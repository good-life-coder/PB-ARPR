"""Independent result verification and read-only manuscript recommendations."""
import json,time,statistics,math,ast
from datetime import datetime
from pathlib import Path
from run_audit import ROOT,OUT,read,rows,save,csvsave,sha,resources,now
T=time.perf_counter()
def write(name,text):
    with (OUT/name).open('x',encoding='utf8') as f:f.write(text)
def cite(rel,needle=None):
    p=ROOT/rel
    if needle:
        lines=p.read_text(encoding='utf-8-sig').splitlines();n=next(i for i,s in enumerate(lines,1) if needle in s)
        return f'`{p}:{n}`'
    return f'`{p}`'
def verify():
    import numpy as np,torch
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(4)
    result=rows(OUT/'image_auroc_unified.csv');per=rows(OUT/'per_image_scores_unified.csv');ids=rows(OUT/'public_image_identity.csv')
    assert len(result)==10 and len(per)==1140
    maxerror=0.;checks=[]
    for r in result:
        sample=[s for s in per if s['seed']==r['seed'] and s['geometry']==r['geometry']]
        assert len(sample)==114 and [s['sample_id'] for s in sample]==[s['sample_id'] for s in ids]
        assert sum(int(s['gt_label']) for s in sample)==90
        v=roc_auc_score([int(s['gt_label']) for s in sample],[float(s['native_max_score']) for s in sample])
        assert abs(v-float(r['unified_image_auroc']))<1e-12
        if r['mode']=='frozen_inference':
            for s in sample:
                p=OUT/f"model_maps_square512_seed{r['seed']}"/(s['sample_id']+'.npy');assert sha(p)==s['model_map_sha256']
                a=np.load(p,allow_pickle=False);assert a.shape==(512,512) and a.dtype==np.float32 and np.isfinite(a).all()
                with torch.inference_mode():b=torch.nn.functional.interpolate(torch.from_numpy(a)[None,None],size=(1056,4224),mode='bilinear',align_corners=False)
                err=abs(float(b.max())-float(s['native_max_score']));maxerror=max(maxerror,err);assert err==0
        checks.append(dict(seed=r['seed'],geometry=r['geometry'],n=114,sklearn_image_auroc=float(v),status='VERIFIED'))
    for m in read(OUT/'inventory_evidence.json')['models']:
        assert sha(m['checkpoint_path'])==m['checkpoint_sha256']
        assert sha(m['config_path'])==m['config_sha256']
        assert sha(Path(m['config_path']).with_name('training_report.json'))==m['training_report_sha256']
    for r in rows(OUT/'geometry_source_manifest.csv'):assert sha(r['path'])==r['sha256']
    for r in rows(OUT/'historical_training_manifest.csv'):assert sha(r['checkpoint_path'])==r['checkpoint_sha256']
    save('independent_verification.json',dict(status='PASS',created=now(),checks=checks,saved_maps_rechecked=456,max_native_score_error=maxerror,checkpoint_hashes_rechecked=22,source_integrity='All inventoried configs/reports/sources/checkpoints unchanged',seconds=time.perf_counter()-T,resources=resources()))
def reports():
    ev=read(OUT/'evaluation_summary.json');geo=read(OUT/'geometry_recheck.json');inv=read(OUT/'inventory_evidence.json');gate=read(OUT/'inference_gate.json');ver=read(OUT/'independent_verification.json')
    paired=rows(OUT/'paired_seed_deltas.csv');result=rows(OUT/'image_auroc_unified.csv')
    table='| Seed | Square512 (%) | PB-ARPR (%) | PB−Square (pp) |\n|---|---:|---:|---:|\n'+''.join(f"| {r['seed']} | {100*float(r['square512']):.6f} | {100*float(r['pb_arpr']):.6f} | {float(r['delta_pp']):+.6f} |\n" for r in paired)
    sm=ev['summary'];summary=f"Square512 **{sm['square512']['mean']*100:.6f}±{sm['square512']['std']*100:.6f}%**；PB-ARPR **{sm['pb_arpr']['mean']*100:.6f}±{sm['pb_arpr']['std']*100:.6f}%**。同 seed 差值 **{sm['paired_delta_pp']['mean']:.6f}±{sm['paired_delta_pp']['std']:.6f} pp**，正增益 **5/5**。Std 为样本标准差（ddof=1），不是置信区间。"
    oldsq=read(ROOT/'experiments/phase24b_sheetmetal_5seed_geometry/07_square512_5seed_summary.json')['metrics']['image_auroc'];oldpb=read(ROOT/'experiments/phase24b_sheetmetal_5seed_geometry/08_pb_arpr_5seed_summary.json')['metrics']['image_auroc']
    comp=f'''# Image AUROC 统一评分复核

STATUS = VERIFIED

{table}
{summary}

## 定义与完整性

每个模型均使用完整 114 张 sheet_metal test_public 图像（24 正常、90 异常），共 1140 个评分。评分为固定校准后的模型异常图，使用 CPU float32 双线性插值恢复至 H=1056、W=4224，align_corners=False 后的最大值。没有逐图归一化、阈值选择、重拟合分位数、样本删除或 seed 选择。AUROC 使用正负样本两两秩比较（同分计 0.5），并与已安装 sklearn.roc_auc_score 独立核对。

10 个 checkpoint 均通过 SHA256、训练完成 70000 步、最终正常验证校准、输入尺寸、strict state_dict、冻结分位数与 teacher 统计核验。114 张当前公开图像的 SHA256 全部与 Phase13C 历史 frozen_inputs 一致。详见 frozen_model_manifest.csv、inference_gate.json、public_image_identity.csv、independent_verification.json。

## 预测复用与补算

- seed42 Square：复用已存 512×512 模型图，再按统一定义恢复原图并取 max。
- seed42 PB：直接复用已存 1056×4224 native combined 图；其 max 与历史分数完全一致。
- seed43–46 PB：复用 image_scores.json。完整 native 数组已随 TemporaryDirectory 删除，但哈希匹配的历史评价代码明确先 native_map 再 max；114 个身份、标签、原 AUROC 与 checkpoint SHA 均匹配。没有假称这些数组仍可读取，也没有重新推理 PB。
- seed43–46 Square：原存分数是模型网格 max，不能反推 native max，故仅这四个冻结模型补做全部 114 张推理，共 456 张次。eval()、requires_grad_(False)、torch.inference_mode()。全部原始 max 与历史分数差为 **0**。已保存 456 个模型图，独立重算 native max 的差也全部为 **0**。

历史差异来源：{cite('scripts/phase24b_evaluate.py','score=float(raw.max()')}，及 {cite('scripts/phase24b_evaluate.py','arr=rect.native_map(raw)')}。恢复函数见 {cite('src/evaluation/rectangular_formal_evaluation.py','def native_map')}。

## 与旧结果/论文材料差异

Square seed42：61.018519% → 61.064815%，+0.046296 pp；seed46：61.944444% → 61.990741%，+0.046296 pp；seed43–45 AUROC 不变（分数变化不必改变排名）。PB 五 seed 均不变。

Square 五 seed 原均值±Std：{oldsq['mean']*100:.6f}±{oldsq['sample_std']*100:.6f}%；现均值提升 0.018519 pp。PB 原均值±Std：{oldpb['mean']*100:.6f}±{oldpb['sample_std']*100:.6f}%，不变。配对平均增益相应减少 0.018519 pp。

明确找到的论文材料：{cite('PB_ARPR_Paper_Materials/03_Tables_Main/Table_03_Geometry_Ablation.md','| Square512 |')}，该行旧值 61.018519 应更新为 61.064815。五 seed 旧值来源为 Phase24B 的 07/08/09 汇总文件，保留原件。当前项目未定位到完整 SIVP 正文，仅找到论文材料、表格和图稿；正文页码/段落编号 UNVERIFIED，不能假称已逐段核对最终稿。

## 重要预处理限制

VERIFIED：Square 历史 evaluator 调用 pre_processor.transform（antialias=True）；PB 调用 pre_processor.forward/export_transform（antialias=False）。入口在 {cite('scripts/phase24b_evaluate.py','x=(model.pre_processor.transform')}；实际序列化对象和 forward 源码保存在 inference_gate.json。这次保留各自原流程，只统一取分位置。不能宣称跨模型全部预处理完全一致，也不能把这次差值当作排除了 antialias 影响的纯几何因果效应。校准参数均原封不动。

本阶段未重算 Pixel AUROC/AP/AUPRO，未读取测试 mask。既有像素指标数值没有因这次 image pooling 修正而改变，但其因果解释仍受预处理差异约束。
'''
    write('image_auroc_comparison.md',comp)
    hist=f'''# 历史输入尺寸选择协议审计

STATUS = PARTIALLY_VERIFIED

VERIFIED=文件、哈希、实际计算直接支持；PARTIALLY_VERIFIED=事实部分成立但历史过程或因果来源不完整；UNVERIFIED=未找到足够证据；BLOCKED=不得据当前证据宣称或推进的事项。

## 实际正式尺寸与本次 normal-only 复核

| 数据集 | 正常梯度训练图像数 | 本次 median(H/W) | 原始选择器输出 | 历史正式 PB H×W | 结果 |
|---|---:|---:|---|---|---|
| sheet_metal |137|0.25|256×1024|256×1024|VERIFIED，数值相同|
| KSDD2 |1876|2.7729257641921397|864×320|864×320|VERIFIED，数值相同|
| Severstal |4131|0.16|192×1280（AE 不可行）|256×1152|VERIFIED，复用历史可行性库存回退后相同|

本次实际读取上述 6144 张正常训练图像的图像头与 SHA256；KSDD2、Severstal 均与冻结 split_manifest 的图像哈希和尺寸匹配。没有重分割。正常训练清单见 normal_training_dimensions.csv。Severstal 仅重排历史 1707 个候选、246 个可行候选的哈希绑定库存，没有重新做网络 probe。KSDD2/Severstal 的冻结 canonical split SHA 已重算匹配，见 geometry_recheck.json。

Square 正式尺寸：sheet_metal 五 seed 为512×512；KSDD2 五 seed 与 Severstal seed42 的历史 STD 为256×256。后两者的 PB 对照并非严格等像素（276480/294912 vs65536）。本次额外核验这12个历史 checkpoint 的 SHA、配置、loss_history 日志末行以及训练报告；见 historical_training_manifest.csv。没有启动、恢复或读取 Phase17 测试预测。

## A/B/C：历史实际执行与尺寸来源

**sheet_metal：VERIFIED 固定尺寸路径；完整自动选择过程 UNVERIFIED。**

Phase13B seed42 在2026-09-14至09-15训练。{cite('src/evaluation/rectangular_efficientad_calibration.py','TARGET=(256,1024)')} 直接固定 TARGET；{cite('scripts/train_rectangular_efficientad_256x1024.py','input_size_hw=list(cal.TARGET)')} 把它写进正式协议。该路径不先统计 normal-train median、不调用联合目标函数或完整候选选择器。Phase24B seed43–46继续复用冻结尺寸，不能称每个 seed 重新运行自动选择器。存在选定形状的前向/训练可行性检查，但不等同完整候选选择。没有充分证据证明更早人工尺寸决策完全不受已有公开测试观察影响，最早选择动机 **UNVERIFIED**；禁止把未找到证据写成“保证没有测试影响”。

**KSDD2：VERIFIED 执行旧联合选择器，但不是 normal-train-only，也不是后来的完整架构选择器。**

{cite('src/evaluation/ksdd2_external_protocol.py','def select_rect')} 定义 J=ln(HW/262144)^2+ln((H/W)/r)^2，候选为96..2048的32倍数，HW≤300000，tie按像素差、H、W。{cite('src/evaluation/ksdd2_external_protocol.py','split=make_split(rows)')} 对全部3335条官方 train/test、正常/异常图像尺寸取median，历史r=2.773913043478261；未限于1876正常训练样本。目标函数只使用几何，不读取标签值或模型性能，但**使用了测试图像尺寸元数据**，不能称严格训练集限定的选择协议。

Phase15A 已冻结864×320；Phase15C/15D以该尺寸正式训练五 seed，配置入口见 {cite('src/evaluation/ksdd2_70k_protocol.py','def config')}。旧选择器没有 AE 全路径候选筛选。后来的 R1/regression_report 验证该尺寸可行，是后验回归证据，不能改写为原 Phase15A 已执行完整架构门控。

**Severstal：VERIFIED 原选择器失败后，在正式70K前执行了完整架构回退。**

{cite('src/evaluation/severstal_protocol.py','ratios=[r')} 在 split 前使用所有12568个 effective annotated-training 来源的尺寸，随后 {cite('src/evaluation/severstal_protocol.py','geometry_regression();hw=select_rect(aspect)')} 选择192×1280。所有原图均256×1600，所以本次正常训练子集的r同为0.16；人口数值无影响不等于历史 normal-only 实施。

{cite('src/evaluation/severstal_geometry_r1.py','def run')} 保存原候选 AE enconv6 输入6×40、kernel8×8的失败证据；原始192×1280不是“性能失败”。R1采用 {cite('src/common/architecture_constrained_geometry.py','def select')}：raw可行直接保留；不可行才遍历所有正32倍数、HW≤300000的候选，按 {cite('src/common/architecture_constrained_geometry.py','def choose')} 的比例误差优先、再像素偏差等规则选择256×1152。fallback不是继续最小化原联合J，候选域也不同。

geometry_protocol_r1 SHA256=bce452c912826b119c237312997a4f57ea172afe6d1a09b3f2ca04952e8aba6d；inventory SHA256=ddc77445aab39b306d8cf62c97725b05509ea87568191bdce0d295db93e1d85f。execution_binding中的9项代码SHA均匹配。正式训练 {cite('src/evaluation/severstal_70k_protocol.py','def assets')} 验证 geometry SHA，并在配置和训练报告绑定256×1152。这提供正式训练前执行回退的证据链。R1记录 DEV_ANOMALY_FORWARD_COUNT=0、FINAL_BLIND_FORWARD_COUNT=0，selection_used_model_performance=False；支持该次回退性能盲，不能扩大为整个研究历史从未观察任何测试结果。

## D/E/F：应当如何陈述

- **VERIFIED**：用正常训练图像离线复核，三个正式尺寸都可复现（Severstal须回退），不需要用测试指标重新选尺寸。
- **VERIFIED**：存在事后统一整理方法与历史实施不完全相同的情况：sheet_metal固定target，KSDD2全数据几何的旧选择器，Severstal再引入full architecture fallback。三者不能统称“正式训练前均执行了同一 normal-only 完整自动算法”。
- **VERIFIED**：KSDD2原目标函数和Severstal回退排序没有异常标签/测试性能优化项。训练正常样本判定需要既有标签/划分，这是监督信息的角色，不应与利用测试表现调参混同。
- **PARTIALLY_VERIFIED**：训练配置/算法中没有发现按测试AUROC选择最终尺寸；Severstal冻结链较强。sheet_metal最初人工决策动机、所有历史依赖/提交时间完整性仍 UNVERIFIED。不能出具全项目“绝无测试信息影响”的证明。
- **BLOCKED**：将本次离线复核或后来的回归检查表述为历史已运行的自动选择过程。重写历史记录也不能解决真实性问题。

全部当前源文件路径、SHA及训练日志路径在 geometry_source_manifest.csv 与 historical_training_manifest.csv；没有靠文件修改时间单独推断先后。没有修改任何历史文件。
'''
    write('historical_geometry_audit.md',hist)
    corrections=f'''# 论文修订依据（本次未修改论文）

STATUS = PARTIALLY_VERIFIED

## 必须修订的内容

1. **Evaluation protocol / Image-level scoring：VERIFIED。** 明确校准后模型图先以bilinear、align_corners=False恢复原图再取最大值。Square512旧实现不是该定义。区分输入插值与异常图恢复；披露Square antialias=True、PB export antialias=False，删除“仅几何不同且全部预处理一致”的绝对化表述。固定分位数不重拟合。
2. **结果表与正文数值：VERIFIED。** Table_03_Geometry_Ablation 的Square512 seed42 Image AUROC改为61.064815%；五seed表、摘要/正文对应均值改为Square62.583333±1.056935%、PB82.101852±3.290719%、paired gain19.518519±4.227767 pp，5/5为正。表中其余方法/分辨率未在此次统一审计覆盖，不能给整表贴上“所有方法同评分”的标签。若要如此宣称，需另行授权核验其评分；不在本阶段扩展样本/模型。
3. **Method / Dataset reference aspect ratio：VERIFIED。** 将理想化normal-only算法定义与历史实施说明分开。sheet_metal手工冻结；KSDD2全官方数据尺寸统计；Severstal初始全effective来源尺寸、R1使用0.16。新增历史协议表和“retrospective normal-training-only recheck”，不得用离线相同尺寸替换原实施历史。
4. **Algorithm / Feasibility fallback：VERIFIED。** 写清raw-first、原J与fallback排序不同、两种候选域不同、完整gate为eval前向可执行性检查。Severstal192×1280失败→256×1152修复发生在正式70K前；sheet_metal和KSDD2没有同样的前置完整选择过程证据。
5. **Contributions / Abstract / Discussion：PARTIALLY_VERIFIED。** 收紧“普适新resize算子”“选择器本身带来性能提升”“硬件导出的300000上限”“严格无形变”“纯几何因果提升”等表述。实际算子是普通Resize，预算/step是协议选择；联合目标与简单算术在这些比例上得到相同原候选，不能由Square对照证明复杂选择器独立必要。当前成绩支持这些冻结系统的观察性比较。
6. **Dataset statistics / Generalization 表：VERIFIED。** 标注实际H×W及像素数：sheet262144对262144；KSDD2276480对65536；Severstal294912对65536。仅sheet该比较是严格等像素；冻结seed范围与normal-only recheck样本数分别明确。
7. **Reproducibility / Limitations：VERIFIED。** 记录10个SHA、所有114样本、预处理差异、normal quantile固定、seed42旧模型复用。KSDD2几何参考含test尺寸应如实披露；不得声称历史全部步骤训练集限定。

## 可定位的现有材料

- {cite('PB_ARPR_Paper_Materials/03_Tables_Main/Table_03_Geometry_Ablation.md','| Square512 |')}：明确数值更新。
- {cite('PB_ARPR_Paper_Materials/09_Paper_Outline/contributions.md','Geometry allocation')}：统一方法表述须补实施历史与预处理限制。
- {cite('PB_ARPR_Paper_Materials/09_Paper_Outline/results_storyline.md','normal-only training/calibration')}：训练/校准normal-only不能扩写成所有几何统计normal-only。
- {cite('PB_ARPR_Paper_Materials/09_Paper_Outline/discussion_limitations.md','Protocol history')}：保留其Severstal时序说明，增补sheet/KSDD差异。
- Table_04_Three_Dataset_Generalization 的像素/尺寸条件应与上述说明一致；本次未重算其像素指标。

**UNVERIFIED：完整投稿版正文没有在本次项目内发现，无法提供真实页码/段号或确认其中是否已经改正。以上是必须核对的语义位置，不是虚构的已读正文引用。** LaTeX/PDF/表格原件均未改写。

## 仍然成立与不能证明的结论

VERIFIED：统一评分后PB五seed仍均为正；网络和训练权重未变；三个正常训练参考比例可复现原正式尺寸。既有Pixel AUROC/AUPRO数值未被本次image pooling更改，但本次不重新认证其计算。

不能把五seed当作五个独立数据集，也不能把同一114张公开测试的重复使用视为独立外部验证；seed Std不是样本泛化置信区间。不能由相关/系统差值证明AE或Student结构优劣，不能由等像素证明等FLOPs或等延迟。antialias差异使严格“只改变几何”的因果主张未成立。

## 投稿门槛与训练需求

**可以进入有针对性的SIVP英文最终修订；不能据此直接宣告可投稿。** 如果仍保留“全部历史实验前均运行完整normal-only自动选择器”“所有预处理完全相同”等已被证据否定的说法，应当 BLOCKED 投稿，直到纠正。当前没有证据要求为本次数字修正重新训练；训练0小时。新训练也不能把历史流程变成当时未执行的流程。

若坚持严格几何因果主张，先另行预注册/授权同checkpoint的预处理敏感性评价，并明确校准协议；不要用公开测试选择最有利版本。若坚持前瞻性normal-only选择算法验证，需要另行设计独立实验，而不能把本次后验复核改称前瞻实验。本阶段仅提出建议，未执行。
'''
    write('manuscript_correction_list.md',corrections)
    elapsed=(datetime.now().astimezone()-datetime.fromisoformat(read(OUT/'protocol.json')['created'])).total_seconds()
    infer=sum(r['inference_decode_preprocess_model_seconds'] for r in ev['timings'])
    timing=f'''## 时间与资源

启动前估计40–100分钟，GPU训练0小时。正式记录起点 protocol.json 至本报告 {elapsed/60:.2f}分钟；起点前的人工源码定位未单独计时，故该值不是全会话精确耗时。复用了已有图/分数，未缩减114张评价范围，计算部分明显短于预计。

| 阶段 | 实测秒 | 说明 |
|---|---:|---|
| 清单/哈希/图像身份 |{inv['seconds']:.3f}|inventory_evidence.json；早期CPU内存API返回0无效，已修正，原记录保留，不把0当真实峰值|
| checkpoint门控 |{gate['seconds']:.3f}|包含加载、配置、分位数、身份核验|
| 统一评价阶段总计 |{ev['seconds']:.3f}|包含上述门控，二者不可相加|
| 四seed冻结推理 |{infer:.3f}|包括读图、预处理、模型及CPU回传；每段CUDA同步；不是纯GPU核时|
| normal-only几何与历史核验 |{geo['seconds']:.3f}|与评价部分并行，不应直接累加为墙钟时间|
| 独立结果验证 |{ver['seconds']:.3f}|456图重新插值、sklearn复算、22 checkpoint终态哈希|

GPU推理峰值 allocated={ev['gpu_peak_allocated_mib']:.3f} MiB，reserved={ev['gpu_peak_reserved_mib']:.3f} MiB；非整机/其他进程显存。评价进程峰值CPU工作集={ev['resources']['cpu_peak_working_set_mib']:.3f} MiB，峰值pagefile/commit指标={ev['resources']['cpu_peak_pagefile_mib']:.3f} MiB。几何进程峰值工作集={geo['resources']['cpu_peak_working_set_mib']:.3f} MiB；验证进程峰值见 independent_verification.json。不把单进程峰值冒充所有并发进程总峰值，整机CPU内存峰值 UNVERIFIED。

评价进程写完全部结果后，由多余的命令分派块触发KeyError('evaluate')，退出码1；错误发生在UNIFIED_PASS与全部结果落盘之后，无额外推理/结果修改。已只修正新审计脚本的重复分派，并通过独立验证全部输出，不以退出码1的运行本身作为唯一成功依据。未重跑模型。原记录未覆盖。
'''
    final=f'''# PB-ARPR SIVP Submission Audit R1

STATUS = PARTIALLY_VERIFIED

评价统一与正常训练尺寸复核：VERIFIED。历史实施差异：已识别；全部早期决策动机、完整投稿正文：UNVERIFIED。不能无条件放行投稿。

## 六项回答

1. **五seed Image AUROC是否统一：是。** 两种几何均按native map max；每模型114张，全部保留。PB43–46复用有代码哈希绑定的native-max分数；Square43–46补推理456张次。
2. **PB是否仍正增益：是，5/5。** {summary}
3. **哪些历史实验未执行完整自动选择器：** sheet_metal Phase13B固定256×1024，Phase24B复用；KSDD2 Phase15A运行全数据尺寸的旧选择器，Phase15C/D复用864×320，不能称已前置执行后来的完整架构选择器。Severstal R1确实先执行完整回退再绑定Phase16C正式训练，但其历史比例并非normal-train-only统计。
4. **必须修改哪些段落：** 评价评分定义、Square512相关数字/五seed表、参考比例人口与历史实施、raw-first与fallback算法、预处理antialias差异、贡献与因果性限制。具体证据和建议在 manuscript_correction_list.md；未发现完整正文，页码/段号UNVERIFIED。
5. **是否需要重新训练：** 本次评价修正和如实描述历史不需要；未训练。若要增加新因果/前瞻性主张，应另行授权独立设计，不能改历史。
6. **是否可以进入SIVP英文最终修订：** 可以按修订清单进入；正式投稿应等待上述不实/过强表述纠正，并核对实际全文，当前不作投稿就绪保证。

{table}
## 真实性与边界

统一评分后观察到的优势成立，但预处理仍存在历史antialias差异，不能称严格纯几何因果对照。数值一致不等于历史流程一致；KSDD2原选择器读取了测试尺寸元数据，Severstal初始使用全部effective来源，sheet_metal最早人工决策原因未知。不得把本次离线normal-only复核写成历史自动选择。若论文保留相反表述，投稿应 BLOCKED。

只对sheet_metal公开测试读取图像标签；没有读取测试mask、私有/盲测图像或预测。KSDD2/Severstal只读取冻结划分元数据及正常训练图像。没有下载、安装、训练、反向传播、修改网络/归一化/数据划分或checkpoint，没有执行/恢复Phase17，没有修改论文。

{timing}

所有输出位于 `{OUT}`。完整文件绝对路径、大小及SHA见 artifact_manifest.csv；456个补算模型图独立存放，未覆盖旧文件。磁盘最终大小见 resource_accounting.json。

本阶段结束，未自动启动后续实验或论文修订。
'''
    write('final_report.md',final);write('final_console_report.txt',final)
    # Bind current implementation, authoritative old results, and all retained score sources.
    source=[]
    paths=[ROOT/'scripts/phase24b_evaluate.py',ROOT/'src/evaluation/rectangular_formal_evaluation.py',ROOT/'experiments/phase13c_rectangular_256x1024_public/precheck_report.json',ROOT/'experiments/phase13c_rectangular_256x1024_public/predictions.csv',ROOT/'experiments/phase18b_sheet_metal_square512_70k_formal/evaluation/predictions.csv']
    for seed in range(43,47):
        for g in ('square512','pb_arpr'):
            for n in ('image_scores.json','evaluation_result.json'):paths.append(ROOT/f'experiments/phase24b_sheetmetal_5seed_geometry/seed{seed}/{g}/{n}')
    for p in paths:source.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    csvsave('evaluation_source_manifest.csv',source)
    save('resource_accounting.json',dict(created=now(),report_generation_seconds=time.perf_counter()-T-ver['seconds'],elapsed_seconds_since_protocol=elapsed,training_seconds=0,inference_decode_preprocess_model_seconds=infer,evaluation_resources=ev['resources'],geometry_resources=geo['resources'],verification_resources=ver['resources'],report_process_resources=resources(),disk_note='Output total at measurement; artifact_manifest.csv itself adds a small amount afterwards; full raw model maps retained, no native-map duplication',initial_inventory_cpu_peak='UNVERIFIED: original zero from API signature error; corrected subsequent stages',whole_session_cpu_peak='UNVERIFIED; per-process peaks recorded'))
    files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file()]
    csvsave('artifact_manifest.csv',files)
    print('STATUS = PARTIALLY_VERIFIED\n'+table+'\n'+summary+'\nAll required reports: '+str(OUT),flush=True)
if __name__=='__main__':verify();reports()
