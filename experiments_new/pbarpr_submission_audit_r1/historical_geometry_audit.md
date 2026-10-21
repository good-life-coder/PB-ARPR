# 历史输入尺寸选择协议审计

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

Phase13B seed42 在2026-09-14至09-15训练。`.\src\evaluation\rectangular_efficientad_calibration.py:17` 直接固定 TARGET；`.\scripts\train_rectangular_efficientad_256x1024.py:62` 把它写进正式协议。该路径不先统计 normal-train median、不调用联合目标函数或完整候选选择器。Phase24B seed43–46继续复用冻结尺寸，不能称每个 seed 重新运行自动选择器。存在选定形状的前向/训练可行性检查，但不等同完整候选选择。没有充分证据证明更早人工尺寸决策完全不受已有公开测试观察影响，最早选择动机 **UNVERIFIED**；禁止把未找到证据写成“保证没有测试影响”。

**KSDD2：VERIFIED 执行旧联合选择器，但不是 normal-train-only，也不是后来的完整架构选择器。**

`.\src\evaluation\ksdd2_external_protocol.py:43` 定义 J=ln(HW/262144)^2+ln((H/W)/r)^2，候选为96..2048的32倍数，HW≤300000，tie按像素差、H、W。`.\src\evaluation\ksdd2_external_protocol.py:99` 对全部3335条官方 train/test、正常/异常图像尺寸取median，历史r=2.773913043478261；未限于1876正常训练样本。目标函数只使用几何，不读取标签值或模型性能，但**使用了测试图像尺寸元数据**，不能称严格训练集限定的选择协议。

Phase15A 已冻结864×320；Phase15C/15D以该尺寸正式训练五 seed，配置入口见 `.\src\evaluation\ksdd2_70k_protocol.py:15`。旧选择器没有 AE 全路径候选筛选。后来的 R1/regression_report 验证该尺寸可行，是后验回归证据，不能改写为原 Phase15A 已执行完整架构门控。

**Severstal：VERIFIED 原选择器失败后，在正式70K前执行了完整架构回退。**

`.\src\evaluation\severstal_protocol.py:159` 在 split 前使用所有12568个 effective annotated-training 来源的尺寸，随后 `.\src\evaluation\severstal_protocol.py:162` 选择192×1280。所有原图均256×1600，所以本次正常训练子集的r同为0.16；人口数值无影响不等于历史 normal-only 实施。

`.\src\evaluation\severstal_geometry_r1.py:11` 保存原候选 AE enconv6 输入6×40、kernel8×8的失败证据；原始192×1280不是“性能失败”。R1采用 `.\src\common\architecture_constrained_geometry.py:31`：raw可行直接保留；不可行才遍历所有正32倍数、HW≤300000的候选，按 `.\src\common\architecture_constrained_geometry.py:25` 的比例误差优先、再像素偏差等规则选择256×1152。fallback不是继续最小化原联合J，候选域也不同。

geometry_protocol_r1 SHA256=bce452c912826b119c237312997a4f57ea172afe6d1a09b3f2ca04952e8aba6d；inventory SHA256=ddc77445aab39b306d8cf62c97725b05509ea87568191bdce0d295db93e1d85f。execution_binding中的9项代码SHA均匹配。正式训练 `.\src\evaluation\severstal_70k_protocol.py:28` 验证 geometry SHA，并在配置和训练报告绑定256×1152。这提供正式训练前执行回退的证据链。R1记录 DEV_ANOMALY_FORWARD_COUNT=0、FINAL_BLIND_FORWARD_COUNT=0，selection_used_model_performance=False；支持该次回退性能盲，不能扩大为整个研究历史从未观察任何测试结果。

## D/E/F：应当如何陈述

- **VERIFIED**：用正常训练图像离线复核，三个正式尺寸都可复现（Severstal须回退），不需要用测试指标重新选尺寸。
- **VERIFIED**：存在事后统一整理方法与历史实施不完全相同的情况：sheet_metal固定target，KSDD2全数据几何的旧选择器，Severstal再引入full architecture fallback。三者不能统称“正式训练前均执行了同一 normal-only 完整自动算法”。
- **VERIFIED**：KSDD2原目标函数和Severstal回退排序没有异常标签/测试性能优化项。训练正常样本判定需要既有标签/划分，这是监督信息的角色，不应与利用测试表现调参混同。
- **PARTIALLY_VERIFIED**：训练配置/算法中没有发现按测试AUROC选择最终尺寸；Severstal冻结链较强。sheet_metal最初人工决策动机、所有历史依赖/提交时间完整性仍 UNVERIFIED。不能出具全项目“绝无测试信息影响”的证明。
- **BLOCKED**：将本次离线复核或后来的回归检查表述为历史已运行的自动选择过程。重写历史记录也不能解决真实性问题。

全部当前源文件路径、SHA及训练日志路径在 geometry_source_manifest.csv 与 historical_training_manifest.csv；没有靠文件修改时间单独推断先后。没有修改任何历史文件。
