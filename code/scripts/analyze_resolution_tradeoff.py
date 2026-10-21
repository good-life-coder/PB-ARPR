"""Phase 08D paired saved-prediction diagnostics; no model construction or inference.

Run manually. Input artifacts are read-only; all analysis outputs stay in Phase 08D.
"""
import csv
import gc
import hashlib
import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluation import exact_aupro as exact
from scripts.analyze_b0_failures import size_group

OUT = ROOT / "experiments/phase08d_resolution_tradeoff"
B0 = ROOT / "experiments/phase07c_baseline256_public"
B1 = ROOT / "experiments/phase08c_baseline384_public"
R1 = ROOT / "experiments/phase07c_r1_aupro_exact"
FPRS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30]
EPS = 1e-10
SAMPLE_BASE = ["sample_id", "image_path", "defect_ratio", "group", "connected_components", "delta_aupro005", "delta_pixel_auroc"]
MODEL_STATS = ["aupro005", "pixel_auroc", "gt_mean", "gt_max", "gt_p95", "gt_background_rank_mean",
               "gt_background_rank_p95", "background_mean", "background_p95", "background_p99", "background_p995",
               "background_p999", "background_max", "global_aupro005_contribution", "global_aupro030_contribution"]
SAMPLE_FIELDS = SAMPLE_BASE + [f"{b}_{key}" for b in ("B0", "B1") for key in MODEL_STATS] + ["delta_global_contribution"]
REGION_STATS = ["rank_mean", "rank_max", "rank_p95", "raw_mean", "raw_max", "raw_p95"]
REGION_FIELDS = ["sample_id", "region_id", "region_size", "region_size_group", "rank_mean_delta"] + [f"{b}_{key}" for b in ("B0", "B1") for key in REGION_STATS]
NORMAL_FIELDS = ["sample_id", "image_path", "normal_rank_shift"] + [f"{b}_{key}" for b in ("B0", "B1") for key in
                 ("official_image_score", "normal_only_rank", "image_percentile_rank", "background_p99", "background_p999", "background_max")]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name, rows, fields):
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def initial_report():
    return dict(status="NOT_RUN", timestamp=datetime.now().astimezone().isoformat(), error_message=None,
                fixed_results_percent={"B0": {"image_auroc":61.57,"pixel_auroc":79.54,"aupro005":6.41,"aupro030":22.33},
                                       "B1": {"image_auroc":56.851851851851855,"pixel_auroc":81.35388138552024,
                                              "aupro005":5.270407401461206,"aupro030":24.860098278293247}},
                hypotheses={}, notes=[
                    "Raw score scales are not comparable across models; primary localization comparisons use within-image background mid-percentile ranks.",
                    "Per-image AU-PRO and the pooled AU-PRO differ: the latter uses the entire public background distribution and equal weights for all regions.",
                    "H7 uses additive per-image contributions to pooled exact AU-PRO, not the mean of per-image AU-PRO.",
                    "Rank changes in normal and anomaly images are coupled in a common ranking; their labels do not identify an independent causal source.",
                    "PRO point values use the same unique-score tie interpolation as exact integration; at vertical jumps the upper/right-continuous value is reported.",
                    "Crossover locations are approximate linear interpolations between the requested diagnostic FPR points, not exact roots.",
                    "Correlation is exploratory, not causal; regions within an image are dependent.",
                    "No operating threshold is selected; no SegF1 or threshold-defined false-positive rate is reported.",
                ], rules={
                    "tiny_pattern":"BROAD_BASED requires >50% of 12 improving and positive median delta; otherwise FEW_SAMPLE_DRIVEN.",
                    "group_D":"OUTLIER_DRIVEN when top two losses supply >=60% of total negative deltas; otherwise >=4/6 worsening is CONSISTENT_GROUP_DROP; else MIXED.",
                    "H7":"SUPPORTED when the two largest negative pooled contributions supply >=50% of total negative contributions; NOT_SUPPORTED when <50% and >45/90 pooled contributions worsen; otherwise INCONCLUSIVE.",
                    "H6":"Strict-PRO penalty plus normal-rank deterioration or reduced positive/background rank separation is PARTIALLY_SUPPORTED. Texture itself is not directly measured, so no causal SUPPORTED label.",
                })


def prepare():
    for name in ("", "figures", "galleries"):
        (OUT/name).mkdir(parents=True, exist_ok=True)


def input_check(np, Image):
    inputs, hashes = {}, {}
    for tag, folder, resolution in (("B0",B0,256),("B1",B1,384)):
        rows = read_csv(folder/"predictions.csv")
        files = list((folder/"raw_anomaly_maps").glob("*.npy"))
        if len(rows)!=114 or len(files)!=114 or len({r["sample_id"] for r in rows})!=114:
            raise ValueError(f"{tag}: expected 114 unique CSV samples and raw maps")
        if len({str(Path(r["image_path"]).resolve()) for r in rows})!=114:
            raise ValueError(f"{tag}: duplicate image paths")
        if [int(r["gt_label"]) for r in rows].count(0)!=24 or [int(r["gt_label"]) for r in rows].count(1)!=90:
            raise ValueError(f"{tag}: expected 24 normal, 90 anomalous")
        used = set()
        for row in rows:
            image_path = Path(row["image_path"]).resolve()
            public = (ROOT/"datasets/MVTec_AD_2/sheet_metal/test_public").resolve()
            if not image_path.is_relative_to(public):
                raise ValueError("Image outside public split")
            with Image.open(image_path) as im:
                row["native_shape"]=(im.height,im.width)
            if row["native_shape"]!=(1056,4224):
                raise ValueError(f"Unexpected native dimensions: {image_path}")
            raw_path = Path(row["raw_map_path"]).resolve()
            if raw_path.parent!=(folder/"raw_anomaly_maps").resolve() or raw_path in used:
                raise ValueError("Duplicate or foreign raw map")
            used.add(raw_path)
            raw=np.load(raw_path,mmap_mode="r",allow_pickle=False)
            if raw.shape!=(resolution,resolution) or raw.dtype!=np.float32 or not np.isfinite(raw).all():
                raise ValueError(f"Invalid raw map: {raw_path}")
            if not math.isclose(float(raw.max()),float(row["official_image_score"]),abs_tol=1e-7,rel_tol=0):
                raise ValueError("Official image score differs from saved continuous-map maximum")
            hashes[str(raw_path)]=digest(raw_path)
            if int(row["gt_label"]):
                mp=Path(row["gt_mask_path"]).resolve()
                if not mp.is_relative_to(public/"ground_truth"):
                    raise ValueError("Mask outside official public ground truth")
                with Image.open(mp) as im:
                    if (im.height,im.width)!=row["native_shape"] or not np.asarray(im).any():
                        raise ValueError("GT shape/foreground invalid")
                hashes[str(mp)]=digest(mp)
        if used!={p.resolve() for p in files}:
            raise ValueError("CSV/raw map directory mismatch")
        hashes[str(folder/"predictions.csv")]=digest(folder/"predictions.csv")
        inputs[tag]={row["sample_id"]:row for row in rows}
    if set(inputs["B0"])!=set(inputs["B1"]):
        raise ValueError("B0/B1 sample IDs differ")
    for sid,row in inputs["B0"].items():
        other=inputs["B1"][sid]
        if int(row["gt_label"])!=int(other["gt_label"]) or Path(row["image_path"]).resolve()!=Path(other["image_path"]).resolve():
            raise ValueError(f"Image/label mismatch: {sid}")
        if int(row["gt_label"]) and Path(row["gt_mask_path"]).resolve()!=Path(other["gt_mask_path"]).resolve():
            raise ValueError(f"Mask mismatch: {sid}")
    return inputs,hashes


def global_cache(tag, rows, np, torch, cv2, Image, report):
    if tag=="B0":
        saved=json.loads((R1/"aupro_report.json").read_text(encoding="utf-8"))
        validation=json.loads((R1/"validation_report.json").read_text(encoding="utf-8"))
        if saved["status"]!="PASS" or validation["status"]!="PASS":
            raise ValueError("B0 exact result and validation must be frozen PASS")
        directory=R1/"temp"
    else:
        outer=json.loads((B1/"evaluation_report.json").read_text(encoding="utf-8"))
        if outer["status"]!="PASS":
            raise ValueError("B1 evaluation must be PASS")
        saved=outer["exact_aupro"]
        directory=B1/"metrics/exact_temp"
    specs=[("background_scores.dat",np.float32,saved["background_pixels"]),
           ("positive_scores.dat",np.float32,saved["positive_pixels"]),
           ("positive_weights.dat",np.float64,saved["positive_pixels"])]
    if not all((directory/name).exists() for name,_,_ in specs):
        directory=OUT/"temp"/tag
        saved={}
        exact.native(list(rows.values()),saved,torch,cv2,np,Image,temp_dir=directory)
    arrays=[]
    for name,dtype,count in specs:
        path=directory/name
        if path.stat().st_size!=np.dtype(dtype).itemsize*count:
            raise ValueError(f"Wrong cache size: {path}")
        arrays.append(np.memmap(path,mode="r",dtype=dtype,shape=(count,)))
    bg,pos,weights=arrays
    for i in range(0,len(bg)-1,1_000_000):
        end=min(i+1_000_000,len(bg)-1)
        if np.any(bg[i:end]>bg[i+1:end+1]) or not np.isfinite(bg[i:end]).all():
            raise ValueError("Invalid sorted background cache")
    values,total=exact.integrate(bg,pos,weights,np)
    if abs(total-1)>=1e-6 or any(abs(values[a]-saved[f"aupro_{a:.2f}"])>1e-7 for a in exact.ALPHAS):
        raise ValueError("Exact cache differs from frozen metrics")
    curve=exact.pro_at_fprs(bg,pos,weights,FPRS,np)
    report.setdefault("global_cache_checks",{})[tag]=dict(directory=str(directory),status="PASS",values=values,
        connected_regions=saved["connected_regions"],background_pixels=len(bg),positive_pixels=len(pos))
    return bg,saved["connected_regions"],values,curve


def midrank(background, values, np):
    return (np.searchsorted(background,values,side="left").astype(np.float64)+
            np.searchsorted(background,values,side="right"))/(2*len(background))


def image_ranks(inputs,np):
    ids=sorted(inputs["B0"])
    result={sid:{} for sid in ids}
    summary={}
    for tag in ("B0","B1"):
        scores=np.array([float(inputs[tag][sid]["official_image_score"]) for sid in ids])
        normal=np.array([int(inputs[tag][sid]["gt_label"])==0 for sid in ids])
        ranked=midrank(np.sort(scores),scores,np)*100
        ranks_normal=midrank(np.sort(scores[normal]),scores,np)*normal.sum()+0.5
        for i,sid in enumerate(ids):
            result[sid][tag+"_official_image_score"]=float(scores[i])
            result[sid][tag+"_image_percentile_rank"]=float(ranked[i])
            if normal[i]:
                result[sid][tag+"_normal_only_rank"]=float(ranks_normal[i])
        n,a=scores[normal],scores[~normal]
        error=float(((n[:,None]>a)+0.5*(n[:,None]==a)).mean())
        summary[tag]=dict(normal_median_rank=float(np.median(ranked[normal])),anomaly_median_rank=float(np.median(ranked[~normal])),
                          normal_mean_rank=float(np.mean(ranked[normal])),anomaly_mean_rank=float(np.mean(ranked[~normal])),
                          pairwise_ranking_error=error,image_auroc_rank_check=1-error)
    normal_up=summary["B1"]["normal_median_rank"]>summary["B0"]["normal_median_rank"]+EPS
    anomaly_down=summary["B1"]["anomaly_median_rank"]<summary["B0"]["anomaly_median_rank"]-EPS
    mechanism="BOTH" if normal_up and anomaly_down else "NORMAL_RANK_INCREASE" if normal_up else "ANOMALY_RANK_DECREASE" if anomaly_down else "INCONCLUSIVE"
    if summary["B1"]["pairwise_ranking_error"]<=summary["B0"]["pairwise_ranking_error"]:
        mechanism="INCONCLUSIVE"
    summary["IMAGE_AUROC_DROP_MECHANISM"]=mechanism
    return result,summary


def analyze_pair(sid,inputs,caches,np,torch,cv2,Image):
    row=inputs["B0"][sid]
    mask=exact.get_mask(row,Image,np)
    labels,sizes=exact.components(mask,cv2,np)
    positive_count=int(mask.sum());k=len(sizes)-1
    sample=dict(sample_id=sid,image_path=row["image_path"],defect_ratio=positive_count/mask.size,
                group=size_group(positive_count/mask.size),connected_components=k)
    regions={i:dict(sample_id=sid,region_id=i,region_size=int(sizes[i])) for i in range(1,k+1)}
    normal=dict(sample_id=sid,image_path=row["image_path"])
    curves={}
    for tag in ("B0","B1"):
        score=exact.get_score(inputs[tag][sid],torch,np)
        bg=score[labels==0]
        bg.sort(kind="quicksort")
        if not k:
            normal.update({tag+"_background_p99":float(np.percentile(bg,99)),tag+"_background_p999":float(np.percentile(bg,99.9)),tag+"_background_max":float(bg[-1])})
            continue
        fg=labels>0;values=score[fg]
        weights=1.0/(k*sizes[labels[fg]].astype(np.float64))
        pro,weight_sum=exact.integrate(bg,values,weights,np)
        if abs(weight_sum-1)>=1e-6:
            raise ValueError("Invalid per-image region weights")
        ranks=midrank(bg,values,np)
        stats=dict(aupro005=pro[0.05],pixel_auroc=float(ranks.mean()),gt_mean=float(values.mean(dtype=np.float64)),
                   gt_max=float(values.max()),gt_p95=float(np.percentile(values,95)),
                   gt_background_rank_mean=float(ranks.mean()),gt_background_rank_p95=float(np.percentile(ranks,95)),
                   background_mean=float(bg.mean(dtype=np.float64)),background_max=float(bg[-1]))
        for suffix,p in (("p95",95),("p99",99),("p995",99.5),("p999",99.9)):
            stats["background_"+suffix]=float(np.percentile(bg,p))
        global_bg,global_k,_,_=caches[tag]
        global_weights=weights*k/global_k
        contribution,_=exact.integrate(global_bg,values,global_weights,np)
        stats["global_aupro005_contribution"]=contribution[0.05]
        stats["global_aupro030_contribution"]=contribution[0.30]
        curves[tag]=exact.pro_at_fprs(global_bg,values,global_weights,FPRS,np)
        sample.update({tag+"_"+key:value for key,value in stats.items()})
        # Use exactly the same connected-component mask for B0 and B1.
        for i in regions:
            v=score[labels==i]
            r=midrank(bg,v,np)
            stats=dict(rank_mean=float(r.mean()),rank_max=float(r.max()),rank_p95=float(np.percentile(r,95)),
                       raw_mean=float(v.mean(dtype=np.float64)),raw_max=float(v.max()),raw_p95=float(np.percentile(v,95)))
            regions[i].update({tag+"_"+key:value for key,value in stats.items()})
    if not k:
        return None,[],normal,curves
    sample.update(delta_aupro005=sample["B1_aupro005"]-sample["B0_aupro005"],
                  delta_pixel_auroc=sample["B1_pixel_auroc"]-sample["B0_pixel_auroc"],
                  delta_global_contribution=sample["B1_global_aupro005_contribution"]-sample["B0_global_aupro005_contribution"])
    for r in regions.values():
        r["rank_mean_delta"]=r["B1_rank_mean"]-r["B0_rank_mean"]
    return sample,list(regions.values()),None,curves


def loss_concentration(values,np):
    losses=np.sort(-np.asarray(values)[np.asarray(values)<-EPS])[::-1]
    return dict(worsened_count=len(losses),total_negative=float(losses.sum()),
                top1_loss_share=float(losses[:1].sum()/losses.sum()) if len(losses) else 0,
                top2_loss_share=float(losses[:2].sum()/losses.sum()) if len(losses) else 0)


def interpret(samples,regions,normals,strict,report,np):
    tiny=[r for r in samples if r["group"]=="A"]
    group_d=[r for r in samples if r["group"]=="D"]
    if len(tiny)!=12 or len(group_d)!=6:
        raise ValueError("Expected 12 tiny and 6 group-D samples")
    delta=np.array([r["delta_aupro005"] for r in tiny])
    improved=int((delta>EPS).sum());worsened=int((delta<-EPS).sum())
    report["tiny"]=dict(n=12,improved_count=improved,worsened_count=worsened,unchanged_count=12-improved-worsened,
                       mean_delta_pp=float(delta.mean()*100),median_delta_pp=float(np.median(delta)*100),
                       pattern="BROAD_BASED" if improved>6 and np.median(delta)>EPS else "FEW_SAMPLE_DRIVEN",samples=tiny)
    d=loss_concentration([r["delta_aupro005"] for r in group_d],np)
    d["GROUP_D_DROP_PATTERN"]="OUTLIER_DRIVEN" if d["top2_loss_share"]>=0.6 else "CONSISTENT_GROUP_DROP" if d["worsened_count"]>=4 else "MIXED"
    d["samples"]=group_d
    d["mean_delta_pp"]=float(np.mean([r["delta_aupro005"] for r in group_d])*100)
    report["group_D"]=d
    pooled=loss_concentration([r["delta_global_contribution"] for r in samples],np)
    pooled["net_delta_pp"]=float(sum(r["delta_global_contribution"] for r in samples)*100)
    pooled["worst_contributors"]=[dict(sample_id=r["sample_id"],delta_pp=r["delta_global_contribution"]*100)
                                    for r in sorted(samples,key=lambda r:r["delta_global_contribution"])[:10]]
    report["pooled_loss_attribution"]=pooled
    crossings=[]
    for a,b in zip(strict,strict[1:]):
        if a["delta_PRO"]*b["delta_PRO"]<0:
            crossings.append(dict(bracket=[a["FPR"],b["FPR"]],approx_fpr=a["FPR"]-a["delta_PRO"]*(b["FPR"]-a["FPR"])/(b["delta_PRO"]-a["delta_PRO"])))
    report["curve_crossovers"]=crossings
    report["curve_equal_grid_points"]=[r["FPR"] for r in strict if abs(r["delta_PRO"])<=EPS]
    strict_drop=[r["FPR"] for r in strict if r["FPR"]<=0.05 and r["delta_PRO"]<0]
    gt_drop=int(sum(r["B1_gt_background_rank_mean"]<r["B0_gt_background_rank_mean"] for r in samples))
    normal_up=int(sum(r["normal_rank_shift"]>EPS for r in normals))
    evidence=dict(strict_fpr_points_with_lower_B1_PRO=strict_drop,normal_images_rank_up=normal_up,
                  anomalies_with_lower_gt_background_percentile=gt_drop,
                  background_rank_definition="local background empirical mid-percentile; each model's background distribution itself has uniform ranks",
                  caveat="These diagnostics can support a strict-background ranking penalty, but cannot establish fine texture as its causal source.")
    h5="SUPPORTED" if delta.mean()>EPS and report["tiny"]["pattern"]=="BROAD_BASED" else "PARTIALLY_SUPPORTED" if delta.mean()>EPS else "NOT_SUPPORTED"
    h6="PARTIALLY_SUPPORTED" if strict_drop and (normal_up>12 or gt_drop>45) else "NOT_SUPPORTED" if not strict_drop else "INCONCLUSIVE"
    h7="SUPPORTED" if pooled["top2_loss_share"]>=0.5 else "NOT_SUPPORTED" if pooled["worsened_count"]>45 else "INCONCLUSIVE"
    report["hypotheses"]={"H5":dict(status=h5,evidence=report["tiny"]),"H6":dict(status=h6,evidence=evidence),"H7":dict(status=h7,evidence=pooled)}
    try:
        from scipy.stats import spearmanr
        def corr(x,y):
            if len(set(x))<2 or len(set(y))<2:
                return dict(status="INCONCLUSIVE",rho=None,p=None)
            rho,p=spearmanr(x,y)
            return dict(status="OK",rho=float(rho),p=float(p))
        report["correlations"]=dict(defect_ratio_vs_delta_aupro005=corr([r["defect_ratio"] for r in samples],[r["delta_aupro005"] for r in samples]),
                                    region_size_vs_response_delta=corr([r["region_size"] for r in regions],[r["rank_mean_delta"] for r in regions]))
    except ImportError:
        report["correlations"]={"status":"NOT_AVAILABLE"}
    quantiles=np.quantile([r["region_size"] for r in regions],[0.2,0.4,0.6,0.8])
    names=["very_tiny","tiny","small","medium","large"]
    for r in regions:
        r["region_size_group"]=names[int(np.searchsorted(quantiles,r["region_size"],side="left"))]
    report["region_size_quantiles"]=quantiles.tolist()
    report["region_group_response"]=[]
    for name in names:
        values=[r["rank_mean_delta"] for r in regions if r["region_size_group"]==name]
        report["region_group_response"].append(dict(group=name,n=len(values),mean_delta=float(np.mean(values)) if values else None,
                                                  median_delta=float(np.median(values)) if values else None))


def plots_and_galleries(samples,regions,normals,strict,inputs,report,np,torch,Image):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for tag in ("B0","B1"):
        ax.plot(FPRS,[r[tag+"_PRO"] for r in strict],marker="o",label=tag)
    ax.axvline(0.05,color="black",linestyle="--",label="FPR=0.05")
    ax.set(xlabel="FPR",ylabel="PRO",title="Exact PRO at fixed FPR points")
    ax.legend();fig.tight_layout();fig.savefig(OUT/"figures/pro_fpr_b0_vs_b1.png",dpi=180);plt.close(fig)
    for data,x,y,name in ((samples,"defect_ratio","delta_aupro005","size_vs_delta_aupro005"),
                          (regions,"region_size","rank_mean_delta","region_size_vs_response_delta")):
        fig,ax=plt.subplots(figsize=(8,5));ax.scatter([r[x] for r in data],[r[y] for r in data],alpha=0.6)
        ax.set(xscale="log",xlabel=x,ylabel=y);ax.axhline(0,color="black",linestyle="--")
        fig.tight_layout();fig.savefig(OUT/f"figures/{name}.png",dpi=180);plt.close(fig)
    albums={"biggest_improvements_10":sorted(samples,key=lambda r:(-r["delta_aupro005"],r["sample_id"]))[:10],
            "biggest_degradations_10":sorted(samples,key=lambda r:(r["delta_aupro005"],r["sample_id"]))[:10],
            "tiny_12":[r for r in samples if r["group"]=="A"],"group_D_6":[r for r in samples if r["group"]=="D"],
            "normal_upward_rank_5":sorted(normals,key=lambda r:(-r["normal_rank_shift"],r["sample_id"]))[:5]}
    report["gallery_selection"]={name:[r["sample_id"] for r in rows] for name,rows in albums.items()}
    report["gallery_count"]=0
    for name,rows in albums.items():
        folder=OUT/"galleries"/name;folder.mkdir(exist_ok=True)
        for r in rows:
            sid=r["sample_id"];base=inputs["B0"][sid];mask=exact.get_mask(base,Image,np)
            with Image.open(base["image_path"]) as image:
                rgb=np.asarray(image.convert("RGB").resize((1200,300),Image.Resampling.BILINEAR))
            gt=np.asarray(Image.fromarray(mask*255).resize((1200,300),Image.Resampling.NEAREST))
            displays=[]
            for tag in ("B0","B1"):
                score=exact.get_score(inputs[tag][sid],torch,np)
                bg=score[mask==0];bg.sort(kind="quicksort")
                display=np.asarray(Image.fromarray(score).resize((1200,300),Image.Resampling.BILINEAR))
                displays.append(midrank(bg,display,np))
            fig,axes=plt.subplots(2,2,figsize=(16,6),constrained_layout=True)
            try:
                for ax,title in zip(axes.flat,("Original","GT","B0 background-percentile heatmap","B1 background-percentile heatmap")):
                    ax.set_title(title);ax.axis("off")
                axes[0,0].imshow(rgb);axes[0,1].imshow(gt,cmap="gray",vmin=0,vmax=255)
                for ax,d in zip(axes[1],displays):ax.imshow(d,cmap="jet",vmin=0,vmax=1)
                caption=(f"delta AU-PRO005={r['delta_aupro005']*100:+.3f} pp" if "delta_aupro005" in r else
                         f"normal percentile-rank shift={r['normal_rank_shift']:+.3f}")
                fig.suptitle(sid+" | "+caption+" | display only")
                fig.savefig(folder/f"{sid}.png",dpi=150)
            finally:plt.close(fig)
            report["gallery_count"]+=1


def summary(r):
    lines=["="*40,"PHASE 08D - RESOLUTION TRADE-OFF ANALYSIS","="*40,"B0: 256","B1: 384","\nTINY SAMPLES"]
    tiny=r.get("tiny",{})
    lines.extend(f"{key}: {tiny.get(key,'N/A')}" for key in ("n","improved_count","worsened_count","unchanged_count","mean_delta_pp","median_delta_pp","pattern"))
    lines.append("\nGROUP D: 0.5%-1%")
    for row in r.get("group_D",{}).get("samples",[]):
        lines.append(f"{row['sample_id']}: B0={row['B0_aupro005']*100:.4f}%, B1={row['B1_aupro005']*100:.4f}%, delta={row['delta_aupro005']*100:+.4f} pp")
    lines.append("Drop pattern: "+r.get("group_D",{}).get("GROUP_D_DROP_PATTERN","NOT_RUN"))
    lines.append("\nPRO-FPR")
    for row in r.get("strict_fpr",[]):
        if row["FPR"] in (0.01,0.05,0.30):lines.append(str(row))
    lines.append("Approximate crossovers: "+str(r.get("curve_crossovers",[])))
    lines.append("IMAGE_AUROC_DROP_MECHANISM: "+r.get("image_ranking",{}).get("IMAGE_AUROC_DROP_MECHANISM","NOT_RUN"))
    lines.extend(f"{key}: {value['status']}" for key,value in r["hypotheses"].items())
    lines.extend(["STATUS:",r["status"],str(r["error_message"] or ""),"="*40])
    return "\n".join(lines)


def save_report(r):
    errors=[]
    for name,text in (("analysis_report.json",json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)),
                      ("analysis_report.txt",summary(r)+"\n\n"+json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False))):
        try:(OUT/name).write_text(text+"\n",encoding="utf-8")
        except OSError as exc:
            traceback.print_exc();errors.append(str(exc))
    if errors:raise OSError("; ".join(errors))


def main():
    report=initial_report();started=time.perf_counter();code=1;stage="INPUT"
    try:
        prepare()
        import numpy as np
        import torch
        import cv2
        from PIL import Image
        inputs,hashes=input_check(np,Image)
        report.update(sample_mapping="PASS",matched_samples=114,input_hashes=hashes,
                      exact_source_sha256=digest(Path(exact.__file__)))
        stage="EXACT_CACHE"
        caches={tag:global_cache(tag,inputs[tag],np,torch,cv2,Image,report) for tag in ("B0","B1")}
        if caches["B0"][1]!=caches["B1"][1]:raise ValueError("Connected-region counts differ")
        ranking,report["image_ranking"]=image_ranks(inputs,np)
        with (OUT/"image_score_ranking.csv").open("w",encoding="utf-8",newline="") as handle:
            fields=["sample_id","gt_label"]+[f"{tag}_{key}" for tag in ("B0","B1") for key in ("official_image_score","image_percentile_rank")]
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
            for sid,row in ranking.items():writer.writerow(dict(sample_id=sid,gt_label=int(inputs["B0"][sid]["gt_label"]),**{k:v for k,v in row.items() if k in fields}))
        samples=[];regions=[];normals=[];summed_curves={tag:np.zeros(len(FPRS)) for tag in caches}
        stage="PAIRED_ANALYSIS"
        for index,sid in enumerate(sorted(inputs["B0"])):
            sample,region_rows,normal,curves=analyze_pair(sid,inputs,caches,np,torch,cv2,Image)
            if sample is not None:samples.append(sample)
            regions.extend(region_rows)
            if normal is not None:
                normal.update(ranking[sid]);normal["normal_rank_shift"]=normal["B1_image_percentile_rank"]-normal["B0_image_percentile_rank"]
                normals.append(normal)
            for tag,curve in curves.items():summed_curves[tag]+=curve
            print(f"Paired sample {index+1}/114",flush=True)
        for tag in caches:
            matches=all(abs(sum(row[tag+"_global_aupro"+suffix+"_contribution"] for row in samples)-caches[tag][2][alpha])<=1e-7
                        for suffix,alpha in (("005",0.05),("030",0.30)))
            if not matches or not np.allclose(summed_curves[tag],caches[tag][3],atol=1e-7,rtol=0):
                raise ValueError("Native cache/provenance mismatch: reconstructed raw-map contributions do not match frozen global result")
        strict=[dict(FPR=f,B0_PRO=float(caches["B0"][3][i]),B1_PRO=float(caches["B1"][3][i]),
                     delta_PRO=float(caches["B1"][3][i]-caches["B0"][3][i])) for i,f in enumerate(FPRS)]
        report["strict_fpr"]=strict;report["connected_regions"]=len(regions)
        interpret(samples,regions,normals,strict,report,np)
        write_csv("paired_sample_delta.csv",samples,SAMPLE_FIELDS)
        write_csv("paired_region_delta.csv",regions,REGION_FIELDS)
        write_csv("normal_score_comparison.csv",normals,NORMAL_FIELDS)
        write_csv("strict_fpr_analysis.csv",strict,["FPR","B0_PRO","B1_PRO","delta_PRO"])
        report["status"]="ANALYSIS_COMPLETE_GALLERIES_PENDING";save_report(report)
        stage="FIGURES"
        plots_and_galleries(samples,regions,normals,strict,inputs,report,np,torch,Image)
        stage="INTEGRITY"
        if any(digest(path)!=value for path,value in hashes.items()):raise ValueError("Input artifact changed")
        report["status"]="PASS";code=0
    except (Exception,KeyboardInterrupt) as exc:
        report.update(status="FAIL",failure_stage=stage,error_message=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["elapsed_seconds"]=time.perf_counter()-started
        try:prepare();save_report(report)
        except Exception as exc:
            traceback.print_exc();report.update(status="FAIL",error_message=str(exc));code=1
        print(summary(report),flush=True)
    return code


if __name__=="__main__":
    sys.exit(main())
