"""CPU-only descriptive diagnostics; no model loading or operating-point tuning."""
from pathlib import Path
import numpy as np
from src.evaluation import exact_aupro as exact

FPRS=[.001,.0025,.005,.01,.02,.03,.04,.05,.075,.10,.15,.20,.25,.30]
EPS=1e-10


def percentiles(background,values):
    if not len(background):raise ValueError('Empty background')
    lower=np.searchsorted(background,values,side='left').astype(np.float64)*100/len(background)
    upper=np.searchsorted(background,values,side='right').astype(np.float64)*100/len(background)
    return lower,upper,(lower+upper)/2


def background_quantile(sorted_scores,q):
    if not 0<=q<=1 or not len(sorted_scores):raise ValueError('Invalid quantile')
    index=(len(sorted_scores)-1)*q;lo=int(np.floor(index));hi=int(np.ceil(index))
    return float(sorted_scores[lo])*(hi-index)+float(sorted_scores[hi])*(index-lo) if hi!=lo else float(sorted_scores[lo])


def tail_mask(scores,threshold):
    return scores>threshold


def enrichment(top_count,background_count,global_density):
    density=top_count/background_count if background_count else None
    return dict(density=density,enrichment=density/global_density if density is not None and global_density>0 else None)


def connected(mask):
    import cv2
    return exact.components(mask,cv2,np)


def pair_samples(b0,p1):
    a={r['sample_id']:r for r in b0};b={r['sample_id']:r for r in p1}
    if len(a)!=len(b0) or len(b)!=len(p1) or set(a)!=set(b):raise ValueError('Sample IDs missing or duplicated')
    result=[]
    for sid in sorted(a):
        if Path(a[sid]['image_path']).resolve()!=Path(b[sid]['image_path']).resolve() or int(a[sid]['gt_label'])!=int(b[sid]['gt_label']):raise ValueError('Image/label mapping differs')
        result.append((a[sid],b[sid]))
    return result


def spatial_zones(shape=(1056,4224),boundaries=(1056,2112,3168),width=96):
    h,w=shape;y,x=np.ogrid[:h,:w]
    outer=np.broadcast_to((y<=width)|(h-1-y<=width)|(x<=width)|(w-1-x<=width),shape).copy()
    seam=np.broadcast_to(np.logical_or.reduce([np.abs(x-b)<=width for b in boundaries]),shape)&~outer
    return dict(OUTER_BOUNDARY=outer,INTERNAL_TILE_SEAM=seam,INTERIOR=~(outer|seam))


def region_pro(background,values,fprs=FPRS):
    weights=np.full(len(values),1/len(values),dtype=np.float64)
    curve=exact.pro_at_fprs(background,values,weights,fprs,np)
    areas,total=exact.integrate(background,values,weights,np)
    if abs(total-1)>1e-6 or any(not 0<=v<=1+1e-10 for v in areas.values()):raise ArithmeticError('Invalid shared exact integral')
    if not np.isfinite(curve).all() or np.any(curve < -EPS) or np.any(curve > 1+EPS):raise ArithmeticError('Invalid shared exact PRO curve')
    return curve,areas


def region_statistics(background,values,quantiles=None):
    lower,upper,mid=percentiles(background,values)
    quantiles=quantiles or {q:background_quantile(background,q) for q in (.7,.95)}
    result=dict(raw_mean=float(np.mean(values,dtype=np.float64)),raw_max=float(np.max(values)),raw_p95=float(np.percentile(values,95)),
        percentile_lower_mean=float(lower.mean()),percentile_upper_mean=float(upper.mean()),percentile_mean=float(mid.mean()),
        percentile_median=float(np.median(mid)),percentile_p90=float(np.percentile(mid,90)),percentile_p95=float(np.percentile(mid,95)),
        percentile_max=float(mid.max()),fraction_pixels_above_bg_q95=float((values>quantiles[.95]).mean()),
        fraction_pixels_above_bg_q70=float((values>quantiles[.7]).mean()))
    return result
