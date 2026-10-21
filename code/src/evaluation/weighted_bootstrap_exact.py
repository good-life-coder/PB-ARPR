"""Exact image-multiplicity bootstrap using cached foreground/background ranks.

Independent of and does not alter exact_aupro.py. No bins, thresholds, score
quantization, pixel/region selection, or model dependencies. All original
foreground pixels are retained; all original background pixels enter ranks.
"""
import json
from pathlib import Path
import time
import numpy as np
from . import exact_aupro as reference


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def bootstrap_indices(labels, count=20, seed=20260924):
    labels=np.asarray(labels)
    normal=np.flatnonzero(labels==0); anomaly=np.flatnonzero(labels==1)
    rng=np.random.default_rng(seed)
    return np.asarray([np.r_[rng.choice(normal,len(normal),replace=True),
                            rng.choice(anomaly,len(anomaly),replace=True)] for _ in range(count)])


def build_gt_cache(rows, directory):
    import cv2
    from PIL import Image
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/'manifest.json'
    if manifest.exists():
        cached=json.loads(manifest.read_text())
        assert cached['sample_ids']==[r['sample_id'] for r in rows]
        return cached
    entries=[]; total_regions=0; cc_calls=0
    for i,r in enumerate(rows):
        mask=reference.get_mask(r,Image,np)
        if r['gt_label']:
            labels,sizes=reference.components(mask,cv2,np)
            cc_calls+=1
            foreground=np.flatnonzero(labels.ravel()).astype(np.uint32)
            ids=labels.ravel()[foreground].astype(np.uint32)
        else:
            foreground=np.empty(0,np.uint32); ids=np.empty(0,np.uint32)
            sizes=np.asarray([mask.size],dtype=np.int64)
        assert int(sizes.sum())==mask.size
        path=directory/f'{i:03d}.npz'
        np.savez(path,foreground=foreground,region_ids=ids,region_sizes=sizes[1:].astype(np.int64))
        k=len(sizes)-1; total_regions+=k
        entries.append(dict(path=str(path.resolve()),sample_id=r['sample_id'],
            foreground_count=len(foreground),background_count=int(sizes[0]),region_count=k,shape=list(mask.shape)))
    result=dict(entries=entries,sample_ids=[r['sample_id'] for r in rows],total_regions=total_regions,
        connected_component_calls=cc_calls,connectivity=8,gt_regions_modified=False)
    dump(manifest,result)
    return result


def build_rank_cache(rows, gt, directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/'manifest.json'
    if manifest.exists():
        cached=json.loads(manifest.read_text())
        assert cached['sample_ids']==[r['sample_id'] for r in rows]
        return cached
    start=time.perf_counter()
    n=len(rows)
    fgcounts=np.asarray([e['foreground_count'] for e in gt['entries']],np.int64)
    bgcounts=np.asarray([e['background_count'] for e in gt['entries']],np.int64)
    p=int(fgcounts.sum())
    if p==0 or bgcounts.sum()==0:
        raise ValueError('Both foreground/background required')
    if np.max(bgcounts)>np.iinfo(np.uint32).max:
        raise ValueError('Per-image rank exceeds uint32 cache representation')
    scores=np.empty(p,np.float32); owners=np.empty(p,np.int32); invsizes=np.empty(p,np.float64)
    offset=0
    for i,(r,e) in enumerate(zip(rows,gt['entries'])):
        a=np.load(r['raw_map_path'],mmap_mode='r',allow_pickle=False)
        assert list(a.shape)==e['shape'] and a.dtype==np.float32 and np.isfinite(a).all()
        with np.load(e['path']) as g:
            idx=g['foreground']; ids=g['region_ids']; sizes=g['region_sizes']
            count=len(idx)
            scores[offset:offset+count]=a.ravel()[idx]
            owners[offset:offset+count]=i
            invsizes[offset:offset+count]=1/sizes[ids-1].astype(np.float64)
            offset+=count
    order=np.argsort(scores,kind='stable')
    scores=scores[order];owners=owners[order];invsizes=invsizes[order]
    del order
    for key,value in [('positive_scores',scores),('positive_owner',owners),('inverse_region_size',invsizes)]:
        np.save(directory/f'{key}.npy',value,allow_pickle=False)
    left=np.lib.format.open_memmap(directory/'left.npy',mode='w+',dtype=np.uint32,shape=(n,p))
    right=np.lib.format.open_memmap(directory/'right.npy',mode='w+',dtype=np.uint32,shape=(n,p))
    pair=np.zeros((n,n),np.int64)
    for j,(r,e) in enumerate(zip(rows,gt['entries'])):
        a=np.load(r['raw_map_path'],mmap_mode='r',allow_pickle=False)
        valid=np.ones(a.size,dtype=bool)
        with np.load(e['path']) as g:
            valid[g['foreground']]=False
        bg=np.asarray(a.ravel()[valid],dtype=np.float32)
        del valid,a
        assert len(bg)==int(bgcounts[j])
        bg.sort(kind='quicksort')
        l=np.searchsorted(bg,scores,side='left')
        rnk=np.searchsorted(bg,scores,side='right')
        left[j]=l;right[j]=rnk
        # Every pair count here is an integer <=2**53, so bincount(float64)
        # is exact; the saved matrix and all multiplicity arithmetic are int64.
        assert 2*int(fgcounts.max())*len(bg)<2**53
        pair[:,j]=np.bincount(owners,weights=l+rnk,minlength=n).astype(np.int64)
        if (j+1)%10==0:
            print(f'Exact rank cache {j+1}/{n}',flush=True)
    left.flush();right.flush()
    del left,right,bg,l,rnk
    np.save(directory/'pair_twice_wins.npy',pair,allow_pickle=False)
    result=dict(directory=str(directory.resolve()),sample_ids=[r['sample_id'] for r in rows],
        positive_count=p,background_counts=bgcounts.tolist(),foreground_counts=fgcounts.tolist(),
        region_counts=[e['region_count'] for e in gt['entries']],rank_dtype='uint32',
        exact_aupro=True,pixel_subsampling_used=False,threshold_approximation_used=False,
        physical_duplicate_materialization=False,preparation_sec=time.perf_counter()-start)
    dump(manifest,result)
    return result


def evaluate(cache,multiplicity):
    m=np.asarray(multiplicity)
    if m.dtype.kind not in 'iu' or m.ndim!=1 or len(m)!=len(cache['sample_ids']) or np.any(m<0):
        raise ValueError('Invalid image multiplicity')
    m=m.astype(np.int64,copy=False)
    nb=int(m@np.asarray(cache['background_counts'],np.int64))
    nf=int(m@np.asarray(cache['foreground_counts'],np.int64))
    k=int(m@np.asarray(cache['region_counts'],np.int64))
    if min(nb,nf,k)<=0:
        raise ValueError('Multiplicity must retain foreground/background/regions')
    directory=Path(cache['directory'])
    begin=time.perf_counter()
    pair=np.load(directory/'pair_twice_wins.npy',mmap_mode='r')
    numerator=int(m@pair@m)
    pixel=numerator/(2*nb*nf)
    pixel_time=time.perf_counter()-begin
    begin=time.perf_counter()
    p=cache['positive_count']
    left=np.zeros(p,np.int64);right=np.zeros(p,np.int64);scratch=np.empty(p,np.int64)
    lr=np.load(directory/'left.npy',mmap_mode='r')
    rr=np.load(directory/'right.npy',mmap_mode='r')
    for j,weight in enumerate(m):
        if weight:
            np.multiply(lr[j],weight,out=scratch,dtype=np.int64);left+=scratch
            np.multiply(rr[j],weight,out=scratch,dtype=np.int64);right+=scratch
    del scratch,lr,rr
    owner=np.load(directory/'positive_owner.npy',mmap_mode='r')
    inv=np.load(directory/'inverse_region_size.npy',mmap_mode='r')
    totals={a:0.0 for a in reference.ALPHAS};weight_sum=0.0
    for start in range(0,p,250_000):
        stop=min(start+250_000,p)
        l=left[start:stop];r=right[start:stop]
        q_gt=(nb-r).astype(np.float64)/nb
        q_ge=(nb-l).astype(np.float64)/nb
        q_eq=(r-l).astype(np.float64)/nb
        w=m[owner[start:stop]].astype(np.float64)*inv[start:stop]/k
        weight_sum+=float(w.sum(dtype=np.float64))
        for alpha in reference.ALPHAS:
            # Identical tie-aware piecewise integration to reference.integrate.
            area=np.zeros(stop-start,np.float64)
            active=alpha>q_gt
            no_tie=active & (r==l)
            full_tie=active & (r>l) & (alpha>=q_ge)
            partial_tie=active & (r>l) & (alpha<q_ge)
            area[no_tie]=alpha-q_gt[no_tie]
            area[full_tie]=q_eq[full_tie]/2+alpha-q_ge[full_tie]
            area[partial_tie]=(alpha-q_gt[partial_tie])**2/(2*q_eq[partial_tie])
            totals[alpha]+=float(np.dot(w,area))
    if abs(weight_sum-1)>1e-10:
        raise ArithmeticError(f'PRO weights sum to {weight_sum}')
    result=dict(pixel_auroc=pixel,aupro005=totals[.05]/.05,aupro030=totals[.30]/.30,
        pixel_sec=pixel_time,aupro_joint_sec=time.perf_counter()-begin,
        background_pixels=nb,positive_pixels=nf,connected_regions=k,weight_sum=weight_sum)
    # Do not clip/change the result: a mathematically perfect area can be
    # 1+one ULP after float64 accumulation, also in reference.integrate.
    assert all(np.isfinite(result[x]) and -1e-12<=result[x]<=1+1e-12
               for x in ['pixel_auroc','aupro005','aupro030'])
    return result
