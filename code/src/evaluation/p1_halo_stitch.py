"""Map full context first, then retain exactly the central core."""
import numpy as np
import torch
from src.evaluation.tile_stitch import seam_diagnostics


def core_map(raw):
    raw=torch.as_tensor(np.asarray(raw,dtype=np.float32))
    assert raw.shape==(256,256) and torch.isfinite(raw).all()
    context=torch.nn.functional.interpolate(raw[None,None],size=(1248,1248),mode='bilinear',align_corners=False)[0,0]
    return context[96:1152,96:1152].numpy().copy()


def stitch(tiles):
    assert len(tiles)==4 and {i for i,_ in tiles}=={0,1,2,3}
    cores={i:core_map(raw) for i,raw in tiles}
    result=np.concatenate([cores[i] for i in range(4)],axis=1)
    assert result.shape==(1056,4224)
    for i in range(4):assert np.array_equal(result[:,i*1056:(i+1)*1056],cores[i])
    return result


def seams(array):
    report=seam_diagnostics(array)
    for threshold,name in ((0,'exact_zero_run_width'),(1e-12,'near_zero_run_width')):
        zero=np.all(np.abs(array)==0 if threshold==0 else np.abs(array)<threshold,axis=0)
        for entry in report['seams']:
            x=entry['boundary'];left=right=x
            while left>0 and zero[left-1]:left-=1
            while right<4224 and zero[right]:right+=1
            entry[name]=right-left
    report['note']='Diagnostic only; context maps bilinear restored to1248 then cropped to1056 cores; no prediction blending.'
    return report
