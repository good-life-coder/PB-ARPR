"""Exact no-overlap tile placement with continuous bilinear upsampling."""
import numpy as np
import torch


def stitch_tiles(tiles):
    if len(tiles)!=4 or {int(i) for i,_ in tiles}!={0,1,2,3}:raise ValueError("Exactly four unique tile IDs required")
    enlarged={}
    for tile_id,raw in tiles:
        raw=np.asarray(raw,dtype=np.float32)
        if raw.shape!=(256,256) or not np.isfinite(raw).all():raise ValueError("Invalid 256 continuous tile map")
        enlarged[int(tile_id)]=torch.nn.functional.interpolate(torch.from_numpy(raw)[None,None],size=(1056,1056),
                                                             mode="bilinear",align_corners=False)[0,0].numpy()
    stitched=np.concatenate([enlarged[i] for i in range(4)],axis=1)
    if stitched.shape!=(1056,4224):raise ValueError("Invalid stitched dimensions")
    for i in range(4):
        if not np.array_equal(stitched[:,i*1056:(i+1)*1056],enlarged[i]):raise ValueError("Tile placement failed")
    return stitched


def seam_diagnostics(stitched):
    rng=np.random.default_rng(42)
    boundaries=[1056,2112,3168]
    columns=rng.choice(np.setdiff1d(np.arange(1,4224),boundaries),size=128,replace=False)
    changes=np.array([np.abs(stitched[:,x]-stitched[:,x-1]).mean() for x in columns])
    seams=[]
    zero_columns=np.all(stitched==0,axis=0)
    for x in boundaries:
        left,right=stitched[:,x-1],stitched[:,x]
        jump=float(np.abs(right-left).mean())
        start,end=x,x
        while start>0 and zero_columns[start-1]:start-=1
        while end<4224 and zero_columns[end]:end+=1
        seams.append(dict(boundary=x,left_column_mean=float(left.mean()),right_column_mean=float(right.mean()),
                          contiguous_zero_band_width=end-start,zero_band_x0=start,zero_band_x1=end,
                          absolute_mean_jump=abs(float(left.mean()-right.mean())),mean_absolute_pixel_jump=jump,
                          nonboundary_percentile=float((changes<=jump).mean()*100),above_nonboundary_p99=jump>float(np.percentile(changes,99))))
    return dict(seams=seams,random_nonboundary_columns=columns.tolist(),nonboundary_jump_mean=float(changes.mean()),
                nonboundary_jump_median=float(np.median(changes)),nonboundary_jump_p99=float(np.percentile(changes,99)),
                note="Diagnostic only, 128 seeded nonboundary column pairs; no blend, smoothing, or overlap. Official EfficientAD defaults to pad_maps=True and pads map borders with zeros. Zero adjacent-column jump does not rule out a low-response seam band.")
