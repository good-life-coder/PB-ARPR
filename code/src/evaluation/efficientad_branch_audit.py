"""Frozen EfficientAD branch extraction; all training and fitting APIs excluded."""
import csv
import hashlib
from pathlib import Path
import numpy as np
import torch
from src.evaluation.strict_fpr_diagnostics import percentiles,region_pro
from src.evaluation.p1_halo_stitch import stitch as native_map

BRANCHES=('ST_RAW','AE_RAW','ST_NORMALIZED','AE_NORMALIZED','COMBINED')
DIRECTORIES={name:name.lower() for name in BRANCHES}
EXPECTED_SHA256='52b75511be348e8361d5177bb0906c3b76ab60cf7fd990b0279fc5dfbd3980b9'


def normalize(raw,qa,qb):
    if not torch.isfinite(qa).all() or not torch.isfinite(qb).all() or not bool((qb>qa).all()):
        raise ValueError('FAIL_QUANTILE_STATE: finite strictly increasing quantiles required; no epsilon or refitting')
    return 0.1*(raw-qa)/(qb-qa)


def branch_maps(st,ae,q):
    if st.shape!=ae.shape or not torch.isfinite(st).all() or not torch.isfinite(ae).all():
        raise ValueError('ST/AE shape or finite-value mismatch')
    sn=normalize(st,q['qa_st'],q['qb_st']);an=normalize(ae,q['qa_ae'],q['qb_ae'])
    return dict(ST_RAW=st.clone(),AE_RAW=ae.clone(),ST_NORMALIZED=sn,AE_NORMALIZED=an,COMBINED=0.5*sn+0.5*an)


def parameter_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        a=p.detach().cpu().contiguous().numpy()
        h.update(name.encode('utf-8'));h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def state_hash(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        a=tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode('utf-8'));h.update(a.tobytes())
    return h.hexdigest()


def extract_context(model,path):
    from anomalib.data.utils import read_image
    if model.training or model.model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Frozen eval model required')
    with torch.no_grad():
        x=read_image(path,as_tensor=True)[None].to(next(model.parameters()).device)
        if tuple(x.shape[-2:])!=(1248,1248):raise ValueError('Context geometry differs')
        x=model.pre_processor(x)
        if tuple(x.shape[-2:])!=(256,256):raise ValueError('Input geometry differs')
        st,ae=model.model.get_maps(x,normalize=False)
        maps=branch_maps(st,ae,model.model.quantiles)
        official=model.model(x).anomaly_map
        error=float((maps['COMBINED']-official).abs().max().cpu())
        if error>1e-5:raise ValueError(f'FAIL_RECONSTRUCTION_MISMATCH: {error}')
        result={k:v[0,0].detach().float().cpu().numpy().copy() for k,v in maps.items()}
    return result,error


def frozen_context_control(root):
    """One real GPU control against official forward AND saved Phase10B context."""
    from scripts.evaluate_p1_halo96_public import CHECKPOINT,sha,load_model
    from src.data.p1_halo_builder import manifest_hash,read_rows
    if sha(CHECKPOINT)!=EXPECTED_SHA256:raise ValueError('FAIL_CHECKPOINT_HASH')
    if not torch.cuda.is_available():raise RuntimeError('GPU0 required for frozen control')
    model=load_model({},manifest_hash(),torch).to('cuda:0')
    before=parameter_hash(model)
    with (root/'experiments/phase10b_p1_halo96_public/context_predictions.csv').open(encoding='utf-8',newline='') as f:
        row=next(csv.DictReader(f))
    with (root/'experiments/phase10b_p1_halo96_public/predictions.csv').open(encoding='utf-8',newline='') as f:
        source=next(a for a in csv.DictReader(f) if a['sample_id']==row['sample_id'])
    context=next(a for a in read_rows() if a['split']=='test_public' and Path(a['source_image_path']).resolve()==Path(source['image_path']).resolve() and int(a['tile_id'])==int(row['tile_id']))
    maps,error=extract_context(model,context['context_path'])
    saved=np.load(row['raw_map_path'],allow_pickle=False)
    result=dict(official_max_abs_error=error,frozen_max_abs_error=float(np.max(np.abs(saved-maps['COMBINED']))),parameters_unchanged=before==parameter_hash(model))
    del model;torch.cuda.empty_cache()
    return result
