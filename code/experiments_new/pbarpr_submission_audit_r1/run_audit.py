"""Submission audit: immutable inputs, exclusive new outputs, no training entrypoints."""
import os, sys, json, csv, hashlib, time, math, statistics, zipfile, pickletools
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT)); sys.dont_write_bytecode=True
START=time.perf_counter()
def now(): return datetime.now().astimezone().isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def save(name,obj):
    with (OUT/name).open('x',encoding='utf8') as f: json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
def csvsave(name,data):
    with (OUT/name).open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
def folder(seed,geo):
    if seed==42: return ROOT/('experiments/phase18b_sheet_metal_square512_70k_formal' if geo=='square512' else 'experiments/phase13b_rectangular_256x1024_seed42')
    return ROOT/f'experiments/phase24b_sheetmetal_5seed_geometry/seed{seed}/{geo}'
def auc(labels,scores):
    pos=[s for y,s in zip(labels,scores) if y];neg=[s for y,s in zip(labels,scores) if not y]
    return sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
def resources():
    import ctypes
    from ctypes import wintypes
    class PMC(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    p=PMC();p.cb=ctypes.sizeof(p)
    ctypes.windll.kernel32.GetCurrentProcess.restype=wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
    assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(p),p.cb)
    return dict(cpu_peak_working_set_mib=p.PeakWorkingSetSize/2**20,cpu_peak_pagefile_mib=p.PeakPagefileUsage/2**20,output_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()))
def inventory():
    save('protocol.json',dict(task='PB-ARPR SIVP Submission Audit R1',created=now(),root=str(ROOT),python=sys.executable,estimate_minutes=[40,100],training_hours=0,seeds=list(range(42,47)),image_score='max(F.interpolate(frozen calibrated model map, native H/W, bilinear, align_corners=False))',normalization='Frozen checkpoint quantiles only; no per-image normalization; no fitting',preprocessing='Preserve historical evaluation entrypoints: square transform, rectangular PreProcessor.forward; audit their actual antialias semantics',population='All 114 sheet_metal public images, 24 normal/90 anomalous',std='sample standard deviation, ddof=1',missing_maps='Seeds43-46 full maps deleted with TemporaryDirectory; square raw maxima require frozen re-inference; PB saved native maxima may be reused only with source/hash/identity checks',geometry_recheck='Median H/W of actual normal training image headers; execute existing raw selector by AST extraction; fallback reuse frozen feasibility inventory',writes='Only this independent directory; exclusive creation',prohibited=['training','backprop','calibration','Phase17 execution','manuscript edits','existing artifact writes','private/blind data access','network/downloads']))
    freeze=read(ROOT/'experiments/phase24b_sheetmetal_5seed_geometry/01_protocol_freeze.json')
    codecheck={n:dict(expected=h,actual=sha(ROOT/'scripts'/n)) for n,h in freeze['scripts_sha256'].items()}
    assert all(v['expected']==v['actual'] for v in codecheck.values())
    refs=rows(folder(42,'square512')/'evaluation/predictions.csv')
    assert len(refs)==114 and len({r['sample_id'] for r in refs})==114 and sum(int(r['gt_label']) for r in refs)==90
    from PIL import Image
    image_inventory=[]
    public=ROOT/'datasets/MVTec_AD_2/sheet_metal/test_public'
    actual={str(p.resolve()) for d in ('good','bad') for p in (public/d).glob('*.png')}
    assert actual=={str(Path(r['image_path']).resolve()) for r in refs}
    for r in refs:
        p=Path(r['image_path']);assert p.is_relative_to(public) and int(r['gt_label'])==int(p.parent.name=='bad')
        with Image.open(p) as im: assert im.size==(4224,1056)
        image_inventory.append(dict(sample_id=r['sample_id'],image_path=str(p),gt_label=int(r['gt_label']),height=1056,width=4224,sha256=sha(p)))
    csvsave('public_image_identity.csv',image_inventory)
    inv=[]
    for seed in range(42,47):
        for geo in ('square512','pb_arpr'):
            d=folder(seed,geo);cp=d/'checkpoints/final.ckpt';tr=read(d/'training_report.json');cfg=read(d/'config.json');h=sha(cp)
            assert h==tr['checkpoint_sha256'] and tr['status']=='PASS' and tr['completed_steps']==70000 and tr['final_validation_completed']
            assert cfg['seed']==seed and cfg.get('input_size',cfg.get('input_size_hw'))==([512,512] if geo=='square512' else [256,1024])
            if seed>42: assert read(d/'evaluation_result.json')['checkpoint_sha256']==h
            with zipfile.ZipFile(cp) as z:
                data=z.read(next(n for n in z.namelist() if n.endswith('/data.pkl')))
                gl=sorted({arg for op,arg,pos in pickletools.genops(data) if op.name=='GLOBAL'})
            inv.append(dict(seed=seed,geometry=geo,checkpoint_path=str(cp),checkpoint_sha256=h,config_path=str(d/'config.json'),config_sha256=sha(d/'config.json'),training_report_sha256=sha(d/'training_report.json'),input_hw=cfg.get('input_size',cfg.get('input_size_hw')),completed_steps=tr['completed_steps'],final_quantiles=tr['final_quantiles'],pickle_globals=gl,has_saved_model_maps=(d/'evaluation/raw_anomaly_maps').exists(),has_saved_native_maps=(d/'native_maps').exists(),status='VERIFIED'))
            print('INVENTORY',seed,geo,h,flush=True)
    save('inventory_evidence.json',dict(models=inv,source_hash_gate=codecheck,seconds=time.perf_counter()-START,resources=resources(),created=now()))
    csvsave('frozen_model_manifest.csv',[{**{k:v for k,v in m.items() if k!='pickle_globals'},'final_quantiles':json.dumps(m['final_quantiles']),'input_hw':str(m['input_hw'])} for m in inv])
    print('INVENTORY_PASS',json.dumps(resources()),flush=True)
def load_checkpoint(path):
    import torch,pickle,types
    from lightning.pytorch.callbacks import Callback
    class InertMonitor(Callback): pass
    class Reader(pickle.Unpickler):
        def find_class(self,module,name):
            if name=='BaselineMonitor' and module in ('train_baseline512','scripts.train_rectangular_efficientad_256x1024','scripts.phase24b_train_square','scripts.phase24b_train_rect'):
                return InertMonitor
            return super().find_class(module,name)
    module=types.ModuleType('audit_pickle');module.Unpickler=Reader;module.load=pickle.load
    return torch.load(path,map_location='cpu',weights_only=False,pickle_module=module)
def inspect_checkpoint():
    s=load_checkpoint(folder(43,'square512')/'checkpoints/final.ckpt')
    print('KEYS',s.keys());print('HPARAMS',s['hyper_parameters']);print('STATE',list(s['state_dict'])[-15:]);print('META',s['phase18b'].keys()); print(resources())
def evaluate():
    import torch,numpy as np,gc,inspect,anomalib,torchvision
    import torch.nn.functional as F
    from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel
    from anomalib.data.utils import read_image
    torch.set_num_threads(4)
    assert anomalib.__version__=='2.6.1' and torch.cuda.is_available()
    refs=rows(OUT/'public_image_identity.csv');ids=[r['sample_id'] for r in refs];labels=[int(r['gt_label']) for r in refs]
    oldrefs=rows(folder(42,'square512')/'evaluation/predictions.csv');pb42={r['sample_id']:r for r in rows(ROOT/'experiments/phase13c_rectangular_256x1024_public/predictions.csv')}
    frozen=read(ROOT/'experiments/phase13c_rectangular_256x1024_public/precheck_report.json')['frozen_inputs']
    for r in refs: assert r['sha256']==sha(r['image_path'])==frozen[r['image_path']]['sha256']
    models=[];gates=[]
    # Complete all provenance gates before any new inference.
    for rec in read(OUT/'inventory_evidence.json')['models']:
        seed=rec['seed'];geo=rec['geometry'];d=folder(seed,geo)
        assert sha(rec['checkpoint_path'])==rec['checkpoint_sha256']
        s=load_checkpoint(rec['checkpoint_path']);hp=s['hyper_parameters'];meta=s['phase18b' if geo=='square512' else 'phase13b']
        assert s['global_step']==70000 and meta['final_validation_completed'] and meta['config']['seed']==seed
        assert meta['config'].get('input_size',meta['config'].get('input_size_hw'))==rec['input_hw']
        assert hp['model_size']=='small' and hp['padding'] is False and hp['pad_maps'] is True and hp['post_processor'] is False
        pre=hp['pre_processor'].eval();assert list(pre.transform.transforms[0].size)==rec['input_hw']
        assert len(pre.transform.transforms)==1 and pre.transform.transforms[0].antialias is True and pre.export_transform.transforms[0].antialias is False
        sd={k.removeprefix('model.'):v for k,v in s['state_dict'].items() if k.startswith('model.')}
        q={k:float(sd['quantiles.'+k]) for k in rec['final_quantiles']};assert q==rec['final_quantiles']
        assert all(math.isfinite(v) for v in q.values()) and q['qb_st']>q['qa_st'] and q['qb_ae']>q['qa_ae']
        model=EfficientAdModel(teacher_out_channels=384,model_size=hp['model_size'],padding=False,pad_maps=True).eval()
        model.load_state_dict(sd,strict=True);model.requires_grad_(False)
        assert torch.isfinite(model.mean_std['mean']).all() and (model.mean_std['std']>0).all()
        old=([float(r['official_image_score']) for r in oldrefs] if geo=='square512' else [float(pb42[sid]['combined_image_score']) for sid in ids]) if seed==42 else [float(r['image_score']) for r in read(d/'image_scores.json')]
        if seed>42:
            sr=read(d/'image_scores.json');assert [r['sample_id'] for r in sr]==ids and [int(r['gt_label']) for r in sr]==labels
            assert abs(auc(labels,old)-read(d/'evaluation_result.json')['image_auroc'])<1e-12
        models.append((rec,model,pre,old))
        gates.append(dict(seed=seed,geometry=geo,status='VERIFIED',checkpoint_sha256=rec['checkpoint_sha256'],quantiles=q,preprocessor=repr(pre),evaluated_preprocessor='transform antialias=True' if geo=='square512' else 'forward/export_transform antialias=False',public_identity='114 original image SHA256 matched Phase13C frozen_inputs',strict_state_load=True))
        del s,sd;gc.collect()
    save('inference_gate.json',dict(status='PASS',created=now(),models=gates,seconds=time.perf_counter()-START,versions=dict(torch=torch.__version__,torchvision=torchvision.__version__,anomalib=anomalib.__version__),resources=resources(),callback_handling='Only serialized BaselineMonitor callbacks replaced with inert Callback for deserialization; no training module imported; strict original model state loaded',preprocessor_forward_source=inspect.getsource(type(models[0][2]).forward)))
    result=[];perimage=[];timings=[]
    torch.cuda.reset_peak_memory_stats()
    for rec,model,pre,old in models:
        t=time.perf_counter();seed=rec['seed'];geo=rec['geometry'];scores=[];maxdiff=0.;inference_seconds=0.
        mode='reuse_saved_native_max' if seed>42 and geo=='pb_arpr' else 'saved_map_native_max' if seed==42 else 'frozen_inference'
        if mode=='frozen_inference': model=model.cuda()
        with torch.inference_mode():
            for i,r in enumerate(refs):
                sid=r['sample_id'];mapsha='';rawsha=''
                if mode=='reuse_saved_native_max': score=old[i]
                else:
                    if seed==42 and geo=='pb_arpr':
                        p=Path(pb42[sid]['combined_path']);arr=np.load(p,allow_pickle=False);assert arr.shape==(1056,4224) and np.isfinite(arr).all();score=float(arr.max());mapsha=sha(p)
                        assert abs(score-old[i])<1e-7
                    else:
                        if seed==42:
                            p=Path(oldrefs[i]['raw_map_path']);raw=torch.from_numpy(np.load(p,allow_pickle=False))[None,None];rawsha=sha(p)
                        else:
                            torch.cuda.synchronize();tt=time.perf_counter()
                            native=read_image(r['image_path'],as_tensor=True);assert tuple(native.shape)==(3,1056,4224)
                            x=pre.transform(native[None]).cuda();raw=model(x).anomaly_map.detach().float().cpu()
                            torch.cuda.synchronize();inference_seconds+=time.perf_counter()-tt
                            del native,x
                        assert tuple(raw.shape)==(1,1,512,512) and torch.isfinite(raw).all()
                        diff=abs(float(raw.max())-old[i]);maxdiff=max(maxdiff,diff)
                        assert diff<=1e-6,('FAIL_FROZEN_REPRODUCTION',seed,sid,diff)
                        arr=F.interpolate(raw,size=(1056,4224),mode='bilinear',align_corners=False)[0,0].numpy()
                        score=float(arr.max());mapsha=hashlib.sha256(arr.tobytes()).hexdigest()
                        if seed>42:
                            mp=OUT/f'model_maps_square512_seed{seed}';mp.mkdir(exist_ok=True);p=mp/(sid+'.npy')
                            with p.open('xb') as f: np.save(f,raw[0,0].numpy(),allow_pickle=False)
                            rawsha=sha(p)
                    assert math.isfinite(score)
                scores.append(score);perimage.append(dict(seed=seed,geometry=geo,sample_id=sid,gt_label=labels[i],old_score=old[i],native_max_score=score,mode=mode,native_map_sha256=mapsha,model_map_sha256=rawsha))
                if (i+1)%38==0: print('SCORES',seed,geo,i+1,'/114',flush=True)
        newauc=auc(labels,scores);oldauc=auc(labels,old)
        result.append(dict(seed=seed,geometry=geo,n=114,normal=24,anomaly=90,old_image_auroc=oldauc,unified_image_auroc=newauc,change_pp=100*(newauc-oldauc),status='VERIFIED',mode=mode))
        timings.append(dict(seed=seed,geometry=geo,wall_seconds=time.perf_counter()-t,inference_decode_preprocess_model_seconds=inference_seconds,max_old_score_reproduction_error=maxdiff))
        print('RESULT',result[-1],flush=True)
        if mode=='frozen_inference':model.cpu();torch.cuda.empty_cache()
    csvsave('image_auroc_unified.csv',result);csvsave('per_image_scores_unified.csv',perimage)
    pairs=[]
    for seed in range(42,47):
        a,b=[r for r in result if r['seed']==seed];pairs.append(dict(seed=seed,square512=a['unified_image_auroc'],pb_arpr=b['unified_image_auroc'],delta_pp=100*(b['unified_image_auroc']-a['unified_image_auroc']),old_delta_pp=100*(b['old_image_auroc']-a['old_image_auroc'])))
    csvsave('paired_seed_deltas.csv',pairs)
    summary={g:dict(mean=statistics.mean(r['unified_image_auroc'] for r in result if r['geometry']==g),std=statistics.stdev(r['unified_image_auroc'] for r in result if r['geometry']==g)) for g in ('square512','pb_arpr')}
    summary['paired_delta_pp']=dict(mean=statistics.mean(r['delta_pp'] for r in pairs),std=statistics.stdev(r['delta_pp'] for r in pairs),positive_seeds=sum(r['delta_pp']>0 for r in pairs))
    save('evaluation_summary.json',dict(status='PASS',summary=summary,timings=timings,seconds=time.perf_counter()-START,gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,gpu_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,resources=resources(),created=now(),native_map_hash_note='seed42 PB hash includes npy header; recomputed maps hash raw float32 bytes; PB43-46 native arrays not retained, exact original score reused with frozen evaluator SHA proof'))
    print('UNIFIED_PASS',json.dumps(summary),flush=True)
if __name__=='__main__':
    {'inventory':inventory,'inspect':inspect_checkpoint,'evaluate':evaluate}[sys.argv[1] if len(sys.argv)>1 else 'inventory']()
