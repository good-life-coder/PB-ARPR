"""Isolated R1 namespaces reuse unchanged Phase16 training/evaluation functions."""
import copy
import gc
import importlib.util
import json
import time
from types import SimpleNamespace
import numpy as np
import torch
from src.evaluation import severstal_protocol as original
from src.evaluation import severstal_geometry_r1 as revision

def runtime_statistics(ms):
    return dict(mean_ms=float(np.mean(ms)),median_ms=float(np.median(ms)),p95_ms=float(np.percentile(ms,95)),FPS=1000/float(np.mean(ms)))

def isolated_modules():
    p=SimpleNamespace(**vars(original));p.OUT=revision.OUT;p.PILOT=original.ROOT/'experiments/phase16b_r1_severstal_10k_pilot'
    p.FLAGS=dict(original.FLAGS,architecture_constrained=True,ORIGINAL_PROTOCOL_STATUS='FAIL_RECT_AE_GEOMETRY',ORIGINAL_RAW_RECT=[192,1280],R1_CHANGE_REASON='architecture feasibility only',PERFORMANCE_USED_FOR_R1=False,DEV_USED_FOR_R1_GEOMETRY=False,BLIND_USED_FOR_R1_GEOMETRY=False,METHOD_DEFINITION_UPDATED=True)
    def assets():
        a=copy.deepcopy(original.assets());r=revision.run();sha=original.digest(revision.OUT/'geometry_protocol_r1.json')
        if a['protocol']['canonical_split_sha256']!=revision.SPLIT_SHA:raise ValueError('FAIL_SPLIT')
        a['geometry']=dict(r,rect_hw=r['selected_hw']);a['protocol']['geometry_sha256']=sha
        a['methods']=original.config(r['selected_hw'])
        a['methods']['STD']['description']='Official EfficientAD-S baseline'
        a['methods']['RECT'].update(description='User-proposed Architecture-Constrained PB-ARPR EfficientAD-S',architecture_constrained=True)
        return a
    p.assets=assets
    def clone(name,path):
        spec=importlib.util.spec_from_file_location(name,original.ROOT/path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    t=clone('src.training._severstal_r1_runner','src/training/severstal_pilot.py');t.p=p
    e=clone('src.evaluation._severstal_r1_dev','src/evaluation/severstal_dev_evaluation.py');e.p=p;e.t=t
    @torch.no_grad()
    def runtime(a,guard):
        result={};row=a['groups']['validation_normal'][0]
        for name in ('STD','RECT'):
            m,c=t.load_final(name,a);before=t.old.state_hash(m);x=t.input_image(m,row,guard,'runtime');hw=tuple(a['methods'][name]['input_hw']);times=[]
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            for i in range(120):
                torch.cuda.synchronize();start=time.perf_counter();out=e.forward(m,x,name,hw);torch.cuda.synchronize()
                if i>=20:times.append((time.perf_counter()-start)*1000)
                del out
            if before!=t.old.state_hash(m):raise ValueError('FAIL_RUNTIME_STATE')
            result[name]=dict(**runtime_statistics(times),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,warmup=20,runs=100,cuda_synchronize=True,source_role='validation_normal')
            del m,c,x;gc.collect();torch.cuda.empty_cache()
        p.save(p.PILOT/'runtime_report.json',result);return result
    e.runtime=runtime
    return p,t,e

def execute(precheck_only=False):
    p,t,e=isolated_modules();a,guard,expected=t.precheck()
    if precheck_only:return
    try:
        for steps in (3,200):
            for name in ('STD','RECT'):t.train(name,steps,a,guard,expected)
        reports={name:t.train(name,10000,a,guard,expected) for name in ('STD','RECT')}
        result=e.run(a,guard,reports)
        result['geometry_revision']=revision.run();result['geometry_sha']=p.digest(p.OUT/'geometry_protocol_r1.json')
        p.save(p.PILOT/'phase16_r1_report.json',result)
        (p.PILOT/'phase16_r1_report.txt').write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result
    except BaseException as exc:
        p.save(p.PILOT/'failure_report.json',dict(status='FAIL',error=repr(exc),**p.FLAGS));raise
