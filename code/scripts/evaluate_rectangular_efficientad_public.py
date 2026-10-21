"""Phase13C frozen rectangular evaluation; no training or adaptation."""
import argparse
import gc
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from PIL import Image
from src.evaluation import rectangular_formal_evaluation as e

def now():return datetime.now().astimezone().isoformat()

def infer(model,ref,r):
    rows={'COMBINED':[],'ST':[]};predictions=[];errors=dict(st_raw=0.,st_normalized=0.,combined=0.)
    for index,sid in enumerate(sorted(ref['mapping'])):
        base=ref['mapping'][sid]
        with torch.no_grad():combined,st,err=e.extract(model,e.input_tensor(model,base['image_path']))
        for k,v in err.items():errors[k]=max(errors[k],v)
        record=dict(sample_id=sid,image_path=base['image_path'],gt_label=base['gt_label'],gt_mask_path=base['gt_mask_path'],shape='1056x4224')
        for tag,raw in (('COMBINED',combined),('ST',st)):
            native=e.native_map(raw);path=e.OUT/('raw_native_combined' if tag=='COMBINED' else 'raw_native_st')/(sid+'.npy')
            if path.exists():raise FileExistsError('FAIL_EXISTING_MAP')
            np.save(path,native,allow_pickle=False);record[tag.lower()+'_image_score']=float(native.max());record[tag.lower()+'_path']=str(path)
            rows[tag].append(dict(base,raw_map_path=str(path),native_shape=e.NATIVE))
        predictions.append(record)
        if (index+1)%5==0:print(f'Formal inference {index+1}/114; reconstruction errors {errors}',flush=True)
    r['reconstruction_errors']=errors;e.write_csv(e.OUT/'predictions.csv',predictions)
    for tag in rows:
        if len(rows[tag])!=114:raise ValueError('FAIL_NATIVE_MAP_COUNT')
    return rows

def analyses(ref,rows,r):
    samples={};curves={};r['metrics']={};regions=[]
    for tag in ('COMBINED','ST'):
        print('Exact native metrics: '+tag,flush=True)
        metric,sample,rr,curve,meta=e.metrics(rows[tag],tag,ref['membership'])
        r['metrics'][tag]=metric;samples[tag]=sample;curves[tag]=curve;regions.extend(rr)
        e.write_json(e.OUT/(tag.lower()+'_metrics.json'),dict(metric,percent={k:metric[k]*100 for k in e.METRIC_KEYS},exact_metadata=meta))
    e.write_csv(e.OUT/'sample_metrics.csv',[a for tag in samples for a in samples[tag].values()])
    e.write_csv(e.OUT/'region_metrics.csv',regions)
    e.write_csv(e.OUT/'pro_fpr.csv',[dict(FPR=f,COMBINED=curves['COMBINED'][i],ST=curves['ST'][i]) for i,f in enumerate(e.diagnostic.FPRS)])
    groups=[]
    for tag in samples:
        for group,n in e.GROUP_COUNTS.items():
            ids=[sid for sid,v in ref['membership'].items() if v['size_group']==group]
            a=dict(branch=tag,size_group=group,n=n)
            for metric in ('PRO005','per_image_AUPRO005'):
                values=[samples[tag][sid][metric] for sid in ids]
                a.update({metric+'_'+key:float(fn(values)) for key,fn in (('mean',np.mean),('median',np.median),('std',np.std))})
            groups.append(a)
    e.write_csv(e.OUT/'size_group_analysis.csv',groups);r['size_groups']=groups
    tiny=[]
    for sid in ref['tiny']:
        b0=float(ref['b0points'][sid]['PRO005'])
        c=samples['COMBINED'][sid];s=samples['ST'][sid]
        tiny.append(dict(sample_id=sid,B0_PRO005=b0,P1384_Combined_PRO005=float(ref['samples']['COMBINED',sid]['PRO005']),
                         P1384_ST_PRO005=float(ref['samples']['ST',sid]['PRO005']),Rect_Combined_PRO005=c['PRO005'],Rect_ST_PRO005=s['PRO005'],
                         Rect_ST_positive_p95_background_percentile=s['percentile_p95'],Combined_delta_vs_B0_pp=100*(c['PRO005']-b0),ST_delta_vs_B0_pp=100*(s['PRO005']-b0)))
    e.write_csv(e.OUT/'tiny_analysis.csv',tiny)
    r['tiny']={'n':12,'metric_definition':'Point PRO at fixed global background FPR0.05; NOT integral AU-PRO',
               'means':{key:float(np.mean([x[key] for x in tiny])) for key in ('B0_PRO005','P1384_Combined_PRO005','P1384_ST_PRO005','Rect_Combined_PRO005','Rect_ST_PRO005')}}
    for tag in ('Combined','ST'):
        delta=np.array([x[tag+'_delta_vs_B0_pp']/100 for x in tiny])
        r['tiny'][tag]=dict(improved=int((delta>e.EPS).sum()),worsened=int((delta<-e.EPS).sum()),unchanged=int((abs(delta)<=e.EPS).sum()),
                            mean_delta_pp=float(delta.mean()*100),median_delta_pp=float(np.median(delta)*100))
    if abs(r['tiny']['means']['P1384_ST_PRO005']*100-31.5432)>.0001:raise ValueError('FAIL_TINY_REFERENCE')
    r['border_groups']=[]
    for tag in ('COMBINED','ST'):
        for outer in (True,False):
            rr=[a for a in regions if a['branch']==tag and a['outer_boundary']==outer]
            r['border_groups'].append(dict(branch=tag,group='OUTER_BOUNDARY_96PX' if outer else 'INTERIOR',regions=len(rr),
                **{key:float(np.mean([a[key] for a in rr])) if rr else None for key in ('PRO005','PRO030','AUPRO005','AUPRO030')}))
    e.write_csv(e.OUT/'outer_boundary_analysis.csv',r['border_groups'])
    r['pareto']=e.pareto(r['metrics']['COMBINED'],r['runtime']['model_only']['mean'])
    comparison(ref,r);plots(ref,curves,groups)

def comparison(ref,r):
    p256=e.read_json(e.B10/'evaluation_report.json');train=e.read_json(e.CHECKPOINT.parent.parent/'training_report.json')
    r['training_cost']=dict(B0_hours=4.2807,P1384_hours=7.2581,Rect_hours=train['training_elapsed_hours'],
                            Rect_vs_P1384_ratio=train['training_elapsed_hours']/7.2581,Rect_vs_B0_ratio=train['training_elapsed_hours']/4.2807)
    table=[]
    def record(name,metrics,hours,trainvram,modelms,endms,infermem):
        table.append(dict(Method=name,**{k+'_percent':metrics[k]*100 for k in e.METRIC_KEYS},Training_hours=hours,Training_peak_allocated_MiB=trainvram,
                          Model_only_ms=modelms,End_to_end_ms=endms,FPS=1000/modelms if modelms else None,Inference_peak_allocated_MiB=infermem,Parameters=8058628))
    record('B0',dict(zip(e.METRIC_KEYS,np.array([61.57,79.54,6.41,22.33])/100)),4.2807,1185.2217,43.1369,None,None)
    record('P1-256',dict(zip(e.METRIC_KEYS,np.array([67.453704,89.938424,7.119403,36.725075])/100)),3.4880,1185.1592,110.9204,None,None)
    record('P1-384 Combined Batch4',e.P1384,7.2581,1561.9448,196.5630,330.2937,1055.9692)
    record('Rect256x1024 Combined',r['metrics']['COMBINED'],train['training_elapsed_hours'],train['peak_gpu_allocated_mib'],r['runtime']['model_only']['mean'],r['runtime']['end_to_end']['mean'],r['runtime']['peak_allocated_mib'])
    r['comparison_table']=table;r['comparison_note']='Frozen user-specified baselines; missing B0/P1-256 end-to-end and inference VRAM left empty, not inferred. Training is slower even if inference is faster.'
    e.write_csv(e.OUT/'comparison_table.csv',table)

def plots(ref,curves,groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    for tag in curves:ax.plot(e.diagnostic.FPRS,curves[tag],label=tag)
    ax.axvline(.05,color='gray',ls='--');ax.set(xlabel='FPR',ylabel='PRO',title='Rectangular native exact PRO');ax.legend();fig.savefig(e.OUT/'figures/pro_fpr.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots();x=np.arange(5)
    for i,tag in enumerate(('COMBINED','ST')):
        ax.bar(x+(i-.5)*.35,[100*a['PRO005_mean'] for a in groups if a['branch']==tag],.35,label=tag)
    ax.set(xticks=x,xticklabels=list('ABCDE'),ylabel='Mean point PRO@0.05 (%)');ax.legend();fig.savefig(e.OUT/'figures/size_group_pro005.png',dpi=150);plt.close(fig)
    for sid in ref['gallery_ids']:
        row=ref['mapping'][sid]
        with Image.open(row['image_path']) as im:original=np.asarray(im.convert('RGB'))
        images=[original,e.exact.get_mask(row,Image,np),e.exact.get_score(row,torch,np),np.load(ref['p1384'][sid]['combined_path']),
                np.load(e.OUT/'raw_native_combined'/f'{sid}.npy'),np.load(e.OUT/'raw_native_st'/f'{sid}.npy')]
        fig,axes=plt.subplots(6,1,figsize=(14,21))
        for index,(ax,a,title) in enumerate(zip(axes,images,['Original','GT','B0','P1-384 Combined','Rect Combined','Rect ST'])):
            ax.imshow(a,cmap='gray' if index==1 else 'inferno');ax.set_title(title);ax.axis('off')
        fig.suptitle(sid+' | fixed reference-selected case; individual display scales only');fig.tight_layout(rect=(0,0,1,.965));fig.savefig(e.OUT/'galleries'/f'{sid}.png',dpi=100);plt.close(fig)
        print('Fixed gallery: '+sid,flush=True)

def summary(r):
    lines=['='*40,'PHASE 13C - RECTANGULAR FORMAL EVALUATION','='*40,f"Checkpoint: {r.get('strict_load')}",f"SHA256: {r.get('checkpoint_sha256')}",
           f"Sources: {r.get('sources')}; Normal: {r.get('normal')}; Anomaly: {r.get('anomaly')}",'Input: 256x1024']
    for tag,metric in r.get('metrics',{}).items():
        lines+=['-'*40,tag]+[f'{k}: {metric[k]*100:.6f}%' for k in e.METRIC_KEYS]
    if 'tiny' in r:
        lines+=['-'*40,'TINY (point PRO@0.05)']+[f'{k}: {v*100:.6f}%' for k,v in r['tiny']['means'].items()]
        for tag in ('Combined','ST'):lines.append(f"{tag} improved/worsened/unchanged vs B0: {r['tiny'][tag]}")
    if 'pareto' in r:
        lines+=['-'*40,'P1-384 COMPARISON']
        for k in e.METRIC_KEYS:lines.append(f'{k}: P1-384={e.P1384[k]*100:.4f}%; Rect={r["metrics"]["COMBINED"][k]*100:.4f}%; retention={r["pareto"]["retention"][k]*100:.4f}%; delta={r["pareto"]["delta_pp"][k]:+.4f} pp')
    if 'runtime' in r:
        a=r['runtime'];lines+=['-'*40,'EFFICIENCY',f'Rect model-only: {a["model_only"]["mean"]:.4f} ms/source','P1-384 Batch4: 196.5630 ms/source',
             f'Latency reduction: {a["latency_reduction_percent"]:.4f}%',f'Speedup: {a["speedup_x"]:.4f}x',f'Rect FPS: {a["model_only"]["fps"]:.4f}',
             f'Rect end-to-end: {a["end_to_end"]["mean"]:.4f} ms/source','P1-384 end-to-end: 330.2937 ms/source',f'End-to-end reduction: {a["end_to_end_reduction_percent"]:.4f}%',
             f'Peak allocated: {a["peak_allocated_mib"]:.4f} MiB',f'Peak reserved: {a["peak_reserved_mib"]:.4f} MiB']
    lines+=['Parameters: '+str(r.get('parameters')),'Training hours: B0=4.2807; P1-384=7.2581; Rect=11.3917']
    if 'pareto' in r:lines+=['-'*40,'PARETO']+[f'{k}: {r["pareto"][k]}' for k in ('RECT_DOMINANT','RECT_STRONG_PARETO','RECT_MODERATE_PARETO','RECT_NOT_COMPETITIVE','classification')]
    lines += [f'Parameter unchanged: {r.get("parameters_unchanged")}; State unchanged: {r.get("state_unchanged")}','Approximation: False','STATUS: '+r['status'],r.get('error_message',''),'='*40]
    return '\n'.join(lines)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--precheck-only',action='store_true');args=parser.parse_args();os.chdir(ROOT)
    # Never let a duplicate invocation overwrite the first formal report or maps.
    if (e.OUT/'evaluation_report.json').exists() or any((e.OUT/f).exists() and any((e.OUT/f).glob('*.npy')) for f in ('raw_native_combined','raw_native_st')):
        print('FAIL_EXISTING_OUTPUT: refusing overwrite; frozen files unchanged',flush=True);return 1
    for f in ('raw_native_combined','raw_native_st','figures','galleries','temp'):(e.OUT/f).mkdir(parents=True,exist_ok=True)
    r=dict(status='RUNNING',timestamp_start=now(),no_training=True,no_adaptation=True,approximation=False,metric_definition='Overall native exact AU-PRO; size groups include separate global-background point PRO and local per-image AU-PRO. Tiny comparison uses point PRO only.')
    model=None
    try:
        torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
        e.no_training_paths();print('Reading frozen references and official public split...',flush=True);ref=e.references()
        r.update(sources=114,normal=24,anomaly=90,regions=450,size_group_counts=e.GROUP_COUNTS,tiny_count=12,fixed_gallery_sets=ref['gallery_sets'],frozen_inputs=ref['integrity'])
        model,details=e.load_model();r.update(details);r['parameters_before']=e.parameter_hash(model);r['state_before']=e.state_hash(model)
        first=ref['mapping'][sorted(ref['mapping'])[0]]
        with torch.no_grad():x=e.input_tensor(model,first['image_path']);g=e.probe(model,x);combined,st,errors=e.extract(model,x)
        expected={'teacher_feature_shape':[1,384,56,248],'student_feature_shape':[1,768,56,248],'st_discrepancy_shape':[1,384,56,248],'ae_feature_shape':[1,384,56,248],'final_anomaly_map_shape':[1,1,256,1024]}
        if any(g[k]!=v for k,v in expected.items()) or g['encoder_layers']['enconv6']['output']!=[1,64,1,25]:raise ValueError('FAIL_SHAPE_GATE')
        r.update(feature_shapes=g,border_audit=dict(invalid_native_equivalent=g['native_invalid_border_equivalent'],invalid_model_border=g['invalid_support_border_model'],outer_region_width=96,definition='Same96px outer subgroup as Phase11C; no internal seams. Support is audited before quantile affine offset, not by thresholding final maps.'),reconstruction_errors=errors,runtime_protocol=e.runtime_protocol())
        if e.parameter_hash(model)!=r['parameters_before'] or e.state_hash(model)!=r['state_before']:raise ValueError('FAIL_STATE_MUTATION')
        if e.shutil.disk_usage(e.OUT).free<10*1024**3:raise OSError('FAIL_DISK_CAPACITY: need10GiB for maps and exact memmaps')
        e.write_json(e.OUT/'feature_shapes.json',g);r['precheck']='PASS'
        print('PRECHECK PASS: SHA / strict load / frozen quantiles / params8058628 / public114=24+90 / input(256,1024) / shapes /450regions /groups12,18,42,6,12 /shared exact /runtime20warmup,5+3rounds /no training',flush=True)
        del x,combined,st
        if args.precheck_only:r['status']='PRECHECK_PASS'
        else:
            rows=infer(model,ref,r);r['runtime']=e.benchmark(model,ref);e.write_json(e.OUT/'runtime_benchmark.json',r['runtime'])
            r['parameters_after']=e.parameter_hash(model);r['state_after']=e.state_hash(model)
            r['parameters_unchanged']=r['parameters_after']==r['parameters_before'];r['state_unchanged']=r['state_after']==r['state_before']
            if not r['parameters_unchanged'] or not r['state_unchanged']:raise ValueError('FAIL_STATE_MUTATION')
            del model;model=None;gc.collect();torch.cuda.empty_cache()
            analyses(ref,rows,r);e.verify_inputs(ref);r['frozen_inputs_unchanged']=True;r['status']='PASS'
    except Exception as exc:
        r.update(status='FAIL',error_message=str(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        if model is not None and 'parameters_before' in r:
            r.update(parameters_after=e.parameter_hash(model),state_after=e.state_hash(model))
            r.update(parameters_unchanged=r['parameters_before']==r['parameters_after'],state_unchanged=r['state_before']==r['state_after'])
        r['timestamp_end']=now();stem='precheck_report' if args.precheck_only else 'evaluation_report'
        if (e.OUT/(stem+'.json')).exists():stem+='_'+datetime.now().strftime('%Y%m%d_%H%M%S')
        e.write_json(e.OUT/(stem+'.json'),r);(e.OUT/(stem+'.txt')).write_text(summary(r),encoding='utf-8');print(summary(r),flush=True)
    return 0 if r['status'] in ('PASS','PRECHECK_PASS') else 1

if __name__=='__main__':raise SystemExit(main())
