"""Phase14B-C: frozen P2-S1 evaluation, no optimization or calibration."""
import argparse
import gc
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import cv2
import numpy as np
import torch
from PIL import Image
from src.evaluation import multiscale_st_s1_evaluation as e


def now():return datetime.now().astimezone().isoformat()


def map_path(tag,sid):
    folder=e.OUT/'raw_native'/e.SLUGS[tag] if tag in e.MODES else e.OUT/'diagnostic_native'/tag.lower()
    return folder/(sid+'.npy')


def infer(model,ref,report):
    rows={k:[] for k in (*e.MODES,*e.AUX)};predictions=[];errors={}
    for index,sid in enumerate(sorted(ref['mapping'])):
        base=ref['mapping'][sid]
        outputs,err,shapes=e.extract(model,e.input_tensor(model,base['image_path']))
        for k,v in err.items():errors[k]=max(errors.get(k,0.),v)
        record={k:base[k] for k in ('sample_id','image_path','gt_label','gt_mask_path')}
        for tag,raw in outputs.items():
            native=e.native_map(raw);p=map_path(tag,sid)
            if p.exists():raise FileExistsError('FAIL_EXISTING_MAP: '+str(p))
            np.save(p,native,allow_pickle=False)
            record[tag+'_image_score']=float(native.max());record[tag+'_path']=str(p)
            rows[tag].append(dict(base,raw_map_path=str(p),native_shape=e.NATIVE))
        predictions.append(record)
        del outputs,native,raw
        if (index+1)%5==0:print(f'Formal inference {index+1}/114; reconstruction {errors}',flush=True)
    e.pairing_gate(rows);report['reconstruction_errors']=errors
    e.write_csv(e.OUT/'predictions.csv',predictions)
    return rows


def s1_percentiles(rows,ref,background_count):
    """Only empirical separation for S1; no alternative performance ranking."""
    folder=e.OUT/'temp/S1';folder.mkdir(parents=True,exist_ok=True)
    bg=np.memmap(folder/'background_scores_r1.dat',mode='w+',dtype=np.float32,shape=(background_count,))
    offset=0
    for row in rows:
        mask=e.exact.get_mask(row,Image,np);a=np.load(row['raw_map_path'],mmap_mode='r')
        v=e.background_values(a,mask);bg[offset:offset+len(v)]=v;offset+=len(v)
    if offset!=background_count:raise ValueError('FAIL_BACKGROUND_MAPPING')
    bg.flush();print('Sorting S1 background for tie-aware Tiny percentile audit...',flush=True);bg.sort(kind='quicksort');bg.flush()
    for i in range(0,len(bg)-1,1000000):
        v=bg[i:min(len(bg),i+1000001)]
        if not np.isfinite(v).all() or np.any(v[1:]<v[:-1]):raise ValueError('FAIL_BACKGROUND_SORT')
    result={}
    for row in rows:
        sid=row['sample_id']
        if sid not in ref['tiny']:continue
        a=np.load(row['raw_map_path'],mmap_mode='r');mask=np.asarray(e.exact.get_mask(row,Image,np),dtype=bool)
        lower,upper,mid=e.rect.diagnostic.percentiles(bg,a[mask])
        result[sid]=dict(percentile_p95=float(np.percentile(mid,95)),lower_p95=float(np.percentile(lower,95)),upper_p95=float(np.percentile(upper,95)))
    del bg
    return result


def analyses(ref,rows,report,*,reuse_metrics=False):
    samples={};regions={};curves={};previous_metrics=report.get('metrics',{});report['metrics']={};metas={}
    for tag in (*e.MODES,'NEW_FINAL_ST'):
        print('Exact native evaluation: '+tag,flush=True)
        filename=e.SLUGS.get(tag,tag.lower())+'_metrics.json'
        prefix=e.SLUGS.get(tag,tag.lower())
        if reuse_metrics:
            saved=e.read_json(e.OUT/filename);metric={k:saved[k] for k in (*e.KEYS,'approximation')};meta=saved['exact_metadata']
            if metric!=previous_metrics[tag]:raise ValueError('FAIL_CACHED_METRICS')
            ss={r['sample_id']:{k:(float(v) if k in ('PRO005','PRO030','per_image_AUPRO005','per_image_AUPRO030','percentile_p95') else v) for k,v in r.items()}
                for r in e.read_csv(e.OUT/(prefix+'_sample_metrics.csv'))}
            rr=[]
            for r in e.read_csv(e.OUT/(prefix+'_region_metrics.csv')):
                rr.append({k:int(v) if k in ('region_id','pixels') else float(v) if k in ('PRO005','PRO030','AUPRO005','AUPRO030') else v=='True' if k=='outer_boundary' else v for k,v in r.items()})
            curve=[float(r[tag]) for r in e.read_csv(e.OUT/'pro_fpr.csv')]
            if set(ss)!=set(ref['membership']) or len(rr)!=450 or meta['connected_regions']!=450:raise ValueError('FAIL_CACHED_MAPPING')
        else:
            metric,ss,rr,curve,meta=e.metric_engine(rows[tag],tag,ref['membership'],output_dir=e.OUT)
        samples[tag]=ss;regions[tag]={(r['sample_id'],r['region_id']):r for r in rr};curves[tag]=curve;metas[tag]=meta
        report['metrics'][tag]=metric
        if not reuse_metrics:e.write_json(e.OUT/filename,dict(metric,percent={k:metric[k]*100 for k in e.KEYS},exact_metadata=meta))
        # Save each completed branch independently, even if a later gate fails.
        if not reuse_metrics:
            e.write_csv(e.OUT/(prefix+'_sample_metrics.csv'),list(ss.values()))
            e.write_csv(e.OUT/(prefix+'_region_metrics.csv'),rr)
    if not reuse_metrics:
        e.write_csv(e.OUT/'sample_metrics.csv',[r for tag in samples for r in samples[tag].values()])
        e.write_csv(e.OUT/'region_metrics.csv',[r for tag in regions for r in regions[tag].values()])
        e.write_csv(e.OUT/'pro_fpr.csv',[dict(FPR=f,**{k:curves[k][i] for k in curves}) for i,f in enumerate(e.rect.diagnostic.FPRS)])
    shallow=s1_percentiles(rows['S1'],ref,metas[e.MODES[0]]['background_pixels'])
    baseline=ref['baseline'];groups=[]
    for tag in ('RECT_COMBINED',*e.MODES):
        for group,n in e.GROUP_COUNTS.items():
            ids=[sid for sid,r in ref['membership'].items() if r['size_group']==group]
            values=[float(baseline['samples']['COMBINED',sid]['PRO005']) if tag=='RECT_COMBINED' else samples[tag][sid]['PRO005'] for sid in ids]
            groups.append(dict(branch=tag,size_group=group,n=n,mean=float(np.mean(values)),median=float(np.median(values)),std=float(np.std(values))))
    e.write_csv(e.OUT/'size_group_analysis.csv',groups);report['size_groups']=groups
    tiny=[]
    for sid in ref['tiny']:
        row=dict(sample_id=sid,Rect_Combined_PRO005=float(baseline['samples']['COMBINED',sid]['PRO005']),Rect_ST_PRO005=float(baseline['samples']['ST',sid]['PRO005']))
        row.update({tag+'_PRO005':samples[tag][sid]['PRO005'] for tag in samples})
        row.update(Rect_ST_p95=float(baseline['samples']['ST',sid]['percentile_p95']),NEW_FINAL_ST_p95=samples['NEW_FINAL_ST'][sid]['percentile_p95'],
                   S1_p95=shallow[sid]['percentile_p95'],MS_ST_p95=samples[e.MODES[1]][sid]['percentile_p95'],
                   S1_lower_p95=shallow[sid]['lower_p95'],S1_upper_p95=shallow[sid]['upper_p95'])
        tiny.append(row)
    e.write_csv(e.OUT/'tiny_analysis.csv',tiny)
    means={k:float(np.mean([r[k] for r in tiny])) for k in tiny[0] if k.endswith('_PRO005')}
    if abs(means['Rect_Combined_PRO005']*100-8.1571)>.0001 or abs(means['Rect_ST_PRO005']*100-27.8785)>.0001:raise ValueError('FAIL_TINY_REFERENCE')
    report['tiny']=dict(n=12,regions=12,definition='Point PRO at global background FPR0.05, NOT AU-PRO',means=means,
        paired_vs_rect_combined={tag:e.paired([r[tag+'_PRO005'] for r in tiny],[r['Rect_Combined_PRO005'] for r in tiny]) for tag in e.MODES},
        ms_st_vs_rect_st=e.paired([r[e.MODES[1]+'_PRO005'] for r in tiny],[r['Rect_ST_PRO005'] for r in tiny]),percentiles={})
    for tag in ('Rect_ST','NEW_FINAL_ST','S1','MS_ST'):
        v=np.array([r[tag+'_p95'] for r in tiny])
        report['tiny']['percentiles'][tag]=dict(median=float(np.median(v)),ge95=int((v>=95).sum()),between90_95=int(((v>=90)&(v<95)).sum()),lt90=int((v<90).sum()))
    pairrows=[]
    for sid in sorted(ref['membership']):
        row=dict(sample_id=sid,size_group=ref['membership'][sid]['size_group'],Rect_Combined_PRO005=float(baseline['samples']['COMBINED',sid]['PRO005']))
        row.update({tag+'_PRO005':samples[tag][sid]['PRO005'] for tag in e.MODES})
        row['delta_pp']=100*(row[e.MODES[2]+'_PRO005']-row['Rect_Combined_PRO005']);pairrows.append(row)
    e.write_csv(e.OUT/'paired_analysis.csv',pairrows)
    report['paired_90']=e.paired([r[e.MODES[2]+'_PRO005'] for r in pairrows],[r['Rect_Combined_PRO005'] for r in pairrows])
    from scipy.stats import wilcoxon
    d=np.array([r['delta_pp'] for r in pairrows])
    if np.any(abs(d)>e.EPS):
        w=wilcoxon(d,zero_method='wilcox',alternative='two-sided')
        report['paired_90']['wilcoxon']=dict(statistic=float(w.statistic),pvalue=float(w.pvalue),auxiliary_only=True)
    else:report['paired_90']['wilcoxon']=dict(statistic=0.,pvalue=1.,all_ties=True)
    regionrows=[];counts=dict(P2_ONLY=0,RECT_ONLY=0,BOTH=0,BOTH_BAD=0)
    for key,r in regions[e.MODES[0]].items():
        old=baseline['regions'][('COMBINED',)+key]
        if int(old['pixels'])!=r['pixels'] or any(key not in regions[tag] or regions[tag][key]['pixels']!=r['pixels'] for tag in regions):raise ValueError('FAIL_REGION_MAPPING')
        a=float(old['PRO005']);b=regions[e.MODES[2]][key]['PRO005']
        label='BOTH' if a>0 and b>0 else 'RECT_ONLY' if a>0 else 'P2_ONLY' if b>0 else 'BOTH_BAD'
        counts[label]+=1
        regionrows.append(dict(sample_id=key[0],region_id=key[1],pixels=r['pixels'],Rect_Combined_PRO005=a,
            **{tag+'_PRO005':regions[tag][key]['PRO005'] for tag in e.MODES},category=label))
    if sum(counts.values())!=450:raise ValueError('FAIL_REGION_COUNT')
    e.write_csv(e.OUT/'region_paired_analysis.csv',regionrows);report['region_level']=dict(regions=450,success_definition='point PRO005 > 0',**counts)
    delta=lambda a,b:{k:100*(a[k]-b[k]) for k in e.KEYS}
    report['effects']=dict(training=delta(report['metrics'][e.MODES[0]],baseline['metrics']['COMBINED']),
        multiscale_inference=delta(report['metrics'][e.MODES[2]],report['metrics'][e.MODES[0]]),
        total=delta(report['metrics'][e.MODES[2]],baseline['metrics']['COMBINED']))
    st_effects={}
    for key in ('aupro005','aupro030'):
        old=baseline['metrics']['ST'][key];new=report['metrics']['NEW_FINAL_ST'][key];ms=report['metrics'][e.MODES[1]][key]
        a=new-old>b_eps();b=ms-new>b_eps()
        st_effects[key]=dict(Rect_ST=old,NEW_FINAL_ST=new,MS_ST=ms,training_delta_pp=(new-old)*100,map_delta_pp=(ms-new)*100,
            classification='BOTH' if a and b else 'AUX_TRAINING' if a else 'MULTISCALE_MAP' if b else 'NEITHER')
    old=means['Rect_ST_PRO005'];new=means['NEW_FINAL_ST_PRO005'];ms=means[e.MODES[1]+'_PRO005']
    a=new-old>b_eps();b=ms-new>b_eps()
    st_effects['tiny_point_PRO005']=dict(Rect_ST=old,NEW_FINAL_ST=new,MS_ST=ms,training_delta_pp=(new-old)*100,map_delta_pp=(ms-new)*100,
        classification='BOTH' if a and b else 'AUX_TRAINING' if a else 'MULTISCALE_MAP' if b else 'NEITHER')
    report['st_effects']=st_effects
    report['success']=e.classify(report['metrics'][e.MODES[2]],report['runtime']['modes'][e.MODES[2]]['model_only']['mean'],means[e.MODES[2]+'_PRO005'])
    comparison(ref,report);plots(ref,report,curves)


def b_eps():return 1e-10


def comparison(ref,report):
    table=[];train=e.read_json(e.CHECKPOINT.parent.parent/'training_report.json')
    def add(name,metric,tiny,ms,end,mem,hours):
        table.append(dict(Method=name,**{k+'_percent':metric[k]*100 for k in e.KEYS},Tiny_mean_point_PRO005_percent=tiny*100,
            Model_only_ms=ms,End_to_end_ms=end,FPS=1000/ms,Peak_allocated_MiB=mem,Parameters=8058628,Training_hours=hours))
    oldruntime=e.read_json(e.BASE/'runtime_benchmark.json')
    add('Rect baseline',ref['baseline']['metrics']['COMBINED'],report['tiny']['means']['Rect_Combined_PRO005'],107.6683,215.2151,oldruntime['peak_allocated_mib'],11.3917)
    p1tiny=np.mean([float(ref['samples']['COMBINED',sid]['PRO005']) for sid in ref['tiny']])
    add('P1-384 Combined',e.rect.P1384,p1tiny,196.563,330.2937,1055.9692,7.2581)
    for tag in e.MODES:
        rt=report['runtime']['modes'][tag]
        add(tag,report['metrics'][tag],report['tiny']['means'][tag+'_PRO005'],rt['model_only']['mean'],rt['end_to_end']['mean'],rt['model_memory']['peak_allocated_mib'],train['training_elapsed_hours'])
    e.write_csv(e.OUT/'comparison_table.csv',table);report['comparison_table']=table


def plots(ref,report,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    for tag,curve in curves.items():ax.plot(e.rect.diagnostic.FPRS,curve,label=tag)
    ax.set(xlabel='FPR',ylabel='PRO',title='Native exact PRO-FPR');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(e.OUT/'figures/pro_fpr.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    names=['Rect','Final-only','MS-ST','MS-Combined']
    values=[ref['baseline']['metrics']['COMBINED']['aupro005']]+[report['metrics'][k]['aupro005'] for k in e.MODES]
    axes[0].bar(names,np.array(values)*100);axes[0].set_ylabel('Exact AU-PRO@0.05 (%)')
    keys=['Rect_Combined_PRO005']+[k+'_PRO005' for k in e.MODES]
    axes[1].bar(names,[report['tiny']['means'][k]*100 for k in keys]);axes[1].set_ylabel('Tiny mean point PRO@0.05 (%)')
    fig.tight_layout();fig.savefig(e.OUT/'figures/strict_and_tiny.png',dpi=160);plt.close(fig)
    for sid in ref['gallery_ids']:
        row=ref['mapping'][sid]
        with Image.open(row['image_path']) as im:original=np.asarray(im.convert('RGB'))
        images=[original,e.exact.get_mask(row,Image,np),np.load(ref['rect'][sid]['combined_path'])]
        images.extend(np.load(map_path(tag,sid)) for tag in e.MODES)
        images.append(np.load(ref['p1384'][sid]['combined_path']))
        titles=['Original','GT','Rect Combined','NEW Final-only','MS-ST','MS-Combined','P1-384 Combined']
        fig,axes=plt.subplots(7,1,figsize=(14,24))
        for i,(ax,a,title) in enumerate(zip(axes,images,titles)):
            ax.imshow(a,cmap='gray' if i==1 else 'inferno');ax.set_title(title);ax.axis('off')
        fig.suptitle(sid+' | frozen reference-selected cases; independent display scales')
        fig.tight_layout(rect=(0,0,1,.97));fig.savefig(e.OUT/'galleries'/f'{sid}.png',dpi=100);plt.close(fig)
        print('Fixed gallery: '+sid,flush=True)


def summary(r):
    lines=['='*40,'PHASE 14B-C - P2-S1 FORMAL EVALUATION','='*40,
        'Checkpoint: '+str(r.get('strict_load','NOT_RUN')),'SHA256: '+str(r.get('checkpoint_sha256','NOT_RUN')),
        'Sources: '+str(r.get('sources','NOT_RUN')),'Parameters: '+str(r.get('parameters','NOT_RUN')),
        'RECT BASELINE: Image87.9167 / Pixel84.0931 / AU-PRO00519.6522 / AU-PRO03044.8072 /107.6683ms']
    for tag,metric in r.get('metrics',{}).items():
        lines+=['-'*40,tag]+[f'{k}: {metric[k]*100:.6f}%' for k in e.KEYS]
    if 'effects' in r:
        lines+=['-'*40,'EFFECT DECOMPOSITION (percentage points)']
        for k,v in r['effects'].items():lines.append(k+': '+str(v))
        lines+=['ST effect decomposition: '+str(r['st_effects'])]
    if 'tiny' in r:
        lines+=['-'*40,'TINY (point PRO@0.05)']+[f'{k}: {v*100:.6f}%' for k,v in r['tiny']['means'].items()]
        lines+=['Tiny paired: '+str(r['tiny']['paired_vs_rect_combined']),'Tiny MS-ST vs RectST: '+str(r['tiny']['ms_st_vs_rect_st']),
                'Tiny p95 percentiles: '+str(r['tiny']['percentiles']),'90 anomaly paired: '+str(r['paired_90']),'450 regions: '+str(r['region_level'])]
    if 'runtime' in r:
        lines+=['-'*40,'RUNTIME']
        for tag,a in r['runtime']['modes'].items():
            lines.append(f"{tag}: model={a['model_only']['mean']:.4f} ms; FPS={a['model_only']['fps']:.4f}; end-to-end={a['end_to_end']['mean']:.4f} ms; allocated/reserved={a['model_memory']['peak_allocated_mib']:.4f}/{a['model_memory']['peak_reserved_mib']:.4f} MiB")
    if 'success' in r:
        lines+=['-'*40,'SUCCESS']+[f'{k}: {v}' for k,v in r['success'].items()]
    lines+=['Parameter/state unchanged: '+str(r.get('parameters_unchanged',False) and r.get('state_unchanged',False)),
            'Approximation: False','STATUS: '+r['status'],r.get('error_message',''),'='*40]
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument('--precheck-only',action='store_true');p.add_argument('--finish-analysis',action='store_true');args=p.parse_args();os.chdir(ROOT)
    if args.finish_analysis:return finish_analysis()
    if (e.OUT/'evaluation_report.json').exists() or any((e.OUT/'raw_native').glob('*/*.npy')):
        print('FAIL_EXISTING_OUTPUT: frozen report/maps preserved',flush=True);return 1
    for tag in (*e.MODES,*e.AUX):map_path(tag,'placeholder').parent.mkdir(parents=True,exist_ok=True)
    for folder in ('figures','galleries','temp'):(e.OUT/folder).mkdir(parents=True,exist_ok=True)
    r=dict(status='RUNNING',timestamp_start=now(),no_training=True,no_quantile_refit=True,no_test_tuning=True,approximation=False,
        classification_operationalization='Clearly improved uses +1.5pp; other Pareto requires no degradation in four metrics and latency. See implementation; fixed before observing results.')
    model=None
    try:
        torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
        e.no_training_paths();print('Checking frozen references / 114 public sources / 450 regions...',flush=True);ref=e.references()
        r.update(sources=114,normal=24,anomaly=90,regions=450,tiny_count=12,size_group_counts=e.GROUP_COUNTS,frozen_inputs=ref['integrity'],fixed_gallery_sets=ref['gallery_sets'])
        model,audit=e.load_model();r.update(audit);r['parameters_before']=e.parameter_hash(model);r['state_before']=e.state_hash(model)
        x=e.input_tensor(model,ref['mapping'][sorted(ref['mapping'])[0]]['image_path'])
        outputs,errors,shapes=e.extract(model,x);r.update(reconstruction_errors=errors,feature_shapes=shapes)
        mode_errors={};ae_calls={}
        for mode in e.MODES:
            count=[];h=model.model.ae.register_forward_hook(lambda *args:count.append(1))
            try:v=e.forward_mode(model,x,mode)
            finally:h.remove()
            mode_errors[mode]=float((v-outputs[mode]).abs().max());ae_calls[mode]=len(count)
        if max(mode_errors.values())>1e-5 or list(ae_calls.values())!=[1,0,1]:raise ValueError('FAIL_MODE_PATH_RECONSTRUCTION')
        r.update(mode_reconstruction_errors=mode_errors,ae_calls_per_mode=ae_calls,runtime_protocol=e.rect.runtime_protocol())
        if e.parameter_hash(model)!=r['parameters_before'] or e.state_hash(model)!=r['state_before']:raise ValueError('FAIL_STATE_MUTATION')
        free=e.rect.shutil.disk_usage(e.OUT).free;r['disk_free_bytes']=free
        if free<25*1024**3:raise OSError('FAIL_DISK_CAPACITY: need25GiB for three formal and two diagnostic map sets / exact memmaps')
        del x,outputs,v
        r['precheck']='PASS';print('PRECHECK PASS: SHA /strict load /six frozen quantiles /lambda /8058628 /input256x1024 /S1+Final shapes /114=24+90 /450regions /Tiny12 /groups12,18,42,6,12 /exact shared /runtime20warmup5+3rounds /no training',flush=True)
        if args.precheck_only:r['status']='PRECHECK_PASS'
        else:
            rows=infer(model,ref,r);r['runtime']=e.benchmark(model,ref);e.write_json(e.OUT/'runtime_benchmark.json',r['runtime'])
            r.update(parameters_after=e.parameter_hash(model),state_after=e.state_hash(model))
            r.update(parameters_unchanged=r['parameters_after']==r['parameters_before'],state_unchanged=r['state_after']==r['state_before'])
            if not r['parameters_unchanged'] or not r['state_unchanged']:raise ValueError('FAIL_STATE_MUTATION')
            del model;model=None;gc.collect();torch.cuda.empty_cache()
            analyses(ref,rows,r);e.rect.verify_inputs(ref);r['frozen_inputs_unchanged']=True;r['status']='PASS'
    except Exception as exc:
        r.update(status='FAIL',error_message=str(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        if model is not None and 'parameters_before' in r:
            r.update(parameters_after=e.parameter_hash(model),state_after=e.state_hash(model))
            r.update(parameters_unchanged=r['parameters_before']==r['parameters_after'],state_unchanged=r['state_before']==r['state_after'])
        r['timestamp_end']=now();stem='precheck_report' if args.precheck_only else 'evaluation_report'
        if (e.OUT/(stem+'.json')).exists():stem+='_'+datetime.now().strftime('%Y%m%d_%H%M%S')
        e.write_json(e.OUT/(stem+'.json'),r);text=summary(r);(e.OUT/(stem+'.txt')).write_text(text,encoding='utf-8');print(text,flush=True)
    return 0 if r['status'] in ('PASS','PRECHECK_PASS') else 1


def finish_analysis():
    """Read-only reuse of completed inference/metrics after the uint8-mask bug.

    Never reload a model, rerun a forward, or overwrite the failed report/maps.
    """
    revision=1
    while (e.OUT/f'evaluation_report_r{revision}.json').exists():revision+=1
    destination=e.OUT/f'evaluation_report_r{revision}.json'
    old=e.read_json(e.OUT/'evaluation_report.json')
    if old['status']!='FAIL' or 'could not broadcast input array' not in old.get('error_message',''):
        raise ValueError('FAIL_CONTINUATION_SCOPE')
    e.continuation_gate(old);e.checkpoint_gate();e.no_training_paths()
    r=dict(old);r.pop('traceback',None);r.pop('error_message',None)
    r.update(status='RUNNING_POSTPROCESS_R1',previous_failure=old['error_message'],previous_report_sha256=e.digest(e.OUT/'evaluation_report.json'),
        continuation_start=now(),continuation_model_forwards=0,
        repair='S1 foreground and background masks explicitly bool before indexing; frozen predictions, metrics, benchmark and protocol unchanged.')
    r['prior_attempts']=[dict(path=str(p),sha256=e.digest(p),status=e.read_json(p)['status'],error=e.read_json(p).get('error_message')) for p in sorted(e.OUT.glob('evaluation_report*.json'))]
    try:
        torch.set_num_threads(4)
        print('Verifying original frozen inputs and completed evaluation artifacts; no model forward...',flush=True)
        e.rect.verify_inputs({'integrity':old['frozen_inputs']})
        ref=e.references();predictions=e.read_csv(e.OUT/'predictions.csv');e.rect.mapping_gate(predictions)
        rows={k:[] for k in (*e.MODES,*e.AUX)}
        protected=list((e.OUT/'raw_native').glob('*/*.npy'))+list((e.OUT/'diagnostic_native').glob('*/*.npy'))
        protected+=[e.OUT/'evaluation_report.json',e.OUT/'predictions.csv',e.OUT/'runtime_benchmark.json',e.OUT/'sample_metrics.csv',e.OUT/'region_metrics.csv',e.OUT/'pro_fpr.csv']
        protected.extend(e.OUT.glob('evaluation_report*.json'))
        for tag in (*e.MODES,'NEW_FINAL_ST'):
            prefix=e.SLUGS.get(tag,tag.lower())
            protected.extend(e.OUT/(prefix+suffix) for suffix in ('_metrics.json','_sample_metrics.csv','_region_metrics.csv'))
        for p in predictions:
            sid=p['sample_id']
            for tag in rows:
                path=map_path(tag,sid)
                if str(path)!=p[tag+'_path']:raise ValueError('FAIL_CACHED_MAP_PATH')
                a=np.load(path,mmap_mode='r',allow_pickle=False)
                if a.shape!=e.NATIVE or a.dtype!=np.float32 or not np.isfinite(a).all() or float(a.max())!=float(p[tag+'_image_score']):raise ValueError('FAIL_CACHED_MAP')
                rows[tag].append(dict(ref['mapping'][sid],raw_map_path=str(path),native_shape=e.NATIVE))
        e.pairing_gate(rows)
        integrity={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=e.digest(p)) for p in protected}
        analyses(ref,rows,r,reuse_metrics=True)
        e.rect.verify_inputs({'integrity':integrity});e.rect.verify_inputs({'integrity':old['frozen_inputs']})
        r.update(status='PASS',frozen_inputs_unchanged=True,completed_inference_artifacts_unchanged=True,continuation_artifact_integrity=integrity)
    except Exception as exc:
        r.update(status='FAIL',error_message=str(exc),traceback=traceback.format_exc());traceback.print_exc()
    r['timestamp_end']=now();e.write_json(destination,r)
    text=summary(r);destination.with_suffix('.txt').write_text(text,encoding='utf-8');print(text,flush=True)
    return 0 if r['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
