"""Phase10C: native saved-map diagnosis only. No model imports or forwards."""
import argparse
import csv
import gc
import hashlib
import json
import re
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from src.evaluation import strict_fpr_diagnostics as d
from src.evaluation import exact_aupro as exact

OUT=ROOT/'experiments/phase10c_strict_fpr_diagnostic'
B0=ROOT/'experiments/phase07c_baseline256_public'
P1=ROOT/'experiments/phase10b_p1_halo96_public'
D0=ROOT/'experiments/phase07d_b0_failure_analysis'
A9=ROOT/'experiments/phase09a_tile_geometry'
GROUP007=[f'{42+i:03d}_007_{c}' for i,c in enumerate(('overexposed','regular','shift_1','shift_2','shift_3','underexposed'))]
QUANTILES=[.5,.9,.95,.975,.99,.995,.999,.9995]


def read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def save_csv(name,rows):
    if not rows:return
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def precheck():
    a=read_csv(B0/'predictions.csv');b=read_csv(P1/'predictions.csv');pairs=d.pair_samples(a,b)
    if len(pairs)!=114:raise ValueError('Expected114 pairs')
    if [int(x['gt_label']) for x,_ in pairs].count(0)!=24:raise ValueError('Expected24 normals')
    if len(list((B0/'raw_anomaly_maps').glob('*.npy')))!=114 or len(list((P1/'raw_native_maps').glob('*.npy')))!=114:raise ValueError('Expected114 maps each')
    formal=json.loads((P1/'evaluation_report.json').read_text(encoding='utf-8'))
    if formal['status']!='PASS':raise ValueError('P1 formal evaluation not PASS')
    membership={r['sample_id']:r for r in read_csv(D0/'sample_analysis.csv')}
    tiny={sid for sid,r in membership.items() if r['size_group']=='A'}
    expected={(r['sample_id'],int(r['region_id'])):r for r in read_csv(A9/'region_tile_analysis.csv')}
    hashes={};total=0;seen=set();mapping={'B0':{},'P1':{}}
    for left,right in pairs:
        sid=left['sample_id'];mask=None
        for tag,row,folder,shape in (('B0',left,B0/'raw_anomaly_maps',(256,256)),('P1',right,P1/'raw_native_maps',(1056,4224))):
            path=Path(row['raw_map_path']).resolve()
            if path.parent!=folder.resolve() or path in seen:raise ValueError('Raw path mismatch/duplicate')
            seen.add(path);raw=np.load(path,mmap_mode='r',allow_pickle=False)
            if raw.dtype!=np.float32 or raw.shape!=shape or not np.isfinite(raw).all():raise ValueError('Invalid raw map')
            with Image.open(row['image_path']) as im:
                if im.size!=(4224,1056):raise ValueError('Native dimensions differ')
            row['native_shape']=(1056,4224);mapping[tag][sid]=row;hashes[str(path)]=digest(path)
            if int(row['gt_label']):
                mp=Path(row['gt_mask_path']).resolve()
                expected_mp=ROOT/'datasets/MVTec_AD_2/sheet_metal/test_public/ground_truth/bad'/(Path(row['image_path']).stem+'_mask.png')
                if mp!=expected_mp.resolve():raise ValueError('GT path mismatch')
                current=exact.get_mask(row,Image,np)
                if mask is not None and not np.array_equal(current,mask):raise ValueError('B0/P1 GT differs')
                mask=current;hashes[str(mp)]=digest(mp)
        if mask is not None:
            labels,sizes=d.connected(mask);total+=len(sizes)-1
            frozen=membership[sid]
            if int(mask.sum())!=int(frozen['defect_pixel_count']):raise ValueError('Frozen area/group mismatch')
            for rid in range(1,len(sizes)):
                if int(expected[(sid,rid)]['pixel_count'])!=int(sizes[rid]):raise ValueError('Region mapping differs')
    if total!=450 or len(expected)!=450 or len(tiny)!=12 or not set(GROUP007)<=set(membership):raise ValueError('450regions/tiny12/007 membership failed')
    for path in (B0/'predictions.csv',P1/'predictions.csv',D0/'sample_analysis.csv',A9/'region_tile_analysis.csv'):
        hashes[str(path)]=digest(path)
    return mapping,membership,tiny,hashes,dict(B0_maps=114,P1_maps=114,GT_images=114,regions=total,tiny_members=12,
        no_model_loaded=True,no_forward_performed=True,status='PASS')


def caches(mapping,report,torch,cv2):
    result={};tails=[];curves={}
    for tag in ('B0','P1'):
        print(f'{tag}: rebuilding exact CPU memmaps from saved native predictions',flush=True)
        path=OUT/'temp'/tag;meta={}
        exact.native(list(mapping[tag].values()),meta,torch,cv2,np,Image,temp_dir=path)
        bg=np.memmap(path/'background_scores.dat',mode='r',dtype=np.float32,shape=(meta['background_pixels'],))
        pos=np.memmap(path/'positive_scores.dat',mode='r',dtype=np.float32,shape=(meta['positive_pixels'],))
        weights=np.memmap(path/'positive_weights.dat',mode='r',dtype=np.float64,shape=(len(pos),))
        curves[tag]=exact.pro_at_fprs(bg,pos,weights,d.FPRS,np).tolist()
        source=ROOT/'experiments/phase07c_r1_aupro_exact/aupro_report.json' if tag=='B0' else P1/'evaluation_report.json'
        frozen=json.loads(source.read_text(encoding='utf-8'));frozen=frozen if tag=='B0' else frozen['metrics']
        if any(abs(meta[f'aupro_{a:.2f}']-frozen[f'aupro_{a:.2f}'])>1e-7 for a in (.05,.30)):raise ValueError('Native exact cache differs from frozen result')
        q={p:d.background_quantile(bg,p) for p in QUANTILES+[.7]}
        density=(len(bg)-np.searchsorted(bg,q[.95],side='right'))/len(bg)
        tails.append(dict(model=tag,N_background=len(bg),minimum=float(bg[0]),**{f'q{100*p:g}':q[p] for p in QUANTILES},maximum=float(bg[-1]),actual_top5_density=float(density)))
        result[tag]=(bg,q,float(density));report.setdefault('exact_cache',{})[tag]=meta
        del pos,weights;gc.collect()
    save_csv('background_tail_analysis.csv',tails)
    report['background_tails']=tails;report['overall_curves']=curves
    return result


def process(mapping,membership,tiny,cache,report,torch):
    zones=d.spatial_zones();sample_rows=[];region_rows=[];spatial=[];tiles=[];normals=[];all_tails=[]
    all_totals={tag:{z:[0,0,0] for z in zones} for tag in cache}
    core_totals={tag:{i:[0,0] for i in range(4)} for tag in cache}
    tiny_curves={tag:[] for tag in cache}
    for index,sid in enumerate(sorted(mapping['B0'])):
        base=mapping['B0'][sid];mask=exact.get_mask(base,Image,np);labels,sizes=d.connected(mask);fg=labels>0
        yy,xx=np.nonzero(fg);ids=labels[fg];k=len(sizes)-1
        condition=re.search(r'(regular|shift_[123]|overexposed|underexposed)$',Path(base['image_path']).stem)
        common=dict(sample_id=sid,gt_label=int(base['gt_label']),condition=condition.group(1) if condition else 'UNKNOWN',
                    group=membership[sid]['size_group'] if k else 'NORMAL',defect_ratio=int(mask.sum())/mask.size)
        for tag in ('B0','P1'):
            score=exact.get_score(mapping[tag][sid],torch,np);bg,q,density=cache[tag]
            high=d.tail_mask(score,q[.95]) & ~fg;nb=int((~fg).sum());nt=int(high.sum())
            record=dict(common,model=tag,top5_count=nt,top5_density=nt/nb,top5_enrichment=(nt/nb)/density)
            for zone,zm in zones.items():
                area=int(zm.sum());background=int((zm&~fg).sum());top=int((zm&high).sum())
                all_totals[tag][zone]=[x+y for x,y in zip(all_totals[tag][zone],(area,background,top))]
                record[zone+'_top5_count']=top;record[zone+'_share_of_hotspots']=top/nt if nt else 0.
            for i in range(4):
                sl=slice(i*1056,(i+1)*1056);background=int((~fg[:,sl]).sum());top=int(high[:,sl].sum())
                core_totals[tag][i][0]+=background;core_totals[tag][i][1]+=top
                record[f'core{i}_top5_count']=top;record[f'core{i}_share_of_hotspots']=top/nt if nt else 0.
            all_tails.append(record)
            if not k:
                normals.append(record);sample_rows.append(dict(common,model=tag,PRO005=None,PRO030=None,relaxation_gain=None,
                    local_PRO005=None,local_PRO030=None,pixel_auroc=None,**{key:None for key in d.region_statistics(bg,np.array([q[.95]]),q)}))
                continue
            values=score[fg];weights=1/(k*sizes[ids].astype(np.float64))
            lower,upper,mid=d.percentiles(bg,values)
            per_pixel=np.empty(len(values),dtype=[('region_id','i4'),('percentile_lower','f8'),('percentile_upper','f8'),('percentile_mid','f8')])
            per_pixel['region_id']=ids;per_pixel['percentile_lower']=lower;per_pixel['percentile_upper']=upper;per_pixel['percentile_mid']=mid
            percentile_dir=OUT/'positive_percentiles'/tag;percentile_dir.mkdir(parents=True,exist_ok=True)
            np.save(percentile_dir/(sid+'.npy'),per_pixel,allow_pickle=False)
            del per_pixel,lower,upper,mid
            curve=exact.pro_at_fprs(bg,values,weights,d.FPRS,np)
            local_bg=score[~fg];local_bg.sort(kind='quicksort')
            local_curve=exact.pro_at_fprs(local_bg,values,weights,[.05,.30],np)
            pixel_auc=float(d.percentiles(local_bg,values)[2].mean()/100)
            stats=d.region_statistics(bg,values,q)
            sr=dict(common,model=tag,PRO005=float(curve[7]),PRO030=float(curve[-1]),relaxation_gain=float(curve[-1]-curve[7]),
                    local_PRO005=float(local_curve[0]),local_PRO030=float(local_curve[1]),pixel_auroc=pixel_auc,**stats)
            sample_rows.append(sr)
            for rid in range(1,k+1):
                select=ids==rid;v=values[select];x=xx[select];y=yy[select]
                stats=d.region_statistics(bg,v,q);rc,area=d.region_pro(bg,v)
                rr=dict(common,model=tag,region_id=rid,region_size=int(sizes[rid]),bbox=json.dumps([int(x.min()),int(y.min()),int(x.max()),int(y.max())]),
                    PRO005=float(rc[7]),PRO030=float(rc[-1]),relaxation_gain=float(rc[-1]-rc[7]),
                    exact_AUPRO005=area[.05],exact_AUPRO030=area[.30],global_PRO005_contribution=float(rc[7])/450,
                    global_PRO030_contribution=float(rc[-1])/450,**stats)
                region_rows.append(rr)
                if sid in tiny:tiny_curves[tag].append(rc)
            del local_bg
        if (index+1)%10==0:print(f'Regions, spatial tails and paired samples:{index+1}/114',flush=True)
    for tag,totals in all_totals.items():
        for zone,(area,nb,nt) in totals.items():spatial.append(dict(model=tag,zone=zone,zone_area_pixels=area,zone_background_pixels=nb,
            top5_background_pixels=nt,**d.enrichment(nt,nb,cache[tag][2])))
        for i,(nb,nt) in core_totals[tag].items():tiles.append(dict(model=tag,core=i,background_pixels=nb,top5_background_pixels=nt,**d.enrichment(nt,nb,cache[tag][2])))
    save_csv('all_region_diagnostic.csv',region_rows);save_csv('tiny_region_diagnostic.csv',[a for a in region_rows if a['sample_id'] in tiny])
    save_csv('sample_diagnostic.csv',sample_rows);save_csv('tiny_sample_diagnostic.csv',[a for a in sample_rows if a['sample_id'] in tiny])
    save_csv('spatial_fp_analysis.csv',spatial);save_csv('tile_fp_analysis.csv',tiles);save_csv('normal_tail_analysis.csv',normals)
    save_csv('sample_background_tail_analysis.csv',all_tails)
    save_csv('group007_background_tail_analysis.csv',[a for a in all_tails if a['sample_id'] in GROUP007])
    report.update(spatial=spatial,cores=tiles,tiny_curves={tag:np.mean(v,axis=0).tolist() for tag,v in tiny_curves.items()})
    return sample_rows,region_rows,normals


def evidence(samples,regions,normals,tiny,mapping,cache,report):
    tr=[a for a in regions if a['sample_id'] in tiny and a['model']=='P1'];n=len(tr)
    p95=np.array([a['percentile_p95'] for a in tr]);p5=np.array([a['PRO005'] for a in tr]);p30=np.array([a['PRO030'] for a in tr])
    h8=(p5<=.01)&(p30>=.10)&(p95>=90);h9=(p30<=.01)&(p95<90)
    def support(mask):return 'SUPPORTED' if mask.sum()>n/2 else 'PARTIALLY_SUPPORTED' if mask.any() else 'NOT_SUPPORTED'
    t=dict(samples=12,regions=n,p95_ge95=int((p95>=95).sum()),p95_90_to95=int(((p95>=90)&(p95<95)).sum()),p95_lt90=int((p95<90).sum()),
        zero005_positive030=int(((p5<=d.EPS)&(p30>d.EPS)).sum()),zero005_zero030=int(((p5<=d.EPS)&(p30<=d.EPS)).sum()),
        mean_PRO005=float(p5.mean()),mean_PRO030=float(p30.mean()),median_percentile_p95=float(np.median(p95)),
        median_fraction_above_q95=float(np.median([a['fraction_pixels_above_bg_q95'] for a in tr])),
        median_fraction_above_q70=float(np.median([a['fraction_pixels_above_bg_q70'] for a in tr])),
        h8_regions=int(h8.sum()),h9_regions=int(h9.sum()),H8_BACKGROUND_TAIL=support(h8),H9_WEAK_TINY_SIGNAL=support(h9),
        interpretation='Descriptive compatibility, not causal identification in feature space. Fixed rules: near-zero PRO<=.01; recovery PRO030>=.10 and p95>=90; weak PRO030<=.01 and p95<90. Strict majority supports, nonzero minority partially supports. No operating threshold is selected.')
    report['tiny']=t
    t['mean_unit']='Equal GT region weight, global full-test background FPR'
    tiny_samples=[a for a in samples if a['sample_id'] in tiny and a['model']=='P1']
    t['sample_mean_PRO005']=float(np.mean([a['PRO005'] for a in tiny_samples]))
    t['sample_mean_PRO030']=float(np.mean([a['PRO030'] for a in tiny_samples]))
    byregion={(a['sample_id'],a['region_id'],a['model']):a for a in regions};complement=[];counts={}
    keys=sorted({(a['sample_id'],a['region_id']) for a in regions})
    for threshold in (0.,.1,.25):
        c={key:0 for key in ('BOTH_GOOD','B0_ONLY','P1_ONLY','BOTH_BAD')}
        for sid,rid in keys:
            b=byregion[(sid,rid,'B0')]['PRO005'];p=byregion[(sid,rid,'P1')]['PRO005'];a=b>threshold;z=p>threshold
            label='BOTH_GOOD' if a and z else 'B0_ONLY' if a else 'P1_ONLY' if z else 'BOTH_BAD';c[label]+=1
            complement.append(dict(sample_id=sid,region_id=rid,good_threshold=threshold,B0_PRO005=b,P1_PRO005=p,category=label))
        union=450-c['BOTH_BAD'];c['success_jaccard']=c['BOTH_GOOD']/union if union else None;counts[str(threshold)]=c
    save_csv('b0_p1_complementarity.csv',complement);report['region_complementarity']=counts
    bysample={(a['sample_id'],a['model']):a for a in samples}
    paired=[]
    for sid in sorted(mapping['B0']):
        if not int(mapping['B0'][sid]['gt_label']):continue
        b=bysample[(sid,'B0')];p=bysample[(sid,'P1')];delta=p['PRO005']-b['PRO005']
        paired.append(dict(sample_id=sid,B0_PRO005=b['PRO005'],P1_PRO005=p['PRO005'],delta=delta,
            comparison='P1_BETTER' if delta>d.EPS else 'B0_BETTER' if delta<-d.EPS else 'TIE'))
    save_csv('sample_complementarity.csv',paired)
    report['sample_complementarity']={label:sum(a['comparison']==label for a in paired) for label in ('P1_BETTER','B0_BETTER','TIE')}
    report['sample_complementarity'].update(mean_delta=float(np.mean([a['delta'] for a in paired])),median_delta=float(np.median([a['delta'] for a in paired])),
        delta_quantiles=np.percentile([a['delta'] for a in paired],[0,10,25,50,75,90,100]).tolist())
    # Condition metadata only; source image scores retain frozen per-model definitions.
    conditions=[]
    for tag in ('B0','P1'):
        ids=sorted(mapping[tag]);scores=np.array([float(mapping[tag][sid]['official_image_score']) for sid in ids])
        ranks=d.percentiles(np.sort(scores),scores)[2];rankmap=dict(zip(ids,ranks))
        for condition in sorted({a['condition'] for a in samples}):
            rows=[a for a in samples if a['condition']==condition and a['model']==tag];an=[a for a in rows if a['gt_label']]
            conditions.append(dict(model=tag,condition=condition,sample_count=len(rows),anomaly_count=len(an),
                image_score_percentile=float(np.mean([rankmap[a['sample_id']] for a in rows])),
                mean_pixel_auroc=float(np.mean([a['pixel_auroc'] for a in an])) if an else None,
                mean_PRO005=float(np.mean([a['PRO005'] for a in an])) if an else None))
    save_csv('condition_analysis.csv',conditions)
    cond_delta=[]
    for b in [a for a in conditions if a['model']=='B0']:
        p=next(a for a in conditions if a['condition']==b['condition'] and a['model']=='P1')
        cond_delta.append(dict(condition=b['condition'],sample_count=b['sample_count'],image_percentile_delta=p['image_score_percentile']-b['image_score_percentile'],
            pixel_auroc_delta=p['mean_pixel_auroc']-b['mean_pixel_auroc'] if b['mean_pixel_auroc'] is not None else None,
            PRO005_delta=p['mean_PRO005']-b['mean_PRO005'] if b['mean_PRO005'] is not None else None))
    save_csv('condition_delta.csv',cond_delta)
    masks={sid:exact.get_mask(mapping['B0'][sid],Image,np) for sid in GROUP007};geometry=[];iou=[]
    for sid,mask in masks.items():
        y,x=np.nonzero(mask);geometry.append(dict(sample_id=sid,area=int(mask.sum()),centroid=[float(x.mean()),float(y.mean())],bbox=[int(x.min()),int(y.min()),int(x.max()),int(y.max())]))
    for i,sid in enumerate(GROUP007):
        for other in GROUP007[i+1:]:
            a=masks[sid].astype(bool);b=masks[other].astype(bool);iou.append(dict(first=sid,second=other,identical=bool(np.array_equal(a,b)),IoU=float((a&b).sum()/(a|b).sum())))
    save_csv('group007_geometry_pairs.csv',iou);save_csv('group007_analysis.csv',[a for a in samples if a['sample_id'] in GROUP007])
    report['group007']=dict(geometry=geometry,pairwise=iou,all_identical=all(a['identical'] for a in iou),min_IoU=min(a['IoU'] for a in iou),
        failure_mechanism='INCONCLUSIVE',reason='Raw score scales are not comparable. Global/local PRO and rank evidence describe separation, but cannot uniquely identify defect response drop versus background-tail increase from saved output maps alone.')
    report['group007']['model_means']={tag:{key:float(np.mean([a[key] for a in samples if a['sample_id'] in GROUP007 and a['model']==tag])) for key in ('PRO005','PRO030','local_PRO005','local_PRO030','percentile_p95','percentile_median','pixel_auroc')} for tag in ('B0','P1')}
    spatial={(a['model'],a['zone']):a for a in report['spatial']}
    highest=max([a for a in report['cores'] if a['model']=='P1'],key=lambda a:a['density']);report['highest_density_core']=highest
    report['top_normal_samples']={tag:sorted([a for a in normals if a['model']==tag],key=lambda a:(-a['top5_enrichment'],a['sample_id']))[:10] for tag in ('B0','P1')}
    questions={}
    questions['Q1']=dict(status='SUPPORTED_FACT' if (p30>p5+d.EPS).sum()>n/2 else 'MIXED' if (p30>p5+d.EPS).any() else 'NOT_SUPPORTED',evidence=t)
    questions['Q2']=dict(status='SUPPORTED_FACT',median_region_p95=t['median_percentile_p95'],bands=[t['p95_ge95'],t['p95_90_to95'],t['p95_lt90']])
    questions['Q3']=dict(status='MIXED' if h8.any() and h9.any() else 'SUPPORTED_FACT' if h8.sum()>n/2 or h9.sum()>n/2 else 'INCONCLUSIVE',H8=t['H8_BACKGROUND_TAIL'],H9=t['H9_WEAK_TINY_SIGNAL'])
    for key,zone in (('Q4','OUTER_BOUNDARY'),('Q5','INTERNAL_TILE_SEAM')):
        b=spatial[('B0',zone)]['enrichment'];p=spatial[('P1',zone)]['enrichment']
        questions[key]=dict(status='SUPPORTED_FACT' if p>=2 else 'MIXED' if p>1 else 'NOT_SUPPORTED',B0_enrichment=b,P1_enrichment=p,
            rule='2x is fixed descriptive strong-enrichment marker; does not prove reflection/tile causation.')
    questions['Q6']=dict(status='SUPPORTED_FACT' if highest['enrichment']>=2 else 'MIXED' if highest['enrichment']>1 else 'NOT_SUPPORTED',evidence=highest)
    questions['Q7']=dict(status='INCONCLUSIVE',evidence=report['group007'])
    stable=all(c['B0_ONLY']>0 and c['P1_ONLY']>0 for c in counts.values())
    questions['Q8']=dict(status='SUPPORTED_FACT' if stable else 'MIXED' if counts['0.0']['B0_ONLY'] and counts['0.0']['P1_ONLY'] else 'NOT_SUPPORTED',
        evidence=counts,note='Stability only across three fixed existence cutoffs on this frozen test set; not across seeds or datasets. No fusion constructed.')
    report['questions']=questions
    return complement


def plots(samples,regions,normals,complement,mapping,cache,tiny,report,torch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def finish(fig,name):fig.tight_layout();fig.savefig(OUT/'figures'/name,dpi=140);plt.close(fig)
    for name,curves in (('tiny_pro_fpr_b0_vs_p1.png',report['tiny_curves']),('overall_pro_fpr_b0_vs_p1.png',report['overall_curves'])):
        fig,ax=plt.subplots(figsize=(8,5))
        for tag,v in curves.items():ax.plot(d.FPRS,v,marker='o',label=tag)
        ax.axvline(.05,ls='--',color='black');ax.set(xlabel='Global background FPR',ylabel='Equal-region PRO',ylim=(0,1));ax.legend();finish(fig,name)
    fig,ax=plt.subplots(figsize=(8,5))
    for tag in ('B0','P1'):ax.hist([a['percentile_p95'] for a in regions if a['sample_id'] in tiny and a['model']==tag],bins=np.linspace(0,100,21),alpha=.5,label=tag)
    ax.set(xlabel='Tiny region positive p95 background percentile',ylabel='Region count');ax.legend();finish(fig,'tiny_positive_percentile_distribution.png')
    for name,source,field,value in (('background_top5_spatial_enrichment.png',report['spatial'],'zone','enrichment'),('background_top5_core_density.png',report['cores'],'core','density')):
        fig,ax=plt.subplots(figsize=(9,5));labels=list(dict.fromkeys(a[field] for a in source));x=np.arange(len(labels))
        for offset,tag in ((-.18,'B0'),(.18,'P1')):ax.bar(x+offset,[next(a[value] for a in source if a['model']==tag and a[field]==z) for z in labels],width=.36,label=tag)
        ax.set_xticks(x,labels);ax.set_ylabel(value);ax.legend();finish(fig,name)
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(6)
    for offset,tag in ((-.18,'B0'),(.18,'P1')):ax.bar(x+offset,[next(a['PRO005'] for a in samples if a['sample_id']==sid and a['model']==tag) for sid in GROUP007],width=.36,label=tag)
    ax.set_xticks(x,[s.split('_',2)[2] for s in GROUP007],rotation=20);ax.set_ylabel('Global-FPR PRO@.05');ax.legend();finish(fig,'group007_b0_vs_p1.png')
    comp=[a for a in complement if a['good_threshold']==0]
    fig,ax=plt.subplots(figsize=(6,6));ax.scatter([a['B0_PRO005'] for a in comp],[a['P1_PRO005'] for a in comp],s=10);ax.plot([0,1],[0,1],'k--');ax.set(xlabel='B0 region PRO@.05',ylabel='P1 region PRO@.05');finish(fig,'region_complementarity_scatter.png')
    selection={'tiny':sorted(tiny),'group007':GROUP007,'normal_tail_top10':[a['sample_id'] for a in report['top_normal_samples']['P1']]}
    for category in ('P1_ONLY','B0_ONLY'):
        selection[category]=[(a['sample_id'],a['region_id']) for a in sorted(comp,key=lambda a:(a['sample_id'],a['region_id'])) if a['category']==category][:10]
    report['gallery_selection']=selection;report['gallery_count']=0
    for group,entries in selection.items():
        folder=OUT/'galleries'/group;folder.mkdir(exist_ok=True)
        for entry in entries:
            sid,rid=(entry,None) if isinstance(entry,str) else entry
            row=mapping['B0'][sid];mask=exact.get_mask(row,Image,np)
            with Image.open(row['image_path']) as im:rgb=np.asarray(im.convert('RGB').resize((1200,300),Image.Resampling.BILINEAR))
            arrays=[exact.get_score(mapping[tag][sid],torch,np) for tag in ('B0','P1')]
            tail=(arrays[1]>cache['P1'][1][.95]) & (mask==0)
            fig,axes=plt.subplots(5,1,figsize=(14,16))
            axes[0].imshow(rgb)
            for ax,data,title in zip(axes[1:],[mask,arrays[0],arrays[1],tail],['GT','B0 (display scale only)','P1 (display scale only)','P1 top5 background']):
                display=np.asarray(Image.fromarray(data.astype(np.float32)).resize((1200,300),Image.Resampling.NEAREST if title in ('GT','P1 top5 background') else Image.Resampling.BILINEAR))
                ax.imshow(display,cmap='gray' if title in ('GT','P1 top5 background') else 'jet');ax.set_title(title)
            for ax in axes:ax.axis('off')
            fig.suptitle(sid+(f' | selected region {rid}' if rid else '')+' | fixed diagnostic selection; independent heatmap display scales')
            fig.tight_layout();fig.savefig(folder/(sid+(f'_r{rid}' if rid else '')+'.png'),dpi=100);plt.close(fig);report['gallery_count']+=1


def format_report(report):
    lines=['='*40,'PHASE 10C - STRICT-FPR BOTTLENECK AUDIT','='*40]
    if not all(key in report for key in ('tiny','spatial','highest_density_core','group007','region_complementarity','sample_complementarity','questions')):
        return '\n'.join(lines+[json.dumps({k:report[k] for k in ('status','error_message','precheck') if k in report},indent=2)])
    t=report['tiny'];n=t['regions']
    lines += [f"Tiny samples: {t['samples']}",f'Tiny regions: {n}','','TINY SIGNAL',
        f"P1 tiny mean PRO@0.05: {t['mean_PRO005']:.6f}",f"P1 tiny mean PRO@0.30: {t['mean_PRO030']:.6f}",
        f"Regions p95 >=95th background percentile: {t['p95_ge95']} / {n}",
        f"Regions p95 90-95: {t['p95_90_to95']} / {n}",f"Regions p95 <90: {t['p95_lt90']} / {n}",
        f"PRO005=0 but PRO030>0: {t['zero005_positive030']} / {n}",f"PRO005=0 and PRO030=0: {t['zero005_zero030']} / {n}",
        f"H8 Background-tail limitation: {t['H8_BACKGROUND_TAIL']}",f"H9 Weak tiny-signal limitation: {t['H9_WEAK_TINY_SIGNAL']}",
        '','BACKGROUND TOP5']
    for tag in ('P1','B0'):
        for row in report['spatial']:
            if row['model']==tag:lines.append(f"{tag} {row['zone']} enrichment: {row['enrichment']:.4f}x")
    lines += [f"Highest-density Core: Core{report['highest_density_core']['core']}",'','GROUP 007',
        f"GT geometry: identical={report['group007']['all_identical']}; minimum IoU={report['group007']['min_IoU']:.6f}",
        f"Failure mechanism: {report['group007']['failure_mechanism']}",json.dumps(report['group007']['model_means'],indent=2),
        '','COMPLEMENTARITY','450 regions:',json.dumps(report['region_complementarity'],indent=2),
        '90 anomaly images:',json.dumps(report['sample_complementarity'],indent=2),'']
    titles=['tiny recovery at 0.30','tiny percentile position','strict-FPR primary mechanism','reflection-boundary FP',
        'internal-seam FP','core-location FP','group007 mechanism','B0/P1 complementarity']
    for i,title in enumerate(titles,1):
        key=f'Q{i}';lines.append(f"{key} {title}: {report['questions'][key]['status']}")
        lines.append(json.dumps(report['questions'][key],indent=2))
    lines += ['',report['protocol'],'STATUS:',report['status']]
    if report.get('error_message'):lines.append(report['error_message'])
    return '\n'.join(lines+['='*40])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--precheck-only',action='store_true');args=parser.parse_args()
    report=dict(status='RUNNING',error_message=None,no_model_loaded=True,no_forward_performed=True,
        native_shape=[1056,4224],connectivity=8,approximation=False,
        positive_percentiles_storage='positive_percentiles/MODEL/sample_id.npy: structured float64 lower/upper/mid percentiles and int32 region_id, in row-major GT-positive pixel order; GT is unchanged.',
        protocol='Global native background empirical percentile. PRO@FPR is pointwise shared exact tie-aware curve; not AU-PRO. Main sample PRO is equal-region mean using GLOBAL background. local_PRO fields additionally use each image background. Neither redefines frozen formal metrics.',
        tail_rule='Fixed q95 linear quantile of sorted full background, strict >; ties excluded equally for both models, actual density reported. Descriptive cutoff only, no deployment threshold chosen.',
        hypothesis_rules='H8:PRO005<=.01,PRO030>=.10,p95>=90. H9:PRO030<=.01,p95<90. Strict majority SUPPORTED; nonzero minority PARTIALLY_SUPPORTED; none NOT_SUPPORTED. Fixed before data analysis.',
        tests='9 tests observed RED for absent implementation, then GREEN; metal_ad_ead uses existing Conda base pytest via temporary sys.path append; no packages installed.')
    start=time.perf_counter()
    try:
        for name in ('','figures','galleries','temp'):(OUT/name).mkdir(parents=True,exist_ok=True)
        mapping,membership,tiny,hashes,checks=precheck();report['precheck']=checks;report['input_hashes']=hashes
        print('PRECHECK: '+json.dumps(checks),flush=True)
        if args.precheck_only:report['status']='PRECHECK_PASS';return 0
        import torch
        import cv2
        cache=caches(mapping,report,torch,cv2)
        samples,regions,normals=process(mapping,membership,tiny,cache,report,torch)
        complement=evidence(samples,regions,normals,tiny,mapping,cache,report)
        # Preserve all numeric diagnostics before rendering galleries.
        (OUT/'diagnostic_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        plots(samples,regions,normals,complement,mapping,cache,tiny,report,torch)
        if any(digest(path)!=value for path,value in hashes.items()):raise ValueError('Frozen maps/GT changed during audit')
        report['frozen_inputs_unchanged']=True
        report['status']='PASS';return 0
    except (Exception,KeyboardInterrupt) as exc:report.update(status='FAIL',error_message=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());traceback.print_exc();return 1
    finally:
        report['elapsed_seconds']=time.perf_counter()-start
        name='precheck_report' if args.precheck_only else 'diagnostic_report'
        (OUT/(name+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        text=format_report(report)
        (OUT/(name+'.txt')).write_text(text,encoding='utf-8');print(text)

if __name__=='__main__':raise SystemExit(main())
