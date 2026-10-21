"""P1 paired analysis using frozen membership and shared exact rank functions."""
import csv
import json
import math
from pathlib import Path
from src.evaluation import exact_aupro as exact
from scripts.analyze_b0_failures import size_group
from scripts.analyze_resolution_tradeoff import midrank, image_ranks, FPRS, EPS, input_check, digest

ROOT=Path(__file__).resolve().parents[2]
D0=ROOT/'experiments/phase07d_b0_failure_analysis'
B1=ROOT/'experiments/phase08c_baseline384_public'
D1=ROOT/'experiments/phase08d_resolution_tradeoff'
A9=ROOT/'experiments/phase09a_tile_geometry'
GROUP_N=dict(A=12,B=18,C=42,D=6,E=12)
B0_GROUP=dict(A=0.,B=7.23,C=10.11,D=45.57,E=29.49)
B1_GROUP=dict(A=5.69146638721465,B=10.788456130361247,C=9.160280921250857,D=23.183775633490434,E=27.890073906016056)


def read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def csv_save(path, rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def references(np,Image):
    inputs,hashes=input_check(np,Image)
    old=json.loads((D1/'analysis_report.json').read_text(encoding="utf-8"))
    if old['status']!='PASS' or any(old['input_hashes'].get(k)!=v for k,v in hashes.items()):
        raise ValueError('B0/B1 artifacts differ from frozen Phase08D')
    for path in (ROOT/'experiments/phase07c_r1_aupro_exact/validation_report.json',
                 ROOT/'experiments/phase07c_r1_aupro_exact/aupro_report.json', B1/'evaluation_report.json'):
        if json.loads(path.read_text(encoding="utf-8"))['status']!='PASS':raise ValueError('Frozen exact/reference gate failed')
    samples={a['sample_id']:a for a in read_csv(D0/'sample_analysis.csv')}
    b1={a['sample_id']:a for a in read_csv(B1/'metrics/per_image_aupro.csv')}
    ids={sid for sid,a in inputs['B0'].items() if int(a['gt_label'])}
    if set(samples)!=set(b1) or set(samples)!=ids or len(ids)!=90:raise ValueError('Anomaly membership mismatch')
    for g,n in GROUP_N.items():
        if sum(a['size_group']==g for a in samples.values())!=n:raise ValueError('Frozen group counts differ')
    for sid,a in samples.items():
        if b1[sid]['group']!=a['size_group']:raise ValueError('B0/B1 size group mismatch')
    geometry={(a['sample_id'],int(a['region_id'])):a for a in read_csv(A9/'region_tile_analysis.csv')}
    regions={(a['sample_id'],int(a['region_id'])):a for a in read_csv(D1/'paired_region_delta.csv')}
    if set(geometry)!=set(regions) or len(regions)!=450:raise ValueError('Frozen450 region IDs mismatch')
    frozen=json.loads((D0/'failure_analysis.json').read_text(encoding="utf-8"))
    galleries={k:frozen['gallery_selection'][k] for k in ('worst_10','highest_normal_response_10')}
    galleries['tiny_12']=sorted(sid for sid,a in samples.items() if a['size_group']=='A')
    galleries['group_D_6']=sorted(sid for sid,a in samples.items() if a['size_group']=='D')
    # Deterministic representatives defined entirely by prior geometry, before P1 scores exist.
    boundary=sorted({sid for (sid,rid),a in geometry.items() if a['crosses_boundary']=='True'})
    galleries['boundary_representatives']=boundary[:10]
    galleries['outer_boundary_representatives']=sorted({sid for (sid,rid),a in geometry.items()
        if min(int(a['bbox_xmin']),4223-int(a['bbox_xmax']),int(a['bbox_ymin']),1055-int(a['bbox_ymax']))<=96})[:10]
    for key,ids0 in galleries.items():
        if len(ids0)!=len(set(ids0)) or not set(ids0)<=set(inputs['B0']):raise ValueError('Invalid fixed gallery')
    curves=read_csv(D1/'strict_fpr_analysis.csv')
    if [float(a['FPR']) for a in curves]!=FPRS:raise ValueError('Frozen PRO curve grid mismatch')
    for path in (D0/'sample_analysis.csv',D0/'failure_analysis.json',B1/'metrics/per_image_aupro.csv',
                 A9/'region_tile_analysis.csv',D1/'paired_region_delta.csv',D1/'strict_fpr_analysis.csv',D1/'analysis_report.json'):
        hashes[str(path)]=digest(path)
    return dict(inputs=inputs,hashes=hashes,samples=samples,b1=b1,geometry=geometry,regions=regions,galleries=galleries,curves=curves)


def paired_stats(delta,np):
    values=np.asarray(delta,dtype=np.float64)
    improved=int((values>EPS).sum());worsened=int((values<-EPS).sum())
    mean=float(values.mean());median=float(np.median(values))
    pattern='NO_IMPROVEMENT' if improved==0 or mean<=EPS else 'BROAD_BASED' if improved>len(values)/2 and median>EPS else 'FEW_SAMPLE_DRIVEN'
    return dict(n=len(values),improved_count=improved,worsened_count=worsened,unchanged_count=len(values)-improved-worsened,
        mean_delta_pp=mean*100,median_delta_pp=median*100,pattern=pattern,
        rule='EPS=1e-10 in unit score; BROAD_BASED requires positive mean, strict majority improved and positive median; no positive mean or no improvements -> NO_IMPROVEMENT; otherwise FEW_SAMPLE_DRIVEN.')


def analyze(rows,ref,out,report,np,torch,cv2,Image):
    paired=[];regions=[]
    for row in rows:
        if not int(row['gt_label']):continue
        sid=row['sample_id'];frozen=ref['samples'][sid]
        mask=exact.get_mask(row,Image,np);labels,sizes=exact.components(mask,cv2,np)
        fg=labels>0;count=int(fg.sum());k=len(sizes)-1;group=size_group(count/mask.size)
        if count!=int(frozen['defect_pixel_count']) or group!=frozen['size_group']:raise ValueError('P1 group membership changed')
        score=exact.get_score(row,torch,np);bg=score[~fg];bg.sort(kind='quicksort')
        values=score[fg];positive_labels=labels[fg]
        weights=1/(k*sizes[positive_labels].astype(np.float64))
        pro,total=exact.integrate(bg,values,weights,np)
        if abs(total-1)>1e-6 or not all(math.isfinite(v) and 0<=v<=1 for v in pro.values()):raise ValueError('Invalid per-image exact PRO')
        old=float(frozen['aupro_005']);b1=float(ref['b1'][sid]['aupro_005'])
        paired.append(dict(sample_id=sid,group=group,defect_ratio=count/mask.size,B0_AUPRO005=old,B1_AUPRO005=b1,
            P1_AUPRO005=pro[.05],P1_minus_B0=pro[.05]-old,P1_minus_B1=pro[.05]-b1))
        ranks=midrank(bg,values,np);yy,xx=np.nonzero(fg)
        for rid in range(1,k+1):
            key=(sid,rid);geom=ref['geometry'][key];baseline=ref['regions'][key];select=positive_labels==rid
            x=xx[select];y=yy[select];v=ranks[select]
            if int(geom['pixel_count'])!=int(sizes[rid]) or int(baseline['region_size'])!=int(sizes[rid]):raise ValueError('Region definition changed')
            affected=any(bool(((x>=b-56)&(x<b+56)).any()) for b in (1056,2112,3168))
            distance=int(min(x.min(),4223-x.max(),y.min(),1055-y.max()))
            b0=float(baseline['B0_rank_mean']);p1=float(v.mean())
            regions.append(dict(sample_id=sid,region_id=rid,pixel_count=int(sizes[rid]),
                boundary_affected=affected,phase09a_crossing=geom['crosses_boundary']=='True',
                outer_border_distance=distance,outer_boundary_group=distance<=96,
                B0_background_percentile_mean=b0,P1_background_percentile_mean=p1,
                P1_minus_B0_rank_pp=(p1-b0)*100,P1_background_percentile_max=float(v.max()),
                P1_background_percentile_p95=float(np.percentile(v,95))))
    if len(paired)!=90 or len(regions)!=450:raise ValueError('Paired sample/region counts invalid')
    csv_save(out/'paired_analysis/all_anomaly_paired.csv',paired)
    csv_save(out/'paired_analysis/region_paired.csv',regions)
    groups=[]
    for g,n in GROUP_N.items():
        part=[a for a in paired if a['group']==g];values=np.array([a['P1_AUPRO005'] for a in part])
        if len(values)!=n:raise ValueError('Group count mismatch')
        groups.append(dict(group=g,n=n,B0_mean_percent=B0_GROUP[g],B1_mean_percent=B1_GROUP[g],
            P1_mean_percent=float(values.mean()*100),P1_median_percent=float(np.median(values)*100),
            P1_std_percent=float(values.std()*100),P1_gain_vs_B0_pp=float(values.mean()*100)-B0_GROUP[g]))
        if g in ('A','B'):
            csv_save(out/f'paired_analysis/group_{g}_paired.csv',part)
            report.setdefault('group_paired',{})[g]=paired_stats([a['P1_minus_B0'] for a in part],np)
    csv_save(out/'metrics/size_group_analysis.csv',groups);report['size_groups']=groups
    report['all_anomaly_paired']=paired_stats([a['P1_minus_B0'] for a in paired],np)
    try:
        from scipy.stats import wilcoxon
        d=np.array([a['P1_minus_B0'] for a in paired])
        w=wilcoxon(d,zero_method='wilcox',alternative='two-sided') if np.any(d!=0) else None
        report['wilcoxon']=dict(statistic=float(w.statistic) if w else 0.,pvalue=float(w.pvalue) if w else 1.,
            note='Auxiliary paired two-sided test; no multiplicity correction; effect sizes reported separately.')
    except ImportError:report['wilcoxon']='NOT_AVAILABLE'
    subgroups=[]
    for field,names in (('boundary_affected',('internal_blind_band_affected','non_boundary')),
                        ('outer_boundary_group',('outer_boundary_le96','interior_gt96'))):
        for flag,name in zip((True,False),names):
            vals=np.array([a['P1_minus_B0_rank_pp'] for a in regions if a[field]==flag])
            subgroups.append(dict(group=name,n=len(vals),mean_rank_gain_pp=float(vals.mean()) if len(vals) else None,
                median_rank_gain_pp=float(np.median(vals)) if len(vals) else None))
    csv_save(out/'paired_analysis/boundary_subgroups.csv',subgroups);report['boundary_subgroups']=subgroups
    report['region_response_definition']='Per-image background empirical midrank, shared Phase08D midrank; equal weight per GT region. Rank gain is not AU-PRO gain. Internal subgroup intersects Phase09C blind bands[b-56,b+56); outer group min pixel distance<=96. No P0 trained performance claim.'
    # Reuse identical Phase08D ranking logic via generic second-model slot.
    ranked,rank_summary=image_ranks({'B0':ref['inputs']['B0'],'B1':{a['sample_id']:a for a in rows}},np)
    renamed={k.replace('B1','P1'):v for k,v in rank_summary.items()}
    if abs(renamed['P1']['image_auroc_rank_check']-report['metrics']['image_auroc'])>1e-12:
        raise ValueError('Image AUROC and pairwise ranking disagree')
    report['image_ranking']=renamed
    normal_rows=[]
    for sid,a in ranked.items():
        if not int(ref['inputs']['B0'][sid]['gt_label']):
            item={k.replace('B1','P1'):v for k,v in a.items()}
            item.update(sample_id=sid,normal_rank_shift=item['P1_image_percentile_rank']-item['B0_image_percentile_rank'])
            normal_rows.append(item)
    csv_save(out/'paired_analysis/normal_ranking.csv',normal_rows)
    report['ranking_interpretation_note']='Normal/anomaly ranks share one ranking and are coupled, not independent causal mechanisms. Frozen B0 score rule retained.'
    return paired


def figures_and_galleries(rows,ref,out,report,np,torch,Image):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curve=read_csv(out/'metrics/pro_fpr_b0_b1_p1.csv')
    fig,ax=plt.subplots(figsize=(9,5))
    for tag in ('B0','B1','P1'):ax.plot(FPRS,[float(a[tag+'_PRO']) for a in curve],marker='o',label=tag)
    ax.axvline(.05,color='black',ls='--');ax.set(xlabel='FPR',ylabel='PRO',ylim=(0,1));ax.legend()
    fig.tight_layout();fig.savefig(out/'visualizations/pro_fpr_b0_b1_p1.png',dpi=160);plt.close(fig)
    current={a['sample_id']:a for a in rows};report['visualizations_saved']=0
    for group,ids in ref['galleries'].items():
        folder=out/'visualizations'/group;folder.mkdir(exist_ok=True)
        for sid in ids:
            row=current[sid]
            with Image.open(row['image_path']) as im:rgb=np.asarray(im.convert('RGB').resize((1200,300),Image.Resampling.BILINEAR))
            mask=exact.get_mask(row,Image,np)
            gt=np.asarray(Image.fromarray(mask*255).resize((1200,300),Image.Resampling.NEAREST))
            fig,axes=plt.subplots(5,1,figsize=(14,17),constrained_layout=True)
            axes[0].imshow(rgb);axes[1].imshow(gt,cmap='gray',vmin=0,vmax=255)
            for ax,tag in zip(axes[2:],('B0','B1','P1')):
                source=row if tag=='P1' else ref['inputs'][tag][sid]
                native=exact.get_score(source,torch,np)
                display=np.asarray(Image.fromarray(native).resize((1200,300),Image.Resampling.BILINEAR))
                ax.imshow(display,cmap='jet',vmin=float(display.min()),vmax=max(float(display.max()),float(display.min())+1e-12))
            for ax,title in zip(axes,('Original','GT','B0','B1','P1')):ax.set_title(title);ax.axis('off')
            fig.suptitle(sid+' | independent display-only min/max scales; raw scores are not cross-model calibrated')
            fig.savefig(folder/(sid+'.png'),dpi=120);plt.close(fig);report['visualizations_saved']+=1
