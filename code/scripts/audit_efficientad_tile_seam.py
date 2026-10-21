"""Phase09C: forward-only border audit and GT-independent overlap geometry."""
import csv
import inspect
import json
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'experiments/phase09c_tile_seam_audit'
W, H, T = 4224, 1056, 1056


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (OUT/name).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k,v in row.items()})


def borders(array, np, threshold=0):
    zero = np.abs(array) == 0 if threshold == 0 else np.abs(array) < threshold
    rows, cols = zero.all(axis=1), zero.all(axis=0)
    def width(flags):
        return next((i for i,v in enumerate(flags) if not v), len(flags))
    return dict(top=width(rows), bottom=width(rows[::-1]), left=width(cols), right=width(cols[::-1]))


def positions(overlap):
    starts = [0]
    while starts[-1] < W-T:
        starts.append(min(starts[-1]+T-overlap, W-T))
    return starts


def simulate(starts, border, np):
    l,r,t,b = [border[k] for k in ('left','right','top','bottom')]
    count = np.zeros((H,W), dtype=np.uint8)
    raw_x = np.zeros(W, dtype=np.uint8)
    mean_den = np.zeros((H,W), dtype=np.float32)
    center_den = np.zeros((H,W), dtype=np.float32)
    # Positive interior triangular weights: fusion cannot create missing support.
    wx = 1-np.abs((np.arange(T-l-r)+0.5)/(T-l-r)*2-1)
    wy = 1-np.abs((np.arange(H-t-b)+0.5)/(H-t-b)*2-1)
    weights = np.outer(wy,wx).astype(np.float32)
    for start in starts:
        raw_x[start:start+T] += 1
        count[t:H-b,start+l:start+T-r] += 1
        mean_den[t:H-b,start+l:start+T-r] += 1
        center_den[t:H-b,start+l:start+T-r] += weights
    assert raw_x.min() >= 1
    assert np.array_equal(mean_den > 0, center_den > 0)
    return count, raw_x


def main():
    r = dict(status='FAIL', timestamp=datetime.now().astimezone().isoformat(), error_message=None,
             no_performance_tuning=True, training_steps=0)
    try:
        os.chdir(ROOT)
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        (OUT/'figures').mkdir(parents=True, exist_ok=True)
        (OUT/'raw_tile_maps').mkdir(exist_ok=True)
        import numpy as np
        import torch
        import cv2
        import anomalib
        from PIL import Image
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from lightning.pytorch import seed_everything
        from anomalib.models import EfficientAd
        from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel
        from anomalib.data.utils import read_image
        from src.data.tile4_builder import DERIVED, manifest_hash
        from src.evaluation.exact_aupro import components
        r['anomalib_version']=anomalib.__version__
        if anomalib.__version__!='2.6.1':raise RuntimeError('Requires inspected Anomalib 2.6.1')
        source = {}
        for obj in (EfficientAd.__init__,EfficientAdModel.__init__,EfficientAdModel.forward,
                    EfficientAdModel.compute_maps,EfficientAdModel.get_maps):
            source[obj.__qualname__]=dict(module=obj.__module__, signature=str(inspect.signature(obj)),
                file=inspect.getsourcefile(obj), line=inspect.getsourcelines(obj)[1], source=inspect.getsource(obj))
        r['source_inspection']=source
        r['source_summary']='compute_maps: channel mean of squared feature distances; F.pad(...,(4,4,4,4)) at feature-map scale; bilinear to input size; optional affine quantile normalization afterwards. Padding width is literal 4, not computed from native tile dimensions.'
        seed_everything(42, workers=True)
        model=EfficientAd(model_size='small',imagenet_dir='./datasets/imagenette',
                          evaluator=False,visualizer=False,post_processor=False)
        model.prepare_pretrained_model()
        model.eval().to('cuda:0')
        r.update(padding=model.model.ae.decoder.padding,pad_maps=model.model.pad_maps,
                 model_input=[256,256],preprocessor=repr(model.pre_processor.transform),
                 weights_note='Phase09B saved no checkpoint. Fixed seed42 fresh student/AE and official pretrained teacher; no optimization or quantile fitting. Audit verifies structural support, not prediction quality. Phase09B saved maps are separately cross-checked.',
                 normalization_note='Quantiles are unset as in Phase09B smoke. With fitted quantiles artificial zero borders may become constant nonzero; nonzero is not proof of valid prediction support.')
        assert r['padding'] is False and r['pad_maps'] is True
        report=json.loads((DERIVED/'dataset_report.json').read_text())
        assert report['status']=='PASS' and report['manifest_sha256']==manifest_hash()
        manifest=read_csv(DERIVED/'manifest.csv')
        chosen=[]
        for label in (0,1):
            candidates=sorted([a for a in manifest if a['split']=='test_public' and int(a['tile_is_anomalous'])==label],key=lambda a:a['tile_image_path'])
            chosen.extend(candidates[:5])
        assert len(chosen)==10
        rows=[]
        with torch.inference_mode():
            for i,a in enumerate(chosen):
                x=model.pre_processor(read_image(a['tile_image_path'],as_tensor=True)[None].to('cuda:0'))
                assert tuple(x.shape[-2:])==(256,256)
                raw=model.model(x).anomaly_map[0,0].float().cpu()
                assert torch.isfinite(raw).all()
                native=torch.nn.functional.interpolate(raw[None,None],size=(H,T),mode='bilinear',align_corners=False)[0,0].numpy()
                arr=raw.numpy(); np.save(OUT/'raw_tile_maps'/f'tile_{i:02d}.npy',arr,allow_pickle=False)
                row=dict(tile_image_path=a['tile_image_path'],tile_is_anomalous=int(a['tile_is_anomalous']),raw_shape=list(arr.shape))
                for scale,data in (('model',arr),('native',native)):
                    for kind,threshold in (('exact',0),('near',1e-12)):
                        for side,n in borders(data,np,threshold).items():row[f'{scale}_{kind}_{side}']=n
                rows.append(row)
        write_csv('effective_region_analysis.csv',rows)
        mb={s:rows[0]['model_exact_'+s] for s in ('left','right','top','bottom')}
        nb={s:rows[0]['native_exact_'+s] for s in mb}
        for row in rows:
            for s in mb:
                assert row['model_exact_'+s]==row['model_near_'+s]==mb[s]
                assert row['native_exact_'+s]==row['native_near_'+s]==nb[s]
        l,rr,t,b=[mb[s] for s in ('left','right','top','bottom')]
        r.update(raw_anomaly_map_shape=rows[0]['raw_shape'],model_border_pixels=mb,
                 native_theoretical_border_pixels={s:n*T/256 for s,n in mb.items()},native_observed_border_pixels=nb,
                 borders_fixed_across_10_tiles=True,valid_rectangle_model=[l,256-rr,t,256-b],
                 valid_width_256=256-l-rr,valid_height_256=256-t-b,
                 native_equivalent_valid_width=(256-l-rr)*T/256,native_equivalent_valid_height=(256-t-b)*T/256,
                 valid_area_fraction=(256-l-rr)*(256-t-b)/256**2,
                 validity_definition='Non-artificial-zero support, not a guarantee of accuracy or fully uncontaminated receptive fields. Bilinear transition pixels mix padding with actual scores.',
                 native_rounding_note='k*1056/256 is an interval-length estimate. Two half-pixel bilinear operations shrink the exactly-zero run; observed native support is used for pixel coverage.')
        # Pure support probe, no network changes: reveal padding-mixed transition.
        support=torch.nn.functional.interpolate(torch.nn.functional.pad(torch.ones(1,1,56,56),(4,4,4,4)),size=(256,256),mode='bilinear',align_corners=False)[0,0].numpy()
        assert all(borders(support,np)[s]==mb[s] for s in mb)
        r['padding_unmixed_center_border_model']=borders((support>=1).astype(np.float32),np)
        r['feature_geometry']={'unpad_map':[56,56],'padded_map':[64,64],'pad_each_side':4,
                               'unmixed_note':'Fully unmixed center is stricter than nonzero support; reported separately, not silently substituted.'}
        p0,rawx=simulate([0,1056,2112,3168],nb,np)
        r['p0_blind_width_observed']=nb['left']+nb['right']
        r['p0_blind_width_theoretical']=(mb['left']+mb['right'])*T/256
        r['p0_internal_boundaries']=[1056,2112,3168]
        smoke=json.loads((ROOT/'experiments/phase09b_p0_tile_smoke/pipeline_smoke_report.json').read_text())
        r['phase09b_crosscheck']={k:[s['contiguous_zero_band_width'] for s in v['seams']['seams']] for k,v in smoke['predictions'].items()}
        overlap_min=(mb['left']+mb['right'])*T/256
        r.update(overlap_min_native=overlap_min,overlap_min_model_equivalent=mb['left']+mb['right'],
                 overlap_min_native_observed_support=nb['left']+nb['right'])
        counts={'P0':p0};candidates=[]
        for overlap in (128,160,192,224,256):
            starts=positions(overlap);count,raw_x=simulate(starts,nb,np);counts[str(overlap)]=count
            mid=count[H//2];interior=mid[nb['left']:W-nb['right']]
            candidates.append(dict(overlap=overlap,sufficiency='SUFFICIENT' if overlap>=overlap_min else 'NOT_SUFFICIENT',
                sufficiency_scope='Internal horizontal seams only',stride=T-overlap,tile_starts=starts,tile_count=len(starts),
                forward_count=len(starts),forward_ratio_B0=len(starts),forward_ratio_P0=len(starts)/4,
                raw_coverage_count_per_x=raw_x.tolist(),valid_midrow_coverage_count_per_x=mid.tolist(),
                internal_x_covered=bool((interior>0).all()),full_width_valid_coverage=bool((mid>0).all()),
                full_image_valid_coverage=bool((count>0).all()),uncovered_pixels=int((count==0).sum()),
                fusion_coverage_equal=True,geometric_status='NOT_GEOMETRICALLY_VALID'))
        r['blending_design']={'simple_mean':'sum supported scores / support count',
            'center_weighted':'positive separable triangular weights inside valid rectangle; divide by accumulated weights',
            'coordinates':'Crop model rectangle and map to matching native rectangle, never stretch it over the entire1056 tile. Coverage simulation uses observed bilinear support bounds.',
            'zero_denominator':'Remain uncovered, never filled or declared valid. Both methods have identical support. No prediction fusion or GT metric comparison performed.'}
        # Same native GT connected components as Phase09A; no score thresholds or tuning.
        samples=read_csv(ROOT/'experiments/phase07c_baseline256_public/predictions.csv')
        assert len(samples)==114 and sum(int(a['gt_label']) for a in samples)==90
        stats={k:dict(regions_with_blind_pixels=0,blind_gt_pixels=0) for k in counts}
        examples=[];region_total=0;internal_affected=0
        for a in samples:
            if not int(a['gt_label']):continue
            with Image.open(a['gt_mask_path']) as im:mask=np.asarray(im)>0
            assert mask.shape==(H,W)
            labels,sizes=components(mask,cv2,np);region_total+=len(sizes)-1
            for name,count in counts.items():
                blind=mask & (count==0)
                ids=np.unique(labels[blind]);stats[name]['regions_with_blind_pixels']+=int((ids>0).sum())
                stats[name]['blind_gt_pixels']+=int(blind.sum())
            internal=np.zeros((H,W),dtype=bool)
            for boundary in (1056,2112,3168):internal[:,boundary-nb['right']:boundary+nb['left']]=True
            ids=np.unique(labels[mask & internal]);internal_affected+=int((ids>0).sum())
            if len(examples)<5:
                crosses=any(set(np.unique(labels[:,:x])) .intersection(set(np.unique(labels[:,x:]))) - {0} for x in (1056,2112,3168))
                if crosses:examples.append(a)
        assert region_total==450
        r.update(connected_regions=region_total,connected_components='Shared Phase09A components: OpenCV connectivity=8',
                 p0_regions_intersecting_internal_blind_seams=internal_affected,gt_coverage=stats)
        for a in candidates:a.update(stats[str(a['overlap'])])
        write_csv('candidate_overlap_analysis.csv',candidates)
        r.update(candidates=candidates,smallest_internal_seam_sufficient_candidate=next(a['overlap'] for a in candidates if a['internal_x_covered']),
                 smallest_geometrically_valid_candidate=None,full_image_valid_coverage=False,
                 recommendation='No candidate meets full-image valid coverage under current constraints. 128px/5 tiles covers internal x seams only. Outer left/right borders and top/bottom strips remain blind for every horizontal overlap. A separate explicit outer-boundary protocol is required before claiming full coverage; no P1 started.')
        fig,ax=plt.subplots(figsize=(6,6));ax.imshow(support,vmin=0,vmax=1,cmap='viridis')
        ax.add_patch(Rectangle((l-.5,t-.5),256-l-rr,256-t-b,fill=False,edgecolor='red'))
        ax.set_title('256 map: zero border / bilinear transition / center');fig.savefig(OUT/'figures/efficientad_valid_region_256.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(12,3));ax.step(np.arange(W),p0[H//2]);ax.set_ylim(-.1,2)
        for x in (1056,2112,3168):ax.axvline(x,color='red',ls='--')
        ax.set(title='P0 valid coverage at middle row (outer borders also blind)',xlabel='Native x',ylabel='Valid tile count')
        fig.tight_layout();fig.savefig(OUT/'figures/p0_seam_coverage.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(5,1,figsize=(12,9),sharex=True)
        for ax,a in zip(axes,candidates):
            ax.step(np.arange(W),counts[str(a['overlap'])][H//2]);ax.set_ylabel(str(a['overlap'])+' px');ax.set_ylim(-.1,3)
        axes[0].set_title('Horizontal valid coverage: every candidate leaves outer borders blind')
        axes[-1].set_xlabel('Native x');fig.tight_layout();fig.savefig(OUT/'figures/overlap_candidate_coverage.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(len(examples),1,figsize=(15,3*len(examples)),squeeze=False)
        for ax,a in zip(axes[:,0],examples):
            with Image.open(a['image_path']) as im:ax.imshow(im)
            with Image.open(a['gt_mask_path']) as im:mask=np.asarray(im)>0
            ax.contour(mask,levels=[.5],colors='lime',linewidths=.6)
            for x in (1056,2112,3168):ax.axvspan(x-nb['right'],x+nb['left'],color='red',alpha=.3)
            ax.set_title(a['sample_id']+'; green GT / red P0 blind seam');ax.axis('off')
        fig.tight_layout();fig.savefig(OUT/'figures/gt_seam_examples.png',dpi=130);plt.close(fig)
        r['status']='PASS'
    except (Exception,KeyboardInterrupt) as exc:
        r.update(error_message=str(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'seam_audit.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
        summary={k:v for k,v in r.items() if k not in ('source_inspection','candidates')}
        summary['overlap_candidate_summary']=[{k:a[k] for k in ('overlap','sufficiency','sufficiency_scope','tile_starts','tile_count','forward_ratio_B0','forward_ratio_P0','full_image_valid_coverage','geometric_status')} for a in r.get('candidates',[])]
        text='='*40+'\nPHASE 09C - TILE SEAM / VALID REGION AUDIT\n'+'='*40+'\n'+json.dumps(summary,indent=2)+'\nSTATUS: '+r['status']
        (OUT/'seam_audit.txt').write_text(text,encoding='utf-8');print(text)
    return 0 if r['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
