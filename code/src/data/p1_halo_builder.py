"""P1 derived contexts, reflected only at full-image outer borders."""
import csv
import hashlib
import json
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DERIVED=ROOT/'datasets/derived/sheet_metal_p1_halo96'
OUT=ROOT/'experiments/phase09d_p1_halo_audit'
HALO,CORE,CONTEXT=96,1056,1248


def read_rows(path=None):
    with (path or DERIVED/'manifest.csv').open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def manifest_hash():
    return hashlib.sha256((DERIVED/'manifest.csv').read_bytes()).hexdigest()


def save_report(name, report):
    OUT.mkdir(parents=True,exist_ok=True)
    text=json.dumps(report,indent=2)
    (OUT/(name+'.json')).write_text(text,encoding='utf-8')
    (OUT/(name+'.txt')).write_text(text,encoding='utf-8')


def reflect_contexts(original):
    import torch
    import numpy as np
    arr=original[:,:,None] if original.ndim==2 else original
    tensor=torch.from_numpy(np.ascontiguousarray(arr)).permute(2,0,1)[None]
    padded=torch.nn.functional.pad(tensor,(96,96,96,96),mode='reflect')[0].permute(1,2,0).numpy()
    if original.ndim==2:padded=padded[:,:,0]
    # Independent reflect101 index construction checks all corners and sides.
    yy=np.pad(np.arange(1056),(96,96),mode='reflect')
    xx=np.pad(np.arange(4224),(96,96),mode='reflect')
    assert np.array_equal(padded,original[yy[:,None],xx[None,:]])
    return [padded[:,i*1056:i*1056+1248].copy() for i in range(4)]


def build():
    if (DERIVED/'manifest.csv').exists():
        print('Existing P1 dataset: refusing overwrite; previous reports preserved.')
        return {'status':'FAIL','error_message':'Dataset already exists'}
    r=dict(status='FAIL',error_message=None,padding='torch.nn.functional.pad(mode=reflect); edge excluded, equivalent to reflect101',
           halo=96,core_size=1056,context_size=1248,source_reconstruction_pass=0,mask_reconstruction_pass=0,
           internal_halo_checks=0,central_core_checks=0,output_format='lossless PNG; no resize',
           future_sampling='548 contexts; four per source; uniform source probability 4/548=1/137. Planned 70000 steps, batch1, seed42; no performance fairness claim.')
    try:
        import numpy as np
        from PIL import Image
        from src.data.tile4_builder import DERIVED as P0,manifest_hash as p0_hash
        p0=json.loads((P0/'dataset_report.json').read_text())
        assert p0['status']=='PASS' and p0['manifest_sha256']==p0_hash()
        sources={a['source_image_id']:a for a in read_rows(P0/'manifest.csv')}
        assert len(sources)==270
        r['source_counts']={s:sum(a['split']==s for a in sources.values()) for s in ('train','validation','test_public')}
        assert r['source_counts']==dict(train=137,validation=19,test_public=114)
        rows=[]
        for n,(sid,a) in enumerate(sorted(sources.items())):
            path=Path(a['source_image_path']);split=a['split'];label=int(a['source_label'])
            with Image.open(path) as im:original=np.array(im)
            assert original.shape[:2]==(1056,4224)
            contexts=reflect_contexts(original)
            mask=np.zeros((1056,4224),dtype=np.uint8)
            if label:
                with Image.open(a['source_mask_path']) as im:mask=np.array(im)
            assert mask.shape==(1056,4224)
            restored=[];restored_masks=[]
            for i,context in enumerate(contexts):
                x0=i*1056;x1=x0+1056;coremask=mask[:,x0:x1]
                positive=int(np.count_nonzero(coremask));cl='anomaly' if positive else 'normal'
                fname=('bad' if label else 'good')+'__'+path.stem+f'__core{i}.png'
                dest=DERIVED/split/'images'/(cl if split=='test_public' else '')/fname
                dest.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(context).save(dest,compress_level=1)
                with Image.open(dest) as im:loaded=np.array(im)
                assert loaded.shape[:2]==(1248,1248) and np.array_equal(loaded,context)
                center=loaded[96:1152,96:1152]
                assert np.array_equal(center,original[:,x0:x1]);r['central_core_checks']+=1
                if i>0:
                    assert np.array_equal(loaded[96:1152,:96],original[:,x0-96:x0]);r['internal_halo_checks']+=1
                if i<3:
                    assert np.array_equal(loaded[96:1152,1152:],original[:,x1:x1+96]);r['internal_halo_checks']+=1
                restored.append(center)
                mp=None
                if split=='test_public':
                    mp=DERIVED/split/'masks'/fname;mp.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(coremask).save(mp,compress_level=1)
                    with Image.open(mp) as im:savedmask=np.array(im)
                    assert savedmask.shape==(1056,1056) and np.array_equal(savedmask,coremask)
                    restored_masks.append(savedmask)
                rows.append(dict(split=split,source_image_id=sid,source_image_path=str(path),source_label=label,
                    source_mask_path=a['source_mask_path'],tile_id=i,context_path=str(dest),core_mask_path=str(mp) if mp else '',
                    core_x0=x0,core_x1=x1,core_y0=0,core_y1=1056,halo=96,core_offset_in_context='[96,96]',
                    context_size=1248,core_size=1056,tile_is_anomalous=int(positive>0),core_positive_pixels=positive,
                    padded_crop_x0=x0,padded_crop_x1=x0+1248,padded_crop_y0=0,padded_crop_y1=1248))
            assert np.array_equal(np.concatenate(restored,axis=1),original);r['source_reconstruction_pass']+=1
            if label:
                assert np.array_equal(np.concatenate(restored_masks,axis=1),mask);r['mask_reconstruction_pass']+=1
            if (n+1)%20==0:print(f'P1 context/core verification: {n+1}/270',flush=True)
        r['tile_counts']={s:sum(a['split']==s for a in rows) for s in r['source_counts']}
        assert r['tile_counts']==dict(train=548,validation=76,test_public=456)
        assert r['source_reconstruction_pass']==270 and r['mask_reconstruction_pass']==90
        with (DERIVED/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        r.update(status='PASS',manifest_sha256=manifest_hash(),test_anomaly_contexts=sum(a['tile_is_anomalous'] for a in rows if a['split']=='test_public'),
                 outer_reflection_verified_sources=270,source_dataset_root=str(ROOT/'datasets/MVTec_AD_2'),
                 source_audit='Phase09B PASS plus all270 sources independently rechecked')
    except (Exception,KeyboardInterrupt) as exc:r.update(error_message=str(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        DERIVED.mkdir(parents=True,exist_ok=True)
        (DERIVED/'dataset_report.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
        save_report('dataset_report',r);print(json.dumps(r,indent=2))
    return r
