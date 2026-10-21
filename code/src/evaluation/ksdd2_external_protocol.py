"""External KSDD2 split, geometry and methods fixed before any model performance."""
import hashlib
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/phase15a_ksdd2_audit'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def mask_name(name):return Path(name).stem+'_GT.png'

def make_split(rows):
    normal=sorted(v['id'] for v in rows if v['official_split']=='train' and v['label']==0)
    order=np.random.default_rng(42).permutation(len(normal));val={normal[i] for i in order[:math.ceil(.1*len(normal))]}
    return [dict(v,split='test' if v['official_split']=='test' else 'excluded_train_defect' if v['label'] else 'validation_normal' if v['id'] in val else 'train_normal') for v in sorted(rows,key=lambda v:v['id'])]

def split_hash(rows):
    return hashlib.sha256(json.dumps([{k:v[k] for k in ['id','split','image_sha']} for v in sorted(rows,key=lambda v:v['id'])],sort_keys=True).encode()).hexdigest()

def duplicate_audit(rows):
    groups=defaultdict(list)
    for v in rows:groups[v['image_sha']].append(v)
    dupe=[v for v in groups.values() if len(v)>1]
    return dict(duplicate_image_groups=[[v['id'] for v in g] for g in dupe],split_leakage=any(len({v.get('split',v['official_split']) for v in g})>1 for g in dupe))

def inventory_gate(expected,actual):
    if not set(expected)<=set(actual):raise ValueError('FAIL_INCOMPLETE_ARCHIVE')
    return True

def canonical_copy_name(name):return name.replace(' (copy)','')

def select_rect(aspect_hw):
    # Joint dimensionless squared log error: equal weight to pixel budget and H/W ratio.
    candidates=[(math.log(h*w/262144)**2+math.log((h/w)/aspect_hw)**2,abs(h*w-262144),h,w) for h in range(96,2049,32) for w in range(96,2049,32) if h*w<=300000]
    _,_,h,w=min(candidates);return h,w

def feature_shapes(hw):
    final=lambda n:((n-3)//2-3)//2-5
    return dict(S1=[128,hw[0]-3,hw[1]-3],Final=[384,final(hw[0]),final(hw[1])])

def size_bins(ratios):
    return dict(boundaries=np.quantile(ratios,[.2,.4,.6,.8]).tolist(),quantiles=[.2,.4,.6,.8],source='GT area ratios before model evaluation',tiny='A: ratio<=20th percentile',ties='upper boundary included in lower group',scope='all official defect masks; structural metadata only, no model scores',group_names=list('ABCDE'))

def consensus(s,f,ae):return .5*(s.relu()*f.relu()).sqrt()+.5*ae

def protocol(rect):
    base=dict(seed=42,batch=1,precision='FP32',fresh_student=True,fresh_ae=True,teacher='official pretrained',optimizer='Adam',lr=.0001,weight_decay=.00001,scheduler='StepLR at66500, gamma0.1',formal_steps=70000,max_epochs=1000,early_stopping=False,official_loss_unchanged=True,imagenette_penalty_count=1)
    return dict(protocol_origin='MVTec AD 2 post-hoc mechanism study',external_dataset_used_for_method_selection=False,external_test_tuning_allowed=False,methods=dict(STD=dict(base,input_hw=[256,256],image_head='Final-only Combined native max',pixel_head='official Combined'),RECT=dict(base,input_hw=list(rect),image_head='Final-only Combined native max',pixel_head='official Combined'),DGNB=dict(base,input_hw=list(rect),image_head='Final-only Combined native max',pixel_head='Consensus Combined',rho=1.,beta=.9,bounds=[.005,5.],s1='Conv1 ReLU',s1_hard_quantile=.999,controller='existing detached shared Conv1 log-space EMA; no long-term tracking assumption')),square384='geometry comparison only, not a fourth trained candidate',rect_algorithm='search H,W in32 multiples96..2048, pixels<=300000; minimize log(H*W/262144)^2+log((H/W)/median_native_H_over_W)^2; tie by pixel gap,H,W',split='official test untouched; official train defects excluded; ceil10% train normals by numpy default_rng42 permutation of sorted IDs for validation',validation_use=['official Final/AE quantiles','S1 quantiles','normal-only calibration'],quantile_probabilities=[.9,.995],s1_quantiles_saved=True,metrics=['Image AUROC','Pixel AUROC','AU-PRO@0.05','AU-PRO@0.30'],native_exact_aupro=True,size_metrics='point PRO@0.05 by GT-only frozen quintiles',runtime=dict(warmup=20,rounds=5,sources='all official test',cuda_synchronize=True,scopes=['model-only','end-to-end']),calibration_steps=200,smoke_steps=3,no_70k_launch=True,no_test_model_forward=True,freeze_before_training=True)

def find_dataset():
    for name in ['KSDD2','ksdd2','KolektorSDD2']:
        root=ROOT/'datasets'/name
        if (root/'train').is_dir() and (root/'test').is_dir():return root
        if root.exists():
            for sub in root.iterdir():
                if sub.is_dir() and (sub/'train').is_dir() and (sub/'test').is_dir():return sub
    return None

def audit(root):
    from PIL import Image
    from scipy.ndimage import label
    rows=[];masks=[];errors=[];maskpaths=set();sizes=Counter();regions=0;copies=[];inventory={}
    archive=root/'KolektorSDD2.zip'
    if archive.exists():
        import zipfile
        with zipfile.ZipFile(archive) as z:expected={v.filename for v in z.infolist() if not v.is_dir()}
        actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()};inventory_gate(expected,actual)
        inventory=dict(official_zip_sha=digest(archive),archive_entries=len(expected),all_entries_present=True)
    for split in ['train','test']:
        folder=root/split
        for p in sorted(folder.glob('*.png')):
            if canonical_copy_name(p.name)!=p.name:
                original=p.with_name(canonical_copy_name(p.name))
                if not original.exists() or digest(p)!=digest(original):errors.append(dict(path=str(p),error='unverified copy'));continue
                copies.append(dict(path=str(p),canonical=str(original),sha=digest(p),excluded=True,reason='byte-identical archive copy'));continue
            if p.stem.endswith('_GT'):continue
            mask=folder/mask_name(p.name)
            try:
                with Image.open(p) as im:im.load();arr=np.asarray(im);w,h=im.size
                if not mask.exists():raise ValueError('missing mask')
                with Image.open(mask) as im:im.load();m=np.asarray(im.convert('L'))
                if m.shape!=(h,w):raise ValueError('image-mask mismatch')
                binary=m>0;_,n=label(binary,structure=np.ones((3,3)));pos=int(binary.sum());regions+=n;sizes[f'{h}x{w}']+=1;maskpaths.add(str(mask))
                row=dict(id=split+'/'+p.stem,official_split=split,image_path=str(p.resolve()),mask_path=str(mask.resolve()),image_sha=digest(p),decoded_image_sha=hashlib.sha256(arr.tobytes()).hexdigest(),label=int(pos>0),height=h,width=w,aspect_hw=h/w,defect_pixels=pos,defect_ratio=pos/(h*w),regions=n);rows.append(row)
                masks.append(dict(id=row['id'],path=str(mask.resolve()),sha=digest(mask),height=h,width=w,foreground=pos,regions=n,unique_values=';'.join(map(str,np.unique(m).tolist()))))
            except Exception as ex:errors.append(dict(path=str(p),error=str(ex)))
    orphan=[str(p) for p in root.rglob('*_GT.png') if str(p) not in maskpaths]
    split=make_split(rows);dupes=duplicate_audit(split);decoded=duplicate_audit([dict(v,image_sha=v['decoded_image_sha']) for v in split]);ratio=[r['defect_ratio'] for r in rows if r['label']];rect=select_rect(float(np.median([r['aspect_hw'] for r in rows])))
    official_counts=Counter((v['official_split'],v['label']) for v in rows)
    if official_counts!=Counter({('train',0):2085,('train',1):246,('test',0):894,('test',1):110}):errors.append(dict(error='official cardinality mismatch',actual=str(official_counts)))
    geometries={}
    for name,hw in [('square256',(256,256)),('square384',(384,384)),('rect',rect)]:
        per=[dict(id=v['id'],scale_x=hw[1]/v['width'],scale_y=hw[0]/v['height'],anisotropy=max(hw[1]/v['width'],hw[0]/v['height'])/min(hw[1]/v['width'],hw[0]/v['height'])) for v in rows]
        geometries[name]=dict(input_hw=list(hw),input_pixels=hw[0]*hw[1],estimated_features=feature_shapes(hw),median_anisotropy=float(np.median([v['anisotropy'] for v in per])),per_image=per)
    result=dict(status='PASS' if not errors and not orphan and not dupes['split_leakage'] and not decoded['split_leakage'] else 'FAIL_DATA_INTEGRITY',images=len(rows),normal=sum(not v['label'] for v in rows),defect=sum(v['label'] for v in rows),masks=len(masks),native_sizes=dict(sizes),aspect_hw=dict(min=min(v['aspect_hw'] for v in rows),median=float(np.median([v['aspect_hw'] for v in rows])),max=max(v['aspect_hw'] for v in rows)),connected_regions=regions,defect_ratio=dict(min=min(ratio),median=float(np.median(ratio)),max=max(ratio)),size_groups=size_bins(ratio),split_counts=dict(Counter(v['split'] for v in split)),split_sha=split_hash(split),errors=errors,orphan_masks=orphan,duplicates=dupes,decoded_duplicates=decoded,split_leakage=dupes['split_leakage'] or decoded['split_leakage'],geometries=geometries,rect_input=list(rect))
    for v in rows:v['size_group']='normal' if not v['label'] else 'ABCDE'[int(np.searchsorted(result['size_groups']['boundaries'],v['defect_ratio'],side='left'))]
    result['archive_inventory']=inventory;result['excluded_verified_copies']=copies
    result['size_group_counts']=dict(Counter(v['size_group'] for v in rows));result['tiny_defect_fraction']=result['size_group_counts'].get('A',0)/result['defect'];return result,rows,masks,split
