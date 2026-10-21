"""Phase24C-R1 Mid formal evaluation; one frozen 70K checkpoint, exact native metrics."""
import csv,hashlib,json,math,os,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/phase24c_r1_architecture_constrained_budget'
RUN=OUT/'08_mid_training'
CKPT=RUN/'checkpoints/final.ckpt'
DATA=ROOT/'datasets/MVTec_AD_2/sheet_metal'
NATIVE=(1056,4224)
accessed=[];blocked=[]
def guard(event,args):
    if event not in ('open','os.listdir','os.scandir') or not args or not isinstance(args[0],(str,bytes)):return
    p=Path(os.fsdecode(args[0])).absolute()
    if p.is_relative_to(ROOT/'datasets'):
        if not p.is_relative_to(DATA/'test_public'):
            blocked.append(str(p));raise PermissionError('Only public test data allowed in formal eval: '+str(p))
        if event=='open':accessed.append(str(p))
sys.addaudithook(guard)
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,obj):
    p=Path(p)
    if p.exists():raise FileExistsError(str(p))
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def sources():
    path=ROOT/'experiments/phase18b_sheet_metal_square512_70k_formal/evaluation/predictions.csv'
    with path.open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
    assert len(rows)==114 and len({r['sample_id'] for r in rows})==114
    assert sum(int(r['gt_label']) for r in rows)==90
    for r in rows:
        assert Path(r['image_path']).is_relative_to(DATA/'test_public')
        if int(r['gt_label']):assert Path(r['gt_mask_path']).is_relative_to(DATA/'test_public/ground_truth/bad')
    return rows
def main():
    evaluation_start=time.perf_counter()
    import cv2,numpy as np,torch
    import torch.nn.functional as F
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    from anomalib.models import EfficientAd
    from anomalib.data.utils import read_image
    from src.evaluation import exact_aupro as exact
    torch.set_num_threads(4)
    selection=read(OUT/'07_mid_selection.json')
    assert selection['selected_geometry']==[256,576] and selection['selection_used_test_metrics'] is False
    train=read(OUT/'08_mid_training_summary.json')
    assert train['status']=='PASS' and train['completed_steps']==70000
    assert train['test_public_files_accessed']==train['test_mask_files_accessed']==0
    assert sha(CKPT)==train['checkpoint_sha256']
    assert sha(ROOT/'src/evaluation/exact_aupro.py')=='3e6ebc80d400419cf3aaf00950e954ea165df2419c6ea895f4296013c67e11a1'
    os.environ['PHASE24C_R1_FORMAL_EVAL']='1'
    saved=torch.load(CKPT,map_location='cpu',weights_only=False)
    assert saved['global_step']==70000
    meta=saved['phase24c_r1']
    assert meta['final_validation_completed'] and meta['config']['geometry']==[256,576]
    del saved
    model=EfficientAd.load_from_checkpoint(str(CKPT),map_location='cpu',strict=True,weights_only=False).eval().cuda()
    assert list(model.pre_processor.transform.transforms[0].size)==[256,576]
    references=sources()
    result_dir=OUT/'09_mid_evaluation_artifacts'
    if result_dir.exists():
        assert not any((result_dir/'native_maps').iterdir()) and len(list(result_dir.iterdir()))==1
    else:result_dir.mkdir()
    maps=result_dir/'native_maps';maps.mkdir(exist_ok=True)
    rows=[];nan=inf=0
    with torch.inference_mode():
        for i,source in enumerate(references):
            native=read_image(source['image_path'],as_tensor=True)
            assert tuple(native.shape[-2:])==NATIVE
            x=model.pre_processor(native[None].cuda())
            raw=model.model(x).anomaly_map.detach().float()
            assert tuple(raw.shape)==(1,1,256,576)
            arr=F.interpolate(raw.cpu(),size=NATIVE,mode='bilinear',align_corners=False)[0,0].numpy().copy()
            score=float(arr.max())
            assert arr.shape==NATIVE and arr.dtype==np.float32
            nan+=int(np.isnan(arr).sum())+int(math.isnan(score))
            inf+=int(np.isinf(arr).sum())+int(math.isinf(score))
            path=maps/(source['sample_id']+'.npy')
            np.save(path,arr,allow_pickle=False)
            rows.append(dict(sample_id=source['sample_id'],image_path=source['image_path'],
                             gt_label=int(source['gt_label']),gt_mask_path=source['gt_mask_path'],
                             native_shape=NATIVE,raw_map_path=str(path),image_score=score))
            if (i+1)%10==0:print('PREDICT',i+1,'/114',flush=True)
    assert len(rows)==114 and len({r['sample_id'] for r in rows})==114 and nan==inf==0
    image=float(roc_auc_score([r['gt_label'] for r in rows],[r['image_score'] for r in rows]))
    with tempfile.TemporaryDirectory(prefix='phase24c_r1_metrics_') as tmp:
        scratch=Path(tmp)
        scores=np.lib.format.open_memmap(scratch/'scores.npy',mode='w+',dtype=np.float32,shape=(114,*NATIVE))
        targets=np.lib.format.open_memmap(scratch/'targets.npy',mode='w+',dtype=np.uint8,shape=(114,*NATIVE))
        for i,row in enumerate(rows):
            scores[i]=np.load(row['raw_map_path'],allow_pickle=False)
            targets[i]=exact.get_mask(row,Image,np)
        scores.flush();targets.flush()
        pixel=float(roc_auc_score(targets.reshape(-1),scores.reshape(-1)))
        del scores,targets
        er={}
        exact.native(rows,er,torch,cv2,np,Image,temp_dir=scratch/'exact')
        assert er['connected_regions']==450
        result=dict(status='PASS',geometry=[256,576],seed=42,pixel_budget=147456,
            test_public_count=114,normal_count=24,anomaly_count=90,gt_region_count=450,
            image_auroc=image,pixel_auroc=pixel,exact_aupro005=er['aupro_0.05'],
            exact_aupro030=er['aupro_0.30'],nan_count=nan,inf_count=inf,
            checkpoint_sha256=train['checkpoint_sha256'],checkpoint_path=str(CKPT),
            exact_source_sha256=sha(ROOT/'src/evaluation/exact_aupro.py'),
            test_private_files_accessed=0,test_private_mixed_files_accessed=0,
            protocol_violation_count=len(blocked),prediction_count=len(rows),
            native_shape=list(NATIVE),map_restore='bilinear_align_corners_false',
            image_score='native_combined_anomaly_map_max',
            evaluation_elapsed_sec=time.perf_counter()-evaluation_start)
        assert not blocked and not any('test_private' in x for x in accessed)
        save(OUT/'09_mid_evaluation.json',result)
        save(result_dir/'image_scores.json',[dict(sample_id=r['sample_id'],image_score=r['image_score'],gt_label=r['gt_label']) for r in rows])
        print('EVALUATION_PASS',result,flush=True)
if __name__=='__main__':main()




