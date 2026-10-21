"""Phase24B public evaluation of a frozen 70K final checkpoint."""
import argparse,csv,hashlib,json,math,os,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
P=ROOT/'experiments/phase24b_sheetmetal_5seed_geometry'
DATA=ROOT/'datasets/MVTec_AD_2/sheet_metal'
NATIVE=(1056,4224)
accessed=[];blocked=[]
def audit(event,args):
    if event not in ('open','os.listdir','os.scandir') or not args or not isinstance(args[0],(str,bytes)):return
    p=Path(os.fsdecode(args[0])).absolute()
    if p.is_relative_to(ROOT/'datasets'):
        if not p.is_relative_to(DATA/'test_public'):
            blocked.append(str(p));raise PermissionError('Forbidden evaluation dataset: '+str(p))
        if event=='open':accessed.append(str(p))
sys.addaudithook(audit)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,obj):
    path=Path(path)
    if path.exists():raise FileExistsError(str(path))
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def sources():
    path=ROOT/'experiments/phase18b_sheet_metal_square512_70k_formal/evaluation/predictions.csv'
    with path.open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
    assert len(rows)==114 and len({r['sample_id'] for r in rows})==114
    assert sum(int(r['gt_label']) for r in rows)==90
    for r in rows:
        assert Path(r['image_path']).is_relative_to(DATA/'test_public')
        assert int(r['gt_label'])==int(Path(r['image_path']).parent.name=='bad')
        if int(r['gt_label']):assert Path(r['gt_mask_path']).is_relative_to(DATA/'test_public/ground_truth/bad')
    return rows
def evaluate(geometry,seed,verify_seed42=False):
    import cv2,numpy as np,torch
    import torch.nn.functional as F
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    from anomalib.models import EfficientAd
    from anomalib.data.utils import read_image
    from src.evaluation import exact_aupro as exact
    from src.evaluation import rectangular_formal_evaluation as rect
    torch.set_num_threads(4)
    if verify_seed42:
        out=ROOT/('experiments/phase18b_sheet_metal_square512_70k_formal' if geometry=='square512' else 'experiments/phase13b_rectangular_256x1024_seed42')
    else:out=P/f'seed{seed}'/geometry
    train=read(out/'training_report.json')
    checkpoint=out/'checkpoints/final.ckpt'
    assert train['status']=='PASS' and train['completed_steps']==70000
    assert sha(checkpoint)==train['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['global_step']==70000
    meta=saved['phase18b' if geometry=='square512' else 'phase13b']
    assert meta['final_validation_completed']
    cfg=meta['config']
    assert cfg['seed']==seed
    assert (cfg['input_size'] if geometry=='square512' else cfg['input_size_hw'])==([512,512] if geometry=='square512' else [256,1024])
    del saved
    model=EfficientAd.load_from_checkpoint(str(checkpoint),map_location='cpu',strict=True,weights_only=False).eval().cuda()
    assert list(model.pre_processor.transform.transforms[0].size)==([512,512] if geometry=='square512' else [256,1024])
    assert sha(ROOT/'src/evaluation/exact_aupro.py')=='3e6ebc80d400419cf3aaf00950e954ea165df2419c6ea895f4296013c67e11a1'
    references=sources()
    if verify_seed42:references=references[:1]
    with tempfile.TemporaryDirectory(prefix='phase24b_eval_') as tmp:
        scratch=Path(tmp)
        rows=[];nan=inf=0
        with torch.inference_mode():
            for i,source in enumerate(references):
                native=read_image(source['image_path'],as_tensor=True)
                assert tuple(native.shape[-2:])==NATIVE
                x=(model.pre_processor.transform(native[None]).cuda() if geometry=='square512' else model.pre_processor(native[None].cuda()))
                if geometry=='square512':
                    raw=model.model(x).anomaly_map.detach().float()
                    assert tuple(raw.shape)==(1,1,512,512)
                    score=float(raw.max().item())
                    arr=F.interpolate(raw.cpu(),size=NATIVE,mode='bilinear',align_corners=False)[0,0].numpy().copy()
                else:
                    raw,_,errors=rect.extract(model,x)
                    assert max(errors.values())<=1e-5
                    arr=rect.native_map(raw)
                    score=float(arr.max())
                assert arr.shape==NATIVE and arr.dtype==np.float32
                nan+=int(np.isnan(arr).sum())+int(math.isnan(score))
                inf+=int(np.isinf(arr).sum())+int(math.isinf(score))
                path=scratch/(source['sample_id']+'.npy');np.save(path,arr,allow_pickle=False)
                rows.append(dict(sample_id=source['sample_id'],image_path=source['image_path'],gt_label=int(source['gt_label']),gt_mask_path=source['gt_mask_path'],native_shape=NATIVE,raw_map_path=str(path),image_score=score))
                if (i+1)%20==0:print(geometry,seed,'PREDICT',i+1,flush=True)
        if verify_seed42:
            sid=rows[0]['sample_id']
            if geometry=='square512':
                frozen=np.load(ROOT/'experiments/phase18b_sheet_metal_square512_70k_formal/evaluation/raw_anomaly_maps'/(sid+'.npy'),allow_pickle=False)
                print('SQUARE_MAP_DIFF',float(np.max(np.abs(raw[0,0].detach().cpu().numpy()-frozen))),'current_score',score,'frozen_score',float(frozen.max()),flush=True)
                assert np.allclose(raw[0,0].detach().cpu().numpy(),frozen,atol=1e-6,rtol=0)
                expected=float(references[0]['official_image_score'])
            else:
                frozen=np.load(ROOT/'experiments/phase13c_rectangular_256x1024_public/raw_native_combined'/(sid+'.npy'),allow_pickle=False)
                assert np.allclose(arr,frozen,atol=1e-6,rtol=0)
                with (ROOT/'experiments/phase13c_rectangular_256x1024_public/predictions.csv').open(newline='',encoding='utf8') as f:
                    expected=float(next(r['combined_image_score'] for r in csv.DictReader(f) if r['sample_id']==sid))
            assert abs(score-expected)<1e-6,(score,expected)
            print('SEED42_EVALUATOR_EQUIVALENCE_PASS',geometry,'score',score,flush=True)
            return
        assert len(rows)==114 and nan==inf==0
        image=float(roc_auc_score([r['gt_label'] for r in rows],[r['image_score'] for r in rows]))
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
        result=dict(status='PASS',geometry=geometry,seed=seed,test_public_count=114,normal_count=24,anomaly_count=90,gt_region_count=450,image_auroc=image,pixel_auroc=pixel,exact_aupro005=er['aupro_0.05'],exact_aupro030=er['aupro_0.30'],nan_count=nan,inf_count=inf,checkpoint_sha256=train['checkpoint_sha256'],checkpoint_path=str(checkpoint),exact_source_sha256=sha(ROOT/'src/evaluation/exact_aupro.py'),test_private_files_accessed=0,test_private_mixed_files_accessed=0,protocol_violation_count=len(blocked))
        assert not blocked and not any('test_private' in x for x in accessed)
        save(out/'evaluation_result.json',result)
        save(out/'image_scores.json',[dict(sample_id=r['sample_id'],image_score=r['image_score'],gt_label=r['gt_label']) for r in rows])
        print('EVALUATION_PASS',geometry,seed,result,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--geometry',choices=['square512','pb_arpr'],required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--verify-seed42',action='store_true')
    args=ap.parse_args()
    if args.verify_seed42:assert args.seed==42
    else:assert args.seed in (43,44,45,46)
    evaluate(args.geometry,args.seed,args.verify_seed42)
