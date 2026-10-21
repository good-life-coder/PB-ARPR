"""Phase24C-R1 selected Mid 256x576 seed42, frozen EfficientAD-S 70K training."""
import argparse, hashlib, json, math, os, sys, time, traceback
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/phase24c_r1_architecture_constrained_budget'
RUN=OUT/'08_mid_training'
FINAL=RUN/'checkpoints/final.ckpt'
LAST=RUN/'checkpoints/last.ckpt'
DATA=ROOT/'datasets/MVTec_AD_2/sheet_metal'
accessed_test=[]
def audit(event,args):
    if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes)):
        p=Path(os.fsdecode(args[0])).absolute()
        if os.environ.get('PHASE24C_R1_FORMAL_EVAL')=='1' and p.is_relative_to(DATA/'test_public'):
            return
        if any(p.is_relative_to(DATA/s) for s in ('test_public','test_private','test_private_mixed')):
            accessed_test.append(str(p));raise PermissionError('Test access forbidden during training: '+str(p))
sys.addaudithook(audit)
sys.path.insert(0,str(ROOT))
import torch
from lightning.pytorch import LightningDataModule,seed_everything
from lightning.pytorch.callbacks import Callback,LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader
from anomalib.callbacks import ModelCheckpoint
from anomalib.data.datamodules.image.mvtecad2 import MVTecAD2Dataset
from anomalib.data.utils import Split
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from scripts.train_baseline384 import check_logs
MIB=1024**2
def now():return datetime.now().astimezone().isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,obj):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*MIB),b''):h.update(b)
    return h.hexdigest()
def protocol():
    return dict(phase='Phase24C-R1',geometry=[256,576],pixel_budget=147456,seed=42,
        dataset='MVTec AD 2',category='sheet_metal',train_normal_count=137,
        validation_normal_count=19,model='EfficientAD-S',model_size='small',
        training_steps=70000,batch_size=1,num_workers=0,precision='32-true',
        optimizer='Adam',learning_rate=1e-4,weight_decay=1e-5,scheduler='StepLR',
        scheduler_step_size=66500,scheduler_gamma=0.1,max_epochs=1000,
        validation_during_fit=False,final_full_validation=True,early_stopping=False,
        imagenette_dir='./datasets/imagenette',teacher_frozen=True,padding=False,
        pad_maps=True,checkpoint_selection='final_70000_only',test_access_during_training=0,
        reference='Phase24B frozen EfficientAD-S')
from pandas import DataFrame
from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.datamodules.base import AnomalibDataModule
from anomalib.data.datasets.image.mvtecad2 import IMG_EXTENSIONS
class GoodOnlyDataset(MVTecAD2Dataset):
    """Official sample layout and __getitem__; enumerate only the requested normal split."""
    def __init__(self, split):
        AnomalibDataset.__init__(self, augmentations=None)
        self.root_category=DATA
        self.split=split
        folder=DATA/('train' if split==Split.TRAIN else 'validation')/'good'
        paths=[p for p in folder.iterdir() if p.is_file() and p.suffix in IMG_EXTENSIONS]
        rows=[(str(DATA),split.value,'good',str(p),None,0) for p in paths]
        frame=DataFrame(rows,columns=['path','split','label','image_path','mask_path','label_index'])
        frame.attrs['task']='segmentation'
        self.samples=frame
class TrainValOnly(LightningDataModule):
    def __init__(self,model_transform):
        super().__init__();self.model_transform=model_transform
    name='MVTecAD2'
    category='sheet_metal'
    train_batch_size=1
    eval_batch_size=1
    num_workers=0
    def setup(self,stage=None):
        if not hasattr(self,'train_data'):
            self.train_data=GoodOnlyDataset(Split.TRAIN)
            self.val_data=GoodOnlyDataset(Split.VAL)
            assert len(self.train_data)==137 and len(self.val_data)==19
            for ds,split in ((self.train_data,'train'),(self.val_data,'validation')):
                folder=DATA/split/'good'
                assert all(Path(p).absolute().is_relative_to(folder) for p in ds.samples.image_path)
                assert all(int(x)==0 for x in ds.samples.label_index)
            for ds in (self.train_data,self.val_data):
                AnomalibDataModule._update_subset_augmentations(ds,None,self.model_transform)
            assert not accessed_test
    def train_dataloader(self):
        return DataLoader(self.train_data,shuffle=True,batch_size=1,num_workers=0,collate_fn=self.train_data.collate_fn)
    def val_dataloader(self):
        return DataLoader(self.val_data,shuffle=False,batch_size=1,num_workers=0,collate_fn=self.val_data.collate_fn)
class Monitor(Callback):
    @property
    def state_key(self):return 'Phase24CR1Monitor'
    def __init__(self,report,config,timing):
        self.report=report;self.config=config;self.timing=timing;self.last_loss=None
    def state_dict(self):return dict(last_loss=self.last_loss,nan=self.report['loss_nan_detected'],inf=self.report['loss_inf_detected'])
    def load_state_dict(self,state):
        self.last_loss=state.get('last_loss')
        self.report['loss_nan_detected']=state.get('nan',False)
        self.report['loss_inf_detected']=state.get('inf',False)
    def on_train_batch_start(self,trainer,module,batch,batch_idx):
        if tuple(batch.image.shape)!=(1,3,256,576):raise ValueError('FAIL_TRAIN_INPUT_GEOMETRY: '+str(tuple(batch.image.shape)))
    def on_train_start(self,trainer,module):
        assert trainer.max_steps==70000 and trainer.precision=='32-true' and trainer.limit_val_batches==0
        assert not accessed_test
    def on_before_backward(self,trainer,module,loss):
        v=float(loss.detach());self.last_loss=v
        self.report['loss_nan_detected']|=math.isnan(v)
        self.report['loss_inf_detected']|=math.isinf(v)
        if not math.isfinite(v):raise FloatingPointError('Nonfinite loss')
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        step=int(trainer.global_step)
        if step==1 or step%100==0:
            elapsed=self.timing['prior']+time.perf_counter()-self.timing['start']
            self.report.update(status='TRAINING',completed_steps=step,global_step=step,
                training_elapsed_seconds=elapsed,training_elapsed_hours=elapsed/3600,
                estimated_remaining_hours=(70000-step)*elapsed/step/3600,
                peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/MIB,
                peak_cuda_reserved_mib=torch.cuda.max_memory_reserved()/MIB,
                last_loss=self.last_loss,progress_timestamp=now())
            write(RUN/'training_report.json',self.report)
            write(OUT/'progress.json',dict(stage='MID_TRAINING',geometry=[256,576],
                completed_steps=step,target_steps=70000,elapsed_hours=elapsed/3600,
                measured_remaining_eta_hours=self.report['estimated_remaining_hours']))
            print(f'Phase24C-R1 Mid {step}/70000 elapsed={elapsed/3600:.4f}h eta={self.report["estimated_remaining_hours"]:.4f}h',flush=True)
    def on_validation_start(self,trainer,module):
        if int(trainer.global_step)!=70000:raise ValueError('Validation before final step')
    def on_validation_batch_end(self,trainer,module,outputs,batch,batch_idx,dataloader_idx=0):
        self.report['final_validation_batches']+=1
    def on_test_start(self,trainer,module):raise PermissionError('Test forbidden during training')
    def on_save_checkpoint(self,trainer,module,checkpoint):
        elapsed=self.timing['prior']+(time.perf_counter()-self.timing['start'] if self.timing['start'] else 0)
        checkpoint['phase24c_r1']=dict(protocol=protocol(),config=self.config,
            training_elapsed_seconds=self.timing['finished'] if self.timing['finished'] is not None else elapsed,
            timestamp_start=self.report['timestamp_start'],log_paths=self.report['log_paths'],
            final_validation_completed=self.report['final_validation_completed'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    os.chdir(ROOT);torch.set_num_threads(4)
    RUN.mkdir(parents=True,exist_ok=True);(RUN/'checkpoints').mkdir(exist_ok=True);(RUN/'logs').mkdir(exist_ok=True)
    if FINAL.exists():raise FileExistsError('Final checkpoint exists')
    if LAST.exists() and not args.resume:raise FileExistsError('Existing last.ckpt requires --resume')
    if (RUN/'training_report.json').exists() and not args.resume:raise FileExistsError('Existing report requires review/resume')
    selected=read(OUT/'07_mid_selection.json')
    assert selected['exact_pixel_nonsquare_feasible'] and selected['selected_geometry']==[256,576]
    assert selected['selection_used_test_metrics'] is False
    assert read(OUT/'01_previous_phase_gate.json')['status']=='PASS'
    prior=read(ROOT/'experiments/phase24c_multibudget_geometry_sensitivity/02_reference_audit.json')
    assert prior['status']=='PASS' and prior['references']['square384']['status']=='PASS'
    assert torch.cuda.is_available()
    seed_everything(42,workers=True)
    b0=read(ROOT/'experiments/phase07b_baseline256_seed42/config.json')
    freeze=protocol()
    for k in ('model_size','batch_size','num_workers','max_steps','max_epochs','optimizer',
              'learning_rate','weight_decay','scheduler','scheduler_step_size','scheduler_gamma',
              'validation_during_fit','final_full_validation','early_stopping','evaluator',
              'visualizer','post_processor'):
        key='training_steps' if k=='max_steps' else k
        if key in freeze:assert freeze[key]==b0[k],(k,freeze[key],b0[k])
    old=EfficientAd.configure_pre_processor()
    pre=EfficientAd.configure_pre_processor(image_size=(256,576))
    assert repr(pre.transform)==repr(old.transform).replace('size=[256, 256]','size=[256, 576]')
    config=dict(freeze,preprocessor=repr(pre.transform),total_parameters=8058628)
    dm=TrainValOnly(pre.transform);dm.setup()
    model=EfficientAd(model_size='small',imagenet_dir='./datasets/imagenette',
        pre_processor=pre,evaluator=False,post_processor=False,visualizer=False)
    assert sum(p.numel() for p in model.parameters())==8058628
    assert model.lr==1e-4 and model.weight_decay==1e-5
    assert not model.model.ae.decoder.padding and model.model.pad_maps
    teacher_asset=EfficientAd.prepare_pretrained_model.__globals__['get_pretrained_weights_dir']()/'efficientad_pretrained_weights/pretrained_teacher_small.pth'
    assert teacher_asset.is_file() and (ROOT/'datasets/imagenette').is_dir()
    config['teacher_sha256']=sha(teacher_asset)
    config['selection_sha256']=sha(OUT/'07_mid_selection.json')
    report=dict(status='PRECHECK_PASS',timestamp_start=now(),timestamp_end=None,
        target_steps=70000,completed_steps=0,global_step=0,geometry=[256,576],seed=42,
        training_elapsed_seconds=None,training_elapsed_hours=None,
        estimated_remaining_hours=None,peak_cuda_allocated_mib=None,
        peak_cuda_reserved_mib=None,loss_nan_detected=False,loss_inf_detected=False,
        final_validation_batches=0,final_validation_completed=False,final_quantiles=None,
        checkpoint_path=None,checkpoint_sha256=None,log_paths=[],resume=args.resume,
        test_public_files_accessed=0,test_mask_files_accessed=0,protocol_violation_count=0)
    timing=dict(prior=0.,start=None,finished=None)
    if args.resume:
        saved=torch.load(LAST,map_location='cpu',weights_only=False)
        meta=saved['phase24c_r1']
        assert meta['protocol']==freeze and meta['config']==config
        assert 0<int(saved['global_step'])<=70000
        assert saved['optimizer_states'] and saved['lr_schedulers']
        report['timestamp_start']=meta['timestamp_start']
        report['log_paths']=meta['log_paths']
        timing['prior']=float(meta['training_elapsed_seconds'])
        del saved
    logger=CSVLogger(save_dir=str(RUN/'logs'),name='baseline',flush_logs_every_n_steps=100)
    report['log_paths'].append(str(Path(logger.log_dir)/'metrics.csv'))
    write(RUN/'config.json',config);logger.log_hyperparams(config)
    monitor=Monitor(report,config,timing)
    recovery=ModelCheckpoint(dirpath=str(RUN/'checkpoints'),monitor=None,save_top_k=0,
        save_last=True,every_n_train_steps=1000,save_on_train_epoch_end=False,
        enable_version_counter=False)
    engine=Engine(max_steps=70000,max_epochs=1000,accelerator='gpu',devices=1,
        precision='32-true',num_sanity_val_steps=0,limit_train_batches=1.0,
        limit_val_batches=0,enable_checkpointing=True,enable_progress_bar=False,
        default_root_dir=str(RUN),logger=logger,log_every_n_steps=1,
        callbacks=[monitor,LearningRateMonitor(logging_interval='step'),recovery])
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    timing['start']=time.perf_counter();write(RUN/'training_report.json',report)
    engine.fit(model=model,datamodule=dm,ckpt_path=str(LAST) if args.resume else None)
    torch.cuda.synchronize();timing['finished']=timing['prior']+time.perf_counter()-timing['start']
    assert engine.trainer.global_step==70000
    logger.save();check_logs(report['log_paths'],report)
    assert not report['loss_nan_detected'] and not report['loss_inf_detected']
    report.update(status='VALIDATING',completed_steps=70000,global_step=70000,
        training_elapsed_seconds=timing['finished'],training_elapsed_hours=timing['finished']/3600,
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/MIB,
        peak_cuda_reserved_mib=torch.cuda.max_memory_reserved()/MIB)
    write(RUN/'training_report.json',report)
    engine.trainer.limit_val_batches=1.0
    engine.trainer.validate(model=model,datamodule=dm,ckpt_path=None,verbose=False)
    q={k:float(v.detach().cpu()) for k,v in model.model.quantiles.items()}
    assert report['final_validation_batches']==19
    assert set(q)=={'qa_st','qb_st','qa_ae','qb_ae'} and all(math.isfinite(x) for x in q.values())
    assert q['qb_st']>q['qa_st'] and q['qb_ae']>q['qa_ae']
    report.update(final_validation_completed=True,final_quantiles=q)
    if FINAL.exists():raise FileExistsError(FINAL)
    engine.trainer.save_checkpoint(str(FINAL),weights_only=False)
    saved=torch.load(FINAL,map_location='cpu',weights_only=False)
    assert saved['global_step']==70000
    assert saved['phase24c_r1']['final_validation_completed']
    assert saved['phase24c_r1']['config']==config
    del saved
    report.update(status='PASS',checkpoint_path=str(FINAL),checkpoint_sha256=sha(FINAL),
        checkpoint_size_bytes=FINAL.stat().st_size,timestamp_end=now(),
        test_public_files_accessed=len(accessed_test),test_mask_files_accessed=0)
    assert not accessed_test
    write(RUN/'training_report.json',report)
    write(OUT/'08_mid_training_summary.json',report)
    write(OUT/'progress.json',dict(stage='MID_TRAINING_COMPLETE',geometry=[256,576],
        completed_steps=70000,target_steps=70000,elapsed_hours=timing['finished']/3600,
        measured_remaining_eta_hours=0))
    print('TRAINING_PASS',report['training_elapsed_hours'],'hours',report['checkpoint_sha256'],flush=True)
if __name__=='__main__':
    try:
        from scripts.phase24c_r1_train_mid import main as run
        run()
    except BaseException as e:
        RUN.mkdir(parents=True,exist_ok=True)
        p=RUN/'failure.json'
        write(p,dict(status='HOLD_MID_TRAINING_FAILED',error=type(e).__name__+': '+str(e),
                     traceback=traceback.format_exc(),test_access_blocked=accessed_test,timestamp=now()))
        traceback.print_exc()
        raise






