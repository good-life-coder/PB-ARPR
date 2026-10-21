"""Fixed native whole-image rectangular calibration; original Anomalib modules only."""
import csv
import inspect
import math
import time
from pathlib import Path
import numpy as np
import torch
from lightning.pytorch import Callback, seed_everything
from anomalib.models import EfficientAd
from anomalib.data import MVTecAD2
from src.evaluation.st_feature_rootcause import geometry, pair, map_from_feature, padding_protocol
from src.evaluation.efficientad_branch_audit import parameter_hash, state_hash
from scripts.audit_efficientad_tile_seam import borders

ROOT=Path(__file__).resolve().parents[2]
TARGET=(256,1024)
LOG_FIELDS=['phase','step','epoch','train_st','train_ae','train_stae','total_loss','learning_rate']


def protocol():
    return dict(image_size=TARGET,native_hw=[1056,4224],model_size='small',batch=1,seed=42,
                precision='32-true',padding=False,pad_maps=True,max_steps=500,max_epochs=1000)


def scaling():
    sy=256/1056;sx=1024/4224
    return dict(native_hw=[1056,4224],target_hw=list(TARGET),native_aspect_ratio=4.,target_aspect_ratio=4.,
                scale_y=sy,scale_x=sx,anisotropy=max(sx,sy)/min(sx,sy),horizontal_detail_gain_vs_B0=sx/(256/4224),
                B0_pixels=256**2,rectangular_pixels=256*1024,P1384_pixels=4*384**2,
                pixel_area_ratio_vs_B0=4.,pixel_area_ratio_vs_P1384=(256*1024)/(4*384**2),
                compute_comparison='PIXEL_AREA_PROXY',B0_anisotropy=4.,
                forward_calls=dict(B0='1 call, batch1',rectangular='1 call, batch1',P1384='1 call, batch4 contexts'))


def seed():seed_everything(42,workers=True)


def shape_gate(x):
    if tuple(x.shape)!=(1,3,256,1024):raise ValueError('FAIL_DIMENSION_ORDER: expected [1,3,256,1024], got '+str(list(x.shape)))
    if x.dtype!=torch.float32:raise ValueError('FAIL_PRECISION')


def trainer_config(steps):
    if steps not in (3,500):raise ValueError('Only3 smoke or500 calibration steps allowed')
    return dict(max_steps=steps,max_epochs=1000,accelerator='gpu',devices=1,precision='32-true',
                limit_val_batches=0,num_sanity_val_steps=0,enable_checkpointing=False,logger=False,barebones=True)


def make_model():
    old=EfficientAd.configure_pre_processor()
    new=EfficientAd.configure_pre_processor(image_size=TARGET)
    expected=repr(old.transform).replace('size=[256, 256]','size=[256, 1024]')
    if list(new.transform.transforms[0].size)!=[256,1024]:raise ValueError('FAIL_DIMENSION_ORDER')
    if repr(new.transform)!=expected:raise ValueError('FAIL_PREPROCESSOR_MISMATCH')
    teacher_root=EfficientAd.prepare_pretrained_model.__globals__['get_pretrained_weights_dir']()
    teacher_path=teacher_root/'efficientad_pretrained_weights/pretrained_teacher_small.pth'
    if not teacher_path.is_file() or not (ROOT/'datasets/imagenette').is_dir():
        raise ValueError('FAIL_ASSETS: require existing official teacher and ImageNette, no download fallback')
    model=EfficientAd(model_size='small',imagenet_dir='./datasets/imagenette',pre_processor=new,
                      evaluator=False,post_processor=False,visualizer=False)
    if sum(p.numel() for p in model.parameters())!=8058628:raise ValueError('FAIL_PARAMETERS')
    if model.model.ae.decoder.padding is not False or model.model.pad_maps is not True:
        raise ValueError('FAIL_DEFAULT_PADDING')
    return model,dict(default_transform=repr(old.transform),rectangular_transform=repr(new.transform),
                      only_resize_geometry_changed=True,tuple_order='H,W',actual_image_size=list(TARGET),
                      normalization='Official ImageNet normalization inside PDN/AE; no extra transform',
                      preprocessor_signature=str(inspect.signature(EfficientAd.configure_pre_processor)),
                      preprocessor_source=inspect.getsource(EfficientAd.configure_pre_processor),teacher_path=str(teacher_path))


def architecture_audit(model):
    targets={'EfficientAd.configure_pre_processor':EfficientAd.configure_pre_processor,
             'EfficientAd.on_train_start':EfficientAd.on_train_start,
             'EfficientAd.prepare_imagenette_data':EfficientAd.prepare_imagenette_data,
             'EfficientAdModel.forward':type(model.model).forward,
             'EfficientAdModel.compute_maps':type(model.model).compute_maps,
             'EfficientAdModel.compute_losses':type(model.model).compute_losses,
             'AutoEncoder.forward':type(model.model.ae).forward,
             'Encoder.forward':type(model.model.ae.encoder).forward,
             'Decoder.forward':type(model.model.ae.decoder).forward}
    result={name:dict(module=fn.__module__,signature=str(inspect.signature(fn)),source=inspect.getsource(fn),
                     source_line=inspect.getsourcelines(fn)[1]) for name,fn in targets.items()}
    return dict(functions=result,padding=False,pad_maps=True,
                decoder_axis_semantics='image_size[0] controls height; image_size[1] controls width in every interpolation target and final upsample',
                explicit_square_requirement_found=False,
                caveat='Source has no H==W constraint; feasibility still gated on actual forward and training.',
                quantile_protocol='Official normalization formula retained. No validation or quantile fitting in this train-only calibration; fitted quantile state remains untested.')


def dataset():
    dm=MVTecAD2(root='./datasets/MVTec_AD_2',category='sheet_metal',train_batch_size=1,
                eval_batch_size=1,num_workers=0,test_type='public')
    dm.setup()
    counts=dict(train=len(dm.train_data),validation=len(dm.val_data),test_public=len(dm.test_public_data))
    if counts!=dict(train=137,validation=19,test_public=114):raise ValueError('FAIL_DATA: '+str(counts))
    labels=dm.test_public_data.samples.label_index
    normal=int((labels==0).sum());anomaly=int((labels==1).sum())
    if (normal,anomaly)!=(24,90):raise ValueError(f'FAIL_DATA: public normal={normal}, anomaly={anomaly}')
    if not (dm.train_data.samples.label_index==0).all():raise ValueError('FAIL_DATA: abnormal training sample')
    paths=dm.train_data.samples.image_path.tolist()
    return dm,dict(counts=counts,test_public_normal=normal,test_public_anomaly=anomaly,
                   test_data_count=len(dm.test_data),test_data_semantics='anomalous public-test subset',
                   train_paths=paths,data_root=str(ROOT/'datasets/MVTec_AD_2'),
                   split_usage='Train only; validation/test metadata counts, no validation or test iteration')


def orientation(pre):
    x=torch.zeros(1,3,1056,4224)
    x[:,:,0:528,0:2112]=torch.tensor([1.,0.,0.])[None,:,None,None]
    x[:,:,0:528,2112:]=torch.tensor([0.,1.,0.])[None,:,None,None]
    x[:,:,528:,0:2112]=torch.tensor([0.,0.,1.])[None,:,None,None]
    x[:,:,528:,2112:]=1.
    with torch.no_grad():y=pre(x)
    shape_gate(y)
    actual=[y[0,:,yy,xx].tolist() for yy,xx in [(20,20),(20,1000),(230,20),(230,1000)]]
    expected=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]]
    if not np.allclose(actual,expected,atol=1e-6):raise ValueError('FAIL_ORIENTATION')
    return dict(status='PASS',marker_order='top-left,top-right,bottom-left,bottom-right',colors=actual)


def input_checks(model,paths):
    from anomalib.data.utils import read_image
    rows=[]
    for index in np.random.default_rng(42).choice(len(paths),5,replace=False):
        path=paths[int(index)];native=read_image(path,as_tensor=True)
        if tuple(native.shape)!=(3,1056,4224):raise ValueError('FAIL_NATIVE_GEOMETRY')
        with torch.no_grad():x=model.pre_processor(native[None].to('cuda:0'))
        shape_gate(x)
        rows.append(dict(path=str(path),native_shape=list(native.shape),preprocessed_shape=list(x[0].shape)))
    return rows


def probe(model,x):
    shape_gate(x)
    if model.model.training:raise ValueError('Probe requires eval')
    before=parameter_hash(model);state_before=state_hash(model)
    captures={};encoder={};decoder={};traces={'teacher':[],'student':[]};handles=[]
    def capture(tag):
        def hook(module,args,out):captures[tag]=out.detach()
        return hook
    def layer_hook(store,name):
        def hook(module,args,out):store[name]=dict(input=list(args[0].shape),output=list(out.shape))
        return hook
    def spatial(tag,name):
        def hook(module,args,out):
            traces[tag].append(dict(name=name,kind=type(module).__name__,kernel=pair(module.kernel_size),
                stride=pair(module.stride or module.kernel_size),padding=pair(module.padding),
                dilation=pair(getattr(module,'dilation',1)),ceil_mode=bool(getattr(module,'ceil_mode',False)),
                observed_output_shape=list(out.shape[-2:])))
        return hook
    try:
        for tag,module in [('teacher',model.model.teacher),('student',model.model.student),('ae',model.model.ae)]:
            handles.append(module.register_forward_hook(capture(tag)))
        for tag in traces:
            for name,module in getattr(model.model,tag).named_modules():
                if isinstance(module,(torch.nn.Conv2d,torch.nn.AvgPool2d,torch.nn.MaxPool2d)):
                    handles.append(module.register_forward_hook(spatial(tag,name)))
        for name,module in model.model.ae.encoder.named_modules():
            if name.startswith('enconv'):handles.append(module.register_forward_hook(layer_hook(encoder,name)))
        for name,module in model.model.ae.decoder.named_modules():
            if name.startswith('deconv'):handles.append(module.register_forward_hook(layer_hook(decoder,name)))
        with torch.no_grad():out=model.model(x)
    finally:
        for h in handles:h.remove()
    c=model.model.teacher_out_channels;teacher=captures['teacher'];student=captures['student'];ae=captures['ae']
    if teacher.shape!=student[:,:c].shape or ae.shape!=student[:,c:].shape:
        raise ValueError('FAIL_RECTANGULAR_SHAPE_MISMATCH: '+str({k:list(v.shape) for k,v in captures.items()}))
    if out.anomaly_map.shape!=(1,1,256,1024) or not torch.isfinite(out.anomaly_map).all():raise ValueError('FAIL_RECTANGULAR_OUTPUT')
    with torch.no_grad():
        student,D=model.model.compute_student_teacher_distance(x)
        raw_st,raw_ae=model.model.compute_maps(x,student,D,normalize=False)
        pad=padding_protocol(model)
        reconstructed=map_from_feature(D.mean(1,keepdim=True),pad,TARGET)
        err=float((raw_st-reconstructed).abs().max())
        if err>1e-6:raise ValueError('FAIL_ST_RECONSTRUCTION')
        support=map_from_feature(torch.ones_like(D[:,:1]),pad,TARGET)[0,0].cpu().numpy()
    tg=geometry(traces['teacher'],TARGET);sg=geometry(traces['student'],TARGET)
    if any(tg[k]!=sg[k] for k in ('jump','rf','start','output_shape')):raise ValueError('FAIL_FEATURE_GEOMETRY')
    edge=borders(support,np);strict=borders((support>=1).astype(np.float32),np)
    after=parameter_hash(model)
    if before!=after or state_before!=state_hash(model):raise ValueError('FAIL_PARAMETER_MUTATION')
    return dict(input_shape=list(x.shape),teacher_feature_shape=list(teacher.shape),student_feature_shape=list(student.shape),
                student_st_shape=list(student[:,:c].shape),st_discrepancy_shape=list(D.shape),
                ae_feature_shape=list(ae.shape),student_ae_half_shape=list(student[:,c:].shape),
                st_shape_match=True,ae_shape_match=True,encoder_layers=encoder,decoder_layers=decoder,
                decoder_interpolation_targets={name:v['input'][-2:] for name,v in decoder.items() if name!='deconv8'},
                decoder_target_observation='Each deconv1..7 pre-convolution input immediately follows official F.interpolate; no patched interpolation or model code.',
                prepad_st_map_shape=[1,1,*D.shape[-2:]],prepad_stae_map_shape=[1,1,*D.shape[-2:]],
                padded_feature_map_shape=[1,1,D.shape[-2]+8,D.shape[-1]+8],feature_padding=list(pad),
                st_raw_map_shape=list(raw_st.shape),stae_raw_map_shape=list(raw_ae.shape),
                final_anomaly_map_shape=list(out.anomaly_map.shape),st_reconstruction_max_error=err,
                teacher_geometry=tg,student_geometry=sg,feature_aspect_ratio=D.shape[-1]/D.shape[-2],input_aspect_ratio=4.,
                invalid_support_border_model=edge,strict_unmixed_border_model=strict,
                native_invalid_border_equivalent={k:v*(1056/256 if k in ('top','bottom') else 4224/1024) for k,v in edge.items()},
                actual_map_borders={name:dict(exact_zero=borders(a[0,0].cpu().numpy(),np),near_zero=borders(a[0,0].cpu().numpy(),np,1e-12)) for name,a in [('st',raw_st),('stae',raw_ae),('combined',out.anomaly_map)]},
                parameter_hash_before=before,parameter_hash_after=after,state_unchanged=True)


def imagenette_probe(model):
    model.prepare_imagenette_data(TARGET)
    x=next(iter(model.imagenet_loader))[0]
    shape_gate(x)
    return dict(received_image_size=list(TARGET),batch_shape=list(x.shape),transform=repr(model.data_transforms_imagenet),status='PASS')


def loss_probe(model,x):
    image=imagenette_probe(model)
    extra=next(iter(model.imagenet_loader))[0].to(x.device);before=parameter_hash(model)
    model.model.train()
    try:
        with torch.no_grad():values=model.model(batch=x,batch_imagenet=extra)
        losses=[float(v) for v in values];losses.append(sum(losses))
        if len(values)!=3 or not all(math.isfinite(v) for v in losses):raise ValueError('FAIL_NUMERICS')
    finally:model.model.eval()
    unchanged=before==parameter_hash(model)
    if not unchanged:raise ValueError('FAIL_PARAMETER_MUTATION')
    return dict(losses=losses,parameters_unchanged=unchanged,imagenette_shape=image['batch_shape'])


class RectangularMonitor(Callback):
    """Observe official loss/input tensors, never replace a training computation."""
    def __init__(self,report,path,steps,phase):
        self.r=report;self.path=Path(path);self.steps=steps;self.phase=phase
        self.file=None;self.hook=None;self.losses=None;self.start=None

    def on_fit_start(self,trainer,pl_module):
        from lightning.pytorch.callbacks import ModelCheckpoint,EarlyStopping
        if any(isinstance(c,(ModelCheckpoint,EarlyStopping)) for c in trainer.callbacks):raise ValueError('FAIL_CALLBACK_PROTOCOL')
        if trainer.precision!='32-true' or trainer.limit_val_batches!=0:raise ValueError('FAIL_TRAINER_PROTOCOL')
        self.file=self.path.open('a',encoding='utf-8',newline='');self.writer=csv.DictWriter(self.file,fieldnames=LOG_FIELDS)
        self.r['optimizer']=[dict(class_name=type(o).__name__,lr=[g['lr'] for g in o.param_groups],weight_decay=[g['weight_decay'] for g in o.param_groups]) for o in trainer.optimizers]
        def capture(module,args,kwargs,out):
            x=kwargs.get('batch',args[0] if args else None);extra=kwargs.get('batch_imagenet')
            shape_gate(x);shape_gate(extra)
            if not isinstance(out,tuple) or len(out)!=3:raise ValueError('FAIL_OFFICIAL_LOSS_TUPLE')
            self.losses=[float(v.detach()) for v in out]
            self.r['observed_train_input_shape']=list(x.shape);self.r['observed_imagenette_shape']=list(extra.shape)
        self.hook=pl_module.model.register_forward_hook(capture,with_kwargs=True)

    def on_before_backward(self,trainer,pl_module,loss):
        self.total=float(loss.detach())
        if self.losses is None:raise ValueError('FAIL_LOSS_OBSERVATION')
        values=self.losses+[self.total]
        self.r['nan_detected']|=any(math.isnan(v) for v in values)
        self.r['inf_detected']|=any(math.isinf(v) for v in values)
        if not all(math.isfinite(v) for v in values):raise ValueError('FAIL_NUMERICS')
        self.r['loss_checks']+=1

    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        step=int(trainer.global_step);self.r['completed_steps']=step
        self.writer.writerow(dict(phase=self.phase,step=step,epoch=int(trainer.current_epoch),
            train_st=self.losses[0],train_ae=self.losses[1],train_stae=self.losses[2],total_loss=self.total,
            learning_rate=trainer.optimizers[0].param_groups[0]['lr']))
        self.r['final_losses']=self.losses+[self.total]
        if step==50:torch.cuda.synchronize();self.start=time.perf_counter()
        if step==500:
            torch.cuda.synchronize();self.r['steady_seconds']=time.perf_counter()-self.start
        if step%50==0 or step==self.steps:
            self.file.flush();print(f'{self.phase}: {step}/{self.steps} optimizer steps; loss={self.total:.6f}',flush=True)

    def on_validation_start(self,trainer,pl_module):raise ValueError('FAIL_VALIDATION_FORBIDDEN')
    def on_test_start(self,trainer,pl_module):raise ValueError('FAIL_TEST_FORBIDDEN')
    def close(self):
        if self.hook is not None:self.hook.remove();self.hook=None
        if self.file is not None:self.file.close();self.file=None
    def on_fit_end(self,trainer,pl_module):self.close()
    def on_exception(self,trainer,pl_module,exception):self.close()
