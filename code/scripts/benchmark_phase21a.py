"""Frozen-model, no-training unified native-image efficiency benchmark."""
import os,sys,json,csv,time,gc,hashlib,platform,ctypes,importlib.util,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'experiments/phase21a_unified_efficiency_complexity';DATA=ROOT/'datasets/MVTec_AD_2/sheet_metal'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(n,d):
    p=OUT/n;p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf8')
def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'completion_verification.json').exists()
    historical={};blocked=[];results=[]
    def report(rel):
        p=ROOT/'experiments'/rel;historical[str(p)]=sha(p);return read(p)
    def guard(event,args):
        if event=='socket.connect':raise PermissionError('offline benchmark')
        if event not in ('open','os.listdir','os.scandir') or not args or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(ROOT/'datasets'):
            ok=(p.is_relative_to(DATA/'validation/good') or p.is_relative_to(DATA/'test_public/good') or p.is_relative_to(DATA/'test_public/bad') or p==ROOT/'datasets/derived/sheet_metal_tile4/manifest.csv' or p.is_relative_to(ROOT/'datasets/derived/sheet_metal_tile4/validation/images'))
            if not ok:blocked.append(str(p));raise PermissionError(str(p))
    sys.addaudithook(guard)
    try:
        import torch,numpy as np,torchvision,timm,importlib.metadata as md
        from anomalib.models import EfficientAd,Patchcore,Fastflow
        from anomalib.data.utils import read_image
        from src.evaluation.tile_stitch import stitch_tiles
        from src.evaluation.rectangular_formal_evaluation import native_map as rect_native
        os.environ['HF_HUB_OFFLINE']='1'
        def sync():torch.cuda.synchronize()
        def forbidden(*args,**kwargs):raise RuntimeError('Training forbidden')
        torch.autograd.backward=forbidden;torch.optim.Optimizer.step=forbidden
        class Memory(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in ['total','avail','page','availpage','virtual','availvirtual','ext']]
        mem=Memory();mem.length=ctypes.sizeof(mem);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as k:cpu=winreg.QueryValueEx(k,'ProcessorNameString')[0]
        env=dict(os=platform.platform(),cpu=cpu,system_ram_gib=mem.total/2**30,python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),gpu_vram_mib=torch.cuda.get_device_properties(0).total_memory/2**20,torchvision=torchvision.__version__,anomalib=md.version('anomalib'),timm=timm.__version__)
        assert env['anomalib']=='2.6.1';save('01_environment.json',env)
        inv=[]
        for name,train,ev,hw in [('EfficientAD-S','phase07b_baseline256_seed42','phase07c_baseline256_public',[256,256]),('Square384','phase08b_baseline384_seed42','phase08c_baseline384_public',[384,384]),('PB-ARPR','phase13b_rectangular_256x1024_seed42','phase13c_rectangular_256x1024_public',[256,1024])]:
            tr=report(train+'/training_report.json');er=report(ev+'/evaluation_report.json')
            if name=='PB-ARPR':m=report(ev+'/combined_metrics.json')
            else:
                m=er['metrics'].copy()
                if name=='EfficientAD-S':m.update({k:v for k,v in report('phase07c_r1_aupro_exact/aupro_report.json').items() if k in ['aupro_0.05','aupro_0.30']})
            metrics=dict(image_auroc=m['image_auroc'],pixel_auroc=m['pixel_auroc'],aupro005=m.get('aupro005',m.get('aupro_0.05')),aupro030=m.get('aupro030',m.get('aupro_0.30')))
            inv.append(dict(method=name,kind='ead',hw=hw,path=tr['checkpoint_path'],sha256=tr['checkpoint_sha256'],training_or_build_time_sec=tr['training_elapsed_seconds'],metrics=metrics))
        for name,folder,kind,hw in [('Square512','phase18b_sheet_metal_square512_70k_formal','ead',[512,512]),('Tile4','phase19b_sheet_metal_tile4_70k_formal','tile',[256,256]),('PatchCore','phase20b_p2_patchcore_formal','patch',[256,256]),('FastFlow','phase20c_p2_fastflow_formal','fast',[256,256])]:
            c=report(folder+'/completion_verification.json');assert c['status']=='PASS'
            cost=c['total_build_sec'] if kind=='patch' else c['training_wall_time_sec'] if kind=='fast' else c['training_wall_time_hours']*3600
            inv.append(dict(method=name,kind=kind,hw=hw,path=c['checkpoint_path'],sha256=c['checkpoint_sha256'],training_or_build_time_sec=cost,
              metrics=dict(image_auroc=c['image_auroc'],pixel_auroc=c['pixel_auroc'],aupro005=c['exact_aupro005'],aupro030=c['exact_aupro030'])))
        order=['EfficientAD-S','Square384','Square512','PB-ARPR','Tile4','PatchCore','FastFlow'];inv.sort(key=lambda r:order.index(r['method']))
        for r in inv:assert sha(r['path'])==r['sha256']
        save('02_frozen_model_inventory.json',inv)
        with (ROOT/'experiments/phase07c_baseline256_public/predictions.csv').open(encoding='utf8') as f:public=[Path(r['image_path']) for r in csv.DictReader(f)]
        warm=sorted((DATA/'validation/good').glob('*.png'))[:10];assert len(public)==len(set(public))==114 and len(warm)==10
        with (ROOT/'datasets/derived/sheet_metal_tile4/manifest.csv').open(encoding='utf8') as f:tiles=list(csv.DictReader(f))
        tiled={}
        for r in tiles:
            if r['split'] in ['validation','test_public']:tiled.setdefault(str(Path(r['source_image_path'])),[]).append(r)
        for p in public+warm:
            rr=sorted(tiled[str(p)],key=lambda r:int(r['tile_id']));assert len(rr)==4
            assert [(int(r['x0']),int(r['x1']),int(r['y0']),int(r['y1'])) for r in rr]==[(i*1056,(i+1)*1056,0,1056) for i in range(4)]
        profiler_tools={n:importlib.util.find_spec(n) is not None for n in ['fvcore','thop','ptflops']}
        save('03_benchmark_protocol.json',dict(order=order,repeats=3,native_count=114,warmup_count=10,precision='FP32',batch_native=1,inference_mode=True,cuda_synchronize=True,std_ddof=1,
          public_sources=[str(p) for p in public],warmup_sources=[str(p) for p in warm],output_residency='native float32 map and scalar score in CPU memory for all methods',
          tile_input='read/decode native once; exact frozen nonoverlap1056 crops, checked against saved validation tiles; original stitch_tiles unmodified',profiler_tools=profiler_tools,
          metrics='READ_FROM_FROZEN_REPORTS; no metric computation',model_only='GPU tensors ready through complete model internal scoring; excludes preprocessing/H2D/restore',e2e='native read through CPU native map and official score; excludes saving/metrics'))
        for idx,r in enumerate(inv):
            print('LOADING',r['method'],flush=True);kind=r['kind'];hw=r['hw']
            saved=torch.load(r['path'],map_location='cpu',weights_only=False)
            if kind in ['ead','tile']:
                model=EfficientAd(model_size='small',imagenet_dir='./datasets/imagenette',pre_processor=EfficientAd.configure_pre_processor(image_size=tuple(hw)),post_processor=False,evaluator=False,visualizer=False)
                assert saved['global_step']==70000
            else:
                # Override only the offline HF cache location, never weights or architecture.
                import huggingface_hub.constants as hc
                cache=ROOT/'experiments'/('phase20b_p1_r3_patchcore_smoke_resume' if kind=='patch' else 'phase20c_p1_fastflow_smoke')/'weights/hub'
                hc.HF_HUB_CACHE=str(cache);hc.HF_HUB_OFFLINE=True;os.environ['HF_HUB_CACHE']=str(cache)
                if kind=='patch':model=Patchcore(backbone='wide_resnet50_2.racm_in1k',layers=['layer2','layer3'],pre_trained=True,coreset_sampling_ratio=0.1,num_neighbors=9,precision='float32',pre_processor=Patchcore.configure_pre_processor(image_size=(256,256),center_crop_size=None),post_processor=False,evaluator=False,visualizer=False)
                else:model=Fastflow(backbone='resnet18',pre_trained=True,flow_steps=8,hidden_ratio=1.0,conv3x3_only=False,pre_processor=Fastflow.configure_pre_processor((256,256)),post_processor=False,evaluator=False,visualizer=False)
            model.load_state_dict(saved['state_dict'],strict=True)
            assert all(torch.equal(v.cpu(),saved['state_dict'][k]) for k,v in model.state_dict().items());del saved
            total=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
            if kind in ['ead','tile']:assert total==8058628
            if kind=='fast':assert (total,trainable)==(7669568,3510528)
            if kind=='patch':assert list(model.model.memory_bank.shape)==[14028,1536]
            model.eval().cuda()
            def pipeline(path):
                sync();t=time.perf_counter();native=read_image(path,as_tensor=True);assert native.shape==(3,1056,4224)
                inputs=[native] if kind!='tile' else [native[:,:1056,i*1056:(i+1)*1056] for i in range(4)]
                tilemaps=[];mts=[]
                for j,img in enumerate(inputs):
                    if kind in ['ead','tile']:x=model.pre_processor(img[None].cuda())
                    else:x=model.pre_processor.transform(img)[None].cuda()
                    assert x.shape==(1,3,*hw) and x.dtype==torch.float32
                    sync();m=time.perf_counter();pred=model.model(x);sync();mts.append((time.perf_counter()-m)*1000)
                    raw=pred.anomaly_map.detach().float()
                    if kind=='tile':tilemaps.append((j,raw.cpu()[0,0].numpy().copy()))
                start=time.perf_counter()
                if kind=='tile':out=stitch_tiles(tilemaps);score=float(out.max())
                elif r['method']=='PB-ARPR':out=rect_native(raw);score=float(out.max())
                else:
                    restore=raw if kind in ['patch','fast'] else raw.cpu()
                    out=torch.nn.functional.interpolate(restore,size=(1056,4224),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
                    score=float(pred.pred_score.item())
                sync();restore_ms=(time.perf_counter()-start)*1000;e2e=(time.perf_counter()-t)*1000
                assert out.shape==(1056,4224) and np.isfinite(out).all() and np.isfinite(score)
                return dict(model_ms=sum(mts),e2e_ms=e2e,per_tile_ms=mts,reconstruction_ms=restore_ms)
            with torch.inference_mode():
                if kind=='tile':
                    # Exact pixels of native crops equal the frozen stored tile data.
                    native=read_image(warm[0],as_tensor=True)
                    for row in sorted(tiled[str(warm[0])],key=lambda z:int(z['tile_id'])):
                        i=int(row['tile_id']);assert torch.equal(native[:,:,i*1056:(i+1)*1056],read_image(row['tile_image_path'],as_tensor=True))
                    del native
                for p in warm:pipeline(p)
                gc.collect();torch.cuda.empty_cache();sync();torch.cuda.reset_peak_memory_stats()
                static_a=torch.cuda.memory_allocated()/2**20;static_r=torch.cuda.memory_reserved()/2**20
                rounds=[]
                for run in range(1,4):
                    rows=[dict(source=str(p),**pipeline(p)) for p in public]
                    rounds.append(dict(run=run,count=len(rows),model_ms=float(np.mean([x['model_ms'] for x in rows])),e2e_ms=float(np.mean([x['e2e_ms'] for x in rows])),rows=rows))
                    print('TIMED_PASS',r['method'],run,rounds[-1]['model_ms'],rounds[-1]['e2e_ms'],flush=True)
                peak_a=torch.cuda.max_memory_allocated()/2**20;peak_r=torch.cuda.max_memory_reserved()/2**20
                # Diagnostic profiler only. with_flops covers conv/matmul, not complete pipelines.
                img=read_image(warm[0],as_tensor=True)
                if kind=='tile':img=img[:,:,:1056]
                x=model.pre_processor(img[None].cuda()) if kind in ['ead','tile'] else model.pre_processor.transform(img)[None].cuda()
                try:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],with_flops=True) as prof:model.model(x);sync()
                    entries=prof.key_averages();partial=sum(e.flops or 0 for e in entries)
                    uncounted=[e.key for e in entries if not e.flops and e.key.startswith('aten::')]
                    profiling=dict(tool='torch.profiler(with_flops=True)',diagnostic_partial_flops=partial,uncounted_operators=uncounted)
                except Exception as e:profiling=dict(tool='torch.profiler',error=str(e))
            a=np.array([q['model_ms'] for q in rounds]);b=np.array([q['e2e_ms'] for q in rounds]);passes=4 if kind=='tile' else 1
            result=dict(r,reload='PASS',input_geometry=hw,forward_passes_per_native=passes,processed_pixels_per_native=passes*hw[0]*hw[1],total_params=total,trainable_params=trainable,
              model_latency_mean_ms=float(a.mean()),model_latency_std_ms=float(a.std(ddof=1)),model_fps_mean=float(np.mean(1000/a)),model_fps_from_mean_latency=1000/float(a.mean()),
              e2e_latency_mean_ms=float(b.mean()),e2e_latency_std_ms=float(b.std(ddof=1)),e2e_fps_mean=float(np.mean(1000/b)),e2e_fps_from_mean_latency=1000/float(b.mean()),
              static_allocated_mib=static_a,static_reserved_mib=static_r,peak_allocated_mib=peak_a,peak_reserved_mib=peak_r,delta_inference_allocated_mib=peak_a-static_a,
              flops=None,flop_profile_complete=False,flops_reason='Profiler counts only supported conv/matmul; pooling, reductions, interpolation, normalization and method-specific scoring are not fully counted. Full-method GFLOPs N/A.',profiling=profiling,rounds=rounds)
            if kind=='patch':
                bank=model.model.memory_bank;result.update(memory_bank_shape=list(bank.shape),memory_bank_elements=bank.numel(),memory_bank_mib=bank.numel()*bank.element_size()/2**20,memory_bank_dtype=str(bank.dtype),neighbors=9,coreset_ratio=0.1);del bank
            if kind=='tile':result.update(per_tile_model_latency_ms=float(a.mean()/4),per_tile_fps=4000/float(a.mean()),reconstruction_latency_ms=float(np.mean([x['reconstruction_ms'] for q in rounds for x in q['rows']])))
            assert sha(r['path'])==r['sha256'];results.append(result)
            filenames=['04_std256','05_square384','06_square512','07_pb_arpr','08_tile4','09_patchcore','10_fastflow'];save(filenames[idx]+'_benchmark.json',result)
            del model,x,img,pipeline
            gc.collect();torch.cuda.empty_cache();sync()
        finalize(results,env,historical,blocked)
    except Exception as e:
        traceback.print_exc();save('completion_verification.json',dict(phase='Phase21A',status='FAIL',error=str(e),completed_methods=[r['method'] for r in results],protocol_violation_count=len(blocked)))
        return 1
    return 0

def finalize(results,env,historical,blocked):
    assert len(results)==7 and not blocked
    assert all(sha(p)==h for p,h in historical.items())
    order=['EfficientAD-S','Square384','Square512','Tile4','PatchCore','FastFlow','PB-ARPR'];results.sort(key=lambda r:order.index(r['method']))
    compact=[{k:v for k,v in r.items() if k not in ['rounds','profiling']} for r in results]
    equal=[r for r in compact if r['method'] in ['Square512','Tile4','PB-ARPR']];assert all(r['processed_pixels_per_native']==262144 for r in equal)
    for n,d in [('11_parameter_counts.json',[{k:r[k] for k in ['method','total_params','trainable_params']} for r in results]),('12_flops_profile.json',[dict(method=r['method'],complete=False,flops=None,reason=r['flops_reason'],diagnostic=r['profiling']) for r in results]),('13_pixel_workload.json',[{k:r[k] for k in ['method','forward_passes_per_native','processed_pixels_per_native']} for r in results]),('14_training_build_cost.json',[dict(method=r['method'],seconds=r['training_or_build_time_sec'],measured=True) for r in results]),('15_runtime_three_repeat_summary.json',compact),('16_memory_summary.json',compact),('17_performance_efficiency_table.json',compact),('18_equal_budget_efficiency_table.json',equal)]:save(n,d)
    save('19_image_score_scope_note.json',dict(STD_Square_FastFlow='model-grid spatial max',PB_ARPR='native map spatial max',Tile4='reconstructed native map spatial max',PatchCore='official weighted NN score',comparison='method-level pipeline comparison; not identical score-grid controlled comparison'))
    save('20_protocol_audit.json',dict(historical_source_hashes=historical,historical_unchanged=True,training_started=False,metrics_recomputed=False,protocol_violation_count=0))
    completion=dict(phase='Phase21A',status='PASS',environment_match=True,gpu=env['gpu'],cuda=env['cuda'],torch_version=env['torch'],benchmark_native_count=114,benchmark_repeats=3,warmup_count=10,methods=compact,equal_budget_three_way_verified=True,historical_metrics_read_only=True,protocol_violation_count=0)
    save('completion_verification.json',completion)
    fields=['method','total_params','trainable_params','processed_pixels_per_native','model_latency_mean_ms','model_latency_std_ms','model_fps_mean','e2e_latency_mean_ms','e2e_latency_std_ms','e2e_fps_mean','peak_allocated_mib','training_or_build_time_sec']
    with (OUT/'performance_efficiency_table.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields+['image_auroc','pixel_auroc','aupro005','aupro030']);w.writeheader();w.writerows([{**{k:r[k] for k in fields},**r['metrics']} for r in compact])
    (OUT/'final_console_report.txt').write_text('PHASE21A UNIFIED EFFICIENCY BENCHMARK\nSTATUS: PASS\n'+json.dumps(completion,indent=2)+'\n',encoding='utf8')
    print('PHASE21A_PASS',flush=True)
if __name__=='__main__':sys.exit(main())
