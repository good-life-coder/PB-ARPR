"""Phase24C-R1 exact-pixel architecture audit (no dataset access)."""
import csv, json, math, os, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'experiments/phase24c_multibudget_geometry_sensitivity'
OUT=ROOT/'experiments/phase24c_r1_architecture_constrained_budget'
DATA=ROOT/'datasets'
def guard(event,args):
    if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes)):
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(DATA):raise PermissionError('Dataset access forbidden during selection: '+str(p))
sys.addaudithook(guard)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(name,obj):
    p=OUT/name
    if p.exists():
        p.unlink()
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def candidates(n):
    return [[h,n//h] for h in range(96,2049,32) if n%h==0 and 96<=n//h<=2048 and (n//h)%32==0 and h<=n//h]
def probe(hw,torch,EfficientAd):
    h,w=hw;torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.cuda.empty_cache()
    r=dict(geometry=hw,pixels=h*w,seed=42,batch_size=1,precision='FP32',synthetic_input='all_zero',
           feasible=False,failure_stage=None,exception=None,teacher_shape=None,student_shape=None,
           ae_shape=None,combined_map_shape=None,nan_count=0,inf_count=0)
    model=None
    try:
        r['failure_stage']='INITIALIZATION'
        pre=EfficientAd.configure_pre_processor(image_size=(h,w))
        model=EfficientAd(model_size='small',imagenet_dir='./datasets/imagenette',pre_processor=pre,
                          evaluator=False,post_processor=False,visualizer=False)
        assert sum(p.numel() for p in model.parameters())==8058628
        assert not model.model.ae.decoder.padding and model.model.pad_maps
        model.model.eval().to('cuda:0')
        x=torch.zeros((1,3,h,w),dtype=torch.float32,device='cuda:0')
        with torch.inference_mode():
            r['failure_stage']='TEACHER';t=model.model.teacher(x);r['teacher_shape']=list(t.shape)
            r['failure_stage']='STUDENT';s=model.model.student(x);r['student_shape']=list(s.shape)
            r['failure_stage']='AE';a=model.model.ae(x,(h,w));r['ae_shape']=list(a.shape)
            r['failure_stage']='BRANCH_COMPATIBILITY'
            assert t.shape[0]==s.shape[0]==a.shape[0]==1
            assert t.shape[1]==a.shape[1]==384 and s.shape[1]==768
            assert t.shape[-2:]==s.shape[-2:]==a.shape[-2:]
            r['failure_stage']='COMBINED_MAP';pred=model.model(x);m=pred.anomaly_map
            r['combined_map_shape']=list(m.shape);assert list(m.shape)==[1,1,h,w]
            r['failure_stage']='FINITE_VALUES'
            values=(t,s,a,m,pred.pred_score)
            r['nan_count']=sum(int(torch.isnan(v).sum().item()) for v in values)
            r['inf_count']=sum(int(torch.isinf(v).sum().item()) for v in values)
            assert r['nan_count']==r['inf_count']==0
        r['failure_stage']=None;r['feasible']=True
    except Exception as e:
        r['exception']=type(e).__name__+': '+str(e);r['traceback']=traceback.format_exc()
    finally:
        del model;torch.cuda.empty_cache()
    return r
def select(rows):
    for r in rows:
        h,w=r['geometry'];r['aspect_deviation']=abs(math.log((h/w)/0.25))
    ok=[r for r in rows if r['feasible'] and r['geometry'][0]!=r['geometry'][1]]
    ordered=sorted(ok,key=lambda r:(round(r['aspect_deviation']/1e-9),
                     abs(r['geometry'][0]/r['geometry'][1]-0.25),-r['geometry'][0],r['geometry'][1],tuple(r['geometry'])))
    best=ordered[0] if ordered else None
    return dict(exact_pixel_nonsquare_feasible=bool(best),selected_geometry=best['geometry'] if best else None,
                selected_score=best['aspect_deviation'] if best else None,selection_used_test_metrics=False,
                selection_reason='minimum log aspect-ratio deviation from 0.25 among feasible non-square candidates' if best else 'No architecture-feasible non-square geometry under strict equal-pixel constraint on the frozen legal grid',
                training_skipped=best is None)
def main():
    t0=time.perf_counter();prior=read(OLD/'completion_verification.json')
    report=(OLD/'final_console_report.txt').read_text(encoding='utf8')
    checks=dict(status=prior['status']=='PASS_MULTIBUDGET_GEOMETRY_SENSITIVITY_COMPLETE',
                low_infeasible=prior['low_budget_strict_rect_feasible'] is False,
                mid_infeasible=prior['mid_budget_strict_rect_feasible'] is False,
                test_during_training=prior['test_accessed_during_training'] is False,
                violations=prior['protocol_violation_count']==0,
                report_status='STATUS = PASS_MULTIBUDGET_GEOMETRY_SENSITIVITY_COMPLETE' in report,
                report_ae_failure='AE enconv6 8x8 kernel' in report)
    if not all(checks.values()):print('FAIL_PREVIOUS_PHASE_GATE',checks,flush=True);return 1
    OUT.mkdir(parents=True,exist_ok=True)
    save('01_previous_phase_gate.json',dict(status='PASS',checks=checks,
         source_report=str(OLD/'final_console_report.txt'),
         source_verification=str(OLD/'completion_verification.json')))
    import torch
    from anomalib.models import EfficientAd
    torch.set_num_threads(4);assert torch.cuda.is_available()
    allrows=[]
    for budget,n,enum,feas,selection in (
        ('low',65536,'02_low_candidate_enumeration.json','03_low_feasibility_results.json','04_low_selection.json'),
        ('mid',147456,'05_mid_candidate_enumeration.json','06_mid_feasibility_results.json','07_mid_selection.json')):
        cs=candidates(n)
        save(enum,dict(budget=budget,pixels=n,reference_ratio=0.25,dimension_multiple=32,
                       dimension_bounds=[96,2048],h_le_w=True,candidates=cs,count=len(cs)))
        print(budget.upper(),'CANDIDATES',cs,flush=True)
        rows=[]
        for hw in cs:
            r=probe(hw,torch,EfficientAd);rows.append(r)
            print(budget.upper(),hw,'FEASIBLE' if r['feasible'] else 'INFEASIBLE',
                  r['failure_stage'],r['exception'],flush=True)
        decision=select(rows)
        save(feas,dict(budget=budget,pixels=n,status='COMPLETE',results=rows))
        save(selection,dict(budget=budget,pixels=n,**decision))
        allrows.extend(dict(budget=budget,**r) for r in rows)
        print(budget.upper(),'SELECTED',decision['selected_geometry'],flush=True)
    p=OUT/'candidate_feasibility.csv'
    if p.exists():p.unlink()
    with p.open('w',newline='',encoding='utf8') as f:
        fields=['budget','height','width','pixels','aspect_ratio','aspect_deviation','feasible',
                'failure_stage','exception','teacher_shape','student_shape','ae_shape',
                'combined_map_shape','nan_count','inf_count']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in allrows:
            h,w=r['geometry']
            writer.writerow(dict(budget=r['budget'],height=h,width=w,pixels=h*w,aspect_ratio=h/w,
                  aspect_deviation=r['aspect_deviation'],feasible=r['feasible'],
                  failure_stage=r['failure_stage'],exception=r['exception'],
                  teacher_shape=r['teacher_shape'],student_shape=r['student_shape'],
                  ae_shape=r['ae_shape'],combined_map_shape=r['combined_map_shape'],
                  nan_count=r['nan_count'],inf_count=r['inf_count']))
    print('AUDIT_ELAPSED_SEC',time.perf_counter()-t0,flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())

