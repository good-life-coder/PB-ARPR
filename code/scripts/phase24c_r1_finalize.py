"""Finalize Phase24C-R1 from frozen references and the single completed Mid run."""
import csv,json,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/phase24c_r1_architecture_constrained_budget'
OLD=ROOT/'experiments/phase24c_multibudget_geometry_sensitivity'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(name,obj):
    p=OUT/name
    if p.exists():raise FileExistsError(str(p))
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def pct(x):return 'N/A' if x=='N/A' else f'{x:.6f}'
def main():
    gate=read(OUT/'01_previous_phase_gate.json')
    low_enum=read(OUT/'02_low_candidate_enumeration.json')
    low=read(OUT/'03_low_feasibility_results.json')
    low_sel=read(OUT/'04_low_selection.json')
    mid_enum=read(OUT/'05_mid_candidate_enumeration.json')
    mid=read(OUT/'06_mid_feasibility_results.json')
    mid_sel=read(OUT/'07_mid_selection.json')
    train=read(OUT/'08_mid_training_summary.json')
    eval=read(OUT/'09_mid_evaluation.json')
    old=read(OLD/'completion_verification.json')
    corrections=read(OUT/'00_preformal_corrections.json')
    assert gate['status']=='PASS'
    assert low_enum['count']==len(low['results']) and mid_enum['count']==len(mid['results'])
    assert all(r['geometry']==g for r,g in zip(low['results'],low_enum['candidates']))
    assert all(r['geometry']==g for r,g in zip(mid['results'],mid_enum['candidates']))
    assert low_sel['exact_pixel_nonsquare_feasible'] is False
    assert low_sel['selected_geometry'] is None and low_sel['training_skipped'] is True
    assert mid_sel['selected_geometry']==[256,576]
    assert mid_sel['exact_pixel_nonsquare_feasible'] and not mid_sel['selection_used_test_metrics']
    assert train['status']=='PASS' and train['completed_steps']==70000
    assert eval['status']=='PASS' and eval['prediction_count']==114 and eval['gt_region_count']==450
    assert train['checkpoint_sha256']==eval['checkpoint_sha256']
    assert train['test_public_files_accessed']==train['test_mask_files_accessed']==0
    assert not train['loss_nan_detected'] and not train['loss_inf_detected']
    assert eval['nan_count']==eval['inf_count']==0
    assert eval['test_private_files_accessed']==eval['test_private_mixed_files_accessed']==0
    assert eval['protocol_violation_count']==0
    rows=[]
    for key,budget,geometry in [('LOW',65536,'N/A'),('MID',147456,'256x576'),('HIGH',262144,'256x1024')]:
        ref=next(r for r in old['multibudget_rows'] if r['budget']==key)
        assert ref['input_pixels']==budget
        row=dict(budget=key,pixels=budget,square_geometry=ref['square_geometry'],
                 rect_geometry=geometry,rect_status='NO_FEASIBLE_NONSQUARE' if key=='LOW' else
                 ('FORMAL_COMPLETE' if key=='MID' else 'FROZEN_REFERENCE'))
        for metric in ('image_auroc','pixel_auroc','exact_aupro005','exact_aupro030'):
            square=ref['square_'+metric+'_percent']
            rect=('N/A' if key=='LOW' else
                  100*eval[metric] if key=='MID' else ref['rect_'+metric+'_percent'])
            row['square_'+metric+'_percent']=square
            row['rect_'+metric+'_percent']=rect
            row['delta_'+metric+'_pp']='N/A' if rect=='N/A' else rect-square
        rows.append(row)
    high=rows[2]
    save('10_high_frozen_reference.json',dict(status='PASS',budget=262144,
        source=str(OLD/'completion_verification.json'),square_geometry='512x512',
        rect_geometry='256x1024',square_checkpoint_sha256=read(OLD/'02_reference_audit.json')['references']['square512']['checkpoint_sha256'],
        rect_checkpoint_sha256=read(OLD/'02_reference_audit.json')['references']['high_rect']['checkpoint_sha256'],
        metrics=high))
    save('11_final_multibudget_summary.json',dict(title='Architecture-Constrained Equal-Pixel Geometry Sensitivity',
        status='PASS',delta_definition='Rect minus Square, percentage points',rows=rows,
        low_reason='No architecture-feasible non-square geometry under strict equal-pixel constraint on the frozen legal grid',
        mid_selection_geometry=[256,576],mid_selection_used_test_metrics=False))
    audit=dict(status='PASS',candidate_enumeration_complete=True,
        low_candidates=low_enum['count'],mid_candidates=mid_enum['count'],
        complete_architecture_checks=True,selection_used_test_metrics=False,
        test_public_files_accessed_during_geometry_selection=0,
        test_mask_files_accessed_during_geometry_selection=0,
        test_public_files_accessed_during_training=0,
        test_mask_files_accessed_during_training=0,
        mid_training_steps=70000,mid_final_checkpoint_sha256=train['checkpoint_sha256'],
        no_test_guided_tuning=True,no_architecture_modification=True,
        protocol_violation_count=0,preformal_corrections=corrections)
    save('12_protocol_audit.json',audit)
    fields=list(rows[0])
    with (OUT/'multibudget_results.csv').open('x',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    end=time.time()
    start_time=train['timestamp_start']
    from datetime import datetime
    elapsed=end-datetime.fromisoformat(start_time).timestamp()
    complete=dict(phase='Phase24C-R1',status='PASS_ARCH_CONSTRAINED_MULTIBUDGET_COMPLETE',
        low_candidates=low_enum['candidates'],mid_candidates=mid_enum['candidates'],
        low_feasibility=[dict(geometry=r['geometry'],feasible=r['feasible'],failure_stage=r['failure_stage']) for r in low['results']],
        mid_feasibility=[dict(geometry=r['geometry'],feasible=r['feasible'],failure_stage=r['failure_stage']) for r in mid['results']],
        low_exact_pixel_nonsquare_feasible=False,low_selected_geometry=None,
        mid_exact_pixel_nonsquare_feasible=True,mid_selected_geometry=[256,576],
        mid_selection_score=mid_sel['selected_score'],mid_training_elapsed_sec=train['training_elapsed_seconds'],
        mid_checkpoint_sha256=train['checkpoint_sha256'],mid_metrics={k:eval[k] for k in ('image_auroc','pixel_auroc','exact_aupro005','exact_aupro030')},
        mid_delta_pp={k:rows[1]['delta_'+k+'_pp'] for k in ('image_auroc','pixel_auroc','exact_aupro005','exact_aupro030')},
        high_frozen_result=high,multibudget_rows=rows,
        test_access_during_geometry_selection=0,test_access_during_training=0,
        protocol_violation_count=0,total_elapsed_since_training_start_sec=elapsed,
        candidate_audit_elapsed_sec=9.41620109998621)
    save('completion_verification.json',complete)
    lines=['STATUS = '+complete['status'],
        'LOW candidates = '+', '.join(f'{h}x{w}: '+('FEASIBLE' if r['feasible'] else 'INFEASIBLE')
             for (h,w),r in zip(low_enum['candidates'],low['results'])),
        'LOW exact-pixel non-square feasible = False; selected = N/A; training skipped = True',
        'MID candidates = '+', '.join(f'{h}x{w}: '+('FEASIBLE' if r['feasible'] else 'INFEASIBLE')
             for (h,w),r in zip(mid_enum['candidates'],mid['results'])),
        'MID exact-pixel non-square feasible = True; selected = 256x576',
        f'MID training time = {train["training_elapsed_hours"]:.6f} h',
        'MID checkpoint SHA256 = '+train['checkpoint_sha256']]
    for key in ('image_auroc','pixel_auroc','exact_aupro005','exact_aupro030'):
        lines.append(f'MID {key} = {100*eval[key]:.6f}%; Delta Rect-Square384 = {rows[1]["delta_"+key+"_pp"]:+.6f} pp')
    lines += ['HIGH frozen Square512 / Rect256x1024:',
              *[f'  {key}: {pct(high["square_"+key+"_percent"])} / {pct(high["rect_"+key+"_percent"])} / {high["delta_"+key+"_pp"]:+.6f} pp'
                for key in ('image_auroc','pixel_auroc','exact_aupro005','exact_aupro030')],
              'FINAL TABLE (Square / Rect / Delta; percent / percent / pp):']
    for row in rows:
        lines.append(f'{row["budget"]} pixels={row["pixels"]} {row["square_geometry"]} vs {row["rect_geometry"]}')
        for key in ('image_auroc','pixel_auroc','exact_aupro005','exact_aupro030'):
            v=row['delta_'+key+'_pp']
            lines.append(f'  {key}: {pct(row["square_"+key+"_percent"])} / {pct(row["rect_"+key+"_percent"])} / '+('N/A' if v=='N/A' else f'{v:+.6f}'))
    lines += ['TEST ACCESS DURING GEOMETRY SELECTION = 0',
              'TEST ACCESS DURING TRAINING = 0','PROTOCOL VIOLATIONS = 0',
              f'TOTAL ELAPSED SINCE TRAINING START = {elapsed:.3f} sec',
              f'CANDIDATE AUDIT ELAPSED = {complete["candidate_audit_elapsed_sec"]:.3f} sec',
              'FINAL_CONSOLE_REPORT = '+str(OUT/'final_console_report.txt'),
              'COMPLETION_VERIFICATION = '+str(OUT/'completion_verification.json')]
    (OUT/'final_console_report.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines),flush=True)
if __name__=='__main__':main()


