"""Freeze R1 architecture geometry without reading any dataset image."""
import json
import math
from pathlib import Path
from src.common import architecture_constrained_geometry as g
from src.evaluation import severstal_protocol as original

OUT=original.ROOT/'experiments/phase16a_r1_severstal_geometry_revision'
SPLIT_SHA='e38c69fa5287a619472b0afb97e5f33e32a95503369e065c096ffc1b44124594'

def run():
    OUT.mkdir(parents=True,exist_ok=True);a=original.assets()
    if a['protocol']['canonical_split_sha256']!=SPLIT_SHA:raise ValueError('FAIL_SPLIT')
    previous=original.read(original.PILOT/'phase16ab_report.json')
    if previous['failure_code']!='FAIL_RECT_AE_GEOMETRY' or previous['DEV_ANOMALY_FORWARD_COUNT'] or previous['FINAL_BLIND_FORWARD_COUNT']:raise ValueError('FAIL_PRIOR_RECORD')
    if original.digest(original.ROOT/'src/evaluation/ksdd2_external_protocol.py')!=a['geometry']['source_file_sha']:raise ValueError('FAIL_OLD_SELECTOR_SHA')
    if g.old_select(.16)!=(192,1280):raise ValueError('FAIL_RAW_GEOMETRY_REGRESSION')
    if (OUT/'geometry_protocol_r1.json').exists():
        p=original.read(OUT/'geometry_protocol_r1.json')
        if original.digest(OUT/'geometry_protocol_r1.json')!=original.read(OUT/'geometry_binding.json')['GEOMETRY_R1_SHA256']:raise ValueError('FAIL_GEOMETRY_SHA')
        if original.digest(OUT/'geometry_candidate_inventory.csv')!=p['inventory_sha']:raise ValueError('FAIL_INVENTORY_SHA')
        return p
    records=[];c=g.frozen_config()
    def check(h,w):
        r=g.feasibility_record(h,w);r.update(aspect_ratio=h/w,ratio_error=abs(math.log((h/w)/.16)),anisotropy=max((h/w)/.16,.16/(h/w)),pixel_budget_deviation=(h*w-c['target_pixels'])/c['target_pixels'])
        records.append(r)
        with (OUT/'feasibility_journal.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
        if len(records)%25==0:print(f'FULL ARCHITECTURE {len(records)} candidates; feasible={sum(v["feasible"] for v in records)}',flush=True)
        return r
    # KSDD2 retains its already-feasible raw result. New ranking is fallback only.
    ratio=original.read(original.ROOT/'experiments/phase15a_ksdd2_audit/dataset_audit.json')['aspect_hw']['median']
    ksdd,ksrecords=g.select(ratio,g.feasibility_record)
    if ksdd!=(864,320):raise ValueError('FAIL_KSDD2_GEOMETRY_REGRESSION')
    mvtec=g.feasibility_record(256,1024)
    if not mvtec['feasible']:raise ValueError('FAIL_MVTEC_GEOMETRY')
    original.save(OUT/'regression_report.json',dict(KSDD2=dict(selected=list(ksdd),records=list(ksrecords.values())),MVTec=mvtec))
    hw,inventory=g.select(.16,check)
    original.csv_save(OUT/'geometry_candidate_inventory.csv',records)
    selected=inventory[hw];raw=inventory[(192,1280)]
    if raw['feasible'] or raw['failure_stage']!='AE':raise ValueError('FAIL_RAW_FEASIBILITY_REGRESSION')
    p=dict(FROZEN=True,architecture_constrained=True,native_hw=[256,1600],raw_candidate_hw=[192,1280],raw_candidate_feasible=False,
        failure='AE geometry',selected_hw=list(hw),selected_pixels=math.prod(hw),pixel_budget=c,ratio_error=selected['ratio_error'],anisotropy=selected['anisotropy'],
        architecture_feasibility=True,selected_forward=selected,selection_used_model_performance=False,dev_used_for_selection=False,blind_used_for_selection=False,
        split_sha=SPLIT_SHA,old_geometry_sha=a['protocol']['geometry_sha256'],original_protocol_status='FAIL_RECT_AE_GEOMETRY',
        R1_CHANGE_REASON='architecture feasibility only',METHOD_DEFINITION_UPDATED=True,selection_rule='retain feasible raw; else minimum ratio error, ties<=1e-9 budget gap, larger pixels, H/W ascending',
        candidate_count=len(records),feasible_count=sum(r['feasible'] for r in records),inventory_sha=original.digest(OUT/'geometry_candidate_inventory.csv'),
        size_boundaries=a['protocol']['size_boundaries'],FINAL_BLIND_FORWARD_COUNT=0,DEV_ANOMALY_FORWARD_COUNT=0)
    original.save(OUT/'geometry_protocol_r1.json',p);original.save(OUT/'geometry_binding.json',dict(GEOMETRY_R1_SHA256=original.digest(OUT/'geometry_protocol_r1.json')))
    print(json.dumps(p,indent=2),flush=True);return p
