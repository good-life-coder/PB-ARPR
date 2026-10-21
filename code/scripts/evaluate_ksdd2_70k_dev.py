"""Development-only evaluation; both own-stage70000 final checkpoints required."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from src.evaluation import ksdd2_70k_protocol as p
from src.evaluation.ksdd2_70k_dev_evaluation import run

if __name__=='__main__':
    torch.set_num_threads(4);a=p.assets();guard=p.base.AccessGuard(a);guard.install_test_block()
    reports={n:p.base.read(p.OUT/p.FOLDERS[n]/'training_report.json') for n in p.NAMES}
    p.all_ready(reports);run(a,guard,reports)
