"""Evaluate development only after all eight repeat finals have passed."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from src.evaluation import ksdd2_repeat_summary as p
from src.training import ksdd2_repeat_training as t

def main():
    torch.set_num_threads(4);a=p.parent.assets();guard=p.base.AccessGuard(a);guard.install_test_block()
    reports={p.key(s,n):p.base.read(p.folder(s,n)/'training_report.json') for s,n in p.RUNS}
    p.all_ready(reports);return p.run(a,guard,reports,t.references())

if __name__=='__main__':main()
