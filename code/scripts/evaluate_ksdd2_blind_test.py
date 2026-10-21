"""Phase15E: one authorized blind pass, no training or tuning entry point."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import msvcrt
import torch
from src.evaluation import ksdd2_blind_test as b

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--precheck-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);b.OUT.mkdir(parents=True,exist_ok=True)
    with (b.OUT/'execution.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        a,rows,binding=b.precheck()
        if not args.precheck_only:return b.run(a,rows,binding)

if __name__=='__main__':main()
