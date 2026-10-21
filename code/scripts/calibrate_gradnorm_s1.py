"""Only Phase14C-B precheck or bounded calibration; no formal training option."""
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from src.evaluation import gradnorm_s1_calibration as cal


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--precheck-only',action='store_true')
    parser.add_argument('--recover-runtime-only',action='store_true')
    args=parser.parse_args()
    result=cal.precheck() if args.precheck_only else cal.recover_runtime_only() if args.recover_runtime_only else cal.run()
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
