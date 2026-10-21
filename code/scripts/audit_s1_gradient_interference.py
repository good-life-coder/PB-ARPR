"""Phase14C-A entry point: diagnostic gradients, never parameter updates."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation import gradient_interference_audit as audit
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--precheck-only',action='store_true');args=parser.parse_args()
    try: audit.run(args.precheck_only)
    except Exception as exc:
        audit.save('failure_precheck.json' if args.precheck_only else 'failure_run.json',dict(status='FAIL',error=str(exc)))
        raise
