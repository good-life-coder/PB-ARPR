"""Tiny synthetic exact-AUPRO checks; not a reproduction of paper metrics."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
from src.evaluation.exact_aupro import integrate
for scores,expected in [(np.array([2.],np.float32),{.05:1.,.30:1.}), (np.array([-1.],np.float32),{.05:0.,.30:0.}), (np.array([1.],np.float32),{.05:.025,.30:.15})]:
 result,total=integrate(np.array([1.,1.],np.float32),scores,np.array([1.]),np)
 assert total==1.
 for alpha,value in expected.items():assert abs(result[alpha]-value)<1e-12,(result,expected)
print('PASS_SYNTHETIC_EXACT_AUPRO: perfect, inverse and all-tied scores at both integration limits')
