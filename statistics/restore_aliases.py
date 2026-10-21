"""Restore only SHA-verified duplicate paths inside an extracted review package."""
from pathlib import Path
import csv,hashlib,shutil
R=Path(__file__).resolve().parents[1]
for row in csv.DictReader((R/'docs/duplicate_audit.csv').open()):
 if not row['action'].startswith('Removed'):continue
 src=R/row['canonical'];dst=R/row['alias']
 assert dst.resolve().is_relative_to(R) and src.resolve().is_relative_to(R)
 assert hashlib.sha256(src.read_bytes()).hexdigest()==row['sha256']
 if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
print('Duplicate aliases restored; no original experiment touched.')
