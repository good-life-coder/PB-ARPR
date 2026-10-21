"""Official Folder loader with explicit validation split; no sampling from test."""
import csv
import json
from anomalib.data import Folder
from anomalib.data.datasets import FolderDataset
from .tile4_builder import DERIVED,manifest_hash


class Tile4DataModule(Folder):
    def __init__(self):
        super().__init__(name="sheet_metal_tile4",root=DERIVED,normal_dir="train/images",
                         normal_test_dir="test_public/images/normal",abnormal_dir="test_public/images/anomaly",
                         mask_dir="test_public/masks/anomaly",train_batch_size=1,eval_batch_size=1,num_workers=0,
                         test_split_mode="from_dir",val_split_mode="from_dir",seed=42)

    def _setup(self,_stage=None):
        report=json.loads((DERIVED/"dataset_report.json").read_text(encoding="utf-8"))
        if report["status"]!="PASS" or report["manifest_sha256"]!=manifest_hash():
            raise ValueError("A validated and unchanged derived manifest is required")
        super()._setup(_stage)
        self.val_data=FolderDataset(name=self.name,root=DERIVED,normal_dir="validation/images",split="train")
        if (len(self.train_data),len(self.val_data),len(self.test_data))!=(548,76,456):
            raise ValueError("Tile loader changed fixed split counts")
        with (DERIVED/"manifest.csv").open(encoding="utf-8",newline="") as f:rows=list(csv.DictReader(f))
        expected={row["tile_image_path"]:row for row in rows}
        for data in (self.train_data,self.val_data,self.test_data):
            for sample in data.samples.itertuples():
                row=expected[str(sample.image_path)]
                if int(sample.label_index)!=int(row["tile_is_anomalous"]):raise ValueError("Folder tile label mismatch")
                if int(sample.label_index) and str(sample.mask_path)!=row["tile_mask_path"]:
                    raise ValueError("Folder tile mask mapping mismatch")
