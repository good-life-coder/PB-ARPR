"""Context inputs and separate native core GT; never resize core GT to context."""
import json
import torch
from anomalib.data import Folder
from anomalib.data.datasets import FolderDataset
from anomalib.data.dataclasses import ImageItem
from anomalib.data.utils import read_image
from .p1_halo_builder import DERIVED,read_rows,manifest_hash


class CoreContextDataset(FolderDataset):
    def __getitem__(self,index):
        sample=self.samples.iloc[index]
        image=read_image(sample.image_path,as_tensor=True)
        if self.augmentations is not None:image=self.augmentations(image)
        # gt_mask intentionally absent: the official paired image/mask transform
        # assumes shared geometry, which 1248context and1056core do not have.
        return ImageItem(image=image,gt_label=torch.tensor(int(sample.label_index)),
                         image_path=sample.image_path,mask_path=self.core_masks[str(sample.image_path)])


class P1HaloDataModule(Folder):
    def __init__(self):
        super().__init__(name='sheet_metal_p1_halo96',root=DERIVED,normal_dir='train/images',
            normal_test_dir='test_public/images/normal',abnormal_dir='test_public/images/anomaly',
            train_batch_size=1,eval_batch_size=1,num_workers=0,test_split_mode='from_dir',val_split_mode='from_dir',seed=42)

    def _setup(self,_stage=None):
        report=json.loads((DERIVED/'dataset_report.json').read_text())
        assert report['status']=='PASS' and report['manifest_sha256']==manifest_hash()
        rows=read_rows();mapping={a['context_path']:a for a in rows}
        self.train_data=FolderDataset(name=self.name,root=DERIVED,normal_dir='train/images',split='train')
        self.val_data=CoreContextDataset(name=self.name,root=DERIVED,normal_dir='validation/images',split='train')
        self.test_data=CoreContextDataset(name=self.name,root=DERIVED,normal_dir='train/images',
            normal_test_dir='test_public/images/normal',abnormal_dir='test_public/images/anomaly',split='test')
        for data in (self.train_data,self.val_data,self.test_data):
            data.core_masks={a['context_path']:a['core_mask_path'] for a in rows}
            for item in data.samples.itertuples():assert int(item.label_index)==int(mapping[str(item.image_path)]['tile_is_anomalous'])
        assert (len(self.train_data),len(self.val_data),len(self.test_data))==(548,76,456)
