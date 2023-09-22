from email.header import make_header
import sys
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    ConcatItemsd,
    RandAffined,
    ToTensord,
    DeleteItemsd,
    EnsureChannelFirstd
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, compute_meandice
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader, Dataset, SmartCacheDataset
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import pytorch_lightning
import time

import os
import glob
import numpy as np

import nibabel as nib


class Net(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = UNet(
            dimensions=3,
            in_channels=2,
            out_channels=2,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        # self.loss_function = DiceLoss(to_onehot_y=True,softmax=True,include_background=False,batch=True)
        # self.post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
        # self.post_label = AsDiscrete(to_onehot=True, n_classes=2)
        # self.best_val_dice = 0
        # self.best_val_epoch = 0

    def forward(self, x):
        return self._model(x)

    def prepare_data(self, data_dir):
        # set up the correct data path
        images_pt = sorted(glob.glob(os.path.join(data_dir, "SUV*")))
        images_ct = sorted(glob.glob(os.path.join(data_dir, "CTres*")))
       
        data_dicts = [
            {'PT': image_name_pt, 'CT': image_name_ct}
            for image_name_pt, image_name_ct in zip(images_pt, images_ct)
        ]
        val_files = data_dicts
        mod_keys = ['CT', 'PT']
        spacing = (2,2,2)
        val_transforms = Compose(
            [
                LoadImaged(keys=mod_keys),
                EnsureChannelFirstd(keys=mod_keys),
                ScaleIntensityRanged(keys=['CT'], a_min=-1024, a_max=1024, b_min=0, b_max=1, clip=True),
                # ScaleIntensityd(keys=['PT'], minv=0, maxv=1),
                CropForegroundd(keys=mod_keys, source_key='CT'),
                Orientationd(keys=mod_keys, axcodes="RAS"),
                Spacingd(keys=mod_keys, pixdim=spacing, mode=('bilinear', 'bilinear', 'nearest')),
                ConcatItemsd(keys=['CT', 'PT'], name='CTPT', dim=0),
                DeleteItemsd(keys=['CT', 'PT'])
            ]
        )

        self.val_ds = Dataset(data=val_files, transform=val_transforms)

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds, batch_size=1, num_workers=0, collate_fn = list_data_collate)
        return val_loader


def segment_PETCT(ckpt_path, data_dir, export_dir):
    print("starting")

    net = Net.load_from_checkpoint(ckpt_path)
    net.eval()

    device = torch.device("cuda:0")
    net.to(device)
    net.prepare_data(data_dir)

    with torch.no_grad():
        for i, val_data in enumerate(net.val_dataloader()):
            roi_size = (192, 192, 192)
            sw_batch_size = 4
            
            mask_out = sliding_window_inference(val_data["CTPT"].to(device), roi_size, sw_batch_size, net)
            mask_out = torch.argmax(mask_out, dim=1).detach().cpu().numpy().squeeze()
            mask_out = mask_out.astype(np.uint8)               
            print("done inference")

            
            PT = nib.load(os.path.join(data_dir,"SUV.nii.gz"))  #needs to be loaded to recover nifti header and export mask
            pet_affine = PT.affine
            PT = PT.get_fdata()
            mask_export = nib.Nifti1Image(mask_out, pet_affine)
            print(os.path.join(export_dir, "PRED.nii.gz"))

            nib.save(mask_export, os.path.join(export_dir, "PRED.nii.gz"))
            print("done writing")


def run_inference(ckpt_path='/opt/algorithm/model_ep=0196.pth', data_dir='/opt/algorithm/', export_dir='/output/images/automated-petct-lesion-segmentation/'):
    segment_PETCT(ckpt_path, data_dir, export_dir)


if __name__ == '__main__':
    run_inference()

