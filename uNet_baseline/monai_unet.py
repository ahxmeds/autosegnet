#%%
from email.header import make_header
import sys
import logging
from monai.utils import set_determinism
from monai.engines import EnsembleEvaluator
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
    MeanEnsembled,
    EnsureChannelFirstd,
    Invertd,
    AsDiscreted,
    SaveImaged,
    Activationsd
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.inferers import SlidingWindowInferer
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader, Dataset, SmartCacheDataset
import torch
# import pytorch_lightning
import time

import os
import glob
import numpy as np

import nibabel as nib


def get_base_model():    
    model = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )
    return model

def get_val_transforms():
    mod_keys = ['CT', 'PT']
    val_transforms = Compose(
        [
            LoadImaged(keys=mod_keys),
            EnsureChannelFirstd(keys=mod_keys),
            ScaleIntensityRanged(keys=['CT'], a_min=-1024, a_max=1024, b_min=0, b_max=1, clip=True),
            # ScaleIntensityd(keys=['PT'], minv=0, maxv=1),
            CropForegroundd(keys=mod_keys, source_key='CT'),
            Orientationd(keys=mod_keys, axcodes="RAS"),
            Spacingd(keys=mod_keys, pixdim=(2,2,2), mode=('bilinear', 'bilinear')),
            ConcatItemsd(keys=mod_keys, name='image', dim=0),
        ]
    )
    return val_transforms 

def prepare_data(data_dir, val_transforms):
    # set up the correct data path
    images_pt = sorted(glob.glob(os.path.join(data_dir, "SUV*")))
    images_ct = sorted(glob.glob(os.path.join(data_dir, "CTres*")))
    
    data_dicts = [
        {'CT': image_name_ct, 'PT': image_name_pt}
        for image_name_ct, image_name_pt,  in zip(images_ct, images_pt)
    ]
    val_files = data_dicts
    val_ds = Dataset(data=val_files, transform=val_transforms)
    return val_ds

def get_val_dataloader(val_ds):
    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=0, collate_fn = list_data_collate)
    return val_loader

def get_post_transforms(val_transforms, export_dir):
    pred_keys_to_use = ['pred0', 'pred1', 'pred2', 'pred3', 'pred4']
    mean_post_transforms = Compose(
    [
        EnsureTyped(keys=pred_keys_to_use),
        MeanEnsembled(
            keys=pred_keys_to_use,
            output_key="pred",
            # weights=[1.0,1.0,1.0,1.0,1.0],
            weights=[0.6358150243759155, 0.6278120875358582, 0.6414686441421509, 0.6304689049720764, 0.6296911239624023],
        ),
        Activationsd(keys="pred", sigmoid=True),
        Invertd(
        keys=["pred"],
        transform=val_transforms,
        orig_keys="PT",
        meta_keys=["pred_meta_dict"],
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, threshold=0.5),
        EnsureTyped(keys='pred', dtype=np.uint8),
        SaveImaged(keys="pred", meta_keys="image_meta_dict", output_dir=export_dir, output_postfix="", output_ext=".nii.gz", separate_folder=False, resample=True, output_dtype=np.uint8)
    ]
    )
    return mean_post_transforms
def ensemble_evaluate(post_transforms, models, device, val_dataloader):
    evaluator = EnsembleEvaluator(
        device=device,
        val_data_loader=val_dataloader,
        # pred_keys=["pred0", "pred1", "pred2", "pred3", "pred4"],
        pred_keys=['pred0', 'pred1', 'pred2', 'pred3', 'pred4'],
        networks=models,
        inferer=SlidingWindowInferer(roi_size=(192,192, 192), sw_batch_size=4, overlap=0.5),
        postprocessing=post_transforms,
    )
    evaluator.run()
    return evaluator


#%%
def segment_PETCT(
    ckpt_path0,
    ckpt_path1,
    ckpt_path2,
    ckpt_path3,
    ckpt_path4, 
    data_dir, 
    export_dir
):
    print("starting")
    device = torch.device("cuda:0")
    chkp_paths = [ckpt_path0, ckpt_path1, ckpt_path2, ckpt_path3, ckpt_path4]
    models = [get_base_model() for _ in range(5)]
    for i in range(5):
        models[i].load_state_dict(torch.load(chkp_paths[i], map_location=device))
        models[i].to(device)
        models[i].eval()
    
    val_transforms = get_val_transforms()
    val_ds = prepare_data(data_dir, val_transforms)
    val_dataloader = get_val_dataloader(val_ds)
    post_transforms = get_post_transforms(val_transforms, export_dir)
    ensemble_evaluate(post_transforms, models, device, val_dataloader)
   
    # data = next(iter(val_dataloader))
    file = os.listdir(export_dir)[0]
    old_path = os.path.join(export_dir, file)
    # old_path = os.path.join(export_dir,  os.path.basename(data[0]['PT_meta_dict']['filename_or_obj']))
    new_path = os.path.join(export_dir, 'PRED.nii.gz')
    os.rename(old_path, new_path)
    

def run_inference(
    ckpt_path0='/opt/algorithm/fold0_model_ep=0262.pth', 
    ckpt_path1='/opt/algorithm/fold1_model_ep=0238.pth', 
    ckpt_path2='/opt/algorithm/fold2_model_ep=0236.pth', 
    ckpt_path3='/opt/algorithm/fold3_model_ep=0230.pth', 
    ckpt_path4='/opt/algorithm/fold4_model_ep=0202.pth', 
    data_dir='/opt/algorithm/', 
    export_dir='/output/images/automated-petct-lesion-segmentation/'
):
    segment_PETCT(
        ckpt_path0,
        ckpt_path1,
        ckpt_path2,
        ckpt_path3,
        ckpt_path4, 
        data_dir, 
        export_dir
    )


if __name__ == '__main__':
    run_inference()

