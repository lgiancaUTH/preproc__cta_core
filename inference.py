'''
Run inference with the trained model for rough CTA brain mask segmentation.

'''

from conf import *


import numpy as np
import torch
import SimpleITK as sitk
import monai 


from dataset import CtaMaskDataset, SAMPLE_CSV_FILE

from monai.utils import set_determinism
# monai.config.print_config()
set_determinism(110)

#==== Parameters

pxHeight = 512
pxWidth = 512
voxSizeHeight = 0.5
voxSizeWidth = 0.5


WEIGHTS_FILE = "best_model_weights-exp2023-09-01 17:21:15.401309.pt"
#=========



# 2D U-Net model
model = monai.networks.nets.UNet(
    spatial_dims=2,
    in_channels=1, # input channels
    out_channels=1, # output channels 
    channels=(4, 8, 16),
    strides=(2, 2),
    num_res_units=2
).to(device)

# load weights
model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))

cta_path='/data/giancardo-group/lgiancardo/slicer_scripts/brain_mask_seg/tmp/ctaAligned_sub-0150-v2.nii.gz'


def segmentBrainMask(brainNeckSi):

    # brainNeckSi = sitk.ReadImage(cta_path)

    # get voxel size
    brainNeckSiVoxSize = brainNeckSi.GetSpacing()
    # get resolution
    brainNeckSiRes = brainNeckSi.GetSize()

    # resample to 0.5mm
    resampler = sitk.ResampleImageFilter()
    # resampler.SetInterpolator(sitk.sitkBSpline) # leave default linear interpolation for speed
    resampler.SetOutputDirection(brainNeckSi.GetDirection())
    resampler.SetOutputOrigin(brainNeckSi.GetOrigin())
    resampler.SetOutputSpacing([voxSizeHeight, voxSizeWidth, brainNeckSiVoxSize[2]])
    resampler.SetSize([pxHeight, pxWidth, brainNeckSiRes[2]])
    brainNeckResSi = resampler.Execute(brainNeckSi)


    ctaArr = sitk.GetArrayFromImage(brainNeckResSi).astype(np.float32)
    # swap axes to have [slice, 0, height, width] (and convert to pytorch tensor)
    inputs = torch.from_numpy(ctaArr[:,None,:,:]).to(device)

    # std preprocessing (clip and normalize)
    inputs = CtaMaskDataset.ctaPreprocess1(inputs)


    TH_FOR_PRED = 0.5
    sig = torch.nn.Sigmoid()

    # run model
    outputs = model(inputs)
    outputsProb = sig(outputs)

    # convert to nifti image
    outputsNp = outputsProb.cpu().detach().numpy()
    outputsNp = outputsNp[:,0,:,:]
    # threshold
    outputsNp = np.where(outputsNp > TH_FOR_PRED, 1, 0)
    # convert type
    outputsNp = outputsNp.astype(np.uint8)


    # convert to sitk image
    outputsSi = sitk.GetImageFromArray(outputsNp)
    outputsSi.CopyInformation(brainNeckResSi)

    #=== keep largest connected component with simpleITK

    # Label connected components
    connected_components = sitk.ConnectedComponent(outputsSi)


    sorted_component_image = sitk.RelabelComponent(connected_components, sortByObjectSize=True)
    outputsLargestSi = sorted_component_image == 1

    #===

    # fill holes with simpleITK
    fillFilter = sitk.BinaryFillholeImageFilter()
    fillFilter.SetForegroundValue(1)
    outputsLargestSi = fillFilter.Execute(outputsLargestSi)

    return outputsLargestSi


# # save
# sitk.WriteImage(outputsLargestSi, 'tmp/mask0150.nii.gz')
