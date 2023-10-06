import pandas as pd
import numpy as np
import torch
import torch.utils.data as tData

import SimpleITK as sitk

import os
import glob

BASE_DIR = '/data/giancardo-group/data/stroke-ct-cta-052023/'
SAMPLE_CSV_FILE = '/data/giancardo-group/lgiancardo/slicer_scripts/brain_mask_seg/sampleLst.csv'

class CtaMaskDataset(tData.Dataset):
    '''
    Class to load the CTA images and the brain mask images
    '''
    def __init__(self, samplesFile, transform=None, slicesPerSubj=30, seed=1234, useCache=False):
        '''
        Args:
            samplesFile (string): Path to the csv file with the samples. Generated with createAndSaveStaticSampleLst
            slicesPerSubj: Number of slices to extract per subject (default 30). Set to -1 to extract all slices
            transform (callable, optional): Optional transform to be applied
                on a sample.
        '''
        self._gtFr = pd.read_csv(samplesFile) 
        self.transform = transform
        self._slicesPerSubj = slicesPerSubj
        self._useCache = useCache
        self._cache = {} 
        if self._useCache:
            for i in range(len(self._gtFr)):
                self._cache[i] = {'brainNeckSi': None, 'maskSiRes': None}

    def __len__(self):
        return len(self._gtFr)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cta_path = self._gtFr.iloc[idx]['cta']
        mask_path = self._gtFr.iloc[idx]['mask']

        # load images
        if (self._useCache and (self._cache[idx]['brainNeckSi'] is None)) or (not self._useCache):
            brainNeckSi = sitk.ReadImage(cta_path)
            maskSi = sitk.ReadImage(mask_path)
            # resample mask to match brainNeckSi
            maskSiRes = sitk.Resample(maskSi, brainNeckSi, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, maskSi.GetPixelID())
            # cache
            if self._useCache:
                self._cache[idx]['brainNeckSi'] = brainNeckSi
                self._cache[idx]['maskSiRes'] = maskSiRes
        
        if self._useCache and self._cache[idx]['brainNeckSi'] is not None:
            brainNeckSi = self._cache[idx]['brainNeckSi']
            maskSiRes = self._cache[idx]['maskSiRes']

        # convert to array
        ctaArr = sitk.GetArrayFromImage(brainNeckSi)
        maskArr = sitk.GetArrayFromImage(maskSiRes)

        # randomly select slices, if needed
        if self._slicesPerSubj > 0:
            slicesIds = np.random.choice( ctaArr.shape[0], self._slicesPerSubj )

            ctaArr = ctaArr[ slicesIds, :, : ]
            maskArr = maskArr[ slicesIds, :, : ]

        # add channel dimension
        ctaArr=ctaArr[None,:,:,:]
        maskArr=maskArr[None,:,:,:]

        sample = {'cta': ctaArr, 'mask': maskArr}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    @staticmethod
    def createAndSaveStaticSampleLst( baseDirIn=BASE_DIR, outPath='sampleLst.csv' ):
        '''
        Create a list of samples from the base directory
        '''
        # Get the list of files
        maskLst = glob.glob( baseDirIn + '/sub-*/cta/*-cta_brain_sm_mask.nii.gz' )
        maskLst=sorted( maskLst )
        ctaLst = [ x.replace( '-cta_brain_sm_mask', '-cta' ) for x in maskLst ]
        


        # Create the list of samples
        sampleLst = []
        for ctaPath, maskPath in zip( ctaLst, maskLst ):
            if os.path.isfile( ctaPath ) and os.path.isfile( maskPath ):
                sampleLst.append( {'cta': ctaPath, 'mask': maskPath} )
        

        # Save the list of samples
        sampleDf = pd.DataFrame( sampleLst )
        sampleDf.to_csv( outPath, index=True )
        
    @staticmethod
    def ctaPreprocess1(ctaTensorIn):
        # clip HU values
        ctaTensorOut = torch.clamp(ctaTensorIn, 0, 100)
        #normalize
        ctaTensorOut = ctaTensorOut/100.

        return ctaTensorOut
    

#main
if __name__ == '__main__':
    # CtaMaskDataset.createAndSaveStaticSampleLst(BASE_DIR, SAMPLE_CSV_FILE)
    # dataDs = CtaMaskDataset(SAMPLE_CSV_FILE)

    # res = dataDs[0]

    # dataLd = tData.DataLoader( dataDs, batch_size=3)

    # # loop through the batches
    # for i_batch, sample_batched in enumerate(dataLd):
    #     print(i_batch, sample_batched['cta'].size(),
    #           sample_batched['mask'].size())

    #     break

    # test caching
    dataDs = CtaMaskDataset('sampleLst5.csv', useCache=True)
    dataLd = tData.DataLoader( dataDs, batch_size=1)

    # loop through the batches
    for i in range(2):
        for i_batch, sample_batched in enumerate(dataLd):
            print(i_batch, sample_batched['cta'].size(),
                sample_batched['mask'].size())

        
    


