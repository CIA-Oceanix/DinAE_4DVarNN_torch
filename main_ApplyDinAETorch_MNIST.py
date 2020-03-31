#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:00:12 2020

@author: rfablet
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:21:45 2019

@author: rfablet
"""

import numpy as np
#from keras import backend as K
#import tensorflow as tf
#import matplotlib.pyplot as plt 
#import os
import argparse
#import pickle
from sklearn import decomposition
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import os
    

#import tensorflow.keras as keras
#import keras
import time
import copy
import torch
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
import torch.nn.functional as F
import dinAE_solver_torch as dinAE

#os.chdir('../Utils')
#import OI        

#import dill as pickle

def eval_InterpPerformance(mask_train,x_train,x_train_missing,x_train_pred,
                           mask_test,x_test,x_test_missing,x_test_pred):
    mse_train      = np.zeros((2))
    mse_train[0]   = np.sum( mask_train * (x_train_pred - x_train_missing)**2 ) / np.sum( mask_train )
    mse_train[1]   = np.mean( (x_train_pred - x_train)**2 )
    #var_Tr        = np.sum( mask_train * (x_train_missing-np.mean(np.mean(x_train_missing,axis=0),axis=0)) ** 2 ) / np.sum( mask_train )
    exp_var_train  = 1. - mse_train #/ var_Tr
            
    mse_test        = np.zeros((2))
    mse_test[0]     = np.sum( mask_test * (x_test_pred - x_test_missing)**2 ) / np.sum( mask_test )
    mse_test[1]     = np.mean( (x_test_pred - x_test)**2 ) 
    #var_Tt       = np.sum( mask_test * (x_test_missing-np.mean(np.mean(x_train_missing,axis=0),axis=0))** 2 ) / np.sum( mask_test )
    exp_var_test = 1. - mse_test #/ var_Tt

    mse_train_interp        = np.sum( (1.-mask_train) * (x_train_pred - x_train)**2 ) / np.sum( 1. - mask_train )
    exp_var_train_interp    = 1. - mse_train_interp #/ var_Tr
    
    mse_test_interp        = np.sum( (1.-mask_test) * (x_test_pred - x_test)**2 ) / np.sum( 1. - mask_test )
    exp_var_test_interp    = 1. - mse_test_interp #/ var_Tr
            
    return mse_train,exp_var_train,mse_test,exp_var_test,mse_train_interp,exp_var_train_interp,mse_test_interp,exp_var_test_interp


# main code
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #parser.add_argument('-d', '--data', help='Image dataset used for learning: cifar, mnist or custom numpy 4D array (samples, rows, colums, channel) using pickle', type=str, default='cifar')
    #parser.add_argument('-e', '--epoch', help='Number of epochs', type=int, default=100)
    #parser.add_argument('-b', '--batchsize', help='Batch size', type=int, default=64)
    #parser.add_argument('-o', '--output', help='Output model name (required) (.h5)', type=str, required = True)
    #parser.add_argument('--optim', help='Optimization method: sgd, adam', type=str, default='sgd')

    flagDisplay   = 0
    flagProcess   = [0,1,2,3,4]
    flagSaveModel = 1
     
    Wsquare     = 4#0 # half-width of holes
    Nsquare     = 3  # number of holes
    DimAE       = 20#20 # Dimension of the latent space
    flagAEType  = 2#7#

    flagDataset = 0 # 0: MNIST, 1: MNIST-FASHION, 2: SST
    flagDataWindowing = 0
    
    alpha_Losss     = np.array([1.0,0.1])
    flagGradModel   = 0 # Gradient computation (0: subgradient, 1: true gradient/autograd)
    flagOptimMethod = 1 # 0: fixed-step gradient descent, 1: ConvNet_step gradient descent, 2: LSTM-based descent
    #NiterProjection = 5 # Number of fixed-point iterations
    #NiterGrad       = 5 # Number of gradient descent step

  
    batch_size        = 512#4#8#12#8#256#8
    NbEpoc            = 20
    Niter             = 50
    NSampleTr         = 445#550#334#
    
    for kk in range(0,len(flagProcess)):
        ###############################################################
        ## read dataset
        if flagProcess[kk] == 0:        
            if flagDataset == 0: ## MNIST
              #(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
              mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
              mnist_testset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)
            
              x_train = mnist_trainset.data.cpu().detach().numpy()
              y_train = mnist_trainset.targets.cpu().detach().numpy()
                          
              x_test = mnist_testset.data.cpu().detach().numpy()
              y_test = mnist_testset.targets.cpu().detach().numpy()  
              
              x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
              x_test  = x_test.reshape((x_test.shape[0],x_train.shape[1],x_train.shape[2],1))
              
              dirSAVE = './ResMNIST/'
              genFilename = 'mnist_DINConvAE_'
              flagloadOIData = 0

              meanTr     = np.mean(x_train)
              x_train    = x_train - meanTr
              x_test     = x_test  - meanTr
            
              # scale wrt std
              stdTr      = np.sqrt( np.mean( x_train**2 ) )
              x_train    = x_train / stdTr
              x_test     = x_test  / stdTr
              
              #x_train   = x_train[0:1000,:,:]
              #x_test    = x_test[0:512,:,:]

            elif flagDataset == 1: ## FASHION MNIST

              (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

              x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
              x_test  = x_test.reshape((x_test.shape[0],x_train.shape[1],x_train.shape[2],1))
              dirSAVE = './MNIST/'
              genFilename = 'fashion_mnist_DINConvAE_'
              flagloadOIData = 0

              meanTr     = np.mean(x_train)
              x_train    = x_train - meanTr
              x_test     = x_test  - meanTr
            
              # scale wrt std
              stdTr      = np.sqrt( np.mean( x_train**2 ) )
              x_train    = x_train / stdTr
              x_test     = x_test  / stdTr

            elif flagDataset == 2: ## NATL60-METOP Dataset
              np.random.seed(100)
              thrMisData = 0.25
              indT       = np.arange(0,5)#np.arange(0,5)#np.arange(0,5)
              indN_Tr    = np.arange(0,415)#np.arange(0,35000)
              indN_Tt    = np.arange(650,800)
              
              if 1*0:
                  indT       = np.arange(2,3)#np.arange(0,5)#np.arange(0,5)
                  indN_Tr    = np.arange(0,800)#np.arange(0,600)#np.arange(0,10)#np.arange(0,35000)
                  indN_Tt    = np.arange(0,300)#np.arange(650,800)

              from netCDF4 import Dataset
              #fileTr = '../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_20080601_20080731_Patch_032_032_005.nc'
              fileTr    = []
              fileTt    = []
              #fileTr.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080601_20080831_Patch_032_032_005.nc')
              #fileTr.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080601_20080630_Patch_064_064_005.nc')
              #fileTr.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080701_20080731_Patch_064_064_005.nc')
              #fileTr.append('/tmp/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080801_20080831_Patch_064_064_005.nc')
                            
              #fileTr.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080601_20080630_Patch_128_128_005.nc')
              #fileTr.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080701_20080731_Patch_128_128_005.nc')
              #fileTr.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080801_20080831_Patch_128_128_005.nc')

              if 1*0 :
                  fileTr.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080701_20080731_Patch_128_512_005.nc')
                  fileTr.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080601_20080630_Patch_128_512_005.nc')
                  fileTr.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080801_20080831_Patch_128_512_005.nc')

                  fileTt.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_Anomaly_20080701_20080731_Patch_128_512_005.nc')
              if 1*0:
                  fileTr.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_20080701_20080731_Patch_128_512_005.nc')
                  fileTr.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_20080601_20080630_Patch_128_512_005.nc')
                  fileTr.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_20080801_20080831_Patch_128_512_005.nc')
  
                  fileTt.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_20080701_20080731_Patch_128_512_005.nc')
              if 1*1:
                  thrMisData = 0.1
                  indT       = np.arange(0,11)#np.arange(0,5)#np.arange(0,5)
                  indN_Tr    = np.arange(0,200)#np.arange(0,800)#np.arange(0,600)#np.arange(0,35000)
                  indN_Tt    = np.arange(0,200)#np.arange(300,600)#np.arange(650,800)
                  SuffixOI   = '_OI_DT11Lx075Ly075Lt003'
                  
                  fileTr.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_ThrData10_20080601_20080630_Patch_128_512_011.nc')
                  #fileTr.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_ThrData10_20080701_20080731_Patch_128_512_011.nc')
                  
                  fileTr.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_ThrData10_20080801_20080831_Patch_128_512_011.nc')
                  fileTr.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_ThrData10_20080901_20080930_Patch_128_512_011.nc')
  
                  #fileTt.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_ThrData10_20080701_20080731_Patch_128_512_011.nc')
                  #fileTt.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_ThrData10_20080801_20080831_Patch_128_512_011.nc')
                  fileTt.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_ThrData10_20080701_20080731_Patch_128_512_011.nc')
                  #fileTt.append('../../Data/DataNATL60/NATL60withMETOPMask/patchDataset_NATL60_Osmosis_METOPMask_ThrData10_20080901_20080930_Patch_128_512_011.nc')

              for ii in range(0,len(fileTr)):
                  print(".... Load SST dataset (training data): "+fileTr[ii])
                  nc_data     = Dataset(fileTr[ii],'r')
              
                  print('.... # samples: %d '%nc_data.dimensions['N'].size)
                  if nc_data.dimensions['N'].size < indN_Tr[-1]:
                      x_train_ii    = nc_data['sst'][:,:,:,indT]
                      mask_train_ii = nc_data['mask'][:,:,:,indT]
                  else:
                      x_train_ii    = nc_data['sst'][indN_Tr,:,:,indT]
                      mask_train_ii = nc_data['mask'][indN_Tr,:,:,indT]
                  print('.... # loaded samples: %d '%x_train_ii.shape[0])
                      
                  # binary mask (remove non-binary labels due to re-gridding)
                  mask_train_ii     = (mask_train_ii > 0.5).astype('float')
                  
                  #x_train_ii    = nc_data['sst'][1000:6000,0:64:2,0:64:2,0]
                  #mask_train_ii = nc_data['mask'][1000:6000,0:64:2,0:64:2,0]
                  
                  nc_data.close()
              
                  if len(indT) == 1:
                      x_train_ii   = x_train_ii.reshape((x_train_ii.shape[0],x_train_ii.shape[1],x_train_ii.shape[2],1))
                      mask_train_ii = mask_train_ii.reshape((mask_train_ii.shape[0],mask_train_ii.shape[1],mask_train_ii.shape[2],1))
                                             
                  # load OI data
                  if flagloadOIData == 1:
                      print(".... Load OI SST dataset (training data): "+fileTr[ii].replace('.nc',SuffixOI+'.nc'))
                      nc_data       = Dataset(fileTr[ii].replace('.nc',SuffixOI+'.nc'),'r')
                      x_train_OI_ii = nc_data['sstOI'][:,:,:]
                      

                  # remove patch if no SST data
                  ss            = np.sum( np.sum( np.sum( x_train_ii < -100 , axis = -1) , axis = -1 ) , axis = -1)
                  ind           = np.where( ss == 0 )
                  
                  #print('... l = %d %d %d'%(ss.shape[0],x_train_ii.shape[0],len(ind[0])))
                  x_train_ii    = x_train_ii[ind[0],:,:,:]
                  mask_train_ii = mask_train_ii[ind[0],:,:,:]
                  if flagloadOIData == 1:
                      x_train_OI_ii = x_train_OI_ii[ind[0],:,:]
                  
                  rateMissDataTr_ii = np.sum( np.sum( np.sum( mask_train_ii , axis = -1) , axis = -1 ) , axis = -1)
                  rateMissDataTr_ii /= mask_train_ii.shape[1]*mask_train_ii.shape[2]*mask_train_ii.shape[3]
                  if 1*1:
                      ind        = np.where( rateMissDataTr_ii  >= thrMisData )              
                      x_train_ii    = x_train_ii[ind[0],:,:,:]
                      mask_train_ii = mask_train_ii[ind[0],:,:,:]                      
                      if flagloadOIData == 1:
                          x_train_OI_ii = x_train_OI_ii[ind[0],:,:]
                      
                  print('.... # remaining samples: %d '%x_train_ii.shape[0])
              
                  if ii == 0:
                      x_train    = np.copy(x_train_ii)
                      mask_train = np.copy(mask_train_ii)
                      if flagloadOIData == 1:
                          x_train_OI = np.copy(x_train_OI_ii)
                  else:
                      x_train    = np.concatenate((x_train,x_train_ii),axis=0)
                      mask_train = np.concatenate((mask_train,mask_train_ii),axis=0)
                      if flagloadOIData == 1:
                          x_train_OI = np.concatenate((x_train_OI,x_train_OI_ii),axis=0)
                                                      
              rateMissDataTr = np.sum( np.sum( np.sum( mask_train , axis = -1) , axis = -1 ) , axis = -1)
              rateMissDataTr /= mask_train.shape[1]*mask_train.shape[2]*mask_train.shape[3]
                  
              if NSampleTr <  x_train.shape[0] :                   
                  ind_rand = np.random.permutation(x_train.shape[0])

                  x_train    = x_train[ind_rand[0:NSampleTr],:,:,:]
                  mask_train = mask_train[ind_rand[0:NSampleTr],:,:,:]
                  if flagloadOIData == 1:
                      x_train_OI = x_train_OI[ind_rand[0:NSampleTr],:,:]
                      
              y_train    = np.ones((x_train.shape[0]))
              #mask_train = maskSet[:,:,:,:]
              #del sstSet
              if flagloadOIData:
                  print("....... # of training patches: %d/%d"%(x_train.shape[0],x_train_OI.shape[0]))
              else:
                  print("....... # of training patches: %d"%(x_train.shape[0]))
              
              if 1*1: 

                  print(".... Load SST dataset (test data): "+fileTt[0])              
                  nc_data     = Dataset(fileTt[0],'r')
                  if nc_data.dimensions['N'].size < indN_Tt[-1]:
                      x_test    = nc_data['sst'][:,:,:,indT]
                      mask_test = nc_data['mask'][:,:,:,indT]
                  else:
                      x_test    = nc_data['sst'][indN_Tt,:,:,indT]
                      mask_test = nc_data['mask'][indN_Tt,:,:,indT]
                  mask_test     = (mask_test > 0.5).astype('float')
                  
                  #x_test    = nc_data['sst'][7000:10000,0:64:2,0:64:2,0]
                  #mask_test = nc_data['mask'][7000:10000,0:64:2,0:64:2,0]
                  
                  nc_data.close()
                  if len(indT) == 1:
                      x_test    = x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],1))
                      mask_test = mask_test.reshape((mask_test.shape[0],mask_test.shape[1],mask_test.shape[2],1))

                  # load OI data
                  if flagloadOIData == 1:
                      print(".... Load OI SST dataset (test data): "+fileTt[0].replace('.nc',SuffixOI+'.nc'))
                      nc_data   = Dataset(fileTt[0].replace('.nc',SuffixOI+'.nc'),'r')
                      x_test_OI = nc_data['sstOI'][:,:,:]
                  
                   # remove patch if no SST data
                  ss        = np.sum( np.sum( np.sum( x_test < -100 , axis = -1) , axis = -1 ) , axis = -1)
                  ind       = np.where( ss == 0 )
                  x_test    = x_test[ind[0],:,:,:]
                  mask_test = mask_test[ind[0],:,:,:]
                  rateMissDataTt = np.sum( np.sum( np.sum( mask_test , axis = -1) , axis = -1 ) , axis = -1)
                  rateMissDataTt /= mask_test.shape[1]*mask_test.shape[2]*mask_test.shape[3]
                  
                  if flagloadOIData == 1:
                      x_test_OI    = x_test_OI[ind[0],:,:]
                                    
                  if 1*0:
                      ind       = np.where( rateMissDataTt >= thrMisData )              
                      x_test    = x_test[ind[0],:,:,:]
                      mask_test = mask_test[ind[0],:,:,:]
                  y_test    = np.ones((x_test.shape[0]))
              else:
                  Nt        = int(np.floor(x_train.shape[0]*0.25))
                  x_test    = np.copy(x_train[0:Nt,:,:,:])
                  mask_test = np.copy(mask_train[0:Nt,:,:,:])
                  y_test    = np.ones((x_test.shape[0]))
                 
                  x_train    = x_train[Nt+1::,:,:,:]
                  mask_train = mask_train[Nt+1::,:,:,:]
                  y_train    = np.ones((x_train.shape[0]))
                           
              if flagloadOIData:
                  print("....... # of test patches: %d /%d"%(x_test.shape[0],x_test_OI.shape[0]))
              else:
                  print("....... # of test patches: %d"%(x_test.shape[0]))

                    
              print("... mean Tr = %f"%(np.mean(x_train)))
              print("... mean Tt = %f"%(np.mean(x_test)))
                    
              print(".... Training set shape %dx%dx%dx%d"%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
              print(".... Test set shape     %dx%dx%dx%d"%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))
            
              dirSAVE = '../../Data/DataNATL60/NATL60withMETOPMask/DINAERes_v2/'
              
              if fileTr[0].find('Anomaly') == -1 :
                  genFilename = 'model_patchDataset_NATL60withMETOP_SST_'+str('%03d'%x_train.shape[1])+str('_%03d'%x_train.shape[2])+str('_%03d'%x_train.shape[3])
              else:
                genFilename = 'model_patchDataset_NATL60withMETOP_SSTAnomaly_'+str('%03d'%x_train.shape[1])+str('_%03d'%x_train.shape[2])+str('_%03d'%x_train.shape[3])
                
              print('....... Generic model filename: '+genFilename)
              
              meanTr     = np.mean(x_train)
              x_train    = x_train - meanTr
              x_test     = x_test  - meanTr
            
              if flagloadOIData:
                  x_train_OI    = x_train_OI - meanTr
                  x_test_OI     = x_test_OI  - meanTr

              # scale wrt std
              stdTr      = np.sqrt( np.mean( x_train**2 ) )
              x_train    = x_train / stdTr
              x_test     = x_test  / stdTr

              print('... Mean and std of training data: %f  -- %f'%(meanTr,stdTr))

              if flagloadOIData == 1:
                  x_train_OI    = x_train_OI / stdTr
                  x_test_OI     = x_test_OI  / stdTr

            if flagDataWindowing == 1:
                HannWindow = np.reshape(np.hanning(x_train.shape[2]),(x_train.shape[1],1)) * np.reshape(np.hanning(x_train.shape[1]),(x_train.shape[2],1)).transpose() 

                x_train = np.moveaxis(np.moveaxis(x_train,3,1) * np.tile(HannWindow,(x_train.shape[0],x_train.shape[3],1,1)),1,3)
                x_test  = np.moveaxis(np.moveaxis(x_test,3,1) * np.tile(HannWindow,(x_test.shape[0],x_test.shape[3],1,1)),1,3)
                print(".... Training set shape %dx%dx%dx%d"%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
                print(".... Test set shape     %dx%dx%dx%d"%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))

            elif flagDataWindowing == 2:
                EdgeWidth  = 4
                EdgeWindow = np.zeros((x_train.shape[1],x_train.shape[2]))
                EdgeWindow[EdgeWidth:x_train.shape[1]-EdgeWidth,EdgeWidth:x_train.shape[2]-EdgeWidth] = 1
                
                x_train = np.moveaxis(np.moveaxis(x_train,3,1) * np.tile(EdgeWindow,(x_train.shape[0],x_train.shape[3],1,1)),1,3)
                x_test  = np.moveaxis(np.moveaxis(x_test,3,1) * np.tile(EdgeWindow,(x_test.shape[0],x_test.shape[3],1,1)),1,3)

                mask_train = np.moveaxis(np.moveaxis(mask_train,3,1) * np.tile(EdgeWindow,(mask_train.shape[0],x_train.shape[3],1,1)),1,3)
                mask_test  = np.moveaxis(np.moveaxis(mask_test,3,1) * np.tile(EdgeWindow,(mask_test.shape[0],x_test.shape[3],1,1)),1,3)
                print(".... Training set shape %dx%dx%dx%d"%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
                print(".... Test set shape     %dx%dx%dx%d"%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))
              
#            else:
#              mini = np.amin(x_train[:])
#              maxi = np.amax(x_train[:])
#              
#              x_train = (x_train - mini ) /(maxi-mini)
#              x_test  = (x_test - mini ) /(maxi-mini)
            
            print("... (after normalization) mean Tr = %f"%(np.mean(x_train)))
            print("... (after normalization) mean Tt = %f"%(np.mean(x_test)))
              
        ###############################################################
        ## generate missing data
        elif flagProcess[kk] == 1:
            
            if Wsquare > 0 :
                print("..... Generate missing data masks: %dx%dx%d "%(Nsquare,Wsquare,Wsquare))
                
                Wsquare = int(Wsquare)
                Nsquare = int(Nsquare)
                
                # random seed 
                np.random.seed(1)
                
                # generate missing data areas for training data
                x_train_missing = np.copy(x_train).astype(float)
                mask_train      = np.zeros((x_train.shape))
                mask_test       = np.zeros((x_test.shape))
                
                
                for ii in range(x_train.shape[0]):
                  # generate mask
                  mask   = np.ones((x_train.shape[1],x_train.shape[2],x_train.shape[3])).astype(float)
                  i_area = np.floor(np.random.uniform(Wsquare,x_train.shape[1]-Wsquare+1,Nsquare)).astype(int)
                  j_area = np.floor(np.random.uniform(Wsquare,x_train.shape[2]-Wsquare+1,Nsquare)).astype(int)
                  
                  for nn in range(Nsquare):
                    mask[i_area[nn]-Wsquare:i_area[nn]+Wsquare,j_area[nn]-Wsquare:j_area[nn]+Wsquare,:] = 0.
                    
                  # apply mask
                  x_train_missing[ii,:,:,:] *= mask
                  mask_train[ii,:,:,:]       = mask     
                  
                ## generate missing data areas for test data
                x_test_missing = np.copy(x_test).astype(float)
                
                for ii in range(x_test.shape[0]):
                  # generate mask
                  mask   = np.ones((x_test.shape[1],x_test.shape[2],x_train.shape[3])).astype(float)
                  i_area = np.floor(np.random.uniform(Wsquare,x_test.shape[1]-Wsquare+1,Nsquare)).astype(int)
                  j_area = np.floor(np.random.uniform(Wsquare,x_test.shape[2]-Wsquare+1,Nsquare)).astype(int)
                  
                  for nn in range(Nsquare):
                    mask[i_area[nn]-Wsquare:i_area[nn]+Wsquare,j_area[nn]-Wsquare:j_area[nn]+Wsquare,:] = 0.
                    
                  # apply mask
                  x_test_missing[ii,:,:,:] *= mask
                  mask_test[ii,:,:,:]      = mask     
            elif Wsquare == 0 :
                print("..... Use missing data masks from file ")
                Nsquare = 0
                x_train_missing = x_train * mask_train
                x_test_missing  = x_test  * mask_test
                 
#            elif Wsquare < 0 :
#                print("..... Use edge window mask ")
#                Nsquare = 0
#                x_train_missing = x_train * mask_train
#                x_test_missing  = np.moveaxis( np.tile(EdgeWindow,(x_train.shape[0],x_train.shape[3],1,1)),1,3)
#                x_test          = np.moveaxis( np.tile(EdgeWindow,(x_test.shape[0],x_test.shape[3],1,1)),1,3)
#                x_test_missing  = x_test  * mask_test

        ###############################################################
        ## define AE architecture
        elif flagProcess[kk] == 2:                    
            DimCAE = DimAE
            shapeData     = np.array(x_train.shape[1:])
            shapeData[0]  = x_train.shape[3]
            shapeData[1:] = x_train.shape[1:3]
            
            print('.. ShapeData: '+str(shapeData))
            
            if flagAEType == 0: ## MLP-AE

                class Encoder(torch.nn.Module):
                    def __init__(self):
                        super(Encoder, self).__init__()
                        self.fc1 = torch.nn.Linear(shapeData[0]*shapeData[1]*shapeData[2],6*DimAE)
                        self.fc2 = torch.nn.Linear(6*DimAE,2*DimAE)
                        self.fc3 = torch.nn.Linear(2*DimAE,DimAE)
                
                    def forward(self, x):
                        #x = self.fc1( torch.nn.Flatten(x) )
                        x = self.fc1( x.view(-1,shapeData[0]*shapeData[1]*shapeData[2]) )
                        x = self.fc2( F.relu(x) )
                        x = self.fc3( F.relu(x) )
                        return x


                class Decoder(torch.nn.Module):
                    def __init__(self):
                        super(Decoder, self).__init__()
                        self.fc1 = torch.nn.Linear(DimAE,10*DimAE)
                        self.fc2 = torch.nn.Linear(10*DimAE,20*DimAE)
                        self.fc3 = torch.nn.Linear(20*DimAE,shapeData[0]*shapeData[1]*shapeData[2])
            
                    def forward(self, x):
                        x = self.fc1( x )
                        x = self.fc2( F.relu(x) )
                        x = self.fc3( F.relu(x) )
                        x = x.view(-1,shapeData[0],shapeData[1],shapeData[2])
                        return x


            elif flagAEType == 1: ## Conv-AE
              Wpool_i = np.floor(  (np.floor((shapeData[1]-2)/2)-2)/2 ).astype(int) 
              Wpool_j = np.floor(  (np.floor((shapeData[2]-2)/2)-2)/2 ).astype(int)
            
            
              class Encoder(torch.nn.Module):
                  def __init__(self):
                      super(Encoder, self).__init__()
                      self.conv1 = torch.nn.Conv2d(shapeData[0],DimAE,(3,3),padding=0)
                      self.pool1 = torch.nn.AvgPool2d((2,2))
                      self.conv2 = torch.nn.Conv2d(DimAE,2*DimAE,(3,3),padding=0)
                      self.pool2 = torch.nn.AvgPool2d((2,2))
                      self.conv3 = torch.nn.Conv2d(2*DimAE,4*DimAE,(Wpool_i,Wpool_j),padding=0)
                      self.conv4 = torch.nn.Conv2d(4*DimAE,DimAE,(1,1),padding=0)
            
                  def forward(self, x):
                      #x = self.fc1( torch.nn.Flatten(x) )
                      x = self.conv1( x )
                      x = self.pool1(x)
                      x = self.conv2( F.relu(x) )
                      x = self.pool2(x)
                      x = self.conv3( F.relu(x) )
                      x = self.conv4( F.relu(x) )
                      x = x.view(-1,DimAE)
                      return x


              class Decoder(torch.nn.Module):
                  def __init__(self):
                      super(Decoder, self).__init__()
                      self.conv1Tr = torch.nn.ConvTranspose2d(DimAE,DimAE,(x_train.shape[1],x_train.shape[2]),stride=(x_train.shape[1],x_train.shape[2]),bias=False)
                      #self.conv1Tr = torch.nn.ConvTranspose2d(DimAE,DimAE,(int(shapeData[1]/2),int(shapeData[2]/2)),stride=(int(shapeData[1]/2),int(shapeData[2]/2)),bias=False)
                      #self.conv11   = torch.nn.Conv2d(DimAE,DimAE,(3,3),padding=1)
                      #self.conv12   = torch.nn.Conv2d(DimAE,DimAE,(3,3),padding=1)
                      #self.conv2Tr = torch.nn.ConvTranspose2d(DimAE,DimAE,(2,2),stride=(2,2),bias=False)
                      #self.resnet  = self._make_ResNet(2,DimAE,5,3,1)
                      self.resnet = dinAE.ResNetConv2D(2,DimAE,2,3,1)
                      self.convF   = torch.nn.Conv2d(DimAE,1,(1,1),padding=0)
                  def _make_ResNet(self,Nblocks,dim,K,kernel_size, padding):
                      layers = []
                      for kk in range(0,Nblocks):
                        layers.append(torch.nn.Conv2d(dim,K*dim,kernel_size,padding=padding,bias=False))
                        layers.append(torch.nn.ReLU())
                        layers.append(torch.nn.Conv2d(K*dim,dim,kernel_size,padding=padding,bias=False))
            
                      return torch.nn.Sequential(*layers)
            
                  def forward(self, x):
                      x = x.view(-1,DimAE,1,1)
                      x = self.conv1Tr( x )
                      #x = torch.add(self.conv12( F.relu( self.conv11(x) ) ),x)
                      #x = torch.add(self.conv12( F.relu( self.conv11(x) ) ),x)
                      #x = self.conv2Tr( x )
                      x = self.resnet(x)
                      x = self.convF(x)
                      return x
                
            elif flagAEType == 2: ## Conv model with no use of the central point
              class Encoder(torch.nn.Module):
                  def __init__(self):
                      super(Encoder, self).__init__()
                      self.pool1 = torch.nn.AvgPool2d((2,2))
                      self.conv1 = dinAE.ConstrainedConv2d(shapeData[0],shapeData[0]*DimAE,(3,3),padding=1)
                      self.conv2 = torch.nn.Conv2d(shapeData[0]*DimAE,2*shapeData[0]*DimAE,(1,1),padding=0)
                      self.conv3 = torch.nn.Conv2d(2*shapeData[0]*DimAE,4*shapeData[0]*DimAE,(1,1),padding=0)
                      self.conv4 = torch.nn.Conv2d(4*shapeData[0]*DimAE,8*shapeData[0]*DimAE,(1,1),padding=0)
                      self.conv2Tr = torch.nn.ConvTranspose2d(8*shapeData[0]*DimAE,8*shapeData[0]*DimAE,(2,2),stride=(2,2),bias=False)          
                      self.conv5 = torch.nn.Conv2d(8*shapeData[0]*DimAE,16*shapeData[0]*DimAE,(3,3),padding=1)
                      self.conv6 = torch.nn.Conv2d(16*shapeData[0]*DimAE,1,(3,3),padding=1)
            
                  def forward(self, x):
                      #x = self.fc1( torch.nn.Flatten(x) )
                      x = self.pool1( x )
                      x = self.conv1(x)
                      x = self.conv2( F.relu(x) )
                      x = self.conv3( F.relu(x) )
                      x = self.conv4( F.relu(x) )
                      x = self.conv2Tr( x )
                      x = self.conv5( x )
                      x = self.conv6( x )
                      x = x.view(-1,shapeData[0],shapeData[1],shapeData[2])
                      return x
            
              class Decoder(torch.nn.Module):
                  def __init__(self):
                      super(Decoder, self).__init__()
            
                  def forward(self, x):
                      return torch.mul(1.,x)
            elif flagAEType == 6: ## Conv-AE for SST
              Wpool_i = np.floor(  (np.floor((shapeData[1]-2)/2)-2)/2 ).astype(int) 
              Wpool_j = np.floor(  (np.floor((shapeData[2]-2)/2)-2)/2 ).astype(int)
            
            
              class Encoder(torch.nn.Module):
                  def __init__(self):
                      super(Encoder, self).__init__()
                      self.conv1 = torch.nn.Conv2d(shapeData[0],DimAE,(3,3),padding=0)
                      self.pool1 = torch.nn.AvgPool2d((2,2))
                      self.conv2 = torch.nn.Conv2d(DimAE,2*DimAE,(3,3),padding=0)
                      self.pool2 = torch.nn.AvgPool2d((2,2))
                      self.conv3 = torch.nn.Conv2d(2*DimAE,2*DimAE,(3,3),padding=0)
                      self.pool3 = torch.nn.AvgPool2d((2,2))
                      self.conv4 = torch.nn.Conv2d(2*DimAE,2*DimAE,(3,3),padding=0)
                      self.pool4 = torch.nn.AvgPool2d((2,2))
                      self.conv5 = torch.nn.Conv2d(2*DimAE,2*DimAE,(3,3),padding=0)
                      self.pool5 = torch.nn.AvgPool2d((2,2))
                      self.conv6 = torch.nn.Conv2d(2*DimAE,DimAE,(1,1),padding=0)
            
                  def forward(self, x):
                      #x = self.fc1( torch.nn.Flatten(x) )
                      x = self.conv1( x )
                      x = self.pool1(x)
                      x = self.conv2( F.relu(x) )
                      x = self.pool2(x)
                      x = self.conv3( F.relu(x) )
                      x = self.pool3(x)
                      x = self.conv4( F.relu(x) )
                      x = self.pool4(x)
                      x = self.conv5( F.relu(x) )
                      x = self.pool5(x)
                      x = self.conv6( x )
                      x = x.view(-1,DimAE)
                      return x


              class Decoder(torch.nn.Module):
                  def __init__(self):
                      super(Decoder, self).__init__()
                      #self.conv1Tr = torch.nn.ConvTranspose2d(DimAE,DimAE,(x_train.shape[1],x_train.shape[2]),stride=(x_train.shape[1],x_train.shape[2]),bias=False)
                      self.conv1Tr = torch.nn.ConvTranspose2d(DimAE,256,(16,16),stride=(16,16),bias=False)
                      self.conv2Tr = torch.nn.ConvTranspose2d(256,64,(2,2),stride=(2,2),bias=False)
                      self.conv3   = torch.nn.Conv2d(64,32,(1,1),padding=0)
                      self.resnet  = dinAE.ResNetConv2D(2,32,2,3,1)
                      self.convF  = torch.nn.Conv2d(32,shapeData[0],(1,1),padding=0)
                  def _make_ResNet(self,Nblocks,dim,K,kernel_size, padding):
                      layers = []
                      for kk in range(0,Nblocks):
                        layers.append(torch.nn.Conv2d(dim,K*dim,kernel_size,padding=padding,bias=False))
                        layers.append(torch.nn.ReLU())
                        layers.append(torch.nn.Conv2d(K*dim,dim,kernel_size,padding=padding,bias=False))
            
                      return torch.nn.Sequential(*layers)
            
                  def forward(self, x):
                      x = self.conv1Tr( x )
                      x = self.conv2Tr( F.relu(x) )
                      x = self.conv3( F.relu(x) )
                      #x = torch.add(self.conv12( F.relu( self.conv11(x) ) ),x)
                      #x = torch.add(self.conv12( F.relu( self.conv11(x) ) ),x)
                      #x = self.conv2Tr( x )
                      x = self.resnet(x)
                      x = self.convF(x)
                      return x

            ## auto-encoder architecture
            class Model_AE(torch.nn.Module):
                def __init__(self):
                    super(Model_AE, self).__init__()
                    self.encoder = Encoder()
                    self.decoder = Decoder()
            
                def forward(self, x):
                    x = self.encoder( x )
                    x = self.decoder( x )
                    return x
              
            model_AE = Model_AE()
            print(model_AE)
            print('Number of trainable parameters = %d'%(sum(p.numel() for p in model_AE.parameters() if p.requires_grad)))  

        ###############################################################
        ## define classifier architecture for performance evaluation
        elif flagProcess[kk] == 3:
            num_classes = (np.max(y_train)+1).astype(int)
            
            class Classifier(torch.nn.Module):
                def __init__(self):
                    super(Classifier, self).__init__()
                    self.fc1 = torch.nn.Linear(DimAE,32)
                    self.fc2 = torch.nn.Linear(32,64)
                    self.fc3 = torch.nn.Linear(64,num_classes)
            
                def forward(self, x):
                    x = self.fc1( x )
                    x = self.fc2( F.relu(x) )
                    x = self.fc3( F.relu(x) )
                    x = F.softmax( x , num_classes )
                    return x

            classifier = Classifier()
 
        ###############################################################
        ## train Conv-AE
        elif flagProcess[kk] == 4:        

            IterUpdate     = [0,2,4,6,8,20]#[0,2,4,6,9,15]
            NbProjection   = [5,5,5,10,10,1]#[0,0,0,0,0,0]#[5,5,5,5,5]##
            NbGradIter     = [0,0,0,0,0,0]#[0,0,1,2,3,3]#[0,2,2,4,5,5]#
            lrUpdate       = [1e-3,1e-4,1e-5,1e-4,1e-5,1e-7]
            if 1*0:
                IterUpdate     = [1,2,4,7,10,20]#[0,2,4,6,9,15]
                NbProjection   = [0,1,1,1,1]#[0,0,0,0,0,0]#[5,5,5,5,5]##
                NbGradIter     = [0,5,5,10,10,14]#[0,0,1,2,3,3]#[0,2,2,4,5,5]#
                lrUpdate       = [1e-3,1e-4,1e-5,1e-5,1e-6,1e-7]
            # PCA decomposition for comparison
            pca              = decomposition.PCA(DimCAE)
            pca.fit(np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3])))
                        
            
            rec_PCA_Tt       = pca.transform(np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])))
            rec_PCA_Tt[:,DimCAE:] = 0.
            rec_PCA_Tt       = pca.inverse_transform(rec_PCA_Tt)
            mse_PCA_Tt       = np.mean( (rec_PCA_Tt - x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])))**2 )
            var_Tt           = np.mean( (x_test-np.mean(x_train,axis=0))** 2 )
            exp_var_PCA_Tt   = 1. - mse_PCA_Tt / var_Tt
            
            print(".......... PCA Dim = %d"%(DimCAE))
            print('.... explained variance PCA (Tr) : %.2f%%'%(100.*np.cumsum(pca.explained_variance_ratio_)[DimCAE-1]))
            print('.... explained variance PCA (Tt) : %.2f%%'%(100.*exp_var_PCA_Tt))
        
            # Modify/Check data format
            x_train         = np.moveaxis(x_train, -1, 1)
            x_train_missing = np.moveaxis(x_train_missing, -1, 1)
            mask_train      = np.moveaxis(mask_train, -1, 1)

            x_test         = np.moveaxis(x_test, -1, 1)
            mask_test      = np.moveaxis(mask_test, -1, 1)
            x_test_missing = np.moveaxis(x_test_missing, -1, 1)
            print("... Training datashape: "+str(x_train.shape))
            print("... Test datashape    : "+str(x_test.shape))

            # mean-squared error loss
            #criterion = torch.nn.MSELoss()
            stdTr    = np.std( x_train )
            stdTt    = np.std( x_test )

            ## Define dataloaders with randomised batches     
            ## no random shuffling for validation/test data
            training_dataset     = torch.utils.data.TensorDataset(torch.Tensor(x_train_missing),torch.Tensor(mask_train),torch.Tensor(x_train)) # create your datset
            test_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_test_missing),torch.Tensor(mask_test),torch.Tensor(x_test)) # create your datset
                        
            dataset_sizes = {'train': len(training_dataset), 'val': len(test_dataset)} 

            ## instantiate model for GPU implementation
            #  use gpu if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #device = torch.device("cuda")
            print(".... Device GPU: "+str(torch.cuda.is_available()))
    
    
            NBProjCurrent = NbProjection[0]
            NBGradCurrent = NbGradIter[0]
            print('..... DinAE learning (initialisation): NBProj = %d -- NGrad = %d'%(NBProjCurrent,NBGradCurrent))
            
            # NiterProjection,NiterGrad: global variables
            # bug for NiterProjection = 0
            shapeData       = x_train.shape[1:]
            #model_AE_GradFP = Model_AE_GradFP(model_AE2,shapeData,NiterProjection,NiterGrad,GradType,OptimType)
            model = dinAE.Model_AE_GradFP(model_AE,shapeData,NBProjCurrent,NBGradCurrent,flagGradModel,flagOptimMethod)
        
            model = model.to(device)

            # create an optimizer object
            # Adam optimizer with learning rate 1e-3
            optimizer        = optim.Adam(model.parameters(), lr=lrUpdate[0])
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.)
                        
            # model compilation
            # model fit
            since = time.time()

            alpha_MaskedLoss = alpha_Losss[0]
            alpha_GTLoss     = 1. - alpha_Losss[0]
            alpha_AE         = alpha_Losss[1]

            best_model_wts = copy.deepcopy(model.state_dict())
                                    
            iterInit    = 0
            comptUpdate = 0
            for iter in range(iterInit,Niter):

                print()
                print('............................................................')
                print('............................................................')
                print('..... Iter  %d '%(iter))
                best_loss      = 1e10
                if iter == IterUpdate[comptUpdate]:
                    # update GradFP parameters
                    NBProjCurrent = NbProjection[comptUpdate]
                    NBGradCurrent = NbGradIter[comptUpdate]

                    print("..... ")
                    print("..... ")
                    print("..... Update/initialize number of projections/Graditer in GradCOnvAE model # %d/%d"%(NbProjection[comptUpdate],NbGradIter[comptUpdate]))

                    # update GradFP architectures
                    print('..... Update model architecture')
                    print("..... ")
                    model = dinAE.Model_AE_GradFP(model_AE,shapeData,NBProjCurrent,NBGradCurrent,flagGradModel,flagOptimMethod)
                    model = model.to(device)
                    
                    # UPDATE optimizer
                    optimizer        = optim.Adam(model.parameters(), lr= lrUpdate[comptUpdate])
                    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.)

                    # copy model parameters from current model
                    model.load_state_dict(best_model_wts)
                                        
                    # update comptUpdate
                    if comptUpdate < len(NbProjection)-1:
                        comptUpdate += 1

                print('..... AE Model type : %d '%(flagAEType))
                print('..... Gradient type : %d '%(flagGradModel))
                print('..... Optim type    : %d '%(flagOptimMethod))
                print('..... DinAE learning: NBProj = %d -- NGrad = %d'%(NBProjCurrent,NBGradCurrent))
                print('..... Learning rate : %f'%lrUpdate[comptUpdate])

                # Daloader during training phase                
                dataloaders = {
                    'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
                    'val': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                }
                
                # Run NbEpoc training epochs
                for epoch in range(NbEpoc):
                    #print('Epoch {}/{}'.format(epoch, NbEpoc - 1))
                    #print('-' * 10)
                    print('Epoc %d/%d'%(epoch,NbEpoc))
                    
                    # Each epoch has a training and validation phase
                    for phase in ['train', 'val']:
                        if phase == 'train':
                            #rint('Learning')
                            model.train()  # Set model to training mode
                            #print('..... Training step')
                        else:
                            #print('Evaluation')
                            model.eval()   # Set model to evaluate mode
                            #print('..... Test step')
            
                        running_loss = 0.0
                        running_loss_All     = 0.
                        running_loss_R       = 0.
                        running_loss_I       = 0.
                        running_loss_AE      = 0.
                        num_loss     = 0
    
                        # Iterate over data.
                        #for inputs_ in dataloaders[phase]:
                        #    inputs = inputs_[0].to(device)
                        compt = 0
                        for inputs_missing,masks,inputs_GT in dataloaders[phase]:
                            compt = compt+1
                            #print('.. batch %d'%compt)
                            
                            inputs_missing = inputs_missing.to(device)
                            masks          = masks.to(device)
                            inputs_GT      = inputs_GT.to(device)
                            #print(inputs.size(0))
            
                            # zero the parameter gradients
                            optimizer.zero_grad()
    
                            # forward
                            # need to evaluate grad/backward during the evaluation and training phase for model_AE
                            with torch.set_grad_enabled(True): 
                            #with torch.set_grad_enabled(phase == 'train'):
                                outputs = model(inputs_missing,masks)
                                #outputs = model(inputs)
                                #loss = criterion( outputs,  inputs)
                                loss_R      = torch.sum((outputs - inputs_GT)**2 * masks )
                                loss_R      = torch.mul(1.0 / torch.sum(masks),loss_R)
                                loss_I      = torch.sum((outputs - inputs_GT)**2 * (1. - masks) )
                                loss_I      = torch.mul(1.0 / torch.sum(1.-masks),loss_I)
                                loss_All    = torch.mean((outputs - inputs_GT)**2 )
                                loss_AE     = torch.mean((model.model_AE(outputs) - outputs)**2 )
                                loss_AE_GT  = torch.mean((model.model_AE(inputs_GT) - inputs_GT)**2 )
                                
                                if alpha_MaskedLoss > 0.:
                                    loss = torch.mul(alpha_MaskedLoss,loss_R)
                                else: 
                                    loss = torch.mul(alpha_GTLoss,loss_All)
                                loss = torch.add(loss,torch.mul(alpha_AE,loss_AE))
            
                                # backward + optimize only if in training phase
                                if phase == 'train':
                                    loss.backward()
                                    optimizer.step()
    
                            # statistics
                            running_loss             += loss.item() * inputs_missing.size(0)
                            running_loss_I           += loss_I.item() * inputs_missing.size(0)
                            running_loss_R           += loss_R.item() * inputs_missing.size(0)
                            running_loss_All         += loss_All.item() * inputs_missing.size(0)
                            running_loss_AE          += loss_AE_GT.item() * inputs_missing.size(0)
                            num_loss                 += inputs_missing.size(0)
                            #running_expvar += torch.sum( (outputs - inputs)**2 ) / torch.sum(
    
                        if phase == 'train':
                            exp_lr_scheduler.step()
    
                        epoch_loss       = running_loss / num_loss
                        epoch_loss_All   = running_loss_All / num_loss
                        epoch_loss_AE    = running_loss_AE / num_loss
                        epoch_loss_I     = running_loss_I / num_loss
                        epoch_loss_R     = running_loss_R / num_loss
                        #epoch_acc = running_corrects.double() / dataset_sizes[phase]
                        if phase == 'train':
                          epoch_nloss_All = epoch_loss_All / stdTr**2
                          epoch_nloss_I   = epoch_loss_I / stdTr**2
                          epoch_nloss_R   = epoch_loss_R / stdTr**2
                          epoch_nloss_AE  = loss_AE / stdTr**2
                        else:
                          epoch_nloss_All = epoch_loss_All / stdTt**2
                          epoch_nloss_I   = epoch_loss_I / stdTt**2
                          epoch_nloss_R   = epoch_loss_R / stdTt**2
                          epoch_nloss_AE   = loss_AE / stdTt**2
    
                        #print('{} Loss: {:.4f} '.format(
                         #   phase, epoch_loss))
                        print('.. {} Loss: {:.4f} NLossAll: {:.4f} NLossR: {:.4f} NLossI: {:.4f} NLossAE: {:.4f}'.format(
                            phase, epoch_loss,epoch_nloss_All,epoch_nloss_R,epoch_nloss_I,epoch_nloss_AE))
            
                        # deep copy the model
                        if phase == 'val' and epoch_loss < best_loss:
                            best_loss = epoch_loss
                            best_model_wts = copy.deepcopy(model.state_dict())
    
                        #print()
                
                    time_elapsed = time.time() - since
                    print('Training complete in {:.0f}m {:.0f}s'.format(
                        time_elapsed // 60, time_elapsed % 60))
                    print('Best val loss: {:4f}'.format(best_loss))


                # load best model weights
                model.load_state_dict(best_model_wts)
            
                ## Performance summary for best model
                # Daloader during training phase                
                dataloaders = {
                    'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                    'val': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                }
                
                ## ouputs for training data
                x_train_pred = []
                for inputs_missing,masks,inputs_GT in dataloaders['train']:
                  inputs_missing = inputs_missing.to(device)
                  masks          = masks.to(device)
                  inputs_GT      = inputs_GT.to(device)
                  with torch.set_grad_enabled(True): 
                  #with torch.set_grad_enabled(phase == 'train'):
                      outputs_ = model(inputs_missing,masks)
                  if len(x_train_pred) == 0:
                      x_train_pred  = torch.mul(1.0,outputs_).cpu().detach()
                  else:
                      x_train_pred  = np.concatenate((x_train_pred,torch.mul(1.0,outputs_).cpu().detach().numpy()),axis=0)

                ## ouputs for test data
                x_test_pred = []
                for inputs_missing,masks,inputs_GT in dataloaders['val']:
                  inputs_missing = inputs_missing.to(device)
                  masks          = masks.to(device)
                  inputs_GT      = inputs_GT.to(device)
                  with torch.set_grad_enabled(True): 
                  #with torch.set_grad_enabled(phase == 'train'):
                      outputs_ = model(inputs_missing,masks)
                  if len(x_test_pred) == 0:
                      x_test_pred  = torch.mul(1.0,outputs_).cpu().detach().numpy()
                  else:
                      x_test_pred  = np.concatenate((x_test_pred,torch.mul(1.0,outputs_).cpu().detach().numpy()),axis=0)

                mse_train,exp_var_train,mse_test,exp_var_test,mse_train_interp,exp_var_train_interp,mse_test_interp,exp_var_test_interp = eval_InterpPerformance(mask_train,x_train,x_train_missing,x_train_pred,
                                           mask_test,x_test,x_test_missing,x_test_pred)
                
                print(".......... iter %d"%(iter))
                print('.... Error for all data (Tr)        : %.2e %.2f%%'%(mse_train[1]*stdTr**2,100.*exp_var_train[1]))
                print('.... Error for all data (Tt)        : %.2e %.2f%%'%(mse_test[1]*stdTr**2,100.*exp_var_test[1]))
                print('....')
                print('.... Error for observed data (Tr)  : %.2e %.2f%%'%(mse_train[0]*stdTr**2,100.*exp_var_train[0]))
                print('.... Error for observed data (Tt)  : %.2e %.2f%%'%(mse_test[0]*stdTr**2,100.*exp_var_test[0]))
                print('....')
                print('.... Error for masked data (Tr)  : %.2e %.2f%%'%(mse_train_interp*stdTr**2,100.*exp_var_train_interp))
                print('.... Error for masked data (Tt)  : %.2e %.2f%%'%(mse_test_interp*stdTr**2,100.*exp_var_test_interp))
                    
                if flagSaveModel == 1:
                    genSuffixModel = 'Torch_Alpha%03d'%(100*alpha_Losss[0]+10*alpha_Losss[1])
#                    if flagUseMaskinEncoder == 1:
#                        genSuffixModel = genSuffixModel+'_MaskInEnc'
#                        if stdMask  > 0:
#                            genSuffixModel = genSuffixModel+'_Std%03d'%(100*stdMask)
                    if alpha_Losss[0] < 1.0:
                        genSuffixModel = genSuffixModel+'_AETRwithTrueData'
                    else:
                        genSuffixModel = genSuffixModel+'_AETRwoTrueData'
                   
                    genSuffixModel = genSuffixModel+str('%02d'%(flagAEType))+'D'+str('%02d'%(DimAE))+'N'+str('%02d'%(Nsquare))+'W'+str('%02d'%(Wsquare))
                    genSuffixModel = genSuffixModel+'_Nproj'+str('%02d'%(NBProjCurrent))
                    genSuffixModel = genSuffixModel+'_Grad_'+str('%02d'%(flagGradModel))+'_'+str('%02d'%(flagOptimMethod))+'_'+str('%02d'%(NBGradCurrent))
      
                    fileMod = dirSAVE+genFilename+genSuffixModel+'_modelAE_iter%03d'%(iter)+'.mod'
                    print('.................. Auto-Encoder '+fileMod)
                    torch.save(model_AE.state_dict(), fileMod)

                    fileMod = dirSAVE+genFilename+genSuffixModel+'_modelAEGradFP_iter%03d'%(iter)+'.mod'
                    print('.................. Auto-Encoder '+fileMod)
                    torch.save(model.state_dict(), fileMod)
                    