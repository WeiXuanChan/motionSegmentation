'''
###############################################################################
MIT License

Copyright (c) 2019 W. X. Chan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
###############################################################################
File: __init__.py
Description: load all class for bfmotionsolver
             Contains linker to main classes
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: w.x.chan@gmail.com         31JAN2018           - Created
Author: w.x.chan@gmail.com         31JAN2019           - v1.2.0
                        -bfSolver version 1.0.0
                        -BsplineFourier version 1.2.0
Author: w.x.chan@gmail.com         12SEP2019           - v2.0.0
                        -bfSolver version 2.0.0
                        -BsplineFourier version 2.0.0
Author: w.x.chan@gmail.com         12SEP2019           - v2.1.0
                        -bfSolver version 2.1.0
                        -BsplineFourier version 2.0.0
Author: w.x.chan@gmail.com         17SEP2019           - v2.2.3
                        -bfSolver version 2.1.0
                        -BsplineFourier version 2.2.0
Author: w.x.chan@gmail.com         23SEP2019           - v2.2.4
                        -bfSolver version 2.2.4
                        -BsplineFourier version 2.2.0
Author: jorry.zhengyu@gmail.com    26SEP2019           - v2.2.5
                        -bfSolver version 2.2.4
                        -BsplineFourier version 2.2.5
Author: w.x.chan@gmail.com    26SEP2019                - v2.2.6
                        -bfSolver version 2.2.4
                        -BsplineFourier version 2.2.6
                        - do import *
Author: w.x.chan@gmail.com    07OCT2019                - v2.2.7
                        -bfSolver version 2.2.4
                        -BsplineFourier version 2.2.7
Author: w.x.chan@gmail.com    07OCT2019                - v2.3.3
                        -bfSolver version 2.2.4
                        -BsplineFourier version 2.3.3
Author: w.x.chan@gmail.com    13NOV2019                - v2.4.1
                        -bfSolver version 2.4.1
                        -BsplineFourier version 2.3.3
Author: w.x.chan@gmail.com    18NOV2019                - v2.4.2
                        -bfSolver version 2.4.1
                        -BsplineFourier version 2.4.2
Author: w.x.chan@gmail.com    18NOV2019                - v2.4.3
                        -bfSolver version 2.4.3
                        -BsplineFourier version 2.4.2
Author: w.x.chan@gmail.com    18NOV2019                - v2.4.5
                        -bfSolver version 2.4.4
                        -BsplineFourier version 2.4.4
Author: w.x.chan@gmail.com    11DEC2019                - v2.4.6
                        -bfSolver version 2.4.6
                        -BsplineFourier version 2.4.4
Author: w.x.chan@gmail.com    13DEC2019                - v2.4.7
                        -bfSolver version 2.4.6
                        -BsplineFourier version 2.4.7
Author: w.x.chan@gmail.com    04FEB2020                - v2.5.1
                        -bfSolver version 2.4.6
                        -BsplineFourier version 2.4.7
                        -motionCorrect version 2.4.7
                        -segment verion 2.5.1
Author: w.x.chan@gmail.com    04FEB2020                - v2.5.7
                        -bfSolver version 2.5.4
                        -BsplineFourier version 2.4.7
                        -motionCorrect version 2.4.7
                        -segment verion 2.5.7
Author: w.x.chan@gmail.com    18FEB2020                - v2.5.10
                        -bfSolver version 2.5.4
                        -BsplineFourier version 2.4.7
                        -motionCorrect version 2.4.7
                        -segment verion 2.5.10
Author: w.x.chan@gmail.com    21FEB2020                - v2.6.2
                        -bfSolver version 2.5.4
                        -BsplineFourier version 2.6.2
                        -motionCorrect version 2.4.7
                        -segment verion 2.6.1
Author: w.x.chan@gmail.com    27FEB2020                - v2.6.3
                        -bfSolver version 2.5.4
                        -BsplineFourier version 2.6.2
                        -motionCorrect version 2.6.3
                        -segment verion 2.6.1
Author: w.x.chan@gmail.com    27FEB2020                - v2.7.7
                        -bfSolver version 2.5.4
                        -BsplineFourier version 2.6.2
                        -motionCorrect version 2.6.3
                        -segment verion 2.7.7
Author: w.x.chan@gmail.com    06MAR2020                - v2.7.9
                        -bfSolver version 2.5.4
                        -BsplineFourier version 2.6.2
                        -motionCorrect version 2.7.8
                        -segment verion 2.7.9
Author: jorry.zhengyu@gmail.com    03June2020                - v2.7.11
                        -bfSolver version 2.7.11
                        -BsplineFourier version 2.6.2
                        -motionCorrect version 2.7.8
                        -segment verion 2.7.9
Author: w.x.chan@gmail.com    15Oct2020                - v2.7.12
                        -bfSolver version 2.7.11
                        -BsplineFourier version 2.7.12
                        -motionCorrect version 2.7.8
                        -segment verion 2.7.9
Author: w.x.chan@gmail.com    07Nov2020                - v2.7.13
                        -bfSolver version 2.7.11
                        -BsplineFourier version 2.7.13
                        -motionCorrect version 2.7.8
                        -segment verion 2.7.9
Author: w.x.chan@gmail.com    20Jan2021                - v2.7.16 -added import exception
                        -bfSolver version 2.7.15
                        -BsplineFourier version 2.7.14
                        -motionCorrect version 2.7.8
                        -segment verion 2.7.14
Author: w.x.chan@gmail.com    20Jan2021                - v2.7.17
                        -bfSolver version 2.7.15
                        -BsplineFourier version 2.7.14
                        -motionCorrect version 2.7.8
                        -segment verion 2.7.17
Author: w.x.chan@gmail.com    25Jan2021                - v2.7.19
                        -bfSolver version 2.7.15
                        -BsplineFourier version 2.7.14
                        -motionCorrect version 2.7.8
                        -segment verion 2.7.19
Author: w.x.chan@gmail.com    08Jul2021                - v2.7.20
                        -bfSolver version 2.7.20
                        -BsplineFourier version 2.7.14
                        -motionCorrect version 2.7.8
                        -segment verion 2.7.19
Author: w.x.chan@gmail.com    15Jul2021                - v2.8.0
                        -bfSolver version 2.8.0
                        -BsplineFourier version 2.7.14
                        -motionCorrect version 2.7.8
                        -segment verion 2.7.19

Author: w.x.chan@gmail.com    21Jul2021                - v2.8.3   replace tab by space
                        -bfSolver version 2.8.0
                        -BsplineFourier version 2.7.14
                        -motionCorrect version 2.7.8
                        -segment verion 2.7.19
Author: w.x.chan@gmail.com    21Jul2021                - v2.8.6   
                                -debug maskImg remained as False 
                        -bfSolver version 2.8.0
                        -BsplineFourier version 2.7.14
                        -motionCorrect version 2.7.8
                        -segment verion 2.7.19
Author: w.x.chan@gmail.com    04Aug2021                - v2.8.7   
                                -added imgfmt to simpleSolver 
                                - added automatic anchor
                        -bfSolver version 2.8.0
                        -BsplineFourier version 2.7.14
                        -motionCorrect version 2.7.8
                        -segment verion 2.7.19
Requirements:
    autoD
    numpy
    re
    scipy
    BsplineFourier
    pickle (optional)
    nfft

Known Bug:
    HSV color format not supported
All rights reserved.
'''
_version='2.8.7'
import logging
logger = logging.getLogger('motionSegmentation v'+_version)
logger.info('motionSegmentation version '+_version)

import os
import sys
import numpy as np
from scipy.ndimage import morphology
import motionSegmentation.BsplineFourier as BsplineFourier
import motionSegmentation.bfSolver as bfSolver
import motionSegmentation.segment as segment
from skimage.segmentation import watershed
import medImgProc as mip
import medImgProc.processFunc as pf
import time

def simpleSolver(savePath,startstep=1,endstep=7,fileScale=None,getCompoundTimeList=None,compoundSchemeList=None,fftLagrangian=True,pngFileFormat=None,period=None,maskImg=True,anchor=None,bgrid=4.,finalShape=None,fourierTerms=4,twoD=False,imgfmt=None):
    '''
    step 1: load image
    step 2: create mask
    step 3: Registrations
    step 4: initialize BSF using fft
    step 5: solve BSF
    step 6: regrid to time
    setp 7: compute compound
    '''
    logging.info(savePath)
    allTime=[]
    allTimeHeader=""
    imregPath=savePath+'/transform/'
    regfile_general='t{0:d}to{1:d}_0.txt'
    saveName='RMSmotion_smooth'
    if anchor is None:
        anchor=[]
    if fileScale is None:
        try:
            fileScale=np.loadtxt(imregPath+'fileScale.txt')
        except:
            fileScale=20.
    if getCompoundTimeList is None:
        try:
            getCompoundTimeList=np.loadtxt(savePath+'/diastole_systole.txt')
        except:
            getCompoundTimeList=[0]
    logging.info(str(getCompoundTimeList))
    if compoundSchemeList is None:
        compoundSchemeList=['SAC','maximum','mean','wavelet']
    

    
    if startstep<=1 and endstep>=1:
        imagedim=np.loadtxt(savePath+'/scale.txt')
        if pngFileFormat is None:
            if twoD:
                pngFileFormat='time{0:03d}.png'
            else:
                pngFileFormat='time{0:03d}/slice{{0:03d}}time{0:03d}.png'
        if twoD:
            image=mip.loadStack(savePath+'/'+pngFileFormat,dimension=['t'],n=1)
            image.dimlen['x']=imagedim[0]
            image.dimlen['y']=imagedim[1]
            image.dimlen['t']=1.
            image.rearrangeDim(['t','y','x'])
        else:
            image=mip.loadStack(savePath+'/'+pngFileFormat,dimension=['t','z'],n=1)
            image.dimlen['x']=imagedim[0]
            image.dimlen['y']=imagedim[1]
            image.dimlen['z']=imagedim[2]
            image.dimlen['t']=1.
            image.rearrangeDim(['t','z','y','x'])
        if imgfmt is not None:
            image.data=image.data.astype(imgfmt)
        image.save(savePath+'/img')
        
    if startstep<=2 and endstep>=2:
        if maskImg:
            image=mip.load(savePath+'/img')
            maskImg=image.clone()
            mask=np.zeros(image.data.shape)
            mask[image.data==0]=1
            mask=np.prod(mask,axis=0)
            if twoD:
                mask=np.tile(mask,(image.data.shape[0],1,1))
                border=np.ones(mask.shape[1:]).astype(bool)
                border[tuple([slice(1,-1)]*len(mask.shape[1:]))]=False
            else:
                mask=np.tile(mask,(image.data.shape[0],1,1,1))
                border=np.ones(mask.shape[2:]).astype(bool)
                border[tuple([slice(1,-1)]*len(mask.shape[2:]))]=False
            for t in range(image.data.shape[0]):
                if twoD:
                    ws=watershed(mask[t])
                    for n in range(ws.max()):
                        if np.count_nonzero(ws==n)>0:
                            if mask[t][ws==n].mean()>=0.5 and not(np.any(border*ws==n)):
                                mask[t][ws==n]=0
                    mask[t]=segment.detectNonregularBoundary(image.data[t].copy(),iterations=30,initArray=mask[t])
                    mask[t]=morphology.binary_erosion(mask[t],iterations=2,border_value=0)
                else:
                    for z in range(image.data.shape[1]):
                        ws=watershed(mask[t,z])
                        for n in range(ws.max()):
                            if np.count_nonzero(ws==n)>0:
                               if mask[t,z][ws==n].mean()>=0.5 and not(np.any(border*ws==n)):
                                    mask[t,z][ws==n]=0
                        mask[t,z]=segment.detectNonregularBoundary(image.data[t,z].copy(),iterations=30,initArray=mask[t,z])
                        mask[t,z]=morphology.binary_erosion(mask[t,z],iterations=2,border_value=0)
            maskImg.data=mask
            maskImg.save(savePath+'/maskBorderImg')
    
    
    if startstep<=3 and endstep>=3:
        image=mip.load(savePath+'/img')
        timestepNo=image.data.shape[0]
        timeList=np.arange(timestepNo)
        timestep=timeList[1]
        if maskImg:
            if type(maskImg)==str:
                maskImg=mip.load(savePath+'/'+maskImg).data
            else:
                maskImg=mip.load(savePath+'/maskBorderImg').data.astype(float)
        else:
            maskImg=None
        startTime=time.process_time()
        if twoD:
            setOrigin=(0.,0.)
        else:
            setOrigin=(0.,0.,0.)
        if fftLagrangian:
            pf.TmapRegister(image,savePath=savePath,origin=setOrigin,bgrid=bgrid,bweight=1.,rms=True,startTime=0,scaleImg=fileScale,maskArray=maskImg,twoD=twoD,cyclic=False)
        else:
            pf.TmapRegister(image,savePath=savePath,origin=setOrigin,bgrid=bgrid,bweight=1.,rms=True,startTime=0,scaleImg=fileScale,maskArray=maskImg,twoD=twoD,cyclic=True)
        if anchor is not None:
            for n in range(len(anchor)):
                if len(anchor[n])==4:
                    image1=mip.load(anchor[n][2])
                    image2=mip.load(anchor[n][3])
                    pf.TmapRegister_img2img(image1,image2,savePath=savePath,fileName='manual_'+regfile_general[:-4].format(anchor[n][0],anchor[n][1]),scaleImg=fileScale)
        regTime=time.process_time()-startTime
        allTime.append(regTime)
        allTimeHeader+="registrationTime,"
        
    if startstep<=4 and endstep>=4:
        image=mip.load(savePath+'/img')
        timestepNo=image.data.shape[0]
        timeList=np.arange(timestepNo)
        timestep=timeList[1]
        timeMapList=[]
        fileList=[]
        for n in range(timestepNo-1):
            timeMapList.append([timeList[n],timeList[n+1]])
            fileList.append(imregPath+regfile_general.format(n,n+1))
        timeMapList.append([timeList[-1],timeList[0]])
        fileList.append(imregPath+regfile_general.format(timestepNo-1,0))
    
        timeMapList.append([timeList[0],timeList[-1]])
        fileList.append(imregPath+regfile_general.format(0,timestepNo-1))
        for n in range(timestepNo-1,0,-1):
            timeMapList.append([timeList[n],timeList[n-1]])
            fileList.append(imregPath+regfile_general.format(n,n-1))
    
        timeMapList2=[]
        fileList2=[]
        for n in range(timestepNo):
            if n!=0:
                timeMapList2.append([0,timeList[n]])
                fileList2.append(imregPath+regfile_general.format(0,n))
        if period is None:
            period=timestep*timestepNo
        
        solver=bfSolver.bfSolver()
        startTime=time.process_time()###
        if fftLagrangian:
            solver.addBsplineFile(fileList2+fileList[:timestepNo-1],timeMapList=timeMapList2+timeMapList[:timestepNo-1],fileScale=fileScale)
            solver.initialize(shape=finalShape,period=period,fourierTerms=fourierTerms,spacingDivision=2.)
            solver.estimateInitialwithRefTime(timestepNo-1,OrderedBsplinesList2=range(timestepNo-1,2*(timestepNo-1)),spacingDivision=2.,gap=0)
        else:
            solver.addBsplineFile(fileList,timeMapList=timeMapList,fileScale=fileScale)
            solver.initialize(shape=finalShape,period=period,fourierTerms=fourierTerms,spacingDivision=2.)
            solver.estimateInitialwithRefTime(timestepNo-1,OrderedBsplinesList2=range(timestepNo,len(fileList)-1),spacingDivision=2.,gap=0,forwardbackward=True)
        mtinitTime=time.process_time()-startTime
        solver.bsFourier.writeCoef(savePath+'/'+saveName+'_fft.txt')
        allTime.append(mtinitTime)
        allTimeHeader+="motiontrackinginitTime,"
    
    if startstep<=5 and endstep>=5:
        image=mip.load(savePath+'/img')
        timestepNo=image.data.shape[0]
        timeList=np.arange(timestepNo)
        timestep=timeList[1]
        timeMapList=[]
        fileList=[]
        for n in range(timestepNo-1):
            timeMapList.append([timeList[n],timeList[n+1]])
            fileList.append(imregPath+regfile_general.format(n,n+1))
        timeMapList.append([timeList[-1],timeList[0]])
        fileList.append(imregPath+regfile_general.format(timestepNo-1,0))
    
        timeMapList.append([timeList[0],timeList[-1]])
        fileList.append(imregPath+regfile_general.format(0,timestepNo-1))
        for n in range(timestepNo-1,0,-1):
            timeMapList.append([timeList[n],timeList[n-1]])
            fileList.append(imregPath+regfile_general.format(n,n-1))
    
        timeMapList2=[]
        fileList2=[]
        for n in range(timestepNo):
            if n!=0:
                timeMapList2.append([0,timeList[n]])
                fileList2.append(imregPath+regfile_general.format(0,n))
        if period is None:
            period=timestep*timestepNo
        
        solver=bfSolver.bfSolver()
        if fftLagrangian:
            solver.addBsplineFile(fileList[:(timestepNo-1)]+[fileList2[-1]],timeMapList=timeMapList[:(timestepNo-1)]+[timeMapList2[-1]],fileScale=fileScale)
        else:
            solver.addBsplineFile(fileList,timeMapList=timeMapList,fileScale=fileScale)
        fft_bsFourier=BsplineFourier.BsplineFourier(savePath+'/'+saveName+'_fft.txt')
        for n in range(len(anchor)):
            if len(anchor[n])==3:
                solver.addBsplineFile(anchor[n][2],timeMapList=[timeList[anchor[n][0]],timeList[anchor[n][1]]],weightList=10.,fileScale=fileScale)
            else:
                solver.addBsplineFile(imregPath+'manual_'+regfile_general.format(anchor[n][0],anchor[n][1]),timeMapList=[timeList[anchor[n][0]],timeList[anchor[n][1]]],weightList=10.,fileScale=fileScale)
                
        solver.bsFourier=fft_bsFourier
        if type(finalShape)!=type(None):
            if np.any(np.array(finalShape)!=np.array(solver.bsFourier.coef.shape)):
                solver.bsFourier=solver.bsFourier.reshape(finalShape)
        solver.initialize(shape=solver.bsFourier.coef.shape,period=period,fourierTerms=fourierTerms,spacingDivision=2.)#,gap=0)
        startTime=time.process_time()###
        solver.solve(maxIteration=1000,convergence=0.5,reportevery=600,method='pointbypoint',tempSave=savePath+'/'+saveName+'_f'+str(fourierTerms)+'_samplingResults.txt')
        mtTime=time.process_time()-startTime
        solver.bsFourier.writeCoef(savePath+'/'+saveName+'_f'+str(fourierTerms)+'.txt')
        solver.writeSamplingResults(savePath+'/'+saveName+'_f'+str(fourierTerms)+'_samplingResults.txt')
        allTime.append(mtTime)
        allTimeHeader+="batchcorrectionTime,"
    if startstep<=6 and endstep>=6:
        image=mip.load(savePath+'/img')
        timestepNo=image.data.shape[0]
        
        if twoD:
            imageSize=image.data.shape[::-1][:2]
            imageSpacing=[image.dimlen['x'],image.dimlen['y']]
        else:
            imageSize=list(image.data.shape[::-1][:3])
            imageSpacing=[image.dimlen['x'],image.dimlen['y'],image.dimlen['z']]
        regridTime=[]
        for t in getCompoundTimeList:
            solver=bfSolver.bfSolver()
            solver.loadSamplingResults(savePath+'/'+saveName+'_f'+str(fourierTerms)+'_samplingResults.txt')
            if t is None:
                solver.bsFourier=BsplineFourier.BsplineFourier(savePath+'/'+saveName+'_fft.txt')
            else:
                solver.bsFourier=BsplineFourier.BsplineFourier(savePath+'/'+saveName+'_f'+str(fourierTerms)+'.txt')
            startTime=time.process_time()###
            solver.bsFourier.regridToTime(solver.points,solver.pointsCoef,t)
            regridTime.append(time.process_time()-startTime)
            if t is None:
                solver.bsFourier.writeCoef(savePath+'/'+saveName+'_f'+str(fourierTerms)+'.txt')
            else:
                solver.bsFourier.writeCoef(savePath+'/'+saveName+'_f'+str(fourierTerms)+'_t'+str(t)+'.txt')
                for t2 in range(timestepNo):
                    if t!=t2:
                        solver.bsFourier.writeBspline(t2,savePath+'/bsfTransform/t'+str(t)+'to'+str(t2)+'.txt',imageSize=imageSize,imageSpacing=imageSpacing)
            del solver
        allTime.append(np.mean(regridTime))
        allTimeHeader+="regridTime,"
    if len(allTime)>0:
        allTime=np.array(allTime)
        allTime=allTime/np.prod(image.data.shape[1:])*(10.**6.)
        np.savetxt(savePath+'/computationalTime_motion.txt',allTime.reshape((1,-1)),header=allTimeHeader+'\n'+' microsecond per pixel ,image shape = '+repr(image.data.shape))
    if startstep<=7 and endstep>=7:
        allTime=[]
        allTimeHeader=""
        image=mip.load(savePath+'/img')
        image.data=image.data.astype(float)
        timestepNo=image.data.shape[0]
        if maskImg:
            maskImg=mip.load(savePath+'/maskBorderImg')
            maskImg.data=maskImg.data.astype(float)
        
        syncTime=[]
        syncMaskTime=[]
        SACTime=[]
        maxTime=[]
        meanTime=[]
        waveletTime=[]
        SACmaxTime=[]
        SACmeanTime=[]
        for t in getCompoundTimeList:
            
            syncImg=image.clone()
            if maskImg:
                syncMask=maskImg.clone()
            syncMaskTime.append([])
            syncTime.append([])
            for t2 in range(image.data.shape[0]):
                if t2!=t:
                    startTime=time.process_time()###
                    syncImg.data[t2]=pf.transform_img2img(image.data[t2].copy(),savePath+r'/bsfTransform/t'+str(t)+'to'+str(t2)+'.txt',savePath=savePath+'/'+str(t),scale=np.array([image.dimlen['x'],image.dimlen['y'],image.dimlen['z']]))
                    syncTime[-1].append(time.process_time()-startTime)###
                    if maskImg:
                        startTime=time.process_time()###
                        syncMask.data[t2]=pf.transform_img2img(maskImg.data[t2].copy(),savePath+r'/bsfTransform/t'+str(t)+'to'+str(t2)+'.txt',savePath=savePath+'/'+str(t),scale=np.array([maskImg.dimlen['x'],maskImg.dimlen['y'],maskImg.dimlen['z']]))
                        syncMaskTime[-1].append(time.process_time()-startTime)###
            syncTime[-1]=np.sum(syncTime[-1])
            syncMaskTime[-1]=np.sum(syncMaskTime[-1])
            syncImg.save(savePath+'/SRimg_sync_t'+str(t))
            
            if maskImg:
                syncMask.data=np.maximum(0,np.minimum(1,syncMask.data))
                syncMask.save(savePath+'/maskBorderImg_sync_t'+str(t))
            
        
            syncImg=mip.load(savePath+'/SRimg_sync_t'+str(t))
            syncImg.data=syncImg.data.astype(float)
            if maskImg:
                syncMask=mip.load(savePath+'/maskBorderImg_sync_t'+str(t))
                syncMask.rearrangeDim(syncImg.dim)
            
            for compoundstr in compoundSchemeList:
                
                syncImg_temp=syncImg.clone()
                if compoundstr!='wavelet' and maskImg:
                    syncImg_temp.data[syncMask.data<0.5]=float('nan')
                if compoundstr=='SAC':
                    startTime=time.process_time()###
                    compoundImg_temp=pf.compound(syncImg_temp,scheme='SAC',schemeArgs=[0.5,None])
                    SACTime.append(time.process_time()-startTime)###
                    for SACargs in [['mean',np.nanmean],['max',np.nanmax]]:
                        compoundImg=compoundImg_temp.clone()
                        startTime=time.process_time()###
                        compoundImg.data=SACargs[1](compoundImg.data,axis=-1)
                        if SACargs[0]=='mean':
                            SACmeanTime.append(time.process_time()-startTime)
                        elif SACargs[0]=='max':
                            SACmaxTime.append(time.process_time()-startTime)
                        compoundImg.data[np.isnan(compoundImg.data)]=0
                        compoundImg.save(savePath+'/SRimg_SAC'+SACargs[0]+'_t'+str(t))
                else:
                    startTime=time.process_time()###
                    compoundImg=pf.compound(syncImg_temp,scheme=compoundstr)
                    if compoundstr=='maximum':
                        maxTime.append(time.process_time()-startTime)
                    elif compoundstr=='mean':
                        meanTime.append(time.process_time()-startTime)
                    elif compoundstr=='wavelet':
                        waveletTime.append(time.process_time()-startTime)
                    if compoundstr!='wavelet':
                        compoundImg.data[np.isnan(compoundImg.data)]=0
                    compoundImg.save(savePath+'/SRimg_'+compoundstr+'_t'+str(t))

    
        allTime=np.array([np.mean(syncTime),np.mean(syncMaskTime),np.mean(SACTime),np.mean(SACTime)+np.mean(SACmaxTime),np.mean(SACTime)+np.mean(SACmeanTime),np.mean(maxTime),np.mean(meanTime),np.mean(waveletTime)])
        allTime=allTime/np.prod(image.data.shape[1:])*(10.**6.)
        np.savetxt(savePath+'/computationalTime.txt',allTime.reshape((1,-1)),header='syncTime,syncMaskTime,SACTime,SACmaxTime,SACmeanTime,maxTime,meanTime,waveletTime\n'+' microsecond per pixel ,image shape = '+repr(image.data.shape))
    
