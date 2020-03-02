'''
File: motionCorrect.py
Description: correct rigid motion in ultrasound scan and dim down non-random motion
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: w.x.chan@gmail.com         04FEB2020           - Created
  Author: w.x.chan@gmail.com         27FEB2020           - v2.6.3
                                                            -add output of raw image without fluid space
  Author: w.x.chan@gmail.com         02MAR2020           - v2.7.8
                                                            -debug focusSlice of raw image and raw image without fluid space
  
Requirements:
    numpy
    scipy
    medImgProc
Known Bug:
    None
All rights reserved.
'''
_version='2.7.8'
import logging
logger = logging.getLogger(__name__)

import os
import sys

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import morphology as morph
from scipy.ndimage import shift
from scipy.special import comb
import medImgProc as mip
import medImgProc.processFunc as pf
import medImgProc.pointSpeckleProc as psp

def NCC(array1,array2,zero=False):
    std1=np.std(array1)
    std2=np.std(array2)
    if std1==0 or std2==0:
        return float('-inf')
    if zero:
        mu1=np.mean(array1)
        mu2=np.mean(array2)
    else:
        mu1=0.
        mu2=0.
    return 1./array1.size/std1/std2*np.sum((array1-mu1)*(array2-mu2))
def ZNCC(array1,array2):
    return NCC(array1,array2,zero=True)
def IRMS(array1,array2):
    return -np.sqrt(np.mean((array1-array2)**2.))
def getMask(shape,maskslice,border):
    mask=np.ones(shape)
    mask[maskslice]=0
    mask[-border:]=0
    mask[:border]=0
    mask[:,-border:]=0
    mask[:,:border]=0
    return mask
def detectPeriod(cyclicTimeStepRef,imageArray,start=0,focusSlice=None,forcePeriod=False):
    if type(focusSlice)==type(None):
        focusSlice=[]
        for n in range(len(imageArray.shape)-1):
            focusSlice.append(slice(None))
        focusSlice=tuple(focusSlice)
    inphase0=[start]
    currentCyclic=cyclicTimeStepRef
    while (imageArray.shape[0]-inphase0[-1])>(currentCyclic+1):
        similarityMax=float('-inf')
        similarityMaxInd=0
        ssss=[]
        if len(inphase0)==1:
            if forcePeriod:
                currentCyclic=currentCyclic
                inphase0.append(inphase0[-1]+currentCyclic)
                continue
            else:
                searchn=range(min(-1,-int(0.1*cyclicTimeStepRef)),max(1,int(0.1*cyclicTimeStepRef))+1)
        else:
            searchn=range(min(-1,-int(0.05*cyclicTimeStepRef)),max(1,int(0.05*cyclicTimeStepRef))+1)
        for n in searchn:
            if (inphase0[-1]+currentCyclic+n)>=imageArray.shape[0]:
                continue
            similarity=ZNCC(imageArray[inphase0[0]][focusSlice],imageArray[inphase0[-1]+currentCyclic+n][focusSlice])
            ssss.append(similarity)
            if similarity>similarityMax:
                similarityMax=similarity
                similarityMaxInd=n
        currentCyclic=currentCyclic+similarityMaxInd
        inphase0.append(inphase0[-1]+currentCyclic)

    refCycleNumber=0
    firstCyclicTimeStep=inphase0[1]-inphase0[0]
    currentCyclic=firstCyclicTimeStep
    while inphase0[0]>=(currentCyclic+1):
        similarityMax=float('-inf')
        similarityMaxInd=0
        ssss=[]
        searchn=range(min(-1,-int(0.05*cyclicTimeStepRef)),max(1,int(0.05*cyclicTimeStepRef))+1)
        for n in searchn:
            if (inphase0[0]-currentCyclic-n)<0:
                continue
            similarity=ZNCC(imageArray[inphase0[refCycleNumber]][focusSlice],imageArray[inphase0[0]-currentCyclic-n][focusSlice])
            ssss.append(similarity)
            if similarity>similarityMax:
                similarityMax=similarity
                similarityMaxInd=n
        currentCyclic=currentCyclic+similarityMaxInd
        inphase0.insert(0,inphase0[0]-currentCyclic)
        refCycleNumber+=1

    cyclicSlice=[]
    cyc_temp=np.array(inphase0)[1:]-np.array(inphase0)[:-1]
    for n in range(firstCyclicTimeStep):
        phase=n/float(firstCyclicTimeStep)
        cyclicSlice.append(phase*cyc_temp+np.array(inphase0)[:-1])
    cyclicSlice=np.roll(np.array(cyclicSlice),-refCycleNumber,axis=1)
    for nn in range(len(cyclicSlice)):
        for n in range(1,cyclicSlice.shape[1]):
            if np.abs(cyclicSlice[nn,n]-np.around(cyclicSlice[nn,n]))>0.3:
                if ZNCC(imageArray[int(np.around(cyclicSlice[nn,0]))][focusSlice],imageArray[int(cyclicSlice[nn,n])][focusSlice])>=ZNCC(imageArray[int(np.around(cyclicSlice[nn,0]))][focusSlice],imageArray[int(cyclicSlice[nn,n])+1][focusSlice]):
                    cyclicSlice[nn,n]=int(cyclicSlice[nn,n])
                else:
                    cyclicSlice[nn,n]=int(cyclicSlice[nn,n])+1
    cyclicSlice=np.around(cyclicSlice).astype(int)     
    return cyclicSlice
def motionCorrect_generalRigid(savePath,case,maskslice,focusSlice,guessPeriod,correlateSlice,includeRotate=False,border=60,avicheck=False):
    #step 1
    ##########general rigid motion correction ###############
    img=mip.load(savePath+'/cropped_'+case)
    mask=getMask(img.data.shape[1:],maskslice,border)
    
    newImg,translateIndexTemp=pf.alignAxes_translate(img,['y','x'],{'t':int(img.data.shape[0]/2)},dimSlice={},fixedRef=False,initTranslate=False,includeRotate=includeRotate,mask=mask)
    translateIndexTemp=np.array(translateIndexTemp)
    translateIndex=np.zeros((img.data.shape[0],translateIndexTemp.shape[1]-1))
    translateIndex[translateIndexTemp[:,0].astype(int)]=translateIndexTemp[:,1:].copy()
    translateIndex=np.sum(translateIndex**2.,axis=1)
    translateIndexmode=translateIndex[:(-guessPeriod*2-1)].copy()/guessPeriod
    for n in range(1,guessPeriod*2+1):
        if n<int(guessPeriod*0.5):
            factor=1./guessPeriod
        elif n>=(int(guessPeriod*0.5)-1) and n>=(int(guessPeriod*0.5)+1):
            factor=guessPeriod*2
        elif n>int(guessPeriod*0.5)+3+guessPeriod:
            factor=1./guessPeriod
        translateIndexmode+=translateIndex[n:(-guessPeriod*2-1+n)]*factor
    stableFrame=np.argmin(translateIndexmode)+int(guessPeriod*0.5)
    np.savetxt(savePath+'/'+case+'/stableFrame.txt',[stableFrame])
    newImg,translateIndexTemp=pf.alignAxes_translate(img,['y','x'],{'t':stableFrame},dimSlice={},fixedRef=True,initTranslate=True,includeRotate=includeRotate,mask=mask)
    translateIndex=np.zeros((img.data.shape[0],translateIndexTemp.shape[1]-1))
    translateIndex[translateIndexTemp[:,0].astype(int)]=translateIndexTemp[:,1:].copy()
    newImg.data=np.maximum(0,newImg.data)
    newImg.save(savePath+'/'+case+'/afterRigidImg')
    if avicheck:
        newImg.mimwrite2D(savePath+'/'+case)
    np.savetxt(savePath+'/'+case+'/translateParameter.txt',translateIndex, header='[y,x] start of stable frame '+str(stableFrame))
    img2=img.clone()
    img2.data=img.data[stableFrame:stableFrame+guessPeriod+1].copy()
    img2.changeColorFormat()
    
    img2.data[...,1]=0
    newslice=(slice(None),*focusSlice[1:])
    img2.data[...,1][newslice]=img2.data[...,0][newslice]
    img2.data[...,2]=0
    newmaskslice=(slice(None),*maskslice)
    img2.data[...,2][newmaskslice]=img2.data[...,0][newmaskslice]
    newslice=(slice(None),*correlateSlice[1:])
    img2.data[...,0][newslice]=0
    img2.mimwrite2D(savePath+'/'+case+'/checkstart',color=1)

def motionCorrect_syncNonRigid(savePath,case,maskslice,focusSlice,guessPeriod,size,correlateSlice,nonCardiacMotion=True,includeRotate=False,border=60,bgridFactor=4.,forcePeriod=False,highIntensityFilter=None,lowIntensityFilter=None,avicheck=False):
    if nonCardiacMotion:
        img=mip.load(savePath+'/'+case+'/afterRigidImg')
        stableFrame=int(np.loadtxt(savePath+'/'+case+'/stableFrame.txt'))
    else:
        img=mip.load(savePath+'/cropped_'+case)
        stableFrame=1
    mask=getMask(img.data.shape[1:],maskslice,border)
    img2=img.clone()
    img2.data=img2.data[focusSlice]
    newImg=img2.data.copy()
    
    fullperiod=detectPeriod(guessPeriod,img.data[correlateSlice].copy(),start=stableFrame,forcePeriod=forcePeriod)
    np.savetxt(savePath+'/'+case+'/fullperiod.txt',fullperiod)
    if avicheck:
        imgg=[]
        for n in range(len(fullperiod[:,0])):
            imgg.append(img.clone())
            imgg[-1].data=img.data[fullperiod[n].astype(int)].copy()
        imgg=mip.stackImage(imgg,'h')
        imgg.mimwrite2D(savePath+'/'+case+'/checkstart2')
        del imgg
    if nonCardiacMotion:
        runImgReg=True
        if os.path.isfile(savePath+'/'+case+'/transform/interCyclefullperiod.txt'):
            if np.all(fullperiod==np.loadtxt(savePath+'/'+case+'/transform/interCyclefullperiod.txt').astype(int)):
                runImgReg=False
        if runImgReg:
            newImg=pf.cyclicNonRigidCorrection(fullperiod,newImg,np.mean(size)*bgridFactor*4.*np.ones(len(size)),nonRigidSavePath=savePath+'/'+case,bgridFactor=bgridFactor,inverse=True,returnSyncPhase=False)
            np.savetxt(savePath+'/'+case+'/transform/interCyclefullperiod.txt',fullperiod)
    else:
        newImg=None

    if includeRotate:
        oriImg=mip.load(savePath+'/cropped_'+case)
        translateIndex=np.loadtxt(savePath+'/'+case+'/translateParameter.txt')
        img3=psp.getSpecklePoint(oriImg,size,prefilter='gaussian')
        for t in range(img3.data.shape[0]):
            if np.any(translateIndex[t]):
                img3.data[t]=pf.translateArray(img3.data[t],translateIndex[t],includeRotate,0,order=0)
        img3=psp.getSpecklePoint(img3,np.ones(len(size)),prefilter='gaussian')
        del oriImg
    else:
        img3=psp.getSpecklePoint(img2,size,prefilter='gaussian')
    img3.data=np.maximum(0,img3.data)
    img4=mip.stackImage([img3.clone()],'h')
    img4.data=np.zeros((*fullperiod.shape,*img3.data.shape[1:]))
    img4.data[:,0]=img3.data[fullperiod[:,0]].copy()
    for n in range(fullperiod.shape[0]):
        mInd=1
        for m in fullperiod[n,1:]:
            if nonCardiacMotion:
                file=savePath+'/'+case+'/transform/t'+str(m)+'to'+str(fullperiod[n,0])+'_0.txt'
                pos=np.transpose(np.nonzero(img3.data[m]))[:,::-1]
                val=img3.data[m][np.nonzero(img3.data[m])].copy()
                newpos=pf.transform_img2img(pos,file,savePath=savePath+'/'+case)
                newpos=np.around(newpos).astype(int)
                get=np.all(np.logical_and(newpos>=0,newpos<np.array(img3.data.shape[1:])[::-1]),axis=-1)
                newpos=newpos[get]
                val=val[get]
                img4.data[n,mInd][tuple(newpos[:,::-1].T)]=val.copy()
            else:
                img4.data[n,mInd]=img3.data[m].copy()
            mInd+=1
            
    #apply intensitycap
    speckleIntensity75=np.percentile(img4.data[img4.data>0],75)
    speckleIntensityMedian=np.percentile(img4.data[img4.data>0],50)
    speckleIntensity25=np.percentile(img4.data[img4.data>0],25)
    speckleIntensityRemoveUpper=speckleIntensity75*2.5-1.5*speckleIntensityMedian
    speckleIntensityRemoveLower=speckleIntensity25*2.5-1.5*speckleIntensityMedian
    np.savetxt(savePath+'/'+case+'/speckleIntensityPercentile.txt',[speckleIntensity75,speckleIntensityMedian,speckleIntensity25])


    #img4.data=np.minimum(speckleIntensity75,img4.data)
    img4.data[img4.data>speckleIntensityRemoveUpper]=1
    img4.data[np.logical_and(img4.data>=1,img4.data<speckleIntensityRemoveLower)]=1
    if highIntensityFilter:
        img4.data[img4.data>highIntensityFilter]=2*highIntensityFilter-img4.data[img4.data>highIntensityFilter]
    if lowIntensityFilter:
        img4.data[img4.data<lowIntensityFilter]=1
    img4.data=np.maximum(0,img4.data)

    if avicheck:
        img4.mimwrite2D(savePath+'/'+case+'/nonrigid')
    img4.save(savePath+'/'+case+'/nonrigid/img')
def motionCorrect_dimNonrandom(savePath,case,size,useCorrDet=True,reduceNonRandomMode=0,dimSigmaFactor=1.,highIntensityFilter=None,avicheck=False):
    img4=mip.load(savePath+'/'+case+'/nonrigid/img')
                
    density=float(img4.data[img4.data>1].size)/img4.data.size
    fullperiod=np.loadtxt(savePath+'/'+case+'/fullperiod.txt').astype(int)
    if reduceNonRandomMode>0:
        multiplier=[psp.reduceNonRandom(img4,np.array(size)*2.,densityApprox=density,dimSigmaFactor=-dimSigmaFactor,average=False,useCorrDet=False)]
        varySigma=max(np.array(size).max()*2.,1)
        for n in range(reduceNonRandomMode):
            normVal_temp=(1./gaussian_filter(np.ones(np.ones(len(size)).astype(int)), np.ones(len(size))*varySigma,mode='constant')).max()
            mean=normVal_temp*density*(img4.data.shape[img4.dim.index('t')]-1)+1
            if mean<2 or varySigma<0.6:
                break
            logger.info('reducing non-random with sigma ='+repr(varySigma))
            multiplier.append(psp.reduceNonRandom(img4,np.ones(len(size))*varySigma,densityApprox=density,dimSigmaFactor=-dimSigmaFactor,average=False,useCorrDet=False))
            varySigma=varySigma*0.5
        img5=img4.clone()
        img5.data*=np.prod(np.array(multiplier),axis=0)**(1./len(multiplier))
        if useCorrDet:
            img5=psp.reduceNonRandom(img5,np.array(size)*2.,densityApprox=density,dimSigmaFactor=0,average=False,useCorrDet=useCorrDet)
    else:
        img5=psp.reduceNonRandom(img4,np.array(size)*2.,densityApprox=density,dimSigmaFactor=dimSigmaFactor,average=False,useCorrDet=useCorrDet)
    
    #remove very low intensity to speed up
    if not(highIntensityFilter):
        img5.data[img5.data<(0.1*np.percentile(img5.data[img5.data>0],99))]=0
    img5.data=np.maximum(0,img5.data)
    if avicheck:
        img5.mimwrite2D(savePath+'/'+case+'/fluidspeckles')
    img5.save(savePath+'/'+case+'/fluidspeckles/img')
    np.savetxt(savePath+'/'+case+'/density.txt',[density])
    dimStd=1+density*(fullperiod.shape[1]-1)*(1./gaussian_filter(np.ones(np.ones(len(size)).astype(int)), size,mode='constant')).max()
    np.savetxt(savePath+'/'+case+'/dimStd.txt',[dimStd])
    img6=psp.spreadSpeckle(img5,size,overlay=True,overlayFunc=np.max,averageSigma=True)
    img6.data=np.maximum(0,img6.data)
    img6.save(savePath+'/'+case+'/pointspecklecorrection/img')
    if avicheck:
        img6.mimwrite2D(savePath+'/'+case+'/pointspecklecorrection',axes=('h','y','x'))
def motionCorrect_intraCycleCompound(savePath,case,maskslice,focusSlice,size,nonCardiacMotion=True,border=60,bgridFactor=4.,highErrorDim=True,highIntensityFilter=None,avicheck=False):
    if nonCardiacMotion:
        img=mip.load(savePath+'/'+case+'/afterRigidImg')
    else:
        img=mip.load(savePath+'/cropped_'+case)
    mask=getMask(img.data.shape[1:],maskslice,border)
    img2=img.clone()
    img2.data=img2.data[focusSlice]

    density=np.loadtxt(savePath+'/'+case+'/density.txt')
    logger.info('Density = {0:.3e}'.format(density))
    fullperiod=np.loadtxt(savePath+'/'+case+'/fullperiod.txt').astype(int)

    img5=mip.load(savePath+'/'+case+'/fluidspeckles/img')

    runImgReg=True
    if os.path.isfile(savePath+'/'+case+'/transform/intraCyclefullperiod.txt'):
        if np.all(fullperiod==np.loadtxt(savePath+'/'+case+'/transform/intraCyclefullperiod.txt').astype(int)):
            runImgReg=False
    if runImgReg:
        pf.nonRigidRegistration(savePath+'/'+case,img2.data[fullperiod[:,0]],np.mean(size)*bgridFactor*np.ones(len(size)),full=False)
        np.savetxt(savePath+'/'+case+'/transform/intraCyclefullperiod.txt',fullperiod)
        
    img55=img5.clone()
    img55.removeDim('t')
    img55.data=img5.data.max(axis=1)
    
    img99=[]
    newDensity=0
    for n in range(1,fullperiod.shape[1]):
        newDensity-=comb(fullperiod.shape[1],n)*(-density)**n
    np.savetxt(savePath+'/'+case+'/intermittentdensity.txt',[newDensity])

    
    if highErrorDim==True or highErrorDim==0:
        e=1./(1.+np.arange(fullperiod.shape[0])/np.arange(fullperiod.shape[0],0,-1))
        acc=1/np.maximum(1,(e*(1-e)*fullperiod.shape[0]))
    elif isinstance(highErrorDim,int):
        e=1./(1.+np.arange(fullperiod.shape[0])/np.arange(fullperiod.shape[0],0,-1))
        e=e*(1-e)*fullperiod.shape[0]
        if highErrorDim<int((len(e)+1)/2):
            acc=1/np.maximum(1,e/e[highErrorDim])
        else:
            acc=np.ones(fullperiod.shape[0])
    else:
        acc=np.ones(fullperiod.shape[0])
    nnd=1-np.prod(1-acc*newDensity)
    np.savetxt(savePath+'/'+case+'/finaldensity.txt',[nnd])
    
    for n in range(fullperiod.shape[0]):
        img7=img55.clone()
        img7.data=psp.speckleTransform(img55.data[n].copy(),savePath+'/'+case+'/transform',n,totalTimeSteps=fullperiod.shape[0],highErrorDim=highErrorDim).copy()
        img99.append(img7.clone())
    img99=mip.stackImage(img99,'t')
    if avicheck:
        trans_img=img.clone()
        trans_img.data=trans_img.data[fullperiod[:,0]]
        trans_img.changeColorFormat()
        trans_img.data[...,0][img99.data[0]>0]=255
        trans_img.data[...,1][img99.data[0]>0]=0
        trans_img.data[...,2][img99.data[0]>0]=0
        trans_img.mimwrite2D(savePath+'/'+case+'/transform/checkt0',color=1)
        del trans_img
    #remove very low intensity to speed up
    if not(highIntensityFilter):
        img99.data[img99.data<(0.1*np.percentile(img99.data[img99.data>0],99))]=0
    
    img99.save(savePath+'/'+case+'/segmentationALL/img')
    if avicheck:
        img99.mimwrite2D(savePath+'/'+case+'/segmentationALL')
def motionCorrect_getSegmentation(savePath,case,size,finalSpread=0.001):
    newDensity=np.loadtxt(savePath+'/'+case+'/intermittentdensity.txt')
    nnd=np.loadtxt(savePath+'/'+case+'/finaldensity.txt')
    sspread=np.sqrt(np.log(finalSpread)/np.log(1.-nnd)/np.pi)
    logger.info('spread='+repr(sspread))
    img99=mip.load(savePath+'/'+case+'/segmentationALL/img')
    
    
    img9=psp.spreadSpeckle(img99,np.array([sspread,sspread]),overlay=True,overlayFunc=np.max,averageSigma=True,dim='t')
    img9.save(savePath+'/'+case+'/segmentation/img')
    img9.mimwrite2D(savePath+'/'+case+'/segmentation',axes=('h','y','x'))
    del img9

    
    img10=img99.clone()
    img10.removeDim('t')
    img10.data=np.max(img99.data,axis=0)
    imgG=img10.clone()
    imgG.data=gaussian_filter(imgG.data,[0,sspread,sspread])
    img11=img10.clone()
    for k in range(50):
        for n in range(img10.data.shape[0]):
            img11.data[n]=morph.binary_closing(img10.data[n],iterations=1).astype('uint8')*255
        if np.all(img11.data==img10.data):
            break
        else:
            img10.data=img11.data.copy()
    for n in range(img11.data.shape[0]):
        img11.data[n]=morph.binary_opening(img11.data[n],iterations=1).astype('uint8')*255
    img11.data[:]=0
    img11.data[img10.data>1]=imgG.data[img10.data>1]
    img11.data=gaussian_filter(img11.data,[0,sspread,sspread])
    img11.data=img11.data/np.percentile(img11.data[img11.data>0],99)*255
    img11.save(savePath+'/'+case+'/segmentation_closing/img')
    img11.mimwrite2D(savePath+'/'+case+'/segmentation_closing',axes=('h','y','x'))
def combineAndSyncSlices(savePath,focusSlice,guessPeriod,stackstr='',translateToStack=True,caseNameFormat='V{0:02d}',nonCardiacMotion=True,border=60):
    try:
        transParaYX=np.loadtxt(savePath+'/sliceSyncPara.txt')
        imgCorrect=mip.load(savePath+'/beforeRegistrationBetweenSlices')
        img=mip.load(savePath+'/spatialadjustmentBetweenSlices')
        logger.info('Read previous spatial translation')
    except:
        try:
            imgCorrect=mip.load(savePath+'/beforeRegistrationBetweenSlices')
            img=mip.load(savePath+'/fullImageStack')
        except:
            img=[]
            imgCorrect=[]
            start=False
            for n in range(1000):
                try:
                    if nonCardiacMotion:
                        imgtemp=mip.load(savePath+'/'+caseNameFormat.format(n)+'/afterRigidImg')
                    else:
                        imgtemp=mip.load(savePath+'/cropped_'+caseNameFormat.format(n))
                    imgCorrected=mip.load(savePath+'/'+caseNameFormat.format(n)+'/segmentation'+stackstr+'/img')
                    start=True
                except:
                    logger.info('File not found: '+savePath+'/'+caseNameFormat.format(n)+'/segmentation'+stackstr+'/img')
                    if start:
                        break
                else:
                    fullperiod=np.loadtxt(savePath+'/'+caseNameFormat.format(n)+'/fullperiod.txt').astype(int)

                    imgtemp.data=imgtemp.data[focusSlice[0]][fullperiod[:,0]]
                    if imgtemp.data.shape[0]!=(guessPeriod+1):
                        imgtemp.data=np.concatenate((imgtemp.data,imgtemp.data[:1]),axis=0)
                        imgtemp.stretch({'t':(guessPeriod+2)},scheme='cubic')
                        imgtemp.data=imgtemp.data[:(guessPeriod+1)]
                        imgCorrected.data=np.concatenate((imgCorrected.data,imgCorrected.data[:1]),axis=0)
                        imgCorrected.stretch({'h':(guessPeriod+2)},scheme='cubic')
                        imgCorrected.data=imgCorrected.data[:(guessPeriod+1)]
                    img.append(imgtemp.clone())
                    imgCorrect.append(imgCorrected.clone())
            img=mip.stackImage(img,'z')
            img.save(savePath+'/fullImageStack')
            imgCorrect=mip.stackImage(imgCorrect,'z')
            imgCorrect.save(savePath+'/beforeRegistrationBetweenSlices')
        selectSlice=int(img.data.shape[0]/2)

        transParaYX=np.zeros((img.data.shape[0],2))
        #correct motion between slice
        if nonCardiacMotion and translateToStack:
            
            newMask=np.ones(img.data.shape[2:])
            newMask[focusSlice[1:]]=0.5
            newMask[-int(border/3):]=0
            newMask[:int(border/3)]=0
            newMask[:,-int(border/3):]=0
            newMask[:,:int(border/3)]=0
            newMask=np.tile(newMask,(guessPeriod+1,1,1))
            #newMask=None
            #correctionArray=np.minimum(imgCorrect.data,np.percentile(imgCorrect.data,99))
            #correctionArray=1-correctionArray/correctionArray.max()
            #imgFluidEmpty=img.clone()
            #imgFluidEmpty.data[(slice(None),slice(None),*focusSlice[1:])]*=correctionArray
            #imgFluidEmpty.mimwrite2D(savePath+'/ImageWithoutFluid'+stackstr,axes=('h','y','x'))
            img,tr=pf.alignAxes_translate(img,['y','x'],{'z':selectSlice},dimSlice={},nres=1,fixedRef=False,initTranslate=False,translateLimit=0.2,includeRotate=False,mask=newMask)
            img.save(savePath+'/spatialadjustmentBetweenSlices')
            del newMask
            transParaYX_temp=np.zeros((img.data.shape[0],2))
            transParaYX_temp[tr[:,0].astype(int)]=tr[:,1:].copy()
            for n in range(len(transParaYX_temp)):
                if n<selectSlice:
                    transParaYX[n]=transParaYX_temp[n:(selectSlice+1)].sum(axis=0)
                elif n>selectSlice:
                    transParaYX[n]=transParaYX_temp[selectSlice:(n+1)].sum(axis=0)
            del transParaYX_temp
        np.savetxt(savePath+'/sliceSyncPara.txt',transParaYX,header=' [y,x]')

    if  transParaYX.shape[1]<3 or np.all(transParaYX[:,0]==0):
        newfocusSlice=(slice(None),slice(None),*focusSlice[1:])

        if nonCardiacMotion and translateToStack:
            imgCorrect_temp=imgCorrect.clone()
            imgCorrect_temp.data=np.zeros(img.data.shape)
            imgCorrect_temp.data[newfocusSlice]=imgCorrect.data.copy()
            for z in range(imgCorrect_temp.data.shape[0]):
                if np.any(transParaYX[z]):
                    imgCorrect_temp.data[z]=pf.translateArray(imgCorrect_temp.data[z],transParaYX[z],False,0)
            imgCorrect.data=imgCorrect_temp.data[newfocusSlice].copy()

        img.data=img.data[newfocusSlice]
        img.data=np.tile(img.data,(1,3,1,1))
        
        selectSlice=int(img.data.shape[0]/2)
        
        XcorWeight=imgCorrect.clone()
        XcorWeight.removeDim('z',selectSlice)
        XcorWeight.data=1-np.maximum(0,np.minimum(1,XcorWeight.data/XcorWeight.data.max()/np.exp(-0.5)))
        
        cyclicmask=np.zeros(img.data.shape[1:])
        cyclicmask[(guessPeriod+1):(2*(guessPeriod+1))]=1
        cyclicmask[slice((guessPeriod+1),(2*(guessPeriod+1)))]*=XcorWeight.data

        img2,tr=pf.alignAxes_translate(img,['t'],{'z':selectSlice},dimSlice={},fixedRef=False,nres=3,initTranslate=False,includeRotate=False,mask=cyclicmask)
        
        tr=np.array(tr)

        transParaT=np.zeros(img.data.shape[0])
        transParaT[tr[:,0].astype(int)]=tr[:,1].copy()
        transPara=np.zeros((img.data.shape[0],3))
        for n in range(len(transParaT)):
            if n<selectSlice:
                transPara[n,0]=transParaT[n:(selectSlice+1)].sum()
                while transPara[n,0]>(guessPeriod+1):
                    transPara[n,0]-=(guessPeriod+1)
                while transPara[n,0]<-(guessPeriod+1):
                    transPara[n,0]+=(guessPeriod+1)
            elif n>selectSlice:
                transPara[n,0]=transParaT[selectSlice:(n+1)].sum()
                while transPara[n,0]>(guessPeriod+1):
                    transPara[n,0]-=(guessPeriod+1)
                while transPara[n,0]<-(guessPeriod+1):
                    transPara[n,0]+=(guessPeriod+1)
        transPara[:,1:]=transParaYX.copy()
        np.savetxt(savePath+'/sliceSyncPara.txt',transPara,header=' [t,y,x]')
    else:
        logger.info('Use previous phase Sync')
        transPara=transParaYX.copy()

    imgCorrect=mip.load(savePath+'/beforeRegistrationBetweenSlices')
    img=mip.load(savePath+'/fullImageStack')
    selectSlice=int(imgCorrect.data.shape[0]/2)
    img.data=np.tile(img.data,(1,3,1,1))
    imgCorrect.data=np.tile(imgCorrect.data,(1,3,1,1))
    for n in range(len(transPara)):
        if n<selectSlice:
            imgCorrect.data[n]=pf.translateArray(imgCorrect.data[n],transPara[n],False,0)
            img.data[n]=pf.translateArray(img.data[n],transPara[n],False,0)
        elif n>selectSlice:
            imgCorrect.data[n]=pf.translateArray(imgCorrect.data[n],transPara[n],False,0)
            img.data[n]=pf.translateArray(img.data[n],transPara[n],False,0)
    imgCorrect.data=imgCorrect.data[:,(guessPeriod+1):(2*(guessPeriod+1))]
    img.data=img.data[:,(guessPeriod+1):(2*(guessPeriod+1))]
    newfocusSlice=(slice(None),slice(None),*focusSlice[1:])
    img.data=img.data[newfocusSlice]
    imgCorrect.save(savePath+'/extractedImage'+stackstr)
    img.save(savePath+'/fullImageStack_adjusted'+stackstr)
    imgCorrect.mimwrite2D(savePath+'/FINALImage'+stackstr,axes=('h','y','x'))
    XcorWeight=imgCorrect.clone()
    XcorWeight.data=1-np.maximum(0,np.minimum(1,XcorWeight.data/XcorWeight.data.max()/np.exp(-0.5)))
    img.data*=XcorWeight.data
    img.save(savePath+'/imgWithoutFluid'+stackstr)
def motionCorrect(savePath,maskslice,focusSlice,guessPeriod,size,correlateSlice=None,caseRange=[],runFromStep=0,stackstr='',translateToStack=True,caseNameFormat='V{0:02d}',aviNameFormat='cropped_V{0:02d}.avi',nonCardiacMotion=True,includeRotate=False,useCorrDet=True,border=60,bgridFactor=4.,forcePeriod=False,highErrorDim=True,reduceNonRandomMode=0,dimSigmaFactor=1.,highIntensityFilter=None,lowIntensityFilter=None,finalSpread=0.001,avicheck=False):
    if type(correlateSlice)==type(None):
        correlateSlice=tuple(focusSlice)
    elif (len(focusSlice)-len(correlateSlice))==1:
        correlateSlice=(focusSlice[0],*correlateSlice)
        
    for caseN in caseRange:
        case=caseNameFormat.format(caseN)

        #step 0
        ###############Adjust image time and save###############
        logger.info(savePath+'/'+case)
        os.makedirs(savePath+'/'+case, exist_ok=True)
        if runFromStep<1 or not(os.path.isfile(savePath+'/cropped_'+case)):
            img=mip.imread(savePath+'/'+aviNameFormat.format(caseN))
            img.changeGreyscaleFormat()
            logger.info('image dimensions='+repr(img.dim))
            logger.info('image shape='+repr(img.data.shape))
            img.save(savePath+'/cropped_'+case)
        ##############################################


        #step 1
        ##########general rigid motion correction ###############
        if runFromStep<2:
            if nonCardiacMotion:
                motionCorrect_generalRigid(savePath,case,maskslice,focusSlice,guessPeriod,correlateSlice,includeRotate=includeRotate,border=border,avicheck=avicheck)
        ################
            

        #step 2
        ####################### phase sync NonRigid Correction ##################
        if runFromStep<3:
            motionCorrect_syncNonRigid(savePath,case,maskslice,focusSlice,guessPeriod,size,correlateSlice,nonCardiacMotion=nonCardiacMotion,includeRotate=includeRotate,border=border,bgridFactor=bgridFactor,forcePeriod=forcePeriod,highIntensityFilter=highIntensityFilter,lowIntensityFilter=lowIntensityFilter,avicheck=avicheck)

            motionCorrect_dimNonrandom(savePath,case,size,useCorrDet=useCorrDet,reduceNonRandomMode=reduceNonRandomMode,dimSigmaFactor=dimSigmaFactor,highIntensityFilter=highIntensityFilter,avicheck=avicheck)

        #################################################



        #step 3
        ################################################# intra cycle correction ############
        if runFromStep<4:
            motionCorrect_intraCycleCompound(savePath,case,maskslice,focusSlice,size,nonCardiacMotion=nonCardiacMotion,border=border,bgridFactor=bgridFactor,highErrorDim=highErrorDim,highIntensityFilter=highIntensityFilter,avicheck=avicheck)
        ###########################################


        #step 4
        ############################### GET segmentation Image################
        if runFromStep<5:
            motionCorrect_getSegmentation(savePath,case,size,finalSpread=finalSpread)
        ################################################



    #step 5
    ###########################   combine and synchonise ###############
    if len(caseRange)==0:
        combineAndSyncSlices(savePath,focusSlice,guessPeriod,stackstr=stackstr,translateToStack=translateToStack,caseNameFormat=caseNameFormat,nonCardiacMotion=nonCardiacMotion,border=border)
