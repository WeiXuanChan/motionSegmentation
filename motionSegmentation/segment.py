'''
File: segment.py
Description: 
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: w.x.chan@gmail.com         04FEB2020           - Created
          w.x.chan@gmail.com         10FEB2020           - v2.5.3
                                                             -added check() for initSnakeStack
                                                             -added border_value determination for SnakeStack.getSnake() and .getBinarySnake()
          w.x.chan@gmail.com         10FEB2020           - v2.5.7
                                                             -added snake with multiple initial pixel for initSnakeStack and add size of init 
          w.x.chan@gmail.com         18FEB2020           - v2.5.10
                                                             -in Snake class, added setBorderValue
                                                             -in initSnakeStack, remove border_value=1 snake from dialating 
          w.x.chan@gmail.com         19FEB2020           - v2.6.0
                                                             -in Snake class and initSnakeStack, added inner_outer_flexi_pixels and setSnakeBlock
                                                             -in Simplified_Mumford_Shah_driver,calculate curvature only when curvatureTerm_coef != 0
          w.x.chan@gmail.com         19FEB2020           - v2.6.1  
                                                             -in snake.getBinary, smoothing without opening first
          w.x.chan@gmail.com         28FEB2020           - v2.7.4  
                                                             -added detectNonregularBoundary
          w.x.chan@gmail.com         28FEB2020           - v2.7.7
                                                             -added option for initArray to detectNonregularBoundary
                                                             - added intiBlocksAxes to initSnakeStack
          w.x.chan@gmail.com         06MAR2020           - v2.7.9
                                                             -added mask to getInnerOuterMeanDiff, Simplified_Mumford_Shah_driver,and initSnakeStack
          w.x.chan@gmail.com         19Jan2021           - v2.7.14
                                                             -added manualSliceBySlice
          w.x.chan@gmail.com         20Jan2021           - v2.7.17
                                                             -debug manualSliceBySlice to get smooth interpolation between segmented slice
          w.x.chan@gmail.com         25Jan2021           - v2.7.19
                                                             -debug manualSliceBySlice
Requirements:
    numpy
Known Bug:
    None
All rights reserved.
'''
_version='2.7.19'
import logging
logger = logging.getLogger(__name__)

import os
import sys
import numpy as np
from scipy.ndimage import morphology
from scipy.ndimage import gaussian_filter
from scipy.ndimage import laplace
from scipy import interpolate
from scipy.ndimage import binary_fill_holes
try:
    from skimage import measure
except:
    pass

def detectNonregularBoundary(imageArray,outofbound_value=0,iterations=1000,smoothingCycle=1,mergeAxes=None,initArray=None,blockInitAxes=None):
    if initArray is None:
        boundaryArray=np.zeros(imageArray.shape)
        addboundaryArray=morphology.binary_dilation(boundaryArray,iterations=1,border_value=1)
        boundaryArray[np.logical_and(addboundaryArray,imageArray==outofbound_value)]=1.
    else:
        boundaryArray=initArray
    if blockInitAxes is not None:
        if isinstance(blockInitAxes,int):
            blockInitAxes=[blockInitAxes]
        if isinstance(blockInitAxes,(list,tuple)):
            sliceList=[]
            for n in range(len(boundaryArray.shape)):
                if n in blockInitAxes:
                    sliceList.append(slice(None))
                else:
                    sliceList.append(slice(1,-1))
            boundaryArray[tuple(sliceList)]=0
        elif isinstance(blockInitAxes,np.ndarray):
            boundaryArray[blockInitAxes.astype(bool)]=0
    s=Snake(imageArray=imageArray,snakesInit=boundaryArray,driver=Specific_value_driver(value=outofbound_value),border_value=-1)
    pixelIncr_lastSmoothingCycle=imageArray.size
    for n in range(iterations):
        snake_incr=s.getDialate()
        if n%smoothingCycle==0:
            if np.count_nonzero(snake_incr)>=pixelIncr_lastSmoothingCycle:
                s+=snake_incr
                break
            else:
                pixelIncr_lastSmoothingCycle=np.count_nonzero(snake_incr)
        logger.info('iteration '+str(n)+' : pixel increment = '+str(np.count_nonzero(snake_incr)))
        if np.any(snake_incr):
            s+=snake_incr
            if n%smoothingCycle==(smoothingCycle-1):
                s.snake=morphology.binary_erosion(s.snake,iterations=1,border_value=1)
                s.snake=morphology.binary_dilation(s.snake,iterations=1,border_value=0)
        else:
            break
    result=morphology.binary_dilation(s.snake.astype(bool),iterations=1,border_value=0)
    if mergeAxes is not None:
        result=np.prod(result,axis=mergeAxes)
    return np.logical_not(result)
    

def gaussianCurvature(addBinary,image,oldsnake,sigmas=3):
    #curvature from -1 to 1, low value means more cells with outer value, reduce driving force to reduce curvature
    curvatureSnake=oldsnake.copy()
    if addBinary is None:
        curvature=(gaussian_filter(curvatureSnake,sigmas)-0.5)*2.
    else:
        curvatureSnake[addBinary]=0.5
        curvature=[((gaussian_filter(curvatureSnake,sigmas)-0.5)*2.)[addBinary]]
        currentSigma=sigmas
        while currentSigma>1:
            currentSigma*=0.5
            curvature.append(((gaussian_filter(curvatureSnake,currentSigma)-0.5)*2.)[addBinary])
        if len(curvature)>1:
            curvatureMin=np.min(curvature,axis=0)
            curvatureMax=np.max(curvature,axis=0)
            curvature=np.zeros(curvatureMin.shape)
            curvature[np.abs(curvatureMin)>curvatureMax]=curvatureMin[np.abs(curvatureMin)>curvatureMax]
            curvature[np.abs(curvatureMin)<curvatureMax]=curvatureMax[np.abs(curvatureMin)<curvatureMax]
            curvature[curvatureMin==curvatureMax]=curvatureMax[curvatureMin==curvatureMax]
        else:
            curvature=curvature[0]
        
    return curvature
def laplaceCurvature(addBinary,image,oldsnake,sigmas=1):
    #curvature from -1 to 1, low value means more cells with outer value, reduce driving force to reduce curvature
    curvature=gaussian_filter(oldsnake,sigmas)  
    curvature=laplace(curvature)
    return curvature[addBinary]
def getInnerOuterMeanDiff(addBinary,image,oldsnake,mask=None):
    if mask is None:
        tempimage=image
        tempoldsnake=oldsnake
        total=image.size
    else:
        tempimage=image*mask
        tempoldsnake=oldsnake*mask
        total=np.sum(mask)
    ones_value=np.sum(tempimage*tempoldsnake)
    ones_size=np.sum(tempoldsnake)
    zeros_value=np.sum(tempimage)-ones_value
    zeros_size=total-ones_size
    mean_ones=ones_value/ones_size
    mean_zeros=zeros_value/zeros_size
    I=image[addBinary]
    return ((I-mean_zeros)**2.-(I-mean_ones)**2.)/max(1,mean_zeros-mean_ones)**2.
class Simplified_Mumford_Shah_driver:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,snakeStackClass=None,curvatureTerm_coef=1.,curvatureSigmas=5,meanTerm_coef=5.,expTerm_coef=1.,calculationMaskToOne=None):
        #calculationMaskToOne is a mask where value with 0 is excluded from calculation and the resultant value of these pixels are always 1
        self.curvatureTerm_coef=curvatureTerm_coef  #+ve
        self.curvatureSigmas=curvatureSigmas
        self.meanTerm_coef=meanTerm_coef  #+ve
        self.expTerm_coef=expTerm_coef  #+ve
        self.snakeStackClass=snakeStackClass
        self.calculationMaskToOne=calculationMaskToOne
    def __call__(self,addBinary,image,oldsnake):
        result=np.zeros(addBinary.shape)
        if self.curvatureTerm_coef==0:
            curvature=0
        else:
            curvature=gaussianCurvature(addBinary,image,oldsnake,self.curvatureSigmas)
            
        if type(self.snakeStackClass)==type(None):
            innerOuterMeanDiff=getInnerOuterMeanDiff(addBinary,image,oldsnake,mask=self.calculationMaskToOne)
        else:
            totalsnakes=self.snakeStackClass.getsnake()
            innerOuterMeanDiff=getInnerOuterMeanDiff(addBinary,image,totalsnakes.snake.copy(),mask=self.calculationMaskToOne)
        result[addBinary]=self.curvatureTerm_coef*curvature+self.meanTerm_coef*innerOuterMeanDiff*np.exp(innerOuterMeanDiff/np.abs(innerOuterMeanDiff)*self.expTerm_coef*curvature)
        if self.calculationMaskToOne is not None:
            result[np.logical_and(addBinary,self.calculationMaskToOne==0)]=1
        return result
class Specific_value_driver:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,snakeStackClass=None,value=0,include=True):
        #if include, addbinary will be True for image pixel = value else, addbinary will be True for image pixel != value
        self.snakeStackClass=snakeStackClass
        self.value=value
        self.include=include
    def __call__(self,addBinary,image,oldsnake):
        result=np.zeros(oldsnake.shape)
        if self.include:
            result[np.logical_and(addBinary,image==self.value)]=1.
        else:
            result[np.logical_and(addBinary,image!=self.value)]=1.
        return result
class none_driver:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,snakeStackClass=None):
        self.snakeStackClass=snakeStackClass
    def __call__(self,addBinary,image,oldsnake):
        return addBinary.astype(float)
class Snake:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,imageArray=None,snakesInit=None,ID=0,driver=None,border_value=None):
        '''
        Initialize all data.
        Algorithm will expand True
        min_nonIntersecti s the number of diallation from merging 2 snakes
        '''
        self.ID=ID
        self.border_value=border_value
        self.border=None
        self.imageArray=imageArray
        if type(snakesInit)!=type(None):
            self.snakeInit=snakesInit.copy()
            self.snake=snakesInit.copy()
            if self.border_value is None:
                self.border_value=int(bool(self.snake[tuple([0]*len(self.snake.shape))]))
        else:
            self.snakeInit=None
            self.snake=None
        self.driver=driver
    def setImage(self,imageArray):
        self.imageArray=imageArray
    def check(self):
        if self.snake is None:
            raise Exception('snake '+str(self.ID)+' not initialized')
        if self.snakeInit is None:
            self.snakeInit=self.snake.copy()
        if self.driver is None:
            self.driver=none_driver()
        if self.border_value is None:
            self.border_value=int(bool(self.snake[tuple([0]*len(self.snake.shape))]))
    def setBorderValue(self,value=None):
        if value is None:
            value=self.border_value
        if value is None:
            logger.warning('No Border value set in Snake '+repr(self.ID))
        elif value>=0 and value<=1:
            if self.border is None:
                self.border=np.zeros(self.snake.shape).astype(bool)
                for n in range(len(self.snake.shape)):
                    self.border[tuple([slice(None)]*n+[0])]=True
                    self.border[tuple([slice(None)]*n+[self.snake.shape[n]-1])]=True
            self.snake[self.border]=value
    def getBinary(self,smoothing=False):
        binarySnake=self.snake.copy()
        binarySnake[binarySnake<1]=0
        binarySnake=binarySnake.astype(bool)
        if smoothing:
            if isinstance(smoothing,(int,float)):
                sigma=float(smoothing)
                while sigma>1:
                    curvature=(gaussian_filter(binarySnake.astype(float),sigma*1., truncate=1.)-0.5)*2.
                    #curvature=gaussianCurvature(None,None,binarySnake.astype(float),smoothing)
                    curvbinarySnake=np.zeros(curvature.shape,dtype=bool)
                    curvbinarySnake[curvature>0.5]=True
                    binarySnake+=curvbinarySnake
                    sigma*=0.5
            #binarySnake=morphology.binary_opening(binarySnake,iterations=1)
            newbinarySnake=binarySnake.copy()
            for k in range(50):
                newbinarySnake=morphology.binary_closing(binarySnake,iterations=1)
                if np.all(newbinarySnake==binarySnake):
                    break
                else:
                    binarySnake=newbinarySnake.copy()
        return binarySnake
    def removeExtras(self,numOfAreas=1):
        binarySnake=self.snake.copy()
        binarySnake[binarySnake<1]=0
        labels = measure.label(binarySnake,connectivity=1,background=-1)
        lab,labCount=np.unique(labels,return_counts=True)
        bgInd=labels[tuple([0]*len(self.snake.shape))]
        lab=np.delete(lab,bgInd-1)
        labCount=np.delete(labCount,bgInd-1)
        nonBgInd=np.argmax(labCount)
        removeArea=labels!=nonBgInd
        for n in range(min(numOfAreas,len(lab))):
            nonBgInd=np.argmax(labCount)
            removeArea*=labels!=lab[nonBgInd]
            lab=np.delete(lab,nonBgInd)
            labCount=np.delete(labCount,nonBgInd)
        self.snake[removeArea]=self.border_value
    def clone(self,ID=None):
        newSnake=Snake()
        if ID is None:
            newSnake.ID=self.ID
        else:
            newSnake.ID=ID
        if self.snakeInit is None:
            newSnake.snakeInit=None
        else:
            newSnake.snakeInit=self.snakeInit.copy()
        if self.snake is None:
            newSnake.snake=None
        else:
            newSnake.snake=self.snake.copy()
        if self.imageArray is None:
            newSnake.imageArray=None
        else:
            newSnake.imageArray=self.imageArray.copy()
        newSnake.driver=self.driver
        newSnake.border_value=self.border_value
        return newSnake
    def __add__(self,array):
        result=self.clone()
        if type(array)==np.ndarray:
            result.snake=np.minimum(1,np.maximum(0,result.snake+array))
        else:
            result.snake=np.minimum(1,np.maximum(0,result.snake+array.snake))
        result.setBorderValue()
        return result
    def reset(self):
        self.snake=self.snakeInit.copy()
    def getDialate(self,numOfDialation=1,filter_with_driver=True):
        binarySnake=self.getBinary()
        addBinary=np.logical_xor(morphology.binary_dilation(binarySnake,iterations=numOfDialation,border_value=min(max(self.border_value,0),1)),binarySnake)
        if filter_with_driver:
            return self.driver(addBinary,self.imageArray.copy(),self.snake.copy())
        else:
            return addBinary
    def __call__(self):
        return self.snake.copy()
class SnakeStack:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,snakesInitList=None,min_nonIntersect=5,inner_outer_flexi_pixels=None):
        '''
        Initialize all data.
        Algorithm will expand True
        min_nonIntersecti s the number of diallation from merging 2 snakes
        inner_outer_flexi_pixels=[np.ndarray,np.ndarray], with mean_zeros and mean_ones updated before every iteration
        '''
        self.inner_outer_flexi_pixels=inner_outer_flexi_pixels
        self.min_nonIntersect=int(min_nonIntersect)
        if snakesInitList is None:
            self.snakes=[]
        elif type(snakesInitList)!=list:
            self.snakes=[snakesInitList]
        else:
            self.snakes=snakesInitList
    def check(self):
        if len(self.snakes)<1:
            raise Exception('Snake not initialized')
        self.shape=self.snakes[0].imageArray.shape
        for n in range(len(self.snakes)):
            self.snakes[n].check()
            if np.any(self.shape!=self.snakes[n].imageArray.shape):
                raise Exception('Snake '+str(n)+' image '+str(self.snakes[n].imageArray.shape)+'is of different shape '+str(self.shape))
    def update_flexi_pixels(self):
        if type(self.inner_outer_flexi_pixels)!=type(None):
            oldsnake=self.getsnake().snake
            ones_size=np.sum(oldsnake)
            zeros_size=oldsnake.size-ones_size
            for sn in self.snakes:
                ones_value=np.sum(sn.imageArray*oldsnake)
                zeros_value=np.sum(sn.imageArray)-ones_value
                mean_ones=ones_value/ones_size
                mean_zeros=zeros_value/zeros_size
                if type(self.inner_outer_flexi_pixels[0])!=type(None):
                    sn.imageArray[self.inner_outer_flexi_pixels[0]]=mean_zeros
                if type(self.inner_outer_flexi_pixels[1])!=type(None):
                    sn.imageArray[self.inner_outer_flexi_pixels[1]]=mean_ones
    def dialate(self,numOfTimes=1,smoothingCycle=10,smoothingSigma=True,recorderList=None):
        for nDialate in range(numOfTimes):
            self.update_flexi_pixels()
            snake_incr=[]
            for n in range(len(self.snakes)):
                snake_incr.append(self.snakes[n].getDialate())
            
            if self.min_nonIntersect>0:
                snake_intersect=np.zeros(self.shape,dtype=bool)
                remove_intersect=np.zeros(self.shape,dtype=bool)
                cumulative_intersect=np.zeros(self.shape,dtype=bool)
                for n in range(len(self.snakes)):
                    addBinary=self.snakes[n].getDialate(numOfDialation=self.min_nonIntersect,filter_with_driver=False)
                    snake_intersect+=cumulative_intersect*addBinary
                    cumulative_intersect+=addBinary
                if np.any(snake_intersect):
                    remove_intersect=morphology.binary_dilation(snake_intersect,iterations=self.min_nonIntersect,border_value=0)
                    for n in range(len(self.snakes)):
                        snake_incr[n][remove_intersect]=0
            #update
            for n in range(len(self.snakes)):
                self.snakes[n]+=snake_incr[n]
            if smoothingCycle>0:
                if nDialate%smoothingCycle==(smoothingCycle-1):
                    for n in range(len(self.snakes)):
                        smoothedSnake=self.snakes[n].getBinary(smoothing=smoothingSigma)
                        self.snakes[n]+=smoothedSnake
                    if type(recorderList)!=type(None):
                        recorderList.append(self.getBinarysnake(True).snake.copy())
    def getsnake(self):
        border_value=0
        for n in range(len(self.snakes)):
            if self.snakes[n].border_value==1:
                border_value=1
        result=Snake(snakesInit=np.zeros(self.shape),border_value=border_value)
        for n in range(len(self.snakes)):
            result+=self.snakes[n]()
        return result
    def getBinarysnake(self,smoothing=False):
        border_value=0
        for n in range(len(self.snakes)):
            if self.snakes[n].border_value==1:
                border_value=1
        result=Snake(snakesInit=np.zeros(self.shape),border_value=border_value)
        for n in range(len(self.snakes)):
            result+=self.snakes[n].getBinary(smoothing=smoothing)
        return result
    def reset(self):
        for n in range(len(self.snakes)):
            self.snakes[n].reset()
    def run(self,numOfTimes=1,reset=True):
        if reset:
            self.reset()
        self.dialate(numOfTimes)
        return self.getsnake()

def initSnakeStack(imageArray,snakeInitCoordList,driver=None,initSize=1,setSnakeBlocks=None,intiBlocksAxes=None,mask=None):
    if setSnakeBlocks is not None:
        if setSnakeBlocks is True:
            setSnakeBlocks=0
        if isinstance(setSnakeBlocks,int):
            padAxis=[]
            for n in range(len(imageArray.shape)):
                if n==setSnakeBlocks:
                    padAxis.append([1,1])
                else:
                    padAxis.append([0,0])
        else:
            padAxis=setSnakeBlocks
        onesArray=np.zeros(imageArray.shape,dtype=bool)
        zerosArray=np.zeros(imageArray.shape,dtype=bool)
        imageArray=np.pad(imageArray,padAxis,constant_values=imageArray.mean())
        onesArray=np.pad(onesArray,padAxis,constant_values=False)
        zerosArray=np.pad(zerosArray,padAxis,constant_values=True)
        imageArray=np.pad(imageArray,1,constant_values=imageArray.min())
        onesArray=np.pad(onesArray,1,constant_values=True)
        zerosArray=np.pad(zerosArray,1,constant_values=False)
    initSnake=[]
    if driver is None:
        driver=Simplified_Mumford_Shah_driver()
    driver.calculationMaskToOne=mask
    for n in range(len(snakeInitCoordList)):
        isOuterSnake=False
        if len(snakeInitCoordList[n].shape)>1:
            initArray=np.zeros(imageArray.shape)
            for m in range(len(snakeInitCoordList[n])):
                if np.all(snakeInitCoordList[n][m]==0):
                    isOuterSnake=True
                    initArray_temp=np.ones(imageArray.shape)
                    if intiBlocksAxes is None:
                        sliceList=[slice(1,-1)]*len(imageArray.shape)
                    else:
                        if isinstance(intiBlocksAxes,int):
                            intiBlocksAxes=[intiBlocksAxes]
                        sliceList=[]
                        for axis in range(len(imageArray.shape)):
                            if axis in intiBlocksAxes:
                                sliceList.append(slice(None))
                            else:
                                sliceList.append(slice(1,-1))
                    initArray_temp[tuple(sliceList)]=0
                else:
                    initArray_temp=np.zeros(imageArray.shape)
                    initArray_temp[tuple(snakeInitCoordList[n][m])]=1
                    if initSize>1:
                        initArray_temp=morphology.binary_dilation(initArray_temp,iterations=initSize-1,border_value=0).astype(float)
                initArray+=initArray_temp
            initArray=np.minimum(initArray,1)
        else:
            if np.all(snakeInitCoordList[n]==0):
                isOuterSnake=True
                initArray=np.ones(imageArray.shape)
                if intiBlocksAxes is None:
                    sliceList=[slice(1,-1)]*len(imageArray.shape)
                else:
                    if isinstance(intiBlocksAxes,int):
                        intiBlocksAxes=[intiBlocksAxes]
                    sliceList=[]
                    for axis in range(len(imageArray.shape)):
                        if axis in intiBlocksAxes:
                            sliceList.append(slice(None))
                        else:
                            sliceList.append(slice(1,-1))
                initArray[tuple(sliceList)]=0
            else:
                initArray=np.zeros(imageArray.shape)
                initArray[tuple(snakeInitCoordList[n])]=1
                if initSize>1:
                    initArray=morphology.binary_dilation(initArray,iterations=initSize-1,border_value=0).astype(float)
        if isOuterSnake and (intiBlocksAxes is not None):
            initSnake.append(Snake(imageArray,initArray.copy(),driver=driver,border_value=-1))
        else:
            initSnake.append(Snake(imageArray,initArray.copy(),driver=driver))
    if setSnakeBlocks:
        resultSnakeStack=SnakeStack(initSnake,inner_outer_flexi_pixels=[zerosArray,onesArray])
    else:
        resultSnakeStack=SnakeStack(initSnake)
    driver.snakeStackClass=resultSnakeStack
    resultSnakeStack.check()
    return resultSnakeStack

def manualSliceBySlice(img,initLineList=None,lengthScaleRatio=0.2):
    if initLineList is not None:
        if isinstance(initLineList,list) and len(initLineList[0].shape)==2:
            a=img.show(disable=['click','swap'],initLineList=initLineList)
        else:
            newInitLineList=[[]]
            currentslice=initLineList[0][0]
            for n in range(len(initLineList)):
                if currentslice!=initLineList[n][0]:
                    newInitLineList.append([])
                    currentslice=initLineList[n][0]
                newInitLineList[-1].append(initLineList[n])
            for n in range(len(newInitLineList)):
                newInitLineList[n]=np.array(newInitLineList[n])
            a=img.show(disable=['click','swap'],initLineList=newInitLineList)
    else:
        a=img.show(disable=['click','swap'])
    while not(a.enter):
        a=img.show(disable=['click','swap'],initLineList=a.lines)
    for n in range(len(a.lines)-1,-1,-1):
        if len(a.lines[n])<3:
            a.lines.pop(n)
    if a.color:
        if len(img.dim)>4:
            raise Exception('Dimension of image more than 3,'+str(img.dim))
    else:
        if len(img.dim)>3:
            raise Exception('Dimension of image more than 3,'+str(img.dim))
    aind=0
    img2=img.clone()
    if a.color:
        img2.changeGreyscaleFormat()
    img2.data[:]=0
    img2.data=img2.data.astype('uint8')
    minArea=float('inf')
    for n in range(img2.data.shape[0]):
        if a.lines[aind][0][0]==n:
            for num in [1000,10000,100000,1000000]:
                ar=np.array(a.lines[aind])
                tck,temp = interpolate.splprep([ar[:,-2], ar[:,-1]], s=0,k=min(4,len(ar))-1)
                cspline_detectline = np.array(interpolate.splev(np.linspace(0, 1,num=num), tck)).T
                cspline_detectline=np.floor(cspline_detectline).astype(int)
                for nn in range(len(cspline_detectline)):
                    img2.data[n][tuple(cspline_detectline[nn])]=1
                ar2=np.array([ar[0],0.5*(ar[0]+ar[-1]),ar[-1]])
                tck,temp = interpolate.splprep([ar2[:,-2], ar2[:,-1]], s=0,k=1)
                cspline_detectline = np.array(interpolate.splev(np.linspace(0, 1,num=num), tck)).T
                cspline_detectline=np.floor(cspline_detectline).astype(int)
                for nn in range(len(cspline_detectline)):
                    img2.data[n][tuple(cspline_detectline[nn])]=1
                countpixel=img2.data[n].sum()
                img2.data[n]=binary_fill_holes(img2.data[n]).astype(img2.data.dtype)
                if img2.data[n].sum()>countpixel:
                    break
            temp_area=img2.data[n].sum()
            if temp_area<minArea:
                minArea=temp_area
            aind+=1
            if aind>=len(a.lines):
                break
    img2.data*=255
    lengthScale=max(2,lengthScaleRatio*minArea**0.5)
    img2.data=gaussian_filter(img2.data,(0,lengthScale,lengthScale))
    img3=img2.clone()
    getind=[]
    for n in range(len(a.lines)):
        getind.append(a.lines[n][0][0])
    getind=np.array(getind).astype(int)
    currentind=0
    for n in range(getind[0]+1,getind[-1]):
        if n==getind[currentind+1]:
            currentind+=1
        else:
            ratio=float(n-getind[currentind])/(getind[currentind+1]-getind[currentind])
            img3.data[n]=(img2.data[getind[currentind+1]]*ratio+img2.data[getind[currentind]]*(1.-ratio)).astype(img2.data.dtype)
    sigma=[]
    if a.color:
        nDim=[0,-3,-2]
    else:
        nDim=[0,-2,-1]
    for n in nDim:
        sigma.append(img3.dimlen[img3.dim[n]])
    sigma=1./np.array(sigma)
    sigma=sigma/sigma[1:].mean()*lengthScale
    img3.data=gaussian_filter(img3.data,sigma)
    return (a.lines,img3)

