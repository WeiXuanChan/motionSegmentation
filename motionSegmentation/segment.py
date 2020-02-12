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
Requirements:
    numpy
Known Bug:
    None
All rights reserved.
'''
_version='2.5.7'
import logging
logger = logging.getLogger(__name__)

import os
import sys
import numpy as np
from scipy.ndimage import morphology
from scipy.ndimage import gaussian_filter
from scipy.ndimage import laplace
try:
    from skimage import measure
except:
    pass

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
def getInnerOuterMeanDiff(addBinary,image,oldsnake):
    ones_value=np.sum(image*oldsnake)
    ones_size=np.sum(oldsnake)
    zeros_value=np.sum(image)-ones_value
    zeros_size=image.size-ones_size
    mean_ones=ones_value/ones_size
    mean_zeros=zeros_value/zeros_size
    I=image[addBinary]
    return ((I-mean_zeros)**2.-(I-mean_ones)**2.)/max(1,mean_zeros-mean_ones)**2.
class Simplified_Mumford_Shah_driver:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,snakeStackClass=None,curvatureTerm_coef=1.,curvatureSigmas=5,meanTerm_coef=5.,expTerm_coef=1.):
        self.curvatureTerm_coef=curvatureTerm_coef  #+ve
        self.curvatureSigmas=curvatureSigmas
        self.meanTerm_coef=meanTerm_coef  #+ve
        self.expTerm_coef=expTerm_coef  #+ve
        self.snakeStackClass=snakeStackClass
    def __call__(self,addBinary,image,oldsnake):
        result=np.zeros(addBinary.shape)
        curvature=gaussianCurvature(addBinary,image,oldsnake,self.curvatureSigmas)
        if type(self.snakeStackClass)==type(None):
            innerOuterMeanDiff=getInnerOuterMeanDiff(addBinary,image,oldsnake)
        else:
            totalsnakes=self.snakeStackClass.getsnake()
            innerOuterMeanDiff=getInnerOuterMeanDiff(addBinary,image,totalsnakes.snake.copy())
        result[addBinary]=self.curvatureTerm_coef*curvature+self.meanTerm_coef*innerOuterMeanDiff*np.exp(innerOuterMeanDiff/np.abs(innerOuterMeanDiff)*self.expTerm_coef*curvature)
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
        if type(snakesInit)!=type(None):
            self.snakeInit=snakesInit.copy()
            self.snake=snakesInit.copy()
        else:
            self.snakeInit=None
            self.snake=None
        self.imageArray=imageArray
        self.driver=driver
        self.border=None
        self.border_value=border_value
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
        self.border=np.zeros(self.snake.shape,dtype=bool)
        for n in range(len(self.snake.shape)):
            self.border[tuple([slice(None)]*n+[0])]=True
            self.border[tuple([slice(None)]*n+[self.snake.shape[n]-1])]=True
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
            binarySnake=morphology.binary_opening(binarySnake,iterations=1)
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
        if self.border is None:
            newSnake.border=None
        else:
            newSnake.border=self.border.copy()
        return newSnake
    def __add__(self,array):
        result=self.clone()
        if type(array)==np.ndarray:
            result.snake=np.minimum(1,np.maximum(0,result.snake+array))
        else:
            result.snake=np.minimum(1,np.maximum(0,result.snake+array.snake))
        if type(result.border)!=type(None):
            result.snake[result.border]=result.border_value
        return result
    def reset(self):
        self.snake=self.snakeInit.copy()
    def getDialate(self,numOfDialation=1,filter_with_driver=True):
        binarySnake=self.getBinary()
        addBinary=np.logical_xor(morphology.binary_dilation(binarySnake,iterations=numOfDialation,border_value=self.border_value),binarySnake)
        if filter_with_driver:
            return self.driver(addBinary,self.imageArray.copy(),self.snake.copy())
        else:
            return addBinary
    def __call__(self):
        return self.snake.copy()
class SnakeStack:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,snakesInitList=None,min_nonIntersect=5):
        '''
        Initialize all data.
        Algorithm will expand True
        min_nonIntersecti s the number of diallation from merging 2 snakes
        '''
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
    def dialate(self,numOfTimes=1,smoothingCycle=10,smoothingSigma=True,recorderList=None):
        for nDialate in range(numOfTimes):
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

def initSnakeStack(imageArray,snakeInitCoordList,driver=None,initSize=1):
    initSnake=[]
    if driver is None:
        driver=Simplified_Mumford_Shah_driver()
    for n in range(len(snakeInitCoordList)):
        if len(snakeInitCoordList[n].shape)>1:
            initArray=np.zeros(imageArray.shape)
            for m in range(len(snakeInitCoordList[n])):
                if np.all(snakeInitCoordList[n][m]==0):
                    initArray_temp=np.ones(imageArray.shape)
                    sliceList=[slice(1,-1)]*len(imageArray.shape)
                    initArray_temp[tuple(sliceList)]=0
                    if initSize>1:
                        initArray_temp=morphology.binary_dilation(initArray_temp,iterations=initSize-1,border_value=1).astype(float)
                else:
                    initArray_temp=np.zeros(imageArray.shape)
                    initArray_temp[tuple(snakeInitCoordList[n][m])]=1
                    if initSize>1:
                        initArray_temp=morphology.binary_dilation(initArray_temp,iterations=initSize-1,border_value=0).astype(float)
                initArray+=initArray_temp
            initArray=np.minimum(initArray,1)
        else:
            if np.all(snakeInitCoordList[n]==0):
                initArray=np.ones(imageArray.shape)
                sliceList=[slice(1,-1)]*len(imageArray.shape)
                initArray[tuple(sliceList)]=0
                if initSize>1:
                    initArray=morphology.binary_dilation(initArray,iterations=initSize-1,border_value=1).astype(float)
            else:
                initArray=np.zeros(imageArray.shape)
                initArray[tuple(snakeInitCoordList[n])]=1
                if initSize>1:
                    initArray=morphology.binary_dilation(initArray,iterations=initSize-1,border_value=0).astype(float)
        initSnake.append(Snake(imageArray,initArray.copy(),driver=driver))
    resultSnakeStack=SnakeStack(initSnake)
    driver.snakeStackClass=resultSnakeStack
    resultSnakeStack.check()
    return resultSnakeStack
    
        
