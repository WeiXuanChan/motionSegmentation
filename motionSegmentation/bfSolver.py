'''
File: bfSolver.py
Description: solving BSplinrFourier Coeficients with input bspline vector maps
             Contains externally usable class
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: w.x.chan@gmail.com         25OCT2018           - Created
  Author: w.x.chan@gmail.com         06Aug2019           - v2.0.0
                                                             -enable 2D for BC-point by point solver only
  Author: w.x.chan@gmail.com         12Sep2019           - v2.1.0
                                                             -save Sampling results at the end of solving
  Author: w.x.chan@gmail.com         12Sep2019           - v2.2.4
                                                             -change bfSolver.points to numpy array in loadSamplingResults
  Author: w.x.chan@gmail.com         13Nov2019           - v2.4.1
                                                             -include mode of initial estimate with forwardbackward
  Author: w.x.chan@gmail.com         18Nov2019           - v2.4.3
                                                             -debug initial estimate with forwardbackward (reshape Fratio)
  Author: w.x.chan@gmail.com         18Nov2019           - v2.4.4
                                                             -change to logging
  Author: jorry.zhengyu@gmail.com    03June2020           - v2.7.11
                                                             -add NFFT initialization to estimateInitialwithRefTime
                                                             -add delimiter option to pointTrace
  Author: w.x.chan@gmail.com         19Jan2021           - v2.7.15
                                                             -remove Bspline2D in addBsplineFile function (bug)
                                                             -debug refTimeStep option in estimateInitialwithRefTime
                                                             -auto detect timeMapList in estimateInitialwithRefTime

Requirements:
    BsplineFourier
    numpy
    scipy
    nfft

Known Bug:
    None
All rights reserved.
'''
_version='2.7.15'

import logging
logger = logging.getLogger(__name__)
import numpy as np
import autoD as ad
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import scipy.sparse as sparse
try:
    from sksparse.cholmod import cholesky
except:
    pass
from numpy.linalg import matrix_rank

try:
    from joblib import Parallel, delayed
except ImportError:
    pass
try:
  import pickle
except ImportError:
  pass
try:
    import medImgProc
except:
    pass
import BsplineFourier
import multiprocessing
import time
import trimesh
import nfft

FourierSeries_type=type(BsplineFourier.FourierSeries(1,1))
def estCoordsThruTime(coord,bsplineList,OrderedBsplinesList,OrderedBsplinesList2=None,mode='Lagrangian-Eulerian'):
    coordsThruTime=[coord.copy()]
    if mode=='Eulerian':
        for n in range(len(OrderedBsplinesList)):
            vector=bsplineList[OrderedBsplinesList[n]].getVector(coordsThruTime[-1])
            coordsThruTime.append(vector+coordsThruTime[-1])
    elif mode=='Lagrangian-Eulerian':
        for n in range(len(OrderedBsplinesList)):
            vector=bsplineList[OrderedBsplinesList[n]].getVector(coordsThruTime[0])
            newcoord=vector+coordsThruTime[0]
            if type(OrderedBsplinesList2)!=type(None) and n>0 and n<(len(OrderedBsplinesList)-1):
                ratio=float(n)*float(len(OrderedBsplinesList)-1-n)/((len(OrderedBsplinesList)-1)/2.)**2.
                vector2=bsplineList[OrderedBsplinesList2[n]].getVector(coordsThruTime[-1])
                newcoord=newcoord*(1.-ratio)+(ratio)*(coordsThruTime[-1]+vector2)
            coordsThruTime.append(newcoord.copy())
    else:
        raise Exception('mode selection error.')
    return coordsThruTime
def getCoordfromCoef(coord,coef,spacing):
    coeftemp=coef[0].copy()
    for m in range(int(coef.shape[0]/2)):#sub in t
        coeftemp=coeftemp+coef[m+1]*np.cos((m+1.)*2.*np.pi/spacing[-1]*coord[-1])+coef[int(coef.shape[0]/2)+m+1]*np.sin((m+1.)*2.*np.pi/spacing[-1]*coord[-1])
    resultDeltaCoord=coeftemp.copy()
    return resultDeltaCoord

def load(file):
    with open(file, 'rb') as input:
        outObj = pickle.load(input)
    return outObj
def SAC(val,cI):
    if len(val)==0:
        value=0
    elif (val.max()-val.min())<=3:
        value=val.mean()
    else:
        value=val.mean()
        bincount= np.bincount(np.around(val).astype(int))
        bincountCompressed = bincount[bincount!=0]
        totalCount=bincountCompressed.sum()
        intensityCompresed=np.nonzero(bincount)[0]
        lowerBoundInd=1
        while bincountCompressed[:lowerBoundInd].sum()<(totalCount*cI):
            lowerBoundInd+=1
        upperBoundInd=len(bincountCompressed)
        while bincountCompressed[(upperBoundInd-1):].sum()<(totalCount*cI):
            upperBoundInd-=1
        bound=np.zeros((257,4),dtype=int)
        for low in range(upperBoundInd):
            for high in range(max(low+1,lowerBoundInd),len(bincountCompressed)+1):
                width=intensityCompresed[high-1]-intensityCompresed[low]+1
                temp_sum=bincountCompressed[low:high].sum()
                if temp_sum>=(totalCount*cI):
                    if temp_sum>np.abs(bound[width,0]):
                        bound[width]=np.array([temp_sum,low,high,width])
                    elif temp_sum==np.abs(bound[width,0]):
                        bound[width]=np.array([-temp_sum,1,0,width])
        newbound=bound[bound[:,0]>0]
        if len(newbound)>0:
            for n in range(len(newbound)):
                if newbound[n,0]>np.abs(bound[:newbound[n,3],0]).max():
                    high=newbound[n,2]
                    low=newbound[n,1]
                    value=np.sum(bincountCompressed[low:high]*intensityCompresed[low:high])/bincountCompressed[low:high].sum()
                    break
    return value

class bfSolver:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self):
        self.bsplines=[]
        self.weights=[]
        self.points=[]
        self.bsFourier=BsplineFourier.BsplineFourier()
        self.wdiag=np.ones(0)
        self.pointsCoef=[]
        self.eqn=[]
        self.eqnWeight=[]
        self.eqnToPts=[]
    def save(self,file):
        with open(file, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
    def writeSamplingResults(self,filepath,delimiter=' '):
        ''' 
        Write sampling results in a single-line in Fortran format
        Parameters:
            filePath:file,str
                File or filename to save to
            delimiter:str, optional
                separation between values
        '''
        saveMatrix=[]
        for n in range(len(self.points)):
            if len(self.pointsCoef)>n:
                saveMatrix.append([*self.points[n],*self.pointsCoef[n].reshape(-1, order='F')])
            else:
                saveMatrix.append([*self.points[n],*np.zeros(self.pointsCoef[0].size)])
        saveMatrix=np.array(saveMatrix)
        np.savetxt(filepath,saveMatrix,delimiter=delimiter,header=str(len(self.pointsCoef))+' points calculated-- Coordinates, Fourier uvw')
    def loadSamplingResults(self,filepath,delimiter=' '):
        ''' 
        Read sampling results in a single-line in Fortran format
        Parameters:
            filePath:file,str
                File or filename to save to
            delimiter:str, optional
                separation between values
        '''
        self.points=[]
        self.pointsCoef=[]
        with open (filepath, "r") as myfile:
            data=myfile.readline()
        coeflen=[int(s) for s in data.split() if s.isdigit()][0]
        loadMatrix=np.loadtxt(filepath,delimiter=delimiter)
        nd=(np.abs(loadMatrix).max(axis=0)==0).argmax(axis=0)
        for n in range(len(loadMatrix)):
            self.points.append(loadMatrix[n,:nd].copy())
            if n<coeflen:
                self.pointsCoef.append(loadMatrix[n,nd:].reshape((-1,nd),order='F'))
        self.points=np.array(self.points)
    def addBsplineFile(self,BsplineFileList=None,timeMapList=None,weightList=None,fileScale=1.,twoD=False):
        ''' 
        add (multiple) BspineFile to solver
        Parameters:
            BsplineFileList: list(file_path)
                List of Bspine File(s) to read and add to solver 
            timeMapList:list([fromTime,toTime],[fromTime,toTime],...)
                Corresponding fromtime->toTime map of bspline
            weightList:list(float), optional, defaults to 1
                Corresponding weight of bspline
            fileScale:float, optional, defaults to 1
                scale of the bsplinefile to resultant bsplinefourier
        '''
        if type(BsplineFileList)!=list:
            BsplineFileList=[BsplineFileList]
            timeMapList=[timeMapList]
            if type(weightList)!=type(None):
                weightList=[weightList]
        for n in range(len(BsplineFileList)):
            if type(weightList)!=list:
                self.weights.append(1.)
            elif len(weightList)<=n:
                self.weights.append(1.)
            else:
                self.weights.append(weightList[n])
            self.bsplines.append(BsplineFourier.Bspline(coefFile=BsplineFileList[n],shape=None,timeMap=timeMapList[n],spacing=None,fileScale=fileScale,delimiter=' ',origin=None))
    def addImgVecFile(self,imgVecFileList=None,timeMapList=None,weightList=None,fileScale=1.):
        ''' 
        add (multiple) BspineFile to solver
        Parameters:
            BsplineFileList: list(file_path)
                List of Bspine File(s) to read and add to solver 
            timeMapList:list([fromTime,toTime],[fromTime,toTime],...)
                Corresponding fromtime->toTime map of bspline
            weightList:list(float), optional, defaults to 1
                Corresponding weight of bspline
            fileScale:float, optional, defaults to 1
                scale of the bsplinefile to resultant bsplinefourier
        '''
        if type(imgVecFileList)!=list:
            imgVecFileList=[imgVecFileList]
            timeMapList=[timeMapList]
            if type(weightList)!=type(None):
                weightList=[weightList]
        for n in range(len(imgVecFileList)):
            if type(weightList)!=list:
                self.weights.append(1.)
            elif len(weightList)<=n:
                self.weights.append(1.)
            else:
                self.weights.append(weightList[n])
            self.bsplines.append(BsplineFourier.ImageVector(coefFile=imgVecFileList[n],timeMap=timeMapList[n],fileScale=fileScale,delimiter=' ',origin=None))
    def addBspline(self,BsplineList,weightList=None):
        ''' 
        add (multiple) Bspine to solver
        Parameters:
            BsplineList: list(BsplineFourier.Bspline)
                List of Bspine to add to solver
            weightList:list(float)
                Corresponding weight of bspline
        '''
        if type(BsplineList)!=list:
            BsplineList=[BsplineList]
        for n in range(len(BsplineList)):
            if type(weightList)!=list:
                self.weights.append(1.)
            elif len(weightList)<=n:
                self.weights.append(1.)
            else:
                self.weights.append(weightList[n])
            
            self.bsplines.append(BsplineList[n])
    def addImgVec(self,imgVecList,weightList=None):
        ''' 
        add (multiple) Bspine to solver
        Parameters:
            BsplineList: list(BsplineFourier.Bspline)
                List of Bspine to add to solver
            weightList:list(float)
                Corresponding weight of bspline
        '''
        if type(imgVecList)!=list:
            imgVecList=[imgVecList]
        for n in range(len(imgVecList)):
            if type(weightList)!=list:
                self.weights.append(1.)
            elif len(weightList)<=n:
                self.weights.append(1.)
            else:
                self.weights.append(weightList[n])
            
            self.bsplines.append(imgVecList[n])
    def initialize(self,shape=None,spacing=None,origin=None,period=1.,fourierTerms=3,spacingDivision=2.,gap=1):
        ''' 
        Initialize the solver
        Parameters:
            shape=[x,y,z,f,uvw]: list(float)
                shape of resultant bsplineFourier
            spacing=[x,y,z,period]:list(float)
                shape of resultant bsplineFourier
            origin=[x,y,z,t]:list(float)
                shape of resultant bsplineFourier
            period:float
                Period of resultant bsplineFourier (only used if spacing is undefined)
            fourierTerms:int
                Number of fourier terms in bsplineFourier (number of cosine or sine terms) (only used if shape is undefined)
            spacingDivision:float
                sampling points density between bsplineFourier grid
            gap:int
                number of sampling points near the boundary removed (1 means that all the sampling points at the boundary are removed)
        '''
        if type(shape)!=type(None) and type(spacing)!=type(None) and type(origin)!=type(None):
            self.bsFourier.initialize(shape,spacing=spacing,origin=origin)
        elif type(self.bsFourier.coef)==type(None):
            orishape=np.array([*self.bsplines[0].coef.shape[:-1],fourierTerms*2+1,self.bsplines[0].coef.shape[-1]])
            spacing=np.array([*self.bsplines[0].spacing,period])
            origin=np.array([*self.bsplines[0].origin,0.])
            self.bsFourier.initialize(orishape,spacing=spacing,origin=origin)
        if type(shape)!=type(None):
            if self.bsFourier.coef.shape!=shape:
                self.bsFourier=self.bsFourier.reshape(shape)
                logger.info('Adjusted to:')
                logger.info('    shape= '+str(self.bsFourier.coef.shape))
                logger.info('    origin= '+str(self.bsFourier.origin))
                logger.info('    spacing= '+str(self.bsFourier.spacing))
        self.points=np.array(self.bsFourier.samplePoints(spacingDivision=spacingDivision,gap=gap))
        logger.info('Initialized with '+str(len(self.points))+'points.')

        
    def solve(self,tRef=None,maxError=0.00001,maxIteration=1000,convergence=0.8,method='pointbypoint',reportevery=1000,tempSave=None,resume=False,rmsBasedWeighted=None,linearConstrainPoints=[],linearConstrainWeight=None):
        ''' 
        Solves for the bsplineFourier
        Parameters:
            maxError: float
                maximum change in coefficients of bsplinefourier to consider converged
            maxIteration:int
                maximum number of iterations
            convergence:float
                maximum ratio of change in coefficient to current coefficient
            method:str, optional, defaults to pointbypoint
                Period of resultant bsplineFourier (only used if spacing is undefined)
            reportevery:int
                print output to report (or save) progress every "reportevery" points solved
            tempSave:str
                file_path to save sampling results
            resume:int
                resume solving with results from tempSave
            rmsBasedWeighted: function, optionsl, defaults to output=input
                function to map rms error to regriding weights
        '''
        if method=='pointbypoint':
            sampleCoefList,rmsList=self.solve_pointbypoint(maxError=maxError,tRef=tRef,maxIteration=maxIteration,convergence=convergence,reportevery=reportevery,tempSave=tempSave,resume=resume)
            if type(rmsBasedWeighted)==type(None):
                rmsweight=None
            else:
                rmsweight=rmsBasedWeighted(rmsList)
        self.bsFourier.regrid(self.points,sampleCoefList,weight=rmsweight,linearConstrainPoints=linearConstrainPoints,linearConstrainWeight=linearConstrainWeight)
        logger.info('BsplineFourier updated')
  
    def solve_pointbypoint(self,tRef=None,maxError=0.00001,maxIteration=1000,convergence=0.8,reportevery=1000,tempSave=None,resume=False,movAvgError=False,lmLambda_init=0.001,lmLambda_incrRatio=5.,lmLambda_max=float('inf'),lmLambda_min=0.00001):
        ''' 
        Solves for the bsplineFourier
        Parameters:
            maxError: float
                maximum change in coefficients of bsplinefourier to consider converged
            maxIteration:int
                maximum number of iterations
            convergence:float
                maximum ratio of change in coefficient to current coefficient
            method:str, optional, defaults to pointbypoint
                Period of resultant bsplineFourier (only used if spacing is undefined)
            reportevery:int
                print output to report (or save) progress every "reportevery" points solved
            tempSave:str
                file_path to save sampling results
            resume:int
                resume solving with results from tempSave
            movAvgError: bool
                determine whether to use moving average error instead of current error
            lmLambda_init: float
                initial Lambda value for Levenberg-Marquardt algorithm
            lmLambda_incrRatio: float
                Ratio to increase of decrease Lambda value for Levenberg-Marquardt algorithm
            lmLambda_max: float
                Maximum Lambda value for Levenberg-Marquardt algorithm
            lmLambda_min: float
                Minimum Lambda value for Levenberg-Marquardt algorithm
        '''
        wdiag=np.ones(0)
        for weight in self.weights:
            wdiag=np.concatenate((wdiag,np.ones(self.points.shape[-1])*weight),axis=0)
        rmsList=[]
        if not(resume):
            self.pointsCoef=[]
        elif type(tempSave)!=type(None):
            self.loadSamplingResults(tempSave)
        for m in range(len(self.pointsCoef),len(self.points)):
            coef=self.bsFourier.getRefCoef(self.points[m])
            coef[0]=0
            if type(tRef)!=type(None):
                coef[0]=-getCoordfromCoef(np.array([*self.points[m],tRef-self.bsFourier.origin[self.points.shape[-1]]]),coef,self.bsFourier.spacing)
            coef_start=coef.copy()
            error=float('inf')
            count=0.
            fx=[]
            pointX=[]
            for n in range(len(self.bsplines)):
                Y=getCoordfromCoef(np.array([*self.points[m],self.bsplines[n].timeMap[1]-self.bsFourier.origin[self.points.shape[-1]]]),coef,self.bsFourier.spacing)+self.points[m]
                X=getCoordfromCoef(np.array([*self.points[m],self.bsplines[n].timeMap[0]-self.bsFourier.origin[self.points.shape[-1]]]),coef,self.bsFourier.spacing)+self.points[m]
                V=np.array(self.bsplines[n].getVector(X))
                pointX.append(X.copy())
                fx.append(Y-X-V)
            fx=np.array(fx).reshape((-1,),order='C')
            rms=np.sqrt(np.mean(wdiag*fx**2.))
            rmsStart=rms
            lmLambda=lmLambda_init
            reductionRatio=1.
            if movAvgError:
                movAvgError=rms*10.
                error=rms
            while error>maxError and count<maxIteration:
                Jmat=np.zeros((len(fx),(coef.shape[0]-1)*coef.shape[1]))
                
                Jcount=0
                for n in range(len(self.bsplines)):
                    fourierdX,fourierdY=self.bsFourier.getdXYdC(self.bsplines[n].timeMap,remove0=True)
                    if type(tRef)!=type(None):
                        fourierRef,=self.bsFourier.getdXYdC([tRef],remove0=True)
                        fourierdX=fourierdX-fourierRef
                        fourierdY=fourierdY-fourierRef
                    dVdX=self.bsplines[n].getdX(pointX[n])
                    for axis in range(self.points.shape[-1]):
                        Jmattemp=[]
                        for xyz in range(self.points.shape[-1]):
                            if axis==xyz:
                                Jmattemp.append(fourierdY-fourierdX*(1.+dVdX[xyz,axis]))
                            else:
                                Jmattemp.append(-fourierdX*dVdX[xyz,axis])   
                        Jmat[Jcount]=np.array(Jmattemp).reshape(-1).copy()
                        Jcount+=1
                matA=Jmat.transpose().dot(np.diag(wdiag).dot(Jmat))
                matA=matA+lmLambda*np.diag(np.diag(matA))##lavenberg-Marquardt correction
                
                natb=Jmat.transpose().dot((-wdiag*fx).transpose())
                dCoef=np.linalg.solve(matA, natb).reshape((coef.shape[0]-1,coef.shape[1]),order='F')
                
                #add constrains
                ####
                
                #calculate error
                
                error=0.
                for n in range(coef.shape[0]-1):
                    for nn in range(coef.shape[1]):
                        if coef[n+1,nn]==0:
                            error=max(error,np.abs(dCoef[n,nn])/self.bsFourier.spacing[:self.points.shape[-1]].min())
                        else:
                            error=max(error,np.abs(dCoef[n,nn]).max()/np.abs(coef[n+1,nn]))     
                #renew
                
                ratio=reductionRatio
                if convergence:
                    if abs(dCoef).max()>self.bsFourier.spacing[:self.points.shape[-1]].min()*convergence:
                        ratio=min(ratio,self.bsFourier.spacing[:self.points.shape[-1]].min()*convergence/abs(dCoef).max())
                
                
                coef_backup=coef.copy()
                fx_backup=fx.copy()
                pointX_backup=pointX.copy()

                coef[1:,:]+=ratio*dCoef
                coef[0]=0
                if type(tRef)!=type(None):
                    coef[0]=-getCoordfromCoef(np.array([*self.points[m],tRef-self.bsFourier.origin[self.points.shape[-1]]]),coef,self.bsFourier.spacing)
                fx=[]
                pointX=[]
                for n in range(len(self.bsplines)):
                    Y=getCoordfromCoef(np.array([*self.points[m],self.bsplines[n].timeMap[1]-self.bsFourier.origin[self.points.shape[-1]]]),coef,self.bsFourier.spacing)+self.points[m]
                    X=getCoordfromCoef(np.array([*self.points[m],self.bsplines[n].timeMap[0]-self.bsFourier.origin[self.points.shape[-1]]]),coef,self.bsFourier.spacing)+self.points[m]
                    V=np.array(self.bsplines[n].getVector(X))
                    pointX.append(X.copy())
                    fx.append(Y-X-V)
                fx=np.array(fx).reshape((-1,),order='C')
                deltarms=np.sqrt(np.mean(wdiag*fx**2.))-rms
                if deltarms>0.:
                    coef=coef_backup.copy()
                    fx=fx_backup.copy()
                    pointX=pointX_backup.copy()
                    #error=float('inf')
                    if (lmLambda*lmLambda_incrRatio)<lmLambda_max:
                        lmLambda*=lmLambda_incrRatio
                    else:
                        lmLambda=lmLambda_max
                        reductionRatio*=0.8
                    count+=0.02
                else:
                    if ratio>0.9 and lmLambda!=lmLambda_min:
                        if (lmLambda/np.sqrt(lmLambda_incrRatio))>lmLambda_min:
                            lmLambda=lmLambda/np.sqrt(lmLambda_incrRatio)
                        else:
                            lmLambda=lmLambda_min
                    elif reductionRatio<0.9:
                        reductionRatio*=1.1
                    rms=np.sqrt(np.mean(wdiag*fx**2.))
                    if movAvgError:
                        error=np.abs(movAvgError-rms)/movAvgError
                        movAvgError=(movAvgError+rms)/2.
                    count+=1
            rmsList.append(rms)
            self.pointsCoef.append(coef.copy())
            if count==maxIteration:
                logger.warning('Maximum iterations reached for point '+str(m)+' '+str(self.points[m]))
            if m%reportevery==0:
                logger.info('Solved for point '+str(m+1)+'/'+str(len(self.points))+' '+str(self.points[m])+',rms start= '+str(rmsStart)+'rms end= '+str(rms)+',max rms= '+str(max(rmsList)))
                if type(tempSave)!=type(None):
                    self.writeSamplingResults(tempSave)
        if type(tempSave)!=type(None):
            self.writeSamplingResults(tempSave)
        rmsList=np.array(rmsList)
        return (self.pointsCoef,rmsList)
    def addEquation(self,equation_AD,eqnToPts=None,weight=1.):
        self.eqn.append(equation_AD)
        if type(eqnToPts)==type(None):
            eqnToPts=range(len(self.points))
        self.eqnWeight.append(weight)
        self.eqnToPts.append(eqnToPts)
    def solve_full(self,tRef=None,flexibleDescent=0,tryDirectSolve=True,minMEMuse=0,maxError=0.00001,maxIteration=1000,convergence=0.8,reportevery=1,tempSave=None,resume=False,movAvgError=False,lmLambda_init=0.001,lmLambda_incrRatio=5.,lmLambda_max=float('inf'),lmLambda_min=0.00001):
        ''' 
        Solves for the bsplineFourier
        Parameters:
            maxError: float
                maximum change in coefficients of bsplinefourier to consider converged
            maxIteration:int
                maximum number of iterations
            convergence:float
                maximum ratio of change in coefficient to current coefficient
            method:str, optional, defaults to pointbypoint
                Period of resultant bsplineFourier (only used if spacing is undefined)
            reportevery:int
                print output to report (or save) progress every "reportevery" points solved
            tempSave:str
                file_path to save sampling results
            resume:int
                resume solving with results from tempSave
            movAvgError: bool
                determine whether to use moving average error instead of current error
            lmLambda_init: float
                initial Lambda value for Levenberg-Marquardt algorithm
            lmLambda_incrRatio: float
                Ratio to increase of decrease Lambda value for Levenberg-Marquardt algorithm
            lmLambda_max: float
                Maximum Lambda value for Levenberg-Marquardt algorithm
            lmLambda_min: float
                Minimum Lambda value for Levenberg-Marquardt algorithm
        '''
        '''
        eqn=[]
        for n in range(len(self.bsplines)):
            eqn.append([self.bsFourier.toBsplineU(self.bsplines[n])-self.bsFourier.U(tVal=self.bsplines[n].timeMap[1]),self.bsFourier.toBsplineV(self.bsplines[n])-self.bsFourier.V(tVal=self.bsplines[n].timeMap[1]),self.bsFourier.toBsplineW(self.bsplines[n])-self.bsFourier.W(tVal=self.bsplines[n].timeMap[1])])
        '''
        wdiag=np.ones(0)
        for n in range(len(self.eqn)):
            wdiag=np.concatenate((wdiag,np.ones(len(self.eqnToPts[n]))*self.eqnWeight[n]),axis=0)        

        error=float('inf')

        numCoef=self.bsFourier.numCoef*3
        logger.info('Solving '+str(numCoef)+'coefficients...')
        
        fxstarmapInput=np.empty( (len(self.points),3), dtype=object)
        for m in range(len(self.points)):
            fxstarmapInput[m,0]={'x':self.points[m][0],'y':self.points[m][1],'z':self.points[m][2]}
            fxstarmapInput[m,1]={}
            fxstarmapInput[m,2]={'C':1}

        fx=np.zeros(0)
        for n in range(len(self.eqn)):
            '''
            if n%1==0:
                sys.stdout.write("\rCalculating fx: {0:.2f}%".format(n/len(self.eqn)*100))
                sys.stdout.flush()
            '''
            if len(self.eqnToPts[n])<multiprocessing.cpu_count():
                fx_temp=[]
                for ptn in range(len(self.eqnToPts[n])):
                    fx_temp.append(self.eqn[n](*fxstarmapInput[self.eqnToPts[n]][ptn,[0,1]]))
            elif minMEMuse:
                pool = multiprocessing.Pool(multiprocessing.cpu_count(),maxtasksperchild=minMEMuse)
                fx_temp=pool.starmap(self.eqn[n],fxstarmapInput[self.eqnToPts[n]][:,[0,1]],chunksize=minMEMuse)
            else:
                pool = multiprocessing.Pool(multiprocessing.cpu_count())
                fx_temp=pool.starmap(self.eqn[n],fxstarmapInput[self.eqnToPts[n]][:,[0,1]])
            pool.close()
            pool.join()
            logger.info('Equation '+str(n)+': rms= '+str(np.sqrt(np.mean(np.array(fx_temp)**2.)))+', wrms= '+str(self.eqnWeight[n]*np.sqrt(np.mean(np.array(fx_temp)**2.))))
            fx=np.concatenate((fx,np.array(fx_temp)))
        sys.stdout.write("\rCalculating fx: 100.00%")
        sys.stdout.flush()
        del fx_temp
        rms=np.sqrt(np.mean(wdiag*fx**2.))
        rmsStart=rms
        lmLambda=lmLambda_init
        reductionRatio=1.
        if movAvgError:
            movAvgError=rms*10.
            error=rms
        count=0.
        flexiCount=0
        recalculateJmat=True
        if type(tempSave)!=type(None):
            temp_path = tempSave[:-4]+'_'
        else:
            temp_path ='/tmp/'
        #tsm=toSparseMatrix(numCoef)
        while error>maxError and count<maxIteration:
            logger.info(' Iteration '+str(count)+'rms= '+str(rms))
            np.savetxt(temp_path+'fx.txt',fx)
            if recalculateJmat or flexibleDescent>0:
                #Jmat=np.empty(0,dtype=object)
                #natEq=np.zeros(0)
                Jmat=[]
                natEq=[]
                for n in range(len(self.eqn)):
                    if n%1==0:
                        sys.stdout.write("\rCalculating dC: {0:.2f}%".format(n/len(self.eqn)*100))
                        sys.stdout.flush()
                    if len(self.eqnToPts[n])<multiprocessing.cpu_count():
                        tempJn=[]
                        for ptn in range(len(self.eqnToPts[n])):
                            tempJn.append(BsplineFourier.toSparseMatrixAD(self.eqn[n],numCoef)(*fxstarmapInput[self.eqnToPts[n]][ptn,[0,2]]))
                    elif minMEMuse:
                        pool = multiprocessing.Pool(multiprocessing.cpu_count(),maxtasksperchild=minMEMuse)
                        tempJn=pool.starmap(BsplineFourier.toSparseMatrixAD(self.eqn[n],numCoef),fxstarmapInput[self.eqnToPts[n]][:,[0,2]],chunksize=minMEMuse)
                    else:
                        pool = multiprocessing.Pool(multiprocessing.cpu_count())
                        tempJn=pool.starmap(BsplineFourier.toSparseMatrixAD(self.eqn[n],numCoef),fxstarmapInput[self.eqnToPts[n]][:,[0,2]])    
                    pool.close()
                    pool.join()
                    
                    tempJn=np.array(tempJn,dtype=object)
                    Jmat+=list(tempJn[:,0])
                    natEq+=list(tempJn[:,1])
                    '''
                    for m in range(len(self.eqnToPts[n])):
                        matrow,natrow=tempJn[m].toSparseMatrix(numCoef)
                        Jmat.append(matrow.copy())
                        natEq.append(natrow)
                    '''    
                sys.stdout.write("\rCalculating dC: 100.00%")
                sys.stdout.flush()
                Jmat=sparse.vstack(Jmat)
                #del tempMatNat
                del tempJn
                if Jmat.shape[0]<Jmat.shape[1]:
                    raise Exception('Not enough equations to support solving!')
                natEq=np.array(natEq)
                if count==0:
                    logger.info('Eqn constant= '+str(np.sqrt(np.mean(natEq**2.))))
            else:
                Jmat=sparse.load_npz(temp_path+'lastJmat.npz')
                recalculateJmat=True
            
            matA=Jmat.transpose().dot(sparse.diags(wdiag).dot(Jmat))
            matA=matA+sparse.diags(lmLambda*matA.diagonal())
            natb=Jmat.transpose().dot((wdiag*(-natEq-fx)))
            
            #stores and clear Jmat
            sparse.save_npz(temp_path+'lastJmat.npz',Jmat)
            del Jmat
            
            dC=None            
            if tryDirectSolve:
                try:
                    if 'sksparse' in sys.modules:
                        dC=cholesky(matA)
                        dC=dC(natb)
                    else:
                        dC=sparse.linalg.spsolve(matA,natb)
                except Exception as e:
                    dC=None 
                    tryDirectSolve=False
                    logger.warning(str(e))
                    logger.warning('Direct solve unsuccessful, trying indirect solving...')
            if type(dC)==type(None):
                dC=sparse.linalg.bicgstab(matA, natb)#,x0=natb/matA.diagonal())
                if dC[1]==0:
                    logger.info(' successful.')
                elif dC[1]>0:
                    logger.warning(' convergence to tolerance not achieved, number of iterations '+str(dC[1]))
                    if (lmLambda*lmLambda_incrRatio)<lmLambda_max:
                        lmLambda*=lmLambda_incrRatio
                    else:
                        lmLambda=lmLambda_max
                        reductionRatio*=0.8
                    count+=1
                    recalculateJmat=False
                    continue
                else:
                    raise Exception(' illegal input or breakdown')
                dC=dC[0]
            if type(dC)!=np.ndarray:
                dC=dC.todense()
            dC=dC.reshape((3,self.bsFourier.coef.shape[3]-1,self.bsFourier.coef.shape[0],self.bsFourier.coef.shape[1],self.bsFourier.coef.shape[2]),order='F').transpose([2,3,4,1,0])
            
            if np.abs(self.bsFourier.coef[:,:,:,1:]).max()==0:
                error=np.abs(dC).max()/self.bsFourier.spacing[:3].min()
            else:
                error=np.abs(dC).max()/np.abs(self.bsFourier.coef[:,:,:,1:]).max()     

            ratio=reductionRatio
            if convergence:
                if np.abs(dC).max()>self.bsFourier.spacing[:3].min()*convergence:
                    ratio=min(ratio,self.bsFourier.spacing[:3].min()*convergence/np.abs(dC).max())
            if flexiCount==0:
                coef_backup=self.bsFourier.coef.copy()
                fx_backup=fx.copy()
            self.bsFourier.coef[:,:,:,1:]+=dC*ratio
            if type(tRef)!=type(None):
                fourierX,fourierY=self.bsFourier.getdXYdC([0,tRef])
                self.bsFourier.coef[:,:,:,0,0]=-self.bsFourier.coef[:,:,:,1:,0].dot(fourierY)
                self.bsFourier.coef[:,:,:,0,1]=-self.bsFourier.coef[:,:,:,1:,1].dot(fourierY)
                self.bsFourier.coef[:,:,:,0,2]=-self.bsFourier.coef[:,:,:,1:,2].dot(fourierY)
            fx=np.zeros(0)
            for n in range(len(self.eqn)):
                if n%1==0:
                    sys.stdout.write("\rCalculating fx: {0:.2f}%".format(n/len(self.eqn)*100))
                    sys.stdout.flush()
                if len(self.eqnToPts[n])<multiprocessing.cpu_count():
                    fx_temp=[]
                    for ptn in range(len(self.eqnToPts[n])):
                        fx_temp.append(self.eqn[n](*fxstarmapInput[self.eqnToPts[n]][ptn,[0,1]]))
                elif minMEMuse:
                    pool = multiprocessing.Pool(multiprocessing.cpu_count(),maxtasksperchild=minMEMuse)
                    fx_temp=pool.starmap(self.eqn[n],fxstarmapInput[self.eqnToPts[n]][:,[0,1]],chunksize=minMEMuse)
                else:
                    pool = multiprocessing.Pool(multiprocessing.cpu_count())
                    fx_temp=pool.starmap(self.eqn[n],fxstarmapInput[self.eqnToPts[n]][:,[0,1]])
                pool.close()
                pool.join()
                fx=np.concatenate((fx,np.array(fx_temp)))
            sys.stdout.write("\rCalculating fx: 100.00%")
            sys.stdout.flush()
            del fx_temp
            deltarms=np.sqrt(np.mean(wdiag*fx**2.))-rms
            if deltarms>0. and flexiCount>=flexibleDescent:
                logger.info('Reverting back to iteration',count-flexiCount,'.')
                if flexibleDescent==0:
                    recalculateJmat=False
                self.bsFourier.coef=coef_backup.copy()
                fx=fx_backup.copy()
                rms=np.sqrt(np.mean(wdiag*fx**2.))
                #error=float('inf')
                if (lmLambda*lmLambda_incrRatio)<lmLambda_max:
                    lmLambda*=lmLambda_incrRatio
                else:
                    lmLambda=lmLambda_max
                    reductionRatio*=0.8
                count+=1-flexiCount
                flexiCount=0
                logger.info(' drms= '+str(deltarms))
            else:
                if deltarms>0.:
                    flexiCount+=1
                else:
                    if type(tempSave)!=type(None):
                        self.bsFourier.writeCoef(tempSave)
                    flexiCount=0
                    if ratio>0.9 and lmLambda!=lmLambda_min:
                        if (lmLambda/np.sqrt(lmLambda_incrRatio))>lmLambda_min:
                            lmLambda=lmLambda/np.sqrt(lmLambda_incrRatio)
                        else:
                            lmLambda=lmLambda_min
                    elif reductionRatio<0.9:
                        reductionRatio*=1.1
                rms=np.sqrt(np.mean(wdiag*fx**2.))
                if movAvgError:
                    error=np.abs(movAvgError-rms)/movAvgError
                    movAvgError=(movAvgError+rms)/2.
                count+=1
        os.remove(temp_path+'lastJmat.npz')
    def solve_eqn_from_pbp(self,tRef=None,flexibleDescent=0,tryDirectSolve=True,minMEMuse=0,maxError=0.00001,maxIteration=1000,convergence=0.8,reportevery=1,tempSave=None,resume=False,movAvgError=False,lmLambda_init=0.001,lmLambda_incrRatio=5.,lmLambda_max=float('inf'),lmLambda_min=0.00001):
        ''' 
        Solves for the bsplineFourier
        Parameters:
            maxError: float
                maximum change in coefficients of bsplinefourier to consider converged
            maxIteration:int
                maximum number of iterations
            convergence:float
                maximum ratio of change in coefficient to current coefficient
            method:str, optional, defaults to pointbypoint
                Period of resultant bsplineFourier (only used if spacing is undefined)
            reportevery:int
                print output to report (or save) progress every "reportevery" points solved
            tempSave:str
                file_path to save sampling results
            resume:int
                resume solving with results from tempSave
            movAvgError: bool
                determine whether to use moving average error instead of current error
            lmLambda_init: float
                initial Lambda value for Levenberg-Marquardt algorithm
            lmLambda_incrRatio: float
                Ratio to increase of decrease Lambda value for Levenberg-Marquardt algorithm
            lmLambda_max: float
                Maximum Lambda value for Levenberg-Marquardt algorithm
            lmLambda_min: float
                Minimum Lambda value for Levenberg-Marquardt algorithm
        '''
        if type(tempSave)!=type(None):
            temp_path = tempSave[:-4]+'_'
        else:
            temp_path ='/tmp/'
        numCoef=self.bsFourier.numCoef*3
        logger.info('Solving '+str(numCoef)+'coefficients...')
        
        wdiag=np.ones(0)#len(self.pointsCoef)*(len(self.pointsCoef[0])-1)*3)
        for n in range(len(self.eqn)):
            wdiag=np.concatenate((wdiag,np.ones(len(self.eqnToPts[n]))*self.eqnWeight[n]),axis=0)        
        error=float('inf')
        
        fxstarmapInput=np.empty( (len(self.points),3), dtype=object)
        for m in range(len(self.points)):
            fxstarmapInput[m,0]={'x':self.points[m][0],'y':self.points[m][1],'z':self.points[m][2]}
            fxstarmapInput[m,1]={}
            fxstarmapInput[m,2]={'C':1}

        #set base matrix
        fx_base=np.array(self.pointsCoef)[:,1:].reshape(-1)
        Jmat_base=[]
        dCList,CIndList=self.bsFourier.getdC(self.points)
        for n in range(len(self.pointsCoef)):
            for fn in range(1,self.bsFourier.coef.shape[3]):
                for axis in range(3):
                    Jmat_base.append(sparse.csr_matrix((np.array(dCList[n]),(np.zeros(len(CIndList[n])),np.array(CIndList[n])*(self.bsFourier.coef.shape[3]-1)*3+(fn-1)*3+axis)),shape=(1,numCoef)))
        Jmat_base=sparse.vstack(Jmat_base)
        #sparse.save_npz(temp_path+'baseJmat.npz',Jmat_base)
        if Jmat_base.shape[0]!=fx_base.shape[0]:
            raise Exception('ERROR length of fx and Jmat base not matched')
        #del Jmat_base
        del dCList
        del CIndList
        logger.info('Base equation : rms= '+str(np.sqrt(np.mean((fx_base-Jmat_base.dot(self.bsFourier.coef[:,:,:,1:].transpose(4,3,0,1,2).reshape(-1,order='F')))**2.))))
        fx=np.zeros(0)
        for n in range(len(self.eqn)):
            '''
            if n%1==0:
                sys.stdout.write("\rCalculating fx: {0:.2f}%".format(n/len(self.eqn)*100))
                sys.stdout.flush()
            '''
            if len(self.eqnToPts[n])<multiprocessing.cpu_count():
                fx_temp=[]
                for ptn in range(len(self.eqnToPts[n])):
                    fx_temp.append(self.eqn[n](*fxstarmapInput[self.eqnToPts[n]][ptn,[0,1]]))
            elif minMEMuse:
                pool = multiprocessing.Pool(multiprocessing.cpu_count(),maxtasksperchild=minMEMuse)
                fx_temp=pool.starmap(self.eqn[n],fxstarmapInput[self.eqnToPts[n]][:,[0,1]],chunksize=minMEMuse)
            else:
                pool = multiprocessing.Pool(multiprocessing.cpu_count())
                fx_temp=pool.starmap(self.eqn[n],fxstarmapInput[self.eqnToPts[n]][:,[0,1]])
            pool.close()
            pool.join()
            logger.info('Equation '+str(n)+': rms= '+str(np.sqrt(np.mean(np.array(fx_temp)**2.)))+', wrms= '+str(self.eqnWeight[n]*np.sqrt(np.mean(np.array(fx_temp)**2.))))
            fx=np.concatenate((fx,np.array(fx_temp)))
        del fx_temp

        rms=np.sqrt(np.mean(np.concatenate((fx_base-Jmat_base.dot(self.bsFourier.coef[:,:,:,1:].transpose(4,3,0,1,2).reshape(-1,order='F')),wdiag*fx))**2.))
        rmsStart=rms
        lmLambda=lmLambda_init
        reductionRatio=1.
        if movAvgError:
            movAvgError=rms*10.
            error=rms
        count=0.
        flexiCount=0
        recalculateJmat=True
        
        #tsm=toSparseMatrix(numCoef)
        while error>maxError and count<maxIteration:
            logger.info(' Iteration '+str(count)+'rms= '+str(rms))
            np.savetxt(temp_path+'fx.txt',fx)
            if recalculateJmat or flexibleDescent>0:
                #Jmat=np.empty(0,dtype=object)
                #natEq=np.zeros(0)
                Jmat=[]
                natEq=[]
                for n in range(len(self.eqn)):
                    if n%1==0:
                        sys.stdout.write("\rCalculating dC: {0:.2f}%".format(n/len(self.eqn)*100))
                        sys.stdout.flush()
                    if len(self.eqnToPts[n])<multiprocessing.cpu_count():
                        tempJn=[]
                        for ptn in range(len(self.eqnToPts[n])):
                            tempJn.append(BsplineFourier.toSparseMatrixAD(self.eqn[n],numCoef)(*fxstarmapInput[self.eqnToPts[n]][ptn,[0,2]]))
                    elif minMEMuse:
                        pool = multiprocessing.Pool(multiprocessing.cpu_count(),maxtasksperchild=minMEMuse)
                        tempJn=pool.starmap(BsplineFourier.toSparseMatrixAD(self.eqn[n],numCoef),fxstarmapInput[self.eqnToPts[n]][:,[0,2]],chunksize=minMEMuse)
                    else:
                        pool = multiprocessing.Pool(multiprocessing.cpu_count())
                        tempJn=pool.starmap(BsplineFourier.toSparseMatrixAD(self.eqn[n],numCoef),fxstarmapInput[self.eqnToPts[n]][:,[0,2]])    
                    pool.close()
                    pool.join()
                    
                    tempJn=np.array(tempJn,dtype=object)
                    Jmat+=list(tempJn[:,0])
                    natEq+=list(tempJn[:,1])
                    '''
                    for m in range(len(self.eqnToPts[n])):
                        matrow,natrow=tempJn[m].toSparseMatrix(numCoef)
                        Jmat.append(matrow.copy())
                        natEq.append(natrow)
                    '''    
                sys.stdout.write("\rCalculating dC: 100.00%")
                sys.stdout.flush()
                Jmat=sparse.vstack(Jmat)
                #del tempMatNat
                del tempJn
                natEq=np.array(natEq)
                if count==0:
                    logger.info('Eqn constant= '+str(np.sqrt(np.mean(natEq**2.))))
            else:
                Jmat=sparse.load_npz(temp_path+'lastJmat.npz')
                recalculateJmat=True

            sparse.save_npz(temp_path+'lastJmat.npz',Jmat)
            
            #Jmat_base=sparse.load_npz(temp_path+'baseJmat.npz')
            Jmat_combine=Jmat.transpose().dot(sparse.diags(wdiag).dot(Jmat))
            Jmat_combine=Jmat_combine+sparse.diags(lmLambda*Jmat_combine.diagonal())
            matA=sparse.vstack([Jmat_base,Jmat_combine])

            natb=matA.transpose().dot(np.concatenate((fx_base,Jmat.transpose().dot(wdiag*(-natEq-fx))+Jmat_combine.dot(self.bsFourier.coef[:,:,:,1:].transpose(4,3,0,1,2).reshape(-1,order='F')))))
            matA=matA.transpose().dot(matA)
            #matA=matA+sparse.diags(lmLambda*matA.diagonal())
            
            #stores and clear Jmat
            del Jmat_combine
            del Jmat
            #del Jmat_base
            
            newC=None            
            if tryDirectSolve:
                try:
                    if 'sksparse' in sys.modules:
                        newC=cholesky(matA)
                        newC=newC(natb)
                    else:
                        newC=sparse.linalg.spsolve(matA,natb)
                except Exception as e:
                    newC=None 
                    tryDirectSolve=False
                    logger.warning(str(e))
                    logger.warning('Direct solve unsuccessful, trying indirect solving...')
            if type(newC)==type(None):
                newC=sparse.linalg.bicgstab(matA, natb)#,x0=natb/matA.diagonal())
                if newC[1]==0:
                    logger.info(' successful.')
                elif newC[1]>0:
                    logger.warning(' convergence to tolerance not achieved, number of iterations '+str(dC[1]))
                    if (lmLambda*lmLambda_incrRatio)<lmLambda_max:
                        lmLambda*=lmLambda_incrRatio
                    else:
                        lmLambda=lmLambda_max
                        reductionRatio*=0.8
                    count+=1
                    recalculateJmat=False
                    continue
                else:
                    raise Exception(' illegal input or breakdown')
                newC=newC[0]
            if type(newC)!=np.ndarray:
                newC=newC.todense()
            newC=newC.reshape((3,self.bsFourier.coef.shape[3]-1,self.bsFourier.coef.shape[0],self.bsFourier.coef.shape[1],self.bsFourier.coef.shape[2]),order='F').transpose([2,3,4,1,0])
            
            if np.abs(self.bsFourier.coef[:,:,:,1:]).max()==0:
                error=np.abs(newC-self.bsFourier.coef[:,:,:,1:]).max()/self.bsFourier.spacing[:3].min()
            else:
                error=np.abs(newC-self.bsFourier.coef[:,:,:,1:]).max()/np.abs(self.bsFourier.coef[:,:,:,1:]).max()     

            ratio=reductionRatio
            if convergence:
                if np.abs(newC-self.bsFourier.coef[:,:,:,1:]).max()>self.bsFourier.spacing[:3].min()*convergence:
                    ratio=min(ratio,self.bsFourier.spacing[:3].min()*convergence/np.abs(newC-self.bsFourier.coef[:,:,:,1:]).max())
            if flexiCount==0:
                coef_backup=self.bsFourier.coef.copy()
                fx_backup=fx.copy()
            self.bsFourier.coef[:,:,:,1:]=ratio*newC+self.bsFourier.coef[:,:,:,1:]*(1-ratio)
            if type(tRef)!=type(None):
                fourierX,fourierY=self.bsFourier.getdXYdC([0,tRef])
                self.bsFourier.coef[:,:,:,0,0]=-self.bsFourier.coef[:,:,:,1:,0].dot(fourierY)
                self.bsFourier.coef[:,:,:,0,1]=-self.bsFourier.coef[:,:,:,1:,1].dot(fourierY)
                self.bsFourier.coef[:,:,:,0,2]=-self.bsFourier.coef[:,:,:,1:,2].dot(fourierY)
            fx=np.zeros(0)
            for n in range(len(self.eqn)):
                if n%1==0:
                    sys.stdout.write("\rCalculating fx: {0:.2f}%".format(n/len(self.eqn)*100))
                    sys.stdout.flush()
                if len(self.eqnToPts[n])<multiprocessing.cpu_count():
                    fx_temp=[]
                    for ptn in range(len(self.eqnToPts[n])):
                        fx_temp.append(self.eqn[n](*fxstarmapInput[self.eqnToPts[n]][ptn,[0,1]]))
                elif minMEMuse:
                    pool = multiprocessing.Pool(multiprocessing.cpu_count(),maxtasksperchild=minMEMuse)
                    fx_temp=pool.starmap(self.eqn[n],fxstarmapInput[self.eqnToPts[n]][:,[0,1]],chunksize=minMEMuse)
                else:
                    pool = multiprocessing.Pool(multiprocessing.cpu_count())
                    fx_temp=pool.starmap(self.eqn[n],fxstarmapInput[self.eqnToPts[n]][:,[0,1]])
                pool.close()
                pool.join()
                fx=np.concatenate((fx,np.array(fx_temp)))
            sys.stdout.write("\rCalculating fx: 100.00%")
            sys.stdout.flush()
            del fx_temp
            deltarms=np.sqrt(np.mean(np.concatenate((fx_base-Jmat_base.dot(self.bsFourier.coef[:,:,:,1:].transpose(4,3,0,1,2).reshape(-1,order='F')),wdiag*fx))**2.))-rms
            if deltarms>0. and flexiCount>=flexibleDescent:
                logger.info('Reverting back to iteration '+str(count-flexiCount)+'.')
                if flexibleDescent==0:
                    recalculateJmat=False
                self.bsFourier.coef=coef_backup.copy()
                fx=fx_backup.copy()
                rms=np.sqrt(np.mean(np.concatenate((fx_base-Jmat_base.dot(self.bsFourier.coef[:,:,:,1:].transpose(4,3,0,1,2).reshape(-1,order='F')),wdiag*fx))**2.))
                #error=float('inf')
                if (lmLambda*lmLambda_incrRatio)<lmLambda_max:
                    lmLambda*=lmLambda_incrRatio
                else:
                    lmLambda=lmLambda_max
                    reductionRatio*=0.8
                count+=1-flexiCount
                flexiCount=0
                logger.info(' drms= '+str(deltarms))
                #logger.info('reduce ratio to',reduceRatio)
            else:
                if deltarms>0.:
                    flexiCount+=1
                else:
                    if type(tempSave)!=type(None):
                        self.bsFourier.writeCoef(tempSave)
                    flexiCount=0
                    if ratio>0.9 and lmLambda!=lmLambda_min:
                        if (lmLambda/np.sqrt(lmLambda_incrRatio))>lmLambda_min:
                            lmLambda=lmLambda/np.sqrt(lmLambda_incrRatio)
                        else:
                            lmLambda=lmLambda_min
                    elif reductionRatio<0.9:
                        reductionRatio*=1.1
                rms=np.sqrt(np.mean(np.concatenate((fx_base-Jmat_base.dot(self.bsFourier.coef[:,:,:,1:].transpose(4,3,0,1,2).reshape(-1,order='F')),wdiag*fx))**2.))
                if movAvgError:
                    error=np.abs(movAvgError-rms)/movAvgError
                    movAvgError=(movAvgError+rms)/2.
                count+=1
        os.remove(temp_path+'lastJmat.npz')
    def superResolution(self,image,otherimages=None,tempSaveFile=None,dimExpand={},scheme='SAC',schemeArgs=None,xList=None,yList=None,zList=None,tList=None,mask=None,reportlevel=0,CPU=1):
        '''
        otherimages=[[image,bsplinefourier,initTransform],...]
        '''
        if type(otherimages)==list:
            if type(otherimages[0])!=list:
                otherimages=[otherimages]
        if scheme=='':
            logger.warning('No scheme selected. Proceding with mean scheme (scheme="weighted" or "SAC")')
        image=image.clone()
        image.rearrangeDim(['x','y','z','t'])
        resultImageshape=list(image.data.shape)
        stretch={}
        for dim in dimExpand:
            stretch[dim]=int(dimExpand[dim]*image.data.shape[image.dim.index(dim)])
            resultImageshape[image.dim.index(dim)]=stretch[dim]
        
        resultImage=image.clone()
        resultImageshape.pop(3)
        resultImage.stretch(stretch,stretchData=False)
        
        if type(xList)==type(None):
            xList=range(resultImageshape[0])
        else:
            resultImageshape[0]=len(xList)
        if type(yList)==type(None):
            yList=range(resultImageshape[1])
        else:
            resultImageshape[1]=len(yList)
        if type(zList)==type(None):
            zList=range(resultImageshape[2])
        else:
            resultImageshape[2]=len(zList)
        resultImage.data=np.zeros(resultImageshape).astype(image.data.dtype)
        
        if type(tList)==type(None):
            tList=range(image.data.shape[3])
        isArray=False
        if type(mask)==np.ndarray:
            isArray=True
        logger.info('Creating Image with shape '+str(resultImage.data.shape))
        if scheme=='SAC-G':
            if type(schemeArgs)==type(None) or type(schemeArgs)==float:
                Gsampling=medImgProc.image.gaussianSampling(resultImage.dim,image.dimlen)
            else:
                Gsampling=medImgProc.image.gaussianSampling(resultImage.dim,image.dimlen,variance=schemeArgs[1])
  
        for xn in range(len(xList)):
            if reportlevel>=0:
                logger.info('    {0:.3f}% completed...'.format(float(xn)/len(xList)*100.))
            for yn in range(len(yList)):
                if reportlevel>=1:
                    logger.info('        {0:.3f}% completed...'.format(float(yn)/len(yList)*100.))
                if scheme[:3]=='SAC' and CPU>1:
                    parallelArgs=[]
                    insertzList=[]
                for zn in range(len(zList)):
                    if reportlevel>=2:
                        logger.info('            {0:.3f}% completed...'.format(float(zn)/len(zList)*100.))
                    if type(mask)!=type(None) and isArray:
                        if mask[z,y,x]<1:
                          if len(val)==0:
                            value=0
                        elif (val.max()-val.min())<=3:
                            value=val.mean()
                        else:  continue
                    xyzRef=np.array([xList[xn]*resultImage.dimlen['x'],yList[yn]*resultImage.dimlen['y'],zList[zn]*resultImage.dimlen['z']])
                    if type(mask)!=type(None) and not(isArray):
                        if mask.getData(xyzRef,fill=0)<1:
                            continue
                    coord=[]
                    for t in tList:
                        coord.append([*xyzRef,t*image.dimlen['t']])
                    coords=self.bsFourier.getCoordFromRef(coord)
                    coords=np.hstack((np.array(coords),np.array(coord)[:,3].reshape((-1,1))))
                    if scheme=='weighted':
                        val=image.getData(coords,fill=0)
                        weight=np.sqrt((np.array(coords)[:,0]-xList[xn]*resultImage.dimlen['x'])**2.+(np.array(coords)[:,1]-yList[yn]*resultImage.dimlen['y'])**2.+(np.array(coords)[:,2]-zList[zn]*resultImage.dimlen['z'])**2.)
                        value=(np.array(val)*weight).sum()/weight.sum()
                    elif scheme[:3]=='SAC':
                        #schemeArgs[0] determines percentage agreement before it is considered accurate
                        #schemeArgs[1] controls the variance is gausian sampling is used
                        if type(schemeArgs)==type(None):
                            schemeArgs=[0.5,None]
                        elif type(schemeArgs)==float:
                            schemeArgs=[schemeArgs,None]
                        elif type(schemeArgs[0])==type(None):
                            schemeArgs[0]=0.5
                        elif schemeArgs[0]>=1.:
                            schemeArgs[0]=0.5
                        if scheme[-2:]=='-G':
                            val=image.getData(coords,fill=None,sampleFunction=Gsampling)
                            if type(val[0])==list:
                                val=sum(val,[])
                        else:
                            val=image.getData(coords,fill=None)
                        if type(otherimages)!=type(None):
                            for n in range(len(otherimages)):
                                tempimage=otherimages[n][0].clone()
                                tempimage.rearrangeDim(['x','y','z','t'])
                                tempval=[None]
                                tempCoord=[]
                                for t in range(tempimage.data.shape[3]):
                                    tempCoord.append([*xyzRef,t*otherimages[n][0].dimlen['t']])
                                tempcoords=otherimages[n][2].getVector(tempCoord)
                                tempcoords=np.hstack((np.array(tempcoords),np.array(tempCoord)[:,3].reshape((-1,1))))
                                tempcoords=otherimages[n][1].getCoordFromRef(tempcoords)
                                tempcoords=np.hstack((np.array(tempcoords),np.array(tempCoord)[:,3].reshape((-1,1))))
                                
                                if scheme[-2:]=='-G':
                                    tempval=tempimage.getData(tempcoords,fill=None,sampleFunction=Gsampling)
                                    if type(tempval[0])==list:
                                        tempval=sum(tempval,[])
                                else:
                                    tempval=tempimage.getData(tempcoords,fill=None)
                            val=val+tempval
                        val=np.array(list(filter(None.__ne__, val)))
                        val=val[val>=1]
                        if CPU>1:
                            parallelArgs.append([val.copy(),schemeArgs[0]])
                            insertzList.append(zn)
                            continue
                        else:
                            value=SAC(val.copy(),schemeArgs[0])
                            
                    elif scheme=='median':
                        val=image.getData(coords,fill=None)
                        val=np.array(list(filter(None.__ne__, val)))
                        if len(val)==0:
                            value=0
                        else:
                            value=np.median(val)
                    elif scheme=='contrast':
                        if type(schemeArgs)==type(None):
                            schemeArgs=3
                        val=image.getData(coords,fill=None)
                        val=np.array(list(filter(None.__ne__, val)))
                        if len(val)==0:
                            value=0
                        else:
                            diff=np.median(val)-val.mean()
                            if diff>schemeArgs:
                                value=val.max()
                            elif diff<-schemeArgs:
                                value=val.min()
                            else:
                                value=val.mean()
                    elif scheme=='threshold':
                        if type(schemeArgs)==type(None):
                            schemeArgs=image.data.min()/2.+image.data.max()/2.
                        val=image.getData(coords,fill=None)
                        val=np.array(list(filter(None.__ne__, val)))
                        if len(val)==0:
                            value=0
                        elif val[val>=schemeArgs].size>val.size:
                            value=255
                        else:
                            value=0
                    else:
                        val=image.getData(coords,fill=None)
                        val=np.array(list(filter(None.__ne__, val)))
                        if len(val)==0:
                            value=0
                        else:
                            value=val.mean()
                    if value>=255:
                        value=255
                    elif value<=0:
                        value=0
                    else:
                        value=int(np.around(value))
                    resultImage.data[xn,yn,zn]=value
                if scheme[:3]=='SAC' and CPU>1:
                    pool = multiprocessing.Pool(CPU)
                    resultImage.data[xn,yn,insertzList]=np.array(pool.starmap(SAC,parallelArgs))
                    pool.close()
                    pool.join()
            if type(tempSaveFile)!=type(None):
                resultImage.save(tempSaveFile)
        return resultImage
    def syncTo(self,image,dimExpand={},xList=None,yList=None,zList=None,tList=None,tempSaveFile=None,reportlevel=0,CPU=1,getResidual=False):
        image=image.clone()
        image.rearrangeDim(['x','y','z','t'])
        fill = 0
        if len(image.dim)>4:
            fill =np.zeros(image.data.shape[4:])
        resultImageshape=list(image.data.shape)
        stretch={}
        for dim in dimExpand:
            stretch[dim]=int(dimExpand[dim]*image.data.shape[image.dim.index(dim)])
            resultImageshape[image.dim.index(dim)]=stretch[dim]
        resultImage=image.clone()
        resultImage.stretch(stretch,stretchData=False)

        if type(xList)==type(None):
            xList=range(resultImageshape[0])
        else:
            resultImageshape[0]=len(xList)
        if type(yList)==type(None):
            yList=range(resultImageshape[1])
        else:
            resultImageshape[1]=len(yList)
        if type(zList)==type(None):
            zList=range(resultImageshape[2])
        else:
            resultImageshape[2]=len(zList)
        if type(tList)==type(None):
            tList=range(resultImageshape[3])
        else:
            resultImageshape[3]=len(tList)
        resultImage.data=np.zeros(resultImageshape).astype(image.data.dtype)
        if getResidual:
            residualImage=resultImage.clone()
            residualImage.data=np.zeros((*resultImageshape[:4],3))
            residualImage.dim=residualImage.dim[:4]+['r']
            residualImage.dimlen={'x':residualImage.dimlen['x'],'y':residualImage.dimlen['y'],'z':residualImage.dimlen['z'],'t':residualImage.dimlen['t'],'r':1}
        logger.info('Creating Image with shape '+str(resultImage.data.shape))
        for xn in range(len(xList)):
            if reportlevel>=0:
                logger.info('    {0:.3f}% completed...'.format(float(xn)/len(xList)*100.))
            for yn in range(len(yList)):
                if reportlevel>=1:
                    logger.info('        {0:.3f}% completed...'.format(float(yn)/len(yList)*100.))
                for zn in range(len(zList)):
                    if reportlevel>=2:
                        logger.info('            {0:.3f}% completed...'.format(float(zn)/len(zList)*100.))
                    xyzRef=np.array([xList[xn]*resultImage.dimlen['x'],yList[yn]*resultImage.dimlen['y'],zList[zn]*resultImage.dimlen['z']])
                    coord=[]
                    for t in tList:
                        coord.append([*xyzRef,t*image.dimlen['t']])
                    coords=self.bsFourier.getCoordFromRef(coord)
                    coords=np.hstack((np.array(coords),np.array(coord)[:,3].reshape((-1,1))))
                    val=image.getData(coords,fill=fill,getResidual=getResidual)
                    if getResidual:
                        val=np.array(val)
                        resultImage.data[xn,yn,zn]=np.array(list(val[:,0]))
                        residualImage.data[xn,yn,zn]=np.array(list(val[:,1]))[...,:3]
                    else:
                        resultImage.data[xn,yn,zn]=np.array(val)
            if type(tempSaveFile)!=type(None):
                resultImage.save(tempSaveFile)
        if getResidual:
            return (resultImage,residualImage)
        else:
            return resultImage
    def forwardImageTransform(self,refImage,time,sampleRate=1.,drawspread=[-2,-1,0,1,2],xList=None,yList=None,zList=None):
        refImage.rearrangeDim(['x','y','z'])
        newImg=refImage.clone()
        distance=np.ones(refImage.data.shape)*float('inf')
        if type(xList)!=type(None) or type(yList)!=type(None) or type(zList)!=type(None):
            maxmotion=np.abs(self.bsFourier.getBspline(time)).max(axis=0).max(axis=0).max(axis=0)/np.array([refImage.dimlen['x'],refImage.dimlen['y'],refImage.dimlen['z']])
            logger.info(str(maxmotion))
        if type(xList)==type(None):
            xList=np.arange(0,refImage.data.shape[0]-1.+1./sampleRate,1./sampleRate)
        else:
            xList=np.arange(max(0,min(xList)-maxmotion[0]),min(refImage.data.shape[0]-1.+1./sampleRate,max(xList)+maxmotion[0]),1./sampleRate)
        if type(yList)==type(None):
            yList=np.arange(0,refImage.data.shape[1]-1.+1./sampleRate,1./sampleRate)
        else:
            yList=np.arange(max(0,min(yList)-maxmotion[1]),min(refImage.data.shape[1]-1.+1./sampleRate,max(yList)+maxmotion[1]),1./sampleRate)
        if type(zList)==type(None):
            zList=np.arange(0,refImage.data.shape[2]-1.+1./sampleRate,1./sampleRate)
        else:
            zList=np.arange(max(0,min(zList)-maxmotion[2]),min(refImage.data.shape[2]-1.+1./sampleRate,max(zList)+maxmotion[2]),1./sampleRate)
        logger.info('Calculating xpixels '+str(min(xList))+'to '+str(max(xList))+', ypixels '+str(min(yList))+'to '+str(max(yList))+', zpixels '+str(min(zList))+'to '+str(max(zList)))
        for x in xList:
            logger.info('    {0:.3f}% completed...'.format(float(list(xList).index(x))/len(xList)*100.))
            for y in yList:
                #logger.info('        {0:.3f}% completed...'.format(float(list(yList).index(y))/len(yList)*100.))
                for z in zList:
                    #logger.info('            {0:.3f}% completed...'.format(float(list(zList).index(z))/len(zList)*100.))
                    xyzRef=np.array([x*refImage.dimlen['x'],y*refImage.dimlen['y'],z*refImage.dimlen['z']])
                    xyzTime=self.bsFourier.getCoordFromRef(np.array([*xyzRef,time]))/np.array([refImage.dimlen['x'],refImage.dimlen['y'],refImage.dimlen['z']])
                    intensity=refImage.getData(xyzRef)
                    baseind=np.around(xyzTime).astype(int)
                    for xalter in drawspread:
                        for yalter in drawspread:
                            for zalter in drawspread:
                                ind=baseind+np.array([xalter,yalter,zalter])
                                if np.all(ind>=0) and np.all(ind<=(np.array(distance.shape)-1)):
                                    dis=np.sum((xyzTime-ind)**2.)
                                    if dis<distance[tuple(ind)]:
                                        newImg.data[tuple(ind)]=intensity
                                        distance[tuple(ind)]=dis
        '''
        xyzRef=np.mgrid[0:refImage.data.shape[0], 0:refImage.data.shape[1],0:refImage.data.shape[2]].reshape(3,*refImage.data.shape[:3]).transpose(1,2,3,0)
        xyzRef=xyzRef.reshape((-1,3))*np.array([refImage.dimlen['x'],refImage.dimlen['y'],refImage.dimlen['z']])
        xyzTime=self.bsFourier.getCoordFromRef(np.hstack((xyzRef,np.ones((len(xyzRef),1))*time)))/np.array([refImage.dimlen['x'],refImage.dimlen['y'],refImage.dimlen['z']])
        intensity=refImage.data.reshape(-1)
        baseind=np.around(xyzTime).astype(int)
        for n in range(len(intensity)):
            for xalter in drawspread:
                for yalter in drawspread:
                    for zalter in drawspread:
                        ind=baseind[n]+np.array([xalter,yalter,zalter])
                        if np.all(ind>=0) and np.all(ind<=(np.array(distance.shape)-1)):
                            dis=np.sum((xyzTime[n]-ind)**2.)
                            if dis<distance[tuple(ind)]:
                                newImg.data[tuple(ind)]=intensity[n]
                                distance[tuple(ind)]=dis
        '''
        xx,yy,zz=np.nonzero(distance==float('inf'))
        logger.info(str(len(xx))+' unaltered pixels')
        logger.info(str(np.array([xx,yy,zz]).transpose()))
        '''
        while len(xx)>0:
            for x in xx:
                for y in yy:
                    for z in zz:
        '''
        return newImg
                    
        
    def estimateInitialwithRefTime(self,OrderedBsplinesList,tRef=None,OrderedBsplinesList2=None,spacingDivision=2.,gap=0,forwardbackward=False,N=20):
        ''' 
        Estimates bsplineFourier with forward marching
        Parameters:
            OrderedBsplinesList: List(int)
                List of index in self.bsplines to use as tref to tn marching
            OrderedBsplinesList2: List(int)
                List of index in self.bsplines to use as tn-1 to tn marching starting from tref to tref+1
            spacingDivision:float
                sampling points density between bsplineFourier grid
            gap:int
                number of sampling points near the boundary removed (1 means that all the sampling points at the boundary are removed)
            backwardforward: boolean
                if True, OrderedBsplinesList is List of index in self.bsplines to use as tn to tn+1 marching while OrderedBsplinesList2 is tn to tn-1
            timeMapList and N: if timeMapList not None, use NFFT initilization; N needs to be even
        '''
        if type(OrderedBsplinesList)==int:
            OrderedBsplinesList=range(OrderedBsplinesList)
        if self.bsplines[OrderedBsplinesList[0]].timeMap[0] is None:
            timeMapList=None
            logger.warning('Unable to determine corresponding timemap of bsplines '+str(n)+', '+str(self.bsplines[OrderedBsplinesList[0]].timeMap)+'.')
        else:
            timeMapList=[self.bsplines[OrderedBsplinesList[0]].timeMap[0]]
            for n in range(len(OrderedBsplinesList)):
                if self.bsplines[OrderedBsplinesList[n]].timeMap[1] is None:
                    timeMapList=None
                    logger.warning('Unable to determine corresponding timemap of bsplines '+str(n)+', '+str(self.bsplines[OrderedBsplinesList[n]].timeMap)+'.')
                    break
                else:
                    timeMapList.append(self.bsplines[OrderedBsplinesList[n]].timeMap[1])
            else:
                timeMapList=np.array(timeMapList)
                for n in range(len(timeMapList)):
                    while timeMapList[n]>=self.bsFourier.spacing[-1]:
                        timeMapList[n]-=self.bsFourier.spacing[-1]
                    while timeMapList[n]<0:
                        timeMapList[n]+=self.bsFourier.spacing[-1]
        if timeMapList is not None:
            locate_coordsThruTime = timeMapList/self.bsFourier.spacing[-1] - 0.5
            weight=[]
            weight.append(1)
            for i in range(int(self.bsFourier.coef.shape[self.points.shape[-1]]/2)):
                weight.append((-1)**(i+1))
            for i in range(int(self.bsFourier.coef.shape[self.points.shape[-1]]/2)):
                weight.append((-1)**i)
        sampleCoord=self.bsFourier.samplePoints(spacingDivision=spacingDivision,gap=gap)
        sampleCoef=[]
        refCoord=[]
        count=0
        logger.info('Estimating coefficients with '+str(len(sampleCoord))+'sample points')
        for coord in sampleCoord:
            count+=1
            if type(OrderedBsplinesList2)!=type(None) and forwardbackward:
                coordsThruTime=np.array(estCoordsThruTime(coord,self.bsplines,OrderedBsplinesList,mode='Eulerian'))
                coordsThruTime2=np.array(estCoordsThruTime(coord,self.bsplines,OrderedBsplinesList2,mode='Eulerian'))
                Fratio=1./(1.+np.arange(len(coordsThruTime))/np.arange(len(coordsThruTime),0,-1))
                coordsThruTime2=np.roll(coordsThruTime2[::-1],1,axis=0)
                coordsThruTime=Fratio.reshape((-1,1))*coordsThruTime+(1-Fratio.reshape((-1,1)))*coordsThruTime2
            else:
                coordsThruTime=estCoordsThruTime(coord,self.bsplines,OrderedBsplinesList,OrderedBsplinesList2=OrderedBsplinesList2,mode='Lagrangian-Eulerian')
            coordsThruTime=np.array(coordsThruTime)
            #deltat=self.bsplines[0].timeMap[1]-self.bsplines[0].timeMap[0]
            #freq=np.fft.rfftfreq(len(coordsThruTime[:,0]))*2.*np.pi/deltat
            sampleCoeftemp=[]
            for axis in range(self.points.shape[-1]):
                if timeMapList is not None:
                    sp = nfft.nfft_adjoint(locate_coordsThruTime, coordsThruTime[:,axis], N)
                    sampleCoeftemp.append(np.array([sp.real[int(N/2)]/len(coordsThruTime),*(sp.real[int(N/2+1):int(self.bsFourier.coef.shape[self.points.shape[-1]]/2+N/2+1)]/len(coordsThruTime)*2.),*(-sp.imag[int(N/2+1):int(self.bsFourier.coef.shape[self.points.shape[-1]]/2+N/2+1)]/len(coordsThruTime)*2.)])*np.array(weight))                  
                else:
                    sp = np.fft.rfft(coordsThruTime[:,axis])
                    sampleCoeftemp.append(np.array([sp.real[0]/len(coordsThruTime),*(sp.real[1:int(self.bsFourier.coef.shape[self.points.shape[-1]]/2+1)]/len(coordsThruTime)*2.),*(-sp.imag[1:int(self.bsFourier.coef.shape[self.points.shape[-1]]/2+1)]/len(coordsThruTime)*2.)]))
            sampleCoeftemp=np.array(sampleCoeftemp)
            sampleCoef.append(sampleCoeftemp.transpose().copy())
            sampleCoef[-1][0,:]=0.
            if type(tRef)!=type(None):
                sampleCoef[-1][0]=-getCoordfromCoef(np.array([*coord[:self.points.shape[-1]],tRef-self.bsFourier.origin[self.points.shape[-1]]]),sampleCoef[-1],self.bsFourier.spacing)
            
            refCoord.append(sampleCoeftemp[:,0].copy())
        logger.info('Calculated '+str(len(refCoord))+'sample points')
        
        self.bsFourier.regrid(refCoord,sampleCoef,tRef=tRef)
        return (refCoord,sampleCoef)
    def pointTrace(self,stlFile,savePath,timeList=None,delimiter=' '):
        os.makedirs(savePath, exist_ok=True)
        if type(timeList)==type(None):
            timeList=10
        if isinstance(timeList,(int)):
            timeList=np.arange(0,self.bsFourier.spacing[self.bsFourier.coef.shape[-1]],self.bsFourier.spacing[self.bsFourier.coef.shape[-1]]/timeList)
        if stlFile[-3:]=='stl':
            ref_mesh=trimesh.load(stlFile)
            oriPos=np.array(ref_mesh.vertices)[:,:self.bsFourier.coef.shape[-1]]
        else:
            oriPos=np.loadtxt(stlFile,delimiter=delimiter)
        
        for time in timeList:
            coords=np.concatenate((oriPos,np.ones((len(oriPos),1))*time),axis=-1)
            newpts=self.bsFourier.getCoordFromRef(coords)
            if stlFile[-3:]=='stl':
                ref_mesh.vertices[:,:self.bsFourier.coef.shape[-1]]=np.array(newpts)
                try:
                    trimesh.io.export.export_mesh(ref_mesh,savePath+'/t{0:.2e}'.format(time)+'.stl')
                except:
                    ref_mesh.export(savePath+'/t{0:.2e}'.format(time)+'.stl')
            else:
                np.savetxt(savePath+'/t{0:.2e}'.format(time)+'.txt',np.array(newpts))
    def estimatedStrainDetect(self,threshold=-1.,outputCoord=True):
        testShape=(np.array(self.bsFourier.coef.shape[:3])-1).astype(int)
        dXYZ=[]
        dXYZ.append(self.bsFourier.coef[1:,:,:]-self.bsFourier.coef[:-1,:,:])
        dXYZ[-1]=(dXYZ[-1][:,1:,1:]+dXYZ[-1][:,:-1,1:]+dXYZ[-1][:,1:,:-1]+dXYZ[-1][:,:-1,:-1])/4.
        dXYZ.append(self.bsFourier.coef[:,1:,:]-self.bsFourier.coef[:,:-1,:])
        dXYZ[-1]=(dXYZ[-1][1:,:,1:]+dXYZ[-1][:-1,:,1:]+dXYZ[-1][1:,:,:-1]+dXYZ[-1][:-1,:,:-1])/4.
        dXYZ.append(self.bsFourier.coef[:,:,1:]-self.bsFourier.coef[:,:,:-1])
        dXYZ[-1]=(dXYZ[-1][1:,1:,:]+dXYZ[-1][:-1,1:,:]+dXYZ[-1][1:,:-1,:]+dXYZ[-1][:-1,:-1,:])/4.
        
        for n in range(3):
            newcoef=np.zeros((*dXYZ[n].shape[:3],int(self.bsFourier.coef.shape[3]/2),3))
            for m in range(int(self.bsFourier.coef.shape[3]/2)):
                newcoef[:,:,:,m,:]=np.sqrt(dXYZ[n][:,:,:,m+1,:]**2.+dXYZ[n][:,:,:,int(self.bsFourier.coef.shape[3]/2)+m+1,:]**2.)
            dXYZ[n]=newcoef.copy()
        axial=[]
        for n in range(3):
            axial.append(dXYZ[n][:,:,:,:,n]/self.bsFourier.spacing[n])
        shear=[]
        for n in range(3):
            axis=[0,1,2]
            axis.pop(n)
            shear.append(-dXYZ[axis[0]][:,:,:,:,axis[1]]*dXYZ[axis[1]][:,:,:,:,axis[0]]/self.bsFourier.spacing[axis[0]]/self.bsFourier.spacing[axis[1]])
        coords=np.zeros((0,3))
        for n in range(3):
            tempcoords=list(np.where(axial[n]<threshold))
            tempcoords=np.vstack((tempcoords[0],tempcoords[1],tempcoords[2])).transpose()
            if len(tempcoords)>0:
                tempcoords[n]=tempcoords[n]+0.5
                coords=np.vstack((coords, tempcoords))
            tempcoords=list(np.where(shear[n]<threshold))
            tempcoords=np.vstack((tempcoords[0],tempcoords[1],tempcoords[2])).transpose()
            if len(tempcoords)>0:
                tempcoords=tempcoords+0.5
                coords=np.vstack((coords, tempcoords))
        if outputCoord:
            for n in range(3):
                coords[:,n]=coords[:,n]*self.bsFourier.spacing[n]+self.bsFourier.origin[n]
        return coords

