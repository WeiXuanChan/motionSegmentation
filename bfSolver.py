'''
File: bfSolver.py
Description: solving BSplinrFourier Coeficients with input bspline vector maps
             Contains externally usable class
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: w.x.chan@gmail.com         25OCT2018           - Created

Requirements:
    BsplineFourier
    numpy
    scipy

Known Bug:
    None
All rights reserved.
'''
print('bfSolver version 1.0.0')

import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import matrix_rank

try:
    from joblib import Parallel, delayed
except ImportError:
    pass
try:
  import pickle
except ImportError:
  pass
sys.path.insert(0, os.path.dirname('/home/yaplab/Programming/python3'))
import medImgProc
import BsplineFourier
def estCoordsThruTime(coord,bsplineList,OrderedBsplinesList,OrderedBsplinesList2=None):
    coordsThruTime=[coord.copy()]
    for n in range(len(OrderedBsplinesList)):
        vector=bsplineList[OrderedBsplinesList[n]].getVector(coordsThruTime[0])
        newcoord=vector+coordsThruTime[0]
        if type(OrderedBsplinesList2)!=type(None) and n>0 and n<(len(OrderedBsplinesList)-1):
            ratio=float(n)*float(len(OrderedBsplinesList)-1-n)/((len(OrderedBsplinesList)-1)/2.)**2.
            vector2=bsplineList[OrderedBsplinesList2[n]].getVector(coordsThruTime[-1])
            newcoord=newcoord*(1.-ratio)+(ratio)*(coordsThruTime[-1]+vector2)
        coordsThruTime.append(newcoord.copy())
    return coordsThruTime
def getCoordfromCoef(coord,coef,spacing):
    coeftemp=np.zeros(coef.shape[1:])
    for m in range(int(coef.shape[0]/2)):#sub in t
        coeftemp=coeftemp+coef[m+1]*np.cos((m+1.)*2.*np.pi/spacing[3]*coord[3])+coef[int(coef.shape[0]/2)+m+1]*np.sin((m+1.)*2.*np.pi/spacing[3]*coord[3])
    resultDeltaCoord=coeftemp.copy()
    return resultDeltaCoord

def load(file):
    with open(file, 'rb') as input:
        outObj = pickle.load(input)
    return outObj

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
                saveMatrix.append([self.points[n][0],self.points[n][1],self.points[n][2],*self.pointsCoef[n].reshape(-1, order='F')])
            else:
                saveMatrix.append([self.points[n][0],self.points[n][1],self.points[n][2],*np.zeros(self.pointsCoef[0].size)])
        saveMatrix=np.array(saveMatrix)
        np.savetxt(filepath,saveMatrix,delimiter=delimiter,header=str(len(self.pointsCoef))+' points calculated-- Coordinates, Fourier u, Fourier v, Fourier w')
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
        for n in range(len(loadMatrix)):
            self.points.append(loadMatrix[n,:3].copy())
            if n<coeflen:
                self.pointsCoef.append(loadMatrix[n,3:].reshape((-1,3),order='F'))
        
    def addBsplineFile(self,BsplineFileList=None,timeMapList=None,weightList=None,fileScale=1.):
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
            if type(weightList)!=list:1
                self.weights.append(1.)
            elif len(weightList)<=n:
                self.weights.append(1.)
            else:
                self.weights.append(weightList[n])
            self.bsplines.append(BsplineFourier.Bspline(coefFile=BsplineFileList[n],shape=None,timeMap=timeMapList[n],spacing=None,fileScale=fileScale,delimiter=' ',origin=None))
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
    
    def initialize(self,shape=None,spacing=None,origin=None,period=1.,fourierTerms=3,spacingDivision=2.,gap=1):
        ''' 
        Initialize the solver
        Parameters:
            shape=[x,y,z,f,uvw]: list(float)
                shape of resultant bsplineFourier
            spacing=[x,y,z,period]:list(float)1
                shape of resultant bsplineFourier
            origin=[x,y,z,t]:list(float)
                shape of resultant bsplineFourier
            period:float1
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
                print('Adjusted to:')
                print('    shape=',self.bsFourier.coef.shape)
                print('    origin=',self.bsFourier.origin)
                print('    spacing=',self.bsFourier.spacing)
        self.points=np.array(self.bsFourier.samplePoints(spacingDivision=spacingDivision,gap=gap))
        print('Initialized with',len(self.points),'points.')

        
    def solve(self,maxError=0.00001,maxIteration=1000,convergence=0.8,method='pointbypoint',reportevery=1000,tempSave=None,resume=False,rmsBasedWeighted=None,linearConstrainPoints=[],linearConstrainWeight=None):
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
            sampleCoefList,rmsList=self.solve_pointbypoint(maxError=maxError,maxIteration=maxIteration,convergence=convergence,reportevery=reportevery,tempSave=tempSave,resume=resume)
            if type(rmsBasedWeighted)==type(None):
                rmsweight=None
            else:
                rmsweight=rmsBasedWeighted(rmsList)
        self.bsFourier.regrid(self.points,sampleCoefList,weight=rmsweight,linearConstrainPoints=linearConstrainPoints,linearConstrainWeight=linearConstrainWeight)
        print('BsplineFourier updated')
  
    def solve_pointbypoint(self,maxError=0.00001,maxIteration=1000,convergence=0.8,reportevery=1000,tempSave=None,resume=False,movAvgError=False,lmLambda_init=0.001,lmLambda_incrRatio=5.,lmLambda_max=float('inf'),lmLambda_min=0.):
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
            wdiag=np.concatenate((wdiag,np.ones(3)*weight),axis=0)
        rmsList=[]
        if not(resume):
            self.pointsCoef=[]
        elif type(tempSave)!=type(None):
            self.loadSamplingResults(tempSave)
        for m in range(len(self.pointsCoef),len(self.points)):
            coef=self.bsFourier.getRefCoef(self.points[m])
            coef_start=coef.copy()
            error=float('inf')
            count=0.
            fx=[]
            pointX=[]
            for n in range(len(self.bsplines)):
                Y=getCoordfromCoef(np.array([*self.points[m][:3],self.bsplines[n].timeMap[1]-self.bsFourier.origin[3]]),coef,self.bsFourier.spacing)+self.points[m][:3]
                X=getCoordfromCoef(np.array([*self.points[m][:3],self.bsplines[n].timeMap[0]-self.bsFourier.origin[3]]),coef,self.bsFourier.spacing)+self.points[m][:3]
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
                    dVdX=self.bsplines[n].getdX(pointX[n])
                    for axis in range(3):1
                        Jmattemp=[]
                        for xyz in range(3):
                            if axis==xyz:
                                Jmattemp.append(fourierdY-fourierdX*(1.+dVdX[xyz,axis]))
                            else:
                                Jmattemp.append(-fourierdX*dVdX[xyz,axis])

                        Jmat[Jcount]=np.array([*Jmattemp[0],*Jmattemp[1],*Jmattemp[2]])
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
                            error=max(error,np.abs(dCoef[n,nn])/self.bsFourier.spacing[:3].min())
                        else:
                            error=max(error,np.abs(dCoef[n,nn]).max()/np.abs(coef[n+1,nn]))     
                #renew
                
                ratio=reductionRatio
                if convergence:
                    if abs(dCoef).max()>self.bsFourier.spacing[:3].min()*convergence:
                        ratio=min(ratio,self.bsFourier.spacing[:3].min()*convergence/abs(dCoef).max())
                
                
                coef_backup=coef.copy()
                fx_backup=fx.copy()
                pointX_backup=pointX.copy()

                coef[1:,:]+=ratio*dCoef
                fx=[]
                pointX=[]
                for n in range(len(self.bsplines)):
                    Y=getCoordfromCoef(np.array([*self.points[m][:3],self.bsplines[n].timeMap[1]-self.bsFourier.origin[3]]),coef,self.bsFourier.spacing)+self.points[m][:3]
                    X=getCoordfromCoef(np.array([*self.points[m][:3],self.bsplines[n].timeMap[0]-self.bsFourier.origin[3]]),coef,self.bsFourier.spacing)+self.points[m][:3]
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
                    #print('deltarms=',deltarms,'from rms=',rms)
                    #print('reduce ratio to',reduceRatio)
                else:
                    if ratio>0.9 and lmLambda!=lmLambda_min:
                        if (lmLambda/np.sqrt(lmLambda_incrRatio))>lmLambda_min:
                            lmLambda=lmLambda/np.sqrt(lmLambda_incrRatio)
                        else:
                            lmLambda=lmLambda_min
                    elif reductionRatio<0.9:
                        reductionRatio*=1.1
                    rms=np.sqrt(np.mean(wdiag*fx**2.))
                    if movAvgError:1
                        error=np.abs(movAvgError-rms)/movAvgError
                        movAvgError=(movAvgError+rms)/2.
                    count+=1
            rmsList.append(rms)
            self.pointsCoef.append(coef.copy())
            if count==maxIteration:
                print('Maximum iterations reached for point',m,self.points[m])
            if m%reportevery==0:
                print('Solved for point',m+1,'/',len(self.points),self.points[m],',rms start=',rmsStart,'rms end=',rms,',max rms=',max(rmsList))
                if type(tempSave)!=type(None):
                    self.writeSamplingResults(tempSave)
        rmsList=np.array(rmsList)
        return (self.pointsCoef,rmsList)
    def estimateInitialwithRefTime(self,OrderedBsplinesList,refTimeStep=0,OrderedBsplinesList2=None,spacingDivision=2.,gap=0):
        ''' 
        Estimates bsplineFourier with forward marching
        Parameters:
            OrderedBsplinesList: List(int)
                List of index in self.bsplines to use as tref to tn marching
            refTimeStep:int
                identify the reference time step tref in OrderedBsplinesList
            OrderedBsplinesList2: List(int)
                List of index in self.bsplines to use as tn-1 to tn marching
            spacingDivision:float
                sampling points density between bsplineFourier grid
            gap:int
                number of sampling points near the boundary removed (1 means that all the sampling points at the boundary are removed)
        '''
        if type(OrderedBsplinesList)==int:
            OrderedBsplinesList=range(OrderedBsplinesList)
        sampleCoord=self.bsFourier.samplePoints(spacingDivision=spacingDivision,gap=gap)
        sampleCoef=[]
        refCoord=[]
        count=0
        print('Estimating coefficients with',len(sampleCoord),'sample points')
        for coord in sampleCoord:
            count+=1
            coordsThruTime=estCoordsThruTime(coord,self.bsplines,OrderedBsplinesList,OrderedBsplinesList2=OrderedBsplinesList2)
            if refTimeStep!=0:
                tempcoordsThruTime=coordsThruTime.copy()
                tempcoordsThruTime[:refTimeStep]=coordsThruTime[-refTimeStep:]
                tempcoordsThruTime[refTimeStep:]=coordsThruTime[:-refTimeStep]
                coordsThruTime=tempcoordsThruTime.copy()
            coordsThruTime=np.array(coordsThruTime)
            #deltat=self.bsplines[0].timeMap[1]-self.bsplines[0].timeMap[0]
            #freq=np.fft.rfftfreq(len(coordsThruTime[:,0]))*2.*np.pi/deltat
            sampleCoeftemp=[]
            for axis in range(3):
                sp = np.fft.rfft(coordsThruTime[:,axis])
                sampleCoeftemp.append(np.array([sp.real[0]/len(coordsThruTime),*(sp.real[1:int(self.bsFourier.coef.shape[3]/2+1)]/len(coordsThruTime)*2.),*(-sp.imag[1:int(self.bsFourier.coef.shape[3]/2+1)]/len(coordsThruTime)*2.)]))
            sampleCoeftemp=np.array(sampleCoeftemp)
            sampleCoef.append(sampleCoeftemp.transpose().copy())
            sampleCoef[-1][0,:]=0. #set a0=0
            refCoord.append(sampleCoeftemp[:,0].copy())
        print('Calculated',len(refCoord),'sample points')
        
        self.bsFourier.regrid(refCoord,sampleCoef)
        return (refCoord,sampleCoef)
