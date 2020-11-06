'''
File: BsplineFourier.py
Description: stores Bspline Coefficients
             stores BsplineFourier Coefficients
             Contains externally usable class
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: w.x.chan@gmail.com         25OCT2018           - Created
  Author: w.x.chan@gmail.com         19DEC2018           - v1.2.0
                                                             -include addition of the a0 fourier term
  Author: w.x.chan@gmail.com         01Feb2018           - v1.3.0
                                                             -add EulerTransform
  Author: w.x.chan@gmail.com         06Aug2019           - v2.0.0
                                                             -enable 2D
  Author: w.x.chan@gmail.com         16Sep2019           - v2.2.0
                                                             -added function fcoefImage
  Author: jorry.zhengyu@gmail.com    26Sep2019          - v2.2.5
                                                             -modify function motionImage - spacing==None  
Author: w.x.chan@gmail.com         16Sep2019           - v2.2.6
                                                             -modify function motionImage - correct spacing value when spacing==None
Author: w.x.chan@gmail.com         07Oct2019           - v2.2.7
                                                             -corrected self.origin[-1] from self.origin[3] to cater to 2D
Author: w.x.chan@gmail.com         07Oct2019           - v2.3.3
                                                             -added evaluateFunc to motionImage
Author: w.x.chan@gmail.com         18Nov2019           - v2.4.2
                                                             -added __call__(self,t) to evaluate FourierSeries
Author: w.x.chan@gmail.com         18Nov2019           - v2.4.4
                                                             -changed to logging
Author: w.x.chan@gmail.com         12Dec2019           - v2.4.6
                                                             -debug writeBspline
Author: w.x.chan@gmail.com         13Dec2019           - v2.4.7
                                                             -debug imageVectorAD
Author: w.x.chan@gmail.com         10FEB2020           - v2.5.4
                                                             -debug writeSITKfile
Author: w.x.chan@gmail.com         21FEB2020           - v2.6.2
                                                             -debug BspreadArray.getbspread , correct mgrid
Author: w.x.chan@gmail.com         21FEB2020           - v2.7.12
                                                             -add regrid to Bspline
Author: w.x.chan@gmail.com         07NOV2020           - v2.7.13
                                                             -debug undefine nFourierRange in Bspline.regrid
Requirements:
    autoD
    numpy
    re
    scipy
    pickle (optional)

Known Bug:
    None
All rights reserved.
'''
_version='2.7.13'

import logging
logger = logging.getLogger(__name__)
import os
import numpy as np
import autoD as ad
import re
import multiprocessing
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

#Optional dependancies
try:
    import pickle
except ImportError:
    pass

try:
    import medImgProc
except:
    pass


b_ad=ad.Scalar('b')
B=[(1.-b_ad)**3./6.,(3.*b_ad**3.-6.*b_ad**2.+4.)/6.,(-3.*b_ad**3.+3.*b_ad**2.+3.*b_ad+1)/6.,b_ad**3./6.]
rndError=0.001

def createGaussianMat(size,sigma=1., mu=0.):
    '''size indicate the length from the middle point'''
    lineSpace=[]
    for length in size:
        lineSpace.append(np.linspace(-1,1,length*2+1))
    x= np.meshgrid(*lineSpace)
    x=np.array([*x])
    d=np.sum(x*x,axis=0)
    d=np.sqrt(d)
    gauss = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return gauss
def load(file):
    with open(file, 'rb') as input:
        outObj = pickle.load(input)
    return outObj
class Bspline:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,coefFile=None,timeMap=[None,None],shape=None,spacing=None,fileScale=1.,delimiter=' ',origin=None):
        '''
        Initialize all data.
        Note:  coef[x,y,z,uvw]
        spacing=[x,y,z]
        origin=[x,y,z]
        timeMap=[t1,t2] vector stored maps t1 to t2
        '''
        self.coef=None
        self.origin=None
        self.spacing=None
        self.timeMap=[None,None]
        self.coordMat =None
        if type(coefFile)!=type(None):
            self.read(coefFile=coefFile,timeMap=timeMap,shape=shape,spacing=spacing,fileScale=fileScale,delimiter=delimiter,origin=origin)
        if type(self.coef)==np.ndarray:
            logger.info('shape= '+str(self.coef.shape))
        logger.info('spacing= '+str(self.spacing))
        logger.info('origin= '+str(self.origin))
        logger.info('timeMap= '+str(self.timeMap))
    def read(self,coefFile=None,timeMap=[None,None],shape=None,spacing=None,fileScale=1.,delimiter=' ',origin=None):
        if type(coefFile)!=type(None):
            if type(shape)!=type(None):
              try:
                  self.coef=np.loadtxt(coefFile,delimiter=delimiter).reshape(shape, order='F')
                  logger.info('Loading '+coefFile)
              except:
                  pass
            if type(self.coef)==type(None):
                try:
                    self.coef=coefFile.copy()
                except:
                    pass
            if type(self.coef)==type(None):
                logger.info('Loading '+str(coefFile))
                with open (coefFile, "r") as myfile:
                    data=myfile.readlines()
                for string in data:
                    if type(origin)==type(None):
                        result = re.search('\(GridOrigin (.*)\)', string)
                    elif type(shape)==type(None):
                        result = re.search('\(GridSize (.*)\)', string)
                    elif type(spacing)==type(None):
                        result = re.search('\(GridSpacing (.*)\)', string)
                    else:
                        result = re.search('\(TransformParameters (.*)\)', string)
                    if result:
                        if type(origin)==type(None):
                            origin=np.fromstring(result.group(1), sep=' ')
                        elif type(shape)==type(None):
                            shape=np.fromstring(result.group(1), sep=' ').astype('uint8')
                        elif type(spacing)==type(None):
                            spacing=np.fromstring(result.group(1), sep=' ')
                        else:
                            self.coef=np.fromstring(result.group(1), sep=' ')
                            if self.coef.size!=np.prod(shape):
                                shape=[*shape,int(np.around((self.coef.size/np.prod(shape))))]
                            self.coef=self.coef.reshape(shape, order='F')
                            break
                if type(self.coef)==type(None):
                    self.coef=np.loadtxt(coefFile,skiprows=3)
                    if self.coef.size!=np.prod(shape):
                        shape=[*shape,int(np.around((self.coef.size/np.prod(shape))))]
                    self.coef=self.coef.reshape(shape, order='F')
            if type(shape)==type(None):
                shape=self.coef.shape
            if type(origin)==type(None):
                self.origin=np.zeros(len(shape)-1)
            else:
                self.origin=np.array(origin)
            if type(spacing)==type(None):
                self.spacing=np.ones(len(shape)-1)
            else:
                self.spacing=spacing
            mgridslice=[]
            for n in range(self.coef.shape[-1]):
                mgridslice.append(slice(self.origin[n],(self.origin[n]+(shape[n]-0.1)*self.spacing[n]),self.spacing[n]))
            self.coordMat = np.mgrid[tuple(mgridslice)].reshape(self.coef.shape[-1],*shape[:self.coef.shape[-1]]).transpose(*tuple(range(1,self.coef.shape[-1]+1)),0)
            if type(timeMap[1])==type(None):
                self.timeMap=[0.,timeMap[0]]
            else:
                self.timeMap=timeMap
            if fileScale!=1.:
                self.scale(1./fileScale)
        self.numCoefXYZ=1
        for n in self.coef.shape[:self.coef.shape[-1]]:
            self.numCoefXYZ*=n
        self.numCoef=int(self.numCoefXYZ)
    def save(self,file):
        with open(file, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
    def writeCoef(self,filepath,delimiter=' '):
        ''' 
        Write coef in a single-line in Fortran format
        Parameters:
            filePath:file,str
                File or filename to save to
            delimiter:str, optional
                separation between values
        '''
        np.savetxt(filepath,self.coef.reshape(-1, order='F'),delimiter=delimiter,comments='',header='(GridOrigin '+' '.join(map(str, self.origin))+')\n(GridSize '+' '.join(map(str, self.coef.shape))+')\n(GridSpacing '+' '.join(map(str, self.spacing))+')')
    def writeSITKfile(self,filepath,imageSize=None,imageSpacing=None):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if type(imageSize)==type(None):
            imageSize=self.coef.shape[:-1]
        if type(imageSpacing)==type(None):
            imageSpacing=self.spacing
        direction=np.eye(self.coef.shape[-1]).reshape(-1)
        imageOrigin=np.zeros(self.coef.shape[-1])
        with open(filepath, 'w') as f: 
            f.write('(BSplineTransformSplineOrder 3.000000)\n')
            f.write('(CompressResultImage "false")\n')
            f.write('(DefaultPixelValue 0.000000)\n')
            f.write('(Direction '+' '.join('{0:.6f}'.format(x) for x in direction)+')\n')
            f.write('(FinalBSplineInterpolationOrder 3.000000)\n')
            f.write('(FixedImageDimension '+str(len(self.origin))+'.000000)\n')
            f.write('(FixedInternalImagePixelType "float")\n')
            f.write('(GridDirection '+' '.join('{0:.6f}'.format(x) for x in direction)+')\n')
            f.write('(GridIndex '+' '.join('{0:.6f}'.format(x) for x in imageOrigin)+')\n')
            f.write('(GridOrigin '+' '.join('{0:.6f}'.format(x) for x in self.origin)+')\n')
            f.write('(GridSize '+' '.join('{0:.6f}'.format(x) for x in self.coef.shape[:-1])+')\n')
            f.write('(GridSpacing '+' '.join('{0:.6f}'.format(x) for x in self.spacing)+')\n')
            f.write('(HowToCombineTransforms "Compose")\n')
            f.write('(Index '+' '.join('{0:.6f}'.format(x) for x in imageOrigin)+')\n')
            f.write('(InitialTransformParametersFileName "NoInitialTransform")\n')
            f.write('(MovingImageDimension '+str(len(self.origin))+'.000000)\n')
            f.write('(MovingInternalImagePixelType "float")\n')
            f.write('(NumberOfParameters '+str(self.coef.size)+'.000000)\n')
            f.write('(Origin '+' '.join('{0:.6f}'.format(x) for x in imageOrigin)+')\n')
            f.write('(ResampleInterpolator "FinalBSplineInterpolator")\n')
            f.write('(Resampler "DefaultResampler")\n')
            f.write('(ResultImageFormat "nii")\n')
            f.write('(ResultImagePixelType "float")\n')
            f.write('(Size '+' '.join('{0:.6f}'.format(x) for x in imageSize)+')\n')
            f.write('(Spacing '+' '.join('{0:.6f}'.format(x) for x in imageSpacing)+')\n')
            f.write('(Transform "BSplineTransform")\n')
            f.write('(TransformParameters '+' '.join('{0:.6f}'.format(x) for x in self.coef.reshape(-1, order='F'))+')\n')
            f.write('(UseCyclicTransform "false")\n')
            f.write('(UseDirectionCosines "true")')
    def U(self,tVal=None,tRef=None,variableIdentifier=''):
        return bsFourierAD('u',self,tVal=tVal,tRef=tRef,variableIdentifier=variableIdentifier)
    def V(self,tVal=None,tRef=None,variableIdentifier=''):
        return bsFourierAD('v',self,tVal=tVal,tRef=tRef,variableIdentifier=variableIdentifier)
    def W(self,tVal=None,tRef=None,variableIdentifier=''):
        return bsFourierAD('w',self,tVal=tVal,tRef=tRef,variableIdentifier=variableIdentifier)
    def UImage(self,imageSize,imageSpacing,accuracy=1,tVal=None,tRef=None,variableIdentifier=''):
        return imageVectorAD('u',self,imageSize,imageSpacing,accuracy=accuracy,tVal=tVal,tRef=tRef,variableIdentifier=variableIdentifier)
    def VImage(self,imageSize,imageSpacing,accuracy=1,tVal=None,tRef=None,variableIdentifier=''):
        return imageVectorAD('v',self,imageSize,imageSpacing,accuracy=accuracy,tVal=tVal,tRef=tRef,variableIdentifier=variableIdentifier)
    def WImage(self,imageSize,imageSpacing,accuracy=1,tVal=None,tRef=None,variableIdentifier=''):
        return imageVectorAD('w',self,imageSize,imageSpacing,accuracy=accuracy,tVal=tVal,tRef=tRef,variableIdentifier=variableIdentifier)
    '''
    Function to receive data
    '''
    def getCoefMat(self,coord):
        ''' 
        Get coef value in Bspline matrix form
        Parameters:
            coord=[x,y,z]:list,np.ndarray
                Coordinate
        Return:
            coef(x,y,zuvw):np.ndarray
                coefficients corresponding to the input coordinate(s)
            uvw=[u,v,w] :np.ndarray
                index of coef in the entire array
        '''
        if (len(self.coef.shape)-2)>len(coord):
            raise Exception('Error, check input coordinates.', coord,self.coef.shape)
        coordSlice=[]
        matSlice=[]
        returnMatsize=[]
        uvw=[]
        for n in range(self.coef.shape[-1]):
            uvw.append((coord[n]-self.origin[n])/self.spacing[n])
            ind=int(uvw[-1])
            if (uvw[-1]-ind)<rndError:
                uvw[-1]=float(ind)
                returnMatsize.append(3)
            elif (uvw[-1]-ind)>(1.-rndError):
                ind+=1
                uvw[-1]=float(ind)
                returnMatsize.append(3)
            else:
                returnMatsize.append(4)
            if ind<1:
                corrector=[-ind+1,0]
            elif ind>=(self.coef.shape[n]-2):
                if uvw[-1]==ind:
                    corrector=[0,self.coef.shape[n]-ind-2]
                else:
                    corrector=[0,self.coef.shape[n]-ind-3]
            else:
                corrector=[0,0]
            if uvw[-1]==ind:
                coordSlice.append(slice(ind-1+corrector[0],max(0,ind+2+corrector[1])))
                matSlice.append(slice(0+corrector[0],max(0,3+corrector[1])))
            else:
                coordSlice.append(slice(ind-1+corrector[0],max(0,ind+3+corrector[1])))
                matSlice.append(slice(0+corrector[0],max(0,4+corrector[1])))
        uvw=np.array(uvw)
        coef=np.zeros((*returnMatsize,*self.coef.shape[self.coef.shape[-1]:]))
        if np.any(np.array(coef[tuple(matSlice)].shape)!=np.array(self.coef[tuple(coordSlice)].shape)):
            raise Exception(uvw,matSlice,coordSlice)
        coef[tuple(matSlice)]=self.coef[tuple(coordSlice)].copy()

        
        if np.any(uvw<-np.array(self.coef.shape[:self.coef.shape[-1]])) or np.any(uvw>(np.array(self.coef.shape[:self.coef.shape[-1]])*2.)):
            logger.warning('WARNING! Coordinates '+str(coord)+'far from active region queried! Grid Coord= '+str(uvw))
        '''
        if np.any(uvw<1) or np.any(uvw>np.array(self.coef.shape[:3])-2):
            coef=self.getExtendedCoef(uvw).copy()
        else:
            coef=self.coeftemp_coef[coordSlice].copy()
        '''
        return (coef,uvw)
    
    def getExtendedCoef(self,uvw):
        ''' 
        Gives zeros padding to coef matrix
        Parameters:
            uvw=[u,v,w]:np.ndarray
                index of coef in the entire array
        Return:
            coef(x,y,zuvw):np.ndarray or list[np.ndarray]
                coefficients or list of coefficients corresponding to the input index
        '''
        
        padNo=max(2-int(uvw.min()),int((uvw-np.array(self.coef.shape[:self.coef.shape[-1]])).max())+2)
        if padNo<0:
            padNo=0
        newuvw=uvw+padNo
        tempcoef=np.zeros((*(np.array(self.coef.shape[:self.coef.shape[-1]])+padNo*2),*self.coef.shape[self.coef.shape[-1]:]))
        tempcoef[padNo:self.coef.shape[0]+padNo,padNo:self.coef.shape[1]+padNo,padNo:self.coef.shape[2]+padNo]=self.coef
        coordSlice=[]
        for n in range(self.coef.shape[-1]):
            ind=int(newuvw[n])
            if (newuvw[n]-ind)<0.001:
                coordSlice.append(slice(ind-1,ind+2))
                newuvw[n]=float(ind)
            elif (newuvw[n]-ind)>0.999:
                ind+=1
                newuvw[n]=float(ind)
                coordSlice.append(slice(ind-1,ind+2))
            else:
                coordSlice.append(slice(ind-1,ind+3))
        
        return tempcoef[coordSlice].copy()
    def getVector(self,coordsList,vec=None,addCoord=False,dxyzt=None,CPU=1):
        ''' 
        Returns vector corresponding to the coordinates
        Parameters:
            coordsList=[[x,y,z],] or [x,y,z]:list,np.ndarray
                Coordinate or list of coordinates
        Return:
            resultVectors=[u,v,w] or [[u,v,w],]:np.ndarray or list[np.ndarray]
                vector or list of vectors corresponding to the input coordinate
        '''
        if type(dxyzt)==type(None):
            dxyzt=list(np.zeros(self.coef.shape[-1]+1,dtype=int))
        singleInput=False
        if not(type(coordsList[0]) in [np.ndarray,list]):
            coordsList=[coordsList]
            singleInput=True
        if CPU>1:
            fxstarmapInput=np.empty( (len(coordsList),4), dtype=object)
            fxstarmapInput[:,0]=list(coordsList)
            fxstarmapInput[:,1]=[vec]*len(coordsList)
            fxstarmapInput[:,2]=addCoord
            fxstarmapInput[:,3]=[dxyzt]*len(coordsList)
            pool = multiprocessing.Pool(CPU)
            resultVectors=pool.starmap(self.getVector,fxstarmapInput)
            pool.close()
            pool.join()
        else:
            if type(vec)==type(None):
                vec=slice(None)
            if dxyzt[self.coef.shape[-1]]%4==0:
                tempVal=[False,1.,1.] #swap, cos multiplier, sin multiplier
            elif dxyzt[self.coef.shape[-1]]%4==1:
                tempVal=[True,1.,-1.]
            elif dxyzt[self.coef.shape[-1]]%4==2:
                tempVal=[False,-1.,-1.]
            elif dxyzt[self.coef.shape[-1]]%4==3:
                tempVal=[True,-1.,1.]
            resultVectors=[]
            noneSlice=[]
            for n in range(self.coef.shape[-1]):
                noneSlice.append(slice(None))
            for n in range(len(coordsList)):
                coef,uvw=self.getCoefMat(coordsList[n])
                if len(coef.shape)==(self.coef.shape[-1]+2):
                    if len(coordsList[n])>self.coef.shape[-1]:
                        coeftemp=coef[tuple(noneSlice+[0,vec])].copy()
                        if dxyzt[-1]>0:
                            coeftemp=coeftemp*0.
                        for m in range(int(coef.shape[self.coef.shape[-1]]/2)):#sub in t
                            if tempVal[0]:
                                coeftemp=coeftemp+coef[tuple(noneSlice+[m+1,vec])]*((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]])**dxyzt[-1]*tempVal[2]*np.sin((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]]*(coordsList[n][self.coef.shape[-1]]-self.origin[self.coef.shape[-1]]))+coef[tuple(noneSlice+[int(coef.shape[self.coef.shape[-1]]/2)+m+1,vec])]*((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]])**dxyzt[self.coef.shape[-1]]*tempVal[1]*np.cos((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]]*(coordsList[n][self.coef.shape[-1]]-self.origin[self.coef.shape[-1]]))
                            else:
                                coeftemp=coeftemp+coef[tuple(noneSlice+[m+1,vec])]*((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]])**dxyzt[-1]*tempVal[1]*np.cos((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]]*(coordsList[n][self.coef.shape[-1]]-self.origin[self.coef.shape[-1]]))+coef[tuple(noneSlice+[int(coef.shape[self.coef.shape[-1]]/2)+m+1,vec])]*((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]])**dxyzt[self.coef.shape[-1]]*tempVal[2]*np.sin((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]]*(coordsList[n][self.coef.shape[-1]]-self.origin[self.coef.shape[-1]]))
                            
                    else:
                        coeftemp=coef[tuple(noneSlice+[slice(None),vec])].copy()
                        if dxyzt[self.coef.shape[-1]]>0:
                            coeftemp[tuple(noneSlice+[0])]=0.
                        for m in range(int(coef.shape[self.coef.shape[-1]]/2)):
                            if tempVal[0]:
                                coeftemp[tuple(noneSlice+[m+1])]=tempVal[1]*coef[tuple(noneSlice+[int(coef.shape[self.coef.shape[-1]]/2)+m+1,vec])].copy()*((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]])**dxyzt[self.coef.shape[-1]]
                                coeftemp[tuple(noneSlice+[int(coef.shape[self.coef.shape[-1]]/2)+m+1])]=tempVal[2]*coef[tuple(noneSlice+[m+1,vec])].copy()*((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]])**dxyzt[self.coef.shape[-1]]
                            else:
                                coeftemp[tuple(noneSlice+[m+1])]=tempVal[1]*coef[tuple(noneSlice+[m+1,vec])].copy()*((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]])**dxyzt[self.coef.shape[-1]]
                                coeftemp[tuple(noneSlice+[int(coef.shape[self.coef.shape[-1]]/2)+m+1])]=tempVal[2]*coef[tuple(noneSlice+[int(coef.shape[self.coef.shape[-1]]/2)+m+1,vec])].copy()*((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]])**dxyzt[self.coef.shape[-1]]
                elif len(coef.shape)==(self.coef.shape[-1]+1):
                    if dxyzt[self.coef.shape[-1]]==0:
                        coeftemp=coef[tuple(noneSlice+[vec])].copy()
                    else:
                        resultVectors.append(coef[tuple(list(np.zeros(self.coef.shape[-1]))+[vec])]*0)
                        continue
                coef=coeftemp.copy()
                for m in range(self.coef.shape[-1]):
                    coeftemp=np.zeros(coef.shape[1:])
                    for k in range(coef.shape[0]):
                        coeftemp=coeftemp+coef[k]*B[k]({'b':uvw[m]%1.},{'b':dxyzt[m]})/(self.spacing[m]**dxyzt[m])
                    coef=coeftemp.copy()
                if addCoord:
                    if dxyzt[-1]==0:
                        if sum(dxyzt[:self.coef.shape[-1]])==0:
                            add=np.array(coordsList[n][:self.coef.shape[-1]])
                        elif sum(dxyzt[:self.coef.shape[-1]])==1:
                            add=np.array(dxyzt[:self.coef.shape[-1]])
                        else:
                            add=np.zeros(self.coef.shape[-1])
                    else:
                        add=np.zeros(self.coef.shape[-1])
                    if len(self.coef.shape)==(self.coef.shape[-1]+2) and len(coordsList[n])==self.coef.shape[-1]:
                        coef[0]=coef[0]+add[vec]
                    else:
                        coef=coef+add[vec]
                resultVectors.append(coef.copy())
            resultVectors=np.array(resultVectors)
        if singleInput:
            resultVectors=resultVectors[0]
        return resultVectors
      
    def samplePoints(self,spacingDivision=2.,gap=0):
        step=np.array(self.spacing[:self.coef.shape[-1]])/spacingDivision
        start=self.coordMat[tuple(np.zeros(self.coef.shape[-1],dtype=int))]+step*gap
        end=self.coordMat[tuple(-np.ones(self.coef.shape[-1],dtype=int))]-step*gap+step/2.
        sampleCoord=[]
        for k in np.arange(start[0],end[0],step[0]):
            for l in np.arange(start[1],end[1],step[1]):
                if self.coef.shape[-1]>2:
                    for m in np.arange(start[2],end[2],step[2]):
                        sampleCoord.append(np.array([k,l,m]))
                else:
                    sampleCoord.append(np.array([k,l]))
        return sampleCoord
    def getdX(self,coordsList):
        ''' 
        Returns vector corresponding to the coordinates
        Parameters:
            coordsList=[[x,y,z],] or [x,y,z] (for BsplineFourier: [[x,y,z,t],] or [x,y,z,t]):list,np.ndarray
                Coordinate or list of coordinates
        Return:
            resultdX=[d[u,v,w]/dx,d[u,v,w]/dy,d[u,v,w]/dz] or[[d[u,v,w]/dx,d[u,v,w]/dy,d[u,v,w]/dz],]:np.ndarray(2d) or list[np.ndarray(2d)]
                gradient of vector or list of gradient of vectors corresponding to the input coordinate
        '''
        singleInput=False
        if not(type(coordsList[0]) in [np.ndarray,list]):
            coordsList=[coordsList]
            singleInput=True
        resultdX=[]
        noneSlice=[]
        for n in range(self.coef.shape[-1]):
            noneSlice.append(slice(None))
            
        for n in range(len(coordsList)):
            coef,uvw=self.getCoefMat(coordsList[n])
            if len(coordsList[n])>self.coef.shape[-1] and len(coef.shape)>(self.coef.shape[-1]+1):
                coeftemp=coef[tuple(noneSlice+[0,slice(None)])].copy()
                for m in range(int(coef.shape[self.coef.shape[-1]]/2)):#sub in t
                    coeftemp=coeftemp+coef[tuple(noneSlice+[m+1,slice(None)])]*np.cos((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]]*(coordsList[n][self.coef.shape[-1]]-self.origin[self.coef.shape[-1]]))+coef[tuple(noneSlice+[int(coef.shape[self.coef.shape[-1]]/2)+m+1,slice(None)])]*np.sin((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]]*(coordsList[n][self.coef.shape[-1]]-self.origin[self.coef.shape[-1]]))
            else:
                coeftemp=coef.copy()
            storecoef=coeftemp.copy()
            dXcoef=[]
            for l in range(self.coef.shape[-1]):
                coef=storecoef.copy()
                for m in range(self.coef.shape[-1]):
                    if m==l:
                        diff=1
                    else:
                        diff=0
                    coeftemp=np.zeros(coef.shape[1:])
                    for k in range(coef.shape[0]):
                        coeftemp=coeftemp+coef[k]*B[k]({'b':uvw[m]%1.},{'b':diff})*(1.-diff*(1-1/self.spacing[m]))
                    coef=coeftemp.copy()
                dXcoef.append(coef.copy())
            resultdX.append(np.array(dXcoef))
        if singleInput:
            resultdX=resultdX[0]
        return resultdX
      
    def getdC(self,coordsList,dxyz=None):#ind=[xIndex,yIndex,zIndex]
        ''' 
        Returns the weights of corresponding control points at respective coordinates
        Parameters:
            coordsList=[[x,y,z],] or [x,y,z]:list,np.ndarray
                Coordinate or list of coordinates
            dxyz=[dx,dy,dy]:list[int]
                the differentiation wrt x, y and z
        Return:
            dCList:float or list[float]
                weight of control point
            CIndList: int or list[int]
                Cumulative index of control point where index = x + y*num_x + z*num_y*num_x
        '''
        singleInput=False
        if type(dxyz)==type(None):
            dxyz=list(np.zeros(self.coef.shape[-1],dtype=int))
        if not(type(coordsList[0]) in [np.ndarray,list]):
            coordsList=[coordsList]
            singleInput=True
        dCList=[]
        CIndList=[]
        for n in range(len(coordsList)):
            coef,uvw=self.getCoefMat(coordsList[n])
            uvw_x=int(uvw[0])
            uvw_y=int(uvw[1])
            if self.coef.shape[-1]>2:
                uvw_z=int(uvw[2])
            else:
                uvw_z=0
                uvw=np.array(list(uvw)+[0])
            dC=[]
            CInd=[]   
            for k in range(coef.shape[0]):
                for l in range(coef.shape[1]):
                    if self.coef.shape[-1]>2:
                        mList=range(coef.shape[2])
                    else:
                        mList=[0]
                    for m in mList:
                        addInd=self.getCIndex([uvw_x+k-1,uvw_y+l-1,uvw_z+m-1][:self.coef.shape[-1]])
                        if addInd>=0:
                            if not(type(dxyz[0]) in [np.ndarray,list]):
                                if self.coef.shape[-1]>2:
                                    dC.append(B[k]({'b':uvw[0]%1.},{'b':dxyz[0]})/(self.spacing[0]**dxyz[0])*B[l]({'b':uvw[1]%1.},{'b':dxyz[1]})/(self.spacing[1]**dxyz[1])*B[m]({'b':uvw[2]%1.},{'b':dxyz[2]})/(self.spacing[2]**dxyz[2]))
                                else:
                                    dC.append(B[k]({'b':uvw[0]%1.},{'b':dxyz[0]})/(self.spacing[0]**dxyz[0])*B[l]({'b':uvw[1]%1.},{'b':dxyz[1]})/(self.spacing[1]**dxyz[1]))
                            else:
                                dC.append([])
                                for nn in range(len(dxyz)):
                                    if self.coef.shape[-1]>2:
                                        dC[-1].append(B[k]({'b':uvw[0]%1.},{'b':dxyz[nn][0]})/(self.spacing[0]**dxyz[nn][0])*B[l]({'b':uvw[1]%1.},{'b':dxyz[nn][1]})/(self.spacing[1]**dxyz[nn][1])*B[m]({'b':uvw[2]%1.},{'b':dxyz[nn][2]})/(self.spacing[2]**dxyz[nn][2]))
                                    else:
                                        dC[-1].append(B[k]({'b':uvw[0]%1.},{'b':dxyz[nn][0]})/(self.spacing[0]**dxyz[nn][0])*B[l]({'b':uvw[1]%1.},{'b':dxyz[nn][1]})/(self.spacing[1]**dxyz[nn][1]))
                            CInd.append(addInd)
            CIndList.append(np.array(CInd))
            dCList.append(np.array(dC))
        if singleInput:
            CIndList=CIndList[0]
            dCList=dCList[0]
        return (dCList,CIndList)
    def getCIndex(self,xyz):
        ''' 
        Returns the weights of corresponding control points at respective coordinates
        Parameters:
            xyz=[x,y,z]:list,np.ndarray
                index of control points
        Return:
            ind: int
                Cumulative index of control point where index = x + y*num_x + z*num_y*num_x
        '''
        xyz=xyz.copy()
        ind=0
        for n in range(self.coef.shape[-1]):
            if xyz[n]>=(self.coef.shape[n]):
                return -1
            elif xyz[n]<0:
                return -1
            ind+=xyz[n]*int(np.prod(self.coef.shape[:n]))
        return ind
    def getCfromIndex(self,indList):
        ''' 
        Returns the index (x,y,z) from the concatenated index
        Parameters:
            indList: int or list[int]
                Cumulative index of control point where index = x + y*num_x + z*num_y*num_x
        Return:
            xyz=[x,y,z] or [[x,y,z],]:list,np.ndarray
                index of control points
        '''
        singleInput=False
        if type(indList)==int:
            indList=[indList]
            singleInput=True
        xyzList=[]
        size=[]
        for n in range(self.coef.shape[-1]-1,-1,-1):
            size.append(int(np.prod(self.coef.shape[:n])))
        for n in range(len(indList)):
            ind=indList[n]
            xyz=np.zeros(self.coef.shape[-1],dtype=int)
            for m in range(self.coef.shape[-1]-1):
                while ind>=size[m]:
                    ind-=size[m]
                    xyz[self.coef.shape[-1]-1-m]+=1
            xyz[0]=ind
            xyzList.append(tuple(xyz))
        if singleInput:
            xyzList=xyzList[0]
        return xyzList
    def regrid(self,coordsList,coefList,tRef=None,weight=None,linearConstrainPoints=[],linearConstrainWeight=None):
        '''
        Regrid the BsplineFourier coefficients from sample points
        Parameters:
            coordsList=[[x,y,z],] or [x,y,z]:list,np.ndarray
                Coordinate or list of coordinates
            coefList(fourier,uvw):list,np.ndarray
                u, v and w fourier coefficients
        '''
        logger.info('Regriding '+str(len(coordsList))+'points.')
        noneSlice=[]
        for n in range(self.coef.shape[-1]):
            noneSlice.append(slice(None))
        if type(weight)==type(None):
            if len(coefList[0].shape)>1:
                weight=np.ones((coefList[0].shape[0],len(coordsList)))
            else:
                weight=np.ones(len(coordsList))
        elif len(weight.shape)==1 and len(coefList[0].shape)>1:
            weight=weight.reshape((1,-1))
            weight_temp=weight.copy()
            for n in range(coefList[0].shape[0]-1):
                weight=np.concatenate((weight,weight_temp),axis=0)
        else:
            weight=np.array(weight)
        Jmat=[]
        dCList,CIndList=self.getdC(coordsList)
        for n in range(len(dCList)):
            tempRow=sparse.csr_matrix((dCList[n].reshape((-1,),order='F'),(np.zeros(len(CIndList[n])),CIndList[n].copy())),shape=(1,self.numCoefXYZ))
            Jmat.append(tempRow.copy())
        if len(linearConstrainPoints)!=0:
            if type(linearConstrainWeight)==type(None):
                linearConstrainWeight=np.ones(len(linearConstrainPoints)*self.coef.shape[-1])
            elif type(linearConstrainWeight) in [int,float]:
                linearConstrainWeight=np.ones(len(linearConstrainPoints)*self.coef.shape[-1])*linearConstrainWeight
            else:
                linearConstrainWeight=(np.ones(self.coef.shape[-1])*linearConstrainWeight).reshape((-1,),order='F')
            weight=np.hstack((weight,linearConstrainWeight))
            dCListX,CIndListX=self.getdC(linearConstrainPoints,dxyz=[1,0,0])
            dCListY,CIndListY=self.getdC(linearConstrainPoints,dxyz=[0,1,0])
            if self.coef.shape[-1]>2:
                dCListZ,CIndListZ=self.getdC(linearConstrainPoints,dxyz=[0,0,1])
            for n in range(len(dCListX)):
                tempRow=sparse.csr_matrix((dCListX[n].reshape((-1,),order='F'),(np.zeros(len(CIndListX[n])),CIndListX[n].copy())),shape=(1,self.numCoefXYZ))
                Jmat.append(tempRow.copy())
                tempRow=sparse.csr_matrix((dCListY[n].reshape((-1,),order='F'),(np.zeros(len(CIndListY[n])),CIndListY[n].copy())),shape=(1,self.numCoefXYZ))
                Jmat.append(tempRow.copy())
                if self.coef.shape[-1]>2:
                    tempRow=sparse.csr_matrix((dCListZ[n].reshape((-1,),order='F'),(np.zeros(len(CIndListZ[n])),CIndListZ[n].copy())),shape=(1,self.numCoefXYZ))
                    Jmat.append(tempRow.copy())
        Jmat=sparse.vstack(Jmat)
        if len(coefList[0].shape)>1:
            for nFourier in range(1,coefList[0].shape[0]):
                matW=sparse.diags(weight[nFourier])
                matA=Jmat.transpose().dot(matW.dot(Jmat))
                for axis in range(coefList[0].shape[1]):
                    natb=Jmat.transpose().dot(weight[nFourier]*np.hstack((np.array(coefList)[:,nFourier,axis],np.zeros(len(linearConstrainPoints)*self.coef.shape[-1]))))
                    C=spsolve(matA, natb)
                    if type(C)!=np.ndarray:
                        C=C.todense()
                    if np.allclose(matA.dot(C), natb):
                        self.coef[tuple(noneSlice+[nFourier,axis])]=C.reshape(self.coef.shape[:self.coef.shape[-1]],order='F')
                    else:
                        logger.warning('Solution error at fourier term '+str(nFourier)+', and axis '+str(axis))
            if type(tRef)==type(None):
                self.coef[tuple(noneSlice+[0])]=0.
            else:
                fourierRef,=self.getdXYdC([tRef],remove0=True)
                self.coef[tuple(noneSlice+[0,0])]=-self.coef[tuple(noneSlice+[slice(1,None),0])].dot(fourierRef)
                self.coef[tuple(noneSlice+[0,1])]=-self.coef[tuple(noneSlice+[slice(1,None),1])].dot(fourierRef)
                if self.coef.shape[-1]>2:
                    self.coef[tuple(noneSlice+[0,2])]=-self.coef[tuple(noneSlice+[slice(1,None),2])].dot(fourierRef)
        else:
            matW=sparse.diags(weight)
            matA=Jmat.transpose().dot(matW.dot(Jmat))
            for axis in range(coefList[0].shape[0]):
                natb=Jmat.transpose().dot(weight*np.hstack((np.array(coefList)[:,axis],np.zeros(len(linearConstrainPoints)*self.coef.shape[-1]))))
                C=spsolve(matA, natb)
                if type(C)!=np.ndarray:
                    C=C.todense()
                if np.allclose(matA.dot(C), natb):
                    self.coef[tuple(noneSlice+[axis])]=C.reshape(self.coef.shape[:self.coef.shape[-1]],order='F')
                else:
                    logger.warning('Solution error at axis '+str(axis))
    def scale(self,s):
        ''' 
        Scales self values in origin, spacing,coordinates and coefficients
        Parameters:
            s:float
                scale factor
        '''
        self.spacing=self.spacing*s
        self.coordMat=self.coordMat*s
        self.coef=self.coef*s
        self.origin=self.origin*s

class ImageVector:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,coefFile=None,timeMap=[None,None],fileScale=1.,delimiter=' ',origin=None):
        '''mgridslice=[]
            for n in range(self.coef.shape[writeSITKfile(self,filepath,imageSize=None,imageSpacing=None)-1]):
                mgridslice.append(slice(self.origin[n],(self.origin[n]+(shape[n]-0.1)*self.spacing[n]),self.spacing[n]))
            self.coordMat = np.mgrid[tuple(mgridslice)].reshape(self.coef.shape[-1],*shape[:self.coef.shape[-1]]).transpose(*tuple(range(1,self.coef.shape[-1]+1)),0)
            
        Initialize all data.
        Note:  coef[x,y,z,uvw]
        spacing=[x,y,z]
        origin=[x,y,z]
        timeMap=[t1,t2] vector stored maps t1 to t2
        '''
        self.coef=None
        self.origin=None
        self.timeMap=timeMap
        if type(coefFile)!=type(None):
            self.read(coefFile=coefFile,timeMap=timeMap,fileScale=fileScale,delimiter=delimiter,origin=origin)
        logger.info('timeMap= '+str(self.timeMap))
    def read(self,coefFile=None,timeMap=[None,None],fileScale=1.,delimiter=' ',origin=None):
        img=medImgProc.imread(coefFile,dimension=['x','y','z'],module='medpy')
        self.coef=[]
        self.coef.append(img.clone())
        self.coef[0].data=self.coef[0].data.transpose(3,1,0,2)
        self.coef.append(self.coef[0].clone())
        self.coef[1].data=self.coef[1].data[0]/fileScale
        self.coef.append(self.coef[0].clone())
        self.coef[2].data=self.coef[2].data[2]/fileScale
        self.coef[0].data=self.coef[0].data[1]/fileScale
        if type(origin)==type(None):
            self.origin=[0.,0.,0.]
        else:
            self.origin=origin
    def getVector(self,coordsList,vec=[0,1,2],CPU=1):
        ''' 
        Returns vector corresponding to the coordinates
        Parameters:
            coordsList=[[x,y,z],] or [x,y,z]:list,np.ndarray
                Coordinate or list of coordinates
        Return:
            resultVectors=[u,v,w] or [[u,v,w],]:np.ndarray or list[np.ndarray]
                vector or list of vectors corresponding to the input coordinate
        '''
        
        singleInput=False
        if not(type(coordsList[0]) in [np.ndarray,list]):
            coordsList=[coordsList]
            singleInput=True
        if singleInput:
            resultVectors=np.zeros((1,3))
        else:
            resultVectors=np.zeros((len(coordsList),3))
        for axis in vec:
            resultVectors[:,axis]=self.coef[axis].getData(coordsList,CPU=CPU)#,sampleFunction=medImgProc.image.linearSampling)
        if singleInput:
            return resultVectors[0,vec]
        else:
            return resultVectors[:,vec]
    def getdX(self,coordsList):
        ''' 
        Returns vector corresponding to the coordinates
        Parameters:
            coordsList=[[x,y,z],] or [x,y,z] (for BsplineFourier: [[x,y,z,t],] or [x,y,z,t]):list,np.ndarray
                Coordinate or list of coordinates
        Return:
            resultdX=[d[u,v,w]/dx,d[u,v,w]/dy,d[u,v,w]/dz] or[[d[u,v,w]/dx,d[u,v,w]/dy,d[u,v,w]/dz],]:np.ndarray(2d) or list[np.ndarray(2d)]
                gradient of vector or list of gradient of vectors corresponding to the input coordinate
        '''
        singleInput=False
        if not(type(coordsList[0]) in [np.ndarray,list]):
            coordsList=[coordsList]
            singleInput=True
        coordsList=np.array(coordsList)
        resultdX=[]
        for axis in range(3):
            coordplus=coordsList
            coordminus=coordsList
            coordplus[:,axis]+=self.coef[0].dimlen[self.coef[0].dim[axis]]
            coordminus[:,axis]-=self.coef[0].dimlen[self.coef[0].dim[axis]]
            resultdX.append((self.getVector(coordplus)-self.getVector(coordminus))/2./self.coef[0].dimlen[self.coef[0].dim[axis]])
        resultdX=np.array(resultdX).transpose(1,0,2)
        if singleInput:
            resultdX=resultdX[0]
        return resultdX
class BsplineFourier(Bspline):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,coefFile=None,shape=None,fourierFormat='fccss',spacing=None,delimiter=' ',origin=None):
        '''
        Initialize all data.
        Note:  coef[x,y,z,fourierTerms,uvw]
        self.coordMat[indexX,indexY,indexZ]=coordinateXYZ
        self.spacing=[x,y,z,t(period in seconds)]
        self.origin=[x,y,z,t(in second)]
        Fourier Terms [a0,a1,a2,...,b1,b2,...]=coordinate+a1cos(w/p*t)+a2cos(2w/p*t)+b1sin(w/p*t)+b2sin(2w/p*t)+...
        a0 is ignored
        self.fourierFormat:str=>f/r (forward/reverse), c =cos, s =sin
        
        '''
        super().__init__()
        self.fourierFormat=None
        self.numCoefXYZ=0
        self.numCoef=0
        if coefFile!=None:
            self.readFile(coefFile=coefFile,shape=shape,fourierFormat=fourierFormat,spacing=spacing,delimiter=delimiter,origin=origin)
    def initialize(self,shape,fourierFormat=None,spacing=None,delimiter=' ',origin=None):
        if type(origin)==type(None):
            if type(self.origin)==type(None):
                self.origin=np.zeros(len(shape)-1)
        else:
            self.origin=origin
        if type(spacing)==type(None):
            if type(self.spacing)==type(None):
                self.spacing=np.ones(len(shape)-1)
        else:
            self.spacing=spacing
        if type(fourierFormat)==type(None):
            if type(self.fourierFormat)==type(None):
                self.fourierFormat='fccss'
        else:
            self.fourierFormat=fourierFormat
        if type(self.coordMat)==type(None):
            mgridslice=[]
            for n in range(shape[-1]):
                mgridslice.append(slice(self.origin[n],(self.origin[n]+(shape[n]-0.1)*self.spacing[n]),self.spacing[n]))
            self.coordMat = np.mgrid[tuple(mgridslice)].reshape(shape[-1],*shape[:shape[-1]]).transpose(*tuple(range(1,shape[-1]+1)),0)
        if type(self.coef)==type(None):
            self.coef=np.zeros(shape)
        logger.info('Origin= '+str(self.origin))
        logger.info('Spacing= '+str(self.spacing))
        logger.info('Fourier Format= '+str(self.fourierFormat))
        self.numCoefXYZ=1
        for n in self.coef.shape[:self.coef.shape[-1]]:
            self.numCoefXYZ*=n
        self.numCoef=int(self.numCoefXYZ*(self.coef.shape[self.coef.shape[-1]]-1))
        
    def readFile(self,coefFile=None,shape=None,fourierFormat='fccss',spacing=None,delimiter=' ',skiprows=1,origin=None):
        if coefFile!=None:
            self.read(coefFile=coefFile,shape=shape,spacing=spacing,delimiter=delimiter,origin=origin)
            self.fourierFormat=fourierFormat
            #get fourier format to fccss
            self.convertToFCCSS()
            self.initialize(self.coef.shape)
    def toBsplineU(self,bspline,tRef=None,variableIdentifier=''):
        return BsplineFunctionofBsplineFourierAD('u',bspline,self,tRef=tRef,variableIdentifier=variableIdentifier)
    def toBsplineV(self,bspline,tRef=None,variableIdentifier=''):
        return BsplineFunctionofBsplineFourierAD('v',bspline,self,tRef=tRef,variableIdentifier=variableIdentifier)
    def toBsplineW(self,bspline,tRef=None,variableIdentifier=''):
        return BsplineFunctionofBsplineFourierAD('w',bspline,self,tRef=tRef,variableIdentifier=variableIdentifier)
    '''
    #arithmatics -- currently not supported
    def __add__(self, other):
        return;
    '''
    '''
    Function to receive data
    '''
    def getRefCoef(self,coord):
        ''' 
        Get coefficients value in matrix form
        Parameters:
            coord=[x,y,z]:list,np.ndarray
                Coordinate
        Return:
            C:np.ndarray
                coefficients corresponding to the input coordinate(s) in the C(fourier,uvw)
        '''
        coef,uvw=self.getCoefMat(coord)
        uvw_x=int(uvw[0])
        uvw_y=int(uvw[1])
        if len(uvw)>2:
            uvw_z=int(uvw[2])
        else:
            uvw_z=0
            uvw=np.array(list(uvw)+[0])
        C=np.zeros(coef.shape[self.coef.shape[-1]:])
        CInd=[]
        for k in range(coef.shape[0]):
            for l in range(coef.shape[1]):
                if self.coef.shape[-1]>2:
                    for m in range(coef.shape[2]):
                        C+=coef[k,l,m]*B[k]({'b':uvw[0]%1.},{})*B[l]({'b':uvw[1]%1.},{})*B[m]({'b':uvw[2]%1.},{})
                else:
                    C+=coef[k,l]*B[k]({'b':uvw[0]%1.},{})*B[l]({'b':uvw[1]%1.},{})
        return C
    def getNormDistanceMat(self,coord,xyzRadius):
        ''' 
        Depriciated: Use with caution.
        '''
        coordSlice=[]
        xyzSize=[]
        for n in range(3):
            xyzSize.append(int(xyzRadius[n]/self.spacing[n]))
        for n in range(3):
            if type(coord)==type(None):
                maxSize=np.array(xyzSize).max()
                coordSlice.append(slice(maxSize-xyzSize[n],maxSize+xyzSize[n]+1))
            else:
                xyzInd=int((coord[n]-self.origin[n])/self.spacing[n])
                coordSlice.append(slice(xyzInd-xyzSize[n],xyzInd+xyzSize[n]+2))
        coordselected=self.coordMat[coordSlice].transpose(3,0,1,2)
        if type(coord)==type(None):
            coord=maxSize*self.spacing[:3]+self.origin[:3]
        dist=np.sqrt(((coordselected[0]-coord[0])/xyzRadius[0])**2.+((coordselected[1]-coord[1])/xyzRadius[1])**2.+((coordselected[2]-coord[2])/xyzRadius[2])**2.)
        return (dist,coordSlice)
    def getFourierCoefFromRef(self,coordsList):
        ''' 
        Get fourier coefficients value in matrix form
        Parameters:
            coordsList=[[x,y,z],] or [x,y,z]:list,np.ndarray
                Coordinate or list of coordinates
        Return:
            resultFourierCoef:list,np.ndarray
                Fourier coefficients corresponding to the input coordinate(s) in the C(fourier,uvw)
        '''
        singleInput=False
        if not(type(coordsList[0]) in [np.ndarray,list]):
            coordsList=[coordsList]
            singleInput=True
        resultFourierCoef=[]
        noneSlice=[]
        for n in range(self.coef.shape[-1]):
            noneSlice.append(slice(None))
            
        for n in range(len(coordsList)):
            coef,uvw=self.getCoefMat(coordsList[n])
            coeftemp=coef.copy()
            for m in range(int(coef.shape[self.coef.shape[-1]]/2)):#sub in t
                coeftemp[tuple(noneSlice+[m+1])]=coef[tuple(noneSlice+[m+1])]*np.cos((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]]*(coordsList[n][self.coef.shape[-1]]-self.origin[self.coef.shape[-1]]))
                coeftemp[tuple(noneSlice+[int(coef.shape[self.coef.shape[-1]]/2)+m+1])]=coef[tuple(noneSlice+[int(coef.shape[self.coef.shape[-1]]/2)+m+1])]*np.sin((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]]*(coordsList[n][self.coef.shape[-1]]-self.origin[self.coef.shape[-1]]))
            coef=coeftemp.copy()
            for m in range(self.coef.shape[-1]):
                coeftemp=np.zeros(coef.shape[1:])
                for k in range(coef.shape[0]):
                    coeftemp=coeftemp+coef[k]*B[k]({'b':uvw[m]%1.},{})
                coef=coeftemp.copy()
            for axis in range(self.coef.shape[-1]):
                coef+=np.array(coordsList[n][axis])
            resultFourierCoef.append(coef.copy())
        if singleInput:
            resultFourierCoef=resultFourierCoef[0]
        return resultFourierCoef
    def getCoordFromRef(self,coordsList,vec=None,dxyzt=None,CPU=1):
        resultCoords=self.getVector(coordsList,vec=vec,addCoord=True,dxyzt=dxyzt,CPU=CPU)
        return resultCoords
    def getRefFromCoord(self,coordsList,maxErrorRatio=0.001,maxIteration=1000,lmLambda_init=0.001,lmLambda_incrRatio=5.,lmLambda_max=float('inf'),lmLambda_min=0.):
        ''' 
        Get coordinates at time t
        Parameters:
            coordsList=[[x,y,z,t],] or [x,y,z,t]:list,np.ndarray
                Coordinate or list of coordinates where x,y,z are at the phantom time point
        Return:
            resultCoords:list,np.ndarray
                resultant coordinates corresponding to the input coordinate(s) in the C(fourier,uvw)
        '''
        singleInput=False
        if not(type(coordsList[0]) in [np.ndarray,list]):
            coordsList=[coordsList]
            singleInput=True
        resultCoords=[]
        for n in range(len(coordsList)):
            lmLambda=lmLambda_init
            reductionRatio=1.
            ref=coordsList[n].copy()
            error=self.getCoordFromRef(ref)[:self.coef.shape[-1]]-coordsList[n][:self.coef.shape[-1]]
            maxerror=self.spacing[:self.coef.shape[-1]]*maxErrorRatio
            count=0
            while np.any(error>maxerror) and count<maxIteration:
                Jmat=self.getdX(ref)
                Jmat=Jmat+lmLambda*np.diag(np.diag(Jmat))
                dX=np.linalg.solve(Jmat, error)
                newref=ref.copy()
                newref[:self.coef.shape[-1]]=newref[:self.coef.shape[-1]]+dX
                newError=self.getCoordFromRef(newref)[:self.coef.shape[-1]]-coordsList[n][:self.coef.shape[-1]]
                if np.sqrt(np.mean(newError**2))<=np.sqrt(np.mean(error**2)):
                    error=newError
                    ref=newref
                    if (lmLambda/np.sqrt(lmLambda_incrRatio))>lmLambda_min:
                        lmLambda=lmLambda/np.sqrt(lmLambda_incrRatio)
                    else:
                        lmLambda=lmLambda_min
                    count+=1
                else:
                    if (lmLambda*lmLambda_incrRatio)<lmLambda_max:
                        lmLambda*=lmLambda_incrRatio
                    else:
                        lmLambda=lmLambda_max
                    count+=0.02
                if count==maxIteration:
                    logger.warning('Maximum iterations reached for point '+str(m)+str(self.points[m]))
            resultCoords.append(ref[:self.coef.shape[-1]].copy())
        if singleInput:
            resultCoords=resultCoords[0]
        return resultCoords

    def getStrain(self,coordsList,constrainVal=-1.):
        ''' 
        Returns strain corresponding to the coordinates
        Parameters:
            coordsList=[[x,y,z],] or [x,y,z]:list,np.ndarray
                Coordinate or list of coordinates
        Return:
            strain=[[du/dx-1,du/dy*dv/dx,du/dz*dw/dx],[du/dy*dv/dx,dv/dy-1,dv/dz*dw/dy],[du/dz*dw/dx,dv/dz*dw/dy,dw/dz-1]]:np.ndarray(2d) or list[np.ndarray(2d)]
                strain test where strain <=-1 represents overlap
        '''
        resultdXList=self.getdX(coordsList)
        singleInput=False
        if type(resultdXList)!=list:
            resultdXList=[resultdXList]
            singleInput=True
        strain=[]
        for n in range(len(resultdXLiist)):
            resultdX=np.zeros(resultdXList[n].shape)
            for m in range(self.coef.shape[-1]):
                for k in range(self.coef.shape[-1]):
                    if m!=k:
                        resultdX[m,k]=resultdXList[n][m,k]*resultdXList[n][k,m]*-1.
            strain.append(resultdX.copy())
        return strain
    def getdXYdC(self,timeMap,remove0=True):#ind=[xIndex,yIndex,zIndex]
        ''' 
        Returns the weights of corresponding control points where the fourier terms are summed
        Parameters:
            timeMap=[t_start,t_end] :list,np.ndarray
                time map from t_start to t_end
        Return:
            dX:np.ndarray
                weight of control points at t_start
            dX:np.ndarray
                weight of control points at t_end
        '''
        dXY=[]
        noneSlice=[]
        for n in range(self.coef.shape[-1]):
            noneSlice.append(slice(None))
            
        for timeN in range(len(timeMap)):
            dX=np.ones(self.coef.shape[self.coef.shape[-1]])
            for m in range(int(self.coef.shape[self.coef.shape[-1]]/2)):#get T matrix
                dX[m+1]=np.cos((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]]*(timeMap[timeN]-self.origin[self.coef.shape[-1]]))
                dX[int(self.coef.shape[self.coef.shape[-1]]/2)+m+1]=np.sin((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]]*(timeMap[timeN]-self.origin[self.coef.shape[-1]]))
            if remove0:
                dX=dX[1:]
            dXY.append(dX.copy())
        return dXY
    def getBspline(self,time,refTime=float('nan')):
        ''' 
        Returns the coef of all control points where with time evaluated wrt ref
        Parameters:
            timeMap=[t_start,t_end] or t_end(wrt t_ref) :list ,np.ndarray or float
                time map from t_start to t_end
        Return:
            vector:np.ndarray
                vector coef of all control points
        '''
        if type(time) in [int,float]:
            time=[None,time]
        noneSlice=[]
        for n in range(self.coef.shape[-1]):
            noneSlice.append(slice(None))
        coef=self.coef[tuple(noneSlice+[0])].copy()
        for m in range(int(self.coef.shape[self.coef.shape[-1]]/2)):#get T matrix
            coef+=self.coef[tuple(noneSlice+[m+1])]*np.cos((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]]*(time[1]-self.origin[self.coef.shape[-1]]))+self.coef[tuple(noneSlice+[int(self.coef.shape[self.coef.shape[-1]]/2)+m+1])]*np.sin((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]]*(time[1]-self.origin[self.coef.shape[-1]]))
        if type(time[0])!=type(None):
            coef-=self.coef[tuple(noneSlice+[0])].copy()
            for m in range(int(self.coef.shape[self.coef.shape[-1]]/2)):#get T matrix
                coef-=self.coef[tuple(noneSlice+[m+1])]*np.cos((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]]*(time[0]-self.origin[self.coef.shape[-1]]))-self.coef[tuple(noneSlice+[int(self.coef.shape[self.coef.shape[-1]]/2)+m+1])]*np.sin((m+1.)*2.*np.pi/self.spacing[self.coef.shape[-1]]*(time[0]-self.origin[self.coef.shape[-1]]))
        b=Bspline(coefFile=coef,timeMap=[refTime,time],spacing=self.spacing[:-1].copy(),origin=self.origin[:-1].copy())
        return b
    def writeBspline(self,time,filepath,refTime=float('nan'),imageSize=None,imageSpacing=None):
        b=self.getBspline(time,refTime=refTime)
        b.writeSITKfile(filepath,imageSize=imageSize,imageSpacing=imageSpacing)
    '''
    Function to change data
    '''

    '''
    Format conversion functions
    '''
    def convertToFCCSS(self):
        newcoef=self.getFCCSS()
        if type(newcoef)!=type(None):
            self.coef=newcoef
            self.fourierFormat='fccss'
    def getFCCSS(self):
        #get fourier format to fccss
        if self.fourierFormat=='fRalpha':
            newcoef=convertfromFRalpha()
            return newcoef
        fourierRearrange=None
        if self.fourierFormat=='fccss':
            return self.coef.copy()
        elif self.fourierFormat=='fcscs':
            fourierRearrange=[0]+list(range(1,ftermsNum,2))+list(range(2,ftermsNum,2))
        elif self.fourierFormat=='fscsc':
            fourierRearrange=[0]+list(range(2,ftermsNum,2))+list(range(1,ftermsNum,2))
        elif self.fourierFormat=='fsscc':
            fourierRearrange=[0]+list(range(int(ftermsNum/2+1),ftermsNum))+list(range(1,int(ftermsNum/2+1)))
        elif self.fourierFormat=='rccss':
            fourierRearrange=[-1]+list(range(0,int(ftermsNum/2)))+list(range(int(ftermsNum/2),ftermsNum-1))
        elif self.fourierFormat=='rcscs':
            fourierRearrange=[-1]+list(range(ftermsNum-3,-1,-2))+list(range(ftermsNum-2,0,-2))
        elif self.fourierFormat=='rscsc':
            fourierRearrange=[-1]+list(range(ftermsNum-2,0,-2))+list(range(ftermsNum-3,-1,-2))
        elif self.fourierFormat=='rsscc':
            fourierRearrange=list(range(ftermsNum-1,-1,-1))
        tempArrList=list(range(self.coef.shape[-1]+2))
        tempArrList.pop(self.coef.shape[-1])
        tempArrList.insert(0,self.coef.shape[-1])
        coeftemp=self.coef.transpose(*tempArrList)
        newcoef=None
        if fourierRearrange!=None:
            coeftemp[:]=coeftemp[fourierRearrange]
            tempArrList=list(range(1,self.coef.shape[-1]+2))
            tempArrList.insert(self.coef.shape[-1],0)
            newcoef=coeftemp.transpose(*tempArrList)
        return newcoef
    def convertfromFRalpha(self):
        outcoef=None
        if self.fourierFormat=='fRalpha':
            ftermsNum=self.coef.shape[-2]
            tempArrList=list(range(self.coef.shape[-1]+2))
            tempArrList.pop(self.coef.shape[-1])
            tempArrList.insert(0,self.coef.shape[-1])
            coeftemp=self.coef.transpose(*tempArrList)
            outcoef=np.zeros(coeftemp.shape)
            for n in range(int(ftermsNum/2)):
                outcoef[n+1]=coeftemp[2*n]*np.sin(coeftemp[2*n+1]*2.*np.pi*(n+1.)/self.spacing[-1])
                outcoef[int(ftermsNum/2+1)+n]=coeftemp[2*n]*np.cos(coeftemp[2*n+1]*2.*np.pi*(n+1.)/self.spacing[-1])
                outcoef[0]=outcoef[0]+outcoef[n+1]
            #transpose fourier terms back to [x,y,z,f,uvw]
            tempArrList=list(range(1,self.coef.shape[-1]+2))
            tempArrList.insert(self.coef.shape[-1],0)
            outcoef=outcoef.transpose(*tempArrList)
        return outcoef
    def convertToFRalpha(self):
        newcoef=None
        if self.fourierFormat=='fccss':
            tempArrList=list(range(self.coef.shape[-1]+2))
            tempArrList.pop(self.coef.shape[-1])
            tempArrList.insert(0,self.coef.shape[-1])
            coeftemp=self.coef.transpose(*tempArrList)
            newcoef=np.zeros(coeftemp.shape)
            for n in range(int(ftermsNum/2)):
                newcoef[2*n]=np.sqrt(coeftemp[n+1]**2.+coeftemp[int(ftermsNum/2+1)+n]**2.)
                newcoef[2*n+1]=np.arctan2(coeftemp[n+1],coeftemp[int(ftermsNum/2+1)+n])/2./np.pi/(n+1.)*self.spacing[-1]
            #transpose fourier terms back to [x,y,z,f,uvw]
            tempArrList=list(range(1,self.coef.shape[-1]+2))
            tempArrList.insert(self.coef.shape[-1],0)
            newcoef=newcoef.transpose(*tempArrList)
        return newcoef
    '''
    Smoothing functions
    '''
    def GaussianAmpEdit(self,coord,radius,ratio,sigma=0.4):
        if type(radius) not in [list,np.ndarray]:
            radius=list(np.ones(self.coef.shape[-1])*radius)
        dist,coordSlice=self.getNormDistanceMat(coord,radius)
        gaussMat = np.exp(-( (dist)**2 / ( 2.0 * sigma**2 ) ) )
        tempArrList=list(range(1,self.coef.shape[-1]+1))+[0]
        gaussMat=np.array([gaussMat,gaussMat,gaussMat]).transpose(*tempArrList)
        tempArrList=list(range(self.coef.shape[-1]+2))
        tempArrList.pop(self.coef.shape[-1])
        tempArrList.insert(0,self.coef.shape[-1])
        tempCoef=self.coef.transpose(*tempArrList)
        for n in range(int(self.coef.shape[-2]/2)):
            tempCoef[2*n][coordSlice]=tempCoef[2*n][coordSlice]*(1+gaussMat*(ratio-1.))
        tempArrList=list(range(1,self.coef.shape[-1]+2))
        tempArrList.insert(self.coef.shape[-1],0)
        self.coef=tempCoef.transpose(*tempArrList)
        return
    def GaussianAmpSmoothing(self,radius,targetCoef=None,coord=None,sigma=0.4,ratio=1.):
        if type(targetCoef)==type(None):
            targetCoef=np.array(range(self.coef.shape[-2]))
        if type(radius)!=list:
            radius=[radius,radius,radius]
        dist,coordSlice=self.getNormDistanceMat(coord,radius)    
        gaussMat= np.exp(-( (dist)**2 / ( 2.0 * sigma**2 ) ) )
        gaussMat=np.array([gaussMat,gaussMat,gaussMat]).transpose(1,2,3,0)
        tempArrList=list(range(self.coef.shape[-1]+2))
        tempArrList.pop(self.coef.shape[-1])
        tempArrList.insert(0,self.coef.shape[-1])
        refcoef=self.coef.transpose(*tempArrList)
        newcoef=self.coef.transpose(*tempArrList)
        tempShape=refcoef.shape[1:]
        for n in targetCoef:
            tempCoef=np.zeros(tempShape)
            totalweight=np.zeros(tempShape)
            if type(coord)==type(None):
                coordInd=[]
                for m in self.coef.shape[-1]:
                    coordInd.append(range(int(dist.shape[m]/2),tempShape[m]-int(dist.shape[m]/2)))
            else:
                coordInd=[]
                for m in self.coef.shape[-1]:
                    coordInd+=range(coordSlice[m].start,coordSlice[m].stop)
            for x in coordInd[0]:
                for y in coordInd[1]:
                    if self.coef.shape[-1]>2:
                        for z in coordInd[2]:
                            coordSlice=[slice(x-int(dist.shape[0]/2),x-int(dist.shape[0]/2)+dist.shape[0]),slice(y-int(dist.shape[1]/2),y-int(dist.shape[1]/2)+dist.shape[1]),slice(z-int(dist.shape[2]/2),z-int(dist.shape[2]/2)+dist.shape[2])]
                            newcoef[n][x,y,z]=np.sum(gaussMat*refcoef[n][coordSlice])/np.sum(gaussMat)*ratio+(1.-ratio)*refcoef[n][x,y,z]
                    else:
                        coordSlice=[slice(x-int(dist.shape[0]/2),x-int(dist.shape[0]/2)+dist.shape[0]),slice(y-int(dist.shape[1]/2),y-int(dist.shape[1]/2)+dist.shape[1])]
                        newcoef[n][x,y]=np.sum(gaussMat*refcoef[n][coordSlice])/np.sum(gaussMat)*ratio+(1.-ratio)*refcoef[n][x,y]
        tempArrList=list(range(1,self.coef.shape[-1]+2))
        tempArrList.insert(self.coef.shape[-1],0)
        self.coef=newcoef.transpose(*tempArrList)
        return
    
    def regridToTime(self,coordsList,coefList,time,shape=None):
        if type(shape)!=type(None):
            self.coef=np.zeros(shape)
        if type(time)!=type(None):            
            p=[]
            for n in range(len(coordsList)):
                p.append(coordsList[n].copy())
                for m in range(int(coefList[n].shape[0]/2)):#sub in t
                    p[-1]=p[-1]+coefList[n][m+1]*np.cos((m+1.)*2.*np.pi*(time-self.origin[-1])/self.spacing[-1])+coefList[n][int(coefList[n].shape[0]/2)+m+1]*np.sin((m+1.)*2.*np.pi*(time-self.origin[-1])/self.spacing[-1])
        else:
            p=coordsList.copy()
        self.regrid(p,coefList,tRef=time)
        
    def reshape(self,shape,translate=None):
        ''' 
        reshape the BsplineFourier coefficients
        Parameters:
            shape=[x,y,z,fourier_terms,3]:list,np.ndarray
                Shape of the final Bspline Fourier
            translate=[[x_start,x_end],[y_start.,y_end],[z_start,z_end]]:list,np.ndarray
                translate the origin by +[x_start,y_start,z_start] and the last control point by +[x_end,y_end,z_end]
        '''
            
        if type(translate)==type(None):
            translate=np.zeros((self.coef.shape[-1],2))
        if np.all(np.array(translate)==0):
            origin=self.origin.copy()
        else:
            origin=[]
            for n in range(self.coef.shape[-1]):
                origin.append(self.origin[n]+translate[n][0])
            origin.append(self.origin[self.coef.shape[-1]])
            origin=np.array(origin)
        
        
        
        newbsFourier=BsplineFourier()
        spacing=[]
        for n in range(self.coef.shape[-1]):
            spacing.append((self.spacing[n]*(self.coef.shape[n]-1)+translate[n][1]-translate[n][0])/(shape[n]-1))
        spacing.append(self.spacing[self.coef.shape[-1]])
        spacing=np.array(spacing)
        newbsFourier.initialize(shape,spacing=spacing,origin=origin)
        
        if type(self.coef)!=type(None): 
            if np.any(np.array(shape)!=np.array(self.coef.shape)) and (self.coef.max()!=0 or self.coef.min()!=0):
                sampleCoef=[]
                samplePoints=np.array(newbsFourier.samplePoints())
                for m in range(len(samplePoints)):
                    sampleCoefTemp=np.zeros(newbsFourier.coef.shape[self.coef.shape[-1]:])
                    sampleCoefTemp[:self.coef.shape[self.coef.shape[-1]],:]=self.getRefCoef(samplePoints[m])
                    sampleCoef.append(sampleCoefTemp.copy())
                newbsFourier.regrid(samplePoints,sampleCoef)              
        return newbsFourier
    def motionImage(self,imageSize=None,spacing=None,coefFourierWeight=None,evaluateFunc=None,xList=None,yList=None,zList=None,scaleFromGrid=None):
        ''' 
        Create an image based on the amplitude of fourier coefficients
        Parameters:
            imageSize=[x,y,z]:list,np.ndarray
                image pixel size
            coefFourierWeight=[fourierterm1,fourierterm2]:list,np.ndarray
                weighs the courier terms of different frequencies cosine and sine are considered as a single term
            xList: range
                fills x pixels
            yList: range
                fills y pixels
            zList: range
                fills z pixels
            scaleFromGrid: float of np.ndarray(3)
                scale image size from bspline grid if image size is not given
        Return:
            imgData:np.ndarray
                image intensity data
            imgDimlen: dict
                conversion of image pixel to real coordinates
        Note: origin is conserved
        '''
        if type(scaleFromGrid)==type(None):
            scaleFromGrid=10.
        if type(imageSize)==type(None):
            imageSize=np.array(self.coef.shape[:self.coef.shape[-1]])*scaleFromGrid
            spacing=np.array(self.spacing[:self.coef.shape[-1]])/scaleFromGrid
        elif type(spacing)==type(None):
            spacing=np.array(self.spacing[:self.coef.shape[-1]]*((np.array(self.coef.shape[:self.coef.shape[-1]])-1-2*self.origin[:self.coef.shape[-1]]/self.spacing[:self.coef.shape[-1]])/(np.array(imageSize)-1)))
        imageSize=np.array(imageSize).astype(int)
        if type(coefFourierWeight)==type(None):
            coefFourierWeight=np.zeros(int(self.coef.shape[self.coef.shape[-1]]/2))
            coefFourierWeight[0]=1.
        maxomega=np.argmax(coefFourierWeight)

        imgData=np.zeros(imageSize)
        if type(xList)==type(None):
            xList=range(imageSize[0])
        else:
            imgData=imgData[xList]
        if type(yList)==type(None):
            yList=range(imageSize[1])
        else:
            imgData=imgData[:,yList]
        if self.coef.shape[-1]>2:
            if type(zList)==type(None):
                zList=range(imageSize[2])
            else:
                imgData=imgData[:,:,zList]
        
        imgDimlen={'x':spacing[0],'y':spacing[1]}
        if self.coef.shape[-1]>2:
            imgDimlen['z']=spacing[2]
        for xn in range(len(xList)):
            logger.info('    {0:.3f}% completed...'.format(float(xn)/len(xList)*100.))
            for yn in range(len(yList)):
                if self.coef.shape[-1]>2:
                    for zn in range(len(zList)):
                        if type(evaluateFunc)==type(None):
                            vec=self.getVector([xList[xn]*imgDimlen['x'],yList[yn]*imgDimlen['y'],zList[zn]*imgDimlen['z']])
                            fvalue=np.zeros(int(self.coef.shape[self.coef.shape[-1]]/2))
                            for m in range(int(self.coef.shape[self.coef.shape[-1]]/2)):
                                fvalue[m]=np.sqrt((vec[m+1]**2.+vec[int(self.coef.shape[self.coef.shape[-1]]/2)+m+1]**2.).sum())
                        else:
                            fvalue=evaluateFunc({'x':xList[xn]*imgDimlen['x'],'y':yList[yn]*imgDimlen['y'],'z':zList[zn]*imgDimlen['z']},{})
                            if not(isinstance(fvalue,(int,float,np.ndarray))):
                                fvalue=np.sqrt(fvalue.cosine**2.+fvalue.sine**2.)
                        imgData[xn,yn,zn]=(fvalue*coefFourierWeight).sum()
                        '''
                        weightRatio=fvalue[maxomega]/coefFourierWeight[maxomega]
                        errorRMS=0.
                        for m in range(int(self.coef.shape[3]/2)):
                            errorRMS+=(weightRatio*coefFourierWeight[m]-fvalue[m])**2.
                        errorRMS=np.sqrt(errorRMS/float(self.coef.shape[3]/2-1))
                        imgData[x,y,z]=weightRatio-errorRMS
                        '''
                else:
                    if type(evaluateFunc)==type(None):
                        vec=self.getVector([xList[xn]*imgDimlen['x'],yList[yn]*imgDimlen['y']])
                        fvalue=np.zeros(int(self.coef.shape[self.coef.shape[-1]]/2))
                        for m in range(int(self.coef.shape[self.coef.shape[-1]]/2)):
                            fvalue[m]=np.sqrt((vec[m+1]**2.+vec[int(self.coef.shape[self.coef.shape[-1]]/2)+m+1]**2.).sum())
                    else:
                        fvalue=evaluateFunc({'x':xList[xn]*imgDimlen['x'],'y':yList[yn]*imgDimlen['y']},{})
                        if not(isinstance(fvalue,(int,float,np.ndarray))):
                            fvalue=np.sqrt(fvalue.cosine**2.+fvalue.sine**2.)
                    imgData[xn,yn]=(fvalue*coefFourierWeight).sum()
        return (imgData,imgDimlen)

    def fcoefImage(self,imageSize=None,spacing=None,coefFourierWeight=None,xList=None,yList=None,zList=None,scaleFromGrid=None):
        ''' 
        Create an image based sampling of fourier coefficients
        Parameters:
            imageSize=[x,y,z]:list,np.ndarray
                image pixel size
            coefFourierWeight=[fourierterm1,fourierterm2]:list,np.ndarray
                weighs the courier terms of different frequencies cosine and sine are considered as a single term
            xList: range
                fills x pixels
            yList: range
                fills y pixels
            zList: range
                fills z pixels
            scaleFromGrid: float of np.ndarray(3)
                scale image size from bspline grid if image size is not given
        Return:
            imgData:np.ndarray
                image intensity data
            imgDimlen: dict
                conversion of image pixel to real coordinates
        Note: origin is conserved
        '''
        if type(scaleFromGrid)==type(None):
            scaleFromGrid=10.
        if type(imageSize)==type(None):
            imageSize=np.array(self.coef.shape[:self.coef.shape[-1]])*scaleFromGrid
            spacing=np.array(self.spacing[:self.coef.shape[-1]])/scaleFromGrid
        elif type(spacing)==type(None):
            spacing=np.array(self.spacing[:self.coef.shape[-1]]*((np.array(self.coef.shape[:self.coef.shape[-1]])-1)/(np.array(imageSize)-1)))
        imageSize=np.array(imageSize).astype(int)
        if type(coefFourierWeight)==type(None):
            coefFourierWeight=np.zeros(int(self.coef.shape[self.coef.shape[-1]]/2))
            coefFourierWeight[0]=1.
        maxomega=np.argmax(coefFourierWeight)
        
        imageSize=np.array([*imageSize,*self.coef.shape[self.coef.shape[-1]:]])
        imgData=np.zeros(imageSize)
        if type(xList)==type(None):
            xList=range(imageSize[0])
        else:
            imgData=imgData[xList]
        if type(yList)==type(None):
            yList=range(imageSize[1])
        else:
            imgData=imgData[:,yList]
        if self.coef.shape[-1]>2:
            if type(zList)==type(None):
                zList=range(imageSize[2])
            else:
                imgData=imgData[:,:,zList]
        
        imgDimlen={'x':spacing[0],'y':spacing[1],'f':1,'u':1}
        if self.coef.shape[-1]>2:
            imgDimlen['z']=spacing[2]

        for xn in range(len(xList)):
            logger.info('    {0:.3f}% completed...'.format(float(xn)/len(xList)*100.))
            for yn in range(len(yList)):
                if self.coef.shape[-1]>2:
                    for zn in range(len(zList)):
                        imgData[xn,yn,zn]=self.getRefCoef([xList[xn]*imgDimlen['x'],yList[yn]*imgDimlen['y'],zList[zn]*imgDimlen['z']])
                else:
                    imgData[xn,yn]=self.getRefCoef([xList[xn]*imgDimlen['x'],yList[yn]*imgDimlen['y']])
        return (imgData,imgDimlen)
      
class Affine:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,coefFile=None,fileScale=1.,delimiter=' ',correctOrigin=True):
        '''
        Initialize all data.
        Note:  coef=np.ndarray(4,4) homogenous transformation
        '''
        self.coef=None
        if type(coefFile)!=type(None):
            self.read(coefFile=coefFile,fileScale=fileScale,delimiter=delimiter,correctOrigin=correctOrigin)
        if type(self.coef)==np.ndarray:
          logger.info('shape= '+str(self.coef.shape))
    def read(self,coefFile=None,fileScale=1.,delimiter=' ',correctOrigin=True):
        if type(coefFile)!=type(None):
            try:
                self.coef=np.loadtxt(coefFile,delimiter=delimiter).reshape((4,4), order='F')
                logger.info('Loading '+coefFile)
            except:
                pass
            if type(self.coef)==type(None):
                try:
                    self.coef=coefFile.copy()
                except:
                    pass
            if type(self.coef)==type(None):
                logger.info('Loading '+str(coefFile))
                with open (coefFile, "r") as myfile:
                    data=myfile.readlines()
                for string in data:
                    result = re.search('\(Transform (.*)\)', string)
                    if result:
                        if 'EulerTransform' in result.group(1):
                            filetype='EulerTransform'
                if filetype=='EulerTransform':#(0)center to rotation, (1) rotate, (2) correct center and translate
                    rotateCenter=None
                    if correctOrigin:
                        origin=None
                    else:
                        origin=np.array([0.,0.,0.])
                    transPara=None
                    for string in data:
                        if type(rotateCenter)==type(None):
                            result = re.search('\(CenterOfRotationPoint (.*)\)', string)
                        elif type(origin)==type(None):
                            result = re.search('\(Origin (.*)\)', string)
                        else:
                            result = re.search('\(TransformParameters (.*)\)', string)
                        if result:
                            if type(rotateCenter)==type(None):
                               rotateCenter=np.fromstring(result.group(1), sep=' ')
                            elif type(origin)==type(None):
                               origin=np.fromstring(result.group(1), sep=' ')
                            else:
                                transPara=np.fromstring(result.group(1), sep=' ')
                                break
                    if fileScale!=1.:
                        transPara[3:6]=transPara[3:6]/fileScale
                        rotateCenter=rotateCenter/fileScale
                        origin=origin/fileScale
                    matrixList=[np.array([[1.,0.,0.,-rotateCenter[0]+origin[0]],[0.,1.,0.,-rotateCenter[1]+origin[1]],[0.,0.,1.,-rotateCenter[2]+origin[2]],[0.,0.,0.,1.]])]
                    for n in range(3):
                        tempMatrix=np.zeros((3,3))
                        tempMatrix[n,n]=1.
                        tempMatrix[n-1,n-1]=np.cos(transPara[n])
                        tempMatrix[n-2,n-1]=-np.sin(transPara[n])
                        tempMatrix[n-1,n-2]=np.sin(transPara[n])
                        tempMatrix[n-2,n-2]=np.cos(transPara[n])
                        matrixList.append(np.zeros((4,4)))
                        matrixList[-1][3,3]=1.
                        matrixList[-1][:3,:3]=tempMatrix.copy()
                    self.coef=matrixList[3]@matrixList[1]@matrixList[2]@matrixList[0]
                    self.coef[0:3,3]+=transPara[3:6]+rotateCenter-origin
    def save(self,file):
        with open(file, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
    def writeCoef(self,filepath,delimiter=' '):
        ''' 
        Write coef in a single-line in Fortran format
        Parameters:
            filePath:file,str
                File or filename to save to
            delimiter:str, optional
                separation between values
        '''
        np.savetxt(filepath,self.coef.reshape(-1, order='F'),delimiter=delimiter)
    def getVector(self,coordsList):
        ''' 
        Returns vector corresponding to the coordinates
        Parameters:
            coordsList=[[x,y,z],] or [x,y,z]:list,np.ndarray
                Coordinate or list of coordinates
        Return:
            resultVectors=[u,v,w] or [[u,v,w],]:np.ndarray or list[np.ndarray]
                vector or list of vectors corresponding to the input coordinate
        '''
        singleInput=False
        if not(type(coordsList[0]) in [np.ndarray,list]):
            coordsList=[coordsList]
            singleInput=True
        resultVectors=[]
        for n in range(len(coordsList)):
            homoCoord=np.zeros((4,1))
            homoCoord[3,0]=1.
            homoCoord[:3,0]=coordsList[n][:3].copy()
            homoCoord=self.coef@homoCoord
            resultTemp=coordsList[n].copy()
            resultTemp[0:3]=homoCoord[:3,0]
            resultVectors.append(resultTemp.copy())
        if singleInput:
            resultVectors=resultVectors[0]
        return resultVectors

class CompositeTransform:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,coefFile=None,fileScale=1.,delimiter=' ',correctOrigin=True):
        '''
        Initialize all data.
        Note:  coef=np.ndarray(4,4) homogenous transformation
        '''
        self.transform=[]
        if type(coefFile)!=type(None):
            self.read(coefFile=coefFile,fileScale=fileScale,delimiter=delimiter,correctOrigin=correctOrigin)
    def addTransform(self,transform,ind=None):
        if type(ind)==type(None):
            self.transform.append(transform)
        else:
            while len(self.transform)<ind:
                self.transform.append(None)
            self.transform[ind]=transform
                
    def read(self,coefFile=None,fileScale=1.,delimiter=' ',correctOrigin=True):
        #NOT IMPLEMENTED YET
        if type(coefFile)!=type(None):
            try:
                self.coef=np.loadtxt(coefFile,delimiter=delimiter).reshape((4,4), order='F')
                logger.info('Loading '+coefFile)
            except:
                pass
            if type(self.coef)==type(None):
                try:
                    self.coef=coefFile.copy()
                except:
                    pass
            if type(self.coef)==type(None):
                logger.info('Loading '+str(coefFile))
                with open (coefFile, "r") as myfile:
                    data=myfile.readlines()
                for string in data:
                    result = re.search('\(Transform (.*)\)', string)
                    if result:
                        if 'EulerTransform' in result.group(1):
                            filetype='EulerTransform'
                if filetype=='EulerTransform':#(0)center to rotation, (1) rotate, (2) correct center and translate
                    rotateCenter=None
                    if correctOrigin:
                        origin=None
                    else:
                        origin=np.array([0.,0.,0.])
                    transPara=None
                    for string in data:
                        if type(rotateCenter)==type(None):
                            result = re.search('\(CenterOfRotationPoint (.*)\)', string)
                        elif type(origin)==type(None):
                            result = re.search('\(Origin (.*)\)', string)
                        else:
                            result = re.search('\(TransformParameters (.*)\)', string)
                        if result:
                            if type(rotateCenter)==type(None):
                               rotateCenter=np.fromstring(result.group(1), sep=' ')
                            elif type(origin)==type(None):
                               origin=np.fromstring(result.group(1), sep=' ')
                            else:
                                transPara=np.fromstring(result.group(1), sep=' ')
                                break
                    if fileScale!=1.:
                        transPara[3:6]=transPara[3:6]/fileScale
                        rotateCenter=rotateCenter/fileScale
                        origin=origin/fileScale
                    matrixList=[np.array([[1.,0.,0.,-rotateCenter[0]+origin[0]],[0.,1.,0.,-rotateCenter[1]+origin[1]],[0.,0.,1.,-rotateCenter[2]+origin[2]],[0.,0.,0.,1.]])]
                    for n in range(3):
                        tempMatrix=np.zeros((3,3))
                        tempMatrix[n,n]=1.
                        tempMatrix[n-1,n-1]=np.cos(transPara[n])
                        tempMatrix[n-2,n-1]=-np.sin(transPara[n])
                        tempMatrix[n-1,n-2]=np.sin(transPara[n])
                        tempMatrix[n-2,n-2]=np.cos(transPara[n])
                        matrixList.append(np.zeros((4,4)))
                        matrixList[-1][3,3]=1.
                        matrixList[-1][:3,:3]=tempMatrix.copy()
                    self.coef=matrixList[3]@matrixList[1]@matrixList[2]@matrixList[0]
                    self.coef[0:3,3]+=transPara[3:6]+rotateCenter-origin
    def save(self,file):
        with open(file, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
    def writeCoef(self,filepath,delimiter=' '):
        #NOT IMPLEMENTED YET
        ''' 
        Write coef in a single-line in Fortran format
        Parameters:
            filePath:file,str
                File or filename to save to
            delimiter:str, optional
                separation between values
        '''
        np.savetxt(filepath,self.coef.reshape(-1, order='F'),delimiter=delimiter)
    def getVector(self,coordsList):
        ''' 
        Returns vector corresponding to the coordinates
        Parameters:
            coordsList=[[x,y,z],] or [x,y,z]:list,np.ndarray
                Coordinate or list of coordinates
        Return:
            resultVectors=[u,v,w] or [[u,v,w],]:np.ndarray or list[np.ndarray]
                vector or list of vectors corresponding to the input coordinate
        '''
        resultVectors=coordsList
        for n in range(len(self.transform)):
            if type(self.transform[n])!=None:
                resultVectors=self.transform[n].getVector(resultVectors)
        return resultVectors


class FourierSeries:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,coef,basefrequency,fourierFormat='fccss'):
        ''' 
        Parameters:
            coef: 1d np.ndarray
                if coef is int, it initialize the Fourier series to 0 with terms=coef
            basefrequency:int,float
                base frequency
        '''
        if type(coef)==int:
            self.constant=0.
            self.terms=coef
            self.cosine=np.zeros(coef)
            self.sine=np.zeros(coef)
        else:
            self.constant=coef[0]
            self.terms=int(len(coef)/2)
            self.cosine=np.array(coef)[1:(self.terms+1)].copy()
            self.sine=np.array(coef)[(self.terms+1):].copy()
        self.freq=basefrequency
    def __str__(self):
        return "FourierSeries with base frequency"+str(self.freq)+"\n    constant: "+str(self.constant)+"\n    cosine terms: "+str(self.cosine)+"\n    sine terms: "+str(self.sine)
    def copy(self):
        return FourierSeries(np.concatenate((np.array([self.constant]),self.cosine.copy(),self.sine.copy())),self.freq)
    def getFSElement(self,element):
        coef=np.concatenate((np.array([self.constant]),self.cosine.copy(),self.sine.copy()))
        return coef[element]
          
    def __pow__(self, val):
        if type(val) not in [int,float]:
            raise Exception('Error: FourierSeries can only be raise to a non-negative interger.')
        elif (val%1)>rndError or val<-rndError:
            raise Exception('Error: FourierSeries can only be raise to a non-negative interger.')
        val=int(np.around(val))
        if val==0:
            return 1.
        elif val==1:
            return self.copy()
        return self*self**(val-1)
    def __rpow__(self, val):
        raise Exception('Error: Raising to FourierSeries not supported.')
    def __add__(self, val):
        if type(val)==type(self):
            if self.freq==val.freq:
                result=FourierSeries(max(self.terms,val.terms),self.freq)
                result.constant=self.constant+val.constant
                result.cosine[:self.terms]+=self.cosine
                result.sine[:self.terms]+=self.sine
                result.cosine[:val.terms]+=val.cosine
                result.sine[:val.terms]+=val.sine
            else:
                Exception('Error: operand of FourierSeries with different base frequency not supported.')
        elif type(val)==CoefficientMatrix:
            result=val.__radd__(self)
        else:
            try:
                result=self.copy()
                result.constant+=val
            except:
                result=val.__radd__(self)
        return result
    def __radd__(self, val):
        return self.__add__(val)
    def __sub__(self, val):
        if type(val)==type(self):
            if self.freq==val.freq:
                result=FourierSeries(max(self.terms,val.terms),self.freq)
                result.constant=self.constant-val.constant
                result.cosine[:self.terms]+=self.cosine
                result.sine[:self.terms]+=self.sine
                result.cosine[:val.terms]-=val.cosine
                result.sine[:val.terms]-=val.sine
            else:
                Exception('Error: operand of FourierSeries with different base frequency not supported.')
        elif type(val)==CoefficientMatrix:
            result=val.__rsub__(self)
        else:
            try:
                result=self.copy()
                result.constant-=val
            except:
                result=val.__rsub__(self)
        return result
    def __rsub__(self,val):
        if type(val)==type(self):
            if self.freq==val.freq:
                result=val.__sub__(self)
            else:
                Exception('Error: operand of FourierSeries with different base frequency not supported.')
        elif type(val)==CoefficientMatrix:
            result=val.__sub__(self)
        else:
            try:
                result=self.copy()
                result.constant=val-self.constant
                result.cosine=-self.cosine
                result.sine=-self.sine
            except:
                result=val.__sub__(self)
        return result
    def __mul__(self, val):
        if type(val)==type(self):
            if self.freq==val.freq:
                result=FourierSeries(self.terms+val.terms,self.freq)
                result.constant+=self.constant*val.constant
                result.cosine[:self.terms]+=val.constant*self.cosine
                result.cosine[:val.terms]+=self.constant*val.cosine
                result.sine[:self.terms]+=val.constant*self.sine
                result.sine[:val.terms]+=self.constant*val.sine
                for n in range(self.terms):
                    for m in range(val.terms):
                        if n==m:
                            result.constant+=0.5*self.sine[n]*val.sine[m]+0.5*self.cosine[n]*val.cosine[m]
                        else:
                            if n>m:
                                togger=1
                            else:
                                togger=-1
                            result.cosine[togger*(n-m)-1]+=0.5*self.sine[n]*val.sine[m]+0.5*self.cosine[n]*val.cosine[m]+0.5*self.sine[n]*val.sine[m]
                            result.sine[togger*(n-m)-1]+=togger*0.5*self.sine[n]*val.cosine[m]-togger*0.5*self.cosine[n]*val.cosine[m]+0.5*self.sine[n]*val.sine[m]
                        result.cosine[n+m+1]+=-0.5*self.sine[n]*val.sine[m]+0.5*self.cosine[n]*val.cosine[m]
                        result.sine[n+m+1]+=0.5*self.cosine[n]*val.sine[m]+0.5*self.sine[n]*val.cosine[m]
            else:
                Exception('Error: operand of FourierSeries with different base frequency not supported.')
        elif type(val)==CoefficientMatrix:
            result=val.__rmul__(self)
        else:
            try:
                result=self.copy()
                result.constant*=val
                result.cosine*=val
                result.sine*=val
            except:
                result=val.__rmul__(self)
        return result
    def __rmul__(self, val):
        return self.__mul__(val)
    def __truediv__(self, val):
        if type(val)==type(self):
            raise Exception('Error: Division by FourierSeries not supported.')
        elif type(val)==CoefficientMatrix:
            result=val.__rtruediv__(self)
        else:
            result=self.copy()
            result.constant=result.constant/val
            result.cosine=result.cosine/val
            result.sine=result.sine/val
            return result
    def __rtruediv__(self, val):
        raise Exception('Error: Division by FourierSeries not supported.')
    def __neg__(self):
        return self.__mul__(-1)
    def differentiate_t(self,val):
        result=self.copy()
        if val==0:
            return result
        if val%4==0:
            tempVal=[False,1.,1.] #swap, cos multiplier, sin multiplier
        elif val%4==1:
            tempVal=[True,1.,-1.]
        elif val%4==2:
            tempVal=[False,-1.,-1.]
        elif val%4==3:
            tempVal=[True,-1.,1.]
        freq_rad=(2.*np.pi*self.freq)**val
        if tempVal[0]:
            result=FourierSeries(np.concatenate((np.array([0]),tempVal[1]*freq_rad*self.sine,tempVal[2]*freq_rad*self.cosine)),self.freq)
        else:
            result=FourierSeries(np.concatenate((np.array([0]),tempVal[1]*freq_rad*self.cosine,tempVal[2]*freq_rad*self.sine)),self.freq)
        return result
    def __call__(self,t):
        freq_rad=2.*np.pi*self.freq
        result=self.constant
        for n in range(len(self.cosine)):
            result+=self.cosine[n]*np.cos(freq_rad*(n+1)*t)
        for n in range(len(self.sine)):
            result+=self.sine[n]*np.sin(freq_rad*(n+1)*t)
        return result
    def integratePeriodAverage(self):
        return self.constant
class CoefficientMatrix:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,fourierTerms=7,dCList=None,CIndList=None,basefreq=None,tVal=None,tDifferentiate=0,tRef=None,vec=None):
        ''' 
        Parameters:
            fourierSeriesDict: dict
                dict of fourier series with fourierSeriesDict[uvw][index of C]=FourierSeries
        '''
        self.fourierTerms=fourierTerms
        self.constant=0.
        self.fsDict=[]
        for n in range(3):
            self.fsDict.append([])
            for m in range(fourierTerms):
                self.fsDict[-1].append({})
        if type(dCList)!=type(None) and type(CIndList)!=type(None):
            if type(vec)==type(None):
                vec=[0,1,2]
            else:
                vec=[vec]
            if tDifferentiate%4==0:
                altcos=lambda a : (2.*np.pi*basefreq)**tDifferentiate*np.cos(a)
                altsin=lambda a : (2.*np.pi*basefreq)**tDifferentiate*np.sin(a)
            elif tDifferentiate%4==1:
                altcos=lambda a : -(2.*np.pi*basefreq)**tDifferentiate*np.sin(a)
                altsin=lambda a : (2.*np.pi*basefreq)**tDifferentiate*np.cos(a)
            elif tDifferentiate%4==2:
                altcos=lambda a : -(2.*np.pi*basefreq)**tDifferentiate*np.cos(a)
                altsin=lambda a : -(2.*np.pi*basefreq)**tDifferentiate*np.sin(a)
            elif tDifferentiate%4==3:
                altcos=lambda a : (2.*np.pi*basefreq)**tDifferentiate*np.sin(a)
                altsin=lambda a : -(2.*np.pi*basefreq)**tDifferentiate*np.cos(a)
            for n in range(len(dCList)):
                for fourierTerm in range(1,self.fourierTerms):
                    
                    if type(tVal)==type(None):
                        if fourierTerm==0 or type(tRef)==type(None):
                            factorRef=0.
                        elif (fourierTerm*2)>self.fourierTerms:
                            factorRef=(fourierTerm-int(self.fourierTerms/2))**tDifferentiate*altsin(2.*np.pi*basefreq*tRef*(fourierTerm-int(self.fourierTerms/2)))
                        else:
                            factorRef=fourierTerm**tDifferentiate*altcos(2.*np.pi*basefreq*tRef*fourierTerm)
                        tempInput=np.zeros(self.fourierTerms)
                        tempInput[fourierTerm]=dCList[n]
                        for axis in vec:
                            self.fsDict[axis][fourierTerm][CIndList[n]]=FourierSeries(tempInput,basefreq).differentiate_t(tDifferentiate)-factorRef*dCList[n]
                    else:
                        if fourierTerm==0:
                            if tDifferentiate>0:
                                factor=0.
                            else:
                                factor=1.
                        elif (fourierTerm*2)>self.fourierTerms:
                            factor=(fourierTerm-int(self.fourierTerms/2))**tDifferentiate*altsin(2.*np.pi*basefreq*tVal*(fourierTerm-int(self.fourierTerms/2)))
                        else:
                            factor=fourierTerm**tDifferentiate*altcos(2.*np.pi*basefreq*tVal*fourierTerm)
                        if fourierTerm==0 or type(tRef)==type(None):
                            factorRef=0.
                        elif (fourierTerm*2)>self.fourierTerms:
                            factorRef=(fourierTerm-int(self.fourierTerms/2))**tDifferentiate*altsin(2.*np.pi*basefreq*tRef*(fourierTerm-int(self.fourierTerms/2)))
                        else:
                            factorRef=fourierTerm**tDifferentiate*altcos(2.*np.pi*basefreq*tRef*fourierTerm)
                        for axis in vec:
                            self.fsDict[axis][fourierTerm][CIndList[n]]=dCList[n]*factor-factorRef*dCList[n]
    def copy(self,fourierTerms=1):
        new=CoefficientMatrix(max(fourierTerms,self.fourierTerms))
        try:
            new.constant=self.constant.copy()
        except:
            new.constant=self.constant
        for axis in range(3):
            for fourierTerm in range(self.fourierTerms):
                for key in self.fsDict[axis][fourierTerm]:
                    try:
                        new.fsDict[axis][fourierTerm][key]=self.fsDict[axis][fourierTerm][key].copy()
                    except:
                        new.fsDict[axis][fourierTerm][key]=self.fsDict[axis][fourierTerm][key]
        return new
    def getFSElement(self,element):
        result=self.copy()
        for axis in range(3):
            for fourierTerm in range(result.fourierTerms):
                if fourierTerm!=element:
                    result.fsDict[axis][fourierTerm]={}
                else:
                    for key in result.fsDict[axis][fourierTerm]:
                        try:
                            result.fsDict[axis][fourierTerm][key]=result.fsDict[axis][fourierTerm][key].getFSElement(element)
                        except:
                            pass

        return result
                
    def __bool__(self):
        if self.fsDict[0] or self.fsDict[1] or self.fsDict[2]:
            return True
        else:
            return False
    def differentiate_t(val):
        result=self.copy()
        if result.constant in [int,float,complex]:
            result.constant=0.
        else:
            result.constant=self.constant.differentiate_t(val)
        for axis in range(3):
            for fourierTerm in range(self.fourierTerms):
                for key in result.fsDict[axis][fourierTerm]:
                    if result.fsDict[axis][fourierTerm][key] in [int,float,complex]:
                        del result.fsDict[axis][fourierTerm][key]
                    else:
                        result.fsDict[axis][fourierTerm][key]=result.fsDict[axis][fourierTerm][key].differentiate_t(val)
        return result
    def integratePeriodAverage(self):
        result=self.copy()
        if type(result.constant) in [FourierSeries]:
            result.constant=self.constant.integratePeriodAverage()
        for axis in range(3):
            for fourierTerm in range(self.fourierTerms):
                for key in result.fsDict[axis][fourierTerm]:
                    if type(result.fsDict[axis][fourierTerm][key]) in [FourierSeries]:
                        result.fsDict[axis][fourierTerm][key]=self.fsDict[axis][fourierTerm][key].integratePeriodAverage()
        return result
    def toSparseMatrix(self,column,skip0fourier=1):
        indarray=[]
        valarray=[]
        for axis in range(3):
            for fourierTerm in range(skip0fourier,self.fourierTerms):
                for key in self.fsDict[axis][fourierTerm]:
                    indarray.append(key*(self.fourierTerms-skip0fourier)*3+(fourierTerm-skip0fourier)*3+axis)
                    valarray.append(self.fsDict[axis][fourierTerm][key])
        matrow=sparse.csr_matrix((np.array(valarray),(np.zeros(len(indarray)),np.array(indarray))),shape=(1,column))
        natrow=self.constant
        return (matrow,natrow)
    def __pow__(self, val):
        
        if type(val) not in [int,float]:
            raise Exception('Error: Please check Equation. You do not need to raise CoefficientMatrix.')
        elif int(np.around(val))==0:
            return 1.
        elif int(np.around(val))==1:
            return self.copy()
        else:
            raise Exception('Error: Please check Equation. You do not need to raise CoefficientMatrix.')
    def __rpow__(self, val):
        raise Exception('Error: Please check Equation. You do not need to raise CoefficientMatrix.')
    def __add__(self, val):
        if type(val)==type(self):
            result=self.copy(max(self.fourierTerms,val.fourierTerms))
            result.constant=self.constant+val.constant
            for axis in range(3):
                for fourierTerm in range(val.fourierTerms):
                    for key in val.fsDict[axis][fourierTerm]:
                        if key in result.fsDict[axis][fourierTerm]:
                            result.fsDict[axis][fourierTerm][key]=result.fsDict[axis][fourierTerm][key]+val.fsDict[axis][fourierTerm][key]
                        else:
                            try:
                                result.fsDict[axis][fourierTerm][key]=val.fsDict[axis][fourierTerm][key].copy()
                            except:
                                result.fsDict[axis][fourierTerm][key]=val.fsDict[axis][fourierTerm][key]
        else:
            result=self.copy()
            result.constant+=val  
        return result
    def __radd__(self, val):
        return self.__add__(val)
    def __sub__(self, val):
        if type(val)==type(self):
            result=self.copy(max(self.fourierTerms,val.fourierTerms))
            result.constant=self.constant-val.constant
            for axis in range(3):
                for fourierTerm in range(val.fourierTerms):
                    for key in val.fsDict[axis][fourierTerm]:
                        if key in result.fsDict[axis][fourierTerm]:
                            result.fsDict[axis][fourierTerm][key]=result.fsDict[axis][fourierTerm][key]-val.fsDict[axis][fourierTerm][key]
                        else:
                            try:
                                result.fsDict[axis][fourierTerm][key]=-val.fsDict[axis][fourierTerm][key].copy()
                            except:
                                result.fsDict[axis][fourierTerm][key]=-val.fsDict[axis][fourierTerm][key]
        else:
            result=self.copy()
            result.constant-=val    
        return result
    def __rsub__(self,val):
        if type(val)==type(self):
            result=val.__sub__(self)
        else:
            result=self.copy()
            result.constant=val-result.constant
        return result
    def __mul__(self, val):
        if type(val)==type(self):
            raise Exception('Error: Please check Equation. You do not need to add a FourierSeries to CoefficientMatrix.')
        else:
            result=self.copy()
            result.constant*=val
            for axis in range(3):
                for fourierTerm in range(result.fourierTerms):
                    for key in result.fsDict[axis][fourierTerm]:
                        result.fsDict[axis][fourierTerm][key]=result.fsDict[axis][fourierTerm][key]*val
        
        return result
    def __rmul__(self, val):
        return self.__mul__(val)
    def __truediv__(self, val):
        if type(val)==type(self):
            raise Exception('Error: Please check Equation. You do not need to add a FourierSeries to CoefficientMatrix.')
        else:
            result=self.copy()
            result.constant=result.constant/val
            for axis in range(3):
                for fourierTerm in range(result.fourierTerms):
                    for key in result.fsDict[axis][fourierTerm]:
                        result.fsDict[axis][fourierTerm][key]=result.fsDict[axis][fourierTerm][key]/val
        return result
    def __rtruediv__(self, val):
        raise Exception('Error: Division by FourierSeries not supported.')
    def __neg__(self):
        return self.__mul__(-1)
    
class BsplineFunctionofBsplineFourierAD(ad.AD):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,name,bspline,bsplineFourier,tRef=None,variableIdentifier=''):
        if not(name in ['u','v','w']):
            raise Exception('Please choose "u","v" or "w"')
        self.name=name
        self.variableIdentifier=variableIdentifier
        self.bspline=bspline
        self.bsplineFourier=bsplineFourier
        self.dependent=[name+variableIdentifier,'C'+variableIdentifier,'x'+variableIdentifier,'y'+variableIdentifier,'z'+variableIdentifier,'t'+variableIdentifier]
        if name=='u':
            self.axis=0
        elif name=='v':
            self.axis=1
        elif name=='w':
            self.axis=2
        else:
            self.axis=None
        self.tRef=tRef
    def __call__(self,x,dOrder={}):
        if 'ALL' not in self.dependent:
            for var in dOrder:
                if dOrder[var]>0 and (var not in self.dependent):
                    self.debugPrint(x,dOrder,0.)
                    return 0.
        diffC=False
        for key in dOrder:
            if key=='C'+self.variableIdentifier and dOrder[key]==1:
                diffC=True
            elif dOrder[key]>0:
                raise Exception('Only BsplineFunctionofBsplineFourierAD differentiated up to "C":1 supported.')            
        coord=[]
        for var in ['x','y','z']:
            coord+=[x[var+self.variableIdentifier]]
        coord+=[self.bspline.timeMap[0]]
        X=self.bsplineFourier.getCoordFromRef(coord)

        if diffC:
            dCList,CIndList=self.bsplineFourier.getdC(coord)
            freq=1./self.bsplineFourier.spacing[3]
            result=0.
            for n in range(3):
                dxyzt=[0,0,0,0]
                dxyzt[n]=1
                vector=self.bspline.getVector(X,vec=self.axis,dxyzt=dxyzt)
                if self.axis==n:
                    vector+=1.
                result+=vector*CoefficientMatrix(fourierTerms=self.bsplineFourier.coef.shape[3],dCList=dCList,CIndList=CIndList,tVal=coord[3],tRef=self.tRef,basefreq=freq,vec=n)
            self.debugPrint(x,dOrder,result)
            return result
        else:
            vector=self.bspline.getVector(X)
            Y=X+vector
            self.debugPrint(x,dOrder,Y[self.axis])
            return Y[self.axis]

class bsFourierAD(ad.AD):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,name,bsFourier,tVal=None,variableIdentifier='',tRef=None):
        self.name=name
        self.variableIdentifier=variableIdentifier
        self.bsFourier=bsFourier
        self.dependent=[name+variableIdentifier,'C'+variableIdentifier,'x'+variableIdentifier,'y'+variableIdentifier,'z'+variableIdentifier,'t'+variableIdentifier]
        self.freq=1./self.bsFourier.spacing[self.bsFourier.coef.shape[-1]]
        if name=='u':
            self.axis=0
        elif name=='v':
            self.axis=1
        elif name=='w':
            self.axis=2
        else:
            self.axis=None
        self.tVal=tVal
        self.tRef=tRef
    def __call__(self,x,dOrder):
        if 'ALL' not in self.dependent:
            for var in dOrder:
                if dOrder[var]>0 and (var not in self.dependent):
                    self.debugPrint(x,dOrder,0.)
                    return 0.
        if 'C' in dOrder:
            if dOrder['C']>1:
                return 0.
        coord=[]
        for var in ['x','y','z']:
            if (var+self.variableIdentifier) in x:
                coord+=[x[var+self.variableIdentifier]]
        tVal=None
        if type(self.tVal)!=type(None):
            coord+=[self.tVal]
            tVal=self.tVal
        elif 't'+self.variableIdentifier in x:
            coord+=[x['t'+self.variableIdentifier]]
            tVal=x['t'+self.variableIdentifier]
        dxyzt=np.zeros(self.bsFourier.coef.shape[-1]+1)
        if 'x'+self.variableIdentifier in dOrder:
            dxyzt[0]=dOrder['x'+self.variableIdentifier]
        if 'y'+self.variableIdentifier in dOrder:
            dxyzt[1]=dOrder['y'+self.variableIdentifier]
        if self.bsFourier.coef.shape[-1]>2:
            if 'z'+self.variableIdentifier in dOrder:
                dxyzt[2]=dOrder['z'+self.variableIdentifier]
        if 't'+self.variableIdentifier in dOrder:
            dxyzt[-1]=dOrder['t'+self.variableIdentifier]
        result=None
        if 'C'+self.variableIdentifier in dOrder:
            if dOrder['C'+self.variableIdentifier]==1:
                dCList,CIndList=self.bsFourier.getdC(coord,dxyz=dxyzt[:self.bsFourier.coef.shape[-1]])
                result=CoefficientMatrix(fourierTerms=self.bsFourier.coef.shape[self.bsFourier.coef.shape[-1]],dCList=dCList,CIndList=CIndList,tVal=tVal,tDifferentiate=dxyzt[self.bsFourier.coef.shape[-1]],tRef=self.tRef,basefreq=self.freq,vec=self.axis)
        if type(result)==type(None):
            if type(self.bsFourier)==BsplineFourier:
                result=self.bsFourier.getCoordFromRef(coord,vec=self.axis,dxyzt=dxyzt)
            elif type(self.bsFourier)==Bspline:
                result=self.bsFourier.getVector(coord,vec=self.axis,dxyzt=dxyzt)
            if type(result)==np.ndarray and type(self.axis)!=type(None):
                result=FourierSeries(result,self.freq)
        self.debugPrint(x,dOrder,result)
        return result
class periodAvgIntegral(ad.AD):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,func):
        self.func=func
        self.dependent=func.dependent[:]
    def __call__(self,x,dOrder):
        if 't' in dOrder:
            if dOrder['t']>1:
                return 0.
        if 'ALL' not in self.dependent:
            for var in dOrder:
                if dOrder[var]>0 and (var not in self.dependent):
                    self.debugPrint(x,dOrder,0.)
                    return 0.
        result=self.func(x,dOrder)
        if type(result) in [CoefficientMatrix,FourierSeries]:
            result=result.integratePeriodAverage()
        self.debugPrint(x,dOrder,result)
        return result
class getFSElementAD(ad.AD):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,func,element):
        self.func=func
        self.element=element
        self.dependent=func.dependent[:]
    def __call__(self,x,dOrder):
        if 'ALL' not in self.dependent:
            for var in dOrder:
                if dOrder[var]>0 and (var not in self.dependent):
                    self.debugPrint(x,dOrder,0.)
                    return 0.
        return self.func(x,dOrder).getFSElement(self.element)
class toSparseMatrixAD(ad.AD):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,func,column):
        self.func=func
        self.column=column
        self.dependent=func.dependent[:]
    def __call__(self,x,dOrder):
        if 'ALL' not in self.dependent:
            for var in dOrder:
                if dOrder[var]>0 and (var not in self.dependent):
                    self.debugPrint(x,dOrder,0.)
                    return 0.
        return self.func(x,dOrder).toSparseMatrix(self.column)
class BspreadArray():
  def __new__(cls, *args, **kwargs):
      return super().__new__(cls)
  def __init__(self,imageSize,imageSpacing,gridSpacing,accuracy=1):
      self.imageSize=np.array(imageSize).astype(int)
      self.accuracy=int(accuracy)
      self.twoD=len(imageSize)
      self.spacing=np.array(imageSpacing)/self.accuracy
      self.numPerGrid=gridSpacing/self.spacing
      self.origin=np.ceil(self.numPerGrid*2+self.accuracy).astype(int)
      self.data={}
  def getbspread(self,nodeImageCoord,key):
        #imageSize=None,spacing=None,scaleFromGrid=None,tVal=None,coefFourierWeight=None,vec=None,dxyzt=None,accuracy=1):
        ''' 
        Create an image based on the amplitude of fourier coefficients
        Parameters:
            imageSize=[x,y,z]:list,np.ndarray
                image pixel size
            coefFourierWeight=[fourierterm1,fourierterm2]:list,np.ndarray
                weighs the courier terms of different frequencies cosine and sine are considered as a single term
            scaleFromGrid: float of np.ndarray(3)
                scale image size from bspline grid if image size is not given
        Return:
            imgData:np.ndarray
                image intensity data
            imgDimlen: dict
                conversion of image pixel to real coordinates
        Note: origin is conserved
        '''
        if key[0]!='d':
            raise Exception('key error.',key)
        elif key in self.data:
            bspread=self.data[key].copy()
        else:
            dxyz=[]
            for n in range(1,len(key)):
                dxyz.append(int(key[n]))
            if np.any(np.array(dxyz)>3):
                return None
            #get standard bspline spread
            bspread=np.zeros(self.origin*2)
            gridList=[]
            for n in range(len(self.origin)):
                gridList.append(slice(self.origin[n]*2))
            bcoord=(np.mgrid[tuple(gridList)].reshape(len(gridList),*(self.origin*2)).transpose(*tuple(range(1,len(gridList)+1)),0)-self.origin)/self.numPerGrid+2
            uvw=np.remainder(bcoord,1.)
            bcoord[bcoord<0]=-1
            k=3-bcoord.astype(int)
            for axis in range(self.twoD):
                for kn in range(0,4):
                    setIndex=(k[...,axis]==kn)
                    validuvw=uvw[setIndex,axis]
                    if axis==0:
                        bspread[setIndex]=B[kn]({'b':validuvw},{'b':dxyz[0]})
                    else:
                        bspread[setIndex]*=B[kn]({'b':validuvw},{'b':dxyz[axis]})
                for kn in [-1,4]:
                    setIndex=(k[...,axis]==kn)
                    bspread[setIndex]*=0
            self.data[key]=bspread.copy()
        imgIndex=np.around(nodeImageCoord).astype(int)
        adjust=np.around((nodeImageCoord-imgIndex)*self.accuracy).astype(int)
        origin=self.origin+adjust
        minget=np.maximum(-imgIndex,-np.ceil(self.numPerGrid/np.float(self.accuracy)*2).astype(int))
        maxget=np.minimum(self.imageSize-imgIndex,np.ceil(self.numPerGrid/np.float(self.accuracy)*2).astype(int))
        bspreadsliceList=[]
        imgsliceList=[]
        if np.any((maxget-minget)<=0):
            return None
        for n in range(self.twoD):
            bspreadsliceList.append(slice(self.origin[n]+minget[n]*self.accuracy,self.origin[n]+maxget[n]*self.accuracy,self.accuracy))
            imgsliceList.append(slice(imgIndex[n]+minget[n],imgIndex[n]+maxget[n]))
        return (bspread[tuple(bspreadsliceList)],imgsliceList)
class imageVectorAD(ad.AD):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self,name,bSpline,imageSize,imageSpacing,accuracy=1,tVal=None,variableIdentifier='',tRef=None):
        self.name=name
        self.variableIdentifier=variableIdentifier
        self.bSpline=bSpline
        self.imageSize=np.array(imageSize).astype(int)[::-1]
        self.imageSpacing=np.array(imageSpacing)[::-1]
        self.twoD=len(imageSpacing)
        self.accuracy=accuracy
        self.dependent=[name+variableIdentifier,'x'+variableIdentifier,'y'+variableIdentifier,'z'+variableIdentifier,'t'+variableIdentifier]
        if name=='u':
            self.axis=0
        elif name=='v':
            self.axis=1
        elif name=='w':
            self.axis=2
        else:
            raise Exception('Select a valid vector (u,v,w)')
        self.tVal=tVal
        self.tRef=tRef
    def __call__(self,x,dOrder):
        if 'ALL' not in self.dependent:
            for var in dOrder:
                if dOrder[var]>0 and (var not in self.dependent):
                    self.debugPrint(x,dOrder,0.)
                    return 0.
        if 'bspread' not in x:
            x['bspread']=BspreadArray(imageSize=self.imageSize,imageSpacing=self.imageSpacing,gridSpacing=self.bSpline.spacing[:self.twoD],accuracy=self.accuracy)
        tVal=None
        if 't'+self.variableIdentifier in x:
            tVal=x['t'+self.variableIdentifier]
        else:
            tVal=self.tVal
        dxyzt=[]
        dkey=''
        axis=['x','y','z']
        for n in range(self.twoD):
            if axis[n]+self.variableIdentifier in dOrder:
                dxyzt.append(dOrder[axis[n]+self.variableIdentifier])
                dkey+=str(dOrder[axis[n]+self.variableIdentifier])
            else:
                dxyzt.append(0)
                dkey+='0'
        if np.any(np.array(dxyzt)>3):
            return 0
        if 't'+self.variableIdentifier in dOrder:
            dxyzt.append(dOrder['t'+self.variableIdentifier])
            dkey+=str(dOrder['t'+self.variableIdentifier])
        else:
            dxyzt.append(0)
            dkey+='0'
        if 'storedImage' not in x:
            x['storedImage']={}
        elif self.name+self.variableIdentifier+'d'+dkey in x['storedImage']:
            return x['storedImage'][self.name+self.variableIdentifier+'d'+dkey].copy().T
        coef=self.bSpline.coef[...,self.axis].copy()
        if len(coef.shape)>self.twoD:
            if dxyzt[-1]%4==0:
                tempVal=[False,1.,1.] #swap, cos multiplier, sin multiplier
            elif dxyzt[-1]%4==1:
                tempVal=[True,1.,-1.]
            elif dxyzt[-1]%4==2:
                tempVal=[False,-1.,-1.]
            elif dxyzt[-1]%4==3:
                tempVal=[True,-1.,1.]
            if tempVal[0]:
                coeftemp=coef.copy()
                coeftemp[...,1:(int(coef.shape[-1]/2)+1)]=coef[...,-(int(coef.shape[-1]/2)+1):].copy()
                coeftemp[...,-(int(coef.shape[-1]/2)+1):]=coef[...,1:(int(coef.shape[-1]/2)+1)].copy()
                coef=coeftemp.copy()
                del coeftemp
            coef[...,1:(int(coef.shape[-1]/2)+1)]*=tempVal[1]
            coef[...,-(int(coef.shape[-1]/2)+1):]*=tempVal[2]
            if dxyzt[-1]>0:
                coef[...,0]=0
                for m in range(int(coef.shape[-1]/2)):
                    coef[...,m+1]*=((m+1.)*2.*np.pi/self.bspline.spacing[self.twoD])**dxyzt[-1]
                    coef[...,-m-1]*=((m+1.)*2.*np.pi/self.bspline.spacing[self.twoD])**dxyzt[-1]
            if type(tVal)!=type(None):
                fValue=self.bSpline.getdXYdC([tVal],remove0=False)[0]
                coef=np.sum(coef*fValue,axis=-1)
        else:
            if dxyzt[-1]>0:
                return 0
        gridList=[]
        for n in range(self.twoD):
            gridList.append(range(self.bSpline.coef.shape[n]))
        gridIndex=np.array(np.meshgrid(*gridList)).T.astype(int)
        gridCoord=(gridIndex*self.bSpline.spacing[:self.twoD]+self.bSpline.origin[:self.twoD])/self.imageSpacing
        gridCoordshape=gridCoord.shape
        
        gridCoord=gridCoord.reshape((-1,gridCoord.shape[-1]))
        gridIndex=gridIndex.reshape((-1,gridIndex.shape[-1]))

        if len(coef.shape)>self.twoD:
            resultImage=np.zeros((*self.imageSize,*coef[self.twoD:]))
        else:
            resultImage=np.zeros(self.imageSize)
        for n in range(len(gridCoord)):
            bspread=x['bspread'].getbspread(gridCoord[n],'d'+dkey[:-1])
            if type(bspread)==type(None):
                continue
            bspread,imgslice=bspread
            if len(coef.shape)>self.twoD:
                resultImage[tuple(imgslice)]+=bspread.reshape((*bspread.shape,*np.ones(len(coef[tuple(gridIndex[n])].shape),dtype=int)))*coef[tuple(gridIndex[n])].reshape((*np.ones(len(bspread.shape),dtype=int)),*coef[tuple(gridIndex[n])].shape)
            else:
                resultImage[tuple(imgslice)]+=bspread*coef[tuple(gridIndex[n])]
        x['storedImage'][self.name+self.variableIdentifier+'d'+dkey]=resultImage.copy()
        return resultImage.T
        
                        
