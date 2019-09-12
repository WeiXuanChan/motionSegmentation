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
print('BsplineFourier version 1.2.0')

#Optional dependancies
try:
  import pickle
except ImportError:
  pass

import numpy as np
import autoD as ad
import re
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
u_ad=ad.Scalar('u')
B=[(1.-u_ad)**3./6.,(3.*u_ad**3.-6.*u_ad**2.+4.)/6.,(-3.*u_ad**3.+3.*u_ad**2.+3.*u_ad+1)/6.,u_ad**3./6.]
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
        print('Warning: only Bspline with [x,y,z,uvw] accepted')
        self.coef=None
        self.origin=None
        self.spacing=None
        self.timeMap=[None,None]
        self.coordMat =None
        if type(coefFile)!=type(None):
            self.read(coefFile=coefFile,timeMap=timeMap,shape=shape,spacing=spacing,fileScale=fileScale,delimiter=delimiter,origin=origin)
        if type(self.coef)==np.ndarray:
          print('shape=',self.coef.shape)
        print('spacing=',self.spacing)
        print('origin=',self.origin)
        print('timeMap=',self.timeMap)
    def read(self,coefFile=None,timeMap=[None,None],shape=None,spacing=None,fileScale=1.,delimiter=' ',origin=None):
        if type(coefFile)!=type(None):
            if type(shape)!=type(None):
              try:
                  self.coef=np.loadtxt(coefFile,delimiter=delimiter).reshape(shape, order='F')
                  print('Loading',coefFile)
              except:
                  pass
            if type(self.coef)==type(None):
                try:
                    self.coef=coefFile.copy()
                except:
                    pass
            if type(self.coef)==type(None):
                print('Loading',coefFile)
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
                            if shape[-1]!=3:
                                shape=[*shape,3]
                        elif type(spacing)==type(None):
                            spacing=np.fromstring(result.group(1), sep=' ')
                        else:
                            self.coef=np.fromstring(result.group(1), sep=' ').reshape(shape, order='F')
                            break
                if type(self.coef)==type(None):
                    self.coef=np.loadtxt(coefFile,skiprows=3).reshape(shape, order='F')
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
            
            self.coordMat = np.mgrid[self.origin[0]:(self.origin[0]+(shape[0]-0.1)*self.spacing[0]):self.spacing[0], self.origin[1]:(self.origin[1]+(shape[1]-0.1)*self.spacing[1]):self.spacing[1],self.origin[2]:(self.origin[2]+(shape[2]-0.1)*self.spacing[2]):self.spacing[2]].reshape(3,*shape[:3]).transpose(1,2,3,0)
            if type(timeMap[1])==type(None):
                self.timeMap=[0.,timeMap[0]]
            else:
                self.timeMap=timeMap
            if fileScale!=1.:
                self.scale(1./fileScale)
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
            print('Error, check input coordinates.')
            return;
        coordSlice=[]
        matSlice=[]
        returnMatsize=[]
        uvw=[]
        for n in range(3):
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
        coef=np.zeros((*returnMatsize,*self.coef.shape[3:]))
        if np.any(np.array(coef[matSlice].shape)!=np.array(self.coef[coordSlice].shape)):
            print(uvw,matSlice,coordSlice)
        coef[matSlice]=self.coef[coordSlice].copy()

        
        if np.any(uvw<-np.array(self.coef.shape[:3])) or np.any(uvw>(np.array(self.coef.shape[:3])*2.)):
            print('WARNING! Coordinates',coord,'far from active region queried! Grid Coord=',uvw)
        '''
        if np.any(uvw<1) or np.any(uvw>np.array(self.coef.shape[:3])-2):
            coef=self.getExtendedCoef(uvw).copy()
        else:
            coef=self.coef[coordSlice].copy()
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
        
        padNo=max(2-int(uvw.min()),int((uvw-np.array(self.coef.shape[:3])).max())+2)
        if padNo<0:
            padNo=0
        newuvw=uvw+padNo
        tempcoef=np.zeros((*(np.array(self.coef.shape[:3])+padNo*2),*self.coef.shape[3:]))
        tempcoef[padNo:self.coef.shape[0]+padNo,padNo:self.coef.shape[1]+padNo,padNo:self.coef.shape[2]+padNo]=self.coef
        coordSlice=[]
        for n in range(3):
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
            coef,uvw=self.getCoefMat(coordsList[n])
            for m in range(3):
                coeftemp=np.zeros(coef.shape[1:])
                for k in range(coef.shape[0]):
                    coeftemp=coeftemp+coef[k]*B[k]({'u':uvw[m]%1.},{})
                coef=coeftemp.copy()
            resultVectors.append(coef.copy())
        if singleInput:
            resultVectors=resultVectors[0]
        return resultVectors
    def samplePoints(self,spacingDivision=2.,gap=0):
        step=np.array(self.spacing[:3])/spacingDivision
        start=self.coordMat[0,0,0]+step*gap
        end=self.coordMat[-1,-1,-1]-step*gap+step/2.
        sampleCoord=[]
        for k in np.arange(start[0],end[0],step[0]):
            for l in np.arange(start[1],end[1],step[1]):
                for m in np.arange(start[2],end[2],step[2]):
                    sampleCoord.append(np.array([k,l,m]))
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
        for n in range(len(coordsList)):
            coef,uvw=self.getCoefMat(coordsList[n])
            if len(coordsList[n])>3 and len(coef.shape)>4:
                coeftemp=coef[:,:,:,0,:].copy()
                for m in range(int(coef.shape[3]/2)):#sub in t
                    coeftemp=coeftemp+coef[:,:,:,m+1,:]*np.cos((m+1.)*2.*np.pi/self.spacing[3]*(coordsList[n][3]-self.origin[3]))+coef[:,:,:,int(coef.shape[3]/2)+m+1,:]*np.sin((m+1.)*2.*np.pi/self.spacing[3]*(coordsList[n][3]-self.origin[3]))
            else:
                coeftemp=coef.copy()
            storecoef=coeftemp.copy()
            dXcoef=[]
            for l in range(3):
                coef=storecoef.copy()
                for m in range(3):
                    if m==l:
                        diff=1
                    else:
                        diff=0
                    coeftemp=np.zeros(coef.shape[1:])
                    for k in range(coef.shape[0]):
                        coeftemp=coeftemp+coef[k]*B[k]({'u':uvw[m]%1.},{'u':diff})*(1.-diff*(1-1/self.spacing[m]))
                    coef=coeftemp.copy()
                dXcoef.append(coef.copy())
            resultdX.append(np.array(dXcoef))
        if singleInput:
            resultdX=resultdX[0]
        return resultdX
      
    def getdC(self,coordsList,dxyz=[0,0,0]):#ind=[xIndex,yIndex,zIndex]
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
        if not(type(coordsList[0]) in [np.ndarray,list]):
            coordsList=[coordsList]
            singleInput=True
        dCList=[]
        CIndList=[]
        for n in range(len(coordsList)):
            coef,uvw=self.getCoefMat(coordsList[n])
            uvw_x=int(uvw[0])
            uvw_y=int(uvw[1])
            uvw_z=int(uvw[2])
            dC=[]
            CInd=[]   
            for k in range(coef.shape[0]):
                for l in range(coef.shape[1]):
                    for m in range(coef.shape[2]):
                        addInd=self.getCIndex([uvw_x+k-1,uvw_y+l-1,uvw_z+m-1])
                        if addInd>=0:
                            dC.append(B[k]({'u':uvw[0]%1.},{'u':dxyz[0]})*B[l]({'u':uvw[1]%1.},{'u':dxyz[1]})*B[m]({'u':uvw[2]%1.},{'u':dxyz[2]}))
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
        for n in range(3):
            if xyz[n]>=(self.coef.shape[n]):
                return -1
            elif xyz[n]<0:
                return -1
        ind=xyz[0]+xyz[1]*self.coef.shape[0]+xyz[2]*self.coef.shape[0]*self.coef.shape[1]
        return ind
    def getCfromIndex(self,indList):
        ''' 
        Returns the weights of corresponding control points at respective coordinates
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
        size=[self.coef.shape[0]*self.coef.shape[1],self.coef.shape[0],1]
        for n in range(len(indList)):
            ind=indList[n]
            xyz=[0,0,0]
            for m in range(2):
                while ind>=size[m]:
                    ind-=size[m]
                    xyz[2-m]+=1
            xyz[0]=ind
            xyzList.append(tuple(xyz))
        if singleInput:
            xyzList=xyzList[0]
        return xyzList

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
        print('Warning: only Bspline fourier with [x,y,z,fourierTerms,uvw] accepted')
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
            self.coordMat = np.mgrid[self.origin[0]:(self.origin[0]+(shape[0]-0.1)*self.spacing[0]):self.spacing[0], self.origin[1]:(self.origin[1]+(shape[1]-0.1)*self.spacing[1]):self.spacing[1],self.origin[2]:(self.origin[2]+(shape[2]-0.1)*self.spacing[2]):self.spacing[2]].reshape(3,*shape[:3]).transpose(1,2,3,0)
        if type(self.coef)==type(None):
            self.coef=np.zeros(shape)
        print('Origin=',self.origin)
        print('Spacing=',self.spacing)
        print('Fourier Format=',self.fourierFormat)
        self.numCoefXYZ=1
        for n in self.coef.shape[:3]:
            self.numCoefXYZ*=n
        self.numCoef=int(self.numCoefXYZ*(self.coef.shape[3]-1))
        
    def readFile(self,coefFile=None,shape=None,fourierFormat='fccss',spacing=None,delimiter=' ',skiprows=1,origin=None):
        if coefFile!=None:
            self.read(coefFile=coefFile,shape=shape,spacing=spacing,delimiter=delimiter,origin=origin)
            self.fourierFormat=fourierFormat
            #get fourier format to fccss
            self.convertToFCCSS()
            self.initialize(self.coef.shape)
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
        uvw_z=int(uvw[2])
        C=np.zeros(coef.shape[3:])
        CInd=[]
        for k in range(coef.shape[0]):
            for l in range(coef.shape[1]):
                for m in range(coef.shape[2]):
                    C+=coef[k,l,m]*B[k]({'u':uvw[0]%1.},{})*B[l]({'u':uvw[1]%1.},{})*B[m]({'u':uvw[2]%1.},{})
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
        for n in range(len(coordsList)):
            coef,uvw=self.getCoefMat(coordsList[n])
            coeftemp=coef.copy()
            for m in range(int(coef.shape[3]/2)):#sub in t
                coeftemp[:,:,:,m+1,:]=coef[:,:,:,m+1,:]*np.cos((m+1.)*2.*np.pi/self.spacing[3]*(coordsList[n][3]-self.origin[3]))
                coeftemp[:,:,:,int(coef.shape[3]/2)+m+1,:]=coef[:,:,:,int(coef.shape[3]/2)+m+1,:]*np.sin((m+1.)*2.*np.pi/self.spacing[3]*(coordsList[n][3]-self.origin[3]))
            coef=coeftemp.copy()
            for m in range(3):
                coeftemp=np.zeros(coef.shape[1:])
                for k in range(coef.shape[0]):
                    coeftemp=coeftemp+coef[k]*B[k]({'u':uvw[m]%1.},{})
                coef=coeftemp.copy()
            for axis in range(3):
                coef+=np.array(coordsList[n][axis])
            resultFourierCoef.append(coef.copy())
        if singleInput:
            resultFourierCoef=resultFourierCoef[0]
        return resultFourierCoef
    def getCoordFromRef(self,coordsList):
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
            coef,uvw=self.getCoefMat(coordsList[n])
            coeftemp=coef[:,:,:,0,:].copy()
            for m in range(int(coef.shape[3]/2)):#sub in t
                coeftemp=coeftemp+coef[:,:,:,m+1,:]*np.cos((m+1.)*2.*np.pi/self.spacing[3]*(coordsList[n][3]-self.origin[3]))+coef[:,:,:,int(coef.shape[3]/2)+m+1,:]*np.sin((m+1.)*2.*np.pi/self.spacing[3]*(coordsList[n][3]-self.origin[3]))
            coef=coeftemp.copy()
            for m in range(3):
                coeftemp=np.zeros(coef.shape[1:])
                for k in range(coef.shape[0]):
                    coeftemp=coeftemp+coef[k]*B[k]({'u':uvw[m]%1.},{})
                coef=coeftemp.copy()
            resultCoords.append(coef+np.array(coordsList[n][:3]))
        if singleInput:
            resultCoords=resultCoords[0]
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
            error=self.getCoordFromRef(ref)[:3]-coordsList[n][:3]
            maxerror=self.spacing[:3]*maxErrorRatio
            count=0
            while np.any(error>maxerror) and count<maxIteration:
                Jmat=self.getdX(ref)
                Jmat=Jmat+lmLambda*np.diag(np.diag(Jmat))
                dX=np.linalg.solve(Jmat, error)
                newref=ref.copy()
                newref[:3]=newref[:3]+dX
                newError=self.getCoordFromRef(newref)[:3]-coordsList[n][:3]
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
                    print('Maximum iterations reached for point',m,self.points[m])
            resultCoords.append(ref[:3].copy())
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
            for m in range(3):
                for k in range(3):
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
        dX=np.ones(self.coef.shape[3])
        dY=np.ones(self.coef.shape[3])
        for m in range(int(self.coef.shape[3]/2)):#get T matrix
            dX[m+1]=np.cos((m+1.)*2.*np.pi/self.spacing[3]*(timeMap[0]-self.origin[3]))
            dX[int(self.coef.shape[3]/2)+m+1]=np.sin((m+1.)*2.*np.pi/self.spacing[3]*(timeMap[0]-self.origin[3]))
            dY[m+1]=np.cos((m+1.)*2.*np.pi/self.spacing[3]*(timeMap[1]-self.origin[3]))
            dY[int(self.coef.shape[3]/2)+m+1]=np.sin((m+1.)*2.*np.pi/self.spacing[3]*(timeMap[1]-self.origin[3]))
        if remove0:
            dX=dX[1:]
            dY=dY[1:]
        return (dX,dY) 
    def getBspline(self,time):
        ''' 
        Returns the coef of all control points where with time evaluated wrt ref
        Parameters:
            timeMap=[t_start,t_end] :list,np.ndarray
                time map from t_start to t_end
        Return:
            vector:np.ndarray
                vector coef of all control points
        '''
        coef=self.coef[:,:,:,0,:].copy()
        for m in range(int(self.coef.shape[3]/2)):#get T matrix
            coef+=self.coef[:,:,:,m+1,:]*np.cos((m+1.)*2.*np.pi/self.spacing[3]*(time-self.origin[3]))+self.coef[:,:,:,int(self.coef.shape[3]/2)+m+1,:]*np.sin((m+1.)*2.*np.pi/self.spacing[3]*(time-self.origin[3]))
        return coef
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
        coeftemp=self.coef.transpose(3,0,1,2,4)
        newcoef=None
        if fourierRearrange!=None:
            coeftemp[:]=coeftemp[fourierRearrange]
            newcoef=coeftemp.transpose(1,2,3,0,4)
        return newcoef
    def convertfromFRalpha(self):
        outcoef=None
        if self.fourierFormat=='fRalpha':
            ftermsNum=self.coef.shape[-2]
            coeftemp=self.coef.transpose(3,0,1,2,4)
            outcoef=np.zeros(coeftemp.shape)
            for n in range(int(ftermsNum/2)):
                outcoef[n+1]=coeftemp[2*n]*np.sin(coeftemp[2*n+1]*2.*np.pi*(n+1.)/self.spacing[-1])
                outcoef[int(ftermsNum/2+1)+n]=coeftemp[2*n]*np.cos(coeftemp[2*n+1]*2.*np.pi*(n+1.)/self.spacing[-1])
                outcoef[0]=outcoef[0]+outcoef[n+1]
            #transpose fourier terms back to [x,y,z,f,uvw]
            outcoef=outcoef.transpose(1,2,3,0,4)
        return outcoef
    def convertToFRalpha(self):
        newcoef=None
        if self.fourierFormat=='fccss':
            coeftemp=self.coef.transpose(3,0,1,2,4)
            newcoef=np.zeros(coeftemp.shape)
            for n in range(int(ftermsNum/2)):
                newcoef[2*n]=np.sqrt(coeftemp[n+1]**2.+coeftemp[int(ftermsNum/2+1)+n]**2.)
                newcoef[2*n+1]=np.arctan2(coeftemp[n+1],coeftemp[int(ftermsNum/2+1)+n])/2./np.pi/(n+1.)*self.spacing[-1]
            #transpose fourier terms back to [x,y,z,f,uvw]
            newcoef=newcoef.transpose(1,2,3,0,4)
        return newcoef
    '''
    Smoothing functions
    '''
    def GaussianAmpEdit(self,coord,radius,ratio,sigma=0.4):
        if type(radius)!=list:
            radius=[radius,radius,radius]
        dist,coordSlice=self.getNormDistanceMat(coord,radius)
        gaussMat = np.exp(-( (dist)**2 / ( 2.0 * sigma**2 ) ) )
        gaussMat=np.array([gaussMat,gaussMat,gaussMat]).transpose(1,2,3,0)
        tempCoef=self.coef.transpose(3,0,1,2,4)
        for n in range(int(self.coef.shape[-2]/2)):
            tempCoef[2*n][coordSlice]=tempCoef[2*n][coordSlice]*(1+gaussMat*(ratio-1.))
        self.coef=tempCoef.transpose(1,2,3,0,4)
        return
    def GaussianAmpSmoothing(self,radius,targetCoef=None,coord=None,sigma=0.4,ratio=1.):
        if type(targetCoef)==type(None):
            targetCoef=np.array(range(self.coef.shape[-2]))
        if type(radius)!=list:
            radius=[radius,radius,radius]
        dist,coordSlice=self.getNormDistanceMat(coord,radius)    
        gaussMat= np.exp(-( (dist)**2 / ( 2.0 * sigma**2 ) ) )
        gaussMat=np.array([gaussMat,gaussMat,gaussMat]).transpose(1,2,3,0)
        refcoef=self.coef.transpose(3,0,1,2,4)
        newcoef=self.coef.transpose(3,0,1,2,4)
        tempShape=refcoef.shape[1:]
        for n in targetCoef:
            tempCoef=np.zeros(tempShape)
            totalweight=np.zeros(tempShape)
            if type(coord)==type(None):
                coordInd=[range(int(dist.shape[0]/2),tempShape[0]-int(dist.shape[0]/2)),range(int(dist.shape[1]/2),tempShape[1]-int(dist.shape[1]/2)),range(int(dist.shape[2]/2),tempShape[2]-int(dist.shape[2]/2))]
            else:
                coordInd=[range(coordSlice[0].start,coordSlice[0].stop),range(coordSlice[1].start,coordSlice[1].stop),range(coordSlice[2].start,coordSlice[2].stop)]
            for x in coordInd[0]:
                for y in coordInd[1]:
                    for z in coordInd[2]:
                        coordSlice=[slice(x-int(dist.shape[0]/2),x-int(dist.shape[0]/2)+dist.shape[0]),slice(y-int(dist.shape[1]/2),y-int(dist.shape[1]/2)+dist.shape[1]),slice(z-int(dist.shape[2]/2),z-int(dist.shape[2]/2)+dist.shape[2])]
                        newcoef[n][x,y,z]=np.sum(gaussMat*refcoef[n][coordSlice])/np.sum(gaussMat)*ratio+(1.-ratio)*refcoef[n][x,y,z]
        self.coef=newcoef.transpose(1,2,3,0,4)
        return
    
    def regrid(self,coordsList,coefList,weight=None,linearConstrainPoints=[],linearConstrainWeight=None):
        '''
        Regrid the BsplineFourier coefficients from sample points
        Parameters:
            coordsList=[[x,y,z],] or [x,y,z]:list,np.ndarray
                Coordinate or list of coordinates
            coefList(fourier,uvw):list,np.ndarray
                u, v and w fourier coefficients
        '''
        print('Regriding',len(coordsList),'points.')
        if type(weight)==type(None):
            weight=np.ones(len(coordsList))
        else:
            weight=np.array(weight)
        Jmat=[]
        dCList,CIndList=self.getdC(coordsList)
        for n in range(len(dCList)):
            tempRow=sparse.csr_matrix((dCList[n].reshape((-1,),order='F'),(np.zeros(len(CIndList[n])),CIndList[n].copy())),shape=(1,self.numCoefXYZ))
            Jmat.append(tempRow.copy())
        if len(linearConstrainPoints)!=0:
            if type(linearConstrainWeight)==type(None):
                linearConstrainWeight=np.ones(len(linearConstrainPoints)*3)
            elif type(linearConstrainWeight) in [int,float]:
                linearConstrainWeight=np.ones(len(linearConstrainPoints)*3)*linearConstrainWeight
            else:
                linearConstrainWeight=np.array([linearConstrainWeight,linearConstrainWeight,linearConstrainWeight]).reshape((-1,),order='F')
            weight=np.hstack((weight,linearConstrainWeight))
            dCListX,CIndListX=self.getdC(linearConstrainPoints,dxyz=[1,0,0])
            dCListY,CIndListY=self.getdC(linearConstrainPoints,dxyz=[0,1,0])
            dCListZ,CIndListZ=self.getdC(linearConstrainPoints,dxyz=[0,0,1])
            for n in range(len(dCListX)):
                tempRow=sparse.csr_matrix((dCListX[n].reshape((-1,),order='F'),(np.zeros(len(CIndListX[n])),CIndListX[n].copy())),shape=(1,self.numCoefXYZ))
                Jmat.append(tempRow.copy())
                tempRow=sparse.csr_matrix((dCListY[n].reshape((-1,),order='F'),(np.zeros(len(CIndListY[n])),CIndListY[n].copy())),shape=(1,self.numCoefXYZ))
                Jmat.append(tempRow.copy())
                tempRow=sparse.csr_matrix((dCListZ[n].reshape((-1,),order='F'),(np.zeros(len(CIndListZ[n])),CIndListZ[n].copy())),shape=(1,self.numCoefXYZ))
                Jmat.append(tempRow.copy())
        Jmat=sparse.vstack(Jmat)
        matW=sparse.diags(weight)
        matA=Jmat.transpose().dot(matW.dot(Jmat))
        for nFourier in range(coefList[0].shape[0]):
            for axis in range(coefList[0].shape[1]):
                natb=Jmat.transpose().dot(weight*np.hstack((np.array(coefList)[:,nFourier,axis],np.zeros(len(linearConstrainPoints)*3))))
                C=spsolve(matA, natb)
                if type(C)!=np.ndarray:
                    C=C.todense()
                if np.allclose(matA.dot(C), natb):
                    self.coef[:,:,:,nFourier,axis]=C.reshape(self.coef.shape[:3],order='F')
                else:
                    print('Solution error at fourier term',nFourier,', and axis',axis)

    def reshape(self,shape,translate=[[0.,0.],[0.,0.],[0.,0.]]):
        ''' 
        reshape the BsplineFourier coefficients
        Parameters:
            shape=[x,y,z,fourier_terms,3]:list,np.ndarray
                Shape of the final Bspline Fourier
            translate=[[x_start,x_end],[y_start.,y_end],[z_start,z_end]]:list,np.ndarray
                translate the origin by +[x_start,y_start,z_start] and the last control point by +[x_end,y_end,z_end]
        '''
            
        
        if np.all(np.array(translate)==0):
            origin=self.origin.copy()
        else:
            origin=[]
            for n in range(3):
                origin.append(self.origin[n]+translate[n][0])
            origin.append(self.origin[3])
            origin=np.array(origin)
        
        
        
        newbsFourier=BsplineFourier()
        spacing=[]
        for n in range(3):
            spacing.append((self.spacing[n]*(self.coef.shape[n]-1)+translate[n][1]-translate[n][0])/(shape[n]-1))
        spacing.append(self.spacing[3])
        spacing=np.array(spacing)
        newbsFourier.initialize(shape,spacing=spacing,origin=origin)
        
        if type(self.coef)!=type(None): 
            if np.all(np.array(shape)!=np.array(self.coef.shape)) and (self.coef.max()!=0 or self.coef.min()!=0):
                sampleCoef=[]
                samplePoints=np.array(newbsFourier.samplePoints())
                for m in range(len(samplePoints)):
                    sampleCoefTemp=np.zeros(newbsFourier.coef.shape[3:])
                    sampleCoefTemp[:self.coef.shape[3],:]=self.getRefCoef(samplePoints[m])
                    sampleCoef.append(sampleCoefTemp.copy())
                newbsFourier.regrid(samplePoints,sampleCoef)              
        return newbsFourier
    def motionImage(self,imageSize=None,coefFourierWeight=None,xList=None,yList=None,zList=None,scaleFromGrid=None):
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
            imageSize=np.array(self.coef.shape[:3])*scaleFromGrid
            spacing=np.array(self.spacing[:3])/scaleFromGrid
        else:
            spacing=np.array(self.spacing[:3]*(np.array(self.coef.shape[:3]-1)/(imageSize-1)))
        imageSize=np.array(imageSize).astype(int)
        if type(coefFourierWeight)==type(None):
            coefFourierWeight=np.zeros(int(self.coef.shape[3]/2))
            coefFourierWeight[0]=1.
        maxomega=np.argmax(coefFourierWeight)
        '''
        weightRatio=np.ones(self.coef.shape[:3])
        
        sincosAmp=[]
        for m in range(int(self.coef.shape[3]/2)):
            sincosAmp.append(np.sqrt((self.coef[:,:,:,m+1,:]**2.+self.coef[:,:,:,int(self.coef.shape[3]/2)+m+1,:]**2.).sum(axis=3)))
        weightRatio=sincosAmp[maxomega]/coefFourierWeight[maxomega]
        errorRMS=np.ones(self.coef.shape[:3])
        for m in range(int(self.coef.shape[3]/2)):
            errorRMS+=(weightRatio*coefFourierWeight[m]-sincosAmp[m])**2.
        errorRMS=np.sqrt(errorRMS/float(self.coef.shape[3]/2))
        coef=weightRatio-errorRMS
        '''
        
        if type(xList)==type(None):
            xList=range(imageSize[0])
        if type(yList)==type(None):
            yList=range(imageSize[1])
        if type(zList)==type(None):
            zList=range(imageSize[2])
        imgData=np.zeros(imageSize)
        imgDimlen={'x':spacing[0],'y':spacing[1],'z':spacing[2]}
        for x in xList:
            print('    {0:.3f}% completed...'.format(float(xList.index(x))/len(xList)*100.))
            for y in yList:
                for z in zList:
                    vec=self.getVector([x*imgDimlen['x'],y*imgDimlen['y'],z*imgDimlen['z']])
                    fvalue=np.zeros(int(self.coef.shape[3]/2))
                    for m in range(int(self.coef.shape[3]/2)):
                        fvalue[m]=np.sqrt((vec[m+1]**2.+vec[int(self.coef.shape[3]/2)+m+1]**2.).sum())
                    weightRatio=fvalue[maxomega]/coefFourierWeight[maxomega]
                    errorRMS=0.
                    for m in range(int(self.coef.shape[3]/2)):
                        errorRMS+=(weightRatio*coefFourierWeight[m]-fvalue[m])**2.
                    errorRMS=np.sqrt(errorRMS/float(self.coef.shape[3]/2-1))
                    imgData[x,y,z]=weightRatio-errorRMS
        return (imgData,imgDimlen)


    
