# motionSegmentation
Explicit spatio-temporal regularization of motion tracking<sup>1</sup> using registered vector maps from SimpleElastix<sup>2</sup>

### Installation
If your default python is python3:
pip install motionSegmentation

### Usage
#### simpleSolver(savePath,startstep=1,endstep=7,fileScale=None,getCompoundTimeList=None,compoundSchemeList=None,fftLagrangian=True,pngFileFormat=None,period=None,maskImg=True,anchor=None,peroid=None,bgrid=4.,finalShape=None,fourierTerms=4,twoD=False)
##### REQUIRED FILES
savePath+'/scale.txt': the length per pixel x, y, z of the image 
##### OPTIONAL FILES
savePath+'/diastole_systole.txt': diastole and systolic time frame
savePath+'/transform/manual_t{0:d}to{1:d}_0.txt'.format(anchor[n][0],anchor[n][1]): additional bspline vectors to support the motion estimation
##### savePath
set the folder path to be used where the generated and required files will be are.
##### startstep, endstep
Set the start and end step for the simpleSolver.

step 1: load image -- Image will be loaded as savePath+'/'+pngFileFormat.format(t,z)

step 2: create mask

step 3: Registrations

step 4: initialize BSF using fft

step 5: solve BSF

step 6: regrid to time

setp 7: compute compound

##### fileScale
set based on the scale of image dimension such that fileScale*AVERAGE_IMAGE_DIMENSION > 1

##### getCompoundTimeList,
set the timestep(s) for compounding in a list, if None, it will use diastolic and systolic timesteps in savePath+'/diastole_systole.txt', if file does not exist, it will be set as [0]

##### compoundSchemeList
set the compounding method(s) to use in a list, if None it will be set for all the supported compounding methods ['SAC','maximum','mean','wavelet'] 

##### fftLagrangian
Set to False if you do not want to use time frame 0 as reference (t -> t+1 and 0 -> t will be registered), then t -> t+1 and t -> t-1 will be registered.

##### pngFileFormat
set the string format to read and stacked multiple images, if None, defaults to 'time{0:03d}/slice{{0:03d}}time{0:03d}.png' for 3D and 'time{0:03d}.png' for 2D.

##### period
Set the period of the motion in time frame (can be a float), if None, it will default as the len(DIMENSION t)

##### maskImg
if True, it will auto detect and mask borders with 0 intensity

##### anchor
set as a List of pairs of time frame to include in the motion estimation in addition to the default: [[anchor_t_n1,anchor_t_m1],[anchor_t_n2,anchor_t_m2],...]

It will find the bspline files in savePath+'/transform/manual_t{0:d}to{1:d}_0.txt'.format(anchor[n][0],anchor[n][1]) for each pair

##### bgrid
set the spacing of the uniform bspline grid use to represent the image registration in terms of pixels of the largest dimension

##### fourierTerms
Number of Fourier terms, number of Fourier coefficients = Fourier terms*2+1

##### finalShape
Set the final uniform bspline grid shape (NUMBER OF CONTROL POINTS IN x,NUMBER OF CONTROL POINTS IN y,NUMBER OF CONTROL POINTS IN z,NUMBER FOURIER COEFFICIENTS PER CONTROL POINT, NUMBER OF DIMENSIONS), if None, it will default to the shape in t0 -> t1 registration.

#### twoD
set to True is the image set is in 2D

### References
<sup>1</sup> Wiputra, H., Chan, W. X., Foo, Y. Y., Ho, S., & Yap, C. H. (2020). Cardiac motion estimation from medical images: a regularisation framework applied on pairwise image registration displacement fields. Scientific reports, 10(1), 1-14.

<sup>2</sup> Bradley Lowekamp, ., gabehart, ., Daniel Blezek, ., Luis Ibanez, ., Matt McCormick, ., Dave Chen, ., … Brad King, . (2015, June 26). SimpleElastix: SimpleElastix v0.9.0 (Version v0.9.0-SimpleElastix). Zenodo. http://doi.org/10.5281/zenodo.19049
