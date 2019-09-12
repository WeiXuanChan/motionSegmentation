'''
File: __init__.py
Description: load all class for bfmotionsolver
             Contains linker to main classes
History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: w.x.chan@gmail.com         31JAN2018           - Created
Author: w.x.chan@gmail.com         31JAN2019           - v1.2.0


Requirements:
    autoD
    numpy
    re
    scipy
    BsplineFourier
    pickle (optional)

Known Bug:
    HSV color format not supported
All rights reserved.
'''
print('motionSegmentation version 1.2.0')

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import bfSolver as solver
import BsplineFourier as bsfourier

#linker to classes
BFSolver=solver.bfSolver
BsplineFourier=bsfourier.BsplineFourier
Bspline=bsfourier.Bspline