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
_version='2.2.7'
print('motionSegmentation version',_version)

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from BsplineFourier import *
from bfSolver import *

