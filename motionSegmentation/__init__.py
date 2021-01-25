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
Author: w.x.chan@gmail.com    07OCT2019                - v2.3.3
						-bfSolver version 2.2.4
						-BsplineFourier version 2.3.3
Author: w.x.chan@gmail.com    13NOV2019                - v2.4.1
						-bfSolver version 2.4.1
						-BsplineFourier version 2.3.3
Author: w.x.chan@gmail.com    18NOV2019                - v2.4.2
						-bfSolver version 2.4.1
						-BsplineFourier version 2.4.2
Author: w.x.chan@gmail.com    18NOV2019                - v2.4.3
						-bfSolver version 2.4.3
						-BsplineFourier version 2.4.2
Author: w.x.chan@gmail.com    18NOV2019                - v2.4.5
						-bfSolver version 2.4.4
						-BsplineFourier version 2.4.4
Author: w.x.chan@gmail.com    11DEC2019                - v2.4.6
						-bfSolver version 2.4.6
						-BsplineFourier version 2.4.4
Author: w.x.chan@gmail.com    13DEC2019                - v2.4.7
						-bfSolver version 2.4.6
						-BsplineFourier version 2.4.7
Author: w.x.chan@gmail.com    04FEB2020                - v2.5.1
						-bfSolver version 2.4.6
						-BsplineFourier version 2.4.7
						-motionCorrect version 2.4.7
						-segment verion 2.5.1
Author: w.x.chan@gmail.com    04FEB2020                - v2.5.7
						-bfSolver version 2.5.4
						-BsplineFourier version 2.4.7
						-motionCorrect version 2.4.7
						-segment verion 2.5.7
Author: w.x.chan@gmail.com    18FEB2020                - v2.5.10
						-bfSolver version 2.5.4
						-BsplineFourier version 2.4.7
						-motionCorrect version 2.4.7
						-segment verion 2.5.10
Author: w.x.chan@gmail.com    21FEB2020                - v2.6.2
						-bfSolver version 2.5.4
						-BsplineFourier version 2.6.2
						-motionCorrect version 2.4.7
						-segment verion 2.6.1
Author: w.x.chan@gmail.com    27FEB2020                - v2.6.3
						-bfSolver version 2.5.4
						-BsplineFourier version 2.6.2
						-motionCorrect version 2.6.3
						-segment verion 2.6.1
Author: w.x.chan@gmail.com    27FEB2020                - v2.7.7
						-bfSolver version 2.5.4
						-BsplineFourier version 2.6.2
						-motionCorrect version 2.6.3
						-segment verion 2.7.7
Author: w.x.chan@gmail.com    06MAR2020                - v2.7.9
						-bfSolver version 2.5.4
						-BsplineFourier version 2.6.2
						-motionCorrect version 2.7.8
						-segment verion 2.7.9
Author: jorry.zhengyu@gmail.com    03June2020                - v2.7.11
						-bfSolver version 2.7.11
						-BsplineFourier version 2.6.2
						-motionCorrect version 2.7.8
						-segment verion 2.7.9
Author: w.x.chan@gmail.com    15Oct2020                - v2.7.12
						-bfSolver version 2.7.11
						-BsplineFourier version 2.7.12
						-motionCorrect version 2.7.8
						-segment verion 2.7.9
Author: w.x.chan@gmail.com    07Nov2020                - v2.7.13
						-bfSolver version 2.7.11
						-BsplineFourier version 2.7.13
						-motionCorrect version 2.7.8
						-segment verion 2.7.9
Author: w.x.chan@gmail.com    20Jan2021                - v2.7.16 -added import exception
						-bfSolver version 2.7.15
						-BsplineFourier version 2.7.14
						-motionCorrect version 2.7.8
						-segment verion 2.7.14
Author: w.x.chan@gmail.com    20Jan2021                - v2.7.17
						-bfSolver version 2.7.15
						-BsplineFourier version 2.7.14
						-motionCorrect version 2.7.8
						-segment verion 2.7.17
Author: w.x.chan@gmail.com    25Jan2021                - v2.7.19
						-bfSolver version 2.7.15
						-BsplineFourier version 2.7.14
						-motionCorrect version 2.7.8
						-segment verion 2.7.19

Requirements:
    autoD
    numpy
    re
    scipy
    BsplineFourier
    pickle (optional)
    nfft

Known Bug:
    HSV color format not supported
All rights reserved.
'''
_version='2.7.19'
import logging
logger = logging.getLogger('motionSegmentation v'+_version)
logger.info('motionSegmentation version '+_version)

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

try:
    from BsplineFourier import *
except Exception as e:
    logger.warning(repr(e))
try:
    from bfSolver import *
except Exception as e:
    logger.warning(repr(e))

