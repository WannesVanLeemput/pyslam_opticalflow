"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import os 
import cv2
import image_slicer
import numpy as np

from utils import Printer,getchar

from orbslam2_features import ORBextractor
from skimage.util import view_as_blocks



kVerbose = True
kImageBlocks = False


# interface for pySLAM 
class Orbslam2Feature2D: 
    def __init__(self,num_features=2000, scale_factor=1.2, num_levels=8): 
        print('Using Orbslam2Feature2D')
        self.orb_extractor = ORBextractor(num_features, scale_factor, num_levels)
    
    # extract keypoints 
    def detect(self, img, mask=None):  #mask is fake: it is not considered by the c++ implementation 
        # detect and compute 
        kps_tuples = self.orb_extractor.detect(img)            
        # convert keypoints 
        kps = [cv2.KeyPoint(*kp) for kp in kps_tuples]
        return kps       
    
    def compute(self, img, kps, mask=None):
        Printer.orange('WARNING: you are supposed to call detectAndCompute() for ORB2 instead of compute()')
        Printer.orange('WARNING: ORB2 is recomputing both kps and des on input frame', img.shape)            
        return self.detectAndCompute(img)
        #return kps, np.array([])
        
    # compute both keypoints and descriptors       
    def detectAndCompute(self, img, mask=None): #mask is fake: it is not considered by the c++ implementation 
        # detect and compute
        if kImageBlocks:
            height, width = img.shape
            n_height = height//2
            n_width = width//2
            image_blocks = view_as_blocks(img, block_shape=(n_height, n_width))
            kps_tuples_NW, des_nw = self.orb_extractor.detectAndCompute(image_blocks[0, 0])
            kps_tuples_NE, des_ne = self.orb_extractor.detectAndCompute(image_blocks[0, 1])
            kps_tuples_SW, des_sw = self.orb_extractor.detectAndCompute(image_blocks[1, 0])
            kps_tuples_SE, des_se = self.orb_extractor.detectAndCompute(image_blocks[1, 1])
            kps_tuples_NE = [(tup[0]+n_width, tup[1], tup[2], tup[3], tup[4], tup[5]) for tup in kps_tuples_NE]
            kps_tuples_SW = [(tup[0], tup[1] + n_height, tup[2], tup[3], tup[4], tup[5]) for tup in kps_tuples_SW]
            kps_tuples_SE = [(tup[0] + n_width, tup[1] + n_height, tup[2], tup[3], tup[4], tup[5]) for tup in kps_tuples_SE]
            # kps_tuples, des = self.orb_extractor.detectAndCompute(img)
            kps_tuples = kps_tuples_NW + kps_tuples_NE + kps_tuples_SW + kps_tuples_SE
            des = np.concatenate((des_nw, des_ne, des_sw, des_se), axis=0)
        else:
            kps_tuples, des = self.orb_extractor.detectAndCompute(img)
        # convert keypoints
        kps = [cv2.KeyPoint(*kp) for kp in kps_tuples]
        print(des.shape)
        return kps, des