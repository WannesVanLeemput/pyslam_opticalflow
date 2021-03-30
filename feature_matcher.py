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
import numpy as np
import cv2

from parameters import Parameters
from enum import Enum
from collections import defaultdict
from utils import Printer
from thirdparty.flownet2.models import FlowNet2
from thirdparty.flownet2.utils.frame_utils import read_gen
import torch
from skimage.util import pad

kRatioTest = Parameters.kFeatureMatchRatioTest
kVerbose = False


class FeatureMatcherTypes(Enum):
    NONE = 0
    BF = 1
    FLANN = 2
    FLOW = 3


def feature_matcher_factory(norm_type=cv2.NORM_HAMMING, cross_check=False, ratio_test=kRatioTest,
                            type=FeatureMatcherTypes.FLANN):
    if type == FeatureMatcherTypes.BF:
        return BfFeatureMatcher(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
    if type == FeatureMatcherTypes.FLANN:
        return FlannFeatureMatcher(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
    if type == FeatureMatcherTypes.FLOW:
        return FlowFeatureMatcher(type=type)
    return None


"""
N.B.: 
The result of matches = matcher.knnMatch() is a list of cv2.DMatch objects. 
A DMatch object has the following attributes:
    DMatch.distance - Distance between descriptors. The lower, the better it is.
    DMatch.trainIdx - Index of the descriptor in train descriptors
    DMatch.queryIdx - Index of the descriptor in query descriptors
    DMatch.imgIdx - Index of the train image.
"""


# base class 
class FeatureMatcher(object):
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check=False, ratio_test=kRatioTest,
                 type=FeatureMatcherTypes.BF):
        self.type = type
        self.norm_type = norm_type
        self.cross_check = cross_check  # apply cross check
        self.matches = []
        self.ratio_test = ratio_test
        self.matcher = None
        self.matcher_name = ''

    # input: des1 = queryDescriptors, des2= trainDescriptors
    # output: idx1, idx2  (vectors of corresponding indexes in des1 and des2, respectively)
    def match(self, des1, des2, ratio_test=None):
        if kVerbose:
            print(self.matcher_name, ', norm ', self.norm_type)
            # print('des1.shape:',des1.shape,' des2.shape:',des2.shape)
        # print('des1.dtype:',des1.dtype,' des2.dtype:',des2.dtype)
        matches = self.matcher.knnMatch(des1, des2, k=2)  # knnMatch(queryDescriptors,trainDescriptors)
        self.matches = matches
        return self.goodMatches(matches, des1, des2, ratio_test)

        # input: des1 = query-descriptors, des2 = train-descriptors, kps1 = query-keypoints, kps2 = train-keypoints

    # output: idx1, idx2  (vectors of corresponding indexes in des1 and des2, respectively)
    # N.B.0: cross checking can be also enabled with the BruteForce Matcher below 
    # N.B.1: after matching there is a model fitting with fundamental matrix estimation 
    # N.B.2: fitting a fundamental matrix has problems in the following cases: [see Hartley/Zisserman Book]
    # - 'geometrical degenerate correspondences', e.g. all the observed features lie on a plane (the correct model for the correspondences is an homography) or lie a ruled quadric 
    # - degenerate motions such a pure rotation (a sufficient parallax is required) or an infinitesimal viewpoint change (where the translation is almost zero)
    # N.B.3: as reported above, in case of pure rotation, this algorithm will compute a useless fundamental matrix which cannot be decomposed to return a correct rotation    
    # Adapted from https://github.com/lzx551402/geodesc/blob/master/utils/opencvhelper.py 
    def matchWithCrossCheckAndModelFit(self, des1, des2, kps1, kps2, ratio_test=None, cross_check=True, err_thld=1,
                                       info=''):
        """Compute putative and inlier matches.
        Args:
            feat: (n_kpts, 128) Local features.
            cv_kpts: A list of keypoints represented as cv2.KeyPoint.
            ratio_test: The threshold to apply ratio test.
            cross_check: (True by default) Whether to apply cross check.
            err_thld: Epipolar error threshold.
            info: Info to print out.
        Returns:
            good_matches: Putative matches.
            mask: The mask to distinguish inliers/outliers on putative matches.
        """
        idx1, idx2 = [], []
        if ratio_test is None:
            ratio_test = self.ratio_test

        init_matches1 = self.matcher.knnMatch(des1, des2, k=2)
        init_matches2 = self.matcher.knnMatch(des2, des1, k=2)

        good_matches = []

        for i, (m1, n1) in enumerate(init_matches1):
            cond = True
            if cross_check:
                cond1 = cross_check and init_matches2[m1.trainIdx][0].trainIdx == i
                cond *= cond1
            if ratio_test is not None:
                cond2 = m1.distance <= ratio_test * n1.distance
                cond *= cond2
            if cond:
                good_matches.append(m1)
                idx1.append(m1.queryIdx)
                idx2.append(m1.trainIdx)

        if type(kps1) is list and type(kps2) is list:
            good_kps1 = np.array([kps1[m.queryIdx].pt for m in good_matches])
            good_kps2 = np.array([kps2[m.trainIdx].pt for m in good_matches])
        elif type(kps1) is np.ndarray and type(kps2) is np.ndarray:
            good_kps1 = np.array([kps1[m.queryIdx] for m in good_matches])
            good_kps2 = np.array([kps2[m.trainIdx] for m in good_matches])
        else:
            raise Exception("Keypoint type error!")
            exit(-1)

        _, mask = cv2.findFundamentalMat(good_kps1, good_kps2, cv2.RANSAC, err_thld, confidence=0.999)
        n_inlier = np.count_nonzero(mask)
        print(info, 'n_putative', len(good_matches), 'n_inlier', n_inlier)
        return idx1, idx2, good_matches, mask

    # input: des1 = query-descriptors, des2 = train-descriptors
    # output: idx1, idx2  (vectors of corresponding indexes in des1 and des2, respectively)
    # N.B.: this returns matches where each trainIdx index is associated to only one queryIdx index    
    def goodMatchesOneToOne(self, matches, des1, des2, ratio_test=None):
        len_des2 = len(des2)
        idx1, idx2 = [], []
        # good_matches = []           
        if ratio_test is None:
            ratio_test = self.ratio_test
        if matches is not None:
            float_inf = float('inf')
            dist_match = defaultdict(lambda: float_inf)
            index_match = dict()
            for m, n in matches:
                if m.distance > ratio_test * n.distance:
                    continue
                dist = dist_match[m.trainIdx]
                if dist == float_inf:
                    # trainIdx has not been matched yet
                    dist_match[m.trainIdx] = m.distance
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    index_match[m.trainIdx] = len(idx2) - 1
                else:
                    if m.distance < dist:
                        # we have already a match for trainIdx: if stored match is worse => replace it
                        # print("double match on trainIdx: ", m.trainIdx)
                        index = index_match[m.trainIdx]
                        assert (idx2[index] == m.trainIdx)
                        idx1[index] = m.queryIdx
                        idx2[index] = m.trainIdx
        return idx1, idx2

    # input: des1 = query-descriptors, des2 = train-descriptors
    # output: idx1, idx2  (vectors of corresponding indexes in des1 and des2, respectively)
    # N.B.: this may return matches where a trainIdx index is associated to two (or more) queryIdx indexes
    def goodMatchesSimple(self, matches, des1, des2, ratio_test=None):
        idx1, idx2 = [], []
        # good_matches = []
        if ratio_test is None:
            ratio_test = self.ratio_test
        if matches is not None:
            for m, n in matches:
                if m.distance < ratio_test * n.distance:
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
        return idx1, idx2

        # input: des1 = query-descriptors, des2 = train-descriptors

    # output: idx1, idx2  (vectors of corresponding indexes in des1 and des2, respectively)
    def goodMatches(self, matches, des1, des2, ratio_test=None):
        # return self.goodMatchesSimple(matches, des1, des2, ratio_test)   # <= N.B.: this generates problem in SLAM since it can produce matches where a trainIdx index is associated to two (or more) queryIdx indexes
        return self.goodMatchesOneToOne(matches, des1, des2, ratio_test)


# Brute-Force Matcher 
class BfFeatureMatcher(FeatureMatcher):
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check=False, ratio_test=kRatioTest,
                 type=FeatureMatcherTypes.BF):
        super().__init__(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
        self.matcher = cv2.BFMatcher(norm_type, cross_check)
        self.matcher_name = 'BfFeatureMatcher'

    # Flann Matcher


class FlannFeatureMatcher(FeatureMatcher):
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check=False, ratio_test=kRatioTest,
                 type=FeatureMatcherTypes.FLANN):
        super().__init__(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
        if norm_type == cv2.NORM_HAMMING:
            # FLANN parameters for binary descriptors 
            FLANN_INDEX_LSH = 6
            self.index_params = dict(algorithm=FLANN_INDEX_LSH,
                                     # Multi-Probe LSH: Efficient Indexing for High-Dimensional Similarity Search
                                     table_number=6,  # 12
                                     key_size=12,  # 20
                                     multi_probe_level=1)  # 2
        if norm_type == cv2.NORM_L2:
            # FLANN parameters for float descriptors 
            FLANN_INDEX_KDTREE = 1
            self.index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
        self.search_params = dict(checks=32)  # or pass empty dictionary
        self.matcher = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        self.matcher_name = 'FlannFeatureMatcher'


class FlowFeatureMatcher(FeatureMatcher):
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check=False, ratio_test=kRatioTest,
                 type=FeatureMatcherTypes.FLOW):
        super().__init__(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
        self.matcher_name = 'FlowFeatureMatcher'
        print("Initializing FlowNet 2")
        # initial a Net
        self.net = FlowNet2(None).cuda()
        # load the state_dict
        dict = torch.load("/home/wannes/GitHub/pyslam/thirdparty/flownet2/checkpoints/FlowNet2_checkpoint.pth.tar")
        self.net.load_state_dict(dict["state_dict"])
        print("FlowNet 2 initialized on GPU")

    # This code is borrowed and slightly adapted from the run_a_pair.py script obtained from the flownet2-pytorch repo
    # https://github.com/NVIDIA/flownet2-pytorch
    def match_non_neighbours(self, f_cur, f_ref, padding=True, crop = True):
        im_cur = f_cur.img
        im_ref = f_ref.img
        height, width, depth = im_cur.shape
        if padding:
            pad_h = 0
            pad_w = 0
            if width % 64 != 0:
                pad_w = (64 - width % 64) // 2
            if height % 64 != 0:
                pad_h = (64 - height % 64) // 2
            im1_padded = pad(im_ref, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant', constant_values=0)
            im2_padded = pad(im_cur, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant', constant_values=0)
            images = [im1_padded, im2_padded]
        else:
            images = [im_ref, im_cur]
        images = np.array(images).transpose(3, 0, 1, 2)
        im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

        # process the image pair to obtian the flow
        result = self.net(im).squeeze()
        flow2d = result.data.cpu().numpy().transpose(1, 2, 0)
        if crop:
            flow2d = np.delete(flow2d, slice(0, 32), 0)
            flow2d = np.delete(flow2d, slice(flow2d.shape[0] -33, -1), 0)
        mvs_ref = flow2d.reshape(-1, flow2d.shape[-1])
        return self.match(f_cur, f_ref, mv_ref=mvs_ref)

    # out: a vector of match index pairs [idx1[i],idx2[i]] such that the keypoint f1.kps[idx1[i]] is matched with f2.kps[idx2[i]]
    def match(self, f_cur, f_ref, ratio_test=None, mv_ref=None):
        # assert type(f_cur) == Frame.__class__ # for debugging purposes
        idx1 = []
        idx2 = []
        if type(mv_ref) != np.ndarray:
            motion_vectors = f_ref.mvs
        else:
            motion_vectors = mv_ref

        keypoints_ref = f_ref.kps
        offset_x = keypoints_ref[0][0]
        offset_y = keypoints_ref[0][1]
        width = Parameters.kWidth
        height = Parameters.kHeight
        count = 0
        stride_h = Parameters.kStrideHorizontal
        stride_v = Parameters.kStrideVertical
        # if f_cur.id > 5:
        #    stride_v *= 4
        #    stride_h *= 4
        for mv, keypoint, idx_ref in zip(motion_vectors, keypoints_ref, range(len(keypoints_ref))):
            if keypoint[0] % stride_h == 0 and keypoint[1] % stride_v == 0:
                new_x = int(round(keypoint[0] + mv[0]))
                new_y = int(round(keypoint[1] - mv[1]))
                match_idx = int(new_x + (new_y - offset_y) * width)
                if offset_x <= new_x <= width - offset_x - 1 and offset_y <= new_y <= height - offset_y - 1 and match_idx < len(
                        f_cur.des):
                    if __debug__:
                        if new_x != int(f_cur.kps[match_idx][0]):
                            print('Error matching frames: height-coordinate mismatch', new_x, '!=',
                                  int(f_cur.kps[match_idx][0]))
                        if new_y != int(f_cur.kps[match_idx][1]):
                            print('Error matching frames: width-coordinate mismatch', new_y, '!=',
                                  int(f_cur.kps[match_idx][1]))
                    # due to rounding motion vectors (we can't use sub-pixel accuracy) only use first match to certain keypoint
                    idx1.append(match_idx)
                    idx2.append(idx_ref)
            count += 1
        inds = []
        unq_idx1_temp = []
        seen = set()
        for i, ele in enumerate(idx1):
            if ele not in seen:
                inds.append(i)
                unq_idx1_temp.append(ele)
                seen.add(ele)
        unq_idx2_temp = list(np.array(idx2)[inds])
        inds = []
        seen = set()
        unq_idx2 = []
        for i, ele in enumerate(unq_idx2_temp):
            if ele not in seen:
                inds.append(i)
                unq_idx2.append(ele)
                seen.add(ele)
        unq_idx1 = list(np.array(unq_idx1_temp)[inds])
        if len(unq_idx1) != len(set(unq_idx1)) or len(unq_idx2) != len(set(unq_idx2)):
            Printer.red("WARNING: matched keypoint multiple times, will result in BA error later!")
        return unq_idx1, unq_idx2
