#!/usr/bin/env python

"""
Code borrowed from https://github.com/darylclimb/cvml_project and https://gitlab.com/xr-lab/ue-generator-scripts/-/blob/master/reconstruct.py
"""
import glob

import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import argparse
import thirdparty.flowlib.flowlib as fl
#from geometry_utils import pixel_coord_np
import re


def pixel_coord_np(width, height):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    x = np.linspace(0, width - 1, width).astype(np.int)
    y = np.linspace(0, height - 1, height).astype(np.int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))


def load_depth(file_depth, downsample=1, max_z=20000.0):
    # method borrowed from UnrealGT
    srcImg = cv2.imread(file_depth, cv2.IMREAD_COLOR)
    H, W, C = srcImg.shape

    H //= downsample
    W //= downsample

    # TODO optimize
    if True:
        img32f = np.zeros((H, W, 1), dtype="float32")
        img32f[:, :] = np.nan
        for y in range(0, H, downsample):
            for x in range(0, W, downsample):
                pixelDepthMM = srcImg[y * downsample, x * downsample][2] + srcImg[y * downsample, x * downsample][
                    1] * 256 + srcImg[y * downsample, x * downsample][0] * 256 * 256
                pixelDepthM = pixelDepthMM / 10  # this depends on the scaling used in the export method in UE4
                if pixelDepthM < max_z:
                    img32f[y, x] = pixelDepthM / 5000  # same remark ^

    print("min", np.min(img32f), "max", np.max(img32f), "median", np.median(img32f))
    return img32f


def load_pose(file_pose):
    with open(file_pose, 'r') as fp:
        line = fp.read()

        # in UnrealGT, Rotator is yaw pitch roll
        # in our plugin implementation it's roll pitch yaw
        tx, ty, tz, rx, ry, rz = line.split(' ')

        # Unreal is left-handed, change to right-handed
        _pose = np.array([ty, tx, tz, rx, ry, rz]).astype(np.float32)

        # prepare transform
        T = np.eye(4)

        # translation
        T[:3, 3] = _pose[:3] / 5000

        # rotation
        yaw = - _pose[5] * np.pi / 180
        pitch = _pose[4] * np.pi / 180
        roll = _pose[3] * np.pi / 180
        R_roll = np.array([
            [np.cos(roll), 0, np.sin(roll)],
            [0, 1, 0],
            [-np.sin(roll), 0, np.cos(roll)],
        ])
        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ])
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ])
        T[:3, :3] = np.dot(R_roll, T[:3, :3])
        T[:3, :3] = np.dot(R_pitch, T[:3, :3])
        T[:3, :3] = np.dot(R_yaw, T[:3, :3])

        return T


def generate_flow(pose1, pose2, depth, K, output_file, visualize=False):
    # Load images
    # img_rgb_original = cv2.imread("/home/wannes/storage/agent_1/rgb/GDumper_A/130.png", cv2.IMREAD_COLOR)
    # rgb = cv2.cvtColor(img_rgb_original, cv2.COLOR_BGR2RGB)

    # pose1 = load_pose("/home/wannes/storage/agent_1/pose/GDumper_A/130.txt")
    # pose2 = load_pose("/home/wannes/storage/agent_1/pose/GDumper_A/131.txt")

    # Depth is stored as float32 in meters
    # depth = load_depth("/home/wannes/storage/agent_1/depth/GDumper_A/130.png")
    # depth2 = load_depth("/home/wannes/storage/agent_1/depth/GDumper_A/131.png")

    # Get intrinsic parameters
    height, width, _ = depth.shape
    K_inv = np.linalg.inv(K)

    # Get pixel coordinates
    pixel_coords = pixel_coord_np(width, height)  # [3, npoints]

    transform = np.linalg.inv(pose1) @ pose2

    # Apply back-projection: K_inv @ pixels * depth
    cam_coords = K_inv[:3, :3] @ pixel_coords * depth.flatten()
    cam_coords_homogeneous = np.vstack((cam_coords, np.ones(width * height)))
    cam2_coords = (K[:3, :3] @ transform[:3, :] @ cam_coords_homogeneous)  # * (depth.flatten() ** -1)

    flow = np.zeros((height, width, 2))
    for i in range(len(cam2_coords[0])):
        w, h, _ = pixel_coords[:, i]
        flow[h, w, 0] = (cam2_coords[0][i] - pixel_coords[0][i]) * -1
        flow[h, w, 1] = (cam2_coords[1][i] - pixel_coords[1][i])
    # flownet_flow = fl.read_flow("/home/wannes/storage/agent_1/flow/000130.flo")
    # img = fl.flow_to_image(flow)
    # image = cv2.imread("/home/wannes/storage/agent_1/rgb/GDumper_A/001.png", cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # flow = np.delete(flow, slice(16), 0)
    # flow = np.delete(flow, slice(height-32,height-16), 0)
    # image = fl.read_image("/home/wannes/storage/agent_1/rgb/GDumper_A/130.png")
    # image = np.delete(image, slice(16), 0)
    # image = np.delete(image, slice(height - 32, height-16), 0)
    # im = fl.flow_to_image(flow)
    # plt.imshow(rgb)

    # X = []
    # Y = []
    # U = []
    # V = []
    # for y in range(0, len(flow), 20):
    #    for x  in range(0, len(flow[0]), 20):
    #        X.append(x)
    #        Y.append(y)
    #        U.append(flow[y, x][0])
    #        V.append(flow[y, x][1])
    # plt.quiver(X, Y, U, V, units='dots')
    # plt.show()
    fl.write_flow(flow, output_file)
    if visualize:
        im = fl.flow_to_image(flow)
        plt.imshow(im)
        plt.show()

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def main(pose_directory, depth_directory, output_directory):
    current_pose = None
    current_depth = None
    index = None

    K = o3d.camera.PinholeCameraIntrinsic(
        width=640, height=480,
        fx=320, fy=240,
        cx=320, cy=240).intrinsic_matrix

    for _pose, _depth in zip(sorted(glob.glob(pose_directory + '*.txt'), key=numericalSort),
                             sorted(glob.glob(depth_directory + '*.png'), key=numericalSort)):
        if current_pose is None and current_depth is None:
            current_pose = load_pose(_pose)
            current_depth = load_depth(_depth)
            index = _depth.split('/')[-1].split('.')[0]
            continue
        else:
            next_pose = load_pose(_pose)
            next_depth = load_depth(_depth)

        output_file = output_directory + index + '.flo'
        generate_flow(current_pose, next_pose, current_depth, K, output_file)
        current_pose = next_pose
        current_depth = next_depth
        index = _depth.split('/')[-1].split('.')[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate ground truth optical flow from poses and depth information.")
    parser.add_argument('pose_directory', type=str, help='The directory containing pose files in UE format.')
    parser.add_argument('depth_directory', type=str, help='The directory containing the png depth images.')
    parser.add_argument('output_directory', type=str, help='The directory to save the generated flow to.')
    args = parser.parse_args()
    main(args.pose_directory, args.depth_directory,
         args.output_directory)
