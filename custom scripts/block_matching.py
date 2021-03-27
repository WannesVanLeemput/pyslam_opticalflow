import sys
import os

sys.path.append(os.path.abspath('/home/wannes/GitHub/pyslam/thirdparty/flowlib'))
import numpy as np
import pandas as pd
import csv
import flowlib
import matplotlib.pyplot as plt
import seaborn as sns
import glob

DEBUG = True
MAC = False


def parse_csv(csvfile):
    motion_vectors = []
    with open(csvfile) as file:
        csv_reader = csv.reader(file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            motion_vectors.append(row)
    return motion_vectors


def str_to_array(string):
    ret = []
    string = string.replace('[', '')
    string = string.replace(']', '')
    temp_list = string.split(' ')
    def_list = [l for l in temp_list if l != '']
    x = float(def_list[0])
    y = float(def_list[-1])
    return [x, y]


def build_array(csvfile, width, height):
    df_flow = pd.read_csv(csvfile)
    array = np.zeros((height, width, 2))
    for index, row in df_flow.iterrows():
        coords = str_to_array(row[0])
        mv_u_orb = float(row[2])
        mv_v_orb = float(row[3])
        mv = [mv_u_orb, mv_v_orb]
        if coords[0] >= 16:
            x = int(round(coords[0]))
            y = int(round(coords[1]))
            array[x, y, 0] = mv[0]
            array[x, y, 1] = mv[1]
    return array


def block_matching(orb, flow, blocksize, width, height):
    block_scores = np.zeros((width // blocksize, height // blocksize))
    block_x = 0
    block_y = 0
    for i in range(0, height - blocksize, blocksize):
        for j in range(0, width - blocksize, blocksize):
            score = 0
            comparisons = 0
            for k in range(blocksize):
                for l in range(blocksize):
                    if (orb[i + k, j + l] != 0).any():
                        comparisons += 1
                        score += np.sum(np.power(orb[i + k, j + l] - flow[j + l, i + k], 2))
            if score != 0:
                block_scores[block_x, block_y] = score / comparisons
            block_x += 1
        block_y += 1
        block_x = 0
    return block_scores


def calculate_block_matching_mse_of(im1, im2, vectors, blocksize):
    image1 = plt.imread(im1)
    image2 = plt.imread(im2)
    mse = []
    height, width, _ = vectors.shape
    for x in range(16, width):
        for y in range(16, height):
            if vectors[y, x].all() == 0:  # handle sparse vectors from orb
                continue
            new_x = int(round(x + vectors[y, x][0]))
            new_y = int(round(y - vectors[y, x][1]))
            start_x = max(0, x - blocksize // 2)
            start_y = max(0, y - blocksize // 2)
            end_x = min(width, x + (blocksize // 2) + 1)
            end_y = min(height, y + (blocksize // 2) + 1)
            block_1 = image1[start_y:end_y, start_x:end_x, :3]

            start_new_x = max(0, new_x - blocksize // 2)
            end_new_x = min(width, new_x + (blocksize // 2) + 1)
            start_new_y = max(0, new_y - blocksize // 2)
            end_new_y = min(height, new_y + (blocksize // 2) + 1)
            block_2 = image2[start_new_y:end_new_y, start_new_x:end_new_x, :3]
            # only compare full blocks
            if block_1.shape == block_2.shape:
                block_diff = (block_1 - block_2) ** 2
                mse.append(np.sum(block_diff) / np.power(blocksize, 2))
    return np.mean(mse)


def calculate_block_matching_mse_orb(image1, image2, vectors, blocksize):
    pass


def show_image_vectors(image, orb, flow, id):
    img = plt.imread(image)
    plt.imshow(img)
    # orb motion vectors
    X = []
    Y = []
    for x in range(len(orb)):
        for y in range(len(orb[0])):
            if orb[x, y].any() != 0:
                X.append(x)
                Y.append(y)
    U = []
    V = []
    U_flow = []
    V_flow = []
    for x, y in zip(X, Y):
        U.append(orb[x, y][0])
        V.append(-orb[x, y][1])
        U_flow.append(flow[y, x][0])
        V_flow.append(-flow[y, x][1])
    name = "/home/wannes/storage/agent_1/motionvectors/comp_6000/" + str(id) + ".png"
    plt.quiver(X, Y, U, V, units='dots', color=(0, 1, 0))
    plt.quiver(X, Y, U_flow, V_flow, units='dots', color=(1, 0, 0))
    plt.savefig(name)
    plt.show()


def main(orb_flow_folder, flownet_flow_folder, image_folder):
    id = 2
    full_str_orb = orb_flow_folder + "/*.csv"
    full_str_flow = flownet_flow_folder + "/*.flo"
    full_str_imgs = image_folder + "/*.png"
    orb_flow_files = sorted([f for f in glob.glob(full_str_orb)], key=os.path.getmtime)
    # orb_flow_files.sort()
    images = [f for f in glob.glob(full_str_imgs)]
    images.sort()
    images = images[2:-1]
    images_2 = images[1:-1]
    flow_flow_files = [f for f in glob.glob(full_str_flow)]
    flow_flow_files.sort()
    flow_flow_files = flow_flow_files[2:-1]
    orb_scores = []
    flow_scores = []
    for flownet_flow, orb_flow, img1, img2 in zip(flow_flow_files, orb_flow_files, images, images_2):
        flownet_mvs = flowlib.read_flow(flownet_flow)
        width = flownet_mvs.shape[0]
        height = flownet_mvs.shape[1]
        orb_mvs = build_array(orb_flow, width, height)
        block_matching_mse_of = calculate_block_matching_mse_of(img1, img2, flownet_mvs, 7)
        block_matching_mse_orb = calculate_block_matching_mse_of(img1, img2, orb_mvs, 7)
        orb_scores.append(block_matching_mse_orb)
        flow_scores.append(block_matching_mse_of)
    return orb_scores, flow_scores


if __name__ == '__main__':
    if DEBUG:
        s1, s2 = main('/home/wannes/storage/agent_1/motionvectors/orb2_6000',
                      '/home/wannes/storage/agent_1/flow',
                      '/home/wannes/storage/agent_1/rgb/images')
        print(np.mean(s1), np.mean(s2))
        with open('bock_scores_flownet.csv', mode='w') as file1:
            writer = csv.writer(file1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(s2)

        with open('bock_scores_orb.csv', mode='w') as file2:
            writer = csv.writer(file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(s1)
    elif MAC:
        img1 = plt.imread('/home/wannes/storage/agent_1/rgb/images/052.png')
        img2 = plt.imread('/home/wannes/storage/agent_1/rgb/images/053.png')
        orb_flow = '/home/wannes/storage/agent_1/motionvectors/orb2_6000/51_52_mvs.csv'
        orb_mvs = build_array(orb_flow, 480, 640)
        mvs = flowlib.read_flow('/home/wannes/storage/agent_1/flow/000051.flo')
        mse = calculate_block_matching_mse_of(img1, img2, orb_mvs, 7)
        print(mse)
    else:
        if len(sys.argv) != 3:
            print("Usage: 'path to CSV file with mvs' 'path to flo file with mvs'")
        else:
            main(sys.argv[1], sys.argv[2])
