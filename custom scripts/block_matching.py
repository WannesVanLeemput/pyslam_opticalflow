import sys
import os
sys.path.append(os.path.abspath('/Users/wannesvanleemput/Documents/School/Thesis/pyslam/thirdparty/flowlib/'))
import numpy as np
import pandas as pd
import csv
import flowlib
import matplotlib.pyplot as plt
import seaborn as sns
import glob

DEBUG = False
MAC = True


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
    block_scores = np.zeros((width//blocksize, height//blocksize))
    block_x = 0
    block_y = 0
    for i in range(0, height-blocksize, blocksize):
        for j in range(0, width-blocksize, blocksize):
            score = 0
            comparisons = 0
            for k in range(blocksize):
                for l in range(blocksize):
                    if (orb[i+k, j+l] != 0).any():
                        comparisons += 1
                        score += np.sum(np.power(orb[i+k, j+l] - flow[j+l, i+k], 2))
            if score != 0:
                block_scores[block_x, block_y] = score / comparisons
            block_x += 1
        block_y += 1
        block_x = 0
    return block_scores


def calculate_block_matching_mse(image1, image2, vectors, blocksize):
    mse = 0
    height, width, _ = vectors.shape
    for x in range(16, width):
        for y in range(16, height):
            new_x = int(round(x + vectors[y, x][0]))
            new_y = int(round(y - vectors[y, x][1]))
            block_error = 0
            for i, l in zip(range(min(0, x - blocksize//2), min(width, x+blocksize//2)), range(min(0, new_x - blocksize // 2),min(width, new_x + blocksize //2))):
                for j, k in zip(range(min(0, y - blocksize//2), min(height, y + blocksize//2)), range(min(0, new_y - blocksize//2), min(height, new_y + blocksize//2))):
                    block_error += np.power(image1[i, j][:3] - image2[l, k][3], 2)
            mse += (block_error/np.power(blocksize, 2))
            print(block_error/np.power(blocksize, 2))
    return mse/vectors.size



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
    plt.quiver(X, Y, U, V, units='dots', color=(0,1,0))
    plt.quiver(X, Y, U_flow, V_flow, units='dots', color=(1,0,0))
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
    flow_flow_files = [f for f in glob.glob(full_str_flow)]
    flow_flow_files.sort()
    flow_flow_files = flow_flow_files[2:-1]
    with open('block_matching.csv', mode='w') as write_file:
        writer = csv.writer(write_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for flownet_flow, orb_flow, img in zip(flow_flow_files, orb_flow_files, images):
            flownet_mvs = flowlib.read_flow(flownet_flow)
            width = flownet_mvs.shape[0]
            height = flownet_mvs.shape[1]
            orb_mvs = build_array(orb_flow, width, height)
            block_scores = block_matching(orb_mvs, flownet_mvs, 5, width, height)
            mse = 0
            for score in block_scores.flatten():
                mse += score
            #print("MSE: ", mse/np.count_nonzero(block_scores))
            #print(np.count_nonzero(block_scores), " motion vectors were compared")
            writer.writerow([mse/np.count_nonzero(block_scores)])
            # show_image_vectors(img, orb_mvs, flownet_mvs, id)
            id += 1
            # ax = sns.heatmap(block_scores, linewidth=0)
            # plt.show()
    return



if __name__ == '__main__':
    if DEBUG:
        main('/home/wannes/storage/agent_1/motionvectors/orb2_6000',
             '/home/wannes/storage/agent_1/flow',
             '/home/wannes/storage/agent_1/rgb/images')
    elif MAC:
        img1 = plt.imread('/Users/wannesvanleemput/OneDrive - UGent/Master 2/Thesis/storage/agent_1/rgb/GDumper_A/Agent00052.png')
        img2 = plt.imread('/Users/wannesvanleemput/OneDrive - UGent/Master 2/Thesis/storage/agent_1/rgb/GDumper_A/Agent00053.png')
        mvs = flownet_mvs = flowlib.read_flow('/Users/wannesvanleemput/OneDrive - UGent/Master 2/Thesis/flow/flowfiles/agent_1/inference/run.epoch-0-flow-field/000051.flo')
        mse = calculate_block_matching_mse(img1, img2, mvs, 7)
        print(mse)
    else:
        if len(sys.argv) != 3:
            print("Usage: 'path to CSV file with mvs' 'path to flo file with mvs'")
        else:
            main(sys.argv[1], sys.argv[2])
