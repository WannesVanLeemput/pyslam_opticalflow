import numpy as np
from scipy.spatial.transform.rotation import Rotation as R

def fix_orb(input_file, output_file):
    out = open(output_file)
    with open(input_file, 'r') as f:
        line = True
        while line:
            line = f.readline()
            timestamp, tx, tz, ty, rx, ry, rz, rw = line[:-1].split(' ')
            pose = np.array([tx, ty, tz, rx, ry, rz, rw]).astype(np.float32)
            out_line = f"{timestamp} {pose[0]} {pose[1]} {-pose[2]} {pose[4]} {pose[5]} {pose[3]} {pose[6]}\n"
            current_pose = R.from_quat()


if __name__ == '__main__':
    #pose_file = open("/home/wannes/storage/agent_1/proposed method.txt", 'r')
    out_file = open("/home/wannes/storage/rgbd_dataset_freiburg1_desk/proposed method_fixed.txt", 'w')
    with  open("/home/wannes/storage/rgbd_dataset_freiburg1_desk/poses_desk_new.txt", 'r') as f:
        line = True
        while line:
            line = f.readline()
            timestamp, tx, tz, ty, rx, ry, rz, rw = line[:-1].split(' ')
            pose = np.array([tx, ty, tz, rx, ry, rz, rw]).astype(np.float32)
            out_line = f"{timestamp} {-pose[0]} {pose[1]} {pose[2]} {-pose[4]} {pose[5]} {-pose[3]} {pose[6]}\n"  # for pyslam output
            #out_line = f"{timestamp} {pose[0]} {pose[1]} {-pose[2]} {pose[4]} {pose[5]} {pose[3]} {pose[6]}\n"
            out_file.write(out_line)