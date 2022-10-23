import argparse
import os
import sys

import numpy as np



# cam (1, 3)
# global_orient (1, 3)
# body_pose (1, 69)
# smpl_betas (1, 10)
# smpl_thetas (1, 72)
# center_preds (1, 2)
# center_confs (1, 1)
# cam_trans (1, 3)
# verts (1, 6890, 3)
# joints (1, 71, 3)
# pj2d_org (1, 71, 2)


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, default='../dataset/wild/monocular/video_results.npz')
parser.add_argument('-o', '--output', type=str, default='../dataset/wild/monocular/novel_poses/novel_poses.npy')

args = parser.parse_args(sys.argv[1:])

npz_path = args.path
if os.path.exists(npz_path):
    print('find video_results.npz in ', npz_path)
else:
    print(npz_path, ' not found')




data = np.load(npz_path., allow_pickle=True)
data_dict = data['results'][()]
# print(data['results']())
pose_output = []
n_frames = len(data_dict.keys())
print('n_frames ', n_frames)
for i, (k, v) in enumerate(data_dict.items()):
    # print(v['global_orient'][0].dtype)
    # v['global_orient'][0] += 180. * np.pi / 180.
    pose_output.append(np.concatenate([v['global_orient'][0], v['body_pose'][0]], axis=0))

# np.save('novel_poses.npy', pose_output)
np.save(args.output, pose_output)
print('save novel_poses.npy in ', args.output)

# novel = np.load('novel_poses.npy')
# print(novel.shape)

# print(pose_output[0])
