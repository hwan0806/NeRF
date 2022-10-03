import numpy as np
import os

# Colmap -> GT, Colmap의 world 좌표계 -> A
colmap_poses = np.load('./data/nerf_llff_data/JH_ICL_2/poses_bounds.npy')
# print(colmap_poses.shape) # [10, 17]
T_Aa = colmap_poses[0,:][:-2].reshape(1, 15)
# print(T_Aa.shape) # [15, 1]
T_Ab = colmap_poses[1,:][:-2].reshape(1, 15)

# 3x4 matrix로 만들기
T_Aa = np.concatenate([T_Aa[:,0:5], T_Aa[:,5:10], T_Aa[:,10:]], axis=0)
# print(T_Aa) # [3, 5]
# print(T_Aa.shape)
T_Aa = T_Aa[:,:4]
# print(T_Aa)
last = np.array([0, 0, 0, 1]).reshape(1, 4)
T_Aa = np.concatenate([T_Aa, last], axis=0)
# print(T_Aa)
# [[-0.0208397   0.94940412  0.31336483  3.47282829]
#  [ 0.99632991 -0.00630579  0.08536365 -1.19782783]
#  [ 0.08302061  0.3139937  -0.94578831 -3.95860496]
#  [ 0.          0.          0.          1.        ]]

T_Ab = np.concatenate([T_Ab[:,0:5], T_Ab[:,5:10], T_Ab[:,10:]], axis=0)
# print(T_Aa) # [3, 5]
# print(T_Aa.shape)
T_Ab = T_Ab[:,:4]
# print(T_Aa)
last = np.array([0, 0, 0, 1]).reshape(1, 4)
T_Ab = np.concatenate([T_Ab, last], axis=0)
# print(T_Ab)
# [[-0.0151411   0.96082757  0.27673295  2.79699096]
#  [ 0.99674535 -0.00741276  0.08027307 -1.31532138]
#  [ 0.07917994  0.2770477  -0.95758817 -2.97596019]
#  [ 0.          0.          0.          1.        ]]

T_ab = np.linalg.inv(T_Aa) @ T_Ab
# print(T_ab)
# [[ 0.9999763  -0.00440825 -0.00528813 -0.02139832]
#  [ 0.00420171  0.99925164 -0.03845143 -0.3323576 ]
#  [ 0.00545367  0.0384283   0.99924648 -1.15118726]
#  [ 0.          0.          0.          1.        ]]

# JH, JH의 world 좌표계 -> B, Colmap과의 좌표축을 맞춰줘야 한다. -> poses_bounds.npy 파일로 인해 이미 맞춰졌다.
JH_poses = np.load('./data/nerf_llff_data/JH_ICL_3/poses_bounds.npy')

# print(JH_poses.shape) # [10, 17]

T_Ba = JH_poses[0,:][:-2].reshape(1, 15)
# print(T_Ba.shape) # [1, 15]

# 3x4 matrix로 만들기
T_Ba = np.concatenate([T_Ba[:,0:5], T_Ba[:,5:10], T_Ba[:,10:]], axis=0)
# print(T_Ba.shape)
# print(T_Ba)

T_Ba = T_Ba[:,:4]
# print(T_Ba.shape)
# print(T_Ba)

last = np.array([0, 0, 0, 1]).reshape(1, 4)
T_Ba = np.concatenate([T_Ba, last], axis=0)
# print(T_Ba.shape)
# print(T_Ba)

# T_Bb
T_Bb = JH_poses[1,:][:-2].reshape(1, 15)
# print(T_Bb.shape) # [1, 15]

# 3x4 matrix로 만들기
T_Bb = np.concatenate([T_Bb[:,0:5], T_Bb[:,5:10], T_Bb[:,10:]], axis=0)
# print(T_Bb.shape)
# print(T_Bb)

T_Bb = T_Bb[:,:4]
# print(T_Bb.shape)
# print(T_Bb)

last = np.array([0, 0, 0, 1]).reshape(1, 4)
T_Bb = np.concatenate([T_Bb, last], axis=0)
# print(T_Bb.shape)
# print(T_Bb)

T_JH_ab = np.linalg.inv(T_Ba) @ T_Bb
print(T_JH_ab)
# [[ 9.99987508e-01 -3.77623179e-04  4.47403800e-03 -5.45509067e-03]
#  [ 5.60056022e-04  9.99163668e-01 -4.08579397e-02 -1.80063975e-02]
#  [-4.45476090e-03  4.08600361e-02  9.99153806e-01 -5.24381591e-02]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]

# [[ 0.9999763  -0.00440825 -0.00528813 -0.02139832]
#  [ 0.00420171  0.99925164 -0.03845143 -0.3323576 ]
#  [ 0.00545367  0.0384283   0.99924648 -1.15118726]
#  [ 0.          0.          0.          1.        ]]
