import numpy as np
import os
import re

# poses_bounds.npy 파일 불러오기

# relative pose 비교해보기 
# colmap -> T_Wa T_Wb -> T_ab = (T_Wa)^(-1) x T_Wb
colmap_poses = np.load('./data/nerf_llff_data/comparison/poses_bounds.npy')
# print(colmap_poses)

# np.savetxt('./data/nerf_llff_data/comparison/colmap_poses_bounds.txt', colmap_poses)

T_Wa = colmap_poses[0][:-2]
# print(T_Wa)
T_Wa = np.array(T_Wa).reshape([1,15])
# print(T_Wa.shape)
T_Wa = np.concatenate([T_Wa[:,:5],T_Wa[:,5:10],T_Wa[:,10:15]], axis=0)
# print(T_Wa.shape)
# print(T_Wa) # [3, 5]
T_Wa = T_Wa[:,:4]
# print(T_Wa)
last = np.array([0., 0., 0., 1.]).reshape([1,4])
# print(last.shape)
T_Wa = np.concatenate([T_Wa, last], axis=0)
# print(T_Wa)

T_Wb = colmap_poses[1][:-2]
# print(T_Wa)
T_Wb = np.array(T_Wb).reshape([1,15])
# print(T_Wa.shape)
T_Wb = np.concatenate([T_Wb[:,:5],T_Wb[:,5:10],T_Wb[:,10:15]], axis=0)
# print(T_Wa.shape)
# print(T_Wa) # [3, 5]
T_Wb = T_Wb[:,:4]
# print(T_Wa)

T_Wb = np.concatenate([T_Wb, last], axis=0)
# print(T_Wb)

# T_ab 
T_ab = np.linalg.inv(T_Wa) @ T_Wb
# print(T_ab) # colmap

# 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95 파일에 해당하는 R|t 가져오기

#각 image에 해당하는 pose 값 불러오기
files = [file for file in os.listdir('./data/nerf_llff_data/comparison/seq-01') if file.endswith('.txt')]

#files 중에서 확장자를 제거한 숫자가 0 ~ 95 + 1 중에서 5의 배수인 것만 추출한다.

file_list = []
for file in files:
    number = re.findall("\d+", file)
    number = np.array(number)
    number = int(number.item())
    if number >= 0 and number <= 95 and number % 5 == 0:
        file_list.append(file)

file_list = sorted(file_list)
# print(file_list)

pose_list = []
for file in file_list:
    pose = np.loadtxt('./data/nerf_llff_data/comparison/seq-01/' + file)
    pose = pose[:3, :]
    pose_list.append(pose)

pose_arr = np.array(pose_list)

#image height, image width, focal length
image_height = 480
image_width = 640
focal_length = 585

hwf = np.array([image_height, image_width, focal_length])
hwf = hwf.reshape([1, 3, 1])
hwfs = hwf
for i in range(20 - 1):
    hwfs = np.concatenate([hwfs, hwf], 0)

pose_hwf = np.concatenate([pose_arr, hwfs], axis = 2)

# # [r, -u, t] -> [-u, r, -t]
# pose_llff = np.concatenate([pose_hwf[:,:,1:2], pose_hwf[:,:,0:1], -pose_hwf[:,:,2:3], pose_hwf[:,:,3:]], axis = 2)

# # [r, -u, -t] -> [-u, r, -t]
# pose_llff = np.concatenate([pose_hwf[:,:,1:2], pose_hwf[:,:,0:1], pose_hwf[:,:,2:3], pose_hwf[:,:,3:]], axis = 2)

# [-r, u, t] -> [-u, r, -t]
pose_llff = np.concatenate([-pose_hwf[:,:,1:2], -pose_hwf[:,:,0:1], -pose_hwf[:,:,2:3], pose_hwf[:,:,3:]], axis = 2)

print(pose_llff.shape) # [20, 3, 5]
JHposes = pose_llff[:,:,:4]
last = np.array([0., 0., 0., 1]).reshape(1, 4)
T_JH_Wa = JHposes[0]
T_JH_Wa = np.concatenate([T_JH_Wa, last], axis=0)
T_JH_Wb = JHposes[1]
T_JH_Wb = np.concatenate([T_JH_Wb, last], axis=0)
# print(T_JH_Wa)
# print(T_JH_Wb)
# print(pose_arr)

T_JH_ab = np.linalg.inv(T_JH_Wa) @ T_JH_Wb
print(T_ab)
print(T_JH_ab)

# pose_hwf = pose_hwf.reshape(10, -1)
# print(pose_hwf.shape)
# print(pose_hwf)

# np.savetxt('./data/nerf_llff_data/comparison/JH_poses_bounds.txt', pose_hwf)