import numpy as np
import os
import cv2

#depth -> focal length 고려

#Rt.txt에서 5개씩의 image의 간격으로 pose matrix를 만들기 -> [3, 4]
rt = np.loadtxt('./data/nerf_llff_data/ICL_NUIM2/Rt.txt')

#500, 505, 510, 515, 520, 525, 530, 535, 540, 545
rt_arr_list = []

for i in range(500, 545 + 1):
    if i % 5 == 0:
        rt_arr = rt[i*3:(i+1)*3]
        rt_arr_list.append(rt_arr)

#pose matrix를 동차 좌표계로 만들기 -> [4, 4]
rt_array_list = []
bottom = np.array([[0, 0, 0, 1]])
for i in range(10):
    rt_array = np.concatenate([rt_arr_list[i], bottom], 0)
    rt_array_list.append(rt_array)

w2c_mats = np.array(rt_array_list)

#inverse 
w2c_mats = np.stack(w2c_mats, 0)
c2w_mats = np.linalg.inv(w2c_mats)

#좌표축 변환 : [r, u, t] -> [-u, r, -t]
image_height = 480 #pixel 단위의 image 크기
image_width = 640 #pixel 단위의 image 크기
focal_length = 481.20 #pixel 단위의 focal length [pixel/meter]

hwf = np.array([image_height, image_width, focal_length]).reshape(3, 1)

poses = c2w_mats[:, :3, :4]
hwf = hwf.reshape(1, 3, 1)
hwfs = hwf
for i in range(10 - 1):
    hwfs = np.concatenate([hwfs, hwf], axis = 0)

poses = np.concatenate([poses, hwfs], 2)
poses = np.concatenate([-poses[:, :, 1:2], poses[:, :, 0:1], -poses[:, :, 2:3], poses[:, :, 3:]], axis = 2)

poses = poses.reshape(10, -1)

#depth -> 16bit로 받아오기
depth_list = []
for depth in os.listdir('./data/nerf_llff_data/ICL_NUIM2/depth'):
    depth_list.append(depth)

depth_list = sorted(depth_list) #오름차순으로 정렬

#depth scaling -> 먼저 255로 나누기
depth_array_list = []
for depth in depth_list:
    depth_image = cv2.imread('./data/nerf_llff_data/ICL_NUIM2/depth/' + depth, -1)
    min_max = np.array([depth_image.min() * 481.20 / 5000, depth_image.max() * 481.20 / 5000]) #m -> pixel
    depth_array_list.append(min_max)
    print(min_max)
depth_array = np.array(depth_array_list)

npy_file = np.concatenate([poses, depth_array], axis = 1)

txt = np.savetxt('./data/nerf_llff_data/ICL_NUIM2/poses_bounds_3.txt', npy_file)

npy = np.loadtxt('./data/nerf_llff_data/ICL_NUIM2/poses_bounds_3.txt', txt)

np.save('./data/nerf_llff_data/ICL_NUIM2/poses_bounds.npy', npy)