import numpy as np
import cv2
import os
import re

#0, 5, 10, 15, 20, 25, 30, 35, 40, 45
#pose.txt파일 차례대로 불러오기 -> 파일 확장자가 pose.txt로 끝나는 파일만 찾아오기 -> 그 중에서 중간 숫자가 n의 배수 찾아오기
#pose의 확장자와 이름 분리 -> 이름에서 숫자만 분리 -> 숫자 중 5의 배수이고, 45 이하의 파일을 불러오기

path = os.listdir('./data/nerf_llff_data/stairs/seq-01/')
pose_list = [pose for pose in path if pose.endswith('.pose.txt')]

num_list = []
for poses in pose_list:
    txt = os.path.splitext(poses)
    pose = os.path.splitext(txt[0])
    pose = pose[0]
    pose = pose.replace('frame-', '')
    pose = int(pose)
    if pose >= 100 and pose <= 145 and pose % 5 == 0:
        print(pose)
        num_list.append(pose)

num_list = sorted(num_list)

pose_txt = []
for pose in num_list:
    new_pose = './data/nerf_llff_data/stairs/seq-01/frame-' + f'{pose:06d}.pose.txt'
    pose_txt.append(new_pose)

#matrix 불러오기
array_list = []
for pose in pose_txt:
    array = np.loadtxt(pose)
    array = array[:-1,:]
    array_list.append(array)

array_mat = np.array(array_list)

#image height, image width, focal length 합치기 -> [10, 3, 1]
image_height = 480
image_width = 640
focal_length = 585

hwf = np.array([image_height, image_width, focal_length])
hwf = hwf.reshape(1, 3, 1)

hwfs = hwf
for i in range(10 - 1):
    hwfs = np.concatenate([hwfs, hwf], axis = 0)

array_hwf = np.concatenate([array_mat, hwfs], axis = 2)

array_hwf = array_hwf.reshape(10, -1)

#depth 구하기 -> depth 폴더의 이미지를 차례로 가져오기 -> 이 때, depth가 얼만큼의 scaling이 되었는가를 계산해야만 한다.
depth_list = os.listdir('./data/nerf_llff_data/stairs/depth/')
minmax_list = []
for depth in depth_list:
    depth_path = './data/nerf_llff_data/stairs/depth/' + depth
    depth_image = cv2.imread(depth_path)
    depth_image = depth_image.astype('float')
    min = np.array(float(depth_image.min()) * 0.5)
    max = np.array(float(depth_image.max()) * 0.5)
    min = min.reshape(1, 1)
    max = max.reshape(1, 1)
    min_max = np.concatenate([min, max], axis = 1)
    minmax_list.append(min_max)

minmax_arr = np.array(minmax_list)
minmax_arr = minmax_arr.reshape(10, -1)

final_arr = np.concatenate([array_hwf, minmax_arr], axis = 1)
#poses_bounds.txt 파일 저장
np.savetxt('./data/nerf_llff_data/stairs/poses_bounds.txt', final_arr)

#poses_bounds.npy 파일 저장
x = np.loadtxt('./data/nerf_llff_data/stairs/poses_bounds.txt')

np.save('./data/nerf_llff_data/stairs/poses_bounds.npy', x)