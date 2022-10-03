import cv2
import os
import numpy as np
import re

#각 image에 해당하는 pose 값 불러오기 -> 950, 955, 960, 965, 970, 975, 980, 985, 990, 995
files = [file for file in os.listdir('./data/nerf_llff_data/chess/seq-01') if file.endswith('.txt')]

#files 중에서 확장자를 제거한 숫자가 0 ~ 45 + 1 중에서 5의 배수인 것만 추출한다.
file_list = []
for file in files:
    number = re.findall("\d+", file)
    number = np.array(number)
    number = int(number.item())
    if number >= 950 and number <= 995 and number % 5 == 0:
        file_list.append(file)

file_list = sorted(file_list)

pose_list = []
for file in file_list:
    pose = np.loadtxt('./data/nerf_llff_data/chess/seq-01/' + file)
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
for i in range(10 - 1):
    hwfs = np.concatenate([hwfs, hwf], 0)

pose_hwf = np.concatenate([pose_arr, hwfs], axis = 2)

#[r, -u, t] -> [-u, r, -t]
pose_llff = np.concatenate([pose_hwf[:,:,1:2], pose_hwf[:,:,0:1], -pose_hwf[:,:,2:3], pose_hwf[:,:,3:]], axis = 2)

pose_hwf = pose_hwf.reshape(10, -1)
# print(pose_hwf.shape)
print(pose_hwf)

# #depth -> 16bit로 받아오기 -> 1000으로 나누어, meter 단위로 맞추기
# depth_list = []
# for i in range(950, 995 + 1):
#     if i % 5 == 0:
#         depth = cv2.imread('./data/nerf_llff_data/chess/depth/' + 'frame-{:06d}.depth.png'.format(i), cv2.CV_32F) # 16bit
#         depth = depth.astype(float)
#         min = np.min(depth)
#         max = np.max(depth)
#         # print(min, max) # 0 65.535
#         depth_list = []
#         # print(depth.shape) # [480, 640]
#         depth = depth.reshape(-1)
#         # print(depth.shape)
#         # print(depth)
#         for i in range(depth.shape[0]):
#             if depth[i] != 0.0 and depth[i] != 255.0:
#                 depth_list.append(depth[i])
        
#         close_depth, inf_depth = np.percentile(depth_list, 0.1), np.percentile(depth_list, 99.9)
#         print(close_depth, inf_depth)

#         # print(depth.shape)
#         # # for i in 
#         # if min == 0.0:
#         #     depth = depth.reshape(-1)
#         #     for i in range(len(depth)):
#         #         if depth[i] != 0 or depth[i] != 65.535:
                    
#         #     # depth = depth.reshape(-1)
#         #     min_list = sorted(depth)
#         #     for i in range(depth.shape[0]): 
#         #         # depth list에서 0인 값 모두 제외
#         #         if min_list[i] != 0:
#         #             min = min_list[i]
#         #             break
#         # if max == 65.535: #큰 값으로 정렬해서, 65535의 값이 나오지 않은 값을 추출하기
#         #     depth = depth.reshape(-1)
#         #     max_list = sorted(depth, reverse = True)
#         #     for i in range(depth.shape[0]):
#         #         if max_list[i] != 65535:
#         #             max = max_list[i]
#         #             break

#         # break
# #         # print(min, max)
# #         # 제일 큰 값은 빼기 -> 즉, 2번째 큰 값을 구해서 넣기
# #         min_max = np.array([close_depth, inf_depth]).reshape(1, 2)
# #         depth_list.append(min_max)
# # depth_arr = np.array(depth_list).reshape(10, -1)

# # final_arr = np.concatenate([pose_hwf, depth_arr], axis = 1)

# # txt = np.savetxt('./data/nerf_llff_data/chess/poses_bounds_2.txt', final_arr)
# # x = np.loadtxt('./data/nerf_llff_data/chess/poses_bounds_2.txt')
# # np.save('./data/nerf_llff_data/chess/poses_bounds.npy', x)
