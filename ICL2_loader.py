import numpy as np
import cv2
import os

n = 5 #image 간격
transform_axis = True #[r, u, t] -> [-u, r, -t]
image_height = 480
image_width = 640
focal_length = 481.20

arr = np.loadtxt('./data/nerf_llff_data/ICL_NUIM2/Rt.txt')

list = []
#500.png ~ 545.png
for i in range(500, 500 + 9 * n + 1):
    if i % n == 0:
        mat = arr[3*i:3*(i+1),:]

        #rotation
        rot = mat[:,:-1]
        trans = mat[:,-1:]

        inverse = np.linalg.inv(rot)
        transform = np.concatenate([inverse, trans], axis = 1)
        list.append(transform)

transformation = np.array(list)

#좌표축을 변환한다면,
if transform_axis == True:

    transformation = np.concatenate([-transformation[:,:,1:2], transformation[:,:,0:1], -transformation[:,:,2:3], transformation[:,:,3:]], axis = 2) 

#image height, image width, focal length 집어넣기
add_arr = np.array([image_height, image_width, focal_length])
add_arr = add_arr.reshape(1, 3, 1) #[3, 1]

add_matrix = add_arr

for i in range(10 - 1):
    add_matrix = np.concatenate([add_matrix, add_arr], axis = 0)

#[10, 3, 4] + [10, 3, 1]
trans_array = np.concatenate([transformation, add_matrix], axis = 2)

final_transformation = trans_array.reshape(10, -1)

#depth 값 추출 -> depth max and min
arr_list = []
for i in range(0, 10):

    if i == 0:
        image = cv2.imread('./data/nerf_llff_data/ICL_NUIM2/depth/{}.png'.format(500 + i), -1) #16bit로 depth 받아오기

    if i != 0:
        image = cv2.imread('./data/nerf_llff_data/ICL_NUIM2/depth/{}.png'.format(500 + i * n), -1)

    print(image)
    image = image.reshape(-1, 3)
    max = (np.max(image)) / 255
    min = (np.min(image)) / 255
    print(max, min)
    max = max.reshape(1, 1)
    min = min.reshape(1, 1)
    min_max = np.concatenate([min, max], 1)
    min_max = min_max.reshape(-1)
    
    arr_list.append(min_max)

depth = np.array(arr_list)
final_array = np.concatenate([final_transformation, depth], axis = 1)

if transform_axis == True:
    np.savetxt('./data/nerf_llff_data/ICL_NUIM2/poses_bounds_w2c_inverse_255_{}.txt'.format(n), final_array)
    txt = np.loadtxt('./data/nerf_llff_data/ICL_NUIM2/poses_bounds_w2c_inverse_255_{}.txt'.format(n))
    np.save('./data/nerf_llff_data/ICL_NUIM2/poses_bounds.npy', txt)
else:
    np.savetxt('./data/nerf_llff_data/ICL_NUIM2/poses_bounds_raw_inverse_{}.txt'.format(n), final_array)
    txt = np.loadtxt('./data/nerf_llff_data/ICL_NUIM2/poses_bounds_raw_inverse_{}.txt'.format(n))
    np.save('./data/nerf_llff_data/ICL_NUIM2/poses_bounds.npy', txt)
    