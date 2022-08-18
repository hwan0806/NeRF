#1. 좌표축을 변환하지 말아보자.
import numpy as np

x = np.loadtxt('./data/nerf_llff_data/ICL_NUIM/Rt.txt')

print(x.shape) #[4524, 4]
print(type(x)) #ndarray

#image0.png
print(x[0,:])
print(x[1,:])
print(x[2,:]) #x[0:3,:] -> 하나의 [R|t] matrix를 이루어야 한다. image0.png의 matrix를 이룬다.
array0 = x[0:3,:]
print('array0 : ', array0)
print(array0.shape)
print(type(array0))

#image1.png
print(x[3,:])
print(x[4,:])
print(x[5,:]) #x[3:6,:] -> image1.png의 matrix를 이룬다.
array1 = x[3:6,:]
print('array1 : ', array1)
print(array1.shape) #[3, 4]
print(type(array1)) #ndarray

#image2.png
print(x[6,:]) #x[image i x 3: image (i+1) x 3]
print(x[7,:])
print(x[8,:]) #image2.png의 matrix를 이룬다.
array2 = x[6:9,:]
print('array2 : ', array2)
print(array2.shape)
print(type(array2))

#image0.png, image5.png, image10.png, image15.png, image20.png, image25.png, image30.png, image35.png, image40.png, image45.png
list = []
for i in range(0, 45 + 1):
    if i % 5 == 0: #바꿔줘야 한다.
        trans_mat = x[3*i:3*(i+1),:]
        print(i, ' ', trans_mat.shape)
        list.append(trans_mat)

print(list)

transformation_matrix = np.array(list)
print(transformation_matrix)
print(transformation_matrix.shape) #[10, 3, 4]


#각 matrix의 좌표축을 변환하여 줘야 한다.
#첫 번째 열과 두 번째 열을 서로 바꾸고, 마지막 z축에 대해서 -의 값을 취해야 한다.
#[:,1,:] + [:,0,:] - [:,2,:] + [:,3,:]
# new_transformation_matrix = np.concatenate([transformation_matrix[:,:,1:2], transformation_matrix[:,:,0:1], -transformation_matrix[:,:,2:3], transformation_matrix[:,:,3:]], axis = 2)
# print('new : ', new_transformation_matrix)

final_transformation_matrix = transformation_matrix.reshape(10, -1)
print(final_transformation_matrix)

# np.savetxt('./data/nerf_llff_data/ICL_NUIM/poses_bounds_raw_5.txt', final_transformation_matrix)

#depth 값 추출 -> depth max and min
import cv2
import os

img = cv2.imread('./data/nerf_llff_data/ICL_NUIM/depth/0.png')

print(img.shape) #[480, 640, 3] Q. 채널의 수도 고려해줘야 하는가?

img_array = img.reshape(-1)

print(img_array.shape) #[92160,]

img_array = img_array.astype('float')
print(np.max(img_array)) #66
print(np.min(img_array)) #32

#채널을 따로 고려하였을 때,

img_array2 = img.reshape(-1, 3)
print(np.max(img_array2)) #66
print(np.min(img_array2)) #32

array_list = []
#image 10개를 반복해서 가져와서, depth의 min과 max를 추출한다.
for i in range(10):
    if i == 0: #image0.png
        array = cv2.imread('./data/nerf_llff_data/ICL_NUIM/depth/{}.png'.format(i))
    else: #10, 20, 30, 40, 50, 60, 70, 80, 90
        array = cv2.imread('./data/nerf_llff_data/ICL_NUIM/depth/{}.png'.format(i * 5))
    array = array.reshape(-1, 3)
    print('image_i.png의 max depth : ', np.max(array), 'image_i.png의 min depth : ', np.min(array))
    max = np.array(np.max(array))
    min = np.array(np.min(array))
    max = max.reshape(1, 1)
    print(max)
    print(max.shape)
    min = min.reshape(1, 1)
    min_max = np.concatenate([max, min], axis = 1)
    print(min_max) #[[64, 32]]
    print(min_max.shape)
    min_max = min_max.reshape(-1)
    print(min_max) #[66 32]
    print(min_max.shape) #[2,]
    array_list.append(min_max)
    # break
print(array_list)

# #txt to npy
