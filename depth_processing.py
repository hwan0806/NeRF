import cv2
import os
import numpy as np

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
    if i == 0:
        array = cv2.imread('./data/nerf_llff_data/ICL_NUIM/depth/{}.png'.format(i))
    else:
        array = cv2.imread('./data/nerf_llff_data/ICL_NUIM/depth/{}.png'.format(i))
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