import numpy as np
import cv2 as cv
import os
# Q = (q_0, q_1, q_2, q_3) = (q_0, q)
def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

# poses_bounds.npy 파일 만들기

# Depth -> 16bit -> 5000으로 나눠야 metric 단위를 만들 수 있다.

# Pose -> camera to world coordinate, global coordinate -> 번호, tx, ty, tz, (qx, qy, qz), qw
# tx, ty, tz -> translation
# qx, qy, qz, qw -> rotation matrix
# focal length -> 525

poses = np.loadtxt('./data/nerf_llff_data/JH_ICL_1/poses.txt')
# print(poses.shape) # [880, 8]

poses_list = []
# 650, 655, 660, 670, 675, 680, 685, 690, 695
for i in range(650, 695 + 1):
    if i % 5 == 0:
        # print(i)
        rotation = poses[i, 4:]
        # print(rotation) # [-0.016716  -0.72753   -0.0332406  0.685067 ]
        # print(rotation.shape) # [0., 0., 0., 1] -> [1, 0., 0., 0.], [4,]
        # print(rotation[:-1].shape) # [3,]
        quaternion = np.concatenate([np.array(rotation[-1]).reshape(1,), rotation[:-1]], axis=0)
        # print(quaternion)
        translation = poses[i, 1:4]
        # print(translation) # [ 0.603922  -0.0637288 -1.40411  ]
        # print(translation.shape) # [3,]
        # print(translation) # [0., 0., -2.25]
        rotation_matrix = quaternion_rotation_matrix(quaternion)
        # print(rotation_matrix) # [3, 3]
        transformation_matrix = np.concatenate([rotation_matrix, np.array(translation).reshape(3,1)], axis=1)
        # print(transformation_matrix)
        poses_list.append(transformation_matrix)
        
# print(poses_list)
poses_arr = np.array(poses_list)
# print(poses_arr.shape) # [10, 3, 4]

height = 480
width = 640
focal = 525
hwf = np.array([height, width, focal]).reshape(1,3,1)

# 좌표축 변환, [down, right, backwards] = [-u, r, -t]
# [r, -u, t] -> [-u, r, -t]
poses_arr = np.concatenate([poses_arr[:,:,1:2], poses_arr[:,:,0:1], -poses_arr[:,:,2:3], poses_arr[:,:,3:]], axis=-1)
# print(poses_arr.shape) # [10, 3, 4]
hwfs = hwf
for i in range(10 - 1):
    hwfs = np.concatenate([hwf, hwfs], axis=0)
# print(hwfs.shape) # [10, 3, 1]

poses_fin = np.concatenate([poses_arr, hwfs], axis=-1)
# print(poses_fin) # [10, 3, 5]

depth_list = []
# Depth image -> 16bit -> 5000으로 나누기
# 650, 655, 660, 665, 670, 675, 680, 685, 690, 695
for i in range(130, 139+1):
    file_name = os.path.join('./data/nerf_llff_data/JH_ICL_3/JH_depth/', '{}.png'.format(5*i))
    depth_image = cv.imread(file_name, -1)
    depth_image = depth_image / 5000
    # print(file_name)
    # print(depth_image)
    close_depth, inf_depth = np.percentile(depth_image, 0.1), np.percentile(depth_image, 99.9)
    # print(close_depth, inf_depth)
    depth_list.append([close_depth, inf_depth])

depth_arr = np.array(depth_list)
# print(depth_arr.shape) # [10, 2]
# print(depth_list)

# pose + depth
poses_fin = poses_fin.reshape(10, -1)
poses_depth = np.concatenate([poses_fin, depth_arr], axis=-1)

print(poses_depth.shape) # [10, 17]

np.savetxt('./data/nerf_llff_data/JH_ICL_3/JH_poses_bounds.txt', poses_depth)

a = np.loadtxt('./data/nerf_llff_data/JH_ICL_3/JH_poses_bounds.txt')

np.save('./data/nerf_llff_data/JH_ICL_3/poses_bounds.npy', a)