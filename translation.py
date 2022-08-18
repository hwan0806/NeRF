import numpy as np

#pts_arr -> [5, 1, 3] -> point, 1, [x, y, z]
#pose -> [2, 3, 5] -> [3, 5, 2]
pts_arr = np.array([['x1', 'y1', 'z1'],
                    ['x2', 'y2', 'z2'],
                    ['x3', 'y3', 'z3'],
                    ['x4', 'y4', 'z4'],
                    ['x5', 'y5', 'z5']])
print(pts_arr.shape) #[5, 3]
# pts_arr = pts_arr[:,np.newaxis,:].transpose([2, 0, 1]) #[3, 5, 1]
# print(pts_arr.shape)

poses = np.array([[['R11', 'R12', 'R13', 't11', 'h'],
                  ['R14', 'R15', 'R16', 't12', 'w'],
                  ['R17', 'R18', 'R19', 't13', 'f']],
                  [['R21', 'R22', 'R23', 't21', 'h'],
                  ['R24', 'R25', 'R26', 't22', 'w'],
                  ['R27', 'R28', 'R29', 't23', 'f']]])

print(poses.shape) #[2, 3, 5]
poses = poses.transpose([1, 2, 0])
print(poses.shape) #[3, 5, 2]

print('1', pts_arr[:,np.newaxis,:].transpose([2, 0, 1])) #[3, 5, 1]

print('2', poses[:3,3:4,:])

print('3', poses[:3,2:3,:])


# #pose -> World to Camera coordinates, camera 좌표를 world 좌표로 변환
# #pts_arr [1924, 1, 3] -> -([3, 1924, 1] - [3, 1, 10] = [t1, t2, t3]) * [3, 1, 10] = [z1, z2, z3]
# print('pts : ', (pts_arr[:,np.newaxis,:].transpose([2,0,1]) - poses[:3,3:4,:]).shape) #[3, 1924, 10]
# #Numpy Broadcasting
# zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
