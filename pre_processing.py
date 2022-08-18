import numpy as np

x = np.array([[['R1', 'R2', 'R3', 'T1', 'HEIGHT'],
             ['R4', 'R5', 'R6', 'T2', 'WIDTH'],
             ['R7', 'R8', 'R9', 'T3', 'FOCAL']],
             [['r1', 'r2', 'r3', 't1', 'height'],
             ['r4', 'r5', 'r6', 't2', 'width'],
             ['r7', 'r8', 'r9', 't3', 'focal']]]) #[2, 3, 5]

x_out = x.transpose([1,2,0]) #[3, 5, 2]

print(x_out)
print(x_out.shape) #[3, 5, 2]

poses = np.concatenate([x_out[:, 1:2, :], x_out[:, 0:1, :], x_out[:, 2:, :]], 1) #맨 앞과 중간의 위치를 바꿨다.

print(poses)
print(poses.shape) #[3, 5, 2]