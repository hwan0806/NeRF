#npy 분석
import numpy as np

x = np.load('./data/nerf_llff_data/JH_ICL_2/poses_bounds.npy')

print(x)

print(x.shape) #[20, 17]

print(type(x)) #array

np.savetxt('./data/nerf_llff_data/JH_ICL_2/LLFF_poses_bounds.txt', x)