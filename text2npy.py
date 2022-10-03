#npy 분석
import numpy as np

x = np.loadtxt('./data/nerf_llff_data/JH_ICL_4/JH_poses_bounds.txt')

print(x)

print(x.shape) #[10, 17]

print(type(x)) #ndarray

np.save('./data/nerf_llff_data/JH_ICL_4/poses_bounds.npy', x)
