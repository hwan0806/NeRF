#npy 분석
import numpy as np

x = np.loadtxt('./data/nerf_llff_data/ICL_NUIM/poses_bounds_raw_5_depth_fern.txt')

print(x)

print(x.shape) #[10, 17]

print(type(x)) #ndarray

np.save('./data/nerf_llff_data/ICL_NUIM/poses_bounds.npy', x)