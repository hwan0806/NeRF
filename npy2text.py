#npy 분석
import numpy as np

x = np.load('./data/nerf_llff_data/LLFF_chess/poses_bounds.npy')

print(x)

print(x.shape) #[20, 17]

print(type(x)) #array

np.savetxt('./data/nerf_llff_data/LLFF_chess/poses_bounds_LLFF_chess.txt', x, fmt = '%f', delimiter = ',')