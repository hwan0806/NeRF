import numpy as np

#txt 파일에서 array 읽어드리기
x = np.loadtxt('./data/nerf_llff_data/LLFF_chess/poses_bounds_JH1.txt')
#array의 값들을 소수점 아래 6자리만 표현하기
x = np.savetxt('./data/nerf_llff_data/LLFF_chess/poses_bounds_JH1_refined.txt', x, fmt = '%.6f')
