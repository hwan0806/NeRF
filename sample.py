import numpy as np

array = np.array([[[1, 2, 3, 111],
                  [4, 5, 6, 222],
                  [7, 8, 9, 333]],
                  [[11, 12, 13, 444],
                  [14, 15, 16, 555],
                  [17, 18, 19, 666]]])

print(array.shape) #[2, 3, 4]

array = np.concatenate([-array[:,:,1:2], array[:,:,0:1], -array[:,:,2:3], array[:,:,3:]], axis = 2)

print(array)