import numpy as np

x = np.loadtxt('./data/nerf_llff_data/ICL_NUIM/Rt.txt')

print(x.shape) #[4524, 4]
print(type(x))

#세 개를 한 번에 묶어서 하나의 array를 만들어야 한다. -> [3, 4] matrix
#1, 5, 10, 1, 81, 101, 121, 141, 161, 181
out_list = []
for i in range(0, 200):
    if i % 20 == 0:
        print(x[3*(i+1)-3:3*(i+1),:])
        out_list.append(x[3*(i+1)-3:3*(i+1),:])

print(out_list)
print(len(out_list))

out_list_array = np.array(out_list)

print(out_list_array)
print(out_list_array)
print(out_list_array.shape) #[10, 3, 4]

#[:, :, 0]과 [:, :, 1]의 열을 바꿔줘야 한다. 서로의 자리를 바꿔줘야 한다.
new_list = out_list_array
print(new_list[:, :, 0:1].shape) #[10, 3, 1]
# a = new_list[:,:,0:1]

#out_list_array[:, :, 0] <-> out_list_array[:, :, 1]

new_list = np.concatenate([new_list[:, :, 1:2], new_list[:, :, 0:1], -new_list[:,:,2:3], new_list[:, :, 3:]], axis = 2)
print(new_list == out_list_array)
print(new_list.shape)
# new_list[:,:,0:1] = out_list_array[:,:,1:2]
# new_list[:,:,1:2] = out_list_array[:,:,0:1]
# print(new_list == out_list_array)
# b = new_list[:,:,0:1]
# print(a == b)

# out_list_final = out_list_array.reshape(10, -1)

# np.savetxt('./data/nerf_llff_data/ICL_NUIM/poses_bounds.txt', out_list_final)

new_list_final = new_list.reshape(10, -1)
np.savetxt('./data/nerf_llff_data/ICL_NUIM/poses_bounds.txt', new_list_final)