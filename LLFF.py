import numpy as np
import os
import sys
import imageio
import skimage.transform

from colmap_wrapper import run_colmap
import colmap_read_models as read_model

realdir = './data/nerf_llff_data/starbucks/'
def load_colmap_data(realdir):
    
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    # print('camdata : ', camdata)
    list_of_keys = list(camdata.keys())
    # print(list_of_keys) #[1]
    cam = camdata[list_of_keys[0]]
    print( 'Cameras', len(cam)) #5

    h, w, f = cam.height, cam.width, cam.params[0]
    # print(h, w, f) #h = 4032, w = 2268, f = 3322.1376
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h,w,f]).reshape([3,1])
    
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    w2c_mats = [] #world to camera -> world 좌표를 camera 좌표로 변환
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    
    names = [imdata[k].name for k in imdata]
    # print(names) #['20220614_013605.jpg', '20220614_013606.jpg', '20220614_013607.jpg', '20220614_013609.jpg', '20220614_013610.jpg', '20220614_013611.jpg', '20220614_013612.jpg', '20220614_013613.jpg', '20220614_013614.jpg', '20220614_013615.jpg']
    print( 'Images #', len(names)) #10
    perm = np.argsort(names)
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
    
    w2c_mats = np.stack(w2c_mats, 0) #COLMAP -> World to Camera : World 좌표를 Camera 좌표로 변환
    c2w_mats = np.linalg.inv(w2c_mats) #Camera to World : Camera 좌표를 World 좌표로 변환
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    # print('poses_shape : ', poses.shape) #[3, 4, 10]
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    # print('poses_shape2 : ', poses.shape) #[3, 5, 10]
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)
    # print('pts3d : ', pts3d)
    # print(type(pts3d)) #dictionary

    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    print('poses_shape3 : ', poses.shape) #[3, 5, 10]
    return poses, pts3d, perm

poses, pts3d, perm = load_colmap_data(realdir)
# print('poses : ', poses) #[3, 5, 10]
# print('pts3d : ', pts3d)
print('perm : ', perm) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -> image index
basedir = './data/nerf_llff_data/starbucks/'

def save_poses(basedir, poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d: #points3D.bin -> 모든 point들을 불러온다.
        print('k :', k) #point의 번호
        pts_arr.append(pts3d[k].xyz) #각 point의 x, y, z pose
        print('poses.shape[-1] : ', poses.shape[-1]) #image의 개수
        cams = [0] * poses.shape[-1]
        print('cams : ', cams) #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        print(type(cams)) #list
        print(pts3d[k].image_ids) #point에 해당하는 image id가 나온다.
        for ind in pts3d[k].image_ids: #pts3d의 image id에 대해서 반복
            # print('ind : ', ind)
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return

            cams[ind-1] = 1 #index로 만들기 위해 image 번호 - 1 = index
        vis_arr.append(cams) #각 World 좌표계 기준의 point들에 의해서 각 point들이 어떠한 이미지에서 발견되었는지에 대한 visibility array

    # print('vis_arr : ', vis_arr)

    pts_arr = np.array(pts_arr) #총 1924개의 각 point들에 의한 [x, y, z] World 좌표계 상의 좌표 값 -> Q. 단위를 모르겠음.
    vis_arr = np.array(vis_arr)
    print( 'Points', pts_arr.shape, 'Visibility', vis_arr.shape ) #Points : [1924, 3], Visibility : [1924, 10] -> 1924개의 World 좌표계를 이루는 모든 ray에 대한 visibility

    # print('pts_arr[:, np.newaxis, :] : ', pts_arr[:, np.newaxis, :].transpose([2, 0, 1]).shape) #[3, 1924, 1]

    #pose -> Camera to World coordinates, Camera 좌표를 World 좌표로 변환
    #pts_arr [1924, 1, 3] -> -([3, 1924, 1] - [3, 1, 10] = [t1, t2, t3]) * [3, 1, 10] = [z1, z2, z3] -> 설마 역이라 -를 붙이는 것인가?
    print('pts : ', (pts_arr[:,np.newaxis,:].transpose([2,0,1]) - poses[:3,3:4,:]).shape) #[3, 1924, 10]


    #Numpy Broadcasting
    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)

    
    
    print(zvals.shape) #[1924, 10] -> 각각의 image 별로 10개의 zvalue가 나온다.
    
    # valid_z = zvals[vis_arr==1] #visibility가 1인 경우의 depth만 고려한다.
    # print(valid_z.shape) #[7151] -> visibility가 1인 경우의 depth
    # print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean() )
    
    save_arr = []
    print(vis_arr.shape) #[1924, 10]
    for i in perm: #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        vis = vis_arr[:, i] 
        print('vis : ', vis.shape) #[1924, ]
        zs = zvals[:, i] #각각의 image에 해당하는 depth
        print('zs : ', zs.shape) #[1924, ]
        zs = zs[vis==1] #visibility가 1인 경우의 z_value만 가져온다.
        print('zs2 : ', zs.shape) #각 이미지에 해당하는 point들의 depth, [481, ]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9) #백분위 0.1 %, 백분위 99.9 %
        print( i, close_depth, inf_depth )
        
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)
    # np.savetxt(os.path.join(basedir, 'poses_bounds_COLMAP.txt'), save_arr)
    # np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)

save_poses(basedir, poses, pts3d, perm)