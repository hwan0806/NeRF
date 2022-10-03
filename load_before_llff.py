import numpy as np
import os, imageio

########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

# 압축
def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors: 
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir): 
            needtoload = True 
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload: 
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])] 
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int): 
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else: 
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name) 
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
    
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')

# basedir = './data/nerf_llff_data/fern'

# load data -> poses_bounds.npy 파일 가져오기
# Custom Dataset -> image + label(poses + bds)
# camera to world coordinate
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy')) # [20, 17]
    # print(poses_arr.shape)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # [3, 5, 20]
    # print(poses.shape)
    bds = poses_arr[:, -2:].transpose([1,0]) # [2, 20]
    # print(bds.shape)
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]


    sh = imageio.imread(img0).shape
    sfx = ''

    # 압축
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    # print('imgdir : ', imgdir)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape # [3024, 4032, 3]
    # print(sh)
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1]) # image height, width resize
    poses[2, 4, :] = poses[2, 4, :] * 1./factor # focal length

    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles] # image -> 255로 나눠 normalize
    imgs = np.stack(imgs, -1)
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    
    return poses, bds, imgs # [3, 5, 20], [2, 20], [height, width, 3, 20] -> 3 : rgb

# poses, bds, imgs = _load_data(basedir) #poses -> [3, 5, 20] / bds -> [2, 20] / img -> [3024, 4032, 3, 20]
# print(poses.shape) #[3, 5, 20]
# print(bds.shape) #[2, 20]
# print(imgs.shape) #[3024, 4032, 3, 20]

def normalize(x):
    return x / np.linalg.norm(x)

# 카메라 위치 : Eye vector, 카메라 타겟 : Look vector, 카메라 Up : up vector
# 1) Eye vector와 Look vector를 통해 z축을 구한다. -> Z축 벡터 = normalize(Look - Eye) 
# 2) z축과 임시 축인 up 벡터의 연산으로 x축을 구한다. -> X축 벡터 = up x Z축 벡터
# 3) 구한 z, x축의 연산을 통해 y축을 구한다. -> Y축 벡터 = Z축 벡터 x X축 벡터
# 4) 각자 구한 x, y, z를 normalize하면 view matrix의 rotation 부분이 된다.
def viewmatrix(z, up, pos): # z = vec2 [R3, R6, R9], up = [R2, R5, R8], center = [t1, t2, t3] 평균

    vec2 = normalize(z) # Z축 벡터
    vec1_avg = up # up 벡터
    vec0 = normalize(np.cross(vec1_avg, vec2)) # up x Z축 벡터 = X축 벡터
    vec1 = normalize(np.cross(vec2, vec0)) # Z축 벡터 x X축 벡터 = Y축 벡터

    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

# 20개 image의 average에 대한 camera to world matrix, new origin
def poses_avg(poses): # poses -> [r, u, -t], [20, 3, 5]
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0) # 20개 image의 translation에 대한 평균 -> [t1, t2, t3]
    # print('1 : ', center.shape) #[3,]
    vec2 = normalize(poses[:, :3, 2].sum(0)) # 20개 image의 [R3, R6, R9]에 대해 normalize -> Z축에 대한 값
    # print('2 : ', vec2.shape)
    up = poses[:, :3, 1].sum(0) # 20개 image의 [R2, R5, R8]에 대해 normalize -> Y축에 대한 값
    # print('3 : ', up.shape) #[3,]

    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w # new origin

# sprial path로 새로운 pose 생성 -> c2w : recentered pose의 view matrix(기존 view matrix와는 다른 새로운 view matrix -> 새로운 pose)
# for making video
def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N): # zrate = 0.5, rots = 2, N = 120
    render_poses = []
    # print('rads : ', rads) # rads -> 20개 image의 translation의 90%, 비교적 멀리 떨어져 있는 translation -> radius
    rads = np.array(list(rads) + [1.])
    # print('rads : ', rads) # [0.39451256 0.12918354 0.04078128 1.        ]
    hwf = c2w[:,4:5]
    # print(np.linspace(0., 2. * np.pi * rots, N+1)) # [121, ]
    # print(np.linspace(0., 2. * np.pi * rots, N+1)[:-1].shape) # [120, ] 마지막 제외
    # np.linspace(구간 시작점, 구간 끝점, 구간 내 숫자 개수)
    # theta = 12.56
    # a = np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]*rads)
    # b = c2w[:3,:4]
    # print('a, b : ', a.shape, b.shape) # [4, ], [3, 4]
    # np.dot(b, a) -> n차원 x m차원 = [3, 4] x [4, ] -> [3, ]
    # c = a * rads 
    # print('c : ', c)

    # 두 바퀴를 회전하는 sprial path를 생성
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]: # 12.56... / 120 = [0, ..., 12.56...] radian 단위 -> degree 단위 (180 / pi)
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) # Look vector -> target의 위치, rads -> 반지름
        # print('c : ', c.shape) # [3, ]
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.]))) # Look vector(target 위치) - eye vector(현재 camera 위치)

        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1)) # 새로운 pose에서의 camera to world matrix
    
    # print(render_poses) # [120, 3, 5] 
    return render_poses
    
# recenter_poses : 20개 image의 average pose에서 본 world 좌표계 상의 물체를 표현
# world 좌표계 -> 20개 image의 average pose
def recenter_poses(poses): # pose -> [r, u, -t], c2w(camera 좌표를 world 좌표로 변환)
    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])

    c2w = poses_avg(poses) # 20개 image의 average pose에서의 world to camera matrix
    # print('2 : ', c2w.shape) # [3, 5]
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    # print('2 : ', c2w.shape) # [4, 4]
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)
    # print('3 : ', poses.shape) # [20, 4, 4]

    poses = np.linalg.inv(c2w) @ poses

    poses_[:,:3,:4] = poses[:,:3,:4] # pose는 newly defined pose origin에 의해서 재정렬된다.
    poses = poses_

    return poses 
    
def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds

basedir = './data/nerf_llff_data/JH_ICL_2'
# Dataloader
def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):

    poses, bds, imgs = _load_data(basedir, factor=factor)
    # print(poses.shape) #[3, 5, 20] 
    # print(bds.shape) #[2, 20]
    # print(imgs.shape) #[378, 504, 3, 20]

    # poses_bounds.npy -> [r, -u, t] => [-u, r, -t]
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1) # [-u, r, -t] => [r, u, -t] (World 좌표계와 같은 좌표축 변환)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32) # poses -> camera to world coordinates
    # print(poses.shape) # [3, 5, 20] -> [20, 3, 5]
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    # print(imgs.shape) # [378, 504, 3] -> [20, 378, 504, 3]
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    # print(bds.shape) # [2, 20] -> [20, 2]

    # poses, bds refinement
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor) 
    poses[:,:3,3] *= sc 
    bds *= sc 

    # poses -> [r, u, -t]

    if recenter: # recenter = True -> 20개 image의 average pose에서 본 object를 표현하기 위한 transformation matrix, for using NDC space
        poses = recenter_poses(poses) # 고정된 위치(average pose) -> 20개 image의 average pose에 대한 view matrix를 취한 pose
        print("after recenter")
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)
        print("after spherify")
    
    else: # spherify = False, poses = recentered poses
        # recentered poses : 20개 image에서의 average pose C(center), T_CA -> Center 좌표계의 좌표로 변환
        c2w = poses_avg(poses) # recentered pose(20개)의 average pose
  
        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0)) # Y -> up vector는 고정, X와 Z 회전하며 새로운 rendering path 생성

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth)) # the depth that the spiral poses look at
        focal = mean_dz

        # Get radii for spiral path -> radii : 반지름
        shrink_factor = .8
        zdelta = close_depth * .2 # delta Z = min_depth x 0.2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T -> translation
        # print('tt : ', tt) # [20, 3]
        rads = np.percentile(np.abs(tt), 90, 0) # image 20개 중에서 90%의 크기에 해당하는 translation, 비교적 멀리 떨어져 있는 translation
        # np.percentile(a, q, axis=None) : input array -> np.abs(tt), float -> 90%, axis = 0
        # print('rads : ', rads)
        c2w_path = c2w # recentered pose의 average pose
        N_views = 120 # 새로 만들 view 개수
        N_rots = 2 # rotation 횟수

        if path_zflat: # False
            # zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path -> spiral path(나선형)로 새로운 pose rendering
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)
    # print('5 : ', render_poses.shape) #[120, 3, 5] -> 120개의 새로운 view
    c2w = poses_avg(poses) # view matrix, 3번째 vm = 2번째 vm -> recenter한 pose(camera가 월드 좌표계의 z축을 바라보도록 배치)의 view matrix
    #[[ 1.0000000e+00  0.0000000e+00  0.0000000e+00  1.4901161e-09]
    # [ 0.0000000e+00  1.0000000e+00 -1.8730975e-09 -9.6857544e-09]
    # [-0.0000000e+00  1.8730975e-09  1.0000000e+00  0.0000000e+00]]
    # print('6 : ', type(c2w), c2w.shape)

    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    # print(c2w)
    # print(c2w[:3,3]) # translation
    # print(poses.shape) # [20, 3, 5]
    # print(poses[:,:3,3]) # 20개의 translation
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1) # dists = square(새로운 view matrix의 translation - recentered pose의 translation)
    print('dists : ', dists)
    i_test = np.argmin(dists) # average pose와 가장 가까운 pose에 대해서 test, for Holdout, Validation
    # np.argmin() : 최소값에 해당하는 index
    # print(i_test)
    print('HOLDOUT view is', i_test) # dists가 가장 작은 값의 index를 test로 설정
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32) # poses -> camera to world(여기서 world 좌표계는 average pose로 recenter 됨)

    return images, poses, bds, render_poses, i_test # poses -> recentered poses

images, poses, bds, render_poses, i_test = load_llff_data(basedir, factor=1, recenter=True, bd_factor=.75, spherify=False, path_zflat=False)

last = np.array([0, 0, 0, 1]).reshape(1, 4)
T_Ba = poses[0,:,:][:,:4] 
T_Ba = np.concatenate([T_Ba, last], axis=0)
print(T_Ba)
T_Bb = poses[1,:,:][:,:4]
T_Bb = np.concatenate([T_Bb, last], axis=0)
T_ab = np.linalg.inv(T_Ba) @ T_Bb
print(T_ab)
# [[ 0.9992517  -0.00420171 -0.03845145 -0.14373397]
#  [ 0.00440825  0.99997634  0.00528813  0.0092541 ]
#  [ 0.03842831 -0.00545367  0.99924652 -0.49785128]
#  [ 0.          0.          0.          1.        ]]