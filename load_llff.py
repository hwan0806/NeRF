import numpy as np
import os, imageio

########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

#minify : 압축
#factors -> list, resolutions -> list
#factors = [factor], resolutions = [height, width]
def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False #neetoload default
    for r in factors: #factor의 모든 원소에 대하여,
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir): #imgdir의 경로에 해당 파일이 없다면,
            needtoload = True #needtoload 해야 한다.
    for r in resolutions: #resolution의 모든 원소에 대하여,
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0])) #images_{width}x{height}
        if not os.path.exists(imgdir): #imgdir의 경로에 해당 파일이 없다면,
            needtoload = True #needtoload 해야 한다.
    if not needtoload: #needtoload가 False라면, 이 함수를 벗어나라.
        return
    
    from shutil import copy #shutil : 파일 복사 혹은 폴더 복사
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images') #image의 경로 = basedir + 'images' = './data/nerf_llff_data/fern/images'
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))] #os.listdir() : images에 있는 모든 파일과 리스트를 리턴 -> sorted() : 정렬
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])] #endswith() : 정의된 문자열이 지정된 접미사로 끝나면 True, 그렇지 않으면 False
    imgdir_orig = imgdir
    
    wd = os.getcwd() #현재 작업경로 가져오기

    for r in factors + resolutions: #factor와 resolution의 모든 원소에 대하여,
        if isinstance(r, int): #r이 int라면 True -> 다음 명령문 수행, r이 factor라면
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r) #resize = (100/8)%로
        else: #r이 int가 아니라면 False -> 다음 명령문 수행, r이 [height, width]라면
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name) #가져올 imgdir는 resize한 이미지를 가져온다.
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir) #print '압축 중'
        
        os.makedirs(imgdir) #imgdir 폴더 형성
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

basedir = './data/nerf_llff_data/fern'

#load data
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy')) #[20, 17]
    # print(poses_arr.shape)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) #[3, 5, 20]
    # print(poses.shape)
    bds = poses_arr[:, -2:].transpose([1,0]) #[2, 20]
    # print(bds.shape)
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]


    sh = imageio.imread(img0).shape
    sfx = ''

    #압축 과정
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
    
    #factor = 8

    imgdir = os.path.join(basedir, 'images' + sfx)
    print('imgdir : ', imgdir)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape #[3024, 4032, 3]
    # print(sh)
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1]) #image height, width resize
    poses[2, 4, :] = poses[2, 4, :] * 1./factor #focal length

    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles] #image -> 255로 나눠 normalize
    imgs = np.stack(imgs, -1)
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    
    return poses, bds, imgs

# poses, bds, imgs = _load_data(basedir) #poses -> [3, 5, 20] / bds -> [2, 20] / img -> [3024, 4032, 3, 20]
# print(poses.shape) #[3, 5, 20]
# print(bds.shape) #[2, 20]
# print(imgs.shape) #[3024, 4032, 3, 20]

def normalize(x):
    return x / np.linalg.norm(x)

#view matrix : from world space into view space(camera space)

#카메라를 월드 시스템의 원점으로 변환하고, 카메라가 양의 z-축을 내려다 보도록 회전시켜야 한다.
#1) Eye vector와 Look vector를 통해 z축을 구한다.
#2) z축과 임시 축인 up 벡터의 연산으로 x축을 구한다.
#3) 구한 z, x축의 연산을 통해 y축을 구한다.
#4) 각자 구한 x, y, z를 normalize하면 view matrix의 rotation 부분이 된다.
def viewmatrix(z, up, pos): #z = vec2 [R3, R6, R9], up = [R2, R5, R8], center = [t1, t2, t3]


    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))

    m = np.stack([vec0, vec1, vec2, pos], 1)


    return m #view matrix -> Q. camera 좌표계의 좌표를 World 좌표계의 좌표로 변환하는 과정?

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses): #poses -> [r, u, -t], [20, 3, 5]
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0) #20개 image의 translation에 대한 평균 -> [t1, t2, t3]
    # print('1 : ', center.shape) #[3,]
    vec2 = normalize(poses[:, :3, 2].sum(0)) #20개 image의 [R3, R6, R9]에 대해 normalize
    # print('2 : ', vec2.shape)
    up = poses[:, :3, 1].sum(0) #20개 image의 [R2, R5, R8]에 대해 normalize
    # print('3 : ', up.shape) #[3,]

    vm = viewmatrix(vec2, up, center)
    # print('1 : ', vm)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w #view matrix + hwf

#spiral path로 새로운 pose를 rendering ******************************
#여기서 c2w_path = c2w -> recenter 한 후의 pose, camera의 z축과 world의 z축이 겹치는 경우
# render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    # print('1 : ', rads) #[0.39451256 0.12918354 0.04078128]
    rads = np.array(list(rads) + [1.]) #radian
    # print('1 : ', rads) #[0.39451256 0.12918354 0.04078128 1.]
    hwf = c2w[:,4:5] #recenter한 pose의 view matrix -> c2w
    # print('2 : ', rots, N) # 2, 120 -> 2번 회전, 120개의 새로운 view
    # print('3 : ', np.linspace(0.,2*np.pi*rots,N+1)[:-1].shape) #[120, ]
    #0~4pi까지 N개의 간격을 갖도록 -> 720도의 각도에서 120개의 새로운 각도를 뽑아낸다.
    #이해가 가지 않는다.********************************************************
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        #c2w[:3, :4] -> hwf를 제외한 rotation + translation view matrix
        #np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads
        #np.dot : 각 자리수끼리 모두 더한다. 내적
        print('4 : ', (np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads).shape) #[4,]

        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) #(rotation + translation)([x, y, z] * rads)
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.]))) #z 값에 대해 focal(mean depth)를 빼준다.

        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
        
    return render_poses #새롭게 만들어낸 c2w pose
    
#recenter_poses : 결국 pose를 내가 보고 있는 방향으로 recenter 하는 일
def recenter_poses(poses): #pose -> [r, u, -t], c2w(camera 좌표를 world 좌표로 변환)
    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])

    c2w = poses_avg(poses) #view matrix
    # print('2 : ', c2w.shape) #[3, 5]
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    # print('2 : ', c2w.shape) #[4, 4]
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)
    print('3 : ', poses.shape) #[20, 4, 4]


    poses = np.linalg.inv(c2w) @ poses 
    
    
    #poses = (viewmatrix, c2w)^(-1) x poses(c2w)
    
    
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_

    return poses #view matrix가 반영된 pose -> Q. World to Camera Coordinate? Camera to World Coordinate? 
    #A. 내가 생각하기에는 Camera to World와 View matrix 자체는 다른 개념이다. World to Camera 혹은 Camera to World coordinate은 보는 관점에 따른 것이 아니라 좌표를 옮기는 것의 문제이고, 
    #view matrix는 내가 보고 있는 축을 중심으로(나는 카메라를 보고 있기 때문) 물체의 측면을 바라보는 것을 의미한다.


#####################


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
    
#LLFF dataloader
def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):

    poses, bds, imgs = _load_data(basedir, factor=factor)
    # print(poses.shape) #[3, 5, 20] 
    # print(bds.shape) #[2, 20]
    # print(imgs.shape) #[378, 504, 3, 20]

    #poses_bounds.npy -> [r, -u, t] => [-u, r, -t]

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1) #[-u, r, -t] => [r, u, -t] (World 좌표계와 같은 좌표축 변환)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    # print(poses.shape) #[20, 3, 5]
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    # print(imgs.shape) #[20, 378, 504, 3]
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    # print(bds.shape) #[20, 2]

    #poses, bds refinement
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor) 
    poses[:,:3,3] *= sc 
    bds *= sc 

    #poses -> [r, u, -t]

    if recenter: #recenter = True -> 물체를 내가 보고 있는 방향(카메라가 보고 있는 방향)으로 recenter 하는 일
        poses = recenter_poses(poses)
        print("after recenter")

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else: #spherify = False, pose는 위의 recenter_poses를 거쳐 정렬되었음.

        c2w = poses_avg(poses) #pose에 대해서 view matrix를 구한다. 2번째 vm 
        #[[ 1.0000000e+00  0.0000000e+00  0.0000000e+00  1.4901161e-09]
        # [ 0.0000000e+00  1.0000000e+00 -1.8730975e-09 -9.6857544e-09]
        # [-0.0000000e+00  1.8730975e-09  1.0000000e+00  0.0000000e+00]]
        print("after spherify")

        
        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0)) 

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5. #bds의 최소값 x 0.9, bds의 최대값 x 5
        dt = .75 #가중치
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth)) #mean depth
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        # print('4 : ', poses.shape) #[20, 3, 5]
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T -> translation
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w #recenter한 pose의 view matrix
        N_views = 120 #새로운 view의 수
        N_rots = 2 #rotation 횟수

        if path_zflat: #False
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path -> spiral path(나선형)로 새로운 pose rendering
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)
    # print('5 : ', render_poses.shape) #[120, 3, 5] -> 120개의 새로운 view
    c2w = poses_avg(poses) #view matrix, 3번째 vm -> recenter한 pose(camera가 월드 좌표계의 z축을 바라보도록 배치)의 view matrix
    #[[ 1.0000000e+00  0.0000000e+00  0.0000000e+00  1.4901161e-09]
    # [ 0.0000000e+00  1.0000000e+00 -1.8730975e-09 -9.6857544e-09]
    # [-0.0000000e+00  1.8730975e-09  1.0000000e+00  0.0000000e+00]]
    # print('6 : ', type(c2w), c2w.shape)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test

images, poses, bds, render_poses, i_test = load_llff_data('./data/nerf_llff_data/fern', factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False)
