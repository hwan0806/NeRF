import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w 
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w 
    return c2w #c2w에는 radius

def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test'] #data -> train, val, test
    metas = {} #meta라는 directory
    for s in splits: #train, val, test에 대해서
        #basedir의 transforms_{}.json을 읽기 모드로 열고, fp로 적는다.
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp) #meta dictionary에 {['train', 'transforms_train.json']}, basedir = './logs/'

    all_imgs = [] #모든 이미지
    all_poses = [] #해당 이미지에 대한 pose
    counts = [0]

    for s in splits: #split의 모든 요소에 대해 반복
        meta = metas[s] #train, val, test 중 하나
        imgs = []
        poses = []

        if s=='train' or testskip==0:
            skip = 1 #train이거나 input으로 들어온 testskip이 0이라면, skip = 1
        else:
            skip = testskip #그게 아니라면, 그대로 skip = testskip
            
        for frame in meta['frames'][::skip]: #.json 파일의 'frames'에 대하여
            fname = os.path.join(basedir, frame['file_path'] + '.png') #각각의 이미지에 대한 경로
            imgs.append(imageio.imread(fname)) #images -> imgs list에 추가
            poses.append(np.array(frame['transform_matrix'])) #.json 파일의 transform matrix를 array로 변환하여 poses list에 추가
        
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        #이 때의 image를 처리하기 위해 255로 나눠준다.

        poses = np.array(poses).astype(np.float32) #pose도 imgs와 같은 float32 데이터 타입으로 변환한다. 
        #float32 -> 1비트 부호, 8비트 정수, 23비트 소수
        counts.append(counts[-1] + imgs.shape[0]) #이 부분은 잘 모르겠음.
        all_imgs.append(imgs) #각각의 image를 전처리하여 all_imgs list에 넣는다.
        all_poses.append(poses) #각각의 pose를 전처리하여 all_poses list에 넣는다.
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0) #array를 dim = 0에 대해서 더한다.
    poses = np.concatenate(all_poses, 0) #array를 dim = 0에 대해서 더한다.
    
    H, W = imgs[0].shape[:2] #800, 800
    camera_angle_x = float(meta['camera_angle_x']) #camera_angle의 list를 float형으로 가져온다.
    focal = .5 * W / np.tan(.5 * camera_angle_x) #tan 함수, 가져온 camera_angle_x를 이용하여 focal length를 구한다.
    
    #Q. 이 부분 잘 이해가 안된다.
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0) #이 때, 마지막 것을 뺀다.
    #np.linspace(구간 시작점, 구간 끝점, 구간 내 숫자 개수) : -180에서 180까지 


    #만약, half_size로 image를 resize한다면, 
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2. #focal은 float여야 하기 때문,

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4)) #[400(image 개수), 400, 400, 4]
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) #interpolation을 통한 image의 resize
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split #image와 pose, render pose, camera intrinsic parameter 등을 출력
   