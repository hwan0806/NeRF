import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

# batchify -> 더 작은 batch size에 적용할 수 있는 fn 설정
# netchunk = 1024*64 = rays/batch * points/ray = 65536 points/batch
def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        # print('inputs : ', inputs.shape) # [65536, 90]
        # print('chunk : ', chunk) # 65536
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
        # range(0, 65536, 65536) -> batch = 1, range(0, 131072, 65536) -> batch = 2
    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    # 하나의 ray에 대한 inputs(x, y, z), viewdirs(X, Y, Z)
    # print('inputs : ', inputs.shape) # [1024, 64, 3] -> 1 batch = 1024 rays, N_c = 64 -> 한 ray에서 sampling한 point 개수, 3 -> [x, y, z], position
    # print('viewdirs : ', viewdirs.shape) # [1024, 3]
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    # print('inputs_flat : ', inputs_flat.shape) # [1024x64, 3], 3 -> [x, y, z]
    embedded = embed_fn(inputs_flat)
    # print('embedded : ', embedded.shape) # [1024x64, 63], 63 = 60 + 3 -> positional encoding

    if viewdirs is not None: # viewdirs = True
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        # print('input_dirs : ', input_dirs.shape) # [1024, 64, 3], 1 batch = 1024 rays, N_c = 64 -> 한 ray에서 sampling한 point 개수, 3 -> [X, Y, Z], direction
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        # print('input_dir_flat : ', input_dirs_flat.shape) # [1024x64, 3], 3 -> [X, Y, Z]
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        # print('embedded_dirs : ', embedded_dirs.shape) # [1024x64, 27], 27 = 24 + 3 -> positional encoding
        embedded = torch.cat([embedded, embedded_dirs], -1)
        # print('concat : ', embedded.shape) # [1024x64, 90] = [65536, 90]
    
    # fn -> NeRF
    outputs_flat = batchify(fn, netchunk)(embedded) # netchunk = 1024*64 -> 1 batch 당 points
    # print('outputs_flat : ', outputs_flat.shape) # [65536, 4] -> 각 point에서의 rgb + density
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    # print('outputs : ', outputs.shape) # [1024, 64, 4]
    return outputs # [batch 당 ray의 수, ray 당 point의 수, rgb + density]

# ray batch화
# all_ret = batchify_rays(rays, chunk, **kwargs)
# rays = [1024, 11]
def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    # print('rays_flat : ', rays_flat.shape) # [1024, 11]
    for i in range(0, rays_flat.shape[0], chunk): # [0, 1024, 1024*32] -> 반복 x
        # print(i) # 0
        ret = render_rays(rays_flat[i:i+chunk], **kwargs) # 
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth. -> 거리가 가까울수록, 해당하는 pixel 값이 커지기 때문
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None: # c2w가 존재한다면,
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs: # use_viewdirs = True
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        # print('viewdirs : ', viewdirs.shape) # [1024, 3]
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()
        # print('viewdirs : ', viewdirs.shape) # [1024, 3], 3 -> [X, Y, Z]

    sh = rays_d.shape # [..., 3]
    if ndc: # ndc = True
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    # print('rays_o : ', rays_o.shape) # [1024, 3]
    # print('rays_d : ', rays_d.shape) # [1024, 3]

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    # print(near.shape) # [1024, 1]
    # print(far.shape) # [1024, 1]
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1) # viewdirs -> normalization of rays_d
        # print('rays : ', rays.shape) # rays = rays_o + rays_d + near + far + viewdirs
        # [1024, 11], 11 = 3 + 3 + 1 + 1 + 3
    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps

# model 설정
def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed) # 10, 0 -> multires = L
    # print('embed_fn, input_ch', embed_fn, input_ch) # input_ch = 63 -> 60(2x10x3) + 3(position)
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs: # use_viewdirs = True
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed) # 4, 0
        # print('embeddirs_fn, input_ch_views', embeddirs_fn, input_ch_views) # input_ch_views = 27 -> 24(2x4x3) + 3(direction)
    output_ch = 5 if args.N_importance > 0 else 4 # Fine Sampling -> output_ch = 5, Coarse Sampling -> output_ch = 4
    skips = [4]
    # Coarse Sampling -> output_ch = 4
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0: # Fine Sampling -> output_ch = 5
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    # inputs(position), viewdirs(direction), network_fn -> 매개변수, run_network -> return
    # network_query_fn -> [1024, 64, 4], 4 -> rgb + density
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train} # render_kwargs_test = render_kwargs_train
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

# 3D rgb + density -> 2D image
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    # lambda 매개변수: 표현식
    # 3D rgb -> 2D rgb로의 weight을 구하는 수식, raw -> 3D color
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1] # 각 sample들 간의 거리
    # print('dists : ', dists.shape) # [1024, 63]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    # print('dists : ', dists) # 마지막 1e10은 무한대를 의미
    # print(dists.shape) # [1024, 64]
    # print('rays_d : ', rays_d.shape) # [1024, 3]
    # print(rays_d[...,None,:].shape) # [1024, 1, 3] -> [X, Y, Z] direction에 대한 normalization
    # print(torch.norm(rays_d[...,None,:], dim=-1).shape) # [1024, 1]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    # print('dists : ', dists.shape) # [1024, 64]
    # print('raw : ', raw.shape) # [1024, 64, 4] -> rgb + density
    rgb = torch.sigmoid(raw[...,:3]) # [N_rays, N_samples, 3] -> density를 제외한 rgb만
    # print('rgb : ', rgb.shape) # [1024, 64, 3]
    noise = 0.
    # print('raw_noise_std : ', raw_noise_std) # 1.0
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std 
        # raw[...,3] -> density
        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples] -> [1024, 64]
    # alpha = 1-exp(-rgb*dists)
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    # torch.cumprod() : 차원 dim에 있는 input 요소의 누적 곱을 반환한다. ex) a = [a, b, c, d] -> torch.cumprod(a, dim=0) = [a, axb, axbxc, axbxcxd]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    # weights의 맨 앞에 붙는 alpha -> T
    # a = torch.cat([torch.ones((alpha.shape[0], 1)), 1-alpha+1e-10], -1) # 마지막 값은 빼줄 것이기 때문에 쓰레기 값을 넣는다.
    # print('a : ', a.shape) # [1024, 65]
    # print('a : ', a)
    # b = torch.cumprod(a, -1) 
    # print('b: ', b) 
    # print('b : ', b.shape) # [1024, 65]
    # c = b[:, :-1] # [1024, 64] -> 마지막 값을 포함하지 않는다.
    # print('c : ', c) 
    # print('c : ', c.shape) # [1024, 64]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    # [1024, 64, 1] * [1024, 64, 3] = [1024, 64, 3] -> [1024, 3]
    # print('rgb_map : ', rgb_map.shape) # [1024, 3]
    # print('rgb_map : ', rgb_map) # 0~1
    depth_map = torch.sum(weights * z_vals, -1) # [1024]
    # print('depth_map : ', depth_map.shape) # [1024]
    # print('depth_map : ', depth_map) # 0~1
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    # depth_map의 inversion
    acc_map = torch.sum(weights, -1) # weight map

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map

# Classic Volume Rendering
# ret = render_rays(rays_flat[i:i+chunk], **kwargs) -> rays_flat = [1024, 11] -> rays_o + rays_d + near + far + viewdirs(the normalization of rays_d)
def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    # print('ray_batch : ', ray_batch.shape) # [1024, 11], 11 = rays_o 3 + rays_d 3 + near 1 + far 1 + viewdirs 3
    N_rays = ray_batch.shape[0] # 1 batch = 1024 rays
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None # viewdirs 이용 = True -> 11 / viewdirs 이용 = False -> 8
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2]) # near, far
    near, far = bounds[...,0], bounds[...,1]

    # print('near : ', near) # [1024, 1]
    # print('far : ', far) # 1
    # print('N_samples : ', N_samples) # 64
    t_vals = torch.linspace(0., 1., steps=N_samples) # min_depth와 max_depth 사이를 64개의 point로 sampling 한다.
    # print('t_vals : ', t_vals.shape) # 64

    if not lindisp: # lindisp -> depth의 inverse에 비례하여 sampling 한다. 가까울 수록, 더 많이 sampling 한다.
        z_vals = near * (1.-t_vals) + far * (t_vals) # z_vals는 0과 1 사이에 있게 한다.
        # print(z_vals) # [0, 1]
        # print(z_vals.shape) # [1024, 64]
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])
    # print('z_vals : ', z_vals.shape) # [1024, 64]

    # Stratified sampling
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1]) # 중간 값
        # print('mids : ', mids) # 0.5 * (두 번째 값 + 세 번째 값), 0.5 * (세 번째 값 + 네 번째 값), ...
        # print(mids.shape) # [1024, 64]
        upper = torch.cat([mids, z_vals[...,-1:]], -1) # mids + 마지막 값
        # print('upper : ', upper.shape) # [1024, 64]
        lower = torch.cat([z_vals[...,:1], mids], -1) # 처음 값 + mids
        # print('lower : ', lower.shape) # [1024, 64]
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape) # 0과 1 사이의 size [1024, 64]의 random한 행렬
        # print('t_rand : ', t_rand)
        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand # random하게 뽑는다.

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    # r(t) = o + td, t = z_vals
    # pts = [1024, 64, 3]
#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    # print('raw : ', raw.shape) # [1024, 64, 4]
    # network_query_fn -> [1024, 64, 4], 4 -> 3D rgb + 3D density
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0: # Fine sampling = True
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

# Hyper parameter
def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, # 1 batch = 4096 rays
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray') # Coarse Sampling
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray') # Fine Sampling
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path') # Test
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path') # Test
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None # K 초기화

    # LLFF Dataloader
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify) # Dataloader
        # print('images : ', images.shape) # [20, 378, 504, 3], 3 -> rgb
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4] # poses -> [3, 4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        # images -> [20, 378, 504, 3] = [image 개수, height, width, rgb], render_poses -> [120, 3, 5], hwf -> [378, 504, 407.5658]
        if not isinstance(i_test, list): # i_test가 list 형식이 아니라면, 
            i_test = [i_test] # i_test를 list로 만들어라.

        # for validation
        if args.llffhold > 0: # llffhold = 8
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold] # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            # np.arange(시작점(생략 시 0), 끝점(미포함), step size(생략 시 1))
            # [::args.llffhold] -> 0 부터 끝까지 args.llffhold 간격으로, i_test = [0, 8, 16]
        i_val = i_test # validation 수행
        # print(i_test)
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)]) # validation에 이용되는 이미지를 제외한 나머지 이미지는 모두 train image로 이용된다.
        # print(i_train) # [1 2 3 4 5 6 7 9 10 11 12 13 14 15 17 18 19]
        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else: # NDC
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
    
    # blender data 형식의 data loader
    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W], # 0.5*W -> c_x 주점
            [0, focal, 0.5*H], # 0.5*H -> c_y 주점
            [0, 0, 1]
        ])

    if args.render_test: # False
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    # near = 0, far = 1
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    
    # boundary depth에 대한 속성 update
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only: # Test
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test] # 12
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand # 1024
    use_batching = not args.no_batching
    # For Batch 학습
    if use_batching: # use_batching = True
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        # print('rays : ', rays.shape) # [20, 2, 378, 504, 3] -> rays_o[378, 504, 3] + rays_d[378, 504, 3], 3 -> [x, y, z] position + [X, Y, Z] direction
        # print('done, concats')
        # print('images : ', images.shape) # [20, 378, 504, 3]
        # print('images_rgb : ', images[:,None].shape) # [20, 1, 378, 504, 3]
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3] -> [20, 3, 378, 504, 3], 3 -> [x, y, z] position + [X, Y, Z] direction + [r, g, b] rgb
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3] -> [20, 378, 504, 3, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        # print(rays_rgb.shape) # [17, 378, 504, 3, 3] -> validation image [0, 8, 16]을 제외한 train image
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3] = [3238704, 3, 3]
        # print('rays_rgb : ', rays_rgb.shape)
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0 # i_batch 초기화

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    # train
    start = start + 1
    for i in trange(start, N_iters): # tqdm(range()) -> process bar
        time0 = time.time()

        # Sample random ray batch
        if use_batching: # use_batching = True
            # Random over all images
            # print('rays_rgb : ', rays_rgb.shape) # [3238704, 3, 3] -> rays_o(x, y, z) + rays_d(X, Y, Z) + rgb
            # 하나의 image의 모든 ray, point에 대해서 batch화 수행
            # i_batch = 0, i_batch+N_rand = 0+1024
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            # print('batch : ', batch.shape) # [1024, 3, 3] -> rays_o(x, y, z) + rays_d(X, Y, Z) + rgb
            batch = torch.transpose(batch, 0, 1)
            # print('batch2 : ', batch.shape) # [3, 1024, 3]
            batch_rays, target_s = batch[:2], batch[2] # [3, 1024] -> 1024 = dimension
            # batch_rays -> rays_o(x, y, z) + rays_d(X, Y, Z)
            # target_s -> rgb
            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]
        
            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        # Classic Volume Rendering : 3D points -> 2D images
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s) # rgb -> classic volume rendering의 결과, target_s -> train image의 실제 rgb
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        # video 생성 -> +Test mode
        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        """
            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)


            if i%args.i_img==0:

                # Log a rendered validation view to Tensorboard
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                    tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
        """

        global_step += 1


if __name__=='__main__': #run_nerf.py가 실행되었을 때만 실행되도록,
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train() #main문을 돌리면 train()
