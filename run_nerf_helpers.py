import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims'] # 3 -> [x, y, z]
        out_dim = 0 # 초기화
        if self.kwargs['include_input']: # True
            embed_fns.append(lambda x : x) # lambda 매개변수 : 표현식
            out_dim += d # out_dim = out_dim + input_dims -> 3

        max_freq = self.kwargs['max_freq_log2'] # 10 - 1 = 9
        N_freqs = self.kwargs['num_freqs'] # 10
        
        if self.kwargs['log_sampling']: # True
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs) # 2^(L-1)
            # torch.linspace(start, end, steps) = torch.linspace(0, 9, 10) -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            # print(freq_bands) # [1., 2., 4., 8., 16., 32., 64., 128., 256., 512.]
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands: # [1., 2., 4., 8., 16., 32., 64., 128., 256., 512.]
            for p_fn in self.kwargs['periodic_fns']: # [torch.sin, torch.cos]
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d # out_dim = out_dim + input_dims -> 3 + 3 * 10 * 2 = 63
                # out_dim = input_dim + 2 * L = 60 + 3
        # print(embed_fns)
        # print(out_dim) # 63
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

# multires = 10 -> L = 10
def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3 # positional embedding x

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1, # max_freq_log2 = 9
                'num_freqs' : multires, # num_freqs = 10
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos], # torch.sin(rad), torch.cos(rad)
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    # def 함수이름(매개변수): return 결과 -> lambda 매개변수: 결과
    embed = lambda x, eo=embedder_obj : eo.embed(x) # eo.embed(x) = Embedder.embed(x)
    return embed, embedder_obj.out_dim # embed -> embedding function, embedder_obj.out_dim -> output dimension

# a, b = get_embedder(10)
# print(a)
# print(b) # 63

# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False): # use_viewdirs = True
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D # 8 -> density를 뽑기 전 block의 개수
        self.W = W # 256 -> node의 개수
        self.input_ch = input_ch # 3 -> position [x, y, z]
        self.input_ch_views = input_ch_views # 3 -> viewing direction [X, Y, Z] Cartesian unit vector
        self.skips = skips # [4] for residual term
        self.use_viewdirs = use_viewdirs # 실제 train -> True
        
        # nn.ModuleList : nn.Sequential과 비슷하게, nn.Module의 list를 input으로 받는다. 하지만 nn.Sequential과 다르게 forward() method가 없다.

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)]) # 7
            # i = 0, 1, 2, ..., 6 만일 i = 4 -> nn.Linear(256 + 3, 256)
            # [3, 256] + [256, 256] + [256, 256] + [256, 256] + [256, 256] + [256 + 3, 256](skip connection) + [256, 256] + [256, 256] 

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)]) # [256 + 3, 128]
        
        # [3 + 256, 128] -> direction + feature space

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs: # viewing direction을 중간에 투입
            self.feature_linear = nn.Linear(W, W) # [256, 256]
            self.alpha_linear = nn.Linear(W, 1) # [256, 1] -> density 출력
            self.rgb_linear = nn.Linear(W//2, 3) # [128, 3] -> RGB 출력
        else:
            self.output_linear = nn.Linear(W, output_ch) # viewing direction을 쓰지 않는다면, [256, 1+3] 한 번에 density와 RGB 출력

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1) # tensor를 [self.input_ch, input_ch_views] size로 나누기
        # input을 position과 viewing direction으로 나누기
        h = input_pts # position
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h) # [256, 256]
            h = F.relu(h)
            if i in self.skips: # i = 4
                h = torch.cat([input_pts, h], -1) # skip connection -> dimension 합침

        if self.use_viewdirs: # viewing direction = True
            alpha = self.alpha_linear(h) # density 추출
            feature = self.feature_linear(h) # feature vector 추출
            h = torch.cat([feature, input_views], -1) # feature + viewing direction
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1) # outputs = rgb color + density
        else:
            outputs = self.output_linear(h)

        return outputs

    # self.use_viewdirs = True
    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        # assert [조건], [오류메시지] -> 조건 = True : 그대로 코드 진행, 조건 = False : 오류메시지 발생
        # Load pts_linears
        for i in range(self.D): # 8
            idx_pts_linears = 2 * i # 0, 2, 4, 6, 8
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D # 2 * 8
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))

# nerf = NeRF(D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=True)

# c2w -> Tensor
# Ray helpers -> Ray Dataloader
def get_rays(H, W, K, c2w): # poses [3, 4] -> recentered poses
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    # i, j -> image 픽셀 상의 좌표
    # torch.linspace(start, end, steps)
    # torch.linspace(0, W-1, W) : 0~W-1 범위 안을 W개 만큼 나눈다. 0, 1, 2, 3, ..., W-1
    # torch.linspace(0, H-1, H) : 0~H-1 범위 안을 H개 만큼 나눈다. 0, 1, 2, 3, ..., H-1
    i = i.t() # tensor화
    j = j.t()
    # print(i.shape) # [3024, 4032] -> Height, Width
    # print(j.shape) # [3024, 4032] -> Height, Width
    # i - c_x / f_x, -(j - c_y / f_y) -> pixel 단위계에서 metric 단위계로 변환
    # print(torch.ones_like(i)) # [3024, 4032]
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # print('dirs : ', dirs)
    # print(dirs.shape) # [3024, 4032, 3] -> ([x, -y, -1] x Width) x Height

    # Rotate ray directions from camera frame to the world frame
    # print(dirs[..., np.newaxis, :].shape) # [3024, 4032, 1, 3]
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1) # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # [3024, 4032, 1, 3] * [3, 3] -> c2w * [x, y, 1] -> 직선의 방정식, 방향만 알면 되기 때문에 z 값이 없어도 된다.
    # print((dirs[..., np.newaxis, :]*c2w[:3, :3]).shape) # [3024, 4032, 3, 3]
    # pixel 단위계에서 metric 단위계로 변환한 camera 좌표계 상의 point들을 camera to world를 통해 world 좌표계 상의 좌표로 변환한다.
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape) # rays_o -> translation, 각 image 별로 같은 translation 적용
    # print(c2w.shape) # [3, 4] -> [R|t]
    # print(c2w[:3, -1]) # translation
    # print(rays_o)
    return rays_o, rays_d # translation, rotation -> world 좌표계 상으로 변환된 픽셀 좌표
    # r(t) = o + td -> t : scale, o : origin, d : direction
# H = 3024
# W = 4032
# focal = 3260.526333
# K = [3, 3] -> K[0][2] = 0.5*W, K[0][0] = focal, K[1][2] = 0.5*H, K[1][1] = focal
# K = np.array([
#             [focal, 0, 0.5*W], # 0.5*W = x축 방향의 주점
#             [0, focal, 0.5*H], # 0.5*H = y축 방향의 주점
#             [0, 0, 1]])
# c2w = torch.FloatTensor([[1, 1, 1, 1],
#                 [1, 1, 1, 1],
#                 [1, 1, 1, 1]]) #[3, 4]

# rays_o, rays_d = get_rays(H, W, K, c2w)
# print(rays_o.shape, rays_d.shape) # [3024, 4032, 3], [3024, 4032, 3] -> 각 픽셀에 대한 모든 ray를 계산한다.

# c2w = np.array([[1, 1, 1, 1],
#                 [1, 1, 1, 1],
#                 [1, 1, 1, 1]])
# c2w -> numpy
def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    # np.arange(시작점(생략 시 0), 끝점(미포함), step size(생략 시 1))
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    # print(rays_d.shape, rays_o.shape) # [3024, 4032, 3], [3024, 4032, 3]
    return rays_o, rays_d # rays_o : ray의 origin
# rays_o, rays_d = get_rays_np(H, W, K, c2w)

# near = 1.
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # rays_o -> [3024, 4032, 3]
    # rays_d -> [3024, 4032, 3]

    # Shift ray origins to near plane
    # print(rays_o[...,2]) # Translation의 Z축 관련
    # print(rays_d[...,2]) # Rotation의 Z축 관련
    t = -(near + rays_o[...,2]) / rays_d[...,2] # t = -(n+o)/d
    rays_o = rays_o + t[...,None] * rays_d # o = o + td
    
    # Projection
    # rays_o[...,0] = o_x, rays_o[...,1] = o_y, rays_o[...,2] = o_z
    # rays_d[...,0] = d_x, rays_d[...,1] = d_y, rays_d[...,2] = d_z
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2] # -f/(W/2)*o_x/o_z
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2] # -f/(H/2)*o_y/o_z
    o2 = 1. + 2. * near / rays_o[...,2] # 1+2n/o_z

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2]) # -f/(W/2)*(d_x/d_z-o_x/o_z)
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2]) # -f/(H/2)*(d_y/d_z-o_y/o_z)
    d2 = -2. * near / rays_o[...,2] # -2n/o_z
    
    rays_o = torch.stack([o0,o1,o2], -1) 
    rays_d = torch.stack([d0,d1,d2], -1)
    # print(rays_o.shape, rays_d.shape) # [3024, 4032, 3], [3024, 4032, 3]

    return rays_o, rays_d

# rays_o, rays_d = ndc_rays(H, W, focal, 1, rays_o, rays_d)

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
