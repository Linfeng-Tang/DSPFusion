import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import os
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from DSPF.utils.base_model import BaseModel
from torch.nn import functional as F
from functools import partial
import numpy as np
from DSPF.utils.beta_schedule import make_beta_schedule, default
from ldm.cddpm import DDPM
from time import time
from thop import profile, clever_format
import statistics


@MODEL_REGISTRY.register()
class DSPF_S2(BaseModel):
    """HI-Diff model for test."""

    def __init__(self, opt):
        super(DSPF_S2, self).__init__(opt)

        # define network
        self.net_dp = build_network(opt['network_dp'])
        self.net_dp = self.model_to_device(self.net_dp, find_unused_parameters=False)
        
        self.net_sp = build_network(opt['network_sp'])
        self.net_sp = self.model_to_device(self.net_sp, find_unused_parameters=False)
        
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g, find_unused_parameters=False)

        self.net_dm = build_network(opt['network_dm'])
        self.net_dm = self.model_to_device(self.net_dm)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_sp', None)
        if load_path is not None:
            load_path = os.path.abspath(load_path.replace('\\', '/'))
            param_key = self.opt['path'].get('param_key_sp', 'params')
            self.load_network(self.net_sp, load_path, self.opt['path'].get('strict_load_sp', True), param_key)
            
        load_path = self.opt['path'].get('pretrain_network_dp', None)
        if load_path is not None:
            load_path = os.path.abspath(load_path.replace('\\', '/'))
            param_key = self.opt['path'].get('param_key_dp', 'params')
            self.load_network(self.net_dp, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            load_path = os.path.abspath(load_path.replace('\\', '/'))
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        load_path = self.opt['path'].get('pretrain_network_dm', None)
        if load_path is not None:
            load_path = os.path.abspath(load_path.replace('\\', '/'))
            param_key = self.opt['path'].get('param_key_dm', 'params')
            self.load_network(self.net_dm, load_path, self.opt['path'].get('strict_load_le_dm', True), param_key)

        
        # diffusion
        self.apply_ldm = self.opt['diffusion_schedule'].get('apply_ldm', None)
        if self.apply_ldm:
            # apply LDM implementation
            self.diffusion = DDPM(denoise=self.net_dm, 
                                  n_feats=opt['network_g']['embed_dim'], 
                                  group=opt['network_g']['group'],
                                  linear_start= self.opt['diffusion_schedule']['linear_start'],
                                  linear_end= self.opt['diffusion_schedule']['linear_end'], 
                                  timesteps = self.opt['diffusion_schedule']['timesteps'])
            self.diffusion = self.model_to_device(self.diffusion)
        else:
            # implemented locally
            self.set_new_noise_schedule(self.opt['diffusion_schedule'], self.device)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.eval()
        self.net_dp.eval()
        self.net_sp.eval()
        self.net_dm.train()
        # if self.apply_ldm:
        #     self.diffusion.train()
        
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            print("TODO")

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            self.cri_pix_diff = build_loss(train_opt['pixel_diff_opt']).to(self.device)
        else:
            self.cri_pix = None
            self.cri_pix_diff = None
            
        if train_opt.get('fidelity_opt'):
            self.cri_fid = build_loss(train_opt['fidelity_opt']).to(self.device)
            ## 能否使用self.mode_to_device()函数将计算指标的类函数加载到GPU上去？
            self.cri_fid = self.cri_fid.to(self.device)
            self.weight_fid = train_opt['fidelity_opt'].get('weight', 1.0)
        else:
            self.cri_fid = None
            
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        if self.apply_ldm:
            for k, v in self.net_dm.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Network Diffusion: Params {k} will not be optimized.')
        else:
            for k, v in self.net_dm.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger = get_root_logger()
                    logger.warning(f'Network LE-DM: Params {k} will not be optimized.')

        optim_type = train_opt['optim_total'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_total = torch.optim.Adam(optim_params, **train_opt['optim_total'])
        elif optim_type == 'AdamW':
            self.optimizer_total = torch.optim.AdamW(optim_params, **train_opt['optim_total'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_total)

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        # β1, β2, ..., βΤ (T)
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['timesteps'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        # α1, α2, ..., αΤ (T)
        alphas = 1. - betas
        # α1, α1α2, ..., α1α2...αΤ (T)
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # 1, α1, α1α2, ...., α1α2...αΤ-1 (T)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        # 1, √α1, √α1α2, ...., √α1α2...αΤ (T+1)
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised=True, condition_x=None, ema_model=False):
        if condition_x is None:
            raise RuntimeError('Must have LQ/LR condition')

        if ema_model:
            print("TODO")
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.net_d(x, condition_x, torch.full(x.shape, t+1, device=self.betas.device, dtype=torch.long)))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance
    
    def p_sample_wo_variance(self, x, t, clip_denoised=True, condition_x=None, ema_model=False):
        model_mean, _ = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, ema_model=ema_model)
        return model_mean
    
    def p_sample_loop_wo_variance(self, x_in, x_noisy, ema_model=False):
        img = x_noisy
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample_wo_variance(img, i, condition_x=x_in, ema_model=ema_model)
        return img

    def p_sample(self, x, t, clip_denoised=True, condition_x=None, ema_model=False):
        model_mean, _ = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, ema_model=ema_model)
        return model_mean

    def p_sample_loop(self, x_in, x_noisy, ema_model=False):
        img = x_noisy
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, i, condition_x=x_in, ema_model=ema_model)
        return img

    def q_sample(self, x_start, sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            sqrt_alpha_cumprod * x_start +
            (1 - sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def feed_data(self, data):        
        self.lq_ir = data['lq_ir'].to(self.device)
        self.lq_vi = data['lq_vi'].to(self.device)
        self.gt_ir = data['gt_ir'].to(self.device)
        self.gt_vi = data['gt_vi'].to(self.device)

    def optimize_parameters(self, current_iter, noise=None):
        # freeze c1 (cpen_s1)
        for p in self.net_g.parameters():
            p.requires_grad = False
        for p in self.net_sp.parameters():
            p.requires_grad = False
        for p in self.net_dp.parameters():
            p.requires_grad = False
        
        self.optimizer_total.zero_grad()
        semantic_prior_lq = self.net_sp(torch.cat([self.lq_ir, self.lq_vi], 1))
        semantic_prior_gt = self.net_sp(torch.cat([self.gt_ir, self.gt_vi], 1))
        deg_prior_ir, _ = self.net_dp(self.lq_ir) ## 提取红外图像的退化先验
        deg_prior_vi, _ = self.net_dp(self.lq_vi) ## 提取红外图像的退化先验
        prior_condition = torch.cat([semantic_prior_lq, deg_prior_ir, deg_prior_vi], dim=-1)

        if self.apply_ldm:
            prior, _ = self.diffusion(self.lq_ir, self.lq_vi, prior_condition, semantic_prior_gt)
        else:
            prior_d = semantic_prior_lq
            # diffusion-forward
            t = self.opt['diffusion_schedule']['timesteps']
            # [b, 4c']
            noise = default(noise, lambda: torch.randn_like(semantic_prior_gt))
            # sample xt/x_noisy (from x0/x_start)
            prior_noisy = self.q_sample(
                x_start=semantic_prior_gt, sqrt_alpha_cumprod=self.alphas_cumprod[t-1],
                noise=noise)
            # diffusion-reverse
            prior = self.p_sample_loop_wo_variance(prior_d, prior_noisy)

        # ir
        self.results = self.net_g(self.lq_ir, self.lq_vi, deg_prior_ir, deg_prior_vi, prior)
        self.output = self.results['fusion']
        self.out_ir = self.results['ir']
        self.out_vi = self.results['vi']
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_fusion, _ = self.cri_pix(self.output, self.gt_ir, self.gt_vi)
            l_fusion_total = l_fusion['loss_fusion']
            l_total += l_fusion_total
            for key, value  in l_fusion.items():
                loss_dict[key] = value
                
        if self.cri_fid:
            l_fid_ir = self.cri_fid(self.out_ir, self.gt_ir, type='ir')
            l_fid_vi = self.cri_fid(self.out_vi, self.gt_vi, type='vi')
            l_fid_total = l_fid_ir['ir_loss_fid'] + l_fid_vi['vi_loss_fid']
            l_total += l_fid_total
            for key, value  in l_fid_ir.items():
                loss_dict[key] = value
            for key, value  in l_fid_vi.items():
                loss_dict[key] = value
                
        if self.cri_pix_diff:
            l_pix_diff = 10 * self.cri_pix_diff(semantic_prior_gt, prior)
            l_total += 1 * l_pix_diff
            loss_dict['l_pix_diff'] = l_pix_diff

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        if self.opt['train']['use_grad_clip']:
            if self.apply_ldm:
                torch.nn.utils.clip_grad_norm_(list(self.diffusion.parameters()), 0.01)
            else:
                torch.nn.utils.clip_grad_norm_(list(self.net_le_dm.parameters()), 0.01)
        self.optimizer_total.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self, idx=None):
        scale = self.opt.get('scale', 1)
        window_size = 8
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq_ir.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img_ir = F.pad(self.lq_ir, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        img_vi = F.pad(self.lq_vi, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        gt_vi = F.pad(self.gt_vi, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        gt_ir = F.pad(self.gt_ir, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        lq = torch.cat([img_ir, img_vi], 1)

        if hasattr(self, 'net_g_ema'):
            print("TODO: wrong")
        else:
            self.net_dp.eval()
            self.net_sp.eval()
            self.net_g.eval()
            self.net_dm.eval()
            self.diffusion.eval()
            if self.apply_ldm:
                self.diffusion.eval()
                with torch.no_grad():                    
                    torch.cuda.synchronize()
                    start_time = time()
                    semantic_prior_lq = self.net_sp(lq) ## 提取高质量的语义先验
                    deg_prior_ir, _ = self.net_dp(img_ir) ## 提取红外图像的退化先验
                    deg_prior_vi, _ = self.net_dp(img_vi) ## 提取红外图像的退化先验
                    prior_condition = torch.cat([semantic_prior_lq, deg_prior_ir, deg_prior_vi], dim=-1)
                    semantic_prior = self.diffusion(self.lq_ir, self.lq_vi, prior_condition)
                    self.results = self.net_g(img_ir, img_vi, deg_prior_ir, deg_prior_vi, semantic_prior)
                    torch.cuda.synchronize()
                    end_time = time()
                    if hasattr(self, 'time_list'):
                        self.time_list.append(end_time - start_time)
                    # if idx == 0:
                        # flops_dp, params_dp = profile(self.net_dp, inputs=(img_ir, ))
                        # flops_sp, params_sp = profile(self.net_sp, inputs=(lq, ))
                        # flops_dm, params_dm = profile(self.diffusion, inputs=(self.lq_ir, self.lq_vi, prior_condition, ))
                        # flops_g, params_g = profile(self.net_g, inputs=(img_ir, img_vi, deg_prior_ir, deg_prior_vi, semantic_prior, ))
                        # flops = flops_dp + flops_sp + flops_dm + flops_g
                        # params = params_dp + params_sp + params_dm + params_g                        
                        # print("Params: {:.3f} M | Flops : {:.3f} G| times : {:.3f}s ".format(params / 1e6, flops / 1e9, end_time - start_time))
                    self.output = self.results['fusion']
                    self.out_ir = self.results['ir']
                    self.out_vi = self.results['vi']
                self.net_dm.train()
                self.diffusion.train()
            else:
                self.net_sp.eval()
                self.net_dp.eval()
                self.net_dm.eval()
                self.net_g.eval()

                with torch.no_grad():
                    semantic_prior_lq = self.net_sp(lq) ## 提取高质量的语义先验
                    # deg_prior_ir = self.net_dp(img_ir) ## 提取红外图像的退化先验
                    # deg_prior_vi = self.net_dp(img_vi) ## 提取红外图像的退化先验
                    ## 不进行增强 
                    deg_prior_ir = self.net_dp(gt_ir) ## 提取红外图像的退化先验
                    deg_prior_vi = self.net_dp(gt_vi) ## 提取红外图像的退化先验
                    
                    prior_condition = torch.cat([semantic_prior_lq, deg_prior_ir, deg_prior_vi], 1)
                    semantic_prior = self.net_dm(self.lq_ir, self.lq_vi, prior_condition)
                    self.results = self.net_g(img_ir, img_vi, deg_prior_ir, deg_prior_vi, semantic_prior)
                    self.output = self.results['fusion']
                    self.out_ir = self.results['ir']
                    self.out_vi = self.results['vi']
                self.net_dm.train()
        _, _, h, w = self.output.size()
        self.output = (self.output - torch.min(self.output)) / (torch.max(self.output) - torch.min(self.output))
        # print(torch.max(self.output), torch.min(self.output))
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        with_rec = self.opt['val'].get('rec_flag', False)
        self.time_list = []
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data1 = dict()
        metric_data2 = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
        if self.opt['is_train']:
            if with_rec:
                save_folder = osp.join(self.opt['path']['visualization'], 'Fusion', '{0}'.format(current_iter))
                ir_save_folder = osp.join(self.opt['path']['visualization'], 'IR_Res', '{0}'.format(current_iter))
                vi_save_folder = osp.join(self.opt['path']['visualization'], 'VI_Res', '{0}'.format(current_iter))                
            else:
                save_folder = osp.join(self.opt['path']['visualization'], '{0}'.format(current_iter))
        else:
            if  dataloader.dataset.opt.get('save_folder') is not None:
                if with_rec:
                    ir_save_folder = osp.join(dataloader.dataset.opt['save_folder'], dataset_name, 'IR_Res')
                    vi_save_folder = osp.join(dataloader.dataset.opt['save_folder'], dataset_name, 'VI_Res')
                save_folder = osp.join(dataloader.dataset.opt['save_folder'], dataset_name)
                # save_folder = osp.join(dataloader.dataset.opt['save_folder'])
            else:
                if with_rec:
                    ir_save_folder = osp.join('./Results', '{}'.format(current_iter), dataset_name, 'IR_Res')
                    vi_save_folder = osp.join('./Results', '{}'.format(current_iter), dataset_name, 'VI_Res')
                    save_folder = osp.join('./Results', '{}'.format(current_iter), dataset_name, 'Fusion')
                else:
                    save_folder = osp.join('./Results', '{}'.format(current_iter), dataset_name)
        os.makedirs(save_folder, exist_ok=True)
        if with_rec:
            os.makedirs(ir_save_folder, exist_ok=True)
            os.makedirs(vi_save_folder, exist_ok=True)
        print("Results will be saved to {}".format(save_folder))
        for idx, val_data in enumerate(dataloader):
            if osp.basename(val_data['lq_path_vi'][0]) != osp.basename(val_data['lq_path_ir'][0]):
                img_name = osp.basename(val_data['lq_path_ir'][0]).split('.')[0] + '_' + osp.basename(val_data['lq_path_vi'][0]).split('_')[-1]
            else:
                img_name = osp.basename(val_data['lq_path_vi'][0])
            self.feed_data(val_data)
            self.test(idx)

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], min_max=(torch.min(visuals['result']), torch.max(visuals['result'])))
            if with_rec:
                rec_ir = tensor2img([visuals['out_ir']], min_max=(torch.min(visuals['out_ir']), torch.max(visuals['out_ir'])))
                rec_vi = tensor2img([visuals['out_vi']], min_max=(torch.min(visuals['out_vi']), torch.max(visuals['out_vi'])))
                
            metric_data1['img'] = sr_img
            metric_data2['img'] = sr_img
            if 'gt_ir' in visuals:
                gt_img_ir = tensor2img([visuals['gt_ir']])
                metric_data1['img2'] = gt_img_ir
                del self.gt_ir
            if 'gt_vi' in visuals:
                gt_img_vi = tensor2img([visuals['gt_vi']])
                metric_data2['img2'] = gt_img_vi
                del self.gt_vi

            # tentative for out of GPU memory
            del self.lq_ir
            del self.lq_vi
            del self.output

            if save_img:
                if self.opt['is_train']:
                    
                    save_img_path = osp.join(save_folder, img_name)
                    if with_rec:
                        save_img_path_ir = osp.join(ir_save_folder, img_name)
                        save_img_path_vi = osp.join(vi_save_folder, img_name)
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(save_folder, img_name)
                        if with_rec:
                            save_img_path_ir = osp.join(ir_save_folder, img_name)
                            save_img_path_vi = osp.join(vi_save_folder, img_name)

                
                imwrite(sr_img, save_img_path)
                if with_rec:
                    imwrite(rec_ir, save_img_path_ir)
                    imwrite(rec_vi, save_img_path_vi)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += 0.5 * calculate_metric(metric_data1, opt_) + 0.5 * calculate_metric(metric_data2, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        # print('The average running time for testing {} images in {} is {:.4f}s'.format(idx + 1, dataset_name, statistics.mean(self.time_list[2:])))
        print("Fusion results are saved to {}".format(save_folder))



    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq_ir'] = self.lq_ir.detach().cpu()
        out_dict['lq_vi'] = self.lq_vi.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt_ir'):
            out_dict['gt_ir'] = self.gt_ir.detach().cpu()
        if hasattr(self, 'gt_vi'):
            out_dict['gt_vi'] = self.gt_vi.detach().cpu()
        if hasattr(self, 'out_ir'):
            out_dict['out_ir'] = self.out_ir.detach().cpu()
        if hasattr(self, 'out_vi'):
            out_dict['out_vi'] = self.out_vi.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            print("TODO")
        else:
            if self.apply_ldm:
                if self.opt['dist']:
                    self.net_dm = self.diffusion.module.model
                else:
                    self.net_dm = self.diffusion.model
            self.save_network(self.net_dm, 'net_dm', current_iter)
        self.save_training_state(epoch, current_iter)
