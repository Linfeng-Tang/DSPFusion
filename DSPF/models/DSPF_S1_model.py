import torch
from collections import OrderedDict
import os
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from DSPF.utils.base_model import BaseModel
from torch.nn import functional as F
from functools import partial
from DSPF.utils.beta_schedule import make_beta_schedule, default
from ldm.ddpm import DDPM
from scipy.io import savemat
from einops import rearrange
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

@MODEL_REGISTRY.register()
class DSPF_S1(BaseModel):
    """HI-Diff model for test."""

    def __init__(self, opt):
        super(DSPF_S1, self).__init__(opt)

        # define network
        '''
        构建两个退化先验编码器 分别用于处理红外和可见光图像
        一个语义先验编码器用于从级联的红外和可见光图像中提取有助于图像恢复的高级语义先验
        一个图像增强&融合网络 包含两路编码器分别从红外和可见光图像中提取特征 编码器包括退化先验调制模块和语义先验调制模块
        '''
        ## 退化类型先验编码网络
        '''
        两个模态使用相同的网络结构，网络参数各自独立
        '''
        self.net_dp = build_network(opt['network_dp'])
        self.net_dp = self.model_to_device(self.net_dp, find_unused_parameters=True)
        
        self.net_sp = build_network(opt['network_sp'])
        self.net_sp = self.model_to_device(self.net_sp, find_unused_parameters=False)
        
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g, find_unused_parameters=False)
        
        load_path = self.opt['path'].get('pretrain_network_dp', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_dp', 'params')
            self.load_network(self.net_dp, load_path, self.opt['path'].get('strict_load_dp', True), param_key)
            
        load_path = self.opt['path'].get('pretrain_network_sp', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_sp', 'params')
            self.load_network(self.net_sp, load_path, self.opt['path'].get('strict_load_sp', True), param_key)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_dp.train()
        self.net_sp.train()
        self.net_g.train()
        # self.net_degra.eval()
        
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            print("TODO")

        # define losses
        ## 构建损失的时候是否需要把网络作为参数传入 还是直接传需要使用的特征？ 采取前者的方案
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            ## 能否使用self.mode_to_device()函数将计算指标的类函数加载到GPU上去？
            self.cri_pix = self.cri_pix.to(self.device)
            self.weight_pix = train_opt['pixel_opt'].get('weight', 1.0)
        else:
            self.cri_pix = None
            
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
        
        if train_opt.get('contrastive_opt'):
            self.cri_contra = build_loss(train_opt['contrastive_opt']).to(self.device)
            self.weight_contra = train_opt['contrastive_opt'].get('weight', 1.0)
        else:
            self.cri_contra = None
            
        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        dp_optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Network G: Params {k} will not be optimized.')

        for k, v in self.net_dp.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
                dp_optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Network C: Params {k} will not be optimized.')
        
        for k, v in self.net_sp.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Network C: Params {k} will not be optimized.')
                
        optim_type = train_opt['optim_total'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_total = torch.optim.Adam(optim_params, **train_opt['optim_total'])
        elif optim_type == 'AdamW':
            self.optimizer_total = torch.optim.AdamW(optim_params, **train_opt['optim_total'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_total)

    def feed_data(self, data):
        self.lq_ir = data['lq_ir'].to(self.device)
        self.lq_vi = data['lq_vi'].to(self.device)
        self.gt_ir = data['gt_ir'].to(self.device)
        self.gt_vi = data['gt_vi'].to(self.device)
        self.gt = torch.cat([self.gt_ir, self.gt_vi], 1)
        if 'neg' in data.keys():
            self.neg = data['neg'].to(self.device)
            self.pos_ir = data['pos_ir'].to(self.device)
            self.pos_vi = data['pos_vi'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_total.zero_grad()
        semantic_prior = self.net_sp(self.gt) ## 提取高质量的语义先验
        deg_prior_ir, deg_type_ir = self.net_dp(self.lq_ir) ## 提取红外图像的退化先验
        deg_prior_vi, deg_type_vi = self.net_dp(self.lq_vi) ## 提取红外图像的退化先验
        '''
        要先把pos_ir, pos_vi 以及neg从通道维度拆到batch维度 然后这提取 然后再拆
        # '''
        pos_irs  = torch.cat(self.pos_ir.chunk(3, dim=1), dim=0)
        pos_vis  = torch.cat(self.pos_vi.chunk(3, dim=1), dim=0)
        negs  = torch.cat(self.neg.chunk(6, dim=1), dim=0)
        
        _, deg_type_ir_pos = self.net_dp(pos_irs)
        _, deg_type_vi_pos = self.net_dp(pos_vis)
        _, deg_type_neg = self.net_dp(negs)
        deg_type_ir_pos = deg_type_ir_pos.chunk(3, dim=0)
        deg_type_vi_pos = deg_type_vi_pos.chunk(3, dim=0)
        deg_type_neg = deg_type_neg.chunk(6, dim=0) ## 负样本两个模态是通用的 再加上另外一个模态作为当前模态的负样本 来构建负样本集
        self.results = self.net_g(self.lq_ir, self.lq_vi, deg_prior_ir, deg_prior_vi, semantic_prior)
        self.output = self.results['fusion']
        self.out_ir = self.results['ir']
        self.out_vi = self.results['vi']
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_fusion, log_img = self.cri_pix(self.output, self.gt_ir, self.gt_vi)
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
        # contrastive loss
        if self.cri_contra:
            l_contra = self.cri_contra(deg_type_ir, deg_type_vi, deg_type_ir_pos, deg_type_vi_pos, deg_type_neg)
            l_total += l_contra
            loss_dict['l_contra'] = l_contra
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
            torch.nn.utils.clip_grad_norm_(list(self.net_sp.parameters()) + list(self.net_g.parameters()) + list(self.net_dp.parameters()), 0.01)
        self.optimizer_total.step()
        self.log_img = log_img
        self.log_dict = self.reduce_loss_dict(loss_dict)
        
    def test_visual(self, data):
        
        lq = data['lq'].to(self.device)
        if hasattr(self, 'net_g_ema'):
            print("TODO: wrong")
        else:
            self.net_dp.eval()
            self.net_sp.eval()
            self.net_g.eval()
            with torch.no_grad():                
                deg_prior_ir, deg_type = self.net_dp(lq) ## 提取红外图像的退化先验
                deg_prior = rearrange(deg_prior_ir, 'b n c -> b (c n)')
        return deg_prior
        
    def test(self):
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
        lq = torch.cat([img_ir, img_vi], 1)

        ## 为了适应CLIP的数据输入,需要将 输入的图像Resize到（224,224）
        gt_ir = F.pad(self.gt_ir, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        gt_vi = F.pad(self.gt_vi, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        gt = F.pad(self.gt, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        if hasattr(self, 'net_g_ema'):
            print("TODO: wrong")
        else:
            self.net_dp.eval()
            self.net_sp.eval()
            self.net_g.eval()
            with torch.no_grad():                
                semantic_prior = self.net_sp(gt) ## 提取高质量的语义先验
                # semantic_prior = self.net_sp(lq) ## 提取高质量的语义先验
                deg_prior_ir, _ = self.net_dp(img_ir) ## 提取红外图像的退化先验
                deg_prior_vi, _ = self.net_dp(img_vi) ## 提取红外图像的退化先验
                self.results = self.net_g(img_ir, img_vi, deg_prior_ir, deg_prior_vi, semantic_prior)
                self.output = self.results['fusion']
                self.out_ir = self.results['ir']
                self.out_vi = self.results['vi']
            self.net_sp.train()
            self.net_dp.train()
            self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
        self.out_ir = self.out_ir[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
        self.out_vi = self.out_vi[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
            # self.nondist_visual(dataloader, current_iter, tb_logger, save_img)

    def nondist_visual(self, dataloader, current_iter, tb_logger, save_img):
        print('*'*15, "Degradation type embedding testing", '*'*15)
        dataset_name = dataloader.dataset.opt['name']
        deg_types = []
        for idx, val_data in enumerate(dataloader):
            deg_type = self.test_visual(val_data) ##[1, 128]
            print(deg_type.unsqueeze(0).size())
            deg_types.append(deg_type)
            # print("# {} degradation type embedding".format(val_data['lq_path'][0]))
        deg_types = torch.cat(deg_types, dim=0)
        # 先将其转换为 numpy 数组
        deg_types_numpy = deg_types.detach().cpu().numpy()  
        # 保存为 .mat 文件
        save_path = '/data/timer/DSPF_Ablation/Results/deg_types_0831.mat'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        savemat(save_path, {'deg_types': deg_types_numpy})  
        print('Degradation types are saved in: {}'.format(save_path))
          
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        with_rec = self.opt['val'].get('rec_flag', False)
        print(with_rec)

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
                save_folder = osp.join(dataloader.dataset.opt['save_folder'], dataset_name, 'DSPF')
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
            self.test()

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
            self.save_network(self.net_g, 'net_g', current_iter)
            self.save_network(self.net_sp, 'net_sp', current_iter)
            self.save_network(self.net_dp, 'net_dp', current_iter)
        self.save_training_state(epoch, current_iter)
