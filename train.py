import datetime
import logging
import math
import random
import time
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str
from DSPF.utils.options import parse_options
from torchvision.utils import make_grid
import os.path as osp
import numpy as np
import torch.distributed as dist


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project') is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, 'should turn on tensorboard when using wandb'
        init_wandb_logger(opt)

    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        # 获取当前时间，并格式化为字符串
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        # 创建新的日志目录，包含时间戳
        log_dir = osp.join(opt['root_path'], 'tb_logger', opt['name'], current_time)
        tb_logger = init_tb_logger(log_dir=log_dir)
    return tb_logger

def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumberinit_distributed_mode of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else: 
        device_id = torch.cuda.current_device()
        # resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        resume_state = torch.load(resume_state_path, map_location='cpu')
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # create model
    model = build_model(opt)
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    # progressive training
    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')

    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])

    logger_j = [True] * len(groups)
    
    scale = opt['scale']
    

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            if current_iter >= total_iters:
                break
            current_iter += 1
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
    
            ### ------Progressive learning ---------------------
            j = ((current_iter>groups) !=True).nonzero()[0]
            if len(j) == 0:
                bs_j = len(groups) - 1
            else:
                bs_j = j[0]

            mini_gt_size = mini_gt_sizes[bs_j]
            mini_batch_size = mini_batch_sizes[bs_j]
            if logger_j[bs_j]:
                logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(mini_gt_size, mini_batch_size*torch.cuda.device_count())) 
                logger_j[bs_j] = False

            lq_ir = train_data['lq_ir']
            gt_ir = train_data['gt_ir']
            lq_vi = train_data['lq_vi']
            gt_vi = train_data['gt_vi']
            Constr_flag = False
            if 'img_neg' in train_data.keys():
                Constr_flag = True
                neg = train_data['img_neg']
                pos_ir = train_data['img_pos_ir']
                pos_vi = train_data['img_pos_vi']

            if mini_batch_size < batch_size:
                indices = random.sample(range(0, batch_size), k=mini_batch_size)##随机选取mini_batch_size个样本进行训练
                lq_ir = lq_ir[indices]
                gt_ir = gt_ir[indices]
                lq_vi = lq_vi[indices]
                gt_vi = gt_vi[indices]
                if Constr_flag:
                    neg = neg[indices]
                    pos_ir = pos_ir[indices]
                    pos_vi = pos_vi[indices]

            if mini_gt_size < gt_size:
                x0 = int((gt_size - mini_gt_size) * random.random())
                
                y0 = int((gt_size - mini_gt_size) * random.random())
                x1 = x0 + mini_gt_size
                y1 = y0 + mini_gt_size
                lq_ir = lq_ir[:,:,x0:x1,y0:y1]
                gt_ir = gt_ir[:,:,x0*scale:x1*scale,y0*scale:y1*scale]
                lq_vi = lq_vi[:,:,x0:x1,y0:y1]
                gt_vi = gt_vi[:,:,x0*scale:x1*scale,y0*scale:y1*scale]
                if Constr_flag:
                    neg = neg[:,:,x0:x1,y0:y1]
                    pos_ir = pos_ir[:,:,x0*scale:x1*scale,y0*scale:y1*scale]
                    pos_vi = pos_vi[:,:,x0*scale:x1*scale,y0*scale:y1*scale]
            ###-------------------------------------------
            
            # training
            if Constr_flag:
                model.feed_data({'lq_ir': lq_ir, 'gt_ir':gt_ir, 'lq_vi': lq_vi, 'gt_vi':gt_vi, 'neg':neg, 'pos_ir':pos_ir, 'pos_vi':pos_vi})
            else:
                model.feed_data({'lq_ir': lq_ir, 'gt_ir':gt_ir, 'lq_vi': lq_vi, 'gt_vi':gt_vi})
            model.optimize_parameters(current_iter)
            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
                ## 在这里尝试自己写 把图片写入tensorboard的代码
                if tb_logger is not None and hasattr(model, 'out_ir') and hasattr(model, 'out_vi'):
                    tb_img = [lq_ir[0, ::].detach().float().cpu(), lq_vi[0, ::].detach().float().cpu(), gt_ir[0, ::].detach().float().cpu(), gt_vi[0, ::].detach().float().cpu(), model.out_ir[0, ::].detach().float().cpu(), model.out_vi[0, ::].detach().float().cpu(),model.output[0, ::].detach().float().cpu()]
                    tb_img = make_grid(tb_img, nrow=7, padding=2)
                elif tb_logger is not None:
                    tb_img = [lq_ir[0, ::].detach().float().cpu(), lq_vi[0, ::].detach().float().cpu(), gt_ir[0, ::].detach().float().cpu(), gt_vi[0, ::].detach().float().cpu(), model.output[0, ::].detach().float().cpu()]
                    tb_img = make_grid(tb_img, nrow=5, padding=2)
                if tb_logger is not None:
                    tb_logger.add_image('images', tb_img, current_iter)
                if tb_logger is not None and hasattr(model, 'log_img'):
                    tb_contra = [model.log_img['contra_A'][0, ::].repeat(3, 1, 1).detach().float().cpu(), model.log_img['contra_B'][0, ::].repeat(3, 1, 1).detach().float().cpu(), model.log_img['contra_max'][0, ::].repeat(3, 1, 1).detach().float().cpu(), model.log_img['contra_f'][0, ::].repeat(3, 1, 1).detach().float().cpu(),]
                    tb_contra = make_grid(tb_contra, nrow=4, padding=2)
                    tb_logger.add_image('contra', tb_contra, current_iter)
                    tb_detail = [model.log_img['grad_A'][0, ::].repeat(3, 1, 1).detach().float().cpu(), model.log_img['grad_B'][0, ::].repeat(3, 1, 1).detach().float().cpu(), model.log_img['grad_max'][0, ::].repeat(3, 1, 1).detach().float().cpu(), model.log_img['grad_f'][0, ::].repeat(3, 1, 1).detach().float().cpu()]
                    tb_detail = make_grid(tb_detail, nrow=4, padding=2)
                    tb_logger.add_image('details', tb_detail, current_iter)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
        # end of iter
        if current_iter >= total_iters:
            break

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest

    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)

