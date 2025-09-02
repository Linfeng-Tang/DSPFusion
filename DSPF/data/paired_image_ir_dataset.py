from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file, paired_paths_from_folder_fusion
from DSPF.utils.transforms import augment, paired_random_crop, random_augmentation, paired_random_crop_fusion, paired_random_crop_fusion_Constr, random_augmentation_Constr
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import os
import random
from PIL import Image, ImageOps
import math
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from natsort import natsorted



@DATASET_REGISTRY.register()
class PairedImageFusionDataset_Hybrid(data.Dataset):
    """Paired image dataset for image Fusion.

    Read LQ_ir (Low-quality infrared images, e.g. LR (Low Resolution), blurry, noisy, etc) and GT_ir image pairs.
    Read LQ_vi (Low-quality visible images, e.g. LL (Low-light ), blurry, noisy, etc) and GT_vi image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageFusionDataset_Hybrid, self).__init__()
        print("Using PairedImageFusionDataset_Hybrid() to construct dataloader!!")
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.degradation = ['LC', 'Norm', 'RN', 'SN']
        # self.degradation = ['']
        self.gt_folder_ir, self.lq_folder_ir = opt['dataroot_gt_ir'], opt['dataroot_lq_ir']
        self.gt_folder_vi, self.lq_folder_vi = opt['dataroot_gt_vi'], opt['dataroot_lq_vi']
        
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            ## 默认使用这个方式获取输入图像的路径
            self.paths = paired_paths_from_folder_fusion([self.lq_folder_ir, self.gt_folder_ir, self.lq_folder_vi, self.gt_folder_vi], ['lq_ir', 'gt_ir', 'lq_vi', 'gt_vi'], self.filename_tmpl)
        # for path in self.paths:
        #     # if 'haze' in path['lq_vi_path']:
        #     print(path['lq_vi_path'])
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR; (和cv2.imwrite一致, arch处理的是RGB, 后续BRG-RGB, 保存前同样RGB-BGR)
        # image range: [0, 1], float32.
        # image range: [0, 1], float32., H W 3
        
        
        
        lq_path_vi = self.paths[index]['lq_vi_path']
        img_bytes = self.file_client.get(lq_path_vi, 'lq')
        img_lq_vi = imfrombytes(img_bytes, float32=True)
        
        lq_vi_name, ext = os.path.splitext(os.path.basename(lq_path_vi))
        name = lq_vi_name.split('_')[0]
        gt_name = name + ext
        lq_path_ir = None
        while lq_path_ir is None:
            ir_deg_opt = random.choice(self.degradation)
            if ir_deg_opt == '':
                lq_ir_name = name + ext
            else:
                lq_ir_name = name + '_' + ir_deg_opt + ext
            
            lq_path_ir = os.path.join(self.lq_folder_ir, lq_ir_name)
            if not os.path.exists(lq_path_ir):
                lq_path_ir = None
        
        
        gt_path_ir = os.path.join(self.gt_folder_ir, gt_name)
        img_bytes = self.file_client.get(gt_path_ir, 'gt')
        img_gt_ir = imfrombytes(img_bytes, float32=True)
        
        # lq_path_ir = os.path.join(self.lq_folder_ir, lq_ir_name)
        img_bytes = self.file_client.get(lq_path_ir, 'lq')
        img_lq_ir = imfrombytes(img_bytes, float32=True)
        
        gt_path_vi = os.path.join(self.gt_folder_vi, gt_name)
        img_bytes = self.file_client.get(gt_path_vi, 'gt')
        img_gt_vi = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # # padding
            # equal in deblurring
            # img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt_ir, img_lq_ir, img_gt_vi, img_lq_vi = paired_random_crop_fusion(img_gt_ir, img_lq_ir, img_gt_vi, img_lq_vi, gt_size, scale, gt_path_ir, gt_path_vi)
            # flip, rotation
            # img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
            img_gt_ir, img_lq_ir, img_gt_vi, img_lq_vi = random_augmentation(img_gt_ir, img_lq_ir, img_gt_vi, img_lq_vi)

        # # color space transform
        # if 'color' in self.opt and self.opt['color'] == 'y':
        #     img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
        #     img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt_ir = img_gt_ir[0:img_lq_ir.shape[0] * scale, 0:img_lq_ir.shape[1] * scale, :]
            img_gt_vi = img_gt_vi[0:img_lq_ir.shape[0] * scale, 0:img_lq_ir.shape[1] * scale, :]
            img_lq_ir = img_lq_ir[0:img_lq_ir.shape[0] * scale, 0:img_lq_ir.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt_ir, img_lq_ir, img_gt_vi, img_lq_vi = img2tensor([img_gt_ir, img_lq_ir, img_gt_vi, img_lq_vi], bgr2rgb=True, float32=True)
        img_ir4clip = self.clip_transform(img_gt_ir)
        img_vi4clip = self.clip_transform(img_gt_vi)
        
        # normalize        
        if self.mean is not None or self.std is not None:
            print("normalizing the input data")
            normalize(img_lq_ir, self.mean, self.std, inplace=True)
            normalize(img_gt_ir, self.mean, self.std, inplace=True)
            normalize(img_lq_vi, self.mean, self.std, inplace=True)
            normalize(img_gt_vi, self.mean, self.std, inplace=True)
            
        # print(torch.min(img_gt_vi), torch.max(img_gt_vi))
        return {'lq_ir': img_lq_ir, 'ir4clip': img_ir4clip, 'gt_ir': img_gt_ir, 'lq_vi': img_lq_vi, 'gt_vi': img_gt_vi, 'vi4clip': img_vi4clip, 'lq_path_ir': lq_path_ir, 'gt_path_ir': gt_path_ir, 'lq_path_vi': lq_path_vi, 'gt_path_vi': gt_path_vi}
    
    def clip_transform(self, tensor, resolution=224, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
        ## 将tensor转换回numpy数组
        img = tensor.clone().detach().cpu().numpy().transpose(1, 2, 0)
        ## 使用PIL 执行图像的填充与裁剪
        img = Image.fromarray((img * 255).astype('uint8'))

        width, height = img.size
        if width == height and width == 224 and height == 224:
            resized_img = img            
        else:
            # 计算调整为正方形后的大小
            if width > height:
                new_size = width
            else:
                new_size = height

            # 将图像调整为正方形
            img = img.resize((new_size, new_size))

           # 计算调整为正方形后的大小
            if width > height:
                new_size = width
                padding_height = (new_size - height) // 2
                padding = (0, padding_height, 0, new_size - height - padding_height)
            else:
                new_size = height
                padding_width = (new_size - width) // 2
                padding = (padding_width, 0, new_size - width - padding_width, 0)


            # 对图像进行填充
            padded_img = ImageOps.expand(img, padding, fill=0)

            # 调整图像大小为固定的大小
            resized_img = padded_img.resize((resolution, resolution), resample=Image.BICUBIC)

        return Compose([
            ToTensor(),
            Normalize(mean, std)
        ])(resized_img)
        
    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class PairedImageFusionDataset(data.Dataset):
    """Paired image dataset for image Fusion.

    Read LQ_ir (Low-quality infrared images, e.g. LR (Low Resolution), blurry, noisy, etc) and GT_ir image pairs.
    Read LQ_vi (Low-quality visible images, e.g. LL (Low-light ), blurry, noisy, etc) and GT_vi image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageFusionDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        # self.degradation = ['LC', 'LR']
        self.degradation = ['']
        self.lq_folder_ir = os.path.abspath(opt['dataroot_lq_ir'].replace('\\', '/'))
        self.lq_folder_vi = os.path.abspath(opt['dataroot_lq_vi'].replace('\\', '/'))
        self.gt_folder_ir = self.lq_folder_ir
        self.gt_folder_vi = self.lq_folder_vi
        
        self.paths = os.listdir(self.lq_folder_ir)
        # 获取两个文件夹中的所有文件名
        ir_files = os.listdir(self.lq_folder_ir)  # 获取红外文件夹中的所有文件名
        vi_files = os.listdir(self.lq_folder_vi)  # 获取可见光文件夹中的所有文件名
        # 保留那些在两个文件夹中都存在的文件名
        self.paths = [file_name for file_name in self.paths if file_name in ir_files and file_name in vi_files]
        # print(self.paths)
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        lq_path_vi = os.path.join(self.lq_folder_vi, self.paths[index])
        img_bytes = self.file_client.get(lq_path_vi, 'lq')
        img_lq_vi = imfrombytes(img_bytes, float32=True)
        
        lq_vi_name, ext = os.path.splitext(os.path.basename(lq_path_vi))
        name = lq_vi_name
        gt_name = name + ext
        ir_deg_opt = random.choice(self.degradation)
        if ir_deg_opt == '':
            lq_ir_name = name + ext
        else:
            lq_ir_name = name + '_' + ir_deg_opt + ext
        
        gt_path_ir = os.path.join(self.gt_folder_ir, gt_name)
        img_bytes = self.file_client.get(gt_path_ir, 'gt')
        img_gt_ir = imfrombytes(img_bytes, float32=True)
        
        lq_path_ir = os.path.join(self.lq_folder_ir, self.paths[index])
        img_bytes = self.file_client.get(lq_path_ir, 'lq')
        img_lq_ir = imfrombytes(img_bytes, float32=True)
        
        gt_path_vi = os.path.join(self.gt_folder_vi, gt_name)     
        img_bytes = self.file_client.get(gt_path_vi, 'gt')
        img_gt_vi = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt_ir, img_lq_ir, img_gt_vi, img_lq_vi = paired_random_crop_fusion(img_gt_ir, img_lq_ir, img_gt_vi, img_lq_vi, gt_size, scale, gt_path_ir, gt_path_vi)
            # flip, rotation
            img_gt_ir, img_lq_ir, img_gt_vi, img_lq_vi = random_augmentation(img_gt_ir, img_lq_ir, img_gt_vi, img_lq_vi)

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt_ir = img_gt_ir[0:img_lq_ir.shape[0] * scale, 0:img_lq_ir.shape[1] * scale, :]
            img_gt_vi = img_gt_vi[0:img_lq_ir.shape[0] * scale, 0:img_lq_ir.shape[1] * scale, :]
            img_lq_ir = img_lq_ir[0:img_lq_ir.shape[0] * scale, 0:img_lq_ir.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt_ir, img_lq_ir, img_gt_vi, img_lq_vi = img2tensor([img_gt_ir, img_lq_ir, img_gt_vi, img_lq_vi], bgr2rgb=True, float32=True)
        
        # normalize        
        if self.mean is not None or self.std is not None:
            print("normalizing the input data")
            normalize(img_lq_ir, self.mean, self.std, inplace=True)
            normalize(img_gt_ir, self.mean, self.std, inplace=True)
            normalize(img_lq_vi, self.mean, self.std, inplace=True)
            normalize(img_gt_vi, self.mean, self.std, inplace=True)            

        return {'lq_ir': img_lq_ir, 'gt_ir': img_gt_ir, 'lq_vi': img_lq_vi, 'gt_vi': img_gt_vi, 'lq_path_ir': lq_path_ir, 'gt_path_ir': gt_path_ir, 'lq_path_vi': lq_path_vi, 'gt_path_vi': gt_path_vi}

    def __len__(self):
        return len(self.paths)
    
    

@DATASET_REGISTRY.register()
class PairedImageFusionDataset_DSPF(data.Dataset):
    """Paired image dataset for image Fusion.

    Read LQ_ir (Low-quality infrared images, e.g. LR (Low Resolution), blurry, noisy, etc) and GT_ir image pairs.
    Read LQ_vi (Low-quality visible images, e.g. LL (Low-light ), blurry, noisy, etc) and GT_vi image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageFusionDataset_DSPF, self).__init__()
        print("Using PairedImageFusionDataset_DSPF() to construct dataloader!!")
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.degradation = ['LC', 'Norm', 'RN', 'SN']
        self.negs = ['LC', 'Norm', 'RN', 'SN', 'Rain', 'Blur', 'OE', 'LL', 'haze']
        self.gt_folder_ir, self.lq_folder_ir = opt['dataroot_gt_ir'], opt['dataroot_lq_ir']
        self.gt_folder_vi, self.lq_folder_vi = opt['dataroot_gt_vi'], opt['dataroot_lq_vi']
        self.Norm_flag = True
        
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            ## 默认使用这个方式获取输入图像的路径
            self.paths = paired_paths_from_folder_fusion([self.lq_folder_ir, self.gt_folder_ir, self.lq_folder_vi, self.gt_folder_vi], ['lq_ir', 'gt_ir', 'lq_vi', 'gt_vi'], self.filename_tmpl)
        
        # for path in self.paths:            
        #     # if 'haze' in path['lq_vi_path']:
        #     print(path['lq_vi_path'])
        ## 此处需要根据paths的结果来遍历文件名 然后罗列各种退化类型的路径
        if not self.Norm_flag:
            ## 此处需要根据paths的结果来遍历文件名 然后罗列各种退化类型的路径
            selected_paths = []
            for path in self.paths:
                if 'Norm' not in path['lq_vi_path']:
                    selected_paths.append(path)
            self.paths = selected_paths
        self.names = []
        self.exts = []
        for path in self.paths:
            path = path['lq_ir_path']
            name, ext = os.path.splitext(os.path.basename(path))
            name = name.split('_')[0]
            if name not in self.names:
                self.names.append(name)
                self.exts.append(ext)
        ## ir_LC_paths
        self.ir_LC_paths = []
        for name, ext in zip(self.names, self.exts):
            path = os.path.join(self.lq_folder_ir, name + '_LC' + ext)
            if os.path.exists(path):
                self.ir_LC_paths.append(path)
        ## ir_Norm_paths
        self.ir_Norm_paths = []
        for name, ext in zip(self.names, self.exts):
            path = os.path.join(self.lq_folder_ir, name + '_Norm' + ext)
            if os.path.exists(path):
                self.ir_Norm_paths.append(path)
        ## ir_RN_paths
        self.ir_RN_paths = []
        for name, ext in zip(self.names, self.exts):
            path = os.path.join(self.lq_folder_ir, name + '_RN' + ext)
            if os.path.exists(path):
                self.ir_RN_paths.append(path)
        ## ir_SN_paths
        self.ir_SN_paths = []
        for name, ext in zip(self.names, self.exts):
            path = os.path.join(self.lq_folder_ir, name + '_SN' + ext)
            if os.path.exists(path):
                self.ir_SN_paths.append(path)
        self.ir_paths = {"LC": self.ir_LC_paths, "Norm": self.ir_Norm_paths, "RN": self.ir_RN_paths, "SN": self.ir_SN_paths}
        
        ## vi_Blur_paths
        self.vi_Blur_paths = []
        for name, ext in zip(self.names, self.exts):
            path = os.path.join(self.lq_folder_vi, name + '_Blur' + ext)
            if os.path.exists(path):
                self.vi_Blur_paths.append(path)
                
        ## vi_Norm_paths
        self.vi_Norm_paths = []
        for name, ext in zip(self.names, self.exts):
            path = os.path.join(self.lq_folder_vi, name + '_Norm' + ext)
            if os.path.exists(path):
                self.vi_Norm_paths.append(path)
                
        ## vi_Rain_paths
        self.vi_Rain_paths = []
        for name, ext in zip(self.names, self.exts):
            path = os.path.join(self.lq_folder_vi, name + '_Rain' + ext)
            if os.path.exists(path):
                self.vi_Rain_paths.append(path)
                
        ## vi_RN_paths
        self.vi_RN_paths = []
        for name, ext in zip(self.names, self.exts):
            path = os.path.join(self.lq_folder_vi, name + '_RN' + ext)
            if os.path.exists(path):
                self.vi_RN_paths.append(path) 
        
        ## vi_LL_paths
        self.vi_LL_paths = []
        for name, ext in zip(self.names, self.exts):
            path = os.path.join(self.lq_folder_vi, name + '_LL' + ext)
            if os.path.exists(path):
                self.vi_LL_paths.append(path)
                
        ## vi_OE_paths
        self.vi_OE_paths = []
        for name, ext in zip(self.names, self.exts):
            path = os.path.join(self.lq_folder_vi, name + '_OE' + ext)
            if os.path.exists(path):
                self.vi_OE_paths.append(path)
        
        self.vi_haze_paths = []
        for name, ext in zip(self.names, self.exts):
            path = os.path.join(self.lq_folder_vi, name + '_haze' + ext)
            if os.path.exists(path):
                self.vi_haze_paths.append(path)
                
        self.vi_paths = {'Blur': self.vi_Blur_paths, 'Norm': self.vi_Norm_paths, 'Rain': self.vi_Rain_paths, 'RN': self.vi_RN_paths, 'LL': self.vi_LL_paths, 'OE': self.vi_OE_paths, 'haze': self.vi_haze_paths}
        
        
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        lq_path_vi = self.paths[index]['lq_vi_path']
        img_bytes = self.file_client.get(lq_path_vi, 'lq')
        img_lq_vi = imfrombytes(img_bytes, float32=True)
        
        lq_vi_name, ext = os.path.splitext(os.path.basename(lq_path_vi))
        name, vi_deg_type = lq_vi_name.split('_')[0], lq_vi_name.split('_')[-1]
        gt_name = name + ext
        
        lq_path_ir = None 
        while lq_path_ir is None:
            ir_deg_type = random.choice(self.degradation)
            if ir_deg_type == '':
                lq_ir_name = name + ext
            else:
                lq_ir_name = name + '_' + ir_deg_type + ext
            
            lq_path_ir = os.path.join(self.lq_folder_ir, lq_ir_name)
            if not os.path.exists(lq_path_ir):
                lq_path_ir = None
        
        gt_path_ir = os.path.join(self.gt_folder_ir, gt_name)
        img_bytes = self.file_client.get(gt_path_ir, 'gt')
        img_gt_ir = imfrombytes(img_bytes, float32=True)
        
        # lq_path_ir = os.path.join(self.lq_folder_ir, lq_ir_name)
        img_bytes = self.file_client.get(lq_path_ir, 'lq')
        img_lq_ir = imfrombytes(img_bytes, float32=True)
        
        gt_path_vi = os.path.join(self.gt_folder_vi, gt_name)
        img_bytes = self.file_client.get(gt_path_vi, 'gt')
        img_gt_vi = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            neg_paths = [os.path.join(folder, name + '_' + neg_deg + ext) for folder in [self.lq_folder_ir, self.lq_folder_vi] for neg_deg in self.negs]
            neg_imgs = []
            for neg_path in neg_paths:
                if not os.path.exists(neg_path) or os.path.samefile(lq_path_ir, neg_path) or os.path.samefile(lq_path_vi, neg_path) :
                    continue
                img_bytes = self.file_client.get(neg_path, 'lq')
                img_neg = imfrombytes(img_bytes, float32=True)
                neg_imgs.append(img_neg)
            ## 负样本路径已经构造出来
            ## 选择三个用于构造红外正样本的路径
            ir_pos_imgs = []
            ir_pos_paths = random.sample(self.ir_paths[ir_deg_type], 3)
            if lq_path_ir in ir_pos_paths:
                ir_pos_paths.remove(lq_path_ir)
                ir_pos_path = random.choice(self.ir_paths[ir_deg_type])
                while ir_pos_path in ir_pos_paths or os.path.samefile(ir_pos_path, lq_path_ir):                        
                    ir_pos_path = random.choice(self.ir_paths[ir_deg_type])
                ir_pos_paths.append(ir_pos_path)                
            for ir_pos_path in ir_pos_paths:
                img_bytes = self.file_client.get(ir_pos_path, 'lq')
                img_pos_ir = imfrombytes(img_bytes, float32=True)
                ir_pos_imgs.append(img_pos_ir)
            
            ## 选择三个用于构造可见光正样本的路径
            vi_pos_imgs = []
            # print(vi_deg_type, len(self.vi_paths[vi_deg_type]))
            vi_pos_paths = random.sample(self.vi_paths[vi_deg_type], 3)
            if lq_path_vi in vi_pos_paths:
                vi_pos_paths.remove(lq_path_vi)
                vi_pos_path = random.choice(self.vi_paths[vi_deg_type])
                while vi_pos_path in vi_pos_paths or os.path.samefile(lq_path_vi, vi_pos_path):                        
                    vi_pos_path = random.choice(self.vi_paths[vi_deg_type])
                vi_pos_paths.append(vi_pos_path)                
            for vi_pos_path in vi_pos_paths:
                img_bytes = self.file_client.get(vi_pos_path, 'lq')
                img_pos_vi = imfrombytes(img_bytes, float32=True)
                vi_pos_imgs.append(img_pos_vi)
            # random crop
            img_gt_ir, img_lq_ir, img_gt_vi, img_lq_vi, img_pos_irs, img_pos_vis, img_negs = paired_random_crop_fusion_Constr(img_gt_ir, img_lq_ir, ir_pos_imgs, img_gt_vi, img_lq_vi, vi_pos_imgs, neg_imgs, gt_size, scale, gt_path_ir, gt_path_vi)
            # flip, rotation
            # img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
            img_gt_ir, img_lq_ir, img_gt_vi, img_lq_vi, img_pos_irs, img_pos_vis, img_negs = random_augmentation_Constr(img_gt_ir, img_lq_ir, img_gt_vi, img_lq_vi, img_pos_irs, img_pos_vis, img_negs)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt_ir, img_lq_ir, img_gt_vi, img_lq_vi = img2tensor([img_gt_ir, img_lq_ir, img_gt_vi, img_lq_vi], bgr2rgb=True, float32=True)
        img_pos_irs = img2tensor(img_pos_irs, bgr2rgb=True, float32=True)
        img_pos_vis = img2tensor(img_pos_vis, bgr2rgb=True, float32=True)
        img_negs = img2tensor(img_negs, bgr2rgb=True, float32=True)
        
        # normalize        
        if self.mean is not None or self.std is not None:
            img_pos_irs_norm = []
            img_pos_vis_norm = []
            img_negs_norm = []
            print("normalizing the input data")
            img_lq_ir = normalize(img_lq_ir, self.mean, self.std, inplace=True)
            img_gt_ir = normalize(img_gt_ir, self.mean, self.std, inplace=True)
            img_lq_vi = normalize(img_lq_vi, self.mean, self.std, inplace=True)
            img_gt_vi = normalize(img_gt_vi, self.mean, self.std, inplace=True)
            for pos_ir in img_pos_irs:
                img_pos_irs_norm.append(normalize(pos_ir, self.mean, self.std, inplace=True))
            for pos_vi in img_pos_vis:
                img_pos_vis_norm.append(normalize(pos_vi, self.mean, self.std, inplace=True))
            for neg in img_negs:
                img_negs_norm.append(normalize(neg, self.mean, self.std, inplace=True))
        else:
            img_pos_irs_norm = img_pos_irs
            img_pos_vis_norm = img_pos_vis
            img_negs_norm = img_negs
        img_pos_irs_norm = torch.cat(img_pos_irs_norm, dim=0)
        img_pos_vis_norm= torch.cat(img_pos_vis_norm, dim=0)
        if len(img_negs_norm) >= 6:
            img_negs_norm = torch.cat(img_negs_norm[:6], dim=0)
        elif len(img_negs_norm) == 5:
            img_negs_norm = torch.cat([*img_negs_norm, img_negs_norm[0]], dim=0)        
        elif len(img_negs_norm) == 4:
            img_negs_norm = torch.cat([*img_negs_norm, img_negs_norm[0], img_negs_norm[1]], dim=0)
        elif len(img_negs_norm) == 3:
            img_negs_norm = torch.cat([*img_negs_norm, *img_negs_norm], dim=0)
        else:
            img_negs_norm = torch.cat(img_negs_norm[:6], dim=0)
        # img_negs_norm = torch.cat(img_negs_norm, dim=0)
        # print(torch.min(img_gt_vi), torch.max(img_gt_vi))
        return {'lq_ir': img_lq_ir, 'gt_ir': img_gt_ir, 'lq_vi': img_lq_vi, 'gt_vi': img_gt_vi, 'lq_path_ir': lq_path_ir, 'gt_path_ir': gt_path_ir, 'lq_path_vi': lq_path_vi, 'gt_path_vi': gt_path_vi, 'img_pos_ir': img_pos_irs_norm, 'img_pos_vi': img_pos_vis_norm, 'img_neg': img_negs_norm}
    
    def clip_transform(self, tensor, resolution=224, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
        ## 将tensor转换回numpy数组
        img = tensor.clone().detach().cpu().numpy().transpose(1, 2, 0)
        ## 使用PIL 执行图像的填充与裁剪
        img = Image.fromarray((img * 255).astype('uint8'))

        width, height = img.size
        if width == height and width == 224 and height == 224:
            resized_img = img            
        else:
            # 计算调整为正方形后的大小
            if width > height:
                new_size = width
            else:
                new_size = height

            # 将图像调整为正方形
            img = img.resize((new_size, new_size))

           # 计算调整为正方形后的大小
            if width > height:
                new_size = width
                padding_height = (new_size - height) // 2
                padding = (0, padding_height, 0, new_size - height - padding_height)
            else:
                new_size = height
                padding_width = (new_size - width) // 2
                padding = (padding_width, 0, new_size - width - padding_width, 0)


            # 对图像进行填充
            padded_img = ImageOps.expand(img, padding, fill=0)

            # 调整图像大小为固定的大小
            resized_img = padded_img.resize((resolution, resolution), resample=Image.BICUBIC)

        return Compose([
            ToTensor(),
            Normalize(mean, std)
        ])(resized_img)
        
    def __len__(self):
        return len(self.paths)
    
    