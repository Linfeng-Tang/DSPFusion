import torch
from torch import nn as nn
from torch.nn import functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image
from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss
import torchvision.transforms as transforms
from info_nce import InfoNCE
from math import exp
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
_reduction_modes = ['none', 'mean', 'sum']

def RGB2YCrCb(rgb_image, with_CbCr=True):
    """
    Convert RGB format to YCrCb format.
    Used in the intermediate results of the color space conversion, because the default size of rgb_image is [B, C, H, W].
    :param rgb_image: image data in RGB format
    :param with_CbCr: boolean flag to determine if Cb and Cr channels should be returned
    :return: Y, CbCr (if with_CbCr is True), otherwise Y, Cb, Cr
    """
    R = rgb_image[:, 0:1, ::]
    G = rgb_image[:, 1:2, ::]
    B = rgb_image[:, 2:3, ::]
    
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.5 * B + 128/255.0
    Cr = 0.5 * R - 0.419 * G - 0.081 * B + 128/255.0

    Y = Y.clamp(0.0, 1.0)
    Cr = Cr.clamp(0.0, 1.0)
    Cb = Cb.clamp(0.0, 1.0)
    
    if with_CbCr:
        CbCr = torch.cat([Cb, Cr], dim=1)
        return Y, CbCr
    
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):
    """
    Convert YcrCb format to RGB format
    :param Y.
    :param Cb.
    :param Cr.
    :return.
    """
    R = Y + 1.402 * (Cr - 128/255.0)
    G = Y - 0.344136 * (Cb - 128/255.0) - 0.714136 * (Cr - 128/255.0)
    B = Y + 1.772 * (Cb - 128/255.0)
    
    RGB = torch.cat([R, G, B], dim=1)
    RGB = RGB.clamp(0,1.0)
    
    return RGB

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        window = self.create_window(window_size, self.channel)        
        self.register_buffer('window', window)
        
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, window):
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=img1.size(1))
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=img2.size(1))
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=img1.size(1)) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=img2.size(1)) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=img1.size(1)) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        ssim = self.ssim(img1, img2, window)
        loss = 1 - ssim
        return loss
    
@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)

    
    
@LOSS_REGISTRY.register()
class Contrastive_loss(nn.Module):
    def __init__(self, model=None):
        super(Contrastive_loss, self).__init__()
        self.loss_InfoNCE = InfoNCE(negative_mode='paired')      
        for p in self.parameters():
            p.requires_grad = False   
            
    def forward(self, anchor_A, anchor_B, pos_A, pos_B, neg):
        anchor_A = F.normalize(anchor_A, dim=-1)
        anchor_B = F.normalize(anchor_B, dim=-1)
        
        pos_A = [F.normalize(pos, dim=-1) for pos in pos_A]
        pos_B = [F.normalize(pos, dim=-1) for pos in pos_B]
        neg = [F.normalize(neg.unsqueeze(1), dim=-1) for neg in neg] ## [batch_size, num_negative, embedding_size]
        # pos_A = torch.cat(pos_A, dim=1)
        # pos_B = torch.cat(pos_B, dim=1)
        # neg_A = [*neg, anchor_B.unsqueeze(1)]
        neg_A = torch.cat([*neg, anchor_B.unsqueeze(1)], dim=1)
        neg_B = torch.cat([*neg, anchor_A.unsqueeze(1)], dim=1)
        contrastive_loss = 0
        for A_pos, B_pos in zip(pos_A, pos_B):
            contrastive_loss += self.loss_InfoNCE(anchor_A, A_pos, neg_A) + self.loss_InfoNCE(anchor_B, B_pos, neg_B)
        return contrastive_loss

@LOSS_REGISTRY.register()
class Fidelity_loss(nn.Module):
    def __init__(self):
        super(Fidelity_loss, self).__init__()
        print('Using Fidelity_loss() as loss function~')
        self.sobelconv = sobel_operation()
        self.loss_func = nn.L1Loss(reduction='mean')

    def Normalize(self, x):
        batch_size, channels, height, width = x.size()
        x_min = x.view(batch_size, channels, -1).min(dim=2, keepdim=True)[0].view(batch_size, channels, 1, 1)
        x_max = x.view(batch_size, channels, -1).max(dim=2, keepdim=True)[0].view(batch_size, channels, 1, 1)
        x = (x - x_min) / (x_max - x_min)
        return x
    
    def forward(self, img_rec, img_ref, type='ir'):
        Y_rec, CbCr_rec = RGB2YCrCb(img_rec)
        Y_ref, CbCr_ref = RGB2YCrCb(img_ref)

        loss_intensity = 50 * self.loss_func(img_rec, img_ref)

        ## 梯度损失
        grad_rec = self.sobelconv(Y_rec)
        grad_ref = self.sobelconv(Y_ref)
        loss_grad = 50 * self.loss_func(grad_rec, grad_ref)

        loss_fidelity = 1 * loss_intensity + 1  * loss_grad
        loss = {
            '{}_loss_intensity'.format(type) : loss_intensity,
            '{}_loss_grad'.format(type) : loss_grad,
            '{}_loss_fid'.format(type) : loss_fidelity
        }
        return loss   
    
@LOSS_REGISTRY.register()
class Fusion_loss(nn.Module):
    def __init__(self, net_degra=None):
        super(Fusion_loss, self).__init__()
        print('Using Fusion_loss() as loss function~')
        self.contrast_flag = False
        self.perceptual_flag = False
        self.ssim_loss = SSIMLoss()
        self.sobelconv = sobel_operation()
        self.loss_func = nn.L1Loss(reduction='mean')
        self.gamma = 1.5

    
    def forward(self, img_fusion, img_A, img_B):
        
        Y_fusion, CbCr_fusion = RGB2YCrCb(img_fusion)
        Y_A, CbCr_A = RGB2YCrCb(img_A)
        Y_B, CbCr_B = RGB2YCrCb(img_B)
        # 计算像素层面的loss，强度损失和梯度损失
        enhanced_Y_A = torch.pow(Y_A, self.gamma)
        
        Y_joint = torch.max(enhanced_Y_A, Y_B)
        loss_intensity = 50 * self.loss_func(Y_fusion, Y_joint)
        loss_color = 10 * self.loss_func(CbCr_fusion, CbCr_B)  ## 色彩损失主要约束融合结果的CbCr通道与彩色的VI图像保持一致

        ## 梯度损失
        grad_A = self.sobelconv(Y_A)
        grad_B = self.sobelconv(Y_B)
        grad_fusion = self.sobelconv(Y_fusion)
        ## 尝试梯度进行归一化  但是似乎梯度已经比较显著了？？？
        grad_fusion = (grad_fusion - torch.min(grad_fusion)) / (torch.max(grad_fusion) - torch.min(grad_fusion))
        grad_A = (grad_A - torch.min(grad_A)) / (torch.max(grad_A) - torch.min(grad_A))
        enhanced_grad_A = torch.pow(grad_A, self.gamma) ## 红外图像先拉伸再压缩 防止红外纹理被完全滤除
        grad_B = (grad_B - torch.min(grad_B)) / (torch.max(grad_B) - torch.min(grad_B))
        enhanced_grad_B = torch.pow(grad_B, 0.7) ## 突出可见光图像中的纹理细节 压缩红外图像中的纹理细节 防止噪声污染
        
        grad_joint = torch.max(enhanced_grad_A, enhanced_grad_B)
        loss_grad = 10 * self.loss_func(grad_fusion, grad_joint)
        loss_ssim = 10 * (0.6 * self.ssim_loss(Y_fusion, Y_A) + 0.5 * self.ssim_loss(Y_fusion, Y_B))
        loss_ssim = loss_grad
        # print(loss_ssim.item())
        if torch.isnan(loss_ssim).any():
            raise ValueError("NaN detected in SSIM loss computation")

        loss_per = loss_grad
        loss_fusion = 1 * loss_intensity + 1 * loss_color + 1 * loss_grad + 1 * loss_ssim
        loss = {
            'loss_intensity' : loss_intensity,
            'loss_color' : loss_color,
            'loss_grad' : loss_grad,
            # 'loss_structure' : loss_contra,
            'loss_ssim' : loss_ssim,
            'loss_fusion' : loss_fusion
        }
        img = {
            'grad_A' : enhanced_grad_A,
            'grad_B' : enhanced_grad_B,
            'grad_max': grad_joint,
            'grad_f' : grad_fusion,
            'contra_A' : enhanced_grad_A,
            'contra_B' : enhanced_grad_B,
            'contra_max': grad_joint,
            'contra_f' : grad_fusion,
        }
        return loss, img 
 
    
class sobel_operation(nn.Module):
    def __init__(self):
        super(sobel_operation, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.register_buffer('weightx', kernelx)
        self.register_buffer('weighty', kernely)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return (torch.abs(sobelx)+torch.abs(sobely)) / 2
 
    
@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
    
@LOSS_REGISTRY.register()
class fusion_loss(nn.Module):
    def __init__(self):
        super(fusion_loss, self).__init__()        
        print('Buliding fusion_loss() as loss function~')
        self.loss_func_ssim = L_SSIM(window_size=13)
        self.loss_func_Grad = GradientMaxLoss()
        self.loss_func_Max = L_Intensity_Max_RGB()
        self.loss_func_Consist = L_Intensity_Consist()
        self.loss_func_color = L_color()
    ## img_fusion, img_A, img_B
    ## max_ratio=15, consist_ratio=1, ssim_ir_ratio=1, ssim_ratio=2, ir_compose=1, color_ratio=30, text_ratio=10, max_mode="l1", consist_mode="l1", regular=False for Normal situation
    
    def forward(self, image_fused, image_infrared, image_visible, max_ratio=15, consist_ratio=1, ssim_ir_ratio=1, ssim_ratio=2, ir_compose=1, color_ratio=20, text_ratio=10, max_mode="l1", consist_mode="l1", regular=False):
        image_visible_gray = self.rgb2gray(image_visible)
        image_infrared_gray = self.rgb2gray(image_infrared)
        image_fused_gray = self.rgb2gray(image_fused)
        loss_ssim = ssim_ratio * (self.loss_func_ssim(image_visible, image_fused) + ssim_ir_ratio * self.loss_func_ssim(image_infrared_gray, image_fused_gray))
        loss_max = max_ratio * self.loss_func_Max(image_visible, image_infrared, image_fused, max_mode)
        loss_consist = consist_ratio * self.loss_func_Consist(image_visible_gray, image_infrared_gray, image_fused_gray, ir_compose, consist_mode)
        loss_color = color_ratio * self.loss_func_color(image_visible, image_fused)
        loss_text = text_ratio * self.loss_func_Grad(image_visible_gray, image_infrared_gray, image_fused_gray, regular)
        total_loss = loss_ssim + loss_max + loss_consist + loss_color + loss_text
                   
        loss = {
            'loss_intensity_max' : loss_max,
            'loss_color' : loss_color,
            'loss_grad' : loss_text,
            'loss_intensity_consist' : loss_consist,
            'loss_ssim' : loss_ssim,
            'loss_fusion' : total_loss
        }
        img = {
            'grad_A' : image_infrared[:, :1, ::],
            'grad_B' : image_infrared[:, :1, ::],
            'grad_max': image_infrared[:, :1, ::],
            'grad_f' : image_infrared[:, :1, ::],
            'contra_A' : image_infrared[:, :1, ::],
            'contra_B' : image_infrared[:, :1, ::],
            'contra_max': image_infrared[:, :1, ::],
            'contra_f' : image_infrared[:, :1, ::],
        }
        return loss, img 
        # return total_loss, loss_ssim + loss_color, loss_max, loss_consist, loss_text

    def rgb2gray(self, image):
        b, c, h, w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray

class L_Intensity_Max(nn.Module):
    def __init__(self):
        super(L_Intensity_Max, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused):
        image_intensity = torch.max(image_visible, image_infrared)
        Loss_intensity = F.l1_loss(image_intensity, image_fused)
        return Loss_intensity

class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, image_visible, image_fused):
        # Convert RGB images to YCbCr
        ycbcr_visible = self.rgb_to_ycbcr(image_visible)
        ycbcr_fused = self.rgb_to_ycbcr(image_fused)

        # Extract CbCr channels
        cb_visible = ycbcr_visible[:, 1, :, :]
        cr_visible = ycbcr_visible[:, 2, :, :]
        cb_fused = ycbcr_fused[:, 1, :, :]
        cr_fused = ycbcr_fused[:, 2, :, :]

        # Compute L1 loss on Cb and Cr channels
        loss_cb = F.l1_loss(cb_visible, cb_fused)
        loss_cr = F.l1_loss(cr_visible, cr_fused)

        # Total color loss
        loss_color = loss_cb + loss_cr

        return loss_color

    def rgb_to_ycbcr(self, image):
        r = image[:, 0, :, :]
        g = image[:, 1, :, :]
        b = image[:, 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b

        ycbcr_image = torch.stack((y, cb, cr), dim=1)

        return ycbcr_image

class L_Intensity_Max_RGB(nn.Module):
    def __init__(self):
        super(L_Intensity_Max_RGB, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused, max_mode="l1"):
        # Convert both visible and infrared images to grayscale
        gray_visible = torch.mean(image_visible, dim=1, keepdim=True)
        gray_infrared = torch.mean(image_infrared, dim=1, keepdim=True)

        # Create a mask based on grayscale intensity comparison
        mask = (gray_infrared > gray_visible).float()

        # Weighted combination of pixel values
        fused_image = mask * image_infrared + (1 - mask) * image_visible

        # Calculate L1 loss between the fused image and the target image
        if max_mode == "l1":
            Loss_intensity = F.l1_loss(fused_image, image_fused)
        else:
            Loss_intensity = F.mse_loss(fused_image, image_fused)

        return Loss_intensity

class L_Intensity_Consist(nn.Module):
    def __init__(self):
        super(L_Intensity_Consist, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused, ir_compose, consist_mode="l1"):
        if consist_mode == "l2":
            Loss_intensity = (F.mse_loss(image_visible, image_fused) + ir_compose * F.mse_loss(image_infrared, image_fused))/2
        else:
            Loss_intensity = (F.l1_loss(image_visible, image_fused) + ir_compose * F.l1_loss(image_infrared, image_fused))/2
        return Loss_intensity

class GradientMaxLoss(nn.Module):
    def __init__(self):
        super(GradientMaxLoss, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1],
                                                       [-2, 0, 2],
                                                       [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False)
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1],
                                                       [0, 0, 0],
                                                       [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False)
        self.padding = (1, 1, 1, 1)

    def forward(self, image_A, image_B, image_fuse, regular):
        gradient_A_x, gradient_A_y = self.gradient(image_A)
        gradient_B_x, gradient_B_y = self.gradient(image_B)
        # 计算融合图像的梯度
        gradient_fuse_x, gradient_fuse_y = self.gradient(image_fuse)
        # 计算梯度幅值最大值一致性损失
        if regular == True:
            loss = 5 * (torch.abs(gradient_A_x - gradient_B_x) * F.l1_loss(gradient_fuse_x, torch.max(gradient_A_x, gradient_B_x)) + torch.abs(gradient_A_y - gradient_B_y) * F.l1_loss(gradient_fuse_y, torch.max(gradient_A_y, gradient_B_y))).mean()
        else:
            loss = F.l1_loss(gradient_fuse_x, torch.max(gradient_A_x, gradient_B_x)) + F.l1_loss(gradient_fuse_y, torch.max(gradient_A_y, gradient_B_y))
        return loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x), torch.abs(gradient_y)

class EdgeTextureLoss(nn.Module):
    def __init__(self):
        super(EdgeTextureLoss, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False)
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False)
        self.padding = (1, 1, 1, 1)

    def forward(self, image_visible, image_infrared, image_fused):
        gray_visible = image_visible
        gray_infrared = image_infrared
        gray_fused = image_fused

        d1 = self.gradient(gray_visible)
        d2 = self.gradient(gray_infrared)
        df = self.gradient(gray_fused)
        edge_loss = F.l1_loss(torch.max(d1, d2), df)

        return edge_loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x) + torch.abs(gradient_y)

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.gradient = Gradient()

    def forward(self, image_visible, image_infrared, image_fused):
        image_visible_Y = self.tensor_RGB2GRAY(image_visible)
        image_infrared_Y = self.tensor_RGB2GRAY(image_infrared)
        image_fused_Y = self.tensor_RGB2GRAY(image_fused)

        gradient_visible = self.gradient(image_visible_Y)
        gradient_infrared = self.gradient(image_infrared_Y)
        gradient_fused = self.gradient(image_fused_Y)

        gradient_max = torch.max(gradient_visible, gradient_infrared)
        loss_gradient = F.mse_loss(gradient_fused, gradient_max)
        return loss_gradient

    def tensor_RGB2GRAY(self, image):
        b,c,h,w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)

        return image_gray

class Gradient(nn.Module):
    def __init__(self):
        super(Gradient, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False)
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False)

    def forward(self, x):
        gradient_x = F.conv2d(x, self.sobel_x, padding=1)
        gradient_y = F.conv2d(x, self.sobel_y, padding=1)
        return torch.abs(gradient_x) + torch.abs(gradient_y)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=24, window=None, size_average=True, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return 1 - ret


# Classes to re-use window
class L_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(L_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        (_, channel_2, _, _) = img2.size()

        if channel != channel_2 and channel == 1:
            img1 = torch.concat([img1, img1, img1], dim=1)
            channel = 3

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


def structure_loss(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1 = F.conv2d(img1, window, padding=padd, groups=channel) - mu1
    sigma2 = F.conv2d(img2, window, padding=padd, groups=channel) - mu2
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C2 = (0.03 * L) ** 2
    loss=(2*sigma12+C2)/(2*sigma1*sigma2+C2)

    if size_average:
        ret = loss.mean()
    else:
        ret = loss.mean(1).mean(1).mean(1)

    if full:
        return 1 - ret
    return ret

"""def show_img(images,imagesl, B):
    for index in range(B):
        img = images[index, :]
        c, h, w =img.shape
        if c == 1:
            img = torch.concat([img, img, img], dim=0)
        img_np = np.array(img.permute(1, 2, 0).detach().cpu())
        plt.figure(1)
        plt.title("decom")
        plt.imshow(img_np)
        img = imagesl[index, :]
        c, h, w =img.shape
        if c == 1:
            img = torch.concat([img, img, img], dim=0)
        img_np = np.array(img.permute(1, 2, 0).detach().cpu())

        plt.figure(2)
        plt.title("origin")
        plt.imshow(img_np)
        plt.show(block=True)"""

def normalize_grad(gradient_orig):
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm