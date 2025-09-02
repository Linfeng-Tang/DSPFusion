import cv2
import random
import numpy as np
import torch


def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs

def paired_random_crop_fusion(img_gts_ir, img_lqs_ir, img_gts_vi, img_lqs_vi, gt_patch_size, scale, gt_path_ir=None, gt_path_vi=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts_ir, list):
        img_gts_ir = [img_gts_ir]
    if not isinstance(img_lqs_ir, list):
        img_lqs_ir = [img_lqs_ir]
    if not isinstance(img_gts_vi, list):
        img_gts_vi = [img_gts_vi]
    if not isinstance(img_lqs_vi, list):
        img_lqs_vi = [img_lqs_vi]
    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts_ir[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq_ir, w_lq_ir = img_lqs_ir[0].size()[-2:]
        h_gt_ir, w_gt_ir = img_gts_ir[0].size()[-2:]
        h_lq_vi, w_lq_vi = img_lqs_vi[0].size()[-2:]
        h_gt_vi, w_gt_vi = img_gts_vi[0].size()[-2:]
    else:
        h_lq_ir, w_lq_ir = img_lqs_ir[0].shape[0:2]
        h_gt_ir, w_gt_ir = img_gts_ir[0].shape[0:2]
        h_lq_vi, w_lq_vi = img_lqs_vi[0].shape[0:2]
        h_gt_vi, w_gt_vi = img_gts_vi[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt_ir != h_lq_ir * scale or w_gt_ir != w_lq_ir * scale :
        raise ValueError(f'Scale mismatches in Infrared images. GT ({h_gt_ir}, {w_gt_ir}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq_ir}, {w_lq_ir}).')
    if h_gt_vi != h_lq_vi * scale or w_gt_vi != w_lq_vi * scale:        
        raise ValueError(f'Scale mismatches in Visible images. GT ({h_gt_vi}, {w_gt_vi}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq_vi}, {w_lq_vi}).')
    if h_lq_ir < lq_patch_size or w_lq_ir < lq_patch_size:
        raise ValueError(f'LQ ({h_lq_ir}, {w_lq_ir}) of Infrared images is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path_ir}.')
    if h_lq_vi < lq_patch_size or w_lq_vi < lq_patch_size:
        raise ValueError(f'LQ ({h_lq_vi}, {w_lq_vi}) of Infrared images is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path_vi}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq_ir - lq_patch_size)
    left = random.randint(0, w_lq_ir - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs_ir = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs_ir]
    else:
        img_lqs_ir = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs_ir]

    if input_type == 'Tensor':
        img_lqs_vi = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs_vi]
    else:
        img_lqs_vi = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs_vi]
    
    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    
    if input_type == 'Tensor':
        img_gts_ir = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts_ir]
    else:
        img_gts_ir = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts_ir]    
        
        
    if input_type == 'Tensor':
        img_gts_vi = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts_vi]
    else:
        img_gts_vi = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts_vi]    
    
    if len(img_gts_ir) == 1:
        img_gts_ir = img_gts_ir[0]
    if len(img_lqs_ir) == 1:
        img_lqs_ir = img_lqs_ir[0]
    
    if len(img_gts_vi) == 1:
        img_gts_vi = img_gts_vi[0]
    if len(img_lqs_vi) == 1:
        img_lqs_vi = img_lqs_vi[0]
    imgs = [img_gts_ir, img_lqs_ir, img_gts_vi, img_lqs_vi]
    return imgs

def get_patch(imgs, input_type, top=0, left=0, patch_size=256):    # crop lq patch
    if input_type == 'Tensor':
        imgs = [v[:, :, top:top + patch_size, left:left + patch_size] for v in imgs]
    else:
        imgs = [v[top:top + patch_size, left:left + patch_size, ...] for v in imgs]
    return imgs

def paired_random_crop_fusion_Constr(img_gts_ir, img_lqs_ir, ir_pos_imgs, img_gts_vi, img_lqs_vi, vi_pos_imgs, img_negs, gt_patch_size, scale, gt_path_ir=None, gt_path_vi=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts_ir, list):
        img_gts_ir = [img_gts_ir]
    if not isinstance(img_lqs_ir, list):
        img_lqs_ir = [img_lqs_ir]
    if not isinstance(img_gts_vi, list):
        img_gts_vi = [img_gts_vi]
    if not isinstance(img_lqs_vi, list):
        img_lqs_vi = [img_lqs_vi]
    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts_ir[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq_ir, w_lq_ir = img_lqs_ir[0].size()[-2:]
    else:
        h_lq_ir, w_lq_ir = img_lqs_ir[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale
    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq_ir - lq_patch_size)
    left = random.randint(0, w_lq_ir - lq_patch_size)
    pos_irs = []
    pos_vis = []
    for img in ir_pos_imgs:        
        if input_type == 'Tensor':
            h_deg, w_deg = img.size()[-2:]
        else:
            h_deg, w_deg = img.shape[0:2]
        top_pos = random.randint(0, h_deg - lq_patch_size)
        left_pos = random.randint(0, w_deg - lq_patch_size)
        pos_ir = get_patch(imgs=[img], input_type=input_type, top=top_pos, left=left_pos, patch_size=lq_patch_size)
        if len(img_gts_ir) == 1:
            pos_ir = pos_ir[0]
        pos_irs.append(pos_ir)
    for img in vi_pos_imgs:   
        if input_type == 'Tensor':
            h_deg, w_deg = img.size()[-2:]
        else:
            h_deg, w_deg = img.shape[0:2]
        top_pos = random.randint(0, h_deg - lq_patch_size)
        left_pos = random.randint(0, w_deg - lq_patch_size)
        pos_vi = get_patch(imgs=[img], input_type=input_type, top=top_pos, left=left_pos, patch_size=lq_patch_size)
        if len(img_gts_ir) == 1:
            pos_vi = pos_vi[0]
        pos_vis.append(pos_vi)   
        
    img_lqs_ir = get_patch(imgs=img_lqs_ir, input_type=input_type, top=top, left=left, patch_size=lq_patch_size)
    img_lqs_vi = get_patch(imgs=img_lqs_vi, input_type=input_type, top=top, left=left, patch_size=lq_patch_size)
    img_gts_ir = get_patch(imgs=img_gts_ir, input_type=input_type, top=top, left=left, patch_size=lq_patch_size)
    img_gts_vi = get_patch(imgs=img_gts_vi, input_type=input_type, top=top, left=left, patch_size=lq_patch_size)    
    if len(img_gts_ir) == 1:
        img_gts_ir = img_gts_ir[0]
    if len(img_lqs_ir) == 1:
        img_lqs_ir = img_lqs_ir[0]
    if len(img_gts_vi) == 1:
        img_gts_vi = img_gts_vi[0]
    if len(img_lqs_vi) == 1:
        img_lqs_vi = img_lqs_vi[0]
     
    patch_negs = []
    for img_neg in img_negs:
        patch_neg = get_patch(imgs=[img_neg], input_type=input_type, top=top, left=left, patch_size=lq_patch_size)
        if len(patch_neg) == 1:
            patch_neg = patch_neg[0]
        patch_negs.append(patch_neg)
        
    imgs = [img_gts_ir, img_lqs_ir, img_gts_vi, img_lqs_vi, pos_irs, pos_vis, patch_negs]
    return imgs



def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img


def data_augmentation(image, mode):
    """
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    """
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out

def random_augmentation(*args):
    out = []
    flag_aug = random.randint(0,7)
    for data in args:    
        out.append(data_augmentation(data, flag_aug).copy())
    return out

def random_augmentation_Constr(*args):
    out = []
    data_pos_irs = []
    data_pos_vis = []
    data_negs = []
    flag_aug = random.randint(0,7)
    for i, data in enumerate(args):
        if i == len(args) - 1 and isinstance(data, list):
            for data_neg in data:
                data_negs.append(data_augmentation(data_neg, flag_aug).copy())
            out.append(data_negs)
        elif i == len(args) - 2 and isinstance(data, list):
            for data_pos_vi in data:
                data_pos_vis.append(data_augmentation(data_pos_vi, flag_aug).copy())
            out.append(data_pos_vis)
        
        elif i == len(args) - 3 and isinstance(data, list):
            for data_pos_ir in data:
                data_pos_irs.append(data_augmentation(data_pos_ir, flag_aug).copy())
            out.append(data_pos_irs)
        else:
            out.append(data_augmentation(data, flag_aug).copy())
    return out