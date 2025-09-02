import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToPILImage
import os
from natsort import  natsorted
import matplotlib.pyplot as plt
import numpy as np
# def visual_weight_map(I):
#     img = (I * 255).to(dtype=torch.uint8).cuda()
#     count = torch.histc(img.flatten().float(), bins=256, min=0, max=255)

#     i_values = torch.arange(256).float().cuda()
#     Sal_Tab = torch.abs(i_values.view(1, -1) - i_values.view(-1, 1))
#     Sal_Tab = torch.matmul(count, Sal_Tab)

#     out = Sal_Tab.view(-1)[img.view(-1)]  # Flatten Sal_Tab before indexing

#     out = torch.div(out, torch.max(out))  # Normalize to [0, 1]
#     return out.cpu()
def visual_weight_map(I):
    img = (I * 255).to(dtype=torch.uint8).cuda()
    count = torch.histc(img.flatten().float(), bins=256, min=0, max=255)

    Sal_Tab = torch.zeros(256).cuda()
    for j in range(256):
        for i in range(256):
            Sal_Tab[j] = Sal_Tab[j] + count[i].item() * abs(j - i)

    out = torch.zeros_like(img, dtype=torch.float32).cuda()
    for i in range(256):
        out[img == i] = Sal_Tab[i]

    out = torch.div(out, torch.max(out))  # Normalize to [0, 1]
    return out.cpu()
ir_folder = '/data/timer/HI-Diff/datasets/MSRS/train/ir'
vi_folder = '/data/timer/HI-Diff/datasets/MSRS/train/vi_enhanced'
ir_weight_folder = '/data/timer/HI-Diff/datasets/MSRS/train/ir_weights'
vi_weight_folder = '/data/timer/HI-Diff/datasets/MSRS/train/vi_weights'
os.makedirs(ir_weight_folder, exist_ok=True)
os.makedirs(vi_weight_folder, exist_ok=True)
filelist = natsorted(os.listdir(ir_folder))
to_pil = ToPILImage()
for item in filelist:
    ir_path = os.path.join(ir_folder, item)
    vi_path = os.path.join(vi_folder, item)
    ir_weight_path = os.path.join(ir_weight_folder, item)
    vi_weight_path = os.path.join(vi_weight_folder, item)
    ir_img = Image.open(ir_path).convert('L')  # 转为灰度图
    vi_img = Image.open(vi_path).convert('L')
    ir_tensor = torch.tensor(np.array(ir_img), dtype=torch.float32) / 255.0
    vi_tensor = torch.tensor(np.array(vi_img), dtype=torch.float32) / 255.0
    # 获取视觉权重图
    ir_weight_map = visual_weight_map(ir_tensor)
    vi_weight_map = 1 - ir_weight_map#visual_weight_map(vi_tensor)
    
    # Combine the tensors into a single tensor
    # combined_weights = torch.stack([ir_weight_map / 0.5, vi_weight_map / 0.5], dim=0)

    # # Apply softmax along the specified dimension (axis)
    # normalized_weights = F.softmax(combined_weights, dim=0)

    # # Access the normalized weights for each original tensor
    # ir_weight_map = normalized_weights[0]
    # vi_weight_map = normalized_weights[1]
    # Convert the tensor to a PIL Image
    ir_weight_map = to_pil(ir_weight_map)
    vi_weight_map = to_pil(vi_weight_map)
    ir_weight_map.save(ir_weight_path)
    vi_weight_map.save(vi_weight_path)
    print(item)