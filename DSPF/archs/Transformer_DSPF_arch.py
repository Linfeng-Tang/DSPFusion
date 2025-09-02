## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from einops.layers.torch import Rearrange

from basicsr.utils.registry import ARCH_REGISTRY


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# w/o shape
class LayerNorm_Without_Shape(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm_Without_Shape, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return self.body(x)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
## Proposed in Restormer
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, embed_dim, group):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # prior
        if group == 1:
            self.ln1 = nn.Linear(embed_dim*4, dim)
            self.ln2 = nn.Linear(embed_dim*4, dim)

    def forward(self, x, prior=None):
        if prior is not None:
            k1 = self.ln1(prior).unsqueeze(-1).unsqueeze(-1)
            k2 = self.ln2(prior).unsqueeze(-1).unsqueeze(-1)
            x = (x * k1) + k2

        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
## Standard channel-based Attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, embed_dim, group):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # prior
        if group == 1:
            self.ln1 = nn.Linear(embed_dim*4, dim)
            self.ln2 = nn.Linear(embed_dim*4, dim)

    def forward(self, x, prior=None):
        b,c,h,w = x.shape
        if prior is not None:
            k1 = self.ln1(prior).unsqueeze(-1).unsqueeze(-1)
            k2 = self.ln2(prior).unsqueeze(-1).unsqueeze(-1)
            x = (x * k1) + k2 ## Similar to SPADE

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        ## q : c x (hw) k : (hw x c)
        ## attn: c x c
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
## Multi-DConv Head Transposed Self-Attention (MDTA)
## Standard channel-based Cross-Attention
class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNorm_type):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.norm = LayerNorm(dim, LayerNorm_type)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x_A, x_B):
        b,c,h,w = x_A.shape
        _X_A = x_A
        x_A = self.norm(x_A)
        x_B = self.norm(x_B)
        q = self.q_dwconv(self.q(x_A))
        kv = self.kv_dwconv(self.kv(x_B))
        k,v = kv.chunk(2, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        ## q : c x (hw) k : (hw x c)
        ## attn: c x c
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return _X_A + out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return x
    

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
        
class Prior_Fusion(nn.Module):
    def __init__(self, dim, num_heads, bias, embed_dim, group):
        super(Prior_Fusion, self).__init__()
        self.num_heads = num_heads
        self.ca = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear(group*group, 1),
            Rearrange('b c n -> b n c'),
            nn.Linear(embed_dim*4, dim*2),
            Rearrange('b n c -> b c n'),
        )
        self.sa = SpatialAttention()
        self.sigmoid = nn.Sigmoid()
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x_A, x_B, prior=None):
        ca  =self.ca(prior).unsqueeze(-1)
        ca_A, ca_B = ca.chunk(2, dim=1)
        sa_A = self.sa(x_A)
        sa_B = self.sa(x_B)
        x = self.sigmoid(ca_A *  sa_A) * x_A + self.sigmoid(ca_B * sa_B) * x_B
        out = self.project_out(x)
        return out

class SPMM(nn.Module):
    ## Semantic Prior Modulation module
    def __init__(self, dim, num_heads, bias, embed_dim, LayerNorm_type, qk_scale=None):
        super(SPMM, self).__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.norm1 = LayerNorm_Without_Shape(dim, LayerNorm_type)
        self.norm2 = LayerNorm_Without_Shape(embed_dim*4, LayerNorm_type)

        self.q = nn.Linear(dim, dim, bias=bias)
        self.kv = nn.Linear(embed_dim*4, 2*dim, bias=bias)
        
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, prior):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        _x = self.norm1(x)
        prior = self.norm2(prior)
        
        q = self.q(_x)
        kv = self.kv(prior)
        k,v = kv.chunk(2, dim=-1)   

        q = rearrange(q, 'b n (head c) -> b head n c', head=self.num_heads)
        k = rearrange(k, 'b n (head c) -> b head n c', head=self.num_heads)
        v = rearrange(v, 'b n (head c) -> b head n c', head=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head n c -> b n (head c)', head=self.num_heads)
        out = self.proj(out)

        # sum
        x = x + out
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()

        return x

class DPMM(nn.Module):
    ## Degradation Prior modulation module
    def __init__(self, dim, num_heads, bias, embed_dim, LayerNorm_type, group=1):
        super(DPMM, self).__init__()

        self.num_heads = num_heads
        self.group = group
        self.weight_linear = nn.Linear(dim, group * group)
        
        self.prior_norm = LayerNorm_Without_Shape(embed_dim*4, LayerNorm_type)
        self.prior_linear = nn.Linear(embed_dim*4, 2*dim, bias=bias)
        
        self.conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        
    def forward(self, x, prior):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prior_weights = F.softmax(self.weight_linear(emb), dim=1)
        prior = prior * prior_weights.unsqueeze(-1)
        prior = torch.sum(prior, dim=1)
        params = self.prior_linear(self.prior_norm(prior))
        alpha, beta = params.chunk(2, dim=-1)
        x = x * alpha.unsqueeze(-1).unsqueeze(-1) + beta.unsqueeze(-1).unsqueeze(-1)
        x = self.conv3x3(x)
        return x

##########################################################################
## Hierarchical Integration Module
class HIM(nn.Module):
    def __init__(self, dim, num_heads, bias, embed_dim, LayerNorm_type, qk_scale=None, group=None):
        super(HIM, self).__init__()
        self.group = group
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.norm1 = LayerNorm_Without_Shape(dim, LayerNorm_type)
        self.norm2_A = LayerNorm_Without_Shape(embed_dim*4, LayerNorm_type)
        self.down_A = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear(group*group, 1),
            Rearrange('b c n -> b n c'),
        )
        self.norm2_B = LayerNorm_Without_Shape(embed_dim*4, LayerNorm_type)
        self.down_B = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear(group*group, 1),
            Rearrange('b c n -> b n c'),
        )

        self.q = nn.Linear(dim, dim, bias=bias)
        ## 为什么embed_dim*4 要×4
        self.kv_A = nn.Linear(embed_dim*4, 2*dim, bias=bias)        
        self.proj_A = nn.Linear(dim, dim, bias=True)
        
        self.kv_B = nn.Linear(embed_dim*4, 2*dim, bias=bias)        
        self.proj_B = nn.Linear(dim, dim, bias=True)
        
        ## feature modulation
        self.kernel_A = nn.Sequential(
            nn.Linear(embed_dim*4, dim*2, bias=False),
        )
        self.kernel_B = nn.Sequential(
            nn.Linear(embed_dim*4, dim*2, bias=False),
        )
        
        
    def forward(self, x, prior_A, prior_B):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        _x = self.norm1(x)
        prior_A = self.norm2_A(prior_A)
        prior_B = self.norm2_B(prior_B)
        Parms_A = self.down_A(prior_A)
        Parms_A = self.kernel_A(Parms_A.squeeze()).view(-1, C * 2, 1, 1)
        alpha_A, beta_A = Parms_A.chunk(2, dim=1)
        Parms_B = self.down_B(prior_B)
        Parms_B = self.kernel_B(Parms_B.squeeze()).view(-1, C * 2, 1, 1)
        alpha_B, beta_B = Parms_B.chunk(2, dim=1)
        
        q = self.q(_x)
        kv_A = self.kv_A(prior_A)
        k_A,v_A = kv_A.chunk(2, dim=-1)
        kv_B = self.kv_B(prior_B)
        k_B, v_B = kv_B.chunk(2, dim=-1)      

        q = rearrange(q, 'b n (head c) -> b head n c', head=self.num_heads)
        k_A = rearrange(k_A, 'b n (head c) -> b head n c', head=self.num_heads)
        v_A = rearrange(v_A, 'b n (head c) -> b head n c', head=self.num_heads)
        
        k_B = rearrange(k_B, 'b n (head c) -> b head n c', head=self.num_heads)
        v_B = rearrange(v_B, 'b n (head c) -> b head n c', head=self.num_heads)

        attn_A = (q @ k_A.transpose(-2, -1)) * self.scale
        attn_A = attn_A.softmax(dim=-1)

        out_A = (attn_A @ v_A)
        out_A = rearrange(out_A, 'b head n c -> b n (head c)', head=self.num_heads)
        out_A = self.proj_A(out_A)

        
        attn_B = (q @ k_B.transpose(-2, -1)) * self.scale
        attn_B = attn_B.softmax(dim=-1)

        out_B = (attn_B @ v_B)
        out_B = rearrange(out_B, 'b head n c -> b n (head c)', head=self.num_heads)
        out_B = self.proj_B(out_B)
        # sum
        x = x + out_A + out_B
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()
        ## fetaure modulation
        x_A = x * alpha_A + beta_A
        x_B = x * alpha_B + beta_B
        x = x_A + x_B        
        return x


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, group):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, embed_dim, group)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, embed_dim, group)

    def forward(self, x, prior=None):
        x = x + self.attn(self.norm1(x), prior)
        x = x + self.ffn(self.norm2(x), prior)

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=True), 
                                  nn.Conv2d(n_feat//2, n_feat//2, kernel_size=3, stride=1, padding=1, bias=True)
                                  )

    def forward(self, x):
        return self.body(x)


class BasicLayer_Decoder(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, num_blocks, group, with_contra=False):

        super().__init__()
        self.group = group
        self.with_contra = with_contra
        # build blocks
        ## 这里是Transformer Block的构造 不需要管
        self.blocks = nn.ModuleList([TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group) for i in range(num_blocks)])
        if self.group > 1:
            self.spmm = SPMM(dim, num_heads, bias, embed_dim, LayerNorm_type)
            if with_contra:
                self.contra_att_sem = ContrastAttention(dim, dim)

    def forward(self, x, prior=None):
        # First inject the prior
        if prior is not None and self.group > 1:
            x = self.spmm(x, prior)
            if self.with_contra:
                x = self.contra_att_sem(x)
        prior=None
        ## Then pass through Transformer Blocks
        for blk in self.blocks:
            x = blk(x, prior)
                
        return x
class ContrastAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super(ContrastAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # 计算每个通道的标准差作为对比度
        channel_std = torch.std(x, dim=(2,3))
        
        # 线性映射
        attention_weights = self.fc(channel_std)
        
        # 将权重应用到输入特征图上
        attended_features = torch.einsum('bchw,bc->bchw', x, attention_weights)
        
        return attended_features
        
class BasicLayer_Encoder(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, type_embed_dim, num_blocks, group, with_contra=False):

        super().__init__()
        self.group = group
        self.with_contra = with_contra
        # build blocks
        ## 这里是Transformer Block的构造 不需要管
        self.blocks = nn.ModuleList([TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group) for i in range(num_blocks)])
        if self.group > 1:
            self.spmm = SPMM(dim, num_heads, bias, embed_dim, LayerNorm_type)
            self.dpmm = DPMM(dim, num_heads, bias, type_embed_dim, LayerNorm_type, group=group)
            self.cross_att = Cross_Attention(dim, num_heads, bias, LayerNorm_type)
            if with_contra:
                self.contra_att_sem = ContrastAttention(dim, dim)
                self.contra_att_deg = ContrastAttention(dim, dim)
            

    def forward(self, x, prior_sem=None, prior_deg=None):
        # First inject the prior
        if prior_sem is not None and prior_deg is not None and self.group > 1:
            x_sem = self.spmm(x, prior_sem)
            x_deg = self.dpmm(x, prior_deg)
            if self.with_contra:
                x_sem = self.contra_att_sem(x_sem)
                x_deg = self.contra_att_deg(x_deg)
            x = self.cross_att(x_sem, x_deg)
        prior = None
        ## Then pass through Transformer Blocks
        for blk in self.blocks:
            x = blk(x, prior)
                
        return x
    

class SFP_Header(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, group, task_name='fusion', out_channels=3, num_blocks=4):
        super().__init__()
        self.task_name = task_name
        if task_name in ['vi', 'ir']:
            self.CA = ChannelAttention(dim, ratio=16)            
            self.refinement = BasicLayer_Decoder(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks, group=group)
        else:   
            self.refinement = BasicLayer_Decoder(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks+1, group=group)

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        
    def forward(self, x):
        if self.task_name in ['vi', 'ir']:
            x = self.CA(x) * x
        x = self.refinement(x)
        x = self.output(x)
        # x = self.Tanh(self.output(x))
        return x

##########################################################################
# The implementation builds on Restormer code https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py
@ARCH_REGISTRY.register()
class Transformer_DSPF(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        embed_dim = 48,
        type_embed_dim=32, 
        group=4,
        with_contra=False,
        with_SFP=False,
    ):

        super(Transformer_DSPF, self).__init__()
         # multi-scale
        self.down_1 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear(group*group, (group*group)//4),
            Rearrange('b c n -> b n c'),
            nn.Linear(embed_dim*4, embed_dim*4)
        )
        self.down_2 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear((group*group)//4, (group*group)//4),
            Rearrange('b c n -> b n c'),
            nn.Linear(embed_dim*4, embed_dim*4)
        )
        self.type_down_1 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear(group*group, (group*group)//4),
            Rearrange('b c n -> b n c'),
            nn.Linear(type_embed_dim*4, type_embed_dim*4)
        )
        
        self.type_down_2 = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear((group*group)//4, (group*group)//4),
            Rearrange('b c n -> b n c'),
            nn.Linear(type_embed_dim*4, type_embed_dim*4)
        )


        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = BasicLayer_Encoder(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, type_embed_dim=type_embed_dim, num_blocks=num_blocks[0], group=group, with_contra=with_contra)
        self.fusion_level1 = Prior_Fusion(dim=dim, num_heads=heads[0], bias=bias, embed_dim=embed_dim, group=group)
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = BasicLayer_Encoder(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, type_embed_dim=type_embed_dim, num_blocks=num_blocks[1], group=group//2, with_contra=with_contra)
        self.fusion_level2 = Prior_Fusion(dim=int(dim*2**1), num_heads=heads[1], bias=bias, embed_dim=embed_dim, group=group//2)

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = BasicLayer_Encoder(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, type_embed_dim=type_embed_dim, num_blocks=num_blocks[2], group=group//2, with_contra=with_contra)
        self.fusion_level3 = Prior_Fusion(dim=int(dim*2**2), num_heads=heads[2], bias=bias, embed_dim=embed_dim, group=group//2)

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = BasicLayer_Encoder(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, type_embed_dim=type_embed_dim, num_blocks=num_blocks[3], group=group//2, with_contra=with_contra)
        self.fusion_latent = Prior_Fusion(dim=int(dim*2**3), num_heads=heads[3], bias=bias, embed_dim=embed_dim, group=group//2)

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = BasicLayer_Decoder(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[2], group=group//2, with_contra=with_contra)


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = BasicLayer_Decoder(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[1], group=group//2, with_contra=with_contra)

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = BasicLayer_Decoder(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_blocks[0], group=group, with_contra=with_contra)
        self.with_SFP = with_SFP
        if with_SFP:
            self.f_header = SFP_Header(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group, task_name='fusion', out_channels=out_channels, num_blocks=num_refinement_blocks)
            self.vi_header = SFP_Header(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group, task_name='vi', out_channels=out_channels, num_blocks=num_refinement_blocks)
            self.ir_header = SFP_Header(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group, task_name='ir', out_channels=out_channels, num_blocks=num_refinement_blocks)
        else:
            self.refinement = BasicLayer_Decoder(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, num_blocks=num_refinement_blocks, group=group, with_contra=with_contra)                
            self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.Tanh = nn.Tanh()

    def forward(self, ir_img, vi_img, deg_prior_A=None, deg_prior_B=None, semantic_prior=None):
        # multi-scale prior 
        img_A = ir_img
        img_B = vi_img
        
        # multi-scale prior
        prior_1_A = deg_prior_A ##[B, group * group, C]
        prior_2_A = self.type_down_1(prior_1_A) ##[B, group * group // 4, C]
        prior_3_A = self.type_down_2(prior_2_A) ##[B, group * group // 4, C]
        
        prior_1_B = deg_prior_B
        prior_2_B = self.type_down_1(prior_1_B)
        prior_3_B = self.type_down_2(prior_2_B)
        
        prior_1 = semantic_prior
        prior_2 = self.down_1(prior_1)
        prior_3 = self.down_2(prior_2)

        inp_enc_level1_A = self.patch_embed(img_A)
        out_enc_level1_A = self.encoder_level1(inp_enc_level1_A, prior_1, prior_1_A)
        
        inp_enc_level1_B = self.patch_embed(img_B)
        out_enc_level1_B = self.encoder_level1(inp_enc_level1_B, prior_1, prior_1_B)
        
        out_enc_level1_fusion = self.fusion_level1(out_enc_level1_A, out_enc_level1_B, prior_1)
        
        inp_enc_level2_A = self.down1_2(out_enc_level1_A)
        out_enc_level2_A = self.encoder_level2(inp_enc_level2_A, prior_2, prior_2_A)
        
        inp_enc_level2_B = self.down1_2(out_enc_level1_B)
        out_enc_level2_B = self.encoder_level2(inp_enc_level2_B, prior_2, prior_2_B)
        
        out_enc_level2_fusion = self.fusion_level2(out_enc_level2_A, out_enc_level2_B, prior_2)

        inp_enc_level3_A = self.down2_3(out_enc_level2_A)
        out_enc_level3_A = self.encoder_level3(inp_enc_level3_A, prior_3, prior_3_A)         
        
        inp_enc_level3_B = self.down2_3(out_enc_level2_B)
        out_enc_level3_B = self.encoder_level3(inp_enc_level3_B, prior_3, prior_3_B) 
        
        out_enc_level3_fusion = self.fusion_level3(out_enc_level3_A, out_enc_level3_B, prior_3)

        inp_enc_level4_A = self.down3_4(out_enc_level3_A)        
        latent_A = self.latent(inp_enc_level4_A, prior_3, prior_3_A) 
        
        inp_enc_level4_B = self.down3_4(out_enc_level3_B) 
        latent_B = self.latent(inp_enc_level4_B, prior_3, prior_3_B) 
        
        latent_fusion = self.fusion_latent(latent_A, latent_B, prior_3)
                        
        inp_dec_level3 = self.up4_3(latent_fusion)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3_fusion], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3, prior_3) ##[B, group * group // 4, C]

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2_fusion], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2, prior_2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1_fusion], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1, prior_1)
        if self.with_SFP:
            vi_dec = self.vi_header(out_dec_level1) + img_B 
            ir_dec = self.ir_header(out_dec_level1) + img_A 
            f_dec = self.f_header(out_dec_level1) + img_B 
             # out_dec_level1 = out_dec_level1.clamp(0, 1)
            f_img = (self.Tanh(f_dec) + 1) / 2
            ir_img = (self.Tanh(ir_dec) + 1) / 2
            vi_img = (self.Tanh(vi_dec) + 1) / 2
            # out_dec_level1 = (out_dec_level1 - torch.min(out_dec_level1)) / (torch.max(out_dec_level1) - torch.min(out_dec_level1))
            results = {'fusion':f_img, 'ir':ir_img, 'vi':vi_img}
            return results
        else:
            out_dec_level1 = self.refinement(out_dec_level1, prior_1)    
            f_dec = self.output(out_dec_level1) + img_B ## 是否跳接？ 尝试一下不跳接的效果            
            f_img = (self.Tanh(f_dec) + 1) / 2
            results = {'fusion':f_img, 'ir':f_img, 'vi':f_img}
            return results
