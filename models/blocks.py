# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np

# import torch
# from torch import nn
# import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair


# 归一化层
class NormLayer(nn.Module):
    """Normalization Layers.
    ------------
    # Arguments
        - channels: input channels, for batch norm and instance norm.
        - input_size: input shape without batch size, for layer norm.
    """
    def __init__(self, channels, normalize_shape=None, norm_type='bn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=True)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(32, channels, affine=True)
        elif norm_type == 'pixel':
            self.norm = lambda x: F.normalize(x, p=2, dim=1)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(normalize_shape)
        elif norm_type == 'none':
            self.norm = lambda x: x
        else:
            assert 1==0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x):
        return self.norm(x)

# RelU激活函数
class ReluLayer(nn.Module):
    """Relu Layer.
    ------------
    # Arguments
        - relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu 
            - SELU
            - none: direct pass
    """
    def __init__(self, channels, relu_type='relu'):
        super(ReluLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'selu':
            self.func = nn.SELU(True)
        elif relu_type == 'none':
            self.func = lambda x: x
        else:
            assert 1==0, 'Relu type {} not support.'.format(relu_type)

    def forward(self, x):
        return self.func(x)

# 卷积
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale='none', norm_type='none', relu_type='none', use_pad=True):
        super(ConvLayer, self).__init__()
        self.use_pad = use_pad
        
        bias = True if norm_type in ['pixel', 'none'] else False 
        stride = 2 if scale == 'down' else 1

        self.scale_func = lambda x: x
        if scale == 'up':
            self.scale_func = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='nearest')

        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2) 
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.relu = ReluLayer(out_channels, relu_type)
        self.norm = NormLayer(out_channels, norm_type=norm_type)

    def forward(self, x):
        out = self.scale_func(x)
        if self.use_pad:
            out = self.reflection_pad(out)
        out = self.conv2d(out)
        out = self.norm(out)
        out = self.relu(out)
        return out

# 加上沙漏块和空间注意力的残差网络
class ResidualBlock(nn.Module):
    """
    Residual block recommended in: http://torch.ch/blog/2016/02/04/resnets.html
    ------------------
    # Args
        - hg_depth: depth of HourGlassBlock. 0: don't use attention map.
        - use_pmask: whether use previous mask as HourGlassBlock input.
    """
    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, att_name='spar'):
        super(ResidualBlock, self).__init__()
        self.c_in      = c_in
        self.c_out     = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth  = hg_depth

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        # 1、进行一个残差
        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)

        # 2、BN + Prelu
        self.preact_func = nn.Sequential(
                    NormLayer(c_in, norm_type=self.norm_type),
                    ReluLayer(c_in, self.relu_type),
                    )

        if scale == 'down':
            scales = ['none', 'down']
        elif scale == 'up':
            scales = ['up', 'none']
        elif scale == 'none':
            scales = ['none', 'none']
        # 3、FAU 第一个卷积层  3*3  scale=none , BN + Prelu
        self.conv1 = ConvLayer(c_in, c_out, 3, scales[0], **kwargs)
        # 4、FAU 第二个卷积层  3*3  scale=none , BN
        self.conv2 = ConvLayer(c_out, c_out, 3, scales[1], norm_type=norm_type, relu_type='none')

        # 5、注意力选择 spar 还是 spar3d ,3d更轻量级
        if att_name.lower() == 'spar':
            c_attn = 1  # 选择spar的话，输出通道是1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out  # 选择spar3d，输出通道和输入通道相同，故称为3d注意力？
        else:
            raise Exception("Attention type {} not implemented".format(att_name))
        # 5、沙漏网络
        self.att_func = HourGlassBlock(self.hg_depth, c_out, c_attn, **kwargs) 
        
    def forward(self, x):
        identity = self.shortcut_func(x)  # shortcut
        out = self.preact_func(x)  # 预激活函数处理
        out = self.conv1(out)  # 经过第一个卷积
        out = self.conv2(out)  # 经过第二个卷积
        out = identity + self.att_func(out)  # 相加
        return out
        
# 沙漏网络
class HourGlassBlock(nn.Module):
    """Simplified HourGlass block.
    Reference: https://github.com/1adrianb/face-alignment
    --------------------------
    """
    def __init__(self, depth, c_in, c_out,
            c_mid=64,
            norm_type='bn',
            relu_type='prelu',
            ):
        super(HourGlassBlock, self).__init__()
        self.depth     = depth
        self.c_in      = c_in
        self.c_mid     = c_mid
        self.c_out     = c_out
        self.kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        if self.depth:
            self._generate_network(self.depth)  # 沙漏块网络
            self.out_block_ISA = Internal_feature_Split_Attention(64, 64, kernel_size=3, stride=1, padding=1, dilation=1,
                                                              groups=1, bias=False, radix=4, norm_layer=nn.BatchNorm2d)
            # 从沙漏模块中出来，经过一个3*3 卷积 和 sigmoid函数
            self.out_block = nn.Conv2d(64, 1, kernel_size=1)
            # self.out_block = nn.Sequential(
            #         ConvLayer(self.c_mid, self.c_out, norm_type='none', relu_type='none'),
            #                     # input = 64，output=1 时 是2d注意力，
            #                     # input = 64，output=64 时，是3d注意力？
            #         nn.Sigmoid()
            #         )

    def _generate_network(self, level):
        if level == self.depth:
            c1, c2 = self.c_in, self.c_mid
        else:
            c1, c2 = self.c_mid, self.c_mid

        self.add_module('b1_' + str(level), ConvLayer(c1, c2, **self.kwargs))
        self.add_module('b2_' + str(level), ConvLayer(c1, c2, scale='down', **self.kwargs))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvLayer(self.c_mid, self.c_mid, **self.kwargs))

        self.add_module('b3_' + str(level), ConvLayer(self.c_mid, self.c_mid, scale='up', **self.kwargs))

    def _forward(self, level, in_x):
        up1 = self._modules['b1_' + str(level)](in_x)
        low1 = self._modules['b2_' + str(level)](in_x)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = self._modules['b2_plus_' + str(level)](low1)

        up2 = self._modules['b3_' + str(level)](low2)
        if up1.shape[2:] != up2.shape[2:]:
            up2 = nn.functional.interpolate(up2, up1.shape[2:])

        return up1 + up2

    def forward(self, x, pmask=None):
        if self.depth == 0: return x
        input_x = x  # 输入沙漏网络的特征x
        x = self._forward(self.depth, x)  # 经过沙漏块输出特征
        isa = self.out_block_ISA(x)
        self.att_map = self.out_block(isa)  # 经过空间注意力处理后的特征
        x = input_x * self.att_map  # 输入沙漏网络的特征x与经过沙漏块和空间注意力处理过后的特征相乘
        return x


# 加上沙漏块和空间注意力的残差网络
class ResidualBlock1(nn.Module):
    """
    Residual block recommended in: http://torch.ch/blog/2016/02/04/resnets.html
    ------------------
    # Args
        - hg_depth: depth of HourGlassBlock. 0: don't use attention map.
        - use_pmask: whether use previous mask as HourGlassBlock input.
    """

    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, att_name='spar'):
        super(ResidualBlock1, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth = hg_depth

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        # 1、进行一个残差
        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)

        # 2、BN + Prelu
        self.preact_func = nn.Sequential(
            NormLayer(c_in, norm_type=self.norm_type),
            ReluLayer(c_in, self.relu_type),
        )

        if scale == 'down':
            scales = ['none', 'down']
        elif scale == 'up':
            scales = ['up', 'none']
        elif scale == 'none':
            scales = ['none', 'none']
        # 3、FAU 第一个卷积层  3*3  scale=none , BN + Prelu
        self.conv1 = ConvLayer(c_in, c_out, 3, scales[0], **kwargs)
        # 4、FAU 第二个卷积层  3*3  scale=none , BN
        self.conv2 = ConvLayer(c_out, c_out, 3, scales[1], norm_type=norm_type, relu_type='none')

        # 5、注意力选择 spar 还是 spar3d ,3d更轻量级
        if att_name.lower() == 'spar':
            c_attn = 1  # 选择spar的话，输出通道是1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out  # 选择spar3d，输出通道和输入通道相同，故称为3d注意力？
        else:
            raise Exception("Attention type {} not implemented".format(att_name))
        # 5、沙漏网络
        self.att_func = HourGlassBlock1(self.hg_depth, c_out, c_attn, **kwargs)

    def forward(self, x):
        identity = self.shortcut_func(x)  # shortcut
        out = self.preact_func(x)  # 预激活函数处理
        out = self.conv1(out)  # 经过第一个卷积
        out = self.conv2(out)  # 经过第二个卷积
        out = identity + self.att_func(out)  # 相加
        return out


#  换成特征注意力的沙漏快
class HourGlassBlock1(nn.Module):
    """Simplified HourGlass block.
    Reference: https://github.com/1adrianb/face-alignment
    --------------------------
    """

    def __init__(self, depth, c_in, c_out,
                 c_mid=64,
                 norm_type='bn',
                 relu_type='prelu',
                 ):
        super(HourGlassBlock1, self).__init__()
        self.depth = depth
        self.c_in = c_in
        self.c_mid = c_mid
        self.c_out = c_out
        self.kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        if self.depth:
            self._generate_network(self.depth)  # 沙漏块网络
            # 从沙漏模块中出来，经过一个3*3 卷积 和 sigmoid函数
            self.out_block = nn.Sequential(
                ConvLayer(self.c_mid, self.c_out, norm_type='none', relu_type='none'),
                # input = 64，output=1 时 是2d注意力，
                # input = 64，output=64 时，是3d注意力？
                nn.Sigmoid()
            )

    def _generate_network(self, level):
        if level == self.depth:
            c1, c2 = self.c_in, self.c_mid
        else:
            c1, c2 = self.c_mid, self.c_mid

        self.add_module('b1_' + str(level), ConvLayer(c1, c2, **self.kwargs))
        self.add_module('b2_' + str(level), ConvLayer(c1, c2, scale='down', **self.kwargs))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvLayer(self.c_mid, self.c_mid, **self.kwargs))

        self.add_module('b3_' + str(level), ConvLayer(self.c_mid, self.c_mid, scale='up', **self.kwargs))

    def _forward(self, level, in_x):
        up1 = self._modules['b1_' + str(level)](in_x)
        low1 = self._modules['b2_' + str(level)](in_x)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = self._modules['b2_plus_' + str(level)](low1)

        up2 = self._modules['b3_' + str(level)](low2)
        if up1.shape[2:] != up2.shape[2:]:
            up2 = nn.functional.interpolate(up2, up1.shape[2:])

        return up1 + up2

    def forward(self, x, pmask=None):
        if self.depth == 0: return x
        input_x = x  # 输入沙漏网络的特征x
        x = self._forward(self.depth, x)  # 经过沙漏块输出特征
        self.att_map = self.out_block(x)  # 经过空间注意力处理后的特征
        x = input_x * self.att_map  # 输入沙漏网络的特征x与经过沙漏块和空间注意力处理过后的特征相乘
        return x


# ISA 注意力机制
class Internal_feature_Split_Attention(Module):

    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True, radix=2, reduction_factor=4, norm_layer=None, **kwargs):
        super(Internal_feature_Split_Attention, self).__init__()
        padding = _pair(padding)
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                             groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.conv1x1_1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.conv1x1_2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.conv1x1_1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.conv1x1_2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x