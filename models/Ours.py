# -*- coding:utf-8 -*-
from models.blocks import *
import torch
from torch import nn
import numpy as np


class Ours(nn.Module):
    """Deep residual network with spatial attention for face SR.
    # Arguments: 实参
        - n_ch: base convolution channels
        - down_steps: how many times to downsample in the encoder 下采样多少次
        - res_depth: depth of residual layers in the main body  主体残差层的深度
        - up_res_depth: depth of residual layers in each upsample block  每个上采样模块的残差层的深度

    """
    def __init__(
        self,
        min_ch=32,
        max_ch=128,
        in_size=128,
        out_size=128,
        min_feat_size=16,
        res_depth=10,
        relu_type='leakyrelu',
        norm_type='bn',
        att_name='spar',
        bottleneck_size=4,
    ):
        super(Ours, self).__init__()
        nrargs = {'norm_type': norm_type, 'relu_type': relu_type}

        ch_clip = lambda x: max(min_ch, min(x, max_ch))

        down_steps = int(np.log2(in_size // min_feat_size))  # 3
        up_steps = int(np.log2(out_size // min_feat_size))  # 3
        n_ch = ch_clip(max_ch // int(np.log2(in_size // min_feat_size) + 1))

        # ------------ define encoder --------------------
        self.encoder = []
        self.encoder.append(ConvLayer(3, n_ch, 3, 1))   # 第一层卷积
        hg_depth = int(np.log2(64 / bottleneck_size))
        for i in range(down_steps):
            cin, cout = ch_clip(n_ch), ch_clip(n_ch * 2)
            self.encoder.append(ResidualBlock(cin, cout, scale='down', hg_depth=hg_depth, att_name=att_name, **nrargs))

            n_ch = n_ch * 2
            hg_depth = hg_depth - 1
        hg_depth = hg_depth + 1
        self.encoder = nn.Sequential(*self.encoder)

        # ------------ define residual layers --------------------
        # self.res_layers = []
        # for i in range(res_depth + 3 - down_steps):
        #     channels = ch_clip(n_ch)
        #     self.res_layers.append(ResidualBlock(channels, channels, hg_depth=hg_depth, att_name=att_name, **nrargs))
        # self.res_layers = nn.Sequential(*self.res_layers)

        channels = ch_clip(n_ch)
        self.R0 = ResidualBlock(channels, channels, hg_depth=hg_depth, att_name=att_name, **nrargs)

        channels = ch_clip(n_ch)
        self.R1 = ResidualBlock(channels, channels, hg_depth=hg_depth, att_name=att_name, **nrargs)

        channels = ch_clip(n_ch)
        self.R2 = ResidualBlock(channels, channels, hg_depth=hg_depth, att_name=att_name, **nrargs)

        channels = ch_clip(n_ch)
        self.R3 = ResidualBlock(channels, channels, hg_depth=hg_depth, att_name=att_name, **nrargs)

        channels = ch_clip(n_ch)
        self.R4 = ResidualBlock(channels, channels, hg_depth=hg_depth, att_name=att_name, **nrargs)

        channels = ch_clip(n_ch)
        self.R5 = ResidualBlock(channels, channels, hg_depth=hg_depth, att_name=att_name, **nrargs)

        channels = ch_clip(n_ch)
        self.R6 = ResidualBlock(channels, channels, hg_depth=hg_depth, att_name=att_name, **nrargs)

        channels = ch_clip(n_ch)
        self.R7 = ResidualBlock(channels, channels, hg_depth=hg_depth, att_name=att_name, **nrargs)

        channels = ch_clip(n_ch)
        self.R8 = ResidualBlock(channels, channels, hg_depth=hg_depth, att_name=att_name, **nrargs)

        channels = ch_clip(n_ch)
        self.R9 = ResidualBlock(channels, channels, hg_depth=hg_depth, att_name=att_name, **nrargs)

        self.fusion = nn.Conv2d(1280, 128, kernel_size=1)
        #------------ define decoder --------------------
        self.decoder = []
        for i in range(up_steps):
            hg_depth = hg_depth + 1
            cin, cout = ch_clip(n_ch), ch_clip(n_ch // 2)
            self.decoder.append(ResidualBlock(cin, cout, scale='up', hg_depth=hg_depth, att_name=att_name, **nrargs))

            n_ch = n_ch // 2
        self.decoder = nn.Sequential(*self.decoder)

        self.out_conv = ConvLayer(ch_clip(n_ch), 3, 3, 1)   # 最后一层卷积
    
    def forward(self, input_img):
        out = self.encoder(input_img)
        # out = self.res_layers(out)
        r0 = self.R0(out)
        r1 = self.R1(r0)
        r2 = self.R2(r1)
        r3 = self.R3(r2)
        r4 = self.R4(r3)
        r5 = self.R5(r4)
        r6 = self.R6(r5)
        r7 = self.R7(r6)
        r8 = self.R8(r7)
        r9 = self.R9(r8)
        # r10 = self.R10(r9)
        # r11 = self.R11(r10)
        # r12 = self.R12(r11)
        # r13 = self.R13(r12)
        r10 = torch.cat([r0, r1, r2, r3, r4, r5, r6, r7, r8, r9], 1)
        fusion = self.fusion(r10)
        # r10 = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9
        out = self.decoder(fusion)
        # out_img = self.out_conv(out)  #增加一条add分支
        out_img = self.out_conv(out) + input_img
        return out_img