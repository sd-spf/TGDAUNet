# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:58:14 2021

@author: angelou
"""
import torchvision.models.densenet
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.pretrain.Res2Net_v1b import  res2net101_v1b_26w_4s
import math
import torchvision.models as models
from Models.conv_layer import Conv, BNPReLU
from Models.partial_decoder import aggregation

from Models.PSA import PSA_p
from Models.swin_transformer import SwinTransformer
from Models.CGRmodes.CGR import CGRModule

    
class TGDAUNet(nn.Module):
    def __init__(self, channel=32):
        super().__init__()
        
         # ---- ResNet Backbone ----
        self.resnet = res2net101_v1b_26w_4s(pretrained=True)

        # Receptive Field Block
        self.rfb2_1 = Conv(512, 32,3,1,padding=1,bn_acti=True)
        self.rfb3_1 = Conv(1024, 32,3,1,padding=1,bn_acti=True)
        self.rfb4_1 = Conv(2048, 32,3,1,padding=1,bn_acti=True)

        self.rfb2_1t = Conv(256, 32, 3, 1, padding=1, bn_acti=True)
        self.rfb3_1t = Conv(512, 32, 3, 1, padding=1, bn_acti=True)
        self.rfb4_1t = Conv(1024, 32, 3, 1, padding=1, bn_acti=True)
        # Partial Decoder
        self.agg1 = aggregation(channel)

        self.trans=SwinTransformer(
            hidden_dim=128,
            layers=[2,2,18,2],
            heads=[4, 8, 16, 32],
            channels=3,
            num_classes=1,
            window_size=12,
            downscaling_factors=[4,2,2,2],
            relative_pos_embedding=True
        )



        self.ra1_conv1 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra2_conv1 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.ra3_conv1 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)
        
        self.aa_kernel_1 = PSA_p(32,32)
        self.aa_kernel_2 = PSA_p(32,32)
        self.aa_kernel_3 = PSA_p(32,32)
        self.up3=nn.Sequential(
            Conv(1024, 512,3,1,padding=1,bn_acti=True),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True))
        self.up4 = nn.Sequential(
            Conv(2048, 1024, 3, 1, padding=1, bn_acti=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

        self.upconv2 = Conv(512, 512, 3, 1, padding=1, bn_acti=True)
        self.upconv3 = Conv(1024, 1024, 3, 1, padding=1, bn_acti=True)
        self.upconv4 = Conv(2048, 2048, 3, 1, padding=1, bn_acti=True)

        self.doubleup=nn.Sequential(
            nn.Upsample(scale_factor=4,mode='bilinear',align_corners=True),
            Conv(2048, 512,3,1,padding=1,bn_acti=True)
        )
        self.downup2= Conv(512, 1024, 3, 2, padding=1, bn_acti=True)
        self.downup3 =Conv(1024, 2048,3,2,padding=1,bn_acti=True)
        self.doubleupdowm=nn.Sequential(
            Conv(512, 1024, 3, 2, padding=1, bn_acti=True),
            Conv(1024, 2048,3,2,padding=1,bn_acti=True))

        self.grath = CGRModule(32,32,4)

    def forward(self, x):
        xa=x
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        
        # ----------- low-level features -------------
        
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)# bs, 2048, 11, 11

        x1t=self.trans.stage1(xa)
        x2t=self.trans.stage2(x1t)
        x3t=self.trans.stage3(x2t)
        x4t=self.trans.stage4(x3t)
        x2td=self.rfb2_1t(x2t)
        x3td = self.rfb3_1t(x3t)
        x4td = self.rfb4_1t(x4t)

        x3_rfbup = self.up3(x3)  # 32 32 32
        x4_rfbdoubleup=self.doubleup(x4)  #32 32 32
        x4_rfbup=self.up4(x4)#32 16 16
        x2_rfbdoublep=self.doubleupdowm(x2) # 32 10 10
        x2_rfbdp=self.downup2(x2) # 32 16 16
        x3_rfbdp=self.downup3(x3)
        # x2_rfb=torch.cat([x2,x3_rfbup,x4_rfbdoubleup],dim=1)
        x2_rfb=x2+x3_rfbup+x4_rfbdoubleup
        # x3_rfb=torch.cat([x3,x2_rfbdp,x4_rfbup],dim=1)
        x3_rfb=x3+x2_rfbdp+x4_rfbup
        # x4_rfb=torch.cat([x4,x2_rfbdoublep,x3_rfbdp],dim=1)
        x4_rfb=x4+x2_rfbdoublep+x3_rfbdp
        x2_rfb=self.upconv2(x2_rfb)
        x3_rfb = self.upconv3(x3_rfb)
        x4_rfb = self.upconv4(x4_rfb)
        x2_rfb = self.rfb2_1(x2_rfb) # 512 - 32 32 32
        x3_rfb = self.rfb3_1(x3_rfb) # 1024 - 32 16 16
        x4_rfb = self.rfb4_1(x4_rfb) # 2048 - 32 8 8

        decoder_1 = self.agg1(x4_rfb,x3_rfb,x2_rfb)
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=8, mode='bilinear')# #20 1 256 256

        # ------------------- atten-one -----------------------

        #PSA3
        aa_atten_3 = self.aa_kernel_3(x4_rfb)
        aa_atten_3=aa_atten_3+x4td

        # aa_atten_3=self.coconv(aa_atten_3)
        #PSA2
        aa_atten_2 = self.aa_kernel_2(x3_rfb)
        aa_atten_2=aa_atten_2+x3td
        # aa_atten_2=self.coconv(aa_atten_2)
        #PSA1
        aa_atten_1 = self.aa_kernel_1(x2_rfb)
        aa_atten_1=aa_atten_1+x2td
        # aa_atten_1=self.coconv(aa_atten_1)
        #RCF1
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.25, mode='bilinear')
        # decoder_2g=self.grathconv(decoder_2)
        grath3 = self.grath(aa_atten_3,decoder_2)
        aa_atten_3=aa_atten_3.mul(grath3)
        # decoder_2co = decoder_2
        # aa_atten_3c = self.coconv(aa_atten_3)
        # aa_atten_3=aa_atten_3c+aa_atten_3


        decoder_2_ra = -1 * (torch.sigmoid(decoder_2)) + 1
        aa_atten_3_o = decoder_2_ra.expand(-1, 32, -1, -1).mul(aa_atten_3)
        ra_3 = self.ra3_conv1(aa_atten_3_o)  # 32 - 32
        ra_3 = self.ra3_conv2(ra_3)  # 32 - 32
        ra_3 = self.ra3_conv3(ra_3)  # 32 - 1
        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3, scale_factor=32, mode='bilinear')
        #RCF2



        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear')
        # aa_atten_3c= F.interpolate(aa_atten_3c,scale_factor=2,mode='bilinear')
        # decoder_3g = self.grathconv(decoder_3)
        grath2 = self.grath(aa_atten_2, decoder_3)
        aa_atten_2=aa_atten_2.mul(grath2)
        # aa_atten_2c = self.coconv(aa_atten_3c+aa_atten_2)
        # aa_atten_2= aa_atten_2c+aa_atten_2

        decoder_3_ra = -1 * (torch.sigmoid(decoder_3)) + 1
        aa_atten_2_o = decoder_3_ra.expand(-1, 32, -1, -1).mul(aa_atten_2)
        ra_2 = self.ra2_conv1(aa_atten_2_o)  # 32 - 32
        ra_2 = self.ra2_conv2(ra_2)  # 32 - 32
        ra_2 = self.ra2_conv3(ra_2)  # 32 - 1
        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2, scale_factor=16, mode='bilinear')
        # RCF1


        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear')
        # aa_atten_2c=F.interpolate(aa_atten_2c, scale_factor=2, mode='bilinear')
        # decoder_4g = self.grathconv(decoder_4)

        grath1=self.grath(aa_atten_1,decoder_4)
        aa_atten_1=aa_atten_1.mul(grath1)
        # decoder_4co = decoder_4
        # aa_atten_1c = self.coconv(aa_atten_2c+ aa_atten_1)
        # aa_atten_1=aa_atten_1c+aa_atten_1
        decoder_4_ra = -1 * (torch.sigmoid(decoder_4)) + 1
        aa_atten_1_o = decoder_4_ra.expand(-1, 32, -1, -1).mul(aa_atten_1)
        ra_1 = self.ra1_conv1(aa_atten_1_o)  # 32 - 32
        ra_1 = self.ra1_conv2(ra_1)  # 32 - 32
        ra_1 = self.ra1_conv3(ra_1)  # 32 - 1
        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1, scale_factor=8, mode='bilinear')
        return lateral_map_5,lateral_map_3,lateral_map_2,lateral_map_1
    


if __name__ == '__main__':
    ras = TGDAUNet().cuda()
    input_tensor = torch.randn(1, 3, 384, 384).cuda()

    out = ras(input_tensor)