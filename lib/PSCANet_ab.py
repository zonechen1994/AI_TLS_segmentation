import torch
import torch.nn as nn
import torch.nn.functional as F
from .res2net import res2net50_v1b_26w_4s
from .att_modules import RFE_Block, CSHAM
from .conv_modules import BasicConv2d, get_model_shape 
from .EfficientNet import *

class RefUnet(nn.Module):
    def __init__(self,in_ch=1, K=32):
        super(RefUnet, self).__init__()
        # Encoder
        self.conv0 = nn.Conv2d(in_ch,K,3,padding=1)

        self.conv1 = nn.Conv2d(K,K,3,padding=1)
        self.bn1 = nn.BatchNorm2d(K)
        self.selu1 = nn.SELU()
        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(K,K,3,padding=1)
        self.bn2 = nn.BatchNorm2d(K)
        self.selu2 = nn.SELU()
        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(K,K,3,padding=1)
        self.bn3 = nn.BatchNorm2d(K)
        self.selu3 = nn.SELU()
        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(K,K,3,padding=1)
        self.bn4 = nn.BatchNorm2d(K)
        self.selu4 = nn.SELU()
        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)
        self.conv_d0 = nn.Conv2d(K,1,3,padding=1)

        self.conv5 = nn.Conv2d(K,K,3,padding=1)
        self.bn5 = nn.BatchNorm2d(K)
        self.selu5 = nn.SELU()

        ##### Decoder
        self.decoder4 = CSHAM(K, groups=4)
        self.decoder3 = CSHAM(K*2, groups=4)
        self.decoder2 = CSHAM(K*2, groups=4)
        self.decoder1 = CSHAM(K*2, groups=4)
        
        self.decoder4_conv = nn.Conv2d(K,K,3,padding=1)
        self.decoder3_conv = nn.Conv2d(K*2,K,3,padding=1)
        self.decoder2_conv = nn.Conv2d(K*2,K,3,padding=1)
        self.decoder1_conv = nn.Conv2d(K*2,K,3,padding=1)
        
        self.rfe4 = RFE_Block(K, K)
        self.rfe3 = RFE_Block(K, K)
        self.rfe2 = RFE_Block(K, K)
        self.rfe1 = RFE_Block(K, K)
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear') 
    def forward(self,x):
        hx = x
        hx = self.conv0(hx)

        hx1 = self.selu1(self.bn1(self.conv1(hx)))
        hx1 = self.pool1(hx1)

        hx2 = self.selu2(self.bn2(self.conv2(hx1)))
        hx2 = self.pool2(hx2)

        hx3 = self.selu3(self.bn3(self.conv3(hx2)))
        hx3 = self.pool3(hx3)

        hx4 = self.selu4(self.bn4(self.conv4(hx3)))
        hx4 = self.pool4(hx4)
        
        hx_4 = self.rfe4(hx4)
        hx_3 = self.rfe3(hx3)
        hx_2 = self.rfe2(hx2)
        hx_1 = self.rfe1(hx1)
        
        d4 = self.decoder4_conv(self.decoder4(hx_4))
        d3 = self.decoder3_conv(self.decoder3(torch.cat((self.upscore2(d4), hx_3), 1)))
        d2 = self.decoder2_conv(self.decoder2(torch.cat((self.upscore2(d3), hx_2), 1)))
        d1 = self.decoder1_conv(self.decoder1(torch.cat((self.upscore2(d2), hx_1), 1)))
        
        residual = self.conv_d0(self.upscore2(d1))
        return x + residual

class PSCANet(nn.Module):
    def __init__(self, cfg):
        super(PSCANet,self).__init__()
        self.cfg = cfg
        print(self.cfg)
        if self.cfg.model == 'res2net':
            self.model = res2net50_v1b_26w_4s(pretrained=True)
            self.channels = [256, 512, 1024, 2048]
        else:
            self.model = EfficientNet.from_pretrained(self.cfg.model, advprop=True)
            self.block_idx, self.channels = get_model_shape()
        channels = [32, 32, 32, 32]
        
        # [16, 32, 64, 128]
        self.upscore4 = nn.Upsample(scale_factor=32,mode='bilinear',align_corners=True)
        self.upscore3 = nn.Upsample(scale_factor=16,mode='bilinear',align_corners=True)
        self.upscore2 = nn.Upsample(scale_factor=8,mode='bilinear',align_corners=True)
        self.upscore1 = nn.Upsample(scale_factor=4,mode='bilinear',align_corners=True)  
        
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.outconv4 = nn.Conv2d(channels[3],1,3,padding=1)
        self.outconv3 = nn.Conv2d(channels[2],1,3,padding=1)
        self.outconv2 = nn.Conv2d(channels[1],1,3,padding=1)
        self.outconv1 = nn.Conv2d(channels[0],1,3,padding=1)
        #self.outconv0 = nn.Conv2d(self.channels[0],1,3,padding=1)
        
        self.decoder4 = CSHAM(channels[3], groups=4)        
        self.decoder3 = CSHAM(channels[3] + channels[2], groups=4) 
        self.decoder2 = CSHAM(channels[2] + channels[1], groups=4)
        self.decoder1 = CSHAM(channels[1] + channels[0], groups=4)        
         
        self.rfe4 = RFE_Block(self.channels[3], channels[3])
        self.rfe3 = RFE_Block(self.channels[2], channels[2])
        self.rfe2 = RFE_Block(self.channels[1], channels[1])
        self.rfe1 = RFE_Block(self.channels[0], channels[0])
        
        self.decoder4_conv = BasicConv2d(channels[3], channels[3], 1)
        self.decoder3_conv = BasicConv2d(channels[3] + channels[2], channels[2], 1)
        self.decoder2_conv = BasicConv2d(channels[2] + channels[1], channels[1], 1)
        self.decoder1_conv = BasicConv2d(channels[1] + channels[0], channels[0], 1)
        self.refine = RefUnet(in_ch=1, K=self.cfg.refine_channels)
    
    def forward(self, inputs):
        if self.cfg.model == 'res2net':
            x = self.model.conv1(inputs)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            # [3, 24, 80, 80], [3, 40, 40, 40], [3, 112, 20, 20], [3, 320, 10, 10] 
            x1 = self.model.layer1(x)      # bs, 256, 88, 88
            x2 = self.model.layer2(x1)     # bs, 512, 44, 44
            x3 = self.model.layer3(x2)     # bs, 1024, 22, 22
            x4 = self.model.layer4(x3)     # bs, 2048, 11, 11
        
        elif self.cfg.model == 'pvt':
            self.model = pvt_v2_b2()
            path = '/cluster/home/zqchen/immune_therapy/algorithms/TLS_VIT/pvt_v2_b2.pth'
            save_model = torch.load(path)
            model_dict = self.model.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)
            self.channels = [64, 128, 320, 512]
            
        else:
            B, C, H, W = inputs.size()
            x = self.model.initial_conv(inputs)
            features = self.model.get_blocks(x, H, W)
            x1 = features[0]
            x2 = features[1]
            x3 = features[2]
            x4 = features[3]
        
        x_4 = self.rfe4(x4)
        x_3 = self.rfe3(x3)
        x_2 = self.rfe2(x2)
        x_1 = self.rfe1(x1)
        d4 = self.decoder4_conv(self.decoder4(x_4))
        out4 = self.upscore4(self.outconv4(d4))  # get output
        
        d3 = self.decoder3_conv(self.decoder3(torch.cat((self.upsample(d4), x_3), dim=1)))
        out3 = self.upscore3(self.outconv3(d3)) # get output

        d2 = self.decoder2_conv(self.decoder2(torch.cat((self.upsample(d3), x_2), dim=1)))
        out2 = self.upscore2(self.outconv2(d2)) # get output

        d1 = self.decoder1_conv(self.decoder1(torch.cat((self.upsample(d2), x_1), dim=1)))
        out1 = self.upscore1(self.outconv1(d1)) # get output
        out = self.refine(out1) # refine
        #return x_4, d4, out, out1, out2, out3, out4
        return out, out1, out2, out3, out4
