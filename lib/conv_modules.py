import torch.nn as nn
import torch
import torchvision


def get_model_shape(model_name):
    if model_name == 'efficientnet-b0':
        block_idx = [2, 4, 10, 15]
        channels = [24, 40, 112, 320]
    elif model_name == 'efficientnet-b1':
        block_idx = [4, 7, 15, 22]
        channels = [24, 40, 112, 320]
    elif model_name == 'efficientnet-b2':
        block_idx = [4, 7, 15, 22]
        channels = [24, 48, 120, 352]
    elif model_name == 'efficientnet-b3':
        block_idx = [4, 7, 17, 25]
        channels = [32, 48, 136, 384]
    elif model_name == 'efficientnet-b4':
        block_idx = [5, 9, 21, 31]
        channels = [32, 56, 160, 448]
    elif model_name == 'efficientnet-b5':
        block_idx = [7, 12, 26, 38]
        channels = [40, 64, 176, 512]
    elif model_name == 'efficientnet-b6':
        block_idx = [8, 14, 30, 44]
        channels = [40, 72, 200, 576]
    elif model_name == 'efficientnet-b7':
        block_idx = [10, 17, 37, 54]
        channels = [48, 80, 224, 640]

    return block_idx, channels



class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.selu = nn.SELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.selu(x)

        return x

class LossNet(torch.nn.Module):
    def __init__(self, resize=True):
        super(LossNet, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target

        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.mse_loss(x, y)
        return loss
