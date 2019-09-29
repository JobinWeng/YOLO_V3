import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy as np

class UpSmapelLayer(nn.Module):
    def __init__(self):
        super(UpSmapelLayer, self).__init__()

    def forward(self, x):
        return F.interpolate(x,scale_factor=2,mode='nearest')

class ConvolutionalLayer(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, strtide, padding, bias=False):
        super(ConvolutionalLayer, self).__init__()
        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strtide, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.sub_module(x)

class ResiduaLayer(nn.Module):
    def __init__(self,in_channels):
        super(ResiduaLayer, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels//2, 1, 1, 0),
            ConvolutionalLayer(in_channels//2, in_channels, 3, 1, 1),
        )

    def forward(self, x):
        return self.sub_module(x) + x

class DownSamplingLayer(nn.Module):
    def __init__(self, in_channels, out_chanels):
        super(DownSamplingLayer, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_channels, out_chanels, 3, 2, 1)
        )

    def forward(self, x):
        return self.sub_module(x)

class ConvolutionalSet(nn.Module):
    def __init__(self, in_chanels, out_chanels):
        super(ConvolutionalSet, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_chanels, out_chanels, 1, 1, 0),
            ConvolutionalLayer(out_chanels, in_chanels, 3, 1, 1),

            ConvolutionalLayer(in_chanels, out_chanels, 1, 1, 0),
            ConvolutionalLayer(out_chanels, in_chanels, 3, 1, 1),

            ConvolutionalLayer(in_chanels, out_chanels, 1, 1, 0),
        )
    def forward(self, x):
        return self.sub_module(x)

class ConvolutionalSets(nn.Module):
    def __init__(self, in_chanels, out_chanels):
        super(ConvolutionalSets, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalLayer(in_chanels, out_chanels, 3, 1, 1),
            ConvolutionalLayer(out_chanels, in_chanels, 1, 1, 0),

            ConvolutionalLayer(in_chanels, out_chanels, 3, 1, 1),
            ConvolutionalLayer(out_chanels, in_chanels, 1, 1, 0),
        )
    def forward(self, x):
        return self.sub_module(x)

class MainNet(nn.Module):
    def __init__(self, cls_num):
        super(MainNet, self).__init__()

        self.trunk_52 = nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            ConvolutionalLayer(32, 64, 3, 2, 1),

            ResiduaLayer(64),
            DownSamplingLayer(64, 128),

            ResiduaLayer(128),
            ResiduaLayer(128),
            DownSamplingLayer(128, 256),

            ResiduaLayer(256),
            ResiduaLayer(256),
            ResiduaLayer(256),
            ResiduaLayer(256),
            ResiduaLayer(256),
            ResiduaLayer(256),
            ResiduaLayer(256),
            ResiduaLayer(256),
        )

        self.trunk_26 = nn.Sequential(
            DownSamplingLayer(256, 512),

            ResiduaLayer(512),
            ResiduaLayer(512),
            ResiduaLayer(512),
            ResiduaLayer(512),
            ResiduaLayer(512),
            ResiduaLayer(512),
            ResiduaLayer(512),
            ResiduaLayer(512),
        )

        self.trunk_13 = nn.Sequential(
            DownSamplingLayer(512, 1024),

            ResiduaLayer(1024),
            ResiduaLayer(1024),
            ResiduaLayer(1024),
            ResiduaLayer(1024),
        )

        self.convset_13 = nn.Sequential(
            ConvolutionalSet(1024,512)
        )

        self.detection_13 = nn.Sequential(
            ConvolutionalLayer(512,1024,3,1,1),
            nn.Conv2d(1024,3*(5+cls_num),1,1,0)
        )

        self.up_26 = nn.Sequential(
            ConvolutionalLayer(512,256,1,1,0),
            UpSmapelLayer()
        )

        self.convset_26 = nn.Sequential(
            ConvolutionalLayer(768,256,1,1,0),  #concat 路由
            ConvolutionalSets(256,512)
        )

        self.detection_26 = nn.Sequential(
            ConvolutionalLayer(256,512,3,1,1),
            nn.Conv2d(512,3*(5+cls_num),1,1,0)
        )

        self.up_52 = nn.Sequential(
            ConvolutionalLayer(256,128,1,1,0),
            UpSmapelLayer()
        )

        self.convset_52 = nn.Sequential(
            ConvolutionalLayer(384, 128, 1, 1, 0),  # concat 路由
            ConvolutionalSets(128, 256)
        )

        self.detection_52 = nn.Sequential(
            ConvolutionalLayer(128,256,3,1,1),
            nn.Conv2d(256, 3*(5+cls_num),1,1,0)
        )

    def forward(self, x):
        h_52 = self.trunk_52(x)
        h_26 = self.trunk_26(h_52)
        h_13 = self.trunk_13(h_26)

        convset_out_13 = self.convset_13(h_13)
        detecttion_out_13 = self.detection_13(convset_out_13)

        up_out_26 = self.up_26(convset_out_13)
        route_out_26 = torch.cat((up_out_26,h_26),dim=1)
        convset_out_26 = self.convset_26(route_out_26)
        detecttion_out_26 = self.detection_26(convset_out_26)

        print("convset_out_26",convset_out_26.shape)
        up_out_52 = self.up_52(convset_out_26)
        route_out_52 = torch.cat((up_out_52,h_52),dim=1)
        convset_out_52 = self.convset_52(route_out_52)
        detecttion_out_52 = self.detection_52(convset_out_52)

        return detecttion_out_13, detecttion_out_26, detecttion_out_52

if __name__ == '__main__':
    input = torch.rand((2,3,416,416))

    net = MainNet(3)
    out_13, out_26, out_52 = net(input)
    print(out_13.shape,out_26.shape,out_52.shape)

    # a = torch.tensor([1,2,5,4])
    # mask = a>2
    # print("a",a)
    # print("mask:",mask)
    # print("a[mask]",a[mask])

    # a = np.array([1,2,5,4])
    # mask = a>2
    # print("a",a)
    # print("mask:",mask.dtype)
    # print("a[mask]",a[mask])

    for name, value in net.named_parameters():
        if name[:8] == "trunk_13":
            print(value)
