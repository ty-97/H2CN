# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.init import normal_, constant_
import torch.nn.functional as F
from nets.base_module import h2cn50



class fhgc(nn.Module):
    def __init__(self, channel, reduction=16, n_segment=8, ME_init=True, chan_expend=1):
        super(fhgc, self).__init__()
        print('========>using ltdin2nd, agg is conv2d')
        self.channel = channel
        self.n_segment = n_segment

        #self.spaAgg =  self.pool = nn.AvgPool2d(kernel_size=(3,3), stride=(1,1), padding=1)    
        self.spaAgg = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel,
            kernel_size=3,
            padding=2,
            groups=channel,
            dilation=2,
            bias=False)
        nn.init.xavier_normal_(self.spaAgg.weight)
        
        self.convt = nn.Conv3d(
            in_channels=self.channel,
            out_channels=self.channel*chan_expend,
            kernel_size=(3,1,1),
            padding=0,
            stride=(3,1,1),
            groups=channel,
            bias=False
        )
        if ME_init == True:
            self.convt.weight.requires_grad = True
            self.convt.weight.data.zero_()
            self.convt.weight.data[:, 0, 2, :, :] = -1 # shift left
            self.convt.weight.data[:, 0, 0, :, :] = 0 # shift right
            self.convt.weight.data[:, 0, 1, :, :] = 1 # fixed
        else:
            nn.init.kaiming_normal_(self.convt.weight, mode='fan_out', nonlinearity='relu') 


    def forward(self, x):
        nt, c, h, w = x.size()

        reshape_x = x.view((-1, self.n_segment) + x.size()[1:])  # n, t, c//r, h, w
        reshape_x = reshape_x.unsqueeze(2)
        #t_fea, __ = reshape_bottleneck.split([self.n_segment-1, 1], dim=1) # n, t-1, c//r, h, w

        # apply transformation conv to t+1 feature
        agg_x = self.spaAgg(x)  # nt, c//r, h, w
        # reshape fea: n, t, c//r, h, w
        reshape_agg_x = agg_x.view((-1, self.n_segment) + agg_x.size()[1:])
        reshape_agg_x = reshape_agg_x.unsqueeze(2)
        zeros = torch.zeros(nt//self.n_segment, 1, 1, *agg_x.size()[1:]).cuda()
        target = torch.cat((reshape_agg_x[:,:-2,:,:,:],reshape_x[:,1:-1,:,:,:],reshape_agg_x[:,2:,:,:,:]),dim=2)
        target = target.view((-1, 3*(self.n_segment-2)) + agg_x.size()[1:])

        target = self.convt(target.transpose(1,2))
        
        target = target.transpose(1,2).contiguous()
        target = target.view(-1, *target.shape[2:])
        return target


class build_h2cn(nn.Module):

    def __init__(self,resnet_model,resnet_model1,apha,belta):
        super(build_h2cn, self).__init__()
        print('initialize h2cn')
        
        
        self.avg_diff = nn.AvgPool2d(kernel_size=2,stride=2)
        self.fhgc1 = nn.Sequential(fhgc(3,n_segment=5),nn.BatchNorm2d(3),nn.ReLU(inplace=True))
        self.fhgc2 = nn.Sequential(fhgc(3,n_segment=3,ME_init=False,chan_expend=3),nn.BatchNorm2d(9),nn.ReLU(inplace=True))

        # implement conv1_5 and inflate weight 
        self.conv1_temp = list(resnet_model1.children())[0]
        params = [x.clone() for x in self.conv1_temp.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3 * 3,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        self.conv_fhgc = nn.Sequential(nn.Conv2d(9,64,kernel_size=7,stride=2,padding=3,bias=False),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.conv_fhgc[0].weight.data = new_kernels

        self.maxpool_fhgc = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.conv1 = list(resnet_model.children())[0]
        self.bn1 = list(resnet_model.children())[1]
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1_bak = nn.Sequential(*list(resnet_model.children())[4])
        self.layer2_bak = nn.Sequential(*list(resnet_model.children())[5])
        self.layer3_bak = nn.Sequential(*list(resnet_model.children())[6])
        self.layer4_bak = nn.Sequential(*list(resnet_model.children())[7])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        
        self.fc = list(resnet_model.children())[8]
        self.apha = apha
        self.belta = belta

    def forward(self, x):
        f = x[:,6:9,:,:]
        f_mo = self.fhgc2(self.fhgc1(self.avg_diff(x.view(-1,x.shape[1]//5,*x.shape[2:]))))
        f_mo = self.conv_fhgc(f_mo)

        f_mo = self.maxpool_fhgc(1.0/1.0*f_mo)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        f_mo = F.interpolate(f_mo, x.size()[2:])
        x = self.apha*x + self.belta*f_mo
 
        x = self.layer1_bak(x)     
        x = self.layer2_bak(x)
        x = self.layer3_bak(x)
        x = self.layer4_bak(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

def h2cn(num_segments=8,pretrained=True, **kwargs):

    resnet_model = h2cn50(num_segments, pretrained)
    resnet_model1 = h2cn50(num_segments, pretrained)

    model = build_h2cn(resnet_model,resnet_model1,apha=0.5,belta=0.5)
   
    return model

