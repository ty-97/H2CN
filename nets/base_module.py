# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

from __future__ import print_function, division, absolute_import
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch



__all__ = ['FBResNet', 'fbresnet50', 'fbresnet101','r21d50','fbresnet50_mgtdin','fbresnet50_tdin_2dis','fbresnet50_tdin_lt','fbresnet50_tdin_ls']

model_urls = {
        'fbresnet50': 'http://data.lip6.fr/cadene/pretrainedmodels/resnet50-19c8e357.pth',
        'fbresnet101': 'http://data.lip6.fr/cadene/pretrainedmodels/resnet101-5d3b4d8f.pth'
}


class chgc(nn.Module):
    """ Motion exciation module
    
    :param reduction=16
    :param n_segment=8/16
    """
    def __init__(self, channel, reduction=16, n_segment=8):
        super(chgc, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.n_segment = n_segment
        self.conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel//self.reduction,
            kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel//self.reduction)

        self.conv2 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel//self.reduction,
            kernel_size=3,
            padding=1,
            groups=channel//self.reduction,
            #dilation=2,
            bias=False)
        self.convt = nn.Conv3d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel//self.reduction,
            kernel_size=(3,1,1),
            padding=0,
            stride=(3,1,1),
            groups=channel//self.reduction,
            bias=False
        )

        # self.convt.weight.requires_grad = True
        # self.convt.weight.data.zero_()
        # self.convt.weight.data[:, 0, 2, :, :] = -1 # shift left
        # self.convt.weight.data[:, 0, 0, :, :] = 0 # shift right
        # self.convt.weight.data[:, 0, 1, :, :] = 1 # fixed
        nn.init.xavier_normal_(self.convt.weight)
        

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)

        self.conv3 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)

        self.identity = nn.Identity()

    def forward(self, x):
        nt, c, h, w = x.size()
        #print('x:{}'.format(x.shape))
        bottleneck = self.conv1(x) # nt, c//r, h, w
        bottleneck = self.bn1(bottleneck) # nt, c//r, h, w

        # t feature
        reshape_bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:])  # n, t, c//r, h, w
        reshape_bottleneck = reshape_bottleneck.unsqueeze(2)
        #t_fea, __ = reshape_bottleneck.split([self.n_segment-1, 1], dim=1) # n, t-1, c//r, h, w

        # apply transformation conv to t+1 feature
        conv_bottleneck = self.conv2(bottleneck)  # nt, c//r, h, w
        # reshape fea: n, t, c//r, h, w
        reshape_conv_bottleneck = conv_bottleneck.view((-1, self.n_segment) + conv_bottleneck.size()[1:])
        #__, tPlusone_fea = reshape_conv_bottleneck.split([1, self.n_segment-1], dim=1)  # n, t-1, c//r, h, w
        reshape_conv_bottleneck = reshape_conv_bottleneck.unsqueeze(2)#.expand((-1, self.n_segment, 2) + conv_bottleneck.size()[1:])
       
        zeros = torch.zeros(nt//self.n_segment, 1, 1, *conv_bottleneck.size()[1:]).cuda()
        target = torch.cat((
                            torch.cat((zeros, reshape_conv_bottleneck[:,:-1,:,:,:]),dim=1),
                            reshape_bottleneck, 
                            torch.cat((reshape_conv_bottleneck[:,1:,:,:,:], zeros),dim=1)
                            ),
                            dim=2)
        target = target.view((-1, 3*self.n_segment) + conv_bottleneck.size()[1:])

        target = self.convt(target.transpose(1,2))
        target = target.transpose(1,2).contiguous().view(-1, *conv_bottleneck.shape[1:])
        y = self.avg_pool(target)  # nt, c//r, 1, 1
 
        y = self.conv3(y)  # nt, c, 1, 1
        y = self.bn3(y)  # nt, c, 1, 1
        y = self.sigmoid(y)  # nt, c, 1, 1
        
        output = x * y.expand_as(x)
        return output



class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, n_segment, inplanes, planes, stride=1, downsample=None, fold_div=8, place='blockres'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.mtdin = chgc(planes, n_segment=n_segment) 
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.n_segment = n_segment
        self.fold_div = fold_div
        self.place = place
        # self.inplace = inplace

        if place in ['block', 'blockres']:
            print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        residual = x

        if self.place == 'blockres':
            out = self.shift(x, self.n_segment, fold_div=self.fold_div)
            out = self.conv1(out)
        else:
            out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.mtdin(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
        #
        if self.place == 'block':
            out = self.shift(out, self.n_segment, fold_div=self.fold_div)
 
        return out
    #
    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
      
   
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError  
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)






class ResNet_h2cn(nn.Module):

    def __init__(self, num_segments, block, layers, num_classes=1000):
        self.inplanes = 64

        self.input_space = None
        self.input_size = (224, 224, 3)
        self.mean = None
        self.std = None
        self.num_segments = num_segments
        super(ResNet_h2cn, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(self.num_segments,block, 64, layers[0])
        self.layer2 = self._make_layer(self.num_segments,block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(self.num_segments,block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(self.num_segments,block, 512, layers[3], stride=2)
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, num_segments ,block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(num_segments, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(num_segments, self.inplanes, planes))

        return nn.Sequential(*layers)


    def features(self, input):
        x = self.conv1(input)
        self.conv1_input = x.clone()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x


    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x



def h2cn50(num_segments=8,pretrained=False,num_classes=1000):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet_h2cn(num_segments,Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
         model.load_state_dict(model_zoo.load_url(model_urls['fbresnet50']),strict=False)
    return model
