import torch
import torch.nn.functional as F
from torch import nn

from .resnet import resnet50, resnet101

from .pspnet import PSPNet

import pdb

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

class ListEncoder(nn.Module):
    def __init__(self, orig_resnet):
        super(ListEncoder, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        # self.conv1 = orig_resnet.conv1
        # self.bn1 = orig_resnet.bn1
        # self.relu1 = orig_resnet.relu1
        # self.conv2 = orig_resnet.conv2
        # self.bn2 = orig_resnet.bn2
        # self.relu2 = orig_resnet.relu2
        # self.conv3 = orig_resnet.conv3
        # self.bn3 = orig_resnet.bn3
        # self.relu3 = orig_resnet.relu3
        # self.maxpool = orig_resnet.maxpool
        self.deep_base = orig_resnet.deep_base
        if not self.deep_base:
            self.relu = orig_resnet.relu
            self.bn1 = orig_resnet.bn1
            self.conv1 = orig_resnet.conv1
        else:
            self.layer0 = orig_resnet.layer0
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4
        self.fc = orig_resnet.fc

    def extract_feat(self, x, return_feature_maps=True):
        conv_out = []

        # x = self.relu1(self.bn1(self.conv1(x)))
        # x = self.relu2(self.bn2(self.conv2(x)))
        # x = self.relu3(self.bn3(self.conv3(x)))
        # x = self.maxpool(x)
        if not self.deep_base:
            x = self.relu(self.bn1(self.conv1(x)))
        else:
            x = self.layer0(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


    def forward(self, x, return_feature_maps = True):
        if return_feature_maps:
            x = self.extract_features(x,return_feature_maps = True)
        else:
            x = self.extract_features(x,return_feature_maps = False)
        return x
        
