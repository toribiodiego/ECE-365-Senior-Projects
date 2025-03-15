################################################
# hubconf.py
################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

########################################
# MODEL DEFINITION (ResNetFace) UTILS
########################################

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class IRBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)
    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.prelu(out)
        return out

class ResNetFace(nn.Module):
    def __init__(self, block, layers, use_se=True, grayscale=True, embedding_size=512):
        super(ResNetFace, self).__init__()
        self.inplanes = 64
        in_ch = 1 if grayscale else 3
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0], use_se)
        self.layer2 = self._make_layer(block, 128, layers[1], use_se, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], use_se, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], use_se, stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.fc5 = nn.Linear(512 * 8 * 8, embedding_size)
        self.bn5 = nn.BatchNorm1d(embedding_size)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, use_se, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.bn5(x)
        return {'fea': x}

def _resnet_face18(use_se=True, grayscale=True, embedding_size=512):
    """Helper to create a ResNetFace-18 model."""
    return ResNetFace(IRBlock, [2, 2, 2, 2],
                      use_se=use_se,
                      grayscale=grayscale,
                      embedding_size=embedding_size)

########################################
# HUB MODEL-LOADING FUNCTIONS
########################################

def resnet18_face(pretrained=False, use_se=True, grayscale=True, embedding_size=512, weights_path="resnet18_110.pth"):
    """
    Example entrypoint for torch.hub.load().
    
    Args:
        pretrained (bool): If True, loads the local weights from `weights_path`.
        use_se (bool): Whether to use squeeze-and-excitation blocks.
        grayscale (bool): Whether the input is 1-channel.
        embedding_size (int): Dimension of final embedding.
        weights_path (str): Path to local .pth file with state_dict for the model.
    """
    model = _resnet_face18(use_se=use_se, grayscale=grayscale, embedding_size=embedding_size)
    if pretrained:
        state_dict = torch.load(weights_path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            # If the model is configured for 3-channel input but the checkpoint has 1-channel conv1 weights, replicate them.
            if new_key == "conv1.weight":
                if (not grayscale) and (v.size(1) == 1):
                    v = v.repeat(1, 3, 1, 1) / 3.0
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
    return model
