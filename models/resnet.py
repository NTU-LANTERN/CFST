"""
This code was based on the file resnet.py (https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
from the pytorch/vision library (https://github.com/pytorch/vision).

The original license is included below:

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from collections import OrderedDict
from types import MethodType

import torch.nn as nn
import torch
from avalanche.models import BaseModel, MultiHeadClassifier, IncrementalClassifier, MultiTaskModule, DynamicModule


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.initial_pool = False
        inplanes = self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def get_layer_output(self, x, layer_to_return):
        hidden_features = None

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        if layer_to_return == 0:
            hidden_features = x

        x = self.layer1(x)
        if layer_to_return == 1:
            hidden_features = x

        x = self.layer2(x)
        if layer_to_return == 2:
            hidden_features = x

        x = self.layer3(x)
        if layer_to_return == 3:
            hidden_features = x

        x = self.layer4(x)
        if layer_to_return in [4, -1]:
            hidden_features = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x, hidden_features

    @property
    def output_size(self):
        return 512


def resnet18(pretrained=False, pretrained_model_path=None, fix=False, **kwargs) -> ResNet:
    """
        Constructs a ResNet-18 feature extractor.
    """

    feature_extractor = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        print('Load pretrained resnet18 model from {}.'.format(pretrained_model_path))
        ckpt_dict = torch.load(pretrained_model_path)
        if 'state_dict' in ckpt_dict:
            feature_extractor.load_state_dict(ckpt_dict['state_dict'])
        else:
            d = OrderedDict()
            for key, item in ckpt_dict.items():
                if key.startswith('resnet'):
                    d['.'.join(key.split('.')[1:])] = item
            feature_extractor.load_state_dict(d)

    # Freeze the parameters of the feature extractor
    if fix:
        for param in feature_extractor.parameters():
            param.requires_grad = False

    return feature_extractor


"""
END: GENERAL RESNET CODE
"""


class NResNet18(nn.Module):
    """
        ResNet18 with normal linear classifier.

    """
    def __init__(self, initial_out_features: int = 2,
                 pretrained=False, pretrained_model_path=None, fix=False):
        super().__init__()
        self.resnet = resnet18()
        self.classifier = nn.Linear(self.resnet.output_size, initial_out_features)

        if pretrained:
            print('Load pretrained resnet18 model from {}.'.format(pretrained_model_path))
            ckpt_dict = torch.load(pretrained_model_path)   # , map_location='cuda:0'
            if 'state_dict' in ckpt_dict:
                self.resnet.load_state_dict(ckpt_dict['state_dict'])
            else:   # load resnet and classifier
                self.load_state_dict(ckpt_dict)

        # Freeze the parameters of the feature extractor
        if fix:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = self.resnet(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


class ResNet18(DynamicModule):
    """
        ResNet18 with classifier.

    """
    def __init__(self, initial_out_features: int = 2,
                 pretrained=False, pretrained_model_path=None, fix=False, masking=True):
        super().__init__()
        self.resnet = resnet18()
        self.classifier = IncrementalClassifier(self.resnet.output_size, initial_out_features=initial_out_features,
                                                masking=masking)

        if pretrained:
            print('Load pretrained resnet18 model from {}.'.format(pretrained_model_path))
            ckpt_dict = torch.load(pretrained_model_path)   # , map_location='cuda:0'
            if 'state_dict' in ckpt_dict:
                self.resnet.load_state_dict(ckpt_dict['state_dict'])
            else:   # load resnet and classifier
                self.load_state_dict(ckpt_dict)

        # Freeze the parameters of the feature extractor
        if fix:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = self.resnet(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


class MTResNet18(MultiTaskModule, DynamicModule):
    """
        MultiTask ResNet18.
        It employs multi-head output layer.
    """

    def __init__(self, initial_out_features: int = 2, pretrained=False, pretrained_model_path=None,
                 fix=False, load_classifier=False, masking=True):
        super().__init__()
        self.resnet = resnet18(pretrained, pretrained_model_path, fix)
        self.classifier = MultiHeadClassifier(self.resnet.output_size, initial_out_features=initial_out_features,
                                              masking=masking)

        # if pretrained:
        #     print('Load pretrained resnet18 model from {}.'.format(pretrained_model_path))
        #     ckpt_dict = torch.load(pretrained_model_path)   # , map_location='cuda:0'
        #     if 'state_dict' in ckpt_dict:
        #         self.resnet.load_state_dict(ckpt_dict['state_dict'])
        #     else:   # load resnet and classifier
        #         self.load_state_dict(ckpt_dict)
        #
        # # Freeze the parameters of the feature extractor
        # if fix:
        #     for param in self.resnet.parameters():
        #         param.requires_grad = False

    def forward_single_task(self, x: torch.Tensor, task_label: int) -> torch.Tensor:
        out = self.resnet(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out, task_label)


def get_resnet(
        multi_head: bool = False,
        initial_out_features: int = 2, pretrained=False, pretrained_model_path=None, fix=False,
        masking=True,
        normal_classifier=False,
):
    if multi_head:
        model = MTResNet18(initial_out_features, pretrained, pretrained_model_path, fix, masking=masking)
    elif normal_classifier:
        model = NResNet18(initial_out_features, pretrained, pretrained_model_path, fix)
    else:
        model = ResNet18(initial_out_features, pretrained, pretrained_model_path, fix, masking=masking)

    return model


__all__ = ['ResNet18', 'MTResNet18', 'get_resnet', 'conv1x1', 'conv3x3', 'BasicBlock']


if __name__ == '__main__':
    # model = resnet18(
    #     pretrained=True,
    #     pretrained_model_path="//172.18.36.77/datasets/pretrained/pretrained_resnet.pt.tar")
    # x = torch.randn(1, 3, 84, 84)
    # features = model(x)
    # print(features.shape)   # whatever input shape, the output is [1, 512]

    model = ResNet(BasicBlock, [2, 2, 2, 2])        # MTResNet18()
    x = torch.randn(1, 3, 128, 128)
    features = model(x)
    print(features.shape)   # whatever input shape, the output is [1, 512]

    from models import get_parameter_number

    d = get_parameter_number(model)
    print(d)
    print(f'Total number of parameters: {d["Total"] / 1024 / 1024:.2f}MB, '
          f'memory size: {d["Total"] * 4 / 1024 / 1024:.2f}MB')
    print(f'Total number of trainable parameters: {d["Trainable"] / 1024 / 1024:.2f}MB, '
          f'memory size: {d["Trainable"] * 4 / 1024 / 1024:.2f}MB')

    get_resnet_model = get_resnet(multi_head=True)

    for key, params in get_resnet_model.classifier.named_parameters():
        print(f'{key}: {params.shape}')
