import torch.nn as nn
import torch.nn.functional as F
import math
import torch


# ResNet的基本模块-Bottleneck类
class Bottleneck(nn.Module):
    expansion = 4  # 通道倍增数

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_planes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, stride, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, self.expansion * planes, 1, bias=False),
            nn.BatchNorm2d(self.expansion * planes),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bottleneck(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# FPN的类，初始化需要一个list，代表ResNet的每一个阶段的Bottleneck的数量
class FPN32x32(nn.Module):
    def __init__(self, layers, classes, overall_dropout):
        super(FPN32x32, self).__init__()
        self.inplanes = 64
        # 处理输入的C1模块（C1代表了RestNet的前几个卷积与池化层）
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        # 搭建自下而上的C2，C3，C4，C5
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], 2)
        self.layer3 = self._make_layer(256, layers[2], 2)
        self.layer4 = self._make_layer(512, layers[3], 2)
        # 对C5减少通道数，得到P5
        self.toplayer = nn.Conv2d(2048, 256, 1, 1, 0)
        # 3x3卷积融合特征
        self.smooth1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)
        # 横向连接，保证通道数相同
        self.latlayer1 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(256, 256, 1, 1, 0)
        # flatten
        self.flat2 = self.flat3 = self.flat4 = self.flat5 = torch.flatten
        self.concate = torch.concat
        self.dropout = nn.Dropout(overall_dropout)
        self.fc = nn.Linear(256 * (8 * 8 + 4 * 4 + 2 * 2 + 1 * 1), classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != Bottleneck.expansion * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, Bottleneck.expansion * planes, 1, stride, bias=False),
                nn.BatchNorm2d(Bottleneck.expansion * planes)
            )
        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    # 自上而下的采样模块
    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        # 自下而上
        c1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # 自上而下
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # 卷积的融合，平滑处理
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        out = self.concate(
            [self.flat2(p2, start_dim=1),
             self.flat3(p3, start_dim=1),
             self.flat4(p4, start_dim=1),
             self.flat5(p5, start_dim=1)],
            dim=1
        )
        out = self.dropout(out)
        out = self.fc(out)

        return out


# ResNet50FPN32x32
def GDBLS(num_classes=None,
          input_shape=None,
          overall_dropout=None,
          filters=None,
          divns=None,
          dropout_rate=None,
          batchsize=None):
    if input_shape is None:
        input_shape = [3, 32, 32]
    return FPN32x32([3, 4, 6, 3], num_classes, overall_dropout)


if __name__ == '__main__':


    x = torch.randn(2, 3, 32, 32)
    net = GDBLS(num_classes=10, overall_dropout=0.2)
    res = net(x)
    print('res.shape: ', res.shape)
