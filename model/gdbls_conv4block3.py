import torch
from torch import nn
from torch import Tensor
from typing import List
from torchvision.utils import _log_api_usage_once
from model.PLVPooling import PLVPooling
from model.customConvs import conv1x1, conv3x3, conv5x5
from model.CBAM import CBAM


class FeatureBlock(nn.Module):
    def __init__(
            self,
            inplanes: int,
            planes: int,
            divn: int = 4,
            dropout_rate: float = 0.1,
            batchs: int = 128,
            islast: bool=False
    ) -> None:
        super().__init__()
        self.planes = planes
        self.batchs = batchs

        self.conv1 = conv3x3(inplanes, planes // 2)
        self.bn1 = nn.BatchNorm2d(planes // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = conv3x3(planes // 2, planes // 2)
        self.bn2 = nn.BatchNorm2d(planes // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv3 = conv3x3(planes // 2, planes // 2)
        self.bn3 = nn.BatchNorm2d(planes // 2)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(dropout_rate)

        if islast:
            self.conv4 = conv5x5(planes // 2, planes, bias=True)
        else:
            self.conv4 = conv3x3(planes // 2, planes, bias=True)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool = PLVPooling()

        self.reshape1 = torch.reshape
        self.fc1 = nn.Linear(planes, planes // divn)
        self.fc2 = nn.Linear(planes // divn, planes)

        self.reshape2 = torch.reshape
        self.multiply = torch.multiply

        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.dropout5 = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.dropout1(out)  # batchsize,planes // 2,32,32

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.bn2(out)
        out = self.dropout2(out)  # batchsize,planes // 2,32,32

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.bn3(out)
        out = self.dropout3(out)  # batchsize,planes // 2,32,32

        out = self.conv4(out)
        out = self.relu4(out)  # batchsize,planes,32,32
        # out = self.bn3(out)

        # se block
        identity = out
        seout = self.pool(out, self.conv4.bias)
        seout = self.reshape1(seout, (self.batchs, self.planes))  # batchsize,planes,1,1 -> batchsize,planes
        seout = self.fc1(seout)
        seout = self.fc2(seout)
        out = self.multiply(self.reshape2(seout, (self.batchs, self.planes, 1, 1,)), identity)

        # out = self.cbam(out)
        if self.downsample is not None:
            out = self.downsample(out)
        out = self.dropout5(out)
        return out


# gdbls base
# GrandDescentBoardLearningSystem
class GDBLS(nn.Module):
    def __init__(
            self,
            num_classes: int = 10,
            batchsize: int = 128,
            input_shape: List[int] = None,
            overall_dropout=0.5,
            filters: List[int] = None,
            divns: List[int] = None,
            dropout_rate: List[float] = None,
    ) -> None:
        super(GDBLS, self).__init__()
        _log_api_usage_once(self)

        assert input_shape is not None
        self.inplanes = input_shape[0]  # in default channels first
        self.num_classes = num_classes

        self.fb1 = FeatureBlock(
            inplanes=self.inplanes,
            planes=filters[0],
            divn=divns[0],
            dropout_rate=dropout_rate[0],
            batchs=batchsize,
        )
        self.fb2 = FeatureBlock(
            inplanes=filters[0],
            planes=filters[1],
            divn=divns[1],
            dropout_rate=dropout_rate[0],
            batchs=batchsize,
        )
        self.fb3 = FeatureBlock(
            inplanes=filters[1],
            planes=filters[2],
            divn=divns[2],
            dropout_rate=dropout_rate[0],
            batchs=batchsize,
            islast=True
        )

        self.flatten1 = torch.flatten

        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2d2 = conv1x1(in_planes=filters[1], out_planes=filters[0])
        self.flatten2 = torch.flatten

        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.conv2d3 = conv1x1(in_planes=filters[2], out_planes=filters[0])
        self.flatten3 = torch.flatten

        self.dropout = nn.Dropout(overall_dropout)
        self.fc = nn.Linear((filters[0] * 16 * 16), self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_impl(self, x: Tensor) -> Tensor:
        p1 = self.fb1(x)  # batchsize,128,16,16
        p2 = self.fb2(p1)  # batchsize,192,8,8
        p3 = self.fb3(p2)  # batchsize,256,4,4

        p1 = self.flatten1(p1, start_dim=1)  # [batchsize,32768]=batchsize*(128*16*16)

        p2 = self.upsample2(p2)  # 8,8 -> 16,16, upsample factor = 2
        p2 = self.conv2d2(p2)  # 1x1 conv
        p2 = self.flatten2(p2, start_dim=1)  # [batchsize,32768]=batchsize*(128*16*16)

        p3 = self.upsample3(p3)  # 4,4 -> 16,16, upsample factor = 4
        p3 = self.conv2d3(p3)  # 1x1 conv
        p3 = self.flatten3(p3, start_dim=1)  # [batchsize,32768]=batchsize*(128*16*16)

        out = self.dropout(p1 + p2 + p3)

        out = self.fc(out)
        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
