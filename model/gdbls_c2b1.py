import torch
from torch import nn
from torch import Tensor
from typing import List
from torchvision.utils import _log_api_usage_once
from model.PLVPooling import PLVPooling
from model.customConvs import conv1x1, conv3x3, conv5x5
from model.CBAM import CBAM
import math
from timm.models.layers import DropPath, trunc_normal_


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        # x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Encoder(nn.Module):
    def __init__(
            self,
            dim, num_heads, mlp_ratio=4., qkv_bias=False,
            drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            sr_ratio=1, linear=False
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)

        # NOTE: drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, batchsize, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = x.reshape(batchsize, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class FeatureBlock(nn.Module):
    def __init__(
            self,
            inplanes: int,
            planes: int,
            divn: int = 4,
            dropout_rate: float = 0.1,
            batchs: int = 128,
            islast: bool = False
    ) -> None:
        super().__init__()
        self.planes = planes
        self.batchs = batchs

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = conv3x3(planes, planes, bias=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool = PLVPooling()
        # self.pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.reshape1 = torch.reshape
        self.reshape2 = torch.reshape
        self.multiply = torch.multiply

        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)

        # se block
        identity = out
        seout = self.pool(out, self.conv2.bias)
        # seout = self.pool(out)
        seout = self.reshape1(seout, (self.batchs, self.planes))  # batchsize,planes,1,1 -> batchsize,planes
        out = self.multiply(self.reshape2(seout, (self.batchs, self.planes, 1, 1,)), identity)

        # out = self.cbam(out)
        if self.downsample is not None:
            out = self.downsample(out)
        out = self.dropout3(out)

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
            num_heads: List[int] = [1, 2],
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

        self.flatten1 = torch.flatten

        self.dropout = nn.Dropout(overall_dropout)
        self.fc = nn.Linear((filters[0] * 16 * 16), self.num_classes)

        self.encoder1 = Encoder(dim=filters[0], num_heads=num_heads[0], mlp_ratio=4, qkv_bias=True,
                                drop=dropout_rate[0], attn_drop=dropout_rate[0], drop_path=dropout_rate[0])

        self.encoder11 = Encoder(dim=filters[0], num_heads=num_heads[0], mlp_ratio=4, qkv_bias=True,
                                drop=dropout_rate[0], attn_drop=dropout_rate[0], drop_path=dropout_rate[0])

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

        batchsize, _, H, W = p1.shape
        p1 = p1.flatten(2).transpose(1, 2)
        p1 = self.encoder1(p1, batchsize, H, W)
        p1 = p1.flatten(2).transpose(1, 2)
        p1 = self.encoder11(p1, batchsize, H, W)

        p1 = self.flatten1(p1, start_dim=1)

        out = self.dropout(p1)
        out = self.fc(out)
        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
