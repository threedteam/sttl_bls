from torch import nn


# for the conv layer without any bn_layer followed, it is not necessary to enable bias.
def conv5x5(
        in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, bias=False
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=5,
        stride=stride,
        padding=2,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


def conv3x3(
        in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, bias=False
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias=False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)
