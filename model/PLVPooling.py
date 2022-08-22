import torch
from torch.nn import Module


class PLVPooling(Module):
    """
    Postive Porpotion Value Pooling Layer

    - given the channel format, the layer applies change on the input tensor's dims.
        for channels_first=True, we compress the 2th, 3th dims. Else, we compress the 1th and 2th dims.
    - we use bias value provided by the last conv layer to judge if the digit in each channels should be set to 1 or 0.
        ATTENTION: the conv layer right before PLVPooling layer should not have any bn layer, otherwise the performance
        will be worse.
    """

    def __init__(self, channels_first=True, **kwargs):
        super(PLVPooling, self).__init__()
        self.dims = [2, 3] if channels_first else [1, 2]

    def forward(self, x, bias):
        # 7.27: METHOD 2, expand dims to the shape of the original input x
        batch, C, H, W = x.shape
        ppvoutput = torch.mean(
            torch.greater(
                x, torch.reshape(bias, (C, 1, 1)).expand((C, H, W))
            ).float(), dim=self.dims)
        return ppvoutput
