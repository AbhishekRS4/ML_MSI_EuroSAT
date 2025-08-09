import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels: int, reduction_ratio: int = 2):
        """
        ---------
        Arguments
        ---------
        num_channels: int
            number of input channels
        reduction_ratio: int
            By how much should the num_channels should be reduced (default: 2)
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        ---------
        Arguments
        ---------
        input_tensor: torch.Tensor
            input torch tensor of shape = (batch_size, num_channels, H, W)

        ------
        Return
        ------
        output_tensor: torch.Tensor
            output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels: int):
        """
        num_channels: int
            number of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor: torch.Tensor, weights=None):
        """
        ---------
        Arguments
        ---------
        input_tensor: torch.Tensor
            input torch tensor of shape = (batch_size, num_channels, H, W)

        ------
        Return
        ------
        output_tensor: torch.Tensor
            output tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        # output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor


class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018, arXiv:1803.02579*
    """

    def __init__(self, num_channels: int, reduction_ratio: int = 2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        ---------
        Arguments
        ---------
        input_tensor: torch.Tensor
            input torch tensor of shape = (batch_size, num_channels, H, W)

        ------
        Return
        ------
        output_tensor: torch.Tensor
            output tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class ResidualBlock(nn.Module):
    """
    Vanilla residual block
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        ---------
        Arguments
        ---------
        in_channels: int
            an integer indicating the number of input channels of the input to the residual block
        out_channels: int
            an integer indicating the number of output channels of the output of the residual block
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.gelu = nn.GELU()
        self.out_channels = out_channels

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ---------
        Arguments
        ---------
        input_tensor: torch.Tensor
            input torch tensor of shape = (batch_size, num_channels, H, W)

        ------
        Return
        ------
        output_tensor: torch.Tensor
            output tensor
        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.gelu(out)
        return out
