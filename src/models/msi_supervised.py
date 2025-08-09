import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from kan import KANLayer


from models.building_blocks import (
    ResidualBlock,
    ChannelSpatialSELayer,
    PolarizedSelfAttention,
)


class MSI_ResNet(nn.Module):
    def __init__(
        self,
        num_input_bands: int = 3,
        num_classes: int = 10,
        list_filters: List[int] = [64, 128, 256],
        list_num_res_units_per_block: List[int] = [4, 4, 4],
        dropout_ratio: float = 0.2,
    ):
        """
        ---------
        Arguments
        ---------
        num_input_bands: int
            number of input bands (default: 3)
        num_classes: int
            an integer indicating the number of classes in the dataset (default: 10)
        list_filters: List[int]
            a list of number of filters to be used in the mdoel (default: [64, 128, 256])
        list_num_res_units_per_block: List[int]
            a list of integers representing number of residual units per block (default: [4, 4, 4])
        dropout_ratio: float
            the dropout ratio to be used in the dropout layer (default: 0.2)
        """
        super().__init__()
        self.list_num_res_units_per_block = list_num_res_units_per_block

        # do not use batch norm layer after the first layer of convolution that
        # is applied on the input data bands from the MSI
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_input_bands,
                out_channels=list_filters[0],
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.GELU(),
        )

        self.residual_block_1 = nn.Sequential(
            *[
                ResidualBlock(list_filters[0], list_filters[0])
                for i in range(self.list_num_res_units_per_block[0])
            ]
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=list_filters[0],
                out_channels=list_filters[1],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(list_filters[1]),
            nn.GELU(),
        )

        self.residual_block_2 = nn.Sequential(
            *[
                ResidualBlock(list_filters[1], list_filters[1])
                for i in range(self.list_num_res_units_per_block[1])
            ]
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=list_filters[1],
                out_channels=list_filters[2],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(list_filters[2]),
            nn.GELU(),
        )

        self.residual_block_3 = nn.Sequential(
            *[
                ResidualBlock(list_filters[2], list_filters[2])
                for i in range(self.list_num_res_units_per_block[2])
            ]
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(list_filters[2], num_classes)
        self.dropout = nn.Dropout(p=dropout_ratio)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
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
        x = self.conv_block_1(x)
        x = self.residual_block_1(x)
        x = self.conv_block_2(x)
        x = self.residual_block_2(x)
        x = self.conv_block_3(x)
        x = self.residual_block_3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class MSI_SE_ResNet(nn.Module):
    def __init__(
        self,
        num_input_bands: int = 3,
        num_classes: int = 10,
        list_filters: List[int] = [64, 128, 256],
        list_num_res_units_per_block: List[int] = [4, 4, 4],
        dropout_ratio: float = 0.2,
    ):
        """
        ---------
        Arguments
        ---------
        num_input_bands: int
            number of input bands (default: 3)
        num_classes: int
            an integer indicating the number of classes in the dataset (default: 10)
        list_filters: List[int]
            a list of number of filters to be used in the mdoel (default: [64, 128, 256])
        list_num_res_units_per_block: List[int]
            a list of integers representing number of residual units per block (default: [4, 4, 4])
        dropout_ratio: float
            the dropout ratio to be used in the dropout layer (default: 0.2)
        """
        super().__init__()
        self.list_num_res_units_per_block = list_num_res_units_per_block

        # do not use batch norm layer after the first layer of convolution that
        # is applied on the input data bands from the MSI
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_input_bands,
                out_channels=list_filters[0],
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.GELU(),
            ChannelSpatialSELayer(list_filters[0]),
        )

        self.residual_block_1 = nn.Sequential(
            *[
                ResidualBlock(list_filters[0], list_filters[0])
                for i in range(self.list_num_res_units_per_block[0])
            ]
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=list_filters[0],
                out_channels=list_filters[1],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(list_filters[1]),
            nn.GELU(),
            ChannelSpatialSELayer(list_filters[1]),
        )
        self.residual_block_2 = nn.Sequential(
            *[
                ResidualBlock(list_filters[1], list_filters[1])
                for i in range(self.list_num_res_units_per_block[1])
            ]
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=list_filters[1],
                out_channels=list_filters[2],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(list_filters[2]),
            nn.GELU(),
            ChannelSpatialSELayer(list_filters[2]),
        )

        self.residual_block_3 = nn.Sequential(
            *[
                ResidualBlock(list_filters[2], list_filters[2])
                for i in range(self.list_num_res_units_per_block[2])
            ]
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(list_filters[2], num_classes)
        self.dropout = nn.Dropout(p=dropout_ratio)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
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
        x = self.conv_block_1(x)
        x = self.residual_block_1(x)
        x = self.conv_block_2(x)
        x = self.residual_block_2(x)
        x = self.conv_block_3(x)
        x = self.residual_block_3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class MSI_PSA_ResNet(nn.Module):
    def __init__(
        self,
        num_input_bands: int = 3,
        num_classes: int = 10,
        list_filters: List[int] = [64, 128, 256],
        list_num_res_units_per_block: List[int] = [4, 4, 4],
        dropout_ratio: float = 0.2,
    ):
        """
        ---------
        Arguments
        ---------
        num_input_bands: int
            number of input bands (default: 3)
        num_classes: int
            an integer indicating the number of classes in the dataset (default: 10)
        list_filters: List[int]
            a list of number of filters to be used in the mdoel (default: [64, 128, 256])
        list_num_res_units_per_block: List[int]
            a list of integers representing number of residual units per block (default: [4, 4, 4])
        dropout_ratio: float
            the dropout ratio to be used in the dropout layer (default: 0.2)
        """
        super().__init__()
        self.list_num_res_units_per_block = list_num_res_units_per_block

        # do not use batch norm layer after the first layer of convolution that
        # is applied on the input data bands from the MSI
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_input_bands,
                out_channels=list_filters[0],
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.GELU(),
            PolarizedSelfAttention(list_filters[0], list_filters[0]),
        )

        self.residual_block_1 = nn.Sequential(
            *[
                ResidualBlock(list_filters[0], list_filters[0])
                for i in range(self.list_num_res_units_per_block[0])
            ]
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=list_filters[0],
                out_channels=list_filters[1],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(list_filters[1]),
            nn.GELU(),
            PolarizedSelfAttention(list_filters[1], list_filters[1]),
        )
        self.residual_block_2 = nn.Sequential(
            *[
                ResidualBlock(list_filters[1], list_filters[1])
                for i in range(self.list_num_res_units_per_block[1])
            ]
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=list_filters[1],
                out_channels=list_filters[2],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(list_filters[2]),
            nn.GELU(),
            PolarizedSelfAttention(list_filters[2], list_filters[2]),
        )

        self.residual_block_3 = nn.Sequential(
            *[
                ResidualBlock(list_filters[2], list_filters[2])
                for i in range(self.list_num_res_units_per_block[2])
            ]
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(list_filters[2], num_classes)
        self.dropout = nn.Dropout(p=dropout_ratio)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
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
        x = self.conv_block_1(x)
        x = self.residual_block_1(x)
        x = self.conv_block_2(x)
        x = self.residual_block_2(x)
        x = self.conv_block_3(x)
        x = self.residual_block_3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class MSI_ResKANet(nn.Module):
    def __init__(
        self,
        num_input_bands: int = 3,
        num_classes: int = 10,
        list_filters: List[int] = [64, 128, 256],
        list_num_res_units_per_block: List[int] = [4, 4, 4],
        dropout_ratio: float = 0.2,
        device: str = "cuda",
    ):
        """
        ---------
        Arguments
        ---------
        num_input_bands: int
            number of input bands (default: 3)
        num_classes: int
            an integer indicating the number of classes in the dataset (default: 10)
        list_filters: List[int]
            a list of number of filters to be used in the mdoel (default: [64, 128, 256])
        list_num_res_units_per_block: List[int]
            a list of integers representing number of residual units per block (default: [4, 4, 4])
        dropout_ratio: float
            the dropout ratio to be used in the dropout layer (default: 0.2)
        device: str
            a string indicating the device on which the KAN layer needs to be initialized
        """
        super().__init__()
        self.list_num_res_units_per_block = list_num_res_units_per_block

        # do not use batch norm layer after the first layer of convolution that
        # is applied on the input data bands from the MSI
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_input_bands,
                out_channels=list_filters[0],
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.GELU(),
        )

        self.residual_block_1 = nn.Sequential(
            *[
                ResidualBlock(list_filters[0], list_filters[0])
                for i in range(self.list_num_res_units_per_block[0])
            ]
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=list_filters[0],
                out_channels=list_filters[1],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(list_filters[1]),
            nn.GELU(),
        )

        self.residual_block_2 = nn.Sequential(
            *[
                ResidualBlock(list_filters[1], list_filters[1])
                for i in range(self.list_num_res_units_per_block[1])
            ]
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=list_filters[1],
                out_channels=list_filters[2],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(list_filters[2]),
            nn.GELU(),
        )

        self.residual_block_3 = nn.Sequential(
            *[
                ResidualBlock(list_filters[2], list_filters[2])
                for i in range(self.list_num_res_units_per_block[2])
            ]
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.kan_layer = KANLayer(list_filters[2], num_classes, device=device)
        self.dropout = nn.Dropout(p=dropout_ratio)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
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
        x = self.conv_block_1(x)
        x = self.residual_block_1(x)
        x = self.conv_block_2(x)
        x = self.residual_block_2(x)
        x = self.conv_block_3(x)
        x = self.residual_block_3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x, _, _, _ = self.kan_layer(x)
        x = self.dropout(x)
        return x


class MSI_SE_ResKANet(nn.Module):
    def __init__(
        self,
        num_input_bands: int = 3,
        num_classes: int = 10,
        list_filters: List[int] = [64, 128, 256],
        list_num_res_units_per_block: List[int] = [4, 4, 4],
        dropout_ratio: float = 0.2,
        device: str = "cuda",
    ):
        """
        ---------
        Arguments
        ---------
        num_input_bands: int
            number of input bands (default: 3)
        num_classes: int
            an integer indicating the number of classes in the dataset (default: 10)
        list_filters: List[int]
            a list of number of filters to be used in the mdoel (default: [64, 128, 256])
        list_num_res_units_per_block: List[int]
            a list of integers representing number of residual units per block (default: [4, 4, 4])
        dropout_ratio: float
            the dropout ratio to be used in the dropout layer (default: 0.2)
        device: str
            a string indicating the device on which the KAN layer needs to be initialized
        """
        super().__init__()
        self.list_num_res_units_per_block = list_num_res_units_per_block

        # do not use batch norm layer after the first layer of convolution that
        # is applied on the input data bands from the MSI
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_input_bands,
                out_channels=list_filters[0],
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.GELU(),
            ChannelSpatialSELayer(list_filters[0]),
        )

        self.residual_block_1 = nn.Sequential(
            *[
                ResidualBlock(list_filters[0], list_filters[0])
                for i in range(self.list_num_res_units_per_block[0])
            ]
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=list_filters[0],
                out_channels=list_filters[1],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(list_filters[1]),
            nn.GELU(),
            ChannelSpatialSELayer(list_filters[1]),
        )

        self.residual_block_2 = nn.Sequential(
            *[
                ResidualBlock(list_filters[1], list_filters[1])
                for i in range(self.list_num_res_units_per_block[1])
            ]
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=list_filters[1],
                out_channels=list_filters[2],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(list_filters[2]),
            nn.GELU(),
            ChannelSpatialSELayer(list_filters[2]),
        )

        self.residual_block_3 = nn.Sequential(
            *[
                ResidualBlock(list_filters[2], list_filters[2])
                for i in range(self.list_num_res_units_per_block[2])
            ]
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.kan_layer = KANLayer(list_filters[2], num_classes, device=device)
        self.dropout = nn.Dropout(p=dropout_ratio)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
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
        x = self.conv_block_1(x)
        x = self.residual_block_1(x)
        x = self.conv_block_2(x)
        x = self.residual_block_2(x)
        x = self.conv_block_3(x)
        x = self.residual_block_3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x, _, _, _ = self.kan_layer(x)
        x = self.dropout(x)
        return x


class MSI_PSA_ResKANet(nn.Module):
    def __init__(
        self,
        num_input_bands: int = 3,
        num_classes: int = 10,
        list_filters: List[int] = [64, 128, 256],
        list_num_res_units_per_block: List[int] = [4, 4, 4],
        dropout_ratio: float = 0.2,
        device: str = "cuda",
    ):
        """
        ---------
        Arguments
        ---------
        num_input_bands: int
            number of input bands (default: 3)
        num_classes: int
            an integer indicating the number of classes in the dataset (default: 10)
        list_filters: List[int]
            a list of number of filters to be used in the mdoel (default: [64, 128, 256])
        list_num_res_units_per_block: List[int]
            a list of integers representing number of residual units per block (default: [4, 4, 4])
        dropout_ratio: float
            the dropout ratio to be used in the dropout layer (default: 0.2)
        device: str
            a string indicating the device on which the KAN layer needs to be initialized
        """
        super().__init__()
        self.list_num_res_units_per_block = list_num_res_units_per_block

        # do not use batch norm layer after the first layer of convolution that
        # is applied on the input data bands from the MSI
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_input_bands,
                out_channels=list_filters[0],
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.GELU(),
            PolarizedSelfAttention(list_filters[0], list_filters[0]),
        )

        self.residual_block_1 = nn.Sequential(
            *[
                ResidualBlock(list_filters[0], list_filters[0])
                for i in range(self.list_num_res_units_per_block[0])
            ]
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=list_filters[0],
                out_channels=list_filters[1],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(list_filters[1]),
            nn.GELU(),
            PolarizedSelfAttention(list_filters[1], list_filters[1]),
        )

        self.residual_block_2 = nn.Sequential(
            *[
                ResidualBlock(list_filters[1], list_filters[1])
                for i in range(self.list_num_res_units_per_block[1])
            ]
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=list_filters[1],
                out_channels=list_filters[2],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(list_filters[2]),
            nn.GELU(),
            PolarizedSelfAttention(list_filters[2], list_filters[2]),
        )

        self.residual_block_3 = nn.Sequential(
            *[
                ResidualBlock(list_filters[2], list_filters[2])
                for i in range(self.list_num_res_units_per_block[2])
            ]
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.kan_layer = KANLayer(list_filters[2], num_classes, device=device)
        self.dropout = nn.Dropout(p=dropout_ratio)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
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
        x = self.conv_block_1(x)
        x = self.residual_block_1(x)
        x = self.conv_block_2(x)
        x = self.residual_block_2(x)
        x = self.conv_block_3(x)
        x = self.residual_block_3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x, _, _, _ = self.kan_layer(x)
        x = self.dropout(x)
        return x
