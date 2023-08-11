"""
This file contains the implementation of the Artificial Neural Network
Original models from https://github.com/OlgaChernytska/2D-Hand-Pose-Estimation-RGB
"""

import torch
from torch import nn, Tensor


class HandPoseEstimationConvBlock(nn.Module):
    """ Convolutional block used in a U-net like neural network for Hand Pose Estimation task """

    def __init__(self, in_channel: int, out_channel: int):
        """
        :param in_channel: number of input channels for the convolutional block
        """

        super().__init__()

        self._in_channel: int = in_channel
        self._out_channel: int = out_channel

        self._double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                      kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                      kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def __str__(self) -> str:
        """
        :return: string representation for the object
        """
        return f"ConvBlock[In-channels: {self._in_channel}; Out-channels: {self._out_channel}]"

    def __repr__(self) -> str:
        """
        :return: string representation for the object
        """
        return str(self)

    def forward(self, x: Tensor) -> Tensor:
        """
        Implementation of forward pass for the network
        :param x: input tensor
        :return: tensor after passing through the convolutional block
        """
        return self._double_conv(x)


class HandPoseEstimationUNet(nn.Module):
    """
    Implementation of U-Net for Hand Pose Estimation
    """

    _NEURONS: int = 16

    def __init__(self, in_channel: int, out_channel: int):
        """

        :param in_channel: number of input channels for the network
        :param out_channel: number of output channels for the network
        """
        super().__init__()

        self._in_channel: int = in_channel
        self._out_channel: int = out_channel

        # ENCODER PART

        self._conv_down1: HandPoseEstimationConvBlock = \
            HandPoseEstimationConvBlock(in_channel=in_channel,
                                        out_channel=self._NEURONS)
        self._conv_down2: HandPoseEstimationConvBlock = \
            HandPoseEstimationConvBlock(in_channel=self._NEURONS,
                                        out_channel=self._NEURONS * 2)
        self._conv_down3: HandPoseEstimationConvBlock = \
            HandPoseEstimationConvBlock(in_channel=self._NEURONS * 2,
                                        out_channel=self._NEURONS * 4)
        self._conv_bottleneck: HandPoseEstimationConvBlock = \
            HandPoseEstimationConvBlock(in_channel=self._NEURONS * 4,
                                        out_channel=self._NEURONS * 8)  # deepest point the network

        # Downsampling
        self._maxpool: nn.MaxPool2d = nn.MaxPool2d(2)

        # DECODER PART

        self._conv_up1: HandPoseEstimationConvBlock = \
            HandPoseEstimationConvBlock(in_channel=self._NEURONS * 8 + self._NEURONS * 4,
                                        out_channel=self._NEURONS * 4)
        self._conv_up2: HandPoseEstimationConvBlock = \
            HandPoseEstimationConvBlock(in_channel=self._NEURONS * 4 + self._NEURONS * 2,
                                        out_channel=self._NEURONS * 2)
        self._conv_up3: HandPoseEstimationConvBlock = \
            HandPoseEstimationConvBlock(in_channel=self._NEURONS * 2 + self._NEURONS,
                                        out_channel=self._NEURONS)
        self._conv_out: nn.Sequential = nn.Sequential(  # final output of the network
            nn.Conv2d(in_channels=self._NEURONS, out_channels=out_channel,
                      kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

        # Upsampling
        self._upsamle: nn.Upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def __str__(self) -> str:
        """
        :return: string representation for the object
        """
        return f"U-Net[In-channels: {self._in_channel}; Out-channels: {self._out_channel}]"

    def __repr__(self) -> str:
        """
        :return: string representation for the object
        """
        return str(self)

    def forward(self, x: Tensor) -> Tensor:
        """
        Definition of the forward pass of data through the U-Net architecture
        :param x: input tensor
        :return: output tensor passed through the U-Net
        """

        # ENCORED
        conv_d1 = self._conv_down1(x)
        conv_d2 = self._conv_down2(self._maxpool(conv_d1))
        conv_d3 = self._conv_down3(self._maxpool(conv_d2))
        conv_b = self._conv_bottleneck(self._maxpool(conv_d3))

        # DECODER
        conv_u1 = self._conv_up1(torch.cat([self._upsamle(conv_b), conv_d3], dim=1))
        conv_u2 = self._conv_up2(torch.cat([self._upsamle(conv_u1), conv_d2], dim=1))
        conv_u3 = self._conv_up3(torch.cat([self._upsamle(conv_u2), conv_d1], dim=1))

        out = self._conv_out(conv_u3)

        return out


class IoULoss(nn.Module):
    """
    Intersection over Union Loss.
    IoU = Area of Overlap / Area of Union
    IoU loss is modified to use for heatmaps.
    """

    _EPSILON = 1e-6  # prevent division by zero

    def __init__(self):
        super(IoULoss, self).__init__()

    def __str__(self) -> str:
        """
        :return: string representation for the object
        """
        return "IoULoss"

    def __repr__(self) -> str:
        """
        :return: string representation for the object
        """
        return str(self)

    @staticmethod
    def _op_sum(x: Tensor):
        """
        It calculates the sum of tensor values along the last two dimensions
        :param x: input tensor
        :return: sum over the last two dimensions
        """
        return x.sum(-1).sum(-1)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Defines loss calculation
        :param y_pred: predicted labels
        :param y_true: ground truth labels
        :return: loss
        """

        # Computing Intersection
        inter = self._op_sum(x=y_true * y_pred)

        # Computing Union
        # A \cup B = A + B - (A \cap B)
        union = (
                self._op_sum(x=y_true ** 2)  # A
                + self._op_sum(x=y_pred ** 2)  # B
                - inter  # (A \cap B)
        )

        # Computing Intersection over Union
        iou: Tensor = (inter + self._EPSILON) / (union + self._EPSILON)
        iou: Tensor = torch.mean(input=iou)

        return 1 - iou
