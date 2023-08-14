"""
Network Architecture
--------------------

This modules contains the implementation of the Artificial Neural Network for Hand Pose Estimation task
Original models from https://github.com/OlgaChernytska/2D-Hand-Pose-Estimation-RGB

Classes:
- HandPoseEstimationConvBlock: Convolutional block used in a U-net like neural network for Hand Pose Estimation task.
- HandPoseEstimationUNet: Implementation of U-Net for Hand Pose Estimation.
- IoULoss: Intersection over Union Loss for evaluating the accuracy of predicted heatmaps.

Note:
The implementation in this file is adapted from the original repository mentioned above.

"""

import torch
from torch import nn, Tensor


class HandPoseEstimationConvBlock(nn.Module):
    """
    Convolutional block used in a U-net like neural network for Hand Pose Estimation task.
    This block consists of two consecutive convolutional layers followed by batch normalization and ReLU activation.

    Attributes:
    - in_channel: number of input channels for the convolutional block.
    - out_channel: number of output channels for the convolutional block.
    - double_conv: sequential container for the double convolutional layers
                   with batch normalization and ReLU activation.
    """

    # CONSTRUCTOR

    def __init__(self, in_channel: int, out_channel: int):
        """
        Initialize an instance of HandPoseEstimationConvBlock.

        :param in_channel: number of input channels for the convolutional block.
        :param out_channel: number of output channels for the convolutional block.
        """

        super().__init__()

        self._in_channel: int = in_channel
        self._out_channel: int = out_channel

        # Define the double convolutional layers with batch normalization and ReLU activation
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

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Returns a string representation of the HandPoseEstimationConvBlock object.
        :return: string representation for the object.
        """

        return f"HandPoseEstimationConvBlock[In-channels: {self._in_channel}; Out-channels: {self._out_channel}]"

    def __repr__(self) -> str:
        """
        Returns a string representation of the HandPoseEstimationConvBlock object.
        :return: string representation for the object. (nn.Upsample)
        """

        return str(self)

    # OVERRIDE

    def forward(self, x: Tensor) -> Tensor:
        """
         Forward pass through the convolutional block.

        :param x: input tensor.
        :return: output tensor after passing through the convolutional block.
        """

        return self._double_conv(x)


class HandPoseEstimationUNet(nn.Module):
    """
    Implementation of U-Net for Hand Pose Estimation.

    The U-Net architecture is used for semantic segmentation tasks, and it consists of an encoder and a decoder.
    The encoder part captures high-level features, while the decoder part recovers spatial information.

    Attributes:
    - in_channel: Number of input channels for the network.
    - out_channel: Number of output channels for the network.
    - NEURONS: Number of neurons used as a base unit in the network.

    - conv_down1: Convolutional block for the first down-sampling step.
    - conv_down2: Convolutional block for the second down-sampling step.
    - conv_down3: Convolutional block for the third down-sampling step.
    - conv_bottleneck: Convolutional block for the deepest point of the network.
    - maxpool: Max pooling layer for down-sampling.

    - conv_up1: Convolutional block for the first up-sampling step.
    - conv_up2: Convolutional block for the second up-sampling step.
    - conv_up3: Convolutional block for the third up-sampling step.
    - conv_out: Final convolutional layer for the network output.
    - upsample: Up-sampling layer for recovering spatial information.
    """

    _NEURONS: int = 16

    def __init__(self, in_channel: int, out_channel: int):
        """
        Initialize an instance of HandPoseEstimationUNet.

        :param in_channel: number of input channels for the network.
        :param out_channel: number of output channels for the network.
        """

        super().__init__()

        self._in_channel: int = in_channel
        self._out_channel: int = out_channel

        # ENCODER PART

        # Initialize convolutional blocks for down-sampling steps
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

        # Initialize max pooling layer for down-sampling
        self._maxpool: nn.MaxPool2d = nn.MaxPool2d(2)

        # Initialize convolutional blocks for up-sampling steps
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

        # Initialize up-sampling layer for recovering spatial information
        self._upsamle: nn.Upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def __str__(self) -> str:
        """
        Returns a string representation of the U-Net.

        :return: string representation for the object
        """

        return f"HandPoseEstimationUNet[In-channels: {self._in_channel}; Out-channels: {self._out_channel}]"

    def __repr__(self) -> str:
        """
        Returns a string representation of the U-Net.

        :return: string representation for the object
        """

        return str(self)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the U-Net architecture.

        :param x: input tensor.
        :returns: output tensor after passing through the U-Net architecture.
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
    Intersection over Union Loss for semantic segmentation tasks.

    The IoU (Intersection over Union) Loss measures the dissimilarity between predicted and ground truth labels.
    It calculates the Intersection over Union metric, which is a measure of the overlap between the two sets.

    """

    _EPSILON = 1e-6  # Prevent division by zero

    # CONSTRUCTOR

    def __init__(self):
        """
        Initialize a new instance of IoULoss.
        """

        super(IoULoss, self).__init__()

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Returns a string representation of the IoU Loss.
        :return: string representation for the object.
        """

        return "IoULoss"

    def __repr__(self) -> str:
        """
        Returns a string representation of the IoU Loss.
        :return: string representation for the object.
        """

        return str(self)

    # OVERRIDE

    @staticmethod
    def _op_sum(x: Tensor):
        """
        It calculates the sum of tensor values along the last two dimensions.

        :param x: input tensor.
        :return: sum over the last two dimensions.
        """
        return x.sum(-1).sum(-1)

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Calculates the IoU loss between predicted and ground truth labels.

        :param y_pred: predicted labels.
        :param y_true: ground truth labels.
        :return: IoU loss value.
        """

        # Computing Intersection
        inter = self._op_sum(x=y_true * y_pred)

        # Computing Union
        union = (
                self._op_sum(x=y_true ** 2)
                + self._op_sum(x=y_pred ** 2)
                - inter
        )

        # Computing Intersection over Union
        iou: Tensor = (inter + self._EPSILON) / (union + self._EPSILON)
        iou: Tensor = torch.mean(input=iou)

        # Convert IoU to loss by subtracting from 1
        loss = 1 - iou

        return loss
