import torch.nn as nn
import torchvision.models as models
from resnet import resnet101
import pdb

from blocks import (
    RefineNetBlock,
    ResidualConvUnit,
    RefineNetBlockImprovedPooling
)


class BaseRefineNet3Cascade(nn.Module):

    def __init__(self, refinenet_block,
                 input_channels=3,
                 features=256,
                 resnet_factory=resnet101,
                 pretrained=False,
                 freeze_resnet=True):
        """Multi-path 4-Cascaded RefineNet for image segmentation

        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True

        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super(BaseRefineNet3Cascade, self).__init__()

        resnet = resnet_factory()

        self.layer1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )

        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        if freeze_resnet:
            layers = [self.layer1, self.layer2, self.layer3]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False

        # adjusting channels for different input layer
        self.layer1_rn = nn.Conv2d(
            256, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(
            512, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(
            1024, features, kernel_size=3, stride=1, padding=1, bias=False)
        # gabriel: for making it deeper to original numbers of channels
        self.out_rn = nn.Conv2d(
            features, 1024, kernel_size=3, stride=1, padding=1, bias=False)

        # (features, input_size // 8), (features, input_size // 4):
        # ((256,     32),              (256,      64))
        # gabriel: incase encounter odd numberi, use 2 or 1
        self.refinenet1 = RefineNetBlock(
            features, (features, 1))
        self.refinenet2 = RefineNetBlock(
            features, (features, 2), (features, 1))
        self.refinenet3 = RefineNetBlock(
            features, (features, 2), (features, 1))

        self.output_conv = nn.Sequential(
            ResidualConvUnit(features),
            ResidualConvUnit(features)
        )
        #self.output_conv = nn.Sequential(
        #    ResidualConvUnit(features),
        #    ResidualConvUnit(features),
        #    nn.Conv2d(features, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        #)

    def forward(self, x):

        layer_1 = self.layer1(x)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)

        # adjusting channels for different input layer
        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)

        path_1 = self.refinenet1(layer_1_rn)
        path_2 = self.refinenet2(layer_2_rn, path_1)
        path_3 = self.refinenet3(layer_3_rn, path_2)
        path_4 = self.output_conv(path_3)
        out = self.out_rn(path_4)
        return out

class RefineNet4CascadePoolingImproved(BaseRefineNet3Cascade):

    def __init__(self, input_shape,
                 features=256,
                 resnet_factory=models.resnet101,
                 pretrained=True,
                 freeze_resnet=True):
        """Multi-path 4-Cascaded RefineNet for image segmentation with improved pooling
        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True
        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__(self, input_shape, RefineNetBlockImprovedPooling,
                         num_classes=num_classes, features=features,
                         resnet_factory=resnet_factory, pretrained=pretrained,
                         freeze_resnet=freeze_resnet)


class RefineNet3Cascade(BaseRefineNet3Cascade):

    def __init__(self, input_channels=3,
                 features=256,
                 resnet_factory=resnet101,
                 pretrained=False,
                 freeze_resnet=True):
        """Multi-path 4-Cascaded RefineNet for image segmentation
        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True
        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super(RefineNet3Cascade, self).__init__(RefineNetBlock,
                         input_channels=3,
                         features=features,
                         resnet_factory=resnet_factory, pretrained=pretrained,
                         freeze_resnet=freeze_resnet)
