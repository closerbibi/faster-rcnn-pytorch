import torch.nn as nn
import pdb


class ResidualConvUnit(nn.Module):

    def __init__(self, features):
        super(ResidualConvUnit, self).__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class MultiResolutionFusion(nn.Module):

    def __init__(self, out_feats, *shapes):
        super(MultiResolutionFusion, self).__init__()

        # ((256,64),(256,32)), get number 1 element in ecah tuple
        _, max_size = max(shapes, key=lambda x: x[1])

        for i, shape in enumerate(shapes):
            feat, size = shape
            if max_size % size != 0:
                raise ValueError("max_size not divisble by shape {:d}".format(i))

            scale_factor = max_size // size
            if scale_factor != 1:
                self.add_module("resolve{:d}".format(i), nn.Sequential(
                    nn.Conv2d(feat, out_feats, kernel_size=3,
                              stride=1, padding=1, bias=False),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # gabriel: from upsampling
                ))
            else:
                self.add_module(
                    "resolve{:d}".format(i),
                    nn.Conv2d(feat, out_feats, kernel_size=3,
                              stride=1, padding=1, bias=False)
                )

    def forward(self, *xs):

        output = self.resolve0(xs[0])

        for i, x in enumerate(xs[1:], 1):
            #output = output + self.__getattr__("resolve{:d}".format(i))(x)
            pass

        return output





class ChainedResidualPool(nn.Module):

    def __init__(self, feats):
        super(ChainedResidualPool, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 4):
            self.add_module("block{:d}".format(i), nn.Sequential(
                nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=False)
            ))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 4):
            path = self.__getattr__("block{:d}".format(i))(path)
            # x += path???
            x = x + path

        return x


class ChainedResidualPoolImproved(nn.Module):

    def __init__(self, feats):
        super(ChainedResidualPoolImproved, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 5):
            self.add_module("block{:d}".format(i), nn.Sequential(
                nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=False),
                nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
            ))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 5):
            path = self.__getattr__("block{:d}".format(i))(path)
            # x += path???
            x = x + path

        return x


class BaseRefineNetBlock(nn.Module):

    def __init__(self, features,
                 residual_conv_unit,
                 multi_resolution_fusion,
                 chained_residual_pool, *shapes):
        super(BaseRefineNetBlock, self).__init__()

        for i, shape in enumerate(shapes):
            feats = shape[0]
            self.add_module("rcu{:d}".format(i), nn.Sequential(
                residual_conv_unit(feats),
                residual_conv_unit(feats)
            ))

        if len(shapes) != 1:
            self.mrf = multi_resolution_fusion(features, *shapes)
        else:
            self.mrf = None

        self.crp = chained_residual_pool(features)
        self.output_conv = residual_conv_unit(features)

    def forward(self, *xs):
        for i, x in enumerate(xs):
            # output x?????????
            x = self.__getattr__("rcu{:d}".format(i))(x)

        if self.mrf is not None:
            out = self.mrf(*xs)
        else:
            out = xs[0]

        out = self.crp(out)
        return self.output_conv(out)


class RefineNetBlock(BaseRefineNetBlock):

    def __init__(self, features, *shapes):
        super(RefineNetBlock, self).__init__(features, ResidualConvUnit,
                         MultiResolutionFusion,
                         ChainedResidualPool, *shapes)


class RefineNetBlockImprovedPooling(nn.Module):

    def __init__(self, features, *shapes):
        super(RefineNetBlockImprovedPooling, self).__init__(features, ResidualConvUnit,
                         MultiResolutionFusion,
                         ChainedResidualPoolImproved, *shapes)
