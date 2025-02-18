# builder
import math

import torch
import torch.nn as nn

from utils.options import args
import utils.conv_matrix


class Builder(object):
    def __init__(self, conv_layer, first_layer = None):
        self.conv_layer = conv_layer
        self.first_layer = first_layer or conv_layer

    def conv(self, kernel_size, in_planes, out_planes, stride = 1, first_layer = False, bias = True if args.arch == 'vgg_cifar' or args.arch == 'googlenet_cifar' else False):
        conv_layer = self.first_layer if first_layer else self.conv_layer

        if first_layer:
            print(f"==> Building first layer with {args.first_layer_type}")

        if kernel_size == 3:
            conv = conv_layer(
                    in_planes,
                    out_planes,
                    kernel_size = 3,
                    stride = stride,
                    padding = 1,
                    bias = bias,
                    )
            # print(conv)
        elif kernel_size == 1:
            conv = conv_layer(
                    in_planes, out_planes, kernel_size = 1, stride = stride, bias = bias
                    )
        elif kernel_size == 5:
            conv = conv_layer(
                    in_planes,
                    out_planes,
                    kernel_size = 5,
                    stride = stride,
                    padding = 2,
                    bias = bias,
                    )
        elif kernel_size == 7:
            conv = conv_layer(
                    in_planes,
                    out_planes,
                    kernel_size = 7,
                    stride = stride,
                    padding = 3,
                    bias = bias,
                    )
        else:
            return None

        # self._init_conv(conv)

        return conv

    def conv2d(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride = 1,
            padding = 0,
            dilation = 1,
            groups = 1,
            bias = True,
            padding_mode = "zeros",
            ):
        return self.conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
                padding_mode,
                )

    def conv3x3(self, in_planes, out_planes, stride = 1, first_layer = False):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, stride = stride, first_layer = first_layer)
        return c

    def conv1x1(self, in_planes, out_planes, stride = 1, first_layer = False):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride = stride, first_layer = first_layer)
        return c

    def conv1x1_fc(self, in_planes, out_planes, stride = 1, first_layer = False):
        """full connect layer"""
        c = self.conv(1, in_planes, out_planes, stride = stride, first_layer = first_layer, bias = True)
        return c

    def conv7x7(self, in_planes, out_planes, stride = 1, first_layer = False):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride = stride, first_layer = first_layer)
        return c

    def conv5x5(self, in_planes, out_planes, stride = 1, first_layer = False):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride = stride, first_layer = first_layer)
        return c



def get_builder():
    print("==> Conv Type: {}".format(args.conv_type))

    conv_layer = getattr(utils.conv_matrix, args.conv_type)

    if args.first_layer_type is not None:
        first_layer = getattr(utils.conv_matrix, args.first_layer_type)
        print(f"==> First Layer Type {args.first_layer_type}")
    else:
        first_layer = None

    builder = Builder(conv_layer = conv_layer, first_layer = first_layer)

    return builder
