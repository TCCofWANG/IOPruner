import torch.nn as nn
from utils.builder import get_builder
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    # 'vgg16': [64, 64,  128, 128, 256, 256, 256, 512, 512, 512,  512, 512, 512],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }


class VGG(nn.Module):
    def __init__(self, builder,vgg_name,num_classes = 10):
        super(VGG, self).__init__()
        self.features = self._make_layers(builder,cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, builder,cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
            else:
                layers += [builder.conv3x3(in_channels, x),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace = True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size = 1, stride = 1)]
        return nn.Sequential(*layers)

class VGG_sparse(nn.Module):
    def __init__(self, builder, vgg_name, layer_cfg=None, num_classes=10):
        super(VGG_sparse, self).__init__()
        self.layer_cfg = layer_cfg
        self.cfg_index = 0
        self.features = self._make_layers(builder,cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, builder, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                x = x if self.layer_cfg is None else self.layer_cfg[self.cfg_index]
                layers += [builder.conv3x3(in_channels, x),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace = True)]
                in_channels = x
                self.cfg_index += 1
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGG_flops(nn.Module):
    def __init__(self,vgg_name,num_classes = 10):
        super(VGG_flops, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self,cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size = 3, padding = 1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace = True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size = 1, stride = 1)]
        return nn.Sequential(*layers)

class VGG_sparse_flops(nn.Module):
    def __init__(self, vgg_name,layer_cfg=None, num_classes=10):
        super(VGG_sparse_flops, self).__init__()
        self.layer_cfg = layer_cfg
        self.cfg_index = 0
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                x = x if self.layer_cfg is None else self.layer_cfg[self.cfg_index]
                layers += [nn.Conv2d(in_channels, x, kernel_size = 3, padding = 1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace = True)]
                in_channels = x
                self.cfg_index += 1
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
# print(VGG(get_builder(),vgg_name = 'vgg16'))


