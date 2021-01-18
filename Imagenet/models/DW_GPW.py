import torch.nn as nn
import torch


class DWConv(nn.Module):
    def __init__(self, planes, stride):
        super(DWConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, groups=planes)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class GPConv(nn.Module):
    def __init__(self, inplanes, planes, g):
        super(GConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=1, stride=1, groups=g)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


'''
按照原先张量的数据组织方式(batchsize,channels, x, y)

(b,c,h,w) -> (b,g,cpg,h,w) ->(b,cgp,g,h,w)->(b,c,h,w)
'''
def shuffle(x, num_group):
    batchsize, channels, height, width = x.size()
    # (b, c, h, w) -> (b, g, cpg, h, w)
    x = x.view(batchsize, num_group, channels // num_group, height, width)
    # (b,g,cpg,h,w) ->(b,cgp,g,h,w)
    x = torch.transpose(x, 1, 2).contiguous()
    # (b,cgp,g,h,w)->(b,c,h,w)
    x = x.view(batchsize, -1, height, width)
    return x


class ConvUnit(nn.Module):
    def __init__(self, inplanes, planes, g, downsample=False, gf=False):
        super(Conv2d, self).__init__()
        self.dwonv = DWonv(inplanes=inplanes, planes=planes, g=g if not gf else 1)
        self.relu = nn.ReLU(inplace=True)
        self.gconv = GConv(inplanes=planes, planes=planes, planes, g=g)
        self.shortcut = downsample
        self.group = g if not gf else 1

    def forward(self, x):
        if self.shortcut:
            identity = self.shortcut(x)
        else:
            identity = x
        x = self.dwconv(x)
        x = shuffle(x, self.group)
        x = self.gconv(x)
        if self.shortcut:
            x = torch.cat((identity, x), dim=1)
        else:
            x += identity
        return x
        return x


class DWGPWNet(nn.Module):
    def __init__(self, g, num_class=1000):
        super(DWGPWNet, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

        # self.stage1 = self._make_stage(inplace=3,planes=64, g=g, num=2)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # self.stage2 = self._make_stage(inplanes=64, planes=128,  g=g, num=5)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # self.stage3 = self._make_stage(inplanes=128 , planes=256, g=g, num=7)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # self.stage4 = self._make_stage(inplanes=256, planes=512, g=g, num=4)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # self.stage5 = self._make_stage(inplanes=512, planes=512, g=g, num=4)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(self.output_channels[3], num_class)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        # x = self.beginning(x)
        # x = self.stage2(x)
        # x = self.maxpool(x)
        # x = self.stage3(x)
        # x = self.stage4(x)
        # x = self.stage5(x)
        # x = self.pool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    # def _make_stage(self, inplanes, planes, g, num):
    #     stage = list() 
    #     for i in range(num):
    #         if i != 0:
    #             stage.append(ConvUnit(self.inplanes, planes, g=g, downsample=False))
    #     return nn.Sequential(*stage)

def make_layers(cfg, g):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers.append(ConvUnit(inplanes=in_channels , planes=v, g=g ))
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'E': [64, 64, 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

def _DWGPW(arch, cfg, g , pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = DWGPWNet(make_layers(cfgs[cfg], g), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model 



def DWGPW_g2(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return net('DWGPW_g2', 'D', 2, pretrained, progress, **kwargs)



def DWGPW_g4(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return net('DWGPW_g4','D', 4, pretrained, progress, **kwargs)


def DWGPW_g8(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return net('DWGPW_g8','D', 8, pretrained, progress, **kwargs)




