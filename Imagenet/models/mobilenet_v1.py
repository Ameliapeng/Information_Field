import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .utils import load_state_dict_from_url

__all__ = ['MobileNetV1', 'mobilenet_v1']


model_urls = {
    'mobilenet_v1': 'https://drive.google.com/file/d/17Z4H8eWScuMXuyK0VmaJg698MTjRjTD-/view',
}

class MobileNetV1(nn.Module):
      def __init__(self, num_classes=1000):
            super(MobileNetV1, self).__init__()
            def conv_bn(inp, oup, stride):
                return nn.Sequential(
                    nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True)
                )
            def conv_dw(inp, oup, stride):
                return nn.Sequential(
                    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True),
                )
            self.model = nn.Sequential(
                conv_bn(  3,  32, 2), 

                conv_dw( 32,  32, 1),
                conv_dw( 32,  64, 1),

                conv_dw( 64, 128, 2),
                conv_dw(128, 128, 1),

                conv_dw(128, 256, 2),
                conv_dw(256, 256, 1),

                conv_dw(256, 512, 2),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),

                conv_dw(512, 1024, 2),
                conv_dw(1024, 1024, 1),
                nn.AvgPool2d(7),
            )
            self.fc = nn.Linear(1024, 1000)

      def forward(self, x):
                x = self.model(x)
                x = x.view(-1, 1024)
                x = self.fc(x)
                return x

def mobilenet_v1(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV1(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v1'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

if __name__ == '__main__':
    model = MobileNetV1()
    from torchsummary import summary
    summary(model.cuda(), (3, 224, 224))