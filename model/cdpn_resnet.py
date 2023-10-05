# adapted from https://github.com/THU-DA-6D-Pose-Group/GDR-Net
import torch.nn as nn
import torch
import logging
from torchvision.models.resnet import BasicBlock, Bottleneck
from mmcv.cnn import normal_init, constant_init

logger = logging.getLogger(__name__)

# Specification
resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], "resnet18"),
    34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], "resnet34"),
    50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], "resnet50"),
    101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], "resnet101"),
    152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], "resnet152"),
}


class ResNetBackbone(nn.Module):
    def __init__(self, block, layers, in_channel=3):
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):  # x.shape [32, 3, 256, 256]
        x = self.conv1(x)  # x.shape [32, 64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        x_low_feature = self.maxpool(x)  # x.shape [32, 64, 64, 64]
        x_f64 = self.layer1(x_low_feature)  # x.shape [32, 256, 64, 64]
        x_f32 = self.layer2(x_f64)  # x.shape [32, 512, 32, 32]
        x_f16 = self.layer3(x_f32)  # x.shape [32, 1024, 16, 16]
        x_high_feature = self.layer4(x_f16)  # x.shape [32, 2048, 8, 8]

        return x_high_feature, x_f64, x_f32, x_f16


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers=3,
        num_filters=256,
        kernel_size=3,
        output_kernel_size=1,
        backbone_num_layers = 34,
        concat = False,
    ):
        super().__init__()

        self.concat = concat
        assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, "Only support kenerl 2, 3 and 4"

        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        assert output_kernel_size == 1 or output_kernel_size == 3, "Only support kenerl 1 and 3"
        if output_kernel_size == 1:
            pad = 0
        elif output_kernel_size == 3:
            pad = 1

        if self.concat:
            _, _, channels, _ = resnet_spec[backbone_num_layers]
            self.features = nn.ModuleList()
            self.features.append(
                nn.ConvTranspose2d(
                    in_channels,
                    num_filters,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))
            for i in range(num_layers):
                self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
                self.features.append(
                    nn.Conv2d(
                        num_filters + channels[-2 - i], num_filters, kernel_size=3, stride=1, padding=1, bias=False
                    )
                )
                self.features.append(nn.BatchNorm2d(num_filters))
                self.features.append(nn.ReLU(inplace=True))

                self.features.append(
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
                )
                self.features.append(nn.BatchNorm2d(num_filters))
                self.features.append(nn.ReLU(inplace=True))
        else:
            self.features = nn.ModuleList()
            self.features.append(
                nn.ConvTranspose2d(
                    in_channels,
                    num_filters,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))
            for i in range(num_layers):
                if i >= 1:
                    self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
                self.features.append(
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
                )
                self.features.append(nn.BatchNorm2d(num_filters))
                self.features.append(nn.ReLU(inplace=True))

                self.features.append(
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
                )
                self.features.append(nn.BatchNorm2d(num_filters))
                self.features.append(nn.ReLU(inplace=True))

        self.out_layer = nn.Conv2d(
                num_filters,
                out_channels,
                kernel_size=output_kernel_size,
                padding=pad,
                bias=True,
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

    def forward(self, x, x_f64=None, x_f32=None, x_f16=None):
        if self.concat:
            for i, l in enumerate(self.features):
                if i == 3:
                    x = torch.cat([x, x_f16], 1)
                elif i == 12:
                    x = torch.cat([x, x_f32], 1)
                elif i == 21:
                    x = torch.cat([x, x_f64], 1)
                x = l(x)
        else:
            for i, l in enumerate(self.features):
                x = l(x)
        out = self.out_layer(x)
        return out, x


class ResNet_cdpn(nn.Module):
    def __init__(self, in_channels, out_channels, pre_trained, **kwargs) -> None:
        super().__init__()
        back_layers_num, concat = kwargs.get('back_layers_num', 34), kwargs.get('concat',False)
        block_type, layers, channels, name = resnet_spec[back_layers_num]

        self.backbone = ResNetBackbone(block_type, layers, in_channels,)
        self.decoder = Decoder(channels[-1], out_channels, concat = concat)
        
        if pre_trained!=False:
            trained_model = torch.load("assets/resnet34-333f7ec4.pth")
            trained_model.pop('fc.weight')
            trained_model.pop('fc.bias')
            self.backbone.load_state_dict(trained_model, strict=True)
            logger.info('loaded resnet backbone from pytorch model zoo')
    
    def forward(self,rgb):
        x_high_feature, x_f64, x_f32, x_f16 = self.backbone(rgb)
        out, feature = self.decoder(x_high_feature, x_f64, x_f32, x_f16)
        return out, feature


def get_network(in_channels, out_channels, **kwargs):
    net = ResNet_cdpn(in_channels, out_channels, True, **kwargs)
    net.feature_dim = 256
    return net
