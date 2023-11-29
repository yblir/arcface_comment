import torch
from torch import nn

__all__ = ['iresnet18', 'iresnet34', 'iresnet50']


class IBasicBlock(nn.Module):
    expansion = 1  # 这个变量是干嘛的？

    def __init__(self, in_planes, out_planes, stride=(1, 1), down_sample=None):
        # groups=1, base_width=64, dilation=1
        super(IBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes)
        # prelu带有一个可学习的参数。在所有通道上学习同一个a，（out_channels）:在每个通道上学习不同的a
        self.prelu = nn.PReLU(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes,
                               kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.down_sample = down_sample

    def forward(self, x):
        identify = x
        # 这是对原始resnet修改，先经过bn层
        out = self.conv1(self.bn1(x))
        out = self.conv2(self.prelu(self.bn2(out)))
        out = self.bn3(out)

        if self.down_sample:
            identify = self.down_sample(x)
        out += identify

        return out


class IResNet(nn.Module):
    fc_scale = 7 * 7  # 这是干嘛的 ？

    def __init__(self, block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.fp16 = fp16  # 指是否有float16与32混合训练， 默认是32精度
        self.in_planes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple')
        self.groups = groups
        self.base_width = width_per_group
        # 3指输入图片的3通道， self.conv1是预处理模块，self.in_planes才是真正的backbone入口。 backbone输入的通道数
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes, eps=1e-05)
        self.prelu = nn.PReLU(self.in_planes)

        # 以下的block指残差模块，layers指每个残差模块重复的次数
        self.layer1 = self._maker_layer(block, 64, layers[0], stride=(2, 2))
        self.layer2 = self._maker_layer(block, 128, layers[1], stride=(2, 2), dilate=replace_stride_with_dilation[0])
        self.layer3 = self._maker_layer(block, 256, layers[2], stride=(2, 2), dilate=replace_stride_with_dilation[1])
        self.layer4 = self._maker_layer(block, 512, layers[3], stride=(2, 2), dilate=replace_stride_with_dilation[2])

        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)  # 对self.fc获得的1维tensor做归一化
        nn.init.constant_(self.features.weight, 1.0)  # todo 初始化权重？
        self.features.weight.requires_grad = False  # todo 为何要不更新梯度。

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _maker_layer(self, block, planes, block_num, stride=(1, 1), dilate=False):
        down_sample = None
        # previous_dilation = self.dilation
        if dilate:  # 是否是空洞卷积
            self.dilation *= stride
            stride = 1

        new_channels = planes * block.expansion
        if stride != 1 or self.in_planes != new_channels:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_planes, new_channels, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(new_channels)
            )

        layers = [block(self.in_planes, planes, stride, down_sample)]

        self.in_planes = new_channels
        for _ in range(1, block_num):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.prelu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)

        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x


def iresnet18(**kwargs):
    model = IResNet(IBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def iresnet34(**kwargs):
    model = IResNet(IBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def iresnet50(**kwargs):
    model = IResNet(IBasicBlock, [3, 4, 14, 3], **kwargs)
    return model


if __name__ == '__main__':
    from torchsummary import summary

    model = iresnet50(dropout=0.0, fp16=False, num_features=512)
    print(type(model))
    # summary(model, (3, 112, 112))
    dic=model.state_dict()
    for k,v in dic.items():
        print(k)