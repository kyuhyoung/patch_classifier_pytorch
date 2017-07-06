import torch.nn as nn
import torch


class Aidentity(nn.Module):
    def __init__(self):
        return
    def forward(self, x):
        return x

class Bottleneck(nn.Module):
    expansion = 4

    #def __init__(self, inplanes, planes, stride=1, downsample=None):
    def __init__(self, n, stride, downsample=None):
        super(Bottleneck, self).__init__()
        m = n * stride
        #self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(n, n, kernel_size=1)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.BatchNorm2d(n)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(n, n, kernel_size=3, stride=stride, padding=1)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(n)
        #self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(n, m, kernel_size=1)
        #self.bn3 = nn.BatchNorm2d(planes * 4)
        self.bn3 = nn.BatchNorm2d(m)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def shortcut(nInputPlane, nOutputPlane, stride, shortcutType):
    useConv = shortcutType == 'C' or \
              (shortcutType == 'B' and nInputPlane != nOutputPlane)
    if useConv:
        # 1x1 convolution
        s = nn.Sequential()
        s.add_module(nn.Conv2d(nInputPlane, nOutputPlane, kernel_size=1, stride=stride))
        s.add_module(nn.BatchNorm2d(nOutputPlane))
        #return s
    elif nInputPlane != nInputPlane:
        # Strided, zero - padded identity shortcut
        s = nn.Sequential()
        #:add(nn.SpatialAveragePooling(1, 1, stride, stride))
        s.add_module(nn.AvgPool2d(7))
        #:add(nn.Concat(2)
        s.add_module(torch.cat(nn.Identity(), nn.MulConstant(0)))
        return s
    else:
        return torch.legacy.nn.Identity()

def sum_and_relu(residual, n, m, stride, shortcut_type):
    s = nn.Sequential()
    s.add_module(residual + shortcut(n, m, stride, shortcut_type))
    s.add_module(nn.ReLU(inplace=True))
    return s


def bottleneck(n, stride, shortcut_type):

    m = n * stride
    r = nn.Sequential()

    r.add_module('c1', nn.Conv2d(n, n, kernel_size=1))
    r.add_module('b1', nn.BatchNorm2d(n))
    r.add_module('a1', nn.ReLU(inplace=True))

    r.add_module('c2', nn.Conv2d(n, n, kernel_size=3, stride=stride, padding=1))
    r.add_module('b2', nn.BatchNorm2d(n))
    r.add_module('a2', nn.ReLU(inplace=True))

    r.add_module('c3', nn.Conv2d(n, m, kernel_size=1))
    r.add_module('b3', nn.BatchNorm2d(n))

    return sum_and_relu(r, n, m, stride, shortcut_type)


def two_way(n, stride, shortcut_type):
    m = n * stride
    r = nn.Sequential()
    r.add_module(bottleneck(n, 1, shortcut_type))
    r.add_module(bottleneck(n, stride, shortcut_type))
    return sum_and_relu(r, n, m, stride, shortcut_type)



def four_way(n, stride, shortcut_type):
    m = n * stride
    r = nn.Sequential()
    r.add_module(two_way(n, 1, shortcut_type))
    r.add_module(two_way(n, stride, shortcut_type))
    return sum_and_relu(r, n, m, stride, shortcut_type)


def eight_way(n, stride, shortcut_type):
    m = n * stride
    r = nn.Sequential()
    r.add_module(four_way(n, 1, shortcut_type))
    r.add_module(four_way(n, stride, shortcut_type))
    return sum_and_relu(r, n, m, stride, shortcut_type)


class Ellie(nn.Module):

    #def __init__(self, block, layers, num_classes=1000):
    def __init__(self, shortcut_type):
        #block = Bottleneck
        layers = [16, 32, 64, 128, 256]
        block = bottleneck
        num_classes = 8
        self.inplanes = 64
        super(Ellie, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer = nn.Sequential()
        for l in layers:
            self.layer.add_module(self._make_layer(l, 1, 2, shortcut_type))

        self.layer16 = self._make_layer(16, 1, 2)
        self.layer32 = self._make_layer(16, 1, 2)
        self.layer64 = self._make_layer(16, 1, 2)
        self.layer128 = self._make_layer(16, 1, 2)
        self.layer256 = self._make_layer(16, 1, 2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    '''
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    '''


    def _make_layer(self, features, count, stride, shortcut_type):

        s = nn.Sequential()
        for i in range(count - 1):
            s.add_module(eight_way(features, 1, shortcut_type))
        s.add_module(eight_way(features, stride, shortcut_type))
        return s
    '''
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    '''
    def forward(self, x_input):

        x_small = self.conv1(x_input)
        x_small = self.bn1(x_small)
        x_small = self.relu(x_small)
        x_small = self.maxpool(x_small)

        x_small = self.layer1(x_small)
        x_small = self.layer2(x_small)
        x_small = self.layer3(x_small)
        x_small = self.layer4(x_small)

        x_small = self.avgpool(x_small)
        x_small = x_small.view(x_small.size(0), -1)
        x_small = self.fc(x_small)

        x_large = self.conv1(x_input)
        x_large = self.bn1(x_large)
        x_large = self.relu(x_large)
        x_large = self.maxpool(x_large)

        x_large = self.layer1(x_large)
        x_large = self.layer2(x_large)
        x_large = self.layer3(x_large)
        x_large = self.layer4(x_large)

        x_large = self.avgpool(x_large)
        x_large = x.view(x_large.size(0), -1)
        x_large = self.fc(x_large)

        x = torch.cat(x_small, x_large)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(2048)
        x = self.fc(x)
        x = nn.Sigmoid()(x)

        return x
