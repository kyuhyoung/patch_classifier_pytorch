import torch.nn as nn
import torch
import math
import numpy as np


class Aidentity(nn.Module):
    def __init__(self):
        super(Aidentity, self).__init__()
        return
    def forward(self, x):
        return x

class Set_Naver_Zero(nn.Module):
    def __init__(self, margin):
        super(Set_Naver_Zero, self).__init__()
        self.margin = margin
        return
    def forward(self, x):
        #n_batch, n_channel, height, width = x.size()
        #x.data[:, :, :, :] = 1
        #t1 = np.count_nonzero(x.data.numpy())
        x.data[:, :, :self.margin, :] = 0
        #t2 = np.count_nonzero(x.data.numpy())
        #t3 = t1 - t2
        #t4 = self.margin * width * n_channel * n_batch
        x.data[:, :, -self.margin:, :] = 0
        #t5 = np.count_nonzero(x.data.numpy())
        #t6 = t2 - t5
        x.data[:, :, :, :self.margin] = 0
        #t7 = np.count_nonzero(x.data.numpy())
        #t8 = t6 - t7
        #t9 = t8 * 3
        x.data[:, :, :, -self.margin:] = 0
        #t10 = np.count_nonzero(x.data.numpy())
        #t11 = t7 - t10
        #t12 = t11 * 3
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
        s.add_module('sc_conv', nn.Conv2d(nInputPlane, nOutputPlane, kernel_size=1, stride=stride))
        s.add_module('sc_bn', nn.BatchNorm2d(nOutputPlane))
        return s
    elif nInputPlane != nInputPlane:
        # Strided, zero - padded identity shortcut
        s = nn.Sequential()
        #:add(nn.SpatialAveragePooling(1, 1, stride, stride))
        s.add_module('sc_ap', nn.AvgPool2d(7))
        #:add(nn.Concat(2)
        s.add_module('sc_cat', torch.cat(nn.Identity(), nn.MulConstant(0)))
        return s
    else:
        #return torch.legacy.nn.Identity()
        return Aidentity()

'''
def sum_and_relu(residual, n, m, stride, shortcut_type):
    s = nn.Sequential()
    s.add_module(residual + shortcut(n, m, stride, shortcut_type))
    s.add_module(nn.ReLU(inplace=True))
    return s
'''
class Sum_and_ReLU(nn.Module):
    def __init__(self, residual, n, m, stride, shortcut_type):
        super(Sum_and_ReLU, self).__init__()

        self.short_cut = shortcut(n, m, stride, shortcut_type)
        self.residual = residual
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        aidentity = self.short_cut(x)
        residual = self.residual(x)
        val = self.relu(aidentity + residual)
        return val


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
    r.add_module('b3', nn.BatchNorm2d(m))

    return Sum_and_ReLU(r, n, m, stride, shortcut_type)


def two_way(n, stride, shortcut_type):
    m = n * stride
    r = nn.Sequential()
    r.add_module('bottle_2_1', bottleneck(n, 1, shortcut_type))
    r.add_module('bottle_2_2', bottleneck(n, stride, shortcut_type))
    return Sum_and_ReLU(r, n, m, stride, shortcut_type)



def four_way(n, stride, shortcut_type):
    m = n * stride
    r = nn.Sequential()
    r.add_module('bottle_4_1', two_way(n, 1, shortcut_type))
    r.add_module('bottle_4_2', two_way(n, stride, shortcut_type))
    return Sum_and_ReLU(r, n, m, stride, shortcut_type)


def eight_way(n, stride, shortcut_type):
    m = n * stride
    r = nn.Sequential()
    r.add_module('bottle_8_1', four_way(n, 1, shortcut_type))
    r.add_module('bottle_8_2', four_way(n, stride, shortcut_type))
    return Sum_and_ReLU(r, n, m, stride, shortcut_type)


class Ellie(nn.Module):

    #def __init__(self, block, layers, num_classes=1000):
    def __init__(self, shortcut_type):
        super(Ellie, self).__init__()
        self.fc = nn.Linear(2048, 8)
        self.sigmoid = nn.Sigmoid()
        self.model_small = nn.Sequential(
            #nn.ZeroPad2d(-240),
            Set_Naver_Zero(240)
            , nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=3, bias=False)
            , nn.BatchNorm2d(16)
            , nn.ReLU(inplace=True)
            , self._make_layer(16, 2, 2, shortcut_type)
            , self._make_layer(32, 2, 2, shortcut_type)
            , self._make_layer(64, 2, 2, shortcut_type)
            , self._make_layer(128, 2, 2, shortcut_type)
            , self._make_layer(256, 2, 2, shortcut_type)
            , self._make_layer(512, 2, 2, shortcut_type)
            , nn.AvgPool2d(kernel_size=4, stride=1)

        )

        self.model_large = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            self._make_layer(16, 2, 2, shortcut_type),
            self._make_layer(32, 2, 2, shortcut_type),
            self._make_layer(64, 2, 2, shortcut_type),
            self._make_layer(128, 2, 2, shortcut_type),
            self._make_layer(256, 2, 2, shortcut_type),
            self._make_layer(512, 2, 2, shortcut_type),
            nn.AvgPool2d(kernel_size=3, stride=1)
        )
        '''
        layers = [16, 32, 64, 128, 256]
        block = bottleneck
        num_classes = 8
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer = nn.Sequential()
        for idx, l in enumerate(layers):
            self.layer.add_module('layer_%d' % (idx), self._make_layer(l, 1, 2, shortcut_type))

        '''
        #for m in self.modules():
        #for idx, m in enumerate(self.modules()):
        for idx, (name, m) in enumerate(self.named_modules()):
            if isinstance(m, nn.Conv2d):
                #print('%d : %s, nn.Conv2d' % (idx, name))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                #print('%d : %s, nn.BatchNorm2d' % (idx, name))
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            else:
                #print('%d : %s, else' % (idx, name))
                a = 0
        return
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
            s.add_module(str(i), eight_way(features, 1, shortcut_type))
        s.add_module('layer_8way', eight_way(features, stride, shortcut_type))
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

        x_small = self.model_small(x_input)
        x_large = self.model_large(x_input)
        x = torch.cat(x_small, x_large)
        x = x.view(2048)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
