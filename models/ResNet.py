import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_dim = 3, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def base_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def base_forward_feature(self, x,layer):
        out = F.relu(self.bn1(self.conv1(x)))

        if layer == 1:
            return self.layer1(out)
        out = self.layer1(out)

        if layer == 2:
            return self.layer2(out)

        out = self.layer2(out)

        if layer == 3:
            return self.layer3(out)
        out = self.layer3(out)

        if layer == 4:
            return self.layer4(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def continue_forward_feature(self, x,layer):
        #out = F.relu(self.bn1(self.conv1(x)))
        if layer == 1:
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
        elif layer == 2:
            out = self.layer2(x)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
        elif layer == 3:
            out = self.layer3(x)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
        elif layer == 4:
            out = self.layer4(x)
            out = F.avg_pool2d(out, 4)
        elif layer == 5:
            out = F.avg_pool2d(x, 4)


        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_feature_maps(self, x):
        feature_maps = {}
        out = F.relu(self.bn1(self.conv1(x)))
        feature_maps['conv1'] = out
        out = self.layer1(out)
        feature_maps['layer1'] = out
        out = self.layer2(out)
        feature_maps['layer2'] = out
        out = self.layer3(out)
        feature_maps['layer3'] = out
        out = self.layer4(out)
        feature_maps['layer4'] = out
        return feature_maps

    def forward(self, x, require_logits = False):
        logits = self.base_forward(x)
        if require_logits:
            return F.log_softmax(logits, dim = 1), logits
        return F.log_softmax(logits, dim = 1)
    
    def forward_w_temperature(self, x, T=1):
        logits = self.base_forward(x)
        scaled_logits = logits/T
        return F.softmax(scaled_logits, dim=1)


def ResNet18(in_dim = 3, num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_dim = in_dim, num_classes=num_classes)

def ResNet9(in_dim = 1, num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1, 1], in_dim = in_dim, num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)



