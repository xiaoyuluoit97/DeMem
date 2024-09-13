import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义自定义激活函数
class NoisyReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, std):
        relu_output = F.relu(input)
        noise = torch.randn_like(relu_output) * std  # 调整噪声的方差
        ctx.save_for_backward(input, noise)
        ctx.std = std
        return relu_output + noise

    @staticmethod
    def backward(ctx, grad_output):
        input, noise = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0  # ReLU's gradient is 0 for input <= 0
        return grad_input, None

class NoisyReLU(nn.Module):
    def __init__(self, std=1.0):
        super(NoisyReLU, self).__init__()
        self.std = std

    def forward(self, input):
        return NoisyReLUFunction.apply(input, self.std)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, std=1.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.noisy_relu = NoisyReLU(std)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.noisy_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.noisy_relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, std=1.0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.noisy_relu = NoisyReLU(std)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.noisy_relu(self.bn1(self.conv1(x)))
        out = self.noisy_relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.noisy_relu(out)
        return out


class _ResNetCustomized(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, std=1.0):
        super(_ResNetCustomized, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, std=std)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, std=std)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, std=std)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, std=std)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.noisy_relu = NoisyReLU(std)

    def _make_layer(self, block, planes, num_blocks, stride, std):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, std=std))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.noisy_relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNetCustomized(num, num_classes=10, std=1.0):
    if num == 18:
        return _ResNetCustomized(BasicBlock, [2, 2, 2, 2], num_classes, std=std)
    elif num == 34:
        return _ResNetCustomized(BasicBlock, [3, 4, 6, 3], num_classes, std=std)
    elif num == 50:
        return _ResNetCustomized(Bottleneck, [3, 4, 6, 3], num_classes, std=std)
    elif num == 101:
        return _ResNetCustomized(Bottleneck, [3, 4, 23, 3], num_classes, std=std)
    elif num == 152:
        return _ResNetCustomized(Bottleneck, [3, 8, 36, 3], num_classes, std=std)
    else:
        raise NotImplementedError
