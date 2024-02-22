import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CompressBNNFC(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.fc(x)


class BatchNormAdapter2d(nn.Module):
    def __init__(
        self,
        num_features: int,
        alpha: float = 0.01,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.running_mean = nn.Parameter(
            torch.zeros(num_features, device=device, dtype=dtype),
            requires_grad=False,
        )
        self.running_var = nn.Parameter(
            torch.ones(num_features, device=device, dtype=dtype),
            requires_grad=False,
        )
        self.weight = nn.Parameter(
            torch.ones(num_features, device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(
            torch.zeros(num_features, device=device, dtype=dtype)
        )
        self.num_batches_tracked = nn.Parameter(
            torch.tensor(0, dtype=torch.long, device=device),
            requires_grad=False,
        )
        self.alpha = alpha
        self.training = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            None,
            None,
            self.training,
            0.1,
            1e-5,
        )
        return self.weight.unsqueeze(-1).unsqueeze(-1) * out * (
            torch.randn_like(x) * self.alpha + 1
        ) + self.bias.unsqueeze(-1).unsqueeze(-1)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1)


class WideBasicCompressbnn(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, alpha, stride=1):
        super().__init__()
        self.bn1 = BatchNormAdapter2d(in_planes, alpha=alpha)

        self.conv1 = nn.Conv2d(
            in_planes, planes, 3, stride=1, padding=1, bias=True
        )

        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = BatchNormAdapter2d(planes, alpha=alpha)

        self.conv2 = nn.Conv2d(
            planes, planes, 3, stride=stride, padding=1, bias=True
        )
        self.bn3 = BatchNormAdapter2d(planes, alpha=alpha)

        self.convs = [self.conv1, self.conv2]
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes, kernel_size=1, stride=stride, bias=True
                ),
                BatchNormAdapter2d(planes, alpha=alpha),
            )

    def update_indices(self, indices):
        for m_conv in self.convs:
            m_conv.update_indices(indices)

    def forward(self, x_dico):
        x = x_dico["out"]

        sample = x_dico["sample"]
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.bn3(self.conv2(F.relu(self.bn2(out))))
        out += self.shortcut(x)

        return {"out": out, "sample": sample}


class WideResNetCompressBNN(nn.Module):
    def __init__(
        self, depth, widen_factor, dropout_rate, num_classes, alpha=0.01
    ):
        super().__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) / 6
        k = widen_factor

        print("| Wide-Resnet %dx%d" % (depth, k))
        n_stages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, n_stages[0], stride=1)

        self.layer1 = self._wide_layer(
            WideBasicCompressbnn, n_stages[1], n, dropout_rate, alpha, stride=1
        )
        self.layer2 = self._wide_layer(
            WideBasicCompressbnn, n_stages[2], n, dropout_rate, alpha, stride=2
        )
        self.layer3 = self._wide_layer(
            WideBasicCompressbnn, n_stages[3], n, dropout_rate, alpha, stride=2
        )

        self.bn1 = BatchNormAdapter2d(n_stages[3], alpha=alpha)

        self.linear = CompressBNNFC(n_stages[3], num_classes)
        self.num_classes = num_classes

    def _wide_layer(
        self, block, planes, num_blocks, dropout_rate, alpha, stride
    ):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(
                block(self.in_planes, planes, dropout_rate, alpha, stride)
            )
            self.in_planes = planes

        return nn.Sequential(*layers)  # return nn.Sequential(*layers())

    def forward(self, x, sample=False):
        out = self.conv1(x)

        out = self.layer1({"out": out, "sample": sample})
        out = self.layer2(out)
        out = self.layer3(out)
        out = out["out"]
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)

        # out=F.softmax(out, dim=1)
        return self.linear(out)


if __name__ == "__main__":
    net = WideResNetCompressBNN(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1, 3, 32, 32)))

    print(y.size())
