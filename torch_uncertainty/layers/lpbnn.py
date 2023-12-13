import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class LpbnnLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_estimators: int,
        hidden_size: int = 32,
        bias=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(num_estimators, in_features))
        self.gamma = nn.Parameter(torch.Tensor(num_estimators, out_features))
        self.encoder_fc1 = nn.Linear(in_features, self.hidden_size)
        self.encoder_fcmean = nn.Linear(self.hidden_size, self.hidden_size)
        self.encoder_fcvar = nn.Linear(self.hidden_size, self.hidden_size)
        self.decoder_fc1 = nn.Linear(self.hidden_size, in_features)
        self.loss_latent = 0
        self.num_estimators = num_estimators
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.num_estimators, out_features)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.alpha, mean=1.0, std=0.5)
        nn.init.normal_(self.gamma, mean=1.0, std=0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def forward(self, x):
        embedded = F.relu(self.encoder_fc1(self.alpha))
        embedded_mean, embedded_logvar = (
            self.encoder_fcmean(embedded),
            self.encoder_fcvar(embedded),
        )
        z_embedded = self.reparameterize(embedded_mean, embedded_logvar)
        alpha_decoded = self.decoder_fc1(z_embedded)
        if self.training:
            mse = F.mse_loss(alpha_decoded, self.alpha)
            kld = -0.5 * torch.sum(
                1
                + embedded_logvar
                - embedded_mean.pow(2)
                - embedded_logvar.exp()
            )
            self.loss_latent = mse + kld

        num_examples_per_model = int(x.size(0) / self.num_estimators)
        # Repeated pattern: [[A,A],[B,B],[C,C]]
        alpha = torch.cat(
            [alpha_decoded for i in range(num_examples_per_model)], dim=1
        ).view([-1, self.in_features])
        gamma = torch.cat(
            [self.gamma for i in range(num_examples_per_model)], dim=1
        ).view([-1, self.out_features])
        out = self.fc(x * alpha) * gamma

        if self.bias is not None:
            bias = torch.cat(
                [self.bias for i in range(num_examples_per_model)], dim=1
            ).view([-1, self.out_features])
            out += bias
        return out


class LpbnnConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_estimators: int,
        stride=1,
        padding=0,
        groups=1,
        hidden_size=32,
        train_gamma=True,
        bias=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_estimators = num_estimators
        self.train_gamma = train_gamma

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.alpha = nn.Parameter(
            torch.Tensor(num_estimators, in_channels), requires_grad=False
        )

        self.encoder_fc1 = nn.Linear(in_channels, self.hidden_size)
        self.decoder_fc1 = nn.Linear(self.hidden_size, in_channels)

        self.encoder_fcmean = nn.Linear(self.hidden_size, self.hidden_size)
        self.encoder_fcvar = nn.Linear(self.hidden_size, self.hidden_size)

        self.loss_latent = torch.zeros(1)
        if train_gamma:
            self.gamma = nn.Parameter(
                torch.Tensor(num_estimators, out_channels)
            )
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.num_estimators, out_channels)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.alpha, mean=1.0, std=0.5)
        if self.train_gamma:
            nn.init.normal_(self.gamma, mean=1.0, std=0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x):
        embedded = F.relu(self.encoder_fc1(self.alpha))
        embedded_mean, embedded_logvar = (
            self.encoder_fcmean(embedded),
            self.encoder_fcvar(embedded),
        )
        z_embedded = self._reparameterize(embedded_mean, embedded_logvar)
        alpha_decoded = self.decoder_fc1(z_embedded)

        alpha_decoded = self.decoder_fc1(embedded.view(len(self.alpha), -1))

        mse = F.mse_loss(alpha_decoded, self.alpha)
        kld = -0.5 * torch.sum(
            1 + embedded_logvar - embedded_mean.pow(2) - embedded_logvar.exp()
        )
        self.loss_latent = mse + kld

        if self.train_gamma:
            num_examples_per_model = int(x.size(0) / self.num_estimators)

            alpha = torch.cat(
                [alpha_decoded for _ in range(num_examples_per_model)], dim=1
            ).view([-1, self.in_channels])
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)
            gamma = torch.cat(
                [self.gamma for _ in range(num_examples_per_model)], dim=1
            ).view([-1, self.out_channels])
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            out = self.conv(x * alpha) * gamma

            if self.bias is not None:
                bias = torch.cat(
                    [self.bias for i in range(num_examples_per_model)], dim=1
                ).view([-1, self.out_channels])
                bias = bias.unsqueeze(-1).unsqueeze(-1)
                out += bias

            return out

        # else
        num_examples_per_model = int(x.size(0) / self.num_estimators)
        # Repeated pattern: [[A,A],[B,B],[C,C]]
        alpha = torch.cat(
            [alpha_decoded for i in range(num_examples_per_model)], dim=1
        ).view([-1, self.in_channels])
        alpha = alpha.unsqueeze(-1).unsqueeze(-1)
        out = self.conv(x * alpha)

        if self.bias is not None:
            bias = torch.cat(
                [self.bias for _ in range(num_examples_per_model)], dim=1
            ).view([-1, self.out_channels])
            bias = bias.unsqueeze(-1).unsqueeze(-1)
            out += bias
        return out
