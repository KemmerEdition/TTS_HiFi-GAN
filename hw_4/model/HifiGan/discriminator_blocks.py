import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from hw_4.model.HifiGan.utils import get_padding


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), (get_padding(5, 1), 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), (get_padding(5, 1), 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), (get_padding(5, 1), 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), (get_padding(5, 1), 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, (2, 0)))
        ])

        self.convs.append(weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0))))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape

        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
        x = x.view(b, c, -1, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = torch.flatten(x, 1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11)
        ])

    def forward(self, y, y_hat):
        y_outs, y_hat_outs = [], []
        y_feats, y_hat_feats = [], []
        for i in self.discriminators:
            y_out, y_feat = i(y)
            y_hat_out, y_hat_feat = i(y_hat)
            y_outs.append(y_out)
            y_hat_outs.append(y_hat_out)
            y_feats.append(y_feat)
            y_hat_feats.append(y_hat_feat)
        return y_outs, y_hat_outs, y_feats, y_hat_feats


class DiscriminatorS(nn.Module):
    def __init__(self, norm=weight_norm):
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            norm(nn.Conv1d(1, 128, 15, 1, 7)),
            norm(nn.Conv1d(128, 128, 41, 2, 20, groups=4)),
            norm(nn.Conv1d(128, 256, 41, 2, 20, groups=16)),
            norm(nn.Conv1d(256, 512, 41, 4, 20, groups=16)),
            norm(nn.Conv1d(512, 1024, 41, 4, 20, groups=16)),
            norm(nn.Conv1d(1024, 1024, 41, 1, 20, groups=16)),
            norm(nn.Conv1d(1024, 1024, 5, 1, 2))
        ])

        self.conv_blocks.append(weight_norm(nn.Conv1d(1024, 1, 3, 1, 1)))

    def forward(self, x):
        fmap = []
        for l in self.conv_blocks:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = torch.flatten(x, 1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(spectral_norm),
            DiscriminatorS(),
            DiscriminatorS()
        ])
        self.pools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        i = 0
        pools = len(self.pools)
        y_outs, y_hat_outs = [], []
        y_feats, y_hat_feats = [], []
        for j in self.discriminators:
            y_out, y_feat = j(y)
            y_hat_out, y_hat_feat = j(y_hat)
            y_outs.append(y_out)
            y_hat_outs.append(y_hat_out)
            y_feats.append(y_feat)
            y_hat_feats.append(y_hat_feat)
            if i < pools:
                y = self.pools[i](y)
                y_hat = self.pools[i](y_hat)
            i += 1
        return y_outs, y_hat_outs, y_feats, y_hat_feats
