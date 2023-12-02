import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from hw_4.model.HifiGan.utils import init_weights, get_padding
from hw_4.utils.configs import TrainConfig as train_config


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2])))
        ])

        self.convs2 = nn.ModuleList([
            weight_norm((nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))),
            weight_norm((nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))),
            weight_norm((nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))))
        ])

        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            skip = x
            x = F.leaky_relu(x, 0.1)
            x = c1(x)
            x = F.leaky_relu(x, 0.1)
            x = c2(x)
            x = x + skip
        return x


class Generator(nn.Module):
    def __init__(self, train_config=train_config):
        super().__init__()
        up_samp_channel = train_config.upsample_channel
        self.num_kernels = len(train_config.resblock_kernel_sizes)
        self.num_upsamples = len(train_config.upsample_rates)
        self.conv_pre = weight_norm(nn.Conv1d(80, up_samp_channel, 7, 1, 3))

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        # params kernel_sizes and upsample_rates
        for i, v in zip(train_config.upsample_kernel_sizes, train_config.upsample_rates):
            self.ups.append(weight_norm(nn.ConvTranspose1d(up_samp_channel,
                                                           up_samp_channel//2,
                                                           i, v,
                                                           (i-v)//2)))
            # params res_kernel_size and dilations
            for res, dil in zip(train_config.resblock_kernel_sizes, train_config.resblock_dilation_sizes):
                self.resblocks.append(ResBlock(up_samp_channel//2, res, dil))

            up_samp_channel = up_samp_channel // 2

        self.conv_post = weight_norm(nn.Conv1d(up_samp_channel, 1, 7, 1, 3))

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            sk = self.resblocks[i * self.num_kernels](x)
            for j in range(1, self.num_kernels):
                sk += self.resblocks[i * self.num_kernels + j](x)
            x = sk / self.num_kernels

        x = torch.tanh(self.conv_post(F.leaky_relu(x, 0.1)))

        return x
