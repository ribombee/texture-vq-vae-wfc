import torch
from torch import nn
from torch.nn import functional as F
from itertools import chain
from math import log

# Gated convolution layers are from
# https://github.com/avalonstrel/GatedConvolution_pytorch


class GatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="same", dilation=1, groups=1,
                 bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class GatedDeConv2dWithActivation(torch.nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding="same", dilation=1,
                 groups=1, bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedDeConv2dWithActivation, self).__init__()
        self.conv2d = GatedConv2dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                                groups, bias, batch_norm, activation)
        self.scale_factor = scale_factor

    def forward(self, input):
        #print(input.size())
        x = F.interpolate(input, scale_factor=self.scale_factor)
        return self.conv2d(x)


# VQ-VAE2 implementation Borrowed mostly from this implementation of the vq-vae-2 model
# https://github.com/rosinality/vq-vae-2-pytorch/blob/master/vqvae.py

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel, gated=False):
        super().__init__()

        if gated:
            self.conv = nn.Sequential(

                nn.LeakyReLU(),
                GatedConv2dWithActivation(in_channels=in_channel, out_channels=channel, kernel_size=3, padding="same"),
                GatedConv2dWithActivation(in_channels=channel, out_channels=in_channel, kernel_size=1)
            )
        else:
            self.conv = nn.Sequential(

                nn.LeakyReLU(),
                nn.Conv2d(in_channel, channel, 3, padding="same"),
                nn.LeakyReLU(),
                nn.Conv2d(channel, in_channel, 1)
            )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride, compress_factor, kernel_size,
                 gated=False):
        super().__init__()

        if gated:
            conv = GatedConv2dWithActivation
        else:
            conv = nn.Conv2d

        num_striding = int(log(compress_factor, stride))

        blocks = [
            conv(in_channels=in_channel, stride=1, out_channels=channel // (2 ** num_striding), kernel_size=kernel_size,
                 padding="same"), nn.LeakyReLU(0.2)]

        shrink_blocks = [[conv(in_channels=channel // (2 ** (num_striding - stride_idx)), stride=1,
                               out_channels=channel // (2 ** (num_striding - stride_idx - 1)),
                               kernel_size=kernel_size, padding="same"),
                          nn.LeakyReLU(0.2),
                          nn.MaxPool2d(stride),
                          conv(in_channels=channel // (2 ** (num_striding - stride_idx - 1)),
                               out_channels=channel // (2 ** (num_striding - stride_idx - 1)), kernel_size=1,
                               padding="same"),
                          nn.LeakyReLU(0.2)] for stride_idx in range(num_striding)]

        blocks.extend(list(chain(*shrink_blocks)))

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.LeakyReLU(0.2, inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
            self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride, compress_factor, kernel_size,
            gated=False
    ):
        super().__init__()

        if gated:
            conv = GatedConv2dWithActivation
        else:
            conv = nn.Conv2d

        num_striding = int(log(compress_factor, stride))
        blocks = [conv(in_channel, channel, 3, padding="same")]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel, gated))

        if gated:

            [blocks.extend([
                nn.LeakyReLU(0.2, inplace=True),
                GatedDeConv2dWithActivation(scale_factor=stride,
                                            in_channels=channel // (2 ** (stride_idx)),
                                            out_channels=channel // (2 ** (stride_idx + 1)),
                                            kernel_size=kernel_size)]) for stride_idx in range(num_striding)]
        else:

            [blocks.extend([nn.LeakyReLU(0.2, inplace=True),
                            nn.ConvTranspose2d(stride=stride,
                                               in_channels=channel // (2 ** stride_idx),
                                               out_channels=channel // (2 ** (stride_idx + 1)),
                                               kernel_size=kernel_size, padding=kernel_size // 2, output_padding=1)

                            ]) for stride_idx in range(num_striding)]
        blocks.append(
            nn.Conv2d(channel // (2 ** num_striding), out_channel, kernel_size=kernel_size, padding="same"))
        blocks.append(nn.Tanh())

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class VQVAE(nn.Module):
    def __init__(
            self,
            conf,
            in_channel=3,
            gated=True
    ):
        super().__init__()
        self.conf = conf

        channel = conf.model.channel
        embed_dim = conf.model.embed_dim
        n_res_block = conf.model.num_res_blocks
        n_res_channel = conf.model.num_res_channel
        n_embed = conf.model.codebook_size
        self.compress_factor = conf.model.compress_factor
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=self.conf.model.stride,
                             compress_factor=conf.model.compress_factor, kernel_size=conf.model.kernel_size,
                             gated=gated)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2,
            compress_factor=conf.model.compress_factor, kernel_size=conf.model.kernel_size, gated=gated
        )
        self.stride = conf.model.stride
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.Upsample(scale_factor=2)
        self.upsample_t_conv = nn.ConvTranspose2d(
            embed_dim, embed_dim, 3, padding=1
        )
        self.dec = Decoder(
            embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=conf.model.stride,
            compress_factor=conf.model.compress_factor,
            kernel_size=conf.model.kernel_size,
            gated=gated
        )

    def forward(self, input):
        quant, diff, id = self.encode(input)
        dec = self.decode(quant, None)
        return dec, diff, id

    def encode(self, input):
        enc = self.enc_b(input)
        quant = self.quantize_conv_t(enc).permute(0, 2, 3, 1)
        quant, diff, id = self.quantize_t(quant)
        quant = quant.permute(0, 3, 1, 2)
        diff = diff.unsqueeze(0)
        return quant, diff, id

    def decode(self, quant_t, quant_b):
        dec = self.dec(quant_t)
        return dec

    def decode_code(self, code_t, code_b):
        quant = self.quantize_t.embed_code(code_t)
        quant = quant.permute(0, 3, 1, 2)
        dec = self.decode(quant, None)
        return dec