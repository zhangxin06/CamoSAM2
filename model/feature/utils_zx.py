import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from einops.layers.torch import Rearrange
import time


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class resblock(nn.Module):
    def __init__(self, dim):
        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(PromptGenBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size)) # (1,5,320,16,16)
        self.linear_layer = nn.Linear(lin_dim, prompt_len) # 384->5
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False) # 320->320

    def forward(self, x): # (bs,384,16,16)
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1)) # (bs,384)
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1) # (bs,5)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B, 1,
                                                                                                                  1, 1,
                                                                                                                  1,
                                                                                                                  1).squeeze(
            1) # (bs,5,320,16,16)
        prompt = torch.sum(prompt, dim=1) # (bs,320,16,16)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear") # (bs,320,16,16)
        prompt = self.conv3x3(prompt) # (bs,320,16,16)

        return prompt


##########################################################################
##---------- PromptIR -----------------------

class PromptIR(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 decoder=False,
                 ):

        super(PromptIR, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.decoder = decoder

        if self.decoder:
            self.prompt1 = PromptGenBlock(prompt_dim=64, prompt_len=5, prompt_size=64, lin_dim=96)
            self.prompt2 = PromptGenBlock(prompt_dim=128, prompt_len=5, prompt_size=32, lin_dim=192)
            self.prompt3 = PromptGenBlock(prompt_dim=320, prompt_len=5, prompt_size=16, lin_dim=384)

        self.chnl_reduce1 = nn.Conv2d(64, 64, kernel_size=1, bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128, 128, kernel_size=1, bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320, 256, kernel_size=1, bias=bias)

        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64, dim, kernel_size=1, bias=bias)
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2

        self.reduce_noise_channel_2 = nn.Conv2d(int(dim * 2 ** 1) + 128, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

        self.reduce_noise_channel_3 = nn.Conv2d(int(dim * 2 ** 2) + 256, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 2))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 1) + 192, int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(dim * 2 ** 2) + 512, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim * 2 ** 2) + 512, int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(dim * 2 ** 1) + 224, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim * 2 ** 1) + 224, int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.noise_level1 = TransformerBlock(dim=int(dim * 2 ** 1) + 64, num_heads=heads[2],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim * 2 ** 1) + 64, int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, noise_emb=None): # (1,3,128,128)

        inp_enc_level1 = self.patch_embed(inp_img) # (bs,48,128,128)

        out_enc_level1 = self.encoder_level1(inp_enc_level1) # (bs,48,128,128)

        inp_enc_level2 = self.down1_2(out_enc_level1) # (bs,96,64,64)

        out_enc_level2 = self.encoder_level2(inp_enc_level2) # (bs,96,64,64)

        inp_enc_level3 = self.down2_3(out_enc_level2) # (bs,192,32,32)

        out_enc_level3 = self.encoder_level3(inp_enc_level3) # (bs,192,32,32)

        inp_enc_level4 = self.down3_4(out_enc_level3) # (bs,384,16,16)
        latent = self.latent(inp_enc_level4) # (bs,384,16,16)
        if self.decoder:
            dec3_param = self.prompt3(latent) # (bs,320,16,16)

            latent = torch.cat([latent, dec3_param], 1) # (bs,704,16,16)
            latent = self.noise_level3(latent) # (bs,704,16,16)  
            latent = self.reduce_noise_level3(latent) # (bs,192,16,16) 

        inp_dec_level3 = self.up4_3(latent) # (1,96,32,32)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1) # (1,288,32,32)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) # (1,192,32,32)

        out_dec_level3 = self.decoder_level3(inp_dec_level3) # (1,192,32,32)
        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3) # (1,128,32,32)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1) # (1,320,32,32)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2) # (1,96,64,64)
        if self.decoder:
            dec1_param = self.prompt1(out_dec_level2) # (1,64,64,64)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1) # (1,160,64,64)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2) # (1,96,64,64)

        inp_dec_level1 = self.up2_1(out_dec_level2) # (1,48,128,128)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1) # (1,96,128,128)

        out_dec_level1 = self.decoder_level1(inp_dec_level1) # (1,96,128,128)

        out_dec_level1 = self.refinement(out_dec_level1) # (1,96,128,128)

        out_dec_level1 = self.output(out_dec_level1) + inp_img # (1,3,128,128)

        return out_dec_level1

class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
class Decoder_zx(nn.Module):
    def __init__(self, channel):  # 32
        super(Decoder_zx, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample2 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample3 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample4 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample5 = ConvBR(2 * channel, 2 * channel, 3, padding=1)
        self.conv_upsample6 = ConvBR(3 * channel, 3 * channel, 3, padding=1)

        self.conv_upsample_new1 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample_new2 = ConvBR(channel, channel, 3, padding=1)

        self.conv_concat2 = ConvBR(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = ConvBR(3 * channel, 3 * channel, 3, padding=1)
        self.conv_concat4 = ConvBR(4 * channel, 4 * channel, 3, padding=1)
        self.conv4 = ConvBR(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)
        self.conv4_new = ConvBR(4 * channel, 4 * channel, 3, padding=1)
        self.conv5_new = nn.Conv2d(4 * channel, 1, 1)

    def forward(self, zt5, zt4, zt3, zt2): # (4,32,44,44),(4,32,44,44),(4,32,44,44),(4,32,88,88)对应x4,x3,x2,x1
        zt5_1 = zt5
        zt4_1 = self.conv_upsample1(zt5) * zt4
        zt3_1 = self.conv_upsample2(zt4_1) * self.conv_upsample3(zt4) * zt3
        # zt2_1 = self.conv_upsample_new1(self.upsample(zt3_1)) * self.conv_upsample_new2(self.upsample(zt3)) * zt2

        zt4_2 = torch.cat((zt4_1, self.conv_upsample4(zt5_1)), 1)
        zt4_2 = self.conv_concat2(zt4_2) # (bs,64,44,44)

        zt3_2 = torch.cat((zt3_1, self.conv_upsample5(zt4_2)), 1)
        zt3_2 = self.conv_concat3(zt3_2) # (bs,96,44,44)

        # zt2_2 = torch.cat((zt2_1, self.conv_upsample6(self.upsample(zt3_2))), 1)
        # zt2_2 = self.conv_concat4(zt2_2) # (bs,128,88,88)

        pc = self.conv4(zt3_2) # (bs,96,44,44)
        pc = self.conv5(pc) # (bs,1,44,44)

        res = F.interpolate(pc, scale_factor=8, mode='bilinear')
        return res

if __name__ == '__main__':
    img = torch.randn(6,3,128,128)
    # model = PromptIR(decoder=True)
    zt5 = torch.randn(1,32,44,44)
    zt4 = torch.randn(1,32,44,44)
    zt3 = torch.randn(1,32,44,44)
    zt2 = torch.randn(1,32,88,88)
    model = Decoder_zx(32)
    output = model(zt5,zt4,zt3,zt2)
