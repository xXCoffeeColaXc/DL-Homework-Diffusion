import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config
from modules import DiffusionUNet

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        return out + x

    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False) -> None:
        super(ResidualBlock, self).__init__()
        self.residual = residual

        if not mid_channels:
            mid_channels = out_channels

        self.res = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )
 

    def forward(self, x):
        if self.residual:
            res = self.res(x)
        else:
            res = x

        x = self.double_conv(x)
        x += res
        return x
    

    

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_depth, emb_dim=256) -> None:
        super(DownBlock, self).__init__()

        layers = nn.ModuleList()
        for i in range(block_depth):
            if i == 0:
                layers.append(ResidualBlock(in_channels, out_channels, residual=True))
            else:
                layers.append(ResidualBlock(out_channels, out_channels, residual=False))

        self.residual_blocks = layers
        self.downsample = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        skip_outputs = []
        for residual_block in self.residual_blocks:
            x = residual_block(x)
            skip_outputs.append(x)
        x = self.downsample(x)
        return x, skip_outputs
    

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, block_depth, emb_dim=256) -> None:
        super(UpBlock, self).__init__()
    
        layers = nn.ModuleList()
        for i in range(block_depth):
            if i == 0:
                layers.append(ResidualBlock(in_channels + skip_channels, out_channels, residual=True))
            else:
                layers.append(ResidualBlock(out_channels + skip_channels, out_channels, residual=True))

        self.residual_blocks = layers
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
   
    def forward(self, x, skip_inputs):
        x = self.upsample(x)
        for residual_block in self.residual_blocks:
            x = torch.cat([x, skip_inputs.pop()], dim=1)
            x = residual_block(x)
        return x
    
# features=[64, 128, 256, 512, 1024]
class UNet(DiffusionUNet):
    def __init__(self, c_in=3, c_out=3, image_size=64, conv_dim=64, block_depth=3, time_emb_dim=256) -> None:
        super(UNet, self).__init__()
        self.requires_alpha_hat_timestep = True

        # base_channel_size = 32
        # down_up_block_number = 3
        # channel_sizes = [i * base_channel_size for i in range(1, block_depth + 1)]
        channel_sizes = [32, 64, 96, 128]
        self.pre_conv = nn.Conv2d(c_in, 32, kernel_size=3, padding=1, bias=False)
        self.embedding_upsample = nn.Upsample(size=(image_size, image_size), mode='nearest')

        self.attn_down1 = SelfAttention(64)
        self.down1 = DownBlock(64, 32, block_depth)
        self.attn_down2 = SelfAttention(32)
        self.down2 = DownBlock(32, 64, block_depth)
        self.attn_down3 = SelfAttention(64)
        self.down3 = DownBlock(64, 96, block_depth)

        self.bottleneck1 = ResidualBlock(96, 128, residual=True)
        self.bottleneck2 = ResidualBlock(128, 128, residual=False)

        self.up1 = UpBlock(128, 96, 96, block_depth)
        self.attn_up1 = SelfAttention(96)
        self.up2 = UpBlock(96, 64, 64, block_depth)
        self.attn_up2 = SelfAttention(64)
        self.up3 = UpBlock(64, 32, 32, block_depth)
        self.attn_up3 = SelfAttention(32)

        self.output = nn.Conv2d(32, c_out, kernel_size=3, padding=1, bias=False)

        # DownBlock:  3, 32     ------>                     # UpBlock: 64, 32, + 32
            # DownBlock: 32, 64     ------>             # UpBlock: 96, 64, + 64
                # DownBlock: 64, 96     ------>     # UpBlock: 128, 96, + 96

                                    # Bottleneck: 96, 128
                                    # Bottleneck: 128, 128


    def sinusoidal_embedding(self, x):
        embedding_min_frequency = 1.0
        embedding_max_frequency = 1000.0
        embedding_dims = 32
        frequencies = torch.exp(
            torch.linspace(
                math.log(embedding_min_frequency),
                math.log(embedding_max_frequency),
                embedding_dims // 2,
            )
        ).to(x.device)
        angular_speeds = 2.0 * math.pi * frequencies
        sin_part = torch.sin(angular_speeds * x)
        cos_part = torch.cos(angular_speeds * x)
        embeddings = torch.cat([sin_part, cos_part], dim=3).permute(0, 3, 1, 2)
        return embeddings

    def forward(self, x, t):
        x = self.pre_conv(x)
        t = self.sinusoidal_embedding(t)
        t = self.embedding_upsample(t)
        x = torch.cat([x, t], dim=1)

        # Downward path
        x = self.attn_down1(x)
        x, skip1 = self.down1(x)
        x = self.attn_down2(x) 
        x, skip2 = self.down2(x)
        x = self.attn_down3(x)  
        x, skip3 = self.down3(x)

        # Bottleneck
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)

        # Upward path
        x = self.up1(x, skip3)
        x = self.attn_up1(x)  
        x = self.up2(x, skip2)
        x = self.attn_up2(x)  
        x = self.up3(x, skip1)
        x = self.attn_up3(x)  

        output = self.output(x)

        return output




if __name__ == '__main__':
    net = UNet(block_depth=2)
    net = net.to('cpu')
    print(sum([p.numel() for p in net.parameters()]))
    print(net)
    x = torch.randn(2, 3, 64, 64)
    t = torch.tensor([[[[0.2860]]],[[[0.2860]]]])
    x = x.to('cpu')
    t = t.to('cpu')
    pred = net(x, t) 
    print(pred.shape)










