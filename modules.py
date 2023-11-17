import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config

class SelfAttention(nn.Module):
    def __init__(self, channels, size) -> None:
        """
        Args:
            channels: Channel dimension.
            size: Current image resolution.
        """
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

# idk, could use this as well
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TimestepEmbedding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.embedding = SinusoidalPositionEmbeddings(channels)
        self.linear = nn.Linear(channels, channels)
        #self.r = nn.ReLU() # kell ?

    def forward(self, time):
        embed = self.embedding(time)
        embed = self.linear(embed)
        return embed
    
# OR ResidualBlock
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False) -> None:
        super(DoubleConv, self).__init__()
        self.residual = residual

        # NOTE this commented lines are from Beres Andras implementation
        # self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_channels) # or group=1
        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)

        # self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        # self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )


    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
        
        # NOTE this commented lines are from Beres Andras implementation
        # x = self.norm1(x)
        # x = F.silu(x)
        # x = self.conv1(x)

        # x = self.norm2(x)
        # x = F.silu(x)
        # x = self.conv2(x)

        # x += residual
        # return x
    

    

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256) -> None:
        super(DownBlock, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), # each downblock will reduce the input size by 2
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels), # bring the time embedding to the hidden dimension
        )
        # NOTE this commented lines are from Beres Andras implementation
        # layers = []
        # for i in range(block_depth):
        #     if i == 0:
        #         layers.append(ResidualBlock(in_channels, out_channels, use_attention))
        #     else:
        #         layers.append(ResidualBlock(out_channels, out_channels, use_attention))

        # self.down_blocks = nn.Sequential(*layers)

        # self.downsample = nn.AvgPool2d(kernel_size=2)
        # self.timestep_embedding = TimestepEmbedding(out_channels)

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
        # NOTE this commented lines are from Beres Andras implementation
        # time_embed = self.timestep_embedding(t)[:, :, None, None]
        # x = self.down_blocks(x) + time_embed
        # return self.downsample(x)
    

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256) -> None:
        super(UpBlock, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1) # apply skip connection
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
    # NOTE this commented lines are from Beres Andras implementation
    #     layers = []
    #     for i in range(block_depth):
    #         if i == 0:
    #             layers.append(ResidualBlock(in_channels*2, out_channels, use_attention))
    #         else:
    #             layers.append(ResidualBlock(out_channels, out_channels, use_attention))

    #     self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    #     self.up_blocks = nn.Sequential(*layers)
    #     self.timestep_embedding = TimestepEmbedding(out_channels)
   
    # def forward(self, x, t):
    #     time_embed = self.timestep_embedding(t)[:, :, None, None]
    #     x = self.up_blocks(x) + time_embed
    #     return self.upsample(x)
    
# features=[64, 128, 256, 512, 1024]
class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, image_size=64, conv_dim=64, block_depth=3, time_emb_dim=256) -> None:
        super(UNet, self).__init__()

        # NOTE Beres Andras did the same, just in a more compact form
        # Resudial block contains double conv and attention
        # Down/Up blocks contain resudial blocks
        
        self.time_dim = time_emb_dim
       
        # Encoder
        self.inc = DoubleConv(c_in, 64)

        self.down1 = DownBlock(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = DownBlock(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = DownBlock(256, 256)
        self.sa3 = SelfAttention(256, 8)

        # Bottleneck
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        # Decoder
        self.up1 = UpBlock(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = UpBlock(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = UpBlock(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)


        # TODO refactor to a more customable form, use conv_dim, image_size and block_depth
        # current_dim = conv_dim
        # current_image_size = image_size
        # # Encoder
        # self.inc = DoubleConv(c_in, current_dim)
        # self.down_blocks = nn.ModuleList()
        # self.sa_blocks_down = nn.ModuleList()

        # # Creating Down/SA Blocks Dynamically
        # for i in range(block_depth):
        #     self.down_blocks.append(DownBlock(current_dim, current_dim * 2))
        #     self.sa_blocks_down.append(SelfAttention(current_dim * 2, current_image_size // 2))
        #     current_dim *= 2
        #     current_image_size //= 2

        # # Bottleneck
        # self.bot1 = DoubleConv(current_dim, current_dim * 2)
        # self.bot2 = DoubleConv(current_dim * 2, current_dim * 2)
        # self.bot3 = DoubleConv(current_dim * 2, current_dim)

        # # Decoder
        # self.up_blocks = nn.ModuleList()
        # self.sa_blocks_up = nn.ModuleList()

        # # Creating Up/SA Blocks Dynamically
        # for i in range(block_depth):
        #     self.up_blocks.append(UpBlock(current_dim * 2, current_dim // 2))
        #     self.sa_blocks_up.append(SelfAttention(current_dim // 2, current_image_size * 2))
        #     current_dim //= 2
        #     current_image_size *= 2

        # self.outc = nn.Conv2d(current_dim, c_out, kernel_size=1)

    # NOTE SinusoidalPositionEmbeddings....
    # for example:
    # t = torch.tensor([100,200,300,400])
    # enc = pos_encoding(t, channels=256)
    # enc.shape: torch.Size([4, 256])

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device='cuda').float() / channels) # TODO fix cuda 
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    
    # NOTE this commented lines are from Beres Andras implementation
    #     self.time_mlp = TimestepEmbedding(time_emb_dim)

    #     # Initial projection
    #     self.conv0 = nn.Conv2d(img_channel, features[0], kernel_size=3, padding=1)

    #     # Downsampling
    #     down_layers = []
    #     for i in range(len(features) - 1):
    #         down_layers.append(DownBlock(features[i], features[i+1], block_depth=block_depth, use_attention=False))  
    #     self.downs = nn.Sequential(*down_layers)

        
    #     # Bottleneck
    #     res_layers = []
    #     for _ in range(block_depth):
    #         res_layers.append(ResidualBlock(features[-1], features[-1], use_attention=False))
    #     self.bottleneck = nn.Sequential(*res_layers)

    #     # Upsampling
    #     up_layers = []
    #     for i in reversed(range(len(features) - 1)):
    #         up_layers.append(UpBlock(features[i], features[i+1], block_depth=block_depth, use_attention=False))
    #     self.ups = nn.Sequential(*up_layers)

    #     # Final
    #     self.output = nn.Conv2d(features[0], img_channel, 1)

    # def forward(self, x, timestep):
    #     # Embedd time
    #     t = self.time_mlp(timestep)
    #     # Initial conv
    #     x = self.conv0(x)
    #     # Unet
    #     residual_inputs = []
    #     for down in self.downs:
    #         x = down(x, t)
    #         residual_inputs.append(x)

    #     #bottleneck
    #     x = self.bottleneck(x)
        
    #     for up in self.ups:
    #         residual_x = residual_inputs.pop()
    #         # Add residual x as additional channels
    #         x = torch.cat((x, residual_x), dim=1)           
    #         x = up(x, t)
    #     return self.output(x)




if __name__ == '__main__':
   
    net = UNet()
    net = net.to('cuda')
    print(sum([p.numel() for p in net.parameters()]))
    print(net)
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    x = x.to('cuda')
    t = t.to('cuda')
    pred = net(x, t) 
    print(pred.shape)
