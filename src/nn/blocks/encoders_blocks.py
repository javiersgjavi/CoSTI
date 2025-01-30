import torch
import torch.nn as nn

from src.nn.utils import init_weights_xavier, apply_group_norm
from src.nn.blocks.basics import TransformerBlockNodeSelf, TransformerBlockTimeSelf, TransformerBlockNodeCross, TransformerBlockTimeCross

# ---------------------------- Encoder blocks ----------------------------------
class DownSpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_factor=1, dropout=0.1, attn='self'):
        super().__init__()
        self.down_factor = down_factor
        self.norm = nn.GroupNorm(4, in_channels)
        self.attn = TransformerBlockNodeSelf(channels=in_channels, dropout=dropout) if attn is 'self' else TransformerBlockNodeCross(channels=in_channels, dropout=dropout)
        self.down_sample = nn.Sequential(
            nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, down_factor),
            stride=(1, down_factor),
            ),
            nn.GroupNorm(4, out_channels),
        ).apply(init_weights_xavier)

    def forward(self, x, x_cfem = None):
        # Attention spa encoding
        if x_cfem is not None:
            x = x + self.attn(x, x_cfem)
        else:
            x = x + self.attn(x)
        x = apply_group_norm(x, self.norm)

        x = x.permute(0, 3, 1, 2) # (B, T, N, C) -> (B, C, T, N)
        x = self.down_sample(x)
        x = x.permute(0, 2, 3, 1) # (B, C, T, N) -> (B, T, N, C)
        return x
    
class DownTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_factor=1, dropout=0.1, attn='self'):
        super().__init__()
        self.down_factor = down_factor
        self.attn = TransformerBlockTimeSelf(in_channels, dropout=dropout) if attn is 'self' else TransformerBlockTimeCross(in_channels, dropout=dropout)
        self.down_sample = nn.Sequential(
            nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(down_factor, 1),
            stride=(down_factor, 1),
            ),
            nn.GroupNorm(4, out_channels),
        ).apply(init_weights_xavier)


    def forward(self, x, x_cfem = None):
        
        if x_cfem is not None:
            x = x + self.attn(x, x_cfem)
        else:
            x = x + self.attn(x)
        x = x.permute(0, 3, 1, 2) # (B, T, N, C) -> (B, C, T, N)
        x = self.down_sample(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_factor_n=1, down_factor_t=1, return_skip=True, dropout=0.1):
        super().__init__()
        self.return_skip = return_skip
        self.encoder_n = DownSpatialBlock(in_channels, out_channels, down_factor=down_factor_n, dropout=dropout)
        self.encoder_t = DownTemporalBlock(out_channels, out_channels, down_factor=down_factor_t, dropout=dropout)

    def forward(self, x):
        x1 = self.encoder_t(x)
        x2 = self.encoder_n(x1)
        if not self.return_skip:
            return x2
        return x2, x1, x 

# ---------------------------- Decoder blocks ----------------------------------

class UpTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_factor=1, dropout=0.1):
        super().__init__()
        self.up_factor = up_factor
        self.out_channels = in_channels if out_channels == in_channels else out_channels
        self.in_attn = in_channels * 2 if out_channels == in_channels else in_channels

        self.attn = TransformerBlockTimeCross(channels=self.in_attn, out_channels=self.out_channels, dropout=dropout)
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(up_factor, 1),
            stride=(up_factor, 1)
            ),
            nn.GroupNorm(4, out_channels),
            nn.GELU()
        ).apply(init_weights_xavier)

        self.t_emb_linear = nn.Linear(128, out_channels)
        self.group_norm = nn.GroupNorm(4, out_channels)
        self.proj_cat = nn.Linear(self.out_channels, self.in_attn)

    def forward(self, x, skip, skip_cfem, t_embd):
        t_embd = self.t_emb_linear(t_embd)
        x = x.permute(0, 3, 1, 2) # (B, T, N, C) -> (B, C, T, N)
        x = self.up_sample(x)
        x = x.permute(0, 2, 3, 1)

        # Add skip connection
        x_cat = torch.cat([x, skip], dim=-1)
        skip_cfem = self.proj_cat(skip_cfem)
        # attention
        x = x + self.attn(x_cat, skip_cfem) + t_embd
        x = apply_group_norm(x, self.group_norm)
        return x
    
    
class UpSpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_factor=1, dropout=0.1):
        super().__init__()
        self.up_factor = up_factor
        self.norm = nn.GroupNorm(4, in_channels)
        self.in_attn = in_channels * 2 if out_channels == in_channels else in_channels
        self.out_channels = in_channels if out_channels == in_channels else out_channels
        self.attn = TransformerBlockNodeCross(channels=self.in_attn, out_channels=self.out_channels, dropout=dropout)
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, up_factor),
            stride=(1, up_factor)
            ),
            nn.GroupNorm(4, out_channels),
            nn.GELU()
        ).apply(init_weights_xavier)

        self.t_emb_linear = nn.Linear(128, out_channels)
        self.proj_skip = nn.Linear(out_channels, self.in_attn)

    def forward(self, x, skip, skip_cfem, t_embd):
        t_embd = self.t_emb_linear(t_embd)

        x = x.permute(0, 3, 1, 2) # (B, T, N, C) -> (B, C, T, N)
        x = self.up_sample(x)
        x = x.permute(0, 2, 3, 1)

        # Add skip connection
        x_cat = torch.cat([x, skip], dim=-1)
        skip_cfem = self.proj_skip(skip_cfem)

        # attention
        x = x + self.attn(x_cat, skip_cfem) + t_embd
        x = apply_group_norm(x, self.norm)
        return x
