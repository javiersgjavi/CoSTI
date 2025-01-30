import torch.nn as nn
from src.nn.blocks.bimamba.modules.mamba_simple import BiMamba
from src.nn.utils import init_weights_kaiming, init_weights_mamba
    
class MLP(nn.Module):
    def __init__(self, channels, scale=4, out_channels=None, dropout=0.0):
        super().__init__()

        mid_channels = scale * channels
        out_channels = out_channels if out_channels is not None else channels
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_channels, out_channels)
        ).apply(init_weights_kaiming)

    def forward(self, x):
        return self.mlp(x)

class CustomBiMamba(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()

        self.mamba_block = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Dropout(dropout),
            BiMamba(d_model=channels, bimamba_type='v2'),
            nn.LayerNorm(channels),
        ).apply(init_weights_mamba)


    def forward(self, x):
        B, T, N, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B*N, T, C) # (B*N, T, F)
        x = self.mamba_block(x) + x
        x = x.reshape(B, N, T, C).permute(0, 2, 1, 3) #(B, T, N, F)
        return x

    
class MambaMLPBlock(nn.Module):
    def __init__(self, channels, dropout=0.1, scale_mlp=2):
        super().__init__()
        self.mamba = CustomBiMamba(channels=channels, dropout=dropout)
        self.mlp = MLP(channels=channels, scale=scale_mlp)
        self.layer_norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = self.mamba(x)
        x = x + self.mlp(self.layer_norm(x))
        return x
    
# ---------------------------- Self-attention ----------------------------------

class SelfAttention(nn.Module):
    def __init__(self, channels, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, 
            num_heads=n_heads, 
            dropout=dropout, 
            bias=False,
            batch_first=True
        )

    def forward(self, x):
        return self.attn(query=x, key=x, value=x, need_weights=False)[0]
    
class TransformerBlockSelf(nn.Module):
    def __init__(self, channels, n_heads=8, dropout=0.1, out_channels=None):
        super().__init__()
        self.in_channels = channels
        self.out_channels = out_channels if out_channels is not None else channels

        self.ln1 = nn.LayerNorm(self.in_channels)
        self.ln2 = nn.LayerNorm(self.in_channels)
        self.attn = SelfAttention(self.in_channels, n_heads=n_heads, dropout=dropout)
        self.mlp = MLP(self.in_channels, out_channels=self.out_channels)

    def forward(self, x):
        # Communication
        x = x + self.attn(self.ln1(x))
        # Computation
        if self.in_channels == self.out_channels:
            x = x + self.mlp(self.ln2(x))
        else:
            x = self.mlp(self.ln2(x))
        return x
    
class TransformerBlockNodeSelf(TransformerBlockSelf):
    def __init__(self, proj_n=None, **kwargs):
        super().__init__(**kwargs)
        if proj_n is not None:
            self.n = proj_n[0]
            self.proj_n = proj_n[1]

            self.proj_n_down = nn.Linear(self.n, self.proj_n)
            self.proj_n_up = nn.Linear(self.proj_n, self.n)

    def regular_forward(self, x):
        '''expect a tensor of shape (B, T, N, C)'''
        B, T, N, C = x.shape
        x = x.reshape(B*T, N, C)
        x =  super().forward(x)
        x = x.reshape(B, T, N, self.out_channels)
        return x


    def proj_forward(self, x):
        B, T, N, C = x.shape

        x = x.permute(0, 1, 3, 2) # (B, T, C, N)
        x = self.proj_n_down(x)
        x = x.permute(0, 1, 3, 2).reshape(B*T, self.proj_n, C) # (B, T, N, C)

        x = super().forward(x)

        x = x.reshape(B, T, self.proj_n, self.out_channels).permute(0, 1, 3, 2) # (B, T, C, N)
        x = self.proj_n_up(x)
        x = x.permute(0, 1, 3, 2) # (B, T, N, C)
        return x
    
    def forward(self, x):
        if hasattr(self, 'proj_n'):
            return self.proj_forward(x)
        else:
            return self.regular_forward(x)
        
    
class TransformerBlockTimeSelf(TransformerBlockSelf):
    def forward(self, x):
        '''expect a tensor of shape (B, T, N, C)'''
        B, T, N, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B*N, T, C) # (B*N, T, C)
        x =  super().forward(x)
        x = x.reshape(B, N, T, self.out_channels).permute(0, 2, 1, 3) # (B, T, N, C)
        return x

# ---------------------------- Cross-attention ---------------------------------

class CrossAttention(SelfAttention):
    def forward(self, x, y):
        return self.attn(query=x, key=y, value=y, need_weights=False)[0]
    
class TransformerBlockCross(nn.Module):
    def __init__(self, channels, n_heads=8, dropout=0.1, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels if out_channels is not None else channels
        self.ln1_1 = nn.LayerNorm(self.channels)
        self.ln1_2 = nn.LayerNorm(self.channels)
        self.ln2 = nn.LayerNorm(self.channels)
        self.attn = CrossAttention(self.channels, n_heads=n_heads, dropout=dropout)
        self.mlp = MLP(channels=self.channels, out_channels=self.out_channels)

    def forward(self, x, y):
        # Communication
        x = x + self.attn(self.ln1_1(x), self.ln1_2(y))
        # Computation
        if self.channels == self.out_channels:
            x = x + self.mlp(self.ln2(x))
        else:
            x = self.mlp(self.ln2(x)) 
        return x
    
class TransformerBlockNodeCross(TransformerBlockCross):
    def forward(self, x, y):
        '''expect a tensor of shape (B, T, N, C)'''
        B, T, N, C = x.shape
        x = x.reshape(B*T, N, C)
        y = y.reshape(B*T, N, C)

        x = super().forward(x, y)

        x = x.reshape(B, T, N, self.out_channels)
        return x
    
class TransformerBlockTimeCross(TransformerBlockCross):
    def forward(self, x, y):
        '''expect a tensor of shape (B, T, N, C)'''
        B, T, N, C = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B*N, T, C) # (B*N, T, C)
        y = y.permute(0, 2, 1, 3).reshape(B*N, T, C) # (B*N, T, C)

        x = super().forward(x, y)

        x = x.reshape(B, N, T, self.out_channels).permute(0, 2, 1, 3) # (B, T, N, C)
        return x
