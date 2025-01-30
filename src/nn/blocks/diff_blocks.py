import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nn.blocks.basics import MLP, MambaMLPBlock, TransformerBlockNodeSelf, TransformerBlockTimeCross, TransformerBlockNodeCross
from src.nn.blocks.encoders_blocks import EncoderBlock
from src.nn.blocks.gnn import AdaptiveGCN

from src.nn.utils import apply_group_norm, init_weights_xavier

class NoiseLevelEmbeddingFourier(nn.Module):
    def __init__(self, channels, scale_factor=0.02):
        super().__init__()
        self.W = nn.Parameter(torch.randn(channels//2)*scale_factor, requires_grad=False)
        self.mlp = MLP(channels)

    def forward(self, t):
        h = t[:, None] * self.W[None, :] * 2 * torch.pi
        h =  torch.cat([torch.sin(h), torch.cos(h)], dim=1)
        return self.mlp(h)
    
class NoiseLevelPositionalEmbedding(nn.Module):
    def __init__(self, channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.channels = channels
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.mlp =  MLP(channels)

    def forward(self, x, mlp_pass=True):
        freqs = torch.arange(start=0, end=self.channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)

        return self.mlp(x) if mlp_pass else x

    
class STFEM(nn.Module):
    def __init__(self, channels, support, n_heads, use_gnn=True, dropout=0.1):
        super().__init__()
        self.support = support
        self.use_gnn = use_gnn

        self.mamba = MambaMLPBlock(channels, dropout=dropout)
        self.attn_s = TransformerBlockNodeSelf(channels=channels, n_heads=n_heads, dropout=dropout)
        self.norm_spa_t = nn.GroupNorm(4, channels)
        self.mlp = MLP(channels, scale=2)

        self.norm_t = nn.GroupNorm(4, channels)
        self.norm_out = nn.GroupNorm(4, channels)

        if self.use_gnn:
            self.gcn = AdaptiveGCN(channels)
            self.norm_spa_gnn = nn.GroupNorm(4, channels)

        self.node_embd_proj = nn.Linear(128, channels)

    def forward(self, x, node_embd):
        B, T, N, C = x.shape

        # Time encoding
        x_time = self.mamba(x)#  + x
        x_time = apply_group_norm(x_time, self.norm_t)

        # Attention spa encoding
        node_embd = self.node_embd_proj(node_embd)
        x_spa_attn = x + self.attn_s(x + node_embd)
        x_spa_attn = apply_group_norm(x_spa_attn, self.norm_spa_t)

        # GNN encoding
        if self.use_gnn:
            x_spa_gnn = x + self.gcn(
                x.permute(0, 3, 2, 1).reshape(B, C, T*N), # (B, C, T*N)
                [B, C, N, T], 
                self.support
            ).reshape(B, C, T, N).permute(0, 2, 3, 1) # (B, T, N, C)
            x_spa_gnn = apply_group_norm(x_spa_gnn, self.norm_spa_gnn)


        # Aggregation
        if self.use_gnn:
            x = x_time + x_spa_attn + x_spa_gnn
        else:
            x = x_time + x_spa_attn

        x = x + self.mlp(x)
        x = apply_group_norm(x, self.norm_out)

        return x
    
class NEM(nn.Module):
    def __init__(self, channels, n_nodes, dropout=0.1, n_heads=8):
        super().__init__()
        self.channels = channels
        self.dropout = dropout

        self.n_nodes = n_nodes
        self.t_emb_proj = nn.Linear(128, channels)
        self.mamba = MambaMLPBlock(channels, dropout=dropout)
        self.attn_t = TransformerBlockTimeCross(channels=channels, n_heads=n_heads, dropout=dropout)
        self.attn_spa = TransformerBlockNodeCross(channels=channels, n_heads=n_heads, dropout=dropout)
        self.mlp = MLP(channels, scale=2, dropout=dropout)

        self.norm_t_attn = nn.GroupNorm(4, channels)
        self.norm_spa_attn = nn.GroupNorm(4, channels)
        self.norm_t_mamba = nn.GroupNorm(4, channels)
        self.norm_mlp = nn.GroupNorm(4, channels)

        self.attn_spa_self = TransformerBlockNodeSelf(channels=channels, n_heads=n_heads, dropout=dropout)
        self.norm_spa_self = nn.GroupNorm(4, channels)


    def forward(self, x_in, cond_info,  t_embd):
        # Add t embedding
        x = x_in + self.t_emb_proj(t_embd)

        # Extract temporal info
        x = self.attn_t(x, cond_info)
        x = apply_group_norm(x, self.norm_t_attn)

        x = self.mamba(x)
        x = apply_group_norm(x, self.norm_t_mamba)

        # Extract spatial info
        x = x + self.attn_spa(x, cond_info)
        x = apply_group_norm(x, self.norm_spa_attn)

        # Self attention
        x = x + self.attn_spa_self(x)
        x = apply_group_norm(x, self.norm_spa_self)

        x = x + self.mlp(x)
        x = apply_group_norm(x, self.norm_mlp)
        return x

    
class NEMBlock(nn.Module):
    def __init__(self, channels, n_heads=8, dropout=0.1, n_nodes=8):
        super().__init__()
        self.channels = channels

        self.nem = NEM(channels, n_heads=n_heads, dropout=dropout, n_nodes=n_nodes)
        self.mlp_gate = nn.Sequential(
            nn.Dropout(dropout),
            MLP(channels, out_channels=2*channels, scale=4, dropout=dropout),
        )
        self.mlp_out = nn.Sequential(
            nn.Dropout(dropout),
            MLP(channels, out_channels=2*channels, scale=4, dropout=dropout)
        )

        self.norm_gate = nn.GroupNorm(4, 2*channels)
        self.norm_out = nn.GroupNorm(4, 2*channels)


    def forward(self, x_in, cond_info, t_embd):
        x = self.nem(x_in, cond_info, t_embd)

        # Gated MLP
        x = self.mlp_gate(x)
        x = apply_group_norm(x, self.norm_gate)
        gate, filter = x.split(self.channels, dim=-1)
        x = F.sigmoid(gate) * F.tanh(filter)

        # Output MLP
        x = self.mlp_out(x)
        x = apply_group_norm(x, self.norm_out)

        residual, skip = x.split(self.channels, dim=-1)
        out = (x_in + residual)/math.sqrt(2.0)

        return out, skip
    

class CFEM(nn.Module):
    def __init__(self, in_channels, out_channels, down_factor_n, down_factor_t, support=None, n_heads=8, dropout=0.1):
        super().__init__()
        self.channels = out_channels
        self.dfn = down_factor_n
        self.dft = down_factor_t

        self.input_proj = nn.Sequential(
            nn.Linear(2, self.channels),
            nn.ReLU()
        ).apply(init_weights_xavier)

        self.fe = STFEM(in_channels, support, n_heads, use_gnn=False, dropout=dropout)

        self.return_skip = True
        self.encoder = EncoderBlock(
             in_channels=self.channels,
            out_channels=self.channels,
            down_factor_n=self.dfn,
            down_factor_t=self.dft,
            return_skip=self.return_skip,
            dropout=dropout
        )

    def forward(self, x, node_embd):
        # Create input encoder
        x = self.input_proj(x)
        # Encode
        x = self.fe(x, node_embd)
        if self.return_skip:
            z, x1, x = self.encoder(x)
            return z, x1, x
        return self.encoder(x)
