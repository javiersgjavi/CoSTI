import torch
import torch.nn as nn

from src.nn.blocks.basics import MLP
from src.nn.blocks.diff_blocks import STFEM, CFEM, NEMBlock, NoiseLevelPositionalEmbedding
from src.nn.blocks.encoders_blocks import UpSpatialBlock, UpTemporalBlock, DownSpatialBlock, DownTemporalBlock

from src.nn.blocks.gnn import compute_support_gwn
from src.nn.utils import init_weights_xavier, apply_group_norm

class Unet(nn.Module):
    def __init__(
            self, 
            channels=64, 
            n_nodes=207, 
            t_steps=24, 
            nem_blocks=4, 
            T=50, 
            dropout=0.1, 
            n_heads=8, 
            undersampling_factor_time=None, 
            undersampling_factor_node=None,
            adj=None
        ):
        super().__init__()

        print(f'Dropout: {dropout}')
        self.uft = undersampling_factor_time
        self.ufn = undersampling_factor_node

        self.channels = channels
        self.n_nodes = n_nodes
        self.t_steps = t_steps
        self.nem_blocks = nem_blocks
        self.n_heads = n_heads
        self.dropout = dropout

        # GNN params
        support = compute_support_gwn(adj)
        nodevec1 = nn.Parameter(torch.randn(self.n_nodes, 10), requires_grad=True)
        nodevec2 = nn.Parameter(torch.randn(10, self.n_nodes), requires_grad=True)
        support.append([nodevec1, nodevec2])


        self.in_proj = nn.Sequential(
            nn.Linear(1, self.channels),
            nn.ReLU()
        ).apply(init_weights_xavier)

        self.t_embd = nn.Embedding(T, 128)
        self.node_embd = nn.Embedding(n_nodes, 128)
        self.register_buffer('node_arange', torch.arange(n_nodes))

        self.nems = nn.ModuleList([NEMBlock(self.channels, dropout=dropout, n_nodes=n_nodes) for _ in range(self.nem_blocks)])

        self.cfem = CFEM(
            in_channels=self.channels,
            out_channels=self.channels,
            support=support,
            down_factor_n=self.ufn,
            down_factor_t=self.uft,
            dropout=dropout
            )

        self.input_sfe = STFEM(self.channels, support, self.n_heads, n_nodes, dropout=dropout)

        self.down_t = DownTemporalBlock(self.channels, self.channels, down_factor=self.uft, dropout=dropout, attn='cross')
        self.down_n = DownSpatialBlock(self.channels, self.channels, down_factor=self.ufn, dropout=dropout, attn='cross')

        self.up_n = UpSpatialBlock(self.channels, self.channels, up_factor=self.ufn, dropout=dropout)
        self.up_t = UpTemporalBlock(self.channels, self.channels, up_factor=self.uft, dropout=dropout)

        self.nems_norm = nn.GroupNorm(4, self.channels)


        self.out_proj = nn.Sequential(
            MLP(self.channels, out_channels=self.channels, scale=4),
            nn.Linear(self.channels, 1).apply(init_weights_xavier)
        )

    def apply_nems(self, x, cond_info, t_embd):
        skip = []
        for i in range(self.nem_blocks):
            x, skip_connection = self.nems[i](x, cond_info, t_embd)
            skip.append(skip_connection)

        x = torch.stack(skip)
        x = torch.sum(x, dim=0)
        x = apply_group_norm(x, self.nems_norm)
        return x
    
    def upsample(self, x):
        x = x.permute(0, 3, 1, 2) # (B, T, N, C) -> (B, C, T, N)
        x = self.upsampler(x)
        x = x.permute(0, 2, 3, 1) # (B, C, T, N) -> (B, T, N, C)
        return x

    def get_node_embd(self):
        return self.node_embd(self.node_arange)
    
    def get_cond_info(self, x_itp, mask):
        node_embd = self.get_node_embd()
        x = torch.cat([x_itp, mask], dim=-1)
        return self.cfem(x, node_embd)
    
    def estimate_noise(self, x_noisy, z_cfem, t):
        # Encode input
        x = x_noisy
        t_emb = self.t_embd(t).view(x.shape[0], 1, 1, 128)

        # Extrar features input
        node_embd = self.node_embd(self.node_arange)
        x = self.in_proj(x)
        x = self.input_sfe(x, node_embd)

        # z_cfem
        z_cfem, x1_cfem, x_cfem = z_cfem

        # Encode input
        x1 = self.down_t(x, x_cfem)
        z = self.down_n(x1, x1_cfem)

        # Apply NEM blocks
        z = self.apply_nems(z, z_cfem, t_emb)

        z = self.up_n(z, x1, x1_cfem, t_emb)
        x = self.up_t(z, x, x_cfem, t_emb)

        x = self.out_proj(x)
        return x

    def forward(self, x_noisy, x_itp, mask, t):
        z_cfem = self.get_cond_info(x_itp, mask)
        return self.estimate_noise(x_noisy, z_cfem, t)
    
class UnetContinuous(Unet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_embd = NoiseLevelPositionalEmbedding(128)
