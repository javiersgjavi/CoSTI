import math

import torch
import torch.nn as nn
from torchinfo import summary

def print_summary_model_ae(model, params, depth=2):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {pytorch_total_params}")
    t = params['time_steps']
    n = params['n_nodes']
    b = params['batch_size']
    summary(
            model,
            input_size=[(b, t, n, 1)],
            dtypes=[torch.float32],
            col_names=['input_size', 'output_size', 'num_params'],
            depth=depth
            )
    
def print_summary_model_diff(model, params, depth=3):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {pytorch_total_params}")
    t = params['time_steps'] // 2
    n = params['n_nodes'] // 9
    b = params['batch_size']
    c = 64
    summary(
            model,
            input_size=[(b, t, n, c), (b, t, n, c), (1,)],
            dtypes=[torch.float32, torch.float32, torch.int64],
            col_names=['input_size', 'output_size', 'num_params'],
            depth=depth
            )

def init_weights_xavier(m: nn.Module) -> None:
    """
    Initialize the weights of the neural network module using the Xavier initialization method.

    Args:
        m (nn.Module): Neural network module

    Returns:
        None
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: 
            m.bias.data.fill_(0.01)

def init_weights_kaiming(m: nn.Module) -> None:
    """
    Initialize the weights of the neural network module using the Kaiming initialization method.

    Args:
        m (nn.Module): Neural network module

    Returns:
        None
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None: 
            m.bias.data.fill_(0.01)

def init_weights_mamba(
    module,
    n_layer=1,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def apply_group_norm(x, layer):
        x = x.permute(0, 3, 1, 2) # (B, T, N, C) -> (B, C, T, N)
        x = layer(x)
        x = x.permute(0, 2, 3, 1) # (B, C, T, N) -> (B, T, N, C)
        return x
    
def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def rescale_vector(vector, tensor):
    res = torch.ones_like(tensor) * vector.view(tensor.shape[0], 1, 1, 1)
    return res

