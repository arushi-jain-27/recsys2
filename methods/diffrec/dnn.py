import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super().__init__()
        self.time_emb_dim = emb_size
        self.norm = norm
        self.emb_layer = nn.Linear(emb_size, emb_size)
        in_dims_temp = [in_dims[0] + emb_size] + in_dims[1:]
        self.in_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])]
        )
        self.out_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims[:-1], out_dims[1:])]
        )
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in list(self.in_layers) + list(self.out_layers) + [self.emb_layer]:
            fan_out, fan_in = layer.weight.size()
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps):
        emb = self.emb_layer(timestep_embedding(timesteps, self.time_emb_dim).to(x.device))
        if self.norm:
            x = F.normalize(x)
        h = torch.cat([self.drop(x), emb], dim=-1)
        for layer in self.in_layers:
            h = torch.tanh(layer(h))
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        return h


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
