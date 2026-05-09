'''
Author: Sicen Liu
Date: 2025-05-16 14:37:10
LastEditTime: 2026-01-18 17:27:47
FilePath: /liusicen/methods/Prompt_pool_IDP/src/model.py
Description: model part

Copyright (c) 2025 by ${liusicen_cs@outlook.com}, All Rights Reserved. 
'''

import torch 
import torch.nn as nn
from src.modules import TransformerEncoder

import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, dim=None, epsilon: float = 1e-12):
    """
    L2-normalize a tensor along a given dim with numerical stability.
    Args:
        x: Tensor
        dim: 需要归一化的维度（或维度元组）；为 None 时对整个张量做归一化
        epsilon: 数值稳定项

    Returns:
        归一化后的张量，范数约为 1（在指定 dim 上）
    """
    if dim is None:
        square_sum = torch.sum(x ** 2)
        inv_norm = torch.rsqrt(torch.clamp(square_sum, min=epsilon))
        return x * inv_norm
    else:
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        inv_norm = torch.rsqrt(torch.clamp(square_sum, min=epsilon))
        return x * inv_norm

#################################################################################################
################################################################################################


from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def same_padding(kernel_size: int, dilation: int = 1) -> int:
    """Return padding for SAME length with stride=1.
    """
    return (kernel_size - 1) // 2 * dilation


def gumbel_softmax_topk(logits: torch.Tensor, k: int, tau: float = 1.0, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable top-k selection using Gumbel-Softmax.
    Args:
        logits: (B, M)
        k: number of items to select
        tau: temperature
        hard: if True, straight-through hard top-k one-hot for the selected k entries
    Returns:
        weights: (B, M) sparse-like soft weights (sum to 1 across selected indices)
        indices: (B, k) the top-k indices (non-differentiable, for inspection)
    """
    B, M = logits.shape
    # Add Gumbel noise for sampling during training; for eval we can skip by tau->0
    gumbel = -torch.empty_like(logits).exponential_().log()  # ~ Gumbel(0,1)
    y = (logits + gumbel) / max(tau, 1e-6)
    # softmax over all, then sparsify by keeping top-k mass only
    probs = F.softmax(y, dim=-1)
    topk_vals, topk_idx = torch.topk(probs, k=k, dim=-1)
    mask = torch.zeros_like(probs)
    mask.scatter_(dim=-1, index=topk_idx, src=torch.ones_like(topk_vals))
    sparse_probs = probs * mask
    # Renormalize over selected entries
    sparse_probs = sparse_probs / (sparse_probs.sum(-1, keepdim=True) + 1e-9)

    if hard:
        hard_mask = torch.zeros_like(probs)
        hard_mask.scatter_(dim=-1, index=topk_idx, src=torch.ones_like(topk_vals))
        sparse_probs = (sparse_probs - sparse_probs.detach()) + hard_mask.detach() / k
    return sparse_probs, topk_idx


# ------------------------------
# Prompt Pool + Warm Start
# ------------------------------

class PromptPool(nn.Module):
    def __init__(self, num_prompts: int, d_model: int):
        super().__init__()
        self.num_prompts = num_prompts
        self.d_model = d_model
        self.prompts = nn.Parameter(torch.randn(num_prompts, d_model) * 0.02)
        # Optional type embeddings for short/long priors
        self.type_embed = nn.Parameter(torch.randn(2, d_model) * 0.02)  # 0: short, 1: long

    @torch.no_grad()
    def warm_start(self,
                   span_embeddings: torch.Tensor,
                   span_labels: torch.Tensor,
                   n_short: int,
                   n_long: int,
                   iters: int = 20,
                   seed: int = 42):
        """Warm-start prompts using simple k-means on span embeddings.
        Args:
            span_embeddings: (N_spans, D) tensor (mean embedding per annotated span)
            span_labels: (N_spans,) int tensor with 0=short, 1=long
            n_short: number of prompts to dedicate to short spans
            n_long: number of prompts to dedicate to long spans
            iters: k-means iterations
        """
        assert n_short + n_long <= self.num_prompts
        torch.manual_seed(seed)

        def kmeans(x: torch.Tensor, k: int, iters: int) -> torch.Tensor:
            N, D = x.size()
            # Randomly pick initial centers
            perm = torch.randperm(N, device=x.device)
            centers = x[perm[:k]].clone()
            for _ in range(iters):
                # Assign
                dist = (x.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(-1)
                idx = dist.argmin(dim=1)
                # Update
                for j in range(k):
                    sel = x[idx == j]
                    if sel.numel() > 0:
                        centers[j] = sel.mean(dim=0)
            return centers

        xs = span_embeddings[span_labels == 0]
        xl = span_embeddings[span_labels == 1]
        if xs.numel() > 0 and n_short > 0:
            cs = kmeans(xs, k=n_short, iters=iters)
            self.prompts[:n_short].copy_(cs)
        if xl.numel() > 0 and n_long > 0:
            cl = kmeans(xl, k=n_long, iters=iters)
            self.prompts[n_short:n_short + n_long].copy_(cl)
        # The rest remain random

    def forward(self) -> torch.Tensor:
        return self.prompts  # (M, D)


# ------------------------------
# Dynamic Depthwise Conv Branch (HyperNet modulation)
# ------------------------------

class DepthwiseBranch(nn.Module):
    """One multi-scale depthwise separable branch with dynamic depthwise weights.

    Effective weights for this branch are generated as a mixture of top-k prompt-conditioned
    kernels: W_eff = sum_m a_m * Gen(p_m), where a_m are routing weights.

    Steps:
      - Generate depthwise conv weights (D, 1, K) via small MLP for each selected prompt
      - Fuse them with routing weights
      - Apply depthwise conv (groups=D), then shared pointwise 1x1 Conv to mix channels
    """
    def __init__(self, d_model: int, hidden: int, kernel_size: int, dilation: int):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.gen_dim = max(64, d_model // 4)
        self.gen_fc1 = nn.Linear(d_model, self.gen_dim)
        self.gen_fc2 = nn.Linear(self.gen_dim, d_model * kernel_size + d_model)

        # Shared pointwise conv after dynamic depthwise conv
        self.pointwise = nn.Conv1d(d_model, hidden, kernel_size=1, bias=True)
        self.norm = nn.LayerNorm(hidden)
        self.act = nn.GELU()

    def generate_depthwise_weight(self, prompt_combined: torch.Tensor) -> torch.Tensor:
        """Generate depthwise weights from combined prompt vector.
        prompt_combined: (B, D) mixture of top-k prompts (already weighted sum)
        Returns: weight (B, D, 1, K) and bias (B, D)
        """
        B, D = prompt_combined.shape
        # Low-rank factorization to keep params small: W = (A p)(B p)^T mapped to (D, K)
        # For simplicity, two linear layers to produce (D*K) + bias D.
        # Using shared generators across batches via linear layer on pooled prompt.
        # To avoid gigantic per-instance parameter tensors, we use a small MLP and reshape.
        # gen_dim = max(64, D // 4)
        # Cache tiny MLP as function attribute (lazy init)
        # if not hasattr(self, 'gen_fc1'):
        #     self.gen_fc1 = nn.Linear(D, gen_dim)
        #     self.gen_fc2 = nn.Linear(gen_dim, self.d_model * self.kernel_size + self.d_model)
        h = F.gelu(self.gen_fc1(prompt_combined))
        out = self.gen_fc2(h)
        w = out[..., : self.d_model * self.kernel_size]
        b = out[..., self.d_model * self.kernel_size :]
        w = w.view(B, self.d_model, 1, self.kernel_size)
        return w, b  # (B, D, 1, K), (B, D)

    def forward(self, x: torch.Tensor, prompt_mix: torch.Tensor) -> torch.Tensor:
        """Forward one branch.
        Args:
            x: (B, L, D)
            prompt_mix: (B, D) combined prompt vector for this sample
        Returns:
            y: (B, L, H)
        """
        B, L, D = x.shape
        # Generate per-batch dynamic depthwise weights
        w_dw, b_dw = self.generate_depthwise_weight(prompt_mix)  # (B, D, 1, K), (B, D)
        x1 = x.transpose(1, 2)  # (B, D, L)
        pad = same_padding(self.kernel_size, self.dilation)
        ys = []
        # Apply per-sample conv by looping B (efficient enough for small batch of long seqs)
        for i in range(B):
            yi = F.conv1d(x1[i:i+1], weight=w_dw[i], bias=b_dw[i], stride=1,
                          padding=pad, dilation=self.dilation, groups=D)  # (1, D, L)
            ys.append(yi)
        y = torch.cat(ys, dim=0)  # (B, D, L)
        y = self.pointwise(y)  # (B, H, L)
        y = y.transpose(1, 2)  # (B, L, H)
        y = self.norm(y)
        y = self.act(y)
        return y


# ------------------------------
# Position-level Gating to mix branches
# ------------------------------

class PositionGate(nn.Module):
    def __init__(self, d_in: int, num_branches: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_in, d_in // 2), nn.GELU(), nn.Linear(d_in // 2, num_branches)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D_in) -> (B, L, K)
        logits = self.gate(x)
        return F.softmax(logits, dim=-1)


# ------------------------------
# Backbones
# ------------------------------

class TCNBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_hidden, kernel_size, padding=same_padding(kernel_size, dilation), dilation=dilation)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(d_hidden, d_model, kernel_size, padding=same_padding(kernel_size, dilation), dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        residual = x
        y = x.transpose(1, 2)  # (B, D, L)
        y = self.conv1(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)
        y = self.norm(y + residual)
        return y


class TCNStack(nn.Module):
    def __init__(self, d_model: int, hidden_size:int, n_layers: int = 4, k: int = 7, base_dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(n_layers):
            dilation = base_dilation * (2 ** i)
            layers.append(TCNBlock(d_model, hidden_size, kernel_size=k, dilation=dilation, dropout=dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class LiteTransformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
                                                   dropout=dropout, batch_first=True, activation='gelu', norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        return self.encoder(x)

# ------------------------------
# Full Model
# ------------------------------

@dataclass
class BranchSpec:
    kernel_size: int
    dilation: int


class ConditionalIDRModel(nn.Module):
    def __init__(self,
                 d_model: int = 768,
                 num_prompts: int = 32,
                 top_k: int = 3,
                 branch_specs: List[BranchSpec] = (BranchSpec(5,1), BranchSpec(15,2), BranchSpec(41,4)),
                 branch_hidden: int = 512,
                 backbone: str = 'tcn',  # 'tcn' | 'transformer' | 'mamba'
                 backbone_depth: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        # d_model, num_prompts, top_k, backbone
        self.d_model = d_model
        self.num_prompts = num_prompts
        self.top_k = top_k
        self.branch_specs = list(branch_specs)
        self.num_branches = len(self.branch_specs)
        self.backbone_ = backbone

        # Prompt pool
        self.prompt_pool = PromptPool(num_prompts=num_prompts, d_model=d_model)

        # Router (sequence-level)
        self.router = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, num_prompts)
        )

        # Branches (shared across samples; weights created dynamically per-sample)
        self.branches = nn.ModuleList([
            DepthwiseBranch(d_model=d_model, hidden=d_model,
                            kernel_size=spec.kernel_size, dilation=spec.dilation)
            for spec in self.branch_specs
        ])


        # Position-level gate to mix branch outputs
        self.pos_gate = PositionGate(d_in=d_model, num_branches=self.num_branches)

        # Backbone choice
        if backbone == 'tcn':
            self.backbone = TCNStack(d_model=d_model, hidden_size=branch_hidden, n_layers=backbone_depth, k=7, base_dilation=1, dropout=dropout)
        elif backbone == 'transformer':
            self.backbone = LiteTransformer(d_model=d_model, n_heads=4, n_layers=backbone_depth, dropout=dropout)
        elif backbone == 'bilstm':
            self.backbone = nn.LSTM(input_size=d_model, hidden_size=branch_hidden, num_layers=1,
                            batch_first=True, bidirectional=True, dropout=dropout)
        else:
            raise ValueError("backbone must be one of {'tcn','transformer','mamba','bilstm}")

        # Prediction head
        # self.head = nn.Sequential(
        #     nn.Linear(branch_hidden, branch_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(branch_hidden, 1)
        # )
        if backbone == 'bilstm':
            self.output_layer = nn.Linear(branch_hidden * 2, 1)  # BiLSTM output is doubled
        else:
            self.output_layer = nn.Linear(d_model, 1)


    def _sequence_query(self, x: torch.Tensor) -> torch.Tensor:
        """Pool sequence to a global query vector q: concat(mean, max, cls-like).
        Args:
            x: (B, L, D)
        Returns:
            q: (B, 3D)
        """
        B, L, D = x.shape
        mean = x.mean(dim=1)
        mx, _ = x.max(dim=1)
        # CLS-like via attention pooling
        attn = torch.softmax((x @ x.mean(dim=1, keepdim=True).transpose(1,2)) / math.sqrt(D), dim=1)  # (B, L, 1)
        cls_like = (attn.transpose(1,2) @ x).squeeze(1)
        return torch.cat([mean, mx, cls_like], dim=-1)

    def _route_prompts(self, x: torch.Tensor, tau: float = 1.0, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute sequence-level routing over prompt pool and produce combined prompt.
        Returns:
            weights: (B, M) sparse-like top-k weights
            idx: (B, k)
            combined: (B, D) = sum_m weights_m * prompt_m
        """
        B, L, D = x.shape
        q = self._sequence_query(x)  # (B, 3D)
        logits = self.router(q)  # (B, M)
        if self.training:
            weights, idx = gumbel_softmax_topk(logits, k=self.top_k, tau=tau, hard=hard)
        else:
            # deterministic top-k softmax over selected
            topk_vals, idx = torch.topk(logits, k=self.top_k, dim=-1)
            weights = torch.zeros_like(logits)
            weights.scatter_(dim=-1, index=idx, src=F.softmax(topk_vals, dim=-1))
        prompts = self.prompt_pool()  # (M, D)
        combined = weights @ prompts  # (B, D)
        return weights, idx, combined

    def forward(self, x: torch.Tensor, tau: float = 1.0, hard_topk: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass.
        Args:
            x: (B, L, D_model) per-residue embeddings
        Returns:
            dict with:
                'logits': (B, L)
                'probs': (B, L)
                'route_weights': (B, M)
                'route_idx': (B, k)
                'gate': (B, L, num_branches)
        """
        B, L, D = x.shape
        # 1) Prompt routing
        route_w, route_idx, prompt_mix = self._route_prompts(x, tau=tau, hard=hard_topk)  # (B, M), (B,k), (B,D)

        # 2) Dynamic multi-branch convs (each branch uses prompt_mix to generate its weights)
        branch_outs = []
        for b in self.branches:
            yb = b(x, prompt_mix)  # (B, L, d_model)
            branch_outs.append(yb)
        Y = torch.stack(branch_outs, dim=-1)  # (B, L, d_model, K)

        # 3) Position-level gating over branches
        # Use the mean across branches as the gating input for stability
        gate_in = Y.mean(dim=-1)  # (B, L, d_model)
        gate = self.pos_gate(gate_in)  # (B, L, K)
        gate = gate.unsqueeze(2)  # (B, L, 1, K)
        Y_mix = (Y * gate).sum(dim=-1)  # (B, L, d_model)

        # residual
        Y_mix = Y_mix + x

        # 4) Backbone (TCN/Transformer/Mamba/BiLSTM)
        if self.backbone_ == 'bilstm':
            Z, _ = self.backbone(Y_mix)  # (B, L, H) BiLSTM returns (B, L, 2H)
        else:
            Z = self.backbone(Y_mix)  # (B, L, H)

        # 5) Prediction
        logits = self.output_layer(Z).squeeze(-1)  # (B, L)
        probs = torch.sigmoid(logits)
        return {
            'logits': logits,
            'probs': probs,
            'route_weights': route_w,
            'route_idx': route_idx,
            'gate': gate.squeeze(2),  # (B, L, K)
        }

