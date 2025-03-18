# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple, List
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import Embedding, Linear
import torch.nn.functional as F
from .auto_adapter import AUTOAdapterLayer

from flash_attn import flash_attn_func
from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
)

if is_flash_attn_2_available() and is_flash_attn_greater_or_equal_2_10():
    print("------ flash attention2 enable -----")
else:
    print("------ flash attention2 unable -----")

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: Optional[bool] = False

    flash_attention2: bool = False
    bf16: bool = False

    max_batch_size: int = 32
    max_seq_len: int = 2048

    w_bias: bool = False # use bias tuning
    # lora
    lora_layers: str = '0-0' # '0-8,24-32'
    lora_rank: int = 8
    lora_targets: str = 'Q,K,V,O,FFN_UP,FFN_DOWN'
    lora_alpha: float = 32

    # parallel adapter
    p_adapter_layers: str='0-0'
    p_adapter_size: int = 16

    # prompt
    prompt_layers: str='0-0'
    prompt_len: int = 10

    # sparse structure moe
    max_threshold: float = 0.5
    bool_weights: bool= False

    ## structure router
    swi_x: int = 0 # 0 is normal Linear, 

    ## moe 
    num_experts: int = 1 # moe when num_experts > 1
    moe_type: str = 'adamole' # adamole, topk
    top_k: int = 2 # top_k experts in topk moe
    noisy_router: bool = False # moe router
    lb_loss: bool = False # moe load balancing loss
    lb_loss_coeff: float = 0
    asym: bool = False #  Asymmetric structure for  LoRA and Parallel adapter
    

class Router(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int 
    ):
        super().__init__()

        self.w1 = Linear(
            in_dim, hidden_dim
        )
        self.w2 = Linear(
            hidden_dim, out_dim
        )
        self.w3 = Linear(
            in_dim, hidden_dim
        )
        
        nn.init.constant_(self.w1.bias.data, 0)
        nn.init.constant_(self.w2.bias.data, 0)
        nn.init.constant_(self.w3.bias.data, 0)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, w_lora=False, w_prompt=False):
        super().__init__()
        self.args = args

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=args.w_bias
        )
        self.wk = Linear(
            args.dim,
            args.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wv = Linear(
            args.dim,
            args.n_kv_heads * self.head_dim,
            bias=False
        )
        self.wo = Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=args.w_bias
        )
        if args.w_bias:
            nn.init.constant_(self.wq.bias.data, 0)
            nn.init.constant_(self.wo.bias.data, 0)

        self.w_lora = w_lora
        if self.w_lora:
            self.lora_targets = args.lora_targets.split(',')
            if 'Q' in self.lora_targets:
                self.lora_Q = AUTOAdapterLayer(num_experts=args.num_experts, moe_type=args.moe_type, top_k=args.top_k, expert_type='lora', noisy_router=args.noisy_router,\
                                                lb_loss=args.lb_loss, asym=args.asym, input_dim=args.dim, output_dim=args.dim, lora_r=args.lora_rank, lora_alpha=args.lora_alpha)
            if 'K' in self.lora_targets:
                self.lora_K = AUTOAdapterLayer(num_experts=args.num_experts, moe_type=args.moe_type, top_k=args.top_k, expert_type='lora', noisy_router=args.noisy_router, \
                                                lb_loss=args.lb_loss, asym=args.asym, input_dim=args.dim, output_dim=args.n_kv_heads * self.head_dim, lora_r=args.lora_rank, lora_alpha=args.lora_alpha)
            if 'V' in self.lora_targets:
                self.lora_V = AUTOAdapterLayer(num_experts=args.num_experts, moe_type=args.moe_type, top_k=args.top_k, expert_type='lora', noisy_router=args.noisy_router, \
                                                lb_loss=args.lb_loss, asym=args.asym, input_dim=args.dim, output_dim=args.n_kv_heads * self.head_dim, lora_r=args.lora_rank, lora_alpha=args.lora_alpha)
            if 'O' in self.lora_targets:
                self.lora_O = AUTOAdapterLayer(num_experts=args.num_experts, moe_type=args.moe_type, top_k=args.top_k, expert_type='lora', noisy_router=args.noisy_router,\
                                                lb_loss=args.lb_loss, asym=args.asym, input_dim=args.dim, output_dim=args.dim, lora_r=args.lora_rank, lora_alpha=args.lora_alpha)
            
        self.w_prompt = w_prompt
        if self.w_prompt:
            self.prompt = nn.Embedding(self.args.num_experts * args.prompt_len, args.dim)
            self.prompt_gate = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))
            if self.args.num_experts >1:
                self.prompt_router = nn.Linear(args.dim, self.args.num_experts)
                
        self.cache_k = None
        self.cache_v = None

    def train(self, mode: bool = True):
        if mode:
            self.cache_k = None
            self.cache_v = None
        else:
            self.cache_k = torch.zeros(
                (self.args.max_batch_size, self.args.max_seq_len, self.n_local_kv_heads, self.head_dim)
            ).cuda()
            self.cache_v = torch.zeros(
                (self.args.max_batch_size, self.args.max_seq_len, self.n_local_kv_heads, self.head_dim)
            ).cuda()
        return super().train(mode)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], type_weight:Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        type_idx = 0
        if self.w_lora:
            if 'Q' in self.lora_targets:
                xq = xq + self.lora_Q(x, type_weight[:,:,type_idx])
                type_idx += 1
            if 'K' in self.lora_targets:
                xk = xk + self.lora_K(x, type_weight[:,:,type_idx])
                type_idx += 1
            if 'V' in self.lora_targets:
                xv = xv + self.lora_V(x, type_weight[:,:,type_idx])
                type_idx += 1

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if not self.training:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            assert start_pos==0
            keys = xk
            values = xv
        
        if self.args.flash_attention2:
            output = flash_attn_func(xq, keys, values, causal=True).view(bsz, seqlen, -1)
        else:
            # repeat k/v heads if n_kv_heads < n_heads
            keys = repeat_kv(
                keys, self.n_rep
            )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
            values = repeat_kv(
                values, self.n_rep
            )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

            xq = xq.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)

            output = output.transpose(
                1, 2
            ).contiguous().view(bsz, seqlen, -1)

        # prompt
        if self.w_prompt:
            if self.args.flash_attention2:
                xq = xq.transpose(1, 2)

            # sparse computing can not work for prompt-tuning, using non-sparse way
            type_weight_prompt = type_weight[:,:,type_idx] 

            if self.args.num_experts == 1:
                prompt = self.prompt.weight
                prompt_k = self.wk(prompt).view(1, self.args.prompt_len, self.n_local_kv_heads, self.head_dim).repeat(bsz, 1, 1, 1)
                prompt_v = self.wv(prompt).view(1, self.args.prompt_len, self.n_local_kv_heads, self.head_dim).repeat(bsz, 1, 1, 1)

                prompt_k = repeat_kv(prompt_k, self.n_rep) # [bs, prompt_len, n_local_heads, head_dim]
                prompt_v = repeat_kv(prompt_v, self.n_rep)

                prompt_k = prompt_k.transpose(1, 2)
                prompt_v = prompt_v.transpose(1, 2) # [bs, n_local_heads, prompt_len, head_dim]

                prompt_scores = torch.matmul(xq, prompt_k.transpose(2, 3)) / math.sqrt(self.head_dim) # [bs, n_local_heads, seqlen, prompt_len]
                
                prompt_scores = self.prompt_gate * F.softmax(prompt_scores.float(), dim=-1).type_as(xq)
                
                prompt_output = torch.matmul(prompt_scores, prompt_v) # [bsz, local_heads, seqlen, head_dim]
                prompt_output = prompt_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
                
            elif self.args.num_experts >1:
                prompt = self.prompt.weight.reshape(self.args.num_experts, self.args.prompt_len, self.args.dim)
                prompt_k = self.wk(prompt).view(1, self.args.num_experts * self.args.prompt_len, self.n_local_kv_heads, self.head_dim).repeat(bsz, 1, 1, 1)
                prompt_v = self.wv(prompt).view(1, self.args.num_experts * self.args.prompt_len, self.n_local_kv_heads, self.head_dim).repeat(bsz, 1, 1, 1)

                prompt_k = repeat_kv(prompt_k, self.n_rep) # [bs, expert_num * prompt_len, n_local_heads, head_dim]
                prompt_v = repeat_kv(prompt_v, self.n_rep)

                prompt_k = prompt_k.transpose(1, 2)
                prompt_v = prompt_v.transpose(1, 2) # [bs, n_local_heads, expert_num * prompt_len, head_dim]

                prompt_scores = torch.matmul(xq, prompt_k.transpose(2, 3)) / math.sqrt(self.head_dim) # [bs, n_local_heads, seqlen, expert_num * prompt_len]
                
                prompt_scores = self.prompt_gate * F.softmax(prompt_scores.float(), dim=-1).type_as(xq)

                prompt_scores = prompt_scores.view(bsz, self.n_local_heads, -1, self.args.num_experts, self.args.prompt_len).transpose(2,3)
                prompt_v = prompt_v.view(bsz, self.n_local_heads, self.args.num_experts, self.args.prompt_len, self.head_dim)
                
                prompt_output = torch.matmul(prompt_scores, prompt_v) # [bsz, local_heads, expertnum, seqlen, head_dim]
                prompt_output = prompt_output.permute(0,3,2,1,4).contiguous().view(bsz,seqlen,self.args.num_experts, -1)
                prompt_weight = nn.functional.softmax(self.prompt_router(x), dim=-1, dtype=torch.float32).to(x.dtype)
                prompt_output = torch.sum(prompt_weight.unsqueeze(-1) * prompt_output, 2, keepdim=True)
                prompt_output = prompt_output.squeeze(2)
            
            prompt_output = prompt_output * type_weight_prompt.unsqueeze(-1)
            output = output + prompt_output

            type_idx += 1

        if self.w_lora and 'O' in self.lora_targets:
            return self.wo(output) + self.lora_O(output, type_weight[:,:,type_idx])
        else:
            return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        args: ModelArgs,
        w_lora=False
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = Linear(
            dim, hidden_dim, bias=args.w_bias
        )
        self.w2 = Linear(
            hidden_dim, dim, bias=args.w_bias
        )
        self.w3 = Linear(
            dim, hidden_dim, bias=args.w_bias
        )
        if args.w_bias:
            nn.init.constant_(self.w1.bias.data, 0)
            nn.init.constant_(self.w2.bias.data, 0)
            nn.init.constant_(self.w3.bias.data, 0)
        self.w_lora = w_lora
        if self.w_lora:
            self.lora_targets = args.lora_targets.split(',')
            if 'FFN_UP' in self.lora_targets:
                self.lora_UP = AUTOAdapterLayer(num_experts=args.num_experts, moe_type=args.moe_type, top_k=args.top_k, expert_type='lora', noisy_router=args.noisy_router, \
                                                lb_loss=args.lb_loss, asym=args.asym, input_dim=args.dim, output_dim=hidden_dim, lora_r=args.lora_rank, lora_alpha=args.lora_alpha)

            if 'FFN_DOWN' in self.lora_targets:
                self.lora_DOWN = AUTOAdapterLayer(num_experts=args.num_experts, moe_type=args.moe_type, top_k=args.top_k, expert_type='lora', noisy_router=args.noisy_router, \
                                                lb_loss=args.lb_loss, asym=args.asym, input_dim=hidden_dim, output_dim=args.dim, lora_r=args.lora_rank, lora_alpha=args.lora_alpha)

    def forward(self, x, type_weight:Optional[torch.Tensor]):
        if self.w_lora:
            type_idx = 0
            if 'FFN_UP' in self.lora_targets:
                out = F.silu(self.w1(x)) * (self.w3(x) + self.lora_UP(x, type_weight[:,:,type_idx]))
                type_idx += 1
            else:
                out = F.silu(self.w1(x)) * self.w3(x)
            
            if 'FFN_DOWN' in self.lora_targets:
                out = self.w2(out) + self.lora_DOWN(out, type_weight[:,:,type_idx])
            else:
                out = self.w2(out)
            return out
        else:
            return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, w_lora=False, w_prompt=False, w_padapter=False):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.bf16 = args.bf16
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, w_lora=w_lora, w_prompt=w_prompt)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of, ffn_dim_multiplier=args.ffn_dim_multiplier, args=args,
            w_lora=w_lora
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.w_padapter = w_padapter
        if self.w_padapter:
            self.p_adapter = AUTOAdapterLayer(num_experts=args.num_experts, moe_type=args.moe_type, top_k=args.top_k, expert_type='padapter', noisy_router=args.noisy_router, \
                                                lb_loss=args.lb_loss, asym=args.asym, input_dim=self.dim, adapter_size=args.p_adapter_size)

        self.adapter_type = 0
        self.attention_type = 0
        self.FFN_type = 0
        if w_lora:
            lora_targets = args.lora_targets.split(',')
            self.adapter_type += len(lora_targets)
            attention_targets = ['Q', 'K', 'V', 'O']
            FFN_targets = ['FFN_UP', 'FFN_DOWN']
            for x in lora_targets:
                if x in attention_targets:
                    self.attention_type += 1
                if x in FFN_targets:
                    self.FFN_type += 1
        if w_prompt:
            self.adapter_type += 1
            self.attention_type += 1
        if w_padapter:
            self.adapter_type += 1
        
        if args.swi_x == 0:
            self.adapter_type_router = nn.Linear(args.dim, self.adapter_type)
        elif args.swi_x > 0:
            self.adapter_type_router = Router(args.dim, self.adapter_type * args.swi_x, self.adapter_type)
        
        self.adapter_threshold_fn = nn.Linear(args.dim, 1)
        self.max_threshold = args.max_threshold
        self.bool_weights = args.bool_weights


    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):

        # type_weights = self.adapter_type * nn.functional.softmax(self.adapter_type_router(x), dim=-1, dtype=torch.float32).to(x.dtype)   # [bsz, seqlen, adapter_type]
        type_weights = nn.functional.sigmoid(self.adapter_type_router(x)).to(x.dtype)   # [bsz, seqlen, adapter_type]
        thresholds = F.sigmoid(self.adapter_threshold_fn(x)) * self.max_threshold # [bsz, seqlen, 1]
        adapted_type_weights = type_weights - thresholds
        selected_experts = torch.ge(adapted_type_weights, 0).to(torch.float)

        if self.bool_weights: # disard experts that weights less than threshold or use 0,1 by selected_experts
            type_weights = selected_experts
        else:
            type_weights = type_weights * selected_experts 

        type_idx = 0
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, type_weight=type_weights[:,:,type_idx:type_idx+self.attention_type])
        # out = h + self.feed_forward.forward(self.ffn_norm(h))
        type_idx += self.attention_type
        residual = h
        h = self.ffn_norm(h)
        out = self.feed_forward.forward(h, type_weight=type_weights[:,:,type_idx:type_idx+self.FFN_type])
        type_idx += self.FFN_type
        if self.w_padapter:
            adapter_states = self.p_adapter(h, type_weight=type_weights[:,:,type_idx])
            out = out + adapter_states
        out = residual + out

        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings = Embedding(
            params.vocab_size, params.dim
        )

        self.lora_layers_id = [x for span in params.lora_layers.split(',') for x in range(int(span.split('-')[0]), int(span.split('-')[1]))]
        print(f'lora_layers_id:{self.lora_layers_id}')

        self.p_adapter_layers_id = [x for span in params.p_adapter_layers.split(',') for x in range(int(span.split('-')[0]), int(span.split('-')[1]))]
        print(f'p_adapter_layers_id:{self.p_adapter_layers_id}')

        self.prompt_layers_id = [x for span in params.prompt_layers.split(',') for x in range(int(span.split('-')[0]), int(span.split('-')[1]))]
        print(f'prompt_layers_id:{self.prompt_layers_id}')

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            w_lora = False
            w_prompt = False
            w_padapter = False
            if layer_id in self.lora_layers_id:
                w_lora = True
            if layer_id in self.p_adapter_layers_id:
                w_padapter = True
            if layer_id in self.prompt_layers_id:
                w_prompt = True 
            self.layers.append(TransformerBlock(layer_id, params, w_lora=w_lora, w_prompt=w_prompt, w_padapter=w_padapter))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = Linear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()
