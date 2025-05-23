# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple, List
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import Embedding, Linear
import torch.nn.functional as F

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
    hydra_moe: bool = False # hydra lora, Asymmetric LoRA

    # parallel adapter
    p_adapter_layers: str='0-0'
    p_adapter_size: int = 16
    p_adapter_hydra: bool = False

    # prompt
    prompt_layers: str='0-0'
    prompt_len: int = 10

    # moe
    expert_num: int = 1
    
    expert_weight: bool= False # weight by expert param number


class MOELoraLayer(nn.Module):
    def __init__(self, input_dim, output_dim, r, expert_num, lora_alpha:float=8, hydra=False):
        super().__init__()

        self.expert_num = expert_num
        self.hydra = hydra # hydra lora
        self.scaling = lora_alpha / r
        self.params_num = 0

        if expert_num == 1:
            self.lora_A = nn.Linear(input_dim, r, bias=False)
            self.lora_B = nn.Linear(r, output_dim, bias=False)
            nn.init.zeros_(self.lora_B.weight)

        elif expert_num > 1: # moe
            self.router = nn.Linear(input_dim, expert_num, bias=False)
            if hydra:
                self.lora_A = nn.Linear(input_dim, r, bias=False)
            else:
                self.lora_A_l = nn.ModuleList()
                for i in range(expert_num):
                    self.lora_A_l.append(nn.Linear(input_dim, r, bias=False))
                
            self.lora_B_l = nn.ModuleList()
            for i in range(expert_num):
                self.lora_B_l.append(nn.Linear(r, output_dim, bias=False))

            # initial lora B to zeros
            for linear in self.lora_B_l:
                nn.init.zeros_(linear.weight)
        else:
            raise Exception("The number of Experts is wrong")
    
    def params_count(self):
        self.params_num = 0
        if self.expert_num == 1:
            self.params_num += torch.numel(self.lora_A.weight)
            self.params_num += torch.numel(self.lora_B.weight)

        elif self.expert_num > 1: # moe
            if self.hydra:
                self.params_num += torch.numel(self.lora_A.weight)
            else:
                for i in range(self.expert_num):
                    self.params_num += torch.numel(self.lora_A_l[i].weight)
                
            for i in range(self.expert_num):
                self.params_num += torch.numel(self.lora_B_l[i].weight)
        return self.params_num


    def forward(self, x: torch.Tensor):
        if self.expert_num == 1:
            return self.lora_B(self.lora_A(x)) * self.scaling
        
        # type_weight: [bsz, seqlen]
        route_weight = nn.functional.softmax(self.router(x), dim=-1, dtype=torch.float32).to(x.dtype) # [bsz, seqlen, expert_num]

        # 收集router权重
        self.route_weight = route_weight

        result = None
        for i in range(self.expert_num):
            if self.hydra:
                tmp = torch.unsqueeze(route_weight[:,:,i], -1) * self.lora_B_l[i](self.lora_A(x)) * self.scaling
            else:
                tmp = torch.unsqueeze(route_weight[:,:,i], -1) * self.lora_B_l[i](self.lora_A_l[i](x)) * self.scaling
            if i == 0:
                result = tmp
            else:
                result = result + tmp
        return result


class PAdapterLayer(nn.Module):
    def __init__(self, hidden_size, adapter_size, expert_num:int=1, hydra:bool=False):
        super(PAdapterLayer, self).__init__()
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        self.expert_num = expert_num
        self.hydra = hydra

        self.adapter_act_fn = nn.SiLU()

        if expert_num == 1:
            self.down_proj = nn.Linear(hidden_size, adapter_size)
            self.up_proj = nn.Linear(adapter_size, hidden_size)
        elif expert_num > 1: # moe
            self.router = nn.Linear(hidden_size, expert_num)
            if hydra:
                self.down_proj = nn.Linear(hidden_size, adapter_size)
            else:
                self.down_proj_l = nn.ModuleList()
                for i in range(expert_num):
                    self.down_proj_l.append(nn.Linear(hidden_size, adapter_size))
                
            self.up_proj_l = nn.ModuleList()
            for i in range(expert_num):
                self.up_proj_l.append(nn.Linear(adapter_size, hidden_size))
        else:
            raise Exception("The number of Experts is wrong")
        
        self.reset_parameters()

    def reset_parameters(self):
        if self.expert_num ==1:
            nn.init.xavier_uniform_(self.down_proj.weight, gain=1e-4)
            nn.init.xavier_uniform_(self.up_proj.weight, gain=1e-4)
            # nn.init.zeros_(self.up_proj.weight) # zero init like lora
            nn.init.constant_(self.down_proj.bias, 0.0)
            nn.init.constant_(self.up_proj.bias, 0.0)
        elif self.expert_num >1:
            if self.hydra:
                nn.init.xavier_uniform_(self.down_proj.weight, gain=1e-4)
                nn.init.constant_(self.down_proj.bias, 0.0)
            else:
                for i in range(self.expert_num):
                    nn.init.xavier_uniform_(self.down_proj_l[i].weight, gain=1e-4)
                    nn.init.constant_(self.down_proj_l[i].bias, 0.0)
            for i in range(self.expert_num):
                nn.init.xavier_uniform_(self.up_proj_l[i].weight, gain=1e-4)
                # nn.init.zeros_(self.up_proj_l[i].weight) # zero init like lora
                nn.init.constant_(self.up_proj_l[i].bias, 0.0)

    def forward(self, x):
        if self.expert_num == 1:
            x = self.down_proj(x)
            x = self.adapter_act_fn(x)
            x = self.up_proj(x)
            return x 

        # type_weight: [bsz, seqlen]
        route_weight = nn.functional.softmax(self.router(x), dim=-1, dtype=torch.float32).to(x.dtype) # [bsz, seqlen, expert_num]

        result = None
        for i in range(self.expert_num):
            if self.hydra:
                tmp = torch.unsqueeze(route_weight[:,:,i], -1) * self.up_proj_l[i](self.adapter_act_fn(self.down_proj(x)))
            else:
                tmp = torch.unsqueeze(route_weight[:,:,i], -1) * self.up_proj_l[i](self.adapter_act_fn(self.down_proj_l[i](x)))
            if i == 0:
                result = tmp
            else:
                result = result + tmp
        return result


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
                self.lora_Q = MOELoraLayer(args.dim, args.dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe)
            if 'K' in self.lora_targets:
                self.lora_K = MOELoraLayer(args.dim, args.n_kv_heads * self.head_dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe)
            if 'V' in self.lora_targets:
                self.lora_V = MOELoraLayer(args.dim, args.n_kv_heads * self.head_dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe)
            if 'O' in self.lora_targets:
                self.lora_O = MOELoraLayer(args.dim, args.dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe)
            
        self.w_prompt = w_prompt
        if self.w_prompt:
            self.prompt = nn.Embedding(args.expert_num * args.prompt_len, args.dim)
            self.prompt_gate = torch.nn.Parameter(torch.zeros(1, self.n_local_heads, 1, 1))
            if self.args.expert_num >1:
                self.prompt_router = nn.Linear(args.dim, self.args.expert_num)
                
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

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        if self.w_lora:
            if 'Q' in self.lora_targets:
                xq = xq + self.lora_Q(x)
            if 'K' in self.lora_targets:
                xk = xk + self.lora_K(x)
            if 'V' in self.lora_targets:
                xv = xv + self.lora_V(x)

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
            prompt = self.prompt.weight.reshape(self.args.expert_num, self.args.prompt_len, self.args.dim)
            prompt_k = self.wk(prompt).view(1, self.args.expert_num * self.args.prompt_len, self.n_local_kv_heads, self.head_dim).repeat(bsz, 1, 1, 1)
            prompt_v = self.wv(prompt).view(1, self.args.expert_num * self.args.prompt_len, self.n_local_kv_heads, self.head_dim).repeat(bsz, 1, 1, 1)

            prompt_k = repeat_kv(prompt_k, self.n_rep) # [bs, expert_num * prompt_len, n_local_heads, head_dim]
            prompt_v = repeat_kv(prompt_v, self.n_rep)

            prompt_k = prompt_k.transpose(1, 2)
            prompt_v = prompt_v.transpose(1, 2) # [bs, n_local_heads, expert_num * prompt_len, head_dim]

            prompt_scores = torch.matmul(xq, prompt_k.transpose(2, 3)) / math.sqrt(self.head_dim) # [bs, n_local_heads, seqlen, expert_num * prompt_len]
            
            prompt_scores = self.prompt_gate * F.softmax(prompt_scores.float(), dim=-1).type_as(xq)

            prompt_scores = prompt_scores.view(bsz, self.n_local_heads, -1, self.args.expert_num, self.args.prompt_len).transpose(2,3)
            prompt_v = prompt_v.view(bsz, self.n_local_heads, self.args.expert_num, self.args.prompt_len, self.head_dim)
            
            experts_output = torch.matmul(prompt_scores, prompt_v) # [bsz, local_heads, expertnum, seqlen, head_dim]
            experts_output = experts_output.permute(0,3,2,1,4).contiguous().view(bsz,seqlen,self.args.expert_num, -1)
            if self.args.expert_num >1:
                prompt_weight = nn.functional.softmax(self.prompt_router(x), dim=-1, dtype=torch.float32).to(x.dtype)
                experts_output = torch.sum(prompt_weight.unsqueeze(-1) * experts_output, 2, keepdim=True)
            experts_output = experts_output.squeeze(2)
            output = output + experts_output

        if self.w_lora and 'O' in self.lora_targets:
            return self.wo(output) + self.lora_O(output)
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
                self.lora_UP = MOELoraLayer(args.dim, hidden_dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe)

            if 'FFN_DOWN' in self.lora_targets:
                self.lora_DOWN = MOELoraLayer(hidden_dim, args.dim, args.lora_rank, args.expert_num, args.lora_alpha, args.hydra_moe)

    def forward(self, x):
        if self.w_lora:
            if 'FFN_UP' in self.lora_targets:
                out = F.silu(self.w1(x)) * (self.w3(x) + self.lora_UP(x))
            else:
                out = F.silu(self.w1(x)) * self.w3(x)
            
            if 'FFN_DOWN' in self.lora_targets:
                out = self.w2(out) + self.lora_DOWN(out)
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
            self.p_adapter = PAdapterLayer(self.dim, args.p_adapter_size, args.expert_num, args.p_adapter_hydra)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):

        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        # out = h + self.feed_forward.forward(self.ffn_norm(h))
        residual = h
        h = self.ffn_norm(h)
        out = self.feed_forward.forward(h)
        if self.w_padapter:
            adapter_states = self.p_adapter(h)
            out = out + adapter_states
        out = residual + out

        if not self.bf16:
            out = out.clamp(min=-65500, max=65500)
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
