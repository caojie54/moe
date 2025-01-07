import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .sparse_moe import LoraLayer, PAdapterLayer, TopkMoE, AdaMoLE

class AUTOAdapterLayer(nn.Module):
    def __init__(self, num_experts=2, moe_type='topk', top_k=2, expert_type='lora', noisy_router=False, lb_loss=False, asym=False, input_dim=None, output_dim=None, lora_r=None, lora_alpha=None, adapter_size=None):
        super().__init__()

        if expert_type == 'lora':
            if not (input_dim and output_dim and lora_r and lora_alpha):
                raise Exception('lora parameter error')
        elif expert_type == 'padapter':
            if not (input_dim and adapter_size):
                raise Exception('padapter parameter error')
        else:
            raise Exception('expert type wrong')
        
        if not output_dim:
            self.output_dim = input_dim
        else:
            self.output_dim = output_dim

        if num_experts == 1:
            if expert_type == 'lora':
                self.expert = LoraLayer(input_dim=input_dim, output_dim=output_dim, r=lora_r, lora_alpha=lora_alpha)
            elif expert_type == 'padapter':
                self.expert = PAdapterLayer(hidden_size=input_dim, adapter_size=adapter_size)
            else:
                raise Exception('expert_type Error')
        elif num_experts > 1:
            if moe_type == 'topk':
                self.expert = TopkMoE(num_experts=num_experts, top_k=top_k, expert_type=expert_type, noisy_router=noisy_router, lb_loss=lb_loss, asym=asym, input_dim=input_dim, output_dim=output_dim, lora_r=lora_r, lora_alpha=lora_alpha, adapter_size=adapter_size)
            elif moe_type == 'adamole':
                self.expert = AdaMoLE(num_experts=num_experts, expert_type=expert_type, noisy_router=noisy_router, lb_loss=lb_loss, asym=asym, input_dim=input_dim, output_dim=output_dim, lora_r=lora_r, lora_alpha=lora_alpha, adapter_size=adapter_size)
            else:
                raise Exception('moe type error')

    
    def forward(self, x: torch.Tensor, type_weight: torch.Tensor):
        # type_weight: [bsz, seqlen]
        results = torch.zeros(x.shape[0], x.shape[1], self.output_dim, dtype=x.dtype, device=x.device) # [bsz, seqlen, output_dim]

        batch_idx = torch.where(type_weight)

        selected_x = x[batch_idx] # [m, dim] ,m tokens selected for this expert
        selected_results = self.expert(selected_x)

        if len(batch_idx[0])>0:
            results[batch_idx] += type_weight[batch_idx].unsqueeze(-1) * selected_results
        
        return results

