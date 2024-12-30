import torch
import torch.nn as nn
from torch.nn import functional as F
# from torch.nn import init

#noisy top-k gating
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, noisy=False):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        #layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noisy = noisy
        if self.noisy:
            self.noise_linear =nn.Linear(n_embed, num_experts)


    def forward(self, x: torch.Tensor):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(x)
        if self.noisy:
            #Noise logits
            noise_logits = self.noise_linear(x)
            #Adding scaled unit gaussian noise to the logits
            noise = torch.randn_like(logits)*F.softplus(noise_logits)
            logits = logits + noise

        # to use the softmax logits compute load_balancing loss
        softmax_logits = F.softmax(logits, dim=-1)
        top_k_logits, selected_experts = softmax_logits.topk(self.top_k, dim=-1)
        weighted_top_k_logits = top_k_logits / torch.sum(top_k_logits, dim=-1, keepdim=True, dtype=x.dtype)
        return weighted_top_k_logits, selected_experts, softmax_logits


#Now create the sparse mixture of experts module
class SparseLoRAMoE(nn.Module):
    def __init__(self, input_dim, output_dim, lora_r, num_experts, lora_alpha, top_k, noisy=False):
        super(SparseLoRAMoE, self).__init__()
        self.router = NoisyTopkRouter(input_dim, num_experts, top_k, noisy=noisy)
        self.experts = nn.ModuleList([LoraLayer(input_dim, output_dim, lora_r, lora_alpha) for _ in range(num_experts)])
        self.top_k = top_k
        self.output_dim = output_dim

    def get_lb_loss(self, gate_logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        """
        Get the load balancing loss by following the Switch Transformer
        """
        num_inputs = gate_logits.shape[0]
        num_experts = len(self.experts)
        expert_counts = torch.bincount(selected_experts.reshape(-1), minlength=num_experts)
        expert_fractions = expert_counts / num_inputs
        expert_probs = torch.sum(gate_logits, dim=0) / num_inputs
        layer_loss = num_experts * torch.sum(expert_fractions * expert_probs)
        return layer_loss

    def forward(self, x:torch.Tensor):

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))

        weighted_top_k_logits, selected_experts, softmax_logits = self.router(flat_x)

        # results = torch.zeros_like(self.experts[0](flat_x)) # todo:  fix 
        results = torch.zeros(x.shape[0]*x.shape[1], self.output_dim, dtype=x.dtype, device=x.device)

        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += \
                weighted_top_k_logits[batch_idx, nth_expert, None] * expert(flat_x[batch_idx])

        results = results.view((*x.shape[:-1], results.shape[-1]))

        if self.training:
            self.load_balancing_loss = self.get_lb_loss(gate_logits=softmax_logits, selected_experts=selected_experts)

        return results
    


class LoraLayer(nn.Module):
    def __init__(self, input_dim, output_dim, r, lora_alpha:float=8):
        super().__init__()

        self.scaling = lora_alpha / r

        self.lora_A = nn.Linear(input_dim, r, bias=False)
        self.lora_B = nn.Linear(r, output_dim, bias=False)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor):
        
        return self.lora_B(self.lora_A(x)) * self.scaling