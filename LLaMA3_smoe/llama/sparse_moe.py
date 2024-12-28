import torch
import torch.nn as nn
from torch.nn import functional as F
# from torch.nn import init

#noisy top-k gating
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        #layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear =nn.Linear(n_embed, num_experts)


    def forward(self, x: torch.Tensor):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(x)

        #Noise logits
        noise_logits = self.noise_linear(x)

        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        # top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1) # topk first, then softmax
        # zeros = torch.full_like(noisy_logits, float('-inf'))
        # sparse_logits = zeros.scatter(-1, indices, top_k_logits) # 
        # router_output = F.softmax(sparse_logits, dim=-1)

        # to use the softmax logits compute load_balancing loss
        softmax_logits = F.softmax(noisy_logits, dim=-1)
        top_k_logits, indices = softmax_logits.topk(self.top_k, dim=-1)
        weighted_top_k_logits = top_k_logits / torch.sum(top_k_logits, dim=-1, keepdim=True, dtype=x.dtype)
        return weighted_top_k_logits, indices, softmax_logits


#Now create the sparse mixture of experts module
class SparseLoRAMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, lora_r, lora_alpha):
        super(SparseLoRAMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([LoraLayer(n_embed, n_embed, lora_r, lora_alpha) for _ in range(num_experts)])
        self.top_k = top_k

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

    def forward(self, x):
        gating_output, indices, softmax_logits = self.router(x)
        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)
        
        if self.training:
            self.load_balancing_loss = self.get_lb_loss(gate_logits=softmax_logits.view(-1, softmax_logits.size(-1)), selected_experts=indices)

        return final_output
    


class LoraLayer(nn.Module):
    def __init__(self, input_dim, output_dim, r, lora_alpha:float=8):
        super().__init__()

        self.scaling = lora_alpha / r

        self.lora_A = nn.Linear(input_dim, r, bias=False)
        self.lora_B = nn.Linear(r, output_dim, bias=False)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor):
        
        return self.lora_B(self.lora_A(x)) * self.scaling