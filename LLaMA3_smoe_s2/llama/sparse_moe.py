import torch
import torch.nn as nn
from torch.nn import functional as F
# from torch.nn import init

class LoraLayer(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, r=None, lora_alpha:float=32):
        super().__init__()

        self.scaling = lora_alpha / r

        self.lora_A = nn.Linear(input_dim, r, bias=False)
        self.lora_B = nn.Linear(r, output_dim, bias=False)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor):
        
        return self.lora_B(self.lora_A(x)) * self.scaling


class PAdapterLayer(nn.Module):
    def __init__(self, hidden_size=None, adapter_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size

        self.adapter_act_fn = nn.SiLU()
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.down_proj.weight, gain=1e-4)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=1e-4)
        nn.init.constant_(self.down_proj.bias, 0.0)
        nn.init.constant_(self.up_proj.bias, 0.0)

    def forward(self, x):
        x = self.up_proj(self.adapter_act_fn(self.down_proj(x)))
        return x


class LoraLayerB(nn.Module):
    def __init__(self, output_dim, r, lora_alpha:float=32):
        super().__init__()

        self.scaling = lora_alpha / r

        self.lora_B = nn.Linear(r, output_dim, bias=False)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor):
        
        return self.lora_B(x) * self.scaling


class PAdapterLayerB(nn.Module):
    def __init__(self, hidden_size, adapter_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size

        self.adapter_act_fn = nn.SiLU()
        
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.up_proj.weight, gain=1e-4)
        nn.init.constant_(self.up_proj.bias, 0.0)

    def forward(self, x):
        x = self.up_proj(self.adapter_act_fn(x))
        return x
    

#noisy top-k gating
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, noisy=False):
        super().__init__()
        self.top_k = top_k
        #layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noisy = noisy
        if self.noisy:
            self.noise_linear =nn.Linear(n_embed, num_experts)


    def forward(self, x: torch.Tensor):
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
class TopkMoE(nn.Module):
    def __init__(self, num_experts=2, top_k=2, expert_type='lora', noisy_router=False, lb_loss=False, asym=False, input_dim=None, output_dim=None, lora_r=None, lora_alpha=None, adapter_size=None):
        super().__init__()
        self.lb_loss = lb_loss
        self.top_k = top_k
        self.asym = asym

        if num_experts <= 1:
            raise Exception('num_experts less than 2')
        
        if not output_dim:
            self.output_dim = input_dim
        else:
            self.output_dim = output_dim

        self.router = NoisyTopkRouter(input_dim, num_experts, top_k, noisy=noisy_router)

        if expert_type == 'lora':
            if self.asym:
                self.A = nn.Linear(input_dim, lora_r, bias=False)
                self.experts = nn.ModuleList([LoraLayerB(output_dim, lora_r, lora_alpha) for _ in range(num_experts)])
            else:
                self.experts = nn.ModuleList([LoraLayer(input_dim, output_dim, lora_r, lora_alpha) for _ in range(num_experts)])
        else:
            if self.asym:
                self.A = nn.Linear(input_dim, adapter_size)
                nn.init.xavier_uniform_(self.A.weight, gain=1e-4)
                nn.init.constant_(self.A.bias, 0.0)
                self.experts = nn.ModuleList([PAdapterLayerB(input_dim, adapter_size=adapter_size) for _ in range(num_experts)])
            else:
                self.experts = nn.ModuleList([PAdapterLayer(input_dim, adapter_size=adapter_size) for _ in range(num_experts)])

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
        # x [seleteced token len, dim]
        # Reshape inputs for batch processing
        # flat_x = x.view(-1, x.size(-1))
        flat_x = x

        weighted_top_k_logits, selected_experts, softmax_logits = self.router(flat_x)

        # results = torch.zeros(x.shape[0]*x.shape[1], self.output_dim, dtype=x.dtype, device=x.device)
        results = torch.zeros(x.shape[0], self.output_dim, dtype=x.dtype, device=x.device)

        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            if self.asym:
                expert_results = expert(self.A(flat_x[batch_idx]))
            else:
                expert_results = expert(flat_x[batch_idx])
            results[batch_idx] += weighted_top_k_logits[batch_idx, nth_expert, None] * expert_results

        # results = results.view((*x.shape[:-1], results.shape[-1]))

        if self.lb_loss and self.training:
            self.load_balancing_loss = self.get_lb_loss(gate_logits=softmax_logits, selected_experts=selected_experts)

        return results
    

# 根据threshold_fn 决定每个token的expert数量
class AdaMoLE(nn.Module):
    def __init__(self, num_experts=2, expert_type='lora', noisy_router=False, lb_loss=False, asym=False, input_dim=None, output_dim=None, lora_r=None, lora_alpha=None, adapter_size=None):
        super().__init__()

        self.lb_loss = lb_loss
        self.asym = asym

        if num_experts <= 1:
            raise Exception('num_experts less than 2')
        
        if not output_dim:
            self.output_dim = input_dim
        else:
            self.output_dim = output_dim

        self.router = nn.Linear(input_dim, num_experts)
        self.noisy = noisy_router
        if self.noisy:
            self.noise_linear =nn.Linear(input_dim, num_experts)

        if expert_type == 'lora':
            if self.asym:
                self.A = nn.Linear(input_dim, lora_r, bias=False)
                self.experts = nn.ModuleList([LoraLayerB(output_dim, lora_r, lora_alpha) for _ in range(num_experts)])
            else:
                self.experts = nn.ModuleList([LoraLayer(input_dim, output_dim, lora_r, lora_alpha) for _ in range(num_experts)])
        else:
            if self.asym:
                self.A = nn.Linear(input_dim, adapter_size)
                nn.init.xavier_uniform_(self.A.weight, gain=1e-4)
                nn.init.constant_(self.A.bias, 0.0)
                self.experts = nn.ModuleList([PAdapterLayerB(input_dim, adapter_size=adapter_size) for _ in range(num_experts)])
            else:
                self.experts = nn.ModuleList([PAdapterLayer(input_dim, adapter_size=adapter_size) for _ in range(num_experts)])

        self.max_threshold = 1 / num_experts
        self.threshold_fn = nn.Linear(input_dim, 1)

    def get_lb_loss(self, gate_logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        """
        Get the load balancing loss by following the Switch Transformer
        """
        num_inputs = gate_logits.shape[0]
        num_experts = len(self.experts)
        expert_counts = torch.sum(selected_experts, dim=0)
        expert_fractions = expert_counts / num_inputs
        expert_probs = torch.sum(gate_logits, dim=0) / num_inputs
        layer_loss = num_experts * torch.sum(expert_fractions * expert_probs)
        return layer_loss

    def forward(self, inputs: torch.Tensor):
        # Reshape inputs for batch processing
        # flattened_inputs = inputs.view(-1, inputs.size(-1))
        flattened_inputs = inputs

        logits = self.router(flattened_inputs)
        if self.noisy:
            #Noise logits
            noise_logits = self.noise_linear(flattened_inputs)
            #Adding scaled unit gaussian noise to the logits
            noise = torch.randn_like(logits)*F.softplus(noise_logits)
            logits = logits + noise

        gate_logits = F.softmax(logits, dim=-1)
        thresholds = F.sigmoid(self.threshold_fn(flattened_inputs)) * self.max_threshold
        adapted_gate_logits = gate_logits - thresholds
        selected_experts = torch.ge(adapted_gate_logits, 0).to(torch.float)
        weights = adapted_gate_logits * selected_experts
        weight_sums = torch.sum(weights, dim=-1, keepdim=True, dtype=inputs.dtype)
        weight_sums = torch.where(weight_sums == 0, torch.ones_like(weight_sums), weight_sums)
        weights = weights / weight_sums

        results = torch.zeros(inputs.shape[0], self.output_dim, dtype=inputs.dtype, device=inputs.device)

        for i, expert in enumerate(self.experts):
            batch_idx = torch.where(selected_experts[:, i])[0]
            if len(batch_idx) > 0:
                if self.asym:
                    expert_results = expert(self.A(flattened_inputs[batch_idx]))
                else:
                    expert_results = expert(flattened_inputs[batch_idx])
                results[batch_idx] += weights[batch_idx, i, None] * expert_results

        # results = results.view((*inputs.shape[:-1], results.shape[-1]))
        
        if self.lb_loss and self.training:
            self.load_balancing_loss = self.get_lb_loss(gate_logits=adapted_gate_logits, selected_experts=selected_experts)
        return results
    

