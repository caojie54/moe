import torch
import json
import os
import fire

def main(
        checkpoint: str
):
    directory = os.path.dirname(checkpoint)
    model = torch.load(checkpoint, map_location="cpu")
    new_model = dict()
    # prompt 
    weight_list = []

    # parallel adapter & lora & prompt
    for k in model['model'].keys():
        if 'adapter' in k or 'lora' in k or 'prompt' in k:
            weight_list.append(k)
            # print(model['model'].get(k))

    for i in range(len(weight_list)):
        tensor = model["model"].get(weight_list[i])
        if type(tensor) == torch.Tensor:
            new_model[weight_list[i]] = tensor

    for x in new_model.keys():
        print(x)
    # print(new_model.keys())
    torch.save(new_model, os.path.join(directory, 'adapter.pth'))

    # adapter params
    args = model.get('args')
    adapter_params = {}
    adapter_params['w_bias'] = args.w_bias
    adapter_params['lora_layers'] = args.lora_layers
    adapter_params['lora_rank'] = args.lora_rank
    adapter_params['lora_targets'] = args.lora_targets
    adapter_params['lora_alpha'] = args.lora_alpha
    
    adapter_params['p_adapter_layers'] = args.p_adapter_layers
    adapter_params['p_adapter_size'] = args.p_adapter_size
    
    adapter_params['prompt_layers'] = args.prompt_layers
    adapter_params['prompt_len'] = args.prompt_len

    adapter_params['max_threshold'] = args.max_threshold
    adapter_params['bool_weights'] = args.bool_weights
    adapter_params['swi_x'] = args.swi_x

    adapter_params['num_experts'] = args.num_experts
    adapter_params['moe_type'] = args.moe_type
    adapter_params['top_k'] = args.top_k
    adapter_params['noisy_router'] = args.noisy_router
    adapter_params['lb_loss'] = args.lb_loss
    adapter_params['lb_loss_coeff'] = args.lb_loss_coeff
    adapter_params['asym'] = args.asym

    adapter_params['max_seq_len'] = args.max_seq_len
    adapter_params['flash_attention2'] = args.flash_attention2
    adapter_params['bf16'] = args.bf16

    print(f'adapter params:{adapter_params}')
    with open(os.path.join(directory,'adapter_params.json'), 'w', encoding='utf-8') as f:
        json_data = json.dumps(adapter_params, ensure_ascii=False)
        f.write(json_data)

    # generate_params = {}
    # generate_params['max_seq_len'] = args.max_seq_len

    # print(f'generate params:{generate_params}')
    # with open(os.path.join(directory,'generate_params.json'), 'w', encoding='utf-8') as f:
    #     json_data = json.dumps(generate_params, ensure_ascii=False)
    #     f.write(json_data)


if __name__ == "__main__":
    fire.Fire(main)