import os
import json
from pathlib import Path

import torch
import torch.nn as nn

from .llama import ModelArgs, Transformer
from .tokenizer import Tokenizer
from .utils import sample_top_p

from typing import List


class LLaMA_adapter(nn.Module):

    def __init__(self, args, llama_ckpt_dir, llama_tokenizer):
        super().__init__()
        
        # load llama configs
        with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
            params = json.loads(f.read())
        
        if args.bf16 and torch.cuda.is_bf16_supported():
            bf16=True
        else:
            bf16=False
            if args.bf16:
                print('------bfloat16 is not supported-----')

        model_args: ModelArgs = ModelArgs(
            max_seq_len=args.max_seq_len,
            max_batch_size=args.max_batch_size,
            w_bias = args.w_bias,

            lora_layers = args.lora_layers,
            lora_rank = args.lora_rank,
            lora_targets = args.lora_targets,
            lora_alpha = args.lora_alpha,
            
            p_adapter_layers = args.p_adapter_layers,
            p_adapter_size = args.p_adapter_size,
            
            prompt_layers = args.prompt_layers,
            prompt_len = args.prompt_len,
            
            max_threshold = args.max_threshold,
            bool_weights = args.bool_weights,
            swi_x = args.swi_x,

            num_experts = args.num_experts,
            moe_type = args.moe_type,
            top_k = args.top_k,
            noisy_router = args.noisy_router,
            lb_loss = args.lb_loss,
            lb_loss_coeff = args.lb_loss_coeff,
            asym = args.asym,

            flash_attention2=args.flash_attention2,
            bf16=bf16,
            **params
        ) # max_batch_size only affects inferenc
        self.model_args = model_args

        # 4. tokenizer
        self.tokenizer = Tokenizer(model_path=llama_tokenizer)

        # 5. llama
        assert model_args.vocab_size == self.tokenizer.n_words
        if model_args.bf16:
            torch.set_default_dtype(torch.bfloat16)
            print('-----bfloat16 for llama-----')
        else:
            torch.set_default_dtype(torch.float16)
            print('-------float16 for llama-----')
        torch.set_default_device('cuda')  # loading llama faster with GPU

        self.llama = Transformer(model_args)

        torch.set_default_dtype(torch.float32)
        torch.set_default_device('cpu')

        ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))
        for ckpt in ckpts:
            ckpt = torch.load(ckpt, map_location='cpu')
            self.llama.load_state_dict(ckpt, strict=False)

         # 6. training criterion
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        # 7. training parameters
        self.get_trainable_params()

        for name, param in self.named_parameters():
            if param.requires_grad:
               print(f"Trainable param: {name}, {param.shape}, {param.dtype}")


    def get_aux_loss(self) -> torch.Tensor:
        """
        Get the load balancing loss for the whole model
        """
        # lb_loss = torch.tensor(0, dtype=torch.float).to(self.llama.device)
        lb_loss = torch.tensor(0, dtype=torch.float).cuda()

        for name, module in self.llama.named_modules():
            if hasattr(module, 'get_lb_loss'):
                load_balancing_loss = module.load_balancing_loss
                lb_loss += load_balancing_loss

        return lb_loss
    

    def get_trainable_params(self):
        for name, para in self.named_parameters():
            para.requires_grad = False

        for name, para in self.named_parameters():
            if name.startswith("llama."):
                if self.model_args.w_bias:
                    if 'norm' in name or 'bias' in name:
                        para.data = para.data.float()
                        para.requires_grad = True
                        
                if 'lora' in name or 'prompt' in name or 'adapter' in name:
                    para.data = para.data.float()
                    para.requires_grad = True

    def forward(self, tokens, labels, prompt_mask):

        _bsz, seqlen = tokens.shape

        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)
        
        for layer in self.llama.layers:
            h = layer(h, 0, freqs_cis, mask)

        h = self.llama.norm(h)
        output = self.llama.output(h)
        output = output[:, :-1, :]
        labels = labels[:, 1:]

        if labels.sum() == 0:
            c_loss = output.mean() * 0
        else:
            # assert self.llama.vocab_size == 32000
            c_loss = self.criterion(output.reshape(-1, self.llama.vocab_size), labels.flatten())
            # load balancing loss
            if self.model_args.lb_loss:
                lb_loss = self.get_aux_loss()
                c_loss += self.model_args.lb_loss_coeff * lb_loss

        return c_loss, c_loss

    @torch.inference_mode()
    def forward_inference(self, tokens, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
        mask = None  # TODO: check mask for cache
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h) # TODO: start_pos 0 is ok?

        for layer in self.llama.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])

        return output.float()

    @torch.inference_mode()
    def generate(
        self, 
        prompts,
        max_gen_len: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.75
    ):
        bsz = len(prompts)
        params = self.llama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        # if isinstance(prompts[0], str):
        #     prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompts])
        max_prompt_size = max([len(t) for t in prompts])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()

        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = torch.tensor(t).cuda().long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            if params.bf16:
                dt = torch.bfloat16
            else:
                dt = torch.float16
            with torch.cuda.amp.autocast(dtype=dt):
                logits = self.forward_inference(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # trick: early stop if bsz==1
            if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):

            # cut to max gen len
            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded