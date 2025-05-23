import os
import json
from pathlib import Path

import torch
import torch.nn as nn

from .llama import ModelArgs, Transformer
from .tokenizer import Tokenizer
from .utils import sample_top_p

from typing import List

import time

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
            expert_num = args.expert_num,
            swi_x= args.swi_x,
            hydra_moe = args.hydra_moe,
            p_adapter_layers = args.p_adapter_layers,
            p_adapter_size = args.p_adapter_size,
            p_adapter_hydra = args.p_adapter_hydra,
            prompt_layers = args.prompt_layers,
            prompt_len = args.prompt_len,
            expert_weight = args.expert_weight,
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


    @torch.inference_mode()
    def generate_time(
        self, 
        prompts,
        max_gen_len: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.75,
        time_gen: bool = False
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

        if time_gen:
            prefill_time = None
            end_time = None
            start_time = time.time()

        for cur_pos in range(start_pos, total_len):
            if params.bf16:
                dt = torch.bfloat16
            else:
                dt = torch.float16
            with torch.cuda.amp.autocast(dtype=dt):
                logits = self.forward_inference(tokens[:, prev_pos:cur_pos], prev_pos)

            # prefill time
            if time_gen and cur_pos == start_pos:
                prefill_time = time.time()

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

        time_cost = None
        if time_gen: # batch time cost, set batch_size to 1 
            end_time = time.time()
            all_cost = end_time - start_time  
            inference_cost = end_time - prefill_time 
            time_cost = {'all_cost':all_cost, 'inference_cost':inference_cost}

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

        return decoded, time_cost

    # router 统计信息
    @torch.inference_mode()
    def forward_inference_router_stat(self, tokens, start_pos: int, layer_stat):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
        mask = None  
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h) 
        
        for i,layer in enumerate(self.llama.layers):
            h, sum_weights = layer(h, start_pos, freqs_cis, mask)
            if start_pos == 0:
                layer_stat[i] = {}
                layer_stat[i]['sum_weights'] =  sum_weights
            else:
                layer_stat[i]['sum_weights'] +=  sum_weights

        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])

        return output.float(), layer_stat


    @torch.inference_mode()
    def generate_router_stat(
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

        layer_stat = {'token_num':0}
        for cur_pos in range(start_pos, total_len):
            if params.bf16:
                dt = torch.bfloat16
            else:
                dt = torch.float16
            with torch.cuda.amp.autocast(dtype=dt):
                logits, layer_stat = self.forward_inference_router_stat(tokens[:, prev_pos:cur_pos], prev_pos, layer_stat)

            layer_stat['token_num'] += bsz * (cur_pos-prev_pos)

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

        return decoded, layer_stat
    
    # router 案例具体分析
    @torch.inference_mode()
    def forward_inference_router_case(self, tokens, start_pos: int, layer_stat):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
        mask = None  
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h) 
        
        for i,layer in enumerate(self.llama.layers):
            h, weights = layer(h, start_pos, freqs_cis, mask)
            if start_pos == 0:
                layer_stat[i] = {}
                layer_stat[i]['weights'] =  weights.clone()
            else:
                layer_stat[i]['weights'] = torch.cat((layer_stat[i]['weights'], weights), dim=1)

        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])

        return output.float(), layer_stat


    @torch.inference_mode()
    def generate_router_case(
        self, 
        prompts,
        max_gen_len: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.75
    ):
        bsz = len(prompts)
        params = self.llama.params
        assert bsz == 1

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

        layer_stat = {}
        for cur_pos in range(start_pos, total_len):
            if params.bf16:
                dt = torch.bfloat16
            else:
                dt = torch.float16
            with torch.cuda.amp.autocast(dtype=dt):
                logits, layer_stat = self.forward_inference_router_case(tokens[:, prev_pos:cur_pos], prev_pos, layer_stat)

            # layer_stat['token_num'] += bsz * (cur_pos-prev_pos)

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
            t0 = t[:]
            # cut to max gen len
            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t0 = t0[: t0.index(self.tokenizer.eos_id)]
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
            print(self.tokenizer.eos_id)
            print(self.tokenizer.pad_id)
            print(t0)
            words = [self.tokenizer.decode([ti]) for ti in t0]
            layer_stat['words'] = words
            assert len(words) == layer_stat[0]['weights'].shape[1]
        return decoded, layer_stat
