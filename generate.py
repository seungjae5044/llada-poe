# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM

# ========================
# Top-K Memory Bank + PoE
# ========================

class TopKMemoryBank:
    """Sparse per-position memory for previous-step distributions.

    Stores top-K log-probabilities, indices, and entropy for still-masked positions.
    Provides an apply() hook to perform sparse log-space PoE on current logits.
    """

    def __init__(self, k: int = 10):
        self.k = int(k)
        self._values = None      # (B, L, K) float32
        self._indices = None     # (B, L, K) long
        self._entropy = None     # (B, L) float32
        self._valid = None       # (B, L) bool

    def _ensure(self, b: int, l: int, device):
        import torch
        if (
            self._values is None
            or self._values.shape[0] != b
            or self._values.shape[1] != l
            or self._values.device != device
        ):
            self._values = torch.zeros((b, l, self.k), dtype=torch.float32, device=device)
            self._indices = torch.zeros((b, l, self.k), dtype=torch.long, device=device)
            self._entropy = torch.zeros((b, l), dtype=torch.float32, device=device)
            self._valid = torch.zeros((b, l), dtype=torch.bool, device=device)

    def update(self, logits, mask_index, slice_offset: int = 0):
        import torch, torch.nn.functional as F
        b, l2, v = logits.shape
        device = logits.device
        self._ensure(b, slice_offset + l2, device)
        flat_mask = mask_index.view(-1)
        if flat_mask.any():
            logits_flat = logits.reshape(-1, v)
            sel_logits = logits_flat[flat_mask]
            log_probs = F.log_softmax(sel_logits.to(torch.float32), dim=-1)
            k_eff = min(self.k, v)
            vals, idx = torch.topk(log_probs, k=k_eff, dim=-1)
            if k_eff < self.k:
                pad = self.k - k_eff
                vals = torch.cat([vals, vals.new_full((vals.size(0), pad), float('-inf'))], dim=-1)
                idx = torch.cat([idx, idx.new_full((idx.size(0), pad), 0)], dim=-1)
            probs_topk = torch.softmax(vals, dim=-1)
            H = -(probs_topk * torch.log(torch.clamp(probs_topk, min=1e-12))).sum(dim=-1)
            pos_idx = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
            batch_ids = pos_idx // l2
            local_pos = pos_idx % l2
            abs_pos = local_pos + slice_offset
            self._values[batch_ids, abs_pos] = vals
            self._indices[batch_ids, abs_pos] = idx
            self._entropy[batch_ids, abs_pos] = H
            self._valid[batch_ids, abs_pos] = True
        # invalidate non-masked in this slice
        if (~mask_index).any():
            b_ids, pos_ids = torch.where(~mask_index)
            abs_pos_ids = pos_ids + slice_offset
            self._valid[b_ids, abs_pos_ids] = False

    def apply_poe(self, logits, mask_index, t_frac: float, alpha_base: float, slice_offset: int = 0, invert: bool = False):
        import torch, numpy as np
        if self._values is None:
            return logits
        b, l2, v = logits.shape
        device = logits.device
        self._ensure(b, slice_offset + l2, device)
        valid_slice = self._valid[:, slice_offset : slice_offset + l2]
        pos_mask = mask_index & valid_slice
        if not pos_mask.any():
            return logits
        vals = self._values[:, slice_offset : slice_offset + l2]
        idxs = self._indices[:, slice_offset : slice_offset + l2]
        H = self._entropy[:, slice_offset : slice_offset + l2]
        centered = vals - vals.mean(dim=-1, keepdim=True)
        denom = max(1.0, float(np.log(max(1, self.k))))
        if invert:
            c = (H / denom).clamp(min=0.0, max=1.0)
        else:
            c = (1.0 - (H / denom)).clamp(min=0.0, max=1.0)
        alpha_eff = float(alpha_base) * float(max(0.0, min(1.0, t_frac)))
        alpha = (c * alpha_eff).unsqueeze(-1)
        logits_flat = logits.view(-1, v)
        pos_mask_flat = pos_mask.view(-1)
        if pos_mask_flat.any():
            rows = torch.nonzero(pos_mask_flat, as_tuple=False).squeeze(1)
            sel_logits = logits_flat[rows]
            sel_alpha = alpha.view(-1, 1)[pos_mask_flat].view(-1, 1)
            sel_centered = centered.view(-1, self.k)[pos_mask_flat]
            sel_indices = idxs.view(-1, self.k)[pos_mask_flat]
            # Fix dtype mismatch: convert src to match sel_logits dtype
            src = (sel_centered * sel_alpha).to(sel_logits.dtype)
            sel_logits = sel_logits.scatter_add(dim=1, index=sel_indices, src=src)
            logits_flat[rows] = sel_logits
        return logits

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None, reuse_topk: bool = False, topk_k: int = 10, alpha_base: float = 0.3, invert: bool = False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    total_steps = steps
    steps = steps // num_blocks

    bank = TopKMemoryBank(k=topk_k) if reuse_topk else None

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            raw_logits = model(x).logits
            logits = raw_logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if bank is not None:
                t_frac = float(num_block * steps + i) / float(max(1, total_steps))
                logits = bank.apply_poe(logits, mask_index, t_frac=t_frac, alpha_base=alpha_base, slice_offset=0, invert=invert)
            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            if bank is not None:
                next_mask = (x == mask_id)
                update_mask = next_mask.clone()
                update_mask[:, : prompt.shape[1] + num_block * block_length] = False
                update_mask[:, prompt.shape[1] + (num_block + 1) * block_length :] = False
                bank.update(raw_logits, update_mask, slice_offset=0)
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe



@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None, reuse_topk: bool = False, topk_k: int = 10, alpha_base: float = 0.3, invert: bool = False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    total_steps = steps
    steps = steps // num_blocks

    bank = TopKMemoryBank(k=topk_k) if reuse_topk else None

    nfe = 0
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        raw_logits = output.logits
        logits = raw_logits
        if bank is not None:
            t_frac = float(num_block * steps + 0) / float(max(1, total_steps))
            logits = bank.apply_poe(logits, mask_index, t_frac=t_frac, alpha_base=alpha_base, slice_offset=0, invert=invert)
        if factor is None:
            x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]
        if bank is not None:
            next_mask = (x == mask_id)
            update_mask = next_mask.clone()
            update_mask[:, : current_block_start] = False
            update_mask[:, current_block_end: ] = False
            bank.update(raw_logits, update_mask, slice_offset=0)

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            raw_logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits = raw_logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                if bank is not None:
                    t_frac = float(num_block * steps + i) / float(max(1, total_steps))
                    logits = bank.apply_poe(logits, mask_index, t_frac=t_frac, alpha_base=alpha_base, slice_offset=current_block_start, invert=invert)
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index,
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                if bank is not None:
                    t_frac = float(num_block * steps + i) / float(max(1, total_steps))
                    logits = bank.apply_poe(logits, mask_index, t_frac=t_frac, alpha_base=alpha_base, slice_offset=current_block_start, invert=invert)
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index,
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            if bank is not None:
                next_mask = (x[:, current_block_start:current_block_end] == mask_id)
                update_mask = torch.zeros_like(mask_index)
                update_mask[:, : block_length] = next_mask
                bank.update(raw_logits, update_mask, slice_offset=current_block_start)
            
            i += 1


    return x, nfe


@ torch.no_grad()
def generate_with_dual_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=None, factor=None, reuse_topk: bool = False, topk_k: int = 10, alpha_base: float = 0.3, invert: bool = False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    total_steps = steps
    steps = steps // num_blocks

    bank = TopKMemoryBank(k=topk_k) if reuse_topk else None

    nfe = 0  
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # cache init and update
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        raw_logits = output.logits
        logits = raw_logits
        if bank is not None:
            t_frac = float(num_block * steps + 0) / float(max(1, total_steps))
            logits = bank.apply_poe(logits, mask_index, t_frac=t_frac, alpha_base=alpha_base, slice_offset=0, invert=invert)
        if factor is None:
            x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]
        if bank is not None:
            next_mask = (x == mask_id)
            update_mask = next_mask.clone()
            update_mask[:, : current_block_start] = False
            update_mask[:, current_block_end: ] = False
            bank.update(raw_logits, update_mask, slice_offset=0)
        nfe += 1

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            # cache position is the position between current_block_start and current_block_end
            raw_logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values, use_cache=True, replace_position=replace_position).logits

            logits = raw_logits
            if factor is None:
                if bank is not None:
                    t_frac = float(num_block * steps + i) / float(max(1, total_steps))
                    logits = bank.apply_poe(logits, mask_index, t_frac=t_frac, alpha_base=alpha_base, slice_offset=current_block_start, invert=invert)
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index,
                                                x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                if bank is not None:
                    t_frac = float(num_block * steps + i) / float(max(1, total_steps))
                    logits = bank.apply_poe(logits, mask_index, t_frac=t_frac, alpha_base=alpha_base, slice_offset=current_block_start, invert=invert)
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index,
                                                x[:, current_block_start:current_block_end], None, factor)
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            if bank is not None:
                next_mask = (x[:, current_block_start:current_block_end] == mask_id)
                bank.update(raw_logits, next_mask, slice_offset=current_block_start)
            i += 1

    return x, nfe


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

def main():
    device = 'cuda'

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
