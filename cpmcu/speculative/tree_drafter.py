from .. import C
from ..llm import LLM

import torch
import torch.nn.functional as F
import time
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

def pack_mask(mask_2d):
    '''
    for static masks, pack them into a int64 per row
    '''
    mask_2d_packed = torch.zeros((mask_2d.shape[0], 2), dtype=torch.uint32, device="cuda")
    for i in range(mask_2d.shape[0]):
        mask_1 = 0
        mask_2 = 0
        for j in range(i + 1):
            if j < 32:
                mask_1 |= (mask_2d[i][j].item() << j)
            else:
                mask_2 |= (mask_2d[i][j].item() << (j - 32))
        mask_2d_packed[i][0] = mask_1
        mask_2d_packed[i][1] = mask_2
    mask_2d_packed = mask_2d_packed.view(torch.int64).view(-1)
    return mask_2d_packed

class LLM_with_tree_drafter(LLM):
    def __init__(self,
                 drafter_type, drafter_path, base_path,
                 tree_size,
                 use_rope: bool=False,
                 **kwargs):
        super().__init__(base_path, **kwargs)

        self.drafter_type = drafter_type
        self.drafter_path = drafter_path
        self.base_path = base_path
        self.use_rope = use_rope

        self.tree_size = tree_size
        self.tree_draft_ids = torch.empty((tree_size), dtype=torch.int32, device="cuda")
        self.tree_position_ids = torch.empty((tree_size), dtype=torch.int32, device="cuda")
        self.tree_gt_ids = torch.empty((tree_size), dtype=torch.int32, device="cuda")
        self.tree_attn_mask = torch.empty((tree_size), dtype=torch.int64, device="cuda")
        self.tree_parent = torch.empty((tree_size), dtype=torch.int32, device="cuda")
        self.tree_position_ids = torch.empty((tree_size), dtype=torch.int32, device="cuda")

        self.cache_length = torch.tensor([0], dtype=torch.int32, device="cuda")

    def load_from_hf(self):
        with torch.no_grad():
            self._load_from_ckpt(self.drafter_path, cls=self.drafter_type)

            if self.use_rope:
                if hasattr(self.config, "rope_scaling") and self.config.rope_scaling is not None:
                    rope_type = self.config.rope_scaling.get("rope_type", self.config.rope_scaling.get("type"))
                else:
                    rope_type = "default"
                # TODO only support "default", "llama3" or "longrope" with long_factor=short_factor
                inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](self.config, "cpu", seq_len=self.max_total_length)
                # attention_scaling = torch.tensor([attention_scaling], dtype=torch.float32, device="cpu")
                self._load(f"{self.drafter_type}.rotary_emb.inv_freq", inv_freq, dtype=torch.float32)
                # self._load("model.rotary_emb.attention_scaling", attention_scaling, dtype=torch.float32)

            super().load_from_hf()

    def generate(self, input_ids, generation_length=100, teminators=[], use_stream=False, progress_callback=None):
        """
        Generate text with optional streaming output for tree drafter.
        Returns (tokens, accept_lengths, decode_time, prefill_time) if use_stream=False, or generator yielding {'token', 'text', 'is_finished', 'accept_length', 'prefill_time', 'decode_time'} if use_stream=True.
        """
        assert input_ids.dtype == torch.int32

        prefix_length = input_ids.numel()
        # Check if input length exceeds maximum supported length
        if prefix_length > self.max_total_length:
            raise ValueError(f"Input token count ({prefix_length}) exceeds maximum supported length ({self.max_total_length}) under current memory limit")
        
        position_ids = torch.arange(prefix_length, dtype=torch.int32, device="cuda")
        
        # Measure prefill time
        torch.cuda.synchronize()
        prefill_start = time.time()
        logits = self.prefill(input_ids, position_ids, progress_callback)
        torch.cuda.synchronize()
        prefill_time = time.time() - prefill_start
        
        if self.temperature > 0.0:
            self.tree_draft_ids[:1].copy_(torch.multinomial(F.softmax(logits[0]/self.temperature, dim=-1), num_samples=1, generator=self.generator))
        else:
            self.tree_draft_ids[:1].copy_(logits[0].argmax(dim=-1))

        if use_stream:
            # Stream generation for tree drafter (optimized)
            def _stream_generator():
                # Keep minimal context for correct spacing
                prev_token = None
                
                # yield first token
                token = self.tree_draft_ids[0].item()
                text = self.tokenizer.decode([token], skip_special_tokens=True)
                prev_token = token
                
                yield {
                    'token': token,
                    'text': text,
                    'is_finished': token in teminators,
                    'accept_length': 1,
                    'prefill_time': prefill_time,
                    'decode_time': 0.0  # First token comes from prefill
                }
                
                if token in teminators:
                    return

                decode_start_time = time.time()

                i = 0
                while i < generation_length-1:
                    self.cache_length[0] = prefix_length + i

                    # draft step
                    C.draft(self.tree_draft_ids.data_ptr(), self.tree_position_ids.data_ptr(), self.cache_length.data_ptr(), self.tree_attn_mask.data_ptr(), self.tree_parent.data_ptr())

                    logits = self.decode(self.tree_draft_ids, self.tree_position_ids, self.cache_length, mask_2d=self.tree_attn_mask)
                    if self.temperature > 0.0:
                        self.tree_gt_ids.copy_(torch.multinomial(F.softmax(logits/self.temperature, dim=-1), num_samples=1, generator=self.generator).squeeze(-1))
                    else:
                        self.tree_gt_ids.copy_(logits.argmax(dim=-1))

                    # verify step
                    accept_length = C.verify_and_fix(
                        self.tree_draft_ids.numel(), self.tree_draft_ids.data_ptr(), self.tree_gt_ids.data_ptr(),
                        self.tree_position_ids.data_ptr(), self.cache_length.data_ptr(),
                        self.tree_attn_mask.data_ptr(), self.tree_parent.data_ptr()
                    )

                    # yield accepted tokens (optimized with minimal context)
                    if accept_length > 0:
                        accepted_tokens = self.tree_draft_ids[:accept_length].tolist()
                        
                        # For correct spacing, decode with previous token context
                        if prev_token is not None:
                            context_tokens = [prev_token] + accepted_tokens
                            context_text = self.tokenizer.decode(context_tokens, skip_special_tokens=True)
                            prev_text = self.tokenizer.decode([prev_token], skip_special_tokens=True)
                            new_text = context_text[len(prev_text):]
                        else:
                            new_text = self.tokenizer.decode(accepted_tokens, skip_special_tokens=True)
                        
                        # Yield tokens with batch text for first token, empty for others
                        for j in range(accept_length):
                            if i + j >= generation_length - 1:
                                break
                                
                            token = accepted_tokens[j]
                            
                            # Give all new text to first token, empty to others
                            if j == 0:
                                text = new_text
                            else:
                                text = ""
                            
                            terminal = token in teminators
                            is_finished = terminal or (i + j == generation_length - 2)
                            
                            # Only calculate time for the last token in the batch to reduce overhead
                            decode_time = time.time() - decode_start_time if j == accept_length - 1 else 0.0
                            
                            yield {
                                'token': token,
                                'text': text,
                                'is_finished': is_finished,
                                'accept_length': accept_length if j == 0 else 0,  # only report accept_length for first token in batch
                                'prefill_time': 0.0,  # Only report prefill_time for first token
                                'decode_time': decode_time
                            }
                            
                            if terminal:
                                return
                        
                        # Update prev_token to the last accepted token
                        prev_token = accepted_tokens[-1]

                    self.tree_draft_ids[0] = self.tree_draft_ids[accept_length - 1]
                    i += accept_length
                    
            return _stream_generator()
        else:
            # Original batch generation
            tokens = torch.empty((generation_length), dtype=torch.int32, device="cuda")
            tokens[0].copy_(self.tree_draft_ids[0])
            accept_lengths = []
            i = 0
            terminal = False
            torch.cuda.synchronize()
            decode_start = time.time()
            while i < generation_length-1 and not terminal:
                self.cache_length[0] = prefix_length + i

                # torch.cuda.nvtx.range_push(f"draft")
                C.draft(self.tree_draft_ids.data_ptr(), self.tree_position_ids.data_ptr(), self.cache_length.data_ptr(), self.tree_attn_mask.data_ptr(), self.tree_parent.data_ptr())
                # torch.cuda.nvtx.range_pop()

                logits = self.decode(self.tree_draft_ids, self.tree_position_ids, self.cache_length, mask_2d=self.tree_attn_mask)
                if self.temperature > 0.0:
                    self.tree_gt_ids.copy_(torch.multinomial(F.softmax(logits/self.temperature, dim=-1), num_samples=1, generator=self.generator).squeeze(-1))
                else:
                    self.tree_gt_ids.copy_(logits.argmax(dim=-1))

                # torch.cuda.nvtx.range_push(f"verify")
                accept_length = C.verify_and_fix(
                    self.tree_draft_ids.numel(), self.tree_draft_ids.data_ptr(), self.tree_gt_ids.data_ptr(),
                    self.tree_position_ids.data_ptr(), self.cache_length.data_ptr(),
                    self.tree_attn_mask.data_ptr(), self.tree_parent.data_ptr()
                )
                # torch.cuda.nvtx.range_pop()

                accept_lengths.append(accept_length)
                for temin in teminators:
                    if temin in self.tree_draft_ids[:accept_length]:
                        terminal = True
                append_length = min(accept_length, generation_length - 1 - i)
                tokens[1+i:1+i+append_length].copy_(self.tree_draft_ids[:append_length])
                self.tree_draft_ids[0] = self.tree_draft_ids[accept_length - 1]
                i += accept_length
            torch.cuda.synchronize()
            decode_time = time.time() - decode_start
            tokens = tokens[:1+i].tolist()

            return tokens, accept_lengths, decode_time, prefill_time