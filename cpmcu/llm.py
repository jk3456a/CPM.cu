from . import C

import os, json, glob
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from safetensors.torch import load_file
import time, math
import torch.nn.functional as F
from .common.logging import logger


dtype_map = {
    torch.float16: 0,
    torch.bfloat16: 1,
}

def dtype_to_int(dtype):
    ret = dtype_map.get(dtype, -1)
    if ret == -1:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return ret

def top_p_filtering(logits, top_p, min_tokens_to_keep=1):
    """
    Filter a distribution of logits using nucleus (top-p) filtering.
    
    Args:
        logits: logits distribution shape (batch_size, vocabulary_size)
        top_p: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
        min_tokens_to_keep: minimum number of tokens to keep
    """
    if top_p >= 1.0:
        return logits
    
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
    
    # Scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    
    return logits

class LLM(torch.nn.Module):
    def __init__(self,
                 path: str, # hf model path
                 memory_limit: float = 0.8,
                 chunk_length: int = 1024,
                 dtype: torch.dtype = None,
                 cuda_graph: bool = False,
                 apply_sparse: bool = False,
                 sink_window_size: int = 1,
                 block_window_size: int = 32,
                 sparse_topk_k: int = 32,
                 sparse_switch: int = 8192,
                 use_compress_lse: bool = False,
                 use_qk_norm: bool = False,
                 use_attn_bias: bool = False,
                 temperature: float = 0.0,
                 random_seed: int = None,
                 top_p: float = 1.0,
    ):
        super().__init__()

        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        self.dtype = dtype if dtype is not None else self.config.torch_dtype
        self.dtype_int = dtype_to_int(self.dtype)
        self.cuda_graph = cuda_graph
        self.temperature = temperature
        self.top_p = top_p

        self.chunk_length = chunk_length
        
        # Initialize random generator if random_seed is provided
        if random_seed is not None:
            self.generator = torch.Generator(device="cuda")
            self.generator.manual_seed(random_seed)
        else:
            self.generator = None
        
        # For Qwen3, head_dim is explicitly specified in config and may not equal hidden_size // num_attention_heads
        if not hasattr(self.config, "head_dim"):
            self.config.head_dim = self.config.hidden_size // self.config.num_attention_heads
        else:
            # Qwen3 models have explicit head_dim that might be different
            logger.info(f"Using explicit head_dim from config: {self.config.head_dim}")
        scale_embed = self.config.scale_emb if hasattr(self.config, "scale_emb") else 1.0
        scale_lmhead = (self.config.dim_model_base / self.config.hidden_size) if hasattr(self.config, "dim_model_base") else 1.0
        scale_residual = self.config.scale_depth / math.sqrt(self.config.num_hidden_layers) if hasattr(self.config, "scale_depth") else 1.0

        if apply_sparse:
            C.init_minicpm4_model(
                memory_limit,
                self.config.vocab_size,
                self.config.num_hidden_layers,
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                self.config.rms_norm_eps,
                self.dtype_int,
                self.chunk_length,
                scale_embed,
                scale_lmhead,
                scale_residual,
                sink_window_size,
                block_window_size,
                sparse_topk_k,
                sparse_switch,
                use_compress_lse,
            )
        else:
            C.init_base_model(
                memory_limit,
                self.config.vocab_size,
                self.config.num_hidden_layers,
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                self.config.rms_norm_eps,
                self.dtype_int,
                self.chunk_length,
                scale_embed,
                scale_lmhead,
                scale_residual,
                use_qk_norm,
                use_attn_bias,
            )

        self.logits = torch.empty((64, self.config.vocab_size), dtype=self.dtype, device="cuda")

    def init_storage(self):
        self.max_total_length = C.init_storage()

    def _load(self, name, param, dtype=None, cls=None):
        if dtype is None:
            if 'rotary_emb' in name:
                dtype = torch.float32
            else:
                dtype = self.dtype

        if 'gate_up_proj' in name:
            self._load(name.replace("gate_up_proj", "gate_proj"), param[:param.shape[0]//2], dtype)
            self._load(name.replace("gate_up_proj", "up_proj"), param[param.shape[0]//2:])
        elif 'qkv_proj' in name:
            self._load(name.replace("qkv_proj", "q_proj"), param[:self.config.num_attention_heads * self.config.head_dim])
            self._load(name.replace("qkv_proj", "k_proj"), param[self.config.num_attention_heads * self.config.head_dim:(self.config.num_attention_heads + self.config.num_key_value_heads) * self.config.head_dim])
            self._load(name.replace("qkv_proj", "v_proj"), param[(self.config.num_attention_heads + self.config.num_key_value_heads) * self.config.head_dim:])
        else:
            param = param.contiguous().to(dtype)
            C.load_model(name, param.data_ptr())

        if "embed_tokens" in name and hasattr(self.config, "tie_word_embeddings") and self.config.tie_word_embeddings:
            self._load("lm_head.weight", param)

    def _load_from_ckpt(self, path, cls=None):
        supported_suffix_1 = ["bin.index.json", "safetensors.index.json"]
        supported_suffix_2 = ["bin", "safetensors", "pt"]
        file = None
        for suffix in supported_suffix_1:
            files = glob.glob(os.path.join(path, f"*.{suffix}"))
            if len(files) > 1:
                raise ValueError(f"Multiple files with suffix {suffix} found in {path}")
            elif len(files) == 1:
                file = files[0]
                break
        else:
            for suffix in supported_suffix_2:
                files = glob.glob(os.path.join(path, f"*.{suffix}"))
                if len(files) > 1:
                    raise ValueError(f"Multiple files with suffix {suffix} found in {path}")
                elif len(files) == 1:
                    file = files[0]
                    break
            else:
                raise ValueError(f"No supported checkpoint file found in {path}, supported suffixes: {supported_suffix_1 + supported_suffix_2}")

        if file.endswith(".index.json"):
            with open(file, "r") as f:
                file_list = set(json.load(f)["weight_map"].values())
            file_list = [os.path.join(path, file) for file in file_list]
        else:
            file_list = [file]

        for file in file_list:
            logger.info(f"load from {file}")
            if file.endswith(".bin") or file.endswith(".pt"):
                ckpt = torch.load(file, map_location="cpu")
            elif file.endswith(".safetensors"):
                ckpt = load_file(file)
            for name, param in ckpt.items():
                self._load(name, param, cls=cls)

    def load_from_hf(self):
        with torch.no_grad():
            self._load_from_ckpt(self.path)

            # rope
            if hasattr(self.config, "rope_scaling") and self.config.rope_scaling is not None:
                rope_type = self.config.rope_scaling.get("rope_type", self.config.rope_scaling.get("type"))
                if rope_type == "longrope" and not hasattr(self.config.rope_scaling, "factor"):
                    self.config.rope_scaling["factor"] = 1.0
            else:
                rope_type = "default"
            # TODO only support "default", "llama3" or "longrope" with long_factor=short_factor
            inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](self.config, "cpu", seq_len=self.max_total_length)
            # attention_scaling = torch.tensor([attention_scaling], dtype=torch.float32, device="cpu")
            self._load("model.rotary_emb.inv_freq", inv_freq, dtype=torch.float32)
            # self._load("model.rotary_emb.attention_scaling", attention_scaling, dtype=torch.float32)

    def prefill(self, input_ids, position_ids, progress_callback=None, logits_buffer=None):
        assert input_ids.dtype == torch.int32
        # Check if input length exceeds maximum supported length
        if input_ids.numel() > self.max_total_length:
            raise ValueError(f"Input token count ({input_ids.numel()}) exceeds maximum supported length ({self.max_total_length}) under current memory limit")
        
        total_length = input_ids.numel()
        num_chunks = (total_length + self.chunk_length - 1) // self.chunk_length
        
        actual_prefill_start = time.time()
        
        # Initialize progress callback if provided
        if progress_callback:
            progress_callback('begin', {'total_tokens': total_length})
        
        # Use per-call logits buffer if provided, else fall back to shared buffer
        logits_buf = self.logits if logits_buffer is None else logits_buffer

        for chunk_idx, i in enumerate(range(0, input_ids.numel(), self.chunk_length)):
            # torch.cuda.nvtx.range_push(f"chunk from {i}")
            C.prefill(
                min(input_ids.numel() - i, self.chunk_length), i,
                input_ids.view(-1)[i:].data_ptr(), position_ids.view(-1)[i:].data_ptr(),
                logits_buf.data_ptr()
            )
            # torch.cuda.nvtx.range_pop()
            
            # Update progress via callback
            if progress_callback:
                current_tokens = min(i + self.chunk_length, total_length)
                progress_callback('advance', {'current_tokens': current_tokens})
        
        # Calculate actual prefill time
        actual_prefill_time = time.time() - actual_prefill_start
        
        # Complete progress via callback
        if progress_callback:
            progress_callback('finish', {'total_time': actual_prefill_time})
        
        # Store the actual prefill time for use in generate method
        self._last_prefill_time = actual_prefill_time
        
        return logits_buf[:1].clone()

    def decode(self, input_ids, position_ids, cache_length, mask_2d = None, logits_buffer=None):
        assert input_ids.dtype == torch.int32
        assert position_ids.dtype == torch.int32
        assert cache_length.dtype == torch.int32
        if mask_2d is not None:
            # assert mask_2d.dtype == torch.int64
            assert input_ids.numel() == mask_2d.shape[0]

        # torch.cuda.nvtx.range_push(f"decode")
        cache_length += input_ids.numel() # temparary add for convinience in flash_attn
        padded_length = (cache_length[0].item() + 128 - 1) // 128 * 128
        logits_buf = self.logits if logits_buffer is None else logits_buffer
        C.decode(
            input_ids.numel(), padded_length,
            input_ids.data_ptr(), position_ids.data_ptr(), cache_length.data_ptr(),
            mask_2d.data_ptr() if mask_2d is not None else 0,
            logits_buf.data_ptr(),
            self.cuda_graph
        )
        cache_length -= input_ids.numel()
        # torch.cuda.nvtx.range_pop()
        return logits_buf[:input_ids.numel()].clone()

    def generate(self, input_ids, generation_length=100, teminators=[], use_stream=False, progress_callback=None, temperature=None):
        """
        Generate text with optional streaming output.
        Returns (tokens, decode_time, prefill_time) if use_stream=False, or generator yielding {'token', 'text', 'is_finished', 'prefill_time', 'decode_time'} if use_stream=True.
        """
        assert input_ids.dtype == torch.int32

        prefix_length = input_ids.numel()
        position_ids = torch.arange(prefix_length, dtype=torch.int32, device="cuda")
        # Per-call logits buffer (avoid shared state across concurrent requests)
        local_logits = None
        # Effective temperature bound to this call
        effective_temperature = self.temperature if temperature is None else temperature
        
        # Measure prefill time
        torch.cuda.synchronize()
        prefill_start = time.time()
        logits = self.prefill(input_ids, position_ids, progress_callback, logits_buffer=local_logits)
        torch.cuda.synchronize()
        prefill_time = time.time() - prefill_start
        
        if effective_temperature > 0.0:
            # Apply temperature scaling
            scaled_logits = logits[0] / effective_temperature
            
            # Apply top_p filtering if needed
            if self.top_p < 1.0:
                filtered_logits = top_p_filtering(scaled_logits.unsqueeze(0), self.top_p)
                token = torch.multinomial(F.softmax(filtered_logits[0], dim=-1), num_samples=1, generator=self.generator)[0].item()
            else:
                token = torch.multinomial(F.softmax(scaled_logits, dim=-1), num_samples=1, generator=self.generator)[0].item()
        else:
            token = logits[0].argmax(dim=-1).item()

        # Use instance-level tiny state tensors (original behavior)
        if not hasattr(self, "input_ids"):
            self.input_ids = torch.tensor([0], dtype=torch.int32, device="cuda")
            self.position_ids = torch.tensor([0], dtype=torch.int32, device="cuda")
            self.cache_length = torch.tensor([0], dtype=torch.int32, device="cuda")

        if use_stream:
            # Stream generation (optimized)
            def _stream_generator():
                nonlocal token
                # Keep minimal context for correct spacing
                prev_token = token
                
                # yield first token
                text = self.tokenizer.decode([token], skip_special_tokens=True)
                
                yield {
                    'token': token,
                    'text': text,
                    'is_finished': token in teminators,
                    'prefill_time': prefill_time,
                    'decode_time': 0.0  # First token comes from prefill
                }
                
                if token in teminators:
                    return

                decode_start_time = time.time()

                for i in range(generation_length-1):
                    self.input_ids[0] = token
                    self.position_ids[0] = prefix_length + i
                    self.cache_length[0] = prefix_length + i

                    logits = self.decode(self.input_ids, self.position_ids, self.cache_length, logits_buffer=local_logits)
                    if effective_temperature > 0.0:
                        # Apply temperature scaling
                        scaled_logits = logits[0] / effective_temperature
                        
                        # Apply top_p filtering if needed
                        if self.top_p < 1.0:
                            filtered_logits = top_p_filtering(scaled_logits.unsqueeze(0), self.top_p)
                            token = torch.multinomial(F.softmax(filtered_logits[0], dim=-1), num_samples=1, generator=self.generator)[0].item()
                        else:
                            token = torch.multinomial(F.softmax(scaled_logits, dim=-1), num_samples=1, generator=self.generator)[0].item()
                    else:
                        token = logits[0].argmax(dim=-1).item()
                    
                    # For correct spacing, decode with previous token context
                    if prev_token is not None:
                        context_tokens = [prev_token, token]
                        context_text = self.tokenizer.decode(context_tokens, skip_special_tokens=True)
                        prev_text = self.tokenizer.decode([prev_token], skip_special_tokens=True)
                        text = context_text[len(prev_text):]
                    else:
                        text = self.tokenizer.decode([token], skip_special_tokens=True)
                    
                    is_finished = token in teminators or i == generation_length - 2
                    
                    # Calculate time only when needed to reduce overhead
                    decode_time = time.time() - decode_start_time
                        
                    yield {
                        'token': token,
                        'text': text,
                        'is_finished': is_finished,
                        'prefill_time': 0.0,  # Only report prefill_time for first token
                        'decode_time': decode_time
                    }
                    
                    if token in teminators:
                        break
                    
                    # Update prev_token
                    prev_token = token
            
            return _stream_generator()
        else:
            # Original batch generation
            tokens = [token]
            torch.cuda.synchronize()
            decode_start = time.time()
            for i in range(generation_length-1):
                self.input_ids[0] = token
                self.position_ids[0] = prefix_length + i
                self.cache_length[0] = prefix_length + i

                logits = self.decode(self.input_ids, self.position_ids, self.cache_length, logits_buffer=local_logits)
                if effective_temperature > 0.0:
                    # Apply temperature scaling
                    scaled_logits = logits[0] / effective_temperature
                    
                    # Apply top_p filtering if needed
                    if self.top_p < 1.0:
                        filtered_logits = top_p_filtering(scaled_logits.unsqueeze(0), self.top_p)
                        token = torch.multinomial(F.softmax(filtered_logits[0], dim=-1), num_samples=1, generator=self.generator)[0].item()
                    else:
                        token = torch.multinomial(F.softmax(scaled_logits, dim=-1), num_samples=1, generator=self.generator)[0].item()
                else:
                    token = logits[0].argmax(dim=-1).item()
                tokens.append(token)
                if token in teminators:
                    break
            torch.cuda.synchronize()
            decode_time = time.time() - decode_start
            return tokens, decode_time, prefill_time

    def print_perf_summary(self):
        C.print_perf_summary()