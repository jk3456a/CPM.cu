#pragma once
#include "../layer.cuh"
#include "attn_eagle3.cuh"
#include "tools_ealge3.cuh"  // 需要concat_embeddings_and_hidden函数

template <typename T>
struct Eagle3Layer {
    Eagle3Attention<T> *attn;
    FFN<T> *ffn;
    T* output;
    int hidden_size;
    float residual_scale;

    Eagle3Layer(int hidden_size, int intermediate_size, int num_attention_heads, int num_key_value_heads, int head_dim, float rms_norm_eps, float residual_scale = 1.0, int window_size = 0, bool use_qk_norm = false, bool use_attn_bias = false) {
        this->attn = new Eagle3Attention<T>(hidden_size, num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps, window_size, use_qk_norm, use_attn_bias);
        this->ffn = new GatedFFN<T>(hidden_size, intermediate_size, rms_norm_eps);
        this->hidden_size = hidden_size;
        this->residual_scale = residual_scale;
    }

    void init_weight_ptr(Memory* memory) {
        this->attn->init_weight_ptr(memory);
        this->ffn->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t attn_end = this->attn->init_output_ptr(memory, num_tokens, offset);
        int64_t ffn_end = this->ffn->init_output_ptr(memory, num_tokens, offset);
        this->output = this->ffn->output;
        return std::max(attn_end, ffn_end);
    }

    void load_to_storage(std::string name, void* ptr) {
        // 处理来自midlayer的权重名称，如 "self_attn.xxx", "mlp.xxx", "input_layernorm.xxx", "post_attention_layernorm.xxx"
        if (name.find("self_attn") != std::string::npos || name.find("input_layernorm") != std::string::npos) {
            this->attn->load_to_storage(name, ptr);
        } else if (name.find("mlp") != std::string::npos || name.find("post_attention_layernorm") != std::string::npos) {
            this->ffn->load_to_storage(name, ptr);
        } else {
            throw std::invalid_argument("Eagle3Layer: Unsupported weight name: " + name);
        }
    }

    // EAGLE3特殊的prefill：需要分别传入embedding和hidden_states
    void prefill(int32_t num_tokens, int32_t num_history_tokens, T* input_emb, T* hidden_states, 
                 int32_t* position_ids, KVCache<T>* kv_cache, T* concat_buffer) {
        // 1. 拼接embedding和hidden_states为8192维
        concat_embeddings_and_hidden(calc_stream, num_tokens, this->hidden_size,
                                     input_emb,       // embedding: 4096维
                                     hidden_states,   // hidden_states: 4096维  
                                     concat_buffer);  // 输出: 8192维
        
        // 2. Attention处理8192维输入，输出4096维
        cuda_perf_start_on_stream_f(PREFILL_ATTN, calc_stream.stream);
        this->attn->prefill(calc_stream, num_tokens, num_history_tokens, 
                            concat_buffer, nullptr, position_ids, kv_cache);
        cuda_perf_stop_on_stream_f(PREFILL_ATTN, calc_stream.stream);
        
        // 3. 对attention输出进行缩放
        elementwise_scale(calc_stream, num_tokens, this->hidden_size, 
                          this->attn->output, this->residual_scale);
        
        // 4. FFN处理：FFN内部的norm会自动进行残差连接(hidden_states + attn_output)
        cuda_perf_start_on_stream_f(PREFILL_FFN, calc_stream.stream);
        this->ffn->prefill(calc_stream, num_tokens, hidden_states, this->attn->output);
        cuda_perf_stop_on_stream_f(PREFILL_FFN, calc_stream.stream);
    }

    // EAGLE3特殊的decode：需要分别传入embedding和hidden_states
    void decode(int32_t num_tokens, int32_t padded_length, T* input_emb, T* hidden_states,
                int32_t* position_ids, int32_t* cache_length, const Mask& mask, 
                KVCache<T>* kv_cache, T* concat_buffer) {
        // 1. 拼接embedding和hidden_states为8192维
        concat_embeddings_and_hidden(calc_stream, num_tokens, this->hidden_size,
                                     input_emb,       // embedding: 4096维
                                     hidden_states,   // hidden_states: 4096维  
                                     concat_buffer);  // 输出: 8192维
        
        // 2. Attention处理8192维输入，输出4096维
        cuda_perf_start_on_stream_f(DECODE_ATTN, calc_stream.stream);
        this->attn->decode(calc_stream, num_tokens, padded_length, 
                           concat_buffer, nullptr, position_ids, cache_length, mask, kv_cache);
        cuda_perf_stop_on_stream_f(DECODE_ATTN, calc_stream.stream);
        
        // 3. 对attention输出进行缩放
        elementwise_scale(calc_stream, num_tokens, this->hidden_size, 
                          this->attn->output, this->residual_scale);
        
        // 4. FFN处理：FFN内部的norm会自动进行残差连接(hidden_states + attn_output)
        cuda_perf_start_on_stream_f(DECODE_FFN, calc_stream.stream);
        this->ffn->decode(calc_stream, num_tokens, hidden_states, this->attn->output);
        cuda_perf_stop_on_stream_f(DECODE_FFN, calc_stream.stream);
    }
};
