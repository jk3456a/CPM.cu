#pragma once
#include "../layer.cuh"
#include "attn_eagle3.cuh"
#include "tools_ealge3.cuh"  // 需要concat_embeddings_and_hidden函数
#include "../elementwise.cuh" // 需要elementwise_add函数

template <typename T>
struct Eagle3Layer {
    Eagle3Attention<T> *attn;
    FFN<T> *ffn;
    Norm<T> *hidden_norm;  // Add hidden_norm for Eagle3
    T* output;
    int hidden_size;
    float residual_scale;

    Eagle3Layer(int hidden_size, int intermediate_size, int num_attention_heads, int num_key_value_heads, int head_dim, float rms_norm_eps, float residual_scale = 1.0, int window_size = 0, bool use_qk_norm = false, bool use_attn_bias = false) {
        this->attn = new Eagle3Attention<T>(hidden_size, num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps, window_size, use_qk_norm, use_attn_bias);
        this->ffn = new GatedFFN<T>(hidden_size, intermediate_size, rms_norm_eps);
        this->hidden_norm = new RMSNorm<T>(hidden_size, rms_norm_eps);  // Initialize hidden_norm
        this->hidden_size = hidden_size;
        this->residual_scale = residual_scale;
    }

    void init_weight_ptr(Memory* memory) {
        this->attn->init_weight_ptr(memory);
        this->ffn->init_weight_ptr(memory);
        this->hidden_norm->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        // attn和ffn共享内存（标准做法）
        int64_t attn_end = this->attn->init_output_ptr(memory, num_tokens, offset);
        int64_t ffn_end = this->ffn->init_output_ptr(memory, num_tokens, offset);
        this->output = this->ffn->output;
        
        // hidden_norm需要独立内存
        int64_t shared_end = std::max(attn_end, ffn_end);
        int64_t hidden_norm_end = this->hidden_norm->init_output_ptr(memory, num_tokens, shared_end);
        return hidden_norm_end;
    }

    void load_to_storage(std::string name, void* ptr) {
        // 处理来自midlayer的权重名称，如 "self_attn.xxx", "mlp.xxx", "input_layernorm.xxx", "post_attention_layernorm.xxx", "hidden_norm.xxx"
        if (name.find("self_attn") != std::string::npos || name.find("input_layernorm") != std::string::npos) {
            this->attn->load_to_storage(name, ptr);
        } else if (name.find("mlp") != std::string::npos || name.find("post_attention_layernorm") != std::string::npos) {
            this->ffn->load_to_storage(name, ptr);
        } else if (name.find("hidden_norm") != std::string::npos) {
            this->hidden_norm->load_to_storage("weight", ptr);  // 传入简化的权重名
        } else {
            throw std::invalid_argument("Eagle3Layer: Unsupported weight name: " + name);
        }
    }

    // EAGLE3特殊的prefill：需要分别传入embedding和FC输出
    void prefill(int32_t num_tokens, int32_t num_history_tokens, T* input_emb, T* fc_output, 
                 int32_t* position_ids, KVCache<T>* kv_cache, T* concat_buffer) {
        // 1. 对FC输出应用hidden_norm归一化
        this->hidden_norm->prefill(calc_stream, num_tokens, fc_output, nullptr);
        
        // 2. 拼接embedding和归一化后的hidden_states为8192维
        concat_embeddings_and_hidden(calc_stream, num_tokens, this->hidden_size,
                                     input_emb,                   // embedding: 4096维
                                     this->hidden_norm->output,   // normalized hidden_states: 4096维  
                                     concat_buffer);              // 输出: 8192维
        
        // 3. Attention处理8192维输入，输出4096维
        cuda_perf_start_on_stream_f(PREFILL_ATTN, calc_stream.stream);
        this->attn->prefill(calc_stream, num_tokens, num_history_tokens, 
                            concat_buffer, nullptr, position_ids, kv_cache);
        cuda_perf_stop_on_stream_f(PREFILL_ATTN, calc_stream.stream);
        
        // 4. 对attention输出进行缩放
        elementwise_scale(calc_stream, num_tokens, this->hidden_size, 
                          this->attn->output, this->residual_scale);
        
        // 5. FFN处理：只接收4096维的normalized_hidden_states作为input，attention_output作为prev_output
        //    FFN内部的norm会自动进行残差连接(normalized_hidden_states + attn_output)
        cuda_perf_start_on_stream_f(PREFILL_FFN, calc_stream.stream);
        this->ffn->prefill(calc_stream, num_tokens, this->hidden_norm->output, this->attn->output);
        cuda_perf_stop_on_stream_f(PREFILL_FFN, calc_stream.stream);
    }

    // EAGLE3特殊的decode：需要分别传入embedding和FC输出
    void decode(int32_t num_tokens, int32_t padded_length, T* input_emb, T* fc_output,
                int32_t* position_ids, int32_t* cache_length, const Mask& mask, 
                KVCache<T>* kv_cache, T* concat_buffer) {
        // 1. 对FC输出应用hidden_norm归一化
        this->hidden_norm->prefill(calc_stream, num_tokens, fc_output, nullptr);
        
        // 2. 拼接embedding和归一化后的hidden_states为8192维
        concat_embeddings_and_hidden(calc_stream, num_tokens, this->hidden_size,
                                     input_emb,                   // embedding: 4096维
                                     this->hidden_norm->output,   // normalized hidden_states: 4096维  
                                     concat_buffer);              // 输出: 8192维
        
        // 3. Attention处理8192维输入，输出4096维
        cuda_perf_start_on_stream_f(DECODE_ATTN, calc_stream.stream);
        this->attn->decode(calc_stream, num_tokens, padded_length, 
                           concat_buffer, nullptr, position_ids, cache_length, mask, kv_cache);
        cuda_perf_stop_on_stream_f(DECODE_ATTN, calc_stream.stream);
        
        // 4. 对attention输出进行缩放
        elementwise_scale(calc_stream, num_tokens, this->hidden_size, 
                          this->attn->output, this->residual_scale);
        
        // 5. FFN处理：只接收4096维的normalized_hidden_states作为input，attention_output作为prev_output
        //    FFN内部的norm会自动进行残差连接(normalized_hidden_states + attn_output)
        cuda_perf_start_on_stream_f(DECODE_FFN, calc_stream.stream);
        this->ffn->decode(calc_stream, num_tokens, this->hidden_norm->output, this->attn->output);
        cuda_perf_stop_on_stream_f(DECODE_FFN, calc_stream.stream);
    }
};
