#pragma once
#include <type_traits>
#include "../tree_drafter.cuh"
#include "../eagle.cuh"
#include "../elementwise.cuh"
#include "minicpm4_model.cuh"
#include "minicpm4_layer.cuh"
#include "minicpm4_w4a16_gptq_marlin_model.cuh"
#include "../w4a16_gptq_marlin/w4a16_gptq_marlin_layer.cuh"
#include "../w4a16_gptq_marlin/w4a16_gptq_marlin_linear.cuh"
#include "../norm.cuh"

namespace {

    __global__ void eagle3_remap_kernel(int32_t num_tokens, const int32_t* input, int32_t* output, const int32_t* token_id_remap) {
        int32_t pos = input[threadIdx.x];
        output[threadIdx.x] = pos + token_id_remap[pos];
    }
    
    void eagle3_remap(const Stream& stream, int32_t num_tokens, const int32_t* input, int32_t* output, const int32_t* token_id_remap) {
        eagle3_remap_kernel<<<1, num_tokens, 0, stream.stream>>>(num_tokens, input, output, token_id_remap);
    }
    
}  // namespace

template<typename T, class ModelType, class LayerType, class Fc1Type, class Fc2Type>
struct MiniCPM4EagleImpl : Model {
    int num_layers;
    int num_iter;
    int topk_per_iter;
    int tree_size;
    int total_tried;
    float residual_scale;
    bool use_input_norm;
    bool use_attn_norm;

    ModelType* model = nullptr;
    KVCacheManager<T>* kv_caches = nullptr;
    std::vector<LayerType*> layers;
    Fc1Type *fc1 = nullptr;
    Fc2Type *fc2 = nullptr;
    Linear<T>* lm_head = nullptr;
    int32_t* token_id_remap = nullptr;
    RMSNorm<T> *input_norm1 = nullptr;
    RMSNorm<T> *input_norm2 = nullptr;
    functions::TopK<T>* topk_func = nullptr;
    functions::TopK<T>* topk_func_2 = nullptr;

    T *prev_hidden_state, *prev_embed;
    int num_prev, num_history_tokens;
    int32_t *eagle_position_ids, *eagle_cache_length;
    int *eagle_original_length, eagle_padded_length;
    uint64_t *eagle_mask_2d, *tmp_mask_2d;
    T* eagle_logits;
    T* tried_history_val; int32_t* tried_history_pos;
    int32_t* tried_history_parent = nullptr;
    bool is_first_draft;
    int frspec_vocab_size;

    int32_t *h_best, *d_best;    

    T* tmp_kvcache;

    T* a_tmp = nullptr;
    float* c_tmp = nullptr;

    MiniCPM4EagleImpl(
        ModelType* model,
        int num_layers,
        int intermediate_size,
        int num_attention_heads,
        int num_key_value_heads,
        int head_dim,
        float rms_norm_eps,
        int num_iter,
        int topk_per_iter,
        int tree_size,
        int group_size = 128,
        int eagle_window_size = 0,
        int frspec_vocab_size = 0,
        float residual_scale = 1.0f,
        bool use_input_norm = true,
        bool use_attn_norm = true
    ) {
        this->model = model;
        this->num_layers = num_layers;
        this->num_iter = num_iter;
        this->topk_per_iter = topk_per_iter;
        this->tree_size = tree_size;
        assert(this->tree_size <= 64);
        this->total_tried = topk_per_iter * topk_per_iter * (num_iter - 1) + topk_per_iter;
        this->frspec_vocab_size = frspec_vocab_size > 0 ? frspec_vocab_size : this->model->vocab_size;
        this->residual_scale = residual_scale;
        this->use_input_norm = use_input_norm;
        this->use_attn_norm = use_attn_norm;

        kv_caches = new KVCacheManager<T>(num_layers, num_key_value_heads, head_dim);
        if constexpr (std::is_same_v<Fc1Type, W4A16GPTQMarlinLinear<T>>) {
            fc1 = new W4A16GPTQMarlinLinear<T>(this->model->hidden_size, this->model->hidden_size, group_size, true, true);
            fc2 = new W4A16GPTQMarlinLinear<T>(this->model->hidden_size, this->model->hidden_size, group_size);
        } else {
            fc1 = new Linear<T>(this->model->hidden_size, this->model->hidden_size, true, true);
            fc2 = new Linear<T>(this->model->hidden_size, this->model->hidden_size);
        }
        if (use_input_norm) {
            input_norm1 = new RMSNorm<T>(this->model->hidden_size, rms_norm_eps);
            input_norm2 = new RMSNorm<T>(this->model->hidden_size, rms_norm_eps);
        }
        for (int i = 0; i < num_layers; i++) {
            if constexpr (std::is_same_v<LayerType, W4A16GPTQMarlinLayer<T>>) {
                layers.push_back(new W4A16GPTQMarlinLayer<T>(this->model->hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps, group_size, this->residual_scale, eagle_window_size));
            } else {
                layers.push_back(new Layer<T>(this->model->hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps, this->residual_scale, eagle_window_size));
            }
        }
        if (this->frspec_vocab_size != this->model->vocab_size) {
            lm_head = new Linear<T>(this->model->hidden_size, this->frspec_vocab_size);
        } else {
            lm_head = this->model->lm_head;
        }

        assert(this->topk_per_iter <= this->tree_size-1);

        topk_func = new functions::TopK<T>(this->frspec_vocab_size, this->topk_per_iter);
        topk_func_2 = new functions::TopK<T>(this->total_tried, this->tree_size-1);
    }

    void init_weight_ptr(Memory* memory) {
        fc1->init_weight_ptr(memory);
        fc2->init_weight_ptr(memory);
        if (use_input_norm) {
            input_norm1->init_weight_ptr(memory);
            input_norm2->init_weight_ptr(memory);
        }
        for (int i = 0; i < num_layers; i++) {
            layers[i]->init_weight_ptr(memory);
        }
        if (this->frspec_vocab_size != this->model->vocab_size) {
            lm_head->init_weight_ptr(memory);
        }
        if (!use_attn_norm) {
            layers[0]->attn->attn_norm = new Skip<T>(this->model->hidden_size);
        }
        kv_caches->rotary_embedding = this->model->kv_caches->rotary_embedding;
        token_id_remap = (int32_t*)memory->allocate_for_model(this->frspec_vocab_size * sizeof(int32_t));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        if constexpr (std::is_same_v<Fc1Type, W4A16GPTQMarlinLinear<T>>) {
            offset = memory->allocate((void**)&this->a_tmp, offset, 2 * num_tokens * this->model->hidden_size * sizeof(T));
            int reduce_max_m = marlin::determine_reduce_max_m(num_tokens, marlin::max_par);
            int reduce_n = 2 * this->model->hidden_size;
            offset = memory->allocate((void**)&this->c_tmp, offset, reduce_max_m * reduce_n * sizeof(float));
        }
        offset = fc1->init_output_ptr(memory, num_tokens, offset);
        offset = fc2->init_output_ptr(memory, num_tokens, offset);
        if (use_input_norm) {
            offset = input_norm1->init_output_ptr(memory, num_tokens, offset);
            offset = input_norm2->init_output_ptr(memory, num_tokens, offset);
        }
        int64_t layer_end = 0;
        for (int i = 0; i < num_layers; i++) {
            layer_end = layers[i]->init_output_ptr(memory, num_tokens, offset);
        }
        offset = layer_end;
        if (this->frspec_vocab_size != this->model->vocab_size) {
            offset = lm_head->init_output_ptr(memory, 64, offset);
        }
        offset = memory->allocate((void**)&eagle_logits, offset, this->topk_per_iter * this->frspec_vocab_size * sizeof(T));
        offset = memory->allocate((void**)&eagle_mask_2d, offset, this->topk_per_iter * sizeof(uint64_t));
        offset = memory->allocate((void**)&tmp_mask_2d, offset, this->topk_per_iter * sizeof(uint64_t));
        offset = memory->allocate((void**)&tried_history_val, offset, this->total_tried * sizeof(T));
        offset = memory->allocate((void**)&tried_history_pos, offset, this->total_tried * sizeof(int32_t));
        if (this->num_iter > 1) {
            offset = memory->allocate((void**)&tried_history_parent, offset, this->topk_per_iter * (this->num_iter - 1) * sizeof(int32_t));
        }
        cudaMallocHost(&eagle_original_length, sizeof(int32_t));

        offset = topk_func->init_output_ptr(memory, this->topk_per_iter, offset);
        offset = topk_func_2->init_output_ptr(memory, 1, offset);

        offset = memory->allocate((void**)&prev_hidden_state, offset, num_tokens * this->model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&prev_embed, offset, num_tokens * this->model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&eagle_position_ids, offset, num_tokens * sizeof(int32_t));
        offset = memory->allocate((void**)&eagle_cache_length, offset, sizeof(int32_t));

        offset = memory->allocate((void**)&d_best, offset, 2 * sizeof(int32_t));
        cudaMallocHost(&h_best, 2 * sizeof(int32_t));
        offset = memory->allocate((void**)&tmp_kvcache, offset, 64 * this->model->kv_caches->num_hidden_layers * 2 * this->model->kv_caches->dim * sizeof(T));
        return offset;
    }

    int init_storage() {
        this->model->init_weight_ptr(this->model->memory);
        this->init_weight_ptr(this->model->memory);
        int64_t offset = this->model->init_output_ptr(this->model->memory, this->model->chunk_length, this->model->memory->model_offset);
        int64_t kv_cache_offset = this->init_output_ptr(this->model->memory, this->model->chunk_length, offset);
        float ratio = float(this->model->num_hidden_layers) / (this->model->num_hidden_layers + this->num_layers);
        if constexpr (std::is_same_v<ModelType, MiniCPM4Impl<T>> || std::is_same_v<ModelType, MiniCPM4W4A16GPTQMarlinModelImpl<T>>) {
            kv_cache_offset = this->model->kv_caches->init_output_ptr(this->model->memory, this->model->chunk_length, kv_cache_offset, ratio);
        } else {
            kv_cache_offset = this->model->kv_caches->init_output_ptr(this->model->memory, kv_cache_offset, ratio);
        }
        kv_caches->init_output_ptr(this->model->memory, kv_cache_offset);
        return min(kv_caches->budget, this->model->kv_caches->budget);
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.substr(0, 5) == "eagle") {
            if (name.substr(0, 9) == "eagle.fc1") {
                fc1->load_to_storage(name, ptr);
            } else if (name.substr(0, 9) == "eagle.fc2") {
                fc2->load_to_storage(name, ptr);
            } else if (name.substr(0, 20) == "eagle.token_id_remap") {
                cudaMemcpy((void*)token_id_remap, ptr, this->frspec_vocab_size * sizeof(int32_t), cudaMemcpyHostToDevice);
            } else if (name.find("eagle.input_norm1") != std::string::npos) {
                if (!use_input_norm) throw std::invalid_argument("norm is not used, but input_norm1 is found");
                input_norm1->load_to_storage(name, ptr);
            } else if (name.find("eagle.input_norm2") != std::string::npos) {
                if (!use_input_norm) throw std::invalid_argument("norm is not used, but input_norm2 is found");
                input_norm2->load_to_storage(name, ptr);
            } else if (name.find("eagle.rotary_emb") != std::string::npos) {
                kv_caches->rotary_embedding->load_to_storage(name, ptr);
            } else {
                std::regex layer_regex("eagle\\.layers\\.(\\d+)\\.(.*)");
                std::smatch matches;
                if (std::regex_search(name, matches, layer_regex)) {
                    int layer_idx = std::stoi(matches[1]);
                    layers[layer_idx]->load_to_storage(matches[2], ptr);
                } else {
                    throw std::invalid_argument("Unsupported name (layer_idx not found): " + name);
                }
            }
        } else {
            this->model->load_to_storage(name, ptr);
            if (name.substr(0, 7) == "lm_head") {
                if (this->frspec_vocab_size != this->model->vocab_size) {
                    remap_copy(calc_stream, this->model->lm_head->weight, this->lm_head->weight, this->model->hidden_size, this->frspec_vocab_size, this->token_id_remap);
                }
            }
        }
    }

    void eagle_prefill(int num_history_tokens) {
        cudaMemcpy(this->prev_embed + (num_prev - 1) * this->model->hidden_size, this->model->embedding->output, this->model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
        if (use_input_norm) {
            this->input_norm1->prefill(calc_stream, num_prev, this->prev_embed, nullptr);
            this->input_norm2->prefill(calc_stream, num_prev, this->prev_hidden_state, nullptr);
            if constexpr (std::is_same_v<Fc1Type, W4A16GPTQMarlinLinear<T>>) {
                this->fc1->prefill(calc_stream, num_prev, this->input_norm1->output, this->a_tmp, this->c_tmp);
                this->fc2->prefill(calc_stream, num_prev, this->input_norm2->output, this->a_tmp, this->c_tmp);
            } else {
                this->fc1->prefill(calc_stream, num_prev, this->input_norm1->output);
                this->fc2->prefill(calc_stream, num_prev, this->input_norm2->output);
            }
        } else {
            if constexpr (std::is_same_v<Fc1Type, W4A16GPTQMarlinLinear<T>>) {
                this->fc1->prefill(calc_stream, num_prev, this->prev_embed, this->a_tmp, this->c_tmp);
                this->fc2->prefill(calc_stream, num_prev, this->prev_hidden_state, this->a_tmp, this->c_tmp);
            } else {
                this->fc1->prefill(calc_stream, num_prev, this->prev_embed);
                this->fc2->prefill(calc_stream, num_prev, this->prev_hidden_state);
            }
        }
        elementwise_add(calc_stream, num_prev, this->model->hidden_size, this->fc1->output, this->fc2->output, this->fc2->output);
        T* layer_output = nullptr;
        for (int i = 0; i < num_layers; i++) {
            this->layers[i]->prefill(num_prev, num_history_tokens, this->fc2->output, layer_output, this->eagle_position_ids, this->kv_caches->caches[i]);
            layer_output = this->layers[i]->output;
        }
        elementwise_scale(calc_stream, num_prev, this->model->hidden_size, layer_output, this->residual_scale);
        elementwise_add(calc_stream, num_prev, this->model->hidden_size, this->fc2->output, layer_output, this->fc2->output);
    }

    void eagle_decode(int32_t* cache_length) {
        if (use_input_norm) {
            this->input_norm1->prefill(calc_stream, num_prev, this->prev_embed, nullptr);
            this->input_norm2->prefill(calc_stream, num_prev, this->prev_hidden_state, nullptr);
            if constexpr (std::is_same_v<Fc1Type, W4A16GPTQMarlinLinear<T>>) {
                this->fc1->prefill(calc_stream, num_prev, this->input_norm1->output, this->a_tmp, this->c_tmp);
                this->fc2->prefill(calc_stream, num_prev, this->input_norm2->output, this->a_tmp, this->c_tmp);
            } else {
                this->fc1->prefill(calc_stream, num_prev, this->input_norm1->output);
                this->fc2->prefill(calc_stream, num_prev, this->input_norm2->output);
            }
        } else {
            if constexpr (std::is_same_v<Fc1Type, W4A16GPTQMarlinLinear<T>>) {
                this->fc1->prefill(calc_stream, num_prev, this->prev_embed, this->a_tmp, this->c_tmp);
                this->fc2->prefill(calc_stream, num_prev, this->prev_hidden_state, this->a_tmp, this->c_tmp);
            } else {
                this->fc1->prefill(calc_stream, num_prev, this->prev_embed);
                this->fc2->prefill(calc_stream, num_prev, this->prev_hidden_state);
            }
        }
        elementwise_add(calc_stream, num_prev, this->model->hidden_size, this->fc1->output, this->fc2->output, this->fc2->output);
        T* layer_output = nullptr;
        for (int i = 0; i < num_layers; i++) {
            this->layers[i]->decode(num_prev, this->eagle_padded_length, this->fc2->output, layer_output, this->eagle_position_ids, cache_length, Mask(nullptr), this->kv_caches->caches[i]);
            layer_output = this->layers[i]->output;
        }
        elementwise_scale(calc_stream, num_prev, this->model->hidden_size, layer_output, this->residual_scale);
        elementwise_add(calc_stream, num_prev, this->model->hidden_size, this->fc2->output, layer_output, this->fc2->output);
    }

    void prefill(int32_t num_tokens, int32_t num_history_tokens, int32_t* input, int32_t* position_ids, void* output) {
        this->model->embedding->prefill(calc_stream, num_tokens, input);
        if (num_history_tokens > 0) {
            this->eagle_prefill(num_history_tokens);
        }

        cudaMemcpy(this->prev_embed, this->model->embedding->output + this->model->hidden_size, (num_tokens - 1) * this->model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
        this->model->prefill_embed(num_tokens, num_history_tokens, this->model->embedding->output, position_ids, output);
        this->prev_hidden_state = this->model->norm->output;
        cudaMemcpy(this->eagle_position_ids, position_ids, num_tokens * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        this->num_prev = num_tokens;

        this->num_history_tokens = num_history_tokens;
        this->is_first_draft = true;
    }

    void decode(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        this->model->decode(num_tokens, padded_length, input, position_ids, cache_length, mask_2d, output);
    }

    void draft(int32_t* tree_draft_ids, int32_t* tree_position_ids, int32_t* cache_length, uint64_t* tree_attn_mask, int32_t* tree_parent) {
        cudaMemcpy(this->eagle_original_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);
        this->eagle_padded_length = (this->eagle_original_length[0] + 256 - 1) / 128 * 128;


        if (this->is_first_draft) {
            this->model->embedding->prefill(calc_stream, 1, tree_draft_ids);
            this->eagle_prefill(this->num_history_tokens);
        } else {
            this->eagle_decode(cache_length);
        }
        cudaMemcpy(this->eagle_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->eagle_position_ids, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        repeat(calc_stream, topk_per_iter, 1, 0, this->eagle_position_ids);

        { // d = 0
            lm_head->prefill(calc_stream, 1, this->fc2->output + (num_prev - 1) * this->model->hidden_size, this->eagle_logits);
            log_softmax(calc_stream, 1, this->frspec_vocab_size, this->eagle_logits);
            this->topk_func->prefill(calc_stream, 1, this->eagle_logits);
            cudaMemcpy(this->tried_history_val, this->topk_func->topk_val, topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->tried_history_pos, this->topk_func->topk_pos, topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            if (this->frspec_vocab_size != this->model->vocab_size) {
                remap(calc_stream, topk_per_iter, this->topk_func->topk_pos, this->topk_func_2->topk_pos, this->token_id_remap);
            } else {
                cudaMemcpy(this->topk_func_2->topk_pos, this->topk_func->topk_pos, topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            }
            cudaMemcpy(this->topk_func_2->topk_val, this->topk_func->topk_val, topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            repeat(calc_stream, topk_per_iter, this->model->hidden_size, num_prev-1, this->fc2->output, this->fc1->output);
            init_tree(calc_stream, topk_per_iter, this->eagle_mask_2d);
        }
        for (int d = 1; d < this->num_iter; ++d) {
            add(calc_stream, 1, this->eagle_cache_length, topk_per_iter);
            this->model->embedding->prefill(calc_stream, topk_per_iter, this->topk_func_2->topk_pos);
            if (use_input_norm) {
                this->input_norm1->prefill(calc_stream, topk_per_iter, this->model->embedding->output, nullptr);
                this->input_norm2->prefill(calc_stream, topk_per_iter, this->fc1->output, nullptr);
                if constexpr (std::is_same_v<Fc1Type, W4A16GPTQMarlinLinear<T>>) {
                    this->fc1->prefill(calc_stream, topk_per_iter, this->input_norm1->output, this->a_tmp, this->c_tmp);
                    this->fc2->prefill(calc_stream, topk_per_iter, this->input_norm2->output, this->a_tmp, this->c_tmp);
                } else {
                    this->fc1->prefill(calc_stream, topk_per_iter, this->input_norm1->output);
                    this->fc2->prefill(calc_stream, topk_per_iter, this->input_norm2->output);
                }
            } else {
                if constexpr (std::is_same_v<Fc1Type, W4A16GPTQMarlinLinear<T>>) {
                    this->fc2->prefill(calc_stream, topk_per_iter, this->fc1->output, this->a_tmp, this->c_tmp);
                    this->fc1->prefill(calc_stream, topk_per_iter, this->model->embedding->output, this->a_tmp, this->c_tmp);
                } else {
                    this->fc2->prefill(calc_stream, topk_per_iter, this->fc1->output);
                    this->fc1->prefill(calc_stream, topk_per_iter, this->model->embedding->output);
                }
            }
            elementwise_add(calc_stream, topk_per_iter, this->model->hidden_size, this->fc1->output, this->fc2->output, this->fc2->output);
            T* layer_output = nullptr;
            for (int i = 0; i < num_layers; i++) {
                this->layers[i]->decode(topk_per_iter, this->eagle_padded_length, this->fc2->output, layer_output, this->eagle_position_ids, this->eagle_cache_length, Mask(eagle_mask_2d, topk_per_iter, topk_per_iter * d), this->kv_caches->caches[i]);
                layer_output = this->layers[i]->output;
            }
            elementwise_scale(calc_stream, topk_per_iter, this->model->hidden_size, layer_output, this->residual_scale);
            elementwise_add(calc_stream, topk_per_iter, this->model->hidden_size, this->fc2->output, layer_output, this->fc2->output);
            add(calc_stream, topk_per_iter, this->eagle_position_ids, 1);

            lm_head->prefill(calc_stream, topk_per_iter, this->fc2->output, this->eagle_logits);
            log_softmax(calc_stream, topk_per_iter, this->frspec_vocab_size, this->eagle_logits);
            this->topk_func->prefill(calc_stream, topk_per_iter, this->eagle_logits);
            cumsum(calc_stream, topk_per_iter, topk_per_iter, this->topk_func->topk_val, this->topk_func_2->topk_val);
            cudaMemcpy(this->tried_history_val + topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter, this->topk_func->topk_val, topk_per_iter * topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->tried_history_pos + topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter, this->topk_func->topk_pos, topk_per_iter * topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            this->topk_func_2->prefill(calc_stream, 1, this->topk_func->topk_val, topk_per_iter * topk_per_iter, topk_per_iter);

            cudaMemcpy(this->tmp_mask_2d, this->eagle_mask_2d, topk_per_iter * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
            set_parent(calc_stream, topk_per_iter, this->tried_history_parent + (d - 1) * topk_per_iter, this->topk_func_2->topk_pos, topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter);
            update_tree(calc_stream, topk_per_iter, topk_per_iter * d, this->eagle_mask_2d, this->tmp_mask_2d, this->topk_func_2->topk_pos);
            remap_hidden(calc_stream, topk_per_iter, this->model->hidden_size, this->topk_func_2->topk_pos, this->fc2->output, this->fc1->output, topk_per_iter);
            if (this->frspec_vocab_size != this->model->vocab_size) {
                remap_id_fr(calc_stream, topk_per_iter, this->topk_func_2->topk_pos, this->topk_func->topk_pos, this->token_id_remap);
            } else {
                remap_id(calc_stream, topk_per_iter, this->topk_func_2->topk_pos, this->topk_func->topk_pos);
            }
        }

        this->topk_func_2->prefill(calc_stream, 1, this->tried_history_val);

        // build tree
        build_dynamic_tree(calc_stream, this->tree_size, this->eagle_original_length[0], this->topk_per_iter, this->tried_history_parent, this->topk_func_2->topk_pos, tree_position_ids, tree_attn_mask, tree_parent);
        if (this->frspec_vocab_size != this->model->vocab_size) {
            remap_id_fr(calc_stream, this->tree_size-1, this->topk_func_2->topk_pos, this->tried_history_pos, this->token_id_remap, tree_draft_ids + 1);
        } else {
            remap_id(calc_stream, this->tree_size-1, this->topk_func_2->topk_pos, this->tried_history_pos, tree_draft_ids + 1);
        }

        this->is_first_draft = false;
    }

    int verify(int32_t num_tokens, int32_t* pred, int32_t* gt, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, int32_t* tree_parent) {
        verify_draft(calc_stream, num_tokens, pred, gt, position_ids, cache_length, mask_2d, tree_parent, this->d_best);
        cudaMemcpyAsync(this->h_best, this->d_best, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, calc_stream.stream);
        cudaStreamSynchronize(calc_stream.stream);

        this->num_prev = h_best[0];
        remap_hidden(calc_stream, this->num_prev, this->model->hidden_size, pred, this->model->norm->output, this->prev_hidden_state);

        fix_kv_cache(calc_stream, h_best[0], this->model->kv_caches->num_hidden_layers * 2, this->model->kv_caches->dim, pred, gt, cache_length, this->model->kv_caches->d_flat_caches, this->tmp_kvcache);

        this->model->embedding->prefill(calc_stream, this->num_prev, pred);
        cudaMemcpy(this->prev_embed, this->model->embedding->output, this->num_prev * this->model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

        make_arange(calc_stream, this->num_prev, cache_length, this->eagle_position_ids);

        if constexpr (std::is_same_v<ModelType, MiniCPM4Impl<T>> || std::is_same_v<ModelType, MiniCPM4W4A16GPTQMarlinModelImpl<T>>) {
            this->model->kv_caches->add_length(h_best[0] - 1);
        }

        return h_best[0];
    }
};

template<typename T, class ModelType, class LayerType, class FcType>
struct MiniCPM4Eagle3Impl : Model {
    int num_layers;
    int num_iter;
    int topk_per_iter;
    int tree_size;
    int total_tried;
    float residual_scale;

    ModelType* model = nullptr;
    KVCacheManager<T>* kv_caches = nullptr;
    LayerType* mid_layer = nullptr;
    FcType *fc = nullptr;
    Linear<T>* lm_head = nullptr;
    int32_t* d2t = nullptr;
    RMSNorm<T> *norm = nullptr;
    functions::TopK<T>* topk_func = nullptr;
    functions::TopK<T>* topk_func_2 = nullptr;

    T *prev_hidden_state, *prev_embed;
    int num_prev, num_history_tokens;
    int32_t *eagle_position_ids, *eagle_cache_length;
    int *eagle_original_length, eagle_padded_length;
    uint64_t *eagle_mask_2d, *tmp_mask_2d;
    T* eagle_logits;
    T* tried_history_val; int32_t* tried_history_pos;
    int32_t* tried_history_parent = nullptr;
    bool is_first_draft;
    int draft_vocab_size;
    float rms_norm_eps;

    int32_t *h_best, *d_best;    

    T* tmp_kvcache;

    T* a_tmp = nullptr;
    float* c_tmp = nullptr;

    T* fc_output = nullptr;

    T* eagle3_hidden = nullptr;

    MiniCPM4Eagle3Impl(
        ModelType* model,
        int num_layers,
        int intermediate_size,
        int num_attention_heads,
        int num_key_value_heads,
        int head_dim,
        float rms_norm_eps,
        int num_iter,
        int topk_per_iter,
        int tree_size,
        int group_size = 128,
        int eagle_window_size = 0,
        int draft_vocab_size = 0,
        float residual_scale = 1.0f
    ) {
        this->model = model;
        this->num_layers = num_layers;
        this->num_iter = num_iter;
        this->topk_per_iter = topk_per_iter;
        this->tree_size = tree_size;
        assert(this->tree_size <= 64);
        this->total_tried = topk_per_iter * topk_per_iter * (num_iter - 1) + topk_per_iter;
        this->draft_vocab_size = draft_vocab_size > 0 ? draft_vocab_size : this->model->vocab_size;
        this->residual_scale = residual_scale;
        this->rms_norm_eps = rms_norm_eps;
        kv_caches = new KVCacheManager<T>(num_layers, num_key_value_heads, head_dim);
        if constexpr (std::is_same_v<FcType, W4A16GPTQMarlinLinear<T>>) {
            fc = new W4A16GPTQMarlinLinear<T>(3 * this->model->hidden_size, this->model->hidden_size, group_size);
        } else {
            fc = new Linear<T>(3 * this->model->hidden_size, this->model->hidden_size);
        }
        
        norm = new RMSNorm<T>(this->model->hidden_size, rms_norm_eps);
        
        assert(num_layers == 1);
        
        if constexpr (std::is_same_v<LayerType, W4A16GPTQMarlinLayer<T>>) {
            mid_layer = new W4A16GPTQMarlinLayer<T>(this->model->hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps, group_size, this->residual_scale, eagle_window_size, false, false, 2);
        } else {
            mid_layer = new Layer<T>(this->model->hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps, this->residual_scale, eagle_window_size, false, false, 2);
        }

        if (this->draft_vocab_size != this->model->vocab_size) {
            lm_head = new Linear<T>(this->model->hidden_size, this->draft_vocab_size);
        } else {
            lm_head = this->model->lm_head;
        }

        assert(this->topk_per_iter <= this->tree_size-1);

        topk_func = new functions::TopK<T>(this->draft_vocab_size, this->topk_per_iter);
        topk_func_2 = new functions::TopK<T>(this->total_tried, this->tree_size-1);
    }

    void init_weight_ptr(Memory* memory) {
        fc->init_weight_ptr(memory);
        norm->init_weight_ptr(memory);
        mid_layer->attn->attn_norm = new RMSNorm2<T>(this->model->hidden_size, this->rms_norm_eps);
        mid_layer->init_weight_ptr(memory);
        if (this->draft_vocab_size != this->model->vocab_size) {
            lm_head->init_weight_ptr(memory);
        }
        kv_caches->rotary_embedding->init_weight_ptr(memory);
        d2t = (int32_t*)memory->allocate_for_model(this->draft_vocab_size * sizeof(int32_t));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        if constexpr (std::is_same_v<FcType, W4A16GPTQMarlinLinear<T>>) {
            offset = memory->allocate((void**)&this->a_tmp, offset, 2 * num_tokens * this->model->hidden_size * sizeof(T));
            int reduce_max_m = marlin::determine_reduce_max_m(num_tokens, marlin::max_par);
            int reduce_n = 2 * this->model->hidden_size;
            offset = memory->allocate((void**)&this->c_tmp, offset, reduce_max_m * reduce_n * sizeof(float));
        }
        offset = fc->init_output_ptr(memory, num_tokens, offset);
        offset = memory->allocate((void**)&fc_output, offset, num_tokens * this->model->hidden_size * sizeof(T));
        offset = norm->init_output_ptr(memory, num_tokens, offset);
        int64_t layer_end = 0;
        layer_end = mid_layer->init_output_ptr(memory, num_tokens, offset);
        offset = layer_end;
        if (this->draft_vocab_size != this->model->vocab_size) {
            offset = lm_head->init_output_ptr(memory, 64, offset);
        }
        offset = memory->allocate((void**)&eagle_logits, offset, this->topk_per_iter * this->draft_vocab_size * sizeof(T));
        offset = memory->allocate((void**)&eagle_mask_2d, offset, this->topk_per_iter * sizeof(uint64_t));
        offset = memory->allocate((void**)&tmp_mask_2d, offset, this->topk_per_iter * sizeof(uint64_t));
        offset = memory->allocate((void**)&tried_history_val, offset, this->total_tried * sizeof(T));
        offset = memory->allocate((void**)&tried_history_pos, offset, this->total_tried * sizeof(int32_t));
        if (this->num_iter > 1) {
            offset = memory->allocate((void**)&tried_history_parent, offset, this->topk_per_iter * (this->num_iter - 1) * sizeof(int32_t));
        }
        cudaMallocHost(&eagle_original_length, sizeof(int32_t));

        offset = topk_func->init_output_ptr(memory, this->topk_per_iter, offset);
        offset = topk_func_2->init_output_ptr(memory, 1, offset);

        offset = memory->allocate((void**)&prev_hidden_state, offset, num_tokens * this->model->hidden_size * 3 * sizeof(T));
        offset = memory->allocate((void**)&prev_embed, offset, num_tokens * this->model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&eagle_position_ids, offset, num_tokens * sizeof(int32_t));
        offset = memory->allocate((void**)&eagle_cache_length, offset, sizeof(int32_t));

        offset = memory->allocate((void**)&d_best, offset, 2 * sizeof(int32_t));
        cudaMallocHost(&h_best, 2 * sizeof(int32_t));
        offset = memory->allocate((void**)&tmp_kvcache, offset, 64 * this->model->kv_caches->num_hidden_layers * 2 * this->model->kv_caches->dim * sizeof(T));
        offset = memory->allocate((void**)&eagle3_hidden, offset, num_tokens * 3 * this->model->hidden_size * sizeof(T));
        return offset;
    }

    int init_storage() {
        this->model->init_weight_ptr(this->model->memory);
        this->init_weight_ptr(this->model->memory);
        int64_t offset = this->model->init_output_ptr(this->model->memory, this->model->chunk_length, this->model->memory->model_offset);
        int64_t kv_cache_offset = this->init_output_ptr(this->model->memory, this->model->chunk_length, offset);
        float ratio = float(this->model->num_hidden_layers) / (this->model->num_hidden_layers + this->num_layers);
        if constexpr (std::is_same_v<ModelType, MiniCPM4Impl<T>> || std::is_same_v<ModelType, MiniCPM4W4A16GPTQMarlinModelImpl<T>>) {
            kv_cache_offset = this->model->kv_caches->init_output_ptr(this->model->memory, this->model->chunk_length, kv_cache_offset, ratio);
        } else {
            kv_cache_offset = this->model->kv_caches->init_output_ptr(this->model->memory, kv_cache_offset, ratio);
        }
        kv_caches->init_output_ptr(this->model->memory, kv_cache_offset);
        return min(kv_caches->budget, this->model->kv_caches->budget);
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.substr(0, 6) == "eagle3") {
            if (name.substr(0, 9) == "eagle3.fc") {
                fc->load_to_storage(name, ptr);
            } else if (name.substr(0, 10) == "eagle3.d2t") {
                cudaMemcpy((void*)d2t, ptr, this->draft_vocab_size * sizeof(int32_t), cudaMemcpyHostToDevice);
            } else if (name.find("eagle3.norm") != std::string::npos) {
                norm->load_to_storage(name, ptr);
            } else if (name.find("eagle3.lm_head") != std::string::npos) { // If eagle3.lm_head is trained, we need to load directly, otherwise we need to remap the weight.
                lm_head->load_to_storage(name, ptr);
            } else if (name.find("eagle3.rotary_emb") != std::string::npos) {
                kv_caches->rotary_embedding->load_to_storage(name, ptr);
            } else if (name.substr(0, 15) == "eagle3.midlayer") {
                std::regex layer_regex("eagle3.midlayer\\.(.*)");
                std::smatch matches;
                if (std::regex_search(name, matches, layer_regex)) {
                    mid_layer->load_to_storage(matches[1], ptr);
                } else {
                    throw std::invalid_argument("Unsupported name : " + name);
                }
            }
        } else {
            this->model->load_to_storage(name, ptr);
            // If eagle3.lm_head is not trained, we need to remap the weight.
            // if (name.substr(0, 7) == "lm_head") {
            //     if (this->draft_vocab_size != this->model->vocab_size) {
            //         remap_copy(calc_stream, this->model->lm_head->weight, this->lm_head->weight, this->model->hidden_size, this->draft_vocab_size, this->d2t);
            //     }
            // }
        }
    }

    void base_model_prefill_embed(int32_t num_tokens, int32_t num_history_tokens, T* embed, int32_t* position_ids, void* output) {
        T* layer_output = nullptr;
        for (int i = 0; i < this->model->num_hidden_layers; i++) {
            if ((i == 2) || (i == this->model->num_hidden_layers / 2) || (i == this->model->num_hidden_layers - 3)) {
                elementwise_add_and_concat3(calc_stream, num_tokens, this->model->hidden_size, this->model->num_hidden_layers, i, embed, layer_output, eagle3_hidden, this->model->residual_scale);
            }
            this->model->layers[i]->prefill(num_tokens, num_history_tokens, embed, layer_output, position_ids, this->model->kv_caches->caches[i]);
            layer_output = this->model->layers[i]->output;
        }
        elementwise_scale(calc_stream, num_tokens, this->model->hidden_size, layer_output, this->model->residual_scale);
        this->model->norm->prefill(calc_stream, num_tokens, embed, layer_output);
        this->model->lm_head->prefill(calc_stream, 1, this->model->norm->output + (num_tokens - 1) * this->model->hidden_size, (T*)output);
    }

    void base_model_decode_embed(int32_t num_tokens, int32_t padded_length, T* embed, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        Mask mask(mask_2d, num_tokens, num_tokens);
        T* layer_output = nullptr;
        for (int i = 0; i < this->model->num_hidden_layers; i++) {
            if ((i == 2) || (i == this->model->num_hidden_layers / 2) || (i == this->model->num_hidden_layers - 3)) {
                elementwise_add_and_concat3(calc_stream, num_tokens, this->model->hidden_size, this->model->num_hidden_layers, i, embed, layer_output, eagle3_hidden, this->model->residual_scale);
            }
            this->model->layers[i]->decode(num_tokens, padded_length, this->model->embedding->output, layer_output, position_ids, cache_length, mask, this->model->kv_caches->caches[i]);
            layer_output = this->model->layers[i]->output;
        }
        elementwise_scale(calc_stream, num_tokens, this->model->hidden_size, layer_output, this->model->residual_scale);
        this->model->norm->prefill(calc_stream, num_tokens, this->model->embedding->output, layer_output);
        this->model->lm_head->prefill(calc_stream, num_tokens, this->model->norm->output, (T*)output);
    }

    void eagle3_prefill(int num_history_tokens) {
        cudaMemcpy(this->prev_embed + (num_prev - 1) * this->model->hidden_size, this->model->embedding->output, this->model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
        if constexpr (std::is_same_v<FcType, W4A16GPTQMarlinLinear<T>>) {
            this->fc->prefill(calc_stream, num_prev, eagle3_hidden, this->a_tmp, this->c_tmp);
        } else {
            this->fc->prefill(calc_stream, num_prev, eagle3_hidden);
        }
        T* layer_output = nullptr;
        this->mid_layer->prefill(num_prev, num_history_tokens, this->fc->output, layer_output, this->eagle_position_ids, this->kv_caches->caches[0], nullptr, this->prev_embed);
        layer_output = this->mid_layer->output;
        elementwise_scale(calc_stream, num_prev, this->model->hidden_size, layer_output, this->residual_scale);
        elementwise_add(calc_stream, num_prev, this->model->hidden_size, this->fc->output, layer_output, this->fc->output);
    }

    void eagle3_decode(int32_t* cache_length) {
        if constexpr (std::is_same_v<FcType, W4A16GPTQMarlinLinear<T>>) {
            this->fc->prefill(calc_stream, num_prev, this->prev_hidden_state, this->a_tmp, this->c_tmp);
        } else {
            this->fc->prefill(calc_stream, num_prev, this->prev_hidden_state);
        }
        T* layer_output = nullptr;
        this->mid_layer->decode(num_prev, this->eagle_padded_length, this->fc->output, layer_output, this->eagle_position_ids, cache_length, Mask(nullptr), this->kv_caches->caches[0], nullptr, this->prev_embed);
        layer_output = this->mid_layer->output;
        elementwise_scale(calc_stream, num_prev, this->model->hidden_size, layer_output, this->residual_scale);
        elementwise_add(calc_stream, num_prev, this->model->hidden_size, this->fc->output, layer_output, this->fc->output);
        
    }

    void prefill(int32_t num_tokens, int32_t num_history_tokens, int32_t* input, int32_t* position_ids, void* output) {
        this->model->embedding->prefill(calc_stream, num_tokens, input);
        if (num_history_tokens > 0) {
            this->eagle3_prefill(num_history_tokens);
        }
        cudaMemcpy(this->prev_embed, this->model->embedding->output + this->model->hidden_size, (num_tokens - 1) * this->model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
        this->base_model_prefill_embed(num_tokens, num_history_tokens, this->model->embedding->output, position_ids, output);
        this->prev_hidden_state = eagle3_hidden;
        cudaMemcpy(this->eagle_position_ids, position_ids, num_tokens * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        this->num_prev = num_tokens;

        this->num_history_tokens = num_history_tokens;
        this->is_first_draft = true;
    }

    void decode(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        this->model->embedding->prefill(calc_stream, num_tokens, input);
        this->base_model_decode_embed(num_tokens, padded_length, this->model->embedding->output, position_ids, cache_length, mask_2d, output);
    }

    void draft(int32_t* tree_draft_ids, int32_t* tree_position_ids, int32_t* cache_length, uint64_t* tree_attn_mask, int32_t* tree_parent) {
        cudaMemcpy(this->eagle_original_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);
        this->eagle_padded_length = (this->eagle_original_length[0] + 256 - 1) / 128 * 128;
        if (this->is_first_draft) {
            this->model->embedding->prefill(calc_stream, 1, tree_draft_ids);
            this->eagle3_prefill(this->num_history_tokens);
        } else {
            this->eagle3_decode(cache_length);
        }
        cudaMemcpy(this->eagle_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->eagle_position_ids, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        repeat(calc_stream, topk_per_iter, 1, 0, this->eagle_position_ids);

        { // d = 0
            this->norm->prefill(calc_stream, 1, this->fc->output + (num_prev - 1) * this->model->hidden_size, nullptr);
            lm_head->prefill(calc_stream, 1, this->norm->output, this->eagle_logits);
            log_softmax(calc_stream, 1, this->draft_vocab_size, this->eagle_logits);
            this->topk_func->prefill(calc_stream, 1, this->eagle_logits);

            cudaMemcpy(this->tried_history_val, this->topk_func->topk_val, topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->tried_history_pos, this->topk_func->topk_pos, topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            
            eagle3_remap(calc_stream, topk_per_iter, this->topk_func->topk_pos, this->topk_func_2->topk_pos, this->d2t);
            
            cudaMemcpy(this->topk_func_2->topk_val, this->topk_func->topk_val, topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            repeat(calc_stream, topk_per_iter, this->model->hidden_size, num_prev-1, this->fc->output); // repeated fc output could be stored in norm output temporarily
            
            init_tree(calc_stream, topk_per_iter, this->eagle_mask_2d);
        }
        
        for (int d = 1; d < this->num_iter; ++d) {
            add(calc_stream, 1, this->eagle_cache_length, topk_per_iter);
            this->model->embedding->prefill(calc_stream, topk_per_iter, this->topk_func_2->topk_pos);
            T* layer_output = nullptr;
            this->mid_layer->decode(topk_per_iter, this->eagle_padded_length, this->fc->output, layer_output, this->eagle_position_ids, this->eagle_cache_length, Mask(eagle_mask_2d, topk_per_iter, topk_per_iter * d), this->kv_caches->caches[0], nullptr, this->model->embedding->output);
            layer_output = this->mid_layer->output;
            elementwise_scale(calc_stream, topk_per_iter, this->model->hidden_size, layer_output, this->residual_scale);
            elementwise_add(calc_stream, topk_per_iter, this->model->hidden_size, this->fc->output, layer_output, this->fc->output);
            add(calc_stream, topk_per_iter, this->eagle_position_ids, 1);
            
            this->norm->prefill(calc_stream, topk_per_iter, this->fc->output, nullptr);
            lm_head->prefill(calc_stream, topk_per_iter, this->norm->output, this->eagle_logits);
            log_softmax(calc_stream, topk_per_iter, this->draft_vocab_size, this->eagle_logits);
            this->topk_func->prefill(calc_stream, topk_per_iter, this->eagle_logits);
            
            cumsum(calc_stream, topk_per_iter, topk_per_iter, this->topk_func->topk_val, this->topk_func_2->topk_val);
            cudaMemcpy(this->tried_history_val + topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter, this->topk_func->topk_val, topk_per_iter * topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->tried_history_pos + topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter, this->topk_func->topk_pos, topk_per_iter * topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            this->topk_func_2->prefill(calc_stream, 1, this->topk_func->topk_val, topk_per_iter * topk_per_iter, topk_per_iter);

            cudaMemcpy(this->tmp_mask_2d, this->eagle_mask_2d, topk_per_iter * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
            set_parent(calc_stream, topk_per_iter, this->tried_history_parent + (d - 1) * topk_per_iter, this->topk_func_2->topk_pos, topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter);
            update_tree(calc_stream, topk_per_iter, topk_per_iter * d, this->eagle_mask_2d, this->tmp_mask_2d, this->topk_func_2->topk_pos);
            remap_hidden(calc_stream, topk_per_iter, this->model->hidden_size, this->topk_func_2->topk_pos, this->fc->output, this->fc_output, topk_per_iter);
            cudaMemcpy(this->fc->output, this->fc_output, topk_per_iter * this->model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

            remap_id(calc_stream, topk_per_iter, this->topk_func_2->topk_pos, this->topk_func->topk_pos);
            eagle3_remap(calc_stream, topk_per_iter, this->topk_func_2->topk_pos, this->topk_func_2->topk_pos, this->d2t);
        }

        this->topk_func_2->prefill(calc_stream, 1, this->tried_history_val);

        // build tree
        build_dynamic_tree(calc_stream, this->tree_size, this->eagle_original_length[0], this->topk_per_iter, this->tried_history_parent, this->topk_func_2->topk_pos, tree_position_ids, tree_attn_mask, tree_parent);
        remap_id(calc_stream, this->tree_size-1, this->topk_func_2->topk_pos, this->tried_history_pos);
        eagle3_remap(calc_stream, this->tree_size-1, this->topk_func_2->topk_pos, tree_draft_ids + 1, this->d2t);

        this->is_first_draft = false;
    }

    int verify(int32_t num_tokens, int32_t* pred, int32_t* gt, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, int32_t* tree_parent) {
        verify_draft(calc_stream, num_tokens, pred, gt, position_ids, cache_length, mask_2d, tree_parent, this->d_best);
        cudaMemcpyAsync(this->h_best, this->d_best, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, calc_stream.stream);
        cudaStreamSynchronize(calc_stream.stream);

        this->num_prev = h_best[0];
        remap_hidden(calc_stream, this->num_prev, this->model->hidden_size * 3, pred, eagle3_hidden, this->prev_hidden_state);

        fix_kv_cache(calc_stream, h_best[0], this->model->kv_caches->num_hidden_layers * 2, this->model->kv_caches->dim, pred, gt, cache_length, this->model->kv_caches->d_flat_caches, this->tmp_kvcache);

        this->model->embedding->prefill(calc_stream, this->num_prev, pred);
        cudaMemcpy(this->prev_embed, this->model->embedding->output, this->num_prev * this->model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

        make_arange(calc_stream, this->num_prev, cache_length, this->eagle_position_ids);

        if constexpr (std::is_same_v<ModelType, MiniCPM4Impl<T>> || std::is_same_v<ModelType, MiniCPM4W4A16GPTQMarlinModelImpl<T>>) {
            this->model->kv_caches->add_length(h_best[0] - 1);
        }

        return h_best[0];
    }
};