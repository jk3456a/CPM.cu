#pragma once
#include "../tree_drafter.cuh"
#include "../eagle.cuh"
#include "layer_eagle3.cuh"
#include "tools_ealge3.cuh"

/*
================================================================================
                            EAGLE3 算法流程说明
================================================================================

EAGLE3 (Extrapolation Algorithm for Greater Language-model Efficiency 3) 是一种
高效的推测解码方法，通过单层草稿模型和多层特征融合来加速大语言模型的推理。

详细的实现历史和设计决策请参考：docs/eagle3_implementation_history.md

主要特点：
1. 单层架构：相比EAGLE2的多层设计，EAGLE3只使用单层decoder，大幅减少参数量
2. 多层特征融合：从基础模型的多个层（通常是最后3层）提取特征，通过FC层融合
3. 独立词汇表：支持draft model使用更小的词汇表以提高效率
4. 树形推测：使用树形结构进行批量推测，一次验证多个候选序列

算法流程：
┌─────────────────────────────────────────────────────────────────┐
│ 1. 初始化阶段 (prefill)                                          │
│    - 基础模型进行常规prefill，获取初始hidden states              │
│    - 保存最后一个token的hidden state和embedding                 │
│    - 初始化EAGLE3的position ids和cache                          │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. 草稿生成阶段 (draft)                                          │
│    a) 第一次草稿 (is_first_draft = true)                        │
│       - 使用基础模型的embedding                                 │
│       - 调用eagle_prefill处理（使用初始multi hidden states）    │
│                                                                 │
│    b) 后续草稿                                                  │
│       - 使用eagle_decode处理（使用新的multi hidden states）     │
│                                                                 │
│    c) 迭代生成树形候选 (num_iter次)                             │
│       - d=0: 使用FC处理后的特征，生成第一层候选和hidden states │
│       - d>0: 使用d=0的hidden states递归生成后续层               │
│       - 每次迭代更新tree mask和parent关系                       │
│                                                                 │
│    d) 构建最终的draft tree                                      │
│       - 选择最优的tree_size-1个候选                            │
│       - 如需要，进行词汇表映射(draft → target)                  │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. 验证阶段 (verify)                                             │
│    - 基础模型并行验证所有草稿token                               │
│    - 找出最长的匹配序列                                         │
│    - 提取Layer 2,16,29的输出并拼接成12288维特征                 │
│    - 调用update_multi_hidden_states保存新特征                   │
│    - 更新hidden states和KV cache                                │
│    - 准备下一轮草稿生成的输入                                   │
└─────────────────────────────────────────────────────────────────┘
                                ↓
                        循环回到步骤2

关键数据结构：
- tree_mask: 64位掩码，表示树中的依赖关系
- tree_parent: 每个节点的父节点索引
- tree_position_ids: 每个节点在序列中的位置
- multi_layer_features: 多层特征的拼接缓冲区

性能优化：
- 使用CUDA kernel并行处理
- 内存对齐优化(eagle_padded_length)
- 批量Top-K选择
- 高效的树形结构构建
================================================================================
*/



template<typename T, class ModelType>
struct Eagle3Impl : Model {
    // ========== 算法参数 ==========
    int num_iter;        // 草稿生成的迭代次数，决定树的深度
    int topk_per_iter;   // 每次迭代选择的top-k个候选token
    int tree_size;       // 最终树的大小（包括根节点）
    int total_tried;     // 总共尝试的候选数量

    ModelType* model;  // 基础模型引用
    KVCacheManager<T>* kv_caches;  // KV cache管理器（只有1层）
    Eagle3Layer<T>* eagle_layer;  // Eagle3的单层decoder（支持8192维输入）
    
    // Eagle3 specific components
    Linear<T> *fc;  // 3x hidden_size输入，用于多层特征融合
    Embedding<T>* eagle_embeddings;  // 复用base model的embeddings（直接指向model->embedding）
    Linear<T>* eagle_lm_head;  // Eagle3自己的lm_head
    RMSNorm<T>* output_norm;  // 输出归一化
    
    // Vocabulary mapping for draft model
    int64_t* d2t_mapping = nullptr;  // draft to target vocab mapping [draft_vocab_size] - torch.int64
    // t2d_mapping已删除 - 仅用于训练，推理时不需要
    // bool* t2d_mapping = nullptr;     // target to draft vocab mapping [target_vocab_size] - torch.bool
    int draft_vocab_size;
    int vocab_size;
    bool need_vocab_mapping;
    
    // Multi-layer feature buffer
    T* multi_layer_features;         // 拼接后的特征缓冲区 (12288维)
    T* base_model_hidden_states = nullptr;     // 从Base Model接收的multi hidden states - 初始化为nullptr
    T* attention_input_buffer;       // attention输入缓冲区 (8192维 = embedding + hidden)
    int num_feature_layers;
    
    // ========== TopK选择器 ==========
    functions::TopK<T>* topk_func;     // 用于从词汇表中选择top-k个候选
    functions::TopK<T>* topk_func_2;   // 用于从所有尝试的候选中选择最终的tree_size-1个

    // ========== 状态缓存 ==========
    T *prev_hidden_state, *prev_embed;  // 前一步的隐藏状态和嵌入
    int num_prev;                       // 前一步接受的token数量
    int num_history_tokens;             // 历史token数量（用于KV cache）
    
    // ========== EAGLE位置和缓存管理 ==========
    int32_t *eagle_position_ids;       // EAGLE模型使用的位置ID
    int32_t *eagle_cache_length;       // EAGLE的KV cache长度
    int *eagle_original_length;        // 原始序列长度（Host内存）
    int eagle_padded_length;           // 对齐后的长度（提高内存访问效率）
    
    // ========== 树形结构管理 ==========
    uint64_t *eagle_mask_2d;           // 当前的2D attention mask（64位掩码）
    uint64_t *tmp_mask_2d;             // 临时mask缓冲区
    T* eagle_logits;                   // EAGLE模型的输出logits
    T* tried_history_val;              // 所有尝试过的候选的概率值
    int32_t* tried_history_pos;        // 所有尝试过的候选的token ID
    int32_t* tried_history_parent;     // 候选的父节点索引（用于构建树）
    bool is_first_draft;               // 是否是第一次生成草稿
    
    // ========== 验证结果 ==========
    int32_t *h_best, *d_best;          // 最佳匹配长度（Host和Device内存）
    
    // ========== 临时缓冲区 ==========
    T* tmp_kvcache;                    // 用于KV cache修复的临时缓冲区

    /**
     * Eagle3Impl 构造函数
     * 
     * @param model             基础大语言模型的引用
     * @param intermediate_size FFN中间层大小
     * @param num_attention_heads 注意力头数量
     * @param num_key_value_heads KV头数量（用于GQA）
     * @param head_dim          每个注意力头的维度
     * @param rms_norm_eps      RMSNorm的epsilon值
     * @param num_iter          草稿生成的迭代次数
     * @param topk_per_iter     每次迭代选择的候选数
     * @param tree_size         最终树的大小
     * @param draft_vocab_size  草稿模型的词汇表大小（0表示与基础模型相同）
     */
    Eagle3Impl(
        ModelType* model,
        int intermediate_size,
        int num_attention_heads,
        int num_key_value_heads,
        int head_dim,
        float rms_norm_eps,
        int num_iter,
        int topk_per_iter,
        int tree_size,
        int draft_vocab_size = 0  // 0 means same as model vocab_size
    ) {
        this->model = model;
        this->num_feature_layers = 3;  // Eagle3 固定使用3层特征
        this->num_iter = num_iter;
        this->topk_per_iter = topk_per_iter;
        this->tree_size = tree_size;
        assert(this->tree_size <= 64); // tree_size must be <= 64
        this->total_tried = topk_per_iter * topk_per_iter * (num_iter - 1) + topk_per_iter;
        
        this->vocab_size = model->vocab_size;  // target vocab size (如：73448)
        this->draft_vocab_size = (draft_vocab_size > 0) ? draft_vocab_size : model->vocab_size;  // draft vocab size (如：32000)
        this->need_vocab_mapping = (this->draft_vocab_size != this->vocab_size);

        kv_caches = new KVCacheManager<T>(1, num_key_value_heads, head_dim);  // Only 1 layer
        
        // Eagle3 specific components
        fc = new Linear<T>(this->model->hidden_size * num_feature_layers, this->model->hidden_size);  // 12288 -> 4096 (3x hidden_size输入)
        eagle_embeddings = this->model->embedding;  // 直接复用base model的embedding，不创建新的
        eagle_lm_head = new Linear<T>(this->model->hidden_size, this->draft_vocab_size);
        output_norm = new RMSNorm<T>(this->model->hidden_size, rms_norm_eps);
        
        // Eagle3's single decoder layer
        // 使用Eagle3Layer，它内部使用Eagle3Attention处理8192维输入
        eagle_layer = new Eagle3Layer<T>(this->model->hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps);

        assert(topk_per_iter <= this->tree_size-1);

        // 修正：topk_func必须使用draft_vocab_size，因为eagle_lm_head输出的是draft_vocab_size维
        topk_func = new functions::TopK<T>(this->draft_vocab_size, topk_per_iter);
        topk_func_2 = new functions::TopK<T>(total_tried, this->tree_size-1);
    }
    
    ~Eagle3Impl() {
        // 释放分配的资源，防止内存泄漏
        delete kv_caches;
        delete fc;
        // eagle_embeddings不需要删除，它指向model->embedding
        delete eagle_lm_head;
        delete output_norm;
        delete eagle_layer;
        delete topk_func;
        delete topk_func_2;
        
        // 释放host内存
        if (eagle_original_length) cudaFreeHost(eagle_original_length);
        if (h_best) cudaFreeHost(h_best);
        
        // 注意：其他GPU内存（如multi_layer_features等）由Memory管理器统一释放
    }

    /**
     * 初始化权重指针
     * 
     * 步骤：
     * 1. 初始化EAGLE3特有的权重（FC、embeddings、lm_head、norm）
     * 2. 初始化单层decoder的权重
     * 3. 设置特殊的Skip层替代标准attention norm
     * 4. 共享基础模型的rotary embedding
     * 5. 如需要，分配词汇表映射内存
     */
    void init_weight_ptr(Memory* memory) {
        // Eagle3 specific weights (不包括embeddings，它复用base model的)
        fc->init_weight_ptr(memory);
        eagle_lm_head->init_weight_ptr(memory);
        output_norm->init_weight_ptr(memory);
        
        // Single layer initialization
        eagle_layer->init_weight_ptr(memory);
        eagle_layer->attn->attn_norm = new Skip<T>(this->model->hidden_size);
        kv_caches->rotary_embedding = this->model->kv_caches->rotary_embedding;
        
        // Initialize vocabulary mapping if needed
        if (need_vocab_mapping) {
            memory->allocate((void**)&d2t_mapping, 0, draft_vocab_size * sizeof(int64_t));
            // t2d_mapping不需要分配内存 - 仅用于训练
            // memory->allocate((void**)&t2d_mapping, 0, vocab_size * sizeof(bool));
        }
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        // Eagle3 specific outputs (embeddings复用base model的输出缓冲区)
        offset = fc->init_output_ptr(memory, num_tokens, offset);
        offset = eagle_lm_head->init_output_ptr(memory, num_tokens, offset);
        offset = output_norm->init_output_ptr(memory, num_tokens, offset);
        
        // Single layer output
        int64_t layer_end = eagle_layer->init_output_ptr(memory, num_tokens, offset);
        offset = layer_end;
        
        // Multi-layer feature buffer
        offset = memory->allocate((void**)&multi_layer_features, offset, 
                                  num_tokens * this->model->hidden_size * num_feature_layers * sizeof(T));
        offset = memory->allocate((void**)&base_model_hidden_states, offset,
                                  num_tokens * this->model->hidden_size * num_feature_layers * sizeof(T));
        // Attention input buffer for embedding + hidden concatenation
        offset = memory->allocate((void**)&attention_input_buffer, offset,
                                  num_tokens * this->model->hidden_size * 2 * sizeof(T));
        
        offset = memory->allocate((void**)&eagle_logits, offset, this->topk_per_iter * this->draft_vocab_size * sizeof(T));
        offset = memory->allocate((void**)&eagle_mask_2d, offset, this->topk_per_iter * sizeof(uint64_t));
        offset = memory->allocate((void**)&tmp_mask_2d, offset, this->topk_per_iter * sizeof(uint64_t));
        offset = memory->allocate((void**)&tried_history_val, offset, this->total_tried * sizeof(T));
        offset = memory->allocate((void**)&tried_history_pos, offset, this->total_tried * sizeof(int32_t));
        offset = memory->allocate((void**)&tried_history_parent, offset, this->topk_per_iter * (this->num_iter - 1) * sizeof(int32_t));
        cudaMallocHost(&eagle_original_length, sizeof(int32_t));

        offset = topk_func->init_output_ptr(memory, this->topk_per_iter, offset);
        offset = topk_func_2->init_output_ptr(memory, 1, offset);

        offset = memory->allocate((void**)&prev_hidden_state, offset, num_tokens * this->model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&prev_embed, offset, num_tokens * this->model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&eagle_position_ids, offset, num_tokens * sizeof(int32_t));
        offset = memory->allocate((void**)&eagle_cache_length, offset, sizeof(int32_t));

        offset = memory->allocate((void**)&d_best, offset, 2 * sizeof(int32_t));
        cudaMallocHost(&h_best, 2 * sizeof(int32_t));
        
        // EAGLE3专用的临时KV cache缓冲区 - 只需要1层，不是base model的多层
        // 大小：max_tree_size * 单层 * K+V * hidden_dim
        offset = memory->allocate((void**)&tmp_kvcache, offset, 
                                  64 * 1 * 2 * this->kv_caches->dim * sizeof(T));
        return offset;
    }

    int init_storage() {
        this->model->init_weight_ptr(this->model->memory);
        this->init_weight_ptr(this->model->memory);
        int64_t offset = this->model->init_output_ptr(this->model->memory, this->model->chunk_length, this->model->memory->model_offset);
        int64_t kv_cache_offset = this->init_output_ptr(this->model->memory, this->model->chunk_length, offset);
        float ratio = float(this->model->num_hidden_layers) / (this->model->num_hidden_layers + 1);  // Eagle3 has 1 layer
        kv_cache_offset = this->model->kv_caches->init_output_ptr(this->model->memory, kv_cache_offset, ratio);
        kv_caches->init_output_ptr(this->model->memory, kv_cache_offset);
        return min(kv_caches->budget + 1, this->model->kv_caches->budget);
    }

    void load_to_storage(std::string name, void* ptr) {
        // 检查是否是EAGLE3的权重（移除"eagle."前缀的检查，直接匹配权重名称）
        if (name.find("midlayer") != std::string::npos) {
            // Handle layer weights with midlayer prefix: "midlayer.self_attn.q_proj.weight" -> "self_attn.q_proj.weight"
            std::string layer_name = name.substr(name.find("midlayer.") + 9);
            eagle_layer->load_to_storage(layer_name, ptr);
        } else if (name == "fc.weight") {
            fc->load_to_storage(name, ptr);
        } else if (name == "lm_head.weight") {
            eagle_lm_head->load_to_storage(name, ptr);
        } else if (name == "norm.weight") {
            output_norm->load_to_storage(name, ptr);
        } else if (name == "d2t") {
            // Load draft to target vocab mapping (torch.int64)
            if (need_vocab_mapping && d2t_mapping != nullptr) {
                cudaMemcpy(d2t_mapping, ptr, draft_vocab_size * sizeof(int64_t), cudaMemcpyHostToDevice);
            }
        // } else if (name == "t2d") {
        //     // Load target to draft vocab mapping (torch.bool)
        //     if (need_vocab_mapping && t2d_mapping != nullptr) {
        //         cudaMemcpy(t2d_mapping, ptr, vocab_size * sizeof(bool), cudaMemcpyHostToDevice);
        //     }
        // t2d仅用于训练，推理时跳过
        // Skip loading t2d mapping - only used for training
        } else {
            // 如果不是EAGLE3权重，传递给基础模型处理
            this->model->load_to_storage(name, ptr);
        }
    }

    /**
     * EAGLE3 prefill阶段
     * 
     * 功能：处理第一次草稿生成或新的序列开始
     * 
     * 步骤：
     * 1. 使用EAGLE3自己的embedding（而非基础模型的）
     * 2. 收集多层特征（通过get_eagle3_layer_outputs获取Layer 2,16,29的真实输出）
     * 3. 通过FC层融合多层特征 (12288→4096)
     * 4. 通过单层decoder处理
     *    - Layer内部会将embedding和hidden_state拼接成8192维
     *    - Attention处理8192维输入
     * 5. 应用输出归一化
     * 
     * 注意：虽然EAGLE3只有单层，但仍需区分prefill/decode，因为：
     * - LlamaDecoderLayeremb需要特殊处理input_emb和hidden_states的拼接
     * - 在树形生成中，KV cache仍能带来性能提升
     * - 保持与基础模型的接口一致性
     * 
     * @param num_history_tokens 历史token数量（用于attention mask）
     */
    void eagle_prefill(int num_history_tokens) {
        // Eagle3: Use eagle's own embedding instead of base model's
        this->eagle_embeddings->prefill(calc_stream, 1, 
                                        reinterpret_cast<int32_t*>(this->model->norm->output));
        
        // Get multi-layer features from base model
        // EAGLE3使用分布式层采样策略，而非简单的最后3层
        int layer_indices[3];
        int total_layers = this->model->num_hidden_layers;
        if (num_feature_layers == 3 && total_layers == 32) {
            // 对于32层模型的标准配置
            layer_indices[0] = 2;                    // 浅层特征 (idx=2)
            layer_indices[1] = total_layers / 2;     // 中层特征 (idx=16)
            layer_indices[2] = total_layers - 3;     // 深层特征 (idx=29)
        } else {
            // 其他层数的模型，使用启发式分布
            layer_indices[0] = std::max(1, total_layers / 16);           // 浅层
            layer_indices[1] = total_layers / 2;                         // 中层
            layer_indices[2] = std::max(total_layers - 3, total_layers * 3 / 4); // 深层
        }
        
        // 检查是否有外部提供的base_model_hidden_states
        if (this->base_model_hidden_states != nullptr) {
            // 使用外部预拼接的特征（优先级较高，用于特殊场景）
            // 输入: [num_prev, 12288] (hidden_size * 3)
            cudaMemcpy(multi_layer_features, this->base_model_hidden_states,
                       num_prev * this->model->hidden_size * num_feature_layers * sizeof(T),
                       cudaMemcpyDeviceToDevice);
        } else {
            // 使用基础模型的真实多层输出
            const T* const* layer_outputs = this->model->get_eagle3_layer_outputs();
            
            // 拼接三层特征: [num_prev, 4096] x 3 -> [num_prev, 12288]
            // layer_outputs[0] = Layer 2 output
            // layer_outputs[1] = Layer 16 output  
            // layer_outputs[2] = Layer 29 output
            multi_layer_concat(calc_stream, num_prev, this->model->hidden_size,
                              layer_outputs[0], layer_outputs[1], layer_outputs[2],
                              multi_layer_features);
        }
        
        // Apply FC to concatenated multi-layer features
        this->fc->prefill(calc_stream, num_prev, multi_layer_features);
        
        // 调用Eagle3Layer的特殊prefill方法，传入分离的embedding和hidden_states
        // Eagle3Layer内部会：
        // 1. 保存hidden_states作为residual
        // 2. 拼接embedding和hidden_states为8192维用于attention
        // 3. attention输出后与residual相加，得到4096维
        // 4. 传给FFN的是4096维
        this->eagle_layer->prefill(num_prev, num_history_tokens, 
                                   this->eagle_embeddings->output,  // embedding: 4096维
                                   this->fc->output,                // hidden_states: 4096维
                                   this->eagle_position_ids, 
                                   this->kv_caches->caches[0],
                                   this->attention_input_buffer);   // concat buffer: 8192维
        
        // Apply output norm and store result
        this->output_norm->prefill(calc_stream, num_prev, this->eagle_layer->output, nullptr);
        cudaMemcpy(this->fc->output, this->output_norm->output, 
                   num_prev * this->model->hidden_size * sizeof(T), 
                   cudaMemcpyDeviceToDevice);
    }

    /**
     * EAGLE3 decode阶段
     * 
     * 功能：处理后续轮次的draft生成（最常用）
     * 
     * 与eagle_prefill的区别：
     * - eagle_prefill: 第一次初始化时调用
     * - eagle_decode: 后续所有轮次都调用这个
     * 
     * 重要：每次调用都会收到新的Base Model multi hidden states！
     * 
     * 步骤：
     * 1. 对候选token进行embedding
     * 2. 获取新的multi hidden states并通过FC层处理
     * 3. 通过单层decoder处理
     * 4. 应用输出归一化
     * 
     * @param cache_length KV cache的当前长度
     */
    void eagle_decode(int32_t* cache_length) {
        // Eagle3: Use eagle's own embeddings for candidate tokens
        this->eagle_embeddings->prefill(calc_stream, num_prev, this->tried_history_pos);
        
        // 检查是否有外部提供的base_model_hidden_states
        if (this->base_model_hidden_states != nullptr) {
            // 使用外部预拼接的特征（优先级较高，用于特殊场景）
            // 输入: [num_prev, 12288] (hidden_size * 3)
            cudaMemcpy(multi_layer_features, this->base_model_hidden_states,
                       num_prev * this->model->hidden_size * num_feature_layers * sizeof(T),
                       cudaMemcpyDeviceToDevice);
        } else {
            // 使用基础模型的真实多层输出（最常用的情况）
            // 这些特征来自最近一次base model的forward过程
            const T* const* base_layer_outputs = this->model->get_eagle3_layer_outputs();
            
            // 拼接三层特征: [num_prev, 4096] x 3 -> [num_prev, 12288]
            // base_layer_outputs[0] = Layer 2 output
            // base_layer_outputs[1] = Layer 16 output  
            // base_layer_outputs[2] = Layer 29 output
            multi_layer_concat(calc_stream, num_prev, this->model->hidden_size,
                              base_layer_outputs[0], base_layer_outputs[1], base_layer_outputs[2],
                              multi_layer_features);
        }
        
        // Apply FC to concatenated multi-layer features
        this->fc->prefill(calc_stream, num_prev, multi_layer_features);
        
        // 调用Eagle3Layer的特殊decode方法，传入分离的embedding和hidden_states
        this->eagle_layer->decode(num_prev, this->eagle_padded_length, 
                                  this->eagle_embeddings->output,  // embedding: 4096维
                                  this->fc->output,                // hidden_states: 4096维
                                  this->eagle_position_ids, cache_length, 
                                  Mask(nullptr), 
                                  this->kv_caches->caches[0],
                                  this->attention_input_buffer);   // concat buffer: 8192维
        
        // Apply output norm and store result
        this->output_norm->prefill(calc_stream, num_prev, this->eagle_layer->output, nullptr);
        cudaMemcpy(this->fc->output, this->output_norm->output, 
                   num_prev * this->model->hidden_size * sizeof(T), 
                   cudaMemcpyDeviceToDevice);
    }

    /**
     * 初始化阶段（Prefill）
     * 
     * 功能：处理输入序列的初始化，为后续的草稿生成做准备
     * 
     * 步骤：
     * 1. 使用基础模型的embedding处理输入
     * 2. 如果有历史token，调用eagle_prefill初始化EAGLE状态
     * 3. 保存最后一个token的embedding（用于后续处理）
     * 4. 调用基础模型的prefill_embed处理
     * 5. 保存最终的hidden state
     * 6. 复制position_ids供EAGLE使用
     * 7. 设置状态标志
     * 
     * @param num_tokens        当前输入的token数量
     * @param num_history_tokens 历史token数量
     * @param input            输入token ID数组
     * @param position_ids     位置ID数组
     * @param output           输出缓冲区
     */
    void prefill(int32_t num_tokens, int32_t num_history_tokens, int32_t* input, int32_t* position_ids, void* output) {
        this->model->embedding->prefill(calc_stream, num_tokens, input);
        if (num_history_tokens > 0) {
            this->eagle_prefill(this->num_history_tokens);
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
    
    /**
     * 更新multi hidden states
     * 
     * 功能：接收Base Model验证阶段产生的multi hidden states
     * 这个方法应该在tree_decoding之后，下一轮draft之前调用
     * 
     * @param multi_hidden_states 来自Base Model的多层拼接特征
     *                           维度：[num_tokens, hidden_size * 3] = [num_tokens, 12288]
     *                           包含Layer 2, 16, 29的输出按顺序拼接
     * @param num_tokens 特征对应的token数量
     */
    void update_multi_hidden_states(T* multi_hidden_states, int32_t num_tokens) {
        // 保存Base Model传来的multi hidden states供后续使用
        // 维度: num_tokens * hidden_size * 3 = num_tokens * 12288
        cudaMemcpy(this->base_model_hidden_states, multi_hidden_states, 
                   num_tokens * this->model->hidden_size * num_feature_layers * sizeof(T), 
                   cudaMemcpyDeviceToDevice);
    }

    /**
     * 草稿生成阶段（Draft）
     * 
     * 功能：生成树形结构的草稿token候选
     * 
     * 主要步骤：
     * 1. 准备阶段：
     *    - 获取当前cache长度并计算对齐后的长度
     *    - 根据is_first_draft决定调用eagle_prefill或eagle_decode
     *    
     * 2. 迭代生成候选（num_iter轮）：
     *    - d=0: 生成第一层候选
     *      * 使用多层特征融合后的结果（已经过FC处理）
     *      * 通过eagle_layer+lm_head生成logits和hidden states
     *      * 选择top-k个候选
     *      * 初始化树结构
     *    - d>0: 递归扩展树结构
     *      * 对当前候选进行embedding
     *      * 使用上一轮的hidden states作为输入
     *      * 通过eagle_layer递归生成新的hidden states和logits
     *      * 累积概率并选择新的候选
     *      * 更新树的mask和parent关系
     *      
     * 3. 最终树构建：
     *    - 从所有候选中选择最优的tree_size-1个
     *    - 构建完整的树形结构（位置、mask、parent）
     *    - 如需要，进行词汇表映射
     * 
     * @param tree_draft_ids    输出的草稿token ID数组
     * @param tree_position_ids 输出的位置ID数组
     * @param cache_length      KV cache长度
     * @param tree_attn_mask    输出的attention mask
     * @param tree_parent       输出的父节点索引数组
     */
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

        { // d = 0: 第一层候选生成
            // Eagle3: Use eagle's own lm_head
            this->eagle_lm_head->prefill(calc_stream, 1, this->fc->output + (num_prev - 1) * this->model->hidden_size, this->eagle_logits);
            log_softmax(calc_stream, 1, this->draft_vocab_size, this->eagle_logits);
            this->topk_func->prefill(calc_stream, 1, this->eagle_logits);
            
            // 保存第一轮的top-k结果
            cudaMemcpy(this->topk_func_2->topk_pos, this->topk_func->topk_pos, topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->topk_func_2->topk_val, this->topk_func->topk_val, topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->tried_history_val, this->topk_func->topk_val, topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->tried_history_pos, this->topk_func->topk_pos, topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            // 注意：tried_history_pos保存的是draft token id，不进行映射
            // 映射只在最后构建树时进行
            
            // 复制hidden state给每个候选
            repeat(calc_stream, topk_per_iter, this->model->hidden_size, num_prev-1, this->fc->output, this->fc->output);
            // 初始化树的mask（每个候选独立）
            init_tree(calc_stream, topk_per_iter, this->eagle_mask_2d);
        }
        // 迭代扩展树结构
        for (int d = 1; d < this->num_iter; ++d) {
            // 更新cache长度
            add(calc_stream, 1, this->eagle_cache_length, topk_per_iter);
            // Eagle3: Use eagle's own embeddings for new candidate tokens
            this->eagle_embeddings->prefill(calc_stream, topk_per_iter, this->topk_func_2->topk_pos);
            
            // 递归处理：调用Eagle3Layer的特殊decode方法
            this->eagle_layer->decode(topk_per_iter, this->eagle_padded_length, 
                                      this->eagle_embeddings->output,  // 新的embedding: 4096维
                                      this->fc->output,                // 上一轮的hidden states: 4096维
                                      this->eagle_position_ids, this->eagle_cache_length, 
                                      Mask(eagle_mask_2d, topk_per_iter, topk_per_iter * d), 
                                      this->kv_caches->caches[0],
                                      this->attention_input_buffer);   // concat buffer: 8192维
            
            // Apply output norm and prepare for next iteration
            this->output_norm->prefill(calc_stream, topk_per_iter, this->eagle_layer->output, nullptr);
            // 将输出存储回fc->output供下一轮使用
            cudaMemcpy(this->fc->output, this->output_norm->output, 
                       topk_per_iter * this->model->hidden_size * sizeof(T), 
                       cudaMemcpyDeviceToDevice);
            
            // 更新位置ID
            add(calc_stream, topk_per_iter, this->eagle_position_ids, 1);

            // 生成下一层的logits
            // Eagle3: Use eagle's own lm_head
            this->eagle_lm_head->prefill(calc_stream, topk_per_iter, this->fc->output, this->eagle_logits);
            log_softmax(calc_stream, topk_per_iter, this->draft_vocab_size, this->eagle_logits);
            
            // 对每个父节点，选择top-k个子节点
            this->topk_func->prefill(calc_stream, topk_per_iter, this->eagle_logits);
            
            // 累积概率：将父节点的概率加到子节点上
            cumsum(calc_stream, topk_per_iter, topk_per_iter, this->topk_func->topk_val, this->topk_func_2->topk_val);
            
            // 保存本轮所有候选
            cudaMemcpy(this->tried_history_val + topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter, this->topk_func->topk_val, topk_per_iter * topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            
            // 保存draft token id，不进行映射
            cudaMemcpy(this->tried_history_pos + topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter, this->topk_func->topk_pos, topk_per_iter * topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            
            // 从累积的候选中选择最好的topk_per_iter个
            this->topk_func_2->prefill(calc_stream, 1, this->topk_func->topk_val, topk_per_iter * topk_per_iter, topk_per_iter);

            // 更新树结构：mask和parent关系
            cudaMemcpy(this->tmp_mask_2d, this->eagle_mask_2d, topk_per_iter * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
            set_parent(calc_stream, topk_per_iter, this->tried_history_parent + (d - 1) * topk_per_iter, this->topk_func_2->topk_pos, topk_per_iter + (d - 1) * topk_per_iter * topk_per_iter);
            update_tree(calc_stream, topk_per_iter, topk_per_iter * d, this->eagle_mask_2d, this->tmp_mask_2d, this->topk_func_2->topk_pos);
            
            // 为下一轮准备：重新映射hidden states和token IDs
            remap_hidden(calc_stream, topk_per_iter, this->model->hidden_size, this->topk_func_2->topk_pos, this->fc->output, this->fc->output, topk_per_iter);
            remap_id(calc_stream, topk_per_iter, this->topk_func_2->topk_pos, this->topk_func->topk_pos);
        }

        // 从所有尝试过的候选中选择最终的tree_size-1个
        this->topk_func_2->prefill(calc_stream, 1, this->tried_history_val);

        // 构建最终的树形结构
        build_dynamic_tree(calc_stream, this->tree_size, this->eagle_original_length[0], this->topk_per_iter, this->tried_history_parent, this->topk_func_2->topk_pos, tree_position_ids, tree_attn_mask, tree_parent);
        
        // 将选中的token ID映射到输出数组（+1是因为第0个位置留给当前token）
        remap_id(calc_stream, this->tree_size-1, this->topk_func_2->topk_pos, this->tried_history_pos, tree_draft_ids + 1);
        
        // 如果使用了词汇表映射，将draft token id映射回target token id
        if (need_vocab_mapping && d2t_mapping != nullptr) {
            vocab_mapping(calc_stream, this->tree_size-1, tree_draft_ids + 1, d2t_mapping, tree_draft_ids + 1);
        }

        // 标记已完成第一次草稿生成
        this->is_first_draft = false;
    }

    /**
     * 验证阶段（Verify）
     * 
     * 功能：验证草稿token，找出最长的匹配序列
     * 
     * 步骤：
     * 1. 调用verify_draft kernel进行并行验证
     * 2. 获取最佳匹配长度（同步到CPU）
     * 3. 根据验证结果更新状态：
     *    - 重新映射hidden states到接受的token
     *    - 修复基础模型的KV cache
     *    - TODO: 可能需要修复EAGLE3的KV cache
     * 4. 为接受的token生成embedding
     * 5. 更新position IDs为下一轮准备
     * 
     * 注意：多层特征提取现在可以通过两种方式：
     * 1. 自动方式（推荐）：直接使用model->get_eagle3_layer_outputs()获取Layer 2,16,29输出
     * 2. 手动方式：外部调用update_multi_hidden_states传递预拼接的12288维特征
     * 
     * @param num_tokens    草稿token总数
     * @param pred         预测的token IDs
     * @param gt           ground truth token IDs
     * @param position_ids  位置IDs
     * @param cache_length  KV cache长度
     * @param mask_2d      2D attention mask
     * @param tree_parent  树的父节点关系
     * @return             接受的token数量
     */
    int verify(int32_t num_tokens, int32_t* pred, int32_t* gt, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, int32_t* tree_parent) {
        verify_draft(calc_stream, num_tokens, pred, gt, position_ids, cache_length, mask_2d, tree_parent, this->d_best);
        cudaMemcpyAsync(this->h_best, this->d_best, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, calc_stream.stream);
        cudaStreamSynchronize(calc_stream.stream);

        this->num_prev = h_best[0];
        remap_hidden(calc_stream, this->num_prev, this->model->hidden_size, pred, this->model->norm->output, this->prev_hidden_state);

        // 修复基础模型的KV cache（EAGLE3与EAGLE2使用完全相同的KV cache处理）
        fix_kv_cache(calc_stream, h_best[0], this->model->kv_caches->num_hidden_layers * 2, this->model->kv_caches->dim, pred, gt, cache_length, this->model->kv_caches->d_flat_caches, this->tmp_kvcache);

        this->model->embedding->prefill(calc_stream, this->num_prev, pred);
        cudaMemcpy(this->prev_embed, this->model->embedding->output, this->num_prev * this->model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

        make_arange(calc_stream, this->num_prev, cache_length, this->eagle_position_ids);

        return h_best[0];
    }
};

/*
================================================================================
                            EAGLE3 实现总结
================================================================================

关键改进点：
1. 架构简化：从多层变为单层，大幅减少参数和计算量
2. 多层特征融合：利用基础模型的多层信息，提高草稿质量
3. 独立组件：拥有自己的embeddings和lm_head，支持词汇表优化

关键设计理解：
1. 多层特征提取：
   - 不是简单的最后3层，而是分布式采样（浅层/中层/深层）
   - 32层模型：Layer 2, 16, 29
   - 每层4096维，拼接后12288维（4096 * 3）
   - 通过FC层压缩：12288 → 4096

2. LlamaDecoderLayeremb的特殊性：
   - 内部拼接：embedding(4096) + hidden_states(4096) = 8192维
   - Attention的q/k/v_proj处理8192维输入
   - 输出仍是4096维
   - 这解释了为什么需要保留prefill/decode区分

3. 词汇表映射：
   - 基础模型：73448 (如MiniCPM4)
   - EAGLE3 draft：32000（更小，提高效率）
   - 仅需要d2t映射表（推理时）
   - t2d仅用于训练，推理时不需要

4. KV Cache处理：
   - EAGLE3与EAGLE2使用完全相同的KV cache机制
   - 无需特殊的eagle3_kv_fix_kernel
   - 直接复用现有的fix_kv_cache函数

待完成工作：
1. ✅ 基础模型多层特征提取（已通过get_eagle3_layer_outputs解决）
2. ✅ embedding+hidden_states拼接（已通过Eagle3Layer和concat函数解决）
3. ✅ KV cache处理（与EAGLE2一致，无需特殊处理）
4. MiniCPM4参数设置（is_minicpm4_native, scale_factor）

性能优化建议：
- 使用更高效的内存布局减少拷贝
- 优化树形结构构建算法
- 考虑使用half精度计算
- 批量处理多个序列

================================================================================
*/