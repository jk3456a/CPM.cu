#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "../utils.cuh"  // 需要Stream类型

// EAGLE3特化算子

// 多层特征拼接算子 - 高效拼接来自不同层的特征
// 将3个[num_tokens, hidden_size]的特征拼接成[num_tokens, hidden_size*3]
template<typename T>
__global__ void multi_layer_concat_kernel(
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t num_layers,     // 固定为3
    const T* layer0,        // Layer 2的输出 [num_tokens, hidden_size]
    const T* layer1,        // Layer 16的输出 [num_tokens, hidden_size]
    const T* layer2,        // Layer 29的输出 [num_tokens, hidden_size]
    T* output              // 拼接后的输出 [num_tokens, hidden_size*3]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * hidden_size;
    
    if (tid < total_elements) {
        int token_idx = tid / hidden_size;
        int hidden_idx = tid % hidden_size;
        
        // 将三层特征按顺序拼接
        // output[token][0:4096] = layer0[token][:]
        // output[token][4096:8192] = layer1[token][:]
        // output[token][8192:12288] = layer2[token][:]
        
        if (layer0 != nullptr) {
            output[token_idx * hidden_size * num_layers + hidden_idx] = layer0[tid];
        }
        if (layer1 != nullptr) {
            output[token_idx * hidden_size * num_layers + hidden_size + hidden_idx] = layer1[tid];
        }
        if (layer2 != nullptr) {
            output[token_idx * hidden_size * num_layers + 2 * hidden_size + hidden_idx] = layer2[tid];
        }
    }
}

// 词汇表映射算子 - 将draft token id映射到target token id
__global__ void vocab_mapping_kernel(
    int32_t num_tokens,
    const int32_t* draft_token_ids,    // draft模型的token id
    const int64_t* d2t_mapping,        // draft到target的映射表 (torch.int64)
    int32_t* target_token_ids          // 输出的target token id
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_tokens) {
        int32_t draft_id = draft_token_ids[tid];
        // d2t_mapping是直接映射表: target_id = d2t_mapping[draft_id]
        target_token_ids[tid] = (int32_t)d2t_mapping[draft_id];
    }
}

// 拼接embedding和hidden states的kernel
template<typename T>
__global__ void concat_embeddings_hidden_kernel(
    int32_t num_tokens,
    int32_t hidden_size,
    const T* embeddings,    // [num_tokens, hidden_size]
    const T* hidden_states, // [num_tokens, hidden_size]  
    T* output              // [num_tokens, hidden_size*2]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * hidden_size;
    
    if (tid < total_elements) {
        int token_idx = tid / hidden_size;
        int hidden_idx = tid % hidden_size;
        
        // 前4096是embedding
        output[token_idx * hidden_size * 2 + hidden_idx] = embeddings[tid];
        // 后4096是hidden_states
        output[token_idx * hidden_size * 2 + hidden_size + hidden_idx] = hidden_states[tid];
    }
}

// 拼接函数封装
template<typename T>
void concat_embeddings_and_hidden(const Stream& stream, int32_t num_tokens, 
                                  int32_t hidden_size,
                                  const T* embeddings, const T* hidden_states, T* output) {
    int block_size = 256;
    int total_elements = num_tokens * hidden_size;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    concat_embeddings_hidden_kernel<<<grid_size, block_size, 0, stream.stream>>>(
        num_tokens, hidden_size, embeddings, hidden_states, output);
    cudaCheck(cudaGetLastError());
}


// EAGLE3特化的多层特征拼接封装
template<typename T>
void multi_layer_concat(const Stream& stream, int32_t num_tokens, int32_t hidden_size,
                        const T* layer0, const T* layer1, const T* layer2, T* output) {
    int block_size = 256;
    int total_elements = num_tokens * hidden_size;
    int grid_size = (total_elements + block_size - 1) / block_size;
    multi_layer_concat_kernel<<<grid_size, block_size, 0, stream.stream>>>(
        num_tokens, hidden_size, 3, layer0, layer1, layer2, output);
    cudaCheck(cudaGetLastError());
}

// EAGLE3特化的词汇表映射封装
void vocab_mapping(const Stream& stream, int32_t num_tokens, const int32_t* draft_token_ids, 
                   const int64_t* d2t_mapping, int32_t* target_token_ids) {
    int block_size = 256;
    int grid_size = (num_tokens + block_size - 1) / block_size;
    vocab_mapping_kernel<<<grid_size, block_size, 0, stream.stream>>>(
        num_tokens, draft_token_ids, d2t_mapping, target_token_ids);
    cudaCheck(cudaGetLastError());
}