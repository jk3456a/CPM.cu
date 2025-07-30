#pragma once
#include <cuda_runtime.h>
#include "../utils.cuh"

namespace {
template <typename T>
__global__ void embedding_kernel(int32_t num_cols, const int32_t* input, const float4* weight, float4* output) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    int offset_output = row * num_cols;
    int offset_weight = input[row] * num_cols;
    for (int i = col; i < num_cols; i += blockDim.x) {
        output[offset_output + i] = weight[offset_weight + i];
    }
}

template <typename T>
void embedding(const Stream& stream, int32_t num_tokens, int32_t hidden_size, const int32_t* input, const T* weight, T* output) {
    embedding_kernel<T><<<num_tokens, 256, 0, stream.stream>>>(hidden_size/(16/sizeof(T)), input, (float4*)weight, (float4*)output);
}
} // namespace

template <typename T>
struct Embedding {
    int vocab_size;
    int hidden_size;
    T* weight;
    T* output;
    float embed_scale;

    Embedding(int vocab_size, int hidden_size, float embed_scale = 1.0f) {
        this->vocab_size = vocab_size;
        this->hidden_size = hidden_size;
        this->embed_scale = embed_scale;
    }

    void init_weight_ptr(Memory* memory) {
        weight = (T*)memory->allocate_for_model(vocab_size * hidden_size * sizeof(T));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        return memory->allocate((void**)&this->output, offset, num_tokens * hidden_size * sizeof(T));
    }

    void load_to_storage(std::string name, void* ptr) {
        cudaMemcpy((void*)weight, ptr, vocab_size * hidden_size * sizeof(T), cudaMemcpyHostToDevice);
    }

    void prefill(const Stream& stream, int32_t num_tokens, int32_t* input) {
        int32_t* h_input = new int32_t[num_tokens];
        cudaMemcpy(h_input, input, num_tokens * sizeof(int32_t), cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_tokens; i++) {
            printf("embedding->prefill input[%d] = %d\n", i, h_input[i]);
        }
        delete[] h_input;
        //查表 查weight表
        embedding(stream, num_tokens, this->hidden_size, input, this->weight, this->output);
        
        // 添加调试信息：显示数据类型信息
        printf("=== Embedding Debug Info ===\n");
        printf("sizeof(T) = %zu bytes\n", sizeof(T));
        printf("num_tokens = %d, hidden_size = %d\n", num_tokens, this->hidden_size);
        printf("Total elements = %d\n", num_tokens * this->hidden_size);
        printf("Total memory = %zu bytes\n", num_tokens * this->hidden_size * sizeof(T));
        
        // 修复：使用正确的数据类型T而不是假设是float
        T* h_output = new T[num_tokens * this->hidden_size];
        cudaMemcpy(h_output, this->output, num_tokens * this->hidden_size * sizeof(T), cudaMemcpyDeviceToHost);
        
        printf("=== Embedding Output Values ===\n");
        for (int i = 0; i < num_tokens; i++) {
            for (int j = 0; j < 10; j++) {
                // 将T类型转换为float进行输出
                printf("embedding->prefill output[%d][%d] = %f\n", i, j, (float)h_output[i * this->hidden_size + j]);
            }
        }
        printf("=============================\n");
        
        delete[] h_output;
        
        elementwise_scale(stream, num_tokens, this->hidden_size, this->output, this->embed_scale);
    }
};