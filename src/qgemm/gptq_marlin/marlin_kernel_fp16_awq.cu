#include "marlin_kernel_impl.cuh"

#ifdef ENABLE_DTYPE_FP16

// AWQ_CALL_IF(vllm::kU4, 16, 4, 256) expansions
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 1, 16, 4, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 1, 16, 4, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 1, 16, 4, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 1, 16, 4, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 2, 16, 4, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 2, 16, 4, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 2, 16, 4, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 2, 16, 4, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 3, 16, 4, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 3, 16, 4, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 3, 16, 4, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 3, 16, 4, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 4, 16, 4, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 4, 16, 4, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 4, 16, 4, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 4, 16, 4, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);


// AWQ_CALL_IF(vllm::kU4, 8, 8, 256) expansions
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 1, 8, 8, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 1, 8, 8, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 1, 8, 8, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 1, 8, 8, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 2, 8, 8, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 2, 8, 8, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 2, 8, 8, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 2, 8, 8, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 3, 8, 8, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 3, 8, 8, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 3, 8, 8, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 3, 8, 8, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 4, 8, 8, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 4, 8, 8, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 4, 8, 8, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 256, 4, 8, 8, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);

// AWQ_CALL_IF(vllm::kU4, 8, 4, 128) expansions
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 1, 8, 4, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 1, 8, 4, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 1, 8, 4, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 1, 8, 4, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 2, 8, 4, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 2, 8, 4, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 2, 8, 4, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 2, 8, 4, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 3, 8, 4, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 3, 8, 4, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 3, 8, 4, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 3, 8, 4, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 4, 8, 4, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 4, 8, 4, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 4, 8, 4, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 4, 8, 4, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);

// AWQ_CALL_IF(vllm::kU4, 4, 8, 128) expansions
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 1, 4, 8, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 1, 4, 8, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 1, 4, 8, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 1, 4, 8, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 2, 4, 8, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 2, 4, 8, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 2, 4, 8, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 2, 4, 8, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 3, 4, 8, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 3, 4, 8, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 3, 4, 8, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 3, 4, 8, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 4, 4, 8, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 4, 4, 8, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 4, 4, 8, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU4.id(), 128, 4, 4, 8, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);

// AWQ_CALL_IF(vllm::kU8, 16, 4, 256) expansions
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 1, 16, 4, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 1, 16, 4, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 1, 16, 4, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 1, 16, 4, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 2, 16, 4, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 2, 16, 4, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 2, 16, 4, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 2, 16, 4, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 3, 16, 4, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 3, 16, 4, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 3, 16, 4, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 3, 16, 4, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 4, 16, 4, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 4, 16, 4, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 4, 16, 4, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 4, 16, 4, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);

// AWQ_CALL_IF(vllm::kU8, 8, 8, 256) expansions
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 1, 8, 8, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 1, 8, 8, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 1, 8, 8, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 1, 8, 8, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 2, 8, 8, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 2, 8, 8, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 2, 8, 8, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 2, 8, 8, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 3, 8, 8, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 3, 8, 8, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 3, 8, 8, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 3, 8, 8, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 4, 8, 8, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 4, 8, 8, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 4, 8, 8, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 256, 4, 8, 8, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);

// AWQ_CALL_IF(vllm::kU8, 8, 4, 128) expansions
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 1, 8, 4, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 1, 8, 4, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 1, 8, 4, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 1, 8, 4, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 2, 8, 4, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 2, 8, 4, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 2, 8, 4, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 2, 8, 4, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 3, 8, 4, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 3, 8, 4, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 3, 8, 4, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 3, 8, 4, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 4, 8, 4, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 4, 8, 4, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 4, 8, 4, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 4, 8, 4, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);

// AWQ_CALL_IF(vllm::kU8, 4, 8, 128) expansions
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 1, 4, 8, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 1, 4, 8, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 1, 4, 8, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 1, 4, 8, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 2, 4, 8, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 2, 4, 8, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 2, 4, 8, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 2, 4, 8, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 3, 4, 8, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 3, 4, 8, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 3, 4, 8, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 3, 4, 8, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 4, 4, 8, 4, false, true, -1>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 4, 4, 8, 4, false, true, 2>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 4, 4, 8, 4, false, true, 4>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);
template __global__ void marlin::Marlin<__half, vllm::kU8.id(), 128, 4, 4, 8, 4, false, true, 8>(const int4*, const int4*, int4*, int4*, const int4*, const int4*, const int*, int, int, int, int, int*, bool);

#endif 