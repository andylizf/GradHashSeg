#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#define MY_CUDA_CHECK(EXPR)                                                                                                                    \
    do {                                                                                                                                       \
        cudaError_t __err = EXPR;                                                                                                              \
        if (__err != cudaSuccess) {                                                                                                            \
            auto error_unused C10_UNUSED = cudaGetLastError();                                                                                 \
            TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(__err), "File: ", __FILE__, "Line: ", __LINE__, "Function: ", __FUNCTION__); \
        }                                                                                                                                      \
    } while (0)

#define CUDA_NUM_THREADS 256
#define CUDA_1D_KERNEL_LOOP(i, n)                                   \
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)
inline int GET_BLOCKS(const uint32_t N)
{
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename scalar_t>
__global__ void grad_hash_kernel(
    const int* vector_ids,
    const int* buckets_index_inv,
    const scalar_t* gradIndex,
    const int* buckets_count,
    scalar_t* grad_Hash_value,
    int n_matrices,
    int num_rows,
    int max_buckets,
    double sigma)
{
    CUDA_1D_KERNEL_LOOP(index, n_matrices * num_rows * max_buckets)
    {
        uint32_t bucket_id = index % max_buckets;
        index = index / max_buckets;
        uint32_t row_id = index % num_rows;
        uint32_t matrix_id = index / num_rows;

        uint32_t vector_id = vector_ids[matrix_id * num_rows + row_id] + 1;
        uint32_t bucket_index = buckets_index_inv[matrix_id * max_buckets + bucket_id] + 1;
        scalar_t grad_index = gradIndex[index];
        uint32_t bucket_count = buckets_count[matrix_id * max_buckets + bucket_id];

        scalar_t grad_Hash_value_temp = static_cast<scalar_t>(vector_id) / static_cast<scalar_t>(bucket_index) - 1;
        scalar_t neg_sigma_sq = -sigma * sigma;

        scalar_t grad_Hash_val = grad_Hash_value_temp / neg_sigma_sq;
        grad_Hash_val *= exp(grad_Hash_value_temp * grad_Hash_val / 2);
        grad_Hash_val *= grad_index;
        grad_Hash_val /= bucket_count;

        grad_Hash_value[index] = isnan(grad_Hash_val) ? 0 : grad_Hash_val;
    }
}

template <typename scalar_t>
__global__ void grad_hash_kernel3(
    const int* __restrict__ vector_ids,
    const int* __restrict__ buckets_index_inv,
    const scalar_t* __restrict__ gradIndex,
    const int* __restrict__ buckets_count,
    scalar_t* __restrict__ grad_Hash_value,
    int n_matrices,
    int num_rows,
    int max_buckets,
    double sigma)
{
    CUDA_1D_KERNEL_LOOP(index, n_matrices * num_rows * max_buckets)
    {
        uint32_t bucket_id = index % max_buckets;
        index = index / max_buckets;
        uint32_t row_id = index % num_rows;
        uint32_t matrix_id = index / num_rows;

        uint32_t vector_id = vector_ids[matrix_id * num_rows + row_id] + 1;
        uint32_t bucket_index = buckets_index_inv[matrix_id * max_buckets + bucket_id] + 1;
        scalar_t grad_index = gradIndex[index];
        uint32_t bucket_count = buckets_count[matrix_id * max_buckets + bucket_id];

        scalar_t grad_Hash_value_temp = static_cast<scalar_t>(vector_id) / static_cast<scalar_t>(bucket_index) - 1;
        scalar_t neg_sigma_sq = -sigma * sigma;

        scalar_t grad_Hash_val = grad_Hash_value_temp / neg_sigma_sq;
        grad_Hash_val *= __expf(grad_Hash_value_temp * grad_Hash_val / 2);
        grad_Hash_val *= grad_index;
        grad_Hash_val /= bucket_count;

        grad_Hash_value[index] = __isnanf(grad_Hash_val) ? 0 : grad_Hash_val;
    }
}

template <typename scalar_t>
__global__ void grad_hash_kernel2(
    const at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> vector_ids,
    const at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> buckets_index_inv,
    const scalar_t* gradIndex,
    const at::PackedTensorAccessor32<int, 2, at::RestrictPtrTraits> buckets_count,
    scalar_t* grad_Hash_value,
    int n_matrices,
    int num_rows,
    int max_buckets,
    double sigma)
{
    CUDA_1D_KERNEL_LOOP(index, n_matrices * num_rows * max_buckets)
    {
        uint32_t bucket_id = index % max_buckets;
        index = index / max_buckets;
        uint32_t row_id = index % num_rows;
        uint32_t matrix_id = index / num_rows;

        unsigned vector_id = vector_ids[matrix_id][row_id] + 1;
        unsigned bucket_index = buckets_index_inv[matrix_id][bucket_id] + 1;
        scalar_t grad_index = gradIndex[index];
        unsigned bucket_count = buckets_count[matrix_id][bucket_id];

        scalar_t grad_Hash_value_temp = static_cast<scalar_t>(vector_id) / static_cast<scalar_t>(bucket_index) - 1;
        scalar_t neg_sigma_sq = -sigma * sigma;

        scalar_t grad_Hash_val = grad_Hash_value_temp / neg_sigma_sq;
        grad_Hash_val *= exp(grad_Hash_value_temp * grad_Hash_val / 2);
        grad_Hash_val *= grad_index;
        grad_Hash_val /= bucket_count;

        grad_Hash_value[index] = isnan(grad_Hash_val) ? 0 : grad_Hash_val;
    }
}

extern int max_buckets, num_rows, n_matrices, param_H;
extern double sigma;
extern at::Tensor vector_ids, buckets_index_inv, buckets_count, gradIndex;

auto grad_hash_cuda()
{
    auto grad_Hash_value = torch::zeros({ n_matrices, num_rows, max_buckets }, gradIndex.options());

    AT_DISPATCH_FLOATING_TYPES(gradIndex.scalar_type(), "grad_hash_kernel", ([&] {
        grad_hash_kernel<scalar_t>
            <<<GET_BLOCKS(num_rows * n_matrices * max_buckets), CUDA_NUM_THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
                vector_ids.data_ptr<int>(),
                buckets_index_inv.data_ptr<int>(),
                gradIndex.data_ptr<scalar_t>(),
                buckets_count.data_ptr<int>(),
                grad_Hash_value.data_ptr<scalar_t>(),
                n_matrices,
                num_rows,
                max_buckets,
                sigma);
    }));
    MY_CUDA_CHECK(cudaGetLastError());
    return grad_Hash_value;
}

auto grad_hash_cuda2()
{
    auto grad_Hash_value = torch::zeros({ n_matrices, num_rows, max_buckets }, gradIndex.options());

    AT_DISPATCH_FLOATING_TYPES(gradIndex.scalar_type(), "grad_hash_kernel2", ([&] {
        grad_hash_kernel2<scalar_t>
            <<<GET_BLOCKS(num_rows * n_matrices * max_buckets), CUDA_NUM_THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
                vector_ids.packed_accessor32<int, 2, at::RestrictPtrTraits>(),
                buckets_index_inv.packed_accessor32<int, 2, at::RestrictPtrTraits>(),
                gradIndex.data_ptr<scalar_t>(),
                buckets_count.packed_accessor32<int, 2, at::RestrictPtrTraits>(),
                grad_Hash_value.data_ptr<scalar_t>(),
                n_matrices,
                num_rows,
                max_buckets,
                sigma);
    }));
    MY_CUDA_CHECK(cudaGetLastError());
    return grad_Hash_value;
}

auto grad_hash_cuda3()
{
    auto grad_Hash_value = torch::zeros({ n_matrices, num_rows, max_buckets }, gradIndex.options());

    AT_DISPATCH_FLOATING_TYPES(gradIndex.scalar_type(), "grad_hash_kernel3", ([&] {
        grad_hash_kernel3<scalar_t>
            <<<GET_BLOCKS(num_rows * n_matrices * max_buckets), CUDA_NUM_THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
                vector_ids.data_ptr<int>(),
                buckets_index_inv.data_ptr<int>(),
                gradIndex.data_ptr<scalar_t>(),
                buckets_count.data_ptr<int>(),
                grad_Hash_value.data_ptr<scalar_t>(),
                n_matrices,
                num_rows,
                max_buckets,
                sigma);
    }));
    MY_CUDA_CHECK(cudaGetLastError());
    return grad_Hash_value;
}