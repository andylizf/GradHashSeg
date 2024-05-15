#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <chrono>
#include <iostream>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/cuda.h>
#include <torch/torch.h>
#include <torch/types.h>

int max_buckets = 170;
int num_rows = 1000;
int n_matrices = 160;
int param_H = 10;
double sigma = 0.000195313;
at::Tensor vector_ids, buckets_index_inv, buckets_count, gradIndex;

using grad_hash_func = at::Tensor (*)();

void some_function(bool useCuda)
{
    using namespace torch::autograd::profiler;
    ProfilerConfig cfg {
        ProfilerState::KINETO,
        false,
        false,
        false,
        false,
        false
    };
    std::set<torch::autograd::profiler::ActivityType> activities { torch::autograd::profiler::ActivityType::CPU };
    if (useCuda) {
        activities.insert(torch::autograd::profiler::ActivityType::CUDA);
    }
    prepareProfiler(cfg, activities);
    enableProfiler(cfg, activities);
    // Your code here
    auto result = disableProfiler();
    result->save("./some_local_file.json");
}

auto grad_hash0() -> at::Tensor
{
    at::Tensor grad_Hash_value = (vector_ids.unsqueeze(2).repeat({ 1, 1, max_buckets }) + 1).to(gradIndex.options()) / (buckets_index_inv.unsqueeze(1).repeat({ 1, num_rows, 1 }) + 1).to(gradIndex.options()) - 1;
    grad_Hash_value = -1 * grad_Hash_value / (sigma * sigma) * exp(-1 * grad_Hash_value * grad_Hash_value / (2 * sigma * sigma)) * gradIndex / buckets_count.unsqueeze(1).repeat({ 1, num_rows, 1 }).to(gradIndex.options());
    at::Tensor zero = at::zeros({ n_matrices, num_rows, max_buckets }, gradIndex.options());
    grad_Hash_value = at::where(grad_Hash_value.isnan(), zero, grad_Hash_value);
    return grad_Hash_value;
}

auto grad_hash1() -> at::Tensor
{
    auto vector_ids_expanded = (vector_ids + 1).unsqueeze(2).expand({ -1, -1, max_buckets });
    auto buckets_index_inv_expanded = (buckets_index_inv + 1).unsqueeze(1).expand({ -1, num_rows, -1 });

    // 计算 grad_Hash_value
    auto grad_Hash_value = (vector_ids_expanded / buckets_index_inv_expanded - 1).to(gradIndex.options());

    // 计算 exp_component，避免重复计算
    auto grad_Hash_value_sq = grad_Hash_value * grad_Hash_value;
    auto neg_sigma_sq = -sigma * sigma;
    grad_Hash_value_sq.div_(2 * neg_sigma_sq).exp_();

    // 计算 grad_Hash_value
    auto buckets_count_expanded = buckets_count.unsqueeze(1).expand({ -1, num_rows, -1 });
    grad_Hash_value.div_(neg_sigma_sq).mul_(gradIndex);
    grad_Hash_value_sq.div_(buckets_count_expanded);
    grad_Hash_value.mul_(grad_Hash_value_sq);

    // 创建 zero 张量
    auto zero = torch::zeros({ n_matrices, num_rows, max_buckets }, gradIndex.options());

    // 使用 where 操作处理 NaN 值
    grad_Hash_value = torch::where(grad_Hash_value.isnan(), zero, grad_Hash_value);

    return grad_Hash_value;
}

auto grad_hash2() -> at::Tensor
{
    auto vector_ids_expanded = (vector_ids + 1).unsqueeze(2).expand({ -1, -1, max_buckets }).to(gradIndex.options());
    auto buckets_index_inv_expanded = (buckets_index_inv + 1).unsqueeze(1).expand({ -1, num_rows, -1 });

    // 计算 grad_Hash_value
    auto grad_Hash_value = vector_ids_expanded.div_(buckets_index_inv_expanded).sub_(1);

    // 计算 exp_component，避免重复计算
    auto grad_Hash_value_sq = grad_Hash_value.square();
    auto neg_sigma_sq = -sigma * sigma;
    grad_Hash_value_sq.div_(2 * neg_sigma_sq).exp_();

    // 计算 grad_Hash_value
    grad_Hash_value.div_(neg_sigma_sq).mul_(gradIndex);
    grad_Hash_value_sq.div_(buckets_count.unsqueeze(1).expand({ -1, num_rows, -1 }));
    grad_Hash_value.mul_(grad_Hash_value_sq);

    // 创建 zero 张量
    auto zero = torch::zeros({ n_matrices, num_rows, max_buckets }, gradIndex.options());

    // 使用 where 操作处理 NaN 值
    grad_Hash_value = torch::where(grad_Hash_value.isnan(), zero, grad_Hash_value);

    return grad_Hash_value;
}

auto grad_hash3() -> at::Tensor
{
    at::Tensor grad_Hash_value = (vector_ids + 1).unsqueeze(2).expand({ -1, -1, max_buckets }) / (buckets_index_inv + 1).unsqueeze(1).expand({ -1, num_rows, -1 }) - 1;
    grad_Hash_value = -grad_Hash_value / (sigma * sigma) * exp(-grad_Hash_value * grad_Hash_value / (2 * sigma * sigma)) * gradIndex / buckets_count.unsqueeze(1).expand({ -1, num_rows, -1 }).to(gradIndex.options());
    at::Tensor zero = at::zeros({ n_matrices, num_rows, max_buckets }, gradIndex.options());
    grad_Hash_value = at::where(grad_Hash_value.isnan(), zero, grad_Hash_value);
    return grad_Hash_value;
}

auto grad_hash4() -> at::Tensor
{
    auto vector_ids_expanded = (vector_ids + 1).unsqueeze(2).expand({ -1, -1, max_buckets }).to(gradIndex.options());
    auto buckets_index_inv_expanded = (buckets_index_inv + 1).unsqueeze(1).expand({ -1, num_rows, -1 });

    // 计算 grad_Hash_value
    auto grad_Hash_value = vector_ids_expanded.div_(buckets_index_inv_expanded).sub_(1);

    // 计算 exp_component，避免重复计算
    auto grad_Hash_value_sq = grad_Hash_value.square();
    auto sigma_sq = sigma * sigma;
    grad_Hash_value_sq.div_(2 * sigma_sq).neg_().exp_();

    // 计算 grad_Hash_value
    grad_Hash_value.div_(sigma_sq).mul_(gradIndex);
    grad_Hash_value_sq.div_(buckets_count.unsqueeze(1).expand({ -1, num_rows, -1 }));
    grad_Hash_value.mul_(grad_Hash_value_sq).neg_();

    // 创建 zero 张量
    auto zero = torch::zeros({ n_matrices, num_rows, max_buckets }, gradIndex.options());

    // 使用 where 操作处理 NaN 值
    grad_Hash_value = torch::where(grad_Hash_value.isnan(), zero, grad_Hash_value);

    return grad_Hash_value;
}

at::Tensor grad_hash_cuda();
at::Tensor grad_hash_cuda2();

std::array funcs = { grad_hash0, grad_hash1, grad_hash2, grad_hash3, grad_hash4, grad_hash_cuda, grad_hash_cuda2 };
std::array funcs_name = { "grad_hash0", "grad_hash1", "grad_hash2", "grad_hash3", "grad_hash4", "grad_hash_cuda", "grad_hash_cuda2" };

auto init_parameters() -> void
{
    vector_ids = torch::randint(0, 10, { n_matrices, num_rows }, torch::dtype(torch::kInt32).device(torch::kCUDA));
    buckets_index_inv = torch::randint(0, 10, { n_matrices, max_buckets }, torch::dtype(torch::kInt32).device(torch::kCUDA));
    gradIndex = torch::randn({ n_matrices, num_rows, max_buckets }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    buckets_count = torch::randint(1, 10, { n_matrices, max_buckets }, torch::dtype(torch::kInt32).device(torch::kCUDA));
}

auto check_correctness(at::Tensor std_result, at::Tensor new_result) -> bool
{
    constexpr double eps = 1e-6;
    auto diff = std_result - new_result;
    auto diff_sum = diff.abs().sum().item<float>();
    return diff_sum < eps;
}

template <unsigned repeat_times = 5>
auto repeat(grad_hash_func grad_hash_new) -> double
{
    c10::cuda::CUDACachingAllocator::emptyCache();
    init_parameters();
    auto std_result = grad_hash0();
    auto new_result = grad_hash_new();
    if (!check_correctness(std_result, new_result)) {
        std::cerr << "Error: the results are not the same\n";
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();
#pragma unroll
    for (int i = 0; i < repeat_times; i++) {
        grad_hash_new();
        torch::cuda::synchronize();
        cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count() * 1000 / repeat_times;
}

auto benchmark() -> void
{
    for (int i = 0; i < funcs.size(); i++) {
        std::cout << "Benchmarking " << funcs_name[i] << "...\n";
        auto time = repeat<5>(funcs[i]);
        std::cout << "Time: " << time << " ms\n";
    }
}

int main()
{
    benchmark();
    return 0;
}
