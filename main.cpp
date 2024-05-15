#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <chrono>
#include <iostream>
#include <torch/torch.h>
#include <torch/types.h>

// 定义宏用于计时
#define TIMER_LAP(str, start)                                                                                                             \
    torch::cuda::synchronize();                                                                                                           \
    std::cout << str << ": " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count() << " seconds\n"; \
    start = std::chrono::high_resolution_clock::now();

int max_buckets = 170;
int num_rows = 1000;
int n_matrices = 160;
int param_H = 10;
double sigma = 0.000195313;
at::Tensor vector_ids, buckets_index_inv, buckets_count, gradIndex;

using grad_hash_func = at::Tensor (*)();

auto grad_hash0() -> at::Tensor
{
    at::Tensor grad_Hash_value = (vector_ids.unsqueeze(2).repeat({ 1, 1, max_buckets }) + 1).to(gradIndex.options()) / (buckets_index_inv.unsqueeze(1).repeat({ 1, num_rows, 1 }) + 1).to(gradIndex.options()) - 1;
    grad_Hash_value = -1 * grad_Hash_value / (sigma * sigma) * exp(-1 * grad_Hash_value * grad_Hash_value / (2 * sigma * sigma)) * gradIndex / buckets_count.unsqueeze(1).repeat({ 1, num_rows, 1 }).to(gradIndex.options());
    at::Tensor power = at::zeros({ n_matrices, max_buckets, param_H }, gradIndex.options());
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
    auto sigma_sq = sigma * sigma;
    auto exp_component = torch::exp(-grad_Hash_value_sq / (2 * sigma_sq));

    // 计算 grad_Hash_value
    auto buckets_count_expanded = buckets_count.unsqueeze(1).expand({ -1, num_rows, -1 });
    grad_Hash_value = -grad_Hash_value / sigma_sq * exp_component * gradIndex / buckets_count_expanded;

    // 创建 zero 张量
    auto zero = torch::zeros({ n_matrices, num_rows, max_buckets }, gradIndex.options());

    // 使用 where 操作处理 NaN 值
    grad_Hash_value = torch::where(grad_Hash_value.isnan(), zero, grad_Hash_value);

    return grad_Hash_value;
}

std::array funcs = { grad_hash0, grad_hash1 };
std::array funcs_name = { "grad_hash0", "grad_hash1" };

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
    std::array<at::Tensor, repeat_times> new_results;

    auto start = std::chrono::high_resolution_clock::now();
#pragma unroll
    for (int i = 0; i < repeat_times; i++) {
        new_results[i] = grad_hash_new();
    }
    auto end = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < repeat_times; i++) {
        if (!check_correctness(std_result, new_results[i])) {
            std::cerr << "Error: the results are not the same\n";
            std::cerr << "std_result: " << std_result << "\n";
            std::cerr << "new_result: " << new_results[i] << "\n";
            return -1;
        }
    }
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
