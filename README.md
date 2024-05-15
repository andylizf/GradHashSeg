# Benchmarking `gradHash`

## Introduction

To calculate the following formula more efficiently:
$$
\text{grad\_Hash\_value} = \left( \frac{(\text{vector\_ids} + 1)}{(\text{buckets\_index\_inv} + 1)} \right) - 1
\text{grad\_Hash\_value} = - \frac{\text{grad\_Hash\_value}}{\sigma^2} \cdot \exp\left(- \frac{\text{grad\_Hash\_value}^2}{2\sigma^2}\right) \cdot \frac{\text{gradIndex}}{\text{buckets\_count}}
$$

## Build

Open VSCode, install the `Dev Container` extension, and press `Ctrl+Shift+P` to open the command palette, then type `Remote-Containers: Reopen in Container`.

After the container is built, open a terminal and run the following commands:

```session
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` -G Ninja ..
cmake --build . --config Release
```

## Benchmark

On Lab3090, we derived the following results:

```
Benchmarking grad_hash0...
Time: 5.91146 ms
Benchmarking grad_hash1...
Time: 3.30413 ms
Benchmarking grad_hash2...
Time: 3.46245 ms
Benchmarking grad_hash3...
Time: 4.13871 ms
Benchmarking grad_hash4...
Time: 3.95917 ms
Benchmarking grad_hash_cuda...
Time: 0.454241 ms
Benchmarking grad_hash_cuda2...
Time: 0.465379 ms
Benchmarking grad_hash_cuda3...
Time: 0.445389 ms
```

We can see that the CUDA implementation is much faster than the PyTorch implementation. That's because our hand-crafted CUDA kernel fused operations into a single kernel, which beats the PyTorch implementation that involves multiple kernels, especially when operations are all element-wise.

In the comparison between `grad_hash_cuda` and `grad_hash_cuda2`, we can see that the use of `at::PackedTensorAccessor32` introduces a slight overhead, though it makes the code more readable.
