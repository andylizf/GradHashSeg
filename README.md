# Benchmarking gradHash

```session
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` -G Ninja ..
cmake --build . --config Release
```
