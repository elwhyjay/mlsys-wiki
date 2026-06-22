> 블로그 출처: https://leimao.github.io/blog/CUDA-Constant-Memory/ 이 글은 Lei Mao의 글이며, 저자의 전재 허가를 받았다.

# CUDA Constant Memory

## 소개

CUDA constant memory는 device의 특수 memory space다. Cache되며 read-only이다.

Constant memory를 사용할 때는 몇 가지 주의할 점이 있다. 이 글에서는 constant memory의 사용 방법과 주의 사항을 논의한다.

## Constant Memory

Device에는 총 64 KB의 constant memory가 있다. Constant memory space는 cache된다. 따라서 constant memory에서 읽을 때 cache miss인 경우에만 device memory에서 한 번 읽으면 되고, 그렇지 않으면 constant cache에서 한 번만 읽으면 된다. 한 warp 안의 thread가 서로 다른 address에 access하면 serialize되므로 비용은 warp 안의 모든 thread가 읽는 unique address 수에 선형적으로 비례한다. 따라서 같은 warp의 thread가 소수의 서로 다른 location에만 access할 때 constant cache가 가장 효과적이다. 한 warp의 모든 thread가 같은 location에 access하면 constant memory는 register access만큼 빠를 수 있다.

## Constant Memory 사용과 performance

아래 예제에서는 array에 addition을 수행한다. Constant input array 중 하나는 global memory에 저장하고, 다른 constant input array는 global memory 또는 constant memory에 저장한다. 서로 다른 access pattern에서 constant memory와 global memory에 access하는 performance을 비교한다.

```c++
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, unsigned int num_repeats = 100,
                          unsigned int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (unsigned int i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (unsigned int i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

// useall constant memory space
// constant memory size is 64KB，divide by sizeof(int) to get number of storable ints
constexpr unsigned int N{64U * 1024U / sizeof(int)};
// declare constant memory array，stored in GPU constant memory
__constant__ int const_values[N];

// magic number for pseudo-random access pattern
constexpr unsigned int magic_number{1357U};

// define enum type for access patterns
enum struct AccessPattern
{
    OneAccessPerBlock,    // one access per block
    OneAccessPerWarp,     // one access per warp
    OneAccessPerThread,   // one access per thread
    PseudoRandom         // pseudo-random access
};

// CPU constant addition function，for generating reference result
void add_constant_cpu(int* sums, int const* inputs, int const* values,
                      unsigned int num_sums, unsigned int num_values,
                      unsigned int block_size, AccessPattern access_pattern)
{
    // iterate over all sums to calculate
    for (unsigned int i{0U}; i < num_sums; ++i)
    {
        // calculateblock ID of current element
        unsigned int const block_id{i / block_size};
        // calculatethread ID of current element within block
        unsigned int const thread_id{i % block_size};
        // calculatewarp ID of current thread（each warp has 32 threads）
        unsigned int const warp_id{thread_id / 32U};
        unsigned int index{0U};

        // determine theconstant arrayindex
        switch (access_pattern)
        {
            case AccessPattern::OneAccessPerBlock:
                // each block accesses the same constant value
                index = block_id % num_values;
                break;
            case AccessPattern::OneAccessPerWarp:
                // each warp accesses the same constant value
                index = warp_id % num_values;
                break;
            case AccessPattern::OneAccessPerThread:
                // each thread accesses a different constant value
                index = thread_id % num_values;
                break;
            case AccessPattern::PseudoRandom:
                // use magic number to generate pseudo-random access pattern
                index = (thread_id * magic_number) % num_values;
                break;
        }

        // perform addition：input value + constant value
        sums[i] = inputs[i] + values[index];
    }
}

// CUDA kernel using global memory
__global__ void add_constant_global_memory(
    int* sums, int const* inputs, int const* values, unsigned int num_sums,
    unsigned int num_values,
    AccessPattern access_pattern = AccessPattern::OneAccessPerBlock)
{
    // calculatecurrent thread globalindex
    unsigned int const i{blockIdx.x * blockDim.x + threadIdx.x};
    // get block ID
    unsigned int const block_id{blockIdx.x};
    // get thread ID within block
    unsigned int const thread_id{threadIdx.x};
    // calculatewarp ID
    unsigned int const warp_id{threadIdx.x / warpSize};
    unsigned int index{0U};

    // determine theglobal memory index
    switch (access_pattern)
    {
        case AccessPattern::OneAccessPerBlock:
            // each block accesses same global memory location
            index = block_id % num_values;
            break;
        case AccessPattern::OneAccessPerWarp:
            // each warp accesses same global memory location
            index = warp_id % num_values;
            break;
        case AccessPattern::OneAccessPerThread:
            // each thread accesses different global memory location
            index = thread_id % num_values;
            break;
        case AccessPattern::PseudoRandom:
            // use magic number to generate pseudo-random access pattern
            index = (thread_id * magic_number) % num_values;
            break;
    }

    // bounds check，ensure no out-of-bounds
    if (i < num_sums)
    {
        // read constant value from global memory and perform addition
        sums[i] = inputs[i] + values[index];
    }
}

// launchkernel using global memory wrapper function
void launch_add_constant_global_memory(int* sums, int const* inputs,
                                       int const* values, unsigned int num_sums,
                                       unsigned int num_values,
                                       unsigned int block_size,
                                       AccessPattern access_pattern,
                                       cudaStream_t stream)
{
    // calculate grid size, ensure all elements can be processed
    add_constant_global_memory<<<(num_sums + block_size - 1) / block_size,
                                 block_size, 0, stream>>>(
        sums, inputs, values, num_sums, num_values, access_pattern);
    // check whether kernel launch succeeded
    CHECK_LAST_CUDA_ERROR();
}

// CUDA kernel using constant memory
__global__ void add_constant_constant_memory(int* sums, int const* inputs,
                                             unsigned int num_sums,
                                             AccessPattern access_pattern)
{
    // calculatecurrent thread globalindex
    unsigned int const i{blockIdx.x * blockDim.x + threadIdx.x};
    // get block ID
    unsigned int const block_id{blockIdx.x};
    // get thread ID within block
    unsigned int const thread_id{threadIdx.x};
    // calculatewarp ID
    unsigned int const warp_id{threadIdx.x / warpSize};
    unsigned int index{0U};

    // determine theconstant memory index
    switch (access_pattern)
    {
        case AccessPattern::OneAccessPerBlock:
            // each block accesses same constant memory location
            index = block_id % N;
            break;
        case AccessPattern::OneAccessPerWarp:
            // each warp accesses same constant memory location
            index = warp_id % N;
            break;
        case AccessPattern::OneAccessPerThread:
            // each thread accesses different constant memory location
            index = thread_id % N;
            break;
        case AccessPattern::PseudoRandom:
            // use magic number to generate pseudo-random access pattern
            index = (thread_id * magic_number) % N;
            break;
    }

    // bounds check，ensure no out-of-bounds
    if (i < num_sums)
    {
        // read constant value from constant memory and perform addition
        sums[i] = inputs[i] + const_values[index];
    }
}

// launchkernel using constant memory wrapper function
void launch_add_constant_constant_memory(int* sums, int const* inputs,
                                         unsigned int num_sums,
                                         unsigned int block_size,
                                         AccessPattern access_pattern,
                                         cudaStream_t stream)
{
    // calculate grid size, ensure all elements can be processed
    add_constant_constant_memory<<<(num_sums + block_size - 1) / block_size,
                                   block_size, 0, stream>>>(
        sums, inputs, num_sums, access_pattern);
    // check whether kernel launch succeeded
    CHECK_LAST_CUDA_ERROR();
}

// function for parsing command-line arguments
void parse_args(int argc, char** argv, AccessPattern& access_pattern,
                unsigned int& block_size, unsigned int& num_sums)
{
    // check whether number of arguments is enough
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <access pattern> <block size> <number of sums>"
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // parse access mode argument
    std::string const access_pattern_str{argv[1]};
    if (access_pattern_str == "one_access_per_block")
    {
        access_pattern = AccessPattern::OneAccessPerBlock;
    }
    else if (access_pattern_str == "one_access_per_warp")
    {
        access_pattern = AccessPattern::OneAccessPerWarp;
    }
    else if (access_pattern_str == "one_access_per_thread")
    {
        access_pattern = AccessPattern::OneAccessPerThread;
    }
    else if (access_pattern_str == "pseudo_random")
    {
        access_pattern = AccessPattern::PseudoRandom;
    }
    else
    {
        std::cerr << "Invalid access pattern: " << access_pattern_str
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // parse block size and number of sums arguments
    block_size = std::stoi(argv[2]);
    num_sums = std::stoi(argv[3]);
}

int main(int argc, char** argv)
{
    // define warmup count and repeat count for performance test
    constexpr unsigned int num_warmups{100U};
    constexpr unsigned int num_repeats{100U};

    // set default parameters
    AccessPattern access_pattern{AccessPattern::OneAccessPerBlock};
    unsigned int block_size{1024U};
    unsigned int num_sums{12800000U};
    // modify access mode, block size, and number of sums from command line
    parse_args(argc, argv, access_pattern, block_size, num_sums);

    // create CUDA stream
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // initialize constant value array in host memory
    int h_values[N];
    // initialize constant values in host memory
    for (unsigned int i{0U}; i < N; ++i)
    {
        h_values[i] = i;
    }
    // initialize constant values in global memory
    int* d_values;
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_values, N * sizeof(int), stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_values, h_values, N * sizeof(int),
                                     cudaMemcpyHostToDevice, stream));
    // initialize constant values in constant memory
    CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(const_values, h_values,
                                             N * sizeof(int), 0,
                                             cudaMemcpyHostToDevice, stream));

    // create input array and initialize to 0
    std::vector<int> inputs(num_sums, 0);
    int* h_inputs{inputs.data()};
    // allocate device input array for constant memory test
    int* d_inputs_for_constant;
    // allocate device input array for global memory test
    int* d_inputs_for_global;
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_inputs_for_constant,
                                     num_sums * sizeof(int), stream));
    CHECK_CUDA_ERROR(
        cudaMallocAsync(&d_inputs_for_global, num_sums * sizeof(int), stream));
    // copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_inputs_for_constant, h_inputs,
                                     num_sums * sizeof(int),
                                     cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_inputs_for_global, h_inputs,
                                     num_sums * sizeof(int),
                                     cudaMemcpyHostToDevice, stream));

    // create result arrays
    std::vector<int> reference_sums(num_sums, 0);      // CPU reference result
    std::vector<int> sums_from_constant(num_sums, 1);  // constant memory result
    std::vector<int> sums_from_global(num_sums, 2);    // global memory result

    // get host array pointer
    int* h_reference_sums{reference_sums.data()};
    int* h_sums_from_constant{sums_from_constant.data()};
    int* h_sums_from_global{sums_from_global.data()};

    // allocate device result array
    int* d_sums_from_constant;
    int* d_sums_from_global;
    CHECK_CUDA_ERROR(
        cudaMallocAsync(&d_sums_from_constant, num_sums * sizeof(int), stream));
    CHECK_CUDA_ERROR(
        cudaMallocAsync(&d_sums_from_global, num_sums * sizeof(int), stream));

    // synchronize stream，ensureall async operations finish
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // calculate reference result on CPU
    add_constant_cpu(h_reference_sums, h_inputs, h_values, num_sums, N,
                     block_size, access_pattern);
    // calculate result on GPU using global memory
    launch_add_constant_global_memory(d_sums_from_global, d_inputs_for_global,
                                      d_values, num_sums, N, block_size,
                                      access_pattern, stream);
    // calculate result on GPU using constant memory
    launch_add_constant_constant_memory(d_sums_from_constant,
                                        d_inputs_for_constant, num_sums,
                                        block_size, access_pattern, stream);

    // copy result from device to host
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_sums_from_constant, d_sums_from_constant,
                                     num_sums * sizeof(int),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_sums_from_global, d_sums_from_global,
                                     num_sums * sizeof(int),
                                     cudaMemcpyDeviceToHost, stream));

    // synchronize stream，ensureall data transfers finish
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // verify result correctness
    for (unsigned int i{0U}; i < num_sums; ++i)
    {
        // check whether constant memory result matches reference result
        if (h_reference_sums[i] != h_sums_from_constant[i])
        {
            std::cerr << "Error at index " << i << " for constant memory."
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }
        // check whether global memory result matches reference result
        if (h_reference_sums[i] != h_sums_from_global[i])
        {
            std::cerr << "Error at index " << i << " for global memory."
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    // measureperformance
    // create bound function for constant memory kernel
    std::function<void(cudaStream_t)> bound_function_constant_memory{
        std::bind(launch_add_constant_constant_memory, d_sums_from_constant,
                  d_inputs_for_constant, num_sums, block_size, access_pattern,
                  std::placeholders::_1)};
    // create bound function for global memory kernel
    std::function<void(cudaStream_t)> bound_function_global_memory{
        std::bind(launch_add_constant_global_memory, d_sums_from_global,
                  d_inputs_for_global, d_values, num_sums, N, block_size,
                  access_pattern, std::placeholders::_1)};
    // measure constant memory performance
    float const latency_constant_memory{measure_performance(
        bound_function_constant_memory, stream, num_repeats, num_warmups)};
    // measure global memory performance
    float const latency_global_memory{measure_performance(
        bound_function_global_memory, stream, num_repeats, num_warmups)};
    // outputperformancetestresult
    std::cout << "Latency for Add using constant memory: "
              << latency_constant_memory << " ms" << std::endl;
    std::cout << "Latency for Add using global memory: "
              << latency_global_memory << " ms" << std::endl;

    // clean up resources
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    CHECK_CUDA_ERROR(cudaFree(d_values));
    CHECK_CUDA_ERROR(cudaFree(d_inputs_for_constant));
    CHECK_CUDA_ERROR(cudaFree(d_inputs_for_global));
    CHECK_CUDA_ERROR(cudaFree(d_sums_from_constant));
    CHECK_CUDA_ERROR(cudaFree(d_sums_from_global));

    return 0;
}
```

이 프로그램은 NVIDIA RTX 3090 GPU에서 컴파일하고 실행했다.

```shell
$ nvcc add_constant.cu -o add_constant
```

block마다 1024 thread를 사용해 12,800,000번의 addition을 수행하는 경우다.

```shell
$ ./add_constant one_access_per_block 1024 12800000
Latency for Add using constant memory: 0.151798 ms
Latency for Add using global memory: 0.171404 ms
$ ./add_constant one_access_per_warp 1024 12800000
Latency for Add using constant memory: 0.164012 ms
Latency for Add using global memory: 0.189501 ms
$ ./add_constant one_access_per_thread 1024 12800000
Latency for Add using constant memory: 0.281967 ms
Latency for Add using global memory: 0.164649 ms
$ ./add_constant pseudo_random 1024 12800000
Latency for Add using constant memory: 1.2925 ms
Latency for Add using global memory: 0.159621 ms
```

block마다 1024 thread를 사용해 128,000번의 addition을 수행하는 경우다.

```shell
$ ./add_constant one_access_per_block 1024 128000
Latency for Add using constant memory: 0.00289792 ms
Latency for Add using global memory: 0.00323584 ms
$ ./add_constant one_access_per_warp 1024 128000
Latency for Add using constant memory: 0.00315392 ms
Latency for Add using global memory: 0.00359392 ms
$ ./add_constant one_access_per_thread 1024 128000
Latency for Add using constant memory: 0.00596992 ms
Latency for Add using global memory: 0.00383264 ms
$ ./add_constant pseudo_random 1024 128000
Latency for Add using constant memory: 0.0215347 ms
Latency for Add using global memory: 0.00482304 ms
```

두 경우 모두 block마다 한 번 access하거나 warp마다 한 번 access하는 경우 constant memory access가 global memory access보다 약 10% 빠르다는 것을 볼 수 있다. Thread마다 한 번 access하는 경우 constant memory access는 global memory access보다 약 70% 느리다. Pseudo-random access인 경우 constant memory access는 global memory access보다 약 800% 느리다.

## 결론

Constant memory를 사용하려면 access pattern을 이해하는 것이 중요하다. Access pattern이 block마다 한 번 또는 warp마다 한 번(보통 broadcast에 사용됨)이라면 constant memory는 좋은 선택이다. Access pattern이 thread마다 한 번이거나 pseudo-random access라면 constant memory는 매우 나쁜 선택이다.

## 참고 자료

- Device Memory Spaces(https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#device-memory-spaces)
- Constant Memory(https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#constant-memory)
- Constant Specifier(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#constant)
