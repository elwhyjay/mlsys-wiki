> 블로그 출처: https://leimao.github.io/blog/CUDA-L2-Persistent-Cache/ 이 글은 Lei Mao의 글이며, 저자의 전재 허가를 받았다.

# CUDA L2 Persistent Cache

## 소개

CUDA 11.0부터 compute capability 8.0 이상의 device는 L2 cache에 있는 data의 persistence에 영향을 줄 수 있는 기능을 가진다. L2 cache는 on-chip에 있으므로 global memory에 대해 더 높은 bandwidth와 더 낮은 latency access를 제공할 수 있다.

이 블로그 글에서는，L2 persistent cache를 사용해 data transfer를 가속하는 방법을 보여주는 CUDA 예제를 만들었다.

## CUDA L2 Persistent Cache

이 예제에서는 특정 값을 담은 작은 constant buffer가 있고, 이 buffer를 사용해 큰 streaming buffer를 reset한다. 예를 들어 constant buffer의 size가 4이고 값이 `[5, 2, 1, 4]`이며 reset할 큰 streaming buffer의 size가 100이면, reset 후 큰 streaming buffer는 `[5, 2, 1, 4, 5, 2, 1, 4, ...]`처럼 constant buffer 값을 반복한다.

Streaming buffer는 constant buffer보다 훨씬 size 때문에 constant buffer의 각 원소는 streaming buffer보다 더 자주 access된다. Global memory에서 buffer에 access하는 것은 매우 비싸다. 자주 access되는 constant buffer를 L2 cache에 cache할 수 있다면, 이 frequent constant buffer access를 가속할 수 있다.

### CUDA Data Reset

Data reset CUDA kernel에 대해 persistent L2 cache 없이 kernel을 실행하는 baseline 버전, 3MB persistent L2 cache를 사용하지만 constant buffer size가 3MB를 넘으면 data thrashing이 발생하는 변형 버전, 그리고 3MB persistent L2 cache를 사용하면서 data thrashing을 제거한 optimized 버전을 만들었다.

```c++
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
// CUDA error check macro，for checking return values of CUDA API calls
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* const func, char const* const file,
           int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// macro checking last CUDA error
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const* const file, int const line)
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

// performancemeasurefunctiontemplate，for measuring CUDA kernel execution time
template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, int num_repeats = 100,
                          int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    // create CUDA events for timing
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // warmup stage，let GPU enter steady state
    for (int i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    // wait for all warmup operations to finish
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // start timing
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    // run multiple tests to obtain average performance
    for (int i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    // finish timing
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    
    // calculatetotalexecutetime
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    
    // clean up event resources
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    // return average latency
    float const latency{time / num_repeats};

    return latency;
}

// CUDA kernel：uselookup tableresetdata stream
// data_streaming: large data stream to reset
// lut_persistent: persistent lookup table（small data，frequently accessed）
// data_streaming_size: data streamsize
// lut_persistent_size: lookup tablesize
__global__ void reset_data(int* data_streaming, int const* lut_persistent,
                           size_t data_streaming_size,
                           size_t lut_persistent_size)
{
    // calculatecurrent thread globalindex
    size_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    // calculategrid stride（total number of all threads）
    size_t const stride{blockDim.x * gridDim.x};
    
    // usegrid strideprocess data in a loop，ensureall data is processed
    for (size_t i{idx}; i < data_streaming_size; i += stride)
    {
        // fill data stream cyclically using lookup table values
        data_streaming[i] = lut_persistent[i % lut_persistent_size];
    }
}

/**
 * @brief uselut_persistentresetdata_streaming，make data_streaming repeat the contents of lut_persistent
 *
 * @param data_streaming data to reset
 * @param lut_persistent values used to reset data_streaming
 * @param data_streaming_size data_streaming size
 * @param lut_persistent_size lut_persistent size
 * @param stream CUDA stream
 */
void launch_reset_data(int* data_streaming, int const* lut_persistent,
                       size_t data_streaming_size, size_t lut_persistent_size,
                       cudaStream_t stream)
{
    // set thread block size
    dim3 const threads_per_block{1024};
    // set grid size
    dim3 const blocks_per_grid{32};
    
    // launchCUDA kernel
    reset_data<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        data_streaming, lut_persistent, data_streaming_size,
        lut_persistent_size);
    CHECK_LAST_CUDA_ERROR();
}

// verify whether data is reset correctly
bool verify_data(int* data, int n, size_t size)
{
    for (size_t i{0}; i < size; ++i)
    {
        if (data[i] != i % n)
        {
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[])
{
    // default persistent data size is 3MB
    size_t num_megabytes_persistent_data{3};
    if (argc == 2)
    {
        num_megabytes_persistent_data = std::atoi(argv[1]);
    }

    // performance test parameters
    constexpr int const num_repeats{100};    // repeat count
    constexpr int const num_warmups{10};     // warmup count

    // get GPU device properties
    cudaDeviceProp device_prop{};
    int current_device{0};
    CHECK_CUDA_ERROR(cudaGetDevice(&current_device));
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&device_prop, current_device));
    
    // print GPU information
    std::cout << "GPU: " << device_prop.name << std::endl;
    std::cout << "L2 Cache Size: " << device_prop.l2CacheSize / 1024 / 1024
              << " MB" << std::endl;
    std::cout << "Max Persistent L2 Cache Size: "
              << device_prop.persistingL2CacheMaxSize / 1024 / 1024 << " MB"
              << std::endl;

    // set data size
    size_t const num_megabytes_streaming_data{1024};  // streaming data 1GB
    if (num_megabytes_persistent_data > num_megabytes_streaming_data)
    {
        std::runtime_error(
            "Try setting persistent data size smaller than 1024 MB.");
    }
    
    // calculate number of array elements
    size_t const size_persistent(num_megabytes_persistent_data * 1024 * 1024 /
                                 sizeof(int));
    size_t const size_streaming(num_megabytes_streaming_data * 1024 * 1024 /
                                sizeof(int));
    
    std::cout << "Persistent Data Size: " << num_megabytes_persistent_data
              << " MB" << std::endl;
    std::cout << "Steaming Data Size: " << num_megabytes_streaming_data << " MB"
              << std::endl;
    cudaStream_t stream;

    // createhost-sidedata
    std::vector<int> lut_persistent_vec(size_persistent, 0);
    // initialize lookup table，values are 0, 1, 2, ...
    for (size_t i{0}; i < lut_persistent_vec.size(); ++i)
    {
        lut_persistent_vec[i] = i;
    }
    std::vector<int> data_streaming_vec(size_streaming, 0);

    // device-sidepointer
    int* d_lut_persistent;
    int* d_data_streaming;
    // host-sidepointer
    int* h_lut_persistent = lut_persistent_vec.data();
    int* h_data_streaming = data_streaming_vec.data();

    // allocate GPU memory
    CHECK_CUDA_ERROR(
        cudaMalloc(&d_lut_persistent, size_persistent * sizeof(int)));
    CHECK_CUDA_ERROR(
        cudaMalloc(&d_data_streaming, size_streaming * sizeof(int)));
    
    // create CUDA stream
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    
    // copy lookup table data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_lut_persistent, h_lut_persistent,
                                size_persistent * sizeof(int),
                                cudaMemcpyHostToDevice));

    // testkernelcorrectness
    launch_reset_data(d_data_streaming, d_lut_persistent, size_streaming,
                      size_persistent, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaMemcpy(h_data_streaming, d_data_streaming,
                                size_streaming * sizeof(int),
                                cudaMemcpyDeviceToHost));
    assert(verify_data(h_data_streaming, size_persistent, size_streaming));

    // benchmark：without persistent L2 cache
    std::function<void(cudaStream_t)> const function{
        std::bind(launch_reset_data, d_data_streaming, d_lut_persistent,
                  size_streaming, size_persistent, std::placeholders::_1)};
    float const latency{
        measure_performance(function, stream, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(3)
              << "Latency Without Using Persistent L2 Cache: " << latency
              << " ms" << std::endl;

    // start using persistent cache
    cudaStream_t stream_persistent_cache;
    size_t const num_megabytes_persistent_cache{3};  // persistent L2 cache size3MB
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream_persistent_cache));

    // set persistent L2 cache size limit
    CHECK_CUDA_ERROR(
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                           num_megabytes_persistent_cache * 1024 * 1024));

    // configure access policy window（version that may cause thrashing）
    cudaStreamAttrValue stream_attribute_thrashing;
    stream_attribute_thrashing.accessPolicyWindow.base_ptr =
        reinterpret_cast<void*>(d_lut_persistent);  // base address of persistent data
    stream_attribute_thrashing.accessPolicyWindow.num_bytes =
        num_megabytes_persistent_data * 1024 * 1024;  // byte count of persistent data
    stream_attribute_thrashing.accessPolicyWindow.hitRatio = 1.0;  // hit ratio100%
    stream_attribute_thrashing.accessPolicyWindow.hitProp =
        cudaAccessPropertyPersisting;  // use persisting property on hit
    stream_attribute_thrashing.accessPolicyWindow.missProp =
        cudaAccessPropertyStreaming;   // use streaming property on miss

    // set access policy for stream
    CHECK_CUDA_ERROR(cudaStreamSetAttribute(
        stream_persistent_cache, cudaStreamAttributeAccessPolicyWindow,
        &stream_attribute_thrashing));

    // test potentially-thrashing persistent L2 cache performance
    float const latency_persistent_cache_thrashing{measure_performance(
        function, stream_persistent_cache, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(3) << "Latency With Using "
              << num_megabytes_persistent_cache
              << " MB Persistent L2 Cache (Potentially Thrashing): "
              << latency_persistent_cache_thrashing << " ms" << std::endl;

    // configure access policy window（version that avoids thrashing）
    cudaStreamAttrValue stream_attribute_non_thrashing{
        stream_attribute_thrashing};
    // adjust hit ratio to avoid thrashing：persistent cachesize / persistent data size
    stream_attribute_non_thrashing.accessPolicyWindow.hitRatio =
        std::min(static_cast<double>(num_megabytes_persistent_cache) /
                     num_megabytes_persistent_data,
                 1.0);
    
    // update stream access policy
    CHECK_CUDA_ERROR(cudaStreamSetAttribute(
        stream_persistent_cache, cudaStreamAttributeAccessPolicyWindow,
        &stream_attribute_non_thrashing));

    // test no-thrashing persistent L2 cache performance
    float const latency_persistent_cache_non_thrashing{measure_performance(
        function, stream_persistent_cache, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(3) << "Latency With Using "
              << num_megabytes_persistent_cache
              << " MB Persistent L2 Cache (Non-Thrashing): "
              << latency_persistent_cache_non_thrashing << " ms" << std::endl;

    // clean up resources
    CHECK_CUDA_ERROR(cudaFree(d_lut_persistent));
    CHECK_CUDA_ERROR(cudaFree(d_data_streaming));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_persistent_cache));
}
```

Data thrashing을 피하려면 `accessPolicyWindow.hitRatio`와 `accessPolicyWindow.num_bytes`의 곱이 `cudaLimitPersistingL2CacheSize`보다 작거나 같아야 한다. `accessPolicyWindow.hitRatio` parameter는 `accessPolicyWindow.hitProp` property(보통 `cudaAccessPropertyPersisting`)를 받는 access 비율을 지정하는 데 사용할 수 있다. `accessPolicyWindow.num_bytes` parameter는 access policy window가 덮는 byte 수를 지정하는 데 사용할 수 있으며, 보통 persistent data의 size다.

실제로는 `accessPolicyWindow.hitRatio`를 persistent L2 cache size와 persistent data size의 비율로 설정할 수 있다. 예를 들어 persistent L2 cache size가 3MB이고 persistent data size가 4MB이면 `accessPolicyWindow.hitRatio`를 3/4 = 0.75로 설정할 수 있다.

### CUDA Data Reset 실행

이 예제는 NVIDIA Ampere GPU에서 빌드하고 실행할 수 있다. 내 경우에는 NVIDIA RTX 3090 GPU를 사용했다.

```shell
$ nvcc l2-persistent.cu -o l2-persistent -std=c++14 --gpu-architecture=compute_80
$ ./l2-persistent
GPU: NVIDIA GeForce RTX 3090
L2 Cache Size: 6 MB
Max Persistent L2 Cache Size: 4 MB
Persistent Data Size: 3 MB
Steaming Data Size: 1024 MB
Latency Without Using Persistent L2 Cache: 3.071 ms
Latency With Using 3 MB Persistent L2 Cache (Potentially Thrashing): 2.436 ms
Latency With Using 3 MB Persistent L2 Cache (Non-Thrashing): 2.443 ms
```

persistent data size가 3MB이고 persistent L2 cache가 3MB일 때 application performance이 약 20% 향상됨을 볼 수 있다.

### Benchmark

persistent data size를 바꾸며 작은 benchmark도 실행할 수 있다.

```shell
$ ./l2-persistent 1
GPU: NVIDIA GeForce RTX 3090
L2 Cache Size: 6 MB
Max Persistent L2 Cache Size: 4 MB
Persistent Data Size: 1 MB
Steaming Data Size: 1024 MB
Latency Without Using Persistent L2 Cache: 1.754 ms
Latency With Using 3 MB Persistent L2 Cache (Potentially Thrashing): 1.685 ms
Latency With Using 3 MB Persistent L2 Cache (Non-Thrashing): 1.674 ms
$ ./l2-persistent 2
GPU: NVIDIA GeForce RTX 3090
L2 Cache Size: 6 MB
Max Persistent L2 Cache Size: 4 MB
Persistent Data Size: 2 MB
Steaming Data Size: 1024 MB
Latency Without Using Persistent L2 Cache: 2.158 ms
Latency With Using 3 MB Persistent L2 Cache (Potentially Thrashing): 1.997 ms
Latency With Using 3 MB Persistent L2 Cache (Non-Thrashing): 2.002 ms
$ ./l2-persistent 3
GPU: NVIDIA GeForce RTX 3090
L2 Cache Size: 6 MB
Max Persistent L2 Cache Size: 4 MB
Persistent Data Size: 3 MB
Steaming Data Size: 1024 MB
Latency Without Using Persistent L2 Cache: 3.095 ms
Latency With Using 3 MB Persistent L2 Cache (Potentially Thrashing): 2.510 ms
Latency With Using 3 MB Persistent L2 Cache (Non-Thrashing): 2.533 ms
$ ./l2-persistent 4
GPU: NVIDIA GeForce RTX 3090
L2 Cache Size: 6 MB
Max Persistent L2 Cache Size: 4 MB
Persistent Data Size: 4 MB
Steaming Data Size: 1024 MB
Latency Without Using Persistent L2 Cache: 3.906 ms
Latency With Using 3 MB Persistent L2 Cache (Potentially Thrashing): 3.632 ms
Latency With Using 3 MB Persistent L2 Cache (Non-Thrashing): 3.706 ms
$ ./l2-persistent 5
GPU: NVIDIA GeForce RTX 3090
L2 Cache Size: 6 MB
Max Persistent L2 Cache Size: 4 MB
Persistent Data Size: 5 MB
Steaming Data Size: 1024 MB
Latency Without Using Persistent L2 Cache: 4.120 ms
Latency With Using 3 MB Persistent L2 Cache (Potentially Thrashing): 4.554 ms
Latency With Using 3 MB Persistent L2 Cache (Non-Thrashing): 3.920 ms
$ ./l2-persistent 6
GPU: NVIDIA GeForce RTX 3090
L2 Cache Size: 6 MB
Max Persistent L2 Cache Size: 4 MB
Persistent Data Size: 6 MB
Steaming Data Size: 1024 MB
Latency Without Using Persistent L2 Cache: 4.194 ms
Latency With Using 3 MB Persistent L2 Cache (Potentially Thrashing): 4.583 ms
Latency With Using 3 MB Persistent L2 Cache (Non-Thrashing): 4.255 ms

```

persistent data size가 persistent L2 cache보다 클 때도 thrashing 없는 persistent L2 cache를 사용하는 latency는 보통 baseline performance보다 나쁘지 않음을 볼 수 있다.

## 자주 묻는 질문

### Persistent Cache VS Shared Memory?

Persistent cache는 shared memory와 다르다. Persistent cache는 GPU의 모든 thread에 보이지만 shared memory는 같은 block 안의 thread에만 보인다.

작은 size의 자주 access되는 data에는 shared memory를 사용해 data access를 가속할 수도 있다. 그러나 shared memory는 thread block마다 48-96KB(GPU에 따라 다름)로 제한되는 반면, persistent cache는 GPU마다 몇 MB로 제한된다.

## 참고 자료

- L2 Cache Access Window(https://docs.nvidia.com/cuda/archive/11.7.0/cuda-c-best-practices-guide/index.html#L2-cache-window)
- Function Binding and Performance Measurement(https://leimao.github.io/blog/Function-Binding-Performance-Measurement/)
