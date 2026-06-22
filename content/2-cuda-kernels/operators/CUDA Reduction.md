> 블로그 출처: https://leimao.github.io/blog/CUDA-Reduction/ 이 글은 Lei Mao의 글이며, 저자의 전재 허가를 받았다. 이후 Lei Mao의 CUDA 관련 Blog를 몇 편 더 전재할 예정이고, 이는 하나의 완결된 칼럼이다. Blog는 비교적 이른 시기의 CUDA 아키텍처부터 현재 최신 CUDA 아키텍처까지 다루며, 실용적인 엔지니어링 기법, 저수준 명령어 분석, Cutlass 분석 등 여러 주제를 포함한다.

# CUDA Reduction 연산

## 소개

Reduction operation은 parallel computing에서 흔한 operation이다. 보통 reduction operation은 일련의 element의 sum, maximum, minimum 또는 product를 calculate하는 데 사용된다.

이 블로그 글에서는，parallel reduction algorithm과 CUDA에서의 implementation을 논의한다.

## Batch Reduction Sum

이 예제에서는 CUDA에서 두 batch reduction sum kernel을 implementation한다. Batch reduction sum kernel은 array batch의 각 array element 합을 calculate한다.

Reduction algorithm의 아이디어는 단순하다. Batch 안의 각 array에 대해 고정된 수의 thread로 구성된 thread block 하나를 할당하여 array element의 sum을 calculate한다. 각 thread는 global memory에서 array의 여러 element에 access하고 partial sum을 register file에 저장한다. 모든 thread가 partial sum을 calculate한 뒤, partial sum을 final sum으로 더 reduce하는 방법은 두 가지가 있다. 한 방법은 shared memory에 partial sum을 저장하고 shared memory 안에서 partial sum을 reduce하는 것이다. 다른 방법은 warp-level primitive(developer.nvidia.com/blog/using-cuda-warp-level-primitives/)를 사용해 warp 안의 register file에서 partial sum을 reduce한 다음, shared memory에서 더 작은 규모의 reduction을 수행하는 것이다.

```c++
#include <cassert>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// CUDA error check macro definition
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* func, char const* file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// check last CUDA error
#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(char const* file, int line)
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

// performancemeasuretemplatefunction
template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, size_t num_repeats = 10,
                          size_t num_warmups = 10)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // warmup stage
    for (size_t i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // formal measurement stage
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (size_t i{0}; i < num_repeats; ++i)
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

// centered string alignment function
std::string std_string_centered(std::string const& s, size_t width,
                                char pad = ' ')
{
    size_t const l{s.length()};
    // throw an exception if width is too small
    if (width < l)
    {
        throw std::runtime_error("Width is too small.");
    }
    size_t const left_pad{(width - l) / 2};
    size_t const right_pad{width - l - left_pad};
    std::string const s_centered{std::string(left_pad, pad) + s +
                                 std::string(right_pad, pad)};
    return s_centered;
}

// shared memoryreduction sumversion1 - use inter-thread synchronization
template <size_t NUM_THREADS>
__device__ float shared_data_reduce_sum_v1(float shared_data[NUM_THREADS])
{
    static_assert(NUM_THREADS % 32 == 0,
                  "NUM_THREADS must be a multiple of 32");
    size_t const thread_idx{threadIdx.x};
    
    // usetreereductionpattern，halve stride each time
#pragma unroll
    for (size_t stride{NUM_THREADS / 2}; stride > 0; stride /= 2)
    {
        if (thread_idx < stride)
        {
            // each thread adds its value with the thread value at stride distance
            shared_data[thread_idx] += shared_data[thread_idx + stride];
        }
        // synchronize all threads，ensureprevious round calculation finished
        __syncthreads();
    }
    return shared_data[0];
}

// shared memoryreduction sumversion2 - for final reduction after warp-level reduction
template <size_t NUM_WARPS>
__device__ float shared_data_reduce_sum_v2(float shared_data[NUM_WARPS])
{
    float sum{0.0f};
    
    // because all threads in a warp access the same shared memory location，broadcast occurs，no bank conflict
#pragma unroll
    for (size_t i{0}; i < NUM_WARPS; ++i)
    {
        // there is no shared memory bank conflict here
        // because multiple threads in the warp access the same shared memory location，the result is broadcast
        sum += shared_data[i];
    }
    return sum;
}

// within warpreduction sum - useshuffle instruction
__device__ float warp_reduce_sum(float val)
{
    constexpr unsigned int FULL_MASK{0xffffffff};
    
    // usebutterfly patternperformwithin warpreduction
#pragma unroll
    for (size_t offset{16}; offset > 0; offset /= 2)
    {
        // __shfl_down_sync passes values from higher-index threads to the current thread
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    // only the first thread in the warp returns the correct result
    return val;
}

// block-levelreduction sumversion1 - useshared memory
template <size_t NUM_THREADS>
__device__ float block_reduce_sum_v1(float const* __restrict__ input_data,
                                     float shared_data[NUM_THREADS],
                                     size_t num_elements)
{
    static_assert(NUM_THREADS % 32 == 0,
                  "NUM_THREADS must be a multiple of 32");
    
    // calculatenumber of elements each thread needs to process
    size_t const num_elements_per_thread{(num_elements + NUM_THREADS - 1) /
                                         NUM_THREADS};
    size_t const thread_idx{threadIdx.x};
    float sum{0.0f};
    
    // each thread processes multiple elements
    for (size_t i{0}; i < num_elements_per_thread; ++i)
    {
        size_t const offset{thread_idx + i * NUM_THREADS};
        if (offset < num_elements)
        {
            sum += input_data[offset];
        }
    }
    
    //  partial sumstore to shared memory
    shared_data[thread_idx] = sum;
    __syncthreads();
    
    // perform final reduction in shared memory
    float const block_sum{shared_data_reduce_sum_v1<NUM_THREADS>(shared_data)};
    return block_sum;
}

// block-levelreduction sumversion2 - combine warp-level primitives and shared memory
template <size_t NUM_THREADS, size_t NUM_WARPS = NUM_THREADS / 32>
__device__ float block_reduce_sum_v2(float const* __restrict__ input_data,
                                     float shared_data[NUM_WARPS],
                                     size_t num_elements)
{
    // calculatenumber of elements each thread needs to process
    size_t const num_elements_per_thread{(num_elements + NUM_THREADS - 1) /
                                         NUM_THREADS};
    size_t const thread_idx{threadIdx.x};
    float sum{0.0f};
    
    // each thread processes multiple elements
    for (size_t i{0}; i < num_elements_per_thread; ++i)
    {
        size_t const offset{thread_idx + i * NUM_THREADS};
        if (offset < num_elements)
        {
            sum += input_data[offset];
        }
    }
    
    // first perform reduction within warp
    sum = warp_reduce_sum(sum);
    
    // first thread of each warp stores warp reduction result to shared memory
    if (threadIdx.x % 32 == 0)
    {
        shared_data[threadIdx.x / 32] = sum;
    }
    __syncthreads();
    
    // perform final reduction across warp reduction results
    float const block_sum{shared_data_reduce_sum_v2<NUM_WARPS>(shared_data)};
    return block_sum;
}

// batchreduction sumkernel version1
template <size_t NUM_THREADS>
__global__ void batched_reduce_sum_v1(float* __restrict__ output_data,
                                      float const* __restrict__ input_data,
                                      size_t num_elements_per_batch)
{
    static_assert(NUM_THREADS % 32 == 0,
                  "NUM_THREADS must be a multiple of 32");
    size_t const block_idx{blockIdx.x};
    size_t const thread_idx{threadIdx.x};
    
    // shared memory used by each block
    __shared__ float shared_data[NUM_THREADS];
    
    // calculate reduction sum of data corresponding to current block
    float const block_sum{block_reduce_sum_v1<NUM_THREADS>(
        input_data + block_idx * num_elements_per_batch, shared_data,
        num_elements_per_batch)};
    
    // only the first thread writes result
    if (thread_idx == 0)
    {
        output_data[block_idx] = block_sum;
    }
}

// batchreduction sumkernel version2
template <size_t NUM_THREADS>
__global__ void batched_reduce_sum_v2(float* __restrict__ output_data,
                                      float const* __restrict__ input_data,
                                      size_t num_elements_per_batch)
{
    static_assert(NUM_THREADS % 32 == 0,
                  "NUM_THREADS must be a multiple of 32");
    constexpr size_t NUM_WARPS{NUM_THREADS / 32};
    size_t const block_idx{blockIdx.x};
    size_t const thread_idx{threadIdx.x};
    
    // shared memory used by each block，only need to store each warp result
    __shared__ float shared_data[NUM_WARPS];
    
    // calculate reduction sum of data corresponding to current block
    float const block_sum{block_reduce_sum_v2<NUM_THREADS, NUM_WARPS>(
        input_data + block_idx * num_elements_per_batch, shared_data,
        num_elements_per_batch)};
    
    // only the first thread writes result
    if (thread_idx == 0)
    {
        output_data[block_idx] = block_sum;
    }
}

// launchbatchreduction sumkernel version1
template <size_t NUM_THREADS>
void launch_batched_reduce_sum_v1(float* output_data, float const* input_data,
                                  size_t batch_size,
                                  size_t num_elements_per_batch,
                                  cudaStream_t stream)
{
    size_t const num_blocks{batch_size};
    batched_reduce_sum_v1<NUM_THREADS><<<num_blocks, NUM_THREADS, 0, stream>>>(
        output_data, input_data, num_elements_per_batch);
    CHECK_LAST_CUDA_ERROR();
}

// launchbatchreduction sumkernel version2
template <size_t NUM_THREADS>
void launch_batched_reduce_sum_v2(float* output_data, float const* input_data,
                                  size_t batch_size,
                                  size_t num_elements_per_batch,
                                  cudaStream_t stream)
{
    size_t const num_blocks{batch_size};
    batched_reduce_sum_v2<NUM_THREADS><<<num_blocks, NUM_THREADS, 0, stream>>>(
        output_data, input_data, num_elements_per_batch);
    CHECK_LAST_CUDA_ERROR();
}

// performanceanalysisfunction
float profile_batched_reduce_sum(
    std::function<void(float*, float const*, size_t, size_t, cudaStream_t)>
        batched_reduce_sum_launch_function,
    size_t batch_size, size_t num_elements_per_batch)
{
    size_t const num_elements{batch_size * num_elements_per_batch};

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // initialize test data
    constexpr float element_value{1.0f};
    std::vector<float> input_data(num_elements, element_value);
    std::vector<float> output_data(batch_size, 0.0f);

    // allocate GPU memory
    float* d_input_data;
    float* d_output_data;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input_data, num_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_data, batch_size * sizeof(float)));

    // copy data to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_input_data, input_data.data(),
                                num_elements * sizeof(float),
                                cudaMemcpyHostToDevice));

    // execute kernel function
    batched_reduce_sum_launch_function(d_output_data, d_input_data, batch_size,
                                       num_elements_per_batch, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // verifykernel  correctness
    CHECK_CUDA_ERROR(cudaMemcpy(output_data.data(), d_output_data,
                                batch_size * sizeof(float),
                                cudaMemcpyDeviceToHost));
    for (size_t i{0}; i < batch_size; ++i)
    {
        if (output_data.at(i) != num_elements_per_batch * element_value)
        {
            std::cout << "Expected: " << num_elements_per_batch * element_value
                      << " but got: " << output_data.at(i) << std::endl;
            throw std::runtime_error("Error: incorrect sum");
        }
    }
    
    // bind function for performance measurement
    std::function<void(cudaStream_t)> const bound_function{std::bind(
        batched_reduce_sum_launch_function, d_output_data, d_input_data,
        batch_size, num_elements_per_batch, std::placeholders::_1)};
    float const latency{measure_performance<void>(bound_function, stream)};
    std::cout << "latency: " << latency << " ms" << std::endl;

    // calculateeffective bandwidth
    size_t num_bytes{num_elements * sizeof(float) + batch_size * sizeof(float)};
    float const bandwidth{(num_bytes * 1e-6f) / latency};
    std::cout << "effective bandwidth: " << bandwidth << " GB/s" << std::endl;

    // clean upGPUmemory
    CHECK_CUDA_ERROR(cudaFree(d_input_data));
    CHECK_CUDA_ERROR(cudaFree(d_output_data));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return latency;
}

int main()
{
    size_t const batch_size{2048};                 // batch size
    size_t const num_elements_per_batch{1024 * 256}; // number of elements per batch

    constexpr size_t string_width{50U};
    std::cout << std_string_centered("", string_width, '~') << std::endl;
    std::cout << std_string_centered("NVIDIA GPU device information", string_width,
                                     ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '~') << std::endl;

    // query device name and peak memory bandwidth
    int device_id{0};
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    std::cout << "device name: " << device_prop.name << std::endl;
    float const memory_size{static_cast<float>(device_prop.totalGlobalMem) /
                            (1 << 30)};
    std::cout << "memorysize: " << memory_size << " GB" << std::endl;
    float const peak_bandwidth{
        static_cast<float>(2.0f * device_prop.memoryClockRate *
                           (device_prop.memoryBusWidth / 8) / 1.0e6)};
    std::cout << "peak bandwidth: " << peak_bandwidth << " GB/s" << std::endl;

    std::cout << std_string_centered("", string_width, '~') << std::endl;
    std::cout << std_string_centered("reduction sumperformanceanalysis", string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '~') << std::endl;

    std::cout << std_string_centered("", string_width, '=') << std::endl;
    std::cout << "batch size: " << batch_size << std::endl;
    std::cout << "number of elements per batch: " << num_elements_per_batch
              << std::endl;
    std::cout << std_string_centered("", string_width, '=') << std::endl;

    constexpr size_t NUM_THREADS_PER_BATCH{256}; // number of threads per batch
    static_assert(NUM_THREADS_PER_BATCH % 32 == 0,
                  "NUM_THREADS_PER_BATCH must be a multiple of 32");
    static_assert(NUM_THREADS_PER_BATCH <= 1024,
                  "NUM_THREADS_PER_BATCH must be less than or equal to 1024");

    std::cout << "batchreduction sum V1" << std::endl;
    float const latency_v1{profile_batched_reduce_sum(
        launch_batched_reduce_sum_v1<NUM_THREADS_PER_BATCH>, batch_size,
        num_elements_per_batch)};
    std::cout << std_string_centered("", string_width, '-') << std::endl;

    std::cout << "batchreduction sum V2" << std::endl;
    float const latency_v2{profile_batched_reduce_sum(
        launch_batched_reduce_sum_v2<NUM_THREADS_PER_BATCH>, batch_size,
        num_elements_per_batch)};
    std::cout << std_string_centered("", string_width, '-') << std::endl;
}
```

빌드하고 실행하려면reduction sum예제，다음 명령을 실행한다。

```shell
$ nvcc reduce_sum.cu -o reduce_sum
$ ./reduce_sum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
              NVIDIA GPU device information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
device name: NVIDIA GeForce RTX 3090
memorysize: 23.6694 GB
peak bandwidth: 936.096 GB/s
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
               reduction sumperformanceanalysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
==================================================
batch size: 2048
number of elements per batch: 262144
==================================================
batchreduction sum V1
latency: 2.42976 ms
effective bandwidth: 883.83 GB/s
--------------------------------------------------
batchreduction sum V2
latency: 2.44303 ms
effective bandwidth: 879.028 GB/s
--------------------------------------------------
```

결과는 두 batch reduction sum kernel이 비슷한 performance을 가진다는 것을 보여준다. Effective bandwidth는 GPU peak bandwidth의 약 94%이다. 내 시스템에서는 effective bandwidth가 하루 중 다른 시간대에 실행마다 750 GB/s에서 900 GB/s까지 변할 수 있음에 유의해야 한다.

## Large Array Reduction Sum

더 큰 array와 더 작은 batch size가 있다면 어떻게 해야 할까? Thread block의 최대 thread 수는 1024이다. 더 큰 array의 element sum을 calculate하기 위해 thread block 하나만 할당하고 batch size가 작다면 GPU utilization과 effective bandwidth는 낮아진다.

이 경우 큰 array를 여러 작은 array로 나누어야 한다. 마치 각 큰 array가 array batch인 것처럼 다룬다. 각 작은 array의 element sum을 calculate하기 위해 여러 thread block을 할당한다. 각 작은 array의 element sum이 calculate되면 batch reduction sum kernel을 다시 사용해 partial sum을 final sum으로 더 reduce한다.

구체적으로 data batch의 shape가 `(batch_size, num_elements_per_batch)`라고 하자. `num_elements_per_batch`가 매우 크고 `batch_size`가 매우 작다면, 언제든 data를 `(batch_size * inner_batch_size, inner_num_elements_per_batch)` shape로 reshape하고 batch reduction sum kernel을 실행할 수 있다. 결과 reduction sum의 shape는 `(batch_size * inner_batch_size, 1)`이 된다. 다시 reduction sum을 `(batch_size, inner_batch_size)`로 reshape하고(이를 다시 `(batch_size, num_elements_per_batch)`라고 부르자) batch reduction sum kernel을 실행할 수 있다. 이 과정은 `num_elements_per_batch`가 너무 크지 않을 때까지 반복할 수 있다.

물론 batch reduction sum kernel을 여러 번 실행하고 synchronize하는 대신 atomic operation을 사용해 각 작은 array의 partial sum을 global memory의 final sum에 더할 수도 있다. 그러나 batch reduction sum kernel을 여러 번 실행하고 synchronize하는 방식과 비교했을 때, 이는 performance 저하가 있을 수도 있고 없을 수도 있다.

## 참고 문헌

- Optimizing Parallel Reduction in CUDA - Mark Harris(https://leimao.github.io/downloads/blog/2024-07-30-CUDA-Reduction/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf)
- Parallel Computation Patterns (Reduction)(https://www.cs.ucr.edu/~mchow009/teaching/cs147/winter20/slides/5-Reduction.pdf)
- Using CUDA Warp-Level Primitives(https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)



