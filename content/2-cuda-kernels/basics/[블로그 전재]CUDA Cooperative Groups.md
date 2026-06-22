> 블로그 출처: leimao.github.io/blog/CUDA-Cooperative-Groups/ 이 글은 Lei Mao의 글이며, 저자의 전재 허가를 받았다. 이후 Lei Mao의 CUDA 관련 Blog를 몇 편 더 전재할 예정이고, 이는 하나의 완결된 칼럼이다. Blog는 비교적 이른 시기의 CUDA 아키텍처부터 현재 최신 CUDA 아키텍처까지 다루며, 실용적인 엔지니어링 기법, 저수준 명령어 분석, Cutlass 분석 등 여러 주제를 포함한다.

# CUDA Cooperative Groups

## 소개

CUDA cooperative groups는 개발자가 서로 synchronize하고 communicate할 수 있는 thread group을 만들고 관리하게 해 주는 기능이다. 전통적인 CUDA programming model과 비교하면 cooperative groups는 GPU에서 parallel algorithm을 작성하는 데 더 유연하고 효율적인 방식을 제공한다.

이 블로그 글에서는，parallel reduction algorithm과 CUDA에서 cooperative groups를 사용한 implementation을 논의한다.

## Cooperative Groups를 사용한 Batch Reduction Sum과 Full Reduction Sum implementation

이 예제에서는 이전 블로그 글 "CUDA Reduction"에서 implementation한 두 batch reduction sum kernel을 수정하여 cooperative groups로 thread 간 synchronization과 communication을 수행하게 한다. Reduction algorithm은 완전히 동일하고 thread group을 synchronize하는 API만 다르다. 또한 cooperative groups를 사용해 단일 kernel launch로 element array를 하나의 값으로 reduce하는 full reduction sum kernel도 implementation했다.

```c++
#include <cassert>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

// CUDA error check macro definition，for automatically checking return values of CUDA API calls
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

// macro definition checking last CUDA error
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

// performancemeasurefunctiontemplate，for measuring CUDA operation execution time
template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, size_t num_repeats = 10,
                          size_t num_warmups = 10)
{
    cudaEvent_t start, stop;
    float time;

    // create CUDA events for timing
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // warmup run，ensure GPU is in steady state
    for (size_t i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // start timing and repeat the requested number of times
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

    // calculateaverage latency
    float const latency{time / num_repeats};

    return latency;
}

// utility function for centered string alignment
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

// thread block reduction sum using cooperative groups（template version）
template <size_t NUM_THREADS>
__device__ float thread_block_reduce_sum(
    cooperative_groups::thread_block_tile<NUM_THREADS> group,
    float shared_data[NUM_THREADS], float val)
{
    static_assert(NUM_THREADS % 32 == 0,
                  "NUM_THREADS must be a multiple of 32");
    size_t thread_idx{group.thread_rank()};
    shared_data[thread_idx] = val;
    group.sync(); // synchronize thread group
#pragma unroll
    // use binary reduction algorithm to perform sum
    for (size_t offset{group.size() / 2}; offset > 0; offset /= 2)
    {
        if (thread_idx < offset)
        {
            shared_data[thread_idx] += shared_data[thread_idx + offset];
        }
        group.sync(); // synchronize after each iteration
    }
    // there is no shared memory bank conflict here
    // because multiple threads in warp access the same shared memory location，causing broadcast
    return shared_data[0];
}

// thread block reduction sum using cooperative groups（dynamic version）
__device__ float thread_block_reduce_sum(cooperative_groups::thread_block group,
                                         float* shared_data, float val)
{
    size_t const thread_idx{group.thread_rank()};
    shared_data[thread_idx] = val;
    group.sync(); // synchronize entire thread block
    // use binary reduction algorithm
    for (size_t stride{group.size() / 2}; stride > 0; stride /= 2)
    {
        if (thread_idx < stride)
        {
            shared_data[thread_idx] += shared_data[thread_idx + stride];
        }
        group.sync(); // synchronize after each iteration
    }
    return shared_data[0];
}

// final thread block reduction sum（process results from multiple warps）
template <size_t NUM_WARPS>
__device__ float thread_block_reduce_sum(float shared_data[NUM_WARPS])
{
    float sum{0.0f};
#pragma unroll
    for (size_t i{0}; i < NUM_WARPS; ++i)
    {
        // there is no shared memory bank conflict here
        // because multiple threads in warp access the same shared memory location，causing broadcast
        sum += shared_data[i];
    }
    return sum;
}

// single-thread reduction sum
__device__ float thread_reduce_sum(float const* __restrict__ input_data,
                                   size_t start_offset, size_t num_elements,
                                   size_t stride)
{
    float sum{0.0f};
    // traverse by stride and accumulate elements
    for (size_t i{start_offset}; i < num_elements; i += stride)
    {
        sum += input_data[i];
    }
    return sum;
}

// warp-level reduction sum
__device__ float
warp_reduce_sum(cooperative_groups::thread_block_tile<32> group, float val)
{
#pragma unroll
    for (size_t offset{group.size() / 2}; offset > 0; offset /= 2)
    {
        // shfl_down function is a warp shuffle operation，exists only for thread block tile of size 32
        val += group.shfl_down(val, offset);
    }
    // only the first thread in the warp returns the correct result
    return val;
}

// thread block reduction sum version 1：useallthreadperformreduction
template <size_t NUM_THREADS>
__device__ float
thread_block_reduce_sum_v1(float const* __restrict__ input_data,
                           size_t num_elements)
{
    static_assert(NUM_THREADS % 32 == 0,
                  "NUM_THREADS must be a multiple of 32");
    __shared__ float shared_data[NUM_THREADS];
    size_t const thread_idx{
        cooperative_groups::this_thread_block().thread_index().x};
    // each thread calculates sum of its own elements
    float sum{
        thread_reduce_sum(input_data, thread_idx, num_elements, NUM_THREADS)};
    shared_data[thread_idx] = sum;
    // note：static thread block cooperative groups are still unsupported
    // this way does not work：
    // cooperative_groups::thread_block_tile<NUM_THREADS> const
    // thread_block{cooperative_groups::tiled_partition<NUM_THREADS>(cooperative_groups::this_thread_block())};
    // float const block_sum{thread_block_reduce_sum<NUM_THREADS>(thread_block,
    // shared_data, sum)};
    // this way works：
    float const block_sum{thread_block_reduce_sum(
        cooperative_groups::this_thread_block(), shared_data, sum)};
    return block_sum;
}

// thread block reduction sum version 2：first reduce within warp，then reduce across warps
template <size_t NUM_THREADS, size_t NUM_WARPS = NUM_THREADS / 32>
__device__ float
thread_block_reduce_sum_v2(float const* __restrict__ input_data,
                           size_t num_elements)
{
    static_assert(NUM_THREADS % 32 == 0,
                  "NUM_THREADS must be a multiple of 32");
    __shared__ float shared_data[NUM_WARPS];
    size_t const thread_idx{
        cooperative_groups::this_thread_block().thread_index().x};
    // each thread calculates sum of its own elements
    float sum{
        thread_reduce_sum(input_data, thread_idx, num_elements, NUM_THREADS)};
    // create warp-level cooperative group
    cooperative_groups::thread_block_tile<32> const warp{
        cooperative_groups::tiled_partition<32>(
            cooperative_groups::this_thread_block())};
    // perform reduction within warp
    sum = warp_reduce_sum(warp, sum);
    // only the first thread of each warp stores result to shared memory
    if (warp.thread_rank() == 0)
    {
        shared_data[cooperative_groups::this_thread_block().thread_rank() /
                    32] = sum;
    }
    cooperative_groups::this_thread_block().sync();
    // final reduction of multiple warp results
    float const block_sum{thread_block_reduce_sum<NUM_WARPS>(shared_data)};
    return block_sum;
}

// batchreduction sumkernelversion1
template <size_t NUM_THREADS>
__global__ void batched_reduce_sum_v1(float* __restrict__ output_data,
                                      float const* __restrict__ input_data,
                                      size_t num_elements_per_batch)
{
    static_assert(NUM_THREADS % 32 == 0,
                  "NUM_THREADS must be a multiple of 32");
    size_t const block_idx{cooperative_groups::this_grid().block_rank()};
    size_t const thread_idx{
        cooperative_groups::this_thread_block().thread_rank()};
    // each block processes one batch of data
    float const block_sum{thread_block_reduce_sum_v1<NUM_THREADS>(
        input_data + block_idx * num_elements_per_batch,
        num_elements_per_batch)};
    // only first thread of each block writes result
    if (thread_idx == 0)
    {
        output_data[block_idx] = block_sum;
    }
}

// batchreduction sumkernelversion2
template <size_t NUM_THREADS>
__global__ void batched_reduce_sum_v2(float* __restrict__ output_data,
                                      float const* __restrict__ input_data,
                                      size_t num_elements_per_batch)
{
    static_assert(NUM_THREADS % 32 == 0,
                  "NUM_THREADS must be a multiple of 32");
    constexpr size_t NUM_WARPS{NUM_THREADS / 32};
    size_t const block_idx{cooperative_groups::this_grid().block_rank()};
    size_t const thread_idx{
        cooperative_groups::this_thread_block().thread_rank()};
    // each block processes one batch of data，use version 2 reduction algorithm
    float const block_sum{thread_block_reduce_sum_v2<NUM_THREADS, NUM_WARPS>(
        input_data + block_idx * num_elements_per_batch,
        num_elements_per_batch)};
    // only first thread of each block writes result
    if (thread_idx == 0)
    {
        output_data[block_idx] = block_sum;
    }
}

// full reduction sum kernel：reduce whole array to single value
template <size_t NUM_THREADS, size_t NUM_BLOCK_ELEMENTS>
__global__ void full_reduce_sum(float* output,
                                float const* __restrict__ input_data,
                                size_t num_elements, float* workspace)
{
    static_assert(NUM_THREADS % 32 == 0,
                  "NUM_THREADS must be a multiple of 32");
    static_assert(NUM_BLOCK_ELEMENTS % NUM_THREADS == 0,
                  "NUM_BLOCK_ELEMENTS must be a multiple of NUM_THREADS");
    // workspace size：num_elements
    size_t const num_grid_elements{
        NUM_BLOCK_ELEMENTS * cooperative_groups::this_grid().num_blocks()};
    float* const workspace_ptr_1{workspace};
    float* const workspace_ptr_2{workspace + num_elements / 2};
    size_t remaining_elements{num_elements};

    // first reduction iteration
    float* workspace_output_data{workspace_ptr_1};
    size_t const num_grid_iterations{
        (remaining_elements + num_grid_elements - 1) / num_grid_elements};
    for (size_t i{0}; i < num_grid_iterations; ++i)
    {
        size_t const grid_offset{i * num_grid_elements};
        size_t const block_offset{grid_offset +
                                  cooperative_groups::this_grid().block_rank() *
                                      NUM_BLOCK_ELEMENTS};
        size_t const num_actual_elements_to_reduce_per_block{
            remaining_elements >= block_offset
                ? min(NUM_BLOCK_ELEMENTS, remaining_elements - block_offset)
                : 0};
        // each block reduces its assigned elements
        float const block_sum{thread_block_reduce_sum_v1<NUM_THREADS>(
            input_data + block_offset,
            num_actual_elements_to_reduce_per_block)};
        if (cooperative_groups::this_thread_block().thread_rank() == 0)
        {
            workspace_output_data
                [i * cooperative_groups::this_grid().num_blocks() +
                 cooperative_groups::this_grid().block_rank()] = block_sum;
        }
    }
    // grid-level synchronization：wait for all blocks to finish
    cooperative_groups::this_grid().sync();
    remaining_elements =
        (remaining_elements + NUM_BLOCK_ELEMENTS - 1) / NUM_BLOCK_ELEMENTS;

    // subsequent reduction iterations
    float* workspace_input_data{workspace_output_data};
    workspace_output_data = workspace_ptr_2;
    while (remaining_elements > 1)
    {
        size_t const num_grid_iterations{
            (remaining_elements + num_grid_elements - 1) / num_grid_elements};
        for (size_t i{0}; i < num_grid_iterations; ++i)
        {
            size_t const grid_offset{i * num_grid_elements};
            size_t const block_offset{
                grid_offset + cooperative_groups::this_grid().block_rank() *
                                  NUM_BLOCK_ELEMENTS};
            size_t const num_actual_elements_to_reduce_per_block{
                remaining_elements >= block_offset
                    ? min(NUM_BLOCK_ELEMENTS, remaining_elements - block_offset)
                    : 0};
            // data in reduction workspace
            float const block_sum{thread_block_reduce_sum_v1<NUM_THREADS>(
                workspace_input_data + block_offset,
                num_actual_elements_to_reduce_per_block)};
            if (cooperative_groups::this_thread_block().thread_rank() == 0)
            {
                workspace_output_data
                    [i * cooperative_groups::this_grid().num_blocks() +
                     cooperative_groups::this_grid().block_rank()] = block_sum;
            }
        }
        // grid-level synchronization
        cooperative_groups::this_grid().sync();
        remaining_elements =
            (remaining_elements + NUM_BLOCK_ELEMENTS - 1) / NUM_BLOCK_ELEMENTS;

        // swap input and output data pointers
        float* const temp{workspace_input_data};
        workspace_input_data = workspace_output_data;
        workspace_output_data = temp;
    }

    // copy final result to output
    workspace_output_data = workspace_input_data;
    if (cooperative_groups::this_grid().thread_rank() == 0)
    {
        *output = workspace_output_data[0];
    }
}

// launchbatchreduction sumversion1 wrapper function
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

// launchbatchreduction sumversion2 wrapper function
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

// launch wrapper function for full reduction sum
template <size_t NUM_THREADS, size_t NUM_BLOCK_ELEMENTS>
void launch_full_reduce_sum(float* output, float const* input_data,
                            size_t num_elements, float* workspace,
                            cudaStream_t stream)
{
    // reference:https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#grid-synchronization
    void const* func{reinterpret_cast<void const*>(
        full_reduce_sum<NUM_THREADS, NUM_BLOCK_ELEMENTS>)};
    int dev{0};
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, dev));
    // use number of device multiprocessors as grid size
    dim3 const grid_dim{
        static_cast<unsigned int>(deviceProp.multiProcessorCount)};
    dim3 const block_dim{NUM_THREADS};

    // this launches a grid that maximally fills the GPU
    // in practice，this is not always the best choice
    // void const* func{reinterpret_cast<void const*>(
    //     full_reduce_sum<NUM_THREADS, NUM_BLOCK_ELEMENTS>)};
    // int dev{0};
    // dim3 const block_dim{NUM_THREADS};
    // int num_blocks_per_sm{0};
    // cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp, dev);
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, func,
    //                                               NUM_THREADS, 0);
    // dim3 const grid_dim{static_cast<unsigned int>(num_blocks_per_sm)};

    void* args[]{static_cast<void*>(&output), static_cast<void*>(&input_data),
                 static_cast<void*>(&num_elements),
                 static_cast<void*>(&workspace)};
    // launch cooperative kernel
    CHECK_CUDA_ERROR(cudaLaunchCooperativeKernel(func, grid_dim, block_dim,
                                                 args, 0, stream));
    CHECK_LAST_CUDA_ERROR();
}

// full reduction sum performance analysis function
float profile_full_reduce_sum(
    std::function<void(float*, float const*, size_t, float*, cudaStream_t)>
        full_reduce_sum_launch_function,
    size_t num_elements)
{
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    constexpr float element_value{1.0f};
    std::vector<float> input_data(num_elements, element_value);
    float output{0.0f};

    float* d_input_data;
    float* d_workspace;
    float* d_output;

    // allocate GPU memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_data, num_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_workspace, num_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, sizeof(float)));

    // copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input_data, input_data.data(),
                                num_elements * sizeof(float),
                                cudaMemcpyHostToDevice));

    // execute reduction operation
    full_reduce_sum_launch_function(d_output, d_input_data, num_elements,
                                    d_workspace, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // verify kernel correctness
    CHECK_CUDA_ERROR(
        cudaMemcpy(&output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    if (output != num_elements * element_value)
    {
        std::cout << "Expected: " << num_elements * element_value
                  << " but got: " << output << std::endl;
        throw std::runtime_error("Error: incorrect sum");
    }
    // measureperformance
    std::function<void(cudaStream_t)> const bound_function{
        std::bind(full_reduce_sum_launch_function, d_output, d_input_data,
                  num_elements, d_workspace, std::placeholders::_1)};
    float const latency{measure_performance<void>(bound_function, stream)};
    std::cout << "Latency: " << latency << " ms" << std::endl;

    // calculateeffective bandwidth
    size_t num_bytes{num_elements * sizeof(float) + sizeof(float)};
    float const bandwidth{(num_bytes * 1e-6f) / latency};
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // clean upGPUmemory
    CHECK_CUDA_ERROR(cudaFree(d_input_data));
    CHECK_CUDA_ERROR(cudaFree(d_workspace));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return latency;
}

// batch reduction sum performance analysis function
float profile_batched_reduce_sum(
    std::function<void(float*, float const*, size_t, size_t, cudaStream_t)>
        batched_reduce_sum_launch_function,
    size_t batch_size, size_t num_elements_per_batch)
{
    size_t const num_elements{batch_size * num_elements_per_batch};

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    constexpr float element_value{1.0f};
    std::vector<float> input_data(num_elements, element_value);
    std::vector<float> output_data(batch_size, 0.0f);

    float* d_input_data;
    float* d_output_data;

    // allocate GPU memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_data, num_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_data, batch_size * sizeof(float)));

    // copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input_data, input_data.data(),
                                num_elements * sizeof(float),
                                cudaMemcpyHostToDevice));

    // execute batch reduction operation
    batched_reduce_sum_launch_function(d_output_data, d_input_data, batch_size,
                                       num_elements_per_batch, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // verify kernel correctness
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
    // measureperformance
    std::function<void(cudaStream_t)> const bound_function{std::bind(
        batched_reduce_sum_launch_function, d_output_data, d_input_data,
        batch_size, num_elements_per_batch, std::placeholders::_1)};
    float const latency{measure_performance<void>(bound_function, stream)};
    std::cout << "Latency: " << latency << " ms" << std::endl;

    // calculateeffective bandwidth
    size_t num_bytes{num_elements * sizeof(float) + batch_size * sizeof(float)};
    float const bandwidth{(num_bytes * 1e-6f) / latency};
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // clean upGPUmemory
    CHECK_CUDA_ERROR(cudaFree(d_input_data));
    CHECK_CUDA_ERROR(cudaFree(d_output_data));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return latency;
}

// main function：demonstrate and test different reduction algorithms
int main()
{
    size_t const batch_size{2048};
    size_t const num_elements_per_batch{1024 * 256};

    constexpr size_t string_width{50U};
    std::cout << std_string_centered("", string_width, '~') << std::endl;
    std::cout << std_string_centered("NVIDIA GPU Device Info", string_width,
                                     ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '~') << std::endl;

    // query device name and peak memory bandwidth
    int device_id{0};
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    std::cout << "Device Name: " << device_prop.name << std::endl;
    float const memory_size{static_cast<float>(device_prop.totalGlobalMem) /
                            (1 << 30)};
    std::cout << "Memory Size: " << memory_size << " GB" << std::endl;
    float const peak_bandwidth{
        static_cast<float>(2.0f * device_prop.memoryClockRate *
                           (device_prop.memoryBusWidth / 8) / 1.0e6)};
    std::cout << "Peak Bandwitdh: " << peak_bandwidth << " GB/s" << std::endl;

    std::cout << std_string_centered("", string_width, '~') << std::endl;
    std::cout << std_string_centered("Reduce Sum Profiling", string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '~') << std::endl;

    std::cout << std_string_centered("", string_width, '=') << std::endl;
    std::cout << "Batch Size: " << batch_size << std::endl;
    std::cout << "Number of Elements Per Batch: " << num_elements_per_batch
              << std::endl;
    std::cout << std_string_centered("", string_width, '=') << std::endl;

    constexpr size_t NUM_THREADS_PER_BATCH{256};
    static_assert(NUM_THREADS_PER_BATCH % 32 == 0,
                  "NUM_THREADS_PER_BATCH must be a multiple of 32");
    static_assert(NUM_THREADS_PER_BATCH <= 1024,
                  "NUM_THREADS_PER_BATCH must be less than or equal to 1024");

    // testbatchreduction sumversion1
    std::cout << "Batched Reduce Sum V1" << std::endl;
    float const latency_v1{profile_batched_reduce_sum(
        launch_batched_reduce_sum_v1<NUM_THREADS_PER_BATCH>, batch_size,
        num_elements_per_batch)};
    std::cout << std_string_centered("", string_width, '-') << std::endl;

    // testbatchreduction sumversion2
    std::cout << "Batched Reduce Sum V2" << std::endl;
    float const latency_v2{profile_batched_reduce_sum(
        launch_batched_reduce_sum_v2<NUM_THREADS_PER_BATCH>, batch_size,
        num_elements_per_batch)};
    std::cout << std_string_centered("", string_width, '-') << std::endl;

    // test full reduction sum
    std::cout << "Full Reduce Sum" << std::endl;
    constexpr size_t NUM_THREADS{256};
    constexpr size_t NUM_BLOCK_ELEMENTS{NUM_THREADS * 1024};
    float const latency_v3{profile_full_reduce_sum(
        launch_full_reduce_sum<NUM_THREADS, NUM_BLOCK_ELEMENTS>,
        batch_size * num_elements_per_batch)};
    std::cout << std_string_centered("", string_width, '-') << std::endl;
}
```

이 reduction sum 예제를 빌드하고 실행하려면 다음 명령을 실행한다.

```shell
$ nvcc reduce_sum_cooperative_groups.cu -o reduce_sum_cooperative_groups
$ ./reduce_sum_cooperative_groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
              NVIDIA GPU Device Info
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Device Name: NVIDIA GeForce RTX 3090
Memory Size: 23.6694 GB
Peak Bandwitdh: 936.096 GB/s
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
               Reduce Sum Profiling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
==================================================
Batch Size: 2048
Number of Elements Per Batch: 262144
==================================================
Batched Reduce Sum V1
Latency: 2.43301 ms
Effective Bandwidth: 882.649 GB/s
--------------------------------------------------
Batched Reduce Sum V2
Latency: 2.43445 ms
Effective Bandwidth: 882.126 GB/s
--------------------------------------------------
Full Reduce Sum
Latency: 2.47788 ms
Effective Bandwidth: 866.663 GB/s
--------------------------------------------------
```

cooperative groups를 사용하는 batch reduction sum kernel의 performance은 traditional CUDA programming model을 사용하는 batch reduction sum kernel performance과 비슷하다.

## Large Array Reduction Sum

large array reduction sum kernel을 implementation하는 방법은 세 가지가 있을 수 있다.

- batch reduction sum kernel launch를 여러 번 사용해 array를 iterative하게 reduce한다.
- full reduction sum kernel launch 하나를 사용해 array를 iterative하게 reduce하며, 이 kernel은 grid cooperative group이 관리한다.

cooperative groups를 사용하지 않으면 thread block 안에서만 thread를 synchronize할 수 있으며, 그래서 첫 번째 방법이 된다. 하지만 여러 번 kernel launch를 하므로 추가 kernel launch overhead가 있다.

cooperative groups를 사용하면 thread block을 넘어 thread를 synchronize할 수 있으며, 그래서 두 번째 방법이 된다. 그러나 첫 번째 방법과 비교해 두 번째 방법에도 단점이 있다. Reduction 후반 단계에서는 reduction problem의 size가 작아지기 때문에 실제로 사용하는 grid 수가 훨씬 적고, 이는 computing resource 낭비다.

## 참고 문헌

- Cooperative Groups: Flexible CUDA Thread Programming(https://developer.nvidia.com/blog/cooperative-groups/)
- Cooperative Groups - NVIDIA GTC 2017(https://leimao.github.io/downloads/blog/2024-08-06-CUDA-Cooperative-Groups/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf)

