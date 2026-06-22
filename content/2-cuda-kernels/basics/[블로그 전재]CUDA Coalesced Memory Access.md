> 블로그 출처: https://leimao.github.io/blog/CUDA-Coalesced-Memory-Access/ 이 글은 Lei Mao의 글이며, 저자의 전재 허가를 받았다.

# CUDA Coalesced Memory Access

## 소개

CUDA 프로그래밍에서 CUDA kernel이 GPU global memory에 access하는 방식은 CUDA kernel performance에 영향을 주는 경우가 많다. Global memory IO를 줄이기 위해 global memory access를 coalesce하여 global memory access 횟수를 줄이고, 재사용 가능한 data를 빠른 shared memory에 cache하기를 원한다.

이 블로그 글에서는，GPU global memory의 read/write access를 coalesce하는 방법을 논의하고, global memory read/write access coalescing이 가져오는 performance 향상을 예제로 보여준다.

## CUDA Matrix Transpose

### implementation

아래 예제에서는 out-of-place matrix transpose를 위한 CUDA kernel 세 개를 implementation했다.

- global memory read access는 coalesced이고, global memory write access는 non-coalesced이다.
- global memory write access는 coalesced이고, global memory read access는 non-coalesced이다.
- global memory read/write access가 모두 coalesced이다. 이는 shared memory를 사용해 implementation한다.

```c++
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>
// CUDA error check macro definition，for checking return values of CUDA API calls
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

// performancemeasurefunctiontemplate，for measuring CUDA kernel execution time
template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, size_t num_repeats = 100,
                          size_t num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    // create CUDA events for timing
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // warmup stage，avoidfirstexecute overheadaffectmeasureresult
    for (size_t i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // start timing and run repeated measurements
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

// helper function for ceiling division
constexpr size_t div_up(size_t a, size_t b) { return (a + b - 1) / b; }

// matrixtransposekernel：global memoryreadcoalesced，writenon-coalesced
// threadby row-majoraccessinputmatrix（coalescedread），but by column-majorwriteoutputmatrix（non-coalescedwrite）
template <typename T>
__global__ void transpose_read_coalesced(T* output_matrix,
                                         T const* input_matrix, size_t M,
                                         size_t N)
{
    // calculatecorresponding to current threadmatrixcoordinate
    size_t const j{threadIdx.x + blockIdx.x * blockDim.x};  // column index
    size_t const i{threadIdx.y + blockIdx.y * blockDim.y};  // row index
    size_t const from_idx{i * N + j};  // inputmatrix linear index（row-major）
    if ((i < M) && (j < N))
    {
        size_t const to_idx{j * M + i};  // outputmatrix linear index（transposed position）
        output_matrix[to_idx] = input_matrix[from_idx];
    }
}

// matrixtransposekernel：global memorywritecoalesced，readnon-coalesced
// threadby column-majoraccessinputmatrix（non-coalescedread），but by row-majorwriteoutputmatrix（coalescedwrite）
template <typename T>
__global__ void transpose_write_coalesced(T* output_matrix,
                                          T const* input_matrix, size_t M,
                                          size_t N)
{
    // note：here coordinatemapping and read_coalesceddifferent
    size_t const j{threadIdx.x + blockIdx.x * blockDim.x};  // outputmatrix column index
    size_t const i{threadIdx.y + blockIdx.y * blockDim.y};  // outputmatrix row index
    size_t const to_idx{i * M + j};  // outputmatrix linear index（row-major，coalescedwrite）
    if ((i < N) && (j < M))
    {
        size_t const from_idx{j * N + i};  // inputmatrix linear index（non-coalescedread）
        output_matrix[to_idx] = input_matrix[from_idx];
    }
}

// launchreadcoalesced matrix transpose kernel
template <typename T>
void launch_transpose_read_coalesced(T* output_matrix, T const* input_matrix,
                                     size_t M, size_t N, cudaStream_t stream)
{
    constexpr size_t const warp_size{32};
    // Use a 32x32 thread block and match warp size to optimize memory access.
    dim3 const threads_per_block{warp_size, warp_size};
    dim3 const blocks_per_grid{static_cast<unsigned int>(div_up(N, warp_size)),
                               static_cast<unsigned int>(div_up(M, warp_size))};
    transpose_read_coalesced<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        output_matrix, input_matrix, M, N);
    CHECK_LAST_CUDA_ERROR();
}

// launchwritecoalesced matrix transpose kernel
template <typename T>
void launch_transpose_write_coalesced(T* output_matrix, T const* input_matrix,
                                      size_t M, size_t N, cudaStream_t stream)
{
    constexpr size_t const warp_size{32};
    dim3 const threads_per_block{warp_size, warp_size};
    // note：griddimension and read_coalesceddifferent，becausecoordinatemappingdifferent
    dim3 const blocks_per_grid{static_cast<unsigned int>(div_up(M, warp_size)),
                               static_cast<unsigned int>(div_up(N, warp_size))};
    transpose_write_coalesced<<<blocks_per_grid, threads_per_block, 0,
                                stream>>>(output_matrix, input_matrix, M, N);
    CHECK_LAST_CUDA_ERROR();
}

// matrixtransposekernel：useshared memoryimplementationboth read and writecoalesced
// throughshared memoryas intermediate buffer，implementationglobal memory coalesced read and coalescedwrite
template <typename T, size_t BLOCK_SIZE = 32>
__global__ void transpose_read_write_coalesced(T* output_matrix,
                                               T const* input_matrix, size_t M,
                                               size_t N)
{
    // useBLOCK_SIZE + 1avoidshared memory bank conflict
    // https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/
    // Try setting this to BLOCK_SIZE instead of BLOCK_SIZE + 1 to observe the performance drop.
    __shared__ T buffer[BLOCK_SIZE][BLOCK_SIZE + 1];

    // calculateinputmatrix coordinate（forcoalescedread）
    size_t const matrix_j{threadIdx.x + blockIdx.x * blockDim.x};
    size_t const matrix_i{threadIdx.y + blockIdx.y * blockDim.y};
    size_t const matrix_from_idx{matrix_i * N + matrix_j};

    // there are two ways matrixdatawrite shared memory：
    // 1. Write transposed matrix data from DRAM to shared memory, then write non-transposed matrix data from shared memory to DRAM.
    // 2. Write non-transposed matrix data from DRAM to shared memory, then write transposed matrix data from shared memory to DRAM.
    // both methods should produce the same performance，even if there isshared memoryaccessbank conflict

    if ((matrix_i < M) && (matrix_j < N))
    {
        // first method： datastore in transposed form in shared memory in 
        buffer[threadIdx.x][threadIdx.y] = input_matrix[matrix_from_idx];
        // second method： datastore in original form in shared memory in 
        // buffer[threadIdx.y][threadIdx.x] = input_matrix[matrix_from_idx];
    }

    // ensure the buffer in the block has been filled
    __syncthreads();

    // calculateoutputmatrix coordinate（forcoalescedwrite）
    size_t const matrix_transposed_j{threadIdx.x + blockIdx.y * blockDim.y};
    size_t const matrix_transposed_i{threadIdx.y + blockIdx.x * blockDim.x};

    if ((matrix_transposed_i < N) && (matrix_transposed_j < M))
    {
        size_t const to_idx{matrix_transposed_i * M + matrix_transposed_j};
        // first method：read from transpose-stored shared memory
        output_matrix[to_idx] = buffer[threadIdx.y][threadIdx.x];
        // second method：read from original-stored shared memory
        // output_matrix[to_idx] = buffer[threadIdx.x][threadIdx.y];
    }
}

// launchboth read and writecoalesced matrix transpose kernel
template <typename T>
void launch_transpose_read_write_coalesced(T* output_matrix,
                                           T const* input_matrix, size_t M,
                                           size_t N, cudaStream_t stream)
{
    constexpr size_t const warp_size{32};
    dim3 const threads_per_block{warp_size, warp_size};
    dim3 const blocks_per_grid{static_cast<unsigned int>(div_up(N, warp_size)),
                               static_cast<unsigned int>(div_up(M, warp_size))};
    transpose_read_write_coalesced<T, warp_size>
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(output_matrix,
                                                            input_matrix, M, N);
    CHECK_LAST_CUDA_ERROR();
}

// comparetwo arrayswhether equal helper function
template <typename T>
bool is_equal(T const* data_1, T const* data_2, size_t size)
{
    for (size_t i{0}; i < size; ++i)
    {
        if (data_1[i] != data_2[i])
        {
            return false;
        }
    }
    return true;
}

// verifymatrixtransposeimplementationcorrectness function
template <typename T>
bool verify_transpose_implementation(
    std::function<void(T*, T const*, size_t, size_t, cudaStream_t)>
        transpose_function,
    size_t M, size_t N)
{
    // fix random seed to ensure reproducibility
    std::mt19937 gen{0};
    cudaStream_t stream;
    size_t const matrix_size{M * N};
    std::vector<T> matrix(matrix_size, 0.0f);
    std::vector<T> matrix_transposed(matrix_size, 1.0f);
    std::vector<T> matrix_transposed_reference(matrix_size, 2.0f);
    std::uniform_real_distribution<T> uniform_dist(-256, 256);
    
    // generate randominputmatrix
    for (size_t i{0}; i < matrix_size; ++i)
    {
        matrix[i] = uniform_dist(gen);
    }
    
    // use CPUcreate referencetransposematrix
    for (size_t i{0}; i < M; ++i)
    {
        for (size_t j{0}; j < N; ++j)
        {
            size_t const from_idx{i * N + j};
            size_t const to_idx{j * M + i};
            matrix_transposed_reference[to_idx] = matrix[from_idx];
        }
    }
    
    // Allocate GPU memory and execute transpose operation.
    T* d_matrix;
    T* d_matrix_transposed;
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, matrix_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_transposed, matrix_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, matrix.data(),
                                matrix_size * sizeof(T),
                                cudaMemcpyHostToDevice));
    transpose_function(d_matrix_transposed, d_matrix, M, N, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaMemcpy(matrix_transposed.data(), d_matrix_transposed,
                                matrix_size * sizeof(T),
                                cudaMemcpyDeviceToHost));
    
    // verifyresultcorrectness
    bool const correctness{is_equal(matrix_transposed.data(),
                                    matrix_transposed_reference.data(),
                                    matrix_size)};
    
    // clean up resources
    CHECK_CUDA_ERROR(cudaFree(d_matrix));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_transposed));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    return correctness;
}

// performancetestfunction
template <typename T>
void profile_transpose_implementation(
    std::function<void(T*, T const*, size_t, size_t, cudaStream_t)>
        transpose_function,
    size_t M, size_t N)
{
    constexpr int num_repeats{100};   // repeat count
    constexpr int num_warmups{10};    // warmup count
    cudaStream_t stream;
    size_t const matrix_size{M * N};
    
    // allocate GPU memory
    T* d_matrix;
    T* d_matrix_transposed;
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, matrix_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_transposed, matrix_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // wrappertransposefunction by for convenienceperformancemeasure
    std::function<void(cudaStream_t)> const transpose_function_wrapped{
        std::bind(transpose_function, d_matrix_transposed, d_matrix, M, N,
                  std::placeholders::_1)};
    
    // measureperformanceand outputresult
    float const transpose_function_latency{measure_performance(
        transpose_function_wrapped, stream, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(3)
              << "Latency: " << transpose_function_latency << " ms"
              << std::endl;
    
    // clean up resources
    CHECK_CUDA_ERROR(cudaFree(d_matrix));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_transposed));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

int main()
{
    // unit test：verify correctness of all implementations
    for (size_t m{1}; m <= 64; ++m)
    {
        for (size_t n{1}; n <= 64; ++n)
        {
            assert(verify_transpose_implementation<float>(
                &launch_transpose_write_coalesced<float>, m, n));
            assert(verify_transpose_implementation<float>(
                &launch_transpose_read_coalesced<float>, m, n));
            assert(verify_transpose_implementation<float>(
                &launch_transpose_read_write_coalesced<float>, m, n));
        }
    }

    // performancetest
    // M: row count
    size_t const M{12800};
    // N: column count
    size_t const N{12800};
    std::cout << M << " x " << N << " Matrix" << std::endl;
    
    std::cout << "Transpose Write Coalesced" << std::endl;
    profile_transpose_implementation<float>(
        &launch_transpose_write_coalesced<float>, M, N);
        
    std::cout << "Transpose Read Coalesced" << std::endl;
    profile_transpose_implementation<float>(
        &launch_transpose_read_coalesced<float>, M, N);
        
    std::cout << "Transpose Read and Write Coalesced" << std::endl;
    profile_transpose_implementation<float>(
        &launch_transpose_read_write_coalesced<float>, M, N);
}

```

### performance

`12800 x 12800` matrix을 사용해 세 CUDA kernel의 performance을 측정했다. performance 측정에 `12800 x 12800` 정방matrix을 사용하는 이유는 global memory coalesced read와 coalesced write의 performance을 가능한 한 공정하게 비교하고 싶었기 때문이다.

`-Xptxas -O0`을 사용하면 CUDA kernel에 대한 모든 NVCC compiler optimization을 비활성화할 수 있다. 적어도 이 사용 사례에서는 global memory coalesced write를 가진 kernel이 global memory coalesced read를 가진 kernel보다 훨씬 빠르다는 것을 볼 수 있다. Kernel에서 global memory coalesced read와 write를 동시에 활성화하면, 이 kernel의 performance이 세 kernel 중 가장 좋다.

```shell
$ nvcc transpose.cu -o transpose -Xptxas -O0
$ ./transpose
12800 x 12800 Matrix
Transpose Write Coalesced
Latency: 5.220 ms
Transpose Read Coalesced
Latency: 7.624 ms
Transpose Read and Write Coalesced
Latency: 4.804 ms
```

`-Xptxas -O3`(compiler 기본 옵션)을 사용하면 CUDA kernel에 대한 모든 NVCC compiler optimization을 활성화할 수 있다. 이 경우에도 세 CUDA kernel의 performance 순서는 변하지 않는다.

```shell
$ nvcc transpose.cu -o transpose -Xptxas -O3
$ ./transpose
12800 x 12800 Matrix
Transpose Write Coalesced
Latency: 2.924 ms
Transpose Read Coalesced
Latency: 5.337 ms
Transpose Read and Write Coalesced
Latency: 2.345 ms
```

모든 측정은 Intel i9-9900K CPU와 NVIDIA RTX 3090 GPU가 있는 플랫폼에서 수행했다.

## 결론

CUDA kernel implementation에서는 global memory read와 write 연산을 가능한 한 coalesce하도록 노력해야 한다.
