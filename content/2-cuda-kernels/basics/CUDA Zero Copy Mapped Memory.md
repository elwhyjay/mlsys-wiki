> > 블로그 출처: https://leimao.github.io/blog/CUDA-Zero-Copy-Mapped-Memory/ , Lei Mao의 글이며 저자의 전재 허가를 받았다.

# CUDA Zero Copy Mapped Memory

## 소개

unified memory는 NVIDIA Drive series와 NVIDIA Jetson series 같은 NVIDIA embedded platform에서 사용된다. CPU와 integrated GPU가 같은 memory를 사용하므로, discrete GPU system에서 보통 발생하는 host와 device 사이의 CUDA memory copy를 제거할 수 있다. 따라서 GPU가 CPU의 output에 직접 접근할 수 있고, CPU도 GPU의 output에 직접 접근할 수 있다. 이렇게 하면 일부 사용 장면에서 system performance를 크게 높일 수 있다.

이 블로그에서는 CUDA mapped pinned memory와 CUDA non-mapped pinned memory를 논의하고, memory-bound kernel에서 이들의 performance를 비교하고자 한다.

## CUDA Pinned Mapped Memory

CUDA pinned mapped memory는 GPU thread가 host memory에 직접 접근할 수 있게 한다. 이를 위해서는 mapped pinned(non-pageable, page-locked) memory(https://leimao.github.io/blog/Page-Locked-Host-Memory-Data-Transfer/)가 필요하다. integrated GPU, 즉 CUDA device property structure의 integrated field가 1로 설정된 GPU에서는 mapped pinned memory가 항상 performance benefit을 준다. 불필요한 copy를 피할 수 있고 integrated GPU와 CPU memory가 물리적으로 동일하기 때문이다. discrete GPU에서는 mapped pinned memory가 일부 경우에만 유리하다. data가 GPU에 cache되지 않기 때문에 mapped pinned memory는 한 번만 read 또는 write되어야 하며, memory를 read/write하는 global load와 store는 coalesced되어야 한다. zero copy는 stream 대신 사용할 수 있다. kernel이 시작한 data transfer가 stream 수를 설정하고 최적 개수를 결정하는 overhead 없이 자동으로 kernel execution과 overlap되기 때문이다.

## CUDA Pinned Memory: Non-Mapped VS Mapped

다음 구현은 memory-bound kernel의 latency와, 필요할 경우 host와 device 사이의 memory copy를 비교한다.

CUDA mapped memory도 pinned memory를 사용한다. CUDA pinned memory의 경우 여전히 device memory를 할당하고 host memory와 device memory 사이에서 data를 transfer해야 한다. 반면 CUDA mapped memory에서는 device memory allocation과 memory transfer가 있다면 그것이 추상화되어 있다.

```c++
#include <cassert>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
// CUDA API call의 return value를 확인하는 CUDA error check macro
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

// 마지막 CUDA error를 확인하는 macro
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

// CUDA kernel execution time을 측정하기 위한 performance measurement template function
template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, int num_repeats = 100,
                          int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    // timing을 위한 CUDA event를 생성한다.
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // warmup 단계. 첫 실행 overhead가 measurement result에 영향을 주지 않게 한다.
    for (int i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    // stream 안의 모든 operation이 완료될 때까지 기다린다.
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // timing을 시작한다.
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    // average performance를 얻기 위해 여러 번 test를 실행한다.
    for (int i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    // timing을 종료한다.
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    
    // total execution time을 계산한다.
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    
    // event resource를 정리한다.
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    // average latency를 반환한다.
    float const latency{time / num_repeats};

    return latency;
}

// CUDA kernel: floating point addition 연산을 수행한다.
__global__ void float_addition(float* output, float const* input_1,
                               float const* input_2, uint32_t n)
{
    // current thread의 global index를 계산한다.
    const uint32_t idx{blockDim.x * blockIdx.x + threadIdx.x};
    // grid stride, 즉 모든 thread block 안의 total thread count를 계산한다.
    const uint32_t stride{blockDim.x * gridDim.x};
    
    // grid-stride loop로 array element를 처리해 모든 element가 처리되도록 한다.
    for (uint32_t i{idx}; i < n; i += stride)
    {
        output[i] = input_1[i] + input_2[i];
    }
}

// non-mapped pinned memory를 사용해 floating point addition kernel을 launch한다.
void launch_float_addition_non_mapped_pinned_memory(
    float* h_output, float const* h_input_1, float const* h_input_2,
    float* d_output, float* d_input_1, float* d_input_2, uint32_t n,
    cudaStream_t stream)
{
    // 입력 data를 host memory에서 device memory로 비동기 copy한다.
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input_1, h_input_1, n * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input_2, h_input_2, n * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));
    
    // kernel launch parameter를 구성한다.
    dim3 const threads_per_block{1024};  // thread block마다 1024개 thread
    dim3 const blocks_per_grid{32};      // grid 안에 32개 thread block
    
    // kernel을 launch해 floating point addition을 수행한다.
    float_addition<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_output, d_input_1, d_input_2, n);
    CHECK_LAST_CUDA_ERROR();
    
    // 결과를 device memory에서 host memory로 비동기 copy한다.
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_output, d_output, n * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
}

// mapped pinned memory를 사용해 floating point addition kernel을 launch한다.
void launch_float_addition_mapped_pinned_memory(float* d_output,
                                                float* d_input_1,
                                                float* d_input_2, uint32_t n,
                                                cudaStream_t stream)
{
    // kernel launch parameter를 구성한다.
    dim3 const threads_per_block{1024};  // thread block마다 1024개 thread
    dim3 const blocks_per_grid{32};      // grid 안에 32개 thread block
    
    // explicit memory copy 없이 kernel을 직접 launch한다. zero copy이다.
    float_addition<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_output, d_input_1, d_input_2, n);
    CHECK_LAST_CUDA_ERROR();
}

// host memory를 초기화하고 모든 element를 지정한 값으로 설정한다.
void initialize_host_memory(float* h_buffer, uint32_t n, float value)
{
    for (int i{0}; i < n; ++i)
    {
        h_buffer[i] = value;
    }
}

// host memory 안의 모든 element가 expected value와 같은지 검증한다.
bool verify_host_memory(float* h_buffer, uint32_t n, float value)
{
    for (int i{0}; i < n; ++i)
    {
        if (h_buffer[i] != value)
        {
            return false;
        }
    }
    return true;
}

int main()
{
    // performance test parameter
    constexpr int const num_repeats{10};   // repeated test count
    constexpr int const num_warmups{10};   // warmup count

    constexpr int const n{1000000};        // array size
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // test data의 initial value
    float const v_input_1{1.0f};
    float const v_input_2{1.0f};
    float const v_output{0.0f};
    float const v_output_reference{v_input_1 + v_input_2};  // expected output value

    // device가 mapped memory를 지원하는지 확인한다.
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    if (!prop.canMapHostMemory)
    {
        throw std::runtime_error{"Device does not supported mapped memory."};
    }

    // 여러 종류의 memory pointer를 선언한다.
    float *h_input_1, *h_input_2, *h_output;    // ordinary pinned memory(host side)
    float *d_input_1, *d_input_2, *d_output;    // device memory

    float *a_input_1, *a_input_2, *a_output;    // mapped pinned memory(host side)
    float *m_input_1, *m_input_2, *m_output;    // mapped pinned memory(device side pointer)

    // ordinary pinned memory를 할당한다.
    CHECK_CUDA_ERROR(cudaMallocHost(&h_input_1, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_input_2, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_output, n * sizeof(float)));

    // device memory를 할당한다.
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_1, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_2, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * sizeof(float)));

    // mapped pinned memory, 즉 GPU가 직접 접근할 수 있는 host memory를 할당한다.
    CHECK_CUDA_ERROR(
        cudaHostAlloc(&a_input_1, n * sizeof(float), cudaHostAllocMapped));
    CHECK_CUDA_ERROR(
        cudaHostAlloc(&a_input_2, n * sizeof(float), cudaHostAllocMapped));
    CHECK_CUDA_ERROR(
        cudaHostAlloc(&a_output, n * sizeof(float), cudaHostAllocMapped));

    // mapped pinned memory의 device side pointer를 얻는다.
    CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&m_input_1, a_input_1, 0));
    CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&m_input_2, a_input_2, 0));
    CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&m_output, a_output, 0));

    // non-mapped pinned memory 구현의 정확성을 검증한다.
    initialize_host_memory(h_input_1, n, v_input_1);
    initialize_host_memory(h_input_2, n, v_input_2);
    initialize_host_memory(h_output, n, v_output);
    launch_float_addition_non_mapped_pinned_memory(
        h_output, h_input_1, h_input_2, d_output, d_input_1, d_input_2, n,
        stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    assert(verify_host_memory(h_output, n, v_output_reference));

    // mapped pinned memory 구현의 정확성을 검증한다.
    initialize_host_memory(a_input_1, n, v_input_1);
    initialize_host_memory(a_input_2, n, v_input_2);
    initialize_host_memory(a_output, n, v_output);
    launch_float_addition_mapped_pinned_memory(m_output, m_input_1, m_input_2,
                                               n, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    assert(verify_host_memory(a_output, n, v_output_reference));

    // 두 방법의 latency performance를 측정한다.
    // non-mapped pinned memory function을 bind한다.
    std::function<void(cudaStream_t)> function_non_mapped_pinned_memory{
        std::bind(launch_float_addition_non_mapped_pinned_memory, h_output,
                  h_input_1, h_input_2, d_output, d_input_1, d_input_2, n,
                  std::placeholders::_1)};
    // mapped pinned memory function을 bind한다.
    std::function<void(cudaStream_t)> function_mapped_pinned_memory{
        std::bind(launch_float_addition_mapped_pinned_memory, m_output,
                  m_input_1, m_input_2, n, std::placeholders::_1)};
    
    // non-mapped pinned memory의 performance를 측정한다.
    float const latency_non_mapped_pinned_memory{measure_performance(
        function_non_mapped_pinned_memory, stream, num_repeats, num_warmups)};
    // mapped pinned memory의 performance를 측정한다.
    float const latency_mapped_pinned_memory{measure_performance(
        function_mapped_pinned_memory, stream, num_repeats, num_warmups)};
    
    // performance test result를 출력한다.
    std::cout << std::fixed << std::setprecision(3)
              << "CUDA Kernel With Non-Mapped Pinned Memory Latency: "
              << latency_non_mapped_pinned_memory << " ms" << std::endl;
    std::cout << std::fixed << std::setprecision(3)
              << "CUDA Kernel With Mapped Pinned Memory Latency: "
              << latency_mapped_pinned_memory << " ms" << std::endl;

    // 할당된 모든 memory resource를 정리한다.
    CHECK_CUDA_ERROR(cudaFree(d_input_1));
    CHECK_CUDA_ERROR(cudaFree(d_input_2));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFreeHost(h_input_1));
    CHECK_CUDA_ERROR(cudaFreeHost(h_input_2));
    CHECK_CUDA_ERROR(cudaFreeHost(h_output));
    CHECK_CUDA_ERROR(cudaFreeHost(a_input_1));
    CHECK_CUDA_ERROR(cudaFreeHost(a_input_2));
    CHECK_CUDA_ERROR(cudaFreeHost(a_output));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

```

### Discrete GPU

이는 Intel Core i9-9900K CPU와 NVIDIA RTX 3090 GPU를 가진 desktop system에서의 latency performance analysis이다.

```shell
$ nvcc mapped_memory.cu -o mapped_memory -std=c++14
$ ./mapped_memory
CUDA Kernel With Non-Mapped Pinned Memory Latency: 0.964 ms
CUDA Kernel With Mapped Pinned Memory Latency: 0.631 ms
```

memory-bound kernel의 경우 discrete GPU, separate host memory와 device memory를 사용하는 platform에서 mapped pinned memory를 사용하는 것이 non-mapped pinned memory를 사용하는 것보다 거의 30% 빠르다는 것을 볼 수 있다.

### Integrated GPU

이는 NVIDIA Jetson Xavier에서의 latency performance analysis이다.

```shell
$ nvcc mapped_memory.cu -o mapped_memory -std=c++14
$ ./mapped_memory
CUDA Kernel With Non-Mapped Pinned Memory Latency: 2.343 ms
CUDA Kernel With Mapped Pinned Memory Latency: 0.431 ms
```

memory-bound kernel의 경우 integrated GPU와 unified memory를 사용하는 platform에서 mapped pinned memory를 사용하는 것이 non-mapped pinned memory를 사용하는 것보다 거의 6배 빠르다는 것을 볼 수 있다. 이는 mapped memory를 사용하면 unified memory에서 host와 device 사이의 memory copy가 실제로 제거되기 때문이다.

### 주의 사항

CUDA zero copy memory는 GPU의 data cache를 비활성화하므로 compute-bound kernel에서는 performance가 하락할 수 있다.

### 참고 자료

- Function Binding and Performance Measurement(https://leimao.github.io/blog/Function-Binding-Performance-Measurement/)
- NVIDIA CUDA Memory Management(https://developer.ridgerun.com/wiki/index.php/NVIDIA_CUDA_Memory_Management)
- Zero Copy Memory - CUDA Best Practice Guide(https://docs.nvidia.com/cuda/archive/11.7.0/cuda-c-best-practices-guide/index.html#zero-copy)
