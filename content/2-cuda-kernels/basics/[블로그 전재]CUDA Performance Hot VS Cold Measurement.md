> 블로그 출처: https://leimao.github.io/blog/CUDA-Performance-Hot-Cold-Measurement/ , Lei Mao의 글이며 저자의 전재 허가를 받았다. 앞으로 Lei Mao의 CUDA 관련 Blog도 일부 전재할 예정이며, 이는 하나의 완전한 칼럼이다. Blog는 조금 이른 CUDA 아키텍처부터 현재 최신 CUDA 아키텍처까지 다루고, 실용적인 엔지니어링 기법, 하위 수준 명령 분석, Cutlass 분석 등 여러 주제도 포함한다. 시간 흐름이 매우 명확한 칼럼이다.

# CUDA 성능 hot measurement와 cold measurement

## 소개

CUDA kernel의 성능을 측정하기 위해 보통 사용자는 kernel을 여러 번 실행하고 실행 시간의 평균을 취한다. 하지만 CUDA kernel의 성능은 cache effect의 영향을 받을 수 있고, 이로 인해 측정된 성능이 실제 성능과 달라질 수 있다.

예를 들어 성능 측정 중에는 CUDA kernel 호출마다 CUDA kernel이 동일한 입력 data에 접근하므로 DRAM에 접근하지 않고 L2 cache에서 read하게 된다. 반면 실제 application에서는 kernel 호출마다 입력 data가 다를 수 있고, 이 경우 kernel은 DRAM에서 read한다. 특정 use case의 성능 측정에서 cache effect를 제거하기 위해, 사용자는 kernel을 실행할 때마다 GPU L2 cache를 flush할 수 있다. 그러면 kernel은 항상 "cold" 상태에서 실행된다.

이 블로그 글에서는 "hot" 상태와 "cold" 상태에서 CUDA kernel의 성능을 측정하는 방법을 논의한다.

## CUDA 성능 hot measurement와 cold measurement

이전 블로그 글 "Function Binding and Performance Measurement"(https://leimao.github.io/blog/Function-Binding-Performance-Measurement/)에서 나는 function binding을 사용해 CUDA kernel 성능을 측정하는 방법을 논의했다. 그 성능 측정 구현은 실제로 CUDA kernel이 "hot" 상태일 때의 성능만 측정할 수 있다. CUDA kernel이 "cold" 상태일 때의 성능을 측정하려면, kernel을 실행할 때마다 L2 cache를 flush하도록 구현을 조금 수정하면 된다.

### L2 cache flush

CUDA에는 GPU L2 cache를 직접 flush하는 API가 없다. 하지만 GPU memory에 L2 cache와 같은 크기의 buffer를 할당하고 여기에 값을 write할 수 있다. 이는 L2 cache에 이전에 cache되어 있던 모든 값을 evict하게 한다. 다음 예시는 "hot" 상태와 "cold" 상태에서 CUDA kernel의 성능을 측정하는 방법을 보여준다.

```c++
#include <functional>
#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>
// CUDA error를 확인하는 macro definition
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
// CUDA error check function. error가 발생하면 error message를 출력하고 program을 종료한다.
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

// performance measurement function template
// bound_function: bound function object
// stream: CUDA stream
// num_repeats: repeated measurement count
// num_warmups: warmup count
// flush_l2_cache: 각 측정 전에 L2 cache를 flush할지 여부
template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, size_t num_repeats = 100,
                          size_t num_warmups = 100, bool flush_l2_cache = false)
{
    int device_id{0};
    int l2_cache_size{0};
    // current device ID를 얻는다.
    CHECK_CUDA_ERROR(cudaGetDevice(&device_id));
    // L2 cache size를 얻는다.
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&l2_cache_size,
                                            cudaDevAttrL2CacheSize, device_id));

    // L2 cache를 flush하는 데 사용할, L2 cache와 같은 크기의 buffer를 할당한다.
    void* l2_flush_buffer{nullptr};
    CHECK_CUDA_ERROR(
        cudaMalloc(&l2_flush_buffer, static_cast<size_t>(l2_cache_size)));

    // timing을 위한 CUDA event를 생성한다.
    cudaEvent_t start, stop;
    float time{0.0f};
    float call_time{0.0f};

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // warmup 단계: kernel을 여러 번 실행해 performance를 안정화한다.
    for (size_t i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    // 모든 warmup operation이 완료될 때까지 기다린다.
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 실제 측정 단계
    for (size_t i{0}; i < num_repeats; ++i)
    {
        // cold measurement가 필요하면 매 실행 전에 L2 cache를 flush한다.
        if (flush_l2_cache)
        {
            // L2 cache 크기의 buffer에 data를 write해 L2 cache를 flush한다.
            CHECK_CUDA_ERROR(cudaMemsetAsync(l2_flush_buffer, 0,
                                             static_cast<size_t>(l2_cache_size),
                                             stream));
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        }
        // start time을 기록한다.
        CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
        // 측정 대상 function을 실행한다.
        CHECK_CUDA_ERROR(bound_function(stream));
        // end time을 기록한다.
        CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
        // stop event가 완료될 때까지 기다린다.
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        // 이번 실행 시간을 계산한다.
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&call_time, start, stop));
        // total time을 누적한다.
        time += call_time;
    }
    // CUDA event를 파괴한다.
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    // L2 cache flush buffer를 해제한다.
    CHECK_CUDA_ERROR(cudaFree(l2_flush_buffer));

    // average latency를 계산한다.
    float const latency{time / num_repeats};

    return latency;
}

// simple array copy kernel
__global__ void copy(float* output, float const* input, size_t n)
{
    // current thread의 global index를 계산한다.
    size_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    // grid stride, 즉 total thread count를 계산한다.
    size_t const stride{blockDim.x * gridDim.x};
    // grid-stride loop로 array element를 처리한다.
    for (size_t i{idx}; i < n; i += stride)
    {
        output[i] = input[i];
    }
}

// copy kernel을 launch하는 wrapper function
cudaError_t launch_copy(float* output, float const* input, size_t n,
                        cudaStream_t stream)
{
    // thread block size를 설정한다.
    dim3 const threads_per_block{1024};
    // grid size를 설정한다.
    dim3 const blocks_per_grid{32};
    // kernel을 launch한다.
    copy<<<blocks_per_grid, threads_per_block, 0, stream>>>(output, input, n);
    // 마지막 CUDA error status를 반환한다.
    return cudaGetLastError();
}

int main()
{
    int device_id{0};
    // current device ID를 얻는다.
    CHECK_CUDA_ERROR(cudaGetDevice(&device_id));
    cudaDeviceProp device_prop;
    // device property를 얻는다.
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&device_prop, device_id));
    // device name을 출력한다.
    std::cout << "Device Name: " << device_prop.name << std::endl;
    // DRAM size(GB)를 계산하고 출력한다.
    float const memory_size{static_cast<float>(device_prop.totalGlobalMem) /
                            (1 << 30)};
    std::cout << "DRAM Size: " << memory_size << " GB" << std::endl;
    // DRAM peak bandwidth(GB/s)를 계산하고 출력한다.
    float const peak_bandwidth{
        static_cast<float>(2.0f * device_prop.memoryClockRate *
                           (device_prop.memoryBusWidth / 8) / 1.0e6)};
    std::cout << "DRAM Peak Bandwitdh: " << peak_bandwidth << " GB/s"
              << std::endl;
    // L2 cache size를 얻는다.
    int const l2_cache_size{device_prop.l2CacheSize};
    // L2 cache size(MB)를 계산하고 출력한다.
    float const l2_cache_size_mb{static_cast<float>(l2_cache_size) / (1 << 20)};
    std::cout << "L2 Cache Size: " << l2_cache_size_mb << " MB" << std::endl;

    // measurement parameter를 설정한다.
    constexpr size_t num_repeats{10000};  // repeated measurement count
    constexpr size_t num_warmups{1000};   // warmup count

    // array size를 계산한다. L2 cache size의 절반이며 float 단위이다.
    size_t const n{l2_cache_size / 2 / sizeof(float)};
    cudaStream_t stream;

    float *d_input, *d_output;

    // GPU memory를 할당한다.
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * sizeof(float)));

    // CUDA stream을 생성한다.
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // bound function object를 생성한다.
    std::function<cudaError_t(cudaStream_t)> function{
        std::bind(launch_copy, d_output, d_input, n, std::placeholders::_1)};

    // hot 상태의 performance를 측정한다. L2 cache를 flush하지 않는다.
    float const hot_latency{
        measure_performance(function, stream, num_repeats, num_warmups, false)};
    std::cout << std::fixed << std::setprecision(4)
              << "Hot Latency: " << hot_latency << " ms" << std::endl;

    // cold 상태의 performance를 측정한다. 매번 L2 cache를 flush한다.
    float const cold_latency{
        measure_performance(function, stream, num_repeats, num_warmups, true)};
    std::cout << std::fixed << std::setprecision(4)
              << "Cold Latency: " << cold_latency << " ms" << std::endl;

    // resource를 정리한다.
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
```

이 예시를 빌드하고 실행하려면 다음 명령을 실행하라.

```shell
$ nvcc measure_performance.cu -o measure_performance -std=c++14
$ ./measure_performance
Device Name: NVIDIA GeForce RTX 3090
DRAM Size: 23.4365 GB
DRAM Peak Bandwitdh: 936.096 GB/s
L2 Cache Size: 6 MB
Hot Latency: 0.0095 ms
Cold Latency: 0.0141 ms
```

"hot" 상태와 "cold" 상태 사이에 성능 차이가 존재한다는 것을 볼 수 있으며, 이 성능 차이는 cache effect로 인해 발생한다. 하지만 kernel이 memory intensive하지 않거나 cache size가 너무 작아 문제에 도움이 되지 않는다면, "hot" 상태와 "cold" 상태 사이의 성능 차이는 무시할 수 있을 정도일 수 있다.

### Nsight Compute

NVIDIA Nsight Compute를 사용해 CUDA kernel 성능을 측정하는 것도 매우 흔하다.

hardware performance counter 값을 더 deterministic하게 만들기 위해, NVIDIA Nsight Compute는 기본적으로 `--cache-control all`을 사용해 각 replay pass 전에 모든 GPU cache를 flush한다. 따라서 각 pass에서 kernel은 깨끗한 cache에 접근하며, kernel이 완전히 격리된 환경에서 실행되는 것처럼 동작한다.

이 동작은 performance analysis에 바람직하지 않을 수 있다. 특히 measurement가 더 큰 application execution 안의 kernel에 집중되어 있고, 수집되는 data가 cache 중심 metric을 대상으로 할 때 그렇다. 이 경우 `--cache-control none`을 사용해 tool이 hardware cache를 flush하지 않도록 비활성화할 수 있다.

```shell
$ ncu --help
  --cache-control arg (=all)            Control the behavior of the GPU caches during profiling. Allowed values:
                                          all
                                          none
```

## 참고 자료

- Measure Cold - NVBench(https://github.com/NVIDIA/nvbench/blob/c03033b50e46748207b27685b1cdfcbe4a2fec59/nvbench/detail/measure_cold.cuh)
- L2 Flush - NVBench(https://github.com/NVIDIA/nvbench/blob/c03033b50e46748207b27685b1cdfcbe4a2fec59/nvbench/detail/l2flush.cuh)
- Cache Control - Nsight Compute(https://docs.nvidia.com/nsight-compute/2025.1/ProfilingGuide/index.html#cache-control)
