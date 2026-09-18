> 블로그 출처: https://leimao.github.io/blog/CUDA-Vectorized-Memory-Access/ 이 글은 Lei Mao의 글이며, 저자의 전재 허가를 받았다. 앞으로도 Lei Mao의 CUDA 관련 블로그를 계속 전재할 예정이며, 이는 하나의 완결된 시리즈이기도 하다. 이 블로그는 다소 이전 세대의 CUDA 아키텍처부터 현재 최신 CUDA 아키텍처까지 다루고, 실용적인 엔지니어링 기법, 저수준 명령어 분석, Cutlass 분석 등 여러 주제를 포함하는, 시간 순서가 매우 뚜렷한 시리즈다.

# CUDA 벡터화 메모리 접근

## 소개

DRAM에서 데이터를 읽고 쓰는 것은 CUDA 프로그래밍의 기본 연산 중 하나다. CUDA 디바이스의 유효 메모리 대역폭은 CUDA 함수의 성능을 좌우하는 가장 중요한 요인 중 하나이며, 특히 CUDA 함수가 memory bound일 때 그렇다.

이 블로그 글에서는 벡터화 메모리 접근을 사용해 CUDA 함수의 유효 메모리 대역폭을 높이는 방법을 보여준다.

## CUDA 벡터화 메모리 접근

아래 예제에서는 단순한 형태의 커스텀 device memcpy 함수를 구현하고, 여러 데이터 타입의 연속된 데이터에 대해 thread당 8바이트 또는 16바이트의 벡터화 메모리 트랜잭션을 사용함으로써 유효 메모리 대역폭을 어떻게 높일 수 있는지 보여준다. thread당 8바이트 또는 16바이트 벡터화 메모리 트랜잭션을 사용하면 데이터 복사에 필요한 메모리 트랜잭션 수가 줄어들고, 이는 거의 모든 사용 사례에서 유효 메모리 대역폭을 높여준다.

```c++
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
// CUDA API 호출의 반환값을 검사하기 위한 CUDA 에러 체크 매크로
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

// 마지막 CUDA 에러를 검사하는 매크로
#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(const char* const file, const int line)
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

// 출력 포맷팅에 사용하는 문자열 가운데 정렬 함수
std::string std_string_centered(std::string const& s, size_t width,
                                char pad = ' ')
{
    size_t const l{s.length()};
    // 폭이 너무 작으면 예외를 던진다
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

// CUDA 함수의 실행 시간을 측정하기 위한 성능 측정 함수 템플릿
template <class T>
float measure_performance(std::function<T(cudaStream_t)> const& bound_function,
                          cudaStream_t stream, unsigned int num_repeats = 100,
                          unsigned int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    // 시간 측정을 위한 CUDA 이벤트 생성
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // 워밍업 실행, 첫 실행의 오버헤드가 측정 결과에 영향을 주지 않도록 한다
    for (unsigned int i{0U}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 계측을 시작하고 여러 번 반복 측정을 수행
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (unsigned int i{0U}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    // 평균 레이턴시 계산
    float const latency{time / num_repeats};

    return latency;
}

// 기본적인 커스텀 device memcpy 커널 함수
// 각 thread가 하나의 데이터 원소를 처리한다
template <typename T>
__global__ void custom_device_memcpy(T* __restrict__ output,
                                     T const* __restrict__ input, size_t n)
{
    // 현재 thread의 전역 인덱스 계산
    size_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    // 전체 thread 수보다 큰 데이터를 처리하기 위한 grid stride 계산
    size_t const stride{blockDim.x * gridDim.x};
    for (size_t i{idx}; i < n; i += stride)
    {
        output[i] = input[i];
    }
}

// 기본 커스텀 device memcpy를 실행하는 래퍼 함수
template <typename T>
void launch_custom_device_memcpy(T* output, T const* input, size_t n,
                                 cudaStream_t stream)
{
    dim3 const threads_per_block{1024};
    // 필요한 block 수 계산, unsigned int의 최댓값을 넘지 않도록 한다
    dim3 const blocks_per_grid{static_cast<unsigned int>(std::min(
        (n + threads_per_block.x - 1U) / threads_per_block.x,
        static_cast<size_t>(std::numeric_limits<unsigned int>::max())))};
    custom_device_memcpy<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        output, input, n);
    CHECK_LAST_CUDA_ERROR();
}

// shared memory를 중간 버퍼로 사용하는 커스텀 device memcpy 커널 함수
template <typename T, unsigned int BLOCK_DIM_X>
__global__ void custom_device_memcpy_shared_memory(T* __restrict__ output,
                                                   T const* __restrict__ input,
                                                   size_t n)
{
    // shared memory를 중간 버퍼로 사용
    __shared__ T shared_memory[BLOCK_DIM_X];
    size_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    size_t const stride{blockDim.x * gridDim.x};
    for (size_t i{idx}; i < n; i += stride)
    {
        // 먼저 global memory에서 shared memory로 데이터를 읽는다
        shared_memory[threadIdx.x] = input[i];
        // 이 경우에는 각 thread가 자신의 shared memory 위치에만 접근하므로 동기화가 필요 없다
        // __syncthreads();
        // 다시 shared memory에서 출력 global memory로 쓴다
        output[i] = shared_memory[threadIdx.x];
    }
}

// shared memory를 사용하는 커스텀 device memcpy를 실행하는 래퍼 함수
template <typename T>
void launch_custom_device_memcpy_shared_memory(T* output, T const* input,
                                               size_t n, cudaStream_t stream)
{
    constexpr dim3 threads_per_block{1024};
    dim3 const blocks_per_grid{static_cast<unsigned int>(std::min(
        (n + threads_per_block.x - 1U) / threads_per_block.x,
        static_cast<size_t>(std::numeric_limits<unsigned int>::max())))};
    custom_device_memcpy_shared_memory<T, threads_per_block.x>
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(output, input, n);
    CHECK_LAST_CUDA_ERROR();
}

// 벡터화 메모리 접근을 사용하는 최적화된 커스텀 device memcpy 커널 함수
// 하나의 thread가 sizeof(R)바이트의 데이터를 복사한다
// 하나의 warp가 몇 번 되지 않는 메모리 트랜잭션으로 32 x sizeof(R)바이트의 데이터를 복사한다
template <typename T, typename R = uint64_t>
__global__ void custom_device_memcpy_optimized(T* __restrict__ output,
                                               T const* __restrict__ input,
                                               size_t n)
{
    size_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    size_t const stride{blockDim.x * gridDim.x};
    // R 타입의 크기에 맞춰 벡터화 접근을 수행
    for (size_t i{idx}; i * sizeof(R) / sizeof(T) < n; i += stride)
    {
        // R 크기의 데이터 블록을 온전히 복사할 수 있는지 검사
        if ((i + 1U) * sizeof(R) / sizeof(T) < n)
        {
            // 벡터화 메모리 접근을 사용해 한 번에 sizeof(R)바이트를 복사
            reinterpret_cast<R*>(output)[i] =
                reinterpret_cast<R const*>(input)[i];
        }
        else
        {
            // R 크기에 미치지 못하는 나머지 데이터를 처리
            size_t const start_index{i * sizeof(R) / sizeof(T)};
            size_t const remaining_units_to_copy{(n - start_index)};
            for (size_t j{0}; j < remaining_units_to_copy; ++j)
            {
                output[start_index + j] = input[start_index + j];
            }
        }
    }
}

// 최적화된 커스텀 device memcpy를 실행하는 래퍼 함수
template <typename T, typename R = uint64_t>
void launch_custom_device_memcpy_optimized(T* output, T const* input, size_t n,
                                           cudaStream_t stream)
{
    dim3 const threads_per_block{1024};
    // 복사해야 할 R 타입 단위의 개수 계산(올림)
    size_t const num_units_to_copy_round_up{(n * sizeof(T) + sizeof(R) - 1U) /
                                            sizeof(R)};
    dim3 const blocks_per_grid{static_cast<unsigned int>(std::min(
        (num_units_to_copy_round_up + threads_per_block.x - 1U) /
            threads_per_block.x,
        static_cast<size_t>(std::numeric_limits<unsigned int>::max())))};
    custom_device_memcpy_optimized<<<blocks_per_grid, threads_per_block, 0,
                                     stream>>>(output, input, n);
    CHECK_LAST_CUDA_ERROR();
}

// CUDA 공식 memcpy 함수를 사용하는 래퍼 함수
template <typename T>
void launch_official_device_memcpy(T* output, T const* input, size_t n,
                                   cudaStream_t stream)
{
    CHECK_CUDA_ERROR(cudaMemcpyAsync(output, input, n * sizeof(T),
                                     cudaMemcpyDeviceToDevice, stream));
}

// 데이터 단위의 값이 자신의 인덱스와 같아지도록 버퍼를 초기화
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
void initialize_buffer(T* buffer, size_t n)
{
    for (size_t i{0}; i < n; ++i)
    {
        buffer[i] = static_cast<T>(
            i % static_cast<size_t>(std::numeric_limits<T>::max()));
    }
}

// 버퍼 데이터의 정확성을 검증
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
void verify_buffer(T* buffer, size_t n)
{
    for (size_t i{0}; i < n; ++i)
    {
        if (buffer[i] != static_cast<T>(i % static_cast<size_t>(
                                                std::numeric_limits<T>::max())))
        {
            std::cerr << "Verification failed at index: " << i << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}

// 커스텀 device memcpy 성능을 측정하는 함수
// 복사할 단위 개수, 사용할 device memcpy 함수, 그리고 반복 횟수와 워밍업 횟수를 받는다
template <typename T>
float measure_custom_device_memcpy_performance(
    size_t n,
    std::function<void(T*, T const*, size_t, cudaStream_t)> const&
        device_memcpy_function,
    int num_repeats = 100, int num_warmups = 100)
{
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // 호스트 측 입력 및 출력 버퍼 준비
    std::vector<T> input(n);
    std::vector<T> output(n, static_cast<T>(0));
    initialize_buffer(input.data(), n);

    // 디바이스 측 메모리 할당
    T* d_input;
    T* d_output;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * sizeof(T)));

    // 호스트에서 디바이스로 데이터 복사
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input, input.data(), n * sizeof(T),
                                     cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_output, output.data(), n * sizeof(T),
                                     cudaMemcpyHostToDevice, stream));
    // 정확성을 확인하기 위해 device memcpy를 한 번 실행
    device_memcpy_function(d_output, d_input, n, stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(output.data(), d_output, n * sizeof(T),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // device memcpy의 정확성 검증
    verify_buffer(output.data(), n);

    // 데이터 크기와 성능 지표 계산
    size_t const num_bytes{n * sizeof(T)};
    float const num_giga_bytes{static_cast<float>(num_bytes) / (1 << 30)};

    // 성능 측정을 위한 바인딩 함수 생성
    std::function<void(cudaStream_t)> function{std::bind(
        device_memcpy_function, d_output, d_input, n, std::placeholders::_1)};

    // 레이턴시를 측정하고 대역폭을 계산
    float const latency{
        measure_performance(function, stream, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(3) << "Latency: " << latency
              << " ms" << std::endl;
    std::cout << "Effective Bandwitdh: "
              << 2.f * num_giga_bytes / (latency / 1000) << " GB/s"
              << std::endl;

    // 디바이스 메모리 정리
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    // 디바이스 이름과 최대 메모리 대역폭 조회
    int device_id{0};
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    float const peak_bandwidth{
        static_cast<float>(2.0 * device_prop.memoryClockRate *
                           (device_prop.memoryBusWidth / 8) / 1.0e6)};
    std::cout << "Percentage of Peak Bandwitdh: "
              << 2.f * num_giga_bytes / (latency / 1000) / peak_bandwidth * 100
              << "%" << std::endl;

    return latency;
}

int main()
{
    constexpr unsigned int num_repeats{10U};
    constexpr unsigned int num_warmups{10U};

    constexpr size_t tensor_size_small{1U * 64U * 64U * 64U};
    constexpr size_t tensor_size_medium{1U * 128U * 128U * 128U};
    constexpr size_t tensor_size_large{1U * 512U * 512U * 512U};

    constexpr size_t string_width{50U};

    std::cout << std_string_centered("", string_width, '~') << std::endl;
    std::cout << std_string_centered("NVIDIA GPU Device Info", string_width,
                                     ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '~') << std::endl;

    // Query deive name and peak memory bandwidth.
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
    std::cout << std::endl;

    // Measure CUDA official memcpy performance for different tensor sizes.
    std::cout << std_string_centered("", string_width, '*') << std::endl;
    std::cout << std_string_centered("CUDA Official Memcpy", string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '*') << std::endl;

    for (size_t tensor_size :
         {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::string const tensor_size_string{std::string("Tensor Size: ") +
                                             std::to_string(tensor_size) +
                                             std::string(" Units")};
        std::cout << std_string_centered("", string_width, '=') << std::endl;
        std::cout << std_string_centered(tensor_size_string, string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '=') << std::endl;

        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 1 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int8_t>(
            tensor_size, launch_official_device_memcpy<int8_t>, num_repeats,
            num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 2 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int16_t>(
            tensor_size, launch_official_device_memcpy<int16_t>, num_repeats,
            num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 4 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int32_t>(
            tensor_size, launch_official_device_memcpy<int32_t>, num_repeats,
            num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 8 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int64_t>(
            tensor_size, launch_official_device_memcpy<int64_t>, num_repeats,
            num_warmups);
    }
    std::cout << std::endl;

    // Measure the latency and bandwidth of custom device memcpy for different
    // tensor sizes.
    std::cout << std_string_centered("", string_width, '*') << std::endl;
    std::cout << std_string_centered("Custom Device Memcpy", string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '*') << std::endl;

    for (size_t tensor_size :
         {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::string const tensor_size_string{std::string("Tensor Size: ") +
                                             std::to_string(tensor_size) +
                                             std::string(" Units")};
        std::cout << std_string_centered("", string_width, '=') << std::endl;
        std::cout << std_string_centered(tensor_size_string, string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '=') << std::endl;

        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 1 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int8_t>(
            tensor_size, launch_custom_device_memcpy<int8_t>, num_repeats,
            num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 2 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int16_t>(
            tensor_size, launch_custom_device_memcpy<int16_t>, num_repeats,
            num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 4 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int32_t>(
            tensor_size, launch_custom_device_memcpy<int32_t>, num_repeats,
            num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 8 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int64_t>(
            tensor_size, launch_custom_device_memcpy<int64_t>, num_repeats,
            num_warmups);
    }
    std::cout << std::endl;

    // Conclusions:
    // 1. The more units of data we copy, the higher the bandwidth.
    // 2. The larger the unit of the data, the higher the bandwidth.

    // Check if shared memory can improve the latency of custom device memcpy.
    std::cout << std_string_centered("", string_width, '*') << std::endl;
    std::cout << std_string_centered("Custom Device Memcpy with Shared Memory",
                                     string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '*') << std::endl;

    for (size_t tensor_size :
         {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::string const tensor_size_string{std::string("Tensor Size: ") +
                                             std::to_string(tensor_size) +
                                             std::string(" Units")};
        std::cout << std_string_centered("", string_width, '=') << std::endl;
        std::cout << std_string_centered(tensor_size_string, string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '=') << std::endl;

        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 1 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int8_t>(
            tensor_size, launch_custom_device_memcpy_shared_memory<int8_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 2 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int16_t>(
            tensor_size, launch_custom_device_memcpy_shared_memory<int16_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 4 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int32_t>(
            tensor_size, launch_custom_device_memcpy_shared_memory<int32_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 8 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int64_t>(
            tensor_size, launch_custom_device_memcpy_shared_memory<int64_t>,
            num_repeats, num_warmups);
    }
    std::cout << std::endl;

    // Conclusions:
    // 1. The effect of using shared memory for improving the latency of custom
    // device memcpy is not obvious.

    // Improve the latency of custom device memcpy when the unit of the data is
    // small.
    std::cout << std_string_centered("", string_width, '*') << std::endl;
    std::cout << std_string_centered(
                     "Custom Device Memcpy 4-Byte Copy Per Thread",
                     string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '*') << std::endl;

    for (size_t tensor_size :
         {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::string const tensor_size_string{std::string("Tensor Size: ") +
                                             std::to_string(tensor_size) +
                                             std::string(" Units")};
        std::cout << std_string_centered("", string_width, '=') << std::endl;
        std::cout << std_string_centered(tensor_size_string, string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '=') << std::endl;

        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 1 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int8_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int8_t, uint32_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 2 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int16_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int16_t, uint32_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 4 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int32_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int32_t, uint32_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 8 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int64_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int64_t, uint32_t>,
            num_repeats, num_warmups);
    }
    std::cout << std::endl;

    std::cout << std_string_centered("", string_width, '*') << std::endl;
    std::cout << std_string_centered(
                     "Custom Device Memcpy 8-Byte Copy Per Thread",
                     string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '*') << std::endl;

    for (size_t tensor_size :
         {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::string const tensor_size_string{std::string("Tensor Size: ") +
                                             std::to_string(tensor_size) +
                                             std::string(" Units")};
        std::cout << std_string_centered("", string_width, '=') << std::endl;
        std::cout << std_string_centered(tensor_size_string, string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '=') << std::endl;

        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 1 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int8_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int8_t, uint64_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 2 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int16_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int16_t, uint64_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 4 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int32_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int32_t, uint64_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 8 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int64_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int64_t, uint64_t>,
            num_repeats, num_warmups);
    }
    std::cout << std::endl;

    std::cout << std_string_centered("", string_width, '*') << std::endl;
    std::cout << std_string_centered(
                     "Custom Device Memcpy 16-Byte Copy Per Thread",
                     string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '*') << std::endl;

    for (size_t tensor_size :
         {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::string const tensor_size_string{std::string("Tensor Size: ") +
                                             std::to_string(tensor_size) +
                                             std::string(" Units")};
        std::cout << std_string_centered("", string_width, '=') << std::endl;
        std::cout << std_string_centered(tensor_size_string, string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '=') << std::endl;

        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 1 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int8_t>(
            tensor_size, launch_custom_device_memcpy_optimized<int8_t, uint4>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 2 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int16_t>(
            tensor_size, launch_custom_device_memcpy_optimized<int16_t, uint4>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 4 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int32_t>(
            tensor_size, launch_custom_device_memcpy_optimized<int32_t, uint4>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 8 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int64_t>(
            tensor_size, launch_custom_device_memcpy_optimized<int64_t, uint4>,
            num_repeats, num_warmups);
    }
    std::cout << std::endl;

    // Conclusions:
    // 1. Copying data in units of 8 bytes or 16 bytes can improve the latency
    // of custom device memcpy.
}
```

이 CUDA 프로그램은 CUDA 12.0이 설치된 NVIDIA RTX 3090 GPU에서 컴파일하고 성능을 측정했다.

```shell
$ nvcc memcpy.cu -o memcpy -std=c++14
$ ./memcpy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
              NVIDIA GPU Device Info
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Device Name: NVIDIA GeForce RTX 3090
Memory Size: 23.6694 GB
Peak Bandwitdh: 936.096 GB/s

**************************************************
               CUDA Official Memcpy
**************************************************
==================================================
            Tensor Size: 262144 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.002 ms
Effective Bandwitdh: 217.362 GB/s
Percentage of Peak Bandwitdh: 23.220%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.002 ms
Effective Bandwitdh: 414.641 GB/s
Percentage of Peak Bandwitdh: 44.295%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 706.425 GB/s
Percentage of Peak Bandwitdh: 75.465%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.004 ms
Effective Bandwitdh: 1030.999 GB/s
Percentage of Peak Bandwitdh: 110.138%
==================================================
            Tensor Size: 2097152 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.004 ms
Effective Bandwitdh: 1059.638 GB/s
Percentage of Peak Bandwitdh: 113.198%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.011 ms
Effective Bandwitdh: 719.754 GB/s
Percentage of Peak Bandwitdh: 76.889%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.023 ms
Effective Bandwitdh: 675.261 GB/s
Percentage of Peak Bandwitdh: 72.136%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.043 ms
Effective Bandwitdh: 719.330 GB/s
Percentage of Peak Bandwitdh: 76.844%
==================================================
           Tensor Size: 134217728 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.321 ms
Effective Bandwitdh: 778.091 GB/s
Percentage of Peak Bandwitdh: 83.121%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.640 ms
Effective Bandwitdh: 781.539 GB/s
Percentage of Peak Bandwitdh: 83.489%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 1.275 ms
Effective Bandwitdh: 784.214 GB/s
Percentage of Peak Bandwitdh: 83.775%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 2.560 ms
Effective Bandwitdh: 781.282 GB/s
Percentage of Peak Bandwitdh: 83.462%

**************************************************
               Custom Device Memcpy
**************************************************
==================================================
            Tensor Size: 262144 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 183.399 GB/s
Percentage of Peak Bandwitdh: 19.592%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 354.443 GB/s
Percentage of Peak Bandwitdh: 37.864%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 681.196 GB/s
Percentage of Peak Bandwitdh: 72.770%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 1192.093 GB/s
Percentage of Peak Bandwitdh: 127.347%
==================================================
            Tensor Size: 2097152 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.010 ms
Effective Bandwitdh: 378.747 GB/s
Percentage of Peak Bandwitdh: 40.460%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.018 ms
Effective Bandwitdh: 445.593 GB/s
Percentage of Peak Bandwitdh: 47.601%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.024 ms
Effective Bandwitdh: 660.732 GB/s
Percentage of Peak Bandwitdh: 70.584%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.042 ms
Effective Bandwitdh: 737.140 GB/s
Percentage of Peak Bandwitdh: 78.746%
==================================================
           Tensor Size: 134217728 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.972 ms
Effective Bandwitdh: 257.207 GB/s
Percentage of Peak Bandwitdh: 27.477%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 1.076 ms
Effective Bandwitdh: 464.543 GB/s
Percentage of Peak Bandwitdh: 49.626%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 1.369 ms
Effective Bandwitdh: 730.586 GB/s
Percentage of Peak Bandwitdh: 78.046%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 2.536 ms
Effective Bandwitdh: 788.727 GB/s
Percentage of Peak Bandwitdh: 84.257%

**************************************************
     Custom Device Memcpy with Shared Memory
**************************************************
==================================================
            Tensor Size: 262144 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 175.995 GB/s
Percentage of Peak Bandwitdh: 18.801%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 328.853 GB/s
Percentage of Peak Bandwitdh: 35.130%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 653.481 GB/s
Percentage of Peak Bandwitdh: 69.809%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 1128.192 GB/s
Percentage of Peak Bandwitdh: 120.521%
==================================================
            Tensor Size: 2097152 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.011 ms
Effective Bandwitdh: 353.213 GB/s
Percentage of Peak Bandwitdh: 37.733%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.018 ms
Effective Bandwitdh: 433.488 GB/s
Percentage of Peak Bandwitdh: 46.308%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.024 ms
Effective Bandwitdh: 650.261 GB/s
Percentage of Peak Bandwitdh: 69.465%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.042 ms
Effective Bandwitdh: 737.864 GB/s
Percentage of Peak Bandwitdh: 78.824%
==================================================
           Tensor Size: 134217728 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 1.011 ms
Effective Bandwitdh: 247.181 GB/s
Percentage of Peak Bandwitdh: 26.406%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 1.113 ms
Effective Bandwitdh: 449.172 GB/s
Percentage of Peak Bandwitdh: 47.984%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 1.391 ms
Effective Bandwitdh: 718.748 GB/s
Percentage of Peak Bandwitdh: 76.781%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 2.546 ms
Effective Bandwitdh: 785.429 GB/s
Percentage of Peak Bandwitdh: 83.905%

**************************************************
   Custom Device Memcpy 4-Byte Copy Per Thread
**************************************************
==================================================
            Tensor Size: 262144 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.002 ms
Effective Bandwitdh: 238.419 GB/s
Percentage of Peak Bandwitdh: 25.469%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.002 ms
Effective Bandwitdh: 437.842 GB/s
Percentage of Peak Bandwitdh: 46.773%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 684.251 GB/s
Percentage of Peak Bandwitdh: 73.096%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.004 ms
Effective Bandwitdh: 1003.868 GB/s
Percentage of Peak Bandwitdh: 107.240%
==================================================
            Tensor Size: 2097152 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.004 ms
Effective Bandwitdh: 968.812 GB/s
Percentage of Peak Bandwitdh: 103.495%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.012 ms
Effective Bandwitdh: 675.168 GB/s
Percentage of Peak Bandwitdh: 72.126%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.024 ms
Effective Bandwitdh: 660.196 GB/s
Percentage of Peak Bandwitdh: 70.527%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.045 ms
Effective Bandwitdh: 690.443 GB/s
Percentage of Peak Bandwitdh: 73.758%
==================================================
           Tensor Size: 134217728 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.366 ms
Effective Bandwitdh: 682.529 GB/s
Percentage of Peak Bandwitdh: 72.912%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.722 ms
Effective Bandwitdh: 692.125 GB/s
Percentage of Peak Bandwitdh: 73.937%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 1.422 ms
Effective Bandwitdh: 703.431 GB/s
Percentage of Peak Bandwitdh: 75.145%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 2.824 ms
Effective Bandwitdh: 708.144 GB/s
Percentage of Peak Bandwitdh: 75.649%

**************************************************
   Custom Device Memcpy 8-Byte Copy Per Thread
**************************************************
==================================================
            Tensor Size: 262144 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.002 ms
Effective Bandwitdh: 238.792 GB/s
Percentage of Peak Bandwitdh: 25.509%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.002 ms
Effective Bandwitdh: 434.723 GB/s
Percentage of Peak Bandwitdh: 46.440%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 681.196 GB/s
Percentage of Peak Bandwitdh: 72.770%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.004 ms
Effective Bandwitdh: 1030.999 GB/s
Percentage of Peak Bandwitdh: 110.138%
==================================================
            Tensor Size: 2097152 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.004 ms
Effective Bandwitdh: 978.128 GB/s
Percentage of Peak Bandwitdh: 104.490%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.012 ms
Effective Bandwitdh: 677.416 GB/s
Percentage of Peak Bandwitdh: 72.366%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.022 ms
Effective Bandwitdh: 696.748 GB/s
Percentage of Peak Bandwitdh: 74.431%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.042 ms
Effective Bandwitdh: 738.924 GB/s
Percentage of Peak Bandwitdh: 78.937%
==================================================
           Tensor Size: 134217728 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.320 ms
Effective Bandwitdh: 781.750 GB/s
Percentage of Peak Bandwitdh: 83.512%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.636 ms
Effective Bandwitdh: 786.536 GB/s
Percentage of Peak Bandwitdh: 84.023%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 1.265 ms
Effective Bandwitdh: 790.547 GB/s
Percentage of Peak Bandwitdh: 84.451%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 2.530 ms
Effective Bandwitdh: 790.419 GB/s
Percentage of Peak Bandwitdh: 84.438%

**************************************************
   Custom Device Memcpy 16-Byte Copy Per Thread
**************************************************
==================================================
            Tensor Size: 262144 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.002 ms
Effective Bandwitdh: 216.744 GB/s
Percentage of Peak Bandwitdh: 23.154%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.002 ms
Effective Bandwitdh: 414.641 GB/s
Percentage of Peak Bandwitdh: 44.295%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.002 ms
Effective Bandwitdh: 829.282 GB/s
Percentage of Peak Bandwitdh: 88.589%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 1192.093 GB/s
Percentage of Peak Bandwitdh: 127.347%
==================================================
            Tensor Size: 2097152 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.003 ms
Effective Bandwitdh: 1128.192 GB/s
Percentage of Peak Bandwitdh: 120.521%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.010 ms
Effective Bandwitdh: 755.386 GB/s
Percentage of Peak Bandwitdh: 80.695%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 0.023 ms
Effective Bandwitdh: 687.333 GB/s
Percentage of Peak Bandwitdh: 73.425%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 0.043 ms
Effective Bandwitdh: 728.343 GB/s
Percentage of Peak Bandwitdh: 77.806%
==================================================
           Tensor Size: 134217728 Units
==================================================
--------------------------------------------------
                Unit Size: 1 Byte
--------------------------------------------------
Latency: 0.321 ms
Effective Bandwitdh: 779.006 GB/s
Percentage of Peak Bandwitdh: 83.219%
--------------------------------------------------
                Unit Size: 2 Byte
--------------------------------------------------
Latency: 0.639 ms
Effective Bandwitdh: 782.639 GB/s
Percentage of Peak Bandwitdh: 83.607%
--------------------------------------------------
                Unit Size: 4 Byte
--------------------------------------------------
Latency: 1.280 ms
Effective Bandwitdh: 781.520 GB/s
Percentage of Peak Bandwitdh: 83.487%
--------------------------------------------------
                Unit Size: 8 Byte
--------------------------------------------------
Latency: 2.552 ms
Effective Bandwitdh: 783.602 GB/s
Percentage of Peak Bandwitdh: 83.710%
```

## 결론

결과로부터 다음을 알 수 있다.

- 복사하는 데이터 단위가 많을수록 유효 메모리 대역폭이 높아진다.
- 데이터 단위가 클수록 유효 메모리 대역폭이 높아진다.
- 대부분의 경우 8바이트 또는 16바이트의 벡터화 단위로 데이터를 복사하면 커스텀 device memcpy의 유효 메모리 대역폭을 높일 수 있으며, 특히 데이터 단위가 작을 때 그렇다.
- shared memory를 사용해 커스텀 device memcpy의 유효 메모리 대역폭을 높이는 효과는 뚜렷하지 않다.

다만 이 사용 사례에서는 CUDA 공식 memcpy 함수를 그대로 사용할 수 있음에도, 커스텀 device memcpy 함수를 작성하고 개선하는 방법을 아는 것은 여전히 가치가 있다는 점에 유의하자. 더 현실적인 CUDA 애플리케이션에서는 복사할 데이터가 메모리상에서 연속적이지 않을 수 있고, 여러 소스에서 여러 목적지로 데이터를 복사해야 할 수도 있기 때문이다.

## 참고 자료
- CUDA Pro Tip: Increase Performance with Vectorized Memory Access(https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)

