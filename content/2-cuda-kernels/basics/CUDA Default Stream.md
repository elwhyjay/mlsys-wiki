> 블로그 출처: https://leimao.github.io/blog/CUDA-Default-Stream/ , Lei Mao의 글이며 저자의 전재 허가를 받았다.

# CUDA default stream

## 서론

CUDA default stream은 서로 다른 상황에서 다른 synchronization behavior를 가질 수 있다. 때로는 서로 다른 kernel에 CUDA stream을 할당할 때 실수를 하더라도, 프로그램이 올바르게 실행되도록 도와줄 수 있다.

이 블로그 글에서는 두 가지 CUDA default stream, 즉 default legacy stream과 default per-thread stream을 소개하고, 서로 다른 상황에서의 synchronization behavior를 논의하고자 한다.

## default stream과 non-default blocking stream

아래 예제에서는 `cudaStreamCreate`를 사용해 non-default blocking stream을 만들었다. 같은 non-default blocking CUDA stream에서 순서대로 실행되어야 하는 CUDA kernel series에 대해, 나는 실수로 그중 하나의 kernel에 default stream을 사용했다.

default stream이 default legacy stream이라면, legacy stream에서 작업(예: kernel launch 또는 `cudaStreamWaitEvent()`)을 실행할 때 legacy stream은 먼저 모든 blocking stream을 기다린 다음 작업을 legacy stream에 enqueue하고, 이후 모든 blocking stream이 legacy stream을 기다린다. 따라서 내가 실수했더라도 CUDA kernel은 여전히 순서대로 실행되며 application correctness는 영향을 받지 않는다.

default stream이 default per-thread stream이라면, 그것은 non-blocking이며 다른 CUDA stream과 synchronize하지 않는다. 따라서 내 실수는 application이 올바르게 실행되지 않게 만든다.

```c++
#include <cassert>
#include <iostream>
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

// CUDA kernel 함수: 배열의 각 element에 지정한 값을 더한다.
__global__ void add_val_in_place(int32_t* data, int32_t val, uint32_t n)
{
    // 현재 thread의 global index 계산
    uint32_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    // grid stride 계산(전체 thread 수)
    uint32_t const stride{blockDim.x * gridDim.x};
    // grid stride loop로 배열 element를 처리해 모든 element가 처리되도록 한다.
    for (uint32_t i{idx}; i < n; i += stride)
    {
        data[i] += val;
    }
}

// CUDA kernel을 launch하는 wrapper 함수
void launch_add_val_in_place(int32_t* data, int32_t val, uint32_t n,
                             cudaStream_t stream)
{
    // thread block당 thread 수 정의
    dim3 const threads_per_block{1024};
    // grid 안의 thread block 수 정의
    dim3 const blocks_per_grid{32};
    // 지정한 stream에서 kernel launch
    add_val_in_place<<<blocks_per_grid, threads_per_block, 0, stream>>>(data,
                                                                        val, n);
    // kernel launch 성공 여부 확인
    CHECK_LAST_CUDA_ERROR();
}

// 배열의 모든 element가 지정한 값과 같은지 확인
bool check_array_value(int32_t const* data, uint32_t n, int32_t val)
{
    for (uint32_t i{0}; i < n; ++i)
    {
        if (data[i] != val)
        {
            return false;
        }
    }
    return true;
}

int main()
{
    // 상수 정의: 배열 크기와 더할 값
    constexpr uint32_t const n{1000000};
    constexpr int32_t const val_1{1};
    constexpr int32_t const val_2{2};
    constexpr int32_t const val_3{3};
    
    // multi-stream application 생성
    cudaStream_t stream_1{0};
    cudaStream_t stream_2{0};
    // stream_1은 non-default blocking stream이다.
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream_1));

    // host에서 배열을 만들고 초기화
    std::vector<int32_t> vec(n, 0);
    int32_t* d_data{nullptr};
    // device에 memory 할당
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, n * sizeof(int32_t)));
    // host에서 device로 data copy
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, vec.data(), n * sizeof(int32_t),
                                cudaMemcpyHostToDevice));
    
    // 같은 CUDA stream에서 CUDA kernel series를 순서대로 실행
    launch_add_val_in_place(d_data, val_1, n, stream_1);
    // 두 번째 kernel launch는 원래 stream_1에서 실행되어야 했다.
    // 하지만 구현에 bug가 있어 kernel launch가 default stream stream_2에서 실행된다.
    launch_add_val_in_place(d_data, val_2, n, stream_2);
    launch_add_val_in_place(d_data, val_3, n, stream_1);

    // stream_1의 모든 작업이 완료될 때까지 대기
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_1));
    // 결과를 device에서 host로 copy
    CHECK_CUDA_ERROR(cudaMemcpy(vec.data(), d_data, n * sizeof(int32_t),
                                cudaMemcpyDeviceToHost));

    // application correctness 확인
    // default stream stream_2가 legacy default stream이라면 결과는 여전히 올바르다.
    assert(check_array_value(vec.data(), n, val_1 + val_2 + val_3));

    // resource 정리
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_1));
}
```

구현에서 세 kernel이 같은 CUDA stream에서 실행되지 않게 만들었지만, 결과는 여전히 올바르다.

```shell
$ nvcc add.cu -o add -std=c++14
$ ./add
```

이는 `--default-stream`의 기본값이 `legacy`이기 때문에 다음 명령을 실행하는 것과 같다.

```shell
$ nvcc add.cu -o add -std=c++14 --default-stream=legacy
$ ./add
```

사용 상황에 따라 이런 오류는 때때로 application performance에 영향을 줄 수 있다. 보통 Nsight Systems(https://leimao.github.io/blog/Docker-Nsight-Systems/) 같은 CUDA performance analysis software로 식별할 수 있다.

하지만 default stream이 `per-thread`가 되면 kernel launch가 더 이상 순서대로 발행되지 않으므로 결과가 더 이상 올바르지 않다.

```shell
$ nvcc add.cu -o add -std=c++14 --default-stream=per-thread
$ ./add
add: add.cu:98: int main(): Assertion `check_array_value(vec.data(), n, val_1 + val_2 + val_3)' failed.
Aborted (core dumped)
```

## default stream과 non-default non-blocking stream

어떤 application에서는 `cudaStreamCreateWithFlags`를 사용해 non-default stream을 만들 수 있으며, 만들어진 non-default stream은 non-blocking이 된다. 이 경우 default stream은 default legacy stream이더라도 non-default non-blocking stream과 synchronize할 수 없다. 따라서 non-default stream이 legacy stream이든 per-thread stream이든 내 실수는 application을 올바르게 실행되지 않게 만든다.

```c++
#include <cassert>
#include <iostream>
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

__global__ void add_val_in_place(int32_t* data, int32_t val, uint32_t n)
{
    uint32_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    uint32_t const stride{blockDim.x * gridDim.x};
    for (uint32_t i{idx}; i < n; i += stride)
    {
        data[i] += val;
    }
}

void launch_add_val_in_place(int32_t* data, int32_t val, uint32_t n,
                             cudaStream_t stream)
{
    dim3 const threads_per_block{1024};
    dim3 const blocks_per_grid{32};
    add_val_in_place<<<blocks_per_grid, threads_per_block, 0, stream>>>(data,
                                                                        val, n);
    CHECK_LAST_CUDA_ERROR();
}

bool check_array_value(int32_t const* data, uint32_t n, int32_t val)
{
    for (uint32_t i{0}; i < n; ++i)
    {
        if (data[i] != val)
        {
            return false;
        }
    }
    return true;
}
int main()
{
    // 상수 정의: 배열 크기와 더할 세 값
    constexpr uint32_t const n{1000000};
    constexpr int32_t const val_1{1};
    constexpr int32_t const val_2{2};
    constexpr int32_t const val_3{3};
    
    // multi-stream application 생성
    cudaStream_t stream_1{0};  // non-default stream
    cudaStream_t stream_2{0};  // default stream(값 0은 default stream을 의미)
    
    // stream_1은 non-default non-blocking stream이다.
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream_1, cudaStreamNonBlocking));

    // host 쪽에서 배열을 만들고 초기화하며, 모든 element를 0으로 초기화한다.
    std::vector<int32_t> vec(n, 0);
    
    // device 쪽에 memory 할당
    int32_t* d_data{nullptr};
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, n * sizeof(int32_t)));
    
    // host data를 device로 copy
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, vec.data(), n * sizeof(int32_t),
                                cudaMemcpyHostToDevice));
    
    // 같은 CUDA stream에서 CUDA kernel series를 순서대로 실행
    // 첫 번째 kernel: stream_1에서 실행하며 배열의 각 element에 val_1을 더한다.
    launch_add_val_in_place(d_data, val_1, n, stream_1);
    
    // 두 번째 kernel launch는 원래 stream_1에서 실행되어야 했다.
    // 그러나 구현에 bug가 있어 kernel launch가 default stream stream_2에서 실행된다.
    // 여기서는 일부러 stream_2를 사용해 default stream의 synchronization behavior를 보여준다.
    launch_add_val_in_place(d_data, val_2, n, stream_2);
    
    // 세 번째 kernel: stream_1에서 실행하며 배열의 각 element에 val_3을 더한다.
    launch_add_val_in_place(d_data, val_3, n, stream_1);

    // stream_1의 모든 작업이 완료될 때까지 대기
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_1));
    
    // 결과를 device에서 host로 copy
    CHECK_CUDA_ERROR(cudaMemcpy(vec.data(), d_data, n * sizeof(int32_t),
                                cudaMemcpyDeviceToHost));

    // application correctness 확인
    // default stream stream_2가 legacy default stream이라면 결과는 여전히 올바르다.
    // legacy default stream이 다른 stream과 synchronize해 kernel이 순서대로 실행되도록 보장하기 때문이다.
    // default stream이 per-thread stream이라면 execution order 오류가 발생할 수 있다.
    assert(check_array_value(vec.data(), n, val_1 + val_2 + val_3));

    // resource 정리: device memory 해제와 stream destroy
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_1));
}
```

```shell
$ nvcc add.cu -o add -std=c++14 --default-stream=legacy
$ ./add
add: add.cu:98: int main(): Assertion `check_array_value(vec.data(), n, val_1 + val_2 + val_3)' failed.
Aborted (core dumped)
```

```shell
$ nvcc add.cu -o add -std=c++14 --default-stream=per-thread
$ ./add
add: add.cu:98: int main(): Assertion `check_array_value(vec.data(), n, val_1 + val_2 + val_3)' failed.
Aborted (core dumped)
```

## 참고 자료

- Stream synchronization behavior(https://docs.nvidia.com/cuda/archive/11.7.1/cuda-runtime-api/stream-sync-behavior.html)

