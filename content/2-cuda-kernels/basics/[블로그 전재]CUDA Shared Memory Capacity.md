> 블로그 출처: https://leimao.github.io/blog/CUDA-Shared-Memory-Capacity/ , Lei Mao의 글이며 저자의 전재 허가를 받았다.

# CUDA 공유 메모리 용량

## 소개

CUDA 공유 메모리는 CUDA kernel 구현과 최적화에서 매우 강력한 기능이다. CUDA 공유 메모리는 칩 위에 있기 때문에, 그 메모리 대역폭은 칩 밖에 있는 전역 메모리보다 훨씬 크다. 따라서 공유 메모리에 메모리 접근을 캐싱하여 CUDA kernel을 최적화하면, 특히 메모리 제한을 받는 연산에서 특정 작업의 성능을 크게 높일 수 있다.

하지만 CUDA 공유 메모리는 thread block마다 크기 제한이 있으며, 기본값은 48 KB이다. 때로는 구현에서 이보다 조금 더 많은 공유 메모리를 사용하고 싶을 수 있다. 이 블로그 글에서는 정적 공유 메모리와 동적 공유 메모리를 할당하는 방법, 그리고 48 KB를 넘는 동적 공유 메모리를 요청하는 방법을 논의하고자 한다.

## stencil kernel

CUDA 공유 메모리 할당을 보여주기 위해 stencil kernel을 구현했다. 이 stencil은 수학적으로는 weight가 정확히 1이고 valid padding을 사용하는 convolution의 특수한 경우와 거의 같다.

예를 들어, 일차원 배열 ${1, 1, 1, 1, 1, 1, 1}$과 반지름이 2인 stencil kernel이 주어지면, 출력 일차원 배열 ${1, 1, 5, 5, 5, 1, 1}$을 얻게 된다.

stencil 연산은 입력 tensor에서 많은 중복 메모리 read를 만들어 내므로 메모리 제한을 받는 연산이다. 메모리 read가 캐싱되지 않고 프로그램이 전역 메모리에서 read한다면 성능은 나빠질 것이다. 따라서 우리는 온칩 공유 메모리를 이용해 메모리 read를 캐싱하고 성능을 높인다.

### 정적 공유 메모리

이 구현에서는 정적 공유 메모리를 할당하며, 그 크기는 컴파일 시점에 알려져 있어야 한다. 이 구현은 임의의 "valid" 배열 크기, 반지름, CUDA thread block 크기도 지원한다. 또한 kernel을 구현할 때 반지름이 CUDA thread block 크기보다 크고 "valid" 배열 크기가 CUDA thread block 크기로 나누어떨어지지 않는 경우를 특별히 주의해야 한다. 이런 경우를 올바르게 구현하는 것은 쉽지 않기 때문이다.

```c++
#include <cassert>
#include <iostream>
#include <vector>

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

template <int BLOCK_SIZE = 1024, int RADIUS = 5>
__global__ void stencil_1d_kernel(int const* d_in, int* d_out,
                                  int valid_array_size)
{
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];

    // This has to be int because we will use negative indices.
    int const gindex{static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x)};
    int const lindex{static_cast<int>(threadIdx.x) + RADIUS};

    int const valid_block_size{
        min(BLOCK_SIZE,
            valid_array_size - static_cast<int>(blockIdx.x * blockDim.x))};

    // Read input elements into shared memory
    if (gindex < valid_array_size)
    {
        temp[lindex] = d_in[gindex];
        if (RADIUS <= valid_block_size)
        {
            if (threadIdx.x < RADIUS)
            {
                temp[lindex - RADIUS] = d_in[gindex - RADIUS];
                temp[lindex + valid_block_size] =
                    d_in[gindex + valid_block_size];
            }
        }
        else
        {
            for (int i{0}; i < RADIUS; i += valid_block_size)
            {
                // Some threads might have to do one more job than other
                // threads.
                if (lindex - RADIUS + i < RADIUS)
                {
                    temp[lindex - RADIUS + i] = d_in[gindex - RADIUS + i];
                    temp[lindex + valid_block_size + i] =
                        d_in[gindex + valid_block_size + i];
                }
            }
        }
    }
    // Synchronize (ensure all the data is available)
    __syncthreads();

    if (gindex >= valid_array_size)
    {
        return;
    }

    // Apply the stencil
    int result{0};
    for (int offset{-RADIUS}; offset <= RADIUS; offset++)
    {
        result += temp[lindex + offset];
    }

    // Store the result
    d_out[gindex] = result;
}

void stencil_1d_cpu(int const* h_in, int* h_out, int radius,
                    int valid_array_size)
{
    for (int i{0}; i < valid_array_size; ++i)
    {
        int result{0};
        for (int offset{-radius}; offset <= radius; offset++)
        {
            result += h_in[i + offset];
        }
        h_out[i] = result;
    }
}

int main(int argc, char** argv)
{
    constexpr int const valid_array_size{1024 * 100 + 1};
    constexpr int const block_size{1024};
    constexpr int const grid_size{(valid_array_size + block_size - 1) /
                                  block_size};
    constexpr int const radius{1025};

    int const array_size{valid_array_size + 2 * radius};
    std::vector<int> const h_in(array_size, 1);
    std::vector<int> h_out{h_in};
    std::vector<int> h_out_reference{h_in};

    stencil_1d_cpu(h_in.data() + radius, h_out_reference.data() + radius,
                   radius, valid_array_size);

    int* d_in;
    int* d_out;

    CHECK_CUDA_ERROR(cudaMalloc(&d_in, array_size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, array_size * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in.data(), array_size * sizeof(int),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_out, h_out.data(), array_size * sizeof(int),
                                cudaMemcpyHostToDevice));

    stencil_1d_kernel<block_size, radius><<<grid_size, block_size>>>(
        d_in + radius, d_out + radius, valid_array_size);
    CHECK_LAST_CUDA_ERROR();

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_out.data(), d_out, array_size * sizeof(int),
                                cudaMemcpyDeviceToHost));

    for (int i{0}; i < h_out_reference.size(); ++i)
    {
        assert(h_out[i] == h_out_reference[i]);
    }

    CHECK_CUDA_ERROR(cudaFree(d_in));
    CHECK_CUDA_ERROR(cudaFree(d_out));
}
```

```shell
$ nvcc stencil_static_shared_memory.cu -o stencil_static_shared_memory
$ ./stencil_static_shared_memory
```

`radius`를 `1025`에서 `6000` 같은 더 큰 값으로 늘리면 다음과 같은 컴파일 오류가 발생한다.

```shell
$ nvcc stencil_static_shared_memory.cu -o stencil_static_shared_memory
ptxas error   : Entry function '_Z17stencil_1d_kernelILi1024ELi6000EEvPKiPii' uses too much shared data (0xcb80 bytes, 0xc000 max)
```

이는 사용자가 최대 48 KB의 CUDA 정적 공유 메모리만 할당할 수 있기 때문이다. 우리 사용 사례에서 `BLOCK_SIZE + 2 * RADIUS = 1024 + 2 × 6000 = 13024`이고, int 하나의 크기는 4바이트이므로 필요한 공유 메모리는 `13024 × 4/1024 = 50.875 KB`이다. 이는 가질 수 있는 최대 정적 공유 메모리보다 크다.

## 동적 공유 메모리

48 KB보다 큰 공유 메모리를 사용하려면 동적 공유 메모리를 사용해야 하며, 이는 아키텍처에 따라 달라진다. 구체적으로, CUDA launch의 `<<<...>>>` 세 번째 인수에 요청하려는 동적 공유 메모리 크기를 지정하는 것 외에도 CUDA Runtime API `cudaFuncSetAttribute`를 호출해야 한다. 일부 아키텍처에서는 런타임에 실패할 수 있으므로 반환값을 항상 확인해야 한다.

플랫폼 GPU는 NVIDIA RTX 2080TI이다. CUDA C 프로그래밍 가이드(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x)에 따르면, compute capability 7.x 장치는 Turing 아키텍처에서 단일 thread block이 최대 64 KB의 공유 메모리를 동적으로 할당하도록 허용한다. 따라서 NVIDIA RTX 2080TI에서 반지름이 `6000`인 stencil 프로그램을 실행할 수 있다.

동적 공유 메모리를 사용하는 이 구현은 정적 공유 메모리를 사용하는 구현과 거의 동일하다.

```c++
#include <cassert>
#include <iostream>
#include <vector>

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

template <int BLOCK_SIZE = 1024, int RADIUS = 5>
__global__ void stencil_1d_kernel(int const* d_in, int* d_out,
                                  int valid_array_size)
{
    extern __shared__ int temp[];

    // This has to be int because we will use negative indices.
    int const gindex{static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x)};
    int const lindex{static_cast<int>(threadIdx.x) + RADIUS};

    int const valid_block_size{
        min(BLOCK_SIZE,
            valid_array_size - static_cast<int>(blockIdx.x * blockDim.x))};

    // Read input elements into shared memory
    if (gindex < valid_array_size)
    {
        temp[lindex] = d_in[gindex];
        if (RADIUS <= valid_block_size)
        {
            if (threadIdx.x < RADIUS)
            {
                temp[lindex - RADIUS] = d_in[gindex - RADIUS];
                temp[lindex + valid_block_size] =
                    d_in[gindex + valid_block_size];
            }
        }
        else
        {
            for (int i{0}; i < RADIUS; i += valid_block_size)
            {
                // Some threads might have to do one more job than other
                // threads.
                if (lindex - RADIUS + i < RADIUS)
                {
                    temp[lindex - RADIUS + i] = d_in[gindex - RADIUS + i];
                    temp[lindex + valid_block_size + i] =
                        d_in[gindex + valid_block_size + i];
                }
            }
        }
    }
    // Synchronize (ensure all the data is available)
    __syncthreads();

    if (gindex >= valid_array_size)
    {
        return;
    }

    // Apply the stencil
    int result{0};
    for (int offset{-RADIUS}; offset <= RADIUS; offset++)
    {
        result += temp[lindex + offset];
    }

    // Store the result
    d_out[gindex] = result;
}

void stencil_1d_cpu(int const* h_in, int* h_out, int radius,
                    int valid_array_size)
{
    for (int i{0}; i < valid_array_size; ++i)
    {
        int result{0};
        for (int offset{-radius}; offset <= radius; offset++)
        {
            result += h_in[i + offset];
        }
        h_out[i] = result;
    }
}

int main(int argc, char** argv)
{
    constexpr int const valid_array_size{1024 * 100 + 1};
    constexpr int const block_size{1024};
    constexpr int const grid_size{(valid_array_size + block_size - 1) /
                                  block_size};
    constexpr int const radius{6000};

    int const array_size{valid_array_size + 2 * radius};
    std::vector<int> const h_in(array_size, 1);
    std::vector<int> h_out{h_in};
    std::vector<int> h_out_reference{h_in};

    stencil_1d_cpu(h_in.data() + radius, h_out_reference.data() + radius,
                   radius, valid_array_size);

    int* d_in;
    int* d_out;

    CHECK_CUDA_ERROR(cudaMalloc(&d_in, array_size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, array_size * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in.data(), array_size * sizeof(int),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_out, h_out.data(), array_size * sizeof(int),
                                cudaMemcpyHostToDevice));

    int const shared_memory_bytes{(block_size + radius * 2) * sizeof(int)};
    CHECK_CUDA_ERROR(cudaFuncSetAttribute(
        stencil_1d_kernel<block_size, radius>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_bytes));
    stencil_1d_kernel<block_size, radius>
        <<<grid_size, block_size, shared_memory_bytes>>>(
            d_in + radius, d_out + radius, valid_array_size);
    CHECK_LAST_CUDA_ERROR();

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_out.data(), d_out, array_size * sizeof(int),
                                cudaMemcpyDeviceToHost));

    for (int i{0}; i < h_out_reference.size(); ++i)
    {
        assert(h_out[i] == h_out_reference[i]);
    }

    CHECK_CUDA_ERROR(cudaFree(d_in));
    CHECK_CUDA_ERROR(cudaFree(d_out));
}
```

```shell
$ nvcc stencil_dynamic_shared_memory.cu -o stencil_dynamic_shared_memory --gpu-architecture=compute_75 --gpu-code=sm_75
$ ./stencil_dynamic_shared_memory
```

## 결론

대용량 공유 메모리를 동적 공유 메모리로만 할당할 수 있는 이유는 모든 GPU 아키텍처가 48 KB보다 큰 특정 크기의 공유 메모리를 지원하는 것은 아니기 때문이다. 48 KB보다 큰 정적 공유 메모리를 허용하면 CUDA 프로그램은 컴파일은 통과하지만 특정 GPU 아키텍처에서는 실행에 실패하게 되며, 이는 바람직하지 않다. 따라서 48 KB보다 큰 공유 메모리를 사용하려면 런타임에 동적 공유 메모리로 요청해야 한다. GPU 아키텍처가 특정 크기의 공유 메모리를 지원하지 않으면 CUDA 런타임 오류가 반환된다.

## 참고 문헌

- Shared Memory - CUDA C Programming Guide(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x)
- CUDA Shared Memory(https://leimao.github.io/downloads/blog/2022-07-04-CUDA-Shared-Memory-Capacity/02-CUDA-Shared-Memory.pdf)
