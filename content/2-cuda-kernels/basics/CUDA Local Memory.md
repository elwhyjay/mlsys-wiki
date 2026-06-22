> 블로그 출처: https://leimao.github.io/blog/CUDA-Local-Memory/ 이 글은 Lei Mao의 글이며, 저자의 전재 허가를 받았다. 이후 Lei Mao의 CUDA 관련 Blog를 몇 편 더 전재할 예정이고, 이는 하나의 완결된 칼럼이다. Blog는 비교적 이른 시기의 CUDA 아키텍처부터 현재 최신 CUDA 아키텍처까지 다루며, 실용적인 엔지니어링 기법, 저수준 명령어 분석, Cutlass 분석 등 여러 주제를 포함한다.

# CUDA Local Memory

## 소개

CUDA 프로그래밍에서 local memory는 실행 중인 thread의 private storage 공간이며, 해당 thread 밖에서는 보이지 않는다. Local memory 공간은 device memory에 있으므로 local memory access는 global memory access와 같은 높은 latency와 낮은 bandwidth 특성을 가지며, 같은 memory coalescing 요구 사항의 제약을 받는다.

> **주석:** local memory는 이름은 "local"이지만 실제로는 device memory(VRAM)에 있으며, 혼동하기 쉬운 개념이다. 실제 access 속도는 global memory만큼 느리다.

`__device__`, `__shared__`, `__constant__` memory space specifier로 선언되지 않은 automatic variable은 compiler에 의해 register 또는 local memory에 배치될 수 있다. 다음 조건 중 하나를 만족하면 변수는 local memory에 배치될 가능성이 높다.

- compiler가 constant index로 access한다고 판단할 수 없는 array
- register 공간을 너무 많이 차지하는 큰 struct 또는 array
- kernel이 사용하는 register 수가 사용 가능한 수를 넘을 때의 모든 변수(이를 register spilling이라고도 한다)

> **주석:** 두 번째와 세 번째 항목은 이해하기 쉽지만 첫 번째 항목은 조금 복잡하다. 아주 작은 array라도 index가 compile-time constant가 아니면 local memory에 배치될 수 있음을 뜻하며, 대부분의 경우 더 나은 performance을 위해 이런 작은 array가 register에 배치되기를 원한다.

두 번째와 세 번째 항목은 매우 직관적으로 이해할 수 있다. 그러나 첫 번째 항목은 다소 복잡하다. 아주 작은 array라도 compiler가 constant index로 access된다고 판단할 수 없으면 register가 아니라 local memory에 배치될 수 있음을 암시하기 때문이다. 대부분의 경우 우리는 더 나은 performance을 위해 이런 작은 array가 register에 배치되기를 원한다.

이 블로그 글에서는，compiler가 array를 register가 아니라 local memory에 배치하기로 결정하는 예를 보이고, 작은 array가 local memory에 배치되는 일을 피하기 위해 사용자가 따를 수 있는 일반 규칙을 논의한다.

## CUDA Local Memory예제

다음 예제에서는 고정된 `window` size에 대해 입력 array의 moving average를 calculate하는 두 CUDA kernel을 만들었다. 두 kernel은 모두 size가 compile time에 알려진 local array `window`를 선언한다. 두 kernel의 implementation은 거의 완전히 같지만, 첫 번째 kernel은 `window` array에 direct index로 access하고 두 번째 kernel은 덜 직관적으로 보이는 index를 사용한다.

> **주석:** 이 예제는 compiler가 array index의 복잡도에 따라 변수의 저장 위치를 어떻게 결정하는지 잘 보여준다.

```c++
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

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

// first kernel ：uses simple indexing，arraywill be placed inregister in 
template <int WindowSize>
__global__ void running_mean_register_array(float const* input, float* output,
                                            int n)
{
    float window[WindowSize];  // this array will be placed in registers
    int const thread_idx{
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    int const stride{static_cast<int>(blockDim.x * gridDim.x)};
    for (int i{thread_idx}; i < n; i += stride)
    {
        // read data into the window
        for (int j{0}; j < WindowSize; ++j)
        {
            int const idx{i - WindowSize / 2 + j};
            window[j] = (idx < 0 || idx >= n) ? 0 : input[idx];
        }
        // calculate average from the window
        float sum{0};
        for (int j{0}; j < WindowSize; ++j)
        {
            // simple constant index is used here j，compilercan handle it easily
            sum += window[j];
        }
        float const mean{sum / WindowSize};
        // write average to output
        output[i] = mean;
    }
}

// second kernel ：uses complex indexing，arraywill be placed inlocal memory in 
template <int WindowSize>
__global__ void running_mean_local_memory_array(float const* input,
                                                float* output, int n)
{
    float window[WindowSize];  // this array will be placed in local memory
    int const thread_idx{
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
    int const stride{static_cast<int>(blockDim.x * gridDim.x)};
    for (int i{thread_idx}; i < n; i += stride)
    {
        // read data into the window
        for (int j{0}; j < WindowSize; ++j)
        {
            int const idx{i - WindowSize / 2 + j};
            window[j] = (idx < 0 || idx >= n) ? 0 : input[idx];
        }
        // calculate average from the window
        float sum{0};
        for (int j{0}; j < WindowSize; ++j)
        {
            // the index accessing the window array cannot be resolved by the compiler at compile time，
            // even though this index does not affect kernel correctness。
            // as a result, the compiler places the window array in local memory instead of the register file。
            int const idx{(j + n) % WindowSize};  // complex index expression
            sum += window[idx];
        }
        float const mean{sum / WindowSize};
        // write average to output
        output[i] = mean;
    }
}

template <int WindowSize>
cudaError_t launch_running_mean_register_array(float const* input,
                                               float* output, int n,
                                               cudaStream_t stream)
{
    dim3 const block_size{256, 1, 1};
    dim3 const grid_size{(n + block_size.x - 1) / block_size.x, 1, 1};
    running_mean_register_array<WindowSize>
        <<<grid_size, block_size, 0, stream>>>(input, output, n);
    return cudaGetLastError();
}

template <int WindowSize>
cudaError_t launch_running_mean_local_memory_array(float const* input,
                                                   float* output, int n,
                                                   cudaStream_t stream)
{
    dim3 const block_size{256, 1, 1};
    dim3 const grid_size{(n + block_size.x - 1) / block_size.x, 1, 1};
    running_mean_local_memory_array<WindowSize>
        <<<grid_size, block_size, 0, stream>>>(input, output, n);
    return cudaGetLastError();
}

// verify correctness of the kernel for the given window size and launcher function
template <int WindowSize>
void verify_running_mean(int n, cudaError_t (*launch_func)(float const*, float*,
                                                           int, cudaStream_t))
{
    std::vector<float> h_input_vec(n, 0.f);
    std::vector<float> h_output_vec(n, 1.f);
    std::vector<float> h_output_vec_ref(n, 2.f);
    // fill input vector with values
    for (int i{0}; i < n; ++i)
    {
        h_input_vec[i] = static_cast<float>(i);
    }
    // calculate reference output vector
    for (int i{0}; i < n; ++i)
    {
        float sum{0};
        for (int j{0}; j < WindowSize; ++j)
        {
            int const idx{i - WindowSize / 2 + j};
            float const val{(idx < 0 || idx >= n) ? 0 : h_input_vec[idx]};
            sum += val;
        }
        h_output_vec_ref[i] = sum / WindowSize;
    }
    // allocate device memory
    float* d_input;
    float* d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * sizeof(float)));
    // copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input_vec.data(), n * sizeof(float),
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_output, h_output_vec.data(),
                                n * sizeof(float), cudaMemcpyHostToDevice));
    // launch kernel 
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_CUDA_ERROR(launch_func(d_input, d_output, n, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    // copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_vec.data(), d_output,
                                n * sizeof(float), cudaMemcpyDeviceToHost));
    // check result
    for (int i{0}; i < n; ++i)
    {
        if (h_output_vec.at(i) != h_output_vec_ref.at(i))
        {
            std::cerr << "Mismatch at index " << i << ": " << h_output_vec.at(i)
                      << " != " << h_output_vec_ref.at(i) << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    // free device memory
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

int main()
{
    // try different window sizes from small to large
    constexpr int WindowSize{32};
    int const n{8192};
    verify_running_mean<WindowSize>(
        n, launch_running_mean_register_array<WindowSize>);
    verify_running_mean<WindowSize>(
        n, launch_running_mean_local_memory_array<WindowSize>);
    return 0;
}
```

예제를 빌드하고 실행하려면 다음 명령을 실행한다. 예제를 실행할 때 오류 메시지는 발생하지 않아야 한다.

```shell
$ nvcc cuda_local_memory.cu -o cuda_local_memory
$ ./cuda_local_memory
```

local array `window`가 register에 배치되었는지 local memory에 배치되었는지 확인하려면 코드를 PTX로 컴파일하고 PTX 코드를 검사하면 된다.

> **주석:** PTX(Parallel Thread Execution)는 NVIDIA GPU의 intermediate representation이며 assembly language와 비슷하다. PTX 코드를 보면 compiler가 CUDA 코드를 어떻게 처리하는지 알 수 있다.

코드를 PTX로 컴파일하려면 다음 명령을 실행한다.

```shell
$ nvcc --ptx cuda_local_memory.cu -o cuda_local_memory.ptx
```

두 kernel의 PTX 코드에서 첫 번째 kernel은 `.local` instruction으로 아무것도 선언하지 않지만, 두 번째 kernel에는 `.local` instruction으로 선언된 local array `__local_depot1`가 있음을 확인할 수 있다. 이는 첫 번째 kernel은 array `window`를 register에 배치했고, 두 번째 kernel은 array `window`를 local memory에 배치했음을 확인해 준다. 두 kernel에서 선언된 local array의 size는 같지만, compiler가 두 번째 kernel에서 사용하는 array를 constant index로 access한다고 판단할 수 없기 때문에 local memory에 배치된 것이다.

> **주석:** `.local` instruction은 변수가 local memory에 저장됨을 나타내며, 이 instruction이 없다는 것은 변수가 register에 저장됨을 의미한다. 이는 변수 저장 위치를 판단하는 핵심 지표다.

```shell
...

	// .globl	_Z27running_mean_register_arrayILi32EEvPKfPfi

.visible .entry _Z27running_mean_register_arrayILi32EEvPKfPfi(
	.param .u64 _Z27running_mean_register_arrayILi32EEvPKfPfi_param_0,
	.param .u64 _Z27running_mean_register_arrayILi32EEvPKfPfi_param_1,
	.param .u32 _Z27running_mean_register_arrayILi32EEvPKfPfi_param_2
)
{
	.reg .pred 	%p<99>;
	.reg .f32 	%f<162>;
	.reg .b32 	%r<41>;
	.reg .b64 	%rd<15>;
...
}
	// .globl	_Z31running_mean_local_memory_arrayILi32EEvPKfPfi
.visible .entry _Z31running_mean_local_memory_arrayILi32EEvPKfPfi(
	.param .u64 _Z31running_mean_local_memory_arrayILi32EEvPKfPfi_param_0,
	.param .u64 _Z31running_mean_local_memory_arrayILi32EEvPKfPfi_param_1,
	.param .u32 _Z31running_mean_local_memory_arrayILi32EEvPKfPfi_param_2
)
{
	.local .align 16 .b8 	__local_depot1[128];  // local memory array is declared here
	.reg .b64 	%SP;
	.reg .b64 	%SPL;
	.reg .pred 	%p<99>;
	.reg .f32 	%f<194>;
	.reg .b32 	%r<232>;
	.reg .b64 	%rd<82>;
...
}
```

## 결론

작은 array가 local memory에 배치되는 일을 피하려면 compiler가 constant인지 판단할 수 없는 매우 복잡한 index 사용을 피해야 한다. 문제는 compiler가 index를 constant로 판단할 수 있는지 우리가 어떻게 알 수 있느냐이다.

> **주석:** 이는 CUDA 최적화에서 중요한 개념이다. compiler의 동작을 이해하면 더 효율적인 코드를 작성하는 데 도움이 된다.

사실 register는 실제로 index로 access할 수 없으며, register에 배치된 array도 마찬가지다. 작은 array가 register에 배치된다면, 그 작은 array의 equivalent constant index 형태도 프로그램에 작성할 수 있다.

예를 들어 첫 번째 kernel `running_mean_register_array`의 다음 implementation은

```c++
constexpr int WindowSize{4};
float window[WindowSize];
float sum{0};
for (int j{0}; j < WindowSize; ++j)
{
    sum += window[j];
}
```

array `window` 선언이 필요 없는 것처럼 다음 equivalent form을 가진다.

```c++
float window0, window1, window2, window3;
float sum{0};
sum += window0;
sum += window1;
sum += window2;
sum += window3;
```

> **주석:** 이 예제는 compiler가 simple index array를 register에 배치할 수 있는 이유를 잘 보여준다. array를 개별 register variable로 "unroll"할 수 있기 때문이다.

반면 두 번째 kernel `running_mean_local_memory_array`의 다음 implementation은

```c++
constexpr int WindowSize{4};
float window[WindowSize];
float sum{0};
for (int j{0}; j < WindowSize; ++j)
{
    int const idx{(j + n) % WindowSize};
    sum += window[idx];
}
```

equivalent form이 없다. `n`의 값은 runtime에만 알 수 있으므로 array `window` 선언이 필요한 것처럼 보인다.

수학적으로는 다음 형태와도 equivalent이지만, compiler가 이를 알아내는 것은 매우 어려운 작업이다.

```c++
float window0, window1, window2, window3;
float sum{0};
sum += window0;
sum += window1;
sum += window2;
sum += window3;
```

> **주석:** 수학적으로 equivalent이더라도 compiler는 이런 복잡한 최적화를 수행할 수 없다. index expression의 equivalence를 증명하려면 복잡한 수학적 추론이 필요하기 때문이다.

실제로 이는 CUDA TensorCore MMA PTX에도 해당한다. TensorCore MMA는 최적 performance을 얻기 위해 register에서 data를 읽어야 하기 때문이다. 예를 들어 CUTLASS의 `SM80_16x8x8_F16F16F16F16_TN` MMA implementation은 다음과 같으며, buffer가 array로 선언되어 있더라도 MMA PTX는 register에만 access한다.

> **주석:** TensorCore는 NVIDIA GPU에서 matrix multiplication을 가속하기 위한 전용 hardware unit이며, 최적 performance을 얻으려면 data가 register에 있어야 한다.

```c++
// MMA 16x8x8 TN
struct SM80_16x8x8_F16F16F16F16_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[1];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
      "{%0, %1},"
      "{%2, %3},"
      "{%4},"
      "{%5, %6};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),
         "r"(b0),
         "r"(c0),  "r"(c1));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM80_16x8x8_F16F16F16F16_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};
```


이 CUTLASS 코드는 Ampere 아키텍처(SM80) TensorCore의 핵심 특성을 보여준다.

**구조체 이름 해석:**
- `SM80_16x8x8_F16F16F16F16_TN`: SM80 아키텍처, M=16/N=8/K=8 차원, F16 data type, A는 transposed이고 B는 normal layout이다.

**register 할당 원리:**
```c++
using ARegisters = uint32_t[2];  // A matrix fragment：2 32-bit registers
using BRegisters = uint32_t[1];  // B matrix fragment：1 32-bit register  
using CRegisters = uint32_t[2];  // C matrix fragment：2 32-bit registers
using DRegisters = uint32_t[2];  // D matrix fragment：2 32-bit registers
```

**핵심 개념:** 여기서 register 수는 전체 matrix을 저장하는 것이 아니라 각 thread의 **matrix fragment**를 저장한다.

**data 분포 메커니즘:**
- 16x8 matrix은 32개 thread(하나의 warp)에 분산된다.
- 각 thread는 2개의 32-bit register로 4개의 F16 값을 저장한다(각 32-bit register는 F16 2개를 pack한다).
- 합계: 32 thread x 2 register x F16 2개 = F16 값 128개 = 16x8 matrix이다.

**핵심 PTX instruction:**
```c++
asm volatile(
  "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
  "{%0, %1},"      // output: d0, d1
  "{%2, %3},"      // Amatrix：a0, a1  
  "{%4},"          // Bmatrix：b0
  "{%5, %6};\n"    // Cmatrix：c0, c1
  : "=r"(d0), "=r"(d1) : "r"(a0), "r"(a1), "r"(b0), "r"(c0), "r"(c1));
```

**설계 장점:**
1. **hardware specialization**: TensorCore는 register에서만 data를 읽을 수 있어 최고 performance을 보장한다.
2. **parallel efficiency**: warp-level data distribution으로 32개 thread를 충분히 활용한다.
3. **register optimization**: 각 thread가 소수의 register만 필요로 하므로 register pressure를 피한다.
4. **memory bandwidth**: 단일 thread가 많은 data를 저장하는 일을 피해서 memory efficiency를 높인다.


## 참고 자료
- Device Memory Accesses - CUDA C Programming Guide(https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-programming-guide/index.html#device-memory-accesses)
