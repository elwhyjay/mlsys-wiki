# [LLM][CUDA] 대형 모델 면접 고빈도 CUDA 손코딩 문제

> 원문: https://zhuanlan.zhihu.com/p/678903537

## 0x00 서문

개인 기술 노트와 CUDA 학습 노트는 LeetCUDA(CUDA Learn Notes with PyTorch)에서 확인할 수 있다. LeetCUDA에는 개인 **LLM/VLM** 글 정리와 **FlashAttention, SGEMM, HGEMM, GEMV** 같은 흔한 **CUDA Kernel**의 **예제 구현**이 포함되어 있다. 현재 누적 **3k+ stars**다.

https://github.com/xlite-dev/LeetCUDA

![](images/v2-cae076e970b2cec6399017ceed59e24a_1440w.png)

CUDA Learn Notes with PyTorch

얼마 전 대형 모델 관련 면접을 몇 번 봤다. 대부분 CUDA를 손으로 작성해야 했다. 그래서 CUDA 최적화 관련 내용을 전반적으로 다시 복습했고, 고빈도 문제의 기본 작성법을 정리했다. 여기에 보관해 두면 나중에 다시 보기 쉽다.

물론 일부 코드는 최적해가 아닐 수 있다. GEMM 같은 경우 면접의 짧은 30분 안에 좋은 GEMM Kernel을 작성하는 것은 꽤 어렵다. 기억에 남는 면접이 하나 있다. 2시간 넘게 진행됐고, 1시간은 프로젝트를 물었고, 남은 1시간은 GEMM을 작성했다. 미리 준비하지 않았다면 최적화 버전을 바로 쓰는 것은 꽤 부담스러웠을 것이다. 개인 경험으로는 적중률이 높았다. 지금까지는 아래 문제들 밖의 것은 만나지 않았다. 깊은 최적화를 요구하는 문제는 보통 면접의 몇십 분 안에 손으로 다 작성하라고 하지는 않는다.

TIPS: 이 글은 개인 복습용으로 정리한 것이다.

### 중요 보충

보충: 2024.09.18. 결과 정확성과 성능을 검증하기 쉽게 하기 위해 CUDA Learn Notes를 리팩터링하고 CUDA Learn Notes with PyTorch로 이름을 바꿨다. 각 예제 kernel은 **"custom CUDA Kernel -> PyTorch bindings -> Run tests(python)"** workflow에 맞춰 다시 구성했다.

개인 기술 노트와 CUDA 학습 노트는 LeetCUDA(CUDA Learn Notes with PyTorch)에서 확인할 수 있다. LeetCUDA에는 개인 **LLM/VLM** 글 정리와 **FlashAttention, SGEMM, HGEMM, GEMV** 같은 흔한 **CUDA Kernel**의 **예제 구현**이 포함되어 있다. 현재 누적 **3k+ stars**다.

https://github.com/xlite-dev/LeetCUDA

![](images/v2-cae076e970b2cec6399017ceed59e24a_1440w.png)

CUDA Learn Notes with PyTorch

## 0x01 고빈도 면접 문제 요약

관련 kernel은 다음과 같다. 원래는 1000줄이 안 됐지만 지금은 더 많다. 외워 두는 것을 권한다. 개인적으로는 암기하는 편을 좋아한다. 외우는 과정에서 세부사항을 서서히 이해하게 된다. 물론 학습 방법은 사람마다 다르므로 편한 방식이면 된다.

- sgemm naive, sgemm + block-tile + k-tile + vec4
- sgemv k32/k128/k16 kernel
- warp/block reduce sum/max, block all reduce + vec4
- dot product, dot product + vec4
- elementwise, elementwise + vec4
- histogram, histogram + vec4
- softmax, softmax + vec4 (grid level memory fence)
- sigmoid, sigmoid + vec4
- relu, relu + vec4
- layer_norm, layer_norm + vec4
- rms_norm, rms_norm + vec4
- ...

대형 모델 관련 포지션에서는 CUDA를 손으로 작성할 확률이 매우 높다. LeetCode는 오히려 적게 쓴다. 얼마 전 개인 경험으로는 대략 4:1 비율이었다. CUDA를 잘 복습하는 것을 권한다. 물론 여기 있는 것은 가장 단순한 kernel 구현이다. `flash_attn`, FMHA 같은 최적화 방법은 이 글에 쓰지 않는다. 하지만 면접에서는 거의 물어본다. FlashAttention 시리즈 원리 설명은 별도 글을 보면 된다.

## 0x02 sgemm naive, sgemm + block-tile + k-tile + vec4

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

// SGEMM: Block Tile + K Tile, with smem
// Block Tile (BM, BN) + K Tile (BK=32)
// grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(BN, BM)
// a: MxK, b: KxN, c: MxN, compute: c = a * b, all row major
__global__ void sgemm(float* a, float* b, float* c, int M, int N, int K) {
  // [1] Block Tile: a 32x32 block computes a 32x32 tile on C
  // [2] K Tile: use shared memory and split K into BK-sized tiles
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 32;
  __shared__ float s_a[BM][BK], s_b[BK][BN];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx; // tid within the block
  // load values to shared memory, 32x32 threads working together
  // to fetch data along the row direction of a and b both for s_a
  // and s_b 32x32x4x2=8KB, we use 32x32 threads within block to
  // load 32x32 elements from global memory to shared memory, namely,
  // each thread will load 1 element.
  int load_smem_a_m = tid / 32; // 0~31, tid / 32, tid / BM, threadIdx.y
  int load_smem_a_k = tid % 32; // 0~31, tid % 32, tid % BK, threadIdx.x
  int load_smem_b_k = tid / 32; // 0~31, tid / 32, tid / BK, threadIdx.y
  int load_smem_b_n = tid % 32; // 0~31, tid % 32, tid % BN, threadIdx.x
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  // if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

  float sum = 0.f;
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    int load_gmem_a_k = bk * BK + load_smem_a_k;
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_addr];
    int load_gmem_b_k = bk * BK + load_smem_b_k;
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_addr];
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      int comp_smem_a_m = load_smem_a_m;
      int comp_smem_b_n = load_smem_b_n;
      sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
    }
    __syncthreads();
  }
  int store_gmem_c_m = load_gmem_a_m;
  int store_gmem_c_n = load_gmem_b_n;
  int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
  c[store_gmem_c_addr] = sum;
}

// SGEMM: Block Tile + Thread Tile + K Tile + Vec4, with smem
// BK:TILE_K=8 BM=BN=128
// TM=TN=8 increases compute density, BM/TM=16 BN/TN=16
// dim3 blockDim(BN/TN, BM/TM);
// dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM)
__global__ void sgemm_thread_tile_vec4(
  float* a, float* b, float* c, int M, int N, int K) {
  // [1]  Block Tile: a 16x16 block computes one 128x128 target tile on C
  // [2] Thread Tile: each thread computes TM*TN(8*8) elements to increase compute density
  // [3]      K Tile: split K into BK-sized tiles and iterate (K+BK-1)/BK times;
  //                  each iteration accumulates partial sums for TM*TN elements
  // [4]   Vectorize: reduce load/store instructions with float4
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 8;
  constexpr int TM = 8;
  constexpr int TN = 8;

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx; // tid within the block
  __shared__ float s_a[BM][BK], s_b[BK][BN]; // 2*128*8*4=8KB

  // 0. Compute indices in shared memory first.
  // Mapping between tid and smem s_a[BM][BK], BM=128 BK=8, load A row-major.
  // Each row of s_a has 8 values; each thread reads 4, so 2 threads are needed per row.
  // There are 128 rows, so 128x2=256 threads exactly.
  int load_smem_a_m = tid / 2; // tid/2 (128/8)*(128/8)=256 threads per block, tid/2->[0,128), BM=128 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 4;  // (tid%2 == 0) ? 0 : 4, col of s_a 0,4
  // Mapping between tid and smem s_b[BK][BN], BK=8 BN=128, load B row-major.
  // Each row of s_b has 128 values; each thread reads 4 values, so 32 threads are needed.
  // There are 8 rows, so 32x8=256 threads are needed.
  int load_smem_b_k = tid / 32; // tid/32, row of s_b 256/32=8 rows 0~7
  int load_smem_b_n = (tid % 32) * 4;  // (tid % 32) * 4, col of s_b 0,4,...,124
  // 1. Compute global memory indices.
  // Elements loaded to s_a correspond to rows in A global memory; each block handles BM*BN tile in C.
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c

  float r_c[TM][TN] = {0.0}; // 8x8
  // 2. Split K into BK-sized tiles.
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    // Load data to shared memory s_a BM*BK 128*8, vectorized by float4.
    int load_gmem_a_k = bk * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    FLOAT4(s_a[load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_gmem_a_addr]);
    // Load data to shared memory s_b BK*BN 8*128, vectorized by float4.
    int load_gmem_b_k = bk * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    FLOAT4(s_b[load_smem_b_k][load_smem_b_n]) = FLOAT4(b[load_gmem_b_addr]);
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BK; k++) {
      // 3. Each thread computes TM*TN(8x8) elements in BM*BN(128x128).
      #pragma unroll
      for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
          // k from 0~7, 0 ~ BK, ty and tx range from 0 to 15, 16x8=128
          int comp_smem_a_m = ty * TM + m;  // 128*8 128/TM(8)=16 threads in M direction
          int comp_smem_b_n = tx * TN + n;  // 8*128 128/TN(8)=16 threads in N direction
          r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int m = 0; m < TM; ++m) {
    int store_gmem_c_m = by * BM + ty * TM + m;
    #pragma unroll
    for (int n = 0; n < TN; n += 4) {
      int store_gmem_c_n = bx * BN + tx * TN + n;
      int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
      FLOAT4(c[store_gmem_c_addr]) = FLOAT4(r_c[m][n]);
    }
  }
}
```

여기 GEMM 구현은 비교적 단순하다. CUDA Cores만 사용했고, Block Tile + K Tile 및 Block Tile + K Tile + Thread Tile + vectorization 버전만 구현했다. 핵심은 gmem의 data를 smem으로 어떻게 load하는지, 즉 global memory의 data index를 shared memory index로 어떻게 mapping하는지다.

핵심 사고방식은 block 안의 thread id를 linear하게 이해한 뒤, 이 linear id를 global memory index 및 shared memory index와 맞추는 것이다. 예를 들어 Block Tile + K Tile 구현에서는 block 안에 총 32x32 Threads가 있고 smem으로 load해야 할 data도 32x32다. 가장 단순한 방법은 각 thread가 서로 중복되지 않는 data 하나를 load하는 것이다.

NOTE: 이 글의 GEMM kernel은 "CUDA 3: 범용 행렬 곱셈, 입문부터 숙련까지"의 구현을 수정한 것이다.

## 0x03 warp/block reduce sum/max

```cpp
// Warp Reduce Sum
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Warp Reduce Max
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// Block reduce sum/max/min device helper for Layer/RMS Norm/Softmax etc.
// grid 1D block 1D, grid(N/128), block(128)
template<const int NUM_THREADS=128>
__device__ __forceinline__ float block_reduce_sum(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];

  val = warp_reduce_sum<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  val = warp_reduce_sum<NUM_WARPS>(val);
  return val;
}

template<const int NUM_THREADS=128>
__device__ __forceinline__ float block_reduce_max(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];

  val = warp_reduce_max<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : -FLT_MAX;
  val = warp_reduce_max<NUM_WARPS>(val);
  return val;
}
```

warp reduce는 대부분 reduce kernel의 표준 작성법에 가깝다. vLLM도 이런 classic 작성법을 사용한다. 따라서 먼저 warp reduce, 즉 여러 warp functions의 사용법을 이해하고 다른 kernel을 작성하면 사고가 훨씬 쉬워진다.

주의할 점은 warp function이 register의 data를 처리한다는 것이다. 따라서 이 시점에는 data를 먼저 smem으로 load한 뒤 reduce할 필요가 없다. register로 직접 load하면 된다. 예전에 이런 작은 실수를 한 적이 있다. Warp Functions는 "CUDA programming introduction to Warp-Level Primitives"를 참고하면 좋다.

## 0x04 block all reduce + vec4

```cpp
// Block All Reduce Sum
// grid(N/128), block(128)
// a: Nx1, y=sum(a)
template<const int NUM_THREADS = 128>
__global__ void block_all_reduce_sum(float* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  // keep the data in register is enougth for warp operaion.
  float sum = (idx < N) ? a[idx] : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum = warp_reduce_sum<WARP_SIZE>(sum);
  // warp leaders store the data to shared memory.
  if (lane == 0) reduce_smem[warp] = sum;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) sum = warp_reduce_sum<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
}

// Block All Reduce Sum + float4
// grid(N/128), block(128/4)
// a: Nx1, y=sum(a)
template<const int NUM_THREADS = 128/4>
__global__ void block_all_reduce_sum_vec4(float* a, float* y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  float4 reg_a = FLOAT4(a[idx]);
  // keep the data in register is enougth for warp operaion.
  float sum = (idx < N) ? (reg_a.x + reg_a.y + reg_a.z + reg_a.w) : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  sum = warp_reduce_sum<WARP_SIZE>(sum);
  // warp leaders store the data to shared memory.
  if (lane == 0) reduce_smem[warp] = sum;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) sum = warp_reduce_sum<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
}
```

block all reduce는 warp reduce 위에 구성한다. `reduce_smem`에 대한 shared memory allocation은 피하기 어렵다. 각 warp가 얻은 local result를 동기화하기 위한 용도다. 마지막에는 global sum을 얻기 위해 block level atomic operation인 `atomicAdd`도 필요하다. `float4` vectorization은 memory access를 최적화해 WarpScheduler가 instruction을 보내는 압력을 줄일 수 있다.

## 0x05 sgemv k32/k128/k16 kernel

```cpp
// SGEMV: Warp SGEMV K32
// Assume K is a multiple of 32; each warp handles one row.
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void sgemv_k32(float* a, float* x, float* y, int M, int K) {
  int tx = threadIdx.x;         // 0~31
  int ty = threadIdx.y;         // 0~4
  int bx = blockIdx.x;          // 0~M/4
  int lane = tx % WARP_SIZE;    // 0~31
  int m = bx * blockDim.y + ty; // (0~M/4) * 4 + (0~3)
  if (m < M) {
    float sum = 0.0f;
    int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
    #pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
      // If NUM_WARPS>=2, first accumulate data from the current row into the first warp.
      int k = w * WARP_SIZE + lane;
      sum += a[m * K + k] * x[k];
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);
    if (lane == 0) y[m] = sum;
  }
}

// SGEMV: Warp SGEMV K128 + Vec4
// Assume K is a multiple of 128, float4.
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void sgemv_k128(float* a, float* x, float* y, int M, int K) {
  // Each thread handles 4 elements; one warp covers 128 elements.
  int tx = threadIdx.x;         // 0~31
  int ty = threadIdx.y;         // 0~3
  int bx = blockIdx.x;          // 0~M/4
  int lane = tx % WARP_SIZE;    // 0~31
  int m = blockDim.y * bx + ty; // (0~M/4) * 4 + (0~3)

  if (m < M) {
    float sum = 0.0f;
    // process 4*WARP_SIZE elements per warp.
    int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
    #pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
      int k = (w * WARP_SIZE + lane) * 4;
      float4 reg_x = FLOAT4(x[k]);
      float4 reg_a = FLOAT4(a[m * K + k]);
      sum += (reg_a.x * reg_x.x + reg_a.y * reg_x.y
            + reg_a.z * reg_x.z + reg_a.w * reg_x.w);
    }
    sum = warp_reduce_sum<WARP_SIZE>(sum);
    if(lane == 0) y[m] = sum;
  }
}

// SGEMV: Warp SGEMV K16
// Assume K is 16 < 32; each warp handles 2 rows, each row has 16 elements.
// NUM_THREADS=128, NUM_WARPS=NUM_THREADS/WARP_SIZE;
// NUM_ROWS=NUM_WARPS * ROW_PER_WARP, grid(M/NUM_ROWS), block(32,NUM_WARPS)
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
template<const int ROW_PER_WARP = 2>
__global__ void sgemv_k16(float* A, float* x, float* y, int M, int K) {
  constexpr int K_WARP_SIZE = (WARP_SIZE + ROW_PER_WARP - 1) / ROW_PER_WARP;
  int tx = threadIdx.x;       // 0~31
  int ty = threadIdx.y;       // 0~NUM_WARPS
  int bx = blockIdx.x;        // 0~M/NUM_ROWS (NUM_ROWS=NUM_WARPS * ROW_PER_WARP)
  int lane = tx % WARP_SIZE;  // 0~31
  int k = lane % K_WARP_SIZE; // 0~15
  // global row of a: MxK and y:Mx1, blockDim.y=NUM_WARPS
  int m = (blockDim.y * bx + ty) * ROW_PER_WARP + lane / K_WARP_SIZE;
  if (m < M) {
    float sum = A[m * K + k] * x[k];
    sum = warp_reduce_sum<K_WARP_SIZE>(sum);
    // Note that this is k == 0, not lane == 0.
    if(k == 0) y[m] = sum;
  }
}
```

어떤 사람들은 sgemv의 다양한 최적화 버전을 거꾸로 서서도 쓸 수 있을 것이다. 핵심 사고는 사실 warp reduce 기반이며, K의 서로 다른 경우를 고려해 최적화하는 것이다. 이 글의 sgemv kernel은 "GPU optimization series: GEMV optimization"의 구현을 수정한 것이다.

## 0x06 dot product, dot product + vec4

```cpp
// Dot Product
// grid(N/128), block(128)
// a: Nx1, b: Nx1, y=sum(elementwise_mul(a,b))
template<const int NUM_THREADS = 128>
__global__ void dot(float* a, float* b, float* y, int N) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  // keep the data in register is enougth for warp operaion.
  float prod = (idx < N) ? a[idx] * b[idx] : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  prod = warp_reduce_sum<WARP_SIZE>(prod);
  // warp leaders store the data to shared memory.
  if (lane == 0) reduce_smem[warp] = prod;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) prod = warp_reduce_sum<NUM_WARPS>(prod);
  if (tid == 0) atomicAdd(y, prod);
}

// Dot Product + Vec4
// grid(N/128), block(128/4)
// a: Nx1, b: Nx1, y=sum(elementwise_mul(a,b))
template<const int NUM_THREADS = 128/4>
__global__ void dot_vec4(float* a, float* b, float* y, int N) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  float4 reg_a = FLOAT4(a[idx]);
  float4 reg_b = FLOAT4(b[idx]);
  float prod = (idx < N) ? (reg_a.x * reg_b.x + reg_a.y * reg_b.y
                          + reg_a.z * reg_b.z + reg_a.w * reg_b.w) : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  // perform warp sync reduce.
  prod = warp_reduce_sum<WARP_SIZE>(prod);
  // warp leaders store the data to shared memory.
  if (lane == 0) reduce_smem[warp] = prod;
  __syncthreads(); // make sure the data is in shared memory.
  // the first warp compute the final sum.
  prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  if (warp == 0) prod = warp_reduce_sum<NUM_WARPS>(prod);
  if (tid == 0) atomicAdd(y, prod);
}
```

dot product kernel의 핵심은 block reduce다. 더 설명할 것은 많지 않다.

## 0x07 elementwise, elementwise + vec4

```cpp
// ElementWise Add
// grid(N/128), block(128)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add(float* a, float* b, float* c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) c[idx] = a[idx] + b[idx];
}

// ElementWise Add + Vec4
// grid(N/128), block(128/4)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_vec4(float* a, float* b, float* c, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FLOAT4(c[idx]) = reg_c;
  }
}
```

elementwise는 vectorization을 조금 넣어 memory access를 최적화할 수 있다.

## 0x08 histogram, histogram + vec4

```cpp
// Histogram
// grid(N/128), block(128)
// a: Nx1, y: count histogram
__global__ void histogram(int* a, int* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) atomicAdd(&(y[a[idx]]), 1);
}

// Histogram + Vec4
// grid(N/128), block(128/4)
// a: Nx1, y: count histogram
__global__ void histogram_vec4(int* a, int* y, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    int4 reg_a = INT4(a[idx]);
    atomicAdd(&(y[reg_a.x]), 1);
    atomicAdd(&(y[reg_a.y]), 1);
    atomicAdd(&(y[reg_a.z]), 1);
    atomicAdd(&(y[reg_a.w]), 1);
  }
}
```

frequency histogram이다. 아주 단순하며 몇 줄이면 된다.

## 0x09 softmax, softmax + vec4 (grid level memory fence)

```cpp
// Softmax x: N, y: N
// grid(N/128), block(K=128)
template<const int NUM_THREADS = 128>
__global__ void softmax(float* x, float* y, float* total, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];

  float sum = (idx < N) ? expf(x[idx]) : 0.0f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  sum = warp_reduce_sum<WARP_SIZE>(sum);
  if (lane == 0) reduce_smem[warp] = sum;
  __syncthreads();
  // compute the final sum in each warp
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
  sum = warp_reduce_sum<NUM_WARPS>(sum); // sum(e^x_0,...,e^x_n-1)
  // get the total sum of all blocks.
  if (tid == 0) atomicAdd(total, sum);
  __threadfence(); // grid level memory fence; grid-level memory synchronization is needed here
  // e^x_i/sum(e^x_0,...,e^x_n-1)
  if (idx < N) y[idx] = block_smem[tid] / (*total);
}

// Softmax x: N, y: N
// grid(N/128), block(K=128)
template<const int NUM_THREADS = 128>
__global__ void softmax_v2(float* x, float* y, float* total, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;

  float exp_val = (idx < N) ? expf(x[idx]) : 0.0f;
  float sum = block_reduce_sum<NUM_THREADS>(exp_val);
  // get the total sum of all blocks.
  if (tid == 0) atomicAdd(total, sum);
  __threadfence(); // grid level memory fence; grid-level memory synchronization is needed here
  // e^x_i/sum(e^x_0,...,e^x_n-1)
  if (idx < N) y[idx] = exp_val / (*total);
}

// Softmax Vec4 x: N, y: N
// grid(N/128), block(128/4)
template<const int NUM_THREADS = 128/4>
__global__ void softmax_v2_vec4(float* x, float* y, float* total, int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * blockDim.x + tid) * 4;

  float4 reg_x = FLOAT4(x[idx]);
  float4 reg_exp;
  reg_exp.x = (idx < N) ? expf(reg_x.x) : 0.0f;
  reg_exp.y = (idx < N) ? expf(reg_x.y) : 0.0f;
  reg_exp.z = (idx < N) ? expf(reg_x.z) : 0.0f;
  reg_exp.w = (idx < N) ? expf(reg_x.w) : 0.0f;
  float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
  float sum = block_reduce_sum<NUM_THREADS>(exp_val);
  // get the total sum of all blocks.
  if (tid == 0) atomicAdd(total, sum);
  __threadfence(); // grid level memory fence; grid-level memory synchronization is needed here
  // e^x_i/sum(e^x_0,...,e^x_n-1)
  if (idx < N) {
    float4 reg_y;
    reg_y.x = reg_exp.x / (*total);
    reg_y.y = reg_exp.y / (*total);
    reg_y.z = reg_exp.z / (*total);
    reg_y.w = reg_exp.w / (*total);
    FLOAT4(y[idx]) = reg_y;
  }
}
```

softmax에서 조금 주의할 부분은 memory synchronization이다. 여기서는 grid level synchronization이 필요하다. block level만으로는 global `exp sum`을 denominator로 얻을 수 없다. 그래서 `__threadfence`라는 grid-level memory synchronization 작업을 사용한다. 다만 효율은 아직 측정하지 않았다. 정말 효율적으로 만들려면 FA2처럼 1-pass + online softmax 구현으로 가야 할 수 있다. 면접에서는 너무 스스로를 괴롭힐 필요 없다. 그래도 FA1/FA2 논문은 classic이므로 여러 번 읽는 것을 강하게 권한다.

## 0x0a sigmoid, sigmoid + vec4

```cpp
// Sigmoid x: N, y: N y=1/(1+exp(-x))
// grid(N/128), block(K=128)
__global__ void sigmoid(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 1.0f / (1.0f + expf(-x[idx]));
}

// Sigmoid x: N, y: N y=1/(1+exp(-x)) Vec4
// grid(N/128), block(128/4)
__global__ void sigmoid_vec4(float* x, float* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = 1.0f / (1.0f + expf(-reg_x.x));
    reg_y.y = 1.0f / (1.0f + expf(-reg_x.y));
    reg_y.z = 1.0f / (1.0f + expf(-reg_x.z));
    reg_y.w = 1.0f / (1.0f + expf(-reg_x.w));
    FLOAT4(y[idx]) = reg_y;
  }
}
```

## 0x0b relu, relu + vec4

```cpp
// Relu x: N, y: N y=max(0,x)
// grid(N/128), block(K=128)
__global__ void relu(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = fmaxf(0.0f, x[idx]);
}

// Relu x: N, y: N y=max(0,x) Vec4
// grid(N/128/4), block(128/4)
__global__ void relu_vec4(float* x, float* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = fmaxf(0.0f, reg_x.x);
    reg_y.y = fmaxf(0.0f, reg_x.y);
    reg_y.z = fmaxf(0.0f, reg_x.z);
    reg_y.w = fmaxf(0.0f, reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}
```

## 0x0c layer_norm, layer_norm + vec4

```cpp
// Layer Norm: x: NxK(K=128<1024), y': NxK, y'=x-mean(x)/std(x) each row
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
template<const int NUM_THREADS=128>
__global__ void layer_norm(float* x, float* y, float g, float b, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x; // 0..N-1
  int idx = bid * blockDim.x + threadIdx.x;
  const float epsilon = 1e-5f;

  __shared__ float s_mean; // shared within block
  __shared__ float s_variance; // shared within block
  float value = (idx < N * K) ? x[idx] : 0.0f; // load once only
  float sum = block_reduce_sum<NUM_THREADS>(value);
  if (tid == 0) s_mean = sum / (float) K;
  // wait for s_mean in shared memory to be ready for all threads
  __syncthreads();
  float variance = (value - s_mean) * (value - s_mean);
  variance = block_reduce_sum<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float) K + epsilon);
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();
  if (idx < N * K) y[idx] = ((value - s_mean) * s_variance) * g + b;
}

// Layer Norm Vec4: x: NxK(K=128<1024), y': NxK, y'=x-mean(x)/std(x) each row
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N*K/K), block(K/4<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
template<const int NUM_THREADS=128/4>
__global__ void layer_norm_vec4(float* x, float* y, float g, float b, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x; // 0..N-1
  int idx = (bid * blockDim.x + threadIdx.x) * 4;
  const float epsilon = 1e-5f;

  __shared__ float s_mean; // shared within block
  __shared__ float s_variance; // shared within block
  float4 reg_x = FLOAT4(x[idx])
  float value = (idx < N * K) ? (reg_x.x + reg_x.y
                               + reg_x.z + reg_x.w) : 0.0f;
  float sum = block_reduce_sum<NUM_THREADS>(value);
  if (tid == 0) s_mean = sum / (float) K;
  // wait for s_mean in shared memory to be ready for all threads
  __syncthreads();
  float4 reg_x_hat;
  reg_x_hat.x = reg_x.x - s_mean;
  reg_x_hat.y = reg_x.y - s_mean;
  reg_x_hat.z = reg_x.z - s_mean;
  reg_x_hat.w = reg_x.w - s_mean;
  float variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y
                 + reg_x_hat.z * reg_x_hat.z + reg_x_hat.w * reg_x_hat.w;
  variance = block_reduce_sum<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float) K + epsilon);
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();
  float4 reg_y;
  reg_y.x = reg_x_hat.x * s_variance * g + b;
  reg_y.y = reg_x_hat.y * s_variance * g + b;
  reg_y.z = reg_x_hat.z * s_variance * g + b;
  reg_y.w = reg_x_hat.w * s_variance * g + b;
  if (idx < N * K) FLOAT4(y[idx]) = reg_y;
}
```

layer norm 구현의 핵심도 block reduce와 warp reduce다. 여기에 vectorization을 조금 넣는다.

## 0x0d rms_norm, rms_norm + vec4

```cpp
// RMS Norm: x: NxK(K=128<1024), y': NxK, y'=x/rms(x) each row
// 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g (g: scale)
template<const int NUM_THREADS=128>
__global__ void rms_norm(float* x, float* y, float g, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x; // 0..N-1
  int idx = bid * blockDim.x + threadIdx.x;
  const float epsilon = 1e-5f;

  __shared__ float s_variance; // shared within block
  float value = (idx < N * K) ? x[idx] : 0.0f; // load once only
  float variance = value * value;
  variance = block_reduce_sum<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float) K + epsilon);
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();
  if (idx < N * K) y[idx] = (value * s_variance) * g;
}

// RMS Norm Vec4: x: NxK(K=128<1024), y': NxK, y'=x/rms(x) each row
// 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
// grid(N*K/K), block(K/4<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g (g: scale)
template<const int NUM_THREADS=128/4>
__global__ void rms_norm_vec4(float* x, float* y, float g, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x; // 0..N-1
  int idx = (bid * blockDim.x + threadIdx.x) * 4;
  const float epsilon = 1e-5f;

  __shared__ float s_variance; // shared within block
  float4 reg_x = FLOAT4(x[idx]);
  float variance = (idx < N * K) ? (reg_x.x * reg_x.x + reg_x.y * reg_x.y
                                  + reg_x.z * reg_x.z + reg_x.w * reg_x.w) : 0.0f;
  variance = block_reduce_sum<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float) K + epsilon);
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();
  float4 reg_y;
  reg_y.x = reg_x.x * s_variance * g;
  reg_y.y = reg_x.y * s_variance * g;
  reg_y.z = reg_x.z * s_variance * g;
  reg_y.w = reg_x.w * s_variance * g;
  if (idx < N * K) FLOAT4(y[idx]) = reg_y;
}
```

rms norm 구현의 핵심도 block reduce와 warp reduce다. 그 위에 `float4` vectorization을 약간 더한다.

## 0x0e NMS

```cpp
struct Box {
  float x1, y1, x2, y2, score;
  float area() const {return (std::abs(x2 - x1 + 1)) * std::abs(y2 - y1 + 1); }
  float iou_of(const Box& other) const{
    float inner_x1 = x1 > other.x1 ? x1 : other.x1;
    float inner_y1 = y1 > other.y1 ? y1 : other.y1;
    float inner_x2 = x2 < other.x2 ? x2 : other.x2;
    float inner_y2 = y2 < other.y2 ? y2 : other.y2;
    float inner_h = inner_y2 - inner_y1 + 1.0f;
    float inner_w = inner_x2 - inner_x1 + 1.0f;
    float inner_area = inner_h * inner_w;
    return (inner_area / (area() + tbox.area() - inner_area));
  }
}
void hard_nms(std::vector<Box> &input, std::vector<Box> &output, float iou_threshold){
  if (input.empty()) return;
  std::sort(input.begin(), input.end(),[](Box& a, Box& b) { return a.score > b.score; });
  int box_num = input.size();
  std::vector<int> merged(box_num, 0);
  for (int i = 0; i < box_num; ++i) {
    if (merged[i]) continue;
    merged[i] = 1;
    for (int j = i + 1; j < box_num; ++j) {
      if (merged[j]) continue;
      float iou = input[i].iou_of(input[j]);
      if (iou > iou_threshold) merged[j] = 1;
    }
    output.push_back(input[i]);
  }
}
```

CV 관련 면접에서는 NMS를 손으로 쓰라고 하는 경우가 많다. 함께 기록해 둔다.

## 0x0f 정리

대부분 kernel의 기본 작성법은 warp reduce와 block reduce에 의존한다. 다양한 상황에서 warp functions를 능숙하게 적용할 수 있으면 큰 문제는 없다. softmax는 grid-level synchronization 문제를 고려해야 하며, 또는 online softmax와 FlashAttention을 고려해야 한다.

SGEMM 최적화는 큰 주제다. 여기 예시처럼 단순하지 않다. 다만 입문 단계에서는 기본적으로 tiling 사고와 index 사이의 mapping을 어떻게 하는지가 핵심이다. SGEMV 최적화는 주로 K의 서로 다른 값을 고려한다. M이 1이기 때문이다. 예를 들어 K=16, 64, 128 같은 경우 warp 단위로 어떻게 처리할지 생각해야 한다.

ReLU, sigmoid 등은 모두 elementwise operation이므로 구현하기 쉽다. 여기에 vectorization을 추가해 memory access를 최적화할 수 있다. layer norm과 rms norm은 수학적으로도 명확하고 단순하다. CUDA kernel로 옮길 때는 token별로 처리하면 된다. head dim이 1024를 넘지 않는 경우, 즉 한 block에 최대 1024 threads를 넣을 수 있는 경우에는 한 block에서 처리할 수 있으므로 parallelization도 쓰기 좋다. 물론 핵심은 여전히 warp reduce와 block reduce다.

### 중요 보충

보충: 2024.09.18. 결과 정확성과 성능을 검증하기 쉽게 하기 위해 CUDA Learn Notes를 리팩터링하고 CUDA Learn Notes with PyTorch로 이름을 바꿨다. 각 예제 kernel은 **"custom CUDA Kernel -> PyTorch bindings -> Run tests(python)"** workflow에 맞춰 다시 구성했다. 현재 누적 **1.5k+ stars**다.

https://github.com/xlite-dev/CUDA-Learn-Notes

![](images/v2-cae076e970b2cec6399017ceed59e24a_1440w.png)

CUDA Learn Notes with PyTorch
