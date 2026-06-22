> 이 글은 @Simon V(https://github.com/simveit)의 허가를 받아 전재 및 번역해 이 공중계정에 게시한 것이다.

# 벡터 합을 매우 빠르게 만들기

2025년 4월 6일

이 블로그 글에서는 벡터 reduction 작업에서 최첨단 성능을 구현하는 방법을 간략히 설명한다. 즉, 프로그램은 다음 작업을 수행해야 한다. 벡터 v가 주어졌을 때 v의 모든 원소 합을 반환한다. 여기서는 벡터가 매우 크다고 가정한다. 즉 `N = 1 << 30 = 2^30`개의 원소를 포함한다고 가정한다.

## 기준 구현

```c++
template <unsigned int threadsPerBlock>
__global__ void kernel_0(const int *d_in, int *d_out, size_t N) {
  extern __shared__ int sums[threadsPerBlock];
  int sum = 0;
  const int tid = threadIdx.x;
  const int global_tid = blockIdx.x * threadsPerBlock + tid;
  const int threads_in_grid = threadsPerBlock * gridDim.x;

  for (int i = global_tid; i < N; i += threads_in_grid) {
    sum += d_in[i];
  }
  sums[tid] = sum;
  __syncthreads();

  for (int activeThreads = threadsPerBlock >> 1; activeThreads;
       activeThreads >>= 1) {
    if (tid < activeThreads) {
      sums[tid] += sums[tid + activeThreads];
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_out[blockIdx.x] = sums[tid];
  }
}

template <int threadsPerBlock>
void kernel_0_launch(const int *d_in, int *d_first, int *d_out, size_t N) {
  const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  kernel_0<threadsPerBlock><<<numBlocks, threadsPerBlock>>>(
      d_in, d_first, N);
  kernel_0<threadsPerBlock><<<1, threadsPerBlock>>>(
      d_first, d_out, numBlocks);
}
```

우리의 기준 구현은 단순한 Two Pass 방식이다. `kernel_0`을 두 번 실행한다. 첫 번째에는 각 block 안의 원소들을 합으로 reduction하고, 그 다음 이 합들을 다시 더한다. 알고리즘의 동작 방식은 다음과 같다.

1. 현재 thread에 대응하는 벡터 원소를 shared memory에 저장한다.
2. 그런 다음 thread block을 절반으로 나눈다. 왼쪽 절반의 첫 번째 thread 결과에 오른쪽 절반 thread의 결과를 누적하고, 이를 같은 방식으로 반복한다.
3. 동기화한다. 즉 모든 thread가 작업을 완료할 때까지 기다린다. 그러면 왼쪽 절반은 각 thread의 누적 결과를 포함한다.
4. 위의 오른쪽 절반은 무시하고, 왼쪽 첫 번째 thread에 도달할 때까지 위 과정을 계속한다. 이렇게 하면 각 block 안의 합을 얻는다.
5. 그런 다음 이것을 또 다른 벡터로 보고, 단일 block 안에서 이 벡터를 reduction해 전체 합을 얻는다.

이 방식은 H100 GPU에서 `639.103 GB/s`의 대역폭을 달성하며, 이는 가능한 대역폭의 `19.3668%`에 해당한다.

## warp 사용하기

GPU에서 하나의 warp는 32개 thread로 구성된다. 각 block 안의 첫 번째 warp를 별도로 처리하고 더 효율적인 `__syncwarp();`를 사용해 동기화를 강제할 수 있다. 블로그 글의 초기 버전에서는 warp 안의 모든 thread가 동기적으로 실행된다고 가정했다는 점에 유의하라. 사실 더 새로운 아키텍처에서는 이것이 틀린 것으로 드러났다. 대부분의 경우 올바른 결과를 얻기는 하지만 race condition을 유발할 수 있으며, 이는 컴파일된 kernel에 `compute-sanitizer --tool racecheck`를 실행하면 발견할 수 있다. 다행히 `__syncwarp();`를 사용해도 대역폭 손실은 약 `1GB/s`에 불과하다. 이 점을 지적해 준 Pauleonix(https://github.com/pauleonix)에게 감사한다!

```c++
template <unsigned int threadsPerBlock>
__global__ void kernel_1(const int *d_in, int *d_out, size_t N) {
  extern __shared__ int sums[threadsPerBlock];
  int sum = 0;
  const int tid = threadIdx.x;
  const int global_tid = blockIdx.x * threadsPerBlock + tid;
  const int threads_in_grid = threadsPerBlock * gridDim.x;

  for (int i = global_tid; i < N; i += threads_in_grid) {
    sum += d_in[i];
  }
  sums[tid] = sum;
  __syncthreads();

#pragma unroll
  for (int activeThreads = threadsPerBlock >> 1; activeThreads > 32;
       activeThreads >>= 1) {
    if (tid < activeThreads) {
      sums[tid] += sums[tid + activeThreads];
    }
    __syncthreads();
  }

  volatile int *volatile_sums = sums;
#pragma unroll
  for (int activeThreads = 32; activeThreads; activeThreads >>= 1) {
    if (tid < activeThreads) {
      volatile_sums[tid] += volatile_sums[tid + activeThreads];
    }
    __syncwarp();
  }

  if (tid == 0) {
    d_out[blockIdx.x] = volatile_sums[tid];
  }
}

template <int threadsPerBlock>
void kernel_1_launch(const int *d_in, int *d_first, int *d_out, size_t N) {
  const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  kernel_1<threadsPerBlock><<<numBlocks, threadsPerBlock>>>(d_in, d_first, N);
  kernel_1<threadsPerBlock><<<1, threadsPerBlock>>>(d_first, d_out, numBlocks);
}
```

이로 인해 성능은 약간 향상되어 661.203 GB/s에 도달하며, 이용률은 20.0365%에 해당한다.

## One Pass 구현

CUDA에서는 `atomicAdd`를 사용해 block을 가로질러 결과를 특정 memory 위치에 더할 수 있다. 이를 사용해 단순한 One Pass Kernel을 구현할 수 있다.

```c++
template <unsigned int threadsPerBlock>
__global__ void kernel_2(const int *d_in, int *d_out, size_t N) {
  extern __shared__ int sums[threadsPerBlock];
  int sum = 0;
  const int tid = threadIdx.x;
  const int global_tid = blockIdx.x * threadsPerBlock + tid;

  if (global_tid == 0) {
    *d_out = 0;
  }

  if (global_tid < N) {
    sum += d_in[global_tid];
  }
  sums[tid] = sum;
  __syncthreads();

#pragma unroll
  for (int activeThreads = threadsPerBlock >> 1; activeThreads > 32;
       activeThreads >>= 1) {
    if (tid < activeThreads) {
      sums[tid] += sums[tid + activeThreads];
    }
    __syncthreads();
  }

  volatile int *volatile_sums = sums;
#pragma unroll
  for (int activeThreads = 32; activeThreads; activeThreads >>= 1) {
    if (tid < activeThreads) {
      volatile_sums[tid] += volatile_sums[tid + activeThreads];
    }
    __syncwarp();
  }

  if (tid == 0) {
    atomicAdd(d_out, volatile_sums[tid]);
  }
}

template <int threadsPerBlock>
void kernel_2_launch(const int *d_in, int *d_out, size_t N) {
  const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  kernel_2<threadsPerBlock><<<numBlocks, threadsPerBlock>>>(d_in, d_out, N);
}
```

이로 인해 성능은 `859.534 GB/s`까지 향상되며, 이용률은 `26.0465%`에 해당한다.

## 산술 강도 높이기

위 kernel을 보면 각 thread가 단지 벡터 안의 대응 원소에 접근하고 이를 shared memory에 쓰고 있다는 점이 명확하다. 각 thread가 한 묶음의 원소를 처리하게 하면 더 잘할 수 있다.

```c++
template <unsigned int threadsPerBlock, unsigned int batchSize>
__global__ void kernel_3(const int *d_in, int *d_out, size_t N) {
  extern __shared__ int sums[threadsPerBlock];
  int sum = 0;
  const int tid = threadIdx.x;  
  const int global_tid = blockIdx.x * threadsPerBlock + tid;
  const int threads_in_grid = threadsPerBlock * gridDim.x;

  if (global_tid == 0) {
    *d_out = 0;
  }

  if (global_tid < N) {
#pragma unroll
    for (int j = 0; j < batchSize; j++) {
      if (global_tid * batchSize + j < N) {
        sum += d_in[global_tid * batchSize + j];
      }
    }
  }
  sums[tid] = sum;
  __syncthreads();

#pragma unroll
  for (int activeThreads = threadsPerBlock >> 1; activeThreads > 32;
       activeThreads >>= 1) {
    if (tid < activeThreads) {
      sums[tid] += sums[tid + activeThreads];
    }
    __syncthreads();
  }

  volatile int *volatile_sums = sums;
#pragma unroll
  for (int activeThreads = 32; activeThreads; activeThreads >>= 1) {
    if (tid < activeThreads) {
      volatile_sums[tid] += volatile_sums[tid + activeThreads];
    }
    __syncwarp();
  }

  if (tid == 0) {
    atomicAdd(d_out, volatile_sums[tid]);
  }
}

template <int threadsPerBlock, int batchSize>
void kernel_3_launch(const int *d_in, int *d_out, size_t N) {
  const int numBlocks = (N + threadsPerBlock * batchSize - 1) /
                        (threadsPerBlock * batchSize);
  kernel_3<threadsPerBlock, batchSize><<<numBlocks, threadsPerBlock>>>(d_in,
                                                                       d_out, N);
}
```

보는 것처럼 이제 실행하는 block 수가 더 적어졌다. 이는 각 thread가 이제 `Batchsize`개의 원소를 처리하기 때문이다. 이 방식은 각 배치의 작업량을 늘리고 성능을 크게 끌어올린다! 이 방법을 사용하면 `3228.5 GB/s`의 대역폭을 얻을 수 있으며, 이는 물리적 최대값에 매우 가깝고 이용률은 `97.8334%`이다.

## 벡터화 로드

CUDA는 사용자 벡터화 데이터 타입 `int4`를 제공한다. 이를 사용하면 데이터를 더 효율적으로 load할 수 있다.

```c++
template <unsigned int threadsPerBlock, unsigned int batchSize>
__global__ void kernel_4(const int4 *d_in, int *d_out, size_t N) {
  extern __shared__ int sums[threadsPerBlock];
  int sum = 0;
  const int tid = threadIdx.x;  
  const int global_tid = blockIdx.x * threadsPerBlock + tid;
  const int threads_in_grid = threadsPerBlock * gridDim.x;

  if (global_tid == 0) {
    *d_out = 0;
  }

  if (global_tid < N) {
#pragma unroll
    for (int i = 0; i < batchSize >> 2; i++) {
      const int4 val = d_in[global_tid * (batchSize >> 2) + i];
      if (global_tid * batchSize + i * 4 < N) {
        sum += val.x + val.y + val.z + val.w;
      }
    }
  }
  sums[tid] = sum;
  __syncthreads();

#pragma unroll
  for (int activeThreads = threadsPerBlock >> 1; activeThreads > 32;
       activeThreads >>= 1) {
    if (tid < activeThreads) {
      sums[tid] += sums[tid + activeThreads];
    }
    __syncthreads();
  }

  volatile int *volatile_sums = sums;
#pragma unroll
  for (int activeThreads = 32; activeThreads; activeThreads >>= 1) {
    if (tid < activeThreads) {
      volatile_sums[tid] += volatile_sums[tid + activeThreads];
    }
    __syncwarp();
  }

  if (tid == 0) {
    atomicAdd(d_out, volatile_sums[tid]);
  }
}

template <int threadsPerBlock, int batchSize>
void kernel_4_launch(const int *d_in, int *d_out, size_t N) {
  const int numBlocks = (N + threadsPerBlock * batchSize - 1) /
                        (threadsPerBlock * batchSize);
  const int4 *d_in_cast = reinterpret_cast<const int4 *>(d_in);
  kernel_4<threadsPerBlock, batchSize><<<numBlocks, threadsPerBlock>>>(d_in_cast,
                                                                       d_out, N);
}
```

이는 위 버전보다 아주 약간 개선되어 `3231.9 GB/s`에 도달하며, 이용률은 `97.9364%`에 해당한다.

## NVIDIA 라이브러리 벤치마크

위 작업에 대한 NVIDIA 네이티브 구현은 다음과 같이 벤치마크할 수 있다.

```c++
void kernel_5_launch(const int *d_in, int *d_out, size_t N) {
  void* d_temp = nullptr;
  size_t temp_storage = 0;

  // 임시 저장 공간 크기를 결정하기 위한 첫 번째 호출
  cub::DeviceReduce::Sum(d_temp, temp_storage, d_in, d_out, N);
  
  // 임시 저장 공간 할당
  assert(temp_storage > 0);
  cudaMalloc(&d_temp, temp_storage);

  cub::DeviceReduce::Sum(d_temp, temp_storage, d_in, d_out, N);
}
```

이는 `3191.42 GB/s`의 대역폭과 `96.7097%`의 이용률을 제공한다. 이는 우리의 방법이 선택한 문제 크기(N = 1 << 30)와 하드웨어(H100)에서 NVIDIA 구현을 넘어섰다는 뜻이다.

## 참고 문헌

이 블로그 글은 CUDA Handbook(https://www.cudahandbook.com/)의 reduction 관련 논의에서 영감을 받았다. 배치 처리 아이디어는 fast.cu (https://github.com/pranjalssh/fast.cu/blob/main/sum.cu) 저장소와 cub 라이브러리 벤치마크용 코드에서 왔다. 그곳에서 사용한 일부 방법은 우리 kernel의 성능을 더 높일 수 있을 것이다. 하지만 나는 초보자도 여전히 쉽게 이해할 수 있는 지점에서 멈추기로 했다. 이 저장소와 고성능 CUDA kernel 작성에 관한 저자의 귀중한 통찰을 담은 블로그 글을 꼭 살펴보기를 강력히 권한다.

이 저장소(https://github.com/simveit/effective_reduction)에서 실험을 재현하고 내 코드를 찾을 수 있다. 나는 H100과 `CUDA 12.8` docker 이미지에서 실험을 실행했다.
