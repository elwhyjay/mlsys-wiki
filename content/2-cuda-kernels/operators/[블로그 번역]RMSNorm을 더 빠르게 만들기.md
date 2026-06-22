> 이 글은 @Simon V(https://github.com/simveit)의 허가를 받아 전재 및 번역해 이 공식 계정에 게시한 것이다. 원문 주소는 https://veitner.bearblog.dev/making-rmsnorm-really-fast/ 이다.

# RMSNorm을 더 빠르게 만들기

2025년 4월 18일

RMS Norm은 현대 LLMs에서 자주 사용되는 연산이다. 벡터 $v$가 주어졌을 때, RMS Norm 계산 방식은 $v_i = \frac{v_i}{RMS(v)} \cdot w_i$이며, 여기서 $w_i$는 weight이고 $RMS(v) = \sqrt{\epsilon + \frac{1}{N}\sum_{i=1,...,N}v_i^2}$다. 이 블로그 글에서는 행렬 $V = [v_1,...,v_{numToken}]$의 각 행에 대한 RMS Norm을 계산한다. 여기서 $v_i = [x_1,...,x_{hiddenDim}]$이고, weight $w = [w_1,...,w_{hiddenDim}]$가 주어진다.

## 순차 구현

우리 kernel의 정확성을 검사하려면 reference로 사용할 기본적인 순차 구현이 필요하다. 아래는 우리가 사용하는 간단한 버전이다.

```c++
template <int numTokens, int hiddenDim>
void launchRmsNormCpu(float *x, float *w, float eps, float *y) {
  float rms;
  for (int token = 0; token < numTokens; token++) {
    rms = 0;
    for (int hidden = 0; hidden < hiddenDim; hidden++) {
      rms += x[token * hiddenDim + hidden] * x[token * hiddenDim + hidden];
    }
    rms = sqrt(rms / hiddenDim + eps);
    for (int hidden = 0; hidden < hiddenDim; hidden++) {
      y[token * hiddenDim + hidden] =
          x[token * hiddenDim + hidden] / rms * w[hidden];
    }
  }
}

```

## 어떻게 병렬화할까?

우리의 병렬화 시도는 매우 단순하다. 각 block은 하나의 token을 처리한다. block 안의 thread 수가 hidden dimension 크기보다 작으면 각 thread가 여러 element를 처리해야 한다. 그런 다음 간단한 reduction 연산을 수행해 RMS Norm을 계산하고 출력을 쓴다. reduction 연산에 익숙하지 않다면 이전 [reduction에 대한 블로그 글](https://mp.weixin.qq.com/s/RklG6tmJnzPbIWxVBKDgLg)을 참고하기 바란다.

## Naive kernel

CUDA에서 naive solution은 다음과 같다.

```c++
template <int hiddenDim, int threadsPerBlock>
__global__ void rmsNormKernelNaive(float *x, float *w, float eps, float *y) {
  __shared__ float squaredPerThread[threadsPerBlock];
  __shared__ float rms_;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  float sum = 0.0f;

  for (int i = tid; i < hiddenDim; i += threadsPerBlock) {
    float x_ = x[bid * hiddenDim + i];
    sum += x_ * x_;
  }
  squaredPerThread[tid] = sum;
  __syncthreads();

  for (int activeThreads = threadsPerBlock / 2; activeThreads > 0;
       activeThreads >>= 1) {
    if (tid < activeThreads) {
      squaredPerThread[tid] += squaredPerThread[tid + activeThreads];
    }
    __syncthreads();
  }

  if (tid == 0) {
    rms_ = rsqrtf(squaredPerThread[tid] / hiddenDim + eps);
  }
  __syncthreads();

  for (int i = tid; i < hiddenDim; i += threadsPerBlock) {
    y[bid * hiddenDim + i] = x[bid * hiddenDim + i] * rms_ * w[i];
  }
}

template <int numTokens, int hiddenDim, int threadsPerBlock>
void launchRmsNormNaive(float *x, float *w, float eps, float *y) {
  rmsNormKernelNaive<hiddenDim, threadsPerBlock>
      <<<numTokens, threadsPerBlock>>>(x, w, eps, y);
}
```

`x`는 memory를 한 번 가로질러 접근하고, `w`도 memory를 한 번 가로질러 접근하며, `y`도 memory를 한 번 가로질러 접근한다. `numTokens = 1 << 18` 및 `hiddenDim = 1 << 12`의 경우 `w`의 영향은 무시할 수 있으므로, bandwidth를 다음 방식으로 계산할 수 있다.

```c++
const size_t size = numTokens * hiddenDim * sizeof(float);
size_t numCrossMemoryBound = 2 * size;
float latency = time / numRounds;
float bandwidth = (numCrossMemoryBound / latency) / 1e6;
```

위 kernel의 결과는 다음과 같다.

```shell
Latency = 2.84878 ms
Bandwidth = 3015.3 GB/s
% of max = 91.3727 %
```

## shared memory 사용

위에서 보았듯이 우리는 `x`의 element에 빈번하게 접근한다. shared memory를 사용하면 memory access를 더 빠르게 할 수 있다.

```c++
template <int hiddenDim, int threadsPerBlock>
__global__ void rmsNormKernelSmem(float *x, float *w, float eps, float *y) {
  __shared__ float squaredPerThread[threadsPerBlock];
  __shared__ float xShared[hiddenDim];
  __shared__ float rms_;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  float sum = 0.0f;

  for (int i = tid; i < hiddenDim; i += threadsPerBlock) {
    int index = bid * hiddenDim + i;
    float x_ = x[index];
    xShared[i] = x_;
    sum += x_ * x_;
  }
  squaredPerThread[tid] = sum;
  __syncthreads();

  for (int activeThreads = threadsPerBlock / 2; activeThreads > 0;
       activeThreads >>= 1) {
    if (tid < activeThreads) {
      squaredPerThread[tid] += squaredPerThread[tid + activeThreads];
    }
    __syncthreads();
  }

  if (tid == 0) {
    rms_ = rsqrtf(squaredPerThread[tid] / hiddenDim + eps);
  }
  __syncthreads();

  for (int i = tid; i < hiddenDim; i += threadsPerBlock) {
    float val = xShared[i] * rms_ * w[i];
    y[bid * hiddenDim + i] = val;
  }
}

template <int numTokens, int hiddenDim, int threadsPerBlock>
void launchRmsNormSmem(float *x, float *w, float eps, float *y) {
  rmsNormKernelSmem<hiddenDim, threadsPerBlock>
      <<<numTokens, threadsPerBlock>>>(x, w, eps, y);
}
```

위 kernel의 결과는 다음과 같다.

```shell
Latency = 2.82101 ms
Bandwidth = 3044.99 GB/s
% of max = 92.2723 %
```

## warp 사용

[prefix sum 연산](https://mp.weixin.qq.com/s/aKBwPEBEsxbLXJc_CKtl-A)에 적용했던 기법과 비슷하게, 여기서도 다음과 같이 할 수 있다.

- 각 warp 안에서 reduction을 수행한다.
- 하나의 warp로 이 배열을 reduction해 최종 reduction 결과를 얻는다. 이 과정의 코드는 다음과 같다.

```c++
#define WARP_SIZE 32

__device__ float warpReduce(float x) {
  float val = x;
  for (int activeThreads = WARP_SIZE >> 1; activeThreads > 0;
       activeThreads >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, activeThreads);
  }
  return val;
}

template <int hiddenDim, int threadsPerBlock>
__global__ void rmsNormKernelWarp(float *x, float *w, float eps, float *y) {
  __shared__ float squaredPerThread[threadsPerBlock];
  __shared__ float xShared[hiddenDim];
  __shared__ float sumPerWarp[WARP_SIZE];
  __shared__ float rms_;

  const int tid = threadIdx.x;
  const int laneId = tid & 31;
  const int warpId = tid >> 5;
  const int warpsPerBlock = threadsPerBlock >> 5;

  const int bid = blockIdx.x;
  float sum = 0.0f;

  for (int i = tid; i < hiddenDim; i += threadsPerBlock) {
    float x_ = x[bid * hiddenDim + i];
    xShared[i] = x_;
    sum += x_ * x_;
  }
  squaredPerThread[tid] = sum;
  __syncthreads();

  float warpSum = warpReduce(squaredPerThread[tid]);
  if (laneId == 0) {
    sumPerWarp[warpId] = warpSum;
  }
  __syncthreads();

  if (tid < WARP_SIZE) {
    sumPerWarp[tid] = warpReduce(tid < warpsPerBlock ? sumPerWarp[tid] : 0);
    if (tid == 0) {
      rms_ = rsqrtf(sumPerWarp[tid] / hiddenDim + eps);
    }
  }
  __syncthreads();

  for (int i = tid; i < hiddenDim; i += threadsPerBlock) {
    y[bid * hiddenDim + i] = xShared[i] * rms_ * w[i];
  }
}

template <int numTokens, int hiddenDim, int threadsPerBlock>
void launchRmsNormWarp(float *x, float *w, float eps, float *y) {
  rmsNormKernelWarp<hiddenDim, threadsPerBlock>
      <<<numTokens, threadsPerBlock>>>(x, w, eps, y);
}
```

위 kernel의 결과는 다음과 같다.

```shell
Latency = 2.82263 ms
Bandwidth = 3043.23 GB/s
% of max = 92.2192 %
```

처음에는 이것이 더 빠를 것이라고 예상했지만, 사실은 그렇지 않았다.

## vectorized load와 store

위 kernel을 performance analysis해 보면 memory load와 store가 가장 많은 명령어를 소비한다는 것을 볼 수 있다. CUDA의 float4 데이터 타입을 사용해 load와 store 연산을 vectorize하면 이를 최적화할 수 있다.

shared memory 방식의 코드는 다음과 같다.

```c++
template <int hiddenDim, int threadsPerBlock>
__global__ void rmsNormKernelSmemFloat4(float4 *x, float4 *w, float eps,
                                        float4 *y) {
  __shared__ float squaredPerThread[threadsPerBlock];
  __shared__ float4 xShared[hiddenDim >> 2];
  __shared__ float rms_;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  float sum = 0.0f;

  for (int i = tid; i < hiddenDim >> 2; i += threadsPerBlock) {
    int index = bid * (hiddenDim >> 2) + i;
    float4 x_ = x[index];
    xShared[i] = x_;
    sum += (x_.x * x_.x) + (x_.y * x_.y) + (x_.z * x_.z) + (x_.w * x_.w);
  }
  squaredPerThread[tid] = sum;
  __syncthreads();

  for (int activeThreads = threadsPerBlock >> 1; activeThreads > 0;
       activeThreads >>= 1) {
    if (tid < activeThreads) {
      squaredPerThread[tid] += squaredPerThread[tid + activeThreads];
    }
    __syncthreads();
  }

  if (tid == 0) {
    rms_ = rsqrtf(squaredPerThread[tid] / hiddenDim + eps);
  }
  __syncthreads();

  for (int i = tid; i < hiddenDim >> 2; i += threadsPerBlock) {
    float4 w_ = w[i];
    float4 x_ = xShared[i];
    float4 val = make_float4(x_.x * rms_ * w_.x, x_.y * rms_ * w_.y,
                             x_.z * rms_ * w_.z, x_.w * rms_ * w_.w);
    y[bid * (hiddenDim >> 2) + i] = val;
  }
}

template <int numTokens, int hiddenDim, int threadsPerBlock>
void launchRmsNormSmemFloat4(float *x, float *w, float eps, float *y) {
  float4 *x_ = reinterpret_cast<float4 *>(x);
  float4 *w_ = reinterpret_cast<float4 *>(w);
  float4 *y_ = reinterpret_cast<float4 *>(y);
  rmsNormKernelSmemFloat4<hiddenDim, threadsPerBlock>
      <<<numTokens, threadsPerBlock>>>(x_, w_, eps, y_);
}
```

위 kernel의 결과는 다음과 같다.

```shell
Latency = 2.80455 ms
Bandwidth = 3062.86 GB/s
% of max = 92.8139 %
```

마찬가지로 warp kernel도 최적화할 수 있다.

```c++
#define WARP_SIZE 32

__device__ float warpReduce(float x) {
  float val = x;
  for (int activeThreads = WARP_SIZE >> 1; activeThreads > 0;
       activeThreads >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, activeThreads);
  }
  return val;
}

template <int hiddenDim, int threadsPerBlock>
__global__ void rmsNormKernelWarpFloat4(float4 *x, float4 *w, float eps,
                                        float4 *y) {
  __shared__ float squaredPerThread[threadsPerBlock];
  __shared__ float4 xShared[hiddenDim >> 2];
  __shared__ float sumPerWarp[WARP_SIZE];
  __shared__ float rms_;

  const int tid = threadIdx.x;
  const int laneId = tid & 31;
  const int warpId = tid >> 5;
  const int warpsPerBlock = threadsPerBlock >> 5;

  const int bid = blockIdx.x;
  float sum = 0.0f;

  for (int i = tid; i < hiddenDim >> 2; i += threadsPerBlock) {
    int index = bid * (hiddenDim >> 2) + i;
    float4 x_ = x[index];
    xShared[i] = x_;
    sum += (x_.x * x_.x) + (x_.y * x_.y) + (x_.z * x_.z) + (x_.w * x_.w);
  }
  squaredPerThread[tid] = sum;
  __syncthreads();

  float warpSum = warpReduce(squaredPerThread[tid]);
  if (laneId == 0) {
    sumPerWarp[warpId] = warpSum;
  }
  __syncthreads();

  if (tid < WARP_SIZE) {
    sumPerWarp[tid] = warpReduce(tid < warpsPerBlock ? sumPerWarp[tid] : 0);
    if (tid == 0) {
      rms_ = rsqrtf(sumPerWarp[tid] / hiddenDim + eps);
    }
  }
  __syncthreads();

  for (int i = tid; i < hiddenDim >> 2; i += threadsPerBlock) {
    float4 w_ = w[i];
    float4 x_ = xShared[i];
    float4 val = make_float4(x_.x * rms_ * w_.x, x_.y * rms_ * w_.y,
                             x_.z * rms_ * w_.z, x_.w * rms_ * w_.w);
    y[bid * (hiddenDim >> 2) + i] = val;
  }
}

template <int numTokens, int hiddenDim, int threadsPerBlock>
void launchRmsNormWarpFloat4(float *x, float *w, float eps, float *y) {
  float4 *x_ = reinterpret_cast<float4 *>(x);
  float4 *w_ = reinterpret_cast<float4 *>(w);
  float4 *y_ = reinterpret_cast<float4 *>(y);

  rmsNormKernelWarpFloat4<hiddenDim, threadsPerBlock>
      <<<numTokens, threadsPerBlock>>>(x_, w_, eps, y_);
}
```

위 kernel의 결과는 다음과 같다.

```shell
Latency = 2.80475 ms
Bandwidth = 3062.63 GB/s
% of max = 92.8071 %
```

## 결론

Reduction의 작동 원리를 이해하고 있다면 고성능 `RMSNorm` 연산 kernel을 구현하는 일은 어렵지 않다는 것을 보았다. 더 나아간 최적화 기회를 발견했다면 기꺼이 의견을 듣고 싶다. 나를 놀라게 한 점은 `#pragma unroll`을 사용해도 성능에 긍정적인 영향이 없었다는 것이다. 이 블로그 글이 마음에 들었다면 LinkedIn(https://www.linkedin.com/in/simon-veitner-174a681b6/)에서 나와 연결해 CUDA나 다른 머신러닝 시스템에 대한 아이디어를 교류하면 좋겠다. 위 결과의 모든 재현 코드는 내 Github(https://github.com/simveit/effective_rms_norm)에서 찾을 수 있다.




