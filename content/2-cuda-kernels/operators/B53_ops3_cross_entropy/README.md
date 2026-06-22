# ops(3): Cross Entropy의 CUDA 구현

> 원문: https://zhuanlan.zhihu.com/p/695594396

**목차**
- 1. Cross Entropy 순전파 구현
- 2. Cross Entropy 역전파 구현
- 참고 자료

딥러닝에서 Cross Entropy는 가장 흔한 손실 함수이고, 다중 클래스 분류에선 softmax와 짝지어 쓰입니다. 그래서 softmax 다음 차례로 Cross Entropy를 구현해 봅니다. Cross Entropy의 원리와 계산 과정은 이전 글에서 상세히 다뤘으므로 자세한 반복은 생략합니다.

![Cross Entropy Loss 병렬화](images/img_001.png)
*紫氣東來: Cross Entropy Loss 병렬화 방안 (43 추천)*

## 1. Cross Entropy 순전파 구현

수식:

```
cross_entropy = − Σₖ (pₖ · log qₖ)
```

`p`는 정답 라벨(one-hot), `q`는 보통 softmax 후의 예측 확률입니다. 학습 중에는 `pₘ_label = 1`이고 나머지는 0이므로 식이 단순화됩니다.

```
cross_entropy = − log qₘ
```

### 1.1 단순 구현

식 그대로 따라가면 됩니다. 입력 `probs`는 softmax 후 확률, 출력 `losses`는 token 레벨입니다.

```cpp
// CPU
void crossentropy_forward_cpu(float* losses,
                              const float* probs, const int* targets,
                              int B, int T, int V) {
    // losses: (B,T) 위치별 손실
    // probs : (B,T,V) 확률
    // targets: (B,T) 정답 인덱스
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* probs_bt = probs + b * T * V + t * V;
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}
```

CUDA는 `[B, T]` 차원에서 병렬화하면 끝:

```cpp
__global__ void crossentropy_forward_kernel1(float* losses,
                                             const float* probs, const int* targets,
                                             int B, int T, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * T) {
        int b = i / T;
        int t = i % T;
        const float* probs_bt = probs + b * T * V + t * V;
        int ix = targets[b * T + t];
        losses[b * T + t] = -logf(probs_bt[ix]);
    }
}
```

계산이 워낙 단순해서 보통 softmax와 함께 처리되므로 추가 최적화는 생략합니다.

```
block_size   32 | time 0.0032 ms | per token 0.39 ns
block_size   64 | time 0.0031 ms | per token 0.38 ns
block_size  128 | time 0.0031 ms | per token 0.38 ns
block_size  256 | time 0.0031 ms | per token 0.38 ns
block_size  512 | time 0.0032 ms | per token 0.39 ns
block_size 1024 | time 0.0037 ms | per token 0.45 ns
```

## 2. Cross Entropy 역전파 구현

Cross Entropy 수식을 사용해 역전파를 유도합니다. `zᵢ = logitsᵢ`, `S(zᵢ) = probsᵢ`(softmax 결과). 손실은:

```
Loss = − ln S(zᵢ)
```

모든 `zⱼ` (j = 1..n)에 대해 미분:

```
∂Loss/∂zⱼ = − (1/S(zᵢ)) · ∂S(zᵢ)/∂zⱼ
```

경우를 나눠 풀면, `i = j`일 때:

```
∂Loss/∂zⱼ = − (1/S(zᵢ)) · S(zᵢ)(1 − S(zᵢ))
           = S(zᵢ) − 1
           = S(zⱼ) − 1
```

`i ≠ j`일 때:

```
∂Loss/∂zⱼ = − (1/S(zᵢ)) · ( − S(zᵢ) · S(zⱼ) )
           = S(zⱼ)
           = S(zⱼ) − 0
```

마지막 `S(zⱼ) − 0`로 적은 이유는 벡터화 식으로 묶었을 때 일관되어 보이게 하기 위함입니다.

### 2.1 단순 구현

CPU 구현:

```cpp
void crossentropy_softmax_backward_cpu(float* dlogits,
                                       const float* dlosses, const float* probs,
                                       const int* targets,
                                       int B, int T, int V) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * V + t * V;
            const float* probs_bt = probs + b * T * V + t * V;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            for (int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = (i == ix) ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}
```

CUDA는 위 로직을 B·T·V 차원으로 병렬화:

```cpp
__global__ void crossentropy_softmax_backward_kernel1(float* dlogits,
                                                      const float* dlosses, const float* probs,
                                                      const int* targets,
                                                      int B, int T, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * T * V) {
        int b = i / (T * V);
        int t = (i / V) % T;
        int v = i % V;
        float* dlogits_bt = dlogits + b * T * V + t * V;
        const float* probs_bt = probs + b * T * V + t * V;
        float dloss = dlosses[b * T + t];
        int ix = targets[b * T + t];
        float p = probs_bt[v];
        float indicator = (v == ix) ? 1.0f : 0.0f;
        dlogits_bt[v] += (p - indicator) * dloss;
    }
}
```

성능:

```
block_size   32 | time 20.2376 ms | per token 2.47 µs
block_size   64 | time 10.0498 ms | per token 1.23 µs
block_size  128 | time  6.2755 ms | per token 0.77 µs
block_size  256 | time  6.2235 ms | per token 0.76 µs
block_size  512 | time  6.2832 ms | per token 0.77 µs
block_size 1024 | time  6.5979 ms | per token 0.81 µs
```

코드는 [crossentropy_forward.cu](https://github.com/ifromeast/cuda_learning/blob/main/04_transformer/ops/crossentropy_forward.cu) / [crossentropy_softmax_backward.cu](https://github.com/ifromeast/cuda_learning/blob/main/04_transformer/ops/crossentropy_softmax_backward.cu) 참고.

## 참고 자료

1. https://github.com/karpathy/llm.c/blob/master/dev/cuda/crossentropy_forward.cu
2. https://github.com/karpathy/llm.c/blob/master/dev/cuda/crossentropy_softmax_backward.cu
3. Cross Entropy Loss 병렬화 방안

> 夢回人遠許多愁, 只在梨花風雨處 — 辛棄疾 《玉樓春》
