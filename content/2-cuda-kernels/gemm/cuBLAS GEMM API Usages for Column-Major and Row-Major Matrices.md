
# column major와 row major 행렬에서 cuBLAS GEMM API를 사용하는 방법

## 서론

cuBLAS GEMM API는 입력 및 출력 행렬의 저장 형식에 대해 매우 엄격한 요구사항을 가진다. 모든 행렬이 column major 형식으로 저장되어 있다면 cuBLAS GEMM API를 직접 사용할 수 있다. 그러나 일부 행렬이 row major 형식으로 저장되어 있다면, 이런 행렬 곱에 대해 cuBLAS GEMM API의 인자를 설정할 때 실수할 수 있다.

이 블로그 글에서는 행렬 전치와 column major 저장 사이의 관계, 그리고 서로 다른 상황에서 cuBLAS GEMM API를 어떻게 사용해야 하는지 논의한다.

## cuBLAS GEMM

### cuBLAS GEMM API

cuBLAS 단정밀도 GEMM API 선언은 다음과 같다.

```c++
cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float *alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                           const float *beta,
                           float *C, int ldc)
```

### column major 행렬의 cuBLAS GEMM

이 함수는 일반 행렬-행렬 곱을 수행한다.

$$C = \alpha \text{op}(A)\text{op}(B) + \beta C$$

여기서 $\alpha$와 $\beta$는 scalar이고, $A$, $B$, $C$는 column major 형식으로 저장된 행렬이다. $\text{op}(A)$의 차원은 $m \times k$, $\text{op}(B)$의 차원은 $k \times n$, $C$의 차원은 $m \times n$이다. 행렬 $A$의 경우:

![](img/blog-repost-cublas-gemm-api-usages-for-column-major-and-row-major-f6fbfd0d/001.png)

### cuBLAS GEMM과 row major 행렬

그렇다면 일부 행렬이 row major 형식으로 저장되어 있으면 어떻게 될까? 몇 가지 예를 보자.

$m' \times k'$ 행렬 $A'$가 row major 형식으로 저장되어 있고, $k' \times n'$ 행렬 $B'$와 $m' \times n'$ 행렬 $C'$는 column major 형식으로 저장되어 있다고 가정하자. $A'$의 전치, 즉 $k' \times m'$ 행렬 $A'^T$는 column major 형식으로 저장되며, 원래 row major 형식으로 저장된 $A'$와 동등하다. 하지만 cuBLAS로 일반 행렬-행렬 곱을 수행하려면 $A'^T$를 $A'$로 전치해야 한다. 이 경우 **transa = CUBLAS_OP_T**, **transb = CUBLAS_OP_N**, $m = m'$, $n = n'$, $k = k'$, $A = A'$, $B = B'$, $C = C'$다.

$m' \times k'$ 행렬 $A'$와 $k' \times n'$ 행렬 $B'$가 column major 형식으로 저장되어 있고, $m' \times n'$ 행렬 $C'$가 row major 형식으로 저장되어 있다고 가정하자. 이 경우 cuBLAS API로 $C'$를 전치할 수 없다.

공식 안의 행렬 $C$를 일반 행렬-행렬 곱 수행 전에 먼저 전치할 수 있다는 점에 주목하자.

$$C'^T = \alpha(\text{op}(A)\text{op}(B))^T + \beta C'^T$$
$$= \alpha\text{op}(B)^T \text{op}(A)^T + \beta C'^T$$
$$= \alpha\text{op}(B^T) \text{op}(A^T) + \beta C'^T$$

따라서 $B^T$, $A^T$, $C^T$가 column major 형식으로 저장되어 있다면, 기존 cuBLAS API로도 일반 행렬-행렬 곱을 수행할 수 있다.

이 경우 $C'$의 전치, 즉 $n' \times m'$ 행렬 $C'^T$는 column major 형식으로 저장되며, 원래 row major 형식으로 저장된 $C'$와 동등하다. 또한 행렬 $A'$와 $B'$도 전치해야 한다. $A'$의 전치, 즉 $k' \times m'$ 행렬 $A'^T$는 row major 형식으로 저장되며, 원래 column major 형식으로 저장된 $A'$와 동등하다. $B'$의 전치, 즉 $n' \times k'$ 행렬 $B'^T$는 row major 형식으로 저장되며, 원래 column major 형식으로 저장된 $B'$와 동등하다. 이 경우 **transa = CUBLAS_OP_T**, **transb = CUBLAS_OP_T**, $m = n'$, $n = m'$, $k = k'$, $A = B'$, $B = A'$, $C = C'$다.

## 결론

cuBLAS API를 사용해 행렬 곱 $C' = \alpha A'B' + \beta C'$를 수행하고 싶다고 가정하자. 여기서 $A'$, $B'$, $C'$는 각각 $m' \times k'$, $k' \times n'$, $m' \times n'$ 형상의 행렬이다. 아래 표는 행렬 $A'$, $B'$, $C'$의 전치와 column major 저장 사이의 관계, 그리고 cuBLAS API를 어떻게 사용해야 하는지를 요약한다.

| $m' \times k'$ 행렬 $A'$ | $k' \times n'$ 행렬 $B'$ | $m' \times n'$ 행렬 $C'$ | **transa** | **transb** | **m** | **n** | **k** | **A** | **B** | **C** |
|---|---|---|---|---|---|---|---|---|---|---|
| column major | column major | column major | CUBLAS_OP_N | CUBLAS_OP_N | $m'$ | $n'$ | $k'$ | $A'$ | $B'$ | $C'$ |
| row major | column major | column major | CUBLAS_OP_T | CUBLAS_OP_N | $m'$ | $n'$ | $k'$ | $A'$ | $B'$ | $C'$ |
| column major | row major | column major | CUBLAS_OP_N | CUBLAS_OP_T | $m'$ | $n'$ | $k'$ | $A'$ | $B'$ | $C'$
| row major | row major | column major | CUBLAS_OP_T | CUBLAS_OP_T | $m'$ | $n'$ | $k'$ | $A'$ | $B'$ | $C'$ |
| column major | column major | row major | CUBLAS_OP_T | CUBLAS_OP_T | $n'$ | $m'$ | $k'$ | $B'$ | $A'$ | $C'$ |
| row major | column major | row major | CUBLAS_OP_T | CUBLAS_OP_N | $n'$ | $m'$ | $k'$ | $B'$ | $A'$ | $C'$ |
| column major | row major | row major | CUBLAS_OP_N | CUBLAS_OP_T | $n'$ | $m'$ | $k'$ | $B'$ | $A'$ | $C'$ |
| row major | row major | row major | CUBLAS_OP_N | CUBLAS_OP_N | $n'$ | $m'$ | $k'$ | $B'$ | $A'$ | $C'$ */


## References
- https://leimao.github.io/blog/cuBLAS-Transpose-Column-Major-Relationship/ 
- `cublas<t>gemm()`(https://docs.nvidia.com/cuda/archive/12.6.2/cublas/#cublas-t-gemm)
- `cuBLAS GEMM Transpose and Column-Major Application`(https://github.com/leimao/CUDA-GEMM-Optimization/blob/747e49b418bca18557581969c78333443055fd43/include/profile_utils.cuh#L113)

