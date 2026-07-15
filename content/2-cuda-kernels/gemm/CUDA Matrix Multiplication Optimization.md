# CUDA 행렬 곱셈 최적화

## 소개

범용 행렬 곱셈(GEMM)은 선형대수의 기본 연산이다. 또한 머신러닝과 딥러닝 등 많은 과학 계산 응용에서 매우 중요한 연산이다.

이 글에서는 CUDA 를 사용해 NVIDIA GPU 상의 FP32 GEMM 성능을 최적화하는 방법과, NVIDIA Tensor Core 를 사용해 FP32 GEMM 최적화를 FP16 GEMM 으로 확장하는 방법을 다룬다.

## 범용 행렬 곱셈

GEMM 연산은 $D = AB + C$ 를 계산한다. 여기서 $D \in \mathbb{R}^{m \times n}$, $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$, $C \in \mathbb{R}^{m \times n}$ 이다. 컴퓨터 프로그램에서 일반적으로 $A$ 와 $B$ 는 상수 입력 행렬이고, $C$ 는 출력 행렬 $D$ 로 덮어써진다.

이 구현에서는 모든 행렬 $A$, $B$, $C$, $D$ 가 메모리에 행 우선(row-major)으로 저장된다고 가정하며, FP32 행렬의 경우 선행 차원(leading dimension)을 64 바이트로, FP16 행렬의 경우 32 바이트로 패딩한다.

## 비병합(non-coalesced) 메모리 접근을 사용하는 단순 구현

단순 구현은 2D 블록을 사용하며, 각 스레드가 출력 행렬의 한 원소를 계산하는 것을 담당한다. 구체적으로, 전역 스레드 인덱스가 $(t_m, t_n)$ 인 각 스레드(여기서 $t_m \in [1, m]$, $t_n \in [1, n]$)는 다음을 계산한다.

$$D_{t_m,t_n} = \sum_{t_k=1}^{k} A_{t_m,t_k} B_{t_k,t_n} + C_{t_m,t_n}.$$

아래 코드 조각은 단순 구현을 보여준다.

```c++
template <typename T>
__global__ void gemm_v00(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // Compute the row and column of C that this thread is responsible for.
    size_t const C_row_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_col_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // Each thread compute
    // C[C_row_idx, C_col_idx] = alpha * A[C_row_idx, :] * B[:, C_col_idx] +
    // beta * C[C_row_idx, C_col_idx].
    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t k_idx{0U}; k_idx < k; ++k_idx)
        {
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
        }
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

template <typename T>
void launch_gemm_kernel_v00(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(m) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(n) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v00<T><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B,
                                                     ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
```
그러나 단순 알고리즘 자체의 다른 단점 외에도, 이 구현에는 한 가지 주요 문제가 있다. 바로 전역 메모리를 읽고 쓸 때의 비병합 메모리 접근이다. 이 구현에서는 부주의하게 빠른 스레드 인덱스를 사용해 행렬 $A$ 와 $C$ 의 행을 인덱싱했기 때문에, 같은 warp 안의 스레드들이 메모리에 행 우선으로 저장된 행렬 $A$ 의 같은 열에서 원소를 읽게 된다. 읽기가 완전히 불연속이므로 비병합 메모리 접근이 발생한다. warp 가 행렬 $C$ 의 원소를 덮어쓸 때도 같은 문제가 발생한다. 같은 warp 안의 스레드들이 행렬 $B$ 의 같은 원소를 읽는 것은 브로드캐스트 메모리 접근을 유발하며, 이는 앞의 부주의한 인덱싱의 영향을 받지 않는다.

이 FP32 GEMM 구현의 성능은 NVIDIA GeForce RTX 3090 GPU 에서 0.27 TFLOPS 에 불과하며, 성능이 매우 나쁘다.

## 병합(coalesced) 메모리 접근을 사용하는 단순 구현

비병합 메모리 접근을 해결하는 방법은, 빠른 스레드 인덱스를 사용해 메모리에 행 우선으로 저장된 행렬의 행을 인덱싱하는 것이다. 이렇게 하면 같은 warp 안의 스레드들이 행렬의 같은 행에 있는 원소를 읽거나 덮어쓸 때 병합이 이루어진다. 이 구현에서는 커널 함수 안에서 빠른 스레드 인덱스와 느린 스레드 인덱스를 교환하기만 하면 된다.

아래 코드 조각은 병합 메모리 접근을 사용하는 단순 구현을 보여준다.

```c++
template <typename T>
__global__ void gemm_v01(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // Compute the row and column of C that this thread is responsible for.
    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // Each thread compute
    // C[C_row_idx, C_col_idx] = alpha * A[C_row_idx, :] * B[:, C_col_idx] +
    // beta * C[C_row_idx, C_col_idx].
    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t k_idx{0U}; k_idx < k; ++k_idx)
        {
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
        }
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

template <typename T>
void launch_gemm_kernel_v01(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_v01<T><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B,
                                                     ldb, *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
```
이제 이 수정 덕분에 같은 warp 안의 스레드들은 메모리에 행 우선으로 저장된 행렬 $B$ 의 같은 행에서 원소를 읽게 되어 병합 메모리 접근이 이루어진다. warp 가 행렬 $C$ 의 원소를 덮어쓸 때도 마찬가지이다. 같은 warp 안의 스레드들이 행렬 $A$ 의 같은 원소를 읽는 것은 브로드캐스트 메모리 접근을 유발한다. 따라서 이 구현은 비병합 메모리 접근 구현보다 성능이 훨씬 좋을 것이다.

이 FP32 GEMM 구현의 성능은 NVIDIA GeForce RTX 3090 GPU 에서 1.72 TFLOPS 가 되어, 이전 구현보다 훨씬 좋아졌다. 그러나 GPU 의 이론적 최대 성능이 35.58 TFLOPS 임을 고려하면, 이 구현의 성능은 여전히 매우 나쁘다.

## 2D 블록 타일링을 사용하는 구현

앞선 구현들은 전역 메모리에 빈번하게 접근하기 때문에 GEMM 구현이 메모리 바운드가 되었다. 공유 메모리 접근은 전역 메모리 접근보다 훨씬 빠르므로, 성능을 높이기 위해 공유 메모리를 사용해 입력 행렬 $A$ 와 $B$ 를 캐시하여 데이터 재사용을 구현할 수 있다.

그러나 공유 메모리 크기가 제한적이므로 입력 행렬 $A$ 와 $B$ 전체를 공유 메모리에 캐시할 수는 없다. 대신, $A$ 와 $B$ 의 2D 타일을 공유 메모리에 캐시하고, 그 2D 타일을 사용해 출력 행렬 $D$ 의 2D 타일을 계산할 수 있다. 그런 다음 $A$ 와 $B$ 의 다음 2D 타일을 공유 메모리에 로드하고 $D$ 의 다음 2D 타일을 계산한다.

수학적으로, GEMM 연산 $D = AB + C$(여기서 $D \in \mathbb{R}^{m \times n}$, $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$, $C \in \mathbb{R}^{m \times n}$)가 주어지면, 행렬은 더 작은 행렬들로 분할할 수 있다.

$$A = \begin{bmatrix}
A_{1,1}^{d_m \times d_{bk}} & A_{1,2}^{d_m \times d_{bk}} & \cdots & A_{1,k/d_{bk}}^{d_m \times d_{bk}} \\
A_{2,1}^{d_m \times d_{bk}} & A_{2,2}^{d_m \times d_{bk}} & \cdots & A_{2,k/d_{bk}}^{d_m \times d_{bk}} \\
\vdots & \vdots & \ddots & \vdots \\
A_{m/d_m,1}^{d_m \times d_{bk}} & A_{m/d_m,2}^{d_m \times d_{bk}} & \cdots & A_{m/d_m,k/d_{bk}}^{d_m \times d_{bk}}
\end{bmatrix}$$

$$B = \begin{bmatrix}
B_{1,1}^{d_{bk} \times d_n} & B_{1,2}^{d_{bk} \times d_n} & \cdots & B_{1,n/d_n}^{d_{bk} \times d_n} \\
B_{2,1}^{d_{bk} \times d_n} & B_{2,2}^{d_{bk} \times d_n} & \cdots & B_{2,n/d_n}^{d_{bk} \times d_n} \\
\vdots & \vdots & \ddots & \vdots \\
B_{k/d_{bk},1}^{d_{bk} \times d_n} & B_{k/d_{bk},2}^{d_{bk} \times d_n} & \cdots & B_{k/d_{bk},n/d_n}^{d_{bk} \times d_n}
\end{bmatrix}$$

$$C = \begin{bmatrix}
C_{1,1}^{d_m \times d_n} & C_{1,2}^{d_m \times d_n} & \cdots & C_{1,n/d_n}^{d_m \times d_n} \\
C_{2,1}^{d_m \times d_n} & C_{2,2}^{d_m \times d_n} & \cdots & C_{2,n/d_n}^{d_m \times d_n} \\
\vdots & \vdots & \ddots & \vdots \\
C_{m/d_m,1}^{d_m \times d_n} & C_{m/d_m,2}^{d_m \times d_n} & \cdots & C_{m/d_m,n/d_n}^{d_m \times d_n}
\end{bmatrix}$$

$$D = \begin{bmatrix}
D_{1,1}^{d_m \times d_n} & D_{1,2}^{d_m \times d_n} & \cdots & D_{1,n/d_n}^{d_m \times d_n} \\
D_{2,1}^{d_m \times d_n} & D_{2,2}^{d_m \times d_n} & \cdots & D_{2,n/d_n}^{d_m \times d_n} \\
\vdots & \vdots & \ddots & \vdots \\
D_{m/d_m,1}^{d_m \times d_n} & D_{m/d_m,2}^{d_m \times d_n} & \cdots & D_{m/d_m,n/d_n}^{d_m \times d_n}
\end{bmatrix}$$

$D$ 안의 각 작은 행렬은 여러 개의 작은 행렬 곱셈과 누산을 통해 계산된다.

$$D_{b_m,b_n}^{d_m \times d_n} = \sum_{b_k=1}^{k/d_{bk}} A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n} + C_{b_m,b_n}^{d_m \times d_n}$$
이 구현에서 블록 인덱스가 $(b_m, b_n)$ 인 각 2D 블록(여기서 $b_m \in [1, m/d_{bm}]$, $b_n \in [1, n/d_{bn}]$)은 하나의 작은 행렬 $D_{b_m,b_n}^{d_m \times d_n}$ 을 계산하는 것을 담당한다. 공유 메모리는 각각 크기가 $d_{bm} \times d_{bk}$ 와 $d_{bk} \times d_{bn}$ 인 $A$ 와 $B$ 의 2D 타일을 캐시하는 데 사용된다. $A$ 의 2D 타일은 $(b_m, b_k)$ 로 인덱싱되며(여기서 $b_m \in [1, m/d_{bm}]$, $b_k \in [1, k/d_{bk}]$), $B$ 의 2D 타일은 $(b_k, b_n)$ 으로 인덱싱된다(여기서 $b_k \in [1, k/d_{bk}]$, $b_n \in [1, n/d_{bn}]$). 캐시와 작은 행렬 곱셈 계산 과정은 작은 행렬 $D_{b_m,b_n}^{d_m \times d_n}$ 전체가 누산 완료될 때까지 $k/d_{bk}$ 번 반복된다.

이전 구현과 유사하게, 각 블록은 작은 행렬 $D_{b_m,b_n}^{d_m \times d_n}$ 을 계산하기 위해 $d_{bm} \times d_{bn}$ 개의 스레드가 필요하다. 블록 내 스레드 인덱스가 $(t_m, t_n)$ 인 각 스레드(여기서 $t_m \in [1, d_{bm}]$, $t_n \in [1, d_{bn}]$)는 작은 행렬의 한 원소를 계산하는 것을 담당한다.

$$\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{t_m,t_n} = \left(\sum_{b_k=1}^{k/d_{bk}} A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n} + C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m,t_n}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_m,t_n} + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m,t_n}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(\sum_{t_k=1}^{d_{bk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{t_m,t_k} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_k,t_n}\right) + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m,t_n}$$

아래 코드 조각은 2D 블록 타일링을 사용하는 구현을 보여준다.

```c++
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS, size_t BLOCK_TILE_SKEW_SIZE_X = 0U, size_t BLOCK_TILE_SKEW_SIZE_K = 0U>
__device__ void load_data_to_shared_memory(T const* A, size_t lda,
                                           T const* B, size_t ldb,
                                           T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
                                           T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
                                           size_t thread_block_tile_idx,
                                           size_t thread_linear_idx,
                                           size_t m, size_t n,
                                           size_t k)
{
    // DRAM 의 행렬 A 데이터를 공유 메모리의 A_thread_block_tile 로 로드한다
#pragma unroll
    for (size_t load_idx{0U};
         load_idx <
         (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) /
             NUM_THREADS;
         ++load_idx)
    {
        // 공유 메모리 타일 안에서의 행/열 인덱스를 계산한다
        size_t const A_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) /
            BLOCK_TILE_SIZE_K};
        size_t const A_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) %
            BLOCK_TILE_SIZE_K};
        // 전역 행렬 A 안에서의 행/열 인덱스를 계산한다
        size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                               A_thread_block_tile_row_idx};
        size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                               A_thread_block_tile_col_idx};

        // 경계 검사는 커널 성능을 어느 정도 떨어뜨릴 수 있으나
        // 서로 다른 모든 GEMM 구성에 대해 커널의 정확성을 보장한다
        T val{static_cast<T>(0)};
        if (A_row_idx < m && A_col_idx < k)
        {
            val = A[A_row_idx * lda + A_col_idx];
        }
        // 이 if 문은 커널 성능을 떨어뜨린다
        // 호스트 코드에 정적 단언(static assert)을 추가해 이 if 조건이 항상 true 임을 보장한다
        static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS ==
                      0U);
        // if (A_thread_block_tile_row_idx < BLOCK_TILE_SIZE_Y &&
        //     A_thread_block_tile_col_idx < BLOCK_TILE_SIZE_K)
        // {
        //     A_thread_block_tile[A_thread_block_tile_row_idx]
        //                        [A_thread_block_tile_col_idx] = val;
        // }
        A_thread_block_tile[A_thread_block_tile_row_idx]
                           [A_thread_block_tile_col_idx] = val;
    }
// DRAM 의 행렬 B 데이터를 공유 메모리의 B_thread_block_tile 로 로드한다
#pragma unroll
    for (size_t load_idx{0U};
         load_idx <
         (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) /
             NUM_THREADS;
         ++load_idx)
    {
        // 공유 메모리 타일 안에서의 행/열 인덱스를 계산한다
        size_t const B_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) /
            BLOCK_TILE_SIZE_X};
        size_t const B_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) %
            BLOCK_TILE_SIZE_X};
        // 전역 행렬 B 안에서의 행/열 인덱스를 계산한다
        size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                               B_thread_block_tile_row_idx};
        size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                               B_thread_block_tile_col_idx};

        // 경계 검사는 커널 성능을 어느 정도 떨어뜨릴 수 있으나
        // 서로 다른 모든 GEMM 구성에 대해 커널의 정확성을 보장한다
        T val{static_cast<T>(0)};
        if (B_row_idx < k && B_col_idx < n)
        {
            val = B[B_row_idx * ldb + B_col_idx];
        }
        // 이 if 문은 커널 성능을 떨어뜨린다
        // 호스트 코드에 정적 단언(static assert)을 추가해 이 if 조건이 항상 true 임을 보장한다
        static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS ==
                      0U);
        // if (B_thread_block_tile_row_idx < BLOCK_TILE_SIZE_K &&
        //     B_thread_block_tile_col_idx < BLOCK_TILE_SIZE_X)
        // {
        //     B_thread_block_tile[B_thread_block_tile_row_idx]
        //                        [B_thread_block_tile_col_idx] = val;
        // }
        B_thread_block_tile[B_thread_block_tile_row_idx]
                           [B_thread_block_tile_col_idx] = val;
    }
}

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K>
__global__ void gemm_v02(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // 블록당 스레드 수로 blockDim.x * blockDim.y 를 사용하는 것을 피한다
    // 이 값은 런타임 상수여서 컴파일러가 이를 기준으로 루프 언롤링을 최적화할 수 없기 때문이다
    // 대신 컴파일 타임 상수를 사용한다
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

    // 이 스레드가 담당하는 행렬 C 의 행/열 인덱스를 계산한다
    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // 데이터 재사용을 위해 A 와 B 의 타일을 공유 메모리에 캐시한다
    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    // 처리해야 할 스레드 블록 타일의 개수를 계산한다
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};

    // 누산 합을 0 으로 초기화한다
    T sum{static_cast<T>(0)};
    // 각 스레드 블록 타일을 순회한다
    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        // 현재 타일의 데이터를 공유 메모리로 로드한다
        load_data_to_shared_memory<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                                   BLOCK_TILE_SIZE_K, NUM_THREADS>(
            A, lda, B, ldb, A_thread_block_tile, B_thread_block_tile,
            thread_block_tile_idx, thread_linear_idx, m, n, k);
        __syncthreads();

        // 현재 타일의 행렬 곱셈을 계산한다
#pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            // 이렇게 하면 성능이 2 TOPS 에 이른다
            // blockDim.x = blockDim.y = 32 라고 가정한다
            // 실제로 하나의 warp 에 대해, 한 번의 반복에서 우리는 A_thread_block_tile 의
            // 공유 메모리 같은 위치에서 값을 읽어 브로드캐스트가 발생하고, B_thread_block_tile 에서도
            // bank 충돌이 없는 32 개의 값을 읽는다. 그렇더라도 모든 값을 공유 메모리에서 읽어야 하므로,
            // 결과적으로 공유 메모리 명령이 매우 밀집되게 실행되면서 단지 간단한 산술 명령으로
            // 소량의 값을 계산하게 되어 비효율적이다
            sum += A_thread_block_tile[threadIdx.y][k_i] *
                   B_thread_block_tile[k_i][threadIdx.x];
        }
        __syncthreads();
    }
    // 최종 결과를 출력 행렬 C 에 기록한다(경계 검사)
    if (C_row_idx < m && C_col_idx < n)
    {
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

template <typename T>
void launch_gemm_kernel_v02(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    // 블록 타일 크기는 자유롭게 조정할 수 있다
    // 알고리즘의 정확성은 항상 보장되어야 한다
    constexpr unsigned int BLOCK_TILE_SIZE_X{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
    constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    // 정적 단언으로 타일 크기가 스레드 수와 호환됨을 보장한다
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS == 0U);
    static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);
    // 블록 차원을 설정한다
    dim3 const block_dim{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, 1U};
    // 그리드 차원을 설정한다
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};
    // GEMM 커널을 실행한다
    gemm_v02<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                              *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
```

이 FP32 GEMM 구현의 성능은 NVIDIA GeForce RTX 3090 GPU 에서 2.66 TFLOPS 가 되어, 이전 구현보다 훨씬 좋아졌다. 그러나 여전히 GPU 의 이론적 최대 성능에는 크게 못 미친다.

이 구현의 문제는 공유 메모리에 빈번하게 접근한다는 점이다. 공유 메모리 접근이 전역 메모리 접근보다 훨씬 빠르더라도, 단지 간단한 산술 명령으로 소량의 값을 계산하기 위해 공유 메모리 명령이 매우 밀집되게 실행되므로 비효율적이다. 따라서 이 구현의 성능은 여전히 메모리 대역폭에 의해 제한되며, 이번에는 그 대상이 공유 메모리이다.

## 2D 블록 타일링과 1D 스레드 타일링을 사용하는 구현

성능을 더 높이기 위해, 입력 행렬 $A$ 와 $B$ 의 더 작은 타일을 스레드의 레지스터에 추가로 캐시하여 공유 메모리 대역폭 문제를 완화할 수 있다. 이번에는 각 스레드가 출력 행렬 $D$ 의 한 원소가 아니라 하나의 작은 타일을 계산하는 것을 담당한다. 레지스터가 가장 빠른 접근 방식이므로, 이 구현의 성능은 이전 구현보다 훨씬 좋을 것이다.

먼저 행렬 $B$ 의 데이터만 공유 메모리에서 레지스터로 캐시한다. 블록 내 스레드 인덱스가 $(t_m, t_n)$ 인 각 스레드(여기서 $t_m \in [1, d_{bm}/d_{tm}]$, $t_n \in [1, d_{bn}]$)는 이제 작은 행렬의 $d_{tm}$ 개 원소를 계산하는 것을 담당한다. 여기서 $d_{tm}$ 은 스레드 타일 크기이다.

$$\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{t_m:t_m+d_{tm},t_n} = \left(\sum_{b_k=1}^{k/d_{bk}} A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n} + C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m:t_m+d_{tm},t_n}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_m:t_m+d_{tm},t_n} + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m:t_m+d_{tm},t_n}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(\sum_{t_k=1}^{d_{bk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{t_m:t_m+d_{tm},t_k} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_k,t_n}\right) + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m:t_m+d_{tm},t_n}$$

스레드 타일링이 없던 이전 구현에서는 작은 행렬의 한 원소를 계산하기 위해, 공유 메모리에 캐시된 행렬 $A$ 에서 $d_{bk}$ 개의 값을, 공유 메모리에 캐시된 행렬 $B$ 에서 $d_{bk}$ 개의 값을 읽어야 했다. 총 $2d_k$ 개의 값을 공유 메모리에서 읽어야 한다.

이제 1D 스레드 타일링을 사용하면, 작은 행렬의 $d_{tm}$ 개 원소를 계산하기 위해 공유 메모리에 캐시된 행렬 $A$ 에서 $d_{bk} \times d_{tm}$ 개의 값을, 공유 메모리에 캐시된 행렬 $B$ 에서 $d_{bk}$ 개의 값만 읽으면 된다. 구체적으로, 각 내부 루프에서 $\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_k,t_n}$ 이 레지스터에 캐시되어 $d_{tm}$ 번 재사용된다. 총 $d_{bk} \times d_{tm} + d_{bk}$ 개의 값을 공유 메모리에서 읽어야 한다. 평균적으로 작은 행렬의 한 원소를 계산하기 위해 공유 메모리에서 $d_{bk} + d_{bk}/d_{tm}$ 개의 값을 읽어야 한다.

$d_{bk} + d_{bk}/d_{tm} < 2d_k$ 이므로 공유 메모리 접근 빈도가 줄어들어, 공유 메모리 대역폭 문제가 완화된다.

아래 코드 조각은 2D 블록 타일링과 1D 스레드 타일링을 사용하는 구현을 보여준다.

```c++
// 2D 블록 타일링과 1D 스레드 타일링을 사용하는 GEMM 커널 템플릿
// T: 데이터 타입
// BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K: 블록 타일 크기
// THREAD_TILE_SIZE_Y: 스레드 타일 크기(각 스레드가 처리하는 행 수)
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t THREAD_TILE_SIZE_Y>
__global__ void gemm_v03(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // 블록당 스레드 수로 blockDim.x * blockDim.y 를 사용하는 것을 피한다
    // 이 값은 런타임 상수여서 컴파일러가 이를 기준으로 루프 언롤링을 최적화할 수 없기 때문이다
    // 대신 컴파일 타임 상수를 사용한다
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
                                 THREAD_TILE_SIZE_Y};
    // 스레드의 선형 인덱스를 계산한다
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

    // 데이터 재사용을 위해 A 와 B 의 타일을 공유 메모리에 캐시한다
    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    // 처리해야 할 스레드 블록 타일의 개수를 계산한다
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};

    // 블록 안의 각 스레드는 THREAD_TILE_SIZE_Y 개의 출력 값을 처리한다
    // 구체적으로 이 값들은 다음에 대응한다
    // C[blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.x / BLOCK_TILE_SIZE_X *
    // THREAD_TILE_SIZE_Y : blockIdx.y * BLOCK_TILE_SIZE_Y + (threadIdx.x /
    // BLOCK_TILE_SIZE_X + 1) * THREAD_TILE_SIZE_Y][blockIdx.x *
    // BLOCK_TILE_SIZE_X + threadIdx.x % BLOCK_TILE_SIZE_X]
    T C_thread_results[THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};

    // 모든 스레드 블록 타일을 순회한다
    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        // 데이터를 전역 메모리에서 공유 메모리로 로드한다
        load_data_to_shared_memory<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                                   BLOCK_TILE_SIZE_K, NUM_THREADS>(
            A, lda, B, ldb, A_thread_block_tile, B_thread_block_tile,
            thread_block_tile_idx, thread_linear_idx, m, n, k);
        // 모든 스레드를 동기화하여 공유 메모리 데이터 로드가 완료되었음을 보장한다
        __syncthreads();

        // K 차원을 순회하며 행렬 곱셈을 계산한다
#pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            size_t const B_thread_block_tile_row_idx{k_i};
            // 공유 메모리 접근 압력을 줄이기 위해 B_val 을 레지스터에 캐시한다
            T const B_val{
                B_thread_block_tile[B_thread_block_tile_row_idx]
                                   [thread_linear_idx % BLOCK_TILE_SIZE_X]};
            // 스레드 타일의 각 행을 순회한다
#pragma unroll
            for (size_t thread_tile_row_idx{0U};
                 thread_tile_row_idx < THREAD_TILE_SIZE_Y;
                 ++thread_tile_row_idx)
            {
                // A 행렬의 공유 메모리 안에서의 행 인덱스를 계산한다
                size_t const A_thread_block_tile_row_idx{
                    thread_linear_idx / BLOCK_TILE_SIZE_X * THREAD_TILE_SIZE_Y +
                    thread_tile_row_idx};
                // A 행렬의 공유 메모리 안에서의 열 인덱스를 계산한다
                size_t const A_thread_block_tile_col_idx{k_i};
                // 공유 메모리에서 A 행렬의 값을 읽는다
                T const A_val{A_thread_block_tile[A_thread_block_tile_row_idx]
                                                 [A_thread_block_tile_col_idx]};
                // 곱셈-누산 연산을 수행한다
                C_thread_results[thread_tile_row_idx] += A_val * B_val;
            }
        }
        // 모든 스레드를 동기화하여 다음 반복을 준비한다
        __syncthreads();
    }

    // 결과를 DRAM 에 기록한다
#pragma unroll
    for (size_t thread_tile_row_idx{0U};
         thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx)
    {
        // 출력 행렬 C 의 행 인덱스를 계산한다
        size_t const C_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                               thread_linear_idx / BLOCK_TILE_SIZE_X *
                                   THREAD_TILE_SIZE_Y +
                               thread_tile_row_idx};
        // 출력 행렬 C 의 열 인덱스를 계산한다
        size_t const C_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                               thread_linear_idx % BLOCK_TILE_SIZE_X};
        // 경계 검사 후 결과를 기록한다
        if (C_row_idx < m && C_col_idx < n)
        {
            C[C_row_idx * ldc + C_col_idx] =
                alpha * C_thread_results[thread_tile_row_idx] +
                beta * C[C_row_idx * ldc + C_col_idx];
        }
    }
}

// GEMM 커널 v03 의 실행 템플릿 함수
template <typename T>
void launch_gemm_kernel_v03(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    // 블록 타일 크기는 자유롭게 조정할 수 있다
    // 알고리즘의 정확성은 항상 보장되어야 한다
    constexpr unsigned int BLOCK_TILE_SIZE_X{64U};      // 블록 타일 X 차원 크기
    constexpr unsigned int BLOCK_TILE_SIZE_Y{64U};      // 블록 타일 Y 차원 크기
    constexpr unsigned int BLOCK_TILE_SIZE_K{8U};       // 블록 타일 K 차원 크기
    // 각 스레드는 C 행렬의 THREAD_TILE_SIZE_Y 개 값을 계산한다
    constexpr unsigned int THREAD_TILE_SIZE_Y{8U};      // 스레드 타일 크기
    // 블록당 스레드 수를 계산한다
    constexpr unsigned int NUM_THREADS_PER_BLOCK{
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y / THREAD_TILE_SIZE_Y};
    
    // 정적 단언으로 파라미터 구성의 정확성을 보장한다
    static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0U);
    
    // 블록 차원(1D 스레드 블록)
    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    // 그리드 차원
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    
    // GEMM 커널을 실행한다
    gemm_v03<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
             THREAD_TILE_SIZE_Y><<<grid_dim, block_dim, 0U, stream>>>(
        m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    // CUDA 오류를 검사한다
    CHECK_LAST_CUDA_ERROR();
}
```

이 FP32 GEMM 구현의 성능은 NVIDIA GeForce RTX 3090 GPU 에서 8.91 TFLOPS 가 되었다. 계속 진전을 이루고 있는 것으로 보인다.


## 2D 블록 타일링과 2D 스레드 타일링을 사용하는 구현

레지스터 개수가 성능 병목이 아니라면, 행렬 $A$ 와 $B$ 의 데이터를 공유 메모리에서 레지스터로 캐시하여 성능을 더 높일 수 있다. 블록 내 스레드 인덱스가 $(t_m, t_n)$ 인 각 스레드(여기서 $t_m \in [1, d_{bm}/d_{tm}]$, $t_n \in [1, d_{bn}/d_{tn}]$)는 이제 작은 행렬의 $d_{tm} \times d_{tn}$ 개 원소를 계산하는 것을 담당한다. 여기서 $d_{tm}$ 과 $d_{tn}$ 은 각각 행과 열의 스레드 타일 크기이다.

$$\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{t_m:t_m+d_{tm},t_n:t_n+d_{tn}} = \left(\sum_{b_k=1}^{k/d_{bk}} A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n} + C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m:t_m+d_{tm},t_n:t_n+d_{tn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_m:t_m+d_{tm},t_n:t_n+d_{tn}} + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m:t_m+d_{tm},t_n:t_n+d_{tn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(\sum_{t_k=1}^{d_{bk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{t_m:t_m+d_{tm},t_k} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_k,t_n:t_n+d_{tn}}\right) + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m:t_m+d_{tm},t_n:t_n+d_{tn}}$$

1D 스레드 타일링을 사용한 이전 구현에서는 작은 행렬의 한 원소를 계산하기 위해 평균적으로 공유 메모리에서 $d_{bk} + d_{bk}/d_{tm}$ 개의 값을 읽어야 했다.

이제 2D 스레드 타일링을 사용하면, 작은 행렬의 $d_{tm} \times d_{tn}$ 개 원소를 계산하기 위해 공유 메모리에서 $d_{bk} \times (d_{tm} + d_{tn})$ 개의 값만 읽으면 된다. 구체적으로, 각 내부 루프에서 $\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{t_m:t_m+d_{tm},t_k}$ 와 $\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_k,t_n:t_n+d_{tn}}$ 가 레지스터에 캐시되어 행렬 곱셈 $\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{t_m:t_m+d_{tm},t_k} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_k,t_n:t_n+d_{tn}}$ 계산에 재사용된다. 총 $d_{bk} \times (d_{tm} + d_{tn})$ 개의 값을 공유 메모리에서 읽어야 한다. 평균적으로 작은 행렬의 한 원소를 계산하기 위해 공유 메모리에서 $d_{bk}/d_{tm} + d_{bk}/d_{tn}$ 개의 값을 읽어야 한다.

$d_{bk}/d_{tm} + d_{bk}/d_{tn} < d_{bk} + d_{bk}/d_{tm}$ 이므로 공유 메모리 접근 빈도가 더욱 줄어들어, 공유 메모리 대역폭 문제가 한층 더 완화된다.

아래는 2D 스레드 타일링 구현을 설명하는 또 다른 방법이다.

수학적으로, 행렬 곱셈-누산 연산 $D_{b_m,b_n}^{d_m \times d_n} = \sum_{b_k=1}^{k/d_{bk}} A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n} + C_{b_m,b_n}^{d_m \times d_n}$(여기서 $D_{b_m,b_n} \in \mathbb{R}^{d_m \times d_n}$, $A_{b_m,b_k} \in \mathbb{R}^{d_m \times d_{bk}}$, $B_{b_k,b_n} \in \mathbb{R}^{d_{bk} \times d_n}$, $C_{b_m,b_n} \in \mathbb{R}^{d_m \times d_n}$)가 주어지면, 행렬은 더 작은 행렬들로 분할할 수 있다.

$$A_{b_m,b_k}^{d_m \times d_{bk}} = \begin{bmatrix}
\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{1,1}^{d_{tm} \times d_{tk}} & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{1,2}^{d_{tm} \times d_{tk}} & \cdots & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{1,d_{bk}/d_{tk}}^{d_{tm} \times d_{tk}} \\
\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{2,1}^{d_{tm} \times d_{tk}} & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{2,2}^{d_{tm} \times d_{tk}} & \cdots & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{2,d_{bk}/d_{tk}}^{d_{tm} \times d_{tk}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{d_m/d_{tm},1}^{d_{tm} \times d_{tk}} & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{d_m/d_{tm},2}^{d_{tm} \times d_{tk}} & \cdots & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{d_m/d_{tm},d_{bk}/d_{tk}}^{d_{tm} \times d_{tk}}
\end{bmatrix}$$

$$B_{b_k,b_n}^{d_{bk} \times d_n} = \begin{bmatrix}
\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{1,1}^{d_{tk} \times d_{tn}} & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{1,2}^{d_{tk} \times d_{tn}} & \cdots & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{1,d_n/d_{tn}}^{d_{tk} \times d_{tn}} \\
\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{2,1}^{d_{tk} \times d_{tn}} & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{2,2}^{d_{tk} \times d_{tn}} & \cdots & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{2,d_n/d_{tn}}^{d_{tk} \times d_{tn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{d_{bk}/d_{tk},1}^{d_{tk} \times d_{tn}} & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{d_{bk}/d_{tk},2}^{d_{tk} \times d_{tn}} & \cdots & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{d_{bk}/d_{tk},d_n/d_{tn}}^{d_{tk} \times d_{tn}}
\end{bmatrix}$$

$$C_{b_m,b_n}^{d_m \times d_n} = \begin{bmatrix}
\left(C_{b_m,b_n}^{d_m \times d_n}\right)_{1,1}^{d_{tm} \times d_{tn}} & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{1,2}^{d_{tm} \times d_{tn}} & \cdots & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{1,d_n/d_{tn}}^{d_{tm} \times d_{tn}} \\
\left(C_{b_m,b_n}^{d_m \times d_n}\right)_{2,1}^{d_{tm} \times d_{tn}} & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{2,2}^{d_{tm} \times d_{tn}} & \cdots & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{2,d_n/d_{tn}}^{d_{tm} \times d_{tn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(C_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{tm},1}^{d_{tm} \times d_{tn}} & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{tm},2}^{d_{tm} \times d_{tn}} & \cdots & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{tm},d_n/d_{tn}}^{d_{tm} \times d_{tn}}
\end{bmatrix}$$

$$D_{b_m,b_n}^{d_m \times d_n} = \begin{bmatrix}
\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{1,1}^{d_{tm} \times d_{tn}} & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{1,2}^{d_{tm} \times d_{tn}} & \cdots & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{1,d_n/d_{tn}}^{d_{tm} \times d_{tn}} \\
\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{2,1}^{d_{tm} \times d_{tn}} & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{2,2}^{d_{tm} \times d_{tn}} & \cdots & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{2,d_n/d_{tn}}^{d_{tm} \times d_{tn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{tm},1}^{d_{tm} \times d_{tn}} & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{tm},2}^{d_{tm} \times d_{tn}} & \cdots & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{tm},d_n/d_{tn}}^{d_{tm} \times d_{tn}}
\end{bmatrix}$$

$D_{b_m,b_n}^{d_m \times d_n}$ 안의 각 작은 행렬은 여러 개의 작은 행렬 곱셈과 누산을 통해 계산된다.

$$\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}} = \left(\sum_{b_k=1}^{k/d_{bk}} A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n} + C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}} + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(\sum_{t_k=1}^{d_{bk}/d_{tk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{t_m,t_k}^{d_{tm} \times d_{tk}} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{t_k,t_n}^{d_{tk} \times d_{tn}}\right) + \left(\left(C_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$$

이를 종합하면, 스레드 타일 $\left(\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$ 는 다음과 같이 계산할 수 있다.

$$\left(\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}} = \left(\sum_{b_k=1}^{k/d_{bk}} \left(\sum_{w_k=1}^{d_{bk}/d_{wk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right) + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(\sum_{w_k=1}^{d_{bk}/d_{wk}} \left(\sum_{t_k=1}^{d_{wk}/d_{tk}} \left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{t_m,t_k}^{d_{tm} \times d_{tk}} \left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{t_k,t_n}^{d_{tk} \times d_{tn}}\right) + \left(\left(C_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}\right)_{t_m,t_n}$$

이 구현에서는 스레드 타일링 알고리즘을 더 단순하게 만들기 위해 $d_{wk} = d_{tk}$ 로 설정한다.

아래 코드 조각은 2D 블록 타일링과 2D 스레드 타일링을 사용하는 구현을 보여준다.

```c++
// GEMM 커널 v04 버전
// 전역 메모리에서 병합 읽기/쓰기 접근을 수행한다
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t THREAD_TILE_SIZE_X,
          size_t THREAD_TILE_SIZE_Y>
__global__ void gemm_v04(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // 블록당 스레드 수로 blockDim.x * blockDim.y 를 사용하는 것을 피한다
    // 이 값은 런타임 상수여서 컴파일러가 이를 기준으로 루프 언롤링을 최적화할 수 없기 때문이다
    // 대신 컴파일 타임 상수를 사용한다
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
                                 (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y)};
    // 스레드의 선형 인덱스를 계산한다
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

    // 데이터 재사용을 위해 A 와 B 의 타일을 공유 메모리에 캐시한다
    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    // 처리해야 할 스레드 블록 타일의 개수를 계산한다
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};

    // 블록 안의 각 스레드는 THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X 개의 출력 값을 처리한다
    // 구체적으로 이 값들은 출력 행렬 C 안의 작은 직사각형 영역에 대응한다
    // C[blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.x / BLOCK_TILE_SIZE_X *
    // THREAD_TILE_SIZE_Y : blockIdx.y * BLOCK_TILE_SIZE_Y + (threadIdx.x /
    // BLOCK_TILE_SIZE_X + 1) * THREAD_TILE_SIZE_Y][blockIdx.x *
    // BLOCK_TILE_SIZE_X + threadIdx.x % BLOCK_TILE_SIZE_X *
    // THREAD_TILE_SIZE_X : blockIdx.x * BLOCK_TILE_SIZE_X + (threadIdx.x %
    // BLOCK_TILE_SIZE_X + 1) * THREAD_TILE_SIZE_X]
    T C_thread_results[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {
        static_cast<T>(0)};
    // A_vals 는 레지스터에 캐시되어, 현재 스레드 타일의 A 행렬 값을 저장한다
    T A_vals[THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};
    // B_vals 는 레지스터에 캐시되어, 현재 스레드 타일의 B 행렬 값을 저장한다
    T B_vals[THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

    // 모든 스레드 블록 타일을 순회한다
    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        // 데이터를 전역 메모리에서 공유 메모리로 로드한다
        load_data_to_shared_memory<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                                   BLOCK_TILE_SIZE_K, NUM_THREADS>(
            A, lda, B, ldb, A_thread_block_tile, B_thread_block_tile,
            thread_block_tile_idx, thread_linear_idx, m, n, k);
        __syncthreads();

        // 현재 타일에 대해 행렬 곱셈을 계산한다
#pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            // 현재 스레드의 A 타일 안에서의 행 인덱스를 계산한다
            size_t const A_thread_block_tile_row_idx{
                thread_linear_idx / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                THREAD_TILE_SIZE_Y};
            // 현재 스레드의 A 타일 안에서의 열 인덱스를 계산한다
            size_t const A_thread_block_tile_col_idx{k_i};

            // 공유 메모리에서 A 행렬의 값을 레지스터로 로드한다
#pragma unroll
            for (size_t thread_tile_row_idx{0U};
                 thread_tile_row_idx < THREAD_TILE_SIZE_Y;
                 ++thread_tile_row_idx)
            {
                // A_thread_block_tile 의 값을 접근할 때 공유 메모리 bank 충돌이 발생한다
                // DRAM 에서 데이터를 로드할 때 A_thread_block_tile 을 전치하면 이를 개선할 수 있다
                A_vals[thread_tile_row_idx] =
                    A_thread_block_tile[A_thread_block_tile_row_idx +
                                        thread_tile_row_idx]
                                       [A_thread_block_tile_col_idx];
            }

            // 현재 스레드의 B 타일 안에서의 행 인덱스를 계산한다
            size_t const B_thread_block_tile_row_idx{k_i};
            // 현재 스레드의 B 타일 안에서의 열 인덱스를 계산한다
            size_t const B_thread_block_tile_col_idx{
                thread_linear_idx % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                THREAD_TILE_SIZE_X};
            
            // 공유 메모리에서 B 행렬의 값을 레지스터로 로드한다
#pragma unroll
            for (size_t thread_tile_col_idx{0U};
                 thread_tile_col_idx < THREAD_TILE_SIZE_X;
                 ++thread_tile_col_idx)
            {
                B_vals[thread_tile_col_idx] =
                    B_thread_block_tile[B_thread_block_tile_row_idx]
                                       [B_thread_block_tile_col_idx +
                                        thread_tile_col_idx];
            }

            // 스레드 타일의 행렬 곱셈을 계산한다
            for (size_t thread_tile_row_idx{0U};
                 thread_tile_row_idx < THREAD_TILE_SIZE_Y;
                 ++thread_tile_row_idx)
            {
                for (size_t thread_tile_col_idx{0U};
                     thread_tile_col_idx < THREAD_TILE_SIZE_X;
                     ++thread_tile_col_idx)
                {
                    // 계산 결과를 누산한다: C += A * B
                    C_thread_results[thread_tile_row_idx]
                                    [thread_tile_col_idx] +=
                        A_vals[thread_tile_row_idx] *
                        B_vals[thread_tile_col_idx];
                }
            }
        }
        __syncthreads();
    }

    // 계산 결과를 전역 메모리(DRAM)에 기록한다
    for (size_t thread_tile_row_idx{0U};
         thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx)
    {
        for (size_t thread_tile_col_idx{0U};
             thread_tile_col_idx < THREAD_TILE_SIZE_X; ++thread_tile_col_idx)
        {
            // 출력 행렬 C 안에서의 행 인덱스를 계산한다
            size_t const C_row_idx{
                blockIdx.y * BLOCK_TILE_SIZE_Y +
                threadIdx.x / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                    THREAD_TILE_SIZE_Y +
                thread_tile_row_idx};
            // 출력 행렬 C 안에서의 열 인덱스를 계산한다
            size_t const C_col_idx{
                blockIdx.x * BLOCK_TILE_SIZE_X +
                threadIdx.x % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                    THREAD_TILE_SIZE_X +
                thread_tile_col_idx};
            // 경계 검사로 범위를 벗어나지 않도록 한다
            if (C_row_idx < m && C_col_idx < n)
            {
                // GEMM 연산을 수행한다: C = alpha * A * B + beta * C
                C[C_row_idx * ldc + C_col_idx] =
                    alpha * C_thread_results[thread_tile_row_idx]
                                            [thread_tile_col_idx] +
                    beta * C[C_row_idx * ldc + C_col_idx];
            }
        }
    }
}

// GEMM 커널 v04 의 실행 함수
template <typename T>
void launch_gemm_kernel_v04(size_t m, size_t n, size_t k, T const* alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const* beta, T* C, size_t ldc,
                            cudaStream_t stream)
{
    // 블록 타일 크기는 자유롭게 조정할 수 있다
    // 알고리즘의 정확성은 항상 보장되어야 한다
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};  // 블록 타일 X 차원 크기
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};  // 블록 타일 Y 차원 크기
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};   // 블록 타일 K 차원 크기
    // 각 스레드는 C 행렬의 THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y 개 값을 계산한다
    constexpr unsigned int THREAD_TILE_SIZE_X{8U};   // 스레드 타일 X 차원 크기
    constexpr unsigned int THREAD_TILE_SIZE_Y{8U};   // 스레드 타일 Y 차원 크기
    // 블록당 스레드 수를 계산한다
    constexpr unsigned int NUM_THREADS_PER_BLOCK{
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
        (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y)};
    
    // 정적 단언으로 파라미터 구성의 정확성을 보장한다
    static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0U);
    static_assert(
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS_PER_BLOCK == 0U);
    static_assert(
        BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS_PER_BLOCK == 0U);
    
    // CUDA 커널 실행 파라미터를 구성한다
    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};  // 블록 차원
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};  // 그리드 차원
    
    // CUDA 커널을 실행한다
    gemm_v04<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
             THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                              *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
```



이 FP32 GEMM 구현의 성능은 NVIDIA GeForce RTX 3090 GPU 에서 13.02 TFLOPS 에 이른다.

## 2D 블록 타일링과 2D 스레드 타일링과 벡터화 메모리 접근을 사용하는 구현

이전 글 “CUDA 벡터화 메모리 접근”에서, 벡터화 메모리 접근을 사용해 간단한 메모리 복사 커널의 성능을 높이는 방법을 보였다. 벡터화 메모리 접근은 메모리 트랜잭션 수를 줄여 메모리 대역폭 활용도를 높인다. 같은 기법을 이 GEMM 커널에 적용하여, 전역 메모리에서 공유 메모리로의 데이터 로드와 공유 메모리에서 레지스터로의 데이터 로드를 가속할 수 있다.

이전 구현에서는 행렬 곱셈을 계산하기 위해 각 스레드가 공유 메모리에서 행렬 $A$ 의 한 열과 행렬 $B$ 의 한 행을 읽어 레지스터에 캐시해야 했다. 행렬 $A$ 의 한 열에서 데이터를 읽는 것은 벡터화 메모리 접근을 막으므로, 전역 메모리에서 공유 메모리로 데이터를 로드할 때 행렬 $A$ 를 전치하고자 한다. 이렇게 하면 각 스레드가 전치된 행렬 $A$ 의 한 행과 행렬 $B$ 의 한 행을 공유 메모리에서 벡터화 방식으로 접근하여 레지스터에 캐시할 수 있다.

아래는 2D 블록 타일링과 2D 스레드 타일링과 벡터화 메모리 접근을 사용하는 구현이다.

```c++
// 벡터화 메모리 접근을 위한 데이터 로드 함수. 행렬 A 를 전치한 뒤 공유 메모리로 로드한다
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS, size_t BLOCK_TILE_SKEW_SIZE_X = 0U, size_t BLOCK_TILE_SKEW_SIZE_Y = 0U, typename VECTOR_TYPE = int4>
__device__ void load_data_to_shared_memory_transposed_vectorized(T const* A, size_t lda,
                                           T const* B, size_t ldb,
                                           T A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y],
                                           T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
                                           size_t thread_block_tile_idx,
                                           size_t thread_linear_idx,
                                           size_t m, size_t n,
                                           size_t k)
{
    // 벡터화 접근의 단위 개수를 계산한다(예: int4 는 4 개의 float 를 포함한다)
    constexpr size_t NUM_VECTOR_UNITS{sizeof(VECTOR_TYPE) / sizeof(T)};
    static_assert(sizeof(VECTOR_TYPE) % sizeof(T) == 0U);
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);
    
    // 벡터화 후의 블록 타일 크기를 계산한다
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_K{BLOCK_TILE_SIZE_K /
                                                  NUM_VECTOR_UNITS};
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_X{BLOCK_TILE_SIZE_X /
                                                  NUM_VECTOR_UNITS};
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);

    // 벡터화 로드를 지원하도록 공유 메모리 데이터가 올바르게 정렬되었는지 보장한다
    // skew 크기는 벡터화 로드 시 공유 메모리 데이터의 정렬에 영향을 줄 수 있다
    static_assert((BLOCK_TILE_SIZE_Y) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_X) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);
    static_assert((BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X) * sizeof(T) % sizeof(VECTOR_TYPE) == 0U);

// DRAM 의 행렬 A 에서 데이터를 공유 메모리의 A_thread_block_tile 로 로드한다(전치 저장)
#pragma unroll
    for (size_t load_idx{0U};
            load_idx < (BLOCK_TILE_SIZE_Y * VECTORIZED_BLOCK_TILE_SIZE_K +
                        NUM_THREADS - 1U) /
                        NUM_THREADS;
            ++load_idx)
    {
        // 현재 스레드의 A_thread_block_tile 안에서의 행 인덱스를 계산한다
        size_t const A_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) /
            VECTORIZED_BLOCK_TILE_SIZE_K};
        // 현재 스레드의 A_thread_block_tile 안에서의 열 인덱스를 계산한다(벡터화 후)
        size_t const A_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) %
            VECTORIZED_BLOCK_TILE_SIZE_K * NUM_VECTOR_UNITS};
        // 전역 행렬 A 안에서의 행 인덱스를 계산한다
        size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                                A_thread_block_tile_row_idx};
        // 전역 행렬 A 안에서의 열 인덱스를 계산한다
        size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                A_thread_block_tile_col_idx};

        // 경계 검사는 커널 성능을 어느 정도 떨어뜨릴 수 있으나
        // 서로 다른 모든 GEMM 구성에 대해 커널의 정확성을 보장한다
        int4 A_row_vector_vals{0, 0, 0, 0};
        if (A_row_idx < m && A_col_idx < k)
        {
            // 행렬 A 의 한 행 데이터를 벡터화하여 읽는다
            A_row_vector_vals = *reinterpret_cast<int4 const*>(
                &A[A_row_idx * lda + A_col_idx]);
        }
        // 행렬 경계를 벗어나면 무효한 원소를 0 으로 설정해야 한다
        if (A_col_idx + NUM_VECTOR_UNITS > k)
        {
            // 마지막 벡터 안의 무효한 원소 개수를 계산한다
            size_t const num_invalid_elements{A_col_idx + NUM_VECTOR_UNITS -
                                                k};
            // 무효한 원소를 마스킹한다
            T* const A_row_vector_vals_ptr{
                reinterpret_cast<T*>(&A_row_vector_vals)};
            for (size_t i{0U}; i < num_invalid_elements; ++i)
            {
                A_row_vector_vals_ptr[NUM_VECTOR_UNITS - 1U - i] =
                    static_cast<T>(0);
            }
        }
        // 아래 조건을 만족하면 다음 if 판정을 제거할 수 있다
        // static_assert(VECTORIZED_BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y %
        // NUM_THREADS ==
        //               0U);
        if (A_thread_block_tile_row_idx < BLOCK_TILE_SIZE_Y &&
            A_thread_block_tile_col_idx < BLOCK_TILE_SIZE_K)
        {
            // 벡터화하여 읽은 데이터를 공유 메모리에 전치하여 저장한다
            for (size_t i{0U}; i < NUM_VECTOR_UNITS; ++i)
            {
                A_thread_block_tile_transposed
                    [A_thread_block_tile_col_idx + i]
                    [A_thread_block_tile_row_idx] =
                        reinterpret_cast<T const*>(&A_row_vector_vals)[i];
            }
        }
    }
// DRAM 의 행렬 B 에서 데이터를 공유 메모리의 B_thread_block_tile 로 로드한다
#pragma unroll
    for (size_t load_idx{0U};
            load_idx < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_X +
                        NUM_THREADS - 1U) /
                        NUM_THREADS;
            ++load_idx)
    {
        // 현재 스레드의 B_thread_block_tile 안에서의 행 인덱스를 계산한다
        size_t const B_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) /
            VECTORIZED_BLOCK_TILE_SIZE_X};
        // 현재 스레드의 B_thread_block_tile 안에서의 열 인덱스를 계산한다(벡터화 후)
        size_t const B_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) %
            VECTORIZED_BLOCK_TILE_SIZE_X * NUM_VECTOR_UNITS};
        // 전역 행렬 B 안에서의 행 인덱스를 계산한다
        size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                                B_thread_block_tile_row_idx};
        // 전역 행렬 B 안에서의 열 인덱스를 계산한다
        size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                                B_thread_block_tile_col_idx};

        // 경계 검사는 커널 성능을 어느 정도 떨어뜨릴 수 있으나
        // 서로 다른 모든 GEMM 구성에 대해 커널의 정확성을 보장한다
        int4 B_row_vector_vals{0, 0, 0, 0};
        if (B_row_idx < k && B_col_idx < n)
        {
            // 행렬 B 의 한 행 데이터를 벡터화하여 읽는다
            B_row_vector_vals = *reinterpret_cast<int4 const*>(
                &B[B_row_idx * ldb + B_col_idx]);
        }
        // 행렬 경계를 벗어나면 무효한 원소를 0 으로 설정해야 한다
        if (B_col_idx + NUM_VECTOR_UNITS > n)
        {
            // 마지막 벡터 안의 무효한 원소 개수를 계산한다
            size_t const num_invalid_elements{B_col_idx + NUM_VECTOR_UNITS -
                                                n};
            // 무효한 원소를 마스킹한다
            T* const B_row_vector_vals_ptr{
                reinterpret_cast<T*>(&B_row_vector_vals)};
            for (size_t i{0U}; i < num_invalid_elements; ++i)
            {
                B_row_vector_vals_ptr[NUM_VECTOR_UNITS - 1U - i] =
                    static_cast<T>(0);
            }
        }
        // 아래 조건을 만족하면 다음 if 판정을 제거할 수 있다
        // static_assert(VECTORIZED_BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K %
        // NUM_THREADS ==
        //               0U);
        if (B_thread_block_tile_row_idx < BLOCK_TILE_SIZE_K &&
            B_thread_block_tile_col_idx < BLOCK_TILE_SIZE_X)
        {
            // 공유 메모리의 B_thread_block_tile 에 벡터화하여 기록한다
            *reinterpret_cast<int4*>(
                &B_thread_block_tile[B_thread_block_tile_row_idx]
                                    [B_thread_block_tile_col_idx]) =
                B_row_vector_vals;
        }
    }
}

// GEMM 커널 v05 버전 - 벡터화 메모리 접근 사용
// 전역 메모리에서 병합 읽기/쓰기를 수행한다
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t THREAD_TILE_SIZE_X,
          size_t THREAD_TILE_SIZE_Y>
__global__ void gemm_v05_vectorized(size_t m, size_t n, size_t k, T alpha,
                                    T const* A, size_t lda, T const* B,
                                    size_t ldb, T beta, T* C, size_t ldc)
{
    // 블록당 스레드 수로 blockDim.x * blockDim.y 를 사용하는 것을 피한다
    // 이 값은 런타임 상수여서 컴파일러가 이를 기준으로 루프 언롤링을 최적화할 수 없기 때문이다
    // 대신 컴파일 타임 상수를 사용한다
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
                                 (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y)};
    // 현재 스레드의 블록 내 선형 인덱스를 계산한다
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

    // 데이터 재사용을 위해 A 와 B 의 타일을 공유 메모리에 캐시한다
    __shared__ T
        A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    // 처리해야 할 스레드 블록 타일의 개수를 계산한다
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};

    // 블록 안의 각 스레드는 THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X 개의 출력 값을 처리한다
    // 구체적으로 이 값들은 다음에 대응한다
    // C[blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.x / BLOCK_TILE_SIZE_X *
    // THREAD_TILE_SIZE_Y : blockIdx.y * BLOCK_TILE_SIZE_Y + (threadIdx.x /
    // BLOCK_TILE_SIZE_X + 1) * THREAD_TILE_SIZE_Y][blockIdx.x *
    // BLOCK_TILE_SIZE_X + threadIdx.x % BLOCK_TILE_SIZE_X *
    // THREAD_TILE_SIZE_X : blockIdx.x * BLOCK_TILE_SIZE_X + (threadIdx.x %
    // BLOCK_TILE_SIZE_X + 1) * THREAD_TILE_SIZE_X]
    T C_thread_results[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {
        static_cast<T>(0)};
    // A_vals 는 레지스터에 캐시된다
    T A_vals[THREAD_TILE_SIZE_Y] = {static_cast<T>(0)};
    // B_vals 는 레지스터에 캐시된다
    T B_vals[THREAD_TILE_SIZE_X] = {static_cast<T>(0)};

    // 벡터화 접근 관련 상수 정의
    constexpr size_t NUM_VECTOR_UNITS{sizeof(int4) / sizeof(T)};
    static_assert(sizeof(int4) % sizeof(T) == 0U);
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_THREAD_TILE_SIZE_X{THREAD_TILE_SIZE_X /
                                                   NUM_VECTOR_UNITS};
    static_assert(THREAD_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);

    // 메인 계산 루프: 모든 스레드 블록 타일을 순회한다
    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        // 데이터를 공유 메모리로 로드한다(A 는 전치, B 는 그대로)
        load_data_to_shared_memory_transposed_vectorized<
            T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            NUM_THREADS>(A, lda, B, ldb, A_thread_block_tile_transposed,
                         B_thread_block_tile, thread_block_tile_idx,
                         thread_linear_idx, m, n, k);
        __syncthreads();

        // K 차원에 대해 행렬 곱셈을 계산한다
#pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            // 현재 스레드의 A_thread_block_tile 안에서의 행 인덱스를 계산한다
            size_t const A_thread_block_tile_row_idx{
                thread_linear_idx / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                THREAD_TILE_SIZE_Y};
            // A 의 열 인덱스는 바로 현재의 k_i 이다
            size_t const A_thread_block_tile_col_idx{k_i};

            // 공유 메모리에서 A 의 데이터를 레지스터로 로드한다
#pragma unroll
            for (size_t thread_tile_row_idx{0U};
                 thread_tile_row_idx < THREAD_TILE_SIZE_Y;
                 ++thread_tile_row_idx)
            {
                A_vals[thread_tile_row_idx] =
                    A_thread_block_tile_transposed[A_thread_block_tile_col_idx]
                                                  [A_thread_block_tile_row_idx +
                                                   thread_tile_row_idx];
            }

            // 현재 스레드의 B_thread_block_tile 안에서의 인덱스를 계산한다
            size_t const B_thread_block_tile_row_idx{k_i};
            size_t const B_thread_block_tile_col_idx{
                thread_linear_idx % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                THREAD_TILE_SIZE_X};
// A_thread_block_tile 에서의 읽기는 벡터화할 수 없지만, B_thread_block_tile 에서의 읽기는 벡터화할 수 있다
#pragma unroll
            for (size_t thread_tile_col_vector_idx{0U};
                 thread_tile_col_vector_idx < VECTORIZED_THREAD_TILE_SIZE_X;
                 ++thread_tile_col_vector_idx)
            {
                // B 의 데이터를 벡터화하여 레지스터로 읽는다
                *reinterpret_cast<int4*>(
                    &B_vals[thread_tile_col_vector_idx * NUM_VECTOR_UNITS]) =
                    *reinterpret_cast<int4 const*>(
                        &B_thread_block_tile[B_thread_block_tile_row_idx]
                                            [B_thread_block_tile_col_idx +
                                             thread_tile_col_vector_idx *
                                                 NUM_VECTOR_UNITS]);
            }

            // 행렬 곱셈-누산을 계산한다
            for (size_t thread_tile_row_idx{0U};
                 thread_tile_row_idx < THREAD_TILE_SIZE_Y;
                 ++thread_tile_row_idx)
            {
                for (size_t thread_tile_col_idx{0U};
                     thread_tile_col_idx < THREAD_TILE_SIZE_X;
                     ++thread_tile_col_idx)
                {
                    C_thread_results[thread_tile_row_idx]
                                    [thread_tile_col_idx] +=
                        A_vals[thread_tile_row_idx] *
                        B_vals[thread_tile_col_idx];
                }
            }
        }
        __syncthreads();
    }

    // 결과를 벡터화하여 DRAM 에 기록한다
    for (size_t thread_tile_row_idx{0U};
         thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx)
    {
        for (size_t thread_tile_col_vector_idx{0U};
             thread_tile_col_vector_idx < VECTORIZED_THREAD_TILE_SIZE_X;
             ++thread_tile_col_vector_idx)
        {
            // 전역 행렬 C 안에서의 행 인덱스를 계산한다
            size_t const C_row_idx{
                blockIdx.y * BLOCK_TILE_SIZE_Y +
                thread_linear_idx / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                    THREAD_TILE_SIZE_Y +
                thread_tile_row_idx};
            // 전역 행렬 C 안에서의 열 인덱스를 계산한다
            size_t const C_col_idx{
                blockIdx.x * BLOCK_TILE_SIZE_X +
                thread_linear_idx % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                    THREAD_TILE_SIZE_X +
                thread_tile_col_vector_idx * NUM_VECTOR_UNITS};
            // C 의 원래 값을 벡터화하여 읽는다
            int4 C_row_vector_vals{*reinterpret_cast<int4 const*>(
                &C[C_row_idx * ldc + C_col_idx])};
            // 계산 결과를 벡터화하여 읽는다
            int4 const C_thread_results_row_vector_vals{
                *reinterpret_cast<int4 const*>(
                    &C_thread_results[thread_tile_row_idx]
                                     [thread_tile_col_vector_idx *
                                      NUM_VECTOR_UNITS])};
            // C_row_vector_vals 의 값을 갱신한다(alpha*결과 + beta*원래값 을 수행)
            for (size_t i{0U}; i < NUM_VECTOR_UNITS; ++i)
            {
                reinterpret_cast<T*>(&C_row_vector_vals)[i] =
                    alpha * reinterpret_cast<T const*>(
                                &C_thread_results_row_vector_vals)[i] +
                    beta * reinterpret_cast<T const*>(&C_row_vector_vals)[i];
            }
            // C 에 벡터화하여 기록한다
            if (C_row_idx < m && C_col_idx < n)
            {
                // 범위를 벗어난 무효한 원소를 마스킹할 필요가 없다,
                // C 행렬의 행이 32 바이트로 정렬되어 있기 때문이다
                *reinterpret_cast<int4*>(&C[C_row_idx * ldc + C_col_idx]) =
                    C_row_vector_vals;
            }
        }
    }
}

// GEMM 커널 v05 벡터화 버전의 실행 함수
template <typename T>
void launch_gemm_kernel_v05_vectorized(size_t m, size_t n, size_t k,
                                       T const* alpha, T const* A, size_t lda,
                                       T const* B, size_t ldb, T const* beta,
                                       T* C, size_t ldc, cudaStream_t stream)
{
    // 블록 타일 크기는 자유롭게 조정할 수 있다
    // 알고리즘의 정확성은 항상 보장되어야 한다
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};   // 블록 타일 X 차원 크기
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};   // 블록 타일 Y 차원 크기
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};    // 블록 타일 K 차원 크기
    // 각 스레드는 C 행렬의 THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y 개 값을 계산한다
    constexpr unsigned int THREAD_TILE_SIZE_X{8U};    // 스레드 타일 X 차원 크기
    constexpr unsigned int THREAD_TILE_SIZE_Y{8U};    // 스레드 타일 Y 차원 크기
    // 블록당 스레드 수를 계산한다
    constexpr unsigned int NUM_THREADS_PER_BLOCK{
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
        (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y)};
    
    // 정적 단언으로 파라미터 구성의 정확성을 보장한다
    static_assert(BLOCK_TILE_SIZE_X % THREAD_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % THREAD_TILE_SIZE_Y == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_K == 0U);
    static_assert(NUM_THREADS_PER_BLOCK % BLOCK_TILE_SIZE_X == 0U);
    static_assert(
        BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS_PER_BLOCK == 0U);
    static_assert(
        BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS_PER_BLOCK == 0U);
    
    // CUDA 커널 실행 파라미터를 구성한다
    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};  // 블록 차원
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};  // 그리드 차원
    
    // CUDA 커널을 실행한다
    gemm_v05_vectorized<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                        BLOCK_TILE_SIZE_K, THREAD_TILE_SIZE_X,
                        THREAD_TILE_SIZE_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                              *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
```


데이터 로드에 벡터화 메모리 접근을 사용한다는 점을 제외하면, 나머지 커널은 2D 블록 타일링과 2D 스레드 타일링을 사용한 이전 구현과 동일하다. 그러나 우리의 사용 사례에서 벡터화 메모리 접근에는 이전 구현에는 없던 주의 사항이 하나 있다. 데이터를 전역 메모리에서 공유 메모리로, 그리고 공유 메모리에서 레지스터로 로드할 때, 행렬이 2D 임을 고려하면 벡터화 메모리 접근 데이터 타입의 데이터 정렬이 올바른지 확인해야 한다. 그렇지 않으면 미정의 동작(undefined behavior)이 발생한다. 예를 들어 벡터화 메모리 접근 데이터 타입으로 $int4$ 를 사용한다면, 데이터 정렬이 16 바이트의 배수인지 확인해야 한다. 이것이 바로 전역 메모리에서 행렬 $A$ 와 행렬 $B$ 의 선행 차원을 패딩해야 하고, 공유 메모리 차원을 신중하게 선택해야 하는 이유이다.

이 FP32 GEMM 구현의 성능은 NVIDIA GeForce RTX 3090 GPU 에서 19.66 TFLOPS 에 이른다.


## 2D 블록 타일링과 2D Warp 타일링과 2D 스레드 타일링과 벡터화 메모리 접근을 사용하는 구현

CUDA 프로그래밍 모델에서 warp 는 32 개의 스레드로 구성되며, 스케줄링과 실행의 최소 단위이다. warp 안의 스레드들이 공유 메모리의 같은 bank 에 접근하면 공유 메모리 bank 충돌(https://leimao.github.io/blog/CUDA-Shared-Memory-Bank-Conflicts/)이 발생할 수 있다. 이전 구현에서는 GEMM CUDA 커널이 warp 중심으로 구성되지 않았기 때문에, 공유 메모리 bank 충돌을 어떻게 피할지가 명확하지 않았다.

이 구현에서는 GEMM CUDA 커널을 warp 중심 방식으로 구성하고 2D warp 타일링과 2D 스레드 타일링을 사용한다. 이렇게 하면 공유 메모리 bank 충돌을 더 쉽게 예측하고 최적화할 수 있다.

warp 타일링을 이해하는 것은 스레드 타일링을 이해하는 것과 거의 완전히 동일하다.

수학적으로, 행렬 곱셈-누산 연산 $D_{b_m,b_n}^{d_m \times d_n} = \sum_{b_k=1}^{k/d_{bk}} A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n} + C_{b_m,b_n}^{d_m \times d_n}$(여기서 $D_{b_m,b_n} \in \mathbb{R}^{d_m \times d_n}$, $A_{b_m,b_k} \in \mathbb{R}^{d_m \times d_{bk}}$, $B_{b_k,b_n} \in \mathbb{R}^{d_{bk} \times d_n}$, $C_{b_m,b_n} \in \mathbb{R}^{d_m \times d_n}$)가 주어지면, 행렬은 더 작은 행렬들로 분할할 수 있다.

$$A_{b_m,b_k}^{d_m \times d_{bk}} = \begin{bmatrix}
\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{1,1}^{d_{wm} \times d_{wk}} & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{1,2}^{d_{wm} \times d_{wk}} & \cdots & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{1,d_{bk}/d_{wk}}^{d_{wm} \times d_{wk}} \\
\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{2,1}^{d_{wm} \times d_{wk}} & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{2,2}^{d_{wm} \times d_{wk}} & \cdots & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{2,d_{bk}/d_{wk}}^{d_{wm} \times d_{wk}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{d_m/d_{wm},1}^{d_{wm} \times d_{wk}} & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{d_m/d_{wm},2}^{d_{wm} \times d_{wk}} & \cdots & \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{d_m/d_{wm},d_{bk}/d_{wk}}^{d_{wm} \times d_{wk}}
\end{bmatrix}$$

$$B_{b_k,b_n}^{d_{bk} \times d_n} = \begin{bmatrix}
\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{1,1}^{d_{wk} \times d_{wn}} & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{1,2}^{d_{wk} \times d_{wn}} & \cdots & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{1,d_n/d_{wn}}^{d_{wk} \times d_{wn}} \\
\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{2,1}^{d_{wk} \times d_{wn}} & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{2,2}^{d_{wk} \times d_{wn}} & \cdots & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{2,d_n/d_{wn}}^{d_{wk} \times d_{wn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{d_{bk}/d_{wk},1}^{d_{wk} \times d_{wn}} & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{d_{bk}/d_{wk},2}^{d_{wk} \times d_{wn}} & \cdots & \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{d_{bk}/d_{wk},d_n/d_{wn}}^{d_{wk} \times d_{wn}}
\end{bmatrix}$$

$$C_{b_m,b_n}^{d_m \times d_n} = \begin{bmatrix}
\left(C_{b_m,b_n}^{d_m \times d_n}\right)_{1,1}^{d_{wm} \times d_{wn}} & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{1,2}^{d_{wm} \times d_{wn}} & \cdots & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{1,d_n/d_{wn}}^{d_{wm} \times d_{wn}} \\
\left(C_{b_m,b_n}^{d_m \times d_n}\right)_{2,1}^{d_{wm} \times d_{wn}} & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{2,2}^{d_{wm} \times d_{wn}} & \cdots & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{2,d_n/d_{wn}}^{d_{wm} \times d_{wn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(C_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{wm},1}^{d_{wm} \times d_{wn}} & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{wm},2}^{d_{wm} \times d_{wn}} & \cdots & \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{wm},d_n/d_{wn}}^{d_{wm} \times d_{wn}}
\end{bmatrix}$$

$$D_{b_m,b_n}^{d_m \times d_n} = \begin{bmatrix}
\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{1,1}^{d_{wm} \times d_{wn}} & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{1,2}^{d_{wm} \times d_{wn}} & \cdots & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{1,d_n/d_{wn}}^{d_{wm} \times d_{wn}} \\
\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{2,1}^{d_{wm} \times d_{wn}} & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{2,2}^{d_{wm} \times d_{wn}} & \cdots & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{2,d_n/d_{wn}}^{d_{wm} \times d_{wn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{wm},1}^{d_{wm} \times d_{wn}} & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{wm},2}^{d_{wm} \times d_{wn}} & \cdots & \left(D_{b_m,b_n}^{d_m \times d_n}\right)_{d_m/d_{wm},d_n/d_{wn}}^{d_{wm} \times d_{wn}}
\end{bmatrix}$$

$D_{b_m,b_n}^{d_m \times d_n}$ 안의 각 작은 행렬은 여러 개의 작은 행렬 곱셈과 누산으로 계산된다.

$$\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}} = \left(\sum_{b_k=1}^{k/d_{bk}} A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n} + C_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}} + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(\sum_{w_k=1}^{d_{bk}/d_{wk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right) + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$$

블록 warp 인덱스가 $(w_m, w_n)$ 인 각 warp(여기서 $w_m \in [1, d_m/d_{wm}]$, $w_n \in [1, d_n/d_{wn}]$)는, 블록 인덱스가 $(b_m, b_n)$ 인 블록(여기서 $b_m \in [1, m/d_m]$, $b_n \in [1, n/d_n]$) 안에서 하나의 작은 행렬 곱셈-누산 $\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$ 을 계산하는 것을 담당한다.

여기까지는 스레드 인덱스와 스레드 타일 크기가 warp 인덱스와 warp 타일 크기로 대체된 것을 제외하면, 모든 것이 2D 스레드 타일링의 수학적 설명과 동일해 보인다.

남은 문제는 블록 warp 인덱스가 $(w_m, w_n)$ 인 warp 안의 32 개 스레드를 모두 사용해 $\left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$ 를 어떻게 계산하느냐이다. 이를 위한 유일한 방법이 있는 것은 아니다. 우리가 선택한 방식은 2D 스레드 타일링을 사용하는 것이다. warp 안에서 행 방향 스레드 수를 $m_t$, 열 방향 스레드 수를 $n_t$ 라고 하면, $m_t \times n_t = 32$ 여야 한다. warp 안의 각 스레드는 $(d_{wm}/m_t) \times (d_{wn}/n_t)$ 개의 출력 행렬 값을 계산하는 것을 담당해야 한다. 이어서 스레드 타일 크기를 행 방향 $d_{tm}$, 열 방향 $d_{tn}$ 으로 설정하여 $(d_{wm}/m_t) \bmod d_{tm} = 0$ 과 $(d_{wn}/n_t) \bmod d_{tn} = 0$ 을 만족시킨다. warp 안의 각 스레드는 크기가 $d_{tm} \times d_{tn}$ 인 출력 행렬 $\left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$ 의 블록을 $((d_{wm}/m_t)/d_{tm}) \times ((d_{wn}/n_t)/d_{tn})$ 개 계산해야 한다.

스레드 타일 인덱스를 $(t_m, t_n)$ 이라 하자(여기서 $t_m \in [1, d_{wm}/d_{tm}]$, $t_n \in [1, d_{wn}/d_{tn}]$). 이 타일 계산을 담당하는 스레드는 warp 스레드 인덱스 $(t_m \bmod m_t, t_n \bmod n_t)$ 를 가진다. 행렬 $\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}$ 는 행 차원을 따라 $d_{wm}/d_{tm}$ 개의 조각으로, 행렬 $\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}$ 는 열 차원을 따라 $d_{wn}/d_{tn}$ 개의 조각으로 분할할 수 있으므로, 다음을 얻는다.

$$\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} = \begin{bmatrix}
\left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{1,1}^{d_{tm} \times d_{tk}} & \left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{1,2}^{d_{tm} \times d_{tk}} & \cdots & \left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{1,d_{wk}/d_{tk}}^{d_{tm} \times d_{tk}} \\
\left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{2,1}^{d_{tm} \times d_{tk}} & \left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{2,2}^{d_{tm} \times d_{tk}} & \cdots & \left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{2,d_{wk}/d_{tk}}^{d_{tm} \times d_{tk}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{d_{wm}/d_{tm},1}^{d_{tm} \times d_{tk}} & \left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{d_{wm}/d_{tm},2}^{d_{tm} \times d_{tk}} & \cdots & \left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{d_{wm}/d_{tm},d_{wk}/d_{tk}}^{d_{tm} \times d_{tk}}
\end{bmatrix}$$

$$\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}} = \begin{bmatrix}
\left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{1,1}^{d_{tk} \times d_{tn}} & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{1,2}^{d_{tk} \times d_{tn}} & \cdots & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{1,d_{wn}/d_{tn}}^{d_{tk} \times d_{tn}} \\
\left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{2,1}^{d_{tk} \times d_{tn}} & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{2,2}^{d_{tk} \times d_{tn}} & \cdots & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{2,d_{wn}/d_{tn}}^{d_{tk} \times d_{tn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{d_{wk}/d_{tk},1}^{d_{tk} \times d_{tn}} & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{d_{wk}/d_{tk},2}^{d_{tk} \times d_{tn}} & \cdots & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{d_{wk}/d_{tk},d_{wn}/d_{tn}}^{d_{tk} \times d_{tn}}
\end{bmatrix}$$

warp 스레드 인덱스가 $(t_m \bmod m_t, t_n \bmod n_t)$ 인 각 스레드는 하나의 작은 행렬 곱셈-누산 $\left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$ 을 계산하는 것을 담당한다.

스레드 타일 $\left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$ 는 다음과 같이 계산할 수 있다.

$$\left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}} = \sum_{t_k=1}^{d_{wk}/d_{tk}} \left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{t_m,t_k}^{d_{tm} \times d_{tk}} \left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{t_k,t_n}^{d_{tk} \times d_{tn}}$$

이를 종합하면, 스레드 타일 $\left(\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$ 는 다음과 같이 계산할 수 있다.

$$\left(\left(D_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}} = \left(\sum_{b_k=1}^{k/d_{bk}} \left(\sum_{w_k=1}^{d_{bk}/d_{wk}} \left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} \left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right) + \left(C_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(\sum_{w_k=1}^{d_{bk}/d_{wk}} \left(\sum_{t_k=1}^{d_{wk}/d_{tk}} \left(\left(A_{b_m,b_k}^{d_m \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{t_m,t_k}^{d_{tm} \times d_{tk}} \left(\left(B_{b_k,b_n}^{d_{bk} \times d_n}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{t_k,t_n}^{d_{tk} \times d_{tn}}\right)\right) + \left(\left(C_{b_m,b_n}^{d_m \times d_n}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$$

이 구현에서는 스레드 타일링 알고리즘을 더 단순하게 만들기 위해 $d_{wk} = d_{tk}$ 로 설정한다.


아래는 2D 블록 타일링과 2D warp 타일링과 2D 스레드 타일링과 벡터화 메모리 접근을 사용하는 구현이다.

```c++
// GEMM kernel v06.
// 각 스레드 블록 안의 스레드는 THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X 개의 출력 값을 처리한다
// 스레드 수는 BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_X / (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X) 이다
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t WARP_TILE_SIZE_X,
          size_t WARP_TILE_SIZE_Y, size_t THREAD_TILE_SIZE_X,
          size_t THREAD_TILE_SIZE_Y, size_t NUM_THREADS_PER_WARP_X,
          size_t NUM_THREADS_PER_WARP_Y>
__global__ void gemm_v06_vectorized(size_t m, size_t n, size_t k, T alpha,
                                    T const* A, size_t lda, T const* B,
                                    size_t ldb, T beta, T* C, size_t ldc)
{
    // 각 warp 가 32 개의 스레드를 가지도록 보장한다
    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32U);
    
    // 각 블록 안의 warp 수를 계산한다
    constexpr size_t NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
    constexpr size_t NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);
    
    // 각 warp 안의 스레드 타일 수를 계산한다
    constexpr unsigned int NUM_THREAD_TILES_PER_WARP_X{
        WARP_TILE_SIZE_X / (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X)};
    constexpr unsigned int NUM_THREAD_TILES_PER_WARP_Y{
        WARP_TILE_SIZE_Y / (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y)};
    static_assert(
        WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0U);
    static_assert(
        WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0U);

    // 전체 스레드 수를 계산한다
    constexpr unsigned int NUM_THREADS_X{NUM_WARPS_X * NUM_THREADS_PER_WARP_X};
    constexpr unsigned int NUM_THREADS_Y{NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y};
    // 블록당 스레드 수로 blockDim.x * blockDim.y 를 사용하는 것을 피한다
    // 이 값은 런타임 상수여서 컴파일러가 이를 기준으로 루프 언롤링을 최적화할 수 없기 때문이다
    // 대신 컴파일 타임 상수를 사용한다
    constexpr size_t NUM_THREADS{NUM_THREADS_X * NUM_THREADS_Y};

    // 데이터 재사용을 위해 A 와 B 의 타일을 공유 메모리에 캐시한다
    __shared__ T
        A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_Y];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    // A_vals 는 레지스터에 캐시된다
    T A_vals[NUM_THREAD_TILES_PER_WARP_Y][THREAD_TILE_SIZE_Y] = {
        static_cast<T>(0)};
    // B_vals 는 레지스터에 캐시된다
    T B_vals[NUM_THREAD_TILES_PER_WARP_X][THREAD_TILE_SIZE_X] = {
        static_cast<T>(0)};

    // 스레드 인덱스를 계산한다
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};
    size_t const warp_linear_idx{thread_linear_idx / 32U};
    size_t const warp_row_idx{warp_linear_idx / NUM_WARPS_X};
    size_t const warp_col_idx{warp_linear_idx % NUM_WARPS_X};
    size_t const thread_linear_idx_in_warp{thread_linear_idx % 32U};
    size_t const thread_linear_row_idx_in_warp{thread_linear_idx_in_warp /
                                               NUM_THREADS_PER_WARP_X};
    size_t const thread_linear_col_idx_in_warp{thread_linear_idx_in_warp %
                                               NUM_THREADS_PER_WARP_X};

    // 내적 합을 수행하는 외부 루프 횟수
    // C_thread_block_tile =
    // \sigma_{thread_block_tile_idx=0}^{num_thread_block_tiles-1} A[:,
    // thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
    // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :]
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};
    // 블록 안의 각 스레드는 NUM_THREAD_TILES_PER_WARP_Y *
    // NUM_THREAD_TILES_PER_WARP_X * THREAD_TILE_SIZE_Y *
    // THREAD_TILE_SIZE_X 개의 출력 값을 처리한다
    T C_thread_results[NUM_THREAD_TILES_PER_WARP_Y][NUM_THREAD_TILES_PER_WARP_X]
                      [THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {
                          static_cast<T>(0)};

    // 벡터화 메모리 접근 설정
    constexpr size_t NUM_VECTOR_UNITS{sizeof(int4) / sizeof(T)};
    static_assert(sizeof(int4) % sizeof(T) == 0U);
    static_assert(BLOCK_TILE_SIZE_K % NUM_VECTOR_UNITS == 0U);
    static_assert(BLOCK_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_THREAD_TILE_SIZE_X{THREAD_TILE_SIZE_X /
                                                   NUM_VECTOR_UNITS};
    static_assert(THREAD_TILE_SIZE_X % NUM_VECTOR_UNITS == 0U);
    constexpr size_t VECTORIZED_THREAD_TILE_SIZE_Y{THREAD_TILE_SIZE_Y /
                                                   NUM_VECTOR_UNITS};
    static_assert(THREAD_TILE_SIZE_Y % NUM_VECTOR_UNITS == 0U);

    // 메인 루프: 모든 스레드 블록 타일을 순회한다
    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        // 데이터를 공유 메모리로 로드한다(전치 및 벡터화)
        load_data_to_shared_memory_transposed_vectorized<
            T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            NUM_THREADS>(A, lda, B, ldb, A_thread_block_tile_transposed,
                         B_thread_block_tile, thread_block_tile_idx,
                         thread_linear_idx, m, n, k);
        __syncthreads();

        // A[:, thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
        // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] 
        // 여기서 A[:, thread_block_tile_idx:BLOCK_TILE_SIZE_K] 와
        // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] 는 각각
        // 공유 메모리에 A_thread_block_tile 과 B_thread_block_tile 로 캐시되어 있다
        // 이 내적은 다시 BLOCK_TILE_SIZE_K 개의 외적으로 분해된다
        // A_thread_block_tile * B_thread_block_tile = 
        // \sigma_{k_i=0}^{BLOCK_TILE_SIZE_K-1} A_thread_block_tile[:, k_i] @ B_thread_block_tile[k_i, :]
        // A_thread_block_tile 과 B_thread_block_tile 은 모두 레지스터에 캐시될 수 있음에 유의한다
#pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            // A 의 데이터를 레지스터로 로드한다
#pragma unroll
            for (size_t thread_tile_repeat_row_idx{0U};
                 thread_tile_repeat_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
                 ++thread_tile_repeat_row_idx)
            {
                size_t const A_thread_block_tile_row_idx{
                    warp_row_idx * WARP_TILE_SIZE_Y +
                    thread_tile_repeat_row_idx *
                        (WARP_TILE_SIZE_Y / NUM_THREAD_TILES_PER_WARP_Y) +
                    thread_linear_row_idx_in_warp * THREAD_TILE_SIZE_Y};
                size_t const A_thread_block_tile_col_idx{k_i};
                // A 의 데이터를 벡터화하여 로드한다
#pragma unroll
                for (size_t thread_tile_y_vector_idx{0U};
                     thread_tile_y_vector_idx < VECTORIZED_THREAD_TILE_SIZE_Y;
                     ++thread_tile_y_vector_idx)
                {
                    *reinterpret_cast<int4*>(
                        &A_vals[thread_tile_repeat_row_idx]
                               [thread_tile_y_vector_idx * NUM_VECTOR_UNITS]) =
                        *reinterpret_cast<int4 const*>(
                            &A_thread_block_tile_transposed
                                [A_thread_block_tile_col_idx]
                                [A_thread_block_tile_row_idx +
                                 thread_tile_y_vector_idx * NUM_VECTOR_UNITS]);
                }
            }
            
            // B 의 데이터를 레지스터로 로드한다
#pragma unroll
            for (size_t thread_tile_repeat_col_idx{0U};
                 thread_tile_repeat_col_idx < NUM_THREAD_TILES_PER_WARP_X;
                 ++thread_tile_repeat_col_idx)
            {
                size_t const B_thread_block_tile_row_idx{k_i};
                size_t const B_thread_block_tile_col_idx{
                    warp_col_idx * WARP_TILE_SIZE_X +
                    thread_tile_repeat_col_idx *
                        (WARP_TILE_SIZE_X / NUM_THREAD_TILES_PER_WARP_X) +
                    thread_linear_col_idx_in_warp * THREAD_TILE_SIZE_X};
                // B 의 데이터를 벡터화하여 로드한다
#pragma unroll
                for (size_t thread_tile_x_vector_idx{0U};
                     thread_tile_x_vector_idx < VECTORIZED_THREAD_TILE_SIZE_X;
                     ++thread_tile_x_vector_idx)
                {
                    *reinterpret_cast<int4*>(
                        &B_vals[thread_tile_repeat_col_idx]
                               [thread_tile_x_vector_idx * NUM_VECTOR_UNITS]) =
                        *reinterpret_cast<int4 const*>(
                            &B_thread_block_tile[B_thread_block_tile_row_idx]
                                                [B_thread_block_tile_col_idx +
                                                 thread_tile_x_vector_idx *
                                                     NUM_VECTOR_UNITS]);
                }
            }

            // NUM_THREAD_TILES_PER_WARP_Y * NUM_THREAD_TILES_PER_WARP_X 개의 외적을 계산한다
#pragma unroll
            for (size_t thread_tile_repeat_row_idx{0U};
                 thread_tile_repeat_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
                 ++thread_tile_repeat_row_idx)
            {
#pragma unroll
                for (size_t thread_tile_repeat_col_idx{0U};
                     thread_tile_repeat_col_idx < NUM_THREAD_TILES_PER_WARP_X;
                     ++thread_tile_repeat_col_idx)
                {
                    // 스레드 타일 수준의 행렬 곱셈을 수행한다
#pragma unroll
                    for (size_t thread_tile_y_idx{0U};
                         thread_tile_y_idx < THREAD_TILE_SIZE_Y;
                         ++thread_tile_y_idx)
                    {
#pragma unroll
                        for (size_t thread_tile_x_idx{0U};
                             thread_tile_x_idx < THREAD_TILE_SIZE_X;
                             ++thread_tile_x_idx)
                        {
                            // 계산 결과를 누산한다
                            C_thread_results[thread_tile_repeat_row_idx]
                                            [thread_tile_repeat_col_idx]
                                            [thread_tile_y_idx]
                                            [thread_tile_x_idx] +=
                                A_vals[thread_tile_repeat_row_idx]
                                      [thread_tile_y_idx] *
                                B_vals[thread_tile_repeat_col_idx]
                                      [thread_tile_x_idx];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // 결과를 DRAM 에 기록한다
#pragma unroll
    for (size_t thread_tile_repeat_row_idx{0U};
         thread_tile_repeat_row_idx < NUM_THREAD_TILES_PER_WARP_Y;
         ++thread_tile_repeat_row_idx)
    {
#pragma unroll
        for (size_t thread_tile_repeat_col_idx{0U};
             thread_tile_repeat_col_idx < NUM_THREAD_TILES_PER_WARP_X;
             ++thread_tile_repeat_col_idx)
        {
#pragma unroll
            for (size_t thread_tile_y_idx{0U};
                 thread_tile_y_idx < THREAD_TILE_SIZE_Y; ++thread_tile_y_idx)
            {
#pragma unroll
                for (size_t thread_tile_x_vector_idx{0U};
                     thread_tile_x_vector_idx < VECTORIZED_THREAD_TILE_SIZE_X;
                     ++thread_tile_x_vector_idx)
                {
                    // 출력 행렬 C 의 인덱스를 계산한다
                    size_t const C_row_idx{
                        blockIdx.y * BLOCK_TILE_SIZE_Y +
                        warp_row_idx * WARP_TILE_SIZE_Y +
                        thread_tile_repeat_row_idx *
                            (WARP_TILE_SIZE_Y / NUM_THREAD_TILES_PER_WARP_Y) +
                        thread_linear_row_idx_in_warp * THREAD_TILE_SIZE_Y +
                        thread_tile_y_idx};
                    size_t const C_col_idx{
                        blockIdx.x * BLOCK_TILE_SIZE_X +
                        warp_col_idx * WARP_TILE_SIZE_X +
                        thread_tile_repeat_col_idx *
                            (WARP_TILE_SIZE_X / NUM_THREAD_TILES_PER_WARP_X) +
                        thread_linear_col_idx_in_warp * THREAD_TILE_SIZE_X +
                        thread_tile_x_vector_idx * NUM_VECTOR_UNITS};

                    // 경계 검사
                    if (C_row_idx < m && C_col_idx < n)
                    {
                        // 원래 C 값을 벡터화하여 읽는다
                        int4 C_vals{*reinterpret_cast<int4 const*>(
                            &C[C_row_idx * ldc + C_col_idx])};
                        // alpha 와 beta 계수를 적용하여 C 값을 갱신한다
#pragma unroll
                        for (size_t i{0U}; i < NUM_VECTOR_UNITS; ++i)
                        {
                            reinterpret_cast<T*>(&C_vals)[i] =
                                alpha *
                                    C_thread_results[thread_tile_repeat_row_idx]
                                                    [thread_tile_repeat_col_idx]
                                                    [thread_tile_y_idx]
                                                    [thread_tile_x_vector_idx *
                                                         NUM_VECTOR_UNITS +
                                                     i] +
                                beta * reinterpret_cast<T const*>(&C_vals)[i];
                        }
                        // 결과를 벡터화하여 다시 기록한다
                        *reinterpret_cast<int4*>(
                            &C[C_row_idx * ldc + C_col_idx]) = C_vals;
                    }
                }
            }
        }
    }
}

// GEMM 커널 v06 벡터화 버전의 실행 함수
template <typename T>
void launch_gemm_kernel_v06_vectorized(size_t m, size_t n, size_t k,
                                       T const* alpha, T const* A, size_t lda,
                                       T const* B, size_t ldb, T const* beta,
                                       T* C, size_t ldc, cudaStream_t stream)
{
    // 블록 타일 크기는 자유롭게 조정할 수 있다
    // 알고리즘의 정확성은 항상 보장되어야 한다
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};  // 블록의 X 방향 크기
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};  // 블록의 Y 방향 크기
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};   // 블록의 K 방향 크기

    constexpr unsigned int WARP_TILE_SIZE_X{32U};    // warp 의 X 방향 크기
    constexpr unsigned int WARP_TILE_SIZE_Y{64U};    // warp 의 Y 방향 크기
    constexpr unsigned int NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    constexpr unsigned int NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);

    constexpr unsigned int THREAD_TILE_SIZE_X{8U};   // 스레드의 X 방향 타일 크기
    constexpr unsigned int THREAD_TILE_SIZE_Y{8U};   // 스레드의 Y 방향 타일 크기

    constexpr unsigned int NUM_THREADS_PER_WARP_X{4U};  // warp 당 X 방향 스레드 수
    constexpr unsigned int NUM_THREADS_PER_WARP_Y{8U};  // warp 당 Y 방향 스레드 수
    static_assert(NUM_THREADS_PER_WARP_X * NUM_THREADS_PER_WARP_Y == 32U);
    static_assert(
        WARP_TILE_SIZE_X % (THREAD_TILE_SIZE_X * NUM_THREADS_PER_WARP_X) == 0U);
    static_assert(
        WARP_TILE_SIZE_Y % (THREAD_TILE_SIZE_Y * NUM_THREADS_PER_WARP_Y) == 0U);

    constexpr unsigned int NUM_THREADS_X{NUM_WARPS_X * NUM_THREADS_PER_WARP_X};
    constexpr unsigned int NUM_THREADS_Y{NUM_WARPS_Y * NUM_THREADS_PER_WARP_Y};

    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_THREADS_X * NUM_THREADS_Y};

    // 블록과 그리드 차원을 설정한다
    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    
    // 커널을 실행한다
    gemm_v06_vectorized<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                        BLOCK_TILE_SIZE_K, WARP_TILE_SIZE_X, WARP_TILE_SIZE_Y,
                        THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y,
                        NUM_THREADS_PER_WARP_X, NUM_THREADS_PER_WARP_Y>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                              *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
```

이 FP32 GEMM 구현의 성능은 NVIDIA GeForce RTX 3090 GPU 에서 20.16 TFLOPS 에 이른다. cuBLAS FP32 GEMM 성능 24.59 TFLOPS 와 비교하면, 이 구현은 이미 상당히 잘 최적화되어 있다.


## 2D 블록 타일링과 2D warp 타일링과 Tensor Core 와 벡터화 메모리 접근을 사용하는 구현

이미 GEMM CUDA 커널을 warp 중심으로 구성했고 NVIDIA Tensor Core 명령이 warp 수준에서 인터페이스되므로, NVIDIA Tensor Core(https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/) WMMA API 를 활용해 GEMM 계산을 추가로 가속하는 것은 매우 간단하다. NVIDIA Tensor Core 는 IEEE FP32 계산을 지원하지 않으므로, 이 CUDA 커널은 FP16 GEMM 을 수행하도록 한다.

2D 블록 타일링과 2D warp 타일링과 2D 스레드 타일링과 벡터화 메모리 접근을 사용하는 구현과 비교하면, 2D 블록 타일링과 2D warp 타일링과 Tensor Core 와 벡터화 메모리 접근을 사용하는 구현이 더 간단하다. 스레드 타일링 과정이 NVIDIA Tensor Core 의 warp 수준 WMMA API 로 추상화되었기 때문이다.


수학적으로, 행렬 곱셈-누산 연산 $D_{b_m,b_n}^{d_{bm} \times d_{bn}} = \sum_{b_k=1}^{k/d_{bk}} A_{b_m,b_k}^{d_{bm} \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_{bn}} + C_{b_m,b_n}^{d_{bm} \times d_{bn}}$(여기서 $D_{b_m,b_n} \in \mathbb{R}^{d_{bm} \times d_{bn}}$, $A_{b_m,b_k} \in \mathbb{R}^{d_{bm} \times d_{bk}}$, $B_{b_k,b_n} \in \mathbb{R}^{d_{bk} \times d_{bn}}$, $C_{b_m,b_n} \in \mathbb{R}^{d_{bm} \times d_{bn}}$)가 주어지면, 이 행렬들은 더 작은 행렬들로 분할할 수 있다.

$$A_{b_m,b_k}^{d_{bm} \times d_{bk}} = \begin{bmatrix}
\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{1,1} & \left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{1,2} & \cdots & \left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{1,d_{bk}/d_{wk}} \\
\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{2,1} & \left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{2,2} & \cdots & \left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{2,d_{bk}/d_{wk}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{d_{bm}/d_{wm},1} & \left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{d_{bm}/d_{wm},2} & \cdots & \left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{d_{bm}/d_{wm},d_{bk}/d_{wk}}
\end{bmatrix}$$

$$B_{b_k,b_n}^{d_{bk} \times d_{bn}} = \begin{bmatrix}
\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{1,1} & \left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{1,2} & \cdots & \left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{1,d_{bn}/d_{wn}} \\
\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{2,1} & \left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{2,2} & \cdots & \left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{2,d_{bn}/d_{wn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{d_{bk}/d_{wk},1} & \left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{d_{bk}/d_{wk},2} & \cdots & \left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{d_{bk}/d_{wk},d_{bn}/d_{wn}}
\end{bmatrix}$$

$$C_{b_m,b_n}^{d_{bm} \times d_{bn}} = \begin{bmatrix}
\left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{1,1} & \left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{1,2} & \cdots & \left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{1,d_{bn}/d_{wn}} \\
\left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{2,1} & \left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{2,2} & \cdots & \left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{2,d_{bn}/d_{wn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{d_{bm}/d_{wm},1} & \left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{d_{bm}/d_{wm},2} & \cdots & \left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{d_{bm}/d_{wm},d_{bn}/d_{wn}}
\end{bmatrix}$$

$$D_{b_m,b_n}^{d_{bm} \times d_{bn}} = \begin{bmatrix}
\left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{1,1} & \left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{1,2} & \cdots & \left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{1,d_{bn}/d_{wn}} \\
\left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{2,1} & \left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{2,2} & \cdots & \left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{2,d_{bn}/d_{wn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{d_{bm}/d_{wm},1} & \left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{d_{bm}/d_{wm},2} & \cdots & \left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{d_{bm}/d_{wm},d_{bn}/d_{wn}}
\end{bmatrix}$$

$D_{b_m,b_n}^{d_{bm} \times d_{bn}}$ 안의 각 작은 행렬은 여러 개의 작은 행렬 곱셈과 누산을 통해 계산된다.

$$\left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}} = \left(\sum_{b_k=1}^{k/d_{bk}} A_{b_m,b_k}^{d_{bm} \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_{bn}} + C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(A_{b_m,b_k}^{d_{bm} \times d_{bk}} B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}} + \left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(\sum_{w_k=1}^{d_{bk}/d_{wk}} \left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} \left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right) + \left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$$

블록 warp 인덱스가 $(w_m, w_n)$ 인 각 warp(여기서 $w_m \in [1, d_{bm}/d_{wm}]$, $w_n \in [1, d_{bn}/d_{wn}]$)는, 블록 인덱스가 $(b_m, b_n)$ 인 블록(여기서 $b_m \in [1, m/d_{bm}]$, $b_n \in [1, n/d_{bn}]$) 안에서 하나의 작은 행렬 곱셈-누산 $\left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$ 을 계산하는 것을 담당한다.

Tensor Core WMMA GEMM 의 크기가 $d_{tm} \times d_{tn} \times d_{tk}$ 라고 하자. 행렬 $\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}$ 는 행 차원을 따라 $d_{wm}/d_{tm}$ 개의 조각으로, 행렬 $\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}$ 는 열 차원을 따라 $d_{wn}/d_{tn}$ 개의 조각으로 분할할 수 있으므로, 다음을 얻는다.

$$\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} = \begin{bmatrix}
\left(\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{1,1}^{d_{tm} \times d_{tk}} & \left(\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{1,2}^{d_{tm} \times d_{tk}} & \cdots & \left(\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{1,d_{wk}/d_{tk}}^{d_{tm} \times d_{tk}} \\
\left(\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{2,1}^{d_{tm} \times d_{tk}} & \left(\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{2,2}^{d_{tm} \times d_{tk}} & \cdots & \left(\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{2,d_{wk}/d_{tk}}^{d_{tm} \times d_{tk}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{d_{wm}/d_{tm},1}^{d_{tm} \times d_{tk}} & \left(\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{d_{wm}/d_{tm},2}^{d_{tm} \times d_{tk}} & \cdots & \left(\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{d_{wm}/d_{tm},d_{wk}/d_{tk}}^{d_{tm} \times d_{tk}}
\end{bmatrix}$$

$$\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}} = \begin{bmatrix}
\left(\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{1,1}^{d_{tk} \times d_{tn}} & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{1,2}^{d_{tk} \times d_{tn}} & \cdots & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{1,d_{wn}/d_{tn}}^{d_{tk} \times d_{tn}} \\
\left(\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{2,1}^{d_{tk} \times d_{tn}} & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{2,2}^{d_{tk} \times d_{tn}} & \cdots & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{2,d_{wn}/d_{tn}}^{d_{tk} \times d_{tn}} \\
\vdots & \vdots & \ddots & \vdots \\
\left(\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{d_{wk}/d_{tk},1}^{d_{tk} \times d_{tn}} & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{d_{wk}/d_{tk},2}^{d_{tk} \times d_{tn}} & \cdots & \left(\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{d_{wk}/d_{tk},d_{wn}/d_{tn}}^{d_{tk} \times d_{tn}}
\end{bmatrix}$$

각 warp 는 스레드 수준 명령을 호출하는 대신 WMMA warp 수준 Tensor Core 를 호출하여, 모든 $\left(\left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$ 를 계산하고 이를 반복적으로 누적해 $\left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}$ 를 계산한다.

$$\left(\left(D_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}} = \left(\sum_{b_k=1}^{k/d_{bk}} \left(\sum_{w_k=1}^{d_{bk}/d_{wk}} \left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}} \left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right) + \left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$$

$$= \sum_{b_k=1}^{k/d_{bk}} \left(\sum_{w_k=1}^{d_{bk}/d_{wk}} \left(\sum_{t_k=1}^{d_{wk}/d_{tk}} \left(\left(A_{b_m,b_k}^{d_{bm} \times d_{bk}}\right)_{w_m,w_k}^{d_{wm} \times d_{wk}}\right)_{t_m,t_k}^{d_{tm} \times d_{tk}} \left(\left(B_{b_k,b_n}^{d_{bk} \times d_{bn}}\right)_{w_k,w_n}^{d_{wk} \times d_{wn}}\right)_{t_k,t_n}^{d_{tk} \times d_{tn}}\right)\right) + \left(\left(C_{b_m,b_n}^{d_{bm} \times d_{bn}}\right)_{w_m,w_n}^{d_{wm} \times d_{wn}}\right)_{t_m,t_n}^{d_{tm} \times d_{tn}}$$

이 구현에서는 WMMA Tensor Core API 의 제약으로 인해 $d_{tm} = 16$, $d_{tn} = 16$, $d_{tk} = 16$ 이다.

아래 코드 조각은 2D 블록 타일링, 2D warp 타일링, Tensor Core, 벡터화 메모리 접근을 사용하는 구현을 보여준다.

```c++
// GEMM kernel v07.
// 각 스레드 블록 안의 각 스레드는 THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X 개의 출력 값을 처리한다
// 스레드 수는 BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_X / (THREAD_TILE_SIZE_Y * THREAD_TILE_SIZE_X) 이다
template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t BLOCK_TILE_SKEW_SIZE_X,
          size_t BLOCK_TILE_SKEW_SIZE_Y, size_t WARP_TILE_SIZE_X,
          size_t WARP_TILE_SIZE_Y, size_t WMMA_TILE_SIZE_X,
          size_t WMMA_TILE_SIZE_Y, size_t WMMA_TILE_SIZE_K, size_t NUM_THREADS>
__global__ void gemm_v07_vectorized(size_t m, size_t n, size_t k, T alpha,
                                    T const* A, size_t lda, T const* B,
                                    size_t ldb, T beta, T* C, size_t ldc)
{
    // X 방향의 warp 수를 계산한다
    constexpr size_t NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    // 블록 타일 크기가 warp 타일 크기로 나누어떨어지도록 보장한다
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);

    // 데이터 재사용을 위해 A 와 B 의 타일을 공유 메모리에 캐시한다
    // A 행렬 타일은 전치하여 저장하고, bank 충돌을 피하기 위해 skew 를 추가한다
    __shared__ T A_thread_block_tile_transposed[BLOCK_TILE_SIZE_K]
                                               [BLOCK_TILE_SIZE_Y +
                                                BLOCK_TILE_SKEW_SIZE_Y];
    // B 행렬 타일은 그대로 저장하고, bank 충돌을 피하기 위해 skew 를 추가한다
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X +
                                                        BLOCK_TILE_SKEW_SIZE_X];

    // 각 warp 의 X 및 Y 방향 WMMA 타일 수를 계산한다
    constexpr size_t NUM_WMMA_TILES_X{WARP_TILE_SIZE_X / WMMA_TILE_SIZE_X};
    static_assert(WARP_TILE_SIZE_X % WMMA_TILE_SIZE_X == 0U);
    constexpr size_t NUM_WMMA_TILES_Y{WARP_TILE_SIZE_Y / WMMA_TILE_SIZE_Y};
    static_assert(WARP_TILE_SIZE_Y % WMMA_TILE_SIZE_Y == 0U);
    // K 방향의 WMMA 타일 수를 계산한다
    constexpr size_t NUM_WMMA_TILES_K{BLOCK_TILE_SIZE_K / WMMA_TILE_SIZE_K};
    static_assert(BLOCK_TILE_SIZE_K % WMMA_TILE_SIZE_K == 0U);

    // WMMA 프래그먼트를 선언한다
    // A 행렬 프래그먼트, 열 우선(col-major) 저장
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_SIZE_Y,
                           WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T,
                           nvcuda::wmma::col_major>
        a_frags[NUM_WMMA_TILES_Y];
    // B 행렬 프래그먼트, 행 우선(row-major) 저장
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_SIZE_Y,
                           WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T,
                           nvcuda::wmma::row_major>
        b_frags[NUM_WMMA_TILES_X];
    // 누산기 프래그먼트, 중간 계산 결과를 저장한다
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_SIZE_Y,
                           WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T>
        acc_frags[NUM_WMMA_TILES_Y][NUM_WMMA_TILES_X];
    // C 행렬 프래그먼트, 최종 결과의 로드와 저장에 사용한다
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_SIZE_Y,
                           WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_K, T>
        c_frag;

// 누산기가 0 에서 시작하도록 보장한다
#pragma unroll
    for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_Y;
         ++wmma_tile_row_idx)
    {
        for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_X;
             ++wmma_tile_col_idx)
        {
            // 누산기 프래그먼트를 0 으로 초기화한다
            nvcuda::wmma::fill_fragment(
                acc_frags[wmma_tile_row_idx][wmma_tile_col_idx],
                static_cast<T>(0));
        }
    }

    // 스레드의 선형 인덱스를 계산한다
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};
    // warp 의 선형 인덱스를 계산한다(각 warp 는 32 개의 스레드를 가진다)
    size_t const warp_linear_idx{thread_linear_idx / 32U};
    // warp 의 Y 방향 인덱스를 계산한다
    size_t const warp_row_idx{warp_linear_idx / NUM_WARPS_X};
    // warp 의 X 방향 인덱스를 계산한다
    size_t const warp_col_idx{warp_linear_idx % NUM_WARPS_X};

    // 내적 합을 수행하기 위한 외부 루프 횟수를 계산한다
    // C_thread_block_tile =
    // \sigma_{thread_block_tile_idx=0}^{num_thread_block_tiles-1} A[:,
    // thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
    // B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :]
    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};

    // 메인 루프: K 방향의 모든 타일을 순회한다
    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        // 벡터화 접근과 전치를 사용해 데이터를 전역 메모리에서 공유 메모리로 로드한다
        load_data_to_shared_memory_transposed_vectorized<
            T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            NUM_THREADS, BLOCK_TILE_SKEW_SIZE_X, BLOCK_TILE_SKEW_SIZE_Y>(
            A, lda, B, ldb, A_thread_block_tile_transposed, B_thread_block_tile,
            thread_block_tile_idx, thread_linear_idx, m, n, k);
        // 모든 스레드를 동기화하여 데이터 로드가 완료되었음을 보장한다
        __syncthreads();

// A[:, thread_block_tile_idx:BLOCK_TILE_SIZE_K] *
// B[thread_block_tile_idx:BLOCK_TILE_SIZE_K, :] 
// 여기서 A 와 B 의 타일은 이미 공유 메모리에 캐시되어 있다
// 이 내적은 다시 BLOCK_TILE_SIZE_K 개의 외적으로 분해된다
// A_thread_block_tile * B_thread_block_tile = 
// \sigma_{k_i=0}^{BLOCK_TILE_SIZE_K-1} A_thread_block_tile[:, k_i] @ B_thread_block_tile[k_i, :]
#pragma unroll
        for (size_t k_i{0U}; k_i < NUM_WMMA_TILES_K; ++k_i)
        {
#pragma unroll
            // Y 방향의 모든 WMMA 타일을 순회한다
            for (size_t wmma_tile_row_idx{0U};
                 wmma_tile_row_idx < NUM_WMMA_TILES_Y; ++wmma_tile_row_idx)
            {
                // 공유 메모리에서 A 행렬 프래그먼트를 로드한다
                nvcuda::wmma::load_matrix_sync(
                    a_frags[wmma_tile_row_idx],
                    &A_thread_block_tile_transposed[k_i * WMMA_TILE_SIZE_K]
                                                   [warp_row_idx *
                                                        WARP_TILE_SIZE_Y +
                                                    wmma_tile_row_idx *
                                                        WMMA_TILE_SIZE_Y],
                    BLOCK_TILE_SIZE_Y + BLOCK_TILE_SKEW_SIZE_Y);
#pragma unroll
                // X 방향의 모든 WMMA 타일을 순회한다
                for (size_t wmma_tile_col_idx{0U};
                     wmma_tile_col_idx < NUM_WMMA_TILES_X; ++wmma_tile_col_idx)
                {
                    // 이 로드 연산들은 매우 느려 성능에 심각한 영향을 준다
                    // 공유 메모리에서 B 행렬 프래그먼트를 로드한다
                    nvcuda::wmma::load_matrix_sync(
                        b_frags[wmma_tile_col_idx],
                        &B_thread_block_tile[k_i * WMMA_TILE_SIZE_K]
                                            [warp_col_idx * WARP_TILE_SIZE_X +
                                             wmma_tile_col_idx *
                                                 WMMA_TILE_SIZE_Y],
                        BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X);

                    // 행렬 곱셈-누산 연산을 수행한다
                    nvcuda::wmma::mma_sync(
                        acc_frags[wmma_tile_row_idx][wmma_tile_col_idx],
                        a_frags[wmma_tile_row_idx], b_frags[wmma_tile_col_idx],
                        acc_frags[wmma_tile_row_idx][wmma_tile_col_idx]);
                }
            }
        }
        // 모든 스레드를 동기화하여 계산이 완료되었음을 보장한다
        __syncthreads();
    }

// 결과를 DRAM 에 기록한다
#pragma unroll
    for (size_t wmma_tile_row_idx{0U}; wmma_tile_row_idx < NUM_WMMA_TILES_Y;
         ++wmma_tile_row_idx)
    {
#pragma unroll
        for (size_t wmma_tile_col_idx{0U}; wmma_tile_col_idx < NUM_WMMA_TILES_X;
             ++wmma_tile_col_idx)
        {
            // 전역 메모리에서 C 행렬 프래그먼트를 로드한다
            nvcuda::wmma::load_matrix_sync(
                c_frag,
                &C[(blockIdx.y * BLOCK_TILE_SIZE_Y +
                    warp_row_idx * WARP_TILE_SIZE_Y +
                    wmma_tile_row_idx * WMMA_TILE_SIZE_Y) *
                       n +
                   blockIdx.x * BLOCK_TILE_SIZE_X +
                   warp_col_idx * WARP_TILE_SIZE_X +
                   wmma_tile_col_idx * WMMA_TILE_SIZE_X],
                n, nvcuda::wmma::mem_row_major);
            // 스케일링과 덧셈 연산을 수행한다: C = alpha * A * B + beta * C
            for (size_t i{0}; i < c_frag.num_elements; ++i)
            {
                c_frag.x[i] =
                    alpha *
                        acc_frags[wmma_tile_row_idx][wmma_tile_col_idx].x[i] +
                    beta * c_frag.x[i];
            }
            // 프래그먼트를 전역 메모리에 다시 저장한다
            nvcuda::wmma::store_matrix_sync(
                &C[(blockIdx.y * BLOCK_TILE_SIZE_Y +
                    warp_row_idx * WARP_TILE_SIZE_Y +
                    wmma_tile_row_idx * WMMA_TILE_SIZE_Y) *
                       n +
                   blockIdx.x * BLOCK_TILE_SIZE_X +
                   warp_col_idx * WARP_TILE_SIZE_X +
                   wmma_tile_col_idx * WMMA_TILE_SIZE_X],
                c_frag, n, nvcuda::wmma::mem_row_major);
        }
    }
}

// GEMM 커널 v07 의 실행 함수
template <typename T>
void launch_gemm_kernel_v07_vectorized(size_t m, size_t n, size_t k,
                                       T const* alpha, T const* A, size_t lda,
                                       T const* B, size_t ldb, T const* beta,
                                       T* C, size_t ldc, cudaStream_t stream)
{
    // 블록 타일 크기는 자유롭게 조정할 수 있다
    // 알고리즘의 정확성은 항상 보장되어야 한다
    constexpr unsigned int BLOCK_TILE_SIZE_X{128U};  // 블록의 X 방향 타일 크기
    constexpr unsigned int BLOCK_TILE_SIZE_Y{128U};  // 블록의 Y 방향 타일 크기
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};   // 블록의 K 방향 타일 크기

    // skew 크기는 공유 메모리의 bank 충돌을 피하는 데 사용된다
    constexpr size_t BLOCK_TILE_SKEW_SIZE_X{16U};
    constexpr size_t BLOCK_TILE_SKEW_SIZE_Y{16U};

    // warp 타일 크기
    constexpr unsigned int WARP_TILE_SIZE_X{32U};    // warp 의 X 방향 타일 크기
    constexpr unsigned int WARP_TILE_SIZE_Y{64U};    // warp 의 Y 방향 타일 크기
    // X 및 Y 방향의 warp 수를 계산한다
    constexpr unsigned int NUM_WARPS_X{BLOCK_TILE_SIZE_X / WARP_TILE_SIZE_X};
    constexpr unsigned int NUM_WARPS_Y{BLOCK_TILE_SIZE_Y / WARP_TILE_SIZE_Y};
    // 블록 타일 크기가 warp 타일 크기로 나누어떨어지도록 보장한다
    static_assert(BLOCK_TILE_SIZE_X % WARP_TILE_SIZE_X == 0U);
    static_assert(BLOCK_TILE_SIZE_Y % WARP_TILE_SIZE_Y == 0U);

    // WMMA 타일 크기(16x16x16 으로 고정)
    constexpr unsigned int WMMA_TILE_SIZE_X{16U};
    constexpr unsigned int WMMA_TILE_SIZE_Y{16U};
    constexpr unsigned int WMMA_TILE_SIZE_K{16U};

    // 블록당 스레드 수
    constexpr unsigned int NUM_THREADS_PER_BLOCK{NUM_WARPS_X * NUM_WARPS_Y *
                                                 32U};

    // 블록과 그리드 차원을 구성한다
    dim3 const block_dim{NUM_THREADS_PER_BLOCK, 1U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) /
            BLOCK_TILE_SIZE_X,
        (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) /
            BLOCK_TILE_SIZE_Y,
        1U};
    // 커널을 실행한다
    gemm_v07_vectorized<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                        BLOCK_TILE_SIZE_K, BLOCK_TILE_SKEW_SIZE_X,
                        BLOCK_TILE_SKEW_SIZE_Y, WARP_TILE_SIZE_X,
                        WARP_TILE_SIZE_Y, WMMA_TILE_SIZE_X, WMMA_TILE_SIZE_Y,
                        WMMA_TILE_SIZE_K, NUM_THREADS_PER_BLOCK>
        <<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                              *beta, C, ldc);
    CHECK_LAST_CUDA_ERROR();
}
```

기본 WMMA 크기가 $16\times 16 \times 16$ 이므로, 같은 warp 안의 32 개 스레드는 공유 메모리에 캐시된 WMMA 프래그먼트에 협력하여 접근해야 한다. 따라서 공유 메모리 bank 충돌이 발생하기 쉽다. 공유 메모리 bank 충돌을 피하려면 공유 메모리 크기를 패딩하여 충돌이 발생하지 않도록 해야 한다. 이것이 바로 선행 차원에서 skew 크기를 사용해 공유 메모리 크기를 패딩해야 하는 이유이다.

이 FP16 GEMM 구현의 성능은 NVIDIA GeForce RTX 3090 GPU 에서 46.78 TFLOPS 에 이른다. cuBLAS FP16 GEMM 성능 138.95 TFLOPS 와 비교하면, 이 구현은 cuBLAS FP16 GEMM 성능의 33.7% 만을 달성한다. 이 구현의 성능 최적화는 향후 과제로 남겨둔다.

## 결론

우리가 GEMM CUDA 커널에 적용한 최적화는 주로 “CUTLASS: Fast Linear Algebra in CUDA C++”(https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)의 도표를 따른다.

![](https://files.mdnice.com/user/59/5681a7ca-2fad-47d3-8068-c0f94e329955.png)

2D 블록 타일링, 2D warp 타일링, 2D 스레드 타일링, 벡터화 메모리 접근 등의 최적화 기법을 사용하면, NVIDIA GeForce RTX 3090 GPU 에서 20.16 TFLOPS 의 FP32 GEMM 성능을 달성할 수 있으며, 이는 cuBLAS FP32 GEMM 성능의 약 80% - 90% 에 해당한다.

## 소스 코드

GEMM CUDA 커널의 소스 코드는 필자의 GitHub 저장소 “CUDA GEMM Optimization”(https://github.com/leimao/CUDA-GEMM-Optimization/)에서 찾을 수 있다.

## 참고

- CUTLASS: Fast Linear Algebra in CUDA C++(https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
- CUDA GEMM Optimization - GitHub(https://github.com/leimao/CUDA-GEMM-Optimization/)
- CUDA Matrix Multiplication(https://leimao.github.io/blog/CUDA-Matrix-Multiplication/)
- CUDA Vectorized Memory Access(https://leimao.github.io/blog/CUDA-Vectorized-Memory-Access/)
- CUDA Data Alignment(https://leimao.github.io/blog/CUDA-Data-Alignment/)
- CUDA Shared Memory Bank(https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/)
- NVIDIA Tensor Core Programming(https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/)
- CUDA Tensor Core GEMM(https://github.com/NVIDIA/cuda-samples/blob/e8568c417356f7e66bb9b7130d6be7e55324a519/Samples/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu)
- How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog(https://siboehm.com/articles/22/CUDA-MMM)





