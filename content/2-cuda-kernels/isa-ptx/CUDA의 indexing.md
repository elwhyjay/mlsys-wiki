> 이 글은 @Simon V(https://github.com/simveit)의 허가를 받아 전재 및 번역하여 본 공중계정에 게시한다. 원문 주소는 https://veitner.bearblog.dev/indexing-in-cuda/ 이다.

# CUDA의 indexing

## 소개

이 블로그 글에서는 CUDA에서 행 우선(row-major) 형식의 행렬이 무엇을 의미하는지 설명하고자 한다. 이는 CUDA kernel과 kernel이 처리하는 행렬을 어떻게 indexing하는지 이해하는 데 매우 중요하다.

형상이 `(M, N)`인 2D 배열 `A`를 생각해 보자. CUDA에서 이런 배열은 기본적으로 행 우선 형식으로 선형화되어, 컴퓨터 메모리 공간의 평평한 구조에 맞춰진다. 실제로 이는 행렬 좌표 `(i,j)`가 `i * N + j`로 매핑된다는 뜻이다. 이 함수를 `f`라고 부르자.

이 공식을 보면 왜 이것을 행 우선이라고 부르는지 알 수 있다. 서로 다른 두 좌표의 메모리 매핑 차이를 살펴보자.

```shell
d = f(i2, j2)-f(i1,j1) = (i2-i1) * N + (j2-j1)
```

인접한 열에 대해서는 `d = 1`이고, 인접한 행에 대해서는 `d = N`임을 볼 수 있다.

이를 더 일반화하여 형상이 `(M1, M2, M3)`인 3D 배열로 확장할 수 있다. 여기서 좌표 `(i, j, l)`은 `l + M3 * (j + i * M2)= i * M2 * M3 + j * M3 + l`로 매핑된다.

## 코드 분석

이제 이 관점을 사용해 CUDA kernel의 indexing을 이해해 보자. 2D Block Tiling(https://siboehm.com/articles/22/CUDA-MMM)에 대한 완전한 설명으로는 2D Tiling 개념을 자세히 설명한 이 훌륭한 블로그 글을 추천한다. 여기서는 indexing 부분에 집중한다. 전체 코드(https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/5_kernel_2D_blocktiling.cuh)는 github에서 볼 수 있으며, 계속 읽기 전에 먼저 읽고 이해해 보아야 한다.

첫 번째 단계에서는 공유 메모리를 할당한다. 각 행렬은 2D 행렬로 해석할 수 있으며, `As`의 형상은 `(BM, BK)`, `Bs`의 형상은 `(BK, BN)`이다.

```c++
__shared__ float As[BM * BK];
__shared__ float Bs[BK * BN];
```

이 공유 메모리는 이후 전역 메모리의 해당 원소들로 채워진다.

우리가 자세히 분석하려는 코드는 다음 부분이다.

```c++
// calculate per-thread results
for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
  // block into registers
  for (uint i = 0; i < TM; ++i) {
    regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
  }
  for (uint i = 0; i < TN; ++i) {
    regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
  }
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      threadResults[resIdxM * TN + resIdxN] +=
          regM[resIdxM] * regN[resIdxN];
    }
  }
```

내가 처음 이 indexing을 보았을 때는 스스로 어떻게 유도해야 할지 확신이 없었지만, 실제로는 위의 설명을 사용하면 그리 어렵지 않다.

### As의 indexing 분석

```c++
(threadRow * TM + i) * BK + dotIdx
= threadRow * TM * BK + i * BK + dotIdx
```

여기서 `As`가 형상이 `(..., BK, TM)`인 3D 배열로 해석된다는 것을 알 수 있다. 위에서 원래 행렬의 형상이 `(BM, BK)`임을 알고 있고, 행렬은 여전히 동일하므로 3D 배열로 해석한 형상은 `(BM/TM, TM, BK)`가 된다. 배열 안의 원소 수가 유지되기 때문이다.

여기서 왜 이렇게 indexing하는지가 명확해진다. 우리는 메모리 블록을 한 번 더 Tiling하고 싶고, `(BM, BK)`를 `(BM/TM, TM, BK)`로 변환함으로써 이를 정확히 달성한다.

`threadRow * TM * BK + i * BK + dotIdx -> (threadRow, i, dotIdx)`.

- `const int threadRow = threadIdx.x / (BN / TN);` 즉, 이것은 하나의 warp에 대응한다. 하나의 warp는 하나의 BM/TM 블록을 처리한다.
- `i`: 이는 레지스터 배열의 인덱스에 대응한다.
- `dotIdx`: 이는 원래 2D 배열의 열 인덱스에 대응한다.

### Bs의 indexing 분석

`dotIdx * BN + threadCol * TN + i = dotIdx * BN/TN * TN + threadCol * TN + i`이다. 위와 비슷한 기법을 사용하면 `Bs`를 형상이 `(BK, BN/TN, TN)`인 3D 배열로 해석한다는 것을 알 수 있다. `dotIdx * BN/TN * TN + threadCol * TN + i -> (dotIdx, threadCol, i)`

- `const int threadCol = threadIdx.x % (BN / TN)`; 즉, 이것은 하나의 thread에 대응한다. 이 경우 thread는 한 열을 처리하는 warp 안의 한 원소이다.
- `i`: 이는 레지스터 배열의 인덱스에 대응한다.
- `dotIdx`: 이는 원래 2D 배열의 열 인덱스에 대응한다.

### 전체 이해

이제 전체 loop를 이해할 수 있다. `float threadResults[TM * TN] = {0.0}`; 따라서 결과는 처음에는 `(TM, TN)`인 2D 배열이다.

```c++
threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
```

알고리즘은 다음과 같이 동작한다.

- 형상이 `(BM, BK)`와 `(BK, BN)`인 두 행렬 `As`와 `Bs`를 곱하려 한다.
- `As`를 `(BM/TM, BK, TM)`로 보고, `Bs`를 `(BK, BN/TN, TN)`으로 본다.
- `As`의 첫 번째 차원과 `Bs`의 두 번째 차원은 하나의 warp가 처리한다.
- `BK` 차원에서 `dotIdx`를 순회한다.
- `TM`을 순회하며 `TM`개의 원소를 레지스터에 쓴다. 각 원소는 하나의 warp와 하나의 `dotIdx`에 대응한다.
- `TN`을 순회하며 `TN`개의 원소를 레지스터에 쓴다. 각 원소는 하나의 warp와 하나의 `dotIdx`에 대응한다.
- 레지스터 안의 원소들을 곱해 모든 `k`의 결과를 크기가 `(TM, TN)`인 2D 배열에 누적한다.

이 작업을 수행한 뒤 결과는 아래와 같이 결과 행렬에 기록된다.

```c++
for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
  for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
    C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
        alpha * threadResults[resIdxM * TN + resIdxN] +
        beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
  }
}
```

`(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN = (threadRow * TM + resIdxM) * N/TN * TN + threadCol * TN + resIdxN`이다. `C`는 처음에 형상이 `(M, N)`인 행렬로 간주되지만, 이제 형상이 `(M/TM, TM, N/TN, TN)`인 행렬로 해석된다. 이는 우리가 메모리에 다시 쓸 때도 Tiling 방식으로 쓴다는 뜻이다.

`(threadRow * TM + resIdxM) * N/TN * TN + threadCol * TN + resIdxN -> (threadRow, resIdxM, threadCol, resIdxN)`이다. 따라서 각 warp는 한 행과 한 열을 쓴다.

이 블로그 글이 CUDA의 indexing을 더 잘 이해하는 데 도움이 되기를 바란다.
