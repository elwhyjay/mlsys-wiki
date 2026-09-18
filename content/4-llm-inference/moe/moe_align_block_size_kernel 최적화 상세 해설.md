# sgl-kernel MoE Align Block Size Kernel 최적화 과정 분석

## 0x0. 서론

이 글은 SGLang의 sgl-kernel에서 `moe_align_kernel.cu`를 최적화한 과정을 기록한 것이다(https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/moe/moe_align_kernel.cu). MoE 모델에는 아주 핵심적인 kernel이 하나 있는데 바로 `moe_align_block_size`이다. 이 kernel은 token들을 expert별로 그룹화해 정렬(align)하여 뒤따르는 expert 계산을 준비하는 역할을 한다.

이 kernel은 최초의 baseline 버전에서 지금에 이르기까지 쭉 최적화되며 몇 개의 버전을 거쳐 왔다:
- 0x1 Baseline: 최초의 구현이다. expert가 적은 경우(num_expert <= 64 && token <= 1024)에는 기본적으로 vLLM의 구현을 그대로 따랐고, 메모리 접근 병합(coalesced access)에 관한 조정을 조금 했다. 그 외의 경우에는 kernel을 새로 만들어 warp 단위로 처리했다.
- 0x2: 벡터화된 padding 연산을 추가했다
- 0x3: Blelloch Scan 알고리즘으로 prefix sum 계산을 병렬화했다
- 0x4: 여기서 더 나아가 Warp Scan으로 동기화 오버헤드를 줄였다. 현재 성능이 가장 좋은 버전이다

**짚고 넘어갈 점은 Baseline 버전은 내가 완성했다는 것이다. 그리고 0x2와 0x3의 핵심 최적화는 https://github.com/ispobock 이 완성했다. 0x4의 핵심 최적화는 https://github.com/yuan-luo 가 완성했다.**

아래에서 각 버전의 최적화 아이디어와 구현 세부 사항을 자세히 설명한다.

## 0x1. Baseline Kernel 상세 분석

### 이 kernel은 도대체 무엇을 하는가

간단히 말하면 이 kernel은 4가지 일을 한다:
1. 각 expert가 token을 몇 개 가지는지 센다
2. 정렬된 prefix sum을 계산한다(block_size에 맞춰 정렬해야 한다)
3. expert_ids 배열을 생성해 각 block이 어느 expert에 대응하는지 기록한다
4. token들을 expert별로 그룹화해 정렬한다

Baseline 버전은 3개의 kernel로 이 작업들을 수행했다:

#### 1. `moe_align_block_size_kernel` - 메인 kernel

```cpp
template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,      // 입력: 각 token에 대응하는 expert id
    int32_t* __restrict__ sorted_token_ids,     // 출력: 정렬된 token 인덱스
    int32_t* __restrict__ expert_ids,           // 출력: 각 block에 대응하는 expert id
    int32_t* __restrict__ total_tokens_post_pad,// 출력: 정렬 후의 총 token 수
    int32_t num_experts,                        // expert 총 개수
    int32_t padded_num_experts,                 // warp_size에 맞춰 정렬한 expert 수
    int32_t experts_per_warp,                   // warp 하나가 처리하는 expert 개수
    int32_t block_size,                         // 정렬 기준 block 크기
    size_t numel,                               // 입력 token 총 개수
    int32_t* __restrict__ cumsum) {             // 출력: prefix sum 배열
  
  extern __shared__ int32_t shared_counts[];
  
  // 단계1: shared memory 카운터 초기화
  // 각 warp는 experts_per_warp개의 expert에 대한 카운팅을 담당한다
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;
  
  // 현재 warp가 담당하는 expert의 카운트를 0으로 초기화
  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      shared_counts[warp_id * experts_per_warp + i] = 0;
    }
  }
  
  __syncthreads();
  
  // 단계2: 각 expert의 token 개수를 센다
  // 모든 스레드가 협력해 전체 token을 순회하며 atomic add로 카운팅한다
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;
  
  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];  // 현재 token의 expert id를 가져온다
    // 해당 expert가 shared_counts에서 차지하는 위치를 계산
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    // atomic add 연산으로 해당 expert의 token 개수를 센다
    atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
  }
  
  __syncthreads();
  
  // 단계3: prefix sum 계산(thread 0만 실행)
  // prefix sum은 각 expert가 출력에서 시작하는 위치를 정하는 데 쓰인다
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      int expert_count = 0;
      int warp_idx = (i - 1) / experts_per_warp;
      int expert_offset = (i - 1) % experts_per_warp;
      expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
      
      // block_size에 맞춰 정렬: CEILDIV(count, block_size) * block_size
      // 이렇게 하면 각 expert의 token 수가 block_size의 배수임을 보장할 수 있다
      cumsum[i] = cumsum[i - 1] + CEILDIV(expert_count, block_size) * block_size;
    }
    *total_tokens_post_pad = cumsum[num_experts];
  }
  
  __syncthreads();
  
  // 단계4: expert_ids 배열 채우기
  // expert_ids[i]는 i번째 block에 대응하는 expert 번호를 나타낸다
  if (threadIdx.x < num_experts) {
    // 각 스레드가 expert 하나를 담당한다
    // cumsum[threadIdx.x]부터 cumsum[threadIdx.x+1]까지의 모든 block은 expert threadIdx.x에 속한다
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }
}
```

이 kernel의 핵심 설계 몇 가지:
- shared memory에 각 expert의 token 카운트를 저장해 global memory 접근을 줄인다
- 카운팅할 때 atomic 연산을 사용해 여러 스레드가 동시에 쓸 때 생기는 문제를 피한다
- prefix sum 계산은 직렬이라 thread 0만 일하고 있는데, 이것이 뒤에 나올 최적화의 핵심 대상이다
- expert_ids 채우기는 병렬이며 각 스레드가 expert 하나를 담당한다

#### 2. `count_and_sort_expert_tokens_kernel` - 정렬 kernel

```cpp
template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    size_t numel) {
  
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  
  // 모든 token을 순회한다
  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    // atomic add로 현재 token이 해당 expert 안에서 가질 위치를 얻는다
    // cumsum_buffer[expert_id]는 해당 expert에 이미 배치된 token 개수를 기록한다
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    // token 인덱스 i를 대응하는 위치에 넣는다
    sorted_token_ids[rank_post_pad] = i;
  }
}
```

이 kernel은 token들을 expert별로 그룹화해 정렬하는 것으로, atomic 연산으로 스레드 안전성을 보장한다.

#### 3. `moe_align_block_size_small_batch_expert_kernel` - 소규모 최적화 버전

```cpp
template <typename scalar_t>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel) {
  
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;
  
  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;  // prefix sum 배열
  int32_t* tokens_cnts = (int32_t*)(shared_mem + num_experts + 1);  // token 카운트 배열
  
  // 단계1: 각 스레드의 로컬 카운터를 초기화
  // tokens_cnts 레이아웃: [blockDim.x+1][num_experts]
  // tokens_cnts[(threadIdx.x + 1) * num_experts + i] 는 스레드 threadIdx.x가 expert i에 대해 센 값을 저장한다
  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(threadIdx.x + 1) * num_experts + i] = 0;
  }
  
  // 단계2: 각 스레드가 자신이 담당하는 token들을 센다
  for (size_t i = tid; i < numel; i += stride) {
    ++tokens_cnts[(threadIdx.x + 1) * num_experts + topk_ids[i]];
  }
  
  __syncthreads();
  
  // 단계3: 각 expert에 대해 모든 스레드의 카운트를 누적한다(prefix sum)
  if (threadIdx.x < num_experts) {
    tokens_cnts[threadIdx.x] = 0;
    for (int i = 1; i <= blockDim.x; ++i) {
      // prefix sum을 누적한다. tokens_cnts[i * num_experts + threadIdx.x]는 최종적으로
      // 앞의 i개 스레드가 expert threadIdx.x에 대해 센 총 개수를 저장한다
      tokens_cnts[i * num_experts + threadIdx.x] += 
          tokens_cnts[(i - 1) * num_experts + threadIdx.x];
    }
  }
  
  __syncthreads();
  
  // 단계4: 정렬 후의 prefix sum을 계산
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] + 
          CEILDIV(tokens_cnts[blockDim.x * num_experts + i - 1], block_size) * block_size;
    }
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }
  
  __syncthreads();
  
  // 단계5: expert_ids 채우기
  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }
  
  // 단계6: token 정렬(kernel 안에서 바로 끝내 추가 kernel 호출을 피한다)
  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    // 현재 token이 출력에서 가질 위치를 계산
    // tokens_cnts[threadIdx.x * num_experts + expert_id]: 현재 스레드보다 앞선 스레드들이 해당 expert에 대해 센 값
    // cumsum[expert_id]: 해당 expert가 출력에서 시작하는 위치
    int32_t rank_post_pad = tokens_cnts[threadIdx.x * num_experts + expert_id] + 
                            cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    // 다음 token을 위해 카운트를 갱신
    ++tokens_cnts[threadIdx.x * num_experts + expert_id];
  }
}
```

이 버전은 소규모 시나리오, 즉 `numel < 1024 && num_experts <= 64` 인 경우에 적합하다.

주요 최적화 포인트:
- 카운팅, prefix sum, 정렬을 전부 하나의 kernel로 융합했다
- 스레드 로컬 카운트를 사용해 atomic 연산의 오버헤드를 피했다
- kernel 실행 횟수를 줄였다

### Baseline 버전의 성능 병목은 어디인가

분석해 보면 주로 2가지 문제가 있다:
1. prefix sum 계산이 직렬이라 thread 0만 일하고 나머지 스레드는 놀고 있어 병렬성을 전혀 활용하지 못한다
2. 카운팅 단계에서 atomicAdd를 대량으로 사용하는데, atomic 연산의 오버헤드가 적지 않다

---

## 0x2. 벡터화 Padding 추가

PR: https://github.com/sgl-project/sglang/pull/7437

### 이 버전에서 무엇을 바꿨는가

Baseline 버전에는 또 하나의 문제가 있었는데, `sorted_token_ids`를 python 레이어에서 numel로 초기화한다는 점이다. 즉 fill kernel을 하나 더 호출하는 셈이 된다. 0x2 버전은 벡터화된 padding 연산을 추가해 kernel 안에서 바로 padding을 수행함으로써 fill에 드는 비용을 줄였다:

```cpp
#define VEC_SIZE 4
using Vec = AlignedArray<int32_t, VEC_SIZE>;

// moe_align_block_size_kernel에 새로 추가된 padding 코드
if (pad_sorted_token_ids) {
    int32_t fill_val = static_cast<int32_t>(numel);  // numel을 채움 값으로 사용
    int32_t total = *total_tokens_post_pad;
    
    // 벡터화된 채움 값을 준비
    Vec fill_vec;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      fill_vec.data[i] = fill_val;
    }
    
    // 벡터화 쓰기, 한 번에 int32_t 4개를 쓴다
    int32_t total_vec_count = (total + VEC_SIZE - 1) / VEC_SIZE;
    Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
    
    for (int32_t idx = tid; idx < total_vec_count; idx += stride) {
      out_ptr[idx] = fill_vec;  // 한 번에 16바이트 쓰기
    }
  }
```

왜 이렇게 하는가:
- 벡터화 메모리 접근: `int4`로 한 번에 int32_t 4개를 쓰면 메모리 대역폭 이용률이 4배 올라간다
- 메모리 트랜잭션 병합: 벡터화 쓰기는 여러 메모리 트랜잭션을 병합할 수 있어 지연이 줄어든다
- 채움 값으로 `numel`을 쓰면 뒤에서 어느 것이 padding인지 식별할 수 있다

`AlignedArray`라는 템플릿 클래스의 역할은 배열이 16바이트 정렬되도록 보장하는 것이며, 사실상 int4와 동등하다. 이로써 컴파일러가 더 효율적인 벡터화 명령어를 생성할 수 있다.

두 kernel 모두에 padding 지원을 추가했다: 메인 kernel은 expert_ids를 계산한 뒤 벡터화 방식으로 sorted_token_ids 배열 전체를 채우고, 뒤따르는 정렬 kernel이 유효한 위치를 덮어쓴다. 소규모 배치 kernel은 정렬 전에 먼저 padding을 하고, 그다음 정렬 연산이 유효한 token 위치를 덮어쓴다.

이 최적화의 이점은 벡터화 쓰기로 메모리 대역폭 이용률을 높이고, 출력 데이터의 결정성을 보장하며, padding 연산이 다른 계산과 병렬로 진행되어 추가 오버헤드가 거의 없다는 것이다.

---

## 0x3. Blelloch Scan으로 prefix sum 병렬화

PR: https://github.com/sgl-project/sglang/pull/7794

### 이 버전이 해결한 문제

0x2 버전에서도 prefix sum은 여전히 직렬이라 thread 0만 일하고 있었고, 이것이 가장 큰 성능 병목이었다. 0x3 버전은 Blelloch Scan 알고리즘을 도입해 prefix sum 계산을 병렬화했다.

### Blelloch Scan 알고리즘은 어떻게 동작하는가

Blelloch Scan은 아주 고전적인 병렬 prefix sum 알고리즘으로, 두 단계로 나뉜다:

- 단계1: Up-Sweep (Reduce Phase)

합 트리를 구성하며 아래에서 위로 부분합을 계산한다:

```
원본 데이터: [3, 1, 7, 0, 4, 1, 6, 3]
         
Step 1:   [3, 4, 7, 7, 4, 5, 6, 9]  // 인접한 원소끼리 둘씩 더한다
Step 2:   [3, 4, 7, 11, 4, 5, 6, 14] // 간격 2로 더한다
Step 3:   [3, 4, 7, 11, 4, 5, 6, 25] // 간격 4로 더해 총합을 얻는다
```

- 단계2: Down-Sweep (Distribution Phase)

위에서 아래로 prefix sum을 분배한다. 이 단계의 핵심 아이디어는 총합을 각 위치로 분배하여, 각 위치보다 앞에 있는 모든 원소의 합을 계산하는 것이다.

상세 단계 설명:

```
Up-sweep 종료 후: [3, 4, 7, 11, 4, 5, 6, 25]  // 마지막 원소가 총합 25이다

Step 0: 마지막 원소를 0으로 두고 down-sweep을 시작한다
        [3, 4, 7, 11, 4, 5, 6, 0]

Down-sweep의 연산: 인덱스 쌍 (ai, bi)에 대해 다음을 수행한다:
  temp = arr[ai]
  arr[ai] = arr[bi]      // ai 위치가 bi의 값을 받는다
  arr[bi] = arr[bi] + temp  // bi 위치에 원래 ai의 값을 누적한다

Step 1: stride=4, 간격이 8인 원소 쌍을 처리
        인덱스 쌍: (3, 7)
        temp = 11, arr[3] = 0, arr[7] = 0 + 11 = 11
        결과: [3, 4, 7, 0, 4, 5, 6, 11]
        
        설명: 인덱스 7 앞에는 11개의 원소가 있다(인덱스 0-3의 총합)

Step 2: stride=2, 간격이 4인 원소 쌍을 처리
        인덱스 쌍: (1, 3), (5, 7)
        
        (1, 3)에 대해: temp = 4, arr[1] = 0, arr[3] = 0 + 4 = 4
        (5, 7)에 대해: temp = 5, arr[5] = 11, arr[7] = 11 + 5 = 16
        결과: [3, 0, 7, 4, 4, 11, 6, 16]
        
        설명: 
        - 인덱스 3 앞에는 4개의 원소가 있다(인덱스 0-1의 총합)
        - 인덱스 7 앞에는 16개의 원소가 있다(인덱스 0-5의 총합)

Step 3: stride=1, 간격이 2인 원소 쌍을 처리
        인덱스 쌍: (0, 1), (2, 3), (4, 5), (6, 7)
        
        (0, 1)에 대해: temp = 3, arr[0] = 0, arr[1] = 0 + 3 = 3
        (2, 3)에 대해: temp = 7, arr[2] = 4, arr[3] = 4 + 7 = 11
        (4, 5)에 대해: temp = 4, arr[4] = 11, arr[5] = 11 + 4 = 15
        (6, 7)에 대해: temp = 6, arr[6] = 16, arr[7] = 16 + 6 = 22
        
최종 결과: [0, 3, 4, 11, 11, 15, 16, 22]  // Exclusive prefix sum!

검증:
- arr[0] = 0 (앞에 원소가 없다)
- arr[1] = 3 (인덱스 0의 값)
- arr[2] = 3+1 = 4 (인덱스 0-1의 합)
- arr[3] = 3+1+7 = 11 (인덱스 0-2의 합)
- arr[4] = 3+1+7+0 = 11 (인덱스 0-3의 합)
- ...
```

핵심 이해:
- Down-sweep은 Up-sweep의 "역과정"이다
- 매 단계마다 앞에서 누적된 합을 "분배"하고 있다
- 교환과 누적을 통해 각 위치의 prefix sum을 절묘하게 계산해 낸다

시간 복잡도: O(n) 작업량, O(log n) 깊이 (병렬)

### 코드 구현 상세

```cpp
// Up-Sweep Phase: 합 트리 구성
int offset = 1;
#pragma unroll
for (int d = scan_size >> 1; d > 0; d >>= 1) {
  if (tid < d) {
    int ai = offset * (2 * tid + 1) - 1;
    int bi = offset * (2 * tid + 2) - 1;
    scan_buf[bi] += scan_buf[ai];  // 누적 합산
  }
  offset <<= 1;
  __syncthreads();
}

// 총합을 저장하고 0으로 설정
if (tid == 0) {
  prefix[num_experts] = scan_buf[scan_size - 1];
  scan_buf[scan_size - 1] = 0;
}
__syncthreads();

// Down-Sweep Phase: prefix sum 분배
#pragma unroll
for (int d = 1; d < scan_size; d <<= 1) {
  offset >>= 1;
  if (tid < d) {
    int ai = offset * (2 * tid + 1) - 1;
    int bi = offset * (2 * tid + 2) - 1;
    if (bi < scan_size) {
      int temp = scan_buf[ai];
      scan_buf[ai] = scan_buf[bi];
      scan_buf[bi] += temp;
    }
  }
  __syncthreads();
}
```

### 핵심 최적화 포인트

#### 1. 병렬 prefix sum 계산

0x2 버전 (직렬):
```cpp
if (threadIdx.x == 0) {
  cumsum[0] = 0;
  for (int i = 1; i <= num_experts; ++i) {
    cumsum[i] = cumsum[i - 1] + padded_count[i-1];
  }
}
```

0x3 버전 (병렬):
- Up-sweep: O(log n) 단계
- Down-sweep: O(log n) 단계
- 모든 스레드가 계산에 참여

성능 향상: num_experts=128에 대해 O(128)에서 O(log 128) = O(7)로 낮아진다

#### 2. expert_ids 채우기 최적화

0x2 버전:
```cpp
// 각 스레드가 expert 하나를 담당하므로 부하가 불균형하다
if (threadIdx.x < num_experts) {
  for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
    expert_ids[i / block_size] = threadIdx.x;
  }
}
```

0x3 버전 (이분 탐색 사용):
```cpp
// 모든 스레드가 모든 block을 병렬로 처리한다
const int32_t num_blocks = s_total_tokens_post_pad / block_size;
for (int32_t i = tid; i < num_blocks; i += stride) {
  int32_t block_start = i * block_size;
  // 이분 탐색으로 대응하는 expert를 찾는다
  int left = 0, right = num_experts;
  while (left < right) {
    int mid = (left + right) >> 1;
    if (prefix[mid] <= block_start) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  expert_ids[i] = left - 1;
}
```

장점:
- 모든 스레드가 참여하므로 부하가 균형을 이룬다
- 이분 탐색의 복잡도는 O(log num_experts)이다
- expert 분포가 균일하지 않은 상황에 적합하다

#### 3. shared memory 레이아웃

```cpp
extern __shared__ int32_t smem[];
int32_t* shared_counts = smem;                  // [num_experts]
int32_t* prefix = shared_counts + num_experts;  // [num_experts + 1]
int32_t* scan_buf = prefix + num_experts + 1;   // [scan_size]
```

scan_size는 반드시 2의 거듭제곱이어야 한다:
```cpp
const size_t scan_size = next_pow2(num_experts);
```

### 성능 분석

시간 복잡도를 비교해 보자:
- prefix sum 계산: 0x2는 O(n) 직렬, 0x3은 O(log n) 병렬
- expert_ids 채우기: 0x2는 O(blocks/experts)로 불균형, 0x3은 O(blocks)로 균형

shared memory 사용량:
```
shared_mem_size = (num_experts + (num_experts + 1) + scan_size) * 4 bytes
```

num_experts=128인 경우: (128 + 129 + 128) * 4 = 1540 bytes

---

## 0x4. Block/Warp Scan 알고리즘 최적화

**PR:** https://github.com/sgl-project/sglang/pull/7884

### 0x3 대비 핵심 개선점

0x3 버전은 Blelloch Scan 알고리즘으로 병렬 prefix sum을 구현했지만, Blelloch Scan은 `__syncthreads()`를 여러 번 필요로 해서 동기화 오버헤드가 비교적 크다. 0x4 버전은 **2단 Warp Scan 알고리즘**을 도입해, warp 내 shuffle 명령어를 활용하여 동기화 오버헤드를 줄였다.

### Warp Scan 알고리즘의 원리

Warp Scan은 warp 내 스레드들이 동기화 없이 바로 통신할 수 있다는 특성을(shuffle 명령어를 통해) 활용해 효율적인 prefix sum 계산을 구현한다.

#### Warp-Level Exclusive Scan

```cpp
__device__ __forceinline__ int warp_exclusive_scan(int v, unsigned mask = 0xffffffffu) {
  int original = v;
  #pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    int n = SHFL_UP(mask, v, offset);  // 앞쪽 스레드에서 값을 가져온다
    if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) v += n;
  }
  return v - original;  // exclusive scan 결과를 반환
}
```

동작 원리:
```
스레드ID:  0   1   2   3   4   5   6   7
입력:    3   1   7   0   4   1   6   3

offset=1: 각 스레드가 앞의 1번째 스레드에서 값을 가져온다
         -   3   1   7   0   4   1   6
결과:    3   4   8   7   4   5   7   9

offset=2: 각 스레드가 앞의 2번째 스레드에서 값을 가져온다
         -   -   3   4   8   7   4   5
결과:    3   4  11  11  12  12  11  14

offset=4: 각 스레드가 앞의 4번째 스레드에서 값을 가져온다
         -   -   -   -   3   4  11  11
결과:    3   4  11  11  15  16  22  25

Exclusive: 원래 값을 뺀다
결과:    0   3   4  11  11  15  16  22
```

장점:
- 동기화가 필요 없다: warp 내 스레드는 본래 동기화되어 있다
- 낮은 지연: shuffle 명령어의 지연이 매우 낮다
- 효율적: O(log 32) = 5회 반복

### 2단 Scan 구조

0x4 버전은 **2단 스캔** 전략을 사용한다:

1. 1단: 각 warp 내부에서 scan을 수행
2. 2단: warp0이 모든 warp의 합에 대해 scan을 수행
3. 병합: 각 스레드가 앞선 모든 warp의 총합을 더한다

```cpp
// 1단: Intra-warp scan
const int warp_id = tid / WARP_SIZE;
const int lane_id = tid & (WARP_SIZE - 1);
const int num_warps_for_scan = (scan_size + WARP_SIZE - 1) / WARP_SIZE;

// 각 warp 내부에서 inclusive scan을 수행
const int warp_sum = warp_exclusive_scan(padded_count) + padded_count;
if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = warp_sum;  // warp 총합을 저장
__syncthreads();

// 2단: warp0이 모든 warp의 합에 대해 scan을 수행
if (tid < WARP_SIZE) {
  int val = (tid < num_warps_for_scan) ? warp_sums[tid] : 0;
  int incl = warp_exclusive_scan(val) + val;  // inclusive scan
  warp_sums[tid] = incl;  // 누적 합을 저장
}
__syncthreads();

// block 전체의 총합을 얻는다
if (tid == 0) {
  prefix[num_experts] = warp_sums[num_warps_for_scan - 1];
  s_total_tokens_post_pad = prefix[num_experts];
  *total_tokens_post_pad = s_total_tokens_post_pad;
}
__syncthreads();
```

### 완전한 prefix sum 계산 흐름

```cpp
// 단계1: scan_buf 준비 (0x3과 동일)
if (tid < num_experts) {
  int32_t count = shared_counts[tid];
  padded_count = (count + block_size - 1) / block_size * block_size;
  scan_buf[tid] = padded_count;
}
if (tid >= num_experts && tid < scan_size) scan_buf[tid] = 0;
__syncthreads();

// 단계2: 2단 warp scan으로 exclusive prefix sum을 계산
int v = (tid < scan_size) ? scan_buf[tid] : 0;
int pre = warp_exclusive_scan(v);  // warp 내 exclusive scan
if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = pre + v;  // warp 총합을 저장
__syncthreads();

// warp0이 모든 warp 총합에 대해 scan을 수행
if (warp_id == 0) {
  int val = (lane_id < num_warps_for_scan) ? warp_sums[lane_id] : 0;
  warp_sums[lane_id] = warp_exclusive_scan(val);  // exclusive scan
}
__syncthreads();

// 단계3: 결과 병합
int offset = warp_sums[warp_id];  // 앞선 모든 warp의 총합
if (tid < scan_size) scan_buf[tid] = pre + offset;  // 최종 exclusive prefix sum
__syncthreads();

// 단계4: 결과 쓰기
if (tid < num_experts) prefix[tid] = scan_buf[tid];
if (tid <= num_experts) {
  cumsum[tid] = prefix[tid];
}
```

### 핵심 최적화 포인트 비교

#### 1. prefix sum 계산

0x3 버전 (Blelloch Scan):
```cpp
// Up-sweep: log(n)회 루프, 매번 __syncthreads()가 필요하다
for (int d = scan_size >> 1; d > 0; d >>= 1) {
  // ... 계산 ...
  __syncthreads();  // 전역 동기화
}

// Down-sweep: log(n)회 루프, 매번 __syncthreads()가 필요하다
for (int d = 1; d < scan_size; d <<= 1) {
  // ... 계산 ...
  __syncthreads();  // 전역 동기화
}
```

동기화 횟수: 2 * log(scan_size)회의 `__syncthreads()`

0x4 버전 (Warp Scan):
```cpp
// warp 내 scan: 동기화가 필요 없다
int pre = warp_exclusive_scan(v);

// 전역 동기화는 3회만 필요하다
__syncthreads();  // 1. warp_sums 쓰기 대기
// warp0 scan
__syncthreads();  // 2. warp0 완료 대기
// 결과 병합
__syncthreads();  // 3. 쓰기 완료 대기
```

동기화 횟수: 3회의 `__syncthreads()`

성능 향상: scan_size=128에 대해 2*log(128)=14회의 동기화에서 3회로 낮아진다

#### 2. shared memory 레이아웃

```cpp
extern __shared__ int32_t smem[];
int32_t* shared_counts = smem;                  // [num_experts]
int32_t* prefix = shared_counts + num_experts;  // [num_experts + 1]
int32_t* scan_buf = prefix + num_experts + 1;   // [scan_size]
int32_t* warp_sums = scan_buf + scan_size;      // [<= 32] - 새로 추가!
```

새로 추가된 warp_sums 배열:
- 각 warp의 누적 합을 저장한다
- 최대 32개 원소 (1024 threads / 32 = 32 warps)
- 추가 오버헤드: 32 * 4 = 128 bytes

#### 3. Shuffle 명령어의 장점

SHFL_UP 명령어:
```cpp
#ifndef __CUDA_ARCH__  // HIP
#define SHFL_UP(mask, val, delta) __shfl_up((val), (delta))
#else  // CUDA
#define SHFL_UP(mask, val, delta) __shfl_up_sync((mask), (val), (delta))
#endif
```

특징:
- 지연이 낮다: 보통 몇 클럭 사이클이면 된다
- 메모리 접근이 없다: 레지스터 사이에서 직접 전달된다

### 성능 분석

시간 복잡도 비교:
- prefix sum 계산: 두 버전 모두 O(log n) 병렬이다
- 동기화 횟수: 0x3은 2*log(scan_size)회가 필요하고, 0x4는 3회면 된다
- Shuffle 명령어: 0x3은 쓰지 않고, 0x4는 warp마다 O(log WARP_SIZE)회 쓴다

실제 성능 향상:
- 동기화 오버헤드 감소로 약 10%의 성능 향상
- 명령어 수준 병렬성이 더 좋다
- num_experts >= 128인 상황에 적합하다

shared memory 사용량:
```
0x3: (num_experts + (num_experts + 1) + scan_size) * 4 bytes
0x4: (num_experts + (num_experts + 1) + scan_size + 32) * 4 bytes
```

추가 오버헤드: 128 bytes (무시할 수 있다)

0x3과 0x4의 주요 차이:
- prefix sum 알고리즘: 0x3은 Blelloch Scan, 0x4는 2단 Warp Scan을 쓴다
- 동기화 횟수: 0x3은 2*log(n)회가 필요하고, 0x4는 3회면 된다
- Shuffle 명령어: 0x3은 쓰지 않고, 0x4는 대량으로 사용한다
- shared memory: 0x4가 조금 더 많다(+128B). 무시할 수 있다
- 적용 상황: 0x3은 범용적이고, 0x4는 num_experts >= 128에 더 적합하다
- 성능 향상: 0x4는 0x3 대비 약 10% 향상된다

---

## 0x5. 추가 최적화: 동적 padding과 병렬 채우기

### 최적화 배경

0x4 버전을 기반으로 커뮤니티에서 두 방향의 최적화를 추가로 제안했다:

1. **작은 batch 상황에서의 max_num_tokens_padded 계산 최적화**: 작은 batch에 대해서는 더 작은 padding 값을 사용한다
2. **sorted_token_ids 병렬 채우기**: 직렬로 실행하는 대신 여분의 스레드 자원을 활용해 병렬로 채운다

### 최적화1: max_num_tokens_padded 동적 조정

**원래 로직:**
```python
max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
if pad_sorted_ids:
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
```

**최적화 후:**
```python
max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
if pad_sorted_ids:
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
# 추가: 작은 batch에 대해서는 더 작은 padding을 사용
if topk_ids.numel() < num_experts:
    max_num_tokens_padded = topk_ids.numel() * block_size
```

**최적화 원리:**
- token 수가 매우 적을 때(expert 수보다 적을 때), 원래 공식은 메모리를 지나치게 많이 할당한다
- 새 로직은 각 token이 최대 하나의 block만 차지하도록 보장해 메모리 낭비를 피한다
- 예를 들어 token 8개, expert 256개, block_size=128인 경우
  - 원래: 8 + 256 * 127 = 32520
  - 최적화: 8 * 128 = 1024 (메모리 96.8% 절약)

### 최적화2: sorted_token_ids 병렬 채우기

#### 메인 kernel의 최적화

**0x4 버전(직렬 채우기):**
```cpp
__global__ void moe_align_block_size_kernel(...) {
  // 시작할 때 직렬로 채운다
  for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x) {
    sorted_token_ids[it] = numel;
  }
  
  // 그다음 카운팅과 prefix sum 계산을 수행
  // ...
}
```

**최적화 버전(여분의 thread block을 사용한 병렬화):**
```cpp
__global__ void moe_align_block_size_kernel(...) {
  // 별도의 thread block을 사용해 채운다
  if (blockIdx.x == 1) {
    for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x) {
      sorted_token_ids[it] = numel;
    }
    return;  // 채우기가 끝나면 바로 반환
  }
  
  // blockIdx.x == 0인 block은 기존 로직을 실행
  // 카운팅, prefix sum, expert_ids 채우기 등
  // ...
}
```

**핵심 변경 사항:**
- kernel 실행을 `<<<1, threads>>>`에서 `<<<2, threads>>>`로 바꾼다
- blockIdx.x == 1이 sorted_token_ids 채우기를 전담한다
- blockIdx.x == 0이 기존의 카운팅과 prefix sum 로직을 실행한다
- 두 block은 완전히 병렬로 실행되며 동기화가 필요 없다

**성능 향상:**
- 채우기 연산과 카운팅 연산이 완전히 병렬화된다
- 주 계산 경로의 지연이 줄어든다
- 특히 대규모 상황(max_num_tokens_padded가 매우 클 때)에 적합하다

#### 작은 batch kernel의 최적화

**0x4 버전(직렬 채우기):**
```cpp
__global__ void moe_align_block_size_small_batch_expert_kernel(...) {
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;
  
  // 모든 스레드가 먼저 채운다
  for (size_t it = tid; it < max_num_tokens_padded; it += stride) {
    sorted_token_ids[it] = numel;
  }
  
  // 그다음 카운팅, prefix sum, 정렬을 수행
  // ...
}
```

**최적화 버전(여분의 스레드 그룹을 사용):**
```cpp
template <typename scalar_t, int32_t fill_threads>
__global__ void moe_align_block_size_small_batch_expert_kernel(...) {
  // 앞쪽 fill_threads개의 스레드가 채우기를 전담한다
  if (threadIdx.x < fill_threads) {
    for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += fill_threads) {
      sorted_token_ids[it] = numel;
    }
    // 다른 스레드가 계산을 끝낼 때까지 대기(동기화 3회)
    __syncthreads();
    __syncthreads();
    __syncthreads();
    return;
  }
  
  // 나머지 스레드는 기존 로직을 실행
  const size_t tid = threadIdx.x - fill_threads;
  const size_t stride = blockDim.x - fill_threads;
  // ...
}
```

**핵심 변경 사항:**
- 템플릿 파라미터 `fill_threads`가 채우기 스레드 수를 지정한다(예: 256)
- kernel 실행을 `<<<1, threads>>>`에서 `<<<1, fill_threads + threads>>>`로 바꾼다
- 앞쪽 256개 스레드는 채우기를 전담하고, 뒤쪽 스레드는 계산을 한다
- 채우기 스레드가 계산 완료를 기다리도록 `__syncthreads()`가 3회 필요하다

**왜 동기화가 3회 필요한가:**
```cpp
// 계산 스레드의 동기화 지점:
for (int i = 0; i < num_experts; ++i) {
  tokens_cnts[(tid + 1) * num_experts + i] = 0;
}
for (size_t i = tid; i < numel; i += stride) {
  ++tokens_cnts[(tid + 1) * num_experts + topk_ids[i]];
}
__syncthreads();  // 동기화 지점1

if (tid < num_experts) {
  // prefix sum 계산
}
__syncthreads();  // 동기화 지점2

if (tid == 0) {
  // cumsum 계산
}
__syncthreads();  // 동기화 지점3

// 채우기 스레드는 이 3개의 동기화 지점을 기다려야 한다
```

### 성능 분석

두 최적화가 성능에 미치는 영향:

1. **동적 max_num_tokens_padded**
   - 메모리 절약: 작은 batch 상황에서 90% 이상의 메모리를 절약한다
   - 성능 향상: 채우기 오버헤드가 줄어 약 5-10% 향상된다

2. **병렬 채우기**
   - 메인 kernel: 채우기와 계산이 완전히 병렬화되어 지연이 20-30% 낮아진다
   - 작은 batch kernel: 채우기와 계산이 부분적으로 병렬화되어 지연이 10-15% 낮아진다

종합 성능 향상:
- 작은 batch 상황: 15-30%
- 큰 batch 상황: 10-20%

---

## 0x6. 정리

Baseline에서 0x5까지, 이 kernel의 최적화 과정은 사실 전형적인 CUDA 성능 최적화 경로 그 자체다:

1. Baseline: 기능은 정확하지만 성능은 평범하고, prefix sum이 직렬이다
2. 0x2: 벡터화 padding을 추가해 메모리 대역폭 이용률을 높였다
3. 0x3: Blelloch Scan으로 prefix sum을 병렬화해 성능이 뚜렷하게 향상됐다
4. 0x4: Warp Scan으로 동기화 오버헤드를 줄여 성능을 한층 더 최적화했다
5. 0x5: 병렬 채우기 + 동적 메모리 할당으로 전방위 최적화를 했다

num_experts >= 128인 대규모 상황에서는 0x4 버전의 Warp Scan이 확실히 유리하며 약 10%의 성능 향상을 가져온다. 0x5 버전은 여기에 더해 병렬 채우기와 동적 메모리 할당으로 추가로 10-30%의 성능 향상을 가져올 수 있다. 소규모 상황에서는 `small_batch_expert_kernel`로 모든 연산을 융합하는 편이 더 적합하다.
