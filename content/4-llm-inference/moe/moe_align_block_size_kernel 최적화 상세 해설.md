# sgl-kernel MoE Align Block Size Kernel 최적화 과정 해설

## 0x0. 머리말

이 글기록SGLang의sgl-kernel중최적화 `moe_align_kernel.cu` 의관련 내용https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/moe/moe_align_kernel.cu）。MoE모델관련 내용있다개관련 내용핵심의kernel관련 내용이다 `moe_align_block_size`,관련 내용의관련 내용사용이다tokens이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert)그룹화정렬,로후관련 내용의expert계산하다관련 내용

이kernel부터초기의baseline버전관련 내용최적화까지관련 내용에서,관련 내용개버전의관련 내용 (:)
- 0x1 Baseline: 초기의구현，작은expert（num_expert <= 64 && token <= 1024）의관련 내용필자는관련 내용사용vLLM의구현，그리고하다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (memory accesscoalesced access)의이 부분은 원문의 해당 기술 설명을 이어서 서술한다하새관련 내용개kernel，로warp로관련 내용와서관련 내용
- 0x2: 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (vectorization)의padding관련 내용
- 0x3: 사용Blelloch Scan이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (prefix sum)계산그리고row관련 내용
- 0x4: 더 나아가사용Warp Scan줄인다동기화 overhead,관련 내용이다현재성능가장 좋다의버전

**관련 내용의이다Baseline버전이다필자는완료의。그다음0x2와0x3의핵심최적화이다 https://github.com/ispobock 완료의。0x4의핵심최적화이다 https://github.com/yuan-luo 완료의。**

아래된다자세히 설명각개버전의최적화관련 내용와구현 세부 사항。

## 0x1. Baseline Kernel 상세 해설

### 이kernel까지관련 내용에서관련 내용

간단히 말하면,이kernel관련 내용하다4관련 내용 (:)
1. 집계각개expert있다많은적은개token
2. 계산정렬후의prefix sum(이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (block_size)정렬)
3. 생성한다expert_ids배열,기록각개block대응관련 내용개expert
4. tokens이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert)그룹화정렬

Baseline버전사용3개kernel와서완료이들관련 내용 (:)

#### 1. `moe_align_block_size_kernel` - 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (kernel)

```cpp
template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,      // 입력: 각개token대응의expert id
    int32_t* __restrict__ sorted_token_ids,     // 출력: 정렬후의token인덱스
    int32_t* __restrict__ expert_ids,           // 출력: 각개block대응의expert id
    int32_t* __restrict__ total_tokens_post_pad,// 출력: 정렬후의이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (token)
    int32_t num_experts,                        // expert총수
    int32_t padded_num_experts,                 // 정렬까지warp_size의expert관련 내용
    int32_t experts_per_warp,                   // 각개warp관련 내용의expert개수
    int32_t block_size,                         // 정렬의block크기
    size_t numel,                               // 입력token총수
    int32_t* __restrict__ cumsum) {             // 출력: prefix sum배열
  
  extern __shared__ int32_t shared_counts[];
  
  // 관련 내용 (1:)초기화shared memory관련 내용
  // 각개warp담당experts_per_warp개expert의관련 내용
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;
  
  // 초기화현재warp담당의expert관련 내용로0
  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      shared_counts[warp_id * experts_per_warp + i] = 0;
    }
  }
  
  __syncthreads();
  
  // 관련 내용 (2:)집계각개expert의token개수
  // 관련 내용있다thread관련 내용,관련 내용있다tokens,관련 내용사용atomic add집계
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;
  
  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];  // 얻는다현재token의expert id
    // 계산이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert)에서shared_counts중의관련 내용
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    // atomic add관련 내용,집계이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert)의token개수
    atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
  }
  
  __syncthreads();
  
  // 관련 내용 (3:)계산prefix sum(만사용thread 0실행한다)
  // prefix sum사용된다관련 내용각개expert에서출력중의관련 내용
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      int expert_count = 0;
      int warp_idx = (i - 1) / experts_per_warp;
      int expert_offset = (i - 1) % experts_per_warp;
      expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
      
      // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (block_size)정렬: CEILDIV(count, block_size) * block_size
      // 관련 내용가능로관련 내용각개expert의token관련 내용이다block_size의관련 내용
      cumsum[i] = cumsum[i - 1] + CEILDIV(expert_count, block_size) * block_size;
    }
    *total_tokens_post_pad = cumsum[num_experts];
  }
  
  __syncthreads();
  
  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (4: expert_ids)배열
  // expert_ids[i]관련 내용제i개block대응의expert관련 내용
  if (threadIdx.x < num_experts) {
    // 각개thread담당관련 내용개expert
    // 부터cumsum[threadIdx.x]까지cumsum[threadIdx.x+1]의관련 내용있다block모두이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert threadIdx.x)
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }
}
```

이kernel의관련 내용개핵심관련 내용 (:)
- 사용shared memory관련 내용각개expert의token관련 내용,줄인다global memory관련 내용
- 집계의관련 내용사용관련 내용,관련 내용많은개thread관련 내용쓰기관련 내용의문제
- prefix sum계산이다관련 내용 (row)의,만있다thread 0에서관련 내용,관련 내용이다후관련 내용최적화의관련 내용
- expert_ids관련 내용이다그리고row의,각개thread담당관련 내용개expert

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
  
  // 관련 내용있다tokens
  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    // 관련 내용사용atomic add얻는다현재token에서이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert)중의관련 내용
    // cumsum_buffer[expert_id]기록이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert)의token개수
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    // 할 것이다token인덱스i관련 내용까지대응관련 내용
    sorted_token_ids[rank_post_pad] = i;
  }
}
```

이kernel관련 내용이다tokens이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert)그룹화정렬,사용이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (thread)

#### 3. `moe_align_block_size_small_batch_expert_kernel` - 작은관련 내용최적화버전

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
  int32_t* cumsum = shared_mem;  // prefix sum배열
  int32_t* tokens_cnts = (int32_t*)(shared_mem + num_experts + 1);  // token관련 내용배열
  
  // 관련 내용 (1:)초기화각개thread의이 부분은 원문의 해당 기술 설명을 이어서 서술한다
  // tokens_cnts관련 내용 (:)[blockDim.x+1][num_experts]
  // tokens_cnts[(threadIdx.x + 1) * num_experts + i] 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (threadthreadIdx.x)대해expert i의관련 내용
  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(threadIdx.x + 1) * num_experts + i] = 0;
  }
  
  // 관련 내용 (2:)각개thread집계관련 내용담당의tokens
  for (size_t i = tid; i < numel; i += stride) {
    ++tokens_cnts[(threadIdx.x + 1) * num_experts + topk_ids[i]];
  }
  
  __syncthreads();
  
  // 관련 내용 (3:)대해각개expert,관련 내용있다thread의이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (prefix sum)
  if (threadIdx.x < num_experts) {
    tokens_cnts[threadIdx.x] = 0;
    for (int i = 1; i <= blockDim.x; ++i) {
      // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (prefix sum),tokens_cnts[i * num_experts + threadIdx.x]관련 내용
      // 전i개thread대해expert threadIdx.x의관련 내용
      tokens_cnts[i * num_experts + threadIdx.x] += 
          tokens_cnts[(i - 1) * num_experts + threadIdx.x];
    }
  }
  
  __syncthreads();
  
  // 관련 내용 (4:)계산정렬후의prefix sum
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] + 
          CEILDIV(tokens_cnts[blockDim.x * num_experts + i - 1], block_size) * block_size;
    }
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }
  
  __syncthreads();
  
  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (5: expert_ids)
  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }
  
  // 관련 내용 (6:)정렬tokens(관련 내용에서kernel중완료,관련 내용의kernel호출한다)
  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    // 계산현재token에서출력중의관련 내용
    // tokens_cnts[threadIdx.x * num_experts + expert_id]: 현재thread이전에는의thread대해이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert)의관련 내용
    // cumsum[expert_id]: 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert)에서출력중의관련 내용
    int32_t rank_post_pad = tokens_cnts[threadIdx.x * num_experts + expert_id] + 
                            cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    // 갱신관련 내용,로하관련 내용개token하다관련 내용
    ++tokens_cnts[threadIdx.x * num_experts + expert_id];
  }
}
```

이버전관련 내용작은관련 내용,도관련 내용이다 `numel < 1024 && num_experts <= 64` 의관련 내용

주요최적화관련 내용 (:)
- 집계、prefix sum、정렬관련 내용융합까지관련 내용개kernel관련 내용
- 사용thread관련 내용,이 부분은 원문의 해당 기술 설명을 이어서 서술한다의overhead
- 줄인다kernel시작관련 내용

### Baseline버전의성능관련 내용에서관련 내용

분석하와서주요있다2개문제:
1. prefix sum계산이다관련 내용 (row)의,만있다thread 0에서관련 내용,이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (thread)모두에서관련 내용,관련 내용사용그리고row관련 내용
2. 집계단계사용큰이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (atomicAdd),이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (overhead)아니작은

---

## 0x2. 관련 내용상vectorizationPadding

PR: https://github.com/sgl-project/sglang/pull/7437

### 이버전관련 내용

Baseline버전관련 내용있다개문제,관련 내용이다`sorted_token_ids`이다에서pythonlayer초기화로numbel의,관련 내용된다많은호출한다관련 내용개fill의kernel。0x2버전이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (vectorization)의padding관련 내용,에서kernel중이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (padding),줄인다fill의overhead:

```cpp
#define VEC_SIZE 4
using Vec = AlignedArray<int32_t, VEC_SIZE>;

// 에서moe_align_block_size_kernel중새관련 내용의padding코드
if (pad_sorted_token_ids) {
    int32_t fill_val = static_cast<int32_t>(numel);  // 관련 내용사용numel관련 내용로관련 내용
    int32_t total = *total_tokens_post_pad;
    
    // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (vectorization)의관련 내용
    Vec fill_vec;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      fill_vec.data[i] = fill_val;
    }
    
    // vectorization쓰기,관련 내용쓰기4개int32_t
    int32_t total_vec_count = (total + VEC_SIZE - 1) / VEC_SIZE;
    Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
    
    for (int32_t idx = tid; idx < total_vec_count; idx += stride) {
      out_ptr[idx] = fill_vec;  // 관련 내용쓰기16관련 내용
    }
  }
```

로관련 내용하다:
- vectorizationmemory access: 사용`int4`관련 내용쓰기4개int32_t,이 부분은 원문의 해당 기술 설명을 이어서 서술한다사용관련 내용향상4관련 내용
- 관련 내용그리고이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (: vectorization)쓰기가능로관련 내용그리고많은개관련 내용,줄인다latency
- 관련 내용사용`numel`,후관련 내용가능로관련 내용이다padding

`AlignedArray`이관련 내용의관련 내용사용관련 내용이다관련 내용배열이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (16)정렬,와int4관련 내용,컴파일관련 내용가능로생성한다더높은관련 내용의vectorization관련 내용

대해관련 내용개kernel모두이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (padding)지원:이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (kernel)에서계산이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert_ids)후사용vectorization이 부분은 원문의 해당 기술 설명을 이어서 서술한다개sorted_token_ids배열,후관련 내용의정렬kernel된다관련 내용있다관련 내용작은이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (kernel)에서정렬전이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (padding),그다음정렬관련 내용있다관련 내용의token관련 내용

이최적화의좋은관련 내용이다vectorization쓰기높인다이 부분은 원문의 해당 기술 설명을 이어서 서술한다사용관련 내용,관련 내용출력이 부분은 원문의 해당 기술 설명을 이어서 서술한다,padding관련 내용와관련 내용계산그리고row,이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (overhead)

---

## 0x3. 사용Blelloch Scan그리고row이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (prefix sum)

PR: https://github.com/sgl-project/sglang/pull/7794

### 관련 내용버전관련 내용문제

0x2버전prefix sum관련 내용이다관련 내용 (row)의,만있다thread 0에서관련 내용,관련 내용이다관련 내용큰의성능이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (0x3)버전이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (Blelloch Scan),prefix sum계산그리고row관련 내용

### Blelloch Scan이 부분은 원문의 해당 기술 설명을 이어서 서술한다의

Blelloch Scan이다개관련 내용의그리고rowprefix sum관련 내용,관련 내용개단계:

- 단계1: Up-Sweep (Reduce Phase)

관련 내용와관련 내용,관련 내용상계산부분와:

```
원본관련 내용 (:)[3, 1, 7, 0, 4, 1, 6, 3]
         
Step 1:   [3, 4, 7, 7, 4, 5, 6, 9]  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다
Step 2:   [3, 4, 7, 11, 4, 5, 6, 14] // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (2)
Step 3:   [3, 4, 7, 11, 4, 5, 6, 25] // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (4),관련 내용까지관련 내용와
```

- 단계2: Down-Sweep (Distribution Phase)

관련 내용하할당prefix sum。이단계의핵심 아이디어이다: 할 것이다관련 내용와할당까지관련 내용개관련 내용,계산각개관련 내용이전에는관련 내용있다관련 내용의와 。

관련 내용설명:

```
Up-sweep관련 내용후: [3, 4, 7, 11, 4, 5, 6, 25]  // 마지막으로관련 내용이다관련 내용와25

Step 0: 할 것이다마지막으로관련 내용 (0),이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (down-sweep)
        [3, 4, 7, 11, 4, 5, 6, 0]

Down-sweep의관련 내용 (:)대해관련 내용인덱스대해(ai, bi),실행한다:
  temp = arr[ai]
  arr[ai] = arr[bi]      // ai이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (bi)의관련 내용
  arr[bi] = arr[bi] + temp  // bi이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (ai)의관련 내용

Step 1: stride=4, 관련 내용로8의관련 내용대해
        인덱스대해: (3, 7)
        temp = 11, arr[3] = 0, arr[7] = 0 + 11 = 11
        결과: [3, 4, 7, 0, 4, 5, 6, 11]
        
        관련 내용 (:)인덱스7이전에는있다11개관련 내용인덱스0-3의관련 내용와)

Step 2: stride=2, 관련 내용로4의관련 내용대해
        인덱스대해: (1, 3), (5, 7)
        
        대해(1, 3): temp = 4, arr[1] = 0, arr[3] = 0 + 4 = 4
        대해(5, 7): temp = 5, arr[5] = 11, arr[7] = 11 + 5 = 16
        결과: [3, 0, 7, 4, 4, 11, 6, 16]
        
        관련 내용 (:)
        - 인덱스3이전에는있다4개관련 내용인덱스0-1의관련 내용와)
        - 인덱스7이전에는있다16개관련 내용인덱스0-5의관련 내용와)

Step 3: stride=1, 관련 내용로2의관련 내용대해
        인덱스대해: (0, 1), (2, 3), (4, 5), (6, 7)
        
        대해(0, 1): temp = 3, arr[0] = 0, arr[1] = 0 + 3 = 3
        대해(2, 3): temp = 7, arr[2] = 4, arr[3] = 4 + 7 = 11
        대해(4, 5): temp = 4, arr[4] = 11, arr[5] = 11 + 4 = 15
        대해(6, 7): temp = 6, arr[6] = 16, arr[7] = 16 + 6 = 22
        
관련 내용결과: [0, 3, 4, 11, 11, 15, 16, 22]  // Exclusive prefix sum!

검증:
- arr[0] = 0 (전관련 내용있다관련 내용
- arr[1] = 3 (인덱스0의관련 내용
- arr[2] = 3+1 = 4 (인덱스0-1의와)
- arr[3] = 3+1+7 = 11 (인덱스0-2의와)
- arr[4] = 3+1+7+0 = 11 (인덱스0-3의와)
-...
```

핵심관련 내용 (:)
- Down-sweep 이다 Up-sweep 의"관련 내용
- 각관련 내용모두에서"할당"전관련 내용의와
- 통해관련 내용와관련 내용,관련 내용계산관련 내용각개관련 내용의prefix sum

관련 내용복잡도: O(n) 관련 내용, O(log n) 관련 내용그리고row)

### 코드구현상세 해설

```cpp
// Up-Sweep Phase: 관련 내용와관련 내용
int offset = 1;
#pragma unroll
for (int d = scan_size >> 1; d > 0; d >>= 1) {
  if (tid < d) {
    int ai = offset * (2 * tid + 1) - 1;
    int bi = offset * (2 * tid + 2) - 1;
    scan_buf[bi] += scan_buf[ai];  // 관련 내용와
  }
  offset <<= 1;
  __syncthreads();
}

// 저장관련 내용와그리고관련 내용 (0)
if (tid == 0) {
  prefix[num_experts] = scan_buf[scan_size - 1];
  scan_buf[scan_size - 1] = 0;
}
__syncthreads();

// Down-Sweep Phase: 할당prefix sum
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

### 핵심최적화관련 내용

#### 1. 그리고rowprefix sum계산

0x2버전 (이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (row:)
```cpp
if (threadIdx.x == 0) {
  cumsum[0] = 0;
  for (int i = 1; i <= num_experts; ++i) {
    cumsum[i] = cumsum[i - 1] + padded_count[i-1];
  }
}
```

0x3버전 (그리고row):
- Up-sweep: O(log n) 관련 내용
- Down-sweep: O(log n) 관련 내용
- 관련 내용있다thread관련 내용와계산

성능향상: 대해이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (num_experts=128), 부터 O(128) 낮춘다까지 O(log 128) = O(7)

#### 2. expert_ids 관련 내용최적화

0x2버전:
```cpp
// 각개thread담당관련 내용개expert,관련 내용아니관련 내용
if (threadIdx.x < num_experts) {
  for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
    expert_ids[i / block_size] = threadIdx.x;
  }
}
```

0x3버전 (관련 내용사용이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (:)
```cpp
// 관련 내용있다thread그리고row관련 내용있다blocks
const int32_t num_blocks = s_total_tokens_post_pad / block_size;
for (int32_t i = tid; i < num_blocks; i += stride) {
  int32_t block_start = i * block_size;
  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다까지대응의expert
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

관련 내용 (:)
- 관련 내용있다thread관련 내용와,관련 내용
- 관련 내용복잡도 O(log num_experts)
- 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert)아니관련 내용의관련 내용

#### 3. shared memory관련 내용

```cpp
extern __shared__ int32_t smem[];
int32_t* shared_counts = smem;                  // [num_experts]
int32_t* prefix = shared_counts + num_experts;  // [num_experts + 1]
int32_t* scan_buf = prefix + num_experts + 1;   // [scan_size]
```

scan_size 반드시이다2의관련 내용 (:)
```cpp
const size_t scan_size = next_pow2(num_experts);
```

### 성능분석

대해관련 내용하관련 내용복잡도:
- prefix sum계산: 0x2이다O(n)관련 내용 (row),0x3이다O(log n)그리고row
- expert_ids이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (: 0x2)이다O(blocks/experts)아니관련 내용,0x3이다O(blocks)관련 내용

shared memoryoverhead:
```
shared_mem_size = (num_experts + (num_experts + 1) + scan_size) * 4 bytes
```

대해이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (num_experts=128: 128 + 129 + 128)* 4 = 1540 bytes

---

## 0x4. Block/Warp Scan 관련 내용최적화

**PR:** https://github.com/sgl-project/sglang/pull/7884

### 비교하면 0x3 의관련 내용

0x3버전관련 내용사용 Blelloch Scan 관련 내용구현그리고rowprefix sum,이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (Blelloch Scan)많은관련 내용`__syncthreads()`,동기화 overhead관련 내용큰。0x4버전관련 내용**이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (layer Warp Scan)**,관련 내용사용 warp 관련 내용의 shuffle 관련 내용줄인다동기화 overhead。

### Warp Scan 관련 내용

Warp Scan 관련 내용사용 warp 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (thread)가능로없음이 부분은 원문의 해당 기술 설명을 이어서 서술한다의관련 내용통해 shuffle 관련 내용,구현높은관련 내용의prefix sum계산。

#### Warp-Level Exclusive Scan

```cpp
__device__ __forceinline__ int warp_exclusive_scan(int v, unsigned mask = 0xffffffffu) {
  int original = v;
  #pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    int n = SHFL_UP(mask, v, offset);  // 부터전관련 내용의thread얻는다관련 내용
    if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) v += n;
  }
  return v - original;  // 반환한다exclusive scan결과
}
```

이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (:)
```
threadID:  0   1   2   3   4   5   6   7
입력:    3   1   7   0   4   1   6   3

offset=1: 각개thread부터전1개thread얻는다관련 내용
         -   3   1   7   0   4   1   6
결과:    3   4   8   7   4   5   7   9

offset=2: 각개thread부터전2개thread얻는다관련 내용
         -   -   3   4   8   7   4   5
결과:    3   4  11  11  12  12  11  14

offset=4: 각개thread부터전4개thread얻는다관련 내용
         -   -   -   -   3   4  11  11
결과:    3   4  11  11  15  16  22  25

Exclusive: 관련 내용가서원본관련 내용
결과:    0   3   4  11  11  15  16  22
```

관련 내용 (:)
- 없음이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (: warp thread)
- 낮은latency: shuffle 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (latency)낮은
- 높은이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (: O log 32 = 5)

### 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (layer Scan)

0x4버전관련 내용사용**이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (layer)**관련 내용 (:)

1. 제이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (layer:)각개 warp 관련 내용수행한다 scan
2. 제이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (layer: warp0)대해관련 내용있다 warp 의와수행한다 scan
3. 관련 내용그리고: 각개thread관련 내용상전관련 내용있다 warp 의관련 내용와

```cpp
// 제이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (layer: Intra-warp scan)
const int warp_id = tid / WARP_SIZE;
const int lane_id = tid & (WARP_SIZE - 1);
const int num_warps_for_scan = (scan_size + WARP_SIZE - 1) / WARP_SIZE;

// 각개warp관련 내용수행한다inclusive scan
const int warp_sum = warp_exclusive_scan(padded_count) + padded_count;
if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = warp_sum;  // 저장warp관련 내용와
__syncthreads();

// 제이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (layer: warp0)대해관련 내용있다warp의와수행한다scan
if (tid < WARP_SIZE) {
  int val = (tid < num_warps_for_scan)? warp_sums[tid]: 0;
  int incl = warp_exclusive_scan(val) + val;  // inclusive scan
  warp_sums[tid] = incl;  // 저장관련 내용와
}
__syncthreads();

// 얻는다관련 내용개block의관련 내용와
if (tid == 0) {
  prefix[num_experts] = warp_sums[num_warps_for_scan - 1];
  s_total_tokens_post_pad = prefix[num_experts];
  *total_tokens_post_pad = s_total_tokens_post_pad;
}
__syncthreads();
```

### 완전한의prefix sum계산관련 내용

```cpp
// 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (1: scan_buf)와0x3관련 내용
if (tid < num_experts) {
  int32_t count = shared_counts[tid];
  padded_count = (count + block_size - 1) / block_size * block_size;
  scan_buf[tid] = padded_count;
}
if (tid >= num_experts && tid < scan_size) scan_buf[tid] = 0;
__syncthreads();

// 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (2: layerwarp scan)계산exclusive prefix sum
int v = (tid < scan_size)? scan_buf[tid]: 0;
int pre = warp_exclusive_scan(v);  // warp이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (exclusive scan)
if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = pre + v;  // 저장warp관련 내용와
__syncthreads();

// warp0대해관련 내용있다warp관련 내용와수행한다scan
if (warp_id == 0) {
  int val = (lane_id < num_warps_for_scan)? warp_sums[lane_id]: 0;
  warp_sums[lane_id] = warp_exclusive_scan(val);  // exclusive scan
}
__syncthreads();

// 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (3:)그리고결과
int offset = warp_sums[warp_id];  // 전관련 내용있다warp의관련 내용와
if (tid < scan_size) scan_buf[tid] = pre + offset;  // 관련 내용의exclusive prefix sum
__syncthreads();

// 관련 내용 (4:)쓰기관련 내용결과
if (tid < num_experts) prefix[tid] = scan_buf[tid];
if (tid <= num_experts) {
  cumsum[tid] = prefix[tid];
}
```

### 핵심최적화관련 내용대해관련 내용

#### 1. prefix sum계산

0x3버전 (Blelloch Scan):
```cpp
// Up-sweep: log(n) 관련 내용,각관련 내용모두이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (__syncthreads)
for (int d = scan_size >> 1; d > 0; d >>= 1) {
  //... 계산...
  __syncthreads();  // 관련 내용
}

// Down-sweep: log(n) 관련 내용,각관련 내용모두이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (__syncthreads)
for (int d = 1; d < scan_size; d <<= 1) {
  //... 계산...
  __syncthreads();  // 관련 내용
}
```

이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (: 2)* log(scan_size) 관련 내용`__syncthreads()`

0x4버전 (Warp Scan):
```cpp
// Warp이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (scan:)없음관련 내용
int pre = warp_exclusive_scan(v);

// 만이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (3)
__syncthreads();  // 1. 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (warp_sums)쓰기
// warp0 scan
__syncthreads();  // 2. 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (warp0)완료
// 관련 내용그리고결과
__syncthreads();  // 3. 관련 내용쓰기완료
```

이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (: 3)`__syncthreads()`

성능향상: 대해이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (scan_size=128), 부터 2*log(128)=14 관련 내용낮춘다까지 3 관련 내용

#### 2. shared memory관련 내용

```cpp
extern __shared__ int32_t smem[];
int32_t* shared_counts = smem;                  // [num_experts]
int32_t* prefix = shared_counts + num_experts;  // [num_experts + 1]
int32_t* scan_buf = prefix + num_experts + 1;   // [scan_size]
int32_t* warp_sums = scan_buf + scan_size;      // [<= 32] - 새관련 내용!
```

새이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (warp_sums)배열:
- 관련 내용각개 warp 의관련 내용와
- 관련 내용많은 32 개이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (1024 threads / 32 = 32 warps)
- 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (overhead: 32)* 4 = 128 bytes

#### 3. Shuffle 관련 내용

SHFL_UP 관련 내용 (:)
```cpp
#ifndef __CUDA_ARCH__  // HIP
#define SHFL_UP(mask, val, delta) __shfl_up((val), (delta))
#else  // CUDA
#define SHFL_UP(mask, val, delta) __shfl_up_sync((mask), (val), (delta))
#endif
```

관련 내용 (:)
- latency낮은: 관련 내용만관련 내용개관련 내용
- 없음이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (:)에서register관련 내용

### 성능분석

관련 내용복잡도대해관련 내용 (:)
- prefix sum계산: 관련 내용개버전모두이다O(log n)그리고row
- 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (: 0x3 2)*log(scan_size)관련 내용,0x4만관련 내용 (3)
- Shuffle이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (: 0x3)아니사용,0x4각개warp사용O(log WARP_SIZE)관련 내용

관련 내용성능향상:
- 줄인다동기화 overhead관련 내용와서~10%성능향상
- 관련 내용그리고row더좋은
- 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (num_experts)>= 128의관련 내용

shared memoryoverhead:
```
0x3: (num_experts + (num_experts + 1) + scan_size) * 4 bytes
0x4: (num_experts + (num_experts + 1) + scan_size + 32) * 4 bytes
```

이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (overhead: 128 bytes)가능관련 내용

0x3와0x4의주요관련 내용 (:)
- prefix sum이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (: 0x3)사용Blelloch Scan,0x4사용이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (layerWarp Scan)
- 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (: 0x3 2)*log(n)관련 내용,0x4만관련 내용 (3)
- Shuffle이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (: 0x3)아니사용,0x4큰관련 내용사용
- shared memory: 0x4관련 내용많은이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (+128B),가능로관련 내용
- 관련 내용사용이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (: 0x3)사용,0x4더이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (num_experts)>= 128
- 성능향상: 0x4비교하면0x3있다~10%향상

---

## 0x5. 더 나아가최적화:이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (padding)와그리고row관련 내용

### 최적화배경

에서0x4버전의관련 내용상,이 부분은 원문의 해당 기술 설명을 이어서 서술한다개관련 내용의최적화:

1. **최적화작은batch관련 내용의max_num_tokens_padded계산**:대해관련 내용작은batch,관련 내용사용더작은의padding관련 내용
2. **그리고row이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (sorted_token_ids)**:관련 내용사용관련 내용의thread관련 내용그리고row관련 내용,관련 내용아니이다관련 내용 (row)실행한다

### 최적화1: 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (max_num_tokens_padded)

**원본관련 내용 (:)**
```python
max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
if pad_sorted_ids:
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
```

**최적화후:**
```python
max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
if pad_sorted_ids:
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
# 새관련 내용 (:)대해관련 내용작은batch,관련 내용사용더작은의padding
if topk_ids.numel() < num_experts:
    max_num_tokens_padded = topk_ids.numel() * block_size
```

**최적화관련 내용 (:)**
- 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (token)개수관련 내용적은관련 내용적은이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (expert)개수),원본관련 내용된다할당관련 내용많은관련 내용
- 새관련 내용각개token관련 내용많은관련 내용사용관련 내용개block,이 부분은 원문의 해당 기술 설명을 이어서 서술한다
- 관련 내용 (:8)개token,256개expert,block_size=128
  - 원본: 8 + 256 * 127 = 32520
  - 최적화: 8 * 128 = 1024 (이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (96.8%)

### 최적화2: 그리고row이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (sorted_token_ids)

#### 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (kernel)의최적화

**0x4버전(이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (row:)**
```cpp
__global__ void moe_align_block_size_kernel(...) {
  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (row)
  for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x) {
    sorted_token_ids[it] = numel;
  }
  
  // 그다음수행한다집계와prefix sum계산
  //...
}
```

**최적화버전(관련 내용사용관련 내용의thread block그리고row):**
```cpp
__global__ void moe_align_block_size_kernel(...) {
  // 관련 내용사용관련 내용의thread block와서관련 내용
  if (blockIdx.x == 1) {
    for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x) {
      sorted_token_ids[it] = numel;
    }
    return;  // 관련 내용완료후관련 내용반환한다
  }
  
  // blockIdx.x == 0 의block실행한다관련 내용있다관련 내용
  // 집계、prefix sum、expert_ids관련 내용
  //...
}
```

**핵심관련 내용 (:)**
- kernel시작부터`<<<1, threads>>>`관련 내용로`<<<2, threads>>>`
- blockIdx.x == 1관련 내용담당이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (sorted_token_ids)
- blockIdx.x == 0실행한다관련 내용있다의집계와prefix sum관련 내용
- 관련 내용개block관련 내용그리고row실행한다,없음관련 내용

**성능향상:**
- 관련 내용와집계관련 내용그리고row
- 줄인다관련 내용계산관련 내용의latency
- 관련 내용큰이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (max_num_tokens_padded)큰관련 내용

#### 작은batch kernel의최적화

**0x4버전(이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (row:)**
```cpp
__global__ void moe_align_block_size_small_batch_expert_kernel(...) {
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;
  
  // 관련 내용있다thread관련 내용
  for (size_t it = tid; it < max_num_tokens_padded; it += stride) {
    sorted_token_ids[it] = numel;
  }
  
  // 그다음집계、prefix sum、정렬
  //...
}
```

**최적화버전(관련 내용사용관련 내용의thread관련 내용 (:)**
```cpp
template <typename scalar_t, int32_t fill_threads>
__global__ void moe_align_block_size_small_batch_expert_kernel(...) {
  // 전fill_threads개thread관련 내용담당관련 내용
  if (threadIdx.x < fill_threads) {
    for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += fill_threads) {
      sorted_token_ids[it] = numel;
    }
    // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (thread)완료계산(3관련 내용
    __syncthreads();
    __syncthreads();
    __syncthreads();
    return;
  }
  
  // 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (thread)실행한다관련 내용있다관련 내용
  const size_t tid = threadIdx.x - fill_threads;
  const size_t stride = blockDim.x - fill_threads;
  //...
}
```

**핵심관련 내용 (:)**
- 관련 내용파라미터`fill_threads`이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (thread 256)
- kernel시작부터`<<<1, threads>>>`관련 내용로`<<<1, fill_threads + threads>>>`
- 전256개thread관련 내용,후관련 내용의thread하다계산
- 관련 내용 (3)`__syncthreads()`이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (thread)계산완료

**로이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (3:)**
```cpp
// 계산thread의관련 내용 (:)
for (int i = 0; i < num_experts; ++i) {
  tokens_cnts[(tid + 1) * num_experts + i] = 0;
}
for (size_t i = tid; i < numel; i += stride) {
  ++tokens_cnts[(tid + 1) * num_experts + topk_ids[i]];
}
__syncthreads();  // 관련 내용 (1)

if (tid < num_experts) {
  // 계산prefix sum
}
__syncthreads();  // 관련 내용 (2)

if (tid == 0) {
  // 계산cumsum
}
__syncthreads();  // 관련 내용 (3)

// 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (thread 3)개관련 내용
```

### 성능분석

관련 내용개최적화의성능관련 내용 (:)

1. **이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (max_num_tokens_padded)**
   - 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (:)작은batch관련 내용하이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (90%+)
   - 성능향상:줄인다이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (overhead),이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (5-10%)향상

2. **그리고row관련 내용**
   - 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (kernel:)와계산관련 내용그리고row,latency낮춘다20-30%
   - 작은batch kernel:관련 내용와계산부분그리고row,latency낮춘다10-15%

관련 내용성능향상:
- 작은batch이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (:15-30%)
- 큰batch이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (:10-20%)

---

## 0x6. 정리

부터Baseline까지0x5,이kernel의최적화이 부분은 원문의 해당 기술 설명을 이어서 서술한다이다관련 내용의CUDA성능최적화관련 내용 (:)

1. Baseline: 관련 내용가능관련 내용성능관련 내용,prefix sum이다관련 내용 (row)의
2. 0x2: 이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (vectorizationpadding),향상이 부분은 원문의 해당 기술 설명을 이어서 서술한다사용관련 내용
3. 0x3: 사용Blelloch Scan그리고row이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (prefix sum),성능향상관련 내용
4. 0x4: 사용Warp Scan줄인다동기화 overhead,더 나아가최적화성능
5. 0x5: 그리고row이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (+)할당,관련 내용최적화

대해이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (num_experts)>= 128의큰관련 내용,0x4버전의Warp Scan관련 내용,가능관련 내용와서~10%의성능향상。0x5버전에서관련 내용상,통해그리고row관련 내용와관련 내용할당,가능관련 내용와서이 부분은 원문의 해당 기술 설명을 이어서 서술한다 (10-30%)의성능향상。대해관련 내용작은관련 내용,사용`small_batch_expert_kernel`융합관련 내용있다관련 내용더관련 내용
