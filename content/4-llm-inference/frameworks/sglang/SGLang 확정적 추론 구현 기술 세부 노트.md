# SGLang 확정적 추론 구현 기술 세부 노트

## 0x0. 서론

최근 SGLang의 확정적 추론 기능(Deterministic Inference) 지원(https://lmsys.org/blog/2025-09-22-sglang-deterministic/) 을 계기로 이 주제를 공부했다. 이것은 Thinking Machines가 최근 제안한, SGLang에서 대규모 언어 모델 추론의 비결정성을 극복하는 엔지니어링 구현이다. Thinking Machines의 원문을 보려면 https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ 또는 그 한국어 버전: https://zhuanlan.zhihu.com/p/1949285893413278978 을 참고한다.

확정적 추론의 핵심 과제는 딥러닝 추론 과정에 여러 가지 무작위성 원천이 존재한다는 점이다. attention 메커니즘에서의 난수 생성, 샘플링 과정의 무작위성, 서로 다른 batch size에서의 계산 순서 차이 등이 그 예다. SGLang은 batch invariant ops 도입, 난수 시드 고정, 특수한 attention backend 설정 등의 기술 수단을 통해 확정적 추론을 성공적으로 구현했다.

이 글은 SGLang이 확정적 추론 흐름을 구현하는 관련 기술 세부 사항을 기록한 것이다.

## 0x1. SGLang 확정적 추론 활성화

SGLang은 `--enable-deterministic-inference` 파라미터로 확정적 추론 모드를 활성화한다:

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend flashinfer \
    --enable-deterministic-inference
```

`server_args.py`에서 이 파라미터가 정의된다:

```python
# 출처: sglang/python/sglang/srt/server_args.py
parser.add_argument(
    "--enable-deterministic-inference",
    action="store_true",
    help="Enable deterministic inference mode with batch invariant ops.",
)
```

전체 확정적 추론은 주로 몇 가지 부분으로 구성된다: Batch Invariant Ops(서로 다른 batch size에서 결과 일관성 보장), 확정적 샘플링(고정 시드로 재현 가능한 난수 시퀀스 생성), FlashInfer와 Triton Attention Backend의 특수 설정, 그리고 각 모듈의 동작을 제어하는 여러 환경 변수.

## 0x2. 배치 불변성 원리와 Batch Invariant Ops

### 0x2.1 배치 불변성 소개

배치 불변성(Batch Invariance)이란 모델이 서로 다른 배치 크기의 입력을 처리할 때, 동일한 입력 데이터에 대해 완전히 동일한 출력 결과를 생성하는 것을 말한다. 간단한 시나리오를 생각해보자: 입력 샘플 x가 있고, batch_size=1과 batch_size=4(4개 샘플 모두 x인 경우)로 각각 추론했을 때, 이론적으로 첫 번째 출력은 완전히 동일해야 한다. 그러나 실제로는 종종 그렇지 않다:

```python
import torch

# 단일 샘플 추론
x = torch.randn(1, 128, device='cuda')
output_single = model(x)

# 배치 추론 (동일 샘플 4개)
x_batch = x.repeat(4, 1)
output_batch = model(x_batch)

# 첫 번째 출력이 동일한지 확인
print(torch.allclose(output_single, output_batch[0:1]))  # 종종 False!
```

이런 불일치의 근본 원인은 GPU의 병렬 계산 특성에 있다. 서로 다른 batch size는 서로 다른 병렬화 전략, 메모리 접근 패턴, 수치 계산 순서를 유발할 수 있어 부동소수점 연산의 누적 오차가 달라진다.

### 0x2.2 배치 불변성 연산자 구현

https://link.zhihu.com/?target=https%3A//thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ 에 따르면, 배치 불변성 연산자를 구현할 때 몇 가지 사항을 고려해야 한다:

- 행렬 곱셈의 병렬화 차이: GPU에서의 행렬 곱셈은 입력 크기에 따라 서로 다른 tile 분해와 thread block 스케줄링 전략을 선택한다. batch_size=1일 때는 어떤 tile 크기를 사용하고, batch_size=64일 때는 다른 크기를 사용할 수 있어 부동소수점 연산 순서가 달라진다.
- 리덕션 연산의 순서 민감성: sum, mean 같은 리덕션 연산은 GPU에서 병렬로 실행되며, 병렬도가 다르면 누적 순서가 달라진다. 부동소수점 덧셈은 결합 법칙을 만족하지 않으므로 결과에 차이가 생긴다.
- 메모리 접근 패턴의 영향: 서로 다른 batch size에서 메모리 레이아웃과 접근 패턴이 다르면, 다른 캐시 동작과 메모리 병합 패턴을 유발해 계산 결과에 간접적인 영향을 준다.
- 연산자 구현의 형태 의존성: 일부 연산자 구현은 입력 형태에 따라 서로 다른 코드 경로를 선택한다. 예를 들어 작은 batch에는 한 가지 구현을, 큰 batch에는 다른 구현을 사용한다.

### 0x2.3 SGLang의 Batch Invariant Ops 구현

SGLang은 먼저 `thinking-machines-lab/batch_invariant_ops` 프로젝트의 Batch Invariant 연산자를 도입해 이 문제들을 해결했다. 그런 다음 Attention Backend와 Sampler 측면에서도 확정적 kernel 지원을 추가해 전체 추론 흐름의 확정성을 달성했다.

#### Matmul Persistent Kernel 설계 사상

전통적인 GPU kernel은 "하나의 thread block이 하나의 출력 tile을 처리"하는 패턴을 사용한다. 이 패턴에서는 입력 크기가 달라지면 서로 다른 수의 thread block이 실행되어 스케줄링 순서의 비결정성이 생긴다.

Persistent kernel은 "고정된 수의 thread block이 각각 여러 tile을 처리"하는 패턴을 채택한다:

```python
# 전통적인 kernel 실행 방식
def traditional_launch(M, N, BLOCK_M, BLOCK_N):
    num_blocks_m = (M + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (N + BLOCK_N - 1) // BLOCK_N
    return (num_blocks_m, num_blocks_n)  # 블록 수가 입력 크기에 따라 변함

# Persistent kernel 실행 방식  
def persistent_launch(M, N, BLOCK_M, BLOCK_N, NUM_SMS):
    total_tiles = ((M + BLOCK_M - 1) // BLOCK_M) * ((N + BLOCK_N - 1) // BLOCK_N)
    return (min(NUM_SMS, total_tiles),)  # 고정적으로 NUM_SMS개 블록 사용
```

핵심 아이디어는 각 SM이 처리하는 작업량을 고정하고, 하드웨어 스케줄링 대신 소프트웨어 스케줄링을 사용해 계산 순서를 확정하는 것이다.

#### 확정적 Tile 스케줄링 알고리즘

Persistent kernel 내부에서는 각 thread block이 어떤 tile을 처리할지 결정하는 확정적 알고리즘도 필요하다. SGLang은 tile ID 기반의 확정적 매핑을 사용한다:

```python
@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    # 현재 tile이 어떤 group에 속하는지 계산
    group_id = tile_id // num_pid_in_group
    
    # 해당 group의 M 차원 시작 위치 계산
    first_pid_m = group_id * GROUP_SIZE_M
    
    # 해당 group의 실제 크기 계산 (경계 처리)
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    # group 내에서의 확정적 매핑
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    
    return pid_m, pid_n
```

이 알고리즘은 동일한 tile_id가 총 tile 수에 상관없이 항상 동일한 (pid_m, pid_n)에 매핑됨을 보장한다.

#### 기타 확정적 kernel 연산자 구현

Matmul 연산자 스케줄링이 확정적이어야 할 뿐만 아니라, 연산자 자체도 Batch Invariant해야 한다. RMSNorm을 예로 들면:

```python
@triton.jit
def rms_norm_kernel(input_ptr, weight_ptr, output_ptr, eps, n_cols, BLOCK_SIZE: tl.constexpr):
    # 각 행을 독립적으로 처리해 batch 간 상호 영향 방지
    row_idx = tl.program_id(0)
    
    # 고정 정밀도의 계산 순서 사용
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # 데이터 로드 후 float32로 변환해 정밀도 확보
    x = tl.load(input_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0)
    x = x.to(tl.float32)
    
    # 고정된 수치 안정 계산
    x_squared = x * x
    mean_x_squared = tl.sum(x_squared, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(mean_x_squared + eps)
    
    # 정규화 및 스케일링
    normalized = x * rstd
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    output = normalized * weight
    
    # 결과 저장
    tl.store(output_ptr + row_idx * n_cols + col_offsets, output, mask=mask)
```

핵심은 고정된 계산 순서와 정밀도를 사용해 서로 다른 병렬도로 인한 수치 차이를 방지하는 것이다.

#### Batch Invariant Ops 도입 및 활성화

SGLang은 `thinking-machines-lab/batch_invariant_ops` 프로젝트의 연산자를 그대로 사용한다. `model_runner.py`에서 확정적 추론을 활성화할 때 batch invariant 모드를 임포트하고 활성화한다:

```python
# 출처: sglang/python/sglang/srt/model_executor/model_runner.py
if server_args.enable_deterministic_inference:
    from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode
    enable_batch_invariant_mode()
```

#### Batch Invariant Ops 연산자 구현 해설 (보충)

##### MatMul Persistent Kernel 해설

먼저 PID(Program ID) 계산 함수는 tile의 스케줄링 순서가 확정적임을 보장한다:

```python
# 출처: sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    """
    확정적인 Program ID를 계산해 tile 스케줄링 순서를 고정

    Args:
        tile_id: 현재 tile의 전역 ID
        num_pid_in_group: 각 group의 PID 수
        num_pid_m: M 차원의 PID 수
        GROUP_SIZE_M: M 차원의 group 크기
        NUM_SMS: SM 수
    """
    # 현재 tile이 어떤 group에 속하는지 계산
    group_id = tile_id // num_pid_in_group
    
    # 해당 group의 M 차원 시작 위치 계산
    first_pid_m = group_id * GROUP_SIZE_M
    
    # 해당 group의 실제 크기 계산 (경계 처리)
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    # M 차원의 PID 계산
    pid_m = first_pid_m + (tile_id % group_size_m)
    
    # N 차원의 PID 계산
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    
    return pid_m, pid_n
```

이 함수의 핵심은 확정적 그룹화에 있다. `group_id = tile_id // num_pid_in_group`으로 tile 그룹화가 확정적임을 보장한다. M과 N 차원의 PID 계산은 모두 tile_id를 기반으로 해, 동일한 tile_id가 항상 동일한 (pid_m, pid_n)에 매핑됨을 보장한다. 경계 처리는 `min(num_pid_m - first_pid_m, GROUP_SIZE_M)`을 사용해 경계 group 처리의 일관성을 보장한다.

```python
# 출처: sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr, b_ptr, c_ptr, bias_ptr,  # 입출력 포인터
    M, N, K,  # 행렬 차원
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,  # 스트라이드
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr,
    A_LARGE: tl.constexpr, B_LARGE: tl.constexpr, C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # 현재 SM의 시작 PID 가져오기
    start_pid = tl.program_id(axis=0)
    
    # tile 수와 분포 계산
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)  # M 차원의 tile 수
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  # N 차원의 tile 수
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)    # K 차원의 tile 수
    num_tiles = num_pid_m * num_pid_n     # 총 tile 수
    
    # 출력 단계의 tile ID 계산용
    tile_id_c = start_pid - NUM_SMS
    
    # 자주 사용하는 오프셋 미리 계산
    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    # Persistent 루프: 각 SM이 여러 tile을 처리
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        # 현재 tile의 PID 계산
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        
        # 현재 tile의 M, N 차원 시작 위치 계산
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        
        # A, B 행렬의 오프셋 계산
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        
        # 큰 행렬의 인덱스 타입 변환 처리
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        
        # 경계 검사 및 마스크 처리
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        
        # 메모리 접근 최적화: 연속 접근 및 정렬 보장
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        
        # 누산기 초기화
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        # K 차원 루프: 행렬 곱셈 실행
        for ki in range(k_tiles):
            # K 차원 오프셋 계산
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            
            # A, B 메모리 주소 계산
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            
            # 데이터 로드 및 경계 처리
            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            
            # 행렬 곱셈 누적 실행
            accumulator = tl.dot(a, b, accumulator)
        
        # 출력 단계: 출력 위치 계산
        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        
        # 출력 행렬 오프셋 계산
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        
        # 큰 행렬의 출력 인덱스 처리
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        
        # 출력 주소 및 마스크 계산
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        
        # bias 처리 (존재하는 경우)
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
        
        # 타입 변환 및 저장
        if c_ptr.dtype.element_ty == tl.float8e4nv:
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)
        
        # 결과 저장
        tl.store(c_ptrs, c, mask=c_mask)
```

##### Persistent MatMul Kernel 설정 및 실행

```python
# 출처: sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
def matmul_persistent(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None):
    # 입력 검증
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert bias is None or bias.dim() == 1, "Currently assuming bias is 1D"
    
    # 하드웨어 정보 가져오기
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    
    # 출력 텐서 할당
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    
    # grid 함수: 실행할 block 수가 SM 수를 넘지 않도록 보장
    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )
    
    # 데이터 타입별 최적화 설정
    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8,
        },
    }
    
    # kernel 실행
    matmul_kernel_persistent[grid](
        a, b, c, bias, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        NUM_SMS=NUM_SMS,
        A_LARGE=a.numel() > 2**31, B_LARGE=b.numel() > 2**31, C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **configs[dtype],
    )
    return c
```

##### 확정적 Log Softmax 연산자

Log Softmax 연산자는 수치적으로 안정적이고 확정적인 softmax 계산을 구현한다:

```python
# 출처: sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
@triton.jit
def _log_softmax_kernel(
    input_ptr, output_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr,
):
    """
    2D 텐서의 마지막 차원에 대해 log_softmax를 계산
    각 block이 입력 텐서의 한 행을 처리
    """
    # 현재 block이 처리할 행 인덱스 가져오기
    row_idx = tl.program_id(0).to(tl.int64)
    
    # 입력 및 출력 행의 기본 주소 계산
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    
    # 1단계: 행의 최댓값 찾기 (수치 안정성)
    max_val = -float("inf")
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        
        # 값 로드
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))
        
        # 최댓값 업데이트
        max_val = tl.max(tl.maximum(vals, max_val))
    
    # 2단계: exp(x - max_val)의 합 계산
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        
        # 값 로드
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        
        # exp(x - max_val) 계산 및 누적
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))
    
    # log(sum_exp) 계산
    log_sum_exp = tl.log(sum_exp)
    
    # 3단계: 최종 log_softmax 값 계산: x - max_val - log_sum_exp
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        
        # 값 로드
        vals = tl.load(row_start_ptr + col_idx, mask=mask)
        
        # log_softmax 계산
        output = vals - max_val - log_sum_exp
        
        # 결과 저장
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)
```

##### 확정적 Mean 연산자

Mean 연산자는 지정된 차원을 따라 평균값을 계산한다:

```python
# 출처: sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
@triton.jit
def mean_kernel(
    input_ptr, output_ptr,
    input_stride0, input_stride1, input_stride2,
    output_stride0, output_stride1,
    M, N, K,  # M: 리덕션 차원 이전 크기, N: 리덕션 차원 크기, K: 리덕션 차원 이후 크기
    BLOCK_SIZE: tl.constexpr,
):
    """
    단일 차원을 따라 평균을 계산하는 kernel
    입력은 (M, N, K)로 보이며, N이 리덕션 차원
    """
    # Program ID가 계산할 출력 원소를 결정
    pid = tl.program_id(0)
    
    # 출력 인덱스 계산
    m_idx = pid // K
    k_idx = pid % K
    
    # 경계 검사
    if m_idx >= M or k_idx >= K:
        return
    
    # 리덕션 차원을 따라 누적 합산
    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N
        
        # 입력 인덱스 계산
        input_idx = (
            m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2
        )
        
        # 로드 및 누적
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)
    
    # 평균 계산 및 저장
    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)
```

##### 연산자 등록 및 교체 메커니즘

SGLang은 Batch Invariant 모드를 활성화할 때 PyTorch의 원래 연산자를 batch invariant 버전으로 교체한다:

```python
# 출처: sglang/python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
def enable_batch_invariant_mode():
    """batch invariant 모드 활성화, PyTorch 원래 연산자 교체"""
    global _batch_invariant_MODE, _batch_invariant_LIB
    if _batch_invariant_MODE:
        return
    
    _batch_invariant_MODE = True
    
    # PyTorch 라이브러리 구현 생성
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")
    
    # 연산자 교체 등록
    _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "CUDA")
    _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, "CUDA")
    _batch_invariant_LIB.impl("aten::_log_softmax", _log_softmax_batch_invariant, "CUDA")
    _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, "CUDA")

# 래퍼 함수
def mm_batch_invariant(a, b):
    """행렬 곱셈의 batch invariant 구현"""
    return matmul_persistent(a, b)

def addmm_batch_invariant(bias, a, b):
    """bias를 포함한 행렬 곱셈의 batch invariant 구현"""
    return matmul_persistent(a, b, bias=bias)

def _log_softmax_batch_invariant(input, dim, _half_to_float):
    """Log softmax의 batch invariant 구현"""
    assert not _half_to_float, "not implemented"
    return log_softmax(input, dim=dim)

def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype | None = None):
    """Mean의 batch invariant 구현"""
    assert dtype is None or dtype == torch.float32, f"unsupported dtype: {dtype}"
    if len(dim) == 1:
        return mean_dim(input, dim[0], keepdim=keepdim)
    else:
        # 다차원 리덕션의 fallback 구현
        assert input.dtype in {torch.float16, torch.bfloat16, torch.float32}
        n_elems = 1
        for d in dim:
            n_elems *= input.shape[d]
        return torch.sum(input, dim=dim, keepdim=keepdim, dtype=torch.float32) / n_elems
```

## 0x3. 확정적 Sampling 메커니즘

### 0x3.1 확정적 Sampling의 원리

#### 일반 샘플링이 비결정적인 이유

텍스트 생성에서 샘플링 과정은 추론 결과의 비결정성의 주요 원인이다. 전통적인 샘플링에는 다음과 같은 문제가 있다:

```python
# 전통적인 샘플링의 비결정성 예시
import torch

def traditional_sampling_demo():
    """전통적인 샘플링의 비결정성 시연"""
    logits = torch.tensor([2.0, 1.0, 0.5, 3.0])  # 고정 logits
    
    # 여러 번 샘플링하면 서로 다른 결과가 나옴
    results = []
    for i in range(5):
        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        results.append(sampled.item())
    
    print(f"Traditional sampling results: {results}")
    # 출력 예: [3, 0, 3, 2, 3] - 매번 다름
    
traditional_sampling_demo()
```

비결정성의 주요 원인: 시스템 난수 생성기가 시스템 시간이나 하드웨어 노이즈에 의존한다. GPU 병렬 실행 시 스레드 실행 순서가 난수 시퀀스에 영향을 준다. 메모리 접근 패턴이 난수 생성기 상태에 영향을 준다.

#### 확정적 샘플링의 원리

SGLang 확정적 샘플링의 핵심 아이디어: 각 token 위치에 고유한 확정적 시드를 할당하고, Gumbel-Max 샘플링 방법(수학적으로 등가이지만 확정적)을 사용하며, 해시 함수로 시드를 고품질 의사 난수로 매핑한다. 구체적으로:

##### Gumbel-Max 샘플링 원리

Gumbel-Max 샘플링은 확정적 샘플링의 수학적 기반이다:

```python
# Gumbel-Max 샘플링 원리 시연
import torch
import math

def gumbel_max_principle():
    """Gumbel-Max 샘플링의 수학적 원리 시연"""
    
    # 원래 확률 분포
    logits = torch.tensor([2.0, 1.0, 0.5, 3.0])
    probs = torch.softmax(logits, dim=-1)
    print(f"Original probabilities: {probs}")
    
    # Gumbel-Max 샘플링 과정
    # 1. Gumbel 노이즈 생성
    uniform = torch.tensor([0.3, 0.7, 0.1, 0.9])  # 확정적인 "랜덤" 수
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-9) + 1e-9)
    print(f"Gumbel noise: {gumbel_noise}")
    
    # 2. log 확률에 더하기
    perturbed_logits = logits + gumbel_noise
    print(f"Perturbed logits: {perturbed_logits}")
    
    # 3. 최댓값에 해당하는 인덱스 선택
    sampled_idx = torch.argmax(perturbed_logits)
    print(f"Sampled index: {sampled_idx.item()}")
    
    # 검증: 동일한 uniform 값을 여러 번 사용하면 동일한 결과
    for i in range(3):
        same_gumbel = -torch.log(-torch.log(uniform + 1e-9) + 1e-9)
        same_result = torch.argmax(logits + same_gumbel)
        print(f"Repeat {i}: {same_result.item()}")

gumbel_max_principle()
```

##### 위치 의존 시드 생성

SGLang은 위치 정보를 사용해 고유한 시드를 생성한다:

```python
# 위치 의존 시드 생성 예시
def position_dependent_seeding():
    """위치 의존 시드 생성 시연"""
    
    base_seed = 42
    sequence_positions = [0, 1, 2, 3, 4]  # token 위치
    
    # SGLang의 시드 생성 알고리즘
    for pos in sequence_positions:
        # 큰 소수로 해시해 서로 다른 위치에 서로 다른 시드 보장
        step_seed = base_seed * 19349663 ^ pos * 73856093
        
        # 추가 해시로 최종 시드 생성
        final_seed = step_seed * 8589934591 % (2**32)
        
        print(f"Position {pos}: base_seed={base_seed} -> step_seed={step_seed} -> final_seed={final_seed}")
    
    # 검증: 동일한 위치는 항상 동일한 시드 생성
    print("\n일관성 검증:")
    for _ in range(3):
        pos = 2
        step_seed = base_seed * 19349663 ^ pos * 73856093
        final_seed = step_seed * 8589934591 % (2**32)
        print(f"Position {pos} (repeat): {final_seed}")

position_dependent_seeding()
```

### 0x3.2 샘플링 시드의 전달 및 구현

PR #10687(https://github.com/sgl-project/sglang/pull/10687) 에 따르면, SGLang은 요청 구조에 `sampling_seed` 필드를 추가해 사용자가 각 요청에 확정적 난수 시드를 지정할 수 있도록 했다:

```python
# 출처: sglang/python/sglang/srt/sampling/sampling_params.py
class SamplingParams:
    def __init__(
        self,
        # ... 기타 파라미터
        sampling_seed: int = 42,  # 새로 추가된 샘플링 시드 파라미터
    ) -> None:
        # ... 기타 파라미터 할당
        self.sampling_seed = sampling_seed
```


`SamplingBatchInfo`에서 샘플링 시드는 확정적 추론이 활성화된 경우에만 처리된다:

```python
# 출처: sglang/python/sglang/srt/sampling/sampling_batch_info.py
def __init__(self, reqs, vocab_size, device):
    # ... 기타 초기화 코드
    
    # 확정적 추론 활성화 여부 확인
    enable_deterministic = global_server_args_dict["enable_deterministic_inference"]
    
    # 확정적 모드에서만 샘플링 시드 텐서 생성
    sampling_seed = (
        torch.tensor(
            [r.sampling_params.sampling_seed for r in reqs],
            dtype=torch.int32,
            device=device,
        )
        if enable_deterministic
        else None
    )
    
    # ... 기타 필드 초기화
    self.sampling_seed = sampling_seed
```

이 설계는 샘플링 시드가 필요할 때만 처리되어 불필요한 메모리 오버헤드를 방지한다.

### 0x3.3 확정적 다항 샘플링 구현

PR #10678(https://github.com/sgl-project/sglang/pull/10678) 에 따르면, SGLang은 온도가 0보다 큰 경우의 확정적 샘플링을 구현했다. 핵심 확정적 샘플링 함수 `multinomial_with_seed`는 시드 기반 확정적 샘플링을 구현한다:

```python
# 출처: sglang/python/sglang/srt/layers/sampler.py
def multinomial_with_seed(
    inputs: torch.Tensor, seed: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """
    고유한 난수 시드로 입력 텐서에 대해 확정적 샘플링 수행

    Args:
        inputs: 형태가 (n, m)인 부동소수점 텐서, n개의 범주 분포를 나타냄
        seed: 형태가 (n,)인 정수 텐서, 각 행에 해당하는 난수 시드 포함
        positions: 시퀀스 내 token 위치, 고유 시드 생성에 사용

    Returns:
        형태가 (n,)인 텐서, 각 원소는 해당 분포에서 샘플링된 인덱스
    """
    n, m = inputs.shape
    col_indices = torch.arange(m, device=inputs.device).unsqueeze(0)
    
    # 위치 의존 시드 생성, 각 위치가 고유한 무작위성 가지도록
    step_seed = seed * 19349663 ^ positions * 73856093
    seed_expanded = step_seed.unsqueeze(-1)
    
    # 해시 함수로 의사 난수 생성
    hashed = seed_expanded * 8589934591 ^ col_indices * 479001599
    uniform_samples = (hashed % (2**24)).float() / (2**24)
    
    # Gumbel-Max 기법으로 샘플링
    epsilon = 1e-9
    gumbel_noise = -torch.log(-torch.log(uniform_samples + epsilon) + epsilon)
    log_probs = torch.log(inputs + epsilon)
    perturbed_log_probs = log_probs + gumbel_noise
    
    return torch.argmax(perturbed_log_probs, dim=1, keepdim=True)
```

이 구현의 핵심 기술 포인트:

1. **위치 의존 시드 생성**: `step_seed = seed * 19349663 ^ positions * 73856093`으로 각 token 위치가 고유한 무작위성을 가지도록 보장
2. **해시 함수**: 큰 소수로 해시해 고품질 의사 난수 생성
3. **Gumbel-Max 샘플링**: Gumbel 노이즈를 통해 확정적인 다항 샘플링 구현

### 0x3.4 Sampler 상위 인터페이스의 변경

`Sampler` 클래스에서 샘플링 시드가 감지되면 확정적 샘플링을 사용한다. 실제 코드 구현에 따르면 확정적 샘플링의 호출 위치는 `sample_token` 함수에 있다:

```python
# 출처: sglang/python/sglang/srt/layers/sampler.py
def sample_token(
    probs: torch.Tensor,
    sampling_seed: Optional[torch.Tensor],
    positions: torch.Tensor,
):
    """
    Token 샘플링 함수

    sampling_seed가 None이 아닐 때 확정적 추론을 활성화하여
    각 요청의 샘플링 시드로 샘플링한다.
    이는 PR #10678에서 구현된 핵심 기능으로, 온도 > 0에서의 확정적 샘플링을 지원한다.
    """
    if sampling_seed is not None:
        # 확정적 샘플링 사용
        sampled_index = multinomial_with_seed(probs, sampling_seed, positions)
    else:
        # 전통적인 랜덤 샘플링 사용
        sampled_index = sampling_from_probs_torch(probs)
    
    return sampled_index
```

#### Sampler의 forward 메서드 통합

`Sampler`의 `forward` 메서드에서 확정적 샘플링이 전체 샘플링 흐름에 통합된다:

```python
# 출처: sglang/python/sglang/srt/layers/sampler.py
def forward(self, logits_output, sampling_info, return_logprob, top_logprobs_nums, 
            token_ids_logprobs, positions):
    # ... logits 전처리
    
    if not sampling_info.is_all_greedy:
        # ... 온도, top_p 등 파라미터 처리
        
        # 핵심: 확정적 샘플링 호출
        batch_next_token_ids = sample_token(
            probs_sort,
            sampling_info.sampling_seed,  # 샘플링 시드 전달
            positions,
        )
    
    # ... 후처리
```

이 설계는 확정적 샘플링이 SGLang의 샘플링 흐름에 원활하게 통합되면서 하위 호환성을 유지하도록 한다.

## 0x4. Attention Backend의 확정적 설정

먼저 https://zhuanlan.zhihu.com/p/1949285893413278978 에서 배치 불변 Attention의 원리 소개를 읽어볼 것을 권장한다.

### 0x4.1 FlashInfer Attention Backend의 확정적 지원

FlashInfer Attention Backend의 확정적 지원은 PR #10645(https://github.com/sgl-project/sglang/pull/10645) 와 FlashInfer 프로젝트의 PR #1675(https://github.com/flashinfer-ai/flashinfer/pull/1675) 에 기반한다. 핵심 구현은 고정 split tile size를 통해 batch invariant한 attention 계산을 보장한다:

```python
# 출처: sglang/python/sglang/srt/layers/attention/flashinfer_backend.py
def __init__(self, model_runner):
    # ... 기타 초기화 코드
    
    # 확정적 추론 설정
    # 확정적 추론 활성화 시 decode 단계에서 tensor cores를 사용해야 함
    # 동시에 환경 변수로 prefill과 decode의 split tile size를 설정하고,
    # cuda graph의 kv split를 비활성화
    # 추가 정보: https://github.com/flashinfer-ai/flashinfer/pull/1675
    self.enable_deterministic = (
        model_runner.server_args.enable_deterministic_inference
    )
    self.prefill_split_tile_size = None
    self.decode_split_tile_size = None
    self.disable_cuda_graph_kv_split = False
    
    if self.enable_deterministic:
        # decode 단계에서 tensor cores 강제 사용
        self.decode_use_tensor_cores = True
        
        # 확정성 보장을 위해 고정 split tile size 설정
        self.prefill_split_tile_size = get_int_env_var(
            "SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096
        )
        self.decode_split_tile_size = get_int_env_var(
            "SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE", 2048
        )
        
        # 확정성 보장을 위해 CUDA Graph의 KV split 비활성화
        self.disable_cuda_graph_kv_split = True
        
        # 더 큰 split을 지원하기 위해 workspace size를 2GB로 증가
        global_config.flashinfer_workspace_size = 2048 * 1024 * 1024
```

FlashInfer는 고정 split size의 FA2 kernel을 사용한다. 전통적인 FlashAttention은 입력 크기에 따라 동적으로 split 전략을 선택하므로, batch size가 다르면 계산 순서가 달라진다. FlashInfer의 해결책:

1. **고정 Split Tile Size**: 환경 변수로 prefill과 decode 단계의 split tile 크기를 미리 설정
2. **Batch Invariant Kernel**: 전용으로 설계된 batch invariant FA2 kernel 사용
3. **동적 최적화 비활성화**: 비결정성을 일으킬 수 있는 동적 KV split 최적화 비활성화

SGLang에서 FlashInfer Attention Backend 확정적 모드의 핵심 설정:

- **SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE**: Prefill 단계의 split tile 크기, 기본값 4096
- **SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE**: Decode 단계의 split tile 크기, 기본값 2048
- **decode_use_tensor_cores**: decode 단계에서 tensor cores 강제 사용
- **disable_cuda_graph_kv_split**: CUDA Graph 모드에서 KV split 최적화 비활성화
- **workspace_size**: 더 큰 split 연산을 지원하기 위해 2GB로 증가

### 0x4.2 Triton Attention Backend의 확정적 지원 및 확정적 kernel 추론 테스트

Triton Attention Backend도 해당하는 확정적 설정이 있다:

```python
# 출처: sglang/python/sglang/srt/layers/attention/triton_backend.py
def __init__(self, model_runner):
    # ... 기타 초기화 코드
    
    self.enable_deterministic = (
        model_runner.server_args.enable_deterministic_inference
    )
    
    if self.enable_deterministic:
        # 환경 변수에서 split tile size 가져오기
        self.split_tile_size = get_int_env_var(
            "SGLANG_TRITON_ATTENTION_SPLIT_TILE_SIZE", None
        )
    else:
        # 서버 파라미터의 설정 사용
        self.split_tile_size = (
            model_runner.server_args.triton_attention_split_tile_size
        )
    
    # split 수 계산
    if self.split_tile_size is not None:
        self.num_splits = (
            self.max_context_len + self.split_tile_size - 1
        ) // self.split_tile_size
```

### 0x4.3 FlashAttention3 Attention Backend의 확정적 지원

FlashAttention3 Attention Backend(FA3)는 split 수를 제어해 확정적 추론을 구현한다. PR #10651(https://github.com/sgl-project/sglang/pull/10651) 의 구현에 따르면, FA3의 확정적 지원은 주로 다음 방식으로 이루어진다:

```python
# 출처: sglang/python/sglang/srt/layers/attention/flashattention_backend.py
def __init__(self, model_runner):
    # ... 기타 초기화 코드
    
    # 확정적 추론 설정: split 수 제어
    # 확정적 추론 활성화 시 num_splits를 1로 설정, 아니면 0(자동 선택)
    # 참고: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
    self.num_splits = (
        1 if model_runner.server_args.enable_deterministic_inference else 0
    )
```

FA3 확정적 추론의 핵심 원리는 split 전략을 고정하는 것이다. 비확정적 모드에서 FlashAttention3는 입력 크기에 따라 자동으로 다른 split 수를 선택해 성능을 최적화할 수 있지만, 이는 batch size가 다르면 계산 순서가 달라지게 한다. `num_splits`를 1로 고정함으로써 계산의 일관성을 보장한다.

PR #10651의 테스트 결과에 따르면, FA3는 확정적 모드에서 모든 테스트를 통과했다:

```bash
# Prefix 모드 테스트 - 서로 다른 prefix 길이의 확정성 테스트
python3 -m sglang.test.test_deterministic --test-mode prefix --n-trials 50
# 결과:
# Prompt 0 with prefix length 1: total samples: 312, Unique samples: 1
# Prompt 1 with prefix length 511: total samples: 334, Unique samples: 1
# Prompt 2 with prefix length 2048: total samples: 326, Unique samples: 1
# Prompt 3 with prefix length 4097: total samples: 303, Unique samples: 1

# Mixed 모드 테스트 - 혼합 길이 prompt의 확정성 테스트
python3 -m sglang.test.test_deterministic --test-mode mixed --n-trials 50
# 결과:
# Prompt 1: total samples: 530, Unique samples: 1
# Prompt 2: total samples: 530, Unique samples: 1
# Long prompt: total samples: 215, Unique samples: 1
```

모든 테스트의 `Unique samples`가 1이어서 실제로 확정성이 달성됐음을 알 수 있다.

또한 FA3의 확정적 구현은 Radix Cache도 지원하며, 테스트 결과는 다음과 같다:

```bash
# Radix Cache를 활성화한 prefix 모드 테스트
# 결과:
# Prompt 0 with prefix length 1: total samples: 315, Unique samples: 1
# Prompt 1 with prefix length 511: total samples: 299, Unique samples: 1
# Prompt 2 with prefix length 1728: total samples: 302, Unique samples: 1
# Prompt 3 with prefix length 2345: total samples: 359, Unique samples: 1
```

FA3의 확정적 구현이 SGLang의 캐시 메커니즘과 호환됨을 알 수 있다.

## 0x5. 확정적 추론에서의 AllReduce 변경

다중 GPU 분산 추론에서 SGLang은 AllReduce 연산에도 특수한 처리를 적용해 통신의 확정성을 보장한다. 전통적인 AllReduce는 통신 순서의 무작위성, 부동소수점 누적 순서의 차이, 알고리즘의 동적 선택 등으로 인해 결과가 비결정적일 수 있다.

SGLang의 해결책은 직접적이다: NCCL의 tree 알고리즘을 강제로 사용하고, 커스텀 AllReduce 구현을 비활성화한다.

```python
# 출처: sglang/python/sglang/srt/server_args.py
def _handle_deterministic_inference(self):
    if self.enable_deterministic_inference:
        if self.tp_size > 1:
            # tree 알고리즘 강제 사용
            os.environ["NCCL_ALGO"] = "allreduce:tree"
            # 커스텀 AllReduce 구현 비활성화
            self.disable_custom_all_reduce = True
```

PR 참고: https://github.com/sgl-project/sglang/pull/10930

## 0x6. SGLang 확정적 추론 환경 변수 및 테스트 스크립트

SGLang은 확정적 추론을 제어하는 여러 환경 변수를 사용한다:


```bash
# FlashInfer prefill 단계의 split tile size, 기본값 4096
export SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE=4096

# FlashInfer decode 단계의 split tile size, 기본값 2048
export SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE=2048
```


```bash
# Triton attention의 split tile size
export SGLANG_TRITON_ATTENTION_SPLIT_TILE_SIZE=2048
```


`server_args.py`에서 확정적 추론은 관련 환경 변수를 자동으로 설정한다:

```python
# 출처: sglang/python/sglang/srt/server_args.py
def prepare_server_env(self):
    # 확정적 추론 환경 변수 설정
    os.environ["SGLANG_DETERMINISTIC_INFERENCE"] = (
        "1" if self.enable_deterministic_inference else "0"
    )
    
    # 기타 환경 변수 설정
    if self.enable_deterministic_inference:
        # 관련 환경 변수 추가 설정이 필요할 수 있음
        pass
```


SGLang은 전용 테스트 스크립트 `test_deterministic.py`를 제공해 확정적 추론의 효과를 검증한다:

```python
# 출처: sglang/python/sglang/test/test_deterministic.py
def test_deterministic(args):
    # 워밍업 단계
    for i in range(3):
        send_single(args, 16, args.profile)
    
    if args.test_mode == "single":
        # 단일 모드: 동일한 prompt로 서로 다른 batch size 테스트
        texts = []
        for i in range(1, args.n_trials + 1):
            batch_size = i
            text = send_single(args, batch_size, args.profile)
            texts.append(text.replace("\n", " "))
        
        print(f"Total samples: {len(texts)}, Unique samples: {len(set(texts))}")
    
    elif args.test_mode == "mixed":
        # 혼합 모드: 동일 batch 내에서 서로 다른 길이의 prompt 혼합
        # ... 혼합 테스트 로직
        
    elif args.test_mode == "prefix":
        # Prefix 모드: 서로 다른 길이의 공통 prefix를 가진 prompt 테스트
        # ... prefix 테스트 로직
```

- Single 모드: 동일한 prompt로 서로 다른 batch size에서의 일관성 테스트
- Mixed 모드: 동일 batch 내에서 서로 다른 길이의 prompt 혼합
- Prefix 모드: 서로 다른 길이의 공통 prefix를 가진 prompt 테스트


이상적인 확정적 추론 테스트 결과는 다음과 같아야 한다:

```bash
# Single 모드 테스트 결과
Total samples: 50, Unique samples: 1

# Mixed 모드 테스트 결과  
Prompt 1: total samples: 459, Unique samples: 1
Prompt 2: total samples: 600, Unique samples: 1
Long prompt: total samples: 216, Unique samples: 1

# Prefix 모드 테스트 결과
Prompt 0 with prefix length 1: total samples: 314, Unique samples: 1
Prompt 1 with prefix length 511: total samples: 297, Unique samples: 1
Prompt 2 with prefix length 2048: total samples: 340, Unique samples: 1
Prompt 3 with prefix length 4097: total samples: 324, Unique samples: 1
```

모든 테스트의 `Unique samples`가 1이어야 한다.

## 0x6. 확정적 kernel 추론 사용 권고 사항 및 모범 사례

### 0x6.1 확정적 kernel 추론 활성화

```bash
# 기본 활성화 방법
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend flashinfer \
    --enable-deterministic-inference

# 커스텀 split tile size
export SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE=4096
export SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE=2048
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend flashinfer \
    --enable-deterministic-inference
```

### 0x6.2 클라이언트 요청 예시

```python
import requests

# 확정적 추론 요청 전송
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Tell me about machine learning",
        "sampling_params": {
            "temperature": 0.7,
            "max_new_tokens": 100,
            "sampling_seed": 42  # 확정적 시드 지정
        }
    }
)
```

## 0x8. 요약

이 노트는 SGLang 확정적 추론의 기술 구현을 분석했다. SGLang은 주로 다섯 가지 측면에서 딥러닝 추론의 무작위성 문제를 해결했다:

1. **Batch Invariant Ops** - 서로 다른 batch size에서 계산 결과 일관성 보장
2. **확정적 샘플링** - Gumbel-Max와 위치 의존 시드 생성 기반
3. **Attention Backend 확정적 설정** - FlashInfer, Triton, FA3 등 다양한 backend 지원
4. **AllReduce 확정적 변경** - 분산 시나리오에서 통신의 확정성 보장
5. **환경 변수 제어** - 각종 환경 변수로 세밀하게 제어

전반적으로 이 방안은 상당히 완전하며, 추론 흐름 중 무작위성이 발생할 수 있는 각 부분을 기본적으로 커버한다. https://lmsys.org/blog/2025-09-22-sglang-deterministic/ 공식 블로그에서는 이 확정적 kernel 구현과 Slime을 결합해 100% 재현 가능한 안정적인 RL 훈련 프레임워크 흐름을 얻었다.
