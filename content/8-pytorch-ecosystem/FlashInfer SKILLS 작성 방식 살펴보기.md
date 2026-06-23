# FlashInfer SKILLS 작성 방식 살펴보기

## 0x0. 배경

최근 FlashInfer 저장소를 보다가 `.claude/skills/` 디렉터리 아래에 세 개의 SKILL 파일이 유지되고 있다는 것을 발견했습니다. SKILL은 개발자가 프로젝트 저장소 안에 구조화된 안내 문서, 보통 `SKILL.md` 파일을 두고, AI 코드 도우미가 작업을 수행할 때 이를 읽고 따르도록 하는 방식입니다. 핵심 아이디어는 프로젝트 고유의 개발 흐름, debug 방법, best practice 같은 지식을 문서로 인코딩하는 것입니다. 그러면 개발자가 Claude Code/Cursor에서 AI에게 관련 작업을 시킬 때, AI는 먼저 대응하는 SKILL 파일을 읽고 그 안의 단계에 따라 실행합니다. 일반 지식으로 추측하지 않는다는 점이 중요합니다.

예를 들어 Cursor에서 AI에게 "FlashInfer에 새 CUDA kernel을 추가해 줘"라고 요청하면, AI는 자동으로 `add-cuda-kernel/SKILL.md`를 읽고 FlashInfer 프로젝트가 자체 정의한 파일 구조, naming 규칙, 테스트 요구 사항을 엄격히 따라 코드를 생성합니다. 일반 템플릿을 쓰지 않습니다. FlashInfer처럼 복잡한 빌드 시스템(TVM-FFI, JIT 컴파일)과 특정 코드 조직 규칙이 있는 프로젝트에는 가치가 큽니다. 반복적인 다회 대화를 많이 줄이고 시간과 많은 token을 아낄 수 있습니다.

FlashInfer는 현재 세 개의 SKILL을 유지하고 있습니다.

- `debug-cuda-crash`: CUDA crash debug 튜토리얼
- `benchmark-kernel`: Kernel 성능 benchmark 가이드
- `add-cuda-kernel`: 새 CUDA kernel 추가 전체 흐름

아래에서 하나씩 간단히 소개합니다. 짧은 번역에 가깝습니다.

## 0x1. debug-cuda-crash: CUDA Crash debug

이 SKILL의 핵심은 FlashInfer의 `@flashinfer_api` logging decorator를 중심으로 CUDA crash debug를 수행하는 것입니다.

### 문제 배경

CUDA 오류, 예를 들어 illegal memory access, out-of-bounds, NaN/Inf 등은 종종 프로그램을 바로 crash시키고, crash 뒤에는 아무 debug 정보도 남기지 않습니다. FlashInfer의 `@flashinfer_api` decorator는 API **실행 전**에 입력 정보를 기록합니다. 그래서 프로그램이 crash 나더라도 마지막 호출의 입력이 무엇이었는지 볼 수 있습니다.

### 사용 방식

환경 변수로 log level과 출력 대상을 제어합니다.

| 변수 | 값 | 설명 |
|------|-----|------|
| `FLASHINFER_LOGLEVEL` | `0` | 기록하지 않음(기본값) |
| | `1` | 함수 이름만 기록 |
| | `3` | 입력/출력 메타정보(shape, dtype, device 등) 기록 |
| | `5` | tensor 통계 정보(min/max/mean/nan_count/inf_count) 추가 기록 |
| `FLASHINFER_LOGDEST` | `stdout` | 콘솔 출력(기본값) |
| | `stderr` | stderr로 출력 |
| | `<path>` | 파일로 출력 |
| | `log_%i.txt` | 다중 프로세스 모드, `%i`가 프로세스 ID로 대체됨 |

전형적인 debug 흐름은 다음과 같습니다.

```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=debug.log
python my_script.py
```

이후 `debug.log`를 보면 마지막 API 호출 기록이 crash 직전의 입력입니다. 예를 들어 shape mismatch 예시는 log에 다음처럼 표시됩니다.

```
[2025-12-18 10:32:15] FlashInfer API Call: batch_decode_with_padded_kv_cache
Positional input arguments:
  arg[0]:
    Tensor(
      shape=(32, 8, 128)      # Query tensor
      ...
    )
Keyword input arguments:
  kv_cache=
    Tensor(
      shape=(1024, 2, 8, 64)  # ❌ Wrong! Should be (..., 128) not (..., 64)
      ...
    )
```

이렇게 하면 `head_dim` mismatch(64 vs 128)를 찾을 수 있습니다.

### 흔한 오류의 조사 방법

문서는 네 가지 흔한 CUDA 오류의 조사 방법을 정리합니다.

1. **Illegal Memory Access**: Level 3으로 tensor shape, CUDA 위에 있는지, stride가 합리적인지, contiguous인지 확인합니다.
2. **NaN/Inf**: Level 5로 `nan_count`, `inf_count`, `min`/`max`가 비정상인지 확인합니다. 흔한 원인은 divide by zero, overflow, uninitialized memory입니다.
3. **Out of Memory**: Level 3으로 tensor shape가 의도치 않게 너무 큰지 확인합니다.
4. **Wrong Dtype**: Level 3으로 dtype 필드를 직접 확인합니다.

### 다중 프로세스 debug

다중 GPU 장면에서는 `%i` 모드로 각 프로세스에 독립 log를 생성할 수 있습니다.

```bash
export FLASHINFER_LOGDEST=debug_rank_%i.txt
torchrun --nproc_per_node=4 my_script.py
```

### 고급 debug

문서는 `compute-sanitizer`와 `cuda-gdb`를 결합하는 방법, 그리고 CUDA kernel에서 `printf()`로 debug하는 방법도 소개합니다. 여기서 주의할 점이 있습니다. warp-specialized kernel의 경우 단순히 `threadIdx.x == 0`을 출력 조건으로 쓰면 안 됩니다. 그러면 warp 0만 출력하므로, kernel 설계에 따라 각 group의 대표 thread를 선택해야 합니다.

또한 문서는 Level 5 통계 정보가 CUDA graph capture 중에는 자동으로 건너뛰어진다고 언급합니다. 동기화를 피하기 위한 것이며 정상 동작입니다. log 기능은 꺼져 있을 때(`LOGLEVEL=0`) zero-overhead이며, decorator는 원래 함수를 직접 반환합니다.

## 0x2. benchmark-kernel: Kernel benchmark

이 SKILL은 FlashInfer의 kernel을 정확히 성능 테스트하는 방법을 소개합니다.

### 계측 방법

FlashInfer는 두 가지 timing 방식을 지원합니다.

1. **CUPTI(권장)**: 하드웨어 수준 profiling으로, host-device 동기화 overhead 없이 순수 GPU 계산 시간을 측정합니다. `cupti-python >= 13.0.0`(CUDA 13+)이 필요합니다.
2. **CUDA Events(fallback)**: 표준 CUDA event timing입니다. CUPTI를 사용할 수 없으면 자동으로 사용합니다. 정밀도는 약간 낮고, 매우 빠른 kernel(5-50 us)에는 동기화 overhead가 있지만 더 긴 kernel에는 영향이 무시할 만합니다.

프레임워크는 CUPTI 사용 가능 여부를 자동 감지하므로 수동 전환이 필요 없습니다. 설치 방식은 `pip install -U cupti-python`입니다.

### 방법 1: flashinfer_benchmark.py 사용

권장 benchmark 방식입니다. 지원하는 테스트 routine은 다음과 같습니다.

- **Attention**: `BatchDecodeWithPagedKVCacheWrapper`, `BatchPrefillWithPagedKVCacheWrapper`, `BatchPrefillWithRaggedKVCacheWrapper`, `BatchMLAPagedAttentionWrapper`
- **GEMM**: `bmm_fp8`, `gemm_fp8_nt_groupwise`, `group_gemm_fp8_nt_groupwise`, `mm_fp4`
- **MOE**: `trtllm_fp4_block_scale_moe`, `trtllm_fp8_block_scale_moe`, `trtllm_fp8_per_tensor_scale_moe`, `cutlass_fused_moe`

decode attention benchmark 예시는 다음과 같습니다.

```bash
python benchmarks/flashinfer_benchmark.py \
    --routine BatchDecodeWithPagedKVCacheWrapper \
    --backends fa2 fa2_tc cudnn \
    --page_size 16 \
    --batch_size 32 \
    --s_qo 1 \
    --s_kv 2048 \
    --num_qo_heads 32 \
    --num_kv_heads 8 \
    --head_dim_qk 128 \
    --head_dim_vo 128 \
    --q_dtype bfloat16 \
    --kv_dtype bfloat16 \
    --num_iters 30 \
    --dry_run_iters 5 \
    --refcheck \
    -vv
```

출력에는 네 가지 핵심 지표가 포함됩니다.

- **median time**: kernel 실행 시간의 median, 낮을수록 좋습니다.
- **std**: 표준편차, 낮을수록 안정적입니다.
- **achieved tflops**: 유효 TFLOPS 처리량
- **achieved tb_per_sec**: 메모리 bandwidth 활용률

여러 파라미터 조합을 testlist 파일에 적어 한 번에 실행하는 batch test도 지원합니다.

```bash
python benchmarks/flashinfer_benchmark.py \
    --testlist my_benchmarks.txt \
    --output_path results.csv \
    --generate_repro_command \
    --refcheck
```

자주 쓰는 flag에는 `--num_iters`(측정 반복 횟수, 기본 30), `--dry_run_iters`(warmup 횟수, 기본 5), `--refcheck`(출력 correctness 검증), `--use_cuda_events`(CUDA events 강제 사용), `--no_cuda_graph`(CUDA graph 비활성화), `--generate_repro_command`(재현 명령 출력) 등이 있습니다.

### 방법 2: Python에서 bench_gpu_time() 사용

커스텀 benchmark 스크립트에서는 FlashInfer가 제공하는 `bench_gpu_time` 함수를 직접 사용할 수 있습니다.

```python
from flashinfer.testing import bench_gpu_time

median_time, std_time = bench_gpu_time(
    my_kernel_wrapper,
    args=(q, k, v),
    enable_cupti=True,          # Prefer CUPTI, automatically fallback to CUDA events.
    num_iters=30,
    dry_run_iters=5,
)
```

`cold_l2_cache=True`를 사용해 cold L2 cache benchmark를 수행할 수도 있습니다.

### 문제 조사

문서는 몇 가지 흔한 문제를 나열합니다. 결과가 불안정하면 warmup과 측정 반복 횟수를 늘리거나 `cold_l2_cache`를 사용할 수 있습니다. reference check가 실패하면 `--allow_output_mismatch`를 추가해 계속 실행할 수 있습니다. 일부 backend가 현재 GPU 아키텍처를 지원하지 않을 때는 명확한 warning 정보가 표시됩니다.

## 0x3. add-cuda-kernel: 새 CUDA Kernel 추가

이 SKILL은 가장 길고 자세한 문서입니다. 간단한 element-wise scale 작업, 즉 `scale(x, factor) = x * factor`를 예로 들어 FlashInfer에 새 kernel을 추가하는 전체 흐름을 10단계로 훑습니다.

### Step 1: include/에 CUDA Kernel 정의

`include/flashinfer/scale.cuh`를 만듭니다. 요구 사항은 **프레임워크 독립적**이어야 한다는 것입니다. Torch header에 의존하지 않고 raw pointer와 template를 사용해 여러 dtype을 지원합니다.

```cpp
namespace flashinfer {

template <typename T>
__global__ void ScaleKernel(const T* input, T* output, T factor, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = input[idx] * factor;
  }
}

template <typename T>
cudaError_t ScaleLauncher(const T* input, T* output, T factor, int n,
                          cudaStream_t stream = nullptr) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  ScaleKernel<T><<<blocks, threads, 0, stream>>>(input, output, factor, n);
  return cudaGetLastError();
}

}  // namespace flashinfer
```

### Step 2: csrc/에 Launcher 생성

`csrc/scale.cu`를 만듭니다. 이 계층은 TVM-FFI의 `TensorView`를 raw pointer로 변환하고, 입력 검증과 dtype dispatch를 담당합니다. `csrc/` 디렉터리 아래에서만 TVM FFI header를 가져올 수 있습니다.

문서는 TVM-FFI의 오류 처리 메커니즘을 자세히 소개합니다.

- `TVM_FFI_THROW(ValueError) << "message"`: 일반 runtime 오류
- `TVM_FFI_LOG_AND_THROW(InternalError) << "message"`: 생성자 또는 초기화 단계처럼 예외가 정상 전파되지 않을 수 있는 경우에 사용

### Step 3: TVM-FFI Binding 생성

`csrc/scale_jit_binding.cu`를 만들고, `TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, scale_launcher)`로 launcher 함수를 TVM-FFI 인터페이스로 export합니다.

### Step 4: JIT Generator 생성

`flashinfer/jit/scale.py`를 만듭니다. 이 파일은 JIT 컴파일 흐름을 담당합니다. 단순 kernel의 경우 Jinja template가 필요 없고, source file을 생성 디렉터리로 복사하면 됩니다. URI는 module config를 고유하게 식별하는 데 쓰입니다. 문서는 **package 디렉터리에 파일을 쓰면 안 된다**는 점을 강조합니다.

이 단계에서는 CUDA 아키텍처 target을 관리하는 `CompilationContext` 메커니즘도 소개합니다. `supported_major_versions` 파라미터로 kernel이 지원하는 SM version을 지정할 수 있습니다.

| 파라미터 | 지원 아키텍처 | 사용 장면 |
|------|-----------|---------|
| `None` | 모든 사용 가능 GPU | 범용 kernel |
| `[9, 10, 11, 12]` | SM90, SM100, SM110, SM120 | Hopper 및 이후 |
| `[10, 11, 12]` | SM100, SM110, SM120 | Blackwell 및 이후 |
| `[12]` | SM120 | 특정 아키텍처 |

환경 변수 `FLASHINFER_CUDA_ARCH_LIST`로 수동 override할 수도 있습니다.

### Step 5: Python API 생성

`flashinfer/scale.py`를 만듭니다. 이것이 사용자가 직접 호출하는 인터페이스입니다. 문서에는 몇 가지 핵심 design pattern이 나옵니다.

- **`@functools.cache`**: 컴파일된 module을 cache해 반복 컴파일을 피합니다.
- **`@flashinfer_api`**: logging 기능을 활성화합니다. `debug-cuda-crash` SKILL의 메커니즘과 같습니다.
- **Destination passing style**: output tensor를 optional parameter로 전달합니다(`out: Optional[torch.Tensor] = None`). 사용자가 미리 할당한 buffer를 전달해 allocation overhead를 피할 수 있습니다.
- **`@backend_requirement` 및 `@supported_compute_capability` decorator**: 입력 검증과 backend 선택을 수행합니다.

`@backend_requirement` decorator에는 세 가지 사용 모드가 있습니다.

1. **단일 backend**: `backend_checks={}`는 backend 선택이 없고 일반 검증만 수행한다는 뜻입니다.
2. **여러 backend**: `backend_checks` dictionary에 각 backend별 독립 검증 함수를 등록합니다.
3. **자동 backend 선택**: `heuristic_func`를 제공해 입력 파라미터에 따라 최적 backend를 자동 선택합니다.

decorator는 함수에 `is_backend_supported()`, `is_compute_capability_supported()`, `has_backend()` 같은 helper method도 추가하고, 성능 핵심 경로에서 검증을 건너뛰기 위한 `skip_check=True` 파라미터도 제공합니다.

### Step 6-10: 테스트, AOT 등록, export, 실행, Benchmark

- **Step 6**: pytest로 unit test를 작성하고 `pytest.mark.parametrize`로 여러 dtype과 size 조합을 테스트하며 reference 구현과 비교합니다. kernel에 아키텍처 요구 사항이 있으면 `pytest.skip`으로 지원하지 않는 GPU를 건너뜁니다.
- **Step 7**: `flashinfer/aot.py`에 등록해 흔한 config를 미리 컴파일합니다. 그러면 `flashinfer-jit-cache`를 설치한 사용자는 JIT 컴파일을 건너뛸 수 있습니다.
- **Step 8**: `flashinfer/__init__.py`에서 API를 export합니다.
- **Step 9**: 테스트를 직접 실행합니다. kernel은 최초 사용 시 자동으로 컴파일됩니다.
- **Step 10**: benchmark 스크립트를 추가하고 `bench_gpu_time` 함수로 성능 테스트를 수행합니다. 문서는 모든 새 kernel에 benchmark가 있어야 한다고 강조합니다.

### 최종 파일 목록

전체 흐름에서 관련되는 파일은 다음과 같습니다.

```
include/flashinfer/scale.cuh              # New: CUDA kernel definition
csrc/scale.cu                              # New: Launcher
csrc/scale_jit_binding.cu                  # New: TVM-FFI binding
flashinfer/jit/scale.py                    # New: JIT generator
flashinfer/scale.py                        # New: Python API
flashinfer/__init__.py                     # Modified: export API
flashinfer/aot.py                          # Modified: register AOT
tests/test_scale.py                        # New: unit tests
benchmarks/bench_scale.py                  # New: benchmark script
```

## 0x4. 요약

FlashInfer의 세 SKILL 파일은 프로젝트에서 가장 핵심적인 세 개발 장면, 즉 debug, 성능 테스트, 새 kernel 추가의 전체 흐름과 best practice를 문서화합니다. 실용적 관점에서 보면 `add-cuda-kernel` SKILL이 FlashInfer에 새 kernel을 기여하려는 개발자에게 가장 도움이 큽니다. CUDA kernel 정의부터 Python API 노출까지 전체 경로를 한 번 훑기 때문입니다. TVM-FFI, JIT 컴파일, decorator pattern 같은 FlashInfer 고유 메커니즘은 이 문서가 없으면 이해 비용이 훨씬 높습니다. `debug-cuda-crash`와 `benchmark-kernel`은 일상 사용에 더 가깝습니다. 전자는 CUDA 오류를 조사할 때 유용할 수 있고, 후자는 성능 비교 시 정확한 kernel-level timing을 제공합니다.

개인적으로는 kernel 추가의 번거로운 흐름을 SKILLS로 만드는 것이 아주 괜찮다고 느낍니다. 모두에게 이롭습니다.
