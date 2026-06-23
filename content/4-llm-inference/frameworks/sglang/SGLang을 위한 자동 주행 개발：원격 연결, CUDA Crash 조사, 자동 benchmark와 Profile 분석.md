# SGLang을 위한 자동 주행 개발：원격 연결, CUDA Crash 조사, 자동 benchmark와 Profile 분석

## 0x0. 서문

이 글은 필자가 최근 자주 사용하는 SGLang 관련 SKILL 몇 가지를 정리한 것입니다. 내용은 debug, benchmark, remote development, performance analysis 등을 포괄합니다. 일부 일상 개발 경험, 검증 흐름, 조사 방법을 추출해 Codex가 skill 형태로 재사용하도록 만든 것이라고 이해할 수 있습니다. 이 내용이 Auto Driven AI Infra 실천에 참고가 되길 바랍니다. 아래에서 언급하는 skill들은 이미 여러 모델과 여러 시나리오에서 검증했지만, 물론 계속 수정해야 할 부분이 남아 있을 수도 있습니다.

SGLang CUDA Debug Crash SKILL과 SGLang Auto-Driven Benchmark SKILL은 SGLang system에 조금 침투해야 정상 동작하므로 각각 다음 위치에 두었습니다: https://github.com/sgl-project/sglang/tree/main/.claude/skills/debug-cuda-crash & https://github.com/sgl-project/sglang/pull/21736

나머지 SKILLS는 모두 여기에 있습니다.

**https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS.git** **많은 관심 부탁드립니다**

목표는 더 많은 SGLang 관련 작업을 점차 Agent가 자동으로 완료하고, 자동으로 분석하고, 자동으로 최적화할 수 있는 흐름으로 넣는 것입니다.

아래에서는 기능별로 이 SKILL들을 소개합니다.

## 0x1. 원격 연결 SKILL

- https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/b200
- https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/h100
- H200 diffusion remote skill(같은 저장소 안에 있으며, 이 글에서는 구체적인 host identifier를 펼치지 않습니다)

이 skill의 역할은 비교적 기본적이지만, 실제 개발에서는 매우 중요합니다. 많은 SGLang 검증 작업이 원래 remote GPU server에 의존하기 때문입니다. 모델 로딩, kernel smoke test, end-to-end service validation, benchmark, profiler 수집 등이 모두 여기에 포함됩니다. 계속 "각 machine마다 agent를 따로 설정하고, 환경 세부사항도 따로 기억하는" 방식을 쓰면 전환 비용이 높고 오류도 나기 쉽습니다.

이 skill의 핵심 목표는 local에서 실행 중인 Codex 또는 Claude Code가 remote GPU machine에 직접 접속하고, remote host를 통일된 execution backend로 만들도록 하는 것입니다. 각 서버에 agent를 반복 배포하지 않아도 됩니다.

주요 기능은 다음과 같습니다.

- remote host, 기본 container, 기본 repo path를 약속해 수동으로 환경을 찾는 비용을 줄입니다.
- `hostname`, `docker ps`, `nvidia-smi`, GPU idle 상태 확인 등 통일된 host check flow를 제공합니다.
- 기본 개발 container에 바로 진입하고 `HF_TOKEN`, Hugging Face cache, FlashInfer 관련 환경이 갖춰졌는지 확인합니다.
- 안전한 remote workflow를 제공합니다. 기본적으로 remote repo 상태를 먼저 확인한 뒤, 직접 사용할지, detached worktree를 만들지, local working tree를 remote temporary directory로 sync할지 결정합니다.
- local의 현재 working tree를 remote validation directory로 streaming sync할 수 있어, "현재 local 변경사항 검증" 같은 시나리오에 적합합니다.
- `py_compile`, `compileall`, `pytest`, GPU smoke test, server-level validation 등을 포함한 통일된 remote validation flow를 제공합니다.
- `b200`, `h100`, `h200`처럼 machine별 전용 skill 버전을 유지해 GPU architecture, container 이름, repo path, cleanup 방식, 적합한 task 유형 등을 skill에 기록하고, 환경을 잘못 고를 확률을 낮춥니다.
- diffusion, `torch.compile`, multi-node serving 같은 특수 검증 시나리오에서도 서로 다른 machine에서 일관된 작업 방식을 재사용할 수 있습니다.

아래는 현재 바로 재사용할 수 있는 remote probe output 예시입니다. 필요한 field만 남겼고, hostname과 container name은 모두 masking했습니다.

```bash
$ ssh h100_sglang 'hostname && whoami'
[redacted-h100-host]
sglang

$ ssh b200 'hostname && whoami'
[redacted-b200-host]
lmsys

$ ssh h200_diffusion 'hostname && whoami'
[redacted-h200-host]
sglang-rl

$ ssh h100_sglang 'docker ps --format "{{.Names}}|{{.Status}}" | sed -n "1,3p"'
spec-dev-container|Up 2 days
sglang_bbuf|Up 4 days
aux-dev-container|Up 3 days

$ ssh b200 'nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | sed -n "1,3p"'
0, NVIDIA B200, 160491, 183359
1, NVIDIA B200, 4, 183359
2, NVIDIA B200, 4, 183359

$ ssh h200_diffusion 'docker ps --format "{{.Names}}|{{.Status}}" | sed -n "1,3p"'
diffusion-dev-a|Up 21 hours
diffusion-dev-b|Up 34 hours
omni-dev-container|Up 33 hours
```

이런 skill은 직접 보여줄 성능 향상 데이터는 없지만, 더 상위의 engineering 문제를 해결합니다. 즉 local에서는 통일된 agent session을 유지하면서도 서로 다른 여러 GPU server에 안정적으로 접근하고, 반복적인 환경 준비 작업을 최소화하는 문제입니다.

## 0x2. SGLang CUDA Debug Crash SKILL

- https://github.com/sgl-project/sglang/tree/main/.claude/skills/debug-cuda-crash

이 skill의 목표는 CUDA crash의 위치 추적 과정을 "예외 stack과 추측만 남은 상태"에서 "kernel/custom op 경계에서 충분한 context를 확보할 수 있는 문제"로 바꾸는 것입니다. SGLang처럼 JIT kernel, custom op, Triton kernel, model wrapper logic이 동시에 포함된 system에서는 특히 중요합니다.

구현 아이디어는 FlashInfer의 API logging에서 왔습니다. crash는 결국 어떤 호출 경계 근처에서 발생하므로, 실행 전에 입력을 기록해 두어야 합니다. 프로그램이 예외로 종료된 뒤 흐릿한 CUDA error 한 줄만 남는 상황을 피하기 위해서입니다.

주요 기능은 다음과 같습니다.

- `register_custom_op(...)`, `register_custom_op_from_extern(...)`, LLM attention/linear/quantization wrapper, diffusion attention/linear/rotary wrapper, 일부 `torch.ops.sglang.*` hot path 등 SGLang의 핵심 kernel boundary를 포괄합니다.
- 필요에 따라 정보량을 제어하기 쉽도록 level별 log output을 지원합니다.
- `SGLANG_KERNEL_API_LOGLEVEL=1`은 API call boundary와 exception boundary를 출력합니다.
- `SGLANG_KERNEL_API_LOGLEVEL=3`은 입력 tensor의 shape, dtype, device, contiguous 등 metadata를 출력합니다.
- `SGLANG_KERNEL_API_LOGLEVEL=5`는 min/max/mean, NaN/Inf statistics를 추가로 출력해 numerical issue 조사에 도움을 줍니다.
- `SGLANG_KERNEL_API_LOGLEVEL=10`은 실행 전에 `inputs.pt`, `metadata.json`을 자동 저장하고, 예외 상황에서 전체 call snapshot을 보존합니다.
- dump를 지정 directory에 저장할 수 있어 offline reproduction에 편리합니다.
- CUDA Graph가 켜져 있으면 안전하지 않은 tensor dump를 자동으로 건너뛰지만, call boundary log는 유지합니다. debugging logic이 graph capture에 역으로 영향을 주는 것을 피하기 위해서입니다.
- LLM과 diffusion 양쪽의 문제 위치 추적에 모두 적용되며, 한 모델 유형에만 국한되지 않습니다.

아래는 필자가 remote H100에서 실제로 이 skill을 실행해 얻은 log excerpt입니다. 테스트 script는 의도적으로 out-of-bound embedding index를 만들어 custom op가 `torch.cuda.synchronize()`에서 `device-side assert`를 일으키게 했습니다.

```bash
===STDERR===
/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1478: indexSelectSmallIndex: block: [0,0,0], thread: [0,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
...
torch.AcceleratorError: CUDA error: device-side assert triggered

===LEVEL3===
[2026-04-01 15:42:32] SGLang Kernel API Call: sglang_llm_crash.mock_llm_cuda_crash
Positional input arguments:
  arg[0]=Tensor(shape=(2,), dtype=torch.int64, device=cuda:0)
  arg[1]=Tensor(shape=(4, 8), dtype=torch.float16, device=cuda:0)
[2026-04-01 15:42:32] SGLang Kernel API Exception: sglang_llm_crash.mock_llm_cuda_crash

===LEVEL10_DUMPS===
/tmp/sglang_kernel_api_demo/.../inputs.pt
/tmp/sglang_kernel_api_demo/.../metadata.json

===LEVEL10_META===
{
  "function_name": "sglang_llm_crash.mock_llm_cuda_crash",
  "execution_status": "exception",
  "input_tensor_keys": ["arg_0", "arg_1"],
  "exception": {"type": "AcceleratorError", "message": "CUDA error: device-side assert triggered"}
}
```

이 출력은 세 가지를 보여 줍니다. 첫째, 예외 발생 시 마지막 kernel API boundary가 기록되어 있습니다. 둘째, `level 3`만으로도 입력 tensor의 shape, dtype, device를 출력할 수 있습니다. 셋째, `level 10`은 이번 호출의 `inputs.pt`와 `metadata.json`을 disk에 저장해 원래는 순간적으로 사라지는 CUDA crash를 계속 offline analysis할 수 있는 문제 sample로 바꿉니다. 더 완전한 검증 과정은 [PR #20910](https://github.com/sgl-project/sglang/pull/20910#issuecomment-4088048145)을 참고하세요.

따라서 이 skill의 주요 가치는 log를 조금 더 많이 출력하는 데 있지 않습니다. 원래 불안정하고 추적하기 어려운 CUDA crash를 offline reproduction과 후속 analysis가 가능한 문제 sample로 바꾸는 데 있습니다.

## 0x3. SGLang Auto-Driven Benchmark SKILL

- https://github.com/sgl-project/sglang/pull/21736

이 skill의 목표는 SGLang에서 가장 시간이 많이 드는 경험 기반 작업 중 하나인 server flag search와 workload benchmark를 자동 실행, 자동 기록, 중단 후 재개가 가능한 flow로 바꾸는 것입니다. 단순한 benchmark script가 아니라, "configuration generation, service startup, load test, SLA 판단, result summary"를 둘러싼 automation loop입니다.

주요 기능은 다음과 같습니다.

- `run`, `convert`, `validate` 세 entrypoint를 제공합니다.
- `convert`는 `sharegpt`, `custom`, `random`, `generated-shared-prefix` 등의 입력 format을 canonical autobench JSONL로 통일합니다.
- `validate`는 canonical autobench data를 구조적으로 검증해 benchmark 실행 중에야 data format 문제를 발견하는 일을 피합니다.
- `run`은 YAML configuration으로 전체 auto benchmark flow를 구동합니다.
- prompt 또는 YAML에 따라 후보 server flags를 자동 생성하며, 시작 명령을 수동으로 하나씩 이어 붙이지 않아도 됩니다.
- data format에 따라 `sglang`, `sglang-oai`, `sglang-oai-chat` 같은 benchmark backend를 자동 선택합니다.
- 세 단계 search level을 지원합니다.
  - `tier 1`: 최소, 가장 빠른 sanity sweep.
  - `tier 2`: 기본 균형 search.
  - `tier 3`: 최대, 가장 느린 full search.
- `max_candidates`를 지원해 search space가 클 때 후보 수를 수동으로 제한할 수 있습니다.
- 고정 QPS list 모드와 `lower / upper / tolerance`가 있는 QPS binary search를 모두 지원합니다.
- 단일 request rate만 보는 대신 `max_concurrency` 차원의 joint search도 지원합니다.
- `tp`, `dp`, `pp`, `ep` 관련 조합 등 parallelism 관련 configuration search와 derivation을 지원합니다.
- 같은 configuration에서 chat, summarization 등 여러 workload scenario를 동시에 생성하는 dataset expansion을 지원합니다.
- base stage와 speculative stage의 2단계 search를 지원하며, 후자는 speculative/EAGLE 관련 parameter search를 계속할 수 있습니다.
- 실시간 progress reporting을 지원하고, 실행 중 `live_results.jsonl`을 계속 기록합니다.
- 중단 시 완료된 trial result를 보존해 긴 실험이 예기치 않게 종료되어도 전체를 잃지 않게 합니다.
- `resume`을 지원해 중단 후 기존 trial result를 읽고 이어서 실행할 수 있으며, 처음부터 다시 돌리지 않아도 됩니다.
- scenario별 `prepared_dataset.jsonl`, `results.jsonl`, `results.csv`, `summary.md`를 출력하고, multi-scenario인 경우 `scenario_summary.jsonl/csv`를 추가 출력합니다.
- 각 trial의 startup argument, request rate, SLA 통과 여부, throughput, latency metric을 자동 기록해 후속 비교가 쉽습니다.

아래는 필자가 remote H100에서 실제로 이 skill을 실행해 얻은 log excerpt입니다. 실행 시간을 제어하기 위해 단일 GPU, 작은 모델, `tier 1`의 최소 search configuration을 사용했습니다.

```bash
=== Auto Benchmark Plan ===
search.tier=1 (tier 1: smallest and fastest sanity sweep)
qps_plan=fixed qps values=[1.0]
Planned base candidates:
  [1/3] {"model_path": "Qwen/Qwen2.5-0.5B-Instruct", "tp_size": 1, ...}
  [2/3] {"model_path": "Qwen/Qwen2.5-0.5B-Instruct", "tp_size": 1, "chunked_prefill_size": 512, ...}
  [3/3] {"model_path": "Qwen/Qwen2.5-0.5B-Instruct", "tp_size": 1, "chunked_prefill_size": 1024, ...}
scenario=demo
prepared_dataset=/tmp/auto_bench_qwen05_demo/prepared_dataset.jsonl
selected_backend=sglang-oai

=== SUMMARY ===
| Candidate | Stage | QPS | Output tok/s | TTFT ms | TPOT ms | SLA |
| 0 | base | 1.0 | 51.78 | 6.26 | 1.20 | pass |
| 1 | base | 1.0 | 51.78 | 6.05 | 1.21 | pass |
| 2 | base | 1.0 | 51.76 | 6.50 | 1.21 | pass |

=== RESUME RUN ===
resume=true loaded_records=3 scenario=demo
results_jsonl=/tmp/auto_bench_qwen05_demo/results.jsonl
```

이 결과는 skill이 후보 생성, data preparation, service startup, load test, result summary, `resume` 재사용까지 온전히 실행할 수 있음을 보여 줍니다. 이 최소 demo에서는 3개 candidate가 모두 SLA를 만족했고, 두 번째 실행은 기존 trial을 바로 재사용합니다. 더 큰 규모의 search result는 [PR #21736](https://github.com/sgl-project/sglang/pull/21736#issuecomment-4159966660)을 참고하세요.

또한 이 skill의 의미는 더 좋은 configuration을 찾는 데만 있지 않습니다. 겉보기에는 합리적이지만 실제 workload에서는 성립하지 않는 parameter 선택을 체계적으로 배제하는 데도 있습니다.

## 0x4. SGLang Torch Profiler Analysis SKILL

- https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/sglang-torch-profiler-analysis

이 skill은 torch profiler analysis flow를 통일해 정리한 것입니다. 과거 SGLang profiler analysis를 할 때는 trace 수집, stage 분리, Perfetto rendering fix, kernel classification, source mapping, overlap 판단, fuse opportunity 식별을 따로 처리해야 했습니다. 이 skill은 이런 동작을 하나의 entrypoint로 통합합니다.

주요 기능은 다음과 같습니다.

- 여러 script를 흩어 써야 하는 대신 통일된 entry script `analyze_sglang_torch_profile.py`를 제공합니다.
- 네 subcommand를 제공합니다.
  - `triage`: 기본 workflow, 바로 세 개의 main table을 출력합니다.
  - `breakdown`: 단일 trace의 kernel/category 비율 분석.
  - `overlap`: 2-stage overlap analysis.
  - `perfetto-fix`: 일부 overlapped event가 Perfetto에서 누락되어 render되는 문제를 수정합니다.
- 기존 `trace.json(.gz)` 또는 profile directory를 바로 분석할 수 있습니다.
- 실행 중인 SGLang server에 `sglang.profiler`를 직접 trigger하고 probe request를 자동으로 보내 실제 workload를 수집할 수 있습니다.
- `profile_by_stage`를 지원해 `extend/prefill`과 `decode`를 나누어 분석합니다. 일반 서비스에서도 이 모드를 권장합니다. PD separation이 켜져 있다면 prefill worker와 decode worker를 각각 수집해야 합니다.
- `triage`는 기본적으로 세 table을 출력합니다.
  - `Kernel Table`
  - `Overlap Opportunity Table`
  - `Fuse Opportunity Table`
- `breakdown`은 attention, communication, MoE, norm, quantization, memory 등 category별 GPU time 비율 통계를 지원합니다.
- `overlap`은 2-stage 방식을 사용합니다.
  - 1단계는 graph-off의 `mapping trace`를 수집해 `kernel -> cpu op -> python scope` 대응을 복원합니다.
  - 2단계는 graph-on의 `formal trace`를 수집해 실제 serving 형태에서의 overlap space를 판단합니다.
- kernel name에만 머무르지 않고 Python code 위치로 역추적할 수 있습니다.
- "아직 overlap headroom이 있는 kernel"과 "이미 다른 compute로 덮여 optimization benefit이 낮은 kernel"을 구분할 수 있습니다.
- dependency risk, actionable overlap rows, ASCII timeline을 출력해 Perfetto를 열지 않고도 1차 판단을 마칠 수 있습니다.
- `fuse-overlap-catalog`와 대조해 분석 결과를 SGLang에 이미 존재하는 fuse 또는 multi-stream overlap pattern과 매칭합니다. 기존 optimization 부재를 새로운 opportunity로 잘못 판단하는 것을 피하기 위해서입니다.
- `perfetto-fix`는 trace를 후처리해 overlapped kernel이 Perfetto에서 불완전하게 표시되는 문제를 수정할 수 있습니다.

아래 예시는 필자가 remote B200에서 실제로 `triage`를 실행한 결과입니다. 입력은 실제 `mapping trace + formal trace` 조합이고, 모델은 `Qwen/Qwen2.5-0.5B-Instruct`입니다. 출력 head는 다음과 같습니다.

```bash
Triage View
Mapping traces: ...TP-0-DECODE.trace.json.gz
Formal traces: ...TP-0-EXTEND.trace.json.gz, ...TP-0-DECODE.trace.json.gz
Model: Qwen/Qwen2.5-0.5B-Instruct
```

이번 실행의 목표는 의도적으로 제거한 `TP all-reduce + residual/RMSNorm` 경로를 다시 식별할 수 있는지 확인하는 것이었습니다. 결과를 보면 skill의 판단은 성립했습니다. `Kernel Table`은 `decode` 단계에서 `cross_device_reduce_1stage<__nv_bfloat16, 2>`를 다시 드러냈고, `Overlap Opportunity Table`은 이를 high-priority headroom으로 표시했으며, `Fuse Opportunity Table`은 문제를 다시 `layernorm.py::_forward_with_allreduce_fusion`로 되돌려 가리켰습니다. 다시 말해 이 skill은 "어디가 느린가"만 보는 것이 아니라, 인위적으로 제거된 optimization도 다시 찾아낼 수 있습니다.

글의 markdown table 폭이 제한되어 있으므로, 아래에는 main table을 발췌한 버전을 붙입니다. field는 압축했지만 정보 출처는 여전히 skill 기본 `triage`의 실제 출력입니다.

`Kernel Table(발췌)`

| Stage | Kernel | Share | Python location |
| --- | --- | ---: | --- |
| extend/prefill | `cross_device_reduce_1stage` | 89.2% | `custom_all_reduce_ops.py:45 all_reduce` |
| decode | `cross_device_reduce_1stage` | 31.0% | `custom_all_reduce_ops.py:45 all_reduce` |
| decode | `nvjet_tst_32x64_64x16_4x1_v_bz_TNN` | 9.9% | `unquant.py:134 apply` |
| decode | `FusedAddRMSNormKernel` | 9.2% | `unresolved` |
| decode | `nvjet_tst_64x8_64x16_4x1_v_bz_TNT` | 7.6% | `unquant.py:134 apply` |
| decode | `act_and_mul_kernel` | 3.9% | `activation.py:73 forward_cuda` |

`Overlap Opportunity Table(발췌)`

| Stage | Priority | Kernel | Python scope | Recommendation |
| --- | --- | --- | --- | --- |
| extend/prefill | P2 | `cross_device_reduce_1stage` | `device_communicators/custom_all_reduce_ops.py:45 all_reduce` | `check deps` |
| decode | P2 | `cross_device_reduce_1stage` | `device_communicators/custom_all_reduce_ops.py:45 all_reduce` | `check deps` |
| decode | P2 | `FusedAddRMSNormKernel` | `layers/quantization/unquant.py:134 apply` | `check deps` |
| decode | P1 | `nvjet_tst_32x64_64x16_4x1_v_bz_TNN` | `layers/quantization/unquant.py:134 apply` | `try fusion` |
| decode | P5 | `vectorized_elementwise_kernel` | `managers/scheduler.py:2583 run_batch` | `skip overlap` |

`Fuse Opportunity Table(발췌)`

| Stage | Pattern | Share | Current location | Candidate fused path |
| --- | --- | ---: | --- | --- |
| extend/prefill | `TP all-reduce + residual/RMSNorm` | 89.2% | `custom_all_reduce_ops.py:45 all_reduce` | `layernorm.py:89 _forward_with_allreduce_fusion` |
| decode | `TP all-reduce + residual/RMSNorm` | 31.0% | `custom_all_reduce_ops.py:45 all_reduce` | `layernorm.py:89 _forward_with_allreduce_fusion` |



따라서 이 skill은 세 가지 질문에 답하는 데 더 적합합니다. 현재 trace의 주요 시간 소모가 어디서 오는지, 어떤 위치에 아직 overlap space가 남아 있는지, 그리고 이 문제가 SGLang에 이미 존재하는 fuse/overlap path로 재사용 가능한지입니다.


## 0x5. 정리

최근 한동안 SGLang 개발, benchmark, debug, profile flow를 SKILL로 계속 정리하면서 비교적 분명히 느낀 점이 있습니다. 과거에는 개인 경험에 크게 의존하던 많은 작업이 점차 Agent에게 맡길 수 있는 형태가 되고 있습니다. 물론 Agent를 연결한다고 문제가 저절로 사라진다는 뜻은 아닙니다. 더 중요한 것은 먼저 경험, context, validation flow를 명확히 정리하는 것입니다.

이 글에서 이야기한 네 종류의 skill은 각각 서로 다른 층위의 문제를 해결합니다.

- remote connection skill은 multi-machine environment switching 문제를 해결합니다.
- CUDA crash skill은 kernel-level debug 문제를 해결합니다.
- auto benchmark skill은 server flag tuning 문제를 해결합니다.
- torch profiler analysis skill은 trace에서 optimization conclusion까지 이어지는 chain을 해결합니다.

이 시리즈는 이후에도 계속 보충할 예정입니다. 한편으로는 기존 skill을 계속 개선하고, 다른 한편으로는 정말 가치 있는 SGLang 개발 flow를 더 정리해서 여러분의 손을 덜어 주고 싶습니다.
