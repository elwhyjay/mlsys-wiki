# SGLang을 위한 자동 주행 개발 SKILLS 업데이트：Serving 문제 해결 SKILLS, 성능 이상, CUDA Crash, 통신 Hang 등

> https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS 에 관심을 가져주기 바란다. Codex/Claude Code로 SGLang 개발을 함께 자동 주행한다.

## 0x0. 서문

AI-Infra-Auto-Driven-SKILLS가 새로운 SKILLS를 업데이트했다. 이번에 업데이트한 것은 실제 serving 문제 해결에 더 가까운 SKILL이다.

https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/sglang-prod-incident-triage

나중에 SGLang 메인 저장소에 병합된다면 대응 디렉터리는 다음과 같다.

https://github.com/sgl-project/sglang/tree/main/.claude/skills/sglang-prod-incident-triage

이 SKILL은 `torch profiler`, `cuda crash debug`, `distributed hang debug` 같은 전문 skill을 대체하는 것이 아니라, 실제 서비스의 문제를 먼저 하나의 표준 경로로 수렴시키는 것이다. 이 SKILL 이전에도 이미 많은 준비가 있었다. 예를 들어 debug cuda crash(https://github.com/sgl-project/sglang/tree/main/.claude/skills/debug-cuda-crash) , debug distributed hang(https://github.com/sgl-project/sglang/tree/main/.claude/skills/debug-distributed-hang), sglang torch profiler analysis(https://github.com/sgl-project/sglang/tree/main/.claude/skills/sglang-torch-profiler-analysis) 등이 있다. 이 SKILL은 실제 serving 문제 해결 기능을 수행할 때 위의 기존 SKILL들을 높은 강도로 호출한다. 다만 전체 production 환경의 문제 해결 흐름을 더 명확하게 만들고, Agent debug에 더 정확한 context 정보를 제공한다.

전체 아이디어는 다음과 같다.

**먼저 현장을 보존하고, 그다음 사고를 replay하며, 그다음 문제를 올바른 전문 skill로 보낸다.**

사용 방법은 이 SKILL을 붙여서 예외가 발생하는 서비스를 debug하도록 요청하는 것이다. 원래 시작 방식과 요청 방식을 Agent에게 알려주기만 하면, 이후 현장 수집, replay, debug가 Agent에 의해 연속적으로 이어질 수 있다.

## 0x1. Motivation

아이디어는 매우 직접적이다.

- 먼저 읽기 전용 bundle을 수집하고, 처음부터 profiler를 켜지 않는다.
- 먼저 문제를 유발하는 request 또는 crash dump를 남기며, 직접 prompt를 짜서 현장을 추측하지 않는다.
- 먼저 replay한다. 같은 현상이 깨끗한 대상에서 재현될 수 있어야 뒤의 trace, torch profile, cuda coredump, `git bisect run`이 의미를 가진다.
- 바퀴를 다시 만들지 않는다. 뒤에서는 여전히 기존의 `debug-cuda-crash`, `debug-distributed-hang`, `sglang-torch-profiler-analysis` 등 SGLang에 이미 있는 전용 SKILL로 연결된다.

이 SKILL은 Ref 안에서 세 가지 예시를 제공한다.

- TTFT 이상, 하지만 queue time은 매우 낮음
- 요청 경로 안의 CUDA Crash(실제 production 장면에서 발생한 문제)
- 다중 카드 TP 통신 Hang

## 0x2. Examples

이 SKILL이 올바르게 트리거하는 문제 해결 흐름을 주로 보여준다. 더 세부적인 정보는 붙이지 않았다. Codex와 GPT5.4 xHigh로 H200에서 재현하면 실제로 각 bug가 발생한 구체적인 코드 위치를 찾을 수 있다.

### 0x2.1 TTFT 이상, 하지만 queue가 주원인은 아님

![TTFT](img/sglang-skills-serving-skills-performance-cuda-crash-hang-f37a3a10/001.svg)

먼저 현장 신호를 보고, 선입견을 갖지 말자.

```text
Health: /health=ok /health_generate=ok
Point-in-time load: running=1 waiting=0 total=1 token_usage=0.410 throughput=29.800 cache_hit_rate=0.970
Metrics: requests=2 prompt_tokens=1540 generation_tokens=128 avg_ttft_s=3.210 avg_e2e_s=4.150 avg_queue_s=0.030
Stage Averages (max across TP ranks): prefill_forward=2.900s, request_process=0.090s
```

<mark>waiting=0</mark>
<mark>avg_queue_s=0.030</mark>
<mark>prefill_forward=2.900s</mark>

이 단계만으로도 충분히 설명된다.

- queue pressure가 밀어붙이는 것이 아니다.
- prefill-side compute 또는 request path slowdown에 더 가깝다.

표준 경로는 다음과 같다.

```text
baseline bundle
  -> save the slow request
  -> replay the same request
  -> trace / torch profile
```

### 0x2.2 CUDA Crash: 죽은 kernel이 반드시 root cause kernel은 아니다

![Crash](img/sglang-skills-serving-skills-performance-cuda-crash-hang-f37a3a10/002.svg)

이 예시는 고립된 kernel 소형 demo를 작성하는 것이 아니라, 일부러 상류 routing kernel 안에 더러운 데이터를 쓰게 해서 하류 `moe_align_block_size_kernel`이 실제로 죽도록 만든 것이다. 실제 production 예시에서 왔다. https://zhuanlan.zhihu.com/p/1984750078074839122 

마지막에 보게 되는 현장 출력은 더 이런 모습에 가깝다.

```text
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
Dumped 1 finished and 1 unfinished requests before crash to /tmp/.../crash_dump_2026-04-20_14-23-15.pkl
CUDA Exception: Warp Out-of-range Address
#0 0x7f7fe1dfac00 _Z27moe_align_block_size_kernelIiEvPKT_PiS3_S3_iimS3_bii
```

<mark>faulting kernel = moe_align_block_size_kernel</mark>
<mark>root cause kernel = topkGatingSoftmax</mark>

전체 흐름은 다음과 같다.

- crash dump가 먼저 실제 request mix를 남긴다.
- replay가 crash를 다시 터뜨릴 수 있다.
- cuda-gdb는 하류 faulting kernel을 가리킨다.
- 하지만 진짜 더러운 데이터를 쓴 것은 앞의 routing kernel이다.

표준 경로는 다음과 같다.

```text
crash dump
  -> summarize dump
  -> replay
  -> CUDA coredump
  -> cuda-gdb
  -> walk one kernel upstream
```

### 0x2.3 통신 Hang: 먼저 사고를 replay하고, 그다음 distributed hang을 깊게 파고든다

![Hang](img/sglang-skills-serving-skills-performance-cuda-crash-hang-f37a3a10/003.svg)

이 예시는 실제 request 경로 안의 TP collective mismatch가 서비스를 멈추도록 일부러 만든 것이다. trigger request는 매우 간단하다.

```text
"hello " * 768
prompt_tokens = 769
```

서비스 로그에서는 먼저 다음을 볼 수 있다.

```text
Prefill batch, #new-seq: 1, #new-token: 769, #cached-token: 0
```

재현 중에 현장을 다시 수집하면 bundle이 나빠지기 시작한다.

```text
health.txt.error.json:
  TimeoutError: timed out

health_generate.txt.error.json:
  TimeoutError: timed out

loads_all.json:
  ConnectionResetError: [Errno 104] Connection reset by peer

loads_core_queues_disagg.json:
  URLError: <urlopen error [Errno 111] Connection refused>
```

그 뒤 watchdog / py-spy는 이런 경로로 떨어진다.

```text
cuEventSynchronize
cudaEventSynchronize
synchronize (torch/cuda/streams.py:231)
process_batch_result_prefill
process_batch_result
event_loop_overlap
```

<mark>서비스는 baseline 때 건강했다</mark>
<mark>같은 trigger request를 replay할 수 있다</mark>
<mark>control plane은 replay로 멈춘 뒤에야 악화되기 시작한다</mark>

표준 경로는 다음과 같다.

```text
baseline bundle
  -> save the trigger request
  -> replay on a clean target
  -> collect replay-time bundle
  -> watchdog / py-spy
  -> debug-distributed-hang
```

## 0x3. 요약

이 SKILL이 진짜로 유용한 지점은 "kernel 하나를 찾아주는 것"이 아니라, 실제 serving의 느림, crash, hang을 다음과 같이 통합하는 것이다.

**먼저 현장을 수집하고, 먼저 replay한 뒤, 이후 trace, 성능 profile, cuda coredump로 들어갈지 아니면 바로 bisect할지 결정한다.**

