# dLLM 추론·서빙 시스템 — 이론에서 커널·엔진까지 (with OSS 기여 타깃)

> 자매 문서: *Diffusion 기초 from Scratch to dLLM*(알고리즘/이론 축). 본 문서는 그 **§8(샘플링·추론)을 시스템·커널 축으로 확장**한다. 이론 문서의 결론 — *dLLM은 결국 "마스크 위치에 대한 가중 MLM cross-entropy"를 학습하고, 추론은 "반복적 언마스킹(remasking)"으로 그 reverse 과정을 푼다* — 을 출발점으로 삼는다.

---

## 0. 한 줄 문제 정의

dLLM이 ARM 대비 *원리적으로* 빠를 수 있는 이유는 한 스텝에 여러 토큰을 병렬로 확정하기 때문이다. 그런데 **오픈소스 dLLM의 실측 추론 속도는 오히려 ARM보다 느렸다.** <cite index="62-1">개방된 Diffusion LLM의 실제 추론 속도가 autoregressive model에 뒤처지는 원인은 (1) KV Cache의 부재와 (2) 여러 토큰을 동시에 디코딩할 때의 품질 저하다.</cite> 이 두 병목이 본 문서의 두 축(§2, §3)이며, 나머지는 그 위에 쌓이는 샘플링 설계(§4), speculative 변주(§5), 엔진 통합(§6), 기여 타깃(§7)이다.

---

## 1. dLLM 추론의 계산 모델 — AR과 무엇이 다른가

생성 지연을 두 항으로 분해한다.

$$
\text{Latency}\ \approx\ \underbrace{N_{\text{step}}}_{\text{스텝 수}}\ \times\ \underbrace{C_{\text{step}}}_{\text{스텝당 비용}}
$$

| | Autoregressive (ARM) | Diffusion (dLLM, 순수 MDM) |
|---|---|---|
| $N_{\text{step}}$ | 출력 길이 $L$ (토큰당 1회) | diffusion 스텝 $T'$ ( $\ll L$ 가능, 병렬 확정 시) |
| $C_{\text{step}}$ | 토큰 1개 forward, **KV cache로 $O(L)$** | **전 시퀀스 forward, 캐시 없으면 $O(L^2)$** |
| 핵심 메트릭 | tokens/s | **NFE**(number of function evaluations) $= T'$ |

**핵심 트레이드오프**: dLLM은 *스텝 수를 줄이는 대신 스텝을 무겁게* 만든다. 따라서 dLLM이 실제로 빠르려면 **두 조건이 동시에** 필요하다 — (a) $N_{\text{step}}$ 을 줄이는 공격적 병렬 디코딩(§3), (b) $C_{\text{step}}$ 을 줄이는 캐싱(§2). 둘 중 하나만으로는 ARM을 못 이긴다. 시스템 연구가 전부 이 두 항을 공격하는 이유다.

---

## 2. 병목 1 — KV Cache 비호환성 (양방향 attention)

### 2.1 왜 표준 KV cache가 깨지는가 (정형화)

ARM은 causal mask를 쓴다. 위치 $i$ 의 $\mathbf K_i,\mathbf V_i$ 는 $x_{\le i}$ 가 확정되면 **불변**이므로, 한 번 계산해 캐시하고 모든 후속 스텝에서 재사용한다($C_{\text{step}}=O(L)$).

dLLM(순수 MDM)은 **양방향(full) attention**이고 매 denoising 스텝마다 마스크 집합이 바뀌어 입력 전체 $\mathbf x_t\to\mathbf x_{t-1}$ 가 갱신된다. 위치 $i$ 의 출력은 미래 위치를 포함한 전 위치에 의존하고, 그 위치들의 토큰이 스텝마다 바뀌므로

$$
\mathbf K_i^{(t-1)},\mathbf V_i^{(t-1)}\ \ne\ \mathbf K_i^{(t)},\mathbf V_i^{(t)}\quad\text{(원칙적으로 모든 }i\text{에서)}.
$$

즉 **모든 K,V를 매 스텝 재계산**해야 한다($C_{\text{step}}=O(L^2)$). <cite index="81-1">양방향 attention 메커니즘 때문에 KV caching 같은 전통적 ARM 가속 기법이 dLLM과 호환되지 않는다.</cite> 이것이 §1 표의 "캐시 없으면 $O(L^2)$"의 근원이다.

### 2.2 탈출구: cross-step KV 유사성이라는 경험적 사실

근사 캐싱의 토대가 되는 관찰: <cite index="79-1">양방향 attention이 직접 캐싱을 막지만, 연속한 디코딩 스텝 간 KV 상태가 거의 동일하다는 점을 이용하면 출력 품질 저하 없이 디코딩을 가속할 수 있다.</cite> 즉 인접 스텝 간 $\mathbf K^{(t)}\!\approx\!\mathbf K^{(t-1)}$ 이므로 "대부분 재사용 + 일부만 갱신"이 가능하다. 해법은 두 갈래다.

### 2.3 해법 A — 근사 캐싱 (training-free)

| 방법 | 핵심 메커니즘 | 비고 |
|---|---|---|
| **Fast-dLLM** (NVIDIA, 2505.22618) | <cite index="60-1">양방향 diffusion에 맞춘 block-wise approximate KV Cache로 성능 저하 거의 없이 캐시 재사용</cite>. **DualCache**: <cite index="64-1">prefix·suffix 토큰을 모두 캐싱, 인접 스텝 간 높은 코사인 유사도를 활용</cite> | block 단위, GSM8K 등에서 수십 배 가속 |
| **dLLM-Cache** (ICML 2026, 2506.06295) | <cite index="80-1">정적 prompt는 long-interval로, 동적 response는 인접 스텝 간 크게 변하는 소수 토큰만 선택 갱신</cite>. **V-verify**: <cite index="80-1">현재·캐시된 V 벡터의 코사인 유사도로 가장 많이 변한 토큰을 식별해 그 토큰만 전체 재계산</cite>. KV뿐 아니라 AttnOut·FFNOut도 캐싱 | LLaDA 8B·Dream 7B에서 최대 9.1× |
| **dKV-Cache** (2505) | <cite index="77-1">one-step delayed KV caching — 디코딩된 토큰을 현재 스텝이 아닌 다음 스텝에 저장</cite> | refresh 메커니즘 결합 |
| **FreeCache** | 마스크 토큰이 먼저 unmask된 토큰에 미치는 기여가 급감함을 이용 | |
| **d2Cache** (2509.23094) | <cite index="77-1">two-stage fine-grained 선택으로 갱신 토큰을 식별하고 나머지는 재사용하는 training-free 근사 KV cache</cite> | |
| **Sparse-dLLM** (2508.02558) | 동적 cache eviction | 메모리 측면 |

> dLLM-Cache의 설계 원리(prompt/response 비대칭)는 **prefix caching의 dLLM판**으로 읽을 수 있다 — Jay의 LMCache connector-기반 offloading 논의와 직접 연결된다(§7).

### 2.4 해법 B — 구조적 (cache-aware, Block Diffusion)

근사 없이 **정확한** KV cache를 되살리는 길은 구조를 바꾸는 것이다. Block Diffusion(BD3-LM)은 블록 간 causal, 블록 내 양방향이므로, *이미 확정된 이전 블록의 K,V는 불변* → 표준 KV cache가 정확히 성립한다(자매 문서 §6, A.7). 대가는 cache-aware training 또는 AR→block 파인튜닝(SDAR, Fast-dLLM v2). LLaDA 2.0이 이 노선을 100B로 스케일했다.

---

## 3. 병목 2 — 병렬 디코딩의 품질 저하 (조건부 독립 가정)

### 3.1 문제의 근원 (정형화)

한 스텝에 여러 마스크 위치를 동시에 확정할 때, 모델의 단일 forward는 **위치별 주변분포** $p_\theta(x_i\mid\mathbf x_t)$ 만 준다. 위치 $i,j$ 를 같이 뽑으면 우리는 곱분포 $p_\theta(x_i\mid\mathbf x_t)\,p_\theta(x_j\mid\mathbf x_t)$ 에서 샘플링하는데, 참 결합 $p(x_i,x_j\mid\mathbf x_t)$ 와 다르다. <cite index="61-1">상호의존적 토큰들을 조건부 독립 가정 하에 동시 샘플링하면 핵심 토큰 의존성이 파괴되어 품질이 저하된다</cite>는 것이 근본 원인이다.

### 3.2 정리 (병렬 언마스킹의 오차 = 조건부 상호정보)

**Proposition 1.** 두 위치 $i,j$ 를 곱분포로 동시 확정할 때, 참 결합과의 괴리는 조건부 상호정보로 정확히 주어진다:

$$
D_{\mathrm{KL}}\big(p(x_i,x_j\mid\mathbf x_t)\,\big\|\,p(x_i\mid\mathbf x_t)\,p(x_j\mid\mathbf x_t)\big)=I(x_i;x_j\mid\mathbf x_t),
$$

그리고 이는 각 위치의 조건부 엔트로피로 상계된다:

$$
I(x_i;x_j\mid\mathbf x_t)\ \le\ \min\big(H(x_i\mid\mathbf x_t),\,H(x_j\mid\mathbf x_t)\big).
$$

**증명.** 첫 등식은 상호정보의 정의 $I=D_{\mathrm{KL}}(p(x_i,x_j)\|p(x_i)p(x_j))$ 그 자체. 둘째는 $I(x_i;x_j)=H(x_j)-H(x_j\mid x_i)\le H(x_j)$ 이고 대칭으로 $\le H(x_i)$ 이므로 최소값으로 상계. $\square$

**함의(confidence-aware decoding의 근거).** 어떤 토큰이 **고신뢰 = 저엔트로피**($H(x_j\mid\mathbf x_t)\approx0$)이면 $I(x_i;x_j\mid\mathbf x_t)\approx0$ — 즉 그 토큰을 병렬 확정해도 의존성 위반 오차가 무시할 만하다. 반대로 두 토큰이 모두 불확실하면 함께 뽑는 것이 위험하다. 이것이 Fast-dLLM 전략의 정보이론적 정당화다: <cite index="67-1">고정된 개수를 뽑는 대신, 신뢰도가 전역 임계값을 넘는 토큰만 동적으로 선택해 안전하게 병렬 디코딩</cite>한다.

### 3.3 실측 효과

confidence threshold만으로도 큰 효과가 보고된다. <cite index="64-1">GSM8K 8-shot·생성길이 1024에서 baseline 대비 27.6× 가속, 정확도 76.0%</cite> 수준이며, <cite index="64-1">모든 과제·모델에서 정확도가 baseline의 1–2점 이내로 유지</cite>되었다. 캐싱(§2)과 병렬 디코딩(§3)은 직교적이라 곱해진다.

---

## 4. 샘플링 스케줄 설계공간

KV cache·병렬 디코딩 위에, *언제·어디서 unmask할지*를 정하는 정책 층이 있다.

- **Remasking 정책**: low-confidence remasking(LLaDA — 확신 높은 토큰만 확정, 나머지 remask), random, semi-AR(블록 단위 좌→우).
- **SlowFast Sampling**: <cite index="76-1">탐색(exploratory)·가속(accelerated) 단계를 동적으로 번갈아 가는 전략으로, certainty·convergence·positional 세 원칙으로 언제·어디서 토큰을 확정할지 통제</cite>하며 dLLM-Cache와 결합 가능.
- **WINO** (draft-and-verify): 공격적으로 다중 토큰을 draft하면서 양방향 컨텍스트로 의심 토큰을 verify·remask.
- **Set Block Decoding**: 표준 next-token prediction(NTP)을 diffusion과 통합.

이들은 모두 §1의 **throughput–latency–quality 삼각 트레이드오프** 위의 서로 다른 운용점이다. NFE를 줄일수록(공격적 병렬) 품질 위험이 커지고, verify/remask가 그 위험을 되사오는 구조.

---

## 5. Speculative Decoding의 dLLM 변주 (Jay의 EAGLE/MTP 표면과 연결)

**관찰: dLLM의 병렬 언마스킹은 speculative decoding과 동형이다.**

| | AR Speculative Decoding | dLLM Parallel Unmasking |
|---|---|---|
| 제안(draft) | 작은 draft 모델이 $\gamma$ 토큰 제안 | 모델이 여러 마스크 위치를 병렬 예측 |
| 검증(verify) | target이 1회 forward로 병렬 검증 | confidence/verify pass로 수락 토큰 선택 |
| 수락 | rejection sampling (**분포 동등성 보장**) | 임계값/근사 (보통 **품질 근사 허용**) |
| 미수락 처리 | 첫 거절 이후 폐기 + target 1토큰 | 나머지 위치 remask |

핵심 차이: AR spec decode는 rejection sampling으로 *target 분포와 정확히 동일*함을 보장(자매 문서 관점의 exactness). dLLM 병렬 디코딩은 일반적으로 근사다(§3.2의 MI 오차를 confidence로 통제). 이 둘을 합치는 흐름:

- **Self-speculative decoding for dLLM** (2510.04147): dLLM 자신을 draft·verify로 활용.
- **DFlash (block-diffusion draft model)**: block diffusion으로 draft를 만들어, *분포 동등성에 가까운* 검증을 dLLM에 도입하려는 변주. Jay가 추적 중인 타깃 — EAGLE/MTP의 "경량 draft + verify" 사고를 dLLM으로 옮긴 형태로 읽으면 진입 비용이 낮다.

> 정리하면, Jay의 spec decoding 자산(rejection sampling 수학, tree attention, drafter taxonomy)은 거의 그대로 dLLM 검증 메커니즘 설계에 이식 가능하다. 차이는 "draft가 좌→우 토큰열"이 아니라 "병렬 마스크 예측"이라는 점뿐이다.

---

## 6. 서빙 엔진 통합 — 실제 OSS 기여 표면

여기가 Jay에게 가장 직접적이다. **2026년 상반기 기준 dLLM 서빙은 아직 미성숙**하다. <cite index="72-1">네이티브 vLLM·SGLang dLLM 지원은 2026년 4월 기준 실험적이거나 부재였고, 권장 경로는 모델의 HuggingFace 구현을 직접 쓰는 커스텀 추론 서버였다.</cite> 즉 **프런티어가 열려 있다.**

### 6.1 SGLang — 활성 로드맵 (가장 구체적인 기여 표면)

SGLang은 production-ready dLLM 프레임워크를 명시적 로드맵으로 추진 중이다(**Issue #14199, "Diffusion LLMs 2025 Q4 & 2026 Q1"**). <cite index="68-1">올해 초 LLaDA가 첫 dLLM을 공개했지만 production-ready dLLM 서빙 엔진이 없었고, SGLang에 가장 고성능의 production-ready dLLM 프레임워크를 구현하려 한다.</cite> 핵심 설계는 **기존 메커니즘 재활용**이다: <cite index="74-1">기존 Chunked-Prefill 메커니즘을 활용해 핵심 아키텍처 변경 없이 SGLang 생태계에 통합하고, 기존 추론 최적화를 그대로 상속하며, 사용자가 diffusion 디코딩 알고리즘을 자유롭게 정의·커스터마이즈할 수 있게 한다.</cite> 다만 근본 과제는 <cite index="74-1">SGLang이 현재 autoregressive 계산 패러다임만 지원하고 dLLM의 diffusion 계산 방식에 아직 적응하지 못했다는 점</cite>이다.

진행 중인 구체 PR(로드맵에 나열): <cite index="68-1">초기 block diffusion 프레임워크(#12588), 문서화(#14358), CI(#14723), 초기 cuda graph 지원(#14203)</cite>. 그리고 LLaDA 2.0 day-0 지원이 이미 들어갔다.

### 6.2 vLLM

vLLM은 (위 Spheron 기준) 네이티브 dLLM 지원이 실험적/부재. ARM 가정에 강하게 묶인 부분 — **continuous batching**, **PD(prefill-decode) disaggregation**, **paged KV cache** — 이 dLLM에서 어떻게 재정의돼야 하는지가 열린 문제다. 예: dLLM은 "decode = 고정 길이 블록의 반복 denoising"이라 토큰 단위 continuous batching의 의미가 달라지고, KV cache가 근사라 paging 정책도 다시 봐야 한다.

### 6.3 엔진 레벨에서 깨지는/재정의되는 가정

- **Continuous batching**: AR은 토큰당 1스텝이라 시퀀스별 진행도를 자연스럽게 섞는다. dLLM은 (블록당) 고정 $T'$ 스텝이라 배칭 단위가 "스텝"이 아니라 "블록 denoising 루프"가 된다.
- **PD disaggregation**: prefill/decode 경계가 흐려진다(블록 내부가 작은 prefill+denoise).
- **CUDA graph**: denoising 루프의 정적 형태는 graph capture에 유리하지만, confidence 기반 동적 토큰 선택은 가변 제어흐름을 만든다 — #14203의 난점.
- **Attention backend (flashinfer)**: block diffusion의 마스크 패턴(블록 내 양방향 + 블록 간 causal)은 **전용 attention 커널** 타깃이다. Jay의 MLA/attention 커널 작업이 그대로 적용되는 지점.

---

## 7. 기여 타깃 매핑 (Jay 표면 → 구체 작업)

| 문제(본 문서) | 관련 OSS repo / issue·PR | 필요 스킬 | Jay 기존 자산과의 연결 |
|---|---|---|---|
| block diffusion attention 커널 (블록 내 양방향 + 블록 간 causal mask) | **flashinfer**, SGLang #12588 | CUDA attention 커널, 마스크 스케줄 | MLA/`sm100_mla.hpp`, FA2/FA4, tile shape·split-KV |
| CUDA graph로 denoising 루프 capture (동적 confidence 선택 처리) | **SGLang #14203** | graph capture, 가변 제어흐름 우회 | SGLang 기여 이력, GDN prefill/decode 컴파일 버그 경험 |
| 근사 KV cache / prompt-response 차등 캐싱 엔진 통합 | **SGLang** dLLM 프레임워크, dLLM-Cache·dKV-Cache 이식 | KV cache 관리, offloading | LMCache connector vs vLLM native offloading 딥다이브 |
| dLLM speculative decoding (DFlash·self-spec) | **SGLang/vLLM** spec decode 경로 | rejection sampling, draft·verify, tree attention | EAGLE/MTP, spec decoding 전문성 (직결) |
| dLLM 양자화 서빙 (W4A16/FP8 in diffusion decode) | **llm-compressor**, flashinfer GEMM | dtype 전략, arch-coupled 양자화 | W4A16/W8A8/FP8/FP4, flashinfer #3438(fp4×bf16) |
| dLLM 서빙 정확성·CI·문서 | **SGLang #14358/#14723** | 벤치 하니스, 회귀 테스트 | onnxruntime#28409식 정밀 버그·벤치 경험 |

**추천 진입점(난이도·레버리지 균형)**: ① flashinfer block-diffusion attention 커널 — 가장 강점이 직결되고 측정 가능한 성능 기여, ② SGLang #14203 cuda graph — 엔진 기여 이력 + 컴파일 버그 경험 활용, ③ DFlash/self-spec 검증 메커니즘 — spec decoding 전문성을 신생 영역에 선점. 셋 다 "측정 가능한 latency/quality 개선 + 명확한 upstream 경로"라는 Jay의 readiness 기준에 부합한다.

---

## 8. 정리 + 읽기 순서

**한 문장 요약**: dLLM 추론은 *스텝 수(병렬 디코딩)* 와 *스텝 비용(캐싱)* 두 항을 동시에 줄여야 ARM을 이기며, 그 과정에서 (1) 양방향 attention의 KV cache 비호환을 근사·구조로 풀고, (2) 병렬 확정의 조건부 상호정보 오차를 confidence로 통제한다. 두 문제 모두 **현재 진행형 OSS 표면**(특히 SGLang)이라 기여 여지가 크다.

**읽기 순서**
1. **자매 문서 §6, 부록 A.5/A.7** — block diffusion·masked diffusion 손실(추론 의미의 토대).
2. **Fast-dLLM** (2505.22618) — KV cache + confidence parallel decoding, 두 병목을 한 논문에서.
3. **dLLM-Cache** (2506.06295) — prompt/response 차등 캐싱, V-verify.
4. **dKV-Cache / d2Cache / Sparse-dLLM** (2505 / 2509.23094 / 2508.02558) — 근사 캐싱 설계공간.
5. **SGLang Issue #14199 + LMSYS blog(2025-12-19)** — 엔진 통합 실제 설계(Chunked-Prefill 재활용).
6. **self-speculative decoding** (2510.04147) + **DFlash** — spec decoding 변주.
7. **Fast-dLLM v2 / SDAR / Eso-LM** (2509.26328 / 2506.01928) — cache-aware 구조적 노선.

---

### 부록: 추론 메트릭 cheat-sheet

| 기호 | 의미 | dLLM에서의 함의 |
|---|---|---|
| $N_{\text{step}}=T'$ | NFE, diffusion 스텝 수 | 병렬 디코딩으로 $\downarrow$, 품질과 trade |
| $C_{\text{step}}$ | 스텝당 forward 비용 | 캐시 없으면 $O(L^2)$, 근사 캐시로 $\to$ 부분 재계산 |
| $\rho$ | (dLLM-Cache) 적응 갱신 비율 | response 토큰 중 재계산 비율 |
| $K_p,K_r$ | prompt·response refresh 간격 | $K_p\gg K_r$ (prompt가 정적) |
| $I(x_i;x_j\mid\mathbf x_t)$ | 두 위치 조건부 상호정보 | 병렬 확정 오차의 정확한 척도(§3.2) |
| confidence threshold | 병렬 확정 기준(저엔트로피) | $I$ 를 작게 유지하는 안전장치 |

*OSS 이슈·PR 번호와 모델 버전(LLaDA 2.0 등)은 빠르게 갱신된다. 기여 착수 전 해당 repo의 최신 이슈/로드맵을 직접 확인할 것 — 특히 SGLang #14199는 활성 트래커다.*
