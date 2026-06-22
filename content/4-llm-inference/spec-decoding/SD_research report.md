# Speculative Decoding과 EAGLE 알고리즘 introduction.

**Speculative Decoding은 LLM 추론의 메모리 바운드 병목을 역이용하여, 작은 드래프트 모델이 후보 토큰을 생성하고 타겟 모델이 이를 병렬 검증하는 "draft-then-verify" 패러다임으로 출력 분포를 수학적으로 보존하면서 2~6.5배 속도 향상을 달성하는 기법이다.** EAGLE 시리즈는 이 패러다임의 최전선에서, feature-level autoregression(EAGLE-1), 동적 드래프트 트리(EAGLE-2), training-time test를 통한 스케일링(EAGLE-3)으로 진화하며 현재 가장 높은 성능을 기록하고 있다. 본 보고서는 수학적 증명부터 PyTorch 구현까지, speculative decoding과 EAGLE 1/2/3를 이해하는 데 필요한 기술적 세부사항을 다룬다.

---

## Autoregressive decoding이 느린 근본적 이유

대규모 언어 모델(LLM)의 추론은 두 단계로 구성된다. **Prefill 단계**에서는 전체 입력 프롬프트를 병렬 처리하며 행렬-행렬 곱셈(GEMM)을 수행하므로 GPU 연산 유닛을 효율적으로 활용한다. 반면 **Decode 단계**에서는 토큰을 하나씩 순차적으로 생성하며, 매 토큰마다 수십~수백 GB에 달하는 전체 모델 가중치를 GPU HBM에서 연산 유닛으로 로드해야 한다. 그러나 실제 수행하는 연산은 단일 토큰에 대한 행렬-벡터 곱셈뿐이다.

이것이 바로 **메모리 바운드(memory-bound)** 문제의 핵심이다. NVIDIA H100 SXM5는 약 **990 TFLOPS**(FP16)의 연산 성능과 **3.35 TB/s**의 메모리 대역폭을 제공하지만, 단일 토큰 디코딩의 산술 강도(arithmetic intensity)는 대역폭 대비 연산 성능의 균형점(약 591 FLOPs/byte)에 한참 미치지 못한다. 결과적으로 디코딩 시 Model FLOPs Utilization(MFU)은 **10% 미만**에 그치며, GPU 연산 코어는 대부분의 시간을 데이터 대기에 소비한다.

**KV 캐시**는 이전 토큰의 Key-Value 행렬을 저장하여 재계산을 방지하지만, 시퀀스 길이에 비례하여 선형 증가하며 긴 컨텍스트에서는 GPU 메모리의 주요 소비원이 된다. 매 디코딩 스텝마다 전체 KV 캐시와 모델 가중치를 읽어야 하지만 생산하는 것은 토큰 하나뿐이라는 근본적인 비효율이 존재한다.

바로 이 유휴 연산 용량(idle compute capacity)이 speculative decoding의 기회가 된다. 메모리 대역폭에 의해 병목이 걸린 상태에서, 여유 연산 자원으로 여러 후보 토큰을 병렬 처리해도 wall-clock 시간이 거의 증가하지 않기 때문이다.

---

## Draft-then-verify: Speculative decoding의 핵심 패러다임

Speculative decoding은 2022년 말 두 연구 그룹에 의해 독립적으로 제안되었다. Leviathan et al.(Google, arXiv:2211.17192, ICML 2023)은 T5-XXL(11B)과 LaMDA(137B)에서 2~3배 속도 향상을 보였고, Chen et al.(DeepMind, arXiv:2302.01318, 2023)은 Chinchilla(70B)에서 2~2.5배 속도 향상을 달성했다.

타겟 모델 $M_p$의 분포를 $p(x_t | x_{<t})$, 작고 빠른 드래프트 모델 $M_q$의 분포를 $q(x_t | x_{<t})$라 하면, speculative decoding은 매 이터레이션마다 세 단계를 수행한다.

**1단계 — 드래프트(Draft):** $M_q$가 $\gamma$개의 후보 토큰 $x_1, x_2, \ldots, x_\gamma$를 autoregressive하게 생성한다. $M_q$가 작으므로 이 과정은 빠르다.

**2단계 — 검증(Verify):** $M_p$가 $\gamma$개의 드래프트 토큰을 **단일 forward pass**로 병렬 처리하여 각 위치의 확률 분포 $p_1(x), p_2(x), \ldots, p_{\gamma+1}(x)$를 동시에 계산한다. 트랜스포머의 causal 구조 덕분에 $\gamma$개 토큰을 병렬 스코어링하는 지연 시간은 단일 토큰 생성과 유사하다.

**3단계 — 수락/거부(Accept/Reject):** 왼쪽에서 오른쪽으로 modified rejection sampling을 적용한다. 드래프트 토큰이 $p(x)$와 일치하면 수락하고, 첫 번째 거부 시점에서 보정 분포에서 재샘플링한 후 중단한다. 모든 $\gamma$개 토큰이 수락되면 $p_{\gamma+1}(x)$에서 추가 토큰 하나를 샘플링한다.

이 메커니즘의 핵심 보장은 **lossless acceleration** — 모델 아키텍처나 학습을 변경하지 않으면서 출력 분포가 타겟 모델의 것과 정확히 동일하다는 점이다.

---

## Acceptance/rejection sampling의 수학적 증명

### 수락 기준과 보정 분포

주어진 프리픽스에서 타겟 모델의 다음 토큰 분포를 $p(x)$, 드래프트 모델의 분포를 $q(x)$라 하자. 드래프트 토큰 $\tilde{x} \sim q(x)$에 대한 수락 확률은 다음과 같다:

$$\alpha(\tilde{x}) = \min\left(1, \frac{p(\tilde{x})}{q(\tilde{x})}\right)$$

구체적으로, $q(\tilde{x}) \leq p(\tilde{x})$이면 항상 수락하고(드래프트 모델이 과소평가), $q(\tilde{x}) > p(\tilde{x})$이면 확률 $p(\tilde{x})/q(\tilde{x})$로 수락한다. 구현에서는 $r \sim U(0,1)$을 샘플링하여 $r \leq p(\tilde{x})/q(\tilde{x})$이면 수락한다.

토큰이 **거부**될 때는 다음의 **잔차(residual) 분포**에서 재샘플링한다:

$$p'(x) = \frac{\max(0,\; p(x) - q(x))}{\sum_{x'} \max(0,\; p(x') - q(x'))}$$

이 분포는 $q$가 커버하지 못한 $p$의 확률 질량을 포착한다.

### 정확성 증명 (Lossless 보장)

**정리 (Leviathan et al., 2023):** 임의의 분포 $p(x)$와 $q(x)$에 대해, speculative sampling으로 생성된 토큰은 정확히 $p(x)$를 따른다.

**증명:** 특정 토큰 값 $x'$의 출력 확률은 두 경로의 합이다:

$$P(\text{output} = x') = \underbrace{q(x') \cdot \alpha(x')}_{\text{드래프트에서 수락}} + \underbrace{\beta \cdot p'(x')}_{\text{거부 후 재샘플링}}$$

**1단계 — 수락 기여:**

$$q(x') \cdot \alpha(x') = q(x') \cdot \min\left(1, \frac{p(x')}{q(x')}\right) = \min(q(x'), p(x'))$$

**2단계 — 전체 거부율 $\beta$:**

$$\beta = 1 - \sum_x q(x)\alpha(x) = 1 - \sum_x \min(p(x), q(x)) = \sum_x \max(0, p(x) - q(x))$$

마지막 등호는 $\sum_x p(x) = 1$이고 $\min(a,b) = a - \max(0, a-b)$이므로 성립한다.

**3단계 — 재샘플링 기여:**

$$\beta \cdot p'(x') = \cancel{\sum_x \max(0, p(x) - q(x))} \cdot \frac{\max(0, p(x') - q(x'))}{\cancel{\sum_x \max(0, p(x) - q(x))}} = \max(0, p(x') - q(x'))$$

$\beta$가 잔차 분포의 정규화 상수와 정확히 상쇄된다.

**4단계 — 결합:**

$$P(\text{output} = x') = \min(p(x'), q(x')) + \max(0, p(x') - q(x'))$$

임의의 비음수 $a, b$에 대해 $\min(a,b) + \max(0, a-b) = a$이므로:

$$\boxed{P(\text{output} = x') = p(x')} \quad \blacksquare$$

### 토큰 수락률과 기대 생성 토큰 수

**전체 수락률** $\alpha$는 드래프트 토큰이 수락될 확률의 기댓값이다:

$$\alpha = \sum_x \min(p(x), q(x)) = 1 - D_{LK}(p, q)$$

여기서 $D_{LK}$는 분포 간 유사도를 측정하는 지표로, $p$와 $q$가 유사할수록 $\alpha$가 증가한다.

i.i.d. 가정 하에서, 한 이터레이션에서 생성되는 **기대 토큰 수**는 cap이 $\gamma$인 기하 변수를 따른다:

$$E[\text{생성 토큰 수}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

**유도:** 첫 거부 전 수락된 토큰 수 $N \in \{0, 1, \ldots, \gamma\}$에 대해 $P(N \geq k) = \alpha^k$ ($k \leq \gamma$)이다. 총 생성 토큰은 $N + 1$(수락 토큰 + 재샘플링/보너스 토큰)이므로:

$$E[N+1] = 1 + \sum_{k=1}^{\gamma} \alpha^k = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

| $\alpha$ | $\gamma$ | 기대 토큰 수 | 속도 향상 |
|----------|----------|-------------|----------|
| 0.7 | 3 | 2.53 | 2.53× |
| 0.8 | 5 | 3.69 | 3.69× |
| 0.9 | 10 | 6.86 | 6.86× |

### Wall-time 속도 향상 공식

비용 계수 $c$를 $M_q$의 1회 실행 시간 대비 $M_p$의 1회 실행 시간 비율이라 하면:

$$\text{Speedup} = \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma c + 1)}$$

Speculative decoding이 유익한 조건은 **$\alpha > c$**이다. 실제로 드래프트 모델이 타겟 모델보다 훨씬 작을 때 $c < 0.05$이므로, 적당한 $\alpha$ 값으로도 상당한 속도 향상이 가능하다.

### Tree-based verification의 수학적 근거

기본 speculative decoding은 단일 선형 체인을 생성하므로, 초기 토큰이 거부되면 이후 모든 토큰이 낭비된다. **트리 기반 방법**은 여러 후보 시퀀스를 트리 구조로 조직화하여 이 문제를 해결한다.

폭(width) $w$의 트리에서, 각 레벨에서 최소 하나의 자식이 수락될 확률은 선형 체인의 $\alpha$에서 $1 - (1-\alpha)^w$로 증가한다. 예를 들어, $\alpha = 0.5$이고 $w = 3$이면 선형 체인의 수락 확률 0.5가 트리에서는 $1 - 0.5^3 = 0.875$로 상승한다. SpecInfer의 실험 데이터에 따르면, 트리 기반 검증은 스텝당 토큰 수락률을 52~57%에서 **96~97%**로 향상시켰다.

트리 어텐션은 **위상 인식 인과 마스크(topology-aware causal mask)**를 사용한다. 각 토큰은 자신과 트리상의 **조상 노드에만** 어텐션할 수 있으며, 형제나 사촌 노드에는 어텐션하지 않는다. 마스크 행렬 $M$은 다음과 같이 정의된다: 노드 $j$가 노드 $i$의 조상이거나 $j = i$이면 $M_{ij} = 1$, 그렇지 않으면 $M_{ij} = 0$.

---

## EAGLE-1: Feature-level autoregression이라는 발상의 전환

EAGLE(Extrapolation Algorithm for Greater Language-model Efficiency)은 Li et al.(Peking University/Microsoft Research)이 제안하여 **ICML 2024**에 발표한 speculative decoding 방법이다. 핵심 통찰은 토큰이 아닌 **feature(은닉 상태)**를 예측하면 드래프트 정확도가 극적으로 향상된다는 것이다.

### "Feature는 토큰보다 예측하기 쉽다"는 제1 원리

EAGLE의 제1 원리(First Principle)는 다음과 같다: **"Feature 시퀀스는 압축 가능(compressible)하여, 이전 feature로부터 다음 feature를 예측하기 쉽다."** 여기서 "feature"란 타겟 LLM의 **최상위에서 두 번째 레이어의 은닉 상태** — LM 헤드 직전의 활성화를 의미한다.

토큰 시퀀스는 자연어의 높은 변동성과 불규칙성을 그대로 반영하여 작은 모델이 직접 예측하기 극히 어렵다. 반면 **feature 시퀀스**는 전체 LLM이 계산한 풍부한 의미 정보를 인코딩하므로 본질적으로 더 규칙적이고 예측 가능하다. 실제 ablation 실험에서 feature 기반 예측은 **1.9배** 속도 향상을, 토큰 기반 예측은 **1.5배**에 그쳤다(Vicuna 7B, MT-bench).

### Sampling uncertainty 해소: Shifted token의 도입

두 번째 핵심 관찰은 **sampling uncertainty**에 관한 것이다. 토큰 "I" 다음에 LLM이 "am" 또는 "always"를 샘플링할 수 있는데, "I am"과 "I always" 이후의 feature는 완전히 다르다. Medusa처럼 "I"의 feature만으로 다음 feature를 예측하면, 동일 입력이 상이한 타겟에 매핑되는 해결 불가능한 모호성에 직면한다.

EAGLE의 해법은 **한 타임스텝 앞선 토큰 시퀀스(shifted token)**를 입력에 포함시키는 것이다. 실제 샘플링된 토큰(예: "am")의 임베딩을 함께 제공하면 입력-출력 매핑이 결정적(deterministic)이 된다. 이 기법으로 속도 향상이 1.9배에서 **2.8배**로 도약했다.

### 드래프트 헤드 아키텍처

EAGLE의 드래프트 모델은 세 모듈로 구성된다:

- **임베딩 레이어** — 타겟 LLM에서 가져온 **동결(frozen)** 파라미터
- **FC(Fully Connected) 레이어** — `2×hidden_dim`을 `hidden_dim`으로 축소 (유일한 학습 가능 컴포넌트의 일부)
- **단일 Transformer 디코더 레이어** — 다음 feature를 autoregressive하게 예측 (학습 가능)
- **LM 헤드** — 타겟 LLM에서 가져온 **동결** 파라미터

입력 처리 흐름은 다음과 같다: 타겟 모델의 feature `(bs, seq_len, hidden_dim)`과 shifted token의 임베딩 `(bs, seq_len, hidden_dim)`을 **연결(concatenate)**하여 `(bs, seq_len, 2×hidden_dim)` → FC 레이어로 `hidden_dim` 축소 → 단일 디코더 레이어로 다음 feature 예측 → 동결된 LM 헤드로 토큰 분포 생성.

학습 가능 파라미터는 모델 크기에 따라 **0.24B~0.99B**로, 타겟 모델 대비 매우 작다. 7B 모델에서 0.24B, 70B 모델에서 0.99B(약 1.4%)에 불과하다.

### 학습 방법론

EAGLE의 학습은 두 가지 손실 함수를 결합한다:

$$\mathcal{L} = \mathcal{L}_{\text{reg}} + 0.1 \times \mathcal{L}_{\text{cls}}$$

$\mathcal{L}_{\text{reg}}$은 feature 예측을 위한 Smooth L1 손실이고, $\mathcal{L}_{\text{cls}}$는 토큰 예측을 위한 Cross Entropy 손실이다. 가중치 0.1은 classification loss가 regression loss보다 약 10배 크기 때문에 균형을 맞추기 위함이다.

학습 데이터로는 **ShareGPT**(약 68,000 대화)를 사용하며, 타겟 LLM은 학습 중 **완전히 동결**된다. 학습 비용은 70B 모델 기준 4×A100(40G)에서 1~2일, 7B/13B 모델 기준 단일 RTX 3090에서 1~2일이다. 특히 학습 데이터에 대한 민감도가 낮아, 타겟 LLM 생성 데이터 사용 시 미미한 개선(2.88× vs 2.78×)만 관찰되었다.

오차 누적을 완화하기 위해 학습 시 feature에 $U(-0.1, 0.1)$의 균일 분포 노이즈를 추가하는 **데이터 증강**을 적용한다.

### EAGLE-1의 정적 트리와 성능

EAGLE-1은 **사전 정의된 정적 트리 구조**를 사용한다. 루트에서 4개의 분기로 시작하여, 드래프트 모델이 5번의 forward pass를 통해 약 **60개의 후보 토큰**으로 구성된 트리를 생성한다. 트리 어텐션의 효과는 상당하여, 체인 대비 평균 수락 길이(τ)가 0.6~0.75 증가한다.

**주요 성능 결과 (Temperature=0, MT-bench 기준):**

| 모델 | 속도 향상 | 평균 수락 길이 (τ) |
|------|----------|-------------------|
| Vicuna 7B | 2.72× | 3.94 |
| Vicuna 13B | 3.07× | 3.98 |
| LLaMA2-Chat 7B | 2.78× | 3.62 |
| LLaMA2-Chat 13B | 3.01×~3.76× | 3.90 |
| LLaMA2-Chat 70B | 2.7×~3.5× | 3.81 |

코드 생성(HumanEval)에서 최대 **3.76배**, 수학(GSM8K)에서 **3.25배** 속도 향상을 달성했다. Medusa 대비 **1.5~1.6배**, Lookahead 대비 **1.7~2.1배** 빠르다.

---

## EAGLE-2: 문맥을 읽는 동적 드래프트 트리

EAGLE-2(EMNLP 2024)는 EAGLE-1의 정적 트리 구조가 지닌 근본적 한계를 해결한다. 정적 트리는 수락률이 토큰의 **위치**에만 의존한다고 암묵적으로 가정하지만, 실제 수락률은 **문맥에 따라 크게 변동**한다. "10+2" 다음의 "="는 모호하지만, "10+2=" 다음의 "1"은 거의 확정적이다. EAGLE-1은 이 차이를 무시하고 동일한 분기 수를 할당하여 연산 예산을 낭비한다.

### EAGLE 드래프트 모델의 보정(calibration) 발견

EAGLE-2의 핵심 발견은 **EAGLE의 드래프트 모델이 잘 보정(well-calibrated)되어 있다**는 것이다. 즉, 드래프트 모델의 신뢰도 점수(출력 확률)가 실제 수락률을 정확히 근사한다. 신뢰도 0.05 미만의 토큰은 실제 수락률 약 0.04, 신뢰도 0.95 초과의 토큰은 실제 수락률 약 0.98로, 보정 곡선이 $y = x$ 선을 밀접하게 따른다. 이로써 **추가 연산 없이** 수락률을 추정할 수 있게 된다.

### 노드 "가치(Value)" 메트릭과 동적 트리 구축

EAGLE-2는 각 노드 $t_i$의 **가치(Value)** $V_i$를 루트에서 해당 노드까지 경로상 모든 신뢰도 점수의 **곱**으로 정의한다:

$$V_i = \prod_{t_j \in \text{Path}(\text{root}, t_i)} c_j$$

여기서 $c_j$는 노드 $t_j$의 드래프트 모델 신뢰도(수락률 근사치)이다. Speculative sampling에서 한 토큰이 거부되면 이후 모든 토큰이 폐기되므로, 한 토큰이 궁극적으로 수락되려면 모든 조상이 수락되어야 한다. 따라서 지역적 신뢰도보다 **전역 확률(조상 신뢰도의 곱)**이 더 정보력 있는 메트릭이다.

동적 트리 구축은 **확장(Expansion)**과 **재순위화(Reranking)** 두 단계로 진행된다:

**확장 단계:** 현재 레이어의 모든 노드 중 가치가 가장 높은 상위 $k$개(기본값 10)를 선택하여 드래프트 모델의 단일 forward pass로 자식 노드를 생성한다. 이를 반복하여 트리를 깊게 확장한다. 높은 신뢰도의 토큰은 높은 가치를 가지므로 자연스럽게 깊은 확장을 받고, 불확실한 예측에서는 여러 얕은 노드로 넓게 확장된다.

**재순위화 단계:** 확장 완료 후, 전체 트리의 **모든** 노드를 풀링하여 가치가 가장 높은 상위 $m$개(7B 모델: 60, 13B: 50, 70B: 48)를 전역적으로 선택한다. 가치가 동일하면 **더 얕은** 노드를 우선시하여 선택된 집합이 연결된 트리를 유지하도록 한다. 선택되지 않은 노드는 암묵적으로 가지치기(pruning)된다.

### EAGLE-2의 성능 도약

| 모델 | EAGLE-1 | EAGLE-2 | 개선율 |
|------|---------|---------|--------|
| Vicuna 7B (T=0) | 2.90× | 3.62× | +25% |
| Vicuna 13B (T=0) | 3.07× | 4.26× | +39% |
| LLaMA2-Chat 7B (T=0) | 2.78× | 3.43× | +23% |
| LLaMA2-Chat 13B (T=0) | 3.03× | 4.21× | +39% |
| LLaMA2-Chat 70B (T=0) | 3.01× | 3.51× | +17% |

Temperature=1(비탐욕적 디코딩)에서는 개선이 더욱 극적이다. Vicuna 13B의 경우 2.32×에서 3.80×로 **64%** 향상되었다. 코드 생성에서 최대 **5배** 속도 향상을 기록했다.

결정적으로 EAGLE-2는 **추가 학습이 필요 없다**. EAGLE-1의 드래프트 모델 체크포인트를 그대로 사용하며, 개선은 순수하게 추론 시간의 트리 구축 전략 변경에서 비롯된다.

---

## EAGLE-3: Training-time test로 스케일링의 벽을 넘다

EAGLE-3(NeurIPS 2025 채택)는 초기에 "Training-Free"로 알려졌으나, 실제 논문 제목은 **"EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"**이다. 여전히 드래프트 헤드 학습이 필요하지만, **Training-Time Test(TTT)**라는 혁신적 학습 기법으로 기존 EAGLE의 근본적 한계를 돌파한다.

### EAGLE-1/2의 스케일링 병목

저자들은 학습 데이터를 8배로 늘려도 EAGLE-1/2의 속도 향상이 거의 변하지 않는 **스케일링 포화(saturation)** 현상을 발견했다. 근본 원인은 **feature 예측 제약**이다. 고차원 은닉 상태를 MSE 손실로 회귀하는 것이 모델의 표현력(expressiveness)에 병목으로 작용했던 것이다.

### 두 가지 핵심 혁신

**1) 직접 토큰 예측 (Feature 예측 제거):** EAGLE-1/2는 다음 feature 벡터를 예측한 후 LM 헤드로 토큰 로짓을 얻었다. EAGLE-3는 **feature 예측 손실($\ell_{\text{fea}}$)을 완전히 제거**하고 직접 토큰을 예측한다:

$$\mathcal{L}_{\text{E3}} = -\sum \log q(t_{t+i} \mid g_{1:t}, a_{t+1:t+i-1})$$

고차원 MSE 제약을 없앰으로써 모델의 표현력이 해방된다.

**2) Training-Time Test (TTT) — 추론을 학습 중에 시뮬레이션:** EAGLE-1/2의 치명적 문제는 **학습-추론 분포 불일치(train-test distribution mismatch)**이다. 학습 시에는 항상 타겟 모델의 정확한 feature를 입력받지만, 추론 시에는 첫 스텝 이후 자신의 (부정확한) 예측을 입력으로 사용해야 한다. 이 불일치가 **오차 누적(error accumulation)**을 유발한다.

EAGLE-3의 TTT는 학습 중에 이 추론 과정을 정확히 시뮬레이션한다. 1단계에서 생성한 출력 $\hat{a}_{t+1}$을 2단계의 입력으로 다시 피드백하며, 트리 구조의 커스텀 어텐션 마스크로 의존성을 처리한다. 결과적으로 드래프트 모델이 **자신의 노이즈 출력에 강건(robust)**해지며, 수락률이 드래프트 위치에 걸쳐 거의 일정하게 유지된다(약 70~80%). EAGLE-1/2에서는 위치가 깊어질수록 수락률이 현저히 하락했다.

### 다층 Feature 융합 (Tri-layer Fusion)

EAGLE-1/2는 최상위 레이어의 feature만 사용했다. 이 feature는 다음 토큰 예측에 최적화되어 있어 더 먼 미래의 토큰 예측에 필요한 정보가 부족하다.

EAGLE-3는 **세 레이어의 feature를 융합**한다:

- **저수준 feature ($l$):** 구문, 형태론, 지역적 토큰 컨텍스트 인코딩
- **중간 수준 feature ($m$):** 의미적 관계와 담화 구조 인코딩
- **고수준 feature ($h$):** 출력 확률 분포 인코딩

이 세 $k$차원 벡터를 연결하여 $3k$차원 벡터를 만든 후, FC 레이어로 $k$차원으로 축소한다:

$$g_t = W_{\text{fuse}}[l_t;\; m_t;\; h_t] \in \mathbb{R}^k$$

### 전례 없는 성능과 스케일링

**Vicuna 13B (Temperature=0) 상세 결과:**

| 태스크 | Medusa | EAGLE-1 | EAGLE-2 | EAGLE-3 |
|--------|--------|---------|---------|---------|
| MT-bench | 2.07× | 3.07× | 4.26× | **5.58×** |
| HumanEval | 2.50× | 3.61× | 4.86× | **6.47×** |
| GSM8K | 2.23× | 2.67× | 3.73× | **4.87×** |
| Alpaca | 2.08× | 2.67× | 3.88× | **5.06×** |
| CNN/DM | 1.71× | 3.25× | 4.37× | **5.58×** |
| **평균** | 2.12× | **3.05×** | **4.22×** | **5.51×** |

EAGLE-3는 EAGLE-2 대비 일관되게 약 **1.4배**, EAGLE-1 대비 약 **1.8배** 개선을 달성했다. 최대 **6.47배** 속도 향상(HumanEval, Vicuna 13B)을 기록했다.

가장 혁신적인 발견은 **스케일링 법칙의 출현**이다. 학습 데이터 1×(ShareGPT)에서 8×(ShareGPT + UltraChat-200K)로 증가시키면, EAGLE-3의 수락률과 속도 향상이 거의 **선형적으로** 증가한다. EAGLE-1/2에서는 관찰되지 않던 현상이다.

대규모 배치에서의 **처리량(throughput)** 성능도 주목할 만하다. SGLang + H100에서 배치 64일 때 EAGLE-3는 **+38% 처리량 향상**을 달성한 반면, EAGLE-2는 배치 24를 넘어서면 성능이 하락한다.

---

## PyTorch 기반 EAGLE 구현 해부

### 드래프트 헤드의 forward pass 구현

EAGLE의 핵심 구현은 `SafeAILab/EAGLE` 리포지토리의 `eagle/model/cnets.py`에 있다. 드래프트 헤드(`Model` 클래스)의 구조와 forward pass는 다음과 같다:

```python
class Model(nn.Module):
    def __init__(self, config, bias=True, total_tokens=63, depth=5, top_k=8):
        super().__init__()
        # 타겟 LLM에서 가져온 동결 임베딩 레이어
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        # 유일한 학습 가능 컴포넌트: FC + 단일 디코더 레이어
        self.fc = nn.Linear(
            2 * config.hidden_size, config.hidden_size, bias=bias
        )
        self.midlayer = LlamaDecoderLayer(config, layer_idx=0)
        # 트리 어텐션 초기 마스크
        self.tree_mask_init = torch.eye(self.top_k)[None, None]

    def forward(self, hidden_states, input_ids, attention_mask=None,
                past_key_values=None, use_cache=True):
        # 1. Shifted token의 임베딩 획득
        token_embeds = self.embed_tokens(input_ids)
        # 2. Feature와 토큰 임베딩 연결
        fused = torch.cat([hidden_states, token_embeds], dim=-1)
        # 3. FC 레이어로 차원 축소: 2*hidden → hidden
        hidden = self.fc(fused)
        # 4. 단일 디코더 레이어 (input_layernorm 생략)
        output = self.midlayer(
            hidden, attention_mask=attention_mask,
            past_key_value=past_key_values, use_cache=use_cache
        )
        # 5. 출력 feature → 타겟 모델의 LM 헤드로 토큰 분포 생성
        return output
```

EAGLE-3에서는 다층 feature 융합이 추가된다:

```python
# EAGLE-3: 3개 레이어의 feature 융합
fused_features = torch.cat([low_feat, mid_feat, high_feat], dim=-1)
g = self.fc_fuse(fused_features)  # 3*hidden → hidden
combined = torch.cat([g, token_embed], dim=-1)  # hidden + hidden
```

### 트리 어텐션 마스크 구현

트리 어텐션 마스크는 각 토큰이 자신의 조상 노드에만 어텐션하도록 보장한다:

```python
def generate_tree_attention_mask(tree_indices, num_nodes):
    """
    tree_indices: 각 노드의 부모 인덱스 리스트
    반환: (num_nodes, num_nodes) 형태의 어텐션 마스크
    """
    mask = torch.zeros(num_nodes, num_nodes)
    for node_idx in range(num_nodes):
        current = node_idx
        while current is not None:
            mask[node_idx, current] = 1
            current = parent_of(current)  # 부모로 거슬러 올라감
    return mask
```

예를 들어, 루트 "The"에서 "cat", "dog", "bird"로 분기하고 "cat" 아래에 "sat", "ran"이 있는 트리의 마스크는:

```
         The  cat  dog  bird sat  ran
The       1    0    0    0    0    0
cat       1    1    0    0    0    0
dog       1    0    1    0    0    0
bird      1    0    0    1    0    0
sat       1    1    0    0    1    0
ran       1    1    0    0    0    1
```

### EAGLE-2의 동적 트리 구축 의사코드

```python
def build_dynamic_tree(draft_model, features, total_tokens=60,
                       depth=6, top_k=10):
    tree = [RootNode(features)]

    for step in range(depth):
        current_layer = get_latest_layer(tree)
        # 현재 레이어에서 가치 상위 k개 노드 선택
        top_nodes = sorted(current_layer, 
                          key=lambda n: n.value, reverse=True)[:top_k]
        # 선택된 노드를 단일 forward pass로 확장
        logits = draft_model(top_nodes)
        probs = softmax(logits)
        for node, node_probs in zip(top_nodes, probs):
            for token_id, conf in top_candidates(node_probs):
                child = TreeNode(token_id,
                                value=node.value * conf)
                node.add_child(child)

    # 재순위화: 전체 트리에서 가치 상위 m개 전역 선택
    all_nodes = flatten(tree)
    # 동일 가치 시 얕은 노드 우선
    selected = sorted(all_nodes, key=lambda n: (-n.value, n.depth)
                     )[:total_tokens]
    return build_tree_mask(selected)
```

### 검증 단계의 구현 로직

```python
def verify_and_accept(target_model, draft_tree, tree_mask, kv_cache):
    # 1. 트리를 1차원 시퀀스로 평탄화
    draft_tokens = flatten_tree(draft_tree)
    
    # 2. 타겟 모델의 단일 forward pass (트리 어텐션)
    target_logits = target_model(
        input_ids=draft_tokens,
        attention_mask=tree_mask,
        past_key_values=kv_cache
    )
    target_probs = softmax(target_logits)
    
    # 3. 루트부터 깊이 우선으로 수락/거부
    accepted_path = []
    for node in depth_first_traversal(draft_tree):
        p_target = target_probs[node.pos][node.token]
        p_draft = node.draft_prob
        # Speculative sampling 수락 기준
        if random() < min(1.0, p_target / p_draft):
            accepted_path.append(node.token)
        else:
            # 거부: 보정 분포에서 재샘플링
            adjusted = clamp(target_probs[node.pos] 
                           - draft_probs[node.pos], min=0)
            adjusted = adjusted / adjusted.sum()
            bonus = multinomial(adjusted, 1)
            accepted_path.append(bonus)
            break  # 첫 거부에서 중단
    
    # 모두 수락 시 보너스 토큰 추가
    if len(accepted_path) == len(draft_tree):
        bonus = sample(target_probs[last_pos + 1])
        accepted_path.append(bonus)
    
    return accepted_path
```

### 프레임워크 통합

EAGLE는 현재 주요 추론 프레임워크에 통합되어 있다. **vLLM**에서는 `speculative_config`로 간단히 활성화할 수 있다:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    speculative_config={
        "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 2,
        "method": "eagle3",
    },
)
```

**SGLang**에서는 커맨드라인으로 설정한다:

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --speculative-algo EAGLE \
    --speculative-draft lmsys/sglang-EAGLE-llama2-chat-7B \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16
```

그 외 NVIDIA TensorRT-LLM, AMD ROCm, Intel Extension for Transformers, PaddleNLP, MLC-LLM 등 10개 이상의 프레임워크에서 지원된다.

---

## EAGLE 1 vs 2 vs 3 종합 비교

| 항목 | EAGLE-1 (ICML'24) | EAGLE-2 (EMNLP'24) | EAGLE-3 (NeurIPS'25) |
|------|-------------------|---------------------|----------------------|
| **핵심 혁신** | Feature-level autoregression | 동적 드래프트 트리 | Training-Time Test + 다층 융합 |
| **예측 대상** | Feature 벡터 (은닉 상태) | Feature 벡터 (은닉 상태) | **직접 토큰** (feature 예측 제거) |
| **Feature 소스** | 최상위 레이어 단일 | 최상위 레이어 단일 | **저/중/고 3개 레이어 융합** |
| **트리 구조** | 정적 (사전 정의) | **동적** (신뢰도 기반) | 동적 (EAGLE-2 계승) |
| **학습-추론 정합성** | 불일치 (오차 누적) | 불일치 (오차 누적) | **TTT로 정합** (강건) |
| **데이터 스케일링** | 포화 | 포화 | **선형 스케일링** |
| **추가 학습 필요** | ✅ (1~2일) | ❌ (EAGLE-1 체크포인트 재사용) | ✅ (1~2시간, H100) |
| **학습 가능 파라미터** | FC + 1 디코더 레이어 | 동일 | FC(3→1) + 1 디코더 레이어 |
| **평균 속도 향상 (T=0)** | 2.7×~3.1× | 3.3×~4.3× | **4.0×~5.5×** |
| **최대 속도 향상** | ~3.8× | ~5.0× | **~6.5×** |
| **대규모 배치 처리량** | 배치 증가 시 하락 | 배치 24+ 하락 | **배치 56까지 유지** |
| **Lossless 보장** | ✅ | ✅ | ✅ |

EAGLE-1의 강점은 **개념적 단순성과 범용성**이다. Feature-level autoregression이라는 우아한 아이디어로 기존 방법 대비 큰 도약을 이루었다. 약점은 정적 트리로 인한 연산 예산 낭비와 학습-추론 분포 불일치로 인한 오차 누적이다.

EAGLE-2는 **추가 학습 없이** 순수 추론 시간 최적화만으로 20~40% 개선을 달성하여 실용성이 극히 높다. 기존 EAGLE-1 체크포인트를 그대로 활용할 수 있어 배포가 즉각적이다. 그러나 근본적인 feature 예측 한계와 오차 누적 문제는 해결하지 못한다.

EAGLE-3는 feature 예측 제거, TTT, 다층 융합이라는 세 가지 혁신을 통합하여 **스케일링의 벽을 돌파**했다. 학습 데이터 증가에 비례하는 성능 향상을 처음으로 실현했으며, 대규모 배치에서의 처리량 유지 능력도 탁월하다. 단, 모델별로 드래프트 헤드를 학습해야 하므로 폐쇄형 API 모델에는 적용 불가하다.

---

## 현재의 한계와 미래 연구 방향

**모델별 학습의 불가피성.** EAGLE 시리즈의 가장 큰 실용적 제약은 타겟 모델마다 전용 드래프트 헤드를 학습해야 한다는 점이다. EAGLE-3도 이 제약에서 자유롭지 않다. 진정한 training-free 접근법의 개발이 중요한 연구 방향이다.

**폐쇄형 모델 비호환성.** EAGLE는 타겟 모델의 은닉 상태에 접근해야 하므로, API만 제공하는 폐쇄형 모델(GPT-4, Claude 등)에는 적용할 수 없다. 출력 분포만으로 speculative decoding을 수행하는 방법론의 발전이 필요하다.

**배치 크기 증가 시 효용 감소.** 배치가 커지면 GPU가 연산 바운드에 가까워져 speculative decoding의 근본 전제(유휴 연산 용량 활용)가 약화된다. EAGLE-3가 배치 56까지 유지하는 것은 발전이나, 수백 배치의 대규모 서빙에서는 여전히 한계가 있다. 배치 서빙과 speculative decoding의 효율적 결합이 활발히 연구되고 있다.

**MoE 모델에서의 낮은 효율.** Mixture-of-Experts 모델에서는 여러 드래프트 토큰 검증 시 2개 이상의 전문가 가중치를 로드해야 할 수 있어, 메모리 대역폭 이점이 줄어든다. Mixtral 8x7B에서 EAGLE-1의 속도 향상은 **1.50배**에 그쳤다.

**학습 데이터 분포 민감도.** EAGLE가 채팅 데이터(ShareGPT)로 학습되면 특정 도메인(예: 독일어→영어 번역)에서 성능이 저하될 수 있다. 도메인 적응적 드래프트 헤드 학습이나 범용 드래프트 모델의 개발이 과제로 남아 있다.

**더 깊은 이론적 이해.** Speculative decoding의 최적 트리 구조에 대한 이론적 분석(SEQUOIA의 동적 프로그래밍 접근), 다양한 디코딩 전략(beam search, nucleus sampling 등)과의 결합, 그리고 최적의 드래프트-타겟 모델 크기 비율에 대한 연구가 진행 중이다. EAGLE-3가 보여준 스케일링 법칙의 이론적 근거를 규명하는 것도 향후 과제이다.

## 결론: Feature에서 시작하여 스케일링으로 도달하다

Speculative decoding은 LLM 추론의 메모리 바운드 특성을 역이용한 정교한 시스템 최적화이며, 그 수학적 정당성은 rejection sampling의 정확성 증명에 의해 완벽히 보장된다. EAGLE 시리즈는 이 패러다임 위에서 세 번의 질적 도약을 이루었다. EAGLE-1이 "feature는 토큰보다 예측하기 쉽다"는 원리로 기반을 놓았고, EAGLE-2가 "트리 구조는 문맥에 따라 달라야 한다"는 통찰로 추론 시간 효율을 극대화했으며, EAGLE-3는 역설적으로 feature 예측 자체를 제거하고 학습-추론 정합성을 확보하여 **데이터 스케일링이라는 새로운 차원을 열었다**. 최대 6.5배의 lossless 가속과 프로덕션 프레임워크 통합은, EAGLE가 학술적 기여를 넘어 실용적 LLM 서빙의 표준 기법으로 자리잡고 있음을 보여준다.