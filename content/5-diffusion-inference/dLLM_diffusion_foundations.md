# Diffusion 기초 from Scratch to dLLM 

> 목적: **Diffusion Language Model (dLLM)** 을 논문 수준에서 이해하기 위해 필요한 diffusion 이론의 최소 충분 집합을, 연속(continuous) → 이산(discrete) → 언어(language) 순으로 수학적으로 정리한다. 각 절은 "왜 이 개념이 dLLM에 필요한가"를 중심으로 작성하였다.

---

## 0. 전체 지도 (Roadmap) (by claude)

dLLM을 제대로 읽으려면 두 갈래의 계보를 모두 잡아야 한다.

```
[연속 diffusion]                         [이산 diffusion]
DDPM (Ho 2020)                           D3PM (Austin 2021)
   │  ε-prediction / ELBO                     │  transition matrix Q, absorbing state
   ▼                                          ▼
Score SDE (Song 2021)                    CTMC / concrete score
   │  VP/VE SDE, reverse SDE, PF-ODE          │  ratio matching
   ▼                                          ▼
   ───────────────────────┌───────────────────┘
                          ▼
            SEDD (Lou 2024) · MDLM/MD4 (Sahoo/Shi 2024) · RADD (Ou 2024)
                          │  masked/absorbing diffusion = weighted MLM
                          ▼
            Block Diffusion / BD3-LM (Arriola 2025) ── AR↔diffusion 보간
                          ▼
            LLaDA 8B (Nie 2025) · Dream 7B · DiffuLLaMA · Mercury · LLaDA 2.0
```

핵심 통찰을 미리 말하면: **현대 대규모 dLLM(LLaDA, Dream 등)은 거의 전부 "absorbing-state(=masking) 이산 diffusion"의 변형**이며, 그 손실 함수는 결국 **시간 가중치가 붙은 Masked Language Modeling(MLM) cross-entropy**로 환원된다. 따라서 연속 diffusion의 SDE/score 이론은 *왜 이것이 원리적인 생성 모델인가*를 이해하는 토대이고, 이산 diffusion 이론은 *실제 dLLM이 무엇을 최적화하는가*를 알려준다.

---

## 1. 동기: 왜 autoregressive 대신 diffusion인가

Autoregressive model(ARM)은 사슬 규칙으로 결합 분포를 분해한다.

$$
\log p_\theta(\mathbf{x}) = \sum_{i=1}^{L} \log p_\theta(x^i \mid x^{<i})
$$

이는 강력하지만 두 가지 구조적 제약을 갖는다. (1) **순차적(left-to-right) 생성** — 토큰당 1회 forward, 병렬화 불가. (2) **단방향(causal) 컨텍스트** — 미래 토큰을 못 본다(이른바 *reversal curse* 의 한 원인).

Diffusion 계열은 결합 분포를 **시간에 걸친 점진적 denoising**으로 분해한다. 텍스트에 적용된 masked diffusion은 양방향(bidirectional) attention을 쓰고 여러 토큰을 한 스텝에 병렬 예측할 수 있어, ARM의 위 두 제약을 원리적으로 우회한다. <cite index="42-1">LLaDA는 forward masking 과정과 reverse 생성 과정을 Transformer로 마스크 토큰 예측에 파라미터화하고, likelihood lower bound를 최적화하는 원리적 생성 접근을 제공한다.</cite> 저자들의 주장은 *지능의 원천은 autoregressive 메커니즘 자체가 아니라 최대우도(maximum likelihood)로 참 언어 분포를 근사하는 생성 모델링 원리*라는 것이다.

> dLLM이 푸는 질문: "ARM 없이도, diffusion으로 scaling·in-context learning·instruction following이 나오는가?" — LLaDA 8B는 LLaMA3 8B와 경쟁적이라는 것을 보였다.

---

## 2. 연속 Diffusion 토대 (1) — DDPM

### 2.1 Forward process (고정된 noising)

DDPM은 데이터 $\mathbf{x}_0 \sim q(\mathbf{x}_0)$ 에 Gaussian noise를 점진적으로 주입하는 **고정된** Markov chain을 정의한다.

$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\!\big(\mathbf{x}_t;\ \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\ \beta_t \mathbf{I}\big),
\qquad
q(\mathbf{x}_{1:T}\mid \mathbf{x}_0)=\prod_{t=1}^{T} q(\mathbf{x}_t\mid\mathbf{x}_{t-1})
$$

여기서 $\beta_t\in(0,1)$ 는 **variance schedule**. $\alpha_t := 1-\beta_t$, $\bar\alpha_t := \prod_{s=1}^t \alpha_s$ 로 두면, Gaussian의 합성성질로 **임의 시점 $t$ 의 주변분포가 닫힌 형태(closed form)** 로 나온다:

$$
\boxed{\,q(\mathbf{x}_t\mid \mathbf{x}_0)=\mathcal{N}\!\big(\mathbf{x}_t;\ \sqrt{\bar\alpha_t}\,\mathbf{x}_0,\ (1-\bar\alpha_t)\mathbf{I}\big)\,}
\quad\Longleftrightarrow\quad
\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,\ \ \boldsymbol\epsilon\sim\mathcal N(0,\mathbf I)
$$

이 **reparameterization** 이 학습을 가능하게 하는 핵심이다: 임의의 $t$ 에서 $\mathbf{x}_t$ 를 한 번에 샘플링할 수 있다. $T\to\infty$ (또는 충분히 큰 $T$)에서 $\bar\alpha_T\approx 0$ 이므로 $q(\mathbf{x}_T)\approx\mathcal N(0,\mathbf I)$.

### 2.2 Reverse process (학습되는 denoising)

생성은 prior $p(\mathbf{x}_T)=\mathcal N(0,\mathbf I)$ 에서 시작해 역방향 chain을 따라간다:

$$
p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t)=\mathcal N\!\big(\mathbf{x}_{t-1};\ \boldsymbol\mu_\theta(\mathbf{x}_t,t),\ \boldsymbol\Sigma_\theta(\mathbf{x}_t,t)\big)
$$

핵심 보조정리: **$\mathbf{x}_0$ 로 조건부화한 forward posterior는 tractable Gaussian** 이다.

$$
q(\mathbf{x}_{t-1}\mid \mathbf{x}_t,\mathbf{x}_0)=\mathcal N\!\big(\mathbf{x}_{t-1};\ \tilde{\boldsymbol\mu}_t(\mathbf{x}_t,\mathbf{x}_0),\ \tilde\beta_t\mathbf I\big),
\quad
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
$$
$$
\tilde{\boldsymbol\mu}_t(\mathbf{x}_t,\mathbf{x}_0)=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\mathbf{x}_0+\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t
$$

### 2.3 변분 하한 (Variational bound / ELBO)

$\mathbf{x}_{1:T}$ 를 잠재변수로 보면 (VAE와 동일 구조; forward가 encoder 역할), 음의 로그우도에 대한 변분 상한을 얻는다:

$$
-\log p_\theta(\mathbf{x}_0)\ \le\ \mathbb{E}_q\Big[\underbrace{D_{\mathrm{KL}}\big(q(\mathbf{x}_T\mid\mathbf{x}_0)\,\|\,p(\mathbf{x}_T)\big)}_{L_T:\ \text{prior matching}}+\sum_{t>1}\underbrace{D_{\mathrm{KL}}\big(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)\,\|\,p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\big)}_{L_{t-1}:\ \text{denoising matching}}\underbrace{-\log p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)}_{L_0}\Big]
$$

두 Gaussian 사이의 $L_{t-1}$ KL은 평균 차이의 제곱으로 닫힌 형태가 된다. $\boldsymbol\mu_\theta$ 를 posterior $\tilde{\boldsymbol\mu}_t$ 형태로 파라미터화하고 $\mathbf{x}_0=(\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon)/\sqrt{\bar\alpha_t}$ 를 대입하면, **noise 예측 $\boldsymbol\epsilon_\theta$** 형태로 정리된다.

### 2.4 단순화된 목적함수 (이게 실제로 학습되는 것)

Ho et al. (2020)은 가중치를 1로 두는 단순화가 더 잘 작동함을 보였다:

$$
\boxed{\ L_{\text{simple}}(\theta)=\mathbb{E}_{t\sim\mathcal U,\,\mathbf{x}_0,\,\boldsymbol\epsilon}\Big[\big\|\boldsymbol\epsilon-\boldsymbol\epsilon_\theta\big(\underbrace{\sqrt{\bar\alpha_t}\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon}_{\mathbf{x}_t},\,t\big)\big\|^2\Big]\ }
$$

즉 **"임의 노이즈 수준에서 주입된 노이즈를 회귀(regress)하라"**. 샘플링은 $\boldsymbol\epsilon_\theta$ 로 $\boldsymbol\mu_\theta$ 를 복원해 $\mathbf{x}_T\to\mathbf{x}_{T-1}\to\cdots\to\mathbf{x}_0$.

> **dLLM 연결점**: 이 "임의 손상 수준 → 원본 복원" 구조가 그대로 텍스트로 옮겨가면 *임의 마스킹 비율 → 마스크 토큰 복원*이 된다(§6). $\boldsymbol\epsilon_\theta$(연속) ↔ mask predictor $p_\theta(x_0\mid x_t)$(이산)가 대응한다.

---

## 3. 연속 Diffusion 토대 (2) — Score matching & SDE

### 3.1 ε-prediction은 사실 score estimation이다 (Tweedie)

Gaussian perturbation kernel의 score(로그밀도의 gradient)는 닫힌 형태다:

$$
\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t\mid\mathbf{x}_0)
=-\frac{\mathbf{x}_t-\sqrt{\bar\alpha_t}\mathbf{x}_0}{1-\bar\alpha_t}
=-\frac{\boldsymbol\epsilon}{\sqrt{1-\bar\alpha_t}}
$$

따라서 DDPM의 noise 예측기와 score 함수는 **단순 스케일 관계**:

$$
\mathbf{s}_\theta(\mathbf{x}_t,t)\approx\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t)=-\frac{\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)}{\sqrt{1-\bar\alpha_t}}
$$

이 등가성이 DDPM(Ho)과 score-based model(Song & Ermon, NCSN)을 하나로 묶는다. *"denoising = score estimation"*.

### 3.2 연속 시간 일반화: Stochastic Differential Equation

<cite index="16-1">Song et al. (2021)은 복잡한 데이터 분포를 노이즈를 천천히 주입해 알려진 prior로 매끄럽게 변환하는 SDE와, 노이즈를 천천히 제거해 prior를 데이터 분포로 되돌리는 reverse-time SDE를 제시했다.</cite> Forward를 Itô SDE로 적는다:

$$
\mathrm d\mathbf{x}=\underbrace{\mathbf f(\mathbf{x},t)}_{\text{drift}}\,\mathrm dt+\underbrace{g(t)}_{\text{diffusion}}\,\mathrm d\mathbf w
$$

<cite index="16-1">결정적으로, reverse-time SDE는 perturbed 데이터 분포의 시간 의존 gradient field(즉 score)에만 의존한다.</cite> (Anderson 1982)

$$
\boxed{\ \mathrm d\mathbf{x}=\big[\mathbf f(\mathbf{x},t)-g(t)^2\,\nabla_{\mathbf{x}}\log p_t(\mathbf{x})\big]\,\mathrm dt+g(t)\,\mathrm d\bar{\mathbf w}\ }
$$

- **VP-SDE** (Variance Preserving): $\mathbf f=-\tfrac12\beta(t)\mathbf{x},\ g=\sqrt{\beta(t)}$ → 연속시간 **DDPM**.
- **VE-SDE** (Variance Exploding): $\mathbf f=0,\ g=\sqrt{\mathrm d[\sigma^2(t)]/\mathrm dt}$ → 연속시간 **SMLD/NCSN**.

### 3.3 Probability Flow ODE

같은 주변분포 $p_t$ 를 갖는 **결정론적** ODE가 존재한다:

$$
\mathrm d\mathbf{x}=\Big[\mathbf f(\mathbf{x},t)-\tfrac12 g(t)^2\,\nabla_{\mathbf{x}}\log p_t(\mathbf{x})\Big]\mathrm dt
$$

이는 정확한 likelihood 계산(연속 정규화 흐름과 동치)과 빠른 결정론적 샘플러(DDIM 등)의 이론적 기반이다.

> **왜 dLLM 학습자가 이걸 알아야 하나**: (1) 이산 diffusion의 "concrete score / ratio"(§5)는 바로 이 score 개념의 *이산 유사물*이다. (2) SEDD의 손실(score entropy)은 score matching의 이산 확장이다. SDE/score 사고틀이 없으면 SEDD 계열 논문이 임의의 손실 함수처럼 보인다.

---

## 4. 이산 Diffusion 토대 (1) — 왜 텍스트는 다른가, 그리고 D3PM

### 4.1 두 가지 경로

텍스트는 **이산 토큰(categorical)** 이라 Gaussian noise를 직접 더할 수 없다. 두 갈래가 생겼다.

- **(A) 연속화(embedding) 경로 — Diffusion-LM (Li et al., 2022)**: 토큰을 연속 embedding으로 올린 뒤 연속 Gaussian diffusion을 돌리고, 마지막에 rounding으로 토큰으로 내린다. 초기 가능성을 보였으나 embedding-rounding 불일치, 학습 난이도 등으로 대규모 확장에서 불리. (참고: SSD-LM, GENIE, DiffuSeq 등 동계열)
- **(B) 진짜 이산(categorical) 경로 — D3PM (Austin et al., 2021)**: 연속공간으로의 relaxation 없이 categorical 상태공간에서 직접 Markov 손상/복원. **현대 dLLM의 주류 계보**.

### 4.2 D3PM: transition matrix로 정의되는 forward

상태(어휘) 크기를 $K$ 라 하고 one-hot 토큰 $\mathbf{x}_{t-1}$ 에 대해

$$
q(\mathbf{x}_t\mid\mathbf{x}_{t-1})=\mathrm{Cat}\big(\mathbf{x}_t;\ \mathbf{p}=\mathbf{x}_{t-1}\mathbf{Q}_t\big),
\qquad [\mathbf{Q}_t]_{ij}=q(x_t=j\mid x_{t-1}=i)
$$

누적 행렬 $\bar{\mathbf Q}_t=\mathbf Q_1\mathbf Q_2\cdots\mathbf Q_t$ 로 임의 시점 주변분포가 행렬곱으로 닫힌 형태가 된다: $q(\mathbf x_t\mid\mathbf x_0)=\mathrm{Cat}(\mathbf x_0\bar{\mathbf Q}_t)$. <cite index="21-1">D3PM은 균일(uniform) 전이확률을 넘어, 연속공간의 Gaussian 커널을 모사하는 행렬, embedding 공간 최근접 이웃 기반 행렬, 그리고 absorbing state를 도입하는 행렬까지 일반화한다.</cite>

### 4.3 핵심: Absorbing state = Masking

가장 중요한 선택지가 **absorbing-state** 전이다. 특수 토큰 $[\text{MASK}]$ (인덱스 $m$)을 흡수 상태로 둔다:

$$
\mathbf Q_t=(1-\beta_t)\mathbf I+\beta_t\,\mathbf 1\,\mathbf e_m^\top
$$

즉 각 토큰은 확률 $\beta_t$ 로 $[\text{MASK}]$ 가 되고, 한 번 마스크되면 영원히 마스크로 남는다. $t$ 가 커질수록 전부 $[\text{MASK}]$ 로 수렴(이게 prior). <cite index="21-1">이 세 번째(absorbing) 선택지가 diffusion model과 autoregressive·mask 기반 생성 모델 사이의 연결을 그릴 수 있게 해 준다.</cite> 실험적으로도 <cite index="30-1">absorbing(=masking) 손상 전략이 텍스트 데이터에 특히 잘 맞아, text8·LM1B에서 uniform·nearest-neighbor 전이를 일관되게 능가했다.</cite>

D3PM의 학습 목적은 DDPM과 같은 구조의 변분 하한이며(다만 KL이 categorical 분포 간 KL), 저자들은 <cite index="30-1">변분 하한에 보조 cross-entropy 손실을 결합해 학습을 안정화하고 샘플 품질을 높였다.</cite>

> **이 절이 dLLM의 분기점**: "absorbing-state 이산 diffusion ≡ 점진적 마스킹/언마스킹"이라는 등식. 이후 SEDD·MDLM·LLaDA·Dream은 전부 이 absorbing(=masked) 정식화의 정련(refinement)이다.

---

## 5. 이산 Diffusion 토대 (2) — 현대적 정식화 (CTMC, SEDD, MDLM, RADD)

### 5.1 연속시간 Markov chain (CTMC) 관점

이산 시점을 연속시간으로 보내면 D3PM은 CTMC가 된다. 확률질량벡터 $\mathbf p_t$ 의 시간 변화는 미분방정식으로:

$$
\frac{\mathrm d\mathbf p_t}{\mathrm dt}=\mathbf Q_t\,\mathbf p_t,\qquad \mathbf p_0\approx p_{\text{data}}
$$

이 chain을 역방향으로 돌리려면 연속 diffusion의 score에 해당하는 양이 필요한데, 그것이 **concrete score / ratio**:

$$
\text{(concrete score)}\quad \frac{p_t(\mathbf y)}{p_t(\mathbf x)}\quad (\mathbf y\ne\mathbf x),
$$

즉 *주변확률들의 비율*이 연속 score $\nabla\log p_t$ 의 이산 유사물이다 (Meng et al. 2022; Sun et al. 2023; Lou et al. 2024).

### 5.2 SEDD — Score Entropy Discrete Diffusion (Lou, Meng, Ermon; ICML 2024 Best Paper)

<cite index="32-1">SEDD는 score matching을 이산공간으로 자연스럽게 확장하는 score entropy라는 새 손실을 제안해, 이산 diffusion model 구축에 매끄럽게 통합하고 성능을 크게 끌어올렸다.</cite> 네트워크 $\mathbf s_\theta(\mathbf x,t)$ 가 미지의 concrete score $\{p_t(\hat{\mathbf x})/p_t(\mathbf x)\}$ 를 학습한다. 학습 가능한(denoising) 형태의 score entropy 목적은 다음 구조를 갖는다:

$$
\mathcal L_{\text{DWDSE}}(\mathbf x_0)=\int_0^T \mathbb E_{\mathbf x_t\sim p_{t|0}(\cdot|\mathbf x_0)}\sum_{\hat{\mathbf x}_t\ne\mathbf x_t} Q_t(\hat{\mathbf x}_t,\mathbf x_t)\,\mathcal I\!\Big(\mathbf s_\theta(\mathbf x_t,t)_{\hat{\mathbf x}_t},\ \tfrac{p_{t|0}(\hat{\mathbf x}_t\mid\mathbf x_0)}{p_{t|0}(\mathbf x_t\mid\mathbf x_0)}\Big)\,\mathrm dt
$$

여기서 $\mathcal I(a,b)=a-b\log a+(\text{const})$ 형태의 Bregman 유형 발산으로, $a$ 가 목표 비율 $b$ 를 맞추도록 강제한다. 성과: <cite index="34-1">동일 모델 크기에서 SEDD는 기존 언어 diffusion 패러다임 대비 perplexity를 25–75% 줄였고, 특히 GPT-2를 능가하며 autoregressive model과 경쟁적이었다.</cite> 샘플링은 Tweedie $\tau$-leaping을 쓴다.

### 5.3 MDLM / MD4 — Masked Diffusion이 사실은 가중 MLM이다 (Sahoo 2024; Shi 2024)

여기가 **실용적으로 가장 중요한 결과**다. Masked(absorbing) diffusion의 forward는 단순하다: 각 토큰을 독립적으로 확률 $1-\alpha_t$ 로 $[\text{MASK}]$ 로 바꾼다($\alpha_t$ 는 $\alpha_0\!\approx\!1\to\alpha_1\!\approx\!0$ 로 감소하는 스케줄). <cite index="38-1">MDLM(Sahoo et al., 2024)은 치환(substitution) 기반 파라미터화(SUBS)를 도입하고, weighted Masked Language Modeling(MLM) 손실과 동치인 단순화된 Rao-Blackwellized ELBO를 유도해, encoder-only 모델의 생성적 학습을 가능케 했다.</cite> 연속시간 NELBO는 다음과 같은 *마스크 위치에서의 가중 cross-entropy* 로 환원된다:

$$
\mathcal L_{\text{NELBO}}=\mathbb E_{t\sim\mathcal U[0,1]}\,\mathbb E_{\mathbf x_t\sim q(\cdot\mid\mathbf x_0)}\Big[\frac{\alpha_t'}{1-\alpha_t}\sum_{i:\,x_t^i=[\text{MASK}]}\!\!\big(-\log p_\theta(x_0^i\mid \mathbf x_t)\big)\Big]
$$

<cite index="38-1">MD4(Shi et al., 2024)는 이 틀을 더 통합해, 연속 diffusion과 유사한 SNR 불변(SNR-invariance) 성질을 갖는 단순 ELBO를 유도하고 상태 의존(state-dependent) 마스킹 스케줄로 일반화했다.</cite> SUBS 파라미터화의 두 트릭은 (i) **zero masking probability**(이미 unmask된 토큰은 다시 mask로 예측하지 않음)와 (ii) **carry-over unmasking**(unmask된 토큰은 그대로 운반)으로, 손실을 마스크된 위치에만 집중시킨다.

### 5.4 RADD — 시간 독립 인수분해 (Ou et al., 2024)

<cite index="38-1">Ou et al.(2024)은 absorbing diffusion에서 concrete score가 "시간 독립 conditional"과 "시간 의존 스칼라"의 곱으로 인수분해됨을 보여 모델을 단순화(RADD)하고 Denoising Cross-Entropy(DCE) 손실을 얻었다.</cite> 함의: absorbing diffusion network는 사실 *시간 $t$ 에 (거의) 의존하지 않는 clean-data conditional* 을 학습한다 — 이는 "마스크 예측기는 time-agnostic하다"는 후속 관찰(Zheng et al. 2024, *MDM are secretly time-agnostic masked models*)로 이어진다.

> 요약 등식 체인:
> **absorbing D3PM → CTMC concrete score(SEDD) → SUBS로 단순화된 ELBO = 가중 MLM(MDLM/MD4) → 시간독립 인수분해(RADD)**.
> 실전 dLLM은 이 마지막 단순형, 즉 "마스크된 토큰에 대한 (시간가중) cross-entropy"를 최적화한다.

---

## 6. AR ↔ Diffusion 보간 — Block Diffusion (BD3-LM)

순수 masked diffusion(MDM)의 두 약점: (1) **고정 길이** 생성(임의/가변 길이 어려움), (2) **KV cache 불가**(양방향 attention이라 매 스텝 전체 재계산) → 추론 비용 큼. <cite index="51-1">이를 잇기 위해 BD3-LM(Arriola et al., 2025)은 시퀀스를 블록으로 나눠 블록 내부는 diffusion으로, 블록 간에는 autoregressive하게 조건화하여, 순수 AR과 순수 diffusion 사이를 보간한다.</cite>

$$
-\log p_\theta(\mathbf x)=-\sum_{b=1}^{B}\log p_\theta(\mathbf x^b\mid \mathbf x^{<b})\ \le\ \sum_{b=1}^{B}\mathcal L_{\text{MDM}}(\mathbf x^b;\ \mathbf x^{<b})
$$

여기서 블록 내부 조건부는 MDLM의 NELBO를 그대로 쓴다. **블록 크기 $L'$ 가 보간 손잡이**: $L'=1$ → 완전한 AR, $L'=L$ → 순수 MDLM. <cite index="51-1">BD3-LM은 가변 길이 생성과 블록 간 KV caching이라는 두 이점을 얻어 추론 효율을 개선한다.</cite> 다만 <cite index="58-1">diffusion 목적의 gradient 분산이 높아 블록 크기 1에서도 AR보다 성능이 떨어지는 문제가 있어, 저자들은 gradient 분산 추정량을 유도하고 분산을 최소화하는 맞춤형 noise 과정을 제안해 perplexity 격차를 좁혔다.</cite>

후속: Eso-LM(Sahoo 2025; MDM과 AR을 결합), SDAR·Fast-dLLM v2(AR 사전학습 모델을 block diffusion으로 파인튜닝), D2F(diffusion forcing 기반 증류) 등이 모두 이 보간 축 위에 있다.

---

## 7. 실제 대규모 dLLM 지형 (2025–2026)

| 모델 | 정식화 | 규모/초기화 | 특징 |
|---|---|---|---|
| **LLaDA** (Nie 2025) | masked(absorbing) diffusion | 8B, **from scratch** | <cite index="42-1">forward masking + reverse 생성, ELBO 최적화; LLaMA3 8B와 in-context learning 경쟁적, SFT 후 instruction following</cite> |
| **Dream 7B** (Ye 2025) | masked diffusion (MDM) | 7B, **AR 가중치로 초기화** | <cite index="43-1">adaptive noise scheduling으로 LLaMA3-8B급 성능을 효율적으로 달성</cite> |
| **DiffuLLaMA / DiffuGPT** (Gong 2025) | masked diffusion | 127M–7B, **AR→diffusion 적응** | <cite index="41-1">기존 AR(GPT-2, LLaMA)을 diffusion으로 적응(adapt)</cite> |
| **Mercury Coder** (Inception Labs 2025) | 상용 dLLM | — | <cite index="41-1">코드 생성에서 상업적 적용성과 효율을 보임</cite> |
| **LLaDA 2.0** | block diffusion | **~100B로 스케일** | <cite index="59-1">block diffusion을 100B로 확장하고 추론 속도를 최적화, 강력한 오픈소스 AR-LM과 경쟁하며 훨씬 빠른 추론</cite> |
| **LLaDA-V / MMaDA / DiffuCoder** | 멀티모달·코드 특화 | 7–8B | 비전 instruction tuning, modality-agnostic diffusion, any-order 코드 생성 |

LLaDA의 학습 손실은 §5.3의 가중 MLM 형태를 그대로 따른다(마스킹 비율 $t\sim\mathcal U[0,1]$, 마스크된 위치만 cross-entropy, $1/t$ 가중). <cite index="45-1">LLaDA는 표준 pretraining·SFT를 따르되 샘플링은 diffusion으로 하며, pretraining에서는 비율 $t\sim U[0,1]$ 로 전 토큰을 무작위 마스킹하고 SFT에서는 응답 토큰만 마스킹한다.</cite>

> 흥미로운 역설(reversal): <cite index="42-1">LLaDA는 토큰을 균일하게 다뤄 inductive bias가 없어, 정방향·역방향 reasoning에서 균형 잡힌 성능을 보인다</cite> — ARM의 reversal curse를 구조적으로 완화하는 사례.

---

## 8. 샘플링·추론 측면 (ML Systems 관점에서 가장 중요)

dLLM은 학습보다 **추론(inference) 설계공간**이 훨씬 풍부하고 까다롭다. inference engineer 관점에서 짚을 핵심:

1. **샘플링 = 반복적 언마스킹**: $t=1$(전부 마스크) → $t=0$. 매 스텝 마스크 예측기로 *모든* 마스크 위치를 동시 예측한 뒤, 일부를 확정(unmask)하고 나머지는 다시 마스크(remask).
2. **Remasking 전략**: LLaDA의 **low-confidence remasking**(가장 확신 높은 토큰만 확정), 또는 무작위/스케줄 기반. 한 스텝에 확정하는 토큰 수가 **품질↔지연(latency) 트레이드오프**를 직접 조절한다.
3. **NFE(number of function evaluations) vs 품질**: diffusion 스텝 수 $T'$ 가 비용을 지배. 스텝을 줄이면 빠르지만 부정확한 categorical 샘플링 오차가 누적(Zheng 2024가 지적한 *inaccurate categorical sampling* 문제).
4. **KV cache 부재 문제**: 순수 MDM은 양방향 attention이라 토큰이 확정돼도 컨텍스트가 매 스텝 바뀌어 표준 KV cache를 못 쓴다. 해법 두 갈래:
   - **구조적**: block diffusion(§6) — 블록 간 KV cache 재사용.
   - **근사 캐싱**: dLLM-Cache(Liu 2025) 등 — <cite index="46-1">MDM은 이산 시퀀스에서 컨텍스트 기반으로 마스크 토큰을 반복 예측하는데, 이 반복성을 이용해 adaptive caching으로 가속한다.</cite>
5. **Speculative / parallel decoding**: self-speculative decoding(2025), WINO(draft-and-verify + 양방향 재마스킹), SlowFast Sampling(탐색/가속 단계 동적 전환), Set Block Decoding(NTP와 결합) 등. (Jay의 EAGLE/MTP·spec decoding 배경과 직접 연결되는 영역)
6. **병렬성의 이론적 상한**: 한 스텝에 여러 토큰을 독립적으로 확정하면, 그 토큰들 간 결합 의존성을 무시하게 된다(독립 가정). 이것이 dLLM 병렬 생성의 근본 품질 한계이며, 적응적 스텝 수·verify 메커니즘이 이를 보정하려는 시도다.

---

## 9. 정리: dLLM을 읽기 위한 최소 수학 체크리스트

- [ ] **Gaussian reparameterization** $\mathbf x_t=\sqrt{\bar\alpha_t}\mathbf x_0+\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon$ 와 tractable posterior
- [ ] **ELBO 분해** → denoising matching KL → simplified $\boldsymbol\epsilon$-prediction 손실
- [ ] **Tweedie**: $\boldsymbol\epsilon$-prediction ≡ score estimation
- [ ] **reverse-time SDE & probability flow ODE** (개념적 토대)
- [ ] **D3PM transition matrix** $\mathbf Q_t$, 특히 **absorbing = masking** 등식
- [ ] **concrete score(ratio)** = 이산 score, **SEDD score entropy**
- [ ] **MDLM/MD4 핵심 결과**: masked diffusion ELBO = **가중 MLM cross-entropy** (+ SUBS, Rao-Blackwell)
- [ ] **RADD**: absorbing의 시간독립 인수분해 (time-agnostic mask predictor)
- [ ] **Block Diffusion 보간**: 블록 크기로 AR↔diffusion, KV cache·가변 길이
- [ ] **추론 설계공간**: remasking, NFE, 병렬 디코딩, 캐싱

## 10. 권장 읽기 순서 (Reading Roadmap)

1. **Ho et al. 2020 — DDPM** (arXiv:2006.11239): §2 전체의 원전.
2. **Song et al. 2021 — Score SDE** (arXiv:2011.13456): §3, score·SDE·PF-ODE 통합 관점.
3. **Austin et al. 2021 — D3PM** (arXiv:2107.03006): §4, 이산화와 absorbing=masking.
4. **Lou et al. 2024 — SEDD** (arXiv:2310.16834, ICML Best Paper): §5.2, 이산 score matching.
5. **Sahoo et al. 2024 — MDLM** (arXiv:2406.07524) + **Shi et al. 2024 — MD4**: §5.3, 가중 MLM 등치 — *실전 dLLM 손실의 정체*.
6. **Ou et al. 2024 — RADD** + **Zheng et al. 2024** (arXiv:2409.02908): §5.4, time-agnostic 통찰.
7. **Arriola et al. 2025 — Block Diffusion / BD3-LM** (arXiv:2503.09573, ICLR Oral): §6, AR 보간·추론 효율.
8. **Nie et al. 2025 — LLaDA** (arXiv:2502.09992) + **Ye et al. 2025 — Dream**: §7, 대규모 실증.
9. (추론 심화) dLLM-Cache(arXiv:2506.06295), Fast-dLLM v2(arXiv:2509.26328), self-speculative decoding(arXiv:2510.04147).

---

## 부록 A: 증명 (Proofs)

> 표기: $\alpha_t=1-\beta_t$, $\bar\alpha_t=\prod_{s\le t}\alpha_s$. 연속(A.1–A.3)에서는 데이터가 $\mathbb R^d$, 이산(A.4–A.7)에서는 토큰이 어휘 $\{1,\dots,K\}$ 의 one-hot. masked diffusion의 스케줄 $\alpha_t$ 는 "원본 유지 확률"로, $\alpha_0=1\to\alpha_1=0$ 으로 감소($[\text{MASK}]$ 인덱스를 $m$).

### A.1 DDPM 변분 하한의 telescoping 유도

**주장.** $-\log p_\theta(\mathbf x_0)\le \mathbb E_q[L_T+\sum_{t\ge2}L_{t-1}+L_0]$, 단
$L_T=D_{\mathrm{KL}}(q(\mathbf x_T|\mathbf x_0)\|p(\mathbf x_T))$, $L_{t-1}=D_{\mathrm{KL}}(q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)\|p_\theta(\mathbf x_{t-1}|\mathbf x_t))$, $L_0=-\log p_\theta(\mathbf x_0|\mathbf x_1)$.

**증명.** Jensen 부등식($-\log$ 볼록)으로 잠재변수 도입:

$$
-\log p_\theta(\mathbf x_0)=-\log\!\int p_\theta(\mathbf x_{0:T})\,\mathrm d\mathbf x_{1:T}
\le \mathbb E_{q(\mathbf x_{1:T}|\mathbf x_0)}\!\Big[-\log\frac{p_\theta(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_0)}\Big]
=\mathbb E_q\Big[-\log p(\mathbf x_T)-\sum_{t\ge1}\log\frac{p_\theta(\mathbf x_{t-1}|\mathbf x_t)}{q(\mathbf x_t|\mathbf x_{t-1})}\Big].
$$

$t\ge2$ 항에서 Markov성으로 $q(\mathbf x_t|\mathbf x_{t-1})=q(\mathbf x_t|\mathbf x_{t-1},\mathbf x_0)$, 이어 Bayes:

$$
q(\mathbf x_t|\mathbf x_{t-1},\mathbf x_0)=q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)\,\frac{q(\mathbf x_t|\mathbf x_0)}{q(\mathbf x_{t-1}|\mathbf x_0)}.
$$

따라서 $\sum_{t=2}^T\log q(\mathbf x_t|\mathbf x_{t-1})=\sum_{t=2}^T\log q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)+\sum_{t=2}^T[\log q(\mathbf x_t|\mathbf x_0)-\log q(\mathbf x_{t-1}|\mathbf x_0)]$. 둘째 합은 **telescoping** 되어 $\log q(\mathbf x_T|\mathbf x_0)-\log q(\mathbf x_1|\mathbf x_0)$. 이를 대입하고 $t=1$ 항을 분리한 뒤 각 항을 KL로 묶으면 주장이 나온다. $\;\square$

### A.2 Denoising matching 항 → $\boldsymbol\epsilon$-손실 (Rao-Blackwellization)

DDPM은 $p_\theta(\mathbf x_{t-1}|\mathbf x_t)=\mathcal N(\boldsymbol\mu_\theta,\sigma_t^2\mathbf I)$ 로 분산을 고정한다. **등분산 Gaussian 간 KL**은 평균 차의 제곱:

$$
L_{t-1}=D_{\mathrm{KL}}\big(\mathcal N(\tilde{\boldsymbol\mu}_t,\sigma_t^2\mathbf I)\,\|\,\mathcal N(\boldsymbol\mu_\theta,\sigma_t^2\mathbf I)\big)=\frac{1}{2\sigma_t^2}\|\tilde{\boldsymbol\mu}_t-\boldsymbol\mu_\theta\|^2 .
$$

$\mathbf x_0=(\mathbf x_t-\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon)/\sqrt{\bar\alpha_t}$ 를 §2.2의 $\tilde{\boldsymbol\mu}_t$ 식에 대입하면 정리되어

$$
\tilde{\boldsymbol\mu}_t(\mathbf x_t,\mathbf x_0)=\frac{1}{\sqrt{\alpha_t}}\Big(\mathbf x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol\epsilon\Big).
$$

같은 대수형으로 $\boldsymbol\mu_\theta=\frac{1}{\sqrt{\alpha_t}}\big(\mathbf x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol\epsilon_\theta\big)$ 라 두면 $\mathbf x_t$ 항이 상쇄되어

$$
L_{t-1}=\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t(1-\bar\alpha_t)}\,\big\|\boldsymbol\epsilon-\boldsymbol\epsilon_\theta(\mathbf x_t,t)\big\|^2 .
$$

가중치 $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\to1$ 로 단순화한 것이 $L_{\text{simple}}$. (Ho et al.은 이 reweighting이 perceptual 품질에 더 낫다고 보고.) $\;\square$

### A.3 Tweedie: 최적 denoiser = posterior mean = score

**보조정리(주변 score = 조건부 score의 posterior 기대).**

$$
\nabla_{\mathbf x_t}\log q_t(\mathbf x_t)=\frac{\int \nabla_{\mathbf x_t} q(\mathbf x_t|\mathbf x_0)q(\mathbf x_0)\,\mathrm d\mathbf x_0}{q_t(\mathbf x_t)}=\mathbb E_{q(\mathbf x_0|\mathbf x_t)}\big[\nabla_{\mathbf x_t}\log q(\mathbf x_t|\mathbf x_0)\big].
$$

Gaussian kernel에서 $\nabla_{\mathbf x_t}\log q(\mathbf x_t|\mathbf x_0)=-(\mathbf x_t-\sqrt{\bar\alpha_t}\mathbf x_0)/(1-\bar\alpha_t)$ 이므로

$$
\nabla_{\mathbf x_t}\log q_t(\mathbf x_t)=\frac{\sqrt{\bar\alpha_t}\,\mathbb E[\mathbf x_0|\mathbf x_t]-\mathbf x_t}{1-\bar\alpha_t}\quad(\textbf{Tweedie}).
$$

MSE 최적 noise 예측기는 $\boldsymbol\epsilon_\theta^\star(\mathbf x_t)=\mathbb E[\boldsymbol\epsilon|\mathbf x_t]$ 이고, $\mathbf x_t=\sqrt{\bar\alpha_t}\mathbf x_0+\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon$ 에서 $\mathbb E[\mathbf x_0|\mathbf x_t]=(\mathbf x_t-\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon_\theta^\star)/\sqrt{\bar\alpha_t}$. 대입하면

$$
\nabla_{\mathbf x_t}\log q_t(\mathbf x_t)=-\frac{\boldsymbol\epsilon_\theta^\star(\mathbf x_t)}{\sqrt{1-\bar\alpha_t}}.
$$

즉 **denoising과 score estimation은 같은 양을 학습**한다(스케일만 다름). 이것이 §3.1의 정당화다. $\;\square$

### A.4 D3PM categorical posterior (Bayes, 벡터형)

row-stochastic $\mathbf Q_t$, one-hot $\mathbf x$. $q(\mathbf x_t|\mathbf x_{t-1})$ 를 $\mathbf x_{t-1}$ 의 함수로 보면 $\mathbf Q_t\mathbf x_t^\top$(열 선택). Bayes·Markov로

$$
q(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)=\mathrm{Cat}\!\Big(\frac{(\mathbf Q_t\mathbf x_t^\top)\odot(\mathbf x_0\bar{\mathbf Q}_{t-1})}{\mathbf x_0\,\bar{\mathbf Q}_t\,\mathbf x_t^\top}\Big),\quad(\odot:\ \text{elementwise}).
$$

**Absorbing 특수화.** marginal에서 $q(\mathbf x_t=\mathbf x_0|\mathbf x_0)=\alpha_t,\ q(\mathbf x_t=m|\mathbf x_0)=1-\alpha_t$. 위 식을 대입하면 토큰별로

- $\mathbf x_t\ne m$ (이미 unmask): posterior는 $\delta_{\mathbf x_t}$ (한 번 unmask면 그대로) — **carry-over**.
- $\mathbf x_t=m$ (mask): 두 점 분포
$$
q(\mathbf x_{t-1}=m\,|\,\mathbf x_t=m,\mathbf x_0)=\frac{1-\alpha_{t-1}}{1-\alpha_t},\qquad
q(\mathbf x_{t-1}=\mathbf x_0\,|\,\mathbf x_t=m,\mathbf x_0)=\frac{\alpha_{t-1}-\alpha_t}{1-\alpha_t}.
$$

$\;\square$

### A.5 (핵심) Masked diffusion NELBO → 가중 MLM cross-entropy

dLLM 손실의 정체를 밝히는 정리. **SUBS 파라미터화**: 모델은 어휘 위 분포 $\mathbf x_\theta^{(i)}(\mathbf x_t)$ 를 출력($m$ 에 질량 0)하고, reverse transition은 A.4의 두 점 분포에서 *알려진 $\mathbf x_0$ 를 예측 분포 $\mathbf x_\theta$ 로 대체*한 형태를 쓴다:

$$
p_\theta(\mathbf x_{t-1}^i=\cdot\,|\,\mathbf x_t)=\underbrace{\tfrac{1-\alpha_{t-1}}{1-\alpha_t}}_{[\text{MASK}]\text{ 유지}}\delta_m+\underbrace{\tfrac{\alpha_{t-1}-\alpha_t}{1-\alpha_t}}_{\text{unmask}}\,\mathbf x_\theta^{(i)}(\mathbf x_t).
$$

**(1) Unmask 토큰은 KL=0.** carry-over로 $q$ 와 $p_\theta$ 가 동일한 $\delta_{\mathbf x_t^i}$.

**(2) Masked 토큰의 per-token KL.** $q$ 와 $p_\theta$ 모두 $[\text{MASK}]$ 에 같은 가중 $\tfrac{1-\alpha_{t-1}}{1-\alpha_t}$ 를 두므로 그 성분은 상쇄. unmask 가중 $\tfrac{\alpha_{t-1}-\alpha_t}{1-\alpha_t}$ 에 대해 $q$ 는 전부 $\mathbf x_0^i$ 에, $p_\theta$ 는 $\mathbf x_\theta^{(i)}$ 로 분산:

$$
L_{t-1}^{(i)}=\frac{\alpha_{t-1}-\alpha_t}{1-\alpha_t}\,D_{\mathrm{KL}}\big(\delta_{\mathbf x_0^i}\,\|\,\mathbf x_\theta^{(i)}\big)=-\frac{\alpha_{t-1}-\alpha_t}{1-\alpha_t}\,\log\big\langle \mathbf x_\theta^{(i)}(\mathbf x_t),\,\mathbf x_0^{(i)}\big\rangle .
$$

**(3) 합산 + 연속시간 극한.** masked 위치 $\{i:\mathbf x_t^i=m\}$ 와 $t$ 에 대해 합하고 $\alpha_{t-1}-\alpha_t\to-\alpha_t'\,\mathrm dt$($\alpha$ 감소이므로 $-\alpha_t'>0$):

$$
\boxed{\ \mathcal L_{\text{NELBO}}=\int_0^1\frac{\alpha_t'}{1-\alpha_t}\,\mathbb E_{\mathbf x_t\sim q(\cdot|\mathbf x_0)}\Big[\sum_{i:\,\mathbf x_t^i=m}\log\big\langle \mathbf x_\theta^{(i)}(\mathbf x_t),\mathbf x_0^{(i)}\big\rangle\Big]\mathrm dt\ }
$$

$\alpha_t'\le0$, $\log\langle\cdot\rangle\le0$ 이므로 적분값은 양수(정당한 상한). 이는 정확히 **마스크된 위치에서의 시간가중 cross-entropy = weighted MLM**. (MD4: $t\to\alpha$ 재매개화로 스케줄 의존성을 제거 → SNR-invariance.) 이것이 LLaDA·Dream이 실제 최적화하는 목적의 본체다. $\;\square$

> **LLaDA 손실과의 일치.** LLaDA는 $t\sim\mathcal U[0,1]$ 로 각 토큰을 독립 마스킹하고 $-\frac1t\sum_i \mathbf 1[\mathbf x_t^i=m]\log p_\theta(\mathbf x_0^i|\mathbf x_t)$ 를 쓴다. 위 일반식에서 선형 스케줄 $\alpha_t=1-t$ 를 넣으면 $\tfrac{-\alpha_t'}{1-\alpha_t}=\tfrac1t$ 로 정확히 환원된다.

### A.6 SEDD: implicit ↔ denoising score entropy 동치 (Vincent 류)

**주장.** 미지의 주변비율 $r_t(\mathbf x,\mathbf y)=p_t(\mathbf y)/p_t(\mathbf x)$ 를 쓰는 implicit score entropy와, 알려진 조건부비율 $r_{t|0}(\mathbf x,\mathbf y)=p_{t|0}(\mathbf y|\mathbf x_0)/p_{t|0}(\mathbf x|\mathbf x_0)$ 를 쓰는 denoising score entropy는 $\theta$-의존 부분이 동일(상수 차).

**증명 핵심(posterior-averaged 조건부비율 = 주변비율).**

$$
\mathbb E_{\mathbf x_0|\mathbf x_t}\big[r_{t|0}(\mathbf x_t,\mathbf y)\big]=\int q(\mathbf x_0|\mathbf x_t)\frac{p_{t|0}(\mathbf y|\mathbf x_0)}{p_{t|0}(\mathbf x_t|\mathbf x_0)}\mathrm d\mathbf x_0
=\frac{1}{p_t(\mathbf x_t)}\int p_0(\mathbf x_0)p_{t|0}(\mathbf y|\mathbf x_0)\,\mathrm d\mathbf x_0=\frac{p_t(\mathbf y)}{p_t(\mathbf x_t)}=r_t(\mathbf x_t,\mathbf y),
$$

(둘째 등호는 $q(\mathbf x_0|\mathbf x_t)=p_{t|0}(\mathbf x_t|\mathbf x_0)p_0(\mathbf x_0)/p_t(\mathbf x_t)$ 대입). score entropy의 유일한 $\theta$-의존 항은 $s_\theta(\mathbf x_t)_{\mathbf y}$(선형)와 cross 항 $-\,(\text{ratio})\log s_\theta(\mathbf x_t)_{\mathbf y}$ 이며, cross 항의 기대에서 tower property로

$$
\mathbb E_{\mathbf x_t\sim p_t}\big[r_t(\mathbf x_t,\mathbf y)\,(\cdot)\big]=\mathbb E_{\mathbf x_0\sim p_0}\,\mathbb E_{\mathbf x_t|\mathbf x_0}\big[r_{t|0}(\mathbf x_t,\mathbf y)\,(\cdot)\big].
$$

즉 주변비율을 조건부비율로 바꿔도 최소화 대상이 동일. 이는 연속 diffusion의 Vincent(2011) denoising score matching 동치의 이산판이며, 학습을 *tractable*하게 만든다(미지의 $p_t$ 불필요). $\;\square$

### A.7 Block diffusion: 블록 크기 극한

블록 분해는 **부등식이 아니라 등식**(chain rule): $-\log p_\theta(\mathbf x)=-\sum_b\log p_\theta(\mathbf x^b|\mathbf x^{<b})$. 각 블록 조건부 NLL을 그 블록의 MDM NELBO로 상한.

- $L'=L$ (단일 블록): 정의상 전체 MDLM NELBO.
- $L'=1$: 각 블록이 토큰 하나. 길이-1 absorbing MDM에서 forward는 그 토큰을 마스킹, reverse는 $\mathbf x^{<b}$ 조건 하에 $[\text{MASK}]\to$ 토큰 복원만 가능. A.5의 적분이 단일 마스크 토큰에 대해 붕괴하여 $-\log p_\theta(x_b|\mathbf x_{<b})$ — **순수 AR cross-entropy**. 따라서 블록 크기가 AR↔diffusion을 잇는 연속 손잡이. (BD3-LM은 $L'=1$ 에서도 gradient 분산 때문에 AR과 격차가 생김을 지적하고, 분산 최소화 noise 과정으로 보정.) $\;\square$

### A.8 Reverse-time SDE: Fokker–Planck 스케치

forward $\mathrm d\mathbf x=\mathbf f\mathrm dt+g\,\mathrm d\mathbf w$ 의 밀도는 FP를 만족: $\partial_t p_t=-\nabla\!\cdot(\mathbf f p_t)+\tfrac12 g^2\Delta p_t$. 역시간 $\tau=T-t$ 의 과정이 같은 주변 $\{p_t\}$ 를 갖도록 drift를 $\tilde{\mathbf f}=\mathbf f-g^2\nabla\log p_t$ 로 두면, 역시간 FP가 forward의 그것과 일치함을 직접 확인할 수 있다(항 정리 시 $g^2\nabla\!\cdot(p_t\nabla\log p_t)=g^2\Delta p_t$ 가 부호를 맞춤). 엄밀한 일반 증명은 Anderson(1982). probability flow ODE는 확산항을 결정론적으로 흡수($\tfrac12 g^2$ 계수)한 동일 주변 흐름. $\;\square$

---

### 부록: 한눈에 보는 대응표 (연속 ↔ 이산)

| 연속 diffusion | 이산(masked) diffusion |
|---|---|
| Gaussian noise 주입 | 토큰 → `[MASK]` (absorbing) |
| $\boldsymbol\epsilon_\theta$ (noise 예측) | $p_\theta(x_0\mid x_t)$ (mask 예측기) |
| score $\nabla_{\mathbf x}\log p_t$ | concrete score (ratio $p_t(\mathbf y)/p_t(\mathbf x)$) |
| denoising score matching | score entropy (SEDD) / 가중 MLM (MDLM) |
| prior $\mathcal N(0,\mathbf I)$ | 전부 `[MASK]` 인 시퀀스 |
| reverse SDE/ODE 적분 | 반복적 언마스킹(remasking) |
| DDIM/PF-ODE 가속 | parallel decoding / 캐싱 / spec decoding |

*본 리포트의 모든 인용은 논문·공식 저장소 기준이며, 수식은 표준 정식화를 따른다. 일부 모델의 최신 버전(LLaDA 2.0 등)은 빠르게 갱신되므로 원 논문/릴리스를 함께 확인할 것.*
