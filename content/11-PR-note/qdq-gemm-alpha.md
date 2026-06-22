# QDQ Gemm Fusion과 α: ONNX Runtime PR #28131에 대한 소고

> 이 문서는 [mul.md](mul.md)의 GEMM 일반 이론을 전제로 한다.
> 특히 §4 (BLAS canonical form $C = \alpha AB + \beta C$), §9 (epilogue fusion), §11 (quantization scaling), §12 (Q/DQ pipeline)을 먼저 읽으면 좋다.

---

## 1. 배경: ONNX QDQ Gemm Fusion

ONNX Runtime은 양자화된 모델을 **QDQ format**으로 표현한다. fp32 graph에 `Quantize` (Q)와 `DeQuantize` (DQ) node를 명시적으로 끼워 넣어 fake-quant 형태로 표현한 뒤, graph optimizer가 다음 패턴을 인식해 **fused integer operator** `QGemm` 으로 rewrite한다.

```
DQ(A_q) -> Gemm(α, β, bias) -> Q(Y)
DQ(B_q) ----^
```

```
   ↓ (graph rewrite)

QGemm(A_q, s_A, z_A, B_q, s_B, z_B, C_fp32, s_Y, z_Y) -> Y_q
```

이 rewrite는 두 단계로 구성된다.

1. **selector**: subgraph pattern을 찾고, fusion이 의미적으로 안전한지 검사
2. **action**: 안전하면 fused op로 교체

selector가 보수적이지 못해 **수학적으로 등가가 아닌 케이스를 통과시키면**, kernel 자체에는 버그가 없어도 graph 변환만으로 wrong output이 발생한다. PR #28131이 다루는 것이 정확히 이 경우다.

---

## 2. Operator-level Math

### 2.1 ONNX Gemm 스펙

ONNX `Gemm` operator의 정의는:

$$
Y = \alpha \cdot (A B) + \beta \cdot C
$$

- $\alpha$: $AB$에만 곱해진다
- $\beta$: bias $C$에만 곱해진다
- 두 scalar는 **분리되어** 적용된다

이게 핵심이다. spec 자체는 `α`와 bias를 독립된 자유도로 분리해 두었다.

### 2.2 QGemm의 내부 연산

Fused QGemm은 다음을 hardware로 수행한다 (symmetric quantization 기준, asymmetric은 zero-point cross-term 추가).

$$
\text{Acc}_{\text{int32}} = \sum_k (A_q - z_A)(B_q - z_B) + C_{\text{int}}
$$

$$
Y_{\text{int}} = \text{round}\!\left(\frac{\alpha \cdot s_A s_B}{s_Y} \cdot \text{Acc}_{\text{int32}}\right) + z_Y
$$

표준 calibration은 bias를 미리 다음과 같이 정수화한다.

$$
C_{\text{int}} = \text{round}\!\left(\frac{\text{bias}_{\text{fp32}}}{s_A s_B}\right)
$$

이 layout의 장점:
- bias 덧셈이 **int32 domain에서** 일어나 fp 변환 없이 accumulator에 합쳐진다
- output scaling 한 번으로 GEMM 결과와 bias가 동시에 fp 도메인으로 환산된다
- $\alpha$는 외부 scaling factor에 합쳐져 추가 비용이 없다

### 2.3 함정: α가 bias에까지 곱해진다

위 식을 풀어 쓰면:

$$
Y_{\text{int}} - z_Y = \frac{\alpha s_A s_B}{s_Y} \sum_k (A_q - z_A)(B_q - z_B) \;+\; \frac{\alpha s_A s_B}{s_Y} \cdot C_{\text{int}}
$$

bias 항을 fp32 도메인으로 환산하면:

$$
\frac{\alpha s_A s_B}{s_Y} \cdot C_{\text{int}} = \frac{\alpha s_A s_B}{s_Y} \cdot \frac{\text{bias}_{\text{fp32}}}{s_A s_B} = \frac{\alpha \cdot \text{bias}_{\text{fp32}}}{s_Y}
$$

즉 fp 도메인에서 본 fused output은:

$$
Y_{\text{fp32}} = \alpha (A_{dq} B_{dq}) + \alpha \cdot \text{bias} \quad \text{(QGemm 실제 동작)}
$$

원래 ONNX Gemm spec이 의도한 의미는:

$$
Y_{\text{fp32}} = \alpha (A_{dq} B_{dq}) + \text{bias} \quad \text{(spec, } \beta = 1\text{)}
$$

`α ≠ 1` 이고 bias가 존재하면 두 식이 일치하지 않는다.

### 2.4 왜 이런 일이 일어나는가

원본 spec은 두 scalar (`α`, `β`)를 분리해 두었는데, fused INT8 pipeline에서는 다음 두 연산이 합쳐진다.

- **Gemm scaling** ($\alpha$): $AB$에만 곱해야 함
- **dequant scaling** ($s_A s_B$): **int32 accumulator 전체** (bias 포함)에 곱해짐

이 두 scaling이 하나의 곱셈으로 fuse되는 순간, $\alpha$를 bias로부터 분리할 수단이 사라진다. β = 1인 한 이 fused layout은 정확하지만, **β = 1을 가정한 채 α만 비-1로 두는 순간 layout이 깨진다.**

bias가 0이거나 α = 1이면 두 식이 일치한다 — issue #28130의 reporter가 관측한 현상과 정확히 부합한다.

---

## 3. 현재 PR (#28131) Scope

### 3.1 변경 사항

`GemmNodeGroupSelector::Check`에 `α == 1.0` 검사를 추가한다. 기존의 `β == 1.0` 검사와 대칭적이다.

```cpp
// 기존
if (beta != 1.0f) return false;

// 추가
if (has_bias && alpha != 1.0f) return false;
```

bias가 없으면 §2.3의 leakage가 발생할 대상 자체가 없으므로 `α ≠ 1`도 안전하다 — 그래서 `has_bias` 가드를 함께 걸었다.

### 3.2 효과

selector가 거부하면 fusion이 일어나지 않고, graph는 원래의 unfused path로 떨어진다.

```
DQ(A_q) -> Gemm(α, β, bias) -> Q(Y)     # fp32 Gemm으로 실행됨
DQ(B_q) ----^
```

이 경로에서는 Gemm이 fp32 도메인에서 실행되므로 `α`와 bias가 spec대로 분리 적용된다. 성능은 잃지만 정확성은 보장된다.

### 3.3 설계 원칙: Fail-safe Rewriting

이 PR이 따르는 원칙은 단순하다.

> **수학적 등가가 보존되지 않는 fusion은, 빠른 잘못된 답보다 느린 정확한 답으로 fallback해야 한다.**

이건 ONNX Runtime에만 적용되는 이야기가 아니다. TensorRT, TVM, MLIR, XLA, OpenVINO 모두 동일한 종류의 pattern matcher / rewriter를 갖고 있고, 같은 함정에 빠질 수 있다. 일반화하면:

> operator-level 수학적 등가성이 깨지는 fusion은, 아무리 매력적이어도 selector 단계에서 잘려야 한다.

### 3.4 테스트 전략

regression을 막기 위해 `QDQTransformerGemmTests`와 fastmath variant에 `alpha_not_one` parameter를 추가:

- `α = 2.0`, bias present → fusion 거부되어야 함 (op count 검사)
- `α = 2.0`, bias absent → fusion 허용되어야 함
- `α = 1.0`, bias present → fusion 허용되어야 함 (기존 동작 유지)

`main` branch에서는 새 케이스가 op count와 numerical comparison 모두에서 실패해야 한다 — 버그가 실제로 reproducible함을 증명한다.

---

## 4. Follow-up: Bias Absorption

PR #28131은 **최소한의 정확성 수정**이다. `α ≠ 1 + bias` 케이스를 fused path에 다시 살려내려면 별도의 graph transformation이 필요하다.

### 4.1 아이디어

graph-transform 시점에 bias를 미리 `α`로 scale한다.

$$
C_{\text{int}}^{\text{new}} = \text{round}\!\left(\frac{C_{\text{int}}}{\alpha}\right) = \text{round}\!\left(\frac{\text{bias}_{\text{fp32}}}{\alpha \cdot s_A s_B}\right)
$$

이 bias를 들고 fused kernel을 그대로 돌리면:

$$
\frac{\alpha s_A s_B}{s_Y}\!\left(\sum_k (\ldots) + C_{\text{int}}^{\text{new}}\right) = \frac{\alpha s_A s_B}{s_Y}\sum_k(\ldots) + \frac{\text{bias}_{\text{fp32}}}{s_Y}
$$

가 되어 spec과 정확히 일치한다. 즉 **bias 안에 `1/α`를 미리 흡수**하여 fused layout의 제약을 회피한다.

### 4.2 안전성 검사

이 변환은 무조건 안전하진 않다. 다음을 정적으로 확인해야 한다.

#### Precision Loss

$C_{\text{int}}^{\text{new}} = \text{round}(C_{\text{int}} / \alpha)$ 에서 추가적인 rounding error가 발생한다.

- $|\alpha| > 1$: 절댓값이 줄어드는 방향 → relative error 증가
- $|\alpha| < 1$: 절댓값이 커지는 방향 → relative error는 작아지지만 overflow risk

#### Overflow

`bias_fp32 / (α · s_A s_B)`가 int32 범위 $[-2^{31}, 2^{31})$ 안에 들어가야 한다.

특히 `|α| < 1`일 때는 bias가 크게 amplify되므로, transform 시점에 max bias 값과 `α`로부터 overflow 가능성을 사전 계산해 안전하지 않으면 transform을 포기해야 한다.

#### Zero-point 상호작용

Asymmetric quantization에서는 cross-term이 등장한다.

$$
A_q B_q = \sum_k (A_q - z_A)(B_q - z_B) + z_A \sum_k B_q + z_B \sum_k A_q - K z_A z_B
$$

이 중 일부 항은 보통 **pre-computed bias offset**으로 흡수되어 `C_int`에 합쳐진다. `α`로 bias를 나누는 변환은 이 pre-computed offset에도 함께 적용되어야 한다 — 일관성 깨지면 또 다른 silent bug.

#### Constant-folding 가능 여부

`α`, bias, scale이 모두 graph 초기화 시점에 알려진 constant여야 transform이 의미가 있다. dynamic input이면 transform 자체가 불가능하다 (그런 경우는 §3의 fallback으로 남긴다).

### 4.3 Decision Tree

```
α == 1?
├── Yes → 기존 path, fusion 허용
└── No
    ├── bias 없음? → fusion 허용 (PR #28131의 has_bias 가드)
    └── bias 있음
        ├── overflow 안전? + precision 손실 허용 범위?
        │   ├── Yes → bias absorption 적용 후 fusion
        │   └── No  → fallback (unfused fp32 Gemm)
```

### 4.4 별도 PR로 분리한 이유

bias absorption은 다음 측면에서 #28131과 별개의 review가 필요하다.

- **수치적 검증**: rounding error, overflow boundary에 대한 정량 분석
- **테스트 커버리지**: `α`, bias scale, weight scale의 cross-product 케이스
- **표준 영향**: ONNX QGemm spec과의 일치 여부 — `α`를 graph attribute에서 흡수해 버리는 것이 표준 호환인지

따라서 #28131은 **fail-safe (correctness guard)** 만 담고, follow-up PR이 **performance recovery (fused path 복구)** 를 담당하는 깔끔한 분리가 된다.

---

## 5. 정리

이 case가 보여주는 일반적 교훈은 다음과 같다 (자세한 이론적 배경은 [mul.md](mul.md) 참조).

| 관점 | 내용 | mul.md 참조 |
|---|---|---|
| BLAS canonical form | $\alpha$와 $\beta$가 분리된 자유도라는 것은 단순한 일반화가 아니다 | §4 |
| Epilogue fusion | scalar를 어디에 곱할지가 layout을 결정한다 | §9 |
| Quantization scaling | $\alpha = s_A s_B$로 BLAS의 α 자리를 점거하면 자유도가 소실된다 | §11.3 |
| Graph rewriter | rewrite가 의미를 보존하지 않으면 kernel과 무관하게 wrong output | §14 |

GEMM에서 `α`는 1바이트짜리 scalar처럼 보이지만, 그것이 fused INT8 pipeline의 어느 layer에 곱해지느냐가 정확성을 결정한다. modern matrix multiplication이 단순한 linear algebra가 아니라 **scaling layout problem**이라는 말의 구체적인 예다.

---

## 참고

- ONNX Runtime PR #28131: Reject QDQ Gemm→QGemm fusion when alpha != 1 with bias
- ONNX Runtime Issue #28130: original bug report
- 이론적 배경: [mul.md](../../10-Theoretical-stuff/mul.md)
