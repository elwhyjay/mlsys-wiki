# 부동소수점 산술 — MLsys 엔지니어용 레퍼런스

> David Goldberg, *What Every Computer Scientist Should Know About Floating-Point Arithmetic*, ACM Computing Surveys 23(1), 1991.
> 표기: β = base, p = precision(significand 자릿수), e = exponent, ε = machine epsilon.

이 문서는 논문의 핵심을 MLsys(특히 CUDA 커널 / 추론 / 양자화 / mixed precision) 관점에서 재정리하고, 두 가지 주제 — **Kahan summation 증명**과 **컴파일러/옵티마이저 위험 사례** — 를 GPU 커널 예제와 함께 깊게 다룬다.

---

## 0. 한 페이지 요약

| 주제 | 논문 결과 | MLsys 함의 |
|---|---|---|
| Machine epsilon | `ε = (β/2)β^(-p)`, 반올림 relative error 상한 | 포맷별 누적 오차 정량화 |
| Wobble | 1/2 ulp의 relative error는 최대 β배 변동 | β=2가 분석에 유리, ulp↔relative 환산 주의 |
| Guard digit | 없으면 뺄셈 error ≤ β−1, 있으면 < 2ε (Thm 1,2) | 거의 모든 HW가 보장, 가정해도 됨 |
| Catastrophic cancellation | 근접 값 뺄셈이 앞선 오차를 드러냄 | softmax/LayerNorm/판별식 재작성의 근거 |
| Sterbenz lemma | `y/2 ≤ x ≤ 2y`면 `x−y` 정확 (Thm 11) | 보정 알고리즘의 핵심 보조정리 |
| Associativity 불성립 | `(x+y)+z ≠ x+(y+z)` | 병렬 reduction 비결정성, fast-math 위험 |
| Kahan summation | 항당 섭동 2ε (naive는 nε) (Thm 8) | reduction 커널 정밀도 전략 |
| Double accumulation | ε → ε² (`nε² ≪ ε`) | fp32 accumulator가 1순위, Kahan은 차선 |
| single×single→double | 곱은 single multiplier보다 약간 비쌀 뿐 | Tensor Core: 낮은 정밀도 곱 + 높은 정밀도 누적 |
| Round-trip | fp32=9자리, fp64=17자리 (Thm 15) | 체크포인트/로그 정밀도 보존 |
| Double rounding | 중간 정밀도 경유 시 오류 발생 가능 | TF32/양자화 round-trip 함정 |
| Denormal/FTZ | gradual underflow가 `x=y ⇔ x−y=0` 보장 | denormal은 성능 함정, FTZ로 0 처리 |

### 포맷별 상수 (β=2)

| 포맷 | exp / mantissa(저장) | p (hidden 포함) | unit roundoff u = 2^(1-p) | dynamic range 비고 |
|---|---|---|---|---|
| fp64 | 11 / 52 | 53 | 2^-52 ≈ 2.22e-16 | 표준 정밀 |
| fp32 | 8 / 23 | 24 | 2^-23 ≈ 1.19e-7 | 표준 누적 정밀 |
| tf32 | 8 / 10 | 11 | 2^-10 ≈ 9.77e-4 | exp=fp32, mantissa만 절삭 |
| fp16 | 5 / 10 | 11 | 2^-10 ≈ 9.77e-4 | exp 5비트 → 좁은 range, loss scaling 필요 |
| bf16 | 8 / 7 | 8 | 2^-7 ≈ 7.81e-3 | exp=fp32 → range 넓음, 학습 선호 |
| fp8 e4m3 | 4 / 3 | 4 | 2^-3 ≈ 1.25e-1 | 추론/가중치 |
| fp8 e5m2 | 5 / 2 | 3 | 2^-2 ≈ 2.5e-1 | range 우선, gradient |

> 논문의 `ε = (β/2)β^(-p) = 2^-p`는 round-to-nearest의 relative error **상한**, 위 표의 `u = 2^(1-p)`는 nearest 반올림의 **최대 상대 간격**(unit roundoff). 둘은 정확히 2배 차이(`u = 2ε`)이며 문헌마다 ε/u를 혼용한다. 분석 시 어느 정의인지 반드시 명시할 것. 여기서는 더 널리 인용되는 u 기준으로 적었다.

핵심: 길이 `n` reduction의 naive 오차는 대략 `n·ε`. bf16(ε≈3.9e-3)으로 n=4096 합을 돌리면 상대오차가 O(1) 수준까지 갈 수 있다 → **accumulator는 반드시 더 높은 정밀도로**.

---

## 1. Rounding Error 핵심

### 1.1 포맷과 hidden bit
- `±d.dd…d × β^e`, significand는 p자리. 선행 0이 아니면 normalized → 표현 유일.
- β=2에서 최상위 비트는 항상 1 → 저장 안 함(**hidden bit**). 그래서 fp32는 23비트 저장이지만 p=24.
- bf16이 fp16과 같은 16비트인데 학습에 선호되는 이유: exponent 8비트(=fp32와 동일 dynamic range)를 유지하고 mantissa를 희생. fp16은 exponent 5비트라 overflow/underflow가 빈번 → loss scaling 필요.

### 1.2 ulp vs relative error, wobble
- ulp: 마지막 자리 단위. nearest 반올림 = 최대 1/2 ulp.
- relative error: 차이/실제값. 상한이 ε.
- **wobble**: 동일 relative error가 ulp로는 최대 β배 차이. β=2일 때 오차 bound가 가장 tight → IEEE가 β=2를 강제한 이유 중 하나. (큰 base는 effective precision이 `4p−3`까지 하락; IBM/370의 β=16 사례.)

### 1.3 Guard digit
- **Theorem 1**: guard digit 없이 p자리로 빼면 relative error 최대 `β−1`. β=2면 결과 전 자릿수 오염.
- **Theorem 2**: guard digit 1개면 `< 2ε`. 비용은 adder 1비트(2% 미만). 현대 HW는 사실상 보장.

### 1.4 Cancellation (가장 실무적)
- **Catastrophic**: 피연산자가 이미 반올림 오차를 가질 때 근접값 뺄셈. 뺄셈이 오차를 *드러낸다*. 예: 판별식 `b²−4ac` (70 ulp 오차).
- **Benign**: 정확한 값끼리 뺄셈, guard digit 있으면 < 2ε.
- 재작성: `x²−y² → (x−y)(x+y)`, 근의 공식 켤레곱(식 5), Heron → Kahan 식(7, Thm 3: ≤11ε), `ln(1+x)` 트릭(Thm 4: ≤5ε).

### 1.5 Exactly rounded & splitting
- **Theorem 6 (Dekker/Kahan splitting)**: exactly rounded면 `x = x_h + x_l`로 정확 분할 → 두 수의 곱을 합으로 정확히 표현. **error-free transformation**과 FMA 정밀도의 토대.
- **Theorem 11 (Sterbenz)**: guard digit 있고 `y/2 ≤ x ≤ 2y`면 `x⊖y` **정확**. 보정 알고리즘 증명의 핵심.

---

## 2. Kahan Summation — 증명까지 끝까지

### 2.1 문제
naive 합 `s_i = (1+δ_i)(s_{i-1} + x_i)`의 오차:

```
computed Σx_j = Σ x_j(1 + δ_j'),   |δ_j'| ≤ (n - j)ε
```

즉 **먼저 더해진 항일수록 더 많이 섭동**(최대 `nε`). 항 수가 수천이면 치명적.

### 2.2 알고리즘 (논문 Thm 8)

```
S = x[1]
C = 0
for j = 2..N {
    Y = x[j] - C        # 직전 루프에서 잃은 비트를 되돌림
    T = S + Y           # 큰 합. Y의 하위 비트가 흡수되어 사라짐
    C = (T - S) - Y     # 사라진 하위 비트를 복원 → 다음 루프 보정값
    S = T
}
```

### 2.3 직관: error-free transformation (2Sum)
핵심은 `(T, C)`가 `S+Y`의 **에러 없는 분해**라는 점이다. 즉 수학적으로 `S + Y = T + (-C)`가 *정확히* 성립(`|S|≥|Y|` 조건 하의 FastTwoSum/Dekker, Knuth 1981 §4.2.2 Thm C). `T`는 반올림된 합, `-C`는 반올림으로 잃은 하위 비트.

- `T = S ⊕ Y` : 반올림된 합. `Y`의 하위 비트가 `S`의 큰 지수에 흡수되어 소실.
- `T ⊖ S` : `T`의 상위 비트(=`Y`의 상위부)를 복원.
- `(T ⊖ S) ⊖ Y` : 거기서 `Y`를 빼면 **소실됐던 하위 비트**가 부호 반전되어 `C`에 남음.
- 다음 루프에서 `x[j] ⊖ C`로 이 잔차를 더해줌 → 누적 오차가 자라지 않음.

각 뺄셈이 Sterbenz/benign cancellation 영역에서 일어나 정확하다는 것이 보장의 뿌리.

### 2.4 형식 증명 (Appendix 재구성)

`s_0 = c_0 = 0`, 그리스 문자는 모두 `|·| ≤ ε`. 정의:

```
Y_k = x_k ⊖ c_{k-1} = (x_k - c_{k-1})(1 + η_k)
S_k = s_{k-1} ⊕ Y_k = (s_{k-1} + Y_k)(1 + σ_k)
C_k = (S_k ⊖ s_{k-1}) ⊖ Y_k = [ (S_k - s_{k-1})(1 + γ_k) - Y_k ](1 + δ_k)
```

관심사는 최종 `s_n`에서 `x_1`의 계수. 논문은 `x_1`의 계수를 `s_k − c_k`와 `c_k`에서 추적하는 게 더 쉽다는 것을 이용한다. `x_i (i>1)` 항을 무시하고 전개.

**k=1:**
```
c_1 = (s_1(1+γ_1) - Y_1)(1+δ_1) = x_1(σ_1 + γ_1 + σ_1γ_1)(1+δ_1)(1+η_1)
s_1 - c_1 = x_1[1 - γ_1 - σ_1δ_1 - σ_1γ_1 - δ_1γ_1 - σ_1γ_1δ_1](1+η_1)
```
`x_1`의 계수를 각각 `C_1`, `S_1`이라 하면:
```
C_1 = 2ε + O(ε²)
S_1 = 1 + η_1 - γ_1 + 4ε² + O(ε³)
```

**일반 점화 (x_i, i>1 무시):**
```
s_k = (s_{k-1} + Y_k)(1+σ_k) = [(s_{k-1} - c_{k-1}) - η_k c_{k-1}](1+σ_k)
```
정리하면 `S_k`, `C_k` (계수)에 대한 점화가 ε² 차수까지:
```
C_k = (σ_k + O(ε²)) S_{k-1} + (-γ_k + O(ε²)) C_{k-1}
S_k = ((1 + 2ε² + O(ε³)) S_{k-1} + (2ε + O(ε²)) C_{k-1}
```

**귀납으로:**
```
C_k = σ_k + O(ε²)
S_k = 1 + η_1 - γ_1 + (4k + 2)ε² + O(ε³)
```

**마무리:** `x_{n+1}=0`, 첨자 `n+1` 그리스 문자=0으로 두면 `s_{n+1} = s_n - c_n`. `s_n`에서 `x_1`의 계수는 `s_{n+1}`의 계수보다 작고, 그 값은
```
S_n = 1 + η_1 - γ_1 + (4n + 2)ε²  =  1 + 2ε + O(nε²)
```

따라서 최종:

```
computed Σ = Σ x_j (1 + δ_j) + O(Nε²) Σ|x_j|,    |δ_j| ≤ 2ε
```

**결론(중요):** 각 항의 섭동이 `nε` → **2ε**로 줄었다. 단, 두 배 정밀도(`ε → ε²`, `nε² ≪ ε`)가 가능하면 그게 Kahan보다도 우월. 우선순위:

```
double(fp32+) accumulator  >  Kahan(같은 정밀도)  >  naive
```

### 2.5 GPU 커널 예제

#### (a) 가장 흔한 해법 — fp32 accumulator (double accumulation 논리)
bf16 입력을 fp32로 누적. 곱도 fp32로 승격해서 한다(아래 3.2 FMA 절 참조).

```cuda
// bf16 벡터의 합을 fp32 accumulator로. 입력 정밀도보다 누적 정밀도를 높이는 게 1순위.
__global__ void reduce_bf16_fp32acc(const __nv_bfloat16* __restrict__ x,
                                    float* __restrict__ out, int n) {
    float acc = 0.0f;                       // ε_fp32 ≈ 6e-8, ε_bf16 ≈ 4e-3
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += gridDim.x * blockDim.x)
        acc += __bfloat162float(x[i]);      // bf16 → fp32 후 누적

    // 워프 내 리덕션 — 순서가 고정이라 결정적(아래 4절 비결정성과 대비)
    for (int o = 16; o > 0; o >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, acc);  // atomicAdd는 순서 비결정!
}
```

#### (b) accumulator를 못 올릴 때 — 디바이스 Kahan
가장 높은 가용 정밀도(fp32)에서 더 줄이고 싶거나, fp32 합 자체가 너무 길 때.

```cuda
// 디바이스 Kahan compensated summation (스레드 로컬 부분합)
struct KahanAcc { float s = 0.f, c = 0.f; };

__device__ __forceinline__ void kahan_add(KahanAcc& k, float x) {
    float y = x - k.c;
    float t = k.s + y;
    k.c = (t - k.s) - y;   // ★ 컴파일러가 0으로 접으면 알고리즘 파괴 (3.2 참조)
    k.s = t;
}
```

> ⚠️ `(t - s) - y`는 대수적으로 0이다. `-ffast-math`/`--use_fast_math`/`-fassociative-math`가 켜지면 컴파일러가 이를 0으로 단순화해 **Kahan을 naive로 되돌린다.** 보호책은 2.6 참조.

#### (c) Flash Attention의 online softmax — 같은 정신의 다른 적용
긴 합을 한 번에 누적하지 않고, running max로 재스케일하며 안전하게 누적. overflow 회피 + catastrophic cancellation 회피.

```cuda
// online softmax 한 블록 갱신 (개념 코드)
// m: running max, l: running denom(sum of exp), acc: running weighted value
__device__ void online_softmax_update(float& m, float& l, float* acc,
                                       const float* scores, const float* v,
                                       int d, int len) {
    for (int j = 0; j < len; ++j) {
        float m_new = fmaxf(m, scores[j]);          // max-subtraction = overflow/cancellation 방지
        float corr  = __expf(m - m_new);            // 기존 누적분 재스케일 계수
        float p     = __expf(scores[j] - m_new);
        l = l * corr + p;                           // 분모도 같은 스케일로 보정
        for (int t = 0; t < d; ++t)
            acc[t] = acc[t] * corr + p * v[j * d + t];
        m = m_new;
    }
}
```
여기서 `scores[j] - m_new`(benign)와 재스케일 `corr`은 정확히 1.4절의 "catastrophic을 benign으로" 재작성과 같은 논리다.

### 2.6 Kahan/보정 코드를 컴파일러로부터 지키기
- 핵심 식만이라도 `volatile` 또는 컴파일러 배리어로 재결합 금지.
- CUDA: 해당 TU를 `--fmad=true`는 두되 `--use_fast_math`는 끄고, 필요한 곳만 intrinsic 사용.
- 또는 `__fadd_rn`/`__fsub_rn`(round-to-nearest, FMA contraction/재결합 비대상)으로 명시:

```cuda
__device__ __forceinline__ void kahan_add_protected(KahanAcc& k, float x) {
    float y = __fsub_rn(x, k.c);
    float t = __fadd_rn(k.s, y);
    k.c = __fsub_rn(__fsub_rn(t, k.s), y);  // 명시 RN 연산은 재결합/contraction 안 됨
    k.s = t;
}
```

---

## 3. Optimizer / 컴파일러 위험 사례 — 확장판

논문의 핵심 명제: 부동소수점은 **`a ⊕ b = (a+b)(1+δ), |δ|≤ε`** 라는 약한 보장만 만족한다. 실수의 결합/분배 법칙을 가정하면 이 위에 세운 알고리즘(Kahan 등)이 무너진다. 실수 의미론은 구현 비용이 비현실적(n비트 곱→2n비트가 무한 증식)이라 채택 불가.

### 3.1 논문이 든 사례
| 변환 | 왜 틀리나 |
|---|---|
| `x/10.0 → x*0.1` | 0.1이 이진수로 부정확. `x/2.0 → x*0.5`만 안전(2의 거듭제곱) |
| `x*y - x*z → x*(y-z)` | `y≈z`면 값이 크게 달라짐(catastrophic) |
| ε 측정 루프 `eps+1>1 ⇔ eps>0` | 완전히 다른 값 계산(ε 대신 β^e_min) |
| Kahan `C=(T-S)-Y → 0` | 알고리즘 무용지물 |
| `1.0E-40` 컴파일타임 변환 | inexact/underflow flag, rounding mode 의존성 변경 |
| `A*B` CSE (사이에 rounding mode 변경) | 같은 식 아님 |
| `x==x → true` | NaN에서 거짓 |
| `-x → 0-x` | `-0`에서 틀림 |
| `x<y → !(x>y)` | NaN은 unordered |

안전한 변환: `x+y=y+x`, `2x=x+x`, `1*x=x`, `0.5*x=x/2` (단 일부 CDC/Cray 예외), instruction scheduling, inline.

### 3.2 FMA contraction — 논문 이후 가장 큰 추가 함정
IEEE 754-2008은 `a*b+c`를 **하나의** 반올림으로 계산하는 FMA를 허용/장려한다. 컴파일러는 `a*b+c`를 FMA로 **contract**할 수 있고(중간 곱이 반올림되지 않음), 이는 결과를 바꾼다.

```cuda
// 같은 수식, 다른 결과:
float r1 = a * b + c;            // 컴파일러가 fma(a,b,c)로 contract 가능 → 곱이 무한정밀
float r2 = __fmul_rn(a, b);      // 곱을 명시적으로 반올림
       r2 = __fadd_rn(r2, c);    // 그 다음 덧셈 → r1과 다를 수 있음
```

- **정밀도 측면 장점:** FMA는 곱의 반올림 오차를 없애 dot product/Horner를 더 정확하게 만든다. 2Prod(error-free 곱)도 `fma(a,b,-a*b)`로 잔차를 정확히 뽑는다.
- **재현성 측면 위험:** contraction on/off, 디바이스마다 결과가 비트 단위로 달라짐. 디바이스 간 mismatch 디버깅의 단골 원인.
- 제어: nvcc `--fmad=true|false`, GCC/Clang `-ffp-contract=fast|on|off`.

```cuda
// 2Prod: 곱과 그 반올림 잔차를 정확히 분리 (error-free transformation)
__device__ void two_prod(float a, float b, float& p, float& e) {
    p = a * b;
    e = __fmaf_rn(a, b, -p);   // e = a*b - p  (정확). FMA 없으면 불가능했던 트릭
}
```

### 3.3 `-ffast-math` / `--use_fast_math` 분해
플래그가 끄는 보장들(각각이 위 알고리즘을 깬다):

- `-fassociative-math` : 재결합 허용 → **병렬 reduction 순서 변동, Kahan 파괴**.
- `-ffinite-math-only` : NaN/Inf 없다고 가정 → **attention mask의 `-inf`가 무력화**, NaN 전파 사라짐.
- `-fno-signed-zeros` : `-0` 무시 → log/branch cut/부호 복원 깨짐.
- `-freciprocal-math` : `a/b → a*(1/b)` → 추가 반올림.
- GPU `__expf/__sinf/rsqrtf` 등 저정밀 intrinsic, **FTZ on(denormal→0)**, division/sqrt 비-정확반올림.

```cuda
// fast-math에서 division이 reciprocal 근사로 바뀜 — softmax 분모처럼 정밀도 민감한 곳 위험
float p = e / denom;            // fast-math: e * rcp_approx(denom), 1~2 ulp 오차
float p_exact = __fdiv_rn(e, denom);  // 정확 반올림 division 강제
```

추론에서 throughput을 위해 fast-math를 켜는 건 흔하지만, **softmax 분모, LayerNorm 분산, loss/gradient reduction, mask 처리**는 빠르고 위험한 경로다. 이런 커널만 별도 TU로 분리해 fast-math 제외하는 패턴을 권장.

### 3.4 Reassociation과 SIMD horizontal sum
자동 벡터화는 reduction을 lane별 부분합으로 쪼갠 뒤 마지막에 합치는 **다른 트리**로 바꾼다. associativity 가정이 들어가므로 `-fassociative-math` 없이는 보통 막혀 있고, 켜면 결과/비결정성이 바뀐다. GPU에서는 warp shuffle 트리 자체가 이미 "다른 순서"라서 CPU 결과와 일치하지 않는 게 정상이다.

### 3.5 양자화/dequant에서의 함정 (Jay의 dequant 커널 맥락)
- `x * scale`의 반올림: scale이 이진수로 부정확하면 dequant마다 1/2 ulp 유입. per-channel scale 곱은 FMA로 묶어 곱 반올림 제거 권장.
- `round(x / scale)`를 `round(x * (1/scale))`로 바꾸면 reciprocal 추가 오차 → zero-point 경계에서 양자화 빈이 어긋날 수 있음.
- AWQ/GPTQ류에서 누산은 fp32, 역양자화 곱은 FMA, 최종 비교/clamp는 NaN-aware로.

---

## 4. 추가로 중요한 요소 (GPU 커널과 함께)

### 4.1 single×single→double 곱 = Tensor Core의 원형 (논문 3.1)
논문: 두 single 피연산자로 double 곱을 내는 명령은 single multiplier보다 약간 비쌀 뿐이고, 근의 공식·iterative improvement·긴 inner product에 매우 유용. → **Tensor Core는 fp16/bf16 곱 + fp32 누적**으로 이 모티프를 그대로 구현.

```cuda
#include <mma.h>
using namespace nvcuda::wmma;
// A,B: __half (fp16/bf16), C: float — "낮은 정밀도 곱, 높은 정밀도 누적"
fragment<matrix_a, 16,16,16, half, row_major> a;
fragment<matrix_b, 16,16,16, half, col_major> b;
fragment<accumulator, 16,16,16, float> acc;
fill_fragment(acc, 0.0f);
load_matrix_sync(a, Aptr, 16);
load_matrix_sync(b, Bptr, 16);
mma_sync(acc, a, b, acc);     // 내부 누적은 fp32 → 논문 3.1의 정확한 실현
```

### 4.2 TF32와 double rounding (논문 4.2)
TF32는 입력을 8exp/10mantissa로 잘라 곱하고 fp32로 누적. fp32 → TF32 → fp32 round-trip은 **double rounding** 위험(논문 12.51→12.5→12, 정답 13 예시와 동형). 정확한 round-trip이 필요하면 TF32 경유를 피하고 full fp32 path 사용.

### 4.3 Round-trip 자릿수 (논문 Thm 15)
- fp32 복원: **9 십진 자리**, fp64: **17 자리**. 8자리로는 fp32 복원 불가(393,216 이진수를 240,000 십진수로 못 담음).
- 체크포인트 직렬화, config 파싱, 로그 출력에서 이 자릿수 미만이면 비트 손실.

```python
# fp32 안전 직렬화
import struct
f = struct.unpack('f', struct.pack('f', x))[0]
s = repr(f)            # Python repr는 round-trip 보장(최소 자릿수)
# 또는 명시적으로
s = f"{x:.9g}"         # fp32: 9, fp64: 17
```

### 4.4 Denormal / FTZ — 성능 함정 (논문 2.2.4)
- gradual underflow가 `x=y ⇔ x−y=0`(식 10)을 보장 → `if(x!=y) z=1/(x-y)`의 spurious div-by-zero 방지.
- 그러나 대부분 고성능 HW는 denormal을 trap→SW 시뮬레이션 → denormal 빈발 시 **현저히 느려짐**(논문 명시).
- GPU 커널은 `--ftz=true`(또는 `-ftz=true`)로 denormal을 0으로 flush해 성능 안정화. 단 작은 gradient/activation이 0으로 죽는 underflow와 트레이드오프.

```cuda
// __expf 같은 intrinsic + FTZ on이면 작은 값이 0으로. 학습 안정성 영향 체크 필요.
// 컴파일: nvcc --ftz=true  vs  --ftz=false
```

### 4.5 signed zero / ∞ / NaN 전파 (논문 2.2)
- `-inf` mask: `score + (-inf) = -inf`, `exp(-inf)=0` → causal/padding mask가 정확히 작동. `-ffinite-math-only`면 이 보장이 깨진다.
- `c/0 = ±∞`(c≠0), `0/0 = NaN`. NaN은 연산 전파되어 디버깅 신호로 유용.
- `+0 == -0`이지만 `1/(+0)=+∞`, `1/(-0)=-∞`. 부호 복원·log·복소 sqrt branch cut에 필요.

```cuda
// NaN/Inf 워치독 — 커널 출력 검증
__global__ void check_finite(const float* x, int n, int* bad) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n && !isfinite(x[i])) atomicExch(bad, 1);  // NaN/Inf 위치 추적
}
```

### 4.6 병렬 reduction 비결정성 — associativity 불성립의 직접 결과
- `(x+y)+z ≠ x+(y+z)`이므로 **reduction 트리/순서가 다르면 결과가 비트 단위로 다르다.**
- `atomicAdd(float)`는 도착 순서가 비결정 → 실행마다 결과 변동. all-reduce 알고리즘(ring vs tree)이 다르면 노드 간 결과도 달라짐.
- 결정적 결과가 필요하면: 고정 트리 reduction, deterministic 알고리즘 강제(`torch.use_deterministic_algorithms(True)`), atomic 대신 정렬된 2단계 reduction.

```cuda
// 비결정적: 순서 보장 없음
atomicAdd(out, partial);

// 결정적 대안: 고정 인덱스로 부분합 기록 후 별도 커널에서 고정 순서 합산
partial_sums[block_id] = block_reduce(partial);   // 1단계
// 2단계: 단일 스레드/고정 트리로 partial_sums 합산 → 매 실행 동일
```

### 4.7 Split-K GEMM 누적 정밀도
Split-K는 K 차원을 여러 블록이 나눠 부분곱을 낸 뒤 합친다. 부분합 결합 순서가 달라 비결정적이고, 부분합을 낮은 정밀도로 합치면 정밀도 손실. fp32 partial + 결정적 결합이 안전. cuBLAS/CUTLASS의 `splitK` 옵션 사용 시 reduction 정밀도/순서 설정 확인.

---

## 5. 체크리스트 (커널 작성/리뷰 시)

- [ ] accumulator 정밀도 ≥ 입력 정밀도? (긴 reduction은 fp32+ 필수)
- [ ] 못 올리면 Kahan/compensated, 그리고 컴파일러가 보정식을 접지 않게 보호(`__fadd_rn` 등)?
- [ ] softmax/LayerNorm/loss/mask 커널이 fast-math에서 분리되어 있나?
- [ ] LayerNorm 분산은 Welford(또는 `(x-μ)`)인가, naive `E[x²]-E[x]²`(catastrophic)인가?
- [ ] FMA contraction on/off가 재현성 요구와 일치하나? (`--fmad`, `-ffp-contract`)
- [ ] mask에 `-inf` 쓰는데 `-ffinite-math-only`가 켜져 있지 않나?
- [ ] denormal 빈발로 성능 저하? FTZ 트레이드오프 검토했나?
- [ ] reduction/atomic 비결정성이 허용되나? 결정성 필요 시 고정 트리?
- [ ] dequant scale 곱을 FMA로 묶었나? reciprocal 치환으로 빈 경계가 어긋나지 않나?
- [ ] 직렬화/로그가 fp32=9, fp64=17 자리를 지키나? (round-trip)
- [ ] TF32 경유로 double rounding이 발생하는 정확도 민감 경로가 있나?

---

## 6. 정리된 Theorem 인덱스

| Thm | 내용 | 활용 |
|---|---|---|
| 1 | guard digit 없는 뺄셈 error ≤ β−1 | guard digit 필요성 |
| 2 | guard digit 뺄셈/덧셈 < 2ε | 기본 연산 정확성 가정 |
| 3 | Heron 재작성(7) ≤ 11ε | cancellation 재작성 |
| 4 | `ln(1+x)` 트릭 ≤ 5ε | 작은 x 정확 log |
| 5 | round-to-even은 drift 없음 | 기본 rounding mode |
| 6 | splitting으로 곱=합 정확 표현 | error-free transform, FMA |
| 7 | β=2에서 `(m⊘n)⊗n=m` | exactly rounded 보장 |
| 8 | Kahan summation, 항당 2ε | reduction 정밀도 |
| 9,10 | 뺄셈/덧셈 ≤ 2ε (Thm 2 구성) | |
| 11 | Sterbenz: `y/2≤x≤2y`면 `x⊖y` 정확 | 보정 알고리즘 핵심 |
| 12,13 | Heron/μ(x) 오차 bound | Thm 3,4 증명 |
| 14 | splitting 정확성 | Thm 6 기반 |
| 15 | fp32=9, fp64=17 round-trip | 직렬화 |
