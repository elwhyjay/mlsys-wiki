> 블로그 출처: https://leimao.github.io/article/CuTe-Layout-Algebra/ 이 글은 Lei Mao의 글이며, 저자의 전재 허가를 받았다. 이후 Lei Mao의 CUDA 관련 Blog를 몇 편 더 전재할 예정이고, 이는 하나의 완결된 칼럼이다. Blog는 비교적 이른 시기의 CUDA 아키텍처부터 현재 최신 CUDA 아키텍처까지 다루며, 실용적인 엔지니어링 기법, 저수준 명령어 분석, Cutlass 분석 등 여러 주제를 포함한다.

# CuTe Layout Algebra

## 서론

CuTe layout algebra(https://github.com/NVIDIA/cutlass/blob/v3.5.1/media/docs/cute/02_layout_algebra.md)는 accelerated computing을 위해 CUTLASS(https://github.com/NVIDIA/cutlass/)를 이해하고 적용하는 데 매우 중요하다. CuTe에는 layout algebra 문서가 있지만, 그 수학적 기초를 먼저 이해하지 않으면 완전히 이해할 수 없다. 나는 CuTe layout algebra에 대한 몇 가지 proof를 직접 만들어 보려 했고, 그것이 매우 큰 작업량이라는 것을 깨달았다. 고맙게도 Jay Shah가 "A Note on the Algebra of CuTe Layouts"(https://leimao.github.io/downloads/article/2024-10-20-CuTe-Layout-Algebra/layout_algebra.pdf)라는 논문을 작성했고, 내가 만들고 싶었던 CuTe layout algebra의 수학적 기초를 완성했다.

내가 교정해 본 바로는 Jay Shah의 논문은 아주 사소한 누락과 오타를 제외하면 대부분 오류가 없었다. 그러나 몇 가지 세부 사항을 건너뛰고 있으며, 그 세부 사항이 없으면 논문을 이해하기가 조금 어렵다. 이 글에서는 Jay Shah의 논문을 바탕으로 CuTe layout algebra에 대해 더 많은 proof와 explanation을 제공하려 한다. 대부분의 definition과 annotation은 Jay Shah의 논문을 따른다.

이 글은 Jay Shah 논문의 보충 자료로 읽을 수 있지만, CuTe layout algebra를 이해하기 위한 독립적인 글이기도 하다.

## Layout Algebra 기초

### 정의 2.1: Layout

*Layout* $L$은 차원이 일치하는 양의 정수 tuple $\mathbf{S}$와 $\mathbf{D}$의 한 쌍이다. $\mathbf{S}$를 *shape*, $\mathbf{D}$를 *stride*라고 부른다. $L = \mathbf{S} : \mathbf{D}$라고 쓴다.

Flattened layout은 shape와 stride 안에 내부 괄호가 없다는 뜻이다. 예를 들어 $L = (5, 2, 2) : (16, 80, 4)$는 flattened layout이지만, $L = (5, (2, 2)) : (16, (80, 4))$는 그렇지 않다. Layout을 flatten해도 layout의 semantics와 operation은 바뀌지 않는다.

### 정의 2.2: Layout Size, Length, Mode

$\alpha \geq 0$를 정수라고 하고, $L = \mathbf{S} : \mathbf{D} = (M_0, M_1, \ldots, M_\alpha) : (d_0, d_1, \ldots, d_\alpha)$를 layout이라고 하자. 그러면 다음과 같다.

• $L$의 *size*는 곱 $M = M_0 \cdot M_1 \cdot \ldots \cdot M_\alpha$이다.

• $L$의 *length*는 정수 $\alpha + 1$이다.

• $L$의 *mode*는 $0 \leq k \leq \alpha$인 entry $(M_k) : (d_k)$ 중 하나다. 이를 length 1 layout으로 볼 수 있다.

### Concatenation

두 layout $L = \mathbf{S} : \mathbf{D}$와 $L' = \mathbf{S}' : \mathbf{D}'$가 주어졌다고 하자. $\mathbf{S}''$와 $\mathbf{D}''$를 각각 $(\mathbf{S}, \mathbf{S}')$와 $(\mathbf{D}, \mathbf{D}')$를 flatten해서 얻은 shape tuple과 stride tuple이라고 하자. 그러면 $L$과 $L'$의 *concatenation*은 다음 layout으로 주어진다.

$$(L, L') = \mathbf{S}'' : \mathbf{D}''$$

$(L, L')$가 $L$과 $L'$로 decomposed된다고 말한다.

귀납적으로 layout $L_0, L_1, \ldots, L_N$이 주어지면 concatenation $(L_0, L_1, \ldots, L_N)$을 만들 수 있다. 반대로 layout $L$이 주어지면 $L$은 그 mode들로 maximal하게 decomposed된다.

### Isomorphism

$\mathbf{S} = (M_0, M_1, \ldots, M_\alpha)$와 $\mathbf{D} = (d_0, d_1, \ldots, d_\alpha)$를 각각 $L = \mathbf{S} : \mathbf{D}$의 shape tuple과 stride tuple이라고 하자. $M = M_0 \cdot M_1 \cdot \ldots \cdot M_\alpha$를 $L$의 size라고 하고, $[0, M) \subset \mathbb{N}$를 $0, 1, 2, \ldots, M - 1$로 주어지는 natural number subset이라고 하자. 그러면 다음 isomorphism이 있다.

$$\iota : [0, M) \cong [0, M_0) \times [0, M_1) \times \ldots \times [0, M_\alpha)$$

임의의 $x \in [0, M)$가 주어지면 isomorphism $\iota$는 $x$를 다음 tuple로 mapping한다.

$$x \mapsto \left(x \bmod M_0, \left\lfloor \frac{x}{M_0} \right\rfloor \bmod M_1, \ldots, \left\lfloor \frac{x}{M_0 \cdot M_1 \cdot \ldots \cdot M_{\alpha-1}} \right\rfloor \bmod M_\alpha\right)$$

이 isomorphism mapping은 bijective하다. 여기서는 임의의 tuple $(x_0, x_1, \ldots, x_\alpha) \in [0, M_0) \times [0, M_1) \times \ldots \times [0, M_\alpha)$가 주어지면 isomorphism inverse mapping은 그 tuple을 다음 정수로 mapping한다.

$$(x_0, x_1, \ldots, x_\alpha) \mapsto x_0 + x_1 \cdot M_0 + x_2 \cdot M_0 \cdot M_1 + \ldots + x_\alpha \cdot M_0 \cdot M_1 \cdot \ldots \cdot M_{\alpha-1}$$

위 isomorphism mapping이 유효함을 검증하고 그 mapping이 bijective임을 증명하는 것(contradiction 사용)은 직접적이다.

Isomorphism은 1D coordinate와 multidimensional coordinate 사이의 mapping으로 생각할 수 있다.

### 정의 2.3: Layout Function

Layout $L$이 주어졌을 때, 그 *layout function*은 function $f_L : [0, M) \to \mathbb{N}$이며 다음 composite으로 정의된다.

$$[0, M) \cong [0, M_0) \times [0, M_1) \times \ldots \times [0, M_\alpha) \subset \mathbb{N}^{\times(\alpha+1)} \xrightarrow{d_0, d_1, \ldots, d_\alpha} \mathbb{N}^{\times(\alpha+1)} \to \mathbb{N}$$

다시 말해 $f_L$는 다음 multilinear function의 composition이다.

$$[0, M_0) \times [0, M_1) \times \ldots \times [0, M_\alpha) \to \mathbb{N}$$
$$(x_0, x_1, \ldots, x_\alpha) \mapsto x_0 \cdot d_0 + x_1 \cdot d_1 + \ldots + x_\alpha \cdot d_\alpha$$

이는 stride에 의해 결정되며, shape에 의해 결정되는 isomorphism $\iota$와 합성된다.

점 $x \in [0, M)$에서 layout function $f_L$의 값을 계산하는 일은 여러 점에서 layout function 값을 계산한 뒤 더하는 것으로 분해될 수 있다. 이는 어떤 점에서 layout function 값을 편리하게 계산할 때 유용할 수 있다.

Layout $L = (M_0, M_1, \ldots, M_\alpha) : (d_0, d_1, \ldots, d_\alpha)$와 $x \in [0, M)$가 주어졌을 때,

$$x \mapsto (x_0, x_1, \ldots, x_\alpha) \mapsto x_0 \cdot d_0 + x_1 \cdot d_1 + \ldots + x_\alpha \cdot d_\alpha$$

다음도 성립한다.

$$x_0' \mapsto (x_0, 0, 0, \ldots, 0) \mapsto x_0 \cdot d_0$$
$$x_1' \mapsto (0, x_1, 0, \ldots, 0) \mapsto x_1 \cdot d_1$$
$$\vdots$$
$$x_\alpha' \mapsto (0, 0, 0, \ldots, x_\alpha) \mapsto x_\alpha \cdot d_\alpha$$

따라서 다음을 얻는다.

$$f_L(x) = f_L(x_0') + f_L(x_1') + \ldots + f_L(x_\alpha')$$

$$x_0' = x \bmod M_0$$

$$x_1' = \left\lfloor \frac{x}{M_0} \right\rfloor \bmod M_1 \cdot M_0$$

$$\vdots$$

$$x_\alpha' = \left\lfloor \frac{x}{M_0 \cdot M_1 \cdot \ldots \cdot M_{\alpha-1}} \right\rfloor \bmod M_\alpha \cdot M_0 \cdot M_1 \cdot \ldots \cdot M_{\alpha-1}$$

예를 들어 layout $L = (3, 2) : (2, 3)$와 $x = 5$가 주어지면 다음을 얻는다.

$$f_L(5) = f_L(5 \bmod 3) + f_L\left(\left\lfloor \frac{5}{3} \right\rfloor \bmod 2 \cdot 3\right)$$

$$= f_L(2) + f_L(3)$$

$$= 2 \cdot 2 + \left\lfloor \frac{3}{3} \right\rfloor \cdot 3$$

$$= 4 + 3$$

$$= 7$$

### Layout Function의 Extension

Layout function의 정의에 기반해 layout function $f_L$의 extension은 function $\hat{f_L} : \mathbb{N} \to \mathbb{N}$이다. 이는 $f_L$의 정의에서 $M_\alpha$를 $\infty$로 바꾸어 정의하며, 즉 다음 composite이다.

$$\mathbb{N} \cong [0, M_0) \times [0, M_1) \times \ldots \times [0, M_{\alpha-1}) \times \mathbb{N} \subset \mathbb{N}^{\times(\alpha+1)} \xrightarrow{d_0, d_1, \ldots, d_\alpha} \mathbb{N}^{\times(\alpha+1)} \to \mathbb{N}$$

여기서 isomorphism $\iota$의 extension $\hat{\iota}$는 다음과 같이 주어진다.

$$x \mapsto \left(x \bmod M_0, \left\lfloor \frac{x}{M_0} \right\rfloor \bmod M_1, \ldots, \left\lfloor \frac{x}{M_0 \cdot M_1 \cdot \ldots \cdot M_{\alpha-2}} \right\rfloor \bmod M_{\alpha-1}, \left\lfloor \frac{x}{M_0 \cdot M_1 \cdot \ldots \cdot M_{\alpha-1}} \right\rfloor\right)$$

Isomorphism extension의 mapping 역시 bijective하다. Isomorphism extension의 inverse mapping도 다음과 같이 주어진다.

$$(x_0, x_1, \ldots, x_{\alpha-1}, x_\alpha) \mapsto x_0 + x_1 \cdot M_0 + x_2 \cdot M_0 \cdot M_1 + \ldots + x_\alpha \cdot M_0 \cdot M_1 \cdot \ldots \cdot M_{\alpha-1}$$

Isomorphism의 extension은 shape의 마지막 dimension을 "batch" dimension으로 정의하고, batch size가 infinite일 수 있게 하는 것으로 생각할 수 있다.

#### Extension Function의 상세 분석

Layout function의 extension $\hat{f_L}$는 original layout function $f_L$를 finite domain에서 infinite domain으로 확장하는 중요한 operation이다. 핵심 아이디어는 "batch dimension" 개념을 통해 dynamic size data를 처리하는 것이다.

**Original layout function vs extended layout function 비교:**

- **Original layout function** $f_L$: domain은 finite set $[0, M)$이며, 여기서 $M = M_0 \cdot M_1 \cdot ... \cdot M_\alpha$이다.
- **Extended layout function** $\hat{f_L}$: domain은 infinite set $\mathbb{N}$이며, 마지막 dimension $M_\alpha$를 $\infty$로 바꾸어 구현한다.

**Extension의 핵심 메커니즘:**

Extended isomorphism mapping에서 앞쪽 dimension들($x_0, x_1, ..., x_{α-1}$)은 여전히 bound를 가지며 순환적으로 사용되지만, 마지막 dimension($x_α$)은 unbounded가 되어 무한히 증가할 수 있다. 이는 coordinate 계산식에 다음처럼 나타난다.

- 앞쪽 dimension: $x_i = \lfloor x/\prod_{j=0}^{i-1} M_j \rfloor \bmod M_i$(mod operation이 있음)
- 마지막 dimension: $x_α = \lfloor x/\prod_{j=0}^{α-1} M_j \rfloor$(mod operation이 없음)

**실제 적용 예:**

Layout $L = (2, 3) : (1, 2)$를 생각하자. 여기서 $M_0 = 2, M_1 = 3, M = 6$이다.

- Original function domain: $\{0, 1, 2, 3, 4, 5\}$
- Extended function domain: $\{0, 1, 2, 3, 4, 5, 6, 7, 8, ...\}$

$x = 7$(original range 밖)에 대해서는 다음과 같다.
```
x₀ = 7 mod 2 = 1
x₁ = ⌊7/2⌋ = 3  // 주의: mod 3 operation이 없다.
coordinate: (1, 3)
function value: f̂_L(7) = 1×1 + 3×2 = 7
```

$x = 8$에 대해서는 다음과 같다.
```
x₀ = 8 mod 2 = 0  
x₁ = ⌊8/2⌋ = 4
coordinate: (0, 4)
function value: f̂_L(8) = 0×1 + 4×2 = 8
```

**CUDA programming에서의 적용 가치:**

1. **batch processing 지원**: 정확한 batch size를 미리 알 필요 없이 임의 크기의 batch data를 처리할 수 있다.
2. **pipeline processing**: 연속적인 data stream 처리를 지원하므로 real-time computing scenario에 적합하다.
3. **memory access pattern 유지**: 앞쪽 dimension의 access pattern은 변하지 않고 마지막 dimension에서만 확장되므로 memory access locality를 보장한다.

Extension function은 본질적으로 finite multidimensional index space를 semi-infinite space로 변환하며, dynamic size data를 처리하기 위한 견고한 수학적 기초를 제공한다. 이는 modern GPU computing에서 중요한 실제적 의미를 가진다.

## Complement

### 정의 2.4: Sorted Layout

$A = (N_0, N_1, \ldots, N_\alpha) : (d_0, d_1, \ldots, d_\alpha)$를 layout이라고 하자. $d_0 \leq d_1 \leq \ldots \leq d_\alpha$이고 모든 $i < j$에 대해 $d_i = d_j$이면 $N_i \leq N_j$일 때, $A$가 *sorted*라고 말한다.

Sorted layout, 더 일반적으로는 layout mode의 순서를 바꾸는 것은 layout의 semantics와 operation을 바꾼다는 점에 주의하라.

예를 들어 layout $A = (2, 4) : (4, 1)$와 layout $B = (4, 2) : (1, 4)$가 있다고 하자. $B$가 $A$의 sorted version임을 볼 수 있다. Lookup table을 사용해 $A$와 $B$의 layout function을 다음과 같이 계산할 수 있다.

$f_A(0) = f_A(0, 0) = 0 \cdot 4 + 0 \cdot 1 = 0$
$f_A(1) = f_A(1, 0) = 1 \cdot 4 + 0 \cdot 1 = 4$
$f_A(2) = f_A(0, 1) = 0 \cdot 4 + 1 \cdot 1 = 1$
$f_A(3) = f_A(1, 1) = 1 \cdot 4 + 1 \cdot 1 = 5$
$f_A(4) = f_A(0, 2) = 0 \cdot 4 + 2 \cdot 1 = 2$
$f_A(5) = f_A(1, 2) = 1 \cdot 4 + 2 \cdot 1 = 6$
$f_A(6) = f_A(0, 3) = 0 \cdot 4 + 3 \cdot 1 = 3$
$f_A(7) = f_A(1, 3) = 1 \cdot 4 + 3 \cdot 1 = 7$

$f_B(0) = f_B(0, 0) = 0 \cdot 1 + 0 \cdot 4 = 0$
$f_B(1) = f_B(1, 0) = 1 \cdot 1 + 0 \cdot 4 = 1$
$f_B(2) = f_B(2, 0) = 2 \cdot 1 + 0 \cdot 4 = 2$
$f_B(3) = f_B(3, 0) = 3 \cdot 1 + 0 \cdot 4 = 3$
$f_B(4) = f_B(0, 1) = 0 \cdot 1 + 1 \cdot 4 = 4$
$f_B(5) = f_B(1, 1) = 1 \cdot 1 + 1 \cdot 4 = 5$
$f_B(6) = f_B(2, 1) = 2 \cdot 1 + 1 \cdot 4 = 6$
$f_B(7) = f_B(3, 1) = 3 \cdot 1 + 1 \cdot 4 = 7$

Layout $B$는 보통 column-major layout이라고 불리고, layout $A$는 보통 row-major layout이라고 불림을 볼 수 있다. 둘은 완전히 다른 layout이다.

더 일반적으로 sorted layout은 column-major layout의 "generalization"과 같다.

### 정의 2.5: Complement의 Admissibility

$A = (N_0, N_1, \ldots, N_\alpha) : (d_0, d_1, \ldots, d_\alpha)$를 layout이라고 하고, $M$을 양의 정수라고 하자. $A$가 sorted가 아니면 $A$를 그 sorted version으로 대체한다. 다음을 만족하면 pair $\{A, M\}$가 *complement admissible*(또는 줄여서 admissible)이라고 말한다.

• 모든 $1 \leq i \leq \alpha$에 대해 $N_{i-1} \cdot d_{i-1}$가 $d_i$를 나눈다.

• $N_\alpha \cdot d_\alpha$가 $M$을 나눈다.

$\{A, M\}$가 complement admissible이라는 것은 다음도 의미한다.

• 모든 $1 \leq i \leq \alpha$에 대해 $N_{i-1} \cdot d_{i-1} \leq d_i$이고 $d_{i-1} \leq d_i$이다.

• $N_\alpha \cdot d_\alpha \leq M$이고 $d_\alpha \leq M$이다.

### 정의 2.6: Complement

$A = (N_0, N_1, \ldots, N_\alpha) : (d_0, d_1, \ldots, d_\alpha)$를 layout이라고 하고, $M$을 양의 정수라고 하자. $\{A, M\}$가 complement admissible이면, $A$가 sorted가 아닐 경우 $A$를 그 sorted version으로 대체한다. $\{A, M\}$의 complement는 다음 layout으로 정의된다.

$$\text{complement}(A, M) = \left(d_0, \frac{d_1}{N_0 d_0}, \frac{d_2}{N_1 d_1}, \ldots, \frac{d_\alpha}{N_{\alpha-1} d_{\alpha-1}}, \frac{M}{N_\alpha d_\alpha}\right) : (1, N_0 d_0, N_1 d_1, \ldots, N_\alpha d_\alpha)$$

$\{A, M\}$의 complement size, 즉 $\text{size}(\text{complement}(A, M))$는 $\frac{M}{\text{size}(A)} = \frac{M}{N_0 \cdot N_1 \cdot \ldots \cdot N_\alpha}$임에 주의하라.

정의상 $\{A, M\}$의 complement는 $A$의 mode 순서에 민감하지 않다. complement를 취하기 전에 항상 sorting하기 때문이다.

$\{A, M\}$의 complement는 strictly increasing이다. 이것이 아주 명확하지 않을 수 있으므로 proof를 보인다.

**증명**

$B = \text{complement}(A, M)$라고 하자. Layout function $f_B$(그 domain은 natural number set)가 strictly increasing임을 증명하려면, 인접한 두 natural number $x$와 $x + 1$에 대해 $0 \leq x < x + 1 < \text{size}(B)$이면 $f_B(x) < f_B(x + 1)$임을 보여야 한다.

Isomorphism에 의해 $x$의 mapping이 다음과 같다고 하자.

$$x \mapsto (x_0, x_1, \ldots, x_\alpha, x_{\alpha+1})$$

Layout function $f_B$의 정의에 따라 다음을 얻는다.

$$f_B(x) = x_0 + x_1 \cdot N_0 d_0 + x_2 \cdot N_1 d_1 + \ldots + x_\alpha \cdot N_{\alpha-1} d_{\alpha-1} + x_{\alpha+1} \cdot N_\alpha d_\alpha$$

$x + 1$의 mapping에는 여러 다른 경우가 있을 수 있다.

가장 단순한 경우에는 다음과 같다.

$$x + 1 \mapsto (x_0 + 1, x_1, \ldots, x_\alpha, x_{\alpha+1})$$

그러면 다음을 얻는다.

$$f_B(x + 1) = x_0 + 1 + x_1 \cdot N_0 d_0 + x_2 \cdot N_1 d_1 + \ldots + x_\alpha \cdot N_{\alpha-1} d_{\alpha-1} + x_{\alpha+1} \cdot N_\alpha d_\alpha$$

$$= f_B(x) + 1$$

$$> f_B(x)$$

더 복잡한 경우로 $x_0 = d_0 - 1$이고 $x_1 < \frac{d_1}{N_0 d_0} - 1$이면 다음을 얻는다.

$$x + 1 \mapsto (0, x_1 + 1, \ldots, x_\alpha, x_{\alpha+1})$$

> **설명:** 이 조건은 multidimensional coordinate에서 "carry"가 발생하는 경우를 설명한다. 구체적인 의미는 다음과 같다.
> 
> - **$x_0 = d_0 - 1$**: 0번째 dimension coordinate가 이미 maximum value에 도달했다(coordinate range는 $[0, d_0)$).
> - **$x_1 < \frac{d_1}{N_0 d_0} - 1$**: 1번째 dimension coordinate는 아직 maximum value에 도달하지 않았다.
> 
> $x$가 $x+1$로 증가할 때 다음 일이 발생한다.
> - 0번째 dimension coordinate가 $d_0-1$에서 overflow되어 $0$으로 reset된다.
> - 1번째 dimension coordinate가 $x_1$에서 $x_1+1$로 증가한다(carry).
> - 다른 dimension coordinate는 변하지 않는다.
> 
> 이 분석은 coordinate에 "carry"가 발생하는 복잡한 경우에도 layout function 값이 여전히 strictly increasing임을 보장한다. 추가적인 $(N_0-1)d_0$ 항은 0번째 dimension reset으로 인한 감소를 보상해 전체 function value가 계속 증가하도록 보장한다.

그러면 다음을 얻는다.

$$f_B(x + 1) = 0 + (x_1 + 1) \cdot N_0 d_0 + x_2 \cdot N_1 d_1 + \ldots + x_\alpha \cdot N_{\alpha-1} d_{\alpha-1} + x_{\alpha+1} \cdot N_\alpha d_\alpha$$

$$= f_B(x) - x_0 + N_0 d_0$$

$$= f_B(x) - (d_0 - 1) + N_0 d_0$$

$$= f_B(x) + 1 + (N_0 - 1) d_0$$

$$> f_B(x)$$

$N_0 \geq 1$이므로 $(N_0 - 1) d_0 \geq 0$이고, 따라서 다음을 얻는다.

$$f_B(x + 1) > f_B(x)$$

일반적으로 $x_0 = d_0 - 1$일 때, 어떤 $k \in [1, \alpha - 1]$에 대해 모든 $i \in [1, k]$에서 $x_i = \frac{d_i}{N_{i-1} d_{i-1}} - 1$이고 $x_{k+1} < \frac{d_{k+1}}{N_k d_k} - 1$이면 다음을 얻는다.

$$x + 1 \mapsto (0, 0, \ldots, 0, x_{k+1} + 1, \ldots, x_\alpha, x_{\alpha+1})$$

그러면 다음을 얻는다.

$$f_B(x + 1) = 0 + 0 \cdot N_0 d_0 + \ldots + 0 \cdot N_{k-1} d_{k-1} + (x_{k+1} + 1) \cdot N_k d_k + \ldots + x_\alpha \cdot N_{\alpha-1} d_{\alpha-1} + x_{\alpha+1} \cdot N_\alpha d_\alpha$$

$$= f_B(x) - x_0 - \left(\sum_{i=1}^{k} x_i \cdot N_{i-1} d_{i-1}\right) + N_k d_k$$

$$= f_B(x) - (d_0 - 1) - \left(\sum_{i=1}^{k} \left(\frac{d_i}{N_{i-1} d_{i-1}} - 1\right) \cdot N_{i-1} d_{i-1}\right) + N_k d_k$$

$$= f_B(x) - (d_0 - 1) - \left(\sum_{i=1}^{k} (d_i - N_{i-1} d_{i-1})\right) + N_k d_k$$

$$= f_B(x) - (d_0 - 1) + \sum_{i=1}^{k} N_{i-1} d_{i-1} - \sum_{i=1}^{k} d_i + N_k d_k$$

$$= f_B(x) + \sum_{i=0}^{k} (N_i - 1) d_i + 1$$

모든 $i$에 대해 $N_i \geq 1$이므로 모든 $i$에 대해 $(N_i - 1) d_i \geq 0$이고, 따라서 다음을 얻는다.

$$f_B(x + 1) > f_B(x)$$

이로써 증명이 완료된다.

마찬가지로 $\{A, M\}$의 complement extension도 strictly increasing임을 증명할 수 있다.

### 명제 2.7

$\{A = (N_0, N_1, \ldots, N_\alpha) : (d_0, d_1, \ldots, d_\alpha), M\}$가 complement admissible이고 $B = \text{complement}(A, M)$라고 하자. $C = (A, B)$를 concatenated layout이라고 하자. 그러면 $C$의 size는 $M$이고, $f_C : [0, M) \to \mathbb{N}$는 restriction상 bijection $[0, M) \cong [0, M)$이다.

**증명**

$\text{size}(A) = \prod_{i=0}^\alpha N_i$이고 $\text{size}(B) = \frac{M}{\prod_{i=0}^\alpha N_i}$이므로 $\text{size}(C) = \text{size}(A) \cdot \text{size}(B) = M$이다. 따라서 $f_C$의 domain은 $[0, M)$이다.

$f_C$의 image는 $C$의 어떤 permutation $C'$에 대한 $f_{C'}$의 image와 같음에 주의하라.

이를 보기 위해 다음 layout $C$와 그 permutation $C'$가 있고, 여기서는 mode 한 쌍만 permutation되었다고 하자.

$$C = (N_0, N_1, \ldots, N_i, \ldots, N_j, \ldots, N_\alpha) : (d_0, d_1, \ldots, d_i, \ldots, d_j, \ldots, d_\alpha)$$
$$C' = (N_0, N_1, \ldots, N_j, \ldots, N_i, \ldots, N_\alpha) : (d_0, d_1, \ldots, d_j, \ldots, d_i, \ldots, d_\alpha)$$

$f_C$와 $f_{C'}$의 domain은 모두 $[0, M)$이다. 임의의 $x_C \in [0, M)$에 대해 다음을 얻는다.

$$x_C \mapsto (x_0, x_1, \ldots, x_i, \ldots, x_j, \ldots, x_\alpha)$$
$$x_{C'} \mapsto (x_0, x_1, \ldots, x_j, \ldots, x_i, \ldots, x_\alpha)$$

그리고 $x_C$와 $x_{C'}$는 bijective하다.

정의에 의해 $f_C(x_C) = f_{C'}(x_{C'})$이므로 $f_C$의 image와 $f_{C'}$의 image는 같다.

$C$의 어떤 permutation $C'$도 $C$의 mode 한 쌍을 한 번씩 permutation하는 과정을 통해 얻을 수 있으며, 매번 $f_C$의 image와 $f_{C'}$의 image는 같다. 따라서 $C$의 어떤 permutation $C'$에 대해서도 $f_C$의 image와 $f_{C'}$의 image는 같다.

$f_C$의 image를 계산할 때 $C$를 sort할 수 있다. Without loss of generality, $A = (N_0, N_1, \ldots, N_\alpha) : (d_0, d_1, \ldots, d_\alpha)$가 이미 sorted라고 가정하자. $C$를 sort한 뒤 sorted $C'$는 다음 형태일 수밖에 없다.

$$C' = \left(d_0, N_0, \frac{d_1}{N_0 d_0}, N_1, \frac{d_2}{N_1 d_1}, N_2, \ldots, \frac{d_\alpha}{N_{\alpha-1} d_{\alpha-1}}, N_\alpha, \frac{M}{N_\alpha d_\alpha}\right) : (1, d_0, N_0 d_0, d_1, N_1 d_1, d_2, \ldots, N_{\alpha-1} d_{\alpha-1}, d_\alpha, N_\alpha d_\alpha)$$

모든 $i$에 대해 $d_i \leq N_i d_i$이고 $N_i d_i \leq d_{i+1}$이다. $N_i = 1$일 때 $N_i \leq \frac{d_{i+1}}{N_i d_i}$이고, $N_i d_i = d_{i+1}$일 때 $\frac{d_{i+1}}{N_i d_i} \leq N_{i+1}$이므로 $C'$는 sorted이다. $C'$의 어떤 permutation도 이를 unsorted로 만든다.

그러면 다음과 같이 다시 쓸 수 있다.

$$C' = (r_0, r_1, r_2, \ldots, r_\beta) : (1, r_0, r_0 r_1, \ldots, r_0 r_1 \ldots r_{\beta-1})$$

여기서 $\beta = 2\alpha + 1$이며, $f_{C'}$가 도달하는 maximum value는 다음과 같이 계산된다.

$$f_{C'}(M - 1) = f_{C'}(r_0 - 1, r_1 - 1, r_2 - 1, \ldots, r_{\beta-1} - 1, r_\beta - 1)$$

$$= (r_0 - 1) + (r_1 - 1) \cdot r_0 + (r_2 - 1) \cdot r_0 r_1 + \ldots + (r_{\beta-1} - 1) \cdot r_0 r_1 \ldots r_{\beta-2} + (r_\beta - 1) \cdot r_0 r_1 \ldots r_{\beta-1}$$

$$= r_0 - 1 + r_0 r_1 - r_0 + r_0 r_1 r_2 - r_0 r_1 + \ldots + r_0 r_1 \ldots r_{\beta-1} - r_0 r_1 \ldots r_{\beta-2} + r_0 r_1 \ldots r_\beta - r_0 r_1 \ldots r_{\beta-1}$$

$$= r_0 r_1 \ldots r_\beta - 1$$

$$= M - 1$$

그러면 이 경우 bijectivity 주장을 세우려면 $f_{C'}(x)$가 injective임을 증명하기만 하면 된다. 즉 임의의 $x, y \in [0, M)$에 대해 $f_{C'}(x) = f_{C'}(y)$이면 $x = y$임을 보이면 된다.

$x$와 $y$의 isomorphism mapping이 다음과 같다고 하자.

$$x \mapsto (x_0, x_1, \ldots, x_\beta)$$
$$y \mapsto (y_0, y_1, \ldots, y_\beta)$$

$f_{C'}(x) = f_{C'}(y)$이므로 다음을 얻는다.

$$x_0 + x_1 \cdot r_0 + x_2 \cdot r_0 r_1 + \ldots + x_\beta \cdot r_0 r_1 \ldots r_{\beta-1} = y_0 + y_1 \cdot r_0 + y_2 \cdot r_0 r_1 + \ldots + y_\beta \cdot r_0 r_1 \ldots r_{\beta-1}$$

Strong induction을 사용해 모든 $i \in [0, \beta]$에 대해 $x_i = y_i$임을 증명한다.

$f_{C'}(x) \bmod r_0 = f_{C'}(y) \bmod r_0$이므로 $x_0 = y_0$이다.

이제 strong induction으로, 주어진 $i \in (0, \beta]$에 대해 모든 $j < i$에서 $x_j = y_j$라고 가정하자. 그러면 다음을 얻는다.

$$x_i \cdot r_0 r_1 \ldots r_{i-1} + x_{i+1} \cdot r_0 r_1 \ldots r_i + \ldots + x_\beta \cdot r_0 r_1 \ldots r_{\beta-1} = y_i \cdot r_0 r_1 \ldots r_{i-1} + y_{i+1} \cdot r_0 r_1 \ldots r_i + \ldots + y_\beta \cdot r_0 r_1 \ldots r_{\beta-1}$$

$x_i \in [0, r_i)$이고 $y_i \in [0, r_i)$이므로, 이 등식에 modulo $r_0 r_1 \ldots r_i$를 취한 뒤 $r_0 r_1 \ldots r_{i-1}$로 나누면 $x_i = y_i$를 얻는다.

$(x_0, x_1, \ldots, x_\beta) = (y_0, y_1, \ldots, y_\beta)$이고 isomorphism mapping은 bijective이므로 $x = y$이다.

따라서 $f_{C'} : [0, M) \to \mathbb{N}$는 restriction상 bijection $[0, M) \cong [0, M)$이다. $f_C$도 마찬가지다.

이로써 증명이 완료된다.

### 따름정리 2.8 Complement Disjointness

따름정리 2.8은 layout의 complement를 취한다는 것의 의미를 설명한다.

명제 2.7의 설정에서 $I = [0, \text{size}(A)) = [0, N_0 N_1 \ldots N_\alpha)$를 $f_A$의 domain이라고 하자. 그러면 다음이 성립한다.

$$f_A(I) \cap \hat{f_B}(I) = \{0\}$$

다시 말해 $\hat{f_A}$와 $\hat{f_B}$는 $f_A$의 domain으로 제한했을 때 0을 제외하면 disjoint image를 가진다.

따름정리에서 $f_A$와 $\hat{f_A}$는 실제로 서로 바꿔 쓸 수 있음에 주의하라. function domain이 $f_A$의 domain으로 제한되기 때문이다.

**증명**

$J = [0, \text{size}(B)) = [0, \frac{M}{N_0 N_1 \ldots N_\alpha})$를 $f_B$의 domain이라고 하자. 그러면 명제 2.7에 의해 다음을 얻는다.

$$f_A(I) \cap f_B(J) = \{0\}$$

이를 이해하기 위해 임의의 $x_A \in I$와 임의의 $x_B \in J$에 대해 isomorphism에 의해 다음을 얻는다.

$$x_A \mapsto (x_{A,0}, x_{A,1}, \ldots, x_{A,\alpha})$$
$$x_B \mapsto (x_{B,0}, x_{B,1}, \ldots, x_{B,\alpha}, x_{B,\alpha+1})$$

그러면 다음을 얻는다.

$$f_A(x_A) = x_{A,0} + x_{A,1} \cdot N_0 + x_{A,2} \cdot N_0 N_1 + \ldots + x_{A,\alpha} \cdot N_0 N_1 \ldots N_{\alpha-1}$$
$$f_B(x_B) = x_{B,0} + x_{B,1} \cdot N_0 d_0 + x_{B,2} \cdot N_1 d_1 + \ldots + x_{B,\alpha} \cdot N_{\alpha-1} d_{\alpha-1} + x_{B,\alpha+1} \cdot N_\alpha d_\alpha$$

Layout $C$에 대해 새로운 coordinate를 다음과 같이 배치한다.

$$x'_A \mapsto (0, x_{A,0}, 0, x_{A,1}, 0, x_{A,2}, \ldots, 0, x_{A,\alpha}, 0)$$
$$x'_B \mapsto (x_{B,0}, 0, x_{B,1}, 0, x_{B,2}, \ldots, x_{B,\alpha}, 0, x_{B,\alpha+1})$$

그러면 다음을 얻는다.

$$f_C(x'_A) = x_{A,0} + x_{A,1} \cdot N_0 + x_{A,2} \cdot N_0 N_1 + \ldots + x_{A,\alpha} \cdot N_0 N_1 \ldots N_{\alpha-1}$$
$$= f_A(x_A)$$
$$f_C(x'_B) = x_{B,0} + x_{B,1} \cdot N_0 d_0 + x_{B,2} \cdot N_1 d_1 + \ldots + x_{B,\alpha} \cdot N_{\alpha-1} d_{\alpha-1} + x_{B,\alpha+1} \cdot N_\alpha d_\alpha$$
$$= f_B(x_B)$$

명제 2.7에 의해 $f_C : [0, M) \to \mathbb{N}$는 restriction상 bijection $[0, M) \cong [0, M)$이다. $x'_A \neq x'_B$이면 $f_C(x'_A) \neq f_C(x'_B)$이고 $f_A(x_A) \neq f_B(x_B)$이다.

명백히 $(0, 0, \ldots, 0)$를 제외하면, $x_{A,0}, x_{A,1}, \ldots, x_{A,\alpha}$와 $x_{B,0}, x_{B,1}, \ldots, x_{B,\alpha}, x_{B,\alpha+1}$의 어떤 값에 대해서도 $(0, x_{A,0}, 0, x_{A,1}, 0, x_{A,2}, \ldots, 0, x_{A,\alpha}, 0) \neq (x_{B,0}, 0, x_{B,1}, 0, x_{B,2}, \ldots, x_{B,\alpha}, 0, x_{B,\alpha+1})$이다. 따라서 $x'_A \neq x'_B$, $f_C(x'_A) \neq f_C(x'_B)$, 그리고 $f_A(x_A) \neq f_B(x_B)$이다.

이는 임의의 $x \in I$ 및 $x \neq 0$에 대해 $f_A(x) = f_B(y)$가 되게 하는 $y \in J$가 존재하지 않음을 뜻한다.

$x = 0$일 때는 $f_A(x) = f_B(x) = 0$이다. 따라서 다음과 같이 주장할 수 있다.

$$f_A(I) \cap f_B(J) = \{0\}$$

정의 2.6: Complement에서 우리는 $\{A, M\}$의 complement, $f_B$, 그리고 그 extension $\hat{f_B}$가 모두 strictly increasing임을 이미 증명했다.

또한 isomorphism extension을 통해 다음을 얻는다.

$$\text{size}(B) \mapsto \left(0, 0, \ldots, 0, \frac{M}{N_\alpha d_\alpha}\right)$$

그러면 다음을 얻는다.

$$\hat{f_B}(\text{size}(B)) = 0 + 0 \cdot 1 + 0 \cdot N_0 d_0 + \ldots + 0 \cdot N_{\alpha-1} d_{\alpha-1} + \frac{M}{N_\alpha d_\alpha} \cdot N_\alpha d_\alpha$$
$$= M$$

$f_A$가 도달하는 maximum value는 $N_0 N_1 \ldots N_\alpha - 1$에서이며, $f_A(N_0 N_1 \ldots N_\alpha - 1) = (N_0 - 1) d_0 + (N_1 - 1) d_1 + \ldots + (N_\alpha - 1) d_\alpha$이다.

$(N_0 - 1) d_0 < N_0 d_0$이고 모든 $i \in [0, \alpha - 1]$에 대해 $N_i d_i \leq d_{i+1}$이며 $N_\alpha d_\alpha \leq M$이므로 다음을 얻는다.

$$f_A(N_0 N_1 \ldots N_\alpha - 1) = (N_0 - 1) d_0 + (N_1 - 1) d_1 + \ldots + (N_\alpha - 1) d_\alpha$$
$$< N_0 d_0 + N_1 d_1 - d_1 + N_2 d_2 - d_2 + \ldots + N_\alpha d_\alpha - d_\alpha$$
$$\leq d_1 + N_1 d_1 - d_1 + N_2 d_2 - d_2 + \ldots + N_\alpha d_\alpha - d_\alpha$$
$$= N_1 d_1 + N_2 d_2 - d_2 + \ldots + N_\alpha d_\alpha - d_\alpha$$
$$\leq d_2 + N_2 d_2 - d_2 + \ldots + N_\alpha d_\alpha - d_\alpha$$
$$\vdots$$
$$\leq d_\alpha + N_\alpha d_\alpha - d_\alpha$$
$$= N_\alpha d_\alpha$$
$$\leq M$$

따라서 $f_A(N_0 N_1 \ldots N_\alpha - 1) < \hat{f_B}(\text{size}(B))$이다.

$I \cap J = I$인 경우, 즉 $\text{size}(A) \leq \text{size}(B)$이면 다음을 얻는다.

$$f_A(I) \cap f_B(I) = \{0\}$$

이 경우에는 $f_B(I) = \hat{f_B}(I)$이므로 다음을 얻는다.

$$f_A(I) \cap \hat{f_B}(I) = \{0\}$$

다른 경우인 $I \cap J = J$, 즉 $\text{size}(A) \geq \text{size}(B)$라고 하자. $f_A$가 도달하는 maximum value는 $f_A(N_0 N_1 \ldots N_\alpha - 1)$이고 $f_A(N_0 N_1 \ldots N_\alpha - 1) < \hat{f_B}(\text{size}(B))$이므로, 임의의 $x \in I/J$에 대해 $f_A(x) < \hat{f_B}(\text{size}(B))$이다.

따라서

$$f_A(I) \cap \hat{f_B}(I/J) = \emptyset$$

따라서

$$f_A(I) \cap \hat{f_B}(I) = f_A(I) \cap \left(\hat{f_B}(I) \cup \hat{f_B}(I/J)\right)$$
$$= f_A(I) \cap \left(f_B(I) \cup \hat{f_B}(I/J)\right)$$
$$= (f_A(I) \cap f_B(I)) \cup \left(f_A(I) \cap \hat{f_B}(I/J)\right)$$
$$= \{0\} \cup \emptyset$$
$$= \{0\}$$

종합하면 다음을 얻는다.

$$f_A(I) \cap \hat{f_B}(I) = \{0\}$$

이로써 증명이 완료된다.

논문에 있는 따름정리 2.8의 원래 proof에 대해 짧게 설명하자면, Jay Shah는 $f_A(I \cap J) \cap f_B(I \cap J) = \{0\}$라고 주장했는데, 이것만으로는 증명하기에 충분하지 않다. 충분한 statement는 $f_A(I) \cap f_B(J) = \{0\}$이어야 한다.
