> 이 글은 @Simon V(https://github.com/simveit)의 허가를 받아 전재 및 번역하여 본 공중계정에 게시한다. 원문 주소는 https://veitner.bearblog.dev/bridging-math-and-code-cute-layout-algebra-in-cutedsl/ 이다.

# 수학과 코드 연결하기: CuTeDSL의 CuTe Layout 대수

19 May, 2025

## 소개

이번 주 `CUTLASS` 팀은 새로운 버전의 `CUTLASS`를 발표하며 `CuTeDSL`이라는 Python 인터페이스를 도입했다. 전용 문서(https://docs.nvidia.com/cutlass/media/docs/pythonDSL/overview.html)에 따르면, 이 인터페이스는 layout, tensor, hardware atom 같은 핵심 개념과 hardware thread 및 data hierarchy에 대한 완전한 제어를 사용자에게 제공한다.

이 블로그 글의 목표는 독자가 `CuTe` layout 대수의 몇 가지 원리를 기본적으로 이해하도록 하는 것이다. layout 함수, merge와 complement 같은 기본 개념을 설명한다. 이는 `CuTeDSL`을 더 깊이 이해하는 데 도움이 된다.

## Layout

우리는 Layout을 다음과 같이 정의한다.

$$L = (M_0, M_1, \ldots, M_n):(d_0, d_1, \ldots, d_n)$$

여기서

$S = (M_0, M_1, \ldots, M_n)$ 은 shape이며, $M = M_0 \cdot \ldots \cdot M_n$ 은 layout의 크기이다. $D = (d_0, d_1, \ldots, d_n)$ 은 stride이다.

pairs $(M_i):(d_i)$ 는 mode라고 부르며, 길이가 1인 layout으로 생각할 수 있다. 일반적으로 우리는 위의 길이를 $n + 1$로 정의한다.

## layout 함수

다음을 두자.

$$
l(x) = (x \bmod M_n, \lfloor \frac{x}{M_n} \rfloor \bmod M_1, \ldots, \lfloor \frac{x}{M_0 \cdot \ldots \cdot M_{n-1}} \rfloor \bmod M_n) = (i_0, i_1, \ldots, i_n)
$$

이는 동형 사상 $[0, M) \cong [0, M_0) \times \ldots \times [0, M_n)$ 이다. 이것은 1차원 index를 다차원 index로 매핑한다.

그다음 layout 함수는 숫자 $x$에 동형 사상을 적용한 뒤(필요하다면), 각 vector 원소를 해당 stride 원소와 곱하고 이들을 합하는 mapping으로 정의된다. 즉 다음과 같다.

$$f_L(x) = i_0 d_0 + i_1 d_1 + \ldots + i_n d_n$$

예를 들어 layout $(2, 4):(2, 2)$를 생각해 보자.

$f_L(3)$을 계산해 보자.

$$l(x) = (3 \bmod 2, \lfloor \frac{3}{2} \rfloor \bmod 4) = (1,1)$$

여기서 다음을 얻는다.

$$f_L(3) = 1 \cdot 2 + 1 \cdot 2 = 2 + 2 = 4$$

이를 `CuTeDSL`에서 계산할 수 있다.

```python
import cutlass               
import cutlass.cute as cute  

@cute.jit
def layout_function_example():
    """
    Layout function in cutlass
    """
    S = (2, 4)
    D = (2, 2)
    L = cute.make_layout(shape=S, stride=D)

    for i in cutlass.range_constexpr(cute.size(S)):
        cute.printf("fL({}) = {}", i, L(i))

layout_function_example()
```

이는 다음을 출력한다.

```shell
fL(0) = 0
fL(1) = 2
fL(2) = 2
fL(3) = 4
fL(4) = 4
fL(5) = 6
fL(6) = 6
fL(7) = 8
```

## Sorted layouts

우리는 정렬된 layout(sorted layout)을 모든 stride가 증가하도록, 즉 $d_{i-1} \leq d_i$가 되도록 정의한다.

정렬은 layout을 그대로 보존하지 않는다. 다음 표현을 생각해 보자.

$$L_1 = (2,2):(3,1)$$
$$L_2 = \text{sorted}(L_1) = (2,2):(1,3)$$

해당 layout 함수들은 일치하지 않으며, `CuTeDSL`로 쉽게 검증할 수 있다.

```shell
import cutlass               
import cutlass.cute as cute  

@cute.jit
def sorted_example():
    """
    Sorting in cutlass
    """
    S1 = (2, 2)
    D1 = (3, 1)
    L1 = cute.make_layout(shape=S1, stride=D1)
    S2 = (2, 2)
    D2 = (1, 3)
    L2 = cute.make_layout(shape=S2, stride=D2)

    for i in cutlass.range_constexpr(cute.size(S1)):
        cute.printf("fL1({}) = {}, fL2({}) = {}", i, L1(i), i, L2(i))

sorted_example()
```

이는 다음을 출력한다.

```shell
fL1(0) = 0, fL2(0) = 0
fL1(1) = 3, fL2(1) = 1
fL1(2) = 1, fL2(2) = 3
fL1(3) = 4, fL2(3) = 4
```

물론 이 간단한 예는 손으로 계산해서도 검증할 수 있다. 위 예를 관찰하면 $L_1$은 row-major이고, $L_2$는 column-major임을 볼 수 있다.

## complement 연산

### 허용 가능성(Admissability)
 $L$을 layout, $K$를 양의 정수라고 하자. $(L,K)$가 허용 가능하다는 것은 다음 조건을 만족할 때이자 그때뿐이다.
$M_{i-1}d_{i-1}$ 가 $d_i$를 나눈다.
$M_n d_n$ 이 $K$를 나눈다.

### complement 연산(Complement)
$(L,K)$가 허용 가능하다면 complement 연산을 다음과 같이 정의한다.
$$\text{complement}(L,K) = \left(d_0, \frac{d_1}{M_0d_0}, \ldots, \frac{d_n}{M_{n-1}d_{n-1}}, \frac{K}{M_nd_n}\right):(1, M_0d_0, \ldots, M_nd_n)$$

### 예시 유도

주어진 값: $L = (2,4):(1,2)$, $K = 16$

- 왼쪽 $(2,4)$: $M_0 = 2, M_1 = 4$
- 오른쪽 $(1,2)$: $d_0 = 1, d_1 = 2$

$M_0 d_0 = 2 \times 1 = 2$ 는 $d_1 = 2$를 나눈다 ✓
$M_1 d_1 = 4 \times 2 = 8$ 은 $K = 16$을 나눈다 ✓
따라서 $(L,K)$는 허용 가능하다.

complement 연산 공식에 따르면:
$$\text{complement}(L,K) = \left(d_0, \frac{d_1}{M_0d_0}, \frac{K}{M_1d_1}\right):(1, M_0d_0, M_1d_1)$$
올바른 수치를 대입하면:
$$\text{complement}(L,K) = \left(1, \frac{2}{2 \times 1}, \frac{16}{4 \times 2}\right):(1, 2 \times 1, 4 \times 2)$$
$$= (1, 1, 2):(1, 2, 8)$$

앞에 두 개의 1이 있으므로 merge할 수 있다.
$$A = 2:8$$

아래 코드를 사용해 계산할 수 있다.

```python
import cutlass               
import cutlass.cute as cute  

@cute.jit
def complement_example():
    """
    Complement in cutlass
    """
    S = (2, 4)
    D = (1, 2)
    L = cute.make_layout(shape=S, stride=D)
    K = 16

    cL = cute.complement(L, K)

    cute.printf("L = {}, cL = {}", L, cL)

complement_example()
```

결과는 위의 유도와 같다.

```shell
L = (2,4):(1,2), cL = 2:8
```

complement 연산은 이렇게 해석할 수 있다.

$A$를 layout과 그 complement 연산의 연결이라고 하자. 연결은 모든 mode를 하나의 layout으로 조합하여 간단히 만들 수 있다. 위 예에서는 다음과 같다.

$$A = (2,4,2):(1,2,8)$$

이것이 우리에게 하나의 전단사 $f_A:[0,M) \to [0,M)$ 를 준다는 것을 증명할 수 있다. 이 블로그의 명제 2.7(https://leimao.github.io/article/CuTe-Layout-Algebra/)을 참조하라.

이를 CuTeDSL로 검증할 수 있다.

```python
import cutlass               
import cutlass.cute as cute  

@cute.jit
def complement_example2():
    """
    Complement in cutlass
    """
    S = (2, 4, 2)
    D = (1, 2, 8)
    L = cute.make_layout(shape=S, stride=D)

    for i in cutlass.range_constexpr(cute.size(L)):
        cute.printf("{} -> {}", i, L(i))
    
complement_example2()

```

```shell
0 -> 0
1 -> 1
2 -> 2
3 -> 3
4 -> 4
5 -> 5
6 -> 6
7 -> 7
8 -> 8
9 -> 9
10 -> 10
11 -> 11
12 -> 12
13 -> 13
14 -> 14
15 -> 15
```

주의하라. 전단사가 반드시 항등 mapping인 것은 아니다.

예를 들어 다음을 취하자.

$$B = 8:2$$ 그리고 $$K = 32$$

$(B,K)$는 허용 가능하다. $8 \times 2 = 16$이 $32$를 나누기 때문이다.

$$\text{complement}(B,K) = (2,2):(1,16)$$

연결한 뒤의 layout은 다음과 같다.

$$A = (8,2,2):(2,1,16)$$

layout 함수의 값을 출력하면 항등 mapping과 같지 않은 전단사가 나온다.

```shell
0 -> 0
1 -> 2
2 -> 4
3 -> 6
4 -> 8
5 -> 10
6 -> 12
7 -> 14
8 -> 1
9 -> 3
10 -> 5
11 -> 7
12 -> 9
13 -> 11
14 -> 13
15 -> 15
16 -> 16
17 -> 18
18 -> 20
19 -> 22
20 -> 24
21 -> 26
22 -> 28
23 -> 30
24 -> 17
25 -> 19
26 -> 21
27 -> 23
28 -> 25
29 -> 27
30 -> 29
31 -> 31
```

## 결론

이 블로그 글이 수학적 개념과 프로그래밍을 연결함으로써 독자에게 간단한 소개를 제공하기를 바란다. 개념에 대한 더 깊은 수학적 설명은 Lei Mao의 블로그(https://leimao.github.io/article/CuTe-Layout-Algebra/)와 CuTe layout 대수에 대한 Jay Shah의 노트를 참조하라.

CuTeDSL의 더 많은 예시는 CUTLASS 저장소를 참조하라.
