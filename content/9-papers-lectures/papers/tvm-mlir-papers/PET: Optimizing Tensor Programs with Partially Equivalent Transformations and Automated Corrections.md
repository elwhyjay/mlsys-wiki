오늘은 OSDI 2021 논문인 《PET: Optimizing Tensor Programs with Partially Equivalent Transformations
and Automated Corrections》를 읽어보겠습니다.

- 논문 링크: https://pacman.cs.tsinghua.edu.cn/~whj/pubs/Pet.pdf
- 오픈소스 코드 링크: https://github.com/thu-pacman/PET

이전에 OSDI 2020의 《Ansor: Generating High-Performance Tensor Programs for Deep Learning》논문도 읽었는데, Ansor가 보다 미시적인 관점에서 코드 생성을 다룬다면, 이번 PET는 보다 거시적인 관점에서 코드 생성을 다룬다고 할 수 있습니다.

Ansor든 PET든 개인적으로 모두 상당히 인상적이라고 생각합니다. 이전에 작성한 Ansor 논문 해설은 이 저장소에 있습니다: https://github.com/BBuf/tvm_mlir_learn . 관심 있는 분들은 한번 읽어보시면 좋고, 이번 글에서는 PET를 살펴보겠습니다.

# 0x1. 제목과 저자

![PET 제목과 저자](https://img-blog.csdnimg.cn/f6f0702ca3034d1f8c6fcaedd8f7ffe3.png)

제목은 다음과 같이 번역할 수 있습니다: **partially equivalent transformation 및 자동 보정에 기반한 Tensor 프로그램 최적화**. 저자 팀은 칭화대학교, CMU, Facebook 등에 소속되어 있습니다. 이 논문의 1저자인 왕하오제(Wang Haojie)는 칭화대학교 소속입니다. 뒤에서 소개하겠지만, 이 논문에서 mutant program 집합을 생성할 때 효율이 가장 높은 K개의 mutant program을 유지하기 위해 TASO의 cost model과 평가 방식을 사용했기 때문에, 저자 중 자즈하오(Jia Zhihao) 대가가 포함되어 있는 것도 놀라운 일이 아닙니다.

# 0x2. 초록
기존 프레임워크들은 그래프 레벨에서 최적화를 할 때 **일반적으로 등가 변환에 기반**합니다. 즉, 변환 전후의 프로그램이 완전히 동등하다는 의미입니다. 여기서 동등하다는 것은 동일한 입력이 주어졌을 때 변환 전후의 프로그램이 반드시 동일한 출력을 얻을 수 있다는 뜻입니다. 그런데 이 논문은 새로운 영역을 개척했습니다. 즉, PET라는 새로운 프레임워크를 만들었으며 **최적화 과정에서 부분적으로 등가인 변환(partially equivalent transformation)을 허용**하고, 완전 등가 변환과 partially equivalent transformation을 조합하여 더 큰 탐색 공간을 탐험할 수 있는 효율적인 탐색 알고리즘을 설계했습니다. 그리고 최종 결과도 비교적 좋습니다.

# 0x3. 머리말
먼저 한 용어의 의미를 짚고 넘어가야 합니다. 바로 **통계적 특성(statistical property)** 입니다. 통계적 특성이란 변환 전후 프로그램이 완전히 수학적으로 등가인 특성을 말합니다. 현재 TVM, TensorFlow, PyTorch, TensorRT 등 프레임워크의 변환 최적화 또는 Pass는 모두 이 특성을 만족합니다. 반면 partially equivalent transformation은 변환 전후 프로그램이 이 통계적 특성을 유지할 것을 **요구하지 않습니다**. 즉, 변환된 프로그램이 동일한 입력에서 원래 프로그램과 비교했을 때 출력의 일부 위치 원소들이 같지 않을 수 있도록 허용합니다. partially equivalent transformation을 지원하면 **(1) 입력 Tensor의 shape과 배열 순서를 변경하여 계산 효율을 높이고 (2) 효율이 더 높은 op로 효율이 낮은 op를 대체하며 (3) 그래프 구조를 변환하여 더 많은 효율적인 최적화 기회를 얻을 수 있습니다**. 그러나 partially equivalent transformation을 지원하는 데에는 두 가지 도전 과제가 있습니다. **첫 번째**는 partially equivalent transformation을 직접 사용하면 모델 정확도가 떨어지므로 동등하지 않은 Tensor 영역을 보정할 필요가 있다는 점입니다. 그러나 어떤 영역이 동등하지 않은지를 빠르게 식별하고 보정 Kernel을 생성하는 것은 매우 어려운 작업이며, 출력의 어떤 위치가 변환 전후에 동등하지 않은지 표시하는 것 또한 난제입니다. **두 번째**는 partially equivalent transformation을 적용한 후 Tensor 프로그램의 탐색 공간이 커진다는 점입니다. 후보 Tensor 프로그램을 생성하는 알고리즘은 그 계산 복잡도를 신중히 관리해야 합니다. **프로그램 옵티마이저**(뒤에 별도의 절에서 설명)는 partially equivalent transformation이 가져오는 이점과 그것이 도입하는 추가 오버헤드 사이의 균형을 맞추고, 완전 등가 변환과 결합하여 고성능 Tensor 프로그램을 얻어야 합니다.

이 논문은 **partially equivalent transformation으로 Tensor 프로그램을 최적화하는 완전히 새로운 프레임워크인 PET를 제안**하며, PET는 주로 3개의 부분으로 구성됩니다.

- **Mutation generator**. 돌연변이 생성기. 입력 Tensor 프로그램에 대해 partially equivalent transformation의 출력 Tensor 프로그램을 생성하는 데 사용됩니다. 각 mutant program과 입력 프로그램은 동일한 입력에서 출력 Tensor의 형태가 동일하지만, 일부 영역의 값은 다를 수 있습니다.
- **Mutation corrector**. 돌연변이 보정기. PET의 mutation corrector는 원본 프로그램과 mutant program 사이의 등가성을 검사하고 보정 Kernel을 자동으로 생성합니다. 그리고 보정 Kernel을 출력 Tensor에 적용하여 전체 변환이 통계적 특성을 만족하도록 보장합니다. 또한 PET는 보정 Kernel을 도입하여 발생하는 추가 오버헤드를 줄이기 위해 보정 Kernel과 Tensor 계산 Kernel을 가능한 한 fusion합니다. partially equivalent transformation을 검사하고 보정하는 것은 매우 어렵습니다. 출력 Tensor가 수백만 개에 달하는 원소를 포함할 수 있으며, 각 출력 원소는 많은 수의 입력 원소와 관련될 수 있기 때문입니다. 하나하나 검증한다면 오버헤드가 매우 클 것입니다. **PET의 핵심 기여 중 하나는 이 검증 과정을 크게 단순화하는 엄밀한 수학 이론을 발견했다는 점입니다**(이 과정의 복잡도를 상수 수준까지 낮추었습니다). 출력 Tensor의 모든 위치를 검사하는 것이 아니라, PET는 몇 개의 대표적인 위치만 검사하면 검증을 완료할 수 있습니다.
- **Program Optimizer.** 먼저 모델이 여러 서브그래프로 분할된 다음, 각 서브그래프에 partially equivalent transformation을 적용하여 더 많은 최적화 기회를 얻습니다. 마지막으로 전체 모델의 각 서브그래프 경계 부분에 일련의 후처리 최적화(중복 제거, op fusion 등)를 적용하여 전체적으로 최적의 성능을 달성합니다.

기여 측면에서는 사실 위의 세 가지인데, PET가 몇 가지 모델에서 평가한 성능을 먼저 언급하겠습니다. ResNet-18에서는 1.2배, CSRNet과 BERT에서는 2.5배의 성능 향상을 보였습니다.


# 0x4. 배경 및 아이디어 출처
이 절은 별로 다룰 내용이 없습니다. Introduction과 약간 중복된다고 느껴지므로, 그림 1만 살펴보면서 partially equivalent transformation이 무엇인지 이해를 돕겠습니다. 우선 그림 1은 다음과 같습니다.

![그림 1](https://img-blog.csdnimg.cn/200bb8fa80384127b2c2bf682130a3f9.png)

먼저 (a)는 일반적인 convolution 연산을 나타냅니다. 여기서 $T_1$은 입력 Tensor이고, 데이터 배치는 [b, c, h, w]로 표기할 수 있습니다. 즉, 배치 크기, 입력 채널 수, 입력 feature map의 높이와 너비입니다. 그런 다음 partially equivalent transformation, 즉 그림 (b)는 reshape와 transpose를 통해 그림에서 **배치 방향**의 인접한 두 feature map을 이어붙입니다. 즉, 다음과 같습니다: [b, c, h, w] -> reshape -> [b / 2, 2, c, h, w] -> transpose -> [b / 2, c, h, w, 2]. 그림에서 $T_1->T_3$ 입니다. 그리고 원본 convolution kernel과 convolution 연산을 수행한 후 $T_4$를 얻고, 다시 reshape와 transpose를 이용해 출력 feature map을 원본 입력 feature map 크기로 복원합니다. 주목할 점은 이 변환 후 출력 Tensor가 경계 부분에서 원본 convolution의 출력 Tensor 값과 같지 않은 경우가 발생한다는 것입니다. 따라서 같지 않은 경계 부분에 대해 보정이 필요하며, 이는 (c) 그림이 보여주는 의미입니다.

뒤의 몇 절에서 **수치적으로 같지 않은 부분이 어디인지 확정하는 방법**과 **이러한 같지 않은 영역들을 어떻게 보정하는지**를 자세히 설명할 것이므로, 지금 이해되지 않아도 괜찮습니다.


# 0x5. 설계 개요
PET는 **partially equivalent transformation을 활용해 Tensor 프로그램을 최적화하는 첫 번째** 프레임워크이며, Tensor 프로그램의 multi-linear 특성을 활용합니다. 먼저 **Multi-linear tensor programs (MLTPs)** 즉 다선형 Tensor 프로그램이 무엇인지 설명해야 합니다. 이후로는 일관되게 MLTPs라는 표현을 사용하겠습니다. n개의 입력 Tensor $I_1, ..., I_n$을 가지는 op가 있을 때, 모든 입력 $I_k$에 대해 선형이라면 이 op는 multi-linear라고 합니다.

![선형의 정의](https://img-blog.csdnimg.cn/a388aa1094254591b5b762122cf1cb8f.png)

여기서 X와 Y는 $I_k$와 동일한 형태를 가지는 임의의 Tensor이며, $\alpha$는 임의의 스칼라입니다. 딥러닝 모델은 일반적으로 선형(Conv, MatMul) 및 비선형(ReLU, Sigmoid 등) operator로 구성되며, PET 프레임워크에서 사용하는 선형 operator는 Table1과 같습니다.

![PET가 사용하는 multi-linear operator](https://img-blog.csdnimg.cn/361a46c6e44a49f18bdd4b9ee4c981a7.png)

이 표는 확장 가능하다는 점에 유의하세요. **프로그램이 multi-linear tensor program(MLTP)이라는 것은 프로그램의 모든 op가 multi-linear인 경우와 동치입니다**. 다음으로 PET의 설계 개요, 즉 Figure2를 살펴보겠습니다.

![PET 개요](https://img-blog.csdnimg.cn/bcfa9db718aa43439a490933723d416e.png)

**먼저 원본 Tensor 프로그램이 PET 프레임워크에 입력되면, PET는 먼저 이 프로그램을 작은 서브 프로그램들로 분해하여 각 서브 프로그램의 탐색 복잡도를 낮춥니다. 각 서브 프로그램에 대해 PET의 Mutation Generator는 서브 프로그램의 MLTPs에 대한 가능한 변형들을 생성하여 partially equivalent transformation 변형 프로그램들을 발견합니다. 각 변형 프로그램은 원본 서브 프로그램과 동일한 입출력 Shape을 갖습니다. 종단간 수치 정확성을 유지하기 위해, PET의 Mutation Corrector는 원본 프로그램과 mutant program 간에 어떤 영역이 같지 않은지 검사하고, 보정 Kernel을 자동으로 생성하여 보정합니다. PET는 엄밀한 수학 이론을 활용해 이 도전적인 작업을 단순화했습니다.**

**보정된 mutant는 PET의 program optimizer로 전달되며, optimizer는 기존의 완전 등가 변환과 partially equivalent transformation을 결합하여 프로그램 최적화를 위한 종합적인 탐색 공간을 구축합니다. Optimizer는 각 서브 프로그램에 대해 풍부한 mutant 집합을 평가하고, 이들의 경계에 후처리 최적화를 적용하여 탐색 공간 안에서 고도로 최적화된 후보 프로그램을 발견합니다.**


# 0x6. Mutation Generator
이 절은 주로 Mutation Generator의 알고리즘 구현 흐름을 설명하고 생성되는 몇 가지 전형적인 돌연변이 패턴을 다룹니다. Mutation Generator의 알고리즘은 아래 그림과 같습니다.

![Mutation Generator Algorithm](https://img-blog.csdnimg.cn/0a045353bd114f88b625e0b55b1de125.png)

먼저 원본 multi-linear tensor program MLTP $P_0$ 와 operator 집합 O가 있습니다. 출력해야 할 것은 적법한 mutant program 집합 $M$ 입니다. 다음으로, $I_0$를 원본 MLTP의 모든 입력 Tensor를 나타내는 것으로 정의합니다. 그리고 M은 빈 집합으로 초기화됩니다. 이어서 BUILD라는 DFS 알고리즘을 실행하여 mutant program 집합을 생성합니다. 8-9번째 줄, 즉 DFS 알고리즘의 반환 조건을 보면, $P$와 $P_0$의 입출력 형태가 완전히 동일할 때 현재 mutant program이 적법함을 의미하므로, 이 mutant program $P$를 집합 $M$에 추가할 수 있습니다. 그리고 n<depth일 때(이 depth는 DFS 재귀의 깊이) 집합 O 안의 op를 계속 순회하며 돌연변이를 진행합니다(11번째 줄). 이어서 각 입력 Tensor $i$를 순회하며, 입력 Tensor $i$가 현재 op에 대해 적법하다면 op를 집합 $P$에 추가하고 그 후로는 일반적인 DFS 동작을 수행합니다.

다음으로 세 가지 전형적인 변형 프로그램을 소개하겠습니다. 간단히 살펴보겠습니다.

- **Reshape + Transpose.** 위의 Figure1에서 이미 이 변형을 설명했습니다. Reshape와 Transpose의 결합을 통해 Tensor의 데이터 배치를 바꿀 수 있습니다. 예를 들어 입력 feature map의 너비를 더 크게 만들면 병렬 계산에 유리합니다. 또한 reshape와 transpose는 자주 함께 사용되므로, PET에서는 이 두 연산을 합쳐 **reshape & transpose** 라고 부릅니다. 이 fusion은 mutant의 크기를 줄이고 더 크고 복잡한 mutant를 탐색할 수 있게 합니다.
- **Single-operator mutants.** PET는 비효율적인 operator를 효율적인 operator로 대체할 수 있습니다. 예를 들어 Dilated Conv를 일반적인 convolution으로 바꾸어 계산 효율을 크게 높일 수 있습니다. Figure3과 같습니다.

![Dilated Conv가 Mutation Generator를 통해 일반적인 Conv 계산으로 변환되어 가속을 얻음](https://img-blog.csdnimg.cn/1a35667158f04d8fa62a2fb3d06748e0.png)

여기서 가속을 얻을 수 있는 이유는 Dilated Conv는 일부 가속 라이브러리에서 큰 최적화가 이루어지지 않은 반면, 일반 convolution은 깊이 있게 최적화되어 있기 때문에 가속 효과가 매우 뚜렷합니다. 여기에도 여전히 보정 과정이 포함된다는 점에 유의하세요.

- **Multi-operator mutants.** 이는 한 operator 집합을 더 효율적인 operator 집합으로 대체하는 것입니다. 예를 들어 InceptionV3에서 비슷한 출력 형태를 가진 일부 Tensor에 대응하는 operator들은 더 큰 convolution으로 결합되어 GPU 활용도를 높이고 Kernel Launch 오버헤드를 줄일 수 있습니다.

# 0x7. Mutation Corrector
PET에서 가장 중요한 단계는 바로 이 Mutation Corrector라고 할 수 있습니다. mutation corrector를 설계할 때는 두 가지 주요 도전 과제가 있습니다. **첫째: 출력 Tensor가 매우 클 수 있어, 등가 검증이 필요한 원소가 수백만 개에 이를 수 있습니다. 출력 Tensor의 각 원소를 개별적으로 검증하는 것은 실현 가능하지 않습니다. 둘째: 각 출력 원소의 검증이 많은 수의 입력 원소에 의존할 수 있습니다. 예를 들어 행렬곱 operator에서 하나의 출력 원소는 두 입력 행렬의 한 행과 한 열의 내적이며, 양쪽 모두 수천 개에 달할 수 있습니다**. 이 두 가지 도전 과제를 해결하기 위해 PET는 두 가지 수학 이론을 제안합니다.

## 0x7.1 이론적 기초
저는 여기서 논문의 서술을 그대로 따르지 않고 제 나름의 이해에 따라 보다 직관적이고 덜 이론적으로 설명하겠습니다. 먼저 $3\times 3$ convolution은 다음 식으로 표현될 수 있습니다.

![3x3 convolution의 수식 표현](https://img-blog.csdnimg.cn/0da52c9535d0449fb42e56e6e910cdb8.png)

여기서 $I_1$과 $I_2$는 각각 입력 Tensor와 convolution Kernel을 나타내며, $D, H, W$는 각각 입력 Tensor $I_1$의 채널 수, 높이, 너비를 나타냅니다. 합 기호 위와 아래의 숫자는 각각 합 구간의 상한과 하한을 나타냅니다. 그리고 이 convolution의 출력 Tensor에 대해 각 원소는 합 영역에 대응합니다. 위에서 정의한 convolution operator의 경우, 좌상단의 출력 위치 즉 $h = 0, w = 0$를 계산하는 데에는 $2\times 2$ Kernel만 관여합니다. 즉 $0<=x<=1, 0<=y<=1$입니다. 이 위치에는 왼쪽이나 위쪽 이웃이 없기 때문입니다. 논문에서는 합 영역이 동일한 위치들을 하나의 Box라고 부르며, 이 convolution 예시의 경우 모든 Box는 Figure 4와 같이 표현할 수 있습니다.

![3x3 convolution 예시는 총 9개의 box를 가짐](https://img-blog.csdnimg.cn/d2cbbcada4ad48f990efa49bff999377.png)

같은 Box의 모든 출력 위치는 동일한 합 구간을 가지며 유사한 수학적 특성을 공유합니다. PET는 프로그램 등가성을 검사할 때 이러한 속성을 활용합니다. PET는 모든 개별 위치에서 두 MLTP의 등가성을 검증할 필요 없이, **각 Box에서 m+1개의 대표 위치에서의 등가성만 검증하면 되며, 여기서 m은 출력 Tensor의 차원 수를 나타냅니다**. 이 정리의 증명은 논문에서 $P_1$과 $P_2$의 입력 변수에 대한 계수 행렬을 비교하여 완성된다고 언급합니다. 여기서는 구체적인 증명 과정에 신경 쓸 필요 없이, 이 정리에 기반하여 등가성 검증 시 모든 출력 원소를 검사할 필요가 없어진다는 점만 알면 됩니다. 이는 보정 검사의 복잡도를 크게 낮춰줍니다.

두 번째 정리의 의미는 다음과 같습니다. 만약 두 개의 $n$개 입력 Tensor를 가지는 MLTPs가 특정 위치 v에서 등가가 아니라면, 범위 F의 분포에서 무작위로 샘플링하여 입력으로 사용할 때, 이 위치 v에서 두 MLTPs가 동일한 출력 값을 산출할 확률은 $n / p$ 입니다. 여기서 $p$는 F의 범위를 나타내며, 이 논문에서 $p$는 매우 큰 소수, 즉 $2^{32}-1$ 입니다.

위의 두 정리를 통해 PET는 매우 적은 위치에서만 검증을 수행해도 어떤 위치가 원본 프로그램과 등가가 아닌지 결정할 수 있습니다. 아래 그림은 정리 1과 정리 2가 검증해야 할 입력 원소 수에 미치는 영향을 보여줍니다.

![Table2](https://img-blog.csdnimg.cn/49ffeea3d2924b8fb2de7a640da91631.png)


## 0x7.2 Mutation Correction 알고리즘

위의 두 정의를 바탕으로 Mutation Correction 알고리즘을 도출할 수 있습니다. 이 알고리즘은 다음 세 단계로 나뉩니다.
- **Step 1: Box propagation**. 첫 번째 단계는 Box Propagation을 통해 주어진 MLTP의 값을 계산하는 것입니다. PET는 Tensor의 각 차원에 대해 분할점 집합을 유지하여 Box의 경계를 식별합니다. multi-linear operator의 경우, **입력 Tensor의 분할점과 operator 종류 및 하이퍼파라미터에 기반하여 출력 Tensor의 분할점을 추론**합니다. Figure5는 Figure1의 돌연변이 예시에 대한 Box 전파 과정을 보여줍니다.

![Box 전파 알고리즘 예시](https://img-blog.csdnimg.cn/7e0608fb09f74c40a686d466aa0aadf2.png)

- **Step 2: Random testing for each box pair**. 두 번째 단계는 입력 MLTP $P_1$과 그 변형 $P_2$에 대해 앞 절에서 소개한 정리를 적용하여 어떤 영역이 수치적으로 등가인지 판단하는 것입니다. 정리 1에 따르면 먼저 m+1개의 위치를 선택해야 하는데, Figure5의 예시에서는 m=4 입니다. 그런 다음 이 m+1개의 위치 각각에 대해 무작위 데이터를 기반으로 $t$회 검사하면, 오판 확률은 $(n/p)^t$가 됩니다. 여기서 $p=2^{32}-1$이며, $t$는 프로그램 검사 오버헤드와 오류 가능성 사이의 균형을 맞추는 데 사용되는 조정 가능한 하이퍼파라미터입니다.
- **Step 3: Correction kernel generation**. 마지막 단계는 모든 출력 Tensor의 수치적으로 같지 않은 영역에 대해 보정 Kernel을 자동으로 생성하는 것입니다. 보정 Kernel의 오버헤드를 줄이기 위해, PET는 보정 Kernel을 기존 계산 Kernel과 가능한 한 fusion합니다.

## 0x7.3 Correction Kernels의 fusion

이 부분은 위의 Step3에 대한 설명입니다. Figure6을 봐주세요.

![Figure6](https://img-blog.csdnimg.cn/b2a1cc1ac1bf4658afd03a391b5452de.png)

Figure6(a)는 표준 convolution 과정을 나타냅니다. 그리고 Figure6(b)는 Figure6(a)에 partially equivalent transformation을 적용한 것을 나타냅니다. Conv-2는 보정 Kernel이며, 여기서 Conv-1과 가중치를 공유하기 때문에 conv1과 conv2를 fusion하여 conv-1-2로 만들 수 있습니다. Figure6(c)와 같습니다. 구체적으로 이 fusion 연산은 $T_1$과 $T_0^{'}$를 단일 Tensor로 결합하고, Conv-1-2의 출력 결과를 $T^2$와 $T_3^{'}$로 분해하는 것입니다. 여기서의 결합과 분해는 데이터 복사만 포함하며, reshape와 transpose로 수행할 수 있습니다.

# 0x8. Program Optimizer
이 절은 PET의 program optimizer를 소개합니다. 등가 변환과 partially equivalent transformation을 결합하여 더 큰 프로그램 최적화 탐색 공간을 탐험할 수 있습니다. **먼저 program optimizer는 입력 프로그램을 더 작은 서브 프로그램으로 분해하여 Mutation Generator로 전달합니다. 그런 다음 각 서브 프로그램을 최적화하기 위해, PET는 풍부한 탐색 공간 안에서 돌연변이에 참여하는 Op 집합과 DFS 탐색 알고리즘의 반복 횟수를 조정하면서 가장 좋은 변형 프로그램을 찾습니다. 마지막으로 모든 최적화된 서브 프로그램을 함께 이어붙일 때, 경계를 가로지르는 추가적인 후처리 최적화(op fusion, 중복 Op 제거 등)를 적용합니다**. 아래의 알고리즘 2는 program optimizer의 전체 흐름을 설명합니다.

![program optimizer의 흐름](https://img-blog.csdnimg.cn/8652855d18bc480e91b2557d11d036ab.png)

전체 알고리즘 흐름은 복잡하지 않습니다. 먼저 8번째 줄에서 전체 프로그램을 잘라 여러 서브 프로그램으로 만듭니다. 각 서브 프로그램 $S$에 대해 GETMUTANTS 함수를 사용해 이 서브 프로그램의 mutant program 집합 mutants를 생성합니다(9번째 줄에 해당). 그런 다음 새로운 스택 $H_{new}$를 초기화합니다. 다시 원본 스택 $H$를 순회하면서, 그 안의 각 프로그램 $P$에 대해 방금 얻은 돌연변이 결과 $M$을 기반으로 $P$ 안의 서브 프로그램에 대해 돌연변이를 수행하여 새로운 mutant program $P_{new}$를 얻고, $P_{new}$를 새로운 스택 $H_{new}$에 push합니다. 마지막으로 원본 스택 $H$를 $H_{new}$로 업데이트하여 현재 서브 프로그램의 돌연변이를 완료합니다.

마지막으로 스택 $H$에서 가장 좋은 성능을 보이는 프로그램을 선택해 후처리 최적화를 진행하면 최종 결과를 얻게 됩니다.

전체 알고리즘 흐름에서 몇 가지 세부 사항에 주의해야 합니다.
- Detail1. 원본 프로그램을 어떻게 분할할 것인가. 논문에서는 ReLU, Sigmoid 등 비선형 Op를 분할 지점으로 삼습니다. 이는 PET의 한 가지 한계이기도 합니다. multi-linear operator로 구성된 서브그래프만 돌연변이시킬 수 있습니다.
- Detail2. 알고리즘의 스택은 성능이 가장 높은 K개의 서브 프로그램을 보존해야 하며, 여기서는 이전 논문 TASO의 cost model 및 성능 평가 방법을 활용합니다.
- Detail3. 서브 프로그램 돌연변이 과정의 탐색 공간 크기와 탐색에 필요한 시공간 비용 사이의 균형을 위해, PET의 program optimizer는 두 개의 하이퍼파라미터를 도입합니다. 하나는 돌연변이의 반복 횟수, 즉 알고리즘 2의 23번째 줄입니다. 또 하나는 서브 프로그램의 Op 개수가 d(여기서는 4)를 초과할 때, PET는 최대 d개의 Op의 모든 가능한 조합을 열거하여 서브 프로그램을 더 작은 Op 부분 집합으로 나누고, 돌연변이는 이 부분 집합에서만 발생하며 다른 Op는 변경하지 않습니다. 이는 알고리즘 2의 26번째 줄에 해당합니다.
- Detail4. PET의 program optimizer는 완전 등가 변환과 완전히 호환됩니다. 완전 등가 변환과 partially equivalent transformation을 조합하여 더 큰 탐색 공간을 탐험할 수 있습니다.

**이 절의 마지막에는 Post-Optimizations에 대해 설명합니다**. 위에서 언급했듯이, 마지막에 모든 서브 프로그램의 mutant들을 이어붙여야 합니다. 그것들의 입력과 출력 Tensor를 연결하는 것 외에도, PET는 **서브 프로그램 경계를 가로지르는 일부 후처리 최적화를 수행하여 프로그램 성능을 더욱 향상**시킵니다. PET의 mutation generator가 서브 프로그램의 시작과 끝 부분에서 특히 많은 Reshape와 Transpose(R/T) operator를 생성한다는 점에 주목할 수 있습니다. 따라서 서브 프로그램을 가로질러 이러한 R/T operator들을 fusion하고, 위에서 언급한 서브 프로그램 최적화에서 제외된 비선형 operator들도 함께 fusion할 기회가 있습니다. Figure7은 두 개의 최적화된 서브 프로그램을 포함하는 예시를 보여줍니다. 서브 프로그램의 경계를 최적화하기 위해, PET는 먼저 비선형 operator와 R/T operator를 재배치하여 두 서브 프로그램 사이의 R/T operator들을 함께 묶습니다. Figure7(b)와 같이, 이 재배치의 정당성은 완전히 보장됩니다. 재배치는 또한 PET가 비선형 활성화 operator를 다른 operator와 fusion할 수 있게 합니다. 예를 들어 Conv와 ReLU를 Conv-ReLU로 fusion하는 것입니다. Figure7(c)와 같습니다.


![후처리 최적화 예시](https://img-blog.csdnimg.cn/69d6cbc835504a339eb6009a16a517c6.png)

따라서 여기에는 세 가지 최적화가 포함됩니다.
- **역원 소거**. 서로 상쇄될 수 있는 R/T operator 쌍을 모두 제거하므로, 결과적으로 no-op과 등가가 됩니다. 이러한 각 쌍을 역원 그룹(inverse group)이라고 부르며, 후처리 최적화 과정에서 직접 삭제합니다. 역원 그룹의 한 예시는 그림 7(b)의 R/T-E와 R/T-G입니다.
- **Operator fusion**. 그림 7(c)와 같이, PET는 남은 연속된 R/T operator들을 단일 operator(예: R/T-DH)로 fusion하여 Kernel Launch 비용을 줄입니다. Tensor 프로그램의 비선형 활성화도 R/T 또는 다른 선형 operator와 fusion됩니다. Operator fusion은 가장 일반적으로 사용되는 비선형 operator 프로그램 최적화입니다. PET는 원본 Tensor 프로그램을 분할할 때 손실된 효율의 대부분을 회복할 수 있습니다.
- **사전 처리**. 모든 입력 Tensor가 정적으로 알려져 있는 경우, 어떤 operator든 사전 처리할 수 있습니다. 예를 들어, 그림 7(b)에서 R/T-B와 R/T-I는 모두 convolution 가중치 Tensor w1과 w2 위에서 사전 처리할 수 있습니다. 사실 이는 **상수 폴딩(constant folding)** 입니다.


# 0x9. 구현
그들의 코드는 오픈소스로 공개되어 있으며, 약 13000줄의 C++ 코드와 1000줄의 Python 코드로 구성됩니다. 이 작업에 관심이 있다면 소스 코드를 살펴볼 수 있습니다. 주소는 다음과 같습니다: https://github.com/thu-pacman/PET .

# 10. 평가
논문에서 제시한 실험 결과는 매우 풍부합니다. 여기서는 각 실험 결과 도표를 자세히 다루지는 않고, 가장 중요한 실험 결과 그림 한 개만 다루겠습니다.

![PET의 다양한 네트워크에서의 성능](https://img-blog.csdnimg.cn/d55032f265824619a7ce142b5e879565.png)

ResNet-18, CSRNet, InceptionV3, BERT, ResNet33D-18에서 현재 일부 인기 있는 프레임워크들과 비교했을 때 모두 뚜렷한 우위를 보입니다. 다만 아쉽게도 여기에는 PyTorch와의 결과 비교가 없습니다.

실험 부분에서는 또한 PET가 TVM 및 Ansor와 쉽게 결합될 수 있어, 생성된 Tensor 프로그램의 효율을 더욱 향상시킬 수 있다고 언급합니다. **PET 안에서는 cuDNN/cuBLAS/TVM/Ansor 등 인기 있는 최적화 라이브러리와 코드 생성 컴파일러를 백엔드로 사용하여 효율적인 Tensor 프로그램을 생성할 수 있습니다**. Figure12는 이러한 프레임워크들이 백엔드로 사용될 때 일반적인 단일 operator의 가속 효과를 보여줍니다.

![다양한 프레임워크 또는 라이브러리가 백엔드로 사용될 때 일반적인 단일 operator의 가속 효과](https://img-blog.csdnimg.cn/cb68a15f817d433599d484704a213508.png)

이는 PET의 확장성이 비교적 우수함을 보여주며, 대부분의 선진적인 코드 생성 작업과 수동 최적화된 operator 라이브러리를 결합할 수 있습니다.

# 11. 결론
이 논문은 PET를 제안했으며, 이는 partially equivalent transformation을 Tensor 프로그램에 적용한 DNN 프레임워크입니다. partially equivalent transformation을 적용하여 더 큰 프로그램 탐색 공간을 탐험할 수 있고, 대부분의 인기 있는 딥러닝 네트워크에서 좋은 가속 효과를 얻을 수 있습니다. 이 논문의 실험 부분은 매우 탄탄하므로 학습해보시기를 권장합니다.









