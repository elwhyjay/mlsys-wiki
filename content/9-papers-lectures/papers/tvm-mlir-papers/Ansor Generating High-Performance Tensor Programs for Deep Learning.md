# Ansor 조사 보고서

- 논문 링크: https://arxiv.org/abs/2006.06762   OSDI 2020
- 사전 지식
  - scheduler에 대한 이해가 필요합니다. 다음 글을 추천합니다: https://zhuanlan.zhihu.com/p/94846767 .
  - Ansor 논문에서는 주로 parallel, cache_read, reorder, unroll, vectorize 같은 scheduler를 사용해 알고리즘 전체를 설명하지만, Ansor의 TVM 오픈소스 구현에서는 이들에 한정되지 않으므로 논문을 읽기 전에 한 번 살펴보는 것을 권장합니다.
  - AutoTVM과 Ansor 이전에는 고성능 텐서화 프로그램을 생성하려면 수동으로 template를 지정해야 했고, 이 template에는 high-level scheduler뿐 아니라 low-level 계산 로직까지 포함되어야 했습니다. 이는 CPU/GPU/ASIC 등 칩의 하드웨어마다 텐서 계산 방식이 다르기 때문입니다. 천 톈치(Tianqi Chen)의 TVM 발표 영상을 추천합니다: https://www.bilibili.com/video/BV1vW411R7Zb?from=search&seid=5300327607655608948 .
- 개인적인 이해
  - Ansor는 어떻게 텐서화 프로그램을 자동으로 생성할까요? 먼저 서브그래프 분할이 필요하며, 분할 규칙은 TVM의 operator fusion과 동일합니다. 각 서브그래프에 대해서는 sketch와 annotation을 통해 대응되는 프로그램을 생성합니다.
  - sketch는 앞서 소개한 scheduler들과 몇 가지 추론 규칙(derivation rule)에 기반해 만들어집니다. 추론 규칙이란 무엇일까요? 예를 들어 convolution이나 matrix multiplication 같은 연산 집약적(compute-intensive) operator에 대해, CPU에서 Ansor는 "SSRSRS"라는 tile 규칙을 정의합니다. matrix multiplication의 경우 "SSRSRS" tile 규칙은 원래의 3중 for 루프 $(i, j, k)$를 $(i_0,j_0,i_1,j_1,k_0,i_2,j_2,k_1,i_3,j_3)$로 확장합니다. 이것이 논문의 Figure 5에 나오는 첫 번째 예시입니다. 이 tile 규칙 외에도 데이터 재사용 operator와 단순 operator의 fusion, 데이터 재사용이 없는 단순 operator에 대한 inline 최적화, 그리고 데이터 재사용 operator의 입력에 대한 계산 윈도우 분할 및 입력의 Cache Read, Cache Write, rfactor 등 여러 규칙이 있습니다. GPU와 CPU는 아키텍처가 다르므로 정의된 규칙도 완전히 동일하지는 않습니다. 예를 들어 matrix multiplication의 다단계 tiling 구조는 "SSRSRS"에서 "SSSRRSRS"로 바뀌어 GPU 아키텍처에 맞춰집니다. tiling의 앞 세 공간 루프는 각각 BlockIdx, virtual thread(bank conflict 감소를 위해 사용), ThreadIdx에 바인딩됩니다. 또한 사용자가 직접 규칙을 정의할 수도 있습니다.
  - annotation은 sketch를 토대로 GPU thread bind, for 루프의 unroll·vectorize·parallelize, Split의 factor 등을 무작위로 결정해 완성된 코드를 생성하는 단계입니다. (외부 루프 병렬화, 내부 루프의 vectorize와 unroll, 이는 GEMM 최적화의 핵심 아이디어와 일치합니다.) 완성된 코드를 생성하더라도 그 코드의 성능은 Evolutionary Search로 보장됩니다. 또한 사용자가 annotation도 직접 정의할 수 있습니다.
  - search space를 더 효율적으로 순회하고 무의미한 search space를 가지치기하는 방법에 대해서는 원 논문의 관련 설명을 참고하시기 바랍니다.
- 장단점
  - Ansor는 매우 큰 search space를 갖기 때문에, 주요 DNN 모델과 하드웨어에서 모두 성능이 우수한 프로그램을 찾아낼 수 있습니다. GPU에서는 TensorRT를, CPU에서는 TensorflowLite, AutoTVM 등을 능가할 수 있습니다.
  - 하지만 Ansor의 search 시간은 TVM 대비 크게 단축되지는 않으며, ResNet50을 TensorRT 이상의 성능으로 search하려면 수 시간이 필요합니다.
- 개선 가능 지점에 대한 고찰
  - search 시간을 어떻게 줄일 수 있을까?
  - 서브그래프 분할 단위가 비교적 세밀한데, 서브그래프가 너무 많아지면 최적화 알고리즘이 local optimum에 빠지지는 않을까?
  - NC4HW4, Winograd 등을 추론 규칙에 추가할 수 있을까?
- 논문 번역
  - Ansor를 더 잘 이해하기 위해 논문을 번역해 보았으니, 여러분의 피드백을 환영합니다.

# Ansor: Generating High-Performance Tensor Programs for Deep Learning

# 초록

고성능 텐서화 프로그램은 deep neural network의 효율적인 실행을 보장하는 데 매우 중요합니다. 그러나 다양한 하드웨어 플랫폼에서 서로 다른 operator마다 효율적인 텐서화 프로그램을 얻는 일은 쉽지 않습니다. 현재 딥러닝 시스템들은 하드웨어 벤더가 제공하는 kernel 라이브러리나 다양한 search 전략에 의존해 고성능 텐서화 프로그램을 얻고 있습니다. 이러한 방식은 플랫폼별 최적화 코드를 개발하기 위해 막대한 엔지니어링 노력을 요구하거나, 제한된 search space와 비효율적인 탐색 전략 때문에 고성능 프로그램을 찾아내지 못합니다.

본 논문에서는 딥러닝 응용을 위한 텐서화 프로그램 생성 프레임워크인 Ansor를 제안합니다. 기존 search 전략과 비교했을 때 Ansor는 search space의 계층적 표현으로부터 프로그램을 sampling함으로써 더 많은 최적화 조합을 탐색합니다. 그런 다음 evolutionary search와 학습 가능한 cost model을 사용해 sampling된 프로그램을 미세 조정하여 최적의 프로그램을 결정합니다. Ansor는 기존 SOTA 방법의 search space 바깥에 존재하는 고성능 프로그램도 찾아낼 수 있습니다. 또한 Ansor는 scheduler를 활용해 deep neural network 내 여러 서브그래프를 동시에 최적화합니다. 실험 결과 Intel CPU, ARM CPU, NVIDIA GPU에서 Ansor는 신경망 실행 성능을 각각 3.8배, 2.6배, 1.7배 향상시킵니다.

# 1. 서론

Deep neural network(DNN)의 저지연(low-latency) 추론은 자율주행[14], 증강현실[3], 언어 번역[15] 등 다양한 AI 응용에서 핵심적인 역할을 합니다. DNN은 directed acyclic graph(DAG)로 표현되며, 노드는 operator(예: Conv, Matmul)를, directed edge는 operator 사이의 의존성을 나타냅니다. 기존 딥러닝 프레임워크(예: TensorFlow, PyTorch, MxNet)는 DNN의 operator를 하드웨어 벤더가 제공하는 고성능 kernel 라이브러리(예: cuDNN, MKL-DNN)에 기반해 구현하여 성능을 확보합니다. 그러나 이러한 kernel 라이브러리들은 각 하드웨어 플랫폼과 각 operator마다 수작업 튜닝을 위해 막대한 엔지니어링 노력이 필요합니다. 각 타깃 가속기마다 효율적인 operator 구현을 만들어내는 데 드는 막대한 수작업은 새로운 operator[7]와 전용 가속기[35]의 개발과 혁신을 가로막고 있습니다.

DNN 성능의 중요성을 고려할 때, 연구자들과 산업계 종사자들은 텐서 프로그램, 즉 텐서 operator의 low-level 구현을 자동으로 생성하기 위해 search 기반 컴파일 기술[2, 11, 32, 49, 59]로 눈을 돌려 왔습니다. 사용자는 하나의 operator나 여러 operator로 구성된 서브그래프에 대해 high-level의 선언적 언어로 계산을 정의하고, 컴파일러가 다양한 하드웨어 플랫폼에 맞춘 프로그램을 search합니다.

고성능 텐서화 프로그램을 찾아내려면 search 기반 방법이 모든 텐서화 프로그램 최적화 전략을 포괄하기에 충분히 큰 search space를 탐색해야 합니다. 그러나 기존 방법들은 사전 정의된 수작업 template(예: TVM[12], FlexTensor[59])에 의존하거나, 미완성 프로그램을 평가(예: Halide auto-scheduler)함으로써 튜닝하기 때문에 효과적인 최적화 조합을 포착하지 못하며, 이로 인해 포괄적인 search space를 다루지 못합니다. search space를 구성하기 위해 사용하는 규칙도 제한적입니다.

본 논문에서는 고성능 텐서화 프로그램을 생성하기 위한 새로운 search 전략을 탐구합니다. 이 전략은 다양한 최적화 전략을 폭넓게 포괄하는 큰 search space를 자동으로 구성할 수 있으므로, 기존 방법이 놓친 고성능 프로그램을 찾아낼 수 있습니다.

이 목표를 달성하는 데에는 여러 가지 어려움이 있습니다. 첫째, 주어진 계산 정의에 대해 가능한 한 많은 텐서화 프로그램을 포괄하도록 큰 search space를 자동으로 구성해야 합니다. 둘째, 기존 template이 다룰 수 있는 범위보다 수십 배 더 큰 search space에서 미완성 프로그램들을 비교하지 않고도 효율적으로 search해야 합니다. 마지막으로, 서브그래프가 많은 DNN 전체를 최적화할 때 end-to-end 성능에 결정적인 서브그래프를 식별하고 우선시해야 합니다.

이를 위해 우리는 텐서화 프로그램 자동 생성 프레임워크인 Ansor를 설계하고 구현했습니다. Ansor는 계층적 표현을 활용해 큰 search space를 다룹니다. 이 표현은 high-level 구조와 low-level 세부 정보를 분리하여, high-level 구조는 유연하게 열거하고 low-level 세부 정보는 효율적으로 sampling할 수 있게 합니다. 계산 정의가 주어지면 search space가 자동으로 구성되고, Ansor는 이 search space에서 완성된 프로그램을 sampling한 뒤 evolutionary search와 학습 가능한 cost model로 sampling된 프로그램을 미세 조정합니다. 여러 서브그래프를 가진 DNN의 성능을 최적화하기 위해, Ansor는 end-to-end 성능을 더 많이 향상시킬 가능성이 큰 서브그래프를 동적으로 우선순위에 둡니다.

우리는 표준 딥러닝 벤치마크와 search 기반 튜닝 프레임워크가 제공하는 벤치마크에서 평가를 진행했습니다. 실험 결과 Ansor는 Intel CPU, ARM CPU, NVIDIA GPU에서 DNN 실행 성능을 각각 3.8배, 2.6배, 1.7배 향상시킵니다. 대부분의 계산 정의에서 Ansor가 찾아낸 최적의 프로그램은 기존 search 기반 방법의 search space 바깥에 위치합니다. 또한 search space가 더 크면서도 기존 search 기반 방법보다 더 효율적으로 search하여 더 짧은 시간에 더 높은 성능의 프로그램을 생성합니다. Ansor는 현재 SOTA 프레임워크(TVM처럼 자동 튜닝이 가능한 프레임워크를 의미)와 동등한 성능을, 한 자릿수 단축된 시간에 얻을 수 있습니다. 게다가 Ansor는 새로운 operator의 수학적 정의만 있으면 자동으로 확장 가능하며, template를 수동으로 작성할 필요가 없습니다.

정리하면, 본 논문은 다음과 같은 기여를 합니다.

- 계산 그래프에 대한 텐서 프로그램을 생성하기 위한, 큰 공간을 다루는 계층적 search 메커니즘.
- 학습 가능한 cost model에 기반한 evolutionary 전략으로 텐서화 프로그램의 성능을 미세 조정.
- DNN의 end-to-end 성능을 최적화할 때 중요한 서브그래프에 우선순위를 부여하는 gradient descent 기반 schedule 알고리즘.
- 위 기법들이 다양한 DNN과 하드웨어 플랫폼에서 SOTA 시스템을 능가함을 보이는 Ansor 시스템의 구현 및 종합 평가.

# 2. 배경

딥러닝 생태계는 CPU, GPU, FPGA, ASIC 등 빠르게 늘어나는 다양한 하드웨어 플랫폼을 받아들이고 있습니다. 이러한 플랫폼에서 DNN을 배포하기 위해서는 DNN이 사용하는 operator에 대해 고성능 텐서화 프로그램이 필요합니다. 필요한 operator에는 일반적으로 표준 operator(예: matmul, conv2d)와 머신러닝 연구자가 새롭게 제안한 operator(예: capsule conv2d[23], dilated conv2d[57])가 혼합되어 있습니다.

다양한 하드웨어 플랫폼에서 이러한 operator를 배포할 때 좋은 성능을 유지하기 위해 여러 컴파일러 기술(예: TVM[11], Halide[41], Tensor Comprehensions[49])이 도입되어 왔습니다. 사용자는 high-level의 선언적 언어로 수학식과 유사한 형태로 계산을 정의하고, 컴파일러는 그 정의로부터 최적화된 텐서 프로그램을 생성합니다. Figure 1은 TVM tensor expression 언어로 표현한 matrix multiplication의 정의를 보여줍니다. 사용자는 주로 텐서의 형태와 출력 텐서의 각 원소를 어떻게 계산할지를 정의하면 됩니다.

![Figure1](https://img-blog.csdnimg.cn/20210717160412873.png)

그러나 high-level 정의로부터 고성능 텐서 프로그램을 자동으로 생성하는 것은 매우 어렵습니다. 컴파일러는 타깃 플랫폼의 아키텍처에 따라 tile, vectorization, parallelization 등 다양한 최적화 방식을 포함한 매우 복잡하고 거대한 공간에서 search해야 합니다. 고성능 프로그램을 찾으려면 search 전략이 종합적인 공간을 포괄하면서 효율적으로 탐색해야 합니다. 본 절에서는 가장 최신의 효과적인 두 가지 방법을 소개하고, 다른 관련 연구는 8장에서 다룹니다.

**Template-guided search** Template 기반 search에서는 search space가 수작업 template로 지정됩니다. Figure 2 a처럼 컴파일러(예: TVM)는 사용자에게 계산 정의용 template를 직접 작성하도록 요구합니다. 이 template는 몇 가지 조정 가능한 매개변수(예: tile size, unrolling factor)로 텐서화 프로그램의 구조를 정의합니다. 그러면 컴파일러는 특정 입력 형태와 특정 하드웨어 타깃에 대해 이 매개변수들의 최적값을 search합니다. 이 방식은 일반적인 딥러닝 operator에서 좋은 성능을 보여 왔습니다. 그러나 template를 개발하려면 막대한 엔지니어링 노력이 필요합니다. 예를 들어 TVM 코드베이스에는 수작업 template를 포함한 코드가 이미 15,000줄이 넘으며, 새로운 operator와 하드웨어 플랫폼이 등장하면서 그 양은 계속 늘어나고 있습니다. 또한 양질의 template를 개발하려면 텐서 operator와 하드웨어에 대한 전문 지식이 필요하고, 상당한 연구 노력이 요구됩니다[32, 55, 59]. template 설계 자체가 복잡함에도 불구하고, 수동으로 지정한 template는 모든 operator의 모든 최적화 선택을 일일이 열거하는 것이 비현실적이기 때문에 제한적인 프로그램 구조만 다룰 수 있습니다. 이 방식은 보통 operator마다 별도의 template를 정의해야 합니다. Flex-Tensor[59]는 여러 operator를 다룰 수 있는 범용 template를 제안했지만, 그 template 역시 단일 operator를 위해 설계되었으며 여러 operator가 관여하는 최적화(예: operator fusion)는 포함하지 않습니다. 여러 operator를 가진 계산 그래프 최적화의 search space에는 이러한 operator들을 결합하는 다양한 방식이 포함되어야 합니다. template 기반 방식은 search 도중에 고정된 template를 분해하고 재조합할 수 없기 때문에 이를 달성할 수 없습니다.

![Figure 2](https://img-blog.csdnimg.cn/20210717162322717.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

**Sequential construction based search.** 이 방식은 프로그램 생성을 일련의 고정된 의사결정 시퀀스로 분해하여 search space를 정의합니다. 그런 다음 컴파일러는 beam search[34] 같은 알고리즘을 사용해 좋은 결정을 search합니다(예: Halide auto-scheduler[2]). 이 방식에서 컴파일러는 계산 그래프의 모든 노드를 순차적으로 unfold하면서 텐서화 프로그램을 구성합니다. 각 노드에 대해 컴파일러는 그 노드를 어떻게 low-level 텐서화 프로그램으로 변환할지에 대한 결정(예: computation location, storage location, tile size 등)을 내립니다. 모든 노드가 unfold되면 완성된 텐서화 프로그램이 만들어집니다. 이 방식은 노드마다 일반화된 unfold 규칙을 사용하므로 수작업 template 없이 자동 search가 가능합니다. 각 결정의 가능한 선택지가 매우 많기 때문에 sequential 진행을 가능하게 하기 위해 이 방식은 각 결정 후 상위 k개의 후보 프로그램만을 유지합니다. 컴파일러는 학습 가능한 cost model을 기반으로 후보 프로그램들의 성능을 평가·비교하여 상위 k개를 선택하고, 나머지는 버립니다. search 과정에서 후보 프로그램들은 미완성 상태입니다. 계산 그래프의 일부만 unfold되었거나 일부 결정만 내려진 상태이기 때문입니다. Figure 2 b가 이 과정을 보여줍니다.

그러나 미완성 프로그램의 최종 성능을 평가하는 데에는 여러 어려움이 있습니다. (1) 완성된 프로그램으로 학습된 cost model은 미완성 프로그램의 최종 성능을 정확하게 예측할 수 없습니다. cost model은 프로그램을 컴파일하고 실행 시간을 측정해 학습 라벨을 얻어야 하므로 완성된 프로그램으로만 학습할 수 있습니다. 이 모델을 그대로 사용해 미완성 프로그램의 최종 성능을 예측하면 정확도가 떨어집니다. 우리는 search space에서 무작위로 추출한 20,000개의 완성된 프로그램으로 cost model을 학습시켜 사례 연구를 수행했고(5.2절), 이 모델로 미완성 프로그램의 최종 성능을 예측해 보았습니다. 미완성 프로그램은 완성된 프로그램의 일부 루프 변환만 적용하여 얻었습니다. 평가에는 두 가지 지표(쌍별 비교 정확도와 top-k 프로그램의 recall, k=10)를 사용했습니다. Figure 3에서처럼 두 곡선은 각각 50%와 0%에서 시작하는데, 이는 정보가 전혀 없는 무작위 추측이 50% 쌍별 비교 정확도와 0% top-k recall을 준다는 의미입니다. 두 곡선은 프로그램의 완성도가 높아질수록 빠르게 상승합니다. 이는 cost model이 완성된 프로그램에서는 매우 잘 동작하지만 미완성 프로그램의 최종 성능은 정확하게 예측하지 못함을 보여줍니다. (2) sequential한 결정 순서가 search space 설계를 제한합니다. 예를 들어 어떤 최적화는 계산 그래프에 새로운 노드를 추가해야 합니다(예: 캐시 노드 추가, rfactor[46] 사용). 그러면 프로그램별 결정 수가 달라져서 미완성 프로그램들을 공정하게 비교하도록 정렬하기 어려워집니다. (3) sequential construction 기반 search는 확장성이 좋지 않습니다. search space를 확장하려면 더 많은 sequential construction 단계를 추가해야 하지만, 이는 누적 오류를 더 심각하게 만듭니다.



![Figure3](https://img-blog.csdnimg.cn/20210717174707527.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

**Ansor's hierarchical approach** Figure 2-c처럼 Ansor는 high-level 구조와 low-level 세부 정보를 분리한 계층적 search space 위에 구축됩니다. Ansor는 계산 그래프의 search space를 자동으로 구성하므로 template를 수동으로 개발할 필요가 없습니다. 그런 다음 Ansor는 이 공간에서 완성된 프로그램을 sampling하고 그 완성된 프로그램에 대해 미세 조정함으로써, 미완성 프로그램에 대한 부정확한 추정을 피합니다. Figure 2는 Ansor 방식과 기존 방식 간의 주요 차이를 보여줍니다.

# 3. 설계 개요

Ansor는 자동 텐서화 프로그램 생성 프레임워크입니다. Figure 4는 Ansor의 전반적인 구조를 보여줍니다. Ansor의 입력은 최적화 대상 DNN 집합입니다. Ansor는 Relay[42]의 operator fusion 알고리즘을 사용해 DNN을 인기 있는 모델 포맷(예: ONNX, TensorFlow PB)에서 작은 서브그래프들로 변환한 뒤, 이 서브그래프에 대한 텐서화 프로그램을 생성합니다. Ansor는 다음 세 가지 핵심 구성 요소를 가집니다. (1) 큰 search space를 구축하고 그 안에서 다양한 프로그램을 sampling하는 program sampler. (2) sampling된 프로그램의 성능을 미세 조정하는 performance tuner. (3) DNN의 여러 서브그래프를 최적화하기 위해 시간 자원을 할당하는 task scheduler.

**Program sampler.** Ansor가 해결해야 하는 핵심 과제 중 하나는 주어진 계산 그래프에 대해 큰 search space를 생성하는 것입니다. high-level 구조와 low-level 세부 사항이 다양한 텐서화 프로그램을 폭넓게 포괄하기 위해, Ansor는 두 단계의 search space를 가진 계층적 표현, 즉 sketch와 annotation(4장)을 활용합니다. Ansor는 프로그램의 high-level 구조를 sketch로 정의하고, 수십억 가지의 low-level 선택(예: tile size, parallel·unroll annotation)을 annotation으로 둡니다. 이 표현은 Ansor가 high-level 구조를 유연하게 열거하고 low-level 세부를 효율적으로 sampling할 수 있게 합니다. Ansor에는 search space에 대한 폭넓은 커버리지를 제공하기 위해 무작위로 프로그램을 sampling하는 program sampler가 포함되어 있습니다.

**Performance tuner.** 무작위로 sampling된 프로그램의 성능이 항상 좋은 것은 아닙니다. 다음 과제는 이를 미세 조정하는 것입니다. Ansor는 evolutionary search와 학습 가능한 cost model을 사용해 반복적으로 미세 조정합니다(5장). 매 반복에서 Ansor는 새롭게 다시 sampling된 프로그램과 이전 반복에서 성능이 괜찮았던 프로그램을 초기 모집단으로 삼아 evolutionary search를 수행합니다. evolutionary search는 mutation과 crossover로 프로그램을 미세 조정하며, 비순차적 재작성을 수행해 sequential construction의 한계를 해소합니다. 학습된 cost model을 조회하는 것은 실제 측정보다 수십 배 빠르므로, 수 초 만에 수천 개의 프로그램을 평가할 수 있습니다.

**Task scheduler.** program sampler와 performance tuner를 사용하면 Ansor는 계산 그래프에 대한 고성능 텐서화 프로그램을 찾을 수 있습니다. 직관적으로는 DNN 전체를 하나의 계산 그래프로 보고 그에 대한 완전한 텐서화 프로그램을 생성하는 것이 최상의 성능을 가져올 수 있습니다. 그러나 이는 search space의 불필요한 지수적 폭발을 다뤄야 하므로 비효율적입니다. 보통 컴파일러는 DNN의 큰 계산 그래프를 여러 작은 서브그래프로 분할합니다[11, 42]. DNN의 layer 단위 구성 특성 덕분에 이 분할이 성능에 미치는 영향은 무시할 만합니다. 이로 인해 Ansor의 마지막 과제가 제기됩니다. 여러 서브그래프에 대한 프로그램을 생성할 때 시간 자원을 어떻게 분배할지의 문제입니다. Ansor의 task scheduler(6장)는 gradient descent 기반 schedule 알고리즘을 사용해 end-to-end DNN 성능을 더 많이 향상시킬 가능성이 있는 서브그래프에 자원을 할당합니다.

 ![Figure 4](https://img-blog.csdnimg.cn/20210717182342864.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)



# 4. Program Sampler

알고리즘 search의 search space는 알고리즘이 찾아낼 수 있는 최적 프로그램을 결정합니다. 기존 방법에서 고려되는 search space는 다음 요인에 의해 제한됩니다. (1) 수작업 열거(예: TVM). template로 모든 가능한 선택을 수동으로 열거하는 것은 비현실적이므로, 기존 수작업 template는 휴리스틱하게 제한된 search space만 다룰 수 있습니다. (2) greedy한 조기 가지치기(예: Halide Auto-scheduler). 미완성 프로그램의 평가를 기반으로 greedy하게 가지치기하여, search space의 일부 영역을 회피합니다.

본 절에서는 위 한계를 해결함으로써 고려하는 search space의 경계를 확장하는 기법을 소개합니다. (1)을 해결하기 위해 우리는 **유연한 추론 규칙(derivation rule) 집합을 재귀적으로 적용해 search space를 자동으로 확장**합니다. (2)를 피하기 위해 우리는 search space에서 완성된 프로그램을 무작위로 sampling합니다. 무작위 sampling은 sampling되는 모든 지점에 동일한 기회를 주므로, 우리의 search 알고리즘은 고려된 공간의 모든 프로그램을 잠재적으로 탐색할 수 있습니다. 우리는 무작위 sampling만으로 최적 프로그램을 찾는 것에 의존하지 않습니다. sampling된 프로그램은 이후에 모두 미세 조정되기 때문입니다(5장).

큰 search space를 다루는 프로그램을 sampling하기 위해, 우리는 두 단계의 계층적 search space인 sketch와 annotation을 정의합니다. 프로그램의 high-level 구조를 sketch로, 수십억 가지의 low-level 선택(예: tile size, parallel·unroll annotation)을 annotation으로 둡니다. 상위 단계에서는 몇 가지 추론 규칙을 재귀적으로 적용해 sketch를 생성합니다. 하위 단계에서는 이 sketch에 무작위로 annotation을 부여해 완성된 프로그램을 얻습니다. 이 표현은 수십억 개의 low-level 선택에서 핵심 구조를 추려내며, high-level 구조의 유연한 열거와 low-level 세부의 효율적인 sampling을 가능하게 합니다.

Ansor는 CPU와 GPU를 모두 지원하지만, 4.1과 4.2에서는 CPU의 sampling 과정을 예시로 설명하고, 4.3에서 GPU 과정에서의 차이점을 다룹니다.

## 4.1 Sketch 생성

Figure 4처럼 program sampler는 서브그래프를 입력으로 받습니다. Figure 5의 첫 번째 열은 두 가지 입력 예시를 보여줍니다. 입력은 세 가지 동등한 형태를 가집니다. 수학식 표현, 루프 인덱스를 직접 펼쳐 얻은 naive program, 그리고 그에 대응하는 계산 그래프(directed acyclic graph, DAG).

여러 노드를 가진 DAG에 대해 sketch를 생성하기 위해, 우리는 모든 노드를 위상 정렬 순서로 방문하면서 구조를 반복적으로 구성합니다. 데이터 재사용 기회가 풍부한 연산 집약적인 계산 노드(예: conv2d, matmul)에 대해서는 기본적인 tile과 fusion 구조를 sketch로 구성합니다. 단순한 element-wise 노드(예: ReLU, element-wise add)에 대해서는 안전하게 inline 처리합니다. sketch 생성 도중 새로운 노드(예: cache 노드, layout 변환 노드)가 DAG에 도입될 수도 있습니다.

우리는 몇 가지 기본 규칙을 재귀적으로 적용해 모든 가능한 sketch를 생성하는 derivation 기반 열거 방식을 제안합니다. 이 방식은 DAG를 입력으로 받아 sketch 리스트를 반환합니다. 상태를 $\sigma=(S, i)$로 정의하며, 여기서 S는 DAG에 대해 부분적으로 생성된 현재의 sketch이고 i는 현재 작업 중인 노드의 인덱스입니다. DAG 내 노드들은 출력에서 입력 방향으로 위상 순서대로 정렬됩니다. derivation은 초기 naive program과 마지막 노드에서 시작하며, 초기 상태는 $\sigma$=(naive program, 마지막 노드의 인덱스)로 쓸 수 있습니다. 그런 다음 모든 derivation 규칙을 상태에 재귀적으로 적용합니다. 각 규칙에 대해 현재 상태가 적용 조건을 만족하면 $\sigma=(S,i)$에 규칙을 적용해 $\sigma^{'}=(S^{'},i^{'})$를 얻으며, $i^{'}<=i$입니다. 이렇게 작업 노드의 인덱스 i는 단조 감소하고, i가 0이 되면 종료 상태가 됩니다. 열거 도중에는 한 상태에 여러 규칙을 적용해 여러 후속 상태를 만들 수 있습니다. 한 규칙도 여러 후속 상태를 만들 수 있으므로, 우리는 모든 중간 상태를 저장하는 큐를 유지합니다. 큐가 비면 절차가 종료됩니다. sketch 생성이 끝났을 때 종료 상태에 있는 모든 $\sigma.S$가 sketch 리스트를 구성합니다. 일반적으로 한 서브그래프에서 생성되는 sketch 수는 10개 미만입니다.

![Figure 5](https://img-blog.csdnimg.cn/20210717194903692.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

**Derivation rules.** Table 1은 CPU에 사용한 derivation 규칙들을 나열합니다. 먼저 몇 가지 술어를 정의합니다. **IsStrictInliable(S, i)**는 서브그래프 S 내 노드 i가 element-wise operator(예: ReLU, 논문에서 inlined로 표현)인지 여부를 의미합니다. **HasDataReuse(S, i)**는 S의 노드 i가 연산 집약적인 operator이며 operator 내부에서 데이터 재사용 기회가 풍부한지(예: convolution, matrix multiplication) 여부를 의미합니다. **HasFusibleConsumer(S, i)**는 S의 노드 i가 단 하나의 consumer j만 가지며 j가 i와 fusion될 수 있는지(예: matmul+bias_add, conv2d+relu) 여부를 의미합니다. **HasMoreReductionParallel(S, i)**는 S의 노드 i가 공간 차원에서는 거의 병렬화할 수 없지만 reduction 차원에서는 병렬 기회가 충분한지(예: 2D matrix multiplication 계산 $C_{2\times 2}=A_{2\times 512} * B_{512\times 2}$) 여부를 의미합니다. 우리는 이 술어 값을 얻기 위해 계산 정의에 대해 정적 분석을 수행합니다. 분석은 수학식의 read/write 패턴을 파싱해 자동으로 이루어집니다. 이어서 각 derivation 규칙의 기능을 소개합니다.

![Tabel1](https://img-blog.csdnimg.cn/20210717225525570.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

규칙 1은 strict inline 대상이 아닌 노드를 단순히 건너뜁니다. 규칙 2는 strict inline 대상 노드에 대해 항상 inline 연산을 수행합니다. 규칙 1과 규칙 2의 조건은 상호 배타적이므로, i>1인 상태에서는 둘 중 하나가 항상 적용 가능해 derivation을 계속할 수 있습니다.

규칙 3, 4, 5는 데이터 재사용을 가진 노드에 대한 multi-level tiling과 fusion을 처리합니다. 규칙 3은 데이터 재사용 노드에 대해 multi-level tiling을 수행합니다. CPU에서는 "SSRSRS" tile 구조를 사용하며, "S"는 공간 루프의 한 tile 단계를, "R"은 reduction 루프의 한 tile 단계를 의미합니다. (tile scheduler에 대해서는 이전 글을 참고하세요: [[밑바닥부터 배우는 딥러닝 컴파일러] 2, TVM의 scheduler](https://mp.weixin.qq.com/s/fPpqKL3uaaJ5QlNS79DZ5Q)). 예를 들어 Figure 5의 Example Input 1에서는 i와 j가 공간 루프이고 k가 reduction 루프입니다. matrix multiplication에서 "SSRSRS" tile 구조는 원래 3중 for 루프 $(i, j, k)$를 $(i_0,j_0,i_1,j_1,k_0,i_2,j_2,k_1,i_3,j_3)$로 확장합니다. 우리가 루프 순서를 직접 치환하지는 않지만, 이러한 multi-level tiling은 일부 reorder(이전 글 참고) 사례도 포괄할 수 있습니다. 예를 들어 위 10단계 루프에서 다른 루프 길이를 1로 설정하면 단순한 reorder 전용으로 사용할 수 있습니다. "SSRSRS" tile 구조는 딥러닝의 연산 집약적 operator(예: matmul, conv2d, conv3d)에 일반적으로 적용됩니다. 이들은 모두 공간 루프와 reduction 루프로 구성되기 때문입니다.

규칙 4는 multi-level tiling을 수행하면서 융합 가능한 consumer를 fusion합니다. 예를 들어 element-wise 노드(예: ReLU, bias add)를 tiling 노드(예: conv2d, matmul)에 fusion합니다. 만약 현재 데이터 재사용 노드에 융합 가능한 consumer가 없으면, 규칙 5는 cache 노드를 추가합니다. 예를 들어 DAG의 최종 출력 노드는 consumer가 없어 기본적으로 결과를 메인 메모리에 직접 기록하는데, 이는 메모리 접근의 높은 지연 때문에 비효율적입니다. cache 노드를 추가하면 DAG에 새로운 융합 가능한 consumer 노드가 도입되며, 이후 규칙 4를 적용해 이 새로 추가된 cache 노드를 최종 출력 노드와 fusion할 수 있습니다. cache 노드 fusion 후, 최종 출력 노드는 결과를 cache block에 기록하다가 block 내 모든 데이터 계산이 끝나면 그 결과를 즉시 메인 메모리에 기록하게 됩니다.

규칙 6은 factor[46]를 사용해 reduction 루프를 공간 루프로 분해함으로써 더 많은 병렬성을 가져올 수 있습니다.

**Examples** Figure 5는 sketch를 생성하는 세 가지 예시를 보여줍니다. sketch는 TVM의 수작업 template와 다릅니다. 수작업 template는 high-level 구조와 low-level 세부 사항을 모두 지정하는 반면, sketch는 high-level 구조만 정의합니다. Example Input 1의 경우 DAG 내 네 노드의 정렬 순서는 (A, B, C, D)입니다. DAG의 sketch를 derivation하기 위해 출력 노드 D(i=4)에서 시작해 노드 하나씩 규칙을 적용합니다. 구체적으로, 생성된 sketch 1의 derivation 규칙은 다음과 같습니다.

![여기에 그림 삽입](https://img-blog.csdnimg.cn/20210717223333757.png)

Example Input 2의 경우 다섯 노드의 정렬 순서는 (A, B, C, D, E)입니다. 마찬가지로 출력 노드 E(i=5)에서 시작해 규칙을 재귀적으로 적용합니다. 생성된 sketch 2는 다음과 같습니다.

![여기에 그림 삽입](https://img-blog.csdnimg.cn/20210717223525393.png)

마찬가지로 sketch 3은 다음 규칙들을 순차적으로 적용해 생성됩니다.

![여기에 그림 삽입](https://img-blog.csdnimg.cn/20210717223621993.png)

**Customization** 제안된 규칙들은 대부분의 operator 구조를 다룰 수 있을 만큼 충분히 실용적이지만 항상 예외가 존재합니다. 예를 들어 일부 특수한 알고리즘(예: Winograd convolution[30])이나 가속기 내부 함수(예: TensorCore[37])는 효율을 위해 특별한 tiling 구조가 필요합니다. template derivation 기반 search 방식(TVM)은 새로운 사례마다 새 template를 작성할 수 있지만, 이는 막대한 설계 노력을 요구합니다. 반면 Ansor의 derivation 기반 sketch 생성은 충분히 유연해서 새로 등장하는 알고리즘과 하드웨어에 필요한 구조를 만들어낼 수 있습니다. 사용자가 새로운 derivation 규칙을 등록해 기존 규칙들과 자연스럽게 통합할 수 있기 때문입니다.

## 4.2 Random Annotation

이전 절에서 생성된 sketch는 미완성 프로그램입니다. tiling 구조만 가지고 있을 뿐 구체적인 tile 크기와 루프 annotation(예: parallel, unroll, vectorize)을 포함하지 않기 때문입니다. 본 절에서는 sketch에 annotation을 부여해 미세 조정과 평가에 사용할 완성된 프로그램으로 만듭니다. 생성된 sketch 리스트가 주어지면, 우리는 무작위로 sketch 하나를 고르고, 무작위로 tile 크기를 채우며, **일부 외부 루프를 parallel화하고, 일부 내부 루프를 vectorize하며, 일부 내부 루프를 unroll**합니다. 또한 프로그램 내 일부 노드의 computation location을 무작위로 변경하여 tile 구조를 미세 조정합니다. 본 절에서 언급한 모든 "무작위"는 모든 유효 값에 대한 균등 분포를 의미합니다. 어떤 알고리즘이 효율을 위해 사용자 정의 annotation이 필요하다면(예: 특별한 unrolling), 사용자가 계산 정의에 간단한 힌트를 주어 annotation 전략을 조정할 수 있습니다. 마지막으로, 상수 텐서의 layout 변경은 컴파일 타임에 가능하며 런타임 오버헤드를 일으키지 않으므로, 우리는 multi-level tile 구조에 따라 상수 텐서의 layout을 가능한 한 cache-friendly하게 재작성합니다. 이 최적화는 convolution이나 fully connected layer의 weight 텐서가 정적 텐서이기 때문에 효과적입니다.

무작위 sampling 예시는 Figure 5에 나와 있습니다. 길이 1인 루프는 단순화되므로, sampling된 프로그램의 루프 수가 sketch보다 적을 수 있습니다.

## 4.3 GPU 지원

GPU의 경우 우리는 multi-level tiling 구조를 "SSRSRS"에서 "SSSRRSRS"로 변경해 GPU 아키텍처에 맞춥니다. tiling의 앞 세 공간 루프는 각각 BlockIdx, virtual thread(bank conflict 감소를 위해 사용), ThreadIdx에 바인딩됩니다. 우리는 두 가지 sketch derivation 규칙을 추가했습니다. 하나는 cache 노드를 삽입해 shared memory를 활용하기 위함이며(규칙 5와 유사), 다른 하나는 cross-thread reduction을 위함입니다(규칙 6과 유사).

# 5. 성능 미세 조정

program sampler가 sampling한 프로그램은 search space 커버리지는 좋지만 품질이 보장되지 않습니다. tiling 구조나 loop annotation 같은 최적화 선택이 모두 무작위로 sampling되기 때문입니다. 본 절에서는 evolutionary search와 학습 가능한 cost model을 통해 sampling된 프로그램의 성능을 미세 조정하는 performance tuner를 소개합니다.

미세 조정은 반복적으로 수행됩니다. 매 반복에서 우리는 먼저 학습된 cost model을 토대로 evolutionary search를 사용해 성능이 괜찮은 작은 프로그램 묶음을 찾아냅니다. 그런 다음 이 프로그램들을 하드웨어에서 측정해 실제 실행 시간 비용을 얻습니다. 마지막으로, 그 측정으로 얻은 데이터를 사용해 cost model을 재학습시켜 더 정확하게 만듭니다.

evolutionary search는 무작위 sampling된 프로그램과 직전 평가에서 우수한 프로그램을 초기 모집단으로 사용하며, mutation과 crossover를 적용해 다음 세대를 생성합니다. 학습 가능한 cost model은 각 프로그램의 성능을 예측하는 데 사용되며, 우리의 경우 그 지표는 프로그램의 throughput입니다. 우리는 일정 횟수의 evolutionary search를 수행하고 그 과정에서 가장 좋은 프로그램을 선택합니다. cost model을 활용하는 이유는 cost model이 비교적 정확한 성능 추정을 제공하면서도 실제 측정보다 수십 배 빠르기 때문입니다. 덕분에 수 초 만에 search space 내 수만 개의 프로그램을 비교하고, 괜찮은 프로그램을 선별해 실제 측정에 넘길 수 있습니다.

## 5.1 Evolutionary Search

Evolutionary search[54]는 생물학적 진화에서 영감을 받은 범용 메타휴리스틱 알고리즘입니다. 우수한 프로그램에 대한 반복적인 mutation을 통해 잠재적으로 더 우수한 새 프로그램을 만들 수 있습니다. 진화는 sampling된 초기 세대에서 시작합니다. 다음 세대를 생성하기 위해 먼저 일정 확률로 현재 세대에서 일부 프로그램을 선택합니다. 프로그램이 선택될 확률은 학습 가능한 cost model(5.2절)이 예측한 적합도에 비례하며, 이는 더 높은 성능의 프로그램이 더 높은 확률로 선택됨을 의미합니다. 선택된 프로그램에 대해 우리는 evolutionary 연산 중 하나를 무작위로 적용해 새로운 프로그램을 만듭니다. 기본적으로 sampling 단계(§4.2)에서 우리가 내린 결정 각각에 대해 그것을 재작성하고 미세 조정하기 위한 evolutionary 연산을 설계해 두었습니다.

**Tile size mutation** 이 연산은 프로그램을 스캔하면서 tiled 루프 하나를 무작위로 고릅니다. 이 tiled 루프에 대해 한 루프를 무작위로 선택해 그 길이를 임의의 수로 나누고, 그 수를 다른 한 루프에 곱합니다. 이 연산은 tile 크기들의 곱을 원래 루프 길이와 동일하게 유지하므로 mutation된 프로그램은 항상 유효합니다.

 **Parallel mutation.** 이 연산은 프로그램을 스캔하면서 parallel annotation이 부여된 루프 하나를 무작위로 선택합니다. 그 루프에 대해 인접한 루프와 fusion하거나 factor로 분해하는 방식으로 parallel 단위를 변경합니다.

**Pragma mutation.** 프로그램의 일부 최적화는 컴파일러 특정 pragma로 지정됩니다. 이 연산은 프로그램을 스캔하면서 컴파일 pragma 하나를 무작위로 선택합니다. 이 pragma에 대해 다른 유효 값으로 무작위 mutation합니다. 예를 들어 우리의 저수준 codegen은 `auto_unroll_max_step=N` pragma를 통해 최대 단계 수의 자동 unroll을 지원합니다. 우리는 숫자 N을 무작위로 조정합니다.

**Computation location mutation.** 이 연산은 프로그램을 스캔하면서 multi-level tile이 적용되지 않은 유연한 노드(예: convolution layer의 padding 노드)를 무작위로 선택합니다. 이 노드에 대해 그 computation location을 다른 유효한 노드로 무작위 변경합니다. (개인적으로 Pad를 미리 수행하는 등의 방법을 의미한다고 이해하며, Feature Map 크기는 변경되지 않습니다.)

**Node-based crossover.** Crossover는 두 명 이상의 부모로부터 유전자를 결합해 새로운 자손을 만들어내는 연산입니다. Ansor에서 프로그램의 유전자는 그 재작성 단계입니다. Ansor가 생성하는 모든 프로그램은 처음의 단순 구현으로부터 재작성된 것입니다. Ansor는 sketch 생성과 random annotation 동안 각 프로그램의 전체 재작성 이력을 보존합니다. 재작성 단계는 그 프로그램이 처음의 naive 프로그램에서 어떻게 형성되었는지를 설명하므로 프로그램의 유전자로 볼 수 있습니다. 이를 바탕으로, 우리는 두 기존 프로그램의 재작성 단계를 결합해 새 프로그램을 생성할 수 있습니다. 그러나 두 프로그램의 재작성 단계를 임의로 결합하면 단계 간 의존성을 깨뜨리고 무효한 프로그램을 만들 수 있습니다. 따라서 Ansor의 crossover 연산 단위는 DAG 내의 노드를 기반으로 합니다. 서로 다른 노드에 걸친 재작성 단계는 일반적으로 의존성이 적기 때문입니다. Ansor는 노드마다 부모 한 명을 무작위로 선택하고 선택된 노드의 재작성 단계를 병합합니다. 노드 사이에 의존성이 있을 때 Ansor는 단순한 휴리스틱으로 단계들을 분석하고 조정합니다. Ansor는 병합된 프로그램이 기능적으로 올바른지 추가로 검증합니다. Ansor는 일부 루프 변환 재작성 단계만 사용하므로 검증이 간단하며, 저수준 codegen이 의존성 분석으로 정확성을 점검할 수 있습니다.

evolutionary search는 mutation과 crossover를 활용해 새로운 후보 집합을 반복적으로 생성하고, 가장 높은 점수를 가진 작은 프로그램 집합을 출력합니다. 이 프로그램들은 타깃 하드웨어에서 컴파일·측정되어 실제 실행 시간 비용을 얻습니다. 그런 다음 수집된 측정 데이터를 사용해 cost model을 갱신합니다. 이러한 방식으로 학습 가능한 cost model의 정확도가 점차 향상되어 타깃 하드웨어에 맞춰집니다. 결과적으로 evolutionary search는 타깃 하드웨어 플랫폼에 점차 더 높은 품질의 프로그램을 생성하게 됩니다.

TVM과 FlexTensor의 search 알고리즘이 고정된 격자형 매개변수 공간에서만 동작하는 것과 달리, Ansor의 evolutionary 연산은 텐서화 프로그램에 특화되어 설계되었습니다. 일반적인 텐서화 프로그램에 적용 가능하며, 의존성이 복잡한 search space를 다룰 수 있습니다. Halide auto-scheduler의 unfold 규칙과 달리, 이 연산들은 프로그램을 비순차적으로 수정할 수 있어 sequential한 제약을 해소합니다.

## 5.2 학습 가능한 Cost Model

cost model은 search 과정에서 프로그램 성능을 빠르게 추정하는 데 필수적입니다. 우리는 관련 연구[2, 12]와 유사하게 학습 가능한 cost model을 채택하며, 새로 설계한 프로그램 특성을 사용합니다. 학습 가능한 cost model 기반 시스템은 단일 모델 설계로 다양한 하드웨어 백엔드에 다른 학습 데이터를 입력해 재사용할 수 있어 이식성이 매우 좋습니다.

우리의 대상 프로그램은 주로 데이터 병렬 텐서화 프로그램으로, 여러 개의 중첩된 루프와 가장 안쪽의 몇 개 대입문으로 구성됩니다. 따라서 cost model은 가장 안쪽 비-루프(non-loop) 문장 하나의 점수를 예측하도록 학습합니다. 완성된 프로그램의 경우 가장 안쪽의 모든 비-루프 문장에 대해 예측한 후, 그 예측 값들을 더해 점수로 사용합니다. 우리는 완성된 프로그램의 컨텍스트에서 특성을 추출해 가장 안쪽 비-루프 문장의 특성 벡터를 구성합니다. 추출된 특성에는 산술 연산 특성과 메모리 접근 특성이 포함됩니다. 추출된 특성의 상세 목록은 부록 B에 있습니다.

우리는 가중 제곱 오차(weighted squared error)를 손실 함수로 사용합니다. 우리가 가장 관심 있는 것은 search space에서 우수한 프로그램을 찾아내는 것이므로, 더 빠르게 동작하는 프로그램에 더 큰 가중치를 둡니다. 구체적으로, 모델 f의 프로그램 P(throughput y)에 대한 손실 함수는 다음과 같습니다.

![여기에 그림 삽입](https://img-blog.csdnimg.cn/20210718091722299.png)

여기서 S(P)는 P 내 가장 안쪽 비-루프 문장의 집합입니다. 우리는 throughput을 그대로 가중치로 사용하며, 저수준 모델 f로 gradient boosting decision tree[9]를 학습합니다. 모든 DAG의 모든 텐서화 프로그램에 대해 하나의 모델을 학습하기 위해, 같은 DAG에서 나온 모든 프로그램의 throughput을 [0, 1] 범위로 정규화합니다. DNN을 최적화할 때 측정되는 프로그램 수는 보통 30,000개 미만입니다. 이렇게 작은 데이터셋에서는 gradient boosting decision tree 학습이 매우 빠르므로, 점진적 갱신 대신 매번 새 모델을 학습합니다.

# 6. Schedule Task

이 부분은 OpenMMLAB의 글이 비교적 명확하게 설명하고 있어 따로 번역하지 않고 그대로 인용합니다.

> 인용 출처: OpenMMLAB https://zhuanlan.zhihu.com/p/360041136

ANSOR는 먼저 계산 그래프를 여러 서브그래프로 나누고 이 서브그래프들을 각각 최적화합니다. 딥러닝 네트워크의 전체 최적화 횟수는 ANSOR 사용자가 주며, Schedule Task 모듈이 이 최적화 횟수를 서로 다른 서브그래프 최적화 task에 어떻게 분배할지를 결정합니다.

핫스팟 서브그래프에 대한 집중적인 최적화를 보장하기 위해, ANSOR는 최적화 효과가 뚜렷한 서브그래프를 가능한 한 우선적으로 선택해 최적화합니다. 예를 들어 DNN 네트워크의 지연을 최소화한다고 할 때, 우선 다음과 같은 최적화 목적 함수를 둡니다.

$arg min f = \sum_{i=1}^nw_i\times g_i(t)$

![Schedule Task 설명](https://img-blog.csdnimg.cn/2021071812044587.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 7. 평가

여기서는 논문의 도표를 바탕으로 Ansor의 성능을 간단히 소개합니다.

![단일 operator의 성능](https://img-blog.csdnimg.cn/2021071812093311.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

Figure 6에서 볼 수 있듯이, 다양한 operator와 BatchSize 설정에서 Ansor는 모두 최고의 성능을 달성했습니다. Ansor의 큰 search space가 성능 향상의 핵심 요인입니다.

![서브그래프에서 Ansor의 성능](https://img-blog.csdnimg.cn/2021071812170117.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

여기서 ConvLayer는 Conv+BN+ReLU를 포함하는 서브그래프이고, TBS는 두 개의 matrix 전치, 하나의 Batch matrix multiplication, 하나의 Softmax를 포함하는 서브그래프입니다. @C는 CPU 측정 결과, @G는 GPU 측정 결과를 의미합니다. CPU와 GPU 모두에서 이러한 일반적인 서브그래프의 최적화에 대해 Ansor가 전반적으로 앞서 있음을 확인할 수 있습니다.

![Figure 9](https://img-blog.csdnimg.cn/20210718122037152.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

Figure 9는 인기 있는 DNN 모델을 Intel CPU, ARM CPU, NVIDIA GPU에서 측정한 결과를 보여 주며, 업계 주류 가속 라이브러리와 비교했을 때도 Ansor가 큰 폭의 성능 우위를 차지합니다.

마지막으로 모두가 비교적 관심을 두는 데이터는 Ansor의 search 시간일 것입니다. Table 3에서 볼 수 있듯이, 인기 있는 DNN 모델에서 Ansor의 search 시간은 search space가 더 크더라도 AutoTVM 대비 향상되었습니다.

![Ansor의 search 시간](https://img-blog.csdnimg.cn/20210718122402281.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 8. 관련 연구

생략. 관심 있는 분은 원 논문을 참고하시기 바랍니다.

# 9. 현재와 향후 연구

생략. 관심 있는 분은 원 논문을 참고하시기 바랍니다.

# 10. 결론

우리는 deep neural network를 위한 고성능 텐서화 프로그램을 자동으로 search·생성하는 프레임워크 Ansor를 제안했습니다. 큰 search space를 효율적으로 탐색하고 성능 병목에 우선순위를 부여함으로써, Ansor는 기존 방법의 search space 바깥에 존재하는 고성능 프로그램들을 찾아냅니다. Ansor는 다양한 신경망과 하드웨어 플랫폼에서 기존의 수작업 라이브러리와 search 기반 프레임워크를 최대 3.8배까지 능가합니다. 더 좋은 프로그램을 자동으로 search함으로써, Ansor가 늘어나는 연산 수요와 제한된 하드웨어 성능 사이의 격차를 좁히는 데 기여하기를 기대합니다. Ansor는 Apache TVM 오픈소스 프로젝트에 통합되어 있습니다.



 



