# Ansor 논문 리딩 노트 & 논문 번역

이 글에서는 Auto-Scheduler의 한 가지 방법인 Ansor를 소개합니다. 이 방법은 이미 TVM에 통합되어 AutoTVM과 함께 고성능 tensor 프로그램을 자동으로 생성하는 데 사용되고 있습니다.

  * 논문 링크: https://arxiv.org/abs/2006.06762 OSDI 2020
  * 사전 지식:
    * scheduler에 대한 이해가 필요합니다. 추천 글: https://zhuanlan.zhihu.com/p/94846767 .
    * Ansor 논문에서는 주로 parallel, cache_read, reorder, unroll, vectorize 등의 scheduler를 사용해 알고리즘을 기술하지만, Ansor의 TVM 오픈소스 구현체에서는 이들에만 한정되지 않습니다. Ansor 논문을 읽기 전에 미리 알아두시기를 권장합니다.
    * AutoTVM과 Ansor 이전에는 고성능 tensor 프로그램을 생성하기 위해 수작업으로 템플릿을 지정해야 했습니다. 이 템플릿은 high-level scheduler뿐 아니라 low-level 계산 로직까지 포함해야 했는데, 이는 CPU/GPU/ASIC 등 칩의 저수준 tensor 계산 방식이 다르기 때문입니다. 천천치(Tianqi Chen)의 TVM 발표 영상도 추천합니다: https://www.bilibili.com/video/BV1vW411R7Zb?from=search&seid=5300327607655608948 .
  * 개인적인 이해
    * Ansor는 구체적으로 어떻게 tensor 프로그램을 자동 생성할까요? 먼저 서브그래프 분할이 필요하며, 분할 규칙은 TVM의 operator fusion과 동일합니다. 각 서브그래프에 대해서는 sketch와 annotation을 통해 대응하는 프로그램을 생성합니다.
    * sketch는 위에서 소개한 scheduler들과 몇 가지 추론 규칙을 기반으로 생성됩니다. 추론 규칙이란 무엇일까요? 예를 들어, 컨볼루션이나 행렬 곱셈처럼 계산 집약적인 operator의 경우, CPU에서 Ansor는 "SSRSRS"라는 tile 규칙을 정의합니다. matmul의 경우 "SSRSRS" tile 규칙은 원래의 3중 for 루프를 확장하는 방식인데, 이는 논문 Figure 5의 첫 번째 예시에 해당합니다. 이 tile 규칙 외에도 데이터 재사용 operator와 단순 operator 간의 fusion, 데이터 재사용이 없는 단순 operator의 inline 최적화, 데이터 재사용 operator의 입력에 대한 계산 윈도우 분할 및 입력의 Cache Read, Cache Write, rfactor 등 다양한 규칙이 있습니다. GPU와 CPU의 아키텍처가 다르기 때문에 정의된 규칙도 완전히 동일하지 않습니다. 예를 들어, matmul의 다단계 tiling 구조는 GPU 아키텍처에 맞추기 위해 "SSRSRS"에서 "SSSRRSRS"로 바뀝니다. tiling의 앞쪽 세 공간 루프는 각각 BlockIdx, virtual thread(bank 충돌 감소를 위한), ThreadIdx에 바인딩됩니다. 그 외에 사용자가 직접 규칙을 정의할 수도 있습니다.
    * annotation은 sketch를 바탕으로 GPU thread bind, for 루프의 unroll, vectorize, parallel, Split factor 등을 무작위로 결정하여 완성된 코드를 생성합니다. (외부 루프 병렬화, 내부 루프 vectorize 및 unroll은 GEMM 최적화의 핵심 아이디어와 일치합니다.) 비록 완성된 코드를 생성하지만, 이 코드의 성능은 Evolutionary Search로 보장됩니다. 또한 사용자가 직접 annotation을 정의할 수도 있습니다.
    * search space를 보다 효율적으로 탐색하고 불필요한 영역을 가지치기하는 방법은 원 논문의 관련 설명을 참고하세요.
  * 장단점
    * Ansor는 매우 큰 search space를 갖기 때문에, 주류 DNN 모델과 하드웨어에서 좋은 성능의 프로그램을 찾을 수 있습니다. GPU에서는 TensorRT를 능가할 수 있고, CPU에서는 TensorflowLite, AutoTVM 등을 능가할 수 있습니다.
    * 그러나 Ansor의 search 시간은 TVM 대비 크게 단축되지는 않았으며, ResNet50을 TensorRT를 능가하는 성능까지 search하려면 몇 시간이 필요합니다.
  * 개선 가능한 부분에 대한 생각
    * search 시간을 어떻게 줄일 것인가?
    * 서브그래프 분할 단위가 비교적 작은데, 서브그래프가 너무 많아지면 최적화 알고리즘이 local optimum에 빠지지 않을까?
    * NC4HW4, Winograd 등을 추론 규칙에 추가할 수 있을까?
  * 논문 번역
    * Ansor를 더 잘 이해하기 위해 논문을 번역해 보았습니다. 오역이 있다면 지적 부탁드립니다.



# Ansor: Generating High-Performance Tensor Programs for Deep Learning

# 초록

고성능 tensor 프로그램은 딥러닝 신경망의 효율적인 실행을 보장하는 데 매우 중요합니다. 그러나 다양한 하드웨어 플랫폼에서 서로 다른 operator에 대해 효율적인 tensor 프로그램을 얻는 것은 매우 도전적인 일입니다. 현재 딥러닝 시스템은 하드웨어 벤더가 제공하는 kernel 라이브러리나 다양한 search 전략에 의존해 고성능 tensor 프로그램을 얻습니다. 이러한 방법은 플랫폼별 최적화 코드를 개발하기 위해 막대한 엔지니어링 작업이 필요하거나, search space가 제한적이고 탐색 전략이 비효율적이어서 고성능 프로그램을 찾지 못합니다.

우리는 딥러닝 응용을 위한 tensor 프로그램 생성 프레임워크인 Ansor를 제안합니다. 기존 search 전략과 비교해, Ansor는 search space의 계층적 표현에서 프로그램을 샘플링함으로써 더 많은 최적화 조합을 탐색합니다. 그런 다음 Ansor는 evolutionary search와 학습 가능한 cost model을 사용해 샘플링된 프로그램을 미세 조정하여 최적의 프로그램을 결정합니다. Ansor는 기존 SOTA 방법의 search space 밖에 있는 고성능 프로그램을 찾을 수 있습니다. 또한 Ansor는 scheduler를 활용해 DNN 내 여러 서브그래프를 동시에 최적화합니다. 실험에 따르면, Intel CPU, ARM CPU, NVIDIA GPU에서 Ansor는 신경망의 실행 성능을 각각 3.8배, 2.6배, 1.7배 향상시켰습니다.

# 1. 서론

DNN(딥러닝 신경망)의 저지연 추론은 자율주행[14], 증강현실[3], 언어 번역[15] 및 기타 AI 응용에서 핵심적인 역할을 합니다. DNN은 DAG(방향성 비순환 그래프)로 표현될 수 있으며, 노드는 operator(예: Conv, Matmul)를, 방향성 간선은 operator 간의 의존성을 나타냅니다. 기존 딥러닝 프레임워크(예: TensorFlow, Pytorch, MxNet)는 DNN의 operator를 하드웨어 벤더가 제공하는 일부 고성능 kernel 라이브러리(예: cuDNN, MKL-DNN)에 기반해 구현하여 고성능을 얻습니다. 그러나 이러한 kernel 라이브러리는 각 하드웨어 플랫폼과 각 operator에 대해 막대한 엔지니어링 작업을 통해 수동으로 튜닝해야 합니다. 각 타겟 가속기에 대해 효율적인 operator 구현을 생성하는 데 필요한 막대한 수작업은 새로운 operator[7]와 전용 가속[35]의 개발과 혁신을 제한합니다.

DNN 성능의 중요성에 비추어, 연구자와 업계 종사자들은 search 기반 컴파일 기술[2, 11, 32, 49, 59]에 눈을 돌려 tensor 프로그램, 즉 tensor operator의 low-level 구현을 자동으로 생성하고 있습니다. 하나의 operator나 여러 operator로 구성된 서브그래프에 대해, 사용자는 high-level 선언적 언어로 계산을 정의하고, 컴파일러는 다양한 하드웨어 플랫폼에 맞는 맞춤 프로그램을 search합니다.

고성능 tensor 프로그램을 찾기 위해서는, search 기반 방법이 모든 tensor 프로그램 최적화 전략을 포괄할 만큼 충분히 큰 search space를 탐색해야 합니다. 그러나 기존 방법들은 효과적인 최적화 조합을 포착하지 못하는데, 이는 사전 정의된 수작업 템플릿(예: TVM[12], FlexTensor[59])에 의존하거나 불완전한 프로그램을 평가하여(예: Halide auto-scheduler) 튜닝하기 때문이며, 이로 인해 포괄적인 search space를 다루지 못합니다. search space를 구축하는 데 사용하는 규칙도 제한적입니다.

본 논문에서는 고성능 tensor 프로그램 생성을 위한 새로운 search 전략을 탐구합니다. 이 전략은 다양한 최적화 전략을 포괄적으로 포함하는 큰 search space를 자동으로 생성합니다. 그 결과 기존 방법이 놓친 고성능 프로그램을 찾을 수 있습니다.

이 목표를 달성하려면 여러 도전 과제를 해결해야 합니다. 먼저, 주어진 계산 정의에 대해 가능한 한 많은 tensor 프로그램을 포함하는 큰 search space를 자동으로 구축해야 합니다. 다음으로, 큰 search space에서 불완전한 프로그램을 비교하지 않고도 더 효율적으로 search해야 하며, 이 search space는 기존 템플릿이 포괄하는 범위보다 수십 배 클 수 있습니다. 마지막으로, 많은 서브그래프가 있는 전체 DNN을 최적화할 때, end-to-end 성능에 결정적인 서브그래프를 식별하고 우선순위를 부여해야 합니다.

이를 위해 우리는 tensor 프로그램을 자동으로 생성하는 프레임워크인 Ansor를 설계하고 구현했습니다. Ansor는 계층적 표현을 활용해 큰 search space를 다룹니다. 이 표현은 high-level 구조와 low-level 세부 사항을 분리하여, high-level 구조의 유연한 열거와 low-level 세부 사항의 효율적인 샘플링을 가능하게 합니다. space는 계산 정의가 주어지면 자동으로 구축되며, Ansor는 search space에서 완성된 프로그램을 샘플링하고 evolutionary search와 학습 가능한 cost model을 사용해 샘플링된 프로그램을 미세 조정합니다. 여러 서브그래프가 있는 DNN의 성능을 최적화하기 위해, Ansor는 end-to-end 성능을 향상시킬 가능성이 더 높은 서브그래프를 동적으로 우선시합니다.

우리는 표준 딥러닝 Benchmark와 search 기반 튜닝 프레임워크에서 제공되는 Benchmark에 대해 평가를 수행했습니다. 실험 결과 Ansor는 Intel CPU, ARM CPU, NVIDIA GPU에서 DNN의 실행 성능을 각각 3.8배, 2.6배, 1.7배 향상시켰습니다. 대부분의 계산 정의에 대해 Ansor가 찾은 최적의 프로그램은 기존 search 기반 방법의 search space 밖에 있었습니다. 결과는 또한 Ansor가 더 큰 search space를 가지고 있음에도 불구하고, 기존 search 기반 방법보다 더 효율적으로 search하여 더 짧은 시간 안에 더 높은 성능의 프로그램을 생성한다는 것을 보여줍니다. Ansor는 기존 SOTA 프레임워크(TVM처럼 자동 튜닝이 가능한 프레임워크)와 동등한 성능을 한 자릿수 적은 시간에 달성할 수 있습니다. 또한 Ansor는 새로운 operator로 자동 확장될 수 있으며, 해당 operator의 수학적 정의만 있으면 되고 수동으로 템플릿을 작성할 필요가 없습니다.

정리하자면, 이 논문은 다음과 같은 기여를 합니다:

  * 계산 그래프에 대한 tensor 프로그램을 생성하기 위한 대규모 space 계층적 search 메커니즘.
  * 학습 가능한 cost model 기반의 진화 전략으로 tensor 프로그램의 성능을 미세 조정.
  * gradient descent 기반 schedule 알고리즘으로, DNN의 end-to-end 성능을 최적화할 때 중요한 서브그래프에 우선순위를 부여.
  * Ansor 시스템의 구현 및 종합 평가를 통해, 위의 기법들이 다양한 DNN과 하드웨어 플랫폼에서 최첨단 시스템보다 우수함을 입증.



# 2. 배경

딥러닝 생태계는 CPU, GPU, FPGA, ASIC 등 빠르게 증가하는 하드웨어 플랫폼의 다양성을 받아들이고 있습니다. 이러한 플랫폼에서 DNN을 배포하기 위해서는, DNN이 사용하는 operator에 고성능 tensor 프로그램이 필요합니다. 필요한 operator에는 보통 표준 operator(예: matmul, conv2d)와 머신러닝 연구자들이 발명한 새로운 operator(예: capsule conv2d[23], dilated conv2d[57])가 혼합되어 있습니다.

다양한 하드웨어 플랫폼에 이러한 operator를 배포할 때 좋은 성능을 유지하기 위해, 여러 컴파일러 기술(예: TVM[11], Halide[41], Tensor Comprehensions[49])이 도입되었습니다. 사용자는 high-level 선언적 언어로 수학식과 유사한 형태의 계산을 정의하고, 컴파일러는 정의에 따라 최적화된 tensor 프로그램을 생성합니다. Figure 1은 TVM tensor 표현 언어에서의 matmul 정의를 보여줍니다. 사용자는 주로 tensor의 shape와 출력 tensor의 각 원소를 어떻게 계산할지를 정의하면 됩니다.

![이미지](images/img_01.png)Figure1

그러나 high-level 정의로부터 고성능 tensor 프로그램을 자동으로 생성하는 것은 매우 어렵습니다. 타겟 플랫폼의 아키텍처에 따라, 컴파일러는 다양한 최적화 방식(예: tile, vectorize, parallel 등)을 포함하는 매우 복잡하고 거대한 space에서 search해야 합니다. 고성능 프로그램을 찾으려면 search 전략이 포괄적인 space를 다루고 효율적으로 search해야 합니다. 본 절에서는 가장 최신이며 효과적인 두 가지 방법을 설명하고, 8장에서 다른 관련 연구를 다룹니다.

**Template-guided search** 템플릿 기반 search에서 search space는 수작업 템플릿으로 지정됩니다. Figure 2 a에서 보이듯, 컴파일러(예: TVM)는 사용자가 계산 정의에 사용할 템플릿을 직접 작성하도록 요구합니다. 이 템플릿은 일부 조정 가능한 매개변수(예: tile size, unrolling factor)로 tensor 프로그램의 구조를 정의합니다. 그런 다음 컴파일러는 특정 입력 shape 설정과 특정 하드웨어 타겟에 대해 이러한 매개변수의 최적값을 search합니다. 이 방식은 일반적인 딥러닝 operator에서 좋은 성능을 보였습니다. 그러나 템플릿 개발에는 막대한 엔지니어링 노력이 필요합니다. 예를 들어, TVM 코드베이스에서 수작업 템플릿이 차지하는 코드는 이미 15000줄을 넘습니다. 새로운 operator와 새로운 하드웨어 플랫폼이 등장하면서 이 숫자는 계속 늘어나고 있습니다. 또한 고품질 템플릿 개발에는 tensor operator와 하드웨어에 대한 전문 지식이 필요합니다. 고품질 템플릿 개발에도 많은 연구 노력이 필요합니다 [32, 55, 59]. 템플릿 설계가 복잡함에도 불구하고, 수동으로 지정한 템플릿은 모든 operator의 모든 최적화 선택을 일일이 열거하는 것이 어렵기 때문에 제한된 프로그램 구조만 다룰 수 있습니다. 이 방식은 보통 각 operator마다 템플릿 하나를 정의해야 합니다. Flex-Tensor [59]는 여러 operator를 다루는 범용 템플릿을 제안했지만, 그 템플릿은 여전히 단일 operator용이며 여러 operator가 관여하는 최적화(예: operator fusion)는 포함하지 않습니다. 여러 operator가 있는 계산 그래프를 최적화하는 search space는 이러한 operator들의 다양한 조합 방식을 포함해야 합니다. 템플릿 기반 방법은 search 과정에서 고정된 템플릿을 분해하고 재조합할 수 없기 때문에 이를 달성할 수 없습니다.

![이미지](images/img_02.png)Figure 2

**Sequential construction based search.** 이 방법은 프로그램 구성을 고정된 결정 시퀀스로 분해하여 search space를 정의합니다. 그런 다음 컴파일러는 beam search [34] 같은 알고리즘을 사용해 좋은 결정을 search합니다(예: Halide auto-scheduler [2]). 이 방식에서 컴파일러는 계산 그래프의 모든 노드를 순차적으로 unfold해 tensor 프로그램을 구성합니다. 각 노드에 대해, 컴파일러는 이를 어떻게 low-level tensor 프로그램으로 변환할지에 대한 일련의 결정을 내립니다(즉, computation location, storage location, tile size 등). 모든 노드가 unfold되면 완성된 tensor 프로그램이 구축됩니다. 이 방식은 각 노드에 대해 일반적인 unfold 규칙 집합을 사용하므로, 수작업 템플릿 없이 자동으로 search할 수 있습니다. 각 결정의 가능한 선택지가 매우 많기 때문에 sequential 과정이 가능하도록 하기 위해, 이 방법은 각 결정 후 상위 k개의 후보 프로그램만 유지합니다. 컴파일러는 학습 가능한 cost model을 기반으로 후보 프로그램의 성능을 평가·비교해 상위 k개를 선택하고 나머지는 버립니다. search 과정에서 후보 프로그램은 불완전한데, 이는 계산 그래프의 일부만 펼쳐졌거나 일부 결정만 내려졌기 때문입니다. Figure 2 b가 이 과정을 보여줍니다.

그러나 불완전한 프로그램의 최종 성능을 평가하는 데에는 몇 가지 어려움이 있습니다: (1) 완성된 프로그램으로 학습된 cost model은 불완전한 프로그램의 최종 성능을 정확히 예측할 수 없습니다. cost model은 완성된 프로그램으로만 학습할 수 있는데, 학습 라벨을 얻기 위해서는 프로그램을 컴파일하고 실행 시간을 측정해야 하기 때문입니다. 이 모델을 직접 사용해 불완전한 프로그램의 최종 성능을 평가하면 정확도가 떨어집니다. 사례 연구로(5.2절), 우리는 search space에서 무작위로 추출한 20,000개의 완성된 프로그램에 대해 cost model을 학습시킨 뒤, 이 모델로 불완전한 프로그램의 최종 성능을 예측했습니다. 불완전한 프로그램은 완성된 프로그램의 일부 루프 변환만 적용해 얻습니다. 우리는 두 가지 평가 지표로 평가합니다: 짝지어진 비교 정확도와 top-k 프로그램의 recall 점수(k=10). Figure 3에서 보이듯, 두 곡선은 각각 50%와 0%에서 시작하는데, 이는 무정보의 무작위 추측이 50%의 짝지어진 비교 정확도와 0%의 top-k recall을 제공한다는 의미입니다. 두 곡선은 프로그램이 완성에 가까워질수록 빠르게 상승하며, 이는 cost model이 완성된 프로그램의 성능에 대해서는 매우 좋지만, 불완전한 프로그램의 최종 성능은 정확히 예측하지 못한다는 것을 의미합니다. (2) 순차적 결정의 고정된 순서는 search space 설계를 제한합니다. 예를 들어, 일부 최적화는 계산 그래프에 새로운 노드를 추가해야 합니다(예: 캐시 노드 추가, rfactor[46] 사용). 서로 다른 프로그램의 결정 수가 다를 수 있어, 불완전한 프로그램들을 공정하게 비교하기 위해 정렬하기 어렵습니다. (3) sequential 구성 기반 search는 확장성이 없습니다. search space를 확장하려면 더 많은 sequential 구성 단계를 추가해야 하지만, 이는 누적 오류를 더 심각하게 만듭니다.

![이미지](images/img_03.png)Figure3

**Ansor's hierarchical approach** Figure 2-c에서 보이듯, Ansor는 high-level 구조와 low-level 세부 사항을 분리하는 계층적 search space에 기반해 구축됩니다. Ansor는 계산 그래프의 search space를 자동으로 구축하므로 수작업으로 템플릿을 개발할 필요가 없습니다. 그런 다음 Ansor는 space에서 완성된 프로그램을 샘플링하고 완성된 프로그램에 대해 미세 조정을 수행하여, 불완전한 프로그램에 대한 부정확한 추정을 피합니다. Figure 2는 Ansor 방법과 기존 방법 간의 주요 차이점을 보여줍니다.

# 3. 설계 개요

Ansor는 자동 tensor 프로그램 생성 프레임워크입니다. Figure 4는 Ansor의 전체 아키텍처를 보여줍니다. Ansor의 입력은 최적화할 DNN 집합입니다. Ansor는 Relay[42]의 operator fusion 알고리즘을 사용해 인기 있는 모델 형식(예: ONNX, TensorFlow PB)에서 DNN을 작은 서브그래프로 변환한 뒤, 이 서브그래프들에 대한 tensor 프로그램을 생성합니다. Ansor는 세 가지 중요한 구성 요소를 갖습니다: (1) 큰 search space를 구축하고 그로부터 다양한 프로그램을 샘플링하는 program sampler. (2) 샘플링된 프로그램의 성능을 미세 조정하는 performance tuner. (3) DNN 내 여러 서브그래프를 최적화하기 위해 시간 자원을 할당하는 task scheduler.

**Program sampler.** Ansor가 해결해야 할 핵심 과제 중 하나는 주어진 계산 그래프에 대한 큰 search space를 생성하는 것입니다. 다양한 high-level 구조와 low-level 세부 사항을 가진 다양한 tensor 프로그램을 다루기 위해, Ansor는 두 단계의 search space 계층적 표현인 sketch와 annotation을 활용합니다(4장). Ansor는 프로그램의 high-level 구조를 sketch로 정의하고, 수십억 개의 low-level 선택(예: tile size, parallel, unroll annotation)을 annotation으로 다룹니다. 이 표현은 Ansor가 high-level 구조를 유연하게 열거하고 low-level 세부 사항을 효율적으로 샘플링할 수 있게 해줍니다. Ansor는 space에서 프로그램을 무작위로 샘플링하는 program sampler를 포함하여 search space를 포괄적으로 다룹니다.

**Performance tuner.** 무작위로 샘플링된 프로그램의 성능이 항상 좋은 것은 아닙니다. 다음 과제는 이를 미세 조정하는 것입니다. Ansor는 evolutionary search와 학습 가능한 cost model을 사용해 반복적으로 미세 조정합니다(5장). 각 반복에서 Ansor는 새로 재샘플링한 프로그램과 이전 반복에서 성능이 괜찮았던 프로그램을 초기 집단으로 사용해 evolutionary search를 수행합니다. evolutionary search는 변이와 교차를 통해 프로그램을 미세 조정하고, 비순서 재작성(out-of-order rewrite)을 수행하여 sequential 구성의 한계를 해결합니다. 학습된 cost model을 query하는 것은 실제 측정보다 수십 배 빠르므로, 우리는 몇 초 안에 수천 개의 프로그램을 평가할 수 있습니다.

**Task scheduler.** program sampler와 performance tuner를 사용하면 Ansor는 계산 그래프에 대한 고성능 tensor 프로그램을 찾을 수 있습니다. 직관적으로, 전체 DNN을 단일 계산 그래프로 보고 그에 대한 완성된 tensor 프로그램을 생성하면 최고의 성능을 달성할 수 있습니다. 그러나 이는 search space의 불필요한 지수적 폭발을 처리해야 하므로 비효율적입니다. 일반적으로 컴파일러는 DNN의 큰 계산 그래프를 몇 개의 작은 서브그래프로 분할합니다 [11, 42]. DNN의 layer 단위 구성 특성으로 인해, 이 분할이 성능에 미치는 영향은 무시할 수 있을 정도입니다. 이는 Ansor의 마지막 과제로 이어집니다: 여러 서브그래프에 대해 프로그램을 생성할 때 어떻게 시간 자원을 분배할 것인가. Ansor의 task scheduler(6장)는 gradient descent 기반 schedule 알고리즘을 사용해, end-to-end DNN 성능을 향상시킬 가능성이 더 높은 서브그래프에 자원을 할당합니다.

![이미지](images/img_04.png)Figure 4

# 4. 프로그램 샘플러

알고리즘 search의 search space는 찾을 수 있는 최적 프로그램을 결정합니다. 기존 방법에서 고려되는 search space는 다음 요인들에 의해 제한됩니다: (1) 수작업 열거(예: TVM). 템플릿을 통해 가능한 모든 선택을 수동으로 열거하는 것은 비현실적이므로, 기존의 수작업 템플릿은 휴리스틱하게 제한된 search space만 다룰 수 있습니다. (2) 탐욕적 조기 가지치기(예: Halide Auto-scheduler). 불완전한 프로그램 평가를 기반으로 탐욕적으로 가지치기하여, 알고리즘 search space의 일부 영역을 회피합니다.

본 절에서는 위의 한계를 해결함으로써 고려되는 search space의 경계를 확장하는 기법을 소개합니다. (1)을 해결하기 위해, 우리는 **유연한 추론 규칙 집합을 재귀적으로 적용해 search space를 자동 확장** 합니다. (2)를 피하기 위해, 우리는 search space에서 완성된 프로그램을 무작위 샘플링합니다. 무작위 샘플링은 샘플링될 각 점에 동등한 기회를 부여하므로, 우리의 search 알고리즘은 고려된 space의 모든 프로그램을 잠재적으로 탐색할 수 있습니다. 우리는 무작위 샘플링에 의존해 최적 프로그램을 찾지는 않습니다. 각 샘플링된 프로그램은 이후에 미세 조정되기 때문입니다(5장).

큰 search space를 다루는 프로그램을 샘플링하기 위해, 우리는 두 단계의 계층적 search space인 sketch와 annotation을 정의합니다. 우리는 프로그램의 high-level 구조를 sketch로 정의하고, 수십억 개의 low-level 선택(예: tile size, parallel, unroll annotation)을 annotation으로 다룹니다. 상위 단계에서는 몇 가지 추론 규칙을 재귀적으로 적용해 sketch를 생성합니다. 하위 단계에서는 이 sketch에 무작위 annotation을 부여해 완성된 프로그램을 얻습니다. 이러한 표현은 수십억 개의 low-level 선택에서 몇 가지 기본 구조를 요약하여, high-level 구조의 유연한 열거와 low-level 세부 사항의 효율적인 샘플링을 가능하게 합니다.

Ansor는 CPU와 GPU를 모두 지원하지만, 4.1과 4.2에서는 예시로 CPU의 샘플링 과정을 설명합니다. 그런 다음 4.3에서 GPU의 과정이 어떻게 다른지 논의합니다.

## 4.1 sketch 생성

Figure 4에서 보이듯, program sampler는 서브그래프를 입력으로 받습니다. Figure 5의 첫 번째 열은 두 가지 입력 예시를 보여줍니다. 입력은 세 가지 동등한 형태를 갖습니다: 수학식 표현, 루프 인덱스를 직접 펼쳐 얻은 대응 naive 프로그램, 그리고 대응 계산 그래프(DAG).

여러 노드가 있는 DAG에 대한 sketch를 생성하기 위해, 우리는 위상 순서로 모든 노드를 방문하며 반복적으로 구조를 구축합니다. 데이터 재사용 기회가 풍부한 계산 집약적 노드(예: conv2d, matmul)에 대해서는, 그 노드들에 대해 기본 tile 및 fusion 구조를 sketch로 구축합니다. 단순한 element-wise 노드(예: ReLU, element-wise add)에 대해서는 안전하게 inline할 수 있습니다. sketch 생성 과정에서 새로운 노드(예: 캐시 노드, 레이아웃 변환 노드)가 DAG에 도입될 수도 있다는 점에 주목합시다.

우리는 몇 가지 기본 규칙을 재귀적으로 적용해 가능한 모든 sketch를 생성하는 추론 기반 열거 방법을 제안합니다. 이 방법은 DAG를 입력으로 받아 sketch 목록을 반환합니다. 우리는 상태를 정의하는데, 여기서 S는 DAG의 일부에 대해 현재 생성된 sketch이고, i는 현재 작업 중인 노드의 인덱스입니다. DAG의 노드는 출력에서 입력으로 위상 순서로 정렬됩니다. 추론은 초기 naive 프로그램과 마지막 노드에서 시작하며, 다시 말해 초기 상태는 (naive 프로그램, 마지막 노드의 인덱스)로 작성될 수 있습니다. 그런 다음 모든 추론 규칙을 재귀적으로 상태에 적용해 봅니다. 각 규칙에 대해, 현재 상태가 적용 조건을 만족하면 규칙을 적용해 다음을 얻습니다. 이렇게 작업 노드의 인덱스 i가 단조 감소하며, i가 0이 되면 종료 상태가 됩니다. 열거 과정에서 한 상태에 여러 규칙을 적용해 여러 후속 상태를 생성할 수 있습니다. 한 규칙도 여러 가능한 후속 상태를 생성할 수 있으므로, 우리는 모든 중간 상태를 저장하는 큐를 유지합니다. 큐가 비면 과정이 종료됩니다. sketch 생성이 끝나면 종료 상태에 있는 모든 것이 생성된 sketch 목록입니다. 일반적으로 한 서브그래프가 생성하는 sketch 수는 10개 미만입니다.

![이미지](images/img_05.png)Figure 5

**Derivation rules.** Table 1은 우리가 CPU에 사용한 추론 규칙들을 나열합니다. 먼저 몇 가지 술어를 선언합니다. 예를 들어 **IsStrictInliable(S, i)** 는 서브그래프 S 내 노드 i가 element-wise OP인지(예: ReLU와 같이 inline 최적화될 수 있는 OP, 논문에서는 inlined라고 표현)를 나타냅니다. **HasDataReuse(S, i)** 는 S 내 노드 i가 계산 집약적 operator이며 operator 내 데이터 재사용 기회가 많은지(예: 컨볼루션, 행렬 곱)를 나타냅니다. **HasFusibleConsumer(S, i)** 는 S 내 노드 i에 단 하나의 소비자 j가 있고 j가 노드 i와 fusion될 수 있는지(예: matmul+bias_add, conv2d+relu)를 나타냅니다. **HasMoreReductionParallel(S, i)** 는 S 내 노드 i가 공간 차원에서는 거의 병렬화할 수 없지만 reduction 차원에서는 충분한 병렬화 기회가 있는지(예: 2D matmul 계산)를 나타냅니다. 우리는 계산 정의에 대한 정적 분석을 수행해 이 술어들의 값을 얻습니다. 분석은 수학식의 read/write 패턴을 파싱하여 자동으로 완료됩니다. 다음으로 각 추론 규칙의 기능을 소개합니다.

![이미지](images/img_06.png)Tabel1

규칙 1은 노드가 엄격하게 inline될 수 없는 경우 단순히 건너뛰는 것입니다. 규칙 2는 엄격하게 inline될 수 있는 노드에 대해 항상 inline 작업을 수행합니다. 규칙 1과 규칙 2의 조건은 상호 배타적이므로, i>1인 상태는 항상 둘 중 하나를 만족하며 추론을 계속할 수 있습니다.

규칙 3, 4, 5는 데이터 재사용이 있는 노드의 다단계 tiling과 fusion을 처리합니다. 규칙 3은 데이터 재사용 가능 노드에 대해 다단계 tiling을 수행합니다. CPU의 경우, "SSRSRS" tile 구조를 사용하는데, 여기서 "S"는 공간 루프의 한 tile 단계를, "R"은 reduction 루프의 한 tile 단계를 의미합니다. (tile이라는 scheduler에 대해서는 제 이전 글을 참고하세요: [【从零开始学深度学习编译器】二，TVM中的scheduler](<https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247493649&idx=1&sn=fb9ddb7ee5a5fd54653fcb926ade4ffc&scene=21#wechat_redirect>)) 예를 들어 Figure 5의 Example Input1의 경우, i와 j는 공간 루프이고 k는 reduction 루프입니다. matmul의 경우 "SSRSRS" tile 구조는 원래의 3중 for 루프를 확장합니다. 우리는 루프 순서를 바꾸지 않지만, 이러한 다단계 tiling은 일부 reorder(앞 글에서도 다룬) 경우를 다룰 수 있습니다. 예를 들어 위의 10단계 루프는 다른 루프의 길이를 1로 설정함으로써 단순한 reorder 전용으로 사용될 수 있습니다. "SSRSRS" tile 구조는 딥러닝의 계산 집약적 OP(예: matmul, conv2d, conv3d)에 대해 일반적으로 적용 가능한데, 이들은 모두 공간 루프와 reduction 루프로 구성되기 때문입니다.

규칙 4는 다단계 tiling을 수행하고 fusion 가능한 소비자를 fusion합니다. 예를 들어, element-wise 노드(예: ReLU, bias add)를 tiling 노드(예: conv2d, matmul)에 fusion합니다. 현재의 데이터 재사용 가능 노드에 fusion 가능한 소비자가 없는 경우, 규칙 5는 캐시 노드를 추가합니다. 예를 들어, DAG의 최종 출력 노드는 소비자가 없으므로, 기본적으로 결과를 메인 메모리에 직접 쓰며, 메모리 접근의 높은 지연으로 인해 효율이 떨어집니다. 캐시 노드를 추가함으로써 우리는 DAG에 새로운 fusion 가능 소비자 노드를 도입하고, 그런 다음 규칙 4를 적용해 이 새로 추가된 캐시 노드를 최종 출력 노드에 fusion할 수 있습니다. 캐시 노드 fusion 후, 이제 최종 출력 노드는 결과를 캐시 블록에 쓰고, 블록 내 모든 데이터가 계산되면 캐시 블록의 결과가 즉시 메인 메모리에 쓰여집니다.

규칙 6은 factor[46]를 사용해 reduction 루프를 공간 루프로 분해해 더 많은 병렬성을 가져올 수 있습니다.

**Examples** Figure 5는 sketch를 생성하는 세 가지 예시를 보여줍니다. sketch는 TVM의 수작업 템플릿과 다릅니다. 수작업 템플릿은 high-level 구조와 low-level 세부 사항을 모두 지정하지만, sketch는 high-level 구조만 정의하기 때문입니다. Example Input 1의 경우, DAG 내 네 노드의 정렬 순서는 (A,B,C,D)입니다. DAG의 sketch를 추론하기 위해, 우리는 출력 노드 D(i=4)에서 시작해 노드에 규칙을 하나씩 적용합니다. 구체적으로, 생성된 sketch 1의 추론 규칙은 다음과 같습니다:

![이미지](images/img_07.png)여기에 이미지 삽입

Example Input 2의 경우, 다섯 노드의 정렬 순서는 (A,B,C,D,E)입니다. 마찬가지로 출력 노드 E(i=5)에서 시작해 재귀적으로 규칙을 적용합니다. 생성된 sketch 2는 다음과 같습니다:

![이미지](images/img_08.png)여기에 이미지 삽입

마찬가지로, sketch 3은 다음 규칙들의 순차 적용으로 생성됩니다:

![이미지](images/img_09.png)여기에 이미지 삽입

**Customization** 제안된 규칙들이 충분히 실용적이어서 대부분의 operator 구조를 다룰 수 있지만, 항상 예외는 있습니다. 예를 들어 일부 특수한 알고리즘(예: Winograd 컨볼루션[30])과 가속기 내부 함수(예: TensorCore[37])는 효과를 보려면 특수한 tiling 구조가 필요합니다. 템플릿 추론 search 방법(TVM)은 새로운 사례마다 새로운 템플릿을 만들 수 있지만, 이는 많은 설계 작업을 요구합니다. 반면 Ansor의 추론 기반 sketch 생성은 충분히 유연해 새로운 알고리즘과 하드웨어에 필요한 구조를 생성할 수 있습니다. 사용자가 새로운 추론 규칙을 등록하고 기존 규칙과 매끄럽게 통합할 수 있게 하기 때문입니다.

## 4.2 random annotation

이전 절에서 생성된 sketch는 불완전한 프로그램입니다. tiling 구조만 있을 뿐, 특정 tiling 크기와 루프 annotation(예: parallel, unroll, vectorize)이 없기 때문입니다. 본 절에서는 sketch에 annotation을 부여해, 미세 조정과 평가에 사용할 완성된 프로그램으로 만듭니다. 생성된 sketch 목록이 주어지면, 우리는 sketch 하나를 무작위로 선택하고, tile 크기를 무작위로 채우고, **일부 외부 루프를 parallel로, 일부 내부 루프를 vectorize, 일부 내부 루프를 unroll** 합니다. 또한 프로그램 내 일부 노드의 계산 위치를 무작위로 변경해 tile 구조를 미세 조정합니다. 본 소절에서 언급되는 모든 "무작위"는 모든 유효 값에 대한 균등 분포를 의미합니다. 일부 알고리즘이 효과를 내기 위해 사용자 정의 annotation이 필요한 경우(예: 특수한 unrolling), 사용자가 계산 정의에 간단한 힌트를 제공해 annotation 전략을 조정할 수 있게 합니다. 마지막으로, 상수 tensor의 레이아웃 변경은 컴파일 시점에 완료할 수 있고 runtime overhead를 발생시키지 않으므로, 다단계 tile 구조에 따라 상수 tensor의 레이아웃을 다시 작성해 가능한 한 캐시 친화적으로 만듭니다. 컨볼루션이나 dense layer의 가중치 tensor는 정적 tensor이기 때문에 이 최적화가 효과적입니다.

random sampling 예시는 Figure 5에 나와 있는데, 길이 1의 루프가 단순화되기 때문에 샘플링된 프로그램의 루프 수가 sketch보다 적을 수 있습니다.

## 4.3 GPU 지원

GPU의 경우, 우리는 다단계 tiling 구조를 "SSRSRS"에서 "SSSRRSRS"로 변경해 GPU 아키텍처에 맞춥니다. tiling의 앞 세 공간 루프는 각각 BlockIdx, virtual thread(bank 충돌 감소를 위한), ThreadIdx에 바인딩됩니다. 우리는 두 가지 sketch 추론 규칙을 추가했습니다. 하나는 캐시 노드를 삽입해 공유 메모리를 활용하기 위한 것(규칙 5와 유사)이고, 다른 하나는 thread 간 reduction을 위한 것(규칙 6과 유사)입니다.

# 5. 성능 미세 조정

program sampler가 샘플링한 프로그램은 좋은 search space 커버리지를 갖지만 품질은 보장되지 않습니다. 이는 tiling 구조와 loop annotation 같은 최적화 선택이 모두 무작위로 샘플링되기 때문입니다. 본 절에서는 evolutionary search와 학습 가능한 cost model을 통해 샘플링된 프로그램의 성능을 미세 조정하는 performance tuner를 소개합니다.

미세 조정은 반복적으로 수행됩니다. 각 반복에서 우리는 먼저 학습된 cost model을 기반으로 evolutionary search를 사용해 성능이 괜찮은 프로그램의 작은 배치를 찾습니다. 그런 다음 이 프로그램들을 하드웨어에서 측정해 실제 실행 시간 비용을 얻습니다. 마지막으로, 그로부터 얻은 분석 데이터를 사용해 cost model을 다시 학습시켜 더 정확하게 만듭니다.

evolutionary search는 무작위로 샘플링된 프로그램과 지난 평가에서의 고품질 프로그램을 초기 집단으로 사용하고, 변이와 교차를 적용해 다음 세대를 생성합니다. 학습 가능한 cost model은 각 프로그램의 성능을 예측하는 데 사용되며, 우리의 경우는 프로그램의 throughput입니다. 우리는 고정된 횟수의 evolutionary search를 수행하고, search 과정에서 최고의 프로그램을 선택합니다. 우리는 학습 가능한 cost model을 활용하는데, cost model은 비교적 정확한 프로그램 성능 추정을 제공하면서 실제 측정보다 수십 배 빠르기 때문입니다. 이 덕분에 우리는 search space의 수만 개 프로그램을 몇 초 안에 비교하고 괜찮은 것들을 골라 실제 평가할 수 있습니다.

## 5.1 evolutionary search

evolutionary search [54]는 생물의 진화에서 영감을 받은 범용 메타휴리스틱 알고리즘입니다. 고품질 프로그램에 대해 반복적으로 변이를 적용함으로써, 잠재적으로 더 높은 품질의 새 프로그램을 생성할 수 있습니다. 진화는 샘플링된 초기 세대에서 시작합니다. 다음 세대를 만들기 위해, 우리는 먼저 일정 확률로 현재 세대에서 일부 프로그램을 선택합니다. 프로그램이 선택될 확률은 학습 가능한 cost model(5.2절)이 예측한 적합도에 비례하는데, 이는 더 높은 성능의 프로그램이 선택될 확률이 더 높다는 의미입니다. 선택된 프로그램에 대해, 우리는 진화 연산 중 하나를 무작위로 적용해 새 프로그램을 생성합니다. 기본적으로, 샘플링 과정에서 우리가 내린 결정들(§4.2)에 대해 우리는 이를 재작성하고 미세 조정하기 위한 대응 진화 연산을 설계했습니다.

**Tile size mutation** 이 연산은 프로그램을 스캔하고 tiled 루프 하나를 무작위로 선택합니다. 이 tiled 루프에 대해, 한 루프를 무작위로 선택해 그 길이를 무작위 수로 나누고, 그 수를 다른 루프에 곱합니다. 이 연산은 tile 크기들의 곱을 원래의 루프 길이와 동일하게 유지하므로, 변이된 프로그램은 항상 유효합니다.

**Parallel mutation.** 이 연산은 프로그램을 스캔하고 parallel annotation이 부여된 루프 하나를 무작위로 선택합니다. 이 루프에 대해, 인접한 루프와 fusion하거나 factor로 분해해 parallel 단위를 변경합니다.

**Pragma mutation.** 프로그램의 일부 최적화는 컴파일러 특정 컴파일 pragma로 지정됩니다. 이 연산은 프로그램을 스캔하고 컴파일 pragma 하나를 무작위로 선택합니다. 이 pragma에 대해 다른 유효한 값으로 무작위로 변이시킵니다. 예를 들어, 우리의 저수준 codegen은 `auto_unroll_max_step=N` pragma를 통해 최대 단계까지의 자동 unroll을 지원합니다. 우리는 N 값을 무작위로 조정합니다.

**Computation location mutation.** 이 연산은 프로그램을 스캔하고 다단계 tiled가 아닌 유연한 노드(예: 컨볼루션 layer의 padding 노드) 하나를 무작위로 선택합니다. 이 노드에 대해, 그 계산 위치를 다른 유효한 노드로 무작위로 변경합니다. (개인적으로 이해하기로는 Pad는 미리 수행될 수 있고, Feature Map 크기는 변하지 않는다는 것입니다.)

**Node-based crossover.** 교차는 둘 이상의 부모로부터 유전자를 결합해 새로운 자손을 생성하는 연산입니다. Ansor에서 프로그램의 유전자는 그 재작성 단계들입니다. Ansor가 생성한 모든 프로그램은 최초의 단순 구현에서 재작성됩니다. Ansor는 sketch generation과 random annotation 동안 각 프로그램의 완전한 재작성 이력을 보존합니다. 우리는 재작성 단계를 프로그램의 유전자로 볼 수 있는데, 이는 이 프로그램이 최초의 naive 프로그램에서 어떻게 형성되었는지를 기술하기 때문입니다. 이를 바탕으로, 두 기존 프로그램의 재작성 단계를 결합해 새 프로그램을 생성할 수 있습니다. 그러나 두 프로그램의 재작성 단계를 임의로 결합하면 단계 간 의존성이 깨지고 무효한 프로그램이 만들어질 수 있습니다. 따라서 Ansor의 교차 연산 단위는 DAG 내 노드 기반이며, 이는 서로 다른 노드 간의 재작성 단계는 일반적으로 의존성이 적기 때문입니다. Ansor는 각 노드에 대해 부모를 무작위로 선택해 선택된 노드의 재작성 단계를 병합합니다. 노드 간 의존성이 있는 경우, Ansor는 단순한 휴리스틱으로 단계를 분석하고 조정하려고 시도합니다. Ansor는 또한 병합된 프로그램을 검증해 기능적 정확성을 보장합니다. 검증은 간단한데, Ansor는 일부 루프 변환 재작성 단계만 사용하고, 저수준 codegen이 의존성 분석을 통해 정확성을 검사할 수 있기 때문입니다.

evolutionary search는 변이와 교차를 활용해 새로운 후보 집합을 반복적으로 생성하고, 가장 높은 점수를 가진 작은 프로그램 집합을 출력합니다. 이 프로그램들은 타겟 하드웨어에서 컴파일·테스트되어 실제 runtime 비용을 얻게 됩니다. 그런 다음 수집된 측정 데이터를 사용해 cost model을 갱신합니다. 이런 방식으로 학습 가능한 cost model의 정확도가 점차 높아져 타겟 하드웨어에 더 잘 맞게 됩니다. 그 결과, evolutionary search는 점차 타겟 하드웨어 플랫폼에 대한 더 높은 품질의 프로그램을 생성합니다.

TVM과 FlexTensor의 search 알고리즘은 고정된 격자 형 매개변수 공간에서만 작동하지만, Ansor의 진화 연산은 tensor 프로그램을 위해 특별히 설계되어 있습니다. 이들은 일반적인 tensor 프로그램에 적용 가능하며, 의존성이 복잡한 search space를 처리할 수 있습니다. Halide auto-scheduler의 unfold 규칙과 달리, 이 연산들은 프로그램에 대해 비순서 수정을 가할 수 있어 sequential 한계를 해결합니다.

## 5.2 학습 가능한 cost model

cost model은 search 과정에서 프로그램의 성능을 빠르게 추정하는 데 필수적입니다. 우리는 관련 연구[2, 12]와 유사한 학습 가능한 cost model을 채택하되, 새로 설계된 프로그램 특징(feature)을 사용합니다. 학습 가능한 cost model 기반 시스템은 단일 모델 설계가 다른 학습 데이터를 입력해 다양한 하드웨어 백엔드에 재사용될 수 있어, 이식성이 매우 좋습니다.

우리의 타겟 프로그램은 주로 데이터 병렬 tensor 프로그램이며, 여러 개의 중첩된 루프 nest로 구성되고 가장 안쪽 문장은 몇 개의 대입문이므로, 우리는 가장 안쪽의 비루프(non-loop) 문 하나의 점수를 예측하도록 cost model을 학습시킵니다. 완성된 프로그램의 경우, 가장 안쪽의 각 비루프 문에 대해 예측을 수행하고 그 예측치들의 합을 점수로 삼습니다. 우리는 완성된 프로그램의 문맥에서 특징을 추출해 가장 안쪽의 비루프 문에 대한 특징 벡터를 구성합니다. 추출되는 특징에는 산술 특징과 메모리 접근 특징이 포함됩니다. 추출된 특징의 자세한 목록은 부록 B에 있습니다.

우리는 weighted squared error를 손실 함수로 사용합니다. search space에서 성능이 좋은 프로그램을 식별하는 것이 가장 중요하므로, 더 빠르게 실행되는 프로그램에 더 큰 가중치를 둡니다. 구체적으로, 모델 f가 프로그램 P에 대해 throughput y를 가질 때의 손실 함수는 다음과 같습니다:

![이미지](images/img_10.png)여기에 이미지 삽입

여기서 S(P)는 P 내 가장 안쪽의 비루프 문 집합입니다. 우리는 throughput을 가중치로 직접 사용하여, 저수준 모델 f로 gradient boosting 의사결정 트리[9]를 학습시킵니다. 모든 DAG의 모든 tensor 프로그램에 대해 하나의 모델을 학습시키기 위해, 우리는 동일한 DAG의 모든 프로그램의 throughput을 [0, 1] 범위로 정규화합니다. DNN 최적화 시 측정되는 프로그램 수는 보통 30000개 미만입니다. 이렇게 작은 데이터셋에서 gradient boosting 의사결정 트리를 학습하는 것은 매우 빠르므로, 우리는 매번 증분 갱신 대신 새 모델을 학습시킵니다.

# 6. Schedule Task

이 부분은 OpenMMLAB의 글이 비교적 잘 설명하고 있어 번역하지 않고, 여기서 직접 인용합니다:

> OpenMMLAB에서 인용 https://zhuanlan.zhihu.com/p/360041136

ANSOR는 먼저 계산 그래프를 여러 서브그래프로 분할하고 이 서브그래프들을 각각 최적화합니다. 딥러닝 네트워크의 전체 최적화 횟수는 ANSOR 사용자가 지정하며, Schedule Task 모듈이 이 최적화 횟수를 서로 다른 서브그래프 최적화 작업에 어떻게 분배할지 결정합니다.

핫스팟 서브그래프에 대한 집중 최적화를 보장하기 위해, ANSOR는 가능한 한 최적화 효과가 뚜렷한 서브그래프를 골라 최적화합니다. 예를 들어 DNN 네트워크의 latency를 최소화하기 위해, 우리는 먼저 최적화의 목적 함수를 다음과 같이 제시합니다:

![이미지](images/img_11.png)Schedule Task 설명

# 7. 평가

여기서는 논문 내 그림을 바탕으로 Ansor의 성능을 간단히 소개합니다.

![이미지](images/img_12.png)단일 operator 성능

Figure 6에서 볼 수 있듯이, 다양한 operator와 BatchSize 설정에서 Ansor는 모두 최고의 성능을 달성했으며, Ansor의 큰 search space가 성능 향상의 핵심 요인입니다.

![이미지](images/img_13.png)서브그래프에서 Ansor의 성능

여기서 ConvLayer는 Conv+BN+ReLU를 포함하는 서브그래프이고, TBS는 두 개의 행렬 transpose, 하나의 Batch matmul, 하나의 Softmax를 포함하는 서브그래프입니다. @C는 CPU 결과, @G는 GPU 결과를 나타냅니다. CPU든 GPU든, 이런 흔한 서브그래프 최적화에서 Ansor가 전반적으로 앞섭니다.

![이미지](images/img_14.png)Figure 9

Figure 9는 인기 있는 DNN 모델들이 Intel CPU, ARM CPU, NVIDIA GPU에서의 성능 결과를 보여줍니다. 업계 주류 가속 라이브러리들과 비교해, Ansor는 큰 폭의 성능 우위를 가집니다.

마지막으로 많은 분들이 관심을 가질 만한 데이터는 Ansor의 search 시간입니다. Table 3에서 볼 수 있듯이, 인기 있는 DNN 모델들에서 Ansor의 search 시간은 AutoTVM 대비 모두 향상되었습니다. Ansor의 search space가 더 큼에도 불구하고 말입니다.

![이미지](images/img_15.png)Ansor의 search 시간

# 8. 관련 연구

생략. 관심 있다면 원 논문을 참고하세요.

# 9. 현재와 향후 작업

생략. 관심 있다면 원 논문을 참고하세요.

# 10. 결론

우리는 Ansor를 제안했습니다. 이는 딥러닝 신경망을 위한 고성능 tensor 프로그램을 생성하는 자동 search 프레임워크입니다. 큰 search space를 효과적으로 탐색하고 성능 병목을 우선시함으로써, Ansor는 기존 방법의 search space 밖에 있는 고성능 프로그램을 찾습니다. Ansor는 다양한 신경망과 하드웨어 플랫폼에서 기존의 수작업 라이브러리와 search 기반 프레임워크보다 최대 3.8배 우수한 성능을 보입니다. 더 나은 프로그램을 자동으로 search함으로써, Ansor가 점점 커지는 컴퓨팅 능력 수요와 제한된 하드웨어 성능 사이의 격차를 줄이는 데 도움이 되기를 바랍니다. Ansor는 Apache TVM 오픈소스 프로젝트에 통합되었습니다.
