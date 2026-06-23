# Roller: Fast and Efficient Tensor Compilation for Deep Learning

오늘은 최근 OSDI에 공개된 Microsoft의 Roller 논문을 읽어보겠습니다. 제목은 《Roller: Fast and Efficient Tensor Compilation
for Deep Learning》입니다.

- 논문 링크: https://www.usenix.org/conference/osdi22/presentation/zhu
- 코드 링크: https://github.com/microsoft/nnfusion/

얼마 전에 OSDI 2021의 《PET: Optimizing Tensor Programs with Partially Equivalent Transformations
and Automated Corrections》라는 논문 해설을 공유한 적이 있습니다. 작년에는 OSDI 2020 《Ansor: Generating High-Performance Tensor Programs for Deep Learning》 논문 해설도 공유했습니다. 이 두 논문의 해설은 다음 주소에서 확인할 수 있습니다: https://github.com/BBuf/tvm_mlir_learn/tree/main/paper_reading 혹은 지후(Zhihu) 메인 페이지에서도 찾을 수 있습니다. Ansor의 주요 기여는 효율적인 schedule(루프 언롤링, 합치기, 분할, 캐시 활용, 병렬도 변경 등)을 자동으로 찾을 수 있게 한 것이며, 더 이상 개발자가 TVM에서 Tensor Expression 기반의 schedule 템플릿을 손으로 작성할 필요가 없게 만들었습니다. 이로써 operator 컴파일러(Tensor Compiler)의 사용성이 크게 향상되었고, 일반적인 operator와 모델에서도 좋은 성능을 보입니다. AutoTVM의 업그레이드 버전이라고 할 수 있습니다(AutoTVM은 검색할 schedule 템플릿을 수동으로 지정해야 하기 때문입니다: https://zhuanlan.zhihu.com/p/508283737). PET는 operator의 schedule에는 관심을 두지 않고, 부분 등가 변환이라는 새로운 관점에서 병렬도를 높이거나 캐시 활용을 개선해 가속 효과를 달성하는데, Roller 논문과는 직접적인 관련이 없으니 읽지 않으셔도 무방합니다.

> 최근 TVM 관련 질문을 많이 받는데, 저도 취미로 보고 있어서 잘 답변드리지 못할 때가 많습니다. TVM 논의를 위한 위챗 그룹을 만들었으니 필요한 분들은 서로 물어보면서 도움을 받을 수 있습니다. 위챗 bbuf23333을 추가하시고 tvm이라고 메모를 남겨 주세요. 그리고 취미로 컴파일러를 접한 1년간 정리한 이 지식 저장소는 이미 500개 이상의 star를 받았습니다. 감사드리며, 더 많은 관심 부탁드립니다.

https://github.com/BBuf/tvm_mlir_learn

Ansor, AutoTVM, PET(코드 생성의 일부도 TVM AutoTVM/Ansor 기반) 모두 동일한 문제에 직면해 있습니다. 바로 operator의 schedule을 탐색할 때 많은 시간이 소요된다는 점입니다. 특정 하드웨어에서 일반적인 비전 모델 하나를 자동 튜닝하고 코드 kernel을 생성하는 데 수 시간이 걸립니다. 이는 AI 컴파일러를 모델 배포에 적용하는 데 큰 걸림돌이 됩니다. 이러한 문제점에 기반해 Roller가 등장했습니다.



## 0x0. 제목 & 저자 & 초록
![제목](https://img-blog.csdnimg.cn/b2dcee579ffa4910b9f103f0308e7555.png)
ROLLER: 딥러닝을 위한 빠르고 효율적인 Tensor 컴파일러. 저자들은 Microsoft Research Asia 및 토론토 대학교 등 여러 대학 출신입니다.


최신 Tensor 컴파일러들은 많은 진전을 이루었지만, 일반적으로 효율적인 Kernel을 탐색하고 생성하는 데 수 시간이 소요됩니다. 이는 기존 Tensor 컴파일러가 지정하는 탐색 공간이 매우 크기 때문입니다. 컴파일 시간이 긴 문제를 해결하기 위해 본 논문은 Roller를 제안합니다. Roller의 핵심은 **rTile**로, 이는 새로운 tile 추상화입니다. **이는 하부 가속기의 핵심 특성과 일치하는 tensor shape을 캡슐화하여, shape 선택을 제한함으로써 효율적인 실행을 가능하게 합니다**. Roller는 rTile 기반의 재귀적 구성 알고리즘을 채택하여 목표 프로그램(rProgram)을 생성합니다. **최종적으로 Roller는 수 초 내에 효율적인 Kernel을 생성할 수 있으며, 현재 주류 가속기에서 다른 Tensor 컴파일러와 견줄 만한 성능을 보이고, IPU 같은 새로운 가속기에서는 더 나은 Kernel을 생성합니다**.

> 아직은 잘 와닿지 않으니 계속 읽어봅시다. 여기서 말하는 tile이란 입력을 분할하여 하드웨어의 메모리 구조에 맞춘다는 의미입니다. 이전 글에서 자세히 다뤘으니, tile에 대한 기초 설명을 보고 싶으신 분은 먼저 이 글을 참고하세요: https://zhuanlan.zhihu.com/p/508283737 .

## 0x1. 머리말
딥러닝 신경망의 중요성이 점점 커지고 있고, 하드웨어에서 효율적인 Kernel을 생성하는 딥러닝 컴파일러의 중요성도 함께 커지고 있으며 많은 성과를 거두었습니다. 그러나 현재의 컴파일러들은 효율적인 Kernel을 생성하기 위해 수 시간 또는 며칠에 걸친 탐색이 필요합니다. 그 이유는 모두 신경망 안의 operator를 다중 중첩 루프로 구현하기 때문입니다. Tensor 컴파일러는 일반적으로 구현된 다중 중첩 루프 계산에 대해 루프 언롤링, 병합, 분할, 캐시 활용, 병렬도 변경 등을 수행하여 하드웨어의 메모리 구조(예: CPU의 3단계 Cache나 CUDA의 global memory, l2 cache, l1 cache 구조)나 하드웨어 특성(예: 벡터화, 병렬화)에 맞춥니다. 여기에는 매우 크고 복잡한 탐색 공간이 관여하므로 탐색 시간이 길어집니다. 본 논문이 제안하는 Roller는 탐색 시간이 긴 문제를 해결하며, 다음과 같은 특징이 있습니다.
- 첫째, Roller는 DNN의 operator 계산을 다중 중첩 루프로 보지 않고, 데이터 처리 파이프라인으로 봅니다. 그 안에서 데이터 블록(tile)이 GPU SM과 같은 병렬 실행 유닛 및 메모리 계층 구조 추상화를 갖춘 하드웨어 위를 이동하고 처리됩니다. 효율적인 Kernel 생성의 목표가 파이프라인의 처리량을 높이는 목표로 바뀝니다.
> Roller는 operator의 계산 과정을 데이터 블록(tile) 기반의 파이프라인으로 모델링합니다. 즉, 다양한 크기의 데이터 블록을 다단계 메모리 구조에서 SM 같은 프로세서로 옮겨 계산하고, 단계별로 다시 기록하는 방식입니다.
- 다음으로, tile 기반 파이프라인의 처리량을 최대화하려면, 각 단계의 데이터 블록(Tile) shape이 하드웨어의 파라미터 설정과 일치해야 합니다(논문에서는 이를 **정렬(alignment)**이라고 부릅니다). 예를 들어 memory bank, memory transaction length, minimum schedulable unit(예: GPU의 warp size) 같은 메모리 대역폭 및 병렬도 관련 설정과 정렬되어야 합니다. 이러한 제약은 tensor 프로그램이 각 메모리 단계에서 좋은 계산 효율을 갖도록 할 뿐 아니라, 이전의 다중 중첩 루프 기반 파라미터 탐색 공간을 크게 줄여 줍니다. 따라서 Tensor 컴파일러가 schedule 탐색에 소요하던 많은 컴파일 시간 문제도 해결됩니다.
- 마지막으로, **하드웨어에 정렬된 데이터 처리 파이프라인의 성능은 매우 예측 가능합니다. 메모리 처리량은 하드웨어 사양이나 Benchmark 테스트로부터 도출할 수 있으므로, 다양한 하드웨어에 정렬한 후 성능을 추정하는 난이도가 크게 낮아지며, 더 이상 하드웨어에 기반한 복잡한 비용 모델을 구축해 성능을 추정할 필요가 없습니다.**

이러한 아이디어들에 기반해 Roller는 rTile을 제안합니다. 이는 새로운 추상화로, 하드웨어 가속기의 핵심 특성 및 입력 tensor shape과 일치하는 데이터 블록(Tile) shape을 캡슐화합니다(뒤에서 자세히 살펴봅니다). 그리고 데이터 처리 파이프라인을 rTile 기반의 프로그램(rProgram)으로 표현하며, rTile에 작용하는 Load, Store, Compute 세 가지 인터페이스로 구성됩니다. 효율적인 rProgram을 구성하기 위해 Roller는 scale-up-then-scale-out 방식을 따릅니다. **먼저 Scale-up 과정을 수행해 rTile 기반의 재귀적 구성 방식(Figure 8)으로 rTile shape 크기를 점진적으로 늘려, 가속기의 단일 실행 유닛(예: SM)을 포화시키는 rProgram을 구성합니다. 그 후 Scale-out 과정을 수행하는데, 딥러닝 계산 패턴과 가속기의 병렬 실행 유닛이 동질적이기 때문에, 생성된 rProgram을 다른 병렬 실행 유닛으로 단순히 복제하기만 하면 됩니다. 여기서 scale-up-then-scale-out은 종적 확장과 횡적 확장이라고 부를 수 있습니다.**

Roller는 큰 오버헤드 없이 다양한 rTile들의 성능을 평가할 수 있습니다. 각 operator는 피크와 대역폭을 간단히 측정해보면 됩니다. 하드웨어 구조에 정렬되어 있기 때문에, rTile의 메모리 부담 같은 다른 핵심 성능 요인들도 하드웨어 규칙 분석으로부터 얻을 수 있습니다. 이렇게 해서 효율적인 마이크로 평가 모델이 만들어졌고, 다른 컴파일러처럼 각 설정을 비싼 비용을 들여 온라인으로 분석할 필요가 없어 컴파일 과정이 크게 가속화됩니다. 또한, 엄격한 정렬 요건 덕분에 재귀적 구성 과정을 통해 원하는 rTile과 rProgram을 빠르게 생산할 수 있습니다. 종합하면, Roller는 수 초 내에 효율적인 Kernel을 생성할 수 있습니다.

저자 팀은 TVM과 Rammer(Rammer는 다음을 참고: https://www.msra.cn/zh-cn/news/features/osdi-2020-rammer) 위에 Roller를 구현하고 코드를 오픈소스로 공개했습니다. 광범위한 실험을 통해 Roller가 수 초 내에 고도로 최적화된 Kernel을 생성할 수 있음을 보였으며, 특히 대규모 사용자 정의 고비용 operator에서 두드러집니다. 컴파일 시간 면에서 3자릿수의 개선을 달성했습니다. Roller가 생성한 Kernel은 최첨단 Tensor 컴파일러나 하드웨어 벤더가 제공하는 가속 라이브러리와 견줄 만하며, 일반적으로 더 좋은 적용성을 갖습니다(새로운 하드웨어에 대한 적용성을 의미). rTile 기반의 세 가지 인터페이스(Load, Store, Compute)로 프로그램을 기술하기 때문에, Roller는 AMD GPU와 Graphcore IPU 같은 다양한 가속기에 쉽게 적응할 수 있습니다.

## 0x2. 동기와 핵심 관찰
- Excessive compilation time: Tensor 컴파일러의 컴파일 시간이 너무 길어 생산성에 영향을 줍니다.
- Observation and insights: 우리는 딥러닝 operator의 계산을 바라보는 다른 시각이 있다는 점을 관찰했습니다. 행렬 곱셈 $C_{m,n}=A_{m,k}\times B_{k, n}$을 예로 들어 보겠습니다. 기존 컴파일러가 MatMul을 $m, n, k$ 세 중첩 루프로 보는 것과 달리, operator의 계산 과정 또한 데이터 처리 파이프라인입니다. 우리는 A와 B에서 두 개의 부분 행렬(tile)을 Load하고, 두 부분 행렬을 Compute하며, 결과를 C 메모리에 Store합니다. 따라서 계산의 성능은 Load-Compute-Store 파이프라인이 하나의 Tile을 얼마나 빠르게 이동시키는지에 달려 있습니다.

파이프라인 모든 단계의 핵심 성능에 영향을 주는 요인은 Tile shape과 1차원 메모리 공간 내의 배치입니다. Figure 1(a)는 C에서 한 원소의 계산과 메모리 접근 패턴을 보여줍니다. 모든 행렬이 행 우선 배치로 저장된다고 가정하면, B에서 열을 로드할 때 1번의 stride 접근이 발생합니다. 메모리 트랜잭션 길이(the memory transaction length)가 4라고 하면, 3/4의 잉여 데이터 읽기가 발생합니다. 따라서 데이터 블록의 형태는 메모리 트랜잭션 길이에 정렬되어 효율적인 메모리 접근이 가능해야 합니다. Figure 1(b)에서 1x4 Tile 단위로 B를 계산하면 메모리 대역폭 낭비가 없습니다. 메모리 정렬 외에도, 데이터의 Tile shape은 병렬 스레드 수 같은 하드웨어 실행 유닛과도 정렬되어야 계산 사이클을 낭비하지 않습니다. 또한 Cache의 존재로 인해 Tile shape은 데이터 재사용 기회에도 영향을 줍니다. 예를 들어 Figure 1(a)에서는 매번 1x1 tile을 계산할 때마다 2mnk개의 데이터를 읽어야 합니다. 그러나 Figure 1(b)에서는 1.25mnk번의 읽기만 필요한데, A에서 한 번 읽은 데이터를 4번 재사용할 수 있기 때문입니다. M 차원의 tile 크기를 4x4로 설정하면 총 reads는 0.5mnk까지 줄일 수 있어, Figure 1(a)보다 데이터 읽기 효율이 10배 향상됩니다.

![Figure1](https://img-blog.csdnimg.cn/a90fb56c27654759aea4a2140dc1212f.png)

## 0x3. 시스템 설계

아래 Figure 2는 Roller의 시스템 설계를 보여줍니다. Roller의 입력은 TE 표현식입니다. 이 표현식은 사용자가 작성하거나 다른 컴파일러로부터 생성됩니다(이 단계에서 일부 fusion 연산이 일어날 수 있습니다). Roller는 TE에서 tensor 형상을 추출하고 하드웨어 사양에 기반해 rTile, 즉 하드웨어에 정렬된 빌딩 블록을 구성합니다. rTile에 기반해 Roller는 횡적 확장 및 종적 확장 재귀 구성 알고리즘을 제안하여, 데이터 처리 파이프라인을 기술하는 효율적인 tensor화된 프로그램(rProgram)을 생성합니다. rProgram을 생성하면서 구성 알고리즘은 마이크로 성능 모델을 통해 구성된 rProgram의 성능을 평가하여 좋은 rTile 설정을 식별합니다. 이는 하드웨어 추상화로 기술된 디바이스 위에서 동작하며, rTile과 관련된 인터페이스인 Load/Save/Compute만 노출합니다. 구성된 rProgram은 최종적으로 codegen을 통해 디바이스별 최종 Kernel로 생성됩니다.

![시스템 개요](https://img-blog.csdnimg.cn/b54a0b2dd06346bcb15970c88a4a0ee8.png)

### 0x3.1 Tensor Expression and rTile
Roller는 TVM에서 도입한 Tensor Expression을 컴파일러의 입력으로 가져옵니다. Tensor Expression은 여기서는 다루지 않으며, 잘 모르신다면 TVM 안에서 chen tianqi가 작성한 문서를 참고하세요. https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html 

**Roller는 rTile을 기본 계산 단위로 도입하여 tensor 계산을 구성합니다**. Figure 3에서 볼 수 있듯이, rTile은 주어진 tensor 표현식 expr의 각 루프 축을 따라 정의된 다차원 tile shape을 캡슐화합니다. shape과 expr이 주어지면, rTile은 관련된 입력과 출력 데이터 블록을 정적으로 추론할 수 있습니다. 예를 들어 축 i, j, k를 따른 tile shape은 위 MatMul 표현식의 rTile을 나타내며, 각 rTile은 A에서 4x2개의 데이터와 B에서 2x4개의 데이터를 로드해 총 4x2x4번의 mul-add 계산을 수행하고, 4x4 데이터 tile을 C에 다시 기록합니다. Figure 4에 나타난 대로입니다.

![Figure3](https://img-blog.csdnimg.cn/8073531426614b9fa2e4d8d0904a3245.png)

![Figure4](https://img-blog.csdnimg.cn/8fe8577b6d40428d86ac778207f052ba.png)

**rTile의 독특한 특성은 주어진 tensor 표현식의 하부 하드웨어 특성 및 입력 Tensor shape과 일치해야 한다는 점입니다. 모든 정렬 방식은 Figure 3의 rTile shape과 storage_padding으로 제어되며, 각각 rTile의 논리적 형태와 물리적 배치를 나타냅니다**. 다음으로 정렬에 대한 상세 요건을 자세히 설명합니다.

- **Alignment with the hardware execution unit**. 첫째, rTile의 shape은 그것이 실행되는 실행 유닛의 병렬도와 정렬되어야 합니다. 예를 들어, GPU에서 실행되는 rTile의 shape 크기는 wrap size의 배수, 예컨대 32여야 최대 계산 효율에 도달할 수 있습니다. NVIDIA GPU에서 TensorCore를 사용할 때는 rTile shape 크기가 16x16x16의 배수여야 합니다.
- **Alignment with memory transaction**. 둘째, 데이터 블록(Tile)의 shape은 메모리 트랜잭션 길이와 일치해야 최적의 메모리 접근이 가능합니다. 구체적으로 rTile의 각 데이터 블록에 대해 Leading dimension(예: 행 우선 Tensor의 가장 안쪽 차원)이 메모리 트랜잭션 길이의 배수가 되어야 합니다. Figure 5(a)에서 보듯이, Roller에서 tensor 메모리는 캐시 정렬 방식으로 할당됩니다. 따라서 rTile은 shape이 메모리 트랜잭션 길이에 정렬되어 있어 메모리 읽기 낭비를 피할 수 있습니다.

> 글로벌 메모리 대역폭을 최대한 활용하고 글로벌 메모리 로드 효율을 높이는 것은 Kernel 최적화의 기본 조건입니다. 정렬되지 않은 메모리는 대역폭 낭비를 초래합니다. 다음을 참고하세요: https://face2ai.com/CUDA-F-4-3-%E5%86%85%E5%AD%98%E8%AE%BF%E9%97%AE%E6%A8%A1%E5%BC%8F/

![Figure5](https://img-blog.csdnimg.cn/6d14f15c8abb4e02b453cb9c9367f8b4.png)

- **Alignment with memory bank.** 셋째, 데이터 블록의 메모리 배치는 Memory Bank와 정렬되어야 읽기 충돌을 피할 수 있습니다. 예를 들어 Figure 5(b)에서 데이터 블록 a(shape이 [3, 4])는 4개의 bank에 걸쳐 메모리에 저장되며, [3, 1] 형태의 블록으로 읽힙니다. 이 [3, 1] 형태의 작은 블록의 데이터를 하나의 bank에 저장하는 단순한 방식은 로드 충돌을 일으킵니다. rTile은 padding을 통해 이러한 비효율을 피합니다. Leading dimension이 N인 데이터 블록이 Leading dimension이 n인 다른 블록에 의해 읽힐 때, N 차원을 따라 padding_size 크기의 padding을 적용합니다.

![여기에 이미지 삽입](https://img-blog.csdnimg.cn/745f3ba68d3e4090b05bbddf77b84da8.png)

여기서 B와 L은 각각 bank 수와 bank의 폭입니다. 각 차원의 padding 크기는 계산되어 Figure 3의 storage_padding 필드에 저장됩니다. Figure 5(b)의 경우 padding_size 1로 채우면 모든 [3x1] 값들이 서로 다른 bank에 분포되어 효율적으로 읽을 수 있습니다.

> GPU Shared Memory bank conflict: https://blog.csdn.net/Bruce_0712/article/details/65447608

- **Alignment with tensor shape**. 마지막으로, **rTile의 shape은 입력 tensor 표현식의 tensor shape과 정렬되어야 합니다**. **그렇지 않으면 계산이 rTile에 의해 균등하게 나뉘지 않아 계산 자원을 낭비하거나 막대한 경계 검사 오버헤드가 발생합니다**. 간단한 해결책은 Tensor의 차원 $i$(크기 $N_i$)에 대해 padding을 적용하여, padding 크기 $P_i$를 두어 $N_i+P_i$가 차원 i에서 rTile shape 크기의 배수가 되도록 하는 것입니다. 그러나 큰 padding kernel은 계산 낭비를 가져오므로, Roller는 tensor padding을 $\varepsilon$ 이내로 제한하며 다음 식을 만족해야 합니다: $\frac{S_i-N_i \mod S_i }{N_i}<= \varepsilon$. 이는 계산 낭비 비율의 상한을 ε으로 보장합니다. 이 제한이 있다면, 이 조건을 만족하는 모든 유효한 rTile 형상을 열거할 수 있습니다.

- **Deriving allrTiles.** 위의 정렬 요건을 고려하여, 특정 tensor 표현식과 하드웨어 디바이스에 대해 Roller는 다음 인터페이스를 사용해 모든 적합한 rTile을 점진적으로 도출합니다:

```cpp
vector<int> GetNextAlignedAxisSize(rTile T, Dev d),
```

디바이스의 지정된 파라미터가 주어지면, 이는 rTile shape의 각 차원에 대한 다음 정렬된 크기를 반환합니다. 이는 모든 정렬 요건을 만족할 때까지 각 차원의 크기를 점진적으로 늘려 계산됩니다. rTile 추상화 덕분에 Roller는 새로운 정렬 요건을 지원하도록 확장할 수 있는데, 이는 `GetNextAlignedAxisSize` 인터페이스를 통해 구현됩니다.

- **Calculating data reuse score**. rTile의 흥미로운 특성 중 하나는 **shape을 조정해 메모리 트래픽을 암묵적으로 제어할 수 있다**는 점입니다. rTile 크기를 늘리면 보통 더 많은 메모리를 점유하는 대가로 프로그램에 더 많은 데이터 재사용 기회를 가져옵니다. 주어진 rTile T와 각 축에서의 다음 정렬 크기를 사용해, 다음 식으로

![여기에 이미지 삽입](https://img-blog.csdnimg.cn/ada59dc8a4564679bd74b482878ffe98.png)

축 $i$의 데이터 재사용 점수 $S_i$를 계산할 수 있습니다. 여기서 $T_i^{'}$는 `GetNextAlignedAxisSize`로 얻은 다음 정렬 크기로 축 $i$의 차원 크기를 대체해 얻은 더 큰 rTile입니다. 함수 Q(T)와 F(T)는 T 단위로 계산을 수행할 때의 메모리 트래픽과 메모리 점유량을 계산하는데, 주어진 tensor 표현식과 하드웨어 메모리 사양으로부터 직접 추론할 수 있습니다(0x3.3절 내용). $S_i$가 클수록 같은 메모리를 사용할 때 메모리 트래픽을 더 많이 절약할 수 있다는 의미입니다. 메모리 재사용 점수는 효율적인 rProgram을 (rTile들로) 구성하는 데 핵심적인 역할을 합니다.

### 0x3.2 Tensor Program Construction
- **rTile program**. rTile과 현대 가속기의 메모리 계층 구조가 주어지면, tensor 계산은 자연스럽게 데이터 흐름 처리 파이프라인으로 볼 수 있습니다. 계산은 가장 낮은 메모리 단계에서 데이터 블록(rTile에서 지정됨)을 로드하고, 가속기의 실행 유닛에서 rTile에 대해 계산하며, 결과 데이터 블록을 가장 낮은 메모리 단계에 다시 기록합니다. **각 메모리 단계마다 그 메모리 단계의 특성과 일치하는 특정 rTile이 정의됩니다.** 따라서 Roller는 tensor 계산을 계층적 rTile 설정을 가진 데이터 처리 파이프라인으로 기술하며, 이를 rProgram이라고 부릅니다.

Figure 6은 세 개의 메모리 계층(L0, L1, L2)을 가진 디바이스에서의 rProgram을 보여줍니다. rProgram은 각 메모리 계층의 rTile과 rTile 명령(Load, Store, Compute)으로 기술됩니다.

![Figure6](https://img-blog.csdnimg.cn/1b0f52ec84aa4dcfa0012759c71c57b9.png)

Figure 7(a)는 Figure 7(b)에 대응하는 MatMul 프로그램을 보여줍니다. Figure 7(c)는 rProgram이 디바이스의 각 메모리 계층에 어떻게 매핑되는지 설명합니다. **구체적으로, 매번 메모리 L2에서 A의 4x4 작은 블록과 B의 4x8 작은 블록을 L1으로 로드합니다. 그런 다음 L1에서 A의 2x1과 B의 1x2 작은 블록을 L0(레지스터)으로 로드합니다. 매번 계산이 끝나면 결과인 2x2 작은 블록이 L0에서 L2로 직접 다시 기록됩니다.**

![여기에 이미지 삽입](https://img-blog.csdnimg.cn/75c7681fde7c4449bb615aacf7d69e3b.png)

데이터 처리 파이프라인이 주어지면, 대응하는 rProgram의 최적화 목표는 파이프라인의 처리량을 최대화하는 것입니다. 이 목표는 다음 세 가지 조건을 만족하는 것으로 변환할 수 있습니다: **1) 계산과 메모리 이동이 하드웨어 특성을 충분히 활용해야 한다. 2) 처리량은 병목 단계(피크에 가까운)에 도달해야 한다. 3) 모든 병렬 실행 유닛을 활용할 수 있는 충분한 병렬도가 있어야 한다**. 따라서 Roller는 다음 rProgram 구성 전략을 제안합니다: 먼저 단일 코어 rProgram을 구성하여 한 코어에서 종적으로 확장하여 Kernel의 하드웨어 활용도를 포화시킵니다. 그런 다음 구성된 단일 Kernel을 복제해 횡적으로 확장하여 하드웨어의 병렬도를 활용합니다.

- **Scaling up an rProgram**. rTile의 정렬 속성이 하드웨어 효율을 보장하므로, Roller는 올바른 rTile shape을 구성하여 각 메모리 계층의 처리량을 최대화하는 데에만 집중하면 됩니다. 0x3.1절에서 정의한 데이터 재사용 점수를 활용해, 단일 코어 rProgram 구성 알고리즘은 초기 rTile에서 시작하여 rTile에서 가장 큰 이득을 주는 축(즉, 가장 큰 재사용 점수를 갖는 축)으로 점차 rTile을 확장합니다. **구성 알고리즘은 정확한 데이터 재사용 점수가 필요하지 않으며, 단지 가장 큰 점수를 선택해 처리량을 최대화합니다. 이 과정에서 메모리 성능은 계산 피크나 최대 메모리 용량에 도달할 때까지 향상됩니다**. 위 과정은 위에서 아래로 각 메모리 계층에 대해 반복되어, 원하는 rProgram이 구성될 때까지 진행됩니다. 만약 일부 tensor 표현식의 데이터 재사용 점수가 변하지 않는 경우(예: elementwise operator), Roller는 최상위 계층에 대한 rTile만 구성하고 가장 낮은 메모리에서 로드합니다.

![Figure8](https://img-blog.csdnimg.cn/35888586d3024aea810f5777200145a0.png)

Figure 8은 상세한 구성 알고리즘을 보여줍니다. **tensor 표현식 expr과 목표 디바이스 dev가 주어지면, 알고리즘은 최상위 메모리 계층에서 초기화된 rTile T를 구성하고 재귀적으로 T를 확장합니다(4번째 줄의 EnlargeTile에 해당). 매 단계마다 데이터 재사용 점수를 최대화하는 다음 더 큰 rTile T'를 열거합니다(10번째 줄의 GetNextRTileShapes에 해당). 만약 T'가 메모리 용량에 도달하거나(13번째 줄), 데이터 블록 로드의 처리량 MemRef(T')가 피크 계산 처리량 MaxComputePerf(T')를 초과하면(17번째 줄), 알고리즘은 현재 rTile을 기록하고 다음 메모리 계층에서 EnlargeTile을 계속합니다. 그렇지 않으면 현재 메모리 계층에서 T'를 계속 확장합니다(20번째 줄). 구성은 가장 낮은 메모리 계층에서 완료되며(6번째 줄), 결과 하나가 생성되고 K개의 rProgram이 생성될 때까지 반복 실행됩니다(컴파일러의 숨겨진 요인 영향을 흡수하기 위해). 여기서 MemPerf(T′)와 MaxComputePerf(T′)는 dev와 0x3.3절의 마이크로 성능 모델을 기반으로 도출됨에 유의하십시오.**

- **Scaling out an rProgram**. 대부분의 DNN operator의 계산 패턴과 가속기의 병렬 실행 유닛이 동질적이라는 점을 감안할 때, **Roller는 계산을 가장 낮은 메모리 계층의 rTile과 같은 크기의 rTile들로 균등 분할하여, 한 실행 유닛에서 구성된 rProgram을 다른 유닛으로 단순히 복제합니다. 이는 모든 rTile들을 모든 실행 유닛에 균등하게 배분함으로써 구현합니다. Roller는 reduce 축을 같은 실행 유닛에 할당하는 것을 선호하는데, 이는 더 높은 메모리 계층에서 reduce 결과를 공유할 수 있기 때문입니다.** Roller는 모든 계산 유닛을 독점한다고 가정하지 않으며, 시스템은 횡적 확장 시 rProgram의 병렬도를 명시적으로 제어할 수 있습니다.
- **Small operator and irregular tensor shape**. 횡적 확장 알고리즘은 본래 충분한 병렬도를 가진 operator에 유리합니다. 예를 들어, 분할 수가 실행 유닛 수보다 훨씬 많은 경우입니다. 작은 operator의 경우, 알고리즘의 전반적인 성능 kernel은 병렬 실행 유닛 활용률이 낮은 것에 영향을 받습니다. 이는 Rammer 컴파일러의 동시에 일부 작은 Kernel들을 schedule하는 방식으로 해결할 수 있습니다. 또 다른 방법으로는 각 rProgram에 대해 **Roller가 가장 작은 데이터 재사용 점수를 가진 축을 따라 rTile을 축소하여 충분한 병렬도를 확보하려 시도**합니다. 다른 정렬 규칙과 마찬가지로, 이 열거 과정은 매번 다음 정렬된 Tile 크기를 반환하는 효율적인 과정이며, 전체 구성 과정과 비교하면 비용이 무시할 만합니다.

또한 큰 operator는 불규칙하고 크기가 작은 tensor 차원을 포함할 수 있으며, 이 경우 Roller는 정렬 요건 때문에 충분한 수의 rProgram을 생성하지 못할 수 있습니다. 이를 해결하기 위해 Roller는 축 융합 pass를 통해 tensor 표현식을 정규 형태로 변환합니다. 구체적으로, 관련된 모든 tensor에서 한 tensor의 인접한 두 축이 다른 모든 tensor에서도 인접해 있거나 모두 없는 경우, Roller는 안전하게 이 두 축을 합칠 수 있습니다. 예를 들어 입력과 출력 tensor 형상이 모두 [17, 11, 3]인 tensor라면 Roller는 이 세 차원을 fuse하여 $[561](17\times 11\times 3)$이 되게 합니다. 축 융합 외에도, Roller는 tensor 패딩 메커니즘에서 파라미터 $\varepsilon$을 점진적으로 늘리는 탐욕적 시도를 하여, kProgram 구성이 완료될 때까지 진행합니다.

### 0x3.3 Efficient Evaluation of an rProgram
구성 알고리즘에서 Roller는 rProgram의 성능을 평가해야 합니다. Roller는 실제 하드웨어 디바이스에서 종단 간 rProgram을 평가할 필요 없이 rTile의 성능만 평가하면 됩니다. Figure 8의 MemPerf와 MaxComputePerf와 같습니다.

이를 위해 Roller는 하드웨어 추상화 계층(HAL)에서 기술된 디바이스에 대한 마이크로 모델을 구성합니다. **HAL은 가속기를 계층적 메모리를 가진 다중 병렬 실행 유닛으로 모델링하며, HAL은 rTile 기반의 세 가지 인터페이스 Load, Save, Compute를 노출합니다. 실행 유닛은 rTile Execution Unit(TEU)로 추상화되며, Compute 인터페이스를 통해 데이터 블록에 대한 계산을 수행합니다**. 여러 TEU를 한 그룹으로 조직하여 협력적으로 Tile을 로드하고 저장할 수 있습니다. HAL은 서로 다른 메모리 계층(예: 레지스터, 공유 메모리, DRAM)을 통일된 한 가지 유형으로 보고, Tile 성능에 영향을 주는 하드웨어 사양을 노출합니다. **하드웨어 사양에는 메모리 용량, 트랜잭션 길이, 캐시 라인 크기, Memory Bank 수가 포함되며**, Figure 9의 getDeviceSpec을 통해 가져올 수 있습니다.

![여기에 이미지 삽입](https://img-blog.csdnimg.cn/c5262d482f714d8fa4707153fc2d202c.png)

- **Micro performance model**. 하드웨어 추상화 계층의 도움으로 Roller는 rTile(과 rProgram)의 성능을 쉽게 도출할 수 있습니다. 첫째, rTile이 주어지면 rTile의 tensor 표현식 expr과 shape(Figure 9의 MemFootprint와 MemTraffic 인터페이스)으로부터 발생하는 메모리 점유량(padding 포함)과 다른 계층 간 메모리 트래픽을 정적으로 추론할 수 있습니다. 데이터 재사용 점수를 계산하고 rTile이 이미 메모리 용량을 초과했는지 확인합니다. 둘째, rTile의 MaxComputePerf를 계산하기 위해 Roller는 Tile shape을 적극적으로 확장해 TEU를 포화시키는 일회성 분석을 수행하여 피크 계산 처리량을 측정합니다. 이 성능 데이터는 Roller에 캐시되어 추후 구성 알고리즘에서 조회됩니다. 마지막으로, 주어진 rTile에 대해 Roller는 MemPerf(낮은 메모리 계층에서 더 높은 계층으로 로드하는 성능) 또한 추정합니다. **rTile에서 정렬된 메모리 접근이 주어졌을 때, 일반적인 Tile 로드의 지연은 단순히 총 트래픽을 메모리 대역폭으로 나누어 모델링할 수 있습니다**. 모든 TEU가 공유하는 메모리 계층에 대해서는 대역폭을 균등 분할합니다. 더 작은 접근 메모리에 대해서는, Roller는 디바이스 유형마다 한 번의 오프라인 분석을 수행하고 결과를 캐시합니다. 주목할 점은, **마이크로 성능 모델은 Tile shape이 완전히 정렬된 경우에만 준비하면 된다는 것입니다. 이는 Roller의 핵심 요구사항입니다.**


## 4. 구현 세부 사항
- **코드 생성**: 고정된 코드 구조(예: Figure 6의 한 rProgram)가 주어지면, Roller는 미리 정의된 템플릿(TVM 내장 schedule 프리미티브)을 기반으로 코드를 생성합니다. 각 메모리 계층에서 데이터 블록을 로드하고 저장하는 것은 TVM의 cache_read와 cache_write 프리미티브로 구현됩니다. rTile의 분할은 split과 fuse로 수행됩니다. 일부 rTile 계산 프리미티브는 TVM 내장 API로 완료됩니다. 템플릿을 기반으로, 주어진 rProgram은 직접 cuda 코드를 생성할 수 있습니다.
- **Tensor Padding**: Roller는 rTile과 tensor shape을 정렬하기 위해 tensor padding에 의존합니다. 실제로 가장 낮은 메모리(예: DRAM)의 대부분의 tensor는 외부 프로그램(예: DNN 프레임워크)에 의해 할당되므로, 우리는 상위 메모리(예: 공유 메모리)에만 padding을 적용하면 됩니다. Roller의 tensor padding은 현재 입력 tensor 표현식이 padding을 허용하는지 여부와 기본 padding 값(예: MatMul operator의 경우 0)을 지정하도록 요구합니다. Memory Bank 정렬을 위한 storage padding의 경우, TVM의 storage_align 프리미티브를 활용해 padding을 추가합니다.
- **Performance profiling**. Roller는 두 개의 성능 분석기를 구현했습니다. 마이크로 성능 분석기와 kernel 분석기입니다. 전자는 micro-benchmark를 통해 메모리 대역폭, 계산 처리량 등 하드웨어 지표를 생성합니다. 이는 디바이스 유형과 tensor 표현식마다 일회성 오프라인 분석입니다. 후자는 top K개의 kProgram 중 가장 빠른 kernel을 기술하는데, k가 1보다 크면 컴파일 결과마다 사용됩니다. 실제로 특정 kernel 코드의 성능은 디바이스 컴파일러와 하드웨어 관련 숨겨진 요인의 영향을 약간 받으며, Roller는 이를 거의 제어할 수 없습니다. 이러한 요인에는 서로 다른 명령어 유형의 명령어 밀도, 레지스터 할당 동작, 디바이스 컴파일러 최적화, warp scheduling 오버헤드 등이 있습니다. 특히 NVIDIA GPU에서 Roller는 nvcc에 의존하여 생성된 CUDA 코드를 머신 코드로 컴파일합니다. 그러나 nvcc의 독자적인 최적화는 프로그램 실행 동작에 부정적 영향을 줄 수 있습니다. 따라서 Roller는 kernel 분석기를 활용해 가장 성능이 좋은 rProgram을 빠르게 평가하고 최적의 것을 선택합니다. 더 큰 K는 일반적으로 kernel 품질을 향상시킬 수 있습니다. 상위 10, 20, 50개의 결과를 평가한 후, 우리의 경험상 상위 10개로 대부분의 경우 최적의 결과를 얻을 수 있다는 것을 보여줍니다. Roller의 kernel 분석기는 이전 컴파일러에서 머신러닝 알고리즘 기반의 평가 과정과는 다르다는 점에 유의하십시오. ML 기반 방법은 보통 수백, 심지어 수천 번의 순차 평가 단계가 필요하지만, ROLLER는 수십 개 후보를 병렬로 분석할 뿐입니다. 향후 우리는 어셈블리 수준 코드 생성을 구현하여 고수준 디바이스 컴파일러의 숨겨진 문제를 완화할 계획입니다.

이외에도 NVIDIA GPU/AMD ROCm/Graphcore IPU 등 구체적인 하드웨어에서의 일부 구현 세부 사항이 있는데, 관심 있는 분은 직접 논문을 살펴보시기 바랍니다.

## 5. 평가

여기서는 주로 cuda에서의 결과를 살펴보겠습니다.

![V100 GPU에서의 operator 성능](https://img-blog.csdnimg.cn/84beeea06c364b1aac75c8e9045886f0.png)

Figure 10은 우리 벤치마크의 119개 operator의 평균 kernel 성능을 operator 유형과 ID 순으로 정렬해 그린 것입니다. 큰 operator(예: kernel 시간이 5ms 초과)는 y축이 로그 스케일인 상단 서브플롯에 그렸고, 하단 4개 서브플롯은 그 외 중소형 operator입니다. 첫째, CUDA 라이브러리(CudaLib)와 비교했을 때, Roller는 81.5%의 operator에 대해 비슷한 성능(즉, 10% 이내)을 얻을 수 있고, 59.7%의 operator에서는 더 빠르기까지 합니다. Roller가 상대적으로 부진한 대부분의 operator는 3×3 이상의 필터를 가진 convolution operator로, cuDNN에서는 보통 더 효율적인 수치 알고리즘(예: Winograd [23])으로 구현되며, tensor 표현식으로 표현하기 어렵습니다. 이런 경우 Ansor와 TVM도 CudaLib보다 느린 이유입니다. 둘째, TVM과 Ansor와 비교했을 때, Roller는 각각 72.3%와 80.7%의 operator에 대해 비슷한 성능을 낼 수 있습니다. 나머지 27.7%와 19.3%는 주로 작은 operator이거나 tensor 형상이 불규칙하여 하드웨어와 정렬하기 어려운 경우입니다. 그러나 이러한 operator의 kernel 실행 시간은 일반적으로 상대적으로 짧으며, 예를 들어 평균 1.65ms와 1.16ms입니다. 모든 operator의 54.6%와 65.5%에서 Roller는 TVM과 Ansor보다도 더 빠른 kernel을 생성할 수 있습니다. 우리는 이러한 operator 대부분이 크고 시간이 많이 걸리는 것임을 관찰했습니다. 위 서브플롯에서 볼 수 있듯이, operator가 5ms를 초과(최고 343ms)할 때 Roller는 이러한 operator 대부분에서 더 좋은 성능을 달성할 수 있으며, 예를 들어 TVM과 Ansor 대비 평균 1.85배와 1.27배의 속도 향상을 보여줍니다.


아래 Figure 11은 operator 컴파일의 평균 시간도 비교합니다.

![여기에 이미지 삽입](https://img-blog.csdnimg.cn/85fe321e5f6347c7a0640d3885822953.png)
TVM과 Ansor에 비해 Roller의 operator 컴파일 시간은 수 초 이내로, TVM과 Ansor의 탐색 시간보다 2자릿수 더 빠릅니다.
![여기에 이미지 삽입](https://img-blog.csdnimg.cn/c9cfc332a503481a93a5fcfbfa219a26.png)

여기 Table은 몇 가지 대표적인 신경망의 성능과 컴파일 시간을 보여줍니다. **Roller가 TVM과 Ansor에 비해 비슷한 성능을 얻을 수 있으면서, 컴파일 시간을 수십 시간에서 수백 초로 단축할 수 있어 모델의 실제 생산 주기를 크게 향상시킬 수 있음을 알 수 있습니다**.


## 6. 정리 & 평가

컴파일 시간이 긴 문제를 해결하기 위해 본 논문은 Roller를 제안합니다. Roller의 핵심은 **rTile**로, 이는 새로운 tile 추상화이며, 하부 가속기의 핵심 특성과 일치하는 tensor shape을 캡슐화하여 shape 선택을 제한함으로써 효율적인 실행을 가능하게 합니다. Roller는 rTile 기반의 재귀적 구성 알고리즘을 채택해 목표 프로그램(rProgram)을 생성합니다. 최종적으로 Roller는 수 초 내에 효율적인 Kernel을 생성할 수 있으며, 현재 주류 가속기에서 다른 Tensor 컴파일러를 능가하는 성능을 보이고, IPU 같은 새로운 가속기에서 더 좋은 Kernel을 생성합니다.


