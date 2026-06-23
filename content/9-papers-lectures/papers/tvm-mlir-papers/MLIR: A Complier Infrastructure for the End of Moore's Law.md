# MLIR: A Complier Infrastructure for the End of Moore's Law

[GiantPandaCV 머리말] 이 글은 MLIR 논문 해설과 실습을 다룹니다. 여기서 실습이란 MLIR의 핵심 요점을 OneFlow Dialect에서 어떻게 구현했는지 대응시켜 보고, 각 요점의 구현 방법과 상호 관계를 설명한다는 의미입니다. MLIR 학습 과정의 중간 정리에 해당합니다. 본문은 크게 두 부분으로 나뉘는데, 1~6절이 MLIR 논문 리딩이고, 7절은 OneFlow Dialect를 통해 논문에서 언급한 MLIR 인프라스트럭처의 핵심 요소(Type, Attribute, Operation, Trait, Interfaces, Region, Block 등)를 설명합니다. 이 글은 어디까지나 입문용 안내라는 위치에서 더 많은 분들이 MLIR이라는 컴파일러 아키텍처를 이해하는 데 도움이 되고자 합니다. 도움이 되셨다면 제가 운영 중인 밑바닥부터 배우는 딥러닝 컴파일러 GitHub 저장소도 살펴봐 주세요: https://github.com/BBuf/tvm_mlir_learn.


## 0x0. 머리말
이전에는 MLIR의 Toy Tutorials를 출발점으로 MLIR을 살짝 둘러본 뒤, MLIR의 ODS, DRR 핵심 요소와 Interfaces 등을 정리했습니다. MLIR 관련 지식을 계속 학습·공유하기 전에 MLIR을 한 번 정리해 두고 싶었습니다. MLIR의 전체 모습을 파악하는 데에는 MLIR 논문을 읽는 것이 좋은 방법입니다. 이 글에서는 논문 리딩에 더해 MLIR Dialect를 구현할 때 사용하는 컴포넌트들을 마인드맵으로 그려 보고, OneFlow Dialect를 예로 들어 이 컴포넌트들이 어떻게 구현되어 있고 서로 어떤 관계를 갖는지 자세히 설명했습니다. MLIR이 익숙하지 않은 분들에게 도움과 영감을 줄 수 있기를 기대합니다. 입문 효과를 노린 글입니다.

이 글의 독해 순서는 대략 다음과 같습니다(숫자는 순서):

1. 제목
2. 초록
3. 서론
4. 결론
5. 관련 연구
6. MLIR 설계 관련
7. 코멘트
8. 참고 문헌

MLIR 논문 링크: https://arxiv.org/pdf/2002.11054.pdf

## 0x1. 제목

![MLIR 논문 제목](https://img-blog.csdnimg.cn/e0c2ba6013454dc49be0db5c3af4b6b3.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAanVzdF9zb3J0,size_20,color_FFFFFF,t_70,g_se,x_16)

논문 제목을 번역하면 MLIR: **무어의 법칙 종말 시대를 위한 컴파일러 인프라스트럭처** 입니다. 제목에서 알 수 있듯이 MLIR은 컴파일러 아키텍처이며, "무어의 법칙 종말"이라는 표현은 다소 직관적이지 않으니 뒤에서 차차 살펴보겠습니다. 또한 MLIR은 LLVM, Clang, Swift 프로젝트의 창시자인 Chris 선생이 주도하고 있다는 점도 확인할 수 있는데, 이는 MLIR 프로젝트의 품질에 큰 신뢰를 줍니다. 현재 MLIR 컴파일러 아키텍처가 인기를 끄는 이유 중 하나도 여기에 있다고 봅니다.

## 0x2. 초록
이 논문은 재사용 가능하고 확장 가능한 컴파일러 인프라스트럭처를 구축하는 새로운 방법으로 MLIR을 제안합니다. MLIR이 해결하고자 하는 것은 **소프트웨어 단편화(software fragmentation)**, **이종(heterogeneous) 하드웨어 컴파일 과정 개선**이며, **도메인 특화 컴파일러 구축 비용을 크게 낮추는 것**, 그리고 **기존 컴파일러들과 상호 연결**되도록 돕는 것입니다. MLIR은 또한 서로 다른 추상화 수준, 서로 다른 응용 도메인, 서로 다른 하드웨어 타깃과 실행 환경에서 code generators, translators, optimizers의 설계와 구현을 개선하는 데 도움을 줍니다. 기여는 다음과 같습니다. **(1) MLIR을 텍스트 기반 연구 산물로 보았을 때 가능한 확장과 진화를 논의하고, 이 새로운 방법이 설계, 의미론, 최적화 명세, 시스템, 엔지니어링 측면에서 가져오는 도전과 기회를 짚어낸다. (2) 컴파일러 구축 비용을 줄이는 범용 아키텍처로서 MLIR을 평가한다. 다양한 use case 설명을 통해, 본 연구가 향후 프로그래밍 언어, 컴파일러, 실행 환경, 컴퓨터 아키텍처 분야의 연구·교육에 어떤 기회를 제공할 수 있는지 보여준다.** 그리고 MLIR의 설계 원칙, 구조, 의미론을 소개합니다.

이 절은 주로 MLIR의 셀링 포인트를 이야기합니다. 즉, MLIR은 새로운 컴파일러 아키텍처이며, 소프트웨어 단편화 해결과 도메인 특화 컴파일러 구축 비용 절감을 핵심 목표로 합니다.

사실 오늘날 시점에서 보면 MLIR이 소프트웨어 단편화 문제를 완전히 해결하지는 못했습니다. 단지 소프트웨어 단편화 문제를 각 Dialect 사이의 단편화 문제로 옮긴 다음, 이 Dialect들이 같은 언어에 속해 혼용 가능하게 함으로써 **소프트웨어 단편화의 영향을 완화**시켰을 뿐입니다. 왜 "완화"이지 "완전 해결"이 아니냐 하면, 우선 제가 이해하는 소프트웨어 단편화는 N개의 프론트엔드 프레임워크(예: TensorFlow, PyTorch …)와 M개의 백엔드(GPU, CPU …) 사이의 적응 문제입니다. 중간 IR 표현이 없다면 적응 작업량은 $N * M$이 됩니다. Microsoft가 제안한 ONNX는 중간 IR 역할을 시도해서 이 $N * M$ 문제를 M으로 줄이려 했죠. 모든 프론트엔드 프레임워크가 ONNX로 변환되면, ONNX 백엔드만 적응시키면 되니까요. 하지만 이상과 현실은 다릅니다. ONNX는 다양한 프론트엔드 프레임워크에 맞추기 위해 더 범용적인 op 집합(opset)을 정의해 각 프레임워크의 op 의미를 매칭하려 했지만, 그 결과 프론트엔드 프레임워크와 ONNX를 상호 변환할 때 새로운 글루(glue) Op들이 끼어들어 IR이 더 복잡해지곤 합니다. 다시 MLIR 이야기로 돌아오면, 각 프론트엔드 프레임워크가 자신의 IR을 MLIR의 Dialect로 옮긴 뒤에도, 코드 생성이 가능한 LLVM IR에 도달하기까지 상당한 양의 DialectConversion을 거쳐야 합니다. 비록 각 Dialect가 혼용 가능해서 ONNX처럼 글루 Op이 추가로 생기는 상황은 피할 수 있지만, Dialect 혼용이 가능하다고 해서 DialectConversion이 매끄럽게 진행된다는 의미는 아닙니다. 가령 Dialect A의 Op X를 Dialect B의 Op로 변환하려는데, Dialect B에 X와 같은 의미의 Op이 없거나 의미가 살짝 다르다면, 우리는 Dialect B를 확장해 요구를 충족시켜야 합니다. 이는 ONNX가 끝없이 Opset을 늘려가는 방식과 본질적으로 다르지 않으며, MLIR의 Dialect 체인은 매우 길어질 수 있어 어떤 면에서는 ONNX보다 더 번거로울 수도 있습니다. 그래도 낙관적으로 보면, MLIR이 오픈소스화된 지 2~3년밖에 되지 않았으니 각 Dialect가 풍부해짐에 따라 이러한 단편화 위험은 점차 줄어들 것이라 기대합니다.

도메인 특화 컴파일러 구축 비용을 낮춘다는 것은, MLIR 생태계가 더 성숙해진 뒤에는 이론적으로 해당 하드웨어용 경계 Dialect 하나만 구현해 그 Dialect 안에 하드웨어의 Operation을 정의하면, 생태계의 기존 Dialect들을 골라 조합해 완전한 컴파일 흐름을 구성할 수 있게 된다는 뜻으로 이해됩니다.

## 0x3. 서론
컴파일러 설계는 코드 생성, 정적 분석, 프로그램 변환 등에 널리 알려진 알고리즘들이 다수 존재하는 성숙한 분야입니다. 컴파일러 설계 영역에서는 LLVM 컴파일러 인프라스트럭처[25], Java Virtual Machine(JVM)[26]과 같은 성숙한 기술 플랫폼들이 발전해 왔으며, 이들은 컴파일러 커뮤니티 전반에 걸쳐 대규모로 사용되고 있습니다. 이러한 인기 시스템들의 공통점은 "one size fits all" 방식, 즉 시스템과 인터페이스하는 추상화 수준이 단일하다는 점입니다. 예를 들어 LLVM IR은 대략 "C with vectors" 수준이고, JVM은 "garbage collector를 갖춘 객체지향 type 시스템(object-oriented type system with a garbage collector)" 추상화를 제공합니다. 이런 "one size fits all" 방식은 매우 가치 있습니다. C/C++나 Java 같은 소스 언어에서 이러한 추상화 영역으로의 매핑이 매우 직관적이기 때문이죠.

한편 어떤 문제들은 더 높거나 더 낮은 추상화 수준에서 모델링하는 편이 낫습니다. 예를 들어 LLVM IR 위에서 C++ 코드의 소스 수준 분석을 수행하기는 매우 어렵습니다. Swift, Rust, Julia, Fortran 같은 많은 언어들이 자체 IR을 개발해 자신만의 도메인 문제(예: 언어/라이브러리 관련 최적화, flow-sensitive type 검사(예: linear types), 최적화된 lowering 과정 구현)를 해결하고 있습니다. 마찬가지로 머신러닝 시스템들도 보통 "ML graphs"를 도메인 특화 추상화로 활용합니다.

도메인 특화 IR 개발 자체는 충분히 연구된 기법이지만, 엔지니어링과 구현 비용은 여전히 높습니다. 이러한 시스템 구현자들에게 인프라스트럭처 품질이 늘 우선순위가 되는 것도 아닙니다. 그 결과 컴파일러 시스템의 구현 품질이 떨어질 수 있고, 사용자가 흔히 마주하는 문제들 — 느린 컴파일 시간, 잘못된 구현, 빈약한 진단 메시지, 최적화된 코드의 디버깅 경험 저하 등 — 이 발생합니다.

MLIR 프로젝트는 새로운 추상화 수준을 손쉽게 정의·도입할 수 있게 하고, "in the box" 인프라스트럭처를 제공해 흔한 컴파일러 엔지니어링 문제들을 해결함으로써, 이러한 프로그래밍 언어 설계·구현상의 도전에 대응하는 것을 목표로 합니다. MLIR이 취하는 방식은 다음과 같습니다. **(1) static single assignment(SSA) 기반 IR 데이터 구조를 표준화한다. (2) IR dialect를 정의하기 위한 선언적 시스템을 제공한다. (3) 광범위한 범용 인프라(문서, parsing/printing 로직, 위치 추적, 멀티스레드 컴파일 지원, pass 관리 등)를 제공한다.**

논문은 MLIR 시스템의 다양한 설계 요점을 살펴보고, 저자들의 경험을 여러 문제에 적용하며, 이 작업이 프로그래밍 언어 설계와 교육에 미칠 수 있는 영향을 논의합니다.

논문의 기여는 다음과 같이 요약됩니다.

- 산업계와 학계에 중요한 활용 가치를 지닌 새로운 컴파일러 인프라스트럭처를 기술한다.
- 확장 가능하고 모듈화된 컴파일러 시스템을 구축하는 새로운 방법을 제안한다.
- 다양한 도메인에서 MLIR이 활용된 사례를 선정해 시스템의 범용성을 보인다.
- MLIR 인프라 위에서 컴파일 시스템을 개발한 경험을 공유한다.

이 절에서는 MLIR의 등장 배경도 언급합니다.

저자들은 우선 현대 머신러닝 프레임워크가 다양한 컴파일러, 그래프 기술, runtime 시스템들로 구성되어 있다는 점(Figure 1 참조)을 인식했습니다. 그러나 이들 부분은 공통 인프라나 설계 관점을 공유하지 않으며, 일부는 컴파일러 설계의 베스트 프랙티스를 따르지도 않습니다. 그 결과 사용자는 불완전한 오류 메시지, 경계 조건의 버그, 예측 불가능한 성능, 그리고 새 하드웨어를 지원하기 어려운 문제 등 여러 불편함을 겪습니다.

![Figure1](https://img-blog.csdnimg.cn/f6c830bbbc5e4be089582cb118c56018.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAanVzdF9zb3J0,size_20,color_FFFFFF,t_70,g_se,x_16)

저자들은 이내 컴파일러 산업 전체에 비슷한 문제가 있음을 깨달았습니다. LLVM과 같은 기존 컴파일러 시스템들은 다양한 언어 구현을 통합·통합하는 데에는 매우 성공적이었지만, 현대의 고수준 언어들은 보통 자체 고수준 IR을 구축하면서 동일한 고수준 추상화 기법을 거듭 재발명하고 있습니다(Figure 2 참조). 또한 LLVM 커뮤니티에서는 병렬 구조를 어떻게 가장 잘 표현할지, C 호출 규약이나 OpenMP 같은 언어 간 기능에 대한 공통 프론트엔드 lowering 인프라를 어떻게 공유할지 등 자주 논쟁이 있어 왔지만, 만족스러운 해법은 나오지 않았습니다.

![Figure2](https://img-blog.csdnimg.cn/383bbcabfb4c4a20b2d35d43417f617f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAanVzdF9zb3J0,size_20,color_FFFFFF,t_70,g_se,x_16)

이러한 도전에 직면한 저자들은 N개의 컴파일러를 각각 개선할 작업량을 감당할 수 없다고 판단해 보다 범용적인 해법을 만들기로 했습니다. 한 번 잘 만든 고품질 인프라에 노력을 쏟으면 여러 영역이 혜택을 보고, 기존 시스템을 점진적으로 업그레이드할 수 있으며, 전용 가속기의 이종 컴파일과 같은 시급한 문제도 보다 쉽게 다룰 수 있습니다. 이제 MLIR 기반 시스템을 구축·배포한 경험이 충분히 쌓인 만큼, MLIR 인프라의 원칙과 설계를 되돌아보고 왜 이 방향으로 가야 하는지 논의할 수 있습니다.

> 이 절은 관련 연구와 MLIR의 등장 배경을 정리해 MLIR의 혁신성과 기여를 강조합니다.

## 0x4. 결론

본 논문은 컴파일러를 구축하기 위한 유연하고 확장 가능한 인프라스트럭처로서 MLIR을 소개합니다. MLIR의 구체적인 설계를 기술하고, 여러 중요한 영역에서의 적용 가능성을 보여주며, 다양한 독창적인 연구·엔지니어링적 함의를 설명했습니다.

앞으로 컴파일러 커뮤니티(예: Clang C/C++ 컴파일러)와 다양한 도메인 전문가들이 보다 고수준의, 언어 특화된 IR로부터 어떤 이득을 얻을 수 있는지 보고 싶습니다. 또한 MLIR이 컴파일러와 IR 설계 기법을 가르치는 새로운 방법을 제공할 수 있을지 궁금하며, 이 인프라가 새 분야의 연구를 가속하는 모습을 보고 싶습니다.

여기서 향후 연구 방향이 다양하게 제시되니, 관심 있다면 직접 살펴보시길 권합니다. 저는 이 부분에 대한 배경 지식이 부족해 더 깊이 다루지 않겠습니다.

## 0x5. 관련 연구
MLIR은 다양한 분야에 걸친 프로젝트입니다. 인프라 자체는 새로운 시스템을 제공하지만, 인프라를 구성하는 각 컴포넌트는 관련 문헌에서 유사한 모듈을 찾아볼 수 있습니다.

MLIR은 LLVM[25]과 유사한 컴파일러 인프라스트럭처입니다. 다만 LLVM은 스칼라 최적화와 동종(homogeneous) 컴파일에서 강점이 있는 반면, MLIR은 텐서 대수와 알고리즘, 그래프 표현, 이종 컴파일 등 다양한 데이터 구조와 알고리즘을 1급(first-class) value와 Operation으로 모델링하는 것을 목표로 합니다. MLIR은 최적화의 mix-and-match를 지원해 컴파일 pass를 컴포넌트로 분해하고 lowering을 재정의할 수 있게 합니다. 이는 주로 패턴 재작성 인프라 덕분으로, 완전한 변환을 작은 지역 패턴들의 조합으로 포착하고, 단일 operation 단위에서 어떤 패턴을 적용해 재작성할지 제어합니다. 재작성 로직의 자동 확장, 형식화, 검증은 중요한 다음 단계가 될 것입니다[9, 27]. 백엔드에서 MLIR의 DDR은 LLVM의 instruction selection 인프라와 비슷하며, 다중 결과 패턴과 명세를 제약(constraint)으로 갖는 확장 가능한 operation을 지원합니다[49].

많은 프로그래밍 언어와 모델이 하드웨어 이종성을 다뤄 왔습니다. 동종 프로그래밍 모델 OpenMP는 StarSs, OpenACC[34, 31]와 같은 초기 제안에 기반해, 가속기로의 task offloading과 병렬 region[32]을 지원하도록 확장되었습니다. C++ AMP, HCC, SyCL은 전통적인 Clang/LLVM 흐름과 C++를 활용해 하드웨어 가속을 위한 고수준 추상화를 제공합니다[46]. 그러나 이들 모두 호스트 언어(보통 C++)의 기존 최적화에 의존해 추상화에 의한 손실을 줄이고, 고수준 구조체를 빠르게 runtime 실행 환경으로의 호출로 lowering 합니다. LLVM IR에 병렬 중간 표현을 더하는 확장은 일부 문제를 해결하지만, 전통적으로 동종 환경에 초점이 맞춰져 있었습니다[23, 42]. 지금까지 가장 야심 찬 작업은 아마도 Liquid Metal[3]일 것입니다. 도메인 특화 언어(DSL)와 함께, managed object의 의미를 정적, 벡터, 또는 reconfigurable 하드웨어로 변환하는 컴파일 흐름이 함께 설계되어 있습니다. 다만 Lime 컴파일러에서는 작업의 대부분이 round object를 square 하드웨어에 끼워 맞추는 일에 집중되어 있습니다(Kou and Palsberg [24]). MLIR은 확장 가능한 Operation/Type 집합을 통해, 이종 특성을 포함하는 고수준 언어를 직접 임베딩할 수단을 제공함과 동시에, 이러한 구조체를 점진적으로 lowering 하면서 서로 다른 타깃 간 공통 컴포넌트를 최대한 재사용할 수 있는 범용 인프라를 제공합니다.

언어 이종성 해결은 메타 프로그래밍 시스템, 특히 multi-stage 프로그래밍의 오랜 목표였습니다. Lightweight Modular Staging(LMS)[39]은 최신 기술 프레임워크이자 runtime code generator로, 효율적인 코드를 생성하고 DSL을 Scala에 임베딩할 수 있는 핵심 컴포넌트 라이브러리를 제공합니다. Delite[45]는 DSL 개발자의 생산성을 크게 향상시킨다고 주장하며, 병렬·이종 실행을 함께 지원합니다. 이런 접근은 MLIR과 상호 보완적이라고 볼 수 있습니다. DSL 임베딩을 위한 더 높은 수준의 추상화를 제공하고 범용 메타 프로그래밍 구조체를 통해 최적화를 구현하기 때문입니다.

언어 syntax 측면에서 더 나아가, ANTLR[33]은 새로운 컴파일러 프론트엔드 개발을 쉽게 해 주는 parser generator 계열입니다. MLIR은 현재 범용 parser 생성, AST 구성/모델링 기능이 없습니다. MLIR을 ANTLR 같은 시스템과 결합하면 사용자 입력에서 코드 생성까지 이어지는 재사용 가능한 컴파일러 라이브러리를 만들 수 있습니다.

XLA[57], Glow[40], TVM[11]은 머신러닝 분야에서 비슷한 이종 컴파일 목표를 다룹니다. 그러나 이들은 매우 구체적인 코드 생성 사례로, 그래프 추상화에서 출발해 가속기의 다차원 벡터 추상화를 타깃으로 합니다. 이들 모두 자체 코드 생성 전략은 그대로 사용하면서 MLIR을 기반 인프라로 활용해 MLIR의 범용 기능을 충분히 누릴 수 있습니다. 마찬가지로 Halide[36]와 TVM의 loop nest 메타 프로그래밍 기법, 더 이전의 loop nest 메타 프로그래밍 문헌[19, 41, 5, 14], 그리고 PolyMage[28], Tensor Comprehensions[52], Stripe[58], Diesel[16], Tiramisu[4]와 같은 완전 자동 흐름들과 그 기반의 polyhedral 컴파일 기법[17, 54, 8, 55]은 MLIR 기반 컴파일 프레임워크 안에서 서로 다른 코드 생성 경로로 공존할 수 있습니다. 직렬화·상호 운용 포맷에서는 ML 프론트엔드 다양성 문제를 푸는 다양한 방법이 있습니다. 예를 들어 ONNX[48]는 서로 다른 프레임워크들이 매핑할 수 있는 공통 op 집합을 제공하는 방식을 택했습니다. ONNX는 MLIR의 한 dialect 선택지가 될 수 있으며, 다른 op들은 이 dialect로 lowering 될 수 있습니다.

## 0x6. MLIR 설계 관련
### 0x6.1 설계 원칙
**내장된 것은 적게, 모든 것은 customizable(Little builtin, everything customizable)** MLIR 시스템은 최소한의 기본 개념 위에 세워지며, 대부분의 IR이 완전히 customizable 합니다. 설계 시에는 IR에서 가장 일반적인 적은 수의 추상(type, operation, attribute) 만으로 나머지 모든 것을 표현하도록 합니다. 그래야 추상화가 더 적고 일관되며, 이해·확장·사용도 쉬워집니다. 넓게 보면 customizability는 컴파일 시스템이 변하는 요구에 적응할 수 있게 하고, 미래의 문제에도 대응할 가능성을 높여줍니다. 이런 의미에서 IR은, 그 위 중간 언어들의 syntax와 semantics를 지원하고, 재사용 가능한 컴포넌트와 프로그래밍 추상을 갖춘 인프라스트럭처로 구성되어야 합니다. **customization** 의 성공 기준은 머신러닝 그래프, AST, 수학적 추상(예: polyhedra), 제어 흐름 그래프(CFG), instruction-level IR(예: LLVM IR) 등 다양한 추상을 표현할 수 있고, 이러한 추상들에서 컴파일 시스템으로 옮길 때 어떤 하드코딩된 개념도 사용하지 않아도 된다는 것입니다. **물론 호환성이 충분치 않을 경우 customizability는 내부 단편화의 위험을 가져옵니다.** 생태계 단편화 문제에 순수 기술적 해결책은 없겠지만, 시스템은 재사용 가능한 추상의 설계를 장려해야 하며, 이러한 추상이 설계자의 예상 범위를 넘어선 곳에서도 사용될 수 있다는 가정을 깔고 있어야 합니다.

**SSA and regions** Static Single Assignment 형태[15]는 컴파일러 IR에서 널리 쓰이는 표현입니다. 데이터 흐름 분석을 단순하고 sparse 하게 만들고, continuation-passing 스타일과의 관계 덕분에 컴파일러 커뮤니티에서 폭넓게 이해되며, 주요 프레임워크들에서 사용되고 있습니다. 많은 기존 IR이 평탄한 선형 CFG를 사용하지만, 더 고수준의 추상을 표현하기 위해서는 nested region을 IR의 1급 개념으로 둘 필요가 있습니다. 이는 전통적인 region 형태를 넘어서 추상화 수준을 끌어올리며(예: loop tree), 컴파일 과정, instruction extraction, SIMD 병렬성 등을 가속합니다[22, 21, 37]. 이종 컴파일을 지원하려면 시스템은 구조화된 제어 흐름, 동시성 구조, 소스 언어의 closure 등을 지원해야 합니다. 한 가지 구체적 도전은 nested region 위에서 CFG 기반 분석·변환을 어떻게 구성할 것인가입니다.

이를 위해 LLVM의 normalization, 때로는 canonicalization 속성을 일부 포기합니다. 다양한 데이터·제어 구조를 더 작은 normalize된 표현 집합으로 lowering 할 수 있는 능력은 컴파일러의 복잡도를 통제하는 데 매우 중요합니다. pre-header, header, latch, body를 갖는 canonical loop 구조는 프론트엔드 언어의 다양한 loop 구조를 선형화된 제어 흐름으로 표현하는 전형적 사례입니다. MLIR은 사용자에게 선택권을 제공합니다. 즉 **컴파일 흐름 내 pass의 알고리즘에 따라, nested loop을 nested region으로 잡을 수도 있고 선형화된 제어 흐름으로 잡을 수도 있습니다**. 이런 선택권을 제공함으로써 MLIR은 LLVM의 normalization-only 방향에서 벗어나면서도, 필요할 때 더 고수준의 추상을 다룰 수 있는 능력을 유지합니다. 이는 추상의 normalization을 어떻게 통제할 것인가라는 새로운 질문으로 이어지는데, 이는 다음 단락의 주제입니다.


**Progressive lowering(점진적 lowering)** 컴파일 시스템은 점진적인 lowering을 지원해야 합니다. 즉 작은 단계로 여러 추상화 수준을 순차적으로 거치며 고수준 표현에서 최저 수준 표현까지 단계적으로 내려가는 것입니다. 다층 추상화가 필요한 이유는, 범용 컴파일러 인프라가 다양한 플랫폼과 프로그래밍 모델을 지원해야 하기 때문입니다. 기존 컴파일러들은 자신들의 파이프라인에 여러 고정 추상화 수준을 도입해 왔습니다. Open64 WHIRL[30]은 5단계 표현을 갖고, Clang/LLVM 컴파일러는 AST에서 LLVM IR, SelectionDAG, MachineInstr, MCInst까지 lowering 합니다. 이러한 lowering 구현은 다소 경직되어 있어, 추상화 수준의 확장성을 지원하려면 더 유연한 설계가 필요합니다. 이는 변환의 phase ordering에 큰 영향을 미칩니다. 컴파일러 전문가들이 점점 더 많은 변환 pass를 구현하면서 pass들 사이에 복잡한 상호작용이 나타나기 시작합니다. 실제로 최적화 pass들을 결합해 실행하면 컴파일러가 더 많은 유용한 프로그램 정보를 발견할 수 있다는 것이 알려져 있습니다. constant propagation, value numbering, dead code elimination을 결합한 시도[13]가 그 좋은 예입니다. 일반적으로 컴파일러 pass는 대략 네 가지 역할로 나뉩니다. (1) 최적화 변환 (2) enabling 변환 (3) lowering (4) cleanup. 컴파일 시스템은 이러한 역할들을 단일 operation 단위에서 mix-and-match 할 수 있게 해야 하며, 전체 컴파일 단위 위에서 pass들을 순차 실행하는 방식에 머물러서는 안 됩니다.

**고수준 semantics 유지(Maintain higher-level semantics)** 시스템은 분석이나 최적화 성능에 필요한 고수준 의미와 계산 구조를 유지해야 합니다. 일단 의미를 낮춰버린 뒤 다시 끌어올리려 하면 잘 되기 어렵고, 그런 정보를 저수준 IR 환경에 억지로 끼워 넣는 방식은 보통 파괴적입니다(예: debug 정보를 사용해 구조를 기록하는 경우, 모든 pass가 이를 검증/재방문해야 합니다). 대신 시스템은 계산 구조를 보존하면서 점진적으로 하드웨어 추상으로 lowering 해야 합니다. 이때 의식적으로 구조 정보를 버릴 수 있는데, 이는 기반 실행 모델에 매칭하기 위해 더 이상 그 구조가 필요 없는 시점에만 일어나야 합니다. **예를 들어, 시스템은 관련된 모든 변환 동안 loop 구조와 같은 구조화된 제어 흐름을 유지해야 합니다. 이 구조를 제거한다는 것, 즉 CFG 기반 제어 흐름으로 가는 것은 본질적으로 더 이상 이 수준에서 어떤 변환도 수행하지 않는다는 의미입니다.** 컴파일러 개발에서 병렬 계산 구조를 모델링하는 최신 기술은 이 일이 얼마나 어려운지를 잘 보여줍니다[23, 42].

컴파일 시스템의 IR 일부는 더 고수준 추상을 유지하고 다른 일부는 IR 수준이 낮춰진 상태가 되도록 허용하려면, 동일 IR 안에서 서로 다른 수준의 추상과 서로 다른 개념이 혼합 가능해야 한다는 것이 시스템의 핵심 속성이 됩니다. 예를 들어, custom 가속기용 컴파일러는 시스템이 제공하는 고수준 구조와 추상을 IR에서 재사용하면서, 동시에 그 IR로 가속기 고유의 기본 스칼라/벡터 instruction도 표현할 수 있습니다.

**IR validation(IR 검증)** 생태계의 개방성은 폭넓은 검증 메커니즘을 요구합니다. 검증과 테스트는 컴파일러 버그를 잡는 데 유용할 뿐 아니라, 확장 가능한 시스템에서는 검증 방법론과 도구의 견고성에 대한 요구가 점점 커집니다. 검증 메커니즘은 정의가 간결하고 실용적이어야 하며, 정확한 결과의 단일 출처(source of truth)로 사용될 수 있어야 합니다. 장기 목표는 변환 검증[35, 29, 50, 51]과 현대 컴파일러 테스트 방법론[12]의 성공 사례를 재현하는 것입니다. 확장 가능한 컴파일러 생태계에서 검증과 테스트는 모두 아직 풀어야 할 두 과제입니다.

**Declarative rewrite patterns(선언적 재작성 패턴)** 표현 변형(modifier)을 정의하는 일은 새로운 추상을 정의하는 것만큼 간단해야 합니다. 일반적인 변환은 선언적으로 표현된 재작성 규칙으로 구현되어야 하며, 머신이 분석 가능한 형식으로 복잡도, 완전성 등 재작성의 속성을 추론할 수 있어야 합니다. 재작성 시스템은 견고하고 효율적이어서 폭넓게 연구되어 왔으며, type system부터 instruction selection까지 다양한 컴파일 문제에 적용되어 왔습니다. MLIR의 목표는 전례 없는 확장성과 점진적 lowering 능력을 달성하는 것이며, 프로그램 변환을 재작성 시스템으로 모델링할 수 있는 다양한 길을 열어 둡니다. 또한 재작성 규칙과 전략을 어떻게 표현할지, 다중 추상화 수준에 걸쳐 재작성 전략을 안내할 수 있는 머신 기술(description)을 어떻게 구성할지에 관한 흥미로운 질문도 제기합니다. 시스템은 이러한 문제를 풀면서도 확장성을 유지하고 합리적이고 단조적이며 재현 가능한 동작을 수행해야 합니다.

**소스 위치 추적과 traceability(Source location tracking and traceability)** Operation의 출처(원래 위치, 적용된 변환)는 시스템 내에서 손쉽게 추적 가능해야 합니다. 이는 복잡한 컴파일 시스템에서 흔히 발생하는 투명성 부족 문제를 해결하기 위함입니다. 복잡한 컴파일 시스템에서는 최종 표현이 원래 표현으로부터 어떻게 만들어졌는지 그 전 과정을 이해하기 어렵습니다. 컴파일 안정성이 중요한 민감 응용에서는 lowering과 최적화 단계 추적이 소프트웨어 인증 프로그램의 중요한 부분이 됩니다[43]. 보안 코드(예: 암호 프로토콜, 프라이버시에 민감한 데이터 처리 알고리즘) 위에서 동작할 때 컴파일러는 종종 중복되거나 번잡해 보이는 계산을 만나는데, 이런 계산에는 소스 프로그램의 기능적 의미만으로는 완전히 포착되지 않는 보안·프라이버시 속성이 묻어 있어, 사이드 채널 노출 방지나 네트워크/장애 공격 강화를 위해 존재합니다. 최적화는 이런 보호 장치를 변경하거나 완전히 무효화할 수 있습니다[56]. 이런 투명성 부족은 보안 컴파일에서 WYSINWYX[6]라고 불립니다. 고수준 정보를 저수준에 정확히 전파하는 간접 목표 중 하나는 안전하고 추적 가능한 컴파일 과정을 돕는 것입니다.

> 이 절은 MLIR이 가진 거시적 특징을 설명합니다. MLIR은 다층 IR 구조를 가진 컴파일러 아키텍처이며, 실제로는 다층 Dialect 구조로서 각 Dialect가 서로 다른 수준의 개념을 모델링합니다. 예를 들어 LLVM Dialect는 시스템 수준 변환을 담당하고, Linalg, Tensor, Vector 등 Dialect는 코드 생성을 협업해 수행하며, Affine, Math 등 Dialect는 저수준 계산을 기술합니다.

### 0x6.2 IR 설계 세부
이 절에서는 앞 절에서 설명한 원칙에 따라 MLIR의 IR 설계를 소개합니다.

#### Operations(연산)
MLIR의 의미 단위는 "operation"이며, 줄여서 Op라 부릅니다. MLIR 시스템에서는 instruction부터 function, module까지 모든 것이 Op으로 모델링됩니다. MLIR은 고정된 Op 집합이 없으며, 사용자 정의 Op 확장을 허용하고 장려합니다. 컴파일러 pass는 알 수 없는 Op에 보수적으로 대응하며, MLIR은 trait, 특수한 Operation hook, Interfaces 등을 통해 pass에게 Op의 의미를 기술할 수 있도록 지원합니다.

Op(Figure 3 참조)는 고유한 opcode를 갖습니다. 문자열로 표기되는 opcode는 자신이 속한 dialect와 operation을 식별합니다. Op은 0개 이상의 value를 operand와 result로 가질 수 있으며, operand와 result는 SSA 형태로 유지됩니다. 모든 value는 LLVM IR과 비슷하게 type을 갖습니다. opcode, operand, result 외에도 Op은 attribute, region, block argument, 위치 정보(**Attributes, Regions, Block Arguments, and Location Information**)를 가질 수 있습니다. Figure 4는 value와 Op을 보여 줍니다. `%` 식별자는 named value(번들)이며, 번들에 다수의 value가 있다면 `:` 뒤에 번들 안 value 개수를 지정합니다(예: Figure 3의 `%results:2`는 반환값이 2개임을 의미). `#`는 특정 value를 의미합니다. 일반 텍스트 표현에서 operation 이름은 따옴표로 감싼 문자열이며, 그 뒤에 괄호로 감싼 operand가 따라옵니다.

![Figure3](https://img-blog.csdnimg.cn/423c326c320d4984ba55f20ae51eb808.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAanVzdF9zb3J0,size_20,color_FFFFFF,t_70,g_se,x_16)

![Figure4](https://img-blog.csdnimg.cn/d21d5f77096740a6bd9c7beb705bde3e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAanVzdF9zb3J0,size_20,color_FFFFFF,t_70,g_se,x_16)

#### Attributes(속성)
MLIR의 attribute는 컴파일 타임 정적 구조 정보로, 정수 상수 값, 문자열 데이터, 상수 부동소수점 값 리스트 등을 담을 수 있습니다. attribute는 type을 가지며, 각 Op 인스턴스는 문자열 이름에서 attribute value로의 열린(open) key-value dictionary 매핑을 갖습니다. 일반 syntax에서 attribute는 **Op operand와 그 type 사이**에 위치하며, key-value 쌍들은 쉼표로 구분되고 전체가 중괄호로 감싸입니다(예: Figure 3의 `{attribute="value" : !d.type}`이나 Figure 4의 `{lower_bound = () -> (0), step = 1 : index, upper_bound = #map3}`). 여기서 `lower_bound`, `step`, `upper_bound`는 attribute 이름입니다. `() -> (0)`은 inline affine form 표기이며, 이 예시에서는 상수 0을 만드는 affine 함수입니다. `#map3`은 attribute alias 표기로, attribute value에 라벨을 미리 연결해 두면 attribute value가 필요한 어디서든 라벨을 사용할 수 있게 해 줍니다. opcode와 마찬가지로 MLIR은 고정된 attribute 집합을 두지 않습니다. attribute의 의미는 Op의 의미나 attribute가 속한 dialect에서 도출됩니다. attribute 또한 확장 가능하며 외부 데이터 구조를 직접 참조할 수 있도록 허용해 기존 시스템과의 통합에 유용합니다. 예를 들어 어떤 attribute는 ML 시스템에서 컴파일 타임에 알려진 데이터 저장소의 내용을 참조할 수 있습니다.

#### Location information(위치 정보)
MLIR은 위치 정보의 컴팩트한 표현 형식을 제공하며, 시스템 전반에서 위치 정보를 처리·전파하도록 권장합니다. 위치 정보는 Op을 만들어낸 소스 프로그램의 stack trace를 보존해 디버그 정보 생성에 사용할 수 있습니다. 위치 정보는 컴파일러가 진단 메시지를 생성하는 방식을 표준화하고 다양한 테스트 도구에서도 사용할 수 있게 합니다. 위치 정보 또한 확장 가능해, 컴파일러가 기존 위치 추적 시스템, 고수준 AST 노드, LLVM 스타일의 file-line-column 주소, DWARF 디버그 정보, 또는 그 외 고품질 컴파일 구현에 필요한 정보를 참조할 수 있도록 합니다.

**위 세 가지 핵심은 Toy 언어의 transpose Op을 통해 더 깊이 이해할 수 있습니다.**

```cpp
%t_tensor = "toy.transpose"(%tensor) {inplace = true} : (tensor<2x3xf64>) -> tensor<3x2xf64> loc("example/file/path":12:1)
```

구조 분해 설명:
- `%t_tensor`: 이 Operation이 정의하는 result의 이름. 앞의 `%`는 충돌을 피하기 위함. https://mlir.llvm.org/docs/LangRef/#identifiers-and-keywords 참조. 한 Operation은 0개 또는 그 이상의 result를 정의할 수 있고(Toy 언어에서는 단일 result만 존재), 이는 SSA value입니다. 이 이름은 parsing 중에만 사용되며 영속적이지 않습니다(예: SSA value의 메모리 표현에 추적되지 않음).
- `"toy.transpose"`: Operation의 이름. 고유한 문자열이며 Dialect의 namespace를 "."로 접두 표기합니다. Toy Dialect의 transpose Operation으로 이해할 수 있습니다.
- `(%tensor)`: 0개 이상의 입력 operand(인자) 리스트. 다른 operation이 정의한 SSA value 또는 block argument에 대한 참조입니다.
- `{ inplace = true }`: 0개 이상의 attribute로 이루어진 dictionary. 이 attribute들은 항상 상수인 특수 operand입니다. 여기서는 `inplace`라는 boolean attribute를 정의했고 상수 값은 true입니다.
- `(tensor<2x3xf64>) -> tensor<3x2xf64>`: 함수 형태로 표현한 operation type으로, 앞이 입력, 뒤가 출력입니다. `<2x3xf64>` 안의 내용은 텐서의 형상 `2x3`과 텐서가 저장하는 데이터 type `f64`를 기술하며, 둘은 `x`로 연결됩니다.
- `loc("example/file/path":12:1)`: 이 operation의 소스 코드 상 위치.


#### Regions and Blocks(영역과 블록)

Op 인스턴스에는 여러 개의 region이 부착될 수 있습니다. region은 MLIR에서 nested 구조의 구현 메커니즘을 제공합니다. 한 region은 일련의 block을 포함하고, 한 block은 일련의 operation을 포함합니다(operation 안에 다시 region이 들어갈 수도 있으며, Figure 3 참조). attribute와 마찬가지로 region의 의미는 그것이 부착된 operation에 의해 정의됩니다. 다만 region 내부의 block들(여러 개라면)은 control flow graph(CFG)를 형성할 수 있습니다. 예를 들어 Figure 4의 `affine.for` operation은 loop이며, `({` 와 `})` 구분자 사이의 block 하나가 region입니다. Op은 region 사이의 제어 흐름을 지정합니다. 이 예시에서는 loop 상한에 도달할 때까지 본문이 반복 수행됩니다. 각 region의 body는 block들의 시퀀스이며, 각 block은 terminator(`terminator`) operation으로 끝나는데, terminator는 후속(successor) block을 가질 수 있어 제어 흐름이 후속 block으로 이동할 수 있습니다. 각 terminator(예: "switch", "conditional branch", "unwind")는 자신만의 의미를 정의합니다. terminator는 같은 region 내 다른 block으로 제어 흐름을 옮기거나, region을 포함한 Op으로 반환할 수 있습니다. 후속 block들의 그래프가 CFG를 정의해, region 내부에서 표준 SSA 기반 제어 흐름이 가능해집니다. MLIR은 $\phi$ 노드를 사용하지 않고 SSA의 functional form을 사용합니다. 즉 terminator가 후속 block에서 정의된 block argument로 value를 전달합니다. 각 block은 (비어 있을 수도 있는) typed block argument 리스트를 가지며, 이 argument들은 일반 value이고 SSA에 부합합니다. terminator Op의 의미는 제어가 이전된 후 block argument들이 어떤 value를 갖게 될지 정의합니다. region의 첫 번째(entry) block의 경우 value는 그 region을 포함하는 Op의 의미에 의해 정의됩니다. 예를 들어 `affine.for`는 entry block argument `%arg4`를 loop 유도(induction) 변수로 사용합니다.

> 여기서 표현하려는 바는, 한 Operation이 여러 Region을 가질 수 있고, Region은 다시 일련의 Block으로 구성되며, Block은 다시 일련의 Op을 포함한다는 것입니다. 이렇게 nested 관계를 형성해 scope와 제어 흐름 관계를 표현할 수 있습니다.

#### Value dominance and visibility
Op은 scope 내에 있는 value만 사용할 수 있습니다. 즉 SSA dominance, nesting, 포함 Operation의 의미상 가시성이 보장된 value만 사용 가능합니다. value가 표준 SSA dominance 관계를 따른다면 CFG에서 그 value들이 보이며, 사용 전에 정의가 반드시 거쳐 가도록 보장됩니다.

region 기반 가시성은 단순한 region nesting에 따라 정의됩니다. Op의 operand가 현재 region 바깥에 있다면, 그 사용이 일어나는 region 위쪽에서 외부 lexical 방식으로 정의되어 있어야 합니다. 이를 통해 `affine.for` operation 내부의 Op이 외부 scope에서 정의된 value를 사용할 수 있습니다.

MLIR은 또한 **isolated-from-above**로 operation을 정의할 수 있도록 허용합니다. 이는 그 operation이 **scope barrier(scope barrier)** 임을 나타냅니다. 예를 들어 "std.func" Op은 함수를 정의하며, 그 함수 내부의 operation들은 함수 바깥에서 정의된 value를 참조할 수 없습니다. 유용한 의미 검사를 제공하는 것 외에도, isolation barrier를 가로지르는 use-def 체인이 존재하지 않기 때문에 isolated-from-above인 Op을 포함하는 Module은 ML 컴파일러가 병렬로 처리할 수 있습니다. 이는 멀티코어 머신에서의 컴파일 활용에 매우 중요합니다.


#### Symbols and symbol tables
Op에는 또한 symbol table이 부착될 수 있습니다. symbol table은 이름(문자열로 표현)을 IR 객체(symbol이라 부름)와 연관 짓는 표준 방식입니다. IR은 symbol의 용도를 규정하지 않고, Op이 정의하도록 맡깁니다. SSA 규칙을 따를 필요가 없는 named entity에 symbol은 유용합니다. symbol은 동일 표 안에서 중복 정의할 수 없지만, 정의 이전에 사용될 수 있습니다. 예를 들어 전역 변수, 함수, 또는 named module은 symbol로 표현할 수 있습니다. 이러한 메커니즘이 없으면 재귀 함수(자신을 정의 안에서 참조)를 정의할 수 없습니다. symbol table이 부착된 Op의 region이 비슷한 Op들을 포함한다면 symbol table은 nested될 수 있습니다. MLIR은 nested symbol을 포함해 Op 안의 symbol을 참조하는 메커니즘을 제공합니다.


#### Dialects
MLIR은 Dialect를 통해 확장성을 관리합니다. Dialect는 고유한 namespace 아래에 Ops, attribute, type을 논리적으로 묶어 제공합니다. Dialect 자체는 어떤 새로운 의미도 도입하지 않으며, 논리적 그룹화 메커니즘 역할을 하고, dialect 공통 op 지원(예: dialect 내 모든 op의 constant folding 동작)을 제공할 수 있습니다. Dialect namespace는 opcode 안에서 "."로 구분된 prefix입니다. 예를 들어 Figure 4가 사용하는 `affine`, `std` dialect가 있습니다.

개념적으로 Op, type, attribute는 Dialect로 추상화할 수 있는데, 이는 모듈식 라이브러리 집합을 설계하는 것과 비슷합니다. 예를 들어 어떤 Dialect는 하드웨어 vector 연산용 Op과 type(예: shuffle, insert/extract element, mask 등)을 담을 수 있고, 다른 Dialect는 대수적 vector 연산용 Op과 type(예: 절댓값, dot product 등)을 담을 수 있습니다. 두 dialect가 동일한 vector type을 사용하는지, 그리고 그 type이 어느 dialect 소속인지는 MLIR 사용자가 설계 시점에 결정할 수 있습니다.

모든 Op, type, attribute를 단일 dialect에 두는 것도 가능하지만, 이는 곧 다수의 개념과 이름 충돌 등으로 dialect를 관리하기 어렵게 만들 것입니다. 각 Op, type, attribute는 정확히 하나의 dialect에 속하지만, MLIR은 점진적 lowering을 가능케 하기 위해 여러 dialect의 혼합을 명시적으로 지원합니다. 서로 다른 dialect의 Op이 IR 어느 수준에서든 공존할 수 있고, 서로 다른 dialect에서 정의된 type을 사용할 수 있습니다. Dialect 혼합은 재사용성, 확장성, 유연성을 강화합니다.

#### type 시스템
MLIR의 모든 value는 type을 가지며, 이 type은 그 value를 만들어낸 Op이나 그 value를 argument로 정의하는 Block에서 지정됩니다. type은 IR에 컴파일 타임 의미를 부여합니다. MLIR의 type 시스템은 사용자 확장 가능하며, 외부 type 시스템(예: llvm::Type, clang::Type)을 참조할 수도 있습니다. MLIR은 엄격한 type equality 검사를 적용하며, type 변환 규칙은 제공하지 않습니다. Op은 함수 꼬리(tail) 형태와 비슷한 syntax로 입력과 result type을 나열합니다. Figure 4의 `affine.load`는 memref와 인덱스 type에서 로드된 value type으로 매핑됩니다. type 이론 관점에서 MLIR은 non-dependent type만 지원합니다. trivial type, parametric type, function type, sum/product type을 포함합니다.

#### 표준 type
또한 MLIR은 임의 정밀도 정수, 표준 부동소수점 type, 그리고 tuple, 다차원 vector, tensor 같은 단순한 범용 컨테이너 등 자주 쓰이는 표준화된 type 집합을 제공합니다. 이러한 type들은 Dialect 개발자의 편의를 위한 것이며 반드시 사용해야 하는 것은 아닙니다.

#### Functions and modules(함수와 모듈)
일반 IR과 마찬가지로 MLIR도 보통 함수와 모듈로 구성되며, 이는 MLIR이 새로 도입한 개념은 아닙니다. 함수와 모듈은 builtin dialect 안에서 Op으로 구현되어 있습니다. 모듈은 단일 region을 가지는 Op이며, 이 region은 단일 block을 포함합니다. 모듈은 제어 흐름을 옮기지 않는 dummy Op으로 종료됩니다.

모듈은 참조 가능한 symbol을 정의합니다. 다른 block과 마찬가지로, 그 본문은 일련의 Op을 포함하는데 이는 함수, 전역 변수, 컴파일러 메타데이터 또는 그 외 최상위 구조일 수 있습니다. 함수는 단일 region을 가지는 Op이며, 그 argument는 함수 매개변수에 대응합니다.

**함수는 이름으로 참조 가능한 symbol을 정의합니다. 함수 호출 Op을 사용하면 제어 흐름이 함수 내부로 이동합니다.** 일단 내부에 들어가면 제어 흐름은 region 내 각 block의 CFG를 따릅니다. "return" terminator는 후속이 없으며, region 실행을 종료해 제어 흐름을 함수의 호출자에게 돌려보냅니다. "return" terminator Op의 operand는 모두 함수의 반환값이 됩니다.

> 위에서 MLIR의 IR 설계 세부를 소개했습니다. MLIR 공식 문서의 syntax 규정과 함께 보면 더 친숙해질 수 있습니다: https://mlir.llvm.org/docs/LangRef/.


### 0x6.3 IR 인프라스트럭처
IR 자체 외에도 MLIR은 dialect, Op, pattern rewrite, 검증, 재사용 가능한 pass 같은 IR 요소를 정의하기 위한 인프라스트럭처를 제공합니다. 새로운 추상을 정의하고 MLIR을 최적화 toolkit으로 사용할 때 MLIR의 인프라스트럭처는 확장성과 사용 편의성을 보장하는 데 핵심적입니다.

#### 0x6.3.1 Operation description(operation 기술)
MLIR은 TableGen[47] 명세를 사용해 operation description(Operation Descriptions, ODS)을 정의하며, 선언적 방식으로 Op의 구조와 그 verifier 컴포넌트를 기술합니다. TableGen은 LLVM에서 널리 사용되는 데이터 모델링 도구로, 도메인 특화 정보를 기록·유지하는 데 도움을 줍니다. ODS는 TableGen 언어에 임베딩되어 MLIR Op을 정의하는 DSL이라 볼 수 있습니다. 따라서 ODS의 syntax는 TableGen이 규정하지만, MLIR 고유의 의미는 ODS가 규정합니다. ODS 정의는 결국 C++ 코드로 변환되며, 컴파일 시스템의 나머지 부분과 상호 운용 가능합니다.

MLIR은 TableGen Op 클래스를 사용해 ODS에서 Op을 모델링합니다. Figure 5는 ODS로 정의된 Op 예시를 보여줍니다. 각 Op 정의는 고유 식별자가 되는 이름을 갖습니다. Op의 trait 리스트는 Op의 속성을 기술합니다. Op의 argument 리스트는 Op의 operand와 attribute를 지정합니다. Op 정의에는 result 리스트도 포함됩니다. Op의 argument와 result는 이름과 type 제약(예: float 또는 int32 형태의 고정 형상 텐서)을 갖습니다. Op 정의는 사람이 읽을 수 있는 Op 설명도 지정할 수 있습니다. ODS가 제공하는 것보다 더 정밀한 제어가 필요할 때는 builder, printer, parser, verifier 구문으로 추가 C++ 코드를 주입할 수 있습니다. Op trait는 "has no side-effects"처럼 일반적일 수도 있고, "has custom exporter"처럼 Dialect 또는 ODS에 특화된 것일 수도 있습니다. ODS의 trait는 trait 동작을 정의하는 C++ 클래스에 의해 지원될 수 있습니다. MLIR에는 고정된 trait 집합이 없지만, 일부 trait나 optimizer(논문 6.1절 참조)는 ODS에 알려진 것이 있습니다(예: "shape result and operand type"은 입력 type만으로 출력 type이 완전히 결정되는 제약을 의미).

type 제약은 argument/result type의 속성을 검사하며, 사용자/dialect가 확장할 수 있습니다. MLIR 인프라는 "any type", "tensor with element satisfying the given constraint", "vector of given rank" 등 여러 사전 정의된 type 제약도 제공합니다. ODS는 trait가 가져오는 제약을 사용하는 operand로부터 result type을 자동 추론하는 기능을 제한적으로 지원합니다. 자세한 내용은 다음 절(논문 4.2절)을 참조하세요.



![Op의 ODS 정의](https://img-blog.csdnimg.cn/473a1be4b1c943dfbbbfca4b5424dabd.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAanVzdF9zb3J0,size_20,color_FFFFFF,t_70,g_se,x_16)


#### 0x6.3.2 Declarative rewrites(선언적 재작성)
많은 MLIR 변환은 Op 조작을 수반하며, 일부 변환은 IR에 복잡한 수정이 필요하지만 다른 많은 변환은 SSA use-def 관계로 정의된 DAG에 대한 단순한 재작성으로 표현할 수 있습니다. MLIR은 그래프 재작성 프레임워크를 제공하고, 선언적 재작성 규칙(Declarative Rewrite Rule, DRR) 시스템을 함께 제공해 pattern 표현을 간단하게 만듭니다.

ODS와 마찬가지로 DRR도 TableGen 언어에 임베딩된 DSL입니다. DRR은 source/target DAG pattern과 제약(동적 제약 포함[49])을 표현하며, pattern 우선순위에서 이득을 봅니다. pattern은 Op의 argument를 capture하고 재사용할 수 있습니다. 개념적으로 DRR은 특정 제약 하의 DAG 등가성을 표현합니다. Figure 6은 DRR pattern의 예를 보여 주며, Figure 5에서 정의된 Op을 `compare`와 `select`로 구성된 범용 저수준 구현으로 변환합니다.

![DRR 그래프 재작성 규칙](https://img-blog.csdnimg.cn/219caae400f74d4cb869b5938f31b906.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAanVzdF9zb3J0,size_20,color_FFFFFF,t_70,g_se,x_16)

DRR은 C++ 코드로 변환되며, 범용 그래프 재작성 프레임워크를 통해 C++로 직접 정의된 더 복잡한 pattern과 혼합 사용할 수 있습니다. 이 기능 덕분에 MLIR은 흔한 use case는 간결하게 유지하면서도 프레임워크의 범용성을 제한하지 않습니다.

#### 0x6.3.3 Pass Manager
MLIR pass manager는 다양한 단위에서 IR pass 시퀀스를 조직·처리하며 pass의 효율적 실행을 보장합니다. 기존 컴파일 시스템의 pass 관리는 보통 고정된 단위(module, function, loop pass manager 등)로 정의됩니다. 그러나 MLIR에서는 module과 function이 특별한 존재가 아니며, region을 가지는 Op일 뿐이고 그 변형도 다양합니다. **따라서 MLIR pass manager 또한 고정된 Op 집합을 대상으로 하지 않고, 임의 nested 수준의 임의 Op을 대상으로 합니다.**

**병렬 컴파일** MLIR의 중요한 요구사항 중 하나는 멀티코어 머신을 활용해 컴파일을 가속하는 것입니다. pass manager는 IR의 동시 순회와 수정(concurrent traversal and modification)을 지원합니다. 이는 Op의 isolated-from-above 속성이 제공하는 불변식을 통해 가능해집니다. SSA use-def 체인이 이런 op의 region 경계를 가로지를 수 없기 때문에, 이런 동작을 갖는 Op(예: "std.func" Op)은 병렬 처리 가능한 region tree를 정의합니다.

이 요구사항은 또한 MLIR이 whole-module use-def 체인을 갖지 않는 이유이기도 합니다(LLVM과 반대). 전역 객체는 symbol table 항목을 통해 참조되고, 상수는 관련 attribute를 갖는 Op으로 구현됩니다.

#### 0x6.4.4 상호 변환 가능한 IR 텍스트 표현
MLIR의 IR과 Op은 텍스트 표현 형태를 갖고 있어 메모리 IR 표현을 완전히 반영할 수 있습니다. 이는 디버깅, 변환 중 IR 이해, 테스트 케이스 작성에 매우 중요합니다. Figure 4의 원본 IR 표현은 길고 이해가 어려우므로, MLIR은 사용자가 Op마다 커스텀 print/parse 형식을 정의할 수 있게 합니다. 그 결과 예시는 Figure 8과 같이 print/parse 되어 사용이 한결 편해집니다. 두 형태는 서로 완전 변환 가능하며, 텍스트 형태를 입력·출력으로 사용해 각 컴파일러 pass를 개별 테스트할 수 있습니다. 숨은 상태가 없으므로 단일 pass 실행 결과는 전체 pass 파이프라인에서 해당 pass를 실행한 결과와 동일합니다. 이 방식은 IR 형식을 손으로 만들 수 있고 IR 변환을 손쉽게 추적할 수 있어 사용자 친화적입니다.


![커스텀 parse 형식의 Affine Dialect IR](https://img-blog.csdnimg.cn/0e5cd7896dc54885905e6b396c7b5cfd.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAanVzdF9zb3J0,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 0x6.4.5 문서
Dialect, Op, Interfaces 모두 그에 대응하는 ODS 기술로부터 문서가 자동 생성됩니다. summary와 더 읽기 좋은 description 외에도, 생성 문서에는 argument와 result type 제약이 포함됩니다. 검증 코드와 문서가 동일 source를 사용하므로, 문서는 runtime 동작과 동기화될 수 있습니다.

#### 0x6.4.6 Verifier
verifier는 IR의 구조적 정확성과 Op의 불변식을 강화하는 데 사용됩니다. pass가 검증된 IR 불변식이 점검되었음을 가정할 수 있게 하며, 디버깅 도구로도 활용됩니다. 검증 과정은 MLIR의 전체 구조 속성 검사부터 시작합니다. 예를 들어 type 매칭이 정확히 일치하는지, value가 한 번만 정의되었고 dominance 규칙과 가시성을 따르는지, symbol 이름이 symbol table 안에서 유일한지, 모든 block이 terminator Op으로 끝나는지 등을 확인합니다. 이후 각 Op과 attribute의 verifier가 적용됩니다. 각 Op은 구조적·의미적 유효성을 점검하는 일련의 규칙을 정의할 수 있습니다. 예를 들어 binary Op은 operand가 두 개인지 검사하고, 일부 Op은 특정 type의 value만 허용하며, 일부 Op은 특정 attribute나 region이 부착되어야 합니다. 마찬가지로 Dialect attribute는 특정 Op에서만 사용을 허용하거나, 부착된 Op에 대해 추가 제약을 부여할 수 있습니다. 예를 들어 어떤 Dialect attribute는 Op이 더 일반적이라 하더라도 Dialect에서 정의한 type만 사용하도록 요구할 수 있습니다. 검증 실패는 invariant violation으로 간주되며 컴파일을 중단시킵니다.

### 0x6.5 평가: MLIR의 응용
MLIR 시스템의 목적은 다양한 컴파일러 프로젝트를 통합하고 견인하는 것이므로, 주된 평가 지표는 MLIR이 어떤 프로젝트들에 채택되었는지를 보여주는 것입니다. 이 절에서는 사용자 커뮤니티 활동을 간단히 소개하고, 몇 가지 use case를 자세히 기술해 MLIR의 범용성과 확장성을 강조하며, MLIR이 어떻게 customization 설계 원칙을 잘 구현하는지 보입니다.

현재 MLIR은 학계와 산업계에 걸친 활발한 사용자 커뮤니티를 가진 오픈소스 프로젝트로 성장 중입니다. 4개 국가의 4개 국립 연구소와 16개 대학 인사들이 고성능 컴퓨팅(HPC)에서의 MLIR 사용을 주제로 한 학술 워크숍에 참여했습니다. MLIR은 또한 14개 다국적 기업의 인정을 받았습니다. LLVM Developer Meeting에서는 100명이 넘는 업계 개발자가 MLIR 관련 라운드 테이블에 참여했습니다. 26개 이상의 dialect가 개발 중이며, 서로 다른 회사의 7개 프로젝트가 자체 컴파일러 인프라를 MLIR로 교체하고 있습니다. 이는 MLIR에 대한 실질 수요와 사용성에 대한 인정을 보여 줍니다.

#### 0x6.5.1 TensorFlow graphs
대부분의 컴파일러 개발자가 다른 표현 형식에도 익숙하겠지만, MLIR의 핵심 use case 중 하나는 머신러닝 프레임워크 개발 지원입니다. 머신러닝 프레임워크의 내부 표현은 보통 동적 실행 의미(dynamic execution semantics)를 갖는 데이터 흐름 그래프에 기반합니다[53].

TensorFlow[1]가 그러한 프레임워크의 한 예입니다. TensorFlow의 표현은 고수준 데이터 흐름 계산이며, 그래프의 노드는 다양한 디바이스(특정 하드웨어 가속기 포함)에 배치 가능한 다양한 계산 과정입니다.

TensorFlow는 이 내부 표현을 모델링하고 Figure 1에서 보인 use case를 위한 변환을 수행하는 데 MLIR을 사용합니다. 단순한 대수적 최적화부터 (하드웨어 가속기로 구성된) 데이터센터 클러스터에서 병렬 실행 가능한 새로운 형태의 그래프로 변환하고, IR을 lowering 해서 XLA[57] 같은 도구로 효율적인 네이티브 코드를 생성하거나 모바일 배포에 적합한 표현을 만들 수 있습니다. MLIR에서의 TensorFlow Graph 표현은 그림 7과 같습니다.

![TensorFlow Graph에 대응하는 MLIR 표현](https://img-blog.csdnimg.cn/86181585c0034346a78ea3928d9a56e5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAanVzdF9zb3J0,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 0x6.5.2 Polyhedral code generation 다면체 코드 생성
MLIR의 초기 동기 중 하나는 가속기를 위한 polyhedral 코드 생성을 탐구하는 것이었습니다. affine dialect는 단순화된 polyhedral 표현 형태로, 점진적 IR lowering을 가능하게 하기 위해 설계되었습니다. 설계 요점에 대한 전반적 논의는 본 논문의 범위를 넘지만, 본 논문은 affine dialect의 몇 가지 측면을 설명해 MLIR의 모델링 능력을 보이고 affine dialect를 과거의 일부 표현 형식과 비교합니다[17, 19, 54, 55, 52].

##### 공통점
MLIR affine dialect는 모든 메모리 접근의 구조화된 다차원 type에 대해 연산할 수 있습니다. 기본적으로 이러한 구조화된 type은 injective(주입적)하여 서로 다른 인덱스가 구성상 alias 되지 않도록 보장합니다. 이는 polyhedral 종속성 분석의 일반적인 전제입니다.

Affine modeling은 두 부분으로 나눌 수 있습니다. attribute는 컴파일 타임에 affine map과 정수 집합을 모델링하는 데 사용되고, Op은 코드에 affine 제약을 적용하는 데 사용됩니다. 즉 `affine.for` Op은 "for" loop이며, 그 경계는 value들의 affine map으로 표현되고, 이 value들은 함수 안에서 불변(invariant)이어야 합니다. 따라서 loop은 정적 제어 흐름을 갖습니다. 마찬가지로 `affine.if`는 affine 정수 집합으로 제약된 조건문입니다. loop과 조건문의 본문은 region이며, 이 region들은 `affine.load`와 `affine.store`를 통해 인덱스를 loop 반복자(iterator)의 affine 형태로 제한합니다. 이렇게 하면 정확한 affine 종속성 분석이 가능하면서, 저수준 표현으로부터 affine 형태를 추론하는 일을 피할 수 있습니다.

##### 차이점
MLIR과 기존 polyhedral 코드 생성 프레임워크 사이의 차이는 많으며, 다음 네 가지로 분류할 수 있습니다.
(1) 풍부한 type: MLIR의 구조화된 memref type은 buffer 인덱스 공간을 실제 주소 공간으로 연결하는 layout map을 포함합니다. 이 두 공간을 분리하면 loop 변환과 데이터 변환의 조합이 개선됩니다. 데이터 layout 수정이 코드에 영향을 주거나 종속성 분석을 오염시키지 않기 때문입니다. 이러한 변환 혼합은 [38]에서 다룬 적이 있지만 흔치는 않습니다.
(2) 추상의 혼합: MLIR에서는 affine loop body를 typed SSA의 Op으로 표현할 수 있습니다. 따라서 모든 전통적인 컴파일러 분석·변환 과정이 그대로 적용 가능하며, polyhedral 변환과 교차 사용할 수 있습니다. 이와 달리 polyhedral 컴파일러는 보통 이런 세부 사항을 완전히 추상화해 버리는데, 그러면 vector type 같은 일부 객체를 다루기가 어려워집니다.
(3) 작은 표현 차이: polyhedral 모델의 주요 특징 중 하나는 type 시스템 안에서 loop 반복 순서를 표현할 수 있다는 점입니다. 그러나 polyhedral 변환은 IR을 원본과 완전히 다른 표현으로 끌어올리는 경향이 있습니다[20, 10]. 게다가 변환된 polyhedral에서 loop으로의 변환은 계산상 어렵습니다[7]. MLIR 기반 표현은 저수준 표현 안에서 고수준 loop 구조를 그대로 유지해 IR을 끌어올릴 필요가 없습니다.
(4) 0x6.3.3 Pass Manager 절에서 설명한 바와 같이 컴파일 속도는 MLIR의 핵심 목표지만, 기존 polyhedral 방법 대부분은 컴파일 속도에 신경 쓰지 않습니다. 이런 polyhedral 방법들은 지수 복잡도 알고리즘에 크게 의존합니다. loop 순서 자동 도출을 위한 정수 선형 계획법, IR을 다시 loop으로 변환하기 위한 polyhedral scanning 알고리즘 등이 그것입니다. MLIR이 채택한 방식은 loop이 IR에 보존되므로 polyhedral scanning에 의존하지 않습니다.



> 논문은 MLIR이 도메인 특화 컴파일러에 적용된 사례와 MLIR 기반의 Fortran IR 등의 예시를 더 들고 있는데, 여기서는 더 이상 다루지 않습니다. 관심 있다면 원문을 직접 살펴보세요.


### 0x6.6 MLIR 설계의 결과
MLIR 설계는 새로운 언어와 컴파일 추상의 모델링을 도우면서, 기존의 범용 컴파일 기법 재사용에도 도움이 됩니다. **MLIR이 많은 문제를 효과적으로 푸는 방식은 "새 operation, 새 type을 추가"하는 것이며, 가능한 경우 그것들을 "어떤 새 dialect"로 모으는 것입니다.** 컴파일러 엔지니어링 측면에서 이는 큰 설계 전환이며, 새로운 기회와 도전, 통찰을 낳습니다. 본 절에서는 그중 일부 관점을 살펴봅니다.

#### 0x6.6.1 재사용 가능한 컴파일러 Pass
하나의 IR 안에서 여러 추상 수준을 표현할 수 있는 능력은 자연스레 여러 추상 수준을 가로질러 동작하는 pass를 작성하자는 아이디어로 이어집니다. MLIR에 자주 나오는 질문은, MLIR이 확장 가능한 operation·type 시스템을 가졌는데 어떻게 컴파일러 pass를 작성하는가입니다. pass는 항상 알 수 없는 구조를 보수적·정확하게 다룰 수 있겠지만, MLIR은 고성능 코드를 만들어 내는 것이 목표이며, 이를 위해 주로 네 가지 방법을 제공합니다.

**기본 operation 특성** "bread and butter"에 해당하는 일부 컴파일러 pass(예: dead code elimination, common subexpression elimination)는 우리가 Op trait로 정의한 단순한 속성("has no side effect", "is commutative" 등)에만 의존합니다. ODS의 Op 정의는 Op 개발자가 이러한 trait를 지정할 수 있게 해 주며, pass는 이 정보를 활용해 다양한 추상 도메인에서 동작을 유지할 수 있습니다.

MLIR의 확장성에는 구조적 attribute도 포함되어 있는데, 다음과 같은 정보를 담습니다. **어떤 operation이 제어 흐름 terminator로 알려져 있는지**, **어떤 operation이 포함하는 region이 isolated-from-above로 알려져 있는지** 등입니다. 이 정보들은 함수, closure, module, 기타 코드 구조의 모델링·처리에 사용될 수 있습니다.

**Privileged operation hooks(Op의 특수 hook)** 일부 trait는 단일 비트로 모델링할 수 있지만, 다른 많은 trait들은 C++ 코드로 구현되어야 하는데, 가령 constant folding 로직이 그렇습니다. MLIR은 다수의 pass에 적용 가능한 일부 hook에 대해 최선의 지원을 제공합니다. 이 hook들은 operation 단위로 구현될 수도 있고, dialect 객체에서 구현될 수도 있습니다. 후자의 방식은 TensorFlow op의 constant folding 같은 pass를 지원하기에 편리하며, 이런 경우 기존 로직에 위임하기 용이합니다.

constant folding 자체도 매우 중요한 기능이지만, 더 흥미로운 hook은 `getCanonicalizationPatterns`인데, 이는 operation에 적용할 folding pattern을 지정하게 해 줍니다. 이 덕분에 중요한 대수 단순화 형태(x − x → 0, min(x, y, y) → min(x, y) 등)가 확장 가능해지고, 일반적인 "Canonicalization" pass를 모든 dialect에 적용하는 데 도움이 됩니다. 이러한 단일 확장 시스템 안에 "InstCombine", "DAGCombine", "PeepholeOptimizer", "SILCombine" 같은 pass와 LLVM 생태계(및 다른 컴파일러)의 그 외 특수 목적 pass들을 담아낼 수 있습니다.

**Optimization interfaces(최적화 인터페이스)** MLIR의 주요 목표는 확장성이며, Op과 type 측면뿐 아니라 변환 측면에서도 확장 가능해야 합니다. canonicalization과 constant folding이 핵심이지만, 많은 표준 변환을 어떤 식으로든 매개변수화해야 변환의 특정 속성이나 코드 모델 등을 기술할 수 있습니다.

이 문제의 해결책이 "optimization interface"라 불리는 서브시스템입니다. MLIR inliner pass를 생각해 봅시다. inliner가 TensorFlow 그래프, Flang 함수, 함수형 언어의 closure 등을 모두 처리할 수 있길 바라지만, inliner는 호출자가 무엇인지, 심지어 호출 대상이 무엇인지조차 모릅니다. inliner가 알아야 할 핵심 속성은 다음과 같습니다.

- 주어진 operation을 주어진 region에 inline 하는 것이 유효한가;
- inline 후 block 중간에 끝나는 terminator operation을 어떻게 처리할 것인가.


이러한 속성을 알기 위해 Inliner pass는 Figure 10의 인터페이스를 정의합니다. 각 operation과 dialect는 MLIR에 이 인터페이스의 operation·dialect 구현을 등록할 수 있고, 범용 Inliner pass의 혜택을 받습니다. 어떤 operation이나 dialect가 인터페이스를 제공하지 않으면 해당 최적화 pass는 그 operation을 보수적으로 처리합니다. 이런 설계 덕분에 dialect 개발자는 빠르게 dialect를 시작·실행할 수 있습니다. 시간이 지나면서 인터페이스 개발에 더 많은 노력을 투입하면 시스템에서 더 많은 이득을 얻을 수 있습니다.


![Inline Pass 인터페이스](https://img-blog.csdnimg.cn/cd689e86f3034ac6b459c9646c2e4a24.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAanVzdF9zb3J0,size_20,color_FFFFFF,t_70,g_se,x_16)

optimization interface는 또한 핵심 컴파일러에 모듈화 측면의 이득을 줍니다. dialect 특화 로직이 핵심 변환이 아니라 dialect 내부에서 구현되기 때문입니다.

**Dialect 특화 pass** 마지막으로, 특정 dialect를 정의하면서 전용 pass를 정의할 수 있고, MLIR 시스템 내의 이런 pass들은 다른 컴파일러 시스템의 pass처럼 유용합니다. 예컨대 코드 생성기가 특정 머신 제약에 따라 머신 instruction을 커스텀 schedule 하기를 원한다면 전용 pass로 그 목적을 달성할 수 있습니다. 이는 새 변환 pass 개발의 출발점으로 활용할 수 있으며, pass의 범용성을 고려하지 않아도 됩니다.



#### 0x6.6.2 Dialect의 혼합
MLIR에서 가장 근본적이면서(또한 가장 이해하기 어려운) 부분 중 하나는 한 프로그램 안에서 서로 다른 dialect의 operation을 혼합 사용하도록 허용·장려한다는 점입니다. 어떤 경우(예: 호스트와 가속기의 계산을 같은 module에 보존)에는 이해하기 쉬운 일이지만, 가장 흥미로운 경우는 MLIR이 dialect를 직접 혼합 사용할 수 있게 한다는 점인데(이는 클래스 전체의 재사용을 가능케 합니다), 다른 시스템에서는 보기 힘든 일입니다.

0x6.5.2 절에서 설명한 affine dialect를 떠올려 봅시다. affine 제어 흐름과 affine map의 정의는 affine region 안에 포함된 operation의 의미와 무관합니다. 우리 사례에서는 affine dialect를 "standard" dialect와 결합해 단순 산술을 (LLVM IR처럼) 타깃 독립적인 형태로 표현할 수도 있고, 내부 가속기를 타깃으로 하기 위해 affine dialect를 여러 타깃 의존 머신 instruction dialect와 결합할 수도 있습니다. 또 어떤 사람들은 affine dialect를 다른 문제 영역의 추상과 결합해 사용하기도 합니다.

범용 polyhedral 변환을 (특정 변환에서 operation의 의미를 얻기 위해 Op Interface를 사용해) 재사용할 수 있는 능력은 컴파일러 인프라를 분해하는 강력한 방법입니다. 또 다른 예로, 다양한 소스 언어 IR에서 OpenMP dialect를 사용·재사용할 수 있습니다.

#### 0x6.6.3 상호 운용성

본 연구는 protobuf 형식의 머신러닝 그래프, LLVM IR을 포함한 여러 컴파일러 IR, 다양한 독자 instruction set 등 수많은 기존 시스템과의 상호 운용을 다룹니다. 어떤 표현 형식에든 크고 작은 결함이 있기 마련이며, 그런 결함이 특정 기존 시스템의 사용 환경에서는 합리적이라 해도 MLIR의 표현 능력은 MLIR을 더 나은 표현 형식으로 만듭니다. importer와 exporter는 테스트가 어렵기 때문에(테스트 케이스가 보통 바이너리 형식임), 그 복잡도를 최소한으로 유지하고 싶었습니다.

해법은 가능한 한 외부 시스템과 직접 대응되는 dialect를 정의해, 단순하고 예측 가능한 방식으로 그 형식을 양방향 변환하는 것입니다. 일단 IR을 MLIR 형식으로 import 하고 나면, MLIR 인프라의 모든 변환을 사용해 import한 IR을 더 적합한 IR 형식으로 끌어올리거나 내릴 수 있고, 이러한 변환 pass들도 다른 모든 MLIR pass와 동일한 방식으로 테스트할 수 있습니다.

이런 dialect의 예시는 많습니다. a) LLVM IR을 MLIR로 매핑할 수 있는 LLVM dialect. b) TensorFlow의 그래프 표현. 이는 TensorFlow의 "switch and merge" 노드 관련 분석·변환을 단순화하기 위해 제안된 표현입니다. c) 함수형 제어 흐름 연산자. "functional while"과 "functional if"는 머신러닝 그래프에서 자주 등장하며, 본문을 외부(out-of-line) 함수가 아니라 region으로 두는 편이 더 편리합니다.

이 접근 방식은 우리에게 잘 맞았으며, MLIR 도구는 외부 바이너리 파일 형식의 테스트 케이스 작성에도 유용합니다.


#### 0x6.6.4 비표준화 설계가 가져오는 새로운 도전
MLIR은 거의 임의의 추상을 정의할 수 있게 해 주지만, 실제로 어떤 방법이 더 잘 작동하고 어떤 방법이 그렇지 않은지에 대한 안내는 거의 제공하지 않습니다. 지금은 일부 엔지니어와 연구자들이 이 분야의 경험을 쌓아 가고 있고, 컴파일러 IR 설계와 추상 설계의 "예술"이 컴파일러·언어 분야에서 잘 이해되어 있지 않다는 점을 깨닫고 있습니다. 많은 사람들이 이미 정착된 시스템의 제약 안에서 일하지만, 자기 손으로 추상을 정의해 볼 기회를 얻은 사람은 상대적으로 적습니다.

이는 도전이지만 동시에 미래 연구의 기회이기도 합니다. MLIR 커뮤니티는 이러한 추상 설계를 통해 전문성을 쌓고 있으며, 시간이 지나면 결실 풍부한 연구 영역이 될 것입니다.

#### 0x6.6.5 기대
MLIR을 여러 시스템에 구축·적용해 본 결과, MLIR의 설계는 다른 컴파일러 인프라와 상당히 다르다는 것을 알 수 있습니다. 우리는 아직 발견되지 않은 응용 영역이 많이 있다고 믿으며, MLIR의 모든 설계 요점을 완전히 이해하고 베스트 프랙티스를 정립하기 위해서는 더 많은 연구 시간이 필요합니다. 예를 들어 out-of-tree dialect의 부상, MLIR을 사용하는 프론트엔드의 수 증가, AST에 대한 가능 응용, 그리고 JSON, protocol buffer 등 구조화된 데이터에 대한 응용 등은 아직 매우 초기 단계이며, 이로부터 흥미로운 새 도전과 기회가 발견될 수 있습니다.

## 0x7. 코멘트(OneFlow Dialect를 예시로)
이상이 MLIR 논문의 대략적 내용입니다. MLIR 논문에서 언급된 컴포넌트들을 마인드맵으로 그리면 대략 다음과 같습니다.

![Dialect의 구성 요소](https://img-blog.csdnimg.cn/74342e27132243a9acc7d2ce6e0fc845.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAanVzdF9zb3J0,size_20,color_FFFFFF,t_70,g_se,x_16)

이제 OneFlow Dialect를 예로 들어 이 그림을 설명해 보겠습니다.

논문에서 언급했듯, MLIR에서 Operation은 MLIR의 기본 의미 단위입니다. 새로운 Dialect를 정의하면 우선 Operation 정의를 고려해야 하고, Operation을 정의하려면 먼저 Attribute와 Type을 정의해야 합니다. OneFlow Dialect는 `oneflow/ir/include/OneFlow/OneFlowDialect.td` 파일에 정의되며, ODS 규칙에 따라 `description`, `cppNamespce` 등 핵심 정보를 설정한 후, MLIR이 제공하는 `mlir-tblgen` 실행 파일을 통해 OneFlow Dialect의 C++ 코드를 자동 생성합니다.

```cpp
def OneFlow_Dialect : Dialect {
    let name = "oneflow";
    let summary = "OneFlow MLIR dialect.";
    let description = [{
        This dialect is the IR of OneFlow.
    }];
    let cppNamespace = "::mlir::oneflow";
    let dependentDialects = [
        "StandardOpsDialect"
    ];
}
```

다음으로 Type 정의는 `oneflow/ir/include/OneFlow/OneFlowBase.td`와 `oneflow/ir/include/OneFlow/OneFlowEnums.td` 두 파일에 있으며, 각각 OneFlow의 Tensor type, 그리고 이후 Operation에서 Attribute 지정에 필요한 Type을 정의합니다. 참고로 OneFlow의 Operation 정의는 아래에서 정의하는 Type 외에도 MLIR이 제공하는 기본 Type을 다수 사용합니다.

```cpp
def OneFlow_Tensor : TensorOf<[AnyType]>;
def SI32ArrayAttr : TypedArrayAttrBase<SI32Attr, "signed 32-bit integer array attribute"> {}

def SI64ArrayAttr : TypedArrayAttrBase<SI64Attr, "signed 64-bit integer array attribute"> {}

def ShapeAttr : TypedArrayAttrBase<SI64Attr, ""> {}
...
```

Attribute 정의는 각 Operation 정의 안에서 `let attrs=`로 지정합니다. LeakyReLU를 예로 OneFlow Dialect의 Operation 정의를 살펴봅시다(`oneflow/ir/include/OneFlow/OneFlowUserOps.td`).

```cpp
def OneFlow_LeakyReluOp : OneFlow_BaseOp<"leaky_relu", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$x
  );
  let output = (outs
    OneFlow_Tensor:$y
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$alpha
  );
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}
```

`OneFlow_LeakyReluOp`가 `OneFlow_BaseOp`를 상속하고 입력·출력·Attribute를 선언하고 있는 것을 볼 수 있습니다. 마지막 4개 표시는 OneFlow가 LLVM의 `table-gen` 위에 추가한 작은 확장으로, Op 정보 추론 인터페이스를 자동 생성하기 위한 것이며 여기서는 신경 쓰지 않아도 됩니다.

위에서 Attribute, Type, Interface를 다뤘으니, 이제 OneFlow Dialect의 Operation에 적용된 Trait와 Constraint에 대해 이야기해 봅시다. MLIR에서 Trait(특성)와 Constraint(제약)의 base 클래스는 `OpTrait` 클래스이며, trait와 constraint는 보통 Operation의 특수 속성과 제약을 지정하는 데 쓰입니다. 예를 들어 Operation에 부수 효과(side effect)가 있는지, Op의 출력이 입력과 동일한 형상을 갖는지 등을 표현합니다.

OneFlow의 Operation 정의는 LeakyReLU의 `NoSideEffect` 같은 MLIR 내장 trait를 사용할 뿐 아니라, `IsOpConfCompatible` 같은 사용자 정의 trait도 사용합니다. `oneflow/ir/include/OneFlow/OneFlowBase.td`의 `def OneFlow_IsOpConfCompatible : NativeOpTrait<"IsOpConfCompatible">;` 이 한 줄은 MLIR이 제공하는 ODS 메서드 `NativeOpTrait`를 사용해 사용자 정의 trait를 선언한 것으로, OneFlow Dialect에서 정의한 Op이 OpName, DeviceDagAttr 등 일부 공통 attribute를 가졌는지 검사하는 용도입니다. 여기서는 ODS에서 사용자 정의 속성을 선언만 했고, 실제 정의는 `oneflow/ir/include/OneFlow/OneFlowOpTraits.h`에 있습니다. 간단히 발췌해 보면 다음과 같습니다.

```cpp
template<typename ConcreteType>
class IsOpConfCompatible : public TraitBase<ConcreteType, IsOpConfCompatible> {
 public:
  static StringRef getOpNameAttr() { return "op_name"; }
  static StringRef getDeviceTagAttr() { return "device_tag"; }
  static StringRef getDeviceNameAttr() { return "device_name"; }
  static StringRef getScopeSymbolIDAttr() { return "scope_symbol_id"; }
  static StringRef getHierarchyAttr() { return "hierarchy"; }
  static LogicalResult verifyTrait(Operation* op) { return impl::VerifyIsOpConfCompatible(op); }
};

LogicalResult VerifyIsOpConfCompatible(Operation* op) {
  for (auto attr : {
           IsOpConfCompatible<void>::getOpNameAttr(),
           IsOpConfCompatible<void>::getDeviceTagAttr(),
       }) {
    if (!op->hasAttrOfType<StringAttr>(attr)) {
      return op->emitError("expected operation to have attribute: " + attr);
    }
  }
  if (!op->hasAttrOfType<ArrayAttr>(IsOpConfCompatible<void>::getDeviceNameAttr())) {
    return op->emitError("expected operation to have attribute: "
                         + IsOpConfCompatible<void>::getDeviceNameAttr());
  }
  return success();
}
```


Trait 외에도 OneFlow는 MLIR이 제공하는 일부 trait, 예컨대 `SameOperandsAndResultType`도 사용합니다. `oneflow/ir/include/OneFlow/OneFlowBase.td`의 `OneFlow_UnaryBaseOp` 정의가 그 예입니다.

```cpp
class OneFlow_UnaryBaseOp<string mnemonic, list<Trait> traits = []> :
        OneFlow_BaseOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType, NoSideEffect])> {
  let summary = "";
  let input = (ins AnyType:$x);
  let output = (outs AnyType:$y);
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}
```

이 trait가 표현하는 의미는, UnaryBaseOp을 상속한 Operation의 operand와 result type이 동일하다는 것입니다. 물론 trait도 constraint와 마찬가지로 사용자 정의가 가능하며, `td` 파일에서 `NativeOpTrait`로 선언한 뒤 실제 구현은 `oneflow/ir/include/OneFlow/OneFlowOpTraits.h`에 두면 됩니다.


여기까지 설명을 통해 MLIR의 Type, Attribute, Operation, Trait, Constraint에 대해 어느 정도 감을 잡으셨을 것입니다. 다음으로 Interfaces에 대해 이야기해 봅시다. Interfaces는 인터페이스로 번역할 수 있으며, MLIR의 Interfaces는 IR과 상호작용하는 일반적 방법을 제공합니다. Interfaces의 설계 목표는 특정 Dialect의 Operation이나 Dialect 고유 지식에 침투하지 않고도 MLIR 표현을 변환·분석할 수 있게 하는 것입니다. 이렇게 하면 변환·분석과 새로운 Dialect 및 그 Operation 추가를 분리(decoupling)할 수 있어 MLIR의 확장성이 크게 강화됩니다. Interfaces의 중요성을 설명하기 위해, 이 시리즈에서는 공식 문서를 참고해 별도 글을 한 편 썼으니 참고하세요: [[밑바닥부터 배우는 딥러닝 컴파일러] 18, MLIR의 Interfaces](https://mp.weixin.qq.com/s/yD-b75p1An4YTpfoIgB8mQ).

OneFlow에서 사용자 정의된 각종 Interfaces는 `oneflow/ir/include/OneFlow/OneFlowInterfaces.td`에 있습니다. `UserOpCompatibleInterface`를 예로 Interface의 구체적 구현을 살펴봅시다.

```cpp
def UserOpCompatibleInterface : OpInterface<"UserOpCompatible"> {
  let description = [{
    Interface to getting the hard-coded bn
  }];

  let methods = [
    StaticInterfaceMethod<"",
        "const std::vector<std::string>*", "inputKeys", (ins), [{
        static std::vector<std::string> val(mlir::oneflow::support::GetInputKeys(ConcreteOp::getOperationName().split('.').second.str()));
        return &val;
    }]>,
    StaticInterfaceMethod<"",
        "const std::vector<std::string>*", "outputKeys", (ins), [{
        static std::vector<std::string> val(mlir::oneflow::support::GetOutputKeys(ConcreteOp::getOperationName().split('.').second.str()));
        return &val;
    }]>,
    InterfaceMethod<"",
        "std::pair<unsigned, unsigned>", "getODSOperandIndexAndLength", (ins "unsigned":$index), [{
        return $_op.getODSOperandIndexAndLength(index);
    }]>,
    InterfaceMethod<"",
        "std::pair<unsigned, unsigned>", "getODSResultIndexAndLength", (ins "unsigned":$index), [{
        return $_op.getODSResultIndexAndLength(index);
    }]>
  ];
}
```

`UserOpCompatibleInterface`가 Interface ODS 규약의 StaticInterfaceMethod와 InterfaceMethod를 사용해 이 Interface의 메서드들을 지정한 것을 확인할 수 있습니다. Operation의 입력 operand 이름, 출력 operand 이름, operand 및 길이, result 및 길이 등을 가져오는 메서드들입니다. 이후 OneFlow의 `oneflow/ir/include/OneFlow/OneFlowUserOps.td`에서 `DeclareOpInterfaceMethods<UserOpCompatibleInterface>`를 사용해 Operation 단위 Interface로 지정하면, 생성된 Operation 코드에 이 Interface 선언이 함께 포함됩니다.

이렇게 하면 어떤 이점이 있을까요? 첫째, OneFlow의 UserOp들이 모두 UserOpCompatibleInterface를 갖고 있으므로, OneFlow의 UserOp용 범용 `GetInputKeys` 함수를 한 번만 구현하면 UserOp을 상속한 모든 Operation이 이 함수 기능을 갖게 됩니다. UserOpCompatibleInterface 인터페이스를 모두 갖추고 있기 때문이죠.

Interface의 더 일반적이고 고전적인 예는 Interface 기반의 범용 pass 개발입니다. 예를 들어 inline pass와 형상 추론 pass가 있는데, 이는 Dialect 단위 Interface 활용 사례입니다. 자세한 내용은 [[밑바닥부터 배우는 딥러닝 컴파일러] 13, MLIR에서 Pass 작성하는 법](https://mp.weixin.qq.com/s/3N9DK7aQtjoLgs-s0lP-jg)을 참고하세요.


마인드맵에서 아직 다루지 않은 것은 Block과 Region입니다. 사실 MLIR 논문의 Region·Block 설명은 이미 충분하다고 생각합니다. 한 Op은 일련의 Region을 가질 수 있고, Region은 MLIR의 nested 구조 구현 메커니즘을 제공합니다. 한 Operation이 일련의 Region을 가지고, Region은 다시 일련의 Block으로 구성되며, Block은 다시 일련의 Op을 포함합니다. 이렇게 nested 관계를 이루어 scope와 제어 흐름 관계를 표현합니다. OneFlow Dialect에서 Region과 Block 활용은 현재 주로 함수 관련 의미에서 이루어집니다. 예를 들어 `oneflow/ir/lib/OneFlow/Passes.cpp`에 구현된 `OutlineMulCast` Pass는 IR 안의 일부 op pattern을 FuncOp type의 Operation으로 외부화(outlining)해 실행할 수 있게 하는데, 이 과정에서 Block을 사용해 이 FuncOp을 IR의 어느 위치에 삽입할지 결정합니다. 또 다른 예로, FuncOp의 매개변수에 접근할 때도 Block이 필요합니다. `oneflow/ir/lib/OneFlow/OneFlowOps.cpp`에서 Job Op을 위한 verify 함수를 구현해, 함수의 매개변수 리스트와 entry Block의 argument 리스트가 정렬되어 있는지 검증합니다.

```cpp
static LogicalResult verify(Job op) {
  // If this function is external there is nothing to do.
  if (op.isExternal()) return success();

  // Verify that the argument list of the function and the arg list of the entry
  // block line up.  The trait already verified that the number of arguments is
  // the same between the signature and the block.
  auto fnInputTypes = op.getType().getInputs();
  Block& entryBlock = op.front();
  for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
    if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
      return op.emitOpError("type of entry block argument #")
             << i << '(' << entryBlock.getArgument(i).getType()
             << ") must match the type of the corresponding argument in "
             << "function signature(" << fnInputTypes[i] << ')';

  return success();
}
```

Job Op의 정의를 보면 `let regions = (region AnyRegion:$body);`를 통해 body라는 이름의 Region을 바인딩하고 있음을 알 수 있습니다. 따라서 위 verify 함수에서 FuncOp의 Block에 접근하는 것은 암묵적으로 그 Block이 속한 Region에 접근하는 것이며, 즉 `op.front()`를 `op.body().front()`로 바꿔도 동일한 효과입니다.

Region과 Block은 MLIR에서 scope에 해당하며, 이를 통해 MLIR의 복잡한 구조에서 제어 흐름 관계를 구현하고, 일반 Op과 FuncOp, ModuleOp을 구분해 MLIR에서 Operation 통일성 원칙을 실현할 수 있습니다.


OneFlow Dialect와 관련된 Pass 메커니즘은 [OneFlow를 예로 MLIR 실제 개발 흐름 살펴보기](https://mp.weixin.qq.com/s/eUIm4QZbKU69B9_h3f109A)에서 이미 소개한 바 있어 여기서는 다시 다루지 않습니다.

전체적으로 MLIR은 재사용성과 확장성이 모두 좋은 컴파일 인프라입니다. 적어도 엔지니어링 개발 관점에서는 따라가 볼 만합니다. 이 부분에서는 OneFlow Dialect의 다양한 컴포넌트와 그 관계를 소개했지만, OneFlow Dialect와 다른 Dialect들의 관계, 그리고 현재 Dialect의 lowering 흐름은 아직 다루지 않았습니다. 사실 MLIR에는 깊이 연구·학습해야 할 기술 세부가 상당히 많이 남아 있어, 본 글은 주로 정리와 영감 부여 역할을 합니다.

## 0x8. 참고 문헌
본 글은 https://arxiv.org/pdf/2002.11054.pdf MLIR 원문 논문과 https://zhuanlan.zhihu.com/p/336543238 번역본을 참고했습니다.




















 












