# Mu Li 식 독서법으로 PyTorch FX 논문 읽기

`torch.fx`는 PyTorch에게는 확실히 좋은 작업이라고 할 수 있는데, 동적 그래프와 정적 그래프 사이의 일부 Gap을 없애주기 때문입니다. 예를 들어 graph transformation 측면에서, `torch.fx`는 PyTorch가 다른 정적 그래프 framework가 하는 operator fusion 최적화 같은 작업을 매우 쉽게 할 수 있도록 만들어줍니다. 또한 `torch.fx`는 post-training quantization과 quantization-aware training, 그리고 AMP 등의 구현 난이도를 크게 낮춰주는데, 이는 우리가 Python 레벨에서 직접 이 IR을 다룰 수 있게 된 덕분입니다. 그래서 저는 이것이 꽤 괜찮은 작업이라고 생각합니다. 특히 PyTorch로 개발하는 알고리즘 엔지니어에게는 이제 이 특성을 기반으로 마음껏 상상력을 펼칠 수 있게 되었습니다. 저도 예전에 FX 주변에서 QAT 작업을 하나 해본 적이 있는데 관심 있다면 다음 글을 읽어보세요: [OneFlow 기반 quantization-aware training 구현](<https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247496930&idx=1&sn=4cd6eebe9d6a691820f6e312d3fb2666&scene=21#wechat_redirect>). `torch.fx`의 셀링 포인트는, 순수 Python 언어로 PyTorch 프로그램의 계산 그래프를 capture하여 IR로 변환할 수 있는 라이브러리를 구현했다는 점, 그리고 이 IR 위에서 매우 편리하게 Pass를 작성할 수 있고, 동시에 변환된 IR을 합법적인 Python 코드로 Codegen하는 기능까지 제공한다는 점입니다. 저는 Eager 모드에서 Pass를 작성하는 일이 마치 연결 리스트의 삽입/삭제 문제를 푸는 것처럼 매끄러워졌다고 느낍니다.

# 0x0. 동기

최근 Mu Li (리무) 선생님이 Bilibili(빌리빌리)에서 일부 고전 논문 리딩 영상을 공유했고, 저도 따라서 Transformer, ViT 등 몇 개를 봤는데 매우 훌륭했습니다. 그래서 저도 Mu Li 선생님의 이런 논문 독서법을 시도해보고자 논문 한 편을 골라 읽어보기로 했습니다. 현재 제가 비교적 관심을 가지고 있는 분야는 엔지니어링 방향의 논문인데, 마침 지난주에 PyTorch에서 `torch.fx`의 논문을 공개해서 이 논문을 예시로 시도해보았습니다. Mu Li 선생님의 논문 독서법은 대략 다음과 같습니다(숫자는 순서를 나타냅니다):

  1. 제목
  2. 초록(Abstract)
  3. 서론(Introduction)
  4. 결론(Conclusion)
  5. 관련 연구(Related Work)
  6. FX 특성
  7. 실험
  8. 코멘트



PyTorch FX 논문 링크는 https://arxiv.org/pdf/2112.08429.pdf 입니다. 아래에서는 Mu Li 선생님의 논문 독서 순서대로 독서 경험을 공유하여, PyTorch FX라는 특성이 도대체 무엇이며, 그것이 PyTorch 안에서 어떤 역할을 할 수 있는지 명확히 이해하는 데 도움을 드리고자 합니다.

# 0x1. 제목

![이미지](images/img_01.png)torch.fx 제목과 저자

다음과 같이 번역할 수 있습니다: TORCH.FX: Python 기반 딥러닝의 실용적인 프로그램 **「capture」** 와 **「변환(transformation)」**. 여기서 "Python 기반 딥러닝의 실용 프로그램"은 PyTorch 기반으로 개발된 모델 프로그램으로 이해할 수 있고, 그 다음 핵심은 capture와 변환입니다. 지금은 여기서 capture와 변환이 무엇을 의미하는지 아직 분명하지 않은데, 계속 읽어 나가면 됩니다. 잠깐 여담을 하자면, 저는 반년 전부터 FX에 관심을 가져왔고, 예전에 OneFlow framework에서도 FX를 성공적으로 통합하여 QAT 작업을 한 번 해봤습니다. 그 당시에는 FX에 아직 논문이 없었기 때문에, 이 논문은 TORCH.FX라는 특성에 대한 정리이자 자리매김 같은 느낌이 듭니다.

# 0x2. 초록

초록 부분에서는 PyTorch처럼 동적 그래프 실행 방식을 기반으로 하는 딥러닝 framework가 비록 사용자의 편의성을 향상시켰지만, 일부 실제 시나리오에서 사용자는 성능 최적화, 시각화, 분석, 하드웨어 튜닝 등을 위해 프로그램 구조(신경망 구조로 직접 이해해도 됩니다)를 capture하고 변환할 필요가 있다는 점을 간단히 지적합니다. 이러한 페인 포인트를 해결하기 위해, PyTorch는 PyTorch 프로그램의 capture와 변환을 위한 `torch.fx`라는 모듈을 설계했고, 이 모듈은 순수 Python으로 개발되었습니다.

이 절은 주로 `torch.fx`의 셀링 포인트에 대해 이야기합니다. 즉, 동적 그래프는 사용성이 매우 강하지만 그래프 구조를 사전에 인식하고 변환할 수 없는데, 이 논문의 `torch.fx` 모듈을 통해 이 일이 가능해졌다는 것입니다.

# 0x3. 서론

초기의 graph 모드 또는 `define-and-run`이라고 부르는 정적 그래프 framework에는 Caffe, TensorFlow 등이 있는데, 이들은 graph를 표현하는 IR을 설계하고 사용자가 framework가 제공하는 API를 호출하여 IR을 구축하도록 했습니다. 그런 다음 이 IR 위에서 프로그램 미분, IR을 device에 분할하여 병렬화 구현, quantization, 성능 최적화 등을 할 수 있습니다. 그러나 이런 일들은 보통 도메인 특화 언어 위에서 해야 합니다. 예를 들어 OneFlow의 정적 그래프 모드를 보면, 그래프 분할, quantization, 성능 최적화 등을 하려면 모두 C++ 기반으로 개발해야 하고, 디버깅도 상대적으로 어렵습니다(pdb, gdb 등 전문 도구의 도움을 받아야 합니다).

현재의 eager 모드 또는 `define-by-run`이라고 부르는 동적 그래프 framework에는 PyTorch, TensorFlow Eager 모드 등이 있습니다. 이들은 사용자가 스크립트 언어를 기반으로 마음껏 프로그래밍할 수 있게 해주고 대부분의 학습(자동 미분 기반)과 추론 작업을 해결할 수 있습니다. 그러나 **「quantization과 operator fusion」** 같은 일부 변환은 직접 할 수 없는데, 이런 작업은 정적 그래프 모드에서는 매우 간단합니다. 이러한 Gap을 없애기 위해, 동적 그래프 framework는 이러한 변환을 가능하게 하기 위해 사용자의 프로그램으로부터 graph 구조를 capture하는 방법이 필요합니다.

사실 이러한 프로그램 capture 기술은 PyTorch에 이미 예전부터 있었는데, 바로 TorchScript입니다. 이는 Python 프로그램의 AST를 기반으로 IR을 구성하고, 전체 Python 프로그램에 대해 포괄적으로 모델링합니다. 그러나 이렇게 하면 여전히 한 가지 문제가 있는데, 바로 프로그램 capture의 기술적 복잡도가 너무 크고, 이 높은 복잡도의 IR 위에서 변환을 작성하는 것이 너무 어렵다는 것입니다. 비교를 위해 우리는 요구사항을 단순화할 수 있습니다. Python 프로그램을 포괄적으로 모델링하는 것에서, quantization과 operator fusion 같은 변환을 할 수 있는 정도까지만 모델링하면 됩니다. 이 두 가지를 하기 위해서는, 사실 프로그램 안에 있는 그 DAG 구조만 있으면 되고, 프로그램 안에 숨겨진 상위 API의 구조(예를 들어 convolution과 BN)는 필요하지 않습니다. 이 말의 의미는, convolution과 BN 같은 여러 operation으로 조립될 수 있는 `nn.Module`에 주목할 필요가 없고, 그 위쪽에서 잘라내면 된다는 것입니다. 즉, 결국 high-level API의 DAG를 얻으면 됩니다.

위와 같은 아이디어를 바탕으로 `torch.fx`가 제안되었습니다. 이는 딥러닝 프로그램에서 DAG에 주목하고, 이 DAG를 얻기 위한 커스터마이징 인터페이스를 제공합니다. 이렇게 하면 `torch.fx`는 대부분의 딥러닝 framework에서의 graph transformation을 구현할 수 있고, 동시에 사용자가 직접 변환을 정의할 수 있도록 돕는 간단하고 사용하기 쉬운 API들도 제공합니다. 정리하자면 `torch.fx`의 핵심 셀링 포인트는 다음과 같습니다:

  1. 딥러닝 프로그램에 매우 중요한 프로그램 capture 및 변환을 위한 실용적인 분석 특성. Trace
  2. 순수 Python으로만 구현된 프로그램 capture 라이브러리로, 다양한 수준의 프로그램 디테일을 capture하도록 커스터마이징 가능. Pure Python
  3. 6개의 명령어만 가지는 단순한 IR로 capture된 프로그램을 표현, 이해와 정적 분석이 쉬움에 중점. IR
   4. 변환된 코드를 호스트 언어 생태계로 되돌리기 위한 코드 생성 시스템. Codegen
  5. torch.fx를 활용하여 실제로 성능 최적화, 프로그램 분석, device lowering 등의 기능을 어떻게 개발하는지에 대한 사례 연구. Eager Pass



# 0x4. 결론

우리는 PyTorch 프로그램을 capture하고 변환하기 위한 순수 Python 시스템인 `torch.fx`를 제안하였습니다. 우리는 관련 시스템을 복잡하게 만드는 요인들, 즉 control flow, mutability, 데이터 모델을 분석하였고, `torch.fx`가 어떻게 일반적인 사용 사례와 커스터마이징 가능성에 집중함으로써 복잡성을 회피하는지 보였습니다. 우리는 최적화, 분석, device lowering 측면에서 `torch.fx`의 다양한 사용 사례를 조사하였고, `torch.fx`의 API 디자인이 어떻게 이러한 결과를 가능하게 하는지 보였습니다.

# 0x5. 관련 연구

프로그램을 capture하고 변환할 때, eager 모드와 graph 모드의 딥러닝 framework는 모두 **「프로그램 구조 capture」**, **「프로그램 특화(specialization)」** 그리고 **「프로그램의 IR 디자인 보존」** 측면에서 선택을 해야 합니다. 이러한 선택의 조합이 framework에서 표현 가능한 **「프로그램 공간」**, **「변환을 작성하기 쉬운 정도」**, 그리고 **「생성된 변환 프로그램의 성능」**을 결정합니다. **「일반적으로, 프로그램의 고성능 실행을 지원하기 위해서는 더 복잡한 capture framework와 IR이 필요하며, 그 결과 변환을 작성하기가 더 어려워집니다」**. 관련 연구의 각 단락을 자세히 다루지는 않고, 각 단락의 핵심이 무엇을 말하고 있는지만 설명하겠습니다. 자세한 내용은 원 논문을 참고해주세요.

## 0x5.1 프로그램 구조 capture

이 절에서는 PyTorch의 `jit.trace`, MxNet Gluon, TensorFlow의 `tf.function` 등의 프로그램 capture 방법을 언급하면서, 이 방법들이 Python의 일부 부분 집합만 처리할 수 있다고 지적합니다. 그 다음, TorchScript는 AST 위에서의 분석을 통해 control flow와 더 많은 Python 문법을 처리할 수 있다고 합니다. 그리고 Julia와 Swift For TensorFlow에서는 프로그램 구조 capture 인터페이스를 Python이 아닌 호스트 언어에 통합했는데, 이를 사용하려면 사용자가 Python 생태계를 포기해야 한다고 언급합니다.

## 0x5.2 프로그램 특화

`a+b`라는 Python 문장의 경우, 이 표현식은 `a`와 `b`의 타입에 제한이 없습니다. 그러나 딥러닝 framework가 프로그램을 capture할 때는 일반적으로 이 두 변수를 특화하여 특정 타입이나 tensor에만 유효하도록 만듭니다. 딥러닝 framework에서 처리되는 대부분의 프로그램은 특화된 타입의 프로그램이며, 특화 정도가 높을수록 처리할 수 있는 입력은 적어집니다. 예를 들어 `torch.jit.trace`는 trace를 실행할 때 합법적인 입력 shape을 갖는 입력만 처리할 수 있습니다. 이어서 LazyTensor와 Jax의 `jit`에 대해 논의하면서, 특화된 프로그램에서의 capture 실패를 더 잘 처리하기 위해 그들이 어떤 노력을 기울였는지 설명합니다.

## 0x5.3 IR 디자인

딥러닝 framework는 모두 자신만의 IR 디자인을 가지고 있는데, Caffe와 TensorFlow는 Protocol Buffers 포맷을 사용합니다. 그리고 PyTorch와 MxNet은 C++ 데이터 구조를 사용해 IR을 표현하고 추가로 Python에 바인딩합니다. 이러한 IR 디자인들은 runtime 단계에서 모두 비교적 좋은 성능을 보이고, 통일적으로 직렬화될 수 있습니다. 그러나 다른 관점에서 보면, 이러한 IR 표현들은 순수 Python 언어 표현에 비해 더 높은 학습 비용을 요구합니다. 이어서, 이 절은 control flow와 상태 문제에 대해 논의하면서, 이러한 문제를 처리하려면 비교적 복잡한 IR을 설계해야 하고 이 IR 위에서 비교적 복잡한 분석을 해야 함을 보여줍니다.

위의 몇 가지 점을 바탕으로, 논문은 `torch.fx`의 기본 디자인 원칙을 제안합니다:

  * 롱테일 분포의 복잡한 사례를 지원하는 것은 피하고, 주로 고전적 모델의 프로그램 capture와 변환에 집중한다.
  * 머신러닝 종사자에게 이미 익숙한 도구와 개념, 예를 들어 Python의 데이터 구조나 PyTorch에서 공개적으로 문서화된 operator를 사용한다.
  * 프로그램 capture 과정을 고도로 설정 가능하게 만들어, 사용자가 롱테일 요구사항에 대해 자신만의 솔루션을 구현할 수 있게 한다.



이 절은 주로 일부 관련 연구를 펼쳐 보여줌으로써 `torch.fx`의 핵심 셀링 포인트를 부각시킵니다. 즉, 비록 TorchScript 같은 IR이 처리할 수 있는 일부 까다로운 케이스(예: 동적 control flow)는 처리하지 못하지만, 신경망이라는 영역 안에서는 충분히 쓸만하다는 것입니다. 가장 핵심적인 것은 구현이 매우 단순하고 순수 Python 라이브러리이기 때문에, 사용자가 변환을 작성하기 매우 쉽고, 학습 비용이 매우 작으며 사용성이 좋다는 점입니다. (간단하다고 강력하지 않다는 의미는 아닙니다!)

# 0x6. FX 특성

단순함을 기본 원칙으로 삼아, `torch.fx`는 symbolic tracing을 통해 프로그램을 capture하고, 6개 명령어로 구성된 단순한 IR로 그것을 표현하며, 이 IR을 기반으로 다시 Python 코드를 생성하여 실행합니다. JIT 특화에서 발생하는 재캡처(re-capture)의 복잡성을 피하기 위해, `torch.fx`는 프로그램 자체에 대해 특화를 수행하지 않고, capture 동안 어떤 특화를 구현할 필요가 있는지를 결정하는 일은 변환에 의존합니다. 사용자는 또한 symbolic tracing 과정을 설정하여 커스텀 capture 요구사항을 구현할 수 있습니다.

Figure 1은 `torch.fx.symbolic_trace`를 사용하여 프로그램을 capture하는 예시를 보여줍니다. 입력은 `torch.nn.Module`이거나 함수가 될 수 있으며, capture 후의 구조는 Graph 객체에 저장됩니다. 이 `Graph` 객체는 `GraphModule` 안의 모듈 파라미터들과 결합되며, `GraphModule`은 `torch.nn.Module`의 서브클래스이고, 그 `forward` 메서드는 capture된 `Graph`를 실행합니다. 우리는 이 graph의 `Nodes`를 출력하여 capture된 IR을 볼 수 있습니다. `placeholder` node는 입력을 표현하고, 단일 `output` node는 `Graph`의 결과를 표현합니다. `call_function` node는 호출할 Python 함수를 직접 참조합니다. `call_method` node는 첫 번째 인자의 메서드를 직접 호출합니다. `Graph`는 호출을 위해 Python 코드(`traced.code`)로 다시 조립됩니다.

![이미지](images/img_02.png)Figure 1

Figure 2는 `torch.fx`를 사용한 변환의 예시를 보여줍니다. 변환은 어떤 activation의 모든 instance를 찾아 다른 것으로 교체하는 것입니다. 여기서는 이를 사용해 `relu`를 `gelu`로 교체합니다.

![이미지](images/img_03.png)Figure 2

## 0x6.1 프로그램 capture

`torch.fx`의 symbolic tracing 메커니즘은 Proxy 데이터 구조를 사용하여 주어진 입력에 대해 어떤 op들을 거쳐갔는지 기록합니다. Proxy는 duck-typed 타입의 Python 클래스로, 그 위에서 일어나는 속성 접근과 메서드 호출을 기록하며, 프로그램 안의 실제 op에 대한 상위 추상화입니다. duck-typed에 대해서는 다음 소개를 참고하면 됩니다: https://zh.wikipedia.org/wiki/%E9%B8%AD%E5%AD%90%E7%B1%BB%E5%9E%8B . PyTorch의 operator 및 Python 부분 집합의 일부 함수들은 모두 이 Proxy로 한 번 wrapping되며, symbolic tracing이 `nn.Module`을 입력받을 때는 이 `nn.Module` 안의 자식 `nn.Module`도 Proxy로 wrapping되고, 물론 입력 데이터도 포함됩니다. 이렇게 하면 프로그램 안의 입력과 다른 op들이 모두 duck-typed 타입의 Proxy 객체가 되어, 우리는 이 프로그램을 실행할 수 있게 됩니다. 즉, 이것이 symbolic tracing 과정입니다. symbolic tracing 과정은 `Tracer` 클래스를 통해 설정되며, 그 메서드들은 어떤 값을 Proxy 객체로 유지할지, 어떤 값을 unpack할지를 제어하기 위해 오버라이드될 수 있습니다. (Proxy가 기록한 op는 unpack될 수 있고, unpack 후에는 실제 Tensor, Parameter, 연산자 등을 얻을 수 있습니다.) Proxy와 Tracer 클래스의 협력을 통해, `torch.fx`는 PyTorch 프로그램의 symbolic tracing을 완성할 수 있습니다. 여기서 symbolic tracing의 의미는, 프록시화된 후의 `nn.Module`의 forward를 한 번 실행하는 것이라는 점에 주의해야 합니다.

## 0x6.2 중간 표현(Intermediate Representation)

`torch.fx`의 중간 표현(IR)은 Python 데이터 구조 `Graph`로 만들어집니다. 이 `Graph`는 사실 일련의 `Node`들을 담고 있는 선형 리스트입니다. node는 문자열 opcode `opcode`를 가지며, 이는 node가 어떤 종류의 operation을 나타내는지를 기술합니다(opcode의 의미는 부록 A.1에서 확인할 수 있습니다). node는 연관된 target을 가지는데, 이는 호출 node(`call_module`, `call_function`, `call_method`)의 호출 target입니다. 마지막으로, node는 `args`와 `kwargs`를 가지며, trace 동안 이들은 함께 Python 호출 규약에서의 target 인자를 표현합니다(각 opcode에 대응하는 `args`와 `kwargs`의 의미는 부록 A.2에서 확인할 수 있습니다). node 사이의 데이터 의존 관계는 `args`와 `kwargs` 안에서 다른 node에 대한 참조로 표현됩니다.

`torch.fx`는 프로그램의 상태를 `GraphModule` 클래스에 저장합니다. `GraphModule`은 변환된 프로그램의 컨테이너로서, 변환 후 생성된 코드를 노출시키고, `nn.Module`과 유사한 파라미터 관리 API를 제공합니다. `GraphModule`은 일반 `nn.Module`을 사용할 수 있는 모든 곳에서 사용 가능하며, 이를 통해 변환된 코드와 PyTorch 생태계의 다른 부분 간의 상호운용성을 제공합니다.

## 0x6.3 source-to-source 변환

`torch.fx` 변환 pipeline의 마지막 단계는 코드 생성입니다. `torch.fx`는 Python 생태계를 빠져나와 맞춤형 runtime으로 들어가는 것이 아니라, 변환된 IR로부터 유효한 Python 소스 코드를 생성합니다. 그런 다음 이 변환된 코드는 Python으로 로드되어, 호출 가능한 Python 객체를 생성하고, `forward` 메서드로서 `GraphModule` 인스턴스에 설치됩니다. 코드 생성을 사용하면 `torch.fx` 변환의 결과를 모델에 설치하여 추가적인 변환에도 사용할 수 있습니다. 예를 들어, 그림 3에서는 원본 프로그램을 trace한 결과를 받아 그것을 새 모듈의 activation function으로 설치합니다.

![이미지](images/img_04.png)Figure 3

여기까지 PyTorch FX 특성에 대한 정독이 끝났습니다. 다만 FX 논문을 보면 Design Decisions라는 절도 있는데, 거기서는 Symbolic Tracing, Configurable Program Capture, AoT Capture without Specialization, Python-based IR and Transforms 등 FX 구현이 의존하고 있는 일부 아이디어와 결정, 그리고 그 장점들을 각각 소개합니다. 저는 이 절이 Introduction의 강화판이라고 이해하고 있으므로, 이 작은 절에 대한 설명은 더 이상 진행하지 않겠습니다. 만약 어떤 디테일을 놓치는 것이 걱정된다면 논문 원문을 읽어보시기 바랍니다.

# 0x7. 실험

`torch.fx`의 한 가지 목표는 딥러닝 모델이 생성하는 IR을 단순화하는 것입니다. 아래 Figure 5는 ResNet50을 예시로 TorchScript IR과 `torch.fx` IR의 차이를 보여줍니다. TorchScript IR과 비교했을 때, `torch.fx` IR은 확실히 더 단순하고 가독성도 더 좋습니다.

![이미지](images/img_05.png)Figure 5

post-quantization 및 quantization-aware training은 프로그램 추론 시의 성능을 끌어올릴 수 있다는 것을 우리는 알고 있습니다. 아래의 Figure 6은 `torch.fx` 기반으로 구현된 post-quantization(FBGEMM quantization operator 사용)을 DeepRecommender 모델에 적용한 후, Intel Xeon Gold 6138 CPU @2.00GHz에서의 성능을 보여줍니다. `torch.fx` 기반으로 구현된 post-quantization 모델의 추론 속도는 float 타입 모델 대비 3.3배 빠릅니다. 그리고 `torch.fx` 기반으로 quantization 작업을 구현하는 것은 TorchScript IR 기반보다 훨씬 간단합니다.

![이미지](images/img_06.png)Figure 6

`torch.fx`는 Op fusion도 할 수 있습니다. Figure 7은 `torch.fx` 기반으로 Conv+BN fusion을 한 후 ResNet50에 적용했을 때, NVIDIA Tesla V100-SXM2 16GB(CUDA version 11.0)와 Intel Xeon Gold 6138 CPU @ 2.00GHz에서의 성능을 보여줍니다. 보시는 것처럼 GPU에서는 약 6%의 latency 감소가 있고, CPU에서는 약 40%의 latency 감소(멀티스레드)와 약 18%의 latency 감소(싱글스레드)가 있습니다.

![이미지](images/img_07.png)Figure 7

이외에도 `torch.fx`는 FLOPs 계산, 메모리 대역폭 사용 분석, 워크로드의 데이터 값 크기 추정 등에 사용되어 프로그램 실행 시의 메모리와 속도를 분석할 수 있습니다. `torch.fx`는 또한 shape inference, 그리고 모델에 대응하는 DAG의 시각화 그리기 등에도 사용할 수 있습니다.

마지막으로, `torch.fx`는 runtime 단계에서 ASIC 가속도 지원합니다(즉 `torch.fx`의 operator를 대응되는 ASIC으로 lowering). 아래 Figure 8은 `torch.fx` 기반으로 ResNet50과 LearningToPaint를 추론하면서 operator를 TensorRT로 lowering한 후의 가속 결과를 보여줍니다:

![이미지](images/img_08.png)Figure 8

# 0x8. 코멘트

`torch.fx`는 PyTorch에게는 확실히 좋은 작업이라고 할 수 있는데, 동적 그래프와 정적 그래프 사이의 일부 Gap을 없애주기 때문입니다. 예를 들어 graph transformation 측면에서, `torch.fx`는 PyTorch가 다른 정적 그래프 framework가 하는 operator fusion 최적화 같은 작업을 매우 쉽게 할 수 있도록 만들어줍니다. 또한 `torch.fx`는 post-training quantization과 quantization-aware training, 그리고 AMP 등의 구현 난이도를 크게 낮춰주는데, 이는 우리가 Python 레벨에서 직접 이 IR을 다룰 수 있게 된 덕분입니다. 그래서 저는 이것이 꽤 괜찮은 작업이라고 생각합니다. 특히 PyTorch로 개발하는 알고리즘 엔지니어에게는 이제 이 특성을 기반으로 마음껏 상상력을 펼칠 수 있게 되었습니다. 저도 예전에 FX 주변에서 QAT 작업을 하나 해본 적이 있는데 관심 있다면 다음 글을 읽어보세요: [OneFlow 기반 quantization-aware training 구현](<https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247496930&idx=1&sn=4cd6eebe9d6a691820f6e312d3fb2666&scene=21#wechat_redirect>).

마지막으로 정리하자면, `torch.fx`의 셀링 포인트는, 순수 Python 언어로 PyTorch 프로그램의 계산 그래프를 capture하여 IR로 변환할 수 있는 라이브러리를 구현했다는 점, 그리고 이 IR 위에서 매우 편리하게 Pass를 작성할 수 있고, 동시에 변환된 IR을 합법적인 Python 코드로 Codegen하는 기능까지 제공한다는 점입니다. 저는 Eager 모드에서 Pass를 작성하는 일이 마치 연결 리스트의 삽입/삭제 문제를 푸는 것처럼 매끄러워졌다고 느낍니다.

Mu Li 선생님의 논문 독서법은 확실히 비교적 과학적이라고 느껴집니다. 글 마지막에서 한 번 더 칭찬을 보냅니다.
