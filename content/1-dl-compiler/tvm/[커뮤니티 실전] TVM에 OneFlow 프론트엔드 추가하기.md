# 【커뮤니티 실전】TVM에 OneFlow 프론트엔드 추가하기

# 0x0. 배경

작년 Summer Code 시기에 마침 TVM에 입문하기 시작했고(지금도 여전히 입문 단계라 의미 있는 작업을 한 적은 없지만), 마침 OneFlow에서 일하게 되어 TVM에 OneFlow 프론트엔드를 추가해 보고 싶었다. 그러나 안타깝게도 Summer Code에서 이 프로젝트를 발의한 후 시스템의 후보자 선발 BUG로 인해 적합한 후보자를 선정하지 못했다. 이후 필자가 사적으로 이 프로젝트에 지원했던 두 번째 후보자인 후자쿠이(Hu Jiakui) 군에게 연락하여 OneFlow에 인턴으로 와서 1~2개월 동안 이 일을 완성해 줄 수 있는지 물었고, 그가 동의하여 인턴 기간 중에 초기 버전을 만들어냈다. 후자쿠이(Hu Jiakui) 군의 기여에 감사드린다.

이 초기 버전을 바탕으로 필자가 일련의 코드 리팩터링, BUG 수정, 문서 작성, 더 많은 operator와 모델 변환을 지원하도록 한 끝에 비교적 안정적인 상태에 도달했다. 그래서 이 글에서는 이 작은 프로젝트를 진행한 경험과 기술적 세부 사항을 공유하고자 하며, 오픈 소스 프로젝트를 해보고 싶지만 아직 해본 적 없는 독자들에게 참고가 되기를 바란다.

# 0x1. 효과

![이미지](images/img_01.png)문서 미리보기

여기에는 전체 화면을 캡처하지 않았으니, 공식 사이트 https://tvm.apache.org/docs/how_to/compile_models/from_oneflow.html 에서 확인할 수 있다.

Python API 미리보기:

![이미지](images/img_02.png)Python API 미리보기

현재 ResNet, MobileNet, ShuffleNet, GhostNet, YOLOv3, SRGAN, Vision Transformer 등 다양한 비전 모델을 성공적으로 지원하고 있으니, 많은 분들의 사용을 환영한다. 사용 방법은 https://tvm.apache.org/docs/how_to/compile_models/from_oneflow.html 를 참고하면 된다.

# 0x2. PR 진행 과정

아래 스크린샷은 이 작업의 PR 흐름을 보여준다. 4월에 기본 기능의 PR이 머지된 이후로는 기본적으로 Op 지원과 모델 지원, 그리고 BUG 수정이 주된 작업이었다.

![이미지](images/img_03.png)PR 진행 과정

PR 과정에서 열정적으로 도움을 주신 TVM 커뮤니티의 **「@masahi」** 께 깊이 감사드린다.

# 0x3. 기술적 세부 사항

사실 별로 자세히 이야기할 만한 세부 사항은 없고, 기본적으로는 OneFlow의 IR을 하나하나 순회하면서 Op 단위로 변환하는 것이다. 이전에 이미 TVM의 ONNX 프론트엔드 기술적 세부 사항을 소개한 적이 있다: [【从零开始学TVM】三，基于ONNX模型结构了解TVM的前端](<https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247494189&idx=1&sn=a0f646dac459d3a47019f6f9bd0db545&scene=21#wechat_redirect>) , 따라서 여기서는 비슷한 세부 사항을 반복하지 않겠다. 여기서는 OneFlow 프론트엔드 구현에서 조금 특별한 세부 사항 몇 가지만 나열한다.

  * shape 및 타입 추론: 입력 Tensor에 대해 shape 및 타입 추론을 수행하며, 기능은 TVM에서 제공한다. 코드는 다음을 참고: https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/common.py#L524-L532
  * shape 및 타입 승격(promotion): concat과 같은 Op의 경우, 입력 Tensor가 서로 다른 타입이거나 서로 다른 shape이고 승격 원칙에 부합한다면, 가장 높은 타입이나 고정된 shape으로 승격한 다음 Relay IR로 변환할 수 있다. 구체적인 구현은 다음을 참고: https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/oneflow.py#L95-L112
  * OneFlow Op의 입력 Tensor 이름을 가져올 때의 무작위성 제거: 이 문제는 OneFlow의 IR이 Protobuf로 직렬화되어 있어, 어떤 Node를 순회할 때 가져오는 입력의 이름이 무작위가 되어 BUG를 유발할 수 있기 때문이다. 이 문제를 해결하기 위해, 이름을 가져올 때 순서가 있는 리스트를 유지했다. 구체적인 구현은 다음을 참고: https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/oneflow.py#L1756-L1765
  * Relay IR 입력의 결정: OneFlow IR에서 입력 노드 이름은 모두 `_input.`이라는 특징을 포함하고 있으므로, 이 특징을 통해 Relay IR의 입력 노드를 결정할 수 있다. 구체적인 구현: https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/oneflow.py#L1816-L1840



입력과 Op 변환 규칙이 결정되고 나면, 전체 Relay IR을 손쉽게 구성할 수 있어 더 이상 말할 만한 것이 없다고 느껴진다. TVM 프론트엔드의 구체적인 세부 사항을 더 알고 싶다면 위의 링크를 참고하기 바란다.

# 0x4. 정리

이 글에서는 필자와 후자쿠이(Hu Jiakui) 군이 TVM에 OneFlow 프론트엔드를 추가한 작업을 간략히 소개했다. 오픈 소스 프로젝트를 해보고 싶지만 아직 해본 적 없는 독자들에게 참고가 되기를 바란다.

# 0x5. 참고 링크

  * https://github.com/apache/tvm
  * https://github.com/Oneflow-Inc/oneflow



  

