# [배포][NCNN] ONNX를 NCNN으로 변환할 때 미지원 op를 우회하는 방법

> 원문: https://zhuanlan.zhihu.com/p/451446147

한동안 글을 갱신하지 않았다. 최근 TNN, MNN, NCNN 사용 시리즈 노트를 정리하려 한다. 좋은 기억력보다 엉성한 기록이 낫다. 기억력도 좋지 않으니, 나중에 같은 구덩이에 빠졌을 때 조금 더 빠르게 빠져나오기 위한 기록이다. 현재 **80개가 넘는 C++** 추론 예제를 lib로 빌드해서 사용할 수 있게 정리해 두었다. 관심이 있으면 보면 된다. 길게 소개하지는 않는다.

프로젝트 설명:

GithubLite.AI.ToolKitA lite C++ toolkit of awesome AI models.

즉, 바로 사용할 수 있는 C++ AI 모델 도구 상자다. 평소 새 알고리즘을 공부할 때 손에 잡히는 대로 만든 것들이고, 현재 80개 이상의 인기 오픈소스 모델을 포함한다. 어느새 거의 800 star에 가까워졌다. star와 issue는 언제나 환영한다.

https://github.com/DefTruth/lite.ai.toolkit

최근 관련 글을 계속 갱신할 예정이다.

### 1. 서문

먼저 분명히 해 둔다. 어떤 inference engine을 선택할지는 완전히 개인 취향이다. 따라서 이 짧은 글은 기술 문제만 기록한다.

오늘은 SCRFD의 C++ inference를 만들려 했다. 내 습관은 보통 여러 version을 동시에 만드는 것이다. 예를 들면 MNN, NCNN, TNN, ONNXRuntime version이다. 오늘 ONNX를 NCNN으로 변환하는 과정에서 변환할 수 없는 문제가 생겼다. 여기서는 **편법 하나**만 간단히 기록한다. **정말 NCNN model file만 얻으면 되고 방식은 신경 쓰지 않는다면**, 이상한 op 때문에 변환이 막힐 때 아래 방법을 시도해 볼 수 있다.

### 2. unsupported op를 만나다

ONNX model file은 아래 repository에서 가져왔다.

https://github.com/ppogg/onnx-scrfd-flask

처음 변환할 때 만난 문제는 다음과 같았다.

```text
➜ onnx2ncnn SCRFD/scrfd_1g.onnx SCRFD/scrfd_1g.param SCRFD/scrfd_1g.bin
Shape not supported yet!
Gather not supported yet!
  # axis=0
Shape not supported yet!
Gather not supported yet!
  # axis=0
Unsupported unsqueeze axes !
Unsupported unsqueeze axes !
Shape not supported yet!
Unknown data type 0
Unsupported Resize scales and sizes are all empty!
Shape not supported yet!
Gather not supported yet!
  # axis=0
Shape not supported yet!
Gather not supported yet!
  # axis=0
Unsupported unsqueeze axes !
...
```

이게 전설의 glue op 묶음인가 싶다. 기본 절차대로 먼저 `onnxsim`을 한번 돌린다.

```bash
➜ python3 -m onnxsim scrfd_1g.onnx scrfd_1g-sim.onnx --dynamic-input-shape --input-shape 1,3,320,320
Simplifying...
Checking 0/3...
Checking 1/3...
Checking 2/3...
Ok!
```

다시 NCNN으로 변환해 본다.

```text
➜ onnx2ncnn SCRFD/scrfd_1g-sim.onnx SCRFD/scrfd_1g.param SCRFD/scrfd_1g.bin
Shape not supported yet!
Gather not supported yet!
  # axis=0
Shape not supported yet!
Gather not supported yet!
  # axis=0
Unsupported unsqueeze axes !
Unsupported unsqueeze axes !
Shape not supported yet!
Unknown data type 0
Unsupported Resize scales and sizes are all empty!
Shape not supported yet!
Gather not supported yet!
  # axis=0
Shape not supported yet!
Gather not supported yet!
  # axis=0
```

![](images/v2-9891e6baa3e069e6ef0cf5600ac935ff_1440w.jpg)

여전히 실패했다. 이 glue op들이 반란을 일으키려는 것인가. 그러면 어떻게 할까.

물론 이 시점에서 다른 글을 참고해 직접 param을 만지는 방법도 생각할 수 있다.

하지만 수동으로 NCNN model structure를 optimize할 때는 크게 두 가지 문제가 있다.

- error를 보면 unsupported op가 너무 많은 위치에 나온다. 손으로 만지는 난도가 너무 높다. 포기한다.
- 수동 작업 자체가 NCNN op에 매우 익숙해야 하고, 동시에 model structure도 잘 알아야 한다. 포기한다.

그래도 정말 NCNN model file만 얻으면 되고 방식은 상관없다면, 이상한 op 때문에 변환이 막힐 때 다른 **편법**을 쓸 수 있다.

### 3. 또 다른 편법

이 방법은 TNN을 이용한다. 맞다. TNN을 **중간상**으로 쓰는 방식이다. 목적은 사용할 수 있는 NCNN file을 얻는 것이므로, 방식이 무엇인지는 따지지 않는다.

TNN을 자주 쓰는 사람이라면 ONNX를 TNN model로 변환할 때 `--optimize` parameter를 지정할 수 있다는 것을 알고 있을 것이다. 이 parameter를 지정하면 TNN은 입력 original ONNX file에 대해 먼저 optimization을 수행한다. 이 과정에는 glue op 제거와 merge가 포함된다. 느낌상 `onnxsim`과 비슷한 작업을 하지만, inference engine 자체에 맞춰 더 최적화된 것처럼 보인다. `onnxsim` 이후에도 남아 있던 일부 glue op가 TNN의 optimize 단계에서 merge되거나 제거된다. 그 결과 중간 file인 `xxx.opt.onnx`가 생성된다. TNN은 이 optimized ONNX file로 최종 TNN model file도 계속 생성한다.

```text
/opt/TNN/tools/convert2tnn# python3 ./converter.py onnx2tnn ./tnn_models/SCRFD/scrfd_1g.onnx -o ./tnn_models/SCRFD/ -optimize -v v1.0 -align -in 1,3,320,320
----------  convert model, please wait a moment ----------
Converter ONNX to TNN Model...
Converter ONNX to TNN check_onnx_dim...
Converter ONNX to TNN check_onnx_dim...
Converter ONNX to TNN model succeed!
----------  align model (tflite or ONNX vs TNN),please wait a moment ----------
input.1: input shape of onnx and tnn is aligned!
Run tnn model_check...
----------  Congratulations!   ----------
The onnx model is aligned with tnn model
```

생성된 file은 다음과 같다.

```text
/opt/TNN/tools/convert2tnn/tnn_models/SCRFD# ls | grep opt
scrfd_1g.opt.onnx
scrfd_1g.opt.tnnmodel
scrfd_1g.opt.tnnproto
scrfd_2.5g.opt.onnx
scrfd_2.5g.opt.tnnmodel
scrfd_2.5g.opt.tnnproto
scrfd_2.5g_bnkps_shape160x160.opt.onnx
scrfd_2.5g_bnkps_shape160x160.opt.tnnmodel
scrfd_2.5g_bnkps_shape160x160.opt.tnnproto
scrfd_2.5g_bnkps_shape320x320.opt.onnx
scrfd_2.5g_bnkps_shape320x320.opt.tnnmodel
scrfd_2.5g_bnkps_shape320x320.opt.tnnproto
scrfd_500m.opt.onnx
scrfd_500m.opt.tnnmodel
scrfd_500m.opt.tnnproto
scrfd_500m_bnkps_shape160x160.opt.onnx
scrfd_500m_bnkps_shape160x160.opt.tnnmodel
scrfd_500m_bnkps_shape160x160.opt.tnnproto
scrfd_500m_bnkps_shape320x320.opt.onnx
scrfd_500m_bnkps_shape320x320.opt.tnnmodel
scrfd_500m_bnkps_shape320x320.opt.tnnproto
```

여기까지 보면 내가 말하는 trick이 무엇인지 알 수 있다. 맞다. 여기서 나온 **`xxx.opt.onnx`**를 이용해 NCNN model을 변환하는 것이다. 정말 야생적인 방법이다. 결과가 어떤지 본다.

```bash
➜  onnx2ncnn SCRFD/scrfd_1g.opt.onnx SCRFD/scrfd_1g.param SCRFD/scrfd_1g.bin
➜  onnx2ncnn SCRFD/scrfd_2.5g.opt.onnx SCRFD/scrfd_2.5g.param SCRFD/scrfd_2.5g.bin
➜  onnx2ncnn SCRFD/scrfd_2.5g_bnkps_shape160x160.opt.onnx SCRFD/scrfd_2.5g_bnkps_shape160x160.param SCRFD/scrfd_2.5g_bnkps_shape160x160.bin
➜  onnx2ncnn SCRFD/scrfd_2.5g_bnkps_shape320x320.opt.onnx SCRFD/scrfd_2.5g_bnkps_shape320x320.param SCRFD/scrfd_2.5g_bnkps_shape320x320.bin
➜  onnx2ncnn SCRFD/scrfd_500m.opt.onnx SCRFD/scrfd_500m.param SCRFD/scrfd_500m.bin
➜  onnx2ncnn SCRFD/scrfd_500m_bnkps_shape160x160.opt.onnx SCRFD/scrfd_500m_bnkps_shape160x160.param SCRFD/scrfd_500m_bnkps_shape160x160.bin
➜  onnx2ncnn SCRFD/scrfd_500m_bnkps_shape320x320.opt.onnx SCRFD/scrfd_500m_bnkps_shape320x320.param SCRFD/scrfd_500m_bnkps_shape320x320.bin
```

모두 변환에 성공했다. TNN의 `--optimize`는 `onnxsim`보다 더 inference engine에 맞춘 처리를 하는 듯하다. `ncnnoptimize`도 한번 돌려서 문제가 없는지 본다.

```text
➜  ncnnoptimize SCRFD/scrfd_1g.param SCRFD/scrfd_1g.bin SCRFD/scrfd_1g.opt.param SCRFD/scrfd_1g.opt.bin 0
fuse_convolution_activation Conv_0 Relu_1
fuse_convolution_activation Conv_4 Relu_5
fuse_convolution_activation Conv_8 Relu_9
fuse_convolution_activation Conv_12 Relu_13
fuse_convolution_activation Conv_16 Relu_17
fuse_convolution_activation Conv_20 Relu_21
fuse_convolution_activation Conv_24 Relu_25
fuse_convolution_activation Conv_28 Relu_29
fuse_convolution_activation Conv_32 Relu_33
fuse_convolution_activation Conv_36 Relu_37
fuse_convolution_activation Conv_40 Relu_41
fuse_convolution_activation Conv_44 Relu_45
fuse_convolution_activation Conv_48 Relu_49
fuse_convolutiondepthwise_activation Conv_2 Relu_3
fuse_convolutiondepthwise_activation Conv_6 Relu_7
fuse_convolutiondepthwise_activation Conv_10 Relu_11
fuse_convolutiondepthwise_activation Conv_14 Relu_15
fuse_convolutiondepthwise_activation Conv_18 Relu_19
fuse_convolutiondepthwise_activation Conv_22 Relu_23
fuse_convolutiondepthwise_activation Conv_26 Relu_27
fuse_convolutiondepthwise_activation Conv_30 Relu_31
fuse_convolutiondepthwise_activation Conv_34 Relu_35
fuse_convolutiondepthwise_activation Conv_38 Relu_39
fuse_convolutiondepthwise_activation Conv_42 Relu_43
fuse_convolutiondepthwise_activation Conv_46 Relu_47
Input layer input.1 without shape info, shape_inference skipped
Input layer input.1 without shape info, estimate_memory_footprint skipped
```

역시 순조롭다. 여기까지 보면 자연스럽게 마지막 의문이 남는다. 이 방법을 쓰려면 결국 TNN을 compile해야 하는 것 아닌가. 사실 **TNN을 compile할 필요는 없다**.

TNN 공식이 `tnn_converter` image를 제공한다. model conversion만 필요하고 중간 `xxx.opt.onnx`만 얻으면 되며, TNN으로 inference를 할 필요가 없다면 `tnn_converter` image를 바로 쓰면 된다. TNN converter 구축은 이전 글을 참고하면 된다.

나중에 시간이 있으면 계속 갱신한다.

이전 글 모음도 계속 갱신한다.
