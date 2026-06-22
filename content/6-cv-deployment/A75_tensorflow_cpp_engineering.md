# [추론 배포] TensorFlow C++ 엔지니어링 지식점

> 원문: https://zhuanlan.zhihu.com/p/449788027

### TensorFlow C++ 엔지니어링 지식점

![TensorFlow C++](images/img_001.jpg)

최근 algorithm inference engineering, 즉 MNN/NCNN/TNN/ONNXRuntime/TF C++ 등을 정리하고 있다. 나중에 같은 문제를 만났을 때 빨리 찾기 위한 기록이다. 관련 C++ 추론 예제는 `Lite.AI.ToolKit`에 있다.

## 참고 blog

- TensorFlow C++ engineering 잡기록, 좋은 case
- TensorFlow C++ API로 online prediction service 구축
- macOS에서 TensorFlow 1.11.0 source build와 C++ API 사용
- macOS에서 TensorFlow C++ compile 중 생기는 문제
- macOS TensorFlow 1.9.0 C++ 설치
- macOS에서 C++로 TensorFlow 호출
- C++ version TensorFlow compile

## TensorFlow C++ training-to-deployment series

- TensorFlow C++ environment setup
- TensorFlow C++ training-to-deployment(2)
- Keras + TensorFlow C++ model deployment

## TensorFlow C++ deployment practice

- 51CTO video tutorial, 유료
- TensorFlow C++에서 pb로 inference
- TensorFlow C++ compile
- TensorFlow C++ compile version matching
- TFLite model deployment
- TFLite official demo deployment
- TFLite mobile deployment summary
- TensorFlow 1.14.1 compile
- TensorFlow 1.14 compile, 유용
- C++ custom operator tutorial
- TensorFlow C++에서 pb로 object detection 호출, 중요
- TensorFlow C++ practice source GitHub
- Training a TensorFlow graph in C++ API
- Loading a TensorFlow graph with the C++ API
- Creating a TensorFlow DNN in C++ Part 1
- TensorFlow C++ model load
- 좋은 TF C++ example, 필독
- TensorFlow 1.4 C++ compile과 API 사용
- `load_model.cc` 사례
- Tensor와 Eigen 유연하게 사용
- TF C++ official example `label_image.cc`

```cpp
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/default_device.h"
using namespace tensorflow;

int main(int argc, char* argv[]) {

    std::string graph_definition = "mlp.pb";
    Session* session;
    GraphDef graph_def;
    SessionOptions opts;
    std::vector<Tensor> outputs; // Store outputs
    TF_CHECK_OK(ReadBinaryProto(Env::Default(), graph_definition, &graph_def));

    // GPU option 설정
    graph::SetDefaultDevice("/gpu:0", &graph_def);
    opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
    opts.config.mutable_gpu_options()->set_allow_growth(true);

    // 새 session 생성
    TF_CHECK_OK(NewSession(opts, &session));

    // graph를 session에 load
    TF_CHECK_OK(session->Create(graph_def));

    // variable 초기화
    TF_CHECK_OK(session->Run({}, {}, {"init_all_vars_op"}, nullptr));

    Tensor x(DT_FLOAT, TensorShape({100, 32}));
    Tensor y(DT_FLOAT, TensorShape({100, 8}));
    auto _XTensor = x.matrix<float>();
    auto _YTensor = y.matrix<float>();

    _XTensor.setRandom();
    _YTensor.setRandom();

    for (int i = 0; i < 10; ++i) {
        TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {"cost"}, {}, &outputs));
        float cost = outputs[0].scalar<float>()(0);
        std::cout << "Cost: " <<  cost << std::endl;
        TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {}, {"train"}, nullptr));
        outputs.clear();
    }

    session->Close();
    delete session;
    return 0;
}
```

## TensorFlow C interface

- TensorFlow C interface 설치는 compile이 필요 없다
- TensorFlow 1.x C interface 사용 사례

## TF C++ official example set

`tensorflow/tensorflow/examples` 안에는 여러 C++ example이 있다.

1. multi_box detector C++
2. TF C++ official example `label_image.cc`
3. speech command C++

## TF C++ 최적화 instruction으로 compile

- How to compile TensorFlow with SSE4.2 and AVX instructions?

```bash
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 --config=monolithic -k //tensorflow:libtensorflow_cc.so
bazel build -c opt --copt=-march=native --config=monolithic -k //tensorflow:libtensorflow_cc.so
```

- MKL-DNN compile, 유용
- MKL로 TensorFlow CPU 가속
- TensorFlow C++ source compile command 해석
- TensorFlow C++ XLA optimization, 중요
- TensorFlow 1.4 XLA + JIT optimization
- XLA는 GPU에도 유용함

```text
몇 가지 실험 결과(macpro 2018):
1. MKL, XLA, JIT option과 AVX/AVX2/FMA/SSE4.2 compile instruction을 사용한 dynamic library는 성능이 불안정했다. 때로 빠르고 때로 느렸다.
2. AVX/AVX2/FMA/SSE4.2 compile instruction만 사용한 dynamic library는 성능이 비교적 안정적이고 Python보다 약간 빨랐다.
3. 작은 model은 multi-thread를 쓸 필요가 없다. single-thread면 충분하다.
```

## C++로 training

- Training TensorFlow models in C++

## 미리 compile된 dynamic library

- Linux/macOS TensorFlow C++ dynamic library collection
- Ubuntu 16.04 TF C++ source compile library

## Windows에서 TensorFlow C++ 사용 시 주의점

- Windows에서 TensorFlow 2.0 C++ dynamic library compile
- TF Windows wheel, 여러 compile library
- Windows 10 + TensorFlow + dll + lib
- Windows 환경 VS2015 Debug mode에서 TensorFlow CPU C++ source compile
- TensorFlow C++ Debug version compile
- TensorFlow debug version compile

## Session parameter 설정

```cpp
BlazeRFB::loadGraph() {
    // 1. session 생성
    tf::ConfigProto config;
    tf::GraphOptions graphOptions;
    tf::OptimizerOptions optimizerOptions;
    tf::SessionOptions sessionOptions;
    // optimizerOptions.set_global_jit_level(tf::OptimizerOptions_GlobalJitLevel_ON_1);
    // optimizerOptions.set_opt_level(tf::OptimizerOptions_Level_Level_MAX);
    optimizerOptions.set_do_constant_folding(true);
    optimizerOptions.set_do_function_inlining(true);
    optimizerOptions.set_do_common_subexpression_elimination(true);
    *graphOptions.mutable_optimizer_options() = optimizerOptions;
    *config.mutable_graph_options() = graphOptions;
    // 2. default thread 수 설정. 작은 model에서는 multi-thread가 오히려 IO 시간을 소모한다.
    // 2.1 thread 수를 1로 설정했을 때 성능이 가장 좋고 Python과 맞출 수 있다. 5ms.
    config.set_intra_op_parallelism_threads(1);
    config.set_use_per_session_threads(true);
    config.set_inter_op_parallelism_threads(1);
    sessionOptions.config = config;
    tf::Status status = tf::NewSession(sessionOptions, &session);
    TF_LOG_STATUS_ERROR(status)
    if (!status.ok()) return false;
    // 3. raw pb model을 GraphDef로 import
    status = tf::ReadBinaryProto(tf::Env::Default(), model_path, &graph_def);
    TF_LOG_STATUS_ERROR(status)
    if (!status.ok()) return false;
    // 4. raw model을 session에 load
    status = session->Create(graph_def);
    TF_LOG_STATUS_ERROR(status)
}
```

원문은 TensorFlow C++를 다룰 때 참고할 blog와 build option을 모아 둔 자료다. 특히 session thread 수, XLA/JIT/MKL 사용 여부, CPU instruction option은 model 규모에 따라 결과가 달라질 수 있으므로 직접 측정해야 한다.

