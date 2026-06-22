# [추론 배포] MNN 참고 자료

> 원문: https://zhuanlan.zhihu.com/p/449761992

# MNN 참고 자료

![](images/v2-e8c3e16a31dd937cb08b3eaf496191a2_1440w.jpg)

최근 TNN, MNN, NCNN, ONNXRuntime 사용 기록을 정리하고 있다. 나중에 같은 문제를 다시 만났을 때 빨리 확인하기 위한 자료 모음이다. 관련 C++ 추론 예제는 `Lite.AI.ToolKit`에 있다.

## 1. 참고 자료

- [1] Alibaba neural network inference framework MNN 사용 방법
- [2] MNN으로 machine learning workflow 구성
- [3] MNN framework 학습(1): 컴파일과 사용
- [4] MNN framework 학습(2): MNN으로 model 배포
- [5] FreeImage를 OpenCV 구현으로 교체
- [6] MNN quantization source 상세 해석
- [7] MNN model 구조 해석
- [8] Inference engine MNN 해석, 유용

## 2. 공식 자료

- [1] MNN Chinese README.md
- [2] MNN Chinese 문서, Yuque 문서
- [3] MNN model converter build 문서
- [4] MNN example project
- [5] MNN inference framework Android build
- [6] MNN inference framework Linux/macOS build
- [7] MNN FAQ: data type 변환
- [8] MNN JNI 공식 사례
- [9] `SkMatrix` class reference
- [10] Skia graphics engine: `SkMatrix`
- [11] Android `Matrix`, `set`/`pre`/`post` 차이
- [12] Android `Matrix`의 `pre`, `post`, `set` 이해, 유용
- [13] MNN CUDA backend
- [14] MNN 1.2.0 release notes, TorchScript 지원
- [15] MNN C++ 공식 demo

## 3. 배포 자료

- [1] PyTorch와 MNN을 결합한 embedded 배포 workflow
- [2] ONNX와 MNN으로 PyTorch model 호출, Python 및 C++
- [3] PyTorch model을 ONNX와 MNN으로 변환
- [4] MNN model output과 ONNX model output 불일치, `NC4HW4`와 `NCHW`
- [5] MNN Android deep learning model inference 삽질 기록, NCHW data type 변환
- [6] 실전 MNN: MobileNet SSD 배포, source 포함
- [7] MNN의 TensorFlow MobileNetSSD C++ 배포 flow 상세 설명
- [8] MNN 사용 상세 가이드, Python과 C++
- [9] MNN 학습 노트
- [10] MNN Tensor interface 사용: batch/channel/height/width

## 4. 모델 변환

- [1] MNN model 변환의 main flow
- [2] MNN: PyTorch에서 학습한 `.pth` model을 MNN으로 변환하고 quantization training
- [3] MNN: quantized model로 C++ inference
- [4] MNN FP16 model 변환

## 5. Source 분석

- [1] MNN inference process source 분석 노트(1): main flow
- [2] FlatBuffers 소개와 사용법
- [3] FlatBuffers C++ 사용 예제

## 6. Open source project

- [1] MNN-APPLICATIONS, 필독
- [2] mnn_example
- [3] ultraface-mnn
- [4] UltraFace-MNN-Android

## 7. Dynamic dimension 처리

- [1] MNN-Segment
- [2] MNN transformer demo
- [3] MNN dynamic dimension
- [4] multi-thread + dynamic input dims
- [5] MNN ONNX dynamic dimension

## 8. 전처리 자료

- [1] `classSkMatrix`
- [2] MNN resize 전처리
- [3] MNN image preprocessing 상세 사례
- [4] image processing module 관련 질문 두 가지
- [5] Matrix image transform 방향 반대 문제, target image(model input)에서 source image로의 변환

지난 글 모음은 계속 업데이트한다.

