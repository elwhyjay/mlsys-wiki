# [추론 배포] ONNX 참고 자료

> 원문: https://zhuanlan.zhihu.com/p/449773663

## ONNX 참고 자료

![](images/v2-317c3ea743fca73641a81cc518a3a328_1440w.jpg)

최근 TNN, MNN, NCNN 사용 기록을 정리하고 있다. 나중에 같은 문제를 다시 만났을 때 빨리 확인하기 위한 자료 모음이다. 관련 C++ 추론 예제는 `Lite.AI.ToolKit`에 있다.

## 참고 문헌

- [1] OpenCV DNN module로 ONNX model 호출
- [2] `onnx-simplifier` 사용
- [3] ONNXRuntime C++ interface source build
- [4] ONNXRuntime Ubuntu source build
- [5] YOLOv3의 ONNX/OpenCV DNN inference acceleration 비교
- [6] Neural network acceleration engine 비교 조사
- [7] ONNX-ONNXRuntime model deployment

## Torch를 ONNX로 변환할 때 주의점

- [1] PyTorch ONNX export 삽질 가이드
- [2] PyTorch -> ONNX 변환 주의점

## ONNX 공식 자료

- [1] ONNXRuntime C/C++/Java full stack
- [2] Homebrew로 ONNXRuntime 설치
- [3] Homebrew로 ONNXRuntime 1.6.0 build script
- [4] ONNX model quantization

## ONNX model quantization

- [1] ONNX `quantize_static` 공식 사례

## ONNX 세부 주의점

- [1] ONNXRuntime C++ interface에서 주의해야 할 큰 함정
- [2] ONNX model conversion opset version 문제

## ONNX source 읽기

- [1] ONNXRuntime source 해석: engine 실행 과정 개요
- [2] PyTorch ONNX `operator_export_type` 설정
- [3] ONNXRuntime과 PyTorch 연결 방법 모음
- [4] ONNXRuntime design philosophy

## ONNX operator 지원

- [1] ONNX supported operator
- [2] ONNX `Squeeze` operator 문제
- [3] ONNX update `Squeeze` op
- [4] PyTorch tensor dimension 확장 방법
- [5] PyTorch tensor expansion 방법
- [6] ONNX `Squeeze` issue
- [7] PyTorch ONNX `operator_export_type` 설정
- [8] ONNXRuntime과 PyTorch 연결 방법 모음

## ONNX Model Zoo

- [1] ONNX model zoo 공식 repository

## Netron 시각화 도구

- Netron GitHub

```bash
brew install netron
```

- Netron 사용 방법

## ONNX 파일 형식

- [1] Model conversion과 ONNX format 분석
- [2] `onnx.proto`
- [3] ONNX 구조 분석
- [4] ONNX 분할

나중에 시간이 나면 계속 업데이트한다.

지난 글 모음도 계속 업데이트한다.

