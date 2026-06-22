# [추론 배포] 딥러닝 모델 변환 자료 정리

> 원문: https://zhuanlan.zhihu.com/p/449759361

## 모델 변환 도구

최근 TNN, MNN, NCNN, ONNXRuntime 사용 기록을 정리하고 있다. 나중에 같은 문제를 다시 만났을 때 빨리 확인하기 위한 자료 모음이다. 관련 C++ 추론 예제는 `Lite.AI.ToolKit`에 있다.

## 1. 참고 문헌

- [0] 여러 format model을 online으로 변환하는 사이트
- [1] `caffemodel2pytorch`
- [2] SfSNet Caffe model을 PyTorch로 변환
- [3] Caffe model을 PyTorch model로 변환
- [4] Python에서 Caffe를 호출해 prediction 수행
- [5] Caffe2: Caffemodel을 Caffe2 pb model로 변환
- [6] Python 3 Caffe 설치
- [7] Python 3 Caffe 설치, 유용
- [8] Caffe 공식 설치 문서
- [9] Caffe Python interface로 inference
- [10] Python 3.8 Caffe 설치
- [11] Mac의 임의 Python 환경에서 Caffe 설치하는 최종 튜토리얼
- [12] OpenCV DNN module 기반 Caffe model 호출
- [13] Caffe model을 ONNX model로 변환
- [14] `caffe2onnx`
- [15] `tf2onnx`
- [16] TensorFlow model을 ONNX model로 변환할 때 만난 문제
- [17] TensorFlow 2.0 환경에서 TensorFlow 1.x 코드 실행

## Torch7 model을 pth로 변환

- `convert_torch_to_pytorch`
- PyTorch JIT를 이용해 `.t7` model을 `.pth`로 변환

## ONNX model 변환

- [1] Model deployment 실패 기록: PyTorch -> ONNX 삽질 실록

지난 글 모음은 계속 업데이트한다.

