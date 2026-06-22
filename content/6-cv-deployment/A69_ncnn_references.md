# [추론 배포] NCNN 참고 자료

> 원문: https://zhuanlan.zhihu.com/p/449765328

## NCNN 참고 자료

![NCNN](images/img_001.jpg)

최근 TNN, MNN, NCNN, ONNXRuntime 사용 기록을 정리하고 있다. 나중에 같은 문제를 다시 만났을 때 빨리 확인하기 위한 자료 모음이다. 관련 C++ 추론 예제는 `Lite.AI.ToolKit`에 있다.

## NCNN 공식 저장소 자료

- NCNN GitHub
- NCNN Release
- ncnn/wiki 공식 문서
- ncnn custom layer 추가
- NCNN 각 layer의 parameter 설정
- NCNN param 파일 의미 해석
- ncnn model 구조 수동 최적화

## NCNN 컴파일 튜토리얼

- [1] NCNN 컴파일 사용(1)
- [2] ncnn 공식 컴파일 튜토리얼
- [3] macOS OpenMP 설치
- [4] ncnn `CMakeLists.txt`에서 OpenMP 지원 수정
- [5] macOS OpenMP 설치

```bash
mkdir build && cd build
cmake .. -DNCNN_OPENMP=OFF -DNCNN_BENCHMARK=OFF -DNCNN_BUILD_EXAMPLES=ON -DNCNN_SHARED_LIB=ON
make -j16
make install
```

## NCNN 학습 참고

- [1] NCNN source 읽기 노트 시리즈
- [2] NCNN LSTM 구현
- [3] NCNN source 읽기와 이해
- [4] NCNN model 구조 수동 최적화
- [5] ncnn 초보자가 꼭 봐야 할 자료
- [6] ncnn 사용 상세(1): PC
- [7] ncnn 사용 상세(2): Android
- [8] ncnn param/bin 파일 분석
- [9] ncnn 컴파일 사용
- [10] ncnn-openmp 비동기 처리 방법 수정
- [11] 2021년 ncnn은 어떻게 발전했나
- [12] ncnn `Extractor`를 올바르게 사용하는 방법

## NCNN 참고 사례

- [1] ncnn-android-squeezenet
- [2] ncnn-android-yolov5
- [3] ncnn-android-styletransfer
- [4] ncnn-chineseocr-lite
- [5] ncnn-pc-mobilessd
- [6] ncnn-android-mobilessd + cmake
- [7] yolov5-onnx-ncnn-android
- [8] Android에서 yolov5 실행

## NCNN 새 op와 layer 생성

- [1] ncnn 공식 문서: 새 layer 생성
- [2] NCNN custom layer 공식 튜토리얼

## NCNN source 자료

- [1] NCNN memory alignment source 읽기
- [2] `ncnn::Mat`
- [3] NCNN source 해석: Mat class(1)
- [4] OpenCV와 함께 ncnn 사용

## Android NCNN

- [1] ncnn-for-android + cmake GitHub
- [2] ncnn Android algorithm 이식
- [3] ncnn-for-android 상세 튜토리얼, 필독

## NCNN 공식 release resource

- [1] ncnn-android-releases

## NCNN open source project

- [1] awesome-ncnn
- [2] YOLOX PyTorch model -> ONNX -> ncnn -> C++ 실행 추론

## NCNN 삽질 기록

- [1] `axis=0`일 때 `Squeeze`/`Unsqueeze`/`Gather`/`Shape` 미지원
- [2] `onnx2ncnn` 여러 미지원 연산 문제
- [3] NCNN 지원 op 목록
- [4] channel 변환 방법, 예: `ncwh` -> `nchw`
- [5] `Expand not supported yet!`
- [6] `unsupported flatten axis 0`

## 7. 좋은 프로젝트

- [1] YOLOX PyTorch model -> ONNX -> ncnn -> C++ 실행 추론

## 8. NCNN 문제 해결 기록

- [1] `Expand not supported yet!`
- [2] `unsupported flatten axis 0`

지난 글 모음은 계속 업데이트한다.

