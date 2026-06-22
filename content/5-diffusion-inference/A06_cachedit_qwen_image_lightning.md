# cache-dit + Qwen-Image-Lightning: 3.5 steps 추론 구현

> 원문: https://zhuanlan.zhihu.com/p/1948696529180295613

**목차**
- 0x00 머리말
- 0x01 비정수 step 캐싱
- 0x02 3.5 steps 추론, 1.14x 가속

## 0x00 머리말

이 글은 cache-dit으로 Qwen-Image-Lightning을 cache 가속하여 3.5 steps 추론을 구현하는 방법을 소개한다. 먼저 증류(distillation) 모델에 캐시를 적용할 때의 고충부터.

8 steps / 4 steps 증류 모델, 특히 4 steps의 경우, 기존 캐시 가속 기법 대부분이 무용지물이 된다. 예를 들어 TaylorSeer는 step 수가 많고 feature 변화가 매끄러운 가속 시나리오에만 유효하다. 반대로 FBCache나 TeaCache 같은 방식으로 1~2 step만 cache해도 생성 이미지가 그대로 뭉개진다.

## 0x01 비정수 step 캐싱

cache-dit의 DBCache는 FnBn 형태로 compute blocks를 유연하게 구성할 수 있어, step 수가 정수가 아닌 캐싱도 가능하다. 예를 들어 Qwen-Image-Lightning에 F16B16(총 block 수 60)을 설정하면 0.5 step만 cache하는 효과를 얻을 수 있다.

## 0x02 3.5 steps 추론, 1.14x 가속

cache-dit 결과를 바로 살펴보자. 왼쪽이 Qwen-Image-Lightning 4 steps baseline, 오른쪽이 cache-dit 3.5 steps 결과로, 두 이미지는 사실상 차이가 없다.

![](images/v2-11f6b47a2886e34ae19f31bfcbced28e_1440w.png)

cache-dit 3.5 steps: 약 1.14x speedup.

## 참고

- 전체 코드: https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_qwen_image_lightning.py
