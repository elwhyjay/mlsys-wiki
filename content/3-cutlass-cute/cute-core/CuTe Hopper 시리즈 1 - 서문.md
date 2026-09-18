# CuTe Hopper 시리즈 1 - 서문

> 원문: https://zhuanlan.zhihu.com/p/1948048832794427610

앞선 "CuTe" 시리즈 글에서는 CuTe의 핵심 개념과 추상을 중점적으로 소개했고, 이 개념과 추상을 바탕으로 **아래에서 위로(bottom-up)** 쌓아 올리는 방식으로 CuTe를 이용한 GEMM 계산의 핵심 프레임워크를 단계적으로 구축했습니다. 이 내용들은 주로 **Ampere 아키텍처**를 기준으로 설명한 것입니다. 2022년 3월, NVIDIA는 GTC에서 Ampere의 후속 하드웨어 아키텍처인 **Hopper 아키텍처**를 발표했고, 같은 해 10월에는 AI 계산 시장을 겨냥한 플래그십 제품 **H100**을 출시했습니다. 지금은 2025년 11월이며, 3년의 시간을 거치면서 이 아키텍처 기반의 칩은 이미 각 데이터센터의 연산 능력을 담당하는 핵심 제품이 되었고, 오늘날 AI 계산의 가장 중심적인 힘을 구성하고 있습니다. 이전의 Ampere 아키텍처와 비교하면 Hopper 아키텍처에는 많은 기능(feature)이 추가되었고, 그 프로그래밍 패러다임에도 일련의 변화가 생겼습니다. 이러한 feature들과 새로운 패러다임이 결합되어 핵심적인 행렬 계산 능력을 함께 구성합니다. 마침 저도 업무상의 이유로 Hopper 아키텍처 제품을 접할 기회가 있었기에 이 시리즈 글을 쓰게 되었습니다. 한편으로는 개인적인 학습과 이해를 정리하고 축적하기 위한 것이고, 다른 한편으로는 CUDA 심화의 길에서 분투하고 있는 동료 여러분과 공유하기 위한 것입니다. 이 글이 Hopper의 프로그래밍 패러다임과 구조를 명확히 정리하고, 여러분에게 도움이 되기를 바랍니다.

전체적인 구상은 대략 다섯 편으로 나누어 이 내용을 소개하는 것입니다.

1. CuTe Hopper MBarrier
2. CuTe Hopper TMA
3. CuTe Hopper WGMMA
4. CuTe Hopper 하드웨어 pipeline
5. CuTe Hopper 고효율 GEMM 구현

## 참고

- CuTe의 Layout https://zhuanlan.zhihu.com/p/661182311
- CuTe Layout의 대수적·기하학적 해석 https://zhuanlan.zhihu.com/p/662089556
- CuTe의 Tensor https://zhuanlan.zhihu.com/p/663093816
- CuTe의 MMA 추상 https://zhuanlan.zhihu.com/p/663092747
- CuTe의 Copy 추상 https://zhuanlan.zhihu.com/p/666232173
- CuTe의 간단한 GEMM 구현 https://zhuanlan.zhihu.com/p/667521327
- CuTe의 GEMM pipeline https://zhuanlan.zhihu.com/p/665082713
- CuTe의 Swizzle https://zhuanlan.zhihu.com/p/671419093
- CuTe의 고효율 GEMM 구현 https://zhuanlan.zhihu.com/p/675308830
- https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c
- 여기서는 CUDA 고성능 계산에 일정한 학습 난이도가 있음을 의미합니다
