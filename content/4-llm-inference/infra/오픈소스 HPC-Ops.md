# 오픈소스 HPC-Ops

> 원문: https://zhuanlan.zhihu.com/p/1998322408797533516

최근 저희는 LLM Inference를 대상으로 하는 고성능 연산자 라이브러리([https://github.com/Tencent/hpc-ops](https://github.com/Tencent/hpc-ops))를 오픈소스로 공개했습니다. 이 라이브러리는 주로 대규모 언어 모델의 핵심 연산자인 Attention과 FusedMoE를 최적화한 것입니다. Attention 부분은 Paged Prefill과 Decode 기능을 제공하며, 동시에 BF16과 FP8 두 가지 데이터 타입을 지원합니다. FusedMoE 부분은 정적 PerTensor quantization과 동적 BlockWise quantization을 지원하고, 병렬화 모드로는 TP와 EP 모드를 지원합니다. 또한 MoE 안의 GroupGEMM과 Activation을 독립된 모듈로 분리해 각각의 호출 인터페이스도 함께 제공하며, 데이터 타입은 FP8입니다.

kernel의 개발과 최적화는 모두 중국 내 추론 주력 카드 기종인 H20에서 진행했으며, 결과 측면에서도 현재 저희가 공개적으로 구할 수 있는 최고 수준의 결과를 어느 정도 넘어설 수 있습니다.

구현 과정에서는 가능한 한 가장 소박한 CuTe 작성 방식을 따랐고, 지나친 추상화와 캡슐화를 피해 스케줄링 파이프라인이 명확하게 드러나도록 했습니다.

앞으로도 지속적으로 유지보수하고 개선해 나가고자 합니다. 많은 관심과 참여로 더 나은 AI Infra 역량을 함께 만들어 가면 좋겠습니다.

마지막으로 수고해 준 동료들에게 감사드리며, 그들의 앞날이 더욱 넓게 열리기를 바랍니다.

[@shaochangxu](https://www.zhihu.com/people/960941fb9285a7a183bf8892c5af6c6f)

[@weishengying](https://www.zhihu.com/people/c75f3234d7fccddfb6e951efc46c9839)

[@程前](https://www.zhihu.com/people/146b334e99204b25641fa0fb2f9d1a3f)

[@FrankJ](https://www.zhihu.com/people/0d595c22024e3be220a1526787645178)

[@薛扬](https://www.zhihu.com/people/21de5e0aaa1c80107eb781d285e1e65c)

[대규모 모델 Infra의 새로운 돌파구! 텐센트 훈위안이 LLM 추론 연산자 라이브러리를 오픈소스로 공개, 추론 throughput 30% 향상](https://zhuanlan.zhihu.com/p/1999441526355420475?share_code=10zPWXFxlTCXN&utm_psn=1999452590216328350)
