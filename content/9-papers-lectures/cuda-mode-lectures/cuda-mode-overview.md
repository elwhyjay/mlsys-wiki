# cuda-mode

CUDA-MODE 강의 학습 노트와 코드, 그리고 관련 기술 블로그 번역이다.

## 디렉터리 구조

```
cuda-mode/
├── code/              # [코드] 강의 관련 CUDA 실험 코드
│   ├── YHs_Sample/    #   CUDA 샘플링 예제 코드
│   └── cudabmk/       #   GPU 마이크로 benchmark 도구(cudabmk)
├── slides/            # [슬라이드] 강의 PPT / PDF 강의 자료
├── lectures/          # [노트] CUDA-MODE 각 강의 노트(Lecture 1~77+)
├── blog-translations/ # [노트] CUDA 관련 블로그 번역
├── cute-dsl/          # [노트] CuTe DSL 학습 노트
├── lei-mao-blogs/     # [노트] Lei Mao CUDA 시리즈 블로그 전재(28편)
├── practice/          # [노트] 강의 과제 실전 연습
└── tech-notes/        # [노트] GPU 기술 주제별 노트
```

## 내용 설명

| 하위 디렉터리 | 설명 |
|--------|------|
| [code/](code/) | YHs CUDA 예제 + cudabmk GPU 마이크로 benchmark 도구 |
| [slides/](slides/) | 강의 원본 PPT/PDF(Lecture 2/7/8/10/16/17/20/29 등)|
| [lectures/](lectures/) | Lecture 1-77+ 각 강의 노트로, Profiling, GEMM, Flash Attention, Triton, NCCL 등을 다룬다 |
| [blog-translations/](blog-translations/) | GPU Assembly, PTX, TMA, CuTe Layout, Prefix Sum, RMSNorm 등 블로그 번역 |
| [cute-dsl/](cute-dsl/) | CUTLASS 4.x CuTe DSL 문서 번역과 학습 노트 |
| [lei-mao-blogs/](lei-mao-blogs/) | Lei Mao의 CUDA 시리즈 블로그: memory, shared memory, tensor core, ldmatrix 등 |
| [practice/](practice/) | 첫 강의 과제 실전(상)(하) |
| [tech-notes/](tech-notes/) | GPU 메모리 발전, 비동기 계산, nsight, Green Context 등 기술 주제 |
