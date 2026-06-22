# x86 gemm optimize src

- 컴파일: `make`
- 컴파일 후 실행 시 리다이렉트 명령만 추가하면 행렬 크기와 GFlops가 기록된 txt 파일을 얻을 수 있습니다. 예: `./unit_test >> now.txt`. 단, `now.txt`는 미리 직접 생성하고 쓰기 권한이 있어야 합니다.
- 현재 테스트한 CPU 모델은 Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz 12 Cores 입니다.

