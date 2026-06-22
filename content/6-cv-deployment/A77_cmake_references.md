# [에세이][CMake] CMake 참고 자료

> 원문: https://zhuanlan.zhihu.com/p/449779892

## CMake 참고 자료

![](images/v2-bffb2bf11422c5ef7d8949788114c2ab.png)

## CMake 참고 문헌

- [1] `cmake-configure_file` 사용법: C++에서 CMake variable 사용. 유용하다.
- [2] CMake 사용자 정의 compile option `cmakedefine`.
- [3] CMake 자동 macro definition `cmakedefine01`.
- [4] CLion config file `CMakeLists.txt` 문법 소개와 예제.
- [5] CMake의 `install` 명령 역할.
- [6] CMake의 `GenerateExportHeader` 응용. 매우 유용하다.
- [7] CMake의 `add_custom_target` 사용법.
- [8] `add_custom_command`와 `add_custom_target`.
- [9] CMake의 `find_package` 검색 경로.
- [10] CMake의 `target_compile_definitions` 함수.
- [11] CMake `GenerateExportHeader` 공식 문서.
- [12] CMake compile option 설정: `add_compile_options`와 `CMAKE_CXX_FLAGS`.
- [13] CMake `CMAKE_TOOLCHAIN_FILE`.
- [14] CMake의 `source_group` 사용법.
- [15] CMake `source_group` 사용 시 주의할 문제.
- [16] `target_compile_definitions`와 `target_compile_options` 사용법.
- [17] CMake의 centralized include.
- [18] CMake의 `set`과 `option` 차이.
- [19] CMake의 `install` 명령.
- [20] `set_properties`와 `target_set_properites`의 차이.

## CMake community 학습 자료

- [1] `cmake-cookbook`.
- [2] `awesome-cmake`.
- [3] `modern-cmake-examples`.

## CMake property list

- [1] `cmake-properties`.
- [2] CMake compile에서 `target_link_libraries`의 `PRIVATE`, `PUBLIC`, `INTERFACE` 의미.
- [3] `PRIVATE`, `PUBLIC`, `INTERFACE`의 의미와 생성된 so library의 관계.

## C++ 참고 문헌

- [1] 컴파일러의 predefined macro: AVX/AVX2 등.
- [2] 서로 다른 compiler에서 자주 쓰는 predefined macro.

## AVX instruction set

- [1] AVX instruction set.
- [2] AVX2 instruction set 해석.
- [3] compile 시 사용하는 instruction set.

## Compiler 참고

- [1] compiler built-in macro의 역할.
- [2] MSDN 문서: compiler predefined macro와 compile option의 관계.

```text
Microsoft-specific predefined macros
MSVC supports these additional predefined macros.

__ATOM__ Defined as 1 when the /favor:ATOM compiler option is set and the compiler target is x86 or x64. Otherwise, undefined.

__AVX__ Defined as 1 when the /arch:AVX, /arch:AVX2, or /arch:AVX512 compiler options are set and the compiler target is x86 or x64. Otherwise, undefined.

__AVX2__ Defined as 1 when the /arch:AVX2 or /arch:AVX512 compiler option is set and the compiler target is x86 or x64. Otherwise, undefined.

__AVX512BW__ Defined as 1 when the /arch:AVX512 compiler option is set and the compiler target is x86 or x64. Otherwise, undefined.

__AVX512CD__ Defined as 1 when the /arch:AVX512 compiler option is set and the compiler target is x86 or x64. Otherwise, undefined.

__AVX512DQ__ Defined as 1 when the /arch:AVX512 compiler option is set and the compiler target is x86 or x64. Otherwise, undefined.

__AVX512F__ Defined as 1 when the /arch:AVX512 compiler option is set and the compiler target is x86 or x64. Otherwise, undefined.

__AVX512VL__ Defined as 1 when the /arch:AVX512 compiler option is set and the compiler target is x86 or x64. Otherwise, undefined.

_CHAR_UNSIGNED Defined as 1 if the default char type is unsigned. This value is defined when the /J (Default char type is unsigned) compiler option is set. Otherwise, undefined.

__CLR_VER Defined as an integer literal that represents the version of the Common Language Runtime (CLR) used to compile the app. The value is encoded in the form Mmmbbbbb, where M is the major version of the runtime, mm is the minor version of
```

- compiler가 지원하는 instruction과 predefined macro 확인.

```bash
➜  ~ gcc -march=native -dM -E - < /dev/null | grep -i SSE
#define __SSE2_MATH__ 1
#define __SSE2__ 1
#define __SSE3__ 1
#define __SSE4_1__ 1
#define __SSE4_2__ 1
#define __SSE_MATH__ 1
#define __SSE__ 1
#define __SSSE3__ 1
➜  ~ gcc -march=native -dM -E - < /dev/null | grep -i AVX
#define __AVX2__ 1
#define __AVX__ 1
```

## CMake와 Make의 차이

- [1] `cmake --build`와 `make`의 차이.

## CMake 학습 서적

- [1] Modern CMake tutorial.

나중에 시간이 있으면 계속 갱신한다.

이전 글 모음도 계속 갱신한다.
