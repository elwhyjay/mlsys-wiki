
# 0x6. Feishu와 cc-connect 상세 설정 튜토리얼

이 절은 최대한 "그대로 따라 쓰면 실행되는" 방식으로 작성하겠습니다. 앞에서 겪었던 함정과 물어봤던 문제를 모두 넣어 둡니다.

## 0x6.1 먼저 준비할 것들

- 장시간 온라인 상태를 유지할 로컬 머신. 이미 `Codex CLI`가 설치되어 있어야 합니다.
- 프로젝트 디렉터리. 예를 들어 제 경우는 `/Users/bbuf/workdir/Common`입니다.
- `cc-connect`
- Feishu 기업 자체 구축 애플리케이션
- Agent가 로컬에서 직접 SSH로 닿을 수 있는 머신, 예를 들어 `b200`에 접근하게 하려면 로컬 머신 자신의 SSH, Docker, 원격 alias도 먼저 설정해야 합니다.

## 0x6.2 Feishu 애플리케이션을 어떻게 만들까

Feishu Open Platform에 들어갑니다.

```text
https://open.feishu.cn/
```

그런 다음 이 순서대로 진행합니다.

1. "기업 자체 구축 애플리케이션"을 만듭니다.
2. `애플리케이션 기능 -> 봇`을 엽니다.
3. `이벤트 구독`을 열고 **장기 연결**을 선택합니다.
4. 이벤트를 추가합니다.

```text
im.message.receive_v1
```

5. `권한 관리`에서 권한을 켭니다.
6. `버전 생성`
7. `게시`

가장 놓치기 쉬운 것은 마지막 두 단계입니다. Feishu 백엔드의 많은 설정은 "이미 체크된 것처럼" 보이지만, 버전을 만들지 않고 게시하지 않으면 실제로는 적용되지 않습니다.

## 0x6.3 권한은 정확히 무엇을 체크해야 할까

먼저 1:1 봇만 실행해 보고 싶다면, 제가 실제로 연결에 성공한 최소 권한 묶음은 다음과 같습니다.

- `contact:user.base:readonly`
- `im:message.p2p_msg:readonly`
- `im:message:send_as_bot`

그룹에서 `@봇`도 동작하게 하려면 다음 두 개를 더 추가합니다.

- `im:message.group_at_msg:readonly`
- `im:message.group_msg`

필요 없는 것은 건드리지 마세요. 예를 들어 출입, 승인, 클라우드 문서, 캘린더, 회의실, 다차원 표 같은 것은 `cc-connect`와 관계가 없습니다.

가장 안정적인 방법은 Feishu 권한 검색창에 위 permission code를 그대로 검색해 하나씩 체크하는 것입니다.

## 0x6.4 자격 증명은 어디서 가져오나

게시가 완료되면 Feishu 애플리케이션 백엔드에서 다음을 얻습니다.

- `App ID`
- `App Secret`

`App ID`는 보통 다음처럼 생겼습니다.

```text
cli_xxxxxxxxxxxxxxxx
```

`App Secret`은 일련의 비밀 문자열입니다.

중요한 보안 알림이 있습니다. `App Secret`을 채팅 기록, 스크린샷, 공개 문서에 보냈다면 나중에 Feishu 백엔드에서 한 번 다시 생성하는 것이 좋습니다.

## 0x6.5 config.toml 작성 방법

`cc-connect`의 전역 설정 파일은 기본적으로 다음 위치에 있습니다.

```text
~/.cc-connect/config.toml
```

Feishu 봇 하나만 먼저 붙인다면, 다음 템플릿에서 시작할 수 있습니다.

```toml
language = "zh"

[log]
level = "info"

[stream_preview]
enabled = false

[[projects]]
name = "common"
quiet = true

[projects.agent]
type = "codex"

[projects.agent.options]
work_dir = "/Users/bbuf/workdir/Common"
mode = "yolo"

[[projects.platforms]]
type = "feishu"

[projects.platforms.options]
app_id = "cli_xxxxxxxxxxxxxxxx"
app_secret = "your_feishu_app_secret"
progress_style = "compact"
```

저처럼 같은 프로젝트에 Feishu 봇 두 개를 붙이고 싶다면, 뒤에 `[[projects.platforms]]` 블록을 하나 더 추가하면 됩니다.

```toml
language = "zh"

[log]
level = "info"

[stream_preview]
enabled = false

[[projects]]
name = "common"
quiet = true

[projects.agent]
type = "codex"

[projects.agent.options]
work_dir = "/Users/bbuf/workdir/Common"
mode = "yolo"

[[projects.platforms]]
type = "feishu"

[projects.platforms.options]
app_id = "cli_first_bot"
app_secret = "first_secret"
progress_style = "compact"

[[projects.platforms]]
type = "feishu"

[projects.platforms.options]
app_id = "cli_second_bot"
app_secret = "second_secret"
progress_style = "compact"
```

이 몇 가지 설정은 아래처럼 이해하면 됩니다.

- `work_dir`
  Codex가 실제로 작업하는 프로젝트 디렉터리입니다.
- `mode = "yolo"`
  원격 개발에는 거의 필수입니다. 켜지 않으면 네트워크, SSH, shell 실행, Docker 진입 같은 동작이 sandbox에 막히기 쉽습니다.
- `quiet = true`
  thinking과 도구 과정을 기본적으로 적게 보냅니다.
- `[stream_preview].enabled = false`
  스트리밍 중간 결과 미리보기를 끕니다. Feishu에 긴 중간 결과가 계속 올라오지 않습니다.
- `progress_style = "compact"`
  진행 상황 표시를 최대한 간결하게 합니다.

## 0x6.6 왜 여기서는 yolo인가

이 지점은 예전에 제가 직접 함정을 밟았습니다.

Feishu 안의 Codex가 로컬 머신이 원래 접근할 수 있는 대상에 실제로 접근하게 하려면, 예를 들어 다음과 같은 작업입니다.

- `ssh b200`
- 로컬 Docker 진입
- benchmark 실행
- 외부 네트워크에서 모델이나 도구 다운로드

`suggest`로는 기본적으로 부족하고, 다음으로 바꿔야 합니다.

```toml
mode = "yolo"
```

이것이 나중에 제가 `b200`을 통과시킨 핵심 중 하나였습니다.

대가도 명확합니다. 권한이 더 커지므로 결과와 diff, benchmark를 더 잘 봐야 하며, 이를 black box처럼 다루면 안 됩니다.

## 0x6.7 cc-connect 시작 방법

설정을 작성한 뒤 새 터미널에서 바로 실행합니다.

```bash
cc-connect -config ~/.cc-connect/config.toml
```

정상적으로 시작되는 것을 확인한 뒤 Feishu에서 봇에게 개인 메시지를 보내 테스트할 수 있습니다.

첫 번째 연결 테스트는 이렇게 하는 것을 추천합니다.

1. 봇에게 개인 메시지로 일반 문장을 보냅니다.
2. 정상적으로 답장하는지 확인합니다.
3. 아주 가벼운 로컬 명령을 하나 시켜 봅니다.
4. 문제가 없음을 확인한 뒤 SSH, benchmark, Docker 진입을 시킵니다.

나중에 수동 시작을 생략하고 싶다면 macOS 백그라운드 서비스로 따로 만들 수 있지만, 수동 시작이 첫 배포와 문제 해결에는 가장 쉽습니다.

## 0x6.8 새 세션을 시작하고 현재 작업을 끝내는 방법

Feishu에서 가장 자주 쓰는 slash 명령은 다음과 같습니다.

- **`/new`**
  새 세션을 엽니다.
- **`/new b200-debug`**
  새 세션을 열고 이름을 붙입니다.
- **`/stop`**
  현재 실행 중인 작업을 멈춥니다.
- **`/list`**
  기존 세션을 봅니다.
- **`/switch <id>`**
  특정 이전 세션으로 전환합니다.
- **`/current`**
  현재 세션이 무엇인지 봅니다.
- **`/history 20`**
  최근 20개 메시지를 다시 봅니다.
- **`/mode yolo`**
  세션 단위로 권한 모드를 바꿉니다.
- **`/reasoning high`**
  reasoning 강도를 높입니다.
- **`/quiet`**
  세션 단위로 중간 과정 출력을 줄입니다.
- **`/help`**
  도움말을 확인합니다.

"완전히 새로운 작업 라인"을 시작하고 싶다면, 저는 바로 이어서 다음을 보내는 것을 추천합니다.

```text
/stop
/new
```

앞 명령은 현재 작업을 멈추고, 뒤 명령은 새 context로 전환합니다.

## 0x6.9 현재 진행 상황 확인 방법

이 부분은 나중에는 slash 명령에 크게 의존하지 않게 되었고, 그냥 자연어로 직접 보냈습니다.

```text
현재 진행 상황, 이미 완료한 수정, 실행 중인 검증, 다음 계획을 요약해 주세요.
```

이런 질문은 본질적으로 "모델이 context를 바탕으로 요약하게 하는 것"이므로, 자연어가 전용 명령을 외우는 것보다 대체로 더 편합니다.

## 0x6.10 token 사용량과 남은 quota는 어떻게 보나

이 점도 따로 물어본 적이 있습니다.

결론은, 제가 쓰는 `cc-connect + Feishu + Codex` 경로에서는 현재 token 사용량이나 남은 quota를 보기 위한 안정적으로 사용할 수 있는 공식 slash 명령을 검증하지 못했습니다.

그러니 먼저 가정하고 가짜 명령을 외우지는 마세요.

```text
/usage
/quota
```

적어도 제 설정에서는 이것들을 안정적인 진입점으로 의존하지 않습니다.

## 0x6.11 Feishu에 Bash와 도구 로그가 잔뜩 올라오지 않게 하는 방법

이것도 실전에서 특히 중요합니다.

아무 수렴 설정을 하지 않으면 Feishu에 다음이 대량으로 올라옵니다.

- `도구 #35: Bash`
- `rg`
- `sed`
- thinking 과정

휴대폰에서는 거의 볼 수 없습니다.

제가 마지막에 사용한 것은 이 세 가지 설정입니다.

```toml
quiet = true

[stream_preview]
enabled = false

progress_style = "compact"
```

이렇게 설정하면 Feishu가 "터미널 미러"가 아니라 "결과 패널"에 더 가까워집니다.

## 0x6.12 한 프로젝트에 여러 Feishu 봇을 붙이는 방법

이것은 사실 매우 간단합니다. 프로젝트 설정을 두 벌 복사할 필요가 없습니다.

같은 `[[projects]]` 아래에 여러 개의 블록을 쓰면 됩니다.

```toml
[[projects.platforms]]
type = "feishu"

[projects.platforms.options]
app_id = "cli_bot_a"
app_secret = "secret_a"
progress_style = "compact"

[[projects.platforms]]
type = "feishu"

[projects.platforms.options]
app_id = "cli_bot_b"
app_secret = "secret_b"
progress_style = "compact"
```

그러면 두 Feishu 봇이 동시에 같은 프로젝트 디렉터리와 같은 Codex agent를 가리키게 할 수 있습니다.

이 방식은 특히 다음에 적합합니다.

- 봇 하나는 자신이 쓰고, 하나는 동료에게 제공
- 하나는 안정적인 진입점, 하나는 실험용 진입점
- 하나는 주 작업 실행, 하나는 debug 전용

## 0x6.13 프로젝트 디렉터리 아래 AGENTS.md도 함께 보강할 수 있다

Agent가 이 프로젝트에서 `cc-connect`로 메시지를 보내거나 cron을 추가하는 방법을 자연스럽게 알게 하고 싶다면, 프로젝트 루트에 `AGENTS.md`를 둘 수 있습니다.

저는 지금 이런 생각으로 두고 있습니다.

```md
# cc-connect Integration

## Scheduled tasks (cron)
cc-connect cron add ...

## Send Message To Current Chat
cc-connect send -m "short message"
```

Feishu 연결의 필수 전제 조건은 아니지만, 이 경로를 장기간 사용하는 사람에게는 꽤 도움이 됩니다.

## 0x6.14 마지막으로 잊기 쉬운 함정 몇 가지

- 권한과 이벤트를 체크한 뒤 반드시 `버전 생성` + `게시`를 해야 합니다.
- Feishu에 과정 정보가 너무 많이 올라오면 먼저 `quiet`, `stream_preview`, `progress_style`을 확인합니다.
- 봇은 연결되는데 SSH가 안 되면, 아직 `suggest`이고 `yolo`가 아닌지 먼저 봅니다.
- `App Secret`을 채팅, 스크린샷, 공개 문서에 보냈다면 나중에 재설정하세요.
- 개인 메시지는 되는데 그룹에서는 안 되면, 그룹 채팅 관련 권한과 애플리케이션 게시 상태를 먼저 다시 확인합니다.
