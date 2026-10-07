#!/usr/bin/env bash
# Sync source markdown files into the wiki content tree via mapping.tsv.
# Run this whenever source repos are updated, then rebuild with mkdocs.
#
# Editing model ("graduation"):
#   - Files listed in mapping.tsv are MANAGED: sync overwrites them from source.
#   - Files NOT in mapping.tsv are LEFT ALONE: hand-edited/graduated articles and
#     new articles you wrote directly in content/ are never touched or deleted.
#   - To graduate an article (stop syncing, edit it freely): delete its line from
#     mapping.tsv. To add a brand-new article: just create it under content/.
#   - Sync only ever deletes a content file if its mapping line is removed AND you
#     confirm it as an orphan (see the orphan report at the end). It never deletes
#     automatically.
#   - unclassified/ 는 소스 3곳의 미러(tvm_mlir_learn / optim_cuda / leetcuda)만
#     매번 재생성한다. 그 밖에 직접 넣어둔 폴더는 보존된다.
set -e

WIKI="$(cd "$(dirname "$0")" && pwd)"
CONTENT="$WIKI/content"
MAPPING="$WIKI/mapping.tsv"

TVM="$HOME/tvm_mlir_learn"
CUDA="$HOME/how-to-optim-algorithm-in-cuda/korean"
LEETCUDA="$HOME/leetcuda/blogs/ko"

# ── helpers ────────────────────────────────────────────────────────────────

# Several source articles can map into the same content/ directory, and many of
# them name their figures generically (images/img_01.png). Copied flat, the last
# mapping line wins and every earlier article renders the wrong figure.
#
# So: when two articles landing in the same directory provide the same figure
# name with DIFFERENT content, both move their images under images/<source-dir>/
# and their own links are rewritten to match. The decision is computed up front
# from mapping.tsv and the sources only — never from what happens to be in
# content/ already — so the tree is the same no matter how often sync runs or
# what was deleted in between.
build_ns_set() {
  CONTENT="$CONTENT" MAPPING="$MAPPING" TVM="$TVM" CUDA="$CUDA" LEETCUDA="$LEETCUDA" \
  python3 - "$NS_SET" <<'PY'
import os, re, sys, hashlib, collections

CONTENT=os.environ["CONTENT"]
ROOT={"tvm":os.environ["TVM"],"cuda":os.environ["CUDA"],"leetcuda":os.environ["LEETCUDA"]}
IMG=(".png",".jpg",".jpeg",".gif",".svg",".webp")

REF=re.compile(r'!\[[^\]]*\]\(([^)\s]+)|<img[^>]*src="([^"]+)"')
def refs(md):
    out=set()
    try: text=open(md, encoding="utf-8", errors="replace").read()
    except OSError: return out
    for m in REF.finditer(text):
        u=(m.group(1) or m.group(2) or "").split("#")[0].split("?")[0].strip()
        if not u or u.startswith(("http","data:","/")): continue
        out.add(u[2:] if u.startswith("./") else u)
    return out

# (destination dir, figure name) -> [source article dir, ...]
claims=collections.defaultdict(list)
for line in open(os.environ["MAPPING"], encoding="utf-8"):
    line=line.rstrip("\n")
    if not line or line.startswith("#"): continue
    f=line.split("\t")
    if len(f)<3 or f[0] not in ROOT: continue
    src=os.path.join(ROOT[f[0]], f[1])
    if not os.path.isfile(src): continue
    dst=os.path.join(CONTENT, f[2])
    if f[2].endswith("/") or os.path.isdir(dst):
        dst=os.path.join(dst.rstrip("/"), os.path.basename(src))
    dd=os.path.dirname(dst)
    sd=os.path.dirname(src)
    # only figures the article links to - those are the ones cp_file copies,
    # so those are the only ones that can collide in the destination folder
    for rel in refs(src):
        if rel.lower().endswith(IMG) and os.path.isfile(os.path.join(sd, rel)):
            claims[(dd, rel)].append(sd)

def sha(path):
    h=hashlib.sha1()
    with open(path,"rb") as fh:
        for chunk in iter(lambda: fh.read(1<<16), b""): h.update(chunk)
    return h.digest()

# A name claimed by one article is fine. Claimed by several with identical bytes
# is also fine - they overwrite each other with the same file. Only differing
# bytes force every claimant to move under its own subdirectory.
need=set()
for (dd, rel), dirs in claims.items():
    if len(dirs)<2: continue
    digests={}
    for sd in dirs:
        try: digests[sd]=sha(os.path.join(sd, rel))
        except OSError: pass
    if len(set(digests.values()))>1:
        need.update(digests)
open(sys.argv[1],"w",encoding="utf-8").write("".join(d+"\n" for d in sorted(need)))
PY
}

# Every image an article links to, as written in the markdown (markdown syntax
# and raw <img src>, local paths only).
refd_images() {
  grep -oE '!\[[^]]*\]\([^)]+\)|<img[^>]*src="[^"]+"' "$1" 2>/dev/null \
    | grep -oE '\(([^)]+)\)|src="[^"]+"' \
    | sed -E 's/^\(//; s/\)$//; s/^src="//; s/"$//; s/[#?].*$//' \
    | grep -viE '^(https?:|data:|/)' \
    | grep -iE '\.(png|jpe?g|gif|svg|webp)$' \
    | sed -E 's|^\./||' \
    | sort -u
}

cp_file() {
  local src="$1" dst="$2"
  if [[ "$dst" == */ ]] || [ -d "$dst" ]; then
    dst="${dst%/}/$(basename "$src")"
  fi
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"

  # A raw <img src="img/x.png"> on its own line renders as-is: mkdocs rewrites
  # relative paths in markdown images but not in HTML attributes, so with
  # directory URLs the browser asks for /<article>/img/x.png and gets a 404.
  # Turn those into markdown so the path gets rewritten. Only whole-line tags,
  # to leave <img> inside an HTML block alone.
  sed -i '' -E 's|^<img src="([^"]*)"[^>]*>[[:space:]]*$|![](\1)|' "$dst"

  local src_dir img_dst slug ns=0
  src_dir="$(dirname "$src")"
  img_dst="$(dirname "$dst")"
  slug="$(basename "$src_dir")"
  grep -qxF "$src_dir" "$NS_SET" && ns=1

  if [ "$ns" = 1 ]; then
    # ](images/x.png  ->  ](images/<slug>/x.png   and the same for src="..."
    # Strip an existing <slug>/ first, so a source copy that already carries the
    # namespaced path (e.g. pushed back by reverse-sync.sh) is rewritten to the
    # same thing instead of images/<slug>/<slug>/x.png.
    local sub
    for sub in images img; do
      sed -i '' \
        -e "s|](${sub}/${slug}/|](${sub}/|g" -e "s|src=\"${sub}/${slug}/|src=\"${sub}/|g" \
        -e "s|](${sub}/|](${sub}/${slug}/|g" -e "s|src=\"${sub}/|src=\"${sub}/${slug}/|g" \
        "$dst"
    done
    NS_COUNT=$((NS_COUNT + 1))
    printf '  [NS] %s -> <images|img>/%s/\n' "${dst#$CONTENT/}" "$slug" >> "$NS_LOG"
  fi

  # Copy only the figures this article actually links to. Copying each source
  # folder wholesale used to drop thousands of unused files into content/.
  local rel out
  while IFS= read -r rel; do
    [ -n "$rel" ] && [ -f "$src_dir/$rel" ] || continue
    out="$rel"
    if [ "$ns" = 1 ]; then
      case "$rel" in
        images/*) out="images/$slug/${rel#images/}" ;;
        img/*)    out="img/$slug/${rel#img/}" ;;
      esac
    fi
    mkdir -p "$img_dst/$(dirname "$out")"
    cp "$src_dir/$rel" "$img_dst/$out"
  done < <(refd_images "$src")
}

mk_index() {
  local dir="$1" title="$2" desc="$3"
  mkdir -p "$dir"
  printf '# %s\n\n%s\n' "$title" "$desc" > "$dir/index.md"
}

# ── managed-file tracking ──────────────────────────────────────────────────
# We no longer wipe content/ wholesale. Instead we record every destination
# that mapping.tsv manages, refresh those, and report anything else as orphan.

MANAGED="$(mktemp)"
trap 'rm -f "$MANAGED"' EXIT

# ── section index pages ────────────────────────────────────────────────────

mk_index "$CONTENT/1-dl-compiler"                    "딥러닝 컴파일러"        "TVM, MLIR, Triton, torch.compile 관련 글 모음"
mk_index "$CONTENT/1-dl-compiler/tvm"                "TVM"                   "TVM 시리즈 (zerodl 1~10) 및 튜토리얼"
mk_index "$CONTENT/1-dl-compiler/mlir"               "MLIR"                  "MLIR 시리즈 (zerodl 11~20) 및 응용"
mk_index "$CONTENT/1-dl-compiler/triton"             "Triton"                "OpenAI Triton DSL 및 커널 예제"
mk_index "$CONTENT/1-dl-compiler/torch-compile"      "torch.compile"         "TorchDynamo, AOTAutograd, TorchInductor"
mk_index "$CONTENT/2-cuda-kernels"                   "CUDA 커널 프로그래밍"   "CUDA 기초부터 고급 최적화까지"
mk_index "$CONTENT/2-cuda-kernels/basics"            "기초 & 메모리"          "메모리 계층, 점유율, 벡터 접근"
mk_index "$CONTENT/2-cuda-kernels/operators"         "연산자 구현"            "LayerNorm, Softmax, Cross Entropy 등"
mk_index "$CONTENT/2-cuda-kernels/gemm"              "GEMM 최적화"           "행렬 곱 단계별 최적화"
mk_index "$CONTENT/2-cuda-kernels/isa-ptx"           "GPU ISA & PTX"         "PTX 명령어, ldmatrix, 인라인 어셈블리"
mk_index "$CONTENT/3-cutlass-cute"                   "CUTLASS / CuTe"        "NVIDIA CUTLASS 라이브러리 및 CuTe DSL"
mk_index "$CONTENT/3-cutlass-cute/cute-core"         "CuTe 핵심"             "Layout, Tensor, Copy, MMA, Swizzle"
mk_index "$CONTENT/3-cutlass-cute/gemm-impl"         "GEMM 구현"             "CuTe 기반 GEMM 구현 시리즈"
mk_index "$CONTENT/3-cutlass-cute/cutlass-deep-dive" "CUTLASS 심층 분석"     "CUTLASS 2.x/3.x 내부 구조"
mk_index "$CONTENT/4-llm-inference"                  "LLM 추론 최적화"        "Attention, 양자화, 프레임워크, 분산 추론"
mk_index "$CONTENT/4-llm-inference/attention"        "Attention"             "FlashAttention, KV 캐시, Flash Decoding"
mk_index "$CONTENT/4-llm-inference/quantization"     "양자화"                "INT4, FP8, GPTQ, AWQ"
mk_index "$CONTENT/4-llm-inference/frameworks/sglang"       "SGLang"         "SGLang 추론 프레임워크"
mk_index "$CONTENT/4-llm-inference/frameworks/vllm"         "vLLM"           "vLLM 구조 및 최적화"
mk_index "$CONTENT/4-llm-inference/frameworks/tensorrt-llm" "TensorRT-LLM"  "TensorRT-LLM 활용"
mk_index "$CONTENT/4-llm-inference/distributed"      "분산 & 병렬"           "NCCL, 텐서 병렬, 파이프라인 병렬"
mk_index "$CONTENT/4-llm-inference/moe"              "MoE 추론"              "Mixture-of-Experts 추론 최적화"
mk_index "$CONTENT/4-llm-inference/training"         "학습 & RL"             "RL 파이프라인, verl, Rollout 가속"
mk_index "$CONTENT/4-llm-inference/infra"            "추론 인프라"            "대규모 서비스, 모델 병렬, 배포 실전"
mk_index "$CONTENT/4-llm-inference/frameworks"       "추론 프레임워크"        "SGLang, vLLM, TensorRT-LLM"
mk_index "$CONTENT/5-diffusion-inference"            "Diffusion 추론"         "Diffusion 모델 가속 및 배포"
mk_index "$CONTENT/6-cv-deployment"                  "CV 배포"               "ONNX, NCNN, MNN, TNN 배포"
mk_index "$CONTENT/7-hardware-arch"                  "하드웨어 & 아키텍처"    "GPU 마이크로아키텍처, TensorCore, TMA"
mk_index "$CONTENT/8-pytorch-ecosystem"              "PyTorch 생태계"         "FSDP, torchao, 프로파일링"
mk_index "$CONTENT/9-papers-lectures"                "논문 & 강의"            "논문 리딩 노트, CUDA-MODE 강의 시리즈"
mk_index "$CONTENT/9-papers-lectures/papers"         "논문"                  "TVM/MLIR/컴파일러 논문 해설"
mk_index "$CONTENT/9-papers-lectures/cuda-mode-lectures" "CUDA-MODE 강의"    "CUDA-MODE 강의 시리즈 번역"

# ── main sync from mapping.tsv ─────────────────────────────────────────────

echo "=== Syncing from mapping.tsv ==="

copied=0
missing=0
NS_COUNT=0
NS_LOG="$(mktemp)"
NS_SET="$(mktemp)"
trap 'rm -f "$MANAGED" "$NS_LOG" "$NS_SET"' EXIT
build_ns_set

while IFS=$'\t' read -r source src_rel wiki_path; do
  # skip comments and blank lines
  [[ "$source" =~ ^#.*$ || -z "$source" ]] && continue

  case "$source" in
    tvm)      src_root="$TVM" ;;
    cuda)     src_root="$CUDA" ;;
    leetcuda) src_root="$LEETCUDA" ;;
    *)        echo "  [WARN] unknown source: $source"; continue ;;
  esac

  src="$src_root/$src_rel"
  dst="$CONTENT/$wiki_path"

  if [ ! -f "$src" ]; then
    echo "  [MISSING] $source/$src_rel"
    missing=$((missing + 1))
    continue
  fi

  # Resolve the final destination (handle dir / trailing-slash targets) so we
  # can record exactly which file mapping.tsv manages.
  if [[ "$dst" == */ ]] || [ -d "$dst" ]; then
    dst="${dst%/}/$(basename "$src")"
  fi
  printf '%s\n' "$dst" >> "$MANAGED"

  cp_file "$src" "$dst"
  copied=$((copied + 1))
done < "$MAPPING"

echo "  Copied: $copied files"
if [ "$NS_COUNT" -gt 0 ]; then
  echo "  Namespaced: $NS_COUNT article(s) whose figure names collided"
  cat "$NS_LOG"
fi
[ "$missing" -gt 0 ] && echo "  Missing in source: $missing files (check mapping.tsv)"

# ── unclassified: mirror source structure, skip already-mapped files ────────

echo ""
echo "=== Unclassified (source mirror, git-ignored) ==="

UNCLASSIFIED="$WIKI/unclassified"
mkdir -p "$UNCLASSIFIED"

# 미러 디렉토리만 관리 대상. unclassified/ 아래 직접 넣어둔 자료
# (colfax_blog, reed_blog 처럼 소스 3곳이 아닌 것)는 sync가 건드리지 않는다.
MIRRORED=""

# Build set of mapped src paths for each source (for fast lookup)
# Format: "source\tsrc_rel"
mapped_srcs=$(grep -v '^#' "$MAPPING" | grep -v '^[[:space:]]*$' | awk -F'\t' '{print $1 "\t" $2}')

mirror_unclassified() {
  local src_root="$1" label="$2" src_key="$3"
  local dst_root="$UNCLASSIFIED/$label"
  rm -rf "$dst_root"
  MIRRORED="$MIRRORED $dst_root"

  find "$src_root" -name "*.md" ! -name "_raw.md" | while read f; do
    rel="${f#$src_root/}"
    if ! printf '%s' "$mapped_srcs" | grep -qF "${src_key}	${rel}"; then
      dst="$dst_root/$rel"
      mkdir -p "$(dirname "$dst")"
      cp "$f" "$dst"
    fi
  done
}

mirror_unclassified "$TVM"      "tvm_mlir_learn" "tvm"
mirror_unclassified "$CUDA"     "optim_cuda"     "cuda"
mirror_unclassified "$LEETCUDA" "leetcuda"       "leetcuda"

unc_count=$(find $MIRRORED -name "*.md" | wc -l | tr -d ' ')
echo "  $unc_count files → $UNCLASSIFIED"
echo "  (원본 폴더 구조 그대로 미러. mapping.tsv에 추가 후 재실행하면 위키에 반영됨)"

# ── orphan report ───────────────────────────────────────────────────────────
# content/*.md files that mapping.tsv does NOT manage. These are either:
#   (a) articles you graduated/edited by hand or wrote directly — KEEP, or
#   (b) stale output from a mapping line you renamed/removed — delete by hand.
# Sync never deletes these automatically; it only lists them so you can decide.

echo ""
echo "=== Orphans (in content/, not managed by mapping.tsv) ==="
sort -u "$MANAGED" > "$MANAGED.sorted"
orphans=0
while IFS= read -r f; do
  [ "$(basename "$f")" = "index.md" ] && continue
  if ! grep -qxF "$f" "$MANAGED.sorted"; then
    echo "  [ORPHAN] ${f#$CONTENT/}"
    orphans=$((orphans + 1))
  fi
done < <(find "$CONTENT" -name "*.md")
rm -f "$MANAGED.sorted"
if [ "$orphans" -eq 0 ]; then
  echo "  none"
else
  echo "  $orphans orphan(s). 졸업/직접작성 글이면 그대로 두고, 옛 산출물이면 직접 삭제하세요."
fi

# ── images referenced but not tracked ──────────────────────────────────────
# content/ 아래 이미지는 .gitignore 로 가려져 있다. 소스에서 딸려오는 수천 개가
# git status 를 덮기 때문이다. 그래서 글이 실제로 쓰는 이미지가 커밋에서 조용히
#빠질 수 있다. 여기서 그런 파일을 찾아 git add -f 명령까지 만들어 준다.

if [ -d "$WIKI/.git" ] && command -v git >/dev/null 2>&1; then
  echo ""
  echo "=== 참조되는데 git에 없는 이미지 ==="
  REFD="$(mktemp)"; TRACKED="$(mktemp)"
  find "$CONTENT" -name '*.md' -print0 | while IFS= read -r -d '' f; do
    d="$(dirname "$f")"
    grep -oE '!\[[^]]*\]\([^)]+\)|<img[^>]*src="[^"]+"' "$f" 2>/dev/null \
      | grep -oE '\(([^)]+)\)|src="[^"]+"' \
      | sed -E 's/^\(//; s/\)$//; s/^src="//; s/"$//' \
      | grep -viE '^(https?:|data:|/)' \
      | grep -iE '\.(png|jpe?g|gif|svg|webp)$' \
      | while IFS= read -r rel; do
          rel="${rel%%#*}"; rel="${rel%%\?*}"
          [ -f "$d/$rel" ] && printf '%s\n' "${d#$WIKI/}/$rel"
        done
  done | sort -u > "$REFD"
  git -C "$WIKI" ls-files -- content | sort -u > "$TRACKED"
  need=$(comm -23 "$REFD" "$TRACKED")
  if [ -z "$need" ]; then
    echo "  none"
  else
    printf '%s\n' "$need" | sed 's/^/  [NEEDS-ADD] /'
    echo ""
    echo "  content/ 이미지는 .gitignore 로 가려져 있으므로 -f 로 추가한다:"
    printf '%s\n' "$need" | sed "s/.*/    git add -f '&'/"
  fi
  rm -f "$REFD" "$TRACKED"
fi

# ── summary ────────────────────────────────────────────────────────────────

echo ""
echo "Sync complete."
echo ""
echo "Counts per section (content/):"
for d in "$CONTENT"/[0-9]*/; do
  count=$(find "$d" -name "*.md" ! -name "index.md" | wc -l | tr -d ' ')
  echo "  $(basename "$d"): $count files"
done
