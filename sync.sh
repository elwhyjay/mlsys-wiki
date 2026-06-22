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
set -e

WIKI="$(cd "$(dirname "$0")" && pwd)"
CONTENT="$WIKI/content"
MAPPING="$WIKI/mapping.tsv"

TVM="$HOME/tvm_mlir_learn"
CUDA="$HOME/how-to-optim-algorithm-in-cuda/korean"
LEETCUDA="$HOME/leetcuda/blogs/ko"

# ── helpers ────────────────────────────────────────────────────────────────

cp_file() {
  local src="$1" dst="$2"
  if [[ "$dst" == */ ]] || [ -d "$dst" ]; then
    dst="${dst%/}/$(basename "$src")"
  fi
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
  local src_dir img_dst
  src_dir="$(dirname "$src")"
  img_dst="$(dirname "$dst")"
  # images/ or img/ subdirectory
  for img_dir in "$src_dir/images" "$src_dir/img"; do
    [ -d "$img_dir" ] && rsync -a "$img_dir/" "$img_dst/$(basename "$img_dir")/" || true
  done
  # sibling image files (e.g. ./img1.webp)
  find "$src_dir" -maxdepth 1 -type f \
    \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \
       -o -name "*.gif" -o -name "*.svg" -o -name "*.webp" \) \
    | while read img; do cp "$img" "$img_dst/"; done
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
[ "$missing" -gt 0 ] && echo "  Missing in source: $missing files (check mapping.tsv)"

# ── unclassified: mirror source structure, skip already-mapped files ────────

echo ""
echo "=== Unclassified (source mirror, git-ignored) ==="

UNCLASSIFIED="$WIKI/unclassified"
rm -rf "$UNCLASSIFIED"
mkdir -p "$UNCLASSIFIED"

# Build set of mapped src paths for each source (for fast lookup)
# Format: "source\tsrc_rel"
mapped_srcs=$(grep -v '^#' "$MAPPING" | grep -v '^[[:space:]]*$' | awk -F'\t' '{print $1 "\t" $2}')

mirror_unclassified() {
  local src_root="$1" label="$2" src_key="$3"
  local dst_root="$UNCLASSIFIED/$label"

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

unc_count=$(find "$UNCLASSIFIED" -name "*.md" | wc -l | tr -d ' ')
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

# ── summary ────────────────────────────────────────────────────────────────

echo ""
echo "Sync complete."
echo ""
echo "Counts per section (content/):"
for d in "$CONTENT"/[0-9]*/; do
  count=$(find "$d" -name "*.md" ! -name "index.md" | wc -l | tr -d ' ')
  echo "  $(basename "$d"): $count files"
done
