#!/usr/bin/env bash
# reverse-sync.sh — push hand-edited wiki files BACK into their source repos.
#
# Why: sync.sh copies source -> content/ and will overwrite any wiki edit to a
# managed file. When you hand-edit a wiki copy (e.g. redraw diagrams), list its
# wiki_path in reverse.tsv and run this to mirror the edit wiki -> source, so the
# next forward sync.sh becomes a no-op for that file instead of reverting it.
#
# Scope: only files explicitly listed in reverse.tsv (opt-in). Each listed
# wiki_path must exist in mapping.tsv; that mapping resolves the source location.
#
# Safety:
#   - DRY-RUN by default. Prints what would change; writes nothing.
#   - Pass --apply to actually write. Before overwriting, the source article
#     directory is backed up to <source_dir>.bak-<timestamp> (source is a plain
#     dir, not git, so this is the only way to recover the originals).
set -e

WIKI="$(cd "$(dirname "$0")" && pwd)"
CONTENT="$WIKI/content"
MAPPING="$WIKI/mapping.tsv"
REVERSE="$WIKI/reverse.tsv"

# Same source roots as sync.sh
TVM="$HOME/tvm_mlir_learn"
CUDA="$HOME/how-to-optim-algorithm-in-cuda/korean"
LEETCUDA="$HOME/leetcuda/blogs/ko"

APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

if [ "$APPLY" -eq 1 ]; then
  echo "=== reverse-sync: APPLY mode (will write to source, with backup) ==="
else
  echo "=== reverse-sync: DRY-RUN (no changes; pass --apply to write) ==="
fi
echo ""

[ -f "$REVERSE" ] || { echo "no reverse.tsv found at $REVERSE"; exit 1; }

src_root_for() {
  case "$1" in
    tvm)      printf '%s' "$TVM" ;;
    cuda)     printf '%s' "$CUDA" ;;
    leetcuda) printf '%s' "$LEETCUDA" ;;
    *)        return 1 ;;
  esac
}

# TS for backup dir names — passed by env so the script itself stays deterministic.
TS="${REVERSE_SYNC_TS:-$(date +%Y%m%d-%H%M%S 2>/dev/null || echo backup)}"

pushed=0
skipped=0

while IFS= read -r wiki_path; do
  # skip comments / blanks
  [[ "$wiki_path" =~ ^[[:space:]]*# ]] && continue
  [[ -z "${wiki_path// }" ]] && continue
  wiki_path="${wiki_path%%$'\r'}"   # strip stray CR

  # Look up this wiki_path in mapping.tsv to find (source, src_rel).
  line="$(grep -v '^[[:space:]]*#' "$MAPPING" | awk -F'\t' -v w="$wiki_path" '$3==w {print; exit}')"
  if [ -z "$line" ]; then
    echo "  [SKIP] not found in mapping.tsv: $wiki_path"
    skipped=$((skipped + 1))
    continue
  fi

  source="$(printf '%s' "$line" | cut -f1)"
  src_rel="$(printf '%s' "$line" | cut -f2)"

  src_root="$(src_root_for "$source")" || { echo "  [SKIP] unknown source '$source' for $wiki_path"; skipped=$((skipped+1)); continue; }

  src_md="$src_root/$src_rel"
  wiki_md="$CONTENT/$wiki_path"

  if [ ! -f "$wiki_md" ]; then
    echo "  [SKIP] wiki file missing: $wiki_md"
    skipped=$((skipped + 1)); continue
  fi
  if [ ! -f "$src_md" ]; then
    echo "  [SKIP] source file missing: $src_md"
    skipped=$((skipped + 1)); continue
  fi

  src_dir="$(dirname "$src_md")"
  wiki_dir="$(dirname "$wiki_md")"
  slug="$(basename "$src_dir")"

  # sync.sh moves an article's figures under images/<slug>/ when its figure
  # names collide with another article's in the same wiki folder. That layout
  # belongs to the wiki, not the source, so strip it back out before writing:
  # otherwise the next forward sync would namespace the already-namespaced path.
  norm_md="$(mktemp)"
  sed -e "s|](images/${slug}/|](images/|g" -e "s|src=\"images/${slug}/|src=\"images/|g" \
      -e "s|](img/${slug}/|](img/|g"       -e "s|src=\"img/${slug}/|src=\"img/|g" \
      "$wiki_md" > "$norm_md"

  echo "  $wiki_path"
  echo "    wiki:   $wiki_md"
  echo "    source: $src_md"

  # --- .md diff ---
  if diff -q "$norm_md" "$src_md" >/dev/null 2>&1; then
    echo "    md:     identical"
  else
    echo "    md:     DIFFERS (wiki -> source):"
    diff "$src_md" "$norm_md" | sed 's/^/        /' | head -40
  fi

  # --- images: only the ones THIS .md references ---
  # The wiki images/ dir is shared by every article in the folder, so we must
  # not mirror it wholesale into the article-specific source dir. Extract the
  # relative image paths the wiki .md actually links (markdown + html <img>) and
  # copy only those. This naturally includes images/redrawn/*.png.
  imgs="$(grep -oE '!\[[^]]*\]\(([^)]+)\)|<img[^>]*src="[^"]+"' "$wiki_md" \
          | grep -oE '\(([^)]+)\)|src="[^"]+"' \
          | sed -E 's/^\(//; s/\)$//; s/^src="//; s/"$//' \
          | grep -vE '^(https?:|/)' \
          | sort -u)"

  to_copy=()
  while IFS= read -r rel; do
    [ -z "$rel" ] && continue
    rel="${rel%%#*}"; rel="${rel%%\?*}"   # strip #anchor / ?query
    wsrc="$wiki_dir/$rel"
    if [ ! -f "$wsrc" ]; then
      echo "    [WARN] referenced image missing in wiki: $rel"
      continue
    fi
    # wiki may hold it at images/<slug>/x.png; in source it is images/x.png
    srel="$rel"
    case "$rel" in
      images/"$slug"/*) srel="images/${rel#images/$slug/}" ;;
      img/"$slug"/*)    srel="img/${rel#img/$slug/}" ;;
    esac
    if [ ! -e "$src_dir/$srel" ] || ! cmp -s "$wsrc" "$src_dir/$srel"; then
      state="new"; [ -e "$src_dir/$srel" ] && state="changed"
      echo "    +img:   $srel ($state in source)"
      to_copy+=("$rel|$srel")
    fi
  done <<< "$imgs"

  if [ "$APPLY" -eq 1 ]; then
    backup="${src_dir}.bak-${TS}"
    if [ ! -e "$backup" ]; then
      cp -R "$src_dir" "$backup"
      echo "    backup: $backup"
    fi
    cp "$norm_md" "$src_md"
    for pair in "${to_copy[@]}"; do
      rel="${pair%%|*}"; srel="${pair#*|}"
      mkdir -p "$src_dir/$(dirname "$srel")"
      cp "$wiki_dir/$rel" "$src_dir/$srel"
    done
    echo "    -> applied (${#to_copy[@]} image(s) copied)"
  fi
  rm -f "$norm_md"
  echo ""
  pushed=$((pushed + 1))
done < "$REVERSE"

echo "Processed: $pushed file(s), skipped: $skipped"
if [ "$APPLY" -eq 0 ]; then
  echo "Dry-run only. Re-run with --apply to write changes."
fi
