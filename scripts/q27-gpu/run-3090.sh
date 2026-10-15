#!/usr/bin/env bash
# RTX 3090 receipt. Never nvidia-smi -pl. Target 350 W; 225 W is the temporary test cap.
set -uo pipefail
ROOT="${Q27_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
ZIG="${ZIG:-zig}"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the 3090}"
: "${Q27_CUBINS:?path to qwen27-exl3-sm86-dflash2-complete-set (aot.json + cubins/)}"
: "${Q27_WEIGHTS:?path to the EXL3 27B folder}"
: "${Q27_PROMPTS:?json object with a story array of token ids}"
unset TENSORFOLD_CUDA_KERNELS
LOG="${Q27_LOG:-$ROOT/scratch/q27-3090.log}"
mkdir -p "$(dirname "$LOG")"
CU=zig/kernels/cuda/q27_rows.cu
OLD='__uint_as_float(0x394ccccdu), eps);'
NEW='__uint_as_float(0x394ccccdu), eps + 1.f);'
BUILD_CMD=("$ZIG" build -Doptimize=ReleaseFast -Dnvcc=nvcc -Dsm=86 -Dstrip=true)
BIN_TEST=zig-out/bin/tf-cuda-test
BIN_STORY=zig-out/bin/tf-qwen27-cuda
BIN_COMMIT=zig-out/bin/BUILD_COMMIT

wipe_bins() { rm -f "$BIN_TEST" "$BIN_STORY" "$BIN_COMMIT"; }
restore_cu() { if [ -f "$CU.good" ]; then mv -f "$CU.good" "$CU"; fi; }
trap restore_cu EXIT

print_bins() {
  echo "BIN_COMMIT:$(cat "$BIN_COMMIT")"
  echo "BIN_TF_CUDA_TEST:$(stat -c '%s %Y' "$BIN_TEST")"
  echo "BIN_TF_QWEN27_CUDA:$(stat -c '%s %Y' "$BIN_STORY")"
}

need_bins() {
  if [ ! -x "$BIN_TEST" ] || [ ! -x "$BIN_STORY" ] || [ ! -f "$BIN_COMMIT" ]; then
    echo "STALE_OR_MISSING_BIN"
    exit 1
  fi
  print_bins
}

build_release() {
  local tag=$1
  wipe_bins
  echo "${BUILD_CMD[*]}"
  if ! "${BUILD_CMD[@]}"; then
    echo "${tag}:1"
    wipe_bins
    exit 1
  fi
  if [ ! -x "$BIN_TEST" ] || [ ! -x "$BIN_STORY" ]; then
    echo "BIN_MISSING after ${tag}"
    echo "${tag}:1"
    wipe_bins
    exit 1
  fi
  git rev-parse HEAD >"$BIN_COMMIT"
  echo "${tag}:0"
  print_bins
}

run() { echo "$1"; eval "$1"; echo "${2:-STATUS}:$?"; }

{
  echo "START $(date -Is)"
  echo "COMMIT $(git rev-parse HEAD)"
  echo "POWER_TARGET:350 W"
  nvidia-smi --query-gpu=index,name,memory.total,power.limit --format=csv,noheader
  echo "KERNELS_UNSET:${TENSORFOLD_CUDA_KERNELS-}"
  build_release BUILD_EXIT
  need_bins
  run '"$BIN_TEST" q27-glue' GLUE_EXIT
  run '"$BIN_TEST" q27-cubin "$Q27_CUBINS"' CUBIN_EXIT
} >"$LOG" 2>&1

cp "$CU" "$CU.good"
if [ "$(grep -c -F "$OLD" "$CU")" != 1 ]; then
  echo "MUTATE_SKIP: expected one 5120-width eps" >>"$LOG"
else
  python3 -c "from pathlib import Path; p=Path('$CU'); t=p.read_text(); p.write_text(t.replace('''$OLD''', '''$NEW''', 1))"
  {
    echo "MUTATION: eps + 1.f inside tf_q27_add_rmsnorm only"
    build_release MUT_BUILD_EXIT
    need_bins
    run '"$BIN_TEST" q27-glue' MUT_GLUE_EXIT
  } >>"$LOG" 2>&1
  restore_cu
  echo "RESTORED" >>"$LOG"
  {
    build_release GOOD_BUILD_EXIT
    need_bins
    run '"$BIN_TEST" q27-glue' PASS_GLUE_EXIT
  } >>"$LOG" 2>&1
fi
need_bins
(
  export CUDA_VISIBLE_DEVICES=
  need_bins
  "$BIN_TEST" q27-glue
  echo "SKIP_EXIT:$?"
) >"${LOG%.log}-skip.log" 2>&1
echo "SKIP_LOG:${LOG%.log}-skip.log" >>"$LOG"
{
  echo "STORY_START $(date -Is)"
  need_bins
  "$BIN_STORY" run "$Q27_WEIGHTS" --prompts "$Q27_PROMPTS" --max-tokens 128
  echo "STORY_EXIT:$?"
  echo "END $(date -Is)"
} >>"$LOG" 2>&1

echo "Wrote $LOG and ${LOG%.log}-skip.log"
echo "Paste BUILD_EXIT, GLUE_EXIT, every CUBIN <kernel> rows=<n> mismatches=<m> line, CUBIN_EXIT,"
echo "every ELEM attn_q / DIFF attn_q line, MUT_GLUE_EXIT (must be nonzero), PASS_GLUE_EXIT,"
echo "the skip log, STORY_EXIT, the story token sha, BIN_COMMIT, and the nvidia-smi power.limit line."
