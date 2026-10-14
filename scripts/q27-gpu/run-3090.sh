#!/usr/bin/env bash
# RTX 3090 receipt: glue, cubin vs captured Triton, add_rms mutation, skip, story. No --kernels.
set -u
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
run() { echo "$1"; eval "$1"; echo "${2:-STATUS}:$?"; }

{
  echo "START $(date -Is)"
  echo "COMMIT $(git rev-parse HEAD)"
  nvidia-smi --query-gpu=index,name,memory.total,power.limit --format=csv,noheader
  echo "KERNELS_UNSET:${TENSORFOLD_CUDA_KERNELS-}"
  run '"$ZIG" build -Doptimize=ReleaseFast -Dnvcc=nvcc -Dsm=86 -Dstrip=true' BUILD_EXIT
  run './zig-out/bin/tf-cuda-test q27-glue' GLUE_EXIT
  run './zig-out/bin/tf-cuda-test q27-cubin "$Q27_CUBINS"' CUBIN_EXIT
} >"$LOG" 2>&1

cp "$CU" "$CU.good"
if [ "$(grep -c -F "$OLD" "$CU")" != 1 ]; then
  echo "MUTATE_SKIP: expected one 5120-width eps" >>"$LOG"
else
  python3 -c "from pathlib import Path; p=Path('$CU'); t=p.read_text(); p.write_text(t.replace('''$OLD''', '''$NEW''', 1))"
  {
    echo "MUTATION: eps + 1.f inside tf_q27_add_rmsnorm only"
    run '"$ZIG" build -Doptimize=ReleaseFast -Dnvcc=nvcc -Dsm=86 -Dstrip=true' MUT_BUILD_EXIT
    run './zig-out/bin/tf-cuda-test q27-glue' MUT_GLUE_EXIT
  } >>"$LOG" 2>&1
  mv "$CU.good" "$CU"
  echo "RESTORED" >>"$LOG"
  {
    run '"$ZIG" build -Doptimize=ReleaseFast -Dnvcc=nvcc -Dsm=86 -Dstrip=true' GOOD_BUILD_EXIT
    run './zig-out/bin/tf-cuda-test q27-glue' PASS_GLUE_EXIT
  } >>"$LOG" 2>&1
fi
(
  export CUDA_VISIBLE_DEVICES=
  ./zig-out/bin/tf-cuda-test q27-glue
  echo "SKIP_EXIT:$?"
) >"${LOG%.log}-skip.log" 2>&1
echo "SKIP_LOG:${LOG%.log}-skip.log" >>"$LOG"
{
  echo "STORY_START $(date -Is)"
  ./zig-out/bin/tf-qwen27-cuda run "$Q27_WEIGHTS" --prompts "$Q27_PROMPTS" --max-tokens 128
  echo "STORY_EXIT:$?"
  echo "END $(date -Is)"
} >>"$LOG" 2>&1

echo "Wrote $LOG and ${LOG%.log}-skip.log"
echo "Paste BUILD_EXIT, GLUE_EXIT, every CUBIN <kernel> rows=<n> mismatches=<m> line, CUBIN_EXIT,"
echo "MUT_GLUE_EXIT (must be nonzero), PASS_GLUE_EXIT, the skip log, STORY_EXIT, and the story token sha."
