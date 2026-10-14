#!/usr/bin/env bash
# Compare host IEEE vs FTZ libdevice sincos against device __nv_sinf/__nv_cosf on the 3090.
set -euo pipefail
ROOT="${Q27_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
NVCC="${NVCC:-nvcc}"
OUT="${TMPDIR:-/tmp}/q27-sinpoly"
mkdir -p "$(dirname "$OUT")"
"$NVCC" -O3 --fmad=false --ftz=false --prec-div=true --prec-sqrt=true \
  -gencode=arch=compute_86,code=sm_86 \
  -o "$OUT" "$ROOT/scripts/q27-gpu/sinpoly.cu"
echo "SINPOLY_BIN:$OUT"
"$OUT"
echo "Paste the SINPOLY n=... line and SINPOLY_EXIT."
