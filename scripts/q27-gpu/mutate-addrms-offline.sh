#!/usr/bin/env bash
# Compile-only: mutate 5120-wide add-RMSNorm eps, cubin bytes must change, then restore.
set -euo pipefail
ROOT="${Q27_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
NVCC="${NVCC:-nvcc}"
CU=zig/kernels/cuda/q27_rows.cu
OUT="${TMPDIR:-/tmp}/q27-mut"
mkdir -p "$OUT"
FLAGS=(-O3 --fmad=false --ftz=false --prec-div=true --prec-sqrt=true
  -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
  --expt-relaxed-constexpr -std=c++20
  -gencode=arch=compute_86,code=sm_86)
OLD='__uint_as_float(0x394ccccdu), eps);'
NEW='__uint_as_float(0x394ccccdu), eps + 1.f);'
"$NVCC" -cubin "${FLAGS[@]}" -o "$OUT/good.cubin" "$CU"
python3 -c "from pathlib import Path; p=Path('$CU'); t=p.read_text();
assert t.count('''$OLD''')==1, 'expected one 5120-width eps'
p.write_text(t.replace('''$OLD''', '''$NEW''', 1))"
"$NVCC" -cubin "${FLAGS[@]}" -o "$OUT/mut.cubin" "$CU"
python3 -c "from pathlib import Path; p=Path('$CU'); t=p.read_text();
p.write_text(t.replace('''$NEW''', '''$OLD''', 1))"
python3 - << PY
from pathlib import Path
g = Path("$OUT/good.cubin").read_bytes()
m = Path("$OUT/mut.cubin").read_bytes()
n = sum(1 for a,b in zip(g,m) if a!=b) + abs(len(g)-len(m))
print(f"MUT_CUBIN_BYTES good={len(g)} mut={len(m)} mismatches={n}")
print("MUT_CUBIN_EXIT:0" if n else "MUT_CUBIN_EXIT:1")
raise SystemExit(0 if n else 1)
PY
