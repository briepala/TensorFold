# sm_86 HMMA notes for Qwen3.8-27B tree and block attention

Triton `tl.dot` on Ampere emits `HMMA.16816.F32.BF16`, which is PTX
`mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32` (NVIDIA PTX ISA, matrix multiply-accumulate;
Ampere tuning guide, tensor-core m16n8k16). Each instruction is a 16x8 f32 tile from a k=16
strip of bf16 A and B. The captured cubins (`_shared` hash `953263b9…`, `_block_attention`
hash `90f7e399…`) start a k-strip from `RZ` then chain the destination registers; D=256 is
256/16 = 16 k-steps. Block attention (D=128) uses the same opcode (176 HMMA in the L=16 cubin).

Ampere tensor cores do not document an f32 add tree that a host `fma` loop can replay. A
matching Zig model would have to land in the same change as `q27_tree.cu` `tile_rows` or
`q27-glue` breaks. Sequential `__fmaf_rn` dots stay; compare f32 accumulators, not the bf16
store. `treeSmem(256)` remains 65792 (two 64-key bf16 tiles plus 64 int flags) in one
file-scope dynamic buffer `q27_tree_smem`.

Observed on Gabriele's RTX 3090 at commit `c200b461`, GPU 1 at 350 W (cubin vs Triton, not HMMA-rewritten):
`tree_p512` mismatches 2/4/17/169 at rows 1/3/17/64; `tree_p4096` row 64 is 625;
`block_attn` is 1/3/2/2. Attention prep and drafter prep are 0 at those widths.
Those tree and block counts are the remaining sequential-dot vs `tl.dot` gap.
