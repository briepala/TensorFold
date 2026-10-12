//! Our .cu kernels as fatbins built at build time with each Python extension's nvcc flags, embedded in the binary.

const std = @import("std");
const options = @import("kernel_options");

fn Blob(comptime import_name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile(import_name).*;
    };
}

/// False in host-only builds (no nvcc, no prebuilt fatbins); every image is then empty.
pub const available = options.with_kernels;

pub const gdn: []const u8 = if (available) &Blob("fatbin_gdn").bytes else &.{};
pub const gdn_prefill: []const u8 = if (available) &Blob("fatbin_gdn_prefill").bytes else &.{};
pub const probe: []const u8 = if (available) &Blob("fatbin_probe").bytes else &.{};
pub const qmm_group: []const u8 = if (available) &Blob("fatbin_qmm_group").bytes else &.{};
pub const qmm_prefill: []const u8 = if (available) &Blob("fatbin_qmm_prefill").bytes else &.{};
pub const experts: []const u8 = if (available) &Blob("fatbin_experts").bytes else &.{};
pub const experts_prefill: []const u8 = if (available) &Blob("fatbin_experts_prefill").bytes else &.{};
pub const experts_pack: []const u8 = if (available) &Blob("fatbin_experts_pack").bytes else &.{};
pub const prefill_attention: []const u8 = if (available) &Blob("fatbin_prefill_attention").bytes else &.{};
pub const scan_rows: []const u8 = if (available) &Blob("fatbin_scan_rows").bytes else &.{};
pub const nemotron_ops: []const u8 = if (available) &Blob("fatbin_nemotron_ops").bytes else &.{};
pub const lane_gemv: []const u8 = if (available) &Blob("fatbin_lane_gemv").bytes else &.{};
pub const affine4_pack: []const u8 = if (available) &Blob("fatbin_affine4_pack").bytes else &.{};
pub const fp8_experts: []const u8 = if (available) &Blob("fatbin_fp8_experts").bytes else &.{};
pub const exlr8: []const u8 = if (available) &Blob("fatbin_exlr8").bytes else &.{};
pub const exlr8_dense: []const u8 = if (available) &Blob("fatbin_exlr8_dense").bytes else &.{};
pub const nvfp4_experts: []const u8 = if (available) &Blob("fatbin_nvfp4_experts").bytes else &.{};
pub const qwen_ops: []const u8 = if (available) &Blob("fatbin_qwen_ops").bytes else &.{};
pub const exl3_linear: []const u8 = if (available) &Blob("fatbin_exl3_linear").bytes else &.{};
pub const b16: []const u8 = if (available) &Blob("fatbin_b16").bytes else &.{};
pub const q27_rows: []const u8 = if (available) &Blob("fatbin_q27_rows").bytes else &.{};
pub const q27_prep: []const u8 = if (available) &Blob("fatbin_q27_prep").bytes else &.{};
pub const q27_tree: []const u8 = if (available) &Blob("fatbin_q27_tree").bytes else &.{};
pub const sample: []const u8 = if (available) &Blob("fatbin_sample").bytes else &.{};
pub const train: []const u8 = if (available) &Blob("fatbin_train").bytes else &.{};
pub const qmmf: []const u8 = if (available) &Blob("fatbin_qmmf").bytes else &.{};
pub const prompt16: []const u8 = if (available) &Blob("fatbin_prompt16").bytes else &.{};
pub const modelopt_pack: []const u8 = if (available) &Blob("fatbin_modelopt_pack").bytes else &.{};
pub const torch_argmax: []const u8 = if (available) &Blob("fatbin_torch_argmax").bytes else &.{};
pub const torch_topk: []const u8 = if (available) &Blob("fatbin_torch_topk").bytes else &.{};
pub const torch_pointwise: []const u8 = if (available) &Blob("fatbin_torch_pointwise").bytes else &.{};
pub const torch_indexing: []const u8 = if (available) &Blob("fatbin_torch_indexing").bytes else &.{};
pub const torch_movement: []const u8 = if (available) &Blob("fatbin_torch_movement").bytes else &.{};
pub const torch_nemotron_constants: []const u8 = if (available) &Blob("fatbin_torch_nemotron_constants").bytes else &.{};
pub const nemotron_norms: []const u8 = if (available) &Blob("fatbin_nemotron_norms").bytes else &.{};
pub const nemotron_route: []const u8 = if (available) &Blob("fatbin_nemotron_route").bytes else &.{};
pub const nemotron_mamba: []const u8 = if (available) &Blob("fatbin_nemotron_mamba").bytes else &.{};
pub const nemotron_attention: []const u8 = if (available) &Blob("fatbin_nemotron_attention").bytes else &.{};
pub const nemotron_keyed: []const u8 = if (available) &Blob("fatbin_nemotron_keyed").bytes else &.{};
pub const torch_norm_rotary: []const u8 = if (available) &Blob("fatbin_torch_norm_rotary").bytes else &.{};

/// Symbols in the gdn image as cuobjdump lists them for the built fatbin (named namespace tf_gdn).
pub const gdn_symbols = struct {
    pub const replay_bf16 = "_ZN6tf_gdn13replay_kernelI13__nv_bfloat16Li8ELi4EEEvPKxiPKiiS5_iPfiii";
    pub const replay_f32 = "_ZN6tf_gdn13replay_kernelIfLi8ELi4EEEvPKxiPKiiS4_iPfiii";
};

/// Every instantiation gdn.cu exports; tests resolve each one so a wrong name cannot hide behind an unused path.
pub const gdn_variants = [_]TreeVariant{
    tree(0, 8, 4, true),  tree(1, 8, 4, false), tree(2, 8, 2, false),  tree(2, 4, 4, false),
    tree(4, 2, 4, false), tree(8, 2, 4, false), tree(16, 2, 2, false), tree(32, 2, 1, false),
};

pub const TreeVariant = struct { slots: u32, r: u32, warps: u32, chain: bool, symbol: [:0]const u8 };

fn tree(comptime slots: u32, comptime r: u32, comptime warps: u32, comptime chain: bool) TreeVariant {
    const name = std.fmt.comptimePrint("_ZN6tf_gdn11tree_kernelI13__nv_bfloat16Li{d}ELi{d}ELi{d}ELb{d}" ++
        "EEEvPKT_S4_PKS1_PKfS8_S8_PKxPKiSC_iPS1_iiiNS_7PendingIS2_EEPfSA_b", .{ slots, r, warps, @intFromBool(chain) });
    return .{ .slots = slots, .r = r, .warps = warps, .chain = chain, .symbol = name };
}

/// The tree_kernel instantiation gdn.cu's dispatch_tree launches for bf16 keys on this GPU.
pub fn treeVariant(slots: u32, streams: u32, major: c_int, minor: c_int, sms: c_int) TreeVariant {
    const wide = streams >= 2 and major == 12 and minor == 0 and sms >= 96;
    if (slots == 0) return tree(0, 8, 4, true);
    if (wide and slots <= 1) return tree(1, 8, 4, false);
    if (wide and slots <= 2) return tree(2, 8, 2, false);
    if (slots <= 2) return tree(2, 4, 4, false);
    if (slots <= 4) return tree(4, 2, 4, false);
    if (slots <= 8) return tree(8, 2, 4, false);
    if (slots <= 16) return tree(16, 2, 2, false);
    return tree(32, 2, 1, false);
}

/// The exl3_linear image's mangled names (namespace tf_exl3_linear), from cuobjdump of the sm_86 build.
pub const exl3_symbols = struct {
    pub const rot_in = "_ZN14tf_exl3_linear13rot_in_kernelEPKviPK6__halfPS2_i";
    /// linear_kernel<K2, CB, WK>: K2 in {2,3,4,5,6,8,10,12,14,16}, CB 0-2, WK 4 or 8; the template args are in the name.
    pub const linear_prefix = "_ZN14tf_exl3_linear13linear_kernelI";
    /// unpack_kernel<K2, CB>: K2 in {2,3,4,5,6,7,8,10,12,14,16}, CB 0-2.
    pub const unpack_prefix = "_ZN14tf_exl3_linear13unpack_kernelI";
};

/// The b16 image's fp16 row-kernel names (template __half), from cuobjdump of the sm_86 build.
pub const b16_f16_symbols = struct {
    pub const rows_1 = "_ZN6tf_b1610b16_kernelI6__halfLi1ELi4EEEvPKT_S4_S4_PS2_iiiS4_S5_i";
    pub const rows_4 = "_ZN6tf_b1610b16_kernelI6__halfLi4ELi8EEEvPKT_S4_S4_PS2_iiiS4_S5_i";
    pub const rows_16 = "_ZN6tf_b1610b16_kernelI6__halfLi16ELi16EEEvPKT_S4_S4_PS2_iiiS4_S5_i";
};

/// The b16 image's mangled names (namespace tf_b16), from cuobjdump of the sm_86 build.
pub const b16_symbols = struct {
    pub const bf16_rows_1 = "_ZN6tf_b1610b16_kernelI13__nv_bfloat16Li1ELi4EEEvPKT_S4_S4_PS2_iiiS4_S5_i";
    pub const bf16_rows_4 = "_ZN6tf_b1610b16_kernelI13__nv_bfloat16Li4ELi8EEEvPKT_S4_S4_PS2_iiiS4_S5_i";
    pub const bf16_rows_16 = "_ZN6tf_b1610b16_kernelI13__nv_bfloat16Li16ELi16EEEvPKT_S4_S4_PS2_iiiS4_S5_i";
    pub const bf16_prompt_32 = "_ZN6tf_b1617b16_prompt_kernelILi32EEEvPK13__nv_bfloat16S3_PS1_iiiS3_S4_i";
    pub const bf16_prompt_64 = "_ZN6tf_b1617b16_prompt_kernelILi64EEEvPK13__nv_bfloat16S3_PS1_iiiS3_S4_i";
    pub const bf16_prompt_128 = "_ZN6tf_b1617b16_prompt_kernelILi128EEEvPK13__nv_bfloat16S3_PS1_iiiS3_S4_i";
};

/// The q27 row/prep/tree images' names (namespace tf_q27), from cuobjdump of the sm_86 build.
pub const q27_symbols = struct {
    pub const add_rmsnorm = "_ZN6tf_q2711add_rmsnormEPK13__nv_bfloat16S2_S2_PS0_S3_Pffi";
    pub const gated_norm = "_ZN6tf_q2710gated_normEPK13__nv_bfloat16S2_S2_PS0_Pffii";
    pub const swiglu = "_ZN6tf_q276swigluEPK13__nv_bfloat16S2_PS0_Pfi";
    pub const gate_mul = "_ZN6tf_q278gate_mulEPK13__nv_bfloat16S2_PS0_Pfii";
    pub const b16_embed = "_ZN6tf_q279b16_embedEPKiPK13__nv_bfloat16PS2_i";
    pub const bf16_to_f16 = "_ZN6tf_q2711bf16_to_f16EPK13__nv_bfloat16P6__halfx";
    pub const gdn_pre = "_ZN6tf_q277gdn_preEPK13__nv_bfloat16S2_S2_PKiS2_S2_PKfS2_PS0_S7_S7_PfS8_iiiii";
    pub const attn_prep = "_ZN6tf_q279attn_prepEPK13__nv_bfloat16S2_S2_S2_PKfS4_PS0_S5_fiiiiiii";
    pub const attn_row = "_ZN6tf_q278attn_rowEPK13__nv_bfloat16S2_S2_PS0_iiixf";
    pub const paths = "_ZN6tf_q2712paths_kernelEPKiPiS2_i";
    pub const shared = "_ZN6tf_q2713shared_kernelEPK13__nv_bfloat16PKyPKiS6_PfS7_S7_iiiiif";
    pub const tail = "_ZN6tf_q2711tail_kernelEPK13__nv_bfloat16S2_S2_PKyPKiS6_S6_S6_PfS7_S7_iiiiif";
    pub const merge = "_ZN6tf_q2712merge_kernelEPKfS1_S1_P13__nv_bfloat16PKiS5_iiiii";
};
