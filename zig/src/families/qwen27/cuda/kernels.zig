//! The 27B's CUDA kernels: our .cu fatbins with the Python wrappers' launch logic, and the captured Triton set.

const std = @import("std");
const cuda = @import("cuda");
pub const torch_ops = @import("nemotron").torch_ops;
pub const Trellis = @import("exl3_scratch.zig").Trellis;

/// Mangled names of the instantiations the copies in zig/kernels/cuda export (cuobjdump -symbols of each fatbin).
const sym = struct {
    const group = "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
    const wide = [_][:0]const u8{
        "_ZN12tf_qmm_group12group_kernelILi64ELi32ELi64ELi1ELi4ELi4ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii",
        "_ZN12tf_qmm_group12group_kernelILi64ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii",
        "_ZN12tf_qmm_group12group_kernelILi64ELi64ELi128ELi2ELi4ELi3ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii",
    };
    const gemv = "_ZN12tf_lane_gemv11gemv_kernelILi64ELi64ELi8EEEvPK13__nv_bfloat16PKfNS_4PartEiii";
    const prefill_mm = "_ZN14tf_qmm_prefill14prefill_kernelILi64ELi128ELi128ELi2ELi2ELi2ELb0EEEvPK13__nv_bfloat16PKjS3_S3_Pviiiiii";
    const pattn = "_ZN20tf_prefill_attention12pattn_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiif";
    const chain128 = "_ZN14tf_gdn_prefill12chain_kernelI13__nv_bfloat16Li128ELi32EEEvPKT_S4_PKS1_PKfS8_S8_PfPS1_iii";
    const chain64 = "_ZN14tf_gdn_prefill12chain_kernelI13__nv_bfloat16Li64ELi32EEEvPKT_S4_PKS1_PKfS8_S8_PfPS1_iii";
    const argmax = "tf_argmax_rows_i32_kernel";
};

/// qmm_group.cu's Part and Parts, passed by value: four projections at most, one used here.
pub const Part = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, sk: c_int, tiles: c_int, first: c_int };
pub const Parts = extern struct { p: [4]Part, count: c_int };

/// lane_gemv.cu's Part: one projection whose column tiles the CTAs share out.
pub const GemvPart = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, sk: c_int, tiles: c_int };

/// gdn.cu's Pending (the tree kernel's rows still to fold): all null here, commits replay instead.
pub const Pending = extern struct { k: u64 = 0, v: u64 = 0, g: u64 = 0, beta: u64 = 0, rows: u64 = 0, row_stride: c_int = 0, counts: u64 = 0, count_stride: c_int = 0 };

comptime {
    std.debug.assert(@sizeOf(Part) == 56 and @sizeOf(Parts) == 232 and @sizeOf(GemvPart) == 48 and @sizeOf(Pending) == 64);
}

pub const group_smem: u32 = 35328; // LaneTile<64, 16, 64, 1, 4, 8>::SMEM
/// qmm_group's GB10 tiles 3-5 for wider rounds: rows a tile, columns a tile, threads, LaneTile::SMEM.
pub const Wide = struct { bm: usize, bn: usize, threads: u32, smem: u32 };
pub const wide = [_]Wide{ .{ .bm = 32, .bn = 64, .threads = 128, .smem = 26112 }, .{ .bm = 64, .bn = 64, .threads = 128, .smem = 43008 }, .{ .bm = 64, .bn = 128, .threads = 256, .smem = 39168 } };
pub const prefill_mm_smem: u32 = 41984; // Tile<64, 128, 128, 2, 2, 2>::SMEM: prompt_tile 9, two stages
pub const pattn_smem: u32 = 98304; // two 32-key slots of 256 dims beside eight warps' staged queries

/// One item of a batched copy: 16-byte aligned pointers, a multiple of 16 bytes.
pub const Copy = extern struct { dst: u64, src: u64, bytes: u64 };

/// One projection in whichever format holds it: `affine4` the MLX layout, `exl3` a trellis, `b16` bf16 rows.
pub const QLinear = union(enum) {
    affine4: Affine4,
    exl3: Exl3W,
    b16: B16W,

    pub fn n(q: QLinear) usize {
        return switch (q) { .affine4 => |a| a.n, .exl3 => |e| e.w.n, .b16 => |b| b.n };
    }

    pub fn k(q: QLinear) usize {
        return switch (q) { .affine4 => |a| a.k, .exl3 => |e| e.w.k, .b16 => |b| b.k };
    }
};

/// The MLX 4-bit tiled weight, byte-identical to the pre-union layout.
pub const Affine4 = struct { w: u64, s: u64, b: u64, n: usize, k: usize, npad: usize };

/// An EXL3 projection: the trellis weight plus the module that launches it.
pub const Exl3W = struct { w: cuda.exl3.Weight, lin: *const cuda.exl3.Linear };

/// An unquantized weight: (n, k) row-major, fp16 or bf16 as stored.
pub const B16W = struct { w: u64, n: usize, k: usize, fp16: bool = false };

pub const Kernels = struct {
    d: *const cuda.Driver,
    mods: [18]cuda.Module,
    mods_loaded: usize = 15, // the 15 base images; slots 15/16 are exl3 and b16 when they loaded
    exl3: cuda.exl3.Linear = undefined,
    b16_rows: [3]cuda.Function = undefined, // R = 1, 4, 16 (bf16)
    b16_prompt: [3]cuda.Function = undefined, // BM = 32, 64, 128 (bf16)
    f16_rows: [3]cuda.Function = undefined, // the fp16 instantiations (in_proj_a/b are fp16 in the packs)
    exl3_on: bool = false,
    b16_on: bool = false,
    b16_embed: cuda.Function = undefined, // q27_rows.cu's b16_embed: bf16 table rows gathered by id
    to_f16: cuda.Function = undefined, // q27_rows.cu's bf16_to_f16: an fp16 weight's inputs, as b16.matmul casts them
    q27_rows_mod: ?cuda.Module = null, // loaded best-effort beside b16
    triton: cuda.aot.Set,
    group: cuda.Function,
    wide: [wide.len]cuda.Function, // tiles 3-5: rows <= 32, <= 64, more
    gemv: cuda.Function,
    prefill_mm: cuda.Function,
    pattn: cuda.Function,
    trees: [cuda.kernels.gdn_variants.len]cuda.Function, // tree_kernel by gdn_variants: [0] the chain one
    chip: [3]c_int, // major, minor, SMs: what dispatch_tree reads
    replay: cuda.Function,
    prompt_chain: cuda.Function,
    prompt_rows: usize, // the prompt chain's value rows a block: 64 when heads < SMs
    pack_mod: cuda.Module, // affine4_pack: the MLX repack (pack_dense, transpose16)
    pack_dense: cuda.Function,
    transpose16: cuda.Function,
    argmax: cuda.Function,
    torch: torch_ops.Functions,
    copy_batch: cuda.Function,
    tap: cuda.Function,
    rms_norm: cuda.Function,
    inv_freq: cuda.Function,
    torch_rms: cuda.Function, // F.rms_norm's bits on bf16 rows of 5120 (torch_ops/norm_rotary.cu)
    rotary: cuda.Function,
    gemv_blocks: usize, // resident lane_gemv CTAs: per SM times SMs
    gemv_rows: usize, // windows of at most this many rows take lane_gemv (none; TF_QWEN27_GEMV_ROWS)
    gb10: bool,
    discrete: bool, // the card has its own memory: checkpoint bytes reach it through page-locked slots

    /// Loads every module; `triton_dir` holds the captured aot.json and cubins for this GPU.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, triton_dir: []const u8, heads: usize) !Kernels {
        const d = ctx.d;
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.d = d;
        const kk = cuda.kernels;
        const images = [_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.prefill_attention, kk.gdn, kk.gdn_prefill, kk.nemotron_ops, kk.torch_argmax, kk.torch_topk, kk.torch_pointwise, kk.torch_indexing, kk.torch_movement, kk.torch_nemotron_constants, kk.qwen_ops, kk.torch_norm_rotary };
        // the sm_86 additions load best-effort: an EXL3 pack needs them, an MLX pack never touches them.
        var loaded_extra: usize = 0;
        if (cuda.Module.load(d, kk.exl3_linear)) |m| {
            k.mods[15] = m;
            loaded_extra += 1;
            k.mods_loaded = 15 + loaded_extra;
            k.exl3 = try cuda.exl3.Linear.fromModule(d, &k.mods[15]);
            k.exl3_on = true;
            if (std.c.getenv("TF_SMOKE_EXL3") != null) {
                // isolation: rotIn on synthetic buffers through the exact serving path
                var s = try cuda.Stream.init(ctx.d, true);
                defer s.deinit();
                const rows2: usize = 4;
                const kk2: usize = 5120;
                var x = try cuda.DeviceBuffer.alloc(ctx.d, rows2 * kk2 * 2);
                defer x.free();
                var suh = try cuda.DeviceBuffer.alloc(ctx.d, kk2 * 2);
                defer suh.free();
                var xh = try cuda.DeviceBuffer.alloc(ctx.d, rows2 * kk2 * 2);
                defer xh.free();
                try k.exl3.rotInPtr(s, x.ptr, suh.ptr, xh.ptr, rows2, kk2);
                try s.synchronize();
                std.debug.print("SMOKE rotIn OK\n", .{});
            }
        } else |e| {
            std.log.err("exl3_linear module failed to load: {}", .{e});
        }
        if (cuda.Module.load(d, kk.b16)) |m| {
            k.mods[16] = m;
            loaded_extra += 1;
            k.mods_loaded = 15 + loaded_extra;
            k.b16_on = true;
            const bs = cuda.kernels.b16_symbols;
            k.b16_rows = .{ try k.mods[16].function(bs.bf16_rows_1), try k.mods[16].function(bs.bf16_rows_4), try k.mods[16].function(bs.bf16_rows_16) };
            const fs = cuda.kernels.b16_f16_symbols;
            k.f16_rows = .{ try k.mods[16].function(fs.rows_1), try k.mods[16].function(fs.rows_4), try k.mods[16].function(fs.rows_16) };
            k.b16_prompt = .{ try k.mods[16].function(bs.bf16_prompt_32), try k.mods[16].function(bs.bf16_prompt_64), try k.mods[16].function(bs.bf16_prompt_128) };
            // prompt_launch's one-time opt-in: BM 64 and 128 stage more than 48 KiB of dynamic shared memory
            for (k.b16_prompt, prompt_bms) |f, bm| try f.allowDynamicShared(promptSmem(bm));
        } else |e| {
            std.log.err("b16 module failed to load: {}", .{e});
        }
        if (cuda.Module.load(d, kk.q27_rows)) |m| {
            k.mods[17] = m;
            loaded_extra += 1;
            k.mods_loaded = 15 + loaded_extra;
            k.b16_embed = try k.mods[17].function(cuda.kernels.q27_symbols.b16_embed);
            k.to_f16 = try k.mods[17].function(cuda.kernels.q27_symbols.bf16_to_f16);
        } else |e| {
            std.log.err("q27_rows module failed to load: {} — b16_embed stays unset", .{e});
        }
        var loaded: usize = 0;
        errdefer for (k.mods[0..loaded]) |*m| m.unload();
        for (images, 0..) |img, i| {
            k.mods[i] = try cuda.Module.load(d, img);
            loaded += 1;
        }
        _ = &loaded;
        k.group = try k.mods[0].function(sym.group);
        for (&k.wide, sym.wide) |*f, name| f.* = try k.mods[0].function(name);
        k.prefill_mm = try k.mods[1].function(sym.prefill_mm);
        k.gemv = try k.mods[2].function(sym.gemv);
        k.pattn = try k.mods[3].function(sym.pattn);
        for (&k.trees, cuda.kernels.gdn_variants) |*f, v| f.* = try k.mods[4].function(v.symbol);
        k.replay = try k.mods[4].function(cuda.kernels.gdn_symbols.replay_bf16);
        // #602 moved the affine-4 repack out of nemotron_ops into its own image, as Nemotron's loader takes it
        k.pack_mod = try cuda.Module.load(d, kk.affine4_pack);
        errdefer k.pack_mod.unload();
        k.pack_dense = try k.pack_mod.function("tf_pack_dense");
        k.transpose16 = try k.pack_mod.function("tf_transpose_pad16");
        k.argmax = try k.mods[7].function(sym.argmax);
        k.torch = try torch_ops.Functions.resolve(k.mods[7..13]);
        k.copy_batch = try k.mods[13].function("tf_copy_batch");
        k.tap = try k.mods[13].function("tf_tap");
        k.rms_norm = try k.mods[13].function("tf_rms_norm");
        k.inv_freq = try k.mods[13].function("tf_inv_freq");
        k.rotary = try k.mods[13].function("tf_rotary");
        k.torch_rms = try k.mods[14].function("tf_dflash_rms5120_bf16_fma_kernel");
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        // gdn_prefill_cuda: a block takes 128 value rows, or 64 when there are fewer heads than SMs
        k.prompt_rows = if (heads >= sms) 128 else 64;
        k.prompt_chain = try k.mods[5].function(if (k.prompt_rows == 128) sym.chain128 else sym.chain64);
        k.triton = try cuda.aot.Set.load(gpa, io, d, ctx.device, triton_dir);
        errdefer k.triton.deinit();
        try k.group.allowDynamicShared(group_smem);
        for (k.wide, wide) |f, t| try f.allowDynamicShared(t.smem);
        try k.gemv.allowDynamicShared(group_smem);
        try k.prefill_mm.allowDynamicShared(prefill_mm_smem);
        try k.pattn.allowDynamicShared(pattn_smem);
        k.gemv_blocks = @max(1, try k.gemv.occupancy(128, group_smem)) * sms;
        const major = try ctx.attribute(.compute_capability_major);
        const minor = try ctx.attribute(.compute_capability_minor);
        k.gb10 = major == 12 and minor == 1;
        k.discrete = try ctx.attribute(.integrated) == 0;
        k.gemv_rows = 0; // a GB10 runs the 27B's projections faster in Python's clusters, 1 to 16 rows alike
        if (std.c.getenv("TF_QWEN27_GEMV_ROWS")) |v| k.gemv_rows = std.fmt.parseInt(usize, std.mem.span(v), 10) catch k.gemv_rows;
        k.chip = .{ major, minor, @intCast(sms) };
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.triton.deinit();
        for (k.mods[0..k.mods_loaded]) |*m| m.unload();
        k.pack_mod.unload();
    }
};

/// qmm.split_k: K slices fixed by the weight's shape, never by the row count.
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + 63) / 64;
    const groups = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 192 and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return sk;
}

fn int(x: usize) c_int {
    return @intCast(x);
}

fn u(x: usize) u32 {
    return @intCast(x);
}

/// b16.cu's prompt tiles: PBN columns a block, PST stages of (BM + PBN) rows of PROW bytes in dynamic shared memory.
const prompt_bn = 64;
const prompt_bms = [3]usize{ 32, 64, 128 };

fn promptSmem(bm: usize) u32 {
    return @intCast(3 * (bm + prompt_bn) * 128);
}

/// prompt_bm: the tallest tile that still gives two blocks an SM (narrow weights take short tiles); never a row's bits.
fn promptBm(rows: usize, n: usize, sms: usize) usize {
    const cols = (n + prompt_bn - 1) / prompt_bn;
    var b: usize = 128;
    while (b > 32 and cols * ((rows + b - 1) / b) < 2 * sms) b /= 2;
    return b;
}

/// The GDN shapes every launch takes: key heads, value heads, value rows a head.
pub const Heads = struct { hk: usize, hv: usize, dv: usize };

/// Launch helpers on one stream; each mirrors the Python wrapper it replaces.
pub const Ops = struct {
    k: *const Kernels,
    s: cuda.Stream,
    trellis: ?*Trellis = null, // the EXL3 decode scratch (xh, z), owned by the engine's scratch

    fn go(o: Ops, f: cuda.Function, grid: [3]usize, block: u32, shared: u32, args: *cuda.Args) !void {
        if (std.c.getenv("TF_TRACE_LAUNCH") != null) {
            std.debug.print("[launch] grid=({d},{d},{d}) block={d} shared={d}\n", .{ grid[0], grid[1], grid[2], block, shared });
        }
        try cuda.launch.launch(f, .{ .grid = .{ .x = u(grid[0]), .y = u(grid[1]), .z = u(grid[2]) }, .block = .{ .x = block }, .shared = shared }, o.s, args);
    }

    /// qmm.matmul by format: affine4 the lane path (sm_12x), exl3 the trellis GEMV, b16 the unquantized rows.
    pub fn dense(o: Ops, x: u64, xs: u64, q: QLinear, out: u64, rows: usize) !void {
        switch (q) {
            .affine4 => |a| {
                const sk = splitK(a.n, a.k);
                if (rows > 16) return o.wideTile(x, xs, a, out, rows, sk);
                // Python's clusters, a CTA a K slice; lane_gemv (byte-equal) when TF_QWEN27_GEMV_ROWS asks for it
                return if (sk > 1 and rows <= o.k.gemv_rows) o.gemv(x, xs, a, out, rows, sk) else o.cluster(x, xs, a, out, rows, sk);
            },
            .exl3 => |e| return o.exl3Dense(e, x, out, rows),
            .b16 => |b| return o.b16Rows(x, b, out, rows),
        }
    }

    /// The trellis GEMV: rot_in (bf16 x -> fp16 xh) then linear, the row-invariant decode path.
    fn exl3Dense(o: Ops, e: Exl3W, x: u64, out: u64, rows: usize) !void {
        if (std.c.getenv("TF_SKIP_EXL3") != null) return; // bisect: skip the trellis launches
        const t = o.trellis orelse return error.TrellisScratchMissing;
        const xh = try t.xhAt(rows, e.w.k);
        try e.lin.rotInPtr(o.s, x, e.w.suh.ptr, xh, rows, e.w.k);
        const sp = e.w.split();
        const z = if (sp.sk > 1) try t.zAt(sp.sk, rows, e.w.n) else 0;
        try e.lin.matmulPtr(o.s, e.w, xh, out, z, rows);
    }

    /// b16 rows: a warp a column over R rows (R by row count: 1, 4, 16), fp16 or bf16 weights.
    fn b16Rows(o: Ops, x: u64, b: B16W, out: u64, rows: usize) !void {
        const i: usize = if (rows <= 1) 0 else if (rows <= 4) 1 else 2;
        const r: usize = switch (i) { 0 => 1, 1 => 4, else => 16 };
        const warps: usize = switch (i) { 0 => 4, 1 => 8, else => 16 };
        const xin = if (b.fp16) try o.toF16(x, rows * b.k) else x;
        var a: cuda.Args = .{};
        a.add(xin);
        a.add(b.w);
        a.add(@as(u64, 0)); // bias: none
        a.add(out);
        for ([_]usize{ rows, b.k, b.n }) |v| a.add(int(v));
        a.add(@as(u64, 0)); // second weight
        a.add(@as(u64, 0)); // second out
        a.add(@as(c_int, 0)); // second n
        const f = if (b.fp16) o.k.f16_rows[i] else o.k.b16_rows[i];
        try o.go(f, .{ (b.n + warps - 1) / warps, (rows + r - 1) / r, 1 }, @intCast(warps * 32), 0, &a);
    }

    /// qmm.matmul at a K split the caller names (matmul_rows takes the stacked weight's for each part).
    pub fn denseSplit(o: Ops, x: u64, xs: u64, q: QLinear, out: u64, rows: usize, sk: usize) !void {
        switch (q) {
            .affine4 => |a| {
                if (rows > 16) return o.wideTile(x, xs, a, out, rows, sk);
                return if (sk > 1 and rows <= o.k.gemv_rows) o.gemv(x, xs, a, out, rows, sk) else o.cluster(x, xs, a, out, rows, sk);
            },
            else => return o.dense(x, xs, q, out, rows), // exl3 and b16 pick their own split by shape
        }
    }

    /// tree_forward's taps: bf16(x + pending) of `rows` rows into columns [col, col + d) of rows `stride` wide.
    pub fn tap(o: Ops, x: u64, pending: u64, out: u64, rows: usize, d: usize, stride: usize, col: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ x, pending, out }) |v| a.add(v);
        for ([_]usize{ d, stride, col }) |v| a.add(int(v));
        try o.go(o.k.tap, .{ rows, 4, 1 }, 256, 0, &a);
    }

    /// lane_gemv: every K slice of a column tile in one CTA, summed in slice order, CTAs looping over the tiles.
    fn gemv(o: Ops, x: u64, xs: u64, q: Affine4, out: u64, rows: usize, sk: usize) !void {
        const tiles = (q.n + 63) / 64;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(GemvPart{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .tiles = int(tiles) });
        for ([_]usize{ rows, q.k, q.k }) |v| a.add(int(v));
        const cfg: cuda.Config = .{ .grid = .{ .x = u(@min(tiles, o.k.gemv_blocks)) }, .block = .{ .x = 128 }, .shared = group_smem, .pdl = o.k.gb10 };
        try cuda.launch.launch(o.k.gemv, cfg, o.s, &a);
    }

    /// qmm_group's tile 2 with one K slice: a CTA a column tile.
    fn cluster(o: Ops, x: u64, xs: u64, q: Affine4, out: u64, rows: usize, sk: usize) !void {
        const tiles = (q.n + 63) / 64;
        var parts: Parts = std.mem.zeroes(Parts);
        parts.count = 1;
        parts.p[0] = .{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .tiles = int(tiles), .first = 0 };
        const rows_t = (rows + 15) / 16;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(parts);
        for ([_]usize{ rows, q.k, q.k, rows_t, sk }) |v| a.add(int(v));
        const cfg: cuda.Config = .{
            .grid = .{ .x = u(rows_t * tiles * sk) },
            .block = .{ .x = 128 },
            .shared = group_smem,
            .cluster = if (sk > 1) .{ .x = u(sk) } else null,
            .pdl = o.k.gb10,
        };
        try cuda.launch.launch(o.k.group, cfg, o.s, &a);
    }

    /// qmm_group's dispatch on a GB10 past 16 rows: tile 3, 4 or 5, a cluster of `sk` CTAs a column tile.
    fn wideTile(o: Ops, x: u64, xs: u64, q: Affine4, out: u64, rows: usize, sk: usize) !void {
        const i: usize = if (rows <= 32) 0 else if (rows <= 64) 1 else 2;
        const t = wide[i];
        const tiles = (q.n + t.bn - 1) / t.bn;
        const rows_t = (rows + t.bm - 1) / t.bm;
        var parts: Parts = std.mem.zeroes(Parts);
        parts.count = 1;
        parts.p[0] = .{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = int(q.n), .npad = int(q.npad), .sk = int(sk), .tiles = int(tiles), .first = 0 };
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(parts);
        for ([_]usize{ rows, q.k, q.k, rows_t, sk }) |v| a.add(int(v));
        const cfg: cuda.Config = .{
            .grid = .{ .x = u(rows_t * tiles * sk) },
            .block = .{ .x = t.threads },
            .shared = t.smem,
            .cluster = if (sk > 1) .{ .x = u(sk) } else null,
            .pdl = o.k.gb10,
        };
        try cuda.launch.launch(o.k.wide[i], cfg, o.s, &a);
    }

    /// qmm.prefill_matmul at prompt_tile 9: weights rounded once to bf16, one fp32 chain over K; bf16 out.
    pub fn prefillDense(o: Ops, x: u64, q: QLinear, out: u64, rows: usize) !void {
        switch (q) {
            .affine4 => |w4| {
                const rows_t = (rows + 127) / 128;
                const band = (12 << 20) / (128 * w4.k * 2);
                const group = @max(1, @min(rows_t, band));
                var a: cuda.Args = .{};
                for ([_]u64{ x, w4.w, w4.s, w4.b, out }) |v| a.add(v);
                for ([_]usize{ rows, w4.n, w4.k, w4.npad, w4.k, group }) |v| a.add(int(v));
                try o.go(o.k.prefill_mm, .{ rows_t * ((w4.n + 127) / 128), 1, 1 }, 128, prefill_mm_smem, &a);
            },
            .exl3 => |e| return o.exl3Prefill(e, x, out, rows),
            .b16 => |b| return o.b16Prompt(x, b, out, rows),
        }
    }

    /// The EXL3 prompt path: rot_in, W_q unpacked once, the captured _gemm with the Hadamard epilogue.
    fn exl3Prefill(o: Ops, e: Exl3W, x: u64, out: u64, rows: usize) !void {
        if (std.c.getenv("TF_SKIP_EXL3") != null) return; // bisect
        const t = o.trellis orelse return error.TrellisScratchMissing;
        const xh = try t.xhAt(rows, e.w.k);
        try e.lin.rotInPtr(o.s, x, e.w.suh.ptr, xh, rows, e.w.k);
        const wq = try t.wqAt(e.w.k, e.w.n);
        try e.lin.unpackPtr(o.s, e.w, wq);
        const h = try t.hadamardAt();
        const bias = if (e.w.bias.ptr != 0) e.w.bias.ptr else e.w.svh.ptr;
        try o.k.triton.run(o.s, "_gemm", .{ @intCast(((rows + 127) / 128) * (e.w.n / 128)), 1, 1 }, &.{
            cuda.aot.ptr("X", "*fp16", xh),   cuda.aot.ptr("W", "*fp16", wq),   cuda.aot.ptr("H", "*bf16", h),
            cuda.aot.ptr("SVH", "*fp16", e.w.svh.ptr), cuda.aot.ptr("BIAS", "*fp16", bias), cuda.aot.ptr("OUT", "*bf16", out),
            cuda.aot.int("M", @intCast(rows)), cuda.aot.int("o_stride", @intCast(e.w.n)),
        }, &.{
            cuda.aot.ci("K", @intCast(e.w.k)), cuda.aot.ci("N", @intCast(e.w.n)), cuda.aot.ci("BM", 128), cuda.aot.ci("BK", 32),
            cuda.aot.ci("GROUP", 8), cuda.aot.ci("HAS_BIAS", @intFromBool(e.w.bias.ptr != 0)),
            cuda.aot.cf("SCALE", @bitCast(had_scale_bits)),
        });
    }

    /// b16.prompt: bf16 weights on the BM-tile mma kernel; an fp16 weight (in_proj_a/b) or K off 64 takes b16.matmul.
    fn b16Prompt(o: Ops, x: u64, b: B16W, out: u64, rows: usize) !void {
        if (b.fp16 or b.k % 64 != 0) return o.b16Rows(x, b, out, rows);
        const bm = promptBm(rows, b.n, @intCast(o.k.chip[2]));
        const i: usize = switch (bm) {
            32 => 0,
            64 => 1,
            else => 2,
        };
        var a: cuda.Args = .{};
        a.add(x);
        a.add(b.w);
        a.add(out);
        for ([_]usize{ rows, b.k, b.n }) |v| a.add(int(v));
        a.add(@as(u64, 0));
        a.add(@as(u64, 0));
        a.add(@as(c_int, 0));
        try o.go(o.k.b16_prompt[i], .{ (b.n + prompt_bn - 1) / prompt_bn, (rows + bm - 1) / bm, 1 }, 128, promptSmem(bm), &a);
    }

    /// bf16 rows to fp16 in the trellis scratch's xh (free between EXL3 calls, which rewrite it before reading).
    fn toF16(o: Ops, x: u64, n: usize) !u64 {
        const t = o.trellis orelse return error.TrellisScratchMissing;
        const y = try t.xhAt(1, n);
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(@as(i64, @intCast(n)));
        try o.go(o.k.to_f16, .{ @min((n + 255) / 256, 4096), 1, 1 }, 256, 0, &a);
        return y;
    }

    /// 1/sqrt(128) as fp32 bits, the _gemm epilogue's SCALE (prefill.py's HAD_SCALE).
    pub const had_scale_bits: u32 = 0x3db504f3;

    /// gdn.tree (dispatch_tree): y from one `state`, or each stream's own by `table` and `starts`; `slots` 0: chains.
    pub fn gdnTree(o: Ops, q: u64, k: u64, v: u64, g: u64, beta: u64, state: u64, table: u64, starts: u64, plan: u64, y: u64, rows: usize, streams: usize, slots: usize, most: usize, h: Heads) !void {
        const variant = cuda.kernels.treeVariant(@intCast(slots), @intCast(streams), o.k.chip[0], o.k.chip[1], o.k.chip[2]);
        const index = for (cuda.kernels.gdn_variants, 0..) |x, n| {
            if (std.mem.eql(u8, x.symbol, variant.symbol)) break n;
        } else unreachable;
        var a: cuda.Args = .{};
        for ([_]u64{ q, k, v, g, beta, state, table, starts, plan }) |x| a.add(x);
        a.add(int(rows));
        a.add(y);
        for ([_]usize{ h.hk, h.hv, h.dv }) |x| a.add(int(x));
        a.add(Pending{});
        a.add(@as(u64, 0));
        a.add(@as(u64, 0));
        a.add(h.dv % variant.r == 0 and v % (2 * @as(u64, @min(variant.r, 8))) == 0);
        const shared = 16 * @as(usize, variant.warps) * variant.slots * variant.r * 32 + (if (variant.chain) 0 else 4 * 3 * most);
        try o.go(o.k.trees[index], .{ (h.dv + variant.r * variant.warps - 1) / (variant.r * variant.warps), h.hv, streams }, 32 * variant.warps, @intCast(shared), &a);
    }

    /// gdn.replay in place: each stream's accepted rows stepped into the states the table names.
    pub fn gdnReplay(o: Ops, table: u64, layers: usize, streams: usize, rows: u64, row_stride: usize, counts: u64, count_stride: usize, h: Heads) !void {
        var a: cuda.Args = .{};
        a.add(table);
        a.add(int(layers));
        a.add(rows);
        a.add(int(row_stride));
        a.add(counts);
        a.add(int(count_stride));
        a.add(@as(u64, 0));
        for ([_]usize{ h.hk, h.hv, h.dv }) |x| a.add(int(x));
        try o.go(o.k.replay, .{ (h.dv + 31) / 32, h.hv, layers * streams }, 128, 0, &a);
    }

    /// gdn.chain: a prompt chunk's rows from `state` (read only) to `last`, chunk-invariant bits.
    pub fn gdnPrompt(o: Ops, q: u64, k: u64, v: u64, g: u64, beta: u64, state: u64, last: u64, y: u64, rows: usize, h: Heads) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ q, k, v, g, beta, state, last, y }) |x| a.add(x);
        for ([_]usize{ rows, h.hk, h.hv }) |x| a.add(int(x));
        const r = o.k.prompt_rows;
        try o.go(o.k.prompt_chain, .{ h.hv, 128 / r, 1 }, u(2 * r), 0, &a);
    }

    /// prefill_attention (head dim 256, two query heads a block): q (rows, heads, 256) at positions p0..
    pub fn prefillAttention(o: Ops, q: u64, kc: u64, vc: u64, out: u64, p0: usize, rows: usize, heads: usize, kv_heads: usize, scale: f32) !void {
        const g = heads / kv_heads;
        const hpc = 2;
        var a: cuda.Args = .{};
        for ([_]u64{ q, kc, vc, out }) |v| a.add(v);
        for ([_]usize{ p0, rows, heads, kv_heads, g }) |v| a.add(int(v));
        a.add(scale);
        const block_rows = 16 * (8 / hpc);
        try o.go(o.k.pattn, .{ (rows + block_rows - 1) / block_rows, kv_heads * (g / hpc), 1 }, 256, pattn_smem, &a);
    }

    /// torch.argmax(rows of bf16 logits) as int32: the first maximum, a NaN first.
    pub fn argmax(o: Ops, logits: u64, vocab: usize, ld: usize, out: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(out);
        for ([_]usize{ rows, vocab, ld }) |v| a.add(@as(i64, @intCast(v)));
        a.add(@as(i32, 0)); // dtype 0: bf16
        try o.go(o.k.argmax, .{ rows, 1, 1 }, 1024, 0, &a);
    }

    /// The torch-op replacements on this stream (topk and casts for sampled draws).
    pub fn torch(o: Ops) torch_ops.Torch {
        return .{ .f = &o.k.torch, .s = o.s };
    }

    /// qwen_ops.cu's tf_copy_batch: `count` (dst, src, bytes) items from the device table `items`, one launch.
    pub fn copies(o: Ops, items: u64, count: usize) !void {
        if (count == 0) return;
        var a: cuda.Args = .{};
        a.add(items);
        try o.go(o.k.copy_batch, .{ count, 4, 1 }, 256, 0, &a);
    }

    pub fn copy(o: Ops, dst: u64, src: u64, bytes: usize) !void {
        if (bytes == 0) return;
        try o.k.d.check(o.k.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, o.s.handle), "cuMemcpyDtoDAsync");
    }

    pub fn fill32(o: Ops, dst: u64, value: u32, words: usize) !void {
        try o.k.d.check(o.k.d.api.cuMemsetD32Async(dst, value, words, o.s.handle), "cuMemsetD32Async");
    }

    pub fn upload(o: Ops, dst: u64, bytes: []const u8) !void {
        if (bytes.len == 0) return;
        try o.k.d.check(o.k.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, o.s.handle), "cuMemcpyHtoDAsync");
    }

    pub fn download(o: Ops, dst: []u8, src: u64) !void {
        if (dst.len == 0) return;
        try o.k.d.check(o.k.d.api.cuMemcpyDtoHAsync_v2(dst.ptr, src, dst.len, o.s.handle), "cuMemcpyDtoHAsync");
    }
};

test "split_k follows the 27B's shapes" {
    try std.testing.expectEqual(@as(usize, 2), splitK(10240, 5120)); // qkv
    try std.testing.expectEqual(@as(usize, 2), splitK(6144, 5120)); // z
    try std.testing.expectEqual(@as(usize, 8), splitK(48, 5120)); // b, a
    try std.testing.expectEqual(@as(usize, 1), splitK(17408, 5120)); // gate, up
    try std.testing.expectEqual(@as(usize, 4), splitK(5120, 17408)); // down
    try std.testing.expectEqual(@as(usize, 4), splitK(5120, 6144)); // out_proj, o_proj
    try std.testing.expectEqual(@as(usize, 1), splitK(12288, 5120)); // q with its gate
    try std.testing.expectEqual(@as(usize, 8), splitK(1024, 5120)); // k, v
    try std.testing.expectEqual(@as(usize, 1), splitK(248320, 5120)); // head
}
