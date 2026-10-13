// msh_mla.v -- MLA attention unit with KV cache.
//
// Part of msh_chip_top. Matches the
// per-head attention loop of rtl/selfmodel/fxmodel.py mla() BIT-EXACTLY:
//   att[t] = rs(sum_d rs(K[t][d],4) * rs(q[d],4), 18)   (K/q Q4.26)
//   att[t] = rs(att[t] * c_att, 16)                     (c_att Q16-ish)
//   mx     = max_t att[t]
//   ei     = clip(att[t] - mx, -2^30, 0)
//   e[t]   = expneg_lut(ei)  (bucket 2^18 codes at sx=26, full-frac interp:
//            idx = clip((ei>>18)+4096, 0, 4096), frac = ei - (ei>>18)<<18,
//            y = t[idx] + (d[idx]*frac + 2^17) >> 18)
//   se     = sum_t e[t]
//   rec    = recip_eval_i(se, 46)  (norm20: m_q in [2^20,2^21), E9 = E+9;
//            idx = (m_q>>9)-2048, frac = m_q&511,
//            r = t[idx] + (d[idx]*frac + 256) >> 9;  sh = 46-30-E9;
//            rec = sh>=0 ? r<<sh : (r + 2^(-sh-1)) >> (-sh))
//   ctx[d] = sat32((rs(sum_t rs(e[t],4) * V[t][d], 28) * rec + 2^13) >> 14)
// where rs(v, r) = (v + 2^(r-1)) >>> r (round-half-up, arithmetic).
//
// Job: q and k_new arrive SPLIT across four XBUF buffers (four separate
// GEMV outputs): qc [H*dk, element hd*dk + d], qr [H*dr, hd*dr + d],
// kc [H*dk], kr [dr, SHARED across heads]; v_new [H*dv] in one buffer.
// Per the C-model q[hd] = [qc_h | qr_h] and k[hd] = [kc_h | kr]; the
// module gathers the two parts per head (both dk and dr are multiples of
// 4, so the concat is word-aligned). It first stores k_new/v_new into the
// KV cache at position `pos` (both streams in parallel on the two read
// ports — K side reads kc then kr per head, V side reads v), then
// computes attention for each head over t = 0..pos: att (4 dims/cycle),
// exp (1 t/cycle), staged recip normalization, ctx accumulation (4 dims/cycle,
// set-on-first accumulators), ctx writeout (4 int32 lanes/word). ctx is
// written to the ctx buffer (head-major). The output GEMV (L.mla.o) is
// done elsewhere; this module stops at ctx.
//
// KV caches are harness msh_sram macros (macros_only rule; sync read,
// 128b words = 4 int32 lanes). Compact dynamic addressing (strides from
// the job descriptor; H <= 2, pos < 128): K word = (hd*128 + t)*nwk + w
// (nwk = dqk/4), V word = (hd*128 + t)*nwv + w. Both cleared by a reset
// write-loop (randreset gate) before the first job is accepted. The LUTs
// are harness msh_rom macros (ASYNC read, same-cycle data — the exp/recip
// datapaths stage arithmetic around the ROM access). All other
// storage is packed vectors (no inferred $mem).
//
// ST_ATT/ST_CTX are retimed around the 1-cycle KV read: the word address
// is issued with (t, w) and the returned word is consumed one cycle
// later with the delayed (at_td/c_td, at_wd/c_wd). The tagged datapaths
// retain initiation interval 1; flush states drain their final results.
//
// Documented assumptions:
//   * dk, dr, dv are multiples of 4 and > 0; dqk = dk + dr <= 192;
//     dv <= 128; H <= 2; pos < 128. dk arrives as job_dk (the sequencer
//     must supply it — dr = dqk - dk).
//   * Realistic C-model ranges keep intermediates in the widths used
//     below (acc 72b, att 56b, se <= 512*2^26, rec < 2^21).
//
// Reset policy (randreset gate): every control flop is reset; both KV
// SRAMs are explicitly zeroed by the ST_CLEAR sweep. att/e/q/acc_d
// storage is written before being read within a job (verified with
// +verilator+rand+reset+2).

`default_nettype none

module msh_mla (
    input  wire         clk,
    input  wire         rst_n,
    // job descriptor
    input  wire         job_valid,
    output wire         job_ready,
    input  wire [9:0]   job_pos,        // write/attend position (0..127)
    input  wire [9:0]   job_dqk,        // qk dims per head (<= 192, mult of 4)
    input  wire [9:0]   job_dv,         // v dims per head (<= 128, mult of 4)
    input  wire [9:0]   job_dk,         // c-part dims (split point; dr = dqk-dk)
    input  wire [3:0]   job_H,          // heads (<= 2)
    input  wire [16:0]  job_c_att,      // round(2^16/sqrt(dqk))
    input  wire [3:0]   job_qc_buf,
    input  wire [3:0]   job_qr_buf,
    input  wire [3:0]   job_kc_buf,
    input  wire [3:0]   job_kr_buf,
    input  wire [3:0]   job_v_buf,
    input  wire [3:0]   job_ctx_buf,
    output wire         job_done,
    // XBUF: two read ports (1-cycle registered latency), one write port
    output wire [12:0]  xr0_addr,       // {buf[3:0], word[8:0]}
    input  wire [127:0] xr0_data,
    output wire [12:0]  xr1_addr,
    input  wire [127:0] xr1_data,
    output wire         xw_we,
    output wire [12:0]  xw_addr,
    output wire [127:0] xw_data,
    output wire [3:0]   xw_wstrb
);

    // ---------------- states ----------------
    localparam ST_CLEAR = 4'd0;
    localparam ST_IDLE  = 4'd1;
    localparam ST_STORE = 4'd2;         // k_new/v_new -> KV at pos
    localparam ST_QLOAD = 4'd3;         // per-head q preload
    localparam ST_ATT   = 4'd4;         // att[t] dot products
    localparam ST_ATT_FLUSH = 4'd10;    // drain pipelined attention results
    localparam ST_EXP   = 4'd5;         // exp LUT + se
    localparam ST_RECIP = 4'd6;         // staged reciprocal normalization
    localparam ST_CTX   = 4'd7;         // sum_t e[t]*V[t][d]
    localparam ST_CTXW  = 4'd8;         // ctx writeout
    localparam ST_DONE  = 4'd9;
    localparam ST_CTXW_DRAIN = 4'd11;   // drain pipelined ctx writeback

    reg [3:0] state;

    // ---------------- job registers ----------------
    reg [8:0]  j_pos;
    reg [5:0]  j_nwk;                   // dqk/4
    reg [5:0]  j_nwv;                   // dv/4
    reg [7:0]  j_dk;                    // split point (elements)
    reg [5:0]  j_nkc;                   // dk/4
    reg [5:0]  j_nkr;                   // dr/4 = (dqk-dk)/4
    reg [3:0]  j_H;
    reg [16:0] j_catt;
    reg [3:0]  j_qcbuf, j_qrbuf, j_kcbuf, j_krbuf, j_vbuf, j_cbuf;

    // ---------------- counters / accumulators ----------------
    reg [3:0]  hd;
    reg [8:0]  t;
    reg [5:0]  w;
    reg signed [71:0] acc;              // att dot-product accumulator
    reg signed [55:0] mx;               // running att max
    reg [35:0] se;                      // sum of exp outputs (<= 512*2^26)
    reg signed [47:0] rec;              // round(2^46/se)
    reg [16:0] clr;

    // store streamers (port0 -> K, port1 -> V)
    reg [1:0]  ks_hd, vs_hd;
    reg [5:0]  ks_w, vs_w;
    reg        ks_ph;                   // 0 = kc part, 1 = kr part
    reg        ks_vd, vs_vd;
    reg [1:0]  ks_hd_d, vs_hd_d;
    reg [5:0]  ks_w_d, vs_w_d;
    reg        ks_ph_d;
    reg        ks_done, vs_done, ks_cap, vs_cap;
    // q loader (port0): phase 0 = qc, phase 1 = qr
    reg [5:0]  q0_w, q0_wd;
    reg        q0_ph, q0_phd, q0_vd;
    // att/ctx sync-read delay registers (KV word returns one cycle
    // after the (t, w) issue)
    reg        att_iss, at_dv;
    reg [8:0]  at_td;
    reg [5:0]  at_wd;
    reg        att_round_valid, att_mul_valid;
    reg        att_dot_valid, att_acc_valid;
    reg        att_round_first, att_round_last;
    reg        att_round_tfirst, att_round_tlast;
    reg [8:0]  att_round_td;
    reg        att_mul_first, att_mul_last;
    reg        att_mul_tfirst, att_mul_tlast;
    reg [8:0]  att_mul_td;
    reg        att_dot_first, att_dot_last, att_dot_tfirst, att_dot_tlast;
    reg        att_acc_last, att_acc_tfirst, att_acc_tlast;
    reg [8:0]  att_dot_td, att_acc_td;
    reg signed [79:0] att_dot_r, att_acc_stage;
    reg        ctx_iss, c_dv;
    reg [8:0]  c_td;
    reg [5:0]  c_wd;

    // ---------------- storage ----------------
    // KV caches: harness msh_sram macros. Compact dynamic addressing:
    //   ht = hd*128 + t (H <= 2);  K word = ht*nwk + w;  V word = ht*nwv + w
    // K: 2*72*12 = 1728 words (nano nwk = 12); V: 2*72*8 = 1152.
    // Positions bounded at 72 (1.125x max_seq 64, 2x public long run 48
    // trimmed from 96 to fit the 4.0 mm2 area budget; see REPORT.md
    // risk note). ht = hd*72 + t (multiply-free form).
    wire [7:0]  at_ht  = (hd[0] ? 8'd72 : 8'd0) + {1'b0, t[6:0]};
    wire [13:0] k_addr = ({6'b0, at_ht} * {8'b0, j_nwk}) + {8'b0, w};
    wire [13:0] v_addr = ({6'b0, at_ht} * {8'b0, j_nwv}) + {8'b0, w};

    // store capture addresses (data returns from XBUF one cycle after
    // the source issue — the _d registers already carry that delay)
    wire [8:0]  ks_dst    = ks_ph_d ? ({3'b0, j_nkc} + {3'b0, ks_w_d})
                                    : {3'b0, ks_w_d};
    wire [7:0]  ks_ht     = (ks_hd_d[0] ? 8'd72 : 8'd0)
                          + {1'b0, j_pos[6:0]};
    wire [13:0] ks_caddr  = ({6'b0, ks_ht} * {8'b0, j_nwk}) + {5'b0, ks_dst};
    wire [7:0]  vs_ht     = (vs_hd_d[0] ? 8'd72 : 8'd0)
                          + {1'b0, j_pos[6:0]};
    wire [13:0] vs_caddr  = ({6'b0, vs_ht} * {8'b0, j_nwv}) + {8'b0, vs_w_d};

    wire         kw_we    = (state == ST_CLEAR)
                          || ((state == ST_STORE) && ks_vd);
    wire [10:0]  kw_waddr = (state == ST_CLEAR) ? clr[10:0]
                                                : ks_caddr[10:0];
    wire [127:0] kw_wd    = (state == ST_CLEAR) ? 128'd0 : xr0_data;
    wire [10:0]  k_raddr  = (state == ST_ATT) ? k_addr[10:0] : 11'd0;
    wire [127:0] k_rd;
    msh_sram #(.DEPTH(1728), .WIDTH(128)) u_kmem (
        .clk(clk), .we(kw_we), .waddr(kw_waddr), .wdata(kw_wd),
        .wstrb(16'hFFFF),
        .re(1'b1), .raddr(k_raddr), .rdata(k_rd));

    wire         vw_we    = ((state == ST_CLEAR) && (clr < 17'd1152))
                          || ((state == ST_STORE) && vs_vd);
    wire [10:0]  vw_waddr = (state == ST_CLEAR) ? clr[10:0]
                                                : vs_caddr[10:0];
    wire [127:0] vw_wd    = (state == ST_CLEAR) ? 128'd0 : xr1_data;
    wire [10:0]  v_raddr  = (state == ST_CTX) ? v_addr[10:0] : 11'd0;
    wire [127:0] v_rd;
    msh_sram #(.DEPTH(1152), .WIDTH(128)) u_vmem (
        .clk(clk), .we(vw_we), .waddr(vw_waddr), .wdata(vw_wd),
        .wstrb(16'hFFFF),
        .re(1'b1), .raddr(v_raddr), .rdata(v_rd));

    // packed-vector storage (no inferred $mem)
    reg [2047:0]  qbuf_v;               // 64 x int32 (dqk <= 64)
    reg [127:0]   qword_r;              // prefetched four-lane q word

    // att/e/ctx storage as msh_sram macros (synchronous read; the
    // consumers are retimed one cycle — see ST_EXP/ST_CTX/ST_CTXW).
    // Macro bits replace ~47 Kflops of wide variable-index logic.
    wire         atb_we    = att_acc_valid && att_acc_last;
    wire [6:0]   atb_waddr = att_acc_td[6:0];
    wire [63:0]  atb_wd    = {8'b0, att_v[55:0]};
    wire [6:0]   atb_raddr = (state == ST_EXP) ? t[6:0] : 7'd0;
    wire [63:0]  atb_rd;
    msh_sram #(.DEPTH(128), .WIDTH(64)) u_atb (
        .clk(clk), .we(atb_we), .waddr(atb_waddr), .wdata(atb_wd),
        .wstrb(8'hFF),
        .re(1'b1), .raddr(atb_raddr), .rdata(atb_rd));

    wire         eb_we    = (state == ST_EXP) && exp_tag5_r[11];
    wire [6:0]   eb_waddr = exp_tag5_r[6:0];
    wire [31:0]  eb_wd    = {5'b0, exp_y_r[26:0]};
    wire [6:0]   eb_raddr = (state == ST_CTX) ? t[6:0] : 7'd0;
    wire [31:0]  eb_rd;
    msh_sram #(.DEPTH(128), .WIDTH(32)) u_eb (
        .clk(clk), .we(eb_we), .waddr(eb_waddr), .wdata(eb_wd),
        .wstrb(4'hF),
        .re(1'b1), .raddr(eb_raddr), .rdata(eb_rd));

    // ctx accumulators: 32 words x 4 x int72 (set-on-first); read at
    // the ST_CTX/ST_CTXW issue, RMW-write one cycle later at consume
    // (same word revisited 8 cycles later — no forwarding needed)
    wire [4:0]   ad_raddr = ((state == ST_CTX) || (state == ST_CTXW))
                          ? w[4:0] : 5'd0;
    wire [287:0] ad_rd;
    wire         ad_we    = (state == ST_CTX) && c_dv;
    wire [4:0]   ad_waddr = c_wd[4:0];
    wire [287:0] ad_wd;
    msh_sram #(.DEPTH(32), .WIDTH(288)) u_accd (
        .clk(clk), .we(ad_we), .waddr(ad_waddr), .wdata(ad_wd),
        .wstrb({36{1'b1}}),
        .re(1'b1), .raddr(ad_raddr), .rdata(ad_rd));
    reg [8:0]  e_td;                  // ST_EXP delayed t (att data phase)
    reg        e_dv;
    reg [11:0] exp_tag1_r, exp_tag2_r, exp_tag3_r, exp_tag4_r, exp_tag5_r;
    reg signed [56:0] exp_diff_r;
    reg [12:0] exp_idx_r;
    reg [17:0] exp_frac1_r, exp_frac2_r;
    reg signed [32:0] exp_delta_r;
    reg [31:0] exp_base_r, exp_base2_r;
    reg signed [39:0] exp_interp_r;
    reg [27:0] exp_y_r;
    reg [2:0] recip_phase;
    reg [5:0] rc_e0_r, rc_amt_r;
    reg signed [7:0] rc_E_r, rc_sh_r;
    reg [63:0] rc_mq_r, rc_ru_r;
    reg [11:0] rc_idx_r;
    reg [8:0] rc_frac_r;
    reg [23:0] rc_base_r;
    reg signed [25:0] rc_di_r, rc_r_r;
    reg [4:0]  cw_wd;                 // ST_CTXW delayed word index
    reg        cw_dv;

    // LUT ROMs (harness msh_rom, ASYNC read — same-cycle data, two
    // ports per lookup). ENDPOINT formulation: the C-model's delta
    // tables satisfy d[idx] == t[idx+1] - t[idx] for every idx (verified
    // numerically for expneg_d and recip_d, including the duplicated
    // final row whose delta is 0), so the deltas are computed on the
    // fly from tab[idx] and tab[idx+1] — bit-identical, and the two
    // _d ROMs are eliminated: deltas are computed on the fly from the
    // endpoint ROM's two async read ports. The reciprocal pipeline registers
    // its index before consuming the async endpoint data.
    wire [31:0] et_rd0, et_rd1;
    wire [23:0] rt_rd0, rt_rd1;
    wire [12:0] rom_e_idx;              // driven below with the exp index
    wire [11:0] rom_r_idx;              // driven below with the recip index
    wire [12:0] rom_e_idx_p1 = rom_e_idx + 13'd1;
    wire [11:0] rom_r_idx_p1 = rom_r_idx + 12'd1;
    msh_rom #(.DEPTH(4098), .WIDTH(32),
              .INIT_FILE("rtl/roms/expneg_msh.hex")) u_rom_et (
        .raddr1(rom_e_idx),    .rdata1(et_rd0),
        .raddr2(rom_e_idx_p1), .rdata2(et_rd1));
    msh_rom #(.DEPTH(2050), .WIDTH(24),
              .INIT_FILE("rtl/roms/recip_msh.hex")) u_rom_rt (
        .raddr1(rom_r_idx),    .rdata1(rt_rd0),
        .raddr2(rom_r_idx_p1), .rdata2(rt_rd1));
    // on-the-fly deltas (== the former _d tables by construction)
    wire signed [32:0] ed_c = $signed(et_rd1) - $signed(et_rd0);
    wire signed [24:0] rd_c = $signed(rt_rd1) - $signed(rt_rd0);

    // ---------------- helpers ----------------
    function signed [79:0] rs80(input signed [79:0] v, input [6:0] r);
        // round-half-up arithmetic shift, r >= 1
        rs80 = (v + (80'sd1 <<< (r - 7'd1))) >>> r;
    endfunction

    function [31:0] sat32(input signed [79:0] v);
        begin
            if (v > 80'sd2147483647)       sat32 = 32'h7FFFFFFF;
            else if (v < -80'sd2147483648) sat32 = 32'h80000000;
            else                           sat32 = v[31:0];
        end
    endfunction

    function [5:0] topbit64(input [63:0] v);
        reg [31:0] x32;
        reg [15:0] x16;
        reg [7:0] x8;
        reg [3:0] x4;
        reg [1:0] x2;
        reg [5:0] idx;
        begin
            idx = 6'd0;
            if (|v[63:32]) begin
                idx[5] = 1'b1;
                x32 = v[63:32];
            end else begin
                x32 = v[31:0];
            end
            if (|x32[31:16]) begin
                idx[4] = 1'b1;
                x16 = x32[31:16];
            end else begin
                x16 = x32[15:0];
            end
            if (|x16[15:8]) begin
                idx[3] = 1'b1;
                x8 = x16[15:8];
            end else begin
                x8 = x16[7:0];
            end
            if (|x8[7:4]) begin
                idx[2] = 1'b1;
                x4 = x8[7:4];
            end else begin
                x4 = x8[3:0];
            end
            if (|x4[3:2]) begin
                idx[1] = 1'b1;
                x2 = x4[3:2];
            end else begin
                x2 = x4[1:0];
            end
            idx[0] = x2[1];
            topbit64 = idx;
        end
    endfunction

    // ---------------- combinational datapath ----------------
    assign job_ready = (state == ST_IDLE);
    assign job_done  = (state == ST_DONE);

    // store source addresses (XBUF words). K side gathers kc (per-head
    // slice) then kr (shared) per head; the KV destination word offset
    // is contiguous across the concatenation.
    wire [8:0] ks_kc_src = ({7'b0, ks_hd} * {3'b0, j_nkc}) + {3'b0, ks_w};
    wire [8:0] ks_kr_src = {3'b0, ks_w};
    wire [8:0] vs_src = ({7'b0, vs_hd} * {3'b0, j_nwv}) + {3'b0, vs_w};

    // q loader source addresses (qc / qr per-head slices)
    wire [8:0] q_qc_src = ({7'b0, hd[1:0]} * {3'b0, j_nkc}) + {3'b0, q0_w};
    wire [8:0] q_qr_src = ({7'b0, hd[1:0]} * {3'b0, j_nkr}) + {3'b0, q0_w};

    assign xr0_addr = (state == ST_STORE && !ks_ph) ? {j_kcbuf, ks_kc_src}
                    : (state == ST_STORE)           ? {j_krbuf, ks_kr_src}
                    : (state == ST_QLOAD && !q0_ph) ? {j_qcbuf, q_qc_src}
                    : (state == ST_QLOAD)           ? {j_qrbuf, q_qr_src}
                    : 13'd0;
    assign xr1_addr = (state == ST_STORE) ? {j_vbuf, vs_src} : 13'd0;

    integer l, al;
    // datapath temps at TRUE value widths (multiplier operands stay
    // narrow+equal so yosys doesn't context-extend them to 80x80; see
    // ("unsigned-concat trap": zero-extend before mixing widths)
    reg signed [31:0] t_k, t_q;         // KV/q lanes (int32)
    reg signed [24:0] t_p;              // rs(e, 4): up to +2^23 (25b!)
    reg signed [79:0] t_acc, att_dot_n;
    reg signed [63:0] att_mul;
    reg [255:0] att_round_n, att_round_r, att_prod_v, att_prod_r;
    reg signed [79:0] att_pair01, att_pair23;
    reg signed [55:0] t_a1;
    reg signed [45:0] t_a2;             // rs(acc_d 72b, 28)
    reg signed [79:0] att_acc_n, att_v;
    reg signed [56:0] t_diff;
    reg [223:0]       ctx_prod_v;       // 4 x int56 (comb temp)
    reg signed [183:0] ctx_a2_n, ctx_a2_r;
    reg signed [227:0] ctx_pp_lo_n, ctx_pp_lo_r;
    reg signed [223:0] ctx_pp_hi_n, ctx_pp_hi_r;
    reg signed [271:0] ctx_prod_n, ctx_prod_r;
    reg signed [55:0] ctx_finish;
    reg [127:0] ctx_data_n;
    reg [127:0]       ctx_word;
    reg [8:0] ctx_word_a_r, ctx_word_b_r, ctx_word_c_r, ctx_word_d_r;
    reg ctx_valid_a, ctx_valid_b, ctx_valid_c, ctx_valid_d;
    wire signed [55:0] t_att_w = atb_rd[55:0];  // att word (macro read)

    /* verilator lint_off WIDTHTRUNC */
    /* verilator lint_off WIDTHEXPAND */
    always @* begin
        att_round_n = 256'd0;
        for (al = 0; al < 4; al = al + 1) begin
            att_round_n[64*al +: 32] =
                rs80($signed(k_rd[32*al +: 32]), 7'd4);
            att_round_n[64*al + 32 +: 32] =
                rs80($signed(qword_r[32*al +: 32]), 7'd4);
        end
    end
    /* verilator lint_on WIDTHEXPAND */
    /* verilator lint_on WIDTHTRUNC */

    // acc_d RMW (read issued one cycle earlier with (t, w))
    reg [287:0] ad_wd_r;
    integer ai;
    /* verilator lint_off WIDTHEXPAND */
    always @* begin
        ad_wd_r = ad_rd;
        for (ai = 0; ai < 4; ai = ai + 1)
            ad_wd_r[72*ai +: 72] = (c_td == 9'd0)
                ? $signed(ctx_prod_v[56*ai +: 56])
                : $signed(ad_rd[72*ai +: 72]) + $signed(ctx_prod_v[56*ai +: 56]);
    end
    /* verilator lint_on WIDTHEXPAND */
    assign ad_wd = ad_wd_r;
    // recip
    wire [63:0]       se_ext  = {28'b0, se};
    wire [5:0] rc_e0_c = topbit64(se_ext);
    wire signed [7:0] rc_E_c = {2'b00, rc_e0_c} - 8'sd20;
    wire [5:0] rc_amt_c = (rc_E_c >= 8'sd0)
                        ? rc_E_c[5:0] : (6'd0 - rc_E_c[5:0]);
    wire [63:0] rc_mq_c = (rc_E_r >= 8'sd0) ? (se_ext >>> rc_amt_r)
                                             : (se_ext <<< rc_amt_r);
    wire [11:0] rc_idx_c = rc_mq_r[20:9] - 12'd2048;
    wire [8:0] rc_frac_c = rc_mq_r[8:0];
    wire signed [7:0] rc_sh_c = 8'sd7 - rc_E_r;  // 46-30-(E+9)
    wire signed [25:0] rc_di_c = rd_c * $signed({1'b0, rc_frac_r});
    wire signed [25:0] rc_r_c = $signed({6'b0, rc_base_r[19:0]})
                              + ((rc_di_r + 26'sd256) >>> 9);
    wire [63:0] rc_ru_c = {38'b0, rc_r_c};
    wire [63:0] rec_n_c = (rc_sh_r >= 8'sd0)
                        ? (rc_ru_r <<< rc_sh_r[5:0])
                        : ((rc_ru_r + (64'd1 <<< (6'd0 - rc_sh_r[5:0] - 6'd1)))
                           >>> (6'd0 - rc_sh_r[5:0]));

    assign rom_e_idx = exp_idx_r;
    assign rom_r_idx = rc_idx_r;

    wire signed [31:0] exp_ei_c = (exp_diff_r > 57'sd0) ? 32'sd0
                                : (exp_diff_r < -57'sd1073741824)
                                  ? -32'sd1073741824
                                  : exp_diff_r[31:0];
    wire signed [31:0] exp_tb_c = exp_ei_c >>> 18;
    wire signed [31:0] exp_idxc_c = exp_tb_c + 32'sd4096;
    wire [12:0] exp_idx_c = (exp_idxc_c < 32'sd0) ? 13'd0
                           : (exp_idxc_c > 32'sd4096) ? 13'd4096
                           : exp_idxc_c[12:0];
    wire [17:0] exp_frac_c = exp_ei_c[17:0];
    wire signed [39:0] exp_interp_c =
        (exp_delta_r * $signed({1'b0, exp_frac2_r}) + 40'sd131072) >>> 18;
    wire [27:0] exp_y_c = {1'b0, exp_base2_r[26:0]} + exp_interp_r[27:0];

    // rs80/sat32 return 80 bits; assignments to the true-width temps
    // intentionally truncate/extend (values always fit)
    /* verilator lint_off WIDTHTRUNC */
    /* verilator lint_off WIDTHEXPAND */
    always @* begin
        ctx_a2_n = 184'sd0;
        ctx_pp_lo_n = 228'sd0;
        ctx_pp_hi_n = 224'sd0;
        ctx_prod_n = 272'sd0;
        ctx_finish = 56'sd0;
        ctx_data_n = 128'd0;
        // ---- att dot-product stage: four rounded products and reduction ----
        for (l = 0; l < 4; l = l + 1) begin
            t_k = $signed(att_round_r[64*l +: 32]);
            t_q = $signed(att_round_r[64*l + 32 +: 32]);
            att_mul = $signed(t_k) * $signed(t_q);
            att_prod_v[64*l +: 64] = att_mul;
        end
        att_pair01 = {{16{att_prod_r[63]}}, att_prod_r[63:0]}
                   + {{16{att_prod_r[127]}}, att_prod_r[127:64]};
        att_pair23 = {{16{att_prod_r[191]}}, att_prod_r[191:128]}
                   + {{16{att_prod_r[255]}}, att_prod_r[255:192]};
        t_acc = att_pair01 + att_pair23;
        att_dot_n = t_acc;
        // ---- attention accumulator stage ----
        att_acc_n = att_dot_first ? att_dot_r
                   : {{8{acc[71]}}, acc} + att_dot_r;
        // ---- completed score scaling stage ----
        t_a1 = rs80(att_acc_stage, 7'd18);
        att_v = rs80(t_a1 * $signed({1'b0, j_catt}), 7'd16);
        // ---- exp input difference; the remaining stages are pipelined ----
        t_diff = {t_att_w[55], t_att_w} - {mx[55], mx};
        // ---- ctx: per-lane products e4 * V (consumes returned V word) ----
        t_p = rs80({53'b0, eb_rd[26:0]}, 7'd4);
        for (l = 0; l < 4; l = l + 1) begin
            t_k = $signed(v_rd[32*l +: 32]);
            ctx_prod_v[56*l +: 56] = t_p * t_k;
        end
        // ---- ctx writeout: rounded operand -> split multiply -> combine -> sat ----
        for (l = 0; l < 4; l = l + 1) begin
            t_a2 = rs80($signed(ad_rd[72*l +: 72]), 7'd28);
            ctx_a2_n[46*l +: 46] = t_a2;
            ctx_pp_lo_n[57*l +: 57] =
                $signed(ctx_a2_r[46*l +: 46])
                * $signed({1'b0, rec_ctx[10:0]});
            ctx_pp_hi_n[56*l +: 56] =
                $signed(ctx_a2_r[46*l +: 46])
                * $signed({1'b0, rec_ctx[20:11]});
            ctx_prod_n[68*l +: 68] =
                $signed({{1{ctx_pp_hi_r[56*l+55]}},
                         ctx_pp_hi_r[56*l +: 56], 11'b0})
                + $signed({{11{ctx_pp_lo_r[57*l+56]}},
                           ctx_pp_lo_r[57*l +: 57]});
            ctx_finish =
                ($signed(ctx_prod_r[68*l +: 68]) + 68'sd8192) >>> 14;
            ctx_data_n[32*l +: 32] = sat32(ctx_finish);
        end
    end
    /* verilator lint_on WIDTHEXPAND */
    /* verilator lint_on WIDTHTRUNC */

    // ctx write port (acc_d read is synchronous: the writeout fires
    // one cycle after the word index issue, with cw_wd)
    wire signed [79:0] rec_ext = {{32{rec[47]}}, rec};
    // rec is non-negative and bounded below 2^21 by the reciprocal contract.
    wire signed [21:0] rec_ctx = {1'b0, rec[20:0]};
    wire signed [79:0] mx_ext  = {{24{mx[55]}}, mx};
    wire [8:0] cw_word = ({7'b0, hd[1:0]} * {3'b0, j_nwv}) + {3'b0, cw_wd};
    assign xw_we    = ((state == ST_CTXW) || (state == ST_CTXW_DRAIN))
                    && ctx_valid_d;
    assign xw_addr  = {j_cbuf, ctx_word_d_r};
    assign xw_data  = ctx_word;
    assign xw_wstrb = 4'b1111;

    // unused spare bits
    wire _unused = &{1'b0, job_pos[9], job_dqk[9:8], job_dv[9:8],
                     job_dk[9:8], job_dqk[1:0], job_dv[1:0], job_dk[1:0],
                     k_addr[13:11], v_addr[13:11],
                     ks_caddr[13:11], vs_caddr[13:11],
                     et_rd0[31:27], et_rd1[31:27],
                     atb_rd[63:56], eb_rd[31:27], rec_ext[79:48],
                      att_acc_n[79:72], att_v[79:56], exp_interp_r[39:28],
                      rc_mq_r[63:21], rec_n_c[63:48], exp_ei_c[31:18]};

    // ---------------- sequential process ----------------
    integer i;

    always @(posedge clk) begin
        if (!rst_n) begin
            state    <= ST_CLEAR;
            j_pos    <= 9'd0;
            j_nwk    <= 6'd0;
            j_nwv    <= 6'd0;
            j_dk     <= 8'd0;
            j_nkc    <= 6'd0;
            j_nkr    <= 6'd0;
            j_H      <= 4'd0;
            j_catt   <= 17'd0;
            j_qcbuf  <= 4'd0;
            j_qrbuf  <= 4'd0;
            j_kcbuf  <= 4'd0;
            j_krbuf  <= 4'd0;
            j_vbuf   <= 4'd0;
            j_cbuf   <= 4'd0;
            hd       <= 4'd0;
            t        <= 9'd0;
            w        <= 6'd0;
            acc      <= 72'sd0;
            mx       <= 56'sd0;
        se       <= 36'd0;
            rec      <= 48'sd0;
            clr      <= 17'd0;
            ks_hd    <= 2'd0;
            vs_hd    <= 2'd0;
            ks_w     <= 6'd0;
            vs_w     <= 6'd0;
            ks_ph    <= 1'b0;
            ks_vd    <= 1'b0;
            vs_vd    <= 1'b0;
            ks_hd_d  <= 2'd0;
            vs_hd_d  <= 2'd0;
            ks_w_d   <= 6'd0;
            vs_w_d   <= 6'd0;
            ks_ph_d  <= 1'b0;
            ks_done  <= 1'b0;
            vs_done  <= 1'b0;
            ks_cap   <= 1'b0;
            vs_cap   <= 1'b0;
            q0_w     <= 6'd0;
            q0_wd    <= 6'd0;
            q0_ph    <= 1'b0;
            q0_phd   <= 1'b0;
            q0_vd    <= 1'b0;
            qword_r  <= 128'd0;
            att_iss  <= 1'b0;
            at_dv    <= 1'b0;
            att_round_valid <= 1'b0;
            att_mul_valid <= 1'b0;
            att_dot_valid <= 1'b0;
            att_acc_valid <= 1'b0;
            att_round_first <= 1'b0;
            att_round_last <= 1'b0;
            att_round_tfirst <= 1'b0;
            att_round_tlast <= 1'b0;
            att_round_td <= 9'd0;
            att_mul_first <= 1'b0;
            att_mul_last <= 1'b0;
            att_mul_tfirst <= 1'b0;
            att_mul_tlast <= 1'b0;
            att_mul_td <= 9'd0;
            att_dot_first <= 1'b0;
            att_dot_last  <= 1'b0;
            att_dot_tfirst <= 1'b0;
            att_dot_tlast <= 1'b0;
            att_acc_last <= 1'b0;
            att_acc_tfirst <= 1'b0;
            att_acc_tlast <= 1'b0;
            att_dot_td <= 9'd0;
            att_acc_td <= 9'd0;
            att_dot_r <= 80'sd0;
            att_round_r <= 256'd0;
            att_prod_r <= 256'd0;
            att_acc_stage <= 80'sd0;
            ctx_iss  <= 1'b0;
            c_dv     <= 1'b0;
            e_dv     <= 1'b0;
            exp_tag1_r <= 12'd0;
            exp_tag2_r <= 12'd0;
            exp_tag3_r <= 12'd0;
            exp_tag4_r <= 12'd0;
            exp_tag5_r <= 12'd0;
            exp_diff_r <= 57'sd0;
            exp_idx_r <= 13'd0;
            exp_frac1_r <= 18'd0;
            exp_frac2_r <= 18'd0;
            exp_delta_r <= 33'sd0;
            exp_base_r <= 32'd0;
            exp_base2_r <= 32'd0;
            exp_interp_r <= 40'sd0;
            exp_y_r <= 28'd0;
            recip_phase <= 3'd0;
            rc_e0_r <= 6'd0;
            rc_amt_r <= 6'd0;
            rc_E_r <= 8'sd0;
            rc_sh_r <= 8'sd0;
            rc_mq_r <= 64'd0;
            rc_ru_r <= 64'd0;
            rc_idx_r <= 12'd0;
            rc_frac_r <= 9'd0;
            rc_base_r <= 24'd0;
            rc_di_r <= 26'sd0;
            rc_r_r <= 26'sd0;
            cw_dv    <= 1'b0;
            ctx_a2_r <= 184'sd0;
            ctx_pp_lo_r <= 228'sd0;
            ctx_pp_hi_r <= 224'sd0;
            ctx_prod_r <= 272'sd0;
            ctx_word <= 128'd0;
            ctx_word_a_r <= 9'd0;
            ctx_word_b_r <= 9'd0;
            ctx_word_c_r <= 9'd0;
            ctx_word_d_r <= 9'd0;
            ctx_valid_a <= 1'b0;
            ctx_valid_b <= 1'b0;
            ctx_valid_c <= 1'b0;
            ctx_valid_d <= 1'b0;
        end else begin
            ctx_valid_a <= (state == ST_CTXW) && cw_dv;
            ctx_valid_b <= ctx_valid_a;
            ctx_valid_c <= ctx_valid_b;
            ctx_valid_d <= ctx_valid_c;
            if ((state == ST_CTXW) && cw_dv) begin
                ctx_a2_r <= ctx_a2_n;
                ctx_word_a_r <= cw_word;
            end
            if (ctx_valid_a) begin
                ctx_pp_lo_r <= ctx_pp_lo_n;
                ctx_pp_hi_r <= ctx_pp_hi_n;
                ctx_word_b_r <= ctx_word_a_r;
            end
            if (ctx_valid_b) begin
                ctx_prod_r <= ctx_prod_n;
                ctx_word_c_r <= ctx_word_b_r;
            end
            if (ctx_valid_c) begin
                ctx_word <= ctx_data_n;
                ctx_word_d_r <= ctx_word_c_r;
            end
`ifdef MSH_DEBUG
            if (ctx_valid_d)
                $display("[mla] ctxw out w=%0d word=%032x", ctx_word_d_r,
                         ctx_word);
`endif
            exp_tag2_r <= exp_tag1_r;
            exp_tag3_r <= exp_tag2_r;
            exp_tag4_r <= exp_tag3_r;
            exp_tag5_r <= exp_tag4_r;
            if (exp_tag1_r[11]) begin
                exp_idx_r <= exp_idx_c;
                exp_frac1_r <= exp_frac_c;
            end
            if (exp_tag2_r[11]) begin
                exp_delta_r <= ed_c;
                exp_base_r <= et_rd0;
                exp_frac2_r <= exp_frac1_r;
            end
            if (exp_tag3_r[11]) begin
                exp_interp_r <= exp_interp_c;
                exp_base2_r <= exp_base_r;
            end
            if (exp_tag4_r[11])
                exp_y_r <= exp_y_c;

            att_round_valid <= at_dv;
            att_mul_valid <= att_round_valid;
            att_dot_valid <= att_mul_valid;
            att_acc_valid <= att_dot_valid;
            if (at_dv) begin
                att_round_r <= att_round_n;
                att_round_first <= (at_wd == 6'd0);
                att_round_last <= (at_wd == j_nwk - 6'd1);
                att_round_tfirst <= (at_td == 9'd0);
                att_round_tlast <= (at_td == j_pos);
                att_round_td <= at_td;
            end
            if (att_round_valid) begin
                att_prod_r <= att_prod_v;
                att_mul_first <= att_round_first;
                att_mul_last <= att_round_last;
                att_mul_tfirst <= att_round_tfirst;
                att_mul_tlast <= att_round_tlast;
                att_mul_td <= att_round_td;
            end
            if (att_mul_valid) begin
                att_dot_r <= att_dot_n;
                att_dot_first <= att_mul_first;
                att_dot_last <= att_mul_last;
                att_dot_tfirst <= att_mul_tfirst;
                att_dot_tlast <= att_mul_tlast;
                att_dot_td <= att_mul_td;
            end
            if (att_dot_valid) begin
                acc <= att_acc_n[71:0];
                att_acc_stage <= att_acc_n;
                att_acc_last <= att_dot_last;
                att_acc_tfirst <= att_dot_tfirst;
                att_acc_tlast <= att_dot_tlast;
                att_acc_td <= att_dot_td;
            end
            if (att_acc_valid && att_acc_last) begin
                if (att_acc_tfirst || (att_v > mx_ext))
                    mx <= att_v[55:0];
            end
            /* verilator lint_off BLKSEQ */
`ifdef MSH_DEBUG
            if (state != ST_CLEAR)
                $display("[mla] st=%0d hd=%0d t=%0d w=%0d", state, hd, t, w);
`endif
            case (state)
            // -------- KV SRAM clear sweep (randreset gate) --------
            ST_CLEAR: begin
                // macro writes are combinational (kw_*/vw_*)
                if (clr == 17'd1727) state <= ST_IDLE;
                else                 clr   <= clr + 17'd1;
            end

            // -------- job accept --------
            ST_IDLE: begin
                if (job_valid) begin
                    j_pos   <= job_pos[8:0];
                    j_nwk   <= job_dqk[7:2];
                    j_nwv   <= job_dv[7:2];
                    j_dk    <= job_dk[7:0];
                    j_nkc   <= job_dk[7:2];
                    j_nkr   <= job_dqk[7:2] - job_dk[7:2];
                    j_H     <= job_H;
                    j_catt  <= job_c_att;
                    j_qcbuf <= job_qc_buf;
                    j_qrbuf <= job_qr_buf;
                    j_kcbuf <= job_kc_buf;
                    j_krbuf <= job_kr_buf;
                    j_vbuf  <= job_v_buf;
                    j_cbuf  <= job_ctx_buf;
                    ks_hd   <= 2'd0;
                    vs_hd   <= 2'd0;
                    ks_w    <= 6'd0;
                    vs_w    <= 6'd0;
                    ks_ph   <= 1'b0;
                    ks_vd   <= 1'b0;
                    vs_vd   <= 1'b0;
                    ks_done <= 1'b0;
                    vs_done <= 1'b0;
                    ks_cap  <= 1'b0;
                    vs_cap  <= 1'b0;
                    state   <= ST_STORE;
                end
            end

            // -------- store k_new (port0) / v_new (port1) into KV --------
            // macro writes are combinational (kw_*/vw_* from ks_vd/vs_vd)
            ST_STORE: begin
                // K side: kc slice (phase 0) then kr shared part (phase 1)
                if (!ks_done) begin
                    ks_vd   <= 1'b1;
                    ks_hd_d <= ks_hd;
                    ks_w_d  <= ks_w;
                    ks_ph_d <= ks_ph;
                    if (!ks_ph) begin
                        if (ks_w == j_nkc - 6'd1) begin
                            ks_w  <= 6'd0;
                            ks_ph <= 1'b1;
                        end else begin
                            ks_w <= ks_w + 6'd1;
                        end
                    end else begin
                        if (ks_w == j_nkr - 6'd1) begin
                            ks_w  <= 6'd0;
                            ks_ph <= 1'b0;
                            if (ks_hd == j_H[1:0] - 2'd1) ks_done <= 1'b1;
                            else                          ks_hd   <= ks_hd + 2'd1;
                        end else begin
                            ks_w <= ks_w + 6'd1;
                        end
                    end
                end else begin
                    ks_vd <= 1'b0;
                end
                if (ks_vd) begin
                    if (ks_ph_d && (ks_hd_d == j_H[1:0] - 2'd1)
                        && (ks_w_d == j_nkr - 6'd1)) ks_cap <= 1'b1;
                end
                // V side
                if (!vs_done) begin
                    vs_vd   <= 1'b1;
                    vs_hd_d <= vs_hd;
                    vs_w_d  <= vs_w;
                    if (vs_w == j_nwv - 6'd1) begin
                        vs_w <= 6'd0;
                        if (vs_hd == j_H[1:0] - 2'd1) vs_done <= 1'b1;
                        else                          vs_hd   <= vs_hd + 2'd1;
                    end else begin
                        vs_w <= vs_w + 6'd1;
                    end
                end else begin
                    vs_vd <= 1'b0;
                end
                if (vs_vd) begin
                    if ((vs_hd_d == j_H[1:0] - 2'd1)
                        && (vs_w_d == j_nwv - 6'd1)) vs_cap <= 1'b1;
                end
                if (ks_cap && vs_cap) begin
                    hd    <= 4'd0;
                    q0_w  <= 6'd0;
                    q0_ph <= 1'b0;
                    q0_vd <= 1'b0;
                    state <= ST_QLOAD;
                end
            end

            // -------- per-head q preload (qc slice then qr slice) --------
            ST_QLOAD: begin
                if ((!q0_ph && (q0_w != j_nkc))
                    || (q0_ph && (q0_w != j_nkr))) begin
                    q0_vd  <= 1'b1;
                    q0_wd  <= q0_w;
                    q0_phd <= q0_ph;
                    q0_w   <= q0_w + 6'd1;
                end else begin
                    q0_vd <= 1'b0;
                end
                if (q0_vd) begin
                    for (i = 0; i < 4; i = i + 1) begin
                        if (!q0_phd)
                            qbuf_v[32*{q0_wd[2:0], i[1:0]} +: 32] <=
                                xr0_data[32*i +: 32];
                        else
                            qbuf_v[32*(j_dk + {3'b0, q0_wd[2:0], i[1:0]})
                                   +: 32] <= xr0_data[32*i +: 32];
                    end
                    if (!q0_phd && (q0_wd == j_nkc - 6'd1)) begin
                        q0_ph <= 1'b1;
                        q0_w  <= 6'd0;
                    end
                    if (q0_phd && (q0_wd == j_nkr - 6'd1)) begin
                        t     <= 9'd0;
                        w     <= 6'd0;
                        state <= ST_ATT;
                    end
                end
            end

            // -------- att[t] over t = 0..pos --------
            // sync KV read: issue (t, w), consume the returned word one
            // cycle later with (at_td, at_wd) — II=1, one drain cycle.
            ST_ATT: begin
                if (!att_iss) begin
                    at_dv <= 1'b1;
                    at_td <= t;
                    at_wd <= w;
                    qword_r <= qbuf_v[128*w[3:0] +: 128];
                    if (w == j_nwk - 6'd1) begin
                        w <= 6'd0;
                        if (t == j_pos) att_iss <= 1'b1;
                        else            t <= t + 9'd1;
                    end else begin
                        w <= w + 6'd1;
                    end
                end else begin
                    at_dv <= 1'b0;
                end
                if (at_dv) begin
                    if (at_wd == j_nwk - 6'd1) begin
                        // att write is combinational (atb_we/atb_wd)
`ifdef MSH_DEBUG
                        $display("[mla] att hd=%0d t=%0d acc=%0d att_v=%0d",
                                 hd, at_td, att_acc_n, att_v);
`endif
                        if (at_td == j_pos) begin
                            t       <= 9'd0;
                            att_iss <= 1'b0;
                            state   <= ST_ATT_FLUSH;
                        end
                    end
                end
            end

            ST_ATT_FLUSH: begin
                if (att_acc_valid && att_acc_last && att_acc_tlast) begin
                    t     <= 9'd0;
                    e_dv  <= 1'b0;
                    exp_tag1_r <= 12'd0;
                    exp_tag2_r <= 12'd0;
                    exp_tag3_r <= 12'd0;
                    exp_tag4_r <= 12'd0;
                    exp_tag5_r <= 12'd0;
                    state <= ST_EXP;
                end
            end

            // -------- e[t] = exp(att[t]-mx), se = sum --------
            // Sync att reads feed an II=1 exp pipeline. The last tagged
            // result both writes e_buf and ends the phase after draining.
            ST_EXP: begin
                if ({1'b0, t} <= {1'b0, j_pos}) begin
                    e_dv <= 1'b1;
                    e_td <= t;
                    t    <= t + 9'd1;
                end else begin
                    e_dv <= 1'b0;
                end
                if (e_dv) begin
                    exp_diff_r <= t_diff;
                    exp_tag1_r <= {1'b1, (e_td == 9'd0),
                                   (e_td == j_pos), e_td};
`ifdef MSH_DEBUG
                    $display("[mla] exp input hd=%0d t=%0d att=%0d mx=%0d diff=%0d",
                             hd, e_td, t_att_w, mx, t_diff);
`endif
                end else begin
                    exp_tag1_r <= 12'd0;
                end
                if (exp_tag5_r[11]) begin
`ifdef MSH_DEBUG
                    $display("[mla] exp output hd=%0d t=%0d e=%0d",
                             hd, exp_tag5_r[8:0], exp_y_r);
`endif
                    if (exp_tag5_r[10]) se <= {8'b0, exp_y_r};
                    else                se <= se + {8'b0, exp_y_r};
                    if (exp_tag5_r[9]) begin
                        t     <= 9'd0;
                        recip_phase <= 3'd0;
                        state <= ST_RECIP;
                    end
                end
            end

            // -------- rec = recip(se), staged around scalar arithmetic --------
            ST_RECIP: begin
                case (recip_phase)
                3'd0: begin
                    rc_e0_r <= rc_e0_c;
                    rc_E_r <= rc_E_c;
                    rc_amt_r <= rc_amt_c;
                    recip_phase <= 3'd1;
                end
                3'd1: begin
                    rc_mq_r <= rc_mq_c;
                    recip_phase <= 3'd2;
                end
                3'd2: begin
                    rc_idx_r <= rc_idx_c;
                    rc_frac_r <= rc_frac_c;
                    rc_sh_r <= rc_sh_c;
                    recip_phase <= 3'd3;
                end
                3'd3: begin
                    rc_di_r <= rc_di_c;
                    rc_base_r <= rt_rd0;
                    recip_phase <= 3'd4;
                end
                3'd4: begin
                    rc_r_r <= rc_r_c;
                    rc_ru_r <= rc_ru_c;
                    recip_phase <= 3'd5;
                end
                default: begin
                    rec <= rec_n_c[47:0];
                    recip_phase <= 3'd0;
                    w <= 6'd0;
                    state <= ST_CTX;
`ifdef MSH_DEBUG
                    $display("[mla] recip hd=%0d se=%0d e0=%0d E=%0d mq=%0d idx=%0d r=%0d sh=%0d rec=%0d",
                             hd, se, rc_e0_r, rc_E_r, rc_mq_r, rc_idx_r,
                             rc_r_r, rc_sh_r, rec_n_c);
`endif
                end
                endcase
            end

            // -------- acc_d += rs(e[t],4) * V[t][d] --------
            // sync KV read: issue (t, w), consume with (c_td, c_wd).
            ST_CTX: begin
                if (!ctx_iss) begin
                    c_dv <= 1'b1;
                    c_td <= t;
                    c_wd <= w;
                    if (w == j_nwv - 6'd1) begin
                        w <= 6'd0;
                        if (t == j_pos) ctx_iss <= 1'b1;
                        else            t <= t + 9'd1;
                    end else begin
                        w <= w + 6'd1;
                    end
                end else begin
                    c_dv <= 1'b0;
                end
                if (c_dv) begin
                    // acc_d RMW is combinational (ad_we/ad_waddr/ad_wd)
                    if ((c_td == j_pos) && (c_wd == j_nwv - 6'd1)) begin
                        ctx_iss <= 1'b0;
                        cw_dv   <= 1'b0;
                        state   <= ST_CTXW;
                    end
                end
            end

            // -------- ctx writeout (sync acc_d read: issue w,
            // consume the returned word next cycle with cw_wd) --------
            ST_CTXW: begin
                if ({1'b0, w} < {1'b0, j_nwv}) begin
                    cw_dv <= 1'b1;
                    cw_wd <= w[4:0];
                    w     <= w + 6'd1;
                end else begin
                    cw_dv <= 1'b0;
                end
                if (cw_dv) begin
`ifdef MSH_DEBUG
                    $display("[mla] ctxw in hd=%0d w=%0d acc0=%0d acc1=%0d acc2=%0d acc3=%0d",
                             hd, cw_wd,
                             ad_rd[72*0 +: 72], ad_rd[72*1 +: 72],
                             ad_rd[72*2 +: 72], ad_rd[72*3 +: 72]);
`endif
                    if (cw_wd == j_nwv[4:0] - 5'd1) begin
                        cw_dv <= 1'b0;
                        state <= ST_CTXW_DRAIN;
                    end
                end
            end

            ST_CTXW_DRAIN: begin
                if (!(ctx_valid_a || ctx_valid_b || ctx_valid_c || ctx_valid_d)) begin
                    if (hd == j_H - 4'd1) begin
                        state <= ST_DONE;
                    end else begin
                        hd    <= hd + 4'd1;
                        q0_w  <= 6'd0;
                        q0_ph <= 1'b0;
                        q0_vd <= 1'b0;
                        state <= ST_QLOAD;
                    end
                end
            end

            // -------- done pulse --------
            ST_DONE: begin
                state <= ST_IDLE;
            end

            default: state <= ST_IDLE;
            endcase
            /* verilator lint_on BLKSEQ */
        end
    end

endmodule

`default_nettype wire
