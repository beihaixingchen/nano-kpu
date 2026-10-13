// msh_deq.v -- embedding-row dequant gather.
//
// Matches rtl/selfmodel/fxmodel.py emb_row BIT-EXACTLY:
//   y[2j]   = sat32((lo_j - z[j//g]) * s[j//g] << 14)
//   y[2j+1] = sat32((hi_j - z[j//g]) * s[j//g] << 14)
// lo_j/hi_j = nibbles of byte j of the token row; g = min(group_size, d);
// <<14 maps the raw (q-z)*s product (sx=0) to Q4.26 (R = -14).
//
// Consumes three byte-stream segments in order (s row, z row, q row)
// with the msh_fetch byte-stream contract, then writes d_model/4 128-bit
// words (4 int32 lanes) to the Y port. d_model is a multiple of 4 in all
// ladder configs.
//
// Staged rows live in packed vectors (no inferred $mem — macros_only).

`default_nettype none

module msh_deq (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         job_valid,
    output wire         job_ready,
    input  wire [15:0]  job_d,          // d_model (mult of 4)
    input  wire [7:0]   job_gsz,        // min(group_size, d)
    input  wire [3:0]   job_ybuf,
    // byte stream (from msh_fetch)
    input  wire         ws_valid,
    output wire         ws_ready,
    input  wire [127:0] ws_data,
    input  wire [4:0]   ws_count,
    input  wire [3:0]   ws_shift,
    input  wire         ws_last,
    // Y write port (128b words, 4 int32 lanes)
    output wire         y_we,
    output wire [11:0]  y_addr,         // word index
    output wire [127:0] y_data,
    output wire [3:0]   y_buf,
    output wire         done
);
    localparam S_IDLE  = 3'd0, S_S = 3'd1, S_Z = 3'd2, S_Q = 3'd3,
               S_WR    = 3'd4, S_WFLUSH = 3'd5, S_WDONE = 3'd6;

    reg [2:0]  state;
    reg [15:0] j_d;
    reg [7:0]  j_gsz;
    reg [3:0]  j_ybuf;

    // staged row data (d/2 <= 128 bytes packed q, ng <= 2 of int16 s, ng z)
    reg [1023:0] qrow_v;                // 128 x 8b
    reg [31:0]   srow_v;                // 2 x 16b
    reg [15:0]   zrow_v;                // 2 x 8b

    // byte-serial beat emitter (one latched beat in flight)
    reg [127:0] beat_r;                 // shift-aligned beat bytes
    reg [4:0]   beat_cnt;               // bytes remaining to emit
    reg         beat_last;              // latched beat ends the segment

    reg [15:0] pos;                     // bytes consumed in current segment
    reg [15:0] wr_i;                    // first element index of output word
    reg signed [5:0]  op_qz0_r, op_qz1_r, op_qz2_r, op_qz3_r;
    reg signed [15:0] op_s0_r, op_s1_r, op_s2_r, op_s3_r;
    reg                op_valid_r, op_last_r;
    reg [11:0]         op_addr_r;
    reg                out_valid_r, out_last_r;
    reg [11:0]         out_addr_r;
    reg [127:0]        out_data_r;

    assign job_ready = (state == S_IDLE);
    // byte-serial intake: accept a beat only when the previous one is
    // fully emitted (one variable-index byte demux instead of 16
    // parallel ones — the demuxes were 96K gates of the module)
    wire emit_idle = (beat_cnt == 5'd0);
    assign ws_ready  = ((state == S_S) || (state == S_Z)
                        || (state == S_Q)) && emit_idle;
    wire ws_fire = ws_valid && ws_ready;

    // Stage 1 selects the nibble, group zero-point and scale. Keeping the
    // multiplier out of this stage is important for the 1 ns screen.
    function automatic [21:0] deq_operands(input [15:0] idx);
        reg [6:0]  f_j;
        reg [7:0]  f_qb;
        reg [3:0]  f_nib;
        reg        f_g;
        reg [7:0]  f_zb;
        reg signed [5:0]  f_qz;
        reg signed [15:0] f_s;
        begin
            f_j   = idx[7:1];
            f_qb  = qrow_v[8*f_j +: 8];
            f_nib = idx[0] ? f_qb[7:4] : f_qb[3:0];
            f_g   = (idx >= {8'b0, j_gsz});
            f_zb  = f_g ? zrow_v[15:8] : zrow_v[7:0];
            f_qz  = $signed({2'b0, f_nib}) - $signed({2'b0, f_zb[3:0]});
            f_s   = f_g ? $signed(srow_v[31:16]) : $signed(srow_v[15:0]);
            deq_operands = {f_s, f_qz};
        end
    endfunction

    function automatic [31:0] deq_product(
        input signed [5:0]  f_qz,
        input signed [15:0] f_s);
        reg signed [21:0] f_acc;
        begin
            f_acc = f_qz * f_s;
            if (f_acc > 22'sd131071)
                deq_product = 32'h7FFFFFFF;
            else if (f_acc < -22'sd131072)
                deq_product = 32'h80000000;
            else
                deq_product = {f_acc[17:0], 14'b0};
        end
    endfunction

    wire [21:0] w_op0 = deq_operands(wr_i);
    wire [21:0] w_op1 = deq_operands(wr_i + 16'd1);
    wire [21:0] w_op2 = deq_operands(wr_i + 16'd2);
    wire [21:0] w_op3 = deq_operands(wr_i + 16'd3);
    wire [31:0] w_y0 = deq_product(op_qz0_r, op_s0_r);
    wire [31:0] w_y1 = deq_product(op_qz1_r, op_s1_r);
    wire [31:0] w_y2 = deq_product(op_qz2_r, op_s2_r);
    wire [31:0] w_y3 = deq_product(op_qz3_r, op_s3_r);

    assign y_we   = out_valid_r;
    assign y_addr = out_addr_r;
    assign y_data = out_data_r;
    assign y_buf  = j_ybuf;
    assign done   = out_valid_r && out_last_r;
`ifdef MSH_DEBUG
    always @(posedge clk) begin
        if (job_valid || state != S_IDLE || y_we)
            $display("[deq] st=%0d pos=%0d wr_i=%0d j_d=%0d ws_v=%0d ws_r=%0d ws_last=%0d y_we=%0d done=%0d",
                     state, pos, wr_i, j_d, ws_valid, ws_ready, ws_last, y_we, done);
    end
`endif
    /* verilator lint_off BLKSEQ */
    always @(posedge clk) begin
        if (!rst_n) begin
            state  <= S_IDLE;
            j_d    <= 16'd0;
            j_gsz  <= 8'd0;
            j_ybuf <= 4'd0;
            pos    <= 16'd0;
            wr_i   <= 16'd0;
            beat_cnt  <= 5'd0;
            beat_last <= 1'b0;
            op_qz0_r <= 6'sd0;
            op_qz1_r <= 6'sd0;
            op_qz2_r <= 6'sd0;
            op_qz3_r <= 6'sd0;
            op_s0_r  <= 16'sd0;
            op_s1_r  <= 16'sd0;
            op_s2_r  <= 16'sd0;
            op_s3_r  <= 16'sd0;
            op_valid_r <= 1'b0;
            op_last_r  <= 1'b0;
            op_addr_r  <= 12'd0;
            out_valid_r <= 1'b0;
            out_last_r  <= 1'b0;
            out_addr_r  <= 12'd0;
            out_data_r  <= 128'd0;
        end else begin
            out_valid_r <= 1'b0;
            out_last_r  <= 1'b0;

            // -------- beat latch (one beat in flight) --------
            if (ws_fire) begin
                beat_r    <= ws_data >> {ws_shift, 3'b000};
                beat_cnt  <= ws_count;
                beat_last <= ws_last;
            end

            // -------- four-byte beat emit --------
            if (beat_cnt != 5'd0) begin
                if (state == S_Q) begin
                    qrow_v[8*pos[6:0] +: 8] <= beat_r[7:0];
                    if (beat_cnt > 5'd1)
                        qrow_v[8*(pos[6:0] + 7'd1) +: 8] <= beat_r[15:8];
                    if (beat_cnt > 5'd2)
                        qrow_v[8*(pos[6:0] + 7'd2) +: 8] <= beat_r[23:16];
                    if (beat_cnt > 5'd3)
                        qrow_v[8*(pos[6:0] + 7'd3) +: 8] <= beat_r[31:24];
                end else if (state == S_S) begin
                    srow_v[8*pos[1:0] +: 8] <= beat_r[7:0];
                    if (beat_cnt > 5'd1)
                        srow_v[8*(pos[1:0] + 2'd1) +: 8] <= beat_r[15:8];
                    if (beat_cnt > 5'd2)
                        srow_v[8*(pos[1:0] + 2'd2) +: 8] <= beat_r[23:16];
                    if (beat_cnt > 5'd3)
                        srow_v[8*(pos[1:0] + 2'd3) +: 8] <= beat_r[31:24];
                end else begin
                    zrow_v[8*pos[0] +: 8] <= beat_r[7:0];
                    if (beat_cnt > 5'd1)
                        zrow_v[8*(pos[0] + 1'b1) +: 8] <= beat_r[15:8];
                end
                if (beat_cnt > 5'd4) begin
                    beat_r   <= {32'b0, beat_r[127:32]};
                    beat_cnt <= beat_cnt - 5'd4;
                    pos      <= pos + 16'd4;
                end else begin
                    beat_r   <= 128'd0;
                    beat_cnt <= 5'd0;
                    pos      <= pos + {11'd0, beat_cnt};
                end
                if ((beat_cnt <= 5'd4) && beat_last) begin
                    pos   <= 16'd0;
                    wr_i  <= 16'd0;
                    op_valid_r <= 1'b0;
                    state <= (state == S_S) ? S_Z
                           : (state == S_Z) ? S_Q : S_WR;
                end
            end

            if (state == S_WR) begin
                // Stage 2 emits the previous word while stage 1 captures the
                // next four operands. This keeps one output word/cycle while
                // removing the variable-index muxes from the multiplier path.
                if (op_valid_r) begin
                    out_valid_r <= 1'b1;
                    out_last_r  <= op_last_r;
                    out_addr_r  <= op_addr_r;
                    out_data_r  <= {w_y3, w_y2, w_y1, w_y0};
                end
                op_qz0_r <= $signed(w_op0[5:0]);
                op_qz1_r <= $signed(w_op1[5:0]);
                op_qz2_r <= $signed(w_op2[5:0]);
                op_qz3_r <= $signed(w_op3[5:0]);
                op_s0_r  <= $signed(w_op0[21:6]);
                op_s1_r  <= $signed(w_op1[21:6]);
                op_s2_r  <= $signed(w_op2[21:6]);
                op_s3_r  <= $signed(w_op3[21:6]);
                op_valid_r <= 1'b1;
                op_last_r  <= (wr_i + 16'd4 >= j_d);
                op_addr_r  <= {2'b0, wr_i[11:2]};
                if (wr_i + 16'd4 >= j_d) state <= S_WFLUSH;
                else                     wr_i  <= wr_i + 16'd4;
            end

            if (state == S_WFLUSH) begin
                if (op_valid_r) begin
                    out_valid_r <= 1'b1;
                    out_last_r  <= op_last_r;
                    out_addr_r  <= op_addr_r;
                    out_data_r  <= {w_y3, w_y2, w_y1, w_y0};
                end
                op_valid_r <= 1'b0;
                state <= S_WDONE;
            end

            if (state == S_WDONE) begin
                state <= S_IDLE;
            end

            if ((state == S_IDLE) && job_valid) begin
                j_d    <= job_d;
                j_gsz  <= job_gsz;
                j_ybuf <= job_ybuf;
                pos    <= 16'd0;
                op_valid_r <= 1'b0;
                state  <= S_S;
            end
        end
    end
    /* verilator lint_on BLKSEQ */
endmodule

`default_nettype wire
