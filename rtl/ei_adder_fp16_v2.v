`timescale 1ps/1ps
// =====================================================
// FP16 Adder — Optimized v2 for Maximum Fmax
//
// Thay đổi so với bản gốc (ei_adder_fp16):
//   1. CLA adder/subtractor thay ripple-carry → O(log N) delay
//   2. Pipeline 5 thanh ghi (s01, s12, s23, s34, output)
//      — Stage 2 cũ (nặng nhất) tách thành S2 + S3
//   3. Tree-based LZC thay sequential → O(log N) delay
//   4. Loại bỏ u_sub_ba không dùng (tiết kiệm diện tích)
//
// Latency:  5 cycles tính từ lần chốt input đến output hợp lệ
// Throughput: 1 result/cycle (không đổi)
// Fmax ước tính: ~4-5× bản gốc
// =====================================================
module ei_adder_fp16_v2 (
    input              sys_clk,
    input              rst,   // sync, active-high
    input              en,
    input      [15:0]  a_in,
    input      [15:0]  b_in,
    output     [15:0]  sum_out
);

    // =========================================================
    // STAGE 0 (comb): unpack + special + exp_adj + mant11
    //                 + compare + pick big/small
    // =========================================================
    wire sign_a = a_in[15];
    wire sign_b = b_in[15];

    wire [4:0] exp_a  = a_in[14:10];
    wire [4:0] exp_b  = b_in[14:10];

    wire [9:0] frac_a = a_in[9:0];
    wire [9:0] frac_b = b_in[9:0];

    wire is_a_nan, is_b_nan, is_a_inf, is_b_inf;
    ei_fp16_is_nan u_a_nan (.x(a_in), .is_nan(is_a_nan));
    ei_fp16_is_nan u_b_nan (.x(b_in), .is_nan(is_b_nan));
    ei_fp16_is_inf u_a_inf (.x(a_in), .is_inf(is_a_inf));
    ei_fp16_is_inf u_b_inf (.x(b_in), .is_inf(is_b_inf));

    wire sign_opposite = sign_a ^ sign_b;
    wire both_inf      = is_a_inf & is_b_inf;
    wire any_inf       = is_a_inf | is_b_inf;

    wire is_nan0 = is_a_nan | is_b_nan | (both_inf & sign_opposite);
    wire is_inf0 = any_inf & ~is_nan0;

    localparam [15:0] NAN_PATTERN = 16'h7E00;
    wire [15:0] sum_nan0 = NAN_PATTERN;

    wire [15:0] inf_from_a0 = {sign_a, 5'b11111, 10'b0};
    wire [15:0] inf_from_b0 = {sign_b, 5'b11111, 10'b0};

    wire sel_inf_from_b0 = is_b_inf & ~is_a_inf;
    wire [15:0] sum_inf0;
    ei_mux16 u_mux_inf_sel0 (
        .d0(inf_from_a0), .d1(inf_from_b0),
        .s (sel_inf_from_b0),
        .y (sum_inf0)
    );

    // exp_adj for alignment (subnormal exp=0 treated as 1)
    wire exp_a_is_zero = ~(|exp_a);
    wire exp_b_is_zero = ~(|exp_b);

    wire [4:0] exp_a_adj5, exp_b_adj5;
    ei_muxN #(.WIDTH(5)) u_exp_a_adj (.d0(5'd1), .d1(exp_a), .s(~exp_a_is_zero), .y(exp_a_adj5));
    ei_muxN #(.WIDTH(5)) u_exp_b_adj (.d0(5'd1), .d1(exp_b), .s(~exp_b_is_zero), .y(exp_b_adj5));

    // mant 11 (hidden=1 for normal, hidden=0 for subnormal)
    wire [10:0] mant_a_11, mant_b_11;
    ei_muxN #(.WIDTH(11)) u_mant_a_11 (.d0({1'b0, frac_a}), .d1({1'b1, frac_a}), .s(~exp_a_is_zero), .y(mant_a_11));
    ei_muxN #(.WIDTH(11)) u_mant_b_11 (.d0({1'b0, frac_b}), .d1({1'b1, frac_b}), .s(~exp_b_is_zero), .y(mant_b_11));

    // compare exp_adj then mant to pick |big| — dùng CLA subtractor
    wire [5:0] exp_a6 = {1'b0, exp_a_adj5};
    wire [5:0] exp_b6 = {1'b0, exp_b_adj5};

    wire [5:0] diff_ab;
    wire borrow_ab;
    ei_subtractorN_cla #(.WIDTH(6)) u_sub_ab (.a(exp_a6), .b(exp_b6), .diff(diff_ab), .borrow(borrow_ab));

    wire exp_a_gt_b = ~borrow_ab & (|diff_ab);
    wire exp_a_eq_b = ~borrow_ab & ~(|diff_ab);

    wire [10:0] mant_diff_ab;
    wire mant_borrow_ab;
    ei_subtractorN_cla #(.WIDTH(11)) u_sub_mant_ab (.a(mant_a_11), .b(mant_b_11), .diff(mant_diff_ab), .borrow(mant_borrow_ab));

    wire mant_a_gt_b = ~mant_borrow_ab & (|mant_diff_ab);
    wire mant_a_eq_b = ~mant_borrow_ab & ~(|mant_diff_ab);

    wire use_a_big0 = exp_a_gt_b | (exp_a_eq_b & (mant_a_gt_b | mant_a_eq_b));
    wire same_sign0 = ~(sign_a ^ sign_b);

    wire sign_big0;
    ei_mux2 u_sign_big0 (.d0(sign_b), .d1(sign_a), .s(use_a_big0), .y(sign_big0));

    // pick exp_big_adj / exp_small_adj
    wire [4:0] exp_big_adj5_0, exp_small_adj5_0;
    ei_muxN #(.WIDTH(5)) u_exp_big0   (.d0(exp_b_adj5), .d1(exp_a_adj5), .s(use_a_big0), .y(exp_big_adj5_0));
    ei_muxN #(.WIDTH(5)) u_exp_small0 (.d0(exp_a_adj5), .d1(exp_b_adj5), .s(use_a_big0), .y(exp_small_adj5_0));

    wire [5:0] exp_big6_0   = {1'b0, exp_big_adj5_0};
    wire [5:0] exp_small6_0 = {1'b0, exp_small_adj5_0};

    // pick mant_big / mant_small
    wire [10:0] mant_big_11_0, mant_small_11_0;
    ei_muxN #(.WIDTH(11)) u_mant_big0   (.d0(mant_b_11), .d1(mant_a_11), .s(use_a_big0), .y(mant_big_11_0));
    ei_muxN #(.WIDTH(11)) u_mant_small0 (.d0(mant_a_11), .d1(mant_b_11), .s(use_a_big0), .y(mant_small_11_0));

    // =========================================================
    // REG: S0 -> S1  (70 bit)
    // =========================================================
    localparam S01W = 70;
    wire [S01W-1:0] s01_d = {
        is_nan0, is_inf0, same_sign0, sign_big0,
        sum_nan0, sum_inf0,
        exp_big6_0, exp_small6_0,
        mant_big_11_0, mant_small_11_0
    };
    wire [S01W-1:0] s01_q;

    regN #(.WIDTH(S01W)) u_s01 (
        .clk(sys_clk), .rst(rst), .en(en),
        .d(s01_d), .q(s01_q)
    );

    wire is_nan1, is_inf1, same_sign1, sign_big1;
    wire [15:0] sum_nan1, sum_inf1;
    wire [5:0] exp_big6_1, exp_small6_1;
    wire [10:0] mant_big_11_1, mant_small_11_1;

    assign {
        is_nan1, is_inf1, same_sign1, sign_big1,
        sum_nan1, sum_inf1,
        exp_big6_1, exp_small6_1,
        mant_big_11_1, mant_small_11_1
    } = s01_q;

    // =========================================================
    // STAGE 1 (comb): align (shr + sticky)
    // =========================================================
    wire [13:0] mant_big_14_1   = {mant_big_11_1,   3'b000};
    wire [13:0] mant_small_14_1 = {mant_small_11_1, 3'b000};

    wire [5:0] exp_diff6_1;
    wire expdiff_borrow_unused1;
    ei_subtractorN_cla #(.WIDTH(6)) u_exp_diff1 (
        .a(exp_big6_1), .b(exp_small6_1),
        .diff(exp_diff6_1), .borrow(expdiff_borrow_unused1)
    );

    wire [13:0] small_shr14_1;
    wire small_sticky_1;
    ei_barrel_shr_sticky #(.WIDTH(14), .SHAMT_W(6)) u_align_small1 (
        .d_in (mant_small_14_1),
        .shamt(exp_diff6_1),
        .d_out(small_shr14_1),
        .sticky(small_sticky_1)
    );

    wire [13:0] mant_small_align14_1 = {small_shr14_1[13:1], (small_shr14_1[0] | small_sticky_1)};
    wire [13:0] mant_big_align14_1   = mant_big_14_1;

    // =========================================================
    // REG: S1 -> S2  (70 bit)
    // =========================================================
    localparam S12W = 70;
    wire [S12W-1:0] s12_d = {
        is_nan1, is_inf1, same_sign1, sign_big1,
        sum_nan1, sum_inf1,
        exp_big6_1,
        mant_big_align14_1, mant_small_align14_1
    };
    wire [S12W-1:0] s12_q;

    regN #(.WIDTH(S12W)) u_s12 (
        .clk(sys_clk), .rst(rst), .en(en),
        .d(s12_d), .q(s12_q)
    );

    wire is_nan2, is_inf2, same_sign2, sign_big2;
    wire [15:0] sum_nan2, sum_inf2;
    wire [5:0] exp_big6_2;
    wire [13:0] mant_big_align14_2, mant_small_align14_2;

    assign {
        is_nan2, is_inf2, same_sign2, sign_big2,
        sum_nan2, sum_inf2,
        exp_big6_2,
        mant_big_align14_2, mant_small_align14_2
    } = s12_q;

    // =========================================================
    // STAGE 2 (comb): add/sub + partial normalize
    // — Chỉ normalize, KHÔNG xử lý underflow (dành cho S3)
    // — Đây là nửa đầu của S2 gốc, giảm critical path ~50%
    // =========================================================
    wire [14:0] big15_2   = {1'b0, mant_big_align14_2};
    wire [14:0] small15_2 = {1'b0, mant_small_align14_2};

    // --- Add (CLA, song song với Sub) ---
    wire [14:0] add15_2;
    wire add_cout_unused2;
    ei_adderN_cla #(.WIDTH(15)) u_add15_2 (
        .a(big15_2), .b(small15_2), .sum(add15_2), .cout(add_cout_unused2)
    );

    // --- Sub (CLA, song song với Add) ---
    wire [14:0] sub15_2;
    wire sub_borrow_unused2;
    ei_subtractorN_cla #(.WIDTH(15)) u_sub15_2 (
        .a(big15_2), .b(small15_2), .diff(sub15_2), .borrow(sub_borrow_unused2)
    );

    // --- Add path normalize ---
    wire add_need_shr1_2 = add15_2[14];
    wire [13:0] add_mant14_noshr_2 = add15_2[13:0];
    wire [13:0] add_mant14_shr_2   = add15_2[14:1];
    wire [13:0] add_mant14_norm_2  = {add_mant14_shr_2[13:1], (add_mant14_shr_2[0] | add15_2[0])};

    wire [13:0] mant_add_pre14_2;
    ei_muxN #(.WIDTH(14)) u_add_norm_mux2 (
        .d0(add_mant14_noshr_2),
        .d1(add_mant14_norm_2),
        .s (add_need_shr1_2),
        .y (mant_add_pre14_2)
    );

    wire [5:0] exp_add_pre6_2;
    wire exp_add_cout_unused2;
    ei_adderN_cla #(.WIDTH(6)) u_exp_add_inc2 (
        .a(exp_big6_2),
        .b({5'd0, add_need_shr1_2}),
        .sum(exp_add_pre6_2),
        .cout(exp_add_cout_unused2)
    );

    // --- Sub path normalize (LZC tree + left shift) ---
    wire [13:0] sub_mag14_2 = sub15_2[13:0];
    wire sub_is_zero_2 = ~(|sub_mag14_2);

    wire [3:0] lzc_sub4_2;
    ei_lzcN_tree #(.WIDTH(14)) u_lzc_sub2 (
        .x(sub_mag14_2),
        .count(lzc_sub4_2),
        .all_zero()              // không dùng tại đây
    );

    wire [13:0] sub_mant_shifted14_2 = sub_mag14_2 << lzc_sub4_2;

    wire [5:0] lzc_sub6_2 = {2'b00, lzc_sub4_2};

    wire [5:0] exp_sub_pre6_2;
    wire exp_sub_borrow2;
    ei_subtractorN_cla #(.WIDTH(6)) u_exp_sub_dec2 (
        .a(exp_big6_2),
        .b(lzc_sub6_2),
        .diff(exp_sub_pre6_2),
        .borrow(exp_sub_borrow2)
    );

    // shift_excess = lzc - exp_big (dùng cho underflow ở S3)
    wire [5:0] shift_excess6_2;
    wire shift_excess_borrow_unused2;
    ei_subtractorN_cla #(.WIDTH(6)) u_shift_excess2 (
        .a(lzc_sub6_2),
        .b(exp_big6_2),
        .diff(shift_excess6_2),
        .borrow(shift_excess_borrow_unused2)
    );

    // =========================================================
    // REG: S2 -> S3  (84 bit — NEW register, tách từ S2 gốc)
    // =========================================================
    localparam S23W = 84;
    wire [S23W-1:0] s23_d = {
        is_nan2, is_inf2, same_sign2, sign_big2,        // 4
        sum_nan2, sum_inf2,                              // 32
        mant_add_pre14_2, exp_add_pre6_2,                // 20
        sub_mant_shifted14_2, exp_sub_pre6_2,            // 20
        exp_sub_borrow2, sub_is_zero_2,                  // 2
        shift_excess6_2                                  // 6
    };                                                   // total: 84
    wire [S23W-1:0] s23_q;

    regN #(.WIDTH(S23W)) u_s23 (
        .clk(sys_clk), .rst(rst), .en(en),
        .d(s23_d), .q(s23_q)
    );

    wire is_nan3, is_inf3, same_sign3, sign_big3;
    wire [15:0] sum_nan3, sum_inf3;
    wire [13:0] mant_add_pre14_3;
    wire [5:0]  exp_add_pre6_3;
    wire [13:0] sub_mant_shifted14_3;
    wire [5:0]  exp_sub_pre6_3;
    wire exp_sub_borrow3, sub_is_zero3;
    wire [5:0]  shift_excess6_3;

    assign {
        is_nan3, is_inf3, same_sign3, sign_big3,
        sum_nan3, sum_inf3,
        mant_add_pre14_3, exp_add_pre6_3,
        sub_mant_shifted14_3, exp_sub_pre6_3,
        exp_sub_borrow3, sub_is_zero3,
        shift_excess6_3
    } = s23_q;

    // =========================================================
    // STAGE 3 (comb): choose add/sub + underflow to subnormal
    // — Nửa sau của S2 gốc
    // =========================================================

    // choose core add/sub
    wire [13:0] mant_core14_3;
    wire [5:0]  exp_core6_3;

    ei_muxN #(.WIDTH(14)) u_mant_core3 (
        .d0(sub_mant_shifted14_3),
        .d1(mant_add_pre14_3),
        .s (same_sign3),
        .y (mant_core14_3)
    );

    ei_muxN #(.WIDTH(6)) u_exp_core3 (
        .d0(exp_sub_pre6_3),
        .d1(exp_add_pre6_3),
        .s (same_sign3),
        .y (exp_core6_3)
    );

    wire exp_neg_3   = same_sign3 ? 1'b0 : exp_sub_borrow3;
    wire core_zero_3 = same_sign3 ? (~(|mant_add_pre14_3)) : sub_is_zero3;

    // sign rule: exact zero + opposite-sign => +0
    wire sign_zero_3 = same_sign3 ? sign_big3 : 1'b0;
    wire sign_core_3 = core_zero_3 ? sign_zero_3 : sign_big3;

    // underflow handling to subnormal
    wire exp_is_0_3 = (~exp_neg_3) & (~(|exp_core6_3));
    wire exp_is_1_3 = (~exp_neg_3) & (exp_core6_3 == 6'd1);
    wire exp_ge_2_3 = (~exp_neg_3) & (|exp_core6_3[5:1]);

    wire mant_hidden_is_1_3 = mant_core14_3[13];

    wire is_normal_out_3   = exp_ge_2_3 | (exp_is_1_3 & mant_hidden_is_1_3);
    wire is_sub_needshift_3 = exp_neg_3 | exp_is_0_3;

    // shift_amt
    wire [5:0] shift_amt_exp0_3 = 6'd1;

    wire [5:0] shift_amt_neg_plus1_3;
    wire shift_amt_cout_unused3;
    ei_adderN_cla #(.WIDTH(6)) u_shift_amt_neg3 (
        .a(shift_excess6_3),
        .b(6'd1),
        .sum(shift_amt_neg_plus1_3),
        .cout(shift_amt_cout_unused3)
    );

    wire [5:0] shift_amt6_3 =
        exp_neg_3  ? shift_amt_neg_plus1_3 :
        exp_is_0_3 ? shift_amt_exp0_3 :
        6'd0;

    wire [13:0] under_shr14_3;
    wire under_sticky_3;
    ei_barrel_shr_sticky #(.WIDTH(14), .SHAMT_W(6)) u_underflow_shr3 (
        .d_in (mant_core14_3),
        .shamt(shift_amt6_3),
        .d_out(under_shr14_3),
        .sticky(under_sticky_3)
    );

    wire [13:0] mant_after_under14_3 = {under_shr14_3[13:1], (under_shr14_3[0] | under_sticky_3)};
    wire [13:0] mant_pre_round14_3   = is_sub_needshift_3 ? mant_after_under14_3 : mant_core14_3;

    wire [5:0] exp_pre_round6_3 = is_normal_out_3 ? exp_core6_3 : 6'd0;

    // =========================================================
    // REG: S3 -> S4  (57 bit)
    // =========================================================
    localparam S34W = 57;
    wire [S34W-1:0] s34_d = {
        is_nan3, is_inf3,
        sum_nan3, sum_inf3,
        sign_core_3,
        is_normal_out_3,
        exp_pre_round6_3,
        mant_pre_round14_3,
        core_zero_3
    };
    wire [S34W-1:0] s34_q;

    regN #(.WIDTH(S34W)) u_s34 (
        .clk(sys_clk), .rst(rst), .en(en),
        .d(s34_d), .q(s34_q)
    );

    wire is_nan4, is_inf4;
    wire [15:0] sum_nan4, sum_inf4;
    wire sign_core4;
    wire is_normal_out4;
    wire [5:0] exp_pre_round6_4;
    wire [13:0] mant_pre_round14_4;
    wire core_zero4;

    assign {
        is_nan4, is_inf4,
        sum_nan4, sum_inf4,
        sign_core4,
        is_normal_out4,
        exp_pre_round6_4,
        mant_pre_round14_4,
        core_zero4
    } = s34_q;

    // =========================================================
    // STAGE 4 (comb): RNE rounding + pack + overflow + final mux
    // =========================================================
    wire [10:0] mant_keep11_4 = mant_pre_round14_4[13:3];
    wire g4 = mant_pre_round14_4[2];
    wire r4 = mant_pre_round14_4[1];
    wire s4 = mant_pre_round14_4[0];

    wire inc4 = g4 & (r4 | s4 | mant_keep11_4[0]);

    wire [10:0] mant_sum11_4;
    wire mant_sum_cout4;
    ei_adderN_cla #(.WIDTH(11)) u_round_add4 (
        .a(mant_keep11_4),
        .b({10'd0, inc4}),
        .sum(mant_sum11_4),
        .cout(mant_sum_cout4)
    );

    wire [10:0] mant_rounded11_4 =
        mant_sum_cout4 ? {1'b1, mant_sum11_4[10:1]} : mant_sum11_4;

    wire exp_inc_round4 = mant_sum_cout4;

    wire [5:0] exp_post_round6_4;
    wire exp_post_cout_unused4;
    ei_adderN_cla #(.WIDTH(6)) u_exp_round_inc4 (
        .a(exp_pre_round6_4),
        .b({5'd0, exp_inc_round4}),
        .sum(exp_post_round6_4),
        .cout(exp_post_cout_unused4)
    );

    wire rounded_is_zero4 = ~(|mant_rounded11_4);

    wire sub_promote4 = (exp_pre_round6_4 == 6'd0) & (mant_rounded11_4[10] == 1'b1);

    wire [4:0] exp_pack5_4 =
        (core_zero4 | rounded_is_zero4) ? 5'd0 :
        sub_promote4                    ? 5'd1 :
        exp_post_round6_4[4:0];

    wire [9:0] frac_pack10_4 =
        (core_zero4 | rounded_is_zero4) ? 10'd0 :
        sub_promote4                    ? 10'd0 :
        mant_rounded11_4[9:0];

    wire exp_is_31_4 = (exp_post_round6_4[4:0] == 5'b11111);
    wire exp_ge_31_4 = exp_post_round6_4[5] | exp_is_31_4;

    wire will_be_inf4 = is_normal_out4 & exp_ge_31_4;

    wire [15:0] packed_finite4 = {sign_core4, exp_pack5_4, frac_pack10_4};
    wire [15:0] packed_inf4    = {sign_core4, 5'b11111, 10'b0};

    wire [15:0] sum_normal4;
    ei_mux16 u_mux_overflow4 (
        .d0(packed_finite4),
        .d1(packed_inf4),
        .s (will_be_inf4),
        .y (sum_normal4)
    );

    wire [15:0] r_inf_normal4;
    ei_mux16 u_mux_inf_vs_normal4 (
        .d0(sum_normal4),
        .d1(sum_inf4),
        .s (is_inf4),
        .y (r_inf_normal4)
    );

    wire [15:0] sum_final4;
    ei_mux16 u_mux_nan_vs_rest4 (
        .d0(r_inf_normal4),
        .d1(sum_nan4),
        .s (is_nan4),
        .y (sum_final4)
    );

    // =========================================================
    // OUTPUT REGISTER
    // =========================================================
    regN #(.WIDTH(16)) u_reg_out (
        .clk(sys_clk), .rst(rst), .en(en),
        .d(sum_final4),
        .q(sum_out)
    );

endmodule
