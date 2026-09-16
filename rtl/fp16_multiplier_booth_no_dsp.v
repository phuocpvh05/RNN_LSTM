`timescale 1ns/1ps

// Exact, four-cycle FP16 multiplier with a radix-4 Booth mantissa core.
// Subnormal inputs are flushed to signed zero to match the trained RTL model.
// No DSP primitive or inferred DSP multiplier is permitted in this module.
module fp16_multiplier_booth_no_dsp (
    input             clk,
    input             rst,
    input             en,
    input      [15:0] a_in,
    input      [15:0] b_in,
    output reg [15:0] result
);
    wire [4:0] exp_a = a_in[14:10];
    wire [4:0] exp_b = b_in[14:10];
    wire [9:0] frac_a = a_in[9:0];
    wire [9:0] frac_b = b_in[9:0];

    wire a_nan = (exp_a == 5'h1f) && (frac_a != 0);
    wire b_nan = (exp_b == 5'h1f) && (frac_b != 0);
    wire a_inf = (exp_a == 5'h1f) && (frac_a == 0);
    wire b_inf = (exp_b == 5'h1f) && (frac_b == 0);
    wire a_zero = (exp_a == 0);
    wire b_zero = (exp_b == 0);
    wire a_normal = (exp_a != 0) && (exp_a != 5'h1f);
    wire b_normal = (exp_b != 0) && (exp_b != 5'h1f);

    reg        sign_s0;
    reg  [5:0] exp_sum_s0;
    reg  [5:0] flags_s0;
    reg [10:0] mant_a_s0;
    reg [10:0] mant_b_s0;

    reg        sign_s1;
    reg  [5:0] exp_sum_s1;
    reg  [5:0] flags_s1;
    (* use_dsp = "no" *) reg signed [23:0] booth_pair01_s1;
    (* use_dsp = "no" *) reg signed [23:0] booth_pair23_s1;
    (* use_dsp = "no" *) reg signed [23:0] booth_pair45_s1;

    reg        sign_s2;
    reg  [5:0] exp_sum_s2;
    reg  [5:0] flags_s2;
    (* use_dsp = "no" *) reg [21:0] product_s2;

    // Radix-4 Booth recoding produces six constant-shift partial products.
    // Registering three pair sums removes the former six-adder combinational
    // chain; the following stage is only a three-input reduction. This costs
    // one extra clock but materially improves the no-DSP critical path.
    function signed [23:0] booth_partial;
        input [10:0] multiplicand;
        input  [2:0] code;
        input integer shift_amount;
        reg signed [23:0] positive_value;
        begin
            positive_value = $signed({1'b0, 12'd0, multiplicand});
            case (code)
                3'b001, 3'b010:
                    booth_partial = positive_value <<< shift_amount;
                3'b011:
                    booth_partial = positive_value <<< (shift_amount + 1);
                3'b100:
                    booth_partial = -(positive_value <<< (shift_amount + 1));
                3'b101, 3'b110:
                    booth_partial = -(positive_value <<< shift_amount);
                default:
                    booth_partial = 24'sd0;
            endcase
        end
    endfunction

    wire [13:0] booth_bits_s0 = {2'b00, mant_b_s0, 1'b0};
    wire signed [23:0] booth_pp0 = booth_partial(
        mant_a_s0, booth_bits_s0[2:0], 0);
    wire signed [23:0] booth_pp1 = booth_partial(
        mant_a_s0, booth_bits_s0[4:2], 2);
    wire signed [23:0] booth_pp2 = booth_partial(
        mant_a_s0, booth_bits_s0[6:4], 4);
    wire signed [23:0] booth_pp3 = booth_partial(
        mant_a_s0, booth_bits_s0[8:6], 6);
    wire signed [23:0] booth_pp4 = booth_partial(
        mant_a_s0, booth_bits_s0[10:8], 8);
    wire signed [23:0] booth_pp5 = booth_partial(
        mant_a_s0, booth_bits_s0[12:10], 10);
    wire signed [23:0] booth_product_s1 =
        booth_pair01_s1 + booth_pair23_s1 + booth_pair45_s1;

    wire norm_shift = product_s2[21];
    wire [10:0] mant_raw = norm_shift ?
        product_s2[21:11] : product_s2[20:10];
    wire guard_bit = mant_raw[0];
    wire round_bit = norm_shift ? product_s2[10] : product_s2[9];
    wire sticky_bit = norm_shift ?
        (|product_s2[9:0]) : (|product_s2[8:0]);
    wire round_up = round_bit & (guard_bit | sticky_bit);
    wire [11:0] mant_rounded = mant_raw + round_up;
    wire round_overflow = mant_rounded[11];
    wire [9:0] final_frac = round_overflow ?
        mant_rounded[10:1] : mant_rounded[9:0];

    wire [6:0] exp_temp = {1'b0, exp_sum_s2} +
        {6'd0, norm_shift} + {6'd0, round_overflow};
    wire exp_underflow = (exp_temp <= 7'd15);
    wire exp_overflow = (exp_temp > 7'd45);
    wire [4:0] final_exp = exp_temp - 7'd15;

    wire s2_a_nan = flags_s2[5];
    wire s2_b_nan = flags_s2[4];
    wire s2_a_zero = flags_s2[3];
    wire s2_b_zero = flags_s2[2];
    wire s2_a_inf = flags_s2[1];
    wire s2_b_inf = flags_s2[0];
    wire nan_result = s2_a_nan | s2_b_nan |
        (s2_a_inf & s2_b_zero) | (s2_a_zero & s2_b_inf);
    wire inf_result = ~nan_result &
        (s2_a_inf | s2_b_inf | exp_overflow);
    wire zero_result = ~nan_result & ~inf_result &
        (s2_a_zero | s2_b_zero | exp_underflow);

    wire [15:0] normal_value = {sign_s2, final_exp, final_frac};
    wire [15:0] inf_value = {sign_s2, 5'h1f, 10'd0};
    wire [15:0] zero_value = {sign_s2, 15'd0};
    wire [15:0] result_s2 = nan_result ? 16'h7e00 :
                            inf_result ? inf_value :
                            zero_result ? zero_value : normal_value;

    always @(posedge clk) begin
        if (rst) begin
            sign_s0 <= 1'b0;
            exp_sum_s0 <= 6'd0;
            flags_s0 <= 6'd0;
            mant_a_s0 <= 11'd0;
            mant_b_s0 <= 11'd0;
            sign_s1 <= 1'b0;
            exp_sum_s1 <= 6'd0;
            flags_s1 <= 6'd0;
            booth_pair01_s1 <= 24'sd0;
            booth_pair23_s1 <= 24'sd0;
            booth_pair45_s1 <= 24'sd0;
            sign_s2 <= 1'b0;
            exp_sum_s2 <= 6'd0;
            flags_s2 <= 6'd0;
            product_s2 <= 22'd0;
            result <= 16'd0;
        end else if (en) begin
            sign_s0 <= a_in[15] ^ b_in[15];
            exp_sum_s0 <= exp_a + exp_b;
            flags_s0 <= {a_nan, b_nan, a_zero, b_zero, a_inf, b_inf};
            mant_a_s0 <= a_normal ? {1'b1, frac_a} : 11'd0;
            mant_b_s0 <= b_normal ? {1'b1, frac_b} : 11'd0;

            sign_s1 <= sign_s0;
            exp_sum_s1 <= exp_sum_s0;
            flags_s1 <= flags_s0;
            booth_pair01_s1 <= booth_pp0 + booth_pp1;
            booth_pair23_s1 <= booth_pp2 + booth_pp3;
            booth_pair45_s1 <= booth_pp4 + booth_pp5;

            sign_s2 <= sign_s1;
            exp_sum_s2 <= exp_sum_s1;
            flags_s2 <= flags_s1;
            product_s2 <= booth_product_s1[21:0];

            result <= result_s2;
        end
    end
endmodule
