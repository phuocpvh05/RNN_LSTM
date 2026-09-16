`timescale 1ns/1ps

module ei_fp16_is_nan (
    input  [15:0] x,       // FP16: [15]=sign, [14:10]=exp, [9:0]=frac
    output        is_nan
);
    wire [4:0] exp  = x[14:10];
    wire [9:0] frac = x[9:0];

    // exp_all_ones = 1 nếu exp = 11111
    wire exp_all_ones  = &exp;     // = exp[4] & exp[3] & exp[2] & exp[1] & exp[0]

    // frac_nonzero = 1 nếu tồn tại ít nhất 1 bit mantissa = 1
    wire frac_nonzero  = |frac;    // = frac[9] | ... | frac[0]

    // NaN khi exponent toàn 1 và mantissa != 0
    assign is_nan = exp_all_ones & frac_nonzero;

endmodule
