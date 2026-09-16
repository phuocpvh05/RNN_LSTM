`timescale 1ns/1ps

module ei_fp16_is_inf (
    input  [15:0] x,       // FP16: [15]=sign, [14:10]=exp, [9:0]=frac
    output        is_inf
);
    wire [4:0] exp  = x[14:10];
    wire [9:0] frac = x[9:0];

    // exp_all_ones = 1 nếu exp = 11111
    wire exp_all_ones  = &exp;     // = exp[4] & exp[3] & ... & exp[0]

    // frac_all_zero = 1 nếu frac = 0000000000
    wire frac_all_zero = ~(|frac); // = ~(frac[9] | ... | frac[0])

    // Inf khi exponent toàn 1 và mantissa = 0
    assign is_inf = exp_all_ones & frac_all_zero;

endmodule
