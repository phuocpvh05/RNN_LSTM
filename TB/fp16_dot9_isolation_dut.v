`timescale 1ns/1ps

module fp16_dot9_isolation_dut (
    input               clk,
    input               rst,

    input               mul_en,
    input      [15:0]   mul_a,
    input      [15:0]   mul_b,
    output     [15:0]   mul_result,

    input               add_en,
    input      [15:0]   add_a,
    input      [15:0]   add_b,
    output     [15:0]   add_result,

    input               pe_valid_i,
    input      [143:0]  pe_weight_i,
    input      [143:0]  pe_activation_i,
    input      [143:0]  pe_psum_i,
    output     [143:0]  pe_psum_o,
    output              pe_valid_o,

    input               reduce_valid_i,
    input      [143:0]  reduce_values_i,
    output     [15:0]   reduce_sum_o,
    output              reduce_valid_o,

    input               array_valid_i,
    input      [7:0]    array_tag_i,
    input      [2303:0] array_weights_i,
    input      [575:0]  array_vector_i,
    input      [63:0]   array_accumulator_i,
    output              array_valid_o,
    output     [7:0]    array_tag_o,
    output     [63:0]   array_result_o
);

    wire [143:0] unused_activation_o;

    fp16_multiplier_booth_no_dsp u_multiplier (
        .clk(clk), .rst(rst), .en(mul_en),
        .a_in(mul_a), .b_in(mul_b), .result(mul_result)
    );

    ei_adder_fp16_v2 u_adder (
        .sys_clk(clk), .rst(rst), .en(add_en),
        .a_in(add_a), .b_in(add_b), .sum_out(add_result)
    );

    fp16_systolic_pe #(
        .MAC_LANES(9), .MAC_LATENCY(9), .MUL_LATENCY(4)
    ) u_pe (
        .clk(clk), .rst(rst),
        .weight_i(pe_weight_i),
        .activation_i(pe_activation_i),
        .psum_i(pe_psum_i),
        .valid_i(pe_valid_i),
        .activation_o(unused_activation_o),
        .psum_o(pe_psum_o),
        .valid_o(pe_valid_o)
    );

    fp16_reduce9_fp16 u_reduce (
        .clk(clk), .rst(rst),
        .values_i(reduce_values_i),
        .valid_i(reduce_valid_i),
        .sum_o(reduce_sum_o),
        .valid_o(reduce_valid_o)
    );

    fp16_systolic_array_4x4 #(
        .PE_LATENCY(9), .TAG_WIDTH(8), .MAC_LANES(9)
    ) u_array (
        .clk(clk), .rst(rst),
        .in_valid(array_valid_i),
        .in_tag(array_tag_i),
        .weights_i(array_weights_i),
        .vector_i(array_vector_i),
        .accumulator_i(array_accumulator_i),
        .out_valid(array_valid_o),
        .out_tag(array_tag_o),
        .result_o(array_result_o)
    );

endmodule
