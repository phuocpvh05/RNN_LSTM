`timescale 1ns/1ps

// Pipelined reduction of nine FP16 values.  ei_adder_fp16_v2 has five cycles
// of latency, so the unpaired ninth value is delayed by 15 cycles before the
// fourth and final adder stage.
module fp16_reduce9_fp16 (
    input              clk,
    input              rst,
    input      [143:0] values_i,
    input              valid_i,
    output     [15:0]  sum_o,
    output             valid_o
);
    localparam integer ADD_LATENCY = 5;
    wire [15:0] s1 [0:3];
    wire [15:0] s2 [0:1];
    wire [15:0] s3;
    wire [15:0] ninth_d1;
    wire [15:0] ninth_d2;
    wire [15:0] ninth_d3;

    genvar pair;
    generate
        for (pair = 0; pair < 4; pair = pair + 1) begin : GEN_STAGE1
            ei_adder_fp16_v2 u_add (
                .sys_clk(clk), .rst(rst), .en(1'b1),
                .a_in(values_i[(pair*2)*16 +: 16]),
                .b_in(values_i[(pair*2+1)*16 +: 16]),
                .sum_out(s1[pair])
            );
        end
        for (pair = 0; pair < 2; pair = pair + 1) begin : GEN_STAGE2
            ei_adder_fp16_v2 u_add (
                .sys_clk(clk), .rst(rst), .en(1'b1),
                .a_in(s1[pair*2]), .b_in(s1[pair*2+1]),
                .sum_out(s2[pair])
            );
        end
    endgenerate

    ei_adder_fp16_v2 u_stage3 (
        .sys_clk(clk), .rst(rst), .en(1'b1),
        .a_in(s2[0]), .b_in(s2[1]), .sum_out(s3)
    );
    ei_adder_fp16_v2 u_stage4 (
        .sys_clk(clk), .rst(rst), .en(1'b1),
        .a_in(s3), .b_in(ninth_d3), .sum_out(sum_o)
    );

    fp16_stream_delay #(.WIDTH(16), .DEPTH(ADD_LATENCY)) u_ninth_d1 (
        .clk(clk), .din(values_i[8*16 +: 16]), .dout(ninth_d1));
    fp16_stream_delay #(.WIDTH(16), .DEPTH(ADD_LATENCY)) u_ninth_d2 (
        .clk(clk), .din(ninth_d1), .dout(ninth_d2));
    fp16_stream_delay #(.WIDTH(16), .DEPTH(ADD_LATENCY)) u_ninth_d3 (
        .clk(clk), .din(ninth_d2), .dout(ninth_d3));
    fp16_stream_delay #(.WIDTH(1), .DEPTH(4*ADD_LATENCY)) u_valid_delay (
        .clk(clk), .din(valid_i), .dout(valid_o));
endmodule
