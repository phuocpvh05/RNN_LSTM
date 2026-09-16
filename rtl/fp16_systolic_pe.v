`timescale 1ns/1ps

// Nine-lane FP16 MAC processing element.
// Each physical PE contains nine independent FP16 multipliers and adders.
// The nine partial sums remain separate while travelling through the four
// systolic columns.  They are reduced once at the end of each array row.
// This avoids placing an eleven-adder dot-product tree inside every PE.
module fp16_systolic_pe #(
    parameter integer MAC_LANES   = 9,
    parameter integer MAC_LATENCY = 9,
    parameter integer MUL_LATENCY = 4
) (
    input                         clk,
    input                         rst,
    input      [MAC_LANES*16-1:0] weight_i,
    input      [MAC_LANES*16-1:0] activation_i,
    input      [MAC_LANES*16-1:0] psum_i,
    input                         valid_i,
    output     [MAC_LANES*16-1:0] activation_o,
    output     [MAC_LANES*16-1:0] psum_o,
    output                        valid_o
);
    localparam integer ADD_LATENCY = MAC_LATENCY - MUL_LATENCY;

    wire [15:0] product [0:MAC_LANES-1];
    wire [15:0] sum [0:MAC_LANES-1];
    reg  [MAC_LANES*16-1:0] activation_delay [0:MAC_LATENCY-1];
    // Flattened memory keeps this source compatible with Verilog-2001; a
    // two-dimensional unpacked array would require SystemVerilog file mode.
    reg  [15:0] psum_delay [0:MAC_LANES*MUL_LATENCY-1];
    reg  [MAC_LATENCY-1:0] valid_delay;

    integer delay_index;
    integer lane_index;
    always @(posedge clk) begin
        if (rst) begin
            valid_delay <= {MAC_LATENCY{1'b0}};
            for (delay_index = 0; delay_index < MAC_LATENCY;
                 delay_index = delay_index + 1)
                activation_delay[delay_index] <= {(MAC_LANES*16){1'b0}};
            for (lane_index = 0; lane_index < MAC_LANES;
                 lane_index = lane_index + 1)
                for (delay_index = 0; delay_index < MUL_LATENCY;
                     delay_index = delay_index + 1)
                    psum_delay[lane_index*MUL_LATENCY+delay_index] <= 16'd0;
        end else begin
            valid_delay[0] <= valid_i;
            activation_delay[0] <= activation_i;
            for (lane_index = 0; lane_index < MAC_LANES;
                 lane_index = lane_index + 1)
                psum_delay[lane_index*MUL_LATENCY] <=
                    psum_i[lane_index*16 +: 16];

            for (delay_index = 1; delay_index < MAC_LATENCY;
                 delay_index = delay_index + 1) begin
                valid_delay[delay_index] <= valid_delay[delay_index-1];
                activation_delay[delay_index] <= activation_delay[delay_index-1];
            end
            for (lane_index = 0; lane_index < MAC_LANES;
                 lane_index = lane_index + 1)
                for (delay_index = 1; delay_index < MUL_LATENCY;
                     delay_index = delay_index + 1)
                    psum_delay[lane_index*MUL_LATENCY+delay_index] <=
                        psum_delay[lane_index*MUL_LATENCY+delay_index-1];
        end
    end

    genvar lane;
    generate
        for (lane = 0; lane < MAC_LANES; lane = lane + 1) begin : GEN_MAC_LANES
            fp16_multiplier_booth_no_dsp u_multiplier (
                .clk(clk),
                .rst(rst),
                .en(1'b1),
                .a_in(weight_i[lane*16 +: 16]),
                .b_in(activation_i[lane*16 +: 16]),
                .result(product[lane])
            );

            ei_adder_fp16_v2 u_adder (
                .sys_clk(clk),
                .rst(rst),
                .en(1'b1),
                .a_in(product[lane]),
                .b_in(psum_delay[lane*MUL_LATENCY+MUL_LATENCY-1]),
                .sum_out(sum[lane])
            );

            assign psum_o[lane*16 +: 16] = sum[lane];
        end
    endgenerate

    // The multiplier has four registered stages and ei_adder_fp16_v2 has
    // five registered stages (s01, s12, s23, s34 and output).  Keep valid
    // aligned to the resulting 4+5=9-cycle datapath.
    // explicit.  It is intentionally checked at elaboration time.
    initial begin
        if (ADD_LATENCY != 5)
            $error("fp16_systolic_pe expects the five-cycle ei_adder_fp16_v2");
    end

    assign activation_o = activation_delay[MAC_LATENCY-1];
    assign valid_o = valid_delay[MAC_LATENCY-1];
endmodule
