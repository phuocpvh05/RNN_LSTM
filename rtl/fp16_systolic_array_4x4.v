`timescale 1ns/1ps

// 4x4 systolic array with nine physical MAC lanes in every PE.
// One request covers 4 rows x (4 columns x 9 inner products) = 144 MACs.
// The PE lanes remain independent across the four columns and are reduced
// only at the end of a row.  This keeps the PE itself to exactly nine MACs.
module fp16_systolic_array_4x4 #(
    parameter integer PE_LATENCY = 9,
    parameter integer TAG_WIDTH  = 8,
    parameter integer MAC_LANES  = 9
) (
    input                       clk,
    input                       rst,
    input                       in_valid,
    input      [TAG_WIDTH-1:0]  in_tag,
    input      [2303:0]         weights_i,
    input      [575:0]          vector_i,
    input      [63:0]           accumulator_i,
    output                      out_valid,
    output     [TAG_WIDTH-1:0]  out_tag,
    output     [63:0]           result_o
);
    localparam integer PE_VECTOR_WIDTH = MAC_LANES*16;
    localparam integer ROW_PE_LATENCY  = 4*PE_LATENCY;
    localparam integer REDUCE_LATENCY  = 20;
    localparam integer TOTAL_LATENCY   = ROW_PE_LATENCY+REDUCE_LATENCY;

    wire [PE_VECTOR_WIDTH-1:0] delayed_vector [0:3];
    wire [PE_VECTOR_WIDTH-1:0] delayed_weight [0:15];
    wire [PE_VECTOR_WIDTH-1:0] pe_psum_out [0:15];
    wire                        pe_valid_out [0:15];
    wire [PE_VECTOR_WIDTH-1:0] unused_activation [0:15];
    wire [3:0] reduce_valid;

    genvar row;
    genvar col;
    generate
        for (col = 0; col < 4; col = col + 1) begin : GEN_VECTOR_DELAY
            fp16_stream_delay #(
                .WIDTH(PE_VECTOR_WIDTH),
                .DEPTH(col*PE_LATENCY)
            ) u_vector_delay (
                .clk(clk),
                .din(vector_i[col*PE_VECTOR_WIDTH +: PE_VECTOR_WIDTH]),
                .dout(delayed_vector[col])
            );
        end

        for (row = 0; row < 4; row = row + 1) begin : GEN_ROWS
            for (col = 0; col < 4; col = col + 1) begin : GEN_COLS
                localparam integer INDEX = row*4 + col;
                wire [PE_VECTOR_WIDTH-1:0] psum_in;
                wire                       valid_in;

                fp16_stream_delay #(
                    .WIDTH(PE_VECTOR_WIDTH),
                    .DEPTH(col*PE_LATENCY)
                ) u_weight_delay (
                    .clk(clk),
                    .din(weights_i[INDEX*PE_VECTOR_WIDTH +: PE_VECTOR_WIDTH]),
                    .dout(delayed_weight[INDEX])
                );

                if (col == 0) begin : GEN_FIRST_COLUMN
                    // Add the scalar accumulator exactly once, in lane zero.
                    assign psum_in = {
                        {(PE_VECTOR_WIDTH-16){1'b0}},
                        accumulator_i[row*16 +: 16]
                    };
                    assign valid_in = in_valid;
                end else begin : GEN_LATER_COLUMNS
                    assign psum_in = pe_psum_out[INDEX-1];
                    assign valid_in = pe_valid_out[INDEX-1];
                end

                fp16_systolic_pe #(
                    .MAC_LANES(MAC_LANES),
                    .MAC_LATENCY(PE_LATENCY),
                    .MUL_LATENCY(4)
                ) u_pe (
                    .clk(clk), .rst(rst),
                    .weight_i(delayed_weight[INDEX]),
                    .activation_i(delayed_vector[col]),
                    .psum_i(psum_in), .valid_i(valid_in),
                    .activation_o(unused_activation[INDEX]),
                    .psum_o(pe_psum_out[INDEX]),
                    .valid_o(pe_valid_out[INDEX])
                );
            end

            fp16_reduce9_fp16 u_reduce_row (
                .clk(clk), .rst(rst),
                .values_i(pe_psum_out[row*4+3]),
                .valid_i(pe_valid_out[row*4+3]),
                .sum_o(result_o[row*16 +: 16]),
                .valid_o(reduce_valid[row])
            );
        end
    endgenerate

    fp16_stream_delay #(
        .WIDTH(TAG_WIDTH), .DEPTH(TOTAL_LATENCY)
    ) u_tag_delay (
        .clk(clk), .din(in_tag), .dout(out_tag)
    );

    assign out_valid = reduce_valid[0];
endmodule

// Always-running delay line. DEPTH=0 becomes a wire bypass. No reset is used,
// allowing long delays to map to SRLs; valid controls whether data is useful.
module fp16_stream_delay #(
    parameter integer WIDTH = 16,
    parameter integer DEPTH = 1
) (
    input                  clk,
    input      [WIDTH-1:0] din,
    output     [WIDTH-1:0] dout
);
    generate
        if (DEPTH == 0) begin : GEN_BYPASS
            assign dout = din;
        end else begin : GEN_PIPE
            (* shreg_extract = "yes" *) reg [WIDTH-1:0] pipe [0:DEPTH-1];
            integer index;
            always @(posedge clk) begin
                pipe[0] <= din;
                for (index = 1; index < DEPTH; index = index + 1)
                    pipe[index] <= pipe[index-1];
            end
            assign dout = pipe[DEPTH-1];
        end
    endgenerate
endmodule
