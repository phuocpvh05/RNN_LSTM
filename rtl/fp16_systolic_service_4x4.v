`timescale 1ns/1ps

// Shared no-DSP systolic service.
//
// Legacy start/done interface:
//   tile_mode=0 : eight independent element-wise FP16 MACs.  The low and high
//                 groups are injected on consecutive clocks.
//   tile_mode=1 : one 4x4 matrix-vector tile.
//
// Streaming interface:
//   one tagged 4x4 tile can be accepted every clock while stream_ready=1.
module fp16_systolic_service_4x4 #(
    parameter integer STREAM_TAG_WIDTH = 6
) (
    input              clk,
    input              rst,

    input              start,
    input              tile_mode,
    input      [127:0] a_vec,
    input      [127:0] b_vec,
    input      [127:0] acc_vec,
    input      [255:0] tile_weights,
    input       [63:0] tile_vector,
    input       [63:0] tile_accumulator,
    output reg [127:0] result_vec,
    output reg  [63:0] tile_result,
    output             busy,
    output reg          done,

    input                         stream_valid,
    input      [STREAM_TAG_WIDTH-1:0] stream_tag,
    input      [2303:0]           stream_weights,
    input      [575:0]            stream_vector,
    input       [63:0]            stream_accumulator,
    output                        stream_ready,
    output                        stream_out_valid,
    output     [STREAM_TAG_WIDTH-1:0] stream_out_tag,
    output      [63:0]            stream_result
);

    localparam S_IDLE       = 3'd0;
    localparam S_ISSUE_LOW  = 3'd1;
    localparam S_ISSUE_HIGH = 3'd2;
    localparam S_WAIT_VEC   = 3'd3;
    localparam S_ISSUE_TILE = 3'd4;
    localparam S_WAIT_TILE  = 3'd5;

    localparam [7:0] TAG_LEGACY_LOW  = 8'h80;
    localparam [7:0] TAG_LEGACY_HIGH = 8'h81;
    localparam [7:0] TAG_LEGACY_TILE = 8'h82;

    reg [2:0] state;
    reg [127:0] a_r;
    reg [127:0] b_r;
    reg [127:0] acc_r;
    reg [255:0] tile_weights_r;
    reg  [63:0] tile_vector_r;
    reg  [63:0] tile_accumulator_r;

    reg          array_in_valid;
    reg  [7:0]   array_in_tag;
    reg [2303:0] array_weights;
    reg [575:0]  array_vector;
    reg  [63:0]  array_accumulator;
    wire         array_out_valid;
    wire [7:0]   array_out_tag;
    wire [63:0]  array_result;

    integer row_index;
    integer col_index;

    assign busy = (state != S_IDLE);
    assign stream_ready = (state == S_IDLE);
    assign stream_out_valid = array_out_valid && !array_out_tag[7];
    assign stream_out_tag = array_out_tag[STREAM_TAG_WIDTH-1:0];
    assign stream_result = array_result;

    // Combinational request mux. Legacy issue states have priority.
    always @(*) begin
        array_in_valid = 1'b0;
        array_in_tag = 8'd0;
        array_weights = 2304'd0;
        array_vector = 576'd0;
        array_accumulator = 64'd0;

        if (state == S_ISSUE_LOW || state == S_ISSUE_HIGH) begin
            array_in_valid = 1'b1;
            array_in_tag = (state == S_ISSUE_LOW) ?
                           TAG_LEGACY_LOW : TAG_LEGACY_HIGH;

            for (row_index = 0; row_index < 4;
                 row_index = row_index + 1) begin
                for (col_index = 0; col_index < 4;
                     col_index = col_index + 1) begin
                    if (row_index == col_index) begin
                        if (state == S_ISSUE_LOW)
                            array_weights[((row_index*4+col_index)*9)*16 +: 16] =
                                a_r[row_index*16 +: 16];
                        else
                            array_weights[((row_index*4+col_index)*9)*16 +: 16] =
                                a_r[(row_index+4)*16 +: 16];
                    end
                end

                if (state == S_ISSUE_LOW) begin
                    array_vector[(row_index*9)*16 +: 16] =
                        b_r[row_index*16 +: 16];
                    array_accumulator[row_index*16 +: 16] =
                        acc_r[row_index*16 +: 16];
                end else begin
                    array_vector[(row_index*9)*16 +: 16] =
                        b_r[(row_index+4)*16 +: 16];
                    array_accumulator[row_index*16 +: 16] =
                        acc_r[(row_index+4)*16 +: 16];
                end
            end
        end else if (state == S_ISSUE_TILE) begin
            array_in_valid = 1'b1;
            array_in_tag = TAG_LEGACY_TILE;
            // Legacy scalar 4x4 tiles occupy lane zero of every dot9 PE.
            for (row_index = 0; row_index < 4;
                 row_index = row_index + 1) begin
                array_vector[(row_index*9)*16 +: 16] =
                    tile_vector_r[row_index*16 +: 16];
                for (col_index = 0; col_index < 4;
                     col_index = col_index + 1)
                    array_weights[((row_index*4+col_index)*9)*16 +: 16] =
                        tile_weights_r[(row_index*4+col_index)*16 +: 16];
            end
            array_accumulator = tile_accumulator_r;
        end else if (state == S_IDLE && stream_valid) begin
            array_in_valid = 1'b1;
            array_in_tag = {2'b00, stream_tag};
            array_weights = stream_weights;
            array_vector = stream_vector;
            array_accumulator = stream_accumulator;
        end
    end

    fp16_systolic_array_4x4 #(
        .PE_LATENCY(9),
        .TAG_WIDTH(8)
    ) u_array (
        .clk(clk),
        .rst(rst),
        .in_valid(array_in_valid),
        .in_tag(array_in_tag),
        .weights_i(array_weights),
        .vector_i(array_vector),
        .accumulator_i(array_accumulator),
        .out_valid(array_out_valid),
        .out_tag(array_out_tag),
        .result_o(array_result)
    );

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            a_r <= 128'd0;
            b_r <= 128'd0;
            acc_r <= 128'd0;
            tile_weights_r <= 256'd0;
            tile_vector_r <= 64'd0;
            tile_accumulator_r <= 64'd0;
            result_vec <= 128'd0;
            tile_result <= 64'd0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;

            if (array_out_valid && array_out_tag[7]) begin
                case (array_out_tag)
                    TAG_LEGACY_LOW:
                        result_vec[63:0] <= array_result;
                    TAG_LEGACY_HIGH: begin
                        result_vec[127:64] <= array_result;
                        done <= 1'b1;
                        state <= S_IDLE;
                    end
                    TAG_LEGACY_TILE: begin
                        tile_result <= array_result;
                        done <= 1'b1;
                        state <= S_IDLE;
                    end
                    default: begin end
                endcase
            end

            case (state)
                S_IDLE: begin
                    if (start) begin
                        if (tile_mode) begin
                            tile_weights_r <= tile_weights;
                            tile_vector_r <= tile_vector;
                            tile_accumulator_r <= tile_accumulator;
                            tile_result <= 64'd0;
                            state <= S_ISSUE_TILE;
                        end else begin
                            a_r <= a_vec;
                            b_r <= b_vec;
                            acc_r <= acc_vec;
                            result_vec <= 128'd0;
                            state <= S_ISSUE_LOW;
                        end
                    end
                end

                S_ISSUE_LOW:
                    state <= S_ISSUE_HIGH;

                S_ISSUE_HIGH:
                    state <= S_WAIT_VEC;

                S_ISSUE_TILE:
                    state <= S_WAIT_TILE;

                S_WAIT_VEC: begin end
                S_WAIT_TILE: begin end

                default:
                    state <= S_IDLE;
            endcase
        end
    end

endmodule
