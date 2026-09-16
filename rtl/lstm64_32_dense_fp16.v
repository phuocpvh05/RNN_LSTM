`timescale 1ns/1ps

// FP16 inference controller using one shared 4x4 systolic MAC service:
//   LSTM(9,64) -> LSTM(64,32) -> concat(stats48) -> Dense(80,6).
//
// Weight memories use the row-tile/column packed format produced by
// train_lstm_fp16.py. Element 0 occupies bits [15:0].
module lstm64_32_dense_fp16 #(
    parameter INPUT_FILE       = "input_sample_fp16.mem",
    parameter INPUT_PACKED_FILE = "input_sample_packed32.mem",
    parameter LSTM1_W_BANK0_FILE = "lstm1_weight_bank0_fp16.mem",
    parameter LSTM1_W_BANK1_FILE = "lstm1_weight_bank1_fp16.mem",
    parameter LSTM1_W_BANK2_FILE = "lstm1_weight_bank2_fp16.mem",
    parameter LSTM1_W_BANK3_FILE = "lstm1_weight_bank3_fp16.mem",
    parameter LSTM1_B_FILE     = "lstm1_bias_packed_fp16.mem",
    parameter LSTM2_W_BANK0_FILE = "lstm2_weight_bank0_fp16.mem",
    parameter LSTM2_W_BANK1_FILE = "lstm2_weight_bank1_fp16.mem",
    parameter LSTM2_W_BANK2_FILE = "lstm2_weight_bank2_fp16.mem",
    parameter LSTM2_W_BANK3_FILE = "lstm2_weight_bank3_fp16.mem",
    parameter LSTM2_B_FILE     = "lstm2_bias_packed_fp16.mem",
    parameter DENSE_W_BANK0_FILE = "dense_weight_bank0_fp16.mem",
    parameter DENSE_W_BANK1_FILE = "dense_weight_bank1_fp16.mem",
    parameter DENSE_W_BANK2_FILE = "dense_weight_bank2_fp16.mem",
    parameter DENSE_W_BANK3_FILE = "dense_weight_bank3_fp16.mem",
    parameter LSTM1_W_DOT9_FILE = "lstm1_weight_dot9_fp16.mem",
    parameter LSTM2_W_DOT9_FILE = "lstm2_weight_dot9_fp16.mem",
    parameter DENSE_W_DOT9_FILE = "dense_weight_dot9_fp16.mem",
    parameter DENSE_B_FILE     = "dense_bias_packed_fp16.mem",
    parameter SIGMOID_FILE     = "sigmoid_lut_fp16.mem",
    parameter TANH_FILE        = "tanh_lut_fp16.mem",
    parameter integer POOL_MEAN = 0
) (
    input              clk,
    input              rst,
    input              start,
    input              input_write_enable,
    input      [9:0]   input_write_word_addr,
    input      [31:0]  input_write_data,
    input      [3:0]   input_write_strb,
    input      [767:0] stats_features,
    output reg         busy,
    output reg         done,
    output reg [2:0]   prediction,
    output reg [31:0]  cycle_count,
    output reg [31:0]  debug_input_word0,
    output wire [95:0] debug_logits,
    output wire [319:0] debug_pipeline,
    output reg         mac_start,
    output reg [127:0] mac_a,
    output reg [127:0] mac_b,
    output reg [127:0] mac_acc,
    output reg         mac_tile_mode,
    output reg [255:0] mac_tile_weights,
    output reg [63:0]  mac_tile_vector,
    output reg [63:0]  mac_tile_accumulator,
    input      [127:0] mac_result,
    input      [63:0]  mac_tile_result,
    input              mac_busy,
    input              mac_done,

    // Fully-pipelined matrix-tile stream. The shared systolic service accepts
    // one independent 4x4 tile per clock and returns the same tag in order.
    output reg         mac_stream_valid,
    output reg [5:0]   mac_stream_tag,
    output reg [2303:0] mac_stream_weights,
    output reg [575:0]  mac_stream_vector,
    output reg [63:0]  mac_stream_accumulator,
    input              mac_stream_ready,
    input              mac_stream_out_valid,
    input      [5:0]   mac_stream_out_tag,
    input      [63:0]  mac_stream_result
);

    localparam [15:0] FP16_ZERO   = 16'h0000;
    localparam [15:0] FP16_ONE    = 16'h3C00;
    localparam [15:0] FP16_INV128 = 16'h2000;

    localparam M_L1_MAT  = 4'd0;
    localparam M_L1_FC   = 4'd1;
    localparam M_L1_IG   = 4'd2;
    localparam M_L1_H    = 4'd3;
    localparam M_L2_MAT  = 4'd4;
    localparam M_L2_FC   = 4'd5;
    localparam M_L2_IG   = 4'd6;
    localparam M_L2_H    = 4'd7;
    localparam M_L2_SUM  = 4'd8;
    localparam M_POOL    = 4'd9;
    localparam M_DENSE   = 4'd10;

    localparam S_IDLE          = 5'd0;
    localparam S_L1_INIT       = 5'd1;
    localparam S_L1_PREP       = 5'd2;
    localparam S_L1_CELL_PREP  = 5'd3;
    localparam S_L2_INIT       = 5'd4;
    localparam S_L2_PREP       = 5'd5;
    localparam S_L2_CELL_PREP  = 5'd6;
    localparam S_L2_SUM_PREP   = 5'd7;
    localparam S_POOL_INIT     = 5'd8;
    localparam S_POOL_PREP     = 5'd9;
    localparam S_DENSE_INIT    = 5'd10;
    localparam S_DENSE_PREP    = 5'd11;
    localparam S_MAC_START     = 5'd12;
    localparam S_MAC_WAIT      = 5'd13;
    localparam S_ARGMAX        = 5'd14;
    localparam S_DONE          = 5'd15;
    localparam S_WEIGHT_WAIT   = 5'd16; // retained for legacy compatibility
    localparam S_L1_DRAIN      = 5'd17;
    localparam S_L2_DRAIN      = 5'd18;
    localparam S_DENSE_DRAIN   = 5'd19;
    localparam S_INPUT_LOAD    = 5'd20;
    localparam S_INPUT_WAIT    = 5'd21;
    localparam S_INPUT_CAPTURE = 5'd22;
    localparam S_ARGMAX_MERGE  = 5'd23;
    localparam S_ARGMAX_FINAL  = 5'd24;
    // Split the recurrent-cell operand path into two registered stages.  The
    // first stage selects the gate/cell words; the second performs the LUT
    // lookup and drives the MAC service.  This removes the long variable-array
    // select -> activation LUT -> MAC-input path seen after implementation.
    localparam S_L1_CELL_APPLY = 5'd25;
    localparam S_L2_CELL_APPLY = 5'd26;

    localparam STREAM_L1       = 2'd0;
    localparam STREAM_L2       = 2'd1;
    localparam STREAM_DENSE    = 2'd2;
    // Dot9 request covers 36 inner products per output row. The four-PE
    // chain plus row reduction is 64 cycles, so a tile is never reused before
    // its previous accumulator has returned.
    localparam integer STREAM_REUSE_GAP = 68;
    localparam integer L1_STREAM_TOTAL = 64*3;
    localparam integer L2_STREAM_TOTAL = 32*3;
    localparam integer DENSE_STREAM_TOTAL = 2*3;
    // Two consecutive FP16 samples share one 32-bit word.  Matching the AXI
    // write width gives this memory a single byte-enabled write address and
    // allows Vivado to infer BRAM instead of 18k input-buffer flip-flops.
    (* ram_style = "block" *)
    reg [31:0] input_mem [0:575];
    (* rom_style = "block" *) reg [2303:0] lstm1_weight_dot9 [0:191];
    (* rom_style = "block" *)
    reg [127:0] lstm1_bias_mem [0:31];
    (* rom_style = "block" *) reg [2303:0] lstm2_weight_dot9 [0:95];
    (* rom_style = "block" *)
    reg [127:0] lstm2_bias_mem [0:15];
    (* rom_style = "block" *) reg [2303:0] dense_weight_dot9 [0:5];
    (* rom_style = "block" *)
    reg [127:0] dense_bias_mem [0:0];
    (* rom_style = "block" *)
    reg [15:0] sigmoid_lut_mem [0:255];
    (* rom_style = "block" *)
    reg [15:0] tanh_lut_mem [0:255];

    initial begin
        $readmemh(INPUT_PACKED_FILE, input_mem);
        $readmemh(LSTM1_W_DOT9_FILE, lstm1_weight_dot9);
        $readmemh(LSTM1_B_FILE, lstm1_bias_mem);
        $readmemh(LSTM2_W_DOT9_FILE, lstm2_weight_dot9);
        $readmemh(LSTM2_B_FILE, lstm2_bias_mem);
        $readmemh(DENSE_W_DOT9_FILE, dense_weight_dot9);
        $readmemh(DENSE_B_FILE, dense_bias_mem);
        $readmemh(SIGMOID_FILE, sigmoid_lut_mem);
        $readmemh(TANH_FILE, tanh_lut_mem);
    end

    // The PS writes one 32-bit AXI word as two consecutive FP16 values.
    // Writes are performed only while the accelerator is idle.
    always @(posedge clk) begin
        if (rst) begin
            debug_input_word0 <= 32'd0;
        end else if (input_write_enable) begin
            if (input_write_strb[0])
                input_mem[input_write_word_addr][7:0] <=
                    input_write_data[7:0];
            if (input_write_strb[1])
                input_mem[input_write_word_addr][15:8] <=
                    input_write_data[15:8];
            if (input_write_strb[2])
                input_mem[input_write_word_addr][23:16] <=
                    input_write_data[23:16];
            if (input_write_strb[3])
                input_mem[input_write_word_addr][31:24] <=
                    input_write_data[31:24];

            if (input_write_word_addr == 10'd0) begin
                if (input_write_strb[0])
                    debug_input_word0[7:0] <= input_write_data[7:0];
                if (input_write_strb[1])
                    debug_input_word0[15:8] <= input_write_data[15:8];
                if (input_write_strb[2])
                    debug_input_word0[23:16] <= input_write_data[23:16];
                if (input_write_strb[3])
                    debug_input_word0[31:24] <= input_write_data[31:24];
            end
        end
    end

    reg [4:0] state;
    reg [3:0] mac_mode;
    reg [7:0] timestep;
    reg [5:0] matrix_tile;
    reg [6:0] matrix_column;
    reg [2:0] hidden_tile;
    reg [1:0] cell_stage;
    reg [2:0] pool_tile;
    reg [127:0] cell_fc_product;
    reg [127:0] new_cell_vector;
    reg [127:0] new_hidden_vector;
    reg [127:0] cell_src_a_pipeline;
    reg [127:0] cell_src_b_pipeline;
    reg [127:0] cell_acc_pipeline;
    reg cell_b_direct_pipeline;

    // Streaming matrix scheduler. req_* drives the synchronous weight ROMs;
    // issue_* is the matching one-cycle delayed metadata for the registered
    // ROM outputs.  Tiles are traversed fastest, columns slowest, which hides
    // the 36-cycle systolic latency behind independent output-row tiles.
    reg [1:0] stream_phase;
    reg [6:0] stream_slot;
    reg [6:0] stream_column;
    reg [11:0] stream_completed;
    reg        req_valid;
    reg        req_last;
    reg [5:0]  req_tile;
    reg [6:0]  req_column;
    reg        issue_valid;
    reg        issue_last;
    reg [5:0]  issue_tile;
    reg [6:0]  issue_column;

    // One synchronous ROM word contains all 16 PE x 9 MAC weights.  Packing
    // nine columns beforehand avoids nine asynchronous ROM reads and the
    // corresponding 2304-bit mux network.
    wire [7:0] lstm1_weight_addr = req_tile*3 + req_column[1:0];
    wire [6:0] lstm2_weight_addr = req_tile*3 + req_column[1:0];
    wire [3:0] dense_weight_addr = req_tile*3 + req_column[1:0];
    reg [2303:0] lstm1_weight_q;
    reg [2303:0] lstm2_weight_q;
    reg [2303:0] dense_weight_q;

    reg [15:0] input_cache [0:8];
    reg [3:0] input_load_channel;
    reg [10:0] input_read_addr;
    reg [31:0] input_read_word_q;
    reg        input_read_half_q;
    wire [15:0] input_read_q = input_read_half_q ?
                               input_read_word_q[31:16] :
                               input_read_word_q[15:0];

    // Three fixed activation groups remove variable indexing of the complete
    // hidden-state arrays from the systolic request path.
    reg [575:0] l1_vector_group [0:2];
    reg [575:0] l2_vector_group [0:2];
    reg [575:0] dense_vector_group [0:2];

    always @(posedge clk) begin
        lstm1_weight_q <= lstm1_weight_dot9[lstm1_weight_addr];
        lstm2_weight_q <= lstm2_weight_dot9[lstm2_weight_addr];
        dense_weight_q <= dense_weight_dot9[dense_weight_addr];
        input_read_word_q <= input_mem[input_read_addr[10:1]];
        input_read_half_q <= input_read_addr[0];

        if (rst) begin
            issue_valid <= 1'b0;
            issue_last <= 1'b0;
            issue_tile <= 6'd0;
            issue_column <= 7'd0;
        end else begin
            issue_valid <= req_valid;
            issue_last <= req_last;
            issue_tile <= req_tile;
            issue_column <= req_column;
        end
    end

    reg [15:0] gate1 [0:255];
    reg [15:0] gate2 [0:127];
    reg [15:0] hidden1 [0:63];
    reg [15:0] cell1 [0:63];
    reg [15:0] hidden2 [0:31];
    reg [15:0] cell2 [0:31];
    reg [15:0] hidden2_sum [0:31];
    reg [15:0] lstm_features [0:31];
    reg [15:0] logits [0:5];

    // First accepted request/result of each streaming matrix phase. These
    // registers are diagnostic only and are cleared for every new sample.
    reg [31:0] debug_l1_input;
    reg [15:0] debug_l1_accumulator;
    reg [15:0] debug_l1_result;
    reg [31:0] debug_l2_input;
    reg [15:0] debug_l2_accumulator;
    reg [15:0] debug_l2_result;
    reg [31:0] debug_dense_input;
    reg [15:0] debug_dense_accumulator;
    reg [15:0] debug_dense_result;
    reg        debug_l1_input_seen;
    reg        debug_l1_result_seen;
    reg        debug_l2_input_seen;
    reg        debug_l2_result_seen;
    reg        debug_dense_input_seen;
    reg        debug_dense_result_seen;

    // Read-only debug visibility for the AXI wrapper. Element 0 occupies the
    // least-significant 16 bits so software can read logits 0..5 in order.
    assign debug_logits = {
        logits[5], logits[4], logits[3],
        logits[2], logits[1], logits[0]
    };

    // Word order, least-significant word first:
    // 0 L1 weight/vector, 1 L1 accumulator/result, 2 hidden1/cell1,
    // 3 L2 weight/vector, 4 L2 accumulator/result, 5 hidden2/cell2,
    // 6 Dense weight/vector, 7 Dense accumulator/result,
    // 8 stream state/count, 9 gate1[0]/gate2[0].
    assign debug_pipeline = {
        gate2[0], gate1[0],
        8'd0, stream_phase, state, stream_completed[11:0], 5'd0,
        debug_dense_result, debug_dense_accumulator,
        debug_dense_input,
        cell2[0], hidden2[0],
        debug_l2_result, debug_l2_accumulator,
        debug_l2_input,
        cell1[0], hidden1[0],
        debug_l1_result, debug_l1_accumulator,
        debug_l1_input
    };

    function [15:0] lane16;
        input [127:0] value;
        input integer lane;
        begin
            lane16 = value[lane*16 +: 16];
        end
    endfunction

    function fp16_less;
        input [15:0] left;
        input [15:0] right;
        reg left_zero;
        reg right_zero;
        begin
            left_zero  = (left[14:0] == 15'd0);
            right_zero = (right[14:0] == 15'd0);
            if (left_zero && right_zero)
                fp16_less = 1'b0;
            else if (left[15] != right[15])
                fp16_less = left[15];
            else if (!left[15])
                fp16_less = (left[14:0] < right[14:0]);
            else
                fp16_less = (left[14:0] > right[14:0]);
        end
    endfunction

    function [7:0] activation_address;
        input [15:0] value;
        integer exponent;
        integer shift;
        integer mantissa;
        integer quotient;
        integer remainder;
        integer halfway;
        integer magnitude;
        integer address_integer;
        begin
            exponent = value[14:10];
            mantissa = 1024 + value[9:0];
            magnitude = 0;
            if (exponent == 0) begin
                magnitude = 0;
            end else if (exponent == 31) begin
                magnitude = value[15] ? 128 : 127;
            end else if (exponent >= 21) begin
                magnitude = mantissa << (exponent - 21);
            end else begin
                shift = 21 - exponent;
                if (shift >= 12) begin
                    magnitude = 0;
                end else begin
                    quotient = mantissa >> shift;
                    remainder = mantissa & ((1 << shift) - 1);
                    halfway = 1 << (shift - 1);
                    if ((remainder > halfway) ||
                        ((remainder == halfway) && (quotient & 1)))
                        quotient = quotient + 1;
                    magnitude = quotient;
                end
            end
            if (value[15]) begin
                if (magnitude > 128)
                    magnitude = 128;
                address_integer = 128 - magnitude;
            end else begin
                if (magnitude > 127)
                    magnitude = 127;
                address_integer = 128 + magnitude;
            end
            activation_address = address_integer[7:0];
        end
    endfunction

    integer index;
    integer lane;
    integer hidden_index;
    reg [15:0] input_scalar;
    // Registered tournament Argmax. The former six-way loop synthesized to
    // a 32-level combinational path from logits[] to prediction[] and missed
    // timing on VC707. One FP16 comparison per stage preserves the original
    // lowest-index-on-tie behavior while removing that critical path.
    (* keep = "true", dont_touch = "true" *)
    reg [15:0] argmax_pair_value0;
    (* keep = "true", dont_touch = "true" *)
    reg [15:0] argmax_pair_value1;
    (* keep = "true", dont_touch = "true" *)
    reg [15:0] argmax_pair_value2;
    (* keep = "true", dont_touch = "true" *)
    reg [2:0]  argmax_pair_index0;
    (* keep = "true", dont_touch = "true" *)
    reg [2:0]  argmax_pair_index1;
    (* keep = "true", dont_touch = "true" *)
    reg [2:0]  argmax_pair_index2;
    (* keep = "true", dont_touch = "true" *)
    reg [15:0] argmax_merge_value;
    (* keep = "true", dont_touch = "true" *)
    reg [2:0]  argmax_merge_index;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            prediction <= 3'd0;
            cycle_count <= 32'd0;
            timestep <= 8'd0;
            matrix_tile <= 6'd0;
            matrix_column <= 7'd0;
            hidden_tile <= 3'd0;
            cell_stage <= 2'd0;
            pool_tile <= 3'd0;
            cell_fc_product <= 128'd0;
            new_cell_vector <= 128'd0;
            new_hidden_vector <= 128'd0;
            cell_src_a_pipeline <= 128'd0;
            cell_src_b_pipeline <= 128'd0;
            cell_acc_pipeline <= 128'd0;
            cell_b_direct_pipeline <= 1'b0;
            argmax_pair_value0 <= FP16_ZERO;
            argmax_pair_value1 <= FP16_ZERO;
            argmax_pair_value2 <= FP16_ZERO;
            argmax_pair_index0 <= 3'd0;
            argmax_pair_index1 <= 3'd2;
            argmax_pair_index2 <= 3'd4;
            argmax_merge_value <= FP16_ZERO;
            argmax_merge_index <= 3'd0;
            mac_start <= 1'b0;
            mac_a <= 128'd0;
            mac_b <= 128'd0;
            mac_acc <= 128'd0;
            mac_tile_mode <= 1'b0;
            mac_tile_weights <= 256'd0;
            mac_tile_vector <= 64'd0;
            mac_tile_accumulator <= 64'd0;
            mac_stream_valid <= 1'b0;
            mac_stream_tag <= 6'd0;
            mac_stream_weights <= 2304'd0;
            mac_stream_vector <= 576'd0;
            mac_stream_accumulator <= 64'd0;
            stream_phase <= STREAM_L1;
            stream_slot <= 7'd0;
            stream_column <= 7'd0;
            stream_completed <= 12'd0;
            req_valid <= 1'b0;
            req_last <= 1'b0;
            req_tile <= 6'd0;
            req_column <= 7'd0;
            debug_l1_input <= 32'd0;
            debug_l1_accumulator <= FP16_ZERO;
            debug_l1_result <= FP16_ZERO;
            debug_l2_input <= 32'd0;
            debug_l2_accumulator <= FP16_ZERO;
            debug_l2_result <= FP16_ZERO;
            debug_dense_input <= 32'd0;
            debug_dense_accumulator <= FP16_ZERO;
            debug_dense_result <= FP16_ZERO;
            debug_l1_input_seen <= 1'b0;
            debug_l1_result_seen <= 1'b0;
            debug_l2_input_seen <= 1'b0;
            debug_l2_result_seen <= 1'b0;
            debug_dense_input_seen <= 1'b0;
            debug_dense_result_seen <= 1'b0;
            input_load_channel <= 4'd0;
            input_read_addr <= 11'd0;
            for (index = 0; index < 9; index = index + 1)
                input_cache[index] <= FP16_ZERO;
            for (index = 0; index < 64; index = index + 1) begin
                hidden1[index] <= FP16_ZERO;
                cell1[index] <= FP16_ZERO;
            end
            for (index = 0; index < 32; index = index + 1) begin
                hidden2[index] <= FP16_ZERO;
                cell2[index] <= FP16_ZERO;
                hidden2_sum[index] <= FP16_ZERO;
                lstm_features[index] <= FP16_ZERO;
            end
            for (index = 0; index < 6; index = index + 1)
                logits[index] <= FP16_ZERO;
        end else begin
            done <= 1'b0;
            mac_start <= 1'b0;
            mac_stream_valid <= 1'b0;
            if (busy)
                cycle_count <= cycle_count + 1'b1;

            // Results can return while later independent tiles are still being
            // issued. Store them immediately by tag.
            if (mac_stream_out_valid) begin
                stream_completed <= stream_completed + 1'b1;
                case (stream_phase)
                    STREAM_L1: begin
                        if (!debug_l1_result_seen) begin
                            debug_l1_result <= mac_stream_result[15:0];
                            debug_l1_result_seen <= 1'b1;
                        end
                        for (lane = 0; lane < 4; lane = lane + 1)
                            gate1[mac_stream_out_tag*4 + lane] <=
                                mac_stream_result[lane*16 +: 16];
                    end
                    STREAM_L2: begin
                        if (!debug_l2_result_seen) begin
                            debug_l2_result <= mac_stream_result[15:0];
                            debug_l2_result_seen <= 1'b1;
                        end
                        for (lane = 0; lane < 4; lane = lane + 1)
                            gate2[mac_stream_out_tag*4 + lane] <=
                                mac_stream_result[lane*16 +: 16];
                    end
                    STREAM_DENSE: begin
                        if (!debug_dense_result_seen) begin
                            debug_dense_result <= mac_stream_result[15:0];
                            debug_dense_result_seen <= 1'b1;
                        end
                        for (lane = 0; lane < 4; lane = lane + 1)
                            if (mac_stream_out_tag*4 + lane < 6)
                                logits[mac_stream_out_tag*4 + lane] <=
                                    mac_stream_result[lane*16 +: 16];
                    end
                    default: begin end
                endcase
            end

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        cycle_count <= 32'd0;
                        debug_l1_input <= 32'd0;
                        debug_l1_accumulator <= FP16_ZERO;
                        debug_l1_result <= FP16_ZERO;
                        debug_l2_input <= 32'd0;
                        debug_l2_accumulator <= FP16_ZERO;
                        debug_l2_result <= FP16_ZERO;
                        debug_dense_input <= 32'd0;
                        debug_dense_accumulator <= FP16_ZERO;
                        debug_dense_result <= FP16_ZERO;
                        debug_l1_input_seen <= 1'b0;
                        debug_l1_result_seen <= 1'b0;
                        debug_l2_input_seen <= 1'b0;
                        debug_l2_result_seen <= 1'b0;
                        debug_dense_input_seen <= 1'b0;
                        debug_dense_result_seen <= 1'b0;
                        timestep <= 8'd0;
                        matrix_tile <= 6'd0;
                        req_valid <= 1'b0;
                        req_last <= 1'b0;
                        for (index = 0; index < 64; index = index + 1) begin
                            hidden1[index] <= FP16_ZERO;
                            cell1[index] <= FP16_ZERO;
                        end
                        for (index = 0; index < 32; index = index + 1) begin
                            hidden2[index] <= FP16_ZERO;
                            cell2[index] <= FP16_ZERO;
                            hidden2_sum[index] <= FP16_ZERO;
                        end
                        state <= S_INPUT_LOAD;
                    end
                end

                S_INPUT_LOAD: begin
                    input_load_channel <= 4'd0;
                    input_read_addr <= timestep*11'd9;
                    state <= S_INPUT_WAIT;
                end

                S_INPUT_WAIT:
                    state <= S_INPUT_CAPTURE;

                S_INPUT_CAPTURE: begin
                    input_cache[input_load_channel] <= input_read_q;
                    if (input_load_channel == 8) begin
                        state <= S_L1_INIT;
                    end else begin
                        input_load_channel <= input_load_channel + 1'b1;
                        input_read_addr <= timestep*11'd9 +
                                           input_load_channel + 1'b1;
                        state <= S_INPUT_WAIT;
                    end
                end

                S_L1_INIT: begin
                    l1_vector_group[0] <= 576'd0;
                    l1_vector_group[1] <= 576'd0;
                    l1_vector_group[2] <= 576'd0;
                    for (lane = 0; lane < 9; lane = lane + 1)
                        l1_vector_group[0][lane*16 +: 16] <= input_cache[lane];
                    for (lane = 0; lane < 27; lane = lane + 1)
                        l1_vector_group[0][(lane+9)*16 +: 16] <= hidden1[lane];
                    for (lane = 0; lane < 36; lane = lane + 1)
                        l1_vector_group[1][lane*16 +: 16] <= hidden1[lane+27];
                    l1_vector_group[2][15:0] <= hidden1[63];
                    stream_phase <= STREAM_L1;
                    stream_completed <= 12'd0;
                    stream_column <= 7'd0;
                    stream_slot <= 7'd1;
                    req_valid <= 1'b1;
                    req_last <= 1'b0;
                    req_tile <= 6'd0;
                    req_column <= 7'd0;
                    state <= S_L1_PREP;
                end

                S_L1_PREP: begin
                    // The service is idle throughout matrix streaming, so ready
                    // is expected to remain asserted. One BRAM result is issued
                    // per cycle after the initial one-cycle prefetch.
                    if (issue_valid && mac_stream_ready) begin
                        mac_stream_valid <= 1'b1;
                        mac_stream_tag <= issue_tile;
                        mac_stream_vector <= l1_vector_group[issue_column[1:0]];
                        mac_stream_weights <= lstm1_weight_q;
                        if (!debug_l1_input_seen) begin
                            debug_l1_input <= {
                                lstm1_weight_q[15:0],
                                l1_vector_group[issue_column[1:0]][15:0]
                            };
                            if (issue_column == 0)
                                debug_l1_accumulator <=
                                    lstm1_bias_mem[issue_tile >> 1][
                                        (issue_tile[0] ? 64 : 0) +: 16
                                    ];
                            else
                                debug_l1_accumulator <= gate1[issue_tile*4];
                            debug_l1_input_seen <= 1'b1;
                        end
                        for (lane = 0; lane < 4; lane = lane + 1) begin
                            if (issue_column == 0)
                                mac_stream_accumulator[lane*16 +: 16] <=
                                    lstm1_bias_mem[issue_tile >> 1][
                                        ((issue_tile[0] ? 4 : 0)+lane)*16 +: 16
                                    ];
                            else
                                mac_stream_accumulator[lane*16 +: 16] <=
                                    gate1[issue_tile*4 + lane];

                        end
                        if (issue_last) begin
                            req_valid <= 1'b0;
                            state <= S_L1_DRAIN;
                        end
                    end

                    if (!issue_last) begin
                        if (stream_slot < 64) begin
                            req_valid <= 1'b1;
                            req_tile <= stream_slot[5:0];
                            req_column <= stream_column;
                            req_last <= (stream_column == 2 && stream_slot == 63);
                        end else begin
                            req_valid <= 1'b0;
                            req_last <= 1'b0;
                        end
                        if (stream_slot == STREAM_REUSE_GAP-1) begin
                            stream_slot <= 7'd0;
                            stream_column <= stream_column + 1'b1;
                        end else begin
                            stream_slot <= stream_slot + 1'b1;
                        end
                    end
                end

                S_L1_CELL_PREP: begin
                    cell_src_a_pipeline <= 128'd0;
                    cell_src_b_pipeline <= 128'd0;
                    cell_acc_pipeline <= (cell_stage == 1) ?
                                         cell_fc_product : 128'd0;
                    cell_b_direct_pipeline <= (cell_stage == 0);
                    for (lane = 0; lane < 8; lane = lane + 1) begin
                        hidden_index = hidden_tile*8 + lane;
                        if (cell_stage == 0) begin
                            cell_src_a_pipeline[lane*16 +: 16] <=
                                gate1[64 + hidden_index];
                            cell_src_b_pipeline[lane*16 +: 16] <=
                                cell1[hidden_index];
                        end else if (cell_stage == 1) begin
                            cell_src_a_pipeline[lane*16 +: 16] <=
                                gate1[hidden_index];
                            cell_src_b_pipeline[lane*16 +: 16] <=
                                gate1[128 + hidden_index];
                        end else begin
                            cell_src_a_pipeline[lane*16 +: 16] <=
                                gate1[192 + hidden_index];
                            cell_src_b_pipeline[lane*16 +: 16] <=
                                lane16(new_cell_vector, lane);
                        end
                    end
                    if (cell_stage == 0)
                        mac_mode <= M_L1_FC;
                    else if (cell_stage == 1)
                        mac_mode <= M_L1_IG;
                    else
                        mac_mode <= M_L1_H;
                    state <= S_L1_CELL_APPLY;
                end

                S_L1_CELL_APPLY: begin
                    mac_tile_mode <= 1'b0;
                    mac_a <= 128'd0;
                    mac_b <= 128'd0;
                    mac_acc <= cell_acc_pipeline;
                    for (lane = 0; lane < 8; lane = lane + 1) begin
                        mac_a[lane*16 +: 16] <= sigmoid_lut_mem[
                            activation_address(
                                lane16(cell_src_a_pipeline, lane)
                            )
                        ];
                        if (cell_b_direct_pipeline)
                            mac_b[lane*16 +: 16] <=
                                lane16(cell_src_b_pipeline, lane);
                        else
                            mac_b[lane*16 +: 16] <= tanh_lut_mem[
                                activation_address(
                                    lane16(cell_src_b_pipeline, lane)
                                )
                            ];
                    end
                    state <= S_MAC_START;
                end

                S_L2_INIT: begin
                    l2_vector_group[0] <= 576'd0;
                    l2_vector_group[1] <= 576'd0;
                    l2_vector_group[2] <= 576'd0;
                    for (lane = 0; lane < 36; lane = lane + 1)
                        l2_vector_group[0][lane*16 +: 16] <= hidden1[lane];
                    for (lane = 0; lane < 28; lane = lane + 1)
                        l2_vector_group[1][lane*16 +: 16] <= hidden1[lane+36];
                    for (lane = 0; lane < 8; lane = lane + 1)
                        l2_vector_group[1][(lane+28)*16 +: 16] <= hidden2[lane];
                    for (lane = 0; lane < 24; lane = lane + 1)
                        l2_vector_group[2][lane*16 +: 16] <= hidden2[lane+8];
                    stream_phase <= STREAM_L2;
                    stream_completed <= 12'd0;
                    stream_column <= 7'd0;
                    stream_slot <= 7'd1;
                    req_valid <= 1'b1;
                    req_last <= 1'b0;
                    req_tile <= 6'd0;
                    req_column <= 7'd0;
                    state <= S_L2_PREP;
                end

                S_L2_PREP: begin
                    if (issue_valid && mac_stream_ready) begin
                        mac_stream_valid <= 1'b1;
                        mac_stream_tag <= issue_tile;
                        mac_stream_vector <= l2_vector_group[issue_column[1:0]];
                        mac_stream_weights <= lstm2_weight_q;
                        if (!debug_l2_input_seen) begin
                            debug_l2_input <= {
                                lstm2_weight_q[15:0],
                                l2_vector_group[issue_column[1:0]][15:0]
                            };
                            if (issue_column == 0)
                                debug_l2_accumulator <=
                                    lstm2_bias_mem[issue_tile >> 1][
                                        (issue_tile[0] ? 64 : 0) +: 16
                                    ];
                            else
                                debug_l2_accumulator <= gate2[issue_tile*4];
                            debug_l2_input_seen <= 1'b1;
                        end
                        for (lane = 0; lane < 4; lane = lane + 1) begin
                            if (issue_column == 0)
                                mac_stream_accumulator[lane*16 +: 16] <=
                                    lstm2_bias_mem[issue_tile >> 1][
                                        ((issue_tile[0] ? 4 : 0)+lane)*16 +: 16
                                    ];
                            else
                                mac_stream_accumulator[lane*16 +: 16] <=
                                    gate2[issue_tile*4 + lane];

                        end
                        if (issue_last) begin
                            req_valid <= 1'b0;
                            state <= S_L2_DRAIN;
                        end
                    end

                    if (!issue_last) begin
                        if (stream_slot < 32) begin
                            req_valid <= 1'b1;
                            req_tile <= stream_slot[5:0];
                            req_column <= stream_column;
                            req_last <= (stream_column == 2 && stream_slot == 31);
                        end else begin
                            req_valid <= 1'b0;
                            req_last <= 1'b0;
                        end

                        if (stream_slot == STREAM_REUSE_GAP-1) begin
                            stream_slot <= 7'd0;
                            stream_column <= stream_column + 1'b1;
                        end else begin
                            stream_slot <= stream_slot + 1'b1;
                        end
                    end
                end

                S_L2_CELL_PREP: begin
                    cell_src_a_pipeline <= 128'd0;
                    cell_src_b_pipeline <= 128'd0;
                    cell_acc_pipeline <= (cell_stage == 1) ?
                                         cell_fc_product : 128'd0;
                    cell_b_direct_pipeline <= (cell_stage == 0);
                    for (lane = 0; lane < 8; lane = lane + 1) begin
                        hidden_index = hidden_tile*8 + lane;
                        if (cell_stage == 0) begin
                            cell_src_a_pipeline[lane*16 +: 16] <=
                                gate2[32 + hidden_index];
                            cell_src_b_pipeline[lane*16 +: 16] <=
                                cell2[hidden_index];
                        end else if (cell_stage == 1) begin
                            cell_src_a_pipeline[lane*16 +: 16] <=
                                gate2[hidden_index];
                            cell_src_b_pipeline[lane*16 +: 16] <=
                                gate2[64 + hidden_index];
                        end else begin
                            cell_src_a_pipeline[lane*16 +: 16] <=
                                gate2[96 + hidden_index];
                            cell_src_b_pipeline[lane*16 +: 16] <=
                                lane16(new_cell_vector, lane);
                        end
                    end
                    if (cell_stage == 0)
                        mac_mode <= M_L2_FC;
                    else if (cell_stage == 1)
                        mac_mode <= M_L2_IG;
                    else
                        mac_mode <= M_L2_H;
                    state <= S_L2_CELL_APPLY;
                end

                S_L2_CELL_APPLY: begin
                    mac_tile_mode <= 1'b0;
                    mac_a <= 128'd0;
                    mac_b <= 128'd0;
                    mac_acc <= cell_acc_pipeline;
                    for (lane = 0; lane < 8; lane = lane + 1) begin
                        mac_a[lane*16 +: 16] <= sigmoid_lut_mem[
                            activation_address(
                                lane16(cell_src_a_pipeline, lane)
                            )
                        ];
                        if (cell_b_direct_pipeline)
                            mac_b[lane*16 +: 16] <=
                                lane16(cell_src_b_pipeline, lane);
                        else
                            mac_b[lane*16 +: 16] <= tanh_lut_mem[
                                activation_address(
                                    lane16(cell_src_b_pipeline, lane)
                                )
                            ];
                    end
                    state <= S_MAC_START;
                end

                S_L2_SUM_PREP: begin
                    mac_tile_mode <= 1'b0;
                    mac_a <= new_hidden_vector;
                    for (lane = 0; lane < 8; lane = lane + 1) begin
                        mac_b[lane*16 +: 16] <= FP16_ONE;
                        mac_acc[lane*16 +: 16] <= hidden2_sum[
                            hidden_tile*8 + lane
                        ];
                    end
                    mac_mode <= M_L2_SUM;
                    state <= S_MAC_START;
                end

                S_POOL_INIT: begin
                    pool_tile <= 3'd0;
                    if (POOL_MEAN != 0)
                        state <= S_POOL_PREP;
                    else begin
                        for (index = 0; index < 32; index = index + 1)
                            lstm_features[index] <= hidden2[index];
                        state <= S_DENSE_INIT;
                    end
                end

                S_POOL_PREP: begin
                    mac_tile_mode <= 1'b0;
                    for (lane = 0; lane < 8; lane = lane + 1) begin
                        mac_a[lane*16 +: 16] <= hidden2_sum[
                            pool_tile*8 + lane
                        ];
                        mac_b[lane*16 +: 16] <= FP16_INV128;
                        mac_acc[lane*16 +: 16] <= FP16_ZERO;
                    end
                    mac_mode <= M_POOL;
                    state <= S_MAC_START;
                end

                S_DENSE_INIT: begin
                    dense_vector_group[0] <= 576'd0;
                    dense_vector_group[1] <= 576'd0;
                    dense_vector_group[2] <= 576'd0;
                    for (lane = 0; lane < 32; lane = lane + 1)
                        dense_vector_group[0][lane*16 +: 16] <= lstm_features[lane];
                    for (lane = 0; lane < 4; lane = lane + 1)
                        dense_vector_group[0][(lane+32)*16 +: 16] <=
                            stats_features[lane*16 +: 16];
                    for (lane = 0; lane < 36; lane = lane + 1)
                        dense_vector_group[1][lane*16 +: 16] <=
                            stats_features[(lane+4)*16 +: 16];
                    for (lane = 0; lane < 8; lane = lane + 1)
                        dense_vector_group[2][lane*16 +: 16] <=
                            stats_features[(lane+40)*16 +: 16];
                    stream_phase <= STREAM_DENSE;
                    stream_completed <= 12'd0;
                    stream_column <= 7'd0;
                    stream_slot <= 7'd1;
                    req_valid <= 1'b1;
                    req_last <= 1'b0;
                    req_tile <= 6'd0;
                    req_column <= 7'd0;
                    state <= S_DENSE_PREP;
                end

                S_DENSE_PREP: begin
                    if (issue_valid && mac_stream_ready) begin
                        mac_stream_valid <= 1'b1;
                        mac_stream_tag <= issue_tile;
                        mac_stream_vector <= dense_vector_group[issue_column[1:0]];
                        mac_stream_weights <= dense_weight_q;
                        if (!debug_dense_input_seen) begin
                            debug_dense_input <= {
                                dense_weight_q[15:0],
                                dense_vector_group[issue_column[1:0]][15:0]
                            };
                            if (issue_column == 0)
                                debug_dense_accumulator <=
                                    dense_bias_mem[0][issue_tile*64 +: 16];
                            else
                                debug_dense_accumulator <=
                                    logits[issue_tile*4];
                            debug_dense_input_seen <= 1'b1;
                        end
                        for (lane = 0; lane < 4; lane = lane + 1) begin
                            if (issue_column == 0) begin
                                if (issue_tile*4 + lane < 6)
                                    mac_stream_accumulator[lane*16 +: 16] <=
                                        dense_bias_mem[0][
                                            (issue_tile*4+lane)*16 +: 16
                                        ];
                                else
                                    mac_stream_accumulator[lane*16 +: 16] <=
                                        FP16_ZERO;
                            end else if (issue_tile*4 + lane < 6)
                                mac_stream_accumulator[lane*16 +: 16] <=
                                    logits[issue_tile*4 + lane];
                            else
                                mac_stream_accumulator[lane*16 +: 16] <=
                                    FP16_ZERO;

                        end
                        if (issue_last) begin
                            req_valid <= 1'b0;
                            state <= S_DENSE_DRAIN;
                        end
                    end

                    if (!issue_last) begin
                        if (stream_slot < 2) begin
                            req_valid <= 1'b1;
                            req_tile <= stream_slot[5:0];
                            req_column <= stream_column;
                            req_last <= (stream_column == 2 && stream_slot == 1);
                        end else begin
                            req_valid <= 1'b0;
                            req_last <= 1'b0;
                        end

                        if (stream_slot == STREAM_REUSE_GAP-1) begin
                            stream_slot <= 7'd0;
                            stream_column <= stream_column + 1'b1;
                        end else begin
                            stream_slot <= stream_slot + 1'b1;
                        end
                    end
                end

                S_WEIGHT_WAIT: begin
                    // Matrix operations use the streaming path.
                    req_valid <= 1'b0;
                    state <= S_IDLE;
                end

                S_MAC_START: begin
                    mac_start <= 1'b1;
                    state <= S_MAC_WAIT;
                end

                S_MAC_WAIT: begin
                    if (mac_done) begin
                        case (mac_mode)
                            M_L1_MAT: begin
                                for (lane = 0; lane < 4; lane = lane + 1)
                                    gate1[matrix_tile*4 + lane] <=
                                        mac_tile_result[lane*16 +: 16];
                                if (matrix_column == 19) begin
                                    if (matrix_tile == 63) begin
                                        hidden_tile <= 3'd0;
                                        cell_stage <= 2'd0;
                                        state <= S_L1_CELL_PREP;
                                    end else begin
                                        matrix_tile <= matrix_tile + 1'b1;
                                        state <= S_L1_INIT;
                                    end
                                end else begin
                                    matrix_column <= matrix_column + 1'b1;
                                    state <= S_L1_PREP;
                                end
                            end

                            M_L1_FC: begin
                                cell_fc_product <= mac_result;
                                cell_stage <= 2'd1;
                                state <= S_L1_CELL_PREP;
                            end

                            M_L1_IG: begin
                                new_cell_vector <= mac_result;
                                cell_stage <= 2'd2;
                                state <= S_L1_CELL_PREP;
                            end

                            M_L1_H: begin
                                for (lane = 0; lane < 8;
                                     lane = lane + 1) begin
                                    cell1[hidden_tile*8 + lane] <=
                                        new_cell_vector[lane*16 +: 16];
                                    hidden1[hidden_tile*8 + lane] <=
                                        mac_result[lane*16 +: 16];
                                end
                                if (hidden_tile == 7) begin
                                    matrix_tile <= 6'd0;
                                    state <= S_L2_INIT;
                                end else begin
                                    hidden_tile <= hidden_tile + 1'b1;
                                    cell_stage <= 2'd0;
                                    state <= S_L1_CELL_PREP;
                                end
                            end

                            M_L2_MAT: begin
                                for (lane = 0; lane < 4; lane = lane + 1)
                                    gate2[matrix_tile*4 + lane] <=
                                        mac_tile_result[lane*16 +: 16];
                                if (matrix_column == 23) begin
                                    if (matrix_tile == 31) begin
                                        hidden_tile <= 3'd0;
                                        cell_stage <= 2'd0;
                                        state <= S_L2_CELL_PREP;
                                    end else begin
                                        matrix_tile <= matrix_tile + 1'b1;
                                        state <= S_L2_INIT;
                                    end
                                end else begin
                                    matrix_column <= matrix_column + 1'b1;
                                    state <= S_L2_PREP;
                                end
                            end

                            M_L2_FC: begin
                                cell_fc_product <= mac_result;
                                cell_stage <= 2'd1;
                                state <= S_L2_CELL_PREP;
                            end

                            M_L2_IG: begin
                                new_cell_vector <= mac_result;
                                cell_stage <= 2'd2;
                                state <= S_L2_CELL_PREP;
                            end

                            M_L2_H: begin
                                new_hidden_vector <= mac_result;
                                for (lane = 0; lane < 8;
                                     lane = lane + 1)
                                    cell2[hidden_tile*8 + lane] <=
                                        new_cell_vector[lane*16 +: 16];
                                if (POOL_MEAN != 0) begin
                                    state <= S_L2_SUM_PREP;
                                end else begin
                                    for (lane = 0; lane < 8;
                                         lane = lane + 1)
                                        hidden2[hidden_tile*8 + lane] <=
                                            mac_result[lane*16 +: 16];
                                    if (hidden_tile == 3) begin
                                        if (timestep == 127)
                                            state <= S_POOL_INIT;
                                        else begin
                                            timestep <= timestep + 1'b1;
                                            matrix_tile <= 6'd0;
                                            state <= S_INPUT_LOAD;
                                        end
                                    end else begin
                                        hidden_tile <= hidden_tile + 1'b1;
                                        cell_stage <= 2'd0;
                                        state <= S_L2_CELL_PREP;
                                    end
                                end
                            end

                            M_L2_SUM: begin
                                for (lane = 0; lane < 8;
                                     lane = lane + 1) begin
                                    hidden2[hidden_tile*8 + lane] <=
                                        new_hidden_vector[lane*16 +: 16];
                                    hidden2_sum[hidden_tile*8 + lane] <=
                                        mac_result[lane*16 +: 16];
                                end
                                if (hidden_tile == 3) begin
                                    if (timestep == 127)
                                        state <= S_POOL_INIT;
                                    else begin
                                        timestep <= timestep + 1'b1;
                                        matrix_tile <= 6'd0;
                                        state <= S_INPUT_LOAD;
                                    end
                                end else begin
                                    hidden_tile <= hidden_tile + 1'b1;
                                    cell_stage <= 2'd0;
                                    state <= S_L2_CELL_PREP;
                                end
                            end

                            M_POOL: begin
                                for (lane = 0; lane < 8;
                                     lane = lane + 1)
                                    lstm_features[pool_tile*8 + lane] <=
                                        mac_result[lane*16 +: 16];
                                if (pool_tile == 3)
                                    state <= S_DENSE_INIT;
                                else begin
                                    pool_tile <= pool_tile + 1'b1;
                                    state <= S_POOL_PREP;
                                end
                            end

                            M_DENSE: begin
                                for (lane = 0; lane < 4; lane = lane + 1)
                                    if (matrix_tile*4 + lane < 6)
                                        logits[matrix_tile*4 + lane] <=
                                            mac_tile_result[lane*16 +: 16];
                                if (matrix_column == 19) begin
                                    if (matrix_tile == 1)
                                        state <= S_ARGMAX;
                                    else begin
                                        matrix_tile <= matrix_tile + 1'b1;
                                        matrix_column <= 7'd0;
                                        state <= S_DENSE_PREP;
                                    end
                                end else begin
                                    matrix_column <= matrix_column + 1'b1;
                                    state <= S_DENSE_PREP;
                                end
                            end

                            default: state <= S_IDLE;
                        endcase
                    end
                end

                S_L1_DRAIN: begin
                    req_valid <= 1'b0;
                    if (mac_stream_out_valid &&
                        stream_completed == L1_STREAM_TOTAL-1) begin
                        hidden_tile <= 3'd0;
                        cell_stage <= 2'd0;
                        state <= S_L1_CELL_PREP;
                    end
                end

                S_L2_DRAIN: begin
                    req_valid <= 1'b0;
                    if (mac_stream_out_valid &&
                        stream_completed == L2_STREAM_TOTAL-1) begin
                        hidden_tile <= 3'd0;
                        cell_stage <= 2'd0;
                        state <= S_L2_CELL_PREP;
                    end
                end

                S_DENSE_DRAIN: begin
                    req_valid <= 1'b0;
                    if (mac_stream_out_valid &&
                        stream_completed == DENSE_STREAM_TOTAL-1) begin
                        state <= S_ARGMAX;
                    end
                end

                S_ARGMAX: begin
                    if (fp16_less(logits[0], logits[1])) begin
                        argmax_pair_value0 <= logits[1];
                        argmax_pair_index0 <= 3'd1;
                    end else begin
                        argmax_pair_value0 <= logits[0];
                        argmax_pair_index0 <= 3'd0;
                    end
                    if (fp16_less(logits[2], logits[3])) begin
                        argmax_pair_value1 <= logits[3];
                        argmax_pair_index1 <= 3'd3;
                    end else begin
                        argmax_pair_value1 <= logits[2];
                        argmax_pair_index1 <= 3'd2;
                    end
                    if (fp16_less(logits[4], logits[5])) begin
                        argmax_pair_value2 <= logits[5];
                        argmax_pair_index2 <= 3'd5;
                    end else begin
                        argmax_pair_value2 <= logits[4];
                        argmax_pair_index2 <= 3'd4;
                    end
                    state <= S_ARGMAX_MERGE;
                end

                S_ARGMAX_MERGE: begin
                    if (fp16_less(argmax_pair_value0,
                                  argmax_pair_value1)) begin
                        argmax_merge_value <= argmax_pair_value1;
                        argmax_merge_index <= argmax_pair_index1;
                    end else begin
                        argmax_merge_value <= argmax_pair_value0;
                        argmax_merge_index <= argmax_pair_index0;
                    end
                    state <= S_ARGMAX_FINAL;
                end

                S_ARGMAX_FINAL: begin
                    if (fp16_less(argmax_merge_value,
                                  argmax_pair_value2))
                        prediction <= argmax_pair_index2;
                    else
                        prediction <= argmax_merge_index;
                    state <= S_DONE;
                end

                S_DONE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
