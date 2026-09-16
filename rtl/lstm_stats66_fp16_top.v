`timescale 1ns/1ps

// On-chip top level for the trained UCI HAR model.
//
// External interface intentionally stays small to avoid FPGA I/O exhaustion.
// All inputs, weights, biases and activation tables are initialized into
// on-chip memories from the exported .mem files.
module lstm_stats66_fp16_top #(
    parameter MEM_DIR =
        ".",
    parameter integer POOL_MEAN = 0
) (
    input             clk,
    input             rst,
    input             start,
    input             input_write_enable,
    input      [9:0]  input_write_word_addr,
    input      [31:0] input_write_data,
    input      [3:0]  input_write_strb,
    output reg        busy,
    output reg        done,
    output reg [2:0]  prediction,
    output reg [31:0] cycle_count,
    output wire [31:0] debug_input_word0,
    output wire [95:0] debug_logits,
    output wire [383:0] debug_stages
);

    localparam S_IDLE       = 3'd0;
    localparam S_STATS_KICK = 3'd1;
    localparam S_STATS_WAIT = 3'd2;
    localparam S_LSTM_KICK  = 3'd3;
    localparam S_LSTM_WAIT  = 3'd4;
    localparam S_DONE       = 3'd5;

    reg [2:0] state;
    reg stats_start;
    reg lstm_start;

    wire stats_busy;
    wire stats_done;
    wire [767:0] stats_features;
    wire [31:0] stats_cycles;

    wire lstm_busy;
    wire lstm_done;
    wire [2:0] lstm_prediction;
    wire [31:0] lstm_cycles;
    wire [319:0] lstm_debug_pipeline;
    reg [31:0] debug_stats_special;
    integer debug_stats_index;

    wire stats_mac_start;
    wire [127:0] stats_mac_a;
    wire [127:0] stats_mac_b;
    wire [127:0] stats_mac_acc;
    wire lstm_mac_start;
    wire [127:0] lstm_mac_a;
    wire [127:0] lstm_mac_b;
    wire [127:0] lstm_mac_acc;
    wire lstm_mac_tile_mode;
    wire [255:0] lstm_mac_tile_weights;
    wire [63:0] lstm_mac_tile_vector;
    wire [63:0] lstm_mac_tile_accumulator;
    wire [127:0] shared_mac_result;
    wire [63:0] shared_mac_tile_result;
    wire shared_mac_busy;
    wire shared_mac_done;

    wire        lstm_stream_valid;
    wire [5:0]  lstm_stream_tag;
    wire [2303:0] lstm_stream_weights;
    wire [575:0] lstm_stream_vector;
    wire [63:0] lstm_stream_accumulator;
    wire        lstm_stream_ready;
    wire        lstm_stream_out_valid;
    wire [5:0]  lstm_stream_out_tag;
    wire [63:0] lstm_stream_result;
    wire stats_owns_mac = (state == S_STATS_KICK) ||
                          (state == S_STATS_WAIT);

    // Debug words 0..1 describe the statistics branch; words 2..11 are
    // supplied by the recurrent/dense controller. A set MSB in word 1 means
    // at least one Stats66 feature was NaN/Inf; bits 22:16 hold its index.
    assign debug_stages = {
        lstm_debug_pipeline,
        debug_stats_special,
        stats_features[31:0]
    };

    // Exactly one physical 4x4 systolic array is shared by the statistics
    // branch and the recurrent network. These phases never overlap.
    fp16_systolic_service_4x4 u_shared_systolic_mac (
        .clk(clk),
        .rst(rst),
        .start(stats_owns_mac ? stats_mac_start : lstm_mac_start),
        // Legacy requests are now only eight-lane element-wise MACs.
        .tile_mode(1'b0),
        .a_vec(stats_owns_mac ? stats_mac_a : lstm_mac_a),
        .b_vec(stats_owns_mac ? stats_mac_b : lstm_mac_b),
        .acc_vec(stats_owns_mac ? stats_mac_acc : lstm_mac_acc),
        .tile_weights(256'd0),
        .tile_vector(64'd0),
        .tile_accumulator(64'd0),
        .result_vec(shared_mac_result),
        .tile_result(shared_mac_tile_result),
        .busy(shared_mac_busy),
        .done(shared_mac_done),

        .stream_valid(!stats_owns_mac && lstm_stream_valid),
        .stream_tag(lstm_stream_tag),
        .stream_weights(lstm_stream_weights),
        .stream_vector(lstm_stream_vector),
        .stream_accumulator(lstm_stream_accumulator),
        .stream_ready(lstm_stream_ready),
        .stream_out_valid(lstm_stream_out_valid),
        .stream_out_tag(lstm_stream_out_tag),
        .stream_result(lstm_stream_result)
    );

    temporal_stats66_fp16 #(
        .INPUT_FILE({MEM_DIR, "/input_sample_fp16.mem"}),
        .INPUT_PACKED_FILE({MEM_DIR, "/input_sample_packed32.mem"}),
        .STATS_W_FILE({MEM_DIR, "/stats_weight_packed_fp16.mem"}),
        .STATS_B_FILE({MEM_DIR, "/stats_bias_packed_fp16.mem"}),
        .TANH_LUT_FILE({MEM_DIR, "/tanh_lut_fp16.mem"})
    ) u_statistics (
        .clk(clk),
        .rst(rst),
        .start(stats_start),
        .input_write_enable(input_write_enable),
        .input_write_word_addr(input_write_word_addr),
        .input_write_data(input_write_data),
        .input_write_strb(input_write_strb),
        .busy(stats_busy),
        .done(stats_done),
        .stats_features(stats_features),
        .cycle_count(stats_cycles),
        .mac_start(stats_mac_start),
        .mac_a(stats_mac_a),
        .mac_b(stats_mac_b),
        .mac_acc(stats_mac_acc),
        .mac_result(shared_mac_result),
        .mac_busy(stats_owns_mac && shared_mac_busy),
        .mac_done(stats_owns_mac && shared_mac_done)
    );

    lstm64_32_dense_fp16 #(
        .INPUT_FILE({MEM_DIR, "/input_sample_fp16.mem"}),
        .INPUT_PACKED_FILE({MEM_DIR, "/input_sample_packed32.mem"}),
        .LSTM1_W_BANK0_FILE({MEM_DIR, "/lstm1_weight_bank0_fp16.mem"}),
        .LSTM1_W_BANK1_FILE({MEM_DIR, "/lstm1_weight_bank1_fp16.mem"}),
        .LSTM1_W_BANK2_FILE({MEM_DIR, "/lstm1_weight_bank2_fp16.mem"}),
        .LSTM1_W_BANK3_FILE({MEM_DIR, "/lstm1_weight_bank3_fp16.mem"}),
        .LSTM1_B_FILE({MEM_DIR, "/lstm1_bias_packed_fp16.mem"}),
        .LSTM2_W_BANK0_FILE({MEM_DIR, "/lstm2_weight_bank0_fp16.mem"}),
        .LSTM2_W_BANK1_FILE({MEM_DIR, "/lstm2_weight_bank1_fp16.mem"}),
        .LSTM2_W_BANK2_FILE({MEM_DIR, "/lstm2_weight_bank2_fp16.mem"}),
        .LSTM2_W_BANK3_FILE({MEM_DIR, "/lstm2_weight_bank3_fp16.mem"}),
        .LSTM2_B_FILE({MEM_DIR, "/lstm2_bias_packed_fp16.mem"}),
        .DENSE_W_BANK0_FILE({MEM_DIR, "/dense_weight_bank0_fp16.mem"}),
        .DENSE_W_BANK1_FILE({MEM_DIR, "/dense_weight_bank1_fp16.mem"}),
        .DENSE_W_BANK2_FILE({MEM_DIR, "/dense_weight_bank2_fp16.mem"}),
        .DENSE_W_BANK3_FILE({MEM_DIR, "/dense_weight_bank3_fp16.mem"}),
        .LSTM1_W_DOT9_FILE({MEM_DIR, "/lstm1_weight_dot9_fp16.mem"}),
        .LSTM2_W_DOT9_FILE({MEM_DIR, "/lstm2_weight_dot9_fp16.mem"}),
        .DENSE_W_DOT9_FILE({MEM_DIR, "/dense_weight_dot9_fp16.mem"}),
        .DENSE_B_FILE({MEM_DIR, "/dense_bias_packed_fp16.mem"}),
        .SIGMOID_FILE({MEM_DIR, "/sigmoid_lut_fp16.mem"}),
        .TANH_FILE({MEM_DIR, "/tanh_lut_fp16.mem"}),
        .POOL_MEAN(POOL_MEAN)
    ) u_lstm (
        .clk(clk),
        .rst(rst),
        .start(lstm_start),
        .input_write_enable(input_write_enable),
        .input_write_word_addr(input_write_word_addr),
        .input_write_data(input_write_data),
        .input_write_strb(input_write_strb),
        .stats_features(stats_features),
        .busy(lstm_busy),
        .done(lstm_done),
        .prediction(lstm_prediction),
        .cycle_count(lstm_cycles),
        .debug_input_word0(debug_input_word0),
        .debug_logits(debug_logits),
        .debug_pipeline(lstm_debug_pipeline),
        .mac_start(lstm_mac_start),
        .mac_a(lstm_mac_a),
        .mac_b(lstm_mac_b),
        .mac_acc(lstm_mac_acc),
        .mac_tile_mode(lstm_mac_tile_mode),
        .mac_tile_weights(lstm_mac_tile_weights),
        .mac_tile_vector(lstm_mac_tile_vector),
        .mac_tile_accumulator(lstm_mac_tile_accumulator),
        .mac_result(shared_mac_result),
        .mac_tile_result(shared_mac_tile_result),
        .mac_busy(!stats_owns_mac && shared_mac_busy),
        .mac_done(!stats_owns_mac && shared_mac_done),
        .mac_stream_valid(lstm_stream_valid),
        .mac_stream_tag(lstm_stream_tag),
        .mac_stream_weights(lstm_stream_weights),
        .mac_stream_vector(lstm_stream_vector),
        .mac_stream_accumulator(lstm_stream_accumulator),
        .mac_stream_ready(!stats_owns_mac && lstm_stream_ready),
        .mac_stream_out_valid(!stats_owns_mac && lstm_stream_out_valid),
        .mac_stream_out_tag(lstm_stream_out_tag),
        .mac_stream_result(lstm_stream_result)
    );

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            stats_start <= 1'b0;
            lstm_start <= 1'b0;
            busy <= 1'b0;
            done <= 1'b0;
            prediction <= 3'd0;
            cycle_count <= 32'd0;
            debug_stats_special <= 32'd0;
        end else begin
            stats_start <= 1'b0;
            lstm_start <= 1'b0;
            done <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        cycle_count <= 32'd0;
                        debug_stats_special <= 32'd0;
                        state <= S_STATS_KICK;
                    end
                end

                S_STATS_KICK: begin
                    stats_start <= 1'b1;
                    state <= S_STATS_WAIT;
                end

                S_STATS_WAIT: begin
                    cycle_count <= cycle_count + 1'b1;
                    if (stats_done) begin
                        for (debug_stats_index = 0;
                             debug_stats_index < 48;
                             debug_stats_index = debug_stats_index + 1)
                            if (stats_features[
                                    debug_stats_index*16 + 10 +: 5
                                ] == 5'h1F)
                                debug_stats_special <= {
                                    1'b1, 8'd0,
                                    debug_stats_index[6:0],
                                    stats_features[
                                        debug_stats_index*16 +: 16
                                    ]
                                };
                        state <= S_LSTM_KICK;
                    end
                end

                S_LSTM_KICK: begin
                    lstm_start <= 1'b1;
                    state <= S_LSTM_WAIT;
                end

                S_LSTM_WAIT: begin
                    cycle_count <= cycle_count + 1'b1;
                    if (lstm_done) begin
                        prediction <= lstm_prediction;
                        state <= S_DONE;
                    end
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
