`timescale 1ns/1ps

// Hardware-friendly temporal statistics and projection branch.
//
// Input:  128 timesteps x 9 FP16 channels.
// Stats:  54 original per-channel values
//       + 9 cross-axis product means
//       + 3 total-acceleration variances
//       = 66 values.
// Output: tanh(FC48(stats)), packed element 0 in bits [15:0].
module temporal_stats66_fp16 #(
    parameter INPUT_FILE       = "input_sample_fp16.mem",
    parameter INPUT_PACKED_FILE = "input_sample_packed32.mem",
    parameter STATS_W_FILE     = "stats_weight_packed_fp16.mem",
    parameter STATS_B_FILE     = "stats_bias_packed_fp16.mem",
    parameter TANH_LUT_FILE    = "tanh_lut_fp16.mem"
) (
    input              clk,
    input              rst,
    input              start,
    input              input_write_enable,
    input      [9:0]   input_write_word_addr,
    input      [31:0]  input_write_data,
    input      [3:0]   input_write_strb,
    output reg         busy,
    output reg         done,
    output reg [767:0] stats_features,
    output reg [31:0]  cycle_count,
    output reg         mac_start,
    output reg [127:0] mac_a,
    output reg [127:0] mac_b,
    output reg [127:0] mac_acc,
    input      [127:0] mac_result,
    input              mac_busy,
    input              mac_done
);

    localparam [15:0] FP16_ZERO   = 16'h0000;
    localparam [15:0] FP16_ONE    = 16'h3C00;
    localparam [15:0] FP16_INV128 = 16'h2000;
    localparam [15:0] FP16_INV127 = 16'h2008;

    localparam MODE_STREAM = 2'd0;
    localparam MODE_FINAL  = 2'd1;
    localparam MODE_FC     = 2'd2;

    localparam OP_SUM      = 4'd0;
    localparam OP_ENERGY   = 4'd1;
    localparam OP_DIFF_RAW = 4'd2;
    localparam OP_DIFF_ACC = 4'd3;
    localparam OP_ADJ      = 4'd4;
    localparam OP_CROSS    = 4'd5;

    localparam F_MEAN      = 4'd0;
    localparam F_ENERGY    = 4'd1;
    localparam F_DIFF      = 4'd2;
    localparam F_ADJ       = 4'd3;
    localparam F_CROSS     = 4'd4;
    localparam F_VARIANCE  = 4'd5;

    localparam S_IDLE          = 4'd0;
    localparam S_LOAD_SAMPLE   = 4'd1;
    localparam S_MINMAX        = 4'd2;
    localparam S_STREAM_PREP   = 4'd3;
    localparam S_FINAL_DIRECT  = 4'd4;
    localparam S_FINAL_PREP    = 4'd5;
    localparam S_FC_INIT       = 4'd6;
    localparam S_FC_PREP       = 4'd7;
    localparam S_MAC_START     = 4'd8;
    localparam S_MAC_WAIT      = 4'd9;
    localparam S_DONE          = 4'd10;
    localparam S_WEIGHT_WAIT   = 4'd11;
    localparam S_LOAD_WAIT     = 4'd12;
    localparam S_LOAD_CAPTURE  = 4'd13;

    // Store two consecutive FP16 samples in each 32-bit word.  This matches
    // the AXI write width, so one bus transfer performs one RAM write.  The
    // former 16-bit organization wrote two independently addressed words in
    // one cycle and therefore could not be mapped to a 7-series block RAM.
    (* ram_style = "block" *)
    reg [31:0] input_mem [0:575];
    (* rom_style = "block" *)
    reg [127:0] stats_weight_mem [0:431];
    (* rom_style = "block" *)
    reg [127:0] stats_bias_mem [0:5];
    (* rom_style = "block" *)
    reg [15:0] tanh_lut_mem [0:255];

    initial begin
        $readmemh(INPUT_PACKED_FILE, input_mem);
        $readmemh(STATS_W_FILE, stats_weight_mem);
        $readmemh(STATS_B_FILE, stats_bias_mem);
        $readmemh(TANH_LUT_FILE, tanh_lut_mem);
    end

    // Mirror AXI input writes into the statistics branch input memory.
    always @(posedge clk) begin
        if (input_write_enable) begin
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
        end
    end

    // The old implementation read all nine sensor channels asynchronously
    // in one cycle. Vivado therefore replicated this memory into a large
    // multi-port LUT mux. Read one channel at a time through a synchronous
    // port so the complete runtime input buffer is inferred as block RAM.
    reg [10:0] input_read_addr;
    reg [31:0] input_read_word_q;
    reg        input_read_half_q;
    wire [15:0] input_read_q = input_read_half_q ?
                               input_read_word_q[31:16] :
                               input_read_word_q[15:0];

    always @(posedge clk) begin
        input_read_word_q <= input_mem[input_read_addr[10:1]];
        input_read_half_q <= input_read_addr[0];
    end

    reg [3:0] state;
    reg [1:0] mac_mode;
    reg [3:0] operation;
    reg       group_index;
    reg [7:0] timestep;
    reg [3:0] load_channel;
    reg [2:0] fc_tile;
    reg [6:0] fc_column;

    reg [127:0] current_vec [0:1];
    reg [127:0] previous_vec [0:1];
    reg [127:0] sum_vec [0:1];
    reg [127:0] energy_vec [0:1];
    reg [127:0] difference_vec [0:1];
    reg [127:0] adjacent_vec [0:1];
    reg [127:0] cross_vec [0:1];
    reg [127:0] minimum_vec [0:1];
    reg [127:0] maximum_vec [0:1];
    reg [127:0] temporary_difference [0:1];
    reg [127:0] mean_vec [0:1];
    reg [127:0] energy_mean_vec [0:1];
    reg [15:0] statistics [0:65];
    reg [127:0] fc_accumulator;

    // Registered ROM output allows the 432x128-bit statistics weights to use
    // block RAM.  The FSM below accounts for the one-cycle BRAM read latency.
    wire [8:0] stats_weight_addr = fc_tile*9'd72 + fc_column;
    reg [127:0] stats_weight_q;

    always @(posedge clk) begin
        stats_weight_q <= stats_weight_mem[stats_weight_addr];
    end

    function [15:0] lane16;
        input [127:0] value;
        input integer lane;
        begin
            lane16 = value[lane*16 +: 16];
        end
    endfunction

    function [15:0] fp16_negate;
        input [15:0] value;
        begin
            fp16_negate = {~value[15], value[14:0]};
        end
    endfunction

    function [15:0] fp16_absolute;
        input [15:0] value;
        begin
            fp16_absolute = {1'b0, value[14:0]};
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

    // Convert an FP16 value to the LUT address used by the Python exporter:
    // address = clamp(round_to_even(value * 16), -128, 127) + 128.
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

    integer lane;
    integer channel;
    integer statistic_index;
    reg [15:0] current_value;
    reg [15:0] min_value;
    reg [15:0] max_value;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            cycle_count <= 32'd0;
            stats_features <= 768'd0;
            mac_start <= 1'b0;
            mac_mode <= MODE_STREAM;
            operation <= OP_SUM;
            group_index <= 1'b0;
            timestep <= 8'd0;
            load_channel <= 4'd0;
            input_read_addr <= 11'd0;
            fc_tile <= 3'd0;
            fc_column <= 7'd0;
            mac_a <= 128'd0;
            mac_b <= 128'd0;
            mac_acc <= 128'd0;
            fc_accumulator <= 128'd0;
            for (channel = 0; channel < 2; channel = channel + 1) begin
                current_vec[channel] <= 128'd0;
                previous_vec[channel] <= 128'd0;
                sum_vec[channel] <= 128'd0;
                energy_vec[channel] <= 128'd0;
                difference_vec[channel] <= 128'd0;
                adjacent_vec[channel] <= 128'd0;
                cross_vec[channel] <= 128'd0;
                minimum_vec[channel] <= 128'd0;
                maximum_vec[channel] <= 128'd0;
                temporary_difference[channel] <= 128'd0;
                mean_vec[channel] <= 128'd0;
                energy_mean_vec[channel] <= 128'd0;
            end
            for (statistic_index = 0; statistic_index < 66;
                 statistic_index = statistic_index + 1)
                statistics[statistic_index] <= 16'd0;
        end else begin
            done <= 1'b0;
            mac_start <= 1'b0;
            if (busy)
                cycle_count <= cycle_count + 1'b1;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        cycle_count <= 32'd0;
                        stats_features <= 768'd0;
                        timestep <= 8'd0;
                        operation <= OP_SUM;
                        group_index <= 1'b0;
                        for (channel = 0; channel < 2;
                             channel = channel + 1) begin
                            previous_vec[channel] <= 128'd0;
                            sum_vec[channel] <= 128'd0;
                            energy_vec[channel] <= 128'd0;
                            difference_vec[channel] <= 128'd0;
                            adjacent_vec[channel] <= 128'd0;
                            cross_vec[channel] <= 128'd0;
                        end
                        state <= S_LOAD_SAMPLE;
                    end
                end

                S_LOAD_SAMPLE: begin
                    load_channel <= 4'd0;
                    input_read_addr <= timestep*11'd9;
                    current_vec[1] <= 128'd0;
                    state <= S_LOAD_WAIT;
                end

                // One wait cycle accounts for the registered BRAM output.
                S_LOAD_WAIT: begin
                    state <= S_LOAD_CAPTURE;
                end

                S_LOAD_CAPTURE: begin
                    if (load_channel < 8)
                        current_vec[0][load_channel*16 +: 16] <=
                            input_read_q;
                    else
                        current_vec[1][15:0] <= input_read_q;

                    if (load_channel == 8) begin
                        state <= S_MINMAX;
                    end else begin
                        load_channel <= load_channel + 1'b1;
                        input_read_addr <= timestep*11'd9
                                           + load_channel + 1'b1;
                        state <= S_LOAD_WAIT;
                    end
                end

                S_MINMAX: begin
                    for (channel = 0; channel < 9;
                         channel = channel + 1) begin
                        if (channel < 8)
                            current_value = lane16(current_vec[0], channel);
                        else
                            current_value = lane16(current_vec[1], 0);

                        if (timestep == 0) begin
                            if (channel < 8) begin
                                minimum_vec[0][channel*16 +: 16] <=
                                    current_value;
                                maximum_vec[0][channel*16 +: 16] <=
                                    current_value;
                            end else begin
                                minimum_vec[1][15:0] <= current_value;
                                maximum_vec[1][15:0] <= current_value;
                            end
                        end else begin
                            if (channel < 8) begin
                                min_value = lane16(minimum_vec[0], channel);
                                max_value = lane16(maximum_vec[0], channel);
                                if (fp16_less(current_value, min_value))
                                    minimum_vec[0][channel*16 +: 16] <=
                                        current_value;
                                if (fp16_less(max_value, current_value))
                                    maximum_vec[0][channel*16 +: 16] <=
                                        current_value;
                            end else begin
                                min_value = lane16(minimum_vec[1], 0);
                                max_value = lane16(maximum_vec[1], 0);
                                if (fp16_less(current_value, min_value))
                                    minimum_vec[1][15:0] <= current_value;
                                if (fp16_less(max_value, current_value))
                                    maximum_vec[1][15:0] <= current_value;
                            end
                        end
                    end
                    operation <= OP_SUM;
                    group_index <= 1'b0;
                    mac_mode <= MODE_STREAM;
                    state <= S_STREAM_PREP;
                end

                S_STREAM_PREP: begin
                    mac_a <= 128'd0;
                    mac_b <= 128'd0;
                    mac_acc <= 128'd0;
                    case (operation)
                        OP_SUM: begin
                            mac_a <= current_vec[group_index];
                            for (lane = 0; lane < 8; lane = lane + 1)
                                mac_b[lane*16 +: 16] <= FP16_ONE;
                            mac_acc <= sum_vec[group_index];
                        end
                        OP_ENERGY: begin
                            mac_a <= current_vec[group_index];
                            mac_b <= current_vec[group_index];
                            mac_acc <= energy_vec[group_index];
                        end
                        OP_DIFF_RAW: begin
                            mac_a <= current_vec[group_index];
                            for (lane = 0; lane < 8; lane = lane + 1) begin
                                mac_b[lane*16 +: 16] <= FP16_ONE;
                                mac_acc[lane*16 +: 16] <= fp16_negate(
                                    lane16(previous_vec[group_index], lane)
                                );
                            end
                        end
                        OP_DIFF_ACC: begin
                            for (lane = 0; lane < 8; lane = lane + 1) begin
                                mac_a[lane*16 +: 16] <= fp16_absolute(
                                    lane16(
                                        temporary_difference[group_index],
                                        lane
                                    )
                                );
                                mac_b[lane*16 +: 16] <= FP16_ONE;
                            end
                            mac_acc <= difference_vec[group_index];
                        end
                        OP_ADJ: begin
                            mac_a <= current_vec[group_index];
                            mac_b <= previous_vec[group_index];
                            mac_acc <= adjacent_vec[group_index];
                        end
                        OP_CROSS: begin
                            if (!group_index) begin
                                mac_a[0*16 +: 16] <= lane16(current_vec[0], 0);
                                mac_b[0*16 +: 16] <= lane16(current_vec[0], 1);
                                mac_a[1*16 +: 16] <= lane16(current_vec[0], 3);
                                mac_b[1*16 +: 16] <= lane16(current_vec[0], 4);
                                mac_a[2*16 +: 16] <= lane16(current_vec[0], 6);
                                mac_b[2*16 +: 16] <= lane16(current_vec[0], 7);
                                mac_a[3*16 +: 16] <= lane16(current_vec[0], 0);
                                mac_b[3*16 +: 16] <= lane16(current_vec[0], 2);
                                mac_a[4*16 +: 16] <= lane16(current_vec[0], 3);
                                mac_b[4*16 +: 16] <= lane16(current_vec[0], 5);
                                mac_a[5*16 +: 16] <= lane16(current_vec[0], 6);
                                mac_b[5*16 +: 16] <= lane16(current_vec[1], 0);
                                mac_a[6*16 +: 16] <= lane16(current_vec[0], 1);
                                mac_b[6*16 +: 16] <= lane16(current_vec[0], 2);
                                mac_a[7*16 +: 16] <= lane16(current_vec[0], 4);
                                mac_b[7*16 +: 16] <= lane16(current_vec[0], 5);
                            end else begin
                                mac_a[15:0] <= lane16(current_vec[0], 7);
                                mac_b[15:0] <= lane16(current_vec[1], 0);
                            end
                            mac_acc <= cross_vec[group_index];
                        end
                        default: begin
                            mac_a <= 128'd0;
                            mac_b <= 128'd0;
                            mac_acc <= 128'd0;
                        end
                    endcase
                    state <= S_MAC_START;
                end

                S_FINAL_DIRECT: begin
                    for (channel = 0; channel < 9;
                         channel = channel + 1) begin
                        if (channel < 8) begin
                            statistics[27 + channel] <=
                                lane16(minimum_vec[0], channel);
                            statistics[36 + channel] <=
                                lane16(maximum_vec[0], channel);
                        end else begin
                            statistics[35] <= lane16(minimum_vec[1], 0);
                            statistics[44] <= lane16(maximum_vec[1], 0);
                        end
                    end
                    operation <= F_MEAN;
                    group_index <= 1'b0;
                    mac_mode <= MODE_FINAL;
                    state <= S_FINAL_PREP;
                end

                S_FINAL_PREP: begin
                    mac_a <= 128'd0;
                    mac_b <= 128'd0;
                    mac_acc <= 128'd0;
                    case (operation)
                        F_MEAN: begin
                            mac_a <= sum_vec[group_index];
                            for (lane = 0; lane < 8; lane = lane + 1)
                                mac_b[lane*16 +: 16] <= FP16_INV128;
                        end
                        F_ENERGY: begin
                            mac_a <= energy_vec[group_index];
                            for (lane = 0; lane < 8; lane = lane + 1)
                                mac_b[lane*16 +: 16] <= FP16_INV128;
                        end
                        F_DIFF: begin
                            mac_a <= difference_vec[group_index];
                            for (lane = 0; lane < 8; lane = lane + 1)
                                mac_b[lane*16 +: 16] <= FP16_INV127;
                        end
                        F_ADJ: begin
                            mac_a <= adjacent_vec[group_index];
                            for (lane = 0; lane < 8; lane = lane + 1)
                                mac_b[lane*16 +: 16] <= FP16_INV127;
                        end
                        F_CROSS: begin
                            mac_a <= cross_vec[group_index];
                            for (lane = 0; lane < 8; lane = lane + 1)
                                mac_b[lane*16 +: 16] <= FP16_INV128;
                        end
                        F_VARIANCE: begin
                            mac_a[0*16 +: 16] <= fp16_negate(
                                lane16(mean_vec[0], 6)
                            );
                            mac_b[0*16 +: 16] <= lane16(mean_vec[0], 6);
                            mac_acc[0*16 +: 16] <=
                                lane16(energy_mean_vec[0], 6);
                            mac_a[1*16 +: 16] <= fp16_negate(
                                lane16(mean_vec[0], 7)
                            );
                            mac_b[1*16 +: 16] <= lane16(mean_vec[0], 7);
                            mac_acc[1*16 +: 16] <=
                                lane16(energy_mean_vec[0], 7);
                            mac_a[2*16 +: 16] <= fp16_negate(
                                lane16(mean_vec[1], 0)
                            );
                            mac_b[2*16 +: 16] <= lane16(mean_vec[1], 0);
                            mac_acc[2*16 +: 16] <=
                                lane16(energy_mean_vec[1], 0);
                        end
                        default: begin
                            mac_a <= 128'd0;
                            mac_b <= 128'd0;
                            mac_acc <= 128'd0;
                        end
                    endcase
                    state <= S_MAC_START;
                end

                S_FC_INIT: begin
                    fc_accumulator <= stats_bias_mem[fc_tile];
                    fc_column <= 7'd0;
                    state <= S_FC_PREP;
                end

                S_FC_PREP: begin
                    for (lane = 0; lane < 8; lane = lane + 1)
                        mac_b[lane*16 +: 16] <=
                            (fc_column < 66)
                            ? statistics[fc_column]
                            : FP16_ZERO;
                    mac_acc <= fc_accumulator;
                    mac_mode <= MODE_FC;
                    state <= S_WEIGHT_WAIT;
                end

                S_WEIGHT_WAIT: begin
                    mac_a <= stats_weight_q;
                    state <= S_MAC_START;
                end

                S_MAC_START: begin
                    mac_start <= 1'b1;
                    state <= S_MAC_WAIT;
                end

                S_MAC_WAIT: begin
                    if (mac_done) begin
                        if (mac_mode == MODE_STREAM) begin
                            case (operation)
                                OP_SUM:
                                    sum_vec[group_index] <= mac_result;
                                OP_ENERGY:
                                    energy_vec[group_index] <= mac_result;
                                OP_DIFF_RAW:
                                    for (lane = 0; lane < 8;
                                         lane = lane + 1)
                                        temporary_difference[group_index]
                                            [lane*16 +: 16] <= {
                                                1'b0,
                                                mac_result[lane*16 +: 15]
                                            };
                                OP_DIFF_ACC:
                                    difference_vec[group_index] <= mac_result;
                                OP_ADJ:
                                    adjacent_vec[group_index] <= mac_result;
                                OP_CROSS:
                                    cross_vec[group_index] <= mac_result;
                            endcase

                            if (!group_index) begin
                                group_index <= 1'b1;
                                state <= S_STREAM_PREP;
                            end else begin
                                group_index <= 1'b0;
                                if (operation == OP_ENERGY &&
                                    timestep == 0) begin
                                    operation <= OP_CROSS;
                                    state <= S_STREAM_PREP;
                                end else if (operation == OP_CROSS) begin
                                    previous_vec[0] <= current_vec[0];
                                    previous_vec[1] <= current_vec[1];
                                    if (timestep == 127) begin
                                        state <= S_FINAL_DIRECT;
                                    end else begin
                                        timestep <= timestep + 1'b1;
                                        state <= S_LOAD_SAMPLE;
                                    end
                                end else begin
                                    operation <= operation + 1'b1;
                                    state <= S_STREAM_PREP;
                                end
                            end
                        end else if (mac_mode == MODE_FINAL) begin
                            case (operation)
                                F_MEAN: begin
                                    mean_vec[group_index] <= mac_result;
                                    for (lane = 0; lane < 8;
                                         lane = lane + 1)
                                        if (group_index*8 + lane < 9)
                                            statistics[
                                                group_index*8 + lane
                                            ] <= mac_result[
                                                lane*16 +: 16
                                            ];
                                end
                                F_ENERGY: begin
                                    energy_mean_vec[group_index] <= mac_result;
                                    for (lane = 0; lane < 8;
                                         lane = lane + 1)
                                        if (group_index*8 + lane < 9)
                                            statistics[
                                                9 + group_index*8 + lane
                                            ] <= mac_result[
                                                lane*16 +: 16
                                            ];
                                end
                                F_DIFF:
                                    for (lane = 0; lane < 8;
                                         lane = lane + 1)
                                        if (group_index*8 + lane < 9)
                                            statistics[
                                                18 + group_index*8 + lane
                                            ] <= mac_result[
                                                lane*16 +: 16
                                            ];
                                F_ADJ:
                                    for (lane = 0; lane < 8;
                                         lane = lane + 1)
                                        if (group_index*8 + lane < 9)
                                            statistics[
                                                45 + group_index*8 + lane
                                            ] <= mac_result[
                                                lane*16 +: 16
                                            ];
                                F_CROSS:
                                    for (lane = 0; lane < 8;
                                         lane = lane + 1)
                                        if (group_index*8 + lane < 9)
                                            statistics[
                                                54 + group_index*8 + lane
                                            ] <= mac_result[
                                                lane*16 +: 16
                                            ];
                                F_VARIANCE: begin
                                    for (lane = 0; lane < 3;
                                         lane = lane + 1)
                                        statistics[63 + lane] <=
                                            mac_result[lane*16 + 15]
                                            ? FP16_ZERO
                                            : mac_result[
                                                lane*16 +: 16
                                            ];
                                end
                            endcase

                            if (operation == F_VARIANCE) begin
                                fc_tile <= 3'd0;
                                state <= S_FC_INIT;
                            end else if (!group_index) begin
                                group_index <= 1'b1;
                                state <= S_FINAL_PREP;
                            end else begin
                                group_index <= 1'b0;
                                operation <= operation + 1'b1;
                                state <= S_FINAL_PREP;
                            end
                        end else begin
                            if (fc_column == 71) begin
                                for (lane = 0; lane < 8;
                                     lane = lane + 1)
                                    stats_features[
                                        (fc_tile*8 + lane)*16 +: 16
                                    ] <= tanh_lut_mem[
                                        activation_address(
                                            mac_result[lane*16 +: 16]
                                        )
                                    ];
                                if (fc_tile == 5) begin
                                    state <= S_DONE;
                                end else begin
                                    fc_tile <= fc_tile + 1'b1;
                                    state <= S_FC_INIT;
                                end
                            end else begin
                                fc_accumulator <= mac_result;
                                fc_column <= fc_column + 1'b1;
                                state <= S_FC_PREP;
                            end
                        end
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
