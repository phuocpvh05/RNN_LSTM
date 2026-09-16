`timescale 1ns/1ps

module tb_fp16_dot9_isolation;
    localparam [15:0] FP16_ZERO = 16'h0000;
    localparam [15:0] FP16_HALF = 16'h3800;
    localparam [15:0] FP16_ONE  = 16'h3C00;
    localparam [15:0] FP16_TWO  = 16'h4000;
    localparam [15:0] FP16_NINE = 16'h4880;
    localparam [15:0] FP16_36   = 16'h5080;

    reg clk = 1'b0;
    reg rst = 1'b1;
    always #5 clk = ~clk;

    reg mul_en = 1'b1;
    reg [15:0] mul_a = FP16_ZERO;
    reg [15:0] mul_b = FP16_ZERO;
    wire [15:0] mul_result;

    reg add_en = 1'b1;
    reg [15:0] add_a = FP16_ZERO;
    reg [15:0] add_b = FP16_ZERO;
    wire [15:0] add_result;

    reg pe_valid_i = 1'b0;
    reg [143:0] pe_weight_i = 144'd0;
    reg [143:0] pe_activation_i = 144'd0;
    reg [143:0] pe_psum_i = 144'd0;
    wire [143:0] pe_psum_o;
    wire pe_valid_o;

    reg reduce_valid_i = 1'b0;
    reg [143:0] reduce_values_i = 144'd0;
    wire [15:0] reduce_sum_o;
    wire reduce_valid_o;

    reg array_valid_i = 1'b0;
    reg [7:0] array_tag_i = 8'd0;
    reg [2303:0] array_weights_i = 2304'd0;
    reg [575:0] array_vector_i = 576'd0;
    reg [63:0] array_accumulator_i = 64'd0;
    wire array_valid_o;
    wire [7:0] array_tag_o;
    wire [63:0] array_result_o;

    integer lane;
    integer error_count = 0;

    fp16_dot9_isolation_dut dut (
        .clk(clk), .rst(rst),
        .mul_en(mul_en), .mul_a(mul_a), .mul_b(mul_b),
        .mul_result(mul_result),
        .add_en(add_en), .add_a(add_a), .add_b(add_b),
        .add_result(add_result),
        .pe_valid_i(pe_valid_i), .pe_weight_i(pe_weight_i),
        .pe_activation_i(pe_activation_i), .pe_psum_i(pe_psum_i),
        .pe_psum_o(pe_psum_o), .pe_valid_o(pe_valid_o),
        .reduce_valid_i(reduce_valid_i),
        .reduce_values_i(reduce_values_i),
        .reduce_sum_o(reduce_sum_o), .reduce_valid_o(reduce_valid_o),
        .array_valid_i(array_valid_i), .array_tag_i(array_tag_i),
        .array_weights_i(array_weights_i),
        .array_vector_i(array_vector_i),
        .array_accumulator_i(array_accumulator_i),
        .array_valid_o(array_valid_o), .array_tag_o(array_tag_o),
        .array_result_o(array_result_o)
    );

    task check16;
        input [8*40-1:0] name;
        input [15:0] actual;
        input [15:0] expected;
        begin
            if (actual !== expected) begin
                $display("FAIL: %0s actual=%04h expected=%04h",
                         name, actual, expected);
                error_count = error_count + 1;
            end else begin
                $display("PASS: %0s = %04h", name, actual);
            end
        end
    endtask

    initial begin
        repeat (12) @(posedge clk);
        @(negedge clk);
        rst = 1'b0;

        // Primitive multiplier: 0.5 x 2.0 = 1.0.
        mul_a = FP16_HALF;
        mul_b = FP16_TWO;
        repeat (6) @(posedge clk);
        #1 check16("multiplier", mul_result, FP16_ONE);

        // Primitive adder: 1.0 + 1.0 = 2.0.
        @(negedge clk);
        add_a = FP16_ONE;
        add_b = FP16_ONE;
        repeat (7) @(posedge clk);
        #1 check16("adder", add_result, FP16_TWO);

        // One PE: every one of its nine independent lanes computes 1x1+0.
        @(negedge clk);
        for (lane = 0; lane < 9; lane = lane + 1) begin
            pe_weight_i[lane*16 +: 16] = FP16_ONE;
            pe_activation_i[lane*16 +: 16] = FP16_ONE;
            pe_psum_i[lane*16 +: 16] = FP16_ZERO;
        end
        pe_valid_i = 1'b1;
        @(negedge clk);
        pe_valid_i = 1'b0;
        @(posedge pe_valid_o);
        #1;
        for (lane = 0; lane < 9; lane = lane + 1)
            check16("PE lane", pe_psum_o[lane*16 +: 16], FP16_ONE);

        // Reduction tree: nine 1.0 values must produce 9.0.
        @(negedge clk);
        for (lane = 0; lane < 9; lane = lane + 1)
            reduce_values_i[lane*16 +: 16] = FP16_ONE;
        reduce_valid_i = 1'b1;
        @(negedge clk);
        reduce_valid_i = 1'b0;
        @(posedge reduce_valid_o);
        #1 check16("reduce9", reduce_sum_o, FP16_NINE);

        // Full 4x4 dot9 array: each row sums 4 columns x 9 lanes = 36.
        @(negedge clk);
        for (lane = 0; lane < 144; lane = lane + 1)
            array_weights_i[lane*16 +: 16] = FP16_ONE;
        for (lane = 0; lane < 36; lane = lane + 1)
            array_vector_i[lane*16 +: 16] = FP16_ONE;
        array_accumulator_i = 64'd0;
        array_tag_i = 8'h5A;
        array_valid_i = 1'b1;
        @(negedge clk);
        array_valid_i = 1'b0;
        @(posedge array_valid_o);
        #1;
        if (array_tag_o !== 8'h5A) begin
            $display("FAIL: array tag actual=%02h expected=5A", array_tag_o);
            error_count = error_count + 1;
        end else begin
            $display("PASS: array tag = %02h", array_tag_o);
        end
        for (lane = 0; lane < 4; lane = lane + 1)
            check16("array row", array_result_o[lane*16 +: 16], FP16_36);

        if (error_count == 0)
            $display("PASS: all isolated FP16 dot9 stages are correct");
        else
            $display("FAIL: isolated FP16 dot9 errors=%0d", error_count);

        $finish;
    end

    initial begin
        #20000;
        $display("FAIL: simulation timeout");
        $finish;
    end
endmodule
