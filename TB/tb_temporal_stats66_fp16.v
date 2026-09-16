`timescale 1ns/1ps

module tb_temporal_stats66_fp16;
    parameter MEM_DIR = ".";

    reg clk = 1'b0;
    reg rst = 1'b1;
    reg start = 1'b0;
    wire busy;
    wire done;
    wire [767:0] stats_features;
    wire [31:0] cycle_count;

    reg [15:0] expected_stats [0:47];
    integer index;
    integer errors;
    integer timeout_cycles;

    always #5 clk = ~clk;

    temporal_stats66_fp16 #(
        .INPUT_FILE({MEM_DIR, "/input_sample_fp16.mem"}),
        .STATS_W_FILE({MEM_DIR, "/stats_weight_packed_fp16.mem"}),
        .STATS_B_FILE({MEM_DIR, "/stats_bias_packed_fp16.mem"}),
        .TANH_LUT_FILE({MEM_DIR, "/tanh_lut_fp16.mem"})
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .busy(busy),
        .done(done),
        .stats_features(stats_features),
        .cycle_count(cycle_count)
    );

    initial begin
        $readmemh(
            {MEM_DIR, "/expected_stats_features_fp16.mem"},
            expected_stats
        );
        repeat (8) @(posedge clk);
        rst <= 1'b0;
        repeat (3) @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        timeout_cycles = 0;
        while (!done && timeout_cycles < 1000000) begin
            @(posedge clk);
            timeout_cycles = timeout_cycles + 1;
        end

        if (!done) begin
            $display("FAIL: Stats66 timeout");
            $finish;
        end

        errors = 0;
        for (index = 0; index < 48; index = index + 1) begin
            if (stats_features[index*16 +: 16] !== expected_stats[index]) begin
                $display(
                    "MISMATCH index=%0d rtl=%h expected=%h",
                    index,
                    stats_features[index*16 +: 16],
                    expected_stats[index]
                );
                errors = errors + 1;
            end
        end

        $display("stats_cycles=%0d errors=%0d", cycle_count, errors);
        if (errors == 0)
            $display("PASS: Stats66 -> FC48 bit-exact");
        else
            $display("FAIL: Stats66 -> FC48");
        $finish;
    end
endmodule
