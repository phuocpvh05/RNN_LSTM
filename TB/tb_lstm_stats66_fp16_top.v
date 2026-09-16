`timescale 1ns/1ps

module tb_lstm_stats66_fp16_top;

    parameter MEM_DIR =
        ".";

    reg clk = 1'b0;
    reg rst = 1'b1;
    reg start = 1'b0;
    wire busy;
    wire done;
    wire [2:0] prediction;
    wire [31:0] cycle_count;

    reg [7:0] expected_class [0:0];
    reg [15:0] expected_stats [0:47];
    reg [15:0] expected_h1 [0:63];
    reg [15:0] expected_c1 [0:63];
    reg [15:0] expected_h2 [0:31];
    reg [15:0] expected_c2 [0:31];
    reg [15:0] expected_logits [0:5];

    integer index;
    integer errors;
    integer timeout_cycles;

    always #5 clk = ~clk;

    lstm_stats66_fp16_top #(
        .MEM_DIR(MEM_DIR),
        .POOL_MEAN(0)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .input_write_enable(1'b0),
        .input_write_word_addr(10'd0),
        .input_write_data(32'd0),
        .input_write_strb(4'd0),
        .busy(busy),
        .done(done),
        .prediction(prediction),
        .cycle_count(cycle_count)
    );

    initial begin
        $readmemh({MEM_DIR, "/expected_class.mem"}, expected_class);
        $readmemh(
            {MEM_DIR, "/expected_stats_features_fp16.mem"},
            expected_stats
        );
        $readmemh(
            {MEM_DIR, "/expected_lstm1_hidden_dot9_fp16.mem"},
            expected_h1
        );
        $readmemh(
            {MEM_DIR, "/expected_lstm1_cell_dot9_fp16.mem"},
            expected_c1
        );
        $readmemh(
            {MEM_DIR, "/expected_lstm2_hidden_dot9_fp16.mem"},
            expected_h2
        );
        $readmemh(
            {MEM_DIR, "/expected_lstm2_cell_dot9_fp16.mem"},
            expected_c2
        );
        $readmemh(
            {MEM_DIR, "/expected_logits_dot9_fp16.mem"},
            expected_logits
        );

        if (^expected_class[0] === 1'bx) begin
            $display(
                "FAIL: .mem files were not loaded from MEM_DIR=%s",
                MEM_DIR
            );
            $finish;
        end

        repeat (8) @(posedge clk);
        rst <= 1'b0;
        repeat (3) @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;

        wait (dut.u_statistics.done);
        @(posedge clk);
        errors = 0;
        for (index = 0; index < 48; index = index + 1) begin
            if (dut.stats_features[index*16 +: 16] !==
                expected_stats[index]) begin
                $display(
                    "STATS MISMATCH index=%0d rtl=%h expected=%h",
                    index,
                    dut.stats_features[index*16 +: 16],
                    expected_stats[index]
                );
                errors = errors + 1;
            end
        end
        if (errors == 0)
            $display("PASS: Stats66 -> FC48 bit-exact");
        else
            $display("FAIL: statistics errors=%0d", errors);

        timeout_cycles = 0;
        while (!done && timeout_cycles < 20000000) begin
            @(posedge clk);
            timeout_cycles = timeout_cycles + 1;
        end

        if (!done) begin
            $display("FAIL: inference timeout");
            $finish;
        end

        for (index = 0; index < 64; index = index + 1) begin
            if (dut.u_lstm.hidden1[index] !== expected_h1[index])
                errors = errors + 1;
            if (dut.u_lstm.cell1[index] !== expected_c1[index])
                errors = errors + 1;
        end
        for (index = 0; index < 32; index = index + 1) begin
            if (dut.u_lstm.hidden2[index] !== expected_h2[index])
                errors = errors + 1;
            if (dut.u_lstm.cell2[index] !== expected_c2[index])
                errors = errors + 1;
        end
        for (index = 0; index < 6; index = index + 1) begin
            if (dut.u_lstm.logits[index] !== expected_logits[index]) begin
                $display(
                    "LOGIT MISMATCH index=%0d rtl=%h expected=%h",
                    index,
                    dut.u_lstm.logits[index],
                    expected_logits[index]
                );
                errors = errors + 1;
            end
        end

        $display(
            "cycles=%0d predicted=%0d expected=%0d",
            cycle_count,
            prediction,
            expected_class[0][2:0]
        );

        if ((prediction === expected_class[0][2:0]) && errors == 0)
            $display("PASS: complete Stats66 + LSTM64 + LSTM32 FP16 RTL");
        else
            $display(
                "FAIL: prediction/state mismatch, total errors=%0d",
                errors
            );
        $finish;
    end

endmodule
