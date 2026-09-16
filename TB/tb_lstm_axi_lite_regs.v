`timescale 1ns/1ps

module tb_lstm_axi_lite_regs;

    reg clk = 1'b0;
    reg aresetn = 1'b0;

    reg [4:0] awaddr = 5'd0;
    reg [2:0] awprot = 3'd0;
    reg awvalid = 1'b0;
    wire awready;
    reg [31:0] wdata = 32'd0;
    reg [3:0] wstrb = 4'hF;
    reg wvalid = 1'b0;
    wire wready;
    wire [1:0] bresp;
    wire bvalid;
    reg bready = 1'b0;

    reg [4:0] araddr = 5'd0;
    reg [2:0] arprot = 3'd0;
    reg arvalid = 1'b0;
    wire arready;
    wire [31:0] rdata;
    wire [1:0] rresp;
    wire rvalid;
    reg rready = 1'b0;

    wire core_start;
    reg core_busy = 1'b0;
    reg core_done = 1'b0;
    reg [2:0] core_prediction = 3'd0;
    reg [31:0] core_cycle_count = 32'd0;

    integer errors = 0;
    reg [31:0] read_value;

    always #5 clk = ~clk;

    lstm_axi_lite_regs dut (
        .S_AXI_ACLK(clk),
        .S_AXI_ARESETN(aresetn),
        .S_AXI_AWADDR(awaddr),
        .S_AXI_AWPROT(awprot),
        .S_AXI_AWVALID(awvalid),
        .S_AXI_AWREADY(awready),
        .S_AXI_WDATA(wdata),
        .S_AXI_WSTRB(wstrb),
        .S_AXI_WVALID(wvalid),
        .S_AXI_WREADY(wready),
        .S_AXI_BRESP(bresp),
        .S_AXI_BVALID(bvalid),
        .S_AXI_BREADY(bready),
        .S_AXI_ARADDR(araddr),
        .S_AXI_ARPROT(arprot),
        .S_AXI_ARVALID(arvalid),
        .S_AXI_ARREADY(arready),
        .S_AXI_RDATA(rdata),
        .S_AXI_RRESP(rresp),
        .S_AXI_RVALID(rvalid),
        .S_AXI_RREADY(rready),
        .core_start(core_start),
        .core_busy(core_busy),
        .core_done(core_done),
        .core_prediction(core_prediction),
        .core_cycle_count(core_cycle_count)
    );

    task axi_write;
        input [4:0] address;
        input [31:0] value;
        begin
            @(posedge clk);
            awaddr  <= address;
            wdata   <= value;
            awvalid <= 1'b1;
            wvalid  <= 1'b1;
            bready  <= 1'b1;
            while (!(awready && wready))
                @(posedge clk);
            @(posedge clk);
            awvalid <= 1'b0;
            wvalid  <= 1'b0;
            while (!bvalid)
                @(posedge clk);
            @(posedge clk);
            bready <= 1'b0;
        end
    endtask

    task axi_read;
        input [4:0] address;
        output [31:0] value;
        begin
            @(posedge clk);
            araddr  <= address;
            arvalid <= 1'b1;
            rready  <= 1'b1;
            while (!arready)
                @(posedge clk);
            @(posedge clk);
            arvalid <= 1'b0;
            while (!rvalid)
                @(posedge clk);
            value = rdata;
            @(posedge clk);
            rready <= 1'b0;
        end
    endtask

    initial begin
        repeat (5) @(posedge clk);
        aresetn <= 1'b1;
        repeat (2) @(posedge clk);

        axi_read(5'h10, read_value);
        if (read_value !== 32'h0001_0000) begin
            $display("ERROR: VERSION=%h", read_value);
            errors = errors + 1;
        end

        axi_read(5'h04, read_value);
        if (read_value[2:0] !== 3'b100) begin
            $display("ERROR: initial STATUS=%h", read_value);
            errors = errors + 1;
        end

        fork
            begin
                axi_write(5'h00, 32'h0000_0001);
            end
            begin
                wait (core_start);
                @(posedge clk);
                #1;
                if (core_start !== 1'b0) begin
                    $display("ERROR: core_start is not one cycle");
                    errors = errors + 1;
                end
                core_busy <= 1'b1;
            end
        join

        axi_read(5'h04, read_value);
        if (read_value[2:0] !== 3'b001) begin
            $display("ERROR: busy STATUS=%h", read_value);
            errors = errors + 1;
        end

        repeat (5) @(posedge clk);
        core_prediction  <= 3'd4;
        core_cycle_count <= 32'd11687294;
        core_busy        <= 1'b0;
        core_done        <= 1'b1;
        @(posedge clk);
        core_done <= 1'b0;
        repeat (2) @(posedge clk);

        axi_read(5'h04, read_value);
        if (read_value[2:0] !== 3'b110) begin
            $display("ERROR: completed STATUS=%h", read_value);
            errors = errors + 1;
        end

        axi_read(5'h08, read_value);
        if (read_value !== 32'd4) begin
            $display("ERROR: PREDICTION=%h", read_value);
            errors = errors + 1;
        end

        axi_read(5'h0C, read_value);
        if (read_value !== 32'd11687294) begin
            $display("ERROR: CYCLES=%0d", read_value);
            errors = errors + 1;
        end

        axi_write(5'h00, 32'h0000_0002);
        axi_read(5'h04, read_value);
        if (read_value[1] !== 1'b0) begin
            $display("ERROR: done_latched did not clear");
            errors = errors + 1;
        end

        if (errors == 0)
            $display("PASS: AXI4-Lite LSTM control/status registers");
        else
            $display("FAIL: AXI4-Lite register errors=%0d", errors);
        $finish;
    end

endmodule
