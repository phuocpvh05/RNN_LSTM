`timescale 1ns/1ps

// VC707 standalone smoke-test top.
//
// After reset is released, one complete 128x9 sample from the initialized
// input memory is inferred automatically.  The result is displayed on LEDs:
//   LED[2:0] = predicted class (0..5)
//   LED[3]   = accelerator busy
//   LED[4]   = inference completed (latched)
//   LED[7:5] = zero
//
// This top intentionally does not depend on Zynq PS.  The AXI/MicroBlaze
// wrapper can be added after the 9-MAC/PE core has passed implementation.
module vc707_lstm_dot9_top (
    input  wire       sys_clk_p,
    input  wire       sys_clk_n,
    input  wire       reset,
    output wire [7:0] leds
);
    wire clk_200mhz;
    wire clk_fb_mmcm;
    wire clk_fb_buf;
    wire clk_100mhz_mmcm;
    wire clk_100mhz;
    wire mmcm_locked;

    IBUFDS #(
        .DIFF_TERM("TRUE"),
        .IBUF_LOW_PWR("FALSE"),
        .IOSTANDARD("LVDS")
    ) u_sysclk_ibuf (
        .I(sys_clk_p),
        .IB(sys_clk_n),
        .O(clk_200mhz)
    );

    // VC707 system clock is 200 MHz.  VCO=1000 MHz, CLKOUT0=100 MHz.
    MMCME2_BASE #(
        .BANDWIDTH("OPTIMIZED"),
        .CLKIN1_PERIOD(5.000),
        .DIVCLK_DIVIDE(1),
        .CLKFBOUT_MULT_F(5.000),
        .CLKOUT0_DIVIDE_F(10.000),
        .STARTUP_WAIT("FALSE")
    ) u_core_mmcm (
        .CLKIN1(clk_200mhz),
        .CLKFBIN(clk_fb_buf),
        .RST(reset),
        .PWRDWN(1'b0),
        .CLKFBOUT(clk_fb_mmcm),
        .CLKOUT0(clk_100mhz_mmcm),
        .LOCKED(mmcm_locked)
    );

    BUFG u_clkfb_bufg (
        .I(clk_fb_mmcm),
        .O(clk_fb_buf)
    );

    BUFG u_coreclk_bufg (
        .I(clk_100mhz_mmcm),
        .O(clk_100mhz)
    );

    reg [3:0] reset_pipe = 4'hf;
    always @(posedge clk_100mhz or negedge mmcm_locked) begin
        if (!mmcm_locked)
            reset_pipe <= 4'hf;
        else
            reset_pipe <= {reset_pipe[2:0], 1'b0};
    end
    wire core_reset = reset_pipe[3];

    reg start_pulse;
    reg started;
    reg done_latched;
    reg [2:0] prediction_latched;
    wire core_busy;
    wire core_done;
    wire [2:0] core_prediction;
    wire [31:0] core_cycles;

    always @(posedge clk_100mhz) begin
        if (core_reset) begin
            start_pulse <= 1'b0;
            started <= 1'b0;
            done_latched <= 1'b0;
            prediction_latched <= 3'd0;
        end else begin
            start_pulse <= !started;
            if (!started)
                started <= 1'b1;
            if (core_done) begin
                done_latched <= 1'b1;
                prediction_latched <= core_prediction;
            end
        end
    end

    lstm_stats66_fp16_top u_accelerator (
        .clk(clk_100mhz),
        .rst(core_reset),
        .start(start_pulse),
        .input_write_enable(1'b0),
        .input_write_word_addr(10'd0),
        .input_write_data(32'd0),
        .input_write_strb(4'd0),
        .busy(core_busy),
        .done(core_done),
        .prediction(core_prediction),
        .cycle_count(core_cycles)
    );

    assign leds = {3'b000, done_latched, core_busy, prediction_latched};

    wire unused_cycles = ^core_cycles;
endmodule
