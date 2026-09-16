`timescale 1ns/1ps

// AXI4-Lite wrapper around the on-chip Stats66 + LSTM64 + LSTM32 accelerator.
module lstm_accel_axi #(
    parameter MEM_DIR =
        ".",
    parameter integer POOL_MEAN = 0,
    parameter integer C_S00_AXI_DATA_WIDTH = 32,
    parameter integer C_S00_AXI_ADDR_WIDTH = 12
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 S00_AXI_ACLK CLK" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S00_AXI_ACLK, ASSOCIATED_BUSIF S00_AXI, ASSOCIATED_RESET s00_axi_aresetn, FREQ_HZ 85000000" *)
    input  wire                                  s00_axi_aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 S00_AXI_ARESETN RST" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S00_AXI_ARESETN, POLARITY ACTIVE_LOW" *)
    input  wire                                  s00_axi_aresetn,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S00_AXI AWADDR" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S00_AXI, PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 12, READ_WRITE_MODE READ_WRITE, FREQ_HZ 85000000" *)
    input  wire [C_S00_AXI_ADDR_WIDTH-1:0]       s00_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S00_AXI AWPROT" *)
    input  wire [2:0]                            s00_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S00_AXI AWVALID" *)
    input  wire                                  s00_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S00_AXI AWREADY" *)
    output wire                                  s00_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S00_AXI WDATA" *)
    input  wire [C_S00_AXI_DATA_WIDTH-1:0]       s00_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S00_AXI WSTRB" *)
    input  wire [(C_S00_AXI_DATA_WIDTH/8)-1:0]   s00_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S00_AXI WVALID" *)
    input  wire                                  s00_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S00_AXI WREADY" *)
    output wire                                  s00_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S00_AXI BRESP" *)
    output wire [1:0]                            s00_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S00_AXI BVALID" *)
    output wire                                  s00_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S00_AXI BREADY" *)
    input  wire                                  s00_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S00_AXI ARADDR" *)
    input  wire [C_S00_AXI_ADDR_WIDTH-1:0]       s00_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S00_AXI ARPROT" *)
    input  wire [2:0]                            s00_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S00_AXI ARVALID" *)
    input  wire                                  s00_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S00_AXI ARREADY" *)
    output wire                                  s00_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S00_AXI RDATA" *)
    output wire [C_S00_AXI_DATA_WIDTH-1:0]       s00_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S00_AXI RRESP" *)
    output wire [1:0]                            s00_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S00_AXI RVALID" *)
    output wire                                  s00_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S00_AXI RREADY" *)
    input  wire                                  s00_axi_rready
);

    wire        core_start;
    wire        core_busy;
    wire        core_done;
    wire [2:0]  core_prediction;
    wire [31:0] core_cycle_count;
    wire [31:0] core_debug_input_word0;
    wire [95:0] core_debug_logits;
    wire [383:0] core_debug_stages;
    wire        input_write_enable;
    wire [9:0]  input_write_word_addr;
    wire [31:0] input_write_data;
    wire [3:0]  input_write_strb;

    lstm_axi_lite_regs #(
        .C_S_AXI_DATA_WIDTH(C_S00_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH(C_S00_AXI_ADDR_WIDTH)
    ) u_axi_registers (
        .S_AXI_ACLK(s00_axi_aclk),
        .S_AXI_ARESETN(s00_axi_aresetn),
        .S_AXI_AWADDR(s00_axi_awaddr),
        .S_AXI_AWPROT(s00_axi_awprot),
        .S_AXI_AWVALID(s00_axi_awvalid),
        .S_AXI_AWREADY(s00_axi_awready),
        .S_AXI_WDATA(s00_axi_wdata),
        .S_AXI_WSTRB(s00_axi_wstrb),
        .S_AXI_WVALID(s00_axi_wvalid),
        .S_AXI_WREADY(s00_axi_wready),
        .S_AXI_BRESP(s00_axi_bresp),
        .S_AXI_BVALID(s00_axi_bvalid),
        .S_AXI_BREADY(s00_axi_bready),
        .S_AXI_ARADDR(s00_axi_araddr),
        .S_AXI_ARPROT(s00_axi_arprot),
        .S_AXI_ARVALID(s00_axi_arvalid),
        .S_AXI_ARREADY(s00_axi_arready),
        .S_AXI_RDATA(s00_axi_rdata),
        .S_AXI_RRESP(s00_axi_rresp),
        .S_AXI_RVALID(s00_axi_rvalid),
        .S_AXI_RREADY(s00_axi_rready),
        .core_start(core_start),
        .core_busy(core_busy),
        .core_done(core_done),
        .core_prediction(core_prediction),
        .core_cycle_count(core_cycle_count),
        .core_debug_input_word0(core_debug_input_word0),
        .core_debug_logits(core_debug_logits),
        .core_debug_stages(core_debug_stages),
        .input_write_enable(input_write_enable),
        .input_write_word_addr(input_write_word_addr),
        .input_write_data(input_write_data),
        .input_write_strb(input_write_strb)
    );

    lstm_stats66_fp16_top #(
        .MEM_DIR(MEM_DIR),
        .POOL_MEAN(POOL_MEAN)
    ) u_lstm_accelerator (
        .clk(s00_axi_aclk),
        .rst(!s00_axi_aresetn),
        .start(core_start),
        .input_write_enable(input_write_enable),
        .input_write_word_addr(input_write_word_addr),
        .input_write_data(input_write_data),
        .input_write_strb(input_write_strb),
        .busy(core_busy),
        .done(core_done),
        .prediction(core_prediction),
        .cycle_count(core_cycle_count),
        .debug_input_word0(core_debug_input_word0),
        .debug_logits(core_debug_logits),
        .debug_stages(core_debug_stages)
    );

endmodule
