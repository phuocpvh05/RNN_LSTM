`timescale 1ns/1ps

// AXI4-Lite control/status register bank and PS-writable input window.
//
// Address map:
//   0x000 CONTROL       bit 0: write 1 to start
//                       bit 1: write 1 to clear done_latched
//   0x004 STATUS        bit 0: busy
//                       bit 1: done_latched
//                       bit 2: ready (= !busy)
//   0x008 PREDICTION    bits [2:0]: predicted class
//   0x00C CYCLES        inference cycle count
//   0x010 VERSION       0x0002_0000
//   0x014 INPUT_WORDS   576 32-bit words per sample
//   0x018 INPUT_WRITES  number of accepted input words in current sample
//   0x01C LAST_ADDRESS  last accepted input word address
//   0x020 LAST_DATA     last accepted 32-bit input word
//   0x024 INPUT_XOR     XOR checksum of accepted input words
//   0x028 INPUT_WORD0   word 0 observed at the LSTM input BRAM write port
//   0x02C BUILD_ID      debug register-map identifier
//   0x030..0x044        six FP16 logits, one per 32-bit register
//   0x400..0xCFF        write-only input window
//
// Each input word contains two consecutive FP16 values:
//   bits [15:0]  = input[2*word_index]
//   bits [31:16] = input[2*word_index + 1]
// A complete sample is 128 timesteps x 9 channels = 1152 FP16 values.
module lstm_axi_lite_regs #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 12
) (
    input  wire                              S_AXI_ACLK,
    input  wire                              S_AXI_ARESETN,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     S_AXI_AWADDR,
    input  wire [2:0]                        S_AXI_AWPROT,
    input  wire                              S_AXI_AWVALID,
    output wire                              S_AXI_AWREADY,
    input  wire [C_S_AXI_DATA_WIDTH-1:0]     S_AXI_WDATA,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0] S_AXI_WSTRB,
    input  wire                              S_AXI_WVALID,
    output wire                              S_AXI_WREADY,
    output wire [1:0]                        S_AXI_BRESP,
    output wire                              S_AXI_BVALID,
    input  wire                              S_AXI_BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     S_AXI_ARADDR,
    input  wire [2:0]                        S_AXI_ARPROT,
    input  wire                              S_AXI_ARVALID,
    output wire                              S_AXI_ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1:0]     S_AXI_RDATA,
    output wire [1:0]                        S_AXI_RRESP,
    output wire                              S_AXI_RVALID,
    input  wire                              S_AXI_RREADY,

    output reg                               core_start,
    input  wire                              core_busy,
    input  wire                              core_done,
    input  wire [2:0]                        core_prediction,
    input  wire [31:0]                       core_cycle_count,
    input  wire [31:0]                       core_debug_input_word0,
    input  wire [95:0]                       core_debug_logits,
    input  wire [383:0]                      core_debug_stages,

    output reg                               input_write_enable,
    output reg [9:0]                         input_write_word_addr,
    output reg [31:0]                        input_write_data,
    output reg [3:0]                         input_write_strb
);

    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_CONTROL =
        {{(C_S_AXI_ADDR_WIDTH-1){1'b0}}, 1'b0};
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_STATUS =
        {{(C_S_AXI_ADDR_WIDTH-3){1'b0}}, 3'h4};
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_PREDICTION =
        {{(C_S_AXI_ADDR_WIDTH-4){1'b0}}, 4'h8};
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_CYCLES =
        {{(C_S_AXI_ADDR_WIDTH-4){1'b0}}, 4'hC};
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_VERSION =
        {{(C_S_AXI_ADDR_WIDTH-5){1'b0}}, 5'h10};
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_INPUT_WORDS =
        {{(C_S_AXI_ADDR_WIDTH-5){1'b0}}, 5'h14};
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_INPUT_WRITES = 12'h018;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_LAST_ADDRESS = 12'h01C;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_LAST_DATA = 12'h020;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_INPUT_XOR = 12'h024;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_INPUT_WORD0 = 12'h028;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_BUILD_ID = 12'h02C;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_LOGIT0 = 12'h030;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_LOGIT1 = 12'h034;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_LOGIT2 = 12'h038;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_LOGIT3 = 12'h03C;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_LOGIT4 = 12'h040;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_LOGIT5 = 12'h044;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_DEBUG0 = 12'h048;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_DEBUG1 = 12'h04C;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_DEBUG2 = 12'h050;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_DEBUG3 = 12'h054;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_DEBUG4 = 12'h058;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_DEBUG5 = 12'h05C;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_DEBUG6 = 12'h060;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_DEBUG7 = 12'h064;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_DEBUG8 = 12'h068;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_DEBUG9 = 12'h06C;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_DEBUG10 = 12'h070;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] ADDR_DEBUG11 = 12'h074;
    localparam [C_S_AXI_ADDR_WIDTH-1:0] INPUT_BASE =
        {{(C_S_AXI_ADDR_WIDTH-11){1'b0}}, 11'h400};
    localparam [C_S_AXI_ADDR_WIDTH-1:0] INPUT_LAST =
        {{(C_S_AXI_ADDR_WIDTH-12){1'b0}}, 12'hCFC};

    reg [C_S_AXI_ADDR_WIDTH-1:0] awaddr_hold;
    reg                          awaddr_valid;
    reg [31:0]                   wdata_hold;
    reg [3:0]                    wstrb_hold;
    reg                          wdata_valid;
    reg [1:0]                    axi_bresp;
    reg                          axi_bvalid;
    reg [31:0]                   axi_rdata;
    reg [1:0]                    axi_rresp;
    reg                          axi_rvalid;

    reg                          done_latched;
    reg [2:0]                    prediction_latched;
    reg [31:0]                   cycle_count_latched;
    reg [9:0]                    input_write_count;
    reg [9:0]                    last_input_address;
    reg [31:0]                   last_input_data;
    reg [31:0]                   input_xor_checksum;

    wire aw_accept = S_AXI_AWVALID && S_AXI_AWREADY;
    wire w_accept  = S_AXI_WVALID && S_AXI_WREADY;
    wire ar_accept = S_AXI_ARVALID && S_AXI_ARREADY;

    wire have_awaddr = awaddr_valid || aw_accept;
    wire have_wdata  = wdata_valid || w_accept;
    wire write_commit = !axi_bvalid && have_awaddr && have_wdata;

    wire [C_S_AXI_ADDR_WIDTH-1:0] commit_addr =
        awaddr_valid ? awaddr_hold : S_AXI_AWADDR;
    wire [31:0] commit_data =
        wdata_valid ? wdata_hold : S_AXI_WDATA;
    wire [3:0] commit_strb =
        wdata_valid ? wstrb_hold : S_AXI_WSTRB;

    wire commit_is_control = (commit_addr == ADDR_CONTROL);
    wire commit_is_input =
        (commit_addr >= INPUT_BASE) && (commit_addr <= INPUT_LAST);

    assign S_AXI_AWREADY = !awaddr_valid && !axi_bvalid;
    assign S_AXI_WREADY  = !wdata_valid && !axi_bvalid;
    assign S_AXI_BRESP   = axi_bresp;
    assign S_AXI_BVALID  = axi_bvalid;
    assign S_AXI_ARREADY = !axi_rvalid;
    assign S_AXI_RDATA   = axi_rdata;
    assign S_AXI_RRESP   = axi_rresp;
    assign S_AXI_RVALID  = axi_rvalid;

    // AW and W are captured independently, as required by AXI4-Lite.
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            awaddr_hold <= {C_S_AXI_ADDR_WIDTH{1'b0}};
            awaddr_valid <= 1'b0;
            wdata_hold <= 32'd0;
            wstrb_hold <= 4'd0;
            wdata_valid <= 1'b0;
            axi_bresp <= 2'b00;
            axi_bvalid <= 1'b0;
        end else begin
            if (aw_accept) begin
                awaddr_hold <= S_AXI_AWADDR;
                awaddr_valid <= 1'b1;
            end

            if (w_accept) begin
                wdata_hold <= S_AXI_WDATA;
                wstrb_hold <= S_AXI_WSTRB;
                wdata_valid <= 1'b1;
            end

            if (write_commit) begin
                awaddr_valid <= 1'b0;
                wdata_valid <= 1'b0;
                axi_bvalid <= 1'b1;
                if (commit_is_control)
                    axi_bresp <= 2'b00;
                else if (commit_is_input && !core_busy)
                    axi_bresp <= 2'b00;
                else
                    axi_bresp <= 2'b10;
            end else if (axi_bvalid && S_AXI_BREADY) begin
                axi_bvalid <= 1'b0;
            end
        end
    end

    // Control pulses, completion latches, and the input-memory write port.
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            core_start <= 1'b0;
            done_latched <= 1'b0;
            prediction_latched <= 3'd0;
            cycle_count_latched <= 32'd0;
            input_write_enable <= 1'b0;
            input_write_word_addr <= 10'd0;
            input_write_data <= 32'd0;
            input_write_strb <= 4'd0;
            input_write_count <= 10'd0;
            last_input_address <= 10'd0;
            last_input_data <= 32'd0;
            input_xor_checksum <= 32'd0;
        end else begin
            core_start <= 1'b0;
            input_write_enable <= 1'b0;

            if (core_done) begin
                done_latched <= 1'b1;
                prediction_latched <= core_prediction;
                cycle_count_latched <= core_cycle_count;
            end

            if (write_commit && commit_is_control && commit_strb[0]) begin
                if (commit_data[0] && !core_busy) begin
                    core_start <= 1'b1;
                    done_latched <= 1'b0;
                end
                if (commit_data[1])
                    done_latched <= 1'b0;
            end

            if (write_commit && commit_is_input && !core_busy) begin
                input_write_enable <= 1'b1;
                input_write_word_addr <=
                    (commit_addr - INPUT_BASE) >> 2;
                input_write_data <= commit_data;
                input_write_strb <= commit_strb;

                // Address zero starts a new sample, so the count/checksum do
                // not depend on an additional control transaction.
                if (((commit_addr - INPUT_BASE) >> 2) == 0) begin
                    input_write_count <= 10'd1;
                    input_xor_checksum <= commit_data;
                end else begin
                    input_write_count <= input_write_count + 1'b1;
                    input_xor_checksum <= input_xor_checksum ^ commit_data;
                end
                last_input_address <= (commit_addr - INPUT_BASE) >> 2;
                last_input_data <= commit_data;
            end
        end
    end

    // Read channel. The input window is intentionally write-only.
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_rdata <= 32'd0;
            axi_rresp <= 2'b00;
            axi_rvalid <= 1'b0;
        end else begin
            if (ar_accept) begin
                axi_rvalid <= 1'b1;
                axi_rresp <= 2'b00;
                case (S_AXI_ARADDR)
                    ADDR_CONTROL:
                        axi_rdata <= 32'd0;
                    ADDR_STATUS:
                        axi_rdata <= {
                            29'd0,
                            !core_busy,
                            done_latched,
                            core_busy
                        };
                    ADDR_PREDICTION:
                        axi_rdata <= {29'd0, prediction_latched};
                    ADDR_CYCLES:
                        axi_rdata <= cycle_count_latched;
                    ADDR_VERSION:
                        axi_rdata <= 32'h0002_0000;
                    ADDR_INPUT_WORDS:
                        axi_rdata <= 32'd576;
                    ADDR_INPUT_WRITES:
                        axi_rdata <= {22'd0, input_write_count};
                    ADDR_LAST_ADDRESS:
                        axi_rdata <= {22'd0, last_input_address};
                    ADDR_LAST_DATA:
                        axi_rdata <= last_input_data;
                    ADDR_INPUT_XOR:
                        axi_rdata <= input_xor_checksum;
                    ADDR_INPUT_WORD0:
                        axi_rdata <= core_debug_input_word0;
                    ADDR_BUILD_ID:
                        axi_rdata <= 32'hD009_0002;
                    ADDR_LOGIT0:
                        axi_rdata <= {16'd0, core_debug_logits[15:0]};
                    ADDR_LOGIT1:
                        axi_rdata <= {16'd0, core_debug_logits[31:16]};
                    ADDR_LOGIT2:
                        axi_rdata <= {16'd0, core_debug_logits[47:32]};
                    ADDR_LOGIT3:
                        axi_rdata <= {16'd0, core_debug_logits[63:48]};
                    ADDR_LOGIT4:
                        axi_rdata <= {16'd0, core_debug_logits[79:64]};
                    ADDR_LOGIT5:
                        axi_rdata <= {16'd0, core_debug_logits[95:80]};
                    ADDR_DEBUG0:
                        axi_rdata <= core_debug_stages[31:0];
                    ADDR_DEBUG1:
                        axi_rdata <= core_debug_stages[63:32];
                    ADDR_DEBUG2:
                        axi_rdata <= core_debug_stages[95:64];
                    ADDR_DEBUG3:
                        axi_rdata <= core_debug_stages[127:96];
                    ADDR_DEBUG4:
                        axi_rdata <= core_debug_stages[159:128];
                    ADDR_DEBUG5:
                        axi_rdata <= core_debug_stages[191:160];
                    ADDR_DEBUG6:
                        axi_rdata <= core_debug_stages[223:192];
                    ADDR_DEBUG7:
                        axi_rdata <= core_debug_stages[255:224];
                    ADDR_DEBUG8:
                        axi_rdata <= core_debug_stages[287:256];
                    ADDR_DEBUG9:
                        axi_rdata <= core_debug_stages[319:288];
                    ADDR_DEBUG10:
                        axi_rdata <= core_debug_stages[351:320];
                    ADDR_DEBUG11:
                        axi_rdata <= core_debug_stages[383:352];
                    default: begin
                        axi_rdata <= 32'd0;
                        if ((S_AXI_ARADDR >= INPUT_BASE) &&
                            (S_AXI_ARADDR <= INPUT_LAST))
                            axi_rresp <= 2'b10;
                    end
                endcase
            end else if (axi_rvalid && S_AXI_RREADY) begin
                axi_rvalid <= 1'b0;
            end
        end
    end

    wire unused_prot = ^{
        S_AXI_AWPROT,
        S_AXI_ARPROT,
        awaddr_hold,
        wdata_hold,
        wstrb_hold
    };

endmodule
