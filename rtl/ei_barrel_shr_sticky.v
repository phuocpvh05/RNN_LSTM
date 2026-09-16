`timescale 1ns/1ps
// =====================================================
// Barrel Shifter (Right Shift) with Sticky bit
// Cấu trúc logarithmic (multiplexer tree).
// Đảm bảo delay nhất quán O(log N) thay vì phụ thuộc synthesis.
// =====================================================
module ei_barrel_shr_sticky #(
    parameter integer WIDTH   = 14,
    parameter integer SHAMT_W = 6
)(
    input  wire [WIDTH-1:0]   d_in,
    input  wire [SHAMT_W-1:0] shamt,
    output wire [WIDTH-1:0]   d_out,
    output wire               sticky
);
    wire [WIDTH-1:0] stage [SHAMT_W:0];
    wire [WIDTH-1:0] sticky_stage [SHAMT_W:0];

    assign stage[0] = d_in;
    assign sticky_stage[0] = {WIDTH{1'b0}};

    genvar i;
    generate
        for (i = 0; i < SHAMT_W; i = i + 1) begin : SHIFT_STAGES
            localparam SHIFT_VAL = 1 << i;
            
            // Nếu bit shamt[i] = 1, dịch phải SHIFT_VAL bit
            // Nếu không, giữ nguyên
            
            wire [WIDTH-1:0] next_stage;
            wire [WIDTH-1:0] next_sticky;
            
            if (SHIFT_VAL < WIDTH) begin
                assign next_stage = shamt[i] ? { {SHIFT_VAL{1'b0}}, stage[i][WIDTH-1 : SHIFT_VAL] } : stage[i];
                // Sticky là OR của sticky cũ, cộng thêm các bit bị dịch ra (nếu có dịch)
                wire [SHIFT_VAL-1:0] shifted_out_bits = stage[i][SHIFT_VAL-1 : 0];
                wire new_sticky = |shifted_out_bits;
                assign next_sticky = shamt[i] ? (sticky_stage[i] | { {(WIDTH-1){1'b0}}, new_sticky }) : sticky_stage[i];
            end else begin
                // Nếu dịch xa hơn hoặc bằng WIDTH, output bằng 0, tất cả bit d_in trở thành sticky
                assign next_stage = shamt[i] ? {WIDTH{1'b0}} : stage[i];
                wire new_sticky = |stage[i];
                assign next_sticky = shamt[i] ? (sticky_stage[i] | { {(WIDTH-1){1'b0}}, new_sticky }) : sticky_stage[i];
            end
            
            assign stage[i+1] = next_stage;
            assign sticky_stage[i+1] = next_sticky;
        end
    endgenerate

    // Check trường hợp shamt >= WIDTH trực tiếp nếu cần thiết để đảm bảo 0 hoàn toàn
    wire shamt_ge_w = (shamt >= WIDTH[SHAMT_W-1:0]);
    assign d_out = shamt_ge_w ? {WIDTH{1'b0}} : stage[SHAMT_W];
    
    // Tổng sticky: bit cuối của quá trình + sticky khi shamt_ge_w
    assign sticky = shamt_ge_w ? (|d_in) : (|sticky_stage[SHAMT_W]);

endmodule
