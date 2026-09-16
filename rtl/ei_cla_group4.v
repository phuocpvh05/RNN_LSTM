`timescale 1ns/1ps
// =====================================================
// 4-bit Carry-Lookahead Adder Group
// Tính sum và tín hiệu Group Propagate / Group Generate
// cho kiến trúc CLA phân cấp.
//
// Delay: ~3 gate levels (vs ~8 cho 4-bit ripple-carry)
// =====================================================
module ei_cla_group4 (
    input  [3:0] a,
    input  [3:0] b,
    input        cin,
    output [3:0] sum,
    output       group_P,   // group propagate
    output       group_G    // group generate
);
    // Bit-level generate và propagate
    wire [3:0] g = a & b;     // g_i = 1 khi bit i tạo carry
    wire [3:0] p = a ^ b;     // p_i = 1 khi bit i truyền carry

    // CLA carry equations — tất cả carry tính song song từ cin
    wire c1 = g[0] | (p[0] & cin);
    wire c2 = g[1] | (p[1] & g[0]) | (p[1] & p[0] & cin);
    wire c3 = g[2] | (p[2] & g[1]) | (p[2] & p[1] & g[0])
            | (p[2] & p[1] & p[0] & cin);

    // Sum = propagate XOR carry-in tại mỗi vị trí bit
    assign sum[0] = p[0] ^ cin;
    assign sum[1] = p[1] ^ c1;
    assign sum[2] = p[2] ^ c2;
    assign sum[3] = p[3] ^ c3;

    // Group propagate: carry truyền qua toàn bộ 4 bit
    assign group_P = p[3] & p[2] & p[1] & p[0];

    // Group generate: nhóm tạo carry không phụ thuộc cin
    assign group_G = g[3] | (p[3] & g[2]) | (p[3] & p[2] & g[1])
                   | (p[3] & p[2] & p[1] & g[0]);

endmodule
