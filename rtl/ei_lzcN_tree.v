`timescale 1ns/1ps
// =====================================================
// Leading Zero Counter — Tree Architecture
// Đếm số bit 0 liên tiếp từ MSB bằng chia đôi đệ quy.
// Delay: O(log2 N)  (vs O(N) cho phiên bản tuần tự)
//
// Ví dụ WIDTH=14:
//   x = 14'b00001xxxxxxxxx  →  count = 4
//   x = 14'b00000000000000  →  count = 14, all_zero = 1
// =====================================================
module ei_lzcN_tree #(
    parameter integer WIDTH   = 14,
    parameter integer COUNT_W = $clog2(WIDTH + 1)
) (
    input  wire [WIDTH-1:0]   x,
    output wire [COUNT_W-1:0] count,
    output wire               all_zero   // 1 nếu x == 0
);
    generate
        // ---- Base case: 1 bit ----
        if (WIDTH == 1) begin : BASE1
            assign count    = ~x[0];   // x=0 → count=1; x=1 → count=0
            assign all_zero = ~x[0];
        end

        // ---- Base case: 2 bit ----
        else if (WIDTH == 2) begin : BASE2
            // x=1x: count=0, x=01: count=1, x=00: count=2
            assign count[1] = ~x[1] & ~x[0];          // = 1 chỉ khi x=00 (count=2)
            assign count[0] = ~x[1] &  x[0];           // = 1 chỉ khi x=01 (count=1)
            assign all_zero = ~x[1] & ~x[0];
        end

        // ---- Recursive case: chia đôi ----
        else begin : RECURSE
            localparam LO_W  = WIDTH / 2;              // nửa thấp (LSBs)
            localparam HI_W  = WIDTH - LO_W;           // nửa cao (MSBs), ≥ LO_W
            localparam HI_CW = $clog2(HI_W + 1);
            localparam LO_CW = $clog2(LO_W + 1);

            wire [HI_CW-1:0] hi_count;
            wire              hi_all_zero;
            wire [LO_CW-1:0] lo_count;
            wire              lo_all_zero;

            // Nửa cao (MSBs) — kiểm tra trước cho leading zeros
            ei_lzcN_tree #(.WIDTH(HI_W)) u_hi (
                .x        (x[WIDTH-1 : LO_W]),
                .count    (hi_count),
                .all_zero (hi_all_zero)
            );

            // Nửa thấp (LSBs)
            ei_lzcN_tree #(.WIDTH(LO_W)) u_lo (
                .x        (x[LO_W-1 : 0]),
                .count    (lo_count),
                .all_zero (lo_all_zero)
            );

            assign all_zero = hi_all_zero & lo_all_zero;

            wire [COUNT_W-1:0] hi_ext;
            if (COUNT_W > HI_CW)
                assign hi_ext = {{(COUNT_W - HI_CW){1'b0}}, hi_count};
            else
                assign hi_ext = hi_count;

            wire [COUNT_W-1:0] lo_ext;
            if (COUNT_W > LO_CW)
                assign lo_ext = {{(COUNT_W - LO_CW){1'b0}}, lo_count};
            else
                assign lo_ext = lo_count;

            // Nếu nửa cao toàn 0: count = HI_W + lo_count
            // Ngược lại:           count = hi_count
            wire [COUNT_W-1:0] hi_w_val  = HI_W[COUNT_W-1:0];
            wire [COUNT_W-1:0] sum_count = hi_w_val + lo_ext;

            assign count = hi_all_zero ? sum_count : hi_ext;
        end
    endgenerate
endmodule
