// MUX 16-bit: chọn giữa d0 và d1 theo s
module ei_mux16 (
    input  [15:0] d0,
    input  [15:0] d1,
    input         s,
    output [15:0] y
);
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : GEN_MUX16
            ei_mux2 u_mux (
                .d0(d0[i]),
                .d1(d1[i]),
                .s (s),
                .y (y[i])
            );
        end
    endgenerate
endmodule
