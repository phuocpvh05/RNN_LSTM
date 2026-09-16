`timescale 1ns / 1ps
module regN #(
    parameter WIDTH = 16
)(
    input                  clk,
    input                  rst,   // reset đồng bộ, active-high
    input                  en,
    input      [WIDTH-1:0] d,
    output reg [WIDTH-1:0] q
);
    always @(posedge clk) begin
        if (rst)
            q <= {WIDTH{1'b0}};
        else if (en)
            q <= d;
    end
endmodule
