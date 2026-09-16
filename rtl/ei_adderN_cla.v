`timescale 1ns/1ps

// Drop-in area implementation for the FP16 adder's integer additions.
// On Virtex-7, this maps to the dedicated CARRY4 chain instead of a
// structural LUT-based CLA tree.  The arithmetic result is unchanged.
module ei_adderN_cla #(
    parameter WIDTH = 16
) (
    input  [WIDTH-1:0] a,
    input  [WIDTH-1:0] b,
    output [WIDTH-1:0] sum,
    output             cout
);
    (* use_dsp = "no" *) wire [WIDTH:0] full_sum =
        {1'b0, a} + {1'b0, b};

    assign sum  = full_sum[WIDTH-1:0];
    assign cout = full_sum[WIDTH];
endmodule
