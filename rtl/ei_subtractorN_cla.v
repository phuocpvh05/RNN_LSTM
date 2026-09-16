`timescale 1ns/1ps

// Drop-in CARRY4 implementation of a-b.  The extra carry bit is the
// no-borrow flag, so borrow remains bit-exact with the original CLA module.
module ei_subtractorN_cla #(
    parameter WIDTH = 6
) (
    input  [WIDTH-1:0] a,
    input  [WIDTH-1:0] b,
    output [WIDTH-1:0] diff,
    output             borrow
);
    (* use_dsp = "no" *) wire [WIDTH:0] full_sub =
        {1'b0, a} + {1'b0, ~b} + {{WIDTH{1'b0}}, 1'b1};

    assign diff   = full_sub[WIDTH-1:0];
    assign borrow = ~full_sub[WIDTH];
endmodule
