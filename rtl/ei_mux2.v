`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/10/2025 10:12:11 AM
// Design Name: 
// Module Name: ei_mux2
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module ei_mux2 (
    input  d0,
    input  d1,
    input  s,
    output y
);
    assign y = (~s & d0) | (s & d1);
endmodule
