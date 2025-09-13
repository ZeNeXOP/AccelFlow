  module buffer #(
      parameter DEPTH = 1024,
      parameter WIDTH = 8
  ) (
      input clk, we,
      input [log2(DEPTH)-1:0] addr,
      input [WIDTH-1:0] data_in,
      output [WIDTH-1:0] data_out
  );
      reg [WIDTH-1:0] mem [0:DEPTH-1];
      always @(posedge clk) begin
          if (we) mem[addr] <= data_in;
      end
      assign data_out = mem[addr];
  endmodule