  module mac_unit #(
      parameter BITWIDTH = 8,
      parameter ACCUM_WIDTH = 32
  ) (
      input signed [BITWIDTH-1:0] a, b,
      input signed [ACCUM_WIDTH-1:0] accum_in,
      output signed [ACCUM_WIDTH-1:0] accum_out
  );
      wire signed [ACCUM_WIDTH-1:0] product = a * b;
      assign accum_out = accum_in + product;
  endmodule