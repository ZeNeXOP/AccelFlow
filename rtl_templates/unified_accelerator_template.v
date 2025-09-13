// unified_accelerator_template.v
`include "systolic_array_template.v"
`include "buffer_template.v"
`include "activation_relu_template.v"

module unified_accelerator #(
    parameter ARRAY_SIZE = 4,
    parameter BITWIDTH = 8,
    parameter USE_BUFFER = 1,  // 1 to enable buffer
    parameter USE_RELU = 1     // 1 to enable ReLU
) (
    input clk, rst,
    // ... ports ...
);

    wire [BITWIDTH-1:0] sys_out [0:ARRAY_SIZE-1];

    systolic_array #(ARRAY_SIZE, BITWIDTH) sys (
        .clk(clk), .rst(rst), /* ... */
        .outputs(sys_out)
    );

    generate
        if (USE_BUFFER) begin
            buffer #(/* params */) buf (/* ... */);
        end
        if (USE_RELU) begin
            relu #(BITWIDTH) act [0:ARRAY_SIZE-1] (/* ... */);
        end
    endgenerate

endmodule