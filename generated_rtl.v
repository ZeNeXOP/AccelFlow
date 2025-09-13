`include "mac_unit_template.v"
// Systolic Array Template for AI Accelerator
module systolic_array #(
    parameter DATA_WIDTH = 8,
    parameter ARRAY_SIZE = 16,
    parameter ACCUM_WIDTH = 32
) (
    input clk,
    input rst,
    input enable,
    
    // Weight inputs (preloaded)
    input signed [DATA_WIDTH-1:0] weights [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1],
    
    // Activation inputs (streaming)
    input signed [DATA_WIDTH-1:0] activations [0:ARRAY_SIZE-1],
    
    // Partial sum outputs
    output signed [ACCUM_WIDTH-1:0] partial_sums [0:ARRAY_SIZE-1]
);

    // Internal registers for systolic dataflow
    reg signed [DATA_WIDTH-1:0] act_reg [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    reg signed [ACCUM_WIDTH-1:0] psum_reg [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

    // Systolic array computation
    genvar i, j;
    generate
        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin : row
            for (j = 0; j < ARRAY_SIZE; j = j + 1) begin : col
                always @(posedge clk) begin
                    if (rst) begin
                        act_reg[i][j] <= 0;
                        psum_reg[i][j] <= 0;
                    end else if (enable) begin
                        // Shift activations horizontally
                        if (j == 0)
                            act_reg[i][j] <= activations[i];
                        else
                            act_reg[i][j] <= act_reg[i][j-1];
                        
                        // MAC operation: multiply-accumulate
                        if (i == 0)
                            psum_reg[i][j] <= weights[i][j] * act_reg[i][j];
                        else
                            psum_reg[i][j] <= psum_reg[i-1][j] + (weights[i][j] * act_reg[i][j]);
                    end
                end
            end
        end
    endgenerate

    // Output assignment (bottom row partial sums)
    assign partial_sums = psum_reg[ARRAY_SIZE-1];

endmodule // Generated for GeneratedModel