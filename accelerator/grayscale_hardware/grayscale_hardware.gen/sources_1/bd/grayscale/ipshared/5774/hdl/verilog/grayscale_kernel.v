// ==============================================================
// Generated by Vitis HLS v2024.2
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps 

(* CORE_GENERATION_INFO="grayscale_kernel_grayscale_kernel,hls_ip_2024_2,{HLS_INPUT_TYPE=cxx,HLS_INPUT_FLOAT=0,HLS_INPUT_FIXED=0,HLS_INPUT_PART=xc7z020-clg400-1,HLS_INPUT_CLOCK=10.000000,HLS_INPUT_ARCH=others,HLS_SYN_CLOCK=7.256000,HLS_SYN_LAT=-1,HLS_SYN_TPT=none,HLS_SYN_MEM=0,HLS_SYN_DSP=0,HLS_SYN_FF=2161,HLS_SYN_LUT=3553,HLS_VERSION=2024_2}" *)

module grayscale_kernel (
        ap_clk,
        ap_rst_n,
        input_stream_TDATA,
        input_stream_TVALID,
        input_stream_TREADY,
        input_stream_TKEEP,
        input_stream_TSTRB,
        input_stream_TLAST,
        output_stream_TDATA,
        output_stream_TVALID,
        output_stream_TREADY,
        output_stream_TKEEP,
        output_stream_TSTRB,
        output_stream_TLAST,
        s_axi_control_AWVALID,
        s_axi_control_AWREADY,
        s_axi_control_AWADDR,
        s_axi_control_WVALID,
        s_axi_control_WREADY,
        s_axi_control_WDATA,
        s_axi_control_WSTRB,
        s_axi_control_ARVALID,
        s_axi_control_ARREADY,
        s_axi_control_ARADDR,
        s_axi_control_RVALID,
        s_axi_control_RREADY,
        s_axi_control_RDATA,
        s_axi_control_RRESP,
        s_axi_control_BVALID,
        s_axi_control_BREADY,
        s_axi_control_BRESP,
        interrupt
);

parameter    ap_ST_fsm_state1 = 7'd1;
parameter    ap_ST_fsm_state2 = 7'd2;
parameter    ap_ST_fsm_state3 = 7'd4;
parameter    ap_ST_fsm_state4 = 7'd8;
parameter    ap_ST_fsm_state5 = 7'd16;
parameter    ap_ST_fsm_state6 = 7'd32;
parameter    ap_ST_fsm_state7 = 7'd64;
parameter    C_S_AXI_CONTROL_DATA_WIDTH = 32;
parameter    C_S_AXI_CONTROL_ADDR_WIDTH = 5;
parameter    C_S_AXI_DATA_WIDTH = 32;

parameter C_S_AXI_CONTROL_WSTRB_WIDTH = (32 / 8);
parameter C_S_AXI_WSTRB_WIDTH = (32 / 8);

input   ap_clk;
input   ap_rst_n;
input  [31:0] input_stream_TDATA;
input   input_stream_TVALID;
output   input_stream_TREADY;
input  [3:0] input_stream_TKEEP;
input  [3:0] input_stream_TSTRB;
input  [0:0] input_stream_TLAST;
output  [31:0] output_stream_TDATA;
output   output_stream_TVALID;
input   output_stream_TREADY;
output  [3:0] output_stream_TKEEP;
output  [3:0] output_stream_TSTRB;
output  [0:0] output_stream_TLAST;
input   s_axi_control_AWVALID;
output   s_axi_control_AWREADY;
input  [C_S_AXI_CONTROL_ADDR_WIDTH - 1:0] s_axi_control_AWADDR;
input   s_axi_control_WVALID;
output   s_axi_control_WREADY;
input  [C_S_AXI_CONTROL_DATA_WIDTH - 1:0] s_axi_control_WDATA;
input  [C_S_AXI_CONTROL_WSTRB_WIDTH - 1:0] s_axi_control_WSTRB;
input   s_axi_control_ARVALID;
output   s_axi_control_ARREADY;
input  [C_S_AXI_CONTROL_ADDR_WIDTH - 1:0] s_axi_control_ARADDR;
output   s_axi_control_RVALID;
input   s_axi_control_RREADY;
output  [C_S_AXI_CONTROL_DATA_WIDTH - 1:0] s_axi_control_RDATA;
output  [1:0] s_axi_control_RRESP;
output   s_axi_control_BVALID;
input   s_axi_control_BREADY;
output  [1:0] s_axi_control_BRESP;
output   interrupt;

 reg    ap_rst_n_inv;
wire    ap_start;
reg    ap_done;
reg    ap_idle;
(* fsm_encoding = "none" *) reg   [6:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg    ap_ready;
wire   [31:0] width;
reg   [31:0] width_0_data_reg;
reg    width_0_vld_reg;
reg    width_0_ack_out;
wire   [31:0] height;
reg   [31:0] height_0_data_reg;
reg    height_0_vld_reg;
reg    height_0_ack_out;
reg   [31:0] height_read_reg_158;
wire    ap_CS_fsm_state2;
reg   [31:0] width_read_reg_163;
wire   [30:0] smax_fu_118_p3;
reg   [30:0] smax_reg_169;
wire   [30:0] smax1_fu_132_p3;
reg   [30:0] smax1_reg_174;
wire    ap_CS_fsm_state3;
wire   [31:0] sub_fu_148_p2;
reg   [31:0] sub_reg_189;
wire    ap_CS_fsm_state4;
wire   [31:0] sub22_fu_153_p2;
reg   [31:0] sub22_reg_194;
wire   [61:0] grp_fu_100_p2;
reg   [61:0] mul_ln12_reg_199;
wire    grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_ap_start;
wire    grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_ap_done;
wire    grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_ap_idle;
wire    grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_ap_ready;
wire    grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TREADY;
wire    grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_input_stream_TREADY;
wire   [31:0] grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TDATA;
wire    grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TVALID;
wire   [3:0] grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TKEEP;
wire   [3:0] grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TSTRB;
wire   [0:0] grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TLAST;
reg    grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_ap_start_reg;
reg   [6:0] ap_NS_fsm;
wire    ap_NS_fsm_state5;
wire    ap_CS_fsm_state6;
reg   [31:0] output_stream_TDATA_reg;
reg   [3:0] output_stream_TKEEP_reg;
reg   [3:0] output_stream_TSTRB_reg;
reg   [0:0] output_stream_TLAST_reg;
wire   [30:0] grp_fu_100_p0;
wire   [30:0] grp_fu_100_p1;
wire   [0:0] empty_fu_112_p2;
wire   [30:0] trunc_ln12_1_fu_108_p1;
wire   [0:0] empty_17_fu_126_p2;
wire   [30:0] trunc_ln12_fu_104_p1;
wire    ap_CS_fsm_state7;
wire    regslice_both_output_stream_V_data_V_U_apdone_blk;
reg    ap_ST_fsm_state1_blk;
wire    ap_ST_fsm_state2_blk;
wire    ap_ST_fsm_state3_blk;
wire    ap_ST_fsm_state4_blk;
wire    ap_ST_fsm_state5_blk;
reg    ap_ST_fsm_state6_blk;
reg    ap_ST_fsm_state7_blk;
wire    regslice_both_input_stream_V_data_V_U_apdone_blk;
wire   [31:0] input_stream_TDATA_int_regslice;
wire    input_stream_TVALID_int_regslice;
reg    input_stream_TREADY_int_regslice;
wire    regslice_both_input_stream_V_data_V_U_ack_in;
wire    regslice_both_input_stream_V_keep_V_U_apdone_blk;
wire   [3:0] input_stream_TKEEP_int_regslice;
wire    regslice_both_input_stream_V_keep_V_U_vld_out;
wire    regslice_both_input_stream_V_keep_V_U_ack_in;
wire    regslice_both_input_stream_V_strb_V_U_apdone_blk;
wire   [3:0] input_stream_TSTRB_int_regslice;
wire    regslice_both_input_stream_V_strb_V_U_vld_out;
wire    regslice_both_input_stream_V_strb_V_U_ack_in;
wire    regslice_both_input_stream_V_last_V_U_apdone_blk;
wire   [0:0] input_stream_TLAST_int_regslice;
wire    regslice_both_input_stream_V_last_V_U_vld_out;
wire    regslice_both_input_stream_V_last_V_U_ack_in;
reg   [31:0] output_stream_TDATA_int_regslice;
wire    output_stream_TVALID_int_regslice;
wire    output_stream_TREADY_int_regslice;
wire    regslice_both_output_stream_V_data_V_U_vld_out;
wire    regslice_both_output_stream_V_keep_V_U_apdone_blk;
reg   [3:0] output_stream_TKEEP_int_regslice;
wire    regslice_both_output_stream_V_keep_V_U_ack_in_dummy;
wire    regslice_both_output_stream_V_keep_V_U_vld_out;
wire    regslice_both_output_stream_V_strb_V_U_apdone_blk;
reg   [3:0] output_stream_TSTRB_int_regslice;
wire    regslice_both_output_stream_V_strb_V_U_ack_in_dummy;
wire    regslice_both_output_stream_V_strb_V_U_vld_out;
wire    regslice_both_output_stream_V_last_V_U_apdone_blk;
reg   [0:0] output_stream_TLAST_int_regslice;
wire    regslice_both_output_stream_V_last_V_U_ack_in_dummy;
wire    regslice_both_output_stream_V_last_V_U_vld_out;
wire   [61:0] grp_fu_100_p00;
wire   [61:0] grp_fu_100_p10;
wire    ap_ce_reg;

// power-on initialization
initial begin
#0 ap_CS_fsm = 7'd1;
#0 width_0_data_reg = 32'd0;
#0 width_0_vld_reg = 1'b0;
#0 height_0_data_reg = 32'd0;
#0 height_0_vld_reg = 1'b0;
#0 grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_ap_start_reg = 1'b0;
end

grayscale_kernel_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2 grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .ap_start(grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_ap_start),
    .ap_done(grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_ap_done),
    .ap_idle(grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_ap_idle),
    .ap_ready(grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_ap_ready),
    .input_stream_TVALID(input_stream_TVALID_int_regslice),
    .output_stream_TREADY(grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TREADY),
    .mul_ln12(mul_ln12_reg_199),
    .width(width_read_reg_163),
    .sub(sub_reg_189),
    .input_stream_TDATA(input_stream_TDATA_int_regslice),
    .input_stream_TREADY(grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_input_stream_TREADY),
    .input_stream_TKEEP(input_stream_TKEEP_int_regslice),
    .input_stream_TSTRB(input_stream_TSTRB_int_regslice),
    .input_stream_TLAST(input_stream_TLAST_int_regslice),
    .sub22(sub22_reg_194),
    .output_stream_TDATA(grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TDATA),
    .output_stream_TVALID(grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TVALID),
    .output_stream_TKEEP(grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TKEEP),
    .output_stream_TSTRB(grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TSTRB),
    .output_stream_TLAST(grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TLAST)
);

grayscale_kernel_control_s_axi #(
    .C_S_AXI_ADDR_WIDTH( C_S_AXI_CONTROL_ADDR_WIDTH ),
    .C_S_AXI_DATA_WIDTH( C_S_AXI_CONTROL_DATA_WIDTH ))
control_s_axi_U(
    .AWVALID(s_axi_control_AWVALID),
    .AWREADY(s_axi_control_AWREADY),
    .AWADDR(s_axi_control_AWADDR),
    .WVALID(s_axi_control_WVALID),
    .WREADY(s_axi_control_WREADY),
    .WDATA(s_axi_control_WDATA),
    .WSTRB(s_axi_control_WSTRB),
    .ARVALID(s_axi_control_ARVALID),
    .ARREADY(s_axi_control_ARREADY),
    .ARADDR(s_axi_control_ARADDR),
    .RVALID(s_axi_control_RVALID),
    .RREADY(s_axi_control_RREADY),
    .RDATA(s_axi_control_RDATA),
    .RRESP(s_axi_control_RRESP),
    .BVALID(s_axi_control_BVALID),
    .BREADY(s_axi_control_BREADY),
    .BRESP(s_axi_control_BRESP),
    .ACLK(ap_clk),
    .ARESET(ap_rst_n_inv),
    .ACLK_EN(1'b1),
    .width(width),
    .height(height),
    .ap_start(ap_start),
    .interrupt(interrupt),
    .ap_ready(ap_ready),
    .ap_done(ap_done),
    .ap_idle(ap_idle)
);

grayscale_kernel_mul_31ns_31ns_62_2_1 #(
    .ID( 1 ),
    .NUM_STAGE( 2 ),
    .din0_WIDTH( 31 ),
    .din1_WIDTH( 31 ),
    .dout_WIDTH( 62 ))
mul_31ns_31ns_62_2_1_U24(
    .clk(ap_clk),
    .reset(ap_rst_n_inv),
    .din0(grp_fu_100_p0),
    .din1(grp_fu_100_p1),
    .ce(1'b1),
    .dout(grp_fu_100_p2)
);

grayscale_kernel_regslice_both #(
    .DataWidth( 32 ))
regslice_both_input_stream_V_data_V_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(input_stream_TDATA),
    .vld_in(input_stream_TVALID),
    .ack_in(regslice_both_input_stream_V_data_V_U_ack_in),
    .data_out(input_stream_TDATA_int_regslice),
    .vld_out(input_stream_TVALID_int_regslice),
    .ack_out(input_stream_TREADY_int_regslice),
    .apdone_blk(regslice_both_input_stream_V_data_V_U_apdone_blk)
);

grayscale_kernel_regslice_both #(
    .DataWidth( 4 ))
regslice_both_input_stream_V_keep_V_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(input_stream_TKEEP),
    .vld_in(input_stream_TVALID),
    .ack_in(regslice_both_input_stream_V_keep_V_U_ack_in),
    .data_out(input_stream_TKEEP_int_regslice),
    .vld_out(regslice_both_input_stream_V_keep_V_U_vld_out),
    .ack_out(input_stream_TREADY_int_regslice),
    .apdone_blk(regslice_both_input_stream_V_keep_V_U_apdone_blk)
);

grayscale_kernel_regslice_both #(
    .DataWidth( 4 ))
regslice_both_input_stream_V_strb_V_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(input_stream_TSTRB),
    .vld_in(input_stream_TVALID),
    .ack_in(regslice_both_input_stream_V_strb_V_U_ack_in),
    .data_out(input_stream_TSTRB_int_regslice),
    .vld_out(regslice_both_input_stream_V_strb_V_U_vld_out),
    .ack_out(input_stream_TREADY_int_regslice),
    .apdone_blk(regslice_both_input_stream_V_strb_V_U_apdone_blk)
);

grayscale_kernel_regslice_both #(
    .DataWidth( 1 ))
regslice_both_input_stream_V_last_V_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(input_stream_TLAST),
    .vld_in(input_stream_TVALID),
    .ack_in(regslice_both_input_stream_V_last_V_U_ack_in),
    .data_out(input_stream_TLAST_int_regslice),
    .vld_out(regslice_both_input_stream_V_last_V_U_vld_out),
    .ack_out(input_stream_TREADY_int_regslice),
    .apdone_blk(regslice_both_input_stream_V_last_V_U_apdone_blk)
);

grayscale_kernel_regslice_both #(
    .DataWidth( 32 ))
regslice_both_output_stream_V_data_V_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(output_stream_TDATA_int_regslice),
    .vld_in(grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TVALID),
    .ack_in(output_stream_TREADY_int_regslice),
    .data_out(output_stream_TDATA),
    .vld_out(regslice_both_output_stream_V_data_V_U_vld_out),
    .ack_out(output_stream_TREADY),
    .apdone_blk(regslice_both_output_stream_V_data_V_U_apdone_blk)
);

grayscale_kernel_regslice_both #(
    .DataWidth( 4 ))
regslice_both_output_stream_V_keep_V_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(output_stream_TKEEP_int_regslice),
    .vld_in(grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TVALID),
    .ack_in(regslice_both_output_stream_V_keep_V_U_ack_in_dummy),
    .data_out(output_stream_TKEEP),
    .vld_out(regslice_both_output_stream_V_keep_V_U_vld_out),
    .ack_out(output_stream_TREADY),
    .apdone_blk(regslice_both_output_stream_V_keep_V_U_apdone_blk)
);

grayscale_kernel_regslice_both #(
    .DataWidth( 4 ))
regslice_both_output_stream_V_strb_V_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(output_stream_TSTRB_int_regslice),
    .vld_in(grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TVALID),
    .ack_in(regslice_both_output_stream_V_strb_V_U_ack_in_dummy),
    .data_out(output_stream_TSTRB),
    .vld_out(regslice_both_output_stream_V_strb_V_U_vld_out),
    .ack_out(output_stream_TREADY),
    .apdone_blk(regslice_both_output_stream_V_strb_V_U_apdone_blk)
);

grayscale_kernel_regslice_both #(
    .DataWidth( 1 ))
regslice_both_output_stream_V_last_V_U(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst_n_inv),
    .data_in(output_stream_TLAST_int_regslice),
    .vld_in(grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TVALID),
    .ack_in(regslice_both_output_stream_V_last_V_U_ack_in_dummy),
    .data_out(output_stream_TLAST),
    .vld_out(regslice_both_output_stream_V_last_V_U_vld_out),
    .ack_out(output_stream_TREADY),
    .apdone_blk(regslice_both_output_stream_V_last_V_U_apdone_blk)
);

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst_n_inv == 1'b1) begin
        grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_ap_start_reg <= 1'b0;
    end else begin
        if (((1'b1 == ap_NS_fsm_state5) & (1'b1 == ap_CS_fsm_state4))) begin
            grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_ap_start_reg <= 1'b1;
        end else if ((grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_ap_ready == 1'b1)) begin
            grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((~((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b0)) & (height_0_ack_out == 1'b1) & (1'b1 == 1'b1) & (height_0_vld_reg == 1'b1)) | (~((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b0)) & (1'b1 == 1'b1) & (height_0_vld_reg == 1'b0)))) begin
        height_0_data_reg <= height;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        height_read_reg_158 <= height_0_data_reg;
        smax1_reg_174 <= smax1_fu_132_p3;
        smax_reg_169 <= smax_fu_118_p3;
        width_read_reg_163 <= width_0_data_reg;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        mul_ln12_reg_199 <= grp_fu_100_p2;
        sub22_reg_194 <= sub22_fu_153_p2;
        sub_reg_189 <= sub_fu_148_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state6) & (grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TVALID == 1'b1))) begin
        output_stream_TDATA_reg <= grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TDATA;
        output_stream_TKEEP_reg <= grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TKEEP;
        output_stream_TLAST_reg <= grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TLAST;
        output_stream_TSTRB_reg <= grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TSTRB;
    end
end

always @ (posedge ap_clk) begin
    if (((~((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b0)) & (width_0_ack_out == 1'b1) & (1'b1 == 1'b1) & (width_0_vld_reg == 1'b1)) | (~((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b0)) & (1'b1 == 1'b1) & (width_0_vld_reg == 1'b0)))) begin
        width_0_data_reg <= width;
    end
end

always @ (*) begin
    if ((ap_start == 1'b0)) begin
        ap_ST_fsm_state1_blk = 1'b1;
    end else begin
        ap_ST_fsm_state1_blk = 1'b0;
    end
end

assign ap_ST_fsm_state2_blk = 1'b0;

assign ap_ST_fsm_state3_blk = 1'b0;

assign ap_ST_fsm_state4_blk = 1'b0;

assign ap_ST_fsm_state5_blk = 1'b0;

always @ (*) begin
    if ((grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_ap_done == 1'b0)) begin
        ap_ST_fsm_state6_blk = 1'b1;
    end else begin
        ap_ST_fsm_state6_blk = 1'b0;
    end
end

always @ (*) begin
    if ((regslice_both_output_stream_V_data_V_U_apdone_blk == 1'b1)) begin
        ap_ST_fsm_state7_blk = 1'b1;
    end else begin
        ap_ST_fsm_state7_blk = 1'b0;
    end
end

always @ (*) begin
    if (((regslice_both_output_stream_V_data_V_U_apdone_blk == 1'b0) & (1'b1 == ap_CS_fsm_state7))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((regslice_both_output_stream_V_data_V_U_apdone_blk == 1'b0) & (1'b1 == ap_CS_fsm_state7))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state2) | ((regslice_both_output_stream_V_data_V_U_apdone_blk == 1'b0) & (1'b1 == ap_CS_fsm_state7)))) begin
        height_0_ack_out = 1'b1;
    end else begin
        height_0_ack_out = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state6)) begin
        input_stream_TREADY_int_regslice = grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_input_stream_TREADY;
    end else begin
        input_stream_TREADY_int_regslice = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state6) & (grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TVALID == 1'b1))) begin
        output_stream_TDATA_int_regslice = grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TDATA;
    end else begin
        output_stream_TDATA_int_regslice = output_stream_TDATA_reg;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state6) & (grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TVALID == 1'b1))) begin
        output_stream_TKEEP_int_regslice = grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TKEEP;
    end else begin
        output_stream_TKEEP_int_regslice = output_stream_TKEEP_reg;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state6) & (grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TVALID == 1'b1))) begin
        output_stream_TLAST_int_regslice = grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TLAST;
    end else begin
        output_stream_TLAST_int_regslice = output_stream_TLAST_reg;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state6) & (grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TVALID == 1'b1))) begin
        output_stream_TSTRB_int_regslice = grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TSTRB;
    end else begin
        output_stream_TSTRB_int_regslice = output_stream_TSTRB_reg;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state2) | ((regslice_both_output_stream_V_data_V_U_apdone_blk == 1'b0) & (1'b1 == ap_CS_fsm_state7)))) begin
        width_0_ack_out = 1'b1;
    end else begin
        width_0_ack_out = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            ap_NS_fsm = ap_ST_fsm_state3;
        end
        ap_ST_fsm_state3 : begin
            ap_NS_fsm = ap_ST_fsm_state4;
        end
        ap_ST_fsm_state4 : begin
            ap_NS_fsm = ap_ST_fsm_state5;
        end
        ap_ST_fsm_state5 : begin
            ap_NS_fsm = ap_ST_fsm_state6;
        end
        ap_ST_fsm_state6 : begin
            if (((1'b1 == ap_CS_fsm_state6) & (grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_ap_done == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state7;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state6;
            end
        end
        ap_ST_fsm_state7 : begin
            if (((regslice_both_output_stream_V_data_V_U_apdone_blk == 1'b0) & (1'b1 == ap_CS_fsm_state7))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state7;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state6 = ap_CS_fsm[32'd5];

assign ap_CS_fsm_state7 = ap_CS_fsm[32'd6];

assign ap_NS_fsm_state5 = ap_NS_fsm[32'd4];

always @ (*) begin
    ap_rst_n_inv = ~ap_rst_n;
end

assign empty_17_fu_126_p2 = (($signed(width_0_data_reg) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign empty_fu_112_p2 = (($signed(height_0_data_reg) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign grp_fu_100_p0 = grp_fu_100_p00;

assign grp_fu_100_p00 = smax_reg_169;

assign grp_fu_100_p1 = grp_fu_100_p10;

assign grp_fu_100_p10 = smax1_reg_174;

assign grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_ap_start = grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_ap_start_reg;

assign grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TREADY = (output_stream_TREADY_int_regslice & ap_CS_fsm_state6);

assign input_stream_TREADY = regslice_both_input_stream_V_data_V_U_ack_in;

assign output_stream_TVALID = regslice_both_output_stream_V_data_V_U_vld_out;

assign output_stream_TVALID_int_regslice = grp_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2_fu_76_output_stream_TVALID;

assign smax1_fu_132_p3 = ((empty_17_fu_126_p2[0:0] == 1'b1) ? trunc_ln12_fu_104_p1 : 31'd0);

assign smax_fu_118_p3 = ((empty_fu_112_p2[0:0] == 1'b1) ? trunc_ln12_1_fu_108_p1 : 31'd0);

assign sub22_fu_153_p2 = ($signed(width_read_reg_163) + $signed(32'd4294967295));

assign sub_fu_148_p2 = ($signed(height_read_reg_158) + $signed(32'd4294967295));

assign trunc_ln12_1_fu_108_p1 = height_0_data_reg[30:0];

assign trunc_ln12_fu_104_p1 = width_0_data_reg[30:0];


reg find_kernel_block = 0;
// synthesis translate_off
`include "grayscale_kernel_hls_deadlock_kernel_monitor_top.vh"
// synthesis translate_on

endmodule //grayscale_kernel

