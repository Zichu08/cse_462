// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2024.2 (win64) Build 5239630 Fri Nov 08 22:35:27 MST 2024
// Date        : Mon Mar 24 01:10:24 2025
// Host        : LAPTOP-D6BOEUU6 running 64-bit major release  (build 9200)
// Command     : write_verilog -force -mode funcsim -rename_top decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix -prefix
//               decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_ design_1_filter_kernel_0_2_sim_netlist.v
// Design      : design_1_filter_kernel_0_2
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xc7z020clg400-1
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

(* CHECK_LICENSE_TYPE = "design_1_filter_kernel_0_2,filter_kernel,{}" *) (* DowngradeIPIdentifiedWarnings = "yes" *) (* IP_DEFINITION_SOURCE = "HLS" *) 
(* X_CORE_INFO = "filter_kernel,Vivado 2024.2" *) (* hls_module = "yes" *) 
(* NotValidForBitStream *)
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix
   (kernel_ce0,
    kernel_we0,
    kernel_ce1,
    kernel_we1,
    s_axi_control_ARADDR,
    s_axi_control_ARREADY,
    s_axi_control_ARVALID,
    s_axi_control_AWADDR,
    s_axi_control_AWREADY,
    s_axi_control_AWVALID,
    s_axi_control_BREADY,
    s_axi_control_BRESP,
    s_axi_control_BVALID,
    s_axi_control_RDATA,
    s_axi_control_RREADY,
    s_axi_control_RRESP,
    s_axi_control_RVALID,
    s_axi_control_WDATA,
    s_axi_control_WREADY,
    s_axi_control_WSTRB,
    s_axi_control_WVALID,
    ap_clk,
    ap_rst_n,
    interrupt,
    kernel_address0,
    kernel_d0,
    kernel_q0,
    kernel_address1,
    kernel_d1,
    kernel_q1,
    input_stream_TDATA,
    input_stream_TKEEP,
    input_stream_TLAST,
    input_stream_TREADY,
    input_stream_TSTRB,
    input_stream_TVALID,
    output_stream_TDATA,
    output_stream_TKEEP,
    output_stream_TLAST,
    output_stream_TREADY,
    output_stream_TSTRB,
    output_stream_TVALID);
  output kernel_ce0;
  output kernel_we0;
  output kernel_ce1;
  output kernel_we1;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_control ARADDR" *) (* X_INTERFACE_MODE = "slave" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME s_axi_control, ADDR_WIDTH 6, DATA_WIDTH 32, PROTOCOL AXI4LITE, READ_WRITE_MODE READ_WRITE, FREQ_HZ 100000000, ID_WIDTH 0, AWUSER_WIDTH 0, ARUSER_WIDTH 0, WUSER_WIDTH 0, RUSER_WIDTH 0, BUSER_WIDTH 0, HAS_BURST 0, HAS_LOCK 0, HAS_PROT 0, HAS_CACHE 0, HAS_QOS 0, HAS_REGION 0, HAS_WSTRB 1, HAS_BRESP 1, HAS_RRESP 1, SUPPORTS_NARROW_BURST 0, NUM_READ_OUTSTANDING 1, NUM_WRITE_OUTSTANDING 1, MAX_BURST_LENGTH 1, PHASE 0.0, CLK_DOMAIN design_1_processing_system7_0_2_FCLK_CLK0, NUM_READ_THREADS 1, NUM_WRITE_THREADS 1, RUSER_BITS_PER_BYTE 0, WUSER_BITS_PER_BYTE 0, INSERT_VIP 0" *) input [5:0]s_axi_control_ARADDR;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_control ARREADY" *) output s_axi_control_ARREADY;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_control ARVALID" *) input s_axi_control_ARVALID;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_control AWADDR" *) input [5:0]s_axi_control_AWADDR;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_control AWREADY" *) output s_axi_control_AWREADY;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_control AWVALID" *) input s_axi_control_AWVALID;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_control BREADY" *) input s_axi_control_BREADY;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_control BRESP" *) output [1:0]s_axi_control_BRESP;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_control BVALID" *) output s_axi_control_BVALID;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_control RDATA" *) output [31:0]s_axi_control_RDATA;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_control RREADY" *) input s_axi_control_RREADY;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_control RRESP" *) output [1:0]s_axi_control_RRESP;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_control RVALID" *) output s_axi_control_RVALID;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_control WDATA" *) input [31:0]s_axi_control_WDATA;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_control WREADY" *) output s_axi_control_WREADY;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_control WSTRB" *) input [3:0]s_axi_control_WSTRB;
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi_control WVALID" *) input s_axi_control_WVALID;
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *) (* X_INTERFACE_MODE = "slave" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME ap_clk, ASSOCIATED_BUSIF s_axi_control:input_stream:output_stream, ASSOCIATED_RESET ap_rst_n, FREQ_HZ 100000000, FREQ_TOLERANCE_HZ 0, PHASE 0.0, CLK_DOMAIN design_1_processing_system7_0_2_FCLK_CLK0, INSERT_VIP 0" *) input ap_clk;
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ap_rst_n RST" *) (* X_INTERFACE_MODE = "slave" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME ap_rst_n, POLARITY ACTIVE_LOW, INSERT_VIP 0" *) input ap_rst_n;
  (* X_INTERFACE_INFO = "xilinx.com:signal:interrupt:1.0 interrupt INTERRUPT" *) (* X_INTERFACE_MODE = "master" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME interrupt, SENSITIVITY LEVEL_HIGH, PortWidth 1" *) output interrupt;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 kernel_address0 DATA" *) (* X_INTERFACE_MODE = "master" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME kernel_address0, LAYERED_METADATA undef" *) output [3:0]kernel_address0;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 kernel_d0 DATA" *) (* X_INTERFACE_MODE = "master" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME kernel_d0, LAYERED_METADATA undef" *) output [31:0]kernel_d0;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 kernel_q0 DATA" *) (* X_INTERFACE_MODE = "slave" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME kernel_q0, LAYERED_METADATA undef" *) input [31:0]kernel_q0;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 kernel_address1 DATA" *) (* X_INTERFACE_MODE = "master" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME kernel_address1, LAYERED_METADATA undef" *) output [3:0]kernel_address1;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 kernel_d1 DATA" *) (* X_INTERFACE_MODE = "master" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME kernel_d1, LAYERED_METADATA undef" *) output [31:0]kernel_d1;
  (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 kernel_q1 DATA" *) (* X_INTERFACE_MODE = "slave" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME kernel_q1, LAYERED_METADATA undef" *) input [31:0]kernel_q1;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 input_stream TDATA" *) (* X_INTERFACE_MODE = "slave" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME input_stream, TUSER_WIDTH 0, TDATA_NUM_BYTES 4, TDEST_WIDTH 0, TID_WIDTH 0, HAS_TREADY 1, HAS_TSTRB 1, HAS_TKEEP 1, HAS_TLAST 1, FREQ_HZ 100000000, PHASE 0.0, CLK_DOMAIN design_1_processing_system7_0_2_FCLK_CLK0, LAYERED_METADATA undef, INSERT_VIP 0" *) input [31:0]input_stream_TDATA;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 input_stream TKEEP" *) input [3:0]input_stream_TKEEP;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 input_stream TLAST" *) input [0:0]input_stream_TLAST;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 input_stream TREADY" *) output input_stream_TREADY;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 input_stream TSTRB" *) input [3:0]input_stream_TSTRB;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 input_stream TVALID" *) input input_stream_TVALID;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 output_stream TDATA" *) (* X_INTERFACE_MODE = "master" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME output_stream, TUSER_WIDTH 0, TDATA_NUM_BYTES 4, TDEST_WIDTH 0, TID_WIDTH 0, HAS_TREADY 1, HAS_TSTRB 1, HAS_TKEEP 1, HAS_TLAST 1, FREQ_HZ 100000000, PHASE 0.0, CLK_DOMAIN design_1_processing_system7_0_2_FCLK_CLK0, LAYERED_METADATA undef, INSERT_VIP 0" *) output [31:0]output_stream_TDATA;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 output_stream TKEEP" *) output [3:0]output_stream_TKEEP;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 output_stream TLAST" *) output [0:0]output_stream_TLAST;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 output_stream TREADY" *) input output_stream_TREADY;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 output_stream TSTRB" *) output [3:0]output_stream_TSTRB;
  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 output_stream TVALID" *) output output_stream_TVALID;

  wire \<const0> ;
  wire ap_clk;
  wire ap_rst_n;
  wire [3:0]input_stream_TKEEP;
  wire input_stream_TREADY;
  wire [3:0]input_stream_TSTRB;
  wire input_stream_TVALID;
  wire interrupt;
  wire [3:1]\^output_stream_TDATA ;
  wire [3:0]output_stream_TKEEP;
  wire [0:0]output_stream_TLAST;
  wire output_stream_TREADY;
  wire [3:0]output_stream_TSTRB;
  wire output_stream_TVALID;
  wire [5:0]s_axi_control_ARADDR;
  wire s_axi_control_ARREADY;
  wire s_axi_control_ARVALID;
  wire [5:0]s_axi_control_AWADDR;
  wire s_axi_control_AWREADY;
  wire s_axi_control_AWVALID;
  wire s_axi_control_BREADY;
  wire s_axi_control_BVALID;
  wire [31:0]s_axi_control_RDATA;
  wire s_axi_control_RREADY;
  wire s_axi_control_RVALID;
  wire [31:0]s_axi_control_WDATA;
  wire s_axi_control_WREADY;
  wire [3:0]s_axi_control_WSTRB;
  wire s_axi_control_WVALID;
  wire NLW_inst_kernel_ce0_UNCONNECTED;
  wire NLW_inst_kernel_ce1_UNCONNECTED;
  wire NLW_inst_kernel_we0_UNCONNECTED;
  wire NLW_inst_kernel_we1_UNCONNECTED;
  wire [3:0]NLW_inst_kernel_address0_UNCONNECTED;
  wire [3:0]NLW_inst_kernel_address1_UNCONNECTED;
  wire [31:0]NLW_inst_kernel_d0_UNCONNECTED;
  wire [31:0]NLW_inst_kernel_d1_UNCONNECTED;
  wire [31:0]NLW_inst_output_stream_TDATA_UNCONNECTED;
  wire [1:0]NLW_inst_s_axi_control_BRESP_UNCONNECTED;
  wire [1:0]NLW_inst_s_axi_control_RRESP_UNCONNECTED;

  assign kernel_address0[3] = \<const0> ;
  assign kernel_address0[2] = \<const0> ;
  assign kernel_address0[1] = \<const0> ;
  assign kernel_address0[0] = \<const0> ;
  assign kernel_address1[3] = \<const0> ;
  assign kernel_address1[2] = \<const0> ;
  assign kernel_address1[1] = \<const0> ;
  assign kernel_address1[0] = \<const0> ;
  assign kernel_ce0 = \<const0> ;
  assign kernel_ce1 = \<const0> ;
  assign kernel_d0[31] = \<const0> ;
  assign kernel_d0[30] = \<const0> ;
  assign kernel_d0[29] = \<const0> ;
  assign kernel_d0[28] = \<const0> ;
  assign kernel_d0[27] = \<const0> ;
  assign kernel_d0[26] = \<const0> ;
  assign kernel_d0[25] = \<const0> ;
  assign kernel_d0[24] = \<const0> ;
  assign kernel_d0[23] = \<const0> ;
  assign kernel_d0[22] = \<const0> ;
  assign kernel_d0[21] = \<const0> ;
  assign kernel_d0[20] = \<const0> ;
  assign kernel_d0[19] = \<const0> ;
  assign kernel_d0[18] = \<const0> ;
  assign kernel_d0[17] = \<const0> ;
  assign kernel_d0[16] = \<const0> ;
  assign kernel_d0[15] = \<const0> ;
  assign kernel_d0[14] = \<const0> ;
  assign kernel_d0[13] = \<const0> ;
  assign kernel_d0[12] = \<const0> ;
  assign kernel_d0[11] = \<const0> ;
  assign kernel_d0[10] = \<const0> ;
  assign kernel_d0[9] = \<const0> ;
  assign kernel_d0[8] = \<const0> ;
  assign kernel_d0[7] = \<const0> ;
  assign kernel_d0[6] = \<const0> ;
  assign kernel_d0[5] = \<const0> ;
  assign kernel_d0[4] = \<const0> ;
  assign kernel_d0[3] = \<const0> ;
  assign kernel_d0[2] = \<const0> ;
  assign kernel_d0[1] = \<const0> ;
  assign kernel_d0[0] = \<const0> ;
  assign kernel_d1[31] = \<const0> ;
  assign kernel_d1[30] = \<const0> ;
  assign kernel_d1[29] = \<const0> ;
  assign kernel_d1[28] = \<const0> ;
  assign kernel_d1[27] = \<const0> ;
  assign kernel_d1[26] = \<const0> ;
  assign kernel_d1[25] = \<const0> ;
  assign kernel_d1[24] = \<const0> ;
  assign kernel_d1[23] = \<const0> ;
  assign kernel_d1[22] = \<const0> ;
  assign kernel_d1[21] = \<const0> ;
  assign kernel_d1[20] = \<const0> ;
  assign kernel_d1[19] = \<const0> ;
  assign kernel_d1[18] = \<const0> ;
  assign kernel_d1[17] = \<const0> ;
  assign kernel_d1[16] = \<const0> ;
  assign kernel_d1[15] = \<const0> ;
  assign kernel_d1[14] = \<const0> ;
  assign kernel_d1[13] = \<const0> ;
  assign kernel_d1[12] = \<const0> ;
  assign kernel_d1[11] = \<const0> ;
  assign kernel_d1[10] = \<const0> ;
  assign kernel_d1[9] = \<const0> ;
  assign kernel_d1[8] = \<const0> ;
  assign kernel_d1[7] = \<const0> ;
  assign kernel_d1[6] = \<const0> ;
  assign kernel_d1[5] = \<const0> ;
  assign kernel_d1[4] = \<const0> ;
  assign kernel_d1[3] = \<const0> ;
  assign kernel_d1[2] = \<const0> ;
  assign kernel_d1[1] = \<const0> ;
  assign kernel_d1[0] = \<const0> ;
  assign kernel_we0 = \<const0> ;
  assign kernel_we1 = \<const0> ;
  assign output_stream_TDATA[31] = \<const0> ;
  assign output_stream_TDATA[30] = \<const0> ;
  assign output_stream_TDATA[29] = \<const0> ;
  assign output_stream_TDATA[28] = \<const0> ;
  assign output_stream_TDATA[27] = \<const0> ;
  assign output_stream_TDATA[26] = \<const0> ;
  assign output_stream_TDATA[25] = \<const0> ;
  assign output_stream_TDATA[24] = \<const0> ;
  assign output_stream_TDATA[23] = \<const0> ;
  assign output_stream_TDATA[22] = \<const0> ;
  assign output_stream_TDATA[21] = \<const0> ;
  assign output_stream_TDATA[20] = \<const0> ;
  assign output_stream_TDATA[19] = \<const0> ;
  assign output_stream_TDATA[18] = \<const0> ;
  assign output_stream_TDATA[17] = \<const0> ;
  assign output_stream_TDATA[16] = \<const0> ;
  assign output_stream_TDATA[15] = \<const0> ;
  assign output_stream_TDATA[14] = \<const0> ;
  assign output_stream_TDATA[13] = \<const0> ;
  assign output_stream_TDATA[12] = \<const0> ;
  assign output_stream_TDATA[11] = \<const0> ;
  assign output_stream_TDATA[10] = \<const0> ;
  assign output_stream_TDATA[9] = \<const0> ;
  assign output_stream_TDATA[8] = \<const0> ;
  assign output_stream_TDATA[7] = \<const0> ;
  assign output_stream_TDATA[6] = \<const0> ;
  assign output_stream_TDATA[5] = \<const0> ;
  assign output_stream_TDATA[4] = \<const0> ;
  assign output_stream_TDATA[3] = \^output_stream_TDATA [3];
  assign output_stream_TDATA[2] = \<const0> ;
  assign output_stream_TDATA[1] = \^output_stream_TDATA [1];
  assign output_stream_TDATA[0] = \<const0> ;
  assign s_axi_control_BRESP[1] = \<const0> ;
  assign s_axi_control_BRESP[0] = \<const0> ;
  assign s_axi_control_RRESP[1] = \<const0> ;
  assign s_axi_control_RRESP[0] = \<const0> ;
  GND GND
       (.G(\<const0> ));
  (* C_S_AXI_CONTROL_ADDR_WIDTH = "6" *) 
  (* C_S_AXI_CONTROL_DATA_WIDTH = "32" *) 
  (* C_S_AXI_CONTROL_WSTRB_WIDTH = "4" *) 
  (* C_S_AXI_DATA_WIDTH = "32" *) 
  (* C_S_AXI_WSTRB_WIDTH = "4" *) 
  (* SDX_KERNEL = "true" *) 
  (* SDX_KERNEL_SYNTH_INST = "inst" *) 
  (* SDX_KERNEL_TYPE = "hls" *) 
  (* ap_ST_fsm_state1 = "7'b0000001" *) 
  (* ap_ST_fsm_state2 = "7'b0000010" *) 
  (* ap_ST_fsm_state3 = "7'b0000100" *) 
  (* ap_ST_fsm_state4 = "7'b0001000" *) 
  (* ap_ST_fsm_state5 = "7'b0010000" *) 
  (* ap_ST_fsm_state6 = "7'b0100000" *) 
  (* ap_ST_fsm_state7 = "7'b1000000" *) 
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel inst
       (.ap_clk(ap_clk),
        .ap_rst_n(ap_rst_n),
        .input_stream_TDATA({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .input_stream_TKEEP(input_stream_TKEEP),
        .input_stream_TLAST(1'b0),
        .input_stream_TREADY(input_stream_TREADY),
        .input_stream_TSTRB(input_stream_TSTRB),
        .input_stream_TVALID(input_stream_TVALID),
        .interrupt(interrupt),
        .kernel_address0(NLW_inst_kernel_address0_UNCONNECTED[3:0]),
        .kernel_address1(NLW_inst_kernel_address1_UNCONNECTED[3:0]),
        .kernel_ce0(NLW_inst_kernel_ce0_UNCONNECTED),
        .kernel_ce1(NLW_inst_kernel_ce1_UNCONNECTED),
        .kernel_d0(NLW_inst_kernel_d0_UNCONNECTED[31:0]),
        .kernel_d1(NLW_inst_kernel_d1_UNCONNECTED[31:0]),
        .kernel_q0({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .kernel_q1({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .kernel_we0(NLW_inst_kernel_we0_UNCONNECTED),
        .kernel_we1(NLW_inst_kernel_we1_UNCONNECTED),
        .output_stream_TDATA({NLW_inst_output_stream_TDATA_UNCONNECTED[31:4],\^output_stream_TDATA ,NLW_inst_output_stream_TDATA_UNCONNECTED[0]}),
        .output_stream_TKEEP(output_stream_TKEEP),
        .output_stream_TLAST(output_stream_TLAST),
        .output_stream_TREADY(output_stream_TREADY),
        .output_stream_TSTRB(output_stream_TSTRB),
        .output_stream_TVALID(output_stream_TVALID),
        .s_axi_control_ARADDR(s_axi_control_ARADDR),
        .s_axi_control_ARREADY(s_axi_control_ARREADY),
        .s_axi_control_ARVALID(s_axi_control_ARVALID),
        .s_axi_control_AWADDR({s_axi_control_AWADDR[5:2],1'b0,1'b0}),
        .s_axi_control_AWREADY(s_axi_control_AWREADY),
        .s_axi_control_AWVALID(s_axi_control_AWVALID),
        .s_axi_control_BREADY(s_axi_control_BREADY),
        .s_axi_control_BRESP(NLW_inst_s_axi_control_BRESP_UNCONNECTED[1:0]),
        .s_axi_control_BVALID(s_axi_control_BVALID),
        .s_axi_control_RDATA(s_axi_control_RDATA),
        .s_axi_control_RREADY(s_axi_control_RREADY),
        .s_axi_control_RRESP(NLW_inst_s_axi_control_RRESP_UNCONNECTED[1:0]),
        .s_axi_control_RVALID(s_axi_control_RVALID),
        .s_axi_control_WDATA(s_axi_control_WDATA),
        .s_axi_control_WREADY(s_axi_control_WREADY),
        .s_axi_control_WSTRB(s_axi_control_WSTRB),
        .s_axi_control_WVALID(s_axi_control_WVALID));
endmodule

(* C_S_AXI_CONTROL_ADDR_WIDTH = "6" *) (* C_S_AXI_CONTROL_DATA_WIDTH = "32" *) (* C_S_AXI_CONTROL_WSTRB_WIDTH = "4" *) 
(* C_S_AXI_DATA_WIDTH = "32" *) (* C_S_AXI_WSTRB_WIDTH = "4" *) (* ap_ST_fsm_state1 = "7'b0000001" *) 
(* ap_ST_fsm_state2 = "7'b0000010" *) (* ap_ST_fsm_state3 = "7'b0000100" *) (* ap_ST_fsm_state4 = "7'b0001000" *) 
(* ap_ST_fsm_state5 = "7'b0010000" *) (* ap_ST_fsm_state6 = "7'b0100000" *) (* ap_ST_fsm_state7 = "7'b1000000" *) 
(* hls_module = "yes" *) 
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel
   (ap_clk,
    ap_rst_n,
    kernel_address0,
    kernel_ce0,
    kernel_we0,
    kernel_d0,
    kernel_q0,
    kernel_address1,
    kernel_ce1,
    kernel_we1,
    kernel_d1,
    kernel_q1,
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
    interrupt);
  input ap_clk;
  input ap_rst_n;
  output [3:0]kernel_address0;
  output kernel_ce0;
  output kernel_we0;
  output [31:0]kernel_d0;
  input [31:0]kernel_q0;
  output [3:0]kernel_address1;
  output kernel_ce1;
  output kernel_we1;
  output [31:0]kernel_d1;
  input [31:0]kernel_q1;
  input [31:0]input_stream_TDATA;
  input input_stream_TVALID;
  output input_stream_TREADY;
  input [3:0]input_stream_TKEEP;
  input [3:0]input_stream_TSTRB;
  input [0:0]input_stream_TLAST;
  output [31:0]output_stream_TDATA;
  output output_stream_TVALID;
  input output_stream_TREADY;
  output [3:0]output_stream_TKEEP;
  output [3:0]output_stream_TSTRB;
  output [0:0]output_stream_TLAST;
  input s_axi_control_AWVALID;
  output s_axi_control_AWREADY;
  input [5:0]s_axi_control_AWADDR;
  input s_axi_control_WVALID;
  output s_axi_control_WREADY;
  input [31:0]s_axi_control_WDATA;
  input [3:0]s_axi_control_WSTRB;
  input s_axi_control_ARVALID;
  output s_axi_control_ARREADY;
  input [5:0]s_axi_control_ARADDR;
  output s_axi_control_RVALID;
  input s_axi_control_RREADY;
  output [31:0]s_axi_control_RDATA;
  output [1:0]s_axi_control_RRESP;
  output s_axi_control_BVALID;
  input s_axi_control_BREADY;
  output [1:0]s_axi_control_BRESP;
  output interrupt;

  wire \<const0> ;
  wire \ap_CS_fsm_reg_n_0_[0] ;
  wire \ap_CS_fsm_reg_n_0_[2] ;
  wire ap_CS_fsm_state2;
  wire ap_CS_fsm_state4;
  wire ap_CS_fsm_state5;
  wire ap_CS_fsm_state6;
  wire ap_CS_fsm_state7;
  wire [6:0]ap_NS_fsm;
  wire ap_block_pp0_stage0_subdone;
  wire ap_clk;
  wire ap_rst_n;
  wire ap_rst_n_inv;
  wire ap_start;
  wire [61:16]buff0_reg__1;
  wire control_s_axi_U_n_7;
  wire data_p2;
  wire grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg;
  wire grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_n_0;
  wire grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_n_1;
  wire grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_n_8;
  wire grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_n_9;
  wire grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TLAST;
  wire grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID;
  wire [31:0]image_height;
  wire image_height_0_data_reg_reg__0_n_0;
  wire image_height_0_data_reg_reg__10_n_0;
  wire image_height_0_data_reg_reg__11_n_0;
  wire image_height_0_data_reg_reg__12_n_0;
  wire image_height_0_data_reg_reg__13_n_0;
  wire image_height_0_data_reg_reg__1_n_0;
  wire image_height_0_data_reg_reg__2_n_0;
  wire image_height_0_data_reg_reg__3_n_0;
  wire image_height_0_data_reg_reg__4_n_0;
  wire image_height_0_data_reg_reg__5_n_0;
  wire image_height_0_data_reg_reg__6_n_0;
  wire image_height_0_data_reg_reg__7_n_0;
  wire image_height_0_data_reg_reg__8_n_0;
  wire image_height_0_data_reg_reg__9_n_0;
  wire \image_height_0_data_reg_reg_n_0_[0] ;
  wire \image_height_0_data_reg_reg_n_0_[10] ;
  wire \image_height_0_data_reg_reg_n_0_[11] ;
  wire \image_height_0_data_reg_reg_n_0_[12] ;
  wire \image_height_0_data_reg_reg_n_0_[13] ;
  wire \image_height_0_data_reg_reg_n_0_[14] ;
  wire \image_height_0_data_reg_reg_n_0_[15] ;
  wire \image_height_0_data_reg_reg_n_0_[16] ;
  wire \image_height_0_data_reg_reg_n_0_[17] ;
  wire \image_height_0_data_reg_reg_n_0_[18] ;
  wire \image_height_0_data_reg_reg_n_0_[19] ;
  wire \image_height_0_data_reg_reg_n_0_[1] ;
  wire \image_height_0_data_reg_reg_n_0_[20] ;
  wire \image_height_0_data_reg_reg_n_0_[21] ;
  wire \image_height_0_data_reg_reg_n_0_[22] ;
  wire \image_height_0_data_reg_reg_n_0_[23] ;
  wire \image_height_0_data_reg_reg_n_0_[24] ;
  wire \image_height_0_data_reg_reg_n_0_[25] ;
  wire \image_height_0_data_reg_reg_n_0_[26] ;
  wire \image_height_0_data_reg_reg_n_0_[27] ;
  wire \image_height_0_data_reg_reg_n_0_[28] ;
  wire \image_height_0_data_reg_reg_n_0_[29] ;
  wire \image_height_0_data_reg_reg_n_0_[2] ;
  wire \image_height_0_data_reg_reg_n_0_[30] ;
  wire \image_height_0_data_reg_reg_n_0_[31] ;
  wire \image_height_0_data_reg_reg_n_0_[3] ;
  wire \image_height_0_data_reg_reg_n_0_[4] ;
  wire \image_height_0_data_reg_reg_n_0_[5] ;
  wire \image_height_0_data_reg_reg_n_0_[6] ;
  wire \image_height_0_data_reg_reg_n_0_[7] ;
  wire \image_height_0_data_reg_reg_n_0_[8] ;
  wire \image_height_0_data_reg_reg_n_0_[9] ;
  wire [31:0]image_height_read_reg_168;
  wire [31:0]image_width;
  wire image_width_0_data_reg_reg__0_n_0;
  wire image_width_0_data_reg_reg__10_n_0;
  wire image_width_0_data_reg_reg__11_n_0;
  wire image_width_0_data_reg_reg__12_n_0;
  wire image_width_0_data_reg_reg__13_n_0;
  wire image_width_0_data_reg_reg__1_n_0;
  wire image_width_0_data_reg_reg__2_n_0;
  wire image_width_0_data_reg_reg__3_n_0;
  wire image_width_0_data_reg_reg__4_n_0;
  wire image_width_0_data_reg_reg__5_n_0;
  wire image_width_0_data_reg_reg__6_n_0;
  wire image_width_0_data_reg_reg__7_n_0;
  wire image_width_0_data_reg_reg__8_n_0;
  wire image_width_0_data_reg_reg__9_n_0;
  wire \image_width_0_data_reg_reg_n_0_[0] ;
  wire \image_width_0_data_reg_reg_n_0_[10] ;
  wire \image_width_0_data_reg_reg_n_0_[11] ;
  wire \image_width_0_data_reg_reg_n_0_[12] ;
  wire \image_width_0_data_reg_reg_n_0_[13] ;
  wire \image_width_0_data_reg_reg_n_0_[14] ;
  wire \image_width_0_data_reg_reg_n_0_[15] ;
  wire \image_width_0_data_reg_reg_n_0_[16] ;
  wire \image_width_0_data_reg_reg_n_0_[17] ;
  wire \image_width_0_data_reg_reg_n_0_[18] ;
  wire \image_width_0_data_reg_reg_n_0_[19] ;
  wire \image_width_0_data_reg_reg_n_0_[1] ;
  wire \image_width_0_data_reg_reg_n_0_[20] ;
  wire \image_width_0_data_reg_reg_n_0_[21] ;
  wire \image_width_0_data_reg_reg_n_0_[22] ;
  wire \image_width_0_data_reg_reg_n_0_[23] ;
  wire \image_width_0_data_reg_reg_n_0_[24] ;
  wire \image_width_0_data_reg_reg_n_0_[25] ;
  wire \image_width_0_data_reg_reg_n_0_[26] ;
  wire \image_width_0_data_reg_reg_n_0_[27] ;
  wire \image_width_0_data_reg_reg_n_0_[28] ;
  wire \image_width_0_data_reg_reg_n_0_[29] ;
  wire \image_width_0_data_reg_reg_n_0_[2] ;
  wire \image_width_0_data_reg_reg_n_0_[30] ;
  wire \image_width_0_data_reg_reg_n_0_[31] ;
  wire \image_width_0_data_reg_reg_n_0_[3] ;
  wire \image_width_0_data_reg_reg_n_0_[4] ;
  wire \image_width_0_data_reg_reg_n_0_[5] ;
  wire \image_width_0_data_reg_reg_n_0_[6] ;
  wire \image_width_0_data_reg_reg_n_0_[7] ;
  wire \image_width_0_data_reg_reg_n_0_[8] ;
  wire \image_width_0_data_reg_reg_n_0_[9] ;
  wire [31:0]image_width_read_reg_173;
  wire [3:0]input_stream_TKEEP;
  wire [3:0]input_stream_TKEEP_int_regslice;
  wire input_stream_TREADY;
  wire input_stream_TREADY_int_regslice;
  wire [3:0]input_stream_TSTRB;
  wire [3:0]input_stream_TSTRB_int_regslice;
  wire input_stream_TVALID;
  wire input_stream_TVALID_int_regslice;
  wire interrupt;
  wire load_p2;
  wire load_p2_0;
  wire mul_31ns_31ns_62_2_1_U13_n_46;
  wire mul_31ns_31ns_62_2_1_U13_n_47;
  wire mul_31ns_31ns_62_2_1_U13_n_48;
  wire mul_31ns_31ns_62_2_1_U13_n_49;
  wire mul_31ns_31ns_62_2_1_U13_n_50;
  wire mul_31ns_31ns_62_2_1_U13_n_51;
  wire mul_31ns_31ns_62_2_1_U13_n_52;
  wire mul_31ns_31ns_62_2_1_U13_n_53;
  wire mul_31ns_31ns_62_2_1_U13_n_54;
  wire mul_31ns_31ns_62_2_1_U13_n_55;
  wire mul_31ns_31ns_62_2_1_U13_n_56;
  wire mul_31ns_31ns_62_2_1_U13_n_57;
  wire mul_31ns_31ns_62_2_1_U13_n_58;
  wire mul_31ns_31ns_62_2_1_U13_n_59;
  wire mul_31ns_31ns_62_2_1_U13_n_60;
  wire mul_31ns_31ns_62_2_1_U13_n_61;
  wire [61:0]mul_ln10_reg_199;
  wire [3:1]\^output_stream_TDATA ;
  wire [3:0]output_stream_TKEEP;
  wire [3:0]output_stream_TKEEP_reg;
  wire [0:0]output_stream_TLAST;
  wire output_stream_TLAST_reg;
  wire output_stream_TREADY;
  wire output_stream_TREADY_int_regslice;
  wire [3:0]output_stream_TSTRB;
  wire [3:0]output_stream_TSTRB_reg;
  wire output_stream_TVALID;
  wire regslice_both_output_stream_V_data_V_U_n_5;
  wire regslice_both_output_stream_V_keep_V_U_n_0;
  wire regslice_both_output_stream_V_last_V_U_n_0;
  wire regslice_both_output_stream_V_last_V_U_n_3;
  wire regslice_both_output_stream_V_strb_V_U_n_0;
  wire [5:0]s_axi_control_ARADDR;
  wire s_axi_control_ARREADY;
  wire s_axi_control_ARVALID;
  wire [5:0]s_axi_control_AWADDR;
  wire s_axi_control_AWREADY;
  wire s_axi_control_AWVALID;
  wire s_axi_control_BREADY;
  wire s_axi_control_BVALID;
  wire [31:0]s_axi_control_RDATA;
  wire s_axi_control_RREADY;
  wire s_axi_control_RVALID;
  wire [31:0]s_axi_control_WDATA;
  wire s_axi_control_WREADY;
  wire [3:0]s_axi_control_WSTRB;
  wire s_axi_control_WVALID;
  wire [31:0]sub7_fu_162_p2;
  wire [31:0]sub7_reg_209;
  wire \sub7_reg_209[12]_i_2_n_0 ;
  wire \sub7_reg_209[12]_i_3_n_0 ;
  wire \sub7_reg_209[12]_i_4_n_0 ;
  wire \sub7_reg_209[12]_i_5_n_0 ;
  wire \sub7_reg_209[16]_i_2_n_0 ;
  wire \sub7_reg_209[16]_i_3_n_0 ;
  wire \sub7_reg_209[16]_i_4_n_0 ;
  wire \sub7_reg_209[16]_i_5_n_0 ;
  wire \sub7_reg_209[20]_i_2_n_0 ;
  wire \sub7_reg_209[20]_i_3_n_0 ;
  wire \sub7_reg_209[20]_i_4_n_0 ;
  wire \sub7_reg_209[20]_i_5_n_0 ;
  wire \sub7_reg_209[24]_i_2_n_0 ;
  wire \sub7_reg_209[24]_i_3_n_0 ;
  wire \sub7_reg_209[24]_i_4_n_0 ;
  wire \sub7_reg_209[24]_i_5_n_0 ;
  wire \sub7_reg_209[28]_i_2_n_0 ;
  wire \sub7_reg_209[28]_i_3_n_0 ;
  wire \sub7_reg_209[28]_i_4_n_0 ;
  wire \sub7_reg_209[28]_i_5_n_0 ;
  wire \sub7_reg_209[31]_i_2_n_0 ;
  wire \sub7_reg_209[31]_i_3_n_0 ;
  wire \sub7_reg_209[31]_i_4_n_0 ;
  wire \sub7_reg_209[4]_i_2_n_0 ;
  wire \sub7_reg_209[4]_i_3_n_0 ;
  wire \sub7_reg_209[4]_i_4_n_0 ;
  wire \sub7_reg_209[4]_i_5_n_0 ;
  wire \sub7_reg_209[8]_i_2_n_0 ;
  wire \sub7_reg_209[8]_i_3_n_0 ;
  wire \sub7_reg_209[8]_i_4_n_0 ;
  wire \sub7_reg_209[8]_i_5_n_0 ;
  wire \sub7_reg_209_reg[12]_i_1_n_0 ;
  wire \sub7_reg_209_reg[12]_i_1_n_1 ;
  wire \sub7_reg_209_reg[12]_i_1_n_2 ;
  wire \sub7_reg_209_reg[12]_i_1_n_3 ;
  wire \sub7_reg_209_reg[16]_i_1_n_0 ;
  wire \sub7_reg_209_reg[16]_i_1_n_1 ;
  wire \sub7_reg_209_reg[16]_i_1_n_2 ;
  wire \sub7_reg_209_reg[16]_i_1_n_3 ;
  wire \sub7_reg_209_reg[20]_i_1_n_0 ;
  wire \sub7_reg_209_reg[20]_i_1_n_1 ;
  wire \sub7_reg_209_reg[20]_i_1_n_2 ;
  wire \sub7_reg_209_reg[20]_i_1_n_3 ;
  wire \sub7_reg_209_reg[24]_i_1_n_0 ;
  wire \sub7_reg_209_reg[24]_i_1_n_1 ;
  wire \sub7_reg_209_reg[24]_i_1_n_2 ;
  wire \sub7_reg_209_reg[24]_i_1_n_3 ;
  wire \sub7_reg_209_reg[28]_i_1_n_0 ;
  wire \sub7_reg_209_reg[28]_i_1_n_1 ;
  wire \sub7_reg_209_reg[28]_i_1_n_2 ;
  wire \sub7_reg_209_reg[28]_i_1_n_3 ;
  wire \sub7_reg_209_reg[31]_i_1_n_2 ;
  wire \sub7_reg_209_reg[31]_i_1_n_3 ;
  wire \sub7_reg_209_reg[4]_i_1_n_0 ;
  wire \sub7_reg_209_reg[4]_i_1_n_1 ;
  wire \sub7_reg_209_reg[4]_i_1_n_2 ;
  wire \sub7_reg_209_reg[4]_i_1_n_3 ;
  wire \sub7_reg_209_reg[8]_i_1_n_0 ;
  wire \sub7_reg_209_reg[8]_i_1_n_1 ;
  wire \sub7_reg_209_reg[8]_i_1_n_2 ;
  wire \sub7_reg_209_reg[8]_i_1_n_3 ;
  wire [31:0]sub_fu_156_p2;
  wire [31:0]sub_reg_204;
  wire \sub_reg_204[12]_i_2_n_0 ;
  wire \sub_reg_204[12]_i_3_n_0 ;
  wire \sub_reg_204[12]_i_4_n_0 ;
  wire \sub_reg_204[12]_i_5_n_0 ;
  wire \sub_reg_204[16]_i_2_n_0 ;
  wire \sub_reg_204[16]_i_3_n_0 ;
  wire \sub_reg_204[16]_i_4_n_0 ;
  wire \sub_reg_204[16]_i_5_n_0 ;
  wire \sub_reg_204[20]_i_2_n_0 ;
  wire \sub_reg_204[20]_i_3_n_0 ;
  wire \sub_reg_204[20]_i_4_n_0 ;
  wire \sub_reg_204[20]_i_5_n_0 ;
  wire \sub_reg_204[24]_i_2_n_0 ;
  wire \sub_reg_204[24]_i_3_n_0 ;
  wire \sub_reg_204[24]_i_4_n_0 ;
  wire \sub_reg_204[24]_i_5_n_0 ;
  wire \sub_reg_204[28]_i_2_n_0 ;
  wire \sub_reg_204[28]_i_3_n_0 ;
  wire \sub_reg_204[28]_i_4_n_0 ;
  wire \sub_reg_204[28]_i_5_n_0 ;
  wire \sub_reg_204[31]_i_2_n_0 ;
  wire \sub_reg_204[31]_i_3_n_0 ;
  wire \sub_reg_204[31]_i_4_n_0 ;
  wire \sub_reg_204[4]_i_2_n_0 ;
  wire \sub_reg_204[4]_i_3_n_0 ;
  wire \sub_reg_204[4]_i_4_n_0 ;
  wire \sub_reg_204[4]_i_5_n_0 ;
  wire \sub_reg_204[8]_i_2_n_0 ;
  wire \sub_reg_204[8]_i_3_n_0 ;
  wire \sub_reg_204[8]_i_4_n_0 ;
  wire \sub_reg_204[8]_i_5_n_0 ;
  wire \sub_reg_204_reg[12]_i_1_n_0 ;
  wire \sub_reg_204_reg[12]_i_1_n_1 ;
  wire \sub_reg_204_reg[12]_i_1_n_2 ;
  wire \sub_reg_204_reg[12]_i_1_n_3 ;
  wire \sub_reg_204_reg[16]_i_1_n_0 ;
  wire \sub_reg_204_reg[16]_i_1_n_1 ;
  wire \sub_reg_204_reg[16]_i_1_n_2 ;
  wire \sub_reg_204_reg[16]_i_1_n_3 ;
  wire \sub_reg_204_reg[20]_i_1_n_0 ;
  wire \sub_reg_204_reg[20]_i_1_n_1 ;
  wire \sub_reg_204_reg[20]_i_1_n_2 ;
  wire \sub_reg_204_reg[20]_i_1_n_3 ;
  wire \sub_reg_204_reg[24]_i_1_n_0 ;
  wire \sub_reg_204_reg[24]_i_1_n_1 ;
  wire \sub_reg_204_reg[24]_i_1_n_2 ;
  wire \sub_reg_204_reg[24]_i_1_n_3 ;
  wire \sub_reg_204_reg[28]_i_1_n_0 ;
  wire \sub_reg_204_reg[28]_i_1_n_1 ;
  wire \sub_reg_204_reg[28]_i_1_n_2 ;
  wire \sub_reg_204_reg[28]_i_1_n_3 ;
  wire \sub_reg_204_reg[31]_i_1_n_2 ;
  wire \sub_reg_204_reg[31]_i_1_n_3 ;
  wire \sub_reg_204_reg[4]_i_1_n_0 ;
  wire \sub_reg_204_reg[4]_i_1_n_1 ;
  wire \sub_reg_204_reg[4]_i_1_n_2 ;
  wire \sub_reg_204_reg[4]_i_1_n_3 ;
  wire \sub_reg_204_reg[8]_i_1_n_0 ;
  wire \sub_reg_204_reg[8]_i_1_n_1 ;
  wire \sub_reg_204_reg[8]_i_1_n_2 ;
  wire \sub_reg_204_reg[8]_i_1_n_3 ;
  wire [3:2]\NLW_sub7_reg_209_reg[31]_i_1_CO_UNCONNECTED ;
  wire [3:3]\NLW_sub7_reg_209_reg[31]_i_1_O_UNCONNECTED ;
  wire [3:2]\NLW_sub_reg_204_reg[31]_i_1_CO_UNCONNECTED ;
  wire [3:3]\NLW_sub_reg_204_reg[31]_i_1_O_UNCONNECTED ;

  assign kernel_address0[3] = \<const0> ;
  assign kernel_address0[2] = \<const0> ;
  assign kernel_address0[1] = \<const0> ;
  assign kernel_address0[0] = \<const0> ;
  assign kernel_address1[3] = \<const0> ;
  assign kernel_address1[2] = \<const0> ;
  assign kernel_address1[1] = \<const0> ;
  assign kernel_address1[0] = \<const0> ;
  assign kernel_ce0 = \<const0> ;
  assign kernel_ce1 = \<const0> ;
  assign kernel_d0[31] = \<const0> ;
  assign kernel_d0[30] = \<const0> ;
  assign kernel_d0[29] = \<const0> ;
  assign kernel_d0[28] = \<const0> ;
  assign kernel_d0[27] = \<const0> ;
  assign kernel_d0[26] = \<const0> ;
  assign kernel_d0[25] = \<const0> ;
  assign kernel_d0[24] = \<const0> ;
  assign kernel_d0[23] = \<const0> ;
  assign kernel_d0[22] = \<const0> ;
  assign kernel_d0[21] = \<const0> ;
  assign kernel_d0[20] = \<const0> ;
  assign kernel_d0[19] = \<const0> ;
  assign kernel_d0[18] = \<const0> ;
  assign kernel_d0[17] = \<const0> ;
  assign kernel_d0[16] = \<const0> ;
  assign kernel_d0[15] = \<const0> ;
  assign kernel_d0[14] = \<const0> ;
  assign kernel_d0[13] = \<const0> ;
  assign kernel_d0[12] = \<const0> ;
  assign kernel_d0[11] = \<const0> ;
  assign kernel_d0[10] = \<const0> ;
  assign kernel_d0[9] = \<const0> ;
  assign kernel_d0[8] = \<const0> ;
  assign kernel_d0[7] = \<const0> ;
  assign kernel_d0[6] = \<const0> ;
  assign kernel_d0[5] = \<const0> ;
  assign kernel_d0[4] = \<const0> ;
  assign kernel_d0[3] = \<const0> ;
  assign kernel_d0[2] = \<const0> ;
  assign kernel_d0[1] = \<const0> ;
  assign kernel_d0[0] = \<const0> ;
  assign kernel_d1[31] = \<const0> ;
  assign kernel_d1[30] = \<const0> ;
  assign kernel_d1[29] = \<const0> ;
  assign kernel_d1[28] = \<const0> ;
  assign kernel_d1[27] = \<const0> ;
  assign kernel_d1[26] = \<const0> ;
  assign kernel_d1[25] = \<const0> ;
  assign kernel_d1[24] = \<const0> ;
  assign kernel_d1[23] = \<const0> ;
  assign kernel_d1[22] = \<const0> ;
  assign kernel_d1[21] = \<const0> ;
  assign kernel_d1[20] = \<const0> ;
  assign kernel_d1[19] = \<const0> ;
  assign kernel_d1[18] = \<const0> ;
  assign kernel_d1[17] = \<const0> ;
  assign kernel_d1[16] = \<const0> ;
  assign kernel_d1[15] = \<const0> ;
  assign kernel_d1[14] = \<const0> ;
  assign kernel_d1[13] = \<const0> ;
  assign kernel_d1[12] = \<const0> ;
  assign kernel_d1[11] = \<const0> ;
  assign kernel_d1[10] = \<const0> ;
  assign kernel_d1[9] = \<const0> ;
  assign kernel_d1[8] = \<const0> ;
  assign kernel_d1[7] = \<const0> ;
  assign kernel_d1[6] = \<const0> ;
  assign kernel_d1[5] = \<const0> ;
  assign kernel_d1[4] = \<const0> ;
  assign kernel_d1[3] = \<const0> ;
  assign kernel_d1[2] = \<const0> ;
  assign kernel_d1[1] = \<const0> ;
  assign kernel_d1[0] = \<const0> ;
  assign kernel_we0 = \<const0> ;
  assign kernel_we1 = \<const0> ;
  assign output_stream_TDATA[31] = \<const0> ;
  assign output_stream_TDATA[30] = \<const0> ;
  assign output_stream_TDATA[29] = \<const0> ;
  assign output_stream_TDATA[28] = \<const0> ;
  assign output_stream_TDATA[27] = \<const0> ;
  assign output_stream_TDATA[26] = \<const0> ;
  assign output_stream_TDATA[25] = \<const0> ;
  assign output_stream_TDATA[24] = \<const0> ;
  assign output_stream_TDATA[23] = \<const0> ;
  assign output_stream_TDATA[22] = \<const0> ;
  assign output_stream_TDATA[21] = \<const0> ;
  assign output_stream_TDATA[20] = \<const0> ;
  assign output_stream_TDATA[19] = \<const0> ;
  assign output_stream_TDATA[18] = \<const0> ;
  assign output_stream_TDATA[17] = \<const0> ;
  assign output_stream_TDATA[16] = \<const0> ;
  assign output_stream_TDATA[15] = \<const0> ;
  assign output_stream_TDATA[14] = \<const0> ;
  assign output_stream_TDATA[13] = \<const0> ;
  assign output_stream_TDATA[12] = \<const0> ;
  assign output_stream_TDATA[11] = \<const0> ;
  assign output_stream_TDATA[10] = \<const0> ;
  assign output_stream_TDATA[9] = \<const0> ;
  assign output_stream_TDATA[8] = \<const0> ;
  assign output_stream_TDATA[7] = \<const0> ;
  assign output_stream_TDATA[6] = \<const0> ;
  assign output_stream_TDATA[5] = \<const0> ;
  assign output_stream_TDATA[4] = \<const0> ;
  assign output_stream_TDATA[3] = \^output_stream_TDATA [3];
  assign output_stream_TDATA[2] = \<const0> ;
  assign output_stream_TDATA[1] = \^output_stream_TDATA [1];
  assign output_stream_TDATA[0] = \<const0> ;
  assign s_axi_control_BRESP[1] = \<const0> ;
  assign s_axi_control_BRESP[0] = \<const0> ;
  assign s_axi_control_RRESP[1] = \<const0> ;
  assign s_axi_control_RRESP[0] = \<const0> ;
  GND GND
       (.G(\<const0> ));
  (* FSM_ENCODING = "none" *) 
  FDSE #(
    .INIT(1'b1)) 
    \ap_CS_fsm_reg[0] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(ap_NS_fsm[0]),
        .Q(\ap_CS_fsm_reg_n_0_[0] ),
        .S(ap_rst_n_inv));
  (* FSM_ENCODING = "none" *) 
  FDRE #(
    .INIT(1'b0)) 
    \ap_CS_fsm_reg[1] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(ap_NS_fsm[1]),
        .Q(ap_CS_fsm_state2),
        .R(ap_rst_n_inv));
  (* FSM_ENCODING = "none" *) 
  FDRE #(
    .INIT(1'b0)) 
    \ap_CS_fsm_reg[2] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(ap_CS_fsm_state2),
        .Q(\ap_CS_fsm_reg_n_0_[2] ),
        .R(ap_rst_n_inv));
  (* FSM_ENCODING = "none" *) 
  FDRE #(
    .INIT(1'b0)) 
    \ap_CS_fsm_reg[3] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\ap_CS_fsm_reg_n_0_[2] ),
        .Q(ap_CS_fsm_state4),
        .R(ap_rst_n_inv));
  (* FSM_ENCODING = "none" *) 
  FDRE #(
    .INIT(1'b0)) 
    \ap_CS_fsm_reg[4] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(ap_CS_fsm_state4),
        .Q(ap_CS_fsm_state5),
        .R(ap_rst_n_inv));
  (* FSM_ENCODING = "none" *) 
  FDRE #(
    .INIT(1'b0)) 
    \ap_CS_fsm_reg[5] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(ap_NS_fsm[5]),
        .Q(ap_CS_fsm_state6),
        .R(ap_rst_n_inv));
  (* FSM_ENCODING = "none" *) 
  FDRE #(
    .INIT(1'b0)) 
    \ap_CS_fsm_reg[6] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(ap_NS_fsm[6]),
        .Q(ap_CS_fsm_state7),
        .R(ap_rst_n_inv));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_control_s_axi control_s_axi_U
       (.D(ap_NS_fsm[1]),
        .E(control_s_axi_U_n_7),
        .\FSM_onehot_rstate_reg[1]_0 (s_axi_control_ARREADY),
        .\FSM_onehot_wstate_reg[1]_0 (s_axi_control_AWREADY),
        .\FSM_onehot_wstate_reg[2]_0 (s_axi_control_WREADY),
        .Q({ap_CS_fsm_state7,ap_CS_fsm_state6,ap_CS_fsm_state5,ap_CS_fsm_state4,\ap_CS_fsm_reg_n_0_[2] ,ap_CS_fsm_state2,\ap_CS_fsm_reg_n_0_[0] }),
        .ap_clk(ap_clk),
        .ap_rst_n(ap_rst_n),
        .ap_rst_n_inv(ap_rst_n_inv),
        .ap_start(ap_start),
        .int_ap_start_reg_0(regslice_both_output_stream_V_data_V_U_n_5),
        .\int_image_height_reg[31]_0 (image_height),
        .\int_image_width_reg[31]_0 (image_width),
        .interrupt(interrupt),
        .s_axi_control_ARADDR(s_axi_control_ARADDR),
        .s_axi_control_ARVALID(s_axi_control_ARVALID),
        .s_axi_control_AWADDR(s_axi_control_AWADDR[5:2]),
        .s_axi_control_AWVALID(s_axi_control_AWVALID),
        .s_axi_control_BREADY(s_axi_control_BREADY),
        .s_axi_control_BVALID(s_axi_control_BVALID),
        .s_axi_control_RDATA(s_axi_control_RDATA),
        .s_axi_control_RREADY(s_axi_control_RREADY),
        .s_axi_control_RVALID(s_axi_control_RVALID),
        .s_axi_control_WDATA(s_axi_control_WDATA),
        .s_axi_control_WSTRB(s_axi_control_WSTRB),
        .s_axi_control_WVALID(s_axi_control_WVALID));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2 grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84
       (.D(ap_NS_fsm[6:5]),
        .E(ap_block_pp0_stage0_subdone),
        .Q(input_stream_TVALID_int_regslice),
        .\ap_CS_fsm_reg[4] (grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_n_8),
        .\ap_CS_fsm_reg[6] ({ap_CS_fsm_state7,ap_CS_fsm_state6,ap_CS_fsm_state5}),
        .\ap_CS_fsm_reg[6]_0 (regslice_both_output_stream_V_data_V_U_n_5),
        .ap_clk(ap_clk),
        .ap_enable_reg_pp0_iter2_reg_0(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_n_0),
        .ap_enable_reg_pp0_iter2_reg_1(load_p2_0),
        .ap_enable_reg_pp0_iter2_reg_2(load_p2),
        .ap_rst_n(ap_rst_n),
        .ap_rst_n_inv(ap_rst_n_inv),
        .cmp6_fu_242_p2_carry__1_0(sub_reg_204),
        .\data_p1_reg[0] (regslice_both_output_stream_V_last_V_U_n_3),
        .data_p2(data_p2),
        .\data_p2_reg[0] (grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_n_1),
        .\data_p2_reg[0]_0 (regslice_both_output_stream_V_last_V_U_n_0),
        .\data_p2_reg[3] (regslice_both_output_stream_V_strb_V_U_n_0),
        .\data_p2_reg[3]_0 (regslice_both_output_stream_V_keep_V_U_n_0),
        .grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg),
        .grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TLAST(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TLAST),
        .icmp_ln34_fu_153_p2_carry__4_0(mul_ln10_reg_199),
        .icmp_ln35_fu_168_p2_carry__2_0(image_width_read_reg_173),
        .\icmp_ln44_reg_312_reg[0]_0 (grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_n_9),
        .\icmp_ln44_reg_312_reg[0]_1 (sub7_reg_209),
        .input_stream_TREADY_int_regslice(input_stream_TREADY_int_regslice),
        .output_stream_TLAST_reg(output_stream_TLAST_reg),
        .output_stream_TREADY_int_regslice(output_stream_TREADY_int_regslice));
  FDRE #(
    .INIT(1'b0)) 
    grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg_reg
       (.C(ap_clk),
        .CE(1'b1),
        .D(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_n_8),
        .Q(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[0] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[0]),
        .Q(\image_height_0_data_reg_reg_n_0_[0] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[10] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[10]),
        .Q(\image_height_0_data_reg_reg_n_0_[10] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[11] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[11]),
        .Q(\image_height_0_data_reg_reg_n_0_[11] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[12] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[12]),
        .Q(\image_height_0_data_reg_reg_n_0_[12] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[13] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[13]),
        .Q(\image_height_0_data_reg_reg_n_0_[13] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[14] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[14]),
        .Q(\image_height_0_data_reg_reg_n_0_[14] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[15] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[15]),
        .Q(\image_height_0_data_reg_reg_n_0_[15] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[16] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[16]),
        .Q(\image_height_0_data_reg_reg_n_0_[16] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[17] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[17]),
        .Q(\image_height_0_data_reg_reg_n_0_[17] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[18] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[18]),
        .Q(\image_height_0_data_reg_reg_n_0_[18] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[19] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[19]),
        .Q(\image_height_0_data_reg_reg_n_0_[19] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[1] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[1]),
        .Q(\image_height_0_data_reg_reg_n_0_[1] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[20] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[20]),
        .Q(\image_height_0_data_reg_reg_n_0_[20] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[21] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[21]),
        .Q(\image_height_0_data_reg_reg_n_0_[21] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[22] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[22]),
        .Q(\image_height_0_data_reg_reg_n_0_[22] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[23] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[23]),
        .Q(\image_height_0_data_reg_reg_n_0_[23] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[24] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[24]),
        .Q(\image_height_0_data_reg_reg_n_0_[24] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[25] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[25]),
        .Q(\image_height_0_data_reg_reg_n_0_[25] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[26] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[26]),
        .Q(\image_height_0_data_reg_reg_n_0_[26] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[27] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[27]),
        .Q(\image_height_0_data_reg_reg_n_0_[27] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[28] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[28]),
        .Q(\image_height_0_data_reg_reg_n_0_[28] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[29] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[29]),
        .Q(\image_height_0_data_reg_reg_n_0_[29] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[2] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[2]),
        .Q(\image_height_0_data_reg_reg_n_0_[2] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[30] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[30]),
        .Q(\image_height_0_data_reg_reg_n_0_[30] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[31] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[31]),
        .Q(\image_height_0_data_reg_reg_n_0_[31] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[3] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[3]),
        .Q(\image_height_0_data_reg_reg_n_0_[3] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[4] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[4]),
        .Q(\image_height_0_data_reg_reg_n_0_[4] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[5] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[5]),
        .Q(\image_height_0_data_reg_reg_n_0_[5] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[6] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[6]),
        .Q(\image_height_0_data_reg_reg_n_0_[6] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[7] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[7]),
        .Q(\image_height_0_data_reg_reg_n_0_[7] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[8] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[8]),
        .Q(\image_height_0_data_reg_reg_n_0_[8] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_height_0_data_reg_reg[9] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[9]),
        .Q(\image_height_0_data_reg_reg_n_0_[9] ),
        .R(1'b0));
  FDRE image_height_0_data_reg_reg__0
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[30]),
        .Q(image_height_0_data_reg_reg__0_n_0),
        .R(1'b0));
  FDRE image_height_0_data_reg_reg__1
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[29]),
        .Q(image_height_0_data_reg_reg__1_n_0),
        .R(1'b0));
  FDRE image_height_0_data_reg_reg__10
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[20]),
        .Q(image_height_0_data_reg_reg__10_n_0),
        .R(1'b0));
  FDRE image_height_0_data_reg_reg__11
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[19]),
        .Q(image_height_0_data_reg_reg__11_n_0),
        .R(1'b0));
  FDRE image_height_0_data_reg_reg__12
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[18]),
        .Q(image_height_0_data_reg_reg__12_n_0),
        .R(1'b0));
  FDRE image_height_0_data_reg_reg__13
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[17]),
        .Q(image_height_0_data_reg_reg__13_n_0),
        .R(1'b0));
  FDRE image_height_0_data_reg_reg__2
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[28]),
        .Q(image_height_0_data_reg_reg__2_n_0),
        .R(1'b0));
  FDRE image_height_0_data_reg_reg__3
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[27]),
        .Q(image_height_0_data_reg_reg__3_n_0),
        .R(1'b0));
  FDRE image_height_0_data_reg_reg__4
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[26]),
        .Q(image_height_0_data_reg_reg__4_n_0),
        .R(1'b0));
  FDRE image_height_0_data_reg_reg__5
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[25]),
        .Q(image_height_0_data_reg_reg__5_n_0),
        .R(1'b0));
  FDRE image_height_0_data_reg_reg__6
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[24]),
        .Q(image_height_0_data_reg_reg__6_n_0),
        .R(1'b0));
  FDRE image_height_0_data_reg_reg__7
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[23]),
        .Q(image_height_0_data_reg_reg__7_n_0),
        .R(1'b0));
  FDRE image_height_0_data_reg_reg__8
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[22]),
        .Q(image_height_0_data_reg_reg__8_n_0),
        .R(1'b0));
  FDRE image_height_0_data_reg_reg__9
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_height[21]),
        .Q(image_height_0_data_reg_reg__9_n_0),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[0] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[0] ),
        .Q(image_height_read_reg_168[0]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[10] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[10] ),
        .Q(image_height_read_reg_168[10]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[11] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[11] ),
        .Q(image_height_read_reg_168[11]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[12] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[12] ),
        .Q(image_height_read_reg_168[12]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[13] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[13] ),
        .Q(image_height_read_reg_168[13]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[14] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[14] ),
        .Q(image_height_read_reg_168[14]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[15] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[15] ),
        .Q(image_height_read_reg_168[15]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[16] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[16] ),
        .Q(image_height_read_reg_168[16]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[17] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[17] ),
        .Q(image_height_read_reg_168[17]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[18] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[18] ),
        .Q(image_height_read_reg_168[18]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[19] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[19] ),
        .Q(image_height_read_reg_168[19]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[1] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[1] ),
        .Q(image_height_read_reg_168[1]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[20] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[20] ),
        .Q(image_height_read_reg_168[20]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[21] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[21] ),
        .Q(image_height_read_reg_168[21]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[22] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[22] ),
        .Q(image_height_read_reg_168[22]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[23] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[23] ),
        .Q(image_height_read_reg_168[23]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[24] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[24] ),
        .Q(image_height_read_reg_168[24]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[25] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[25] ),
        .Q(image_height_read_reg_168[25]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[26] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[26] ),
        .Q(image_height_read_reg_168[26]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[27] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[27] ),
        .Q(image_height_read_reg_168[27]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[28] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[28] ),
        .Q(image_height_read_reg_168[28]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[29] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[29] ),
        .Q(image_height_read_reg_168[29]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[2] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[2] ),
        .Q(image_height_read_reg_168[2]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[30] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[30] ),
        .Q(image_height_read_reg_168[30]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[31] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[31] ),
        .Q(image_height_read_reg_168[31]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[3] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[3] ),
        .Q(image_height_read_reg_168[3]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[4] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[4] ),
        .Q(image_height_read_reg_168[4]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[5] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[5] ),
        .Q(image_height_read_reg_168[5]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[6] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[6] ),
        .Q(image_height_read_reg_168[6]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[7] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[7] ),
        .Q(image_height_read_reg_168[7]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[8] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[8] ),
        .Q(image_height_read_reg_168[8]),
        .R(1'b0));
  FDRE \image_height_read_reg_168_reg[9] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_height_0_data_reg_reg_n_0_[9] ),
        .Q(image_height_read_reg_168[9]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[0] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[0]),
        .Q(\image_width_0_data_reg_reg_n_0_[0] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[10] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[10]),
        .Q(\image_width_0_data_reg_reg_n_0_[10] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[11] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[11]),
        .Q(\image_width_0_data_reg_reg_n_0_[11] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[12] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[12]),
        .Q(\image_width_0_data_reg_reg_n_0_[12] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[13] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[13]),
        .Q(\image_width_0_data_reg_reg_n_0_[13] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[14] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[14]),
        .Q(\image_width_0_data_reg_reg_n_0_[14] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[15] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[15]),
        .Q(\image_width_0_data_reg_reg_n_0_[15] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[16] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[16]),
        .Q(\image_width_0_data_reg_reg_n_0_[16] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[17] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[17]),
        .Q(\image_width_0_data_reg_reg_n_0_[17] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[18] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[18]),
        .Q(\image_width_0_data_reg_reg_n_0_[18] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[19] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[19]),
        .Q(\image_width_0_data_reg_reg_n_0_[19] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[1] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[1]),
        .Q(\image_width_0_data_reg_reg_n_0_[1] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[20] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[20]),
        .Q(\image_width_0_data_reg_reg_n_0_[20] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[21] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[21]),
        .Q(\image_width_0_data_reg_reg_n_0_[21] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[22] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[22]),
        .Q(\image_width_0_data_reg_reg_n_0_[22] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[23] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[23]),
        .Q(\image_width_0_data_reg_reg_n_0_[23] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[24] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[24]),
        .Q(\image_width_0_data_reg_reg_n_0_[24] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[25] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[25]),
        .Q(\image_width_0_data_reg_reg_n_0_[25] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[26] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[26]),
        .Q(\image_width_0_data_reg_reg_n_0_[26] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[27] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[27]),
        .Q(\image_width_0_data_reg_reg_n_0_[27] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[28] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[28]),
        .Q(\image_width_0_data_reg_reg_n_0_[28] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[29] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[29]),
        .Q(\image_width_0_data_reg_reg_n_0_[29] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[2] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[2]),
        .Q(\image_width_0_data_reg_reg_n_0_[2] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[30] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[30]),
        .Q(\image_width_0_data_reg_reg_n_0_[30] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[31] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[31]),
        .Q(\image_width_0_data_reg_reg_n_0_[31] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[3] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[3]),
        .Q(\image_width_0_data_reg_reg_n_0_[3] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[4] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[4]),
        .Q(\image_width_0_data_reg_reg_n_0_[4] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[5] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[5]),
        .Q(\image_width_0_data_reg_reg_n_0_[5] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[6] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[6]),
        .Q(\image_width_0_data_reg_reg_n_0_[6] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[7] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[7]),
        .Q(\image_width_0_data_reg_reg_n_0_[7] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[8] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[8]),
        .Q(\image_width_0_data_reg_reg_n_0_[8] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \image_width_0_data_reg_reg[9] 
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[9]),
        .Q(\image_width_0_data_reg_reg_n_0_[9] ),
        .R(1'b0));
  FDRE image_width_0_data_reg_reg__0
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[30]),
        .Q(image_width_0_data_reg_reg__0_n_0),
        .R(1'b0));
  FDRE image_width_0_data_reg_reg__1
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[29]),
        .Q(image_width_0_data_reg_reg__1_n_0),
        .R(1'b0));
  FDRE image_width_0_data_reg_reg__10
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[20]),
        .Q(image_width_0_data_reg_reg__10_n_0),
        .R(1'b0));
  FDRE image_width_0_data_reg_reg__11
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[19]),
        .Q(image_width_0_data_reg_reg__11_n_0),
        .R(1'b0));
  FDRE image_width_0_data_reg_reg__12
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[18]),
        .Q(image_width_0_data_reg_reg__12_n_0),
        .R(1'b0));
  FDRE image_width_0_data_reg_reg__13
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[17]),
        .Q(image_width_0_data_reg_reg__13_n_0),
        .R(1'b0));
  FDRE image_width_0_data_reg_reg__2
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[28]),
        .Q(image_width_0_data_reg_reg__2_n_0),
        .R(1'b0));
  FDRE image_width_0_data_reg_reg__3
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[27]),
        .Q(image_width_0_data_reg_reg__3_n_0),
        .R(1'b0));
  FDRE image_width_0_data_reg_reg__4
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[26]),
        .Q(image_width_0_data_reg_reg__4_n_0),
        .R(1'b0));
  FDRE image_width_0_data_reg_reg__5
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[25]),
        .Q(image_width_0_data_reg_reg__5_n_0),
        .R(1'b0));
  FDRE image_width_0_data_reg_reg__6
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[24]),
        .Q(image_width_0_data_reg_reg__6_n_0),
        .R(1'b0));
  FDRE image_width_0_data_reg_reg__7
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[23]),
        .Q(image_width_0_data_reg_reg__7_n_0),
        .R(1'b0));
  FDRE image_width_0_data_reg_reg__8
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[22]),
        .Q(image_width_0_data_reg_reg__8_n_0),
        .R(1'b0));
  FDRE image_width_0_data_reg_reg__9
       (.C(ap_clk),
        .CE(control_s_axi_U_n_7),
        .D(image_width[21]),
        .Q(image_width_0_data_reg_reg__9_n_0),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[0] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[0] ),
        .Q(image_width_read_reg_173[0]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[10] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[10] ),
        .Q(image_width_read_reg_173[10]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[11] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[11] ),
        .Q(image_width_read_reg_173[11]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[12] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[12] ),
        .Q(image_width_read_reg_173[12]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[13] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[13] ),
        .Q(image_width_read_reg_173[13]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[14] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[14] ),
        .Q(image_width_read_reg_173[14]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[15] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[15] ),
        .Q(image_width_read_reg_173[15]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[16] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[16] ),
        .Q(image_width_read_reg_173[16]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[17] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[17] ),
        .Q(image_width_read_reg_173[17]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[18] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[18] ),
        .Q(image_width_read_reg_173[18]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[19] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[19] ),
        .Q(image_width_read_reg_173[19]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[1] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[1] ),
        .Q(image_width_read_reg_173[1]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[20] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[20] ),
        .Q(image_width_read_reg_173[20]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[21] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[21] ),
        .Q(image_width_read_reg_173[21]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[22] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[22] ),
        .Q(image_width_read_reg_173[22]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[23] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[23] ),
        .Q(image_width_read_reg_173[23]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[24] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[24] ),
        .Q(image_width_read_reg_173[24]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[25] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[25] ),
        .Q(image_width_read_reg_173[25]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[26] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[26] ),
        .Q(image_width_read_reg_173[26]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[27] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[27] ),
        .Q(image_width_read_reg_173[27]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[28] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[28] ),
        .Q(image_width_read_reg_173[28]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[29] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[29] ),
        .Q(image_width_read_reg_173[29]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[2] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[2] ),
        .Q(image_width_read_reg_173[2]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[30] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[30] ),
        .Q(image_width_read_reg_173[30]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[31] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[31] ),
        .Q(image_width_read_reg_173[31]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[3] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[3] ),
        .Q(image_width_read_reg_173[3]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[4] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[4] ),
        .Q(image_width_read_reg_173[4]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[5] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[5] ),
        .Q(image_width_read_reg_173[5]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[6] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[6] ),
        .Q(image_width_read_reg_173[6]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[7] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[7] ),
        .Q(image_width_read_reg_173[7]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[8] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[8] ),
        .Q(image_width_read_reg_173[8]),
        .R(1'b0));
  FDRE \image_width_read_reg_173_reg[9] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state2),
        .D(\image_width_0_data_reg_reg_n_0_[9] ),
        .Q(image_width_read_reg_173[9]),
        .R(1'b0));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_mul_31ns_31ns_62_2_1 mul_31ns_31ns_62_2_1_U13
       (.D(image_width[16:0]),
        .E(control_s_axi_U_n_7),
        .Q(ap_CS_fsm_state2),
        .ap_clk(ap_clk),
        .buff0_reg_0({image_height_0_data_reg_reg__0_n_0,image_height_0_data_reg_reg__1_n_0,image_height_0_data_reg_reg__2_n_0,image_height_0_data_reg_reg__3_n_0,image_height_0_data_reg_reg__4_n_0,image_height_0_data_reg_reg__5_n_0,image_height_0_data_reg_reg__6_n_0,image_height_0_data_reg_reg__7_n_0,image_height_0_data_reg_reg__8_n_0,image_height_0_data_reg_reg__9_n_0,image_height_0_data_reg_reg__10_n_0,image_height_0_data_reg_reg__11_n_0,image_height_0_data_reg_reg__12_n_0,image_height_0_data_reg_reg__13_n_0}),
        .buff0_reg__0_0({buff0_reg__1,mul_31ns_31ns_62_2_1_U13_n_46,mul_31ns_31ns_62_2_1_U13_n_47,mul_31ns_31ns_62_2_1_U13_n_48,mul_31ns_31ns_62_2_1_U13_n_49,mul_31ns_31ns_62_2_1_U13_n_50,mul_31ns_31ns_62_2_1_U13_n_51,mul_31ns_31ns_62_2_1_U13_n_52,mul_31ns_31ns_62_2_1_U13_n_53,mul_31ns_31ns_62_2_1_U13_n_54,mul_31ns_31ns_62_2_1_U13_n_55,mul_31ns_31ns_62_2_1_U13_n_56,mul_31ns_31ns_62_2_1_U13_n_57,mul_31ns_31ns_62_2_1_U13_n_58,mul_31ns_31ns_62_2_1_U13_n_59,mul_31ns_31ns_62_2_1_U13_n_60,mul_31ns_31ns_62_2_1_U13_n_61}),
        .buff0_reg__0_1({image_width_0_data_reg_reg__0_n_0,image_width_0_data_reg_reg__1_n_0,image_width_0_data_reg_reg__2_n_0,image_width_0_data_reg_reg__3_n_0,image_width_0_data_reg_reg__4_n_0,image_width_0_data_reg_reg__5_n_0,image_width_0_data_reg_reg__6_n_0,image_width_0_data_reg_reg__7_n_0,image_width_0_data_reg_reg__8_n_0,image_width_0_data_reg_reg__9_n_0,image_width_0_data_reg_reg__10_n_0,image_width_0_data_reg_reg__11_n_0,image_width_0_data_reg_reg__12_n_0,image_width_0_data_reg_reg__13_n_0}),
        .tmp_product__19_0(image_height[16:0]),
        .tmp_product_i_3_0({\image_width_0_data_reg_reg_n_0_[31] ,\image_width_0_data_reg_reg_n_0_[30] ,\image_width_0_data_reg_reg_n_0_[29] ,\image_width_0_data_reg_reg_n_0_[28] ,\image_width_0_data_reg_reg_n_0_[27] ,\image_width_0_data_reg_reg_n_0_[26] ,\image_width_0_data_reg_reg_n_0_[25] ,\image_width_0_data_reg_reg_n_0_[24] ,\image_width_0_data_reg_reg_n_0_[23] ,\image_width_0_data_reg_reg_n_0_[22] ,\image_width_0_data_reg_reg_n_0_[21] ,\image_width_0_data_reg_reg_n_0_[20] ,\image_width_0_data_reg_reg_n_0_[19] ,\image_width_0_data_reg_reg_n_0_[18] ,\image_width_0_data_reg_reg_n_0_[17] ,\image_width_0_data_reg_reg_n_0_[16] ,\image_width_0_data_reg_reg_n_0_[15] ,\image_width_0_data_reg_reg_n_0_[14] ,\image_width_0_data_reg_reg_n_0_[13] ,\image_width_0_data_reg_reg_n_0_[12] ,\image_width_0_data_reg_reg_n_0_[11] ,\image_width_0_data_reg_reg_n_0_[10] ,\image_width_0_data_reg_reg_n_0_[9] ,\image_width_0_data_reg_reg_n_0_[8] ,\image_width_0_data_reg_reg_n_0_[7] ,\image_width_0_data_reg_reg_n_0_[6] ,\image_width_0_data_reg_reg_n_0_[5] ,\image_width_0_data_reg_reg_n_0_[4] ,\image_width_0_data_reg_reg_n_0_[3] ,\image_width_0_data_reg_reg_n_0_[2] ,\image_width_0_data_reg_reg_n_0_[1] ,\image_width_0_data_reg_reg_n_0_[0] }),
        .tmp_product_i_4_0({\image_height_0_data_reg_reg_n_0_[31] ,\image_height_0_data_reg_reg_n_0_[30] ,\image_height_0_data_reg_reg_n_0_[29] ,\image_height_0_data_reg_reg_n_0_[28] ,\image_height_0_data_reg_reg_n_0_[27] ,\image_height_0_data_reg_reg_n_0_[26] ,\image_height_0_data_reg_reg_n_0_[25] ,\image_height_0_data_reg_reg_n_0_[24] ,\image_height_0_data_reg_reg_n_0_[23] ,\image_height_0_data_reg_reg_n_0_[22] ,\image_height_0_data_reg_reg_n_0_[21] ,\image_height_0_data_reg_reg_n_0_[20] ,\image_height_0_data_reg_reg_n_0_[19] ,\image_height_0_data_reg_reg_n_0_[18] ,\image_height_0_data_reg_reg_n_0_[17] ,\image_height_0_data_reg_reg_n_0_[16] ,\image_height_0_data_reg_reg_n_0_[15] ,\image_height_0_data_reg_reg_n_0_[14] ,\image_height_0_data_reg_reg_n_0_[13] ,\image_height_0_data_reg_reg_n_0_[12] ,\image_height_0_data_reg_reg_n_0_[11] ,\image_height_0_data_reg_reg_n_0_[10] ,\image_height_0_data_reg_reg_n_0_[9] ,\image_height_0_data_reg_reg_n_0_[8] ,\image_height_0_data_reg_reg_n_0_[7] ,\image_height_0_data_reg_reg_n_0_[6] ,\image_height_0_data_reg_reg_n_0_[5] ,\image_height_0_data_reg_reg_n_0_[4] ,\image_height_0_data_reg_reg_n_0_[3] ,\image_height_0_data_reg_reg_n_0_[2] ,\image_height_0_data_reg_reg_n_0_[1] ,\image_height_0_data_reg_reg_n_0_[0] }));
  FDRE \mul_ln10_reg_199_reg[0] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(mul_31ns_31ns_62_2_1_U13_n_61),
        .Q(mul_ln10_reg_199[0]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[10] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(mul_31ns_31ns_62_2_1_U13_n_51),
        .Q(mul_ln10_reg_199[10]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[11] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(mul_31ns_31ns_62_2_1_U13_n_50),
        .Q(mul_ln10_reg_199[11]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[12] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(mul_31ns_31ns_62_2_1_U13_n_49),
        .Q(mul_ln10_reg_199[12]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[13] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(mul_31ns_31ns_62_2_1_U13_n_48),
        .Q(mul_ln10_reg_199[13]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[14] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(mul_31ns_31ns_62_2_1_U13_n_47),
        .Q(mul_ln10_reg_199[14]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[15] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(mul_31ns_31ns_62_2_1_U13_n_46),
        .Q(mul_ln10_reg_199[15]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[16] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[16]),
        .Q(mul_ln10_reg_199[16]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[17] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[17]),
        .Q(mul_ln10_reg_199[17]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[18] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[18]),
        .Q(mul_ln10_reg_199[18]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[19] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[19]),
        .Q(mul_ln10_reg_199[19]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[1] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(mul_31ns_31ns_62_2_1_U13_n_60),
        .Q(mul_ln10_reg_199[1]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[20] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[20]),
        .Q(mul_ln10_reg_199[20]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[21] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[21]),
        .Q(mul_ln10_reg_199[21]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[22] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[22]),
        .Q(mul_ln10_reg_199[22]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[23] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[23]),
        .Q(mul_ln10_reg_199[23]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[24] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[24]),
        .Q(mul_ln10_reg_199[24]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[25] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[25]),
        .Q(mul_ln10_reg_199[25]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[26] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[26]),
        .Q(mul_ln10_reg_199[26]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[27] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[27]),
        .Q(mul_ln10_reg_199[27]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[28] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[28]),
        .Q(mul_ln10_reg_199[28]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[29] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[29]),
        .Q(mul_ln10_reg_199[29]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[2] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(mul_31ns_31ns_62_2_1_U13_n_59),
        .Q(mul_ln10_reg_199[2]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[30] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[30]),
        .Q(mul_ln10_reg_199[30]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[31] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[31]),
        .Q(mul_ln10_reg_199[31]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[32] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[32]),
        .Q(mul_ln10_reg_199[32]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[33] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[33]),
        .Q(mul_ln10_reg_199[33]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[34] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[34]),
        .Q(mul_ln10_reg_199[34]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[35] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[35]),
        .Q(mul_ln10_reg_199[35]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[36] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[36]),
        .Q(mul_ln10_reg_199[36]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[37] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[37]),
        .Q(mul_ln10_reg_199[37]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[38] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[38]),
        .Q(mul_ln10_reg_199[38]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[39] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[39]),
        .Q(mul_ln10_reg_199[39]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[3] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(mul_31ns_31ns_62_2_1_U13_n_58),
        .Q(mul_ln10_reg_199[3]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[40] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[40]),
        .Q(mul_ln10_reg_199[40]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[41] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[41]),
        .Q(mul_ln10_reg_199[41]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[42] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[42]),
        .Q(mul_ln10_reg_199[42]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[43] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[43]),
        .Q(mul_ln10_reg_199[43]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[44] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[44]),
        .Q(mul_ln10_reg_199[44]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[45] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[45]),
        .Q(mul_ln10_reg_199[45]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[46] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[46]),
        .Q(mul_ln10_reg_199[46]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[47] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[47]),
        .Q(mul_ln10_reg_199[47]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[48] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[48]),
        .Q(mul_ln10_reg_199[48]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[49] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[49]),
        .Q(mul_ln10_reg_199[49]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[4] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(mul_31ns_31ns_62_2_1_U13_n_57),
        .Q(mul_ln10_reg_199[4]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[50] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[50]),
        .Q(mul_ln10_reg_199[50]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[51] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[51]),
        .Q(mul_ln10_reg_199[51]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[52] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[52]),
        .Q(mul_ln10_reg_199[52]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[53] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[53]),
        .Q(mul_ln10_reg_199[53]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[54] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[54]),
        .Q(mul_ln10_reg_199[54]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[55] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[55]),
        .Q(mul_ln10_reg_199[55]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[56] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[56]),
        .Q(mul_ln10_reg_199[56]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[57] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[57]),
        .Q(mul_ln10_reg_199[57]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[58] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[58]),
        .Q(mul_ln10_reg_199[58]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[59] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[59]),
        .Q(mul_ln10_reg_199[59]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[5] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(mul_31ns_31ns_62_2_1_U13_n_56),
        .Q(mul_ln10_reg_199[5]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[60] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[60]),
        .Q(mul_ln10_reg_199[60]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[61] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(buff0_reg__1[61]),
        .Q(mul_ln10_reg_199[61]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[6] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(mul_31ns_31ns_62_2_1_U13_n_55),
        .Q(mul_ln10_reg_199[6]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[7] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(mul_31ns_31ns_62_2_1_U13_n_54),
        .Q(mul_ln10_reg_199[7]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[8] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(mul_31ns_31ns_62_2_1_U13_n_53),
        .Q(mul_ln10_reg_199[8]),
        .R(1'b0));
  FDRE \mul_ln10_reg_199_reg[9] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state4),
        .D(mul_31ns_31ns_62_2_1_U13_n_52),
        .Q(mul_ln10_reg_199[9]),
        .R(1'b0));
  FDRE \output_stream_TKEEP_reg_reg[0] 
       (.C(ap_clk),
        .CE(input_stream_TREADY_int_regslice),
        .D(input_stream_TKEEP_int_regslice[0]),
        .Q(output_stream_TKEEP_reg[0]),
        .R(1'b0));
  FDRE \output_stream_TKEEP_reg_reg[1] 
       (.C(ap_clk),
        .CE(input_stream_TREADY_int_regslice),
        .D(input_stream_TKEEP_int_regslice[1]),
        .Q(output_stream_TKEEP_reg[1]),
        .R(1'b0));
  FDRE \output_stream_TKEEP_reg_reg[2] 
       (.C(ap_clk),
        .CE(input_stream_TREADY_int_regslice),
        .D(input_stream_TKEEP_int_regslice[2]),
        .Q(output_stream_TKEEP_reg[2]),
        .R(1'b0));
  FDRE \output_stream_TKEEP_reg_reg[3] 
       (.C(ap_clk),
        .CE(input_stream_TREADY_int_regslice),
        .D(input_stream_TKEEP_int_regslice[3]),
        .Q(output_stream_TKEEP_reg[3]),
        .R(1'b0));
  FDRE \output_stream_TLAST_reg_reg[0] 
       (.C(ap_clk),
        .CE(input_stream_TREADY_int_regslice),
        .D(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TLAST),
        .Q(output_stream_TLAST_reg),
        .R(1'b0));
  FDRE \output_stream_TSTRB_reg_reg[0] 
       (.C(ap_clk),
        .CE(input_stream_TREADY_int_regslice),
        .D(input_stream_TSTRB_int_regslice[0]),
        .Q(output_stream_TSTRB_reg[0]),
        .R(1'b0));
  FDRE \output_stream_TSTRB_reg_reg[1] 
       (.C(ap_clk),
        .CE(input_stream_TREADY_int_regslice),
        .D(input_stream_TSTRB_int_regslice[1]),
        .Q(output_stream_TSTRB_reg[1]),
        .R(1'b0));
  FDRE \output_stream_TSTRB_reg_reg[2] 
       (.C(ap_clk),
        .CE(input_stream_TREADY_int_regslice),
        .D(input_stream_TSTRB_int_regslice[2]),
        .Q(output_stream_TSTRB_reg[2]),
        .R(1'b0));
  FDRE \output_stream_TSTRB_reg_reg[3] 
       (.C(ap_clk),
        .CE(input_stream_TREADY_int_regslice),
        .D(input_stream_TSTRB_int_regslice[3]),
        .Q(output_stream_TSTRB_reg[3]),
        .R(1'b0));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_regslice_both regslice_both_input_stream_V_data_V_U
       (.Q(input_stream_TVALID_int_regslice),
        .ack_in_t_reg_0(input_stream_TREADY),
        .ap_clk(ap_clk),
        .ap_rst_n_inv(ap_rst_n_inv),
        .input_stream_TREADY_int_regslice(input_stream_TREADY_int_regslice),
        .input_stream_TVALID(input_stream_TVALID),
        .output_stream_TREADY_int_regslice(output_stream_TREADY_int_regslice),
        .\state_reg[1]_0 (ap_CS_fsm_state6),
        .\state_reg[1]_1 (grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_n_0));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_regslice_both__parameterized0 regslice_both_input_stream_V_keep_V_U
       (.Q(input_stream_TKEEP_int_regslice),
        .ap_clk(ap_clk),
        .ap_rst_n_inv(ap_rst_n_inv),
        .input_stream_TKEEP(input_stream_TKEEP),
        .input_stream_TREADY_int_regslice(input_stream_TREADY_int_regslice),
        .input_stream_TVALID(input_stream_TVALID));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_regslice_both__parameterized0_0 regslice_both_input_stream_V_strb_V_U
       (.Q(input_stream_TSTRB_int_regslice),
        .ap_clk(ap_clk),
        .ap_rst_n_inv(ap_rst_n_inv),
        .input_stream_TREADY_int_regslice(input_stream_TREADY_int_regslice),
        .input_stream_TSTRB(input_stream_TSTRB),
        .input_stream_TVALID(input_stream_TVALID));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_regslice_both_1 regslice_both_output_stream_V_data_V_U
       (.D(ap_NS_fsm[0]),
        .E(ap_block_pp0_stage0_subdone),
        .Q({ap_CS_fsm_state7,ap_CS_fsm_state6,\ap_CS_fsm_reg_n_0_[0] }),
        .\ap_CS_fsm_reg[6] (regslice_both_output_stream_V_data_V_U_n_5),
        .ap_clk(ap_clk),
        .ap_rst_n_inv(ap_rst_n_inv),
        .ap_start(ap_start),
        .\data_p2_reg[1]_0 (input_stream_TVALID_int_regslice),
        .\data_p2_reg[1]_1 (grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_n_0),
        .grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .output_stream_TDATA({\^output_stream_TDATA [3],\^output_stream_TDATA [1]}),
        .output_stream_TREADY(output_stream_TREADY),
        .output_stream_TREADY_int_regslice(output_stream_TREADY_int_regslice),
        .output_stream_TVALID(output_stream_TVALID));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_regslice_both__parameterized0_2 regslice_both_output_stream_V_keep_V_U
       (.D(input_stream_TKEEP_int_regslice),
        .E(load_p2),
        .ack_in_t_reg_0(regslice_both_output_stream_V_keep_V_U_n_0),
        .ap_clk(ap_clk),
        .ap_rst_n_inv(ap_rst_n_inv),
        .grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .input_stream_TREADY_int_regslice(input_stream_TREADY_int_regslice),
        .output_stream_TKEEP(output_stream_TKEEP),
        .output_stream_TKEEP_reg(output_stream_TKEEP_reg),
        .output_stream_TREADY(output_stream_TREADY));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_regslice_both__parameterized1 regslice_both_output_stream_V_last_V_U
       (.\FSM_sequential_state_reg[1]_0 (regslice_both_output_stream_V_last_V_U_n_3),
        .ack_in_t_reg_0(regslice_both_output_stream_V_last_V_U_n_0),
        .ap_clk(ap_clk),
        .ap_rst_n_inv(ap_rst_n_inv),
        .\data_p1_reg[0]_0 (grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_n_1),
        .data_p2(data_p2),
        .\data_p2_reg[0]_0 (grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_n_9),
        .grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .output_stream_TLAST(output_stream_TLAST),
        .output_stream_TREADY(output_stream_TREADY));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_regslice_both__parameterized0_3 regslice_both_output_stream_V_strb_V_U
       (.D(input_stream_TSTRB_int_regslice),
        .E(load_p2_0),
        .ack_in_t_reg_0(regslice_both_output_stream_V_strb_V_U_n_0),
        .ap_clk(ap_clk),
        .ap_rst_n_inv(ap_rst_n_inv),
        .grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .input_stream_TREADY_int_regslice(input_stream_TREADY_int_regslice),
        .output_stream_TREADY(output_stream_TREADY),
        .output_stream_TSTRB(output_stream_TSTRB),
        .output_stream_TSTRB_reg(output_stream_TSTRB_reg));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[0]_i_1 
       (.I0(image_width_read_reg_173[0]),
        .O(sub7_fu_162_p2[0]));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[12]_i_2 
       (.I0(image_width_read_reg_173[12]),
        .O(\sub7_reg_209[12]_i_2_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[12]_i_3 
       (.I0(image_width_read_reg_173[11]),
        .O(\sub7_reg_209[12]_i_3_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[12]_i_4 
       (.I0(image_width_read_reg_173[10]),
        .O(\sub7_reg_209[12]_i_4_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[12]_i_5 
       (.I0(image_width_read_reg_173[9]),
        .O(\sub7_reg_209[12]_i_5_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[16]_i_2 
       (.I0(image_width_read_reg_173[16]),
        .O(\sub7_reg_209[16]_i_2_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[16]_i_3 
       (.I0(image_width_read_reg_173[15]),
        .O(\sub7_reg_209[16]_i_3_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[16]_i_4 
       (.I0(image_width_read_reg_173[14]),
        .O(\sub7_reg_209[16]_i_4_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[16]_i_5 
       (.I0(image_width_read_reg_173[13]),
        .O(\sub7_reg_209[16]_i_5_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[20]_i_2 
       (.I0(image_width_read_reg_173[20]),
        .O(\sub7_reg_209[20]_i_2_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[20]_i_3 
       (.I0(image_width_read_reg_173[19]),
        .O(\sub7_reg_209[20]_i_3_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[20]_i_4 
       (.I0(image_width_read_reg_173[18]),
        .O(\sub7_reg_209[20]_i_4_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[20]_i_5 
       (.I0(image_width_read_reg_173[17]),
        .O(\sub7_reg_209[20]_i_5_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[24]_i_2 
       (.I0(image_width_read_reg_173[24]),
        .O(\sub7_reg_209[24]_i_2_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[24]_i_3 
       (.I0(image_width_read_reg_173[23]),
        .O(\sub7_reg_209[24]_i_3_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[24]_i_4 
       (.I0(image_width_read_reg_173[22]),
        .O(\sub7_reg_209[24]_i_4_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[24]_i_5 
       (.I0(image_width_read_reg_173[21]),
        .O(\sub7_reg_209[24]_i_5_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[28]_i_2 
       (.I0(image_width_read_reg_173[28]),
        .O(\sub7_reg_209[28]_i_2_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[28]_i_3 
       (.I0(image_width_read_reg_173[27]),
        .O(\sub7_reg_209[28]_i_3_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[28]_i_4 
       (.I0(image_width_read_reg_173[26]),
        .O(\sub7_reg_209[28]_i_4_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[28]_i_5 
       (.I0(image_width_read_reg_173[25]),
        .O(\sub7_reg_209[28]_i_5_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[31]_i_2 
       (.I0(image_width_read_reg_173[31]),
        .O(\sub7_reg_209[31]_i_2_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[31]_i_3 
       (.I0(image_width_read_reg_173[30]),
        .O(\sub7_reg_209[31]_i_3_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[31]_i_4 
       (.I0(image_width_read_reg_173[29]),
        .O(\sub7_reg_209[31]_i_4_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[4]_i_2 
       (.I0(image_width_read_reg_173[4]),
        .O(\sub7_reg_209[4]_i_2_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[4]_i_3 
       (.I0(image_width_read_reg_173[3]),
        .O(\sub7_reg_209[4]_i_3_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[4]_i_4 
       (.I0(image_width_read_reg_173[2]),
        .O(\sub7_reg_209[4]_i_4_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[4]_i_5 
       (.I0(image_width_read_reg_173[1]),
        .O(\sub7_reg_209[4]_i_5_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[8]_i_2 
       (.I0(image_width_read_reg_173[8]),
        .O(\sub7_reg_209[8]_i_2_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[8]_i_3 
       (.I0(image_width_read_reg_173[7]),
        .O(\sub7_reg_209[8]_i_3_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[8]_i_4 
       (.I0(image_width_read_reg_173[6]),
        .O(\sub7_reg_209[8]_i_4_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub7_reg_209[8]_i_5 
       (.I0(image_width_read_reg_173[5]),
        .O(\sub7_reg_209[8]_i_5_n_0 ));
  FDRE \sub7_reg_209_reg[0] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[0]),
        .Q(sub7_reg_209[0]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[10] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[10]),
        .Q(sub7_reg_209[10]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[11] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[11]),
        .Q(sub7_reg_209[11]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[12] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[12]),
        .Q(sub7_reg_209[12]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \sub7_reg_209_reg[12]_i_1 
       (.CI(\sub7_reg_209_reg[8]_i_1_n_0 ),
        .CO({\sub7_reg_209_reg[12]_i_1_n_0 ,\sub7_reg_209_reg[12]_i_1_n_1 ,\sub7_reg_209_reg[12]_i_1_n_2 ,\sub7_reg_209_reg[12]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI(image_width_read_reg_173[12:9]),
        .O(sub7_fu_162_p2[12:9]),
        .S({\sub7_reg_209[12]_i_2_n_0 ,\sub7_reg_209[12]_i_3_n_0 ,\sub7_reg_209[12]_i_4_n_0 ,\sub7_reg_209[12]_i_5_n_0 }));
  FDRE \sub7_reg_209_reg[13] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[13]),
        .Q(sub7_reg_209[13]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[14] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[14]),
        .Q(sub7_reg_209[14]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[15] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[15]),
        .Q(sub7_reg_209[15]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[16] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[16]),
        .Q(sub7_reg_209[16]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \sub7_reg_209_reg[16]_i_1 
       (.CI(\sub7_reg_209_reg[12]_i_1_n_0 ),
        .CO({\sub7_reg_209_reg[16]_i_1_n_0 ,\sub7_reg_209_reg[16]_i_1_n_1 ,\sub7_reg_209_reg[16]_i_1_n_2 ,\sub7_reg_209_reg[16]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI(image_width_read_reg_173[16:13]),
        .O(sub7_fu_162_p2[16:13]),
        .S({\sub7_reg_209[16]_i_2_n_0 ,\sub7_reg_209[16]_i_3_n_0 ,\sub7_reg_209[16]_i_4_n_0 ,\sub7_reg_209[16]_i_5_n_0 }));
  FDRE \sub7_reg_209_reg[17] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[17]),
        .Q(sub7_reg_209[17]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[18] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[18]),
        .Q(sub7_reg_209[18]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[19] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[19]),
        .Q(sub7_reg_209[19]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[1] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[1]),
        .Q(sub7_reg_209[1]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[20] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[20]),
        .Q(sub7_reg_209[20]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \sub7_reg_209_reg[20]_i_1 
       (.CI(\sub7_reg_209_reg[16]_i_1_n_0 ),
        .CO({\sub7_reg_209_reg[20]_i_1_n_0 ,\sub7_reg_209_reg[20]_i_1_n_1 ,\sub7_reg_209_reg[20]_i_1_n_2 ,\sub7_reg_209_reg[20]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI(image_width_read_reg_173[20:17]),
        .O(sub7_fu_162_p2[20:17]),
        .S({\sub7_reg_209[20]_i_2_n_0 ,\sub7_reg_209[20]_i_3_n_0 ,\sub7_reg_209[20]_i_4_n_0 ,\sub7_reg_209[20]_i_5_n_0 }));
  FDRE \sub7_reg_209_reg[21] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[21]),
        .Q(sub7_reg_209[21]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[22] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[22]),
        .Q(sub7_reg_209[22]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[23] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[23]),
        .Q(sub7_reg_209[23]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[24] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[24]),
        .Q(sub7_reg_209[24]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \sub7_reg_209_reg[24]_i_1 
       (.CI(\sub7_reg_209_reg[20]_i_1_n_0 ),
        .CO({\sub7_reg_209_reg[24]_i_1_n_0 ,\sub7_reg_209_reg[24]_i_1_n_1 ,\sub7_reg_209_reg[24]_i_1_n_2 ,\sub7_reg_209_reg[24]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI(image_width_read_reg_173[24:21]),
        .O(sub7_fu_162_p2[24:21]),
        .S({\sub7_reg_209[24]_i_2_n_0 ,\sub7_reg_209[24]_i_3_n_0 ,\sub7_reg_209[24]_i_4_n_0 ,\sub7_reg_209[24]_i_5_n_0 }));
  FDRE \sub7_reg_209_reg[25] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[25]),
        .Q(sub7_reg_209[25]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[26] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[26]),
        .Q(sub7_reg_209[26]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[27] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[27]),
        .Q(sub7_reg_209[27]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[28] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[28]),
        .Q(sub7_reg_209[28]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \sub7_reg_209_reg[28]_i_1 
       (.CI(\sub7_reg_209_reg[24]_i_1_n_0 ),
        .CO({\sub7_reg_209_reg[28]_i_1_n_0 ,\sub7_reg_209_reg[28]_i_1_n_1 ,\sub7_reg_209_reg[28]_i_1_n_2 ,\sub7_reg_209_reg[28]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI(image_width_read_reg_173[28:25]),
        .O(sub7_fu_162_p2[28:25]),
        .S({\sub7_reg_209[28]_i_2_n_0 ,\sub7_reg_209[28]_i_3_n_0 ,\sub7_reg_209[28]_i_4_n_0 ,\sub7_reg_209[28]_i_5_n_0 }));
  FDRE \sub7_reg_209_reg[29] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[29]),
        .Q(sub7_reg_209[29]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[2] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[2]),
        .Q(sub7_reg_209[2]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[30] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[30]),
        .Q(sub7_reg_209[30]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[31] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[31]),
        .Q(sub7_reg_209[31]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \sub7_reg_209_reg[31]_i_1 
       (.CI(\sub7_reg_209_reg[28]_i_1_n_0 ),
        .CO({\NLW_sub7_reg_209_reg[31]_i_1_CO_UNCONNECTED [3:2],\sub7_reg_209_reg[31]_i_1_n_2 ,\sub7_reg_209_reg[31]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,image_width_read_reg_173[30:29]}),
        .O({\NLW_sub7_reg_209_reg[31]_i_1_O_UNCONNECTED [3],sub7_fu_162_p2[31:29]}),
        .S({1'b0,\sub7_reg_209[31]_i_2_n_0 ,\sub7_reg_209[31]_i_3_n_0 ,\sub7_reg_209[31]_i_4_n_0 }));
  FDRE \sub7_reg_209_reg[3] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[3]),
        .Q(sub7_reg_209[3]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[4] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[4]),
        .Q(sub7_reg_209[4]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \sub7_reg_209_reg[4]_i_1 
       (.CI(1'b0),
        .CO({\sub7_reg_209_reg[4]_i_1_n_0 ,\sub7_reg_209_reg[4]_i_1_n_1 ,\sub7_reg_209_reg[4]_i_1_n_2 ,\sub7_reg_209_reg[4]_i_1_n_3 }),
        .CYINIT(image_width_read_reg_173[0]),
        .DI(image_width_read_reg_173[4:1]),
        .O(sub7_fu_162_p2[4:1]),
        .S({\sub7_reg_209[4]_i_2_n_0 ,\sub7_reg_209[4]_i_3_n_0 ,\sub7_reg_209[4]_i_4_n_0 ,\sub7_reg_209[4]_i_5_n_0 }));
  FDRE \sub7_reg_209_reg[5] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[5]),
        .Q(sub7_reg_209[5]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[6] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[6]),
        .Q(sub7_reg_209[6]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[7] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[7]),
        .Q(sub7_reg_209[7]),
        .R(1'b0));
  FDRE \sub7_reg_209_reg[8] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[8]),
        .Q(sub7_reg_209[8]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \sub7_reg_209_reg[8]_i_1 
       (.CI(\sub7_reg_209_reg[4]_i_1_n_0 ),
        .CO({\sub7_reg_209_reg[8]_i_1_n_0 ,\sub7_reg_209_reg[8]_i_1_n_1 ,\sub7_reg_209_reg[8]_i_1_n_2 ,\sub7_reg_209_reg[8]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI(image_width_read_reg_173[8:5]),
        .O(sub7_fu_162_p2[8:5]),
        .S({\sub7_reg_209[8]_i_2_n_0 ,\sub7_reg_209[8]_i_3_n_0 ,\sub7_reg_209[8]_i_4_n_0 ,\sub7_reg_209[8]_i_5_n_0 }));
  FDRE \sub7_reg_209_reg[9] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub7_fu_162_p2[9]),
        .Q(sub7_reg_209[9]),
        .R(1'b0));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[0]_i_1 
       (.I0(image_height_read_reg_168[0]),
        .O(sub_fu_156_p2[0]));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[12]_i_2 
       (.I0(image_height_read_reg_168[12]),
        .O(\sub_reg_204[12]_i_2_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[12]_i_3 
       (.I0(image_height_read_reg_168[11]),
        .O(\sub_reg_204[12]_i_3_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[12]_i_4 
       (.I0(image_height_read_reg_168[10]),
        .O(\sub_reg_204[12]_i_4_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[12]_i_5 
       (.I0(image_height_read_reg_168[9]),
        .O(\sub_reg_204[12]_i_5_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[16]_i_2 
       (.I0(image_height_read_reg_168[16]),
        .O(\sub_reg_204[16]_i_2_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[16]_i_3 
       (.I0(image_height_read_reg_168[15]),
        .O(\sub_reg_204[16]_i_3_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[16]_i_4 
       (.I0(image_height_read_reg_168[14]),
        .O(\sub_reg_204[16]_i_4_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[16]_i_5 
       (.I0(image_height_read_reg_168[13]),
        .O(\sub_reg_204[16]_i_5_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[20]_i_2 
       (.I0(image_height_read_reg_168[20]),
        .O(\sub_reg_204[20]_i_2_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[20]_i_3 
       (.I0(image_height_read_reg_168[19]),
        .O(\sub_reg_204[20]_i_3_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[20]_i_4 
       (.I0(image_height_read_reg_168[18]),
        .O(\sub_reg_204[20]_i_4_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[20]_i_5 
       (.I0(image_height_read_reg_168[17]),
        .O(\sub_reg_204[20]_i_5_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[24]_i_2 
       (.I0(image_height_read_reg_168[24]),
        .O(\sub_reg_204[24]_i_2_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[24]_i_3 
       (.I0(image_height_read_reg_168[23]),
        .O(\sub_reg_204[24]_i_3_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[24]_i_4 
       (.I0(image_height_read_reg_168[22]),
        .O(\sub_reg_204[24]_i_4_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[24]_i_5 
       (.I0(image_height_read_reg_168[21]),
        .O(\sub_reg_204[24]_i_5_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[28]_i_2 
       (.I0(image_height_read_reg_168[28]),
        .O(\sub_reg_204[28]_i_2_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[28]_i_3 
       (.I0(image_height_read_reg_168[27]),
        .O(\sub_reg_204[28]_i_3_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[28]_i_4 
       (.I0(image_height_read_reg_168[26]),
        .O(\sub_reg_204[28]_i_4_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[28]_i_5 
       (.I0(image_height_read_reg_168[25]),
        .O(\sub_reg_204[28]_i_5_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[31]_i_2 
       (.I0(image_height_read_reg_168[31]),
        .O(\sub_reg_204[31]_i_2_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[31]_i_3 
       (.I0(image_height_read_reg_168[30]),
        .O(\sub_reg_204[31]_i_3_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[31]_i_4 
       (.I0(image_height_read_reg_168[29]),
        .O(\sub_reg_204[31]_i_4_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[4]_i_2 
       (.I0(image_height_read_reg_168[4]),
        .O(\sub_reg_204[4]_i_2_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[4]_i_3 
       (.I0(image_height_read_reg_168[3]),
        .O(\sub_reg_204[4]_i_3_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[4]_i_4 
       (.I0(image_height_read_reg_168[2]),
        .O(\sub_reg_204[4]_i_4_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[4]_i_5 
       (.I0(image_height_read_reg_168[1]),
        .O(\sub_reg_204[4]_i_5_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[8]_i_2 
       (.I0(image_height_read_reg_168[8]),
        .O(\sub_reg_204[8]_i_2_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[8]_i_3 
       (.I0(image_height_read_reg_168[7]),
        .O(\sub_reg_204[8]_i_3_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[8]_i_4 
       (.I0(image_height_read_reg_168[6]),
        .O(\sub_reg_204[8]_i_4_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \sub_reg_204[8]_i_5 
       (.I0(image_height_read_reg_168[5]),
        .O(\sub_reg_204[8]_i_5_n_0 ));
  FDRE \sub_reg_204_reg[0] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[0]),
        .Q(sub_reg_204[0]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[10] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[10]),
        .Q(sub_reg_204[10]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[11] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[11]),
        .Q(sub_reg_204[11]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[12] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[12]),
        .Q(sub_reg_204[12]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \sub_reg_204_reg[12]_i_1 
       (.CI(\sub_reg_204_reg[8]_i_1_n_0 ),
        .CO({\sub_reg_204_reg[12]_i_1_n_0 ,\sub_reg_204_reg[12]_i_1_n_1 ,\sub_reg_204_reg[12]_i_1_n_2 ,\sub_reg_204_reg[12]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI(image_height_read_reg_168[12:9]),
        .O(sub_fu_156_p2[12:9]),
        .S({\sub_reg_204[12]_i_2_n_0 ,\sub_reg_204[12]_i_3_n_0 ,\sub_reg_204[12]_i_4_n_0 ,\sub_reg_204[12]_i_5_n_0 }));
  FDRE \sub_reg_204_reg[13] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[13]),
        .Q(sub_reg_204[13]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[14] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[14]),
        .Q(sub_reg_204[14]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[15] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[15]),
        .Q(sub_reg_204[15]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[16] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[16]),
        .Q(sub_reg_204[16]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \sub_reg_204_reg[16]_i_1 
       (.CI(\sub_reg_204_reg[12]_i_1_n_0 ),
        .CO({\sub_reg_204_reg[16]_i_1_n_0 ,\sub_reg_204_reg[16]_i_1_n_1 ,\sub_reg_204_reg[16]_i_1_n_2 ,\sub_reg_204_reg[16]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI(image_height_read_reg_168[16:13]),
        .O(sub_fu_156_p2[16:13]),
        .S({\sub_reg_204[16]_i_2_n_0 ,\sub_reg_204[16]_i_3_n_0 ,\sub_reg_204[16]_i_4_n_0 ,\sub_reg_204[16]_i_5_n_0 }));
  FDRE \sub_reg_204_reg[17] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[17]),
        .Q(sub_reg_204[17]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[18] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[18]),
        .Q(sub_reg_204[18]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[19] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[19]),
        .Q(sub_reg_204[19]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[1] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[1]),
        .Q(sub_reg_204[1]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[20] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[20]),
        .Q(sub_reg_204[20]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \sub_reg_204_reg[20]_i_1 
       (.CI(\sub_reg_204_reg[16]_i_1_n_0 ),
        .CO({\sub_reg_204_reg[20]_i_1_n_0 ,\sub_reg_204_reg[20]_i_1_n_1 ,\sub_reg_204_reg[20]_i_1_n_2 ,\sub_reg_204_reg[20]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI(image_height_read_reg_168[20:17]),
        .O(sub_fu_156_p2[20:17]),
        .S({\sub_reg_204[20]_i_2_n_0 ,\sub_reg_204[20]_i_3_n_0 ,\sub_reg_204[20]_i_4_n_0 ,\sub_reg_204[20]_i_5_n_0 }));
  FDRE \sub_reg_204_reg[21] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[21]),
        .Q(sub_reg_204[21]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[22] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[22]),
        .Q(sub_reg_204[22]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[23] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[23]),
        .Q(sub_reg_204[23]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[24] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[24]),
        .Q(sub_reg_204[24]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \sub_reg_204_reg[24]_i_1 
       (.CI(\sub_reg_204_reg[20]_i_1_n_0 ),
        .CO({\sub_reg_204_reg[24]_i_1_n_0 ,\sub_reg_204_reg[24]_i_1_n_1 ,\sub_reg_204_reg[24]_i_1_n_2 ,\sub_reg_204_reg[24]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI(image_height_read_reg_168[24:21]),
        .O(sub_fu_156_p2[24:21]),
        .S({\sub_reg_204[24]_i_2_n_0 ,\sub_reg_204[24]_i_3_n_0 ,\sub_reg_204[24]_i_4_n_0 ,\sub_reg_204[24]_i_5_n_0 }));
  FDRE \sub_reg_204_reg[25] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[25]),
        .Q(sub_reg_204[25]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[26] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[26]),
        .Q(sub_reg_204[26]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[27] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[27]),
        .Q(sub_reg_204[27]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[28] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[28]),
        .Q(sub_reg_204[28]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \sub_reg_204_reg[28]_i_1 
       (.CI(\sub_reg_204_reg[24]_i_1_n_0 ),
        .CO({\sub_reg_204_reg[28]_i_1_n_0 ,\sub_reg_204_reg[28]_i_1_n_1 ,\sub_reg_204_reg[28]_i_1_n_2 ,\sub_reg_204_reg[28]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI(image_height_read_reg_168[28:25]),
        .O(sub_fu_156_p2[28:25]),
        .S({\sub_reg_204[28]_i_2_n_0 ,\sub_reg_204[28]_i_3_n_0 ,\sub_reg_204[28]_i_4_n_0 ,\sub_reg_204[28]_i_5_n_0 }));
  FDRE \sub_reg_204_reg[29] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[29]),
        .Q(sub_reg_204[29]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[2] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[2]),
        .Q(sub_reg_204[2]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[30] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[30]),
        .Q(sub_reg_204[30]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[31] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[31]),
        .Q(sub_reg_204[31]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \sub_reg_204_reg[31]_i_1 
       (.CI(\sub_reg_204_reg[28]_i_1_n_0 ),
        .CO({\NLW_sub_reg_204_reg[31]_i_1_CO_UNCONNECTED [3:2],\sub_reg_204_reg[31]_i_1_n_2 ,\sub_reg_204_reg[31]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,image_height_read_reg_168[30:29]}),
        .O({\NLW_sub_reg_204_reg[31]_i_1_O_UNCONNECTED [3],sub_fu_156_p2[31:29]}),
        .S({1'b0,\sub_reg_204[31]_i_2_n_0 ,\sub_reg_204[31]_i_3_n_0 ,\sub_reg_204[31]_i_4_n_0 }));
  FDRE \sub_reg_204_reg[3] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[3]),
        .Q(sub_reg_204[3]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[4] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[4]),
        .Q(sub_reg_204[4]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \sub_reg_204_reg[4]_i_1 
       (.CI(1'b0),
        .CO({\sub_reg_204_reg[4]_i_1_n_0 ,\sub_reg_204_reg[4]_i_1_n_1 ,\sub_reg_204_reg[4]_i_1_n_2 ,\sub_reg_204_reg[4]_i_1_n_3 }),
        .CYINIT(image_height_read_reg_168[0]),
        .DI(image_height_read_reg_168[4:1]),
        .O(sub_fu_156_p2[4:1]),
        .S({\sub_reg_204[4]_i_2_n_0 ,\sub_reg_204[4]_i_3_n_0 ,\sub_reg_204[4]_i_4_n_0 ,\sub_reg_204[4]_i_5_n_0 }));
  FDRE \sub_reg_204_reg[5] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[5]),
        .Q(sub_reg_204[5]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[6] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[6]),
        .Q(sub_reg_204[6]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[7] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[7]),
        .Q(sub_reg_204[7]),
        .R(1'b0));
  FDRE \sub_reg_204_reg[8] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[8]),
        .Q(sub_reg_204[8]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \sub_reg_204_reg[8]_i_1 
       (.CI(\sub_reg_204_reg[4]_i_1_n_0 ),
        .CO({\sub_reg_204_reg[8]_i_1_n_0 ,\sub_reg_204_reg[8]_i_1_n_1 ,\sub_reg_204_reg[8]_i_1_n_2 ,\sub_reg_204_reg[8]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI(image_height_read_reg_168[8:5]),
        .O(sub_fu_156_p2[8:5]),
        .S({\sub_reg_204[8]_i_2_n_0 ,\sub_reg_204[8]_i_3_n_0 ,\sub_reg_204[8]_i_4_n_0 ,\sub_reg_204[8]_i_5_n_0 }));
  FDRE \sub_reg_204_reg[9] 
       (.C(ap_clk),
        .CE(ap_CS_fsm_state5),
        .D(sub_fu_156_p2[9]),
        .Q(sub_reg_204[9]),
        .R(1'b0));
endmodule

module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_control_s_axi
   (ap_rst_n_inv,
    interrupt,
    \FSM_onehot_rstate_reg[1]_0 ,
    s_axi_control_RVALID,
    \FSM_onehot_wstate_reg[2]_0 ,
    D,
    ap_start,
    E,
    \int_image_height_reg[31]_0 ,
    \int_image_width_reg[31]_0 ,
    \FSM_onehot_wstate_reg[1]_0 ,
    s_axi_control_BVALID,
    s_axi_control_RDATA,
    ap_clk,
    s_axi_control_ARVALID,
    s_axi_control_RREADY,
    s_axi_control_ARADDR,
    s_axi_control_WVALID,
    s_axi_control_WDATA,
    s_axi_control_WSTRB,
    Q,
    ap_rst_n,
    s_axi_control_AWVALID,
    s_axi_control_BREADY,
    int_ap_start_reg_0,
    s_axi_control_AWADDR);
  output ap_rst_n_inv;
  output interrupt;
  output \FSM_onehot_rstate_reg[1]_0 ;
  output s_axi_control_RVALID;
  output \FSM_onehot_wstate_reg[2]_0 ;
  output [0:0]D;
  output ap_start;
  output [0:0]E;
  output [31:0]\int_image_height_reg[31]_0 ;
  output [31:0]\int_image_width_reg[31]_0 ;
  output \FSM_onehot_wstate_reg[1]_0 ;
  output s_axi_control_BVALID;
  output [31:0]s_axi_control_RDATA;
  input ap_clk;
  input s_axi_control_ARVALID;
  input s_axi_control_RREADY;
  input [5:0]s_axi_control_ARADDR;
  input s_axi_control_WVALID;
  input [31:0]s_axi_control_WDATA;
  input [3:0]s_axi_control_WSTRB;
  input [6:0]Q;
  input ap_rst_n;
  input s_axi_control_AWVALID;
  input s_axi_control_BREADY;
  input int_ap_start_reg_0;
  input [3:0]s_axi_control_AWADDR;

  wire [0:0]D;
  wire [0:0]E;
  wire \FSM_onehot_rstate[1]_i_1_n_0 ;
  wire \FSM_onehot_rstate[2]_i_1_n_0 ;
  wire \FSM_onehot_rstate_reg[1]_0 ;
  wire \FSM_onehot_wstate[1]_i_1_n_0 ;
  wire \FSM_onehot_wstate[2]_i_1_n_0 ;
  wire \FSM_onehot_wstate[3]_i_1_n_0 ;
  wire \FSM_onehot_wstate_reg[1]_0 ;
  wire \FSM_onehot_wstate_reg[2]_0 ;
  wire [6:0]Q;
  wire \ap_CS_fsm[1]_i_2_n_0 ;
  wire ap_clk;
  wire ap_rst_n;
  wire ap_rst_n_inv;
  wire ap_start;
  wire ar_hs;
  wire auto_restart_status_i_1_n_0;
  wire auto_restart_status_reg_n_0;
  wire int_ap_idle_i_1_n_0;
  wire int_ap_ready;
  wire int_ap_ready_i_1_n_0;
  wire int_ap_start5_out;
  wire int_ap_start_i_1_n_0;
  wire int_ap_start_reg_0;
  wire int_auto_restart_i_1_n_0;
  wire int_auto_restart_i_2_n_0;
  wire int_gie_i_1_n_0;
  wire int_gie_reg_n_0;
  wire \int_ier[0]_i_1_n_0 ;
  wire \int_ier[1]_i_1_n_0 ;
  wire \int_ier[1]_i_2_n_0 ;
  wire \int_ier_reg_n_0_[0] ;
  wire [31:0]int_image_height0;
  wire \int_image_height[31]_i_1_n_0 ;
  wire [31:0]\int_image_height_reg[31]_0 ;
  wire [31:0]int_image_width0;
  wire \int_image_width[31]_i_1_n_0 ;
  wire [31:0]\int_image_width_reg[31]_0 ;
  wire int_interrupt0;
  wire int_isr7_out;
  wire \int_isr[0]_i_1_n_0 ;
  wire \int_isr[1]_i_1_n_0 ;
  wire \int_isr_reg_n_0_[0] ;
  wire \int_isr_reg_n_0_[1] ;
  wire [31:0]int_kernel_factor0;
  wire \int_kernel_factor[31]_i_1_n_0 ;
  wire \int_kernel_factor_reg_n_0_[0] ;
  wire \int_kernel_factor_reg_n_0_[10] ;
  wire \int_kernel_factor_reg_n_0_[11] ;
  wire \int_kernel_factor_reg_n_0_[12] ;
  wire \int_kernel_factor_reg_n_0_[13] ;
  wire \int_kernel_factor_reg_n_0_[14] ;
  wire \int_kernel_factor_reg_n_0_[15] ;
  wire \int_kernel_factor_reg_n_0_[16] ;
  wire \int_kernel_factor_reg_n_0_[17] ;
  wire \int_kernel_factor_reg_n_0_[18] ;
  wire \int_kernel_factor_reg_n_0_[19] ;
  wire \int_kernel_factor_reg_n_0_[1] ;
  wire \int_kernel_factor_reg_n_0_[20] ;
  wire \int_kernel_factor_reg_n_0_[21] ;
  wire \int_kernel_factor_reg_n_0_[22] ;
  wire \int_kernel_factor_reg_n_0_[23] ;
  wire \int_kernel_factor_reg_n_0_[24] ;
  wire \int_kernel_factor_reg_n_0_[25] ;
  wire \int_kernel_factor_reg_n_0_[26] ;
  wire \int_kernel_factor_reg_n_0_[27] ;
  wire \int_kernel_factor_reg_n_0_[28] ;
  wire \int_kernel_factor_reg_n_0_[29] ;
  wire \int_kernel_factor_reg_n_0_[2] ;
  wire \int_kernel_factor_reg_n_0_[30] ;
  wire \int_kernel_factor_reg_n_0_[31] ;
  wire \int_kernel_factor_reg_n_0_[3] ;
  wire \int_kernel_factor_reg_n_0_[4] ;
  wire \int_kernel_factor_reg_n_0_[5] ;
  wire \int_kernel_factor_reg_n_0_[6] ;
  wire \int_kernel_factor_reg_n_0_[7] ;
  wire \int_kernel_factor_reg_n_0_[8] ;
  wire \int_kernel_factor_reg_n_0_[9] ;
  wire int_task_ap_done;
  wire int_task_ap_done0__3;
  wire int_task_ap_done_i_1_n_0;
  wire int_task_ap_done_i_3_n_0;
  wire interrupt;
  wire p_0_in;
  wire [7:2]p_3_in;
  wire [9:0]rdata;
  wire \rdata[0]_i_2_n_0 ;
  wire \rdata[0]_i_3_n_0 ;
  wire \rdata[10]_i_1_n_0 ;
  wire \rdata[11]_i_1_n_0 ;
  wire \rdata[12]_i_1_n_0 ;
  wire \rdata[13]_i_1_n_0 ;
  wire \rdata[14]_i_1_n_0 ;
  wire \rdata[15]_i_1_n_0 ;
  wire \rdata[16]_i_1_n_0 ;
  wire \rdata[17]_i_1_n_0 ;
  wire \rdata[18]_i_1_n_0 ;
  wire \rdata[19]_i_1_n_0 ;
  wire \rdata[1]_i_2_n_0 ;
  wire \rdata[1]_i_3_n_0 ;
  wire \rdata[20]_i_1_n_0 ;
  wire \rdata[21]_i_1_n_0 ;
  wire \rdata[22]_i_1_n_0 ;
  wire \rdata[23]_i_1_n_0 ;
  wire \rdata[24]_i_1_n_0 ;
  wire \rdata[25]_i_1_n_0 ;
  wire \rdata[26]_i_1_n_0 ;
  wire \rdata[27]_i_1_n_0 ;
  wire \rdata[28]_i_1_n_0 ;
  wire \rdata[29]_i_1_n_0 ;
  wire \rdata[2]_i_2_n_0 ;
  wire \rdata[30]_i_1_n_0 ;
  wire \rdata[31]_i_1_n_0 ;
  wire \rdata[31]_i_3_n_0 ;
  wire \rdata[31]_i_4_n_0 ;
  wire \rdata[31]_i_5_n_0 ;
  wire \rdata[3]_i_2_n_0 ;
  wire \rdata[4]_i_1_n_0 ;
  wire \rdata[5]_i_1_n_0 ;
  wire \rdata[6]_i_1_n_0 ;
  wire \rdata[7]_i_2_n_0 ;
  wire \rdata[8]_i_1_n_0 ;
  wire \rdata[9]_i_2_n_0 ;
  wire \rdata[9]_i_3_n_0 ;
  wire [5:0]s_axi_control_ARADDR;
  wire s_axi_control_ARVALID;
  wire [3:0]s_axi_control_AWADDR;
  wire s_axi_control_AWVALID;
  wire s_axi_control_BREADY;
  wire s_axi_control_BVALID;
  wire [31:0]s_axi_control_RDATA;
  wire s_axi_control_RREADY;
  wire s_axi_control_RVALID;
  wire [31:0]s_axi_control_WDATA;
  wire [3:0]s_axi_control_WSTRB;
  wire s_axi_control_WVALID;
  wire waddr;
  wire \waddr_reg_n_0_[2] ;
  wire \waddr_reg_n_0_[3] ;
  wire \waddr_reg_n_0_[4] ;
  wire \waddr_reg_n_0_[5] ;

  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT4 #(
    .INIT(16'hF747)) 
    \FSM_onehot_rstate[1]_i_1 
       (.I0(s_axi_control_ARVALID),
        .I1(\FSM_onehot_rstate_reg[1]_0 ),
        .I2(s_axi_control_RVALID),
        .I3(s_axi_control_RREADY),
        .O(\FSM_onehot_rstate[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT4 #(
    .INIT(16'h8F88)) 
    \FSM_onehot_rstate[2]_i_1 
       (.I0(s_axi_control_ARVALID),
        .I1(\FSM_onehot_rstate_reg[1]_0 ),
        .I2(s_axi_control_RREADY),
        .I3(s_axi_control_RVALID),
        .O(\FSM_onehot_rstate[2]_i_1_n_0 ));
  (* FSM_ENCODED_STATES = "RDIDLE:010,RDDATA:100,iSTATE:001" *) 
  FDRE #(
    .INIT(1'b0)) 
    \FSM_onehot_rstate_reg[1] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\FSM_onehot_rstate[1]_i_1_n_0 ),
        .Q(\FSM_onehot_rstate_reg[1]_0 ),
        .R(ap_rst_n_inv));
  (* FSM_ENCODED_STATES = "RDIDLE:010,RDDATA:100,iSTATE:001" *) 
  FDRE #(
    .INIT(1'b0)) 
    \FSM_onehot_rstate_reg[2] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\FSM_onehot_rstate[2]_i_1_n_0 ),
        .Q(s_axi_control_RVALID),
        .R(ap_rst_n_inv));
  LUT5 #(
    .INIT(32'h888BFF8B)) 
    \FSM_onehot_wstate[1]_i_1 
       (.I0(s_axi_control_BREADY),
        .I1(s_axi_control_BVALID),
        .I2(\FSM_onehot_wstate_reg[2]_0 ),
        .I3(\FSM_onehot_wstate_reg[1]_0 ),
        .I4(s_axi_control_AWVALID),
        .O(\FSM_onehot_wstate[1]_i_1_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \FSM_onehot_wstate[2]_i_1 
       (.I0(s_axi_control_AWVALID),
        .I1(\FSM_onehot_wstate_reg[1]_0 ),
        .I2(s_axi_control_WVALID),
        .I3(\FSM_onehot_wstate_reg[2]_0 ),
        .O(\FSM_onehot_wstate[2]_i_1_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \FSM_onehot_wstate[3]_i_1 
       (.I0(s_axi_control_WVALID),
        .I1(\FSM_onehot_wstate_reg[2]_0 ),
        .I2(s_axi_control_BREADY),
        .I3(s_axi_control_BVALID),
        .O(\FSM_onehot_wstate[3]_i_1_n_0 ));
  (* FSM_ENCODED_STATES = "WRDATA:0100,WRRESP:1000,WRIDLE:0010,iSTATE:0001" *) 
  FDRE #(
    .INIT(1'b0)) 
    \FSM_onehot_wstate_reg[1] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\FSM_onehot_wstate[1]_i_1_n_0 ),
        .Q(\FSM_onehot_wstate_reg[1]_0 ),
        .R(ap_rst_n_inv));
  (* FSM_ENCODED_STATES = "WRDATA:0100,WRRESP:1000,WRIDLE:0010,iSTATE:0001" *) 
  FDRE #(
    .INIT(1'b0)) 
    \FSM_onehot_wstate_reg[2] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\FSM_onehot_wstate[2]_i_1_n_0 ),
        .Q(\FSM_onehot_wstate_reg[2]_0 ),
        .R(ap_rst_n_inv));
  (* FSM_ENCODED_STATES = "WRDATA:0100,WRRESP:1000,WRIDLE:0010,iSTATE:0001" *) 
  FDRE #(
    .INIT(1'b0)) 
    \FSM_onehot_wstate_reg[3] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\FSM_onehot_wstate[3]_i_1_n_0 ),
        .Q(s_axi_control_BVALID),
        .R(ap_rst_n_inv));
  LUT1 #(
    .INIT(2'h1)) 
    ack_in_t_i_1
       (.I0(ap_rst_n),
        .O(ap_rst_n_inv));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT5 #(
    .INIT(32'h00001000)) 
    \ap_CS_fsm[1]_i_1 
       (.I0(Q[6]),
        .I1(Q[1]),
        .I2(Q[0]),
        .I3(ap_start),
        .I4(\ap_CS_fsm[1]_i_2_n_0 ),
        .O(D));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \ap_CS_fsm[1]_i_2 
       (.I0(Q[4]),
        .I1(Q[5]),
        .I2(Q[2]),
        .I3(Q[3]),
        .O(\ap_CS_fsm[1]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hEFAA)) 
    auto_restart_status_i_1
       (.I0(p_3_in[7]),
        .I1(ap_start),
        .I2(Q[0]),
        .I3(auto_restart_status_reg_n_0),
        .O(auto_restart_status_i_1_n_0));
  FDRE #(
    .INIT(1'b0)) 
    auto_restart_status_reg
       (.C(ap_clk),
        .CE(1'b1),
        .D(auto_restart_status_i_1_n_0),
        .Q(auto_restart_status_reg_n_0),
        .R(ap_rst_n_inv));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT2 #(
    .INIT(4'h2)) 
    int_ap_idle_i_1
       (.I0(Q[0]),
        .I1(ap_start),
        .O(int_ap_idle_i_1_n_0));
  FDRE #(
    .INIT(1'b0)) 
    int_ap_idle_reg
       (.C(ap_clk),
        .CE(1'b1),
        .D(int_ap_idle_i_1_n_0),
        .Q(p_3_in[2]),
        .R(ap_rst_n_inv));
  LUT4 #(
    .INIT(16'h4F44)) 
    int_ap_ready_i_1
       (.I0(p_3_in[7]),
        .I1(int_ap_start_reg_0),
        .I2(int_task_ap_done0__3),
        .I3(int_ap_ready),
        .O(int_ap_ready_i_1_n_0));
  FDRE #(
    .INIT(1'b0)) 
    int_ap_ready_reg
       (.C(ap_clk),
        .CE(1'b1),
        .D(int_ap_ready_i_1_n_0),
        .Q(int_ap_ready),
        .R(ap_rst_n_inv));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT4 #(
    .INIT(16'hFBF8)) 
    int_ap_start_i_1
       (.I0(p_3_in[7]),
        .I1(int_ap_start_reg_0),
        .I2(int_ap_start5_out),
        .I3(ap_start),
        .O(int_ap_start_i_1_n_0));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT4 #(
    .INIT(16'h0080)) 
    int_ap_start_i_2
       (.I0(s_axi_control_WDATA[0]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(int_auto_restart_i_2_n_0),
        .I3(\waddr_reg_n_0_[2] ),
        .O(int_ap_start5_out));
  FDRE #(
    .INIT(1'b0)) 
    int_ap_start_reg
       (.C(ap_clk),
        .CE(1'b1),
        .D(int_ap_start_i_1_n_0),
        .Q(ap_start),
        .R(ap_rst_n_inv));
  LUT5 #(
    .INIT(32'hEFFF2000)) 
    int_auto_restart_i_1
       (.I0(s_axi_control_WDATA[7]),
        .I1(\waddr_reg_n_0_[2] ),
        .I2(int_auto_restart_i_2_n_0),
        .I3(s_axi_control_WSTRB[0]),
        .I4(p_3_in[7]),
        .O(int_auto_restart_i_1_n_0));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT5 #(
    .INIT(32'h00000040)) 
    int_auto_restart_i_2
       (.I0(\waddr_reg_n_0_[4] ),
        .I1(\FSM_onehot_wstate_reg[2]_0 ),
        .I2(s_axi_control_WVALID),
        .I3(\waddr_reg_n_0_[5] ),
        .I4(\waddr_reg_n_0_[3] ),
        .O(int_auto_restart_i_2_n_0));
  FDRE #(
    .INIT(1'b0)) 
    int_auto_restart_reg
       (.C(ap_clk),
        .CE(1'b1),
        .D(int_auto_restart_i_1_n_0),
        .Q(p_3_in[7]),
        .R(ap_rst_n_inv));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT5 #(
    .INIT(32'hBFFF8000)) 
    int_gie_i_1
       (.I0(s_axi_control_WDATA[0]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(int_auto_restart_i_2_n_0),
        .I3(\waddr_reg_n_0_[2] ),
        .I4(int_gie_reg_n_0),
        .O(int_gie_i_1_n_0));
  FDRE #(
    .INIT(1'b0)) 
    int_gie_reg
       (.C(ap_clk),
        .CE(1'b1),
        .D(int_gie_i_1_n_0),
        .Q(int_gie_reg_n_0),
        .R(ap_rst_n_inv));
  LUT5 #(
    .INIT(32'hFBFF0800)) 
    \int_ier[0]_i_1 
       (.I0(s_axi_control_WDATA[0]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\waddr_reg_n_0_[2] ),
        .I3(\int_ier[1]_i_2_n_0 ),
        .I4(\int_ier_reg_n_0_[0] ),
        .O(\int_ier[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT5 #(
    .INIT(32'hFBFF0800)) 
    \int_ier[1]_i_1 
       (.I0(s_axi_control_WDATA[1]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\waddr_reg_n_0_[2] ),
        .I3(\int_ier[1]_i_2_n_0 ),
        .I4(p_0_in),
        .O(\int_ier[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT5 #(
    .INIT(32'h00002000)) 
    \int_ier[1]_i_2 
       (.I0(\waddr_reg_n_0_[3] ),
        .I1(\waddr_reg_n_0_[4] ),
        .I2(\FSM_onehot_wstate_reg[2]_0 ),
        .I3(s_axi_control_WVALID),
        .I4(\waddr_reg_n_0_[5] ),
        .O(\int_ier[1]_i_2_n_0 ));
  FDRE #(
    .INIT(1'b0)) 
    \int_ier_reg[0] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\int_ier[0]_i_1_n_0 ),
        .Q(\int_ier_reg_n_0_[0] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_ier_reg[1] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\int_ier[1]_i_1_n_0 ),
        .Q(p_0_in),
        .R(ap_rst_n_inv));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[0]_i_1 
       (.I0(s_axi_control_WDATA[0]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_image_height_reg[31]_0 [0]),
        .O(int_image_height0[0]));
  (* SOFT_HLUTNM = "soft_lutpair21" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[10]_i_1 
       (.I0(s_axi_control_WDATA[10]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_image_height_reg[31]_0 [10]),
        .O(int_image_height0[10]));
  (* SOFT_HLUTNM = "soft_lutpair25" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[11]_i_1 
       (.I0(s_axi_control_WDATA[11]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_image_height_reg[31]_0 [11]),
        .O(int_image_height0[11]));
  (* SOFT_HLUTNM = "soft_lutpair25" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[12]_i_1 
       (.I0(s_axi_control_WDATA[12]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_image_height_reg[31]_0 [12]),
        .O(int_image_height0[12]));
  (* SOFT_HLUTNM = "soft_lutpair28" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[13]_i_1 
       (.I0(s_axi_control_WDATA[13]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_image_height_reg[31]_0 [13]),
        .O(int_image_height0[13]));
  (* SOFT_HLUTNM = "soft_lutpair28" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[14]_i_1 
       (.I0(s_axi_control_WDATA[14]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_image_height_reg[31]_0 [14]),
        .O(int_image_height0[14]));
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[15]_i_1 
       (.I0(s_axi_control_WDATA[15]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_image_height_reg[31]_0 [15]),
        .O(int_image_height0[15]));
  (* SOFT_HLUTNM = "soft_lutpair32" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[16]_i_1 
       (.I0(s_axi_control_WDATA[16]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_image_height_reg[31]_0 [16]),
        .O(int_image_height0[16]));
  (* SOFT_HLUTNM = "soft_lutpair32" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[17]_i_1 
       (.I0(s_axi_control_WDATA[17]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_image_height_reg[31]_0 [17]),
        .O(int_image_height0[17]));
  (* SOFT_HLUTNM = "soft_lutpair35" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[18]_i_1 
       (.I0(s_axi_control_WDATA[18]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_image_height_reg[31]_0 [18]),
        .O(int_image_height0[18]));
  (* SOFT_HLUTNM = "soft_lutpair35" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[19]_i_1 
       (.I0(s_axi_control_WDATA[19]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_image_height_reg[31]_0 [19]),
        .O(int_image_height0[19]));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[1]_i_1 
       (.I0(s_axi_control_WDATA[1]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_image_height_reg[31]_0 [1]),
        .O(int_image_height0[1]));
  (* SOFT_HLUTNM = "soft_lutpair38" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[20]_i_1 
       (.I0(s_axi_control_WDATA[20]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_image_height_reg[31]_0 [20]),
        .O(int_image_height0[20]));
  (* SOFT_HLUTNM = "soft_lutpair38" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[21]_i_1 
       (.I0(s_axi_control_WDATA[21]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_image_height_reg[31]_0 [21]),
        .O(int_image_height0[21]));
  (* SOFT_HLUTNM = "soft_lutpair41" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[22]_i_1 
       (.I0(s_axi_control_WDATA[22]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_image_height_reg[31]_0 [22]),
        .O(int_image_height0[22]));
  (* SOFT_HLUTNM = "soft_lutpair41" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[23]_i_1 
       (.I0(s_axi_control_WDATA[23]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_image_height_reg[31]_0 [23]),
        .O(int_image_height0[23]));
  (* SOFT_HLUTNM = "soft_lutpair44" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[24]_i_1 
       (.I0(s_axi_control_WDATA[24]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_image_height_reg[31]_0 [24]),
        .O(int_image_height0[24]));
  (* SOFT_HLUTNM = "soft_lutpair44" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[25]_i_1 
       (.I0(s_axi_control_WDATA[25]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_image_height_reg[31]_0 [25]),
        .O(int_image_height0[25]));
  (* SOFT_HLUTNM = "soft_lutpair47" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[26]_i_1 
       (.I0(s_axi_control_WDATA[26]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_image_height_reg[31]_0 [26]),
        .O(int_image_height0[26]));
  (* SOFT_HLUTNM = "soft_lutpair47" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[27]_i_1 
       (.I0(s_axi_control_WDATA[27]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_image_height_reg[31]_0 [27]),
        .O(int_image_height0[27]));
  (* SOFT_HLUTNM = "soft_lutpair50" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[28]_i_1 
       (.I0(s_axi_control_WDATA[28]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_image_height_reg[31]_0 [28]),
        .O(int_image_height0[28]));
  (* SOFT_HLUTNM = "soft_lutpair50" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[29]_i_1 
       (.I0(s_axi_control_WDATA[29]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_image_height_reg[31]_0 [29]),
        .O(int_image_height0[29]));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[2]_i_1 
       (.I0(s_axi_control_WDATA[2]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_image_height_reg[31]_0 [2]),
        .O(int_image_height0[2]));
  (* SOFT_HLUTNM = "soft_lutpair53" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[30]_i_1 
       (.I0(s_axi_control_WDATA[30]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_image_height_reg[31]_0 [30]),
        .O(int_image_height0[30]));
  LUT6 #(
    .INIT(64'h0000400000000000)) 
    \int_image_height[31]_i_1 
       (.I0(\waddr_reg_n_0_[2] ),
        .I1(\waddr_reg_n_0_[3] ),
        .I2(\FSM_onehot_wstate_reg[2]_0 ),
        .I3(s_axi_control_WVALID),
        .I4(\waddr_reg_n_0_[5] ),
        .I5(\waddr_reg_n_0_[4] ),
        .O(\int_image_height[31]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair53" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[31]_i_2 
       (.I0(s_axi_control_WDATA[31]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_image_height_reg[31]_0 [31]),
        .O(int_image_height0[31]));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[3]_i_1 
       (.I0(s_axi_control_WDATA[3]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_image_height_reg[31]_0 [3]),
        .O(int_image_height0[3]));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[4]_i_1 
       (.I0(s_axi_control_WDATA[4]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_image_height_reg[31]_0 [4]),
        .O(int_image_height0[4]));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[5]_i_1 
       (.I0(s_axi_control_WDATA[5]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_image_height_reg[31]_0 [5]),
        .O(int_image_height0[5]));
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[6]_i_1 
       (.I0(s_axi_control_WDATA[6]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_image_height_reg[31]_0 [6]),
        .O(int_image_height0[6]));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[7]_i_1 
       (.I0(s_axi_control_WDATA[7]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_image_height_reg[31]_0 [7]),
        .O(int_image_height0[7]));
  (* SOFT_HLUTNM = "soft_lutpair21" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[8]_i_1 
       (.I0(s_axi_control_WDATA[8]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_image_height_reg[31]_0 [8]),
        .O(int_image_height0[8]));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_height[9]_i_1 
       (.I0(s_axi_control_WDATA[9]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_image_height_reg[31]_0 [9]),
        .O(int_image_height0[9]));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[0] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[0]),
        .Q(\int_image_height_reg[31]_0 [0]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[10] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[10]),
        .Q(\int_image_height_reg[31]_0 [10]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[11] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[11]),
        .Q(\int_image_height_reg[31]_0 [11]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[12] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[12]),
        .Q(\int_image_height_reg[31]_0 [12]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[13] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[13]),
        .Q(\int_image_height_reg[31]_0 [13]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[14] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[14]),
        .Q(\int_image_height_reg[31]_0 [14]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[15] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[15]),
        .Q(\int_image_height_reg[31]_0 [15]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[16] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[16]),
        .Q(\int_image_height_reg[31]_0 [16]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[17] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[17]),
        .Q(\int_image_height_reg[31]_0 [17]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[18] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[18]),
        .Q(\int_image_height_reg[31]_0 [18]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[19] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[19]),
        .Q(\int_image_height_reg[31]_0 [19]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[1] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[1]),
        .Q(\int_image_height_reg[31]_0 [1]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[20] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[20]),
        .Q(\int_image_height_reg[31]_0 [20]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[21] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[21]),
        .Q(\int_image_height_reg[31]_0 [21]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[22] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[22]),
        .Q(\int_image_height_reg[31]_0 [22]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[23] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[23]),
        .Q(\int_image_height_reg[31]_0 [23]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[24] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[24]),
        .Q(\int_image_height_reg[31]_0 [24]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[25] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[25]),
        .Q(\int_image_height_reg[31]_0 [25]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[26] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[26]),
        .Q(\int_image_height_reg[31]_0 [26]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[27] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[27]),
        .Q(\int_image_height_reg[31]_0 [27]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[28] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[28]),
        .Q(\int_image_height_reg[31]_0 [28]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[29] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[29]),
        .Q(\int_image_height_reg[31]_0 [29]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[2] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[2]),
        .Q(\int_image_height_reg[31]_0 [2]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[30] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[30]),
        .Q(\int_image_height_reg[31]_0 [30]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[31] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[31]),
        .Q(\int_image_height_reg[31]_0 [31]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[3] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[3]),
        .Q(\int_image_height_reg[31]_0 [3]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[4] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[4]),
        .Q(\int_image_height_reg[31]_0 [4]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[5] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[5]),
        .Q(\int_image_height_reg[31]_0 [5]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[6] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[6]),
        .Q(\int_image_height_reg[31]_0 [6]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[7] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[7]),
        .Q(\int_image_height_reg[31]_0 [7]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[8] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[8]),
        .Q(\int_image_height_reg[31]_0 [8]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_height_reg[9] 
       (.C(ap_clk),
        .CE(\int_image_height[31]_i_1_n_0 ),
        .D(int_image_height0[9]),
        .Q(\int_image_height_reg[31]_0 [9]),
        .R(ap_rst_n_inv));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[0]_i_1 
       (.I0(s_axi_control_WDATA[0]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_image_width_reg[31]_0 [0]),
        .O(int_image_width0[0]));
  (* SOFT_HLUTNM = "soft_lutpair24" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[10]_i_1 
       (.I0(s_axi_control_WDATA[10]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_image_width_reg[31]_0 [10]),
        .O(int_image_width0[10]));
  (* SOFT_HLUTNM = "soft_lutpair24" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[11]_i_1 
       (.I0(s_axi_control_WDATA[11]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_image_width_reg[31]_0 [11]),
        .O(int_image_width0[11]));
  (* SOFT_HLUTNM = "soft_lutpair27" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[12]_i_1 
       (.I0(s_axi_control_WDATA[12]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_image_width_reg[31]_0 [12]),
        .O(int_image_width0[12]));
  (* SOFT_HLUTNM = "soft_lutpair27" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[13]_i_1 
       (.I0(s_axi_control_WDATA[13]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_image_width_reg[31]_0 [13]),
        .O(int_image_width0[13]));
  (* SOFT_HLUTNM = "soft_lutpair30" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[14]_i_1 
       (.I0(s_axi_control_WDATA[14]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_image_width_reg[31]_0 [14]),
        .O(int_image_width0[14]));
  (* SOFT_HLUTNM = "soft_lutpair30" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[15]_i_1 
       (.I0(s_axi_control_WDATA[15]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_image_width_reg[31]_0 [15]),
        .O(int_image_width0[15]));
  (* SOFT_HLUTNM = "soft_lutpair33" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[16]_i_1 
       (.I0(s_axi_control_WDATA[16]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_image_width_reg[31]_0 [16]),
        .O(int_image_width0[16]));
  (* SOFT_HLUTNM = "soft_lutpair33" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[17]_i_1 
       (.I0(s_axi_control_WDATA[17]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_image_width_reg[31]_0 [17]),
        .O(int_image_width0[17]));
  (* SOFT_HLUTNM = "soft_lutpair36" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[18]_i_1 
       (.I0(s_axi_control_WDATA[18]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_image_width_reg[31]_0 [18]),
        .O(int_image_width0[18]));
  (* SOFT_HLUTNM = "soft_lutpair36" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[19]_i_1 
       (.I0(s_axi_control_WDATA[19]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_image_width_reg[31]_0 [19]),
        .O(int_image_width0[19]));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[1]_i_1 
       (.I0(s_axi_control_WDATA[1]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_image_width_reg[31]_0 [1]),
        .O(int_image_width0[1]));
  (* SOFT_HLUTNM = "soft_lutpair39" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[20]_i_1 
       (.I0(s_axi_control_WDATA[20]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_image_width_reg[31]_0 [20]),
        .O(int_image_width0[20]));
  (* SOFT_HLUTNM = "soft_lutpair39" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[21]_i_1 
       (.I0(s_axi_control_WDATA[21]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_image_width_reg[31]_0 [21]),
        .O(int_image_width0[21]));
  (* SOFT_HLUTNM = "soft_lutpair42" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[22]_i_1 
       (.I0(s_axi_control_WDATA[22]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_image_width_reg[31]_0 [22]),
        .O(int_image_width0[22]));
  (* SOFT_HLUTNM = "soft_lutpair42" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[23]_i_1 
       (.I0(s_axi_control_WDATA[23]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_image_width_reg[31]_0 [23]),
        .O(int_image_width0[23]));
  (* SOFT_HLUTNM = "soft_lutpair45" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[24]_i_1 
       (.I0(s_axi_control_WDATA[24]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_image_width_reg[31]_0 [24]),
        .O(int_image_width0[24]));
  (* SOFT_HLUTNM = "soft_lutpair45" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[25]_i_1 
       (.I0(s_axi_control_WDATA[25]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_image_width_reg[31]_0 [25]),
        .O(int_image_width0[25]));
  (* SOFT_HLUTNM = "soft_lutpair48" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[26]_i_1 
       (.I0(s_axi_control_WDATA[26]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_image_width_reg[31]_0 [26]),
        .O(int_image_width0[26]));
  (* SOFT_HLUTNM = "soft_lutpair48" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[27]_i_1 
       (.I0(s_axi_control_WDATA[27]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_image_width_reg[31]_0 [27]),
        .O(int_image_width0[27]));
  (* SOFT_HLUTNM = "soft_lutpair51" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[28]_i_1 
       (.I0(s_axi_control_WDATA[28]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_image_width_reg[31]_0 [28]),
        .O(int_image_width0[28]));
  (* SOFT_HLUTNM = "soft_lutpair51" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[29]_i_1 
       (.I0(s_axi_control_WDATA[29]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_image_width_reg[31]_0 [29]),
        .O(int_image_width0[29]));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[2]_i_1 
       (.I0(s_axi_control_WDATA[2]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_image_width_reg[31]_0 [2]),
        .O(int_image_width0[2]));
  (* SOFT_HLUTNM = "soft_lutpair54" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[30]_i_1 
       (.I0(s_axi_control_WDATA[30]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_image_width_reg[31]_0 [30]),
        .O(int_image_width0[30]));
  LUT6 #(
    .INIT(64'h0000100000000000)) 
    \int_image_width[31]_i_1 
       (.I0(\waddr_reg_n_0_[2] ),
        .I1(\waddr_reg_n_0_[3] ),
        .I2(\FSM_onehot_wstate_reg[2]_0 ),
        .I3(s_axi_control_WVALID),
        .I4(\waddr_reg_n_0_[5] ),
        .I5(\waddr_reg_n_0_[4] ),
        .O(\int_image_width[31]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair54" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[31]_i_2 
       (.I0(s_axi_control_WDATA[31]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_image_width_reg[31]_0 [31]),
        .O(int_image_width0[31]));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[3]_i_1 
       (.I0(s_axi_control_WDATA[3]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_image_width_reg[31]_0 [3]),
        .O(int_image_width0[3]));
  (* SOFT_HLUTNM = "soft_lutpair17" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[4]_i_1 
       (.I0(s_axi_control_WDATA[4]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_image_width_reg[31]_0 [4]),
        .O(int_image_width0[4]));
  (* SOFT_HLUTNM = "soft_lutpair17" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[5]_i_1 
       (.I0(s_axi_control_WDATA[5]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_image_width_reg[31]_0 [5]),
        .O(int_image_width0[5]));
  (* SOFT_HLUTNM = "soft_lutpair19" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[6]_i_1 
       (.I0(s_axi_control_WDATA[6]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_image_width_reg[31]_0 [6]),
        .O(int_image_width0[6]));
  (* SOFT_HLUTNM = "soft_lutpair19" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[7]_i_1 
       (.I0(s_axi_control_WDATA[7]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_image_width_reg[31]_0 [7]),
        .O(int_image_width0[7]));
  (* SOFT_HLUTNM = "soft_lutpair22" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[8]_i_1 
       (.I0(s_axi_control_WDATA[8]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_image_width_reg[31]_0 [8]),
        .O(int_image_width0[8]));
  (* SOFT_HLUTNM = "soft_lutpair22" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_image_width[9]_i_1 
       (.I0(s_axi_control_WDATA[9]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_image_width_reg[31]_0 [9]),
        .O(int_image_width0[9]));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[0] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[0]),
        .Q(\int_image_width_reg[31]_0 [0]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[10] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[10]),
        .Q(\int_image_width_reg[31]_0 [10]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[11] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[11]),
        .Q(\int_image_width_reg[31]_0 [11]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[12] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[12]),
        .Q(\int_image_width_reg[31]_0 [12]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[13] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[13]),
        .Q(\int_image_width_reg[31]_0 [13]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[14] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[14]),
        .Q(\int_image_width_reg[31]_0 [14]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[15] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[15]),
        .Q(\int_image_width_reg[31]_0 [15]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[16] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[16]),
        .Q(\int_image_width_reg[31]_0 [16]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[17] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[17]),
        .Q(\int_image_width_reg[31]_0 [17]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[18] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[18]),
        .Q(\int_image_width_reg[31]_0 [18]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[19] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[19]),
        .Q(\int_image_width_reg[31]_0 [19]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[1] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[1]),
        .Q(\int_image_width_reg[31]_0 [1]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[20] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[20]),
        .Q(\int_image_width_reg[31]_0 [20]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[21] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[21]),
        .Q(\int_image_width_reg[31]_0 [21]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[22] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[22]),
        .Q(\int_image_width_reg[31]_0 [22]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[23] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[23]),
        .Q(\int_image_width_reg[31]_0 [23]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[24] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[24]),
        .Q(\int_image_width_reg[31]_0 [24]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[25] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[25]),
        .Q(\int_image_width_reg[31]_0 [25]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[26] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[26]),
        .Q(\int_image_width_reg[31]_0 [26]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[27] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[27]),
        .Q(\int_image_width_reg[31]_0 [27]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[28] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[28]),
        .Q(\int_image_width_reg[31]_0 [28]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[29] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[29]),
        .Q(\int_image_width_reg[31]_0 [29]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[2] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[2]),
        .Q(\int_image_width_reg[31]_0 [2]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[30] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[30]),
        .Q(\int_image_width_reg[31]_0 [30]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[31] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[31]),
        .Q(\int_image_width_reg[31]_0 [31]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[3] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[3]),
        .Q(\int_image_width_reg[31]_0 [3]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[4] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[4]),
        .Q(\int_image_width_reg[31]_0 [4]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[5] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[5]),
        .Q(\int_image_width_reg[31]_0 [5]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[6] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[6]),
        .Q(\int_image_width_reg[31]_0 [6]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[7] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[7]),
        .Q(\int_image_width_reg[31]_0 [7]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[8] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[8]),
        .Q(\int_image_width_reg[31]_0 [8]),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_image_width_reg[9] 
       (.C(ap_clk),
        .CE(\int_image_width[31]_i_1_n_0 ),
        .D(int_image_width0[9]),
        .Q(\int_image_width_reg[31]_0 [9]),
        .R(ap_rst_n_inv));
  LUT3 #(
    .INIT(8'hA8)) 
    int_interrupt_i_1
       (.I0(int_gie_reg_n_0),
        .I1(\int_isr_reg_n_0_[1] ),
        .I2(\int_isr_reg_n_0_[0] ),
        .O(int_interrupt0));
  FDRE #(
    .INIT(1'b0)) 
    int_interrupt_reg
       (.C(ap_clk),
        .CE(1'b1),
        .D(int_interrupt0),
        .Q(interrupt),
        .R(ap_rst_n_inv));
  LUT5 #(
    .INIT(32'hF777F888)) 
    \int_isr[0]_i_1 
       (.I0(s_axi_control_WDATA[0]),
        .I1(int_isr7_out),
        .I2(\int_ier_reg_n_0_[0] ),
        .I3(int_ap_start_reg_0),
        .I4(\int_isr_reg_n_0_[0] ),
        .O(\int_isr[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT3 #(
    .INIT(8'h80)) 
    \int_isr[0]_i_2 
       (.I0(s_axi_control_WSTRB[0]),
        .I1(\int_ier[1]_i_2_n_0 ),
        .I2(\waddr_reg_n_0_[2] ),
        .O(int_isr7_out));
  LUT5 #(
    .INIT(32'hF777F888)) 
    \int_isr[1]_i_1 
       (.I0(s_axi_control_WDATA[1]),
        .I1(int_isr7_out),
        .I2(p_0_in),
        .I3(int_ap_start_reg_0),
        .I4(\int_isr_reg_n_0_[1] ),
        .O(\int_isr[1]_i_1_n_0 ));
  FDRE #(
    .INIT(1'b0)) 
    \int_isr_reg[0] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\int_isr[0]_i_1_n_0 ),
        .Q(\int_isr_reg_n_0_[0] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_isr_reg[1] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\int_isr[1]_i_1_n_0 ),
        .Q(\int_isr_reg_n_0_[1] ),
        .R(ap_rst_n_inv));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[0]_i_1 
       (.I0(s_axi_control_WDATA[0]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_kernel_factor_reg_n_0_[0] ),
        .O(int_kernel_factor0[0]));
  (* SOFT_HLUTNM = "soft_lutpair23" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[10]_i_1 
       (.I0(s_axi_control_WDATA[10]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_kernel_factor_reg_n_0_[10] ),
        .O(int_kernel_factor0[10]));
  (* SOFT_HLUTNM = "soft_lutpair23" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[11]_i_1 
       (.I0(s_axi_control_WDATA[11]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_kernel_factor_reg_n_0_[11] ),
        .O(int_kernel_factor0[11]));
  (* SOFT_HLUTNM = "soft_lutpair26" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[12]_i_1 
       (.I0(s_axi_control_WDATA[12]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_kernel_factor_reg_n_0_[12] ),
        .O(int_kernel_factor0[12]));
  (* SOFT_HLUTNM = "soft_lutpair26" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[13]_i_1 
       (.I0(s_axi_control_WDATA[13]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_kernel_factor_reg_n_0_[13] ),
        .O(int_kernel_factor0[13]));
  (* SOFT_HLUTNM = "soft_lutpair29" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[14]_i_1 
       (.I0(s_axi_control_WDATA[14]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_kernel_factor_reg_n_0_[14] ),
        .O(int_kernel_factor0[14]));
  (* SOFT_HLUTNM = "soft_lutpair29" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[15]_i_1 
       (.I0(s_axi_control_WDATA[15]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_kernel_factor_reg_n_0_[15] ),
        .O(int_kernel_factor0[15]));
  (* SOFT_HLUTNM = "soft_lutpair31" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[16]_i_1 
       (.I0(s_axi_control_WDATA[16]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_kernel_factor_reg_n_0_[16] ),
        .O(int_kernel_factor0[16]));
  (* SOFT_HLUTNM = "soft_lutpair31" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[17]_i_1 
       (.I0(s_axi_control_WDATA[17]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_kernel_factor_reg_n_0_[17] ),
        .O(int_kernel_factor0[17]));
  (* SOFT_HLUTNM = "soft_lutpair34" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[18]_i_1 
       (.I0(s_axi_control_WDATA[18]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_kernel_factor_reg_n_0_[18] ),
        .O(int_kernel_factor0[18]));
  (* SOFT_HLUTNM = "soft_lutpair34" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[19]_i_1 
       (.I0(s_axi_control_WDATA[19]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_kernel_factor_reg_n_0_[19] ),
        .O(int_kernel_factor0[19]));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[1]_i_1 
       (.I0(s_axi_control_WDATA[1]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_kernel_factor_reg_n_0_[1] ),
        .O(int_kernel_factor0[1]));
  (* SOFT_HLUTNM = "soft_lutpair37" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[20]_i_1 
       (.I0(s_axi_control_WDATA[20]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_kernel_factor_reg_n_0_[20] ),
        .O(int_kernel_factor0[20]));
  (* SOFT_HLUTNM = "soft_lutpair37" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[21]_i_1 
       (.I0(s_axi_control_WDATA[21]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_kernel_factor_reg_n_0_[21] ),
        .O(int_kernel_factor0[21]));
  (* SOFT_HLUTNM = "soft_lutpair40" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[22]_i_1 
       (.I0(s_axi_control_WDATA[22]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_kernel_factor_reg_n_0_[22] ),
        .O(int_kernel_factor0[22]));
  (* SOFT_HLUTNM = "soft_lutpair40" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[23]_i_1 
       (.I0(s_axi_control_WDATA[23]),
        .I1(s_axi_control_WSTRB[2]),
        .I2(\int_kernel_factor_reg_n_0_[23] ),
        .O(int_kernel_factor0[23]));
  (* SOFT_HLUTNM = "soft_lutpair43" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[24]_i_1 
       (.I0(s_axi_control_WDATA[24]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_kernel_factor_reg_n_0_[24] ),
        .O(int_kernel_factor0[24]));
  (* SOFT_HLUTNM = "soft_lutpair43" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[25]_i_1 
       (.I0(s_axi_control_WDATA[25]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_kernel_factor_reg_n_0_[25] ),
        .O(int_kernel_factor0[25]));
  (* SOFT_HLUTNM = "soft_lutpair46" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[26]_i_1 
       (.I0(s_axi_control_WDATA[26]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_kernel_factor_reg_n_0_[26] ),
        .O(int_kernel_factor0[26]));
  (* SOFT_HLUTNM = "soft_lutpair46" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[27]_i_1 
       (.I0(s_axi_control_WDATA[27]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_kernel_factor_reg_n_0_[27] ),
        .O(int_kernel_factor0[27]));
  (* SOFT_HLUTNM = "soft_lutpair49" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[28]_i_1 
       (.I0(s_axi_control_WDATA[28]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_kernel_factor_reg_n_0_[28] ),
        .O(int_kernel_factor0[28]));
  (* SOFT_HLUTNM = "soft_lutpair49" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[29]_i_1 
       (.I0(s_axi_control_WDATA[29]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_kernel_factor_reg_n_0_[29] ),
        .O(int_kernel_factor0[29]));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[2]_i_1 
       (.I0(s_axi_control_WDATA[2]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_kernel_factor_reg_n_0_[2] ),
        .O(int_kernel_factor0[2]));
  (* SOFT_HLUTNM = "soft_lutpair52" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[30]_i_1 
       (.I0(s_axi_control_WDATA[30]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_kernel_factor_reg_n_0_[30] ),
        .O(int_kernel_factor0[30]));
  LUT6 #(
    .INIT(64'h0000000000002000)) 
    \int_kernel_factor[31]_i_1 
       (.I0(\waddr_reg_n_0_[5] ),
        .I1(\waddr_reg_n_0_[4] ),
        .I2(\FSM_onehot_wstate_reg[2]_0 ),
        .I3(s_axi_control_WVALID),
        .I4(\waddr_reg_n_0_[3] ),
        .I5(\waddr_reg_n_0_[2] ),
        .O(\int_kernel_factor[31]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair52" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[31]_i_2 
       (.I0(s_axi_control_WDATA[31]),
        .I1(s_axi_control_WSTRB[3]),
        .I2(\int_kernel_factor_reg_n_0_[31] ),
        .O(int_kernel_factor0[31]));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[3]_i_1 
       (.I0(s_axi_control_WDATA[3]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_kernel_factor_reg_n_0_[3] ),
        .O(int_kernel_factor0[3]));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[4]_i_1 
       (.I0(s_axi_control_WDATA[4]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_kernel_factor_reg_n_0_[4] ),
        .O(int_kernel_factor0[4]));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[5]_i_1 
       (.I0(s_axi_control_WDATA[5]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_kernel_factor_reg_n_0_[5] ),
        .O(int_kernel_factor0[5]));
  (* SOFT_HLUTNM = "soft_lutpair18" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[6]_i_1 
       (.I0(s_axi_control_WDATA[6]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_kernel_factor_reg_n_0_[6] ),
        .O(int_kernel_factor0[6]));
  (* SOFT_HLUTNM = "soft_lutpair18" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[7]_i_1 
       (.I0(s_axi_control_WDATA[7]),
        .I1(s_axi_control_WSTRB[0]),
        .I2(\int_kernel_factor_reg_n_0_[7] ),
        .O(int_kernel_factor0[7]));
  (* SOFT_HLUTNM = "soft_lutpair20" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[8]_i_1 
       (.I0(s_axi_control_WDATA[8]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_kernel_factor_reg_n_0_[8] ),
        .O(int_kernel_factor0[8]));
  (* SOFT_HLUTNM = "soft_lutpair20" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \int_kernel_factor[9]_i_1 
       (.I0(s_axi_control_WDATA[9]),
        .I1(s_axi_control_WSTRB[1]),
        .I2(\int_kernel_factor_reg_n_0_[9] ),
        .O(int_kernel_factor0[9]));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[0] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[0]),
        .Q(\int_kernel_factor_reg_n_0_[0] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[10] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[10]),
        .Q(\int_kernel_factor_reg_n_0_[10] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[11] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[11]),
        .Q(\int_kernel_factor_reg_n_0_[11] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[12] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[12]),
        .Q(\int_kernel_factor_reg_n_0_[12] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[13] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[13]),
        .Q(\int_kernel_factor_reg_n_0_[13] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[14] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[14]),
        .Q(\int_kernel_factor_reg_n_0_[14] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[15] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[15]),
        .Q(\int_kernel_factor_reg_n_0_[15] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[16] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[16]),
        .Q(\int_kernel_factor_reg_n_0_[16] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[17] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[17]),
        .Q(\int_kernel_factor_reg_n_0_[17] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[18] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[18]),
        .Q(\int_kernel_factor_reg_n_0_[18] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[19] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[19]),
        .Q(\int_kernel_factor_reg_n_0_[19] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[1] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[1]),
        .Q(\int_kernel_factor_reg_n_0_[1] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[20] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[20]),
        .Q(\int_kernel_factor_reg_n_0_[20] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[21] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[21]),
        .Q(\int_kernel_factor_reg_n_0_[21] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[22] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[22]),
        .Q(\int_kernel_factor_reg_n_0_[22] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[23] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[23]),
        .Q(\int_kernel_factor_reg_n_0_[23] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[24] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[24]),
        .Q(\int_kernel_factor_reg_n_0_[24] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[25] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[25]),
        .Q(\int_kernel_factor_reg_n_0_[25] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[26] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[26]),
        .Q(\int_kernel_factor_reg_n_0_[26] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[27] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[27]),
        .Q(\int_kernel_factor_reg_n_0_[27] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[28] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[28]),
        .Q(\int_kernel_factor_reg_n_0_[28] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[29] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[29]),
        .Q(\int_kernel_factor_reg_n_0_[29] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[2] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[2]),
        .Q(\int_kernel_factor_reg_n_0_[2] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[30] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[30]),
        .Q(\int_kernel_factor_reg_n_0_[30] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[31] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[31]),
        .Q(\int_kernel_factor_reg_n_0_[31] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[3] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[3]),
        .Q(\int_kernel_factor_reg_n_0_[3] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[4] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[4]),
        .Q(\int_kernel_factor_reg_n_0_[4] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[5] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[5]),
        .Q(\int_kernel_factor_reg_n_0_[5] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[6] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[6]),
        .Q(\int_kernel_factor_reg_n_0_[6] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[7] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[7]),
        .Q(\int_kernel_factor_reg_n_0_[7] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[8] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[8]),
        .Q(\int_kernel_factor_reg_n_0_[8] ),
        .R(ap_rst_n_inv));
  FDRE #(
    .INIT(1'b0)) 
    \int_kernel_factor_reg[9] 
       (.C(ap_clk),
        .CE(\int_kernel_factor[31]_i_1_n_0 ),
        .D(int_kernel_factor0[9]),
        .Q(\int_kernel_factor_reg_n_0_[9] ),
        .R(ap_rst_n_inv));
  LUT6 #(
    .INIT(64'h7520FFFF75207520)) 
    int_task_ap_done_i_1
       (.I0(auto_restart_status_reg_n_0),
        .I1(p_3_in[2]),
        .I2(int_ap_idle_i_1_n_0),
        .I3(int_ap_start_reg_0),
        .I4(int_task_ap_done0__3),
        .I5(int_task_ap_done),
        .O(int_task_ap_done_i_1_n_0));
  LUT6 #(
    .INIT(64'h0000000100000000)) 
    int_task_ap_done_i_2
       (.I0(s_axi_control_ARADDR[2]),
        .I1(s_axi_control_ARADDR[3]),
        .I2(int_task_ap_done_i_3_n_0),
        .I3(s_axi_control_ARADDR[4]),
        .I4(s_axi_control_ARADDR[5]),
        .I5(ar_hs),
        .O(int_task_ap_done0__3));
  LUT2 #(
    .INIT(4'hE)) 
    int_task_ap_done_i_3
       (.I0(s_axi_control_ARADDR[1]),
        .I1(s_axi_control_ARADDR[0]),
        .O(int_task_ap_done_i_3_n_0));
  FDRE #(
    .INIT(1'b0)) 
    int_task_ap_done_reg
       (.C(ap_clk),
        .CE(1'b1),
        .D(int_task_ap_done_i_1_n_0),
        .Q(int_task_ap_done),
        .R(ap_rst_n_inv));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \rdata[0]_i_2 
       (.I0(\int_isr_reg_n_0_[0] ),
        .I1(\int_ier_reg_n_0_[0] ),
        .I2(\rdata[31]_i_4_n_0 ),
        .I3(int_gie_reg_n_0),
        .I4(\rdata[31]_i_5_n_0 ),
        .I5(ap_start),
        .O(\rdata[0]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[0]_i_3 
       (.I0(\int_kernel_factor_reg_n_0_[0] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [0]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [0]),
        .O(\rdata[0]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[10]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[10] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [10]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [10]),
        .O(\rdata[10]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[11]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[11] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [11]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [11]),
        .O(\rdata[11]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[12]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[12] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [12]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [12]),
        .O(\rdata[12]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[13]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[13] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [13]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [13]),
        .O(\rdata[13]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[14]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[14] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [14]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [14]),
        .O(\rdata[14]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[15]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[15] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [15]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [15]),
        .O(\rdata[15]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[16]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[16] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [16]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [16]),
        .O(\rdata[16]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[17]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[17] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [17]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [17]),
        .O(\rdata[17]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[18]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[18] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [18]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [18]),
        .O(\rdata[18]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[19]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[19] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [19]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [19]),
        .O(\rdata[19]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hA0A0CFC0)) 
    \rdata[1]_i_2 
       (.I0(\int_isr_reg_n_0_[1] ),
        .I1(p_0_in),
        .I2(\rdata[31]_i_4_n_0 ),
        .I3(int_task_ap_done),
        .I4(\rdata[31]_i_5_n_0 ),
        .O(\rdata[1]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[1]_i_3 
       (.I0(\int_kernel_factor_reg_n_0_[1] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [1]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [1]),
        .O(\rdata[1]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[20]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[20] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [20]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [20]),
        .O(\rdata[20]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[21]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[21] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [21]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [21]),
        .O(\rdata[21]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[22]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[22] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [22]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [22]),
        .O(\rdata[22]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[23]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[23] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [23]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [23]),
        .O(\rdata[23]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[24]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[24] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [24]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [24]),
        .O(\rdata[24]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[25]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[25] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [25]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [25]),
        .O(\rdata[25]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[26]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[26] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [26]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [26]),
        .O(\rdata[26]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[27]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[27] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [27]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [27]),
        .O(\rdata[27]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[28]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[28] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [28]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [28]),
        .O(\rdata[28]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[29]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[29] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [29]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [29]),
        .O(\rdata[29]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00A000A0C0CFC0C0)) 
    \rdata[2]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[2] ),
        .I1(\rdata[2]_i_2_n_0 ),
        .I2(\rdata[9]_i_3_n_0 ),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(p_3_in[2]),
        .I5(\rdata[31]_i_4_n_0 ),
        .O(rdata[2]));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \rdata[2]_i_2 
       (.I0(\int_image_height_reg[31]_0 [2]),
        .I1(\rdata[31]_i_5_n_0 ),
        .I2(\int_image_width_reg[31]_0 [2]),
        .O(\rdata[2]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[30]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[30] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [30]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [30]),
        .O(\rdata[30]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000000008)) 
    \rdata[31]_i_1 
       (.I0(\FSM_onehot_rstate_reg[1]_0 ),
        .I1(s_axi_control_ARVALID),
        .I2(s_axi_control_ARADDR[1]),
        .I3(s_axi_control_ARADDR[0]),
        .I4(s_axi_control_ARADDR[4]),
        .I5(s_axi_control_ARADDR[5]),
        .O(\rdata[31]_i_1_n_0 ));
  LUT2 #(
    .INIT(4'h8)) 
    \rdata[31]_i_2 
       (.I0(s_axi_control_ARVALID),
        .I1(\FSM_onehot_rstate_reg[1]_0 ),
        .O(ar_hs));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[31]_i_3 
       (.I0(\int_kernel_factor_reg_n_0_[31] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [31]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [31]),
        .O(\rdata[31]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFEFEFEFFFE)) 
    \rdata[31]_i_4 
       (.I0(s_axi_control_ARADDR[1]),
        .I1(s_axi_control_ARADDR[0]),
        .I2(s_axi_control_ARADDR[5]),
        .I3(s_axi_control_ARADDR[3]),
        .I4(s_axi_control_ARADDR[4]),
        .I5(s_axi_control_ARADDR[2]),
        .O(\rdata[31]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFEEA)) 
    \rdata[31]_i_5 
       (.I0(s_axi_control_ARADDR[1]),
        .I1(s_axi_control_ARADDR[4]),
        .I2(s_axi_control_ARADDR[5]),
        .I3(s_axi_control_ARADDR[3]),
        .I4(s_axi_control_ARADDR[0]),
        .I5(s_axi_control_ARADDR[2]),
        .O(\rdata[31]_i_5_n_0 ));
  LUT6 #(
    .INIT(64'h00A000A0C0CFC0C0)) 
    \rdata[3]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[3] ),
        .I1(\rdata[3]_i_2_n_0 ),
        .I2(\rdata[9]_i_3_n_0 ),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(int_ap_ready),
        .I5(\rdata[31]_i_4_n_0 ),
        .O(rdata[3]));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \rdata[3]_i_2 
       (.I0(\int_image_height_reg[31]_0 [3]),
        .I1(\rdata[31]_i_5_n_0 ),
        .I2(\int_image_width_reg[31]_0 [3]),
        .O(\rdata[3]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[4]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[4] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [4]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [4]),
        .O(\rdata[4]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[5]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[5] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [5]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [5]),
        .O(\rdata[5]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[6]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[6] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [6]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [6]),
        .O(\rdata[6]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00A000A0C0CFC0C0)) 
    \rdata[7]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[7] ),
        .I1(\rdata[7]_i_2_n_0 ),
        .I2(\rdata[9]_i_3_n_0 ),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(p_3_in[7]),
        .I5(\rdata[31]_i_4_n_0 ),
        .O(rdata[7]));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \rdata[7]_i_2 
       (.I0(\int_image_height_reg[31]_0 [7]),
        .I1(\rdata[31]_i_5_n_0 ),
        .I2(\int_image_width_reg[31]_0 [7]),
        .O(\rdata[7]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'h30BB3088)) 
    \rdata[8]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[8] ),
        .I1(\rdata[31]_i_4_n_0 ),
        .I2(\int_image_height_reg[31]_0 [8]),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(\int_image_width_reg[31]_0 [8]),
        .O(\rdata[8]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00A000A0C0CFC0C0)) 
    \rdata[9]_i_1 
       (.I0(\int_kernel_factor_reg_n_0_[9] ),
        .I1(\rdata[9]_i_2_n_0 ),
        .I2(\rdata[9]_i_3_n_0 ),
        .I3(\rdata[31]_i_5_n_0 ),
        .I4(interrupt),
        .I5(\rdata[31]_i_4_n_0 ),
        .O(rdata[9]));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \rdata[9]_i_2 
       (.I0(\int_image_height_reg[31]_0 [9]),
        .I1(\rdata[31]_i_5_n_0 ),
        .I2(\int_image_width_reg[31]_0 [9]),
        .O(\rdata[9]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \rdata[9]_i_3 
       (.I0(s_axi_control_ARADDR[5]),
        .I1(s_axi_control_ARADDR[4]),
        .I2(s_axi_control_ARADDR[0]),
        .I3(s_axi_control_ARADDR[1]),
        .O(\rdata[9]_i_3_n_0 ));
  FDRE \rdata_reg[0] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(rdata[0]),
        .Q(s_axi_control_RDATA[0]),
        .R(1'b0));
  MUXF7 \rdata_reg[0]_i_1 
       (.I0(\rdata[0]_i_2_n_0 ),
        .I1(\rdata[0]_i_3_n_0 ),
        .O(rdata[0]),
        .S(\rdata[9]_i_3_n_0 ));
  FDRE \rdata_reg[10] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[10]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[10]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[11] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[11]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[11]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[12] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[12]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[12]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[13] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[13]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[13]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[14] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[14]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[14]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[15] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[15]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[15]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[16] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[16]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[16]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[17] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[17]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[17]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[18] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[18]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[18]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[19] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[19]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[19]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[1] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(rdata[1]),
        .Q(s_axi_control_RDATA[1]),
        .R(1'b0));
  MUXF7 \rdata_reg[1]_i_1 
       (.I0(\rdata[1]_i_2_n_0 ),
        .I1(\rdata[1]_i_3_n_0 ),
        .O(rdata[1]),
        .S(\rdata[9]_i_3_n_0 ));
  FDRE \rdata_reg[20] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[20]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[20]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[21] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[21]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[21]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[22] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[22]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[22]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[23] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[23]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[23]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[24] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[24]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[24]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[25] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[25]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[25]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[26] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[26]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[26]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[27] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[27]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[27]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[28] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[28]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[28]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[29] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[29]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[29]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[2] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(rdata[2]),
        .Q(s_axi_control_RDATA[2]),
        .R(1'b0));
  FDRE \rdata_reg[30] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[30]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[30]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[31] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[31]_i_3_n_0 ),
        .Q(s_axi_control_RDATA[31]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[3] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(rdata[3]),
        .Q(s_axi_control_RDATA[3]),
        .R(1'b0));
  FDRE \rdata_reg[4] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[4]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[4]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[5] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[5]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[5]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[6] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[6]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[6]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[7] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(rdata[7]),
        .Q(s_axi_control_RDATA[7]),
        .R(1'b0));
  FDRE \rdata_reg[8] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(\rdata[8]_i_1_n_0 ),
        .Q(s_axi_control_RDATA[8]),
        .R(\rdata[31]_i_1_n_0 ));
  FDRE \rdata_reg[9] 
       (.C(ap_clk),
        .CE(ar_hs),
        .D(rdata[9]),
        .Q(s_axi_control_RDATA[9]),
        .R(1'b0));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT2 #(
    .INIT(4'hB)) 
    tmp_product__0_i_1
       (.I0(ap_start),
        .I1(Q[0]),
        .O(E));
  LUT2 #(
    .INIT(4'h8)) 
    \waddr[5]_i_1 
       (.I0(\FSM_onehot_wstate_reg[1]_0 ),
        .I1(s_axi_control_AWVALID),
        .O(waddr));
  FDRE \waddr_reg[2] 
       (.C(ap_clk),
        .CE(waddr),
        .D(s_axi_control_AWADDR[0]),
        .Q(\waddr_reg_n_0_[2] ),
        .R(1'b0));
  FDRE \waddr_reg[3] 
       (.C(ap_clk),
        .CE(waddr),
        .D(s_axi_control_AWADDR[1]),
        .Q(\waddr_reg_n_0_[3] ),
        .R(1'b0));
  FDRE \waddr_reg[4] 
       (.C(ap_clk),
        .CE(waddr),
        .D(s_axi_control_AWADDR[2]),
        .Q(\waddr_reg_n_0_[4] ),
        .R(1'b0));
  FDRE \waddr_reg[5] 
       (.C(ap_clk),
        .CE(waddr),
        .D(s_axi_control_AWADDR[3]),
        .Q(\waddr_reg_n_0_[5] ),
        .R(1'b0));
endmodule

module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2
   (ap_enable_reg_pp0_iter2_reg_0,
    \data_p2_reg[0] ,
    input_stream_TREADY_int_regslice,
    ap_enable_reg_pp0_iter2_reg_1,
    ap_enable_reg_pp0_iter2_reg_2,
    grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TLAST,
    D,
    \ap_CS_fsm_reg[4] ,
    \icmp_ln44_reg_312_reg[0]_0 ,
    ap_clk,
    E,
    ap_rst_n_inv,
    data_p2,
    \data_p1_reg[0] ,
    output_stream_TLAST_reg,
    Q,
    output_stream_TREADY_int_regslice,
    \ap_CS_fsm_reg[6] ,
    \data_p2_reg[0]_0 ,
    \data_p2_reg[3] ,
    \data_p2_reg[3]_0 ,
    grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg,
    \ap_CS_fsm_reg[6]_0 ,
    ap_rst_n,
    icmp_ln34_fu_153_p2_carry__4_0,
    icmp_ln35_fu_168_p2_carry__2_0,
    \icmp_ln44_reg_312_reg[0]_1 ,
    cmp6_fu_242_p2_carry__1_0);
  output ap_enable_reg_pp0_iter2_reg_0;
  output \data_p2_reg[0] ;
  output input_stream_TREADY_int_regslice;
  output [0:0]ap_enable_reg_pp0_iter2_reg_1;
  output [0:0]ap_enable_reg_pp0_iter2_reg_2;
  output grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TLAST;
  output [1:0]D;
  output \ap_CS_fsm_reg[4] ;
  output \icmp_ln44_reg_312_reg[0]_0 ;
  input ap_clk;
  input [0:0]E;
  input ap_rst_n_inv;
  input data_p2;
  input \data_p1_reg[0] ;
  input output_stream_TLAST_reg;
  input [0:0]Q;
  input output_stream_TREADY_int_regslice;
  input [2:0]\ap_CS_fsm_reg[6] ;
  input \data_p2_reg[0]_0 ;
  input \data_p2_reg[3] ;
  input \data_p2_reg[3]_0 ;
  input grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg;
  input \ap_CS_fsm_reg[6]_0 ;
  input ap_rst_n;
  input [61:0]icmp_ln34_fu_153_p2_carry__4_0;
  input [31:0]icmp_ln35_fu_168_p2_carry__2_0;
  input [31:0]\icmp_ln44_reg_312_reg[0]_1 ;
  input [31:0]cmp6_fu_242_p2_carry__1_0;

  wire [1:0]D;
  wire [0:0]E;
  wire [0:0]Q;
  wire [30:1]add_ln35_fu_210_p2;
  wire \ap_CS_fsm[6]_i_2_n_0 ;
  wire \ap_CS_fsm_reg[4] ;
  wire [2:0]\ap_CS_fsm_reg[6] ;
  wire \ap_CS_fsm_reg[6]_0 ;
  wire ap_block_pp0_stage0_11001__1;
  wire ap_clk;
  wire ap_enable_reg_pp0_iter1;
  wire ap_enable_reg_pp0_iter1_i_1_n_0;
  wire ap_enable_reg_pp0_iter2_i_1_n_0;
  wire ap_enable_reg_pp0_iter2_reg_0;
  wire [0:0]ap_enable_reg_pp0_iter2_reg_1;
  wire [0:0]ap_enable_reg_pp0_iter2_reg_2;
  wire ap_rst_n;
  wire ap_rst_n_inv;
  wire cmp6_fu_242_p2;
  wire cmp6_fu_242_p2_carry__0_i_1_n_0;
  wire cmp6_fu_242_p2_carry__0_i_2_n_0;
  wire cmp6_fu_242_p2_carry__0_i_3_n_0;
  wire cmp6_fu_242_p2_carry__0_i_4_n_0;
  wire cmp6_fu_242_p2_carry__0_n_0;
  wire cmp6_fu_242_p2_carry__0_n_1;
  wire cmp6_fu_242_p2_carry__0_n_2;
  wire cmp6_fu_242_p2_carry__0_n_3;
  wire [31:0]cmp6_fu_242_p2_carry__1_0;
  wire cmp6_fu_242_p2_carry__1_i_1_n_0;
  wire cmp6_fu_242_p2_carry__1_i_2_n_0;
  wire cmp6_fu_242_p2_carry__1_i_3_n_0;
  wire cmp6_fu_242_p2_carry__1_n_2;
  wire cmp6_fu_242_p2_carry__1_n_3;
  wire cmp6_fu_242_p2_carry_i_1_n_0;
  wire cmp6_fu_242_p2_carry_i_2_n_0;
  wire cmp6_fu_242_p2_carry_i_3_n_0;
  wire cmp6_fu_242_p2_carry_i_4_n_0;
  wire cmp6_fu_242_p2_carry_n_0;
  wire cmp6_fu_242_p2_carry_n_1;
  wire cmp6_fu_242_p2_carry_n_2;
  wire cmp6_fu_242_p2_carry_n_3;
  wire [30:0]col_fu_70;
  wire \col_fu_70[30]_i_2_n_0 ;
  wire \col_fu_70_reg[12]_i_1_n_0 ;
  wire \col_fu_70_reg[12]_i_1_n_1 ;
  wire \col_fu_70_reg[12]_i_1_n_2 ;
  wire \col_fu_70_reg[12]_i_1_n_3 ;
  wire \col_fu_70_reg[16]_i_1_n_0 ;
  wire \col_fu_70_reg[16]_i_1_n_1 ;
  wire \col_fu_70_reg[16]_i_1_n_2 ;
  wire \col_fu_70_reg[16]_i_1_n_3 ;
  wire \col_fu_70_reg[20]_i_1_n_0 ;
  wire \col_fu_70_reg[20]_i_1_n_1 ;
  wire \col_fu_70_reg[20]_i_1_n_2 ;
  wire \col_fu_70_reg[20]_i_1_n_3 ;
  wire \col_fu_70_reg[24]_i_1_n_0 ;
  wire \col_fu_70_reg[24]_i_1_n_1 ;
  wire \col_fu_70_reg[24]_i_1_n_2 ;
  wire \col_fu_70_reg[24]_i_1_n_3 ;
  wire \col_fu_70_reg[28]_i_1_n_0 ;
  wire \col_fu_70_reg[28]_i_1_n_1 ;
  wire \col_fu_70_reg[28]_i_1_n_2 ;
  wire \col_fu_70_reg[28]_i_1_n_3 ;
  wire \col_fu_70_reg[30]_i_3_n_3 ;
  wire \col_fu_70_reg[4]_i_1_n_0 ;
  wire \col_fu_70_reg[4]_i_1_n_1 ;
  wire \col_fu_70_reg[4]_i_1_n_2 ;
  wire \col_fu_70_reg[4]_i_1_n_3 ;
  wire \col_fu_70_reg[8]_i_1_n_0 ;
  wire \col_fu_70_reg[8]_i_1_n_1 ;
  wire \col_fu_70_reg[8]_i_1_n_2 ;
  wire \col_fu_70_reg[8]_i_1_n_3 ;
  wire \data_p1_reg[0] ;
  wire data_p2;
  wire \data_p2_reg[0] ;
  wire \data_p2_reg[0]_0 ;
  wire \data_p2_reg[3] ;
  wire \data_p2_reg[3]_0 ;
  wire flow_control_loop_pipe_sequential_init_U_n_0;
  wire flow_control_loop_pipe_sequential_init_U_n_1;
  wire flow_control_loop_pipe_sequential_init_U_n_5;
  wire flow_control_loop_pipe_sequential_init_U_n_6;
  wire grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg;
  wire grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TLAST;
  wire icmp_ln34_fu_153_p2;
  wire icmp_ln34_fu_153_p2_carry__0_i_1_n_0;
  wire icmp_ln34_fu_153_p2_carry__0_i_2_n_0;
  wire icmp_ln34_fu_153_p2_carry__0_i_3_n_0;
  wire icmp_ln34_fu_153_p2_carry__0_i_4_n_0;
  wire icmp_ln34_fu_153_p2_carry__0_n_0;
  wire icmp_ln34_fu_153_p2_carry__0_n_1;
  wire icmp_ln34_fu_153_p2_carry__0_n_2;
  wire icmp_ln34_fu_153_p2_carry__0_n_3;
  wire icmp_ln34_fu_153_p2_carry__1_i_1_n_0;
  wire icmp_ln34_fu_153_p2_carry__1_i_2_n_0;
  wire icmp_ln34_fu_153_p2_carry__1_i_3_n_0;
  wire icmp_ln34_fu_153_p2_carry__1_i_4_n_0;
  wire icmp_ln34_fu_153_p2_carry__1_n_0;
  wire icmp_ln34_fu_153_p2_carry__1_n_1;
  wire icmp_ln34_fu_153_p2_carry__1_n_2;
  wire icmp_ln34_fu_153_p2_carry__1_n_3;
  wire icmp_ln34_fu_153_p2_carry__2_i_1_n_0;
  wire icmp_ln34_fu_153_p2_carry__2_i_2_n_0;
  wire icmp_ln34_fu_153_p2_carry__2_i_3_n_0;
  wire icmp_ln34_fu_153_p2_carry__2_i_4_n_0;
  wire icmp_ln34_fu_153_p2_carry__2_n_0;
  wire icmp_ln34_fu_153_p2_carry__2_n_1;
  wire icmp_ln34_fu_153_p2_carry__2_n_2;
  wire icmp_ln34_fu_153_p2_carry__2_n_3;
  wire icmp_ln34_fu_153_p2_carry__3_i_1_n_0;
  wire icmp_ln34_fu_153_p2_carry__3_i_2_n_0;
  wire icmp_ln34_fu_153_p2_carry__3_i_3_n_0;
  wire icmp_ln34_fu_153_p2_carry__3_i_4_n_0;
  wire icmp_ln34_fu_153_p2_carry__3_n_0;
  wire icmp_ln34_fu_153_p2_carry__3_n_1;
  wire icmp_ln34_fu_153_p2_carry__3_n_2;
  wire icmp_ln34_fu_153_p2_carry__3_n_3;
  wire [61:0]icmp_ln34_fu_153_p2_carry__4_0;
  wire icmp_ln34_fu_153_p2_carry_i_1_n_0;
  wire icmp_ln34_fu_153_p2_carry_i_2_n_0;
  wire icmp_ln34_fu_153_p2_carry_i_3_n_0;
  wire icmp_ln34_fu_153_p2_carry_i_4_n_0;
  wire icmp_ln34_fu_153_p2_carry_n_0;
  wire icmp_ln34_fu_153_p2_carry_n_1;
  wire icmp_ln34_fu_153_p2_carry_n_2;
  wire icmp_ln34_fu_153_p2_carry_n_3;
  wire icmp_ln35_fu_168_p2_carry__0_i_1_n_0;
  wire icmp_ln35_fu_168_p2_carry__0_i_2_n_0;
  wire icmp_ln35_fu_168_p2_carry__0_i_3_n_0;
  wire icmp_ln35_fu_168_p2_carry__0_i_4_n_0;
  wire icmp_ln35_fu_168_p2_carry__0_i_5_n_0;
  wire icmp_ln35_fu_168_p2_carry__0_i_6_n_0;
  wire icmp_ln35_fu_168_p2_carry__0_i_7_n_0;
  wire icmp_ln35_fu_168_p2_carry__0_i_8_n_0;
  wire icmp_ln35_fu_168_p2_carry__0_n_0;
  wire icmp_ln35_fu_168_p2_carry__0_n_1;
  wire icmp_ln35_fu_168_p2_carry__0_n_2;
  wire icmp_ln35_fu_168_p2_carry__0_n_3;
  wire icmp_ln35_fu_168_p2_carry__1_i_1_n_0;
  wire icmp_ln35_fu_168_p2_carry__1_i_2_n_0;
  wire icmp_ln35_fu_168_p2_carry__1_i_3_n_0;
  wire icmp_ln35_fu_168_p2_carry__1_i_4_n_0;
  wire icmp_ln35_fu_168_p2_carry__1_i_5_n_0;
  wire icmp_ln35_fu_168_p2_carry__1_i_6_n_0;
  wire icmp_ln35_fu_168_p2_carry__1_i_7_n_0;
  wire icmp_ln35_fu_168_p2_carry__1_i_8_n_0;
  wire icmp_ln35_fu_168_p2_carry__1_n_0;
  wire icmp_ln35_fu_168_p2_carry__1_n_1;
  wire icmp_ln35_fu_168_p2_carry__1_n_2;
  wire icmp_ln35_fu_168_p2_carry__1_n_3;
  wire [31:0]icmp_ln35_fu_168_p2_carry__2_0;
  wire icmp_ln35_fu_168_p2_carry__2_i_1_n_0;
  wire icmp_ln35_fu_168_p2_carry__2_i_2_n_0;
  wire icmp_ln35_fu_168_p2_carry__2_i_3_n_0;
  wire icmp_ln35_fu_168_p2_carry__2_i_4_n_0;
  wire icmp_ln35_fu_168_p2_carry__2_i_5_n_0;
  wire icmp_ln35_fu_168_p2_carry__2_i_6_n_0;
  wire icmp_ln35_fu_168_p2_carry__2_i_7_n_0;
  wire icmp_ln35_fu_168_p2_carry__2_i_8_n_0;
  wire icmp_ln35_fu_168_p2_carry__2_n_0;
  wire icmp_ln35_fu_168_p2_carry__2_n_1;
  wire icmp_ln35_fu_168_p2_carry__2_n_2;
  wire icmp_ln35_fu_168_p2_carry__2_n_3;
  wire icmp_ln35_fu_168_p2_carry_i_1_n_0;
  wire icmp_ln35_fu_168_p2_carry_i_2_n_0;
  wire icmp_ln35_fu_168_p2_carry_i_3_n_0;
  wire icmp_ln35_fu_168_p2_carry_i_4_n_0;
  wire icmp_ln35_fu_168_p2_carry_i_5_n_0;
  wire icmp_ln35_fu_168_p2_carry_i_6_n_0;
  wire icmp_ln35_fu_168_p2_carry_i_7_n_0;
  wire icmp_ln35_fu_168_p2_carry_i_8_n_0;
  wire icmp_ln35_fu_168_p2_carry_n_0;
  wire icmp_ln35_fu_168_p2_carry_n_1;
  wire icmp_ln35_fu_168_p2_carry_n_2;
  wire icmp_ln35_fu_168_p2_carry_n_3;
  wire icmp_ln44_fu_205_p2;
  wire icmp_ln44_fu_205_p2_carry__0_i_1_n_0;
  wire icmp_ln44_fu_205_p2_carry__0_i_2_n_0;
  wire icmp_ln44_fu_205_p2_carry__0_i_3_n_0;
  wire icmp_ln44_fu_205_p2_carry__0_i_4_n_0;
  wire icmp_ln44_fu_205_p2_carry__0_i_5_n_0;
  wire icmp_ln44_fu_205_p2_carry__0_i_6_n_0;
  wire icmp_ln44_fu_205_p2_carry__0_i_7_n_0;
  wire icmp_ln44_fu_205_p2_carry__0_i_8_n_0;
  wire icmp_ln44_fu_205_p2_carry__0_n_0;
  wire icmp_ln44_fu_205_p2_carry__0_n_1;
  wire icmp_ln44_fu_205_p2_carry__0_n_2;
  wire icmp_ln44_fu_205_p2_carry__0_n_3;
  wire icmp_ln44_fu_205_p2_carry__1_i_1_n_0;
  wire icmp_ln44_fu_205_p2_carry__1_i_2_n_0;
  wire icmp_ln44_fu_205_p2_carry__1_i_3_n_0;
  wire icmp_ln44_fu_205_p2_carry__1_i_4_n_0;
  wire icmp_ln44_fu_205_p2_carry__1_i_5_n_0;
  wire icmp_ln44_fu_205_p2_carry__1_n_2;
  wire icmp_ln44_fu_205_p2_carry__1_n_3;
  wire icmp_ln44_fu_205_p2_carry_i_1_n_0;
  wire icmp_ln44_fu_205_p2_carry_i_2_n_0;
  wire icmp_ln44_fu_205_p2_carry_i_3_n_0;
  wire icmp_ln44_fu_205_p2_carry_i_4_n_0;
  wire icmp_ln44_fu_205_p2_carry_i_5_n_0;
  wire icmp_ln44_fu_205_p2_carry_i_6_n_0;
  wire icmp_ln44_fu_205_p2_carry_i_7_n_0;
  wire icmp_ln44_fu_205_p2_carry_i_8_n_0;
  wire icmp_ln44_fu_205_p2_carry_n_0;
  wire icmp_ln44_fu_205_p2_carry_n_1;
  wire icmp_ln44_fu_205_p2_carry_n_2;
  wire icmp_ln44_fu_205_p2_carry_n_3;
  wire icmp_ln44_reg_312;
  wire \icmp_ln44_reg_312_reg[0]_0 ;
  wire [31:0]\icmp_ln44_reg_312_reg[0]_1 ;
  wire \indvar_flatten_fu_78[0]_i_3_n_0 ;
  wire [61:0]indvar_flatten_fu_78_reg;
  wire \indvar_flatten_fu_78_reg[0]_i_2_n_0 ;
  wire \indvar_flatten_fu_78_reg[0]_i_2_n_1 ;
  wire \indvar_flatten_fu_78_reg[0]_i_2_n_2 ;
  wire \indvar_flatten_fu_78_reg[0]_i_2_n_3 ;
  wire \indvar_flatten_fu_78_reg[0]_i_2_n_4 ;
  wire \indvar_flatten_fu_78_reg[0]_i_2_n_5 ;
  wire \indvar_flatten_fu_78_reg[0]_i_2_n_6 ;
  wire \indvar_flatten_fu_78_reg[0]_i_2_n_7 ;
  wire \indvar_flatten_fu_78_reg[12]_i_1_n_0 ;
  wire \indvar_flatten_fu_78_reg[12]_i_1_n_1 ;
  wire \indvar_flatten_fu_78_reg[12]_i_1_n_2 ;
  wire \indvar_flatten_fu_78_reg[12]_i_1_n_3 ;
  wire \indvar_flatten_fu_78_reg[12]_i_1_n_4 ;
  wire \indvar_flatten_fu_78_reg[12]_i_1_n_5 ;
  wire \indvar_flatten_fu_78_reg[12]_i_1_n_6 ;
  wire \indvar_flatten_fu_78_reg[12]_i_1_n_7 ;
  wire \indvar_flatten_fu_78_reg[16]_i_1_n_0 ;
  wire \indvar_flatten_fu_78_reg[16]_i_1_n_1 ;
  wire \indvar_flatten_fu_78_reg[16]_i_1_n_2 ;
  wire \indvar_flatten_fu_78_reg[16]_i_1_n_3 ;
  wire \indvar_flatten_fu_78_reg[16]_i_1_n_4 ;
  wire \indvar_flatten_fu_78_reg[16]_i_1_n_5 ;
  wire \indvar_flatten_fu_78_reg[16]_i_1_n_6 ;
  wire \indvar_flatten_fu_78_reg[16]_i_1_n_7 ;
  wire \indvar_flatten_fu_78_reg[20]_i_1_n_0 ;
  wire \indvar_flatten_fu_78_reg[20]_i_1_n_1 ;
  wire \indvar_flatten_fu_78_reg[20]_i_1_n_2 ;
  wire \indvar_flatten_fu_78_reg[20]_i_1_n_3 ;
  wire \indvar_flatten_fu_78_reg[20]_i_1_n_4 ;
  wire \indvar_flatten_fu_78_reg[20]_i_1_n_5 ;
  wire \indvar_flatten_fu_78_reg[20]_i_1_n_6 ;
  wire \indvar_flatten_fu_78_reg[20]_i_1_n_7 ;
  wire \indvar_flatten_fu_78_reg[24]_i_1_n_0 ;
  wire \indvar_flatten_fu_78_reg[24]_i_1_n_1 ;
  wire \indvar_flatten_fu_78_reg[24]_i_1_n_2 ;
  wire \indvar_flatten_fu_78_reg[24]_i_1_n_3 ;
  wire \indvar_flatten_fu_78_reg[24]_i_1_n_4 ;
  wire \indvar_flatten_fu_78_reg[24]_i_1_n_5 ;
  wire \indvar_flatten_fu_78_reg[24]_i_1_n_6 ;
  wire \indvar_flatten_fu_78_reg[24]_i_1_n_7 ;
  wire \indvar_flatten_fu_78_reg[28]_i_1_n_0 ;
  wire \indvar_flatten_fu_78_reg[28]_i_1_n_1 ;
  wire \indvar_flatten_fu_78_reg[28]_i_1_n_2 ;
  wire \indvar_flatten_fu_78_reg[28]_i_1_n_3 ;
  wire \indvar_flatten_fu_78_reg[28]_i_1_n_4 ;
  wire \indvar_flatten_fu_78_reg[28]_i_1_n_5 ;
  wire \indvar_flatten_fu_78_reg[28]_i_1_n_6 ;
  wire \indvar_flatten_fu_78_reg[28]_i_1_n_7 ;
  wire \indvar_flatten_fu_78_reg[32]_i_1_n_0 ;
  wire \indvar_flatten_fu_78_reg[32]_i_1_n_1 ;
  wire \indvar_flatten_fu_78_reg[32]_i_1_n_2 ;
  wire \indvar_flatten_fu_78_reg[32]_i_1_n_3 ;
  wire \indvar_flatten_fu_78_reg[32]_i_1_n_4 ;
  wire \indvar_flatten_fu_78_reg[32]_i_1_n_5 ;
  wire \indvar_flatten_fu_78_reg[32]_i_1_n_6 ;
  wire \indvar_flatten_fu_78_reg[32]_i_1_n_7 ;
  wire \indvar_flatten_fu_78_reg[36]_i_1_n_0 ;
  wire \indvar_flatten_fu_78_reg[36]_i_1_n_1 ;
  wire \indvar_flatten_fu_78_reg[36]_i_1_n_2 ;
  wire \indvar_flatten_fu_78_reg[36]_i_1_n_3 ;
  wire \indvar_flatten_fu_78_reg[36]_i_1_n_4 ;
  wire \indvar_flatten_fu_78_reg[36]_i_1_n_5 ;
  wire \indvar_flatten_fu_78_reg[36]_i_1_n_6 ;
  wire \indvar_flatten_fu_78_reg[36]_i_1_n_7 ;
  wire \indvar_flatten_fu_78_reg[40]_i_1_n_0 ;
  wire \indvar_flatten_fu_78_reg[40]_i_1_n_1 ;
  wire \indvar_flatten_fu_78_reg[40]_i_1_n_2 ;
  wire \indvar_flatten_fu_78_reg[40]_i_1_n_3 ;
  wire \indvar_flatten_fu_78_reg[40]_i_1_n_4 ;
  wire \indvar_flatten_fu_78_reg[40]_i_1_n_5 ;
  wire \indvar_flatten_fu_78_reg[40]_i_1_n_6 ;
  wire \indvar_flatten_fu_78_reg[40]_i_1_n_7 ;
  wire \indvar_flatten_fu_78_reg[44]_i_1_n_0 ;
  wire \indvar_flatten_fu_78_reg[44]_i_1_n_1 ;
  wire \indvar_flatten_fu_78_reg[44]_i_1_n_2 ;
  wire \indvar_flatten_fu_78_reg[44]_i_1_n_3 ;
  wire \indvar_flatten_fu_78_reg[44]_i_1_n_4 ;
  wire \indvar_flatten_fu_78_reg[44]_i_1_n_5 ;
  wire \indvar_flatten_fu_78_reg[44]_i_1_n_6 ;
  wire \indvar_flatten_fu_78_reg[44]_i_1_n_7 ;
  wire \indvar_flatten_fu_78_reg[48]_i_1_n_0 ;
  wire \indvar_flatten_fu_78_reg[48]_i_1_n_1 ;
  wire \indvar_flatten_fu_78_reg[48]_i_1_n_2 ;
  wire \indvar_flatten_fu_78_reg[48]_i_1_n_3 ;
  wire \indvar_flatten_fu_78_reg[48]_i_1_n_4 ;
  wire \indvar_flatten_fu_78_reg[48]_i_1_n_5 ;
  wire \indvar_flatten_fu_78_reg[48]_i_1_n_6 ;
  wire \indvar_flatten_fu_78_reg[48]_i_1_n_7 ;
  wire \indvar_flatten_fu_78_reg[4]_i_1_n_0 ;
  wire \indvar_flatten_fu_78_reg[4]_i_1_n_1 ;
  wire \indvar_flatten_fu_78_reg[4]_i_1_n_2 ;
  wire \indvar_flatten_fu_78_reg[4]_i_1_n_3 ;
  wire \indvar_flatten_fu_78_reg[4]_i_1_n_4 ;
  wire \indvar_flatten_fu_78_reg[4]_i_1_n_5 ;
  wire \indvar_flatten_fu_78_reg[4]_i_1_n_6 ;
  wire \indvar_flatten_fu_78_reg[4]_i_1_n_7 ;
  wire \indvar_flatten_fu_78_reg[52]_i_1_n_0 ;
  wire \indvar_flatten_fu_78_reg[52]_i_1_n_1 ;
  wire \indvar_flatten_fu_78_reg[52]_i_1_n_2 ;
  wire \indvar_flatten_fu_78_reg[52]_i_1_n_3 ;
  wire \indvar_flatten_fu_78_reg[52]_i_1_n_4 ;
  wire \indvar_flatten_fu_78_reg[52]_i_1_n_5 ;
  wire \indvar_flatten_fu_78_reg[52]_i_1_n_6 ;
  wire \indvar_flatten_fu_78_reg[52]_i_1_n_7 ;
  wire \indvar_flatten_fu_78_reg[56]_i_1_n_0 ;
  wire \indvar_flatten_fu_78_reg[56]_i_1_n_1 ;
  wire \indvar_flatten_fu_78_reg[56]_i_1_n_2 ;
  wire \indvar_flatten_fu_78_reg[56]_i_1_n_3 ;
  wire \indvar_flatten_fu_78_reg[56]_i_1_n_4 ;
  wire \indvar_flatten_fu_78_reg[56]_i_1_n_5 ;
  wire \indvar_flatten_fu_78_reg[56]_i_1_n_6 ;
  wire \indvar_flatten_fu_78_reg[56]_i_1_n_7 ;
  wire \indvar_flatten_fu_78_reg[60]_i_1_n_3 ;
  wire \indvar_flatten_fu_78_reg[60]_i_1_n_6 ;
  wire \indvar_flatten_fu_78_reg[60]_i_1_n_7 ;
  wire \indvar_flatten_fu_78_reg[8]_i_1_n_0 ;
  wire \indvar_flatten_fu_78_reg[8]_i_1_n_1 ;
  wire \indvar_flatten_fu_78_reg[8]_i_1_n_2 ;
  wire \indvar_flatten_fu_78_reg[8]_i_1_n_3 ;
  wire \indvar_flatten_fu_78_reg[8]_i_1_n_4 ;
  wire \indvar_flatten_fu_78_reg[8]_i_1_n_5 ;
  wire \indvar_flatten_fu_78_reg[8]_i_1_n_6 ;
  wire \indvar_flatten_fu_78_reg[8]_i_1_n_7 ;
  wire input_stream_TREADY_int_regslice;
  wire output_stream_TLAST_reg;
  wire output_stream_TREADY_int_regslice;
  wire \regslice_both_output_stream_V_last_V_U/load_p2 ;
  wire \row_fu_74[0]_i_2_n_0 ;
  wire [30:0]row_fu_74_reg;
  wire \row_fu_74_reg[0]_i_1_n_0 ;
  wire \row_fu_74_reg[0]_i_1_n_1 ;
  wire \row_fu_74_reg[0]_i_1_n_2 ;
  wire \row_fu_74_reg[0]_i_1_n_3 ;
  wire \row_fu_74_reg[0]_i_1_n_4 ;
  wire \row_fu_74_reg[0]_i_1_n_5 ;
  wire \row_fu_74_reg[0]_i_1_n_6 ;
  wire \row_fu_74_reg[0]_i_1_n_7 ;
  wire \row_fu_74_reg[12]_i_1_n_0 ;
  wire \row_fu_74_reg[12]_i_1_n_1 ;
  wire \row_fu_74_reg[12]_i_1_n_2 ;
  wire \row_fu_74_reg[12]_i_1_n_3 ;
  wire \row_fu_74_reg[12]_i_1_n_4 ;
  wire \row_fu_74_reg[12]_i_1_n_5 ;
  wire \row_fu_74_reg[12]_i_1_n_6 ;
  wire \row_fu_74_reg[12]_i_1_n_7 ;
  wire \row_fu_74_reg[16]_i_1_n_0 ;
  wire \row_fu_74_reg[16]_i_1_n_1 ;
  wire \row_fu_74_reg[16]_i_1_n_2 ;
  wire \row_fu_74_reg[16]_i_1_n_3 ;
  wire \row_fu_74_reg[16]_i_1_n_4 ;
  wire \row_fu_74_reg[16]_i_1_n_5 ;
  wire \row_fu_74_reg[16]_i_1_n_6 ;
  wire \row_fu_74_reg[16]_i_1_n_7 ;
  wire \row_fu_74_reg[20]_i_1_n_0 ;
  wire \row_fu_74_reg[20]_i_1_n_1 ;
  wire \row_fu_74_reg[20]_i_1_n_2 ;
  wire \row_fu_74_reg[20]_i_1_n_3 ;
  wire \row_fu_74_reg[20]_i_1_n_4 ;
  wire \row_fu_74_reg[20]_i_1_n_5 ;
  wire \row_fu_74_reg[20]_i_1_n_6 ;
  wire \row_fu_74_reg[20]_i_1_n_7 ;
  wire \row_fu_74_reg[24]_i_1_n_0 ;
  wire \row_fu_74_reg[24]_i_1_n_1 ;
  wire \row_fu_74_reg[24]_i_1_n_2 ;
  wire \row_fu_74_reg[24]_i_1_n_3 ;
  wire \row_fu_74_reg[24]_i_1_n_4 ;
  wire \row_fu_74_reg[24]_i_1_n_5 ;
  wire \row_fu_74_reg[24]_i_1_n_6 ;
  wire \row_fu_74_reg[24]_i_1_n_7 ;
  wire \row_fu_74_reg[28]_i_1_n_2 ;
  wire \row_fu_74_reg[28]_i_1_n_3 ;
  wire \row_fu_74_reg[28]_i_1_n_5 ;
  wire \row_fu_74_reg[28]_i_1_n_6 ;
  wire \row_fu_74_reg[28]_i_1_n_7 ;
  wire \row_fu_74_reg[4]_i_1_n_0 ;
  wire \row_fu_74_reg[4]_i_1_n_1 ;
  wire \row_fu_74_reg[4]_i_1_n_2 ;
  wire \row_fu_74_reg[4]_i_1_n_3 ;
  wire \row_fu_74_reg[4]_i_1_n_4 ;
  wire \row_fu_74_reg[4]_i_1_n_5 ;
  wire \row_fu_74_reg[4]_i_1_n_6 ;
  wire \row_fu_74_reg[4]_i_1_n_7 ;
  wire \row_fu_74_reg[8]_i_1_n_0 ;
  wire \row_fu_74_reg[8]_i_1_n_1 ;
  wire \row_fu_74_reg[8]_i_1_n_2 ;
  wire \row_fu_74_reg[8]_i_1_n_3 ;
  wire \row_fu_74_reg[8]_i_1_n_4 ;
  wire \row_fu_74_reg[8]_i_1_n_5 ;
  wire \row_fu_74_reg[8]_i_1_n_6 ;
  wire \row_fu_74_reg[8]_i_1_n_7 ;
  wire [30:0]select_ln34_fu_185_p3;
  wire [30:0]select_ln34_reg_307;
  wire \select_ln34_reg_307[3]_i_2_n_0 ;
  wire \select_ln34_reg_307_reg[11]_i_1_n_0 ;
  wire \select_ln34_reg_307_reg[11]_i_1_n_1 ;
  wire \select_ln34_reg_307_reg[11]_i_1_n_2 ;
  wire \select_ln34_reg_307_reg[11]_i_1_n_3 ;
  wire \select_ln34_reg_307_reg[15]_i_1_n_0 ;
  wire \select_ln34_reg_307_reg[15]_i_1_n_1 ;
  wire \select_ln34_reg_307_reg[15]_i_1_n_2 ;
  wire \select_ln34_reg_307_reg[15]_i_1_n_3 ;
  wire \select_ln34_reg_307_reg[19]_i_1_n_0 ;
  wire \select_ln34_reg_307_reg[19]_i_1_n_1 ;
  wire \select_ln34_reg_307_reg[19]_i_1_n_2 ;
  wire \select_ln34_reg_307_reg[19]_i_1_n_3 ;
  wire \select_ln34_reg_307_reg[23]_i_1_n_0 ;
  wire \select_ln34_reg_307_reg[23]_i_1_n_1 ;
  wire \select_ln34_reg_307_reg[23]_i_1_n_2 ;
  wire \select_ln34_reg_307_reg[23]_i_1_n_3 ;
  wire \select_ln34_reg_307_reg[27]_i_1_n_0 ;
  wire \select_ln34_reg_307_reg[27]_i_1_n_1 ;
  wire \select_ln34_reg_307_reg[27]_i_1_n_2 ;
  wire \select_ln34_reg_307_reg[27]_i_1_n_3 ;
  wire \select_ln34_reg_307_reg[30]_i_1_n_2 ;
  wire \select_ln34_reg_307_reg[30]_i_1_n_3 ;
  wire \select_ln34_reg_307_reg[3]_i_1_n_0 ;
  wire \select_ln34_reg_307_reg[3]_i_1_n_1 ;
  wire \select_ln34_reg_307_reg[3]_i_1_n_2 ;
  wire \select_ln34_reg_307_reg[3]_i_1_n_3 ;
  wire \select_ln34_reg_307_reg[7]_i_1_n_0 ;
  wire \select_ln34_reg_307_reg[7]_i_1_n_1 ;
  wire \select_ln34_reg_307_reg[7]_i_1_n_2 ;
  wire \select_ln34_reg_307_reg[7]_i_1_n_3 ;
  wire [3:0]NLW_cmp6_fu_242_p2_carry_O_UNCONNECTED;
  wire [3:0]NLW_cmp6_fu_242_p2_carry__0_O_UNCONNECTED;
  wire [3:3]NLW_cmp6_fu_242_p2_carry__1_CO_UNCONNECTED;
  wire [3:0]NLW_cmp6_fu_242_p2_carry__1_O_UNCONNECTED;
  wire [3:1]\NLW_col_fu_70_reg[30]_i_3_CO_UNCONNECTED ;
  wire [3:2]\NLW_col_fu_70_reg[30]_i_3_O_UNCONNECTED ;
  wire [3:0]NLW_icmp_ln34_fu_153_p2_carry_O_UNCONNECTED;
  wire [3:0]NLW_icmp_ln34_fu_153_p2_carry__0_O_UNCONNECTED;
  wire [3:0]NLW_icmp_ln34_fu_153_p2_carry__1_O_UNCONNECTED;
  wire [3:0]NLW_icmp_ln34_fu_153_p2_carry__2_O_UNCONNECTED;
  wire [3:0]NLW_icmp_ln34_fu_153_p2_carry__3_O_UNCONNECTED;
  wire [3:1]NLW_icmp_ln34_fu_153_p2_carry__4_CO_UNCONNECTED;
  wire [3:0]NLW_icmp_ln34_fu_153_p2_carry__4_O_UNCONNECTED;
  wire [3:0]NLW_icmp_ln35_fu_168_p2_carry_O_UNCONNECTED;
  wire [3:0]NLW_icmp_ln35_fu_168_p2_carry__0_O_UNCONNECTED;
  wire [3:0]NLW_icmp_ln35_fu_168_p2_carry__1_O_UNCONNECTED;
  wire [3:0]NLW_icmp_ln35_fu_168_p2_carry__2_O_UNCONNECTED;
  wire [3:0]NLW_icmp_ln44_fu_205_p2_carry_O_UNCONNECTED;
  wire [3:0]NLW_icmp_ln44_fu_205_p2_carry__0_O_UNCONNECTED;
  wire [3:3]NLW_icmp_ln44_fu_205_p2_carry__1_CO_UNCONNECTED;
  wire [3:0]NLW_icmp_ln44_fu_205_p2_carry__1_O_UNCONNECTED;
  wire [3:1]\NLW_indvar_flatten_fu_78_reg[60]_i_1_CO_UNCONNECTED ;
  wire [3:2]\NLW_indvar_flatten_fu_78_reg[60]_i_1_O_UNCONNECTED ;
  wire [3:2]\NLW_row_fu_74_reg[28]_i_1_CO_UNCONNECTED ;
  wire [3:3]\NLW_row_fu_74_reg[28]_i_1_O_UNCONNECTED ;
  wire [3:2]\NLW_select_ln34_reg_307_reg[30]_i_1_CO_UNCONNECTED ;
  wire [3:3]\NLW_select_ln34_reg_307_reg[30]_i_1_O_UNCONNECTED ;

  LUT6 #(
    .INIT(64'h8000000088888888)) 
    \ap_CS_fsm[6]_i_2 
       (.I0(ap_enable_reg_pp0_iter1),
        .I1(icmp_ln34_fu_153_p2),
        .I2(\ap_CS_fsm_reg[6] [1]),
        .I3(output_stream_TREADY_int_regslice),
        .I4(Q),
        .I5(ap_enable_reg_pp0_iter2_reg_0),
        .O(\ap_CS_fsm[6]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hCC000888)) 
    ap_enable_reg_pp0_iter1_i_1
       (.I0(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg),
        .I1(ap_rst_n),
        .I2(icmp_ln34_fu_153_p2),
        .I3(ap_enable_reg_pp0_iter1),
        .I4(ap_block_pp0_stage0_11001__1),
        .O(ap_enable_reg_pp0_iter1_i_1_n_0));
  FDRE #(
    .INIT(1'b0)) 
    ap_enable_reg_pp0_iter1_reg
       (.C(ap_clk),
        .CE(1'b1),
        .D(ap_enable_reg_pp0_iter1_i_1_n_0),
        .Q(ap_enable_reg_pp0_iter1),
        .R(1'b0));
  LUT5 #(
    .INIT(32'h88880C00)) 
    ap_enable_reg_pp0_iter2_i_1
       (.I0(ap_enable_reg_pp0_iter2_reg_0),
        .I1(ap_rst_n),
        .I2(icmp_ln34_fu_153_p2),
        .I3(ap_enable_reg_pp0_iter1),
        .I4(ap_block_pp0_stage0_11001__1),
        .O(ap_enable_reg_pp0_iter2_i_1_n_0));
  FDRE #(
    .INIT(1'b0)) 
    ap_enable_reg_pp0_iter2_reg
       (.C(ap_clk),
        .CE(1'b1),
        .D(ap_enable_reg_pp0_iter2_i_1_n_0),
        .Q(ap_enable_reg_pp0_iter2_reg_0),
        .R(1'b0));
  CARRY4 cmp6_fu_242_p2_carry
       (.CI(1'b0),
        .CO({cmp6_fu_242_p2_carry_n_0,cmp6_fu_242_p2_carry_n_1,cmp6_fu_242_p2_carry_n_2,cmp6_fu_242_p2_carry_n_3}),
        .CYINIT(1'b1),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(NLW_cmp6_fu_242_p2_carry_O_UNCONNECTED[3:0]),
        .S({cmp6_fu_242_p2_carry_i_1_n_0,cmp6_fu_242_p2_carry_i_2_n_0,cmp6_fu_242_p2_carry_i_3_n_0,cmp6_fu_242_p2_carry_i_4_n_0}));
  CARRY4 cmp6_fu_242_p2_carry__0
       (.CI(cmp6_fu_242_p2_carry_n_0),
        .CO({cmp6_fu_242_p2_carry__0_n_0,cmp6_fu_242_p2_carry__0_n_1,cmp6_fu_242_p2_carry__0_n_2,cmp6_fu_242_p2_carry__0_n_3}),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(NLW_cmp6_fu_242_p2_carry__0_O_UNCONNECTED[3:0]),
        .S({cmp6_fu_242_p2_carry__0_i_1_n_0,cmp6_fu_242_p2_carry__0_i_2_n_0,cmp6_fu_242_p2_carry__0_i_3_n_0,cmp6_fu_242_p2_carry__0_i_4_n_0}));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    cmp6_fu_242_p2_carry__0_i_1
       (.I0(select_ln34_reg_307[21]),
        .I1(cmp6_fu_242_p2_carry__1_0[21]),
        .I2(cmp6_fu_242_p2_carry__1_0[23]),
        .I3(select_ln34_reg_307[23]),
        .I4(cmp6_fu_242_p2_carry__1_0[22]),
        .I5(select_ln34_reg_307[22]),
        .O(cmp6_fu_242_p2_carry__0_i_1_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    cmp6_fu_242_p2_carry__0_i_2
       (.I0(select_ln34_reg_307[18]),
        .I1(cmp6_fu_242_p2_carry__1_0[18]),
        .I2(cmp6_fu_242_p2_carry__1_0[20]),
        .I3(select_ln34_reg_307[20]),
        .I4(cmp6_fu_242_p2_carry__1_0[19]),
        .I5(select_ln34_reg_307[19]),
        .O(cmp6_fu_242_p2_carry__0_i_2_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    cmp6_fu_242_p2_carry__0_i_3
       (.I0(select_ln34_reg_307[15]),
        .I1(cmp6_fu_242_p2_carry__1_0[15]),
        .I2(cmp6_fu_242_p2_carry__1_0[17]),
        .I3(select_ln34_reg_307[17]),
        .I4(cmp6_fu_242_p2_carry__1_0[16]),
        .I5(select_ln34_reg_307[16]),
        .O(cmp6_fu_242_p2_carry__0_i_3_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    cmp6_fu_242_p2_carry__0_i_4
       (.I0(select_ln34_reg_307[12]),
        .I1(cmp6_fu_242_p2_carry__1_0[12]),
        .I2(cmp6_fu_242_p2_carry__1_0[14]),
        .I3(select_ln34_reg_307[14]),
        .I4(cmp6_fu_242_p2_carry__1_0[13]),
        .I5(select_ln34_reg_307[13]),
        .O(cmp6_fu_242_p2_carry__0_i_4_n_0));
  CARRY4 cmp6_fu_242_p2_carry__1
       (.CI(cmp6_fu_242_p2_carry__0_n_0),
        .CO({NLW_cmp6_fu_242_p2_carry__1_CO_UNCONNECTED[3],cmp6_fu_242_p2,cmp6_fu_242_p2_carry__1_n_2,cmp6_fu_242_p2_carry__1_n_3}),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(NLW_cmp6_fu_242_p2_carry__1_O_UNCONNECTED[3:0]),
        .S({1'b0,cmp6_fu_242_p2_carry__1_i_1_n_0,cmp6_fu_242_p2_carry__1_i_2_n_0,cmp6_fu_242_p2_carry__1_i_3_n_0}));
  LUT3 #(
    .INIT(8'h09)) 
    cmp6_fu_242_p2_carry__1_i_1
       (.I0(select_ln34_reg_307[30]),
        .I1(cmp6_fu_242_p2_carry__1_0[30]),
        .I2(cmp6_fu_242_p2_carry__1_0[31]),
        .O(cmp6_fu_242_p2_carry__1_i_1_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    cmp6_fu_242_p2_carry__1_i_2
       (.I0(select_ln34_reg_307[27]),
        .I1(cmp6_fu_242_p2_carry__1_0[27]),
        .I2(cmp6_fu_242_p2_carry__1_0[29]),
        .I3(select_ln34_reg_307[29]),
        .I4(cmp6_fu_242_p2_carry__1_0[28]),
        .I5(select_ln34_reg_307[28]),
        .O(cmp6_fu_242_p2_carry__1_i_2_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    cmp6_fu_242_p2_carry__1_i_3
       (.I0(select_ln34_reg_307[24]),
        .I1(cmp6_fu_242_p2_carry__1_0[24]),
        .I2(cmp6_fu_242_p2_carry__1_0[26]),
        .I3(select_ln34_reg_307[26]),
        .I4(cmp6_fu_242_p2_carry__1_0[25]),
        .I5(select_ln34_reg_307[25]),
        .O(cmp6_fu_242_p2_carry__1_i_3_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    cmp6_fu_242_p2_carry_i_1
       (.I0(select_ln34_reg_307[9]),
        .I1(cmp6_fu_242_p2_carry__1_0[9]),
        .I2(cmp6_fu_242_p2_carry__1_0[11]),
        .I3(select_ln34_reg_307[11]),
        .I4(cmp6_fu_242_p2_carry__1_0[10]),
        .I5(select_ln34_reg_307[10]),
        .O(cmp6_fu_242_p2_carry_i_1_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    cmp6_fu_242_p2_carry_i_2
       (.I0(select_ln34_reg_307[6]),
        .I1(cmp6_fu_242_p2_carry__1_0[6]),
        .I2(cmp6_fu_242_p2_carry__1_0[8]),
        .I3(select_ln34_reg_307[8]),
        .I4(cmp6_fu_242_p2_carry__1_0[7]),
        .I5(select_ln34_reg_307[7]),
        .O(cmp6_fu_242_p2_carry_i_2_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    cmp6_fu_242_p2_carry_i_3
       (.I0(select_ln34_reg_307[3]),
        .I1(cmp6_fu_242_p2_carry__1_0[3]),
        .I2(cmp6_fu_242_p2_carry__1_0[5]),
        .I3(select_ln34_reg_307[5]),
        .I4(cmp6_fu_242_p2_carry__1_0[4]),
        .I5(select_ln34_reg_307[4]),
        .O(cmp6_fu_242_p2_carry_i_3_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    cmp6_fu_242_p2_carry_i_4
       (.I0(select_ln34_reg_307[0]),
        .I1(cmp6_fu_242_p2_carry__1_0[0]),
        .I2(cmp6_fu_242_p2_carry__1_0[2]),
        .I3(select_ln34_reg_307[2]),
        .I4(cmp6_fu_242_p2_carry__1_0[1]),
        .I5(select_ln34_reg_307[1]),
        .O(cmp6_fu_242_p2_carry_i_4_n_0));
  LUT6 #(
    .INIT(64'h4000000044444444)) 
    \col_fu_70[30]_i_2 
       (.I0(icmp_ln34_fu_153_p2),
        .I1(ap_enable_reg_pp0_iter1),
        .I2(\ap_CS_fsm_reg[6] [1]),
        .I3(output_stream_TREADY_int_regslice),
        .I4(Q),
        .I5(ap_enable_reg_pp0_iter2_reg_0),
        .O(\col_fu_70[30]_i_2_n_0 ));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[0] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(flow_control_loop_pipe_sequential_init_U_n_0),
        .Q(col_fu_70[0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[10] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[10]),
        .Q(col_fu_70[10]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[11] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[11]),
        .Q(col_fu_70[11]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[12] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[12]),
        .Q(col_fu_70[12]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \col_fu_70_reg[12]_i_1 
       (.CI(\col_fu_70_reg[8]_i_1_n_0 ),
        .CO({\col_fu_70_reg[12]_i_1_n_0 ,\col_fu_70_reg[12]_i_1_n_1 ,\col_fu_70_reg[12]_i_1_n_2 ,\col_fu_70_reg[12]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(add_ln35_fu_210_p2[12:9]),
        .S(col_fu_70[12:9]));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[13] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[13]),
        .Q(col_fu_70[13]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[14] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[14]),
        .Q(col_fu_70[14]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[15] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[15]),
        .Q(col_fu_70[15]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[16] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[16]),
        .Q(col_fu_70[16]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \col_fu_70_reg[16]_i_1 
       (.CI(\col_fu_70_reg[12]_i_1_n_0 ),
        .CO({\col_fu_70_reg[16]_i_1_n_0 ,\col_fu_70_reg[16]_i_1_n_1 ,\col_fu_70_reg[16]_i_1_n_2 ,\col_fu_70_reg[16]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(add_ln35_fu_210_p2[16:13]),
        .S(col_fu_70[16:13]));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[17] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[17]),
        .Q(col_fu_70[17]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[18] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[18]),
        .Q(col_fu_70[18]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[19] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[19]),
        .Q(col_fu_70[19]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[1] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[1]),
        .Q(col_fu_70[1]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[20] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[20]),
        .Q(col_fu_70[20]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \col_fu_70_reg[20]_i_1 
       (.CI(\col_fu_70_reg[16]_i_1_n_0 ),
        .CO({\col_fu_70_reg[20]_i_1_n_0 ,\col_fu_70_reg[20]_i_1_n_1 ,\col_fu_70_reg[20]_i_1_n_2 ,\col_fu_70_reg[20]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(add_ln35_fu_210_p2[20:17]),
        .S(col_fu_70[20:17]));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[21] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[21]),
        .Q(col_fu_70[21]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[22] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[22]),
        .Q(col_fu_70[22]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[23] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[23]),
        .Q(col_fu_70[23]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[24] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[24]),
        .Q(col_fu_70[24]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \col_fu_70_reg[24]_i_1 
       (.CI(\col_fu_70_reg[20]_i_1_n_0 ),
        .CO({\col_fu_70_reg[24]_i_1_n_0 ,\col_fu_70_reg[24]_i_1_n_1 ,\col_fu_70_reg[24]_i_1_n_2 ,\col_fu_70_reg[24]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(add_ln35_fu_210_p2[24:21]),
        .S(col_fu_70[24:21]));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[25] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[25]),
        .Q(col_fu_70[25]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[26] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[26]),
        .Q(col_fu_70[26]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[27] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[27]),
        .Q(col_fu_70[27]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[28] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[28]),
        .Q(col_fu_70[28]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \col_fu_70_reg[28]_i_1 
       (.CI(\col_fu_70_reg[24]_i_1_n_0 ),
        .CO({\col_fu_70_reg[28]_i_1_n_0 ,\col_fu_70_reg[28]_i_1_n_1 ,\col_fu_70_reg[28]_i_1_n_2 ,\col_fu_70_reg[28]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(add_ln35_fu_210_p2[28:25]),
        .S(col_fu_70[28:25]));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[29] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[29]),
        .Q(col_fu_70[29]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[2] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[2]),
        .Q(col_fu_70[2]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[30] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[30]),
        .Q(col_fu_70[30]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \col_fu_70_reg[30]_i_3 
       (.CI(\col_fu_70_reg[28]_i_1_n_0 ),
        .CO({\NLW_col_fu_70_reg[30]_i_3_CO_UNCONNECTED [3:1],\col_fu_70_reg[30]_i_3_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\NLW_col_fu_70_reg[30]_i_3_O_UNCONNECTED [3:2],add_ln35_fu_210_p2[30:29]}),
        .S({1'b0,1'b0,col_fu_70[30:29]}));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[3] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[3]),
        .Q(col_fu_70[3]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[4] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[4]),
        .Q(col_fu_70[4]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \col_fu_70_reg[4]_i_1 
       (.CI(1'b0),
        .CO({\col_fu_70_reg[4]_i_1_n_0 ,\col_fu_70_reg[4]_i_1_n_1 ,\col_fu_70_reg[4]_i_1_n_2 ,\col_fu_70_reg[4]_i_1_n_3 }),
        .CYINIT(col_fu_70[0]),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(add_ln35_fu_210_p2[4:1]),
        .S(col_fu_70[4:1]));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[5] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[5]),
        .Q(col_fu_70[5]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[6] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[6]),
        .Q(col_fu_70[6]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[7] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[7]),
        .Q(col_fu_70[7]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[8] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[8]),
        .Q(col_fu_70[8]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \col_fu_70_reg[8]_i_1 
       (.CI(\col_fu_70_reg[4]_i_1_n_0 ),
        .CO({\col_fu_70_reg[8]_i_1_n_0 ,\col_fu_70_reg[8]_i_1_n_1 ,\col_fu_70_reg[8]_i_1_n_2 ,\col_fu_70_reg[8]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(add_ln35_fu_210_p2[8:5]),
        .S(col_fu_70[8:5]));
  FDRE #(
    .INIT(1'b0)) 
    \col_fu_70_reg[9] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(add_ln35_fu_210_p2[9]),
        .Q(col_fu_70[9]),
        .R(flow_control_loop_pipe_sequential_init_U_n_6));
  LUT6 #(
    .INIT(64'hB888BBBBB8888888)) 
    \data_p1[0]_i_2 
       (.I0(data_p2),
        .I1(\data_p1_reg[0] ),
        .I2(cmp6_fu_242_p2),
        .I3(icmp_ln44_reg_312),
        .I4(input_stream_TREADY_int_regslice),
        .I5(output_stream_TLAST_reg),
        .O(\data_p2_reg[0] ));
  LUT6 #(
    .INIT(64'h8F80FFFF8F800000)) 
    \data_p2[0]_i_1 
       (.I0(cmp6_fu_242_p2),
        .I1(icmp_ln44_reg_312),
        .I2(input_stream_TREADY_int_regslice),
        .I3(output_stream_TLAST_reg),
        .I4(\regslice_both_output_stream_V_last_V_U/load_p2 ),
        .I5(data_p2),
        .O(\icmp_ln44_reg_312_reg[0]_0 ));
  LUT5 #(
    .INIT(32'h80000000)) 
    \data_p2[0]_i_2 
       (.I0(ap_enable_reg_pp0_iter2_reg_0),
        .I1(Q),
        .I2(output_stream_TREADY_int_regslice),
        .I3(\ap_CS_fsm_reg[6] [1]),
        .I4(\data_p2_reg[0]_0 ),
        .O(\regslice_both_output_stream_V_last_V_U/load_p2 ));
  (* SOFT_HLUTNM = "soft_lutpair55" *) 
  LUT5 #(
    .INIT(32'h80000000)) 
    \data_p2[3]_i_1 
       (.I0(ap_enable_reg_pp0_iter2_reg_0),
        .I1(Q),
        .I2(output_stream_TREADY_int_regslice),
        .I3(\ap_CS_fsm_reg[6] [1]),
        .I4(\data_p2_reg[3] ),
        .O(ap_enable_reg_pp0_iter2_reg_1));
  LUT5 #(
    .INIT(32'h80000000)) 
    \data_p2[3]_i_1__0 
       (.I0(ap_enable_reg_pp0_iter2_reg_0),
        .I1(Q),
        .I2(output_stream_TREADY_int_regslice),
        .I3(\ap_CS_fsm_reg[6] [1]),
        .I4(\data_p2_reg[3]_0 ),
        .O(ap_enable_reg_pp0_iter2_reg_2));
  decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_flow_control_loop_pipe_sequential_init flow_control_loop_pipe_sequential_init_U
       (.CO(icmp_ln35_fu_168_p2_carry__2_n_0),
        .D(D),
        .Q(Q),
        .S(flow_control_loop_pipe_sequential_init_U_n_5),
        .\ap_CS_fsm_reg[5] (\ap_CS_fsm[6]_i_2_n_0 ),
        .\ap_CS_fsm_reg[6] (\ap_CS_fsm_reg[6] ),
        .\ap_CS_fsm_reg[6]_0 (\ap_CS_fsm_reg[6]_0 ),
        .ap_block_pp0_stage0_11001__1(ap_block_pp0_stage0_11001__1),
        .ap_clk(ap_clk),
        .ap_done_cache_reg_0(icmp_ln34_fu_153_p2),
        .ap_enable_reg_pp0_iter1(ap_enable_reg_pp0_iter1),
        .ap_enable_reg_pp0_iter1_reg(flow_control_loop_pipe_sequential_init_U_n_6),
        .ap_rst_n(ap_rst_n),
        .ap_rst_n_inv(ap_rst_n_inv),
        .col_fu_70(col_fu_70[0]),
        .\col_fu_70_reg[0] (flow_control_loop_pipe_sequential_init_U_n_0),
        .\col_fu_70_reg[0]_0 (\col_fu_70[30]_i_2_n_0 ),
        .grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg),
        .grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg_reg(flow_control_loop_pipe_sequential_init_U_n_1),
        .icmp_ln34_fu_153_p2_carry__4(indvar_flatten_fu_78_reg[61:60]),
        .icmp_ln34_fu_153_p2_carry__4_0(icmp_ln34_fu_153_p2_carry__4_0[61:60]),
        .\indvar_flatten_fu_78_reg[61] (ap_enable_reg_pp0_iter2_reg_0),
        .output_stream_TREADY_int_regslice(output_stream_TREADY_int_regslice));
  LUT5 #(
    .INIT(32'hEFFFAAAA)) 
    grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg_i_1
       (.I0(\ap_CS_fsm_reg[6] [0]),
        .I1(ap_block_pp0_stage0_11001__1),
        .I2(ap_enable_reg_pp0_iter1),
        .I3(icmp_ln34_fu_153_p2),
        .I4(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg),
        .O(\ap_CS_fsm_reg[4] ));
  CARRY4 icmp_ln34_fu_153_p2_carry
       (.CI(1'b0),
        .CO({icmp_ln34_fu_153_p2_carry_n_0,icmp_ln34_fu_153_p2_carry_n_1,icmp_ln34_fu_153_p2_carry_n_2,icmp_ln34_fu_153_p2_carry_n_3}),
        .CYINIT(1'b1),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(NLW_icmp_ln34_fu_153_p2_carry_O_UNCONNECTED[3:0]),
        .S({icmp_ln34_fu_153_p2_carry_i_1_n_0,icmp_ln34_fu_153_p2_carry_i_2_n_0,icmp_ln34_fu_153_p2_carry_i_3_n_0,icmp_ln34_fu_153_p2_carry_i_4_n_0}));
  CARRY4 icmp_ln34_fu_153_p2_carry__0
       (.CI(icmp_ln34_fu_153_p2_carry_n_0),
        .CO({icmp_ln34_fu_153_p2_carry__0_n_0,icmp_ln34_fu_153_p2_carry__0_n_1,icmp_ln34_fu_153_p2_carry__0_n_2,icmp_ln34_fu_153_p2_carry__0_n_3}),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(NLW_icmp_ln34_fu_153_p2_carry__0_O_UNCONNECTED[3:0]),
        .S({icmp_ln34_fu_153_p2_carry__0_i_1_n_0,icmp_ln34_fu_153_p2_carry__0_i_2_n_0,icmp_ln34_fu_153_p2_carry__0_i_3_n_0,icmp_ln34_fu_153_p2_carry__0_i_4_n_0}));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    icmp_ln34_fu_153_p2_carry__0_i_1
       (.I0(indvar_flatten_fu_78_reg[21]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[21]),
        .I2(icmp_ln34_fu_153_p2_carry__4_0[23]),
        .I3(indvar_flatten_fu_78_reg[23]),
        .I4(icmp_ln34_fu_153_p2_carry__4_0[22]),
        .I5(indvar_flatten_fu_78_reg[22]),
        .O(icmp_ln34_fu_153_p2_carry__0_i_1_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    icmp_ln34_fu_153_p2_carry__0_i_2
       (.I0(indvar_flatten_fu_78_reg[18]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[18]),
        .I2(icmp_ln34_fu_153_p2_carry__4_0[20]),
        .I3(indvar_flatten_fu_78_reg[20]),
        .I4(icmp_ln34_fu_153_p2_carry__4_0[19]),
        .I5(indvar_flatten_fu_78_reg[19]),
        .O(icmp_ln34_fu_153_p2_carry__0_i_2_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    icmp_ln34_fu_153_p2_carry__0_i_3
       (.I0(indvar_flatten_fu_78_reg[15]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[15]),
        .I2(icmp_ln34_fu_153_p2_carry__4_0[17]),
        .I3(indvar_flatten_fu_78_reg[17]),
        .I4(icmp_ln34_fu_153_p2_carry__4_0[16]),
        .I5(indvar_flatten_fu_78_reg[16]),
        .O(icmp_ln34_fu_153_p2_carry__0_i_3_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    icmp_ln34_fu_153_p2_carry__0_i_4
       (.I0(indvar_flatten_fu_78_reg[12]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[12]),
        .I2(icmp_ln34_fu_153_p2_carry__4_0[14]),
        .I3(indvar_flatten_fu_78_reg[14]),
        .I4(icmp_ln34_fu_153_p2_carry__4_0[13]),
        .I5(indvar_flatten_fu_78_reg[13]),
        .O(icmp_ln34_fu_153_p2_carry__0_i_4_n_0));
  CARRY4 icmp_ln34_fu_153_p2_carry__1
       (.CI(icmp_ln34_fu_153_p2_carry__0_n_0),
        .CO({icmp_ln34_fu_153_p2_carry__1_n_0,icmp_ln34_fu_153_p2_carry__1_n_1,icmp_ln34_fu_153_p2_carry__1_n_2,icmp_ln34_fu_153_p2_carry__1_n_3}),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(NLW_icmp_ln34_fu_153_p2_carry__1_O_UNCONNECTED[3:0]),
        .S({icmp_ln34_fu_153_p2_carry__1_i_1_n_0,icmp_ln34_fu_153_p2_carry__1_i_2_n_0,icmp_ln34_fu_153_p2_carry__1_i_3_n_0,icmp_ln34_fu_153_p2_carry__1_i_4_n_0}));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    icmp_ln34_fu_153_p2_carry__1_i_1
       (.I0(indvar_flatten_fu_78_reg[33]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[33]),
        .I2(icmp_ln34_fu_153_p2_carry__4_0[35]),
        .I3(indvar_flatten_fu_78_reg[35]),
        .I4(icmp_ln34_fu_153_p2_carry__4_0[34]),
        .I5(indvar_flatten_fu_78_reg[34]),
        .O(icmp_ln34_fu_153_p2_carry__1_i_1_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    icmp_ln34_fu_153_p2_carry__1_i_2
       (.I0(indvar_flatten_fu_78_reg[30]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[30]),
        .I2(icmp_ln34_fu_153_p2_carry__4_0[32]),
        .I3(indvar_flatten_fu_78_reg[32]),
        .I4(icmp_ln34_fu_153_p2_carry__4_0[31]),
        .I5(indvar_flatten_fu_78_reg[31]),
        .O(icmp_ln34_fu_153_p2_carry__1_i_2_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    icmp_ln34_fu_153_p2_carry__1_i_3
       (.I0(indvar_flatten_fu_78_reg[27]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[27]),
        .I2(icmp_ln34_fu_153_p2_carry__4_0[29]),
        .I3(indvar_flatten_fu_78_reg[29]),
        .I4(icmp_ln34_fu_153_p2_carry__4_0[28]),
        .I5(indvar_flatten_fu_78_reg[28]),
        .O(icmp_ln34_fu_153_p2_carry__1_i_3_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    icmp_ln34_fu_153_p2_carry__1_i_4
       (.I0(indvar_flatten_fu_78_reg[24]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[24]),
        .I2(icmp_ln34_fu_153_p2_carry__4_0[26]),
        .I3(indvar_flatten_fu_78_reg[26]),
        .I4(icmp_ln34_fu_153_p2_carry__4_0[25]),
        .I5(indvar_flatten_fu_78_reg[25]),
        .O(icmp_ln34_fu_153_p2_carry__1_i_4_n_0));
  CARRY4 icmp_ln34_fu_153_p2_carry__2
       (.CI(icmp_ln34_fu_153_p2_carry__1_n_0),
        .CO({icmp_ln34_fu_153_p2_carry__2_n_0,icmp_ln34_fu_153_p2_carry__2_n_1,icmp_ln34_fu_153_p2_carry__2_n_2,icmp_ln34_fu_153_p2_carry__2_n_3}),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(NLW_icmp_ln34_fu_153_p2_carry__2_O_UNCONNECTED[3:0]),
        .S({icmp_ln34_fu_153_p2_carry__2_i_1_n_0,icmp_ln34_fu_153_p2_carry__2_i_2_n_0,icmp_ln34_fu_153_p2_carry__2_i_3_n_0,icmp_ln34_fu_153_p2_carry__2_i_4_n_0}));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    icmp_ln34_fu_153_p2_carry__2_i_1
       (.I0(indvar_flatten_fu_78_reg[45]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[45]),
        .I2(icmp_ln34_fu_153_p2_carry__4_0[47]),
        .I3(indvar_flatten_fu_78_reg[47]),
        .I4(icmp_ln34_fu_153_p2_carry__4_0[46]),
        .I5(indvar_flatten_fu_78_reg[46]),
        .O(icmp_ln34_fu_153_p2_carry__2_i_1_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    icmp_ln34_fu_153_p2_carry__2_i_2
       (.I0(indvar_flatten_fu_78_reg[42]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[42]),
        .I2(icmp_ln34_fu_153_p2_carry__4_0[44]),
        .I3(indvar_flatten_fu_78_reg[44]),
        .I4(icmp_ln34_fu_153_p2_carry__4_0[43]),
        .I5(indvar_flatten_fu_78_reg[43]),
        .O(icmp_ln34_fu_153_p2_carry__2_i_2_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    icmp_ln34_fu_153_p2_carry__2_i_3
       (.I0(indvar_flatten_fu_78_reg[39]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[39]),
        .I2(icmp_ln34_fu_153_p2_carry__4_0[41]),
        .I3(indvar_flatten_fu_78_reg[41]),
        .I4(icmp_ln34_fu_153_p2_carry__4_0[40]),
        .I5(indvar_flatten_fu_78_reg[40]),
        .O(icmp_ln34_fu_153_p2_carry__2_i_3_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    icmp_ln34_fu_153_p2_carry__2_i_4
       (.I0(indvar_flatten_fu_78_reg[36]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[36]),
        .I2(icmp_ln34_fu_153_p2_carry__4_0[38]),
        .I3(indvar_flatten_fu_78_reg[38]),
        .I4(icmp_ln34_fu_153_p2_carry__4_0[37]),
        .I5(indvar_flatten_fu_78_reg[37]),
        .O(icmp_ln34_fu_153_p2_carry__2_i_4_n_0));
  CARRY4 icmp_ln34_fu_153_p2_carry__3
       (.CI(icmp_ln34_fu_153_p2_carry__2_n_0),
        .CO({icmp_ln34_fu_153_p2_carry__3_n_0,icmp_ln34_fu_153_p2_carry__3_n_1,icmp_ln34_fu_153_p2_carry__3_n_2,icmp_ln34_fu_153_p2_carry__3_n_3}),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(NLW_icmp_ln34_fu_153_p2_carry__3_O_UNCONNECTED[3:0]),
        .S({icmp_ln34_fu_153_p2_carry__3_i_1_n_0,icmp_ln34_fu_153_p2_carry__3_i_2_n_0,icmp_ln34_fu_153_p2_carry__3_i_3_n_0,icmp_ln34_fu_153_p2_carry__3_i_4_n_0}));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    icmp_ln34_fu_153_p2_carry__3_i_1
       (.I0(indvar_flatten_fu_78_reg[57]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[57]),
        .I2(icmp_ln34_fu_153_p2_carry__4_0[59]),
        .I3(indvar_flatten_fu_78_reg[59]),
        .I4(icmp_ln34_fu_153_p2_carry__4_0[58]),
        .I5(indvar_flatten_fu_78_reg[58]),
        .O(icmp_ln34_fu_153_p2_carry__3_i_1_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    icmp_ln34_fu_153_p2_carry__3_i_2
       (.I0(indvar_flatten_fu_78_reg[54]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[54]),
        .I2(icmp_ln34_fu_153_p2_carry__4_0[56]),
        .I3(indvar_flatten_fu_78_reg[56]),
        .I4(icmp_ln34_fu_153_p2_carry__4_0[55]),
        .I5(indvar_flatten_fu_78_reg[55]),
        .O(icmp_ln34_fu_153_p2_carry__3_i_2_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    icmp_ln34_fu_153_p2_carry__3_i_3
       (.I0(indvar_flatten_fu_78_reg[51]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[51]),
        .I2(icmp_ln34_fu_153_p2_carry__4_0[53]),
        .I3(indvar_flatten_fu_78_reg[53]),
        .I4(icmp_ln34_fu_153_p2_carry__4_0[52]),
        .I5(indvar_flatten_fu_78_reg[52]),
        .O(icmp_ln34_fu_153_p2_carry__3_i_3_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    icmp_ln34_fu_153_p2_carry__3_i_4
       (.I0(indvar_flatten_fu_78_reg[48]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[48]),
        .I2(icmp_ln34_fu_153_p2_carry__4_0[50]),
        .I3(indvar_flatten_fu_78_reg[50]),
        .I4(icmp_ln34_fu_153_p2_carry__4_0[49]),
        .I5(indvar_flatten_fu_78_reg[49]),
        .O(icmp_ln34_fu_153_p2_carry__3_i_4_n_0));
  CARRY4 icmp_ln34_fu_153_p2_carry__4
       (.CI(icmp_ln34_fu_153_p2_carry__3_n_0),
        .CO({NLW_icmp_ln34_fu_153_p2_carry__4_CO_UNCONNECTED[3:1],icmp_ln34_fu_153_p2}),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(NLW_icmp_ln34_fu_153_p2_carry__4_O_UNCONNECTED[3:0]),
        .S({1'b0,1'b0,1'b0,flow_control_loop_pipe_sequential_init_U_n_5}));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    icmp_ln34_fu_153_p2_carry_i_1
       (.I0(indvar_flatten_fu_78_reg[9]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[9]),
        .I2(icmp_ln34_fu_153_p2_carry__4_0[11]),
        .I3(indvar_flatten_fu_78_reg[11]),
        .I4(icmp_ln34_fu_153_p2_carry__4_0[10]),
        .I5(indvar_flatten_fu_78_reg[10]),
        .O(icmp_ln34_fu_153_p2_carry_i_1_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    icmp_ln34_fu_153_p2_carry_i_2
       (.I0(indvar_flatten_fu_78_reg[6]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[6]),
        .I2(icmp_ln34_fu_153_p2_carry__4_0[8]),
        .I3(indvar_flatten_fu_78_reg[8]),
        .I4(icmp_ln34_fu_153_p2_carry__4_0[7]),
        .I5(indvar_flatten_fu_78_reg[7]),
        .O(icmp_ln34_fu_153_p2_carry_i_2_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    icmp_ln34_fu_153_p2_carry_i_3
       (.I0(indvar_flatten_fu_78_reg[3]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[3]),
        .I2(icmp_ln34_fu_153_p2_carry__4_0[5]),
        .I3(indvar_flatten_fu_78_reg[5]),
        .I4(icmp_ln34_fu_153_p2_carry__4_0[4]),
        .I5(indvar_flatten_fu_78_reg[4]),
        .O(icmp_ln34_fu_153_p2_carry_i_3_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    icmp_ln34_fu_153_p2_carry_i_4
       (.I0(indvar_flatten_fu_78_reg[0]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[0]),
        .I2(icmp_ln34_fu_153_p2_carry__4_0[2]),
        .I3(indvar_flatten_fu_78_reg[2]),
        .I4(icmp_ln34_fu_153_p2_carry__4_0[1]),
        .I5(indvar_flatten_fu_78_reg[1]),
        .O(icmp_ln34_fu_153_p2_carry_i_4_n_0));
  (* COMPARATOR_THRESHOLD = "11" *) 
  CARRY4 icmp_ln35_fu_168_p2_carry
       (.CI(1'b0),
        .CO({icmp_ln35_fu_168_p2_carry_n_0,icmp_ln35_fu_168_p2_carry_n_1,icmp_ln35_fu_168_p2_carry_n_2,icmp_ln35_fu_168_p2_carry_n_3}),
        .CYINIT(1'b0),
        .DI({icmp_ln35_fu_168_p2_carry_i_1_n_0,icmp_ln35_fu_168_p2_carry_i_2_n_0,icmp_ln35_fu_168_p2_carry_i_3_n_0,icmp_ln35_fu_168_p2_carry_i_4_n_0}),
        .O(NLW_icmp_ln35_fu_168_p2_carry_O_UNCONNECTED[3:0]),
        .S({icmp_ln35_fu_168_p2_carry_i_5_n_0,icmp_ln35_fu_168_p2_carry_i_6_n_0,icmp_ln35_fu_168_p2_carry_i_7_n_0,icmp_ln35_fu_168_p2_carry_i_8_n_0}));
  (* COMPARATOR_THRESHOLD = "11" *) 
  CARRY4 icmp_ln35_fu_168_p2_carry__0
       (.CI(icmp_ln35_fu_168_p2_carry_n_0),
        .CO({icmp_ln35_fu_168_p2_carry__0_n_0,icmp_ln35_fu_168_p2_carry__0_n_1,icmp_ln35_fu_168_p2_carry__0_n_2,icmp_ln35_fu_168_p2_carry__0_n_3}),
        .CYINIT(1'b0),
        .DI({icmp_ln35_fu_168_p2_carry__0_i_1_n_0,icmp_ln35_fu_168_p2_carry__0_i_2_n_0,icmp_ln35_fu_168_p2_carry__0_i_3_n_0,icmp_ln35_fu_168_p2_carry__0_i_4_n_0}),
        .O(NLW_icmp_ln35_fu_168_p2_carry__0_O_UNCONNECTED[3:0]),
        .S({icmp_ln35_fu_168_p2_carry__0_i_5_n_0,icmp_ln35_fu_168_p2_carry__0_i_6_n_0,icmp_ln35_fu_168_p2_carry__0_i_7_n_0,icmp_ln35_fu_168_p2_carry__0_i_8_n_0}));
  LUT4 #(
    .INIT(16'h2F02)) 
    icmp_ln35_fu_168_p2_carry__0_i_1
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[14]),
        .I1(col_fu_70[14]),
        .I2(col_fu_70[15]),
        .I3(icmp_ln35_fu_168_p2_carry__2_0[15]),
        .O(icmp_ln35_fu_168_p2_carry__0_i_1_n_0));
  LUT4 #(
    .INIT(16'h2F02)) 
    icmp_ln35_fu_168_p2_carry__0_i_2
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[12]),
        .I1(col_fu_70[12]),
        .I2(col_fu_70[13]),
        .I3(icmp_ln35_fu_168_p2_carry__2_0[13]),
        .O(icmp_ln35_fu_168_p2_carry__0_i_2_n_0));
  LUT4 #(
    .INIT(16'h2F02)) 
    icmp_ln35_fu_168_p2_carry__0_i_3
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[10]),
        .I1(col_fu_70[10]),
        .I2(col_fu_70[11]),
        .I3(icmp_ln35_fu_168_p2_carry__2_0[11]),
        .O(icmp_ln35_fu_168_p2_carry__0_i_3_n_0));
  LUT4 #(
    .INIT(16'h2F02)) 
    icmp_ln35_fu_168_p2_carry__0_i_4
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[8]),
        .I1(col_fu_70[8]),
        .I2(col_fu_70[9]),
        .I3(icmp_ln35_fu_168_p2_carry__2_0[9]),
        .O(icmp_ln35_fu_168_p2_carry__0_i_4_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    icmp_ln35_fu_168_p2_carry__0_i_5
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[14]),
        .I1(col_fu_70[14]),
        .I2(icmp_ln35_fu_168_p2_carry__2_0[15]),
        .I3(col_fu_70[15]),
        .O(icmp_ln35_fu_168_p2_carry__0_i_5_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    icmp_ln35_fu_168_p2_carry__0_i_6
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[12]),
        .I1(col_fu_70[12]),
        .I2(icmp_ln35_fu_168_p2_carry__2_0[13]),
        .I3(col_fu_70[13]),
        .O(icmp_ln35_fu_168_p2_carry__0_i_6_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    icmp_ln35_fu_168_p2_carry__0_i_7
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[10]),
        .I1(col_fu_70[10]),
        .I2(icmp_ln35_fu_168_p2_carry__2_0[11]),
        .I3(col_fu_70[11]),
        .O(icmp_ln35_fu_168_p2_carry__0_i_7_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    icmp_ln35_fu_168_p2_carry__0_i_8
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[8]),
        .I1(col_fu_70[8]),
        .I2(icmp_ln35_fu_168_p2_carry__2_0[9]),
        .I3(col_fu_70[9]),
        .O(icmp_ln35_fu_168_p2_carry__0_i_8_n_0));
  (* COMPARATOR_THRESHOLD = "11" *) 
  CARRY4 icmp_ln35_fu_168_p2_carry__1
       (.CI(icmp_ln35_fu_168_p2_carry__0_n_0),
        .CO({icmp_ln35_fu_168_p2_carry__1_n_0,icmp_ln35_fu_168_p2_carry__1_n_1,icmp_ln35_fu_168_p2_carry__1_n_2,icmp_ln35_fu_168_p2_carry__1_n_3}),
        .CYINIT(1'b0),
        .DI({icmp_ln35_fu_168_p2_carry__1_i_1_n_0,icmp_ln35_fu_168_p2_carry__1_i_2_n_0,icmp_ln35_fu_168_p2_carry__1_i_3_n_0,icmp_ln35_fu_168_p2_carry__1_i_4_n_0}),
        .O(NLW_icmp_ln35_fu_168_p2_carry__1_O_UNCONNECTED[3:0]),
        .S({icmp_ln35_fu_168_p2_carry__1_i_5_n_0,icmp_ln35_fu_168_p2_carry__1_i_6_n_0,icmp_ln35_fu_168_p2_carry__1_i_7_n_0,icmp_ln35_fu_168_p2_carry__1_i_8_n_0}));
  LUT4 #(
    .INIT(16'h2F02)) 
    icmp_ln35_fu_168_p2_carry__1_i_1
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[22]),
        .I1(col_fu_70[22]),
        .I2(col_fu_70[23]),
        .I3(icmp_ln35_fu_168_p2_carry__2_0[23]),
        .O(icmp_ln35_fu_168_p2_carry__1_i_1_n_0));
  LUT4 #(
    .INIT(16'h2F02)) 
    icmp_ln35_fu_168_p2_carry__1_i_2
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[20]),
        .I1(col_fu_70[20]),
        .I2(col_fu_70[21]),
        .I3(icmp_ln35_fu_168_p2_carry__2_0[21]),
        .O(icmp_ln35_fu_168_p2_carry__1_i_2_n_0));
  LUT4 #(
    .INIT(16'h2F02)) 
    icmp_ln35_fu_168_p2_carry__1_i_3
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[18]),
        .I1(col_fu_70[18]),
        .I2(col_fu_70[19]),
        .I3(icmp_ln35_fu_168_p2_carry__2_0[19]),
        .O(icmp_ln35_fu_168_p2_carry__1_i_3_n_0));
  LUT4 #(
    .INIT(16'h2F02)) 
    icmp_ln35_fu_168_p2_carry__1_i_4
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[16]),
        .I1(col_fu_70[16]),
        .I2(col_fu_70[17]),
        .I3(icmp_ln35_fu_168_p2_carry__2_0[17]),
        .O(icmp_ln35_fu_168_p2_carry__1_i_4_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    icmp_ln35_fu_168_p2_carry__1_i_5
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[22]),
        .I1(col_fu_70[22]),
        .I2(icmp_ln35_fu_168_p2_carry__2_0[23]),
        .I3(col_fu_70[23]),
        .O(icmp_ln35_fu_168_p2_carry__1_i_5_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    icmp_ln35_fu_168_p2_carry__1_i_6
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[20]),
        .I1(col_fu_70[20]),
        .I2(icmp_ln35_fu_168_p2_carry__2_0[21]),
        .I3(col_fu_70[21]),
        .O(icmp_ln35_fu_168_p2_carry__1_i_6_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    icmp_ln35_fu_168_p2_carry__1_i_7
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[18]),
        .I1(col_fu_70[18]),
        .I2(icmp_ln35_fu_168_p2_carry__2_0[19]),
        .I3(col_fu_70[19]),
        .O(icmp_ln35_fu_168_p2_carry__1_i_7_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    icmp_ln35_fu_168_p2_carry__1_i_8
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[16]),
        .I1(col_fu_70[16]),
        .I2(icmp_ln35_fu_168_p2_carry__2_0[17]),
        .I3(col_fu_70[17]),
        .O(icmp_ln35_fu_168_p2_carry__1_i_8_n_0));
  (* COMPARATOR_THRESHOLD = "11" *) 
  CARRY4 icmp_ln35_fu_168_p2_carry__2
       (.CI(icmp_ln35_fu_168_p2_carry__1_n_0),
        .CO({icmp_ln35_fu_168_p2_carry__2_n_0,icmp_ln35_fu_168_p2_carry__2_n_1,icmp_ln35_fu_168_p2_carry__2_n_2,icmp_ln35_fu_168_p2_carry__2_n_3}),
        .CYINIT(1'b0),
        .DI({icmp_ln35_fu_168_p2_carry__2_i_1_n_0,icmp_ln35_fu_168_p2_carry__2_i_2_n_0,icmp_ln35_fu_168_p2_carry__2_i_3_n_0,icmp_ln35_fu_168_p2_carry__2_i_4_n_0}),
        .O(NLW_icmp_ln35_fu_168_p2_carry__2_O_UNCONNECTED[3:0]),
        .S({icmp_ln35_fu_168_p2_carry__2_i_5_n_0,icmp_ln35_fu_168_p2_carry__2_i_6_n_0,icmp_ln35_fu_168_p2_carry__2_i_7_n_0,icmp_ln35_fu_168_p2_carry__2_i_8_n_0}));
  LUT3 #(
    .INIT(8'h04)) 
    icmp_ln35_fu_168_p2_carry__2_i_1
       (.I0(col_fu_70[30]),
        .I1(icmp_ln35_fu_168_p2_carry__2_0[30]),
        .I2(icmp_ln35_fu_168_p2_carry__2_0[31]),
        .O(icmp_ln35_fu_168_p2_carry__2_i_1_n_0));
  LUT4 #(
    .INIT(16'h2F02)) 
    icmp_ln35_fu_168_p2_carry__2_i_2
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[28]),
        .I1(col_fu_70[28]),
        .I2(col_fu_70[29]),
        .I3(icmp_ln35_fu_168_p2_carry__2_0[29]),
        .O(icmp_ln35_fu_168_p2_carry__2_i_2_n_0));
  LUT4 #(
    .INIT(16'h2F02)) 
    icmp_ln35_fu_168_p2_carry__2_i_3
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[26]),
        .I1(col_fu_70[26]),
        .I2(col_fu_70[27]),
        .I3(icmp_ln35_fu_168_p2_carry__2_0[27]),
        .O(icmp_ln35_fu_168_p2_carry__2_i_3_n_0));
  LUT4 #(
    .INIT(16'h2F02)) 
    icmp_ln35_fu_168_p2_carry__2_i_4
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[24]),
        .I1(col_fu_70[24]),
        .I2(col_fu_70[25]),
        .I3(icmp_ln35_fu_168_p2_carry__2_0[25]),
        .O(icmp_ln35_fu_168_p2_carry__2_i_4_n_0));
  LUT3 #(
    .INIT(8'h09)) 
    icmp_ln35_fu_168_p2_carry__2_i_5
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[30]),
        .I1(col_fu_70[30]),
        .I2(icmp_ln35_fu_168_p2_carry__2_0[31]),
        .O(icmp_ln35_fu_168_p2_carry__2_i_5_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    icmp_ln35_fu_168_p2_carry__2_i_6
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[28]),
        .I1(col_fu_70[28]),
        .I2(icmp_ln35_fu_168_p2_carry__2_0[29]),
        .I3(col_fu_70[29]),
        .O(icmp_ln35_fu_168_p2_carry__2_i_6_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    icmp_ln35_fu_168_p2_carry__2_i_7
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[26]),
        .I1(col_fu_70[26]),
        .I2(icmp_ln35_fu_168_p2_carry__2_0[27]),
        .I3(col_fu_70[27]),
        .O(icmp_ln35_fu_168_p2_carry__2_i_7_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    icmp_ln35_fu_168_p2_carry__2_i_8
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[24]),
        .I1(col_fu_70[24]),
        .I2(icmp_ln35_fu_168_p2_carry__2_0[25]),
        .I3(col_fu_70[25]),
        .O(icmp_ln35_fu_168_p2_carry__2_i_8_n_0));
  LUT4 #(
    .INIT(16'h2F02)) 
    icmp_ln35_fu_168_p2_carry_i_1
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[6]),
        .I1(col_fu_70[6]),
        .I2(col_fu_70[7]),
        .I3(icmp_ln35_fu_168_p2_carry__2_0[7]),
        .O(icmp_ln35_fu_168_p2_carry_i_1_n_0));
  LUT4 #(
    .INIT(16'h2F02)) 
    icmp_ln35_fu_168_p2_carry_i_2
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[4]),
        .I1(col_fu_70[4]),
        .I2(col_fu_70[5]),
        .I3(icmp_ln35_fu_168_p2_carry__2_0[5]),
        .O(icmp_ln35_fu_168_p2_carry_i_2_n_0));
  LUT4 #(
    .INIT(16'h2F02)) 
    icmp_ln35_fu_168_p2_carry_i_3
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[2]),
        .I1(col_fu_70[2]),
        .I2(col_fu_70[3]),
        .I3(icmp_ln35_fu_168_p2_carry__2_0[3]),
        .O(icmp_ln35_fu_168_p2_carry_i_3_n_0));
  LUT4 #(
    .INIT(16'h2F02)) 
    icmp_ln35_fu_168_p2_carry_i_4
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[0]),
        .I1(col_fu_70[0]),
        .I2(col_fu_70[1]),
        .I3(icmp_ln35_fu_168_p2_carry__2_0[1]),
        .O(icmp_ln35_fu_168_p2_carry_i_4_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    icmp_ln35_fu_168_p2_carry_i_5
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[6]),
        .I1(col_fu_70[6]),
        .I2(icmp_ln35_fu_168_p2_carry__2_0[7]),
        .I3(col_fu_70[7]),
        .O(icmp_ln35_fu_168_p2_carry_i_5_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    icmp_ln35_fu_168_p2_carry_i_6
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[4]),
        .I1(col_fu_70[4]),
        .I2(icmp_ln35_fu_168_p2_carry__2_0[5]),
        .I3(col_fu_70[5]),
        .O(icmp_ln35_fu_168_p2_carry_i_6_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    icmp_ln35_fu_168_p2_carry_i_7
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[2]),
        .I1(col_fu_70[2]),
        .I2(icmp_ln35_fu_168_p2_carry__2_0[3]),
        .I3(col_fu_70[3]),
        .O(icmp_ln35_fu_168_p2_carry_i_7_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    icmp_ln35_fu_168_p2_carry_i_8
       (.I0(icmp_ln35_fu_168_p2_carry__2_0[0]),
        .I1(col_fu_70[0]),
        .I2(icmp_ln35_fu_168_p2_carry__2_0[1]),
        .I3(col_fu_70[1]),
        .O(icmp_ln35_fu_168_p2_carry_i_8_n_0));
  CARRY4 icmp_ln44_fu_205_p2_carry
       (.CI(1'b0),
        .CO({icmp_ln44_fu_205_p2_carry_n_0,icmp_ln44_fu_205_p2_carry_n_1,icmp_ln44_fu_205_p2_carry_n_2,icmp_ln44_fu_205_p2_carry_n_3}),
        .CYINIT(1'b1),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(NLW_icmp_ln44_fu_205_p2_carry_O_UNCONNECTED[3:0]),
        .S({icmp_ln44_fu_205_p2_carry_i_1_n_0,icmp_ln44_fu_205_p2_carry_i_2_n_0,icmp_ln44_fu_205_p2_carry_i_3_n_0,icmp_ln44_fu_205_p2_carry_i_4_n_0}));
  CARRY4 icmp_ln44_fu_205_p2_carry__0
       (.CI(icmp_ln44_fu_205_p2_carry_n_0),
        .CO({icmp_ln44_fu_205_p2_carry__0_n_0,icmp_ln44_fu_205_p2_carry__0_n_1,icmp_ln44_fu_205_p2_carry__0_n_2,icmp_ln44_fu_205_p2_carry__0_n_3}),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(NLW_icmp_ln44_fu_205_p2_carry__0_O_UNCONNECTED[3:0]),
        .S({icmp_ln44_fu_205_p2_carry__0_i_1_n_0,icmp_ln44_fu_205_p2_carry__0_i_2_n_0,icmp_ln44_fu_205_p2_carry__0_i_3_n_0,icmp_ln44_fu_205_p2_carry__0_i_4_n_0}));
  LUT4 #(
    .INIT(16'h8700)) 
    icmp_ln44_fu_205_p2_carry__0_i_1
       (.I0(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I1(col_fu_70[21]),
        .I2(\icmp_ln44_reg_312_reg[0]_1 [21]),
        .I3(icmp_ln44_fu_205_p2_carry__0_i_5_n_0),
        .O(icmp_ln44_fu_205_p2_carry__0_i_1_n_0));
  LUT4 #(
    .INIT(16'h8700)) 
    icmp_ln44_fu_205_p2_carry__0_i_2
       (.I0(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I1(col_fu_70[18]),
        .I2(\icmp_ln44_reg_312_reg[0]_1 [18]),
        .I3(icmp_ln44_fu_205_p2_carry__0_i_6_n_0),
        .O(icmp_ln44_fu_205_p2_carry__0_i_2_n_0));
  LUT4 #(
    .INIT(16'h8700)) 
    icmp_ln44_fu_205_p2_carry__0_i_3
       (.I0(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I1(col_fu_70[15]),
        .I2(\icmp_ln44_reg_312_reg[0]_1 [15]),
        .I3(icmp_ln44_fu_205_p2_carry__0_i_7_n_0),
        .O(icmp_ln44_fu_205_p2_carry__0_i_3_n_0));
  LUT4 #(
    .INIT(16'h8700)) 
    icmp_ln44_fu_205_p2_carry__0_i_4
       (.I0(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I1(col_fu_70[12]),
        .I2(\icmp_ln44_reg_312_reg[0]_1 [12]),
        .I3(icmp_ln44_fu_205_p2_carry__0_i_8_n_0),
        .O(icmp_ln44_fu_205_p2_carry__0_i_4_n_0));
  LUT5 #(
    .INIT(32'h90000393)) 
    icmp_ln44_fu_205_p2_carry__0_i_5
       (.I0(col_fu_70[22]),
        .I1(\icmp_ln44_reg_312_reg[0]_1 [22]),
        .I2(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I3(col_fu_70[23]),
        .I4(\icmp_ln44_reg_312_reg[0]_1 [23]),
        .O(icmp_ln44_fu_205_p2_carry__0_i_5_n_0));
  LUT5 #(
    .INIT(32'h90000393)) 
    icmp_ln44_fu_205_p2_carry__0_i_6
       (.I0(col_fu_70[19]),
        .I1(\icmp_ln44_reg_312_reg[0]_1 [19]),
        .I2(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I3(col_fu_70[20]),
        .I4(\icmp_ln44_reg_312_reg[0]_1 [20]),
        .O(icmp_ln44_fu_205_p2_carry__0_i_6_n_0));
  LUT5 #(
    .INIT(32'h90000393)) 
    icmp_ln44_fu_205_p2_carry__0_i_7
       (.I0(col_fu_70[16]),
        .I1(\icmp_ln44_reg_312_reg[0]_1 [16]),
        .I2(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I3(col_fu_70[17]),
        .I4(\icmp_ln44_reg_312_reg[0]_1 [17]),
        .O(icmp_ln44_fu_205_p2_carry__0_i_7_n_0));
  LUT5 #(
    .INIT(32'h90000393)) 
    icmp_ln44_fu_205_p2_carry__0_i_8
       (.I0(col_fu_70[13]),
        .I1(\icmp_ln44_reg_312_reg[0]_1 [13]),
        .I2(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I3(col_fu_70[14]),
        .I4(\icmp_ln44_reg_312_reg[0]_1 [14]),
        .O(icmp_ln44_fu_205_p2_carry__0_i_8_n_0));
  CARRY4 icmp_ln44_fu_205_p2_carry__1
       (.CI(icmp_ln44_fu_205_p2_carry__0_n_0),
        .CO({NLW_icmp_ln44_fu_205_p2_carry__1_CO_UNCONNECTED[3],icmp_ln44_fu_205_p2,icmp_ln44_fu_205_p2_carry__1_n_2,icmp_ln44_fu_205_p2_carry__1_n_3}),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(NLW_icmp_ln44_fu_205_p2_carry__1_O_UNCONNECTED[3:0]),
        .S({1'b0,icmp_ln44_fu_205_p2_carry__1_i_1_n_0,icmp_ln44_fu_205_p2_carry__1_i_2_n_0,icmp_ln44_fu_205_p2_carry__1_i_3_n_0}));
  LUT4 #(
    .INIT(16'h0087)) 
    icmp_ln44_fu_205_p2_carry__1_i_1
       (.I0(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I1(col_fu_70[30]),
        .I2(\icmp_ln44_reg_312_reg[0]_1 [30]),
        .I3(\icmp_ln44_reg_312_reg[0]_1 [31]),
        .O(icmp_ln44_fu_205_p2_carry__1_i_1_n_0));
  LUT4 #(
    .INIT(16'h8700)) 
    icmp_ln44_fu_205_p2_carry__1_i_2
       (.I0(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I1(col_fu_70[27]),
        .I2(\icmp_ln44_reg_312_reg[0]_1 [27]),
        .I3(icmp_ln44_fu_205_p2_carry__1_i_4_n_0),
        .O(icmp_ln44_fu_205_p2_carry__1_i_2_n_0));
  LUT4 #(
    .INIT(16'h8700)) 
    icmp_ln44_fu_205_p2_carry__1_i_3
       (.I0(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I1(col_fu_70[24]),
        .I2(\icmp_ln44_reg_312_reg[0]_1 [24]),
        .I3(icmp_ln44_fu_205_p2_carry__1_i_5_n_0),
        .O(icmp_ln44_fu_205_p2_carry__1_i_3_n_0));
  LUT5 #(
    .INIT(32'h90000393)) 
    icmp_ln44_fu_205_p2_carry__1_i_4
       (.I0(col_fu_70[28]),
        .I1(\icmp_ln44_reg_312_reg[0]_1 [28]),
        .I2(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I3(col_fu_70[29]),
        .I4(\icmp_ln44_reg_312_reg[0]_1 [29]),
        .O(icmp_ln44_fu_205_p2_carry__1_i_4_n_0));
  LUT5 #(
    .INIT(32'h90000393)) 
    icmp_ln44_fu_205_p2_carry__1_i_5
       (.I0(col_fu_70[25]),
        .I1(\icmp_ln44_reg_312_reg[0]_1 [25]),
        .I2(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I3(col_fu_70[26]),
        .I4(\icmp_ln44_reg_312_reg[0]_1 [26]),
        .O(icmp_ln44_fu_205_p2_carry__1_i_5_n_0));
  LUT4 #(
    .INIT(16'h8700)) 
    icmp_ln44_fu_205_p2_carry_i_1
       (.I0(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I1(col_fu_70[9]),
        .I2(\icmp_ln44_reg_312_reg[0]_1 [9]),
        .I3(icmp_ln44_fu_205_p2_carry_i_5_n_0),
        .O(icmp_ln44_fu_205_p2_carry_i_1_n_0));
  LUT4 #(
    .INIT(16'h8700)) 
    icmp_ln44_fu_205_p2_carry_i_2
       (.I0(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I1(col_fu_70[6]),
        .I2(\icmp_ln44_reg_312_reg[0]_1 [6]),
        .I3(icmp_ln44_fu_205_p2_carry_i_6_n_0),
        .O(icmp_ln44_fu_205_p2_carry_i_2_n_0));
  LUT4 #(
    .INIT(16'h8700)) 
    icmp_ln44_fu_205_p2_carry_i_3
       (.I0(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I1(col_fu_70[3]),
        .I2(\icmp_ln44_reg_312_reg[0]_1 [3]),
        .I3(icmp_ln44_fu_205_p2_carry_i_7_n_0),
        .O(icmp_ln44_fu_205_p2_carry_i_3_n_0));
  LUT4 #(
    .INIT(16'h8700)) 
    icmp_ln44_fu_205_p2_carry_i_4
       (.I0(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I1(col_fu_70[0]),
        .I2(\icmp_ln44_reg_312_reg[0]_1 [0]),
        .I3(icmp_ln44_fu_205_p2_carry_i_8_n_0),
        .O(icmp_ln44_fu_205_p2_carry_i_4_n_0));
  LUT5 #(
    .INIT(32'h90000393)) 
    icmp_ln44_fu_205_p2_carry_i_5
       (.I0(col_fu_70[10]),
        .I1(\icmp_ln44_reg_312_reg[0]_1 [10]),
        .I2(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I3(col_fu_70[11]),
        .I4(\icmp_ln44_reg_312_reg[0]_1 [11]),
        .O(icmp_ln44_fu_205_p2_carry_i_5_n_0));
  LUT5 #(
    .INIT(32'h90000393)) 
    icmp_ln44_fu_205_p2_carry_i_6
       (.I0(col_fu_70[7]),
        .I1(\icmp_ln44_reg_312_reg[0]_1 [7]),
        .I2(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I3(col_fu_70[8]),
        .I4(\icmp_ln44_reg_312_reg[0]_1 [8]),
        .O(icmp_ln44_fu_205_p2_carry_i_6_n_0));
  LUT5 #(
    .INIT(32'h90000393)) 
    icmp_ln44_fu_205_p2_carry_i_7
       (.I0(col_fu_70[4]),
        .I1(\icmp_ln44_reg_312_reg[0]_1 [4]),
        .I2(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I3(col_fu_70[5]),
        .I4(\icmp_ln44_reg_312_reg[0]_1 [5]),
        .O(icmp_ln44_fu_205_p2_carry_i_7_n_0));
  LUT5 #(
    .INIT(32'h90000393)) 
    icmp_ln44_fu_205_p2_carry_i_8
       (.I0(col_fu_70[1]),
        .I1(\icmp_ln44_reg_312_reg[0]_1 [1]),
        .I2(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I3(col_fu_70[2]),
        .I4(\icmp_ln44_reg_312_reg[0]_1 [2]),
        .O(icmp_ln44_fu_205_p2_carry_i_8_n_0));
  FDRE \icmp_ln44_reg_312_reg[0] 
       (.C(ap_clk),
        .CE(E),
        .D(icmp_ln44_fu_205_p2),
        .Q(icmp_ln44_reg_312),
        .R(1'b0));
  LUT1 #(
    .INIT(2'h1)) 
    \indvar_flatten_fu_78[0]_i_3 
       (.I0(indvar_flatten_fu_78_reg[0]),
        .O(\indvar_flatten_fu_78[0]_i_3_n_0 ));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[0] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[0]_i_2_n_7 ),
        .Q(indvar_flatten_fu_78_reg[0]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \indvar_flatten_fu_78_reg[0]_i_2 
       (.CI(1'b0),
        .CO({\indvar_flatten_fu_78_reg[0]_i_2_n_0 ,\indvar_flatten_fu_78_reg[0]_i_2_n_1 ,\indvar_flatten_fu_78_reg[0]_i_2_n_2 ,\indvar_flatten_fu_78_reg[0]_i_2_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b1}),
        .O({\indvar_flatten_fu_78_reg[0]_i_2_n_4 ,\indvar_flatten_fu_78_reg[0]_i_2_n_5 ,\indvar_flatten_fu_78_reg[0]_i_2_n_6 ,\indvar_flatten_fu_78_reg[0]_i_2_n_7 }),
        .S({indvar_flatten_fu_78_reg[3:1],\indvar_flatten_fu_78[0]_i_3_n_0 }));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[10] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[8]_i_1_n_5 ),
        .Q(indvar_flatten_fu_78_reg[10]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[11] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[8]_i_1_n_4 ),
        .Q(indvar_flatten_fu_78_reg[11]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[12] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[12]_i_1_n_7 ),
        .Q(indvar_flatten_fu_78_reg[12]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \indvar_flatten_fu_78_reg[12]_i_1 
       (.CI(\indvar_flatten_fu_78_reg[8]_i_1_n_0 ),
        .CO({\indvar_flatten_fu_78_reg[12]_i_1_n_0 ,\indvar_flatten_fu_78_reg[12]_i_1_n_1 ,\indvar_flatten_fu_78_reg[12]_i_1_n_2 ,\indvar_flatten_fu_78_reg[12]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\indvar_flatten_fu_78_reg[12]_i_1_n_4 ,\indvar_flatten_fu_78_reg[12]_i_1_n_5 ,\indvar_flatten_fu_78_reg[12]_i_1_n_6 ,\indvar_flatten_fu_78_reg[12]_i_1_n_7 }),
        .S(indvar_flatten_fu_78_reg[15:12]));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[13] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[12]_i_1_n_6 ),
        .Q(indvar_flatten_fu_78_reg[13]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[14] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[12]_i_1_n_5 ),
        .Q(indvar_flatten_fu_78_reg[14]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[15] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[12]_i_1_n_4 ),
        .Q(indvar_flatten_fu_78_reg[15]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[16] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[16]_i_1_n_7 ),
        .Q(indvar_flatten_fu_78_reg[16]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \indvar_flatten_fu_78_reg[16]_i_1 
       (.CI(\indvar_flatten_fu_78_reg[12]_i_1_n_0 ),
        .CO({\indvar_flatten_fu_78_reg[16]_i_1_n_0 ,\indvar_flatten_fu_78_reg[16]_i_1_n_1 ,\indvar_flatten_fu_78_reg[16]_i_1_n_2 ,\indvar_flatten_fu_78_reg[16]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\indvar_flatten_fu_78_reg[16]_i_1_n_4 ,\indvar_flatten_fu_78_reg[16]_i_1_n_5 ,\indvar_flatten_fu_78_reg[16]_i_1_n_6 ,\indvar_flatten_fu_78_reg[16]_i_1_n_7 }),
        .S(indvar_flatten_fu_78_reg[19:16]));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[17] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[16]_i_1_n_6 ),
        .Q(indvar_flatten_fu_78_reg[17]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[18] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[16]_i_1_n_5 ),
        .Q(indvar_flatten_fu_78_reg[18]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[19] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[16]_i_1_n_4 ),
        .Q(indvar_flatten_fu_78_reg[19]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[1] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[0]_i_2_n_6 ),
        .Q(indvar_flatten_fu_78_reg[1]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[20] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[20]_i_1_n_7 ),
        .Q(indvar_flatten_fu_78_reg[20]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \indvar_flatten_fu_78_reg[20]_i_1 
       (.CI(\indvar_flatten_fu_78_reg[16]_i_1_n_0 ),
        .CO({\indvar_flatten_fu_78_reg[20]_i_1_n_0 ,\indvar_flatten_fu_78_reg[20]_i_1_n_1 ,\indvar_flatten_fu_78_reg[20]_i_1_n_2 ,\indvar_flatten_fu_78_reg[20]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\indvar_flatten_fu_78_reg[20]_i_1_n_4 ,\indvar_flatten_fu_78_reg[20]_i_1_n_5 ,\indvar_flatten_fu_78_reg[20]_i_1_n_6 ,\indvar_flatten_fu_78_reg[20]_i_1_n_7 }),
        .S(indvar_flatten_fu_78_reg[23:20]));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[21] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[20]_i_1_n_6 ),
        .Q(indvar_flatten_fu_78_reg[21]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[22] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[20]_i_1_n_5 ),
        .Q(indvar_flatten_fu_78_reg[22]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[23] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[20]_i_1_n_4 ),
        .Q(indvar_flatten_fu_78_reg[23]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[24] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[24]_i_1_n_7 ),
        .Q(indvar_flatten_fu_78_reg[24]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \indvar_flatten_fu_78_reg[24]_i_1 
       (.CI(\indvar_flatten_fu_78_reg[20]_i_1_n_0 ),
        .CO({\indvar_flatten_fu_78_reg[24]_i_1_n_0 ,\indvar_flatten_fu_78_reg[24]_i_1_n_1 ,\indvar_flatten_fu_78_reg[24]_i_1_n_2 ,\indvar_flatten_fu_78_reg[24]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\indvar_flatten_fu_78_reg[24]_i_1_n_4 ,\indvar_flatten_fu_78_reg[24]_i_1_n_5 ,\indvar_flatten_fu_78_reg[24]_i_1_n_6 ,\indvar_flatten_fu_78_reg[24]_i_1_n_7 }),
        .S(indvar_flatten_fu_78_reg[27:24]));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[25] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[24]_i_1_n_6 ),
        .Q(indvar_flatten_fu_78_reg[25]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[26] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[24]_i_1_n_5 ),
        .Q(indvar_flatten_fu_78_reg[26]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[27] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[24]_i_1_n_4 ),
        .Q(indvar_flatten_fu_78_reg[27]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[28] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[28]_i_1_n_7 ),
        .Q(indvar_flatten_fu_78_reg[28]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \indvar_flatten_fu_78_reg[28]_i_1 
       (.CI(\indvar_flatten_fu_78_reg[24]_i_1_n_0 ),
        .CO({\indvar_flatten_fu_78_reg[28]_i_1_n_0 ,\indvar_flatten_fu_78_reg[28]_i_1_n_1 ,\indvar_flatten_fu_78_reg[28]_i_1_n_2 ,\indvar_flatten_fu_78_reg[28]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\indvar_flatten_fu_78_reg[28]_i_1_n_4 ,\indvar_flatten_fu_78_reg[28]_i_1_n_5 ,\indvar_flatten_fu_78_reg[28]_i_1_n_6 ,\indvar_flatten_fu_78_reg[28]_i_1_n_7 }),
        .S(indvar_flatten_fu_78_reg[31:28]));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[29] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[28]_i_1_n_6 ),
        .Q(indvar_flatten_fu_78_reg[29]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[2] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[0]_i_2_n_5 ),
        .Q(indvar_flatten_fu_78_reg[2]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[30] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[28]_i_1_n_5 ),
        .Q(indvar_flatten_fu_78_reg[30]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[31] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[28]_i_1_n_4 ),
        .Q(indvar_flatten_fu_78_reg[31]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[32] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[32]_i_1_n_7 ),
        .Q(indvar_flatten_fu_78_reg[32]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \indvar_flatten_fu_78_reg[32]_i_1 
       (.CI(\indvar_flatten_fu_78_reg[28]_i_1_n_0 ),
        .CO({\indvar_flatten_fu_78_reg[32]_i_1_n_0 ,\indvar_flatten_fu_78_reg[32]_i_1_n_1 ,\indvar_flatten_fu_78_reg[32]_i_1_n_2 ,\indvar_flatten_fu_78_reg[32]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\indvar_flatten_fu_78_reg[32]_i_1_n_4 ,\indvar_flatten_fu_78_reg[32]_i_1_n_5 ,\indvar_flatten_fu_78_reg[32]_i_1_n_6 ,\indvar_flatten_fu_78_reg[32]_i_1_n_7 }),
        .S(indvar_flatten_fu_78_reg[35:32]));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[33] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[32]_i_1_n_6 ),
        .Q(indvar_flatten_fu_78_reg[33]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[34] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[32]_i_1_n_5 ),
        .Q(indvar_flatten_fu_78_reg[34]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[35] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[32]_i_1_n_4 ),
        .Q(indvar_flatten_fu_78_reg[35]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[36] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[36]_i_1_n_7 ),
        .Q(indvar_flatten_fu_78_reg[36]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \indvar_flatten_fu_78_reg[36]_i_1 
       (.CI(\indvar_flatten_fu_78_reg[32]_i_1_n_0 ),
        .CO({\indvar_flatten_fu_78_reg[36]_i_1_n_0 ,\indvar_flatten_fu_78_reg[36]_i_1_n_1 ,\indvar_flatten_fu_78_reg[36]_i_1_n_2 ,\indvar_flatten_fu_78_reg[36]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\indvar_flatten_fu_78_reg[36]_i_1_n_4 ,\indvar_flatten_fu_78_reg[36]_i_1_n_5 ,\indvar_flatten_fu_78_reg[36]_i_1_n_6 ,\indvar_flatten_fu_78_reg[36]_i_1_n_7 }),
        .S(indvar_flatten_fu_78_reg[39:36]));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[37] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[36]_i_1_n_6 ),
        .Q(indvar_flatten_fu_78_reg[37]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[38] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[36]_i_1_n_5 ),
        .Q(indvar_flatten_fu_78_reg[38]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[39] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[36]_i_1_n_4 ),
        .Q(indvar_flatten_fu_78_reg[39]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[3] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[0]_i_2_n_4 ),
        .Q(indvar_flatten_fu_78_reg[3]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[40] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[40]_i_1_n_7 ),
        .Q(indvar_flatten_fu_78_reg[40]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \indvar_flatten_fu_78_reg[40]_i_1 
       (.CI(\indvar_flatten_fu_78_reg[36]_i_1_n_0 ),
        .CO({\indvar_flatten_fu_78_reg[40]_i_1_n_0 ,\indvar_flatten_fu_78_reg[40]_i_1_n_1 ,\indvar_flatten_fu_78_reg[40]_i_1_n_2 ,\indvar_flatten_fu_78_reg[40]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\indvar_flatten_fu_78_reg[40]_i_1_n_4 ,\indvar_flatten_fu_78_reg[40]_i_1_n_5 ,\indvar_flatten_fu_78_reg[40]_i_1_n_6 ,\indvar_flatten_fu_78_reg[40]_i_1_n_7 }),
        .S(indvar_flatten_fu_78_reg[43:40]));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[41] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[40]_i_1_n_6 ),
        .Q(indvar_flatten_fu_78_reg[41]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[42] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[40]_i_1_n_5 ),
        .Q(indvar_flatten_fu_78_reg[42]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[43] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[40]_i_1_n_4 ),
        .Q(indvar_flatten_fu_78_reg[43]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[44] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[44]_i_1_n_7 ),
        .Q(indvar_flatten_fu_78_reg[44]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \indvar_flatten_fu_78_reg[44]_i_1 
       (.CI(\indvar_flatten_fu_78_reg[40]_i_1_n_0 ),
        .CO({\indvar_flatten_fu_78_reg[44]_i_1_n_0 ,\indvar_flatten_fu_78_reg[44]_i_1_n_1 ,\indvar_flatten_fu_78_reg[44]_i_1_n_2 ,\indvar_flatten_fu_78_reg[44]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\indvar_flatten_fu_78_reg[44]_i_1_n_4 ,\indvar_flatten_fu_78_reg[44]_i_1_n_5 ,\indvar_flatten_fu_78_reg[44]_i_1_n_6 ,\indvar_flatten_fu_78_reg[44]_i_1_n_7 }),
        .S(indvar_flatten_fu_78_reg[47:44]));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[45] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[44]_i_1_n_6 ),
        .Q(indvar_flatten_fu_78_reg[45]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[46] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[44]_i_1_n_5 ),
        .Q(indvar_flatten_fu_78_reg[46]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[47] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[44]_i_1_n_4 ),
        .Q(indvar_flatten_fu_78_reg[47]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[48] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[48]_i_1_n_7 ),
        .Q(indvar_flatten_fu_78_reg[48]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \indvar_flatten_fu_78_reg[48]_i_1 
       (.CI(\indvar_flatten_fu_78_reg[44]_i_1_n_0 ),
        .CO({\indvar_flatten_fu_78_reg[48]_i_1_n_0 ,\indvar_flatten_fu_78_reg[48]_i_1_n_1 ,\indvar_flatten_fu_78_reg[48]_i_1_n_2 ,\indvar_flatten_fu_78_reg[48]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\indvar_flatten_fu_78_reg[48]_i_1_n_4 ,\indvar_flatten_fu_78_reg[48]_i_1_n_5 ,\indvar_flatten_fu_78_reg[48]_i_1_n_6 ,\indvar_flatten_fu_78_reg[48]_i_1_n_7 }),
        .S(indvar_flatten_fu_78_reg[51:48]));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[49] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[48]_i_1_n_6 ),
        .Q(indvar_flatten_fu_78_reg[49]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[4] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[4]_i_1_n_7 ),
        .Q(indvar_flatten_fu_78_reg[4]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \indvar_flatten_fu_78_reg[4]_i_1 
       (.CI(\indvar_flatten_fu_78_reg[0]_i_2_n_0 ),
        .CO({\indvar_flatten_fu_78_reg[4]_i_1_n_0 ,\indvar_flatten_fu_78_reg[4]_i_1_n_1 ,\indvar_flatten_fu_78_reg[4]_i_1_n_2 ,\indvar_flatten_fu_78_reg[4]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\indvar_flatten_fu_78_reg[4]_i_1_n_4 ,\indvar_flatten_fu_78_reg[4]_i_1_n_5 ,\indvar_flatten_fu_78_reg[4]_i_1_n_6 ,\indvar_flatten_fu_78_reg[4]_i_1_n_7 }),
        .S(indvar_flatten_fu_78_reg[7:4]));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[50] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[48]_i_1_n_5 ),
        .Q(indvar_flatten_fu_78_reg[50]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[51] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[48]_i_1_n_4 ),
        .Q(indvar_flatten_fu_78_reg[51]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[52] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[52]_i_1_n_7 ),
        .Q(indvar_flatten_fu_78_reg[52]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \indvar_flatten_fu_78_reg[52]_i_1 
       (.CI(\indvar_flatten_fu_78_reg[48]_i_1_n_0 ),
        .CO({\indvar_flatten_fu_78_reg[52]_i_1_n_0 ,\indvar_flatten_fu_78_reg[52]_i_1_n_1 ,\indvar_flatten_fu_78_reg[52]_i_1_n_2 ,\indvar_flatten_fu_78_reg[52]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\indvar_flatten_fu_78_reg[52]_i_1_n_4 ,\indvar_flatten_fu_78_reg[52]_i_1_n_5 ,\indvar_flatten_fu_78_reg[52]_i_1_n_6 ,\indvar_flatten_fu_78_reg[52]_i_1_n_7 }),
        .S(indvar_flatten_fu_78_reg[55:52]));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[53] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[52]_i_1_n_6 ),
        .Q(indvar_flatten_fu_78_reg[53]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[54] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[52]_i_1_n_5 ),
        .Q(indvar_flatten_fu_78_reg[54]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[55] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[52]_i_1_n_4 ),
        .Q(indvar_flatten_fu_78_reg[55]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[56] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[56]_i_1_n_7 ),
        .Q(indvar_flatten_fu_78_reg[56]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \indvar_flatten_fu_78_reg[56]_i_1 
       (.CI(\indvar_flatten_fu_78_reg[52]_i_1_n_0 ),
        .CO({\indvar_flatten_fu_78_reg[56]_i_1_n_0 ,\indvar_flatten_fu_78_reg[56]_i_1_n_1 ,\indvar_flatten_fu_78_reg[56]_i_1_n_2 ,\indvar_flatten_fu_78_reg[56]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\indvar_flatten_fu_78_reg[56]_i_1_n_4 ,\indvar_flatten_fu_78_reg[56]_i_1_n_5 ,\indvar_flatten_fu_78_reg[56]_i_1_n_6 ,\indvar_flatten_fu_78_reg[56]_i_1_n_7 }),
        .S(indvar_flatten_fu_78_reg[59:56]));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[57] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[56]_i_1_n_6 ),
        .Q(indvar_flatten_fu_78_reg[57]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[58] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[56]_i_1_n_5 ),
        .Q(indvar_flatten_fu_78_reg[58]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[59] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[56]_i_1_n_4 ),
        .Q(indvar_flatten_fu_78_reg[59]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[5] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[4]_i_1_n_6 ),
        .Q(indvar_flatten_fu_78_reg[5]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[60] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[60]_i_1_n_7 ),
        .Q(indvar_flatten_fu_78_reg[60]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \indvar_flatten_fu_78_reg[60]_i_1 
       (.CI(\indvar_flatten_fu_78_reg[56]_i_1_n_0 ),
        .CO({\NLW_indvar_flatten_fu_78_reg[60]_i_1_CO_UNCONNECTED [3:1],\indvar_flatten_fu_78_reg[60]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\NLW_indvar_flatten_fu_78_reg[60]_i_1_O_UNCONNECTED [3:2],\indvar_flatten_fu_78_reg[60]_i_1_n_6 ,\indvar_flatten_fu_78_reg[60]_i_1_n_7 }),
        .S({1'b0,1'b0,indvar_flatten_fu_78_reg[61:60]}));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[61] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[60]_i_1_n_6 ),
        .Q(indvar_flatten_fu_78_reg[61]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[6] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[4]_i_1_n_5 ),
        .Q(indvar_flatten_fu_78_reg[6]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[7] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[4]_i_1_n_4 ),
        .Q(indvar_flatten_fu_78_reg[7]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[8] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[8]_i_1_n_7 ),
        .Q(indvar_flatten_fu_78_reg[8]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \indvar_flatten_fu_78_reg[8]_i_1 
       (.CI(\indvar_flatten_fu_78_reg[4]_i_1_n_0 ),
        .CO({\indvar_flatten_fu_78_reg[8]_i_1_n_0 ,\indvar_flatten_fu_78_reg[8]_i_1_n_1 ,\indvar_flatten_fu_78_reg[8]_i_1_n_2 ,\indvar_flatten_fu_78_reg[8]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\indvar_flatten_fu_78_reg[8]_i_1_n_4 ,\indvar_flatten_fu_78_reg[8]_i_1_n_5 ,\indvar_flatten_fu_78_reg[8]_i_1_n_6 ,\indvar_flatten_fu_78_reg[8]_i_1_n_7 }),
        .S(indvar_flatten_fu_78_reg[11:8]));
  FDRE #(
    .INIT(1'b0)) 
    \indvar_flatten_fu_78_reg[9] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\indvar_flatten_fu_78_reg[8]_i_1_n_6 ),
        .Q(indvar_flatten_fu_78_reg[9]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* SOFT_HLUTNM = "soft_lutpair55" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \output_stream_TKEEP_reg[3]_i_1 
       (.I0(ap_enable_reg_pp0_iter2_reg_0),
        .I1(Q),
        .I2(output_stream_TREADY_int_regslice),
        .I3(\ap_CS_fsm_reg[6] [1]),
        .O(input_stream_TREADY_int_regslice));
  LUT2 #(
    .INIT(4'h8)) 
    \output_stream_TLAST_reg[0]_i_1 
       (.I0(cmp6_fu_242_p2),
        .I1(icmp_ln44_reg_312),
        .O(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TLAST));
  LUT2 #(
    .INIT(4'h9)) 
    \row_fu_74[0]_i_2 
       (.I0(icmp_ln35_fu_168_p2_carry__2_n_0),
        .I1(row_fu_74_reg[0]),
        .O(\row_fu_74[0]_i_2_n_0 ));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[0] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[0]_i_1_n_7 ),
        .Q(row_fu_74_reg[0]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \row_fu_74_reg[0]_i_1 
       (.CI(1'b0),
        .CO({\row_fu_74_reg[0]_i_1_n_0 ,\row_fu_74_reg[0]_i_1_n_1 ,\row_fu_74_reg[0]_i_1_n_2 ,\row_fu_74_reg[0]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,row_fu_74_reg[0]}),
        .O({\row_fu_74_reg[0]_i_1_n_4 ,\row_fu_74_reg[0]_i_1_n_5 ,\row_fu_74_reg[0]_i_1_n_6 ,\row_fu_74_reg[0]_i_1_n_7 }),
        .S({row_fu_74_reg[3:1],\row_fu_74[0]_i_2_n_0 }));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[10] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[8]_i_1_n_5 ),
        .Q(row_fu_74_reg[10]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[11] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[8]_i_1_n_4 ),
        .Q(row_fu_74_reg[11]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[12] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[12]_i_1_n_7 ),
        .Q(row_fu_74_reg[12]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \row_fu_74_reg[12]_i_1 
       (.CI(\row_fu_74_reg[8]_i_1_n_0 ),
        .CO({\row_fu_74_reg[12]_i_1_n_0 ,\row_fu_74_reg[12]_i_1_n_1 ,\row_fu_74_reg[12]_i_1_n_2 ,\row_fu_74_reg[12]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\row_fu_74_reg[12]_i_1_n_4 ,\row_fu_74_reg[12]_i_1_n_5 ,\row_fu_74_reg[12]_i_1_n_6 ,\row_fu_74_reg[12]_i_1_n_7 }),
        .S(row_fu_74_reg[15:12]));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[13] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[12]_i_1_n_6 ),
        .Q(row_fu_74_reg[13]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[14] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[12]_i_1_n_5 ),
        .Q(row_fu_74_reg[14]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[15] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[12]_i_1_n_4 ),
        .Q(row_fu_74_reg[15]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[16] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[16]_i_1_n_7 ),
        .Q(row_fu_74_reg[16]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \row_fu_74_reg[16]_i_1 
       (.CI(\row_fu_74_reg[12]_i_1_n_0 ),
        .CO({\row_fu_74_reg[16]_i_1_n_0 ,\row_fu_74_reg[16]_i_1_n_1 ,\row_fu_74_reg[16]_i_1_n_2 ,\row_fu_74_reg[16]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\row_fu_74_reg[16]_i_1_n_4 ,\row_fu_74_reg[16]_i_1_n_5 ,\row_fu_74_reg[16]_i_1_n_6 ,\row_fu_74_reg[16]_i_1_n_7 }),
        .S(row_fu_74_reg[19:16]));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[17] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[16]_i_1_n_6 ),
        .Q(row_fu_74_reg[17]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[18] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[16]_i_1_n_5 ),
        .Q(row_fu_74_reg[18]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[19] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[16]_i_1_n_4 ),
        .Q(row_fu_74_reg[19]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[1] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[0]_i_1_n_6 ),
        .Q(row_fu_74_reg[1]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[20] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[20]_i_1_n_7 ),
        .Q(row_fu_74_reg[20]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \row_fu_74_reg[20]_i_1 
       (.CI(\row_fu_74_reg[16]_i_1_n_0 ),
        .CO({\row_fu_74_reg[20]_i_1_n_0 ,\row_fu_74_reg[20]_i_1_n_1 ,\row_fu_74_reg[20]_i_1_n_2 ,\row_fu_74_reg[20]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\row_fu_74_reg[20]_i_1_n_4 ,\row_fu_74_reg[20]_i_1_n_5 ,\row_fu_74_reg[20]_i_1_n_6 ,\row_fu_74_reg[20]_i_1_n_7 }),
        .S(row_fu_74_reg[23:20]));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[21] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[20]_i_1_n_6 ),
        .Q(row_fu_74_reg[21]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[22] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[20]_i_1_n_5 ),
        .Q(row_fu_74_reg[22]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[23] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[20]_i_1_n_4 ),
        .Q(row_fu_74_reg[23]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[24] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[24]_i_1_n_7 ),
        .Q(row_fu_74_reg[24]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \row_fu_74_reg[24]_i_1 
       (.CI(\row_fu_74_reg[20]_i_1_n_0 ),
        .CO({\row_fu_74_reg[24]_i_1_n_0 ,\row_fu_74_reg[24]_i_1_n_1 ,\row_fu_74_reg[24]_i_1_n_2 ,\row_fu_74_reg[24]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\row_fu_74_reg[24]_i_1_n_4 ,\row_fu_74_reg[24]_i_1_n_5 ,\row_fu_74_reg[24]_i_1_n_6 ,\row_fu_74_reg[24]_i_1_n_7 }),
        .S(row_fu_74_reg[27:24]));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[25] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[24]_i_1_n_6 ),
        .Q(row_fu_74_reg[25]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[26] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[24]_i_1_n_5 ),
        .Q(row_fu_74_reg[26]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[27] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[24]_i_1_n_4 ),
        .Q(row_fu_74_reg[27]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[28] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[28]_i_1_n_7 ),
        .Q(row_fu_74_reg[28]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \row_fu_74_reg[28]_i_1 
       (.CI(\row_fu_74_reg[24]_i_1_n_0 ),
        .CO({\NLW_row_fu_74_reg[28]_i_1_CO_UNCONNECTED [3:2],\row_fu_74_reg[28]_i_1_n_2 ,\row_fu_74_reg[28]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\NLW_row_fu_74_reg[28]_i_1_O_UNCONNECTED [3],\row_fu_74_reg[28]_i_1_n_5 ,\row_fu_74_reg[28]_i_1_n_6 ,\row_fu_74_reg[28]_i_1_n_7 }),
        .S({1'b0,row_fu_74_reg[30:28]}));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[29] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[28]_i_1_n_6 ),
        .Q(row_fu_74_reg[29]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[2] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[0]_i_1_n_5 ),
        .Q(row_fu_74_reg[2]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[30] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[28]_i_1_n_5 ),
        .Q(row_fu_74_reg[30]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[3] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[0]_i_1_n_4 ),
        .Q(row_fu_74_reg[3]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[4] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[4]_i_1_n_7 ),
        .Q(row_fu_74_reg[4]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \row_fu_74_reg[4]_i_1 
       (.CI(\row_fu_74_reg[0]_i_1_n_0 ),
        .CO({\row_fu_74_reg[4]_i_1_n_0 ,\row_fu_74_reg[4]_i_1_n_1 ,\row_fu_74_reg[4]_i_1_n_2 ,\row_fu_74_reg[4]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\row_fu_74_reg[4]_i_1_n_4 ,\row_fu_74_reg[4]_i_1_n_5 ,\row_fu_74_reg[4]_i_1_n_6 ,\row_fu_74_reg[4]_i_1_n_7 }),
        .S(row_fu_74_reg[7:4]));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[5] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[4]_i_1_n_6 ),
        .Q(row_fu_74_reg[5]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[6] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[4]_i_1_n_5 ),
        .Q(row_fu_74_reg[6]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[7] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[4]_i_1_n_4 ),
        .Q(row_fu_74_reg[7]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[8] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[8]_i_1_n_7 ),
        .Q(row_fu_74_reg[8]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  (* ADDER_THRESHOLD = "11" *) 
  CARRY4 \row_fu_74_reg[8]_i_1 
       (.CI(\row_fu_74_reg[4]_i_1_n_0 ),
        .CO({\row_fu_74_reg[8]_i_1_n_0 ,\row_fu_74_reg[8]_i_1_n_1 ,\row_fu_74_reg[8]_i_1_n_2 ,\row_fu_74_reg[8]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\row_fu_74_reg[8]_i_1_n_4 ,\row_fu_74_reg[8]_i_1_n_5 ,\row_fu_74_reg[8]_i_1_n_6 ,\row_fu_74_reg[8]_i_1_n_7 }),
        .S(row_fu_74_reg[11:8]));
  FDRE #(
    .INIT(1'b0)) 
    \row_fu_74_reg[9] 
       (.C(ap_clk),
        .CE(\col_fu_70[30]_i_2_n_0 ),
        .D(\row_fu_74_reg[8]_i_1_n_6 ),
        .Q(row_fu_74_reg[9]),
        .R(flow_control_loop_pipe_sequential_init_U_n_1));
  LUT2 #(
    .INIT(4'h9)) 
    \select_ln34_reg_307[3]_i_2 
       (.I0(row_fu_74_reg[0]),
        .I1(icmp_ln35_fu_168_p2_carry__2_n_0),
        .O(\select_ln34_reg_307[3]_i_2_n_0 ));
  FDRE \select_ln34_reg_307_reg[0] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[0]),
        .Q(select_ln34_reg_307[0]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[10] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[10]),
        .Q(select_ln34_reg_307[10]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[11] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[11]),
        .Q(select_ln34_reg_307[11]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \select_ln34_reg_307_reg[11]_i_1 
       (.CI(\select_ln34_reg_307_reg[7]_i_1_n_0 ),
        .CO({\select_ln34_reg_307_reg[11]_i_1_n_0 ,\select_ln34_reg_307_reg[11]_i_1_n_1 ,\select_ln34_reg_307_reg[11]_i_1_n_2 ,\select_ln34_reg_307_reg[11]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(select_ln34_fu_185_p3[11:8]),
        .S(row_fu_74_reg[11:8]));
  FDRE \select_ln34_reg_307_reg[12] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[12]),
        .Q(select_ln34_reg_307[12]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[13] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[13]),
        .Q(select_ln34_reg_307[13]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[14] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[14]),
        .Q(select_ln34_reg_307[14]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[15] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[15]),
        .Q(select_ln34_reg_307[15]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \select_ln34_reg_307_reg[15]_i_1 
       (.CI(\select_ln34_reg_307_reg[11]_i_1_n_0 ),
        .CO({\select_ln34_reg_307_reg[15]_i_1_n_0 ,\select_ln34_reg_307_reg[15]_i_1_n_1 ,\select_ln34_reg_307_reg[15]_i_1_n_2 ,\select_ln34_reg_307_reg[15]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(select_ln34_fu_185_p3[15:12]),
        .S(row_fu_74_reg[15:12]));
  FDRE \select_ln34_reg_307_reg[16] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[16]),
        .Q(select_ln34_reg_307[16]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[17] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[17]),
        .Q(select_ln34_reg_307[17]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[18] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[18]),
        .Q(select_ln34_reg_307[18]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[19] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[19]),
        .Q(select_ln34_reg_307[19]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \select_ln34_reg_307_reg[19]_i_1 
       (.CI(\select_ln34_reg_307_reg[15]_i_1_n_0 ),
        .CO({\select_ln34_reg_307_reg[19]_i_1_n_0 ,\select_ln34_reg_307_reg[19]_i_1_n_1 ,\select_ln34_reg_307_reg[19]_i_1_n_2 ,\select_ln34_reg_307_reg[19]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(select_ln34_fu_185_p3[19:16]),
        .S(row_fu_74_reg[19:16]));
  FDRE \select_ln34_reg_307_reg[1] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[1]),
        .Q(select_ln34_reg_307[1]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[20] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[20]),
        .Q(select_ln34_reg_307[20]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[21] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[21]),
        .Q(select_ln34_reg_307[21]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[22] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[22]),
        .Q(select_ln34_reg_307[22]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[23] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[23]),
        .Q(select_ln34_reg_307[23]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \select_ln34_reg_307_reg[23]_i_1 
       (.CI(\select_ln34_reg_307_reg[19]_i_1_n_0 ),
        .CO({\select_ln34_reg_307_reg[23]_i_1_n_0 ,\select_ln34_reg_307_reg[23]_i_1_n_1 ,\select_ln34_reg_307_reg[23]_i_1_n_2 ,\select_ln34_reg_307_reg[23]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(select_ln34_fu_185_p3[23:20]),
        .S(row_fu_74_reg[23:20]));
  FDRE \select_ln34_reg_307_reg[24] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[24]),
        .Q(select_ln34_reg_307[24]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[25] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[25]),
        .Q(select_ln34_reg_307[25]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[26] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[26]),
        .Q(select_ln34_reg_307[26]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[27] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[27]),
        .Q(select_ln34_reg_307[27]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \select_ln34_reg_307_reg[27]_i_1 
       (.CI(\select_ln34_reg_307_reg[23]_i_1_n_0 ),
        .CO({\select_ln34_reg_307_reg[27]_i_1_n_0 ,\select_ln34_reg_307_reg[27]_i_1_n_1 ,\select_ln34_reg_307_reg[27]_i_1_n_2 ,\select_ln34_reg_307_reg[27]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(select_ln34_fu_185_p3[27:24]),
        .S(row_fu_74_reg[27:24]));
  FDRE \select_ln34_reg_307_reg[28] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[28]),
        .Q(select_ln34_reg_307[28]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[29] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[29]),
        .Q(select_ln34_reg_307[29]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[2] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[2]),
        .Q(select_ln34_reg_307[2]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[30] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[30]),
        .Q(select_ln34_reg_307[30]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \select_ln34_reg_307_reg[30]_i_1 
       (.CI(\select_ln34_reg_307_reg[27]_i_1_n_0 ),
        .CO({\NLW_select_ln34_reg_307_reg[30]_i_1_CO_UNCONNECTED [3:2],\select_ln34_reg_307_reg[30]_i_1_n_2 ,\select_ln34_reg_307_reg[30]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\NLW_select_ln34_reg_307_reg[30]_i_1_O_UNCONNECTED [3],select_ln34_fu_185_p3[30:28]}),
        .S({1'b0,row_fu_74_reg[30:28]}));
  FDRE \select_ln34_reg_307_reg[3] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[3]),
        .Q(select_ln34_reg_307[3]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \select_ln34_reg_307_reg[3]_i_1 
       (.CI(1'b0),
        .CO({\select_ln34_reg_307_reg[3]_i_1_n_0 ,\select_ln34_reg_307_reg[3]_i_1_n_1 ,\select_ln34_reg_307_reg[3]_i_1_n_2 ,\select_ln34_reg_307_reg[3]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,row_fu_74_reg[0]}),
        .O(select_ln34_fu_185_p3[3:0]),
        .S({row_fu_74_reg[3:1],\select_ln34_reg_307[3]_i_2_n_0 }));
  FDRE \select_ln34_reg_307_reg[4] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[4]),
        .Q(select_ln34_reg_307[4]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[5] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[5]),
        .Q(select_ln34_reg_307[5]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[6] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[6]),
        .Q(select_ln34_reg_307[6]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[7] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[7]),
        .Q(select_ln34_reg_307[7]),
        .R(1'b0));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \select_ln34_reg_307_reg[7]_i_1 
       (.CI(\select_ln34_reg_307_reg[3]_i_1_n_0 ),
        .CO({\select_ln34_reg_307_reg[7]_i_1_n_0 ,\select_ln34_reg_307_reg[7]_i_1_n_1 ,\select_ln34_reg_307_reg[7]_i_1_n_2 ,\select_ln34_reg_307_reg[7]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(select_ln34_fu_185_p3[7:4]),
        .S(row_fu_74_reg[7:4]));
  FDRE \select_ln34_reg_307_reg[8] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[8]),
        .Q(select_ln34_reg_307[8]),
        .R(1'b0));
  FDRE \select_ln34_reg_307_reg[9] 
       (.C(ap_clk),
        .CE(E),
        .D(select_ln34_fu_185_p3[9]),
        .Q(select_ln34_reg_307[9]),
        .R(1'b0));
endmodule

module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_flow_control_loop_pipe_sequential_init
   (\col_fu_70_reg[0] ,
    grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg_reg,
    D,
    ap_block_pp0_stage0_11001__1,
    S,
    ap_enable_reg_pp0_iter1_reg,
    ap_rst_n_inv,
    ap_clk,
    col_fu_70,
    \col_fu_70_reg[0]_0 ,
    CO,
    \ap_CS_fsm_reg[6] ,
    grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg,
    \ap_CS_fsm_reg[5] ,
    \ap_CS_fsm_reg[6]_0 ,
    ap_rst_n,
    ap_enable_reg_pp0_iter1,
    ap_done_cache_reg_0,
    \indvar_flatten_fu_78_reg[61] ,
    Q,
    output_stream_TREADY_int_regslice,
    icmp_ln34_fu_153_p2_carry__4,
    icmp_ln34_fu_153_p2_carry__4_0);
  output \col_fu_70_reg[0] ;
  output grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg_reg;
  output [1:0]D;
  output ap_block_pp0_stage0_11001__1;
  output [0:0]S;
  output ap_enable_reg_pp0_iter1_reg;
  input ap_rst_n_inv;
  input ap_clk;
  input [0:0]col_fu_70;
  input \col_fu_70_reg[0]_0 ;
  input [0:0]CO;
  input [2:0]\ap_CS_fsm_reg[6] ;
  input grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg;
  input \ap_CS_fsm_reg[5] ;
  input \ap_CS_fsm_reg[6]_0 ;
  input ap_rst_n;
  input ap_enable_reg_pp0_iter1;
  input [0:0]ap_done_cache_reg_0;
  input \indvar_flatten_fu_78_reg[61] ;
  input [0:0]Q;
  input output_stream_TREADY_int_regslice;
  input [1:0]icmp_ln34_fu_153_p2_carry__4;
  input [1:0]icmp_ln34_fu_153_p2_carry__4_0;

  wire [0:0]CO;
  wire [1:0]D;
  wire [0:0]Q;
  wire [0:0]S;
  wire \ap_CS_fsm_reg[5] ;
  wire [2:0]\ap_CS_fsm_reg[6] ;
  wire \ap_CS_fsm_reg[6]_0 ;
  wire ap_block_pp0_stage0_11001__1;
  wire ap_clk;
  wire ap_done_cache;
  wire ap_done_cache_i_1_n_0;
  wire [0:0]ap_done_cache_reg_0;
  wire ap_enable_reg_pp0_iter1;
  wire ap_enable_reg_pp0_iter1_reg;
  wire ap_loop_init_int;
  wire ap_loop_init_int_i_1_n_0;
  wire ap_rst_n;
  wire ap_rst_n_inv;
  wire [0:0]col_fu_70;
  wire \col_fu_70_reg[0] ;
  wire \col_fu_70_reg[0]_0 ;
  wire grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg;
  wire grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg_reg;
  wire [1:0]icmp_ln34_fu_153_p2_carry__4;
  wire [1:0]icmp_ln34_fu_153_p2_carry__4_0;
  wire \indvar_flatten_fu_78_reg[61] ;
  wire output_stream_TREADY_int_regslice;

  LUT5 #(
    .INIT(32'hAAAAEEAE)) 
    \ap_CS_fsm[5]_i_1 
       (.I0(\ap_CS_fsm_reg[6] [0]),
        .I1(\ap_CS_fsm_reg[6] [1]),
        .I2(ap_done_cache),
        .I3(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg),
        .I4(\ap_CS_fsm_reg[5] ),
        .O(D[0]));
  LUT6 #(
    .INIT(64'hAA08FFFFAA08AA08)) 
    \ap_CS_fsm[6]_i_1 
       (.I0(\ap_CS_fsm_reg[6] [1]),
        .I1(ap_done_cache),
        .I2(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg),
        .I3(\ap_CS_fsm_reg[5] ),
        .I4(\ap_CS_fsm_reg[6]_0 ),
        .I5(\ap_CS_fsm_reg[6] [2]),
        .O(D[1]));
  LUT5 #(
    .INIT(32'h40FF4040)) 
    ap_done_cache_i_1
       (.I0(ap_block_pp0_stage0_11001__1),
        .I1(ap_done_cache_reg_0),
        .I2(ap_enable_reg_pp0_iter1),
        .I3(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg),
        .I4(ap_done_cache),
        .O(ap_done_cache_i_1_n_0));
  FDRE #(
    .INIT(1'b0)) 
    ap_done_cache_reg
       (.C(ap_clk),
        .CE(1'b1),
        .D(ap_done_cache_i_1_n_0),
        .Q(ap_done_cache),
        .R(ap_rst_n_inv));
  LUT6 #(
    .INIT(64'hCFCFCFCFFF4F4F4F)) 
    ap_loop_init_int_i_1
       (.I0(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg),
        .I1(ap_loop_init_int),
        .I2(ap_rst_n),
        .I3(ap_enable_reg_pp0_iter1),
        .I4(ap_done_cache_reg_0),
        .I5(ap_block_pp0_stage0_11001__1),
        .O(ap_loop_init_int_i_1_n_0));
  FDRE #(
    .INIT(1'b1)) 
    ap_loop_init_int_reg
       (.C(ap_clk),
        .CE(1'b1),
        .D(ap_loop_init_int_i_1_n_0),
        .Q(ap_loop_init_int),
        .R(1'b0));
  LUT4 #(
    .INIT(16'h006E)) 
    \col_fu_70[0]_i_1 
       (.I0(col_fu_70),
        .I1(\col_fu_70_reg[0]_0 ),
        .I2(CO),
        .I3(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg_reg),
        .O(\col_fu_70_reg[0] ));
  LUT6 #(
    .INIT(64'h00000000FF040404)) 
    \col_fu_70[30]_i_1 
       (.I0(ap_done_cache_reg_0),
        .I1(ap_enable_reg_pp0_iter1),
        .I2(CO),
        .I3(ap_loop_init_int),
        .I4(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg),
        .I5(ap_block_pp0_stage0_11001__1),
        .O(ap_enable_reg_pp0_iter1_reg));
  LUT4 #(
    .INIT(16'h2AAA)) 
    \col_fu_70[30]_i_4 
       (.I0(\indvar_flatten_fu_78_reg[61] ),
        .I1(Q),
        .I2(output_stream_TREADY_int_regslice),
        .I3(\ap_CS_fsm_reg[6] [1]),
        .O(ap_block_pp0_stage0_11001__1));
  LUT4 #(
    .INIT(16'h9009)) 
    icmp_ln34_fu_153_p2_carry__4_i_1
       (.I0(icmp_ln34_fu_153_p2_carry__4[0]),
        .I1(icmp_ln34_fu_153_p2_carry__4_0[0]),
        .I2(icmp_ln34_fu_153_p2_carry__4[1]),
        .I3(icmp_ln34_fu_153_p2_carry__4_0[1]),
        .O(S));
  LUT6 #(
    .INIT(64'h8000000088888888)) 
    \indvar_flatten_fu_78[0]_i_1 
       (.I0(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg),
        .I1(ap_loop_init_int),
        .I2(\ap_CS_fsm_reg[6] [1]),
        .I3(output_stream_TREADY_int_regslice),
        .I4(Q),
        .I5(\indvar_flatten_fu_78_reg[61] ),
        .O(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_ap_start_reg_reg));
endmodule

module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_mul_31ns_31ns_62_2_1
   (buff0_reg__0_0,
    Q,
    ap_clk,
    buff0_reg_0,
    buff0_reg__0_1,
    tmp_product_i_4_0,
    tmp_product_i_3_0,
    E,
    D,
    tmp_product__19_0);
  output [61:0]buff0_reg__0_0;
  input [0:0]Q;
  input ap_clk;
  input [13:0]buff0_reg_0;
  input [13:0]buff0_reg__0_1;
  input [31:0]tmp_product_i_4_0;
  input [31:0]tmp_product_i_3_0;
  input [0:0]E;
  input [16:0]D;
  input [16:0]tmp_product__19_0;

  wire [16:0]D;
  wire [0:0]E;
  wire [0:0]Q;
  wire ap_clk;
  wire \buff0_reg[16]__0_n_0 ;
  wire [13:0]buff0_reg_0;
  wire [61:0]buff0_reg__0_0;
  wire [13:0]buff0_reg__0_1;
  wire buff0_reg__0_n_100;
  wire buff0_reg__0_n_101;
  wire buff0_reg__0_n_102;
  wire buff0_reg__0_n_103;
  wire buff0_reg__0_n_104;
  wire buff0_reg__0_n_105;
  wire buff0_reg__0_n_58;
  wire buff0_reg__0_n_59;
  wire buff0_reg__0_n_60;
  wire buff0_reg__0_n_61;
  wire buff0_reg__0_n_62;
  wire buff0_reg__0_n_63;
  wire buff0_reg__0_n_64;
  wire buff0_reg__0_n_65;
  wire buff0_reg__0_n_66;
  wire buff0_reg__0_n_67;
  wire buff0_reg__0_n_68;
  wire buff0_reg__0_n_69;
  wire buff0_reg__0_n_70;
  wire buff0_reg__0_n_71;
  wire buff0_reg__0_n_72;
  wire buff0_reg__0_n_73;
  wire buff0_reg__0_n_74;
  wire buff0_reg__0_n_75;
  wire buff0_reg__0_n_76;
  wire buff0_reg__0_n_77;
  wire buff0_reg__0_n_78;
  wire buff0_reg__0_n_79;
  wire buff0_reg__0_n_80;
  wire buff0_reg__0_n_81;
  wire buff0_reg__0_n_82;
  wire buff0_reg__0_n_83;
  wire buff0_reg__0_n_84;
  wire buff0_reg__0_n_85;
  wire buff0_reg__0_n_86;
  wire buff0_reg__0_n_87;
  wire buff0_reg__0_n_88;
  wire buff0_reg__0_n_89;
  wire buff0_reg__0_n_90;
  wire buff0_reg__0_n_91;
  wire buff0_reg__0_n_92;
  wire buff0_reg__0_n_93;
  wire buff0_reg__0_n_94;
  wire buff0_reg__0_n_95;
  wire buff0_reg__0_n_96;
  wire buff0_reg__0_n_97;
  wire buff0_reg__0_n_98;
  wire buff0_reg__0_n_99;
  wire \buff0_reg_n_0_[0] ;
  wire \buff0_reg_n_0_[10] ;
  wire \buff0_reg_n_0_[11] ;
  wire \buff0_reg_n_0_[12] ;
  wire \buff0_reg_n_0_[13] ;
  wire \buff0_reg_n_0_[14] ;
  wire \buff0_reg_n_0_[15] ;
  wire \buff0_reg_n_0_[16] ;
  wire \buff0_reg_n_0_[1] ;
  wire \buff0_reg_n_0_[2] ;
  wire \buff0_reg_n_0_[3] ;
  wire \buff0_reg_n_0_[4] ;
  wire \buff0_reg_n_0_[5] ;
  wire \buff0_reg_n_0_[6] ;
  wire \buff0_reg_n_0_[7] ;
  wire \buff0_reg_n_0_[8] ;
  wire \buff0_reg_n_0_[9] ;
  wire buff0_reg_n_100;
  wire buff0_reg_n_101;
  wire buff0_reg_n_102;
  wire buff0_reg_n_103;
  wire buff0_reg_n_104;
  wire buff0_reg_n_105;
  wire buff0_reg_n_58;
  wire buff0_reg_n_59;
  wire buff0_reg_n_60;
  wire buff0_reg_n_61;
  wire buff0_reg_n_62;
  wire buff0_reg_n_63;
  wire buff0_reg_n_64;
  wire buff0_reg_n_65;
  wire buff0_reg_n_66;
  wire buff0_reg_n_67;
  wire buff0_reg_n_68;
  wire buff0_reg_n_69;
  wire buff0_reg_n_70;
  wire buff0_reg_n_71;
  wire buff0_reg_n_72;
  wire buff0_reg_n_73;
  wire buff0_reg_n_74;
  wire buff0_reg_n_75;
  wire buff0_reg_n_76;
  wire buff0_reg_n_77;
  wire buff0_reg_n_78;
  wire buff0_reg_n_79;
  wire buff0_reg_n_80;
  wire buff0_reg_n_81;
  wire buff0_reg_n_82;
  wire buff0_reg_n_83;
  wire buff0_reg_n_84;
  wire buff0_reg_n_85;
  wire buff0_reg_n_86;
  wire buff0_reg_n_87;
  wire buff0_reg_n_88;
  wire buff0_reg_n_89;
  wire buff0_reg_n_90;
  wire buff0_reg_n_91;
  wire buff0_reg_n_92;
  wire buff0_reg_n_93;
  wire buff0_reg_n_94;
  wire buff0_reg_n_95;
  wire buff0_reg_n_96;
  wire buff0_reg_n_97;
  wire buff0_reg_n_98;
  wire buff0_reg_n_99;
  wire \mul_ln10_reg_199[19]_i_2_n_0 ;
  wire \mul_ln10_reg_199[19]_i_3_n_0 ;
  wire \mul_ln10_reg_199[19]_i_4_n_0 ;
  wire \mul_ln10_reg_199[23]_i_2_n_0 ;
  wire \mul_ln10_reg_199[23]_i_3_n_0 ;
  wire \mul_ln10_reg_199[23]_i_4_n_0 ;
  wire \mul_ln10_reg_199[23]_i_5_n_0 ;
  wire \mul_ln10_reg_199[27]_i_2_n_0 ;
  wire \mul_ln10_reg_199[27]_i_3_n_0 ;
  wire \mul_ln10_reg_199[27]_i_4_n_0 ;
  wire \mul_ln10_reg_199[27]_i_5_n_0 ;
  wire \mul_ln10_reg_199[31]_i_2_n_0 ;
  wire \mul_ln10_reg_199[31]_i_3_n_0 ;
  wire \mul_ln10_reg_199[31]_i_4_n_0 ;
  wire \mul_ln10_reg_199[31]_i_5_n_0 ;
  wire \mul_ln10_reg_199[35]_i_2_n_0 ;
  wire \mul_ln10_reg_199[35]_i_3_n_0 ;
  wire \mul_ln10_reg_199[35]_i_4_n_0 ;
  wire \mul_ln10_reg_199[35]_i_5_n_0 ;
  wire \mul_ln10_reg_199[39]_i_2_n_0 ;
  wire \mul_ln10_reg_199[39]_i_3_n_0 ;
  wire \mul_ln10_reg_199[39]_i_4_n_0 ;
  wire \mul_ln10_reg_199[39]_i_5_n_0 ;
  wire \mul_ln10_reg_199[43]_i_2_n_0 ;
  wire \mul_ln10_reg_199[43]_i_3_n_0 ;
  wire \mul_ln10_reg_199[43]_i_4_n_0 ;
  wire \mul_ln10_reg_199[43]_i_5_n_0 ;
  wire \mul_ln10_reg_199[47]_i_2_n_0 ;
  wire \mul_ln10_reg_199[47]_i_3_n_0 ;
  wire \mul_ln10_reg_199[47]_i_4_n_0 ;
  wire \mul_ln10_reg_199[47]_i_5_n_0 ;
  wire \mul_ln10_reg_199[51]_i_2_n_0 ;
  wire \mul_ln10_reg_199[51]_i_3_n_0 ;
  wire \mul_ln10_reg_199[51]_i_4_n_0 ;
  wire \mul_ln10_reg_199[51]_i_5_n_0 ;
  wire \mul_ln10_reg_199[55]_i_2_n_0 ;
  wire \mul_ln10_reg_199[55]_i_3_n_0 ;
  wire \mul_ln10_reg_199[55]_i_4_n_0 ;
  wire \mul_ln10_reg_199[55]_i_5_n_0 ;
  wire \mul_ln10_reg_199[59]_i_2_n_0 ;
  wire \mul_ln10_reg_199[59]_i_3_n_0 ;
  wire \mul_ln10_reg_199[59]_i_4_n_0 ;
  wire \mul_ln10_reg_199[59]_i_5_n_0 ;
  wire \mul_ln10_reg_199[61]_i_2_n_0 ;
  wire \mul_ln10_reg_199[61]_i_3_n_0 ;
  wire \mul_ln10_reg_199_reg[19]_i_1_n_0 ;
  wire \mul_ln10_reg_199_reg[19]_i_1_n_1 ;
  wire \mul_ln10_reg_199_reg[19]_i_1_n_2 ;
  wire \mul_ln10_reg_199_reg[19]_i_1_n_3 ;
  wire \mul_ln10_reg_199_reg[23]_i_1_n_0 ;
  wire \mul_ln10_reg_199_reg[23]_i_1_n_1 ;
  wire \mul_ln10_reg_199_reg[23]_i_1_n_2 ;
  wire \mul_ln10_reg_199_reg[23]_i_1_n_3 ;
  wire \mul_ln10_reg_199_reg[27]_i_1_n_0 ;
  wire \mul_ln10_reg_199_reg[27]_i_1_n_1 ;
  wire \mul_ln10_reg_199_reg[27]_i_1_n_2 ;
  wire \mul_ln10_reg_199_reg[27]_i_1_n_3 ;
  wire \mul_ln10_reg_199_reg[31]_i_1_n_0 ;
  wire \mul_ln10_reg_199_reg[31]_i_1_n_1 ;
  wire \mul_ln10_reg_199_reg[31]_i_1_n_2 ;
  wire \mul_ln10_reg_199_reg[31]_i_1_n_3 ;
  wire \mul_ln10_reg_199_reg[35]_i_1_n_0 ;
  wire \mul_ln10_reg_199_reg[35]_i_1_n_1 ;
  wire \mul_ln10_reg_199_reg[35]_i_1_n_2 ;
  wire \mul_ln10_reg_199_reg[35]_i_1_n_3 ;
  wire \mul_ln10_reg_199_reg[39]_i_1_n_0 ;
  wire \mul_ln10_reg_199_reg[39]_i_1_n_1 ;
  wire \mul_ln10_reg_199_reg[39]_i_1_n_2 ;
  wire \mul_ln10_reg_199_reg[39]_i_1_n_3 ;
  wire \mul_ln10_reg_199_reg[43]_i_1_n_0 ;
  wire \mul_ln10_reg_199_reg[43]_i_1_n_1 ;
  wire \mul_ln10_reg_199_reg[43]_i_1_n_2 ;
  wire \mul_ln10_reg_199_reg[43]_i_1_n_3 ;
  wire \mul_ln10_reg_199_reg[47]_i_1_n_0 ;
  wire \mul_ln10_reg_199_reg[47]_i_1_n_1 ;
  wire \mul_ln10_reg_199_reg[47]_i_1_n_2 ;
  wire \mul_ln10_reg_199_reg[47]_i_1_n_3 ;
  wire \mul_ln10_reg_199_reg[51]_i_1_n_0 ;
  wire \mul_ln10_reg_199_reg[51]_i_1_n_1 ;
  wire \mul_ln10_reg_199_reg[51]_i_1_n_2 ;
  wire \mul_ln10_reg_199_reg[51]_i_1_n_3 ;
  wire \mul_ln10_reg_199_reg[55]_i_1_n_0 ;
  wire \mul_ln10_reg_199_reg[55]_i_1_n_1 ;
  wire \mul_ln10_reg_199_reg[55]_i_1_n_2 ;
  wire \mul_ln10_reg_199_reg[55]_i_1_n_3 ;
  wire \mul_ln10_reg_199_reg[59]_i_1_n_0 ;
  wire \mul_ln10_reg_199_reg[59]_i_1_n_1 ;
  wire \mul_ln10_reg_199_reg[59]_i_1_n_2 ;
  wire \mul_ln10_reg_199_reg[59]_i_1_n_3 ;
  wire \mul_ln10_reg_199_reg[61]_i_1_n_3 ;
  wire p_0_in;
  wire smax1_reg_184;
  wire smax_reg_179;
  wire tmp_product__0_n_0;
  wire tmp_product__10_n_0;
  wire tmp_product__11_n_0;
  wire tmp_product__12_n_0;
  wire tmp_product__13_n_0;
  wire tmp_product__14_n_0;
  wire tmp_product__15_n_0;
  wire tmp_product__16_n_0;
  wire tmp_product__17_n_100;
  wire tmp_product__17_n_101;
  wire tmp_product__17_n_102;
  wire tmp_product__17_n_103;
  wire tmp_product__17_n_104;
  wire tmp_product__17_n_105;
  wire tmp_product__17_n_106;
  wire tmp_product__17_n_107;
  wire tmp_product__17_n_108;
  wire tmp_product__17_n_109;
  wire tmp_product__17_n_110;
  wire tmp_product__17_n_111;
  wire tmp_product__17_n_112;
  wire tmp_product__17_n_113;
  wire tmp_product__17_n_114;
  wire tmp_product__17_n_115;
  wire tmp_product__17_n_116;
  wire tmp_product__17_n_117;
  wire tmp_product__17_n_118;
  wire tmp_product__17_n_119;
  wire tmp_product__17_n_120;
  wire tmp_product__17_n_121;
  wire tmp_product__17_n_122;
  wire tmp_product__17_n_123;
  wire tmp_product__17_n_124;
  wire tmp_product__17_n_125;
  wire tmp_product__17_n_126;
  wire tmp_product__17_n_127;
  wire tmp_product__17_n_128;
  wire tmp_product__17_n_129;
  wire tmp_product__17_n_130;
  wire tmp_product__17_n_131;
  wire tmp_product__17_n_132;
  wire tmp_product__17_n_133;
  wire tmp_product__17_n_134;
  wire tmp_product__17_n_135;
  wire tmp_product__17_n_136;
  wire tmp_product__17_n_137;
  wire tmp_product__17_n_138;
  wire tmp_product__17_n_139;
  wire tmp_product__17_n_140;
  wire tmp_product__17_n_141;
  wire tmp_product__17_n_142;
  wire tmp_product__17_n_143;
  wire tmp_product__17_n_144;
  wire tmp_product__17_n_145;
  wire tmp_product__17_n_146;
  wire tmp_product__17_n_147;
  wire tmp_product__17_n_148;
  wire tmp_product__17_n_149;
  wire tmp_product__17_n_150;
  wire tmp_product__17_n_151;
  wire tmp_product__17_n_152;
  wire tmp_product__17_n_153;
  wire tmp_product__17_n_24;
  wire tmp_product__17_n_25;
  wire tmp_product__17_n_26;
  wire tmp_product__17_n_27;
  wire tmp_product__17_n_28;
  wire tmp_product__17_n_29;
  wire tmp_product__17_n_30;
  wire tmp_product__17_n_31;
  wire tmp_product__17_n_32;
  wire tmp_product__17_n_33;
  wire tmp_product__17_n_34;
  wire tmp_product__17_n_35;
  wire tmp_product__17_n_36;
  wire tmp_product__17_n_37;
  wire tmp_product__17_n_38;
  wire tmp_product__17_n_39;
  wire tmp_product__17_n_40;
  wire tmp_product__17_n_41;
  wire tmp_product__17_n_42;
  wire tmp_product__17_n_43;
  wire tmp_product__17_n_44;
  wire tmp_product__17_n_45;
  wire tmp_product__17_n_46;
  wire tmp_product__17_n_47;
  wire tmp_product__17_n_48;
  wire tmp_product__17_n_49;
  wire tmp_product__17_n_50;
  wire tmp_product__17_n_51;
  wire tmp_product__17_n_52;
  wire tmp_product__17_n_53;
  wire tmp_product__17_n_58;
  wire tmp_product__17_n_59;
  wire tmp_product__17_n_60;
  wire tmp_product__17_n_61;
  wire tmp_product__17_n_62;
  wire tmp_product__17_n_63;
  wire tmp_product__17_n_64;
  wire tmp_product__17_n_65;
  wire tmp_product__17_n_66;
  wire tmp_product__17_n_67;
  wire tmp_product__17_n_68;
  wire tmp_product__17_n_69;
  wire tmp_product__17_n_70;
  wire tmp_product__17_n_71;
  wire tmp_product__17_n_72;
  wire tmp_product__17_n_73;
  wire tmp_product__17_n_74;
  wire tmp_product__17_n_75;
  wire tmp_product__17_n_76;
  wire tmp_product__17_n_77;
  wire tmp_product__17_n_78;
  wire tmp_product__17_n_79;
  wire tmp_product__17_n_80;
  wire tmp_product__17_n_81;
  wire tmp_product__17_n_82;
  wire tmp_product__17_n_83;
  wire tmp_product__17_n_84;
  wire tmp_product__17_n_85;
  wire tmp_product__17_n_86;
  wire tmp_product__17_n_87;
  wire tmp_product__17_n_88;
  wire tmp_product__17_n_89;
  wire tmp_product__17_n_90;
  wire tmp_product__17_n_91;
  wire tmp_product__17_n_92;
  wire tmp_product__17_n_93;
  wire tmp_product__17_n_94;
  wire tmp_product__17_n_95;
  wire tmp_product__17_n_96;
  wire tmp_product__17_n_97;
  wire tmp_product__17_n_98;
  wire tmp_product__17_n_99;
  wire [16:0]tmp_product__19_0;
  wire tmp_product__19_n_0;
  wire tmp_product__1_n_0;
  wire tmp_product__20_n_0;
  wire tmp_product__21_n_0;
  wire tmp_product__22_n_0;
  wire tmp_product__23_n_0;
  wire tmp_product__24_n_0;
  wire tmp_product__25_n_0;
  wire tmp_product__26_n_0;
  wire tmp_product__27_n_0;
  wire tmp_product__28_n_0;
  wire tmp_product__29_n_0;
  wire tmp_product__2_n_0;
  wire tmp_product__30_n_0;
  wire tmp_product__31_n_0;
  wire tmp_product__32_n_0;
  wire tmp_product__33_n_0;
  wire tmp_product__34_n_0;
  wire tmp_product__35_n_0;
  wire tmp_product__3_n_0;
  wire tmp_product__4_n_0;
  wire tmp_product__5_n_0;
  wire tmp_product__6_n_0;
  wire tmp_product__7_n_0;
  wire tmp_product__8_n_0;
  wire tmp_product__9_n_0;
  wire tmp_product_i_10_n_0;
  wire tmp_product_i_11_n_0;
  wire tmp_product_i_12_n_0;
  wire tmp_product_i_13_n_0;
  wire tmp_product_i_14_n_0;
  wire tmp_product_i_14_n_1;
  wire tmp_product_i_14_n_2;
  wire tmp_product_i_14_n_3;
  wire tmp_product_i_15_n_0;
  wire tmp_product_i_16_n_0;
  wire tmp_product_i_17_n_0;
  wire tmp_product_i_18_n_0;
  wire tmp_product_i_19_n_0;
  wire tmp_product_i_20_n_0;
  wire tmp_product_i_21_n_0;
  wire tmp_product_i_22_n_0;
  wire tmp_product_i_23_n_0;
  wire tmp_product_i_23_n_1;
  wire tmp_product_i_23_n_2;
  wire tmp_product_i_23_n_3;
  wire tmp_product_i_24_n_0;
  wire tmp_product_i_25_n_0;
  wire tmp_product_i_26_n_0;
  wire tmp_product_i_27_n_0;
  wire tmp_product_i_28_n_0;
  wire tmp_product_i_29_n_0;
  wire tmp_product_i_30_n_0;
  wire tmp_product_i_31_n_0;
  wire tmp_product_i_32_n_0;
  wire tmp_product_i_32_n_1;
  wire tmp_product_i_32_n_2;
  wire tmp_product_i_32_n_3;
  wire tmp_product_i_33_n_0;
  wire tmp_product_i_34_n_0;
  wire tmp_product_i_35_n_0;
  wire tmp_product_i_36_n_0;
  wire tmp_product_i_37_n_0;
  wire tmp_product_i_38_n_0;
  wire tmp_product_i_39_n_0;
  wire [31:0]tmp_product_i_3_0;
  wire tmp_product_i_3_n_1;
  wire tmp_product_i_3_n_2;
  wire tmp_product_i_3_n_3;
  wire tmp_product_i_40_n_0;
  wire tmp_product_i_41_n_0;
  wire tmp_product_i_41_n_1;
  wire tmp_product_i_41_n_2;
  wire tmp_product_i_41_n_3;
  wire tmp_product_i_42_n_0;
  wire tmp_product_i_43_n_0;
  wire tmp_product_i_44_n_0;
  wire tmp_product_i_45_n_0;
  wire tmp_product_i_46_n_0;
  wire tmp_product_i_47_n_0;
  wire tmp_product_i_48_n_0;
  wire tmp_product_i_49_n_0;
  wire [31:0]tmp_product_i_4_0;
  wire tmp_product_i_4_n_0;
  wire tmp_product_i_4_n_1;
  wire tmp_product_i_4_n_2;
  wire tmp_product_i_4_n_3;
  wire tmp_product_i_50_n_0;
  wire tmp_product_i_50_n_1;
  wire tmp_product_i_50_n_2;
  wire tmp_product_i_50_n_3;
  wire tmp_product_i_51_n_0;
  wire tmp_product_i_52_n_0;
  wire tmp_product_i_53_n_0;
  wire tmp_product_i_54_n_0;
  wire tmp_product_i_55_n_0;
  wire tmp_product_i_56_n_0;
  wire tmp_product_i_57_n_0;
  wire tmp_product_i_58_n_0;
  wire tmp_product_i_59_n_0;
  wire tmp_product_i_5_n_0;
  wire tmp_product_i_5_n_1;
  wire tmp_product_i_5_n_2;
  wire tmp_product_i_5_n_3;
  wire tmp_product_i_60_n_0;
  wire tmp_product_i_61_n_0;
  wire tmp_product_i_62_n_0;
  wire tmp_product_i_63_n_0;
  wire tmp_product_i_64_n_0;
  wire tmp_product_i_65_n_0;
  wire tmp_product_i_66_n_0;
  wire tmp_product_i_67_n_0;
  wire tmp_product_i_68_n_0;
  wire tmp_product_i_69_n_0;
  wire tmp_product_i_6_n_0;
  wire tmp_product_i_70_n_0;
  wire tmp_product_i_71_n_0;
  wire tmp_product_i_72_n_0;
  wire tmp_product_i_73_n_0;
  wire tmp_product_i_74_n_0;
  wire tmp_product_i_7_n_0;
  wire tmp_product_i_8_n_0;
  wire tmp_product_i_9_n_0;
  wire tmp_product_n_100;
  wire tmp_product_n_101;
  wire tmp_product_n_102;
  wire tmp_product_n_103;
  wire tmp_product_n_104;
  wire tmp_product_n_105;
  wire tmp_product_n_106;
  wire tmp_product_n_107;
  wire tmp_product_n_108;
  wire tmp_product_n_109;
  wire tmp_product_n_110;
  wire tmp_product_n_111;
  wire tmp_product_n_112;
  wire tmp_product_n_113;
  wire tmp_product_n_114;
  wire tmp_product_n_115;
  wire tmp_product_n_116;
  wire tmp_product_n_117;
  wire tmp_product_n_118;
  wire tmp_product_n_119;
  wire tmp_product_n_120;
  wire tmp_product_n_121;
  wire tmp_product_n_122;
  wire tmp_product_n_123;
  wire tmp_product_n_124;
  wire tmp_product_n_125;
  wire tmp_product_n_126;
  wire tmp_product_n_127;
  wire tmp_product_n_128;
  wire tmp_product_n_129;
  wire tmp_product_n_130;
  wire tmp_product_n_131;
  wire tmp_product_n_132;
  wire tmp_product_n_133;
  wire tmp_product_n_134;
  wire tmp_product_n_135;
  wire tmp_product_n_136;
  wire tmp_product_n_137;
  wire tmp_product_n_138;
  wire tmp_product_n_139;
  wire tmp_product_n_140;
  wire tmp_product_n_141;
  wire tmp_product_n_142;
  wire tmp_product_n_143;
  wire tmp_product_n_144;
  wire tmp_product_n_145;
  wire tmp_product_n_146;
  wire tmp_product_n_147;
  wire tmp_product_n_148;
  wire tmp_product_n_149;
  wire tmp_product_n_150;
  wire tmp_product_n_151;
  wire tmp_product_n_152;
  wire tmp_product_n_153;
  wire tmp_product_n_58;
  wire tmp_product_n_59;
  wire tmp_product_n_60;
  wire tmp_product_n_61;
  wire tmp_product_n_62;
  wire tmp_product_n_63;
  wire tmp_product_n_64;
  wire tmp_product_n_65;
  wire tmp_product_n_66;
  wire tmp_product_n_67;
  wire tmp_product_n_68;
  wire tmp_product_n_69;
  wire tmp_product_n_70;
  wire tmp_product_n_71;
  wire tmp_product_n_72;
  wire tmp_product_n_73;
  wire tmp_product_n_74;
  wire tmp_product_n_75;
  wire tmp_product_n_76;
  wire tmp_product_n_77;
  wire tmp_product_n_78;
  wire tmp_product_n_79;
  wire tmp_product_n_80;
  wire tmp_product_n_81;
  wire tmp_product_n_82;
  wire tmp_product_n_83;
  wire tmp_product_n_84;
  wire tmp_product_n_85;
  wire tmp_product_n_86;
  wire tmp_product_n_87;
  wire tmp_product_n_88;
  wire tmp_product_n_89;
  wire tmp_product_n_90;
  wire tmp_product_n_91;
  wire tmp_product_n_92;
  wire tmp_product_n_93;
  wire tmp_product_n_94;
  wire tmp_product_n_95;
  wire tmp_product_n_96;
  wire tmp_product_n_97;
  wire tmp_product_n_98;
  wire tmp_product_n_99;
  wire NLW_buff0_reg_CARRYCASCOUT_UNCONNECTED;
  wire NLW_buff0_reg_MULTSIGNOUT_UNCONNECTED;
  wire NLW_buff0_reg_OVERFLOW_UNCONNECTED;
  wire NLW_buff0_reg_PATTERNBDETECT_UNCONNECTED;
  wire NLW_buff0_reg_PATTERNDETECT_UNCONNECTED;
  wire NLW_buff0_reg_UNDERFLOW_UNCONNECTED;
  wire [29:0]NLW_buff0_reg_ACOUT_UNCONNECTED;
  wire [17:0]NLW_buff0_reg_BCOUT_UNCONNECTED;
  wire [3:0]NLW_buff0_reg_CARRYOUT_UNCONNECTED;
  wire [47:0]NLW_buff0_reg_PCOUT_UNCONNECTED;
  wire NLW_buff0_reg__0_CARRYCASCOUT_UNCONNECTED;
  wire NLW_buff0_reg__0_MULTSIGNOUT_UNCONNECTED;
  wire NLW_buff0_reg__0_OVERFLOW_UNCONNECTED;
  wire NLW_buff0_reg__0_PATTERNBDETECT_UNCONNECTED;
  wire NLW_buff0_reg__0_PATTERNDETECT_UNCONNECTED;
  wire NLW_buff0_reg__0_UNDERFLOW_UNCONNECTED;
  wire [29:0]NLW_buff0_reg__0_ACOUT_UNCONNECTED;
  wire [17:0]NLW_buff0_reg__0_BCOUT_UNCONNECTED;
  wire [3:0]NLW_buff0_reg__0_CARRYOUT_UNCONNECTED;
  wire [47:0]NLW_buff0_reg__0_PCOUT_UNCONNECTED;
  wire [3:1]\NLW_mul_ln10_reg_199_reg[61]_i_1_CO_UNCONNECTED ;
  wire [3:2]\NLW_mul_ln10_reg_199_reg[61]_i_1_O_UNCONNECTED ;
  wire NLW_tmp_product_CARRYCASCOUT_UNCONNECTED;
  wire NLW_tmp_product_MULTSIGNOUT_UNCONNECTED;
  wire NLW_tmp_product_OVERFLOW_UNCONNECTED;
  wire NLW_tmp_product_PATTERNBDETECT_UNCONNECTED;
  wire NLW_tmp_product_PATTERNDETECT_UNCONNECTED;
  wire NLW_tmp_product_UNDERFLOW_UNCONNECTED;
  wire [29:0]NLW_tmp_product_ACOUT_UNCONNECTED;
  wire [17:0]NLW_tmp_product_BCOUT_UNCONNECTED;
  wire [3:0]NLW_tmp_product_CARRYOUT_UNCONNECTED;
  wire NLW_tmp_product__17_CARRYCASCOUT_UNCONNECTED;
  wire NLW_tmp_product__17_MULTSIGNOUT_UNCONNECTED;
  wire NLW_tmp_product__17_OVERFLOW_UNCONNECTED;
  wire NLW_tmp_product__17_PATTERNBDETECT_UNCONNECTED;
  wire NLW_tmp_product__17_PATTERNDETECT_UNCONNECTED;
  wire NLW_tmp_product__17_UNDERFLOW_UNCONNECTED;
  wire [17:0]NLW_tmp_product__17_BCOUT_UNCONNECTED;
  wire [3:0]NLW_tmp_product__17_CARRYOUT_UNCONNECTED;
  wire [3:0]NLW_tmp_product_i_14_O_UNCONNECTED;
  wire [3:0]NLW_tmp_product_i_23_O_UNCONNECTED;
  wire [3:0]NLW_tmp_product_i_3_O_UNCONNECTED;
  wire [3:0]NLW_tmp_product_i_32_O_UNCONNECTED;
  wire [3:0]NLW_tmp_product_i_4_O_UNCONNECTED;
  wire [3:0]NLW_tmp_product_i_41_O_UNCONNECTED;
  wire [3:0]NLW_tmp_product_i_5_O_UNCONNECTED;
  wire [3:0]NLW_tmp_product_i_50_O_UNCONNECTED;

  (* METHODOLOGY_DRC_VIOS = "{SYNTH-10 {cell *THIS*} {string 15x15 4}}" *) 
  DSP48E1 #(
    .ACASCREG(1),
    .ADREG(1),
    .ALUMODEREG(0),
    .AREG(1),
    .AUTORESET_PATDET("NO_RESET"),
    .A_INPUT("DIRECT"),
    .BCASCREG(1),
    .BREG(1),
    .B_INPUT("DIRECT"),
    .CARRYINREG(0),
    .CARRYINSELREG(0),
    .CREG(1),
    .DREG(1),
    .INMODEREG(0),
    .MASK(48'h3FFFFFFFFFFF),
    .MREG(0),
    .OPMODEREG(0),
    .PATTERN(48'h000000000000),
    .PREG(1),
    .SEL_MASK("MASK"),
    .SEL_PATTERN("PATTERN"),
    .USE_DPORT("FALSE"),
    .USE_MULT("MULTIPLY"),
    .USE_PATTERN_DETECT("NO_PATDET"),
    .USE_SIMD("ONE48")) 
    buff0_reg
       (.A({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,buff0_reg_0}),
        .ACIN({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .ACOUT(NLW_buff0_reg_ACOUT_UNCONNECTED[29:0]),
        .ALUMODE({1'b0,1'b0,1'b0,1'b0}),
        .B({1'b0,1'b0,1'b0,1'b0,buff0_reg__0_1}),
        .BCIN({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .BCOUT(NLW_buff0_reg_BCOUT_UNCONNECTED[17:0]),
        .C({1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1}),
        .CARRYCASCIN(1'b0),
        .CARRYCASCOUT(NLW_buff0_reg_CARRYCASCOUT_UNCONNECTED),
        .CARRYIN(1'b0),
        .CARRYINSEL({1'b0,1'b0,1'b0}),
        .CARRYOUT(NLW_buff0_reg_CARRYOUT_UNCONNECTED[3:0]),
        .CEA1(1'b0),
        .CEA2(Q),
        .CEAD(1'b0),
        .CEALUMODE(1'b0),
        .CEB1(1'b0),
        .CEB2(Q),
        .CEC(1'b0),
        .CECARRYIN(1'b0),
        .CECTRL(1'b0),
        .CED(1'b0),
        .CEINMODE(1'b0),
        .CEM(1'b0),
        .CEP(1'b1),
        .CLK(ap_clk),
        .D({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .INMODE({1'b0,1'b0,1'b0,1'b0,1'b0}),
        .MULTSIGNIN(1'b0),
        .MULTSIGNOUT(NLW_buff0_reg_MULTSIGNOUT_UNCONNECTED),
        .OPMODE({1'b1,1'b0,1'b1,1'b0,1'b1,1'b0,1'b1}),
        .OVERFLOW(NLW_buff0_reg_OVERFLOW_UNCONNECTED),
        .P({buff0_reg_n_58,buff0_reg_n_59,buff0_reg_n_60,buff0_reg_n_61,buff0_reg_n_62,buff0_reg_n_63,buff0_reg_n_64,buff0_reg_n_65,buff0_reg_n_66,buff0_reg_n_67,buff0_reg_n_68,buff0_reg_n_69,buff0_reg_n_70,buff0_reg_n_71,buff0_reg_n_72,buff0_reg_n_73,buff0_reg_n_74,buff0_reg_n_75,buff0_reg_n_76,buff0_reg_n_77,buff0_reg_n_78,buff0_reg_n_79,buff0_reg_n_80,buff0_reg_n_81,buff0_reg_n_82,buff0_reg_n_83,buff0_reg_n_84,buff0_reg_n_85,buff0_reg_n_86,buff0_reg_n_87,buff0_reg_n_88,buff0_reg_n_89,buff0_reg_n_90,buff0_reg_n_91,buff0_reg_n_92,buff0_reg_n_93,buff0_reg_n_94,buff0_reg_n_95,buff0_reg_n_96,buff0_reg_n_97,buff0_reg_n_98,buff0_reg_n_99,buff0_reg_n_100,buff0_reg_n_101,buff0_reg_n_102,buff0_reg_n_103,buff0_reg_n_104,buff0_reg_n_105}),
        .PATTERNBDETECT(NLW_buff0_reg_PATTERNBDETECT_UNCONNECTED),
        .PATTERNDETECT(NLW_buff0_reg_PATTERNDETECT_UNCONNECTED),
        .PCIN({tmp_product_n_106,tmp_product_n_107,tmp_product_n_108,tmp_product_n_109,tmp_product_n_110,tmp_product_n_111,tmp_product_n_112,tmp_product_n_113,tmp_product_n_114,tmp_product_n_115,tmp_product_n_116,tmp_product_n_117,tmp_product_n_118,tmp_product_n_119,tmp_product_n_120,tmp_product_n_121,tmp_product_n_122,tmp_product_n_123,tmp_product_n_124,tmp_product_n_125,tmp_product_n_126,tmp_product_n_127,tmp_product_n_128,tmp_product_n_129,tmp_product_n_130,tmp_product_n_131,tmp_product_n_132,tmp_product_n_133,tmp_product_n_134,tmp_product_n_135,tmp_product_n_136,tmp_product_n_137,tmp_product_n_138,tmp_product_n_139,tmp_product_n_140,tmp_product_n_141,tmp_product_n_142,tmp_product_n_143,tmp_product_n_144,tmp_product_n_145,tmp_product_n_146,tmp_product_n_147,tmp_product_n_148,tmp_product_n_149,tmp_product_n_150,tmp_product_n_151,tmp_product_n_152,tmp_product_n_153}),
        .PCOUT(NLW_buff0_reg_PCOUT_UNCONNECTED[47:0]),
        .RSTA(smax_reg_179),
        .RSTALLCARRYIN(1'b0),
        .RSTALUMODE(1'b0),
        .RSTB(smax1_reg_184),
        .RSTC(1'b0),
        .RSTCTRL(1'b0),
        .RSTD(1'b0),
        .RSTINMODE(1'b0),
        .RSTM(1'b0),
        .RSTP(1'b0),
        .UNDERFLOW(NLW_buff0_reg_UNDERFLOW_UNCONNECTED));
  FDRE \buff0_reg[0] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product_n_105),
        .Q(\buff0_reg_n_0_[0] ),
        .R(1'b0));
  FDRE \buff0_reg[0]__0 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product__17_n_105),
        .Q(buff0_reg__0_0[0]),
        .R(1'b0));
  FDRE \buff0_reg[10] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product_n_95),
        .Q(\buff0_reg_n_0_[10] ),
        .R(1'b0));
  FDRE \buff0_reg[10]__0 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product__17_n_95),
        .Q(buff0_reg__0_0[10]),
        .R(1'b0));
  FDRE \buff0_reg[11] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product_n_94),
        .Q(\buff0_reg_n_0_[11] ),
        .R(1'b0));
  FDRE \buff0_reg[11]__0 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product__17_n_94),
        .Q(buff0_reg__0_0[11]),
        .R(1'b0));
  FDRE \buff0_reg[12] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product_n_93),
        .Q(\buff0_reg_n_0_[12] ),
        .R(1'b0));
  FDRE \buff0_reg[12]__0 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product__17_n_93),
        .Q(buff0_reg__0_0[12]),
        .R(1'b0));
  FDRE \buff0_reg[13] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product_n_92),
        .Q(\buff0_reg_n_0_[13] ),
        .R(1'b0));
  FDRE \buff0_reg[13]__0 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product__17_n_92),
        .Q(buff0_reg__0_0[13]),
        .R(1'b0));
  FDRE \buff0_reg[14] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product_n_91),
        .Q(\buff0_reg_n_0_[14] ),
        .R(1'b0));
  FDRE \buff0_reg[14]__0 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product__17_n_91),
        .Q(buff0_reg__0_0[14]),
        .R(1'b0));
  FDRE \buff0_reg[15] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product_n_90),
        .Q(\buff0_reg_n_0_[15] ),
        .R(1'b0));
  FDRE \buff0_reg[15]__0 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product__17_n_90),
        .Q(buff0_reg__0_0[15]),
        .R(1'b0));
  FDRE \buff0_reg[16] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product_n_89),
        .Q(\buff0_reg_n_0_[16] ),
        .R(1'b0));
  FDRE \buff0_reg[16]__0 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product__17_n_89),
        .Q(\buff0_reg[16]__0_n_0 ),
        .R(1'b0));
  FDRE \buff0_reg[1] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product_n_104),
        .Q(\buff0_reg_n_0_[1] ),
        .R(1'b0));
  FDRE \buff0_reg[1]__0 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product__17_n_104),
        .Q(buff0_reg__0_0[1]),
        .R(1'b0));
  FDRE \buff0_reg[2] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product_n_103),
        .Q(\buff0_reg_n_0_[2] ),
        .R(1'b0));
  FDRE \buff0_reg[2]__0 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product__17_n_103),
        .Q(buff0_reg__0_0[2]),
        .R(1'b0));
  FDRE \buff0_reg[3] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product_n_102),
        .Q(\buff0_reg_n_0_[3] ),
        .R(1'b0));
  FDRE \buff0_reg[3]__0 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product__17_n_102),
        .Q(buff0_reg__0_0[3]),
        .R(1'b0));
  FDRE \buff0_reg[4] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product_n_101),
        .Q(\buff0_reg_n_0_[4] ),
        .R(1'b0));
  FDRE \buff0_reg[4]__0 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product__17_n_101),
        .Q(buff0_reg__0_0[4]),
        .R(1'b0));
  FDRE \buff0_reg[5] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product_n_100),
        .Q(\buff0_reg_n_0_[5] ),
        .R(1'b0));
  FDRE \buff0_reg[5]__0 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product__17_n_100),
        .Q(buff0_reg__0_0[5]),
        .R(1'b0));
  FDRE \buff0_reg[6] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product_n_99),
        .Q(\buff0_reg_n_0_[6] ),
        .R(1'b0));
  FDRE \buff0_reg[6]__0 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product__17_n_99),
        .Q(buff0_reg__0_0[6]),
        .R(1'b0));
  FDRE \buff0_reg[7] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product_n_98),
        .Q(\buff0_reg_n_0_[7] ),
        .R(1'b0));
  FDRE \buff0_reg[7]__0 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product__17_n_98),
        .Q(buff0_reg__0_0[7]),
        .R(1'b0));
  FDRE \buff0_reg[8] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product_n_97),
        .Q(\buff0_reg_n_0_[8] ),
        .R(1'b0));
  FDRE \buff0_reg[8]__0 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product__17_n_97),
        .Q(buff0_reg__0_0[8]),
        .R(1'b0));
  FDRE \buff0_reg[9] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product_n_96),
        .Q(\buff0_reg_n_0_[9] ),
        .R(1'b0));
  FDRE \buff0_reg[9]__0 
       (.C(ap_clk),
        .CE(1'b1),
        .D(tmp_product__17_n_96),
        .Q(buff0_reg__0_0[9]),
        .R(1'b0));
  (* METHODOLOGY_DRC_VIOS = "{SYNTH-10 {cell *THIS*} {string 18x15 4}}" *) 
  DSP48E1 #(
    .ACASCREG(0),
    .ADREG(1),
    .ALUMODEREG(0),
    .AREG(0),
    .AUTORESET_PATDET("NO_RESET"),
    .A_INPUT("CASCADE"),
    .BCASCREG(1),
    .BREG(1),
    .B_INPUT("DIRECT"),
    .CARRYINREG(0),
    .CARRYINSELREG(0),
    .CREG(1),
    .DREG(1),
    .INMODEREG(0),
    .MASK(48'h3FFFFFFFFFFF),
    .MREG(0),
    .OPMODEREG(0),
    .PATTERN(48'h000000000000),
    .PREG(1),
    .SEL_MASK("MASK"),
    .SEL_PATTERN("PATTERN"),
    .USE_DPORT("FALSE"),
    .USE_MULT("MULTIPLY"),
    .USE_PATTERN_DETECT("NO_PATDET"),
    .USE_SIMD("ONE48")) 
    buff0_reg__0
       (.A({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .ACIN({tmp_product__17_n_24,tmp_product__17_n_25,tmp_product__17_n_26,tmp_product__17_n_27,tmp_product__17_n_28,tmp_product__17_n_29,tmp_product__17_n_30,tmp_product__17_n_31,tmp_product__17_n_32,tmp_product__17_n_33,tmp_product__17_n_34,tmp_product__17_n_35,tmp_product__17_n_36,tmp_product__17_n_37,tmp_product__17_n_38,tmp_product__17_n_39,tmp_product__17_n_40,tmp_product__17_n_41,tmp_product__17_n_42,tmp_product__17_n_43,tmp_product__17_n_44,tmp_product__17_n_45,tmp_product__17_n_46,tmp_product__17_n_47,tmp_product__17_n_48,tmp_product__17_n_49,tmp_product__17_n_50,tmp_product__17_n_51,tmp_product__17_n_52,tmp_product__17_n_53}),
        .ACOUT(NLW_buff0_reg__0_ACOUT_UNCONNECTED[29:0]),
        .ALUMODE({1'b0,1'b0,1'b0,1'b0}),
        .B({1'b0,1'b0,1'b0,1'b0,buff0_reg__0_1}),
        .BCIN({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .BCOUT(NLW_buff0_reg__0_BCOUT_UNCONNECTED[17:0]),
        .C({1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1}),
        .CARRYCASCIN(1'b0),
        .CARRYCASCOUT(NLW_buff0_reg__0_CARRYCASCOUT_UNCONNECTED),
        .CARRYIN(1'b0),
        .CARRYINSEL({1'b0,1'b0,1'b0}),
        .CARRYOUT(NLW_buff0_reg__0_CARRYOUT_UNCONNECTED[3:0]),
        .CEA1(1'b0),
        .CEA2(1'b0),
        .CEAD(1'b0),
        .CEALUMODE(1'b0),
        .CEB1(1'b0),
        .CEB2(Q),
        .CEC(1'b0),
        .CECARRYIN(1'b0),
        .CECTRL(1'b0),
        .CED(1'b0),
        .CEINMODE(1'b0),
        .CEM(1'b0),
        .CEP(1'b1),
        .CLK(ap_clk),
        .D({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .INMODE({1'b0,1'b0,1'b0,1'b0,1'b0}),
        .MULTSIGNIN(1'b0),
        .MULTSIGNOUT(NLW_buff0_reg__0_MULTSIGNOUT_UNCONNECTED),
        .OPMODE({1'b1,1'b0,1'b1,1'b0,1'b1,1'b0,1'b1}),
        .OVERFLOW(NLW_buff0_reg__0_OVERFLOW_UNCONNECTED),
        .P({buff0_reg__0_n_58,buff0_reg__0_n_59,buff0_reg__0_n_60,buff0_reg__0_n_61,buff0_reg__0_n_62,buff0_reg__0_n_63,buff0_reg__0_n_64,buff0_reg__0_n_65,buff0_reg__0_n_66,buff0_reg__0_n_67,buff0_reg__0_n_68,buff0_reg__0_n_69,buff0_reg__0_n_70,buff0_reg__0_n_71,buff0_reg__0_n_72,buff0_reg__0_n_73,buff0_reg__0_n_74,buff0_reg__0_n_75,buff0_reg__0_n_76,buff0_reg__0_n_77,buff0_reg__0_n_78,buff0_reg__0_n_79,buff0_reg__0_n_80,buff0_reg__0_n_81,buff0_reg__0_n_82,buff0_reg__0_n_83,buff0_reg__0_n_84,buff0_reg__0_n_85,buff0_reg__0_n_86,buff0_reg__0_n_87,buff0_reg__0_n_88,buff0_reg__0_n_89,buff0_reg__0_n_90,buff0_reg__0_n_91,buff0_reg__0_n_92,buff0_reg__0_n_93,buff0_reg__0_n_94,buff0_reg__0_n_95,buff0_reg__0_n_96,buff0_reg__0_n_97,buff0_reg__0_n_98,buff0_reg__0_n_99,buff0_reg__0_n_100,buff0_reg__0_n_101,buff0_reg__0_n_102,buff0_reg__0_n_103,buff0_reg__0_n_104,buff0_reg__0_n_105}),
        .PATTERNBDETECT(NLW_buff0_reg__0_PATTERNBDETECT_UNCONNECTED),
        .PATTERNDETECT(NLW_buff0_reg__0_PATTERNDETECT_UNCONNECTED),
        .PCIN({tmp_product__17_n_106,tmp_product__17_n_107,tmp_product__17_n_108,tmp_product__17_n_109,tmp_product__17_n_110,tmp_product__17_n_111,tmp_product__17_n_112,tmp_product__17_n_113,tmp_product__17_n_114,tmp_product__17_n_115,tmp_product__17_n_116,tmp_product__17_n_117,tmp_product__17_n_118,tmp_product__17_n_119,tmp_product__17_n_120,tmp_product__17_n_121,tmp_product__17_n_122,tmp_product__17_n_123,tmp_product__17_n_124,tmp_product__17_n_125,tmp_product__17_n_126,tmp_product__17_n_127,tmp_product__17_n_128,tmp_product__17_n_129,tmp_product__17_n_130,tmp_product__17_n_131,tmp_product__17_n_132,tmp_product__17_n_133,tmp_product__17_n_134,tmp_product__17_n_135,tmp_product__17_n_136,tmp_product__17_n_137,tmp_product__17_n_138,tmp_product__17_n_139,tmp_product__17_n_140,tmp_product__17_n_141,tmp_product__17_n_142,tmp_product__17_n_143,tmp_product__17_n_144,tmp_product__17_n_145,tmp_product__17_n_146,tmp_product__17_n_147,tmp_product__17_n_148,tmp_product__17_n_149,tmp_product__17_n_150,tmp_product__17_n_151,tmp_product__17_n_152,tmp_product__17_n_153}),
        .PCOUT(NLW_buff0_reg__0_PCOUT_UNCONNECTED[47:0]),
        .RSTA(1'b0),
        .RSTALLCARRYIN(1'b0),
        .RSTALUMODE(1'b0),
        .RSTB(smax1_reg_184),
        .RSTC(1'b0),
        .RSTCTRL(1'b0),
        .RSTD(1'b0),
        .RSTINMODE(1'b0),
        .RSTM(1'b0),
        .RSTP(1'b0),
        .UNDERFLOW(NLW_buff0_reg__0_UNDERFLOW_UNCONNECTED));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[19]_i_2 
       (.I0(buff0_reg__0_n_103),
        .I1(\buff0_reg_n_0_[2] ),
        .O(\mul_ln10_reg_199[19]_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[19]_i_3 
       (.I0(buff0_reg__0_n_104),
        .I1(\buff0_reg_n_0_[1] ),
        .O(\mul_ln10_reg_199[19]_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[19]_i_4 
       (.I0(buff0_reg__0_n_105),
        .I1(\buff0_reg_n_0_[0] ),
        .O(\mul_ln10_reg_199[19]_i_4_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[23]_i_2 
       (.I0(buff0_reg__0_n_99),
        .I1(\buff0_reg_n_0_[6] ),
        .O(\mul_ln10_reg_199[23]_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[23]_i_3 
       (.I0(buff0_reg__0_n_100),
        .I1(\buff0_reg_n_0_[5] ),
        .O(\mul_ln10_reg_199[23]_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[23]_i_4 
       (.I0(buff0_reg__0_n_101),
        .I1(\buff0_reg_n_0_[4] ),
        .O(\mul_ln10_reg_199[23]_i_4_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[23]_i_5 
       (.I0(buff0_reg__0_n_102),
        .I1(\buff0_reg_n_0_[3] ),
        .O(\mul_ln10_reg_199[23]_i_5_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[27]_i_2 
       (.I0(buff0_reg__0_n_95),
        .I1(\buff0_reg_n_0_[10] ),
        .O(\mul_ln10_reg_199[27]_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[27]_i_3 
       (.I0(buff0_reg__0_n_96),
        .I1(\buff0_reg_n_0_[9] ),
        .O(\mul_ln10_reg_199[27]_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[27]_i_4 
       (.I0(buff0_reg__0_n_97),
        .I1(\buff0_reg_n_0_[8] ),
        .O(\mul_ln10_reg_199[27]_i_4_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[27]_i_5 
       (.I0(buff0_reg__0_n_98),
        .I1(\buff0_reg_n_0_[7] ),
        .O(\mul_ln10_reg_199[27]_i_5_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[31]_i_2 
       (.I0(buff0_reg__0_n_91),
        .I1(\buff0_reg_n_0_[14] ),
        .O(\mul_ln10_reg_199[31]_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[31]_i_3 
       (.I0(buff0_reg__0_n_92),
        .I1(\buff0_reg_n_0_[13] ),
        .O(\mul_ln10_reg_199[31]_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[31]_i_4 
       (.I0(buff0_reg__0_n_93),
        .I1(\buff0_reg_n_0_[12] ),
        .O(\mul_ln10_reg_199[31]_i_4_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[31]_i_5 
       (.I0(buff0_reg__0_n_94),
        .I1(\buff0_reg_n_0_[11] ),
        .O(\mul_ln10_reg_199[31]_i_5_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[35]_i_2 
       (.I0(buff0_reg__0_n_87),
        .I1(buff0_reg_n_104),
        .O(\mul_ln10_reg_199[35]_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[35]_i_3 
       (.I0(buff0_reg__0_n_88),
        .I1(buff0_reg_n_105),
        .O(\mul_ln10_reg_199[35]_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[35]_i_4 
       (.I0(buff0_reg__0_n_89),
        .I1(\buff0_reg_n_0_[16] ),
        .O(\mul_ln10_reg_199[35]_i_4_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[35]_i_5 
       (.I0(buff0_reg__0_n_90),
        .I1(\buff0_reg_n_0_[15] ),
        .O(\mul_ln10_reg_199[35]_i_5_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[39]_i_2 
       (.I0(buff0_reg__0_n_83),
        .I1(buff0_reg_n_100),
        .O(\mul_ln10_reg_199[39]_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[39]_i_3 
       (.I0(buff0_reg__0_n_84),
        .I1(buff0_reg_n_101),
        .O(\mul_ln10_reg_199[39]_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[39]_i_4 
       (.I0(buff0_reg__0_n_85),
        .I1(buff0_reg_n_102),
        .O(\mul_ln10_reg_199[39]_i_4_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[39]_i_5 
       (.I0(buff0_reg__0_n_86),
        .I1(buff0_reg_n_103),
        .O(\mul_ln10_reg_199[39]_i_5_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[43]_i_2 
       (.I0(buff0_reg__0_n_79),
        .I1(buff0_reg_n_96),
        .O(\mul_ln10_reg_199[43]_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[43]_i_3 
       (.I0(buff0_reg__0_n_80),
        .I1(buff0_reg_n_97),
        .O(\mul_ln10_reg_199[43]_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[43]_i_4 
       (.I0(buff0_reg__0_n_81),
        .I1(buff0_reg_n_98),
        .O(\mul_ln10_reg_199[43]_i_4_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[43]_i_5 
       (.I0(buff0_reg__0_n_82),
        .I1(buff0_reg_n_99),
        .O(\mul_ln10_reg_199[43]_i_5_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[47]_i_2 
       (.I0(buff0_reg__0_n_75),
        .I1(buff0_reg_n_92),
        .O(\mul_ln10_reg_199[47]_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[47]_i_3 
       (.I0(buff0_reg__0_n_76),
        .I1(buff0_reg_n_93),
        .O(\mul_ln10_reg_199[47]_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[47]_i_4 
       (.I0(buff0_reg__0_n_77),
        .I1(buff0_reg_n_94),
        .O(\mul_ln10_reg_199[47]_i_4_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[47]_i_5 
       (.I0(buff0_reg__0_n_78),
        .I1(buff0_reg_n_95),
        .O(\mul_ln10_reg_199[47]_i_5_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[51]_i_2 
       (.I0(buff0_reg__0_n_71),
        .I1(buff0_reg_n_88),
        .O(\mul_ln10_reg_199[51]_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[51]_i_3 
       (.I0(buff0_reg__0_n_72),
        .I1(buff0_reg_n_89),
        .O(\mul_ln10_reg_199[51]_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[51]_i_4 
       (.I0(buff0_reg__0_n_73),
        .I1(buff0_reg_n_90),
        .O(\mul_ln10_reg_199[51]_i_4_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[51]_i_5 
       (.I0(buff0_reg__0_n_74),
        .I1(buff0_reg_n_91),
        .O(\mul_ln10_reg_199[51]_i_5_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[55]_i_2 
       (.I0(buff0_reg__0_n_67),
        .I1(buff0_reg_n_84),
        .O(\mul_ln10_reg_199[55]_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[55]_i_3 
       (.I0(buff0_reg__0_n_68),
        .I1(buff0_reg_n_85),
        .O(\mul_ln10_reg_199[55]_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[55]_i_4 
       (.I0(buff0_reg__0_n_69),
        .I1(buff0_reg_n_86),
        .O(\mul_ln10_reg_199[55]_i_4_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[55]_i_5 
       (.I0(buff0_reg__0_n_70),
        .I1(buff0_reg_n_87),
        .O(\mul_ln10_reg_199[55]_i_5_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[59]_i_2 
       (.I0(buff0_reg__0_n_63),
        .I1(buff0_reg_n_80),
        .O(\mul_ln10_reg_199[59]_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[59]_i_3 
       (.I0(buff0_reg__0_n_64),
        .I1(buff0_reg_n_81),
        .O(\mul_ln10_reg_199[59]_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[59]_i_4 
       (.I0(buff0_reg__0_n_65),
        .I1(buff0_reg_n_82),
        .O(\mul_ln10_reg_199[59]_i_4_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[59]_i_5 
       (.I0(buff0_reg__0_n_66),
        .I1(buff0_reg_n_83),
        .O(\mul_ln10_reg_199[59]_i_5_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[61]_i_2 
       (.I0(buff0_reg__0_n_61),
        .I1(buff0_reg_n_78),
        .O(\mul_ln10_reg_199[61]_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \mul_ln10_reg_199[61]_i_3 
       (.I0(buff0_reg__0_n_62),
        .I1(buff0_reg_n_79),
        .O(\mul_ln10_reg_199[61]_i_3_n_0 ));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \mul_ln10_reg_199_reg[19]_i_1 
       (.CI(1'b0),
        .CO({\mul_ln10_reg_199_reg[19]_i_1_n_0 ,\mul_ln10_reg_199_reg[19]_i_1_n_1 ,\mul_ln10_reg_199_reg[19]_i_1_n_2 ,\mul_ln10_reg_199_reg[19]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({buff0_reg__0_n_103,buff0_reg__0_n_104,buff0_reg__0_n_105,1'b0}),
        .O(buff0_reg__0_0[19:16]),
        .S({\mul_ln10_reg_199[19]_i_2_n_0 ,\mul_ln10_reg_199[19]_i_3_n_0 ,\mul_ln10_reg_199[19]_i_4_n_0 ,\buff0_reg[16]__0_n_0 }));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \mul_ln10_reg_199_reg[23]_i_1 
       (.CI(\mul_ln10_reg_199_reg[19]_i_1_n_0 ),
        .CO({\mul_ln10_reg_199_reg[23]_i_1_n_0 ,\mul_ln10_reg_199_reg[23]_i_1_n_1 ,\mul_ln10_reg_199_reg[23]_i_1_n_2 ,\mul_ln10_reg_199_reg[23]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({buff0_reg__0_n_99,buff0_reg__0_n_100,buff0_reg__0_n_101,buff0_reg__0_n_102}),
        .O(buff0_reg__0_0[23:20]),
        .S({\mul_ln10_reg_199[23]_i_2_n_0 ,\mul_ln10_reg_199[23]_i_3_n_0 ,\mul_ln10_reg_199[23]_i_4_n_0 ,\mul_ln10_reg_199[23]_i_5_n_0 }));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \mul_ln10_reg_199_reg[27]_i_1 
       (.CI(\mul_ln10_reg_199_reg[23]_i_1_n_0 ),
        .CO({\mul_ln10_reg_199_reg[27]_i_1_n_0 ,\mul_ln10_reg_199_reg[27]_i_1_n_1 ,\mul_ln10_reg_199_reg[27]_i_1_n_2 ,\mul_ln10_reg_199_reg[27]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({buff0_reg__0_n_95,buff0_reg__0_n_96,buff0_reg__0_n_97,buff0_reg__0_n_98}),
        .O(buff0_reg__0_0[27:24]),
        .S({\mul_ln10_reg_199[27]_i_2_n_0 ,\mul_ln10_reg_199[27]_i_3_n_0 ,\mul_ln10_reg_199[27]_i_4_n_0 ,\mul_ln10_reg_199[27]_i_5_n_0 }));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \mul_ln10_reg_199_reg[31]_i_1 
       (.CI(\mul_ln10_reg_199_reg[27]_i_1_n_0 ),
        .CO({\mul_ln10_reg_199_reg[31]_i_1_n_0 ,\mul_ln10_reg_199_reg[31]_i_1_n_1 ,\mul_ln10_reg_199_reg[31]_i_1_n_2 ,\mul_ln10_reg_199_reg[31]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({buff0_reg__0_n_91,buff0_reg__0_n_92,buff0_reg__0_n_93,buff0_reg__0_n_94}),
        .O(buff0_reg__0_0[31:28]),
        .S({\mul_ln10_reg_199[31]_i_2_n_0 ,\mul_ln10_reg_199[31]_i_3_n_0 ,\mul_ln10_reg_199[31]_i_4_n_0 ,\mul_ln10_reg_199[31]_i_5_n_0 }));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \mul_ln10_reg_199_reg[35]_i_1 
       (.CI(\mul_ln10_reg_199_reg[31]_i_1_n_0 ),
        .CO({\mul_ln10_reg_199_reg[35]_i_1_n_0 ,\mul_ln10_reg_199_reg[35]_i_1_n_1 ,\mul_ln10_reg_199_reg[35]_i_1_n_2 ,\mul_ln10_reg_199_reg[35]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({buff0_reg__0_n_87,buff0_reg__0_n_88,buff0_reg__0_n_89,buff0_reg__0_n_90}),
        .O(buff0_reg__0_0[35:32]),
        .S({\mul_ln10_reg_199[35]_i_2_n_0 ,\mul_ln10_reg_199[35]_i_3_n_0 ,\mul_ln10_reg_199[35]_i_4_n_0 ,\mul_ln10_reg_199[35]_i_5_n_0 }));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \mul_ln10_reg_199_reg[39]_i_1 
       (.CI(\mul_ln10_reg_199_reg[35]_i_1_n_0 ),
        .CO({\mul_ln10_reg_199_reg[39]_i_1_n_0 ,\mul_ln10_reg_199_reg[39]_i_1_n_1 ,\mul_ln10_reg_199_reg[39]_i_1_n_2 ,\mul_ln10_reg_199_reg[39]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({buff0_reg__0_n_83,buff0_reg__0_n_84,buff0_reg__0_n_85,buff0_reg__0_n_86}),
        .O(buff0_reg__0_0[39:36]),
        .S({\mul_ln10_reg_199[39]_i_2_n_0 ,\mul_ln10_reg_199[39]_i_3_n_0 ,\mul_ln10_reg_199[39]_i_4_n_0 ,\mul_ln10_reg_199[39]_i_5_n_0 }));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \mul_ln10_reg_199_reg[43]_i_1 
       (.CI(\mul_ln10_reg_199_reg[39]_i_1_n_0 ),
        .CO({\mul_ln10_reg_199_reg[43]_i_1_n_0 ,\mul_ln10_reg_199_reg[43]_i_1_n_1 ,\mul_ln10_reg_199_reg[43]_i_1_n_2 ,\mul_ln10_reg_199_reg[43]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({buff0_reg__0_n_79,buff0_reg__0_n_80,buff0_reg__0_n_81,buff0_reg__0_n_82}),
        .O(buff0_reg__0_0[43:40]),
        .S({\mul_ln10_reg_199[43]_i_2_n_0 ,\mul_ln10_reg_199[43]_i_3_n_0 ,\mul_ln10_reg_199[43]_i_4_n_0 ,\mul_ln10_reg_199[43]_i_5_n_0 }));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \mul_ln10_reg_199_reg[47]_i_1 
       (.CI(\mul_ln10_reg_199_reg[43]_i_1_n_0 ),
        .CO({\mul_ln10_reg_199_reg[47]_i_1_n_0 ,\mul_ln10_reg_199_reg[47]_i_1_n_1 ,\mul_ln10_reg_199_reg[47]_i_1_n_2 ,\mul_ln10_reg_199_reg[47]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({buff0_reg__0_n_75,buff0_reg__0_n_76,buff0_reg__0_n_77,buff0_reg__0_n_78}),
        .O(buff0_reg__0_0[47:44]),
        .S({\mul_ln10_reg_199[47]_i_2_n_0 ,\mul_ln10_reg_199[47]_i_3_n_0 ,\mul_ln10_reg_199[47]_i_4_n_0 ,\mul_ln10_reg_199[47]_i_5_n_0 }));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \mul_ln10_reg_199_reg[51]_i_1 
       (.CI(\mul_ln10_reg_199_reg[47]_i_1_n_0 ),
        .CO({\mul_ln10_reg_199_reg[51]_i_1_n_0 ,\mul_ln10_reg_199_reg[51]_i_1_n_1 ,\mul_ln10_reg_199_reg[51]_i_1_n_2 ,\mul_ln10_reg_199_reg[51]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({buff0_reg__0_n_71,buff0_reg__0_n_72,buff0_reg__0_n_73,buff0_reg__0_n_74}),
        .O(buff0_reg__0_0[51:48]),
        .S({\mul_ln10_reg_199[51]_i_2_n_0 ,\mul_ln10_reg_199[51]_i_3_n_0 ,\mul_ln10_reg_199[51]_i_4_n_0 ,\mul_ln10_reg_199[51]_i_5_n_0 }));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \mul_ln10_reg_199_reg[55]_i_1 
       (.CI(\mul_ln10_reg_199_reg[51]_i_1_n_0 ),
        .CO({\mul_ln10_reg_199_reg[55]_i_1_n_0 ,\mul_ln10_reg_199_reg[55]_i_1_n_1 ,\mul_ln10_reg_199_reg[55]_i_1_n_2 ,\mul_ln10_reg_199_reg[55]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({buff0_reg__0_n_67,buff0_reg__0_n_68,buff0_reg__0_n_69,buff0_reg__0_n_70}),
        .O(buff0_reg__0_0[55:52]),
        .S({\mul_ln10_reg_199[55]_i_2_n_0 ,\mul_ln10_reg_199[55]_i_3_n_0 ,\mul_ln10_reg_199[55]_i_4_n_0 ,\mul_ln10_reg_199[55]_i_5_n_0 }));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \mul_ln10_reg_199_reg[59]_i_1 
       (.CI(\mul_ln10_reg_199_reg[55]_i_1_n_0 ),
        .CO({\mul_ln10_reg_199_reg[59]_i_1_n_0 ,\mul_ln10_reg_199_reg[59]_i_1_n_1 ,\mul_ln10_reg_199_reg[59]_i_1_n_2 ,\mul_ln10_reg_199_reg[59]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({buff0_reg__0_n_63,buff0_reg__0_n_64,buff0_reg__0_n_65,buff0_reg__0_n_66}),
        .O(buff0_reg__0_0[59:56]),
        .S({\mul_ln10_reg_199[59]_i_2_n_0 ,\mul_ln10_reg_199[59]_i_3_n_0 ,\mul_ln10_reg_199[59]_i_4_n_0 ,\mul_ln10_reg_199[59]_i_5_n_0 }));
  (* ADDER_THRESHOLD = "35" *) 
  CARRY4 \mul_ln10_reg_199_reg[61]_i_1 
       (.CI(\mul_ln10_reg_199_reg[59]_i_1_n_0 ),
        .CO({\NLW_mul_ln10_reg_199_reg[61]_i_1_CO_UNCONNECTED [3:1],\mul_ln10_reg_199_reg[61]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,buff0_reg__0_n_62}),
        .O({\NLW_mul_ln10_reg_199_reg[61]_i_1_O_UNCONNECTED [3:2],buff0_reg__0_0[61:60]}),
        .S({1'b0,1'b0,\mul_ln10_reg_199[61]_i_2_n_0 ,\mul_ln10_reg_199[61]_i_3_n_0 }));
  (* METHODOLOGY_DRC_VIOS = "{SYNTH-10 {cell *THIS*} {string 15x18 4}}" *) 
  DSP48E1 #(
    .ACASCREG(1),
    .ADREG(1),
    .ALUMODEREG(0),
    .AREG(1),
    .AUTORESET_PATDET("NO_RESET"),
    .A_INPUT("DIRECT"),
    .BCASCREG(1),
    .BREG(1),
    .B_INPUT("DIRECT"),
    .CARRYINREG(0),
    .CARRYINSELREG(0),
    .CREG(1),
    .DREG(1),
    .INMODEREG(0),
    .MASK(48'h3FFFFFFFFFFF),
    .MREG(0),
    .OPMODEREG(0),
    .PATTERN(48'h000000000000),
    .PREG(0),
    .SEL_MASK("MASK"),
    .SEL_PATTERN("PATTERN"),
    .USE_DPORT("FALSE"),
    .USE_MULT("MULTIPLY"),
    .USE_PATTERN_DETECT("NO_PATDET"),
    .USE_SIMD("ONE48")) 
    tmp_product
       (.A({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,tmp_product__0_n_0,tmp_product__1_n_0,tmp_product__2_n_0,tmp_product__3_n_0,tmp_product__4_n_0,tmp_product__5_n_0,tmp_product__6_n_0,tmp_product__7_n_0,tmp_product__8_n_0,tmp_product__9_n_0,tmp_product__10_n_0,tmp_product__11_n_0,tmp_product__12_n_0,tmp_product__13_n_0,tmp_product__14_n_0,tmp_product__15_n_0,tmp_product__16_n_0}),
        .ACIN({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .ACOUT(NLW_tmp_product_ACOUT_UNCONNECTED[29:0]),
        .ALUMODE({1'b0,1'b0,1'b0,1'b0}),
        .B({1'b0,1'b0,1'b0,1'b0,buff0_reg_0}),
        .BCIN({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .BCOUT(NLW_tmp_product_BCOUT_UNCONNECTED[17:0]),
        .C({1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1}),
        .CARRYCASCIN(1'b0),
        .CARRYCASCOUT(NLW_tmp_product_CARRYCASCOUT_UNCONNECTED),
        .CARRYIN(1'b0),
        .CARRYINSEL({1'b0,1'b0,1'b0}),
        .CARRYOUT(NLW_tmp_product_CARRYOUT_UNCONNECTED[3:0]),
        .CEA1(1'b0),
        .CEA2(Q),
        .CEAD(1'b0),
        .CEALUMODE(1'b0),
        .CEB1(1'b0),
        .CEB2(Q),
        .CEC(1'b0),
        .CECARRYIN(1'b0),
        .CECTRL(1'b0),
        .CED(1'b0),
        .CEINMODE(1'b0),
        .CEM(1'b0),
        .CEP(1'b0),
        .CLK(ap_clk),
        .D({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .INMODE({1'b0,1'b0,1'b0,1'b0,1'b0}),
        .MULTSIGNIN(1'b0),
        .MULTSIGNOUT(NLW_tmp_product_MULTSIGNOUT_UNCONNECTED),
        .OPMODE({1'b0,1'b0,1'b0,1'b0,1'b1,1'b0,1'b1}),
        .OVERFLOW(NLW_tmp_product_OVERFLOW_UNCONNECTED),
        .P({tmp_product_n_58,tmp_product_n_59,tmp_product_n_60,tmp_product_n_61,tmp_product_n_62,tmp_product_n_63,tmp_product_n_64,tmp_product_n_65,tmp_product_n_66,tmp_product_n_67,tmp_product_n_68,tmp_product_n_69,tmp_product_n_70,tmp_product_n_71,tmp_product_n_72,tmp_product_n_73,tmp_product_n_74,tmp_product_n_75,tmp_product_n_76,tmp_product_n_77,tmp_product_n_78,tmp_product_n_79,tmp_product_n_80,tmp_product_n_81,tmp_product_n_82,tmp_product_n_83,tmp_product_n_84,tmp_product_n_85,tmp_product_n_86,tmp_product_n_87,tmp_product_n_88,tmp_product_n_89,tmp_product_n_90,tmp_product_n_91,tmp_product_n_92,tmp_product_n_93,tmp_product_n_94,tmp_product_n_95,tmp_product_n_96,tmp_product_n_97,tmp_product_n_98,tmp_product_n_99,tmp_product_n_100,tmp_product_n_101,tmp_product_n_102,tmp_product_n_103,tmp_product_n_104,tmp_product_n_105}),
        .PATTERNBDETECT(NLW_tmp_product_PATTERNBDETECT_UNCONNECTED),
        .PATTERNDETECT(NLW_tmp_product_PATTERNDETECT_UNCONNECTED),
        .PCIN({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .PCOUT({tmp_product_n_106,tmp_product_n_107,tmp_product_n_108,tmp_product_n_109,tmp_product_n_110,tmp_product_n_111,tmp_product_n_112,tmp_product_n_113,tmp_product_n_114,tmp_product_n_115,tmp_product_n_116,tmp_product_n_117,tmp_product_n_118,tmp_product_n_119,tmp_product_n_120,tmp_product_n_121,tmp_product_n_122,tmp_product_n_123,tmp_product_n_124,tmp_product_n_125,tmp_product_n_126,tmp_product_n_127,tmp_product_n_128,tmp_product_n_129,tmp_product_n_130,tmp_product_n_131,tmp_product_n_132,tmp_product_n_133,tmp_product_n_134,tmp_product_n_135,tmp_product_n_136,tmp_product_n_137,tmp_product_n_138,tmp_product_n_139,tmp_product_n_140,tmp_product_n_141,tmp_product_n_142,tmp_product_n_143,tmp_product_n_144,tmp_product_n_145,tmp_product_n_146,tmp_product_n_147,tmp_product_n_148,tmp_product_n_149,tmp_product_n_150,tmp_product_n_151,tmp_product_n_152,tmp_product_n_153}),
        .RSTA(smax1_reg_184),
        .RSTALLCARRYIN(1'b0),
        .RSTALUMODE(1'b0),
        .RSTB(smax_reg_179),
        .RSTC(1'b0),
        .RSTCTRL(1'b0),
        .RSTD(1'b0),
        .RSTINMODE(1'b0),
        .RSTM(1'b0),
        .RSTP(1'b0),
        .UNDERFLOW(NLW_tmp_product_UNDERFLOW_UNCONNECTED));
  FDRE tmp_product__0
       (.C(ap_clk),
        .CE(E),
        .D(D[16]),
        .Q(tmp_product__0_n_0),
        .R(1'b0));
  FDRE tmp_product__1
       (.C(ap_clk),
        .CE(E),
        .D(D[15]),
        .Q(tmp_product__1_n_0),
        .R(1'b0));
  FDRE tmp_product__10
       (.C(ap_clk),
        .CE(E),
        .D(D[6]),
        .Q(tmp_product__10_n_0),
        .R(1'b0));
  FDRE tmp_product__11
       (.C(ap_clk),
        .CE(E),
        .D(D[5]),
        .Q(tmp_product__11_n_0),
        .R(1'b0));
  FDRE tmp_product__12
       (.C(ap_clk),
        .CE(E),
        .D(D[4]),
        .Q(tmp_product__12_n_0),
        .R(1'b0));
  FDRE tmp_product__13
       (.C(ap_clk),
        .CE(E),
        .D(D[3]),
        .Q(tmp_product__13_n_0),
        .R(1'b0));
  FDRE tmp_product__14
       (.C(ap_clk),
        .CE(E),
        .D(D[2]),
        .Q(tmp_product__14_n_0),
        .R(1'b0));
  FDRE tmp_product__15
       (.C(ap_clk),
        .CE(E),
        .D(D[1]),
        .Q(tmp_product__15_n_0),
        .R(1'b0));
  FDRE tmp_product__16
       (.C(ap_clk),
        .CE(E),
        .D(D[0]),
        .Q(tmp_product__16_n_0),
        .R(1'b0));
  (* METHODOLOGY_DRC_VIOS = "{SYNTH-10 {cell *THIS*} {string 18x18 4}}" *) 
  DSP48E1 #(
    .ACASCREG(1),
    .ADREG(1),
    .ALUMODEREG(0),
    .AREG(1),
    .AUTORESET_PATDET("NO_RESET"),
    .A_INPUT("DIRECT"),
    .BCASCREG(1),
    .BREG(1),
    .B_INPUT("DIRECT"),
    .CARRYINREG(0),
    .CARRYINSELREG(0),
    .CREG(1),
    .DREG(1),
    .INMODEREG(0),
    .MASK(48'h3FFFFFFFFFFF),
    .MREG(0),
    .OPMODEREG(0),
    .PATTERN(48'h000000000000),
    .PREG(0),
    .SEL_MASK("MASK"),
    .SEL_PATTERN("PATTERN"),
    .USE_DPORT("FALSE"),
    .USE_MULT("MULTIPLY"),
    .USE_PATTERN_DETECT("NO_PATDET"),
    .USE_SIMD("ONE48")) 
    tmp_product__17
       (.A({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,tmp_product__19_n_0,tmp_product__20_n_0,tmp_product__21_n_0,tmp_product__22_n_0,tmp_product__23_n_0,tmp_product__24_n_0,tmp_product__25_n_0,tmp_product__26_n_0,tmp_product__27_n_0,tmp_product__28_n_0,tmp_product__29_n_0,tmp_product__30_n_0,tmp_product__31_n_0,tmp_product__32_n_0,tmp_product__33_n_0,tmp_product__34_n_0,tmp_product__35_n_0}),
        .ACIN({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .ACOUT({tmp_product__17_n_24,tmp_product__17_n_25,tmp_product__17_n_26,tmp_product__17_n_27,tmp_product__17_n_28,tmp_product__17_n_29,tmp_product__17_n_30,tmp_product__17_n_31,tmp_product__17_n_32,tmp_product__17_n_33,tmp_product__17_n_34,tmp_product__17_n_35,tmp_product__17_n_36,tmp_product__17_n_37,tmp_product__17_n_38,tmp_product__17_n_39,tmp_product__17_n_40,tmp_product__17_n_41,tmp_product__17_n_42,tmp_product__17_n_43,tmp_product__17_n_44,tmp_product__17_n_45,tmp_product__17_n_46,tmp_product__17_n_47,tmp_product__17_n_48,tmp_product__17_n_49,tmp_product__17_n_50,tmp_product__17_n_51,tmp_product__17_n_52,tmp_product__17_n_53}),
        .ALUMODE({1'b0,1'b0,1'b0,1'b0}),
        .B({1'b0,tmp_product__0_n_0,tmp_product__1_n_0,tmp_product__2_n_0,tmp_product__3_n_0,tmp_product__4_n_0,tmp_product__5_n_0,tmp_product__6_n_0,tmp_product__7_n_0,tmp_product__8_n_0,tmp_product__9_n_0,tmp_product__10_n_0,tmp_product__11_n_0,tmp_product__12_n_0,tmp_product__13_n_0,tmp_product__14_n_0,tmp_product__15_n_0,tmp_product__16_n_0}),
        .BCIN({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .BCOUT(NLW_tmp_product__17_BCOUT_UNCONNECTED[17:0]),
        .C({1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1,1'b1}),
        .CARRYCASCIN(1'b0),
        .CARRYCASCOUT(NLW_tmp_product__17_CARRYCASCOUT_UNCONNECTED),
        .CARRYIN(1'b0),
        .CARRYINSEL({1'b0,1'b0,1'b0}),
        .CARRYOUT(NLW_tmp_product__17_CARRYOUT_UNCONNECTED[3:0]),
        .CEA1(1'b0),
        .CEA2(Q),
        .CEAD(1'b0),
        .CEALUMODE(1'b0),
        .CEB1(1'b0),
        .CEB2(Q),
        .CEC(1'b0),
        .CECARRYIN(1'b0),
        .CECTRL(1'b0),
        .CED(1'b0),
        .CEINMODE(1'b0),
        .CEM(1'b0),
        .CEP(1'b0),
        .CLK(ap_clk),
        .D({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .INMODE({1'b0,1'b0,1'b0,1'b0,1'b0}),
        .MULTSIGNIN(1'b0),
        .MULTSIGNOUT(NLW_tmp_product__17_MULTSIGNOUT_UNCONNECTED),
        .OPMODE({1'b0,1'b0,1'b0,1'b0,1'b1,1'b0,1'b1}),
        .OVERFLOW(NLW_tmp_product__17_OVERFLOW_UNCONNECTED),
        .P({tmp_product__17_n_58,tmp_product__17_n_59,tmp_product__17_n_60,tmp_product__17_n_61,tmp_product__17_n_62,tmp_product__17_n_63,tmp_product__17_n_64,tmp_product__17_n_65,tmp_product__17_n_66,tmp_product__17_n_67,tmp_product__17_n_68,tmp_product__17_n_69,tmp_product__17_n_70,tmp_product__17_n_71,tmp_product__17_n_72,tmp_product__17_n_73,tmp_product__17_n_74,tmp_product__17_n_75,tmp_product__17_n_76,tmp_product__17_n_77,tmp_product__17_n_78,tmp_product__17_n_79,tmp_product__17_n_80,tmp_product__17_n_81,tmp_product__17_n_82,tmp_product__17_n_83,tmp_product__17_n_84,tmp_product__17_n_85,tmp_product__17_n_86,tmp_product__17_n_87,tmp_product__17_n_88,tmp_product__17_n_89,tmp_product__17_n_90,tmp_product__17_n_91,tmp_product__17_n_92,tmp_product__17_n_93,tmp_product__17_n_94,tmp_product__17_n_95,tmp_product__17_n_96,tmp_product__17_n_97,tmp_product__17_n_98,tmp_product__17_n_99,tmp_product__17_n_100,tmp_product__17_n_101,tmp_product__17_n_102,tmp_product__17_n_103,tmp_product__17_n_104,tmp_product__17_n_105}),
        .PATTERNBDETECT(NLW_tmp_product__17_PATTERNBDETECT_UNCONNECTED),
        .PATTERNDETECT(NLW_tmp_product__17_PATTERNDETECT_UNCONNECTED),
        .PCIN({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .PCOUT({tmp_product__17_n_106,tmp_product__17_n_107,tmp_product__17_n_108,tmp_product__17_n_109,tmp_product__17_n_110,tmp_product__17_n_111,tmp_product__17_n_112,tmp_product__17_n_113,tmp_product__17_n_114,tmp_product__17_n_115,tmp_product__17_n_116,tmp_product__17_n_117,tmp_product__17_n_118,tmp_product__17_n_119,tmp_product__17_n_120,tmp_product__17_n_121,tmp_product__17_n_122,tmp_product__17_n_123,tmp_product__17_n_124,tmp_product__17_n_125,tmp_product__17_n_126,tmp_product__17_n_127,tmp_product__17_n_128,tmp_product__17_n_129,tmp_product__17_n_130,tmp_product__17_n_131,tmp_product__17_n_132,tmp_product__17_n_133,tmp_product__17_n_134,tmp_product__17_n_135,tmp_product__17_n_136,tmp_product__17_n_137,tmp_product__17_n_138,tmp_product__17_n_139,tmp_product__17_n_140,tmp_product__17_n_141,tmp_product__17_n_142,tmp_product__17_n_143,tmp_product__17_n_144,tmp_product__17_n_145,tmp_product__17_n_146,tmp_product__17_n_147,tmp_product__17_n_148,tmp_product__17_n_149,tmp_product__17_n_150,tmp_product__17_n_151,tmp_product__17_n_152,tmp_product__17_n_153}),
        .RSTA(smax_reg_179),
        .RSTALLCARRYIN(1'b0),
        .RSTALUMODE(1'b0),
        .RSTB(smax1_reg_184),
        .RSTC(1'b0),
        .RSTCTRL(1'b0),
        .RSTD(1'b0),
        .RSTINMODE(1'b0),
        .RSTM(1'b0),
        .RSTP(1'b0),
        .UNDERFLOW(NLW_tmp_product__17_UNDERFLOW_UNCONNECTED));
  FDRE tmp_product__19
       (.C(ap_clk),
        .CE(E),
        .D(tmp_product__19_0[16]),
        .Q(tmp_product__19_n_0),
        .R(1'b0));
  FDRE tmp_product__2
       (.C(ap_clk),
        .CE(E),
        .D(D[14]),
        .Q(tmp_product__2_n_0),
        .R(1'b0));
  FDRE tmp_product__20
       (.C(ap_clk),
        .CE(E),
        .D(tmp_product__19_0[15]),
        .Q(tmp_product__20_n_0),
        .R(1'b0));
  FDRE tmp_product__21
       (.C(ap_clk),
        .CE(E),
        .D(tmp_product__19_0[14]),
        .Q(tmp_product__21_n_0),
        .R(1'b0));
  FDRE tmp_product__22
       (.C(ap_clk),
        .CE(E),
        .D(tmp_product__19_0[13]),
        .Q(tmp_product__22_n_0),
        .R(1'b0));
  FDRE tmp_product__23
       (.C(ap_clk),
        .CE(E),
        .D(tmp_product__19_0[12]),
        .Q(tmp_product__23_n_0),
        .R(1'b0));
  FDRE tmp_product__24
       (.C(ap_clk),
        .CE(E),
        .D(tmp_product__19_0[11]),
        .Q(tmp_product__24_n_0),
        .R(1'b0));
  FDRE tmp_product__25
       (.C(ap_clk),
        .CE(E),
        .D(tmp_product__19_0[10]),
        .Q(tmp_product__25_n_0),
        .R(1'b0));
  FDRE tmp_product__26
       (.C(ap_clk),
        .CE(E),
        .D(tmp_product__19_0[9]),
        .Q(tmp_product__26_n_0),
        .R(1'b0));
  FDRE tmp_product__27
       (.C(ap_clk),
        .CE(E),
        .D(tmp_product__19_0[8]),
        .Q(tmp_product__27_n_0),
        .R(1'b0));
  FDRE tmp_product__28
       (.C(ap_clk),
        .CE(E),
        .D(tmp_product__19_0[7]),
        .Q(tmp_product__28_n_0),
        .R(1'b0));
  FDRE tmp_product__29
       (.C(ap_clk),
        .CE(E),
        .D(tmp_product__19_0[6]),
        .Q(tmp_product__29_n_0),
        .R(1'b0));
  FDRE tmp_product__3
       (.C(ap_clk),
        .CE(E),
        .D(D[13]),
        .Q(tmp_product__3_n_0),
        .R(1'b0));
  FDRE tmp_product__30
       (.C(ap_clk),
        .CE(E),
        .D(tmp_product__19_0[5]),
        .Q(tmp_product__30_n_0),
        .R(1'b0));
  FDRE tmp_product__31
       (.C(ap_clk),
        .CE(E),
        .D(tmp_product__19_0[4]),
        .Q(tmp_product__31_n_0),
        .R(1'b0));
  FDRE tmp_product__32
       (.C(ap_clk),
        .CE(E),
        .D(tmp_product__19_0[3]),
        .Q(tmp_product__32_n_0),
        .R(1'b0));
  FDRE tmp_product__33
       (.C(ap_clk),
        .CE(E),
        .D(tmp_product__19_0[2]),
        .Q(tmp_product__33_n_0),
        .R(1'b0));
  FDRE tmp_product__34
       (.C(ap_clk),
        .CE(E),
        .D(tmp_product__19_0[1]),
        .Q(tmp_product__34_n_0),
        .R(1'b0));
  FDRE tmp_product__35
       (.C(ap_clk),
        .CE(E),
        .D(tmp_product__19_0[0]),
        .Q(tmp_product__35_n_0),
        .R(1'b0));
  FDRE tmp_product__4
       (.C(ap_clk),
        .CE(E),
        .D(D[12]),
        .Q(tmp_product__4_n_0),
        .R(1'b0));
  FDRE tmp_product__5
       (.C(ap_clk),
        .CE(E),
        .D(D[11]),
        .Q(tmp_product__5_n_0),
        .R(1'b0));
  FDRE tmp_product__6
       (.C(ap_clk),
        .CE(E),
        .D(D[10]),
        .Q(tmp_product__6_n_0),
        .R(1'b0));
  FDRE tmp_product__7
       (.C(ap_clk),
        .CE(E),
        .D(D[9]),
        .Q(tmp_product__7_n_0),
        .R(1'b0));
  FDRE tmp_product__8
       (.C(ap_clk),
        .CE(E),
        .D(D[8]),
        .Q(tmp_product__8_n_0),
        .R(1'b0));
  FDRE tmp_product__9
       (.C(ap_clk),
        .CE(E),
        .D(D[7]),
        .Q(tmp_product__9_n_0),
        .R(1'b0));
  LUT2 #(
    .INIT(4'h2)) 
    tmp_product_i_1
       (.I0(Q),
        .I1(p_0_in),
        .O(smax1_reg_184));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_10
       (.I0(tmp_product_i_3_0[30]),
        .I1(tmp_product_i_3_0[31]),
        .O(tmp_product_i_10_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_11
       (.I0(tmp_product_i_3_0[28]),
        .I1(tmp_product_i_3_0[29]),
        .O(tmp_product_i_11_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_12
       (.I0(tmp_product_i_3_0[26]),
        .I1(tmp_product_i_3_0[27]),
        .O(tmp_product_i_12_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_13
       (.I0(tmp_product_i_3_0[24]),
        .I1(tmp_product_i_3_0[25]),
        .O(tmp_product_i_13_n_0));
  (* COMPARATOR_THRESHOLD = "11" *) 
  CARRY4 tmp_product_i_14
       (.CI(tmp_product_i_32_n_0),
        .CO({tmp_product_i_14_n_0,tmp_product_i_14_n_1,tmp_product_i_14_n_2,tmp_product_i_14_n_3}),
        .CYINIT(1'b0),
        .DI({tmp_product_i_33_n_0,tmp_product_i_34_n_0,tmp_product_i_35_n_0,tmp_product_i_36_n_0}),
        .O(NLW_tmp_product_i_14_O_UNCONNECTED[3:0]),
        .S({tmp_product_i_37_n_0,tmp_product_i_38_n_0,tmp_product_i_39_n_0,tmp_product_i_40_n_0}));
  LUT2 #(
    .INIT(4'h2)) 
    tmp_product_i_15
       (.I0(tmp_product_i_4_0[30]),
        .I1(tmp_product_i_4_0[31]),
        .O(tmp_product_i_15_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_16
       (.I0(tmp_product_i_4_0[28]),
        .I1(tmp_product_i_4_0[29]),
        .O(tmp_product_i_16_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_17
       (.I0(tmp_product_i_4_0[26]),
        .I1(tmp_product_i_4_0[27]),
        .O(tmp_product_i_17_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_18
       (.I0(tmp_product_i_4_0[24]),
        .I1(tmp_product_i_4_0[25]),
        .O(tmp_product_i_18_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_19
       (.I0(tmp_product_i_4_0[30]),
        .I1(tmp_product_i_4_0[31]),
        .O(tmp_product_i_19_n_0));
  LUT2 #(
    .INIT(4'h2)) 
    tmp_product_i_2
       (.I0(Q),
        .I1(tmp_product_i_4_n_0),
        .O(smax_reg_179));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_20
       (.I0(tmp_product_i_4_0[28]),
        .I1(tmp_product_i_4_0[29]),
        .O(tmp_product_i_20_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_21
       (.I0(tmp_product_i_4_0[26]),
        .I1(tmp_product_i_4_0[27]),
        .O(tmp_product_i_21_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_22
       (.I0(tmp_product_i_4_0[24]),
        .I1(tmp_product_i_4_0[25]),
        .O(tmp_product_i_22_n_0));
  (* COMPARATOR_THRESHOLD = "11" *) 
  CARRY4 tmp_product_i_23
       (.CI(tmp_product_i_41_n_0),
        .CO({tmp_product_i_23_n_0,tmp_product_i_23_n_1,tmp_product_i_23_n_2,tmp_product_i_23_n_3}),
        .CYINIT(1'b0),
        .DI({tmp_product_i_42_n_0,tmp_product_i_43_n_0,tmp_product_i_44_n_0,tmp_product_i_45_n_0}),
        .O(NLW_tmp_product_i_23_O_UNCONNECTED[3:0]),
        .S({tmp_product_i_46_n_0,tmp_product_i_47_n_0,tmp_product_i_48_n_0,tmp_product_i_49_n_0}));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_24
       (.I0(tmp_product_i_3_0[22]),
        .I1(tmp_product_i_3_0[23]),
        .O(tmp_product_i_24_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_25
       (.I0(tmp_product_i_3_0[20]),
        .I1(tmp_product_i_3_0[21]),
        .O(tmp_product_i_25_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_26
       (.I0(tmp_product_i_3_0[18]),
        .I1(tmp_product_i_3_0[19]),
        .O(tmp_product_i_26_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_27
       (.I0(tmp_product_i_3_0[16]),
        .I1(tmp_product_i_3_0[17]),
        .O(tmp_product_i_27_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_28
       (.I0(tmp_product_i_3_0[22]),
        .I1(tmp_product_i_3_0[23]),
        .O(tmp_product_i_28_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_29
       (.I0(tmp_product_i_3_0[20]),
        .I1(tmp_product_i_3_0[21]),
        .O(tmp_product_i_29_n_0));
  (* COMPARATOR_THRESHOLD = "11" *) 
  CARRY4 tmp_product_i_3
       (.CI(tmp_product_i_5_n_0),
        .CO({p_0_in,tmp_product_i_3_n_1,tmp_product_i_3_n_2,tmp_product_i_3_n_3}),
        .CYINIT(1'b0),
        .DI({tmp_product_i_6_n_0,tmp_product_i_7_n_0,tmp_product_i_8_n_0,tmp_product_i_9_n_0}),
        .O(NLW_tmp_product_i_3_O_UNCONNECTED[3:0]),
        .S({tmp_product_i_10_n_0,tmp_product_i_11_n_0,tmp_product_i_12_n_0,tmp_product_i_13_n_0}));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_30
       (.I0(tmp_product_i_3_0[18]),
        .I1(tmp_product_i_3_0[19]),
        .O(tmp_product_i_30_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_31
       (.I0(tmp_product_i_3_0[16]),
        .I1(tmp_product_i_3_0[17]),
        .O(tmp_product_i_31_n_0));
  (* COMPARATOR_THRESHOLD = "11" *) 
  CARRY4 tmp_product_i_32
       (.CI(tmp_product_i_50_n_0),
        .CO({tmp_product_i_32_n_0,tmp_product_i_32_n_1,tmp_product_i_32_n_2,tmp_product_i_32_n_3}),
        .CYINIT(1'b0),
        .DI({tmp_product_i_51_n_0,tmp_product_i_52_n_0,tmp_product_i_53_n_0,tmp_product_i_54_n_0}),
        .O(NLW_tmp_product_i_32_O_UNCONNECTED[3:0]),
        .S({tmp_product_i_55_n_0,tmp_product_i_56_n_0,tmp_product_i_57_n_0,tmp_product_i_58_n_0}));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_33
       (.I0(tmp_product_i_4_0[22]),
        .I1(tmp_product_i_4_0[23]),
        .O(tmp_product_i_33_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_34
       (.I0(tmp_product_i_4_0[20]),
        .I1(tmp_product_i_4_0[21]),
        .O(tmp_product_i_34_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_35
       (.I0(tmp_product_i_4_0[18]),
        .I1(tmp_product_i_4_0[19]),
        .O(tmp_product_i_35_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_36
       (.I0(tmp_product_i_4_0[16]),
        .I1(tmp_product_i_4_0[17]),
        .O(tmp_product_i_36_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_37
       (.I0(tmp_product_i_4_0[22]),
        .I1(tmp_product_i_4_0[23]),
        .O(tmp_product_i_37_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_38
       (.I0(tmp_product_i_4_0[20]),
        .I1(tmp_product_i_4_0[21]),
        .O(tmp_product_i_38_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_39
       (.I0(tmp_product_i_4_0[18]),
        .I1(tmp_product_i_4_0[19]),
        .O(tmp_product_i_39_n_0));
  (* COMPARATOR_THRESHOLD = "11" *) 
  CARRY4 tmp_product_i_4
       (.CI(tmp_product_i_14_n_0),
        .CO({tmp_product_i_4_n_0,tmp_product_i_4_n_1,tmp_product_i_4_n_2,tmp_product_i_4_n_3}),
        .CYINIT(1'b0),
        .DI({tmp_product_i_15_n_0,tmp_product_i_16_n_0,tmp_product_i_17_n_0,tmp_product_i_18_n_0}),
        .O(NLW_tmp_product_i_4_O_UNCONNECTED[3:0]),
        .S({tmp_product_i_19_n_0,tmp_product_i_20_n_0,tmp_product_i_21_n_0,tmp_product_i_22_n_0}));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_40
       (.I0(tmp_product_i_4_0[16]),
        .I1(tmp_product_i_4_0[17]),
        .O(tmp_product_i_40_n_0));
  (* COMPARATOR_THRESHOLD = "11" *) 
  CARRY4 tmp_product_i_41
       (.CI(1'b0),
        .CO({tmp_product_i_41_n_0,tmp_product_i_41_n_1,tmp_product_i_41_n_2,tmp_product_i_41_n_3}),
        .CYINIT(1'b0),
        .DI({tmp_product_i_59_n_0,tmp_product_i_60_n_0,tmp_product_i_61_n_0,tmp_product_i_62_n_0}),
        .O(NLW_tmp_product_i_41_O_UNCONNECTED[3:0]),
        .S({tmp_product_i_63_n_0,tmp_product_i_64_n_0,tmp_product_i_65_n_0,tmp_product_i_66_n_0}));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_42
       (.I0(tmp_product_i_3_0[14]),
        .I1(tmp_product_i_3_0[15]),
        .O(tmp_product_i_42_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_43
       (.I0(tmp_product_i_3_0[12]),
        .I1(tmp_product_i_3_0[13]),
        .O(tmp_product_i_43_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_44
       (.I0(tmp_product_i_3_0[10]),
        .I1(tmp_product_i_3_0[11]),
        .O(tmp_product_i_44_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_45
       (.I0(tmp_product_i_3_0[8]),
        .I1(tmp_product_i_3_0[9]),
        .O(tmp_product_i_45_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_46
       (.I0(tmp_product_i_3_0[14]),
        .I1(tmp_product_i_3_0[15]),
        .O(tmp_product_i_46_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_47
       (.I0(tmp_product_i_3_0[12]),
        .I1(tmp_product_i_3_0[13]),
        .O(tmp_product_i_47_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_48
       (.I0(tmp_product_i_3_0[10]),
        .I1(tmp_product_i_3_0[11]),
        .O(tmp_product_i_48_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_49
       (.I0(tmp_product_i_3_0[8]),
        .I1(tmp_product_i_3_0[9]),
        .O(tmp_product_i_49_n_0));
  (* COMPARATOR_THRESHOLD = "11" *) 
  CARRY4 tmp_product_i_5
       (.CI(tmp_product_i_23_n_0),
        .CO({tmp_product_i_5_n_0,tmp_product_i_5_n_1,tmp_product_i_5_n_2,tmp_product_i_5_n_3}),
        .CYINIT(1'b0),
        .DI({tmp_product_i_24_n_0,tmp_product_i_25_n_0,tmp_product_i_26_n_0,tmp_product_i_27_n_0}),
        .O(NLW_tmp_product_i_5_O_UNCONNECTED[3:0]),
        .S({tmp_product_i_28_n_0,tmp_product_i_29_n_0,tmp_product_i_30_n_0,tmp_product_i_31_n_0}));
  (* COMPARATOR_THRESHOLD = "11" *) 
  CARRY4 tmp_product_i_50
       (.CI(1'b0),
        .CO({tmp_product_i_50_n_0,tmp_product_i_50_n_1,tmp_product_i_50_n_2,tmp_product_i_50_n_3}),
        .CYINIT(1'b0),
        .DI({tmp_product_i_67_n_0,tmp_product_i_68_n_0,tmp_product_i_69_n_0,tmp_product_i_70_n_0}),
        .O(NLW_tmp_product_i_50_O_UNCONNECTED[3:0]),
        .S({tmp_product_i_71_n_0,tmp_product_i_72_n_0,tmp_product_i_73_n_0,tmp_product_i_74_n_0}));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_51
       (.I0(tmp_product_i_4_0[14]),
        .I1(tmp_product_i_4_0[15]),
        .O(tmp_product_i_51_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_52
       (.I0(tmp_product_i_4_0[12]),
        .I1(tmp_product_i_4_0[13]),
        .O(tmp_product_i_52_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_53
       (.I0(tmp_product_i_4_0[10]),
        .I1(tmp_product_i_4_0[11]),
        .O(tmp_product_i_53_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_54
       (.I0(tmp_product_i_4_0[8]),
        .I1(tmp_product_i_4_0[9]),
        .O(tmp_product_i_54_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_55
       (.I0(tmp_product_i_4_0[14]),
        .I1(tmp_product_i_4_0[15]),
        .O(tmp_product_i_55_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_56
       (.I0(tmp_product_i_4_0[12]),
        .I1(tmp_product_i_4_0[13]),
        .O(tmp_product_i_56_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_57
       (.I0(tmp_product_i_4_0[10]),
        .I1(tmp_product_i_4_0[11]),
        .O(tmp_product_i_57_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_58
       (.I0(tmp_product_i_4_0[8]),
        .I1(tmp_product_i_4_0[9]),
        .O(tmp_product_i_58_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_59
       (.I0(tmp_product_i_3_0[6]),
        .I1(tmp_product_i_3_0[7]),
        .O(tmp_product_i_59_n_0));
  LUT2 #(
    .INIT(4'h2)) 
    tmp_product_i_6
       (.I0(tmp_product_i_3_0[30]),
        .I1(tmp_product_i_3_0[31]),
        .O(tmp_product_i_6_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_60
       (.I0(tmp_product_i_3_0[4]),
        .I1(tmp_product_i_3_0[5]),
        .O(tmp_product_i_60_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_61
       (.I0(tmp_product_i_3_0[2]),
        .I1(tmp_product_i_3_0[3]),
        .O(tmp_product_i_61_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_62
       (.I0(tmp_product_i_3_0[0]),
        .I1(tmp_product_i_3_0[1]),
        .O(tmp_product_i_62_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_63
       (.I0(tmp_product_i_3_0[6]),
        .I1(tmp_product_i_3_0[7]),
        .O(tmp_product_i_63_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_64
       (.I0(tmp_product_i_3_0[4]),
        .I1(tmp_product_i_3_0[5]),
        .O(tmp_product_i_64_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_65
       (.I0(tmp_product_i_3_0[2]),
        .I1(tmp_product_i_3_0[3]),
        .O(tmp_product_i_65_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_66
       (.I0(tmp_product_i_3_0[0]),
        .I1(tmp_product_i_3_0[1]),
        .O(tmp_product_i_66_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_67
       (.I0(tmp_product_i_4_0[6]),
        .I1(tmp_product_i_4_0[7]),
        .O(tmp_product_i_67_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_68
       (.I0(tmp_product_i_4_0[4]),
        .I1(tmp_product_i_4_0[5]),
        .O(tmp_product_i_68_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_69
       (.I0(tmp_product_i_4_0[2]),
        .I1(tmp_product_i_4_0[3]),
        .O(tmp_product_i_69_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_7
       (.I0(tmp_product_i_3_0[28]),
        .I1(tmp_product_i_3_0[29]),
        .O(tmp_product_i_7_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_70
       (.I0(tmp_product_i_4_0[0]),
        .I1(tmp_product_i_4_0[1]),
        .O(tmp_product_i_70_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_71
       (.I0(tmp_product_i_4_0[6]),
        .I1(tmp_product_i_4_0[7]),
        .O(tmp_product_i_71_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_72
       (.I0(tmp_product_i_4_0[4]),
        .I1(tmp_product_i_4_0[5]),
        .O(tmp_product_i_72_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_73
       (.I0(tmp_product_i_4_0[2]),
        .I1(tmp_product_i_4_0[3]),
        .O(tmp_product_i_73_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    tmp_product_i_74
       (.I0(tmp_product_i_4_0[0]),
        .I1(tmp_product_i_4_0[1]),
        .O(tmp_product_i_74_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_8
       (.I0(tmp_product_i_3_0[26]),
        .I1(tmp_product_i_3_0[27]),
        .O(tmp_product_i_8_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    tmp_product_i_9
       (.I0(tmp_product_i_3_0[24]),
        .I1(tmp_product_i_3_0[25]),
        .O(tmp_product_i_9_n_0));
endmodule

module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_regslice_both
   (ack_in_t_reg_0,
    Q,
    ap_rst_n_inv,
    ap_clk,
    input_stream_TREADY_int_regslice,
    input_stream_TVALID,
    \state_reg[1]_0 ,
    output_stream_TREADY_int_regslice,
    \state_reg[1]_1 );
  output ack_in_t_reg_0;
  output [0:0]Q;
  input ap_rst_n_inv;
  input ap_clk;
  input input_stream_TREADY_int_regslice;
  input input_stream_TVALID;
  input [0:0]\state_reg[1]_0 ;
  input output_stream_TREADY_int_regslice;
  input \state_reg[1]_1 ;

  wire \FSM_sequential_state[0]_i_1__5_n_0 ;
  wire \FSM_sequential_state[1]_i_1_n_0 ;
  wire [0:0]Q;
  wire ack_in_t_i_2_n_0;
  wire ack_in_t_reg_0;
  wire ap_clk;
  wire ap_rst_n_inv;
  wire input_stream_TREADY_int_regslice;
  wire input_stream_TVALID;
  wire output_stream_TREADY_int_regslice;
  wire [1:1]state;
  wire \state[0]_i_1_n_0 ;
  wire \state[1]_i_1__0_n_0 ;
  wire [1:0]state__0;
  wire [0:0]\state_reg[1]_0 ;
  wire \state_reg[1]_1 ;

  LUT4 #(
    .INIT(16'hAEFF)) 
    \FSM_sequential_state[0]_i_1__5 
       (.I0(input_stream_TREADY_int_regslice),
        .I1(state__0[0]),
        .I2(input_stream_TVALID),
        .I3(state__0[1]),
        .O(\FSM_sequential_state[0]_i_1__5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair56" *) 
  LUT5 #(
    .INIT(32'hFCCC4C4C)) 
    \FSM_sequential_state[1]_i_1 
       (.I0(input_stream_TREADY_int_regslice),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(ack_in_t_reg_0),
        .I4(input_stream_TVALID),
        .O(\FSM_sequential_state[1]_i_1_n_0 ));
  (* FSM_ENCODED_STATES = "ZERO:01,TWO:10,ONE:11,iSTATE:00" *) 
  FDSE #(
    .INIT(1'b0)) 
    \FSM_sequential_state_reg[0] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\FSM_sequential_state[0]_i_1__5_n_0 ),
        .Q(state__0[0]),
        .S(ap_rst_n_inv));
  (* FSM_ENCODED_STATES = "ZERO:01,TWO:10,ONE:11,iSTATE:00" *) 
  FDRE #(
    .INIT(1'b0)) 
    \FSM_sequential_state_reg[1] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\FSM_sequential_state[1]_i_1_n_0 ),
        .Q(state__0[1]),
        .R(ap_rst_n_inv));
  (* SOFT_HLUTNM = "soft_lutpair56" *) 
  LUT5 #(
    .INIT(32'hB0FFFAF0)) 
    ack_in_t_i_2
       (.I0(input_stream_TREADY_int_regslice),
        .I1(input_stream_TVALID),
        .I2(ack_in_t_reg_0),
        .I3(state__0[1]),
        .I4(state__0[0]),
        .O(ack_in_t_i_2_n_0));
  FDRE #(
    .INIT(1'b0)) 
    ack_in_t_reg
       (.C(ap_clk),
        .CE(1'b1),
        .D(ack_in_t_i_2_n_0),
        .Q(ack_in_t_reg_0),
        .R(ap_rst_n_inv));
  LUT5 #(
    .INIT(32'hFCCC4C4C)) 
    \state[0]_i_1 
       (.I0(input_stream_TREADY_int_regslice),
        .I1(Q),
        .I2(state),
        .I3(ack_in_t_reg_0),
        .I4(input_stream_TVALID),
        .O(\state[0]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h8080FF80FFFFFFFF)) 
    \state[1]_i_1__0 
       (.I0(\state_reg[1]_0 ),
        .I1(output_stream_TREADY_int_regslice),
        .I2(\state_reg[1]_1 ),
        .I3(state),
        .I4(input_stream_TVALID),
        .I5(Q),
        .O(\state[1]_i_1__0_n_0 ));
  FDRE #(
    .INIT(1'b0)) 
    \state_reg[0] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\state[0]_i_1_n_0 ),
        .Q(Q),
        .R(ap_rst_n_inv));
  FDSE #(
    .INIT(1'b0)) 
    \state_reg[1] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\state[1]_i_1__0_n_0 ),
        .Q(state),
        .S(ap_rst_n_inv));
endmodule

(* ORIG_REF_NAME = "filter_kernel_regslice_both" *) 
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_regslice_both_1
   (output_stream_TREADY_int_regslice,
    E,
    output_stream_TVALID,
    grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID,
    D,
    \ap_CS_fsm_reg[6] ,
    output_stream_TDATA,
    ap_rst_n_inv,
    ap_clk,
    Q,
    \data_p2_reg[1]_0 ,
    \data_p2_reg[1]_1 ,
    output_stream_TREADY,
    ap_start);
  output output_stream_TREADY_int_regslice;
  output [0:0]E;
  output output_stream_TVALID;
  output grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID;
  output [0:0]D;
  output \ap_CS_fsm_reg[6] ;
  output [1:0]output_stream_TDATA;
  input ap_rst_n_inv;
  input ap_clk;
  input [2:0]Q;
  input [0:0]\data_p2_reg[1]_0 ;
  input \data_p2_reg[1]_1 ;
  input output_stream_TREADY;
  input ap_start;

  wire [0:0]D;
  wire [0:0]E;
  wire \FSM_sequential_state[0]_i_1__2_n_0 ;
  wire \FSM_sequential_state[1]_i_1__5_n_0 ;
  wire [2:0]Q;
  wire ack_in_t_i_1__2_n_0;
  wire \ap_CS_fsm_reg[6] ;
  wire ap_clk;
  wire ap_rst_n_inv;
  wire ap_start;
  wire \data_p1[1]_i_1_n_0 ;
  wire \data_p1[3]_i_1_n_0 ;
  wire \data_p2[1]_i_1_n_0 ;
  wire \data_p2[3]_i_1_n_0 ;
  wire [0:0]\data_p2_reg[1]_0 ;
  wire \data_p2_reg[1]_1 ;
  wire \data_p2_reg_n_0_[1] ;
  wire \data_p2_reg_n_0_[3] ;
  wire grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID;
  wire [1:0]output_stream_TDATA;
  wire output_stream_TREADY;
  wire output_stream_TREADY_int_regslice;
  wire output_stream_TVALID;
  wire [1:1]state;
  wire \state[0]_i_1__0_n_0 ;
  wire \state[1]_i_1_n_0 ;
  wire [1:0]state__0;

  (* SOFT_HLUTNM = "soft_lutpair62" *) 
  LUT4 #(
    .INIT(16'hAEFF)) 
    \FSM_sequential_state[0]_i_1__2 
       (.I0(output_stream_TREADY),
        .I1(state__0[0]),
        .I2(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .I3(state__0[1]),
        .O(\FSM_sequential_state[0]_i_1__2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair59" *) 
  LUT5 #(
    .INIT(32'hFCCC4C4C)) 
    \FSM_sequential_state[1]_i_1__5 
       (.I0(output_stream_TREADY),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(output_stream_TREADY_int_regslice),
        .I4(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .O(\FSM_sequential_state[1]_i_1__5_n_0 ));
  (* FSM_ENCODED_STATES = "ZERO:01,TWO:10,ONE:11,iSTATE:00" *) 
  FDSE #(
    .INIT(1'b0)) 
    \FSM_sequential_state_reg[0] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\FSM_sequential_state[0]_i_1__2_n_0 ),
        .Q(state__0[0]),
        .S(ap_rst_n_inv));
  (* FSM_ENCODED_STATES = "ZERO:01,TWO:10,ONE:11,iSTATE:00" *) 
  FDRE #(
    .INIT(1'b0)) 
    \FSM_sequential_state_reg[1] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\FSM_sequential_state[1]_i_1__5_n_0 ),
        .Q(state__0[1]),
        .R(ap_rst_n_inv));
  (* SOFT_HLUTNM = "soft_lutpair59" *) 
  LUT5 #(
    .INIT(32'hB0FFFAF0)) 
    ack_in_t_i_1__2
       (.I0(output_stream_TREADY),
        .I1(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .I2(output_stream_TREADY_int_regslice),
        .I3(state__0[1]),
        .I4(state__0[0]),
        .O(ack_in_t_i_1__2_n_0));
  FDRE #(
    .INIT(1'b0)) 
    ack_in_t_reg
       (.C(ap_clk),
        .CE(1'b1),
        .D(ack_in_t_i_1__2_n_0),
        .Q(output_stream_TREADY_int_regslice),
        .R(ap_rst_n_inv));
  LUT6 #(
    .INIT(64'hF2F222F222F222F2)) 
    \ap_CS_fsm[0]_i_1 
       (.I0(Q[0]),
        .I1(ap_start),
        .I2(Q[2]),
        .I3(state__0[1]),
        .I4(state__0[0]),
        .I5(output_stream_TREADY),
        .O(D));
  (* SOFT_HLUTNM = "soft_lutpair62" *) 
  LUT4 #(
    .INIT(16'hA222)) 
    \ap_CS_fsm[6]_i_3 
       (.I0(Q[2]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(output_stream_TREADY),
        .O(\ap_CS_fsm_reg[6] ));
  LUT6 #(
    .INIT(64'hFBFBFFFFF8083000)) 
    \data_p1[1]_i_1 
       (.I0(\data_p2_reg_n_0_[1] ),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .I4(output_stream_TREADY),
        .I5(output_stream_TDATA[0]),
        .O(\data_p1[1]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFBFBFFFFF8083000)) 
    \data_p1[3]_i_1 
       (.I0(\data_p2_reg_n_0_[3] ),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .I4(output_stream_TREADY),
        .I5(output_stream_TDATA[1]),
        .O(\data_p1[3]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair60" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \data_p1[3]_i_2__1 
       (.I0(Q[1]),
        .I1(output_stream_TREADY_int_regslice),
        .I2(\data_p2_reg[1]_0 ),
        .I3(\data_p2_reg[1]_1 ),
        .O(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID));
  FDRE #(
    .INIT(1'b0)) 
    \data_p1_reg[1] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\data_p1[1]_i_1_n_0 ),
        .Q(output_stream_TDATA[0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p1_reg[3] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\data_p1[3]_i_1_n_0 ),
        .Q(output_stream_TDATA[1]),
        .R(1'b0));
  (* SOFT_HLUTNM = "soft_lutpair60" *) 
  LUT5 #(
    .INIT(32'hEAAAAAAA)) 
    \data_p2[1]_i_1 
       (.I0(\data_p2_reg_n_0_[1] ),
        .I1(\data_p2_reg[1]_1 ),
        .I2(\data_p2_reg[1]_0 ),
        .I3(Q[1]),
        .I4(output_stream_TREADY_int_regslice),
        .O(\data_p2[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair61" *) 
  LUT5 #(
    .INIT(32'hEAAAAAAA)) 
    \data_p2[3]_i_1 
       (.I0(\data_p2_reg_n_0_[3] ),
        .I1(\data_p2_reg[1]_1 ),
        .I2(\data_p2_reg[1]_0 ),
        .I3(Q[1]),
        .I4(output_stream_TREADY_int_regslice),
        .O(\data_p2[3]_i_1_n_0 ));
  FDRE #(
    .INIT(1'b0)) 
    \data_p2_reg[1] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\data_p2[1]_i_1_n_0 ),
        .Q(\data_p2_reg_n_0_[1] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p2_reg[3] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\data_p2[3]_i_1_n_0 ),
        .Q(\data_p2_reg_n_0_[3] ),
        .R(1'b0));
  (* SOFT_HLUTNM = "soft_lutpair61" *) 
  LUT4 #(
    .INIT(16'h80FF)) 
    \icmp_ln44_reg_312[0]_i_1 
       (.I0(Q[1]),
        .I1(output_stream_TREADY_int_regslice),
        .I2(\data_p2_reg[1]_0 ),
        .I3(\data_p2_reg[1]_1 ),
        .O(E));
  LUT5 #(
    .INIT(32'hFCCC4C4C)) 
    \state[0]_i_1__0 
       (.I0(output_stream_TREADY),
        .I1(output_stream_TVALID),
        .I2(state),
        .I3(output_stream_TREADY_int_regslice),
        .I4(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .O(\state[0]_i_1__0_n_0 ));
  LUT4 #(
    .INIT(16'hAEFF)) 
    \state[1]_i_1 
       (.I0(output_stream_TREADY),
        .I1(state),
        .I2(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .I3(output_stream_TVALID),
        .O(\state[1]_i_1_n_0 ));
  FDRE #(
    .INIT(1'b0)) 
    \state_reg[0] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\state[0]_i_1__0_n_0 ),
        .Q(output_stream_TVALID),
        .R(ap_rst_n_inv));
  FDSE #(
    .INIT(1'b0)) 
    \state_reg[1] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\state[1]_i_1_n_0 ),
        .Q(state),
        .S(ap_rst_n_inv));
endmodule

(* ORIG_REF_NAME = "filter_kernel_regslice_both" *) 
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_regslice_both__parameterized0
   (Q,
    ap_rst_n_inv,
    ap_clk,
    input_stream_TVALID,
    input_stream_TREADY_int_regslice,
    input_stream_TKEEP);
  output [3:0]Q;
  input ap_rst_n_inv;
  input ap_clk;
  input input_stream_TVALID;
  input input_stream_TREADY_int_regslice;
  input [3:0]input_stream_TKEEP;

  wire [3:0]Q;
  wire ack_in_t_i_1__0_n_0;
  wire ack_in_t_reg_n_0;
  wire ap_clk;
  wire ap_rst_n_inv;
  wire \data_p1[0]_i_1__1_n_0 ;
  wire \data_p1[1]_i_1__1_n_0 ;
  wire \data_p1[2]_i_1__1_n_0 ;
  wire \data_p1[3]_i_2__2_n_0 ;
  wire [3:0]data_p2;
  wire [3:0]input_stream_TKEEP;
  wire input_stream_TREADY_int_regslice;
  wire input_stream_TVALID;
  wire load_p1;
  wire load_p2;
  wire [1:0]next__0;
  wire [1:0]state__0;

  LUT4 #(
    .INIT(16'hAEFF)) 
    \FSM_sequential_state[0]_i_1__3 
       (.I0(input_stream_TREADY_int_regslice),
        .I1(state__0[0]),
        .I2(input_stream_TVALID),
        .I3(state__0[1]),
        .O(next__0[0]));
  (* SOFT_HLUTNM = "soft_lutpair57" *) 
  LUT5 #(
    .INIT(32'hFCCC4C4C)) 
    \FSM_sequential_state[1]_i_1__1 
       (.I0(input_stream_TREADY_int_regslice),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(ack_in_t_reg_n_0),
        .I4(input_stream_TVALID),
        .O(next__0[1]));
  (* FSM_ENCODED_STATES = "ZERO:01,TWO:10,ONE:11,iSTATE:00" *) 
  FDSE #(
    .INIT(1'b0)) 
    \FSM_sequential_state_reg[0] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(next__0[0]),
        .Q(state__0[0]),
        .S(ap_rst_n_inv));
  (* FSM_ENCODED_STATES = "ZERO:01,TWO:10,ONE:11,iSTATE:00" *) 
  FDRE #(
    .INIT(1'b0)) 
    \FSM_sequential_state_reg[1] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(next__0[1]),
        .Q(state__0[1]),
        .R(ap_rst_n_inv));
  (* SOFT_HLUTNM = "soft_lutpair57" *) 
  LUT5 #(
    .INIT(32'hB0FFFAF0)) 
    ack_in_t_i_1__0
       (.I0(input_stream_TREADY_int_regslice),
        .I1(input_stream_TVALID),
        .I2(ack_in_t_reg_n_0),
        .I3(state__0[1]),
        .I4(state__0[0]),
        .O(ack_in_t_i_1__0_n_0));
  FDRE #(
    .INIT(1'b0)) 
    ack_in_t_reg
       (.C(ap_clk),
        .CE(1'b1),
        .D(ack_in_t_i_1__0_n_0),
        .Q(ack_in_t_reg_n_0),
        .R(ap_rst_n_inv));
  LUT4 #(
    .INIT(16'hFB08)) 
    \data_p1[0]_i_1__1 
       (.I0(data_p2[0]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(input_stream_TKEEP[0]),
        .O(\data_p1[0]_i_1__1_n_0 ));
  LUT4 #(
    .INIT(16'hFB08)) 
    \data_p1[1]_i_1__1 
       (.I0(data_p2[1]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(input_stream_TKEEP[1]),
        .O(\data_p1[1]_i_1__1_n_0 ));
  LUT4 #(
    .INIT(16'hFB08)) 
    \data_p1[2]_i_1__1 
       (.I0(data_p2[2]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(input_stream_TKEEP[2]),
        .O(\data_p1[2]_i_1__1_n_0 ));
  LUT4 #(
    .INIT(16'hE240)) 
    \data_p1[3]_i_1__0 
       (.I0(state__0[1]),
        .I1(state__0[0]),
        .I2(input_stream_TVALID),
        .I3(input_stream_TREADY_int_regslice),
        .O(load_p1));
  LUT4 #(
    .INIT(16'hFB08)) 
    \data_p1[3]_i_2__2 
       (.I0(data_p2[3]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(input_stream_TKEEP[3]),
        .O(\data_p1[3]_i_2__2_n_0 ));
  FDRE #(
    .INIT(1'b0)) 
    \data_p1_reg[0] 
       (.C(ap_clk),
        .CE(load_p1),
        .D(\data_p1[0]_i_1__1_n_0 ),
        .Q(Q[0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p1_reg[1] 
       (.C(ap_clk),
        .CE(load_p1),
        .D(\data_p1[1]_i_1__1_n_0 ),
        .Q(Q[1]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p1_reg[2] 
       (.C(ap_clk),
        .CE(load_p1),
        .D(\data_p1[2]_i_1__1_n_0 ),
        .Q(Q[2]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p1_reg[3] 
       (.C(ap_clk),
        .CE(load_p1),
        .D(\data_p1[3]_i_2__2_n_0 ),
        .Q(Q[3]),
        .R(1'b0));
  LUT2 #(
    .INIT(4'h8)) 
    \data_p2[3]_i_1__1 
       (.I0(input_stream_TVALID),
        .I1(ack_in_t_reg_n_0),
        .O(load_p2));
  FDRE #(
    .INIT(1'b0)) 
    \data_p2_reg[0] 
       (.C(ap_clk),
        .CE(load_p2),
        .D(input_stream_TKEEP[0]),
        .Q(data_p2[0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p2_reg[1] 
       (.C(ap_clk),
        .CE(load_p2),
        .D(input_stream_TKEEP[1]),
        .Q(data_p2[1]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p2_reg[2] 
       (.C(ap_clk),
        .CE(load_p2),
        .D(input_stream_TKEEP[2]),
        .Q(data_p2[2]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p2_reg[3] 
       (.C(ap_clk),
        .CE(load_p2),
        .D(input_stream_TKEEP[3]),
        .Q(data_p2[3]),
        .R(1'b0));
endmodule

(* ORIG_REF_NAME = "filter_kernel_regslice_both" *) 
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_regslice_both__parameterized0_0
   (Q,
    ap_rst_n_inv,
    ap_clk,
    input_stream_TVALID,
    input_stream_TREADY_int_regslice,
    input_stream_TSTRB);
  output [3:0]Q;
  input ap_rst_n_inv;
  input ap_clk;
  input input_stream_TVALID;
  input input_stream_TREADY_int_regslice;
  input [3:0]input_stream_TSTRB;

  wire [3:0]Q;
  wire ack_in_t_i_1__1_n_0;
  wire ack_in_t_reg_n_0;
  wire ap_clk;
  wire ap_rst_n_inv;
  wire \data_p1[0]_i_1__2_n_0 ;
  wire \data_p1[1]_i_1__2_n_0 ;
  wire \data_p1[2]_i_1__2_n_0 ;
  wire \data_p1[3]_i_2__3_n_0 ;
  wire [3:0]data_p2;
  wire input_stream_TREADY_int_regslice;
  wire [3:0]input_stream_TSTRB;
  wire input_stream_TVALID;
  wire load_p1;
  wire load_p2;
  wire [1:0]next__0;
  wire [1:0]state__0;

  LUT4 #(
    .INIT(16'hAEFF)) 
    \FSM_sequential_state[0]_i_1__4 
       (.I0(input_stream_TREADY_int_regslice),
        .I1(state__0[0]),
        .I2(input_stream_TVALID),
        .I3(state__0[1]),
        .O(next__0[0]));
  (* SOFT_HLUTNM = "soft_lutpair58" *) 
  LUT5 #(
    .INIT(32'hFCCC4C4C)) 
    \FSM_sequential_state[1]_i_1__0 
       (.I0(input_stream_TREADY_int_regslice),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(ack_in_t_reg_n_0),
        .I4(input_stream_TVALID),
        .O(next__0[1]));
  (* FSM_ENCODED_STATES = "ZERO:01,TWO:10,ONE:11,iSTATE:00" *) 
  FDSE #(
    .INIT(1'b0)) 
    \FSM_sequential_state_reg[0] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(next__0[0]),
        .Q(state__0[0]),
        .S(ap_rst_n_inv));
  (* FSM_ENCODED_STATES = "ZERO:01,TWO:10,ONE:11,iSTATE:00" *) 
  FDRE #(
    .INIT(1'b0)) 
    \FSM_sequential_state_reg[1] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(next__0[1]),
        .Q(state__0[1]),
        .R(ap_rst_n_inv));
  (* SOFT_HLUTNM = "soft_lutpair58" *) 
  LUT5 #(
    .INIT(32'hB0FFFAF0)) 
    ack_in_t_i_1__1
       (.I0(input_stream_TREADY_int_regslice),
        .I1(input_stream_TVALID),
        .I2(ack_in_t_reg_n_0),
        .I3(state__0[1]),
        .I4(state__0[0]),
        .O(ack_in_t_i_1__1_n_0));
  FDRE #(
    .INIT(1'b0)) 
    ack_in_t_reg
       (.C(ap_clk),
        .CE(1'b1),
        .D(ack_in_t_i_1__1_n_0),
        .Q(ack_in_t_reg_n_0),
        .R(ap_rst_n_inv));
  LUT4 #(
    .INIT(16'hFB08)) 
    \data_p1[0]_i_1__2 
       (.I0(data_p2[0]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(input_stream_TSTRB[0]),
        .O(\data_p1[0]_i_1__2_n_0 ));
  LUT4 #(
    .INIT(16'hFB08)) 
    \data_p1[1]_i_1__2 
       (.I0(data_p2[1]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(input_stream_TSTRB[1]),
        .O(\data_p1[1]_i_1__2_n_0 ));
  LUT4 #(
    .INIT(16'hFB08)) 
    \data_p1[2]_i_1__2 
       (.I0(data_p2[2]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(input_stream_TSTRB[2]),
        .O(\data_p1[2]_i_1__2_n_0 ));
  LUT4 #(
    .INIT(16'hE240)) 
    \data_p1[3]_i_1 
       (.I0(state__0[1]),
        .I1(state__0[0]),
        .I2(input_stream_TVALID),
        .I3(input_stream_TREADY_int_regslice),
        .O(load_p1));
  LUT4 #(
    .INIT(16'hFB08)) 
    \data_p1[3]_i_2__3 
       (.I0(data_p2[3]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(input_stream_TSTRB[3]),
        .O(\data_p1[3]_i_2__3_n_0 ));
  FDRE #(
    .INIT(1'b0)) 
    \data_p1_reg[0] 
       (.C(ap_clk),
        .CE(load_p1),
        .D(\data_p1[0]_i_1__2_n_0 ),
        .Q(Q[0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p1_reg[1] 
       (.C(ap_clk),
        .CE(load_p1),
        .D(\data_p1[1]_i_1__2_n_0 ),
        .Q(Q[1]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p1_reg[2] 
       (.C(ap_clk),
        .CE(load_p1),
        .D(\data_p1[2]_i_1__2_n_0 ),
        .Q(Q[2]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p1_reg[3] 
       (.C(ap_clk),
        .CE(load_p1),
        .D(\data_p1[3]_i_2__3_n_0 ),
        .Q(Q[3]),
        .R(1'b0));
  LUT2 #(
    .INIT(4'h8)) 
    \data_p2[3]_i_1__2 
       (.I0(input_stream_TVALID),
        .I1(ack_in_t_reg_n_0),
        .O(load_p2));
  FDRE #(
    .INIT(1'b0)) 
    \data_p2_reg[0] 
       (.C(ap_clk),
        .CE(load_p2),
        .D(input_stream_TSTRB[0]),
        .Q(data_p2[0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p2_reg[1] 
       (.C(ap_clk),
        .CE(load_p2),
        .D(input_stream_TSTRB[1]),
        .Q(data_p2[1]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p2_reg[2] 
       (.C(ap_clk),
        .CE(load_p2),
        .D(input_stream_TSTRB[2]),
        .Q(data_p2[2]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p2_reg[3] 
       (.C(ap_clk),
        .CE(load_p2),
        .D(input_stream_TSTRB[3]),
        .Q(data_p2[3]),
        .R(1'b0));
endmodule

(* ORIG_REF_NAME = "filter_kernel_regslice_both" *) 
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_regslice_both__parameterized0_2
   (ack_in_t_reg_0,
    output_stream_TKEEP,
    ap_rst_n_inv,
    ap_clk,
    D,
    input_stream_TREADY_int_regslice,
    output_stream_TKEEP_reg,
    grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID,
    output_stream_TREADY,
    E);
  output ack_in_t_reg_0;
  output [3:0]output_stream_TKEEP;
  input ap_rst_n_inv;
  input ap_clk;
  input [3:0]D;
  input input_stream_TREADY_int_regslice;
  input [3:0]output_stream_TKEEP_reg;
  input grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID;
  input output_stream_TREADY;
  input [0:0]E;

  wire [3:0]D;
  wire [0:0]E;
  wire ack_in_t_i_1__3_n_0;
  wire ack_in_t_reg_0;
  wire ap_clk;
  wire ap_rst_n_inv;
  wire \data_p1[0]_i_1_n_0 ;
  wire \data_p1[1]_i_1_n_0 ;
  wire \data_p1[2]_i_1_n_0 ;
  wire \data_p1[3]_i_2_n_0 ;
  wire [3:0]data_p2;
  wire grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID;
  wire input_stream_TREADY_int_regslice;
  wire load_p1;
  wire [1:0]next__0;
  wire [3:0]output_stream_TKEEP;
  wire [3:0]output_stream_TKEEP_reg;
  wire output_stream_TREADY;
  wire [1:0]state__0;

  LUT4 #(
    .INIT(16'hAEFF)) 
    \FSM_sequential_state[0]_i_1 
       (.I0(output_stream_TREADY),
        .I1(state__0[0]),
        .I2(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .I3(state__0[1]),
        .O(next__0[0]));
  (* SOFT_HLUTNM = "soft_lutpair63" *) 
  LUT5 #(
    .INIT(32'hFCCC4C4C)) 
    \FSM_sequential_state[1]_i_1__4 
       (.I0(output_stream_TREADY),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(ack_in_t_reg_0),
        .I4(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .O(next__0[1]));
  (* FSM_ENCODED_STATES = "ZERO:01,TWO:10,ONE:11,iSTATE:00" *) 
  FDSE #(
    .INIT(1'b0)) 
    \FSM_sequential_state_reg[0] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(next__0[0]),
        .Q(state__0[0]),
        .S(ap_rst_n_inv));
  (* FSM_ENCODED_STATES = "ZERO:01,TWO:10,ONE:11,iSTATE:00" *) 
  FDRE #(
    .INIT(1'b0)) 
    \FSM_sequential_state_reg[1] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(next__0[1]),
        .Q(state__0[1]),
        .R(ap_rst_n_inv));
  (* SOFT_HLUTNM = "soft_lutpair63" *) 
  LUT5 #(
    .INIT(32'hB0FFFAF0)) 
    ack_in_t_i_1__3
       (.I0(output_stream_TREADY),
        .I1(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .I2(ack_in_t_reg_0),
        .I3(state__0[1]),
        .I4(state__0[0]),
        .O(ack_in_t_i_1__3_n_0));
  FDRE #(
    .INIT(1'b0)) 
    ack_in_t_reg
       (.C(ap_clk),
        .CE(1'b1),
        .D(ack_in_t_i_1__3_n_0),
        .Q(ack_in_t_reg_0),
        .R(ap_rst_n_inv));
  LUT6 #(
    .INIT(64'hFB08FBFBFB080808)) 
    \data_p1[0]_i_1 
       (.I0(data_p2[0]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(D[0]),
        .I4(input_stream_TREADY_int_regslice),
        .I5(output_stream_TKEEP_reg[0]),
        .O(\data_p1[0]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFB08FBFBFB080808)) 
    \data_p1[1]_i_1 
       (.I0(data_p2[1]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(D[1]),
        .I4(input_stream_TREADY_int_regslice),
        .I5(output_stream_TKEEP_reg[1]),
        .O(\data_p1[1]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFB08FBFBFB080808)) 
    \data_p1[2]_i_1 
       (.I0(data_p2[2]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(D[2]),
        .I4(input_stream_TREADY_int_regslice),
        .I5(output_stream_TKEEP_reg[2]),
        .O(\data_p1[2]_i_1_n_0 ));
  LUT4 #(
    .INIT(16'hE240)) 
    \data_p1[3]_i_1__2 
       (.I0(state__0[1]),
        .I1(state__0[0]),
        .I2(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .I3(output_stream_TREADY),
        .O(load_p1));
  LUT6 #(
    .INIT(64'hFB08FBFBFB080808)) 
    \data_p1[3]_i_2 
       (.I0(data_p2[3]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(D[3]),
        .I4(input_stream_TREADY_int_regslice),
        .I5(output_stream_TKEEP_reg[3]),
        .O(\data_p1[3]_i_2_n_0 ));
  FDRE #(
    .INIT(1'b0)) 
    \data_p1_reg[0] 
       (.C(ap_clk),
        .CE(load_p1),
        .D(\data_p1[0]_i_1_n_0 ),
        .Q(output_stream_TKEEP[0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p1_reg[1] 
       (.C(ap_clk),
        .CE(load_p1),
        .D(\data_p1[1]_i_1_n_0 ),
        .Q(output_stream_TKEEP[1]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p1_reg[2] 
       (.C(ap_clk),
        .CE(load_p1),
        .D(\data_p1[2]_i_1_n_0 ),
        .Q(output_stream_TKEEP[2]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p1_reg[3] 
       (.C(ap_clk),
        .CE(load_p1),
        .D(\data_p1[3]_i_2_n_0 ),
        .Q(output_stream_TKEEP[3]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p2_reg[0] 
       (.C(ap_clk),
        .CE(E),
        .D(D[0]),
        .Q(data_p2[0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p2_reg[1] 
       (.C(ap_clk),
        .CE(E),
        .D(D[1]),
        .Q(data_p2[1]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p2_reg[2] 
       (.C(ap_clk),
        .CE(E),
        .D(D[2]),
        .Q(data_p2[2]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p2_reg[3] 
       (.C(ap_clk),
        .CE(E),
        .D(D[3]),
        .Q(data_p2[3]),
        .R(1'b0));
endmodule

(* ORIG_REF_NAME = "filter_kernel_regslice_both" *) 
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_regslice_both__parameterized0_3
   (ack_in_t_reg_0,
    output_stream_TSTRB,
    ap_rst_n_inv,
    ap_clk,
    D,
    input_stream_TREADY_int_regslice,
    output_stream_TSTRB_reg,
    grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID,
    output_stream_TREADY,
    E);
  output ack_in_t_reg_0;
  output [3:0]output_stream_TSTRB;
  input ap_rst_n_inv;
  input ap_clk;
  input [3:0]D;
  input input_stream_TREADY_int_regslice;
  input [3:0]output_stream_TSTRB_reg;
  input grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID;
  input output_stream_TREADY;
  input [0:0]E;

  wire [3:0]D;
  wire [0:0]E;
  wire ack_in_t_i_1__4_n_0;
  wire ack_in_t_reg_0;
  wire ap_clk;
  wire ap_rst_n_inv;
  wire \data_p1[0]_i_1__0_n_0 ;
  wire \data_p1[1]_i_1__0_n_0 ;
  wire \data_p1[2]_i_1__0_n_0 ;
  wire \data_p1[3]_i_2__0_n_0 ;
  wire [3:0]data_p2;
  wire grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID;
  wire input_stream_TREADY_int_regslice;
  wire load_p1;
  wire [1:0]next__0;
  wire output_stream_TREADY;
  wire [3:0]output_stream_TSTRB;
  wire [3:0]output_stream_TSTRB_reg;
  wire [1:0]state__0;

  LUT4 #(
    .INIT(16'hAEFF)) 
    \FSM_sequential_state[0]_i_1__0 
       (.I0(output_stream_TREADY),
        .I1(state__0[0]),
        .I2(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .I3(state__0[1]),
        .O(next__0[0]));
  (* SOFT_HLUTNM = "soft_lutpair66" *) 
  LUT5 #(
    .INIT(32'hFCCC4C4C)) 
    \FSM_sequential_state[1]_i_1__3 
       (.I0(output_stream_TREADY),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(ack_in_t_reg_0),
        .I4(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .O(next__0[1]));
  (* FSM_ENCODED_STATES = "ZERO:01,TWO:10,ONE:11,iSTATE:00" *) 
  FDSE #(
    .INIT(1'b0)) 
    \FSM_sequential_state_reg[0] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(next__0[0]),
        .Q(state__0[0]),
        .S(ap_rst_n_inv));
  (* FSM_ENCODED_STATES = "ZERO:01,TWO:10,ONE:11,iSTATE:00" *) 
  FDRE #(
    .INIT(1'b0)) 
    \FSM_sequential_state_reg[1] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(next__0[1]),
        .Q(state__0[1]),
        .R(ap_rst_n_inv));
  (* SOFT_HLUTNM = "soft_lutpair66" *) 
  LUT5 #(
    .INIT(32'hB0FFFAF0)) 
    ack_in_t_i_1__4
       (.I0(output_stream_TREADY),
        .I1(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .I2(ack_in_t_reg_0),
        .I3(state__0[1]),
        .I4(state__0[0]),
        .O(ack_in_t_i_1__4_n_0));
  FDRE #(
    .INIT(1'b0)) 
    ack_in_t_reg
       (.C(ap_clk),
        .CE(1'b1),
        .D(ack_in_t_i_1__4_n_0),
        .Q(ack_in_t_reg_0),
        .R(ap_rst_n_inv));
  LUT6 #(
    .INIT(64'hFB08FBFBFB080808)) 
    \data_p1[0]_i_1__0 
       (.I0(data_p2[0]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(D[0]),
        .I4(input_stream_TREADY_int_regslice),
        .I5(output_stream_TSTRB_reg[0]),
        .O(\data_p1[0]_i_1__0_n_0 ));
  LUT6 #(
    .INIT(64'hFB08FBFBFB080808)) 
    \data_p1[1]_i_1__0 
       (.I0(data_p2[1]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(D[1]),
        .I4(input_stream_TREADY_int_regslice),
        .I5(output_stream_TSTRB_reg[1]),
        .O(\data_p1[1]_i_1__0_n_0 ));
  LUT6 #(
    .INIT(64'hFB08FBFBFB080808)) 
    \data_p1[2]_i_1__0 
       (.I0(data_p2[2]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(D[2]),
        .I4(input_stream_TREADY_int_regslice),
        .I5(output_stream_TSTRB_reg[2]),
        .O(\data_p1[2]_i_1__0_n_0 ));
  LUT4 #(
    .INIT(16'hE240)) 
    \data_p1[3]_i_1__1 
       (.I0(state__0[1]),
        .I1(state__0[0]),
        .I2(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .I3(output_stream_TREADY),
        .O(load_p1));
  LUT6 #(
    .INIT(64'hFB08FBFBFB080808)) 
    \data_p1[3]_i_2__0 
       (.I0(data_p2[3]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(D[3]),
        .I4(input_stream_TREADY_int_regslice),
        .I5(output_stream_TSTRB_reg[3]),
        .O(\data_p1[3]_i_2__0_n_0 ));
  FDRE #(
    .INIT(1'b0)) 
    \data_p1_reg[0] 
       (.C(ap_clk),
        .CE(load_p1),
        .D(\data_p1[0]_i_1__0_n_0 ),
        .Q(output_stream_TSTRB[0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p1_reg[1] 
       (.C(ap_clk),
        .CE(load_p1),
        .D(\data_p1[1]_i_1__0_n_0 ),
        .Q(output_stream_TSTRB[1]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p1_reg[2] 
       (.C(ap_clk),
        .CE(load_p1),
        .D(\data_p1[2]_i_1__0_n_0 ),
        .Q(output_stream_TSTRB[2]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p1_reg[3] 
       (.C(ap_clk),
        .CE(load_p1),
        .D(\data_p1[3]_i_2__0_n_0 ),
        .Q(output_stream_TSTRB[3]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p2_reg[0] 
       (.C(ap_clk),
        .CE(E),
        .D(D[0]),
        .Q(data_p2[0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p2_reg[1] 
       (.C(ap_clk),
        .CE(E),
        .D(D[1]),
        .Q(data_p2[1]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p2_reg[2] 
       (.C(ap_clk),
        .CE(E),
        .D(D[2]),
        .Q(data_p2[2]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p2_reg[3] 
       (.C(ap_clk),
        .CE(E),
        .D(D[3]),
        .Q(data_p2[3]),
        .R(1'b0));
endmodule

(* ORIG_REF_NAME = "filter_kernel_regslice_both" *) 
module decalper_eb_ot_sdeen_pot_pi_dehcac_xnilix_filter_kernel_regslice_both__parameterized1
   (ack_in_t_reg_0,
    data_p2,
    output_stream_TLAST,
    \FSM_sequential_state_reg[1]_0 ,
    ap_rst_n_inv,
    ap_clk,
    \data_p2_reg[0]_0 ,
    output_stream_TREADY,
    grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID,
    \data_p1_reg[0]_0 );
  output ack_in_t_reg_0;
  output data_p2;
  output [0:0]output_stream_TLAST;
  output \FSM_sequential_state_reg[1]_0 ;
  input ap_rst_n_inv;
  input ap_clk;
  input \data_p2_reg[0]_0 ;
  input output_stream_TREADY;
  input grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID;
  input \data_p1_reg[0]_0 ;

  wire \FSM_sequential_state_reg[1]_0 ;
  wire ack_in_t_i_1__5_n_0;
  wire ack_in_t_reg_0;
  wire ap_clk;
  wire ap_rst_n_inv;
  wire \data_p1[0]_i_1__3_n_0 ;
  wire \data_p1_reg[0]_0 ;
  wire data_p2;
  wire \data_p2_reg[0]_0 ;
  wire grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID;
  wire [1:0]next__0;
  wire [0:0]output_stream_TLAST;
  wire output_stream_TREADY;
  wire [1:0]state__0;

  (* SOFT_HLUTNM = "soft_lutpair65" *) 
  LUT4 #(
    .INIT(16'hAEFF)) 
    \FSM_sequential_state[0]_i_1__1 
       (.I0(output_stream_TREADY),
        .I1(state__0[0]),
        .I2(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .I3(state__0[1]),
        .O(next__0[0]));
  (* SOFT_HLUTNM = "soft_lutpair64" *) 
  LUT5 #(
    .INIT(32'hFCCC4C4C)) 
    \FSM_sequential_state[1]_i_1__2 
       (.I0(output_stream_TREADY),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(ack_in_t_reg_0),
        .I4(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .O(next__0[1]));
  (* FSM_ENCODED_STATES = "ZERO:01,TWO:10,ONE:11,iSTATE:00" *) 
  FDSE #(
    .INIT(1'b0)) 
    \FSM_sequential_state_reg[0] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(next__0[0]),
        .Q(state__0[0]),
        .S(ap_rst_n_inv));
  (* FSM_ENCODED_STATES = "ZERO:01,TWO:10,ONE:11,iSTATE:00" *) 
  FDRE #(
    .INIT(1'b0)) 
    \FSM_sequential_state_reg[1] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(next__0[1]),
        .Q(state__0[1]),
        .R(ap_rst_n_inv));
  (* SOFT_HLUTNM = "soft_lutpair64" *) 
  LUT5 #(
    .INIT(32'hB0FFFAF0)) 
    ack_in_t_i_1__5
       (.I0(output_stream_TREADY),
        .I1(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .I2(ack_in_t_reg_0),
        .I3(state__0[1]),
        .I4(state__0[0]),
        .O(ack_in_t_i_1__5_n_0));
  FDRE #(
    .INIT(1'b0)) 
    ack_in_t_reg
       (.C(ap_clk),
        .CE(1'b1),
        .D(ack_in_t_i_1__5_n_0),
        .Q(ack_in_t_reg_0),
        .R(ap_rst_n_inv));
  LUT6 #(
    .INIT(64'hABFBEFFFA8082000)) 
    \data_p1[0]_i_1__3 
       (.I0(\data_p1_reg[0]_0 ),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(grp_filter_kernel_Pipeline_VITIS_LOOP_34_1_VITIS_LOOP_35_2_fu_84_output_stream_TVALID),
        .I4(output_stream_TREADY),
        .I5(output_stream_TLAST),
        .O(\data_p1[0]_i_1__3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair65" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \data_p1[0]_i_3 
       (.I0(state__0[1]),
        .I1(state__0[0]),
        .O(\FSM_sequential_state_reg[1]_0 ));
  FDRE #(
    .INIT(1'b0)) 
    \data_p1_reg[0] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\data_p1[0]_i_1__3_n_0 ),
        .Q(output_stream_TLAST),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \data_p2_reg[0] 
       (.C(ap_clk),
        .CE(1'b1),
        .D(\data_p2_reg[0]_0 ),
        .Q(data_p2),
        .R(1'b0));
endmodule
`ifndef GLBL
`define GLBL
`timescale  1 ps / 1 ps

module glbl ();

    parameter ROC_WIDTH = 100000;
    parameter TOC_WIDTH = 0;
    parameter GRES_WIDTH = 10000;
    parameter GRES_START = 10000;

//--------   STARTUP Globals --------------
    wire GSR;
    wire GTS;
    wire GWE;
    wire PRLD;
    wire GRESTORE;
    tri1 p_up_tmp;
    tri (weak1, strong0) PLL_LOCKG = p_up_tmp;

    wire PROGB_GLBL;
    wire CCLKO_GLBL;
    wire FCSBO_GLBL;
    wire [3:0] DO_GLBL;
    wire [3:0] DI_GLBL;
   
    reg GSR_int;
    reg GTS_int;
    reg PRLD_int;
    reg GRESTORE_int;

//--------   JTAG Globals --------------
    wire JTAG_TDO_GLBL;
    wire JTAG_TCK_GLBL;
    wire JTAG_TDI_GLBL;
    wire JTAG_TMS_GLBL;
    wire JTAG_TRST_GLBL;

    reg JTAG_CAPTURE_GLBL;
    reg JTAG_RESET_GLBL;
    reg JTAG_SHIFT_GLBL;
    reg JTAG_UPDATE_GLBL;
    reg JTAG_RUNTEST_GLBL;

    reg JTAG_SEL1_GLBL = 0;
    reg JTAG_SEL2_GLBL = 0 ;
    reg JTAG_SEL3_GLBL = 0;
    reg JTAG_SEL4_GLBL = 0;

    reg JTAG_USER_TDO1_GLBL = 1'bz;
    reg JTAG_USER_TDO2_GLBL = 1'bz;
    reg JTAG_USER_TDO3_GLBL = 1'bz;
    reg JTAG_USER_TDO4_GLBL = 1'bz;

    assign (strong1, weak0) GSR = GSR_int;
    assign (strong1, weak0) GTS = GTS_int;
    assign (weak1, weak0) PRLD = PRLD_int;
    assign (strong1, weak0) GRESTORE = GRESTORE_int;

    initial begin
	GSR_int = 1'b1;
	PRLD_int = 1'b1;
	#(ROC_WIDTH)
	GSR_int = 1'b0;
	PRLD_int = 1'b0;
    end

    initial begin
	GTS_int = 1'b1;
	#(TOC_WIDTH)
	GTS_int = 1'b0;
    end

    initial begin 
	GRESTORE_int = 1'b0;
	#(GRES_START);
	GRESTORE_int = 1'b1;
	#(GRES_WIDTH);
	GRESTORE_int = 1'b0;
    end

endmodule
`endif
