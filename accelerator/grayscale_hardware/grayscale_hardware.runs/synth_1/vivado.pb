
>
Refreshing IP repositories
234*coregenZ19-234h px� 
�
 Loaded user IP repository '%s'.
1135*coregen2>
<d:/cse_462/fgpa_convolution_acceleration/grayscale/grayscaleZ19-1700h px� 
j
"Loaded Vivado IP repository '%s'.
1332*coregen2!
C:/Xilinx/Vivado/2024.2/data/ipZ19-2313h px� 
�
�The host OS only allows 260 characters in a normal path. The IP cache path is more than 80 characters. If you experience issues with IP caching, please consider changing the IP cache to a location with a shorter path. Alternately consider using the OS subst command to map part of the path to a drive letter. 
Current IP cache path is %s 2293*coregen2Y
Wd:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.cache/ipZ19-4995h px� 
l
Command: %s
53*	vivadotcl2;
9synth_design -top grayscale_wrapper -part xc7z020clg400-1Z4-113h px� 
:
Starting synth_design
149*	vivadotclZ4-321h px� 
z
@Attempting to get a license for feature '%s' and/or device '%s'
308*common2
	Synthesis2	
xc7z020Z17-347h px� 
j
0Got license for feature '%s' and/or device '%s'
310*common2
	Synthesis2	
xc7z020Z17-349h px� 
D
Loading part %s157*device2
xc7z020clg400-1Z21-403h px� 
o
HMultithreading enabled for synth_design using a maximum of %s processes.4828*oasys2
2Z8-7079h px� 
a
?Launching helper process for spawning children vivado processes4827*oasysZ8-7078h px� 
N
#Helper process launched with PID %s4824*oasys2
12408Z8-7075h px� 
�
%s*synth2{
yStarting RTL Elaboration : Time (s): cpu = 00:00:04 ; elapsed = 00:00:05 . Memory (MB): peak = 1130.949 ; gain = 466.762
h px� 
�
synthesizing module '%s'%s4497*oasys2
grayscale_wrapper2
 2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/hdl/grayscale_wrapper.v2
138@Z8-6157h px� 
�
synthesizing module '%s'%s4497*oasys2
	grayscale2
 2
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
138@Z8-6157h px� 
�
synthesizing module '%s'%s4497*oasys2
grayscale_axi_dma_0_12
 2�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/.Xil/Vivado-27244-LAPTOP-D6BOEUU6/realtime/grayscale_axi_dma_0_1_stub.v2
68@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2
grayscale_axi_dma_0_12
 2
02
12�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/.Xil/Vivado-27244-LAPTOP-D6BOEUU6/realtime/grayscale_axi_dma_0_1_stub.v2
68@Z8-6155h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
mm2s_prmry_reset_out_n2
grayscale_axi_dma_0_12
	axi_dma_02
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
2318@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
s2mm_prmry_reset_out_n2
grayscale_axi_dma_0_12
	axi_dma_02
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
2318@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
mm2s_introut2
grayscale_axi_dma_0_12
	axi_dma_02
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
2318@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
s2mm_introut2
grayscale_axi_dma_0_12
	axi_dma_02
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
2318@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
axi_dma_tstvec2
grayscale_axi_dma_0_12
	axi_dma_02
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
2318@Z8-7071h px� 
�
Kinstance '%s' of module '%s' has %s connections declared, but only %s given4757*oasys2
	axi_dma_02
grayscale_axi_dma_0_12
642
592
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
2318@Z8-7023h px� 
�
synthesizing module '%s'%s4497*oasys2
grayscale_axi_mem_intercon_02
 2
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
5868@Z8-6157h px� 
�
synthesizing module '%s'%s4497*oasys2
m00_couplers_imp_1BA43ZX2
 2
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
11178@Z8-6157h px� 
�
synthesizing module '%s'%s4497*oasys2*
(grayscale_axi_mem_intercon_imp_auto_pc_02
 2�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/.Xil/Vivado-27244-LAPTOP-D6BOEUU6/realtime/grayscale_axi_mem_intercon_imp_auto_pc_0_stub.v2
68@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2*
(grayscale_axi_mem_intercon_imp_auto_pc_02
 2
02
12�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/.Xil/Vivado-27244-LAPTOP-D6BOEUU6/realtime/grayscale_axi_mem_intercon_imp_auto_pc_0_stub.v2
68@Z8-6155h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2
m00_couplers_imp_1BA43ZX2
 2
02
12
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
11178@Z8-6155h px� 
�
synthesizing module '%s'%s4497*oasys2
s00_couplers_imp_DYEVKX2
 2
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
14438@Z8-6157h px� 
�
synthesizing module '%s'%s4497*oasys2*
(grayscale_axi_mem_intercon_imp_auto_us_02
 2�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/.Xil/Vivado-27244-LAPTOP-D6BOEUU6/realtime/grayscale_axi_mem_intercon_imp_auto_us_0_stub.v2
68@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2*
(grayscale_axi_mem_intercon_imp_auto_us_02
 2
02
12�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/.Xil/Vivado-27244-LAPTOP-D6BOEUU6/realtime/grayscale_axi_mem_intercon_imp_auto_us_0_stub.v2
68@Z8-6155h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
m_axi_arregion2*
(grayscale_axi_mem_intercon_imp_auto_us_02	
auto_us2
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
15408@Z8-7071h px� 
�
Kinstance '%s' of module '%s' has %s connections declared, but only %s given4757*oasys2	
auto_us2*
(grayscale_axi_mem_intercon_imp_auto_us_02
342
332
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
15408@Z8-7023h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2
s00_couplers_imp_DYEVKX2
 2
02
12
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
14438@Z8-6155h px� 
�
synthesizing module '%s'%s4497*oasys2
s01_couplers_imp_1RQ8JZT2
 2
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
15768@Z8-6157h px� 
�
synthesizing module '%s'%s4497*oasys2*
(grayscale_axi_mem_intercon_imp_auto_us_12
 2�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/.Xil/Vivado-27244-LAPTOP-D6BOEUU6/realtime/grayscale_axi_mem_intercon_imp_auto_us_1_stub.v2
68@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2*
(grayscale_axi_mem_intercon_imp_auto_us_12
 2
02
12�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/.Xil/Vivado-27244-LAPTOP-D6BOEUU6/realtime/grayscale_axi_mem_intercon_imp_auto_us_1_stub.v2
68@Z8-6155h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
m_axi_awregion2*
(grayscale_axi_mem_intercon_imp_auto_us_12	
auto_us2
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
16918@Z8-7071h px� 
�
Kinstance '%s' of module '%s' has %s connections declared, but only %s given4757*oasys2	
auto_us2*
(grayscale_axi_mem_intercon_imp_auto_us_12
402
392
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
16918@Z8-7023h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2
s01_couplers_imp_1RQ8JZT2
 2
02
12
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
15768@Z8-6155h px� 
�
synthesizing module '%s'%s4497*oasys2'
%grayscale_axi_mem_intercon_imp_xbar_02
 2�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/.Xil/Vivado-27244-LAPTOP-D6BOEUU6/realtime/grayscale_axi_mem_intercon_imp_xbar_0_stub.v2
68@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2'
%grayscale_axi_mem_intercon_imp_xbar_02
 2
02
12�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/.Xil/Vivado-27244-LAPTOP-D6BOEUU6/realtime/grayscale_axi_mem_intercon_imp_xbar_0_stub.v2
68@Z8-6155h px� 
�
Pwidth (%s) of port connection '%s' does not match port width (%s) of module '%s'689*oasys2
12
s_axi_arready2
22'
%grayscale_axi_mem_intercon_imp_xbar_02
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
10888@Z8-689h px� 
�
Pwidth (%s) of port connection '%s' does not match port width (%s) of module '%s'689*oasys2
642
s_axi_rdata2
1282'
%grayscale_axi_mem_intercon_imp_xbar_02
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
11058@Z8-689h px� 
�
Pwidth (%s) of port connection '%s' does not match port width (%s) of module '%s'689*oasys2
12
s_axi_rlast2
22'
%grayscale_axi_mem_intercon_imp_xbar_02
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
11068@Z8-689h px� 
�
Pwidth (%s) of port connection '%s' does not match port width (%s) of module '%s'689*oasys2
22
s_axi_rresp2
42'
%grayscale_axi_mem_intercon_imp_xbar_02
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
11088@Z8-689h px� 
�
Pwidth (%s) of port connection '%s' does not match port width (%s) of module '%s'689*oasys2
12
s_axi_rvalid2
22'
%grayscale_axi_mem_intercon_imp_xbar_02
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
11098@Z8-689h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
	s_axi_bid2'
%grayscale_axi_mem_intercon_imp_xbar_02
xbar2
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
10388@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
	s_axi_rid2'
%grayscale_axi_mem_intercon_imp_xbar_02
xbar2
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
10388@Z8-7071h px� 
�
Kinstance '%s' of module '%s' has %s connections declared, but only %s given4757*oasys2
xbar2'
%grayscale_axi_mem_intercon_imp_xbar_02
782
762
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
10388@Z8-7023h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2
grayscale_axi_mem_intercon_02
 2
02
12
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
5868@Z8-6155h px� 
�
synthesizing module '%s'%s4497*oasys2
grayscale_axi_smc_02
 2�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/.Xil/Vivado-27244-LAPTOP-D6BOEUU6/realtime/grayscale_axi_smc_0_stub.v2
68@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2
grayscale_axi_smc_02
 2
02
12�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/.Xil/Vivado-27244-LAPTOP-D6BOEUU6/realtime/grayscale_axi_smc_0_stub.v2
68@Z8-6155h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
M00_AXI_awprot2
grayscale_axi_smc_02	
axi_smc2
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
3678@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
M00_AXI_arprot2
grayscale_axi_smc_02	
axi_smc2
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
3678@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
M01_AXI_awprot2
grayscale_axi_smc_02	
axi_smc2
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
3678@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
M01_AXI_wstrb2
grayscale_axi_smc_02	
axi_smc2
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
3678@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
M01_AXI_arprot2
grayscale_axi_smc_02	
axi_smc2
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
3678@Z8-7071h px� 
�
Kinstance '%s' of module '%s' has %s connections declared, but only %s given4757*oasys2	
axi_smc2
grayscale_axi_smc_02
782
732
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
3678@Z8-7023h px� 
�
synthesizing module '%s'%s4497*oasys2 
grayscale_grayscale_kernel_0_22
 2�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/.Xil/Vivado-27244-LAPTOP-D6BOEUU6/realtime/grayscale_grayscale_kernel_0_2_stub.v2
68@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2 
grayscale_grayscale_kernel_0_22
 2
02
12�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/.Xil/Vivado-27244-LAPTOP-D6BOEUU6/realtime/grayscale_grayscale_kernel_0_2_stub.v2
68@Z8-6155h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
	interrupt2 
grayscale_grayscale_kernel_0_22
grayscale_kernel_02
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
4418@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
output_stream_TSTRB2 
grayscale_grayscale_kernel_0_22
grayscale_kernel_02
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
4418@Z8-7071h px� 
�
Kinstance '%s' of module '%s' has %s connections declared, but only %s given4757*oasys2
grayscale_kernel_02 
grayscale_grayscale_kernel_0_22
322
302
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
4418@Z8-7023h px� 
�
synthesizing module '%s'%s4497*oasys2"
 grayscale_processing_system7_0_12
 2�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/.Xil/Vivado-27244-LAPTOP-D6BOEUU6/realtime/grayscale_processing_system7_0_1_stub.v2
68@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2"
 grayscale_processing_system7_0_12
 2
02
12�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/.Xil/Vivado-27244-LAPTOP-D6BOEUU6/realtime/grayscale_processing_system7_0_1_stub.v2
68@Z8-6155h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
USB0_PORT_INDCTL2"
 grayscale_processing_system7_0_12
processing_system7_02
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
4728@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
USB0_VBUS_PWRSELECT2"
 grayscale_processing_system7_0_12
processing_system7_02
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
4728@Z8-7071h px� 
�
Kinstance '%s' of module '%s' has %s connections declared, but only %s given4757*oasys2
processing_system7_02"
 grayscale_processing_system7_0_12
1062
1042
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
4728@Z8-7023h px� 
�
synthesizing module '%s'%s4497*oasys2
grayscale_rst_ps7_0_100M_02
 2�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/.Xil/Vivado-27244-LAPTOP-D6BOEUU6/realtime/grayscale_rst_ps7_0_100M_0_stub.v2
68@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2
grayscale_rst_ps7_0_100M_02
 2
02
12�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/.Xil/Vivado-27244-LAPTOP-D6BOEUU6/realtime/grayscale_rst_ps7_0_100M_0_stub.v2
68@Z8-6155h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2

mb_reset2
grayscale_rst_ps7_0_100M_02
rst_ps7_0_100M2
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
5778@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
bus_struct_reset2
grayscale_rst_ps7_0_100M_02
rst_ps7_0_100M2
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
5778@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
peripheral_reset2
grayscale_rst_ps7_0_100M_02
rst_ps7_0_100M2
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
5778@Z8-7071h px� 
�
9port '%s' of module '%s' is unconnected for instance '%s'4818*oasys2
interconnect_aresetn2
grayscale_rst_ps7_0_100M_02
rst_ps7_0_100M2
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
5778@Z8-7071h px� 
�
Kinstance '%s' of module '%s' has %s connections declared, but only %s given4757*oasys2
rst_ps7_0_100M2
grayscale_rst_ps7_0_100M_02
102
62
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
5778@Z8-7023h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2
	grayscale2
 2
02
12
{d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/synth/grayscale.v2
138@Z8-6155h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2
grayscale_wrapper2
 2
02
12�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/hdl/grayscale_wrapper.v2
138@Z8-6155h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
M_ACLK2
s01_couplers_imp_1RQ8JZTZ8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
	M_ARESETN2
s01_couplers_imp_1RQ8JZTZ8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
M_ACLK2
s00_couplers_imp_DYEVKXZ8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
	M_ARESETN2
s00_couplers_imp_DYEVKXZ8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
M_ACLK2
m00_couplers_imp_1BA43ZXZ8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
	M_ARESETN2
m00_couplers_imp_1BA43ZXZ8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
M_AXI_bid[2]2
m00_couplers_imp_1BA43ZXZ8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
M_AXI_bid[1]2
m00_couplers_imp_1BA43ZXZ8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
M_AXI_rid[2]2
m00_couplers_imp_1BA43ZXZ8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
M_AXI_rid[1]2
m00_couplers_imp_1BA43ZXZ8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2

M00_ACLK2
grayscale_axi_mem_intercon_0Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
M00_ARESETN2
grayscale_axi_mem_intercon_0Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2

S00_ACLK2
grayscale_axi_mem_intercon_0Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
S00_ARESETN2
grayscale_axi_mem_intercon_0Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2

S01_ACLK2
grayscale_axi_mem_intercon_0Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
S01_ARESETN2
grayscale_axi_mem_intercon_0Z8-7129h px� 
�
%s*synth2{
yFinished RTL Elaboration : Time (s): cpu = 00:00:06 ; elapsed = 00:00:07 . Memory (MB): peak = 1246.680 ; gain = 582.492
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
;
%s
*synth2#
!Start Handling Custom Attributes
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Handling Custom Attributes : Time (s): cpu = 00:00:06 ; elapsed = 00:00:07 . Memory (MB): peak = 1246.680 ; gain = 582.492
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished RTL Optimization Phase 1 : Time (s): cpu = 00:00:06 ; elapsed = 00:00:07 . Memory (MB): peak = 1246.680 ; gain = 582.492
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
Netlist sorting complete. 2

00:00:002
00:00:00.0252

1246.6802
0.000Z17-268h px� 
K
)Preparing netlist for logic optimization
349*projectZ1-570h px� 
>

Processing XDC Constraints
244*projectZ1-262h px� 
=
Initializing timing engine
348*projectZ1-569h px� 
�
$Parsing XDC File [%s] for cell '%s'
848*designutils2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_processing_system7_0_1/grayscale_processing_system7_0_1/grayscale_processing_system7_0_1_in_context.xdc2$
 grayscale_i/processing_system7_0	8Z20-848h px� 
�
-Finished Parsing XDC File [%s] for cell '%s'
847*designutils2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_processing_system7_0_1/grayscale_processing_system7_0_1/grayscale_processing_system7_0_1_in_context.xdc2$
 grayscale_i/processing_system7_0	8Z20-847h px� 
�
$Parsing XDC File [%s] for cell '%s'
848*designutils2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_grayscale_kernel_0_2/grayscale_grayscale_kernel_0_2/grayscale_grayscale_kernel_0_2_in_context.xdc2"
grayscale_i/grayscale_kernel_0	8Z20-848h px� 
�
-Finished Parsing XDC File [%s] for cell '%s'
847*designutils2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_grayscale_kernel_0_2/grayscale_grayscale_kernel_0_2/grayscale_grayscale_kernel_0_2_in_context.xdc2"
grayscale_i/grayscale_kernel_0	8Z20-847h px� 
�
$Parsing XDC File [%s] for cell '%s'
848*designutils2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_axi_smc_0/grayscale_axi_smc_0/grayscale_axi_smc_0_in_context.xdc2
grayscale_i/axi_smc	8Z20-848h px� 
�
-Finished Parsing XDC File [%s] for cell '%s'
847*designutils2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_axi_smc_0/grayscale_axi_smc_0/grayscale_axi_smc_0_in_context.xdc2
grayscale_i/axi_smc	8Z20-847h px� 
�
$Parsing XDC File [%s] for cell '%s'
848*designutils2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_rst_ps7_0_100M_0/grayscale_rst_ps7_0_100M_0/grayscale_rst_ps7_0_100M_0_in_context.xdc2
grayscale_i/rst_ps7_0_100M	8Z20-848h px� 
�
-Finished Parsing XDC File [%s] for cell '%s'
847*designutils2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_rst_ps7_0_100M_0/grayscale_rst_ps7_0_100M_0/grayscale_rst_ps7_0_100M_0_in_context.xdc2
grayscale_i/rst_ps7_0_100M	8Z20-847h px� 
�
$Parsing XDC File [%s] for cell '%s'
848*designutils2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_axi_dma_0_1/grayscale_axi_dma_0_1/grayscale_axi_dma_0_1_in_context.xdc2
grayscale_i/axi_dma_0	8Z20-848h px� 
�
-Finished Parsing XDC File [%s] for cell '%s'
847*designutils2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_axi_dma_0_1/grayscale_axi_dma_0_1/grayscale_axi_dma_0_1_in_context.xdc2
grayscale_i/axi_dma_0	8Z20-847h px� 
�
$Parsing XDC File [%s] for cell '%s'
848*designutils2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_axi_mem_intercon_imp_xbar_0/grayscale_axi_mem_intercon_imp_xbar_0/grayscale_axi_mem_intercon_imp_xbar_0_in_context.xdc2%
!grayscale_i/axi_mem_intercon/xbar	8Z20-848h px� 
�
-Finished Parsing XDC File [%s] for cell '%s'
847*designutils2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_axi_mem_intercon_imp_xbar_0/grayscale_axi_mem_intercon_imp_xbar_0/grayscale_axi_mem_intercon_imp_xbar_0_in_context.xdc2%
!grayscale_i/axi_mem_intercon/xbar	8Z20-847h px� 
�
$Parsing XDC File [%s] for cell '%s'
848*designutils2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_axi_mem_intercon_imp_auto_us_0/grayscale_axi_mem_intercon_imp_auto_us_0/grayscale_axi_mem_intercon_imp_auto_us_0_in_context.xdc25
1grayscale_i/axi_mem_intercon/s00_couplers/auto_us	8Z20-848h px� 
�
-Finished Parsing XDC File [%s] for cell '%s'
847*designutils2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_axi_mem_intercon_imp_auto_us_0/grayscale_axi_mem_intercon_imp_auto_us_0/grayscale_axi_mem_intercon_imp_auto_us_0_in_context.xdc25
1grayscale_i/axi_mem_intercon/s00_couplers/auto_us	8Z20-847h px� 
�
$Parsing XDC File [%s] for cell '%s'
848*designutils2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_axi_mem_intercon_imp_auto_us_1/grayscale_axi_mem_intercon_imp_auto_us_1/grayscale_axi_mem_intercon_imp_auto_us_1_in_context.xdc25
1grayscale_i/axi_mem_intercon/s01_couplers/auto_us	8Z20-848h px� 
�
-Finished Parsing XDC File [%s] for cell '%s'
847*designutils2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_axi_mem_intercon_imp_auto_us_1/grayscale_axi_mem_intercon_imp_auto_us_1/grayscale_axi_mem_intercon_imp_auto_us_1_in_context.xdc25
1grayscale_i/axi_mem_intercon/s01_couplers/auto_us	8Z20-847h px� 
�
$Parsing XDC File [%s] for cell '%s'
848*designutils2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_axi_mem_intercon_imp_auto_pc_0/grayscale_axi_mem_intercon_imp_auto_pc_0/grayscale_axi_mem_intercon_imp_auto_pc_0_in_context.xdc25
1grayscale_i/axi_mem_intercon/m00_couplers/auto_pc	8Z20-848h px� 
�
-Finished Parsing XDC File [%s] for cell '%s'
847*designutils2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_axi_mem_intercon_imp_auto_pc_0/grayscale_axi_mem_intercon_imp_auto_pc_0/grayscale_axi_mem_intercon_imp_auto_pc_0_in_context.xdc25
1grayscale_i/axi_mem_intercon/m00_couplers/auto_pc	8Z20-847h px� 
�
Parsing XDC File [%s]
179*designutils2n
jD:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/dont_touch.xdc8Z20-179h px� 
�
Finished Parsing XDC File [%s]
178*designutils2n
jD:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/dont_touch.xdc8Z20-178h px� 
H
&Completed Processing XDC Constraints

245*projectZ1-263h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
Netlist sorting complete. 2

00:00:002
00:00:00.0012

1246.6802
0.000Z17-268h px� 
l
!Unisim Transformation Summary:
%s111*project2'
%No Unisim elements were transformed.
Z1-111h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2"
 Constraint Validation Runtime : 2

00:00:002
00:00:00.0102

1246.6802
0.000Z17-268h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
Finished Constraint Validation : Time (s): cpu = 00:00:13 ; elapsed = 00:00:15 . Memory (MB): peak = 1246.680 ; gain = 582.492
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
D
%s
*synth2,
*Start Loading Part and Timing Information
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
8
%s
*synth2 
Loading part: xc7z020clg400-1
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Loading Part and Timing Information : Time (s): cpu = 00:00:13 ; elapsed = 00:00:15 . Memory (MB): peak = 1246.680 ; gain = 582.492
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
H
%s
*synth20
.Start Applying 'set_property' XDC Constraints
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished applying 'set_property' XDC Constraints : Time (s): cpu = 00:00:13 ; elapsed = 00:00:15 . Memory (MB): peak = 1246.680 ; gain = 582.492
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished RTL Optimization Phase 2 : Time (s): cpu = 00:00:13 ; elapsed = 00:00:16 . Memory (MB): peak = 1246.680 ; gain = 582.492
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
:
%s
*synth2"
 Start RTL Component Statistics 
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
9
%s
*synth2!
Detailed RTL Component Info : 
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
=
%s
*synth2%
#Finished RTL Component Statistics 
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
6
%s
*synth2
Start Part Resource Summary
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
q
%s
*synth2Y
WPart Resources:
DSPs: 220 (col length:60)
BRAMs: 280 (col length: RAMB18 60 RAMB36 30)
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
9
%s
*synth2!
Finished Part Resource Summary
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
E
%s
*synth2-
+Start Cross Boundary and Area Optimization
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
H
&Parallel synthesis criteria is not met4829*oasysZ8-7080h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2

M00_ACLK2
grayscale_axi_mem_intercon_0Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
M00_ARESETN2
grayscale_axi_mem_intercon_0Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
M00_AXI_bid[2]2
grayscale_axi_mem_intercon_0Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
M00_AXI_bid[1]2
grayscale_axi_mem_intercon_0Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
M00_AXI_rid[2]2
grayscale_axi_mem_intercon_0Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
M00_AXI_rid[1]2
grayscale_axi_mem_intercon_0Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2

S00_ACLK2
grayscale_axi_mem_intercon_0Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
S00_ARESETN2
grayscale_axi_mem_intercon_0Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2

S01_ACLK2
grayscale_axi_mem_intercon_0Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
S01_ARESETN2
grayscale_axi_mem_intercon_0Z8-7129h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Cross Boundary and Area Optimization : Time (s): cpu = 00:00:15 ; elapsed = 00:00:17 . Memory (MB): peak = 1246.680 ; gain = 582.492
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
@
%s
*synth2(
&Start Applying XDC Timing Constraints
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Applying XDC Timing Constraints : Time (s): cpu = 00:00:20 ; elapsed = 00:00:23 . Memory (MB): peak = 1410.906 ; gain = 746.719
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
4
%s
*synth2
Start Timing Optimization
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2
}Finished Timing Optimization : Time (s): cpu = 00:00:20 ; elapsed = 00:00:23 . Memory (MB): peak = 1411.277 ; gain = 747.090
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
3
%s
*synth2
Start Technology Mapping
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2~
|Finished Technology Mapping : Time (s): cpu = 00:00:20 ; elapsed = 00:00:23 . Memory (MB): peak = 1421.402 ; gain = 757.215
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
-
%s
*synth2
Start IO Insertion
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
?
%s
*synth2'
%Start Flattening Before IO Insertion
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
B
%s
*synth2*
(Finished Flattening Before IO Insertion
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
6
%s
*synth2
Start Final Netlist Cleanup
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
9
%s
*synth2!
Finished Final Netlist Cleanup
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2x
vFinished IO Insertion : Time (s): cpu = 00:00:24 ; elapsed = 00:00:26 . Memory (MB): peak = 1637.988 ; gain = 973.801
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
=
%s
*synth2%
#Start Renaming Generated Instances
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Renaming Generated Instances : Time (s): cpu = 00:00:24 ; elapsed = 00:00:26 . Memory (MB): peak = 1637.988 ; gain = 973.801
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
:
%s
*synth2"
 Start Rebuilding User Hierarchy
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Rebuilding User Hierarchy : Time (s): cpu = 00:00:24 ; elapsed = 00:00:26 . Memory (MB): peak = 1637.988 ; gain = 973.801
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
9
%s
*synth2!
Start Renaming Generated Ports
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Renaming Generated Ports : Time (s): cpu = 00:00:24 ; elapsed = 00:00:26 . Memory (MB): peak = 1637.988 ; gain = 973.801
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
;
%s
*synth2#
!Start Handling Custom Attributes
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Handling Custom Attributes : Time (s): cpu = 00:00:24 ; elapsed = 00:00:26 . Memory (MB): peak = 1637.988 ; gain = 973.801
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
8
%s
*synth2 
Start Renaming Generated Nets
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Renaming Generated Nets : Time (s): cpu = 00:00:24 ; elapsed = 00:00:26 . Memory (MB): peak = 1637.988 ; gain = 973.801
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
9
%s
*synth2!
Start Writing Synthesis Report
h p
x
� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
/
%s
*synth2

Report BlackBoxes: 
h p
x
� 
X
%s
*synth2@
>+------+-----------------------------------------+----------+
h p
x
� 
X
%s
*synth2@
>|      |BlackBox name                            |Instances |
h p
x
� 
X
%s
*synth2@
>+------+-----------------------------------------+----------+
h p
x
� 
X
%s
*synth2@
>|1     |grayscale_axi_dma_0_1                    |         1|
h p
x
� 
X
%s
*synth2@
>|2     |grayscale_axi_mem_intercon_imp_xbar_0    |         1|
h p
x
� 
X
%s
*synth2@
>|3     |grayscale_axi_mem_intercon_imp_auto_pc_0 |         1|
h p
x
� 
X
%s
*synth2@
>|4     |grayscale_axi_mem_intercon_imp_auto_us_0 |         1|
h p
x
� 
X
%s
*synth2@
>|5     |grayscale_axi_mem_intercon_imp_auto_us_1 |         1|
h p
x
� 
X
%s
*synth2@
>|6     |grayscale_axi_smc_0                      |         1|
h p
x
� 
X
%s
*synth2@
>|7     |grayscale_grayscale_kernel_0_2           |         1|
h p
x
� 
X
%s
*synth2@
>|8     |grayscale_processing_system7_0_1         |         1|
h p
x
� 
X
%s
*synth2@
>|9     |grayscale_rst_ps7_0_100M_0               |         1|
h p
x
� 
X
%s
*synth2@
>+------+-----------------------------------------+----------+
h p
x
� 
/
%s*synth2

Report Cell Usage: 
h px� 
R
%s*synth2:
8+------+---------------------------------------+------+
h px� 
R
%s*synth2:
8|      |Cell                                   |Count |
h px� 
R
%s*synth2:
8+------+---------------------------------------+------+
h px� 
R
%s*synth2:
8|1     |grayscale_axi_dma_0                    |     1|
h px� 
R
%s*synth2:
8|2     |grayscale_axi_mem_intercon_imp_auto_pc |     1|
h px� 
R
%s*synth2:
8|3     |grayscale_axi_mem_intercon_imp_auto_us |     2|
h px� 
R
%s*synth2:
8|5     |grayscale_axi_mem_intercon_imp_xbar    |     1|
h px� 
R
%s*synth2:
8|6     |grayscale_axi_smc                      |     1|
h px� 
R
%s*synth2:
8|7     |grayscale_grayscale_kernel_0           |     1|
h px� 
R
%s*synth2:
8|8     |grayscale_processing_system7_0         |     1|
h px� 
R
%s*synth2:
8|9     |grayscale_rst_ps7_0_100M               |     1|
h px� 
R
%s*synth2:
8+------+---------------------------------------+------+
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Writing Synthesis Report : Time (s): cpu = 00:00:24 ; elapsed = 00:00:26 . Memory (MB): peak = 1637.988 ; gain = 973.801
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
a
%s
*synth2I
GSynthesis finished with 0 errors, 0 critical warnings and 11 warnings.
h p
x
� 
�
%s
*synth2�
Synthesis Optimization Runtime : Time (s): cpu = 00:00:16 ; elapsed = 00:00:24 . Memory (MB): peak = 1637.988 ; gain = 973.801
h p
x
� 
�
%s
*synth2�
�Synthesis Optimization Complete : Time (s): cpu = 00:00:24 ; elapsed = 00:00:26 . Memory (MB): peak = 1637.988 ; gain = 973.801
h p
x
� 
B
 Translating synthesized netlist
350*projectZ1-571h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
Netlist sorting complete. 2

00:00:002
00:00:00.0252

1637.9882
0.000Z17-268h px� 
K
)Preparing netlist for logic optimization
349*projectZ1-570h px� 
Q
)Pushed %s inverter(s) to %s load pin(s).
98*opt2
02
0Z31-138h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
Netlist sorting complete. 2

00:00:002

00:00:002

1643.0942
0.000Z17-268h px� 
l
!Unisim Transformation Summary:
%s111*project2'
%No Unisim elements were transformed.
Z1-111h px� 
V
%Synth Design complete | Checksum: %s
562*	vivadotcl2

1e7ccad6Z4-1430h px� 
C
Releasing license: %s
83*common2
	SynthesisZ17-83h px� 

G%s Infos, %s Warnings, %s Critical Warnings and %s Errors encountered.
28*	vivadotcl2
452
632
02
0Z4-41h px� 
L
%s completed successfully
29*	vivadotcl2
synth_designZ4-42h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
synth_design: 2

00:00:282

00:00:342

1647.0782

1236.344Z17-268h px� 
�
 The %s '%s' has been generated.
621*common2

checkpoint2s
qD:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/synth_1/grayscale_wrapper.dcpZ17-1381h px� 
�
Executing command : %s
56330*	planAhead2m
kreport_utilization -file grayscale_wrapper_utilization_synth.rpt -pb grayscale_wrapper_utilization_synth.pbZ12-24828h px� 
\
Exiting %s at %s...
206*common2
Vivado2
Thu Mar  6 03:30:25 2025Z17-206h px� 


End Record