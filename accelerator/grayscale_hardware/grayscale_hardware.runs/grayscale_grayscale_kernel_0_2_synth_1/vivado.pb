
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
t
-IPCACHE: Running cache check for IP inst: %s
2546*coregen2 
grayscale_grayscale_kernel_0_2Z19-6924h px� 
�
Command: %s
53*	vivadotcl2s
qsynth_design -top grayscale_grayscale_kernel_0_2 -part xc7z020clg400-1 -incremental_mode off -mode out_of_contextZ4-113h px� 
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
26040Z8-7075h px� 
�
%s*synth2{
yStarting RTL Elaboration : Time (s): cpu = 00:00:09 ; elapsed = 00:00:16 . Memory (MB): peak = 1082.031 ; gain = 469.133
h px� 
�
synthesizing module '%s'%s4497*oasys2 
grayscale_grayscale_kernel_0_22
 2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_grayscale_kernel_0_2/synth/grayscale_grayscale_kernel_0_2.v2
538@Z8-6157h px� 
�
synthesizing module '%s'%s4497*oasys2
grayscale_kernel2
 2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel.v2
98@Z8-6157h px� 
�
synthesizing module '%s'%s4497*oasys2L
Jgrayscale_kernel_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_22
 2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2.v2
98@Z8-6157h px� 
�
synthesizing module '%s'%s4497*oasys21
/grayscale_kernel_fadd_32ns_32ns_32_5_full_dsp_12
 2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_fadd_32ns_32ns_32_5_full_dsp_1.v2
98@Z8-6157h px� 
�
synthesizing module '%s'%s4497*oasys24
2grayscale_kernel_fadd_32ns_32ns_32_5_full_dsp_1_ip2
 2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/ip/grayscale_kernel_fadd_32ns_32ns_32_5_full_dsp_1_ip.v2
538@Z8-6157h px� 
�
synthesizing module '%s'%s4497*oasys2
LUT12
 29
5C:/Xilinx/Vivado/2024.2/scripts/rt/data/unisim_comp.v2	
840878@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2
LUT12
 2
02
129
5C:/Xilinx/Vivado/2024.2/scripts/rt/data/unisim_comp.v2	
840878@Z8-6155h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys24
2grayscale_kernel_fadd_32ns_32ns_32_5_full_dsp_1_ip2
 2
02
12�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/ip/grayscale_kernel_fadd_32ns_32ns_32_5_full_dsp_1_ip.v2
538@Z8-6155h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys21
/grayscale_kernel_fadd_32ns_32ns_32_5_full_dsp_12
 2
02
12�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_fadd_32ns_32ns_32_5_full_dsp_1.v2
98@Z8-6155h px� 
�
synthesizing module '%s'%s4497*oasys20
.grayscale_kernel_fmul_32ns_32ns_32_4_max_dsp_12
 2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_fmul_32ns_32ns_32_4_max_dsp_1.v2
98@Z8-6157h px� 
�
synthesizing module '%s'%s4497*oasys23
1grayscale_kernel_fmul_32ns_32ns_32_4_max_dsp_1_ip2
 2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/ip/grayscale_kernel_fmul_32ns_32ns_32_4_max_dsp_1_ip.v2
538@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys23
1grayscale_kernel_fmul_32ns_32ns_32_4_max_dsp_1_ip2
 2
02
12�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/ip/grayscale_kernel_fmul_32ns_32ns_32_4_max_dsp_1_ip.v2
538@Z8-6155h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys20
.grayscale_kernel_fmul_32ns_32ns_32_4_max_dsp_12
 2
02
12�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_fmul_32ns_32ns_32_4_max_dsp_1.v2
98@Z8-6155h px� 
�
synthesizing module '%s'%s4497*oasys2,
*grayscale_kernel_uitofp_32ns_32_6_no_dsp_12
 2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_uitofp_32ns_32_6_no_dsp_1.v2
98@Z8-6157h px� 
�
synthesizing module '%s'%s4497*oasys2/
-grayscale_kernel_uitofp_32ns_32_6_no_dsp_1_ip2
 2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/ip/grayscale_kernel_uitofp_32ns_32_6_no_dsp_1_ip.v2
538@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2/
-grayscale_kernel_uitofp_32ns_32_6_no_dsp_1_ip2
 2
02
12�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/ip/grayscale_kernel_uitofp_32ns_32_6_no_dsp_1_ip.v2
538@Z8-6155h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2,
*grayscale_kernel_uitofp_32ns_32_6_no_dsp_12
 2
02
12�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_uitofp_32ns_32_6_no_dsp_1.v2
98@Z8-6155h px� 
�
synthesizing module '%s'%s4497*oasys29
7grayscale_kernel_flow_control_loop_pipe_sequential_init2
 2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_flow_control_loop_pipe_sequential_init.v2
118@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys29
7grayscale_kernel_flow_control_loop_pipe_sequential_init2
 2
02
12�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_flow_control_loop_pipe_sequential_init.v2
118@Z8-6155h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2L
Jgrayscale_kernel_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_22
 2
02
12�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_grayscale_kernel_Pipeline_VITIS_LOOP_38_1_VITIS_LOOP_39_2.v2
98@Z8-6155h px� 
�
synthesizing module '%s'%s4497*oasys2 
grayscale_kernel_control_s_axi2
 2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_control_s_axi.v2
98@Z8-6157h px� 
�
-case statement is not full and has no default155*oasys2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_control_s_axi.v2
2118@Z8-155h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2 
grayscale_kernel_control_s_axi2
 2
02
12�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_control_s_axi.v2
98@Z8-6155h px� 
�
synthesizing module '%s'%s4497*oasys2'
%grayscale_kernel_mul_31ns_31ns_62_2_12
 2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_mul_31ns_31ns_62_2_1.v2
58@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2'
%grayscale_kernel_mul_31ns_31ns_62_2_12
 2
02
12�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_mul_31ns_31ns_62_2_1.v2
58@Z8-6155h px� 
�
synthesizing module '%s'%s4497*oasys2 
grayscale_kernel_regslice_both2
 2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_regslice_both.v2
88@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2 
grayscale_kernel_regslice_both2
 2
02
12�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_regslice_both.v2
88@Z8-6155h px� 
�
synthesizing module '%s'%s4497*oasys20
.grayscale_kernel_regslice_both__parameterized02
 2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_regslice_both.v2
88@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys20
.grayscale_kernel_regslice_both__parameterized02
 2
02
12�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_regslice_both.v2
88@Z8-6155h px� 
�
synthesizing module '%s'%s4497*oasys20
.grayscale_kernel_regslice_both__parameterized12
 2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_regslice_both.v2
88@Z8-6157h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys20
.grayscale_kernel_regslice_both__parameterized12
 2
02
12�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_regslice_both.v2
88@Z8-6155h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2
grayscale_kernel2
 2
02
12�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel.v2
98@Z8-6155h px� 
�
'done synthesizing module '%s'%s (%s#%s)4495*oasys2 
grayscale_grayscale_kernel_0_22
 2
02
12�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_grayscale_kernel_0_2/synth/grayscale_grayscale_kernel_0_2.v2
538@Z8-6155h px� 
�
+Unused sequential element %s was removed. 
4326*oasys2
int_ap_done_reg2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_control_s_axi.v2
2768@Z8-6014h px� 
s
@FSM extraction disabled for register '%s' through user attribute3641*oasys2
ap_CS_fsm_regZ8-4490h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
reset2'
%grayscale_kernel_mul_31ns_31ns_62_2_1Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
	AWADDR[1]2 
grayscale_kernel_control_s_axiZ8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
	AWADDR[0]2 
grayscale_kernel_control_s_axiZ8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
CLK2'
%xbip_pipe_v3_0_10_viv__parameterized3Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
CE2'
%xbip_pipe_v3_0_10_viv__parameterized3Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SCLR2'
%xbip_pipe_v3_0_10_viv__parameterized3Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SSET2'
%xbip_pipe_v3_0_10_viv__parameterized3Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SINIT2'
%xbip_pipe_v3_0_10_viv__parameterized3Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
CLK2(
&xbip_pipe_v3_0_10_viv__parameterized55Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
CE2(
&xbip_pipe_v3_0_10_viv__parameterized55Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SCLR2(
&xbip_pipe_v3_0_10_viv__parameterized55Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SSET2(
&xbip_pipe_v3_0_10_viv__parameterized55Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SINIT2(
&xbip_pipe_v3_0_10_viv__parameterized55Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SCLR2(
&xbip_pipe_v3_0_10_viv__parameterized17Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SSET2(
&xbip_pipe_v3_0_10_viv__parameterized17Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SINIT2(
&xbip_pipe_v3_0_10_viv__parameterized17Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
DEC_OP_STATE[11]2
flt_dec_op__parameterized0Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
DEC_OP_STATE[10]2
flt_dec_op__parameterized0Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
DEC_OP_STATE[9]2
flt_dec_op__parameterized0Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
DEC_OP_STATE[8]2
flt_dec_op__parameterized0Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
DIVIDE_BY_ZERO_IN2
flt_dec_op__parameterized0Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SCLR2(
&xbip_pipe_v3_0_10_viv__parameterized33Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SSET2(
&xbip_pipe_v3_0_10_viv__parameterized33Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SINIT2(
&xbip_pipe_v3_0_10_viv__parameterized33Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SCLR2(
&xbip_pipe_v3_0_10_viv__parameterized87Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SSET2(
&xbip_pipe_v3_0_10_viv__parameterized87Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SINIT2(
&xbip_pipe_v3_0_10_viv__parameterized87Z8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[31]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[30]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[29]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[28]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[27]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[26]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[25]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[24]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[23]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[22]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[21]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[20]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[19]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[18]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[17]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[16]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[15]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[14]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[13]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[12]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[11]2
fix_to_flt_conv_expZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[10]2
fix_to_flt_conv_expZ8-7129h px� 
z
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[9]2
fix_to_flt_conv_expZ8-7129h px� 
z
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[8]2
fix_to_flt_conv_expZ8-7129h px� 
z
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[7]2
fix_to_flt_conv_expZ8-7129h px� 
z
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[6]2
fix_to_flt_conv_expZ8-7129h px� 
z
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[5]2
fix_to_flt_conv_expZ8-7129h px� 
z
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[4]2
fix_to_flt_conv_expZ8-7129h px� 
z
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[3]2
fix_to_flt_conv_expZ8-7129h px� 
z
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[2]2
fix_to_flt_conv_expZ8-7129h px� 
z
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[1]2
fix_to_flt_conv_expZ8-7129h px� 
z
9Port %s in module %s is either unconnected or has no load4866*oasys2
A[0]2
fix_to_flt_conv_expZ8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SCLR2(
&xbip_pipe_v3_0_10_viv__parameterized83Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SSET2(
&xbip_pipe_v3_0_10_viv__parameterized83Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SINIT2(
&xbip_pipe_v3_0_10_viv__parameterized83Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SCLR2(
&xbip_pipe_v3_0_10_viv__parameterized85Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SSET2(
&xbip_pipe_v3_0_10_viv__parameterized85Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SINIT2(
&xbip_pipe_v3_0_10_viv__parameterized85Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
B[11]2
carry_chain__parameterized9Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
CLK2(
&xbip_pipe_v3_0_10_viv__parameterized81Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
CE2(
&xbip_pipe_v3_0_10_viv__parameterized81Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SCLR2(
&xbip_pipe_v3_0_10_viv__parameterized81Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SSET2(
&xbip_pipe_v3_0_10_viv__parameterized81Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SINIT2(
&xbip_pipe_v3_0_10_viv__parameterized81Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
CLK2(
&xbip_pipe_v3_0_10_viv__parameterized79Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
CE2(
&xbip_pipe_v3_0_10_viv__parameterized79Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SCLR2(
&xbip_pipe_v3_0_10_viv__parameterized79Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SSET2(
&xbip_pipe_v3_0_10_viv__parameterized79Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SINIT2(
&xbip_pipe_v3_0_10_viv__parameterized79Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
CLK2(
&xbip_pipe_v3_0_10_viv__parameterized25Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
CE2(
&xbip_pipe_v3_0_10_viv__parameterized25Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SCLR2(
&xbip_pipe_v3_0_10_viv__parameterized25Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SSET2(
&xbip_pipe_v3_0_10_viv__parameterized25Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SINIT2(
&xbip_pipe_v3_0_10_viv__parameterized25Z8-7129h px� 
|
9Port %s in module %s is either unconnected or has no load4866*oasys2
NORMALIZE[1]2
flt_round_bitZ8-7129h px� 
{
9Port %s in module %s is either unconnected or has no load4866*oasys2
FIXED_POINT2
flt_round_bitZ8-7129h px� 
}
9Port %s in module %s is either unconnected or has no load4866*oasys2
FIX_MANT_SIGN2
flt_round_bitZ8-7129h px� 
t
9Port %s in module %s is either unconnected or has no load4866*oasys2
SIGN2
flt_round_bitZ8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
CLK2(
&xbip_pipe_v3_0_10_viv__parameterized51Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
CE2(
&xbip_pipe_v3_0_10_viv__parameterized51Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SCLR2(
&xbip_pipe_v3_0_10_viv__parameterized51Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SSET2(
&xbip_pipe_v3_0_10_viv__parameterized51Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SINIT2(
&xbip_pipe_v3_0_10_viv__parameterized51Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
CLK2(
&xbip_pipe_v3_0_10_viv__parameterized43Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
CE2(
&xbip_pipe_v3_0_10_viv__parameterized43Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SCLR2(
&xbip_pipe_v3_0_10_viv__parameterized43Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SSET2(
&xbip_pipe_v3_0_10_viv__parameterized43Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SINIT2(
&xbip_pipe_v3_0_10_viv__parameterized43Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
CLK2(
&xbip_pipe_v3_0_10_viv__parameterized29Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
CE2(
&xbip_pipe_v3_0_10_viv__parameterized29Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SCLR2(
&xbip_pipe_v3_0_10_viv__parameterized29Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SSET2(
&xbip_pipe_v3_0_10_viv__parameterized29Z8-7129h px� 
�
9Port %s in module %s is either unconnected or has no load4866*oasys2
SINIT2(
&xbip_pipe_v3_0_10_viv__parameterized29Z8-7129h px� 
�
�Message '%s' appears more than %s times and has been disabled. User can change this message limit to see more message instances.
14*common2
Synth 8-71292
100Z17-14h px� 
�
%s*synth2{
yFinished RTL Elaboration : Time (s): cpu = 00:00:19 ; elapsed = 00:00:29 . Memory (MB): peak = 1337.516 ; gain = 724.617
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
�Finished Handling Custom Attributes : Time (s): cpu = 00:00:19 ; elapsed = 00:00:30 . Memory (MB): peak = 1337.516 ; gain = 724.617
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
�Finished RTL Optimization Phase 1 : Time (s): cpu = 00:00:19 ; elapsed = 00:00:30 . Memory (MB): peak = 1337.516 ; gain = 724.617
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
00:00:00.2862

1337.5162
0.000Z17-268h px� 
U
-Analyzing %s Unisim elements for replacement
17*netlist2
457Z29-17h px� 
X
2Unisim Transformation completed in %s CPU seconds
28*netlist2
0Z29-28h px� 
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
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_grayscale_kernel_0_2/constraints/grayscale_kernel_ooc.xdc2
inst	8Z20-848h px� 
�
-Finished Parsing XDC File [%s] for cell '%s'
847*designutils2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ip/grayscale_grayscale_kernel_0_2/constraints/grayscale_kernel_ooc.xdc2
inst	8Z20-847h px� 
�
Parsing XDC File [%s]
179*designutils2�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/grayscale_grayscale_kernel_0_2_synth_1/dont_touch.xdc8Z20-179h px� 
�
Finished Parsing XDC File [%s]
178*designutils2�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/grayscale_grayscale_kernel_0_2_synth_1/dont_touch.xdc8Z20-178h px� 
H
&Completed Processing XDC Constraints

245*projectZ1-263h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
Netlist sorting complete. 2

00:00:002
00:00:00.0072

1337.5162
0.000Z17-268h px� 
�
!Unisim Transformation Summary:
%s111*project2H
F  A total of 6 instances were transformed.
  FDE => FDRE: 6 instances
Z1-111h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2"
 Constraint Validation Runtime : 2

00:00:002
00:00:00.1732

1338.2732
0.758Z17-268h px� 
Q
!Incremental synthesis strategy %s2274*designutils2
offZ20-5008h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
Finished Constraint Validation : Time (s): cpu = 00:00:25 ; elapsed = 00:00:46 . Memory (MB): peak = 1338.273 ; gain = 725.375
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
�Finished Loading Part and Timing Information : Time (s): cpu = 00:00:25 ; elapsed = 00:00:46 . Memory (MB): peak = 1338.273 ; gain = 725.375
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
�Finished applying 'set_property' XDC Constraints : Time (s): cpu = 00:00:25 ; elapsed = 00:00:46 . Memory (MB): peak = 1338.273 ; gain = 725.375
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
3inferred FSM for state register '%s' in module '%s'802*oasys2

wstate_reg2 
grayscale_kernel_control_s_axiZ8-802h px� 
�
3inferred FSM for state register '%s' in module '%s'802*oasys2

rstate_reg2 
grayscale_kernel_control_s_axiZ8-802h px� 
�
3inferred FSM for state register '%s' in module '%s'802*oasys2
	state_reg2 
grayscale_kernel_regslice_bothZ8-802h px� 
�
3inferred FSM for state register '%s' in module '%s'802*oasys2
	state_reg20
.grayscale_kernel_regslice_both__parameterized0Z8-802h px� 
�
3inferred FSM for state register '%s' in module '%s'802*oasys2
	state_reg20
.grayscale_kernel_regslice_both__parameterized1Z8-802h px� 
s
@FSM extraction disabled for register '%s' through user attribute3641*oasys2
ap_CS_fsm_regZ8-4490h px� 
s
@FSM extraction disabled for register '%s' through user attribute3641*oasys2
ap_CS_fsm_regZ8-4490h px� 
s
@FSM extraction disabled for register '%s' through user attribute3641*oasys2
ap_CS_fsm_regZ8-4490h px� 
~
%s
*synth2f
d---------------------------------------------------------------------------------------------------
h p
x
� 
z
%s
*synth2b
`                   State |                     New Encoding |                Previous Encoding 
h p
x
� 
~
%s
*synth2f
d---------------------------------------------------------------------------------------------------
h p
x
� 
y
%s
*synth2a
_                  iSTATE |                             0001 |                               11
h p
x
� 

%s
*synth2
*
h p
x
� 
y
%s
*synth2a
_                  WRIDLE |                             0010 |                               00
h p
x
� 
y
%s
*synth2a
_                  WRDATA |                             0100 |                               01
h p
x
� 
y
%s
*synth2a
_                  WRRESP |                             1000 |                               10
h p
x
� 
~
%s
*synth2f
d---------------------------------------------------------------------------------------------------
h p
x
� 
�
Gencoded FSM with state register '%s' using encoding '%s' in module '%s'3353*oasys2

wstate_reg2	
one-hot2 
grayscale_kernel_control_s_axiZ8-3354h px� 
~
%s
*synth2f
d---------------------------------------------------------------------------------------------------
h p
x
� 
z
%s
*synth2b
`                   State |                     New Encoding |                Previous Encoding 
h p
x
� 
~
%s
*synth2f
d---------------------------------------------------------------------------------------------------
h p
x
� 
y
%s
*synth2a
_                  iSTATE |                              001 |                               10
h p
x
� 

%s
*synth2
*
h p
x
� 
y
%s
*synth2a
_                  RDIDLE |                              010 |                               00
h p
x
� 
y
%s
*synth2a
_                  RDDATA |                              100 |                               01
h p
x
� 
~
%s
*synth2f
d---------------------------------------------------------------------------------------------------
h p
x
� 
�
Gencoded FSM with state register '%s' using encoding '%s' in module '%s'3353*oasys2

rstate_reg2	
one-hot2 
grayscale_kernel_control_s_axiZ8-3354h px� 
~
%s
*synth2f
d---------------------------------------------------------------------------------------------------
h p
x
� 
z
%s
*synth2b
`                   State |                     New Encoding |                Previous Encoding 
h p
x
� 
~
%s
*synth2f
d---------------------------------------------------------------------------------------------------
h p
x
� 
y
%s
*synth2a
_                  iSTATE |                               00 |                               00
h p
x
� 

%s
*synth2
*
h p
x
� 
y
%s
*synth2a
_                    ZERO |                               01 |                               10
h p
x
� 
y
%s
*synth2a
_                     ONE |                               11 |                               11
h p
x
� 
y
%s
*synth2a
_                     TWO |                               10 |                               01
h p
x
� 
~
%s
*synth2f
d---------------------------------------------------------------------------------------------------
h p
x
� 
�
Gencoded FSM with state register '%s' using encoding '%s' in module '%s'3353*oasys2
	state_reg2

sequential2 
grayscale_kernel_regslice_bothZ8-3354h px� 
~
%s
*synth2f
d---------------------------------------------------------------------------------------------------
h p
x
� 
z
%s
*synth2b
`                   State |                     New Encoding |                Previous Encoding 
h p
x
� 
~
%s
*synth2f
d---------------------------------------------------------------------------------------------------
h p
x
� 
y
%s
*synth2a
_                  iSTATE |                               00 |                               00
h p
x
� 

%s
*synth2
*
h p
x
� 
y
%s
*synth2a
_                    ZERO |                               01 |                               10
h p
x
� 
y
%s
*synth2a
_                     ONE |                               11 |                               11
h p
x
� 
y
%s
*synth2a
_                     TWO |                               10 |                               01
h p
x
� 
~
%s
*synth2f
d---------------------------------------------------------------------------------------------------
h p
x
� 
�
Gencoded FSM with state register '%s' using encoding '%s' in module '%s'3353*oasys2
	state_reg2

sequential20
.grayscale_kernel_regslice_both__parameterized0Z8-3354h px� 
~
%s
*synth2f
d---------------------------------------------------------------------------------------------------
h p
x
� 
z
%s
*synth2b
`                   State |                     New Encoding |                Previous Encoding 
h p
x
� 
~
%s
*synth2f
d---------------------------------------------------------------------------------------------------
h p
x
� 
y
%s
*synth2a
_                  iSTATE |                               00 |                               00
h p
x
� 

%s
*synth2
*
h p
x
� 
y
%s
*synth2a
_                    ZERO |                               01 |                               10
h p
x
� 
y
%s
*synth2a
_                     ONE |                               11 |                               11
h p
x
� 
y
%s
*synth2a
_                     TWO |                               10 |                               01
h p
x
� 
~
%s
*synth2f
d---------------------------------------------------------------------------------------------------
h p
x
� 
�
Gencoded FSM with state register '%s' using encoding '%s' in module '%s'3353*oasys2
	state_reg2

sequential20
.grayscale_kernel_regslice_both__parameterized1Z8-3354h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished RTL Optimization Phase 2 : Time (s): cpu = 00:00:26 ; elapsed = 00:00:49 . Memory (MB): peak = 1338.273 ; gain = 725.375
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
$decloning instance '%s' (%s) to '%s'223*oasys2�
�grayscale_kernel_fadd_32ns_32ns_32_5_full_dsp_1:/grayscale_kernel_fadd_32ns_32ns_32_5_full_dsp_1_ip_u/inst/i_synth/ADDSUB_OP.ADDSUB/DSP.OP/DSP48E1_BODY.ALIGN_ADD/SUB_DELAY2
delay__parameterized02�
�grayscale_kernel_fadd_32ns_32ns_32_5_full_dsp_1:/grayscale_kernel_fadd_32ns_32ns_32_5_full_dsp_1_ip_u/inst/i_synth/ADDSUB_OP.ADDSUB/DSP.OP/DSP48E1_BODY.ALIGN_ADD/SUB_ADD_IP_DELAYZ8-223h px� 
�
$decloning instance '%s' (%s) to '%s'223*oasys2�
�grayscale_kernel_fadd_32ns_32ns_32_5_full_dsp_1:/grayscale_kernel_fadd_32ns_32ns_32_5_full_dsp_1_ip_u/inst/i_synth/DELAY_DIVIDE_BY_ZERO2
delay__parameterized02�
�grayscale_kernel_fadd_32ns_32ns_32_5_full_dsp_1:/grayscale_kernel_fadd_32ns_32ns_32_5_full_dsp_1_ip_u/inst/i_synth/DELAY_ACCUM_OVERFLOWZ8-223h px� 
�
$decloning instance '%s' (%s) to '%s'223*oasys2�
�grayscale_kernel_fadd_32ns_32ns_32_5_full_dsp_1:/grayscale_kernel_fadd_32ns_32ns_32_5_full_dsp_1_ip_u/inst/i_synth/DELAY_DIVIDE_BY_ZERO2
delay__parameterized02�
�grayscale_kernel_fadd_32ns_32ns_32_5_full_dsp_1:/grayscale_kernel_fadd_32ns_32ns_32_5_full_dsp_1_ip_u/inst/i_synth/DELAY_ACCUM_INPUT_OVERFLOWZ8-223h px� 
�
$decloning instance '%s' (%s) to '%s'223*oasys2�
�grayscale_kernel_fmul_32ns_32ns_32_4_max_dsp_1:/grayscale_kernel_fmul_32ns_32ns_32_4_max_dsp_1_ip_u/inst/i_synth/DELAY_DIVIDE_BY_ZERO2
delay__parameterized02�
�grayscale_kernel_fmul_32ns_32ns_32_4_max_dsp_1:/grayscale_kernel_fmul_32ns_32ns_32_4_max_dsp_1_ip_u/inst/i_synth/DELAY_ACCUM_OVERFLOWZ8-223h px� 
�
$decloning instance '%s' (%s) to '%s'223*oasys2�
�grayscale_kernel_fmul_32ns_32ns_32_4_max_dsp_1:/grayscale_kernel_fmul_32ns_32ns_32_4_max_dsp_1_ip_u/inst/i_synth/DELAY_DIVIDE_BY_ZERO2
delay__parameterized02�
�grayscale_kernel_fmul_32ns_32ns_32_4_max_dsp_1:/grayscale_kernel_fmul_32ns_32ns_32_4_max_dsp_1_ip_u/inst/i_synth/DELAY_ACCUM_INPUT_OVERFLOWZ8-223h px� 
�
$decloning instance '%s' (%s) to '%s'223*oasys2�
�grayscale_kernel_uitofp_32ns_32_6_no_dsp_1:/grayscale_kernel_uitofp_32ns_32_6_no_dsp_1_ip_u/inst/i_synth/FIX_TO_FLT_OP.SPD.OP/ROUND/EXTRA_LSB_BIT_DEL2
delay__parameterized02�
�grayscale_kernel_uitofp_32ns_32_6_no_dsp_1:/grayscale_kernel_uitofp_32ns_32_6_no_dsp_1_ip_u/inst/i_synth/FIX_TO_FLT_OP.SPD.OP/ROUND/EXTRA_LSB_RND1_DELZ8-223h px� 
�
$decloning instance '%s' (%s) to '%s'223*oasys2�
�grayscale_kernel_uitofp_32ns_32_6_no_dsp_1:/grayscale_kernel_uitofp_32ns_32_6_no_dsp_1_ip_u/inst/i_synth/FIX_TO_FLT_OP.SPD.OP/ROUND/EXP_INC_DELAY2
delay__parameterized02�
�grayscale_kernel_uitofp_32ns_32_6_no_dsp_1:/grayscale_kernel_uitofp_32ns_32_6_no_dsp_1_ip_u/inst/i_synth/FIX_TO_FLT_OP.SPD.OP/ROUND/EXP_INC_RND1_DELAYZ8-223h px� 
�
$decloning instance '%s' (%s) to '%s'223*oasys2{
ygrayscale_kernel_uitofp_32ns_32_6_no_dsp_1:/grayscale_kernel_uitofp_32ns_32_6_no_dsp_1_ip_u/inst/i_synth/DELAY_INVALID_OP2
delay__parameterized02
}grayscale_kernel_uitofp_32ns_32_6_no_dsp_1:/grayscale_kernel_uitofp_32ns_32_6_no_dsp_1_ip_u/inst/i_synth/DELAY_DIVIDE_BY_ZEROZ8-223h px� 
�
$decloning instance '%s' (%s) to '%s'223*oasys2{
ygrayscale_kernel_uitofp_32ns_32_6_no_dsp_1:/grayscale_kernel_uitofp_32ns_32_6_no_dsp_1_ip_u/inst/i_synth/DELAY_INVALID_OP2
delay__parameterized02y
wgrayscale_kernel_uitofp_32ns_32_6_no_dsp_1:/grayscale_kernel_uitofp_32ns_32_6_no_dsp_1_ip_u/inst/i_synth/DELAY_OVERFLOWZ8-223h px� 
�
$decloning instance '%s' (%s) to '%s'223*oasys2{
ygrayscale_kernel_uitofp_32ns_32_6_no_dsp_1:/grayscale_kernel_uitofp_32ns_32_6_no_dsp_1_ip_u/inst/i_synth/DELAY_INVALID_OP2
delay__parameterized02z
xgrayscale_kernel_uitofp_32ns_32_6_no_dsp_1:/grayscale_kernel_uitofp_32ns_32_6_no_dsp_1_ip_u/inst/i_synth/DELAY_UNDERFLOWZ8-223h px� 
�
$decloning instance '%s' (%s) to '%s'223*oasys2{
ygrayscale_kernel_uitofp_32ns_32_6_no_dsp_1:/grayscale_kernel_uitofp_32ns_32_6_no_dsp_1_ip_u/inst/i_synth/DELAY_INVALID_OP2
delay__parameterized02
}grayscale_kernel_uitofp_32ns_32_6_no_dsp_1:/grayscale_kernel_uitofp_32ns_32_6_no_dsp_1_ip_u/inst/i_synth/DELAY_ACCUM_OVERFLOWZ8-223h px� 
�
$decloning instance '%s' (%s) to '%s'223*oasys2{
ygrayscale_kernel_uitofp_32ns_32_6_no_dsp_1:/grayscale_kernel_uitofp_32ns_32_6_no_dsp_1_ip_u/inst/i_synth/DELAY_INVALID_OP2
delay__parameterized02�
�grayscale_kernel_uitofp_32ns_32_6_no_dsp_1:/grayscale_kernel_uitofp_32ns_32_6_no_dsp_1_ip_u/inst/i_synth/DELAY_ACCUM_INPUT_OVERFLOWZ8-223h px� 
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
�
+Unused sequential element %s was removed. 
4326*oasys2&
$mul_31ns_31ns_62_2_1_U24/tmp_product2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_mul_31ns_31ns_62_2_1.v2
478@Z8-6014h px� 
�
+Unused sequential element %s was removed. 
4326*oasys2$
"mul_31ns_31ns_62_2_1_U24/buff0_reg2�
�d:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.gen/sources_1/bd/grayscale/ipshared/5774/hdl/verilog/grayscale_kernel_mul_31ns_31ns_62_2_1.v2
478@Z8-6014h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2S
QADDSUB_OP.ADDSUB/DSP.OP/DSP48E1_BODY.NORM_RND/FULL_USAGE_DSP.LOD/dist_norm_reg[1]2
floating_point_v7_1_19_viv__1Z8-3332h px�
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2S
QADDSUB_OP.ADDSUB/DSP.OP/DSP48E1_BODY.NORM_RND/FULL_USAGE_DSP.LOD/dist_norm_reg[0]2
floating_point_v7_1_19_viv__1Z8-3332h px�
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2S
QADDSUB_OP.ADDSUB/DSP.OP/DSP48E1_BODY.NORM_RND/FULL_USAGE_DSP.LOD/dist_norm_reg[1]2
floating_point_v7_1_19_vivZ8-3332h px�
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2S
QADDSUB_OP.ADDSUB/DSP.OP/DSP48E1_BODY.NORM_RND/FULL_USAGE_DSP.LOD/dist_norm_reg[0]2
floating_point_v7_1_19_vivZ8-3332h px�
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2*
(control_s_axi_U/FSM_onehot_wstate_reg[0]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2*
(control_s_axi_U/FSM_onehot_rstate_reg[0]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2C
Aregslice_both_input_stream_V_last_V_U/FSM_sequential_state_reg[1]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2C
Aregslice_both_input_stream_V_last_V_U/FSM_sequential_state_reg[0]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[47]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[46]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[45]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[44]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[43]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[42]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[41]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[40]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[39]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[38]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[37]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[36]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[35]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[34]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[33]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[32]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[31]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[30]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[29]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[28]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[27]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[26]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[25]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[24]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[23]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[22]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[21]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[20]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[19]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[18]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2(
&mul_31ns_31ns_62_2_1_U24/buff0_reg[17]2
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[47]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[46]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[45]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[44]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[43]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[42]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[41]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[40]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[39]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[38]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[37]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[36]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[35]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[34]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[33]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[32]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[31]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[30]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[29]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[28]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[27]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[26]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[25]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[24]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[23]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[22]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[21]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[20]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[19]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[18]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2+
)mul_31ns_31ns_62_2_1_U24/buff0_reg[17]__02
grayscale_kernelZ8-3332h px� 
�
ESequential element (%s) is unused and will be removed from module %s.3332*oasys2*
(mul_31ns_31ns_62_2_1_U24/tmp_product__182
grayscale_kernelZ8-3332h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
�
%s*synth2�
�Finished Cross Boundary and Area Optimization : Time (s): cpu = 00:00:31 ; elapsed = 00:01:03 . Memory (MB): peak = 1338.273 ; gain = 725.375
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
r
%s
*synth2Z
X Sort Area is  mul_31ns_31ns_62_2_1_U24/tmp_product_0 : 0 0 : 3137 5871 : Used 1 time 0
h p
x
� 
r
%s
*synth2Z
X Sort Area is  mul_31ns_31ns_62_2_1_U24/tmp_product_0 : 0 1 : 2734 5871 : Used 1 time 0
h p
x
� 
r
%s
*synth2Z
X Sort Area is  mul_31ns_31ns_62_2_1_U24/tmp_product_3 : 0 0 : 2667 5230 : Used 1 time 0
h p
x
� 
r
%s
*synth2Z
X Sort Area is  mul_31ns_31ns_62_2_1_U24/tmp_product_3 : 0 1 : 2563 5230 : Used 1 time 0
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
�Finished Applying XDC Timing Constraints : Time (s): cpu = 00:00:35 ; elapsed = 00:01:18 . Memory (MB): peak = 1499.402 ; gain = 886.504
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
}Finished Timing Optimization : Time (s): cpu = 00:00:37 ; elapsed = 00:01:20 . Memory (MB): peak = 1539.773 ; gain = 926.875
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
|Finished Technology Mapping : Time (s): cpu = 00:00:38 ; elapsed = 00:01:22 . Memory (MB): peak = 1544.117 ; gain = 931.219
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
%s*synth2y
wFinished IO Insertion : Time (s): cpu = 00:00:42 ; elapsed = 00:01:26 . Memory (MB): peak = 1732.797 ; gain = 1119.898
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
�Finished Renaming Generated Instances : Time (s): cpu = 00:00:42 ; elapsed = 00:01:26 . Memory (MB): peak = 1732.797 ; gain = 1119.898
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
�Finished Rebuilding User Hierarchy : Time (s): cpu = 00:00:43 ; elapsed = 00:01:27 . Memory (MB): peak = 1732.797 ; gain = 1119.898
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
�Finished Renaming Generated Ports : Time (s): cpu = 00:00:43 ; elapsed = 00:01:27 . Memory (MB): peak = 1732.797 ; gain = 1119.898
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
�Finished Handling Custom Attributes : Time (s): cpu = 00:00:43 ; elapsed = 00:01:27 . Memory (MB): peak = 1732.797 ; gain = 1119.898
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
�Finished Renaming Generated Nets : Time (s): cpu = 00:00:43 ; elapsed = 00:01:27 . Memory (MB): peak = 1732.797 ; gain = 1119.898
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
W
%s
*synth2?
=
DSP Final Report (the ' indicates corresponding REG is set)
h p
x
� 
�
%s
*synth2�
�+------------------------------------+-------------------+--------+--------+--------+--------+--------+------+------+------+------+-------+------+------+
h p
x
� 
�
%s
*synth2�
�|Module Name                         | DSP Mapping       | A Size | B Size | C Size | D Size | P Size | AREG | BREG | CREG | DREG | ADREG | MREG | PREG | 
h p
x
� 
�
%s
*synth2�
�+------------------------------------+-------------------+--------+--------+--------+--------+--------+------+------+------+------+-------+------+------+
h p
x
� 
�
%s
*synth2�
�|dsp48e1_wrapper_281                 | Dynamic           | -      | -      | -      | -      | 40     | -    | -    | -    | -    | 0     | 0    | 1    | 
h p
x
� 
�
%s
*synth2�
�|dsp48e1_wrapper__parameterized0_235 | C+A'*B'           | 24     | 16     | 35     | -      | 35     | 1    | 1    | 0    | -    | -     | 0    | 0    | 
h p
x
� 
�
%s
*synth2�
�|dsp48e1_wrapper                     | Dynamic           | -      | -      | -      | -      | 40     | -    | -    | -    | -    | 0     | 0    | 1    | 
h p
x
� 
�
%s
*synth2�
�|dsp48e1_wrapper__parameterized0     | C+A'*B'           | 24     | 16     | 35     | -      | 35     | 1    | 1    | 0    | -    | -     | 0    | 0    | 
h p
x
� 
�
%s
*synth2�
�|dsp48e1_wrapper__parameterized1_172 | (A*B)'            | 24     | 17     | -      | -      | 41     | 0    | 0    | -    | -    | -     | 0    | 1    | 
h p
x
� 
�
%s
*synth2�
�|dsp48e1_wrapper__parameterized2_173 | PCIN>>17+(A*B)'   | 0      | 7      | -      | -      | 31     | 0    | 0    | -    | -    | -     | 1    | 0    | 
h p
x
� 
�
%s
*synth2�
�|dsp48e1_wrapper__parameterized3_170 | PCIN+A:B+C        | 22     | 0      | -      | -      | 48     | 0    | 0    | -    | -    | -     | 0    | 0    | 
h p
x
� 
�
%s
*synth2�
�|dsp48e1_wrapper__parameterized1_146 | (A*B)'            | 24     | 17     | -      | -      | 41     | 0    | 0    | -    | -    | -     | 0    | 1    | 
h p
x
� 
�
%s
*synth2�
�|dsp48e1_wrapper__parameterized2_147 | PCIN>>17+(A*B)'   | 0      | 7      | -      | -      | 31     | 0    | 0    | -    | -    | -     | 1    | 0    | 
h p
x
� 
�
%s
*synth2�
�|dsp48e1_wrapper__parameterized3_144 | PCIN+A:B+C        | 22     | 0      | -      | -      | 48     | 0    | 0    | -    | -    | -     | 0    | 0    | 
h p
x
� 
�
%s
*synth2�
�|dsp48e1_wrapper__parameterized1     | (A*B)'            | 24     | 17     | -      | -      | 41     | 0    | 0    | -    | -    | -     | 0    | 1    | 
h p
x
� 
�
%s
*synth2�
�|dsp48e1_wrapper__parameterized2     | PCIN>>17+(A*B)'   | 0      | 7      | -      | -      | 31     | 0    | 0    | -    | -    | -     | 1    | 0    | 
h p
x
� 
�
%s
*synth2�
�|dsp48e1_wrapper__parameterized3     | PCIN+A:B+C        | 22     | 0      | -      | -      | 48     | 0    | 0    | -    | -    | -     | 0    | 0    | 
h p
x
� 
�
%s
*synth2�
�|grayscale_kernel                    | A'*B'             | 17     | 14     | -      | -      | 48     | 1    | 1    | -    | -    | -     | 0    | 0    | 
h p
x
� 
�
%s
*synth2�
�|grayscale_kernel                    | (PCIN>>17+A'*B')' | 14     | 14     | -      | -      | 48     | 1    | 1    | -    | -    | -     | 0    | 1    | 
h p
x
� 
�
%s
*synth2�
�|grayscale_kernel                    | A'*B'             | 17     | 17     | -      | -      | 48     | 1    | 1    | -    | -    | -     | 0    | 0    | 
h p
x
� 
�
%s
*synth2�
�|grayscale_kernel                    | (PCIN>>17+A*B')'  | 0      | 14     | -      | -      | 48     | 0    | 1    | -    | -    | -     | 0    | 1    | 
h p
x
� 
�
%s
*synth2�
�+------------------------------------+-------------------+--------+--------+--------+--------+--------+------+------+------+------+-------+------+------+

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
8
%s
*synth2 
+-+--------------+----------+
h p
x
� 
8
%s
*synth2 
| |BlackBox name |Instances |
h p
x
� 
8
%s
*synth2 
+-+--------------+----------+
h p
x
� 
8
%s
*synth2 
+-+--------------+----------+
h p
x
� 
/
%s*synth2

Report Cell Usage: 
h px� 
3
%s*synth2
+------+--------+------+
h px� 
3
%s*synth2
|      |Cell    |Count |
h px� 
3
%s*synth2
+------+--------+------+
h px� 
3
%s*synth2
|1     |CARRY4  |   106|
h px� 
3
%s*synth2
|2     |DSP48E1 |    17|
h px� 
3
%s*synth2
|9     |LUT1    |   127|
h px� 
3
%s*synth2
|10    |LUT2    |   245|
h px� 
3
%s*synth2
|11    |LUT3    |   723|
h px� 
3
%s*synth2
|12    |LUT4    |   242|
h px� 
3
%s*synth2
|13    |LUT5    |   306|
h px� 
3
%s*synth2
|14    |LUT6    |   486|
h px� 
3
%s*synth2
|15    |MUXCY   |   290|
h px� 
3
%s*synth2
|16    |SRL16E  |     9|
h px� 
3
%s*synth2
|17    |SRLC32E |     9|
h px� 
3
%s*synth2
|18    |XORCY   |   130|
h px� 
3
%s*synth2
|19    |FDE     |     6|
h px� 
3
%s*synth2
|20    |FDRE    |  2419|
h px� 
3
%s*synth2
|21    |FDSE    |    10|
h px� 
3
%s*synth2
+------+--------+------+
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
�Finished Writing Synthesis Report : Time (s): cpu = 00:00:43 ; elapsed = 00:01:27 . Memory (MB): peak = 1732.797 ; gain = 1119.898
h px� 
l
%s
*synth2T
R---------------------------------------------------------------------------------
h p
x
� 
b
%s
*synth2J
HSynthesis finished with 0 errors, 0 critical warnings and 596 warnings.
h p
x
� 
�
%s
*synth2�
�Synthesis Optimization Runtime : Time (s): cpu = 00:00:21 ; elapsed = 00:01:22 . Memory (MB): peak = 1732.797 ; gain = 1119.141
h p
x
� 
�
%s
*synth2�
�Synthesis Optimization Complete : Time (s): cpu = 00:00:43 ; elapsed = 00:01:27 . Memory (MB): peak = 1732.797 ; gain = 1119.898
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

00:00:012
00:00:00.1162

1732.7972
0.000Z17-268h px� 
U
-Analyzing %s Unisim elements for replacement
17*netlist2
549Z29-17h px� 
X
2Unisim Transformation completed in %s CPU seconds
28*netlist2
0Z29-28h px� 
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

00:00:002
00:00:00.0022

1732.7972
0.000Z17-268h px� 
�
!Unisim Transformation Summary:
%s111*project2q
o  A total of 83 instances were transformed.
  (MUXCY,XORCY) => CARRY4: 77 instances
  FDE => FDRE: 6 instances
Z1-111h px� 
V
%Synth Design complete | Checksum: %s
562*	vivadotcl2

f129a8d3Z4-1430h px� 
C
Releasing license: %s
83*common2
	SynthesisZ17-83h px� 
�
G%s Infos, %s Warnings, %s Critical Warnings and %s Errors encountered.
28*	vivadotcl2
852
1722
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

00:00:452

00:01:352

1732.7972

1349.938Z17-268h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
Write ShapeDB Complete: 2

00:00:002
00:00:00.0072

1732.7972
0.000Z17-268h px� 
�
 The %s '%s' has been generated.
621*common2

checkpoint2�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/grayscale_grayscale_kernel_0_2_synth_1/grayscale_grayscale_kernel_0_2.dcpZ17-1381h px� 
�
<Added synthesis output to IP cache for IP %s, cache-ID = %s
485*coretcl2 
grayscale_grayscale_kernel_0_22
c902049aeb468a2aZ2-1648h px� 
@
Renamed %s cell refs.
330*coretcl2
401Z2-1174h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
Write ShapeDB Complete: 2

00:00:002
00:00:00.0072

1732.7972
0.000Z17-268h px� 
�
 The %s '%s' has been generated.
621*common2

checkpoint2�
�D:/cse_462/fgpa_convolution_acceleration/grayscale_hardware/grayscale_hardware.runs/grayscale_grayscale_kernel_0_2_synth_1/grayscale_grayscale_kernel_0_2.dcpZ17-1381h px� 
�
Executing command : %s
56330*	planAhead2�
�report_utilization -file grayscale_grayscale_kernel_0_2_utilization_synth.rpt -pb grayscale_grayscale_kernel_0_2_utilization_synth.pbZ12-24828h px� 
\
Exiting %s at %s...
206*common2
Vivado2
Thu Mar  6 03:29:35 2025Z17-206h px� 


End Record