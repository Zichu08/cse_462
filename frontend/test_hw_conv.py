import time
import numpy as np
from PIL import Image

from pynq import Overlay, allocate
from myapp.hardware_globals import filter_overlay, filter_dma, filter_kernel_ip
from myapp.hardware import hardware_conv2d
from myapp.utils import get_kernel_by_type, compute_factor_for_kernel, convert_float_kernel_to_int

overlay = Overlay("/home/xilinx/pynq/overlays/filter/filter.bit")
dma = overlay.axi_dma_0
ip = overlay.filter_kernel_0

img_path = "/home/xilinx/jupyter_notebooks/220px.png"
img = Image.open(img_path).convert("RGB")
img_array = np.array(img)

float_kernel = get_kernel_by_type("edge_detect_3x3")
factor = compute_factor_for_kernel(float_kernel)
int_kernel = convert_float_kernel_to_int(float_kernel, factor)

result, t = hardware_conv2d(img_array, int_kernel, factor)
print(f"Hardware conv took {t:.4f}s, result shape = {result.shape}")
