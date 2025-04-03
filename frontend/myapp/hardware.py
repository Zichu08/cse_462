# myapp/hardware.py
from pynq import DefaultIP, Overlay, allocate
import numpy as np
import time

################################################################################
# Hardware grayscale logic
################################################################################
grayscale_overlay = Overlay('/home/xilinx/pynq/overlays/grayscale/grayscale.bit')
grayscale_ip = grayscale_overlay.grayscale_kernel_0   # IP block
dma = grayscale_overlay.axi_dma_0                     # AXI DMA

def hardware_grayscale(img_array):
    """
    Convert an RGB image (NumPy array shape (H, W, 3)) to grayscale
    using the hardware IP block on the FPGA.

    Parameters
    ----------
    img_array : np.ndarray
        A NumPy array of shape (H, W, 3) in uint8, representing a color image.

    Returns
    -------
    np.ndarray
        The resulting hardware-processed grayscale image, shape (H, W, 3).
        (Although it's "grayscale," the hardware IP might return three identical channels.)
    """
    start_time = time.perf_counter()  # Start timing here

    # 1) Get dimensions
    height, width, _ = img_array.shape

    # 2) Pack the 3 channels into 32-bit: (R << 16) | (G << 8) | B
    combined_array = ((img_array[:,:,0].astype(np.uint32) << 16) |
                      (img_array[:,:,1].astype(np.uint32) <<  8) |
                       img_array[:,:,2].astype(np.uint32))

    # 3) Allocate input and output DMA buffers
    in_buffer = allocate(shape=combined_array.shape, dtype=np.uint32)
    out_buffer = allocate(shape=combined_array.shape, dtype=np.uint32)

    # 4) Copy to the in_buffer
    in_buffer[:] = combined_array

    # 5) Program the grayscale IP control registers
    grayscale_ip.write(0x10, width)
    grayscale_ip.write(0x18, height)
    # Start IP
    grayscale_ip.write(0x00, 0x01)

    # 6) Start the DMA transfers
    dma.sendchannel.transfer(in_buffer)
    dma.recvchannel.transfer(out_buffer)

    # Wait for DMA completion
    dma.sendchannel.wait()
    dma.recvchannel.wait()

    # 7) Optionally wait for the IP to signal it’s done
    while (grayscale_ip.read(0x00) & 0x4) == 0:
        pass

    # 8) Unpack the 32-bit data back into separate channels
    r_channel = (out_buffer >> 16) & 0xFF
    g_channel = (out_buffer >>  8) & 0xFF
    b_channel =  out_buffer        & 0xFF

    # 9) Stack along the last axis => shape (height, width, 3)
    gray_result = np.stack((r_channel, g_channel, b_channel), axis=-1).astype(np.uint8)

    # 10) Free the buffers
    in_buffer.freebuffer()
    out_buffer.freebuffer()

    end_time = time.perf_counter()  # End timing
    hw_time = end_time - start_time

    # Return both the resulting image and the hardware processing time
    return gray_result, hw_time


################################################################################
# Hardware convolution logic
################################################################################
filter_overlay = Overlay('/home/xilinx/pynq/overlays/filter/filter.bit')
filter_dma = filter_overlay.axi_dma_0

# class FilterKernel(DefaultIP):
#     bindto = ['xilinx.com:hls:filter_kernel:1.0']

#     def __init__(self, description):
#         super().__init__(description=description)
#         self.width_addr = self.register_map.image_width.address
#         self.height_addr = self.register_map.image_height.address
#         self.factor_addr = self.register_map.kernel_factor.address
#         # Kernel base address (0x40)
#         self.kernel_base_addr = 0x40

#     @property
#     def width(self):
#         return self.read(self.width_addr)

#     @width.setter
#     def width(self, value):
#         self.write(self.width_addr, value)

#     @property
#     def height(self):
#         return self.read(self.height_addr)

#     @height.setter
#     def height(self, value):
#         self.write(self.height_addr, value)

#     @property
#     def factor(self):
#         return self.read(self.factor_addr)

#     @factor.setter
#     def factor(self, value):
#         self.write(self.factor_addr, value)

#     @property
#     def kernel(self):
#         # Read 9 int32 coefficients (3x3)
#         coeffs = []
#         for i in range(9):
#             val = self.read(self.kernel_base_addr + (4*i))
#             coeffs.append(val)
#         return np.array(coeffs, dtype=np.int32).reshape(3,3)

#     @kernel.setter
#     def kernel(self, matrix):
#         matrix = np.array(matrix, dtype=np.int32)
#         if matrix.shape != (3,3):
#             raise ValueError(f"Kernel must be 3x3, got {matrix.shape}")
#         flat = matrix.flatten()
#         for i, val in enumerate(flat):
#             self.write(self.kernel_base_addr + (4*i), int(val))

# Reference to the IP block instance
filter_kernel_ip = filter_overlay.filter_kernel_0  # The filter_kernel IP

def hardware_conv2d(img_array, kernel_3x3, factor=1):
    """
    Perform a hardware convolution on the FPGA using the 'filter_kernel'
    overlay (with DMA). The kernel is a 3x3 integer matrix, factor is an int
    that can be used to scale results inside the hardware.

    Parameters
    ----------
    img_array : np.ndarray of shape (H, W, 3) in uint8
        The input image data (RGB).
    kernel_3x3 : np.ndarray of shape (3, 3), integer
        The convolution kernel (e.g. edge detect, sharpen, etc.).
    factor : int
        The factor register on the IP that scales the accumulation (like dividing
        by the sum of the kernel, or 1 for unscaled).

    Returns
    -------
    (result_image, hw_time)
        result_image : np.ndarray of shape (H, W, 3) in uint8
        hw_time : float, seconds
    """
    start_time = time.perf_counter()

    # Dimensions
    height, width, _ = img_array.shape

    # 1) Pack channels into 32-bit
    combined_array = ((img_array[:,:,0].astype(np.uint32) << 16) |
                      (img_array[:,:,1].astype(np.uint32) <<  8) |
                       img_array[:,:,2].astype(np.uint32))

    # 2) Allocate DMA buffers
    in_buffer = allocate(shape=combined_array.shape, dtype=np.uint32)
    out_buffer = allocate(shape=combined_array.shape, dtype=np.uint32)
    in_buffer[:] = combined_array

    # 3) Program the filter_kernel IP
    filter_kernel_ip.width = width
    filter_kernel_ip.height = height
    filter_kernel_ip.factor = factor
    filter_kernel_ip.kernel = kernel_3x3  # 3x3 int matrix

    # 4) DMA
    filter_dma.sendchannel.transfer(in_buffer)
    filter_dma.recvchannel.transfer(out_buffer)

    # Start IP
    filter_kernel_ip.write(0x00, 1)

    filter_dma.sendchannel.wait()
    filter_dma.recvchannel.wait()

    # 5) Unpack the result
    r_channel = (out_buffer >> 16) & 0xFF
    g_channel = (out_buffer >>  8) & 0xFF
    b_channel =  out_buffer        & 0xFF
    conv_result = np.stack((r_channel, g_channel, b_channel), axis=-1).astype(np.uint8)

    # 6) Free buffers
    in_buffer.freebuffer()
    out_buffer.freebuffer()

    end_time = time.perf_counter()
    hw_time = end_time - start_time
    return conv_result, hw_time
