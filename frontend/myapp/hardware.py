# myapp/hardware.py
from pynq import DefaultIP, Overlay, allocate
import numpy as np
import time
import threading
from .hardware_globals import filter_overlay, filter_dma, filter_kernel_ip

################################################################################
# Hardware convolution logic
################################################################################
class FilterKernel(DefaultIP):
    bindto = ['xilinx.com:hls:filter_kernel:1.0']

    def __init__(self, description):
        super().__init__(description=description)
        self.width_addr = self.register_map.image_width.address
        self.height_addr = self.register_map.image_height.address
        self.factor_addr = self.register_map.kernel_factor.address
        # Kernel base address (0x40)
        self.kernel_base_addr = 0x40

    @property
    def width(self):
        return self.read(self.width_addr)

    @width.setter
    def width(self, value):
        self.write(self.width_addr, value)

    @property
    def height(self):
        return self.read(self.height_addr)

    @height.setter
    def height(self, value):
        self.write(self.height_addr, value)

    @property
    def factor(self):
        return self.read(self.factor_addr)

    @factor.setter
    def factor(self, value):
        self.write(self.factor_addr, value)

    @property
    def kernel(self):
        # Read 9 int32 coefficients (3x3)
        coeffs = []
        for i in range(9):
            val = self.read(self.kernel_base_addr + (4*i))
            coeffs.append(val)
        return np.array(coeffs, dtype=np.int32).reshape(3,3)

    @kernel.setter
    def kernel(self, matrix):
        matrix = np.array(matrix, dtype=np.int32)
        if matrix.shape != (3,3):
            raise ValueError(f"Kernel must be 3x3, got {matrix.shape}")
        flat = matrix.flatten()
        for i, val in enumerate(flat):
            self.write(self.kernel_base_addr + (4*i), int(val))


hardware_lock = threading.Lock()
def hardware_conv2d(img_array, kernel_3x3, factor=1):
    """
    Perform a hardware convolution on the FPGA using the 'filter_kernel'
    overlay (with DMA). The kernel is a 3×3 integer matrix, factor is an int
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
    with hardware_lock:
        print("[DEBUG] hardware_conv2d: using overlay from hardware_globals")
        filter_dma = filter_overlay.axi_dma_0
        filter_kernel_ip = filter_overlay.filter_kernel_0

        start_time = time.perf_counter()

        # Dimensions
        height, width, _ = img_array.shape
        print(f"[DEBUG] image shape = {img_array.shape} (height={height}, width={width})")

        # 1) Pack channels into 32-bit
        print("[DEBUG] Packing channels into 32-bit format...")
        combined_array = (
            (img_array[:,:,0].astype(np.uint32) << 16) |
            (img_array[:,:,1].astype(np.uint32) <<  8) |
            img_array[:,:,2].astype(np.uint32)
        )
        print("[DEBUG] combined_array shape =", combined_array.shape)

        # 2) Allocate DMA buffers
        print("[DEBUG] Allocating DMA buffers...")
        in_buffer = allocate(shape=combined_array.shape, dtype=np.uint32)
        out_buffer = allocate(shape=combined_array.shape, dtype=np.uint32)
        in_buffer[:] = combined_array
        print("[DEBUG] Copied data to in_buffer.")

        # 3) Program the filter_kernel IP
        print(f"[DEBUG] Programming filter_kernel IP with width={width}, height={height}, factor={factor}")
        print(f"[DEBUG] kernel_3x3:\n{kernel_3x3}")
        filter_kernel_ip.width = width
        filter_kernel_ip.height = height
        filter_kernel_ip.factor = factor
        filter_kernel_ip.kernel = kernel_3x3

        # 4) DMA
        print("[DEBUG] Starting DMA transfers...")
        filter_dma.sendchannel.transfer(in_buffer)
        filter_dma.recvchannel.transfer(out_buffer)

        # Start IP
        print("[DEBUG] Writing IP start bit (0x00 => 1)")
        filter_kernel_ip.write(0x00, 1)

        print("[DEBUG] Waiting for sendchannel DMA to finish...")
        filter_dma.sendchannel.wait()
        print("[DEBUG] sendchannel finished. Waiting for recvchannel DMA to finish...")
        filter_dma.recvchannel.wait()
        print("[DEBUG] recvchannel finished. DMA complete.")

        # 5) Unpack the result
        print("[DEBUG] Unpacking output buffer to R,G,B channels...")
        r_channel = (out_buffer >> 16) & 0xFF
        g_channel = (out_buffer >>  8) & 0xFF
        b_channel =  out_buffer        & 0xFF
        conv_result = np.stack((r_channel, g_channel, b_channel), axis=-1).astype(np.uint8)
        print("[DEBUG] conv_result shape =", conv_result.shape)

        # 6) Free buffers
        print("[DEBUG] Freeing DMA buffers...")
        in_buffer.freebuffer()
        out_buffer.freebuffer()

        end_time = time.perf_counter()
        hw_time = end_time - start_time
        print(f"[DEBUG] hardware_conv2d() done. Took {hw_time:.4f} seconds.")
        return conv_result, hw_time
