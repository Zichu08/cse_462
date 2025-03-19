# myapp/hardware.py
from pynq import DefaultIP, Overlay, allocate
import numpy as np

class AddDriver(DefaultIP):
    def __init__(self, description):
        super().__init__(description=description)
    bindto = ['xilinx.com:hls:add:1.0']
    def add(self, a, b):
        self.write(0x10, a)
        self.write(0x18, b)
        return self.read(0x20)

# Load the overlay for the addition accelerator.
adder_overlay = Overlay('/home/xilinx/pynq/overlays/adder/adder.bit')
scalar_add = adder_overlay.scalar_add

# Load the overlay once.
grayscale_overlay = Overlay('/home/xilinx/pynq/overlays/grayscale/grayscale.bit')
grayscale_ip = grayscale_overlay.grayscale_kernel_0   # IP block
dma = grayscale_overlay.axi_dma_0                     # AXI DMA

def hardware_grayscale(img_array):
    """
    Convert an RGB image (NumPy array shape (H, W, 3)) to grayscale
    using the hardware IP block on the FPGA.
    """
    # 1) Get dimensions
    height, width, _ = img_array.shape

    # 2) Pack the 3 channels into 32-bit
    #    (R << 16) | (G << 8) | B
    combined_array = ((img_array[:,:,0].astype(np.uint32) << 16) |
                      (img_array[:,:,1].astype(np.uint32) <<  8) |
                      (img_array[:,:,2].astype(np.uint32)))

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

    # 7) Optionally wait for the IP to signal it’s done (poll the status register)
    #    This is sometimes optional if the IP is well synchronized with the DMA done.
    while (grayscale_ip.read(0x00) & 0x4) == 0:
        pass

    # 8) Unpack the 32-bit data back into separate channels
    r_channel = (out_buffer >> 16) & 0xFF
    g_channel = (out_buffer >>  8) & 0xFF
    b_channel =  out_buffer        & 0xFF

    # 9) Stack along last axis => shape (height, width, 3)
    gray_result = np.stack((r_channel, g_channel, b_channel), axis=-1).astype(np.uint8)

    # 10) Free the buffers (optional cleanup)
    in_buffer.freebuffer()
    out_buffer.freebuffer()

    return gray_result
