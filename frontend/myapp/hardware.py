# myapp/hardware.py
from pynq import DefaultIP, Overlay, allocate
import numpy as np
import time

# Load the overlay once. (Ensure the .bit file path is correct for your setup.)
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
