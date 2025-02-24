import numpy as np
from PIL import Image

def naive_conv2d(image_array, kernel):
    """
    Perform a 2D convolution of image_array (grayscale) with the given kernel.
    Both are expected to be NumPy arrays.
    This is a simplistic implementation for demonstration.
    """

    # Image dimensions
    h, w = image_array.shape
    # Kernel dimensions
    k_h, k_w = kernel.shape

    # Calculate padding
    pad_h = k_h // 2
    pad_w = k_w // 2

    # Create a padded version of the image
    padded_image = np.pad(image_array, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    # Output array
    output = np.zeros_like(image_array, dtype=np.float32)

    # Convolve
    for i in range(h):
        for j in range(w):
            # Region of interest
            region = padded_image[i:i + k_h, j:j + k_w]
            # Element-wise multiply and sum
            val = np.sum(region * kernel)
            output[i, j] = val

    # Clip to valid [0..255], and convert to uint8 for image display
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output


def get_kernel_by_type(kernel_type):
    """
    Returns a NumPy kernel based on the user-selected string.
    Adjust or add more kernels as you wish.
    """
    if kernel_type == 'edge_detect':
        # Simple 3x3 edge detection kernel
        return np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32)

    elif kernel_type == 'blur':
        # Simple normalized box blur kernel 3x3
        return np.ones((3, 3), dtype=np.float32) / 9.0

    elif kernel_type == 'sharpen':
        # A simple sharpen kernel 3x3
        return np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ], dtype=np.float32)

    elif kernel_type == 'identity':
        # Returns the same image
        return np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)

    # Fallback if something unexpected
    return np.zeros((3, 3), dtype=np.float32)
