# myapp/utils.py
import numpy as np

def flexible_conv2d(image_array, kernel, padding='reflect'):
    """
    Perform a 2D convolution on a grayscale or color (multi-channel) image using a generic kernel.
    This function uses pure Python + NumPy, without external libraries like SciPy/OpenCV,
    so it can serve as a reference for hardware or custom implementations.

    Parameters
    ----------
    image_array : np.ndarray
        The input image as a NumPy array.
          - Shape (H, W) for a single-channel (grayscale).
          - Shape (H, W, C) for color images, where C is the number of channels (e.g., 3 for RGB).
        Typically, this would be np.uint8 in [0..255] or a float array in [0..1].
    
    kernel : np.ndarray
        The 2D convolution kernel, shape (kH, kW).
        For example, a 3×3 edge detection kernel or a 5×5 Gaussian blur, etc.

    padding : str, optional
        The type of padding to use, default is 'reflect'.
        Supported modes:
            - 'reflect': Reflect the image about the border.
            - 'constant': Zero-pad the image border.
            - 'edge': Extend the border pixels.
        You can add more modes if desired.

    Returns
    -------
    np.ndarray
        The convolved image, with the same shape (H, W[, C]) as input.
        - If input was uint8, the output is also uint8, clipped to [0..255].
        - If input was floating-point, the output remains float32 (no clipping unless you add it explicitly).

    Notes
    -----
    1. This function handles bigger kernels by automatically computing the necessary padding.
    2. A triple nested loop is used if the image has multiple channels:
          - Outer loop over channels (if color).
          - Two inner loops for the height and width.
       This makes the logic very explicit, which is useful for hardware reference.
    3. For even-sized kernels, you can still do the same approach. The padding is kH//2 and kW//2.

    Examples
    --------
    # Example usage with a color image and a 3×3 edge detection kernel:
    import numpy as np
    from PIL import Image

    # Load an image and convert to a NumPy array (RGB)
    pil_img = Image.open("sample.jpg").convert('RGB')
    img_array = np.array(pil_img)

    # Define an edge detection kernel (3×3)
    edge_kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)

    # Perform convolution
    result = flexible_conv2d(img_array, edge_kernel, padding='reflect')

    # Convert back to PIL for saving/viewing:
    result_pil = Image.fromarray(result)
    result_pil.save("convolved_output.png")
    """

    # 1. Validate the kernel dimension
    if kernel.ndim != 2:
        raise ValueError("The kernel must be a 2D array, e.g. shape (kH, kW).")

    # 2. Validate the image dimension
    if image_array.ndim not in (2, 3):
        raise ValueError("Image must be 2D (grayscale) or 3D (color).")

    # Remember the original dtype so we can restore/clamp at the end if needed
    orig_dtype = image_array.dtype

    # Convert to float32 for convolution math (avoid overflow in int)
    image_float = image_array.astype(np.float32, copy=False)
    kernel_float = kernel.astype(np.float32, copy=False)

    # 3. Determine the image shape
    if image_float.ndim == 2:
        # Single channel
        H, W = image_float.shape
        C = 1
    else:
        # Multi-channel: e.g. H, W, 3 for RGB
        H, W, C = image_float.shape

    # 4. Determine kernel shape
    kH, kW = kernel_float.shape

    # 5. Compute required padding for 'SAME' style convolution
    padH = kH // 2
    padW = kW // 2

    # 6. Prepare an output array (float32 internally)
    if C == 1:
        output = np.zeros((H, W), dtype=np.float32)
    else:
        output = np.zeros((H, W, C), dtype=np.float32)

    # 7. Convert the user-friendly padding string to NumPy's pad mode
    valid_padding_modes = {'reflect': 'reflect', 'constant': 'constant', 'edge': 'edge'}
    if padding not in valid_padding_modes:
        raise ValueError(f"Unsupported padding mode. Choose from {list(valid_padding_modes.keys())}")
    np_pad_mode = valid_padding_modes[padding]

    # 8. Convolve for each channel
    for c in range(C):

        # Extract the channel data if color, or just the whole array if single-channel
        if C == 1:
            channel_data = image_float
        else:
            channel_data = image_float[..., c]

        # Pad the channel
        padded = np.pad(
            channel_data,
            pad_width=((padH, padH), (padW, padW)),
            mode=np_pad_mode
        )

        # 9. Perform the convolution:
        #    For each pixel (i, j) in the original (un-padded) domain,
        #    compute the sum of elementwise multiplication of the kernel
        #    with the overlapped region in 'padded'.
        for i in range(H):
            for j in range(W):
                # Region of interest in the padded image
                region = padded[i : i + kH, j : j + kW]
                # Multiply element-wise and sum
                val = np.sum(region * kernel_float)
                if C == 1:
                    output[i, j] = val
                else:
                    output[i, j, c] = val

    # 10. If original was uint8 (e.g. a typical image), clamp to [0..255] and convert back to uint8
    if np.issubdtype(orig_dtype, np.integer):
        output = np.clip(output, 0, 255).astype(np.uint8)
    else:
        # If it was float, keep as float (or you can clamp differently if you like)
        pass

    return output


def get_kernel_by_type(kernel_type):
    """
    Returns a NumPy kernel of various sizes for demonstration or testing.
    Extend or modify as needed.
    """
    # Example built-in kernels. You can add 5×5, 7×7, etc. as needed.
    if kernel_type == 'edge_detect_3x3':
        # Basic 3x3 edge detection
        return np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32)

    elif kernel_type == 'blur_3x3':
        # Simple normalized box blur (3×3)
        return np.ones((3, 3), dtype=np.float32) / 9.0

    elif kernel_type == 'sharpen_3x3':
        # Simple sharpen (3×3)
        return np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ], dtype=np.float32)

    elif kernel_type == 'identity_3x3':
        # 3×3 identity
        return np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)

    elif kernel_type == 'blur_5x5':
        # Example 5×5 blur
        return np.ones((5, 5), dtype=np.float32) / 25.0

    # Default fallback
    return np.zeros((3, 3), dtype=np.float32)
