# myapp/utils.py
import numpy as np

def flexible_conv2d(image_array, kernel, padding='reflect'):
    """
    Perform a 2D convolution on a grayscale or color (multi-channel) image using a generic kernel.
    This function uses pure Python + NumPy (no SciPy/OpenCV) for educational clarity.

    Parameters
    ----------
    image_array : np.ndarray
        The input image as a NumPy array.
          - Shape (H, W) for a single-channel (grayscale).
          - Shape (H, W, C) for color images, where C is the number of channels (e.g., 3 for RGB).
    kernel : np.ndarray
        The 2D convolution kernel, shape (kH, kW).
    padding : str
        The type of padding to use (e.g. 'reflect', 'constant', 'edge').

    Returns
    -------
    np.ndarray
        The convolved image, same shape as input (H, W[, C]).
    """
    # 1. Validate kernel dimension
    if kernel.ndim != 2:
        raise ValueError("The kernel must be a 2D array, e.g. shape (kH, kW).")

    # 2. Validate image dimension
    if image_array.ndim not in (2, 3):
        raise ValueError("Image must be 2D (grayscale) or 3D (color).")

    # Remember the original dtype so we can restore/clamp at the end if needed
    orig_dtype = image_array.dtype

    # Convert to float32 for internal math
    image_float = image_array.astype(np.float32, copy=False)
    kernel_float = kernel.astype(np.float32, copy=False)

    # 3. Determine the image shape
    if image_float.ndim == 2:
        H, W = image_float.shape
        C = 1
    else:
        H, W, C = image_float.shape

    # 4. Determine kernel shape
    kH, kW = kernel_float.shape

    # 5. Compute required padding for 'SAME' style convolution
    padH = kH // 2
    padW = kW // 2

    # 6. Prepare an output array (float32)
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

        # 9. Perform the convolution
        for i in range(H):
            for j in range(W):
                region = padded[i : i + kH, j : j + kW]
                val = np.sum(region * kernel_float)
                if C == 1:
                    output[i, j] = val
                else:
                    output[i, j, c] = val

    # 10. If original was uint8, clamp to [0..255] and convert back
    if np.issubdtype(orig_dtype, np.integer):
        output = np.clip(output, 0, 255).astype(np.uint8)

    return output


def get_kernel_by_type(kernel_type):
    """
    Returns a NumPy kernel of various sizes for demonstration or testing.
    Extend or modify as needed.
    """
    import numpy as np

    if kernel_type == 'edge_detect_3x3':
        # Basic 3×3 edge detection
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


def software_grayscale(img_array):
    """
    Convert an RGB image (shape: H x W x 3) to grayscale using simple
    average weighting in software.

    Parameters
    ----------
    img_array : np.ndarray
        A NumPy array of shape (H, W, 3) in uint8 or float32.

    Returns
    -------
    np.ndarray
        Grayscale image as a 2D array of shape (H, W) in the same dtype domain (uint8 or float32).
    """
    if img_array.ndim == 3 and img_array.shape[2] == 3:
        # Convert to float32 to avoid overflow if needed
        orig_dtype = img_array.dtype
        float_img = img_array.astype(np.float32)

        # Simple luminance (many possible ways; here is a straightforward average)
        gray_float = 0.299 * float_img[:, :, 0] + \
                     0.587 * float_img[:, :, 1] + \
                     0.114 * float_img[:, :, 2]

        # Convert back to original dtype
        if np.issubdtype(orig_dtype, np.integer):
            gray_float = np.clip(gray_float, 0, 255)
            return gray_float.astype(np.uint8)
        else:
            return gray_float
    else:
        # If it's already single-channel, just return it
        return img_array
