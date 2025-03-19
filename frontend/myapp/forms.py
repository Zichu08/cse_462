# myapp/forms.py
from django import forms

# Define several kernel options here:
KERNEL_CHOICES = [
    ('edge_detect_3x3', 'Edge Detection (3×3)'),
    ('blur_3x3', 'Box Blur (3×3)'),
    ('sharpen_3x3', 'Sharpen (3×3)'),
    ('identity_3x3', 'Identity (3×3)'),
    ('blur_5x5', 'Box Blur (5×5)'),
]

class SoftwareConv2DForm(forms.Form):
    """
    A form that collects:
      1. An uploaded image (grayscale or color).
      2. A choice of convolution kernel from the updated list.
    """
    image = forms.ImageField(label="Upload an Image")
    kernel_type = forms.ChoiceField(
        choices=KERNEL_CHOICES,
        label="Choose a Convolution Kernel",
        initial='edge_detect_3x3'  # default selection
    )
