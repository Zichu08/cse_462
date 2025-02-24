# myapp/forms.py
from django import forms

# Define several kernel options here:
KERNEL_CHOICES = [
    ('edge_detect', 'Edge Detection'),
    ('blur', 'Blur'),
    ('sharpen', 'Sharpen'),
    ('identity', 'Identity'),
]

class SoftwareConv2DForm(forms.Form):
    """
    A form that collects:
      1. An uploaded image.
      2. A choice of convolution kernel.
    """
    image = forms.ImageField(label="Upload an Image")
    kernel_type = forms.ChoiceField(
        choices=KERNEL_CHOICES,
        label="Choose a Convolution Kernel",
        initial='edge_detect'
    )
