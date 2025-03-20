# myapp/views.py
from django.shortcuts import render
from .forms import SoftwareConv2DForm
from .utils import flexible_conv2d, get_kernel_by_type
from io import BytesIO
from PIL import Image
from django.views import View
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageUploadSerializer
from rest_framework.parsers import MultiPartParser, FormParser
import numpy as np
import time
import base64
from .hardware import hardware_grayscale


# Create your views here.
class Conv2DReferenceView(View):
    template_name = "myapp/conv2d_reference.html"

    def get(self, request):
        form = SoftwareConv2DForm()
        return render(request, self.template_name, {"form": form})

    def post(self, request):
        form = SoftwareConv2DForm(request.POST, request.FILES)
        if form.is_valid():
            # 1. Upload
            image_file = form.cleaned_data['image']
            pil_image = Image.open(image_file)

            # 2. Kernel selection for software reference
            kernel_type = form.cleaned_data['kernel_type']
            kernel = get_kernel_by_type(kernel_type)

            # 3. Convert to NumPy for software convolution
            image_np = np.array(pil_image)

            # 4. Time the software convolution
            start_time = time.perf_counter()
            convolved = flexible_conv2d(image_np, kernel, padding='reflect')
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            # 5. Convert the convolved array back to a PIL image
            if convolved.ndim == 2:
                convolved_image = Image.fromarray(convolved, mode='L')
            else:
                convolved_image = Image.fromarray(convolved)

            # 6. ALWAYS call hardware grayscale on the original image
            #    Ensure it's RGB so hardware_grayscale gets (H,W,3)
            original_rgb = pil_image.convert("RGB")
            hw_in_np = np.array(original_rgb)
            hw_out_np = hardware_grayscale(hw_in_np)
            hw_gray_image = Image.fromarray(hw_out_np)  # shape(H,W,3) with R=G=B

            # 7. Convert original and convolved images to base64
            #    (We also convert hardware grayscale to base64.)
            original_buffer = BytesIO()
            pil_image.save(original_buffer, format="PNG")
            original_base64 = base64.b64encode(original_buffer.getvalue()).decode('utf-8')

            convolved_buffer = BytesIO()
            convolved_image.save(convolved_buffer, format="PNG")
            convolved_base64 = base64.b64encode(convolved_buffer.getvalue()).decode('utf-8')

            hw_gray_buffer = BytesIO()
            hw_gray_image.save(hw_gray_buffer, format="PNG")
            hw_gray_base64 = base64.b64encode(hw_gray_buffer.getvalue()).decode('utf-8')

            # 8. Render
            return render(request, self.template_name, {
                "form": form,
                "original_image": original_base64,
                "encoded_result": convolved_base64,
                "hw_gray_image": hw_gray_base64,
                "elapsed_time": f"{elapsed_time:.4f} seconds"
            })
        else:
            return render(request, self.template_name, {"form": form})
