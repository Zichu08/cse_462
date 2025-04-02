# myapp/views.py

from django.shortcuts import render
from .forms import SoftwareConv2DForm
from .utils import (
    flexible_conv2d,
    get_kernel_by_type,
    software_grayscale,
    fast_conv_scipy
)
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

from .hardware import (
    hardware_grayscale,
    hardware_conv2d
)

##############################################################################
# Main Page View
##############################################################################

class Conv2DReferenceView(View):
    """
    Renders the template with a form for uploading an image and choosing
    a kernel. The actual software and hardware processing are done via
    AJAX calls to REST endpoints.
    """
    template_name = "myapp/conv2d_reference.html"

    def get(self, request):
        form = SoftwareConv2DForm()
        return render(request, self.template_name, {"form": form})

    def post(self, request):
        form = SoftwareConv2DForm(request.POST, request.FILES)
        use_scipy = request.POST.get('use_scipy', '')  # 'on' if checked, '' if unchecked
        if form.is_valid():
            image_file = form.cleaned_data['image']
            pil_image = Image.open(image_file)

            # Convert the original image to base64 for front-end JavaScript use
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            original_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            kernel_type = form.cleaned_data['kernel_type']

            return render(request, self.template_name, {
                "form": form,
                "original_image": original_base64,
                "selected_kernel": kernel_type,
                "use_scipy": use_scipy
            })
        else:
            return render(request, self.template_name, {"form": form})


##############################################################################
# REST API Endpoints
##############################################################################

class SoftwareProcessAPIView(APIView):
    """
    Receives an image + kernel_type, then performs:
      1) Software grayscale
      2) Software convolution (either the custom flexible_conv2d or SciPy).
    and returns:
      - software_gray_image (base64)
      - software_gray_time (seconds)
      - software_conv_image (base64)
      - software_conv_time (seconds)
    """
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            # Extract uploaded image
            image_file = serializer.validated_data['image']
            kernel_type = request.POST.get('kernel_type', 'edge_detect_3x3')
            use_scipy = request.POST.get('use_scipy', '')  # 'on' or ''

            # Convert to a PIL Image (ensure RGB)
            pil_image = Image.open(image_file).convert("RGB")
            image_np = np.array(pil_image)

            # 1) Software grayscale
            start_gray = time.perf_counter()
            gray_np = software_grayscale(image_np)  # shape (H, W)
            end_gray = time.perf_counter()
            gray_time = end_gray - start_gray

            # Convert grayscale result to base64
            gray_pil = Image.fromarray(gray_np)
            gray_buffer = BytesIO()
            gray_pil.save(gray_buffer, format="PNG")
            sw_gray_b64 = base64.b64encode(gray_buffer.getvalue()).decode('utf-8')

            # 2) Software convolution
            kernel = get_kernel_by_type(kernel_type)

            start_conv = time.perf_counter()
            if use_scipy == 'on':
                # Use SciPy's convolve2d for each channel
                conv_np = fast_conv_scipy(image_np, kernel, padding='reflect')
            else:
                # Use the existing pure-Python function
                conv_np = flexible_conv2d(image_np, kernel, padding='reflect')
            end_conv = time.perf_counter()
            conv_time = end_conv - start_conv

            # Convert convolved result to base64
            if conv_np.ndim == 2:
                conv_pil = Image.fromarray(conv_np, mode='L')
            else:
                conv_pil = Image.fromarray(conv_np)
            conv_buffer = BytesIO()
            conv_pil.save(conv_buffer, format="PNG")
            sw_conv_b64 = base64.b64encode(conv_buffer.getvalue()).decode('utf-8')

            # Return JSON response
            return Response({
                "software_gray_image": sw_gray_b64,
                "software_gray_time": f"{gray_time:.4f} seconds",
                "software_conv_image": sw_conv_b64,
                "software_conv_time": f"{conv_time:.4f} seconds",
            }, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class HardwareGrayscaleAPIView(APIView):
    """
    A REST endpoint that:
      - Receives an uploaded image via POST
      - Calls hardware_grayscale on the FPGA
      - Returns the grayscale as base64
      - Also returns the time it took to run on hardware
    """
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            # Extract the image file from the request
            image_file = serializer.validated_data['image']

            # Convert to an RGB PIL Image
            pil_image = Image.open(image_file).convert("RGB")
            # Convert to NumPy
            input_np = np.array(pil_image)

            # Run hardware grayscale
            hw_result, hw_time = hardware_grayscale(input_np)

            # Convert back to PNG in memory
            out_pil = Image.fromarray(hw_result)
            buffer = BytesIO()
            out_pil.save(buffer, format="PNG")

            # Encode as base64
            hw_gray_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return Response({
                "hw_gray_image": hw_gray_b64,
                "hw_gray_time": f"{hw_time:.4f} seconds"
            }, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class HardwareConv2DAPIView(APIView):
    """
    A REST endpoint that:
      - Receives an uploaded image via POST
      - Receives a kernel choice (e.g. 'edge_detect_3x3')
      - Converts that kernel to a 3x3 int matrix + factor if needed
      - Calls hardware_conv2d to perform FPGA-based convolution
      - Returns the convolved image in base64
      - Returns the time taken on hardware
    """
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image_file = serializer.validated_data['image']
            kernel_type = request.POST.get('kernel_type', 'edge_detect_3x3')

            # Convert PIL image => NumPy
            pil_image = Image.open(image_file).convert("RGB")
            input_np = np.array(pil_image)

            # Convert float kernel to int for the IP
            float_kernel = get_kernel_by_type(kernel_type)  # shape (3, 3), float
            # Example approach: multiply float kernel by some factor, then cast to int
            # For small 3x3 filters like sharpen, edge detection, factor=1 is OK.
            # For blur (1/9), you might set factor=9.
            # We'll do a simple guess based on sum of kernel.
            sum_of_elems = np.sum(float_kernel)
            if abs(sum_of_elems) < 1e-6:
                # e.g. for edge detection, sum=0, let's just use factor=1
                factor = 1
                int_kernel = float_kernel.astype(np.int32)
            else:
                # e.g. for blur_3x3 => sum=1
                # or blur_5x5 => sum=1
                # or sharpen => sum=1
                # We'll just do factor=1 if sum=1. If sum=5, factor=5, etc.
                # Adjust as you see fit for your hardware design.
                factor = int(np.round(sum_of_elems))
                if factor == 0:
                    factor = 1  # fallback
                int_kernel = (float_kernel * factor).astype(np.int32)

            # Run hardware convolution
            hw_conv_result, hw_conv_time = hardware_conv2d(input_np, int_kernel, factor=factor)

            # Convert result to base64
            out_pil = Image.fromarray(hw_conv_result)
            buffer = BytesIO()
            out_pil.save(buffer, format="PNG")
            hw_conv_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return Response({
                "hw_conv_image": hw_conv_b64,
                "hw_conv_time": f"{hw_conv_time:.4f} seconds"
            }, status=status.HTTP_200_OK)

        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
