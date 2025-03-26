# myapp/views.py
from django.shortcuts import render
from .forms import SoftwareConv2DForm
from .utils import flexible_conv2d, get_kernel_by_type, software_grayscale, fast_conv_scipy
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
        """
        In this refactored version, the POST just re-encodes the user-uploaded
        image into base64 for consumption by the front-end. The software/hardware
        processes are done via their respective REST APIs.
        """
        form = SoftwareConv2DForm(request.POST, request.FILES)
        use_scipy = request.POST.get('use_scipy', '')  # 'on' if checked, '' if unchecked
        if form.is_valid():
            image_file = form.cleaned_data['image']
            pil_image = Image.open(image_file)

            # Convert the original image to base64 for front-end JavaScript use
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            original_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # We also store the chosen kernel type so the frontend can pass it to the software API
            kernel_type = form.cleaned_data['kernel_type']

            # Render template with the base64 image (and kernel choice)
            # The front-end JS will use fetch() to call the software/hardware endpoints
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

            # Kernel type might come from the validated_data or from request.POST
            kernel_type = request.POST.get('kernel_type', 'edge_detect_3x3')

            # Check if user wants SciPy approach
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

            # Run hardware grayscale (which also returns the timing)
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
