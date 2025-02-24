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
from .serializers import AdditionSerializer, ImageUploadSerializer
from .hardware import scalar_add, conv2d_driver
from rest_framework.parsers import MultiPartParser, FormParser
import numpy as np
import time
import base64


# Create your views here.
# Original addition API.
class AdditionAPIView(APIView):
    def post(self, request):
        serializer = AdditionSerializer(data=request.data)
        if serializer.is_valid():
            num1 = serializer.validated_data['num1']
            num2 = serializer.validated_data['num2']
            # Use the hardware accelerator to perform addition.
            result = scalar_add.add(num1, num2)
            return Response({"sum": result}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# New convolution API for hardware part computation (not complete yet)
class ConvolutionAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            # Retrieve the uploaded image.
            image_file = serializer.validated_data['image']

            # Open the image with Pillow.
            image = Image.open(image_file)

            # Convert to grayscale if your accelerator expects a 2D (single channel) image.
            image = image.convert('L')

            # Convert image to a NumPy array.
            image_np = np.array(image)

            # Define a convolution kernel.
            # Here we use a simple edge detection kernel.
            edge_detect_kernel = np.array([
                [-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]
            ], dtype=np.int8)

            # Pass the image and kernel to the hardware accelerator.
            result_np = conv2d_driver.conv2d(image_np, edge_detect_kernel)

            # Convert the resulting NumPy array back into an image.
            result_image = Image.fromarray(result_np)

            # Save the result image to a bytes buffer.
            buffer = BytesIO()
            result_image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

            # Encode the image bytes as a Base64 string.
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')

            return Response({"result_image": encoded_image}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class Conv2DReferenceView(View):
    template_name = "myapp/conv2d_reference.html"

    def get(self, request):
        form = SoftwareConv2DForm()
        return render(request, self.template_name, {"form": form})

    def post(self, request):
        form = SoftwareConv2DForm(request.POST, request.FILES)
        if form.is_valid():
            # 1. Retrieve the uploaded image (DO NOT convert to L if you want color)
            image_file = form.cleaned_data['image']
            pil_image = Image.open(image_file)  
            # If you want to handle both grayscale or color automatically,
            # you could do:
            # pil_image = pil_image.convert("RGB")

            # 2. Retrieve the selected kernel type
            kernel_type = form.cleaned_data['kernel_type']
            # Get the corresponding kernel
            kernel = get_kernel_by_type(kernel_type)
            
            # 2a. If the user selected an edge-detect kernel, 
            #     convert the image to grayscale automatically.
            if 'edge_detect' in kernel_type.lower():
                pil_image = pil_image.convert('L')
            
            # --- A) Convert the *original* PIL image to base64 for display ---
            # (We'll encode it *before* any modifications, so the user sees the original uploaded image.)
            original_buffer = BytesIO()
            # Save as PNG (or the original format, but PNG is usually safe)
            pil_image.save(original_buffer, format="PNG")
            original_bytes = original_buffer.getvalue()
            original_image_b64 = base64.b64encode(original_bytes).decode('utf-8')
            
            # Convert to NumPy
            image_np = np.array(pil_image)
            # shape could be (H, W) for grayscale PNG or (H, W, 3/4) for color

            # 3. Time the software convolution
            start_time = time.perf_counter()
            convolved = flexible_conv2d(image_np, kernel, padding='reflect')
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            # 4. Convert the result back to an image for display
            # If result is 2D, we must specify mode='L' in PIL
            # If result has shape (H,W,C)=3 channels, we specify mode='RGB'
            if convolved.ndim == 2:
                result_image = Image.fromarray(convolved, mode='L')
            else:
                # Usually the result shape is (H,W,3) for RGB
                # If there's an alpha channel, handle that separately
                result_image = Image.fromarray(convolved)

            # 5. Encode the image in base64 to display inline
            result_buffer = BytesIO()
            result_image.save(result_buffer, format="PNG")
            result_bytes = result_buffer.getvalue()
            encoded_result = base64.b64encode(result_bytes).decode('utf-8')

            return render(request, self.template_name, {
                "form": form,
                "original_image": original_image_b64,
                "encoded_result": encoded_result,
                "elapsed_time": f"{elapsed_time:.4f} seconds"
            })
        else:
            return render(request, self.template_name, {"form": form})
