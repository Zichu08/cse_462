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
            # 1) The user’s uploaded image
            image_file = form.cleaned_data['image']
            pil_image = Image.open(image_file)

            # 2) The chosen kernel => software conv
            kernel_type = form.cleaned_data['kernel_type']
            kernel = get_kernel_by_type(kernel_type)
            image_np = np.array(pil_image)

            # 3) Software convolution
            start_time = time.perf_counter()
            convolved = flexible_conv2d(image_np, kernel, padding='reflect')
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            # 4) Convert software result to PIL
            if convolved.ndim == 2:
                convolved_image = Image.fromarray(convolved, mode='L')
            else:
                convolved_image = Image.fromarray(convolved)

            # 5) Encode the original image in base64 for later use in the template
            #    (We’ll re‐send it to the hardware REST endpoint from JavaScript)
            original_buffer = BytesIO()
            # Convert the original to a standard format like PNG
            pil_image.save(original_buffer, format="PNG")
            original_base64 = base64.b64encode(original_buffer.getvalue()).decode('utf-8')

            # 6) Encode the convolved image
            convolved_buffer = BytesIO()
            convolved_image.save(convolved_buffer, format="PNG")
            convolved_base64 = base64.b64encode(convolved_buffer.getvalue()).decode('utf-8')

            return render(request, self.template_name, {
                "form": form,
                "original_image": original_base64,    # we embed this for hardware calls
                "encoded_result": convolved_base64,   # software convolution
                "elapsed_time": f"{elapsed_time:.4f} seconds"
            })
        else:
            return render(request, self.template_name, {"form": form})


class HardwareGrayscaleAPIView(APIView):
    """
    A REST endpoint that:
      - Receives an uploaded image via POST
      - Calls hardware_grayscale on the FPGA
      - Returns the grayscale as base64
    """
    parser_classes = (MultiPartParser, FormParser)  # so DRF can parse the uploaded file

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
            output_np = hardware_grayscale(input_np)

            # Convert back to PNG in memory
            out_pil = Image.fromarray(output_np)
            buffer = BytesIO()
            out_pil.save(buffer, format="PNG")

            # Encode as base64
            hw_gray_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return Response({"hw_gray_image": hw_gray_b64}, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
