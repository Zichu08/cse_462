# myapp/views.py
from django.shortcuts import render
import time
import base64
from io import BytesIO
from PIL import Image
from django.views import View
import numpy as np

from .forms import SoftwareConv2DForm
from .utils import flexible_conv2d, get_kernel_by_type

# Create your views here.
# from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import AdditionSerializer

# from .hardware import overlay

# class AdditionAPIView(APIView):
#     def post(self, request):
#         serializer = AdditionSerializer(data=request.data)
#         if serializer.is_valid():
#             num1 = serializer.validated_data['num1']
#             num2 = serializer.validated_data['num2']
#             # result = num1 + num2

#             # hardware implementation
#             result = overlay.scalar_add.add(num1, num2)

#             return Response({"sum": result}, status=status.HTTP_200_OK)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


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

            # Convert to NumPy
            image_np = np.array(pil_image)
            # shape could be (H, W) for grayscale PNG or (H, W, 3/4) for color

            # 2. Retrieve the selected kernel type
            kernel_type = form.cleaned_data['kernel_type']
            # Get the corresponding kernel
            kernel = get_kernel_by_type(kernel_type)

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
            buffer = BytesIO()
            result_image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            encoded_result = base64.b64encode(image_bytes).decode('utf-8')

            return render(request, self.template_name, {
                "form": form,
                "encoded_result": encoded_result,
                "elapsed_time": f"{elapsed_time:.4f} seconds"
            })
        else:
            return render(request, self.template_name, {"form": form})
