from django.shortcuts import render
import time
import base64
from io import BytesIO
from PIL import Image
from django.views import View
import numpy as np

from .forms import SoftwareConv2DForm
from .utils import naive_conv2d, get_kernel_by_type

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
    """
    Renders a page with a form to upload an image and choose a kernel.
    It then displays the convolved image (software reference) and timing info.
    """
    template_name = "myapp/conv2d_reference.html"

    def get(self, request):
        form = SoftwareConv2DForm()
        return render(request, self.template_name, {"form": form})

    def post(self, request):
        form = SoftwareConv2DForm(request.POST, request.FILES)
        if form.is_valid():
            # 1. Retrieve the uploaded image
            image_file = form.cleaned_data['image']

            # 2. Retrieve the selected kernel type
            kernel_type = form.cleaned_data['kernel_type']

            # 3. Convert the image to grayscale if needed
            pil_image = Image.open(image_file).convert('L')
            image_np = np.array(pil_image)

            # 4. Fetch the chosen kernel
            kernel = get_kernel_by_type(kernel_type)

            # 5. Time the software convolution
            start_time = time.perf_counter()
            convolved = naive_conv2d(image_np, kernel)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            # 6. Convert result back to an image for display
            result_image = Image.fromarray(convolved)

            # 7. Encode the image as base64 to display in the template
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
