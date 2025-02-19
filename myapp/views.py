# myapp/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import AdditionSerializer, ImageUploadSerializer
from .hardware import scalar_add, conv2d_driver

from PIL import Image
import numpy as np
from io import BytesIO
import base64

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


# New convolution API.
# We need to support file uploads so we use the MultiPartParser.
from rest_framework.parsers import MultiPartParser, FormParser

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
