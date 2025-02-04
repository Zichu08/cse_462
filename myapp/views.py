from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import AdditionSerializer

from .hardware import overlay

class AdditionAPIView(APIView):
    def post(self, request):
        serializer = AdditionSerializer(data=request.data)
        if serializer.is_valid():
            num1 = serializer.validated_data['num1']
            num2 = serializer.validated_data['num2']
            # result = num1 + num2

            # hardware implementation
            result = overlay.scalar_add.add(num1, num2)

            return Response({"sum": result}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
