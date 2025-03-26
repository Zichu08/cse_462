# myapp/serializers.py
from rest_framework import serializers

class ImageUploadSerializer(serializers.Serializer):
    """
    Serializer for handling an image upload and an optional kernel type.
    """
    image = serializers.ImageField()
    kernel_type = serializers.CharField(required=False)  # We'll parse kernel_type from the request if needed
