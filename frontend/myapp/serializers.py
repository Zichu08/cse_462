# myapp/serializers.py
from rest_framework import serializers

# Serializer for handling image uploads.
class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()
