# myapp/serializers.py
from rest_framework import serializers

# class AdditionSerializer(serializers.Serializer):
#     num1 = serializers.IntegerField(min_value=-2147483648, max_value=2147483647)
#     num2 = serializers.IntegerField(min_value=-2147483648, max_value=2147483647)

# Serializer for handling image uploads.
class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()
