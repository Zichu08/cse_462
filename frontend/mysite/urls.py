# mysite/urls.py
"""
URL configuration for mysite project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path
from myapp.views import (
    Conv2DReferenceView,
    HardwareGrayscaleAPIView,
    SoftwareProcessAPIView,
    # HardwareConv2DAPIView
)

urlpatterns = [
    path('', Conv2DReferenceView.as_view(), name='conv2d_reference'),
    
    # REST endpoints
    path('api/software_process/', SoftwareProcessAPIView.as_view(), name='software_process_api'),
    path('api/hw_grayscale/', HardwareGrayscaleAPIView.as_view(), name='hw_grayscale_api'),
    # path('api/hw_conv2d/', HardwareConv2DAPIView.as_view(), name='hw_conv2d_api'),
]
