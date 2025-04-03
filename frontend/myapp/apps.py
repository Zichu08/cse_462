# myapp/apps.py

from django.apps import AppConfig
from django.conf import settings
import sys
import os

class MyappConfig(AppConfig):
    name = 'myapp'

    def ready(self):
        from . import hardware_globals

        if hardware_globals.filter_overlay is not None:
            print("[DEBUG] MyAppConfig.ready(): overlay already loaded, skipping re-load.")
            return

        print("[DEBUG] MyAppConfig.ready(): Loading filter overlay in main thread.")
        from pynq import Overlay
        overlay = Overlay("/home/xilinx/pynq/overlays/filter/filter.bit")

        hardware_globals.filter_overlay = overlay
        hardware_globals.filter_dma = overlay.axi_dma_0
        hardware_globals.filter_kernel_ip = overlay.filter_kernel_0
        print("[DEBUG] Overlay loaded and references stored in hardware_globals.")
