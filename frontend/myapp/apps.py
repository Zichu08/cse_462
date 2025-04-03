# myapp/apps.py

from django.apps import AppConfig
from django.conf import settings
import sys

class MyappConfig(AppConfig):
    name = 'myapp'

    def ready(self):
        # If we already have overlay, skip to avoid second load
        from . import hardware_globals
        if hardware_globals.overlay is not None:
            print("[DEBUG] MyAppConfig.ready(): overlay already loaded, skipping re-load.")
            return

        print("[DEBUG] MyAppConfig.ready(): Loading filter overlay in main thread.")
        from pynq import Overlay

        filter_overlay = Overlay("/home/xilinx/pynq/overlays/filter/filter.bit")
        hardware_globals.overlay = filter_overlay
        hardware_globals.filter_dma = filter_overlay.axi_dma_0
        hardware_globals.filter_kernel_ip = filter_overlay.filter_kernel_0
        print("[DEBUG] Overlay loaded and references stored in hardware_globals.")
