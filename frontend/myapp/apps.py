# myapp/apps.py

from django.apps import AppConfig
import sys

class MyappConfig(AppConfig):
    name = 'myapp'
    
    def ready(self):
        print("[DEBUG] MyAppConfig.ready(): Loading filter overlay in main thread.")
        from pynq import Overlay
        filter_overlay = Overlay("/home/xilinx/pynq/overlays/filter/filter.bit")

        # store references on a global object or a module-level variable
        from . import hardware_globals
        hardware_globals.overlay = filter_overlay
        hardware_globals.filter_dma = filter_overlay.axi_dma_0
        hardware_globals.filter_kernel_ip = filter_overlay.filter_kernel_0
        print("[DEBUG] Overlay loaded and references stored in hardware_globals.")
