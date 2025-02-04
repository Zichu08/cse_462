from pynq import DefaultIP
from pynq import Overlay


class AddDriver(DefaultIP):
    def __init__(self, description):
        super().__init__(description=description)

    bindto = ['xilinx.com:hls:add:1.0']

    def add(self, a, b):
        self.write(0x10, a)
        self.write(0x18, b)
        return self.read(0x20)


overlay = Overlay('/home/xilinx/pynq/overlays/adder/adder.bit')
