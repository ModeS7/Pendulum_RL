"""
Serial port configuration and data packet definition.

Defines COM port settings for Arduino/QUBE hardware connection. Provides Packet class
for thread-safe data exchange between control loop and GUI, containing PID parameters,
plot data, and encoder reset flags.
"""

from PID import *

class Packet:
    def __init__(self):
        self.pid = PID()
        self.plot_data = [[] * 9]
        self.resetEncoders = False

    def unpack(self):
        return [
            self.pid,
            self.resetEncoders,
        ]