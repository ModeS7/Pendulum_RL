"""
Dependency installer script for QUBE Python control environment.

Automatically installs required Python packages: pyserial (hardware communication),
PyQt5 (GUI framework), and pyqtgraph (real-time plotting). Run once to set up
development environment with all necessary dependencies.
"""

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("pyserial")
install("PyQt5")
install("pyqtgraph")