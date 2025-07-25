"""
Workaround to not require setting python path or running example scripts as modules
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from examples.caller import call
