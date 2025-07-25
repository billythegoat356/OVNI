from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "Driver_Wrapper",
        ["NvEncVersionCheck.cpp"],
        cxx_std=14,  # Specify C++14 standard
    ),
]

setup(
    name="Driver_Wrapper",
    version="0.1.0",
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.7"
)