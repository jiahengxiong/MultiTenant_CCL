from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "simcore_cpp",
        ["CCL_Simulator/simcore_cpp/bindings.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"],
    ),
]

setup(
    name="simcore_cpp",
    ext_modules=ext_modules,
)