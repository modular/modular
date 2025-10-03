"""Wrapper macro for py_library"""

load("@rules_python//python:defs.bzl", "py_library")

def modular_py_library(
        name,
        visibility = ["//visibility:public"],
        **kwargs):
    """Creates a py_library target

    Args:
        name: The name of the underlying py_library
        visibility: The visibility of the target, defaults to public
        **kwargs: Extra arguments passed through to py_library
    """

    py_library(
        name = name,
        visibility = visibility,
        **kwargs
    )
