"""Wrapper macro for py_library"""

load("@rules_python//python:defs.bzl", "py_library")

def modular_py_library(
        name,
        visibility = None,
        tags = [],
        **kwargs):
    """Creates a py_library target

    Args:
        name: The name of the underlying py_library
        visibility: The visibility of the target, defaults to public
        tags: Tags to add to the target
        **kwargs: Extra arguments passed through to py_library
    """

    if "manual" in tags:
        fail("modular_py_library targets cannot be manual. Remove 'manual' from the tags list.")

    py_library(
        name = name,
        visibility = visibility,
        tags = tags,
        **kwargs
    )
