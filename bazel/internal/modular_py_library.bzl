"""Wrapper macro for py_library"""

load("@rules_python//python:defs.bzl", "py_library")

_MAX_PYTHON_ROOT = "max/python/max/"
_IGNORED_PACKAGES = [
    "max/python/max/_core/internal/mlir_nanobind/tblgen",
]

def modular_py_library(
        name,
        visibility = None,
        imports = None,
        tags = [],
        **kwargs):
    """Creates a py_library target

    Args:
        name: The name of the underlying py_library
        visibility: The visibility of the target, defaults to public
        imports: The imports path. For max/python/max packages, this is
            automatically computed and should not be passed.
        tags: Tags to add to the target
        **kwargs: Extra arguments passed through to py_library
    """
    package_name = native.package_name()
    if (package_name + "/").startswith(_MAX_PYTHON_ROOT) and package_name not in _IGNORED_PACKAGES:
        if imports != None:
            fail(
                "Do not pass 'imports' to modular_py_library for packages " +
                "under {}. The imports path is automatically computed.".format(_MAX_PYTHON_ROOT),
            )

        # max/python/max/foo/bar -> ../..
        relative_path = package_name.removeprefix("max/python/")
        depth = len(relative_path.split("/"))
        imports = ["/".join([".."] * depth)]

    if "manual" in tags:
        fail("modular_py_library targets cannot be manual. Remove 'manual' from the tags list.")

    py_library(
        name = name,
        visibility = visibility,
        imports = imports,
        tags = tags,
        **kwargs
    )
