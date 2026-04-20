"""Wrapper macro for py_library"""

load("@rules_python//python:defs.bzl", "py_library")
load("//bazel/pip/pydeps:pydeps_test.bzl", "pydeps_test")

_MAX_PYTHON_ROOT = "max/python/max/"
_IGNORED_PACKAGES = [
    "max/python/max/_core/internal/mlir_nanobind/tblgen",
]

def modular_py_library(
        name,
        visibility = None,
        ignore_extra_deps = [],
        ignore_unresolved_imports = [],
        imports = [],
        tags = [],
        **kwargs):
    """Creates a py_library target

    Args:
        name: The name of the underlying py_library
        visibility: The visibility of the target, defaults to public
        ignore_extra_deps: Forwarded to pydeps_test
        ignore_unresolved_imports: Forwarded to pydeps_test
        imports: The imports path. For max/python/max packages, this is
            automatically computed and should not be passed.
        tags: Tags to add to the target
        **kwargs: Extra arguments passed through to py_library
    """
    package_name = native.package_name()
    if (package_name + "/").startswith(_MAX_PYTHON_ROOT) and package_name not in _IGNORED_PACKAGES:
        if len(imports) > 0:
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

    if len(imports) > 1:
        fail("modular_py_library only supports a single import path.")

    py_library(
        name = name,
        visibility = visibility,
        imports = imports,
        tags = tags,
        **kwargs
    )

    if "no-pydeps" not in tags:
        pydeps_test(
            name = name + ".pydeps_test",
            deps = kwargs.get("deps", []),
            data = kwargs.get("data", []),
            ignore_extra_deps = ignore_extra_deps,
            ignore_unresolved_imports = ignore_unresolved_imports,
            imports = imports if imports != None else [],
            srcs = kwargs.get("srcs", []) + kwargs.get("pyi_srcs", []),
            tags = ["pydeps"],
        )
