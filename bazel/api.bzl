"""Public API accessors to reduce the number of load statements needed in BUILD.bazel files."""

load("@modular_test_deps//:requirements.bzl", _requirement = "requirement")
load("@rules_mojo//mojo:mojo_binary.bzl", _mojo_binary = "mojo_binary")
load("@rules_mojo//mojo:mojo_library.bzl", _mojo_library = "mojo_library")
load("@rules_mojo//mojo:mojo_test.bzl", _mojo_test = "mojo_test")
load("@rules_python//python:py_library.bzl", "py_library")
load("//bazel/internal:binary_test.bzl", "binary_test")

modular_py_library = py_library
modular_run_binary_test = binary_test
mojo_binary = _mojo_binary
mojo_test = _mojo_test
requirement = _requirement

# buildifier: disable=function-docstring
def mojo_library(
        validate_missing_docs = False,  # buildifier: disable=unused-variable
        build_docs = False,  # buildifier: disable=unused-variable
        target_compatible_with = [],
        deps = [],
        **kwargs):
    extra_target_compatible_with = []
    if any([dep.startswith(("//Kernels/", "//SDK/")) for dep in deps]):
        extra_target_compatible_with = ["@platforms//:incompatible"]
        deps = []

    _mojo_library(
        deps = deps,
        target_compatible_with = target_compatible_with + extra_target_compatible_with,
        **kwargs
    )

def lit_tests(**_kwargs):
    pass

def mojo_doc(**_kwargs):
    pass

def modular_py_binary(**_kwargs):
    pass

def mojo_kgen_lib(**_kwargs):
    pass
