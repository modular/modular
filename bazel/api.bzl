"""Public API accessors to reduce the number of load statements needed in BUILD.bazel files."""

load("@com_github_grpc_grpc//bazel:python_rules.bzl", _py_grpc_library = "py_grpc_library")
load("@rules_cc//cc:cc_binary.bzl", _cc_binary = "cc_binary")
load("@rules_cc//cc:cc_library.bzl", _cc_library = "cc_library")
load("@rules_pkg//pkg:mappings.bzl", _pkg_filegroup = "pkg_filegroup", _pkg_files = "pkg_files", _strip_prefix = "strip_prefix")
load("@rules_proto//proto:defs.bzl", _proto_library = "proto_library")
load("//bazel/internal:lit.bzl", _lit_tests = "lit_tests")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_multi_py_version_test.bzl", _modular_multi_py_version_test = "modular_multi_py_version_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_binary.bzl", _modular_py_binary = "modular_py_binary")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_library.bzl", _modular_py_library = "modular_py_library")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_test.bzl", _modular_py_test = "modular_py_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_venv.bzl", _modular_py_venv = "modular_py_venv")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_run_binary_test.bzl", _modular_run_binary_test = "modular_run_binary_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_binary.bzl", _mojo_binary = "mojo_binary")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_filecheck_test.bzl", _mojo_filecheck_test = "mojo_filecheck_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_library.bzl", _mojo_library = "mojo_library")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_test.bzl", _mojo_test = "mojo_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_test_environment.bzl", _mojo_test_environment = "mojo_test_environment")  # buildifier: disable=bzl-visibility
load("//bazel/pip:pip_requirement.bzl", _requirement = "pip_requirement")

lit_tests = _lit_tests
modular_cc_binary = _cc_binary
modular_cc_library = _cc_library
modular_multi_py_version_test = _modular_multi_py_version_test
modular_py_binary = _modular_py_binary
modular_py_venv = _modular_py_venv
mojo_binary = _mojo_binary
mojo_library = _mojo_library
mojo_test = _mojo_test
mojo_filecheck_test = _mojo_filecheck_test
mojo_test_environment = _mojo_test_environment
pkg_files = _pkg_files
pkg_filegroup = _pkg_filegroup
proto_library = _proto_library
py_grpc_library = _py_grpc_library
requirement = _requirement
strip_prefix = _strip_prefix

def _is_internal_reference(dep):
    """Check if a dependency is an internal reference."""
    return dep.startswith((
        "//max/tests/integration:",
        "//max/tests/integration/pipelines/python",
    ))

def _has_internal_reference(deps):
    return any([_is_internal_reference(dep) for dep in deps])

# buildifier: disable=function-docstring
def modular_py_library(
        name,
        deps = [],
        visibility = ["//visibility:public"],
        **kwargs):
    if name == "_mlir":
        native.alias(name = name, actual = "@modular_wheel//:wheel", visibility = visibility)
        return

    if _has_internal_reference(deps):
        return

    _modular_py_library(
        name = name,
        deps = deps,
        visibility = visibility,
        **kwargs
    )

# buildifier: disable=function-docstring
def modular_py_test(
        deps = [],
        data = [],
        **kwargs):
    if _has_internal_reference(deps) or _has_internal_reference(data):
        return

    _modular_py_test(
        data = data,
        deps = deps,
        **kwargs
    )

# buildifier: disable=function-docstring
def modular_run_binary_test(external_noop = False, **kwargs):
    if external_noop:
        return

    _modular_run_binary_test(
        **kwargs
    )

def modular_generate_stubfiles(name, **_kwargs):
    native.alias(name = name, actual = "@modular_wheel//:wheel", visibility = ["//visibility:public"])

def _noop(**_kwargs):
    pass

copy_files = _noop
mojo_kgen_lib = _noop
modular_nanobind_extension = _noop
