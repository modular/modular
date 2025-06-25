"""Public API accessors to reduce the number of load statements needed in BUILD.bazel files."""

load("@aspect_rules_py//py:defs.bzl", "py_binary", "py_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", _py_grpc_library = "py_grpc_library")
load("@protobuf//bazel:py_proto_library.bzl", _py_proto_library = "py_proto_library")
load("@rules_mojo//mojo:mojo_binary.bzl", _mojo_binary = "mojo_binary")
load("@rules_mojo//mojo:mojo_library.bzl", _mojo_library = "mojo_library")
load("@rules_mojo//mojo:mojo_test.bzl", _mojo_test = "mojo_test")
load("@rules_pkg//pkg:mappings.bzl", _strip_prefix = "strip_prefix")
load("@rules_proto//proto:defs.bzl", _proto_library = "proto_library")
load("//bazel/internal:binary_test.bzl", "binary_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_filecheck_test.bzl", _mojo_filecheck_test = "mojo_filecheck_test")  # buildifier: disable=bzl-visibility
load("//bazel/pip:pip_requirement.bzl", _requirement = "pip_requirement")

mojo_filecheck_test = _mojo_filecheck_test
mojo_test = _mojo_test
proto_library = _proto_library
py_grpc_library = _py_grpc_library
py_proto_library = _py_proto_library
requirement = _requirement
strip_prefix = _strip_prefix

# Deps that aren't open source so we need to remap to pull from the wheel instead.
_DEPS_FROM_WHEEL = [
    "//max",
    "//max/driver",
    "//max/dtype",
    "//max/engine",
    "//max/graph",
    "//max/mojo",
    "//max/profiler",
    "//max/support",
    "//max:_core",
]

def _is_internal_reference(dep):
    """Check if a dependency is an internal reference."""
    return dep.startswith(("//GenericML", "//KGEN/", "//Kernels/", "//Support/"))

def _has_internal_reference(deps):
    return any([_is_internal_reference(dep) for dep in deps])

def _remove_internal_data(data):
    # TODO: This is a hack that potentially breaks things at runtime
    if type(data) != type([]):
        return []
    return [d for d in data if not _is_internal_reference(d)]

def _rewrite_deps(deps):
    """Rewrite dependencies to use the open-source package names, or to come from the wheel."""
    new_deps = []
    for dep in deps:
        if dep.startswith("//SDK/lib/API/python/"):
            replaced_dep = dep.replace("//SDK/lib/API/python/", "//")
            if replaced_dep in _DEPS_FROM_WHEEL:
                replaced_dep = "@modular_wheel//:wheel"
            if replaced_dep not in new_deps:
                new_deps.append(replaced_dep)
        else:
            new_deps.append(dep)
    return new_deps

def modular_py_library(
        data = [],
        deps = [],
        visibility = ["//visibility:public"],
        **kwargs):
    py_library(
        data = _remove_internal_data(data),
        deps = _rewrite_deps(deps),
        visibility = visibility,
        **kwargs
    )

# buildifier: disable=function-docstring
def modular_py_binary(
        name,
        deps = [],
        data = [],
        env = {},
        visibility = ["//visibility:public"],
        use_sitecustomize = False,  # buildifier: disable=unused-variable # TODO: support
        mojo_deps = [],  # buildifier: disable=unused-variable # TODO: support
        **kwargs):
    if name == "pipelines":
        # TODO: Fix this hack, there is a layering issue with what is open source right now
        deps.append("//max/entrypoints:mojo")
        data = []
        env = {}

    # TODO: There is some data we can fix by pulling from the wheel
    if _has_internal_reference(deps) or _has_internal_reference(data):
        return

    py_binary(
        name = name,
        data = data,
        env = env,
        deps = _rewrite_deps(deps),
        visibility = visibility,
        **kwargs
    )

# buildifier: disable=function-docstring
def mojo_library(
        validate_missing_docs = False,  # buildifier: disable=unused-variable
        build_docs = False,  # buildifier: disable=unused-variable
        deps = [],
        **kwargs):
    if _has_internal_reference(deps):
        return

    _mojo_library(
        deps = deps,
        **kwargs
    )

def mojo_binary(
        data = [],
        deps = [],
        **kwargs):
    if _has_internal_reference(deps) or _has_internal_reference(data):
        return
    _mojo_binary(
        data = data,
        deps = deps,
        **kwargs
    )

# buildifier: disable=function-docstring
def modular_run_binary_test(name, external_noop = False, **kwargs):
    if external_noop:
        return
    if name.endswith(".example-test"):
        return  # TODO: Fix custom_ops python examples
    binary_test(
        name = name,
        **kwargs
    )

def _noop(**_kwargs):
    pass

lit_tests = _noop
modular_py_test = _noop
mojo_doc = _noop
mojo_kgen_lib = _noop
pkg_attributes = _noop
pkg_filegroup = _noop
pkg_files = _noop
