"""Public API accessors to reduce the number of load statements needed in BUILD.bazel files."""

load("@rules_pkg//pkg:mappings.bzl", _pkg_filegroup = "pkg_filegroup", _pkg_files = "pkg_files", _strip_prefix = "strip_prefix")
load("//bazel/internal:copy_files.bzl", _copy_files = "copy_files")  # buildifier: disable=bzl-visibility
load("//bazel/internal:lit.bzl", _lit_tests = "lit_tests")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_cc_binary.bzl", _modular_cc_binary = "modular_cc_binary")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_cc_library.bzl", _modular_cc_library = "modular_cc_library")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_cc_test.bzl", _modular_cc_test = "modular_cc_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_multi_py_version_test.bzl", _modular_multi_py_version_test = "modular_multi_py_version_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_binary.bzl", _modular_py_binary = "modular_py_binary")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_library.bzl", _modular_py_library = "modular_py_library")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_test.bzl", _modular_py_test = "modular_py_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_py_venv.bzl", _modular_py_venv = "modular_py_venv")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_run_binary_test.bzl", _modular_run_binary_test = "modular_run_binary_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_sphinx_docs.bzl", _modular_sphinx_docs = "modular_sphinx_docs")  # buildifier: disable=bzl-visibility
load("//bazel/internal:modular_versioned_expand_template.bzl", _modular_versioned_expand_template = "modular_versioned_expand_template")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_binary.bzl", _mojo_binary = "mojo_binary")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_filecheck_test.bzl", _mojo_filecheck_test = "mojo_filecheck_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_library.bzl", _mojo_library = "mojo_library")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_shared_library.bzl", _mojo_shared_library = "mojo_shared_library")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_test.bzl", _mojo_test = "mojo_test")  # buildifier: disable=bzl-visibility
load("//bazel/internal:mojo_test_environment.bzl", _mojo_test_environment = "mojo_test_environment")  # buildifier: disable=bzl-visibility
load("//bazel/internal:py_repl.bzl", _py_repl = "py_repl")  # buildifier: disable=bzl-visibility
load("//bazel/pip:pip_requirement.bzl", _requirement = "pip_requirement")

lit_tests = _lit_tests
modular_multi_py_version_test = _modular_multi_py_version_test
modular_py_binary = _modular_py_binary
modular_py_library = _modular_py_library
modular_py_venv = _modular_py_venv
modular_run_binary_test = _modular_run_binary_test
modular_versioned_expand_template = _modular_versioned_expand_template
mojo_binary = _mojo_binary
mojo_library = _mojo_library
mojo_shared_library = _mojo_shared_library
mojo_test = _mojo_test
mojo_filecheck_test = _mojo_filecheck_test
modular_sphinx_docs = _modular_sphinx_docs
mojo_test_environment = _mojo_test_environment
pkg_files = _pkg_files
pkg_filegroup = _pkg_filegroup
py_repl = _py_repl
requirement = _requirement
strip_prefix = _strip_prefix

def modular_py_test(tags = [], **kwargs):
    if "external-exclusive" in tags:
        tags.append("exclusive")
    _modular_py_test(tags = tags, use_resource_tags = True, **kwargs)

def _process_cc_deps(data, deps):
    # TODO: This will break in the presence of select()s
    extra_data = []
    new_deps = []
    for dep in deps:
        if dep == "//max/internal:max":
            new_deps.append("@modular_wheel//:max_lib")
            extra_data.append("@modular_wheel//:wheel")
        else:
            new_deps.append(dep)

    return {
        "deps": new_deps,
        "data": data + extra_data,
    }

def modular_cc_binary(data = [], deps = [], **kwargs):
    _modular_cc_binary(
        **(kwargs | _process_cc_deps(
            data = data,
            deps = deps,
        ))
    )

def modular_cc_library(data = [], deps = [], **kwargs):
    _modular_cc_library(
        **(kwargs | _process_cc_deps(
            data = data,
            deps = deps,
        ))
    )

def modular_cc_test(data = [], deps = [], **kwargs):
    _modular_cc_test(
        **(kwargs | _process_cc_deps(
            data = data,
            deps = deps,
        ))
    )

def modular_generate_stubfiles(name, pyi_srcs, deps = [], tags = [], **_kwargs):
    modular_py_library(
        name = name,
        pyi_srcs = pyi_srcs,
        deps = deps + ["@modular_wheel//:wheel"],
        tags = tags + ["no-pydeps"],  # Pydeps works internally but not externally
    )

# buildifier: disable=function-docstring
def copy_files(srcs, **kwargs):
    new_srcs = []
    for src in srcs:
        if src.startswith("//GenericML:"):
            if "@modular_wheel//:tblgen_python_srcs" not in new_srcs:
                new_srcs.append("@modular_wheel//:tblgen_python_srcs")
        else:
            new_srcs.append(src)

    _copy_files(srcs = new_srcs, **kwargs)

def _noop(**_kwargs):
    pass

install_docs = _noop
modular_nanobind_extension = _noop
