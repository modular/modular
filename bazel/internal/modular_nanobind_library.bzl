"""Wrapper for the upstream nanobind_library rule to apply modular specific defaults."""

load(":modular_cc_library.bzl", "modular_cc_library")

def modular_nanobind_library(
        name,
        copts = [],
        deps = [],
        defines = [],
        **kwargs):
    """A wrapper for cc_library

    https://bazel.build/reference/be/c-cpp#cc_library

    Args:
        name: The name of the core target, the prefix of the name of the other python specific targets
        copts: See cc_library docs
        defines: See cc_library docs
        deps: See cc_library docs
        **kwargs: Everything else passed through to cc_library without processing
    """

    modular_cc_library(
        name = name,
        copts = copts + [
            "-fexceptions",
            "-fno-stack-protector",
            "-fno-strict-aliasing",
            "-frtti",
        ],
        defines = defines + select({
            "@rules_python//python/config_settings:is_py_freethreaded": ["NB_FREE_THREADED"],
            "//conditions:default": [],
        }),
        deps = deps + [
            "@nanobind",
        ],
        **kwargs
    )
