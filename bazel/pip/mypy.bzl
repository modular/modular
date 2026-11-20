"""Configuration for mypy aspect"""

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@rules_mypy//mypy:mypy.bzl", "mypy")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

_TYPES_DEPS = [
    "protobuf",
    "pyyaml",
    "setuptools",
    "tabulate",
]

_PIP_REPO = "@@rules_pycross++lock_file+modular_pip_lock_file_repo//deps:"

# Each type-stub mapping goes through an alias made by mypy_toggle() below.
# (An alias reports the label of its target, so mypy matches stubs as before.)
_TYPES = {
    "@@//bazel/pip:mypy_types_key_" + dep: "@@//bazel/pip:mypy_types_value_" + dep
    for dep in _TYPES_DEPS
}

mypy_aspect = mypy(
    mypy_ini = "@@//:pyproject.toml",
    mypy_cli = "@@//bazel/pip:mypy_cli",
    suppression_tags = ["no-mypy"],
    types = _TYPES,
    color = False,
    cache = False,
)

# buildifier: disable=unnamed-macro
def mypy_toggle():
    """Defines the targets that let --config=disable-mypy turn mypy off.

    mypy and its pip packages are reached through aliases. Normally they point
    at the real targets. With --config=disable-mypy they point at empty
    placeholders, so Bazel never has to load the pip packages at all.
    """
    bool_flag(
        name = "enable_mypy",
        build_setting_default = True,
    )

    native.config_setting(
        name = "mypy_enabled",
        flag_values = {":enable_mypy": "True"},
    )

    sh_binary(
        name = "mypy_disabled_cli",
        srcs = ["mypy_disabled_cli.sh"],
        tags = ["no-mypy"],
    )

    native.filegroup(
        name = "mypy_disabled_empty",
        srcs = [],
    )

    _toggle("mypy_cli", "//bazel/internal:mypy_cli", ":mypy_disabled_cli")
    for dep in _TYPES_DEPS:
        _toggle("mypy_types_key_" + dep, _PIP_REPO + dep, ":mypy_disabled_empty")
        _toggle("mypy_types_value_" + dep, _PIP_REPO + "types-" + dep, ":mypy_disabled_empty")

def _toggle(name, target, placeholder):
    """An alias to `target`, or to `placeholder` when mypy is turned off."""
    native.alias(
        name = name,
        actual = select({
            ":mypy_enabled": target,
            "//conditions:default": placeholder,
        }),
    )
