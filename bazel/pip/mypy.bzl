"""Configuration for mypy aspect"""

load("@rules_mypy//mypy:mypy.bzl", "mypy")

_TYPES_DEPS = [
    "protobuf",
    "pyyaml",
    "setuptools",
    "tabulate",
]

_TYPES = {
    "@@rules_pycross++lock_file+modular_pip_lock_file_repo//deps:" + dep: "@@rules_pycross++lock_file+modular_pip_lock_file_repo//deps:types-" + dep
    for dep in _TYPES_DEPS
}

mypy_aspect = mypy(
    mypy_ini = "@@//:mypy.ini",
    mypy_cli = "@@//bazel/internal:mypy_cli",
    suppression_tags = ["no-mypy"],
    types = _TYPES,
    color = False,
    cache = False,
)
