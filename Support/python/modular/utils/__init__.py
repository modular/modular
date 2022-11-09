# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

__doc__ = """
=======================
Modular utility library
=======================

This library is a collection of various modules that each provide utilities to
be used by Python scripts and packages across our codebase. Since, by
definition, this library is heavily depended on, it strives to provide reliable,
modular, and well documented functionality.

This library is a subpackage of the `modular` namespace package, but it is not
a namespace package itself. It is, however, implemented in such a way that it
avoids loading submodules unless users explicitly request them. This allows one
to provide utility modules with exotic dependencies that may be optional, or
slow to load. For more context and rationale for this design, see:
https://www.notion.so/modularai/Modular-Python-support-library-045f201fa45d4b1e8769944f28367ffe

With the exception of the `modular.utils.misc` module, each utility module is
expected to provide rationale for its scope and existence.

For the time being there are no strict rules on what necessitates a new module.
There are a few cases where the decision is relatively simple:
* Wrappers around standard library modules (e.g. typing, logging)
* Utilities complementing standard library modules (e.g. subprocess)
* Python interfaces for commonly used tools (e.g. git, gh)
* Modules to hide external dependencies (e.g. yaml)
"""
