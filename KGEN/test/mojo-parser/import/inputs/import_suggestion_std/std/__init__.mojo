# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Minimal standalone `std` for the missing-import suggestion tests (MOCO-1051).

This is a purpose-built, fully self-contained stand-in for the standard library,
precompiled to a `std.mojoc` so the tests exercise the feature's real bytecode
path (the form shipped to users). It is intentionally tiny: the Mojo compiler
only hard-requires the `std`, `std.prelude`, and `std.builtin` packages to
*exist*, so their `__init__.mojo` files can be empty. The `import_suggestion`
sub-package holds the actual test fixtures.
"""
