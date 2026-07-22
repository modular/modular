# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Declares `foreign_only_symbol`, which no package re-exports from its own
subtree — only the foreign `sole_reexporter` package surfaces it. Exercises the
foreign-only fallback: with no native owner, the re-exporter is suggested."""


def foreign_only_symbol():
    pass
