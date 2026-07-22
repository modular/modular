# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Native owner of `native_owned_symbol`: declares it directly. Paired with
`aggregator`, which re-exports the same name from this foreign package, to test
that a native owner is preferred over a convenience re-exporter."""


def native_owned_symbol():
    pass
