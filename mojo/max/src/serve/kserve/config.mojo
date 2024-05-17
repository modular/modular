# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides compile-time and runtime config operations."""

from sys.param_env import is_defined

alias STATS_ENABLED = is_defined["MODULAR_MAX_SERVE_STATS_ENABLED"]()
alias BATCH_HEAT_MAP_ENABLED = is_defined[
    "MODULAR_MAX_SERVE_BATCH_HEAT_MAP_ENABLED"
]()
