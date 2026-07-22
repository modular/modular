# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Convenience aggregator: re-exports `native_owned_symbol` from the foreign
package `std.import_suggestion.owner` (an absolute cross-package import), the
way `std.ffi` re-exports `std.os.abort`. The suggestion must prefer the native
owner (`std.import_suggestion.owner`), not this aggregator."""

from std.import_suggestion.owner import native_owned_symbol
