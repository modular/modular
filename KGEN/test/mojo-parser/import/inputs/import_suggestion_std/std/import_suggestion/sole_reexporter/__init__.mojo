# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Sole exposer of `foreign_only_symbol`: re-exports it from the foreign sibling
module `std.import_suggestion.solo_mod` (an absolute cross-package import). No
package owns the name natively, so this re-exporter is the fallback suggestion."""

from std.import_suggestion.solo_mod import foreign_only_symbol
