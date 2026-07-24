# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Round-trip test: actually apply the fix-it in place (not just export its
# replacement text to YAML, as missing_movable_conformance_fixit.mojo does)
# and confirm the mutated source is both textually correct and still valid,
# warning-free Mojo.

# RUN: %parse-mojo-isolated %s -mojo-diagnose-missing-movable-conformance -experimental-fixit -o /dev/null | FileCheck %s --check-prefix=APPLIED
# APPLIED: Fixits applied.

# Re-running against the now-fixed-up source must find nothing left to warn
# about or fix -- this is what actually proves the fix-it produces valid,
# conforming Mojo, not just correct replacement text.
# RUN: %parse-mojo-isolated %s -mojo-diagnose-missing-movable-conformance -experimental-fixit -o /dev/null | FileCheck %s --check-prefix=IDEMPOTENT
# IDEMPOTENT: No fixits to apply.
# RUN: %parse-mojo-isolated %s -mojo-diagnose-missing-movable-conformance -verify-diagnostics

# The mutated source itself must read back with the synthesized conformance
# list in the right place -- and, for RefinesMovableTransitively below, must
# read back completely unchanged.
# RUN: cat %s | grep -v "^#" | FileCheck %s --check-prefix=SOURCE

# SOURCE: struct NoConformanceList(Movable where False):
struct NoConformanceList:
    pass


# Already conforms to Movable transitively (RegisterPassable refines
# Movable), so the fix-it must leave this struct untouched even though
# another struct in the same file does need fixing up.
# SOURCE: struct RefinesMovableTransitively(RegisterPassable):
struct RefinesMovableTransitively(RegisterPassable):
    pass
