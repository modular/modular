# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# A module inside a package cannot reach a sibling module - or a sibling's
# symbols - by bare name without an explicit import. Siblings are unlisted
# children of the package, so a contained file's upward name lookup walks up
# into the package's (empty) scope and finds nothing; it never sees a sibling,
# nor __init__'s re-exports. The package's `__init__.mojo` deliberately
# re-exports `producer`, which makes `no_sibling_leak.producer` reachable from
# *outside* but must NOT make `producer` visible to its siblings.
#
# As a migration aid, a bare reference to a sibling *module* still resolves for
# one release, with a deprecation warning (it will become a hard error later); a
# bare reference to a sibling's *symbol* was never reachable this way and is an
# error. See inputs/no_sibling_leak/.

# RUN: not %parse-mojo-isolated -split-input-file -I=%S/inputs %s 2>&1 | FileCheck %s

# A sibling MODULE resolves with a deprecation warning (temporary migration aid).
from no_sibling_leak.consumer_module import consume

def main():
    _ = consume()

# CHECK: consumer_module.mojo:{{[0-9]+}}:{{[0-9]+}}: warning: implicit reference to sibling module 'producer' without an import is deprecated

# // -----

# A sibling's SYMBOL that __init__ does NOT re-export is not visible without an
# import - it never leaked into the package scope, so this stays a hard error.
from no_sibling_leak.consumer_symbol import consume

def main():
    _ = consume()

# CHECK: consumer_symbol.mojo:{{[0-9]+}}:{{[0-9]+}}: error: use of unknown declaration 'producer_fn'

# // -----

# A symbol that __init__ DOES re-export is reachable from a sibling only via the
# same temporary deprecation warning (the package used to wildcard-import
# __init__ into its scope). It too becomes a hard error in a future release.
from no_sibling_leak.consumer_reexport import consume

def main():
    _ = consume()

# CHECK: consumer_reexport.mojo:{{[0-9]+}}:{{[0-9]+}}: warning: implicit reference to 'reexported_fn' from the enclosing package's '__init__' is deprecated
