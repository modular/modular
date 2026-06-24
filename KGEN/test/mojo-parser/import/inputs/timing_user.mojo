# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import timing_pkg.sub
import timing_pkg.nested.leaf


# Body use of the dotted imports.
def body_use():
    var _x = timing_pkg.sub.VALUE
    var _y = timing_pkg.nested.leaf.DEEP


# Default-argument use of the same dotted imports. The full path is navigated,
# so every segment (timing_pkg, timing_pkg.nested, timing_pkg.nested.leaf) must
# be bound — including the middle `nested`.
def default_use(
    x: Int = timing_pkg.sub.VALUE, y: Int = timing_pkg.nested.leaf.DEEP
):
    pass
