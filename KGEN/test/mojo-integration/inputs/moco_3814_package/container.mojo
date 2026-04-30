# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .inner import Inner


struct Container:
    var count: Int

    def __init__(out self, count: Int):
        self.count = count

    def get_count(self) -> Int:
        return self.count

    # This method references Inner in its return type but is not called by
    # the importing test.  When Container is body-resolved, this FnOp is
    # materialized (placed in parsedDeclList as 'unparsed') with Inner
    # appearing in its function-type attribute.
    def get_inner(self) -> Inner:
        return Inner(self.count)
