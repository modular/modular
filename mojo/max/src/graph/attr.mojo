# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .mobicc import AttrMapPtr, AttrPtr


@value
struct AttrMap:
    var m: AttrMapPtr

    # ===------------------------------------------------------------------=== #
    # Basic constructors
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self, *attrs: AttrPtr):
        self.__init__(attrs)

    fn __init__(inout self, attrs: VariadicList[AttrPtr]):
        self.m = mobicc.attr_map_new()
        for i in range(len(attrs)):
            mobicc.attr_map_add_attr(self.m, attrs[i])

    # ===------------------------------------------------------------------=== #
    # Basic accessors
    # ===------------------------------------------------------------------=== #

    fn __len__(self) -> Int:
        return mobicc.attr_map_size(self.m)
