# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .capi import AttrMapPtr

import mlir


@value
struct AttrMap:
    var m: AttrMapPtr

    # ===------------------------------------------------------------------=== #
    # Basic constructors
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self, *attrs: mlir.NamedAttribute):
        self.m = capi.attr_map_new()
        for attr in attrs:
            capi.attr_map_add_attr(self.m, attr[])

    # TODO: parser crash
    #     self.__init__(attrs)

    # fn __init__(inout self, attrs: VariadicListMem[mlir.NamedAttribute]):
    #     self.m = capi.attr_map_new()
    #     for i in range(len(attrs)):
    #         capi.attr_map_add_attr(self.m, attrs[i])

    # ===------------------------------------------------------------------=== #
    # Basic accessors
    # ===------------------------------------------------------------------=== #

    fn __len__(self) -> Int:
        return capi.attr_map_size(self.m)
