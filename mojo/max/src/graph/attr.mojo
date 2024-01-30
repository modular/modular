# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import mlir


@value
struct AttrMap(Sized):
    var attrs: DynamicVector[mlir.NamedAttribute]

    # ===------------------------------------------------------------------=== #
    # Basic constructors
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self, *attrs: mlir.NamedAttribute):
        self.attrs = DynamicVector[mlir.NamedAttribute]()
        for attr in attrs:
            self.attrs.append(attr[])

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
        return len(self.attrs)
