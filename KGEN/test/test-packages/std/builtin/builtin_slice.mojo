# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct Slice(ImplicitlyCopyable):
    @always_inline
    def __init__(
        out self,
        start: Optional[Int],
        end: Optional[Int],
        step: Optional[Int],
        __slice_literal__: NoneType = None,
    ):
        pass


struct ContiguousSlice(ImplicitlyCopyable):
    @always_inline
    def __init__(
        out self,
        start: Optional[Int],
        end: Optional[Int],
        stride: NoneType,
        __slice_literal__: NoneType = None,
    ):
        pass
