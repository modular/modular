# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


struct IndexList[size: Int, *, element_type: DType = DType.int]():
    @always_inline
    def __init__(out self, *elems: Int, __list_literal__: NoneType = None):
        pass

    @always_inline
    def __init__(out self, fill: Int):
        pass


struct SomeThing[rank: Int](Movable where False):
    def __init__(out self):
        pass


def baz[
    rank: Int,
    *,
    output_shape: IndexList[rank + 1] = IndexList[rank + 1](fill=0),
](cfg: SomeThing[rank]):
    pass


def main():
    var cfg = SomeThing[2]()

    # `output_shape` must be inferred to the explicitly-provided
    # `IndexList[3](1, 2, 3)`, not the `fill=0` default.
    #
    # CHECK: lit.call @{{.*}}::@"baz
    # CHECK-SAME: <:!Int {:scalar<index> 2}, :!lit.struct<#IndexList <:!Int {:scalar<index> 3}, :!DType {:dtype index}>>
    # CHECK-SAME: store_to_mem([store_to_mem({:scalar<index> 1}), store_to_mem({:scalar<index> 2}), store_to_mem({:scalar<index> 3})])
    baz[output_shape=IndexList[3](1, 2, 3)](cfg)
