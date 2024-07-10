# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from . import IntTuple, Layout
from .runtime_tuple import RuntimeTuple, product, crd2idx
from .int_tuple import flatten, to_int, UNKNOWN_VALUE
from .layout import (
    coalesce as coalesce_layout,
    composition as composition_layout,
    is_tuple,
)


# A `Layout` like type that uses RuntimeTuple as its storage instead of
# IntTuple.


@register_passable("trivial")
struct RuntimeLayout[layout: Layout](Stringable, Formattable):
    var shape: RuntimeTuple[layout.shape]
    var stride: RuntimeTuple[layout.stride]

    fn __init__(inout self):
        constrained[
            layout.all_dims_known(), "Static layout with known dims is required"
        ]()
        self.shape = RuntimeTuple[layout.shape]()
        self.stride = RuntimeTuple[layout.stride]()

    fn __init__(
        inout self,
        shape: RuntimeTuple[layout.shape],
        stride: RuntimeTuple[layout.stride],
    ):
        self.shape = shape
        self.stride = stride

    @always_inline
    fn __call__[t: IntTuple](self, idx: RuntimeTuple[t]) -> Int:
        return crd2idx(idx, self.shape, self.stride)

    @always_inline
    fn size(self) -> Int:
        return product(self.shape)

    @no_inline
    fn __str__(self) -> String:
        return String.format_sequence(self)

    @no_inline
    fn format_to(self, inout f: Formatter):
        f.write_str["("]()
        self.shape.format_to(f)
        f.write_str[":"]()
        self.stride.format_to(f)
        f.write_str[")"]()

    fn sublayout[i: Int](self) -> RuntimeLayout[layout[i]]:
        return RuntimeLayout[layout[i]](
            rebind[RuntimeTuple[layout[i].shape]](self.shape[i]),
            rebind[RuntimeTuple[layout[i].stride]](self.stride[i]),
        )

    @staticmethod
    fn __len__() -> Int:
        return len(layout)


fn coalesce[
    l: Layout, keep_rank: Bool = False
](layout: RuntimeLayout[l]) -> RuntimeLayout[coalesce_layout(l, keep_rank)]:
    constrained[not keep_rank, "Unsupported coalesce mode"]()

    var res_shape = RuntimeTuple[coalesce_layout(l, keep_rank).shape]()
    var res_stride = RuntimeTuple[coalesce_layout(l, keep_rank).stride]()

    res_shape.value[0] = 1
    res_stride.value[0] = 0

    var idx = 0

    @parameter
    for i in range(len(flatten(l.shape))):
        alias shape = to_int(l.shape[i])
        alias stride = to_int(l.stride[i])

        # If dynamic, append new mode
        if UNKNOWN_VALUE in (shape, stride):
            res_shape.value[idx] = layout.shape.value[i]
            res_stride.value[idx] = layout.stride.value[i]
            idx += 1
            continue

        # skip their shape-1s
        if shape == 1:
            continue

        # replace our shape-1 with anything
        if res_shape.value[idx] == 1:
            res_shape.value[idx] = layout.shape.value[i]
            res_stride.value[idx] = layout.stride.value[i]

        # merge modes if the shape*stride match
        elif res_shape.value[idx] * res_shape.value[idx] == stride:
            res_shape.value[idx] = res_shape.value[idx] * shape
        # append a new mode
        else:
            res_shape.value[idx] = layout.shape.value[i]
            res_stride.value[idx] = layout.stride.value[i]
            idx += 1

    return RuntimeLayout[coalesce_layout(l, keep_rank)](res_shape, res_stride)
