# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from utils import IndexList
from sys import bitwidthof

from . import IntTuple, Layout
from .int_tuple import UNKNOWN_VALUE, flatten, to_int
from .layout import coalesce as coalesce_layout
from .layout import composition as composition_layout
from .layout import is_tuple
from .layout import make_layout as make_layout_static
from .runtime_tuple import RuntimeTuple, crd2idx, product

# A `Layout` like type that uses RuntimeTuple as its storage instead of
# IntTuple.


@register_passable("trivial")
struct RuntimeLayout[layout: Layout, /, *, bitwidth: Int = bitwidthof[Int]()](
    Stringable, Writable
):
    var shape: RuntimeTuple[
        layout.shape, element_bitwidth=bitwidth, unsigned=True
    ]
    # stride can be huge so default to int64 for now
    var stride: RuntimeTuple[layout.stride, unsigned=True]

    @always_inline
    fn __init__(inout self):
        constrained[
            layout.all_dims_known(), "Static layout with known dims is required"
        ]()
        self.shape = RuntimeTuple[
            layout.shape, element_bitwidth=bitwidth, unsigned=True
        ]()
        self.stride = RuntimeTuple[layout.stride, unsigned=True]()

    @always_inline
    fn __init__(
        inout self,
        shape: RuntimeTuple[
            layout.shape, element_bitwidth=bitwidth, unsigned=True
        ],
        stride: RuntimeTuple[layout.stride, unsigned=True],
    ):
        self.shape = shape
        self.stride = stride

    @always_inline
    fn __call__[t: IntTuple](self, idx: RuntimeTuple[t, **_]) -> Int:
        return crd2idx(idx, self.shape, self.stride)

    @always_inline
    fn size(self) -> Int:
        return product(self.shape)

    @always_inline
    fn cast[
        type: DType
    ](self) -> RuntimeLayout[layout, bitwidth = bitwidthof[type]()] as result:
        return __type_of(result)(self.shape.cast[type](), self.stride)

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @staticmethod
    fn row_major[
        rank: Int, //
    ](shape: IndexList[rank, **_]) -> RuntimeLayout[
        layout, bitwidth = shape.element_bitwidth
    ] as result:
        var stride = __type_of(shape)()
        var c_stride = 1
        stride[rank - 1] = c_stride

        @parameter
        for i in reversed(range(rank - 1)):
            var dim = shape[i + 1]
            stride[i] = dim * c_stride
            c_stride *= dim
        return __type_of(result)(shape, stride)

    @staticmethod
    fn col_major[
        rank: Int, //
    ](shape: IndexList[rank, **_]) -> RuntimeLayout[
        layout, bitwidth = shape.element_bitwidth
    ] as result:
        var stride = __type_of(shape)()
        var c_stride = 1
        stride[0] = c_stride

        @parameter
        for i in range(1, rank):
            var dim = shape[i - 1]
            stride[i] = dim * c_stride
            c_stride *= dim
        return __type_of(result)(shape, stride)

    @no_inline
    fn write_to[W: Writer](self, inout writer: W):
        writer.write("(")
        writer.write(self.shape)
        writer.write(":")
        writer.write(self.stride)
        writer.write(")")

    fn sublayout[
        i: Int
    ](self) -> RuntimeLayout[layout[i], bitwidth=bitwidth] as result:
        return __type_of(result)(
            rebind[
                RuntimeTuple[
                    layout[i].shape,
                    element_bitwidth=bitwidth,
                    unsigned=True,
                ],
            ](self.shape[i]),
            rebind[RuntimeTuple[layout[i].stride, unsigned=True]](
                self.stride[i]
            ),
        )

    fn dim(self, i: Int) -> Int:
        return self.shape.value[i]

    @staticmethod
    fn __len__() -> Int:
        return len(layout)


fn coalesce[
    l: Layout,
    keep_rank: Bool = False,
](layout: RuntimeLayout[l, **_]) -> RuntimeLayout[
    coalesce_layout(l, keep_rank), bitwidth = layout.bitwidth
] as result:
    constrained[not keep_rank, "Unsupported coalesce mode"]()

    var res_shape = RuntimeTuple[
        coalesce_layout(l, keep_rank).shape,
        element_bitwidth = layout.bitwidth,
        unsigned=True,
    ]()
    var res_stride = RuntimeTuple[
        coalesce_layout(l, keep_rank).stride,
        unsigned=True,
    ]()

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

    return __type_of(result)(res_shape, res_stride)


fn make_layout[
    l1: Layout, l2: Layout
](
    a: RuntimeLayout[l1, **_],
    b: RuntimeLayout[l2, **_],
) -> RuntimeLayout[
    make_layout_static(l1, l2), bitwidth = b.bitwidth
] as result:
    var res_shape = RuntimeTuple[
        make_layout_static(l1, l2).shape,
        element_bitwidth = b.bitwidth,
        unsigned=True,
    ]()
    var res_stride = RuntimeTuple[
        make_layout_static(l1, l2).stride, unsigned=True
    ]()

    alias a_length = len(flatten(l1.shape))
    alias b_length = len(flatten(l2.shape))

    @parameter
    for i in range(a_length):
        res_shape.value[i] = a.shape.value[i]
        res_stride.value[i] = a.stride.value[i]

    @parameter
    for i in range(b_length):
        res_shape.value[a_length + i] = b.shape.value[i]
        res_stride.value[a_length + i] = b.stride.value[i]

    return __type_of(result)(res_shape, res_stride)
