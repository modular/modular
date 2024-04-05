# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from builtin.io import _printf
from builtin.string import _calc_initial_buffer_size_int32

from .dynamic_tuple import *
from .int_tuple import (
    IntTuple,
    abs,
    crd2idx,
    flatten,
    inner_product,
    int,
    is_int,
    is_tuple,
    min,
    mul,
    prefix_product,
    product,
    reverse,
    shape_div,
    sorted,
    zip,
)


fn make_layout(*layouts: Layout) -> Layout:
    var shape = IntTuple()
    var stride = IntTuple()
    for i in range(len(layouts)):
        shape.append(layouts[i].shape)
        stride.append(layouts[i].stride)
    return Layout(shape, stride)


@value
struct _LayoutIter:
    var index: Int
    var layout: Layout

    fn __next__(inout self) -> Layout:
        self.index += 1
        return Layout(
            self.layout.shape[self.index - 1],
            self.layout.stride[self.index - 1],
        )

    fn __len__(self) -> Int:
        return len(self.layout.shape) - self.index


struct Layout(Sized, Stringable, CollectionElement, EqualityComparable):
    var shape: IntTuple
    var stride: IntTuple

    @always_inline
    fn __init__(inout self):
        self.shape = IntTuple()
        self.stride = IntTuple()

    @always_inline
    fn __init__(inout self, shape: IntTuple, stride: IntTuple = IntTuple()):
        self.shape = shape
        if len(stride) == 0:
            self.stride = prefix_product(self.shape)
        else:
            self.stride = stride

    @staticmethod
    fn col_major(*dims: Int) -> Layout:
        var shape = IntTuple()
        var stride = IntTuple()
        var c_stride = 1
        for dim in dims:
            stride.append(c_stride)
            shape.append(dim)
            c_stride *= dim
        return Layout(shape, stride)

    @staticmethod
    fn row_major(*dims: Int) -> Layout:
        var shape = IntTuple()
        var stride = IntTuple()
        var c_stride = 1
        stride.append(c_stride)
        for i in range(len(dims) - 1):
            var dim = dims[len(dims) - 1 - i]
            stride.append(dim * c_stride)
            c_stride *= dim
        for dim in dims:
            shape.append(dim)
        return Layout(shape, reverse(stride))

    @always_inline
    fn __moveinit__(inout self: Self, owned existing: Self):
        self.shape = existing.shape^
        self.stride = existing.stride^

    @always_inline
    fn __copyinit__(inout self, existing: Self):
        self.shape = existing.shape
        self.stride = existing.stride

    fn __str__(self) -> String:
        return "(" + str(self.shape) + ":" + str(self.stride) + ")"

    fn __eq__(self, other: Layout) -> Bool:
        return self.shape == other.shape and self.stride == other.stride

    fn __ne__(self, other: Layout) -> Bool:
        return not (self == other)

    @always_inline
    fn __len__(self) -> Int:
        return len(self.shape)

    @always_inline
    fn __iter__(self) -> _LayoutIter:
        return _LayoutIter(0, self)

    @always_inline
    fn size(self) -> Int:
        return product(self.shape)

    @always_inline
    fn cosize(self) -> Int:
        return self(self.size() - 1) + 1
        # return math.max(1, inner_product(self.shape, self.stride))

    @always_inline
    fn __getitem__(self, index: Int) -> Self:
        return Layout(self.shape[index], self.stride[index])

    @always_inline
    fn rank(self) -> Int:
        return len(self.shape)

    @always_inline
    fn __call__(self, idx: IntTuple) -> Int:
        return crd2idx(idx, self.shape, self.stride)

    @always_inline
    fn append(inout self, item: Layout):
        self.shape.append(item.shape)
        self.stride.append(item.stride)


fn size(l: Layout) -> Int:
    return l.size()


fn cosize(l: Layout) -> Int:
    return l.cosize()


alias LayoutList = List[Layout]


fn MakeLayoutList(v0: Layout, v1: Layout) -> LayoutList:
    var layout_list = LayoutList(capacity=2)
    layout_list.append(v0)
    layout_list.append(v1)
    return layout_list


fn MakeTileLayoutList[*tile_sizes: Int]() -> LayoutList:
    @parameter
    fn num_tiles() -> Int:
        return __mlir_op.`pop.variadic.size`(tile_sizes)

    var layout_list = LayoutList(capacity=num_tiles())

    @parameter
    fn append_tile[i: Int]():
        alias arg = tile_sizes[i]
        layout_list.append(Layout(arg, 1))

    unroll[append_tile, num_tiles()]()
    return layout_list


# Layout coalesce -- flatten and combine as many modes as possible while preserving
# the int-to-int function.
# The CUTE version has a second input to specify which modes to coalesce. We simplify
# it to flag to indicate keeping the original rank.
fn coalesce(layout: Layout, keep_rank: Bool = False) -> Layout:
    if keep_rank:
        # Coalesce each mode and concat the results to perserve rank.
        var shapes = IntTuple()
        var strides = IntTuple()
        for mode in layout:
            var coalesced_mode = coalesce(mode, False)
            shapes.append(coalesced_mode.shape)
            strides.append(coalesced_mode.stride)

        return Layout(shapes, strides)

    var result_shape = IntTuple(1)
    var result_stride = IntTuple(0)

    for z in zip(flatten(layout.shape), flatten(layout.stride)):
        var shape = int(z[0])
        var stride = int(z[1])

        # skip their shape-1s
        if shape == 1:
            continue
        # replace our shape-1 with anything
        elif result_shape[-1] == 1:
            result_shape[-1] = shape
            result_stride[-1] = stride
        # merge modes if the shape*stride match
        elif int(result_shape[-1]) * int(result_stride[-1]) == stride:
            result_shape[-1] = int(result_shape[-1]) * shape
        # append a new mode
        else:
            result_shape.append(shape)
            result_stride.append(stride)

    if len(result_shape) == 1:
        return Layout(result_shape[0], result_stride[0])
    return Layout(result_shape, result_stride)


# Layout composition
fn composition(layoutA: Layout, layoutB: Layout) -> Layout:
    if len(layoutB) == 0:
        return layoutA

    if is_tuple(layoutB.shape):
        var r = Layout()
        for layoutB_i in layoutB:
            r.append(composition(layoutA, layoutB_i))
        return r

    if layoutB.stride == 0:
        return Layout(layoutB.shape, 0)
    else:
        var result_shape = IntTuple()
        var result_stride = IntTuple()
        var rest_shape = layoutB.shape
        var rest_stride = layoutB.stride

        for z in zip(flatten(layoutA.shape)[:-1], flatten(layoutA.stride)[:-1]):
            var s = int(z[0])
            var d = int(z[1])

            var s1 = shape_div(s, rest_stride)
            result_shape.append(min(s1, rest_shape))
            result_stride.append(mul(rest_stride, d))
            rest_shape = shape_div(rest_shape, abs(s1))
            rest_stride = shape_div(rest_stride, s)

        result_shape.append(rest_shape)
        result_stride.append(mul(rest_stride, int(flatten(layoutA.stride)[-1])))

        return coalesce(Layout(result_shape, result_stride))


# Tuple of layouts
fn composition(layout_a: Layout, tiler: LayoutList) -> Layout:
    var result = Layout()
    for t in tiler:
        result.append(composition(layout_a, t[]))
    return result


fn complement(layout: Layout, size: Int = 1) -> Layout:
    var result_shape = IntTuple()
    var result_stride = IntTuple()
    var current_idx = 1

    for z in sorted(zip(flatten(layout.stride), flatten(layout.shape))):
        var stride = int(z[0])
        var shape = int(z[1])

        if stride == 0 or shape == 1:
            continue

        var in_bound = current_idx <= shape * stride
        if not in_bound:
            abort("Complement out of bounds.")

        result_shape.append(stride // current_idx)
        result_stride.append(current_idx)
        current_idx = shape * stride

    result_shape.append((size + current_idx - 1) // current_idx)  # ceil_div
    result_stride.append(current_idx)

    return coalesce(Layout(result_shape, result_stride))


@always_inline
fn apply_tiler[
    func: fn (Layout, Layout) -> Layout
](layout_a: Layout, tiler: LayoutList) -> Layout:
    if len(tiler) == 0:
        return layout_a
    var result = Layout()
    for i in range(len(tiler)):
        result.append(func(layout_a[i], tiler[i]))
    return result


fn logical_divide(layout_a: Layout, _layout_b: Layout) -> Layout:
    return composition(
        layout_a, make_layout(_layout_b, complement(_layout_b, layout_a.size()))
    )


fn logical_divide(layout_a: Layout, tiler: LayoutList) -> Layout:
    return apply_tiler[logical_divide](layout_a, tiler)


fn logical_product(_layout_a: Layout, layout_b: Layout) -> Layout:
    return make_layout(
        _layout_a,
        composition(
            complement(_layout_a, _layout_a.size() * layout_b.cosize()),
            layout_b,
        ),
    )


fn logical_product(layout_a: Layout, tiler: LayoutList) -> Layout:
    if len(tiler) == 1:
        return logical_product(layout_a, tiler[0])
    return apply_tiler[logical_product](layout_a, tiler)


fn hier_unzip[
    splitter: fn (Layout, Layout) -> Layout
](layout_a: Layout, tiler: LayoutList) -> Layout:
    var split = Layout()
    for i in range(len(tiler)):
        split.append(hier_unzip[splitter](layout_a[i], tiler[i]))
    var res_1 = Layout()
    for i in range(len(tiler)):
        res_1.append(split[i][0])
    var res_2 = Layout()
    for i in range(len(tiler)):
        res_2.append(split[i][1])
    for i in range(len(tiler), len(layout_a)):
        res_2.append(layout_a[i])
    var res = Layout()
    res.append(res_1)
    res.append(res_2)
    return res


fn hier_unzip[
    splitter: fn (Layout, Layout) -> Layout
](layout_a: Layout, layout_b: Layout,) -> Layout:
    return splitter(layout_a, layout_b)


fn zipped_divide(layout_a: Layout, layout_b: Layout) -> Layout:
    return hier_unzip[logical_divide](layout_a, layout_b)


fn zipped_divide(layout_a: Layout, tiler: LayoutList) -> Layout:
    return hier_unzip[logical_divide](layout_a, tiler)


fn print_layout(layout: Layout):
    if layout.rank() != 2:
        abort("print_layout only supports 2D layouts")

    var idx_width = _calc_initial_buffer_size_int32(layout.cosize()) + 2
    var delim = "+-----------------------"
    print(layout)
    _printf("    ")
    for n in range(layout[1].size()):
        _printf("  %*d ", idx_width - 2, n)
    _printf("\n")

    for m in range(layout[0].size()):
        _printf("    ")
        for n in range(layout[1].size()):
            _printf("%.*s", idx_width + 1, delim)
        _printf("+\n")

        _printf("%2d  ", m)
        for n in range(layout[1].size()):
            _printf("| %*d ", idx_width - 2, int(layout(IntTuple(m, n))))
        _printf("|\n")
    _printf("    ")
    for n in range(layout[1].size()):
        _printf("%.*s", idx_width + 1, delim)
    _printf("+\n")
