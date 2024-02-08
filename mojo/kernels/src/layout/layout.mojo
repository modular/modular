# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .int_tuple import *

from builtin.io import _printf
from builtin.string import _calc_initial_buffer_size_int32


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
        return self.layout.shape.__len__() - self.index


struct Layout(Sized, Stringable, CollectionElement):
    var shape: IntTuple
    var stride: IntTuple

    fn __init__(inout self):
        self.shape = IntTuple()
        self.stride = IntTuple()

    fn __init__(inout self, shape: IntTuple, stride: IntTuple):
        self.shape = shape
        self.stride = stride

    @always_inline
    fn __moveinit__(inout self: Self, owned existing: Self):
        self.shape = existing.shape ^
        self.stride = existing.stride ^

    @always_inline
    fn __copyinit__(inout self, existing: Self):
        self.shape = existing.shape
        self.stride = existing.stride

    fn __str__(self) -> String:
        return (
            "Layout(" + self.shape.__str__() + ":" + self.stride.__str__() + ")"
        )

    @always_inline
    fn __len__(self) -> Int:
        return self.shape.__len__()

    @always_inline
    fn __iter__(self) -> _LayoutIter:
        return _LayoutIter(0, self)

    @always_inline
    fn size(self) -> Int:
        return product(self.shape)

    @always_inline
    fn cosize(self) raises -> Int:
        return math.max(1, inner_product(self.shape, self.stride))

    @always_inline
    fn __getitem__(self, index: Int) -> Self:
        return Layout(self.shape[index], self.stride[index])

    @always_inline
    fn rank(self) -> Int:
        return self.shape.value.get[IntTupleBase]().elts.size

    @always_inline
    fn __call__(self, idx: IntTuple) raises -> Int:
        return crd2idx(idx, self.shape, self.stride)

    @always_inline
    fn append(inout self, item: Layout):
        self.shape.append(item.shape)
        self.stride.append(item.stride)


fn coalesce(src: Layout) -> Layout:
    let flatten_shape = flatten(src.shape)
    let flatten_stride = flatten(src.stride)

    debug_assert(
        flatten_shape.__len__() == flatten_stride.__len__(),
        "flattened shapes and stride should be the same length",
    )

    var result_shape = IntTuple(1)
    var result_stride = IntTuple(0)

    for i in range(flatten_shape.__len__()):
        let shape = flatten_shape[i]
        let stride = flatten_stride[i]

        let shape_last_idx = result_shape.__len__() - 1
        let stride_last_idx = result_stride.__len__() - 1

        if int(shape) == 1:
            continue
        elif int(result_shape[shape_last_idx]) == 1:
            result_shape[shape_last_idx] = shape
            result_stride[stride_last_idx] = stride
        elif int(result_shape[shape_last_idx]) * int(
            result_shape[stride_last_idx]
        ) == int(stride):
            result_shape[shape_last_idx] = int(
                result_shape[shape_last_idx]
            ) * int(stride)
        else:
            result_shape.append(shape)
            result_stride.append(stride)

    if len(result_shape) == 1:
        return Layout(result_shape[0], result_stride[0])
    else:
        return Layout(result_shape, result_stride)


fn composition(layout_a: Layout, layout_b: Layout) raises -> Layout:
    # Note: len(flatten(layout_b.shape)) > 1 is needed because we cant
    # get 1:2 layout by default everything is an int!?
    if is_tuple(layout_b.shape) and len(flatten(layout_b.shape)) > 1:
        var res_layout_shape = IntTuple()
        var res_layout_stride = IntTuple()
        for layout_b_i in layout_b:
            let a_b_i = composition(layout_a, layout_b_i)
            res_layout_shape.append(a_b_i.shape)
            res_layout_stride.append(a_b_i.stride)
        return Layout(res_layout_shape, res_layout_stride)

    if int(layout_b.stride) == 0:
        return Layout(layout_b.shape, 0)
    else:
        var result_shape = IntTuple()
        var result_stride = IntTuple()

        var rest_shape = layout_b.shape
        var rest_stride = layout_b.stride
        let flatten_a_shapes = flatten(layout_a.shape)
        let flatten_a_strides = flatten(layout_a.stride)

        for i in range(len(flatten_a_shapes) - 1):
            let s = int(flatten_a_shapes[i])
            let d = int(flatten_a_strides[i])
            let s1 = shape_div(s, rest_stride)
            result_shape.append(elementwise_min(s1, rest_shape))
            result_stride.append(rest_stride * d)
            rest_shape = shape_div(rest_shape, s1)
            rest_stride = shape_div(rest_stride, s)

        result_shape.append(rest_shape)
        result_stride.append(
            rest_stride * int(flatten_a_strides[len(flatten_a_strides) - 1])
        )

        return coalesce(Layout(result_shape, result_stride))


fn composition(layout_a: Layout, tiler: DynamicVector[Layout]) raises -> Layout:
    var res_shape = IntTuple()
    var res_stride = IntTuple()
    for i in range(len(tiler)):
        let res = composition(layout_a, tiler[i])
        res_shape.append(res.shape)
        res_stride.append(res.stride)
    return Layout(res_shape, res_stride)


fn complement(layout: Layout, size: Int = 1) -> Layout:
    var result_shape = IntTuple()
    var result_stride = IntTuple()
    var current_idx = 1

    var flat_shape = flatten(layout.shape)
    var flat_stride = flatten(layout.stride)

    # Ugly O(n^2) sored idx, should remove later..
    # use max_int
    for i in range(len(flat_stride)):
        var min_val = 4294967295
        var min_idx = -1
        for j in range(i, len(flat_stride)):
            if int(flat_stride[j]) < min_val:
                min_val = int(flat_stride[j])
                min_idx = j
        let tmp_stride = flat_stride[i]
        let tmp_shape = flat_shape[i]
        flat_stride[i] = flat_stride[min_idx]
        flat_shape[i] = flat_shape[min_idx]
        flat_shape[min_idx] = tmp_shape
        flat_stride[min_idx] = tmp_stride

    for i in range(len(flat_stride)):
        let stride_i = int(flat_stride[i])
        let shape_i = int(flat_shape[i])

        if stride_i == 0 or shape_i == 1:
            continue

        result_shape.append(stride_i // current_idx)
        result_stride.append(current_idx)
        current_idx = shape_i * stride_i

    result_shape.append((size + current_idx - 1) // current_idx)  # ceil_div ?
    result_stride.append(current_idx)

    return coalesce(Layout(result_shape, result_stride))


fn apply_tiler(
    func: fn (Layout, Layout) raises -> Layout,
    layout_a: Layout,
    tiler: DynamicVector[Layout],
) raises -> Layout:
    if tiler.size == 0:
        return layout_a
    var res = Layout()
    for i in range(len(tiler)):
        let layout_b = tiler[i]
        res.append(func(layout_a[i], layout_b))
    return res


fn logical_divide(layout_a: Layout, layout_b: Layout) raises -> Layout:
    let res_comp = complement(layout_b, layout_a.size())
    var res = layout_b
    res.shape.append(res_comp.shape)
    res.stride.append(res_comp.stride)
    return composition(layout_a, res)


fn logical_divide(
    layout_a: Layout, tiler: DynamicVector[Layout]
) raises -> Layout:
    return apply_tiler(logical_divide, layout_a, tiler)


fn logical_product(layout_a: Layout, layout_b: Layout) raises -> Layout:
    let a_comp = complement(layout_a, layout_a.size() * layout_b.cosize())
    var com_res = composition(a_comp, layout_b)
    var res = layout_a
    res.shape.append(com_res.shape)
    res.stride.append(com_res.stride)
    return res


fn logical_product(
    layout_a: Layout, tiler: DynamicVector[Layout]
) raises -> Layout:
    if tiler.size == 1:
        return logical_product(layout_a, tiler[0])
    return apply_tiler(logical_product, layout_a, tiler)


fn hier_unzip(
    splitter: fn (Layout, Layout) raises -> Layout,
    layout_a: Layout,
    tiler: DynamicVector[Layout],
) raises -> Layout:
    var split = Layout()
    for i in range(len(tiler)):
        split.append(hier_unzip(splitter, layout_a[i], tiler[i]))
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


fn hier_unzip(
    splitter: fn (Layout, Layout) raises -> Layout,
    layout_a: Layout,
    layout_b: Layout,
) raises -> Layout:
    return splitter(layout_a, layout_b)


fn zipped_divide(layout_a: Layout, layout_b: Layout) raises -> Layout:
    return hier_unzip(logical_divide, layout_a, layout_b)


fn zipped_divide(
    layout_a: Layout, tiler: DynamicVector[Layout]
) raises -> Layout:
    return hier_unzip(logical_divide, layout_a, tiler)


fn print_layout(layout: Layout) raises:
    if layout.rank() != 2:
        raise Error("print_layout only supports 2D layouts")

    let idx_width = _calc_initial_buffer_size_int32(layout.cosize()) + 2
    let delim = "+-----------------------"
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
