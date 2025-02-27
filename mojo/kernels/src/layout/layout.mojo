# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import sys
from collections import InlineArray, Optional
from collections.string.string import _calc_initial_buffer_size_int32
from os import abort

from buffer.dimlist import DimList

from utils import Writable, Writer

from .int_tuple import (
    UNKNOWN_VALUE,
    INT_TUPLE_VALIDATION,
    IntTuple,
    IntArray,
    abs,
    crd2idx,
    flatten,
    idx2crd,
    inner_product,
    is_int,
    is_tuple,
    mul,
    prefix_product,
    product,
    propagate_unknown,
    reverse,
    shape_div,
    sorted,
    to_unknown,
    tuple_min,
    zip,
    compact_order,
    product_each,
)

# ===-----------------------------------------------------------------------===#
# Layout Trait                                                                 #
# ===-----------------------------------------------------------------------===#


trait LayoutTrait(Copyable):
    """The LayoutTrait trait destribles layouts, swizzles, and composed layouts.

    They are used to map indices, e.g. from thread id to memory location.

    """

    fn __call__(self, index: IntTuple) -> Int:
        """Get the output index."""
        ...

    fn size(self) -> Int:
        """Return the size defined by shape.

        E.g. shape (m, n) has size m * n.
        """
        ...

    fn cosize(self) -> Int:
        """Return the size of the domain spanned by output indices.

        E.g. shape (m, n) and stride (r, s) has cosize (m - 1) * r + (n -1) * s.
        """
        ...

    @staticmethod
    fn has_shape() -> Bool:
        """Return whether the object has valid shape.

        Layout and ComposedLayout with at least one Layout have valid shapes.
        They can be used in layout algebra. Swizzle doesn't have shape and
        should be excluded from layout algebra.
        """
        ...


# ===-----------------------------------------------------------------------===#
# Layout                                                                       #
# ===-----------------------------------------------------------------------===#


fn make_layout(*layouts: Layout) -> Layout:
    var shape = IntTuple()
    var stride = IntTuple()
    for i in range(len(layouts)):
        shape.append(layouts[i].shape)
        stride.append(layouts[i].stride)
    return Layout(shape, stride)


# Workaround MOCO-976.
fn make_layout(layout_a: Layout, layout_b: Layout) -> Layout:
    var shape = IntTuple()
    var stride = IntTuple()
    shape.append(layout_a.shape)
    shape.append(layout_b.shape)
    stride.append(layout_a.stride)
    stride.append(layout_b.stride)
    return Layout(shape, stride)


# Make a compact/bijective layout with `shape` and `stride` following the order
# induced by `order`.
# For example:
# `make_ordered_layout(IntTuple(2, 3, 4, 5),  IntTuple(1, 4, 3, 5)) == (2, 3, 4,
# 5):(1, 8, 2, 24)`
# `make_ordered_layout(IntTuple(2, IntTuple(3, 4), 5),  IntTuple(1, IntTuple(2,
# 3), 4)) == ((2, (3, 4)), 5):(1, (2, 6), 24)`
fn make_ordered_layout(shape: IntTuple, order: IntTuple) -> Layout:
    var stride = compact_order(shape, order)
    return Layout(shape, stride)


@value
struct _LayoutIter[origin: ImmutableOrigin]:
    var index: Int
    var layout: Pointer[Layout, origin]

    fn __next__(mut self) -> Layout:
        var idx = self.index
        self.index += 1
        return Layout(
            self.layout[].shape[idx],
            self.layout[].stride[idx],
        )

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        return len(self.layout[].shape) - self.index


struct Layout(
    LayoutTrait,
    Sized,
    Stringable,
    Writable,
    CollectionElement,
    EqualityComparable,
):
    var shape: IntTuple
    var stride: IntTuple

    # ===------------------------------------------------------------------===#
    # Initializers
    # ===------------------------------------------------------------------===#

    @always_inline
    fn __init__(out self):
        self.shape = IntTuple()
        self.stride = IntTuple()

    @always_inline
    @implicit
    fn __init__(out self, shape: IntTuple):
        # FIXME: all these owned_copy() calls are annoying, fix __copyinit__
        self.shape = shape.owned_copy()
        self.stride = prefix_product(self.shape)

    @always_inline
    fn __init__(out self, shape: IntTuple, stride: IntTuple):
        self.shape = shape.owned_copy()
        if len(stride) == 0:
            self.stride = prefix_product(self.shape)
        else:
            self.stride = stride.owned_copy()

    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    @always_inline
    fn idx2crd(self, idx: IntTuple) -> IntTuple:
        return idx2crd(idx, self.shape, self.stride)

    @staticmethod
    fn col_major(*dims: Int) -> Layout:
        var shape = IntTuple()
        for dim in dims:
            shape.append(dim)
        return Self.col_major(shape)

    @staticmethod
    fn col_major(shape: IntTuple) -> Layout:
        return Layout(shape, prefix_product(shape))

    @staticmethod
    fn row_major(*dims: Int) -> Layout:
        var shape = IntTuple()
        for dim in dims:
            shape.append(dim)
        return Self.row_major(shape)

    @staticmethod
    fn row_major[rank: Int](dims: DimList) -> Layout:
        var shape = IntTuple()
        var stride = IntTuple()
        var unknown_flag = False
        var c_stride = 1
        stride.append(c_stride)

        @parameter
        for i in range(rank):
            var dim = dims.get[i]()
            shape.append(dim if dims.has_value[i]() else UNKNOWN_VALUE)

        @parameter
        for i in range(rank - 1):
            var dim = dims.get[rank - 1 - i]()
            if not dims.has_value[rank - 1 - i]():
                unknown_flag = True
            stride.append(UNKNOWN_VALUE if unknown_flag else dim * c_stride)
            c_stride *= dim

        return Layout(shape, reverse(stride))

    @staticmethod
    fn row_major(shape: IntTuple) -> Layout:
        return Layout(shape, reverse(prefix_product(reverse(shape))))

    @always_inline
    fn make_shape_unknown[axis: Int = UNKNOWN_VALUE](self) -> Layout:
        """Mark shape (along axis) unknown.
        This is useful for tiling a tensor with runtime sizses where the tile's
        shape is unknown but the stride is preserved.
        """

        @parameter
        if axis == UNKNOWN_VALUE:
            return Layout(to_unknown(self.shape), self.stride)
        else:
            # var shape_with_unknown = self.shape
            # shape_with_unknown[axis] = to_unknown(self.shape[axis])
            var shape_with_unknown = IntTuple()
            for i in range(len(self.shape)):
                if i != axis:
                    shape_with_unknown.append(self.shape[i])
                else:
                    shape_with_unknown.append(to_unknown(self.shape[i]))
            return Layout(shape_with_unknown, self.stride)

    @always_inline
    fn __moveinit__(out self, owned existing: Self):
        self.shape = existing.shape^
        self.stride = existing.stride^

    @always_inline
    fn __copyinit__(out self, existing: Self):
        self.shape = existing.shape
        self.stride = existing.stride

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        return self

    # ===------------------------------------------------------------------===#
    # Methods
    # ===------------------------------------------------------------------===#

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.shape, ":", self.stride, ")")

    fn __eq__(self, other: Layout) -> Bool:
        return self.shape == other.shape and self.stride == other.stride

    fn __ne__(self, other: Layout) -> Bool:
        return not (self == other)

    @always_inline
    fn __len__(self) -> Int:
        return len(self.shape)

    @always_inline
    fn __iter__(self) -> _LayoutIter[__origin_of(self)]:
        return _LayoutIter(0, Pointer.address_of(self))

    @always_inline
    fn size(self) -> Int:
        return product(self.shape)

    @always_inline
    fn cosize(self) -> Int:
        return self(self.size() - 1) + 1
        # return math.max(1, inner_product(self.shape, self.stride))

    @staticmethod
    @always_inline
    fn has_shape() -> Bool:
        return True

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
    fn append(mut self, item: Layout):
        self.shape.append(item.shape)
        self.stride.append(item.stride)

    # Returns `True` if values in shape and stride are known, otherwise `False`.
    #
    @always_inline
    fn all_dims_known(self) -> Bool:
        for shape_i in flatten(self.shape):
            if Int(shape_i) == UNKNOWN_VALUE:
                return False
        for stride_i in flatten(self.stride):
            if Int(stride_i) == UNKNOWN_VALUE:
                return False
        return True

    # Returns `True` if values in shape are known, otherwise `False`.
    #
    @always_inline
    fn known_shape(self) -> Bool:
        for shape_i in flatten(self.shape):
            if Int(shape_i) == UNKNOWN_VALUE:
                return False
        return True


fn size(l: Layout) -> Int:
    return l.size()


fn cosize(l: Layout) -> Int:
    return l.cosize()


alias LayoutList = List[Layout]


fn MakeLayoutList(v0: Layout, v1: Layout) -> LayoutList:
    return LayoutList(v0, v1)


fn MakeTileLayoutList[*tile_sizes: Int]() -> LayoutList:
    @parameter
    fn num_tiles() -> Int:
        return __mlir_op.`pop.variadic.size`(tile_sizes)

    var layout_list = LayoutList(capacity=num_tiles())

    @parameter
    for i in range(num_tiles()):
        alias arg = tile_sizes[i]
        layout_list.append(Layout(arg, 1))

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
        var shape = Int(z[0])
        var stride = Int(z[1])

        # skip their shape-1s
        if shape == 1:
            continue
        # replace our shape-1 with anything
        elif result_shape[-1] == 1:
            # result_shape[-1] = shape
            result_shape.replace_entry(len(result_shape) - 1, int_value=shape)
            # result_stride[-1] = stride
            result_stride.replace_entry(
                len(result_stride) - 1, int_value=stride
            )

        # merge modes if the shape*stride match and computable.
        elif Int(result_shape[-1]) * Int(
            result_stride[-1]
        ) == stride and UNKNOWN_VALUE not in (
            shape,
            stride,
            Int(result_shape[-1]),
            Int(result_stride[-1]),
        ):
            # result_shape[-1] = to_int(result_shape[-1]) * shape
            result_shape.replace_entry(
                len(result_shape) - 1,
                int_value=Int(result_shape[-1]) * shape,
            )

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
            var s = Int(z[0])
            var d = Int(z[1])

            var s1 = shape_div(s, rest_stride)
            result_shape.append(tuple_min(s1, rest_shape))
            result_stride.append(mul(rest_stride, d))
            rest_shape = shape_div(rest_shape, abs(s1))
            rest_stride = shape_div(rest_stride, s)

        result_shape.append(rest_shape)
        result_stride.append(mul(rest_stride, Int(flatten(layoutA.stride)[-1])))

        return coalesce(Layout(result_shape, result_stride))


# Tuple of layouts
fn composition(layout_a: Layout, tiler: LayoutList) -> Layout:
    var result = Layout()
    for i in range(len(tiler)):
        result.append(composition(layout_a[i], tiler[i]))

    # Remainder if tiler is shorter.
    for i in range(len(tiler), len(layout_a)):
        result.append(layout_a[i])

    return result


fn complement(layout: Layout, size: Int = 1) -> Layout:
    var current_idx = 1
    var sorted = sorted(zip(flatten(layout.stride), flatten(layout.shape)))
    var sorted_len = len(sorted)

    var result_shape = IntTuple(num_elems=sorted_len + 1, size=sorted_len + 2)
    var result_stride = IntTuple(num_elems=sorted_len + 1, size=sorted_len + 2)

    # for z in sorted(zip(flatten(layout.stride), flatten(layout.shape))):
    var i = 0
    for z in sorted:
        var stride = Int(z[0])
        var shape = Int(z[1])

        if UNKNOWN_VALUE in (shape, stride) or stride == 0 or shape == 1:
            continue

        var in_bound = current_idx <= shape * stride
        if not in_bound:
            abort("Complement out of bounds.")

        result_shape.replace_entry(i, int_value=stride // current_idx)
        result_stride.replace_entry(i, int_value=current_idx)
        i += 1
        current_idx = shape * stride

    result_shape.replace_entry(
        i, int_value=(size + current_idx - 1) // current_idx
    )  # ceil_div
    result_stride.replace_entry(i, int_value=current_idx)
    i += 1

    result_shape._store[0] = i
    result_stride._store[0] = i

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


fn zip_modes(layout_a: Layout, layout_b: Layout) -> Layout:
    var zipped = Layout()
    for i in range(layout_a.rank()):
        zipped.append(make_layout(layout_a[i], layout_b[i]))
    return zipped


fn blocked_product(layout_a: Layout, layout_b: Layout) -> Layout:
    # ((a_0, a_1, ...), (tile_0, tile_1, ...))
    var lp = logical_product(layout_a, layout_b)
    # ((a_0, tile_0), (a_1, tile_1), ...)
    return zip_modes(lp[0], lp[1])


fn tile_to_shape(
    tile: Layout, target_shape: IntTuple, order: Optional[IntTuple] = None
) -> Layout:
    var flat_tile_shape = product_each(tile.shape)
    var flat_target_shape = product_each(target_shape)
    var tiler_shape = IntTuple()
    for i in range(len(flat_tile_shape)):
        var a = Int(flat_target_shape[i])
        var b = Int(flat_tile_shape[i])
        if a % b != 0:
            abort(
                "Tile to shape failed: target shape is not divisible by tile"
                " shape"
            )
        tiler_shape.append(a // b)

    var new_order: IntTuple
    if order:
        new_order = order.value()
    else:
        new_order = prefix_product(tiler_shape)  # default to column major
    var tiler = make_ordered_layout(tiler_shape, new_order)
    return blocked_product(tile, tiler)


fn logical_product(layout_a: Layout, tiler: LayoutList) -> Layout:
    if len(tiler) == 1:
        return logical_product(layout_a, tiler[0])
    return apply_tiler[logical_product](layout_a, tiler)


fn hier_unzip(layout_a: Layout, tiler: LayoutList) -> Layout:
    var res_1 = Layout()
    var res_2 = Layout()

    for i in range(len(tiler)):
        var split = hier_unzip(layout_a[i], tiler[i])
        res_1.append(split[0])
        res_2.append(split[1])

    # Remainder if tiler is shorter.
    for i in range(len(tiler), len(layout_a)):
        res_2.append(layout_a[i])

    var res = Layout()
    res.shape.append(res_1.shape, res_2.shape)
    res.stride.append(res_1.stride, res_2.stride)
    return res


fn hier_unzip(
    layout_a: Layout,
    layout_b: Layout,
) -> Layout:
    return logical_divide(layout_a, layout_b)


fn zipped_divide(layout_a: Layout, layout_b: Layout) -> Layout:
    return hier_unzip(layout_a, layout_b)


fn zipped_divide(layout_a: Layout, tiler: LayoutList) -> Layout:
    return hier_unzip(layout_a, tiler)


fn print_layout(layout: Layout):
    if layout.rank() != 2:
        abort("print_layout only supports 2D layouts")

    print(layout)
    # make stdout mutable
    var stdout = sys.stdout
    format_layout(layout, stdout)


fn format_layout[W: Writer](layout: Layout, mut writer: W):
    @parameter
    fn _write_divider(column_count: Int, cell_width: Int):
        for _ in range(column_count):
            writer.write("+")
            for _ in range(cell_width):
                writer.write("-")
        writer.write("+\n")

    var idx_width = _calc_initial_buffer_size_int32(layout.cosize()) + 2

    # Print column labels
    writer.write("    ")
    for n in range(layout[1].size()):
        writer.write("  ")
        n.write_padded(writer, width=idx_width - 2)

        var is_last_column = n + 1 == layout[1].size()

        if not is_last_column:
            writer.write(" ")

    writer.write("\n")

    for m in range(layout[0].size()):
        writer.write("    ")
        _write_divider(layout[1].size(), idx_width)

        # Print row label
        m.write_padded(writer, width=2)
        writer.write("  ")

        for n in range(layout[1].size()):
            writer.write("| ")
            Int(layout(IntTuple(m, n))).write_padded(
                writer,
                width=idx_width - 2,
            )
            writer.write(" ")
        writer.write("|\n")

    writer.write("    ")

    # Write the final horizontal dividing line
    _write_divider(layout[1].size(), idx_width)


# Returns the sublayout with specific modes e.g sublayout(Layout(3,4,5), 0, 2)
# returns Layout(3, 5)
#
fn sublayout(layout: Layout, *modes: Int) -> Layout:
    var shape = IntTuple()
    var stride = IntTuple()
    for mode in modes:
        shape.append(layout.shape[mode])
        stride.append(layout.stride[mode])
    return Layout(shape, stride)


fn expand_strides(shape: IntTuple, stride: Int) -> IntTuple:
    var new_stride = IntTuple()
    var cumulative_stride: Int = stride
    for sz in shape:
        if sz.is_tuple():
            new_stride.append(expand_strides(sz, cumulative_stride))
            cumulative_stride *= size(sz)
        else:
            new_stride.append(cumulative_stride)
            cumulative_stride *= sz.value()
    return new_stride


fn expand_modes_alike(
    shape_a: IntTuple, stride_a: IntTuple, shape_b: IntTuple, stride_b: IntTuple
) -> InlineArray[IntTuple, 3]:
    if shape_a.is_tuple() and shape_b.is_tuple():
        var new_shape = IntTuple()
        var new_stride_a = IntTuple()
        var new_stride_b = IntTuple()
        # TODO: lift this limitation, and instead require
        # that size(shape_a) != size(shape_b):
        # Ideally, we'd be able to call `coalesce` on two layouts prior
        # to `uncoalesce` in functions like `copy_to` to first simplify.
        if len(shape_a) != len(shape_b):
            abort()
        for i in range(len(shape_a)):
            if shape_a[i].is_value() and shape_b[i].is_value():
                new_shape.append(shape_a[i].value())
                new_stride_a.append(stride_a[i].value())
                new_stride_b.append(stride_b[i].value())
            elif shape_a[i].is_value():
                new_shape.append(shape_b[i])
                new_stride_b.append(stride_b[i])
                new_stride_a.append(
                    expand_strides(shape_b[i], stride_a[i].value())
                )
            elif shape_b[i].is_value():
                new_shape.append(shape_a[i])
                new_stride_a.append(stride_a[i])
                new_stride_b.append(
                    expand_strides(shape_a[i], stride_b[i].value())
                )
            else:
                var uc = expand_modes_alike(
                    shape_a[i], stride_a[i], shape_b[i], stride_b[i]
                )
                new_shape.append(uc[0])
                new_stride_a.append(uc[1])
                new_stride_b.append(uc[2])

        return InlineArray[IntTuple, 3](new_shape, new_stride_a, new_stride_b)
    elif shape_a.is_tuple():
        return InlineArray[IntTuple, 3](
            shape_a.owned_copy(),
            stride_a.owned_copy(),
            expand_strides(shape_a, stride_b.value()),
        )
    elif shape_b.is_tuple():
        return InlineArray[IntTuple, 3](
            shape_b.owned_copy(),
            expand_strides(shape_b.owned_copy(), stride_a.value()),
            stride_b.owned_copy(),
        )
    else:
        return InlineArray[IntTuple, 3](
            shape_b.owned_copy(), stride_a.owned_copy(), stride_b.owned_copy()
        )


fn expand_modes_alike(
    layout_a: Layout, layout_b: Layout
) -> InlineArray[Layout, 2]:
    """
    Tiles both layouts such that they mirror one another.

    For example:

        ```mojo
        alias layout_0 = Layout(
             IntTuple(IntTuple(3, IntTuple(5, 2)), 4),
             IntTuple(IntTuple(1, IntTuple(24, 12)), 3),
         )
         alias layout_1 = Layout(
             IntTuple(30, IntTuple(2, 2)), IntTuple(2, IntTuple(60, 1))
         )
         alias uc = uncoalesce(layout_0, layout_1)
         print(uc[0])
         # (((3, (5, 2)), (2, 2)):((1, (24, 12)), (3, 6)))
         print(uc[1])
         # (((3, (5, 2)), (2, 2)):((2, (6, 30)), (60, 1)))
         ```

    The shapes are equal. This can be useful for another algorithm
    that wants to iterate over layouts in a blockwise fasion.
    """
    var uc = expand_modes_alike(
        layout_a.shape, layout_a.stride, layout_b.shape, layout_b.stride
    )
    return InlineArray[Layout, 2](Layout(uc[0], uc[1]), Layout(uc[0], uc[2]))


fn right_inverse(layout: Layout) -> Layout:
    var flat_layout = coalesce(layout)
    var rstride = prefix_product(flat_layout.shape)
    var shape = IntTuple()
    var stride = IntTuple()
    var next_stride = 1
    for _ in range(flat_layout.rank()):
        for j in range(flat_layout.rank()):
            if Int(flat_layout.stride[j]) == next_stride:
                var shape_j = flat_layout.shape[j]
                shape.append(shape_j)
                stride.append(rstride[j])
                next_stride *= Int(shape_j)
                break
    return Layout(shape, stride)


fn is_row_major[rank: Int](layout: Layout) -> Bool:
    var flat_shape = flatten(layout.shape)
    var flat_stride = flatten(layout.stride)
    var flat_rank = len(flat_shape)

    if flat_rank != rank:
        return False

    if flat_stride[flat_rank - 1].value() != 1:
        return False

    correct_stride = flat_shape[flat_rank - 1].value()

    for i in reversed(range(flat_rank - 1)):
        stride_i = flat_stride[i].value()
        if stride_i != correct_stride:
            return False
        correct_stride *= flat_shape[i].value()

    return True
