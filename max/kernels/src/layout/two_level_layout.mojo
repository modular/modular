# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Provides a two-level `TensorLayout` conformer.

A `TwoLevelLayout` splits a layout into an integer outer part and an F2-linear
inner part. The inner part is the map a hardware swizzle applies inside one
atom's footprint; the outer part carries the footprint bases with ordinary
integer strides. A coordinate's contribution is

```
Outer(o) + (x0 ^ Lambda(y))
```

where `x0` is a `w`-bit XOR base. `x0` is zero for a whole tensor and moves
when a tile starts inside a footprint.

The inner map is given as comptime basis images, one image per inner index
bit. The low bits of each mode are its inner bits, packed mode by mode from
the low bit up. `Lambda` is injective, so `TwoLevelLayout` is not affine and
`is_affine` is `False`: code that steps through memory by adding a stride
must not do so for this layout.

Use [`two_level_layout_from_swizzle`](/api/mojo/layout/two_level_layout/two_level_layout_from_swizzle)
to turn a `Swizzle` and an integer `Layout` into an exact `TwoLevelLayout`.

You can import these APIs from the `layout` package:

```mojo
from layout.two_level_layout import TwoLevelLayout
```
"""

from std.bit import log2_floor
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder

from .coord import (
    ComptimeInt,
    Coord,
    CoordLike,
    DynamicCoord,
    _IntToComptimeInt,
    _NestedDynamicCoord,
)
from .swizzle import Swizzle
from .tile_layout import Layout, TensorLayout


# ===----------------------------------------------------------------------=== #
# Inner-map type-level helpers
# ===----------------------------------------------------------------------=== #


comptime _InnerBitCount[
    Inner: TypeList[Trait=CoordLike, ...], i: Int
]: Int = Inner[i].__len__()
"""The number of inner index bits the layout takes from mode `i`."""


comptime _BitCountReducer[
    upto: Int, Prev: Int, Element: CoordLike, list_idx: Int
]: Int = Prev + Element.__len__() if list_idx < upto else Prev


comptime _InnerBitOffset[
    Inner: TypeList[Trait=CoordLike, ...], upto: Int
]: Int = Inner.reduce_idx[0, _BitCountReducer[upto, ...]]
"""The packed bit position of mode `upto`'s lowest inner index bit."""


comptime _InnerTotalBits[Inner: TypeList[Trait=CoordLike, ...]]: Int = (
    _InnerBitOffset[Inner, Inner.length]
)
"""The total number of inner index bits."""


comptime _InnerImage[
    Inner: TypeList[Trait=CoordLike, ...], mode: Int, j: Int
]: Int = Inner[mode].ParamListType[j].static_value
"""The basis image of inner index bit `j` of mode `mode`."""


comptime _FirstImageReducer[Prev: Int, Element: CoordLike]: Int = (
    Prev if Prev != -1 else Element.static_value
)


comptime _LowestImage[
    Inner: TypeList[Trait=CoordLike, ...], i: Int
]: Int = Inner[i].ParamListType.reduce[-1, _FirstImageReducer]
"""The image of mode `i`'s lowest inner bit, or `-1` when mode `i` is outer."""


comptime _ImageOrReducer[Prev: Int, Element: CoordLike]: Int = (
    Prev | Element.static_value
)


comptime _ModeMask[Mode: CoordLike]: Int = Mode.ParamListType.reduce[
    0, _ImageOrReducer
]


comptime _MaskReducer[Prev: Int, Element: CoordLike, list_idx: Int]: Int = (
    Prev | _ModeMask[Element]
)


comptime _InnerMask[Inner: TypeList[Trait=CoordLike, ...]]: Int = (
    Inner.reduce_idx[0, _MaskReducer]
)
"""The union of the low offset bits the inner map reaches."""


def _bit_width(value: Int) -> Int:
    """Returns the number of bits needed to hold `value`."""
    var n = value
    var width = 0
    while n > 0:
        width += 1
        n >>= 1
    return width


comptime _InnerMaskWidth[Inner: TypeList[Trait=CoordLike, ...]]: Int = (
    _bit_width(_InnerMask[Inner])
)
"""The width `w` of the low offset bits the inner map reaches."""


def _log2_or_zero(value: Int) -> Int:
    """Returns `log2_floor(value)`, or `0` when `value` is not positive."""
    return log2_floor(value) if value > 0 else 0


comptime _PowerOfTwo[value: Int]: Bool = (
    value > 0 and (value & (value - 1)) == 0
)

comptime _IntMin[a: Int, b: Int]: Int = a if a < b else b


comptime _IntMax[a: Int, b: Int]: Int = a if a > b else b


comptime _BitOwnerReducer[
    Inner: TypeList[Trait=CoordLike, ...],
    b: Int,
    Prev: Int,
    Element: CoordLike,
    list_idx: Int,
]: Int = (
    list_idx if Prev == -1
    and _InnerBitOffset[Inner, list_idx] <= b
    and b < _InnerBitOffset[Inner, list_idx] + Element.__len__() else Prev
)


comptime _BitOwner[
    Inner: TypeList[Trait=CoordLike, ...], b: Int
]: Int = Inner.reduce_idx[-1, _BitOwnerReducer[Inner, b, ...]]
"""The mode that owns packed inner index bit `b`."""


comptime _FlatImage[
    Inner: TypeList[Trait=CoordLike, ...], b: Int
]: Int = _InnerImage[
    Inner,
    _BitOwner[Inner, b],
    b - _InnerBitOffset[Inner, _BitOwner[Inner, b]],
]
"""The basis image of packed inner index bit `b`."""


def _invert_inner_image[Inner: TypeList[Trait=CoordLike, ...], p: Int]() -> Int:
    """Returns the inner index whose image is offset bit `p`.

    Parameters:
        Inner: The inner map's basis image lists, one list per mode.
        p: The offset bit to invert.

    Returns:
        The inner index `y` with `Lambda(y) == 1 << p`, or `-1` when `p` is
        not in the image.
    """
    comptime K = _InnerTotalBits[Inner]
    comptime W = _InnerMaskWidth[Inner]
    var cols = Array[Int, K]()
    var combos = Array[Int, K]()
    var pivoted = Array[Bool, K](fill=False)
    var pivot_col = Array[Int, W](fill=-1)
    comptime for b in range(K):
        cols[b] = _FlatImage[Inner, b]
        combos[b] = 1 << b
    # Forward elimination: give every reachable offset bit one pivot column.
    for bit in range(W):
        var pick = -1
        for c in range(K):
            if not pivoted[c] and ((cols[c] >> bit) & 1) == 1:
                pick = c
                break
        if pick == -1:
            continue
        for c in range(K):
            if c != pick and ((cols[c] >> bit) & 1) == 1:
                cols[c] ^= cols[pick]
                combos[c] ^= combos[pick]
        pivoted[pick] = True
        pivot_col[bit] = pick
    # Back substitution: clear the higher bits a pivot column still carries.
    for step in range(W):
        var bit = W - 1 - step
        var pivot = pivot_col[bit]
        if pivot == -1:
            continue
        for lower in range(bit):
            var other = pivot_col[lower]
            if other != -1 and ((cols[other] >> bit) & 1) == 1:
                cols[other] ^= cols[pivot]
                combos[other] ^= combos[pivot]
    if p >= W or pivot_col[p] == -1:
        return -1
    return combos[pivot_col[p]]


# ===----------------------------------------------------------------------=== #
# Full-shape helpers
# ===----------------------------------------------------------------------=== #


comptime _FullShapeTabulator[
    Outer: TensorLayout, Inner: TypeList[Trait=CoordLike, ...], i: Int
]: CoordLike = (
    ComptimeInt[
        Outer._shape_types[i].static_value << _InnerBitCount[Inner, i]
    ] if Outer._shape_types[i].is_static_value else Scalar[
        Outer._shape_types[i].DTYPE
    ]
)


comptime _FullShapeTypes[
    Outer: TensorLayout, Inner: TypeList[Trait=CoordLike, ...]
] = TypeList.tabulate[
    Trait=CoordLike, Outer.rank, _FullShapeTabulator[Outer, Inner, _]
]()
"""The full shapes: the outer shape scaled by each mode's inner bit count."""


comptime _FullStrideTabulator[
    Outer: TensorLayout, Inner: TypeList[Trait=CoordLike, ...], i: Int
]: CoordLike = (
    ComptimeInt[_LowestImage[Inner, i]] if _InnerBitCount[Inner, i]
    > 0 else Outer._stride_types[i]
)


comptime _FullStrideTypes[
    Outer: TensorLayout, Inner: TypeList[Trait=CoordLike, ...]
] = TypeList.tabulate[
    Trait=CoordLike, Outer.rank, _FullStrideTabulator[Outer, Inner, _]
]()
"""The effective stride per mode: the outer stride, or `Lambda` of the
mode's lowest inner bit."""


comptime _ReverseInnerTabulator[
    Inner: TypeList[Trait=CoordLike, ...], j: Int
]: CoordLike = Inner[Inner.length - 1 - j]


comptime _ReverseInner[
    Inner: TypeList[Trait=CoordLike, ...]
] = TypeList.tabulate[
    Trait=CoordLike, Inner.length, _ReverseInnerTabulator[Inner, _]
]()
"""The inner map with the modes in reverse order, for `transpose`."""


# ===----------------------------------------------------------------------=== #
# Nested tiles
# ===----------------------------------------------------------------------=== #


comptime _TileIsWholeFootprint[
    Inner: TypeList[Trait=CoordLike, ...], i: Int, size: Int
]: Bool = (
    _InnerBitCount[Inner, i] == 0 or size % (1 << _InnerBitCount[Inner, i]) == 0
)
"""Whether a tile of `size` elements on mode `i` holds whole footprints."""


comptime _TileIsInsideFootprint[
    Inner: TypeList[Trait=CoordLike, ...], i: Int, size: Int
]: Bool = (
    size > 0 and _PowerOfTwo[size] and size <= (1 << _InnerBitCount[Inner, i])
)
"""Whether a tile of `size` elements on mode `i` is an aligned
power-of-two box inside one footprint."""


comptime _TileModeAdmissible[
    Inner: TypeList[Trait=CoordLike, ...], i: Int, size: Int
]: Bool = (
    _TileIsWholeFootprint[Inner, i, size]
    or _TileIsInsideFootprint[Inner, i, size]
)


comptime _NestedOuterSize[
    Inner: TypeList[Trait=CoordLike, ...], i: Int, size: Int
]: Int = (
    size if _InnerBitCount[Inner, i]
    == 0 else (
        size
        >> _InnerBitCount[Inner, i] if size % (1 << _InnerBitCount[Inner, i])
        == 0 else 1
    )
)
"""The outer extent of a tile of `size` elements on mode `i`."""


comptime _NestedKeptBits[
    Inner: TypeList[Trait=CoordLike, ...], i: Int, size: Int
]: Int = (
    0 if _InnerBitCount[Inner, i]
    == 0 else (
        _InnerBitCount[Inner, i] if size % (1 << _InnerBitCount[Inner, i])
        == 0 else _IntMin[_log2_or_zero(size), _InnerBitCount[Inner, i]]
    )
)
"""How many of mode `i`'s low inner bits a tile of `size` elements keeps."""


comptime _NestedInnerEntry[
    Inner: TypeList[Trait=CoordLike, ...], i: Int, size: Int
] = Coord[*Inner[i].ParamListType.slice[0, _NestedKeptBits[Inner, i, size]]()]
"""The restricted inner map entry for mode `i` of a tile."""


comptime _NestedInnerTabulator[
    Outer: TensorLayout,
    Inner: TypeList[Trait=CoordLike, ...],
    TileShape: TypeList[Trait=CoordLike, ...],
    i: Int,
]: CoordLike = _NestedInnerEntry[Inner, i, TileShape[i].static_value]


comptime _NestedInnerTypes[
    Outer: TensorLayout,
    Inner: TypeList[Trait=CoordLike, ...],
    TileShape: TypeList[Trait=CoordLike, ...],
] = TypeList.tabulate[
    Trait=CoordLike,
    Outer.rank,
    _NestedInnerTabulator[Outer, Inner, TileShape, _],
]()


comptime _NestedOuterTabulator[
    Outer: TensorLayout,
    Inner: TypeList[Trait=CoordLike, ...],
    TileShape: TypeList[Trait=CoordLike, ...],
    i: Int,
]: CoordLike = ComptimeInt[
    _NestedOuterSize[Inner, i, TileShape[i].static_value]
]


comptime _NestedOuterTypes[
    Outer: TensorLayout,
    Inner: TypeList[Trait=CoordLike, ...],
    TileShape: TypeList[Trait=CoordLike, ...],
] = TypeList.tabulate[
    Trait=CoordLike,
    Outer.rank,
    _NestedOuterTabulator[Outer, Inner, TileShape, _],
]()


comptime _TileShapeKnownReducer[Prev: Bool, Element: CoordLike]: Bool = (
    Prev and Element.is_static_value
)


comptime _TileModeOkReducer[
    Inner: TypeList[Trait=CoordLike, ...],
    Prev: Bool,
    Element: CoordLike,
    list_idx: Int,
]: Bool = Prev and _TileModeAdmissible[Inner, list_idx, Element.static_value]


comptime _NestedTileAdmissible[
    Outer: TensorLayout,
    Inner: TypeList[Trait=CoordLike, ...],
    TileShape: TypeList[Trait=CoordLike, ...],
]: Bool = (
    TileShape.length == Outer.rank
    and TileShape.reduce[True, _TileShapeKnownReducer]
    and TileShape.reduce_idx[True, _TileModeOkReducer[Inner, ...]]
)


comptime _BadTileModeReducer[
    Inner: TypeList[Trait=CoordLike, ...],
    Prev: Int,
    Element: CoordLike,
    list_idx: Int,
]: Int = (
    Prev if Prev
    != -1 else (
        list_idx if not _TileModeAdmissible[
            Inner, list_idx, Element.static_value
        ] else Prev
    )
)


comptime _FirstBadTileMode[
    Inner: TypeList[Trait=CoordLike, ...],
    TileShape: TypeList[Trait=CoordLike, ...],
]: Int = TileShape.reduce_idx[-1, _BadTileModeReducer[Inner, ...]]


def _nested_box_check[
    Outer: TensorLayout,
    Inner: TypeList[Trait=CoordLike, ...],
    TileShape: TypeList[Trait=CoordLike, ...],
]() -> Bool:
    """Rejects a tile shape a `TwoLevelLayout` cannot express.

    A tile must hold whole footprints on every mode, or be an aligned
    power-of-two box inside one footprint. Anything else would need a
    different inner map per position and is not a static tiling.

    Returns:
        True when the tile shape is admissible.
    """
    comptime if not _NestedTileAdmissible[Outer, Inner, TileShape]:
        comptime bad = _FirstBadTileMode[Inner, TileShape]
        comptime assert False, String(
            "TwoLevelLayout cannot tile mode ",
            bad,
            " (footprint ",
            1 << _InnerBitCount[Inner, bad],
            ") with tile size ",
            TileShape[bad].static_value,
            (
                ": use a whole number of footprints or an aligned"
                " power-of-two box inside one footprint"
            ),
        )
    return True


# ===----------------------------------------------------------------------=== #
# The conformer
# ===----------------------------------------------------------------------=== #


struct _TwoLevelLayout[
    Outer: TensorLayout,
    FullShapes: TypeList[Trait=CoordLike, ...],
    FullStrides: TypeList[Trait=CoordLike, ...],
    Inner: TypeList[Trait=CoordLike, ...],
](
    DevicePassable,
    ImplicitlyCopyable,
    TensorLayout,
    TrivialRegisterPassable,
    Writable,
):
    """The two-level layout struct behind `TwoLevelLayout`.

    The full shape and stride type lists are parameters so the trait members
    can index them. See `TwoLevelLayout` for the public form.

    Parameters:
        Outer: The integer outer layout.
        FullShapes: The full shape types, one per mode.
        FullStrides: The effective stride types, one per mode.
        Inner: The inner map's basis images, one list per mode.
    """

    comptime _shape_types = Self.FullShapes
    comptime _stride_types = Self.FullStrides
    comptime __shape_types = Self._shape_types.values
    comptime __stride_types = Self._stride_types.values

    var _outer: Self.Outer
    """The integer outer layout."""

    var _x0: Int
    """The `w`-bit XOR base inside a footprint."""

    var _shape: Coord[*Self._shape_types]
    """The full shape, with each mode scaled by its inner bit count."""

    var _stride: Coord[*Self._stride_types]
    """The effective stride: the outer stride, or the lowest inner image."""

    comptime rank = Self.Outer.rank
    """The number of dimensions in the layout."""
    comptime flat_rank = Self.Outer.flat_rank
    """The number of dimensions after flattening nested coordinates."""
    comptime shape_known = Self.Outer.shape_known
    """Whether all shape dimensions are known at compile time."""
    comptime stride_known = Self.Outer.stride_known
    """Whether all stride dimensions are known at compile time."""
    comptime is_affine = False
    """Whether the layout's offset is a sum of coordinate times stride.

    `TwoLevelLayout` is never affine: the inner part XORs the images of the
    inner index bits.
    """

    comptime static_shape[i: Int]: Int = (
        Self._shape_types[i]
        .static_value if Self._shape_types[i]
        .is_static_value else -1
    )
    """Returns the compile-time value of the i-th flattened shape dimension.

    Parameters:
        i: The dimension index.
    """

    comptime static_stride[i: Int]: Int = (
        _LowestImage[Self.Inner, i] if _InnerBitCount[Self.Inner, i]
        > 0 else Self._stride_types[i].static_value
    )
    """Returns the compile-time value of the i-th flattened stride dimension.

    For an inner mode this is `Lambda` of its lowest bit; for an outer mode it
    is the outer stride.

    Parameters:
        i: The dimension index.
    """

    comptime static_product = Coord[*Self._shape_types]().static_product
    """The compile-time product of all shape dimensions."""

    comptime static_cosize = Self.Outer.static_cosize
    """The compile-time size of the memory region spanned by the layout."""

    comptime _TransposedLayout: TensorLayout = TwoLevelLayout[
        Self.Outer._TransposedLayout, _ReverseInner[Self.Inner]
    ]
    """The layout returned by `transpose`."""

    comptime _DynamicLayout[dtype: DType]: TensorLayout = TwoLevelLayout[
        Self.Outer._DynamicLayout[dtype], Self.Inner
    ]
    """The layout returned by `make_dynamic[dtype]`, with outer dimensions as
    runtime `Scalar[dtype]` values and the inner map unchanged.

    Parameters:
        dtype: The data type for the resulting `Scalar` values.
    """

    comptime _NestedTileLayout[
        TileShape: TypeList[Trait=CoordLike, ...]
    ]: TensorLayout = _CheckedNestedTileLayout[
        Self.Outer,
        Self.Inner,
        TileShape,
        _nested_box_check[Self.Outer, Self.Inner, TileShape](),
    ]
    """The layout of `TileTensor.tile` with tile shape `TileShape`.

    A tile that holds whole footprints keeps `Lambda` and takes `Outer`'s
    sub-layout. An aligned power-of-two box inside one footprint restricts
    `Lambda` to the box's low bits; its `x0` moves by `Lambda` of the box
    origin and is set when the tile view is built. Any other tile shape is
    rejected by `_nested_box_check` with a compile error naming the mode.

    Parameters:
        TileShape: The tile's extent along each mode.
    """

    @inline(.nodebug)
    def __init__(out self, var outer: Self.Outer, x0: Int = 0):
        """Initializes the layout from an outer layout and an XOR base.

        Args:
            outer: The integer outer layout.
            x0: The `w`-bit XOR base inside a footprint.
        """
        comptime assert (
            Self.Outer.rank == Self.Outer.flat_rank
        ), "TwoLevelLayout requires a flat outer layout"
        self._outer = outer
        self._x0 = x0
        self._shape = Coord[*Self._shape_types]()
        self._stride = Coord[*Self._stride_types]()
        comptime for i in range(Self.rank):
            comptime k = _InnerBitCount[Self.Inner, i]
            comptime if not Self._shape_types[i].is_static_value:
                Pointer(to=self._shape[i]).write(
                    rebind[type_of(self._shape[i])](
                        Scalar[Self._shape_types[i].DTYPE](
                            Int(self._outer.shape_coord()[i]) << k
                        )
                    )
                )
            comptime if not Self._stride_types[i].is_static_value:
                Pointer(to=self._stride[i]).write(
                    rebind[type_of(self._stride[i])](self._outer.stride[i]())
                )

    @inline(.nodebug)
    def __call__[
        index_type: CoordLike,
        *,
        linear_idx_type: DType = .int64,
    ](self, index: index_type) -> Scalar[linear_idx_type]:
        """Maps a logical coordinate to a linear memory index.

        The coordinate splits per mode into an outer part and `k[i]` low inner
        bits. The result is the outer offset plus the `x0`-based XOR of the
        inner images.

        Parameters:
            index_type: The coordinate type.
            linear_idx_type: The data type for the returned linear index.

        Args:
            index: The logical coordinates to map.

        Returns:
            `Outer(o) + (x0 ^ Lambda(y))`.
        """
        comptime R = Self.rank
        var values = Array[Int, R]()
        comptime if index_type.is_tuple:
            comptime assert (
                index_type.__len__() == R
            ), "TwoLevelLayout index rank does not match the layout rank"
            var coords = index.tuple()
            comptime for i in range(R):
                comptime assert not index_type.ParamListType[
                    i
                ].is_tuple, "TwoLevelLayout does not accept nested coordinates"
                values[i] = Int(coords[i].value())
        else:
            var n = Int(index.value())
            comptime for i in range(R):
                comptime shape_type = Self._shape_types[i]
                var mode_size: Int
                comptime if shape_type.is_static_value:
                    mode_size = shape_type.static_value
                else:
                    mode_size = (
                        Int(self._outer.shape_coord()[i])
                        << _InnerBitCount[Self.Inner, i]
                    )
                var quotient, remainder = divmod(n, mode_size)
                values[i] = remainder
                n = quotient

        var outer_coord = DynamicCoord[.int64, R]()
        var inner_bits = 0
        comptime for i in range(R):
            comptime k = _InnerBitCount[Self.Inner, i]
            var value = values[i]
            comptime if k > 0:
                var low = value & ((1 << k) - 1)
                comptime for b in range(k):
                    if ((low >> b) & 1) == 1:
                        inner_bits ^= _InnerImage[Self.Inner, i, b]
                value = value >> k
            Pointer(to=outer_coord[i]).write(
                rebind[type_of(outer_coord[i])](Scalar[.int64](value))
            )

        var result = Int(self._outer(outer_coord)) + (self._x0 ^ inner_bits)
        return Scalar[linear_idx_type](result)

    @inline(.nodebug)
    def idx2crd[
        *,
        out_dtype: DType = .int64,
    ](self, idx: Int) -> Coord[
        *_NestedDynamicCoord[out_dtype, *Self._shape_types]
    ]:
        """Maps a linear memory index back to logical coordinates.

        The low `w` bits are inverted through `Lambda` after the `x0` XOR; the
        outer part is recovered by `Outer`'s division.

        Parameters:
            out_dtype: The data type for the output coordinate values.

        Args:
            idx: The linear memory index to convert to coordinates.

        Returns:
            A Coord with the logical coordinates of `idx`.
        """
        comptime ResultType = Coord[
            *_NestedDynamicCoord[out_dtype, *Self._shape_types]
        ]
        var result = ResultType()
        var low = idx & _InnerMask[Self.Inner]
        var value = low ^ self._x0
        var y = 0
        comptime for p in range(_InnerMaskWidth[Self.Inner]):
            comptime inverse = _invert_inner_image[Self.Inner, p]()
            if inverse != -1 and ((value >> p) & 1) == 1:
                y ^= inverse
        var outer_coord = self._outer.idx2crd[out_dtype=out_dtype](idx)
        comptime for i in range(Self.rank):
            comptime k = _InnerBitCount[Self.Inner, i]
            var outer_value = Int(outer_coord[i])
            var coordinate: Int
            comptime if k > 0:
                var inner = (y >> _InnerBitOffset[Self.Inner, i]) & (
                    (1 << k) - 1
                )
                coordinate = (outer_value << k) | inner
            else:
                coordinate = outer_value
            Pointer(to=result[i]).write(
                rebind[ResultType.element_types[i]](
                    Scalar[out_dtype](coordinate)
                )
            )
        return result

    @inline(.nodebug)
    def product(self) -> Int:
        """Returns the total number of elements in the layout's domain.

        Returns:
            The product of all shape dimensions.
        """
        var result: Int = 1
        comptime for i in range(Self.rank):
            result *= Int(self._shape[i])
        return result

    @inline(.nodebug)
    def size(self) -> Int:
        """Returns the total number of elements. Alias for `product()`.

        Returns:
            The product of all shape dimensions.
        """
        return self.product()

    @inline(.nodebug)
    def shape[i: Int](self) -> Self._shape_types[i]:
        """Returns the i-th shape dimension.

        Parameters:
            i: The dimension index.

        Returns:
            The full shape value for dimension `i`.
        """
        return self._shape[i]

    @inline(.nodebug)
    def stride[i: Int](self) -> Self._stride_types[i]:
        """Returns the i-th stride dimension.

        For an inner mode this is `Lambda` of its lowest bit; for an outer
        mode it is the outer stride.

        Parameters:
            i: The dimension index.

        Returns:
            The effective stride value for dimension `i`.
        """
        return self._stride[i]

    @inline(.nodebug)
    def shape_coord(self) -> Coord[*Self._shape_types]:
        """Returns the full shape as a `Coord`.

        Returns:
            A Coord with the full shape dimensions.
        """
        return self._shape

    @inline(.nodebug)
    def stride_coord(self) -> Coord[*Self._stride_types]:
        """Returns the full stride as a `Coord`.

        Returns:
            A Coord with the effective strides.
        """
        return self._stride

    @inline(.nodebug)
    def transpose(self) -> Self._TransposedLayout:
        """Transposes the layout by reversing the order of dimensions.

        Returns:
            A new `TwoLevelLayout` with shape, strides and inner modes
            reversed.
        """
        return TwoLevelLayout[
            Self.Outer._TransposedLayout, _ReverseInner[Self.Inner]
        ](self._outer.transpose(), self._x0)

    @inline(.nodebug)
    def make_dynamic[dtype: DType](self) -> Self._DynamicLayout[dtype]:
        """Converts all outer dimensions to runtime values of the given dtype.

        The inner map stays comptime; only `Outer`'s shape and stride become
        `Scalar[dtype]`.

        Parameters:
            dtype: The data type for the resulting `Scalar` values.

        Returns:
            A new `TwoLevelLayout` with outer dimensions as `Scalar[dtype]`.
        """
        return TwoLevelLayout[Self.Outer._DynamicLayout[dtype], Self.Inner](
            self._outer.make_dynamic[dtype](), self._x0
        )

    @staticmethod
    def require_tile_shape[TileShape: TypeList[Trait=CoordLike, ...]]() -> Bool:
        """Rejects a tile shape the layout cannot tile statically.

        A tile must hold whole footprints on every mode, or be an aligned
        power-of-two box inside one footprint. Any other box fails to compile
        with a message naming the mode.

        Parameters:
            TileShape: The tile's extent along each mode.

        Returns:
            True when the tile shape is admissible.
        """
        return _nested_box_check[Self.Outer, Self.Inner, TileShape]()

    @inline(.nodebug)
    def write_to(self, mut writer: Some[Writer]):
        """Writes the layout representation to a Writer.

        Args:
            writer: The object to write to.
        """
        writer.write(
            t"TwoLevelLayout({self._shape}:{self._stride}, x0={self._x0},"
            t" mask={_InnerMask[Self.Inner]})"
        )

    @staticmethod
    def get_type_name() -> String:
        """Gets the name of the host type.

        Returns:
            The host type's name.
        """
        return String(
            t"TwoLevelLayout[{Coord[*Self._shape_types].get_type_name()},"
            t" {Coord[*Self._stride_types].get_type_name()}]"
        )

    comptime device_type = Self
    """Device-side type for GPU kernel parameter passing."""

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)


comptime TwoLevelLayout[
    Outer: TensorLayout, Inner: TypeList[Trait=CoordLike, ...]
]: TensorLayout = _TwoLevelLayout[
    Outer,
    _FullShapeTypes[Outer, Inner],
    _FullStrideTypes[Outer, Inner],
    Inner,
]
"""A layout with an integer outer part and an F2-linear inner part.

`Outer` maps each mode's high coordinate bits to a footprint base; every one
of its offsets is a multiple of `2 ^ w`. `Inner` holds one list of basis
images per mode, one image per low inner index bit. A coordinate `x` splits
per mode into `x[i] >> k[i]` (outer) and the low `k[i]` bits (inner), and
maps to

```
Outer(o) + (x0 ^ Lambda(y))
```

where `Lambda` XORs the images of the set inner index bits and `x0` is a
`w`-bit XOR base, zero for a whole tensor. Because the inner part is not a
sum of coordinate times stride, `is_affine` is `False`.

Parameters:
    Outer: The integer outer layout.
    Inner: The inner map's basis images, one list per mode. A mode with
        `k[i]` inner index bits contributes `k[i]` images, low bit first; a
        mode without inner bits contributes an empty list.
"""


comptime _CheckedNestedTileLayout[
    Outer: TensorLayout,
    Inner: TypeList[Trait=CoordLike, ...],
    TileShape: TypeList[Trait=CoordLike, ...],
    checked: Bool,
]: TensorLayout = TwoLevelLayout[
    Outer._NestedTileLayout[_NestedOuterTypes[Outer, Inner, TileShape]],
    _NestedInnerTypes[Outer, Inner, TileShape],
]
"""The nested tile layout type. `checked` carries the result of
`_nested_box_check`, so the check is written where the type is built."""


# ===----------------------------------------------------------------------=== #
# Swizzle constructor
# ===----------------------------------------------------------------------=== #


comptime _SwizzleMask[bits: Int, base: Int, shift: Int]: Int = (
    (1 << bits) - 1
) << (base + (shift if shift > 0 else 0))


comptime _SwizzleImage[bits: Int, base: Int, shift: Int, value: Int]: Int = (
    value ^ ((value & _SwizzleMask[bits, base, shift]) >> shift) if shift
    > 0 else value ^ ((value & _SwizzleMask[bits, base, shift]) << -shift)
)


comptime _SwizzlePeriod[bits: Int, base: Int, shift: Int]: Int = (
    base + bits + (shift if shift > 0 else -shift)
)
"""The offset bit above which a `Swizzle` changes nothing."""


comptime _SwizzleInnerBits[
    LayoutType: TensorLayout, bits: Int, base: Int, shift: Int, i: Int
]: Int = (
    _IntMin[
        _log2_or_zero(LayoutType._shape_types[i].static_value),
        _IntMax[
            0,
            _SwizzlePeriod[bits, base, shift]
            - _log2_or_zero(LayoutType._stride_types[i].static_value),
        ],
    ] if _PowerOfTwo[LayoutType._shape_types[i].static_value]
    and _PowerOfTwo[LayoutType._stride_types[i].static_value] else 0
)
"""How many low bits of mode `i` the swizzle's period covers."""


comptime _SwizzleImageType[
    LayoutType: TensorLayout, bits: Int, base: Int, shift: Int, i: Int, j: Int
]: CoordLike = ComptimeInt[
    _SwizzleImage[
        bits,
        base,
        shift,
        LayoutType._stride_types[i].static_value << j,
    ]
]


comptime _SwizzleInnerEntry[
    LayoutType: TensorLayout, bits: Int, base: Int, shift: Int, i: Int
] = Coord[
    *TypeList.tabulate[
        Trait=CoordLike,
        _SwizzleInnerBits[LayoutType, bits, base, shift, i],
        _SwizzleImageType[LayoutType, bits, base, shift, i, _],
    ]()
]


comptime _SwizzleInnerTabulator[
    LayoutType: TensorLayout, bits: Int, base: Int, shift: Int, i: Int
]: CoordLike = _SwizzleInnerEntry[LayoutType, bits, base, shift, i]


comptime _SwizzleInner[
    LayoutType: TensorLayout, bits: Int, base: Int, shift: Int
] = TypeList.tabulate[
    Trait=CoordLike,
    LayoutType.rank,
    _SwizzleInnerTabulator[LayoutType, bits, base, shift, _],
]()


comptime _SwizzleOuterShapeTabulator[
    LayoutType: TensorLayout, bits: Int, base: Int, shift: Int, i: Int
]: CoordLike = ComptimeInt[
    LayoutType._shape_types[i].static_value
    >> _SwizzleInnerBits[LayoutType, bits, base, shift, i]
]


comptime _SwizzleOuterStrideTabulator[
    LayoutType: TensorLayout, bits: Int, base: Int, shift: Int, i: Int
]: CoordLike = ComptimeInt[
    LayoutType._stride_types[i].static_value
    << _SwizzleInnerBits[LayoutType, bits, base, shift, i]
]


comptime _SwizzleOuter[
    LayoutType: TensorLayout, bits: Int, base: Int, shift: Int
] = Layout[
    TypeList.tabulate[
        Trait=CoordLike,
        LayoutType.rank,
        _SwizzleOuterShapeTabulator[LayoutType, bits, base, shift, _],
    ](),
    TypeList.tabulate[
        Trait=CoordLike,
        LayoutType.rank,
        _SwizzleOuterStrideTabulator[LayoutType, bits, base, shift, _],
    ](),
]
"""The outer layout of the swizzle split: shapes and strides pushed above
the swizzle's period."""


comptime SwizzledTwoLevelLayout[
    LayoutType: TensorLayout, bits: Int, base: Int, shift: Int
]: TensorLayout = TwoLevelLayout[
    _SwizzleOuter[LayoutType, bits, base, shift],
    _SwizzleInner[LayoutType, bits, base, shift],
]
"""The exact `TwoLevelLayout` of a `Swizzle[bits, base, shift]` applied to a
fully-static integer layout.

Parameters:
    LayoutType: The integer layout the swizzle is applied to.
    bits: The swizzle mask width.
    base: The swizzle's constant low bits.
    shift: The swizzle's mask shift.
"""


def _swizzle_layout_valid[LayoutType: TensorLayout]() -> Bool:
    """Checks that a layout admits the swizzle split."""
    if not LayoutType.all_dims_known:
        return False
    comptime if LayoutType.rank != LayoutType.flat_rank:
        return False
    var valid = True
    comptime for i in range(LayoutType.rank):
        if not _PowerOfTwo[LayoutType._shape_types[i].static_value]:
            valid = False
        if not _PowerOfTwo[LayoutType._stride_types[i].static_value]:
            valid = False
    return valid


def _swizzle_inner_bits_disjoint[
    LayoutType: TensorLayout, bits: Int, base: Int, shift: Int
]() -> Bool:
    """Checks that the inner bit contributions of the modes share no offset
    bit, so the split has no carries."""
    var seen = 0
    var disjoint = True
    comptime for i in range(LayoutType.rank):
        comptime stride_log = _log2_or_zero(
            LayoutType._stride_types[i].static_value
        )
        comptime k = _SwizzleInnerBits[LayoutType, bits, base, shift, i]
        for b in range(k):
            var bit = stride_log + b
            if ((seen >> bit) & 1) == 1:
                disjoint = False
            seen |= 1 << bit
    return disjoint


def two_level_layout_from_swizzle[
    LayoutType: TensorLayout, bits: Int, base: Int, shift: Int
](layout: LayoutType,) -> SwizzledTwoLevelLayout[LayoutType, bits, base, shift]:
    """Builds the exact `TwoLevelLayout` of a swizzle and an integer layout.

    The layout splits as `L = L_out + L_in`, where `L_in` is every coordinate
    contribution below the swizzle's period. `Lambda` is `Swizzle` composed
    with `L_in`, so the result evaluates to `Swizzle(L(x))` for every
    coordinate `x`.

    Parameters:
        LayoutType: The integer layout type. All dimensions must be static
            powers of two, and the mode strides must not overlap on the bits
            the swizzle's period covers.
        bits: The swizzle mask width.
        base: The swizzle's constant low bits.
        shift: The swizzle's mask shift.

    Args:
        layout: The integer layout, used to anchor the result's type.

    Returns:
        The two-level layout with `Outer = L_out` and `Lambda = Swizzle o L_in`.
    """
    comptime assert _swizzle_layout_valid[LayoutType](), (
        "two_level_layout_from_swizzle requires a flat layout whose shape and"
        " stride are all static powers of two"
    )
    comptime assert _swizzle_inner_bits_disjoint[
        LayoutType, bits, base, shift
    ](), "two_level_layout_from_swizzle: mode inner bits overlap"
    comptime OuterType = _SwizzleOuter[LayoutType, bits, base, shift]
    return SwizzledTwoLevelLayout[LayoutType, bits, base, shift](OuterType(), 0)


def two_level_layout_from_swizzle[
    LayoutType: TensorLayout, swizzle: Swizzle
](
    layout: LayoutType,
) -> SwizzledTwoLevelLayout[
    LayoutType, swizzle.bits, swizzle.base, swizzle.shift
]:
    """Builds the exact `TwoLevelLayout` of a swizzle and an integer layout.

    Parameters:
        LayoutType: The integer layout type.
        swizzle: The swizzle to fold into the inner map.

    Args:
        layout: The integer layout, used to anchor the result's type.

    Returns:
        The two-level layout with `Outer = L_out` and `Lambda = swizzle o L_in`.
    """
    return two_level_layout_from_swizzle[
        LayoutType, swizzle.bits, swizzle.base, swizzle.shift
    ](layout)
