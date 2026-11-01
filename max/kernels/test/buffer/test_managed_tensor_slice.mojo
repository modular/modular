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
"""Tests for ManagedTensorSlice - a tensor view type for custom graph operations.
"""

from extensibility import get_row_major_tensor_spec_static
from extensibility import ManagedTensorSlice, IOSpec
from extensibility.managed_tensor_slice import (
    OutputFusion,
    StaticTensorSpec,
    _IndexListToTileLayout,
)
from layout import coord_to_index_list
from std.ffi import external_call
from std.memory import AddressSpace
from std.sys import align_of, size_of
from std.sys.info import CompilationTarget
from max.gpu.host import DeviceContext
from extensibility import foreach
from layout import Coord
from std.testing import assert_equal, assert_true, TestSuite

from std.utils import IndexList


def test_basic_construction() raises:
    """Test basic ManagedTensorSlice construction from pointer and shape."""
    var storage = Array[Float32, 3 * 4](fill={})
    # Shape-only constructor computes row-major strides automatically
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 3, 4]()
    var tensor = ManagedTensorSlice[io_spec=IOSpec.Unknown, static_spec=spec](
        storage.unsafe_ptr(), (3, 4)
    )

    assert_equal(tensor.rank, 2)
    assert_equal(tensor.size(), 12)


def test_shape_and_strides() raises:
    """Test shape() and strides() methods."""
    var storage = Array[Float32, 2 * 3 * 4](fill={})
    comptime spec = get_row_major_tensor_spec_static[
        DType.float32, 3, 2, 3, 4
    ]()
    var tensor = ManagedTensorSlice[io_spec=IOSpec.Unknown, static_spec=spec](
        storage.unsafe_ptr(), (2, 3, 4)
    )

    var shape = tensor.shape()
    assert_equal(shape[0], 2)
    assert_equal(shape[1], 3)
    assert_equal(shape[2], 4)

    var strides = tensor.strides()
    assert_equal(strides[0], 12)
    assert_equal(strides[1], 4)
    assert_equal(strides[2], 1)


def test_dim_size() raises:
    """Test dim_size methods (compile-time and runtime)."""
    var storage = Array[Float32, 5 * 7](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 5, 7]()
    var tensor = ManagedTensorSlice[io_spec=IOSpec.Unknown, static_spec=spec](
        storage.unsafe_ptr(), (5, 7)
    )

    # Test compile-time dim_size
    assert_equal(tensor.dim_size[0](), 5)
    assert_equal(tensor.dim_size[1](), 7)

    # Test runtime dim_size
    assert_equal(tensor.dim_size(0), 5)
    assert_equal(tensor.dim_size(1), 7)


def test_getitem_setitem() raises:
    """Test __getitem__ and __setitem__ operations."""
    var storage = Array[Float32, 2 * 3](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 2, 3]()
    var tensor = ManagedTensorSlice[
        mut=True, io_spec=IOSpec.Unknown, static_spec=spec
    ](storage.unsafe_ptr(), (2, 3))

    # Set values
    tensor[0, 0] = 1.0
    tensor[0, 1] = 2.0
    tensor[0, 2] = 3.0
    tensor[1, 0] = 4.0
    tensor[1, 1] = 5.0
    tensor[1, 2] = 6.0

    # Get values using variadic indices
    assert_equal(tensor[0, 0], 1.0)
    assert_equal(tensor[0, 1], 2.0)
    assert_equal(tensor[0, 2], 3.0)
    assert_equal(tensor[1, 0], 4.0)
    assert_equal(tensor[1, 1], 5.0)
    assert_equal(tensor[1, 2], 6.0)

    # Get values using IndexList
    assert_equal(tensor[(0, 2)], 3.0)
    assert_equal(tensor[(1, 0)], 4.0)


def test_simd_load_store() raises:
    """Test SIMD load and store operations."""
    var storage = Array[Float32, 8](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 1, 8]()
    var tensor = ManagedTensorSlice[
        mut=True, io_spec=IOSpec.Unknown, static_spec=spec
    ](storage.unsafe_ptr(), (8,))

    # Store a SIMD vector
    var vec = SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0)
    tensor.store((0,), vec)

    var vec2 = SIMD[.float32, 4](5.0, 6.0, 7.0, 8.0)
    tensor.store((4,), vec2)

    # Load and verify
    var loaded = tensor.load[4]((0,))
    assert_equal(loaded, SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0))

    var loaded2 = tensor.load[4]((4,))
    assert_equal(loaded2, SIMD[.float32, 4](5.0, 6.0, 7.0, 8.0))


def test_stride_length() raises:
    """Test stride_length methods."""
    var storage = Array[Float32, 3 * 5](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 3, 5]()
    var tensor = ManagedTensorSlice[io_spec=IOSpec.Unknown, static_spec=spec](
        storage.unsafe_ptr(), (3, 5)
    )

    # Test compile-time stride_length
    assert_equal(tensor.stride_length[0](), 5)
    assert_equal(tensor.stride_length[1](), 1)

    # Test runtime stride_length
    assert_equal(tensor.stride_length(0), 5)
    assert_equal(tensor.stride_length(1), 1)


def test_simd_load_store_2d() raises:
    """Test SIMD load and store operations on 2D tensor."""
    var storage = Array[Float32, 4 * 8](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 4, 8]()
    var tensor = ManagedTensorSlice[
        mut=True, io_spec=IOSpec.Unknown, static_spec=spec
    ](storage.unsafe_ptr(), (4, 8))

    # Store vectors in each row
    for i in range(4):
        var vec = SIMD[.float32, 4](
            Float32(i * 10),
            Float32(i * 10 + 1),
            Float32(i * 10 + 2),
            Float32(i * 10 + 3),
        )
        tensor.store((i, 0), vec)

        var vec2 = SIMD[.float32, 4](
            Float32(i * 10 + 4),
            Float32(i * 10 + 5),
            Float32(i * 10 + 6),
            Float32(i * 10 + 7),
        )
        tensor.store((i, 4), vec2)

    # Load and verify
    var loaded_row0 = tensor.load[4]((0, 0))
    assert_equal(loaded_row0, SIMD[.float32, 4](0.0, 1.0, 2.0, 3.0))

    var loaded_row2 = tensor.load[4]((2, 4))
    assert_equal(loaded_row2, SIMD[.float32, 4](24.0, 25.0, 26.0, 27.0))

    var loaded_row3 = tensor.load[4]((3, 0))
    assert_equal(loaded_row3, SIMD[.float32, 4](30.0, 31.0, 32.0, 33.0))


def test_to_tile_tensor() raises:
    """Test to_tile_tensor() conversion preserves dtype, layout, and pointer."""
    var storage = Array[Float32, 3 * 4](
        fill_with=lambda (i: Int) -> Float32: Float32(i)
    )
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 3, 4]()
    var tensor = ManagedTensorSlice[
        mut=True, io_spec=IOSpec.Unknown, static_spec=spec
    ](storage.unsafe_ptr(), (3, 4))

    # Convert to TileTensor
    var tile_tensor = tensor.to_tile_tensor[.int64]()

    # Verify dtype and rank are carried across.
    comptime assert tile_tensor.dtype == DType.float32
    comptime assert tile_tensor.flat_rank == 2

    # Verify the tile tensor views the same data.
    assert_equal(tile_tensor[0, 0], 0.0)
    assert_equal(tile_tensor[1, 1], 5.0)
    assert_equal(tile_tensor[2, 3], 11.0)

    # Verify the tile tensor is a view over the slice, not a copy.
    assert_equal(Int(tile_tensor.unsafe_ptr()), Int(tensor.unsafe_ptr()))

    # Verify dimensions and strides match the slice.
    assert_equal(tile_tensor.layout.shape[0]().value(), 3)
    assert_equal(tile_tensor.layout.shape[1]().value(), 4)
    assert_equal(tile_tensor.layout.stride[0]().value(), 4)
    assert_equal(tile_tensor.layout.stride[1]().value(), 1)

    # TODO(GEX-4147): ManagedTensorSlice needs to carry the Array's origin
    # `tensor` holds an untracked pointer into `storage`; keep `storage` alive
    # until the last read through it.
    _ = storage^


def test_shape_coord_static() raises:
    """Test shape_coord() preserves fully-static shape information."""
    var storage = Array[Float32, 3 * 4](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 3, 4]()
    var tensor = ManagedTensorSlice[io_spec=IOSpec.Unknown, static_spec=spec](
        storage.unsafe_ptr(), (3, 4)
    )

    var shape = tensor.shape_coord()

    # Every dimension is statically known, so this is a compile-time fact.
    comptime assert shape.all_dims_known
    comptime assert shape.element_types[0].static_value == 3
    comptime assert shape.element_types[1].static_value == 4

    # The runtime values still round-trip correctly.
    var index_list = coord_to_index_list(shape)
    assert_equal(index_list[0], 3)
    assert_equal(index_list[1], 4)


def test_shape_coord_mixed() raises:
    """Test shape_coord() encodes static dims while filling dynamic ones."""
    var storage = Array[Float32, 2 * 4](fill={})
    # dim 0 is dynamic (-1), dim 1 is static (4); strides are row-major.
    comptime mixed_layout = _IndexListToTileLayout[
        IndexList[2](-1, 4), IndexList[2](4, 1)
    ]
    comptime mixed_spec = StaticTensorSpec[
        DType.float32, 2, static_layout=mixed_layout
    ](align_of[DType.float32](), AddressSpace.GENERIC)
    var tensor = ManagedTensorSlice[
        io_spec=IOSpec.Unknown, static_spec=mixed_spec
    ](storage.unsafe_ptr(), (2, 4))

    var shape = tensor.shape_coord()

    # The static/dynamic structure is preserved in the Coord's type.
    comptime assert not shape.all_dims_known
    comptime assert not shape.element_types[0].is_static_value
    comptime assert shape.element_types[1].is_static_value
    comptime assert shape.element_types[1].static_value == 4

    # The dynamic dimension is filled from the runtime shape.
    var index_list = coord_to_index_list(shape)
    assert_equal(index_list[0], 2)
    assert_equal(index_list[1], 4)


def test_strides_coord_static() raises:
    """Test strides_coord() preserves fully-static stride information."""
    var storage = Array[Float32, 3 * 4](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 3, 4]()
    var tensor = ManagedTensorSlice[io_spec=IOSpec.Unknown, static_spec=spec](
        storage.unsafe_ptr(), (3, 4)
    )

    var strides = tensor.strides_coord()

    # Row-major strides are statically known, so this is a compile-time fact.
    comptime assert strides.all_dims_known
    comptime assert strides.element_types[0].static_value == 4
    comptime assert strides.element_types[1].static_value == 1

    # The runtime values still round-trip correctly.
    var index_list = coord_to_index_list(strides)
    assert_equal(index_list[0], 4)
    assert_equal(index_list[1], 1)


def test_strides_coord_mixed() raises:
    """Test strides_coord() encodes static strides while filling dynamic ones.
    """
    var storage = Array[Float32, 2 * 4](fill={})
    # Shape is static (2, 4); stride 0 is dynamic (-1) and stride 1 is static.
    comptime mixed_layout = _IndexListToTileLayout[
        IndexList[2](2, 4), IndexList[2](-1, 1)
    ]
    comptime mixed_spec = StaticTensorSpec[
        DType.float32, 2, static_layout=mixed_layout
    ](align_of[DType.float32](), AddressSpace.GENERIC)
    var tensor = ManagedTensorSlice[
        io_spec=IOSpec.Unknown, static_spec=mixed_spec
    ](storage.unsafe_ptr(), (2, 4), (4, 1))

    var strides = tensor.strides_coord()

    # The static/dynamic structure is preserved in the Coord's type.
    comptime assert not strides.all_dims_known
    comptime assert not strides.element_types[0].is_static_value
    comptime assert strides.element_types[1].is_static_value
    comptime assert strides.element_types[1].static_value == 1

    # The dynamic stride is filled from the runtime strides.
    var index_list = coord_to_index_list(strides)
    assert_equal(index_list[0], 4)
    assert_equal(index_list[1], 1)


# ===----------------------------------------------------------------------=== #
# `foreach` — the value (unified-closure) overload and its parametric twin.
#
# Every case runs from a parametric helper: the callback's return type has to
# name `dtype` as a parameter reference for it to unify with the overload's
# inferred `dtype`. A literal (`SIMD[DType.float32, width]`) does not.
#
# A test that needs two buffers carves both out of one `Array`. Two separate
# `Array`s can end up sharing a stack slot once `unsafe_ptr()` is taken, which
# silently aliases them.
# ===----------------------------------------------------------------------=== #


def _lane_ramp[dtype: DType, width: Int](start: Int) -> SIMD[dtype, width]:
    """`[start, start + 1, ...]` — a body that varies per lane, so a wrong
    index or a collapsed vector shows up as a wrong element."""
    var v = SIMD[dtype, width](0)
    for lane in range(width):
        v[lane] = Scalar[dtype](start + lane)
    return v


def _flat_view[
    dtype: DType, n: Int
](
    ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    out result: ManagedTensorSlice[
        mut=True,
        io_spec=IOSpec.Unknown,
        static_spec=get_row_major_tensor_spec_static[dtype, 1, n](),
    ],
):
    """A readable/writable rank-1 view, for setting up inputs and checking
    results without going back through the raw pointer."""
    return {ptr, (n,)}


def _flat_output[
    dtype: DType, n: Int
](
    ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    out result: ManagedTensorSlice[
        io_spec=IOSpec.Output,
        static_spec=get_row_major_tensor_spec_static[dtype, 1, n](),
    ],
):
    return {ptr, (n,)}


def _flat_input[
    dtype: DType, n: Int
](
    ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    out result: ManagedTensorSlice[
        io_spec=IOSpec.Input,
        static_spec=get_row_major_tensor_spec_static[dtype, 1, n](),
    ],
):
    return {ptr, (n,)}


def _fill_ramp[
    dtype: DType, n: Int
](tensor: ManagedTensorSlice[mut=True, dtype=dtype, rank=1, ...]):
    for i in range(n):
        tensor.store((i,), SIMD[dtype, 1](Scalar[dtype](i)))


def _check_value_form_rank1[dtype: DType]() raises:
    comptime N = 32
    var storage = Array[Scalar[dtype], N](fill={})
    var base = storage.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var out = _flat_output[dtype, N](base)
    var view = _flat_view[dtype, N](base)

    var bias = Scalar[dtype](3)

    @inline(.always)
    def body[width: Int](idx: Coord) {var bias} -> SIMD[dtype, width]:
        return SIMD[dtype, width](bias)

    var ctx = DeviceContext(api="cpu")
    foreach(body, out, ctx)

    for i in range(N):
        assert_equal(view.load[1]((i,)), Scalar[dtype](3))
    _ = storage^


def test_foreach_value_form_rank1() raises:
    """The value overload resolves and runs over a rank-1 tensor."""
    _check_value_form_rank1[.float32]()


def _check_value_form_rank2[dtype: DType]() raises:
    comptime ROWS = 3
    comptime COLS = 4
    var storage = Array[Scalar[dtype], ROWS * COLS](fill={})
    var base = storage.unsafe_ptr()
    comptime spec = get_row_major_tensor_spec_static[dtype, 2, ROWS, COLS]()
    var out = ManagedTensorSlice[io_spec=IOSpec.Output, static_spec=spec](
        base, (ROWS, COLS)
    )
    var view = ManagedTensorSlice[
        mut=True, io_spec=IOSpec.Unknown, static_spec=spec
    ](base, (ROWS, COLS))

    var scale = Scalar[dtype](10)

    # `Coord` carries the index of the vector's first element, so the body
    # fills the remaining lanes along the last axis itself. A wrong index or a
    # collapsed vector shows up as a wrong element.
    @inline(.always)
    def body[width: Int](idx: Coord) {var scale} -> SIMD[dtype, width]:
        var il = coord_to_index_list(idx)
        var base_val = Scalar[dtype](il[0]) * scale + Scalar[dtype](il[1])
        var v = SIMD[dtype, width](0)
        for lane in range(width):
            v[lane] = base_val + Scalar[dtype](lane)
        return v

    var ctx = DeviceContext(api="cpu")
    foreach(body, out, ctx)

    for r in range(ROWS):
        for c in range(COLS):
            assert_equal(view.load[1]((r, c)), Scalar[dtype](r * 10 + c))
    _ = storage^


def test_foreach_value_form_rank2() raises:
    """The value overload handles rank 2, including `Coord` conversion."""
    _check_value_form_rank2[.float32]()


def _check_capture_outer_tensor[dtype: DType]() raises:
    comptime N = 32
    var storage = Array[Scalar[dtype], 2 * N](fill={})
    var base = storage.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var in_ptr = base
    var out_ptr = base.unsafe_offset(N)

    _fill_ramp[dtype, N](_flat_view[dtype, N](in_ptr))
    # A mut view, not an `IOSpec.Input` one, even though the migrated examples
    # capture inputs: `IOSpec.Input` means `mut=False`, which
    # `simd_load_from_managed_tensor_slice` turns into an LLVM
    # `!invariant.load`. Filling a buffer and reading it back in one function
    # breaks that promise — at N=32 the optimizer forwarded the `Array`
    # zero-init past `_fill_ramp` for the first vector. Real ops never write
    # their own input, and the overload does not depend on the `IOSpec`.
    var x = _flat_view[dtype, N](in_ptr)
    var out = _flat_output[dtype, N](out_ptr)
    var view = _flat_view[dtype, N](out_ptr)

    # The shape every migrated `max/examples/` caller uses: capture the input
    # tensor and read it through `load`.
    @inline(.always)
    def body[width: Int](idx: Coord) {var x} -> SIMD[dtype, width]:
        return x.load[width](idx) + 1

    var ctx = DeviceContext(api="cpu")
    foreach(body, out, ctx)

    for i in range(N):
        assert_equal(view.load[1]((i,)), Scalar[dtype](i + 1))
    _ = storage^


def test_foreach_captures_outer_tensor() raises:
    """A captured input tensor, the shape the migrated examples use."""
    _check_capture_outer_tensor[.float32]()


def _check_capture_tensor_and_scalar[dtype: DType]() raises:
    comptime N = 32
    var storage = Array[Scalar[dtype], 2 * N](fill={})
    var base = storage.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var in_ptr = base
    var out_ptr = base.unsafe_offset(N)

    _fill_ramp[dtype, N](_flat_view[dtype, N](in_ptr))
    # A mut view, for the invariant-load reason given above.
    var x = _flat_view[dtype, N](in_ptr)
    var out = _flat_output[dtype, N](out_ptr)
    var view = _flat_view[dtype, N](out_ptr)
    var addend = Scalar[dtype](7)

    # The `add_constant` shape: a tensor and a loose scalar in one capture list.
    @inline(.always)
    def body[width: Int](idx: Coord) {var x, var addend} -> SIMD[dtype, width]:
        return x.load[width](idx) + addend

    var ctx = DeviceContext(api="cpu")
    foreach(body, out, ctx)

    for i in range(N):
        assert_equal(view.load[1]((i,)), Scalar[dtype](i + 7))
    _ = storage^


def test_foreach_captures_tensor_and_scalar() raises:
    """Mixed capture list: an outer tensor alongside an outer scalar."""
    _check_capture_tensor_and_scalar[.float32]()


@inline(.never)
def _foreach_over_input[
    dtype: DType, N: Int
](
    in_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    out_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
) raises:
    """Runs a `foreach` whose body captures an `IOSpec.Input` tensor.

    Never inlined, on purpose. `IOSpec.Input` loads are LLVM invariant loads,
    and GVN resolves an invariant load by walking past every store to the
    allocation, so a load in the same function as the `Array` that backs it
    reads the allocation's undefined initial value, not what the fill wrote
    (observed on Linux x86 as NaN for the first vector). Behind a call
    boundary the walk stops at the pointer argument.
    """
    var x = _flat_input[dtype, N](in_ptr)
    var out = _flat_output[dtype, N](out_ptr)

    @inline(.always)
    def body[width: Int](idx: Coord) {var x} -> SIMD[dtype, width]:
        return x.load[width](idx) + _lane_ramp[dtype, width](
            coord_to_index_list(idx)[0]
        )

    var ctx = DeviceContext(api="cpu")
    foreach(body, out, ctx)


def _check_capture_input_tensor[dtype: DType]() raises:
    comptime N = 32
    comptime FILL = 7
    # The one test that captures a real `IOSpec.Input` tensor, so the shape the
    # migrated examples use is covered with its invariant loads intact. The
    # ramp lives in the body instead of the buffer, so the expected value still
    # varies per index.
    var in_storage = Array[Scalar[dtype], N](fill=Scalar[dtype](FILL))
    var out_storage = Array[Scalar[dtype], N](fill={})
    var in_ptr = in_storage.unsafe_ptr().unsafe_origin_cast[
        MutUntrackedOrigin
    ]()
    var out_ptr = out_storage.unsafe_ptr().unsafe_origin_cast[
        MutUntrackedOrigin
    ]()

    _foreach_over_input[dtype, N](in_ptr, out_ptr)

    var view = _flat_view[dtype, N](out_ptr)
    for i in range(N):
        assert_equal(view.load[1]((i,)), Scalar[dtype](FILL + i))
    _ = in_storage^
    _ = out_storage^


def test_foreach_captures_input_tensor() raises:
    """A captured `IOSpec.Input` tensor, invariant loads and all."""
    _check_capture_input_tensor[.float32]()


def _check_simd_width_one[dtype: DType]() raises:
    comptime N = 32
    var storage = Array[Scalar[dtype], N](fill={})
    var base = storage.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var out = _flat_output[dtype, N](base)
    var view = _flat_view[dtype, N](base)
    var one = Scalar[dtype](1)

    # `simd_width=1` is what `image_pipeline.mojo` and `grayscale.mojo` pass;
    # the body asserts it actually arrives.
    @inline(.always)
    def body[width: Int](idx: Coord) {var one} -> SIMD[dtype, width]:
        comptime assert width == 1, "simd_width=1 was not honored"
        return SIMD[dtype, width](
            Scalar[dtype](coord_to_index_list(idx)[0]) * one
        )

    var ctx = DeviceContext(api="cpu")
    foreach[simd_width=1](body, out, ctx)

    for i in range(N):
        assert_equal(view.load[1]((i,)), Scalar[dtype](i))
    _ = storage^


def test_foreach_honors_simd_width() raises:
    """A non-default `simd_width` reaches the body."""
    _check_simd_width_one[.float32]()


@fieldwise_init
struct _MarkingOutFusion[fusion_dtype: DType](OutputFusion):
    """Records stores through the fusion path, offset by 100 so a store that
    bypassed the fusion is distinguishable from one that used it."""

    var dst: Pointer[Scalar[Self.fusion_dtype], MutUntrackedOrigin]

    def store[
        dtype: DType,
        rank: Int,
        simd_width: SIMDLength,
        element_alignment: Int = 1,
    ](self, idx: IndexList[rank], val: SIMD[dtype, simd_width]):
        var start = idx[rank - 1]
        for lane in range(Int(simd_width)):
            self.dst.unsafe_offset(start + lane).unsafe_store(
                rebind[Scalar[Self.fusion_dtype]](val[lane]) + 100
            )


def _check_fused_store[dtype: DType]() raises:
    comptime N = 32
    # `direct` is what the tensor points at; `fused` is where the fusion
    # writes. A `foreach` that stored straight to the data pointer would fill
    # `direct` and leave `fused` untouched — the regression this guards.
    var storage = Array[Scalar[dtype], 2 * N](fill={})
    var base = storage.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var direct_ptr = base
    var fused_ptr = base.unsafe_offset(N)

    var direct_view = _flat_view[dtype, N](direct_ptr)
    var fused_view = _flat_view[dtype, N](fused_ptr)
    for i in range(N):
        direct_view.store((i,), SIMD[dtype, 1](Scalar[dtype](-1)))
        fused_view.store((i,), SIMD[dtype, 1](Scalar[dtype](-1)))

    var out = _flat_output[dtype, N](direct_ptr)
    var fused_out = out._bind_to_fused_output(_MarkingOutFusion(fused_ptr))
    var one = Scalar[dtype](1)

    @inline(.always)
    def body[width: Int](idx: Coord) {var one} -> SIMD[dtype, width]:
        return SIMD[dtype, width](
            Scalar[dtype](coord_to_index_list(idx)[0]) * one
        )

    var ctx = DeviceContext(api="cpu")
    foreach[simd_width=1](body, fused_out, ctx)

    for i in range(N):
        assert_equal(fused_view.load[1]((i,)), Scalar[dtype](i + 100))
        assert_equal(direct_view.load[1]((i,)), Scalar[dtype](-1))
    _ = storage^


def test_foreach_routes_through_fused_store() raises:
    """The value overload stores through `_fused_store`, not the data pointer.

    Without this the wrapper could write straight to the tensor and silently
    break output fusion while every other test still passed.
    """
    _check_fused_store[.float32]()


# ===----------------------------------------------------------------------=== #
# Strided SIMD loads
# ===----------------------------------------------------------------------=== #
#
# A view whose innermost stride is a static value in 2...8 is loaded with one
# contiguous window plus a lane shuffle on CPU, falling back to a gather when
# the window would run past the end of the view. These tests compare every
# load against a scalar reference, with the buffer ending exactly where the
# view does so that an overrun would touch a protected page.

comptime _GUARD_PAGE_BYTES = 16384
comptime _SENTINEL = 255


struct _GuardedBuffer[dtype: DType](Movable):
    """`count` elements that end exactly at an inaccessible page, so any read
    past the last element faults instead of silently returning garbage."""

    var _base: Int
    var ptr: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]

    def __init__(out self, count: Int) raises:
        comptime read_write = Int32(3)
        comptime no_access = Int32(0)
        # MAP_PRIVATE | MAP_ANONYMOUS
        comptime flags = Int32(
            0x1002
        ) if CompilationTarget.is_macos() else Int32(0x22)
        var bytes = count * size_of[Self.dtype]()
        if bytes > _GUARD_PAGE_BYTES:
            raise Error("buffer does not fit in one page")
        self._base = external_call["mmap", Int](
            Int(0),
            Int(2 * _GUARD_PAGE_BYTES),
            read_write,
            flags,
            Int32(-1),
            Int(0),
        )
        if self._base == -1:
            raise Error("mmap failed")
        var guard = external_call["mprotect", Int32](
            Int(self._base + _GUARD_PAGE_BYTES),
            Int(_GUARD_PAGE_BYTES),
            no_access,
        )
        if guard != 0:
            raise Error("mprotect failed")
        self.ptr = Pointer[Scalar[Self.dtype], MutUntrackedOrigin](
            unsafe_from_address=self._base + _GUARD_PAGE_BYTES - bytes
        )

    def __deinit__(deinit self):
        _ = external_call["munmap", Int32](
            self._base, Int(2 * _GUARD_PAGE_BYTES)
        )


def _ramp_value[dtype: DType](i: Int) -> Scalar[dtype]:
    """Distinct small values (all below `_SENTINEL`, exact in bfloat16)."""
    return Scalar[dtype]((i * 5 + 1) % 251)


def _check_view[
    dtype: DType, stride: Int, width: Int
](
    ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    rows: Int,
    cols: Int,
    row_stride: Int,
) raises:
    """Loads every in-row window of a `rows x cols` view with innermost stride
    `stride` and compares it with the scalar element at the same offset."""
    comptime layout = _IndexListToTileLayout[
        IndexList[2](-1, -1), IndexList[2](-1, stride)
    ]
    comptime spec = StaticTensorSpec[dtype, 2, layout](1, AddressSpace.GENERIC)
    var view = ManagedTensorSlice[io_spec=IOSpec.Input, static_spec=spec](
        ptr, (rows, cols), (row_stride, stride)
    )
    for r in range(rows):
        for c in range(cols - width + 1):
            var got = view.load[width]((r, c))
            for lane in range(width):
                var expected = ptr.unsafe_load(
                    r * row_stride + (c + lane) * stride
                )
                assert_equal(got[lane], expected)
                assert_true(got[lane] != Scalar[dtype](_SENTINEL))


def _check_stride_width[dtype: DType, stride: Int, width: Int]() raises:
    # 37 is not a multiple of any tested width, so the last window of a row
    # does not line up with the end of the row.
    comptime rows = 3
    comptime cols = 37

    # A step view over an exactly-sized buffer, the last lane of the last row
    # being the last element: every window near the end must fall back.
    var row_stride = cols * stride
    var extent = (rows - 1) * row_stride + (cols - 1) * stride + 1
    var buf = _GuardedBuffer[dtype](extent)
    for i in range(extent):
        buf.ptr.unsafe_offset(i).unsafe_write(_ramp_value[dtype](i))
    _check_view[dtype, stride, width](buf.ptr, rows, cols, row_stride)

    # The same buffer with the rows reversed (negative outer stride), so the
    # view's first row is the one at the end of the buffer.
    _check_view[dtype, stride, width](
        buf.ptr.unsafe_offset((rows - 1) * row_stride), rows, cols, -row_stride
    )
    # `ptr` does not keep the mapping alive, so end the buffers' lifetimes
    # explicitly.
    _ = buf^

    # An NHWC -> NCHW style permute: `stride` channels, innermost stride
    # `stride`, outer stride 1.
    var permuted_extent = stride * cols
    var permuted = _GuardedBuffer[dtype](permuted_extent)
    for i in range(permuted_extent):
        permuted.ptr.unsafe_offset(i).unsafe_write(_ramp_value[dtype](i))
    _check_view[dtype, stride, width](permuted.ptr, stride, cols, 1)
    _ = permuted^

    # A view that is smaller than its buffer: the buffer, including the gaps
    # between lanes and rows, is sentinel outside the view, so a window that
    # fits in the buffer but not in the view must not leak those values.
    var padded_row_stride = (cols + 5) * stride
    var view_extent = (rows - 1) * padded_row_stride + (cols - 1) * stride + 1
    var total = view_extent + 64
    var padded = _GuardedBuffer[dtype](total)
    for i in range(total):
        padded.ptr.unsafe_offset(i).unsafe_write(Scalar[dtype](_SENTINEL))
    for r in range(rows):
        for c in range(cols):
            var i = r * padded_row_stride + c * stride
            padded.ptr.unsafe_offset(i).unsafe_write(_ramp_value[dtype](i))
    _check_view[dtype, stride, width](padded.ptr, rows, cols, padded_row_stride)
    _ = padded^


def _check_strided_loads[dtype: DType]() raises:
    # 1 and 9 take the contiguous and gather paths; 2...8 take the window.
    comptime for stride in range(1, 10):
        _check_stride_width[dtype, stride, 1]()
        _check_stride_width[dtype, stride, 4]()
        _check_stride_width[dtype, stride, 16]()


def test_strided_load_uint8() raises:
    _check_strided_loads[.uint8]()


def test_strided_load_bfloat16() raises:
    _check_strided_loads[.bfloat16]()


def test_strided_load_float32() raises:
    _check_strided_loads[.float32]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
