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
"""Composable traits for graph-compiler kernel tensor arguments.

A kernel declares each tensor argument by combining small, orthogonal traits
with trait composition instead of naming a single monolithic tensor type.

The traits fall into three groups:

- Role traits (`Input`, `Output`, `MutableInput`) declare how the kernel uses
  the argument: read-only, write-only destination, or read-write in place.
  Exactly one role applies to each argument.
- The `Fused` marker declares that the operand participates in prologue or
  epilogue fusion.
- The `Stable` marker declares that the argument's storage is a stable address
  slot, read through `StableAddrEngine` (device only).
- Metadata and access traits describe what the kernel can read off the argument
  (`Tensor`) and how it touches individual elements.

Element access is split into marker / capability pairs. A graph-visible marker
(`LoadAccess`, `StoreAccess`) records the access form in the distilled contract
but adds no method; the matching capability (`Loadable`, `Storable`)
additionally supplies the method the kernel body calls. A kernel typically binds
the marker in its signature and refines to the capability inside its body, so
the same argument type works whether it stays on the host or is projected to a
device.

`_LegacyComputeOutput` is the exception: it doesn't guarantee its capability
`_LegacyComputable`, and the graph compiler may deliver `Storable` instead. It
is legacy machinery; read its docstring before using it.
"""

from std.memory import AddressSpace
from std.reflection import reflect

from layout import (
    Coord,
    CoordLike,
    CurrentTensorEngine,
    TensorEngine,
    TensorLayout,
    TileTensor,
)


# ===----------------------------------------------------------------------=== #
# Role traits
# ===----------------------------------------------------------------------=== #


trait Input:
    """Tells the graph compiler this argument is a read-only input."""

    pass


trait Output:
    """Tells the graph compiler this argument is an output."""

    pass


trait MutableInput:
    """Tells the graph compiler this argument is a mutable input."""

    pass


# ===----------------------------------------------------------------------=== #
# Fusion marker
# ===----------------------------------------------------------------------=== #


trait Fused:
    """Marker trait: the operand participates in fusion."""

    pass


trait UnsafeFusedView(Fused, TrivialRegisterPassable):
    """Flags a fused trait argument whose host and device views share a
    byte-identical layout.

    Conformance here is the legacy gate for the `unsafe_fused_view` rebind;
    it exists to keep that one host-built-closure construction compiling
    while kernels migrate. Prefer the supported pattern in new accelerator
    kernels: pass the fused trait arg straight through `enqueue_function`
    to a device kernel parameterized by `XHost: DevicePassable`, where the
    capability is reached via `downcast[XHost.device_type, ...]`. The
    canonical exemplars in `GraphCompiler/test/kernels/testing_kernels.mojo`
    follow that shape:

    - `TraitFusedLoadNegate` (`_gpu_fused_load_negate_kernel`).
    - `TraitFusedStoreDouble` (`_gpu_fused_store_double_kernel`).
    - `TraitFusedTransformBias` (`_gpu_fused_transform_bias_kernel`).

    `FusedViewType` is the trait's contract field -- the access-enabled
    view the rebind returns. It is `Self.device_type` on every conformer
    today, so the free function next door is the single place the
    byte-identity check lives. The day the two stop being byte-identical
    (e.g., NPU), the rebind becomes a real builder on the concrete type
    and the free function routes through it.
    """

    comptime FusedViewType: TrivialRegisterPassable


@inline(.always)
def unsafe_fused_view[
    T: UnsafeFusedView,
    *,
](t: T) -> T.FusedViewType:
    """Rebinds a fused trait argument to its access-enabled projected view.

    On `_Mogg*FusionTensor` conformers this is `Self.device_type`: the same
    fields with `_device_passable` and `_access_enabled` swapped. The host
    proxy and projected view are byte-identical today, so the rebind is just a
    bit cast; the `comptime assert` below is the guard for the day that
    property stops holding (e.g., NPU).

    Parameters:
        T: The fused trait argument type. Constrained to `UnsafeFusedView`.

    Args:
        t: The fused trait argument to rebind.

    Returns:
        The access-enabled projected view, byte-identical to `t`.
    """
    comptime assert type_of(reflect[T].field_types()) == type_of(
        reflect[T.FusedViewType].field_types()
    ), "fused view rebind: proxy and view no longer share field types"
    return rebind[T.FusedViewType](t)


# ===----------------------------------------------------------------------=== #
# Storage marker
# ===----------------------------------------------------------------------=== #


trait Stable:
    """Marker trait: the argument's data pointer lives in a stable address slot.

    The graph compiler hands the kernel the slot's address (a `StableAddr`)
    rather than the data pointer, and the argument's `Engine` is
    `StableAddrEngine`, so every `TileTensor` projected from it reads the data
    pointer out of the slot at the point of use.
    """

    pass


# ===----------------------------------------------------------------------=== #
# Metadata traits
# ===----------------------------------------------------------------------=== #


trait Tensor:
    """Metadata trait exposing a tensor's element type, known rank, and shape.
    """

    comptime dtype: DType
    comptime rank: Int

    # The shape is a `Coord` over these element types, not an opaque
    # `CoordLike`, so a generic kernel body sees a `Coord` and overloads that
    # take one (`row_major`, `coord_to_index_list`) resolve as they would on a
    # concrete tensor.
    comptime __shape_types: TypeList[Trait=CoordLike, _]._mlir_type
    comptime _shape_types: TypeList[Trait=CoordLike, Self.__shape_types]

    def shape(self) -> Coord[*Self._shape_types]:
        """Returns the tensor's runtime shape.

        Returns:
            The shape as a `Coord`, one element per dimension.
        """
        ...


# ===----------------------------------------------------------------------=== #
# Element access: marker / capability pairs
# ===----------------------------------------------------------------------=== #


trait LoadAccess(Tensor):
    """Marker trait: this tensor supports SIMD loads.

    The marker only advertises the capability; it carries no method, so a
    kernel can bind it in its signature before the argument reaches an
    execution context where loads are legal. Inside its body the kernel refines
    to `Loadable`, which supplies the `load` method.
    """

    pass


trait Loadable(LoadAccess):
    """`LoadAccess` refined with the callable `load` method."""

    def load[
        width: Int,
        element_alignment: Int = 1,
    ](self, idx: Coord) -> SIMD[Self.dtype, width]:
        """Loads a SIMD vector of elements at `idx`.

        Parameters:
            width: Number of elements to load.
            element_alignment: Assumed alignment of the load in elements.

        Args:
            idx: The element coordinate to load from.

        Returns:
            The loaded `SIMD[Self.dtype, width]` value.
        """
        ...


trait StoreAccess(Tensor):
    """Marker trait: this tensor supports SIMD stores.

    The marker only advertises the capability; it carries no method, so a
    kernel can bind it in its signature before the argument reaches an
    execution context where stores are legal. Inside its body the kernel refines
    to `Storable`, which supplies the `store` method.
    """

    pass


trait Storable(StoreAccess):
    """`StoreAccess` refined with the callable `store` method."""

    def store[
        width: Int,
        element_alignment: Int = 1,
    ](self, idx: Coord, value: SIMD[Self.dtype, width]):
        """Stores a SIMD vector of elements at `idx`.

        Parameters:
            width: Number of elements to store.
            element_alignment: Assumed alignment of the store in elements.

        Args:
            idx: The element coordinate to store to.
            value: The `SIMD[Self.dtype, width]` value to write.
        """
        ...


trait _LegacyComputeOutput(Tensor):
    """Opt-in: this output can take its fused epilogue as a compute lambda.

    Unlike `LoadAccess` and `StoreAccess`, this opt-in doesn't guarantee a
    single callable refinement. The argument arrives as either
    `_LegacyComputable` or `Storable`, depending on the fused epilogue, and the
    kernel must handle both.

    It is the trait-lane counterpart of the legacy `_FusedComputeOutputTensor`
    and follows the same rules.

    This trait and `_LegacyComputable` are slated for deprecation and will be
    removed once more principled epilogue forms are available. They couple the
    epilogue's value transform to the kernel's store, and their fallback rules
    leak into every kernel that uses them. Use them only when a kernel can't
    write its output element by element, such as a matmul that stages output
    tiles in shared memory and writes them with an asynchronous bulk copy (TMA
    on SM100, DMA on an NPU). For now, every other kernel should declare
    `StoreAccess`.

    Declare it as `Some[Output & Fused & _LegacyComputeOutput]`, then branch on
    what the graph compiler delivered:

    - `_LegacyComputable & TileTensorable`: `compute` applies the epilogue's
      value transform and returns the result without storing it. The kernel
      writes the result through `to_tile_tensor`. An output with no fused
      epilogue arrives this way too, with an identity `compute`.
    - `Storable`: `store` runs the whole epilogue, including the write, as for
      a fused `StoreAccess` output. The kernel stores in the dtype it computed
      and has no buffer to write a tile into.

    ```mojo
    comptime if conforms_to(type_of(z), _LegacyComputable):
        # Transform each value, then stage and write the tile.
        ...
    else:
        # `store` each value; it applies the whole epilogue.
        ...
    ```

    The graph compiler delivers `Storable` when a compute lambda can't express
    the epilogue, because it changes the store index (a fused reshape) or the
    dtype (a fused cast), or when compute lambdas are disabled.
    """

    pass


trait _LegacyComputable(_LegacyComputeOutput):
    """`_LegacyComputeOutput` refined with the callable `compute` method.

    Only refine to it in the kernel body, after checking `conforms_to`; the
    argument may arrive as `Storable` instead. A kernel signature must not
    declare it, and the graph compiler rejects one that does. See
    `_LegacyComputeOutput` for when each arrives and when to use it.
    """

    def compute[
        width: Int,
        element_alignment: Int = 1,
    ](self, idx: Coord, value: SIMD[Self.dtype, width]) -> SIMD[
        Self.dtype, width
    ]:
        """Applies the fused compute epilogue to a computed value at `idx`.

        Parameters:
            width: Number of elements in the value.
            element_alignment: Assumed alignment in elements.

        Args:
            idx: The element coordinate being produced.
            value: The computed `SIMD[Self.dtype, width]` value to transform.

        Returns:
            The transformed value; the parent kernel performs the final store.
        """
        ...


# ===----------------------------------------------------------------------=== #
# Executable projection
# ===----------------------------------------------------------------------=== #


trait TileTensorable(Tensor):
    """A `Tensor` backed by a dense layout that can project to a `TileTensor`
    view over its storage.

    Adds the layout type, preferred alignment, and engine on top of `Tensor`.
    Lets the kernel work on a non-fused argument directly as a `TileTensor`
    (via `to_tile_tensor`) instead of going through the fused
    load/store/transform access. Bind it in the signature when the kernel
    projects the argument on every target it supports, or refine to it in the
    body with `conforms_to` when that varies by target.

    The projection's origin is `UntrackedOrigin` because the backing memory
    belongs to the graph executor: its liveness guarantee is invisible to the
    origin system, and the generated argument has no call-site origin it could
    name instead.
    """

    comptime LayoutType: TensorLayout
    comptime alignment: Int
    comptime Engine: TensorEngine = CurrentTensorEngine[AddressSpace.GENERIC]
    """How the argument's backing memory is referenced.

    Per-target instantiation metadata like `alignment`, not part of the
    distilled contract: the graph compiler picks it per device label when it
    instantiates the argument, so a host-pointer argument and a device-pointer
    argument differ here while sharing one contract.
    """
    comptime mut: Bool

    def layout(self) -> Self.LayoutType:
        """Returns the tensor's layout (shape and strides).

        Returns:
            The argument's `LayoutType` instance.
        """
        ...

    def _to_tile_tensor(
        self,
    ) -> TileTensor[
        Self.dtype,
        LayoutType=Self.LayoutType,
        origin=UntrackedOrigin[mut=Self.mut],
        Engine=Self.Engine,
    ]:
        """Projects the argument to a `TileTensor` over its storage.

        Conformers implement this; kernels call `to_tile_tensor`. The two
        names are distinct so a call without parameters always reaches the
        method with role-aware mutability.

        Returns:
            A `TileTensor` view with the argument's dtype, layout, and `mut`.
        """
        ...

    # The role test only makes `mut` foldable through an `Output` or
    # `MutableInput` bound, so the view reaches `mut=True` parameters; it is
    # `Self.mut` in every case. It uses `|` because `or` lowers to
    # `a if a else b`, which folds only when `a` is provable. Elsewhere the
    # default stays symbolic: an input still reaches immutable parameters
    # through `TileTensor`'s implicit `Immut` conversion, and a role-less
    # writer passes `mut=True`.
    @inline(.always)
    def to_tile_tensor[
        dtype: DType = Self.dtype,
        *,
        mut: Bool = True if conforms_to(Self, Output)
        | conforms_to(Self, MutableInput) else Self.mut,
    ](
        self,
        out result: TileTensor[
            mut=mut,
            dtype,
            LayoutType=Self.LayoutType,
            origin=UntrackedOrigin[mut=mut],
            Engine=Self.Engine,
        ],
    ):
        """Projects the argument to a `TileTensor` view over its storage.

        Treat this method as final: conformers implement `_to_tile_tensor`
        and must not override `to_tile_tensor`. The role-based `mut` default
        and the dtype and mutability checks live here, so an override would
        bypass them, and Mojo cannot mark a default method final.

        Parameters:
            dtype: The argument's dtype, restated so the projection unifies
                with another argument's dtype parameter (defaults to the
                argument's own dtype). It must equal the argument's dtype; it
                does not convert or reinterpret.
            mut: Whether to project a mutable view (defaults to the
                argument's own `mut`).

        Returns:
            A `TileTensor` view over the argument's storage.

        Constraints:
            `dtype` must equal the argument's dtype, and a mutable view needs
            a mutable argument.
        """
        comptime assert dtype == Self.dtype, _dtype_mismatch_message[
            Self, dtype
        ]()
        comptime if mut:
            comptime assert Self.mut, "a mutable view needs a mutable argument"
        return rebind[type_of(result)](self._to_tile_tensor())


def _dtype_mismatch_message[TensorType: Tensor, dtype: DType]() -> String:
    return String(
        t"to_tile_tensor[{dtype}] restates the argument's dtype for the type"
        t" checker and does not convert, but '{reflect[TensorType].name()}'"
        t" has dtype {TensorType.dtype}; use TileTensor.bitcast to reinterpret"
        t" the storage"
    )
