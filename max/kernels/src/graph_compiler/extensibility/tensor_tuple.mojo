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
"""A tuple of tensor arguments, for kernels with variadic tensor arguments."""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.builtin.rebind import downcast
from std.math import align_up
from std.sys import CompilationTarget, align_of, size_of


comptime _DeviceTypeOf[T: AnyType]: AnyType = downcast[
    T, DevicePassable
].device_type if conforms_to(T, DevicePassable) else T
"""The device representation of a tuple element: its `device_type` when it is
`DevicePassable`, otherwise the element type itself."""


struct TensorTuple[*Ts: AnyType](
    Copyable where Ts.all_conforms_to[Copyable](),
    Deinitable,
    DevicePassable where Ts.all_conforms_to[DevicePassable](),
    ImplicitlyCopyable where Ts.all_conforms_to[ImplicitlyCopyable](),
    Movable where Ts.all_conforms_to[Movable](),
    RegisterPassable where Ts.all_conforms_to[RegisterPassable](),
    Sized,
):
    """A heterogeneous tuple of a kernel's tensor arguments, held by value.

    A kernel takes one `TensorTuple` per variadic tensor argument, with the
    element types in an infer-only `TypeList` bound by tensor-argument traits:

    ```mojo
    def execute[
        Outs: TypeList[Trait=Output & DenseTensor, ...],
        Ins: TypeList[Trait=Input & DenseTensor, ...],
        //,
        target: StaticString,
    ](outputs: TensorTuple[*Outs], inputs: TensorTuple[*Ins], ctx: DeviceContext):
        comptime for i in range(len(Ins)):
            var x = inputs[i]
    ```

    The struct is a `Tuple` of tensor arguments in all but two respects. It puts
    no bound of its own on the element types, so a kernel's list needs only
    its tensor-argument bound; the elements must be `Copyable` to be held
    and `Deinitable` to be dropped, and that is checked where the tuple is
    instantiated. And its storage is a plain aggregate rather than a
    parameter pack, so it crosses a kernel launch as one argument (see
    `_mlir_type`).

    Like `TileTensor`, the tuple is its own device representation, over its
    elements' device types: when every element is `DevicePassable`, so is the
    tuple, and `enqueue_function` projects it to its `device_type` by running
    each element's own `_to_device_type` in place. A device kernel taking
    `TensorTuple[*Ts].device_type` then indexes element `k` with the access its
    bound grants.

    Parameters:
        Ts: The element types.
    """

    # A plain aggregate, not a parameter pack: `isParamPack` marks a struct
    # whose elements the calling convention expands into separate arguments,
    # which would leave a launch with more device parameters than the host
    # passed. Without it the tuple crosses as one argument. A plain struct's
    # element list must be typed by `!kgen.type` rather than by the elements'
    # trait bound, as `VariadicPack` spells its own.
    comptime _element_types = rebind[
        __mlir_type[`!kgen.param_list<`, __mlir_type.`!kgen.type`, `>`]
    ](Self.Ts.values)
    comptime _mlir_type = __mlir_type[
        `!kgen.struct<:`,
        type_of(Self._element_types),
        Self._element_types,
        `>`,
    ]
    var _mlir_value: Self._mlir_type

    comptime device_type: AnyType = TensorTuple[*Self.Ts.map[_DeviceTypeOf]()]

    @inline(.always)
    def __init__(out self, *elements: *Self.Ts):
        """Constructs a tuple holding copies of `elements`.

        Args:
            elements: The tensor arguments to hold.

        Constraints:
            Every element type must be `Copyable`.
        """
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._mlir_value)
        )
        comptime for i in range(Self.Ts.length):
            comptime assert conforms_to(
                type_of(elements[i]), Copyable
            ), "a TensorTuple element must be Copyable"
            Pointer(to=self[i]).unsafe_write(copy=elements[i])

    @inline(.always)
    def __init__(
        out self, *, copy: Self
    ) where Self.Ts.all_conforms_to[Copyable]():
        """Copies a tuple element by element.

        Args:
            copy: The tuple to copy.
        """
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._mlir_value)
        )
        comptime for i in range(Self.Ts.length):
            Pointer(to=self[i]).unsafe_write(copy=copy[i])

    @inline(.always)
    def __init__(
        out self, *, deinit move: Self
    ) where Self.Ts.all_conforms_to[Movable]():
        """Moves a tuple element by element.

        Args:
            move: The tuple to move from.
        """
        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._mlir_value)
        )
        comptime for i in range(Self.Ts.length):
            Pointer(to=self[i]).unsafe_write_move_from(Pointer(to=move[i]))

    def __deinit__(deinit self):
        """Destroys each element.

        Constraints:
            Every element type must be `Deinitable`.
        """
        # Unconditional, unlike `Tuple`'s, so a kernel generic over a
        # tensor-argument bound can let its tuple go out of scope. The
        # obligation lands on the concrete elements instead.
        comptime for i in range(Self.Ts.length):
            comptime assert conforms_to(
                Self.Ts[i], Deinitable
            ), "a TensorTuple element must be Deinitable"
            Pointer(to=self[i]).unsafe_deinit_pointee()

    @inline(.always)
    def __len__(self) -> Int:
        """Returns the number of elements.

        Returns:
            The number of elements in the tuple.
        """
        return Self.Ts.length

    @inline(.always)
    def __getitem_param__[idx: Int](ref self) -> ref[self] Self.Ts[idx]:
        """Returns a reference to the element at `idx`.

        Parameters:
            idx: The element's index.

        Returns:
            A reference to the element.
        """
        var storage = Pointer(to=self._mlir_value)._get_kgen_pointer()
        var element = __mlir_op.`kgen.struct.gep`[
            index=idx.__mlir_index__(),
            _type=Pointer[Self.Ts[idx]]._mlir_type,
        ](storage)
        return Pointer[_, origin_of(self)](_mlir_value=element)[]

    @inline(.always)
    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ) where Self.Ts.all_conforms_to[DevicePassable]():
        # Each element runs its own `_to_device_type`, so a dense element's
        # pointer translates to its device address and a fused proxy flips to
        # its access-enabled view, at the element's offset in the device tuple.
        comptime for i in range(Self.Ts.length):
            comptime assert conforms_to(Self.Ts[i], DevicePassable)
            comptime offset = TensorTuple[
                *Self.Ts.map[_DeviceTypeOf]()
            ]._element_offset[i, target=encoder.target()]()
            self[i]._to_device_type(
                encoder,
                target.unsafe_bitcast[UInt8]()
                .unsafe_offset(offset)
                .unsafe_bitcast[NoneType](),
            )

    @staticmethod
    def get_type_name() -> String:
        """Returns the host type's name, for kernel-launch diagnostics.

        Returns:
            The name `TensorTuple`.
        """
        return String("TensorTuple")

    @staticmethod
    def _element_offset[idx: Int, *, target: CompilationTarget]() -> Int:
        """Returns the byte offset of element `idx` in `target`'s layout.

        The storage is a plain aggregate, so each element sits at the first
        multiple of its alignment past the previous one.
        """
        var offset = 0
        comptime for i in range(idx + 1):
            offset = align_up(offset, align_of[Self.Ts[i], target=target]())
            comptime if i < idx:
                offset += size_of[Self.Ts[i], target=target]()
        return offset
