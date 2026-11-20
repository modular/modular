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
"""Helpers for projecting dense tensor arguments to `TileTensor` views.

The `to_tile_tensor` free functions are deprecated: call the
`TileTensorable.to_tile_tensor` method instead.
"""

from layout import TileTensor
from std.builtin.rebind import downcast
from std.reflection import reflect

from .tensor_arg_traits import (
    MutableInput,
    Output,
    Tensor,
    TileTensorable,
)


def _not_tile_tensorable_message[TensorType: AnyType]() -> String:
    return String(
        t"to_tile_tensor requires a TileTensorable dense argument, but"
        t" '{reflect[TensorType].name()}' is not"
    )


def _require_tile_tensorable[TensorType: AnyType]() -> Bool:
    comptime assert conforms_to(
        TensorType, TileTensorable
    ), _not_tile_tensorable_message[TensorType]()
    return True


# `_require_tile_tensorable` returns `True` or stops compilation with its
# message, so the `Never` arm is unreachable. The arm makes `downcast` depend on
# the check. Thus the check runs first, and its message replaces the generic
# witness-table error that `downcast` gives for a type that does not conform.
comptime _AsTileTensorable[TensorType: Tensor] = downcast[
    TensorType if _require_tile_tensorable[TensorType]() else Never,
    TileTensorable,
]


comptime _TileProjection[mut: Bool, TensorType: Tensor]: AnyType = TileTensor[
    mut=mut,
    TensorType.dtype,
    LayoutType=_AsTileTensorable[TensorType].LayoutType,
    origin=UntrackedOrigin[mut=mut],
    Engine=_AsTileTensorable[TensorType].Engine,
]


@inline(.always)
def to_tile_tensor[
    TensorType: Tensor, //, *, mut: Bool = False
](tensor: TensorType) -> _TileProjection[mut, TensorType]:
    """Projects a tensor argument to a `TileTensor` view.

    Deprecated: call `tensor.to_tile_tensor()`, the `TileTensorable` method,
    instead. Refine a `Tensor`-bound argument with `conforms_to` first.

    Parameters:
        TensorType: The tensor-argument type (inferred).
        mut: Whether to project a mutable view (defaults to immutable).

    Args:
        tensor: The tensor argument to project.

    Returns:
        A `TileTensor` view over the argument's storage.

    Constraints:
        `TensorType` must conform to `TileTensorable`.
    """
    # The return type also does this check. The elaboration order of the
    # caller decides which check fails first, so both give the same message.
    comptime assert conforms_to(
        TensorType, TileTensorable
    ), _not_tile_tensorable_message[TensorType]()
    comptime if mut:
        comptime assert TensorType.mut
    return rebind[_TileProjection[mut, TensorType]](
        tensor.to_tile_tensor[mut=mut]()
    )


@inline(.always)
def to_tile_tensor[
    TensorType: Output & Tensor, //
](tensor: TensorType) -> _TileProjection[True, TensorType]:
    """Projects an output tensor argument to a mutable `TileTensor` view.

    Deprecated: call `tensor.to_tile_tensor()`, the `TileTensorable` method,
    instead. Refine a `Tensor`-bound argument with `conforms_to` first.

    Output tensors are writable by construction, so the trait bound selects the
    mutable projection without call sites passing `mut=True`.

    Parameters:
        TensorType: The output tensor-argument type (inferred).

    Args:
        tensor: The output tensor argument to project.

    Returns:
        A mutable `TileTensor` view over the argument's storage.

    Constraints:
        `TensorType` must conform to `TileTensorable`.
    """
    comptime assert conforms_to(
        TensorType, TileTensorable
    ), _not_tile_tensorable_message[TensorType]()
    comptime assert TensorType.mut
    return rebind[_TileProjection[True, TensorType]](
        tensor.to_tile_tensor[mut=True]()
    )


@inline(.always)
def to_tile_tensor[
    TensorType: MutableInput & Tensor, //
](tensor: TensorType) -> _TileProjection[True, TensorType]:
    """Projects a mutable-input tensor argument to a mutable `TileTensor` view.

    Deprecated: call `tensor.to_tile_tensor()`, the `TileTensorable` method,
    instead. Refine a `Tensor`-bound argument with `conforms_to` first.

    Mutable inputs are read-write by construction, so the trait bound selects
    the mutable projection without call sites passing `mut=True`.

    Parameters:
        TensorType: The mutable-input tensor-argument type (inferred).

    Args:
        tensor: The mutable-input tensor argument to project.

    Returns:
        A mutable `TileTensor` view over the argument's storage.

    Constraints:
        `TensorType` must conform to `TileTensorable`.
    """
    comptime assert conforms_to(
        TensorType, TileTensorable
    ), _not_tile_tensorable_message[TensorType]()
    comptime assert TensorType.mut
    return rebind[_TileProjection[True, TensorType]](
        tensor.to_tile_tensor[mut=True]()
    )
