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
"""The workspace buffer a kernel requests from the graph compiler."""

from std.builtin.device_passable import DevicePassable
from std.memory import AddressSpace
from std.sys import size_of

from layout import (
    CoordLike,
    RowMajorLayout,
    TensorEngine,
    TileTensor,
    row_major,
)


trait Workspace(DevicePassable):
    """A buffer of bytes the graph compiler allocates for one call of a kernel.

    The argument takes no operand of the op. Its contents are unspecified on
    entry and are not preserved after the call returns.

    A kernel takes one workspace argument on `execute` or `execute_device`,
    after its operands, and declares a `workspace_size` static method that
    returns the size in bytes. `workspace_size` takes one `Some[Tensor]`
    argument per tensor input of the entry point, so the size depends on the
    input shapes but never on the input data:

    ```mojo
    @staticmethod
    def workspace_size(a: Some[Tensor], b: Some[Tensor]) -> Int:
        return Int(a.shape()[0].value()) * 2

    @staticmethod
    def execute_device(
        c: Some[Output & TileTensorable],
        a: Some[Input & TileTensorable],
        b: Some[Input & TileTensorable],
        workspace: Some[Workspace],
    ):
        ...
    ```

    Use `workspace_to_tile_tensor` to view the buffer as a tensor.
    """

    comptime Engine: TensorEngine
    """How the buffer's memory is referenced."""

    def storage(
        self,
    ) -> Self.Engine.StorageType[
        DType.uint8, MutUntrackedOrigin, AddressSpace.GENERIC
    ]:
        """Returns the handle to the buffer's bytes.

        Returns:
            The storage handle, as the engine references it.
        """
        ...

    def size(self) -> Int:
        """Returns the size of the buffer.

        Returns:
            The size in bytes.
        """
        ...


@inline(.always)
def workspace_to_tile_tensor[
    dtype: DType,
    *CoordTypes: CoordLike,
](ws: Some[Workspace], var *coords: *CoordTypes) -> TileTensor[
    dtype,
    LayoutType=RowMajorLayout[*CoordTypes],
    origin=MutUntrackedOrigin,
    Engine=type_of(ws).Engine,
]:
    """Views the start of a workspace buffer as a row-major tensor.

    Parameters:
        dtype: The element type of the view.
        CoordTypes: The types of the dimensions.

    Args:
        ws: The workspace buffer.
        coords: The shape of the view, one value per dimension.

    Returns:
        A mutable `TileTensor` over the first bytes of the buffer.
    """
    # `row_major` names its result type through its own parameter pack, which
    # the checker does not unify with the pack here.
    var layout = rebind[RowMajorLayout[*CoordTypes]](row_major(*coords^))
    debug_assert(
        layout.size() * size_of[Scalar[dtype]]() <= ws.size(),
        "the view is larger than the workspace buffer",
    )
    return {
        type_of(ws).Engine.unsafe_cast[
            dtype, MutUntrackedOrigin, AddressSpace.GENERIC
        ](ws.storage()),
        layout,
    }
