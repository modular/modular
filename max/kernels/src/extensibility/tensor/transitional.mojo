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

"""Utilities for transitional period during NDBuffer deprecation."""

from buffer import NDBuffer
from compiler_internal.directives import StaticTensorSpec
from tensor.io_spec import IO
from tensor.managed_tensor_slice import ManagedTensorSlice


@always_inline
fn managed_tensor_slice_to_ndbuffer[
    spec: StaticTensorSpec,
    //,
](
    tensor: ManagedTensorSlice[static_spec=spec],
    out result: NDBuffer[
        spec.dtype,
        spec.rank,
        AnyOrigin[mut = not tensor.io_spec.input.value == IO.Input.value],
        spec.shape,
        spec.strides,
        # alignment2 = spec.alignment,
        address_space = spec.address_space,
        # exclusive = spec.exclusive,
    ],
):
    comptime assert not tensor.io_spec.input == IO.FusedInput
    var ptr = tensor._ptr.address_space_cast[spec.address_space]()
    return {
        ptr.mut_cast[result.mut](),
        tensor.shape(),
        tensor._runtime_strides,
    }
