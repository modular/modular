# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from tensor_internal.io_spec import IO
from tensor_internal.managed_tensor_slice import ManagedTensorSlice


@always_inline
fn managed_tensor_slice_to_ndbuffer[
    spec: StaticTensorSpec, //
](tensor: ManagedTensorSlice[static_spec=spec]) -> NDBuffer[
    spec.dtype,
    spec.rank,
    MutableAnyOrigin,
    spec.shape,
    spec.strides,
    alignment = spec.alignment,
    address_space = spec.address_space,
    exclusive = spec.exclusive,
]:
    constrained[not tensor.io_spec.input == IO.FusedInput]()
    var ptr = tensor._ptr.address_space_cast[spec.address_space]()
    return NDBuffer[
        spec.dtype,
        spec.rank,
        _,
        spec.shape,
        spec.strides,
        alignment = spec.alignment,
        address_space = spec.address_space,
        exclusive = spec.exclusive,
    ](ptr, tensor.shape(), tensor._runtime_strides)
