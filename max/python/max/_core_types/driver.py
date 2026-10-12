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

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DLPackArray(Protocol):
    """Protocol for objects that exchange data through the DLPack interface.

    Any array-like object implementing ``__dlpack__`` and
    ``__dlpack_device__`` satisfies this protocol, including NumPy arrays
    and PyTorch tensors. MAX APIs that consume external array data accept
    values matching this protocol.
    """

    def __dlpack__(self, *, stream: None = None) -> Any: ...

    def __dlpack_device__(self) -> Any: ...
