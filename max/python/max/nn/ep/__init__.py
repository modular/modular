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
"""Expert parallelism over a device mesh.

Plain functions over mesh-sharded tensors. The exchange's buffers are
passed explicitly as one :class:`EPBuffers` argument, which a decoder
receives ready from the framework, and no call keeps state for a later one:
:func:`dispatch` returns the :class:`Route` that :func:`combine` takes.
:func:`mega_ffn_fused` and :func:`mega_ffn_combine_send` fuse the routed
experts' FFN into the exchange. :mod:`max.nn.ep.init` is the host side the
framework runs.
"""

from max.nn.ep_config import EPConfig

from .buffers import EPBuffers, check_supported
from .exchange import (
    Grouped,
    Route,
    combine,
    combine_finish,
    combine_local,
    combine_start,
    dispatch,
    dispatch_finish,
    dispatch_local,
    dispatch_start,
    grouped_matmul_bounds,
)
from .mega_ffn import mega_ffn_combine_send, mega_ffn_fused

__all__ = [
    "EPBuffers",
    "EPConfig",
    "Grouped",
    "Route",
    "check_supported",
    "combine",
    "combine_finish",
    "combine_local",
    "combine_start",
    "dispatch",
    "dispatch_finish",
    "dispatch_local",
    "dispatch_start",
    "grouped_matmul_bounds",
    "mega_ffn_combine_send",
    "mega_ffn_fused",
]
