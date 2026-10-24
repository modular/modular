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

"""``F.functional`` wrappers for the Hy4 kernels that ModuleV3 lacks."""

from collections.abc import Callable
from typing import Any

from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorValue
from max.nn import kernels


def _functional(op: Callable[..., Any]) -> Callable[..., Any]:
    """Wraps ``op``, handing it graph values for its tensor arguments.

    ``F.functional`` leaves a tensor passed directly for the op to convert,
    but these kernels read ``.device`` and build typed ops before converting.
    """

    def call(*args: Any, **kwargs: Any) -> Any:
        return op(
            *(TensorValue(a) if isinstance(a, Tensor) else a for a in args),
            **kwargs,
        )

    call.__name__ = op.__name__
    return F.functional(call)


mla_fp8_index_top_k = _functional(kernels.mla_fp8_index_top_k)
rope_ragged = _functional(kernels.rope_ragged)
store_k_cache_ragged = _functional(kernels.store_k_cache_ragged)
store_k_scale_cache_ragged = _functional(kernels.store_k_scale_cache_ragged)
