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
"""Kimi K3's fused q/k/v short convolution (ModuleV3).

Direct V3 port of ``kimi_k3/layers/short_conv.py``'s ``kimi_short_conv_qkv``:
three depthwise causal convs (one each on q, k, v, no bias, each followed by
``silu``) fused into one call over the concatenated ``[q ; k ; v]`` channels,
reusing MAX's shared ``gated_delta_conv1d_fwd`` kernel (also used by Qwen3.5)
rather than adding a new one. Bridges ``Tensor`` to the kernel's
``TensorValue``/``BufferValue`` interface the same way
``nemotron_h_modulev3/layers/mamba2.py`` bridges its own conv/SSM pools.
"""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import BufferValue, TensorValue, ops
from max.nn.legacy.state_space import gated_delta_conv1d_fwd


def _kimi_short_conv_qkv(
    x_qkv_ragged: Tensor,
    conv_weight: Tensor,
    conv_pool: Tensor,
    conv_row_id: Tensor,
    input_row_offsets: Tensor,
) -> TensorValue:
    # The shared kernel binds one `work_dtype` across its input, weight and
    # output, and its MAX wrapper declares that output fp32 unconditionally
    # -- so the working dtype is fp32 whatever the model runs at; the pool
    # alone is free to stay narrower.
    out_dtype = TensorValue(x_qkv_ragged).dtype
    conv = gated_delta_conv1d_fwd(
        qkv_input_ragged=ops.cast(TensorValue(x_qkv_ragged), DType.float32),
        conv_weight=ops.cast(TensorValue(conv_weight), DType.float32),
        conv_state=BufferValue(conv_pool),
        slot_idx=TensorValue(conv_row_id),
        input_row_offsets=TensorValue(input_row_offsets),
    )
    return ops.cast(ops.silu(conv), out_dtype)


kimi_short_conv_qkv = F.functional(_kimi_short_conv_qkv)
