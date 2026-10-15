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
"""Mixture of Experts (MoE) modules for DeepseekV3.2, in the ModuleV3 API.

DeepSeek changed MoE datatypes in the V3 to V3.2 upgrade: accumulations run in
float32. The reference implementations differ only in the casts:

- V3:   ``self.w2(F.silu(self.w1(x)) * self.w3(x))``
- V3.2: ``self.w2((F.silu(self.w1(x).float()) * self.w3(x).float()).type_as(x))``

Each class below is the V3 module with those accumulation dtypes set; see
``QuantizedMoE.gate_up_accum_dtype`` and ``combine_accum_dtype``.
"""

from __future__ import annotations

from typing import ClassVar

from max.dtype import DType

from ...deepseekV3_modulev3.layers.quant_moe import (
    ExpertParallelMoE,
    QuantizedMoE,
    TensorParallelMoE,
)


class DeepseekV3_2MoE(QuantizedMoE):
    """Single-device (or replicated) V3.2 MoE."""

    gate_up_accum_dtype: ClassVar[DType | None] = DType.float32
    combine_accum_dtype: ClassVar[DType | None] = DType.float32


class DeepseekV3_2TensorParallelMoE(TensorParallelMoE):
    """Tensor-parallel V3.2 MoE."""

    gate_up_accum_dtype: ClassVar[DType | None] = DType.float32
    combine_accum_dtype: ClassVar[DType | None] = DType.float32


class DeepseekV3_2ExpertParallelMoE(ExpertParallelMoE):
    """Expert-parallel V3.2 MoE.

    Neither upcast applies. V2's EP deployment runs the shared expert-parallel
    path, which takes the fused NVFP4 SwiGLU with no float32 cast, and the EP
    combine already accumulates in the dispatch dtype. A float32 gate/up
    accumulation here would rule out the fused SwiGLU and with it the
    graph compiler's MegaFFN fusion, which decode latency depends on.
    """
