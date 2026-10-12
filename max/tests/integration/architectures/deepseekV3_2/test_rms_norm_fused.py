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
"""Tests for RMSNormFused layer interface."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.pipelines.architectures.deepseekV3_2.layers import RMSNormFused


def test_rms_norm_fused_without_residual() -> None:
    """Tests RMSNormFused interface works without residual."""
    dim = 64
    norm = RMSNormFused(dim=dim, eps=1e-6)

    def forward(x: TensorValue) -> TensorValue:
        result = norm(x)
        assert isinstance(result, TensorValue)
        return result

    g = Graph(
        "test_without_residual",
        forward=forward,
        input_types=[TensorType(DType.bfloat16, (2, 10, dim), DeviceRef.CPU())],
    )

    # Verify the graph was constructed successfully
    assert g is not None


def test_rms_norm_fused_with_residual() -> None:
    """Tests RMSNormFused interface works with residual."""
    dim = 64
    norm = RMSNormFused(dim=dim, eps=1e-6)

    def forward(
        x: TensorValue, residual: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        result = norm(x, residual)
        assert isinstance(result, tuple)
        return result

    g = Graph(
        "test_with_residual",
        forward=forward,
        input_types=[
            TensorType(DType.bfloat16, (2, 10, dim), DeviceRef.CPU()),
            TensorType(DType.bfloat16, (2, 10, dim), DeviceRef.CPU()),
        ],
    )

    # Verify the graph was constructed successfully
    assert g is not None
