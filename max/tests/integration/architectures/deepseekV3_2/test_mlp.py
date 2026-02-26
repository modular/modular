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
"""Tests for DeepseekV3.2 MLP layer."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn import (
    Float8Config,
    Float8InputScaleSpec,
    Float8ScaleGranularity,
    Float8ScaleOrigin,
    Float8WeightScaleSpec,
)
from max.pipelines.architectures.deepseekV3_2.layers import DeepseekV3_2MLP


def test_mlp_bfloat16() -> None:
    """Tests MLP works with bfloat16 inputs."""
    hidden_dim = 64
    feed_forward_length = 128

    mlp = DeepseekV3_2MLP(
        dtype=DType.bfloat16,
        quantization_encoding=None,
        hidden_dim=hidden_dim,
        feed_forward_length=feed_forward_length,
        devices=[DeviceRef.CPU()],
    )

    # Initialize weight names to avoid collisions.
    mlp.state_dict()

    def forward(x: TensorValue) -> TensorValue:
        return mlp(x)

    g = Graph(
        "test_mlp_bfloat16",
        forward=forward,
        input_types=[
            TensorType(DType.bfloat16, (10, hidden_dim), DeviceRef.CPU())
        ],
    )

    # Verify the graph was constructed successfully.
    assert g is not None


def test_mlp_float32() -> None:
    """Tests MLP works with float32 inputs."""
    hidden_dim = 64
    feed_forward_length = 128

    mlp = DeepseekV3_2MLP(
        dtype=DType.float32,
        quantization_encoding=None,
        hidden_dim=hidden_dim,
        feed_forward_length=feed_forward_length,
        devices=[DeviceRef.CPU()],
    )

    # Initialize weight names to avoid collisions.
    mlp.state_dict()

    def forward(x: TensorValue) -> TensorValue:
        return mlp(x)

    g = Graph(
        "test_mlp_float32",
        forward=forward,
        input_types=[
            TensorType(DType.float32, (10, hidden_dim), DeviceRef.CPU())
        ],
    )

    # Verify the graph was constructed successfully.
    assert g is not None


def test_mlp_with_float8_config() -> None:
    """Tests MLP works with Float8Config for weight quantization on GPU.

    Float8 matmul requires:
    - Weight dtype of float8_e4m3fn (not bfloat16)
    - GPU device
    - Proper weight scales in the registry
    """
    hidden_dim = 64
    feed_forward_length = 128

    float8_config = Float8Config(
        weight_scale=Float8WeightScaleSpec(
            dtype=DType.float32,
            granularity=Float8ScaleGranularity.ROWWISE,
        ),
        input_scale=Float8InputScaleSpec(
            dtype=DType.float32,
            granularity=Float8ScaleGranularity.COLWISE,
            origin=Float8ScaleOrigin.DYNAMIC,
        ),
        mlp_in_float8=set(),
        attn_qkv_in_float8=set(),
    )

    # Float8 requires float8 weight dtype, not bfloat16
    mlp = DeepseekV3_2MLP(
        dtype=DType.float8_e4m3fn,
        quantization_encoding=None,
        hidden_dim=hidden_dim,
        feed_forward_length=feed_forward_length,
        devices=[DeviceRef.CPU()],
        float8_config=float8_config,
    )

    # Initialize weight names to avoid collisions.
    mlp.state_dict()

    def forward(x: TensorValue) -> TensorValue:
        return mlp(x)

    g = Graph(
        "test_mlp_float8_config",
        forward=forward,
        input_types=[
            TensorType(DType.bfloat16, (10, hidden_dim), DeviceRef.GPU())
        ],
    )

    # Verify the graph was constructed successfully.
    assert g is not None

    # Verify the float8 config was passed to the linear layers.
    assert mlp.gate_proj.float8_config is not None
    assert mlp.down_proj.float8_config is not None
    assert mlp.up_proj.float8_config is not None
