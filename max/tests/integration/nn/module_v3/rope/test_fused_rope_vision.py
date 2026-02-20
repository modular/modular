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
"""Tests for fused_qk_rope_vision kernel."""

from __future__ import annotations

from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.rope import fused_qk_rope_vision
from max.tensor import Tensor


def test_fused_rope_vision_import() -> None:
    """Test that fused_qk_rope_vision is properly exported from max.nn.rope."""
    assert callable(fused_qk_rope_vision)


def test_fused_rope_vision_execution() -> None:
    """Test that the kernel can be compiled and executed via MAX graph."""

    # shapes
    batch = 1
    seq_len = 8
    num_q_heads = 2
    num_k_heads = 2
    head_dim = 64  # mult of 2

    # Q/K: [B, S, H, D]
    input_shape = (batch, seq_len, num_q_heads, head_dim)
    # Cos/Sin: [S, D]
    freq_shape = (seq_len, head_dim)

    # Select device
    device = CPU() if accelerator_count() == 0 else Accelerator()
    device_ref = DeviceRef.from_device(device)

    # Define graph with custom_extensions
    graph = Graph(
        name="fused_rope_test",
        forward=lambda q, k, cos, sin: fused_qk_rope_vision(
            q, k, cos, sin, repeat_interleave=True
        ),
        input_types=[
            TensorType(DType.bfloat16, input_shape, device=device_ref),
            TensorType(DType.bfloat16, input_shape, device=device_ref),
            TensorType(DType.float32, freq_shape, device=device_ref),
            TensorType(DType.float32, freq_shape, device=device_ref),
        ],
    )

    # Load and compile
    session = InferenceSession(devices=[device])
    model = session.load(graph)

    # Create dummy inputs
    q = Tensor.zeros(input_shape, dtype=DType.bfloat16).to(device)
    k = Tensor.zeros(input_shape, dtype=DType.bfloat16).to(device)
    cos = Tensor.ones(freq_shape, dtype=DType.float32).to(device)
    sin = Tensor.ones(freq_shape, dtype=DType.float32).to(device)

    # Execute
    results = model.execute(q, k, cos, sin)
    q_out = results[0]
    k_out = results[1]

    # Basic output check
    assert q_out.shape == input_shape
    assert k_out.shape == input_shape
