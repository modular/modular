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
"""Targeted smoke tests for grouped MXFP4 matmul."""

from __future__ import annotations

import numpy as np

from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import (
    grouped_dynamic_scaled_mxfp4_matmul,
    grouped_matmul_ragged,
    mxfp4_dequant,
)


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hidden_states = np.zeros((4, 32), dtype=np.float32)
    packed_weight = np.zeros((2, 16, 16), dtype=np.uint8)
    expert_start_indices = np.array([0, 2, 4], dtype=np.uint32)
    expert_ids = np.array([0, 1], dtype=np.int32)
    return hidden_states, packed_weight, expert_start_indices, expert_ids


def _constants() -> tuple[np.ndarray, list[list[list[float]]]]:
    expert_usage_stats = np.array([2, 2], dtype=np.uint32)
    scale_literal = [[[1.0] for _ in range(16)] for _ in range(2)]
    return expert_usage_stats, scale_literal


def _random_case() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(7)
    hidden_states = rng.standard_normal((6, 64), dtype=np.float32)
    packed_weight = rng.integers(
        0, 256, size=(4, 24, 32), dtype=np.uint8
    )
    # Use exact powers of two so the float8_e8m0fnu conversion is well defined.
    weight_scales = np.array(
        [[[1.0, 2.0]] * 24, [[0.5, 1.0]] * 24, [[2.0, 4.0]] * 24, [[1.0, 0.5]] * 24],
        dtype=np.float32,
    )
    expert_start_indices = np.array([0, 3, 4, 6], dtype=np.uint32)
    expert_ids = np.array([2, 0, 3], dtype=np.int32)
    expert_usage_stats = np.array([3, 3], dtype=np.uint32)
    return (
        hidden_states,
        packed_weight,
        weight_scales,
        expert_start_indices,
        expert_ids,
        expert_usage_stats,
    )


def _float8_buffer(values: np.ndarray) -> Buffer:
    tensor = Buffer(DType.float8_e8m0fnu, values.shape, CPU())
    for idx in np.ndindex(values.shape):
        tensor[idx] = values[idx]
    return tensor


def _execute(graph: Graph, session: InferenceSession, hidden_states: np.ndarray):
    compiled = session.load(graph)
    hidden_states_gpu = Buffer.from_numpy(hidden_states).to(
        compiled.input_devices[0]
    )
    return compiled.execute(hidden_states_gpu)


def test_grouped_mxfp4_reference_path_smoke(session: InferenceSession) -> None:
    device = DeviceRef.GPU(0)
    hidden_states, packed_weight, expert_start_indices, expert_ids = _inputs()
    expert_usage_stats, scale_literal = _constants()

    with Graph(
        "grouped_mxfp4_reference_smoke",
        input_types=[TensorType(DType.float32, (4, 32), device=device)],
    ) as graph:
        hidden_states_t = ops.cast(graph.inputs[0].tensor, DType.bfloat16)
        packed_weight_t = ops.constant(
            packed_weight, dtype=DType.uint8, device=device
        )
        scales_t = ops.constant(
            scale_literal, dtype=DType.float8_e8m0fnu, device=device
        )
        expert_start_indices_t = ops.constant(
            expert_start_indices, dtype=DType.uint32, device=device
        )
        expert_ids_t = ops.constant(
            expert_ids, dtype=DType.int32, device=device
        )
        expert_usage_stats_t = ops.constant(
            expert_usage_stats, dtype=DType.uint32, device=DeviceRef.CPU()
        )

        reference = grouped_matmul_ragged(
            hidden_states_t,
            mxfp4_dequant(packed_weight_t, scales_t, out_type=DType.bfloat16),
            expert_start_indices_t,
            expert_ids_t,
            expert_usage_stats_t,
        )
        graph.output(ops.cast(reference, DType.float32))

    (reference_out,) = _execute(graph, session, hidden_states)
    np.testing.assert_equal(reference_out.to_numpy(), 0.0)


def test_grouped_mxfp4_fused_path_smoke(session: InferenceSession) -> None:
    device = DeviceRef.GPU(0)
    hidden_states, packed_weight, expert_start_indices, expert_ids = _inputs()
    expert_usage_stats, scale_literal = _constants()

    with Graph(
        "grouped_mxfp4_fused_smoke",
        input_types=[TensorType(DType.float32, (4, 32), device=device)],
    ) as graph:
        hidden_states_t = ops.cast(graph.inputs[0].tensor, DType.bfloat16)
        packed_weight_t = ops.constant(
            packed_weight, dtype=DType.uint8, device=device
        )
        scales_t = ops.constant(
            scale_literal, dtype=DType.float8_e8m0fnu, device=device
        )
        expert_start_indices_t = ops.constant(
            expert_start_indices, dtype=DType.uint32, device=device
        )
        expert_ids_t = ops.constant(
            expert_ids, dtype=DType.int32, device=device
        )
        expert_usage_stats_t = ops.constant(
            expert_usage_stats, dtype=DType.uint32, device=DeviceRef.CPU()
        )

        fused = grouped_dynamic_scaled_mxfp4_matmul(
            hidden_states_t,
            packed_weight_t,
            scales_t,
            expert_start_indices_t,
            expert_ids_t,
            expert_usage_stats_t,
        )
        graph.output(ops.cast(fused, DType.float32))

    (fused_out,) = _execute(graph, session, hidden_states)
    np.testing.assert_equal(fused_out.to_numpy(), 0.0)


def test_grouped_mxfp4_fused_matches_reference(
    session: InferenceSession,
) -> None:
    device = DeviceRef.GPU(0)
    (
        hidden_states,
        packed_weight,
        weight_scales,
        expert_start_indices,
        expert_ids,
        expert_usage_stats,
    ) = _random_case()
    active_packed_weight = packed_weight[expert_ids]
    active_weight_scales = weight_scales[expert_ids]
    dense_expert_ids = np.arange(expert_ids.shape[0], dtype=np.int32)

    with Graph(
        "grouped_mxfp4_fused_matches_reference",
        input_types=[TensorType(DType.float32, (6, 64), device=device)],
    ) as graph:
        hidden_states_t = ops.cast(graph.inputs[0].tensor, DType.bfloat16)
        active_packed_weight_t = ops.constant(
            active_packed_weight, dtype=DType.uint8, device=device
        )
        active_weight_scales_t = ops.constant(
            _float8_buffer(active_weight_scales),
            dtype=DType.float8_e8m0fnu,
            device=device,
        )
        dense_expert_ids_t = ops.constant(
            dense_expert_ids, dtype=DType.int32, device=device
        )
        packed_weight_t = ops.constant(
            packed_weight, dtype=DType.uint8, device=device
        )
        weight_scales_t = ops.constant(
            _float8_buffer(weight_scales),
            dtype=DType.float8_e8m0fnu,
            device=device,
        )
        expert_start_indices_t = ops.constant(
            expert_start_indices, dtype=DType.uint32, device=device
        )
        expert_ids_t = ops.constant(
            expert_ids, dtype=DType.int32, device=device
        )
        expert_usage_stats_t = ops.constant(
            expert_usage_stats, dtype=DType.uint32, device=DeviceRef.CPU()
        )

        reference = grouped_matmul_ragged(
            hidden_states_t,
            mxfp4_dequant(
                active_packed_weight_t,
                active_weight_scales_t,
                out_type=DType.bfloat16,
            ),
            expert_start_indices_t,
            dense_expert_ids_t,
            expert_usage_stats_t,
        )
        fused = grouped_dynamic_scaled_mxfp4_matmul(
            hidden_states_t,
            packed_weight_t,
            weight_scales_t,
            expert_start_indices_t,
            expert_ids_t,
            expert_usage_stats_t,
        )
        graph.output(
            ops.cast(reference, DType.float32),
            ops.cast(fused, DType.float32),
        )

    reference_out, fused_out = _execute(graph, session, hidden_states)
    reference_np = reference_out.to_numpy()
    fused_np = fused_out.to_numpy()
    max_abs_diff = np.max(np.abs(reference_np - fused_np))
    np.testing.assert_allclose(
        fused_np,
        reference_np,
        atol=0.0,
        rtol=0.0,
        err_msg=f"max_abs_diff={max_abs_diff}",
    )
