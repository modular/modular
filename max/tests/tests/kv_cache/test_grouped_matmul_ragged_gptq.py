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

"""CPU-safe graph construction tests for grouped GPTQ MoE matmuls."""

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.graph.quantization import QuantizationConfig
from max.nn.kernels import grouped_matmul_ragged_gptq


def _build_grouped_gptq_graph(desc_act: bool) -> None:
    hidden_dim = 128
    out_dim = 64
    num_experts = 4
    group_size = 128
    packed_rows = hidden_dim // 2 + (hidden_dim // group_size) * 2

    input_type = TensorType(
        DType.bfloat16, ["num_tokens", hidden_dim], device=DeviceRef.GPU()
    )
    weight_type = TensorType(
        DType.uint8,
        [num_experts, packed_rows, out_dim],
        device=DeviceRef.GPU(),
    )
    expert_start_indices_type = TensorType(
        DType.uint32, [num_experts + 1], device=DeviceRef.GPU()
    )
    expert_ids_type = TensorType(
        DType.int32, [num_experts], device=DeviceRef.GPU()
    )
    usage_stats_type = TensorType(DType.uint32, [2], device=DeviceRef.GPU())
    perm_idx_type = TensorType(
        DType.int32, [num_experts, hidden_dim], device=DeviceRef.GPU()
    )

    quantization_config = QuantizationConfig(
        quant_method="gptq",
        bits=4,
        group_size=group_size,
        desc_act=desc_act,
        sym=True,
    )

    input_types = [
        input_type,
        weight_type,
        expert_start_indices_type,
        expert_ids_type,
        usage_stats_type,
    ]
    if desc_act:
        input_types.append(perm_idx_type)

    with Graph(
        "grouped_gptq_moe_test", input_types=tuple(input_types)
    ) as graph:
        perm_idx = graph.inputs[5].tensor if desc_act else None
        output = grouped_matmul_ragged_gptq(
            x=graph.inputs[0].tensor,
            weight=graph.inputs[1].tensor,
            expert_start_indices=graph.inputs[2].tensor,
            expert_ids=graph.inputs[3].tensor,
            usage_stats=graph.inputs[4].tensor,
            quantization_config=quantization_config,
            perm_idx=perm_idx,
        )
        graph.output(output)


def test_grouped_matmul_ragged_gptq_builds_without_perm_idx() -> None:
    _build_grouped_gptq_graph(desc_act=False)


def test_grouped_matmul_ragged_gptq_builds_with_perm_idx() -> None:
    _build_grouped_gptq_graph(desc_act=True)
