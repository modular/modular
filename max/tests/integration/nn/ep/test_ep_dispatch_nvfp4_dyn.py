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
"""NVFP4 EP dispatch with a dynamic global scale per token, end to end.

Both fused dispatch ops are driven through the batch manager, so this covers
the registrations, the buffer sizing in `ep.init`, and the position of
`output_rowwise_scales` in the outputs, as well as the numerics.
"""

from __future__ import annotations

import os

import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn.comm.ep import EPBatchManager, EPCommInitializer, EPConfig
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from test_common.graph_utils import is_b100_b200

N_DEVICES = 2
HIDDEN_DIM = 2048
NUM_EXPERTS = 64
TOP_K = 8
MAX_TOKENS_PER_RANK = 64
N_LOCAL_EXPERTS = NUM_EXPERTS // N_DEVICES

# E2M1 max times E4M3 max: the global scale maps a token's max onto it, and
# the dispatch stores that scale's inverse.
_GLOBAL_SCALE_NUMERATOR = 2688.0
# From `linalg/fp4_utils.mojo`.
_SF_ATOM_M = (32, 4)
_SF_ATOM_K = 4
_SF_MN_GROUP_SIZE = _SF_ATOM_M[0] * _SF_ATOM_M[1]
_NVFP4_SF_VECTOR_SIZE = 16
_E2M1_TO_FLOAT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    + [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def _nvfp4_config() -> QuantConfig:
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.STATIC,
            dtype=DType.float32,
            block_size=(1, 16),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e4m3fn,
            block_size=(1, 16),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        embedding_output_dtype=None,
        format=QuantFormat.NVFP4,
    )


def _varied_magnitude_tokens(n_tokens: int) -> torch.Tensor:
    """Tokens spread over 32 binades, with some zeroed out, so that no single
    static scale could quantize them all."""
    tokens = torch.randn(n_tokens, HIDDEN_DIM, dtype=torch.float32)
    exponents = 4 * (torch.arange(n_tokens) % 9) - 16
    factors = torch.pow(2.0, exponents.float())
    factors[torch.arange(n_tokens) % 17 == 0] = 0.0
    return (tokens * factors[:, None]).to(torch.bfloat16)


def _dequantize_rows(
    tokens: torch.Tensor,
    scales: torch.Tensor,
    rowwise_scales: torch.Tensor,
    rows: torch.Tensor,
    scales_block_base: int,
) -> torch.Tensor:
    """Dequantizes one expert's received rows.

    `rows[j]` is the `j`-th row of the expert, whose block scales sit in the
    5D scale-factor layout starting at scale block `scales_block_base`.
    """
    packed = tokens[rows].long()
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(1)
    values = _E2M1_TO_FLOAT[codes]

    j = torch.arange(rows.numel())[:, None]
    col = torch.arange(HIDDEN_DIM)[None, :]
    block_scales = scales[
        scales_block_base + j // _SF_MN_GROUP_SIZE,
        col // (_NVFP4_SF_VECTOR_SIZE * _SF_ATOM_K),
        j % _SF_ATOM_M[0],
        (j % _SF_MN_GROUP_SIZE) // _SF_ATOM_M[0],
        (col // _NVFP4_SF_VECTOR_SIZE) % _SF_ATOM_K,
    ]
    return values * block_scales * rowwise_scales[rows][:, None]


def _check_rows(
    tokens: torch.Tensor,
    scales: torch.Tensor,
    rowwise_scales: torch.Tensor,
    rows: torch.Tensor,
    scales_block_base: int,
    sources: torch.Tensor,
    what: str,
) -> None:
    sources = sources.float()
    row_max = sources.abs().amax(dim=1)

    expected_scales = torch.where(
        row_max > 0,
        (row_max / _GLOBAL_SCALE_NUMERATOR).to(torch.bfloat16).float(),
        1.0,
    )
    # Within one BF16 rounding of the float32 quotient.
    torch.testing.assert_close(
        rowwise_scales[rows],
        expected_scales,
        rtol=1e-2,
        atol=0.0,
        msg=f"{what}: inverse global scale mismatch",
    )

    dequantized = _dequantize_rows(
        tokens, scales, rowwise_scales, rows, scales_block_base
    )
    # Relative to each row's own max, since the rows span 32 binades. A zero
    # row gets no absolute slack, so its payload must dequantize to zero.
    tolerance = torch.maximum(
        0.25 * torch.maximum(dequantized.abs(), sources.abs()),
        0.0625 * row_max[:, None],
    )
    bad = (dequantized - sources).abs() > tolerance
    assert not bad.any(), (
        f"{what}: {int(bad.sum())} dequantized values out of tolerance, first"
        f" at {bad.nonzero()[0].tolist()}"
    )


def _verify_device(
    device_idx: int,
    outputs: list[torch.Tensor],
    inputs: list[torch.Tensor],
    topk_ids: list[torch.Tensor],
    fused_shared_expert: bool,
) -> None:
    (
        tokens,
        scales,
        rowwise_scales,
        row_offsets,
        scales_offsets,
        expert_ids,
        src_info,
    ) = outputs
    shared_offset = 1 if fused_shared_expert else 0
    n_groups = N_LOCAL_EXPERTS + shared_offset
    assert expert_ids.tolist() == list(range(n_groups))

    for group in range(n_groups):
        start, end = int(row_offsets[group]), int(row_offsets[group + 1])
        scales_block_base = start // _SF_MN_GROUP_SIZE + int(
            scales_offsets[group]
        )
        if group < shared_offset:
            # The shared expert's rows are this device's own tokens, in order.
            assert end - start == inputs[device_idx].shape[0]
            _check_rows(
                tokens,
                scales,
                rowwise_scales,
                torch.arange(start, end),
                scales_block_base,
                inputs[device_idx],
                f"device {device_idx} shared expert",
            )
            continue

        # A routed expert's rows are grouped by source rank, in rank order.
        expert = device_idx * N_LOCAL_EXPERTS + group - shared_offset
        rows_from_rank_start = start
        sources = []
        for rank in range(N_DEVICES):
            n_rows = int((topk_ids[rank] == expert).sum())
            for row in range(
                rows_from_rank_start, rows_from_rank_start + n_rows
            ):
                token, slot = src_info[row].tolist()
                assert topk_ids[rank][token, slot] == expert
                sources.append(inputs[rank][token])
            rows_from_rank_start += n_rows
        assert rows_from_rank_start == end
        if end > start:
            _check_rows(
                tokens,
                scales,
                rowwise_scales,
                torch.arange(start, end),
                scales_block_base,
                torch.stack(sources),
                f"device {device_idx} expert {expert}",
            )


@pytest.mark.skipif(
    accelerator_api() == "hip", reason="NVFP4 dispatch is NVIDIA-only"
)
@pytest.mark.skipif(
    not is_b100_b200(), reason="NVFP4 dispatch requires B100 or B200"
)
@pytest.mark.parametrize(
    "distributed, fused_shared_expert",
    [(False, False), (True, True)],
    ids=["per-device-op", "multi-device-op-shared-expert"],
)
def test_ep_dispatch_nvfp4_dyn_global_scales(
    distributed: bool, fused_shared_expert: bool
) -> None:
    assert N_DEVICES <= accelerator_count(), (
        "Devices are not enough to run EP test"
    )

    devices = [Accelerator(i) for i in range(N_DEVICES)]
    session = InferenceSession(devices=devices)

    config = EPConfig(
        dispatch_dtype=DType.uint8,
        combine_dtype=DType.bfloat16,
        hidden_size=HIDDEN_DIM,
        top_k=TOP_K,
        n_experts=NUM_EXPERTS,
        max_tokens_per_rank=MAX_TOKENS_PER_RANK,
        n_gpus_per_node=N_DEVICES,
        n_nodes=int(os.environ.get("SHMEM_TOTAL_NODES", "1")),
        dispatch_quant_config=_nvfp4_config(),
        fused_shared_expert=fused_shared_expert,
        nvfp4_dyn_global_scales=True,
    )
    ep_comm_init = EPCommInitializer(config)
    ep_comm_init.ep_init(session)
    ep_batch_manager = EPBatchManager(config)

    torch.manual_seed(0)
    input_lengths = [MAX_TOKENS_PER_RANK, MAX_TOKENS_PER_RANK // 2 + 3]
    inputs = [_varied_magnitude_tokens(n) for n in input_lengths]
    topk_ids = [
        torch.topk(torch.randn(n, NUM_EXPERTS), TOP_K, dim=1).indices.to(
            torch.int32
        )
        for n in input_lengths
    ]

    input_types = [
        TensorType(
            DType.bfloat16, (f"input_len_{i}", HIDDEN_DIM), DeviceRef.GPU(i)
        )
        for i in range(N_DEVICES)
    ] + [
        TensorType(DType.int32, (f"input_len_{i}", TOP_K), DeviceRef.GPU(i))
        for i in range(N_DEVICES)
    ]
    with Graph(
        "ep_dispatch_nvfp4_dyn",
        input_types=[*input_types, *ep_batch_manager.input_types()],
    ) as graph:
        xs = [v.tensor for v in graph.inputs[:N_DEVICES]]
        ids = [v.tensor for v in graph.inputs[N_DEVICES : 2 * N_DEVICES]]
        ep_batch_manager.fetch_buffers(graph.inputs[2 * N_DEVICES :])

        device_ids = list(range(N_DEVICES))
        dispatched: list[tuple[TensorValue, ...]]
        if distributed:
            dispatched = ep_batch_manager.ep_dispatch_all(xs, ids, device_ids)
        else:
            dispatched = [
                ep_batch_manager.ep_dispatch(xs[i], ids[i], i)
                for i in device_ids
            ]

        outputs: list[TensorValue] = []
        for i, device_results in enumerate(dispatched):
            # Everything but the trailing grouped-matmul metadata.
            *dispatch_outputs, _ = device_results
            assert len(dispatch_outputs) == 6
            assert dispatch_outputs[2].dtype == DType.bfloat16
            src_info = ep_batch_manager._src_info[i]
            assert src_info is not None
            outputs.extend([*dispatch_outputs, src_info])
        graph.output(*outputs)

    compiled = session.load(graph)
    results = compiled.execute(
        *[Buffer.from_dlpack(x).to(devices[i]) for i, x in enumerate(inputs)],
        *[
            Buffer.from_dlpack(ids).to(devices[i])
            for i, ids in enumerate(topk_ids)
        ],
        *ep_comm_init.model_inputs(),
    )

    n_outputs = 7
    for i in range(N_DEVICES):
        buffers = results[i * n_outputs : (i + 1) * n_outputs]
        device_outputs = [
            torch.from_dlpack(b.view(DType.uint8) if j == 1 else b).cpu()
            for j, b in enumerate(buffers)
        ]
        # The block scales come back as raw E4M3 bytes; the rowwise scales
        # are compared in float32.
        device_outputs[1] = device_outputs[1].view(torch.float8_e4m3fn).float()
        device_outputs[2] = device_outputs[2].float()
        _verify_device(i, device_outputs, inputs, topk_ids, fused_shared_expert)
