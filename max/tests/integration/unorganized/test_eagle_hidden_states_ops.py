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

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.pipelines.lib.bfloat16_utils import float32_to_bfloat16_as_uint16
from max.pipelines.lib.speculative_decoding import (
    accepted_hidden_states_extractor,
    call_per_device_graph,
    compute_extractor_inputs,
    compute_filter_indices,
    filter_hidden_states,
)


@pytest.fixture
def session() -> InferenceSession:
    return InferenceSession(devices=[Accelerator()])


@pytest.fixture
def multi_gpu_session() -> InferenceSession:
    return InferenceSession(devices=[Accelerator(id=0), Accelerator(id=1)])


def test_hidden_states_extractor(session: InferenceSession) -> None:
    device = DeviceRef.GPU()
    model = session.load(accepted_hidden_states_extractor([device]))

    hidden_dim = 64
    hidden_states = torch.randn(12, hidden_dim, dtype=torch.bfloat16).cuda()
    hidden_states[0] = 1.0
    hidden_states[1] = 2.0
    hidden_states[4] = 10.0
    hidden_states[5] = 20.0
    hidden_states[6] = 30.0
    hidden_states[7] = 40.0
    hidden_states[9] = 100.0

    logit_offsets = torch.tensor([0, 4, 9, 12], dtype=torch.int64).cuda()
    first_rejected_tokens_np = np.array([1, 3, 0], dtype=np.int64)
    total_range, output_offsets = compute_extractor_inputs(
        first_rejected_tokens_np
    )

    (result,) = model(
        Buffer.from_dlpack(hidden_states),
        Buffer.from_dlpack(logit_offsets),
        Buffer.from_dlpack(torch.from_numpy(total_range).cuda()),
        Buffer.from_dlpack(torch.from_numpy(output_offsets).cuda()),
    )

    expected = torch.cat(
        [hidden_states[0:2], hidden_states[4:8], hidden_states[9:10]], dim=0
    )
    torch.testing.assert_close(
        torch.from_dlpack(result), expected, rtol=1e-2, atol=1e-3
    )


def test_hidden_states_filter(session: InferenceSession) -> None:
    device = DeviceRef.GPU()
    model = session.load(filter_hidden_states([device]))
    hidden_dim = 64

    hidden_states = torch.randn(3, hidden_dim, dtype=torch.bfloat16).cuda()
    keep_indices, _ = compute_filter_indices(
        np.array([0, 0, 0], dtype=np.int64), [0, 1, 2]
    )
    (result,) = model(
        Buffer.from_dlpack(hidden_states),
        Buffer.from_dlpack(torch.from_numpy(keep_indices).cuda()),
    )
    torch.testing.assert_close(
        torch.from_dlpack(result), hidden_states, rtol=1e-2, atol=1e-3
    )

    hidden_states = torch.randn(6, hidden_dim, dtype=torch.bfloat16).cuda()
    keep_indices, _ = compute_filter_indices(
        np.array([1, 1, 1], dtype=np.int64), [0, 2]
    )
    (result,) = model(
        Buffer.from_dlpack(hidden_states),
        Buffer.from_dlpack(torch.from_numpy(keep_indices).cuda()),
    )
    expected = torch.cat([hidden_states[0:2], hidden_states[4:6]], dim=0)
    torch.testing.assert_close(
        torch.from_dlpack(result), expected, rtol=1e-2, atol=1e-3
    )

    first_rejected_np = np.array([2, 1, 3], dtype=np.int64)
    total_tokens = int(np.sum(first_rejected_np + 1))
    hidden_states = torch.randn(
        total_tokens, hidden_dim, dtype=torch.bfloat16
    ).cuda()
    keep_indices, _ = compute_filter_indices(first_rejected_np, [0, 2])
    (result,) = model(
        Buffer.from_dlpack(hidden_states),
        Buffer.from_dlpack(torch.from_numpy(keep_indices).cuda()),
    )
    expected = torch.cat([hidden_states[0:3], hidden_states[5:9]], dim=0)
    torch.testing.assert_close(
        torch.from_dlpack(result), expected, rtol=1e-2, atol=1e-3
    )


def test_multi_device_extractor(multi_gpu_session: InferenceSession) -> None:
    devices = [DeviceRef.GPU(id=0), DeviceRef.GPU(id=1)]
    extractors = [
        multi_gpu_session.load(accepted_hidden_states_extractor([dev]))
        for dev in devices
    ]
    hidden_dim = 64

    hs0_np = np.random.randn(9, hidden_dim).astype(np.float32)
    hs0_np[0], hs0_np[1], hs0_np[4], hs0_np[5], hs0_np[6] = (
        1.0,
        2.0,
        10.0,
        20.0,
        30.0,
    )
    hs0_bf16 = float32_to_bfloat16_as_uint16(hs0_np)
    hs0 = Buffer.from_numpy(hs0_bf16).view(DType.bfloat16).to(Accelerator(id=0))
    logit_offsets0 = np.array([0, 4, 9], dtype=np.int64)
    first_rejected0 = np.array([1, 2], dtype=np.int64)

    hs1_np = np.random.randn(7, hidden_dim).astype(np.float32)
    hs1_np[0], hs1_np[3] = 100.0, 200.0
    hs1_bf16 = float32_to_bfloat16_as_uint16(hs1_np)
    hs1 = Buffer.from_numpy(hs1_bf16).view(DType.bfloat16).to(Accelerator(id=1))
    logit_offsets1 = np.array([0, 3, 7], dtype=np.int64)
    first_rejected1 = np.array([0, 3], dtype=np.int64)

    total_range0, output_offsets0 = compute_extractor_inputs(first_rejected0)
    total_range1, output_offsets1 = compute_extractor_inputs(first_rejected1)

    (result0,) = extractors[0](
        hs0,
        Buffer.from_numpy(logit_offsets0).to(Accelerator(id=0)),
        Buffer.from_numpy(total_range0).to(Accelerator(id=0)),
        Buffer.from_numpy(output_offsets0).to(Accelerator(id=0)),
    )
    (result1,) = extractors[1](
        hs1,
        Buffer.from_numpy(logit_offsets1).to(Accelerator(id=1)),
        Buffer.from_numpy(total_range1).to(Accelerator(id=1)),
        Buffer.from_numpy(output_offsets1).to(Accelerator(id=1)),
    )

    expected0_np = np.concatenate([hs0_np[0:2], hs0_np[4:7]], axis=0)
    expected0 = torch.from_numpy(
        float32_to_bfloat16_as_uint16(expected0_np)
    ).view(torch.bfloat16)
    torch.testing.assert_close(
        torch.from_dlpack(result0).cpu(), expected0.cpu(), rtol=1e-2, atol=1e-3
    )

    expected1_np = np.concatenate([hs1_np[0:1], hs1_np[3:7]], axis=0)
    expected1 = torch.from_numpy(
        float32_to_bfloat16_as_uint16(expected1_np)
    ).view(torch.bfloat16)
    torch.testing.assert_close(
        torch.from_dlpack(result1).cpu(), expected1.cpu(), rtol=1e-2, atol=1e-3
    )


def test_multi_device_filter(multi_gpu_session: InferenceSession) -> None:
    devices = [DeviceRef.GPU(id=0), DeviceRef.GPU(id=1)]
    filters = [
        multi_gpu_session.load(filter_hidden_states([dev])) for dev in devices
    ]
    hidden_dim = 64

    hs0_np = np.random.randn(5, hidden_dim).astype(np.float32)
    hs0_bf16 = float32_to_bfloat16_as_uint16(hs0_np)
    hs0 = Buffer.from_numpy(hs0_bf16).view(DType.bfloat16).to(Accelerator(id=0))
    keep_indices0, _ = compute_filter_indices(
        np.array([1, 2], dtype=np.int64), [1]
    )

    hs1_np = np.random.randn(4, hidden_dim).astype(np.float32)
    hs1_bf16 = float32_to_bfloat16_as_uint16(hs1_np)
    hs1 = Buffer.from_numpy(hs1_bf16).view(DType.bfloat16).to(Accelerator(id=1))
    keep_indices1, _ = compute_filter_indices(
        np.array([1, 1], dtype=np.int64), [0, 1]
    )

    (result0,) = filters[0](
        hs0, Buffer.from_numpy(keep_indices0).to(Accelerator(id=0))
    )
    (result1,) = filters[1](
        hs1, Buffer.from_numpy(keep_indices1).to(Accelerator(id=1))
    )

    expected0 = torch.from_numpy(
        float32_to_bfloat16_as_uint16(hs0_np[2:5])
    ).view(torch.bfloat16)
    torch.testing.assert_close(
        torch.from_dlpack(result0).cpu(), expected0.cpu(), rtol=1e-2, atol=1e-3
    )

    expected1 = torch.from_numpy(hs1_bf16).view(torch.bfloat16)
    torch.testing.assert_close(
        torch.from_dlpack(result1).cpu(), expected1.cpu(), rtol=1e-2, atol=1e-3
    )


def test_call_per_device_graph_with_empty_inputs(
    multi_gpu_session: InferenceSession,
) -> None:
    devices = [DeviceRef.GPU(id=0), DeviceRef.GPU(id=1)]
    filters = [
        multi_gpu_session.load(filter_hidden_states([dev])) for dev in devices
    ]
    hidden_dim = 64

    hs0_np = np.random.randn(3, hidden_dim).astype(np.float32)
    hs0 = (
        Buffer.from_numpy(float32_to_bfloat16_as_uint16(hs0_np))
        .view(DType.bfloat16)
        .to(Accelerator(id=0))
    )
    hs1_np = np.random.randn(4, hidden_dim).astype(np.float32)
    hs1 = (
        Buffer.from_numpy(float32_to_bfloat16_as_uint16(hs1_np))
        .view(DType.bfloat16)
        .to(Accelerator(id=1))
    )

    per_device_inputs = [
        [
            hs0,
            Buffer.from_numpy(np.array([0, 2], dtype=np.int64)).to(
                Accelerator(id=0)
            ),
        ],
        [
            hs1,
            Buffer.from_numpy(np.array([1, 3], dtype=np.int64)).to(
                Accelerator(id=1)
            ),
        ],
    ]
    results = call_per_device_graph(filters, per_device_inputs)
    assert len(results) == 2
    assert results[0].shape[0] == 2
    assert results[1].shape[0] == 2

    hs1_empty = (
        Buffer.from_numpy(
            float32_to_bfloat16_as_uint16(
                np.zeros((0, hidden_dim), dtype=np.float32)
            )
        )
        .view(DType.bfloat16)
        .to(Accelerator(id=1))
    )
    per_device_inputs = [
        [
            hs0,
            Buffer.from_numpy(np.array([0, 1], dtype=np.int64)).to(
                Accelerator(id=0)
            ),
        ],
        [
            hs1_empty,
            Buffer.from_numpy(np.array([], dtype=np.int64)).to(
                Accelerator(id=1)
            ),
        ],
    ]
    results = call_per_device_graph(filters, per_device_inputs)
    assert len(results) == 2
    assert results[0].shape[0] == 2
    assert results[1].shape[0] == 0

    hs0_empty = (
        Buffer.from_numpy(
            float32_to_bfloat16_as_uint16(
                np.zeros((0, hidden_dim), dtype=np.float32)
            )
        )
        .view(DType.bfloat16)
        .to(Accelerator(id=0))
    )
    per_device_inputs = [
        [
            hs0_empty,
            Buffer.from_numpy(np.array([], dtype=np.int64)).to(
                Accelerator(id=0)
            ),
        ],
        [
            hs1_empty,
            Buffer.from_numpy(np.array([], dtype=np.int64)).to(
                Accelerator(id=1)
            ),
        ],
    ]
    results = call_per_device_graph(filters, per_device_inputs)
    assert len(results) == 2
    assert results[0].shape[0] == 0
    assert results[1].shape[0] == 0
