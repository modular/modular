# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.pipelines.lib.speculative_decoding import (
    accepted_hidden_states_extractor,
    compute_extractor_inputs,
    compute_filter_indices,
    filter_hidden_states,
)


@pytest.fixture
def session() -> InferenceSession:
    return InferenceSession(devices=[Accelerator()])


def test_accepted_hidden_states_extractor_multiple_batches(
    session: InferenceSession,
) -> None:
    """Test extraction with multiple batch elements with different acceptance counts."""
    device = DeviceRef.GPU()
    graph = accepted_hidden_states_extractor(device)
    model = session.load(graph)

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
    first_rejected_tokens = np.array([1, 3, 0], dtype=np.int64)

    total_range_np, output_offsets_np = compute_extractor_inputs(
        first_rejected_tokens
    )
    total_range = torch.from_numpy(total_range_np).cuda()
    output_offsets = torch.from_numpy(output_offsets_np).cuda()

    (result,) = model(
        Buffer.from_dlpack(hidden_states),
        Buffer.from_dlpack(logit_offsets),
        Buffer.from_dlpack(total_range),
        Buffer.from_dlpack(output_offsets),
    )

    result_max = torch.from_dlpack(result)

    expected = torch.cat(
        [
            hidden_states[0:2],
            hidden_states[4:8],
            hidden_states[9:10],
        ],
        dim=0,
    )

    torch.testing.assert_close(result_max, expected, rtol=1e-2, atol=1e-3)


def test_filter_hidden_states_keep_all(
    session: InferenceSession,
) -> None:
    """Test filtering when all hidden states are kept."""
    device = DeviceRef.GPU()
    graph = filter_hidden_states(device)
    model = session.load(graph)

    hidden_dim = 64
    hidden_states = torch.randn(3, hidden_dim, dtype=torch.bfloat16).cuda()
    gather_indices = torch.tensor([0, 1, 2], dtype=torch.int64).cuda()

    (result,) = model(
        Buffer.from_dlpack(hidden_states),
        Buffer.from_dlpack(gather_indices),
    )

    result_max = torch.from_dlpack(result)

    torch.testing.assert_close(result_max, hidden_states, rtol=1e-2, atol=1e-3)


def test_filter_hidden_states_remove_middle_sequence(
    session: InferenceSession,
) -> None:
    """Test filtering to remove middle sequence's hidden states."""
    device = DeviceRef.GPU()
    graph = filter_hidden_states(device)
    model = session.load(graph)

    hidden_dim = 64
    hidden_states = torch.randn(6, hidden_dim, dtype=torch.bfloat16).cuda()
    gather_indices = torch.tensor([0, 1, 4, 5], dtype=torch.int64).cuda()

    (result,) = model(
        Buffer.from_dlpack(hidden_states),
        Buffer.from_dlpack(gather_indices),
    )

    result_max = torch.from_dlpack(result)

    expected = torch.cat(
        [
            hidden_states[0:2],
            hidden_states[4:6],
        ],
        dim=0,
    )
    torch.testing.assert_close(result_max, expected, rtol=1e-2, atol=1e-3)


def test_filter_hidden_states_with_active_contexts(
    session: InferenceSession,
) -> None:
    """Test filter logic with compute_filter_indices (positive case: some sequences terminate)."""
    device = DeviceRef.GPU()
    graph = filter_hidden_states(device)
    model = session.load(graph)

    hidden_dim = 64
    first_rejected_tokens = np.array([2, 1, 3], dtype=np.int64)
    active_context_indices = [0, 2]

    keep_indices_np, offsets = compute_filter_indices(
        first_rejected_tokens, active_context_indices
    )

    total_hidden_states = int(offsets[-1])
    hidden_states = torch.randn(
        total_hidden_states, hidden_dim, dtype=torch.bfloat16
    ).cuda()

    gather_indices = torch.from_numpy(keep_indices_np).cuda()

    (result,) = model(
        Buffer.from_dlpack(hidden_states),
        Buffer.from_dlpack(gather_indices),
    )

    result_max = torch.from_dlpack(result)

    expected = torch.cat(
        [
            hidden_states[0:3],
            hidden_states[5:9],
        ],
        dim=0,
    )
    torch.testing.assert_close(result_max, expected, rtol=1e-2, atol=1e-3)


def test_filter_hidden_states_all_active(
    session: InferenceSession,
) -> None:
    """Test filter logic with compute_filter_indices (negative case: all sequences active)."""
    device = DeviceRef.GPU()
    graph = filter_hidden_states(device)
    model = session.load(graph)

    hidden_dim = 64
    first_rejected_tokens = np.array([1, 2, 1], dtype=np.int64)
    active_context_indices = [0, 1, 2]

    keep_indices_np, offsets = compute_filter_indices(
        first_rejected_tokens, active_context_indices
    )

    total_hidden_states = int(offsets[-1])
    hidden_states = torch.randn(
        total_hidden_states, hidden_dim, dtype=torch.bfloat16
    ).cuda()

    gather_indices = torch.from_numpy(keep_indices_np).cuda()

    (result,) = model(
        Buffer.from_dlpack(hidden_states),
        Buffer.from_dlpack(gather_indices),
    )

    result_max = torch.from_dlpack(result)

    torch.testing.assert_close(result_max, hidden_states, rtol=1e-2, atol=1e-3)
