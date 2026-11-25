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
"""Checks confidence scores against the embedding of each previous token."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from dspark_test_support import _to_max
from max.driver import CPU
from max.dtype import DType
from max.pipelines.architectures.deepseekV41_modulev3.dspark_drafter import (
    DSparkDrafter,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)
from max.tensor import default_device
from max.tensor.realization_context import ensure_context
from numpy.typing import NDArray


def _numpy(
    embed: NDArray[np.float32],
    proj: NDArray[np.float32],
    hidden: NDArray[np.float32],
    anchors: NDArray[np.int32],
    drafted: NDArray[np.int32],
) -> NDArray[np.float32]:
    prev = np.concatenate([anchors[:, None], drafted[:, :-1]], axis=1)
    steps = hidden.reshape(*drafted.shape, hidden.shape[-1])
    combined = np.concatenate([steps, embed[prev]], axis=-1)
    return np.squeeze(combined @ proj.T, axis=-1)


def test_confidence_scores_match_the_numpy_reference() -> None:
    embed = torch.tensor(
        [[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.bfloat16
    )
    proj = torch.tensor([[1, 2, 4, 8, 10]], dtype=torch.float32)
    hidden = torch.eye(4, dtype=torch.bfloat16)
    anchors = torch.tensor([0, 3], dtype=torch.int32)
    drafted = torch.tensor([[1, 2], [5, 6]], dtype=torch.int32)

    with default_device(CPU()), ensure_context():
        markov_head = DSparkMarkovHead(
            vocab_size=embed.shape[0],
            draft_vocab_size=embed.shape[0],
            markov_rank=embed.shape[1],
        )
        markov_head.embed.weight = _to_max(embed)
        head = DSparkConfidenceHead(
            hidden_size=hidden.shape[1], markov_rank=embed.shape[1]
        )
        head.proj.weight = _to_max(proj)
        drafter = Mock(
            spec=DSparkDrafter,
            config=SimpleNamespace(hidden_size=hidden.shape[1]),
            block_size=drafted.shape[1],
            markov_head=markov_head,
            confidence_head=head,
        )
        assert isinstance(drafter, DSparkDrafter)
        scores = DSparkDrafter.confidence_scores(
            drafter, _to_max(hidden), _to_max(anchors), _to_max(drafted)
        )

    assert scores.dtype == DType.float32
    want = _numpy(
        embed.float().numpy(),
        proj.numpy(),
        hidden.float().numpy(),
        anchors.numpy(),
        drafted.numpy(),
    )
    np.testing.assert_allclose(scores.to_numpy(), want, rtol=1e-6, atol=1e-6)
