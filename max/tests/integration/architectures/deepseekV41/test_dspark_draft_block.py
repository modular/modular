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
"""Checks the DSpark draft loop's greedy tokens against a float64 loop.

A greedy draft adds each position's Markov bias inside the fused
``argmax_with_bias``, reading the bf16 logits in place; a sampled one widens
them and adds in the graph. Both must pick the tokens a float64 loop picks.
The controls drop the bias and the feedback of each drawn token, and require
the tokens to move, so the comparison cannot pass on data where neither
matters.
"""

from __future__ import annotations

import pytest
import torch
from max.driver import Accelerator
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn.legacy.layer import Module
from max.pipelines.architectures.deepseekV41.dspark_drafter import draft_block
from max.pipelines.architectures.deepseekV41.layers.dspark import (
    DSparkMarkovHead,
)
from torch.utils.dlpack import from_dlpack

BLOCK = 6
RANK = 256


class _Head(Module):
    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.markov_head = DSparkMarkovHead(
            vocab_size=vocab,
            draft_vocab_size=vocab,
            markov_rank=RANK,
            device=DeviceRef.GPU(),
        )

    def __call__(self) -> None:
        raise NotImplementedError("a weight holder; trace with its members")


def _sampled(logits: TensorValue, step: int) -> TensorValue:
    del step
    return ops.squeeze(ops.argmax(logits, axis=-1), axis=-1)


def _weights(vocab: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(vocab)
    return {
        "markov_head.embed.weight": torch.randn(vocab, RANK, generator=gen).to(
            torch.bfloat16
        ),
        "markov_head.head.weight": (
            torch.randn(vocab, RANK, generator=gen) / RANK**0.5
        ).to(torch.bfloat16),
    }


def _model(
    session: InferenceSession, vocab: int, weights: dict[str, torch.Tensor]
) -> Model:
    module = _Head(vocab)
    module.load_state_dict(weights)
    with Graph(
        f"dspark_draft_block_{vocab}",
        input_types=[
            TensorType(DType.bfloat16, ["b", BLOCK, vocab], DeviceRef.GPU()),
            TensorType(DType.int32, ["b"], DeviceRef.GPU()),
        ],
    ) as graph:
        logits, anchors = (x.tensor for x in graph.inputs)
        graph.output(
            draft_block(logits, anchors, module.markov_head),
            draft_block(logits, anchors, module.markov_head, _sampled),
        )
    return session.load(graph, weights_registry=module.state_dict())


def _reference(
    logits: torch.Tensor,
    anchors: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    bias: bool = True,
    feedback: bool = True,
) -> torch.Tensor:
    embed = weights["markov_head.embed.weight"].double().cuda()
    head = weights["markov_head.head.weight"].double().cuda()
    current = anchors.long()
    drafted = []
    for i in range(BLOCK):
        step = logits[:, i].double()
        if bias:
            prev = current if feedback else anchors.long()
            step = step + embed[prev] @ head.T
        current = step.argmax(dim=-1)
        drafted.append(current)
    return torch.stack(drafted, dim=1).int()


# A vocabulary off the fused kernel's eight-element loads, and an aligned one.
VOCABS = (1031, 4096)


@pytest.fixture(scope="module")
def models() -> dict[int, tuple[dict[str, torch.Tensor], Model]]:
    session = InferenceSession(devices=[Accelerator()])
    out = {}
    for vocab in VOCABS:
        weights = _weights(vocab)
        out[vocab] = (weights, _model(session, vocab, weights))
    return out


@pytest.mark.parametrize("vocab", VOCABS)
@pytest.mark.parametrize("rows", [1, 7])
def test_draft_block_matches_a_float64_loop(
    models: dict[int, tuple[dict[str, torch.Tensor], Model]],
    vocab: int,
    rows: int,
) -> None:
    weights, model = models[vocab]
    gen = torch.Generator(device="cuda").manual_seed(rows)
    logits = torch.randn(rows, BLOCK, vocab, generator=gen, device="cuda").to(
        torch.bfloat16
    )
    anchors = torch.randint(
        vocab, (rows,), generator=gen, device="cuda", dtype=torch.int32
    )
    fused, sampled = (
        from_dlpack(t).clone() for t in model.execute(logits, anchors)
    )
    want = _reference(logits, anchors, weights)
    torch.testing.assert_close(fused, want, rtol=0, atol=0)
    torch.testing.assert_close(sampled, want, rtol=0, atol=0)

    no_bias = _reference(logits, anchors, weights, bias=False)
    no_feedback = _reference(logits, anchors, weights, feedback=False)
    assert not torch.equal(no_bias, want), "the bias never moves a token"
    assert not torch.equal(no_feedback[:, 1:], want[:, 1:]), (
        "no position reads the token drawn before it"
    )
