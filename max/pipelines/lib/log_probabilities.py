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

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.interfaces import LogProbabilities


def log_probabilities_ragged_graph(device: DeviceRef, *, levels: int) -> Graph:
    """Create a graph to compute log probabilities over ragged inputs.

    A model obtained by this graph is a required input to
    'compute_log_probabilities_ragged'.

    Args:
        device: The type of device this graph will need to run on.
        levels: log2(max_k+1) for the desired maximum top-k you'd like to
            support.  To support the OpenAI API maximum of 5 logprobs, use
            levels=3.  Higher levels can be used to support higher k.
    """
    out_per_token = 2**levels if levels > 0 else 1
    logit_dtype = DType.float32
    token_dtype = DType.uint32
    offset_dtype = DType.uint32
    host_device = DeviceRef.CPU()
    with Graph(
        "ragged_logprobs",
        input_types=[
            TensorType(logit_dtype, ("bseq_or_b", "vocab"), device),  # logits
            TensorType(token_dtype, ("batch_seq",), device),  # tokens
            TensorType(token_dtype, ("batch",), device),  # sampled_tokens
            TensorType(offset_dtype, ("batchp1",), device),  # logit_row_offsets
            TensorType(offset_dtype, ("batchp1",), device),  # token_row_offsets
            TensorType(offset_dtype, ("batchp1",), device),  # lp_output_offsets
            TensorType(offset_dtype, ("batchp1",), host_device),  # ^ for host
        ],
    ) as g:
        g.output(
            *ops.custom(
                "compute_log_probabilities_ragged",
                device,
                list(g.inputs),
                [
                    TensorType(  # lp_logits
                        logit_dtype, ("out_batch_seq", out_per_token), device
                    ),
                    TensorType(  # lp_tokens
                        # TODO(GEX-2198): out_batch_seq2 should be the same as
                        # out_batch_seq, but doing so causes a failure in KGEN.
                        token_dtype,
                        ("out_batch_seq2", out_per_token),
                        device,
                    ),
                ],
                {"levels": levels},
            )
        )
        return g


def compute_log_probabilities_ragged(
    device: Device,
    model: Model,
    *,
    input_row_offsets: npt.NDArray[np.integer[Any]],
    logits: Tensor | None,
    next_token_logits: Tensor,
    tokens: npt.NDArray[np.integer[Any]],
    sampled_tokens: npt.NDArray[np.integer[Any]],
    batch_top_n: Sequence[int],
    batch_echo: Sequence[bool],
) -> list[LogProbabilities | None]:
    """Computes the log probabilities for ragged model outputs.

    Args:
        device: Device on which to do the bulk of the log probabilities
            computation.  A small amount of computation still occurs on the
            host regardless of this setting.
        model: A compiled version of a graph from the
            'log_probabilities_ragged_graph' function.
        input_row_offsets: Token offsets into token-indexed buffers, by batch
            index.  Should have 1 more element than there are batches (batch n
            is token indices [input_row_offsets[n], input_row_offsets[n+1])).
        logits: (tokens, vocab_dim) tensor full of tensor logits.  Token
            dimension mapped to batches using input_row_offsets.  May be
            omitted only if all 'batch_echo' values are False.
        next_token_logits: (batch_dim, vocab_dim) tensor full of tensor logits
            for the next token in each batch item.
        sampled_tokens: (batch_dim,) tensor of sampled token per batch
        batch_top_n: Number of top log probabilities to return per input in
            the batch. For any element where `top_n == 0`, the
            LogProbabilities is skipped.
        batch_echo: Whether to include input tokens in the returned log
            probabilities.

    Returns:
        Computed log probabilities for each item in the batch.
    """
    assert len(input_row_offsets.shape) == 1
    if logits is not None:
        assert len(logits.shape) == 2
    assert len(next_token_logits.shape) == 2
    assert len(tokens.shape) == 1
    assert len(sampled_tokens.shape) == 1
    assert (
        len(batch_top_n)
        == len(batch_echo)
        == input_row_offsets.shape[0] - 1
        == next_token_logits.shape[0]
        == sampled_tokens.shape[0]
    )
    batch_size = len(batch_top_n)
    if logits is not None:
        assert logits.shape[0] == tokens.shape[0]
        assert logits.shape[1] == next_token_logits.shape[1]
    vocab_size = next_token_logits.shape[1]
    if logits is not None:
        assert logits.device == device
        assert logits.dtype == DType.float32
    assert next_token_logits.device == device
    assert next_token_logits.dtype == DType.float32
    logit_row_offsets: npt.NDArray[np.integer[Any]]
    if logits is None:
        assert not any(batch_echo)
        kernel_logits = next_token_logits
        logit_row_offsets = np.arange(batch_size + 1, dtype=np.uint32)
    else:
        kernel_logits = logits
        logit_row_offsets = input_row_offsets
    output_counts = np.array(
        [
            input_row_offsets[i + 1] - input_row_offsets[i] if echo else 1
            for i, echo in enumerate(batch_echo)
        ],
        dtype=np.uint32,
    )
    output_row_offsets = np.concatenate(
        [np.zeros(1, dtype=output_counts.dtype), np.cumsum(output_counts)]
    )
    token_row_offsets = input_row_offsets
    assert kernel_logits.dtype == DType.float32
    model_outputs = model.execute(
        kernel_logits,
        Tensor.from_numpy(tokens.astype(np.uint32)).to(device),
        Tensor.from_numpy(sampled_tokens.astype(np.uint32)).to(device),
        Tensor.from_numpy(logit_row_offsets.astype(np.uint32)).to(device),
        Tensor.from_numpy(token_row_offsets.astype(np.uint32)).to(device),
        Tensor.from_numpy(output_row_offsets.astype(np.uint32)).to(device),
        Tensor.from_numpy(output_row_offsets.astype(np.uint32)),  # for host
    )
    assert isinstance(model_outputs[0], Tensor)
    assert isinstance(model_outputs[1], Tensor)
    lp_logits = model_outputs[0].to_numpy()
    lp_tokens = model_outputs[1].to_numpy()

    def compute_top(output_index: int, top_n: int) -> dict[int, float]:
        if top_n > lp_logits.shape[1] - 1:
            raise ValueError(
                "top_n too large for this graph -- "
                "rebuild with larger levels & rerun"
            )
        top_assoc = [
            (int(token), float(logit))
            for token, logit in zip(
                lp_tokens[output_index, :-1], lp_logits[output_index, :-1]
            )
            if token < vocab_size
        ]
        top_assoc.sort(key=lambda item: item[1], reverse=True)
        del top_assoc[top_n:]
        top = dict(top_assoc)
        # Special case: If sampled token not in top-n, include it anyway.
        top[int(lp_tokens[output_index, -1])] = float(
            lp_logits[output_index, -1]
        )
        return top

    outputs: list[LogProbabilities | None] = []
    for batch_index in range(input_row_offsets.shape[0] - 1):
        if batch_top_n[batch_index] == 0:
            outputs.append(None)
            continue
        start = output_row_offsets[batch_index]
        end = output_row_offsets[batch_index + 1]
        outputs.append(
            LogProbabilities(
                token_log_probabilities=list(
                    map(float, lp_logits[start:end, -1])
                ),
                top_log_probabilities=[
                    compute_top(i, batch_top_n[batch_index])
                    for i in range(start, end)
                ],
            )
        )
    return outputs
