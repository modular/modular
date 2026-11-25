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
"""Stub graph pieces shared by the speculative driver step tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.graph import BufferValue, DeviceRef, TensorValue, ops
from max.nn._graph_ops.sampling import AcceptanceSampler
from max.nn.kv_cache import KVCacheInputsPerDevice
from max.pipelines.speculative.driver import SequentialBatch, SequentialDriver
from max.pipelines.speculative.ragged_token_merger import RaggedTokenMerger
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec
from max.pipelines.speculative.spec_target import Verified


def stub_kv_leaf(
    cache_lengths: TensorValue, kv_blocks: BufferValue, cpu: DeviceRef
) -> KVCacheInputsPerDevice[TensorValue, BufferValue]:
    """Returns a cache leaf for stubbed passes, which read only
    ``cache_lengths``."""
    scalar = ops.constant(np.zeros(1, np.int64), DType.int64, cpu)
    return KVCacheInputsPerDevice(
        kv_blocks=kv_blocks,
        cache_lengths=cache_lengths,
        lookup_table=scalar,
        max_prompt_length=scalar,
        max_cache_length=scalar,
        page_stride=scalar,
    )


def stub_sequential_driver(
    proposer: object,
    *,
    device: DeviceRef,
    num_draft_steps: int,
    per_row: bool,
    head_free: bool = False,
    draft_proposal: Literal["argmax", "sampled"] = "argmax",
    vocab_size: int | None = None,
    distributed: bool = False,
    observe_batch: Callable[[SequentialBatch], None] | None = None,
) -> SequentialDriver[None]:
    """Returns a driver that verifies with the logits supplied by the test."""

    def verify(batch: SequentialBatch) -> Verified[list[TensorValue]]:
        if observe_batch is not None:
            observe_batch(batch)
        logits = batch.extra["logits"]
        return Verified(logits=logits, hidden=[logits])

    driver = SequentialDriver.__new__(SequentialDriver)
    vars(driver).update(
        _proposer=proposer,
        _verify=verify,
        _draft_proposal=draft_proposal,
        _vocab_size=vocab_size,
        _skips_draft_rows=False,
        _head_free_scores=head_free,
        num_draft_steps=num_draft_steps,
        data_parallel_degree=1,
        devices=[device],
        merger=RaggedTokenMerger(device),
        _input_spec=SpecDecodeInputTypeSpec(
            devices=[device],
            distributed=distributed,
            include_per_row_verify=per_row,
            num_draft_positions=num_draft_steps if per_row else None,
        ),
        acceptance_sampler=AcceptanceSampler(
            num_draft_steps=num_draft_steps,
            use_stochastic=True,
            draft_proposal=draft_proposal,
            vocab_size=vocab_size,
        ),
    )
    return driver


def unrestricted_sampling_buffers(
    batch: int, vocab: int, device: Device
) -> list[Buffer]:
    """Returns seed, temperature, top_k, max_k, top_p and min_top_p for a
    step that samples from the full distribution."""
    return [
        Buffer.from_numpy(np.arange(batch, dtype=np.uint64)).to(device),
        Buffer.from_numpy(np.ones(batch, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.full(batch, vocab, dtype=np.int64)).to(device),
        Buffer.from_numpy(np.array(vocab, dtype=np.int64)),
        Buffer.from_numpy(np.ones(batch, dtype=np.float32)).to(device),
        Buffer.from_numpy(np.array(1.0, dtype=np.float32)),
    ]


def outputs_to_numpy(outputs: Sequence[object]) -> list[np.ndarray]:
    """Returns each model output as a numpy array."""
    results = []
    for out in outputs:
        assert isinstance(out, Buffer)
        results.append(out.to_numpy())
    return results
