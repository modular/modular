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
"""Tests for exactly-sized model input buffers and the bounds they use."""

from __future__ import annotations

from typing import Any, cast

import pytest
from max.driver import CPU
from max.dtype import DType
from max.nn.transformer import ReturnLogits
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessorRuntime,
    ModelInputBuffers,
)

MAX_TOKENS = 64


@pytest.fixture
def buffers() -> ModelInputBuffers:
    """Buffers holding one declared input, sized for ``MAX_TOKENS`` tokens."""
    buffers = ModelInputBuffers()
    buffers.declare(
        name="ragged_input_tokens",
        dtype=DType.int64,
        max_shape=(MAX_TOKENS,),
    )
    return buffers


def test_different_shapes_share_one_backing(
    buffers: ModelInputBuffers,
) -> None:
    """Ragged shapes under one name are prefixes of a single allocation."""
    device = CPU()

    first = buffers.view(name="ragged_input_tokens", shape=(8,), device=device)
    second = buffers.view(
        name="ragged_input_tokens", shape=(32,), device=device
    )
    first_again = buffers.view(
        name="ragged_input_tokens", shape=(8,), device=device
    )

    assert first._data_ptr() == second._data_ptr()
    # Graph replay skips the input copy only for the same Buffer object.
    assert first_again is first
    assert first.shape == (8,)
    assert first.dtype == DType.int64
    assert second.shape == (32,)
    assert second.dtype == DType.int64


def test_the_declared_maximum_fits_the_backing(
    buffers: ModelInputBuffers,
) -> None:
    """The backing is sized for the declared maximum, exactly."""
    device = CPU()

    full = buffers.view(
        name="ragged_input_tokens", shape=(MAX_TOKENS,), device=device
    )
    tiny = buffers.view(name="ragged_input_tokens", shape=(1,), device=device)

    assert full.shape == (MAX_TOKENS,)
    assert full._data_ptr() == tiny._data_ptr()


def test_different_names_do_not_alias(buffers: ModelInputBuffers) -> None:
    device = CPU()
    buffers.declare(
        name="ragged_input_row_offsets",
        dtype=DType.uint32,
        max_shape=(9,),
    )

    tokens = buffers.view(name="ragged_input_tokens", shape=(8,), device=device)
    row_offsets = buffers.view(
        name="ragged_input_row_offsets", shape=(9,), device=device
    )

    assert tokens._data_ptr() != row_offsets._data_ptr()
    assert row_offsets.dtype == DType.uint32


def test_zero_sized_shape_uses_the_backing(
    buffers: ModelInputBuffers,
) -> None:
    device = CPU()

    empty = buffers.view(name="ragged_input_tokens", shape=(0,), device=device)
    nonempty = buffers.view(
        name="ragged_input_tokens", shape=(4,), device=device
    )

    assert empty.shape == (0,)
    assert empty._data_ptr() == nonempty._data_ptr()


def test_multi_dimensional_shapes_fit_the_element_bound(
    buffers: ModelInputBuffers,
) -> None:
    """A declared maximum bounds elements, not one leading dimension."""
    device = CPU()
    buffers.declare(name="padded_tokens", dtype=DType.int64, max_shape=(4, 16))

    rows = buffers.view(name="padded_tokens", shape=(2, 16), device=device)
    full = buffers.view(name="padded_tokens", shape=(4, 16), device=device)

    assert rows.shape == (2, 16)
    assert rows._data_ptr() == full._data_ptr()


def test_redeclaring_a_name_differently_raises(
    buffers: ModelInputBuffers,
) -> None:
    buffers.declare(
        name="ragged_input_tokens",
        dtype=DType.int64,
        max_shape=(MAX_TOKENS,),
    )

    with pytest.raises(ValueError, match="already declared"):
        buffers.declare(
            name="ragged_input_tokens",
            dtype=DType.int64,
            max_shape=(MAX_TOKENS * 2,),
        )
    with pytest.raises(ValueError, match="already declared"):
        buffers.declare(
            name="ragged_input_tokens",
            dtype=DType.uint32,
            max_shape=(MAX_TOKENS,),
        )


def test_undeclared_name_raises(buffers: ModelInputBuffers) -> None:
    with pytest.raises(KeyError):
        buffers.view(name="token_positions", shape=(4,), device=CPU())


def test_outgrowing_the_declaration_raises(
    buffers: ModelInputBuffers,
) -> None:
    """The declared maximum is the real bound, so overrunning it is a bug."""
    device = CPU()

    with pytest.raises(RuntimeError) as excinfo:
        buffers.view(
            name="ragged_input_tokens",
            shape=(MAX_TOKENS + 1,),
            device=device,
        )

    message = str(excinfo.value)
    assert "ragged_input_tokens" in message
    assert str(MAX_TOKENS + 1) in message
    assert str(MAX_TOKENS) in message
    assert str(device) in message


def test_a_raise_leaves_the_backing_usable(
    buffers: ModelInputBuffers,
) -> None:
    """Nothing is cached for the shape that failed."""
    device = CPU()
    within = buffers.view(name="ragged_input_tokens", shape=(4,), device=device)

    with pytest.raises(RuntimeError):
        buffers.view(
            name="ragged_input_tokens",
            shape=(MAX_TOKENS + 1,),
            device=device,
        )

    after = buffers.view(name="ragged_input_tokens", shape=(8,), device=device)
    assert after._data_ptr() == within._data_ptr()
    with pytest.raises(RuntimeError):
        buffers.view(
            name="ragged_input_tokens",
            shape=(MAX_TOKENS + 1,),
            device=device,
        )


def _runtime(
    *,
    max_batch_size: int,
    max_seq_len: int,
    max_batch_input_tokens: int,
    enable_chunked_prefill: bool = True,
    data_parallel_degree: int = 1,
) -> BatchProcessorRuntime:
    """A runtime carrying only the batching dimensions these tests read."""
    return BatchProcessorRuntime(
        pipeline_config=cast(Any, None),
        devices=[],
        return_logits=ReturnLogits.LAST_TOKEN,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        max_batch_input_tokens=max_batch_input_tokens,
        enable_chunked_prefill=enable_chunked_prefill,
        data_parallel_degree=data_parallel_degree,
    )


def test_chunked_prefill_bounds_active_tokens_by_the_budget() -> None:
    """Chunking holds a batch's active windows inside the CE budget."""
    runtime = _runtime(
        max_batch_size=32, max_seq_len=8192, max_batch_input_tokens=2048
    )

    assert runtime.max_batch_active_tokens == 2048


def test_unchunked_prefill_admits_one_oversized_request() -> None:
    """Without chunking the budget is soft: one whole sequence can overrun."""
    runtime = _runtime(
        max_batch_size=32,
        max_seq_len=8192,
        max_batch_input_tokens=2048,
        enable_chunked_prefill=False,
    )

    assert runtime.max_batch_active_tokens == 2048 + 8192


def test_active_tokens_never_exceed_a_full_batch_of_sequences() -> None:
    """A budget above what the batch can physically hold does not size it."""
    runtime = _runtime(
        max_batch_size=2,
        max_seq_len=128,
        max_batch_input_tokens=8192,
        enable_chunked_prefill=False,
    )

    assert runtime.max_batch_active_tokens == 2 * 128


def test_active_tokens_cover_one_token_per_context() -> None:
    """A decode step still needs a slot for every context in the batch."""
    runtime = _runtime(
        max_batch_size=16, max_seq_len=8192, max_batch_input_tokens=8
    )

    assert runtime.max_batch_active_tokens == 16


def test_data_parallel_replicas_each_get_their_own_budget() -> None:
    """Replica batches are staged into one stream, so the bounds multiply."""
    runtime = _runtime(
        max_batch_size=8,
        max_seq_len=8192,
        max_batch_input_tokens=2048,
        data_parallel_degree=4,
    )

    assert runtime.max_global_batch_size == 32
    assert runtime.max_batch_active_tokens == 4 * 2048


def test_one_replica_leaves_the_bounds_alone() -> None:
    runtime = _runtime(
        max_batch_size=8, max_seq_len=8192, max_batch_input_tokens=2048
    )

    assert runtime.max_global_batch_size == 8
    assert runtime.max_batch_active_tokens == 2048
