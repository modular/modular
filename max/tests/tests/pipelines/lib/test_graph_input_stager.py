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
"""Tests for the graph input stager and the bounds it sizes from."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytest
from max.driver import CPU, Device
from max.dtype import DType
from max.nn.transformer import ReturnLogits
from max.pipelines.graph_input_stager import GraphInputStager, InputDescriptor
from max.pipelines.lib.interfaces.batch_processor import BatchProcessorRuntime

MAX_TOKENS = 64
TOKENS = "ragged_input_tokens"
OFFSETS = "ragged_input_row_offsets"


def describe(
    name: str,
    *,
    dtype: DType = DType.int64,
    max_shape: tuple[int, ...] = (MAX_TOKENS,),
    destinations: Sequence[Device] | None = None,
) -> InputDescriptor:
    """An input description, defaulted to the token stream's shape."""
    return InputDescriptor(
        name=name,
        dtype=dtype,
        max_shape=max_shape,
        destinations=[CPU()] if destinations is None else destinations,
    )


@pytest.fixture
def stager() -> GraphInputStager:
    """A stager holding one input, sized for ``MAX_TOKENS`` tokens."""
    return GraphInputStager([describe(TOKENS)])


def test_get_returns_host_staging_and_its_destinations(
    stager: GraphInputStager,
) -> None:
    with stager.stage() as staging:
        host, destinations = staging.get(TOKENS, (4,))

    assert host.shape == (4,)
    assert host.dtype == DType.int64
    assert len(destinations) == 1
    assert destinations[0].shape == (4,)


def test_nothing_is_sent_until_the_scope_closes(
    stager: GraphInputStager,
) -> None:
    with stager.stage() as staging:
        settled, _ = staging.get(TOKENS, (4,))
        settled.to_numpy()[:] = [1, 2, 3, 4]

    with stager.stage() as staging:
        host, (again,) = staging.get(TOKENS, (4,))
        host.to_numpy()[:] = [9, 9, 9, 9]
        assert again.to_numpy().tolist() == [1, 2, 3, 4]

    assert again.to_numpy().tolist() == [9, 9, 9, 9]


def test_a_raising_body_sends_nothing(stager: GraphInputStager) -> None:
    """Its host writes did not finish, so what staging holds is not input."""
    with stager.stage() as staging:
        settled, (device,) = staging.get(TOKENS, (4,))
        settled.to_numpy()[:] = [1, 2, 3, 4]

    with pytest.raises(ZeroDivisionError):
        with stager.stage() as staging:
            host, _ = staging.get(TOKENS, (4,))
            host.to_numpy()[:] = [9, 9, 9, 9]
            raise ZeroDivisionError

    assert device.to_numpy().tolist() == [1, 2, 3, 4]


def test_a_raising_body_leaves_the_stager_usable(
    stager: GraphInputStager,
) -> None:
    """The scope closes even when its body does not finish."""
    with pytest.raises(ZeroDivisionError):
        with stager.stage():
            raise ZeroDivisionError

    with stager.stage() as staging:
        host, (device,) = staging.get(TOKENS, (2,))
        host.to_numpy()[:] = [5, 6]

    assert device.to_numpy().tolist() == [5, 6]


def test_one_scope_sends_every_input_staged_in_it() -> None:
    stager = GraphInputStager(
        [
            describe(TOKENS),
            describe(OFFSETS, dtype=DType.uint32, max_shape=(9,)),
        ]
    )

    with stager.stage() as staging:
        tokens_host, (tokens,) = staging.get(TOKENS, (3,))
        offsets_host, (offsets,) = staging.get(OFFSETS, (2,))
        tokens_host.to_numpy()[:] = [4, 5, 6]
        offsets_host.to_numpy()[:] = [0, 3]

    assert tokens.to_numpy().tolist() == [4, 5, 6]
    assert offsets.to_numpy().tolist() == [0, 3]


def test_a_scope_sends_only_what_it_staged() -> None:
    """An input this forward skipped is not re-sent from a stale host buffer.

    The one in ``overlap_text_generation`` is read back into after its scope
    closes, so a later forward re-sending it would push that back over the
    device buffer.
    """
    stager = GraphInputStager(
        [
            describe(TOKENS),
            describe(OFFSETS, dtype=DType.uint32, max_shape=(9,)),
        ]
    )

    with stager.stage() as staging:
        offsets_host, (offsets,) = staging.get(OFFSETS, (2,))
        offsets_host.to_numpy()[:] = [0, 2]

    # Whoever owns the input is free to reuse the host buffer afterwards.
    offsets_host.to_numpy()[:] = [9, 9]
    with stager.stage() as staging:
        tokens_host, (tokens,) = staging.get(TOKENS, (2,))
        tokens_host.to_numpy()[:] = [1, 2]

    assert tokens.to_numpy().tolist() == [1, 2]
    assert offsets.to_numpy().tolist() == [0, 2]


def test_an_input_never_asked_for_has_nothing_to_send() -> None:
    """A description no step has staged has nothing to send."""
    stager = GraphInputStager(
        [
            describe(TOKENS),
            describe("unused", dtype=DType.uint32, max_shape=(4,)),
        ]
    )

    with stager.stage() as staging:
        host, (device,) = staging.get(TOKENS, (2,))
        host.to_numpy()[:] = [3, 4]

    assert device.to_numpy().tolist() == [3, 4]


def test_one_input_asked_for_twice_in_a_scope_is_one_host_write(
    stager: GraphInputStager,
) -> None:
    """Every shard of a replica asks for the metadata it shares.

    They have to land in one host buffer, or the second ask would replace the
    first's staging and drop what had been written to it.
    """
    with stager.stage() as staging:
        first_host, first_destinations = staging.get(TOKENS, (2,))
        first_host.to_numpy()[:] = [7, 8]
        second_host, second_destinations = staging.get(TOKENS, (2,))

        assert second_host is first_host
        assert second_destinations == first_destinations
        assert second_host.to_numpy().tolist() == [7, 8]

    assert first_destinations[0].to_numpy().tolist() == [7, 8]


def test_different_shapes_share_one_backing(
    stager: GraphInputStager,
) -> None:
    """Ragged shapes under one name are prefixes of a single allocation."""
    with stager.stage() as staging:
        _, (first,) = staging.get(TOKENS, (8,))
    with stager.stage() as staging:
        _, (second,) = staging.get(TOKENS, (32,))
    with stager.stage() as staging:
        _, (first_again,) = staging.get(TOKENS, (8,))

    # What has to stay put is the allocation, not the view over it: replay's
    # refresh copy is dropped for a pair that names the same memory.
    assert first._data_ptr() == second._data_ptr()
    assert first_again._data_ptr() == first._data_ptr()
    assert first.shape == (8,)
    assert second.shape == (32,)


def test_one_name_staged_at_two_shapes_in_a_scope_raises(
    stager: GraphInputStager,
) -> None:
    """A forward has one shape per input, and only one is sent.

    Letting the second call through would silently drop what the first wrote.
    """
    with pytest.raises(RuntimeError, match="staged as"):
        with stager.stage() as staging:
            staging.get(TOKENS, (8,))
            staging.get(TOKENS, (32,))


def test_a_host_device_gets_pageable_staging() -> None:
    """Callers do not choose: a host device cannot pin, so the stager does.

    An accelerator gets ``DevicePinnedBuffer`` so its H2D is async. There is
    no accelerator here to assert that half on.
    """
    stager = GraphInputStager([describe(TOKENS)])

    with stager.stage() as staging:
        host, _ = staging.get(TOKENS, (4,))

    assert not host.pinned


def test_host_staging_is_fresh_every_step(
    stager: GraphInputStager,
) -> None:
    """Never recycled by us: overlap would clobber an in-flight copy."""
    with stager.stage() as staging:
        first_host, _ = staging.get(TOKENS, (4,))
        first_host.to_numpy()[:] = [1, 2, 3, 4]
    with stager.stage() as staging:
        second_host, _ = staging.get(TOKENS, (4,))
        second_host.to_numpy()[:] = [9, 9, 9, 9]

    assert first_host is not second_host
    assert first_host.to_numpy().tolist() == [1, 2, 3, 4]


def test_the_described_maximum_fits_the_backing(
    stager: GraphInputStager,
) -> None:
    """The backing is sized for the described maximum, exactly."""
    with stager.stage() as staging:
        _, (full,) = staging.get(TOKENS, (MAX_TOKENS,))
    with stager.stage() as staging:
        _, (tiny,) = staging.get(TOKENS, (1,))

    assert full.shape == (MAX_TOKENS,)
    assert full._data_ptr() == tiny._data_ptr()


def test_different_names_do_not_alias() -> None:
    stager = GraphInputStager(
        [
            describe(TOKENS),
            describe(OFFSETS, dtype=DType.uint32, max_shape=(9,)),
        ]
    )

    with stager.stage() as staging:
        _, (tokens,) = staging.get(TOKENS, (8,))
        _, (row_offsets,) = staging.get(OFFSETS, (9,))

    assert tokens._data_ptr() != row_offsets._data_ptr()
    assert row_offsets.dtype == DType.uint32


def test_replicas_do_not_share_a_buffer_on_one_device() -> None:
    """Data parallelism on CPU hands every replica the same ``Device``.

    The driver calls two such devices equal, so the name is what keeps one
    replica's batch from overwriting the other's.
    """
    first = f"0/{TOKENS}"
    second = f"1/{TOKENS}"
    stager = GraphInputStager([describe(first), describe(second)])

    with stager.stage() as staging:
        first_host, (first_device,) = staging.get(first, (3,))
        second_host, (second_device,) = staging.get(second, (2,))
        first_host.to_numpy()[:] = [1, 2, 3]
        second_host.to_numpy()[:] = [7, 8]

    assert first_device.to_numpy().tolist() == [1, 2, 3]
    assert second_device.to_numpy().tolist() == [7, 8]


def test_tensor_parallel_shards_each_get_a_buffer() -> None:
    """One name, one host write, a destination per shard."""
    stager = GraphInputStager([describe(TOKENS, destinations=[CPU(), CPU()])])

    with stager.stage() as staging:
        host, destinations = staging.get(TOKENS, (3,))
        host.to_numpy()[:] = [5, 6, 7]

    assert len(destinations) == 2
    assert destinations[0]._data_ptr() != destinations[1]._data_ptr()
    for destination in destinations:
        assert destination.to_numpy().tolist() == [5, 6, 7]


def test_zero_sized_shape_uses_the_backing(
    stager: GraphInputStager,
) -> None:
    with stager.stage() as staging:
        _, (empty,) = staging.get(TOKENS, (0,))
    with stager.stage() as staging:
        _, (nonempty,) = staging.get(TOKENS, (4,))

    assert empty.shape == (0,)
    assert empty._data_ptr() == nonempty._data_ptr()


def test_multi_dimensional_shapes_fit_the_element_bound() -> None:
    """A described maximum bounds elements, not one leading dimension."""
    stager = GraphInputStager([describe("padded_tokens", max_shape=(4, 16))])

    with stager.stage() as staging:
        _, (rows,) = staging.get("padded_tokens", (2, 16))
    with stager.stage() as staging:
        _, (full,) = staging.get("padded_tokens", (4, 16))

    assert rows.shape == (2, 16)
    assert rows._data_ptr() == full._data_ptr()


def test_describing_a_name_twice_raises() -> None:
    """A name belongs to whoever owns the input.

    Even two identical descriptions: owners that happen to agree today still
    disagree about who sizes it.
    """
    with pytest.raises(ValueError, match="described twice"):
        GraphInputStager([describe(TOKENS), describe(TOKENS)])
    with pytest.raises(ValueError, match="described twice"):
        GraphInputStager(
            [describe(TOKENS), describe(TOKENS, max_shape=(MAX_TOKENS * 2,))]
        )


def test_describing_without_a_destination_raises() -> None:
    with pytest.raises(ValueError, match="at least one destination"):
        GraphInputStager([describe(TOKENS, destinations=[])])


def test_an_undescribed_name_raises(stager: GraphInputStager) -> None:
    with pytest.raises(KeyError):
        with stager.stage() as staging:
            staging.get("token_positions", (4,))


def test_outgrowing_the_description_raises(
    stager: GraphInputStager,
) -> None:
    """The described maximum is the real bound, so overrunning it is a bug."""
    with pytest.raises(RuntimeError) as excinfo:
        with stager.stage() as staging:
            staging.get(TOKENS, (MAX_TOKENS + 1,))

    message = str(excinfo.value)
    assert TOKENS in message
    assert str(MAX_TOKENS + 1) in message
    assert str(MAX_TOKENS) in message


def test_a_raise_leaves_the_backing_usable(
    stager: GraphInputStager,
) -> None:
    """Nothing is cached for the shape that failed."""
    with stager.stage() as staging:
        _, (within,) = staging.get(TOKENS, (4,))

    with pytest.raises(RuntimeError):
        with stager.stage() as staging:
            staging.get(TOKENS, (MAX_TOKENS + 1,))

    with stager.stage() as staging:
        _, (after,) = staging.get(TOKENS, (8,))
    assert after._data_ptr() == within._data_ptr()


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
