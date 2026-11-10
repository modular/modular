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
"""The mixed-step split record the KV cache managers fill for every batch."""

from __future__ import annotations

import numpy as np
import pytest
from max import tree
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheInputsPerDevice, MHAKVCacheParams
from max.nn.kv_cache.input_types import MIXED_STEP_SPLIT_SIZE
from max.pipelines.kv_cache import PagedKVCacheManager
from max.pipelines.kv_cache.paged_kv_cache._mixed_step_split import (
    MIXED_STEP_SPLIT_SWITCH,
    mixed_step_decode_width,
    mixed_step_split_record,
)
from max.pipelines.kv_cache.paged_kv_cache.jenga_cache_manager import (
    JengaKVCacheManager,
)
from test_common.context_utils import create_text_context

K = 3
WIDTH = mixed_step_decode_width(K)


def _record(widths: list[int], valid: list[int]) -> list[int]:
    return mixed_step_split_record(widths, valid, WIDTH).tolist()


def test_decode_rows_first_make_the_head_view() -> None:
    widths = [4, 4, 604]
    valid = [100, 200, 1_700]
    assert _record(widths, valid) == [
        2,
        8,
        4,
        200,
        604,
        1_700,
        1_700,
    ]


def test_decode_rows_last_make_the_tail_view() -> None:
    widths = [604, 9, 4, 4]
    valid = [1_700, 50, 100, 200]
    assert _record(widths, valid) == [
        2,
        613,
        604,
        1_700,
        4,
        200,
        1_750,
    ]


def test_the_longer_end_run_takes_the_decode_view() -> None:
    widths = [4, 604, 4, 4]
    record = _record(widths, [10, 20, 30, 40])
    # The trailing end run wins; a decode row at either end runs the decode view.
    assert record[0] == 2
    assert record[4] == WIDTH


def test_a_decode_row_after_a_prompt_row_stays_in_the_prefill_view() -> None:
    record = _record([4, 604, 4, 700], [10, 20, 30, 40])
    assert record[:2] == [1, 4]
    assert record[6] == 20 + 30 + 40


def test_a_one_token_chunk_rides_the_decode_view() -> None:
    # A one-token prompt chunk is as wide as a verify row, so it rides the decode view.
    assert _record([4, 4, 300], [1, 2, 3])[0] == 2


def test_short_prompt_rows_wider_than_verify_stay_in_the_prefill_view() -> None:
    assert _record([4, 6, 300], [1, 2, 3])[0] == 1


@pytest.mark.parametrize(
    "widths",
    [[4, 4, 4], [600, 700], [4, 8], [600, 4, 700]],
    ids=["all_decode", "all_prompt", "all_spec_width", "decode_in_middle"],
)
def test_no_split_gives_zeros(widths: list[int]) -> None:
    assert _record(widths, [1] * len(widths)) == [0] * MIXED_STEP_SPLIT_SIZE


def test_single_token_decode_rows_split_without_drafts() -> None:
    record = mixed_step_split_record(
        [1, 1, 500], [5, 6, 7], mixed_step_decode_width(0)
    )
    assert record[:3].tolist() == [2, 2, 1]


def test_decode_width_stops_at_the_spec_route() -> None:
    assert mixed_step_decode_width(3) == 4
    assert mixed_step_decode_width(7) == 8
    assert mixed_step_decode_width(15) == 8


def test_seven_drafts_put_eight_token_rows_in_the_decode_view() -> None:
    width = mixed_step_decode_width(7)
    # Mach order: a chunk with its MAGIC drafts, then verify rows; a
    # chunk just past the decode width stays in the prefill view.
    assert mixed_step_split_record(
        [607, 8, 8], [700, 40, 50], width
    ).tolist() == [
        1,
        607,
        607,
        700,
        8,
        50,
        700,
    ]
    assert mixed_step_split_record([8, 8, 9], [1, 2, 3], width)[
        :2
    ].tolist() == [2, 16]
    assert (
        mixed_step_split_record([8, 8, 8], [1, 2, 3], width).tolist()
        == [0] * MIXED_STEP_SPLIT_SIZE
    )


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        mixed_step_split_record([4, 500], [1], WIDTH)


def test_record_is_uint32() -> None:
    assert mixed_step_split_record([4, 500], [1, 2], WIDTH).dtype == np.uint32


def _manager(
    *, data_parallel_degree: int = 1, num_draft_tokens: int = 0
) -> PagedKVCacheManager:
    params = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=1,
        head_dim=16,
        num_layers=1,
        page_size=128,
        devices=[DeviceRef.CPU() for _ in range(data_parallel_degree)],
        data_parallel_degree=data_parallel_degree,
        num_draft_tokens=num_draft_tokens,
    )
    return PagedKVCacheManager(
        params=params,
        session=InferenceSession(devices=[CPU()]),
        total_num_pages=16,
        max_batch_size=8,
    )


def _leaves(
    manager: PagedKVCacheManager | JengaKVCacheManager,
    prompt_lengths: list[list[int]],
) -> list[KVCacheInputsPerDevice[Buffer, Buffer]]:
    batches = []
    for replica_idx, lengths in enumerate(prompt_lengths):
        batch = []
        for length in lengths:
            ctx = create_text_context(np.zeros(length, dtype=np.int64))
            manager.claim(ctx, replica_idx=replica_idx)
            manager.alloc(ctx)
            batch.append(ctx)
        batches.append(batch)
    return list(
        tree.leaves(
            manager.runtime_inputs(batches), leaf=KVCacheInputsPerDevice
        )
    )


def _record_of(leaf: KVCacheInputsPerDevice[Buffer, Buffer]) -> list[int]:
    assert leaf.mixed_step_split is not None
    return leaf.mixed_step_split.to_numpy().tolist()


def test_the_manager_fills_the_record_of_a_mixed_batch() -> None:
    (leaf,) = _leaves(_manager(), [[1, 1, 300]])
    assert _record_of(leaf) == [2, 2, 1, 1, 300, 300, 300]
    assert leaf.max_prompt_length.to_numpy().tolist() == [300]
    assert leaf.max_cache_length.to_numpy().tolist() == [300]


@pytest.mark.parametrize("lengths", [[1, 1, 1], [300, 200]])
def test_a_batch_without_a_split_gets_zeros(lengths: list[int]) -> None:
    (leaf,) = _leaves(_manager(), [lengths])
    assert _record_of(leaf) == [0] * MIXED_STEP_SPLIT_SIZE


def test_spec_widens_the_decode_view() -> None:
    # Spec widens the decode view, so a short prompt row rides it.
    (leaf,) = _leaves(_manager(num_draft_tokens=2), [[3, 3, 300]])
    assert _record_of(leaf)[:3] == [2, 6, 3]


@pytest.mark.parametrize("mixed_replica", [0, 1])
def test_each_replica_gets_its_own_record(mixed_replica: int) -> None:
    lengths = [[200, 200], [200, 200]]
    lengths[mixed_replica] = [1, 300]
    leaves = _leaves(_manager(data_parallel_degree=2), lengths)
    for replica, leaf in enumerate(leaves):
        want = (
            [1, 1, 1, 1, 300, 300, 300]
            if replica == mixed_replica
            else [0] * MIXED_STEP_SPLIT_SIZE
        )
        assert _record_of(leaf) == want, f"replica {replica}"


def test_the_switch_fills_zeros(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MIXED_STEP_SPLIT_SWITCH, "0")
    (leaf,) = _leaves(_manager(), [[1, 1, 300]])
    assert _record_of(leaf) == [0] * MIXED_STEP_SPLIT_SIZE
    assert leaf.max_prompt_length.to_numpy().tolist() == [300]


def _jenga_manager(*, num_draft_tokens: int = 0) -> JengaKVCacheManager:
    params = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=1,
        head_dim=16,
        num_layers=1,
        page_size=1,
        devices=[DeviceRef.CPU()],
        num_draft_tokens=num_draft_tokens,
    )
    return JengaKVCacheManager.create(
        params=params,
        available_bytes=1 << 20,
        max_batch_size=8,
        max_seq_len=512,
    )


def test_the_jenga_manager_fills_the_record_of_a_mixed_batch() -> None:
    (leaf,) = _leaves(_jenga_manager(), [[1, 1, 300]])
    assert _record_of(leaf) == [2, 2, 1, 1, 300, 300, 300]
    assert leaf.max_prompt_length.to_numpy().tolist() == [300]
    assert leaf.max_cache_length.to_numpy().tolist() == [300]


@pytest.mark.parametrize("lengths", [[1, 1, 1], [300, 200]])
def test_a_jenga_decode_only_batch_gets_zeros(lengths: list[int]) -> None:
    (leaf,) = _leaves(_jenga_manager(), [lengths])
    assert _record_of(leaf) == [0] * MIXED_STEP_SPLIT_SIZE
