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
"""The decoder's host-planned kept rows, and the switch that exports them.

Runs the real module methods eagerly on a one-device CPU mesh: no weights and
no kernels. The plans come from Bo's planner (``deepseekV41.replay``), the
same one the Mach engine's Rust planner is pinned to.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import abi_contract
import numpy as np
import numpy.typing as npt
import pytest
from max import functional as F
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.legacy.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.deepseekV41.replay import (
    ReplayPlan,
    build_replay_plan,
)
from max.pipelines.architectures.deepseekV41_modulev3.batch_processor import (
    DeepseekV41BatchProcessor,
    keep_all_plan,
)
from max.pipelines.architectures.deepseekV41_modulev3.deepseekV41 import (
    Batch,
    DeepseekV41,
    _replay_decoder_start,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    V41Config,
    build_kv_params,
    extra_input_names,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.attention import (
    RaggedRows,
    absolute_positions,
    row_ids,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
)
from max.pipelines.context import TextContext
from max.pipelines.context.tokens import TokenBuffer
from max.pipelines.lib import KVCacheConfig
from max.tensor import Tensor, default_device
from max.tensor.sharding import DeviceMesh

WINDOW = 128
DRAFT_WIDTH = 5
MIN_TOKENS = 256


def _text(**overrides: object) -> SimpleNamespace:
    text = SimpleNamespace(**dict(abi_contract.hf_config("f1"))["text_config"])
    for key, value in overrides.items():
        setattr(text, key, value)
    return text


def _config(text: SimpleNamespace) -> DeepseekV41Config:
    interface = V41Config.from_hf({"text_config": vars(text)})
    kv = build_kv_params(
        interface, KVCacheConfig(kv_cache_page_size=128), [DeviceRef.CPU()]
    )
    return DeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=kv,
        devices=[DeviceRef.CPU()],
        max_seq_len=256,
        **DeepseekV41Config.resolve_text_fields(text),
    )


def _module() -> DeepseekV41:
    """The trunk shell the replay methods need; no weights, no layers."""
    module = object.__new__(DeepseekV41)
    module.config = _config(_text())
    module.mesh = DeviceMesh((CPU(),), (1,), ("tp",))
    module.dp = 1
    return module


def _t(array: npt.ArrayLike, dtype: DType) -> Tensor:
    return Tensor.from_dlpack(np.ascontiguousarray(array)).cast(dtype)


def _context(start: int, length: int) -> TextContext:
    tokens = TokenBuffer(np.arange(start + length, dtype=np.int64))
    tokens.skip_processing(start)
    return TextContext(max_length=1_000_000, tokens=tokens)


def _batch(merged: list[int], cache: list[int]) -> Batch:
    """A staged batch over ``merged`` rows per request, one leaf.

    Nothing stores or attends here; every field the cut reads (cache lengths,
    the prompt bound) carries real values.
    """
    total = sum(merged)
    batch = len(merged)
    offsets = _t(np.cumsum([0, *merged]), DType.uint32)
    leaf = PagedCacheValues(
        kv_blocks=Tensor.from_dlpack(np.zeros((1, 1, 1, 4, 1, 512), np.uint8)),
        cache_lengths=_t(cache, DType.uint32),
        lookup_table=_t(np.zeros((batch, 2)), DType.uint32),
        max_prompt_length=_t([max(merged)], DType.uint32),
        max_cache_length=_t([max(cache)], DType.uint32),
        page_stride=_t([1], DType.int64),
    )
    positions = absolute_positions(offsets, leaf.cache_lengths, total)
    return Batch(
        tokens=_t(np.arange(total), DType.int64),
        rows=RaggedRows(
            offsets,
            *row_ids(offsets, total),
            positions,
            F.cast(offsets[1:] - offsets[:-1], DType.int32),
        ),
        swa_begin=_t(np.zeros(batch), DType.int32),
        global_write_floor=_t(np.zeros(batch), DType.int32),
        image_token_mask=_t(np.zeros(total), DType.uint8),
        lookback=_t(np.zeros((batch, 3)), DType.int32),
        lookback_dead=_t(np.zeros((batch, 3)), DType.uint8),
        tables=[],
        kv={"swa": leaf},
    )


def _replay_batch(b: Batch, plan: ReplayPlan) -> Batch:
    """``b`` with the plan staged, as ``batch_from_parts`` would."""
    extras = {
        "replay.kept_offsets": _t(plan.kept_offsets, DType.uint32),
        "replay.kept_skip": _t(plan.kept_skip, DType.int32),
        "replay.row_indices": _t(plan.kept_indices, DType.int32),
        "replay.rank_longest": _t(
            plan.kept_longest([0, len(plan.kept_skip)]), DType.uint32
        ),
    }
    replay = _module()._replay_rows(
        b.rows, b.swa_begin, b.kv["swa"].cache_lengths, extras, None
    )
    return replace(b, replay=replay)


def _cut(b: Batch, plan: ReplayPlan) -> Batch:
    """The batch the decoder runs under the plan."""
    return _module()._cut_to_replay(_replay_batch(b, plan))


def test_trim_plan_keeps_its_suffix_at_absolute_positions() -> None:
    """A planned step keeps the plan's rows with their full-batch coordinates."""
    plan = build_replay_plan(
        [_context(0, 300)], window_size=WINDOW, min_tokens=MIN_TOKENS
    )
    assert plan is not None
    assert plan.kept_offsets.tolist() == [0, 172]
    with default_device(CPU()):
        cut = _cut(_batch([300], [0]), plan)

    np.testing.assert_array_equal(cut.row_offsets.to_numpy(), [0, 172])
    np.testing.assert_array_equal(cut.positions.to_numpy(), np.arange(128, 300))
    np.testing.assert_array_equal(cut.kv["swa"].cache_lengths.to_numpy(), [128])
    np.testing.assert_array_equal(
        cut.kv["swa"].max_prompt_length.to_numpy(), [172]
    )
    # No window KV below the first kept row was written this step.
    np.testing.assert_array_equal(cut.swa_begin.to_numpy(), [128])


def test_continuation_chunk_keeps_the_previous_tail() -> None:
    """A later chunk's skip is relative to its own start, not absolute."""
    contexts = [_context(1024, 4096), _context(4096, 3)]
    plan = build_replay_plan(
        contexts, window_size=WINDOW, min_tokens=MIN_TOKENS
    )
    assert plan is not None
    with default_device(CPU()):
        cut = _cut(_batch([4096, 3], [1024, 4096]), plan)

    # End 5120 is window-aligned: the request keeps its last 128 rows, the
    # short continuation keeps all three.
    np.testing.assert_array_equal(cut.row_offsets.to_numpy(), [0, 128, 131])
    np.testing.assert_array_equal(
        cut.kv["swa"].cache_lengths.to_numpy(), [1024 + 3968, 4096]
    )
    # The untrimmed request's bound is its own window visibility, so its
    # kept rows still read the previous chunk's window through the cache.
    assert cut.swa_begin.to_numpy()[1] == 4096 - WINDOW


def test_long_chunk_outputs_gather_at_the_kept_offsets() -> None:
    """The last-row gather must read the kept offsets, not the original ones.

    A chunk much longer than ``window + draft_width`` keeps only a tail, so
    the original ``rows.offsets`` index past the kept hidden states. The
    hidden rows here carry their source row, so the gather returns exactly
    each request's last merged row only when it uses ``row_offsets``.
    """
    lengths = [700, 40, 1]
    plan = build_replay_plan(
        [_context(0, n) for n in lengths],
        window_size=WINDOW,
        min_tokens=MIN_TOKENS,
        verify_width=DRAFT_WIDTH,
    )
    assert plan is not None
    merged = [n + DRAFT_WIDTH for n in lengths]
    assert max(merged) > WINDOW + DRAFT_WIDTH
    with default_device(CPU()):
        b = _batch(merged, [0] * len(lengths))
        staged = _replay_batch(b, plan)
        hidden = _t(plan.kept_indices, DType.int32).reshape([-1, 1])
        got = DeepseekV41._gather_last_tokens(hidden, staged.row_offsets)
    np.testing.assert_array_equal(got.to_numpy().ravel(), np.cumsum(merged) - 1)
    assert staged.row_offsets.to_numpy().tolist() != (
        b.rows.offsets.to_numpy().tolist()
    )
    # The kept rows hold the plan's rows only: 128 + 700 % 128 + 5.
    assert hidden.shape[0] == int(plan.kept_offsets[-1]) < sum(merged)


def test_keep_all_plan_runs_the_full_batch() -> None:
    """The plan a step feeds when nothing trims leaves every row where it was."""
    contexts = [_context(0, 257), _context(1024, 40)]
    plan = keep_all_plan(contexts, DRAFT_WIDTH)
    merged = [257 + DRAFT_WIDTH, 40 + DRAFT_WIDTH]
    with default_device(CPU()):
        b = _batch(merged, [0, 1024])
        cut = _cut(b, plan)
    np.testing.assert_array_equal(
        cut.row_offsets.to_numpy(), b.rows.offsets.to_numpy()
    )
    np.testing.assert_array_equal(
        cut.positions.to_numpy(), b.positions.to_numpy()
    )
    np.testing.assert_array_equal(
        cut.kv["swa"].cache_lengths.to_numpy(), [0, 1024]
    )
    # Nothing dropped, so no read moves: the bound never rises above a kept
    # row's own window start.
    counts = np.diff(cut.row_offsets.to_numpy())
    bound = np.repeat(cut.swa_begin.to_numpy(), counts)
    np.testing.assert_array_equal(
        np.maximum(bound, cut.positions.to_numpy() - WINDOW + 1),
        np.maximum(0, b.positions.to_numpy() - WINDOW + 1),
    )


def test_default_export_declares_no_replay_input() -> None:
    off = V41Config.from_hf({"text_config": vars(_text())})
    on = V41Config.from_hf({"text_config": vars(_text(decoder_replay=True))})
    assert not off.decoder_replay
    assert not any(n.startswith("replay.") for n in extra_input_names(off))
    assert [n for n in extra_input_names(on) if n.startswith("replay.")] == [
        "replay.kept_offsets",
        "replay.kept_skip",
        "replay.row_indices",
        "replay.rank_longest",
    ]
    assert extra_input_names(off) == [
        n for n in extra_input_names(on) if not n.startswith("replay.")
    ]
    assert extra_input_names(on, 4)[-2:] == [
        "replay.host_kept_offsets",
        "replay.rank_bases",
    ]


def test_replay_rejects_configs_that_need_every_row() -> None:
    config = replace(
        _config(_text()), engram_layer_ids=[], dspark_target_layer_ids=[]
    )
    start = max(config.kv_source_layer_ids) + 1
    assert _replay_decoder_start(config) == start
    with pytest.raises(ValueError, match="all-row"):
        _replay_decoder_start(replace(config, return_logits=ReturnLogits.ALL))
    with pytest.raises(ValueError, match="all-row"):
        _replay_decoder_start(
            replace(
                config, return_hidden_states=ReturnHiddenStates.ALL_NORMALIZED
            )
        )
    with pytest.raises(ValueError, match="Engram"):
        _replay_decoder_start(replace(config, engram_layer_ids=[start]))
    with pytest.raises(ValueError, match="tap"):
        _replay_decoder_start(
            replace(config, dspark_target_layer_ids=[start - 1])
        )
    with pytest.raises(ValueError, match="layer after"):
        _replay_decoder_start(
            replace(config, kv_source_layer_ids=[config.num_hidden_layers - 1])
        )
    with pytest.raises(ValueError, match="KV source"):
        _replay_decoder_start(replace(config, kv_source_layer_ids=[]))


def test_trunk_without_replay_builds_without_kv_source_layers() -> None:
    """The replay switch alone decides whether the trunk needs a cut layer."""
    text = _text(
        compress_ratios=[0] * _text().num_hidden_layers,
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        candidate_source_layer_id=-1,
        engram_layer_ids=[],
        engram_num_embeddings=[],
    )
    config = _config(text)
    mesh = DeviceMesh((CPU(),), (1,), ("tp",))
    kv = build_kv_params(
        V41Config.from_hf({"text_config": vars(text)}),
        KVCacheConfig(kv_cache_page_size=128),
        [DeviceRef.CPU()],
    )
    with F.lazy(), default_device(mesh):
        module = DeepseekV41(
            config,
            kv,
            mesh,
            text_config=text,
            token_map=np.arange(text.vocab_size, dtype=np.int32),
        )
    assert module.decoder_start is None


def test_batch_processor_stages_the_keep_all_plan_only_when_exported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both exports run through the same processor: the plan inputs exist iff on."""
    contexts = [_context(0, 6), _context(4, 1)]
    offsets = np.array([0, 6, 7])
    for on in (False, True):
        text = _text(
            decoder_replay=on, engram_layer_ids=[], engram_num_embeddings=[]
        )
        processor = object.__new__(DeepseekV41BatchProcessor)
        processor._v41 = V41Config.from_hf({"text_config": vars(text)})
        processor._engram_static = {
            "engram_token_map": Buffer.from_numpy(np.zeros(1, np.int32))
        }
        processor._verifies = False
        processor._dp_degree = 1
        monkeypatch.setattr(
            processor,
            "runtime",
            SimpleNamespace(devices=[CPU()]),
            raising=False,
        )
        extras = dict(
            zip(
                extra_input_names(processor._v41),
                processor._extras(contexts, offsets),
                strict=True,
            )
        )
        assert any(name.startswith("replay.") for name in extras) == on
        if on:
            np.testing.assert_array_equal(
                extras["replay.kept_offsets"].to_numpy(), [0, 6, 7]
            )
            np.testing.assert_array_equal(
                extras["replay.kept_skip"].to_numpy(), [0, 0]
            )
            np.testing.assert_array_equal(
                extras["replay.row_indices"].to_numpy(), np.arange(7)
            )
            np.testing.assert_array_equal(
                extras["replay.rank_longest"].to_numpy(), [6]
            )
