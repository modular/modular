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
"""Checks DSpark decoder replay: the verify, the gate and the batch hooks."""

from __future__ import annotations

from functools import cache
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.legacy.kv_cache import PagedCacheValues
from max.nn.legacy.transformer import ReturnHiddenStates
from max.pipelines.architectures.deepseekV41 import spec_adapters as adapters
from max.pipelines.architectures.deepseekV41.dspark_drafter import (
    DeepseekV41DSparkDrafter,
)
from max.pipelines.architectures.deepseekV41.layers.ragged import (
    ReplayRows,
    ReplicaTokens,
)
from max.pipelines.architectures.deepseekV41.model import (
    DeepseekV41BatchProcessor,
    DeepseekV41Inputs,
)
from max.pipelines.architectures.deepseekV41.model_config import (
    DeepseekV41Config,
)
from max.pipelines.architectures.deepseekV41.replay import (
    build_replay_plan,
    replay_min_tokens,
)
from max.pipelines.architectures.deepseekV41.replay_writes import (
    DSparkWriteBatch,
)
from max.pipelines.architectures.deepseekV41.spec_adapters import (
    TrunkTaps,
    _DSparkProposer,
    _Verify,
    local_replay_rows,
)
from max.pipelines.context import TextContext
from max.pipelines.context.tokens import TokenBuffer
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    SpecDecodePipelineModel,
    _BatchProcessor,
    _Inputs,
)

ENV = "MODULAR_DEEPSEEK_V41_REPLAY_MIN_TOKENS"


def _config(taps: list[int] | None = None) -> DeepseekV41Config:
    config = object.__new__(DeepseekV41Config)
    config.num_hidden_layers = 40
    config.kv_source_layer_ids = [0, 20]
    config.dspark_target_layer_ids = [37, 38, 39] if taps is None else taps
    config.sliding_window = 128
    config.engram_layer_ids = []
    return config


@cache
def mapping_graph() -> Model:
    cpu = DeviceRef.CPU()
    with Graph(
        "dspark_replay_dp_mapping",
        input_types=[
            TensorType(DType.int64, ["k"], cpu),
            TensorType(DType.uint32, ["b1"], cpu),
            TensorType(DType.int32, ["b"], cpu),
            TensorType(DType.uint32, ["b"], cpu),
            TensorType(DType.uint32, ["b1"], cpu),
            TensorType(DType.int64, [3], cpu),
            TensorType(DType.uint32, [2], cpu),
        ],
    ) as graph:
        (
            indices,
            offsets,
            skip,
            cache_lengths,
            original_offsets,
            splits,
            longest,
        ) = [v.tensor for v in graph.inputs]
        kept = ReplicaTokens.from_splits(offsets, splits, indices.shape[0], 2)
        original = ReplicaTokens.from_splits(original_offsets, splits, 12, 2)
        # Each device's cache lengths cover only its own replica's requests.
        replica_lengths = [
            cache_lengths[
                ((slice(splits[i], splits[i + 1]), f"replica_lengths_{i}"),)
            ]
            for i in range(2)
        ]
        local = local_replay_rows(
            [indices, indices],
            [skip, skip],
            offsets,
            splits,
            kept,
            original,
            replica_lengths,
            128,
            longest,
        )
        graph.output(
            *(
                value
                for item in local
                for value in (
                    item.indices,
                    item.rows.offsets,
                    item.rows.starts,
                    item.swa_valid_from,
                    item.longest,
                )
            )
        )
    return InferenceSession(devices=[CPU()]).load(graph)


@pytest.mark.parametrize(
    ("splits", "longest"),
    [([0, 1, 3], [2, 1]), ([0, 0, 3], [0, 2]), ([0, 3, 3], [2, 0])],
)
def test_replica_major_mapping_including_empty_replica(
    splits: list[int], longest: list[int]
) -> None:
    indices = np.array([2, 3, 8, 11], dtype=np.int64)
    offsets = np.array([0, 2, 3, 4], dtype=np.uint32)
    skip = np.array([30, 14, 0], dtype=np.int32)
    cache_lengths = np.array([100, 250, 499], dtype=np.uint32)
    # A skipped row starts at its cache length plus the skip and floors there;
    # a kept row floors one window below its cache length.
    starts = np.array([130, 264, 499], dtype=np.int32)
    floor = np.array([130, 264, 371], dtype=np.int32)
    original = np.array([0, 4, 9, 12], dtype=np.uint32)
    outputs = mapping_graph().execute(
        *(
            Buffer.from_numpy(array)
            for array in (
                indices,
                offsets,
                skip,
                cache_lengths,
                original,
                np.array(splits, dtype=np.int64),
                np.array(longest, dtype=np.uint32),
            )
        )
    )
    for i in range(2):
        a, b = splits[i : i + 2]
        lo, hi = int(offsets[a]), int(offsets[b])
        expected = (
            indices[lo:hi] - original[a],
            offsets[a : b + 1] - offsets[a],
            starts[a:b],
            floor[a:b],
            longest[i : i + 1],
        )
        for out, reference in zip(
            outputs[i * 5 : i * 5 + 5], expected, strict=True
        ):
            np.testing.assert_array_equal(out.to_numpy(), reference)


def test_materialize_preserves_verified_kept_rows() -> None:
    proposer = object.__new__(_DSparkProposer)
    stages = [Mock(), Mock()]
    proposer._stages = cast(list[DeepseekV41DSparkDrafter], stages)
    taps, rows, caches = [Mock(), Mock()], [Mock(), Mock()], [Mock(), Mock()]
    proposer.materialize(
        Mock(),
        cast(TrunkTaps, (taps, rows)),
        cast(list[PagedCacheValues], caches),
    )
    for stage, tap, row, window_cache in zip(
        stages, taps, rows, caches, strict=True
    ):
        stage.store_context.assert_called_once_with(tap, row, window_cache)


def test_full_inputs_have_no_replay_buffers() -> None:
    inputs = object.__new__(DeepseekV41Inputs)
    assert inputs.replay_plan is None
    assert inputs.replay_buffers == ()


@pytest.mark.parametrize("threshold", [None, "0"])
def test_replay_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch, threshold: str | None
) -> None:
    if threshold is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, threshold)
    assert replay_min_tokens(_config(), None) == (0, None)
    processor = object.__new__(DeepseekV41BatchProcessor)
    assert processor.replay_min_tokens == 0


@pytest.mark.parametrize(
    "capture, expected", [(False, 256), (True, 256), (None, 0)]
)
def test_replay_needs_resolved_capture(
    monkeypatch: pytest.MonkeyPatch, capture: bool | None, expected: int
) -> None:
    monkeypatch.setenv(ENV, "256")
    assert replay_min_tokens(_config(), capture)[0] == expected


@pytest.mark.parametrize("taps", [[20, 38, 39], [37, 38, 40], []])
def test_replay_requires_all_taps_after_cut(
    monkeypatch: pytest.MonkeyPatch, taps: list[int]
) -> None:
    monkeypatch.setenv(ENV, "256")
    threshold, why = replay_min_tokens(_config(taps), False)
    assert threshold == 0 and why is not None


def test_replay_rejects_negative_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV, "-1")
    with pytest.raises(ValueError, match="nonnegative"):
        replay_min_tokens(_config(), False)


@pytest.mark.parametrize("return_n_logits", [1, 3])
def test_a_decode_step_is_never_trimmed(
    monkeypatch: pytest.MonkeyPatch, return_n_logits: int
) -> None:
    monkeypatch.setenv(ENV, "256")
    decoding = TextContext(
        max_length=100000, tokens=TokenBuffer(np.arange(4096, dtype=np.int64))
    )
    decoding.update(42)
    assert not decoding.is_initial_prompt
    assert (
        build_replay_plan(
            [decoding],
            window_size=128,
            min_tokens=256,
            verify_width=return_n_logits - 1,
        )
        is None
    )


def _inputs(
    width: int, plan_width: int | None = None
) -> tuple[DeepseekV41Inputs, TextContext]:
    ctx = TextContext(
        max_length=100000, tokens=TokenBuffer(np.arange(4096, dtype=np.int64))
    )
    plan = build_replay_plan(
        [ctx], window_size=128, min_tokens=256, verify_width=plan_width or width
    )
    assert plan is not None
    inputs = object.__new__(DeepseekV41Inputs)
    inputs.replay_plan = plan
    inputs.replay_buffers = (Mock(),)
    inputs.replay_write_batch = DSparkWriteBatch.capture(
        [ctx], plan, leaf_id="swa", verified_width=width, previous=()
    )
    inputs.tokens = Buffer.from_numpy(np.zeros(4096, dtype=np.int64))
    return inputs, ctx


def test_settle_engages_replay_when_widths_agree() -> None:
    processor = object.__new__(DeepseekV41BatchProcessor)
    inputs, _ = _inputs(width=4)
    assert processor.settle_replay(inputs, 4)
    assert inputs.replay_plan is not None
    assert processor._submitted_replay_writes is None
    processor.record_submission(inputs)
    assert processor._submitted_replay_writes is inputs.replay_write_batch


def test_settle_drops_a_plan_whose_snapshot_has_another_width() -> None:
    processor = object.__new__(DeepseekV41BatchProcessor)
    inputs, _ = _inputs(width=4)
    assert not processor.settle_replay(inputs, 2)
    assert inputs.replay_plan is None
    assert inputs.replay_buffers == ()
    assert inputs.replay_write_batch is None
    processor.record_submission(inputs)
    assert processor._submitted_replay_writes is None


def test_settle_rejects_a_plan_built_for_another_width() -> None:
    processor = object.__new__(DeepseekV41BatchProcessor)
    inputs, _ = _inputs(width=4, plan_width=2)
    with pytest.raises(RuntimeError, match="verify width"):
        processor.settle_replay(inputs, 4)


@pytest.mark.parametrize("capture", [False, True, None])
@pytest.mark.parametrize("threshold", [0, 256])
def test_prepare_only_records_opted_in_resolved_submissions(
    monkeypatch: pytest.MonkeyPatch, capture: bool | None, threshold: int
) -> None:
    monkeypatch.setenv(ENV, str(threshold))
    processor = object.__new__(DeepseekV41BatchProcessor)
    processor.config = _config()
    processor._replay_leaf_id = "swa"
    processor.runtime = Mock()
    processor.runtime.pipeline_config.runtime.device_graph_capture = capture
    processor.runtime.return_hidden_states = ReturnHiddenStates.NONE
    processor.runtime.pipeline_config.model.enable_echo = False
    inputs = object.__new__(DeepseekV41Inputs)
    inputs.replay_plan = None
    inputs.replay_buffers = ()
    inputs.replay_write_batch = None
    processor._prepare_replay(inputs, [[]], [], 3)
    assert (inputs.replay_write_batch is not None) is (
        threshold > 0 and capture is not None
    )
    assert processor._submitted_replay_writes is None
    assert processor._previous_replay_writes == ()
    if threshold == 0 or capture is None:
        assert inputs.replay_plan is None
        assert inputs.replay_buffers == ()


def test_replay_adapter_inputs_name_the_extra_keys() -> None:
    config = _config()
    config.devices = [DeviceRef.CPU()]
    config.data_parallel_degree = 2
    types = adapters.replay_input_types(config)
    assert len(types) == 5
    config.data_parallel_degree = 1
    assert len(adapters.replay_input_types(config)) == 4
    values = [Mock(tensor=i) for i in range(4)]
    assert adapters.replay_extra(config, values) == {
        "kept_indices": 0,
        "kept_offsets": 1,
        "kept_skip": 2,
        "kept_longest": 3,
    }
    with pytest.raises(ValueError):
        adapters.replay_extra(config, values[:3])


def test_replay_is_disabled_by_an_engram_layer_after_the_cut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV, "256")
    config = _config()
    config.engram_layer_ids = [1, 21]
    threshold, why = replay_min_tokens(config, False)
    assert threshold == 0 and why is not None and "Engram" in why
    config.engram_layer_ids = [1, 14, 20]
    assert replay_min_tokens(config, False) == (256, None)


def test_settle_treats_a_missing_draft_width_as_a_mismatch() -> None:
    processor = object.__new__(DeepseekV41BatchProcessor)
    inputs, _ = _inputs(width=4)
    assert not processor.settle_replay(inputs, None)
    assert inputs.replay_plan is None and inputs.replay_write_batch is None


def test_captured_success_without_a_draft_width_is_an_error() -> None:
    processor = object.__new__(DeepseekV41BatchProcessor)
    inputs, _ = _inputs(width=2)
    inputs.replay_plan = None
    inputs.replay_buffers = ()
    with pytest.raises(RuntimeError, match="verify width"):
        processor.on_captured_submission_success(inputs, None)
    assert processor._submitted_replay_writes is None


@pytest.mark.parametrize("data_parallel", [False, True])
def test_replay_buffers_match_the_declared_replay_inputs(
    monkeypatch: pytest.MonkeyPatch, data_parallel: bool
) -> None:
    monkeypatch.setenv(ENV, "256")
    config = _config()
    config.devices = [DeviceRef.CPU()]
    config.data_parallel_degree = 2 if data_parallel else 1
    processor = object.__new__(DeepseekV41BatchProcessor)
    processor.config = config
    processor._replay_leaf_id = "swa"
    processor.runtime = Mock()
    processor.runtime.devices = [CPU()]
    processor.runtime.pipeline_config.runtime.device_graph_capture = False
    processor.runtime.return_hidden_states = ReturnHiddenStates.NONE
    processor.runtime.pipeline_config.model.enable_echo = False

    def prompt() -> TextContext:
        return TextContext(
            max_length=100000,
            tokens=TokenBuffer(np.arange(4096, dtype=np.int64)),
        )

    contexts = [prompt(), prompt()]
    batches = [[contexts[0]], [contexts[1]]] if data_parallel else [contexts]
    inputs = object.__new__(DeepseekV41Inputs)
    inputs.replay_plan = None
    inputs.replay_buffers = ()
    inputs.replay_write_batch = None
    processor._prepare_replay(inputs, batches, contexts, 3)
    assert inputs.replay_plan is not None
    buffers = adapters.batch_replay_buffers(inputs)
    types = adapters.replay_input_types(config)
    assert len(buffers) == len(types) == (5 if data_parallel else 4)
    for buffer, declared in zip(buffers, types, strict=True):
        assert buffer.dtype == declared.dtype
        assert DeviceRef.from_device(buffer.device) == declared.device


@pytest.mark.parametrize("launch_failure", [False, True])
def test_only_a_returned_replay_submission_reaches_finalize(
    monkeypatch: pytest.MonkeyPatch, launch_failure: bool
) -> None:
    dsv41 = object.__new__(DeepseekV41BatchProcessor)
    dsv41._previous_replay_writes = ()
    dsv41._submitted_replay_writes = None
    batching = object.__new__(_BatchProcessor)
    batching.__dict__["_target_batching"] = dsv41
    model = object.__new__(SpecDecodePipelineModel)
    model._batch_processor = batching
    model.model = Mock()
    model.replay_model = Mock()
    model.replay_model.execute.return_value = [Mock(), Mock(), Mock()]
    target, ctx = _inputs(width=2)
    inputs = object.__new__(_Inputs)
    inputs.target_inputs = target
    inputs.replay_trailing = [Mock()]
    inputs.draft_tokens = Buffer.from_numpy(np.empty((1, 2), dtype=np.int64))
    monkeypatch.setattr(_Inputs, "buffers", property(lambda _: ()))
    if launch_failure:
        model.replay_model.execute.side_effect = RuntimeError("launch failed")
        with pytest.raises(RuntimeError, match="launch failed"):
            model.execute(inputs)
    else:
        model.execute(inputs)
        assert dsv41._submitted_replay_writes is target.replay_write_batch
    ctx.update(42)
    model.finalize_kv_cache_writes([ctx])
    expected = {} if launch_failure else {"swa": (3968, 4096)}
    assert ctx.kv_cache_write_ranges == expected
    model.model.execute.assert_not_called()


def _block_batch(**fields: Any) -> adapters.BlockBatch:
    """A real BlockBatch holding only the fields the verify reads."""
    batch = object.__new__(adapters.BlockBatch)
    batch.__dict__.update(fields)
    return batch


@pytest.mark.parametrize(
    "data_parallel, replica_logits",
    [(False, False), (True, False), (True, True)],
)
@pytest.mark.parametrize("replay_enabled", [False, True])
def test_verify_reads_named_logits_and_preserves_selected_rows(
    monkeypatch: pytest.MonkeyPatch,
    data_parallel: bool,
    replica_logits: bool,
    replay_enabled: bool,
) -> None:
    cpu = DeviceRef.CPU()
    n = 2 if data_parallel else 1
    with Graph("dspark_named_verify", input_types=[]) as graph:
        offsets = ops.constant([0, 3], DType.uint32, cpu)
        kept_offsets = ops.constant([0, 2], DType.uint32, cpu)
        starts = ops.constant([0], DType.int32, cpu)
        kept = ReplayRows.from_inputs(
            ops.constant([1, 2], DType.int64, cpu),
            kept_offsets,
            ops.constant([1], DType.int32, cpu),
            starts,
            128,
            ops.constant([2], DType.uint32, cpu),
        )
        rows = adapters.RaggedRows.from_offsets(offsets, 3, starts)
        count = 2 if replay_enabled else 3
        hidden = ops.constant(
            np.arange(count * 2, dtype=np.float32).reshape(1, count, 2),
            DType.float32,
            cpu,
        )
        taps = [hidden] * n
        lead = SimpleNamespace(
            trunk=SimpleNamespace(
                forward_with_taps=Mock(return_value=(hidden, hidden)),
                layers=[],
            ),
            _engram_operands=Mock(return_value=None),
            norm=lambda x: x,
            _lm_head=lambda x: x,
            ep_batch_manager=None,
        )
        verify = object.__new__(_Verify)
        verify.config = cast(
            DeepseekV41Config,
            SimpleNamespace(
                hidden_size=2,
                dspark_target_layer_ids=[37, 38, 39],
                sliding_window=128,
                kv_leaf_specs=[],
            ),
        )
        verify.takes_distributed_inputs = data_parallel
        verify.replica_logits = replica_logits
        cache = SimpleNamespace(
            swa_collection=SimpleNamespace(
                cache_lengths=ops.constant([0], DType.uint32, cpu)
            )
        )
        monkeypatch.setattr(verify, "replicas", lambda _: [lead] * n)
        monkeypatch.setattr(adapters, "leaf_caches", lambda *_: [cache] * n)
        monkeypatch.setattr(
            adapters,
            "VisionMerge",
            lambda *_: SimpleNamespace(for_rows=lambda *_args: None),
        )
        monkeypatch.setattr(
            adapters, "broadcast_per_device", lambda value, *_: [value] * n
        )
        monkeypatch.setattr(
            adapters, "gather_rows", lambda *_: ops.reshape(hidden, [count, 2])
        )
        split = SimpleNamespace(
            tokens=[hidden] * n,
            batch_tokens=[hidden] * n,
            rows=[rows] * n,
            lookbacks=[None] * n,
            replica_tokens=Mock(),
        )
        monkeypatch.setattr(
            adapters.DeepseekV41,
            "split_replicas",
            lambda *_args, **_kwargs: split,
        )
        monkeypatch.setattr(
            adapters.DeepseekV41,
            "_engram_operands_data_parallel",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            adapters, "local_replay_rows", lambda *_: [kept] * n
        )
        dp_calls: list[dict[str, Any]] = []

        def data_parallel_with_taps(*_args: Any, **kwargs: Any) -> Any:
            dp_calls.append(kwargs)
            return [hidden] * n, taps

        monkeypatch.setattr(
            adapters.DeepseekV41Trunk,
            "data_parallel_with_taps",
            data_parallel_with_taps,
        )
        replica_calls: list[Any] = []

        def replica_verify_logits(*args: Any) -> list[Any]:
            replica_calls.append(args)
            # The real function keeps the last return_n_logits rows per request.
            return [ops.reshape(hidden, [count, 2])[count - 2 :]] * n

        monkeypatch.setattr(
            adapters, "replica_verify_logits", replica_verify_logits
        )
        extra: dict[str, Any] = {}
        if replay_enabled:
            # The driver adds the kept_* keys only to the replay graph's
            # batch, so the verify keys replay on their presence.
            extra = {
                "kept_indices": kept.indices,
                "kept_offsets": kept_offsets,
                "kept_skip": ops.constant([1], DType.int32, cpu),
                "kept_longest": ops.constant([2], DType.uint32, cpu),
                "host_kept_offsets": kept_offsets,
            }
        batch = _block_batch(
            kv_collections=[],
            passthrough_kv={},
            data_parallel_degree=n,
            devices=[cpu] * n,
            merged_tokens=ops.constant([0, 1, 2], DType.int64, cpu),
            merged_offsets=offsets,
            merged_offsets_per_dev=[offsets] * n,
            host_merged_offsets=offsets,
            data_parallel_splits=ops.constant([0, 1, 1], DType.int64, cpu),
            signal_buffers=[],
            vision_embeddings=[hidden] * n,
            vision_scatter_indices=[starts] * n,
            extra=extra,
            return_n_logits=ops.constant([2], DType.int64, cpu),
        )
        verified = verify(batch)
        assert verified.hidden is not None
        assert len(verified.hidden[1]) == n
        if replica_logits:
            ((_, _xs, offsets_per_dev, *_rest),) = replica_calls
            expected_offsets = kept_offsets if replay_enabled else offsets
            assert all(o is expected_offsets for o in offsets_per_dev)
            assert verified.replica_logits is not None
            assert len(verified.replica_logits) == n
        if data_parallel:
            (kwargs,) = dp_calls
            if replay_enabled:
                assert all(row is kept.rows for row in verified.hidden[1])
                assert all(r is kept for r in kwargs["replay"])
                assert len(kwargs["batch_kept_indices"]) == n
                assert all(
                    i is kept.indices for i in kwargs["batch_kept_indices"]
                )
                assert isinstance(kwargs["kept_replica_tokens"], ReplicaTokens)
            else:
                assert kwargs["replay"] is None
                assert kwargs["batch_kept_indices"] is None
                assert kwargs["kept_replica_tokens"] is None
        else:
            forwarded = lead.trunk.forward_with_taps.call_args.kwargs["replay"]
            if replay_enabled:
                assert isinstance(forwarded, ReplayRows)
                assert forwarded.rows is verified.hidden[1][0]
            else:
                assert forwarded is None
        graph.output(verified.logits)
    (output,) = InferenceSession(devices=[CPU()]).load(graph).execute()
    expected = np.arange(count * 2, dtype=np.float32).reshape(count, 2)[-2:]
    np.testing.assert_array_equal(output.to_numpy(), expected)
