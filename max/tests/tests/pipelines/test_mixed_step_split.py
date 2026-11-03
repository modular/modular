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
"""The host rule that cuts a mixed speculative step for split-route ops."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn.legacy.kernels import _MSA_SPLIT_ARGS_SIZE, _msa_split_args_value
from max.nn.legacy.kv_cache import (
    KVCacheInputs,
    MHAKVCacheParams,
    MultiKVCacheParams,
)
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _Inputs,
)
from max.pipelines.modeling.types import RequestID
from max.pipelines.speculative._mixed_step_split import (
    fill_mixed_step_split,
    mixed_step_decode_width,
    mixed_step_split_buffers,
    mixed_step_split_record,
    mixed_step_split_record_for_contexts,
)
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.spec_input_types import (
    SpecDecodeInputTypeSpec,
    _mixed_step_split_input_types,
    build_spec_decode_input_types,
    decode_spec_decode_input_values,
)

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
    # Two trailing decode rows beat one leading one.
    assert record[0] == 2
    assert record[4] == WIDTH


def test_a_decode_row_after_a_prompt_row_stays_in_the_prefill_view() -> None:
    record = _record([4, 604, 4, 700], [10, 20, 30, 40])
    assert record[:2] == [1, 4]
    assert record[6] == 20 + 30 + 40


def test_a_one_token_chunk_rides_the_decode_view() -> None:
    # A prompt chunk of one token is 1 + K wide, like a verify row.
    assert _record([4, 4, 300], [1, 2, 3])[0] == 2


def test_short_prompt_rows_wider_than_verify_stay_in_the_prefill_view() -> None:
    assert _record([4, 6, 300], [1, 2, 3])[0] == 1


@pytest.mark.parametrize(
    "widths",
    [[4, 4, 4], [600, 700], [4, 8], [600, 4, 700]],
    ids=["all_decode", "all_prompt", "all_spec_width", "decode_in_middle"],
)
def test_no_split_gives_zeros(widths: list[int]) -> None:
    assert _record(widths, [1] * len(widths)) == [0] * _MSA_SPLIT_ARGS_SIZE


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
    # two-token chunk (nine tokens) is one token too wide for the decode view.
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
        == [0] * _MSA_SPLIT_ARGS_SIZE
    )


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        mixed_step_split_record([4, 500], [1], WIDTH)


def test_record_is_uint32() -> None:
    assert mixed_step_split_record([4, 500], [1, 2], WIDTH).dtype == np.uint32


def test_the_option_is_off_by_default() -> None:
    field = SpeculativeConfig.model_fields["mixed_step_split_inputs"]
    assert field.default is False


def test_the_spec_declares_no_records_by_default() -> None:
    spec = SpecDecodeInputTypeSpec(
        devices=[DeviceRef.GPU(0), DeviceRef.GPU(1)], distributed=True
    )
    assert _mixed_step_split_input_types(spec) == []


def test_the_spec_declares_one_host_record_per_device() -> None:
    devices = [DeviceRef.GPU(i) for i in range(8)]
    spec = SpecDecodeInputTypeSpec(
        devices=devices,
        distributed=True,
        data_parallel_degree=2,
        mixed_step_split=True,
    )
    records = _mixed_step_split_input_types(spec)
    assert len(records) == len(devices)
    for record in records:
        assert record.dtype == DType.uint32
        assert [int(d) for d in record.shape] == [_MSA_SPLIT_ARGS_SIZE]
        assert record.device == DeviceRef.CPU()


def _row(prompt_len: int, drafts: int) -> TextContext:
    ctx = TextContext(
        request_id=RequestID(),
        max_length=4_096,
        tokens=TokenBuffer(np.arange(prompt_len, dtype=np.int64)),
    )
    ctx.spec_decoding_state.draft_tokens_to_verify = list(range(drafts))
    return ctx


def test_fill_writes_each_replicas_record_on_its_devices() -> None:
    mixed = [_row(600, K), _row(1, K)]
    decode_only = [_row(1, K), _row(1, K)]
    records = mixed_step_split_buffers(4)
    for buffer in records:
        buffer.to_numpy()[:] = 99
    fill_mixed_step_split(records, [mixed, decode_only], K)
    expected = mixed_step_split_record_for_contexts(mixed, K)
    assert expected[0] == 1, "the prompt row is the head view"
    for buffer in records[:2]:
        assert buffer.to_numpy().tolist() == expected.tolist()
    for buffer in records[2:]:
        assert buffer.to_numpy().tolist() == [0] * _MSA_SPLIT_ARGS_SIZE


def _decode_graph_spec(
    *, on_kv_leaves: bool = False
) -> tuple[SpecDecodeInputTypeSpec, MHAKVCacheParams]:
    spec = SpecDecodeInputTypeSpec(
        devices=[DeviceRef.CPU()],
        distributed=False,
        mixed_step_split=True,
        mixed_step_split_on_kv_leaves=on_kv_leaves,
    )
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=64,
        num_layers=1,
        devices=[DeviceRef.CPU()],
        n_kv_heads=1,
    )
    return spec, kv_params


def test_the_records_decode_after_a_models_trailing_group() -> None:
    spec, kv_params = _decode_graph_spec()
    own = TensorType(DType.int64, [3], device=DeviceRef.CPU())
    input_types = [
        *build_spec_decode_input_types(spec, kv_params=kv_params),
        own,
        *_mixed_step_split_input_types(spec),
    ]
    with Graph("decode", input_types=input_types) as graph:
        decoded = decode_spec_decode_input_values(
            graph.inputs, spec, kv_params=kv_params, allow_trailing=True
        )
        assert [v.type for v in decoded.trailing] == [own]
        assert [v.type for v in decoded.mixed_step_split] == list(
            _mixed_step_split_input_types(spec)
        )


def test_the_records_ride_on_the_target_kv_leaves_only() -> None:
    spec, kv_params = _decode_graph_spec(on_kv_leaves=True)
    unified = MultiKVCacheParams.from_params(
        {"target": kv_params, "draft": kv_params}
    )
    input_types = [
        *build_spec_decode_input_types(spec, kv_params=unified),
        *_mixed_step_split_input_types(spec),
    ]
    with Graph("decode", input_types=input_types) as graph:
        decoded = decode_spec_decode_input_values(
            graph.inputs, spec, kv_params=unified
        )
        (target,) = decoded.kv("target")
        (draft,) = decoded.kv("draft")
        assert target.mixed_step_split is decoded.mixed_step_split[0]
        assert draft.mixed_step_split is None


def test_declared_records_leave_the_kv_leaves_bare_without_the_opt_in() -> None:
    """A target that reads the records itself must keep its unsplit ops."""
    spec, kv_params = _decode_graph_spec()
    unified = MultiKVCacheParams.from_params(
        {"target": kv_params, "draft": kv_params}
    )
    for params in (kv_params, unified):
        input_types = [
            *build_spec_decode_input_types(spec, kv_params=params),
            *_mixed_step_split_input_types(spec),
        ]
        with Graph("decode", input_types=input_types) as graph:
            decoded = decode_spec_decode_input_values(
                graph.inputs, spec, kv_params=params
            )
            assert len(decoded.mixed_step_split) == 1
            paths = [("target",), ("draft",)] if params is unified else [()]
            leaves = [leaf for path in paths for leaf in decoded.kv(*path)]
            assert leaves
            assert all(leaf.mixed_step_split is None for leaf in leaves)


@pytest.mark.parametrize("records", [False, True], ids=["off", "on"])
@pytest.mark.parametrize(
    "distributed, traced, signal_index",
    [
        (False, False, None),
        (True, False, None),
        (False, True, None),
        (True, True, None),
        (False, True, 2),
        (True, True, 4),
    ],
    ids=[
        "plain",
        "distributed",
        "traced",
        "distributed_traced",
        "traced_signals_placed",
        "distributed_traced_signals_placed",
    ],
)
def test_the_batch_packs_the_records_where_the_graph_declares_them(
    distributed: bool, traced: bool, signal_index: int | None, records: bool
) -> None:
    """The records follow the trailing and replay inputs; a traced graph's
    signal buffers keep their own place."""
    devices = [DeviceRef.CPU()]
    spec = SpecDecodeInputTypeSpec(
        devices=devices,
        distributed=distributed,
        include_signal_buffers=True,
        trace_owns_signal_buffers=traced,
        signal_buffer_index=signal_index,
        mixed_step_split=records,
    )
    _, kv_params = _decode_graph_spec()
    own = TensorType(DType.int64, [3], device=DeviceRef.CPU())
    replay = TensorType(DType.int64, [2], device=DeviceRef.CPU())
    declared = [
        *build_spec_decode_input_types(spec, kv_params=kv_params),
        own,
        replay,
        *_mixed_step_split_input_types(spec),
    ]

    def buffer() -> Buffer:
        return Buffer.zeros(shape=[1], dtype=DType.int64)

    signals = [buffer() for _ in devices]
    split = mixed_step_split_buffers(len(devices)) if records else []
    inputs = _Inputs(
        tokens=buffer(),
        input_row_offsets=buffer(),
        return_n_logits=buffer(),
        signal_buffers=signals,
        host_input_row_offsets=buffer(),
        data_parallel_splits=buffer(),
        batch_context_lengths=[buffer() for _ in devices],
        kv_cache_inputs=cast(
            "KVCacheInputs[Buffer, Buffer]",
            {"kv": [buffer() for _ in kv_params.flattened_kv_inputs()]},
        ),
        trailing=[buffer()],
        replay_trailing=[buffer()],
        draft_tokens=buffer(),
        seed=buffer(),
        temperature=buffer(),
        top_k=buffer(),
        max_k=buffer(),
        top_p=buffer(),
        min_top_p=buffer(),
        mixed_step_split=split,
        spec=spec,
    )

    packed = inputs.buffers
    placed = signals if traced else []
    assert len(packed) == len(declared) + len(placed)
    at = len(declared) if signal_index is None else signal_index
    assert all(
        got is want
        for got, want in zip(packed[at : at + len(placed)], placed, strict=True)
    )
    first_record = len(declared) - len(split)
    if at <= first_record:
        first_record += len(placed)
    assert all(
        got is want
        for got, want in zip(
            packed[first_record : first_record + len(split)], split, strict=True
        )
    )


def test_a_graph_without_its_records_is_refused_at_decode() -> None:
    spec, kv_params = _decode_graph_spec()
    input_types = build_spec_decode_input_types(spec, kv_params=kv_params)
    with Graph("decode", input_types=input_types) as graph:
        with pytest.raises(ValueError, match="mixed-step split records"):
            decode_spec_decode_input_values(
                graph.inputs, spec, kv_params=kv_params
            )


@pytest.mark.parametrize(
    "size", [_MSA_SPLIT_ARGS_SIZE - 1, _MSA_SPLIT_ARGS_SIZE + 1]
)
def test_a_record_of_another_size_is_refused(size: int) -> None:
    record = TensorType(DType.uint32, [size], device=DeviceRef.CPU())
    with Graph("split_args", input_types=[record]) as graph:
        with pytest.raises(
            ValueError, match=f"{_MSA_SPLIT_ARGS_SIZE} elements"
        ):
            _msa_split_args_value(graph.inputs[0].tensor)


@pytest.mark.parametrize(
    "record, field",
    [
        (
            TensorType(DType.int32, [_MSA_SPLIT_ARGS_SIZE], DeviceRef.CPU()),
            "dtype",
        ),
        (
            TensorType(
                DType.uint32, [1, _MSA_SPLIT_ARGS_SIZE], DeviceRef.CPU()
            ),
            "rank",
        ),
        (
            TensorType(DType.uint32, [_MSA_SPLIT_ARGS_SIZE], DeviceRef.GPU()),
            "device",
        ),
    ],
    ids=["int32", "rank_2", "on_the_gpu"],
)
def test_a_record_off_the_host_layout_is_refused(
    record: TensorType, field: str
) -> None:
    with Graph("split_args", input_types=[record]) as graph:
        with pytest.raises(ValueError, match=f"split_args.{field}"):
            _msa_split_args_value(graph.inputs[0].tensor)
