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
"""Tests the batch processor against the ABI it must feed.

``_extras`` stages the inputs after the KV leaves in
:func:`~.interface.extra_input_names` order, and a declared name with no
feeder must fail loudly instead of silently serving a wrong tensor. The
Engram tables and token map are the model's to bind once, so the binding is
held to the same contract: every declared Engram name is bound, the tables
land as bit-preserving E4M3/E8M0 views of their raw bytes and the token
map as int32 on the first device.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import abi_contract
import numpy as np
import pytest
from max.driver import CPU, Buffer, Usage
from max.dtype import DType
from max.nn.legacy.transformer import ReturnLogits
from max.pipelines.architectures.deepseekV41.layers.engram_hash import (
    engram_lookback_token_ids,
)
from max.pipelines.architectures.deepseekV41_modulev3.batch_processor import (
    DeepseekV41BatchProcessor,
    decoder_row_count,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    V41Config,
    extra_input_names,
)
from max.pipelines.architectures.deepseekV41_modulev3.model import (
    DeepseekV41Model,
)
from max.pipelines.architectures.deepseekV41_modulev3.weight_adapters import (
    EngramTable,
)
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessorRuntime,
)


def _hf() -> Any:
    return abi_contract.hf_config("f1")


def _processor() -> DeepseekV41BatchProcessor:
    """A processor over a minimal runtime: no engine, CPU only.

    The config and the pipeline config are stand-ins carrying only the
    fields the processor reads (the same stand-in the serve-capture
    tests build).
    """
    runtime = BatchProcessorRuntime(
        pipeline_config=cast(
            Any,
            SimpleNamespace(
                model=SimpleNamespace(
                    huggingface_config=_hf(), data_parallel_degree=1
                )
            ),
        ),
        devices=[CPU()],
        return_logits=ReturnLogits.LAST_TOKEN,
        max_batch_size=8,
        max_seq_len=64,
        max_batch_input_tokens=64,
    )
    return DeepseekV41BatchProcessor(cast(Any, SimpleNamespace()), runtime)


def _context(history: list[int], processed: int) -> TextContext:
    """A request carrying ``history``, of which ``processed`` are handled."""
    ctx = TextContext(
        max_length=64, tokens=TokenBuffer(np.array(history, dtype=np.int64))
    )
    ctx.tokens.skip_processing(processed)
    return ctx


def test_extras_follow_the_abi_names() -> None:
    assert extra_input_names(V41Config.from_hf(_hf())) == [
        "swa_begin",
        "global_write_floor",
        "image_token_mask",
        "engram_lookback",
        "engram_lookback_dead",
        "engram_table_0_weight",
        "engram_table_0_scale",
        "engram_table_1_weight",
        "engram_table_1_scale",
        "engram_token_map",
        "decoder_rows",
    ]


def test_extras_stage_the_static_inputs_in_abi_order() -> None:
    processor = _processor()
    config = V41Config.from_hf(_hf())
    static = {
        name: Buffer.zeros(
            [1],
            (
                DType.float8_e4m3fn
                if name.endswith("_weight")
                else DType.float8_e8m0fnu
            ),
        )
        for name in extra_input_names(config)
        if name.startswith("engram_table_")
    }
    static["engram_token_map"] = Buffer.zeros([1], DType.int32)
    processor.bind_engram(static)

    extras = processor._extras(
        [_context([5, 6, 7, 8], 0), _context([9, 9], 1)], np.array([0, 4, 6])
    )

    by_name = dict(zip(extra_input_names(config), extras, strict=True))
    for name, buffer in static.items():
        assert by_name[name] is buffer
    assert by_name["swa_begin"].dtype == DType.int32
    assert by_name["swa_begin"].shape == (2,)
    assert by_name["image_token_mask"].dtype == DType.uint8
    assert by_name["image_token_mask"].shape == (6,)
    assert by_name["decoder_rows"].shape == (6,)
    assert by_name["engram_lookback"].dtype == DType.int32
    assert by_name["engram_lookback"].shape == (2, 3)
    assert by_name["engram_lookback_dead"].shape == (2, 3)


def test_extras_fail_loudly_without_the_engram_inputs() -> None:
    with pytest.raises(RuntimeError, match="engram_table_0_weight"):
        _processor()._extras([_context([1, 2, 3], 2)], np.array([0, 3]))


def test_the_model_binds_the_token_map_and_raw_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The binding feeds every Engram name ``_extras`` requires.

    The token map rides the first device as the ABI declares it, and the
    tables are staged as bit-preserving E4M3/E8M0 views of their raw
    bytes -- the ABI's table inputs carry the kernel's dtypes, never a
    byte-typed copy. Pinning itself needs the accelerator
    (``test_pinned_tables_gpu`` runs the real allocation), so this test
    stands in for the pinned copy with a CPU staging buffer viewed the
    same way the model views it.
    """
    processor = _processor()
    config = V41Config.from_hf(_hf())
    model = object.__new__(DeepseekV41Model)
    tables = [
        EngramTable(
            layer=layer,
            rows=2,
            weight_codes=np.frombuffer(
                bytes([ordinal + 1]) * 2 * 256, np.uint8
            ).reshape(2, 256),
            scale_codes=np.frombuffer(
                bytes([ordinal + 7]) * 2 * 8, np.uint8
            ).reshape(2, 8),
        )
        for ordinal, layer in enumerate(config.engram_layer_ids)
    ]
    names = [table.table_names(ordinal) for ordinal, table in enumerate(tables)]
    model._engram_tables = tables
    model._token_map = np.arange(config.vocab_size, dtype=np.int32)
    device = CPU()
    model.devices = [device]
    model._batch_processor = processor

    def fake_pinned_table(
        pin_device: object,
        name: str,
        rows: int,
        width: int,
        codes: np.ndarray,
        dtype: DType,
    ) -> Buffer:
        del pin_device
        buffer = Buffer(
            shape=(rows, width),
            dtype=DType.uint8,
            device=device,
            usage=Usage.STAGING,
        )
        np.copyto(buffer.to_numpy(), codes)
        return buffer.view(dtype)

    monkeypatch.setattr(
        "max.pipelines.architectures.deepseekV41_modulev3.model._pinned_table",
        fake_pinned_table,
    )

    model._bind_engram_static_inputs()

    static = processor._engram_static
    assert sorted(static) == sorted(
        name
        for name in extra_input_names(config)
        if name.startswith("engram_table_") or name == "engram_token_map"
    )
    token_map = static["engram_token_map"]
    assert token_map.dtype == DType.int32
    assert token_map.shape == (config.vocab_size,)
    for weight_name, scale_name in names:
        assert static[weight_name].dtype == DType.float8_e4m3fn, weight_name
        assert static[weight_name].shape == (2, 256)
        assert static[scale_name].dtype == DType.float8_e8m0fnu, scale_name
        assert static[scale_name].shape == (2, 8)
    # The binding owns the pinned copies: the adapter's checkpoint views
    # are dropped so a second mapping is not held for the model's lifetime.
    assert model._engram_tables == []


@pytest.mark.parametrize(
    ("lengths", "want"),
    [
        ([4, 2], 6),
        ([300, 5, 128, 129], 128 + 5 + 128 + 128),
        ([1, 1, 1], 3),
        ([], 0),
    ],
)
def test_decoder_rows_keep_each_requests_last_window(
    lengths: list[int], want: int
) -> None:
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.uint32)
    assert decoder_row_count(offsets, 128) == want


@pytest.mark.parametrize(
    ("history", "start", "want"),
    [
        # Prefill from position 0 sees nothing before it.
        ([5, 6, 7, 8, 9], 0, [-1, -1, -1]),
        # A chunk continuation sees the three tokens before it, newest first.
        ([5, 6, 7, 8, 9], 4, [8, 7, 6]),
        # Fewer than three before it: the rest is before the sequence start.
        ([5, 6, 7, 8, 9], 2, [6, 5, -1]),
        # A decode step.
        ([5, 6, 7, 8, 9], 5, [9, 8, 7]),
    ],
)
def test_lookback_is_the_real_predecessors(
    history: list[int], start: int, want: list[int]
) -> None:
    got = engram_lookback_token_ids([np.array(history)], [start], 3).astype(
        np.int32
    )
    assert got.dtype == np.int32
    assert got.tolist() == [want]


def test_lookback_of_two_requests_at_different_offsets() -> None:
    got = engram_lookback_token_ids(
        [np.array([1, 2, 3, 4]), np.array([7, 8, 9])], [4, 1], 3
    ).astype(np.int32)
    assert got.tolist() == [[4, 3, 2], [7, -1, -1]]
