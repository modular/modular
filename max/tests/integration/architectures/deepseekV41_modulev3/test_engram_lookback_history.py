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
"""The Engram lookback's device history, as the overlap export takes it.

Under the overlap scheduler the host does not yet know a step's accepted
drafts when it builds the next step's window, so the export's graph reads
the window the previous step recorded in-graph (the V2 mechanism): the host
window arrives as int64 ids beside a per-row source map and a persistent
history buffer, and the verify records each row's window for the next step.
The default export keeps the host-built window alone, so its input list is
unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from abi_contract import hf_config
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.kv_cache import RecurrentStateParams
from max.pipelines.architectures.deepseekV41.engram_lookback import (
    EngramLookbackBatchState,
)
from max.pipelines.architectures.deepseekV41_modulev3.batch_processor import (
    DeepseekV41BatchProcessor,
)
from max.pipelines.architectures.deepseekV41_modulev3.history import (
    _mask,
    _read,
    _record,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    LOOKBACK_HISTORY_ROW,
    V41Config,
    build_kv_params,
    extra_input_names,
    kv_input_names,
)
from max.pipelines.context import TextContext
from max.pipelines.context.tokens import TokenBuffer
from max.pipelines.lib import KVCacheConfig


def _text(**overrides: object) -> SimpleNamespace:
    text = SimpleNamespace(**dict(hf_config("f1"))["text_config"])
    for key, value in overrides.items():
        setattr(text, key, value)
    return text


def _context(start: int, length: int) -> TextContext:
    """A request whose first ``start`` tokens are already processed."""
    tokens = TokenBuffer(np.arange(start + length, dtype=np.int64))
    tokens.skip_processing(start)
    return TextContext(max_length=1_000_000, tokens=tokens)


def _processor(
    monkeypatch: pytest.MonkeyPatch, *, engram_lookback_history: bool
) -> DeepseekV41BatchProcessor:
    # The staging under test is the lookback's; the Engram tables the
    # trunk pins at load stay out of the fixture, as the replay tests do.
    text = _text(
        engram_lookback_history=engram_lookback_history,
        engram_layer_ids=[],
        engram_num_embeddings=[],
    )
    processor = object.__new__(DeepseekV41BatchProcessor)
    processor._v41 = V41Config.from_hf({"text_config": vars(text)})
    processor._engram_static = {
        "engram_token_map": Buffer.from_numpy(np.zeros(1, np.int32))
    }
    processor._verifies = False
    processor._dp_degree = 1
    processor._lookback = EngramLookbackBatchState()
    monkeypatch.setattr(
        processor,
        "runtime",
        SimpleNamespace(devices=[CPU()], max_global_batch_size=8),
        raising=False,
    )
    return processor


def _extras(
    processor: DeepseekV41BatchProcessor,
    contexts: list[TextContext],
    offsets: np.ndarray,
) -> dict[str, Buffer]:
    return dict(
        zip(
            extra_input_names(processor._v41),
            processor._extras(contexts, offsets),
            strict=True,
        )
    )


def test_default_export_declares_no_lookback_history() -> None:
    """The history inputs exist iff the export asks for them.

    The default input list is the host-built window alone, so a default
    export's ABI is byte-for-byte what it was before the mechanism.
    """
    off = V41Config.from_hf({"text_config": vars(_text())})
    on = V41Config.from_hf(
        {"text_config": vars(_text(engram_lookback_history=True))}
    )
    assert not off.engram_lookback_history
    assert on.engram_lookback_history
    names_off = extra_input_names(off)
    names_on = extra_input_names(on)
    assert "engram_lookback" in names_off
    assert not any(
        name in names_off
        for name in (
            "engram_lookback_ids",
            "engram_lookback_source",
            "engram_lookback_history",
        )
    )
    # A bare MEF keeps the registry contract's history buffer input.
    at = names_on.index("engram_lookback_dead")
    assert names_on[at - 3 : at] == [
        "engram_lookback_ids",
        "engram_lookback_source",
        "engram_lookback_history",
    ]
    window_at = names_off.index("engram_lookback")
    assert names_on[: at - 3] == names_off[:window_at]
    assert names_on[at + 1 :] == names_off[window_at + 2 :]
    # An artifact's history travels with its recurrent KV leaf instead.
    artifact = V41Config.from_hf(
        {"text_config": vars(_text(engram_lookback_history=True))},
        artifact_export=True,
    )
    names_artifact = extra_input_names(artifact)
    assert "engram_lookback_history" not in names_artifact
    # (An artifact also turns on the decoder replay, whose inputs trail.)
    bare_names = [n for n in names_on if n != "engram_lookback_history"]
    assert names_artifact[: len(bare_names)] == bare_names


def test_batch_processor_stages_the_history_only_when_exported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both exports run through the same processor: the history exists iff on.

    Flag on, the host window arrives as int64 ids, every row's source is
    fresh (-1) until a previous batch carries it, and a bare MEF's history
    buffer is allocated once and reused.
    """
    contexts = [_context(6, 1), _context(13, 2)]
    offsets = np.array([0, 1, 3])
    processor = _processor(monkeypatch, engram_lookback_history=True)
    extras = _extras(processor, contexts, offsets)
    assert "engram_lookback" not in extras
    ids = extras["engram_lookback_ids"].to_numpy()
    np.testing.assert_array_equal(ids[0], [5, 4, 3])
    np.testing.assert_array_equal(ids[1], [12, 11, 10])
    assert ids.dtype == np.int64
    np.testing.assert_array_equal(
        extras["engram_lookback_source"].to_numpy(), [-1, -1]
    )
    history = extras["engram_lookback_history"]
    assert history is processor._lookback.history
    assert history.shape == (8, 3)
    assert _extras(processor, contexts, offsets)["engram_lookback_history"] is (
        history
    )

    # The default export stages the int32 host window as before.
    processor = _processor(monkeypatch, engram_lookback_history=False)
    extras = _extras(processor, contexts, offsets)
    assert "engram_lookback_ids" not in extras
    lookback = extras["engram_lookback"].to_numpy()
    assert lookback.dtype == np.int32
    np.testing.assert_array_equal(lookback[0], [5, 4, 3])


def test_batch_processor_sources_the_rows_the_previous_batch_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row whose previous step is still in flight reads its history row.

    The predicate is the one the overlap pipeline realizes the cache
    lengths with: a request carrying drafts under verification whose
    generated tokens exist starts after those drafts, which only the
    device knows, so the graph reads the window the previous step
    recorded at the row the request sat in.
    """
    request = _context(6, 1)
    joined = _context(13, 2)
    processor = _processor(monkeypatch, engram_lookback_history=True)
    # The first batch seats the request at row 0.
    _extras(processor, [request], np.array([0, 1]))
    # Its step is still in flight when the next batch forms: it comes
    # back carrying drafts under verification and a generated token.
    request.tokens.advance_with_token(6)
    request.spec_decoding_state.maybe_accepted_draft_tokens = [7]
    extras = _extras(processor, [request, joined], np.array([0, 2, 4]))
    np.testing.assert_array_equal(
        extras["engram_lookback_source"].to_numpy(), [0, -1]
    )
    # A batch without the request sources no row; the history keeps it.
    extras = _extras(processor, [joined], np.array([0, 2]))
    np.testing.assert_array_equal(
        extras["engram_lookback_source"].to_numpy(), [-1]
    )


@pytest.mark.parametrize("replicas", [1, 4, 8])
def test_history_is_request_scratch_in_the_cache_tree(replicas: int) -> None:
    bare = V41Config.from_hf(
        {"text_config": vars(_text(engram_lookback_history=True))}
    )
    devices = [DeviceRef.GPU(i) for i in range(replicas)]
    # A bare MEF keeps the registry contract's six KV groups.
    assert "lookback_history" not in (
        build_kv_params(
            bare,
            KVCacheConfig(kv_cache_page_size=128),
            devices,
            data_parallel_degree=replicas,
        ).children
    )
    config = V41Config.from_hf(
        {"text_config": vars(_text(engram_lookback_history=True))},
        artifact_export=True,
    )
    params = build_kv_params(
        config,
        KVCacheConfig(kv_cache_page_size=128),
        devices,
        data_parallel_degree=replicas,
    )
    history = params.children["lookback_history"]
    assert isinstance(history, RecurrentStateParams)
    assert len(history.regions) == 1
    assert history.regions[0].scratch
    assert history.regions[0].row_shape == (LOOKBACK_HISTORY_ROW,)
    # Mach's KV pool rejects a leaf whose row is not a multiple of 16 bytes.
    assert history.regions[0].bytes_per_page % 16 == 0
    assert history.regions[0].dtype == DType.int64
    assert len(kv_input_names(params)) == len(params.flattened_kv_inputs())


@pytest.mark.parametrize("draft_width", [0, 5])
def test_history_graph_uses_request_rows_and_accepted_prefix(
    draft_width: int,
) -> None:
    device = DeviceRef.CPU()
    types: list[BufferType | TensorType] = [
        BufferType(DType.int64, [16, LOOKBACK_HISTORY_ROW], device=device),
        TensorType(DType.uint32, [1, 3], device),
        TensorType(DType.int64, [3, 3], device),
        TensorType(DType.int64, [3], device),
        TensorType(DType.int64, [9], device),
        TensorType(DType.uint32, [4], device),
        TensorType(DType.int64, [3, draft_width], device),
        TensorType(DType.int64, [3], device),
    ]
    with Graph("arena_lookback", input_types=types) as graph:
        pool, row_ids, host, source, tokens, offsets, drafts, accepted = (
            graph.inputs
        )
        lookback = _read(
            pool.buffer, row_ids.tensor, host.tensor, source.tensor
        )
        _record(
            pool.buffer,
            row_ids.tensor,
            lookback,
            tokens.tensor,
            offsets.tensor,
            drafts.tensor,
            accepted.tensor,
        )
        graph.output(lookback)
    assert lookback.dtype == DType.int64
    assert list(lookback.shape) == [3, 3]


def test_history_commits_mixed_accepted_prefixes_by_request() -> None:
    """The gather/accept/scatter composition must preserve request history.

    Individual op checks cannot catch committing a rejected draft or writing
    a batch index instead of a live request row; either corrupts overlap.
    This tiny CPU graph exercises that composition before merge.
    """
    device = DeviceRef.CPU()
    types: list[BufferType | TensorType] = [
        BufferType(DType.int64, [8, LOOKBACK_HISTORY_ROW], device=device),
        TensorType(DType.uint32, [1, 3], device),
        TensorType(DType.int64, [3, 3], device),
        TensorType(DType.int64, [3], device),
        TensorType(DType.int64, [6], device),
        TensorType(DType.uint32, [4], device),
        TensorType(DType.int64, [3, 5], device),
        TensorType(DType.int64, [3], device),
    ]
    with Graph("accepted_request_history", input_types=types) as graph:
        pool, ids, host, source, tokens, offsets, drafts, accepted = (
            graph.inputs
        )
        lookback = _read(pool.buffer, ids.tensor, host.tensor, source.tensor)
        _record(
            pool.buffer,
            ids.tensor,
            lookback,
            tokens.tensor,
            offsets.tensor,
            drafts.tensor,
            accepted.tensor,
        )
        dead = ops.constant(
            np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], np.uint8),
            DType.uint8,
            device,
        )
        mask = _mask(source.tensor, lookback, dead, 10)
        graph.output(lookback, ops.buffer_load(pool.buffer), mask)
    model = InferenceSession(devices=[CPU()]).load(graph)
    expected = np.full((8, LOOKBACK_HISTORY_ROW), -987654, np.int64)
    expected[1, :3] = [10, 9, 8]
    expected[6, :3] = [60, 59, 58]
    expected[4, :3] = [400, 399, 398]
    pool_buffer = Buffer.from_numpy(expected.copy())
    offsets_array = np.array([0, 1, 3, 6], np.uint32)
    for row_ids, sources, base in [
        ([4, 1, 6], [-1, 0, 0], 40),
        ([6, 4, 1], [0, 0, 0], 70),
        ([7, 6, 1], [-1, 0, 0], 700),
    ]:
        host_array = np.array(
            [[base, base - 1, base - 2], [900, 899, 898], [800, 799, 798]],
            np.int64,
        )
        token_array = np.array([base + 1, 11, 12, 61, 62, 63], np.int64)
        draft_array = np.array(
            [
                [base + 2, base + 3, base + 4, base + 5, base + 6],
                [13, 14, 15, 16, 17],
                [64, 65, 66, 67, 68],
            ],
            np.int64,
        )
        counts = np.array([0, 1, 5], np.int64)
        before = np.stack(
            [
                host_array[i] if sources[i] < 0 else expected[row, :3]
                for i, row in enumerate(row_ids)
            ]
        )
        for i, row in enumerate(row_ids):
            sequence = [
                *before[i, ::-1],
                *token_array[offsets_array[i] : offsets_array[i + 1]],
                *draft_array[i, : counts[i]],
            ]
            expected[row] = [*sequence[-3:][::-1], 0]
        result = model.execute(
            pool_buffer,
            Buffer.from_numpy(np.array([row_ids], np.uint32)),
            Buffer.from_numpy(host_array),
            Buffer.from_numpy(np.array(sources, np.int64)),
            Buffer.from_numpy(token_array),
            Buffer.from_numpy(offsets_array),
            Buffer.from_numpy(draft_array),
            Buffer.from_numpy(counts),
        )
        np.testing.assert_array_equal(result[0].to_numpy(), before)
        np.testing.assert_array_equal(result[1].to_numpy(), expected)
        expected_mask = np.where(
            np.array(sources)[:, None] >= 0,
            before == 10,
            np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], np.uint8),
        )
        np.testing.assert_array_equal(result[2].to_numpy(), expected_mask)
