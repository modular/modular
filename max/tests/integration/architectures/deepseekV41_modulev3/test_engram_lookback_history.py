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
from max.pipelines.architectures.deepseekV41.engram_lookback import (
    EngramLookbackBatchState,
)
from max.pipelines.architectures.deepseekV41_modulev3.batch_processor import (
    DeepseekV41BatchProcessor,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    V41Config,
    extra_input_names,
)
from max.pipelines.context import TextContext
from max.pipelines.context.tokens import TokenBuffer


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
    # The history triple replaces the window, where the window sat; the
    # dead companion keeps its place after it.
    at = names_on.index("engram_lookback_dead")
    assert names_on[at - 3 : at] == [
        "engram_lookback_ids",
        "engram_lookback_source",
        "engram_lookback_history",
    ]
    window_at = names_off.index("engram_lookback")
    assert names_on[: at - 3] == names_off[:window_at]
    assert names_on[at + 1 :] == names_off[window_at + 2 :]


def test_batch_processor_stages_the_history_only_when_exported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both exports run through the same processor: the history exists iff on.

    Flag on, the host window arrives as int64 ids, every row's source is
    fresh (-1) until a previous batch carries it, and the history buffer
    is allocated once and reused.
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
    assert history.dtype == DType.int64
    assert history.shape == (8, 3)
    again = _extras(processor, contexts, offsets)
    assert again["engram_lookback_history"] is history, "allocated once"

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
