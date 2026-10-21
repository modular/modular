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
"""AutoModel with device graph capture on a real GPU model.

Every decode step must replay a captured graph, and the greedy tokens and
the logits GenerateIterator records must match an AutoModel built without
capture. The basic example from the AutoModel design runs here as written.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from max.driver import DeviceSpec
from max.pipelines import AutoModel, AutoTokenizer, TextGenerationPipeline
from max.pipelines.lib import ModelOutputs, OverlapTextGenerationPipeline

MODEL = "HuggingFaceTB/SmolLM2-135M"
PROMPT_LENGTHS = (5, 17, 40, 80)
MAX_NEW_TOKENS = 16
PASSAGE = (
    "The history of computing is a story of steadily shrinking machines and "
    "growing ambitions. Early mechanical calculators could add and subtract, "
    "but the first programmable electronic computers filled entire rooms and "
    "consumed enormous amounts of power. Transistors replaced vacuum tubes, "
    "integrated circuits replaced discrete transistors, and each generation "
    "of hardware made the previous one look slow and expensive. Personal "
    "computers brought this power into homes and offices during the late "
    "twentieth century."
)


def _load(device_graph_capture: bool) -> AutoModel:
    return AutoModel(
        MODEL,
        max_batch_size=len(PROMPT_LENGTHS),
        device_graph_capture=device_graph_capture,
        device_specs=[DeviceSpec.accelerator()],
        quantization_encoding="bfloat16",
        max_length=512,
    )


def test_basic_example_runs_as_written() -> None:
    tokenizer = AutoTokenizer("HuggingFaceTB/SmolLM2-135M")
    model = AutoModel("HuggingFaceTB/SmolLM2-135M", max_batch_size=2)
    try:
        inputs = tokenizer(["The capital of France is", "2 + 2 ="])
        outputs = model.generate(
            **inputs, max_new_tokens=16, output_logits=True
        )
        texts = tokenizer.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
        print(texts)
        vocab_size = model.pipeline_config.model.huggingface_config.vocab_size
    finally:
        model.close()

    for sequence, prompt_ids in zip(
        outputs.sequences, inputs["input_ids"], strict=True
    ):
        assert sequence[: len(prompt_ids)] == prompt_ids
    for text, prompt in zip(
        texts, ["The capital of France is", "2 + 2 ="], strict=True
    ):
        assert text.startswith(prompt)
    assert outputs.logits is not None
    new_tokens = len(outputs.sequences[0]) - len(inputs["input_ids"][0])
    assert outputs.logits[0].shape == (new_tokens, vocab_size)


def test_graph_capture_replays_and_matches_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The capture model loads last, since references such as the runner
    # patched below keep its pipeline alive after close().
    tokens = AutoTokenizer(MODEL)(PASSAGE)["input_ids"]
    assert len(tokens) >= max(PROMPT_LENGTHS)
    input_ids = [tokens[:length] for length in PROMPT_LENGTHS]
    with _load(device_graph_capture=False) as model:
        assert isinstance(model.pipeline, TextGenerationPipeline)
        eager = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            top_k=1,
            ignore_eos=True,
            output_logits=True,
        )

    with _load(device_graph_capture=True) as model:
        pipeline = model.pipeline
        assert isinstance(pipeline, OverlapTextGenerationPipeline)
        assert pipeline.overlap_active is False
        assert pipeline.graph_capture_ready
        runner = pipeline._graph_capture_runner
        assert runner is not None
        replay: Callable[..., ModelOutputs] = runner.replay
        replays = 0

        def count_replay(*args: object, **kwargs: object) -> ModelOutputs:
            nonlocal replays
            replays += 1
            return replay(*args, **kwargs)

        monkeypatch.setattr(runner, "replay", count_replay)

        steps = 0
        with model.generate_iter(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            top_k=1,
            ignore_eos=True,
            output_logits=True,
        ) as batch:
            for _ in batch:
                steps += 1
        assert steps == MAX_NEW_TOKENS
        # Only the prefill step runs eagerly.
        assert replays == steps - 1
    assert eager.sequences == batch.sequences
    assert eager.logits is not None
    for expected, actual in zip(eager.logits, batch.logits, strict=True):
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=1e-2, rtol=1e-3)
