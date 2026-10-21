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

"""Scores JevBench cases with the decider reference implementation.

Run this in an environment with ``torch`` and a recent ``transformers`` (it is
not a Bazel target). It uses the reference package's own prompt builder and
model wrapper, in fp32 by default, so the probabilities it writes are the
ground truth ``parity.py`` compares MAX against.

The reference package (``decider/``) ships inside the model repository:
``huggingface-cli download Mapika/decider-0.8b --local-dir DIR`` and pass
``--reference-dir DIR``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import torch
from cases import load_cases, make_prediction
from huggingface_hub import hf_hub_download
from metrics import format_table, summarize_by_slice
from predictions import Prediction, write_predictions


def _temperatures(model_path: str) -> tuple[float, dict[str, float]]:
    """The model's global temperature and its per-question-type overrides."""
    config_path = Path(model_path) / "decider_config.json"
    if not config_path.exists():
        config_path = Path(hf_hub_download(model_path, "decider_config.json"))
    config = json.loads(config_path.read_text())
    return float(config["temperature"]), dict(
        config.get("temperature_by_type", {})
    )


@click.command()
@click.option("--model-path", required=True)
@click.option(
    "--reference-dir",
    type=click.Path(path_type=Path, exists=True),
    required=True,
)
@click.option(
    "--cases-dir", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--slices", default=None)
@click.option("--limit-per-slice", type=int, default=None)
@click.option("--max-options", type=int, default=10, show_default=True)
@click.option("--max-state-tokens", type=int, default=None)
@click.option(
    "--dtype", type=click.Choice(["float32", "bfloat16"]), default="float32"
)
def main(
    model_path: str,
    reference_dir: Path,
    cases_dir: Path,
    output_dir: Path,
    slices: str | None,
    limit_per_slice: int | None,
    max_options: int,
    max_state_tokens: int | None,
    dtype: str,
) -> None:
    sys.path.insert(0, str(reference_dir))
    from decider import systemone as reference_systemone
    from decider.infer import Example, Q
    from decider.model import DecisionModel, collate
    from decider.prompt import MAX_OPTIONS, build

    class _KeepOrder:
        """Stops the reference builder from shuffling options."""

        def shuffle(self, items: list[Any]) -> None:
            pass

        def sample(self, items: list[Any], count: int) -> list[Any]:
            return items[:count]

    global_temperature, by_type = _temperatures(model_path)
    state_limit = max_state_tokens or 32768
    model = (
        DecisionModel(model_path, dtype=getattr(torch, dtype), grad_ckpt=False)
        .to("cuda")
        .eval()
    )
    cases = load_cases(
        cases_dir,
        slices=slices.split(",") if slices else None,
        max_options=max_options,
        limit_per_slice=limit_per_slice,
    )

    predictions: list[Prediction] = []
    for done, case in enumerate(cases, start=1):
        state = reference_systemone.render_state(case["state"])
        question = reference_systemone.render_question(case["question"])
        rows, index = reference_systemone.plan_rows({"decision": question})
        # Isolated score levels are yes/no rows; every row of a question is
        # read at that question type's temperature.
        temperature = by_type.get(question["type"], global_temperature)
        row_probabilities: list[list[float]] = []
        for row in rows:
            item = build(
                Example(state, [Q(row["question"], list(row["options"]), 0)]),
                model.tok,
                _KeepOrder(),
                max_options=MAX_OPTIONS,
                max_ctx_tokens=state_limit,
            )
            batch = collate([item], model.tok.pad_token_id)
            with torch.no_grad():
                logits = model.slot_logits(
                    batch["input_ids"].cuda(),
                    batch["attention_mask"].cuda(),
                    batch["slot_idx"].cuda(),
                    batch["slot_batch"].cuda(),
                    batch["nopts"].cuda(),
                )
            probs = torch.softmax(logits / temperature, -1)[0]
            row_probabilities.append(probs[: len(row["options"])].tolist())
        if index[0][1] == "iso":
            # Each level's chance of fitting, normalized over the levels.
            probabilities, _ = reference_systemone.combine_isolated(
                [probs[1] for probs in row_probabilities]
            )
        else:
            probabilities = row_probabilities[0]
        predictions.append(make_prediction(case, probabilities))
        if done % 100 == 0:
            click.echo(f"{done}/{len(cases)}")

    scores = summarize_by_slice(predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(output_dir / "predictions.jsonl", predictions)
    (output_dir / "scores.json").write_text(
        json.dumps(
            {"model": model_path, "dtype": dtype, "scores": scores}, indent=2
        )
    )
    click.echo(format_table(scores))


if __name__ == "__main__":
    main()
