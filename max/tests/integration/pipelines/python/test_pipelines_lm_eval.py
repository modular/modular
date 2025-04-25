# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test for running pipelines_lm_eval."""

import json
import logging
from pathlib import Path

import hf_repo_lock
import pipelines_lm_eval
from click.testing import CliRunner
from max.pipelines.lib import generate_local_model_path

REPO_ID = "HuggingFaceTB/SmolLM2-135M"
REVISION = hf_repo_lock.revision_for_hf_repo(REPO_ID)

logger = logging.getLogger("max.pipelines")


def test_pipelines_lm_eval_smollm(tmp_path: Path):
    runner = CliRunner()
    output_dir = tmp_path / "lm-eval-output"

    try:
        local_model_path = generate_local_model_path(REPO_ID, REVISION)
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {str(e)}")
        logger.warning(
            f"Falling back to repo_id: {REPO_ID} as config to PipelineConfig"
        )
        local_model_path = REPO_ID

    result = runner.invoke(
        pipelines_lm_eval.main,
        [
            "--pipelines-probe-port=8000",
            "--pipelines-probe-timeout=240",
            "--pipelines-arg=serve",
            f"--pipelines-arg=--model-path={local_model_path}",
            "--pipelines-arg=--quantization-encoding=float32",
            "--pipelines-arg=--max-length=512",
            "--pipelines-arg=--max-new-tokens=10",
            "--pipelines-arg=--device-memory-utilization=0.3",
            "--lm-eval-arg=--model=local-completions",
            "--lm-eval-arg=--tasks=smol_task",
            f"--lm-eval-arg=--model_args=model={local_model_path},base_url=http://localhost:8000/v1/completions,tokenized_requests=False,num_concurrent=20,max_retries=3,max_length=512",
            f"--lm-eval-arg=--output_path={output_dir}",
            "--lm-eval-arg=--log_samples",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    results_dir = next(output_dir.iterdir())

    # Print predictions in case the test fails.
    samples_file = next(results_dir.glob("samples_*"))
    for n, line in enumerate(samples_file.read_text().split("\n")):
        line = line.strip()
        if not line:
            continue
        samples = json.loads(line)
        prompt = samples["doc"]["prompt"]
        response = samples["filtered_resps"][0]
        print(f"Prompt {n}: {prompt}")
        print(f"Response {n}: {response}")

    # Confirm that 3 out of the 4 results match (in test_prompts.json, one of
    # the "expected" predictions is incorrect).
    results_file = next(results_dir.glob("results_*"))
    results = json.loads(results_file.read_text().strip())
    assert results["results"]["smol_task"]["results_match,none"] == 0.75
