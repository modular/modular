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
from __future__ import annotations

"""
Smoke test for an already-serving endpoint.

Runs the smoke-test eval against an OpenAI-compatible endpoint this process
does not own, so the result describes what a deployment serves rather than MAX
at a commit. Shares the eval plumbing with ``smoke_test.py``; there is simply
no server to launch here, and no weights or recipe to resolve.

Set ``OPENAI_API_KEY`` when the endpoint needs a bearer token.
"""

import logging
import os
import sys
from pathlib import Path
from pprint import pformat

import click
from smoke_tests.eval_runner import (
    TEXT_TASK,
    VISION_TASK,
    build_eval_summary,
    call_eval,
    print_samples,
    safe_model_name,
    test_single_request,
    validate_hf_token,
    write_results,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.argument("model", type=str, required=True)
@click.option(
    "--base-url",
    type=str,
    required=True,
    help=(
        "Root URL of the endpoint, e.g. https://gateway.example.com. The "
        "OpenAI chat-completions path is appended."
    ),
)
@click.option(
    "--task",
    type=click.Choice([TEXT_TASK, VISION_TASK]),
    default=TEXT_TASK,
    help="Eval task to run.",
)
@click.option(
    "--output-path",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=None,
    help="If provided, a summary json file and the eval result are written here",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=64,
    help="Maximum concurrent requests to send to the endpoint",
)
@click.option(
    "--num-questions",
    type=int,
    default=320,
    help="Number of questions to ask the model",
)
@click.option(
    "--min-accuracy",
    type=float,
    default=None,
    help=(
        "Exit non-zero if accuracy falls below this. Left unset, accuracy is "
        "reported without a verdict and only an unservable endpoint fails."
    ),
)
@click.option(
    "--print-responses",
    is_flag=True,
    default=False,
    help="Print question/response pairs from eval samples after the run finishes",
)
@click.option(
    "--print-cot",
    is_flag=True,
    default=False,
    help="Print the model's chain-of-thought reasoning for each sample. Must be used with --print-responses",
)
@click.option(
    "--disable-timeouts",
    is_flag=True,
    default=False,
    help="Disable all timeouts. Useful when debugging hangs.",
)
def endpoint_smoke_test(
    model: str,
    base_url: str,
    task: str,
    output_path: Path | None,
    max_concurrent: int,
    num_questions: int,
    min_accuracy: float | None,
    print_responses: bool,
    print_cot: bool,
    disable_timeouts: bool,
) -> None:
    """
    Run the smoke-test eval against the endpoint serving MODEL at --base-url.

    MODEL is whatever name the endpoint routes on, which need not be a
    resolvable HuggingFace repo.

    Example:
        ./bazelw run //...:endpoint_smoke_test -- my-model \\
            --base-url https://gateway.example.com
    """
    validate_hf_token()

    if print_cot and not print_responses:
        raise ValueError("--print-cot must be used with --print-responses")

    build_workspace = os.getenv("BUILD_WORKSPACE_DIRECTORY")
    if output_path and build_workspace and not output_path.is_absolute():
        output_path = Path(build_workspace) / output_path

    result_dir = None
    if output_path is not None:
        result_dir = output_path / safe_model_name(model)
        result_dir.mkdir(parents=True, exist_ok=True)

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    test_single_request(
        url,
        model,
        task,
        disable_timeouts,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    # No metrics_url: an endpoint exposes its OpenAI routes without the
    # Prometheus ones behind them, so token counts stay unreported.
    result, samples = call_eval(
        url,
        model,
        task,
        max_concurrent=max_concurrent,
        num_questions=num_questions,
        disable_timeouts=disable_timeouts,
    )

    if print_responses:
        print_samples(samples, print_cot)

    # Nothing was started here, so there is no startup time to report.
    summary = build_eval_summary([result], startup_time_seconds=0.0)
    logger.info(pformat(summary, indent=2))

    if result_dir is not None:
        write_results(result_dir, summary, [result], [samples], [task])

    accuracy = summary[0].accuracy
    if min_accuracy is not None and accuracy < min_accuracy:
        logger.error(
            f"{task} accuracy {accuracy:.4f} is below the required "
            f"{min_accuracy:.4f}"
        )
        sys.exit(1)


if __name__ == "__main__":
    endpoint_smoke_test()
