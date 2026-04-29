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
Smoke test for disaggregated inference (DI) deployments.

Boots ``run_dist`` (workers + proxy load balancer) from a deployment YAML and
runs the gsm8k_cot_llama lm-eval task through the proxy. Shares the eval
plumbing with ``smoke_test.py``; only the launch + readiness path is DI
specific.
"""

import logging
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from pprint import pformat

import click
import requests
import yaml
from inference_server_harness import start_server
from smoke_tests.eval_runner import (
    TEXT_TASK,
    build_eval_summary,
    call_eval,
    print_samples,
    safe_model_name,
    test_single_request,
    validate_hf_token,
    write_github_output,
    write_results,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DI_SERVER_TIMEOUT_SECONDS = 2700


def build_run_dist_cmd(
    config_yaml: Path, proxy_host: str, proxy_port: int
) -> list[str]:
    """Build the ``run_dist`` command that boots the DI workers + proxy.

    The returned cmd shells out to ``./bazelw``, so callers must spawn it
    with ``cwd`` set to the workspace root.
    """
    return [
        "./bazelw",
        "run",
        "//max/examples/internal/di:run_dist",
        "--",
        "--config-yaml",
        str(config_yaml),
        "--start-proxy-server",
        "--proxy-host",
        proxy_host,
        "--proxy-port",
        str(proxy_port),
    ]


def make_proxy_health_probe(base_url: str) -> Callable[[], bool]:
    """Return a probe that GETs ``/health`` on the DI proxy."""

    def probe() -> bool:
        try:
            return (
                requests.get(f"{base_url}/health", timeout=2).status_code == 200
            )
        except requests.exceptions.RequestException:
            return False

    return probe


def _section_model_path(section: object) -> str | None:
    if not isinstance(section, Mapping):
        return None
    model_config = section.get("model_config")
    if not isinstance(model_config, Mapping):
        return None
    model_path = model_config.get("model_path")
    return model_path if isinstance(model_path, str) else None


def resolve_model_path(config_yaml: Path) -> str:
    """Pull the (single) ``model_path`` declared in a run_dist deployment YAML."""
    data = yaml.safe_load(config_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected mapping at top level of {config_yaml}")
    sections: list[object] = [data.get("defaults") or {}]
    worker_groups = data.get("worker_groups") or {}
    if isinstance(worker_groups, Mapping):
        sections.extend(worker_groups.values())
    paths: set[str] = {
        path
        for section in sections
        if (path := _section_model_path(section)) is not None
    }
    if len(paths) != 1:
        raise ValueError(
            f"Expected exactly one model_path in {config_yaml}, "
            f"found {sorted(paths)!r}"
        )
    return paths.pop()


@click.command()
@click.argument(
    "config_yaml",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output-path",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=None,
    help="If provided, a summary json file and the eval result are written here",
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
    "--max-concurrent",
    type=int,
    default=64,
    help="Maximum concurrent requests to send to the proxy",
)
@click.option(
    "--num-questions",
    type=int,
    default=320,
    help="Number of questions to ask the model",
)
@click.option(
    "--proxy-host",
    type=str,
    default="127.0.0.1",
    help="Proxy bind host",
)
@click.option(
    "--proxy-port",
    type=int,
    default=9999,
    help="Proxy bind port",
)
@click.option(
    "--disable-timeouts",
    is_flag=True,
    default=False,
    help="Disable all timeouts. Useful when debugging hangs.",
)
def di_smoke_test(
    config_yaml: Path,
    output_path: Path | None,
    print_responses: bool,
    print_cot: bool,
    max_concurrent: int,
    num_questions: int,
    proxy_host: str,
    proxy_port: int,
    disable_timeouts: bool,
) -> None:
    """
    Run a DI smoke test against the deployment described by CONFIG_YAML.

    Spawns ``run_dist`` (workers + proxy load balancer) and routes the
    gsm8k_cot_llama lm-eval task through the proxy at PROXY_HOST:PROXY_PORT.
    The model under test is the ``model_path`` declared in the YAML.

    Example:
        ./bazelw run //...:di_smoke_test -- \\
            max/examples/internal/di/configs/1p1d-intranode.yaml
    """
    validate_hf_token()

    if print_cot and not print_responses:
        raise ValueError("--print-cot must be used with --print-responses")

    build_workspace = os.getenv("BUILD_WORKSPACE_DIRECTORY")
    if build_workspace and not config_yaml.is_absolute():
        config_yaml = Path(build_workspace) / config_yaml
    if not config_yaml.exists():
        raise FileNotFoundError(f"config-yaml not found: {config_yaml}")
    if output_path and build_workspace and not output_path.is_absolute():
        output_path = Path(build_workspace) / output_path

    # Use the YAML model_path verbatim: run_dist registers the server under
    # exactly this string, and the OpenAI request's `model` field must match
    # what the server reports. HF canonical-id normalization would produce a
    # different name (e.g. Meta- prefix redirects) that the server does not
    # know about.
    model = resolve_model_path(config_yaml)
    base_url = f"http://{proxy_host}:{proxy_port}"
    url = f"{base_url}/v1/chat/completions"
    cmd = build_run_dist_cmd(config_yaml, proxy_host, proxy_port)

    timeout = sys.maxsize if disable_timeouts else DI_SERVER_TIMEOUT_SECONDS

    logger.info(f"Starting DI deployment with command:\n {' '.join(cmd)}")
    with start_server(
        cmd,
        timeout,
        readiness_probe=make_proxy_health_probe(base_url),
        cwd=build_workspace,
        poll_interval=5.0,
    ) as server:
        logger.info(f"Proxy ready in {server.startup_time:.2f} seconds")
        write_github_output("startup_time", f"{server.startup_time:.2f}")

        test_single_request(
            url, model, TEXT_TASK, disable_timeouts=disable_timeouts
        )
        result, samples = call_eval(
            url,
            model,
            TEXT_TASK,
            max_concurrent=max_concurrent,
            num_questions=num_questions,
            disable_timeouts=disable_timeouts,
        )
        if print_responses:
            print_samples(samples, print_cot)

    summary = build_eval_summary(
        [result], startup_time_seconds=server.startup_time
    )

    if output_path is not None:
        path = output_path / safe_model_name(model)
        path.mkdir(parents=True, exist_ok=True)
        write_results(path, summary, [result], [samples], [TEXT_TASK])

    logger.info(pformat(summary, indent=2))


if __name__ == "__main__":
    di_smoke_test()
