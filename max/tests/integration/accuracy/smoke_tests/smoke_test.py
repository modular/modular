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
This script is used for the CI "Max Serve Smoke Test" workflow.
It does two things:
    1. Starts the MAX/SGLang/VLLM inference server for the given model
    2. Runs a tiny evaluation task using against the chat/completions API

Currently there is a hard dependency that two virtualenvs are already created:
    - .venv-serve (not needed for max-ci, which uses bazel)
    - .venv-eval

Where the serve environment should already have either MAX/VLLM/SGLang installed.
The eval environment should already have lm-eval installed.
These dependencies are to be removed once this script
has been integrated into bazel.

Note that if you're running this script inside bazel, only available for max-ci,
then the virtualenvs are not needed.
"""

import csv
import json
import logging
import os
import shlex
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from pprint import pformat
from subprocess import Popen, TimeoutExpired, check_call, check_output
from tempfile import TemporaryDirectory
from typing import Any

import click
import requests

DUMMY_2X2_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP8zwACTGCSAQANHQEDgslx/wAAAABJRU5ErkJggg=="
)
IMAGE_PROMPT = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Say 'hello image'"},
        {"type": "image_url", "image_url": {"url": DUMMY_2X2_IMAGE}},
    ],
}
TEXT_PROMPT = {"role": "user", "content": "Say: 'hello world'"}
URL = "http://127.0.0.1:8000/v1/chat/completions"

TEXT_TASK = "gsm8k_cot_llama"
VISION_TASK = "chartqa"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EvalResults = dict[str, Any]
EvalSamples = list[dict[str, Any]]


def is_deepseek(model: str) -> bool:
    """Temporary workaround for large DeepSeek models."""
    return "deepseek" in model and "lite" not in model


def validate_hf_token() -> None:
    if os.getenv("HF_TOKEN") is None:
        raise ValueError(
            "Environment variable `HF_TOKEN` must be set. "
            "See https://www.notion.so/modularai/HuggingFace-Access-Token-29d1044d37bb809fbe70e37428faf9da"
        )


def _inside_bazel() -> bool:
    return os.getenv("BUILD_WORKSPACE_DIRECTORY") is not None


@cache
def _load_hf_repo_lock() -> dict[str, str]:
    """Read hf-repo-lock.tsv, return {lowercase_repo: revision} mapping."""
    tsv = Path(__file__).resolve().parent.parent.parent / "hf-repo-lock.tsv"
    if not tsv.exists():
        logger.warning("hf-repo-lock.tsv not found, skipping revision pinning")
        return {}
    db = {}
    with open(tsv) as f:
        for row in csv.DictReader(f, dialect="excel-tab"):
            db[row["hf_repo"].lower()] = row["revision"]
    return db


def test_single_request(model: str, task: str) -> None:
    is_vision = task == VISION_TASK
    m = [IMAGE_PROMPT if is_vision else TEXT_PROMPT]

    r = requests.post(
        URL,
        json={"model": model, "messages": m, "max_tokens": 8},
        timeout=(30, 180),  # Initial req can be slow for huge models
    )
    r.raise_for_status()
    resp = r.json()["choices"][0]["message"]["content"]
    logger.info(f"Test single request OK. Response: {resp}")


@cache
def get_gpu_name_and_count() -> tuple[str, int]:
    """Returns the name and number of the available GPUs, e.g. ('MI300', 2)"""
    amd = ["amd-smi", "static", "--json"]
    nv = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    try:
        result = check_output(amd, text=True)
        data = json.loads(result.strip())["gpu_data"]
        return data[0]["asic"]["market_name"], len(data)
    except:
        try:
            lines = check_output(nv, text=True).strip().split("\n")
            return lines[0].strip(), len(lines)
        except:
            logger.warning("nvidia-smi and amd-smi both failed")
            return "N/A", 0


def server_is_ready() -> bool:
    health_url = "http://127.0.0.1:8000/health"
    try:
        return requests.get(health_url, timeout=1).status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_server_cmd(
    framework: str,
    model: str,
    *,
    serve_extra_args: str = "",
) -> list[str]:
    gpu_model, gpu_count = get_gpu_name_and_count()
    sglang_backend = "triton" if "b200" in gpu_model.lower() else "fa3"
    SGLANG = f"sglang.launch_server --attention-backend {sglang_backend} --mem-fraction-static 0.8"
    # limit-mm-per-prompt.video is for InternVL3 on B200
    VLLM = "vllm.entrypoints.openai.api_server --max-model-len 16384 --limit-mm-per-prompt.video 0"
    MAX = "max.entrypoints.pipelines serve"

    is_huge_model = is_deepseek(model)
    if is_huge_model and framework != "sglang":
        MAX += f" --device-memory-utilization 0.8 --devices gpu:{','.join(str(i) for i in range(gpu_count))} --ep-size {gpu_count} --data-parallel-degree {gpu_count} --max-batch-input-tokens 1024"
        VLLM += f" --enable-chunked-prefill --gpu-memory-utilization 0.8 --data-parallel-size={gpu_count} --enable-expert-parallel"
        # Have not been successful in getting SGLang to work with R1 yet
    elif gpu_count > 1:
        MAX += f" --devices gpu:{','.join(str(i) for i in range(gpu_count))}"
        VLLM += f" --tensor-parallel-size={gpu_count}"
        SGLANG += f" --tp-size={gpu_count}"

    if _inside_bazel():
        assert framework == "max-ci", "bazel invocation only supports max-ci"
        cmd = [sys.executable, "-m", *MAX.split()]
    else:
        assert framework != "max-ci", "max-ci must be run through bazel"
        interpreter = [".venv-serve/bin/python", "-m"]
        commands = {
            "sglang": [*interpreter, *SGLANG.split()],
            "vllm": [*interpreter, *VLLM.split()],
            "max": [*interpreter, *MAX.split()],
        }
        cmd = commands[framework]

    cmd = cmd + ["--port", "8000", "--trust-remote-code", "--model", model]

    # GPT-OSS uses repetition_penalty in lm_eval to prevent reasoning loops,
    # so we need to enable penalties on the server
    if "gpt-oss" in model and framework in ["max-ci", "max"]:
        cmd += ["--enable-penalties"]

    if framework in ["max-ci", "max"] and "tbmod" in model:
        cmd += ["--prefer-module-v3"]

    revision = _load_hf_repo_lock().get(model)
    if revision:
        if framework in ("max", "max-ci"):
            cmd += [
                "--huggingface-model-revision",
                revision,
                "--huggingface-weight-revision",
                revision,
            ]
        else:  # vllm, sglang
            cmd += ["--revision", revision]
        logger.info(f"Pinned to revision {revision[:12]}")
    else:
        logger.warning(f"No locked revision for {model}")

    if serve_extra_args:
        if framework in ["max-ci", "max"]:
            cmd += shlex.split(serve_extra_args)
        else:
            logger.warning(
                "Ignoring --serve-extra-args for framework %s", framework
            )
    return cmd


def safe_model_name(model: str) -> str:
    return model.replace("/", "__")


def call_eval(
    model: str, task: str, *, max_concurrent: int, num_questions: int
) -> tuple[EvalResults, EvalSamples]:
    extra_gen_kwargs = ""
    is_reasoning_model = any(
        kw in model
        for kw in (
            "academic-ds",
            "deepseek-r1",
            "gpt-oss",
            "internvl3_5",
            "qwen3",
        )
    )
    # Reasoning models needs extra tokens for .. reasoning
    if is_reasoning_model:
        extra_gen_kwargs = ",max_gen_toks=4096"

    # GPT-OSS sometimes gets stuck in a reasoning loop. To ensure consistency
    # in CI, we add a repetition penalty which helps prevent the loop
    if "gpt-oss" in model:
        extra_gen_kwargs = extra_gen_kwargs + ",repetition_penalty=1.1"

    interpreter = sys.executable if _inside_bazel() else ".venv-eval/bin/python"

    include_path = str(Path(__file__).parent.resolve() / "tasks")
    with TemporaryDirectory() as tempdir:
        eval_cmd = [
            "lm_eval",
            f"--tasks={task}",
            "--model=local-chat-completions",
            "--log_samples",
            f"--model_args=model={model},base_url={URL},num_concurrent={max_concurrent},max_retries=1",
            "--apply_chat_template",
            f"--output_path={tempdir}",
            f"--limit={num_questions}",
            "--seed=42",
            f"--gen_kwargs=seed=42,temperature=0{extra_gen_kwargs}",
            f"--include_path={include_path}",
            "--fewshot_as_multiturn",
        ]

        args = [interpreter, "-m", *eval_cmd]
        logger.info(f"Running eval with:\n {' '.join(args)}")
        check_call(args, timeout=600)

        return parse_eval_results(Path(tempdir))


def parse_eval_results(loc: Path) -> tuple[EvalResults, EvalSamples]:
    samples = []
    for line in open(next(loc.glob("**/samples*.jsonl")), encoding="utf-8"):
        samples.append(json.loads(line))

    results = json.loads(next(loc.glob("**/results*.json")).read_text("utf-8"))

    return results, samples


def write_github_output(key: str, value: str) -> None:
    path = os.getenv("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"{key}={value}\n")


def gracefully_stop_process(process: Popen[bytes]) -> None:
    start_time = time.time()
    process.send_signal(signal.SIGINT)
    try:
        process.wait(25)
        shutdown_seconds = int(time.time() - start_time)
        logger.info(f"Server shutdown took {shutdown_seconds} seconds")
    except TimeoutExpired:
        logger.warning("Server did not stop after ctrl-c, trying SIGTERM")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(5)
        except ProcessLookupError:
            pass
        except TimeoutExpired:
            logger.warning("Process did not terminate gracefully, forcing kill")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(5)


@dataclass
class EvalSummary:
    gpu_name: str
    gpu_count: int
    startup_time_seconds: float
    eval_task: str
    task_type: str
    accuracy: float
    accuracy_stderr: float
    total_evaluation_time_seconds: float
    task_hash: str


def build_eval_summary(
    results: Sequence[Mapping[str, Any]],
    startup_time_seconds: float,
) -> list[EvalSummary]:
    """
    Extract the metrics from the eval results and build a summary for each task.
    """
    summaries = []

    for result in results:
        task = next(iter(result["results"].keys()))
        metrics = result["results"][task]
        total_secs = float(result["total_evaluation_time_seconds"])

        if VISION_TASK in task:
            accuracy = metrics["relaxed_accuracy,none"]
            accuracy_stderr = metrics["relaxed_accuracy_stderr,none"]
            task_type = "vision"
        elif task == TEXT_TASK:
            accuracy = metrics["exact_match,flexible-extract"]
            accuracy_stderr = metrics["exact_match_stderr,flexible-extract"]
            task_type = "text"
        else:
            raise ValueError(f"Unknown task: {task}")

        gpu_name, gpu_count = get_gpu_name_and_count()
        summaries.append(
            EvalSummary(
                gpu_name=gpu_name,
                gpu_count=gpu_count,
                startup_time_seconds=round(startup_time_seconds, 2),
                eval_task=task,
                task_type=task_type,
                accuracy=accuracy,
                accuracy_stderr=accuracy_stderr,
                total_evaluation_time_seconds=total_secs,
                task_hash=result["task_hashes"][task],
            )
        )

    return summaries


def print_samples(samples: EvalSamples, print_cot: bool) -> None:
    """
    Print question and the model's responses to each sample
    Assumes 'resps' is [[str]] (one decode) and GSM8K uses 'question',
    ChartQA uses 'query'.
    """
    for item in samples:
        doc = item.get("doc", {})
        question = doc.get("question") or doc.get("query")

        filt = item["filtered_resps"]
        extracted = filt[0] if isinstance(filt, list) and filt else str(filt)

        status = "✅" if item["exact_match"] == 1.0 else "❌"
        prefix_q = "🧮" if "question" in doc else "📊"

        logger.info(f"{prefix_q} {question}")
        if print_cot:
            logger.info(f"🤖💭 {item['resps'][0][0]}")
        logger.info(f"{status} {extracted}")


def start_server(cmd: list[str], timeout: int) -> tuple[Popen[bytes], float]:
    env = os.environ.copy()

    if not _inside_bazel():
        # SGLang depends on ninja which is in the serve environment
        env["PYTHONSAFEPATH"] = "1"  # Avoids root dir `max` shadowing
        venv_bin = os.path.abspath(".venv-serve/bin")
        prev_path = env.get("PATH")
        env["PATH"] = f"{venv_bin}:{prev_path}" if prev_path else venv_bin

    start = time.monotonic()
    proc = Popen(cmd, start_new_session=True, env=env)
    try:
        deadline = start + timeout
        while time.monotonic() < deadline:
            if server_is_ready():
                break
            if proc.poll() is not None:
                raise RuntimeError("Server process terminated unexpectedly")
            time.sleep(0.5)
        else:
            raise TimeoutError(f"Server did not start in {timeout} seconds")
        return proc, time.monotonic() - start
    except:
        gracefully_stop_process(proc)
        raise


def write_results(
    path: Path,
    summary: list[EvalSummary],
    results: list[EvalResults],
    all_samples: list[EvalSamples],
    tasks: list[str],
) -> None:
    summary_file = path / "eval_metrics.json"
    summary_json = json.dumps([asdict(s) for s in summary], indent=2)
    summary_file.write_text(summary_json, encoding="utf-8")
    for result, samples, task in zip(results, all_samples, tasks, strict=True):
        timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
        result_fp = path / f"results_{task}_{timestamp}.json"
        result_fp.write_text(json.dumps(result, indent=2), encoding="utf-8")

        samples_fp = path / f"samples_{task}_{timestamp}.jsonl"
        with open(samples_fp, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")


@click.command()
@click.argument(
    "hf_model_path",
    type=str,
    required=True,
)
@click.option(
    "--framework",
    type=click.Choice(["sglang", "vllm", "max", "max-ci"]),
    default="max-ci",
    required=False,
    help="Framework to use for the smoke test. Only max-ci is supported when running in bazel.",
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
    help="Maximum concurrent requests to send to the server",
)
@click.option(
    "--num-questions",
    type=int,
    default=320,
    help="Number of questions to ask the model",
)
@click.option(
    "--serve-extra-args",
    type=str,
    default="",
    help=(
        "Extra args appended to MAX serve command, for example: "
        '"--device-graph-capture --max-batch-size=16"'
    ),
)
def smoke_test(
    hf_model_path: str,
    framework: str,
    output_path: Path | None,
    print_responses: bool,
    print_cot: bool,
    max_concurrent: int,
    num_questions: int,
    serve_extra_args: str,
) -> None:
    """
    Example usage: ./bazelw run smoke-test -- meta-llama/llama-3.2-1b-instruct

    This command asks 320 questions against the model behind the given hf_model_path.
    It runs the provided framework (defaulting to MAX serve) in the background,
    and fires off HTTP requests to chat/completions API.
    Note: Only models with a chat template (typically -instruct, -it, -chat, etc.) are supported.

    Accuracy is reported at the end, with higher values being better.
    A 1.0 value means 100% accuracy.

    """
    validate_hf_token()

    if print_cot and not print_responses:
        raise ValueError("--print-cot must be used with --print-responses")

    build_workspace = os.getenv("BUILD_WORKSPACE_DIRECTORY")
    if output_path and build_workspace and not output_path.is_absolute():
        output_path = Path(build_workspace) / output_path

    model = hf_model_path.lower().strip()
    cmd = get_server_cmd(
        framework,
        model,
        serve_extra_args=serve_extra_args,
    )

    # TODO Refactor this to a model list/matrix specifying type of model
    is_vision_model = any(
        kw in model
        for kw in (
            "gemma-3",
            "idefics",
            "internvl",
            "olmocr",
            "pixtral",
            "qwen2.5-vl",
            "qwen3-vl",
            "vision",
        )
    )
    # 1b is non-vision
    if "gemma-3-1b" in model:
        is_vision_model = False

    tasks = [TEXT_TASK]
    if is_vision_model:
        tasks = [VISION_TASK] + tasks

    logger.info(f"Starting server with command:\n {' '.join(cmd)}")
    results = []
    all_samples = []
    timeout = 900
    if is_deepseek(model):
        timeout = 1800

    server_process, startup_time = start_server(cmd, timeout)
    try:
        logger.info(f"Server started in {startup_time:.2f} seconds")
        write_github_output("startup_time", f"{startup_time:.2f}")

        for task in tasks:
            test_single_request(model, task)
            result, samples = call_eval(
                model,
                task,
                max_concurrent=max_concurrent,
                num_questions=num_questions,
            )
            if print_responses:
                print_samples(samples, print_cot)

            results.append(result)
            all_samples.append(samples)
    finally:
        try:
            gracefully_stop_process(server_process)
        except Exception:
            logger.exception(f"Failed to shutdown {framework.upper()}")

    if results:
        summary = build_eval_summary(results, startup_time_seconds=startup_time)

        if output_path is not None:
            path = output_path / safe_model_name(model)
            path.mkdir(parents=True, exist_ok=True)
            write_results(path, summary, results, all_samples, tasks)

            logger.info(pformat(summary, indent=2))


if __name__ == "__main__":
    smoke_test()
