# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
This script is used for the CI "Max Serve Smoke Test" workflow.
It does two things:
    1. Starts the MAX/SGLang/VLLM inference server for the given model
    2. Runs a tiny evaluation task using against the chat/completions API

Currently there is a hard dependency that two virtualenvs are already created:
    - .venv-serve (not needed for max-ci, which uses bazel)
    - .venv-lm-eval

Where the serve environment should already have either MAX/VLLM/SGLang installed.
The lm-eval environment should already have lm-eval installed.
These dependencies are to be removed once this script
has been integrated into bazel.

Note that if you're running this script inside bazel, only available for max-ci,
then the virtualenvs are not needed.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from subprocess import Popen
from tempfile import TemporaryDirectory

import click
import requests

DUMMY_2X2_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP8zwACTGCSAQANHQEDgslx/wAAAABJRU5ErkJggg=="
)
URL = "http://localhost:8000/v1/chat/completions"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _inside_bazel() -> bool:
    return os.getenv("BUILD_WORKSPACE_DIRECTORY") is not None


def test_single_request(model: str, task: str) -> None:
    is_vision = task == "chartqa"
    if is_vision:
        m = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Say 'hello image'",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": DUMMY_2X2_IMAGE},
                    },
                ],
            }
        ]
    else:
        m = [{"role": "user", "content": "Say: 'hello world'"}]

    connection_timeout = 10
    read_timeout = 60
    r = requests.post(
        URL,
        json={"model": model, "messages": m, "max_tokens": 8, "temperature": 0},
        timeout=(connection_timeout, read_timeout),
    )
    r.raise_for_status()
    resp = r.json()["choices"][0]["message"]["content"]
    logger.info(f"Test single request OK. Response: {resp}")


def get_gpu_model() -> str:
    try:
        # Try AMD first
        result = subprocess.check_output(
            ["rocm-smi", "--showid", "--json"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        data = json.loads(result)
        return next(iter(data.values()))["Device Name"]
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        StopIteration,
        KeyError,
    ):
        # Fallback to NVIDIA
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
                "-i",
                "0",
            ],
            text=True,
        ).strip()


def server_is_ready() -> bool:
    health_url = "http://localhost:8000/health"
    try:
        return requests.get(health_url, timeout=1).status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_server_cmd(framework: str, model: str) -> list[str]:
    if _inside_bazel():
        assert framework == "max-ci", (
            "Only max-ci supported when running in bazel"
        )
        cmd = [sys.executable, "-m", "max.entrypoints.pipelines", "serve"]
    else:
        interpreter = [".venv-serve/bin/python", "-m"]
        commands = {
            "sglang": [
                *interpreter,
                "sglang.launch_server",
                "--host",
                "0.0.0.0",
                "--attention-backend",
                "triton" if "b200" in get_gpu_model().lower() else "fa3",
                "--mem-fraction-static",
                "0.8",
            ],
            "vllm": [
                *interpreter,
                "vllm.entrypoints.openai.api_server",
                "--host",
                "0.0.0.0",
                "--max-model-len",
                "16384",
                "--limit-mm-per-prompt.video",  # Needed for InternVL 3 on B200
                "0",
            ],
            "max": [*interpreter, "max.entrypoints.pipelines", "serve"],
            "max-ci": [
                "./bazelw",
                "run",
                "--config=ci",
                "--config=disable-mypy",
                "max",
                "--",
                "serve",
            ],
        }
        cmd = commands[framework]
    return cmd + [
        "--port",
        "8000",
        "--trust-remote-code",
        "--model",
        model,
    ]


def safe_model_name(model: str) -> str:
    return model.replace("/", "__")


def get_lm_eval_cmd(model: str, task: str, output_path: Path) -> list[str]:
    max_gen_toks = {
        "unsloth/gpt-oss-20b-bf16": ",max_gen_toks=50000",
        "qwen/qwen3-8b": ",max_gen_toks=4096",
        "opengvlab/internvl3_5-8b-instruct": ",max_gen_toks=4096",
    }.get(model, "")

    gen_params = {
        "qwen/qwen3-8b": ",temperature=0.6,top_p=0.95,top_k=20",
        "opengvlab/internvl3_5-8b-instruct": ",temperature=0.6,top_p=0.95,top_k=20",
    }.get(model, ",top_p=1,top_k=1,temperature=0")

    interpreter = (
        sys.executable if _inside_bazel() else ".venv-lm-eval/bin/python"
    )

    return [
        interpreter,
        "-m",
        "lm_eval",
        "--model",
        "local-chat-completions",
        "--model_args",
        f"model={model},base_url={URL},num_concurrent=64",
        "--include_path",
        str(Path(__file__).parent.resolve() / "chartqa_modular"),
        "--tasks",
        task,
        "--fewshot_as_multiturn",
        "--apply_chat_template",
        "--output_path",
        f"{output_path}/lm_eval_output",
        "--limit",
        "320",
        "--seed",
        "42",
        "--gen_kwargs",
        f"seed=42{max_gen_toks}{gen_params}",
        "--log_samples",
    ]


def write_github_output(key: str, value: str) -> None:
    path = os.getenv("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"{key}={value}\n")


def gracefully_stop_process(process: Popen) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(5)
    except subprocess.TimeoutExpired:
        logger.warning("Process did not terminate gracefully, forcing kill")
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait(5)


def parse_results(
    model: str, tasks: list[str], output_path: Path
) -> list[dict]:
    """
    Returns a list of the lm-eval results for the given model and tasks.
    """
    sanitized_name = safe_model_name(model)
    results_dir = output_path / "lm_eval_output" / sanitized_name
    candidates = sorted(
        results_dir.glob("results_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if len(candidates) < len(tasks):
        raise FileNotFoundError(
            f"No results_*.json under {results_dir} for all tasks"
        )

    result_dicts = []
    for index, task in enumerate(reversed(tasks)):
        result = json.loads(candidates[index].read_text())
        if task not in result["results"]:
            raise FileNotFoundError(
                f"Did not find results for task {task} in {result}"
            )
        result_dicts.append(result)

    return result_dicts


@dataclass
class EvalSummary:
    startup_time_seconds: float
    eval_task: str
    accuracy: float
    accuracy_stderr: float
    total_evaluation_time_seconds: float
    task_hash: str


def build_eval_summary(
    results: Sequence[Mapping], startup_time_seconds: float
) -> list[EvalSummary]:
    """
    Extract the metrics from the lm-eval results and build a summary for each task.
    """
    summaries = []

    for result in results:
        assert len(result["results"]) == 1, "Expected exactly one task result"
        task = next(iter(result["results"].keys()))
        metrics = result["results"][task]
        total_secs = float(result["total_evaluation_time_seconds"])

        if "chartqa" in task:
            accuracy = metrics["relaxed_accuracy,none"]
            accuracy_stderr = metrics["relaxed_accuracy_stderr,none"]
        elif task == "gsm8k_cot_llama":
            accuracy = metrics["exact_match,flexible-extract"]
            accuracy_stderr = metrics["exact_match_stderr,flexible-extract"]
        else:
            raise ValueError(f"Unknown task: {task}")

        summaries.append(
            EvalSummary(
                startup_time_seconds=round(startup_time_seconds, 2),
                eval_task=task,
                accuracy=accuracy,
                accuracy_stderr=accuracy_stderr,
                total_evaluation_time_seconds=total_secs,
                task_hash=result["task_hashes"][task],
            )
        )

    return summaries


def _latest_samples_files(
    model: str, tasks: Sequence[str], output_path: Path
) -> list[Path]:
    """
    Return the newest samples_*.jsonl file for each task we ran.
    Assumes lm-eval was invoked with --log_samples.
    """
    out: list[Path] = []
    results_dir = output_path / "lm_eval_output" / safe_model_name(model)
    if not results_dir.exists():
        return out
    for task in tasks:
        candidates = sorted(
            results_dir.glob(f"samples_{task}*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            out.append(candidates[0])
    return out


def _print_samples(files: Sequence[Path], print_cot: bool) -> None:
    """
    Print question/query and the model's first response from samples files.
    Assumes 'resps' is [[str]] (one decode) and GSM8K uses 'question',
    ChartQA uses 'query'.
    """
    if not files:
        logger.info("No lm-eval sample files found to print.")
        return

    for fpath in files:
        logger.info(f"\n--- Printing model responses from {fpath.name} ---\n")
        with fpath.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                item = json.loads(line)
                doc = item.get("doc", {})
                question = doc.get("question") or doc.get("query") or ""
                resps = item["resps"]
                resp_text: str = resps[0][0]

                target = str(item.get("target")) if "target" in item else None

                filt = item["filtered_resps"]
                extracted = (
                    filt[0] if isinstance(filt, list) and filt else str(filt)
                )

                correct = target is not None and extracted == target
                status = "✅" if correct else "❌"

                prefix_q = "🧮" if "question" in doc else "📊"
                logger.info(f"{prefix_q} {question}")
                if print_cot:
                    logger.info(f"🤖💭 {resp_text}")
                logger.info(f"{status} {extracted}")


def smoke_test(
    hf_model_path: str,
    framework: str,
    output_path: Path,
    write_summary: bool,
    print_responses: bool,
    print_cot: bool,
) -> None:
    model = hf_model_path.lower().strip()
    cmd = get_server_cmd(framework, model)

    # TODO Refactor this to a model list/matrix specifying type of model
    is_vision_model = any(
        kw in model
        for kw in ("qwen2.5-vl", "vision", "internvl", "idefics", "pixtral")
    )
    tasks = ["gsm8k_cot_llama"]
    if is_vision_model:
        tasks.append("chartqa")

    # SGLang depends on ninja which is in the serve environment
    env = os.environ.copy()
    venv_bin = os.path.abspath(".venv-serve/bin")
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

    logger.info(f"Starting server with command:\n {cmd}")
    with Popen(cmd, start_new_session=True, env=env) as server_process:
        script_start_time = time.perf_counter()
        while not server_is_ready():
            if server_process.poll() is not None:
                raise Exception("Server process terminated unexpectedly")
            elif time.perf_counter() - script_start_time > 600:
                raise Exception("Server did not start in 600 seconds")
            time.sleep(0.5)

        startup_time = time.perf_counter() - script_start_time
        logger.info(f"Server started in {startup_time:.2f} seconds")
        write_github_output("startup_time", f"{startup_time:.2f}")

        try:
            for task in tasks:
                test_single_request(model, task)

                lm_eval_cmd = get_lm_eval_cmd(model, task, output_path)
                logger.info(f"Starting lm-eval with command:\n {lm_eval_cmd}")
                with Popen(
                    lm_eval_cmd, start_new_session=True
                ) as lm_eval_process:
                    try:
                        rc = lm_eval_process.wait(600)
                    except subprocess.TimeoutExpired:
                        logger.warning("lm-eval timed out, killing process")
                        gracefully_stop_process(lm_eval_process)
                        raise
                    if rc != 0:
                        raise Exception(
                            f"lm-eval exited with non-zero status {rc}"
                        )
        finally:
            gracefully_stop_process(server_process)

    results = parse_results(model, tasks, output_path)
    summary = build_eval_summary(results, startup_time_seconds=startup_time)

    summary_json = json.dumps([asdict(s) for s in summary], indent=2)
    if write_summary:
        file_name = safe_model_name(hf_model_path) + "_eval_metrics.json"
        output_file = output_path / file_name
        output_file.write_text(summary_json)
        logger.info(f"Wrote EvalSummary JSON to {output_file.resolve()}")

    if print_responses:
        files = _latest_samples_files(model, tasks, output_path)
        _print_samples(files, print_cot)

    time.sleep(1)
    logger.info("Smoke test completed")
    logger.info(f"EvalSummary: {summary_json}")


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
    help="If provided, a summary json file and the lm-eval result are written here",
)
@click.option(
    "--print-responses",
    is_flag=True,
    default=False,
    help="Print question/response pairs from lm-eval samples after the run finishes",
)
@click.option(
    "--print-cot",
    is_flag=True,
    default=False,
    help="Print the model's chain-of-thought reasoning for each sample. Must be used with --print-responses",
)
def smoke_test_cli(
    hf_model_path: str,
    framework: str,
    output_path: Path | None,
    print_responses: bool,
    print_cot: bool,
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
    if print_cot and not print_responses:
        raise ValueError("--print-cot must be used with --print-responses")

    if output_path is None:
        with TemporaryDirectory() as tempdir:
            output_path = Path(tempdir)
            smoke_test(
                hf_model_path,
                framework,
                output_path,
                False,
                print_responses,
                print_cot,
            )
    else:
        smoke_test(
            hf_model_path,
            framework,
            output_path,
            True,
            print_responses,
            print_cot,
        )


if __name__ == "__main__":
    smoke_test_cli()
