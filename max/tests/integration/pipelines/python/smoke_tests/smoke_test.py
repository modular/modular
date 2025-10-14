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
"""

import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from subprocess import Popen

import click
import requests

DUMMY_2X2_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP8zwACTGCSAQANHQEDgslx/wAAAABJRU5ErkJggg=="
)
URL = "http://localhost:8000/v1/chat/completions"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    r = requests.post(
        URL,
        json={"model": model, "messages": m, "max_tokens": 8, "temperature": 0},
        timeout=10,
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
    return commands[framework] + [
        "--port",
        "8000",
        "--trust-remote-code",
        "--model",
        model,
    ]


def safe_model_name(model: str) -> str:
    return model.replace("/", "__")


def get_lm_eval_cmd(model: str, task: str) -> list[str]:
    max_gen_toks = {
        "unsloth/gpt-oss-20b-bf16": ",max_gen_toks=50000",
        "qwen/qwen3-8b": ",max_gen_toks=4096",
    }.get(model, "")

    if task == "chartqa":
        # Chartqa has a bug in lm-eval, so we use a local version until it's fixed
        task = str(Path(__file__).parent.resolve() / "chartqa")

    return [
        ".venv-lm-eval/bin/python",
        "-m",
        "lm_eval",
        "--model",
        "local-chat-completions",
        "--model_args",
        f"model={model},base_url={URL},num_concurrent=64",
        "--tasks",
        task,
        "--fewshot_as_multiturn",
        "--apply_chat_template",
        "--output_path",
        "./lm_eval_output",
        "--limit",
        "320",
        "--seed",
        "42",
        "--gen_kwargs",
        f"top_p=1,top_k=1,temperature=0,seed=42{max_gen_toks}",
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


def locate_results_json(model: str) -> Path:
    sanitized_name = safe_model_name(model)
    results_dir = Path("./lm_eval_output") / Path(sanitized_name)
    candidates = sorted(
        results_dir.glob("results_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No results_*.json under {results_dir}")
    return candidates[0]


@dataclass
class EvalSummary:
    startup_time_seconds: float
    eval_task: str
    accuracy: float
    accuracy_stderr: float
    total_evaluation_time_seconds: float
    task_hash: str


def build_eval_summary(
    results_json: Path, startup_time_seconds: float, task: str
) -> EvalSummary:
    """
    Parse lm-eval's results JSON and extract the metrics
    """
    data = json.loads(results_json.read_text())
    metrics = data["results"][task]
    total_secs = float(data["total_evaluation_time_seconds"])

    if task == "chartqa":
        accuracy = metrics["relaxed_accuracy,none"]
        accuracy_stderr = metrics["relaxed_accuracy_stderr,none"]
    elif task == "gsm8k_cot_llama":
        accuracy = metrics["exact_match,flexible-extract"]
        accuracy_stderr = metrics["exact_match_stderr,flexible-extract"]
    else:
        raise ValueError(f"Unknown task: {task}")

    return EvalSummary(
        startup_time_seconds=round(startup_time_seconds, 2),
        eval_task=task,
        accuracy=accuracy,
        accuracy_stderr=accuracy_stderr,
        total_evaluation_time_seconds=total_secs,
        task_hash=data["task_hashes"][task],
    )


@click.command()
@click.option(
    "--framework",
    type=click.Choice(["sglang", "vllm", "max", "max-ci"]),
    required=True,
)
@click.option(
    "--model",
    type=str,
    required=True,
    help="huggingface model path, for example: unsloth/gpt-oss-20b-bf16",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="If provided, write the summary of the smoke test to this file",
)
def smoke_test(framework: str, model: str, output_file: Path | None) -> None:
    model = model.lower().strip()
    cmd = get_server_cmd(framework, model)

    # TODO Refactor this to a model list/matrix specifying type of model
    is_vision_model = any(
        kw in model
        for kw in ("qwen2.5-vl", "vision", "internvl", "idefics", "pixtral")
    )
    task = "chartqa" if is_vision_model else "gsm8k_cot_llama"

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
            test_single_request(model, task)
        except Exception as e:
            gracefully_stop_process(server_process)
            raise Exception(f"Test single request failed: {e}") from e

        try:
            lm_eval_cmd = get_lm_eval_cmd(model, task)
            logger.info(f"Starting lm-eval with command:\n {lm_eval_cmd}")
            with Popen(lm_eval_cmd, start_new_session=True) as lm_eval_process:
                try:
                    rc = lm_eval_process.wait(600)
                except subprocess.TimeoutExpired:
                    gracefully_stop_process(lm_eval_process)
                    raise
                if rc != 0:
                    raise Exception(f"lm-eval exited with non-zero status {rc}")
        finally:
            gracefully_stop_process(server_process)

    results_json = locate_results_json(model)
    summary = build_eval_summary(
        results_json, startup_time_seconds=startup_time, task=task
    )

    if output_file is not None:
        output_file.write_text(json.dumps(asdict(summary), indent=2))
        logger.info(f"Wrote EvalSummary JSON to {output_file.resolve()}")

    time.sleep(1)
    logger.info("Smoke test completed")
    logger.info(f"EvalSummary: {json.dumps(asdict(summary), indent=2)}")


if __name__ == "__main__":
    smoke_test()
