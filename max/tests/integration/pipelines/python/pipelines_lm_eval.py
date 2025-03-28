# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import logging
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Optional

import click
import python.runfiles
import requests

logger = logging.getLogger("pipelines_lm_eval")


def _must_rlocation_str(runfiles: python.runfiles.Runfiles, rloc: str) -> str:
    loc = runfiles.Rlocation(rloc)
    if loc is None:
        raise FileNotFoundError(
            f"Required rlocation {rloc!r} could not be resolved"
        )
    return loc


def _must_rlocation(runfiles: python.runfiles.Runfiles, rloc: str) -> Path:
    return Path(_must_rlocation_str(runfiles, rloc))


class PipelineSitter:
    """Owns the pipelines process and manages its startup/shutdown."""

    _args: Sequence[str]
    _proc: Optional[subprocess.Popen]

    def __init__(self, args: Sequence[str]) -> None:
        self._args = args
        self._proc = None

    def __enter__(self) -> PipelineSitter:
        self.start()
        return self

    def __exit__(
        self, exc_type: Any, exc_value: Any, exc_tb: Any
    ) -> Literal[False]:
        self.stop()
        return False

    def start(self) -> None:
        if self._proc:
            return
        logger.info(
            f"Starting pipelines process with provided args: {self._args}"
        )
        self._proc = subprocess.Popen(self._args)
        logger.info("Pipelines process started")

    def stop(self) -> None:
        if not self._proc:
            return
        logger.info("Sending pipelines process SIGTERM")
        self._proc.terminate()
        logger.info("Waiting for pipelines process to terminate")
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Pipelines process did not terminate after 5 seconds, sending"
                " SIGKILL"
            )
            self._proc.kill()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Well, we tried our best.
                logger.error(
                    "Pipelines process still did not die; continuing anyway"
                )
            else:
                logger.info("Pipelines process terminated")
        else:
            logger.info("Pipelines process terminated")
        self._proc = None

    def wait_for_alive(
        self, probe_port: int, *, timeout: Optional[float]
    ) -> None:
        assert self._proc is not None
        probe_url = f"http://127.0.0.1:{probe_port}/metrics"
        start_time = time.time()
        deadline: Optional[float]
        if timeout is None:
            deadline = None
        else:
            deadline = start_time + timeout
        logger.info("Waiting for pipelines server to begin accepting requests")
        while deadline is None or (now := time.time()) < deadline:
            try:
                self._proc.wait(timeout=0)
            except subprocess.TimeoutExpired:
                pass
            else:
                self._proc = None
                logger.error(
                    "Pipelines server died while waiting for readiness"
                )
                raise Exception(
                    "Pipelines server died while waiting for readiness"
                )
            attempt_timeout = 5.0
            if deadline is not None:
                remaining_time = deadline - now
                attempt_timeout = min(attempt_timeout, remaining_time)
            try:
                requests.get(probe_url, timeout=attempt_timeout)
            except Exception:
                pass
            else:
                logger.info(
                    "Pipelines server seems to now be accepting requests"
                )
                return
        logger.error(
            "Pipelines server did not begin accepting requests within deadline"
            f" of {timeout} seconds"
        )
        raise Exception("Pipelines server did not come up within timeout")


@click.command()
@click.option(
    "--override-pipelines",
    type=click.Path(
        exists=True, executable=True, dir_okay=False, path_type=Path
    ),
)
@click.option("--pipelines-probe-port", type=int)
@click.option("--pipelines-probe-timeout", type=float)
@click.option("--pipelines-arg", "pipelines_args", multiple=True)
@click.option("--evaluator", type=str, default="lm-eval")
@click.option(
    "--override-lm-eval",
    type=click.Path(
        exists=True, executable=True, dir_okay=False, path_type=Path
    ),
)
@click.option("--lm-eval-arg", "lm_eval_args", multiple=True)
@click.option(
    "--override-mistral-evals",
    type=click.Path(
        exists=True, executable=True, dir_okay=False, path_type=Path
    ),
)
@click.option("--mistral-evals-arg", "mistral_evals_args", multiple=True)
def main(
    override_pipelines: Optional[Path],
    pipelines_probe_port: Optional[int],
    pipelines_probe_timeout: Optional[int],
    pipelines_args: Sequence[str],
    evaluator: str,
    override_lm_eval: Optional[Path],
    lm_eval_args: Sequence[str],
    override_mistral_evals: Optional[Path],
    mistral_evals_args: Sequence[str],
) -> None:
    """Start pipelines server, run an evaluator, and then shut down server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(name)s: %(message)s",
    )

    runfiles = python.runfiles.Create()
    if runfiles is None:
        raise FileNotFoundError("Unable to find runfiles tree")
    if override_pipelines is not None:
        pipelines_program = [str(override_pipelines)]
    else:
        pipelines_program = [
            _must_rlocation_str(
                runfiles,
                "_main/SDK/lib/API/python/max/entrypoints/pipelines",
            )
        ]
    if override_lm_eval is not None:
        lm_eval_program = [str(override_lm_eval)]
    else:
        lm_eval_program = [
            sys.executable,
            _must_rlocation_str(
                runfiles,
                "_main/SDK/integration-test/pipelines/python/run_lm_eval.py",
            ),
        ]
    if override_mistral_evals is not None:
        mistral_evals_program = [str(override_mistral_evals)]
    else:
        mistral_evals_program = [
            _must_rlocation_str(runfiles, "mistral-evals/evaluate")
        ]
    logger.debug("Pipelines binary at: %r", pipelines_program)
    evaluator_args: list[str] = []
    if evaluator == "lm-eval":
        evaluator_program = lm_eval_program
        evaluator_args.extend(lm_eval_args)
    elif evaluator == "mistral-evals":
        evaluator_program = mistral_evals_program
        evaluator_args.extend(mistral_evals_args)
    else:
        logger.error("Unrecognized evaluator %r", evaluator)
        sys.exit(1)
    logger.debug("Evaluator binary at: %r", evaluator_program)

    if evaluator == "lm-eval" and not any(
        arg.startswith("--include_path") for arg in evaluator_args
    ):
        include_path = _must_rlocation(
            runfiles,
            "_main/SDK/integration-test/pipelines/python/eval_tasks/BUILD.bazel",
        ).parent
        evaluator_args.append(f"--include_path={include_path}")
        logger.debug("Including path: %s", include_path)

    with PipelineSitter(
        pipelines_program + list(pipelines_args)
    ) as pipeline_sitter:
        if pipelines_probe_port is not None:
            pipeline_sitter.wait_for_alive(
                probe_port=pipelines_probe_port, timeout=pipelines_probe_timeout
            )
        logger.info(
            "Running evaluator %r with provided args: %r",
            evaluator_program,
            evaluator_args,
        )
        evaluator_proc = subprocess.run(evaluator_program + evaluator_args)
        logger.info(
            "Evaluator exited with status code %s", evaluator_proc.returncode
        )
    sys.exit(evaluator_proc.returncode)


if __name__ == "__main__":
    main()
