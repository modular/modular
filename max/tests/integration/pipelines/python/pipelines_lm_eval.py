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
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import click
import python.runfiles
import requests

logger = logging.getLogger("pipelines_lm_eval")


def _must_rlocation(runfiles: python.runfiles.Runfiles, rloc: str) -> Path:
    loc = runfiles.Rlocation(rloc)
    if loc is None:
        raise FileNotFoundError(
            f"Required rlocation {rloc!r} could not be resolved"
        )
    return Path(loc)


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
@click.option(
    "--override-lm-eval",
    type=click.Path(
        exists=True, executable=True, dir_okay=False, path_type=Path
    ),
)
@click.option("--lm-eval-arg", "lm_eval_args", multiple=True)
def main(
    override_pipelines: Optional[Path],
    pipelines_probe_port: Optional[int],
    pipelines_probe_timeout: Optional[int],
    pipelines_args: Sequence[str],
    override_lm_eval: Optional[Path],
    lm_eval_args: Sequence[str],
) -> None:
    """Start pipelines server, run lm-eval, and then shut down server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(name)s: %(message)s",
    )

    runfiles = python.runfiles.Create()
    if runfiles is None:
        raise FileNotFoundError("Unable to find runfiles tree")
    if override_pipelines is not None:
        pipelines_path = override_pipelines
    else:
        pipelines_path = _must_rlocation(
            runfiles, "_main/SDK/public/max-repo/pipelines/python/pipelines"
        )
    if override_lm_eval is not None:
        lm_eval_path = override_lm_eval
    else:
        lm_eval_path = _must_rlocation(
            runfiles,
            "_main/SDK/integration-test/pipelines/python/run_lm_eval.py",
        )
    logger.debug("Pipelines binary at: %s", pipelines_path)
    logger.debug("lm-eval binary at: %s", lm_eval_path)

    lm_eval_args = list(lm_eval_args)
    if not any(arg.startswith("--include_path") for arg in lm_eval_args):
        include_path = _must_rlocation(
            runfiles,
            "_main/SDK/integration-test/pipelines/python/eval_tasks/BUILD.bazel",
        ).parent
        lm_eval_args.append(f"--include_path={include_path}")
        logger.debug("Including path: %s", include_path)

    with PipelineSitter(
        [str(pipelines_path)] + list(pipelines_args)
    ) as pipeline_sitter:
        if pipelines_probe_port is not None:
            pipeline_sitter.wait_for_alive(
                probe_port=pipelines_probe_port, timeout=pipelines_probe_timeout
            )
        logger.info(f"Running lm-eval with provided args: {lm_eval_args}")
        lm_eval_proc = subprocess.run(
            [sys.executable, str(lm_eval_path)] + lm_eval_args
        )
        logger.info(
            f"lm-eval exited with status code {lm_eval_proc.returncode}"
        )
    sys.exit(lm_eval_proc.returncode)


if __name__ == "__main__":
    main()
