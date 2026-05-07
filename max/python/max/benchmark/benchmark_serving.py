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

"""Benchmark online serving throughput."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import shlex
import subprocess
import sys
import time
from collections.abc import (
    Generator,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np
import yaml
from cyclopts import App, Parameter
from cyclopts.config import Env
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from max.benchmark.benchmark_shared.server_metrics import ParsedMetrics
    from max.diagnostics.gpu import BackgroundRecorder as GPUBackgroundRecorder
    from max.diagnostics.gpu import GPUStats

from max.benchmark.benchmark_shared.config import (
    CACHE_RESET_ENDPOINT_MAP,
    PIXEL_GEN_DEFAULT_ENDPOINT,
    PIXEL_GENERATION_ENDPOINTS,
    PIXEL_GENERATION_TASKS,
    Backend,
    BenchmarkTask,
    Endpoint,
    ServingBenchmarkConfig,
)
from max.benchmark.benchmark_shared.datasets.all import sample_requests
from max.benchmark.benchmark_shared.datasets.types import (
    ChatSamples,
    RequestSamples,
    Samples,
)
from max.benchmark.benchmark_shared.lora_benchmark_manager import (
    LoRABenchmarkManager,
)
from max.benchmark.benchmark_shared.metrics import (
    PixelGenerationBenchmarkResult,
    SpecDecodeMetrics,
    TextGenerationBenchmarkResult,
    calculate_spec_decode_stats,
)
from max.benchmark.benchmark_shared.multi_turn import (
    prime_prefix_turns,
    run_kv_cache_stress_benchmark,
    run_multiturn_benchmark,
)
from max.benchmark.benchmark_shared.request import (
    BaseRequestFuncOutput,
    PixelGenerationRequestFuncOutput,
    ProgressBarRequestDriver,
    RequestDriver,
    RequestFuncOutput,
    get_request_driver_class,
)
from max.benchmark.benchmark_shared.server_metrics import (
    collect_benchmark_metrics,
    fetch_spec_decode_metrics,
)
from max.benchmark.benchmark_shared.serving_metrics import (
    build_pixel_generation_result,
    build_text_generation_result,
    compute_output_len,
)
from max.benchmark.benchmark_shared.serving_result_output import (
    print_benchmark_summary,
    print_input_prompts,
    print_workload_stats,
    save_output_lengths,
    save_result_json,
)
from max.benchmark.benchmark_shared.single_turn import (
    prime_shared_contexts,
    run_single_test_prompt,
    run_single_turn_benchmark,
)
from max.benchmark.benchmark_shared.utils import (
    argmedian,
    get_tokenizer,
    int_or_none,
    is_castable_to_int,
    parse_comma_separated,
    print_section,
    set_ulimit,
    wait_for_server_ready,
)
from max.benchmark.benchmark_shared.warmup import (
    log_warmup_sampling_report,
    pick_warmup_population,
)
from max.diagnostics.cpu import (
    CPUMetrics,
    CPUMetricsCollector,
    collect_pids_for_port,
)
from max.diagnostics.gpu import GPUDiagContext
from openai.types.chat.completion_create_params import ResponseFormat
from pydantic import TypeAdapter, ValidationError

BENCHMARK_SERVING_ARGPARSER_DESCRIPTION = (
    "This command runs comprehensive benchmark tests on a model server to"
    " measure performance metrics including throughput, latency, and resource"
    " utilization. Make sure that the MAX server is running and hosting a model"
    " before running this command."
)

logger = logging.getLogger(__name__)


def parse_response_format(arg: str) -> ResponseFormat:
    """Parse response format from CLI arg (inline JSON or @filepath).

    Args:
        arg: Either a JSON string or '@path/to/schema.json' to load from file.

    Returns:
        Validated ResponseFormat.

    Raises:
        ValueError: If the JSON is invalid, the file cannot be read, or the
            value does not match a recognised OpenAI response format.
    """
    if arg.startswith("@"):
        # Load from file
        file_path = Path(arg[1:])
        try:
            raw = file_path.read_text()
        except FileNotFoundError as e:
            raise ValueError(
                f"Response format file not found: {file_path}"
            ) from e
        try:
            return TypeAdapter(ResponseFormat).validate_json(raw)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(
                f"Invalid response format in file {file_path}: {e}"
            ) from e

    # Parse inline JSON
    try:
        return TypeAdapter(ResponseFormat).validate_json(arg)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Invalid response format: {e}") from e


def get_default_trace_path() -> str:
    """Get the default trace output path."""
    workspace_path = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace_path:
        return os.path.join(workspace_path, "profile.nsys-rep")
    return "./profile.nsys-rep"


def assert_nvidia_gpu() -> None:
    """Raise an exception if no NVIDIA GPUs are available."""
    with GPUDiagContext() as ctx:
        stats = ctx.get_stats()
        if not stats:
            raise RuntimeError(
                "No GPUs detected. The --trace flag currently only works with NVIDIA GPUs."
            )
        if not any(gpu_name.startswith("nv") for gpu_name in stats):
            raise RuntimeError(
                "The --trace flag currently only works with NVIDIA GPUs. "
                f"Found GPUs: {list(stats.keys())}"
            )


@contextlib.contextmanager
def under_nsys_tracing(
    output_path: str, session_name: str | None = None
) -> Generator[None, None, None]:
    """Run some code under nsys tracing."""
    start_cmd = ["nsys", "start", "-o", output_path, "--force-overwrite=true"]
    stop_cmd = ["nsys", "stop"]
    if session_name:
        start_cmd.extend(["--session", session_name])
        stop_cmd.extend(["--session", session_name])
    logger.info(f"Starting nsys trace: {shlex.join(start_cmd)}")
    subprocess.run(start_cmd, check=True)
    try:
        yield
    finally:
        logger.info(f"Stopping nsys trace: {shlex.join(stop_cmd)}")
        subprocess.run(stop_cmd, check=True)


def create_benchmark_pbar(disable_tqdm: bool, samples: Samples) -> tqdm | None:
    """Create a progress bar for benchmark runs.

    Args:
        disable_tqdm: Whether to disable the progress bar.
        samples: Samples that will be benchmarked with.

    Returns:
        A tqdm progress bar instance or None if disabled.
    """
    if disable_tqdm:
        return None

    if isinstance(samples, RequestSamples):
        # single-turn chat scenario
        return tqdm(total=len(samples.requests))
    else:
        # multi-turn chat scenario
        num_qa_turns = [session.num_turns for session in samples.chat_sessions]
        return tqdm(total=sum(num_qa_turns))


async def benchmark(
    args: ServingBenchmarkConfig,
    session: BenchmarkSession,
    max_concurrency: int | None,
    request_rate: float,
) -> TextGenerationBenchmarkResult | PixelGenerationBenchmarkResult:
    """Run a single benchmark invocation.

    ``session.orig_skip_first`` / ``session.orig_skip_last`` are the
    user-supplied values (``None`` = auto-derive from *max_concurrency*).
    """
    backend: Backend = args.backend

    skip_first = session.orig_skip_first
    skip_last = session.orig_skip_last
    if request_rate != float("inf"):
        # Finite rate → steady drip with no ramp-up / ramp-down artifacts,
        # so skip nothing (PERF-878).
        if skip_first is None:
            skip_first = 0
        if skip_last is None:
            skip_last = 0
    elif max_concurrency is not None and max_concurrency > 1:
        if skip_first is None:
            skip_first = max_concurrency
            logger.info(
                f"Auto-setting skip_first_n_requests={skip_first}"
                f" (max_concurrency={max_concurrency})"
            )
        if skip_last is None:
            skip_last = max_concurrency
            logger.info(
                f"Auto-setting skip_last_n_requests={skip_last}"
                f" (max_concurrency={max_concurrency})"
            )
    # max_concurrency=1 → sequential requests, no ramp-up to trim.
    # max_concurrency=None → no cap; default to 0.
    # Both leave auto values unset → fall through to 0 below.
    if skip_first is None:
        skip_first = 0
    if skip_last is None:
        skip_last = 0

    if args.warm_shared_prefix:
        fit_with_sys = args.fit_distributions and args.dataset_name in (
            "instruct-coder",
            "agentic-code",
        )
        if (
            args.dataset_name not in ("random", "synthetic")
            and not fit_with_sys
        ):
            raise ValueError(
                f"--warm-shared-prefix is not supported for dataset"
                f" '{args.dataset_name}'. Use random/synthetic, or"
                " instruct-coder/agentic-code with --fit-distributions."
            )
        if args.random_sys_prompt_ratio <= 0:
            raise ValueError(
                "--warm-shared-prefix requires --random-sys-prompt-ratio > 0."
            )

    logger.info("Starting benchmark run")
    assert args.num_prompts is not None

    if (
        args.ignore_first_turn_stats
        and skip_first
        and not args.warmup_to_steady_state
    ):
        # Without --warmup-to-steady-state, sessions all start at turn 0,
        # so --ignore-first-turn-stats already drops the same head requests
        # that --skip-first-n-requests would target. Combining them just
        # trims deeper into the run than the user asked for.
        # With --warmup-to-steady-state, sessions begin at randomized turn
        # offsets, so the two features filter different requests and we
        # want them to compose.
        logger.warning(
            "--ignore-first-turn-stats and --skip-first-n-requests both set"
            " without --warmup-to-steady-state. --ignore-first-turn-stats"
            " already drops every session's first turn, so"
            " --skip-first-n-requests would trim deeper than expected."
            " Ignoring --skip-first-n-requests."
        )
        skip_first = 0

    # Benchmark LoRA loading if manager provided
    if session.lora_manager:
        logger.info("Starting LoRA loading benchmark...")
        await session.lora_manager.benchmark_loading(
            api_url=session.base_url,
        )

    # Generate a single run-level unique prefix so all requests in this run
    # share the same constant prefix. This prevents cross-run KV-cache
    # pollution while preserving within-run system-prompt prefix caching
    # (requests with the same system prompt still share a common token prefix).
    run_prefix: str | None = None
    run_prefix_len: int = 0
    if args.force_unique_runs:
        if session.benchmark_task == "image-to-image":
            raise ValueError(
                "--force-unique-runs is not supported for image-to-image:"
                " the primary input is the image, not text, and systems may"
                " cache vision embeddings independently, so we can't guarantee"
                " uniqueness across benchmark runs."
            )
        run_prefix = f"{uuid4()}: "
        if session.benchmark_task not in PIXEL_GENERATION_TASKS:
            # prompt_len is not tracked for pixel generation tasks, so
            # run_prefix_len is not needed there.
            assert session.tokenizer is not None
            run_prefix_len = len(
                session.tokenizer.encode(run_prefix, add_special_tokens=False)
            )

    request_driver_class: type[RequestDriver] = get_request_driver_class(
        session.api_url, task=session.benchmark_task
    )
    # Create a request driver instance without pbar for test prompt
    # (pbar will be set later for the actual benchmark runs)
    test_request_driver: RequestDriver = request_driver_class(
        tokenizer=session.tokenizer
    )

    if args.warm_shared_prefix:
        await prime_shared_contexts(
            model_id=session.model_id,
            api_url=session.api_url,
            samples=session.samples,
            request_driver=test_request_driver,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            run_prefix=run_prefix,
            run_prefix_len=run_prefix_len,
        )

    if not args.skip_test_prompt:
        logger.info("Starting initial single prompt test run...")
        test_output = await run_single_test_prompt(
            benchmark_task=session.benchmark_task,
            model_id=session.model_id,
            api_url=session.api_url,
            samples=session.samples,
            request_driver=test_request_driver,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_output_len=args.max_output_len,
            run_prefix=run_prefix,
            run_prefix_len=run_prefix_len,
        )
        if not test_output.success:
            raise ValueError(
                "Initial test run failed - Please make sure benchmark"
                " arguments are correctly specified. Error:"
                f" {test_output.error}"
            )
        logger.info(
            "Initial test run completed. Starting main benchmark run..."
        )

    if args.burstiness == 1.0:
        distribution = "Poisson process"
    else:
        distribution = "Gamma distribution"

    logger.info(f"Input request rate: {request_rate}")
    logger.info(f"Burstiness factor: {args.burstiness} ({distribution})")
    logger.info(f"Maximum request concurrency: {max_concurrency}")

    semaphore: contextlib.AbstractAsyncContextManager[None]
    if max_concurrency:
        semaphore = asyncio.Semaphore(max_concurrency)
    else:
        semaphore = contextlib.nullcontext()

    with contextlib.ExitStack() as benchmark_stack:
        gpu_recorder: GPUBackgroundRecorder | None = None
        spec_decode_metrics_before: SpecDecodeMetrics | None = None
        spec_decode_metrics_after: SpecDecodeMetrics | None = None
        if args.collect_gpu_stats:
            try:
                from max.diagnostics.gpu import BackgroundRecorder
            except ImportError:
                logger.warning(
                    "max.diagnostics not available, skipping GPU stats"
                    " collection"
                )
            else:
                gpu_recorder = benchmark_stack.enter_context(
                    BackgroundRecorder()
                )

        cpu_collector = None
        if args.collect_cpu_stats:
            try:
                pids = collect_pids_for_port(
                    int(urlparse(session.api_url).port or 8000)
                )
                cpu_collector = benchmark_stack.enter_context(
                    CPUMetricsCollector(pids)
                )
            except Exception:
                logger.warning(
                    "Cannot access max-serve PIDs, skipping CPU stats"
                    " collection"
                )

        # Start nsys trace if enabled (before timing to exclude trace overhead)
        if session.trace_path is not None:
            benchmark_stack.enter_context(
                under_nsys_tracing(session.trace_path, args.trace_session)
            )

        # Create pbar for actual benchmark runs
        pbar = create_benchmark_pbar(
            disable_tqdm=args.disable_tqdm, samples=session.samples
        )

        # Create base driver and wrap with ProgressBarRequestDriver if pbar is provided
        request_driver = base_driver = request_driver_class(
            tokenizer=session.tokenizer
        )
        if pbar is not None:
            request_driver = ProgressBarRequestDriver(request_driver, pbar)

        # Prime prefix turns before the benchmark timer starts. Only the
        # initial concurrent population keeps its prefix_turns; sessions
        # arriving mid-benchmark get reset to 0 and don't need priming.
        # Bound: kv-cache-stress uses max_concurrent_conversations;
        # multiturn uses max_concurrency (may be None for unbounded, in
        # which case all sessions keep prefix_turns and are all primed).
        if isinstance(session.samples, ChatSamples):
            assert session.tokenizer is not None
            prime_bound = (
                args.max_concurrent_conversations
                if args.max_concurrent_conversations is not None
                else max_concurrency
            )
            await prime_prefix_turns(
                sessions=session.samples.chat_sessions,
                request_driver=base_driver,
                model_id=session.model_id,
                api_url=session.api_url,
                max_chat_len=session.tokenizer.model_max_length,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_sessions=prime_bound,
            )

        # Capture baseline server metrics after priming so priming requests
        # don't affect the delta calculation.
        baseline_endpoints: Mapping[str, ParsedMetrics] = {}
        if args.collect_server_stats:
            try:
                baseline_endpoints = collect_benchmark_metrics(
                    args.metrics_urls, backend, session.base_url
                )
                logger.info("Captured baseline server metrics")
            except Exception as e:
                logger.warning(
                    f"Failed to capture baseline server metrics: {e}"
                )

        if session.benchmark_task == "text-generation":
            spec_decode_metrics_before = fetch_spec_decode_metrics(
                backend, session.base_url
            )

        # Marker consumed by utils/benchmarking/serving/analyze_batch_logs.py
        # to slice the batch log by concurrency and exclude warmup/test-prompt
        # phases.
        logger.info(
            f"=== BATCH LOG MARKER: Benchmark started "
            f"(max_concurrency={max_concurrency}, "
            f"request_rate={request_rate}) ==="
        )
        benchmark_start_time = time.perf_counter_ns()
        if args.max_benchmark_duration_s is None:
            benchmark_should_end_time = None
        else:
            benchmark_should_end_time = benchmark_start_time + int(
                args.max_benchmark_duration_s * 1e9
            )

        all_outputs: Sequence[BaseRequestFuncOutput]
        outputs_by_session: dict[str, list[RequestFuncOutput]] | None = None
        if isinstance(session.samples, RequestSamples):
            if args.max_concurrent_conversations is not None:
                raise ValueError(
                    "--max-concurrent-conversations is only valid for "
                    "multi-turn workloads. Set --num-chat-sessions to "
                    "enable multi-turn mode."
                )
            # single-turn chat scenario
            all_outputs = await run_single_turn_benchmark(
                input_requests=session.samples.requests,
                benchmark_task=session.benchmark_task,
                request_rate=request_rate,
                burstiness=args.burstiness,
                timing_data=None,
                semaphore=semaphore,
                benchmark_should_end_time=benchmark_should_end_time,
                request_driver=request_driver,
                model_id=session.model_id,
                api_url=session.api_url,
                max_output_len=args.max_output_len,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                lora_manager=session.lora_manager,
                run_prefix=run_prefix,
                run_prefix_len=run_prefix_len,
            )
        elif args.max_concurrent_conversations is not None:
            # KV-cache stress benchmark: two independent concurrency knobs.
            # max_concurrent_conversations caps active session workers;
            # max_concurrency (semaphore) caps in-flight turns globally.
            if (
                max_concurrency is not None
                and max_concurrency > args.max_concurrent_conversations
            ):
                raise ValueError(
                    f"--max-concurrency ({max_concurrency}) must be <= "
                    f"--max-concurrent-conversations "
                    f"({args.max_concurrent_conversations}): to stress the "
                    "server's KV-cache, more sessions must be open than "
                    "turns in-flight."
                )
            assert session.tokenizer is not None
            assert isinstance(args.max_concurrent_conversations, int)
            outputs_by_session = await run_kv_cache_stress_benchmark(
                chat_sessions=session.samples.chat_sessions,
                max_requests=args.num_prompts,
                max_concurrent_conversations=args.max_concurrent_conversations,
                semaphore=semaphore,
                benchmark_should_end_time=benchmark_should_end_time,
                request_driver=request_driver,
                model_id=session.model_id,
                api_url=session.api_url,
                tokenizer=session.tokenizer,
                ignore_first_turn_stats=args.ignore_first_turn_stats,
                lora_manager=session.lora_manager,
                warmup_delay_ms=args.chat_warmup_delay_ms,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                randomize_session_start=args.randomize_session_start,
                warmup_to_steady_state=args.warmup_to_steady_state,
                warmup_oversample_factor=args.warmup_oversample_factor,
                num_chat_sessions=args.num_chat_sessions or 0,
                seed=args.seed,
                run_prefix=run_prefix,
                run_prefix_len=run_prefix_len,
            )
            all_outputs = [
                out for outs in outputs_by_session.values() for out in outs
            ]
        else:
            # multi-turn chat scenario
            outputs_by_session = await run_multiturn_benchmark(
                chat_sessions=session.samples.chat_sessions,
                max_requests=args.num_prompts,
                semaphore=semaphore,
                benchmark_should_end_time=benchmark_should_end_time,
                request_driver=request_driver,
                model_id=session.model_id,
                api_url=session.api_url,
                tokenizer=session.tokenizer,
                ignore_first_turn_stats=args.ignore_first_turn_stats,
                lora_manager=session.lora_manager,
                warmup_delay_ms=args.chat_warmup_delay_ms,
                max_concurrency=max_concurrency,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                randomize_session_start=args.randomize_session_start,
                warmup_to_steady_state=args.warmup_to_steady_state,
                warmup_oversample_factor=args.warmup_oversample_factor,
                num_chat_sessions=args.num_chat_sessions or 0,
                seed=args.seed,
                run_prefix=run_prefix,
                run_prefix_len=run_prefix_len,
            )
            all_outputs = [
                out for outs in outputs_by_session.values() for out in outs
            ]

        # Close pbar if it was created
        if pbar is not None:
            pbar.close()

        benchmark_duration = (
            time.perf_counter_ns() - benchmark_start_time
        ) / 1e9

    if session.benchmark_task == "text-generation":
        spec_decode_metrics_after = fetch_spec_decode_metrics(
            backend, session.base_url
        )
    spec_decode_stats = None
    if (
        spec_decode_metrics_before is not None
        and spec_decode_metrics_after is not None
    ):
        spec_decode_stats = calculate_spec_decode_stats(
            spec_decode_metrics_before,
            spec_decode_metrics_after,
        )

    if args.print_inputs_and_outputs:
        if session.benchmark_task == "text-generation":
            assert session.tokenizer is not None
            print("Generated output text:")
            for req_id, output in enumerate(all_outputs):
                assert isinstance(output, RequestFuncOutput)
                output_len = compute_output_len(session.tokenizer, output)
                print(
                    {
                        "req_id": req_id,
                        "output_len": output_len,
                        "output": output.generated_text,
                    }
                )
        elif session.benchmark_task in PIXEL_GENERATION_TASKS:
            print("Generated pixel generation outputs:")
            for req_id, output in enumerate(all_outputs):
                assert isinstance(output, PixelGenerationRequestFuncOutput)
                print(
                    {
                        "req_id": req_id,
                        "num_generated_outputs": output.num_generated_outputs,
                        "latency_s": output.latency,
                        "success": output.success,
                        "error": output.error,
                    }
                )

    if session.lora_manager:
        await session.lora_manager.benchmark_unloading(
            api_url=session.base_url,
        )

    gpu_metrics: list[dict[str, GPUStats]] | None = None
    if args.collect_gpu_stats and gpu_recorder is not None:
        gpu_metrics = gpu_recorder.stats

    cpu_metrics_result: CPUMetrics | None = None
    if cpu_collector is not None:
        cpu_metrics_result = cpu_collector.get_stats()

    # Collect server-side metrics from Prometheus endpoint (with delta from baseline)
    endpoint_metrics: Mapping[str, ParsedMetrics] = {}
    if args.collect_server_stats:
        try:
            endpoint_metrics = collect_benchmark_metrics(
                args.metrics_urls,
                backend,
                session.base_url,
                baseline=baseline_endpoints,
            )
            logger.info("Collected server metrics (final)")
        except Exception as e:
            logger.warning(f"Failed to collect server metrics: {e}")

    achieved_request_rate = 0.0

    result: PixelGenerationBenchmarkResult | TextGenerationBenchmarkResult
    if session.benchmark_task in PIXEL_GENERATION_TASKS:
        result = build_pixel_generation_result(
            outputs=all_outputs,
            benchmark_duration=benchmark_duration,
            gpu_metrics=gpu_metrics,
            cpu_metrics=cpu_metrics_result,
            max_concurrency=max_concurrency,
            collect_gpu_stats=args.collect_gpu_stats,
            metrics_by_endpoint=endpoint_metrics,
        )
    else:
        text_result = build_text_generation_result(
            outputs=all_outputs,
            benchmark_duration=benchmark_duration,
            tokenizer=session.tokenizer,
            gpu_metrics=gpu_metrics,
            cpu_metrics=cpu_metrics_result,
            skip_first_n_requests=skip_first,
            skip_last_n_requests=skip_last,
            max_concurrency=max_concurrency,
            max_concurrent_conversations=args.max_concurrent_conversations,
            collect_gpu_stats=args.collect_gpu_stats,
            metrics_by_endpoint=endpoint_metrics,
            spec_decode_stats=spec_decode_stats,
        )
        if outputs_by_session is not None:
            text_result.session_server_stats = {
                sid: [out.server_token_stats for out in outs]
                for sid, outs in sorted(
                    outputs_by_session.items(),
                    key=lambda kv: _session_sort_key(kv[0]),
                )
            }
        else:
            text_result.aggregate_server_stats = [
                out.server_token_stats
                for out in all_outputs
                if isinstance(out, RequestFuncOutput)
            ]
        result = text_result
    if session.lora_manager is not None:
        result.lora_metrics = session.lora_manager.metrics

    print_benchmark_summary(
        metrics=result.metrics,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        achieved_request_rate=achieved_request_rate,
        collect_gpu_stats=args.collect_gpu_stats,
        collect_cpu_stats=args.collect_cpu_stats,
        spec_decode_stats=spec_decode_stats,
        lora_manager=session.lora_manager,
    )

    ok, validation_errors = result.validate_metrics()
    if not ok:
        for err in validation_errors:
            logger.error(f"Benchmark result validation failed: {err}")
        logger.info("finished benchmark run: Failed.")
        sys.exit(1)

    logger.info("finished benchmark run: Success.")
    return result


def validate_task_and_endpoint(
    benchmark_task: BenchmarkTask, endpoint: Endpoint
) -> None:
    if benchmark_task == "text-generation":
        if endpoint in ("/v1/responses", "/v1/images/generations"):
            raise ValueError(
                f"--benchmark-task text-generation does not support "
                f"--endpoint {endpoint}"
            )
    elif benchmark_task in PIXEL_GENERATION_TASKS:
        if endpoint not in PIXEL_GENERATION_ENDPOINTS:
            raise ValueError(
                f"--benchmark-task {benchmark_task} requires --endpoint"
                f" to be one of {sorted(PIXEL_GENERATION_ENDPOINTS)},"
                f" got {endpoint!r}"
            )


def _apply_workload_to_config(
    config: ServingBenchmarkConfig, workload: Mapping[str, object]
) -> None:
    """Set workload YAML values as fields on *config*.

    Keys are converted from kebab-case to snake_case.  Path objects are
    stringified and env vars in string values are expanded.

    Fields already in `config.model_fields_set` (i.e. explicitly provided
    by the caller, whether via CLI args or direct construction) are left
    unchanged so that CLI values always take precedence over workload YAML.
    """
    for k, v in workload.items():
        field_name = k.replace("-", "_")
        if field_name not in ServingBenchmarkConfig.model_fields:
            logger.warning(f"Ignoring unknown workload key: {k}")
            continue
        if field_name in config.model_fields_set:
            logger.info(
                f"CLI flag --{k} takes precedence over workload YAML"
                f" (CLI: {getattr(config, field_name)!r},"
                f" workload: {v!r})"
            )
            continue
        if isinstance(v, Path):
            v = str(v)
        elif isinstance(v, str):
            v = os.path.expandvars(v)
        logger.info(f"Applying workload YAML value: --{k}={v!r}")
        setattr(config, field_name, v)


def flush_prefix_cache(
    backend: Backend, host: str, port: int, dry_run: bool
) -> None:
    """Flush the serving engine's prefix cache via HTTP POST."""
    if backend not in CACHE_RESET_ENDPOINT_MAP:
        raise ValueError(
            f"Cannot flush prefix cache for {backend} backend: this backend"
            " does not support prefix cache flush."
        )
    import requests as _http_requests  # lazy - avoid hard dep for non-sweep use

    api_url = f"http://{host}:{port}{CACHE_RESET_ENDPOINT_MAP[backend]}"
    if dry_run:
        logger.info(f"Dry-run flush: POST {api_url}")
        return
    response = _http_requests.post(api_url)
    if response.status_code == 400:
        logger.warning(
            f"Prefix caching is not enabled on backend {backend} at {api_url};"
            " skipping cache flush."
        )
    elif response.status_code == 404:
        logger.warning(
            f"Prefix cache reset is not supported at {api_url} (HTTP 404);"
            " skipping cache flush."
        )
    elif response.status_code != 200:
        # Mammoth's proxy wraps engine 404s in a 502 with per-endpoint statuses
        # in the JSON body; treat unanimous 404s the same as a direct 404 above
        # (e.g. vLLM builds without /reset_prefix_cache exposed).
        try:
            body = response.json() if response.content else None
        except ValueError:
            body = None
        results = body.get("results") if isinstance(body, dict) else None
        if (
            isinstance(results, list)
            and results
            and all(
                isinstance(r, dict) and r.get("statusCode") == 404
                for r in results
            )
        ):
            logger.warning(
                f"Prefix cache reset is not supported at {api_url} "
                "(proxy reported 404 from all engine endpoints);"
                " skipping cache flush."
            )
            return
        raise RuntimeError(
            f"Failed to flush prefix cache for backend {backend} at {api_url}: "
            f"status={response.status_code} body={response.text}"
        )


@dataclass
class BenchmarkRunResult:
    """Result of one (max_concurrency, request_rate) benchmark configuration.

    Yielded by :func:`main_with_parsed_args` — one entry per (mc, rr) combo
    after median selection across ``num_iters`` iterations.
    """

    max_concurrency: int | None
    request_rate: float
    num_prompts: int
    result: (
        TextGenerationBenchmarkResult | PixelGenerationBenchmarkResult | None
    ) = None


@dataclass
class BenchmarkSession:
    """Resolved, session-level state shared across all sweep iterations.

    Created once after argument parsing / dataset loading in
    :func:`main_with_parsed_args` and threaded into each
    :func:`benchmark` call.
    """

    benchmark_task: BenchmarkTask
    endpoint: Endpoint
    api_url: str
    base_url: str
    model_id: str
    tokenizer_id: str
    tokenizer: PreTrainedTokenizerBase | None
    samples: Samples
    lora_manager: LoRABenchmarkManager | None
    trace_path: str | None
    orig_skip_first: int | None
    orig_skip_last: int | None


def _session_sort_key(sid: str) -> tuple[int, int, str]:
    """Sort numeric session ids first by integer value, then anonymous ids."""
    try:
        return (0, int(sid), "")
    except ValueError:
        return (1, 0, sid)


def _load_workload_yaml(args: ServingBenchmarkConfig) -> None:
    if not args.workload_config:
        return
    with open(args.workload_config) as workload_file:
        workload = yaml.safe_load(workload_file)
    # Resolve relative paths against the YAML's directory.
    for key in ("dataset-path", "output-lengths"):
        if workload.get(key) is not None:
            if is_castable_to_int(str(workload[key])):
                continue
            path = Path(os.path.expandvars(workload[key]))
            if not path.is_absolute():
                path = Path(args.workload_config).parent / path
            workload[key] = path
    # Resolve max_concurrency: CLI > YAML.
    yaml_max_concurrency = workload.pop("max-concurrency", None)
    if yaml_max_concurrency is not None and args.max_concurrency is None:
        args.max_concurrency = str(yaml_max_concurrency)
    # Resolve num_prompts: CLI > YAML > default (deferred).
    cli_num_prompts = args.num_prompts is not None
    yaml_num_prompts = workload.pop("num-prompts", None)
    if not cli_num_prompts:
        if yaml_num_prompts is not None:
            args.num_prompts = int(yaml_num_prompts)
    # Resolve max_benchmark_duration_s: CLI > YAML.
    w_duration = workload.pop("max-benchmark-duration-s", None)
    if w_duration is not None and args.max_benchmark_duration_s is None:
        args.max_benchmark_duration_s = int(w_duration)
    _apply_workload_to_config(args, workload)
    args.skip_test_prompt = True


def _apply_run_length_defaults(args: ServingBenchmarkConfig) -> None:
    has_prompts = args.num_prompts is not None
    has_duration = args.max_benchmark_duration_s is not None
    has_multiplier = args.num_prompts_multiplier is not None
    # The multiplier dynamically computes num_prompts per-mc, but only
    # when no explicit duration also constrains the run.
    multiplier_will_resolve = has_multiplier and not has_duration
    if not has_prompts and not has_duration and not has_multiplier:
        logger.warning(
            "Neither --num-prompts nor --max-benchmark-duration-s is"
            " specified. Defaulting to --num-prompts 1000 and"
            " --max-benchmark-duration-s 300"
        )
        args.num_prompts = 1000
        args.max_benchmark_duration_s = 300
    elif not has_prompts and not multiplier_will_resolve:
        args.num_prompts = 1000


def _apply_dynamic_num_prompts(
    args: ServingBenchmarkConfig,
    concurrency_range: list[int | None],
) -> bool:
    use_dynamic_num_prompts = (
        args.num_prompts_multiplier is not None
        and args.num_prompts is None
        and args.max_benchmark_duration_s is None
    )
    if use_dynamic_num_prompts:
        assert args.num_prompts_multiplier is not None
        max_mc = max(
            (mc for mc in concurrency_range if mc is not None), default=1
        )
        args.num_prompts = args.num_prompts_multiplier * max_mc
        # When using num_prompts_multiplier without explicit duration, default to
        # 300s timeout per MC config to prevent indefinitely long benchmark runs.
        logger.info(
            "Using --num-prompts-multiplier without --max-benchmark-duration-s."
            " Defaulting to 300s timeout per max-concurrency configuration."
        )
        args.max_benchmark_duration_s = 300
    return use_dynamic_num_prompts


def _build_session(args: ServingBenchmarkConfig) -> BenchmarkSession:
    assert args.model is not None
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_ulimit()
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    benchmark_task: BenchmarkTask = args.benchmark_task
    endpoint: Endpoint = args.endpoint

    # Auto-select the correct endpoint for pixel generation based on the
    # backend. Each pixel-gen backend requires a specific endpoint (e.g.,
    # sglang needs /v1/images/generations, vllm needs
    # /v1/chat/completions). We auto-select when the current endpoint
    # doesn't match this backend's expected pixel-gen endpoint.
    if benchmark_task in PIXEL_GENERATION_TASKS:
        backend_key = args.backend.removesuffix("-chat")
        if backend_key in PIXEL_GEN_DEFAULT_ENDPOINT:
            expected = PIXEL_GEN_DEFAULT_ENDPOINT[backend_key]
            if endpoint != expected:
                logger.info(
                    "Auto-selected endpoint %s for backend %s"
                    " (pixel generation task)",
                    expected,
                    args.backend,
                )
                endpoint = expected
        else:
            raise ValueError(
                f"Backend {args.backend!r} does not have a default"
                f" pixel-generation endpoint. Explicitly pass --endpoint"
                f" with one of {sorted(PIXEL_GENERATION_ENDPOINTS)}."
            )

    validate_task_and_endpoint(benchmark_task, endpoint)
    # chat is only meaningful for text-generation (enables chat template
    # formatting). For pixel generation via /v1/chat/completions
    # (vllm pixel gen), the pixel-gen code path ignores this flag.
    chat = endpoint == "/v1/chat/completions"

    if args.base_url is not None:
        base_url = args.base_url
    else:
        base_url = f"http://{args.host}:{args.port}"

    api_url = f"{base_url}{endpoint}"
    tokenizer: PreTrainedTokenizerBase | None = None

    if benchmark_task == "text-generation":
        logger.info(f"getting tokenizer. api url: {api_url}")
        tokenizer = get_tokenizer(
            tokenizer_id,
            model_max_length=args.model_max_length,
            trust_remote_code=args.trust_remote_code,
        )

    samples = sample_requests(
        args=args,
        benchmark_task=benchmark_task,
        tokenizer=tokenizer,
        chat=chat,
    )

    # Inject response_format into all sampled requests if specified
    if args.response_format is not None:
        response_format = parse_response_format(args.response_format)
        if isinstance(samples, RequestSamples):
            for request in samples.requests:
                request.response_format = response_format
            logger.info(
                f"Injected response_format into {len(samples.requests)} requests"
            )
        else:
            logger.warning(
                "response_format is only supported for single-turn benchmarks, "
                "ignoring for multi-turn chat sessions"
            )

    lora_manager = None
    if args.lora_paths:
        num_requests = (
            len(samples.requests)
            if isinstance(samples, RequestSamples)
            else len(samples.chat_sessions)
        )
        lora_manager = LoRABenchmarkManager(
            lora_paths=args.lora_paths,
            num_requests=num_requests,
            traffic_ratios=args.per_lora_traffic_ratio
            if args.per_lora_traffic_ratio
            else None,
            uniform_ratio=args.lora_uniform_traffic_ratio,
            seed=args.seed,
            max_concurrent_lora_ops=args.max_concurrent_lora_ops,
        )
        lora_manager.log_traffic_distribution()

    # Handle trace flag
    trace_path = None
    if args.trace:
        assert_nvidia_gpu()
        trace_path = (
            args.trace_file if args.trace_file else get_default_trace_path()
        )
        logger.info(f"Tracing enabled, output: {trace_path}")

    return BenchmarkSession(
        benchmark_task=benchmark_task,
        endpoint=endpoint,
        api_url=api_url,
        base_url=base_url,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        tokenizer=tokenizer,
        samples=samples,
        lora_manager=lora_manager,
        trace_path=trace_path,
        orig_skip_first=args.skip_first_n_requests,
        orig_skip_last=args.skip_last_n_requests,
    )


def _run_dry_run_sweep(
    args: ServingBenchmarkConfig,
    session: BenchmarkSession,
    concurrency_range: list[int | None],
    request_rate_range: list[float],
) -> Iterator[BenchmarkRunResult]:
    if not args.print_workload_stats:
        print_workload_stats(session.samples)
    if isinstance(session.samples, ChatSamples) and args.warmup_to_steady_state:
        rng = np.random.default_rng(args.seed or 0)
        for mc in concurrency_range:
            warmup_count = (
                args.max_concurrent_conversations
                or mc
                or len(session.samples.chat_sessions)
            )
            print_section(
                title=f" Warmup sampling preview (max_concurrency={mc}) ",
                char="=",
            )
            _, report = pick_warmup_population(
                session.samples.chat_sessions,
                warmup_count,
                warmup_to_steady_state=True,
                warmup_oversample_factor=args.warmup_oversample_factor,
                main_pool_target=args.num_chat_sessions or 0,
                rng=rng,
            )
            if report is not None:
                log_warmup_sampling_report(report)
    for mc in concurrency_range:
        for rr in request_rate_range:
            print(
                f"Dry run: model={args.model}"
                f" host={args.host} port={args.port}"
                f" endpoint={args.endpoint}"
                f" max_concurrency={mc}"
                f" request_rate={rr}"
                f" num_prompts={args.num_prompts}"
                f" max_benchmark_duration_s="
                f"{args.max_benchmark_duration_s}"
            )
            yield BenchmarkRunResult(
                max_concurrency=mc,
                request_rate=rr,
                num_prompts=args.num_prompts or 0,
            )


def _run_benchmark_sweep(
    args: ServingBenchmarkConfig,
    session: BenchmarkSession,
    concurrency_range: list[int | None],
    request_rate_range: list[float],
    use_dynamic_num_prompts: bool,
) -> Iterator[BenchmarkRunResult]:
    # ---- Sweep loop ----
    for mc in concurrency_range:
        if use_dynamic_num_prompts:
            assert args.num_prompts_multiplier is not None
            assert mc is not None
            args.num_prompts = args.num_prompts_multiplier * mc
            logger.info(
                f"Using num_prompts = {args.num_prompts_multiplier}"
                f" * {mc} = {args.num_prompts}"
            )

        for rr in request_rate_range:
            # Temporarily write the per-iteration values so that downstream
            # code reading args.max_concurrency / args.request_rate sees the
            # correct scalar value.
            args.max_concurrency = str(mc) if mc is not None else None
            args.request_rate = str(rr)

            iteration_results: list[
                TextGenerationBenchmarkResult | PixelGenerationBenchmarkResult
            ] = []
            for _iteration in range(args.num_iters):
                if args.flush_prefix_cache:
                    flush_prefix_cache(
                        args.backend, args.host, args.port, args.dry_run
                    )

                args.seed = int(np.random.randint(0, 10000))

                result = asyncio.run(benchmark(args, session, mc, rr))
                iteration_results.append(result)

            # Median selection when running multiple iterations.
            if len(iteration_results) > 1:
                throughputs = np.asarray(
                    [r.metrics.request_throughput for r in iteration_results]
                )
                idx = argmedian(throughputs)
            else:
                idx = 0
            best_result = iteration_results[idx]

            # JSON result file (for the median iteration).
            save_result_json(
                args.result_filename,
                args,
                best_result,
                benchmark_task=session.benchmark_task,
                model_id=session.model_id,
                tokenizer_id=session.tokenizer_id,
                request_rate=rr,
            )

            # Output lengths recording (for the median iteration).
            save_output_lengths(args, best_result, session.benchmark_task)

            yield BenchmarkRunResult(
                mc, rr, args.num_prompts or 0, result=best_result
            )


def main_with_parsed_args(
    args: ServingBenchmarkConfig,
) -> Iterator[BenchmarkRunResult]:
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)s: %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    logger.info(args)

    if args.model is None:
        raise ValueError("--model is required when running benchmark")

    _load_workload_yaml(args)
    _apply_run_length_defaults(args)

    concurrency_range = parse_comma_separated(args.max_concurrency, int_or_none)
    request_rate_range = parse_comma_separated(args.request_rate, float)
    use_dynamic_num_prompts = _apply_dynamic_num_prompts(
        args, concurrency_range
    )

    session = _build_session(args)

    if args.print_workload_stats:
        print_workload_stats(session.samples)
    if args.print_inputs_and_outputs:
        print_input_prompts(session.samples)

    if args.dry_run:
        yield from _run_dry_run_sweep(
            args, session, concurrency_range, request_rate_range
        )
        return

    wait_for_server_ready(
        args.host, args.port, timeout_s=args.server_ready_timeout_s
    )

    yield from _run_benchmark_sweep(
        args,
        session,
        concurrency_range,
        request_rate_range,
        use_dynamic_num_prompts,
    )


def _extract_metadata_args(
    args: list[str],
) -> tuple[list[str], list[str]]:
    """Extract --metadata values from args before passing to cyclopts.

    cyclopts interprets bare ``key=value`` tokens as keyword assignments. When
    a token like ``enable_prefix_caching=True`` matches a real model field, it
    is routed to that field rather than consumed as a ``--metadata`` list item,
    leaving subsequent tokens as orphaned positionals (which then fail).

    This function peels off all space-separated values after ``--metadata``
    (until the next ``--flag``) and returns them separately so cyclopts never
    sees them.

    Returns:
        A 2-tuple of (clean_args, metadata_values).
    """
    clean_args: list[str] = []
    metadata_values: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--metadata":
            i += 1
            while i < len(args) and not args[i].startswith("-"):
                metadata_values.append(args[i])
                i += 1
        else:
            clean_args.append(args[i])
            i += 1
    return clean_args, metadata_values


def parse_args(
    args: Sequence[str] | None = None,
    *,
    app_name: str = "benchmark_serving",
    description: str = BENCHMARK_SERVING_ARGPARSER_DESCRIPTION,
) -> ServingBenchmarkConfig:
    """Parse command line arguments into a ServingBenchmarkConfig.

    Args:
        args: Command line arguments to parse. If None, parse from sys.argv.
        app_name: Name shown in --help output.
        description: Description shown in --help output.
    """
    raw_args = list(sys.argv[1:] if args is None else args)

    clean_args, metadata_values = _extract_metadata_args(raw_args)

    parsed_configs: list[ServingBenchmarkConfig] = []

    app = App(
        name=app_name,
        help=description,
        help_formatter="plain",
        config=[Env(prefix="MODULAR_")],
        result_action="return_value",
    )

    @app.default
    def _capture(
        config: Annotated[
            ServingBenchmarkConfig, Parameter(name="*")
        ] = ServingBenchmarkConfig(),
    ) -> None:
        parsed_configs.append(config)

    app(clean_args)
    if not parsed_configs:
        raise SystemExit(0)
    config = parsed_configs[0]
    if metadata_values:
        config.metadata = metadata_values
    return config
