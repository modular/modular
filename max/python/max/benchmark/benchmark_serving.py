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
import logging
import os
import random
import shlex
import subprocess
import sys
import time
from collections.abc import (
    Callable,
    Generator,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np
import psutil
from cyclopts import App, Parameter
from cyclopts.config import Env
from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from max.benchmark.benchmark_shared.server_metrics import ParsedMetrics

from max.benchmark.benchmark_shared.config import (
    CACHE_RESET_ENDPOINT_MAP,
    PIXEL_GENERATION_ENDPOINTS,
    PIXEL_GENERATION_TASKS,
    VIDEO_GENERATION_TASKS,
    Backend,
    BenchmarkTask,
    Endpoint,
    ServingBenchmarkConfig,
    get_pixel_gen_endpoint,
)
from max.benchmark.benchmark_shared.datasets.all import sample_requests
from max.benchmark.benchmark_shared.datasets.chat_judge import (
    ChatJudgeChatSamples,
)
from max.benchmark.benchmark_shared.datasets.image_augmentation import (
    augment_samples_with_images,
)
from max.benchmark.benchmark_shared.datasets.response_format_augmentation import (
    augment_samples_with_response_format,
)
from max.benchmark.benchmark_shared.datasets.tool_augmentation import (
    augment_samples_with_tools,
)
from max.benchmark.benchmark_shared.datasets.types import (
    ChatSamples,
    ChatSession,
    RequestSamples,
    Samples,
)
from max.benchmark.benchmark_shared.gpu_metrics_prometheus import (
    PrometheusGPURecorder,
)
from max.benchmark.benchmark_shared.gpu_metrics_scraper import (
    DCGMBackgroundRecorder,
    GPUStatsSnapshot,
)
from max.benchmark.benchmark_shared.iteration import (
    report_and_validate,
    resolve_seed,
    resolve_skip_counts,
    retain_result,
    seed_for_concurrency,
)
from max.benchmark.benchmark_shared.lora_benchmark_manager import (
    LoRABenchmarkManager,
)
from max.benchmark.benchmark_shared.metrics import (
    BenchmarkResult,
    calculate_spec_decode_stats,
)
from max.benchmark.benchmark_shared.multi_turn import (
    prerun_warmup_turns,
    run_chat_judge_benchmark,
    run_kv_cache_stress_benchmark,
    run_multiturn_benchmark,
)
from max.benchmark.benchmark_shared.request import (
    BaseRequestFuncOutput,
    PixelGenerationRequestFuncOutput,
    RequestDriver,
    RequestFuncOutput,
    get_request_driver_class,
    progressbar_request_driver,
)
from max.benchmark.benchmark_shared.server_metrics import (
    MissingEngineMetricsError,
    collect_benchmark_metrics,
    fetch_spec_decode_metrics,
)
from max.benchmark.benchmark_shared.serving_metrics import (
    build_pixel_generation_result,
    build_text_generation_result,
    compute_output_len,
)
from max.benchmark.benchmark_shared.serving_result_output import (
    BenchmarkRunResultLabels,
    print_input_prompts,
    print_workload_stats,
)
from max.benchmark.benchmark_shared.single_turn import (
    prime_shared_contexts,
    run_single_test_prompt,
    run_single_turn_benchmark,
)
from max.benchmark.benchmark_shared.utils import (
    argmedian,
    fetch_server_max_model_len,
    get_tokenizer,
    openai_bearer_auth_headers,
    print_section,
    resolve_revision,
    set_ulimit,
    wait_for_server_ready,
)
from max.benchmark.benchmark_shared.warmup import (
    log_warmup_sampling_report,
    pick_warmup_population,
)
from max.benchmark.benchmark_shared.workload import (
    apply_workload,
    load_workload_file,
)
from max.benchmark.benchmark_shared.workload_stats import (
    WorkloadStats,
    compute_workload_stats,
    write_workload_stats,
)
from max.profiler.cpu import (
    CPUMetrics,
    CPUMetricsCollector,
    collect_pids_for_port,
)
from max.profiler.gpu import GPUDiagContext, GpuStatsRecorder

BENCHMARK_SERVING_ARGPARSER_DESCRIPTION = (
    "This command runs comprehensive benchmark tests on a model server to"
    " measure performance metrics including throughput, latency, and resource"
    " utilization. Make sure that the MAX server is running and hosting a model"
    " before running this command."
)

logger = logging.getLogger(__name__)


def _expand_pids(pids: list[int]) -> list[int]:
    """Returns pids plus all their descendants."""
    ppid_to_children: dict[int, list[int]] = {}
    for proc in psutil.process_iter(["pid", "ppid"]):
        ppid_to_children.setdefault(proc.info["ppid"], []).append(
            proc.info["pid"]
        )

    result: set[int] = set()
    queue = list(pids)
    while queue:
        pid = queue.pop()
        if pid in result:
            continue
        result.add(pid)
        queue.extend(ppid_to_children.get(pid, []))
    return list(result)


def _make_gpu_recorder(
    *,
    collect_gpu_stats: bool,
    gpu_metrics_host: str,
    gpu_metrics_prometheus_url: str = "",
    gpu_metrics_selector: str = "",
) -> GpuStatsRecorder | None:
    """Choose the GPU-stats recorder for this run, or None to disable it.

    The remote sources are source selectors, not additive toggles: either one
    means the load generator runs off the accelerator node (a disaggregated
    Mammoth bench pod) with no local engine GPUs, so stats must come from the
    engine nodes rather than local NVML. Prometheus wins over a direct DCGM
    scrape because its pod labels narrow the device set to the engine's own
    GPUs. Running more than one would double-count and destabilize the device
    set :func:`_aggregate_gpu_stats` aggregates over. Returned un-entered; the
    caller is responsible for entering it as a context manager.

    Args:
        collect_gpu_stats: Whether GPU-stats collection is enabled at all.
        gpu_metrics_host: Remote DCGM endpoint(s); empty selects local NVML.
        gpu_metrics_prometheus_url: Prometheus that scrapes the engine's DCGM
            exporter; takes precedence over ``gpu_metrics_host``.
        gpu_metrics_selector: Label matchers for the engine's GPUs in that
            Prometheus.

    Returns:
        A recorder to sample GPU stats, or None when disabled/unavailable.

    Raises:
        ValueError: If a Prometheus URL is given without a selector.
    """
    if not collect_gpu_stats:
        return None
    if gpu_metrics_prometheus_url:
        return PrometheusGPURecorder(
            gpu_metrics_prometheus_url, gpu_metrics_selector
        )
    if gpu_metrics_host:
        return DCGMBackgroundRecorder(gpu_metrics_host)
    try:
        from max.profiler.gpu import BackgroundRecorder
    except ImportError:
        logger.warning(
            "max.profiler not available, skipping GPU stats collection"
        )
        return None
    return BackgroundRecorder()


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


def _benchmark_request_total(samples: Samples) -> int:
    """Return the number of requests a benchmark run will issue for *samples*."""
    if isinstance(samples, RequestSamples):
        # single-turn chat scenario
        return len(samples.requests)
    # multi-turn chat scenario
    return sum(session.num_turns for session in samples.chat_sessions)


async def benchmark(
    args: ServingBenchmarkConfig,
    session: BenchmarkSession,
    max_concurrency: int | None,
    request_rate: float,
) -> tuple[BenchmarkResult, bool]:
    """Run a single benchmark invocation.

    ``session.orig_skip_first`` / ``session.orig_skip_last`` are the
    user-supplied values (``None`` = auto-derive from *max_concurrency*).
    """
    backend: Backend = args.backend

    skip_first, skip_last = resolve_skip_counts(
        orig_skip_first=session.orig_skip_first,
        orig_skip_last=session.orig_skip_last,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        ignore_first_turn_stats=args.ignore_first_turn_stats,
        warmup_to_steady_state=args.warmup_to_steady_state,
    )

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

    if isinstance(session.samples, ChatJudgeChatSamples):
        if args.warmup_to_steady_state:
            raise NotImplementedError(
                "--warmup-to-steady-state is not yet implemented for"
                " chat-judge."
            )

    logger.info("Starting benchmark run")
    assert args.num_prompts is not None

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
    if args.extra_body and session.benchmark_task != "text-generation":
        logger.warning(
            "extra_body is ignored for %s; it only applies to "
            "text-generation requests.",
            session.benchmark_task,
        )
    # Create a request driver instance without pbar for test prompt
    # (pbar will be set later for the actual benchmark runs)
    test_request_driver: RequestDriver = request_driver_class(
        tokenizer=session.tokenizer,
        extra_body=args.extra_body,
        backend=args.backend,
    )

    if args.warm_shared_prefix:
        await prime_shared_contexts(
            model_id=session.model_id,
            api_url=session.api_url,
            samples=session.samples,
            request_driver=test_request_driver,
            sampling=args.sampling,
            run_prefix=run_prefix,
            run_prefix_len=run_prefix_len,
            max_concurrency=args.warmup_concurrency,
            disable_tqdm=args.disable_tqdm,
        )

    if not args.skip_test_prompt:
        logger.info("Starting initial single prompt test run...")
        test_output = await run_single_test_prompt(
            benchmark_task=session.benchmark_task,
            model_id=session.model_id,
            api_url=session.api_url,
            samples=session.samples,
            request_driver=test_request_driver,
            sampling=args.sampling,
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

    base_driver = request_driver_class(
        tokenizer=session.tokenizer,
        extra_body=args.extra_body,
        backend=args.backend,
    )

    # Warm up the initial-slot sessions before starting the timer.
    # pick_warmup_population assigns each picked session a random
    # prefix_turns; prerun_warmup_turns fires one request per session
    # covering that prefix. Sessions arriving mid-benchmark start cold.
    chat_sessions: Sequence[ChatSession] | None = None
    if isinstance(session.samples, ChatSamples):
        assert session.tokenizer is not None
        chat_sessions = session.samples.chat_sessions
        warmup_count = (
            args.max_concurrent_conversations
            if args.max_concurrent_conversations is not None
            else max_concurrency
        )
        if args.warmup_to_steady_state and warmup_count:
            chat_sessions, report = pick_warmup_population(
                chat_sessions,
                warmup_count,
                warmup_to_steady_state=True,
                warmup_oversample_factor=args.warmup_oversample_factor,
                main_pool_target=args.num_chat_sessions or len(chat_sessions),
                rng=np.random.default_rng(args.seed),
                delay_biased=args.warmup_delay_biased,
                est_ttft_ms=args.warmup_delay_estimated_ttft_ms,
                est_tpot_ms=args.warmup_delay_estimated_tpot_ms,
            )
            if report is not None:
                log_warmup_sampling_report(report)
            cold_count = sum(
                1 for s in chat_sessions[:warmup_count] if s.prefix_turns == 0
            )
            logger.info(
                "Warming up to steady state: %d sessions (%d will"
                " pre-run, %d drew prefix_turns=0 and start cold).",
                warmup_count,
                warmup_count - cold_count,
                cold_count,
            )
        await prerun_warmup_turns(
            sessions=chat_sessions,
            request_driver=base_driver,
            model_id=session.model_id,
            api_url=session.api_url,
            max_chat_len=session.tokenizer.model_max_length,
            sampling=args.sampling,
            disable_tqdm=args.disable_tqdm,
            max_concurrency=args.warmup_concurrency,
            use_session_id_as_cache_salt=args.use_session_id_as_cache_salt,
            max_output_len=args.max_output_len,
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
            # No required-metric verification on the baseline: the exporter
            # omits a histogram until its first data point, so a fresh idle
            # engine has no batch families yet.
            logger.warning(f"Failed to capture baseline server metrics: {e}")

    if session.benchmark_task == "text-generation":
        spec_decode_metrics_before = fetch_spec_decode_metrics(
            backend, session.base_url, metrics_urls=args.metrics_urls
        )
    else:
        spec_decode_metrics_before = None

    semaphore: contextlib.AbstractAsyncContextManager[None]
    if max_concurrency:
        semaphore = asyncio.Semaphore(max_concurrency)
    else:
        semaphore = contextlib.nullcontext()

    with contextlib.ExitStack() as benchmark_stack:
        gpu_recorder = _make_gpu_recorder(
            collect_gpu_stats=args.collect_gpu_stats,
            gpu_metrics_host=args.gpu_metrics_host,
            gpu_metrics_prometheus_url=args.gpu_metrics_prometheus_url,
            gpu_metrics_selector=args.gpu_metrics_selector,
        )
        if gpu_recorder is not None:
            gpu_recorder = benchmark_stack.enter_context(gpu_recorder)

        cpu_collector = None
        if args.collect_cpu_stats:
            try:
                if args.server_pids:
                    pids = _expand_pids(args.server_pids)
                else:
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
        request_driver = benchmark_stack.enter_context(
            progressbar_request_driver(
                base_driver,
                _benchmark_request_total(session.samples),
                disable_tqdm=args.disable_tqdm,
            )
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
                sampling=args.sampling,
                lora_manager=session.lora_manager,
                run_prefix=run_prefix,
                run_prefix_len=run_prefix_len,
                disable_ignore_eos=args.disable_ignore_eos,
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
            assert chat_sessions is not None
            outputs_by_session = await run_kv_cache_stress_benchmark(
                chat_sessions=chat_sessions,
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
                sampling=args.sampling,
                run_prefix=run_prefix,
                run_prefix_len=run_prefix_len,
                request_rate=request_rate,
                burstiness=args.burstiness,
                est_ttft_ms=args.warmup_delay_estimated_ttft_ms,
                est_tpot_ms=args.warmup_delay_estimated_tpot_ms,
                use_session_id_as_cache_salt=args.use_session_id_as_cache_salt,
                disable_ignore_eos=args.disable_ignore_eos,
            )
            all_outputs = [
                out for outs in outputs_by_session.values() for out in outs
            ]
        elif isinstance(session.samples, ChatJudgeChatSamples):
            # chat-judge: each turn already has its history inlined in the
            # user message, so the driver sends [system, user] per turn
            # and never accumulates assistant responses.
            outputs_by_session = await run_chat_judge_benchmark(
                chat_sessions=session.samples.chat_sessions,
                max_output_tokens=args.max_output_len or 32,
                max_requests=args.num_prompts,
                semaphore=semaphore,
                benchmark_should_end_time=benchmark_should_end_time,
                request_driver=request_driver,
                model_id=session.model_id,
                api_url=session.api_url,
                lora_manager=session.lora_manager,
                warmup_delay_ms=args.chat_warmup_delay_ms,
                max_concurrency=max_concurrency,
                sampling=args.sampling,
                use_session_id_as_cache_salt=args.use_session_id_as_cache_salt,
            )
            all_outputs = [
                out for outs in outputs_by_session.values() for out in outs
            ]
        else:
            # multi-turn chat scenario
            assert chat_sessions is not None
            outputs_by_session = await run_multiturn_benchmark(
                chat_sessions=chat_sessions,
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
                sampling=args.sampling,
                run_prefix=run_prefix,
                run_prefix_len=run_prefix_len,
                request_rate=request_rate,
                burstiness=args.burstiness,
                est_ttft_ms=args.warmup_delay_estimated_ttft_ms,
                est_tpot_ms=args.warmup_delay_estimated_tpot_ms,
                use_session_id_as_cache_salt=args.use_session_id_as_cache_salt,
                disable_ignore_eos=args.disable_ignore_eos,
                max_output_len=args.max_output_len,
            )
            all_outputs = [
                out for outs in outputs_by_session.values() for out in outs
            ]

        benchmark_duration = (
            time.perf_counter_ns() - benchmark_start_time
        ) / 1e9

    if session.benchmark_task == "text-generation":
        spec_decode_metrics_after = fetch_spec_decode_metrics(
            backend, session.base_url, metrics_urls=args.metrics_urls
        )
    else:
        spec_decode_metrics_after = None
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

    gpu_metrics: list[GPUStatsSnapshot] | None = None
    gpu_device_groups: dict[str, str] = {}
    if args.collect_gpu_stats and gpu_recorder is not None:
        gpu_metrics = gpu_recorder.stats
        if isinstance(gpu_recorder, PrometheusGPURecorder):
            gpu_device_groups = gpu_recorder.device_groups

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
                verify_required_metrics=True,
            )
            logger.info("Collected server metrics (final)")
        except MissingEngineMetricsError:
            # A reachable endpoint missing the metrics this backend always
            # emits must fail the run, not degrade to null fields.
            raise
        except Exception as e:
            logger.warning(f"Failed to collect server metrics: {e}")

    result: BenchmarkResult
    if session.benchmark_task in PIXEL_GENERATION_TASKS:
        result = build_pixel_generation_result(
            outputs=all_outputs,
            benchmark_duration=benchmark_duration,
            gpu_metrics=gpu_metrics,
            cpu_metrics=cpu_metrics_result,
            max_concurrency=max_concurrency,
            collect_gpu_stats=args.collect_gpu_stats,
            metrics_by_endpoint=endpoint_metrics,
            gpu_device_groups=gpu_device_groups,
        )
    else:
        result = build_text_generation_result(
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
            kv_block_size=args.kv_block_size,
            record_request_text=args.record_request_text,
            gpu_device_groups=gpu_device_groups,
            outputs_by_session=outputs_by_session,
        )
    if session.lora_manager is not None:
        result.lora_metrics = session.lora_manager.metrics

    ok = report_and_validate(
        result,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        collect_gpu_stats=args.collect_gpu_stats,
        collect_cpu_stats=args.collect_cpu_stats,
        lora_manager=session.lora_manager,
    )
    return result, ok


def validate_task_and_endpoint(
    benchmark_task: BenchmarkTask, endpoint: Endpoint
) -> None:
    if benchmark_task == "text-generation":
        if endpoint in (
            "/v1/responses",
            "/v1/images/generations",
            "/v1/videos/sync",
            "/v1/videos",
        ):
            raise ValueError(
                f"--benchmark-task text-generation does not support "
                f"--endpoint {endpoint}"
            )
    elif benchmark_task in PIXEL_GENERATION_TASKS:
        if (
            endpoint in ("/v1/videos/sync", "/v1/videos")
            and benchmark_task not in VIDEO_GENERATION_TASKS
        ):
            raise ValueError(
                f"--endpoint {endpoint} is only valid for video tasks"
                f" {VIDEO_GENERATION_TASKS}, got {benchmark_task!r}"
            )
        if endpoint not in PIXEL_GENERATION_ENDPOINTS:
            raise ValueError(
                f"--benchmark-task {benchmark_task} requires --endpoint"
                f" to be one of {sorted(PIXEL_GENERATION_ENDPOINTS)},"
                f" got {endpoint!r}"
            )


def flush_prefix_cache(
    backend: Backend,
    host: str,
    port: int,
    dry_run: bool,
    base_url: str | None = None,
) -> None:
    """Flush the serving engine's prefix cache via HTTP POST.

    When ``base_url`` is set (a remote endpoint, for example, ``--base-url``), it
    replaces the ``http://<host>:<port>`` prefix and the request carries an
    ``Authorization: Bearer $OPENAI_API_KEY`` header, matching the auth the
    benchmark requests themselves send.
    """
    if backend not in CACHE_RESET_ENDPOINT_MAP:
        raise ValueError(
            f"Cannot flush prefix cache for {backend} backend: this backend"
            " does not support prefix cache flush."
        )
    import requests as _http_requests  # lazy - avoid hard dep for non-sweep use

    headers: Mapping[str, str] | None = None
    if base_url is not None:
        api_url = f"{base_url.rstrip('/')}{CACHE_RESET_ENDPOINT_MAP[backend]}"
        headers = openai_bearer_auth_headers()
    else:
        api_url = f"http://{host}:{port}{CACHE_RESET_ENDPOINT_MAP[backend]}"
    if dry_run:
        logger.info(f"Dry-run flush: POST {api_url}")
        return
    response = _http_requests.post(api_url, headers=headers)
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


BenchmarkStatus = Literal["pass", "fail"]
"""The outcome of one benchmark configuration's validation."""


@dataclass
class BenchmarkRunResult:
    """Result of one (max_concurrency, request_rate) benchmark configuration."""

    max_concurrency: int | None
    request_rate: float
    num_prompts: int
    labels: BenchmarkRunResultLabels
    """The benchmark configuration :attr:`result` was produced with, for its
    result JSON."""

    result: BenchmarkResult | None = None
    status: BenchmarkStatus = "pass"
    """Whether the result passed or failed validation."""


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


def _load_workload_yaml(args: ServingBenchmarkConfig) -> None:
    if not args.workload_config:
        return
    workload_path = Path(args.workload_config)
    apply_workload(
        args, load_workload_file(workload_path), base_dir=workload_path.parent
    )


_DEFAULT_NUM_PROMPTS = 1000
_DEFAULT_MAX_BENCHMARK_DURATION_S = 300


@dataclass(frozen=True)
class EndCondition:
    """When each run in a sweep stops sending requests.

    A run ends once it has sent its request count or reached
    ``max_duration_s``, whichever comes first. The count is either fixed
    (``num_prompts``) or scales with the run's max concurrency
    (``num_prompts_multiplier``); exactly one of the two is set.
    """

    num_prompts: int | None
    num_prompts_multiplier: int | None
    max_duration_s: int | None

    def num_prompts_for(self, max_concurrency: int | None) -> int:
        """Returns the request count for a run at ``max_concurrency``."""
        if self.num_prompts_multiplier is None:
            assert self.num_prompts is not None
            return self.num_prompts
        assert max_concurrency is not None
        return self.num_prompts_multiplier * max_concurrency


def _resolve_end_condition(
    args: ServingBenchmarkConfig,
    concurrency_range: Sequence[int | None],
) -> EndCondition:
    num_prompts = args.num_prompts
    multiplier = args.num_prompts_multiplier
    duration_s = args.max_benchmark_duration_s
    if num_prompts is not None and multiplier is not None:
        raise ValueError(
            "--num-prompts and --num-prompts-multiplier both set the request"
            " count; pass only one."
        )
    if multiplier is not None:
        if None in concurrency_range:
            raise ValueError(
                "--num-prompts-multiplier scales with --max-concurrency, so"
                " every --max-concurrency value must be bounded; got"
                f" {list(concurrency_range)} (the default is unbounded)."
            )
        # Without a time limit, the large counts the multiplier produces at
        # high concurrency could run indefinitely.
        if duration_s is None:
            logger.info(
                "Using --num-prompts-multiplier without"
                " --max-benchmark-duration-s. Defaulting to"
                f" {_DEFAULT_MAX_BENCHMARK_DURATION_S}s per max-concurrency"
                " level."
            )
            duration_s = _DEFAULT_MAX_BENCHMARK_DURATION_S
    elif num_prompts is None:
        if duration_s is None:
            logger.warning(
                "Neither --num-prompts nor --max-benchmark-duration-s is"
                f" specified. Defaulting to --num-prompts {_DEFAULT_NUM_PROMPTS}"
                " and --max-benchmark-duration-s"
                f" {_DEFAULT_MAX_BENCHMARK_DURATION_S}"
            )
            duration_s = _DEFAULT_MAX_BENCHMARK_DURATION_S
        num_prompts = _DEFAULT_NUM_PROMPTS
    return EndCondition(
        num_prompts=num_prompts,
        num_prompts_multiplier=multiplier,
        max_duration_s=duration_s,
    )


def _run_prefix_len(
    args: ServingBenchmarkConfig,
    benchmark_task: BenchmarkTask,
    tokenizer: PreTrainedTokenizerBase | None,
) -> int:
    """Tokens ``--force-unique-runs`` adds to a session's first measured turn.

    Zero when the flag is off, when there is no tokenizer, and for
    pixel-generation tasks, which do not track ``prompt_len`` at all.

    ``benchmark()`` builds the real prefix, but that runs after sampling, so
    the budget check here has to measure an equivalent one. Only the length
    matters and every ``uuid4()`` has the same shape, so a throwaway one is
    as good as the prefix the run will actually use.
    """
    if not args.force_unique_runs or tokenizer is None:
        return 0
    if benchmark_task in PIXEL_GENERATION_TASKS:
        return 0
    return len(tokenizer.encode(f"{uuid4()}: ", add_special_tokens=False))


def _sample_for_seed(
    args: ServingBenchmarkConfig,
    benchmark_task: BenchmarkTask,
    tokenizer: PreTrainedTokenizerBase | None,
    chat: bool,
    seed: int | None,
) -> Samples:
    """Draw one deterministic workload sample from ``seed``."""
    random.seed(seed)
    np.random.seed(seed)
    samples = sample_requests(
        args=args,
        benchmark_task=benchmark_task,
        tokenizer=tokenizer,
        chat=chat,
    )

    if args.tools is not None:
        augment_samples_with_tools(
            samples,
            tools=args.tools,
            fraction=args.tools_fraction,
            tokenizer=tokenizer,
            seed=seed,
        )

    if args.image_fraction > 0:
        augment_samples_with_images(
            samples,
            fraction=args.image_fraction,
            image_count=args.image_count,
            image_long_side=args.image_long_side,
            image_aspect_ratio=args.image_aspect_ratio,
            turn=args.image_turn,
            num_turns=args.image_num_turns,
            max_chat_len=(
                tokenizer.model_max_length if tokenizer is not None else None
            ),
            run_prefix_len=_run_prefix_len(args, benchmark_task, tokenizer),
            # Offset from the response-format mixer's stream so the two pick
            # independently rather than in lockstep.
            seed=None if seed is None else seed + 1,
        )

    # After images, which can cut sessions: a constraint on a cut turn
    # would never be sent.
    response_format = args.openai_response_format
    if response_format is not None:
        augment_samples_with_response_format(
            samples,
            response_format=response_format,
            fraction=args.response_format_fraction,
            turn=args.response_format_turn,
            seed=seed,
        )

    return samples


def _build_session(args: ServingBenchmarkConfig) -> BenchmarkSession:
    assert args.model is not None
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
        try:
            expected = get_pixel_gen_endpoint(args.backend, benchmark_task)
        except ValueError:
            raise ValueError(
                f"Backend {args.backend!r} does not have a default"
                f" pixel-generation endpoint. Explicitly pass --endpoint"
                f" with one of {sorted(PIXEL_GENERATION_ENDPOINTS)}."
            ) from None
        if endpoint != expected:
            logger.info(
                "Auto-selected endpoint %s for backend %s (%s task)",
                expected,
                args.backend,
                benchmark_task,
            )
            endpoint = expected

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
        model_max_length = args.model_max_length
        if model_max_length is None:
            # Best-effort: when the server is already up (e.g. benchmarking a
            # running deployment), adopt its real context limit so the
            # context-length guards derived from tokenizer.model_max_length
            # bind even when the tokenizer reports the HF unbounded sentinel.
            #
            # --dry-run asks this too. It returns before the later clamp in
            # main_with_parsed_args, so this is its only chance to learn the
            # real bound, and without it a preview reports a workload the
            # server would truncate. Costs one 2s-timeout request when no
            # server is listening.
            model_max_length = fetch_server_max_model_len(
                base_url, model_id, timeout_s=2.0
            )
            if model_max_length is not None:
                logger.info(
                    "Using server-reported max_model_len=%d from %s/v1/models",
                    model_max_length,
                    base_url,
                )
            elif args.dry_run:
                logger.warning(
                    "No server reachable at %s; --dry-run is using the"
                    " tokenizer's context limit, so any workload the server"
                    " would truncate is reported here as if it fit.",
                    base_url,
                )
        logger.info(f"getting tokenizer. api url: {api_url}")
        tokenizer = get_tokenizer(
            tokenizer_id,
            # An explicit revision wins: ``resolve_revision`` asks the Hub and
            # returns None for a repo it cannot see, which loads ``main``.
            revision=args.tokenizer_revision or resolve_revision(tokenizer_id),
            local_files_only=args.tokenizer_local_files_only,
            model_max_length=model_max_length,
            trust_remote_code=args.trust_remote_code,
        )

    samples = _sample_for_seed(args, benchmark_task, tokenizer, chat, args.seed)

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
            traffic_ratios=args.per_lora_traffic_ratio or None,
            uniform_ratio=args.lora_uniform_traffic_ratio,
            seed=args.seed,
            max_concurrent_lora_ops=args.max_concurrent_lora_ops,
        )
        lora_manager.log_traffic_distribution()

    # Handle trace flag
    trace_path = None
    if args.trace:
        assert_nvidia_gpu()
        trace_path = args.trace_file or get_default_trace_path()
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


def _compute_session_stats(
    args: ServingBenchmarkConfig, session: BenchmarkSession
) -> WorkloadStats:
    """Stats of the requests this run would send.

    Multi-turn sessions end at the tokenizer's context limit, so the stats
    cut them at the same one, and ``--force-unique-runs`` prepends its
    prefix to every request, so the stats count it.
    """
    return compute_workload_stats(
        session.samples,
        context_limit=session.tokenizer.model_max_length
        if session.tokenizer
        else None,
        run_prefix_len=_run_prefix_len(
            args, session.benchmark_task, session.tokenizer
        ),
    )


def _emit_workload_stats(
    args: ServingBenchmarkConfig,
    session: BenchmarkSession,
    stats: WorkloadStats,
) -> None:
    if args.workload_stats_json is not None:
        write_workload_stats(stats, args.workload_stats_json)
        logger.info("Wrote workload stats to %s", args.workload_stats_json)
    if args.print_workload_stats:
        print_workload_stats(session.samples, stats)


def _run_dry_run_sweep(
    args: ServingBenchmarkConfig,
    session: BenchmarkSession,
    end_condition: EndCondition,
    concurrency_range: Sequence[int | None],
    request_rate_range: Sequence[float],
    stats: WorkloadStats,
) -> Iterator[BenchmarkRunResult]:
    if not args.print_workload_stats:
        print_workload_stats(session.samples, stats)
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
                delay_biased=args.warmup_delay_biased,
                est_ttft_ms=args.warmup_delay_estimated_ttft_ms,
                est_tpot_ms=args.warmup_delay_estimated_tpot_ms,
            )
            if report is not None:
                log_warmup_sampling_report(report)
    for mc in concurrency_range:
        num_prompts = end_condition.num_prompts_for(mc)
        for rr in request_rate_range:
            print(
                f"Dry run: model={args.model}"
                f" host={args.host} port={args.port}"
                f" endpoint={args.endpoint}"
                f" max_concurrency={mc}"
                f" request_rate={rr}"
                f" num_prompts={num_prompts}"
                f" max_benchmark_duration_s="
                f"{end_condition.max_duration_s}"
            )
            yield BenchmarkRunResult(
                max_concurrency=mc,
                request_rate=rr,
                num_prompts=num_prompts,
                labels=BenchmarkRunResultLabels.from_config(
                    args,
                    benchmark_task=session.benchmark_task,
                    model_id=session.model_id,
                    tokenizer_id=session.tokenizer_id,
                ),
            )


def _run_benchmark_sweep(
    args: ServingBenchmarkConfig,
    session: BenchmarkSession,
    end_condition: EndCondition,
) -> Iterator[BenchmarkRunResult]:
    assert args.seed is not None, "resolve_seed must run first"
    # ---- Sweep loop ----
    for mc in args.max_concurrency:
        num_prompts = end_condition.num_prompts_for(mc)
        if end_condition.num_prompts_multiplier is not None:
            logger.info(
                f"Using num_prompts = {end_condition.num_prompts_multiplier}"
                f" * {mc} = {num_prompts}"
            )
        # Sampling, the run itself, and the result JSON read the count
        # from args.
        args.num_prompts = num_prompts

        # Each concurrency level draws its own sample, derived from the base
        # seed, so levels are reproducible individually without replaying
        # the exact same requests at every level.
        mc_seed = seed_for_concurrency(args.seed, mc)
        session.samples = _sample_for_seed(
            args,
            session.benchmark_task,
            session.tokenizer,
            session.endpoint == "/v1/chat/completions",
            mc_seed,
        )

        for rr in args.request_rate:
            iteration_results: list[BenchmarkResult] = []
            validation_passed = True
            for _iteration in range(args.num_iters):
                if args.flush_prefix_cache:
                    flush_prefix_cache(
                        args.backend,
                        args.host,
                        args.port,
                        args.dry_run,
                        base_url=args.base_url,
                    )

                logger.info("mc=%s seed=%d", mc, mc_seed)

                result, ok = asyncio.run(benchmark(args, session, mc, rr))
                iteration_results.append(result)
                if not ok:
                    validation_passed = False

            # Median selection when running multiple iterations.
            if len(iteration_results) > 1:
                throughputs = np.asarray(
                    [
                        agg.request_throughput
                        for r in iteration_results
                        if (agg := r.aggregates) is not None
                    ]
                )
                idx = argmedian(throughputs)
            else:
                idx = 0
            best_result = iteration_results[idx]

            yield BenchmarkRunResult(
                max_concurrency=mc,
                request_rate=rr,
                num_prompts=num_prompts,
                result=retain_result(
                    best_result,
                    passed=validation_passed,
                    always_save_result=args.always_save_result,
                ),
                labels=BenchmarkRunResultLabels.from_config(
                    args,
                    benchmark_task=session.benchmark_task,
                    model_id=session.model_id,
                    tokenizer_id=session.tokenizer_id,
                ),
                status="pass" if validation_passed else "fail",
            )


def main_with_parsed_args(
    args: ServingBenchmarkConfig,
    *,
    server_liveness: Callable[[], bool] | None = None,
) -> Generator[BenchmarkRunResult, None, None]:
    """Runs a serving benchmark sweep.

    Yields one :class:`BenchmarkRunResult` per (concurrency, request rate)
    combination.

    Args:
        args: The benchmark configuration.
        server_liveness: A predicate that reports whether the server process
            is alive, so the server-ready wait aborts promptly on a crashed
            bring-up, or ``None`` to wait until the timeout.

    Yields:
        Each combination's result, as soon as it completes.

    Raises:
        ValueError: If ``args.model`` is not set.
    """
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)s: %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    logger.info(args)

    if args.model is None:
        raise ValueError("--model is required when running benchmark")

    _load_workload_yaml(args)
    resolve_seed(args)

    end_condition = _resolve_end_condition(
        args=args, concurrency_range=args.max_concurrency
    )
    args.max_benchmark_duration_s = end_condition.max_duration_s
    # The session samples once up front, so size it for the largest level.
    args.num_prompts = max(
        (end_condition.num_prompts_for(mc) for mc in args.max_concurrency),
        default=end_condition.num_prompts_for(1),
    )

    session = _build_session(args)

    if args.dry_run:
        # Dry-run exits before the server-ready clamp below, so the
        # tokenizer's limit is the tightest bound it can apply.
        stats = _compute_session_stats(args, session)
        _emit_workload_stats(args, session, stats)
        if args.print_inputs_and_outputs:
            print_input_prompts(session.samples)
        yield from _run_dry_run_sweep(
            args=args,
            session=session,
            end_condition=end_condition,
            concurrency_range=args.max_concurrency,
            request_rate_range=args.request_rate,
            stats=stats,
        )
        return

    wait_for_server_ready(
        args.host,
        args.port,
        timeout_s=args.server_ready_timeout_s,
        backend=args.backend,
        liveness_check=server_liveness,
        base_url=args.base_url,
    )

    # The server may not have been up during session build (it is launched
    # concurrently with dataset sampling). Now that it is ready, adopt its
    # context limit when it is tighter than the tokenizer's, so the
    # multi-turn max_chat_len guards bind to the real bound.
    if args.model_max_length is None and session.tokenizer is not None:
        max_model_len = fetch_server_max_model_len(
            session.base_url, session.model_id
        )
        if (
            max_model_len is not None
            and max_model_len < session.tokenizer.model_max_length
        ):
            logger.info(
                "Clamping tokenizer model_max_length %d to server-reported"
                " max_model_len %d",
                session.tokenizer.model_max_length,
                max_model_len,
            )
            session.tokenizer.model_max_length = max_model_len

    # After the clamp above, which may have just tightened the limit the
    # stats cut multi-turn sessions at, and still before any request is
    # sent: the stats must describe what this run will actually send.
    if args.print_workload_stats or args.workload_stats_json:
        _emit_workload_stats(
            args, session, _compute_session_stats(args, session)
        )
    if args.print_inputs_and_outputs:
        print_input_prompts(session.samples)

    yield from _run_benchmark_sweep(
        args=args, session=session, end_condition=end_condition
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
        ] = ServingBenchmarkConfig(),  # noqa: B008
    ) -> None:
        parsed_configs.append(config)

    app(clean_args)
    if not parsed_configs:
        raise SystemExit(0)
    config = parsed_configs[0]
    if metadata_values:
        config.metadata = metadata_values
    return config
