# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
# File contains code from the vllm project
# https://github.com/vllm-project/vllm/blob/v0.6.0/benchmarks/benchmark_throughput.py
# used under the Apache 2 licenced

"""Benchmark offline inference throughput."""

from __future__ import annotations

import argparse
import asyncio
import enum
import json
import os
import random
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet
from benchmark_shared.config import BaseBenchmarkConfig
from benchmark_shared.datasets import (
    BenchmarkDataset,
    CodeDebugBenchmarkDataset,
)
from huggingface_hub import hf_hub_download
from max.entrypoints.cli import DevicesOptionType
from max.interfaces import (
    PipelinesFactory,
    PipelineTask,
    RequestID,
    SamplingParams,
    SamplingParamsInput,
    TextGenerationRequest,
)
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    SupportedEncoding,
    TextTokenizer,
)
from max.serve.config import Settings
from max.serve.pipelines.llm import (
    EmbeddingsGenerationOutput,
    TokenGeneratorOutput,
    TokenGeneratorPipeline,
)
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.scheduler.queues import SchedulerZmqConfigs
from max.serve.telemetry.metrics import NoopClient
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class ThroughputBenchmarkConfig(BaseBenchmarkConfig):
    """Configuration class for throughput benchmarks (benchmark_throughput.py).

    Inherits from BaseBenchmarkConfig and adds throughput-specific parameters:
    - Quantization and model loading configuration
    - Batching and performance parameters
    - KV Cache configuration
    - Device and memory configuration
    - Pipeline configuration
    - Async engine settings
    """

    # Backend configuration (throughput-specific)
    backend: str = field(
        default="modular",
        metadata={
            "group": "Backend Configuration",
            "group_description": "Backend selection and basic configuration",
        },
    )
    """Backend (throughput benchmarks typically use modular backend only)."""

    # Model configuration (throughput-specific extensions)
    quantization_encoding: str | None = field(
        default=None,
        metadata={
            "group": "Model Configuration",
            "group_description": "Model loading and quantization settings",
        },
    )
    """Quantization encoding. Choices: q4_0, q4_k, q6_k, bfloat16, float32, null"""

    weight_path: str | None = field(
        default=None, metadata={"group": "Model Configuration"}
    )
    """Path for already-downloaded pretrained weight file."""

    # Input/Output configuration (throughput-specific)
    input_len: int | None = field(
        default=None,
        metadata={
            "group": "Input/Output Configuration",
            "group_description": "Parameters controlling input and output lengths",
        },
    )
    """Input prompt length for each request."""

    output_len: int | None = field(
        default=None, metadata={"group": "Input/Output Configuration"}
    )
    """Output length for each request (overrides dataset output length)."""

    # Batching and performance configuration (throughput-specific)
    max_batch_size: int | None = field(
        default=None,
        metadata={
            "group": "Batching and Performance",
            "group_description": "Parameters controlling batching and performance optimization",
        },
    )
    """Maximum number of requests to include in a single batch."""

    max_num_steps: int = field(
        default=10, metadata={"group": "Batching and Performance"}
    )
    """Maximum number of steps to run per multi-step scheduling call."""

    async_engine: bool = field(
        default=True, metadata={"group": "Batching and Performance"}
    )
    """Use Modular async pipeline engine rather than LLM class."""

    # KV Cache configuration (throughput-specific)
    cache_strategy: KVCacheStrategy = field(
        default=KVCacheStrategy.PAGED,
        metadata={
            "group": "KV Cache Configuration",
            "group_description": "Parameters controlling KV cache behavior and memory management",
        },
    )
    """The KVCache strategy to use."""

    kv_cache_page_size: int | None = field(
        default=None, metadata={"group": "KV Cache Configuration"}
    )
    """Number of tokens in a single page in the paged kv cache."""

    enable_prefix_caching: bool = field(
        default=False, metadata={"group": "KV Cache Configuration"}
    )
    """Enable prefix caching of kv cache entries when using paged attention."""

    enable_kvcache_swapping_to_host: bool = field(
        default=False, metadata={"group": "KV Cache Configuration"}
    )
    """Enable swapping KVCache blocks to host memory."""

    host_kvcache_swap_space_gb: float = field(
        default=50.0, metadata={"group": "KV Cache Configuration"}
    )
    """Amount of host memory for host KVCache in GiB."""

    # Device and memory configuration (throughput-specific)
    device_memory_utilization: float | None = field(
        default=None,
        metadata={
            "group": "Device and Memory Configuration",
            "group_description": "Parameters controlling device selection and memory usage",
        },
    )
    """Fraction of available device memory to consume."""

    devices: str | None = field(
        default=None, metadata={"group": "Device and Memory Configuration"}
    )
    """Device ID to target (GPU configuration)."""

    # Pipeline configuration (throughput-specific)
    pipeline_task: PipelineTask = field(
        default=PipelineTask.TEXT_GENERATION,
        metadata={
            "group": "Pipeline Configuration",
            "group_description": "Parameters controlling pipeline behavior and task execution",
        },
    )
    """Type of task to complete using the pipeline."""

    max_length: int | None = field(
        default=None, metadata={"group": "Pipeline Configuration"}
    )
    """Maximum length of sequence (including prompt and output)."""

    # Sampling parameters
    top_k: int | None = None

    # Output configuration (throughput-specific)
    output_json: str | None = field(
        default=None,
        metadata={
            "group": "Output Configuration",
            "group_description": "Parameters controlling output format and display",
        },
    )
    """Path to save throughput results in JSON format."""

    show_text: bool = field(
        default=False, metadata={"group": "Output Configuration"}
    )
    """Whether to show generated text."""

    @staticmethod
    def help() -> dict[str, str]:
        """Documentation for throughput benchmark config parameters.

        Returns:
            Dictionary of config options and their descriptions.
        """
        # Get base help and extend with throughput-specific parameters
        base_help = BaseBenchmarkConfig.help()
        throughput_help = {
            "backend": "Backend (throughput benchmarks typically use modular backend only).",
            "quantization_encoding": "Quantization encoding. Choices: q4_0, q4_k, q6_k, bfloat16, float32, null",
            "weight_path": "Path for already-downloaded pretrained weight file.",
            "input_len": "Input prompt length for each request.",
            "output_len": "Output length for each request (overrides dataset output length).",
            "max_batch_size": "Maximum number of requests to include in a single batch.",
            "max_num_steps": "Maximum number of steps to run per multi-step scheduling call.",
            "async_engine": "Use Modular async pipeline engine rather than LLM class.",
            "cache_strategy": "The KVCache strategy to use.",
            "kv_cache_page_size": "Number of tokens in a single page in the paged kv cache.",
            "enable_prefix_caching": "Enable prefix caching of kv cache entries when using paged attention.",
            "enable_kvcache_swapping_to_host": "Enable swapping KVCache blocks to host memory.",
            "host_kvcache_swap_space_gb": "Amount of host memory for host KVCache in GiB.",
            "device_memory_utilization": "Fraction of available device memory to consume.",
            "devices": "Device ID to target (GPU configuration).",
            "pipeline_task": "Type of task to complete using the pipeline.",
            "max_length": "Maximum length of sequence (including prompt and output).",
            "output_json": "Path to save throughput results in JSON format.",
            "show_text": "Whether to show generated text.",
        }
        return {**base_help, **throughput_help}

    @classmethod
    def _get_enum_mapping_impl(cls) -> Mapping[str, type[enum.Enum]]:
        """Get the enum mapping for ThroughputBenchmarkConfig."""
        return {
            "KVCacheStrategy": KVCacheStrategy,
            "PipelineTask": PipelineTask,
        }


@dataclass
class RequestPayload:
    prompt: str
    prompt_len: int
    output_len: int
    image: bytes | None


def load_parquet_dataset(
    dataset_path: str,
) -> tuple[list[tuple[str, str]], list[bytes]]:
    dataset_table = pyarrow.parquet.read_table(dataset_path)
    dataset: list[tuple[str, str]] = []
    images: list[bytes] = []
    for row in dataset_table.to_pylist():
        conversation = json.loads(row["conversation"])
        # Filter out the conversations with less than 2 turns.
        if len(conversation) < 2:
            continue
        assert conversation[0]["role"] == "user"
        assert len(conversation[0]["content"]) == 2
        assert conversation[0]["content"][0]["type"] == "image"
        assert conversation[0]["content"][1]["type"] == "text"
        assert conversation[1]["role"] == "assistant"
        assert len(conversation[1]["content"]) == 1
        assert conversation[1]["content"][0]["type"] == "text"
        # Only keep the first two turns of each conversation.
        dataset.append(
            (
                conversation[0]["content"][1]["text"],
                conversation[1]["content"][0]["text"],
            )
        )
        images.append(row["image"]["bytes"])
    return dataset, images


# TODO: We should just consolidate this with the sample_requests methods of each
# BenchmarkDataset subclass.
def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: int | None,
    max_length: int | None,
) -> list[RequestPayload]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    dataset_ext = os.path.splitext(dataset_path)[1]
    images: list[bytes] = []
    if dataset_ext == ".json":
        with open(dataset_path) as f:
            dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [
            (
                data["conversations"][0]["value"],
                data["conversations"][1]["value"],
            )
            for data in dataset
        ]
    elif dataset_ext == ".parquet":
        dataset, images = load_parquet_dataset(dataset_path)
    else:
        raise ValueError(
            f"Don't know how to parse datasets with extension {dataset_ext}"
        )

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: list[RequestPayload] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        if max_length:
            prompt_token_ids = tokenizer(
                prompt, max_length=max_length, truncation=True
            ).input_ids

            # If the ids are truncated, update the prompt.
            if len(prompt_token_ids) == max_length:
                prompt = tokenizer.decode(
                    prompt_token_ids, skip_special_tokens=True
                )
        else:
            prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids)
            if fixed_output_len is None
            else fixed_output_len
        )
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        image = images[i] if i < len(images) else None
        filtered_dataset.append(
            RequestPayload(prompt, prompt_len, output_len, image)
        )

    return filtered_dataset


async def all_tokens(
    model_name: str,
    pipeline: TokenGeneratorPipeline,
    pbar: tqdm,
    request_id: int,
    request_payload: RequestPayload,
    top_k: int | None,
) -> tuple[str, list[TokenGeneratorOutput]]:
    """Generate all tokens for a request."""
    prompt = request_payload.prompt
    output_len = request_payload.output_len
    image = request_payload.image

    params = SamplingParamsInput(
        max_new_tokens=output_len,
        ignore_eos=True,
        top_k=top_k,
    )
    sampling_params = SamplingParams.from_input(params)
    request = TextGenerationRequest(
        request_id=RequestID(str(request_id)),
        model_name=model_name,
        prompt=prompt,
        images=[image] if image is not None else None,
        sampling_params=sampling_params,
    )

    # Generate this request until complete
    tokens = await pipeline.all_tokens(request)
    pbar.update(1)
    return (str(request.request_id), tokens)


def print_results(
    requests: list[RequestPayload],
    results: dict[str, list[TokenGeneratorOutput] | EmbeddingsGenerationOutput],
) -> None:
    for i, outputs in results.items():
        if isinstance(outputs, EmbeddingsGenerationOutput):
            output_text = str(outputs.embeddings)
        else:
            output_tokens = [
                generated_token.decoded_token for generated_token in outputs
            ]
            output_text = "".join(
                token for token in output_tokens if token is not None
            )
        print(f'task#{i}: {{"{requests[int(i)].prompt}", "{output_text}"}}')


async def pipeline_encode(
    model_name: str,
    pipeline: TokenGeneratorPipeline,
    pbar: tqdm,
    request_id: int,
    request_payload: RequestPayload,
) -> tuple[str, EmbeddingsGenerationOutput]:
    """Encodes the request."""
    prompt = request_payload.prompt
    output_len = request_payload.output_len

    request = TextGenerationRequest(
        request_id=RequestID(str(request_id)),
        model_name=model_name,
        prompt=prompt,
    )

    # Generate this request until complete
    tokens = await pipeline.encode(request)
    assert tokens is not None
    pbar.update(1)
    return (str(request.request_id), tokens)


async def run_max_async(
    model_name: str,
    requests: list[RequestPayload],
    config: PipelineConfig,
    model_factory: PipelinesFactory,
    tokenizer: TextTokenizer,
    show_text: bool,
    pipeline_task: PipelineTask,
    top_k: int | None,
) -> tuple[float, list[int]]:
    scheduler_zmq_configs = SchedulerZmqConfigs(pipeline_task)
    async with (
        # Start the model worker process.
        start_model_worker(
            model_factory=model_factory,
            pipeline_config=config,
            settings=Settings(),
            metric_client=NoopClient(),
            scheduler_zmq_configs=scheduler_zmq_configs,
        ) as worker_monitor,
        # Create dynamic and continuous batching workers and associated queues
        # to feed the model worker process.
        TokenGeneratorPipeline(
            model_name=model_name,
            tokenizer=tokenizer,  # type: ignore
            scheduler_zmq_configs=scheduler_zmq_configs,
            worker_monitor=worker_monitor,
        ) as pipeline,
    ):
        # Start timing and create a progress bar.
        pbar = tqdm(total=len(requests))

        start = time.perf_counter()

        # Submit all request for execution in the model worker.
        if pipeline_task == PipelineTask.TEXT_GENERATION:
            all_tokens_tasks = [
                all_tokens(model_name, pipeline, pbar, i, request, top_k)
                for i, request in enumerate(requests)
            ]
        elif pipeline_task == PipelineTask.EMBEDDINGS_GENERATION:
            all_tokens_tasks = [
                pipeline_encode(model_name, pipeline, pbar, i, request)  # type: ignore
                for i, request in enumerate(requests)
            ]
        else:
            raise ValueError(f"Benchmarking does not support {pipeline_task}.")
        all_results = dict(await asyncio.gather(*all_tokens_tasks))

        # Wind down timing and the progress bar.
        end = time.perf_counter()
        pbar.close()

        assert all_results is not None
        assert len(all_results) == len(requests)

        if show_text:
            print_results(requests, all_results)  # type: ignore

        generated_tokens_len = [0] * len(all_results)
        if pipeline_task == PipelineTask.TEXT_GENERATION:
            for i, (request_id, generated_tokens) in enumerate(
                all_results.items()
            ):
                if len(generated_tokens) == 0:
                    warnings.warn(
                        f"WARNING: task#{request_id}: Empty response."
                    )
                    continue
                generated_tokens_len[i] = len(generated_tokens)

        return float(end - start), generated_tokens_len


def load_model_config(
    model_id: str,
    devices: str | list[int] | None,
    weight_path: str | None,
    quantization_encoding: str | None,
    max_length: int | None,
    max_batch_size: int | None,
    kv_cache_page_size: int | None,
    enable_prefix_caching: bool | None,
    enable_kvcache_swapping_to_host: bool | None,
    host_kvcache_swap_space_gb: float | None,
    device_memory_utilization: float | None,
    max_num_steps: int | None,
    trust_remote_code: bool | None,
    pipeline_task: PipelineTask,
) -> tuple[PipelinesFactory, PipelineConfig, TextTokenizer]:
    config_kwargs: dict[str, Any] = {}

    # Match what we already do in SDK/lib/API/python/max/entrypoints/cli/config.py
    if not devices:
        devices = "cpu"
    elif isinstance(devices, str) and devices.startswith("gpu:"):
        devices = [int(id) for id in devices[4:].split(",")]
    elif devices != "gpu":
        raise ValueError(
            f"Invalid devices: {devices}, must be 'cpu', 'gpu' or 'gpu:<id1,id2,...>'"
        )

    config_kwargs["device_specs"] = DevicesOptionType.device_specs(devices)

    if trust_remote_code:
        config_kwargs["trust_remote_code"] = trust_remote_code

    if weight_path:
        if not (
            os.path.isfile(weight_path) and os.access(weight_path, os.R_OK)
        ):
            raise ValueError(f"Invalid path: {weight_path}")
        config_kwargs["weight_path"] = [Path(weight_path)]

    config_kwargs["model_path"] = model_id

    if quantization_encoding:
        config_kwargs["quantization_encoding"] = SupportedEncoding(
            quantization_encoding
        )

    if max_length:
        config_kwargs["max_length"] = max_length

    if max_batch_size:
        config_kwargs["max_batch_size"] = max_batch_size

    if kv_cache_page_size:
        config_kwargs["kv_cache_page_size"] = kv_cache_page_size

    if enable_prefix_caching:
        config_kwargs["enable_prefix_caching"] = enable_prefix_caching

    if enable_kvcache_swapping_to_host:
        config_kwargs["enable_kvcache_swapping_to_host"] = (
            enable_kvcache_swapping_to_host
        )

    if host_kvcache_swap_space_gb:
        config_kwargs["host_kvcache_swap_space_gb"] = host_kvcache_swap_space_gb

    if device_memory_utilization:
        config_kwargs["device_memory_utilization"] = device_memory_utilization

    if max_num_steps:
        config_kwargs["max_num_steps"] = max_num_steps

    config = PipelineConfig(**config_kwargs)

    if len(config.model_config.weight_path) == 0:
        hf_file_kwargs = {}
        hf_file_kwargs["encoding"] = config.model_config.quantization_encoding

    tokenizer, pipeline_factory = PIPELINE_REGISTRY.retrieve_factory(
        config, task=pipeline_task
    )
    return pipeline_factory, config, tokenizer  # type: ignore


def fetch_dataset_from_hf(dataset_name: str) -> str:
    if dataset_name == "sharegpt":
        return hf_hub_download(
            repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
            filename="ShareGPT_V3_unfiltered_cleaned_split.json",
            repo_type="dataset",
        )
    elif dataset_name == "code_debug":
        return hf_hub_download(
            repo_id="xinrongzhang2022/InfiniteBench",
            filename="code_debug.jsonl",
            repo_type="dataset",
        )
    elif dataset_name == "mm-mt-bench":
        return hf_hub_download(
            repo_id="mistralai/MM-MT-Bench",
            filename="data/eval-00000-of-00001.parquet",
            repo_type="dataset",
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main(args: argparse.Namespace) -> None:
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        model_max_length=args.model_max_length,
        trust_remote_code=args.trust_remote_code,
    )

    # TODO: benchmark_throughput.py should be refactored to use the BenchmarkDataset class.
    # and not use its own fetch_dataset_from_hf() here.
    if args.dataset_name:
        dataset_path = fetch_dataset_from_hf(args.dataset_name)
    elif args.dataset_path:
        dataset_path = args.dataset_path
    else:
        dataset_path = None

    # Sample the requests.
    if dataset_path:
        optional_kwargs = {}
        optional_kwargs["fixed_output_len"] = args.output_len

        # TODO: benchmark_throughput.py should be refactored to use the BenchmarkDataset class.
        # Some of the fetch_dataset_from_hf() logic have different filenames
        # than the ones defined in benchmark_shared.datasets. These should be reconciled.
        if args.dataset_name == "code_debug":
            benchmark_dataset = BenchmarkDataset.from_flags(
                dataset_name=args.dataset_name,
                dataset_path=dataset_path,
            )
            assert isinstance(benchmark_dataset, CodeDebugBenchmarkDataset), (
                "code_debug dataset must be a CodeDebugBenchmarkDataset"
            )

            # code_debug is a long-context dataset based on InfiniteBench
            def sample_requests_func(  # noqa: ANN202
                dataset_path: str,
                num_requests: int,
                tokenizer: PreTrainedTokenizerBase,
                **kwargs,
            ):
                # CodeDebugBenchmarkDataset.sample_requests doesn't take dataset_path
                # because it already knows its dataset path
                sampled = benchmark_dataset.sample_requests(
                    num_requests=num_requests, tokenizer=tokenizer, **kwargs
                )
                converted = []
                for request in sampled:
                    # keep mypy happy
                    assert request.output_len is not None, (
                        "output_len is required for CodeDebugBenchmarkDataset"
                    )
                    assert isinstance(request.prompt_formatted, str)
                    converted.append(
                        RequestPayload(
                            request.prompt_formatted,
                            request.prompt_len,
                            request.output_len,
                            None,
                        )
                    )
                return converted

        else:
            sample_requests_func = sample_requests  # type: ignore
            optional_kwargs["max_length"] = args.max_length

        requests = sample_requests_func(
            dataset_path=dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            **optional_kwargs,
        )
    else:
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (args.input_len - 1)
        requests = [
            RequestPayload(prompt, args.input_len, args.output_len, None)
            for _ in range(args.num_prompts)
        ]

    if args.backend == "modular":
        print("\nLoading...")
        load_kwargs = dict(
            model_id=args.model,
            devices=args.devices,
            weight_path=args.weight_path,
            quantization_encoding=args.quantization_encoding,
            max_length=args.max_length,
            max_batch_size=args.max_batch_size,
            kv_cache_page_size=args.kv_cache_page_size,
            enable_prefix_caching=args.enable_prefix_caching,
            enable_kvcache_swapping_to_host=args.enable_kvcache_swapping_to_host,
            host_kvcache_swap_space_gb=args.host_kvcache_swap_space_gb,
            device_memory_utilization=args.device_memory_utilization,
            max_num_steps=args.max_num_steps,
            trust_remote_code=args.trust_remote_code,
            pipeline_task=args.pipeline_task,
        )
        model_factory, config, model_tokenizer = load_model_config(
            **load_kwargs
        )
        print(f"INFO: MODEL config = {config}")

        run_kwargs = dict(
            model_name=args.model,
            requests=requests,
            config=config,
            model_factory=model_factory,
            tokenizer=model_tokenizer,
            show_text=args.show_text,
            pipeline_task=args.pipeline_task,
            top_k=args.top_k,
        )

        print("\nExecuting...")
        if args.async_engine:
            elapsed_time, generated_tokens_len = asyncio.run(
                run_max_async(**run_kwargs)
            )
        else:
            raise ValueError("Non-async LLM Engine not supported yet")
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    total_num_input_tokens = sum(request.prompt_len for request in requests)
    total_num_output_tokens = sum(generated_tokens_len)

    if args.show_text and args.pipeline_task == PipelineTask.TEXT_GENERATION:
        print("\nPrompt Size [Input, Output_Real(Output_Expected)]:")
        for i, request in enumerate(requests):
            prompt_len = request.prompt_len
            output_len = request.output_len
            output_real = generated_tokens_len[i]
            print(
                f"task#{i}: [{prompt_len}, {output_real}({output_len})]", end=""
            )
            if output_real + prompt_len >= config.max_length:
                print(
                    (
                        "  # [WARNING] limited by maximum sequence length"
                        f" ({config.max_length}) from the model config."
                    ),
                    end="",
                )
            print()

    total_num_tokens = total_num_input_tokens + total_num_output_tokens
    dataset_basename = (
        os.path.basename(dataset_path) if dataset_path else "Synthetic-hi"
    )
    results = {
        "dataset_basename": dataset_basename,
        "elapsed_time_ms": elapsed_time * 1000.0,
        "num_requests": len(requests),
        "total_num_input_tokens": total_num_input_tokens,
        "total_num_output_tokens": total_num_output_tokens,
        "total_num_tokens": total_num_tokens,
        "requests_per_second": len(requests) / elapsed_time,
        "tokens_per_second": total_num_tokens / elapsed_time,
    }
    print("\nBenchmark Result:")
    print("--------")
    for key, value in results.items():
        print(f"{key}: {value}")

    print()
    # Output JSON results if specified
    if args.output_json:
        output_filename = args.output_json
        if not output_filename.endswith(".json"):
            output_filename += ".json"
        print(
            f"INFO: Write result to {output_filename}...",
            end="",
        )
        with open(output_filename, "w") as f:
            json.dump(results, f, indent=4)
        print("DONE")


if __name__ == "__main__":
    # Load configuration from YAML file and create argument parser
    config_path = os.path.join(
        os.path.dirname(__file__),
        "configs",
        "throughput_config.yaml",
    )
    config = ThroughputBenchmarkConfig.from_config_file(config_path)
    parser = config.cli_arg_parsers()

    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model

    if (
        args.enable_prefix_caching
        and args.cache_strategy != KVCacheStrategy.PAGED
    ):
        raise ValueError(
            "prefix caching is only supported with paged attention"
        )

    if args.dataset_name is None and args.dataset_path is None:
        if args.input_len is None:
            raise ValueError("Unknown input length to synthetic prompts")
        if args.output_len is None:
            raise ValueError("Unknown output length for each request")
    else:
        if args.input_len is not None:
            raise ValueError(
                "Unable to set input length. The input length will be derived"
                " from the dataset"
            )

    if __debug__:
        print(f"\nINFO: args = {args}")

    main(args)
