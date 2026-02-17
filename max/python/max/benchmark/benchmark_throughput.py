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
# File contains code from the vllm project
# https://github.com/vllm-project/vllm/blob/v0.6.0/benchmarks/benchmark_throughput.py
# used under the Apache 2 licenced

"""Benchmark offline inference throughput."""

from __future__ import annotations

import asyncio
import enum
import json
import os
import random
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import pyarrow.parquet
from cyclopts import App, Parameter
from cyclopts.config import Env
from huggingface_hub import hf_hub_download
from max.benchmark.benchmark_shared.config import (
    BenchmarkCommonConfig,
    SamplingConfig,
)
from max.benchmark.benchmark_shared.datasets import (
    BenchmarkDataset,
    CodeDebugBenchmarkDataset,
)
from max.config import ConfigFileModel
from max.interfaces import (
    PipelinesFactory,
    PipelineTask,
    RequestID,
    SamplingParams,
    SamplingParamsInput,
    TextGenerationOutput,
    TextGenerationRequest,
)
from max.nn.legacy.kv_cache import KVCacheStrategy
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    TextAndVisionContext,
    TextContext,
    TextTokenizer,
)
from max.pipelines.lib.device_specs import (
    device_specs_from_normalized_device_handle,
    normalize_device_specs_input,
)
from max.serve.config import Settings
from max.serve.pipelines.llm import (
    EmbeddingsGenerationOutput,
    TokenGeneratorOutput,
    TokenGeneratorPipeline,
)
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.telemetry.metrics import NoopClient
from max.serve.worker_interface.zmq_interface import ZmqModelWorkerInterface
from pydantic import Field
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

DESCRIPTION = """
    Synthetic benchmark to measure batched pipeline performance without
    serving or scheduling overhead.
"""


class ThroughputBenchmarkCommonConfig(BenchmarkCommonConfig):
    """Configuration class for throughput benchmark common options."""

    num_prompts: int = Field(default=1000)
    """Number of prompts to process."""


@Parameter(name="*")
class ThroughputBenchmarkConfig(ConfigFileModel):
    """Configuration class for benchmark options."""

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    """Configuration class for pipeline options."""

    other: ThroughputBenchmarkCommonConfig = ThroughputBenchmarkCommonConfig()
    """Configuration class for throughput benchmark common options."""

    sampling: SamplingConfig = SamplingConfig()
    """Configuration class for sampling options."""

    backend: str = Field(default="modular")
    """Backend (throughput benchmarks typically use modular backend only)."""

    # Input/Output configuration (throughput-specific)
    input_len: int | None = Field(default=None)
    """Input prompt length for each request."""

    output_len: int | None = Field(default=None)
    """Output length for each request (overrides dataset output length)."""

    async_engine: bool = Field(default=True)
    """Use Modular async pipeline engine rather than LLM class."""

    # TODO: These have different default values than ones configured via
    # PipelineConfig constructor.
    # KV Cache configuration (throughput-specific)
    cache_strategy: KVCacheStrategy = Field(
        default=KVCacheStrategy.PAGED,
    )
    """The KVCache strategy to use."""

    kv_cache_page_size: int | None = Field(default=None)
    """Number of tokens in a single page in the paged kv cache."""

    enable_prefix_caching: bool = Field(default=False)
    """Enable prefix caching of kv cache entries when using paged attention."""

    devices: str | None = Field(default=None)
    """Device spec string (e.g. cpu, gpu, gpu:0,1). Overrides pipeline devices."""

    # Pipeline configuration (throughput-specific)
    pipeline_task: PipelineTask = Field(
        default=PipelineTask.TEXT_GENERATION,
    )
    """Type of task to complete using the pipeline."""

    output_json: str | None = Field(default=None)
    """Path to save throughput results in JSON format."""

    show_text: bool = Field(default=False)
    """Whether to show generated text."""

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
    config: PipelineConfig,
) -> tuple[str, list[TokenGeneratorOutput]]:
    """Generate all tokens for a request."""
    prompt = request_payload.prompt
    output_len = request_payload.output_len
    images: list[bytes] = []
    if request_payload.image:
        images.append(request_payload.image)

    params = SamplingParamsInput(
        max_new_tokens=output_len,
        ignore_eos=True,
        top_k=top_k,
    )
    sampling_params = SamplingParams.from_input_and_generation_config(
        params,
        sampling_params_defaults=config.model.sampling_params_defaults,
    )
    request = TextGenerationRequest(
        request_id=RequestID(str(request_id)),
        model_name=model_name,
        prompt=prompt,
        images=images,
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
            output_text = "".join(
                chunk.decoded_tokens
                for chunk in outputs
                if chunk.decoded_tokens is not None
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
    model_factory: PipelinesFactory,  # type: ignore[type-arg]  # TODO
    tokenizer: TextTokenizer,
    show_text: bool,
    pipeline_task: PipelineTask,
    top_k: int | None,
) -> tuple[float, list[int]]:
    model_worker_interface = ZmqModelWorkerInterface[
        TextAndVisionContext | TextContext, TextGenerationOutput
    ](
        pipeline_task,
        context_type=PIPELINE_REGISTRY.retrieve_context_type(config),
    )
    async with (
        # Start the model worker process.
        start_model_worker(
            model_factory=model_factory,
            pipeline_config=config,
            settings=Settings(),
            metric_client=NoopClient(),
            model_worker_interface=model_worker_interface,
        ) as model_worker,
    ):
        # Create dynamic and continuous batching workers and associated queues
        # to feed the model worker process.
        pipeline = TokenGeneratorPipeline(
            model_name=model_name,
            tokenizer=tokenizer,
            model_worker=model_worker,
        )
        # Start timing and create a progress bar.
        pbar = tqdm(total=len(requests))

        start = time.perf_counter()

        # Submit all request for execution in the model worker.
        if pipeline_task == PipelineTask.TEXT_GENERATION:
            all_tokens_tasks = [
                all_tokens(
                    model_name, pipeline, pbar, i, request, top_k, config
                )
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


def run(benchmark_config: ThroughputBenchmarkConfig) -> None:
    """Run the throughput benchmark."""

    random.seed(benchmark_config.other.seed)

    pipeline_config = benchmark_config.pipeline
    if benchmark_config.devices is not None:
        device_handle = normalize_device_specs_input(benchmark_config.devices)
        pipeline_config.model.device_specs = (
            device_specs_from_normalized_device_handle(device_handle)
        )

    defer_resolve = os.getenv("MODULAR_PIPELINE_DEFER_RESOLVE", "").lower()
    if defer_resolve in {"1", "true", "yes"}:
        pipeline_config.resolve()

    tokenizer = AutoTokenizer.from_pretrained(
        benchmark_config.other.tokenizer or pipeline_config.model.model_path,
        model_max_length=benchmark_config.other.model_max_length,
        trust_remote_code=benchmark_config.other.trust_remote_code,
    )

    # TODO: benchmark_throughput.py should be refactored to use the BenchmarkDataset class.
    # and not use its own fetch_dataset_from_hf() here.
    if benchmark_config.other.dataset_name:
        dataset_path = fetch_dataset_from_hf(
            benchmark_config.other.dataset_name
        )
    elif benchmark_config.other.dataset_path:
        dataset_path = benchmark_config.other.dataset_path
    else:
        dataset_path = None

    # Sample the requests.
    if dataset_path:
        optional_kwargs = {}
        optional_kwargs["fixed_output_len"] = benchmark_config.output_len

        # TODO: benchmark_throughput.py should be refactored to use the BenchmarkDataset class.
        # Some of the fetch_dataset_from_hf() logic have different filenames
        # than the ones defined in benchmark_shared.datasets. These should be reconciled.
        if benchmark_config.other.dataset_name == "code_debug":
            benchmark_dataset = BenchmarkDataset.from_flags(
                dataset_name=benchmark_config.other.dataset_name,
                dataset_path=dataset_path,
            )
            assert isinstance(benchmark_dataset, CodeDebugBenchmarkDataset), (
                "code_debug dataset must be a CodeDebugBenchmarkDataset"
            )

            # code_debug is a long-context dataset based on InfiniteBench
            def sample_requests_func(
                dataset_path: str,
                num_requests: int,
                tokenizer: PreTrainedTokenizerBase,
                **kwargs,
            ) -> list[RequestPayload]:
                # CodeDebugBenchmarkDataset.sample_requests doesn't take dataset_path
                # because it already knows its dataset path
                sampled = benchmark_dataset.sample_requests(
                    num_requests=num_requests, tokenizer=tokenizer, **kwargs
                )
                converted = []
                for request in sampled.requests:
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
            optional_kwargs["max_length"] = pipeline_config.max_length

        requests = sample_requests_func(
            dataset_path=dataset_path,
            num_requests=benchmark_config.other.num_prompts,
            tokenizer=tokenizer,
            **optional_kwargs,
        )
    else:
        # Synthesize a prompt with the given input length.
        assert benchmark_config.input_len is not None
        assert benchmark_config.output_len is not None
        prompt = "hi" * (benchmark_config.input_len - 1)
        requests = [
            RequestPayload(
                prompt=prompt,
                prompt_len=benchmark_config.input_len,
                output_len=benchmark_config.output_len,
                image=None,
            )
            for _ in range(benchmark_config.other.num_prompts)
        ]

    if benchmark_config.backend == "modular":
        print("\nLoading...")
        model_tokenizer, model_factory = PIPELINE_REGISTRY.retrieve_factory(
            pipeline_config, task=benchmark_config.pipeline_task
        )
        print(f"INFO: MODEL config = {pipeline_config}")

        assert isinstance(model_tokenizer, TextTokenizer)
        print("\nExecuting...")
        if benchmark_config.async_engine:
            elapsed_time, generated_tokens_len = asyncio.run(
                run_max_async(
                    model_name=pipeline_config.model.model_name,
                    requests=requests,
                    config=pipeline_config,
                    model_factory=model_factory,
                    tokenizer=model_tokenizer,
                    show_text=benchmark_config.show_text,
                    pipeline_task=benchmark_config.pipeline_task,
                    top_k=benchmark_config.sampling.top_k,
                )
            )
        else:
            raise ValueError("Non-async LLM Engine not supported yet")
    else:
        raise ValueError(f"Unknown backend: {benchmark_config.backend}")

    total_num_input_tokens = sum(request.prompt_len for request in requests)
    total_num_output_tokens = sum(generated_tokens_len)

    if (
        benchmark_config.show_text
        and benchmark_config.pipeline_task == PipelineTask.TEXT_GENERATION
    ):
        print("\nPrompt Size [Input, Output_Real(Output_Expected)]:")
        for i, request in enumerate(requests):
            prompt_len = request.prompt_len
            output_len = request.output_len
            output_real = generated_tokens_len[i]
            print(
                f"task#{i}: [{prompt_len}, {output_real}({output_len})]", end=""
            )
            if (
                pipeline_config.max_length is not None
                and output_real + prompt_len >= pipeline_config.max_length
            ):
                print(
                    (
                        "  # [WARNING] limited by maximum sequence length"
                        f" ({pipeline_config.max_length}) from the pipeline config."
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
    if benchmark_config.output_json:
        output_filename = benchmark_config.output_json
        if not output_filename.endswith(".json"):
            output_filename += ".json"
        print(
            f"INFO: Write result to {output_filename}...",
            end="",
        )
        with open(output_filename, "w") as f:
            json.dump(results, f, indent=4)
        print("DONE")


def main() -> None:
    """Main entry point for the pipeline latency benchmark."""
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    os.environ.setdefault("MODULAR_PIPELINE_DEFER_RESOLVE", "1")

    # Create cyclopts app with environment variable config source
    # This must be set early so that --help can display MODULAR_ env vars
    app = App(
        name="benchmark_throughput",
        help=DESCRIPTION,
        help_formatter="plain",
        config=[
            # Load from environment variables with MODULAR_ prefix
            # Environment variable names follow pattern: MODULAR_<PARAM_NAME>
            Env(prefix="MODULAR_"),
        ],
    )

    # Define benchmark command with cyclopts
    @app.default
    def benchmark(
        benchmark_config: ThroughputBenchmarkConfig = ThroughputBenchmarkConfig(),
    ) -> None:
        """Run the pipeline latency benchmark."""
        # Validate that model is provided (required for benchmark).
        if not benchmark_config.pipeline.model.model_path:
            raise ValueError(
                "model is required. Please provide --pipeline.model.model_path argument, set it in the config file, "
                "or set MODULAR_PIPELINE_MODEL_MODEL_PATH environment variable."
            )

        if benchmark_config.other.tokenizer is None:
            benchmark_config.other.tokenizer = (
                benchmark_config.pipeline.model.model_path
            )

        # Validate cache strategy
        if (
            benchmark_config.enable_prefix_caching
            and benchmark_config.cache_strategy != KVCacheStrategy.PAGED
        ):
            raise ValueError(
                "prefix caching is only supported with paged attention"
            )

        if (
            benchmark_config.other.dataset_name is None
            and benchmark_config.other.dataset_path is None
        ):
            if benchmark_config.input_len is None:
                raise ValueError("Unknown input length to synthetic prompts")
            if benchmark_config.output_len is None:
                raise ValueError("Unknown output length for each request")
        else:
            if benchmark_config.input_len is not None:
                raise ValueError(
                    "Unable to set input length. The input length will be derived"
                    " from the dataset"
                )

        if __debug__:
            print(f"\nINFO: benchmark_config = {benchmark_config}")

        run(benchmark_config=benchmark_config)

    app()


if __name__ == "__main__":
    main()
