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

"""Benchmark online serving throughput."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import itertools
import json
import logging
import math
import os
import random
import resource
import statistics
import sys
import time
import traceback
import warnings
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import aiohttp
import numpy as np
import yaml
from safetensors.numpy import save_file
from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

if TYPE_CHECKING:
    from max.diagnostics.gpu import BackgroundRecorder as GPUBackgroundRecorder
    from max.diagnostics.gpu import GPUStats

try:
    from .benchmark_shared.config import (  # type: ignore[import-not-found, unused-ignore, no-redef]
        ServingBenchmarkConfig,
        parse_benchmark_args,
    )
    from .benchmark_shared.cpu_metrics import (  # type: ignore[import-not-found, unused-ignore, no-redef]
        CpuMetricsCollector,
        collect_pids_for_port,
    )
    from .benchmark_shared.datasets import (  # type: ignore[import-not-found, unused-ignore, no-redef]
        ArxivSummarizationBenchmarkDataset,
        AxolotlBenchmarkDataset,
        BatchJobBenchmarkDataset,
        BenchmarkDataset,
        ChatSession,
        CodeDebugBenchmarkDataset,
        ObfuscatedConversationsBenchmarkDataset,
        OpenAIImage,
        RandomBenchmarkDataset,
        SampledRequest,
        ShareGPTBenchmarkDataset,
        SonnetBenchmarkDataset,
        VisionArenaBenchmarkDataset,
    )
    from .benchmark_shared.metrics import (  # type: ignore[import-not-found, unused-ignore, no-redef]
        BenchmarkMetrics,
        LoRAMetrics,
        StandardPercentileMetrics,
        ThroughputMetrics,
    )
except ImportError:
    from benchmark_shared.config import (  # type: ignore[import-not-found, unused-ignore, no-redef]
        ServingBenchmarkConfig,
        parse_benchmark_args,
    )
    from benchmark_shared.cpu_metrics import (  # type: ignore[import-not-found, unused-ignore, no-redef]
        CpuMetricsCollector,
        collect_pids_for_port,
    )
    from benchmark_shared.datasets import (  # type: ignore[import-not-found, unused-ignore, no-redef]
        ArxivSummarizationBenchmarkDataset,
        AxolotlBenchmarkDataset,
        BatchJobBenchmarkDataset,
        BenchmarkDataset,
        ChatSession,
        CodeDebugBenchmarkDataset,
        ObfuscatedConversationsBenchmarkDataset,
        OpenAIImage,
        RandomBenchmarkDataset,
        SampledRequest,
        ShareGPTBenchmarkDataset,
        SonnetBenchmarkDataset,
        VisionArenaBenchmarkDataset,
    )
    from benchmark_shared.metrics import (  # type: ignore[import-not-found, unused-ignore, no-redef]
        BenchmarkMetrics,
        LoRAMetrics,
        StandardPercentileMetrics,
        ThroughputMetrics,
    )


# 30 minute timeout per request session
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=30 * 60)

BENCHMARK_SERVING_ARGPARSER_DESCRIPTION = (
    "This command runs comprehensive benchmark tests on a model server to"
    " measure performance metrics including throughput, latency, and resource"
    " utilization. Make sure that the MAX server is running and hosting a model"
    " before running this command."
)

logger = logging.getLogger("benchmark_serving")


@dataclass
class RequestFuncInput:
    prompt: str | list[dict[str, Any]]
    images: list[OpenAIImage]
    api_url: str
    prompt_len: int
    max_tokens: int | None
    ignore_eos: bool
    model: str
    session_id: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None


@dataclass
class RequestFuncOutput:
    cancelled: bool = False
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    # List of inter-token latencies
    itl: list[float] = field(default_factory=list)
    prompt_len: int = 0
    error: str = ""


class RequestCounter:
    def __init__(
        self,
        max_requests: int,
        req_counter_lock: asyncio.locks.Lock,
        total_sent_requests: int = 0,
    ) -> None:
        self.max_requests = max_requests
        self.req_counter_lock = req_counter_lock
        self.total_sent_requests = total_sent_requests

    async def advance_until_max(self) -> bool:
        """
        Checks if the number of sent requests has reached max_requests.
        If not, increment by one.

        Returns:
        bool: True if the request hasn't reached max and can advance, otherwise False.
        """
        async with self.req_counter_lock:
            if self.total_sent_requests >= self.max_requests:
                logger.warning(
                    f"Ending run: max requests {self.max_requests} have been"
                    " sent"
                )
                return False

            self.total_sent_requests += 1
            return True


def min_ignore_none(x: Sequence[int | None]) -> int | None:
    filtered = [elem for elem in x if elem is not None]
    return min(filtered, default=None)


def compute_output_len(
    tokenizer: PreTrainedTokenizerBase, output: RequestFuncOutput
) -> int:
    return len(
        tokenizer(
            output.generated_text,
            add_special_tokens=False,
        ).input_ids
    )


async def async_request_trt_llm(
    request_func_input: RequestFuncInput, pbar: tqdm | None = None
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input.prompt,
            "temperature": request_func_input.temperature,
            "top_p": request_func_input.top_p,
            "ignore_eos": request_func_input.ignore_eos,
            "stream": True,
        }

        if request_func_input.max_tokens is not None:
            payload["max_tokens"] = request_func_input.max_tokens
        if request_func_input.top_k is not None:
            payload["top_k"] = request_func_input.top_k

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(
                            chunk_bytes.decode("utf-8"), "data:"
                        )

                        data = json.loads(chunk)
                        output.generated_text += data["text_output"]
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True

                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_openai_completions(
    request_func_input: RequestFuncInput, pbar: tqdm | None = None
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(("completions", "profile")), (
        "OpenAI Completions API URL must end with 'completions' or 'profile'."
    )

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": request_func_input.temperature,
            "top_p": request_func_input.top_p,
            "best_of": 1,
            "stream": True,
            "ignore_eos": request_func_input.ignore_eos,
        }

        if request_func_input.max_tokens is not None:
            payload["max_tokens"] = request_func_input.max_tokens

        if request_func_input.top_k is not None:
            payload["top_k"] = request_func_input.top_k

        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        has_content = False
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(
                            chunk_bytes.decode("utf-8"), "data: "
                        )
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["choices"][0]["text"]:
                                has_content = True

                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(
                                        timestamp - most_recent_timestamp
                                    )

                                most_recent_timestamp = timestamp
                                generated_text += data["choices"][0]["text"]
                    if not has_content:
                        output.error = (
                            "No content returned, there could be an issue with"
                            " accuracy"
                        )
                        output.success = False
                    else:
                        output.generated_text = generated_text
                        output.success = True
                        output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False

        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput, pbar: tqdm | None = None
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("chat/completions"), (
        "OpenAI Chat Completions API URL must end with 'chat/completions'."
    )

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        if isinstance(request_func_input.prompt, str):  # question only
            content = [{"type": "text", "text": request_func_input.prompt}]
            messages_data = [
                {"role": "user", "content": content},
            ]
        else:  # conversation
            messages_data = request_func_input.prompt

        payload = {
            "model": request_func_input.model,
            "messages": messages_data,
            "temperature": request_func_input.temperature,
            "top_p": request_func_input.top_p,
            "stream": True,
            "ignore_eos": request_func_input.ignore_eos,
        }

        if request_func_input.max_tokens is not None:
            payload["max_tokens"] = request_func_input.max_tokens

        if request_func_input.top_k is not None:
            payload["top_k"] = request_func_input.top_k

        for img in request_func_input.images:
            # TODO: Remove this type ignore
            # (error: Value of type "object" is not indexable)
            payload["messages"][0]["content"].append(img)  # type: ignore[index]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }
        if request_func_input.session_id:
            headers["X-Session-ID"] = request_func_input.session_id

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        has_content = False
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(
                            chunk_bytes.decode("utf-8"), "data: "
                        )
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            pass
                        else:
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            delta = data["choices"][0]["delta"]
                            if delta.get("content", None):
                                has_content = True

                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(
                                        timestamp - most_recent_timestamp
                                    )

                                generated_text += delta["content"]

                            most_recent_timestamp = timestamp

                    if not has_content:
                        output.error = (
                            "No content returned, there could be an issue with"
                            " accuracy"
                        )
                        output.success = False
                    else:
                        output.generated_text = generated_text
                        output.success = True
                        output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False

        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


def generate_lora_adapter(
    base_model_id: str,
    output_dir: str,
    target_modules: list[str],
    lora_rank: int,
    adapter_name: str,
) -> None:
    """Generate a minimal LoRA adapter for testing.

    Args:
        base_model_id: HuggingFace model ID to generate adapter for
        output_dir: Directory to save the adapter files
        lora_rank: LoRA rank (r parameter)
        lora_alpha: LoRA alpha parameter for scaling
        target_modules: List of module names to apply LoRA to (q, k, v, o)
        adapter_name: Name for the adapter (used in metadata)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load base model config to get dimensions
    config = AutoConfig.from_pretrained(base_model_id)

    # Create adapter config
    adapter_config: dict[str, Any] = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": base_model_id,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": lora_rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }

    # Save adapter config
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    # Generate minimal LoRA weights
    lora_weights = {}

    # For each layer and target module, create LoRA A and B matrices
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size

    # Handle grouped query attention - k_proj and v_proj have different dimensions
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads

    # Determine output dimensions for attention projection modules
    attn_module_dims = {
        "q_proj": hidden_size,  # hidden_size -> hidden_size
        "k_proj": num_kv_heads * head_dim,  # hidden_size -> kv_hidden_size
        "v_proj": num_kv_heads * head_dim,  # hidden_size -> kv_hidden_size
        "o_proj": hidden_size,  # hidden_size -> hidden_size
    }

    for layer_idx in range(num_layers):
        for module in target_modules:
            # Validate that module is supported (attention only for now)
            if module not in attn_module_dims:
                raise ValueError(
                    f"Unsupported target module '{module}'. Only attention"
                    " modules are currently supported:"
                    f" {list(attn_module_dims.keys())}"
                )

            # Get the output dimension for this module type
            out_dim = attn_module_dims[module]
            in_dim = (
                hidden_size  # Input is always hidden_size for attention modules
            )

            # LoRA A: shape should be (lora_rank, in_dim)
            lora_a_key = f"base_model.model.layers.{layer_idx}.self_attn.{module}.lora_A.weight"
            lora_weights[lora_a_key] = (
                np.random.randn(lora_rank, in_dim).astype(np.float32) * 0.01
            )

            # LoRA B: shape should be (out_dim, lora_rank)
            lora_b_key = f"base_model.model.layers.{layer_idx}.self_attn.{module}.lora_B.weight"
            lora_weights[lora_b_key] = np.zeros(
                (out_dim, lora_rank), dtype=np.float32
            )

    # Save weights in safetensors format
    save_file(
        lora_weights, os.path.join(output_dir, "adapter_model.safetensors")
    )


def generate_loras(
    model_id: str,
    lora_rank: int,
    num_loras: int,
    lora_target_modules: list[str],
    lora_paths: list[str],
    lora_output_dir: str,
    lora_server_path: str,
) -> dict[str, str]:
    # Generate test LoRA adapters if needed
    lora_configs: dict[str, str] = {}

    if lora_paths:
        # Use provided LoRA paths
        logger.info("Using provided LoRA paths")
        for i, path in enumerate(lora_paths):
            abs_path = os.path.abspath(path)
            lora_configs[f"adapter_{i}"] = abs_path
    else:
        # Generate test LoRA adapters
        logger.info(f"Preparing {num_loras} test LoRA adapters...")

        # Use custom output directory if specified, otherwise use temp directory
        base_output_dir = os.path.abspath(os.path.expanduser(lora_output_dir))
        os.makedirs(base_output_dir, exist_ok=True)

        for i in range(num_loras):
            adapter_name = f"generated_adapter_{i}"
            adapter_path = os.path.join(base_output_dir, adapter_name)

            generate_lora_adapter(
                base_model_id=model_id,
                output_dir=adapter_path,
                target_modules=lora_target_modules,
                lora_rank=lora_rank,
                adapter_name=adapter_name,
            )

            # Use server path if specified, otherwise use absolute local path
            if lora_server_path:
                relative_path = os.path.relpath(adapter_path, base_output_dir)
                server_path = os.path.join(lora_server_path, relative_path)
            else:
                server_path = os.path.abspath(adapter_path)

            lora_configs[adapter_name] = server_path
    return lora_configs


async def async_request_lora_load(
    api_url: str, lora_name: str, lora_path: str
) -> tuple[bool, float]:
    """Load a LoRA adapter via the API.

    Returns:
        Tuple of (success, load_time_ms)
    """
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {"lora_name": lora_name, "lora_path": lora_path}
        headers = {"Content-Type": "application/json"}
        logger.debug(f"Loading LoRA '{lora_name}' from path: {lora_path}")

        start_time = time.perf_counter()
        try:
            async with session.post(
                url=f"{api_url}/v1/load_lora_adapter",
                json=payload,
                headers=headers,
            ) as response:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                if response.status == 200:
                    logger.debug(
                        f"Successfully loaded LoRA '{lora_name}' in"
                        f" {elapsed_ms:.2f}ms"
                    )
                    return True, elapsed_ms
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to load LoRA '{lora_name}': {error_text}"
                    )
                    return False, elapsed_ms
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(f"Exception loading LoRA '{lora_name}'")
            return False, elapsed_ms


async def async_request_lora_unload(
    api_url: str, lora_name: str
) -> tuple[bool, float]:
    """Unload a LoRA adapter via the API.

    Returns:
        Tuple of (success, unload_time_ms)
    """
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {"lora_name": lora_name}
        headers = {"Content-Type": "application/json"}

        start_time = time.perf_counter()
        try:
            async with session.post(
                url=f"{api_url}/v1/unload_lora_adapter",
                json=payload,
                headers=headers,
            ) as response:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                if response.status == 200:
                    logger.debug(
                        f"Successfully unloaded LoRA '{lora_name}' in"
                        f" {elapsed_ms:.2f}ms"
                    )
                    return True, elapsed_ms
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to unload LoRA '{lora_name}': {error_text}"
                    )
                    return False, elapsed_ms
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(f"Exception loading LoRA '{lora_name}'")
            return False, elapsed_ms


async def benchmark_lora_unloading(
    api_url: str,
    lora_configs: dict[str, str],
    metrics: LoRAMetrics,
    max_concurrent: int = 1,
) -> None:
    """Benchmark LoRA unloading performance.

    Args:
        api_url: Base API URL
        lora_names: List of LoRA adapter names to unload
        max_concurrent_unloads: Maximum concurrent unloading operations

    Returns:
        LoRAOperationMetrics with unloading performance data
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def unload_with_semaphore(name: str) -> None:
        async with semaphore:
            success, unload_time = await async_request_lora_unload(
                api_url, name
            )
            if success:
                metrics.unload_times_ms.append(unload_time)
                metrics.total_unloads += 1
            else:
                logger.warning(f"Failed to unload LoRA '{name}'")

    tasks = [unload_with_semaphore(name) for name in lora_configs]
    await tqdm.gather(*tasks, desc="Unloading LoRAs...")


async def benchmark_lora_loading(
    api_url: str,
    lora_configs: dict[str, str],  # List of (name, path) tuples
    metrics: LoRAMetrics,
    max_concurrent: int = 1,
) -> None:
    """Benchmark LoRA loading performance.

    Args:
        api_url: Base API URL
        lora_configs: List of (name, path) tuples for LoRA adapters
        max_concurrent_loads: Maximum concurrent loading operations
        LoRAMetrics with loading performance data
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def load_with_semaphore(name: str, path: str) -> None:
        async with semaphore:
            success, load_time = await async_request_lora_load(
                api_url, name, path
            )
            if success:
                metrics.load_times_ms.append(load_time)
                metrics.total_loads += 1
            else:
                logger.warning(f"Failed to load LoRA '{name}'")

    tasks = [
        load_with_semaphore(name, path) for name, path in lora_configs.items()
    ]
    await tqdm.gather(*tasks, desc="Loading LoRAs...")


# Since vllm must support Python 3.8, we can't use str.removeprefix(prefix)
# introduced in Python 3.9
def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def get_tokenizer(
    pretrained_model_name_or_path: str,
    model_max_length: int | None,
    trust_remote_code: bool,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        model_max_length=model_max_length,
        trust_remote_code=trust_remote_code,
    )


# TODO: The keys here should match the backend enum in benchmark_config.py
ASYNC_REQUEST_FUNCS = {
    "vllm": async_request_openai_completions,
    "vllm-chat": async_request_openai_chat_completions,
    "trt-llm": async_request_trt_llm,
    "modular": async_request_openai_completions,
    "modular-chat": async_request_openai_chat_completions,
    "sglang": async_request_openai_completions,
    "sglang-chat": async_request_openai_chat_completions,
}


# from https://github.com/sgl-project/sglang/blob/v0.4.0/python/sglang/bench_serving.py#L1283
def set_ulimit(target_soft_limit: int = 65535) -> None:
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")


async def get_request(
    input_requests: Sequence[SampledRequest],
    request_rate: float,
    timing_data: dict[str, list[float]],
    burstiness: float = 1.0,
) -> AsyncGenerator[SampledRequest, None]:
    """
    Asynchronously generates requests at a specified rate
    with OPTIONAL burstiness.

    Args:
        input_requests:
            A list of input requests, each represented as a SampledRequest.
        request_rate:
            The rate at which requests are generated (requests/s).
        burstiness (optional):
            The burstiness factor of the request generation.
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests, while a higher burstiness value
            (burstiness > 1) results in a more uniform arrival of requests.
        timing_data:
            Dictionary where timing data will be collected with keys:
            - 'intervals': List of actual time intervals between requests
    """

    # Calculate scale parameter theta to maintain the desired request_rate.
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}."
    )
    theta = 1.0 / (request_rate * burstiness)

    # Initialize timing data collection - always enabled
    if timing_data is None:
        timing_data = {}
    timing_data.setdefault("intervals", [])

    start_time = time.perf_counter()
    last_request_time = start_time

    for request in input_requests:
        current_time = time.perf_counter()

        # Record timestamp when request is yielded
        if last_request_time != start_time:
            actual_interval = current_time - last_request_time
            timing_data["intervals"].append(actual_interval)

        yield request

        # Update last_request_time for next iteration
        last_request_time = current_time

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the gamma distribution.
        # If burstiness is 1, it follows exponential distribution.
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def print_section(title: str, char: str = "-") -> None:
    """Helper function to print a section with formatted header."""
    print("{s:{c}^{n}}".format(s=title, n=50, c=char))


def calculate_metrics(
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    gpu_metrics: list[dict[str, GPUStats]] | None,
    cpu_metrics: dict[str, Any],
    skip_first_n_requests: int,
    max_concurrency: int | None,
    collect_gpu_stats: bool,
) -> tuple[BenchmarkMetrics, list[int]]:
    actual_output_lens: list[int] = []
    nonempty_response_chunks = 0
    total_input = 0
    completed = 0
    max_input = 0
    max_output = 0
    max_total = 0
    failures = 0
    failed_responses = []
    itls: list[float] = []
    tpots: list[float] = []
    ttfts: list[float] = []
    latencies: list[float] = []
    input_throughputs: list[float] = []
    output_throughputs: list[float] = []
    for i in range(len(outputs)):
        # If the request was cancelled due to max_benchmark_duration_s, we skip it
        # and don't count it towards the metrics
        if outputs[i].cancelled:
            continue
        if outputs[i].success:
            completed += 1
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note : this may inflate the output token count slightly
            total_input += outputs[i].prompt_len
            output_len = compute_output_len(tokenizer, outputs[i])
            actual_output_lens.append(output_len)
            nonempty_response_chunks += 1 if outputs[i].ttft != 0 else 0
            nonempty_response_chunks += len(outputs[i].itl)

            max_input = max(max_input, outputs[i].prompt_len)
            max_output = max(max_output, output_len)
            max_total = max(max_total, outputs[i].prompt_len + output_len)

            # We only skip these requests for client experience metrics like
            # TTFT, ITL, TPOT, E2E. They are still considered for overall token
            # counts and throughputs.
            if i < skip_first_n_requests:
                continue

            if output_len > 1:
                tpots.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1)
                )
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            # Input throughput is fully calculated once we reach the first output token.
            input_throughputs.append(outputs[i].prompt_len / outputs[i].ttft)
            # output throughput ignores the first token.
            # It is just timing for the chain of output tokens.
            output_throughputs.append(
                (output_len - 1) / (outputs[i].latency - outputs[i].ttft)
            )
            latencies.append(outputs[i].latency)
        else:
            actual_output_lens.append(0)
            failures = failures + 1
            failed_responses.append(outputs[i])

    if len(outputs) == 0:
        warnings.warn(
            "No responses were received from the server.", stacklevel=2
        )

    if failures != 0:
        warnings.warn(
            (
                "Some requests failed. The responses returned are displayed "
                "below. Please check server logs for more information."
            ),
            stacklevel=2,
        )
        for f in failed_responses:
            logger.error(f"Failed :: {f}")

    if completed == 0:
        warnings.warn(
            (
                "All requests failed. This is likely due to a misconfiguration "
                "on the benchmark arguments."
            ),
            stacklevel=2,
        )

    peak_gpu_memory_mib = []
    available_gpu_memory_mib = []
    gpu_utilization = []
    if collect_gpu_stats and gpu_metrics:
        # Simplification: We assume that whatever devices are available at the
        # start of benchmarking stays the same throughout the run.  If someone
        # is hotplugging GPUs during a benchmark this may not be true, but that
        # doesn't seem likely.
        all_devices = list(gpu_metrics[0].keys())
        if not all_devices:
            logger.warning("No GPUs found, so there are no GPU stats to report")

        BYTES_PER_MIB = 1024 * 1024
        for device_name in all_devices:
            peak_gpu_memory_mib.append(
                max(
                    snapshot[device_name].memory.used_bytes
                    for snapshot in gpu_metrics
                )
                / BYTES_PER_MIB
            )
            available_gpu_memory_mib.append(
                min(
                    snapshot[device_name].memory.free_bytes
                    for snapshot in gpu_metrics
                )
                / BYTES_PER_MIB
            )
            gpu_utilization.append(
                statistics.mean(
                    snapshot[device_name].utilization.gpu_usage_percent
                    for snapshot in gpu_metrics
                )
            )

    metrics = BenchmarkMetrics(
        completed=completed,
        failures=failures,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        nonempty_response_chunks=nonempty_response_chunks,
        max_concurrency=max_concurrency or len(outputs),
        request_throughput=completed / dur_s,
        # Use specialized metric classes that handle percentile calculations automatically
        input_throughput=ThroughputMetrics(
            input_throughputs or [float("nan")], unit="tok/s"
        ),
        output_throughput=ThroughputMetrics(
            output_throughputs or [float("nan")], unit="tok/s"
        ),
        ttft_ms=StandardPercentileMetrics(
            ttfts or [float("nan")], scale_factor=1000.0, unit="ms"
        ),
        tpot_ms=StandardPercentileMetrics(
            tpots or [float("nan")], scale_factor=1000.0, unit="ms"
        ),
        itl_ms=StandardPercentileMetrics(
            itls or [float("nan")], scale_factor=1000.0, unit="ms"
        ),
        latency_ms=StandardPercentileMetrics(
            latencies or [float("nan")], scale_factor=1000.0, unit="ms"
        ),
        max_input=max_input,
        max_output=max_output,
        max_total=max_total,
        peak_gpu_memory_mib=peak_gpu_memory_mib,
        available_gpu_memory_mib=available_gpu_memory_mib,
        gpu_utilization=gpu_utilization,
        cpu_utilization_user=cpu_metrics.get("user_percent"),
        cpu_utilization_system=cpu_metrics.get("system_percent"),
    )

    return metrics, actual_output_lens


async def chat_session_driver(
    model_id: str,
    api_url: str,
    request_func: Callable[[RequestFuncInput], Awaitable[RequestFuncOutput]],
    request_counter: RequestCounter,
    chat_session: ChatSession,
    max_chat_len: int,
    delay_between_chat_turns: int | None,
    skip_session_count: int | None = None,
    ignore_first_turn_stats: bool = False,
) -> list[RequestFuncOutput]:
    request_func_input = RequestFuncInput(
        model=model_id,
        prompt=[],
        api_url=api_url,
        prompt_len=0,
        max_tokens=0,
        ignore_eos=True,
        images=[],
        session_id=str(chat_session.id),
    )
    content_idx = 0  # Assume user initiates the conversation

    session_outputs = []
    message_history: list[dict[str, Any]] = []
    chat_len = 0

    messages = chat_session.messages
    while content_idx + 1 < len(messages):
        chat_len += messages[content_idx].num_tokens
        output_len = messages[content_idx + 1].num_tokens
        if chat_len + output_len > max_chat_len:
            logger.warning(
                f"Ending conversation: hitting max chat length {max_chat_len}"
            )
            break

        advance_request = await request_counter.advance_until_max()
        if not advance_request:  # reached max_requests
            break

        user_prompt = messages[content_idx].content
        message_history.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            }
        )
        request_func_input.prompt = message_history
        request_func_input.prompt_len = chat_len
        request_func_input.max_tokens = output_len
        response = await request_func(request_func_input)
        if (
            skip_session_count is None
            or chat_session.id is None
            or chat_session.id >= skip_session_count
        ) and not (ignore_first_turn_stats and content_idx == 0):
            session_outputs.append(response)

        if not response.success:
            if not response.cancelled:
                logger.error(
                    f"Ending chat session {chat_session.id} due to server error"
                    f" response: {response.error}"
                )
            break

        content_idx += 2
        message_history.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response.generated_text}],
            }
        )
        chat_len += output_len

        if delay_between_chat_turns:
            # todo parameterize the distribution and scale
            # e.g. N(mean, std) or U(lower, upper)
            delay_ms = np.random.normal(
                loc=delay_between_chat_turns,
                scale=delay_between_chat_turns * 0.5,
            )
            await asyncio.sleep(delay_ms / 1000)

    return session_outputs


async def benchmark(  # noqa: ANN201
    backend: str,
    chat: bool,
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: Sequence[SampledRequest],
    chat_sessions: Sequence[ChatSession],
    request_rate: float,
    burstiness: float,
    max_concurrency: int | None,
    disable_tqdm: bool,
    do_test_prompt: bool,
    collect_gpu_stats: bool,
    collect_cpu_stats: bool,
    print_inputs_and_outputs: bool,
    max_requests: int,
    num_chat_sessions: int | None,
    delay_between_chat_turns: int | None,
    skip_first_n_requests: int,
    max_output_len: int | None,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_benchmark_duration_s: int | None,
    warmup_delay_ms: float = 0,
    ignore_first_turn_stats: bool = False,
    timing_data: dict[str, list[float]] | None = None,
    lora_request_ratio: float = 0.0,
    lora_configs: dict[str, str] | None = None,
    max_concurrent_lora_ops: int = 1,
):
    if ignore_first_turn_stats and skip_first_n_requests:
        logger.warning(
            "--ignore-first-turn-stats and --skip-first-n-requests both set."
            " Ignoring --skip-first-n-requests due to first turn in each chat"
            " already being ignored."
        )
        skip_first_n_requests = 0

    # Initialize LoRA metrics
    lora_metrics = LoRAMetrics()

    # Benchmark LoRA loading if configs provided
    if lora_configs:
        logger.info("Starting LoRA loading benchmark...")
        await benchmark_lora_loading(
            api_url=base_url,
            lora_configs=lora_configs,
            metrics=lora_metrics,
            max_concurrent=max_concurrent_lora_ops,
        )

    # Handle backend construction: add "-chat" only if not already present
    # and if the resulting backend exists in ASYNC_REQUEST_FUNCS
    if chat and not backend.endswith("-chat"):
        potential_chat_backend = backend + "-chat"
        if potential_chat_backend in ASYNC_REQUEST_FUNCS:
            full_backend = potential_chat_backend
        else:
            # If chat variant doesn't exist, use the base backend
            full_backend = backend
    else:
        full_backend = backend

    if full_backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[full_backend]
    else:
        raise ValueError(f"Unknown backend: {full_backend}")

    if do_test_prompt:
        logger.info("Starting initial single prompt test run...")
        test_prompt: str | list[dict[str, Any]]
        if num_chat_sessions:
            test_question = chat_sessions[0].messages[0]
            test_answer = chat_sessions[0].messages[1]
            test_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": test_question.content}
                    ],
                }
            ]
            test_prompt_len = test_question.num_tokens
            test_max_tokens: int | None = test_answer.num_tokens
            test_ignore_eos = True
            test_images = []
        else:
            test_request = input_requests[0]
            test_prompt = test_request.prompt_formatted
            test_prompt_len = test_request.prompt_len
            test_max_tokens = min_ignore_none(
                (test_request.output_len, max_output_len)
            )
            test_ignore_eos = test_request.ignore_eos
            test_images = test_request.encoded_images

        test_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=api_url,
            prompt_len=test_prompt_len,
            max_tokens=test_max_tokens,
            ignore_eos=test_ignore_eos,
            images=test_images,
            top_k=top_k,
        )
        test_output = await request_func(
            request_func_input=test_input,
        )
        if not test_output.success:
            raise ValueError(
                "Initial test run failed - Please make sure benchmark"
                " arguments are correctly specified. Error:"
                f" {test_output.error}"
            )
        else:
            logger.info(
                "Initial test run completed. Starting main benchmark run..."
            )

    if burstiness == 1.0:
        distribution = "Poisson process"
    else:
        distribution = "Gamma distribution"

    logger.info(f"Input request rate: {request_rate}")
    logger.info(f"Burstiness factor: {burstiness} ({distribution})")
    logger.info(f"Maximum request concurrency: {max_concurrency}")

    # This can be used once the minimum Python version is 3.10 or higher,
    # and it will simplify the code in limited_request_func.
    #    semaphore = (asyncio.Semaphore(max_concurrency)
    #                 if max_concurrency else contextlib.nullcontext())
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    with contextlib.ExitStack() as benchmark_stack:
        gpu_recorder: GPUBackgroundRecorder | None = None
        if collect_gpu_stats:
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
        if collect_cpu_stats:
            try:
                pids = collect_pids_for_port(
                    int(urlparse(api_url).port or 8000)
                )
                cpu_collector = CpuMetricsCollector(pids)
                cpu_collector.start()
            except:
                logger.warning(
                    "Cannot access max-serve PIDs, skipping CPU stats"
                    " collection"
                )

        benchmark_start_time = time.perf_counter_ns()
        if max_benchmark_duration_s is None:
            benchmark_should_end_time = None
        else:
            benchmark_should_end_time = (
                benchmark_start_time + max_benchmark_duration_s * 1e9
            )
        tasks: list[asyncio.Task] = []  # type: ignore[type-arg, unused-ignore]
        outputs: list[RequestFuncOutput] = []
        if not num_chat_sessions:
            # single-turn chat scenario
            if timing_data is None:
                timing_data = {}
            pbar = None if disable_tqdm else tqdm(total=len(input_requests))

            async def limited_request_func(
                request_func_input: RequestFuncInput,
            ) -> RequestFuncOutput:
                if semaphore is None:
                    return await request_func(
                        request_func_input=request_func_input, pbar=pbar
                    )
                async with semaphore:
                    if benchmark_should_end_time is not None:
                        if time.perf_counter_ns() >= benchmark_should_end_time:
                            return RequestFuncOutput(cancelled=True)
                    return await request_func(
                        request_func_input=request_func_input, pbar=pbar
                    )

            async for request in get_request(
                input_requests, request_rate, timing_data, burstiness
            ):
                # If we've hit the time limit, then don't issue any more rquests
                if benchmark_should_end_time is not None:
                    if time.perf_counter_ns() >= benchmark_should_end_time:
                        break

                # Use the ignore_eos setting from the dataset.
                # Each dataset determines whether to respect EOS based on its own logic.
                ignore_eos = request.ignore_eos
                max_tokens = min_ignore_none(
                    (request.output_len, max_output_len)
                )

                lora_id = None
                if lora_configs and random.random() < lora_request_ratio:
                    lora_id = random.choice(list(lora_configs.keys()))

                request_func_input = RequestFuncInput(
                    model=model_id if lora_id is None else lora_id,
                    prompt=request.prompt_formatted,
                    api_url=api_url,
                    prompt_len=request.prompt_len,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    ignore_eos=ignore_eos,
                    images=request.encoded_images,
                )
                tasks.append(
                    asyncio.create_task(
                        limited_request_func(request_func_input)
                    )
                )
            outputs = await asyncio.gather(*tasks)

        else:
            # multi-turn chat scenario
            if disable_tqdm:
                pbar = None
            else:
                num_qa_turns = [
                    (len(session.messages) // 2) for session in chat_sessions
                ]
                pbar = tqdm(total=sum(num_qa_turns))

            # Track total sent requests among chat sessions
            request_counter = RequestCounter(
                max_requests=max_requests,
                req_counter_lock=asyncio.Lock(),
                total_sent_requests=0,
            )

            # Limit the request function only to deal with timeouts.
            async def limited_request_func(
                request_func_input: RequestFuncInput,
            ) -> RequestFuncOutput:
                if benchmark_should_end_time is not None:
                    if time.perf_counter_ns() >= benchmark_should_end_time:
                        return RequestFuncOutput(cancelled=True)
                return await request_func(
                    request_func_input=request_func_input, pbar=pbar
                )

            # apply the semaphore at the session level
            # ex: with max_concurrency = 1,
            # the first session finishes before the second session starts
            async def limited_chat_session_driver(
                chat_session: ChatSession,
            ) -> list[RequestFuncOutput]:
                lora_id = None
                if lora_configs and random.random() < lora_request_ratio:
                    lora_id = random.choice(list(lora_configs.keys()))

                if semaphore is None:
                    return await chat_session_driver(
                        model_id if lora_id is None else lora_id,
                        api_url,
                        limited_request_func,
                        request_counter,
                        chat_session,
                        tokenizer.model_max_length,
                        delay_between_chat_turns,
                        skip_first_n_requests,
                        ignore_first_turn_stats,
                    )
                async with semaphore:
                    return await chat_session_driver(
                        model_id if lora_id is None else lora_id,
                        api_url,
                        limited_request_func,
                        request_counter,
                        chat_session,
                        tokenizer.model_max_length,
                        delay_between_chat_turns,
                        skip_first_n_requests,
                        ignore_first_turn_stats,
                    )

            for idx, chat_session in enumerate(chat_sessions):
                if (
                    warmup_delay_ms > 0
                    and max_concurrency
                    and idx < max_concurrency
                ):
                    await asyncio.sleep(warmup_delay_ms / 1000)
                tasks.append(
                    asyncio.create_task(
                        limited_chat_session_driver(chat_session)
                    )
                )

            session_outputs = await asyncio.gather(*tasks)
            outputs = [
                output for sublist in session_outputs for output in sublist
            ]

        benchmark_duration = (
            time.perf_counter_ns() - benchmark_start_time
        ) / 1e9

        if pbar is not None:
            pbar.close()

    if print_inputs_and_outputs:
        print("Generated output text:")
        for req_id, output in enumerate(outputs):
            output_len = compute_output_len(tokenizer, output)
            print(
                {
                    "req_id": req_id,
                    "output_len": output_len,
                    "output": output.generated_text,
                }
            )

    if lora_configs:
        await benchmark_lora_unloading(
            api_url=base_url,
            lora_configs=lora_configs,
            metrics=lora_metrics,
            max_concurrent=max_concurrent_lora_ops,
        )

    gpu_metrics: list[dict[str, GPUStats]] | None = None
    if collect_gpu_stats and gpu_recorder is not None:
        gpu_metrics = gpu_recorder.stats

    if collect_cpu_stats and cpu_collector is not None:
        cpu_collector.stop()
        cpu_metrics = cpu_collector.dump_stats()
    else:
        cpu_metrics = {}

    metrics, actual_output_lens = calculate_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        gpu_metrics=gpu_metrics,
        cpu_metrics=cpu_metrics,
        skip_first_n_requests=skip_first_n_requests,
        max_concurrency=max_concurrency,
        collect_gpu_stats=collect_gpu_stats,
    )
    achieved_request_rate = 0.0
    if timing_data and timing_data.get("intervals"):
        mean_interval = sum(timing_data["intervals"]) / len(
            timing_data["intervals"]
        )
        achieved_request_rate = (
            round(1.0 / mean_interval, 3) if mean_interval > 0 else 0.0
        )

    print_section(title=" Serving Benchmark Result ", char="=")
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10}".format("Failed requests:", metrics.failures))
    print(
        "{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration)
    )
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print(
        "{:<40} {:<10}".format("Total generated tokens:", metrics.total_output)
    )
    # We found that response chunks can be empty in content and the token number
    # can be different with the re-tokenization in one pass or chunk-by-chunk.
    # Let's count the number of nonempty_response_chunks for all serving backends.
    # With the move to zero-overhead single step scheduling, this should generally
    # exactly match the number of requested output tokens.
    print(
        "{:<40} {:<10}".format(
            "Total nonempty serving response chunks:",
            metrics.nonempty_response_chunks,
        )
    )
    offline_benchmark = math.isinf(request_rate) and max_concurrency is None
    print(
        "{:<40} {:<10.5f}".format(
            "Input request rate (req/s):",
            float("inf") if offline_benchmark else achieved_request_rate,
        )
    )
    print(
        "{:<40} {:<10.5f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print_section(title="Client Experience Metrics")
    print("{:<40} {:<10}".format("Max Concurrency:", metrics.max_concurrency))
    print(
        metrics.input_throughput.format_with_prefix(
            prefix="input token throughput", unit="tok/s"
        )
    )
    print(
        metrics.output_throughput.format_with_prefix(
            prefix="output token throughput", unit="tok/s"
        )
    )
    print_section(title="Time to First Token")
    print(metrics.ttft_ms.format_with_prefix(prefix="TTFT", unit="ms"))
    print_section(title="Time per Output Token (excl. 1st token)")
    print(metrics.tpot_ms.format_with_prefix(prefix="TPOT", unit="ms"))
    print_section(title="Inter-token Latency")
    print(metrics.itl_ms.format_with_prefix(prefix="ITL", unit="ms"))
    print_section(title="Per-Request E2E Latency")
    print(
        metrics.latency_ms.format_with_prefix(
            prefix="Request Latency", unit="ms"
        )
    )
    print_section(title="Token Stats")
    print("{:<40} {:<10}".format("Max input tokens:", metrics.max_input))
    print("{:<40} {:<10}".format("Max output tokens:", metrics.max_output))
    print("{:<40} {:<10}".format("Max total tokens:", metrics.max_total))
    if collect_gpu_stats:
        for i in range(len(metrics.gpu_utilization)):
            print_section(title=f"GPU Stats {i}")
            print(
                "{:<40} {:<10.2f}".format(
                    "GPU Utilization (%):", metrics.gpu_utilization[i]
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Peak GPU Memory Used (MiB):",
                    metrics.peak_gpu_memory_mib[i],
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "GPU Memory Available (MiB):",
                    metrics.available_gpu_memory_mib[i],
                )
            )

    if collect_cpu_stats and metrics.cpu_utilization_user is not None:
        print_section(title="CPU Stats")
        print(
            "{:<40} {:<10.2f}".format(
                "CPU User Utilization (%):",
                metrics.cpu_utilization_user or 0.0,
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "CPU System Utilization (%):",
                metrics.cpu_utilization_system or 0.0,
            )
        )

    print("=" * 50)

    # Print LoRA benchmark results
    if lora_configs:
        print_section(title=" LoRA Adapter Benchmark Results ", char="=")
        print(
            "{:<40} {:<10}".format(
                "Total LoRA loads:", lora_metrics.total_loads
            )
        )
        print(
            "{:<40} {:<10}".format(
                "Total LoRA unloads:", lora_metrics.total_unloads
            )
        )

        if lora_metrics.load_times_ms:
            print_section(title="LoRA Load Times")
            print(
                "{:<40} {:<10.2f}".format(
                    "Mean load time:",
                    statistics.mean(lora_metrics.load_times_ms),
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Median load time:",
                    statistics.median(lora_metrics.load_times_ms),
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Min load time:", min(lora_metrics.load_times_ms)
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Max load time:", max(lora_metrics.load_times_ms)
                )
            )
            if len(lora_metrics.load_times_ms) > 1:
                print(
                    "{:<40} {:<10.2f}".format(
                        "Std dev load time:",
                        statistics.stdev(lora_metrics.load_times_ms),
                    )
                )

        if lora_metrics.unload_times_ms:
            print_section(title="LoRA Unload Times")
            print(
                "{:<40} {:<10.2f}".format(
                    "Mean unload time:",
                    statistics.mean(lora_metrics.unload_times_ms),
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Median unload time:",
                    statistics.median(lora_metrics.unload_times_ms),
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Min unload time:", min(lora_metrics.unload_times_ms)
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Max unload time:", max(lora_metrics.unload_times_ms)
                )
            )
            if len(lora_metrics.unload_times_ms) > 1:
                print(
                    "{:<40} {:<10.2f}".format(
                        "Std dev unload time:",
                        statistics.stdev(lora_metrics.unload_times_ms),
                    )
                )

        print("=" * 50)

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "failures": metrics.failures,
        "max_concurrency": metrics.max_concurrency,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "mean_input_throughput": metrics.input_throughput.mean,
        "std_input_throughput": metrics.input_throughput.std,
        "median_input_throughput": metrics.input_throughput.median,
        "p90_input_throughput": metrics.input_throughput.p90,
        "p95_input_throughput": metrics.input_throughput.p95,
        "p99_input_throughput": metrics.input_throughput.p99,
        "mean_output_throughput": metrics.output_throughput.mean,
        "std_output_throughput": metrics.output_throughput.std,
        "median_output_throughput": metrics.output_throughput.median,
        "p90_output_throughput": metrics.output_throughput.p90,
        "p95_output_throughput": metrics.output_throughput.p95,
        "p99_output_throughput": metrics.output_throughput.p99,
        "mean_ttft_ms": metrics.ttft_ms.mean,
        "median_ttft_ms": metrics.ttft_ms.median,
        "std_ttft_ms": metrics.ttft_ms.std,
        "p90_ttft_ms": metrics.ttft_ms.p90,
        "p95_ttft_ms": metrics.ttft_ms.p95,
        "p99_ttft_ms": metrics.ttft_ms.p99,
        "mean_tpot_ms": metrics.tpot_ms.mean,
        "median_tpot_ms": metrics.tpot_ms.median,
        "std_tpot_ms": metrics.tpot_ms.std,
        "p90_tpot_ms": metrics.tpot_ms.p90,
        "p95_tpot_ms": metrics.tpot_ms.p95,
        "p99_tpot_ms": metrics.tpot_ms.p99,
        "mean_itl_ms": metrics.itl_ms.mean,
        "median_itl_ms": metrics.itl_ms.median,
        "std_itl_ms": metrics.itl_ms.std,
        "p90_itl_ms": metrics.itl_ms.p90,
        "p95_itl_ms": metrics.itl_ms.p95,
        "p99_itl_ms": metrics.itl_ms.p99,
        "mean_latency_ms": metrics.latency_ms.mean,
        "median_latency_ms": metrics.latency_ms.median,
        "std_latency_ms": metrics.latency_ms.std,
        "p90_latency_ms": metrics.latency_ms.p90,
        "p95_latency_ms": metrics.latency_ms.p95,
        "p99_latency_ms": metrics.latency_ms.p99,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        "peak_gpu_memory_mib": metrics.peak_gpu_memory_mib,
        "available_gpu_memory_mib": metrics.available_gpu_memory_mib,
        "gpu_utilization": metrics.gpu_utilization,
    }

    # Add LoRA metrics to result if available
    if lora_configs:
        result["lora_metrics"] = {
            "total_loads": lora_metrics.total_loads,
            "total_unloads": lora_metrics.total_unloads,
            "load_times_ms": lora_metrics.load_times_ms,
            "unload_times_ms": lora_metrics.unload_times_ms,
        }

    return result


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)s: %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    logger.info(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # benchmarks can create a large number of concurrent in-flight requests
    # so bump the file limit to make room for them
    set_ulimit()

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.endpoint not in [
        "/v1/completions",
        "/v1/chat/completions",
        "/v2/models/ensemble/generate_stream",
    ]:
        raise ValueError(f"Unknown endpoint: {args.endpoint}")
    chat = args.endpoint == "/v1/chat/completions"

    if args.base_url is not None:
        base_url = args.base_url
    else:
        base_url = f"http://{args.host}:{args.port}"

    api_url = f"{base_url}{args.endpoint}"

    logger.info(f"getting tokenizer. api url: {api_url}")
    tokenizer = get_tokenizer(
        tokenizer_id,
        args.model_max_length,
        trust_remote_code=args.trust_remote_code,
    )

    benchmark_dataset = BenchmarkDataset.from_flags(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
    )

    if (
        args.num_chat_sessions
        and not benchmark_dataset.has_multiturn_chat_support
    ):
        raise ValueError(
            f"Multiturn chat is not supported for dataset {benchmark_dataset}"
        )

    logger.info("sampling requests")
    if args.dataset_name is None:
        raise ValueError(
            "Please specify '--dataset-name' and the corresponding "
            "'--dataset-path' if required."
        )

    # Build output_lengths array
    if args.num_prompts is not None:
        num_requests = args.num_prompts
    else:
        num_requests = args.num_chat_sessions

    if args.output_lengths is None:
        output_lengths = None
    elif os.path.exists(args.output_lengths):
        with open(args.output_lengths) as f:
            output_lengths = yaml.safe_load(f)["output_lengths"]
    else:
        output_lengths = [int(args.output_lengths)] * num_requests

    input_requests: Sequence[SampledRequest] = []
    chat_sessions: Sequence[ChatSession] = []
    if isinstance(benchmark_dataset, CodeDebugBenchmarkDataset):
        # code_debug is a long-context dataset based on InfiniteBench
        if args.num_chat_sessions:
            if args.output_lengths is not None:
                raise NotImplementedError(
                    "TODO: Add support for fixed output lengths with multi-turn"
                    " code-debug"
                )
            chat_sessions = benchmark_dataset.gen_twoturn_longcontext_requests(
                num_chat_sessions=args.num_chat_sessions,
                tokenizer=tokenizer,
            )
        else:
            input_requests = benchmark_dataset.sample_requests(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                output_lengths=output_lengths,
                shuffle=(
                    args.output_lengths is None
                    and not args.record_output_lengths
                ),
            )

    elif isinstance(benchmark_dataset, ShareGPTBenchmarkDataset):
        input_requests = benchmark_dataset.sample_requests(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_lengths=output_lengths,
            shuffle=(
                args.output_lengths is None and not args.record_output_lengths
            ),
        )

    elif isinstance(benchmark_dataset, SonnetBenchmarkDataset):
        # For sonnet, formatting depends on the endpoint
        apply_chat_template = chat
        # Sample sonnet requests with common parameters
        input_requests = benchmark_dataset.sample_requests(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_lengths=output_lengths,
            input_len=args.sonnet_input_len,
            prefix_len=args.sonnet_prefix_len,
            apply_chat_template=apply_chat_template,
        )

    elif isinstance(benchmark_dataset, VisionArenaBenchmarkDataset):
        input_requests = benchmark_dataset.sample_requests(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_lengths=output_lengths,
        )
    elif isinstance(benchmark_dataset, ArxivSummarizationBenchmarkDataset):
        if output_lengths:
            ValueError(
                "Arxiv summarization dataset does not support --output-lengths."
                " Please use --max-output-len"
            )
        input_requests = benchmark_dataset.sample_requests(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            shuffle=not args.record_output_lengths,
            input_len=args.arxiv_summarization_input_len,
            max_output_len=args.max_output_len,
        )
    elif isinstance(benchmark_dataset, RandomBenchmarkDataset):
        if args.num_chat_sessions:
            chat_sessions = benchmark_dataset.gen_multiturn_random_requests(
                input_len=args.random_input_len,
                output_len=args.random_output_len,
                num_chat_sessions=args.num_chat_sessions,
                num_turns=args.random_num_turns,
                coefficient_of_variation=args.random_coefficient_of_variation,
                tokenizer=tokenizer,
                sys_prompt_ratio=args.random_sys_prompt_ratio,
                max_num_unique_sys_prompt=args.random_max_num_unique_sys_prompt,
                distribution_type=args.random_distribution_type,
                first_turn_ratio=args.random_first_turn_ratio,
            )
        else:
            input_requests = benchmark_dataset.sample_requests(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                input_len=args.random_input_len,
                output_len=args.random_output_len,
                coefficient_of_variation=args.random_coefficient_of_variation,
                sys_prompt_ratio=args.random_sys_prompt_ratio,
                max_num_unique_sys_prompt=args.random_max_num_unique_sys_prompt,
                distribution_type=args.random_distribution_type,
                image_size=args.random_image_size,
                image_count=args.random_image_count,
            )
    elif isinstance(benchmark_dataset, AxolotlBenchmarkDataset):
        input_requests = benchmark_dataset.sample_requests(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_lengths=output_lengths,
            shuffle=(
                args.output_lengths is None and not args.record_output_lengths
            ),
        )
    elif isinstance(benchmark_dataset, ObfuscatedConversationsBenchmarkDataset):
        if output_lengths is None:
            output_scale = (
                args.obfuscated_conversations_average_output_len
                * args.obfuscated_conversations_coefficient_of_variation
            )
            output_lengths = np.random.normal(
                loc=args.obfuscated_conversations_average_output_len,
                scale=output_scale,
                size=num_requests,
            ).tolist()
            output_lengths = np.round(output_lengths).astype(int).tolist()
            output_lengths = [
                max(output_len, 1) for output_len in output_lengths
            ]
        input_requests = benchmark_dataset.sample_requests(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_lengths=output_lengths,
            shuffle=args.obfuscated_conversations_shuffle,
            seed=args.seed,
        )
    elif isinstance(benchmark_dataset, BatchJobBenchmarkDataset):
        input_requests = benchmark_dataset.sample_requests(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_lengths=output_lengths,
            shuffle=(
                args.output_lengths is None and not args.record_output_lengths
            ),
            image_dir=args.batch_job_image_dir,
        )
    else:
        raise ValueError(f"Unknown / unsupported dataset: {benchmark_dataset}")

    if args.print_inputs_and_outputs:
        if args.num_chat_sessions:
            raise NotImplementedError(
                "Printing out multi-turn chats is not supported."
            )

        print("Input prompts:")
        for req_id, request in enumerate(input_requests):
            print(
                {
                    "req_id": req_id,
                    "output_len": request.output_len,
                    "prompt_len": request.prompt_len,
                    "prompt": request.prompt_formatted,
                    "encoded_images": request.encoded_images,
                }
            )

    # Generate LoRA configurations if needed
    lora_configs = None
    if args.num_loras > 0 or args.lora_paths:
        lora_configs = generate_loras(
            model_id,
            args.lora_rank,
            args.num_loras,
            args.lora_target_modules,
            args.lora_paths,
            args.lora_output_dir,
            args.lora_server_path,
        )

    if args.max_concurrency is not None:
        try:
            args.max_concurrency = int(args.max_concurrency)
        except ValueError as e:
            raise ValueError(
                f"Expected a single integer value for max_concurrency, got {args.max_concurrency}"
            ) from e
    if args.request_rate is not None:
        try:
            args.request_rate = float(args.request_rate)
        except ValueError as e:
            raise ValueError(
                f"Expected a single float value for request_rate, got {args.request_rate}"
            ) from e

    logger.info("starting benchmark run")
    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            chat=chat,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            chat_sessions=chat_sessions,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            max_concurrency=args.max_concurrency,
            disable_tqdm=args.disable_tqdm,
            do_test_prompt=not args.skip_test_prompt,
            collect_gpu_stats=args.collect_gpu_stats,
            collect_cpu_stats=args.collect_cpu_stats,
            print_inputs_and_outputs=args.print_inputs_and_outputs,
            max_requests=args.num_prompts,
            num_chat_sessions=args.num_chat_sessions,
            delay_between_chat_turns=args.delay_between_chat_turns,
            skip_first_n_requests=args.skip_first_n_requests,
            max_output_len=args.max_output_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_benchmark_duration_s=args.max_benchmark_duration_s,
            warmup_delay_ms=args.chat_warmup_delay_ms,
            ignore_first_turn_stats=args.ignore_first_turn_stats,
            lora_request_ratio=args.lora_request_ratio,
            lora_configs=lora_configs,
            max_concurrent_lora_ops=args.max_concurrent_lora_ops,
        )
    )

    # Benchmark run failed if any failed requests
    if benchmark_result["failures"] != 0:
        logger.info("finished benchmark run: Failed.")
        sys.exit(1)

    # Save config and results to json
    if args.save_result:
        logger.info("saving results")
        result_json: dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["num_prompts"] = benchmark_result["completed"]
        result_json["server_args"] = args.server_args
        result_json["dataset_name"] = args.dataset_name
        result_json["client_args"] = dict(vars(args))
        # json doesn't allow infinity as numeric, so cast this to string
        result_json["client_args"]["request_rate"] = str(
            result_json["client_args"]["request_rate"]
        )

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    key = kvstring[0].strip()
                    value = kvstring[1].strip()

                    if key == "server_cpu":
                        # Map server_cpu to cpu for consistency with existing data pipeline
                        result_json["cpu"] = value
                    else:
                        result_json[key] = value
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf"
        )
        result_json["burstiness"] = args.burstiness
        result_json["max_concurrency"] = args.max_concurrency

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Add LoRA metrics if present
        if "lora_metrics" in benchmark_result:
            result_json["lora_metrics"] = benchmark_result["lora_metrics"]

        # Save to file
        if args.result_filename:
            file_name = os.path.join(
                args.result_dir or "", args.result_filename
            )
        else:
            base_model_id = model_id.split("/")[-1]
            max_concurrency_str = (
                f"-concurrency{args.max_concurrency}"
                if args.max_concurrency is not None
                else ""
            )
            # When auto-generating file names, add suffixes if we have to to
            # ensure we're not overwriting an existing file (best effort,
            # subject to TOCTTOU).
            for uniq_count in itertools.count(1):
                if uniq_count == 1:
                    uniq_suffix = ""
                else:
                    uniq_suffix = f"-{uniq_count}"
                file_name = (
                    f"{backend}-{args.request_rate}qps{max_concurrency_str}-"
                    f"{base_model_id}-{current_dt}{uniq_suffix}.json"
                )
                file_name = os.path.join(args.result_dir or "", file_name)
                if not os.path.exists(file_name):
                    break
        logger.info(f"Writing file: {file_name}")
        if os.path.isfile(file_name):
            logger.warning(
                "This is going to overwrite an existing file.  "
                f"The existing file will be moved to {file_name}.orig."
            )
            os.rename(file_name, f"{file_name}.orig")
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)

    # Save output lengths if requested
    if args.record_output_lengths:
        # Save relevant input args for context
        args_to_save = (
            "backend",
            "burstiness",
            "dataset_name",
            "dataset_path",
            "endpoint",
            "max_concurrency",
            "max_output_len",
            "model",
            "request_rate",
            "seed",
            "temperature",
            "top_p",
        )
        output_lens_dict = {}
        output_lens_dict["args"] = {x: vars(args)[x] for x in args_to_save}
        output_lens_dict["output_lengths"] = benchmark_result["output_lens"]
        with open(args.record_output_lengths, "w") as f:
            yaml.dump(output_lens_dict, f)

    logger.info("finished benchmark run: Success.")


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments using ServingBenchmarkConfig with enhanced cli_parse_args().

    This function uses the generalized parse_benchmark_args function to handle
    config file inheritance and CLI argument parsing.

    Args:
        args: Command line arguments to parse. If None, parse from sys.argv.
    """
    return parse_benchmark_args(
        config_class=ServingBenchmarkConfig,
        default_config_path=Path(__file__).parent
        / "configs"
        / "serving_config.yaml",
        description=BENCHMARK_SERVING_ARGPARSER_DESCRIPTION,
        args=args,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
