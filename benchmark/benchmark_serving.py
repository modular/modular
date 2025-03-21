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

import argparse
import asyncio
import json
import logging
import os
import random
import resource
import sys
import time
import traceback
import warnings
from argparse import ArgumentParser as FlexibleArgumentParser
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
from huggingface_hub import hf_hub_download
from tqdm.asyncio import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

# 10 minute timeout per request session
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=10 * 60)

logger = logging.getLogger("benchmark_serving")


CODE_DEBUG_TEMPLATE = "There is ONLY ONE function in the large project that is deliberately made to include an obvious error. Please find the function that contains the most obvious errors. I will give you four options to narrow your scope. You can inspect the options and think. Eventually, tell me the answer using one single letter (A, B, C, or D).\n\n{context}\n\nWhich funtion has deliberate error?\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nYou should first find the functions in the options. Repeat their content, inspect through code, and at last give me your answer for the function that has the deliberate and obvious error in A, B, C, or D."


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(
        default_factory=list
    )  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""


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
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input.prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
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
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(("completions", "profile")), (
        "OpenAI Completions API URL must end with 'completions' or 'profile'."
    )

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "best_of": 1,
            "max_tokens": request_func_input.output_len,
            "stream": True,
            "ignore_eos": True,
        }

        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
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
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("chat/completions"), (
        "OpenAI Chat Completions API URL must end with 'chat/completions'."
    )

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "messages": [
                {
                    "role": "user",
                    "content": request_func_input.prompt,
                },
            ],
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
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


# Since vllm must support Python 3.8, we can't use str.removeprefix(prefix)
# introduced in Python 3.9
def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def get_tokenizer(
    pretrained_model_name_or_path: str, trust_remote_code: bool
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code
    )


ASYNC_REQUEST_FUNCS = {
    "vllm": async_request_openai_completions,
    "trt-llm": async_request_trt_llm,
    "modular": async_request_openai_completions,
    "modular-chat": async_request_openai_chat_completions,
}


# from https://github.com/sgl-project/sglang/blob/v0.4.0/python/sglang/bench_serving.py#L1283
def set_ulimit(target_soft_limit=65535):
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")


@dataclass
class BenchmarkMetrics:
    completed: int
    failures: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    p99_itl_ms: float
    max_input: int
    max_output: int
    max_total: int
    peak_gpu_memory_mib: float  # 'benchmark/gpu:0/memory_used (MiB)/max'
    available_gpu_memory_mib: float  # 'benchmark/gpu:0/memory_free (MiB)/min'
    gpu_utilization: float  # 'benchmark/gpu:0/gpu_utilization (%)/mean'


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
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
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
        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset


def sample_sonnet_requests(
    dataset_path: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    prefix_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, str, int, int]]:
    assert input_len > prefix_len, (
        "'args.sonnet-input-len' must be greater than 'args.prefix-input-len'."
    )

    # Load the dataset.
    with open(dataset_path) as f:
        poem_lines = f.readlines()

    # Tokenize the poem lines.
    poem_token_ids = tokenizer(poem_lines).input_ids
    average_poem_len = sum(
        len(token_ids) for token_ids in poem_token_ids
    ) / len(poem_token_ids)

    # Base prefix for all requests.
    base_prompt = "Pick as many lines as you can from these poem lines:\n"
    base_message = [
        {
            "role": "user",
            "content": base_prompt,
        }
    ]
    base_prompt_formatted = tokenizer.apply_chat_template(
        base_message, add_generation_prompt=True, tokenize=False
    )
    base_prompt_offset = len(tokenizer(base_prompt_formatted).input_ids)

    assert input_len > base_prompt_offset, (
        f"Please set 'args.sonnet-input-len' higher than {base_prompt_offset}."
    )
    num_input_lines = round((input_len - base_prompt_offset) / average_poem_len)

    # First approximately `prefix_len` number of tokens in the
    # prompt are fixed poem lines.
    assert prefix_len > base_prompt_offset, (
        f"Please set 'args.sonnet-prefix-len' higher than {base_prompt_offset}."
    )

    num_prefix_lines = round(
        (prefix_len - base_prompt_offset) / average_poem_len
    )
    prefix_lines = poem_lines[:num_prefix_lines]

    # Sample the rest of lines per request.
    sampled_requests: List[Tuple[str, str, int, int]] = []
    for _ in range(num_requests):
        sampled_lines = "".join(
            prefix_lines
            + random.sample(poem_lines, num_input_lines - num_prefix_lines)
        )

        prompt = f"{base_prompt}{sampled_lines}"
        message = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        prompt_formatted = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False
        )
        prompt_len = len(tokenizer(prompt_formatted).input_ids)
        sampled_requests.append(
            (prompt, prompt_formatted, prompt_len, output_len)
        )

    return sampled_requests


def sample_random_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    input_lens = np.random.randint(
        int(input_len * range_ratio),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )
    offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
    input_requests = []
    for i in range(num_prompts):
        prompt = tokenizer.decode(
            [
                (offsets[i] + i + j) % tokenizer.vocab_size
                for j in range(input_lens[i])
            ]
        )
        input_requests.append((prompt, int(input_lens[i]), int(output_lens[i])))

    return input_requests


def format_code_debug_context(request_features: dict):
    code = request_features["context"]
    prompt = CODE_DEBUG_TEMPLATE.format(
        context=code,
        OPTION_A=request_features["options"][0],
        OPTION_B=request_features["options"][1],
        OPTION_C=request_features["options"][2],
        OPTION_D=request_features["options"][3],
    )
    return prompt


def get_code_debug_answer(request_features: dict):
    OPTIONS = "ABCD"
    if isinstance(request_features["answer"], list):
        if len(request_features["answer"]) == 1:
            ret = OPTIONS[
                request_features["options"].index(request_features["answer"][0])
            ]
        else:
            raise ValueError("More than 1 answers")
    else:
        raise ValueError("Invalid answer type")
    return ret


def sample_longcontext_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int]]:
    """
    The Long-Context dataset workload is based on InfiniteBench Code.debug
    """
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    with open(dataset_path, "r") as jsonl_file:
        json_list = list(jsonl_file)
    dataset = [json.loads(json_str) for json_str in json_list]

    # format context/options/answer -> template of (prompt, completion)
    dataset = [
        (format_code_debug_context(data), get_code_debug_answer(data))
        for data in dataset
    ]

    # Filter out data with no LICENSE
    dataset = [data for data in dataset if "LICENSE" in data[0]]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    model_max_length = tokenizer.model_max_length
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids)
            if fixed_output_len is None
            else fixed_output_len
        )
        if (
            prompt_len > model_max_length
            or prompt_len + output_len > model_max_length
        ):
            # Prune too long sequences.
            print(
                f"Skip too long sequences ({prompt_len} > {model_max_length})..."
            )
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    if __debug__:
        from statistics import mean

        list_prompt_len = [data[1] for data in filtered_dataset]
        print(
            f"INFO: Sampled {len(filtered_dataset)} Long-Context Requests: "
            f"Input Tokens(Average: {mean(list_prompt_len)}, "
            f"Min: {min(list_prompt_len)}, Max: {max(list_prompt_len)})"
        )

    return filtered_dataset


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    gpu_metrics: Dict[str, Any],
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    max_input = 0
    max_output = 0
    max_total = 0
    failures = 0
    failed_responses = []
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note : this may inflate the output token count slightly
            output_len = compute_output_len(tokenizer, outputs[i])
            actual_output_lens.append(output_len)
            total_input += input_requests[i][1]
            if output_len > 1:
                tpots.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1)
                )
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            completed += 1
            max_input = max(max_input, input_requests[i][1])
            max_output = max(max_output, output_len)
            max_total = max(max_total, input_requests[i][1] + output_len)
        else:
            actual_output_lens.append(0)
            failures = failures + 1
            failed_responses.append(outputs[i])

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
    metrics = BenchmarkMetrics(
        completed=completed,
        failures=failures,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        mean_ttft_ms=float(np.mean(ttfts or 0))
        * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=float(np.median(ttfts or 0)) * 1000,
        std_ttft_ms=float(np.std(ttfts or 0)) * 1000,
        p99_ttft_ms=float(np.percentile(ttfts or 0, 99)) * 1000,
        mean_tpot_ms=float(np.mean(tpots or 0)) * 1000,
        median_tpot_ms=float(np.median(tpots or 0)) * 1000,
        std_tpot_ms=float(np.std(tpots or 0)) * 1000,
        p99_tpot_ms=float(np.percentile(tpots or 0, 99)) * 1000,
        mean_itl_ms=float(np.mean(itls or 0)) * 1000,
        median_itl_ms=float(np.median(itls or 0)) * 1000,
        std_itl_ms=float(np.std(itls or 0)) * 1000,
        p99_itl_ms=float(np.percentile(itls or 0, 99)) * 1000,
        max_input=max_input,
        max_output=max_output,
        max_total=max_total,
        peak_gpu_memory_mib=float(
            gpu_metrics.get("benchmark/gpu:0/memory_used (MiB)/max") or 0
        ),
        available_gpu_memory_mib=float(
            gpu_metrics.get("benchmark/gpu:0/memory_free (MiB)/min") or 0
        ),
        gpu_utilization=float(
            gpu_metrics.get("benchmark/gpu:0/gpu_utilization (%)/mean") or 0
        ),
    )

    return metrics, actual_output_lens


async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    disable_tqdm: bool,
    do_test_prompt: bool,
    collect_gpu_stats: bool,
    print_inputs_and_outputs: bool,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if do_test_prompt:
        logger.info("Starting initial single prompt test run...")
        test_prompt, test_prompt_len, test_output_len = input_requests[0]
        test_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=api_url,
            prompt_len=test_prompt_len,
            output_len=test_output_len,
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

    logger.info(f"Traffic request rate: {request_rate}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))
    if collect_gpu_stats:
        from nvitop import ResourceMetricCollector

        collector = ResourceMetricCollector()
        collector.start("benchmark")

    benchmark_start_time = time.perf_counter_ns()
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
        )
        tasks.append(
            asyncio.create_task(
                request_func(
                    request_func_input=request_func_input,
                    pbar=pbar,
                )
            )
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = (time.perf_counter_ns() - benchmark_start_time) / 1e9

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

    if collect_gpu_stats:
        gpu_metrics = collector.collect()
        collector.stop()
    else:
        gpu_metrics = {}

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        gpu_metrics=gpu_metrics,
    )

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10}".format("Failed requests:", metrics.failures))
    print(
        "{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration)
    )
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print(
        "{:<40} {:<10}".format("Total generated tokens:", metrics.total_output)
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", metrics.input_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print(
        "{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms)
    )
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print(
        "{s:{c}^{n}}".format(
            s="Time per Output Token (excl. 1st token)", n=50, c="-"
        )
    )
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print(
        "{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms)
    )
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("{s:{c}^{n}}".format(s="Token Stats", n=50, c="-"))
    print("{:<40} {:<10}".format("Max input tokens:", metrics.max_input))
    print("{:<40} {:<10}".format("Max output tokens:", metrics.max_output))
    print("{:<40} {:<10}".format("Max total tokens:", metrics.max_total))
    if collect_gpu_stats:
        print("{s:{c}^{n}}".format(s="GPU Stats", n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                "GPU Utilization (%):", metrics.gpu_utilization
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Peak GPU Memory Used (MiB):", metrics.peak_gpu_memory_mib
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "GPU Memory Available (MiB):", metrics.available_gpu_memory_mib
            )
        )

    print("=" * 50)

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "std_ttft_ms": metrics.std_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "std_tpot_ms": metrics.std_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "std_itl_ms": metrics.std_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
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
    return result


def main(args: argparse.Namespace):
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

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    logger.info(f"getting tokenizer. api url: {api_url}, base_url: {base_url}")
    tokenizer = get_tokenizer(
        tokenizer_id, trust_remote_code=args.trust_remote_code
    )

    logger.info("sampling requests")
    if args.dataset is not None:
        warnings.warn(
            (
                "The '--dataset' argument will be deprecated in the next "
                "release. Please use '--dataset-name' and "
                "'--dataset-path' in the future runs."
            ),
            stacklevel=2,
        )
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
        )
    elif args.dataset_name == "code_debug":
        # code_debug is a long-context dataset based on InfiniteBench
        input_requests = sample_longcontext_requests(
            dataset_path=args.dataset_path
            or fetch_dataset_from_hf(args.dataset_name),
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
        )

    elif args.dataset_name == "sharegpt":
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path
            or fetch_dataset_from_hf(args.dataset_name),
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
        )

    elif args.dataset_name == "sonnet":
        # Sample sonnet requests with common parameters
        sonnet_requests = sample_sonnet_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            input_len=args.sonnet_input_len,
            output_len=args.sonnet_output_len,
            prefix_len=args.sonnet_prefix_len,
            tokenizer=tokenizer,
        )

        # Do not format the prompt, pass to message directly
        if args.backend == "openai-chat":
            # For chat API, use raw prompt without formatting
            input_requests = [
                (prompt, prompt_len, output_len)
                for prompt, _, prompt_len, output_len in sonnet_requests
            ]
        else:
            # For non-chat API, ensure model has chat template and use formatted prompt
            assert tokenizer.chat_template or tokenizer.default_chat_template, (
                "Tokenizer/model must have chat template for sonnet dataset."
            )
            input_requests = [
                (prompt_formatted, prompt_len, output_len)
                for _, prompt_formatted, prompt_len, output_len in sonnet_requests
            ]

    elif args.dataset_name == "random":
        input_requests = sample_random_requests(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
        )

    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    if args.print_inputs_and_outputs:
        print("Input prompts:")
        for req_id, (prompt_formatted, prompt_len, output_len) in enumerate(
            input_requests
        ):
            print(
                {
                    "req_id": req_id,
                    "output_len": output_len,
                    "prompt_len": prompt_len,
                    "prompt": prompt_formatted,
                }
            )

    logger.info("starting benchmark run")
    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            request_rate=args.request_rate,
            disable_tqdm=args.disable_tqdm,
            do_test_prompt=not args.skip_test_prompt,
            collect_gpu_stats=args.collect_gpu_stats,
            print_inputs_and_outputs=args.print_inputs_and_outputs,
        )
    )

    # Benchmark run failed if any failed requests
    if args.num_prompts != benchmark_result["completed"]:
        logger.info("finished benchmark run: Failed.")
        sys.exit(1)

    # Save config and results to json
    if args.save_result:
        logger.info("saving results")
        result_json: Dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["num_prompts"] = args.num_prompts
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
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf"
        )

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = f"{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"  # noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)

    logger.info("finished benchmark run: Success.")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="modular",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Path to the ShareGPT dataset, will be deprecated in the "
            "next release."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["sharegpt", "sonnet", "random", "code_debug"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None, help="Path to the dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=(  # noqa: E501
            "Name or path of the tokenizer, if not using the default tokenizer."
        ),
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help=(
            "Output length for each request. Overrides the output length "
            "from the ShareGPT dataset."
        ),
    )
    parser.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help=(
            "Number of input tokens per request, used only for sonnet dataset."
        ),
    )
    parser.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help=(
            "Number of output tokens per request, used only for sonnet dataset."
        ),
    )
    parser.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help=(
            "Number of prefix tokens per request, used only for sonnet dataset."
        ),
    )
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help=(
            "Number of input tokens per request, used only for random sampling."
        ),
    )
    parser.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help=(
            "Number of output tokens per request, used only for random"
            " sampling."
        ),
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=1.0,
        help=(
            "Range of sampled ratio of input/output length, "
            "used only for random sampling."
        ),
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help=(
            "Number of requests per second. If this is inf, "
            "then all the requests are sent at time 0. "
            "Otherwise, we use Poisson process to synthesize "
            "the request arrival times."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--skip-test-prompt",
        action="store_true",
        help="Skip the test prompt.  Useful when doing external profiling.",
    )
    parser.add_argument(
        "--collect-gpu-stats",
        action="store_true",
        help="Collect GPU stats with NVML (NVIDIA only).",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help=(
            "Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
            "for metadata of this run to be saved in the result JSON file "
            "for record keeping purposes."
        ),
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help=(
            "Specify directory to save benchmark json results."
            "If not specified, results are saved in the current directory."
        ),
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help=(
            "Specify the filename to save benchmark json results."
            "If not specified, results will be saved in "
            "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
            " format."
        ),
    )
    parser.add_argument(
        "--print-inputs-and-outputs",
        action="store_true",
        help="Print all input and outputs to console.",
    )

    parser.add_argument(
        "--server-args",
        type=str,
        default="",
        help="Server args",
    )

    args = parser.parse_args()
    main(args)
