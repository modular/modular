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

"""BenchmarkConfig MAXConfig subclass for MAX benchmark configurations."""

from dataclasses import dataclass, field
from typing import Any, Optional

from max.pipelines.lib import MAXConfig


@dataclass
class BenchmarkConfig(MAXConfig):
    """Configuration class for MAX benchmark parameters.

    This class can handle both known parameters (validated against the dataclass)
    and unknown parameters (stored as additional fields via _unknown_fields).
    This provides flexibility for extensions while maintaining validation for core parameters.

    Base configuration is loaded from base_benchmark_config.yaml with optional overrides.
    This ensures that CLI parsing doesn't need to know default values since they're
    all specified in the configuration file.
    """

    # Config file section name for MAXConfig interface
    _config_file_section_name: str = "benchmark_config"
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    # Backend and API configuration
    # TODO: Make this an enum.
    backend: str = "modular"
    """Backend to use for benchmarking. Choices: vllm, vllm-chat, trt-llm, modular, modular-chat, sglang, sglang-chat"""

    base_url: Optional[str] = None
    """Server or API base url if not using http host and port."""

    host: str = "localhost"
    """Server host."""

    port: int = 8000
    """Server port."""

    # TODO: Make this an enum.
    endpoint: str = "/v1/completions"
    """API endpoint. Choices: /v1/completions, /v1/chat/completions, /v2/models/ensemble/generate_stream"""

    # Dataset configuration
    dataset_name: str = "sharegpt"
    """Name of the dataset to benchmark on."""

    dataset_path: Optional[str] = None
    """Path to the dataset."""

    # Request configuration
    max_concurrency: Optional[int] = None
    """Maximum number of concurrent requests. This can be used to help simulate an environment where a higher level component is enforcing a maximum number of concurrent requests. While the --request-rate argument controls the rate at which requests are initiated, this argument will control how many are actually allowed to execute at a time. This means that when used in combination, the actual request rate may be lower than specified with --request-rate, if the server is not processing requests fast enough to keep up."""

    model: Optional[str] = None
    """Name of the model. Required when running benchmark."""

    lora: Optional[str] = None
    """Name of the lora."""

    tokenizer: Optional[str] = None
    """Name or path of the tokenizer, if not using the default tokenizer."""

    # Workload configuration
    num_prompts: int = 1000
    """Number of prompts to process."""

    max_benchmark_duration_s: Optional[int] = None
    """Maximum duration of the benchmark in seconds. If specified, the benchmark will run for max_benchmark_duration_s seconds. It works if max_concurrency is not None. If not specified, the benchmark will run for num_prompts requests."""

    num_chat_sessions: Optional[int] = None
    """Number of multiturn chat sessions to spawn. Single turn mode if not specified / set to None."""

    delay_between_chat_turns: Optional[int] = None
    """Optional delay before sending next chat turn request in ms."""

    # Output control
    output_lengths: Optional[str] = None
    """Path to YAML file containing list of output lengths, or an int. If an int is given, all responses are forced to the given length. Default: None"""

    max_output_len: Optional[int] = None
    """Max output length for each request."""

    temperature: float = 0.0
    """Temperature used for token sampling."""

    # Traffic control
    request_rate: float = float("inf")
    """Number of requests per second. If this is inf, then all the requests are sent at time 0. Otherwise, we use Poisson process to synthesize the request arrival times."""

    burstiness: float = 1.0
    """Burstiness factor of the request generation. Only take effect when request_rate is not inf. Default value is 1, which follows Poisson process. Otherwise, the request intervals follow a gamma distribution. A lower burstiness value (0 < burstiness < 1) results in more bursty requests. A higher burstiness value (burstiness > 1) results in a more uniform arrival of requests."""

    ttft_skip_requests: int = 0
    """Number of requests to skip when measuring TTFT latencies; i.e, ignore the first N requests in TTFT calculations. This mitigates the effect of artificial queuing, particularly at infinite request rates. Defaults to zero."""

    chat_warmup_delay_ms: float = 0.0
    """Delay between starting chat sessions when the number of active sessions is below max_concurrency, in ms. This prevents all of the chat sessions from starting at the same time. Default: 0.0"""

    # Dataset-specific parameters
    sonnet_input_len: int = 550
    """Number of input tokens per request, used only for sonnet dataset."""

    sonnet_prefix_len: int = 200
    """Number of prefix tokens per request, used only for sonnet dataset."""

    arxiv_summarization_input_len: int = 15000
    """Number of input tokens per request, used only for arxiv-summarization dataset."""

    random_input_len: int = 1024
    """Number of input tokens per request, used only for random sampling."""

    random_output_len: int = 128
    """Number of output tokens per request, used only for random sampling."""

    random_coefficient_of_variation: str = "0.3,0.7"
    """Coefficient of variation for input/output length, used only for random sampling."""

    random_image_size: str = ""
    """Size of random images to generate."""

    random_sys_prompt_ratio: float = 0.0
    """Ratio to determine the system prompt length, used only for random sampling."""

    random_first_turn_ratio: float = 1.0
    """Ratio of the length of the first turn to the length of subsequent turns."""

    random_max_num_unique_sys_prompt: int = 1
    """Maximum number of unique system prompts, used only for random sampling."""

    random_distribution_type: str = "normal"
    """Type of probability distribution for sampled input/output length. Choices: uniform, normal"""

    random_num_turns: int = 1
    """Number of turns per session, used only for random sampling and --num-chat-sessions."""

    # Control flags
    seed: int = 0
    """Random seed."""

    trust_remote_code: bool = False
    """Trust remote code from huggingface."""

    disable_tqdm: bool = False
    """Specify to disable tqdm progress bar."""

    skip_test_prompt: bool = False
    """Skip the test prompt. Useful when doing external profiling."""

    collect_gpu_stats: bool = False
    """Collect GPU stats with NVML (NVIDIA only)."""

    save_result: bool = False
    """Specify to save benchmark results to a json file."""

    print_inputs_and_outputs: bool = False
    """Print all input and outputs to console."""

    # Result saving
    metadata: list[str] = field(default_factory=list)
    """Key-value pairs (e.g, --metadata version=0.3.3 tp=1) for metadata of this run to be saved in the result JSON file for record keeping purposes."""

    result_dir: Optional[str] = None
    """Specify directory to save benchmark json results. If not specified, results are saved in the current directory."""

    result_filename: Optional[str] = None
    """Specify the filename to save benchmark json results. If not specified, results will be saved in {backend}-{args.request_rate}qps(-concurrency{args.max_concurrency})?-{base_model_id}-{current_dt}.json format. (Note that, concurrency exists in filename when args.max_concurrency is specified.)"""

    server_args: str = ""
    """Server args."""

    # File operations (Note: record_output_lengths is handled specially in argparse)
    record_output_lengths: Optional[str] = None
    """Path to save output lengths in YAML format. If not specified, output lengths are not saved."""

    # Unknown fields storage (not a dataclass field)
    _unknown_fields: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )

    @staticmethod
    def help() -> dict[str, str]:
        """Documentation for this config class.

        Returns:
            Dictionary of config options and their descriptions.
        """

        return {
            "backend": "Backend to use for benchmarking. Choices: vllm, vllm-chat, trt-llm, modular, modular-chat, sglang, sglang-chat",
            "base_url": "Server or API base url if not using http host and port.",
            "host": "Server host.",
            "port": "Server port.",
            "endpoint": "API endpoint. Choices: /v1/completions, /v1/chat/completions, /v2/models/ensemble/generate_stream",
            "dataset_name": "Name of the dataset to benchmark on.",
            "dataset_path": "Path to the dataset.",
            "max_concurrency": "Maximum number of concurrent requests. This can be used to help simulate an environment where a higher level component is enforcing a maximum number of concurrent requests. While the --request-rate argument controls the rate at which requests are initiated, this argument will control how many are actually allowed to execute at a time. This means that when used in combination, the actual request rate may be lower than specified with --request-rate, if the server is not processing requests fast enough to keep up.",
            "model": "Name of the model. Required when running benchmark.",
            "lora": "Name of the lora.",
            "tokenizer": "Name or path of the tokenizer, if not using the default tokenizer.",
            "num_prompts": "Number of prompts to process.",
            "max_benchmark_duration_s": "Maximum duration of the benchmark in seconds. If specified, the benchmark will run for max_benchmark_duration_s seconds. It works if max_concurrency is not None. If not specified, the benchmark will run for num_prompts requests.",
            "num_chat_sessions": "Number of multiturn chat sessions to spawn. Single turn mode if not specified / set to None.",
            "delay_between_chat_turns": "Optional delay before sending next chat turn request in ms.",
            "output_lengths": "Path to YAML file containing list of output lengths, or an int. If an int is given, all responses are forced to the given length. Default: None",
            "max_output_len": "Max output length for each request.",
            "temperature": "Temperature used for token sampling.",
            "request_rate": "Number of requests per second. If this is inf, then all the requests are sent at time 0. Otherwise, we use Poisson process to synthesize the request arrival times.",
            "burstiness": "Burstiness factor of the request generation. Only take effect when request_rate is not inf. Default value is 1, which follows Poisson process. Otherwise, the request intervals follow a gamma distribution. A lower burstiness value (0 < burstiness < 1) results in more bursty requests. A higher burstiness value (burstiness > 1) results in a more uniform arrival of requests.",
            "ttft_skip_requests": "Number of requests to skip when measuring TTFT latencies; i.e, ignore the first N requests in TTFT calculations. This mitigates the effect of artificial queuing, particularly at infinite request rates. Defaults to zero.",
            "chat_warmup_delay_ms": "Delay between starting chat sessions when the number of active sessions is below max_concurrency, in ms. This prevents all of the chat sessions from starting at the same time. Default: 0.0",
            "sonnet_input_len": "Number of input tokens per request, used only for sonnet dataset.",
            "sonnet_prefix_len": "Number of prefix tokens per request, used only for sonnet dataset.",
            "arxiv_summarization_input_len": "Number of input tokens per request, used only for arxiv-summarization dataset.",
            "random_input_len": "Number of input tokens per request, used only for random sampling.",
            "random_output_len": "Number of output tokens per request, used only for random sampling.",
            "random_coefficient_of_variation": "Coefficient of variation for input/output length, used only for random sampling.",
            "random_image_size": "Size of random images to generate.",
            "random_sys_prompt_ratio": "Ratio to determine the system prompt length, used only for random sampling.",
            "random_first_turn_ratio": "Ratio of the length of the first turn to the length of subsequent turns.",
            "random_max_num_unique_sys_prompt": "Maximum number of unique system prompts, used only for random sampling.",
            "random_distribution_type": "Type of probability distribution for sampled input/output length. Choices: uniform, normal",
            "random_num_turns": "Number of turns per session, used only for random sampling and --num-chat-sessions.",
            "seed": "Random seed.",
            "trust_remote_code": "Trust remote code from huggingface.",
            "disable_tqdm": "Specify to disable tqdm progress bar.",
            "skip_test_prompt": "Skip the test prompt. Useful when doing external profiling.",
            "collect_gpu_stats": "Collect GPU stats with NVML (NVIDIA only).",
            "save_result": "Specify to save benchmark results to a json file.",
            "print_inputs_and_outputs": "Print all input and outputs to console.",
            "metadata": "Key-value pairs (e.g, --metadata version=0.3.3 tp=1) for metadata of this run to be saved in the result JSON file for record keeping purposes.",
            "result_dir": "Specify directory to save benchmark json results. If not specified, results are saved in the current directory.",
            "result_filename": "Specify the filename to save benchmark json results. If not specified, results will be saved in {backend}-{args.request_rate}qps(-concurrency{args.max_concurrency})?-{base_model_id}-{current_dt}.json format. (Note that, concurrency exists in filename when args.max_concurrency is specified.)",
            "server_args": "Server args.",
            "record_output_lengths": "Path to save output lengths in YAML format. If not specified, output lengths are not saved.",
        }
