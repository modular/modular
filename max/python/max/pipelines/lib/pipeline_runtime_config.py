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

"""Model-agnostic runtime configuration for pipeline execution."""

from __future__ import annotations

import os

from max.config import ConfigFileModel
from max.pipelines.diffusion.cache import DenoisingCacheConfig
from max.pipelines.modeling.config_enums import PipelineRole
from pydantic import Field, PrivateAttr

# Default max batch input tokens for chunked prefill and memory estimation.
DEFAULT_MAX_BATCH_INPUT_TOKENS = 8192

# Sentinel value users can pass to ``reasoning_parser`` / ``tool_parser`` to
# explicitly disable the parser, overriding any architecture default. The value
# is matched case-insensitively (e.g. ``"none"``, ``"None"``, ``"NONE"``).
DISABLE_PARSER_SENTINEL = "none"


class PipelineRuntimeConfig(ConfigFileModel):
    """Model-agnostic runtime settings for pipeline execution.

    Contains batching, scheduling, and execution configuration that is
    independent of any particular model architecture.
    """

    pipeline_role: PipelineRole = Field(
        default="prefill_and_decode",
        description=(
            "Whether the pipeline should serve both a prefill or decode role or "
            "both."
        ),
    )

    max_batch_size: int | None = Field(
        default=None,
        description=(
            "Maximum batch size to execute with the model. When not specified "
            "(``None``), this value is determined dynamically. For server "
            "launches, set this higher based on server capacity."
        ),
    )

    max_queue_size_tg: int | None = Field(
        default=None,
        description=(
            "Maximum number of requests in decode queue. By default, this is "
            "``max_batch_size``."
        ),
    )

    min_batch_size_tg: int | None = Field(
        default=None,
        description=(
            "Soft floor on the decode batch size. If the TG batch size is "
            "larger, the scheduler continues TG batches; if it falls below, the "
            "scheduler prioritizes CE. This is not a strict minimum. By "
            "default, this is ``max_queue_size_tg``."
        ),
    )

    ep_size: int = Field(
        default=1,
        description=(
            "The expert parallelism size. Needs to be 1 (no expert parallelism) "
            "or the total number of GPUs across nodes."
        ),
    )

    ep_use_allreduce: bool = Field(
        default=False,
        description=(
            "Whether to use allreduce for the cross-device communication in "
            "expert parallelism."
        ),
    )

    eplb_profile: bool = Field(
        default_factory=lambda: os.getenv("MAX_SERVE_EPLB_PROFILE", "").lower()
        in ("1", "true", "yes"),
        description=(
            "When True, enables expert-parallel load balancing (EPLB) MoE "
            "routing histogram profiling in the pipeline. Mirrors "
            "Settings.eplb_profile for pipeline code that doesn't have "
            "access to Settings."
        ),
    )
    ce_delay_ms: float = Field(
        default=0.0,
        description=(
            "Duration of scheduler sleep prior to starting a prefill batch."
        ),
    )

    enable_prioritize_first_decode: bool = Field(
        default=False,
        description=(
            "When enabled, the scheduler always runs a TG batch immediately "
            "after a CE batch with the same requests. This may reduce "
            "time-to-first-chunk latency."
        ),
    )

    enable_chunked_prefill: bool = Field(
        default=True,
        description=(
            "Enable chunked prefill to split context encoding requests into "
            "multiple chunks based on ``max_batch_input_tokens``."
        ),
    )

    enable_in_flight_batching: bool = Field(
        default=False,
        description=(
            "When enabled, prioritizes token generation by batching it with "
            "context encoding requests."
        ),
    )

    max_num_steps: int = Field(
        default=1,
        description=(
            "Deprecated. Multi-step pipeline execution is no longer supported; "
            "the pipeline always runs single-step decode. Values other than "
            "``1`` (including the legacy default ``-1``) are ignored after "
            "logging a warning."
        ),
    )

    max_batch_input_tokens: int = Field(
        default=DEFAULT_MAX_BATCH_INPUT_TOKENS,
        description=(
            "The target number of un-encoded tokens to include in each batch. "
            "This value is used for chunked prefill and memory estimation."
        ),
    )

    use_experimental_kernels: str = Field(
        default=os.environ.get("USE_EXPERIMENTAL_KERNELS", "false"),
        description=(
            "Enables using experimental Mojo kernels with ``max serve``. The "
            "kernels could be unstable or incorrect."
        ),
    )

    use_vendor_blas: str = Field(
        default=os.environ.get("MAX_SERVE_USE_VENDOR_BLAS", "false"),
        description=(
            "Enables using vendor BLAS libraries (``cublas``, ``hipblas``, "
            "etc.) with ``max serve``. Currently, this just replaces "
            "``matmul`` calls."
        ),
    )

    use_vendor_ccl: str = Field(
        default=os.environ.get("MAX_SERVE_USE_VENDOR_CCL", "false"),
        description=(
            "Enables using vendor CCL libraries (NCCL/RCCL) for collective "
            "operations such as allreduce in multi-GPU inference."
        ),
    )

    custom_architectures: list[str] = Field(
        default_factory=list,
        description=(
            "Custom architecture implementations to register. Each input is "
            "either a path to a single custom-architecture module directory "
            "or an ``IMPORT_PATH:MODULE_NAME`` colon-form. Each module must "
            "expose a top-level ``ARCHITECTURES`` list of "
            "``SupportedArchitecture`` instances."
        ),
    )

    execute_empty_batches: bool = Field(
        default=False,
        description=(
            "When enabled, the scheduler runs the model's forward pass even "
            "for an empty batch, so expert-parallel and data-parallel replicas "
            "still reach their collective barrier points; output processing is "
            "skipped. The architecture must support empty batches."
        ),
    )

    max_batch_total_tokens: int | None = Field(
        default=None,
        description=(
            "Ensures the sum of page-aligned context lengths in a batch does "
            "not exceed ``max_batch_total_tokens``. Alignment uses the KV "
            "cache page size. If ``None``, the sum is not limited."
        ),
    )

    device_graph_capture: bool | None = Field(
        default=None,
        description=(
            "Enable device graph capture and replay for graph execution. "
            "If unset, automatically enabled for some selected architectures. "
            "Use ``--no-device-graph-capture`` to explicitly "
            "disable."
        ),
    )

    force: bool = Field(
        default=False,
        description=(
            "Skip validation of user provided flags against the architecture's "
            "required arguments."
        ),
    )

    kvcache_ce_watermark: float = Field(
        default=0.95,
        description=(
            "Projected cache usage threshold for scheduling CE requests, "
            "considering current and incoming requests. CE is scheduled if "
            "either projected usage stays below this threshold or no active "
            "requests exist. Higher values can cause more preemptions."
        ),
    )

    decode_stall_timeout_s: float | None = Field(
        default=float(os.environ["MODULAR_DECODE_STALL_TIMEOUT_S"])
        if "MODULAR_DECODE_STALL_TIMEOUT_S" in os.environ
        else None,
        description=(
            "Seconds of no-batch-activity after which the decode worker exits "
            "to trigger a pod restart. ``None`` (the default) disables the "
            "watchdog. Set with the ``MODULAR_DECODE_STALL_TIMEOUT_S`` environment "
            "variable."
        ),
    )

    decode_request_ttl_s: float | None = Field(
        default=float(os.environ["MODULAR_DECODE_REQUEST_TTL_S"])
        if "MODULAR_DECODE_REQUEST_TTL_S" in os.environ
        else None,
        description=(
            "Per-request TTL in seconds for the decode-side ``prefill_reqs`` "
            "and ``inflight_transfers`` dicts. Entries older than this are "
            "evicted individually (KV blocks released, failure surfaced to "
            "the client) before the stall watchdog fires. ``None`` (the "
            "default) disables eviction. Set with the "
            "``MODULAR_DECODE_REQUEST_TTL_S`` environment variable."
        ),
    )

    enable_overlap_scheduler: bool = Field(
        default=False,
        description=(
            "Whether to enable the overlap scheduler. This feature allows the scheduler "
            "to run alongside GPU execution. This helps improve GPU utilization. "
            "This is an experimental feature which may crash and burn. "
            "This feature will be enabled by default for some selected architectures. "
            "You can forcibly disable this by setting "
            "``--no-enable-overlap-scheduler --force``."
        ),
    )

    allow_unsupported_logprobs: bool = Field(
        default=False,
        description=(
            "When ``True``, OpenAI-compatible requests that ask for "
            "``logprobs`` against a runtime configuration that cannot honor "
            "them will raise a warning, and served as if ``logprobs`` were not "
            "requested. Each response chunk carries ``logprobs: null``. "
            "When ``False`` (default), such requests are rejected with a 400."
        ),
    )

    allow_extra_request_fields: bool = Field(
        default=False,
        description=(
            "When ``True``, unknown top-level fields on OpenAI-compatible "
            "request bodies are dropped with a warning before pydantic "
            " validation, instead of producing a 400."
        ),
    )

    prefer_module_v3: bool = Field(
        default=False,
        description=(
            "Whether to prefer the eager API architecture over the graph API architecture. "
            "When ``False`` (default), the inference server uses the graph API architecture. "
            "When ``True``, the server uses the eager API architecture when available and "
            "falls back to the graph API architecture."
        ),
    )

    reasoning_parser: str | None = Field(
        default=None,
        description=(
            "Name of the reasoning output parser. The parser extracts "
            "thinking blocks to populate the ``reasoning`` field in chat "
            "completion responses. When unset, the server applies the "
            "architecture's default reasoning parser, if any. Pass "
            '``"none"`` (case-insensitive) to explicitly disable reasoning '
            "parsing even when the architecture declares a default."
        ),
    )

    tool_parser: str | None = Field(
        default=None,
        description=(
            "Name of the tool call parser. The parser extracts tool calls "
            "from model output in chat completion responses. When unset, "
            "the server applies the architecture's default tool parser, "
            'if any. Pass ``"none"`` (case-insensitive) to explicitly '
            "disable tool parsing even when the architecture declares a "
            "default."
        ),
    )

    emit_reasoning_content: bool = Field(
        default=False,
        description=(
            "When ``True``, chat completion responses emit a thinking model's "
            "chain-of-thought under ``reasoning_content`` only (``reasoning`` "
            "is omitted). The ``reasoning_content`` alias is used by vLLM, "
            "SGLang, and the DeepSeek API; some clients require it. When "
            "``False`` (default), responses emit reasoning under ``reasoning`` "
            "only."
        ),
    )

    temperature: float | None = Field(
        default=None,
        description=(
            "Default sampling temperature. Controls randomness of token selection—"
            "higher values (e.g. 1.0) produce more random outputs, lower values "
            "(e.g. 0.2) produce more deterministic outputs. When set, this "
            "server-level default applies to all requests that do not explicitly "
            "provide ``temperature``."
        ),
    )

    thinking_temperature: float | None = Field(
        default=None,
        description=(
            "Default temperature override for tokens inside ``<think>...</think>`` "
            "blocks. When set, this server-level default applies to all requests "
            "that do not explicitly provide ``thinking_temperature``. Requires "
            "a reasoning parser to be configured; ignored otherwise."
        ),
    )

    max_vision_cache_entries: int = Field(
        default=256,
        description=(
            "Maximum number of images cached in the vision encoder cache. "
            "Each entry stores the vision encoder output for one image, "
            "avoiding re-encoding across chunks and requests. Set to ``0`` "
            "to disable caching. Only used by VLMs."
        ),
    )

    denoising_cache: DenoisingCacheConfig = Field(
        default_factory=DenoisingCacheConfig,
        description=(
            "Cache configuration for diffusion model denoising "
            "(FBCache, TaylorSeer)."
        ),
    )

    _config_file_section_name: str = PrivateAttr(default="runtime")
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""
