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
from max.serve.worker_interface.zmq_queue import generate_zmq_ipc_path
from pydantic import Field, PrivateAttr

from .config.config_enums import PipelineRole
from .interfaces.cache_mixin import DenoisingCacheConfig

# Default max batch input tokens for chunked prefill and memory estimation.
DEFAULT_MAX_BATCH_INPUT_TOKENS = 8192


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
            "default, this is ``max_queue_size_tg``. Experimental for the TTS "
            "scheduler."
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

    ce_delay_ms: float = Field(
        default=0.0,
        description=(
            "Duration of scheduler sleep prior to starting a prefill batch. "
            "Experimental for the TTS scheduler."
        ),
    )

    enable_prioritize_first_decode: bool = Field(
        default=False,
        description=(
            "When enabled, the scheduler always runs a TG batch immediately "
            "after a CE batch with the same requests. This may reduce "
            "time-to-first-chunk latency. Experimental for the TTS scheduler."
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
        default=-1,
        description=(
            "The number of steps to run for multi-step scheduling. ``-1`` "
            "specifies a default value based on configuration and platform. "
            "Ignored for models which are not auto-regressive (for example, "
            "embedding models)."
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
            "Custom architecture implementations to register. Each input can "
            "either be a raw module name or an import path followed by a colon "
            "and the module name. Each module must expose an ``ARCHITECTURES`` list "
            "of architectures to register."
        ),
    )

    zmq_endpoint_base: str = Field(
        default_factory=generate_zmq_ipc_path,
        description=(
            "Prefix for ZMQ endpoints used for IPC. This ensures unique "
            "endpoints across MAX Serve instances on the same host. Example: "
            "``lora_request_zmq_endpoint = "
            'f"{zmq_endpoint_base}-lora_request"``.'
        ),
    )

    execute_empty_batches: bool = Field(
        default=False,
        description="Whether the scheduler should execute empty batches.",
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
            "completion responses."
        ),
    )

    tool_parser: str | None = Field(
        default=None,
        description=(
            "Name of the tool call parser. The parser extracts tool calls "
            "from model output in chat completion responses."
        ),
    )

    # TODO(SERVSYS-1096): Remove this field once we've reworked how required
    # config fields are validated.
    defer_resolve: bool = Field(
        default=False,
        description="Whether to defer resolving the pipeline config.",
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
            "(FBCache, TaylorSeer, TeaCache)."
        ),
    )

    _config_file_section_name: str = PrivateAttr(default="runtime")
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""
