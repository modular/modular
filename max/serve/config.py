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

"""
Placeholder file for any configs (runtime, models, pipelines, etc)
"""

from __future__ import annotations

import logging
from enum import Enum, IntEnum
from pathlib import Path

from max.serve.kvcache_agent.dispatcher_factory import DispatcherConfig
from max.serve.queue.zmq_queue import generate_zmq_ipc_path
from max.support.human_readable_formatter import to_human_readable_bytes
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("max.serve")


class APIType(Enum):
    KSERVE = "kserve"
    OPENAI = "openai"
    SAGEMAKER = "sagemaker"


class RunnerType(Enum):
    PYTORCH = "pytorch"
    TOKEN_GEN = "token_gen"


class MetricLevel(IntEnum):
    """Metric levels in increasing granularity"""

    # no metrics
    NONE = 0
    # basic api-worker and model worker metrics. minimal performance impact.
    BASIC = 10
    # high detail metrics. may impact performance
    DETAILED = 20


class MetricRecordingMethod(Enum):
    """How should metrics be recorded?"""

    # Do not record metrics
    NOOP = "NOOP"
    # Synchronously record metrics
    SYNC = "SYNC"
    # Record metrics asynchronously using asyncio
    ASYNCIO = "ASYNCIO"
    # Send metric observations to a separate process for recording
    PROCESS = "PROCESS"


class Settings(BaseSettings):
    # env files, direct initialization, and aliases interact in some confusing
    # ways.  this is the way:
    #   1. extra="allow"
    #      This allows .env files to include entries for non-modular use cases.  eg HF_TOKEN
    #   2. populate_by_name=True
    #      Allow both field names and aliases to be used for initialization, but aliases are preferred for clarity.
    #   3. initialize with alias names `Settings(MAX_SERVE_HOST="host")`
    #
    # Known sharp edges:
    #   1. .env files can use both the Settings attr name (eg host) as well as the alias MAX_SERVE_HOST.
    #   2. Environment variables can only use the alias (MAX_SERVE_...)
    #   3. Explicit overrides can only use the alias (Settings(MAX_SERVE_HOST=...)
    #   4. Explicit overrides using the wrong name silently do nothing (Settings(host=...)) has no effect.

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="", extra="allow", populate_by_name=True
    )

    # Server configuration
    api_types: list[APIType] = Field(
        description="List of exposed API types.",
        default=[APIType.OPENAI, APIType.SAGEMAKER],
    )
    offline_inference: bool = Field(
        description="If True, the server is run in offline inference mode. While it will still spin up workers, it will not spin up the API endpoint or use an HTTP port.",
        default=False,
        alias="MAX_SERVE_OFFLINE_INFERENCE",
    )
    headless: bool = Field(
        default=False,
        description="If True, runs a model worker and dispatch worker without starting an API server.",
        alias="MAX_SERVE_HEADLESS",
    )
    host: str = Field(
        description="Hostname to use", default="0.0.0.0", alias="MAX_SERVE_HOST"
    )
    port: int = Field(
        description="Port to use", default=8000, alias="MAX_SERVE_PORT"
    )

    metrics_port: int = Field(
        description="Port to use for the metrics endpoint",
        default=8001,
        alias="MAX_SERVE_METRICS_ENDPOINT_PORT",
    )

    # File URI configuration
    allowed_image_roots: list[str] = Field(
        description="List of allowed root directories for file:// URI access",
        default_factory=list,
        alias="MAX_SERVE_ALLOWED_IMAGE_ROOTS",
    )
    max_local_image_bytes: int = Field(
        description="Maximum size in bytes for local image files accessed via file:// URIs",
        default=20 * 1024 * 1024,  # 20MiB
        alias="MAX_SERVE_MAX_LOCAL_IMAGE_BYTES",
    )

    # Telemetry and logging configuration
    logs_console_level: str | None = Field(
        default="INFO",
        description="Logging level",
        alias="MAX_SERVE_LOGS_CONSOLE_LEVEL",
    )
    logs_otlp_level: str | None = Field(
        default=None,
        description="OTLP log level",
        alias="MAX_SERVE_LOGS_OTLP_LEVEL",
    )
    logs_file_level: str | None = Field(
        default=None,
        description="File log level",
        alias="MAX_SERVE_LOGS_FILE_LEVEL",
    )
    logs_file_path: str | None = Field(
        default=None,
        description="Logs file path",
        alias="MAX_SERVE_LOGS_FILE_PATH",
    )
    structured_logging: bool = Field(
        default=False,
        description="Structured logging for deployed services",
        alias="MODULAR_STRUCTURED_LOGGING",
    )
    logs_enable_components: str | None = Field(
        default=None,
        description="Comma separated list of additional components to enable for logging",
        alias="MAX_SERVE_LOGS_ENABLE_COMPONENTS",
    )

    disable_telemetry: bool = Field(
        default=False,
        description="Disable remote telemetry",
        alias="MAX_SERVE_DISABLE_TELEMETRY",
    )

    # Model worker configuration
    use_heartbeat: bool = Field(
        default=False,
        description="When True, uses a periodic heart beat to confirm model worker liveness. This can result in false negatives if a single batch takes longer than the heartbeat interval to process (as may be the case for large context prefill)",
        alias="MAX_SERVE_USE_HEARTBEAT",
    )
    mw_timeout_s: float = Field(
        default=20 * 60.0, description="", alias="MAX_SERVE_MW_TIMEOUT"
    )
    mw_health_fail_s: float = Field(
        # TODO: we temporarily set it to 1 minute to handle long context input
        default=60.0,
        description="Maximum time to wait for a heartbeat & remain healthy.  This should be longer than ITL",
        alias="MAX_SERVE_MW_HEALTH_FAIL",
    )

    telemetry_worker_spawn_timeout: float = Field(
        default=60.0,
        description="Amount of time in seconds to wait for the telemetry worker to spawn and turn healthy",
        alias="MAX_SERVE_TELEMETRY_WORKER_SPAWN_TIMEOUT",
    )

    metric_recording: MetricRecordingMethod = Field(
        default=MetricRecordingMethod.PROCESS,
        description="How metrics should be recorded?",
        alias="MAX_SERVE_METRIC_RECORDING_METHOD",
    )

    metric_level: MetricLevel = Field(
        default=MetricLevel.BASIC,
        description="Determines the level of detail in the metrics emitted. Metrics tagged at a higher level will be dropped. This does nothing if metric recording is disabled.",
        alias="MAX_SERVE_METRIC_LEVEL",
    )

    detailed_metric_buffer_factor: int = Field(
        default=20,
        description="How many detailed metrics to buffer before sending them to the telemetry worker",
        alias="MAX_SERVE_DETAILED_METRIC_BUFFER_FACTOR",
    )

    @field_validator("metric_level", mode="before")
    def validate_metric_level(cls, value: str | MetricLevel) -> MetricLevel:
        # Support string values ("BASIC") even though Metric is an IntEnum
        if isinstance(value, str):
            return MetricLevel[value]
        return value

    transaction_recording_file: Path | None = Field(
        default=None,
        description="File to record all HTTP transactions to",
        alias="MAX_SERVE_TRANSACTION_RECORDING_FILE",
    )

    @field_validator("transaction_recording_file", mode="after")
    def validate_transaction_recording_file(
        cls, path: Path | None
    ) -> Path | None:
        if path is None:
            return None
        if not path.name.endswith(".rec.jsonl"):
            raise ValueError(
                "Transaction recording files must have a '.rec.jsonl' file extension."
            )
        return path

    transaction_recording_include_responses: bool = Field(
        default=False,
        description="When recording HTTP transactions, whether to include responses",
        alias="MAX_SERVE_TRANSACTION_RECORDING_INCLUDE_RESPONSES",
    )

    kv_cache_events_zmq_endpoint: str = Field(
        default_factory=generate_zmq_ipc_path,
        description="Expose KV Cache Events ZMQ Socket for communication between the KV Cache Agent and MAX Serve",
        alias="MAX_SERVE_KV_CACHE_EVENTS_ZMQ_ENDPOINT",
    )

    dispatcher_config: DispatcherConfig = Field(
        default_factory=DispatcherConfig,
        description="Expose Dispatcher Config for use in inter-node communication.",
        alias="MAX_SERVE_DISPATCHER_CONFIG",
    )

    log_prefix: str | None = Field(
        default=None,
        description="Prefix to prepend to all log messages for this service instance.",
        alias="MAX_SERVE_LOG_PREFIX",
    )

    def log_server_info(self) -> None:
        """Log comprehensive server configuration information.

        Displays all server settings in a consistent visual format similar to
        pipeline configuration logging.
        """
        # Build API types string
        api_types_str = ", ".join(api_type.value for api_type in self.api_types)

        # Build operation mode string
        mode_flags = []
        if self.offline_inference:
            mode_flags.append("offline_inference")
        if self.headless:
            mode_flags.append("headless")
        mode_str = ", ".join(mode_flags) if mode_flags else "standard"

        # Build allowed roots string
        allowed_roots_str = (
            ", ".join(self.allowed_image_roots)
            if self.allowed_image_roots
            else "None"
        )

        # Log Server Configuration
        logger.info("")
        logger.info("Server Config")
        logger.info("=" * 60)
        logger.info(f"    host                   : {self.host}")
        logger.info(f"    port                   : {self.port}")
        logger.info(f"    metrics_port           : {self.metrics_port}")
        logger.info(f"    api_types              : {api_types_str}")
        logger.info(f"    operation_mode         : {mode_str}")
        logger.info("")

        # File System Configuration
        logger.info("File System Config")
        logger.info("=" * 60)
        logger.info(f"    allowed_image_roots    : {allowed_roots_str}")
        logger.info(
            f"    max_local_image_bytes  : {to_human_readable_bytes(self.max_local_image_bytes)}"
        )
        logger.info("")

        # Metrics and Telemetry Configuration
        logger.info("Metrics and Telemetry Config")
        logger.info("=" * 60)
        logger.info(
            f"    metric_recording       : {self.metric_recording.value}"
        )
        logger.info(
            f"    metric_level           : {self.metric_level.name} ({self.metric_level.value})"
        )
        logger.info(
            f"    detailed_buffer_factor : {self.detailed_metric_buffer_factor}"
        )
        logger.info(f"    disable_telemetry      : {self.disable_telemetry}")

        # Transaction recording (part of telemetry)
        if self.transaction_recording_file:
            logger.info(
                f"    transaction_recording  : {self.transaction_recording_file}"
            )
            logger.info(
                f"    include_responses      : {self.transaction_recording_include_responses}"
            )
        else:
            logger.info("    transaction_recording  : None")
        logger.info("")

        # Model Worker Configuration
        logger.info("Model Worker Config")
        logger.info("=" * 60)
        logger.info(f"    use_heartbeat          : {self.use_heartbeat}")
        logger.info(f"    timeout                : {self.mw_timeout_s:.1f}s")
        logger.info(
            f"    health_fail_timeout    : {self.mw_health_fail_s:.1f}s"
        )
        logger.info(
            f"    telemetry_spawn_timeout: {self.telemetry_worker_spawn_timeout:.1f}s"
        )
        logger.info("")
