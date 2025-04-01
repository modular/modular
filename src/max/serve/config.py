# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""
Placeholder file for any configs (runtime, models, pipelines, etc)
"""

import socket
from enum import Enum, IntEnum
from pathlib import Path
from typing import Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    # Send metric observations to a seperate process for recording
    PROCESS = "PROCESS"


class Settings(BaseSettings):
    # env files, direct initialization, and aliases interact in some confusing
    # ways.  this is the way:
    #   1. extra="allow"
    #      This allows .env files to include entries for non-modular use cases.  eg HF_TOKEN
    #   2. populate_by_name=False
    #      Reduce the number of ways a setting can be spelled so it is easier to reason about priority among env vars, .env, and explicit settings.
    #   3. initialize with alias names `Settings(MAX_SERVE_HOST="host")`
    #
    # Known sharp edges:
    #   1. .env files can use both the Settings attr name (eg host) as well as the alias MAX_SERVE_HOST.
    #   2. Environment variables can only use the alias (MAX_SERVE_...)
    #   3. Explicit overrides can only use the alias (Settings(MAX_SERVE_HOST=...)
    #   4. Explicit overrides using the wrong name silently do nothing (Settings(host=...)) has no effect.

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="allow",
        populate_by_name=False,
    )

    # Server configuration
    api_types: list[APIType] = Field(
        description="List of exposed API types.",
        default=[APIType.OPENAI, APIType.SAGEMAKER],
    )
    host: str = Field(
        description="Hostname to use", default="0.0.0.0", alias="MAX_SERVE_HOST"
    )
    port: int = Field(
        description="Port to use", default=8000, alias="MAX_SERVE_PORT"
    )

    @field_validator("port")
    def validate_port(cls, port: int):
        # check if port is already in use
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("", port))
                return port
            except OSError as e:
                raise ValueError(f"port {port} is already in use") from e

    # Telemetry and logging configuration
    logs_console_level: str = Field(
        default="INFO",
        description="Logging level",
        alias="MAX_SERVE_LOGS_CONSOLE_LEVEL",
    )
    logs_otlp_level: Union[str, None] = Field(
        default=None,
        description="OTLP log level",
        alias="MAX_SERVE_LOGS_OTLP_LEVEL",
    )
    logs_file_level: Union[str, None] = Field(
        default=None,
        description="File log level",
        alias="MAX_SERVE_LOGS_FILE_LEVEL",
    )
    logs_file_path: Union[str, None] = Field(
        default=None,
        description="Logs file path",
        alias="MAX_SERVE_LOGS_FILE_PATH",
    )
    structured_logging: bool = Field(
        default=False,
        description="Structured logging for deployed services",
        alias="MODULAR_STRUCTURED_LOGGING",
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
        default=20 * 60.0,
        description="",
        alias="MAX_SERVE_MW_TIMEOUT",
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
        default=MetricRecordingMethod.ASYNCIO,
        description="How metrics should be recorded?",
        alias="MAX_SERVE_METRIC_RECORDING_METHOD",
    )

    metric_level: MetricLevel = Field(
        default=MetricLevel.BASIC,
        description="",
        alias="MAX_SERVE_METRIC_LEVEL",
    )

    @field_validator("metric_level", mode="before")
    def validate_metric_level(
        cls, value: Union[str, MetricLevel]
    ) -> MetricLevel:
        # Support string values ("BASIC") even though Metric is an IntEnum
        if isinstance(value, str):
            return MetricLevel[value]
        return value

    transaction_recording_file: Optional[Path] = Field(
        default=None,
        description="File to record all HTTP transactions to",
        alias="MAX_SERVE_TRANSACTION_RECORDING_FILE",
    )

    @field_validator("transaction_recording_file", mode="after")
    def validate_transaction_recording_file(
        cls, path: Optional[Path]
    ) -> Optional[Path]:
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

    experimental_enable_kvcache_agent: bool = Field(
        default=False,
        description="Experimental: Enable KV Cache Agent support.",
        alias="MAX_SERVE_EXPERIMENTAL_ENABLE_KVCACHE_AGENT",
    )


def api_prefix(settings: Settings, api_type: APIType):
    return "/" + str(api_type) if len(settings.api_types) > 1 else ""
