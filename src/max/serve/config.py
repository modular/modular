# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""
Placeholder file for any configs (runtime, models, pipelines, etc)
"""

import socket
from enum import Enum
from typing import Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIType(Enum):
    KSERVE = "kserve"
    OPENAI = "openai"


class RunnerType(Enum):
    PYTORCH = "pytorch"
    TOKEN_GEN = "token_gen"


class Settings(BaseSettings):
    # env files, direct initialization, and aliases interact in some confusing
    # ways.  this is the way:
    #   1. extra="forbid"
    #   2. populate_by_name=False
    #   3. initialize with alias names `Settings(MAX_SERVE_HOST="host")`
    #
    # If there is an alias, there is ambiguity in which name to use when giving
    # a setting a valuie. (alias or attribute name on Settings?).
    # extra-"forbid" ensures that misspellings of a setting are always errors.
    # populate_by_name=False ensures that only the alias value is used to
    # initialized.  In the presence of both a .env value & intializer value,
    # pydantic is able to resolve both declarations to the same settings &
    # correctly evaluate priority.

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="forbid",
        populate_by_name=False,
    )

    # Server configuration
    api_types: list[APIType] = Field(
        description="List of exposed API types.", default=[APIType.OPENAI]
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
            except socket.error as e:
                raise ValueError(f"port {port} is already in use") from e

    # Telemetry and logging configuration
    logs_console_level: str = Field(
        default="INFO",
        description="Logging level",
        alias="MAX_SERVE_LOGS_CONSOLE_LEVEL",
    )
    logs_otlp_level: str = Field(
        default="INFO",
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

    runner_type: RunnerType = Field(
        description="Type of execution runner.",
        default=RunnerType.PYTORCH,
    )


def api_prefix(settings: Settings, api_type: APIType):
    return "/" + str(api_type) if len(settings.api_types) > 1 else ""
