# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import logging
import os
import platform
import uuid
from dataclasses import dataclass
from typing import Union

from max.serve.config import Settings
from opentelemetry.sdk.resources import Resource

otelBaseUrl = "https://telemetry.modular.com:443"


def _getCloudProvider() -> str:
    providers = ["amazon", "google", "microsoft", "oracle"]
    path = "/sys/class/dmi/id/"
    if os.path.isdir(path):
        for idFile in os.listdir(path):
            try:
                with open(idFile, "r") as file:
                    contents = file.read().lower()
                    for provider in providers:
                        if provider in contents:
                            return provider
            except Exception:
                pass
    return ""


def _getGPUInfo() -> str:
    try:
        import torch  # type: ignore

        device_properties = torch.cuda.get_device_properties(0)
        return f"{torch.cuda.device_count()}:{device_properties.total_memory}:{device_properties.name}"
    except Exception:
        return ""


logs_resource = Resource.create(
    {
        "event.domain": "serve",
        "telemetry.session": uuid.uuid4().hex,
        "enduser.id": os.environ.get("MODULAR_USER_ID", ""),
        "os.type": platform.system(),
        "os.version": platform.release(),
        "cpu.description": platform.processor(),
        "cpu.arch": platform.architecture()[0],
        # MAGIC-55: disable gpu info for now
        # Because it initilizes the CUDA driver in the API process
        # while we initialize models in the Model worker process
        # CUDA doesn't like it and crashes.
        # "system.gpu": _getGPUInfo(),
        "system.cloud": _getCloudProvider(),
        "deployment.id": os.environ.get("MAX_SERVE_DEPLOYMENT_ID", ""),
    }
)

metrics_resource = Resource.create(
    {
        "enduser.id": os.environ.get("MODULAR_USER_ID", ""),
        "deployment.id": os.environ.get("MAX_SERVE_DEPLOYMENT_ID", ""),
    }
)


@dataclass
class TelemetryConfig:
    """Telemetry Configuration"""

    console_level: Union[int, str] = logging.INFO
    file_path: Union[str, None] = None
    file_level: Union[int, str, None] = None
    otlp_level: Union[int, str, None] = None
    metrics_egress_enabled: bool = False
    async_metrics: bool = True
    egress_enabled: bool = False

    @classmethod
    def from_config(cls, config: Settings) -> "TelemetryConfig":
        """Read the telemetry config from env variables"""
        otlp_level: Union[int, str, None] = logging.getLevelName(
            config.logs_otlp_level
        )

        metrics_egress_enabled = True

        if config.disable_telemetry:
            otlp_level = None
            metrics_egress_enabled = False

        return cls(
            console_level=config.logs_console_level,
            file_path=config.logs_file_path,
            file_level=config.logs_file_level,
            otlp_level=otlp_level,
            metrics_egress_enabled=metrics_egress_enabled,
        )
