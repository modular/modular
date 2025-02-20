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
from time import time
from typing import Union

import requests
from max.serve.config import Settings
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from pythonjsonlogger import jsonlogger

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


def _getWebUserId() -> str:
    try:
        idFile = os.path.expanduser("~") + "/.modular/webUserId"
        with open(idFile) as file:
            return file.readline().rstrip("\n")
    except Exception:
        return ""


logs_resource = Resource.create(
    {
        "event.domain": "serve",
        "telemetry.session": uuid.uuid4().hex,
        "web.user.id": _getWebUserId(),
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

    @classmethod
    def from_config(cls, config: Settings) -> "TelemetryConfig":
        """Read the telemetry config from env variables"""
        return cls(
            console_level=config.logs_console_level,
            file_path=config.logs_file_path,
            file_level=config.logs_file_level,
            otlp_level=config.logs_otlp_level,
            metrics_egress_enabled=not config.disable_telemetry,
        )


# Configure logging to console and OTEL.  This should be called before any
# 3rd party imports whose logging you wish to capture.
def configure_logging(server_settings: Settings) -> None:
    default_config = TelemetryConfig.from_config(server_settings)
    console_level = default_config.console_level
    file_path = default_config.file_path
    file_level = default_config.file_level
    otlp_level = default_config.otlp_level
    egress_enabled = default_config.metrics_egress_enabled

    logging_handlers: list[logging.Handler] = []

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_formatter: logging.Formatter
    if os.getenv("MODULAR_STRUCTURED_LOGGING") == "1":
        console_formatter = jsonlogger.JsonFormatter()
    else:
        console_formatter = logging.Formatter(
            (
                "%(asctime)s.%(msecs)03d %(levelname)s: %(process)d %(threadName)s:"
                " %(name)s: %(message)s"
            ),
            datefmt="%H:%M:%S",
        )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(console_level)
    logging_handlers.append(console_handler)

    if file_level is not None and file_path is not None:
        # Create a file handler
        file_handler = logging.FileHandler(file_path)
        file_formatter: logging.Formatter
        if os.getenv("MODULAR_STRUCTURED_LOGGING") == "1":
            file_formatter = jsonlogger.JsonFormatter()
        else:
            file_formatter = logging.Formatter(
                (
                    "%(asctime)s.%(msecs)03d %(levelname)s: %(process)d %(threadName)s:"
                    " %(name)s: %(message)s"
                ),
                datefmt="%y:%m:%d-%H:%M:%S",
            )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(file_level)
        logging_handlers.append(file_handler)

    if egress_enabled and otlp_level is not None:
        # Create an OTEL handler
        logger_provider = LoggerProvider(logs_resource)
        set_logger_provider(logger_provider)
        exporter = OTLPLogExporter(endpoint=otelBaseUrl + "/v1/logs")
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(exporter)
        )
        otlp_handler = LoggingHandler(
            level=logging.getLevelName(otlp_level),
            logger_provider=logger_provider,
        )
        logging_handlers.append(otlp_handler)

    # Configure root logger level
    logger_level = min(h.level for h in logging_handlers)
    logger = logging.getLogger()
    logger.setLevel(logger_level)
    for handler in logging_handlers:
        logger.addHandler(handler)

    # TODO use FastAPIInstrumentor once Motel supports traces.
    # For now, manually configure uvicorn.
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    # Explicit levels to reduce noise
    logging.getLogger("sse_starlette.sse").setLevel(
        max(logger_level, logging.INFO)
    )
    logger.info(
        "Logging initialized: Console: %s, File: %s, Telemetry: %s",
        console_level,
        file_level,
        egress_enabled and otlp_level,
    )


def configure_metrics(server_settings: Settings):
    default_config = TelemetryConfig.from_config(server_settings)
    metrics_egress_enabled = default_config.metrics_egress_enabled

    meterProviders: list[MetricReader] = [PrometheusMetricReader(True)]
    if metrics_egress_enabled:
        meterProviders.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=otelBaseUrl + "/v1/metrics")
            )
        )
    set_meter_provider(MeterProvider(meterProviders, metrics_resource))


# Send a simple one-time structured log, avoiding the buggy OTEL SDK
# (see MAXSERV-904)
def send_telemetry_log(model_name: str):
    request_body = f"""{{
  "resourceLogs": [
    {{
      "resource": {{
        "attributes": [
          {{"key": "model", "value": {{"stringValue": "{model_name}"}}}},
          {{"key": "web.user.id", "value": {{"stringValue": "{logs_resource.attributes["web.user.id"]}"}}}},
          {{"key": "enduser.id", "value": {{"stringValue": "{logs_resource.attributes["enduser.id"]}"}}}},
          {{"key": "deployment.id", "value": {{"stringValue": "{logs_resource.attributes["deployment.id"]}"}}}},
          {{"key": "os.type", "value": {{"stringValue": "{logs_resource.attributes["os.type"]}"}}}},
          {{"key": "os.version", "value": {{"stringValue": "{logs_resource.attributes["os.version"]}"}}}},
          {{"key": "cpu.description", "value": {{"stringValue": "{logs_resource.attributes["cpu.description"]}"}}}},
          {{"key": "cpu.arch", "value": {{"stringValue": "{logs_resource.attributes["cpu.arch"]}"}}}},
          {{"key": "system.cloud", "value": {{"stringValue": "{logs_resource.attributes["system.cloud"]}"}}}},
          {{"key": "service.name", "value": {{"stringValue": "unknown_service"}}}},
          {{"key": "telemetry.sdk.language", "value": {{"stringValue": "python"}}}},
          {{"key": "telemetry.sdk.version", "value": {{"stringValue": "0.0.0"}}}},
          {{"key": "telemetry.sdk.name", "value": {{"stringValue": "opentelemetry"}}}}
        ]
      }},
      "scopeLogs": [
        {{
          "logRecords": [
            {{
              "attributes": [
                {{"key": "event.domain", "value": {{"stringValue": "modular"}}}},
                {{"key": "event.name", "value": {{"stringValue": "serve.telemetry.log"}}}}
              ],
              "body": {{"stringValue": ""}},
              "observedTimeUnixNano": "{int(time() * 1_000_000_000)}",
              "severityNumber": 9,
              "severityText": "INFO"
            }}
          ],
          "scope": {{"name": "modular_logger"}}
        }}
      ]
    }}
  ]
}}"""

    requests.post(
        otelBaseUrl + "/v1/logs",
        data=request_body,
        headers={"Content-Type": "application/json"},
        timeout=2,
    )
