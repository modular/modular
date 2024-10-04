# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import logging
from typing import Optional
import uuid

from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource


# Configure logging to console and OTEL.  This should be called before any
# 3rd party imports whose logging you wish to capture.
def configureLogging(console_level: int, otlp_level: Optional[int] = None):
    logging_handlers: list[logging.Handler] = []

    # Create a console handler
    console_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(console_level)
    logging_handlers.append(console_handler)

    if otlp_level is not None:
        # Create an OTEL handler
        logger_provider = LoggerProvider(
            resource=Resource.create(
                {
                    "event.domain": "serve",
                    "telemetry.session": uuid.uuid4().hex,
                }
            ),
        )
        set_logger_provider(logger_provider)
        exporter = OTLPLogExporter(
            endpoint="https://telemetry-dev.modular.com:443/v1/logs"
        )
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(exporter)
        )
        otlp_handler = LoggingHandler(
            level=otlp_level, logger_provider=logger_provider
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
    logging.getLogger("uvicorn").setLevel(console_level)
    # Explicit levels to reduce noise
    logging.getLogger("sse_starlette.sse").setLevel(
        max(logger_level, logging.INFO)
    )
