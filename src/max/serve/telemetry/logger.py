# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import logging
import os
import uuid
from typing import Any

from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource


# Configure logging to console and OTEL.  This should be called before any
# 3rd party imports whose logging you wish to capture.
def configureLogging(console_level, otlp_level):
    # Create a log formatter
    log_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(name)s: %(message)s", "%H:%M:%S"
    )

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(console_level)

    # Configure the root logger, all children will bubble up
    logger = logging.getLogger()
    logger.addHandler(console_handler)
    logger.setLevel(logging.NOTSET)

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
        otlp_handler.setFormatter(log_formatter)

        logger.addHandler(otlp_handler)
        logger.setLevel(otlp_level)

    # TODO use FastAPIInstrumentor once Motel supports traces.
    # For now, manually configure uvicorn.
    logging.getLogger("uvicorn").setLevel(console_level)
