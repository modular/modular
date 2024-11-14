# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from opentelemetry import trace

tracer = trace.get_tracer("max.serve")
