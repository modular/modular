# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""APIs to serve MAX models and graphs."""

from .scheduler import queues

from .router import kserve_routes
from .router import openai_routes
