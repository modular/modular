# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import logging
import uuid
from typing import Callable

from fastapi import FastAPI, Request

logger = logging.getLogger(__name__)


def register_request(app: FastAPI):
    @app.middleware("http")
    async def request_session(request: Request, call_next: Callable):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
