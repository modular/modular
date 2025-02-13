# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import uuid
from typing import Callable

from fastapi import FastAPI, HTTPException, Request, Response
from max.loggers import get_logger
from max.serve.telemetry.stopwatch import StopWatch

logger = get_logger(__name__)


def register_request(app: FastAPI):
    @app.middleware("http")
    async def request_session(request: Request, call_next: Callable):
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        request.state.request_timer = StopWatch()
        try:
            response: Response = await call_next(request)
            status_code = response.status_code
        except HTTPException as e:
            status_code = e.status_code
            raise e
        except Exception as e:
            logger.exception("Exception in request session : %s", request_id)
            status_code = 500
            raise HTTPException(
                status_code=500,
                headers={"X-Request-ID": request_id},
            ) from e
        response.headers["X-Request-ID"] = request_id
        return response
