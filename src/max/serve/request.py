# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import logging
import uuid
from time import perf_counter_ns
from typing import Callable

from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import StopWatch

from fastapi import FastAPI, HTTPException, Request, Response

logger = logging.getLogger(__name__)


def register_request(app: FastAPI):
    @app.middleware("http")
    async def request_session(request: Request, call_next: Callable):
        with StopWatch() as requestTimer:
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            request.state.recv_time_ns = perf_counter_ns()
            try:
                response: Response = await call_next(request)
                status_code = response.status_code
            except HTTPException as e:
                status_code = e.status_code
                raise e
            except Exception as e:
                logger.exception(
                    "Exception in request session : %s", request_id
                )
                status_code = 500
                raise HTTPException(
                    status_code=500,
                    headers={"X-Request-ID": request_id},
                ) from e
            finally:
                METRICS.requestCount(status_code, request.url.path)
                METRICS.requestTime(requestTimer.elapsed_ms, request.url.path)
        response.headers["X-Request-ID"] = request_id
        return response
