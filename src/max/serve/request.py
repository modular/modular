# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import logging
import uuid
from time import perf_counter_ns
from typing import Callable

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_async.aio import time
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

REQ_TIME = Histogram("req_time_seconds", "Time spent in requests", ["path"])
REQ_COUNTER = Counter("req_count", "Http request count", ["code", "path"])


def register_request(app: FastAPI):
    @app.middleware("http")
    async def request_session(request: Request, call_next: Callable):
        with REQ_TIME.labels(request.url.path).time():
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
                REQ_COUNTER.labels(status_code, request.url.path).inc()
            response.headers["X-Request-ID"] = request_id
            return response
