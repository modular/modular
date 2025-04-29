# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

router = APIRouter(prefix="/v2")


@router.get("/health/live")
async def live() -> Response:
    """Returns server liveness status."""
    return Response()


@router.get("/health/ready")
async def ready() -> Response:
    """Returns server ready status."""
    return Response()


@router.post("/models/{model_name}/versions/{model_version}/infer")
async def infer(
    model_name: str, model_version: str, request: Request
) -> Response:
    """Process a model inference request."""
    await request.json()

    # TODO - parse this request and hand it off.
    print(request)

    return JSONResponse({})
