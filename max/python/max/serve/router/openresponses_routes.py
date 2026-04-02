# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""OpenResponses API route handlers.

This module provides a clean implementation of the OpenResponses API standard
without inheriting technical debt from other API endpoints.

Spec: https://www.openresponses.org/reference
"""

from __future__ import annotations

import logging
from http import HTTPStatus

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from max.interfaces import OpenResponsesRequest
from max.interfaces.generation import GenerationOutput
from max.interfaces.provider_options import GeneratedMediaResponseFormat
from max.interfaces.request.open_responses import (
    OutputContent,
    OutputImageContent,
    OutputVideoContent,
    ResponseResource,
)
from max.serve.dependencies import create_request_parser
from max.serve.media import (
    GeneratedMediaStore,
    StoredMediaAsset,
    encode_video_bytes_b64,
)
from starlette.datastructures import State

router = APIRouter(prefix="/v1")
logger = logging.getLogger("max.serve")

# Create a reusable dependency for parsing OpenResponses requests
ParseOpenResponsesRequest = Depends(create_request_parser(OpenResponsesRequest))


@router.post("/responses")
async def create_response(
    request: Request,
    open_responses_request: OpenResponsesRequest = ParseOpenResponsesRequest,
) -> JSONResponse:
    """Create a response using the OpenResponses API schema.

    This endpoint provides a clean implementation of the OpenResponses
    standard for generating responses from AI models.

    Args:
        request: The incoming FastAPI request containing OpenResponses data.
        open_responses_request: Parsed and validated OpenResponses request
            (automatically injected via dependency injection). Validation
            includes checking that streaming is not requested.

    Returns:
        A JSONResponse with the generated response data.

    Raises:
        HTTPException: If request parsing or validation fails, including
            if streaming is requested (not currently supported).
    """

    # Request is already parsed and validated via dependency injection
    # (including validation that streaming is not requested)
    logger.debug(
        "OpenResponses request parsed successfully - "
        "request_id=%s, model=%s, stream=%s",
        open_responses_request.request_id.value,
        open_responses_request.body.model,
        open_responses_request.body.stream,
    )
    _validate_requested_model(
        request=request,
        requested_model=open_responses_request.body.model,
    )

    # Generate response using the GeneralPipelineHandler from app state
    logger.debug("Starting response generation")

    # Get the first chunk from the handler (raises StopAsyncIteration if empty)
    generator = request.app.state.handler.next(open_responses_request)
    final_output = await anext(generator)
    logger.debug(
        "Received chunk - is_done=%s, status=%s",
        final_output.is_done,
        final_output.final_status,
    )

    # Continue consuming chunks until we get is_done=True
    if not final_output.is_done:
        async for chunk in generator:
            logger.debug(
                "Received chunk - is_done=%s, status=%s",
                chunk.is_done,
                chunk.final_status,
            )
            final_output = chunk
            if chunk.is_done:
                break

    final_output = _persist_generated_media(
        request=request,
        open_responses_request=open_responses_request,
        final_output=final_output,
    )

    # Convert GenerationOutput to ResponseResource format
    response = ResponseResource.from_generation_output(
        final_output, model=open_responses_request.body.model
    )

    logger.debug(
        "Returning response for request_id=%s",
        open_responses_request.request_id.value,
    )
    return JSONResponse(
        content=response.model_dump(mode="json", exclude_none=True),
        status_code=HTTPStatus.OK,
    )


@router.get("/images/{image_id}/content", name="get_generated_image_content")
async def get_generated_image_content(
    request: Request, image_id: str
) -> FileResponse:
    asset = _require_asset(
        _get_media_store(request).get_image(image_id),
        kind="image",
        asset_id=image_id,
    )
    return FileResponse(
        path=asset.path,
        media_type=asset.media_type,
        filename=asset.filename,
    )


@router.get("/videos/{video_id}/content", name="get_generated_video_content")
async def get_generated_video_content(
    request: Request, video_id: str
) -> FileResponse:
    asset = _require_asset(
        _get_media_store(request).get_video(video_id),
        kind="video",
        asset_id=video_id,
    )
    return FileResponse(
        path=asset.path,
        media_type=asset.media_type,
        filename=asset.filename,
    )


def _persist_generated_media(
    request: Request,
    open_responses_request: OpenResponsesRequest,
    final_output: GenerationOutput,
) -> GenerationOutput:
    media_store = getattr(request.app.state, "media_store", None)
    if not isinstance(media_store, GeneratedMediaStore):
        return final_output

    video_options = open_responses_request.body.provider_options.video
    if (
        video_options is not None
        and final_output.output
        and all(
            isinstance(content, OutputImageContent)
            for content in final_output.output
        )
    ):
        video_response_format = video_options.response_format
        frames = [
            content
            for content in final_output.output
            if isinstance(content, OutputImageContent)
        ]
        if video_response_format == GeneratedMediaResponseFormat.b64_json:
            video_bytes = media_store.encode_video_frames(
                frame_contents=frames,
                frames_per_second=video_options.frames_per_second or 16,
            )
            return final_output.model_copy(
                update={
                    "output": [
                        OutputVideoContent(
                            video_data=encode_video_bytes_b64(video_bytes),
                            format="mp4",
                            frames_per_second=video_options.frames_per_second
                            or 16,
                            num_frames=len(frames),
                        )
                    ]
                }
            )

        video_asset = media_store.save_video_frames(
            frame_contents=frames,
            frames_per_second=video_options.frames_per_second or 16,
        )
        video_url = str(
            request.url_for(
                "get_generated_video_content",
                video_id=video_asset.asset_id,
            )
        )
        return final_output.model_copy(
            update={
                "output": [
                    OutputVideoContent(
                        video_url=video_url,
                        format="mp4",
                        frames_per_second=video_options.frames_per_second or 16,
                        num_frames=len(frames),
                    )
                ]
            }
        )

    image_options = open_responses_request.body.provider_options.image
    image_response_format = (
        image_options.response_format
        if image_options is not None
        else GeneratedMediaResponseFormat.url
    )
    if image_response_format == GeneratedMediaResponseFormat.b64_json:
        return final_output

    persisted_output: list[OutputContent] = []
    for content in final_output.output:
        if not isinstance(content, OutputImageContent):
            persisted_output.append(content)
            continue

        image_asset = media_store.save_image_content(content)
        image_url = str(
            request.url_for(
                "get_generated_image_content",
                image_id=image_asset.asset_id,
            )
        )
        persisted_output.append(
            content.model_copy(
                update={
                    "image_url": image_url,
                    "image_data": None,
                }
            )
        )

    return final_output.model_copy(update={"output": persisted_output})


def _get_media_store(request: Request) -> GeneratedMediaStore:
    media_store = getattr(request.app.state, "media_store", None)
    if not isinstance(media_store, GeneratedMediaStore):
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Generated media storage is not available.",
        )
    return media_store


def _require_asset(
    asset: StoredMediaAsset | None, kind: str, asset_id: str
) -> StoredMediaAsset:
    if asset is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"Unknown generated {kind}: {asset_id}",
        )
    return asset


def _validate_requested_model(request: Request, requested_model: str) -> None:
    app_state: State = request.app.state
    handler = getattr(app_state, "handler", None)
    pipeline_config = getattr(app_state, "pipeline_config", None)

    served_models: list[str] = []
    if handler is not None and getattr(handler, "model_name", None):
        served_models.append(handler.model_name)

    if (
        pipeline_config is not None
        and getattr(pipeline_config, "model", None) is not None
        and getattr(pipeline_config.model, "model_name", None)
    ):
        served_models.append(pipeline_config.model.model_name)

    served_models = list(dict.fromkeys(served_models))
    if not served_models:
        return

    if requested_model not in served_models:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=(
                f"Unknown model '{requested_model}', currently serving "
                f"'{served_models}'."
            ),
        )
