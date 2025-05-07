# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
