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


"""Pipeline cli utilities."""

from .config import (
    config_to_flag,
    get_default,
    get_field_type,
    is_flag,
    is_multiple,
    is_optional,
    pipeline_config_options,
    validate_field_type,
)
from .device_options import DevicesOptionType
from .encode import pipeline_encode
from .generate import generate_text_for_pipeline, stream_text_to_console
from .list import list_pipelines_to_console, list_pipelines_to_json
from .metrics import TextGenerationMetrics
from .serve import serve_api_server_and_model_worker

__all__ = [
    "DevicesOptionType",
    "TextGenerationMetrics",
    "config_to_flag",
    "generate_text_for_pipeline",
    "get_default",
    "get_field_type",
    "is_flag",
    "is_multiple",
    "is_optional",
    "list_pipelines_to_console",
    "list_pipelines_to_json",
    "pipeline_config_options",
    "pipeline_encode",
    "serve_api_server_and_model_worker",
    "stream_text_to_console",
    "validate_field_type",
]
