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

"""Shared types for LoRA queue operations."""

from __future__ import annotations

from enum import Enum

import msgspec

LORA_REQUEST_ENDPOINT = "lora_request"
LORA_RESPONSE_ENDPOINT = "lora_response"


class LoRAType(Enum):
    """Enumeration for LoRA Types."""

    A = "lora_A"
    """Represents the LoRA A matrix (high rank tensor to low rank tensor)."""

    B = "lora_B"
    """Represents the LoRA B matrix. (low rank tensor to high rank tensor)"""

    B_KV = "lora_B_kv"
    """Represents the combined K and V LoRA B matrices for QKV fusion."""

    BIAS = "lora.bias"
    """Represents the LoRA bias matrix. (added to matrix B)"""


class LoRAOperation(Enum):
    """Enum for different LoRA operations."""

    LOAD = "load"
    UNLOAD = "unload"


class LoRAStatus(Enum):
    """Enum for LoRA operation status."""

    SUCCESS = "success"
    LOAD_NAME_EXISTS = "load_name_exists"
    UNLOAD_NAME_NONEXISTENT = "unload_name_nonexistent"
    LOAD_ERROR = "load_error"
    UNLOAD_ERROR = "unload_error"
    LOAD_INVALID_PATH = "load_invalid_path"
    LOAD_INVALID_ADAPTER = "load_invalid_adapter"
    UNSPECIFIED_ERROR = "unspecified_error"


class LoRARequest(msgspec.Struct, omit_defaults=True):
    """Container for LoRA adapter requests."""

    operation: LoRAOperation
    lora_name: str
    lora_path: str | None = None


class LoRAResponse(msgspec.Struct):
    """Response from LoRA operations."""

    status: LoRAStatus
    message: str | list[str]
