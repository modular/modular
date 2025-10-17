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

from __future__ import annotations

from max.pipelines.core import TextContext
from max.pipelines.lib import LoRAManager


def can_allocate_lora_request(
    ctx: TextContext, active_loras: set[str], lora_manager: LoRAManager | None
) -> bool:
    """Checks if the LoRA request can be allocated and serviced"""
    # This should only be called when lora_manager exists
    assert lora_manager is not None

    # Non-LoRA requests can always be allocated
    if not lora_manager.is_lora(ctx.model_name):
        return True

    # LoRA requests can be allocated if:
    # - Already in active set for this batch (no additional LoRA slot needed)
    # - There's room for more LoRAs in this batch
    return (
        ctx.model_name in active_loras
        or lora_manager.is_active_lora(ctx.model_name)
        or len(active_loras) < lora_manager.max_num_loras
    )


def is_lora(ctx: TextContext, lora_manager: LoRAManager | None) -> bool:
    """Helper that checks the manager is not None and if the context is a lora"""
    return bool(lora_manager and lora_manager.is_lora(ctx.model_name))


def is_active_lora(ctx: TextContext, lora_manager: LoRAManager | None) -> bool:
    """Helper that checks the manager is not None and if the LoRA is active"""
    return bool(lora_manager and lora_manager.is_active_lora(ctx.model_name))
