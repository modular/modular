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

from typing import Protocol, runtime_checkable

from max.graph import TensorValue


@runtime_checkable
class SupportsLoRA(Protocol):
    """Base class for supporting LoRA functionality in Modules"""

    def set_lora_batch_info(
        self,
        lora_ids: TensorValue,
        lora_ranks: TensorValue,
    ) -> None: ...

    def apply_lora(self, *args, **kwargs) -> None: ...
