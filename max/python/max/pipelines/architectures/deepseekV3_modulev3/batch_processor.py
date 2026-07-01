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
"""Input batching for DeepseekV3 ModuleV3 pipeline models."""

from __future__ import annotations

from collections.abc import Sequence

from max.graph import BufferType, DeviceRef, TensorType
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_ragged_kv_symbolic_inputs,
)

from ..deepseekV2_modulev3.batch_processor import (
    DeepseekV2ModuleV3BatchProcessor,
)


class DeepseekV3ModuleV3BatchProcessor(DeepseekV2ModuleV3BatchProcessor):
    """Ragged batching for DeepseekV3 ModuleV3 models.

    EP communication buffers are owned by the pipeline model and appended in
    ``execute()``; this processor prepares the standard ragged token/KV inputs
    only.
    """

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
        extra_input_types: Sequence[TensorType | BufferType] = (),
    ) -> list[TensorType | BufferType]:
        """Returns ModuleV3 symbolic inputs plus optional EP buffer types."""
        return [
            *modulev3_ragged_kv_symbolic_inputs(
                kv_params=kv_params,
                device_refs=device_refs,
            ),
            *extra_input_types,
        ]
