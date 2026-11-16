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
"""Constructs the registered Nemotron-H DSpark draft."""

from max.dtype import DType
from max.pipelines.lib.registration_draft import DraftRegistration
from max.tensor import default_dtype

from .dspark_qwen3 import DSparkQwen3
from .model_config import DSparkQwen3Config, configure_dspark_draft


class DSparkQwen3Draft(DSparkQwen3):
    """Builds the existing draft computation from its finalized config."""

    def __init__(self, config: DSparkQwen3Config) -> None:
        devices = [ref.to_device() for ref in config.kv_params.devices]
        with default_dtype(DType.bfloat16):
            super().__init__(
                config,
                config.kv_params,
                num_devices=len(devices),
                max_seq_len=config.max_seq_len,
                device=devices[0],
            )


dspark_draft_registration = DraftRegistration[
    DSparkQwen3Draft, DSparkQwen3Config
](method="dspark", configure=configure_dspark_draft)
