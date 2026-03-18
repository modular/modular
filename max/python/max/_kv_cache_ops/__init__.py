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

import mojo.importer
from max.driver import Device

from .kv_cache_ops import (  # type: ignore[import-not-found]
    mha_decode_num_partitions as _mha_decode_num_partitions,
)


def mha_decode_num_partitions(
    batch_size: int,
    max_cache_valid_length: int,
    n_kv_heads: int,
    device: Device,
) -> int:
    """Returns the MHA decode partition count for the given request."""
    return int(
        _mha_decode_num_partitions(
            batch_size,
            max_cache_valid_length,
            n_kv_heads,
            device._device_context_ptr(),
        )
    )


__all__ = ["mha_decode_num_partitions"]
