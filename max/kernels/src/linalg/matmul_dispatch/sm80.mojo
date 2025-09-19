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
from hashlib import default_comp_time_hasher
from utils import IndexList

from ..utils_gpu import (
    MatmulConfig,
)


fn create_matmul_configs_ampere[
    key: String, a_type: DType, b_type: DType, c_type: DType, transpose_b: Bool
]() -> MatmulConfig[a_type, b_type, c_type, transpose_b]:
    var dict = get_dispatch_table[a_type, b_type, c_type, transpose_b]()
    try:
        return dict[key]
    except error:
        return MatmulConfig[a_type, b_type, c_type, transpose_b](
            num_pipeline_stages=0,
        )  # 128x128_4


fn get_dispatch_table[
    a_type: DType, b_type: DType, c_type: DType, transpose_b: Bool
]() -> Dict[
    String,
    MatmulConfig[a_type, b_type, c_type, transpose_b],
    default_comp_time_hasher,
]:
    var tile_configs = Dict[
        String,
        MatmulConfig[a_type, b_type, c_type, transpose_b],
        default_comp_time_hasher,
    ]()

    # TODO(PAQ-1284):
    #   The configs below were optimal than the
    #   default config, leading to performance regressions that were previously
    #   masked by a bug. Retune and update before using them again. Return an
    #   empty dict in the meantime.
    #
    # The old code is in https://github.com/modularml/modular/pull/69274
    #
    return tile_configs^
