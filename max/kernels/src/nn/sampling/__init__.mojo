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
"""Token-sampling kernels: penalties, top-k/top-p masking and sampling.

`topk_fi` holds the single-block kernels, which serve every target;
`topk_fi_cluster` holds thread-block-cluster variants of the hot spec-decode
kernels for NVIDIA SM90+ devices. The dispatchers in this file pick between
them by device, so callers import from this package and never choose a
launch shape themselves.
"""

from max.gpu.host import DeviceContext
from std.sys.info import _has_sm_9x_or_newer
from layout import ComptimeInt, Coord, TensorLayout, TileTensor
from layout.tile_layout import Layout

from .sampling import (
    apply_penalties_to_logits,
    update_frequency_data,
)
from .topk_fi import (
    topk_mask_logits,
    topk_sampling_from_prob,
    topk_softmax_sample,
    topk_topp_sampling_from_prob,
)
from .topk_fi import topk_topp_masked_probs as _topk_topp_masked_probs_single
from .topk_fi_cluster import topk_topp_masked_probs_cluster


def topk_topp_masked_probs[
    dtype: DType,
    block_size: Int = 1024,
    TopKArrLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    TopPArrLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    TemperatureLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    ProbsLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64, Int64].element_types,
        stride_types=Coord[Int64, ComptimeInt[1]].element_types,
    ],
](
    ctx: DeviceContext,
    logits: TileTensor[mut=False, dtype, ...],
    probs: TileTensor[DType.float32, ProbsLayoutType, MutAnyOrigin],
    top_k_val: Int,
    top_p_val: Float32 = 1.0,
    top_k_arr: Optional[
        TileTensor[DType.int64, TopKArrLayoutType, ImmutAnyOrigin]
    ] = None,
    top_p_arr: Optional[
        TileTensor[DType.float32, TopPArrLayoutType, ImmutAnyOrigin]
    ] = None,
    temperature: Optional[
        TileTensor[DType.float32, TemperatureLayoutType, ImmutAnyOrigin]
    ] = None,
) raises:
    """Computes per-row top-k/top-p masked softmax.

    Dispatches by device: NVIDIA SM90+ takes the cluster-capable launcher,
    every other target takes the single-block one. The branch is comptime,
    so a target without thread-block clusters never instantiates the cluster
    kernels.

    See `topk_fi.topk_topp_masked_probs` for the parameters, arguments and
    output contract; both sides share them.
    """
    comptime if _has_sm_9x_or_newer():
        topk_topp_masked_probs_cluster[dtype, block_size](
            ctx,
            logits,
            probs,
            top_k_val,
            top_p_val,
            top_k_arr,
            top_p_arr,
            temperature,
        )
    else:
        _topk_topp_masked_probs_single[dtype, block_size](
            ctx,
            logits,
            probs,
            top_k_val,
            top_p_val,
            top_k_arr,
            top_p_arr,
            temperature,
        )
