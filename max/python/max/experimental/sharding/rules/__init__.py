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

"""Placement propagation rules for distributed ops."""

from ._common import RuleSignature
from .conv import conv2d_rule, conv2d_transpose_rule, conv3d_rule
from .elementwise import (
    binary_rule,
    linear_binary_rule,
    linear_unary_rule,
    ternary_rule,
    unary_rule,
)
from .matmul import matmul_rule
from .misc import (
    as_interleaved_complex_rule,
    band_part_rule,
    buffer_store_slice_rule,
    cond_rule,
    fold_rule,
    irfft_rule,
    reject_distributed_rule,
    resize_linear_rule,
    resize_rule,
    while_loop_rule,
)
from .norm import normalization_rule
from .pooling import linear_pool_rule, pool_rule
from .reduction import linear_reduce_rule, reduce_rule
from .shape import (
    argsort_rule,
    broadcast_to_rule,
    chunk_rule,
    flatten_rule,
    gather_nd_rule,
    gather_rule,
    masked_scatter_rule,
    nonzero_rule,
    outer_rule,
    pad_rule,
    passthrough_rule,
    permute_rule,
    repeat_interleave_rule,
    reshape_rule,
    same_placement_multi_input_rule,
    scatter_add_rule,
    scatter_nd_add_rule,
    scatter_nd_rule,
    scatter_rule,
    slice_tensor_rule,
    split_rule,
    squeeze_rule,
    stack_rule,
    tile_rule,
    top_k_rule,
    transpose_rule,
    unsqueeze_rule,
)
