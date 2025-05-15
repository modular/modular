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
"""Implements the algorithm package."""

from .functional import (
    BinaryTile1DTileUnitFunc,
    Dynamic1DTileUnitFunc,
    Dynamic1DTileUnswitchUnitFunc,
    Static1DTileUnitFunc,
    Static1DTileUnitFuncWithFlags,
    Static1DTileUnswitchUnitFunc,
    Static2DTileUnitFunc,
    SwitchedFunction,
    SwitchedFunction2,
    elementwise,
    map,
    parallelize,
    parallelize_over_rows,
    stencil,
    stencil_gpu,
    sync_parallelize,
    tile,
    tile_and_unswitch,
    tile_middle_unswitch_boundaries,
    unswitch,
    vectorize,
)
from .memory import parallel_memcpy
from .reduction import (
    all_true,
    any_true,
    cumsum,
    map_reduce,
    max,
    mean,
    min,
    none_true,
    product,
    reduce,
    reduce_boolean,
    sum,
    variance,
)
