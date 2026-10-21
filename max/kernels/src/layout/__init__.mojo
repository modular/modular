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
"""Provides layout and tensor types, which abstract memory layout for
multidimensional data.

- The [`Layout`](/api/mojo/layout/layout/Layout) type represents a mapping
  between a set of logical coordinates and a linear index. It can be used, for
  example, to map logical tensor coordinates to a memory address, or to map GPU
  threads to tiles of data.

- The [`TileTensor`](/api/mojo/layout/tile_tensor/TileTensor) type is the
  tensor with explicit memory layout, and
  [`TileTensorIter`](/api/mojo/layout/tile_iterator/TileTensorIter) steps
  through a buffer of equally shaped tiles.
"""
from .coord import (
    _CoordReplaceAt,
    Coord,
    CoordLike,
    ComptimeInt,
    Idx,
    coord,
    coord_to_index_list,
)
from .int_tuple import (
    UNKNOWN_VALUE,
    IntTuple,
    to_index_list,
)
from .layout import Layout, LayoutList, composition, print_layout
from .tile_layout import (
    TensorLayout,
    Layout as MixedLayout,
    RowMajorLayout,
    ColMajorLayout,
    row_major,
    col_major,
)
from .numpy import from_numpy, to_numpy
from .tile_tensor import (
    ImmTileTensor,
    LTToTTLayout,
    MutTileTensor,
    TileTensor,
    flatten_leading,
    stack_allocation,
)
from .tensor_engine import (
    TensorEngine,
    DefaultEngine,
)
from .tmem_engine import (
    TMemEngine,
    TMem16x256bLayout,
    TMem16x256bRegisters,
    TMemStorage,
    TMEM_NUM_COLS,
    TMEM_NUM_LANES,
    tmem_cell,
    tmem_copy_async,
    tmem_16x256b,
)
from ._plugin import (
    CurrentLayoutPlugin,
    CurrentTensorEngine,
    DefaultLayoutPlugin,
    LayoutPluginForTarget,
    LayoutPluginHooks,
    TensorEngineForTarget,
)
