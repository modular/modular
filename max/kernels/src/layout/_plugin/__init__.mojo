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
"""Target-keyed selection of the `TensorEngine` that backs a tile.

`TileTensor` is parameterized on a `TensorEngine` so generic code can be
written against the engine the target uses:

```mojo
from layout import CurrentTensorEngine, TensorEngine, TensorLayout, TileTensor

def generic[
    L: TensorLayout, Engine: TensorEngine = CurrentTensorEngine[L]
](tile: TileTensor[_, L, _, Engine=Engine]):
    ...
```

The default build resolves to `DefaultEngine`. A hardware plugin implements
`LayoutPluginHooks`, registers it in `_overlay.mojo`, and the same source picks
up its engines.
"""

from ._impl import (
    CurrentLayoutPlugin,
    CurrentTensorEngine,
    DeviceGlobalAddressSpace,
    DeviceTensorEngine,
    LayoutPluginForTarget,
    TensorEngineForTarget,
)
from ._overlay import LAYOUT_PLUGINS
from ._trait import DefaultLayoutPlugin, LayoutPluginHooks
