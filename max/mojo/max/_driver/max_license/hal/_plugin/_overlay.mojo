# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from ._trait import DefaultHalPlugin
from .cuda import CudaHalPlugin
from .hip import HipHalPlugin
from .metal import MetalHalPlugin
from std._plugin.selector import PluginSelector

comptime HAL_PLUGINS = PluginSelector[
    DefaultHalPlugin, MetalHalPlugin, CudaHalPlugin, HipHalPlugin
]
