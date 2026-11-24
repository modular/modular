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
"""Lists the target collections and the name-keyed MAX plugins of a build.

Overlay builds shadow this file to add their own collection and plugin.
"""

from std._plugin.selector import PluginSelector
from std._gpu.host.info import TargetAcceleratorCollection

from ._trait import DefaultTargets, MaxPlugin
from .cuda.targets import CudaTargets
from .hip.targets import HipTargets
from .metal.targets import MetalTargets

# Builtin targets carry their MAX plugin in their payload, so this lists none
# of them. It serves overlay collections defined in stdlib packages, which
# cannot name MAX types.
comptime MAX_PLUGINS = PluginSelector[Trait=MaxPlugin]

# `DefaultTargets` must stay first: it holds the "no accelerator" placeholder
# that target-list consumers skip by position.
comptime TARGETS = TypeList.of[
    Trait=TargetAcceleratorCollection,
    DefaultTargets,
    CudaTargets,
    HipTargets,
    MetalTargets,
]()
