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

from .._trait import HalPluginHooks
from ..._machine import DeviceRef
from ...execution_config.gpu_execution_config import GPUExecutionConfiguration
from ...extensions.cuda import CUDAHALExtensions


# Hopper, datacenter Blackwell and Thor support thread block clusters.
comptime _CLUSTER_TARGETS: List[StaticString] = [
    "nvidia:sm_90a",
    "nvidia:sm_100a",
    "nvidia:sm_103a",
    "nvidia:sm_110a",
]


struct CudaHalPluginHooks(HalPluginHooks):
    comptime ExecutionConfigType[device: DeviceRef] = GPUExecutionConfiguration[
        has_cluster_dim=device.spec.name in _CLUSTER_TARGETS,
        has_launch_attributes=True,
        has_constant_memory=True,
    ]
    """Returns the execution configuration for the given device."""

    comptime extensions = CUDAHALExtensions
