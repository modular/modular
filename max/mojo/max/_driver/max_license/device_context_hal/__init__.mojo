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
"""The HAL `DeviceContext` compat layer.

Compiles in every configuration as `max._driver.max_license.device_context_hal`.
Under --config=hal-device-context, `_dcc_overlay.mojo` replaces
`max/gpu/host/_overlay.mojo`, so `max.gpu.host` resolves to the DeviceContext
defined here.
"""
