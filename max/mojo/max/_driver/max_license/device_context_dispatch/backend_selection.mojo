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
"""The `DeviceContext` backend this `max` package is built for."""

comptime HAL_DEVICE_CONTEXT = False
"""Whether this `max` package uses the HAL `DeviceContext`.

The HAL `max` packages stack `//max/mojo/max/_driver/max_license/hal_backend`
over this module, which sets it to `True`. The package carries the choice, so
every program that imports it matches the runtime it ships with, including
programs that `mojo` builds outside Bazel.
"""
