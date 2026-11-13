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
"""Backend-dispatching `DeviceContext` and its buffers.

Each struct here forwards to the legacy or the HAL implementation, whichever
`use_legacy_backend` names for the target. Signatures follow the legacy
backend where the two disagree, because that is what existing call sites are
written against.
"""
