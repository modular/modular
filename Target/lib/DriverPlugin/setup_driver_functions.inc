//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
// SDK HAL API functions (types defined in SDK/include/HAL/M_driver_*.h)
SETUP_FUNC(status_message)
SETUP_FUNC(create)
SETUP_FUNC(destroy)
SETUP_FUNC(property)
SETUP_FUNC(device_count)
// Plugin-specific extensions (types defined locally in DriverPlugin.cpp)
SETUP_FUNC(device_get_native)
SETUP_FUNC(device_arch_name_primary)
SETUP_FUNC(create_compilation_device)
SETUP_FUNC(destroy_compilation_device)
