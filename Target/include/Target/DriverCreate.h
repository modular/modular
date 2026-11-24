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
//
// The host side of `M_driver_create`: version negotiation and the failed-create
// contract, as `SDK/include/HAL/M_driver_driver.h` specifies them. Every C++
// loader of a HAL plugin creates its driver through `createDriver`. The Mojo
// HAL implements the same rules in `hal/plugins/__init__.mojo`.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_DRIVERCREATE_H
#define TARGET_DRIVERCREATE_H

#include "HAL/M_driver_api.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/StringRef.h"

namespace M::Driver::Plugin {

/// The driver interface version this build of Modular implements.
inline constexpr M_driver_version hostDriverVersion = {
    M_DRIVER_INTERFACE_VERSION_MAJOR, M_DRIVER_INTERFACE_VERSION_MINOR,
    M_DRIVER_INTERFACE_VERSION_PATCH};

/// Whether Modular, built against `host`, may use a plugin that implements
/// `plugin`, by the rule `M_driver_version` documents.
bool isCompatibleDriverVersion(const M_driver_version &host,
                               const M_driver_version &plugin);

/// Creates a plugin's driver through its `M_driver_create` and accepts it only
/// at a compatible version. On failure, it destroys any handle the plugin
/// returned and the error names `path`, the plugin's message, and for a
/// version mismatch both versions.
ErrorOr<M_driver_driver_t>
createDriver(llvm::StringRef path, M_driver_create_func_ptr_t create,
             M_driver_destroy_func_ptr_t destroy,
             M_driver_status_message_func_ptr_t statusMessage);

} // namespace M::Driver::Plugin

#endif // TARGET_DRIVERCREATE_H
