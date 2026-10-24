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
// The two names one accelerator goes by, and the translation between them.
//
// A device API is named after its runtime ("cuda", "hip"); the Mojo
// `--target-accelerator` vendor is named after the LLVM backend ("nvidia",
// "amdgpu"). Both vocabularies meet inside this layer --
// `Device::getAcceleratorArchOrEmpty()` reports a vendor-prefixed arch built
// out of an API name, and `CompilationDevice::create` takes either -- so the
// translation lives here once instead of in each TU that happens to need it.
//
// Header-only and dependency-free on purpose: `CompilerDeviceArch.cpp`
// provides `getAcceleratorArchOrEmpty` for the compiler family, which exists
// precisely so that family links no DeviceContext runtime at all.
//
// Spelled as literals rather than the `Device::*API` constants because this
// header is included by the headers that declare them. `DeviceAPINamesTest`
// in HALVirtualDeviceTest.cpp pins the two in agreement, in both backends.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_DEVICECONTEXT_DEVICEAPINAMES_H
#define MLRT_DRIVER_DEVICECONTEXT_DEVICEAPINAMES_H

#include "Support/LLVMForwardDecls.h"

#include "llvm/ADT/StringRef.h"

namespace M::Driver {

/// The Mojo `--target-accelerator` vendor name for device API `api`, for
/// building a `vendor:arch` accelerator string. An API with no distinct vendor
/// name -- "metal", and any plugin API -- is returned unchanged.
inline StringRef apiToVendorPrefix(StringRef api) {
  if (api == "cuda")
    return "nvidia";
  if (api == "hip")
    return "amdgpu";
  return api;
}

/// The device API name for `api`, which may already be one or may be the Mojo
/// `--target-accelerator` vendor name for the same device. Lets a caller
/// holding a target-accelerator value pass it through unchanged. "metal" needs
/// no entry: its two names already coincide. "amd" is accepted because
/// `--target-accelerator` accepts it too (see
/// `BuiltinTargets.normalize_target_arch` in
/// Mojo/stdlib/std/_gpu/host/_builtin_targets.mojo).
inline StringRef canonicalDeviceAPI(StringRef api) {
  if (api == "nvidia")
    return "cuda";
  if (api == "amdgpu" || api == "amd")
    return "hip";
  return api;
}

} // namespace M::Driver

#endif // MLRT_DRIVER_DEVICECONTEXT_DEVICEAPINAMES_H
