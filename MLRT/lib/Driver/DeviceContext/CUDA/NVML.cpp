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
// This file provides access to the NVML API.
//
//===----------------------------------------------------------------------===//

#include "NVML.h"
#include "Support/DLFcn.h"
#include "Support/PlatformUtils.h"
#include "llvm/Support/raw_ostream.h"
#include <stddef.h>

namespace M::Driver {

#if MODULAR_LINUX
const char nvmlLibName[] = "libnvidia-ml.so.1";
#else
const char nvmlLibName[] = "";
#endif

ErrorOr<std::unique_ptr<NVML>> NVML::allocAndInitNVML() {
  // Attempt to open the NVML shared library.
  void *nvmlLib = dlopen(nvmlLibName, RTLD_NOW);
  if (nvmlLib == nullptr) {
    if (strcmp(nvmlLibName, "") == 0) {
      return Error("NVML library name is not defined for this platform");
    }
    return Error(llvm::Twine("Failed to open library \"") + nvmlLibName + "\"");
  }

  auto nvml = std::make_unique<NVML>();

#define NVML_FUNC(name, ...)                                                   \
  nvml->name##_ = reinterpret_cast<t##name *>(dlsym(nvmlLib, #name));          \
  if (nvml->name##_ == nullptr) {                                              \
    return Error(llvm::Twine("Failed to find symbol \"") + #name +             \
                 "\" in \"" + nvmlLibName + "\"");                             \
  }

#define NVML_FUNC_NO_PARAMS(name, ret)                                         \
  nvml->name##_ = reinterpret_cast<t##name *>(dlsym(nvmlLib, #name));          \
  if (nvml->name##_ == nullptr) {                                              \
    return Error(llvm::Twine("Failed to find symbol \"") + #name +             \
                 "\" in \"" + nvmlLibName + "\"");                             \
  }

#include "NVML.inc"
#undef NVML_FUNC_NO_PARAMS
#undef NVML_FUNC

  auto result = nvml->nvmlInit_v2();
  if (result != SUCCESS) {
    return Error(llvm::Twine("Failed to initialize NVML: ") +
                 std::to_string(result));
  }

  return nvml;
}

NVML::~NVML() { nvmlShutdown(); }

ErrorOr<std::string> NVML::GetDriverVersion() const {
  CUDA_NVML_CALL_LOG();

  // The maximum version buffer size for the NVML system driver version.
  constexpr int nvmlSystemDriverVersionBufferSize = 80;
  char buffer[nvmlSystemDriverVersionBufferSize] = {};

  auto result =
      nvmlSystemGetDriverVersion(buffer, nvmlSystemDriverVersionBufferSize);
  if (result != SUCCESS) {
    return Error(llvm::Twine("Failed to get driver version: ") +
                 nvmlErrorString(result));
  }

  // Construct from C string so the std::string's size matches the version
  // length rather than the buffer capacity. Otherwise the trailing NULs in
  // the buffer become embedded NULs in the std::string and silently truncate
  // any downstream C-string conversion (e.g. when the version appears inside
  // an error message that is later passed through a const char* boundary).
  return std::string(buffer);
}

ErrorOr<const NVML &> NVML::get() {
  static ErrorOr<std::unique_ptr<NVML>> errOrNVML = allocAndInitNVML();

  if (errOrNVML) {
    return Error(errOrNVML.getError());
  }
  return *errOrNVML.get();
}

} // namespace M::Driver
