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
// This file implements the CUDA compilation driver.
//
//===----------------------------------------------------------------------===//

#include "CUDACompilationDriver.h"

#include "Support/BazelRunfiles.h"
#include "Support/DLFcn.h"
#include "Support/ErrorOr.h"
#include "Support/PlatformUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

namespace M::Driver {

#if MODULAR_LINUX
const char nvptxLibName[] = "libNVPTX.so";
#else
const char nvptxLibName[] = "";
#endif

namespace {

/// Opens the NVPTX compiler library.
///
/// Unlike libcuda / libnvidia-ml, which the NVIDIA driver installs into the
/// loader's search path, libNVPTX.so is one of our own artifacts — it wraps
/// NVIDIA's static libnvptxcompiler archive, which ships no shared object of
/// its own. Nothing links against it any more either, so no RPATH points at
/// it and a bare name does not resolve. It has to be located explicitly.
void *openNVPTXLibrary() {
  if (auto path = llvm::sys::Process::GetEnv("MODULAR_NVPTX_LIBRARY_PATH"))
    return dlopen(path->c_str(), RTLD_NOW | RTLD_LOCAL);

  // Under Bazel the library is staged in runfiles, whose layout only the
  // runfiles library can decode.
  if (auto runfilesPath = M::findConfigWithRunfiles("mojo-max.nvptx_path"))
    if (void *lib = dlopen(runfilesPath->c_str(), RTLD_NOW | RTLD_LOCAL))
      return lib;

  // Installed layouts place it alongside the other shared libraries. mingw
  // has no dladdr, and on Windows this function is unreachable anyway
  // (nvptxLibName is empty), so the branch is fenced rather than emulated.
#ifndef _WIN32
  Dl_info info;
  if (dladdr(reinterpret_cast<const void *>(&openNVPTXLibrary), &info) != 0 &&
      info.dli_fname != nullptr) {
    llvm::SmallString<128> moduleDir(info.dli_fname);
    llvm::sys::path::remove_filename(moduleDir);
    for (llvm::StringRef subdir : {".", "../lib"}) {
      llvm::SmallString<128> candidate(moduleDir);
      llvm::sys::path::append(candidate, subdir, nvptxLibName);
      if (void *lib = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL))
        return lib;
    }
  }
#endif // _WIN32

  // Last resort: let the loader search (LD_LIBRARY_PATH, ldconfig).
  return dlopen(nvptxLibName, RTLD_NOW | RTLD_LOCAL);
}

} // namespace

ErrorOr<std::unique_ptr<CUDACompilationDriver>>
CUDACompilationDriver::allocAndInitDriver() {
  if (nvptxLibName[0] == '\0')
    return Error("NVPTX compiler library is not available on this platform");

  void *nvptxLib = openNVPTXLibrary();
  if (nvptxLib == nullptr)
    return Error(llvm::Twine("Failed to open library \"") + nvptxLibName +
                 "\". Set MODULAR_NVPTX_LIBRARY_PATH to its location, or "
                 "MODULAR_NVPTX_COMPILER_PATH to an external ptxas binary.");

  auto driver = std::make_unique<CUDACompilationDriver>();

#define NVPTX_FUNC(name, ...)                                                  \
  driver->name##_ = reinterpret_cast<t##name *>(dlsym(nvptxLib, #name));       \
  if (driver->name##_ == nullptr) {                                            \
    return Error(llvm::Twine("Failed to find symbol \"") + #name +             \
                 "\" in \"" + nvptxLibName + "\"");                            \
  }

#include "CUDACompilationDriver.inc"
#undef NVPTX_FUNC

  return driver;
}

ErrorOr<const CUDACompilationDriver &> CUDACompilationDriver::get() {
  static ErrorOr<std::unique_ptr<CUDACompilationDriver>> errOrDriver =
      allocAndInitDriver();

  if (errOrDriver.isError()) {
    return Error(errOrDriver.getError());
  }
  return *errOrDriver.get();
}

} // namespace M::Driver
