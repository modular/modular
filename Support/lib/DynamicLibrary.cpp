//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DynamicLibrary.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/Twine.h"
#if defined(__linux__)
#include <dlfcn.h>
#endif // defined(__linux__)

using namespace M;
using llvm::sys::DynamicLibrary;

ErrorOr<DynamicLibrary>
M::permanentPluginLibrary(const std::filesystem::path &libFilepath,
                          const bool exposeSymbols) {
  std::string errorMessage;
#if defined(__linux__)
  uint64_t dlopenFlags = RTLD_LAZY;
  dlopenFlags |= exposeSymbols ? RTLD_GLOBAL : RTLD_LOCAL;
#if (!(LLVM_ADDRESS_SANITIZER_BUILD || LLVM_THREAD_SANITIZER_BUILD))
  // Only use RTLD_DEEPBIND without sanitizers
  // (https://github.com/google/sanitizers/issues/611).
  dlopenFlags |= RTLD_DEEPBIND;
#endif // !LLVM_ADDRESS_SANITIZER_BUILD

  // TODO(#27162): Upstream dlopen flags to LLVM getPermanentLibrary.
  void *handle = ::dlopen(libFilepath.c_str(), dlopenFlags);
  if (!handle) {
    Dl_info info;
    std::string loader = "unknown binary";
    if (dladdr((void *)&M::permanentPluginLibrary, &info) != 0)
      loader = info.dli_fname;

    return Error(loader + Twine(": encountered errors loading ") +
                 libFilepath.c_str() + Twine(":\n") + ::dlerror());
  }

  auto dylib = DynamicLibrary::addPermanentLibrary(handle, &errorMessage);
#else
  std::string path = libFilepath.string();
  auto dylib = DynamicLibrary::getPermanentLibrary(path.c_str(), &errorMessage);
#endif // defined(__linux__)
  if (!dylib.isValid())
    return Error(errorMessage);

  return dylib;
}
