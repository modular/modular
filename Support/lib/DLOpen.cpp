//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DLOpen.h"

#if defined(_WIN32) || defined(_WIN64)
#include "llvm/Support/DynamicLibrary.h"
#else
#include <dlfcn.h>
#endif // defined(_WIN32) || defined(_WIN64)

namespace M {

void *loadLibrary(const std::string &path) {
#if defined(_WIN32) || defined(_WIN64)
  // Use `llvm::sys::getPermanentLibrary()` on Windows where the default export
  // problem that `RTLD_DEEPBIND` fixes does not exist.
  llvm::sys::DynamicLibrary dylib =
      llvm::sys::DynamicLibrary::getPermanentLibrary(path.c_str());
  if (!dylib.isValid())
    return nullptr;

  return dylib.getOSSpecificHandle();
#else
  int flags = RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE;
#if defined(__linux__)
  // `RTLD_DEEPBIND` is specific to Linux.
  // MacOS behaves as if `RTLD_DEEPBIND` is always set by default.
  flags |= RTLD_DEEPBIND;
#endif // defined(__linux__)
  return dlopen(path.c_str(), flags);
#endif // defined(_WIN32) || defined(_WIN64)
}

} // namespace M
