//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DLOpen.h"

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace M {

void *loadLibrary(const std::string &path) {
#if defined(_WIN32) || defined(_WIN64)
  return LoadLibrary(path.c_str());
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
