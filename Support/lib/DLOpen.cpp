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
  return dlopen(path.c_str(),
                RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE | RTLD_DEEPBIND);
#endif
}

} // namespace M
