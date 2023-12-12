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
M::permanentPluginLibrary(const std::filesystem::path &libFilepath) {
  std::string errorMessage;
#if defined(__linux__)
  // TODO(#27162): Upstream dlopen flags to LLVM getPermanentLibrary.
  void *handle =
      ::dlopen(libFilepath.c_str(), RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND);
  if (!handle) {
    return Error(Twine("encountered errors loading ") + libFilepath.c_str() +
                 Twine(":\n") + ::dlerror());
  }

  auto dylib = DynamicLibrary::addPermanentLibrary(handle, &errorMessage);
#else
  std::string path = libFilepath.string();
  auto dylib = DynamicLibrary::getPermanentLibrary(path.c_str(), &errorMessage);
#endif // defined(__linux__)
  if (!errorMessage.empty())
    return Error(errorMessage);

  return dylib;
}
