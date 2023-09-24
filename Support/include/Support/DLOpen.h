//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DLOPEN_H
#define SUPPORT_DLOPEN_H

#include <string>

namespace M {

/// Open the shared object file from `path` and map it in using
/// RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE | RTLD_DEEPBIND (on *nix systems) and
/// return a handle. The function can be used when
/// llvm::sys::DynamicLibrary::getPermanentLibrary() isn't suitable (due to
/// RTLD_LAZY | RTLD_GLOBAL flags).
void *loadLibrary(const std::string &path);

} // namespace M

#endif // SUPPORT_DLOPEN_H
