//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DLOPEN_H
#define SUPPORT_DLOPEN_H

#include <string>

namespace M {

/// Opens the shared object file from `path` and maps it in using
/// RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE | RTLD_DEEPBIND (on *nix systems).
/// Returns a handle to the opened shared object file.
///
/// On *nix the function is useful when
/// `llvm::sys::DynamicLibrary::getPermanentLibrary()` isn't suitable due to
/// RTLD_LAZY | RTLD_GLOBAL flags.
/// On Windows this function calls
/// `llvm::sys::DynamicLibrary::getPermanentLibrary()`.
void *loadLibrary(const std::string &path);

} // namespace M

#endif // SUPPORT_DLOPEN_H
