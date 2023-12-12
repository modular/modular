//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DYNAMICLIBRARY_H
#define SUPPORT_DYNAMICLIBRARY_H

#include "Support/ForwardDecls.h"
#include "llvm/Support/DynamicLibrary.h"
#include <filesystem>

namespace M {

/// Opens a permanent `DynamicLibrary` handle to a plugin library.
/// Differs from llvm::DynamicLibrary::getPermanentLibrary in symbol resolution:
/// `permanentPluginLibrary()` guarantees symbols first resolve locally then
/// globally, while LLVM's `getPermanentLibrary()` does not.
ErrorOr<llvm::sys::DynamicLibrary>
permanentPluginLibrary(const std::filesystem::path &libFilepath);

} // namespace M

#endif // SUPPORT_DYNAMICLIBRARY_H
