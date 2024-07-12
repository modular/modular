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
/// If exposeSymbols is true, the library will be loaded with RTLD_GLOBAL
/// exposing its symbols to any other loaded object.
ErrorOr<llvm::sys::DynamicLibrary>
permanentPluginLibrary(const std::filesystem::path &libFilepath,
                       bool exposeSymbols = false);

} // namespace M

#endif // SUPPORT_DYNAMICLIBRARY_H
