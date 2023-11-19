//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CUDA_UTILS_H
#define SUPPORT_CUDA_UTILS_H

#include "Support/ErrorOr.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/DynamicLibrary.h"

namespace M::CUDA {
ErrorOr<llvm::sys::DynamicLibrary> getCUDADriverLibrary();

template <typename SymbolTy>
ErrorOr<SymbolTy> fallibleGetSymbol(llvm::sys::DynamicLibrary &dylib,
                                    llvm::StringRef symbolName) {

  static_assert(std::is_pointer_v<SymbolTy>, "Should be a pointer type");

  void *symbolOr = dylib.getAddressOfSymbol(symbolName.data());
  if (!symbolOr)
    return Error(Twine("failed to get symbol '") + symbolName +
                 "' from the CUDA library");
  return reinterpret_cast<SymbolTy>(symbolOr);
}

} // namespace M::CUDA
#endif // SUPPORT_CUDA_UTILS_H
