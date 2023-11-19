//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Utils.h"
#include "Support/CUDA/Constants.h"
#include "llvm/ADT/Twine.h"

using namespace M;
using namespace M::CUDA;

ErrorOr<llvm::sys::DynamicLibrary> M::CUDA::getCUDADriverLibrary() {
  std::string errorMessage;
  static auto library = llvm::sys::DynamicLibrary::getPermanentLibrary(
      kCUDADriverPath.data(), &errorMessage);

  if (library.isValid())
    return library;
  return Error(Twine("failed to load CUDA library: ") + errorMessage);
}
