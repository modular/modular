//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ErrorOr.h"
#include "llvm/Support/Error.h"

using namespace M;

ErrorOrSuccess M::toModularErrorOr(llvm::Error error) {
  if (error)
    return Error(llvm::toString(std::move(error)));
  return success();
}
