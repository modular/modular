//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/ErrorOr.h"
#include "Support/Error.h"
#include "Support/LogicalResult.h"
#include "llvm/Support/Error.h"
#include <utility>

using namespace M;

ErrorOrSuccess M::toModularErrorOr(llvm::Error error) {
  if (error)
    return Error(llvm::toString(std::move(error)));
  return success();
}
