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

const char *CodedErrorOrSuccess::getComponentAsString() const {
  const char *retStr;
  switch (component) {
#define X(val)                                                                 \
  case CodedErrorComponent::val:                                               \
    retStr = #val;                                                             \
    break;
    ERROR_COMPONENT_EXPR
#undef X
  }
  return retStr;
}
