//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/ErrorTree.h"
#include "mlir/IR/Diagnostics.h"

using namespace M;
using namespace KGEN;

ErrorTree::ErrorTree(Location loc, Error error, ErrorTree causes)
    : loc(loc), error(std::move(error)) {
  addCause(std::move(causes));
}

ErrorTree::ErrorTree(Location loc, Error error,
                     MutableArrayRef<ErrorTree> causes)
    : loc(loc), error(std::move(error)) {
  addCauses(causes);
}

ErrorTree ErrorTree::copy() const {
  ErrorTree copy(loc, error.copy());
  copy.causes.reserve(causes.size());
  for (const ErrorTree &cause : causes)
    copy.causes.push_back(cause.copy());
  return copy;
}

void ErrorTree::emit(
    function_ref<InFlightDiagnostic(Location)> emitError) const {
  // Emit the main error.
  InFlightDiagnostic diag = emitError(loc) << getMessage();
  // Emit the causes
  emit(diag, causes, /*indentDepth=*/2);
}

void ErrorTree::emit(InFlightDiagnostic &diag, ArrayRef<ErrorTree> errors,
                     unsigned indentDepth) {
  if (errors.empty())
    return;

  std::string spaces(indentDepth, ' ');
  for (const ErrorTree &err : errors) {
    diag.attachNote(err.loc) << spaces << err.getMessage();
    emit(diag, err.causes, indentDepth + 2);
  }
}
