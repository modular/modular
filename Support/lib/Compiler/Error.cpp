//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/Error.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

void M::emitLimitedError(function_ref<InFlightDiagnostic()> emitError,
                         ErrorLimit &limit) {
  ++limit.errorCount;
  if (limit.errorLimit > 0 && limit.errorCount > limit.errorLimit) {
    return;
  }

  InFlightDiagnostic diag = emitError();

  // Emit message if hits error limit.
  if (limit.errorCount == limit.errorLimit)
    diag.attachNote() << "too many errors emitted, stopping now";
}

bool M::isLocationInPrelude(const Location &loc) {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << loc;
  return str.find("_startup.mojo") != std::string::npos;
}
