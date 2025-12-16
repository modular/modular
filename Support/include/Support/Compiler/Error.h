//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_SUPPORT_ERROR_H
#define KGEN_SUPPORT_ERROR_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"

namespace M {
struct ErrorLimit {
  int errorLimit = 0;
  int errorCount = 0;
};

/// Emit error with a limit check.
void emitLimitedError(function_ref<InFlightDiagnostic()> emitError,
                      ErrorLimit &limit);

/// Helper function to check if a location is from the mojo startup module or
/// not.
bool isLocationInPrelude(const Location &loc);

} // namespace M

#endif // KGEN_SUPPORT_ERROR_H
