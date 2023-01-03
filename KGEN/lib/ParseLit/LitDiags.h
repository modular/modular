//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines the preferred diagnostics machinery used by the frontend.
// It is intentionally similar to the MLIR diagnostics machinery, but disallows
// printing MLIR types (forcing use of ASTTypes) and supports additional
// features like source ranges.
//
//===----------------------------------------------------------------------===//

#ifndef LITDIAGS_H
#define LITDIAGS_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace M::KGEN::LIT {
using llvm::SMLoc;
using llvm::SourceMgr;

class LitDiags {
public:
  LitDiags(SourceMgr &sourceMgr, MLIRContext *context);

  llvm::SourceMgr &sourceMgr;
  MLIRContext *const context;

  /// Return the identifier for the main buffer in the SourceMgr.
  StringAttr getBufferNameIdentifier() const;

  bool isErrorEmitted() const { return errorEmitted; }

  /// Emit an error through the parser's logic.
  InFlightDiagnostic emitError(Location loc, const Twine &twine);

  /// Emit an error through the parser's logic.
  InFlightDiagnostic emitError(llvm::SMLoc loc, const Twine &twine);

  /// Encode the specified source location information into a Location object
  /// for attachment to the IR or error reporting.  This always returns a
  /// FileLineColLoc.
  Location translateLocation(llvm::SMLoc loc) const;

private:
  /// This is the StringAttr for the main buffer identifier.  It is type erased
  /// to void* to reduce header polution.
  const void *const bufferNameIdentifier;

  /// This is set to true if an error occurred at any point processing the
  /// file.
  bool errorEmitted = false;
};

} // namespace M::KGEN::LIT

#endif // LITDIAGS_H
