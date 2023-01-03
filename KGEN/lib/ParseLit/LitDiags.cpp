//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LitDiags.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// Get the name of the main buffer so we can rapidly build Location objects
/// on demand.
static StringAttr makeBufferNameIdentifier(const SourceMgr &sourceMgr,
                                           size_t bufferID,
                                           MLIRContext *context) {
  auto mainBuffer = sourceMgr.getMemoryBuffer(bufferID);
  StringRef bufferName = mainBuffer->getBufferIdentifier();
  if (bufferName.empty())
    bufferName = "<unknown>";
  return StringAttr::get(context, bufferName);
}

/// This sets up the buffer name identifier for the main buffer.
static const void *makeMainBufferNameIdentifier(const SourceMgr &sourceMgr,
                                                MLIRContext *context) {
  return makeBufferNameIdentifier(sourceMgr, sourceMgr.getMainFileID(), context)
      .getAsOpaquePointer();
}

LitDiags::LitDiags(SourceMgr &sourceMgr, MLIRContext *context)
    : sourceMgr(sourceMgr), context(context),
      bufferNameIdentifier(makeMainBufferNameIdentifier(sourceMgr, context)) {}

/// Return the identifier for the main buffer in the SourceMgr.
StringAttr LitDiags::getBufferNameIdentifier() const {
  return StringAttr::getFromOpaquePointer(bufferNameIdentifier);
}

/// Emit an error through the parser's logic.
InFlightDiagnostic LitDiags::emitError(Location loc, const Twine &twine) {
  errorEmitted = true;
  return mlir::emitError(loc, twine);
}

/// Emit an error through the parser's logic.
InFlightDiagnostic LitDiags::emitError(llvm::SMLoc loc, const Twine &twine) {
  return emitError(translateLocation(loc), twine);
}

/// Encode the specified source location information into a Location object
/// for attachment to the IR or error reporting.  This always returns a
/// FileLineColLoc.
Location LitDiags::translateLocation(SMLoc loc) const {
  // TODO: Implement a cache here to speed up location translation.
  unsigned bufferID = sourceMgr.FindBufferContainingLoc(loc);
  auto lineAndColumn = sourceMgr.getLineAndColumn(loc, bufferID);

  StringAttr bufferName;
  if (bufferID == sourceMgr.getMainFileID())
    bufferName = getBufferNameIdentifier();
  else
    bufferName = makeBufferNameIdentifier(sourceMgr, bufferID, context);

  return FileLineColLoc::get(bufferName, lineAndColumn.first,
                             lineAndColumn.second);
}
