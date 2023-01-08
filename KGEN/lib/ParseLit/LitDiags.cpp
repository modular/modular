//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LitDiags.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// LitDiags implementation
//===----------------------------------------------------------------------===//

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
LitDiagnostic LitDiags::emitError(Location loc, const Twine &message) {
  errorEmitted = true;
  return LitDiagnostic(loc, *this) << message;
}

/// Emit an error through the parser's logic.
LitDiagnostic LitDiags::emitError(llvm::SMLoc loc, const Twine &message) {
  return emitError(translateLocation(loc), message);
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

//===----------------------------------------------------------------------===//
// LitDiagnostic Implementation
//===----------------------------------------------------------------------===//

/// Each message in a diagnostic must have a location and text, and may
/// have any number of highlighted ranges and fixit hints.
struct LitDiagnostic::Message {
  Location loc;
  std::string text;
  // TODO: ranges and fixits.
};

LitDiagnostic::LitDiagnostic(LitDiagnostic &&other)
    : messages(std::move(other.messages)), diags(other.diags) {
  // Do not emit the other diagnostic.
  other.diags = nullptr;
}

LitDiagnostic::LitDiagnostic(Location loc, LitDiags &diags) : diags(&diags) {
  messages.push_back({loc, ""});
}

LitDiagnostic::~LitDiagnostic() {
  // If the diagnostic got abandoned, just drop it.
  if (!diags)
    return;

  // Build the MLIR diagnostic and hand it off to its diagnostic machinery.
  // TODO: Support warnings.
  Diagnostic mlirDiag(messages.front().loc, mlir::DiagnosticSeverity::Error);
  mlirDiag << messages.front().text;
  for (auto &note : llvm::drop_begin(messages))
    mlirDiag.attachNote(note.loc) << note.text;

  // Emit the diagnostic through the MLIR machinery.
  diags->context->getDiagEngine().emit(std::move(mlirDiag));
}

/// Add a note to this diagnostic at the specified location, and change the
/// emission point to start filling it in.
LitDiagnostic LitDiagnostic::attachNote(Location loc) && {
  messages.push_back({loc, ""});
  return std::move(*this);
}
LitDiagnostic &LitDiagnostic::attachNote(Location loc) & {
  messages.push_back({loc, ""});
  return *this;
}

void LitDiagnostic::addText(const Twine &text) {
  messages.back().text += text.str();
}

// Allow inserting string-like things.
void LIT::appendText(const Twine &text, LitDiagnostic &diag) {
  diag.addText(text);
}

void LIT::appendText(char text, LitDiagnostic &diag) {
  diag.addText(Twine(text));
}

void LIT::appendText(size_t number, LitDiagnostic &diag) {
  diag.addText(Twine(number));
}

void LIT::appendText(Attribute attr, LitDiagnostic &diag) {
  if (auto strAttr = dyn_cast<StringAttr>(attr)) {
    diag.addText(Twine("'"));
    diag.addText(strAttr.getValue());
    diag.addText(Twine("'"));
    return;
  }

  SmallString<128> str;
  llvm::raw_svector_ostream os(str);
  attr.print(os);
  diag.addText(os.str());
}
