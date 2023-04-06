//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LitDiags.h"
#include "LitLexer.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/SourceMgr.h"

using llvm::SMRange;
using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// LitSourceRange implementation
//===----------------------------------------------------------------------===//

LitSourceRange::LitSourceRange(SMLoc start, SMLoc end)
    : start(start.getPointer()), end(end.getPointer()) {
  assert(start.isValid() == end.isValid() &&
         "Start and End should either both be valid or both be invalid!");
}

LitSourceRange LitSourceRange::getByteLevel(SMLoc start, SMLoc end) {
  auto result = LitSourceRange(start, end);
  result.byteLevel = true;
  return result;
}

SMLoc LitSourceRange::getStart() const { return SMLoc::getFromPointer(start); }
SMLoc LitSourceRange::getEnd() const { return SMLoc::getFromPointer(end); }

//===----------------------------------------------------------------------===//
// SourceMgrLocationMapper implementation
//===----------------------------------------------------------------------===//

/// This is a helper that allows us to translate mlir::Location objects to
/// llvm::SMLoc objects.
///
/// TODO(mlir upstream): This was refactored out of
/// SourceMgrDiagnosticHandlerImpl; upstream this.
class LitDiags::SourceMgrLocationMapper {
public:
  /// Constant string that we can use to signify an un-named file (usually means
  /// reading from stdin or something).
  static constexpr StringLiteral kUnnamedFileSigil = "<unknown>";

  /// Return the SrcManager buffer id for the specified file, or zero if none
  /// can be found.
  unsigned getBufferIDForFile(SourceMgr &sourceMgr, StringAttr filename);

  /// Convert a location to SMLoc.
  SMLoc convertLocToSMLoc(SourceMgr &sourceMgr, FileLineColLoc loc);
  SMLoc convertLocToSMLoc(SourceMgr &sourceMgr, Location loc);

private:
  /// Mapping between file name and buffer ID's.
  llvm::DenseMap<StringAttr, unsigned> filenameToBufId;
};

unsigned LitDiags::SourceMgrLocationMapper::getBufferIDForFile(
    llvm::SourceMgr &sourceMgr, StringAttr filename) {
  // Check for an existing mapping to the buffer id for this file.
  auto bufferIt = filenameToBufId.find(filename);
  if (bufferIt != filenameToBufId.end())
    return bufferIt->second;

  // Look for a buffer in the manager that has this filename.
  for (unsigned i = 1, e = sourceMgr.getNumBuffers() + 1; i != e; ++i) {
    auto *buf = sourceMgr.getMemoryBuffer(i);
    if (buf->getBufferIdentifier() == filename.getValue())
      return filenameToBufId[filename] = i;
  }

  // Otherwise, try to load the source file.
  std::string ignored;
  unsigned id = sourceMgr.AddIncludeFile(filename.str(), SMLoc(), ignored);
  filenameToBufId[filename] = id;
  return id;
}

/// Get a memory buffer for the given file, or the main file of the source
/// manager if one doesn't exist. This always returns non-null.
SMLoc LitDiags::SourceMgrLocationMapper::convertLocToSMLoc(SourceMgr &sourceMgr,
                                                           FileLineColLoc loc) {
  // The column and line may be zero to represent unknown column and/or unknown
  /// line/column information.
  if (loc.getLine() == 0 || loc.getColumn() == 0)
    return SMLoc();

  // Default the buffer ID to 0 - this is the 'unknown' buffer ID since the
  // SourceMgr starts at 1. This will result in a default-constructed SMLoc
  // below unless we can match the filename in the loc.
  unsigned bufferId = 0;
  if (loc.getFilename() == kUnnamedFileSigil)
    bufferId = sourceMgr.getMainFileID();
  else
    bufferId = getBufferIDForFile(sourceMgr, loc.getFilename());

  if (!bufferId)
    return SMLoc();
  return sourceMgr.FindLocForLineAndColumn(bufferId, loc.getLine(),
                                           loc.getColumn());
}

SMLoc LitDiags::SourceMgrLocationMapper::convertLocToSMLoc(SourceMgr &sourceMgr,
                                                           Location loc) {
  if (auto fileLineCol = dyn_cast<FileLineColLoc>(loc))
    return convertLocToSMLoc(sourceMgr, fileLineCol);
  return SMLoc();
}

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
    bufferName = LitDiags::SourceMgrLocationMapper::kUnnamedFileSigil;
  return StringAttr::get(context, bufferName);
}

/// This sets up the buffer name identifier for the main buffer.
static const void *makeMainBufferNameIdentifier(const SourceMgr &sourceMgr,
                                                MLIRContext *context) {
  return makeBufferNameIdentifier(sourceMgr, sourceMgr.getMainFileID(), context)
      .getAsOpaquePointer();
}

LitDiags::LitDiags(SourceMgr &sourceMgr, MLIRContext *context,
                   bool useMLIRDiagnostics)
    : sourceMgr(sourceMgr), context(context),
      bufferNameIdentifier(makeMainBufferNameIdentifier(sourceMgr, context)),
      useMLIRDiagnostics(useMLIRDiagnostics) {

  if (!useMLIRDiagnostics)
    sourceMgrMapper = std::make_unique<SourceMgrLocationMapper>();
}

LitDiags::~LitDiags() {}

/// Return the identifier for the main buffer in the SourceMgr.
StringAttr LitDiags::getBufferNameIdentifier() const {
  return StringAttr::getFromOpaquePointer(bufferNameIdentifier);
}

/// Emit an error through the parser's logic.
LitDiagnostic LitDiags::emitError(Location loc, const Twine &message) {
  diagnosticEmitted = errorEmitted = true;
  return LitDiagnostic(loc, *this, /*isWarning=*/false) << message;
}

/// Emit an error through the parser's logic.
LitDiagnostic LitDiags::emitError(llvm::SMLoc loc, const Twine &message) {
  return emitError(translateLocation(loc), message);
}

/// Emit a warning.
LitDiagnostic LitDiags::emitWarning(Location loc, const Twine &message) {
  diagnosticEmitted = true;
  return LitDiagnostic(loc, *this, /*isWarning=*/true) << message;
}
LitDiagnostic LitDiags::emitWarning(llvm::SMLoc loc, const Twine &message) {
  return emitWarning(translateLocation(loc), message);
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
  std::vector<SMRange> ranges;
  std::vector<SMFixIt> fixIts;
};

LitDiagnostic::LitDiagnostic(LitDiagnostic &&other)
    : messages(std::move(other.messages)), diags(other.diags),
      isWarning(other.isWarning) {
  // Do not emit the other diagnostic.
  other.diags = nullptr;
}

LitDiagnostic::LitDiagnostic(Location loc, LitDiags &diags, bool isWarning)
    : diags(&diags), isWarning(isWarning) {
  messages.push_back({loc, /*message=*/"", /*ranges=*/{}, /*fixIts=*/{}});
}

LitDiagnostic::~LitDiagnostic() {
  // If the diagnostic got abandoned, just drop it.
  if (!diags)
    return;

  if (diags->useMLIRDiagnostics)
    emitMLIRDiagnostic();
  else
    emitSourceMgrDiagnostic();
}

/// Build the MLIR diagnostic and hand it off to its diagnostic machinery.
void LitDiagnostic::emitMLIRDiagnostic() {
  Location loc = messages.front().loc;
  InFlightDiagnostic mlirDiag =
      isWarning ? mlir::emitWarning(loc) : mlir::emitError(loc);
  mlirDiag << messages.front().text;
  for (auto &note : llvm::drop_begin(messages))
    mlirDiag.attachNote(note.loc) << note.text;
}

/// Print the diagnostic + each note through SourceMgr.
void LitDiagnostic::emitSourceMgrDiagnostic() {
  auto &sourceMgr = diags->sourceMgr;

  SourceMgr::DiagKind kind =
      isWarning ? SourceMgr::DK_Warning : SourceMgr::DK_Error;
  for (auto &message : messages) {
    auto loc =
        diags->sourceMgrMapper->convertLocToSMLoc(sourceMgr, message.loc);

    // If we have an exotic MLIR location, give up.  Lit shouldn't be producing
    // these, so just pick a weird-but-valid location.
    if (!loc.isValid())
      loc = sourceMgr.FindLocForLineAndColumn(sourceMgr.getMainFileID(), 0, 0);

    sourceMgr.PrintMessage(loc, kind, message.text, message.ranges,
                           message.fixIts);
    // Subsequent diagnostics are all notes.
    kind = SourceMgr::DK_Note;
  }
}

/// Add a note to this diagnostic at the specified location, and change the
/// emission point to start filling it in.
LitDiagnostic LitDiagnostic::attachNote(Location loc) && {
  messages.push_back({loc, /*message=*/"", /*ranges=*/{}, /*fixIts=*/{}});
  return std::move(*this);
}
LitDiagnostic &LitDiagnostic::attachNote(Location loc) & {
  messages.push_back({loc, /*message=*/"", /*ranges=*/{}, /*fixIts=*/{}});
  return *this;
}

LitDiagnostic LitDiagnostic::attachNote(SMLoc loc) && {
  // If the diagnostic has been detached then we cannot translate the location,
  // but we don't care if we are anyway.
  if (!diags)
    return std::move(*this);
  return std::move(*this).attachNote(diags->translateLocation(loc));
}

LitDiagnostic &LitDiagnostic::attachNote(SMLoc loc) & {
  // If the diagnostic has been detached then we cannot translate the location,
  // but we don't care if we are anyway.
  if (!diags)
    return *this;
  return attachNote(diags->translateLocation(loc));
}

void LitDiagnostic::addText(const Twine &text) {
  messages.back().text += text.str();
}

static SMRange translateToSMRange(LitSourceRange range, LitDiags *diags) {
  SMRange byteLevelRange{range.getStart(), range.getEnd()};

  // LitSourceRange typically represents the end of range in terms of the start
  // of the end location.  Convert to a SMRange with a byte-level end position
  // if needed.
  if (diags && !range.isByteLevel() && diags && !diags->useMLIRDiagnostics &&
      diags->tokenEndPointAdjustmentFn)
    diags->tokenEndPointAdjustmentFn(byteLevelRange.End);
  return byteLevelRange;
}

void LitDiagnostic::addSourceRange(LitSourceRange range) {
  messages.back().ranges.push_back(translateToSMRange(range, diags));
}

void LitDiagnostic::addFixIt(LitFixIt fixIt) {
  messages.back().fixIts.push_back(
      SMFixIt(translateToSMRange(fixIt.range, diags), fixIt.replacement));
}

LitFixIt::LitFixIt(LitSourceRange range, const Twine &replacement)
    : range(range), replacement(replacement.str()) {}

/// This constructor creates a fixit that replaces the one token at the
/// specified location with some text.
LitFixIt LitFixIt::replaceToken(SMLoc loc, const Twine &text) {
  return LitFixIt({loc, loc}, text);
}

/// This constructor creates a fixit that inserts some text before the token
/// at the specified location, without replacing the token.
LitFixIt LitFixIt::insertBeforeToken(SMLoc loc, const Twine &text) {
  // Set the replacement range to an empty byte-level range before the token.
  return LitFixIt(LitSourceRange::getByteLevel(loc, loc), text);
}

/// This constructor creates a fixit that inserts some text after the token
/// at the specified location.
LitFixIt LitFixIt::insertAfterToken(SMLoc loc, const Twine &text,
                                    LitSharedState &shared) {
  // Find end of token.
  size_t tokenSize = LitLexer::getTokenLength(shared, loc);
  loc = SMLoc::getFromPointer(loc.getPointer() + tokenSize);
  return LitFixIt(LitSourceRange::getByteLevel(loc, loc), text);
}

//===----------------------------------------------------------------------===//
// addToDiagnostic helpers
//===----------------------------------------------------------------------===//

// Allow inserting string-like things.
void LIT::addToDiagnostic(const Twine &text, LitDiagnostic &diag) {
  diag.addText(text);
}

void LIT::addToDiagnostic(char text, LitDiagnostic &diag) {
  diag.addText(Twine(text));
}

void LIT::addToDiagnostic(size_t number, LitDiagnostic &diag) {
  diag.addText(Twine(number));
}

void LIT::addToDiagnostic(StringAttr attr, LitDiagnostic &diag) {
  diag.addText(Twine("'"));
  diag.addText(attr.getValue());
  diag.addText(Twine("'"));
}

/// This adds a source range highlight.
void LIT::addToDiagnostic(LitSourceRange range, LitDiagnostic &diag) {
  diag.addSourceRange(range);
}

/// This adds a fixit hint.
void LIT::addToDiagnostic(LitFixIt fixIt, LitDiagnostic &diag) {
  diag.addFixIt(fixIt);
}
