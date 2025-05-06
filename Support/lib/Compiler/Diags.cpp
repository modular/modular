//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/Diags.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/SourceMgr.h"

using llvm::SMRange;
using namespace M;

//===----------------------------------------------------------------------===//
// SourceRange implementation
//===----------------------------------------------------------------------===//

SourceRange::SourceRange(SMLoc start, SMLoc end)
    : start(start.getPointer()), end(end.getPointer()) {
  assert(start.isValid() == end.isValid() &&
         "Start and End should either both be valid or both be invalid!");
}

SourceRange::SourceRange(SMRange range) : SourceRange(range.Start, range.End) {}

SourceRange SourceRange::getByteLevel(SMLoc start, SMLoc end) {
  auto result = SourceRange(start, end);
  result.byteLevel = true;
  return result;
}

SMLoc SourceRange::getStart() const { return SMLoc::getFromPointer(start); }
SMLoc SourceRange::getEnd() const { return SMLoc::getFromPointer(end); }

//===----------------------------------------------------------------------===//
// SourceMgrLocationMapper implementation
//===----------------------------------------------------------------------===//

/// This is a helper that allows us to translate mlir::Location objects to
/// llvm::SMLoc objects.
///
/// TODO(mlir upstream): This was refactored out of
/// SourceMgrDiagnosticHandlerImpl; upstream this.
class Diags::SourceMgrLocationMapper {
public:
  SourceMgrLocationMapper(MLIRContext *context, StringRef stripFilenamePrefix)
      : unknownBufferNameIdentifier(StringAttr::get(
            context, Diags::SourceMgrLocationMapper::kUnnamedFileSigil)),
        context(context), stripFilenamePrefix(stripFilenamePrefix) {}

  /// Constant string that we can use to signify an un-named file (usually means
  /// reading from stdin or something).
  static constexpr StringLiteral kUnnamedFileSigil = "<unknown>";
  /// StringAttr version of `kUnnamedFileSigil`.
  StringAttr unknownBufferNameIdentifier;

  /// Return the SrcManager buffer id for the specified file, or zero if none
  /// can be found.
  unsigned getBufferIDForFile(SourceMgr &sourceMgr, StringAttr filename);
  /// Get the name of the buffer so we can rapidly build Location objects on
  /// demand.
  StringAttr getFilenameFromBufferId(const SourceMgr &sourceMgr,
                                     unsigned bufferID);

  /// Convert a location to SMLoc.
  SMLoc convertLocToSMLoc(SourceMgr &sourceMgr, FileLineColLoc loc);
  SMLoc convertLocToSMLoc(SourceMgr &sourceMgr, Location loc);

  /// Convert a SMLoc to location.
  FileLineColLoc convertSMLocToLoc(SourceMgr &sourceMgr, SMLoc loc);

  /// Canonicalize a filename by stripping `stripFilenamePrefix` prefix from it
  /// if applicable.
  std::string getCanonicalFilename(StringRef filename) const;
  StringRef getStripFilenamePrefix() const { return stripFilenamePrefix; }

private:
  MLIRContext *context;
  std::string stripFilenamePrefix;
  /// Maps known StringAttr filenames to buffer IDs.
  llvm::DenseMap<StringAttr, unsigned> filenameToBufId;
  /// Maps buffer IDs to their canonical filenames.
  llvm::DenseMap<unsigned, StringAttr> bufIdToCanonicalFilename;
};

unsigned
Diags::SourceMgrLocationMapper::getBufferIDForFile(llvm::SourceMgr &sourceMgr,
                                                   StringAttr filename) {
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

StringAttr Diags::SourceMgrLocationMapper::getFilenameFromBufferId(
    const SourceMgr &sourceMgr, unsigned bufferID) {
  // Lookup the cache first.
  auto bufferIt = bufIdToCanonicalFilename.find(bufferID);
  if (bufferIt != bufIdToCanonicalFilename.end())
    return bufferIt->second;

  if (sourceMgr.getNumBuffers() == 0)
    return unknownBufferNameIdentifier;

  auto mainBuffer = sourceMgr.getMemoryBuffer(bufferID);
  StringRef bufferName = mainBuffer->getBufferIdentifier();
  StringAttr filename;
  if (bufferName.empty()) {
    filename = unknownBufferNameIdentifier;
  } else {
    std::string relFilename = getCanonicalFilename(bufferName);
    filename = StringAttr::get(context, relFilename);
  }

  bufIdToCanonicalFilename[bufferID] = filename;
  filenameToBufId[filename] = bufferID;
  return filename;
}

/// Get a memory buffer for the given file, or the main file of the source
/// manager if one doesn't exist. This always returns non-null.
SMLoc Diags::SourceMgrLocationMapper::convertLocToSMLoc(SourceMgr &sourceMgr,
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

SMLoc Diags::SourceMgrLocationMapper::convertLocToSMLoc(SourceMgr &sourceMgr,
                                                        Location loc) {
  if (auto fileLineCol = dyn_cast<FileLineColLoc>(loc))
    return convertLocToSMLoc(sourceMgr, fileLineCol);
  return SMLoc();
}

FileLineColLoc
Diags::SourceMgrLocationMapper::convertSMLocToLoc(SourceMgr &sourceMgr,
                                                  SMLoc loc) {
  unsigned bufferID = sourceMgr.FindBufferContainingLoc(loc);
  if (!bufferID)
    return FileLineColLoc::get(unknownBufferNameIdentifier, 0, 0);
  auto lineAndColumn = sourceMgr.getLineAndColumn(loc, bufferID);

  StringAttr bufferName = getFilenameFromBufferId(sourceMgr, bufferID);
  return FileLineColLoc::get(bufferName, lineAndColumn.first,
                             lineAndColumn.second);
}

std::string Diags::SourceMgrLocationMapper::getCanonicalFilename(
    StringRef filenameRef) const {
  if (!stripFilenamePrefix.empty() &&
      filenameRef.starts_with(stripFilenamePrefix))
    return filenameRef.drop_front(stripFilenamePrefix.size()).str();
  return std::string(filenameRef);
}

//===----------------------------------------------------------------------===//
// Diags implementation
//===----------------------------------------------------------------------===//

/// Diag Handler that expects a `Diags` object via `context`, and invokes the
/// `prefixStrippingDiagHandler` on the `Diags` object.
static void prefixStrippingDiagHandler(const llvm::SMDiagnostic &diagnostic,
                                       void *context) {
  ((Diags *)context)->diagHandler(diagnostic);
}

Diags::Diags(SourceMgr &sourceMgr, MLIRContext *context,
             bool useMLIRDiagnostics, int maxNotesPerDiagnostic,
             StringRef stripFilenamePrefix, bool disableWarnings,
             void *extraContext)
    : sourceMgr(sourceMgr), context(context), extraContext(extraContext),
      sourceMgrMapper(std::make_unique<SourceMgrLocationMapper>(
          context, stripFilenamePrefix)),
      useMLIRDiagnostics(useMLIRDiagnostics),
      maxNotesPerDiagnostic(maxNotesPerDiagnostic),
      disableWarnings(disableWarnings) {
  // Install a prefix-stripping diag handler if necessary.
  if (!stripFilenamePrefix.empty()) {
    prevDiagHandler = sourceMgr.getDiagHandler();
    prevDiagContext = sourceMgr.getDiagContext();
    sourceMgr.setDiagHandler(::prefixStrippingDiagHandler, this);
  }
}

Diags::~Diags() {
  // Remove the prefix-stripping diag handler if previously installed.
  if (!sourceMgrMapper->getStripFilenamePrefix().empty())
    sourceMgr.setDiagHandler(prevDiagHandler, prevDiagContext);
}

/// Return the identifier for the main buffer in the SourceMgr.
StringAttr Diags::getBufferNameIdentifier() const {
  if (!bufferNameIdentifier) {
    if (sourceMgr.getNumBuffers() == 0)
      return sourceMgrMapper->unknownBufferNameIdentifier;
    bufferNameIdentifier =
        sourceMgrMapper
            ->getFilenameFromBufferId(sourceMgr, sourceMgr.getMainFileID())
            .getAsOpaquePointer();
  }

  return StringAttr::getFromOpaquePointer(*bufferNameIdentifier);
}

/// Emit an error through the parser's logic.
InflightDiag Diags::emitError(Location loc, const Twine &message) {
  return InflightDiag(loc, *this, /*isWarning=*/false) << message;
}

/// Emit an error through the parser's logic.
InflightDiag Diags::emitError(llvm::SMLoc loc, const Twine &message) {
  return emitError(translateLocation(loc), message);
}

/// Emit a warning.
InflightDiag Diags::emitWarning(Location loc, const Twine &message) {
  return InflightDiag(loc, *this, /*isWarning=*/true) << message;
}
InflightDiag Diags::emitWarning(llvm::SMLoc loc, const Twine &message) {
  return emitWarning(translateLocation(loc), message);
}

/// Encode the specified source location information into a Location object
/// for attachment to the IR or error reporting.  This always returns a
/// FileLineColLoc.
FileLineColLoc Diags::translateLocation(SMLoc loc) const {
  return sourceMgrMapper->convertSMLocToLoc(sourceMgr, loc);
}

std::string Diags::getCanonicalFilename(StringRef filenameRef) const {
  return sourceMgrMapper->getCanonicalFilename(filenameRef);
}

SMLoc Diags::convertLocToSMLoc(LocationAttr loc) const {
  if (!loc)
    return SMLoc();
  if (FileLineColLoc fileLoc = loc.findInstanceOf<FileLineColLoc>())
    return sourceMgrMapper->convertLocToSMLoc(sourceMgr, fileLoc);
  return SMLoc();
}

llvm::SMRange Diags::convertToSMRange(SourceRange range) const {
  SMRange byteLevelRange{range.getStart(), range.getEnd()};

  // SourceRange typically represents the end of range in terms of the start
  // of the end location.  Convert to a SMRange with a byte-level end position
  // if needed.
  if (!range.isByteLevel() && tokenEndPointAdjustmentFn &&
      byteLevelRange.End.isValid())
    tokenEndPointAdjustmentFn(byteLevelRange.End);
  return byteLevelRange;
}

void Diags::printIncludeStack(SMLoc includeLoc, raw_ostream &OS) const {
  if (includeLoc == SMLoc())
    return; // Top of stack.

  unsigned bufferID = sourceMgr.FindBufferContainingLoc(includeLoc);
  assert(bufferID && "Invalid or unspecified location!");
  printIncludeStack(sourceMgr.getBufferInfo(bufferID).IncludeLoc, OS);

  StringAttr bufferName =
      sourceMgrMapper->getFilenameFromBufferId(sourceMgr, bufferID);
  auto lineAndColumn = sourceMgr.getLineAndColumn(includeLoc, bufferID);
  OS << "Included from " << bufferName << ":" << lineAndColumn.first << ":\n";
}

void Diags::emitDiagnostic(const llvm::SMDiagnostic &diag) const {
  if (prevDiagHandler) {
    prevDiagHandler(diag, prevDiagContext);
    return;
  }

  if (diag.getLoc().isValid()) {
    SMLoc loc = diag.getLoc();
    unsigned bufferID = sourceMgr.FindBufferContainingLoc(loc);
    assert(bufferID && "Invalid or unspecified location!");
    printIncludeStack(sourceMgr.getBufferInfo(bufferID).IncludeLoc,
                      llvm::errs());
  }

  diag.print(nullptr, llvm::errs());
}

void Diags::diagHandler(const llvm::SMDiagnostic &diag) {
  StringRef origFilename = diag.getFilename();
  if (origFilename.empty() || origFilename == "-" ||
      origFilename == sourceMgrMapper->kUnnamedFileSigil) {
    emitDiagnostic(diag);
    return;
  }

  std::string filename = getCanonicalFilename(origFilename);
  llvm::SMDiagnostic newDiag(
      *diag.getSourceMgr(), diag.getLoc(), filename, diag.getLineNo(),
      diag.getColumnNo(), diag.getKind(), diag.getMessage(),
      diag.getLineContents(), diag.getRanges(), diag.getFixIts());
  emitDiagnostic(newDiag);
}

//===----------------------------------------------------------------------===//
// InflightDiag Implementation
//===----------------------------------------------------------------------===//

/// Each message in a diagnostic must have a location and text, and may
/// have any number of highlighted ranges and fixit hints.
struct InflightDiag::Message {
  Location loc;
  std::string text;
  std::vector<SMRange> ranges;
  std::vector<SMFixIt> fixIts;
};

InflightDiag::InflightDiag(InflightDiag &&other)
    : messages(std::move(other.messages)), diags(other.diags),
      isWarning(other.isWarning) {
  // Do not emit the other diagnostic.
  other.diags = nullptr;
}

InflightDiag &InflightDiag::operator=(InflightDiag &&other) {
  messages = std::move(other.messages);
  diags = other.diags;
  isWarning = other.isWarning;

  // Do not emit the other diagnostic.
  other.diags = nullptr;
  return *this;
}

InflightDiag::InflightDiag(Location loc, Diags &diags, bool isWarning)
    : diags(&diags), isWarning(isWarning) {
  messages.push_back({loc, /*message=*/"", /*ranges=*/{}, /*fixIts=*/{}});
}

InflightDiag::~InflightDiag() {
  // If the diagnostic got abandoned, just drop it.
  if (!diags)
    return;

  if (diags->disableWarnings && isWarning)
    return;

  diags->diagnosticEmitted = true;
  if (!isWarning)
    diags->errorEmitted = true;
  if (diags->useMLIRDiagnostics)
    emitMLIRDiagnostic();
  else
    emitSourceMgrDiagnostic();
}

/// Build the MLIR diagnostic and hand it off to its diagnostic machinery.
void InflightDiag::emitMLIRDiagnostic() {
  Location loc = messages.front().loc;
  InFlightDiagnostic mlirDiag =
      isWarning ? mlir::emitWarning(loc) : mlir::emitError(loc);
  mlirDiag << messages.front().text;
  for (auto &note : llvm::drop_begin(messages))
    mlirDiag.attachNote(note.loc) << note.text;
}

/// Print the diagnostic + each note through SourceMgr.
void InflightDiag::emitSourceMgrDiagnostic() {
  auto &sourceMgr = diags->sourceMgr;

  int nMessagesPrinted = 0;
  SourceMgr::DiagKind kind =
      isWarning ? SourceMgr::DK_Warning : SourceMgr::DK_Error;
  for (auto &message : messages) {
    auto loc = diags->convertLocToSMLoc(message.loc);

    // If we have an exotic MLIR location, give up.  Mojo shouldn't be producing
    // these, so just pick a weird-but-valid location.
    if (!loc.isValid())
      loc = sourceMgr.FindLocForLineAndColumn(sourceMgr.getMainFileID(), 0, 0);

    // Limit number of notes to print
    ++nMessagesPrinted;
    llvm::raw_string_ostream text(message.text);
    auto nOmitted = messages.size() - nMessagesPrinted;
    if (nMessagesPrinted > diags->maxNotesPerDiagnostic && nOmitted > 0)
      text << " (" << nOmitted << " more notes omitted.)";

    sourceMgr.PrintMessage(loc, kind, text.str(), message.ranges,
                           message.fixIts);
    if (nMessagesPrinted > diags->maxNotesPerDiagnostic)
      break;

    // Subsequent diagnostics are all notes.
    kind = SourceMgr::DK_Note;
  }
}

/// Add a note to this diagnostic at the specified location, and change the
/// emission point to start filling it in.
InflightDiag InflightDiag::attachNote(Location loc) && {
  messages.push_back({loc, /*message=*/"", /*ranges=*/{}, /*fixIts=*/{}});
  return std::move(*this);
}
InflightDiag &InflightDiag::attachNote(Location loc) & {
  messages.push_back({loc, /*message=*/"", /*ranges=*/{}, /*fixIts=*/{}});
  return *this;
}

InflightDiag InflightDiag::attachNote(SMLoc loc) && {
  // If the diagnostic has been detached then we cannot translate the location,
  // but we don't care if we are anyway.
  if (!diags)
    return std::move(*this);
  return std::move(*this).attachNote(diags->translateLocation(loc));
}

InflightDiag &InflightDiag::attachNote(SMLoc loc) & {
  // If the diagnostic has been detached then we cannot translate the location,
  // but we don't care if we are anyway.
  if (!diags)
    return *this;
  return attachNote(diags->translateLocation(loc));
}

void InflightDiag::addDiag(InflightDiag &&otherDiag) {
  auto otherMessages = std::move(otherDiag.messages);
  Message &otherPrimary = otherMessages[0];
  Message &lastMsg = messages.back();
  lastMsg.text += otherPrimary.text;
  llvm::append_range(lastMsg.ranges, std::move(otherPrimary.ranges));
  llvm::append_range(lastMsg.fixIts, std::move(otherPrimary.fixIts));
  llvm::append_range(messages, llvm::drop_begin(std::move(otherMessages)));
  otherDiag.abandon();
}

void InflightDiag::addText(const Twine &text) {
  messages.back().text += text.str();
}

static SMRange translateToSMRange(SourceRange range, Diags *diags) {
  if (!diags || diags->useMLIRDiagnostics)
    return {range.getStart(), range.getEnd()};
  return diags->convertToSMRange(range);
}

void InflightDiag::addSourceRange(SourceRange range) {
  messages.back().ranges.push_back(translateToSMRange(range, diags));
}

void InflightDiag::addFixIt(FixIt fixIt) {
  messages.back().fixIts.push_back(
      SMFixIt(translateToSMRange(fixIt.range, diags), fixIt.replacement));
}

FixIt::FixIt(SourceRange range, const Twine &replacement)
    : range(range), replacement(replacement.str()) {}

/// This constructor creates a fixit that removes the specified token.
FixIt FixIt::remove(SMLoc loc) { return FixIt({loc, loc}, Twine()); }

/// This constructor creates a fixit that removes the specified token range.
FixIt FixIt::remove(SourceRange range) { return FixIt(range, Twine()); }

/// This constructor creates a fixit that replaces the one token at the
/// specified location with some text.
FixIt FixIt::replaceToken(SMLoc loc, const Twine &text) {
  return FixIt({loc, loc}, text);
}

/// This constructor creates a fixit that inserts some text before the token
/// at the specified location, without replacing the token.
FixIt FixIt::insertBeforeToken(SMLoc loc, const Twine &text) {
  // Set the replacement range to an empty byte-level range before the token.
  return FixIt(SourceRange::getByteLevel(loc, loc), text);
}

/// This constructor creates a fixit that inserts some text after the token
/// at the specified location.
FixIt FixIt::insertAfterToken(SMLoc loc, const Twine &text, Diags &diags) {
  // Find end of token if we have a token end point adjustment function.
  if (diags.tokenEndPointAdjustmentFn && loc.isValid())
    diags.tokenEndPointAdjustmentFn(loc);
  return FixIt(SourceRange::getByteLevel(loc, loc), text);
}

//===----------------------------------------------------------------------===//
// addToDiagnostic helpers
//===----------------------------------------------------------------------===//

// Allow inserting string-like things.
void M::addToDiagnostic(const Twine &text, InflightDiag &diag) {
  diag.addText(text);
}

void M::addToDiagnostic(char text, InflightDiag &diag) {
  diag.addText(Twine(text));
}

void M::addToDiagnostic(size_t number, InflightDiag &diag) {
  diag.addText(Twine(number));
}

void M::addToDiagnostic(StringAttr attr, InflightDiag &diag) {
  diag.addText(Twine("'"));
  diag.addText(attr.getValue());
  diag.addText(Twine("'"));
}

/// This adds a source range highlight.
void M::addToDiagnostic(SourceRange range, InflightDiag &diag) {
  diag.addSourceRange(range);
}

/// This adds a fixit hint.
void M::addToDiagnostic(FixIt fixIt, InflightDiag &diag) {
  diag.addFixIt(fixIt);
}

void M::addToDiagnostic(InflightDiag &&otherDiag, InflightDiag &diag) {
  diag.addDiag(std::move(otherDiag));
}
