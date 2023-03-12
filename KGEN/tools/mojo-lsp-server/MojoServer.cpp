//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoServer.h"
#include "KGEN/CompilationOptions.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/ParseLit.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/Timing.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"
#include "mlir/Tools/lsp-server-support/SourceMgrUtils.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <optional>

namespace lsp = mlir::lsp;
using namespace M;
using namespace M::KGEN::LIT;
using llvm::SMLoc;

/// Returns a language server uri for the given source location. `mainFileURI`
/// corresponds to the uri for the main file of the source manager.
static std::optional<lsp::URIForFile>
getURIFromLoc(llvm::SourceMgr &mgr, SMLoc loc,
              const lsp::URIForFile &mainFileURI) {
  int bufferId = mgr.FindBufferContainingLoc(loc);
  if (bufferId == 0 || bufferId == static_cast<int>(mgr.getMainFileID()))
    return mainFileURI;
  llvm::Expected<lsp::URIForFile> fileForLoc = lsp::URIForFile::fromFile(
      mgr.getBufferInfo(bufferId).Buffer->getBufferIdentifier());
  if (fileForLoc)
    return *fileForLoc;
  lsp::Logger::error("Failed to create URI for include file: {0}",
                     llvm::toString(fileForLoc.takeError()));
  return std::nullopt;
}

/// Returns true if the given location is in the main file of the source
/// manager.
static bool isMainFileLoc(llvm::SourceMgr &mgr, SMLoc loc) {
  return mgr.FindBufferContainingLoc(loc) == mgr.getMainFileID();
}

/// Returns a language server range from the given diagnostic.
static lsp::Range getRangeFromDiag(llvm::SourceMgr &mgr,
                                   const llvm::SMDiagnostic &diag) {
  lsp::Range range(mgr, lsp::convertTokenLocToRange(diag.getLoc()));
  if (!diag.getRanges().empty()) {
    range.start.character = diag.getRanges()[0].first;
    range.end.character = diag.getRanges()[0].second;
  }
  return range;
}

/// Returns a language server location from the given diagnostic.
static std::optional<lsp::Location>
getLocationFromDiag(llvm::SourceMgr &mgr, const llvm::SMDiagnostic &diag,
                    const lsp::URIForFile &uri) {
  std::optional<lsp::URIForFile> diagUri =
      getURIFromLoc(mgr, diag.getLoc(), uri);
  if (!diagUri)
    return std::nullopt;
  return lsp::Location(*diagUri, getRangeFromDiag(mgr, diag));
}

//===----------------------------------------------------------------------===//
// MojoDocument
//===----------------------------------------------------------------------===//

namespace {
/// This class represents all of the information pertaining to a specific Mojo
/// document.
struct MojoDocument {
  MojoDocument(const lsp::URIForFile &uri, StringRef contents, int64_t version,
               std::vector<lsp::Diagnostic> &diagnostics,
               LLCL::Runtime &runtime);
  MojoDocument(const MojoDocument &) = delete;
  MojoDocument &operator=(const MojoDocument &) = delete;

  /// Return the current version of this document.
  int64_t getVersion() const { return version; }

  /// Initialize the document based on the current set of contents.
  void initialize(const lsp::URIForFile &uri,
                  std::vector<lsp::Diagnostic> &diagnostics);

  /// Update the file to the new version using the provided set of content
  /// changes. Returns failure if the update was unsuccessful.
  LogicalResult update(const lsp::URIForFile &uri, int64_t newVersion,
                       ArrayRef<lsp::TextDocumentContentChangeEvent> changes,
                       std::vector<lsp::Diagnostic> &diagnostics);

  //===--------------------------------------------------------------------===//
  // LSP Queries
  //===--------------------------------------------------------------------===//

  //===--------------------------------------------------------------------===//
  // Diagnostics

  std::optional<lsp::Diagnostic>
  buildLspDiagnoticFromSMDiagnostic(llvm::SourceMgr &sourceMgr,
                                    ArrayRef<llvm::SMDiagnostic> diags,
                                    const lsp::URIForFile &uri);

  //===--------------------------------------------------------------------===//
  // Code Actions

  void getCodeActions(const lsp::URIForFile &uri, const lsp::Range &pos,
                      const lsp::CodeActionContext &context,
                      std::vector<lsp::CodeAction> &actions);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  /// The full string contents of the file.
  std::string contents;

  /// The version of this file.
  int64_t version = 0;

  /// An ordered set of fixits for diagnostics emitted for the current version
  /// of the file.
  std::map<std::pair<lsp::Range, std::string>, std::vector<lsp::CodeAction>>
      fixits;

  /// The runtime used when parsing the file.
  LLCL::Runtime &runtime;
};
} // namespace

MojoDocument::MojoDocument(const lsp::URIForFile &uri, StringRef contents,
                           int64_t version,
                           std::vector<lsp::Diagnostic> &diagnostics,
                           LLCL::Runtime &runtime)
    : contents(contents.str()), version(version), runtime(runtime) {
  initialize(uri, diagnostics);
}

void MojoDocument::initialize(const lsp::URIForFile &uri,
                              std::vector<lsp::Diagnostic> &diagnostics) {
  auto memBuffer = llvm::MemoryBuffer::getMemBufferCopy(contents, uri.file());
  if (!memBuffer) {
    lsp::Logger::error("Failed to create memory buffer for {0}", uri.file());
    return;
  }

  // Reset the source manager and parse the file.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), SMLoc());

  // Build a wrapper diagnostic handler for the source manager to capture
  // diagnostics emitted when parsing the mojo file.
  struct DiagHandlerContext {
    /// A set of diagnostic groups, where the first diagnostic is the main
    /// diagnostic and the rest are notes.
    std::vector<std::vector<llvm::SMDiagnostic>> smDiagnostics;
  };
  auto handlerFn = [](const llvm::SMDiagnostic &diag, void *ctx) {
    auto &handlerCtx = *static_cast<DiagHandlerContext *>(ctx);

    // If this is a note, add it to the last diagnostic group.
    if (diag.getKind() == llvm::SourceMgr::DK_Note) {
      if (!handlerCtx.smDiagnostics.empty())
        handlerCtx.smDiagnostics.back().push_back(diag);
      return;
    }
    handlerCtx.smDiagnostics.push_back({diag});
  };
  DiagHandlerContext handlerCtx;
  sourceMgr.setDiagHandler(handlerFn, &handlerCtx);

  // Parse the mojo file. Ignore the result for now as we aren't doing anything
  // other than collecting diagnostics at this point.
  MLIRContext context(MLIRContext::Threading::DISABLED);
  mlir::TimingScope ts;
  M::importLitFile(sourceMgr, &context, ts, KGEN::CompilationOptions(),
                   /*useMLIRDiagnostics=*/false, runtime);

  // Process the collected diagnostics.
  for (ArrayRef<llvm::SMDiagnostic> diags : handlerCtx.smDiagnostics) {
    if (auto lspDiag = buildLspDiagnoticFromSMDiagnostic(sourceMgr, diags, uri))
      diagnostics.push_back(*lspDiag);
  }
}

LogicalResult
MojoDocument::update(const lsp::URIForFile &uri, int64_t newVersion,
                     ArrayRef<lsp::TextDocumentContentChangeEvent> changes,
                     std::vector<lsp::Diagnostic> &diagnostics) {
  if (failed(lsp::TextDocumentContentChangeEvent::applyTo(changes, contents))) {
    lsp::Logger::error("Failed to update contents of {0}", uri.file());
    return failure();
  }
  version = newVersion;
  fixits.clear();

  // If the file contents were properly changed, reinitialize the text file.
  // TODO: We shouldn't need to reinitialize the entire file here, we should be
  // able to selectively update the parts that actually changed.
  initialize(uri, diagnostics);
  return success();
}

//===----------------------------------------------------------------------===//
// MojoDocument: Diagnostics
//===----------------------------------------------------------------------===//

/// Sanitizes a piece for presenting it in a synthesized fix message. Ensures
/// the result is not too large and does not contain newlines.
static void writeCodeToFixMessage(raw_ostream &os, StringRef code) {
  constexpr unsigned kMaxLen = 50;
  if (code == "\n") {
    os << "\\n";
    return;
  }

  // Only show the first line if there are many.
  StringRef result = code.split('\n').first;

  // Shorten the message if it's too long.
  result = result.take_front(kMaxLen);

  os << result;
  if (result.size() != result.size())
    os << "…";
}

static std::optional<lsp::CodeAction>
buildCodeActionFromSMFixit(const llvm::SMFixIt &fixit, llvm::SourceMgr &mgr,
                           const lsp::URIForFile &mainFileURI) {
  llvm::SMRange range = fixit.getRange();
  if (!range.isValid())
    return std::nullopt;

  // Get the file this fixit is in.
  auto uri = getURIFromLoc(mgr, range.Start, mainFileURI);
  if (!uri)
    return std::nullopt;

  // Build the code action.
  lsp::CodeAction action;
  action.kind = lsp::CodeAction::kQuickFix.str();

  // Construct a title based on what the fixit is doing.
  {
    llvm::raw_string_ostream titleOS(action.title);

    StringRef removedText(range.Start.getPointer(),
                          range.End.getPointer() - range.Start.getPointer());
    StringRef insertedText = fixit.getText();
    if (!removedText.empty() && !insertedText.empty()) {
      titleOS << "change '";
      writeCodeToFixMessage(titleOS, removedText);
      titleOS << "' to '";
      writeCodeToFixMessage(titleOS, insertedText);
      titleOS << "'";
    } else if (!removedText.empty()) {
      titleOS << "remove '";
      writeCodeToFixMessage(titleOS, removedText);
      titleOS << "'";
    } else if (!insertedText.empty()) {
      titleOS << "insert '";
      writeCodeToFixMessage(titleOS, insertedText);
      titleOS << "'";
    }

    // Don't allow source code to inject newlines into diagnostics.
    std::replace(action.title.begin(), action.title.end(), '\n', ' ');
  }

  // Build the edit.
  action.edit.emplace();
  action.edit->changes[uri->uri().str()].push_back(
      {lsp::Range(mgr, range), fixit.getText().str()});
  return action;
}

/// Convert the given MLIR diagnostic to the LSP form.
std::optional<lsp::Diagnostic> MojoDocument::buildLspDiagnoticFromSMDiagnostic(
    llvm::SourceMgr &sourceMgr, ArrayRef<llvm::SMDiagnostic> diags,
    const lsp::URIForFile &uri) {
  const llvm::SMDiagnostic &mainDiag = diags[0];

  // Skip diagnostics that weren't emitted within the main file.
  if (!isMainFileLoc(sourceMgr, mainDiag.getLoc()))
    return std::nullopt;

  lsp::Diagnostic lspDiag;
  lspDiag.source = "mojo";
  lspDiag.category = "Parse Error";
  lspDiag.range = getRangeFromDiag(sourceMgr, mainDiag);

  // Convert the severity for the diagnostic.
  switch (mainDiag.getKind()) {
  case llvm::SourceMgr::DK_Note:
    llvm_unreachable("expected notes to be handled separately");
  case llvm::SourceMgr::DK_Warning:
    lspDiag.severity = lsp::DiagnosticSeverity::Warning;
    break;
  case llvm::SourceMgr::DK_Error:
    lspDiag.severity = lsp::DiagnosticSeverity::Error;
    break;
  case llvm::SourceMgr::DK_Remark:
    lspDiag.severity = lsp::DiagnosticSeverity::Information;
    break;
  }
  lspDiag.message = mainDiag.getMessage().str();

  // Attach any notes to the main diagnostic as related information.
  if (diags.size() > 1) {
    std::vector<lsp::DiagnosticRelatedInformation> relatedDiags;
    for (const llvm::SMDiagnostic &note : diags.drop_front())
      if (auto loc = getLocationFromDiag(sourceMgr, note, uri))
        relatedDiags.emplace_back(*loc, note.getMessage().str());
    lspDiag.relatedInformation = std::move(relatedDiags);
  }

  // Collect fixits for the diagnostic.
  std::vector<lsp::CodeAction> diagFixits;
  for (const llvm::SMFixIt &fixit : mainDiag.getFixIts())
    if (auto action = buildCodeActionFromSMFixit(fixit, sourceMgr, uri))
      diagFixits.push_back(*action);
  if (!diagFixits.empty()) {
    // If there is only one fixit, mark it as preferred.
    if (diagFixits.size() == 1)
      diagFixits[0].isPreferred = true;

    fixits.emplace(std::make_pair(lspDiag.range, lspDiag.message),
                   std::move(diagFixits));
  }

  return lspDiag;
}

//===----------------------------------------------------------------------===//
// MojoDocument: Code Action
//===----------------------------------------------------------------------===//

void MojoDocument::getCodeActions(const lsp::URIForFile &uri,
                                  const lsp::Range &pos,
                                  const lsp::CodeActionContext &context,
                                  std::vector<lsp::CodeAction> &actions) {
  // Create actions for any diagnostics in this file.
  for (auto &diag : context.diagnostics) {
    if (diag.source != "mojo")
      continue;

    // Find the fixits for this diagnostic.
    auto it = fixits.find(std::make_pair(diag.range, diag.message));
    if (it == fixits.end())
      continue;
    for (auto &action : it->second) {
      actions.emplace_back(action);
      actions.back().diagnostics = {diag};
    }
  }
}

//===----------------------------------------------------------------------===//
// MojoServer::Impl
//===----------------------------------------------------------------------===//

struct MojoServer::Impl {
  /// The runtime used when processing files.
  LLCL::Runtime runtime{LLCL::createMallocAllocator(),
                        LLCL::createThreadPoolWorkQueue()};

  /// The files held by the server, mapped by their URI file name.
  llvm::StringMap<std::unique_ptr<MojoDocument>> files;
};

//===----------------------------------------------------------------------===//
// MojoServer
//===----------------------------------------------------------------------===//

MojoServer::MojoServer() : impl(std::make_unique<Impl>()) {}
MojoServer::~MojoServer() = default;

void MojoServer::addDocument(const lsp::URIForFile &uri, StringRef contents,
                             int64_t version,
                             std::vector<lsp::Diagnostic> &diagnostics) {
  impl->files[uri.file()] = std::make_unique<MojoDocument>(
      uri, contents, version, diagnostics, impl->runtime);
}

void MojoServer::updateDocument(
    const lsp::URIForFile &uri,
    ArrayRef<lsp::TextDocumentContentChangeEvent> changes, int64_t version,
    std::vector<lsp::Diagnostic> &diagnostics) {
  auto it = impl->files.find(uri.file());
  if (it == impl->files.end())
    return;

  // Try to update the document. If we fail, erase the file from the server. A
  // failed updated generally means we've fallen out of sync somewhere.
  if (failed(it->second->update(uri, version, changes, diagnostics)))
    impl->files.erase(it);
}

std::optional<int64_t> MojoServer::removeDocument(const lsp::URIForFile &uri) {
  auto it = impl->files.find(uri.file());
  if (it == impl->files.end())
    return std::nullopt;

  int64_t version = it->second->getVersion();
  impl->files.erase(it);
  return version;
}

void MojoServer::getCodeActions(const lsp::URIForFile &uri,
                                const lsp::Range &pos,
                                const lsp::CodeActionContext &context,
                                std::vector<lsp::CodeAction> &actions) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->getCodeActions(uri, pos, context, actions);
}
