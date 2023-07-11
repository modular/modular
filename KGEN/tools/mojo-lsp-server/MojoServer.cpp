//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoServer.h"
#include "KGEN/CompilationOptions.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser.h"
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

/// Returns a `Range` for a given `text` that starts at the location `loc`.
static lsp::Range getRangeForText(llvm::SourceMgr &sourceMgr, SMLoc loc,
                                  StringRef text) {
  auto [line, col] = sourceMgr.getLineAndColumn(loc);
  return {lsp::Position(line - 1, col - 1),
          lsp::Position(line - 1, col + text.size() - 1)};
}

//===----------------------------------------------------------------------===//
// Symbol
//===----------------------------------------------------------------------===//

namespace {
/// Common representation for any kind of symbol.
struct Symbol {
  Symbol(MojoASTDeclRef declRef, StringRef identifier,
         const lsp::Range &identifierRange)
      : identifier(identifier), declRef(declRef),
        identifierRange(identifierRange) {}

  Symbol(const Symbol &) = delete;
  Symbol &operator=(const Symbol &) = delete;

  /// Identifier of the symbol as specified in the source code.
  std::string identifier;

  /// API for accessing the internals of this decl.
  MojoASTDeclRef declRef;

  /// The document range where the symbol name was declared.
  lsp::Range identifierRange;
};
} // namespace

//===----------------------------------------------------------------------===//
// SymbolPrinter
//===----------------------------------------------------------------------===//

namespace {
/// Class used to print user readable representations of symbols and their
/// metadata.
class SymbolPrinter {
public:
  SymbolPrinter(const Symbol &symbol) : symbol(symbol) {}

  /// Return a code snippet that summarized the declaration of the symbol.
  std::string getDeclarationCodeSnippet() const;

  /// Return a nicely formatted markdown text of the declaration of this symbol.
  std::string getMarkdownDeclaration() const;

  StringRef getSymbolKindAsString() const;

private:
  const Symbol &symbol;
};
} // namespace

StringRef SymbolPrinter::getSymbolKindAsString() const {
  return TypeSwitch<Operation &, StringRef>(*symbol.declRef.getIfOperation())
      .Case<VarLetDeclOp, LetRegDeclOp>([&](auto op) { return "variable"; });
}

std::string SymbolPrinter::getDeclarationCodeSnippet() const {
  std::string buff;
  llvm::raw_string_ostream os(buff);

  auto &rawOp = *symbol.declRef.getIfOperation();
  TypeSwitch<Operation &>(rawOp)
      .Case<VarLetDeclOp>([&](auto op) {
        os << (op.getIsVar() ? "var" : "let");

        os << " " << symbol.identifier;
        if (auto typeRef = symbol.declRef.getType())
          os << ": " << typeRef.getAsString();
      })
      .Case<LetRegDeclOp>([&](auto op) {
        os << "let " << symbol.identifier;
        if (auto typeRef = symbol.declRef.getType())
          os << ": " << typeRef.getAsString();
      });
  return buff;
}

std::string SymbolPrinter::getMarkdownDeclaration() const {
  return llvm::formatv(R"(### {0} `{1}`

---

###
```mojo
{2}
```)",
                       getSymbolKindAsString(), symbol.identifier,
                       getDeclarationCodeSnippet());
}

//===----------------------------------------------------------------------===//
// SymbolIndex
//===----------------------------------------------------------------------===//

namespace {
/// Database of symbols in a single file.
class SymbolIndex {
public:
  SymbolIndex() : rangeToSymbol(allocator) {}

  /// Store a new symbol in this index.
  template <typename... Args>
  void registerSymbol(MojoASTDeclRef declRef, Args &&...args);

  /// Store a new reference to a symbol in this index. No error is thrown if the
  /// expected symbol doesn't exist in the index.
  void registerRef(MojoASTDeclRef declRef, const lsp::Range &refRange);

  /// Look for the symbol whose declaration or references contain the given
  /// position in the document.
  Symbol *getSymbolAt(const lsp::Position &position) const;

  /// Remove all symbols and references in this index.
  void clear();

private:
  using MapT = llvm::IntervalMap<
      lsp::Position, Symbol *,
      llvm::IntervalMapImpl::NodeSizer<lsp::Position, Symbol *>::LeafSize,
      llvm::IntervalMapHalfOpenInfo<lsp::Position>>;

  MapT::Allocator allocator;
  MapT rangeToSymbol;
  /// Mapping from an opaque pointer of a MojoASTDeclRef to an LSP Symbol.
  llvm::DenseMap<void *, std::unique_ptr<Symbol>> symbolTable;
};
} // namespace

template <typename... Args>
void SymbolIndex::registerSymbol(MojoASTDeclRef declRef, Args &&...args) {
  auto [it, _] = symbolTable.try_emplace(
      declRef.getAsVoidPointer(), std::make_unique<Symbol>(declRef, args...));
  Symbol *symbol = it->second.get();
  rangeToSymbol.insert(symbol->identifierRange.start,
                       symbol->identifierRange.end, symbol);
}

void SymbolIndex::registerRef(MojoASTDeclRef declRef,
                              const lsp::Range &refRange) {
  auto it = symbolTable.find(declRef.getAsVoidPointer());
  if (it == symbolTable.end())
    return;
  rangeToSymbol.insert(refRange.start, refRange.end, it->getSecond().get());
}

Symbol *SymbolIndex::getSymbolAt(const lsp::Position &position) const {
  return rangeToSymbol.lookup(position, nullptr);
}

void SymbolIndex::clear() {
  symbolTable.clear();
  rangeToSymbol.clear();
}

//===----------------------------------------------------------------------===//
// LSPParserListener
//===----------------------------------------------------------------------===//

namespace {
struct MojoDocument;

/// Class that is used to connect the LSP with the Mojo parser to enable
/// features like symbol indices.
class LSPParserListener : public MojoParserListener {
public:
  LSPParserListener(MojoDocument &mainDoc, llvm::SourceMgr &sourceMgr)
      : mainDoc(mainDoc), sourceMgr(sourceMgr) {}

  void onVariableDecl(MojoASTDeclRef declRef, SMLoc identifierLoc) override;

  void onRef(MojoASTDeclRef declRef, StringRef spelling, SMLoc loc) override;

private:
  /// The main doc for which parsing was initiated.
  MojoDocument &mainDoc;
  llvm::SourceMgr &sourceMgr;
};
} // namespace

//===----------------------------------------------------------------------===//
// MojoDocument
//===----------------------------------------------------------------------===//

namespace {
/// This class represents all of the information pertaining to a specific Mojo
/// document.
struct MojoDocument {
public:
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
  buildLspDiagnosticFromSMDiagnostic(llvm::SourceMgr &sourceMgr,
                                     ArrayRef<llvm::SMDiagnostic> diags,
                                     const lsp::URIForFile &uri);

  //===--------------------------------------------------------------------===//
  // Code Actions

  void getCodeActions(const lsp::URIForFile &uri, const lsp::Range &pos,
                      const lsp::CodeActionContext &context,
                      std::vector<lsp::CodeAction> &actions);

  //===--------------------------------------------------------------------===//
  // Language Features

  std::optional<lsp::Location> onDefinition(const lsp::Position &pos) const;

  std::optional<lsp::Hover> onHover(const lsp::Position &pos) const;

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  /// A collection of MLIR and Mojo related entities used to invoke the parser.
  /// Its lifetime is tied to that of the AST objects gotten from the parser.
  /// It also sets up a SourceMgr with the given MojoDocument as its main file.
  struct Context {
    Context(LLCL::Runtime &runtime, MojoDocument &mainDoc)
        : mlirContext(MLIRContext::Threading::DISABLED),
          parserConfig(&mlirContext, runtime, compilationOptions),
          parserListener(mainDoc, sourceMgr) {
      // We add the main doc to the SourceMgr here to ensure it's considered the
      // "main" file.
      auto buffer = llvm::MemoryBuffer::getMemBuffer(mainDoc.contents,
                                                     mainDoc.uri.file());
      sourceMgr.AddNewSourceBuffer(std::move(buffer), SMLoc());

      parserConfig.validateDocStrings = true;
      parserConfig.parserListener = &parserListener;
      parserContext =
          std::make_unique<MojoParserContext>(sourceMgr, parserConfig);
    }

    KGEN::CompilationOptions compilationOptions;
    MLIRContext mlirContext;
    MojoParserConfig parserConfig;
    llvm::SourceMgr sourceMgr;
    LSPParserListener parserListener;
    std::unique_ptr<MojoParserContext> parserContext;
  };

  /// The uri of the file.
  lsp::URIForFile uri;

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

  /// An index of all symbols in this document.
  SymbolIndex symbolIndex;

  /// The overall parser context.
  std::unique_ptr<Context> context;
};
} // namespace

MojoDocument::MojoDocument(const lsp::URIForFile &uri, StringRef contents,
                           int64_t version,
                           std::vector<lsp::Diagnostic> &diagnostics,
                           LLCL::Runtime &runtime)
    : uri(uri), contents(contents.str()), version(version), runtime(runtime) {}

void MojoDocument::initialize(const lsp::URIForFile &uri,
                              std::vector<lsp::Diagnostic> &diagnostics) {
  // Reset the source manager and parse the file.
  context = std::make_unique<Context>(runtime, *this);

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
  context->sourceMgr.setDiagHandler(handlerFn, &handlerCtx);

  context->parserContext->parseFile(context->sourceMgr.getMainFileID());

  // Process the collected diagnostics.
  for (ArrayRef<llvm::SMDiagnostic> diags : handlerCtx.smDiagnostics) {
    if (auto lspDiag =
            buildLspDiagnosticFromSMDiagnostic(context->sourceMgr, diags, uri))
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
  symbolIndex.clear();

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
  StringRef result = code.ltrim().split('\n').first;

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
std::optional<lsp::Diagnostic> MojoDocument::buildLspDiagnosticFromSMDiagnostic(
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
// MojoDocument: Language Features
//===----------------------------------------------------------------------===//

std::optional<lsp::Location>
MojoDocument::onDefinition(const lsp::Position &pos) const {
  if (Symbol *symbol = symbolIndex.getSymbolAt(pos))
    return lsp::Location(uri, symbol->identifierRange);

  return std::nullopt;
}

std::optional<lsp::Hover>
MojoDocument::onHover(const lsp::Position &pos) const {
  Symbol *symbol = symbolIndex.getSymbolAt(pos);
  if (!symbol)
    return std::nullopt;

  lsp::Hover hover(symbol->identifierRange);
  hover.contents.kind = mlir::lsp::MarkupKind::Markdown;
  hover.contents.value = SymbolPrinter(*symbol).getMarkdownDeclaration();
  return hover;
}

//===----------------------------------------------------------------------===//
// LSPParserListener
//===----------------------------------------------------------------------===//

void LSPParserListener::onVariableDecl(MojoASTDeclRef declRef,
                                       SMLoc identifierLoc) {
  // For now we don't index files other than the main one.
  if (!isMainFileLoc(sourceMgr, identifierLoc))
    return;

  if (std::optional<StringRef> identifier = declRef.getName()) {
    mainDoc.symbolIndex.registerSymbol(
        declRef, *identifier,
        getRangeForText(sourceMgr, identifierLoc, *identifier));
  }
}

void LSPParserListener::onRef(MojoASTDeclRef declRef, StringRef spelling,
                              SMLoc loc) {
  // For now we don't index files other than the main one.
  if (!isMainFileLoc(sourceMgr, loc) ||
      !isMainFileLoc(sourceMgr, declRef.getLoc()))
    return;

  mainDoc.symbolIndex.registerRef(declRef,
                                  getRangeForText(sourceMgr, loc, spelling));
}

//===----------------------------------------------------------------------===//
// MojoServer::Impl
//===----------------------------------------------------------------------===//

struct MojoServer::Impl {
  /// Retrieve the document that matches completely the given filename. Return
  /// `nullptr` if no document is found.
  MojoDocument *findDocument(StringRef filename) {
    auto it = files.find(filename);
    if (it == files.end())
      return nullptr;
    return it->second.get();
  }

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
  auto [it, _] = impl->files.try_emplace(
      uri.file(), std::make_unique<MojoDocument>(uri, contents, version,
                                                 diagnostics, impl->runtime));
  auto &document = *it->second;
  document.initialize(uri, diagnostics);
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
  if (MojoDocument *doc = impl->findDocument(uri.file()))
    doc->getCodeActions(uri, pos, context, actions);
}

std::optional<lsp::Location>
MojoServer::onDefinition(const lsp::URIForFile &uri, const lsp::Position &pos) {
  if (MojoDocument *doc = impl->findDocument(uri.file()))
    return doc->onDefinition(pos);

  return std::nullopt;
}

std::optional<lsp::Hover> MojoServer::onHover(const lsp::URIForFile &uri,
                                              const lsp::Position &pos) {
  if (MojoDocument *doc = impl->findDocument(uri.file()))
    return doc->onHover(pos);

  return std::nullopt;
}
