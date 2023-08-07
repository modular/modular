//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoServer.h"
#include "KGEN/CompilationOptions.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser.h"
#include "KGEN/MojoParser/ASTDeclRef.h"
#include "KGEN/MojoParser/ASTDeclView.h"
#include "KGEN/MojoParser/CodeComplete.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/AnyAsyncValueRef.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/ReferenceCounted.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"
#include "mlir/Tools/lsp-server-support/SourceMgrUtils.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/StringMap.h"
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
  if (bufferId == 0)
    return std::nullopt;
  if (bufferId == static_cast<int>(mgr.getMainFileID()))
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
static bool isMainFileLoc(const llvm::SourceMgr &mgr, SMLoc loc) {
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
static lsp::Range getRangeForText(const llvm::SourceMgr &sourceMgr, SMLoc loc,
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
  Symbol(MojoASTDeclRef declRef, StringRef identifier, SMLoc identifierLoc)
      : identifier(identifier), declRef(declRef), identifierLoc(identifierLoc) {
  }

  Symbol(const Symbol &) = delete;
  Symbol &operator=(const Symbol &) = delete;

  /// Return a nicely formatted markdown text of the declaration of this symbol.
  std::string getMarkdownDeclaration() const;

  /// Get the LSP range of the identifier at the declaration point.
  lsp::Range getIdentifierRange(const llvm::SourceMgr &sourceMgr) const {
    return getRangeForText(sourceMgr, identifierLoc, identifier);
  }

  /// Identifier of the symbol as specified in the source code.
  std::string identifier;

  /// API for accessing the internals of this decl.
  MojoASTDeclRef declRef;

  /// The location of the identifier of this decl.
  SMLoc identifierLoc;
};
} // namespace

std::string Symbol::getMarkdownDeclaration() const {
  auto processView = [&](const DeclView &view) -> std::string {
    std::string buff;
    llvm::raw_string_ostream os(buff);
    os << formatv("### {0} `{1}`\n", view.getKindAsString(), identifier);
    if (auto docString = view.getMarkdownDocString(); !docString.empty()) {
      os << llvm::formatv(R"(
---

###
{0}
)",
                          docString);
    }

    if (auto snippet = view.getDeclarationSnippet(); !snippet.empty()) {
      os << llvm::formatv(R"(
---

###
```mojo
{0}
```)",
                          snippet);
    }
    return buff;
  };

  if (auto view = declRef.getView())
    return processView(*view);
  // If didn't get a view, we fall back to simply printing the name of the
  // entity.
  if (auto name = declRef.getName())
    return formatv("### `{0}`", *name);
  return {};
}

//===----------------------------------------------------------------------===//
// SymbolIndex
//===----------------------------------------------------------------------===//

namespace {
/// Database of symbols in a single file.
class SymbolIndex {
public:
  /// This struct represents a reference or a declaration in the doc managed by
  /// this index to a symbol that might be defined elsewhere.
  struct SymbolRef {
    SymbolRef(const Symbol &symbol, const lsp::Range &range)
        : symbol(symbol), range(range) {}

    /// The symbol being referenced.
    const Symbol &symbol;
    /// The range in the index's doc where the symbol is being referenced.
    lsp::Range range;
  };

  SymbolIndex(const llvm::SourceMgr &sourceMgr)
      : sourceMgr(sourceMgr), rangeToSymbol(allocator) {}

  /// Store a new symbol in this index, unless its name is empty.
  /// If the symbol is effectively stored, a pointer to it is returned,
  /// otherwise nullptr is returned.
  Symbol *registerSymbol(MojoASTDeclRef declRef,
                         std::optional<StringRef> identifier,
                         SMLoc identifierLoc);

  /// Store a new reference to a symbol. No error is thrown if the expected
  /// symbol doesn't exist in the index.
  void registerRef(MojoASTDeclRef declRef, SMLoc loc, StringRef spelling);

  /// Look for the symbol whose declaration or references contain the given
  /// position in the document.
  std::optional<SymbolIndex::SymbolRef>
  getSymbolAt(const lsp::Position &position) const;

  /// Look for the symbol corresponding to the given decl in the symbol table.
  /// Return nullptr if not found.
  Symbol *findSymbol(MojoASTDeclRef declRef);

private:
  /// Store the range corresponding to the reference or the declaration of a
  /// symbol in the main doc.
  void insertRangeInMainDoc(const lsp::Range &range, Symbol &symbol);

  using MapT = llvm::IntervalMap<
      lsp::Position, Symbol *,
      llvm::IntervalMapImpl::NodeSizer<lsp::Position, Symbol *>::LeafSize,
      llvm::IntervalMapHalfOpenInfo<lsp::Position>>;

  const llvm::SourceMgr &sourceMgr;
  MapT::Allocator allocator;
  MapT rangeToSymbol;
  /// Mapping from an opaque pointer of a MojoASTDeclRef to an LSP Symbol.
  llvm::DenseMap<void *, std::unique_ptr<Symbol>> symbolTable;
};
} // namespace

Symbol *SymbolIndex::findSymbol(MojoASTDeclRef declRef) {
  if (auto it = symbolTable.find(declRef.getAsVoidPointer());
      it != symbolTable.end())
    return it->getSecond().get();
  return nullptr;
}

void SymbolIndex::insertRangeInMainDoc(const lsp::Range &range,
                                       Symbol &symbol) {
  if (!rangeToSymbol.overlaps(range.start, range.end))
    rangeToSymbol.insert(range.start, range.end, &symbol);
}

Symbol *SymbolIndex::registerSymbol(MojoASTDeclRef declRef,
                                    std::optional<StringRef> identifier,
                                    SMLoc identifierLoc) {
  // We don't index symbols without a proper name.
  if (!identifier.has_value() || identifier->empty())
    return nullptr;

  auto [it, _] = symbolTable.try_emplace(
      declRef.getAsVoidPointer(),
      std::make_unique<Symbol>(declRef, *identifier, identifierLoc));
  Symbol &symbol = *it->second;

  // We only add symbols to the range map if they belong to the main file.
  if (isMainFileLoc(sourceMgr, symbol.identifierLoc))
    insertRangeInMainDoc(symbol.getIdentifierRange(sourceMgr), symbol);
  return &symbol;
}

void SymbolIndex::registerRef(MojoASTDeclRef declRef, SMLoc loc,
                              StringRef spelling) {
  // We don't index empty spellings nor references in files other than the main
  // one.
  if (spelling.empty() || !isMainFileLoc(sourceMgr, loc))
    return;

  Symbol *symbol = findSymbol(declRef);

  // If we don't have the symbol in the symbol table, we try to register it,
  // as it might come from a non-main doc.
  if (!symbol)
    symbol = registerSymbol(declRef, declRef.getName(), declRef.getLoc());

  if (symbol)
    insertRangeInMainDoc(getRangeForText(sourceMgr, loc, spelling), *symbol);
}

std::optional<SymbolIndex::SymbolRef>
SymbolIndex::getSymbolAt(const lsp::Position &position) const {
  if (auto it = rangeToSymbol.find(position);
      it.valid() && it.start() <= position) {
    return SymbolIndex::SymbolRef(*it.value(),
                                  lsp::Range(it.start(), it.stop()));
  }
  return std::nullopt;
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
  LSPParserListener(MojoDocument &mainDoc, const llvm::SourceMgr &sourceMgr)
      : mainDoc(mainDoc), sourceMgr(sourceMgr) {}

  bool isInterestedInLoc(llvm::SMLoc parserLoc) override {
    // We're only interested in locations in the main file.
    return isMainFileLoc(sourceMgr, parserLoc);
  }

  void onAliasDecl(MojoASTDeclRef declRef, llvm::SMLoc identifierLoc) override;

  void onArgumentDecl(MojoASTDeclRef declRef,
                      llvm::SMLoc identifierLoc) override;

  void onFunctionDecl(MojoASTDeclRef declRef, SMLoc identifierLoc) override;

  void onModuleDecl(MojoASTDeclRef declRef, SMLoc identifierLoc) override;

  void onModuleImport(MojoASTDeclRef declRef, StringRef spelling,
                      SMLoc loc) override;

  void onParameterDecl(MojoASTDeclRef declRef, SMLoc identifierLoc) override;

  void onStructDecl(MojoASTDeclRef declRef, SMLoc identifierLoc) override;

  void onStructFieldDecl(MojoASTDeclRef declRef, SMLoc identifierLoc) override;

  void onVariableDecl(MojoASTDeclRef declRef, SMLoc identifierLoc) override;

  void onRef(MojoASTDeclRef declRef, StringRef spelling, SMLoc loc) override;

private:
  /// The main doc for which parsing was initiated.
  MojoDocument &mainDoc;
  const llvm::SourceMgr &sourceMgr;
};
} // namespace

//===----------------------------------------------------------------------===//
// MojoDocument
//===----------------------------------------------------------------------===//

using SendDiagnosticsFnRef =
    function_ref<void(const mlir::lsp::PublishDiagnosticsParams &)>;

namespace {
/// This class represents all of the information pertaining to a specific Mojo
/// document.
struct MojoDocument : public LLCL::ReferenceCounted<MojoDocument> {
public:
  MojoDocument(const lsp::URIForFile &uri, std::string &&contents,
               int64_t version, SendDiagnosticsFnRef sendDiagnosticsFn,
               LLCL::Runtime &runtime, LLCL::AnyAsyncValueRef chain);
  MojoDocument(const MojoDocument &) = delete;
  MojoDocument &operator=(const MojoDocument &) = delete;

  /// Return the contents of this document.
  StringRef getContents() const { return contents; }

  /// Return the version of this document.
  int64_t getVersion() const { return version; }

  /// Invalidate this document.
  void invalidate() {
    isInvalidated = true;

    // Mark the document as parsed to unblock chained events, and let them
    // invalidate themselves.
    markDocumentParsed();
  }

  /// Return a chain that will be ready when the document is parsed.
  AnyAsyncValueRef getDocumentReadyChain() const {
    return isDocumentParsed.copy();
  }

  //===--------------------------------------------------------------------===//
  // Asynchronous LSP Queries
  //===--------------------------------------------------------------------===//

  //===--------------------------------------------------------------------===//
  // Code Actions

  void getCodeActions(const lsp::Range &pos,
                      const lsp::CodeActionContext &context,
                      OnResultFn<std::vector<mlir::lsp::CodeAction>> onActions);

  //===--------------------------------------------------------------------===//
  // Language Features

  void onCodeCompletion(const lsp::Position &completePos,
                        OnResultFn<mlir::lsp::CompletionList> onCompletionFn);

  void
  onDefinition(const lsp::Position &pos,
               OnResultFn<std::optional<mlir::lsp::Location>> onDefinitionFn);

  void onHover(const lsp::Position &pos,
               OnResultFn<std::optional<mlir::lsp::Hover>> onHoverFn);

private:
  /// Parse the document and populate the index based on the current contents.
  void parseDocument();

  /// Mark the current document as being finished parsing.
  void markDocumentParsed() {
    std::lock_guard<std::mutex> lock(isDocumentParsedMutex);
    if (!isDocumentParsed.isReady())
      isDocumentParsed.copy().emplace();
  }

  /// Start a task that depends on the document being parsed.
  template <typename FnT>
  void startTaskAfterParsing(FnT &&fn) {
    isDocumentParsed.andThenAsync([doc = LLCL::RCRef<MojoDocument>::copy(this),
                                   fn = std::forward<FnT>(fn)]() mutable {
      // If the document has been invalidated, there's nothing to do here.
      if (!doc->isInvalidated)
        fn(*doc);
    });
  }

  //===--------------------------------------------------------------------===//
  // Synchronous LSP Queries
  //===--------------------------------------------------------------------===//

  //===--------------------------------------------------------------------===//
  // Diagnostics

  std::optional<lsp::Diagnostic>
  buildLspDiagnosticFromSMDiagnostic(llvm::SourceMgr &sourceMgr,
                                     ArrayRef<llvm::SMDiagnostic> diags,
                                     const lsp::URIForFile &uri);

  //===--------------------------------------------------------------------===//
  // Code Actions

  std::vector<lsp::CodeAction>
  getCodeActionsSync(const lsp::Range &pos,
                     const lsp::CodeActionContext &context);

  //===--------------------------------------------------------------------===//
  // Language Features

  lsp::CompletionList
  onCodeCompletionSync(const lsp::Position &completePos) const;

  std::optional<lsp::Location> onDefinitionSync(const lsp::Position &pos) const;

  std::optional<lsp::Hover> onHoverSync(const lsp::Position &pos) const;

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
          symbolIndex(sourceMgr), parserListener(mainDoc, sourceMgr) {
      // We add the main doc to the SourceMgr here to ensure it's considered the
      // "main" file.
      auto buffer = llvm::MemoryBuffer::getMemBuffer(mainDoc.contents,
                                                     mainDoc.uri.file());
      sourceMgr.AddNewSourceBuffer(std::move(buffer), SMLoc());

      parserConfig.validateDocStrings = true;
      parserConfig.parserListener = &parserListener;

      // TODO: Enable full caching here when we can symbolize references from
      // IR. We can enable references from imported modules though, as we just
      // need definitions from cached IR.
      parserConfig.moduleCachingLevel = MojoParserConfig::kCacheImports;
      parserContext =
          std::make_unique<MojoParserContext>(sourceMgr, parserConfig);
    }

    KGEN::CompilationOptions compilationOptions;
    MLIRContext mlirContext;
    MojoParserConfig parserConfig;
    llvm::SourceMgr sourceMgr;
    SymbolIndex symbolIndex;
    LSPParserListener parserListener;
    std::unique_ptr<MojoParserContext> parserContext;
  };

  //===--------------------------------------------------------------------===//
  // Static Fields

  /// The following fields are always available for access and don't require
  /// additional synchronization.

  /// The uri of the file.
  lsp::URIForFile uri;

  /// The full string contents of the file.
  std::string contents;

  /// The version of this file.
  int64_t version = 0;

  /// The function used to send diagnostics for this document.
  SendDiagnosticsFnRef sendDiagnosticsFn;

  /// The runtime used when parsing the file.
  LLCL::Runtime &runtime;

  /// A flag indicating if this document version has been invalidated.
  std::atomic<bool> isInvalidated = false;

  //===--------------------------------------------------------------------===//
  // Parsed Fields

  /// The following fields are only available after the document has been
  /// parsed, when `isDocumentParsed` is ready.

  /// Allow access to the parser fields.
  friend LSPParserListener;

  /// An async value readied when the document is parsed.
  AsyncValueRef<Chain> isDocumentParsed;
  std::mutex isDocumentParsedMutex;

  /// An ordered set of fixits for diagnostics emitted for the current version
  /// of the file.
  std::map<std::pair<lsp::Range, std::string>, std::vector<lsp::CodeAction>>
      fixits;

  /// The overall parser context.
  std::unique_ptr<Context> context;
};

using MojoDocumentRef = LLCL::RCRef<MojoDocument>;
} // namespace

MojoDocument::MojoDocument(const lsp::URIForFile &uri, std::string &&contents,
                           int64_t version,
                           SendDiagnosticsFnRef sendDiagnosticsFn,
                           LLCL::Runtime &runtime, LLCL::AnyAsyncValueRef chain)
    : uri(uri), contents(std::move(contents)), version(version),
      sendDiagnosticsFn(sendDiagnosticsFn), runtime(runtime),
      isDocumentParsed(AsyncValueRef<Chain>::allocate(runtime)) {
  // Start a task to resolve the document.
  chain.andThenAsync(
      [doc = MojoDocumentRef::copy(this)] { doc->parseDocument(); });
}

void MojoDocument::parseDocument() {
  // If we've already been invalidated, bail out early.
  if (isInvalidated)
    return markDocumentParsed();

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

  // If we've already been invalidated, bail out early.
  if (isInvalidated)
    return markDocumentParsed();

  // Process the collected diagnostics.
  lsp::PublishDiagnosticsParams diagParams(uri, version);
  for (ArrayRef<llvm::SMDiagnostic> diags : handlerCtx.smDiagnostics) {
    if (auto lspDiag =
            buildLspDiagnosticFromSMDiagnostic(context->sourceMgr, diags, uri))
      diagParams.diagnostics.push_back(*lspDiag);
  }
  sendDiagnosticsFn(diagParams);

  // Mark the document as fully parsed now that we're done.
  markDocumentParsed();
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

void MojoDocument::getCodeActions(
    const lsp::Range &pos, const lsp::CodeActionContext &context,
    OnResultFn<std::vector<mlir::lsp::CodeAction>> onActions) {
  startTaskAfterParsing([pos, context, onActions = std::move(onActions)](
                            MojoDocument &doc) mutable {
    onActions(doc.getCodeActionsSync(pos, context));
  });
}

std::vector<lsp::CodeAction>
MojoDocument::getCodeActionsSync(const lsp::Range &pos,
                                 const lsp::CodeActionContext &context) {
  // Create actions for any diagnostics in this file.
  std::vector<lsp::CodeAction> actions;
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
  return actions;
}

//===----------------------------------------------------------------------===//
// MojoDocument: Code Completion
//===----------------------------------------------------------------------===//

void MojoDocument::onCodeCompletion(
    const lsp::Position &completePos,
    OnResultFn<mlir::lsp::CompletionList> onCompletionFn) {
  startTaskAfterParsing(
      [completePos,
       onCompletionFn = std::move(onCompletionFn)](MojoDocument &doc) mutable {
        onCompletionFn(doc.onCodeCompletionSync(completePos));
      });
}

lsp::CompletionList
MojoDocument::onCodeCompletionSync(const lsp::Position &completePos) const {
  if (!context)
    return lsp::CompletionList();
  SMLoc posLoc = completePos.getAsSMLoc(context->sourceMgr);
  if (!posLoc.isValid())
    return lsp::CompletionList();
  unsigned locBuffer = context->sourceMgr.FindBufferContainingLoc(posLoc);
  const llvm::MemoryBuffer *buffer =
      context->sourceMgr.getMemoryBuffer(locBuffer);

  // Query the mojo parser for potential completion results.
  uint64_t rawCompletePos = posLoc.getPointer() - buffer->getBuffer().data();
  MLIRContext mlirContext(MLIRContext::Threading::DISABLED);
  std::vector<KGEN::Mojo::CodeCompletionResult> results =
      KGEN::Mojo::codeComplete(*buffer, rawCompletePos, &mlirContext, runtime,
                               context->compilationOptions);

  // Map the Mojo results to LSP results.
  lsp::CompletionList completionList;
  for (const KGEN::Mojo::CodeCompletionResult &it : results) {
    lsp::CompletionItem item;
    item.label = it.label;
    item.sortText = std::to_string(static_cast<unsigned>(it.kind));

    switch (it.kind) {
    case KGEN::Mojo::CodeCompletionResult::kUnknown:
      item.kind = lsp::CompletionItemKind::Missing;
      break;
    case KGEN::Mojo::CodeCompletionResult::kModule:
      item.kind = lsp::CompletionItemKind::Module;
      break;
    case KGEN::Mojo::CodeCompletionResult::kPackage:
      item.kind = lsp::CompletionItemKind::Folder;
      break;
    case KGEN::Mojo::CodeCompletionResult::kStruct:
      item.kind = lsp::CompletionItemKind::Struct;
      break;
    case KGEN::Mojo::CodeCompletionResult::kFunction:
      item.kind = lsp::CompletionItemKind::Function;
      break;
    case KGEN::Mojo::CodeCompletionResult::kField:
      item.kind = lsp::CompletionItemKind::Field;
      break;
    }

    if (!it.documentation.empty())
      item.documentation = {lsp::MarkupKind::Markdown, it.documentation};
    completionList.items.push_back(item);
  }
  return completionList;
}

//===----------------------------------------------------------------------===//
// MojoDocument: Definitions and References
//===----------------------------------------------------------------------===//

void MojoDocument::onDefinition(
    const lsp::Position &pos,
    OnResultFn<std::optional<mlir::lsp::Location>> onDefinitionFn) {
  startTaskAfterParsing([pos, onDefinitionFn = std::move(onDefinitionFn)](
                            MojoDocument &doc) mutable {
    onDefinitionFn(doc.onDefinitionSync(pos));
  });
}

std::optional<lsp::Location>
MojoDocument::onDefinitionSync(const lsp::Position &pos) const {
  if (auto symbolRef = context->symbolIndex.getSymbolAt(pos)) {
    auto &symbol = symbolRef->symbol;
    if (auto symbolUri =
            getURIFromLoc(context->sourceMgr, symbol.identifierLoc, uri)) {
      return lsp::Location(*symbolUri, getRangeForText(context->sourceMgr,
                                                       symbol.identifierLoc,
                                                       symbol.identifier));
    }
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// MojoDocument: Hover
//===----------------------------------------------------------------------===//

void MojoDocument::onHover(
    const lsp::Position &pos,
    OnResultFn<std::optional<mlir::lsp::Hover>> onHoverFn) {
  startTaskAfterParsing(
      [pos, onHoverFn = std::move(onHoverFn)](MojoDocument &doc) mutable {
        onHoverFn(doc.onHoverSync(pos));
      });
}

std::optional<lsp::Hover>
MojoDocument::onHoverSync(const lsp::Position &pos) const {
  if (auto symbolRef = context->symbolIndex.getSymbolAt(pos)) {
    lsp::Hover hover(symbolRef->range);
    hover.contents.kind = mlir::lsp::MarkupKind::Markdown;
    hover.contents.value = symbolRef->symbol.getMarkdownDeclaration();
    return hover;
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// LSPParserListener
//===----------------------------------------------------------------------===//

void LSPParserListener::onAliasDecl(MojoASTDeclRef declRef,
                                    SMLoc identifierLoc) {
  mainDoc.context->symbolIndex.registerSymbol(declRef, declRef.getName(),
                                              identifierLoc);
}

void LSPParserListener::onArgumentDecl(MojoASTDeclRef declRef,
                                       SMLoc identifierLoc) {
  mainDoc.context->symbolIndex.registerSymbol(declRef, declRef.getName(),
                                              identifierLoc);
}

void LSPParserListener::onFunctionDecl(MojoASTDeclRef declRef,
                                       SMLoc identifierLoc) {
  mainDoc.context->symbolIndex.registerSymbol(declRef, declRef.getName(),
                                              identifierLoc);
}

void LSPParserListener::onModuleDecl(MojoASTDeclRef declRef,
                                     SMLoc identifierLoc) {
  // We don't index the module of the main file.
  if (!isMainFileLoc(sourceMgr, identifierLoc))
    mainDoc.context->symbolIndex.registerSymbol(declRef, declRef.getName(),
                                                identifierLoc);
}

void LSPParserListener::onModuleImport(MojoASTDeclRef declRef,
                                       StringRef spelling, SMLoc loc) {
  mainDoc.context->symbolIndex.registerRef(declRef, loc, spelling);
}

void LSPParserListener::onStructFieldDecl(MojoASTDeclRef declRef,
                                          SMLoc identifierLoc) {
  mainDoc.context->symbolIndex.registerSymbol(declRef, declRef.getName(),
                                              identifierLoc);
}

void LSPParserListener::onParameterDecl(MojoASTDeclRef declRef,
                                        SMLoc identifierLoc) {
  mainDoc.context->symbolIndex.registerSymbol(declRef, declRef.getName(),
                                              identifierLoc);
}

void LSPParserListener::onStructDecl(MojoASTDeclRef declRef,
                                     SMLoc identifierLoc) {
  mainDoc.context->symbolIndex.registerSymbol(declRef, declRef.getName(),
                                              identifierLoc);
}

void LSPParserListener::onVariableDecl(MojoASTDeclRef declRef,
                                       SMLoc identifierLoc) {
  mainDoc.context->symbolIndex.registerSymbol(declRef, declRef.getName(),
                                              identifierLoc);
}

void LSPParserListener::onRef(MojoASTDeclRef declRef, StringRef spelling,
                              SMLoc loc) {
  mainDoc.context->symbolIndex.registerRef(declRef, loc, spelling);
}

//===----------------------------------------------------------------------===//
// MojoServer::Impl
//===----------------------------------------------------------------------===//

struct MojoServer::Impl {
  Impl(std::unique_ptr<LLCL::WorkQueue> workQueue, bool waitOnShutdown,
       SendDiagnosticsFn sendDiagnosticsFn)
      : runtime(std::make_unique<LLCL::Runtime>(LLCL::createMallocAllocator(),
                                                std::move(workQueue))),
        waitOnShutdown(waitOnShutdown),
        sendDiagnosticsFn(std::move(sendDiagnosticsFn)) {}

  /// Begin the shutdown process for the server.
  void shutdown() {
    if (isShuttingDown())
      return;
    // Invalidate all of the current documents if we aren't waiting for
    // shutdown, otherwise wait for them to parse and resolve actions.
    for (auto &it : files) {
      if (waitOnShutdown)
        LLCL::await(it.second->getDocumentReadyChain());
      else
        it.second->invalidate();
    }
    files.clear();
    runtime.reset();
  }

  /// Return if the server is shutting down.
  bool isShuttingDown() const { return !runtime; }

  /// Retrieve the document that matches completely the given filename. Return
  /// `nullptr` if no document is found.
  MojoDocumentRef findDocument(StringRef filename) {
    auto it = files.find(filename);
    if (it == files.end())
      return MojoDocumentRef();
    return it->second.copy();
  }

  /// The runtime used when processing files.
  std::unique_ptr<LLCL::Runtime> runtime;

  /// A flag indicating if the server should not invalidate requests on
  /// shutdown, and instead wait for them to complete.
  bool waitOnShutdown;

  /// The function used to send diagnostics to the client.
  SendDiagnosticsFn sendDiagnosticsFn;

  /// The files held by the server, mapped by their URI file name.
  llvm::StringMap<MojoDocumentRef> files;
};

//===----------------------------------------------------------------------===//
// MojoServer
//===----------------------------------------------------------------------===//

MojoServer::MojoServer(std::unique_ptr<LLCL::WorkQueue> workQueue,
                       bool waitOnShutdown, SendDiagnosticsFn sendDiagnosticsFn)
    : impl(std::make_unique<Impl>(std::move(workQueue), waitOnShutdown,
                                  std::move(sendDiagnosticsFn))) {}
MojoServer::~MojoServer() { shutdown(); }

void MojoServer::shutdown() { impl->shutdown(); }

void MojoServer::addDocument(const lsp::URIForFile &uri, std::string &&contents,
                             int64_t version) {
  if (impl->isShuttingDown())
    return;
  auto [it, _] = impl->files.try_emplace(uri.file(), MojoDocumentRef());

  // If a document already exists, invalidate that version.
  AnyAsyncValueRef chain = AsyncValueRef<Chain>::createReady(*impl->runtime);
  if (it->second) {
    it->second->invalidate();

    // Chain the new document to the old one.
    chain = it->second->getDocumentReadyChain();
  }

  // Create a new document.
  it->second = MojoDocumentRef::create(uri, std::move(contents), version,
                                       impl->sendDiagnosticsFn, *impl->runtime,
                                       std::move(chain));
}

void MojoServer::updateDocument(
    const lsp::URIForFile &uri,
    ArrayRef<lsp::TextDocumentContentChangeEvent> changes, int64_t version) {
  auto it = impl->files.find(uri.file());
  if (it == impl->files.end())
    return;

  // Try to update the document. If we fail, erase the file from the server. A
  // failed updated generally means we've fallen out of sync somewhere.
  std::string contents = it->second->getContents().str();
  if (failed(lsp::TextDocumentContentChangeEvent::applyTo(changes, contents))) {
    lsp::Logger::error("Failed to update contents of {0}", uri.file());
    return removeDocument(uri);
  }

  // Overrite the original document with the new contents.
  addDocument(uri, std::move(contents), version);
}

void MojoServer::removeDocument(const lsp::URIForFile &uri) {
  auto it = impl->files.find(uri.file());
  if (it == impl->files.end())
    return;

  // Empty out the diagnostics shown for this document. This will clear out
  // anything currently displayed by the client for this document (e.g. in the
  // "Problems" pane of VSCode).
  impl->sendDiagnosticsFn(
      lsp::PublishDiagnosticsParams(uri, it->second->getVersion()));
  it->second->invalidate();
  impl->files.erase(it);
}

void MojoServer::getCodeActions(
    const lsp::URIForFile &uri, const lsp::Range &pos,
    const lsp::CodeActionContext &context,
    OnResultFn<std::vector<mlir::lsp::CodeAction>> onActionsFn) {
  if (MojoDocumentRef doc = impl->findDocument(uri.file()))
    doc->getCodeActions(pos, context, std::move(onActionsFn));
}

void MojoServer::onCodeCompletion(
    const lsp::URIForFile &uri, const lsp::Position &completePos,
    OnResultFn<mlir::lsp::CompletionList> onCompletionFn) {
  if (MojoDocumentRef doc = impl->findDocument(uri.file()))
    doc->onCodeCompletion(completePos, std::move(onCompletionFn));
}

void MojoServer::onDefinition(
    const lsp::URIForFile &uri, const lsp::Position &pos,
    OnResultFn<std::optional<mlir::lsp::Location>> onDefinitionFn) {
  if (MojoDocumentRef doc = impl->findDocument(uri.file()))
    doc->onDefinition(pos, std::move(onDefinitionFn));
}

void MojoServer::onHover(
    const lsp::URIForFile &uri, const lsp::Position &pos,
    OnResultFn<std::optional<mlir::lsp::Hover>> onHoverFn) {
  if (MojoDocumentRef doc = impl->findDocument(uri.file()))
    doc->onHover(pos, std::move(onHoverFn));
}
