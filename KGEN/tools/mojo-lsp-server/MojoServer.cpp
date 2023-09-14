//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoServer.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoTooling/ASTDeclRef.h"
#include "KGEN/MojoTooling/ASTDeclView.h"
#include "KGEN/MojoTooling/CodeComplete.h"
#include "KGEN/MojoTooling/ParserDriver.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/AnyAsyncValueRef.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/ReferenceCounted.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"
#include "mlir/Tools/lsp-server-support/SourceMgrUtils.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include <optional>

namespace lsp = mlir::lsp;
using namespace M;
using namespace M::KGEN::LIT;
using llvm::SMLoc;
using llvm::SMRange;

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
static SMRange getRangeForText(SMLoc loc, StringRef text) {
  if (!loc.isValid())
    return {};
  return {loc, SMLoc::getFromPointer(loc.getPointer() + text.size())};
}

/// Define ordering operators for SMLoc for use in IntervalMap.
namespace llvm {
bool operator<(const SMLoc &lhs, const SMLoc &rhs) {
  return lhs.getPointer() < rhs.getPointer();
}
bool operator<=(const SMLoc &lhs, const SMLoc &rhs) {
  return lhs.getPointer() <= rhs.getPointer();
}
} // namespace llvm

//===----------------------------------------------------------------------===//
// Symbol
//===----------------------------------------------------------------------===//

namespace {
/// Common representation for any kind of symbol.
struct Symbol {
  Symbol(MojoASTDeclRef declRef, StringRef identifier, SMLoc identifierLoc)
      : identifier(identifier), declRef(declRef),
        range(getRangeForText(identifierLoc, identifier)) {}

  Symbol(const Symbol &) = delete;
  Symbol &operator=(const Symbol &) = delete;

  /// Return a nicely formatted markdown text of the declaration of this symbol.
  std::string getMarkdownDeclaration() const;

  /// Identifier of the symbol as specified in the source code.
  std::string identifier;

  /// API for accessing the internals of this decl.
  MojoASTDeclRef declRef;

  /// The location of the identifier of this decl.
  SMRange range;
};
} // namespace

/// Return if the given view kind should be included in the markdown
/// declaration.
static bool shouldIncludeViewKindInMarkdown(DeclView::DeclViewKind kind) {
  return kind != DeclView::DK_AliasDeclView &&
         kind != DeclView::DK_StructDeclView;
}

std::string Symbol::getMarkdownDeclaration() const {
  auto processView = [&](const DeclView &view) -> std::string {
    std::string buff;
    llvm::raw_string_ostream os(buff);
    if (auto snippet = view.getDeclarationSnippet(); !snippet.empty()) {
      // Add the decl prefix to the snippet, unless it's superfluous.
      std::string declPrefix;
      if (shouldIncludeViewKindInMarkdown(view.getKind()))
        declPrefix = llvm::formatv("({0}) ", view.getKindAsString()).str();

      os << llvm::formatv(R"(```mojo
{0}{1}
```)",
                          declPrefix, snippet);
    } else {
      os << formatv("### {0} `{1}`\n", view.getKindAsString(), identifier);
    }

    if (auto docString = view.getMarkdownDocString(); !docString.empty()) {
      os << llvm::formatv(R"(
---

###
{0}
)",
                          docString);
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
// SymbolRef
//===----------------------------------------------------------------------===//

namespace {
/// This struct represents a reference or a declaration in the doc managed by
/// this index to a symbol that might be defined elsewhere.
struct SymbolRef {
  SymbolRef(ArrayRef<const Symbol *> symbols, SMRange range)
      : symbols(symbols), range(range) {}
  SymbolRef(const Symbol &symbol, SMRange range) : SymbolRef(&symbol, range) {}

  /// Return a nicely formatted markdown text of this reference.
  std::string getMarkdownDeclaration() const;

  /// The symbols being referenced.
  SmallVector<const Symbol *, 1> symbols;
  /// The range in the index's doc where the symbol is being referenced.
  SMRange range;
};
} // namespace

std::string SymbolRef::getMarkdownDeclaration() const {
  // If there is only one symbol, we can simply return its markdown declaration.
  if (symbols.size() == 1)
    return symbols[0]->getMarkdownDeclaration();

  // Otherwise, build a markdown string that lists all the symbols.
  std::string output;
  llvm::raw_string_ostream os(output);
  llvm::interleave(
      symbols,
      [&](const Symbol *symbol) { os << symbol->getMarkdownDeclaration(); },
      [&] { os << "\n---\n\n"; });
  return os.str();
}

//===----------------------------------------------------------------------===//
// SymbolIndex
//===----------------------------------------------------------------------===//

namespace {
/// Database of symbols in a single file.
class SymbolIndex {
public:
  SymbolIndex(const llvm::SourceMgr &sourceMgr)
      : sourceMgr(sourceMgr), rangeToSymbolRef(allocator) {}

  /// Store a new symbol in this index, unless its name is empty.
  /// If the symbol is effectively stored, a pointer to it is returned,
  /// otherwise nullptr is returned.
  Symbol *registerSymbol(MojoASTDeclRef declRef,
                         std::optional<StringRef> identifier,
                         SMLoc identifierLoc);

  /// Store a new reference to a given set of symbols. No error is thrown if the
  /// expected symbol doesn't exist in the index.
  void registerRef(ArrayRef<MojoASTDeclRef> declRefs, SMLoc loc,
                   StringRef spelling);

  /// Look for the symbols whose declaration or references contain the given
  /// position in the document.
  SymbolRef *getSymbolAt(SMLoc loc) const;

  /// Look for the symbol corresponding to the given decl in the symbol table.
  /// Return nullptr if not found.
  Symbol *findSymbol(MojoASTDeclRef declRef);

private:
  /// Store the range corresponding to the reference or the declaration of a
  /// symbol in the main doc.
  void insertRangeInMainDoc(SymbolRef &&symbolRef);

  using MapT = llvm::IntervalMap<
      SMLoc, SymbolRef *,
      llvm::IntervalMapImpl::NodeSizer<SMLoc, Symbol *>::LeafSize,
      llvm::IntervalMapHalfOpenInfo<SMLoc>>;

  const llvm::SourceMgr &sourceMgr;
  MapT::Allocator allocator;
  MapT rangeToSymbolRef;
  SmallVector<std::unique_ptr<SymbolRef>> symbolRefs;

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

void SymbolIndex::insertRangeInMainDoc(SymbolRef &&symbolRef) {
  SMRange range = symbolRef.range;

  // If an existing mapping is found, overwrite with the new reference. We may
  // resolve more specific references as the parser progresses.
  if (auto it = rangeToSymbolRef.find(range.Start); it.valid()) {
    if (it.start() == range.Start && it.stop() == range.End &&
        it.value()->symbols.size() > symbolRef.symbols.size()) {
      it.value()->symbols = std::move(symbolRef.symbols);
      return;
    }
  }

  // Otherwise, insert a new mapping.
  if (!rangeToSymbolRef.overlaps(range.Start, range.End)) {
    symbolRefs.push_back(std::make_unique<SymbolRef>(std::move(symbolRef)));
    rangeToSymbolRef.insert(range.Start, range.End, symbolRefs.back().get());
  }
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
  if (isMainFileLoc(sourceMgr, symbol.range.Start))
    insertRangeInMainDoc({symbol, symbol.range});
  return &symbol;
}

void SymbolIndex::registerRef(ArrayRef<MojoASTDeclRef> declRefs, SMLoc loc,
                              StringRef spelling) {
  // We don't index empty spellings nor references in files other than the main
  // one.
  if (spelling.empty() || !isMainFileLoc(sourceMgr, loc))
    return;

  SmallVector<Symbol *> symbols;
  for (MojoASTDeclRef ref : declRefs) {
    // Capture the symbol if it exists, otherwise try to register it, as it
    // might come from a non-main doc.
    if (Symbol *symbol = findSymbol(ref))
      symbols.push_back(symbol);
    else if (Symbol *symbol = registerSymbol(ref, ref.getName(), ref.getLoc()))
      symbols.push_back(symbol);
  }

  if (!symbols.empty())
    insertRangeInMainDoc({symbols, getRangeForText(loc, spelling)});
}

SymbolRef *SymbolIndex::getSymbolAt(SMLoc loc) const {
  if (auto it = rangeToSymbolRef.find(loc); it.valid() && it.start() <= loc)
    return it.value();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// LSPParserListener
//===----------------------------------------------------------------------===//

namespace {
struct MojoDocument;

/// Class that is used to connect the LSP with the Mojo parser to enable
/// features like symbol indices.
class LSPParserListener : public ParserListener {
public:
  LSPParserListener(MojoDocument &mainDoc, const llvm::SourceMgr &sourceMgr)
      : mainDoc(mainDoc), sourceMgr(sourceMgr) {}

  void addSymbolDecl(ASTDecl *decl, SMLoc loc);

  bool isInterestedInLoc(llvm::SMLoc parserLoc) override {
    // We're only interested in locations in the main file.
    return isMainFileLoc(sourceMgr, parserLoc);
  }

  void onAliasDecl(ASTDecl *decl, SMLoc identifierLoc) override;

  void onArgumentDecl(ASTDecl *decl, SMLoc identifierLoc) override;

  void onFunctionDecl(ASTDecl *decl, SMLoc identifierLoc) override;

  void onModuleDecl(ASTDecl *decl, SMLoc identifierLoc) override;

  void onModuleImport(ASTDecl *decl, StringRef spelling, SMLoc loc) override;

  void onParameterDecl(ASTDecl *decl, SMLoc identifierLoc) override;

  void onStructDecl(ASTDecl *decl, SMLoc identifierLoc) override;

  void onStructFieldDecl(ASTDecl *decl, SMLoc identifierLoc) override;

  void onVariableDecl(ASTDecl *decl, SMLoc identifierLoc) override;

  void onRef(ArrayRef<ASTDecl *> decls, StringRef spelling, SMLoc loc) override;

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
struct MojoDocument : public ReferenceCounted<MojoDocument> {
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

  /// Return the source location from the given LSP position.
  SMLoc getLocFromPos(const lsp::Position &pos) {
    return pos.getAsSMLoc(context->sourceMgr);
  }

  /// Return the source range from the given LSP range.
  SMRange getLocFromPos(const lsp::Range &range) {
    return {getLocFromPos(range.start), getLocFromPos(range.end)};
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
               OnResultFn<std::vector<mlir::lsp::Location>> onDefinitionFn);

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
    isDocumentParsed.andThenAsync(
        [doc = RCRef<MojoDocument>::copy(this),
         fn = std::forward<FnT>(fn)]() mutable { fn(*doc); });
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
  getCodeActionsSync(SMRange range, const lsp::CodeActionContext &context);

  //===--------------------------------------------------------------------===//
  // Language Features

  lsp::CompletionList onCodeCompletionSync(SMLoc completeLoc) const;

  std::vector<lsp::Location> onDefinitionSync(SMLoc loc) const;

  std::optional<lsp::Hover> onHoverSync(SMLoc loc) const;

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

      parserConfig.parserListener = &parserListener;

      // TODO: Enable full caching here when we can symbolize references from
      // IR. We can enable references from imported modules though, as we just
      // need definitions from cached IR.
      parserConfig.moduleCachingLevel = ParserConfig::kCacheImports;
      parserContext =
          std::make_unique<MojoParserContext>(sourceMgr, parserConfig);
    }

    KGEN::CompilationOptions compilationOptions;
    MLIRContext mlirContext;
    ParserConfig parserConfig;
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

using MojoDocumentRef = RCRef<MojoDocument>;
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

  llvm::CrashRecoveryContext::Enable();
  llvm::CrashRecoveryContext crc;
  crc.DumpStackAndCleanupOnFailure = true;

  if (!crc.RunSafelyOnThread([&]() {
        context->parserContext->parseFile(context->sourceMgr.getMainFileID());
      })) {
    lsp::Logger::error("Crash recovered: CrashRecoveryContext::RetCode (on "
                       "POSIX: signal number + 128) = {0}",
                       crc.RetCode);
    lsp::Logger::error(
        "A crash happened in the mojo parser when processing the "
        "file {0}.\nPlease report this issue in "
        "https://github.com/modularml/mojo/issues along with all the relevant "
        "source codes with the contents they had at crash time.",
        uri);
    isInvalidated = true;
    lsp::PublishDiagnosticsParams diagParams(uri, version);
    lsp::Diagnostic lspDiag;
    lspDiag.source = "mojo";
    lspDiag.severity = lsp::DiagnosticSeverity::Error;
    lspDiag.message =
        "A crash happened in the mojo parser with the current version of this "
        "file. Please report this issue in "
        "https://github.com/modularml/mojo/issues along with all the relevant "
        "source codes with their current contents.";
    diagParams.diagnostics.push_back(lspDiag);
    sendDiagnosticsFn(diagParams);
  }

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
    if (doc.isInvalidated)
      return onActions({});
    onActions(doc.getCodeActionsSync(doc.getLocFromPos(pos), context));
  });
}

std::vector<lsp::CodeAction>
MojoDocument::getCodeActionsSync(SMRange range,
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
        if (doc.isInvalidated)
          return onCompletionFn({});
        SMLoc completeLoc = doc.getLocFromPos(completePos);
        if (!completeLoc.isValid())
          return onCompletionFn({});
        onCompletionFn(doc.onCodeCompletionSync(completeLoc));
      });
}

lsp::CompletionList
MojoDocument::onCodeCompletionSync(SMLoc completeLoc) const {
  if (!context)
    return lsp::CompletionList();
  unsigned locBuffer = context->sourceMgr.FindBufferContainingLoc(completeLoc);
  const llvm::MemoryBuffer *buffer =
      context->sourceMgr.getMemoryBuffer(locBuffer);

  // Query the mojo parser for potential completion results.
  uint64_t rawCompleteLoc =
      completeLoc.getPointer() - buffer->getBuffer().data();
  MLIRContext mlirContext(MLIRContext::Threading::DISABLED);
  std::vector<KGEN::Mojo::CodeCompletionResult> results =
      MojoParserContext::codeComplete(*buffer, rawCompleteLoc, &mlirContext,
                                      runtime, context->compilationOptions);

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
    OnResultFn<std::vector<mlir::lsp::Location>> onDefinitionFn) {
  startTaskAfterParsing([pos, onDefinitionFn = std::move(onDefinitionFn)](
                            MojoDocument &doc) mutable {
    if (doc.isInvalidated)
      return onDefinitionFn({});
    SMLoc loc = doc.getLocFromPos(pos);
    if (!loc.isValid())
      return onDefinitionFn({});
    onDefinitionFn(doc.onDefinitionSync(loc));
  });
}

std::vector<lsp::Location> MojoDocument::onDefinitionSync(SMLoc loc) const {
  SymbolRef *symbolRef = context->symbolIndex.getSymbolAt(loc);
  if (!symbolRef)
    return {};

  std::vector<lsp::Location> locations;
  for (const Symbol *symbol : symbolRef->symbols) {
    if (auto symbolUri =
            getURIFromLoc(context->sourceMgr, symbol->range.Start, uri)) {
      locations.emplace_back(*symbolUri,
                             lsp::Range(context->sourceMgr, symbol->range));
    }
  }

  return locations;
}

//===----------------------------------------------------------------------===//
// MojoDocument: Hover
//===----------------------------------------------------------------------===//

void MojoDocument::onHover(
    const lsp::Position &pos,
    OnResultFn<std::optional<mlir::lsp::Hover>> onHoverFn) {
  startTaskAfterParsing(
      [pos, onHoverFn = std::move(onHoverFn)](MojoDocument &doc) mutable {
        if (doc.isInvalidated)
          return onHoverFn({});
        SMLoc loc = doc.getLocFromPos(pos);
        onHoverFn(loc.isValid() ? doc.onHoverSync(loc) : std::nullopt);
      });
}

std::optional<lsp::Hover> MojoDocument::onHoverSync(SMLoc loc) const {
  if (auto symbolRef = context->symbolIndex.getSymbolAt(loc)) {
    lsp::Hover hover(lsp::Range(context->sourceMgr, symbolRef->range));
    hover.contents.kind = mlir::lsp::MarkupKind::Markdown;
    hover.contents.value = symbolRef->getMarkdownDeclaration();
    return hover;
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// LSPParserListener
//===----------------------------------------------------------------------===//

void LSPParserListener::addSymbolDecl(ASTDecl *decl, SMLoc loc) {
  MojoASTDeclRef declRef(decl);
  mainDoc.context->symbolIndex.registerSymbol(declRef, declRef.getName(), loc);
}

void LSPParserListener::onAliasDecl(ASTDecl *decl, SMLoc identifierLoc) {
  addSymbolDecl(decl, identifierLoc);
}

void LSPParserListener::onArgumentDecl(ASTDecl *decl, SMLoc identifierLoc) {
  addSymbolDecl(decl, identifierLoc);
}

void LSPParserListener::onFunctionDecl(ASTDecl *decl, SMLoc identifierLoc) {
  addSymbolDecl(decl, identifierLoc);
}

void LSPParserListener::onModuleDecl(ASTDecl *decl, SMLoc identifierLoc) {
  // We don't index the module of the main file.
  if (!isMainFileLoc(sourceMgr, identifierLoc))
    addSymbolDecl(decl, identifierLoc);
}

void LSPParserListener::onModuleImport(ASTDecl *decl, StringRef spelling,
                                       SMLoc loc) {
  mainDoc.context->symbolIndex.registerRef(MojoASTDeclRef(decl), loc, spelling);
}

void LSPParserListener::onStructFieldDecl(ASTDecl *decl, SMLoc identifierLoc) {
  addSymbolDecl(decl, identifierLoc);
}

void LSPParserListener::onParameterDecl(ASTDecl *decl, SMLoc identifierLoc) {
  addSymbolDecl(decl, identifierLoc);
}

void LSPParserListener::onStructDecl(ASTDecl *decl, SMLoc identifierLoc) {
  addSymbolDecl(decl, identifierLoc);
}

void LSPParserListener::onVariableDecl(ASTDecl *decl, SMLoc identifierLoc) {
  addSymbolDecl(decl, identifierLoc);
}

void LSPParserListener::onRef(ArrayRef<ASTDecl *> decls, StringRef spelling,
                              SMLoc loc) {
  mainDoc.context->symbolIndex.registerRef(
      llvm::map_to_vector(decls,
                          [](ASTDecl *decl) -> MojoASTDeclRef { return decl; }),
      loc, spelling);
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
    OnResultFn<std::vector<mlir::lsp::Location>> onDefinitionFn) {
  if (MojoDocumentRef doc = impl->findDocument(uri.file()))
    doc->onDefinition(pos, std::move(onDefinitionFn));
}

void MojoServer::onHover(
    const lsp::URIForFile &uri, const lsp::Position &pos,
    OnResultFn<std::optional<mlir::lsp::Hover>> onHoverFn) {
  if (MojoDocumentRef doc = impl->findDocument(uri.file()))
    doc->onHover(pos, std::move(onHoverFn));
}
