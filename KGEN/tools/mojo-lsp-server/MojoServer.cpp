//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoServer.h"
#include "MojoDocument.h"

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
using namespace M::Mojo::LSP;
using namespace M::KGEN::LIT;
using llvm::SMLoc;
using llvm::SMRange;

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

/// Returns a `SMRange` for a given `text` that starts at the location `loc`.
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
  SymbolIndex(MojoDocument &mainDoc)
      : mainDoc(mainDoc), sourceMgr(mainDoc.getSourceMgr()),
        rangeToSymbolRef(allocator) {}

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

  MojoDocument &mainDoc;
  const llvm::SourceMgr &sourceMgr;
  MapT::Allocator allocator;
  MapT rangeToSymbolRef;
  SmallVector<std::unique_ptr<SymbolRef>> symbolRefs;

  /// Mapping from an ASTDecl to an LSP Symbol.
  llvm::DenseMap<ASTDecl *, std::unique_ptr<Symbol>> symbolTable;
};
} // namespace

Symbol *SymbolIndex::findSymbol(MojoASTDeclRef declRef) {
  if (auto it = symbolTable.find(&*declRef); it != symbolTable.end())
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
      &*declRef, std::make_unique<Symbol>(declRef, *identifier, identifierLoc));
  Symbol &symbol = *it->second;

  // We only add symbols to the range map if they belong to the main file.
  if (mainDoc.containsLocation(symbol.range.Start))
    insertRangeInMainDoc({symbol, symbol.range});
  return &symbol;
}

void SymbolIndex::registerRef(ArrayRef<MojoASTDeclRef> declRefs, SMLoc loc,
                              StringRef spelling) {
  // We don't index empty spellings nor references in files other than the main
  // one.
  if (spelling.empty() || !mainDoc.containsLocation(loc))
    return;

  SmallVector<Symbol *> symbols;
  for (MojoASTDeclRef ref : declRefs) {
    // Capture the symbol if it exists, otherwise try to register it, as it
    // might come from a non-main doc.
    if (Symbol *symbol = findSymbol(ref))
      symbols.push_back(symbol);
    else if (Symbol *symbol = registerSymbol(
                 ref, ref.getName(), mainDoc.translateParserLoc(ref.getLoc())))
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
/// Class that is used to connect the LSP with the Mojo parser to enable
/// features like symbol indices.
class LSPParserListener : public ParserListener {
public:
  LSPParserListener(MojoDocument &mainDoc, SymbolIndex &symbolIndex)
      : mainDoc(mainDoc), symbolIndex(symbolIndex) {}

  void addSymbolDecl(ASTDecl *decl, SMLoc loc);

  bool isInterestedInLoc(SMLoc parserLoc) override {
    // We're only interested in locations in the main file.
    return mainDoc.containsLocation(mainDoc.translateParserLoc(parserLoc));
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
  SymbolIndex &symbolIndex;
};
} // namespace

void LSPParserListener::addSymbolDecl(ASTDecl *decl, SMLoc loc) {
  MojoASTDeclRef declRef(decl);
  symbolIndex.registerSymbol(declRef, declRef.getName(),
                             mainDoc.translateParserLoc(loc));
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
  if (!mainDoc.containsLocation(mainDoc.translateParserLoc(identifierLoc)))
    addSymbolDecl(decl, identifierLoc);
}

void LSPParserListener::onModuleImport(ASTDecl *decl, StringRef spelling,
                                       SMLoc loc) {
  symbolIndex.registerRef(MojoASTDeclRef(decl), mainDoc.translateParserLoc(loc),
                          spelling);
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
  symbolIndex.registerRef(
      llvm::map_to_vector(decls,
                          [](ASTDecl *decl) -> MojoASTDeclRef { return decl; }),
      mainDoc.translateParserLoc(loc), spelling);
}

//===----------------------------------------------------------------------===//
// LSPMojoREPLListener
//===----------------------------------------------------------------------===//

namespace {
/// This class implements a parser listener that communicates between the Mojo
/// parser and the LSP.
class LSPMojoREPLListener : public MojoParserREPLListener {
public:
  LSPMojoREPLListener(
      llvm::SourceMgr &sourceMgr,
      SmallVectorImpl<std::pair<StringRef, mlir::Type>> &newPersistentVariables)
      : newPersistentVariables(newPersistentVariables),
        diagHandler(sourceMgr.getDiagHandler()),
        diagHandlerContext(sourceMgr.getDiagContext()) {}
  ~LSPMojoREPLListener() override = default;

  //===--------------------------------------------------------------------===//
  // Notifications

  void notifyWrappedExpr(StringRef wrappedExpr) override {}
  void notifyFixedExpr(StringRef fixedExpr) override {}
  void notifyDiagnostics(ArrayRef<llvm::SMDiagnostic> diagnostics) override {
    for (const llvm::SMDiagnostic &diag : diagnostics)
      diagHandler(diag, diagHandlerContext);
  }

  //===--------------------------------------------------------------------===//
  // Queries

  bool shouldPersistVariable(StringRef name, mlir::Type type) override {
    newPersistentVariables.emplace_back(name, type);
    return true;
  }

private:
  StringRef currentModuleName;
  SmallVectorImpl<std::pair<StringRef, mlir::Type>> &newPersistentVariables;

  /// The main diagnostic handler used to notify diagnostics.
  llvm::SourceMgr::DiagHandlerTy diagHandler;
  void *diagHandlerContext;
};
} // namespace

//===----------------------------------------------------------------------===//
// MojoDocument::Context
//===----------------------------------------------------------------------===//

/// A collection of MLIR and Mojo related entities used to invoke the parser.
/// Its lifetime is tied to that of the AST objects gotten from the parser.
/// It also sets up a SourceMgr with the given MojoDocument as its main file.
struct MojoDocument::Context {
  Context(MojoDocument &mainDoc)
      : mlirContext(MLIRContext::Threading::DISABLED),
        parserConfig(&mlirContext, mainDoc.getRuntime(), compilationOptions),
        symbolIndex(mainDoc), parserListener(mainDoc, symbolIndex) {
    parserConfig.parserListener = &parserListener;

    // TODO: Enable full caching here when we can symbolize references from
    // IR. We can enable references from imported modules though, as we just
    // need definitions from cached IR.
    parserConfig.moduleCachingLevel = isa<MojoTextDocument>(mainDoc)
                                          ? ParserConfig::kCacheImports
                                          : ParserConfig::kCacheNone;
    parserContext =
        std::make_unique<MojoParserContext>(mainDoc.sourceMgr, parserConfig);
  }

  KGEN::CompilationOptions compilationOptions;
  MLIRContext mlirContext;
  ParserConfig parserConfig;
  SymbolIndex symbolIndex;
  LSPParserListener parserListener;
  std::unique_ptr<MojoParserContext> parserContext;
};

//===----------------------------------------------------------------------===//
// MojoDocument
//===----------------------------------------------------------------------===//

MojoDocument::MojoDocument(Kind kind, ArrayRef<lsp::URIForFile> uris,
                           int64_t version,
                           SendDiagnosticsFnRef sendDiagnosticsFn,
                           LLCL::Runtime &runtime, LLCL::AnyAsyncValueRef chain)
    : kind(kind), uris(uris), version(version),
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

  llvm::CrashRecoveryContext::Enable();
  llvm::CrashRecoveryContext crc;
  crc.DumpStackAndCleanupOnFailure = true;

  context = std::make_unique<Context>(*this);
  if (!crc.RunSafelyOnThread([&]() { parseDocumentImpl(); })) {
    lsp::Logger::error("Crash recovered: CrashRecoveryContext::RetCode (on "
                       "POSIX: signal number + 128) = {0}",
                       crc.RetCode);
    lsp::Logger::error(
        "A crash happened in the mojo parser when processing the "
        "file {0}.\nPlease report this issue in "
        "https://github.com/modularml/mojo/issues along with all the relevant "
        "source codes with the contents they had at crash time.",
        uris.front());
    isInvalidated = true;
    lsp::PublishDiagnosticsParams diagParams(uris.front(), version);
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
  llvm::StringMap<std::optional<lsp::PublishDiagnosticsParams>> fileToDiags;
  for (auto &uri : uris)
    fileToDiags[uri.file()].emplace(uri, version);

  for (ArrayRef<llvm::SMDiagnostic> diags : handlerCtx.smDiagnostics) {
    // Skip diagnostics that weren't emitted within the main file.
    if (!containsLocation(diags.front().getLoc()))
      continue;
    // Get the URI for the file this diagnostic is in. In the case of a text
    // document, this is always the main URI.
    lsp::URIForFile diagUri = uris.front();
    if (uris.size() > 1) {
      std::optional<lsp::URIForFile> optDiagUri =
          getURIFromLoc(diags.front().getLoc());
      if (!optDiagUri)
        continue;
      diagUri = *optDiagUri;
    }

    // Build the LSP diagnostic.
    if (auto lspDiag =
            buildLspDiagnosticFromSMDiagnostic(sourceMgr, diags, diagUri))
      fileToDiags[diagUri.file()]->diagnostics.push_back(*lspDiag);
  }
  for (auto &params : llvm::make_second_range(fileToDiags))
    sendDiagnosticsFn(*params);

  // Mark the document as fully parsed now that we're done.
  markDocumentParsed();
}

const KGEN::CompilationOptions &MojoDocument::getCompilationOptions() const {
  return context->compilationOptions;
}

MojoParserContext &MojoDocument::getParserContext() const {
  return *context->parserContext;
}

void MojoDocument::invalidate() {
  if (isInvalidated)
    return;
  isInvalidated = true;

  // Mark the document as parsed to unblock chained events, and let them
  // invalidate themselves.
  markDocumentParsed();
}

void MojoDocument::markDocumentParsed() {
  std::lock_guard<std::mutex> lock(isDocumentParsedMutex);
  if (!isDocumentParsed.isReady())
    isDocumentParsed.copy().emplace();
}

//===----------------------------------------------------------------------===//
// MojoDocument: Document Utilities
//===----------------------------------------------------------------------===//

std::optional<lsp::URIForFile> MojoDocument::getURIFromLoc(SMLoc loc) {
  int bufferId = sourceMgr.FindBufferContainingLoc(loc);
  if (bufferId == 0)
    return std::nullopt;

  // If this is a contained location, we can directly get the URI for it.
  if (containsLocation(loc))
    return getURIFromContainedLoc(loc);

  llvm::Expected<lsp::URIForFile> fileForLoc = lsp::URIForFile::fromFile(
      sourceMgr.getBufferInfo(bufferId).Buffer->getBufferIdentifier(), "file");
  if (fileForLoc)
    return *fileForLoc;
  lsp::Logger::error("Failed to create URI for include file: {0}",
                     llvm::toString(fileForLoc.takeError()));
  return std::nullopt;
}

std::optional<lsp::Location>
MojoDocument::getLocationFromDiag(const llvm::SMDiagnostic &diag) {
  if (std::optional<lsp::URIForFile> diagUri = getURIFromLoc(diag.getLoc()))
    return lsp::Location(*diagUri, getRangeFromDiag(sourceMgr, diag));
  return std::nullopt;
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
buildCodeActionFromSMFixit(const llvm::SMFixIt &fixit, MojoDocument &doc,
                           const lsp::URIForFile &mainFileURI) {
  llvm::SMRange range = fixit.getRange();
  if (!range.isValid())
    return std::nullopt;

  // Get the file this fixit is in.
  auto uri = doc.getURIFromLoc(range.Start);
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
      {lsp::Range(doc.getSourceMgr(), range), fixit.getText().str()});
  return action;
}

/// Convert the given MLIR diagnostic to the LSP form.
std::optional<lsp::Diagnostic> MojoDocument::buildLspDiagnosticFromSMDiagnostic(
    llvm::SourceMgr &sourceMgr, ArrayRef<llvm::SMDiagnostic> diags,
    const lsp::URIForFile &uri) {
  const llvm::SMDiagnostic &mainDiag = diags[0];

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
      if (auto loc = getLocationFromDiag(note))
        relatedDiags.emplace_back(*loc, note.getMessage().str());
    lspDiag.relatedInformation = std::move(relatedDiags);
  }

  // Collect fixits for the diagnostic.
  std::vector<lsp::CodeAction> diagFixits;
  for (const llvm::SMFixIt &fixit : mainDiag.getFixIts())
    if (auto action = buildCodeActionFromSMFixit(fixit, *this, uri))
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
    const mlir::lsp::URIForFile &uri, const lsp::Range &pos,
    const lsp::CodeActionContext &context,
    OnResultFn<std::vector<mlir::lsp::CodeAction>> onActions) {
  startTaskAfterParsing([uri, pos, context, onActions = std::move(onActions)](
                            MojoDocument &doc) mutable {
    if (doc.isInvalidated)
      return onActions({});
    onActions(doc.getCodeActionsSync(doc.getLocFromPos(uri, pos), context));
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
    const mlir::lsp::URIForFile &uri, const lsp::Position &completePos,
    OnResultFn<mlir::lsp::CompletionList> onCompletionFn) {
  startTaskAfterParsing(
      [uri, completePos,
       onCompletionFn = std::move(onCompletionFn)](MojoDocument &doc) mutable {
        if (doc.isInvalidated)
          return onCompletionFn({});
        SMLoc completeLoc = doc.getLocFromPos(uri, completePos);
        if (!completeLoc.isValid())
          return onCompletionFn({});
        onCompletionFn(doc.onCodeCompletionSync(completeLoc));
      });
}

lsp::CompletionList MojoDocument::onCodeCompletionSync(SMLoc completeLoc) {
  if (!context)
    return lsp::CompletionList();

  // Map the Mojo results to LSP results.
  lsp::CompletionList completionList;
  for (const KGEN::Mojo::CodeCompletionResult &it :
       onCodeCompletionSyncImpl(completeLoc)) {
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
    const mlir::lsp::URIForFile &uri, const lsp::Position &pos,
    OnResultFn<std::vector<mlir::lsp::Location>> onDefinitionFn) {
  startTaskAfterParsing([uri, pos, onDefinitionFn = std::move(onDefinitionFn)](
                            MojoDocument &doc) mutable {
    if (doc.isInvalidated)
      return onDefinitionFn({});
    SMLoc loc = doc.getLocFromPos(uri, pos);
    if (!loc.isValid())
      return onDefinitionFn({});
    onDefinitionFn(doc.onDefinitionSync(loc));
  });
}

std::vector<lsp::Location> MojoDocument::onDefinitionSync(SMLoc loc) {
  SymbolRef *symbolRef = context->symbolIndex.getSymbolAt(loc);
  if (!symbolRef)
    return {};

  std::vector<lsp::Location> locations;
  for (const Symbol *symbol : symbolRef->symbols)
    if (auto uri = getURIFromLoc(symbol->range.Start))
      locations.emplace_back(*uri, lsp::Range(getSourceMgr(), symbol->range));
  return locations;
}

//===----------------------------------------------------------------------===//
// MojoDocument: Hover
//===----------------------------------------------------------------------===//

void MojoDocument::onHover(
    const mlir::lsp::URIForFile &uri, const lsp::Position &pos,
    OnResultFn<std::optional<mlir::lsp::Hover>> onHoverFn) {
  startTaskAfterParsing(
      [uri, pos, onHoverFn = std::move(onHoverFn)](MojoDocument &doc) mutable {
        if (doc.isInvalidated)
          return onHoverFn({});
        SMLoc loc = doc.getLocFromPos(uri, pos);
        if (!loc.isValid())
          return onHoverFn({});
        onHoverFn(doc.onHoverSync(loc));
      });
}

std::optional<lsp::Hover> MojoDocument::onHoverSync(SMLoc loc) {
  if (auto symbolRef = context->symbolIndex.getSymbolAt(loc)) {
    lsp::Hover hover(lsp::Range(getSourceMgr(), symbolRef->range));
    hover.contents.kind = mlir::lsp::MarkupKind::Markdown;
    hover.contents.value = symbolRef->getMarkdownDeclaration();
    return hover;
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// MojoTextDocument
//===----------------------------------------------------------------------===//

MojoTextDocument::MojoTextDocument(const lsp::URIForFile &uri,
                                   std::string &&contents, int64_t version,
                                   SendDiagnosticsFnRef sendDiagnosticsFn,
                                   LLCL::Runtime &runtime,
                                   LLCL::AnyAsyncValueRef chain)
    : MojoDocument(Kind::kTextDocument, uri, version, sendDiagnosticsFn,
                   runtime, std::move(chain)),
      contents(std::move(contents)) {
  // We add the main doc to the SourceMgr here to ensure it's considered the
  // "main" file.
  getSourceMgr().AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(this->contents, uri.file()), SMLoc());
}

void MojoTextDocument::parseDocumentImpl() {
  getParserContext().parseFile(getSourceMgr().getMainFileID());
}

const mlir::lsp::URIForFile &
MojoTextDocument::getURIFromContainedLoc(SMLoc loc) {
  return getURIs().front();
}

bool MojoTextDocument::containsLocation(SMLoc loc) {
  return getSourceMgr().FindBufferContainingLoc(loc) ==
         getSourceMgr().getMainFileID();
}

SMLoc MojoTextDocument::getLocFromPos(const mlir::lsp::URIForFile &uri,
                                      mlir::lsp::Position position) {
  return position.getAsSMLoc(getSourceMgr());
}

//===----------------------------------------------------------------------===//
// MojoTextDocument: Code Completion
//===----------------------------------------------------------------------===//

std::vector<KGEN::Mojo::CodeCompletionResult>
MojoTextDocument::onCodeCompletionSyncImpl(SMLoc completeLoc) {
  llvm::SourceMgr &sourceMgr = getSourceMgr();
  const llvm::MemoryBuffer *buffer =
      sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());

  // Query the mojo parser for potential completion results.
  uint64_t rawCompletePos =
      completeLoc.getPointer() - buffer->getBuffer().data();
  MLIRContext mlirContext(MLIRContext::Threading::DISABLED);
  return MojoParserContext::codeComplete(*buffer, rawCompletePos, &mlirContext,
                                         getRuntime(), getCompilationOptions());
}

//===----------------------------------------------------------------------===//
// MojoNotebookDocument
//===----------------------------------------------------------------------===//

MojoNotebookDocument::MojoNotebookDocument(
    ArrayRef<mlir::lsp::URIForFile> notebookAndCellURIs, int64_t version,
    ArrayRef<mlir::lsp::NotebookCell> cellInfos,
    ArrayRef<mlir::lsp::TextDocumentItem> cellDocuments,
    SendDiagnosticsFnRef sendDiagnosticsFn, LLCL::Runtime &runtime,
    LLCL::AnyAsyncValueRef chain)
    : MojoDocument(Kind::kNotebookDocument, notebookAndCellURIs, version,
                   sendDiagnosticsFn, runtime, std::move(chain)) {
  for (unsigned i = 0, e = cellInfos.size(); i < e; ++i) {
    if (cellInfos[i].kind != lsp::NotebookCellKind::Code)
      continue;
    auto &doc = cellDocuments[i];

    auto &cell = cells.emplace_back(std::make_unique<Cell>(doc.uri, doc.text));
    cell->bufferId = getSourceMgr().AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBuffer(cell->contents, cell->uri.file()),
        SMLoc());
    uriToCell.try_emplace(doc.uri.file(), cells.back().get());
  }
}

void MojoNotebookDocument::parseDocumentImpl() {
  SmallVector<std::pair<StringRef, mlir::Type>> persistentVariables;
  LSPMojoREPLListener listener(getSourceMgr(), persistentVariables);

  // Parse each of the cells in the notebook.
  MojoParserContext &ctx = getParserContext();
  for (Cell &cell : getCells()) {
    // Ignore cells that contain python expressions.
    // TODO: Extract the variables that are implicitly imported into mojo and
    // create stub definitions so that future cells can reference them without
    // error.
    if (StringRef(cell.contents).starts_with("%%python"))
      continue;

    cell.decl = ctx.parseREPLExpresion(listener, cell.bufferId, "lsp_repl_main",
                                       persistentVariables);
  }
}

const mlir::lsp::URIForFile &
MojoNotebookDocument::getURIFromContainedLoc(SMLoc loc) {
  size_t bufferId = getSourceMgr().FindBufferContainingLoc(loc);
  assert(bufferId && bufferId <= cells.size() &&
         "expected to find buffer containing location");
  return cells[bufferId - 1]->uri;
}

bool MojoNotebookDocument::containsLocation(SMLoc loc) {
  int locBufferId = getSourceMgr().FindBufferContainingLoc(loc);
  if (locBufferId == 0)
    return false;
  // Check that the buffer corresponds to one of the cells.
  return locBufferId <= static_cast<int>(cells.size());
}

SMLoc MojoNotebookDocument::translateParserLoc(SMLoc loc) {
  auto newLoc = getParserContext().getREPLLocMapper().mapLocation(loc);
  return newLoc.isValid() ? newLoc : loc;
}

SMLoc MojoNotebookDocument::getLocFromPos(const mlir::lsp::URIForFile &uri,
                                          mlir::lsp::Position position) {
  Cell &cell = *uriToCell[uri.file()];
  return getSourceMgr().FindLocForLineAndColumn(
      cell.bufferId, position.line + 1, position.character + 1);
}

//===----------------------------------------------------------------------===//
// MojoNotebookDocument: Code Completion
//===----------------------------------------------------------------------===//

std::vector<KGEN::Mojo::CodeCompletionResult>
MojoNotebookDocument::onCodeCompletionSyncImpl(SMLoc completeLoc) {
  // TODO: Support notebook documents.
  return {};
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
    notebookCellToFile.clear();
    runtime.reset();
  }

  /// Return if the server is shutting down.
  bool isShuttingDown() const { return !runtime; }

  /// Retrieve the document that matches completely the given filename. Return
  /// `nullptr` if no document is found.
  MojoDocumentRef findDocument(StringRef filename) {
    if (auto it = files.find(filename); it != files.end())
      return it->second.copy();

    auto it = notebookCellToFile.find(filename);
    return it != notebookCellToFile.end() ? it->second.copy()
                                          : MojoDocumentRef();
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

  /// A mapping from individual notebook cells to their documents.
  llvm::StringMap<MojoDocumentRef> notebookCellToFile;
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

//===----------------------------------------------------------------------===//
// Document Management

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
  it->second = MojoTextDocumentRef::create(uri, std::move(contents), version,
                                           impl->sendDiagnosticsFn,
                                           *impl->runtime, std::move(chain));
}

void MojoServer::updateDocument(
    const lsp::URIForFile &uri,
    ArrayRef<lsp::TextDocumentContentChangeEvent> changes, int64_t version) {
  auto it = impl->files.find(uri.file());
  if (it == impl->files.end())
    return;
  MojoTextDocument *textDoc = dyn_cast<MojoTextDocument>(&*it->second);
  if (!textDoc) {
    lsp::Logger::error("Updating a non-text document: {0}", uri.file());
    return;
  }

  // Try to update the document. If we fail, erase the file from the server. A
  // failed updated generally means we've fallen out of sync somewhere.
  std::string contents = textDoc->getContents().str();
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

//===----------------------------------------------------------------------===//
// Notebook Document Management

void MojoServer::addNotebookDocument(
    const mlir::lsp::URIForFile &uri, ArrayRef<mlir::lsp::NotebookCell> cells,
    int64_t version, ArrayRef<mlir::lsp::TextDocumentItem> cellDocuments) {
  if (impl->isShuttingDown())
    return;
  MojoDocumentRef &file = impl->files[uri.file()];

  // If a document already exists, invalidate that version.
  AnyAsyncValueRef chain = AsyncValueRef<Chain>::createReady(*impl->runtime);
  if (file) {
    file->invalidate();

    // Chain the new document to the old one.
    chain = file->getDocumentReadyChain();
  }

  // Build the list of URIs for the document and cells.
  SmallVector<lsp::URIForFile> docURIs(1, uri);
  for (auto [index, cell] : llvm::enumerate(cellDocuments))
    docURIs.push_back(cell.uri);

  // Create a new document.
  file = MojoNotebookDocumentRef::create(docURIs, version, cells, cellDocuments,
                                         impl->sendDiagnosticsFn,
                                         *impl->runtime, std::move(chain));
  for (const mlir::lsp::TextDocumentItem &cell : cellDocuments)
    impl->notebookCellToFile[cell.uri.file()] = file.copy();
}

void MojoServer::removeNotebookDocument(
    const mlir::lsp::URIForFile &uri,
    ArrayRef<mlir::lsp::TextDocumentIdentifier> cellDocuments) {
  // Remove the document from the server using the same flow as a normal text
  // document.
  removeDocument(uri);

  // Clear out mappings from the cell documents to the notebook document.
  for (const mlir::lsp::TextDocumentIdentifier &cell : cellDocuments)
    impl->notebookCellToFile.erase(cell.uri.file());
}

void MojoServer::updateNotebookDocument(
    const lsp::URIForFile &uri, int64_t version,
    const lsp::NotebookDocumentChangeEvent &change) {
  auto it = impl->files.find(uri.file());
  if (it == impl->files.end())
    return;
  MojoNotebookDocument *doc = dyn_cast<MojoNotebookDocument>(&*it->second);
  if (!doc) {
    lsp::Logger::error("Updating a non-notebook document: {0}", uri.file());
    return;
  }

  // Grab all of the current cells and their documents.
  std::vector<lsp::NotebookCell> cells;
  std::vector<lsp::TextDocumentItem> cellDocs;
  for (MojoNotebookDocument::Cell &cell : doc->getCells()) {
    cells.push_back({lsp::NotebookCellKind::Code, cell.uri});
    cellDocs.push_back({cell.uri, "mojo", cell.contents, version});
  }

  // Apply updates to the cells.
  if (change.cells) {
    // Check for structure changes.
    if (auto &cellStructure = change.cells->structure) {
      auto &array = cellStructure->array;

      // Erase the deleted cells.
      for (const lsp::NotebookCell &cell :
           ArrayRef(cells).slice(array.start, array.deleteCount)) {
        impl->notebookCellToFile.erase(cell.document.file());
      }
      cells.erase(cells.begin() + array.start,
                  cells.begin() + array.start + array.deleteCount);
      cellDocs.erase(cellDocs.begin() + array.start,
                     cellDocs.begin() + array.start + array.deleteCount);

      // Insert any new cells.
      cells.insert(cells.begin() + array.start, array.cells.begin(),
                   array.cells.end());
      for (const lsp::NotebookCell &cell : llvm::reverse(array.cells)) {
        lsp::TextDocumentItem docItem{cell.document, "mojo", "", version};
        cellDocs.insert(cellDocs.begin() + array.start, docItem);
      }
    }

    // Map the cell uri the index of the cell.
    llvm::StringMap<unsigned> cellURIToIndex;
    for (auto [index, cell] : llvm::enumerate(cells))
      cellURIToIndex.try_emplace(cell.document.file(), index);

    // Apply updates to the cell properties.
    for (auto &cellUpdate : change.cells->data) {
      auto it = cellURIToIndex.find(cellUpdate.document.file());
      if (it != cellURIToIndex.end())
        cells[it->second].kind = cellUpdate.kind;
    }

    // Apply updates to the cell contents.
    for (auto &content : change.cells->textContent) {
      auto it = cellURIToIndex.find(content.document.uri.file());
      if (it == cellURIToIndex.end())
        continue;
      // Try to update the document. If we fail, erase the file from the
      // server. A failed updated generally means we've fallen out of sync
      // somewhere.
      if (failed(lsp::TextDocumentContentChangeEvent::applyTo(
              content.changes, cellDocs[it->second].text))) {
        lsp::Logger::error("Failed to update contents of {0}", uri.file());

        SmallVector<lsp::TextDocumentIdentifier> cellDocuments;
        for (auto &cell : cells)
          cellDocuments.push_back({cell.document});
        return removeNotebookDocument(uri, cellDocuments);
      }
    }
  }

  // Overrite the original document with the new contents.
  addNotebookDocument(uri, cells, version, cellDocs);
}

//===----------------------------------------------------------------------===//
// Queries

void MojoServer::getCodeActions(
    const lsp::URIForFile &uri, const lsp::Range &pos,
    const lsp::CodeActionContext &context,
    OnResultFn<std::vector<mlir::lsp::CodeAction>> onActionsFn) {
  if (MojoDocumentRef doc = impl->findDocument(uri.file()))
    doc->getCodeActions(uri, pos, context, std::move(onActionsFn));
}

void MojoServer::onCodeCompletion(
    const lsp::URIForFile &uri, const lsp::Position &completePos,
    OnResultFn<mlir::lsp::CompletionList> onCompletionFn) {
  if (MojoDocumentRef doc = impl->findDocument(uri.file()))
    doc->onCodeCompletion(uri, completePos, std::move(onCompletionFn));
}

void MojoServer::onDefinition(
    const lsp::URIForFile &uri, const lsp::Position &pos,
    OnResultFn<std::vector<mlir::lsp::Location>> onDefinitionFn) {
  if (MojoDocumentRef doc = impl->findDocument(uri.file()))
    doc->onDefinition(uri, pos, std::move(onDefinitionFn));
}

void MojoServer::onHover(
    const lsp::URIForFile &uri, const lsp::Position &pos,
    OnResultFn<std::optional<mlir::lsp::Hover>> onHoverFn) {
  if (MojoDocumentRef doc = impl->findDocument(uri.file()))
    doc->onHover(uri, pos, std::move(onHoverFn));
}
