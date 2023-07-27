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
#include "LLCL/Runtime/Runtime.h"
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
  SymbolIndex(const llvm::SourceMgr &sourceMgr)
      : sourceMgr(sourceMgr), rangeToSymbol(allocator) {}

  /// Store a new symbol in this index.
  template <typename... Args>
  void registerSymbol(MojoASTDeclRef declRef,
                      std::optional<StringRef> identifier, Args &&...args);

  /// Store a new reference to a symbol. No error is thrown if the expected
  /// symbol doesn't exist in the index.
  void registerRef(MojoASTDeclRef declRef, SMLoc loc, StringRef spelling);

  /// Look for the symbol whose declaration or references contain the given
  /// position in the document.
  Symbol *getSymbolAt(const lsp::Position &position) const;

private:
  /// Store the range corresponding to the reference or the declaration of a
  /// symbol in the main doc.
  void insertRangeInMainDoc(const lsp::Range &range, Symbol &symbol) {
    rangeToSymbol.insert(range.start, range.end, &symbol);
  }

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

template <typename... Args>
void SymbolIndex::registerSymbol(MojoASTDeclRef declRef,
                                 std::optional<StringRef> identifier,
                                 Args &&...args) {
  // We don't index symbols without a proper name.
  if (!identifier.has_value() || identifier->empty())
    return;

  auto [it, _] = symbolTable.try_emplace(
      declRef.getAsVoidPointer(),
      std::make_unique<Symbol>(declRef, *identifier, args...));
  Symbol &symbol = *it->second;

  // We only add symbols to the range map if they belong to the main file.
  if (isMainFileLoc(sourceMgr, symbol.identifierLoc))
    insertRangeInMainDoc(symbol.getIdentifierRange(sourceMgr), symbol);
}

void SymbolIndex::registerRef(MojoASTDeclRef declRef, SMLoc loc,
                              StringRef spelling) {
  // We don't index empty spellings nor references in files other than the main
  // one.
  if (spelling.empty() || !isMainFileLoc(sourceMgr, loc))
    return;

  auto it = symbolTable.find(declRef.getAsVoidPointer());
  // If haven't indexed the decl, we do nothing.
  if (it == symbolTable.end())
    return;

  insertRangeInMainDoc(getRangeForText(sourceMgr, loc, spelling),
                       *it->getSecond());
}

Symbol *SymbolIndex::getSymbolAt(const lsp::Position &position) const {
  return rangeToSymbol.lookup(position, nullptr);
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

  void onAliasDecl(MojoASTDeclRef declRef, llvm::SMLoc identifierLoc) override;

  void onArgumentDecl(MojoASTDeclRef declRef,
                      llvm::SMLoc identifierLoc) override;

  void onFunctionDecl(MojoASTDeclRef declRef, SMLoc identifierLoc) override;

  void onModuleDecl(MojoASTDeclRef declRef, SMLoc identifierLoc) override;

  void onModuleImport(MojoASTDeclRef declRef, StringRef spelling,
                      SMLoc loc) override;

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
struct MojoDocument {
public:
  MojoDocument(const lsp::URIForFile &uri, std::string &&contents,
               int64_t version, SendDiagnosticsFnRef sendDiagnosticsFn,
               LLCL::Runtime &runtime);
  MojoDocument(const MojoDocument &) = delete;
  MojoDocument &operator=(const MojoDocument &) = delete;

  /// Initialize the document based on the current set of contents.
  void initialize(const lsp::URIForFile &uri);

  /// Return the contents of this document.
  StringRef getContents() const { return contents; }

  /// Return the version of this document.
  int64_t getVersion() const { return version; }

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

  void
  onCodeCompletion(const lsp::Position &completePos,
                   OnResultFn<mlir::lsp::CompletionList> onCompletionFn) const;

  void onDefinition(
      const lsp::Position &pos,
      OnResultFn<std::optional<mlir::lsp::Location>> onDefinitionFn) const;

  void onHover(const lsp::Position &pos,
               OnResultFn<std::optional<mlir::lsp::Hover>> onHoverFn) const;

private:
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

  //===--------------------------------------------------------------------===//
  // Parsed Fields

  /// Allow access to the parser fields.
  friend LSPParserListener;

  /// An ordered set of fixits for diagnostics emitted for the current version
  /// of the file.
  std::map<std::pair<lsp::Range, std::string>, std::vector<lsp::CodeAction>>
      fixits;

  /// The overall parser context.
  std::unique_ptr<Context> context;
};
} // namespace

MojoDocument::MojoDocument(const lsp::URIForFile &uri, std::string &&contents,
                           int64_t version,
                           SendDiagnosticsFnRef sendDiagnosticsFn,
                           LLCL::Runtime &runtime)
    : uri(uri), contents(std::move(contents)), version(version),
      sendDiagnosticsFn(sendDiagnosticsFn), runtime(runtime) {}

void MojoDocument::initialize(const lsp::URIForFile &uri) {
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
  lsp::PublishDiagnosticsParams diagParams(uri, version);
  for (ArrayRef<llvm::SMDiagnostic> diags : handlerCtx.smDiagnostics) {
    if (auto lspDiag =
            buildLspDiagnosticFromSMDiagnostic(context->sourceMgr, diags, uri))
      diagParams.diagnostics.push_back(*lspDiag);
  }
  sendDiagnosticsFn(diagParams);
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
  onActions(getCodeActionsSync(pos, context));
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
    OnResultFn<mlir::lsp::CompletionList> onCompletionFn) const {
  onCompletionFn(onCodeCompletionSync(completePos));
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
    OnResultFn<std::optional<mlir::lsp::Location>> onDefinitionFn) const {
  onDefinitionFn(onDefinitionSync(pos));
}

std::optional<lsp::Location>
MojoDocument::onDefinitionSync(const lsp::Position &pos) const {
  if (Symbol *symbol = context->symbolIndex.getSymbolAt(pos))
    if (auto symbolUri =
            getURIFromLoc(context->sourceMgr, symbol->identifierLoc, uri))
      return lsp::Location(*symbolUri, getRangeForText(context->sourceMgr,
                                                       symbol->identifierLoc,
                                                       symbol->identifier));

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// MojoDocument: Hover
//===----------------------------------------------------------------------===//

void MojoDocument::onHover(
    const lsp::Position &pos,
    OnResultFn<std::optional<mlir::lsp::Hover>> onHoverFn) const {
  onHoverFn(onHoverSync(pos));
}

std::optional<lsp::Hover>
MojoDocument::onHoverSync(const lsp::Position &pos) const {
  Symbol *symbol = context->symbolIndex.getSymbolAt(pos);
  if (!symbol)
    return std::nullopt;

  lsp::Hover hover(symbol->getIdentifierRange(context->sourceMgr));
  hover.contents.kind = mlir::lsp::MarkupKind::Markdown;
  hover.contents.value = symbol->getMarkdownDeclaration();
  return hover;
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
  Impl(SendDiagnosticsFn sendDiagnosticsFn)
      : sendDiagnosticsFn(std::move(sendDiagnosticsFn)) {}

  /// Retrieve the document that matches completely the given filename. Return
  /// `nullptr` if no document is found.
  MojoDocument *findDocument(StringRef filename) {
    auto it = files.find(filename);
    if (it == files.end())
      return nullptr;
    return it->second.get();
  }

  /// The function used to send diagnostics to the client.
  SendDiagnosticsFn sendDiagnosticsFn;

  /// The runtime used when processing files.
  LLCL::Runtime runtime{LLCL::createMallocAllocator(),
                        LLCL::createThreadPoolWorkQueue()};

  /// The files held by the server, mapped by their URI file name.
  llvm::StringMap<std::unique_ptr<MojoDocument>> files;
};

//===----------------------------------------------------------------------===//
// MojoServer
//===----------------------------------------------------------------------===//

MojoServer::MojoServer(SendDiagnosticsFn sendDiagnosticsFn)
    : impl(std::make_unique<Impl>(std::move(sendDiagnosticsFn))) {}
MojoServer::~MojoServer() = default;

void MojoServer::addDocument(const lsp::URIForFile &uri, std::string &&contents,
                             int64_t version) {
  auto [it, _] = impl->files.try_emplace(uri.file(), nullptr);
  it->second =
      std::make_unique<MojoDocument>(uri, std::move(contents), version,
                                     impl->sendDiagnosticsFn, impl->runtime);
  it->second->initialize(uri);
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
  impl->files.erase(it);
}

void MojoServer::getCodeActions(
    const lsp::URIForFile &uri, const lsp::Range &pos,
    const lsp::CodeActionContext &context,
    OnResultFn<std::vector<mlir::lsp::CodeAction>> onActionsFn) {
  if (MojoDocument *doc = impl->findDocument(uri.file()))
    doc->getCodeActions(pos, context, std::move(onActionsFn));
}

void MojoServer::onCodeCompletion(
    const lsp::URIForFile &uri, const lsp::Position &completePos,
    OnResultFn<mlir::lsp::CompletionList> onCompletionFn) {
  if (MojoDocument *doc = impl->findDocument(uri.file()))
    doc->onCodeCompletion(completePos, std::move(onCompletionFn));
}

void MojoServer::onDefinition(
    const lsp::URIForFile &uri, const lsp::Position &pos,
    OnResultFn<std::optional<mlir::lsp::Location>> onDefinitionFn) {
  if (MojoDocument *doc = impl->findDocument(uri.file()))
    doc->onDefinition(pos, std::move(onDefinitionFn));
}

void MojoServer::onHover(
    const lsp::URIForFile &uri, const lsp::Position &pos,
    OnResultFn<std::optional<mlir::lsp::Hover>> onHoverFn) {
  if (MojoDocument *doc = impl->findDocument(uri.file()))
    doc->onHover(pos, std::move(onHoverFn));
}
