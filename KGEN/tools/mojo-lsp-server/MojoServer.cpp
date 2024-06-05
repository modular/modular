//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoServer.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "MojoDocument.h"

#include "../common/lsp-protocol/SemanticTokens.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoTooling/ASTDeclRef.h"
#include "KGEN/MojoTooling/ASTDeclView.h"
#include "KGEN/MojoTooling/CodeComplete.h"
#include "KGEN/MojoTooling/ParserDriver.h"
#include "KGEN/MojoTooling/REPLPythonExprUtils.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/AnyAsyncValueRef.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/Config.h"
#include "Support/Context.h"
#include "Support/Init/Init.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/ReferenceCounted.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Protocol.h"
#include "mlir/Tools/lsp-server-support/SourceMgrUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
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

/// The message used when emitting a diagnostic for missing documentation.
static constexpr StringRef kMissingDocMessage =
    "Unexpected empty documentation string";

//===----------------------------------------------------------------------===//
// Inlay Hints
//===----------------------------------------------------------------------===//

lsp::InlayHint MojoInlayHint::toLspInlayHint(SourceMgr &sourceMgr) const {
  lsp::InlayHint hint(kind, lsp::Position(sourceMgr, loc));
  if (leftIndent)
    hint.label.assign(leftIndent, ' ');
  hint.label += label.str();

  hint.paddingLeft = paddingLeft;
  hint.paddingRight = paddingRight;
  return hint;
}

//===----------------------------------------------------------------------===//
// Symbol
//===----------------------------------------------------------------------===//

namespace {
/// Common representation for any kind of symbol.
struct SymbolRef;

struct Symbol {
  Symbol(MojoASTDeclRef declRef, StringRef identifier, SMLoc identifierLoc)
      : identifier(identifier), declRef(declRef),
        approximateViewKind(declRef.getApproximateViewKind()) {
    // Modules/Packages just point to the direct location, they don't have a
    // name in the source code.
    if (isa<FileModuleOp, PackageOp>(*declRef))
      range = {identifierLoc, identifierLoc};
    else
      range = getRangeForText(identifierLoc, identifier);
  }

  Symbol(const Symbol &) = delete;
  Symbol &operator=(const Symbol &) = delete;

  /// Return a nicely formatted markdown text of the declaration of this symbol.
  std::string getMarkdownDeclaration() const;

  /// Identifier of the symbol as specified in the source code.
  std::string identifier;

  /// API for accessing the internals of this symbol.
  MojoASTDeclRef declRef;

  /// The approximate view kind for the decl of this symbol. This can provide a
  /// rough estimate for what kind of decl this is.
  std::optional<DeclViewKind> approximateViewKind;

  /// The location of the identifier of this symbol.
  SMRange range;

  /// A list of symbolRefs that point to this symbol.
  llvm::SetVector<SymbolRef *> symbolRefs;
};
} // namespace

/// Return if the given view kind should be included in the markdown
/// declaration.
static bool shouldIncludeViewKindInMarkdown(DeclViewKind kind) {
  return kind != DeclViewKind::DK_AliasDeclView &&
         kind != DeclViewKind::DK_StructDeclView;
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
  SymbolRef(ArrayRef<Symbol *> symbols, SMRange range)
      : symbols(symbols), range(range) {}
  SymbolRef(Symbol &symbol, SMRange range) : SymbolRef(&symbol, range) {}

  /// Return a nicely formatted markdown text of this reference.
  std::string getMarkdownDeclaration() const;

  /// The symbols being referenced.
  SmallVector<Symbol *, 1> symbols;
  /// The range in the index's doc where the symbol is being referenced.
  SMRange range;

  /// Remove the existing links from Symbol -> SymbolRef for this SymbolRef.
  void removeSymbolToSymbolRefMapping() {
    for (Symbol *symbol : symbols)
      symbol->symbolRefs.remove(this);
  }

  /// Create the links from Symbol -> SymbolRef for reverse lookups.
  void createSymbolToSymbolRefMapping() {
    for (Symbol *symbol : symbols)
      symbol->symbolRefs.insert(this);
  }
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
  void registerRef(ArrayRef<MojoASTDeclRef> declRefs, SMRange range,
                   StringRef spelling);

  /// Look for the symbols whose declaration or references contain the given
  /// position in the document.
  SymbolRef *getSymbolAt(SMLoc loc) const;

  /// Look for the symbol corresponding to the given decl in the symbol table.
  /// Return nullptr if not found.
  Symbol *findSymbol(MojoASTDeclRef declRef);

  /// Walk all symbols in the index, regardless of range.
  void walkSymbols(function_ref<void(Symbol &)> callback);

  /// Walk the symbol references in the given range, including references that
  /// are only partially overlapped.
  void walkSymbolRefs(SMRange range, function_ref<void(SymbolRef &)> callback);

private:
  /// Store the range corresponding to the reference or the declaration of a
  /// symbol in the main doc.
  /// In addition, this method maintains the mapping from Symbol to SymbolRef,
  /// as affectively here is where we commit changes to the index.
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
  llvm::MapVector<ASTDecl *, std::unique_ptr<Symbol>> symbolTable;
};
} // namespace

Symbol *SymbolIndex::findSymbol(MojoASTDeclRef declRef) {
  if (auto it = symbolTable.find(&*declRef); it != symbolTable.end())
    return it->second.get();
  return nullptr;
}

void SymbolIndex::walkSymbols(function_ref<void(Symbol &)> callback) {
  for (const std::unique_ptr<Symbol> &symbol :
       llvm::make_second_range(symbolTable))
    callback(*symbol);
}

void SymbolIndex::walkSymbolRefs(SMRange range,
                                 function_ref<void(SymbolRef &)> callback) {
  auto startIt = rangeToSymbolRef.find(range.Start);
  auto endIt = rangeToSymbolRef.find(range.End);
  if (!startIt.valid())
    return;
  // Include partial overlaps at the end.
  if (endIt.valid())
    ++endIt;

  for (SymbolRef *ref : llvm::make_range(startIt, endIt))
    callback(*ref);
}

void SymbolIndex::insertRangeInMainDoc(SymbolRef &&symbolRef) {
  SMRange range = symbolRef.range;

  // If an existing mapping is found, overwrite with the new reference. We may
  // resolve more specific references as the parser progresses.
  if (auto it = rangeToSymbolRef.find(range.Start); it.valid()) {
    SymbolRef *existingSymbolRef = it.value();
    if (it.start() == range.Start && it.stop() == range.End &&
        existingSymbolRef->symbols.size() > symbolRef.symbols.size()) {

      existingSymbolRef->removeSymbolToSymbolRefMapping();
      it.value()->symbols = std::move(symbolRef.symbols);
      existingSymbolRef->createSymbolToSymbolRefMapping();
      return;
    }
  }

  // Otherwise, insert a new mapping.
  if (!rangeToSymbolRef.overlaps(range.Start, range.End)) {
    symbolRefs.push_back(std::make_unique<SymbolRef>(std::move(symbolRef)));
    SymbolRef *symbolRef = symbolRefs.back().get();
    rangeToSymbolRef.insert(range.Start, range.End, symbolRef);
    symbolRef->createSymbolToSymbolRefMapping();
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
  if (mainDoc.containsLocation(symbol.range.Start)) {
    // Don't register modules as they don't have a proper location in the file.
    if (!isa<FileModuleOp>(*declRef))
      insertRangeInMainDoc({symbol, symbol.range});
  }
  return &symbol;
}

void SymbolIndex::registerRef(ArrayRef<MojoASTDeclRef> declRefs, SMRange range,
                              StringRef spelling) {
  // We don't index empty spellings nor references in files other than the main
  // one.
  if (spelling.empty() || !mainDoc.containsLocation(range.Start))
    return;

  SmallVector<Symbol *> symbols;
  for (MojoASTDeclRef ref : declRefs) {
    // Capture the symbol if it exists, otherwise try to register it, as it
    // might come from a non-main doc.
    if (Symbol *symbol = findSymbol(ref)) {
      symbols.push_back(symbol);
      continue;
    }

    // Grab the name of the decl.
    std::optional<StringRef> symName = ref.getName();
    if (!symName) {
      // If we can't compute one, handle the edge case where we're dealing with
      // a reference to an argument which could manifest while type checking
      // the signature (so the normal way of computing the name isn't ready
      // yet).
      if (ref.getApproximateViewKind() == DeclViewKind::DK_ArgumentDeclView)
        symName = spelling;
    }

    if (Symbol *symbol = registerSymbol(
            ref, symName, mainDoc.translateParserLoc(ref.getLoc())))
      symbols.push_back(symbol);
  }

  if (!symbols.empty())
    insertRangeInMainDoc({symbols, range});
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

  void addSymbolDecl(ASTDecl *decl, SMLoc loc,
                     std::optional<StringRef> identifier = {});

  bool isInterestedInLoc(SMLoc parserLoc) override {
    // We're only interested in locations in the main file.
    return mainDoc.containsLocation(mainDoc.translateParserLoc(parserLoc));
  }

  void onAliasDecl(ASTDecl *decl, SMLoc identifierLoc) override;

  void onArgumentDecl(ASTDecl *decl, StringRef argName,
                      SMLoc identifierLoc) override;

  void onFunctionDecl(ASTDecl *decl, SMLoc identifierLoc) override;

  void onModuleDecl(ASTDecl *decl, SMLoc identifierLoc) override;

  void onModuleImport(ASTDecl *decl, StringRef spelling, SMLoc loc) override;

  void onParameterDecl(ASTDecl *decl, SMLoc identifierLoc) override;

  void onStructDecl(ASTDecl *decl, SMLoc identifierLoc) override;

  void onStructFieldDecl(ASTDecl *decl, SMLoc identifierLoc) override;

  void onTraitDecl(ASTDecl *decl, SMLoc identifierLoc) override;

  void onVariableDecl(ASTDecl *decl, SMLoc identifierLoc) override;

  void onRef(ArrayRef<ASTDecl *> decls, StringRef spelling,
             llvm::SMRange range) override;

private:
  /// The main doc for which parsing was initiated.
  MojoDocument &mainDoc;
  SymbolIndex &symbolIndex;
};
} // namespace

void LSPParserListener::addSymbolDecl(ASTDecl *decl, SMLoc loc,
                                      std::optional<StringRef> identifier) {
  MojoASTDeclRef declRef(decl);
  if (!identifier)
    identifier = declRef.getName();
  symbolIndex.registerSymbol(declRef, identifier,
                             mainDoc.translateParserLoc(loc));
}

void LSPParserListener::onAliasDecl(ASTDecl *decl, SMLoc identifierLoc) {
  addSymbolDecl(decl, identifierLoc);
}

void LSPParserListener::onArgumentDecl(ASTDecl *decl, StringRef argName,
                                       SMLoc identifierLoc) {
  addSymbolDecl(decl, identifierLoc, argName);
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
  loc = mainDoc.translateParserLoc(loc);
  symbolIndex.registerRef(MojoASTDeclRef(decl), getRangeForText(loc, spelling),
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

void LSPParserListener::onTraitDecl(ASTDecl *decl, SMLoc identifierLoc) {
  addSymbolDecl(decl, identifierLoc);
}

void LSPParserListener::onRef(ArrayRef<ASTDecl *> decls, StringRef spelling,
                              llvm::SMRange range) {
  symbolIndex.registerRef(
      llvm::map_to_vector(decls,
                          [](ASTDecl *decl) -> MojoASTDeclRef { return decl; }),
      mainDoc.translateParserLoc(range), spelling);
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
    // Ignore non-user variables.
    if (MojoParserContext::isHiddenPersistentVariable(name))
      return false;

    auto [it, inserted] =
        nameToVariable.insert({name, newPersistentVariables.size()});
    if (inserted)
      newPersistentVariables.emplace_back(name, type);
    else
      newPersistentVariables[it->second].second = type;
    return true;
  }

private:
  llvm::StringMap<unsigned> nameToVariable;
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
        parserConfig(&mlirContext, compilationOptions), symbolIndex(mainDoc),
        parserListener(mainDoc, symbolIndex) {
    parserConfig.parserListener = &parserListener;

    DialectRegistry registry;
    registerAllKGENDialects(registry);
    parserConfig.context->appendDialectRegistry(registry);

    parserContext =
        std::make_unique<MojoParserContext>(mainDoc.sourceMgr, parserConfig);
  }

  KGEN::CompilationOptions compilationOptions;
  MLIRContext mlirContext{MLIRContext::Threading::DISABLED};
  ParserConfig parserConfig;
  SymbolIndex symbolIndex;
  LSPParserListener parserListener;
  std::unique_ptr<MojoParserContext> parserContext;
};

//===----------------------------------------------------------------------===//
// MojoDocument
//===----------------------------------------------------------------------===//

static bool
shouldSkipArgument(DenseMap<const ASTDecl *, std::unique_ptr<DeclView>> &views,
                   MojoASTDeclRef decl) {
  assert(decl.getParentDecl() && "Arguments must always have a parent decl");
  MojoASTDeclRef grandparent = decl.getParentDecl().getParentDecl();

  // If there's not a parent of this decl, it's a normal function, which does
  // not have a self argument.
  if (!grandparent)
    return false;

  std::optional<DeclViewKind> grandparentKind =
      grandparent.getApproximateViewKind();

  // All arguments to trait methods/functions are, by definition, unused.
  if (grandparentKind == DeclViewKind::DK_TraitDeclView)
    return true;

  // We want to ignore an unused first argument for struct methods, as this
  // may be the `self` variable.
  if (grandparentKind == DeclViewKind::DK_StructDeclView) {
    MojoASTDeclRef parent = decl.getParentDecl();

    DeclView *parentView;
    if (auto it = views.find(&*parent); it != views.end()) {
      parentView = it->getSecond().get();
    } else {
      std::unique_ptr<DeclView> view = parent.getView();
      parentView = view.get();
      views[&*parent] = std::move(view);
    }

    if (const auto *parentFnView = dyn_cast<FunctionDeclView>(parentView)) {
      // The first argument of a non-static method is the `self` variable,
      // but it may not be named that. Since argument names must be unique,
      // if the first argument's name is the symbol in question, then we're
      // looking at the `self` variable.
      if (parentFnView->isMethod() &&
          parentFnView->getArguments().front().getName() == decl.getName()) {
        return true;
      }
    }
  }

  return false;
}

static std::vector<lsp::Diagnostic> checkUnusedVariables(SymbolIndex &index,
                                                         MojoDocument &doc) {
  std::vector<lsp::Diagnostic> diags;
  DenseMap<const ASTDecl *, std::unique_ptr<DeclView>> parentViews;

  index.walkSymbols([&](const Symbol &symbol) {
    if (!doc.containsLocation(symbol.declRef->getLoc()))
      return;

    // This indicates that the symbol is referenced outside of its decl, so it's
    // being used.
    if (symbol.symbolRefs.size() > 1)
      return;

    // Ignore variables beginning with _. This is the traditional syntax for an
    // intentionally unused variable and allows users to silence the diagnostic
    // if necessary.
    if (StringRef(symbol.identifier).starts_with("_"))
      return;

    std::optional<DeclViewKind> declKind =
        symbol.declRef.getApproximateViewKind();

    if (declKind != DeclViewKind::DK_VariableDeclView &&
        declKind != DeclViewKind::DK_ArgumentDeclView)
      return;

    if (declKind == DeclViewKind::DK_ArgumentDeclView &&
        shouldSkipArgument(parentViews, symbol.declRef))
      return;

    StringRef kind = DeclView::getKindAsString(*declKind);

    lsp::Diagnostic lspDiag;
    lspDiag.source = "mojo";
    lspDiag.severity = lsp::DiagnosticSeverity::Warning;
    lspDiag.message =
        llvm::formatv("unused {0} '{1}'", kind, symbol.identifier).str();
    lspDiag.range = lsp::Range(doc.getSourceMgr(), symbol.range);
    lspDiag.tags.push_back(lsp::DiagnosticTag::Unnecessary);

    diags.emplace_back(std::move(lspDiag));
  });

  return diags;
}

MojoDocument::MojoDocument(Kind kind, ArrayRef<lsp::URIForFile> uris,
                           int64_t version,
                           SendDiagnosticsFnRef sendDiagnosticsFn,
                           LLCL::Runtime &runtime, LLCL::AnyAsyncValueRef chain,
                           ArrayRef<std::string> includeDirs)
    : kind(kind), uris(uris), version(version),
      sendDiagnosticsFn(sendDiagnosticsFn), runtime(runtime),
      isDocumentParsed(AsyncValueRef<Chain>::allocate(runtime)),
      isQuiescent(runtime.getReadyChain().copy()) {
  // Add the parent directory of the main uri as an available include directory.
  std::string parentDir =
      std::filesystem::path(uris[0].file().str()).parent_path().string();

  std::vector<std::string> allIncludeDirs{parentDir};
  llvm::append_range(allIncludeDirs, includeDirs);
  getSourceMgr().setIncludeDirs(allIncludeDirs);

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
    DiagHandlerContext(MojoDocument &doc) : doc(doc) {}

    /// A reference to the document.
    MojoDocument &doc;
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
    // Remember errors found during parsing.
    if (diag.getKind() == llvm::SourceMgr::DK_Error)
      handlerCtx.doc.hasParserErrors = true;

    handlerCtx.smDiagnostics.push_back({diag});
  };
  DiagHandlerContext handlerCtx(*this);
  sourceMgr.setDiagHandler(handlerFn, &handlerCtx);
  context = std::make_unique<Context>(*this);

  parseDocumentImpl();

  // If we've already been invalidated, bail out early.
  if (isInvalidated)
    return markDocumentParsed();

  // Process the collected diagnostics.
  llvm::StringMap<std::optional<lsp::PublishDiagnosticsParams>> fileToDiags;
  for (auto &uri : uris)
    fileToDiags[uri.file()].emplace(uri, version);

  // Find unused variables in this document.
  std::vector<lsp::Diagnostic> unusedDiags =
      checkUnusedVariables(context->symbolIndex, *this);
  std::vector<lsp::Diagnostic> &fileDiags =
      fileToDiags[uris.front().file()]->diagnostics;
  fileDiags.insert(fileDiags.end(), unusedDiags.begin(), unusedDiags.end());

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

  // Process the fixit actions, using a custom title for certain actions.
  auto missingDocFixitsIt = fixits.find(kMissingDocMessage);
  if (missingDocFixitsIt != fixits.end()) {
    for (auto &[range, actions] : missingDocFixitsIt->second)
      actions[0].title = "Generate documentation";
  }

  // Sort any inlay hints computed during parsing.
  llvm::stable_sort(inlayHints);

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

AsyncValueRef<Chain> MojoDocument::newTaskChain() {
  std::lock_guard<std::mutex> lock(isDocumentParsedMutex);
  // Join these two tasks into one chain.
  AsyncValueRef<Chain> n =
      AsyncValueRef<Chain>::allocate(isQuiescent.getRuntime());
  AsyncValueRef<Chain> rval =
      AsyncValueRef<Chain>::allocate(isQuiescent.getRuntime());
  llvm::SmallVector<AnyAsyncValueRef> refs;
  refs.emplace_back(std::move(isQuiescent));
  refs.emplace_back(rval.copy());
  LLCL::andThenSyncCopying(
      llvm::ArrayRef<AnyAsyncValueRef>(refs),
      [n = n.copy()](llvm::ArrayRef<AnyAsyncValueRef> elems) mutable {
        std::move(n).emplace();
      });
  isQuiescent = std::move(n);
  return rval;
}

void MojoDocument::checkModuleSemantics(MojoASTDeclRef decl) {
  // Don't check the semantics of the module if there were parser errors.
  if (hasParserErrors || !decl || !decl.getIfOperation())
    return;

  // Clone the module this decl is in so that we don't mess with the AST, as
  // this is used for other LSP queries.
  OwningOpRef<ModuleOp> tempModuleOp = cloneDeclModuleForCompilation(*decl);

  // Build a wrapper diagnostic handler for the source manager to capture
  // diagnostics emitted when processing the module.
  mlir::SourceMgrDiagnosticHandler sourceMgrDiagHandler(
      sourceMgr, tempModuleOp->getContext());

  // Run the high level verification pipeline.
  KGEN::KGENCompiler kgenCompiler(*tempModuleOp->getContext(),
                                  getCompilationOptions());

  if (failed(kgenCompiler.runCheckLITPipeline(*tempModuleOp))) {
    lsp::Logger::debug("The 'check' pipeline failed to run on the module {0}",
                       decl.getName().value_or("<unnamed>"));
  }
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
  if (lspDiag.message.empty())
    lspDiag.message = "Mojo parser error";

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

    fixits[lspDiag.message].emplace(lspDiag.range, std::move(diagFixits));
  }

  return lspDiag;
}

//===----------------------------------------------------------------------===//
// MojoDocument: Code Action
//===----------------------------------------------------------------------===//

void MojoDocument::getCodeActions(
    const lsp::URIForFile &uri, const lsp::Range &pos,
    const lsp::CodeActionContext &context,
    LSPResponder<std::vector<lsp::CodeAction>> responder) {
  startTaskAfterParsing([uri, pos, context, responder = std::move(responder)](
                            MojoDocument &doc) mutable {
    if (doc.isInvalidated)
      return responder.replyOutdatedRequest();
    responder.reply(
        doc.getCodeActionsSync(doc.getLocFromPos(uri, pos), context));
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
    auto fixitIt = fixits.find(diag.message);
    if (fixitIt == fixits.end())
      continue;
    auto it = fixitIt->second.find(diag.range);
    if (it == fixitIt->second.end())
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
    const lsp::URIForFile &uri, const lsp::Position &completePos,
    LSPResponder<lsp::CompletionList> responder) {
  startTaskAfterParsing([uri, completePos, responder = std::move(responder)](
                            MojoDocument &doc) mutable {
    if (doc.isInvalidated)
      return responder.replyOutdatedRequest();
    SMLoc completeLoc = doc.getLocFromPos(uri, completePos);
    if (!completeLoc.isValid())
      return responder.replyInvalidRequest();
    responder.reply(doc.onCodeCompletionSync(completeLoc));
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
    case KGEN::Mojo::CodeCompletionResult::kVariable:
      item.kind = lsp::CompletionItemKind::Variable;
      break;
    case KGEN::Mojo::CodeCompletionResult::kTrait:
      item.kind = lsp::CompletionItemKind::Interface;
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
    const lsp::URIForFile &uri, const lsp::Position &pos,
    LSPResponder<std::vector<lsp::Location>> responder) {
  startTaskAfterParsing(
      [uri, pos, responder = std::move(responder)](MojoDocument &doc) mutable {
        if (doc.isInvalidated)
          return responder.replyOutdatedRequest();
        SMLoc loc = doc.getLocFromPos(uri, pos);
        if (!loc.isValid())
          return responder.replyInvalidRequest();
        responder.reply(doc.onDefinitionSync(loc));
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
// MojoDocument: Document Symbols
//===----------------------------------------------------------------------===//

void MojoDocument::getDocumentSymbols(
    MojoASTDeclRef decl, std::vector<lsp::DocumentSymbol> &symbols) {
  const llvm::MemoryBuffer *declBuffer = sourceMgr.getMemoryBuffer(
      getSourceMgr().FindBufferContainingLoc(decl.getLoc()));
  StringRef declBufferRef = declBuffer->getBuffer();
  getDocumentSymbols(decl, symbols, [declBufferRef](MojoASTDeclRef decl) {
    // Imported decls may be in a different buffer, only consider decls in the
    // same buffer as the main decl.
    const char *declLoc = decl.getLoc().getPointer();
    return declBufferRef.begin() <= declLoc && declLoc <= declBufferRef.end();
  });
}

void MojoDocument::getDocumentSymbols(
    MojoASTDeclRef decl, std::vector<lsp::DocumentSymbol> &symbols,
    function_ref<bool(MojoASTDeclRef)> shouldIncludeDecl) {
  std::vector<lsp::DocumentSymbol> *nestedSymbols = &symbols;

  // Utility functor to add a new document symbol.
  auto addSymbol = [&](const Twine &name, lsp::SymbolKind kind,
                       lsp::Range range, std::string detail = "") {
    auto &docSym = symbols.emplace_back(name, kind, range, range);
    docSym.detail = std::move(detail);
    nestedSymbols = &docSym.children;
  };

  // Check for symbol information for this decl.
  auto *symbol = context->symbolIndex.findSymbol(decl);
  if (symbol && symbol->range.isValid()) {
    if (std::unique_ptr<DeclView> declView = decl.getView()) {
      lsp::Range range(getSourceMgr(), symbol->range);

      TypeSwitch<DeclView *>(declView.get())
          .Case([&](AliasDeclView *alias) {
            // We only consider global aliases here, we don't want to show every
            // conceivable decl.
            if (!alias->isGlobal())
              return;

            addSymbol(alias->getName(), lsp::SymbolKind::Property, range,
                      alias->getValue().str());
          })
          .Case([&](FunctionDeclView *fn) {
            addSymbol(fn->getName(), lsp::SymbolKind::Function, range,
                      fn->getSignature());
          })
          .Case([&](StructDeclView *structDecl) {
            addSymbol(structDecl->getName(), lsp::SymbolKind::Struct, range);
          })
          .Case([&](StructFieldDeclView *field) {
            addSymbol(field->getName(), lsp::SymbolKind::Field, range,
                      field->getType().str());
          })
          .Case([&](TraitDeclView *traitDecl) {
            addSymbol(traitDecl->getName(), lsp::SymbolKind::Interface, range);
          })
          .Case([&](VariableDeclView *var) {
            // We only consider global variables here, we don't want to show
            // every conceivable decl.
            if (!var->isGlobal())
              return;

            addSymbol(var->getName(), lsp::SymbolKind::Variable, range,
                      var->getType().str());
          });
    }
  }

  // Traverse the child decls.
  for (const auto &childIt : decl.getChildren()) {
    for (MojoASTDeclRef child : childIt.getDecls())
      if (shouldIncludeDecl(child))
        getDocumentSymbols(child, *nestedSymbols);
  }
}

void MojoDocument::onDocumentSymbol(
    const lsp::URIForFile &uri,
    LSPResponder<std::vector<lsp::DocumentSymbol>> responder) {
  startTaskAfterParsing(
      [uri, responder = std::move(responder)](MojoDocument &doc) mutable {
        if (doc.isInvalidated)
          return responder.replyOutdatedRequest();
        responder.reply(doc.onDocumentSymbolSync(uri));
      });
}

//===----------------------------------------------------------------------===//
// MojoDocument: Folding Ranges
//===----------------------------------------------------------------------===//

void MojoDocument::onFoldingRange(
    const lsp::URIForFile &uri,
    LSPResponder<std::vector<lsp::FoldingRange>> responder) {
  startTaskAfterParsing(
      [uri, responder = std::move(responder)](MojoDocument &doc) mutable {
        if (doc.isInvalidated)
          return responder.replyOutdatedRequest();
        responder.reply(doc.onFoldingRangeSync(uri));
      });
}

//===----------------------------------------------------------------------===//
// MojoDocument: Document String Code Blocks
//===----------------------------------------------------------------------===//

void MojoDocument::processDocStrings(
    MojoDocStrings &docStrings, MojoASTDeclRef decl, unsigned bufferId,
    function_ref<bool(MojoASTDeclRef)> shouldIncludeDecl,
    MojoASTDeclRef curReplDecl) {
  docStrings.addDocString(*this, decl, curReplDecl, bufferId);

  // Traverse the child decls.
  for (const auto &childIt : decl.getChildren()) {
    for (MojoASTDeclRef child : childIt.getDecls())
      if (shouldIncludeDecl(child))
        processDocStrings(docStrings, child, bufferId, shouldIncludeDecl,
                          curReplDecl);
  }
}

void MojoDocument::processDocStrings(MojoDocStrings &docStrings,
                                     MojoASTDeclRef decl,
                                     MojoASTDeclRef curReplDecl) {
  unsigned bufferId = sourceMgr.FindBufferContainingLoc(decl.getLoc());
  StringRef declBufferRef = sourceMgr.getMemoryBuffer(bufferId)->getBuffer();
  auto processFn = [declBufferRef](MojoASTDeclRef decl) {
    // Imported decls may be in a different buffer, only consider decls in the
    // same buffer as the main decl.
    const char *declLoc = decl.getLoc().getPointer();
    return declBufferRef.begin() <= declLoc && declLoc <= declBufferRef.end();
  };
  processDocStrings(docStrings, decl, bufferId, processFn, curReplDecl);
}

//===----------------------------------------------------------------------===//
// MojoDocument: Hover
//===----------------------------------------------------------------------===//

void MojoDocument::onHover(
    const lsp::URIForFile &uri, const lsp::Position &pos,
    LSPResponder<std::optional<mlir::lsp::Hover>> responder) {
  startTaskAfterParsing(
      [uri, pos, responder = std::move(responder)](MojoDocument &doc) mutable {
        if (doc.isInvalidated)
          return responder.replyOutdatedRequest();
        SMLoc loc = doc.getLocFromPos(uri, pos);
        if (!loc.isValid())
          return responder.replyInvalidRequest();
        responder.reply(doc.onHoverSync(loc));
      });
}

std::optional<lsp::Hover> MojoDocument::onHoverSync(SMLoc loc) {
  if (auto symbolRef = context->symbolIndex.getSymbolAt(loc)) {
    lsp::Hover hover(lsp::Range(getSourceMgr(), symbolRef->range));
    hover.contents.kind = lsp::MarkupKind::Markdown;
    hover.contents.value = symbolRef->getMarkdownDeclaration();
    return hover;
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// MojoDocument: Inlay Hints
//===----------------------------------------------------------------------===//

void MojoDocument::onInlayHint(
    const lsp::URIForFile &uri, const lsp::Range &range,
    LSPResponder<std::vector<lsp::InlayHint>> responder) {
  startTaskAfterParsing([uri, range, responder = std::move(responder)](
                            MojoDocument &doc) mutable {
    if (doc.isInvalidated)
      return responder.replyOutdatedRequest();
    SMRange smRange = doc.getLocFromPos(uri, range);
    if (!smRange.isValid())
      return responder.replyInvalidRequest();
    responder.reply(doc.onInlayHintSync(smRange));
  });
}

std::vector<lsp::InlayHint> MojoDocument::onInlayHintSync(SMRange range) {
  std::vector<lsp::InlayHint> hints;

  // Grab the set of inlay hints contained within the given range.
  auto hintIt = llvm::partition_point(
      inlayHints, [&](const auto &hint) { return hint.loc < range.Start; });
  auto hintEndIt = llvm::partition_point(
      inlayHints, [&](const auto &hint) { return hint.loc < range.End; });
  for (const MojoInlayHint &hint : llvm::make_range(hintIt, hintEndIt))
    hints.push_back(hint.toLspInlayHint(sourceMgr));

  return hints;
}

//===----------------------------------------------------------------------===//
// MojoDocument: References
//===----------------------------------------------------------------------===//

void MojoDocument::onReferences(
    const lsp::URIForFile &uri, const lsp::Position &position,
    bool includeDeclaration,
    LSPResponder<std::vector<lsp::Location>> responder) {
  startTaskAfterParsing(
      [uri, position, includeDeclaration,
       responder = std::move(responder)](MojoDocument &doc) mutable {
        if (doc.isInvalidated)
          return responder.replyOutdatedRequest();
        SMLoc smLoc = doc.getLocFromPos(uri, position);
        if (!smLoc.isValid())
          return responder.replyInvalidRequest();
        responder.reply(doc.onReferencesSync(smLoc, includeDeclaration));
      });
}

std::vector<lsp::Location>
MojoDocument::onReferencesSync(SMLoc smLoc, bool includeDeclaration) {
  SymbolRef *symbolRef = context->symbolIndex.getSymbolAt(smLoc);
  if (!symbolRef)
    return {};

  std::vector<lsp::Location> locations;
  for (const Symbol *symbol : symbolRef->symbols) {
    for (SymbolRef *symbolRef : symbol->symbolRefs) {
      if (!includeDeclaration && symbol->range.Start == symbolRef->range.Start)
        continue;
      if (auto uri = getURIFromLoc(symbolRef->range.Start))
        locations.emplace_back(*uri,
                               lsp::Range(getSourceMgr(), symbolRef->range));
    }
  }
  llvm::sort(locations);
  return locations;
}

//===----------------------------------------------------------------------===//
// MojoDocument: Signature Help
//===----------------------------------------------------------------------===//

void MojoDocument::onSemanticTokens(
    const lsp::URIForFile &uri,
    OnSemanticTokensResultFn<std::optional<std::vector<SemanticToken>>>
        onSemanticTokens) {
  startTaskAfterParsing([uri, onSemanticTokens = std::move(onSemanticTokens)](
                            MojoDocument &doc) mutable {
    if (doc.isInvalidated)
      return onSemanticTokens({}, /*outdated=*/true, /*invalid=*/false);
    // Get a document range for the given uri.
    SMRange range = doc.getFullRangeForURI(uri);
    if (!range.isValid())
      return onSemanticTokens({}, /*outdated=*/false, /*invalid=*/true);
    onSemanticTokens(doc.onSemanticTokensSync(range), /*outdated=*/false,
                     /*invalid=*/false);
  });
}

/// Return if the given decl corresponds to the `self` argument
/// of a method.
static bool isSelfArgument(MojoASTDeclRef decl) {
  if (decl.getName() == "self") {
    auto parentFunc = dyn_cast<KGEN::LIT::FuncOp>(decl->getParentDecl());
    return parentFunc &&
           isa<KGEN::LIT::StructDeclOp, KGEN::LIT::TraitDeclOp>(
               parentFunc->getParentOp()) &&
           !parentFunc.getIsStatic();
  }
  return false;
}

/// Return a semantic token kind for the given ast decl.
static SemanticTokenKind
getSemanticTokenKind(MojoASTDeclRef symDecl,
                     std::optional<DeclViewKind> declKind) {
  // If we can't decipher the kind, it's nearly always a variable.
  if (!declKind)
    return SemanticTokenKind::kVariable;

  switch (*declKind) {
  case DeclViewKind::DK_AliasDeclView: {
    auto aliasOp = cast<KGEN::LIT::AliasDeclOp>(symDecl->getIfOperation());
    Attribute aliasValue = aliasOp.getValue();

    // Try to decipher a token kind from the alias value.
    if (isa<ModuleAttr>(aliasValue))
      return SemanticTokenKind::kModule;
    if (isa<KGEN::TypeConstantAttr>(aliasValue))
      return SemanticTokenKind::kType;
    return SemanticTokenKind::kVariable;
  }
  case DeclViewKind::DK_ArgumentDeclView:
    if (isSelfArgument(symDecl))
      return SemanticTokenKind::kSpecialVariable;
    return SemanticTokenKind::kVariable;
  case DeclViewKind::DK_FunctionDeclView:
    return SemanticTokenKind::kFunction;
  case DeclViewKind::DK_ModuleDeclView:
  case DeclViewKind::DK_PackageDeclView:
    return SemanticTokenKind::kModule;
  case DeclViewKind::DK_ParameterDeclView:
    return SemanticTokenKind::kParameter;
  case DeclViewKind::DK_StructDeclView:
    return SemanticTokenKind::kClass;
  case DeclViewKind::DK_StructFieldDeclView:
    return SemanticTokenKind::kField;
  case DeclViewKind::DK_TraitDeclView:
    return SemanticTokenKind::kTrait;
  case DeclViewKind::DK_VariableDeclView:
    return SemanticTokenKind::kVariable;
  }
  llvm_unreachable("invalid decl kind");
}

std::optional<std::vector<SemanticToken>>
MojoDocument::onSemanticTokensSync(SMRange range) {
  if (!context)
    return std::nullopt;

  // Compute the set of semantic tokens in the document.
  std::vector<SemanticToken> tokens;

  // Compute tokens for known symbol references.
  context->symbolIndex.walkSymbolRefs(range, [&](SymbolRef &ref) {
    const Symbol *symbol = ref.symbols.front();
    tokens.emplace_back(
        getSemanticTokenKind(symbol->declRef, symbol->approximateViewKind),
        lsp::Range(sourceMgr, ref.range));
  });
  llvm::sort(tokens);

  return tokens;
}

//===----------------------------------------------------------------------===//
// MojoDocument: Signature Help
//===----------------------------------------------------------------------===//

void MojoDocument::onSignatureHelp(
    const lsp::URIForFile &uri, const lsp::Position &pos,
    LSPResponder<lsp::SignatureHelp2> responder) {
  startTaskAfterParsing(
      [uri, pos, responder = std::move(responder)](MojoDocument &doc) mutable {
        if (doc.isInvalidated)
          return responder.replyOutdatedRequest();
        SMLoc loc = doc.getLocFromPos(uri, pos);
        if (!loc.isValid())
          return responder.replyInvalidRequest();

        lsp::SignatureHelp help = doc.onSignatureHelpSync(loc);
        lsp::SignatureHelp2 help2;
        help2.activeParameter = help.activeParameter;
        help2.activeSignature = help.activeSignature;
        for (auto &sig : help.signatures) {
          lsp::SignatureInformation2 sig2;
          sig2.label = sig.label;
          sig2.documentation =
              lsp::MarkupContent{lsp::MarkupKind::Markdown, sig.documentation};
          sig2.parameters = std::move(sig.parameters);
          help2.signatures.push_back(sig2);
        }

        responder.reply(std::move(help2));
      });
}

lsp::SignatureHelp MojoDocument::onSignatureHelpSync(SMLoc loc) {
  if (!context)
    return lsp::SignatureHelp();

  // Call the parser to get the signature help.
  std::optional<KGEN::Mojo::SignatureHelpResult> result =
      onSignatureHelpSyncImpl(loc);
  if (!result)
    return lsp::SignatureHelp();

  // Map the Mojo results to LSP results.
  lsp::SignatureHelp help;
  help.activeSignature = result->activeSignature;
  help.activeParameter = result->activeParameter;
  for (const auto &signature : result->signatures) {
    lsp::SignatureInformation info;
    info.label = signature.label;
    info.documentation = signature.documentation;
    for (const auto &param : signature.parameters)
      info.parameters.push_back({"", param.labelOffset, param.documentation});
    help.signatures.push_back(info);
  }
  return help;
}

//===----------------------------------------------------------------------===//
// MojoDocument: Renaming
//===----------------------------------------------------------------------===//

void MojoDocument::onRename(const lsp::URIForFile &uri,
                            const mlir::lsp::Position &pos, StringRef newName,
                            LSPResponder<lsp::WorkspaceEdit> responder) {
  // Convert newName to a string now, because the string ref may be invalidated
  // by the time we process this request.
  startTaskAfterParsing([uri, pos, newName = newName.str(),
                         responder =
                             std::move(responder)](MojoDocument &doc) mutable {
    if (doc.isInvalidated)
      return responder.replyOutdatedRequest();

    SMLoc loc = doc.getLocFromPos(uri, pos);
    if (!loc.isValid())
      return responder.replyInvalidRequest();

    ErrorOr<std::vector<lsp::TextEdit>> edits = doc.onRenameSync(loc, newName);

    if (auto err = edits.getError())
      return responder.replyError(
          lsp::LSPError(err, lsp::ErrorCode::InvalidRequest));

    std::map<std::string, std::vector<lsp::TextEdit>> workspaceEdits;
    workspaceEdits.insert({uri.file().str(), std::move(edits.get())});

    return responder.reply(lsp::WorkspaceEdit{std::move(workspaceEdits)});
  });
}

static bool isLocalVariable(const Symbol &symbol) {
  if (symbol.approximateViewKind != DeclViewKind::DK_VariableDeclView)
    return false;

  std::unique_ptr<DeclView> declView = symbol.declRef.getView();
  if (const auto *varDeclView = dyn_cast<VariableDeclView>(declView.get()))
    return !varDeclView->isGlobal();

  return false;
}

ErrorOr<std::vector<mlir::lsp::TextEdit>>
MojoDocument::onRenameSync(SMLoc loc, const std::string &newName) {
  SymbolRef *symbolRef = context->symbolIndex.getSymbolAt(loc);
  if (!symbolRef)
    return Error("no identified symbol at this position");

  const char *errorMessage = "renaming is only available for local variables";

  // Variables have only one symbol for a given SymbolRef. Overloaded function
  // sets will reference more than one symbol, but we aren't going to rename
  // those.
  if (symbolRef->symbols.size() > 1)
    return Error(errorMessage);

  Symbol &symbol = *symbolRef->symbols.front();

  if (!isLocalVariable(symbol))
    return Error(errorMessage);

  std::vector<lsp::TextEdit> edits;
  edits.reserve(symbol.symbolRefs.size());
  for (SymbolRef *symbolRef : symbol.symbolRefs) {
    edits.push_back(
        lsp::TextEdit{lsp::Range(getSourceMgr(), symbolRef->range), newName});
  }

  llvm::stable_sort(edits, [](const lsp::TextEdit &a, const lsp::TextEdit &b) {
    return a.range < b.range;
  });

  return std::move(edits);
}

//===----------------------------------------------------------------------===//
// MojoDocStrings
//===----------------------------------------------------------------------===//

void MojoDocStrings::addDocString(MojoDocument &mainDoc, MojoASTDeclRef decl,
                                  MojoASTDeclRef curReplDecl,
                                  unsigned bufferId) {
  // Check if the decl has a doc string with a decipherable location.
  std::optional<KGEN::LIT::DocString> docString = decl->getParsedDocString();
  if (!docString)
    return;
  FileLineColLoc docLocAttr = docString->getLoc();
  if (!docLocAttr)
    return;
  SourceMgr &sourceMgr = mainDoc.getSourceMgr();
  SMLoc docStartLoc = sourceMgr.FindLocForLineAndColumn(
      bufferId, docLocAttr.getLine(), docLocAttr.getColumn());
  if (!docStartLoc.isValid())
    return;
  StringRef rawDocStr = decl->getDocString().getString();

  // Handle the case where the doc string is empty. In this case, we add a
  // code action for inserting a doc string.
  if (rawDocStr.empty()) {
    // Grab the indent for the doc string.
    size_t indent = docLocAttr.getColumn() - 1;
    const char *docStrStartLocPtr = docStartLoc.getPointer() - 1;
    while (indent > 0 && *docStrStartLocPtr == '\"') {
      --indent;
      --docStrStartLocPtr;
    }

    std::optional<std::string> docStr =
        generateDocStringTemplate(*decl, indent);
    if (docStr) {
      // Use the full string token for the warning.
      const char *docStrEndLocPtr = docStrStartLocPtr + 1;
      while (*docStrEndLocPtr == '\"')
        ++docStrEndLocPtr;
      SMLoc docStrStartLoc = SMLoc::getFromPointer(docStrStartLocPtr + 1);
      SMLoc docStrEndLoc = SMLoc::getFromPointer(docStrEndLocPtr);

      // Emit a warning with a fixit.
      sourceMgr.PrintMessage(docStrStartLoc, llvm::SourceMgr::DK_Warning,
                             kMissingDocMessage,
                             SMRange(docStrStartLoc, docStrEndLoc),
                             llvm::SMFixIt(docStartLoc, *docStr));
    }
  }

  // Translate a doc string location to a source location in the main buffer.
  auto translateLoc = [&](const char *loc) -> const char * {
    if (!docStartLoc.isValid())
      return nullptr;
    return docStartLoc.getPointer() + (loc - rawDocStr.data());
  };
  auto translateEndLocToMainDoc = [&](const char *loc) {
    // When translating an end location to the main document, we need to take
    // extra care when handling translation (the end position may not be
    // mapped).
    SMLoc lastLoc = SMLoc::getFromPointer(loc - 1);
    return SMLoc::getFromPointer(
        mainDoc.translateParserLoc(lastLoc).getPointer() + 1);
  };

  // Add the doc string.
  docStrings.emplace_back(llvm::SMRange(
      mainDoc.translateParserLoc(docStartLoc),
      translateEndLocToMainDoc(docStartLoc.getPointer() + rawDocStr.size())));

  // Process the code blocks in the doc string.
  SmallVector<std::pair<StringRef, mlir::Type>> persistentVariables;
  MojoParserContext &ctx = mainDoc.getParserContext();
  MojoASTDeclRef prevDecl = curReplDecl;
  LSPMojoREPLListener listener(sourceMgr, persistentVariables);
  for (const auto &block : docString->getCodeBlocks()) {
    StringRef blockContents = block.getRawCode();
    const char *blockStartLoc = translateLoc(blockContents.data());
    const char *blockEndLoc = translateLoc(blockContents.end());
    if (!blockStartLoc || !blockEndLoc || blockStartLoc == blockEndLoc)
      continue;
    StringRef contents(blockStartLoc, blockEndLoc - blockStartLoc);

    // Make sure the contents are mapped to the main document (important in the
    // case of the REPL, whose parsed buffer is different from the input).
    SMLoc docStartLoc =
        mainDoc.translateParserLoc(SMLoc::getFromPointer(contents.data()));
    SMLoc docEndLoc =
        translateEndLocToMainDoc(contents.data() + contents.size());
    StringRef mainDocContents(docStartLoc.getPointer(),
                              docEndLoc.getPointer() -
                                  docStartLoc.getPointer());

    // Create the new codeblock.
    CodeBlock *codeBlock =
        codeBlocks.emplace_back(new (codeBlockAllocator.Allocate()) CodeBlock(
            mainDocContents, persistentVariables, block.getRawIndentLevel()));

    // Parse the code block.
    auto [moduleDecl, exprFnDecl] = ctx.parseREPLExpression(
        listener, bufferId, contents, "__mojo_repl_lsp_main",
        persistentVariables, prevDecl);
    prevDecl = codeBlock->decl = moduleDecl;
    if (exprFnDecl)
      mainDoc.checkModuleSemantics(codeBlock->decl);

    // For the range of the code block, consume past the end to also include the
    // newline. This allows for more easily using the end of the last line for
    // different requests (like code completion).
    SMLoc docRangeEndLoc = SMLoc::getFromPointer(docEndLoc.getPointer() + 1);

    // Map the code block location to the main buffer.
    rangeToCodeBlock.insert(docStartLoc, docRangeEndLoc, codeBlock);
  }
}

auto MojoDocStrings::findContainingCodeBlock(SMLoc loc) -> CodeBlock * {
  if (auto it = rangeToCodeBlock.find(loc); it.valid() && it.start() <= loc)
    return it.value();
  return nullptr;
}

void MojoDocStrings::getFoldingRanges(SourceMgr &sourceMgr,
                                      std::vector<lsp::FoldingRange> &ranges) {
  for (DocString &docString : docStrings) {
    ranges.emplace_back(lsp::Range(sourceMgr, docString.range),
                        lsp::FoldingRange::kCommentKind);
  }
}

std::vector<KGEN::Mojo::CodeCompletionResult>
MojoDocStrings::CodeBlock::onCodeCompletion(llvm::SMLoc completeLoc,
                                            MojoParserContext &ctx) {
  uint64_t rawCompletePos = completeLoc.getPointer() - contents.data();
  return ctx.codeCompleteREPLExpression(contents, rawCompletePos,
                                        persistentVariables, decl);
}

std::optional<KGEN::Mojo::SignatureHelpResult>
MojoDocStrings::CodeBlock::onSignatureHelp(llvm::SMLoc loc,
                                           MojoParserContext &ctx) {
  uint64_t rawPos = loc.getPointer() - contents.data();
  return ctx.signatureHelpREPLExpression(contents, rawPos, persistentVariables,
                                         decl);
}

//===----------------------------------------------------------------------===//
// MojoTextDocument
//===----------------------------------------------------------------------===//

MojoTextDocument::MojoTextDocument(const lsp::URIForFile &uri,
                                   std::string &&contents, int64_t version,
                                   SendDiagnosticsFnRef sendDiagnosticsFn,
                                   LLCL::Runtime &runtime,
                                   LLCL::AnyAsyncValueRef chain,
                                   ArrayRef<std::string> includeDirs)
    : MojoDocument(Kind::kTextDocument, uri, version, sendDiagnosticsFn,
                   runtime, std::move(chain), includeDirs),
      contents(std::move(contents)) {
  // We add the main doc to the SourceMgr here to ensure it's considered the
  // "main" file.
  getSourceMgr().AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(this->contents, uri.file()), SMLoc());
}

void MojoTextDocument::parseDocumentImpl() {
  // Parse the file and make sure not to kill off things that weren't parsed,
  // these may get referenced later.
  parsedDecl = getParserContext().parseFile(getSourceMgr().getMainFileID(),
                                            /*eraseUnparsedDecls=*/false);
  checkModuleSemantics(parsedDecl);
  processDocStrings(docStrings, parsedDecl);
}

const lsp::URIForFile &MojoTextDocument::getURIFromContainedLoc(SMLoc loc) {
  return getURIs().front();
}

bool MojoTextDocument::containsLocation(SMLoc loc) {
  return getSourceMgr().FindBufferContainingLoc(loc) ==
         getSourceMgr().getMainFileID();
}

SMLoc MojoTextDocument::translateParserLoc(SMLoc loc) {
  if (containsLocation(loc))
    return loc;

  // If the location isn't in the main document, try to map it (e.g. in the case
  // of a location within a doc string code block).
  auto newLoc = getParserContext().getREPLLocMapper().mapLocation(loc);
  return newLoc.isValid() ? newLoc : loc;
}

SMLoc MojoTextDocument::getLocFromPos(const lsp::URIForFile &uri,
                                      lsp::Position position) {
  return position.getAsSMLoc(getSourceMgr());
}

SMRange MojoTextDocument::getFullRangeForURI(const lsp::URIForFile &uri) {
  return SMRange(SMLoc::getFromPointer(contents.data()),
                 SMLoc::getFromPointer(StringRef(contents).end()));
}

//===----------------------------------------------------------------------===//
// MojoTextDocument: Code Completion
//===----------------------------------------------------------------------===//

std::vector<KGEN::Mojo::CodeCompletionResult>
MojoTextDocument::onCodeCompletionSyncImpl(SMLoc completeLoc) {
  // Check for code completion within a code block.
  if (auto *codeBlock = docStrings.findContainingCodeBlock(completeLoc))
    return codeBlock->onCodeCompletion(completeLoc, getParserContext());

  // Otherwise, perform completion using the main doc.
  llvm::SourceMgr &sourceMgr = getSourceMgr();
  const llvm::MemoryBuffer *buffer =
      sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());

  // Query the mojo parser for potential completion results.
  uint64_t rawCompletePos =
      completeLoc.getPointer() - buffer->getBuffer().data();
  MLIRContext mlirContext{MLIRContext::Threading::DISABLED};
  return MojoParserContext::codeComplete(*buffer, rawCompletePos, &mlirContext,
                                         getCompilationOptions());
}

//===----------------------------------------------------------------------===//
// MojoTextDocument: Document Symbol
//===----------------------------------------------------------------------===//

std::vector<lsp::DocumentSymbol>
MojoTextDocument::onDocumentSymbolSync(const lsp::URIForFile &uri) {
  if (!parsedDecl)
    return {};
  std::vector<lsp::DocumentSymbol> symbols;
  getDocumentSymbols(parsedDecl, symbols);
  return symbols;
}

//===----------------------------------------------------------------------===//
// MojoTextDocument: Document Symbol
//===----------------------------------------------------------------------===//

std::vector<lsp::FoldingRange>
MojoTextDocument::onFoldingRangeSync(const lsp::URIForFile &uri) {
  if (!parsedDecl)
    return {};
  std::vector<lsp::FoldingRange> ranges;
  docStrings.getFoldingRanges(getSourceMgr(), ranges);
  return ranges;
}

//===----------------------------------------------------------------------===//
// MojoTextDocument: Signature Help
//===----------------------------------------------------------------------===//

std::optional<KGEN::Mojo::SignatureHelpResult>
MojoTextDocument::onSignatureHelpSyncImpl(SMLoc loc) {
  // Check for signature help within a code block.
  if (auto *codeBlock = docStrings.findContainingCodeBlock(loc))
    return codeBlock->onSignatureHelp(loc, getParserContext());

  // Otherwise, perform signature help using the main doc.
  llvm::SourceMgr &sourceMgr = getSourceMgr();
  const llvm::MemoryBuffer *buffer =
      sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());

  // Query the mojo parser for potential help results.
  uint64_t rawPos = loc.getPointer() - buffer->getBuffer().data();
  MLIRContext mlirContext{MLIRContext::Threading::DISABLED};
  return MojoParserContext::signatureHelp(*buffer, rawPos, &mlirContext,
                                          getCompilationOptions());
}

//===----------------------------------------------------------------------===//
// MojoNotebookDocument
//===----------------------------------------------------------------------===//

MojoNotebookDocument::MojoNotebookDocument(
    ArrayRef<lsp::URIForFile> notebookAndCellURIs, int64_t version,
    ArrayRef<lsp::NotebookCell> cellInfos,
    ArrayRef<lsp::TextDocumentItem> cellDocuments,
    SendDiagnosticsFnRef sendDiagnosticsFn, LLCL::Runtime &runtime,
    LLCL::AnyAsyncValueRef chain, ArrayRef<std::string> includeDirs)
    : MojoDocument(Kind::kNotebookDocument, notebookAndCellURIs, version,
                   sendDiagnosticsFn, runtime, std::move(chain), includeDirs) {
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
  MojoASTDeclRef prevDecl;
  for (Cell &cell : getCells()) {
    // If the cell isn't a python expression, parse it as normal.
    StringRef contents(cell.contents);
    if (!contents.consume_front("%%python")) {
      cell.persistentVariables = persistentVariables;
      MojoParserContext::ParsedREPLExpr result = ctx.parseREPLExpression(
          listener, cell.bufferId, cell.contents, "__mojo_repl_lsp_main",
          persistentVariables, prevDecl);
      prevDecl = cell.decl = result.moduleDecl;
      if (result.isValid())
        checkModuleSemantics(cell.decl);
      processDocStrings(cell.docStrings, cell.decl, /*curReplDecl=*/cell.decl);
      continue;
    }

    // Otherwise, this is a python expression. Extract the variables that are
    // implicitly imported into mojo and create stub definitions so that future
    // cells can reference them without error.
    ErrorOr<std::vector<std::unique_ptr<KGEN::Mojo::ExtractedPythonSymbol>>>
        symbolsOr = KGEN::Mojo::extractPythonSymbolsFromReplExpr(contents);
    if (failed(symbolsOr)) {
      getSourceMgr().PrintMessage(SMLoc::getFromPointer(contents.data()),
                                  llvm::SourceMgr::DK_Warning,
                                  symbolsOr.getError());
      continue;
    }

    // Build a new expression string containing variables for each of the
    // imported symbols.
    std::string pythonCell;
    llvm::raw_string_ostream os(pythonCell);
    os << "from python.object import PythonObject\n\n";
    for (auto &symbol : *symbolsOr)
      os << "var " << symbol->getName() << ": PythonObject\n";
    int pythonCellId = getSourceMgr().AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBuffer(pythonCell,
                                         (cell.uri.file() + "_py").str()),
        SMLoc());
    MojoParserContext::ParsedREPLExpr result = ctx.parseREPLExpression(
        listener, pythonCellId, pythonCell, "__mojo_repl_lsp_main",
        persistentVariables, prevDecl);
    prevDecl = result.moduleDecl;
  }
}

const lsp::URIForFile &MojoNotebookDocument::getURIFromContainedLoc(SMLoc loc) {
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
  if (!newLoc.isValid())
    return loc;

  // If this location isn't contained, try to map it again, this can happen if
  // the location is in a doc string.
  if (containsLocation(newLoc))
    return newLoc;
  newLoc = getParserContext().getREPLLocMapper().mapLocation(newLoc);
  return newLoc.isValid() ? newLoc : loc;
}

SMLoc MojoNotebookDocument::getLocFromPos(const lsp::URIForFile &uri,
                                          lsp::Position position) {
  Cell &cell = *uriToCell[uri.file()];
  return getSourceMgr().FindLocForLineAndColumn(
      cell.bufferId, position.line + 1, position.character + 1);
}

SMRange MojoNotebookDocument::getFullRangeForURI(const lsp::URIForFile &uri) {
  StringRef cellContents = uriToCell[uri.file()]->contents;
  return SMRange(SMLoc::getFromPointer(cellContents.data()),
                 SMLoc::getFromPointer(cellContents.end()));
}

//===----------------------------------------------------------------------===//
// MojoNotebookDocument: Code Completion
//===----------------------------------------------------------------------===//

std::vector<KGEN::Mojo::CodeCompletionResult>
MojoNotebookDocument::onCodeCompletionSyncImpl(SMLoc completeLoc) {
  llvm::SourceMgr &sourceMgr = getSourceMgr();
  int cellBufferId = sourceMgr.FindBufferContainingLoc(completeLoc);
  assert(cellBufferId > 0 && cellBufferId <= static_cast<int>(cells.size()) &&
         "expected to find cell buffer containing location");
  Cell &cell = *cells[cellBufferId - 1];

  // Check for code completion within a code block.
  if (auto *codeBlock = cell.docStrings.findContainingCodeBlock(completeLoc))
    return codeBlock->onCodeCompletion(completeLoc, getParserContext());

  // Query the mojo parser for potential completion results.
  uint64_t rawCompletePos = completeLoc.getPointer() - cell.contents.data();
  return getParserContext().codeCompleteREPLExpression(
      cell.contents, rawCompletePos, cell.persistentVariables, cell.decl);
}

//===----------------------------------------------------------------------===//
// MojoNotebookDocument: Document Symbol
//===----------------------------------------------------------------------===//

std::vector<lsp::DocumentSymbol>
MojoNotebookDocument::onDocumentSymbolSync(const lsp::URIForFile &uri) {
  auto cellIt = uriToCell.find(uri.file());
  if (cellIt == uriToCell.end() || !cellIt->second->decl ||
      cellIt->second->isPythonCell())
    return {};

  std::vector<lsp::DocumentSymbol> symbols;
  getDocumentSymbols(cellIt->second->decl, symbols);
  return symbols;
}

//===----------------------------------------------------------------------===//
// MojoNotebookDocument: Document Symbol
//===----------------------------------------------------------------------===//

std::vector<lsp::FoldingRange>
MojoNotebookDocument::onFoldingRangeSync(const lsp::URIForFile &uri) {
  auto cellIt = uriToCell.find(uri.file());
  if (cellIt == uriToCell.end() || !cellIt->second->decl ||
      cellIt->second->isPythonCell())
    return {};

  std::vector<lsp::FoldingRange> ranges;
  cellIt->second->docStrings.getFoldingRanges(getSourceMgr(), ranges);
  return ranges;
}

//===----------------------------------------------------------------------===//
// MojoNotebookDocument: Signature Help
//===----------------------------------------------------------------------===//

std::optional<KGEN::Mojo::SignatureHelpResult>
MojoNotebookDocument::onSignatureHelpSyncImpl(SMLoc loc) {
  llvm::SourceMgr &sourceMgr = getSourceMgr();
  int cellBufferId = sourceMgr.FindBufferContainingLoc(loc);
  assert(cellBufferId > 0 && cellBufferId <= static_cast<int>(cells.size()) &&
         "expected to find cell buffer containing location");
  Cell &cell = *cells[cellBufferId - 1];

  // Check for signature help within a code block.
  if (auto *codeBlock = cell.docStrings.findContainingCodeBlock(loc))
    return codeBlock->onSignatureHelp(loc, getParserContext());

  // Query the mojo parser for potential help results.
  uint64_t rawPos = loc.getPointer() - cell.contents.data();
  return getParserContext().signatureHelpREPLExpression(
      cell.contents, rawPos, cell.persistentVariables, cell.decl);
}

//===----------------------------------------------------------------------===//
// MojoServer::Impl
//===----------------------------------------------------------------------===//

struct MojoServer::Impl {
  Impl(ContextRef ctx, bool waitOnShutdown, SendDiagnosticsFn sendDiagnosticsFn,
       ArrayRef<std::string> includeDirs)
      : ctx(ctx.copy()),
        lspTelemetryContext(*ctx->get<M::Telemetry::TelemetryContext>()),
        waitOnShutdown(waitOnShutdown),
        sendDiagnosticsFn(std::move(sendDiagnosticsFn)),
        includeDirs(includeDirs) {}

  /// Begin the shutdown process for the server.
  void shutdown() {
    if (shuttingDown.exchange(true))
      return;

    // Invalidate all of the current documents if we aren't waiting for
    // shutdown, otherwise wait for them to parse and resolve actions. The
    // document ready chain is set and all related notifications are fired
    // synchronously.
    for (auto &[filename, file] : files) {
      if (waitOnShutdown)
        LLCL::await(file->getDocumentReadyChain());
      else
        file->invalidate();
    }

    // We always need to block on outstanding tasks, they may simple have been
    // cancelled above and we can expect them to finish quickly.
    for (auto &[filename, file] : files) {
      LLCL::await(file->getQuiescentChain());
    }

    files.clear();
    notebookCellToFile.clear();
  }

  /// Return if the server is shutting down.
  bool isShuttingDown() const { return shuttingDown; }

  /// Retrieve the document that matches completely the given filename. Return
  /// `nullptr` if no document is found.
  MojoDocumentRef findDocument(StringRef filename) {
    if (auto it = files.find(filename); it != files.end())
      return it->second.copy();

    auto it = notebookCellToFile.find(filename);
    return it != notebookCellToFile.end() ? it->second.copy()
                                          : MojoDocumentRef();
  }

  /// The global context.
  ContextRef ctx;

  /// Manages the telemetry instance use to record metrics and events.
  LSPTelemetryContext lspTelemetryContext;

  /// Records whether we are shutting down. Note that we can't clear
  /// the context, as this may teardown the associated telemetry.
  std::atomic<bool> shuttingDown = false;

  /// A flag indicating if the server should not invalidate requests on
  /// shutdown, and instead wait for them to complete.
  bool waitOnShutdown;

  /// The function used to send diagnostics to the client.
  SendDiagnosticsFn sendDiagnosticsFn;

  /// The files held by the server, mapped by their URI file name.
  llvm::StringMap<MojoDocumentRef> files;

  /// A mapping from individual notebook cells to their documents.
  llvm::StringMap<MojoDocumentRef> notebookCellToFile;

  /// A mapping from file to the last set of semantic tokens sent to the client.
  llvm::StringMap<lsp::SemanticTokens> prevSemanticTokensForFile;
  std::mutex lastSemanticTokensMutex;

  /// Additional directories to append to the search paths list.
  std::vector<std::string> includeDirs;
};

//===----------------------------------------------------------------------===//
// MojoServer
//===----------------------------------------------------------------------===//

MojoServer::MojoServer(std::unique_ptr<Impl> &&impl) : impl(std::move(impl)) {}
MojoServer::MojoServer(MojoServer &&) = default;

ErrorOr<MojoServer> MojoServer::create(bool singleThreaded, bool waitOnShutdown,
                                       SendDiagnosticsFn sendDiagnosticsFn,
                                       ArrayRef<std::string> includeDirs) {
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "mojo-lsp-server",
      Init::Options().withRuntimeOptions(LLCL::RuntimeOptions()
                                             .withSingleThreaded(singleThreaded)
                                             .withMainWillNotDonate()));
  if (ctxOr.isError())
    return ctxOr.takeError();
  auto impl = std::make_unique<Impl>(ctxOr->copy(), waitOnShutdown,
                                     std::move(sendDiagnosticsFn), includeDirs);
  MojoServer server(std::move(impl));
  return server;
}

MojoServer::~MojoServer() { shutdown(); }

LSPTelemetryContext &MojoServer::getLSPTelemetryContext() {
  return impl->lspTelemetryContext;
}

void MojoServer::shutdown() {
  if (impl)
    impl->shutdown();
}

//===----------------------------------------------------------------------===//
// Document Management

void MojoServer::addDocument(const lsp::URIForFile &uri, std::string &&contents,
                             int64_t version) {
  if (impl->isShuttingDown() ||
      !llvm::is_contained({"file", "test"}, uri.scheme()))
    return;
  auto [it, _] = impl->files.try_emplace(uri.file(), MojoDocumentRef());

  // If a document already exists, invalidate that version.
  LLCL::Runtime &runtime = *impl->ctx->get<LLCL::Runtime>();
  AnyAsyncValueRef chain = AsyncValueRef<Chain>::createReady(runtime);
  if (it->second) {
    it->second->invalidate();

    // Chain the new document to the old one.
    chain = it->second->getDocumentReadyChain();
  }

  // Create a new document.
  it->second = MojoTextDocumentRef::create(uri, std::move(contents), version,
                                           impl->sendDiagnosticsFn, runtime,
                                           std::move(chain), impl->includeDirs);
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

  { // Clear out the semantic token state for the file.
    std::lock_guard<std::mutex> lock(impl->lastSemanticTokensMutex);
    impl->prevSemanticTokensForFile.erase(uri.file());
  }

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
    const lsp::URIForFile &uri, ArrayRef<lsp::NotebookCell> cells,
    int64_t version, ArrayRef<lsp::TextDocumentItem> cellDocuments) {
  if (impl->isShuttingDown())
    return;
  MojoDocumentRef &file = impl->files[uri.file()];

  // If a document already exists, invalidate that version.
  LLCL::Runtime &runtime = *impl->ctx->get<LLCL::Runtime>();
  AnyAsyncValueRef chain = AsyncValueRef<Chain>::createReady(runtime);
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
                                         impl->sendDiagnosticsFn, runtime,
                                         std::move(chain), impl->includeDirs);
  for (const lsp::TextDocumentItem &cell : cellDocuments)
    impl->notebookCellToFile[cell.uri.file()] = file.copy();
}

void MojoServer::removeNotebookDocument(
    const lsp::URIForFile &uri,
    ArrayRef<lsp::TextDocumentIdentifier> cellDocuments) {
  // Remove the document from the server using the same flow as a normal text
  // document.
  removeDocument(uri);

  // Clear out mappings from the cell documents to the notebook document.
  for (const lsp::TextDocumentIdentifier &cell : cellDocuments) {
    impl->notebookCellToFile.erase(cell.uri.file());

    { // Clear out the semantic token state for the cell.
      std::lock_guard<std::mutex> lock(impl->lastSemanticTokensMutex);
      impl->prevSemanticTokensForFile.erase(cell.uri.file());
    }
  }
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
    const lsp::CodeActionParams &params,
    LSPResponder<std::vector<lsp::CodeAction>> responder) {
  lsp::URIForFile uri = params.textDocument.uri;

  // Check whether a particular CodeActionKind is included in the response.
  auto isKindAllowed = [only(params.context.only)](StringRef kind) {
    if (only.empty())
      return true;
    return llvm::any_of(only, [&](StringRef base) {
      return kind.consume_front(base) &&
             (kind.empty() || kind.starts_with("."));
    });
  };

  // We provide a code action for fixes on the specified diagnostics.
  if (!isKindAllowed(lsp::CodeAction::kQuickFix))
    return responder.reply(std::vector<lsp::CodeAction>());

  if (MojoDocumentRef doc = impl->findDocument(uri.file()))
    doc->getCodeActions(uri, params.range.start, params.context,
                        std::move(responder));
  else
    responder.replyInvalidRequest();
}

void MojoServer::onCodeCompletion(const lsp::CompletionParams &params,
                                  LSPResponder<lsp::CompletionList> responder) {
  if (MojoDocumentRef doc = impl->findDocument(params.textDocument.uri.file()))
    doc->onCodeCompletion(params.textDocument.uri, params.position,
                          std::move(responder));
  else
    responder.replyInvalidRequest();
}

void MojoServer::onDefinition(
    const lsp::TextDocumentPositionParams &params,
    LSPResponder<std::vector<lsp::Location>> responder) {
  if (MojoDocumentRef doc = impl->findDocument(params.textDocument.uri.file()))
    doc->onDefinition(params.textDocument.uri, params.position,
                      std::move(responder));
  else
    responder.replyInvalidRequest();
}

void MojoServer::onDocumentSymbol(
    const lsp::DocumentSymbolParams &params,
    LSPResponder<std::vector<lsp::DocumentSymbol>> responder) {
  if (MojoDocumentRef doc = impl->findDocument(params.textDocument.uri.file()))
    doc->onDocumentSymbol(params.textDocument.uri, std::move(responder));
  else
    responder.replyInvalidRequest();
}

void MojoServer::onFoldingRange(
    const lsp::FoldingRangeParams &params,
    LSPResponder<std::vector<lsp::FoldingRange>> responder) {
  if (MojoDocumentRef doc = impl->findDocument(params.textDocument.uri.file()))
    doc->onFoldingRange(params.textDocument.uri, std::move(responder));
  else
    responder.replyInvalidRequest();
}

void MojoServer::onHover(
    const lsp::TextDocumentPositionParams &params,
    LSPResponder<std::optional<mlir::lsp::Hover>> responder) {
  if (MojoDocumentRef doc = impl->findDocument(params.textDocument.uri.file()))
    doc->onHover(params.textDocument.uri, params.position,
                 std::move(responder));
  else
    responder.replyInvalidRequest();
}

void MojoServer::onInlayHint(
    const lsp::InlayHintsParams &params,
    LSPResponder<std::vector<lsp::InlayHint>> responder) {
  if (MojoDocumentRef doc = impl->findDocument(params.textDocument.uri.file()))
    doc->onInlayHint(params.textDocument.uri, params.range,
                     std::move(responder));
  else
    responder.replyInvalidRequest();
}

void MojoServer::onReferences(
    const lsp::ReferenceParams &params,
    LSPResponder<std::vector<lsp::Location>> responder) {
  if (MojoDocumentRef doc = impl->findDocument(params.textDocument.uri.file()))
    doc->onReferences(params.textDocument.uri, params.position,
                      params.context.includeDeclaration, std::move(responder));
  else
    responder.replyInvalidRequest();
}

/// Increment a numeric string: "" -> 1 -> 2 -> ... -> 9 -> 10 -> 11 ...
static void incrementNumericString(std::string &str) {
  for (char &c : llvm::reverse(str)) {
    if (c != '9') {
      ++c;
      return;
    }
    c = '0';
  }
  str.insert(str.begin(), '1');
}

void MojoServer::onSemanticTokens(
    const lsp::SemanticTokensParams &params,
    LSPResponder<std::optional<lsp::SemanticTokens>> responder) {
  MojoDocumentRef doc = impl->findDocument(params.textDocument.uri.file());
  if (!doc)
    return responder.replyInvalidRequest();

  doc->onSemanticTokens(
      params.textDocument.uri,
      [this, uri = params.textDocument.uri.file().str(),
       responder = std::move(responder)](
          std::optional<std::vector<SemanticToken>> tokens, bool outdated,
          bool invalid) mutable {
        if (outdated)
          return responder.replyOutdatedRequest();
        if (invalid)
          return responder.replyInvalidRequest();
        if (!tokens)
          return responder.reply({});

        lsp::SemanticTokens result(toLspSemanticTokens(*tokens));
        {
          std::lock_guard<std::mutex> lock(impl->lastSemanticTokensMutex);
          lsp::SemanticTokens &prevResult =
              impl->prevSemanticTokensForFile[uri];

          prevResult.tokens = result.tokens;
          incrementNumericString(prevResult.resultId);
          result.resultId = prevResult.resultId;
        }
        responder.reply(std::move(result));
      });
}

void MojoServer::onSemanticTokensDelta(
    const lsp::SemanticTokensDeltaParams &params,
    LSPResponder<std::optional<lsp::SemanticTokensOrDelta>> responder) {
  MojoDocumentRef doc = impl->findDocument(params.textDocument.uri.file());
  if (!doc)
    return responder.replyInvalidRequest();

  doc->onSemanticTokens(
      params.textDocument.uri,
      [this, uri = params.textDocument.uri.file().str(),
       prevId = params.previousResultId, responder = std::move(responder)](
          std::optional<std::vector<SemanticToken>> tokens, bool outdated,
          bool invalid) mutable {
        if (outdated)
          return responder.replyOutdatedRequest();
        if (invalid)
          return responder.replyInvalidRequest();
        if (!tokens)
          return responder.reply({});
        std::vector<lsp::SemanticToken> lspTokens =
            toLspSemanticTokens(*tokens);

        lsp::SemanticTokensOrDelta result;
        {
          std::lock_guard<std::mutex> lock(impl->lastSemanticTokensMutex);
          lsp::SemanticTokens &prevResult =
              impl->prevSemanticTokensForFile[uri];

          if (prevResult.resultId == prevId) {
            result.edits = diffTokens(prevResult.tokens, lspTokens);
          } else {
            lsp::Logger::debug(
                "semanticTokens/full/delta: wanted edits vs {0} but last "
                "result had ID {1}. Returning full token list.",
                prevId, prevResult.resultId);
            result.tokens = lspTokens;
          }

          prevResult.tokens = std::move(lspTokens);
          incrementNumericString(prevResult.resultId);
          result.resultId = prevResult.resultId;
        }
        responder.reply(std::move(result));
      });
}

void MojoServer::getSignatureHelp(const lsp::TextDocumentPositionParams &params,
                                  LSPResponder<lsp::SignatureHelp2> responder) {
  if (MojoDocumentRef doc = impl->findDocument(params.textDocument.uri.file()))
    doc->onSignatureHelp(params.textDocument.uri, params.position,
                         std::move(responder));
  else
    responder.replyInvalidRequest();
}

void MojoServer::onRename(const lsp::RenameParams &params,
                          LSPResponder<lsp::WorkspaceEdit> responder) {
  if (MojoDocumentRef doc =
          impl->findDocument(params.textDocument.uri.file())) {
    doc->onRename(params.textDocument.uri, params.position, params.newName,
                  std::move(responder));
  } else {
    responder.replyInvalidRequest();
  }
}
