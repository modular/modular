//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_LSP_SERVER_MOJODOCUMENT_H
#define KGEN_TOOLS_MOJO_LSP_SERVER_MOJODOCUMENT_H

#include "AsyncRT/Runtime/Runtime.h"
#include "KGEN/MojoParser/DocString.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoTooling/ASTDeclRef.h"
#include "KGEN/MojoTooling/ParserDriver.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "MojoServer.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ReferenceCounted.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/MapVector.h"

/// Define ordering operators for SMLoc for use in IntervalMap.
namespace llvm {
inline bool operator<(const SMLoc &lhs, const SMLoc &rhs) {
  return lhs.getPointer() < rhs.getPointer();
}
inline bool operator<=(const SMLoc &lhs, const SMLoc &rhs) {
  return lhs.getPointer() <= rhs.getPointer();
}
} // namespace llvm

namespace M::Mojo::LSP {
class MojoDocStrings;
struct SemanticToken;

//===----------------------------------------------------------------------===//
// MojoInlayHint
//===----------------------------------------------------------------------===//

/// This class is used to represent an inlay hint in the document. This is a bit
/// more stripped, optimized, and mojo specific compared to lsp::InlayHint.
struct MojoInlayHint {
  MojoInlayHint(mlir::lsp::InlayHintKind kind, StringRef label, SMLoc loc)
      : label(label), loc(loc), leftIndent(0), kind(kind), paddingLeft(false),
        paddingRight(false) {}

  /// Generate an LSP inlay hint from this inlay hint.
  mlir::lsp::InlayHint toLspInlayHint(SourceMgr &sourceMgr) const;

  /// Order inlay hints by their location.
  bool operator<(const MojoInlayHint &other) const {
    return loc.getPointer() < other.loc.getPointer();
  }

  /// The label of the inlay hint.
  StringRef label;

  /// The location of the inlay hint.
  SMLoc loc;

  /// An optional left indent for the inlay hint.
  unsigned leftIndent : 28;

  /// The kind of the inlay hint.
  mlir::lsp::InlayHintKind kind : 2;

  /// If the hint should be padded to the left.
  bool paddingLeft : 1;

  /// If the hint should be padded to the right.
  bool paddingRight : 1;
};

//===----------------------------------------------------------------------===//
// MojoDocument
//===----------------------------------------------------------------------===//

/// This class represents all of the information pertaining to a specific Mojo
/// document.
struct MojoDocument : public ReferenceCounted<MojoDocument> {
public:
  MojoDocument(const MojoDocument &) = delete;
  MojoDocument &operator=(const MojoDocument &) = delete;
  virtual ~MojoDocument() = default;

  /// Return the version of this document.
  int64_t getVersion() const { return version; }

  /// Return the runtime used for this document.
  LLCL::Runtime &getRuntime() const { return runtime; }

  /// Return the URIs of this document.
  ArrayRef<mlir::lsp::URIForFile> getURIs() const { return uris; }

  /// Return the source manager used for this document.
  llvm::SourceMgr &getSourceMgr() { return sourceMgr; }

  /// Return the compilation options for this document.
  const KGEN::CompilationOptions &getCompilationOptions() const;

  /// Return the parser context for this document.
  MojoParserContext &getParserContext() const;

  /// Invalidate this document.
  void invalidate();

  /// Return a chain that will be ready when the document is parsed.
  AnyAsyncValueRef getDocumentReadyChain() const {
    return isDocumentParsed.copy();
  }

  /// Return a chain that will be ready when currently scheduled tasks are done.
  AnyAsyncValueRef getQuiescentChain() {
    std::lock_guard<std::mutex> lk(isDocumentParsedMutex);
    return isQuiescent.copy();
  }

  //===--------------------------------------------------------------------===//
  // RTTI Utilities
  //===--------------------------------------------------------------------===//

  /// The kind of document this is.
  enum class Kind {
    kTextDocument,
    kNotebookDocument,
  };

  /// Return the kind of this document.
  Kind getKind() const { return kind; }

  //===--------------------------------------------------------------------===//
  // Document Utilities
  //===--------------------------------------------------------------------===//

  /// Returns true if the document contains the given location.
  virtual bool containsLocation(llvm::SMLoc loc) = 0;

  /// Returns true if the document contains the given location.
  virtual llvm::SMLoc getLocFromPos(const mlir::lsp::URIForFile &uri,
                                    mlir::lsp::Position position) = 0;

  /// Return the source range from the given LSP range.
  llvm::SMRange getLocFromPos(const mlir::lsp::URIForFile &uri,
                              const mlir::lsp::Range &range) {
    return llvm::SMRange(getLocFromPos(uri, range.start),
                         getLocFromPos(uri, range.end));
  }

  /// Return a location range for the document of the given uri.
  virtual llvm::SMRange
  getFullRangeForURI(const mlir::lsp::URIForFile &uri) = 0;

  /// Translate the given parser location into one usable by the language
  /// server.
  virtual llvm::SMLoc translateParserLoc(llvm::SMLoc loc) { return loc; }
  llvm::SMRange translateParserLoc(llvm::SMRange range) {
    llvm::SMLoc newStart = translateParserLoc(range.Start);
    auto newEnd = llvm::SMLoc::getFromPointer(
        newStart.getPointer() +
        (range.End.getPointer() - range.Start.getPointer()));
    return {newStart, newEnd};
  }

  /// Returns a language server uri for the given source location. `mainFileURI`
  /// corresponds to the uri for the main file of the source manager.
  std::optional<mlir::lsp::URIForFile> getURIFromLoc(llvm::SMLoc loc);

  /// Returns a language server location from the given diagnostic.
  std::optional<mlir::lsp::Location>
  getLocationFromDiag(const llvm::SMDiagnostic &diag);

  /// Get a document symbol with the given ASTDecl, appending it to the given
  /// vector.
  void getDocumentSymbols(MojoASTDeclRef decl,
                          std::vector<mlir::lsp::DocumentSymbol> &symbols);

  /// Get a document symbol with the given ASTDecl, appending it to the given
  /// vector. The provided functor defines whether a decl should be included in
  /// the symbol list.
  void getDocumentSymbols(MojoASTDeclRef decl,
                          std::vector<mlir::lsp::DocumentSymbol> &symbols,
                          function_ref<bool(MojoASTDeclRef)> shouldIncludeDecl);

  /// Recursively process the document strings in decls nested within `decl`.
  /// The provided functor defines whether a decl should be processed. If the
  /// main document represents a REPL module, `curReplDecl` is the AST decl for
  /// the REPL module that contains `decl`. In the case of a normal text
  /// document, `curReplDecl` is null.
  void processDocStrings(MojoDocStrings &docStrings, MojoASTDeclRef decl,
                         unsigned bufferId,
                         function_ref<bool(MojoASTDeclRef)> shouldIncludeDecl,
                         MojoASTDeclRef curReplDecl = {});

  /// Recursively process the document strings in decls nested within `decl`. If
  /// the main document represents a REPL module, `curReplDecl` is the AST decl
  /// for the REPL module that contains `decl`. In the case of a normal text
  /// document, `curReplDecl` is null.
  void processDocStrings(MojoDocStrings &docStrings, MojoASTDeclRef decl,
                         MojoASTDeclRef curReplDecl = {});

  /// Check the given the parsed module decl for high-level semantic issues. Any
  /// errors are reported to the source manager.
  void checkModuleSemantics(MojoASTDeclRef decl);

  //===--------------------------------------------------------------------===//
  // Asynchronous LSP Queries
  //===--------------------------------------------------------------------===//

  //===--------------------------------------------------------------------===//
  // Code Actions

  void
  getCodeActions(const mlir::lsp::URIForFile &uri, const mlir::lsp::Range &pos,
                 const mlir::lsp::CodeActionContext &context,
                 LSPResponder<std::vector<mlir::lsp::CodeAction>> responder);

  //===--------------------------------------------------------------------===//
  // Language Features

  void onCodeCompletion(const mlir::lsp::URIForFile &uri,
                        const mlir::lsp::Position &completePos,
                        LSPResponder<mlir::lsp::CompletionList> responder);

  void onDefinition(const mlir::lsp::URIForFile &uri,
                    const mlir::lsp::Position &pos,
                    LSPResponder<std::vector<mlir::lsp::Location>> responder);

  void onDocumentSymbol(
      const mlir::lsp::URIForFile &uri,
      LSPResponder<std::vector<mlir::lsp::DocumentSymbol>> responder);

  void
  onFoldingRange(const mlir::lsp::URIForFile &uri,
                 LSPResponder<std::vector<mlir::lsp::FoldingRange>> responder);

  void onHover(const mlir::lsp::URIForFile &uri, const mlir::lsp::Position &pos,
               LSPResponder<std::optional<mlir::lsp::Hover>> responder);

  void onInlayHint(const mlir::lsp::URIForFile &uri,
                   const mlir::lsp::Range &range,
                   LSPResponder<std::vector<mlir::lsp::InlayHint>> responder);

  void onReferences(const mlir::lsp::URIForFile &uri,
                    const mlir::lsp::Position &position,
                    bool includeDeclaration,
                    LSPResponder<std::vector<mlir::lsp::Location>> responder);

  void onSemanticTokens(
      const mlir::lsp::URIForFile &uri,
      OnSemanticTokensResultFn<std::optional<std::vector<SemanticToken>>>
          onSemanticTokens);

  void onSignatureHelp(const mlir::lsp::URIForFile &uri,
                       const mlir::lsp::Position &pos,
                       LSPResponder<mlir::lsp::SignatureHelp2> responder);

  void onRename(const mlir::lsp::URIForFile &uri,
                const mlir::lsp::Position &pos, StringRef newName,
                LSPResponder<mlir::lsp::WorkspaceEdit> responder);

protected:
  MojoDocument(Kind kind, ArrayRef<mlir::lsp::URIForFile> uris, int64_t version,
               SendDiagnosticsFnRef sendDiagnosticsFn, LLCL::Runtime &runtime,
               LLCL::AnyAsyncValueRef chain, ArrayRef<std::string> includeDirs);

  /// A collection of MLIR and Mojo related entities used to invoke the parser.
  /// Its lifetime is tied to that of the AST objects gotten from the parser.
  /// It also sets up a SourceMgr with the given MojoDocument as its main file.
  struct Context;

  //===--------------------------------------------------------------------===//
  // Derived Document Hooks
  //===--------------------------------------------------------------------===//

  /// Hook that is invoked to perform the raw document parsing process.
  virtual void parseDocumentImpl() = 0;

  /// Hook that returns the URI for the given contained location.
  virtual const mlir::lsp::URIForFile &
  getURIFromContainedLoc(llvm::SMLoc loc) = 0;

  //===--------------------------------------------------------------------===//
  // Language Features

  /// Hook that is invoked to perform code completion at the given position.
  virtual std::vector<KGEN::Mojo::CodeCompletionResult>
  onCodeCompletionSyncImpl(llvm::SMLoc completeLoc) = 0;

  /// Hook that returns the symbols within the document.
  virtual std::vector<mlir::lsp::DocumentSymbol>
  onDocumentSymbolSync(const mlir::lsp::URIForFile &uri) = 0;

  /// Hook that returns the folding ranges within the document.
  virtual std::vector<mlir::lsp::FoldingRange>
  onFoldingRangeSync(const mlir::lsp::URIForFile &uri) = 0;

  /// Hook that is invoked to perform signature help at the given position.
  virtual std::optional<KGEN::Mojo::SignatureHelpResult>
  onSignatureHelpSyncImpl(llvm::SMLoc loc) = 0;

private:
  /// Parse the document and populate the index based on the current contents.
  void parseDocument();

  /// Mark the current document as being finished parsing.
  void markDocumentParsed();

  /// Get a new chain for a new task to mark when finished.
  AsyncValueRef<Chain> newTaskChain();

  /// Start a task that depends on the document being parsed.
  template <typename FnT>
  void startTaskAfterParsing(FnT &&fn) {
    AsyncValueRef<Chain> done = newTaskChain();
    isDocumentParsed.andThenAsync([doc = RCRef<MojoDocument>::copy(this),
                                   fn = std::forward<FnT>(fn),
                                   done = std::move(done)]() mutable {
      fn(*doc);
      std::move(done).emplace();
    });
  }

  //===--------------------------------------------------------------------===//
  // Synchronous LSP Queries
  //===--------------------------------------------------------------------===//

  //===--------------------------------------------------------------------===//
  // Diagnostics

  std::optional<mlir::lsp::Diagnostic>
  buildLspDiagnosticFromSMDiagnostic(llvm::SourceMgr &sourceMgr,
                                     ArrayRef<llvm::SMDiagnostic> diags,
                                     const mlir::lsp::URIForFile &uri);

  //===--------------------------------------------------------------------===//
  // Code Actions

  std::vector<mlir::lsp::CodeAction>
  getCodeActionsSync(llvm::SMRange range,
                     const mlir::lsp::CodeActionContext &context);

  //===--------------------------------------------------------------------===//
  // Language Features

  mlir::lsp::CompletionList onCodeCompletionSync(llvm::SMLoc completeLoc);

  std::vector<mlir::lsp::Location> onDefinitionSync(llvm::SMLoc loc);

  std::optional<mlir::lsp::Hover> onHoverSync(llvm::SMLoc loc);

  std::vector<mlir::lsp::InlayHint> onInlayHintSync(llvm::SMRange range);

  std::vector<mlir::lsp::Location> onReferencesSync(SMLoc smLoc,
                                                    bool includeDeclaration);

  ErrorOr<std::vector<mlir::lsp::TextEdit>>
  onRenameSync(SMLoc loc, const std::string &newName);

  std::optional<std::vector<SemanticToken>>
  onSemanticTokensSync(llvm::SMRange range);

  mlir::lsp::SignatureHelp onSignatureHelpSync(llvm::SMLoc loc);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  //===--------------------------------------------------------------------===//
  // Static Fields

  /// The following fields are always available for access and don't require
  /// additional synchronization.

  /// The kind of this document.
  Kind kind;

  /// The uri of the file.
  SmallVector<mlir::lsp::URIForFile> uris;

  /// The version of this file.
  int64_t version = 0;

  /// The function used to send diagnostics for this document.
  SendDiagnosticsFnRef sendDiagnosticsFn;

  /// The runtime used when parsing the file.
  LLCL::Runtime &runtime;

  /// A flag indicating if this document version has been invalidated.
  std::atomic<bool> isInvalidated = false;

  /// The source manager used to parse the document.
  llvm::SourceMgr sourceMgr;

  //===--------------------------------------------------------------------===//
  // Parsed Fields

  /// The following fields are only available after the document has been
  /// parsed, when `isDocumentParsed` is ready.

  /// An async value readied when the document is parsed, and one that is
  /// signaled when all asynchronous jobs are completed.
  AsyncValueRef<Chain> isDocumentParsed;
  AsyncValueRef<Chain> isQuiescent;
  std::mutex isDocumentParsedMutex;

  /// A set of fixits for diagnostics emitted for the current version of the
  /// file.
  llvm::StringMap<
      std::map<mlir::lsp::Range, std::vector<mlir::lsp::CodeAction>>>
      fixits;

  /// An ordered set of inlay hints for the current version of the file.
  std::vector<MojoInlayHint> inlayHints;

  /// Indicates if the document produced parser errors.
  bool hasParserErrors = false;

  /// The overall parser context.
  std::unique_ptr<Context> context;
};

using MojoDocumentRef = RCRef<MojoDocument>;

//===----------------------------------------------------------------------===//
// MojoDocStrings
//===----------------------------------------------------------------------===//

/// This class represents all of the doc string state within a Mojo document,
/// including code block state. Code blocks somewhat function as independent
/// documents, as they are parsed and processed separately from the main
/// document, but are still tied to the main document (e.g. for locations,
/// requests, etc.).
class MojoDocStrings {
public:
  MojoDocStrings() : rangeToCodeBlock(allocator) {}

  /// This class represents an individual code block within a doc string.
  struct CodeBlock {
    CodeBlock(StringRef contents,
              SmallVector<std::pair<StringRef, Type>> persistentVariables,
              unsigned contentsIndent)
        : contents(contents),
          persistentVariables(std::move(persistentVariables)),
          contentsIndent(contentsIndent) {}

    /// Attempt to perform code completion at the given location.
    std::vector<KGEN::Mojo::CodeCompletionResult>
    onCodeCompletion(llvm::SMLoc loc, MojoParserContext &ctx);

    /// Attempt to compute signature help at the given location.
    std::optional<KGEN::Mojo::SignatureHelpResult>
    onSignatureHelp(llvm::SMLoc loc, MojoParserContext &ctx);

    /// The contents of the code block.
    StringRef contents;

    /// The persistent REPL variables defined in code blocks defined before this
    /// one in the same doc string.
    SmallVector<std::pair<StringRef, Type>> persistentVariables;

    /// The AST decl for the module containing this code block.
    MojoASTDeclRef decl;

    /// The indent of the code within the contents.
    unsigned contentsIndent;
  };

  /// This class represents an individual doc string within a Mojo document.
  struct DocString {
    DocString(llvm::SMRange range) : range(range) {}

    /// The range of the doc string.
    llvm::SMRange range;
  };

  /// Add the doc string and any code blocks for the given decl. `bufferId` is
  /// the source manager buffer for the main document. If the main document
  /// represents a REPL module, `curReplDecl` is the AST decl for the REPL
  /// module that contains `decl`. In the case of a normal text document,
  /// `curReplDecl` is null.
  void addDocString(MojoDocument &mainDoc, MojoASTDeclRef decl,
                    MojoASTDeclRef curReplDecl, unsigned bufferId);

  /// Find the code block that contains the given location.
  CodeBlock *findContainingCodeBlock(llvm::SMLoc loc);

  /// Get the folding ranges for held doc strings.
  void getFoldingRanges(SourceMgr &sourceMgr,
                        std::vector<mlir::lsp::FoldingRange> &ranges);

private:
  using MapT = llvm::IntervalMap<
      SMLoc, CodeBlock *,
      llvm::IntervalMapImpl::NodeSizer<SMLoc, CodeBlock *>::LeafSize,
      llvm::IntervalMapHalfOpenInfo<SMLoc>>;

  /// An allocator to use for code blocks.
  llvm::SpecificBumpPtrAllocator<CodeBlock> codeBlockAllocator;

  /// The code blocks within the document.
  SmallVector<CodeBlock *> codeBlocks;

  /// The doc strings within the document.
  SmallVector<DocString> docStrings;

  /// A map of source locations within the main document to code blocks.
  MapT::Allocator allocator;
  MapT rangeToCodeBlock;
};

//===----------------------------------------------------------------------===//
// MojoTextDocument
//===----------------------------------------------------------------------===//

/// This class represents all of the information pertaining to a specific Mojo
/// text document, i.e. a .mojo or .🔥 file.
struct MojoTextDocument : public MojoDocument {
public:
  MojoTextDocument(const mlir::lsp::URIForFile &uri, std::string &&contents,
                   int64_t version, SendDiagnosticsFnRef sendDiagnosticsFn,
                   LLCL::Runtime &runtime, LLCL::AnyAsyncValueRef chain,
                   ArrayRef<std::string> includeDirs);
  MojoTextDocument(const MojoDocument &) = delete;
  MojoTextDocument &operator=(const MojoDocument &) = delete;

  /// Return the contents of this document.
  StringRef getContents() const { return contents; }

  /// Support LLVM RTTI.
  static bool classof(const MojoDocument *doc) {
    return doc->getKind() == Kind::kTextDocument;
  }

private:
  //===--------------------------------------------------------------------===//
  // Derived Document Hooks
  //===--------------------------------------------------------------------===//

  /// Hook that is invoked to perform the raw document parsing process.
  void parseDocumentImpl() override;

  /// Hook that returns the URI for the given contained location.
  const mlir::lsp::URIForFile &getURIFromContainedLoc(llvm::SMLoc loc) override;

  /// Returns true if the document contains the given location.
  bool containsLocation(llvm::SMLoc loc) override;

  /// Translate the given parser location into one usable by the language
  /// server.
  llvm::SMLoc translateParserLoc(llvm::SMLoc loc) override;

  /// Returns true if the document contains the given location.
  llvm::SMLoc getLocFromPos(const mlir::lsp::URIForFile &uri,
                            mlir::lsp::Position position) override;

  /// Return a location range for the document of the given uri.
  llvm::SMRange getFullRangeForURI(const mlir::lsp::URIForFile &uri) override;

  //===--------------------------------------------------------------------===//
  // Language Features

  std::vector<KGEN::Mojo::CodeCompletionResult>
  onCodeCompletionSyncImpl(llvm::SMLoc completeLoc) override;

  std::vector<mlir::lsp::DocumentSymbol>
  onDocumentSymbolSync(const mlir::lsp::URIForFile &uri) override;

  std::vector<mlir::lsp::FoldingRange>
  onFoldingRangeSync(const mlir::lsp::URIForFile &uri) override;

  std::optional<KGEN::Mojo::SignatureHelpResult>
  onSignatureHelpSyncImpl(llvm::SMLoc loc) override;

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  /// The full string contents of the file.
  std::string contents;

  /// The AST decl for the module containing this document.
  MojoASTDeclRef parsedDecl;

  /// The doc strings within this document.
  MojoDocStrings docStrings;
};

using MojoTextDocumentRef = RCRef<MojoTextDocument>;

//===----------------------------------------------------------------------===//
// MojoNotebookDocument
//===----------------------------------------------------------------------===//

/// This class represents all of the information pertaining to a specific Mojo
/// notebook document, e.g. a jupyter notebook file.
struct MojoNotebookDocument : public MojoDocument {
public:
  /// This class represents a cell within the notebook.
  struct Cell {
    Cell(mlir::lsp::URIForFile uri, StringRef contents)
        : uri(std::move(uri)), contents(contents.str()) {}

    /// Return if this cell is a python cell.
    bool isPythonCell() const {
      return StringRef(contents).starts_with("%%python");
    }

    /// The uri of the cell
    mlir::lsp::URIForFile uri;

    /// The contents of the cell.
    std::string contents;

    /// The buffer id of the cell contents within the source manager.
    unsigned bufferId = 0;

    /// The AST decl for the module containing this cell.
    MojoASTDeclRef decl;

    /// The persistent REPL variables defined before this cell.
    SmallVector<std::pair<StringRef, Type>> persistentVariables;

    /// The doc strings within this cell.
    MojoDocStrings docStrings;
  };

  MojoNotebookDocument(ArrayRef<mlir::lsp::URIForFile> notebookAndCellURIs,
                       int64_t version,
                       ArrayRef<mlir::lsp::NotebookCell> cellInfos,
                       ArrayRef<mlir::lsp::TextDocumentItem> cellDocuments,
                       SendDiagnosticsFnRef sendDiagnosticsFn,
                       LLCL::Runtime &runtime, LLCL::AnyAsyncValueRef chain,
                       ArrayRef<std::string> includeDirs);
  MojoNotebookDocument(const MojoDocument &) = delete;
  MojoNotebookDocument &operator=(const MojoDocument &) = delete;

  /// Return the cells within this document.
  auto getCells() { return llvm::make_pointee_range(cells); }

  /// Support LLVM RTTI.
  static bool classof(const MojoDocument *doc) {
    return doc->getKind() == Kind::kNotebookDocument;
  }

private:
  //===--------------------------------------------------------------------===//
  // Derived Document Hooks
  //===--------------------------------------------------------------------===//

  /// Hook that is invoked to perform the raw document parsing process.
  void parseDocumentImpl() override;

  /// Returns true if the document contains the given location.
  bool containsLocation(llvm::SMLoc loc) override;

  /// Translate the given parser location into one usable by the language
  /// server.
  llvm::SMLoc translateParserLoc(llvm::SMLoc loc) override;

  /// Returns true if the document contains the given location.
  llvm::SMLoc getLocFromPos(const mlir::lsp::URIForFile &uri,
                            mlir::lsp::Position position) override;

  /// Return a location range for the document of the given uri.
  llvm::SMRange getFullRangeForURI(const mlir::lsp::URIForFile &uri) override;

  /// Hook that returns the URI for the given contained location.
  const mlir::lsp::URIForFile &getURIFromContainedLoc(llvm::SMLoc loc) override;

  //===--------------------------------------------------------------------===//
  // Language Features

  std::vector<KGEN::Mojo::CodeCompletionResult>
  onCodeCompletionSyncImpl(llvm::SMLoc completeLoc) override;

  std::vector<mlir::lsp::DocumentSymbol>
  onDocumentSymbolSync(const mlir::lsp::URIForFile &uri) override;

  std::vector<mlir::lsp::FoldingRange>
  onFoldingRangeSync(const mlir::lsp::URIForFile &uri) override;

  std::optional<KGEN::Mojo::SignatureHelpResult>
  onSignatureHelpSyncImpl(llvm::SMLoc loc) override;

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  //===--------------------------------------------------------------------===//
  // Static Fields

  /// The following fields are always available for access and don't require
  /// additional synchronization.

  /// The cells within the document, mapped from the uri of the cell.
  llvm::StringMap<Cell *> uriToCell;
  std::vector<std::unique_ptr<Cell>> cells;
};

using MojoNotebookDocumentRef = RCRef<MojoNotebookDocument>;
} // namespace M::Mojo::LSP

#endif // KGEN_TOOLS_MOJO_LSP_SERVER_MOJODOCUMENT_H
