//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoTooling/CodeComplete.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/CallEmission.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoParser/ExprNode.h"
#include "KGEN/MojoParser/Lexer.h"
#include "KGEN/MojoParser/SharedState.h"
#include "KGEN/MojoTooling/ASTDeclRef.h"
#include "KGEN/MojoTooling/ASTDeclView.h"
#include "KGEN/MojoTooling/ParserDriver.h"
#include "Support/Filesystem/Paths.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;
using namespace M::KGEN::Mojo;

using llvm::SMLoc;
using llvm::SMRange;

/// Returns if the given range, inclusive, contains `loc`.
static bool containsLoc(SMRange range, SMLoc loc) {
  return range.Start.getPointer() <= loc.getPointer() &&
         loc.getPointer() <= range.End.getPointer();
}

//===----------------------------------------------------------------------===//
// BaseCompletionListener
//===----------------------------------------------------------------------===//

namespace {
/// This class implements a base listener for completion or signature help that
/// handles shared listener setup.
struct BaseCompletionListener : public ParserListener {
  BaseCompletionListener(SourceMgr &sourceMgr) : sourceMgr(sourceMgr) {}
  ~BaseCompletionListener() override = default;

  /// The source manager.
  llvm::SourceMgr &sourceMgr;

  /// The range of acceptable locations for the completion.
  llvm::SMRange completionRange;

  /// The current parser context.
  MojoParserContext *parserContext = nullptr;
};
} // namespace

//===----------------------------------------------------------------------===//
// Code Completion: Listener
//===----------------------------------------------------------------------===//

/// Returns true if the given member should be shown during lookup within
/// `decl`. If `isModuleLookup` is true, we are looking up nested modules.
static bool showDeclDuringLookup(MojoASTDeclRef decl, StringRef &member,
                                 MojoASTDeclRef child,
                                 bool isModuleLookup = false) {
  if (llvm::isa_and_present<PackageOp>(decl.getIfOperation())) {
    bool childIsPackageOrModule =
        llvm::isa_and_present<FileModuleOp, PackageOp>(child.getIfOperation());
    // If this is a module lookup, we only want to show non-init modules defined
    // within the package.
    if (isModuleLookup)
      return childIsPackageOrModule && member != "__init__";
    // Otherwise, show everything but internally defined modules.
    return !childIsPackageOrModule;
  }
  return true;
}

namespace {
/// This class implements a listener that collects code completion results.
struct CodeCompletionListener : public BaseCompletionListener {
  CodeCompletionListener(std::vector<CodeCompletionResult> &results,
                         llvm::SourceMgr &sourceMgr)
      : BaseCompletionListener(sourceMgr), results(results) {}
  ~CodeCompletionListener() override = default;

  /// Returns true if the listener is interested in being notified for the given
  /// location.
  bool isInterestedInLoc(SMLoc parserLoc) override {
    return containsLoc(completionRange, parserLoc);
  }

  /// Notify the listener that an import is currently being resolved.
  void onImport(SMLoc importLoc) override {
    // Simple helper for adding completion results and dropping duplicates.
    StringSet<> addedImports;
    auto addImportCompletion = [&](const std::filesystem::path &path,
                                   bool isPackage) {
      std::string name = path.stem().string();
      if (!addedImports.insert(name).second)
        return;
      results.emplace_back(name, isPackage ? CodeCompletionResult::kPackage
                                           : CodeCompletionResult::kModule);

      // Grab the documentation for the import. Do this out of the current
      // context to avoid pulling in a bunch of unwanted state.
      MLIRContext ctx{MLIRContext::Threading::DISABLED};
      ParserConfig config(&ctx, parserContext->getCompilationOptions());
      MojoParserContext importContext(sourceMgr, config);
      if (auto module = importContext.parseFileOrPackageNonRecursive(path)) {
        if (auto view = module.getView())
          results.back().documentation = view->getFullMarkdownString();
      }
    };

    // Standard library packages are exposed as top-level imports, even though
    // they are defined inside the 'stdlib' package.
    addedImports.insert("stdlib");
    onImport(
        [&]() {
          return &parserContext->getSharedState().importModule(
              "stdlib", PackageOp(), SMLoc());
        },
        importLoc);
    for (CodeCompletionResult &result : results)
      addedImports.insert(result.label);

    // Compute the viable imports for the given location.
    for (const std::string &dir :
         parserContext->getModuleSearchDirectories(sourceMgr.getMainFileID())) {
      std::error_code ec;
      for (const auto &it : std::filesystem::directory_iterator(dir, ec)) {
        if (ec)
          continue;
        if (Filesystem::isMojoSourceFile(it.path()))
          addImportCompletion(it.path(), /*isPackage=*/false);
        else if (Filesystem::isMojoBinaryPackagePath(it.path()) ||
                 Filesystem::isMojoSourcePackagePath(it.path()))
          addImportCompletion(it.path(), /*isPackage=*/true);
      }
    }
  }

  /// Notify the listener that an import of a module within the given package is
  /// currently being resolved.
  void onImport(ResolveInputDeclFn getPackageDecl, SMLoc importLoc) override {
    MojoASTDeclRef packageDecl = getPackageDecl();
    for (MojoASTDeclRef::ChildEntry child : packageDecl.getChildren()) {
      StringRef name = child.getName();
      MojoASTDeclRef childDecl = *child.getDecls().begin();
      if (!showDeclDuringLookup(packageDecl, name, childDecl,
                                /*isModuleLookup=*/true))
        continue;

      addCompletionForOp(name, childDecl, [](Operation *op) {
        return isa<FileModuleOp, PackageOp>(op);
      });
    }
  }

  /// Notify the listener that a member within the given decl is being looked
  /// up.
  void onMemberLookup(ResolveInputDeclFn getDeclFn, llvm::SMLoc lookupLoc,
                      bool searchParentScopes) override {
    MojoASTDeclRef decl = getDeclFn();

    auto collectDeclChildren = [&](MojoASTDeclRef decl) {
      for (MojoASTDeclRef::ChildEntry child : decl.getChildren()) {
        StringRef name = child.getName();
        MojoASTDeclRef childDecl = *child.getDecls().begin();
        if (!showDeclDuringLookup(decl, name, childDecl))
          continue;

        // TODO: Include information about overloads here and just handle multi
        // decls in general.
        addCompletionForOp(name, childDecl);
      }
    };

    // Collect all of the decls in the current scope.
    if (!searchParentScopes)
      return collectDeclChildren(decl);

    // Collect all of the decls in the current scope and all parent scopes.
    do {
      collectDeclChildren(decl);
      decl = decl.getParentDecl();
    } while (
        !llvm::isa_and_present<PackageOp, ModuleOp>(decl.getIfOperation()));
  }

  /// Utility function to add a completion result for the given decl. An
  /// optional filter that returns which operations should be considered.
  void addCompletionForOp(StringRef name, MojoASTDeclRef decl,
                          function_ref<bool(Operation *)> filter = {}) {
    if (!addedResults.insert(&*decl).second)
      return;

    Operation *op = decl.getIfOperation();
    if (!op || (filter && !filter(op)))
      return;
    auto kind =
        TypeSwitch<Operation *, CodeCompletionResult::Kind>(op)
            .Case([](FileModuleOp) { return CodeCompletionResult::kModule; })
            .Case([](PackageOp) { return CodeCompletionResult::kPackage; })
            .Case([](StructDeclOp) { return CodeCompletionResult::kStruct; })
            .Case([](TraitDeclOp) { return CodeCompletionResult::kTrait; })
            .Case([](FuncOp) { return CodeCompletionResult::kFunction; })
            .Case([](StructFieldOp) { return CodeCompletionResult::kField; })
            .Case([](VarDeclOp op) { return CodeCompletionResult::kVariable; })
            .Default(CodeCompletionResult::kUnknown);

    CodeCompletionResult result(name, kind);
    if (auto view = decl.getView())
      result.documentation = view->getFullMarkdownString();
    results.emplace_back(result);
  }

  /// The results that have been collected so far.
  DenseSet<ASTDecl *> addedResults;
  std::vector<CodeCompletionResult> &results;
};
} // namespace

//===----------------------------------------------------------------------===//
// Signature Help: Listener
//===----------------------------------------------------------------------===//

namespace {
/// This class implements a listener that collects signature help results.
struct SignatureHelpListener : public BaseCompletionListener {
  SignatureHelpListener(llvm::SourceMgr &sourceMgr, SignatureHelpResult &result)
      : BaseCompletionListener(sourceMgr), result(result) {}
  ~SignatureHelpListener() override = default;

  /// Returns true if the listener is interested in being notified for the given
  /// location.
  bool isInterestedInLoc(SMLoc loc) override {
    // Filter at a high level for locations in the main document, we'll filter
    // further when examining calls.
    return sourceMgr.getMainFileID() == sourceMgr.FindBufferContainingLoc(loc);
  }

  void onCall(ArrayRef<ASTDecl *> decls, llvm::SMLoc rparenLoc,
              const CallOperands &operands) override {
    auto findInterestedOperand = [&]() -> std::optional<size_t> {
      for (const auto &[index, operand] :
           llvm::enumerate(operands.posOperands)) {
        if (containsLoc(completionRange, operand.expr->getRangeStart()))
          return index;
      }

      // Consider the rparen location if it is within the completion range.
      if (operands.getNumKwOperands() == 0 &&
          containsLoc(completionRange, rparenLoc))
        return operands.posOperands.size();

      // TODO: Consider kwargs.
      return std::nullopt;
    };

    // Check if any of the operands are within the completion range.
    std::optional<size_t> operandIndex = findInterestedOperand();
    if (!operandIndex)
      return;
    result.activeParameter = *operandIndex;

    // Collect the signatures for each of the decls.
    for (MojoASTDeclRef decl : decls) {
      std::unique_ptr<DeclView> declView = decl.getView();
      if (!declView)
        continue;
      if (auto *fnView = dyn_cast<FunctionDeclView>(declView.get())) {
        if (operands.posOperands.size() > fnView->getArguments().size())
          continue;

        // If this is the first function and it's a method, bump the active
        // parameter past the self argument.
        if (result.signatures.empty() && fnView->isMethod())
          ++result.activeParameter;

        SignatureHelpResult::Signature signature;
        SmallVector<std::pair<unsigned, unsigned>> argOffsets;
        signature.label = fnView->getDeclarationSnippet(
            /*parameterOffsets=*/nullptr, &argOffsets);
        addDeclDocAndParametersToSignature(signature, *fnView,
                                           fnView->getArguments(), argOffsets);
        result.signatures.emplace_back(std::move(signature));
      }
    }
  }

  void onParameterBinding(ArrayRef<ASTDecl *> decls, llvm::SMLoc rsquareLoc,
                          ArrayRef<ExprNode *> parameters) override {
    auto findInterestedParam = [&]() -> std::optional<size_t> {
      for (const auto &[index, param] : llvm::enumerate(parameters))
        if (containsLoc(completionRange, param->getRangeStart()))
          return index;

      // Consider the rparen location if it is within the completion range.
      if (containsLoc(completionRange, rsquareLoc))
        return parameters.size();
      return std::nullopt;
    };

    // Check if any of the operands are within the completion range.
    std::optional<size_t> paramIndex = findInterestedParam();
    if (!paramIndex)
      return;
    result.activeParameter = *paramIndex;

    // Collect the signatures for each of the decls.
    for (MojoASTDeclRef decl : decls) {
      std::unique_ptr<DeclView> declView = decl.getView();
      if (!declView)
        continue;
      TypeSwitch<DeclView *>(declView.get())
          .Case<FunctionDeclView, StructDeclView>([&](auto *declView) {
            if (parameters.size() > declView->getParameters().size())
              return;
            SignatureHelpResult::Signature signature;
            SmallVector<std::pair<unsigned, unsigned>> paramOffsets;
            signature.label = declView->getDeclarationSnippet(&paramOffsets);
            addDeclDocAndParametersToSignature(
                signature, *declView, declView->getParameters(), paramOffsets);
            result.signatures.emplace_back(std::move(signature));
          });
    }
  }

  /// Utility function for adding the documentation and parameter information
  /// form the given decl to a signature.
  template <typename RangeT>
  static void addDeclDocAndParametersToSignature(
      SignatureHelpResult::Signature &signature, DeclView &declView,
      RangeT &&range, ArrayRef<std::pair<unsigned, unsigned>> offsets) {
    signature.documentation = declView.getFullMarkdownString();
    for (const auto &[arg, offset] : llvm::zip(range, offsets))
      signature.parameters.push_back({offset, arg.getMarkdownDocString()});
  }

  /// The result that has been collected so far.
  SignatureHelpResult &result;
};
} // namespace

//===----------------------------------------------------------------------===//
// Entrypoint
//===----------------------------------------------------------------------===//

/// Parse the given buffer for completion results using the given listener
/// implementation.
static void parseCompletionImpl(
    llvm::MemoryBufferRef buffer, uint64_t completionPosition,
    MLIRContext *context, const KGEN::CompilationOptions &options,
    function_ref<void(MojoParserContext &, int)> parserCallback,
    BaseCompletionListener &listener, bool disableModuleCaching) {
  if (buffer.getBufferSize() < completionPosition)
    return;
  listener.sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(buffer), SMLoc());

  // Add a diagnostic handler that consumes anything emitted during parsing. We
  // don't care about diagnostics here, there will almost always be a diagnostic
  // emitted when grabbing completion results from a partial file.
  listener.sourceMgr.setDiagHandler([](const llvm::SMDiagnostic &, void *) {});

  ParserConfig config(context, options);
  config.parserListener = &listener;

  // Disable as much of the diagnostic machinery as possible, we don't care
  // about diagnostics for completion results.
  config.maxNotesPerDiagnostic = 0;

  // Build the parser context with our listener.
  MojoParserContext parserContext(listener.sourceMgr, config);
  listener.parserContext = &parserContext;

  // Compute the start completion location. We first trim the buffer to the
  // last non-whitespace, and then to the start of any identifier. We often get
  // completion requests for lookups that are partially formed already (e.g. a
  // completion on `p` to get things like `print`).
  StringRef completionPosStr =
      buffer.getBuffer().take_front(completionPosition).rtrim();
  while (!completionPosStr.empty()) {
    char c = completionPosStr.back();
    if (!(llvm::isAlpha(c) || llvm::isDigit(c) || c == '_' || c == '$'))
      break;
    completionPosStr = completionPosStr.drop_back();
  }
  listener.completionRange.Start =
      SMLoc::getFromPointer(completionPosStr.end());

  // Compute the end completion location by finding the next token from the
  // input completion position.
  completionPosStr = buffer.getBuffer().drop_front(completionPosition);
  Lexer lexer(parserContext.getSharedState().diags, completionPosStr,
              completionPosStr.data());
  listener.completionRange.End = lexer.getToken().getLoc();

  parserCallback(parserContext, listener.sourceMgr.getMainFileID());
}

//===----------------------------------------------------------------------===//
// Code Completion

std::vector<CodeCompletionResult> MojoParserContext::codeComplete(
    llvm::MemoryBufferRef buffer, uint64_t completionPosition,
    MLIRContext *context, const KGEN::CompilationOptions &options) {
  return codeComplete(
      buffer, completionPosition, context, options,
      [](MojoParserContext &ctx, int fileID) { ctx.parseFile(fileID); });
}

std::vector<CodeCompletionResult> MojoParserContext::codeComplete(
    llvm::MemoryBufferRef buffer, uint64_t completionPosition,
    MLIRContext *context, const KGEN::CompilationOptions &options,
    function_ref<void(MojoParserContext &, int)> parserCallback,
    bool disableModuleCaching) {
  llvm::SourceMgr sourceMgr;
  std::vector<CodeCompletionResult> results;
  CodeCompletionListener listener(results, sourceMgr);
  parseCompletionImpl(buffer, completionPosition, context, options,
                      parserCallback, listener, disableModuleCaching);
  return results;
}

//===----------------------------------------------------------------------===//
// Signature Help

std::optional<SignatureHelpResult>
MojoParserContext::signatureHelp(llvm::MemoryBufferRef buffer,
                                 uint64_t position, MLIRContext *context,
                                 const KGEN::CompilationOptions &options) {
  return signatureHelp(
      buffer, position, context, options,
      [](MojoParserContext &ctx, int fileID) { ctx.parseFile(fileID); });
}

std::optional<SignatureHelpResult> MojoParserContext::signatureHelp(
    llvm::MemoryBufferRef buffer, uint64_t completionPosition,
    MLIRContext *context, const KGEN::CompilationOptions &options,
    function_ref<void(MojoParserContext &, int)> parserCallback,
    bool disableModuleCaching) {
  llvm::SourceMgr sourceMgr;
  SignatureHelpResult result;
  SignatureHelpListener listener(sourceMgr, result);
  parseCompletionImpl(buffer, completionPosition, context, options,
                      parserCallback, listener, disableModuleCaching);
  return result.signatures.empty() ? std::nullopt : std::optional(result);
}
