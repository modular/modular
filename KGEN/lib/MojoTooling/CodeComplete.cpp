//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoTooling/CodeComplete.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoLexer/Lexer.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoParser/SharedState.h"
#include "KGEN/MojoTooling/ASTDeclRef.h"
#include "KGEN/MojoTooling/ASTDeclView.h"
#include "KGEN/MojoTooling/ParserDriver.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;
using namespace M::KGEN::Mojo;

using llvm::SMLoc;

/// Returns true if the given member should be shown during lookup within
/// `decl`. If `isModuleLookup` is true, we are looking up nested modules.
static bool showDeclDuringLookup(MojoASTDeclRef decl, StringRef &member,
                                 bool isModuleLookup = false) {
  if (llvm::isa_and_present<PackageOp>(decl.getIfOperation())) {
    // If this is a module lookup, we only want to show non-init modules defined
    // within the package.
    if (isModuleLookup)
      return member.consume_front("$") && member != "__init__";
    // Otherwise, show everything but internally defined modules.
    return !member.starts_with("$");
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Listener
//===----------------------------------------------------------------------===//

namespace {
/// This class implements a listener that collects code completion results.
struct CodeCompletionListener : public ParserListener {
  CodeCompletionListener(std::vector<CodeCompletionResult> &results,
                         llvm::SourceMgr &sourceMgr)
      : results(results), sourceMgr(sourceMgr) {}
  ~CodeCompletionListener() override = default;

  /// Returns true if the listener is interested in being notified for the given
  /// location.
  bool isInterestedInLoc(SMLoc parserLoc) override {
    return completionRange.Start.getPointer() <= parserLoc.getPointer() &&
           parserLoc.getPointer() <= completionRange.End.getPointer();
  }

  /// Notify the listener that an import is currently being resolved.
  void onImport(SMLoc importLoc) override {
    // Simple helper for adding completion results and dropping duplicates.
    StringSet<> addedImports;
    auto addImportCompletion = [&](StringRef name, bool isPackage) {
      if (addedImports.insert(name).second)
        results.emplace_back(name, isPackage ? CodeCompletionResult::kPackage
                                             : CodeCompletionResult::kModule);
    };

    // Compute the viable imports for the given location.
    for (const std::string &dir :
         parserContext->getModuleSearchDirectories(sourceMgr.getMainFileID())) {
      std::error_code ec;
      for (const auto &it : std::filesystem::directory_iterator(dir, ec)) {
        if (ec)
          continue;
        std::string extension = it.path().extension().string();
        if (extension == ".mojo" || extension == ".🔥")
          addImportCompletion(it.path().stem().string(), /*isPackage=*/false);
        else if (extension == ".mojopkg" || extension == ".📦" ||
                 isMojoSourcePackagePath(it.path()))
          addImportCompletion(it.path().stem().string(), /*isPackage=*/true);
      }
    }
  }

  /// Notify the listener that an import of a module within the given package is
  /// currently being resolved.
  void onImport(ResolveInputDeclFn getPackageDecl, SMLoc importLoc) override {
    MojoASTDeclRef packageDecl = getPackageDecl();
    for (MojoASTDeclRef::ChildEntry child : packageDecl.getChildren()) {
      StringRef name = child.getName();
      if (!showDeclDuringLookup(packageDecl, name, /*isModuleLookup=*/true))
        continue;

      addCompletionForOp(name, *child.getDecls().begin(), [](Operation *op) {
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
        if (!showDeclDuringLookup(decl, name))
          continue;

        // TODO: Include information about overloads here and just handle multi
        // decls in general.
        addCompletionForOp(name, *child.getDecls().begin());
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
            .Case([](FuncOp) { return CodeCompletionResult::kFunction; })
            .Case([](StructFieldOp) { return CodeCompletionResult::kField; })
            .Case<LetRegDeclOp, VarLetDeclOp>(
                [](auto op) { return CodeCompletionResult::kVariable; })
            .Default(CodeCompletionResult::kUnknown);

    CodeCompletionResult result(name, kind);
    if (auto view = decl.getView())
      result.documentation = view->getFullMarkdownString();
    results.emplace_back(result);
  }

  /// The results that have been collected so far.
  DenseSet<ASTDecl *> addedResults;
  std::vector<CodeCompletionResult> &results;

  /// The range of acceptable locations for the completion.
  llvm::SMRange completionRange;

  /// The source manager.
  llvm::SourceMgr &sourceMgr;

  /// The current parser context.
  MojoParserContext *parserContext = nullptr;
};
} // namespace

//===----------------------------------------------------------------------===//
// Entrypoint
//===----------------------------------------------------------------------===//

std::vector<CodeCompletionResult>
MojoParserContext::codeComplete(llvm::MemoryBufferRef buffer,
                                uint64_t completionPosition,
                                MLIRContext *context, LLCL::Runtime &runtime,
                                const KGEN::CompilationOptions &options) {
  return codeComplete(
      buffer, completionPosition, context, runtime, options,
      [](MojoParserContext &ctx, int fileID) { ctx.parseFile(fileID); });
}

std::vector<CodeCompletionResult> MojoParserContext::codeComplete(
    llvm::MemoryBufferRef buffer, uint64_t completionPosition,
    MLIRContext *context, LLCL::Runtime &runtime,
    const KGEN::CompilationOptions &options,
    function_ref<void(MojoParserContext &, int)> parserCallback) {
  if (buffer.getBufferSize() < completionPosition)
    return {};
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(buffer),
                               SMLoc());

  // Add a diagnostic handler that consumes anything emitted during parsing. We
  // don't care about diagnostics here, there will almost always be a diagnostic
  // emitted when grabbing completion results from a partial file.
  sourceMgr.setDiagHandler([](const llvm::SMDiagnostic &, void *) {});

  // Build the listener that collects the results.
  std::vector<CodeCompletionResult> results;
  CodeCompletionListener listener(results, sourceMgr);

  ParserConfig config(context, runtime, options);
  config.parserListener = &listener;

  // We don't want to cache the main module, but imported modules can be cached.
  config.moduleCachingLevel = ParserConfig::kCacheImports;

  // Disable as much of the diagnostic machinery as possible, we don't care
  // about diagnostics for completion results.
  config.maxNotesPerDiagnostic = 0;

  // Build the parser context with our listener.
  MojoParserContext parserContext(sourceMgr, config);
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

  parserCallback(parserContext, sourceMgr.getMainFileID());
  return results;
}
