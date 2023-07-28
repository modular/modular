//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the main entrypoints for the Mojo parser.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser.h"

#include "ASTDecl.h"
#include "DeclResolver.h"
#include "DocString.h"
#include "Lexer.h"
#include "ParserBase.h"
#include "ParserDriverImpl.h"
#include "SharedState.h"

#include "KGEN/CompilationOptions.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ASTDeclRef.h"
#include "KGEN/MojoParser/ASTDeclView.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/Telemetry/Telemetry.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "mlir/Bytecode/Encoding.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/Timing.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"

#include <filesystem>

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

using llvm::SourceMgr;

//===----------------------------------------------------------------------===//
// MojoParserListener
//===----------------------------------------------------------------------===//

void MojoParserListener::onAliasDecl(MojoASTDeclRef declRef,
                                     llvm::SMLoc identifierLoc) {}
void MojoParserListener::onArgumentDecl(MojoASTDeclRef declRef,
                                        llvm::SMLoc identifierLoc) {}
void MojoParserListener::onFunctionDecl(MojoASTDeclRef declRef,
                                        llvm::SMLoc identifierLoc) {}
void MojoParserListener::onImport(llvm::SMLoc importLoc) {}
void MojoParserListener::onImport(MojoASTDeclRef packageDecl,
                                  llvm::SMLoc importLoc) {}
void MojoParserListener::onMemberLookup(MojoASTDeclRef decl, llvm::SMLoc loc) {}
void MojoParserListener::onModuleImport(MojoASTDeclRef declRef,
                                        StringRef spelling,
                                        llvm::SMLoc importLoc) {}
void MojoParserListener::onModuleDecl(MojoASTDeclRef declRef,
                                      llvm::SMLoc identifierLoc) {}
void MojoParserListener::onStructDecl(MojoASTDeclRef declRef,
                                      llvm::SMLoc identifierLoc) {}
void MojoParserListener::onStructFieldDecl(MojoASTDeclRef declRef,
                                           llvm::SMLoc identifierLoc) {}
void MojoParserListener::onVariableDecl(MojoASTDeclRef declRef,
                                        llvm::SMLoc identifierLoc) {}
void MojoParserListener::onRef(MojoASTDeclRef declRef, StringRef spelling,
                               llvm::SMLoc loc) {}

//===----------------------------------------------------------------------===//
// MojoParserContext::Impl
//===----------------------------------------------------------------------===//

MojoParserContext::Impl::Impl(llvm::SourceMgr &sourceMgr,
                              MojoParserConfig &config)
    : sharedState(sourceMgr, config) {
  // Create the top-level outer decl, which will contain all things we parse.
  module = ModuleOp::create(UnknownLoc::get(sharedState.getContext()));
  topLevelDecl = &sharedState.declResolver->addDecl(
      *module, SMLoc(), StringAttr(), /*parentDecl=*/nullptr, LexerCursor(),
      LexerCursor(), /*indentation=*/-1);
  sharedState.initialize(*topLevelDecl);
}

//===----------------------------------------------------------------------===//
// MojoParserContext
//===----------------------------------------------------------------------===//

MojoParserContext::MojoParserContext(SourceMgr &sourceMgr,
                                     MojoParserConfig &config)
    : impl(std::make_unique<Impl>(sourceMgr, config)) {}
MojoParserContext::~MojoParserContext() {
  // Finalize any imported bytecode now that we've resolved everything. This
  // avoids dangling references to operations from the bytecode.
  (void)impl->sharedState.finalizeImportedBytecodeModules();
}

ModuleOp MojoParserContext::getModule() {
  return cast<ModuleOp>(impl->topLevelDecl->getIfOperation());
}

llvm::SourceMgr &MojoParserContext::getSourceMgr() {
  return impl->sharedState.getSourceMgr();
}

std::vector<std::string>
MojoParserContext::getModuleSearchDirectories(unsigned fileId) {
  std::vector<std::string> searchDirs;
  impl->sharedState.traverseImportDirectories(fileId, [&](StringRef dir) {
    searchDirs.push_back(dir.str());
    return WalkResult::advance();
  });
  return searchDirs;
}

const KGEN::CompilationOptions &MojoParserContext::getCompilationOptions() {
  return impl->sharedState.options;
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

/// Sort the order of uses of the given value using the given set of operation
/// IDs. This ensures a deterministic order of uses.
static void sortValueUses(Value value,
                          DenseMap<Operation *, unsigned> &operationIDs) {
  if (value.use_empty() || value.hasOneUse())
    return;

  // The current use list order for each of the uses of the value.
  SmallVector<std::pair<unsigned, uint64_t>> currentOrder;

  // Functor to record a new use to the current order.
  auto addUse = [&](unsigned index, OpOperand &operand) {
    uint64_t nextID =
        mlir::bytecode::getUseID(operand, operationIDs.at(operand.getOwner()));
    currentOrder.emplace_back(index, nextID);
    return nextID;
  };

  // Compute the current order of the use-list with respect to the global
  // ordering. Detect if the order is already sorted while doing so.
  uint64_t currentID = addUse(0, *value.use_begin());
  bool isSorted = true;
  for (auto it : llvm::drop_begin(llvm::enumerate(value.getUses()))) {
    uint64_t nextID = addUse(it.index(), it.value());
    isSorted &= std::exchange(currentID, nextID) >= nextID;
  }
  // If we know the list is already sorted, there is nothing left to do.
  if (isSorted)
    return;

  // Sort the order based on the operation user, and build the shuffled order
  // mapping.
  llvm::sort(currentOrder, [](auto elem1, auto elem2) {
    return elem1.second > elem2.second;
  });
  SmallVector<unsigned> shuffledOrder(currentOrder.size());
  for (unsigned i = 0, e = currentOrder.size(); i != e; ++i)
    shuffledOrder[currentOrder[i].first] = i;
  value.shuffleUseList(shuffledOrder);
}

/// Sort the order of uses of values defined within the given top level
/// operation. This enforces a deterministic order to the uses, and ensures that
/// any generated bytecode is stable across compilations (regardless of how the
/// IR was parsed).
static void sortValueUses(Operation *topLevelOp) {
  unsigned operationID = 0;
  DenseMap<Operation *, unsigned> operationIDs;
  topLevelOp->walk([&](Operation *op, mlir::WalkStage stage) {
    // Before visiting any of the regions of the operation, we simply check if
    // we should record this operation's ID (i.e. if it has operands).
    if (stage.isBeforeAllRegions()) {
      if (op->getNumOperands())
        operationIDs.insert({op, operationID++});
      return;
    }

    // After the walk has finished, sort the uses of all values defined within
    // this operation.
    if (stage.isAfterAllRegions()) {
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (BlockArgument &arg : block.getArguments())
            sortValueUses(arg, operationIDs);
          for (Operation &op : block)
            for (Value result : op.getResults())
              sortValueUses(result, operationIDs);
        }
      }
    }
  });
}

/// Erase all declarations unreachable from the main module. This is primarily
/// useful when there are imported modules that were not lazily loaded from the
/// cache. This function should run after textual modules are saved to the
/// cache, reducing the amount of IR reaching the compiler. In addition, it
/// provides the compiler a canonical form of IR coming out of the parser. This,
/// for instance, ensures the cache key computed on parser output does not
/// depend on whether the parser has cache hits for lazy loading.
static void eraseUnreachableDecls(ASTDecl &decl) {
  // Don't purge decls when parsing a package.
  if (isa<PackageOp>(decl))
    return;

  TimeTraceScope traceScope("eraseUnreachableDecls");
  // Start by erasing unresolved imports. This puts the module in a canonical
  // form.
  auto declModule = cast<FileModuleOp>(decl);
  auto module = cast<ModuleOp>(declModule->getParentOp());
  module.walk([](Operation *op) {
    // Imports are not found underneath structs and functions.
    if (isa<StructDeclOp, LIT::FuncOp>(op))
      return WalkResult::skip();
    if (isa<UnresolvedImportOp, UnresolvedWildcardImportOp>(op))
      op->erase();
    return WalkResult::advance();
  });

  // The remaining skippable decls are all symbol operations. Run symbol DCE
  // rooted at the main module.
  mlir::SymbolTableCollection symtab;
  DenseSet<Operation *> liveSymbols;
  std::vector<Operation *> worklist;
  declModule.walk([&](mlir::SymbolOpInterface symbol) {
    liveSymbols.insert(symbol);
    worklist.push_back(symbol);
  });

  // This walker will mark all referenced symbols as live.
  mlir::AttrTypeWalker refCollector;
  auto markLive = [&](Operation *op) {
    if (liveSymbols.insert(op).second)
      worklist.push_back(op);
  };
  refCollector.addWalk([&](SymbolRefAttr ref) {
    // Mark all referenced symbols as live. Invalid symbol references will get
    // caught by the verifier.
    SmallVector<Operation *, 4> symbols;
    (void)symtab.lookupSymbolIn(module, ref, symbols);
    for (Operation *symbol : symbols)
      markLive(symbol);
  });

  // Propagate liveness.
  while (!worklist.empty()) {
    Operation *cur = worklist.back();
    worklist.pop_back();
    // Collect symbol references between this symbol table and any child symbol
    // tables. Nested `lit.func` operations are trickier, however.
    cur->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
      if (op != cur) {
        // Mark nested functions as live, which also recurses on them.
        if (auto func = dyn_cast<LIT::FuncOp>(op);
            func && func.getParamDeclAttr())
          markLive(func);
        if (op->hasTrait<OpTrait::SymbolTable>())
          return WalkResult::skip();
      }
      refCollector.walk(op->getAttrDictionary());
      for (Type type : op->getResultTypes())
        refCollector.walk(type);
      return WalkResult::advance();
    });
  }

  // Walk post-order to erase dead symbols.
  module.walk([&](mlir::SymbolOpInterface symbol) {
    if (!liveSymbols.contains(symbol))
      symbol.erase();
  });
}

/// Parse a mojo module or package into the specified MLIR context. Returns the
/// resultant IR, and the decl for the module or package. This abstracts away
/// the shared setup between module and package parsing.
static std::tuple<OwningOpRef<mlir::ModuleOp>, ASTDecl *>
importMojoImpl(StringRef moduleIdentifier, SourceMgr &sourceMgr,
               SharedState &sharedState, mlir::TimingScope &ts,
               SmallVectorImpl<std::string> *includedFiles,
               function_ref<ASTDecl &(ModuleOp)> buildDeclFn) {
  MLIRContext *context = sharedState.getContext();
  [[maybe_unused]] auto timeScope =
      sharedState.runtime.getTelemetryContext()->createUInt64Timer(
          "mojo.parser.compile.time");
  [[maybe_unused]] auto flushTelemetry =
      sharedState.runtime.getTelemetryContext()->autoFlush();

  // This is the result module we are parsing into.
  auto fileLoc = FileLineColLoc::get(context, moduleIdentifier, /*line=*/0,
                                     /*column=*/0);
  mlir::OwningOpRef<ModuleOp> module(ModuleOp::create(fileLoc));

  // Build the decl for the main module.
  ASTDecl &moduleDecl = buildDeclFn(*module);

  // Resolve everything within the main input module.
  sharedState.declResolver->resolveAllReferencedFrom(moduleDecl);

  // Finalize the imported bytecode now that we've resolved everything. This
  // will drop bytecode operations that never got referenced.
  if (failed(sharedState.finalizeImportedBytecodeModules()))
    return {nullptr, nullptr};

  // We fail either if we have a non-recoverable parse error, or if we emitted
  // an error and then recovered.  In either case, the IR will not be valid and
  // the caller should not verify it.
  if (sharedState.diags.isErrorEmitted())
    return {nullptr, nullptr};

  // Make sure the parse module has no other structural problems detected by
  // the verifier.
  {
    auto verificationTimer = ts.nest("Verify module");
    if (failed(verify(*module)))
      return {};
  }

  // Now that resolution is finished, cache the state of modules we have parsed.
  // TODO: We should be able to cache even in the presence of warnings and
  // errors. We can store the diagnostics and replay on cache load.
  if (!sharedState.diags.isDiagnosticEmitted()) {
    sharedState.cacheParsedModules();
    eraseUnreachableDecls(moduleDecl);
    sortValueUses(*module);
  }

  // Set the included files if requested.
  if (includedFiles)
    llvm::append_range(*includedFiles, sharedState.getIncludedFiles());
  return {std::move(module), &moduleDecl};
}

/// Parse the specified Mojo file into the specified MLIR context. Returns the
/// resultant IR, and the decl for the module represented by the input file.
static std::tuple<OwningOpRef<mlir::ModuleOp>, ASTDecl *>
importMojoFileImpl(SourceMgr &sourceMgr, SharedState &sharedState,
                   mlir::TimingScope &ts,
                   SmallVectorImpl<std::string> *includedFiles = nullptr) {
  auto sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  StringRef bufName = sourceBuf->getBufferIdentifier();
  DebugInfo::DIBuilder::ScopeGuard fileGuard;

  return importMojoImpl(
      bufName, sourceMgr, sharedState, ts, includedFiles,
      [&](ModuleOp module) -> ASTDecl & {
        Lexer lexer(sharedState, sourceBuf);
        auto startSMLoc = lexer.getToken().getLoc();

        // Create the top-level outer decl, which will contain all things we
        // parse.
        ASTDecl &topLevelDecl = sharedState.declResolver->addDecl(
            module, startSMLoc, StringAttr(), /*parentDecl=*/nullptr,
            lexer.getCursor(), LexerCursor::getEOF(sourceBuf), -1);
        sharedState.initialize(topLevelDecl);

        // If we are emitting debug info, create a file entry for this file.
        if (sharedState.diBuilder)
          fileGuard = sharedState.diBuilder->pushFile(bufName, "/");

        // Grab a module name for the current input, choosing a dummy name if we
        // don't have one that's valid.
        std::string moduleName =
            std::filesystem::path(bufName.str()).stem().string();
        if (moduleName.empty())
          moduleName = "<input>";

        // Build the input module.
        return sharedState.createModule(moduleName, sourceBuf,
                                        cast<FileLineColLoc>(module->getLoc()));
      });
}

bool M::isMojoSourcePackagePath(const std::filesystem::path &path) {
  if (std::filesystem::is_directory(path)) {
    std::error_code ec;
    return std::filesystem::exists(path / "__init__.mojo", ec) ||
           std::filesystem::exists(path / "__init__.🔥", ec);
  }
  return false;
}

std::pair<OwningOpRef<ModuleOp>, PackageOp>
M::importMojoPackage(StringRef path, StringRef packageName,
                     llvm::SourceMgr &sourceMgr, MojoParserConfig &config,
                     mlir::TimingScope &ts,
                     SmallVectorImpl<std::string> *includedFiles) {
  // Emit an error if the path doesn't actually correspond with a package.
  if (!isMojoSourcePackagePath(path.str())) {
    sourceMgr.PrintMessage({}, llvm::SourceMgr::DK_Error,
                           "provided path '" + path +
                               "' does not correspond to a package");
    return {};
  }
  SharedState sharedState(sourceMgr, config);
  auto [module, packageDecl] = importMojoImpl(
      path, sourceMgr, sharedState, ts, includedFiles,
      [&](ModuleOp module) -> ASTDecl & {
        // Create the top-level outer decl, which will contain all things we
        // parse.
        ASTDecl &topLevelDecl = sharedState.declResolver->addDecl(
            module, SMLoc(), StringAttr(), /*parentDecl=*/nullptr,
            LexerCursor(), LexerCursor(), /*indentation=*/-1);
        sharedState.initialize(topLevelDecl);

        // Build the package.
        return sharedState.createPackage(path, packageName);
      });
  if (!module)
    return {};
  return {std::move(module), cast<PackageOp>(*packageDecl)};
}

OwningOpRef<mlir::ModuleOp>
M::importMojoFile(llvm::SourceMgr &sourceMgr, MojoParserConfig &config,
                  mlir::TimingScope &ts,
                  SmallVectorImpl<std::string> *includedFiles) {
  SharedState sharedState(sourceMgr, config);
  auto [module, topLevelDecl] =
      importMojoFileImpl(sourceMgr, sharedState, ts, includedFiles);
  return std::move(module);
}

MojoASTDeclRef MojoParserContext::parseFile(unsigned fileId) {
  llvm::SourceMgr &sourceMgr = getSourceMgr();

  const llvm::MemoryBuffer *sourceBuf = sourceMgr.getMemoryBuffer(fileId);

  StringRef filepath = sourceBuf->getBufferIdentifier();
  auto fileLoc = FileLineColLoc::get(impl->sharedState.getContext(), filepath,
                                     /*line=*/0, /*column=*/0);
  std::string moduleName =
      std::filesystem::path(filepath.data()).stem().string();
  ASTDecl &moduleDecl =
      impl->sharedState.createModule(moduleName, sourceBuf, fileLoc);
  impl->sharedState.declResolver->resolveAllReferencedFrom(moduleDecl);

  // Now that resolution is finished, cache the state of modules we have parsed.
  // TODO: We should be able to cache even in the presence of warnings and
  // errors. We can store the diagnostics and replay on cache load.
  if (!impl->sharedState.diags.isDiagnosticEmitted())
    impl->sharedState.cacheParsedModules();

  return MojoASTDeclRef(&moduleDecl);
}
