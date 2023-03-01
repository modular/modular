//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of the LitSharedState class.
//
//===----------------------------------------------------------------------===//

#include "LitSharedState.h"
#include "ASTDecl.h"
#include "ASTType.h"
#include "IRValues.h"

#include "KGEN/CompilationOptions.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LitDecls.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/HLCFDialect/HLCFDialect.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include <filesystem>

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

using llvm::SMLoc;
using llvm::SourceMgr;

static void adjustTokenEndPoint(LitSharedState &shared, SMLoc &loc);

/// Return the path containing the standard library. Returns nullopt if the
/// standard library cannot be found.
static std::optional<std::string> getStandardLibraryPath() {
  // TODO: Eventually we should resolve the standard library path to an actual
  // install of lit, for now though try to resolve the standard library path
  // within modular.

  // Check if we already have the path set.
  if (auto envDir = llvm::sys::Process::GetEnv("MODULAR_PATH"))
    return (std::filesystem::path(*envDir) / "Kernels" / "lit-stdlib").string();

  // Otherwise, try to find modular relative to the current directory.
  std::filesystem::path path = std::filesystem::current_path();
  while (!path.empty()) {
    if (path.stem() == "modular")
      return (path / "Kernels" / "lit-stdlib").string();
    path = path.parent_path();
  }
  return std::nullopt;
}

class LitSharedState::Impl {
public:
  SymbolTableCollection symbolTables;

  /// The path of the standard library, or nullopt if it is not available.
  std::optional<std::string> stdlibPath;

  /// The top-level decl containing everything being parsed.
  ASTDecl *topLevelDecl = nullptr;

  /// This is the AST type that corresponds to TypeCheckErrorType.
  ASTType typeCheckErrorType;
  /// This is the decl for the builtin 'kgen.none' type.
  ASTType noneType;

  /// The current set of imported modules.
  DenseMap<StringAttr, ASTDecl *> importedModules;
  /// A list of included files used when importing modules. These are used to
  /// generate dependency files.
  SmallVector<std::string> includedFiles;
};

LitSharedState::LitSharedState(llvm::SourceMgr &sourceMgr, MLIRContext *context,
                               const CompilationOptions &options,
                               bool useMLIRDiagnostics)
    : diags(sourceMgr, context, useMLIRDiagnostics), options(options),
      declResolver(std::make_unique<DeclResolver>(*this)),
      impl(std::make_unique<Impl>()) {
  impl->stdlibPath = getStandardLibraryPath();

  context->loadDialect<DebugInfo::DebugInfoDialect, HLCF::HLCFDialect,
                       POP::POPDialect, LITDialect, mlir::index::IndexDialect,
                       KGENDialect>();

  // Tell the diagnostics machinery how to find the end of a token lazily when
  // it needs it.
  diags.setTokenEndPointAdjustmentFn(
      [=](SMLoc &loc) { adjustTokenEndPoint(*this, loc); });

  if (options.getDebugInfoLevelForInput()) {
    diBuilder = std::make_unique<DebugInfo::DIBuilder>(context);

    // TODO: Dwarf technically has a language for python, but it's not really
    // what we want here AFAICT (our compilation model isn't the same as
    // python's). Figure out what we actually want here (though C works well
    // enough for now).
    diBuilder->initializeCompileUnit(
        llvm::dwarf::DW_LANG_C,
        diBuilder->createFile(diags.getBufferNameIdentifier(), "/"), "Lit",
        /*isOptimized=*/true, options.getDIEmissionKind());
  }
}

LitSharedState::~LitSharedState() { declResolver.reset(); }

void LitSharedState::initialize(ASTDecl &topLevelDecl) {
  assert(!impl->topLevelDecl && "already initialized");
  impl->topLevelDecl = &topLevelDecl;

  // Build the builtins decl.
  // TODO: Add these:
  // https://docs.python.org/3/library/functions.html#built-in-funcs
  // https://docs.python.org/3/reference/executionmodel.html#naming-and-binding
  ASTDecl &builtinsDecl = declResolver->addDecl(
      topLevelDecl.getIfOperation(), topLevelDecl.getLoc(), StringAttr(),
      nullptr, topLevelDecl.getCursor(), topLevelDecl.getCursor(), -1);
  addBuiltinTypes(builtinsDecl);
  builtinsDecl.resolvedness = DeclResolvedness::fully;

  // The outermost scope contains all of the __builtins__ function definitions.
  for (auto &[name, decls] : builtinsDecl.declsInScope)
    declResolver->aliasDecls(decls, name, topLevelDecl.getLoc(), topLevelDecl);

  // Top level is fully resolved now.
  topLevelDecl.resolvedness = DeclResolvedness::fully;
}

LitDiagnostic LitSharedState::emitError(Location loc, const Twine &message) {
  return diags.emitError(loc, message);
}

/// Emit an error through the parser's logic.
LitDiagnostic LitSharedState::emitError(llvm::SMLoc loc, const Twine &message) {
  return diags.emitError(loc, message);
}

/// Emit a warning.
LitDiagnostic LitSharedState::emitWarning(Location loc, const Twine &message) {
  return diags.emitWarning(loc, message);
}
LitDiagnostic LitSharedState::emitWarning(llvm::SMLoc loc,
                                          const Twine &message) {
  return diags.emitWarning(loc, message);
}

/// Inflate a lightweight SMLoc into an MLIR Location object for addition
/// into the IR.
Location LitSharedState::translateLocation(llvm::SMLoc loc) const {
  auto fileLoc = diags.translateLocation(loc);
  return diBuilder ? diBuilder->createScopedLoc(fileLoc) : fileLoc;
}

ASTType LitSharedState::getTypeCheckErrorType() const {
  return impl->typeCheckErrorType;
}
ASTType LitSharedState::getNoneType() const { return impl->noneType; }

/// Add declarations for magic things to the builtins decl.
void LitSharedState::addBuiltinTypes(ASTDecl &builtinsDecl) {
  DeclResolver &resolver = *declResolver;
  MLIRContext *context = getContext();

  // Add a declarations for builtin types.
  impl->noneType = LIT::NoneType::get(context);

  // Make the type check error type.  Anything that references this will
  // considering it erroneous and already declared as such.
  impl->typeCheckErrorType = TypeCheckErrorType::get(context);

  // Add an empty struct with the specified name to the resolver.
  auto addMagicMLIRDecl = [&](StringRef name, Type magicType) {
    TypedAttr value = ConcreteTypeConstantAttr::get(magicType);
    resolver.addFullyResolvedDecl(PRValue(value), name, builtinsDecl.getLoc(),
                                  &builtinsDecl);
  };

  addMagicMLIRDecl("__mlir_attr", MagicMLIRAttrType::get(context));
  addMagicMLIRDecl("__mlir_op", MagicMLIROpType::get(context));
  addMagicMLIRDecl("__mlir_type", MagicMLIRTypeType::get(context));
}

/// Set the symbol for the specified declaration (known to be an operation)
/// into the MLIR symbol table for its container.  If the symbol is already
/// declared in the same MLIR scope, then return the conflicting operation.
Operation *LitSharedState::setResolvedDeclSymbol(Operation *declOp) {
  assert(declOp && "Cannot set a symbol for non-operation decl");

  // We look up the symbol in the enclosing symbol table.  For example, for a
  // method in a struct, we use the struct as the symbol table.  For atop-level
  // function we use the global module.
  Operation *parentSymbolTableOp =
      SymbolTable::getNearestSymbolTable(declOp->getParentOp());
  SymbolTable &symTab = impl->symbolTables.getSymbolTable(parentSymbolTableOp);

  // Insert the operation into the symbol table and see if it got renamed.
  // Restore the original position of the operation after.
  auto origName = SymbolTable::getSymbolName(declOp);
  Block *prevBlock = declOp->getBlock();
  Block::iterator prevPos = std::next(declOp->getIterator());
  declOp->remove();
  auto resetPos =
      llvm::make_scope_exit([&] { declOp->moveBefore(prevBlock, prevPos); });
  if (symTab.insert(declOp) == origName)
    return nullptr; // No conflict, done.

  return symTab.lookup(origName);
}

//===----------------------------------------------------------------------===//
// Name Lookup
//===----------------------------------------------------------------------===//

/// Perform a name lookup in the specified scope and return the named
/// declaration as a LookupResult.
auto LitSharedState::lookupAndResolveDecl(StringRef name, SMLoc loc,
                                          ASTDecl &scope,
                                          bool searchParentScopes)
    -> LookupResult {

  // Ensure the context is fully resolved, so all its members are known.  It
  // would be bad to look something up in a scope without all members known.
  // FIXME(Issue#5975): FuncOp shouldn't be special cased.
  if (!isa<FuncOp>(scope)) {
    if (failed(declResolver->resolveFully(scope, loc)))
      return LookupResult::getErroneous();
  }

  auto nameAttr = StringAttr::get(getContext(), name);

  // Look up the name.
  auto lookupInScope = [&](ASTDecl &scope) -> const TinyPtrVector<ASTDecl *> * {
    // Check if we already have a declaration for this name in the current
    // scope.
    if (auto *result = scope.lookupInCurrentScope(nameAttr))
      return result;

    // If the lookup failed, try to resolve any wildcard imports in the scope.
    // Don't try wildcard imports if we wouldn't import this name anyways.
    if (name.startswith("_"))
      return nullptr;

    // We don't know if these imports will actually provide the decl we are
    // looking for, so we have to try until we find one that does.
    while (!scope.unresolvedWildcardImports.empty()) {
      auto it = scope.unresolvedWildcardImports.begin();
      auto [moduleName, loc] = *it;
      scope.unresolvedWildcardImports.erase(it);

      // Resolve the import. If it fails, don't fail the search immediately,
      // keep checking for something that can resolve the decl we care about.
      if (failed(declResolver->importWildCardDeclsFromModule(scope, moduleName,
                                                             loc)))
        continue;
      // Re-check the lookup in the scope now that the wildcard import has
      // been resolved.
      if (auto *result = scope.lookupInCurrentScope(nameAttr))
        return result;
    }

    return nullptr;
  };

  auto getEntry = [&]() {
    const TinyPtrVector<ASTDecl *> *e;
    if (searchParentScopes) {
      ASTDecl *curSearchScope = &scope;
      do {
        if ((e = lookupInScope(*curSearchScope)))
          break;
      } while ((curSearchScope = curSearchScope->parentDecl));
    } else {
      e = lookupInScope(scope);
    }
    return e;
  };

  const TinyPtrVector<ASTDecl *> *entry = getEntry();

  // If nothing was found, return a failure.
  if (!entry)
    return LookupResult::getFailure();

  // If the lookup succeeded, make sure the signature for the referenced decls
  // are understood.
  for (auto *decl : *entry) {
    if (failed(
            declResolver->resolve(*decl, DeclResolvedness::signature, loc))) {
      // If the decl was erroneous somehow, then don't form a reference to it,
      // the error has already been diagnosed.
      return LookupResult::getErroneous();
    }
  }
  // Get again the entry pointer since it might have been invalidated by
  // declResolver->resolve above.
  entry = getEntry();
  // If we are resolving an unresolved import, do another lookup now that import
  // has been resolved. The scope map should be updated with the proper decls.
  if (!entry->empty() && isa<UnresolvedImportOp>(*entry->front()))
    return lookupAndResolveDecl(name, loc, scope, searchParentScopes);

  // We return a pointer into the TinyPtrVector entry in the scope.  This should
  // be stable because you can't perform a lookup into a decl that has unknown
  // entries, and we just resolved all the signatures for all the decls.
  return LookupResult::getSuccess(*entry);
}

/// Perform a name lookup for a member in the specified type.
auto LitSharedState::lookupAndResolveDecl(StringRef name, SMLoc loc,
                                          ASTType scope,
                                          bool searchParentScopes)
    -> LookupResult {
  if (auto *decl = scope.getDecl(*this))
    return lookupAndResolveDecl(name, loc, *decl, searchParentScopes);
  return LookupResult::getFailure();
}

ASTType LitSharedState::lookupNonparameterizedNamedType(StringRef name,
                                                        llvm::SMLoc loc,
                                                        ASTDecl &context) {
  LookupResult result =
      lookupAndResolveDecl(name, loc, context, /*searchParentScopes=*/true);
  if (result.isErroneous())
    return {};
  if (result.isFailure()) {
    emitError(loc, "could not find an '") << name << "' type";
    return {};
  }
  // The overload set may contain multiple entries, but if it is a struct, it
  // must be a single entry and therefore we can just check that one.
  ASTDecl &firstDecl = *result.getIfSuccess()[0];
  auto structOp = dyn_cast<StructDeclOp>(firstDecl);
  if (!structOp) {
    auto diag = emitError(loc, "'") << name << "' doesn't resolve to a type";
    diag.attachNote(firstDecl.getLoc()) << "'" << name << "' declared here";
    return {};
  }
  if (!structOp.getInputParamDecls().empty()) {
    auto diag = emitError(loc, "'")
                << name << "' resolves to a parameterized type";
    diag.attachNote(firstDecl.getLoc()) << "'" << name << "' declared here";
    return {};
  }
  return firstDecl.getSelfType();
}

/// Lookup the `object` type in the specified context and return it if found,
/// otherwise emit an error and return null.
ASTType LitSharedState::lookupObjectType(llvm::SMLoc loc, ASTDecl &context) {
  return lookupNonparameterizedNamedType("object", loc, context);
}

/// Lookup the `Error` type in the current context and return it if found,
/// otherwise emit an error and return null.
ASTType LitSharedState::lookupErrorType(SMLoc loc, ASTDecl &context) {
  return lookupNonparameterizedNamedType("Error", loc, context);
}

/// Resolve the absolute path for a given module name. Returns nullopt if the
/// module cannot be found.
static std::optional<std::string>
resolveModulePath(StringRef moduleName, const Optional<std::string> &stdLibDir,
                  llvm::SourceMgr &sourceMgr, llvm::SMLoc includeLoc) {
  // Python has lots of magic rules surrounding how modules get resolved. For
  // now, we just use the available include directories within the source
  // manager and the working directory of where the module is included.
  auto checkPath = [&](StringRef includeDir) -> std::optional<std::string> {
    std::string path = (Twine(includeDir) + "/" + moduleName + ".lit").str();
    if (std::filesystem::exists(path))
      return path;
    return std::nullopt;
  };

  // Check the standard library first.
  if (stdLibDir) {
    if (auto path = checkPath(*stdLibDir))
      return path;
  }

  // Check the working directory.
  const llvm::MemoryBuffer *includeBuffer =
      sourceMgr.getMemoryBuffer(sourceMgr.FindBufferContainingLoc(includeLoc));
  assert(includeBuffer && "must be in a source buffer");
  auto includerPath =
      std::filesystem::path(includeBuffer->getBufferIdentifier().str());
  if (auto path = checkPath(includerPath.parent_path().string()))
    return path;

  // Then check the include directories.
  for (StringRef includeDir : sourceMgr.getIncludeDirs())
    if (auto path = checkPath(includeDir))
      return path;
  return std::nullopt;
}

/// Return a mangled version of the given module name. This is used to avoid
/// conflicts with symbols that are actually visible.
static StringAttr getMangledModuleName(MLIRContext *ctx, StringRef moduleName) {
  return StringAttr::get(ctx, "$" + moduleName);
}

ASTDecl &LitSharedState::importModule(StringRef moduleName, llvm::SMLoc loc) {
  // Mangle the module name during import to avoid conflicts with symbols that
  // are actually visible. We may import a module, but not directly expose it
  // via its module name.
  auto mangledName = getMangledModuleName(getContext(), moduleName);

  // Check to see if we've already imported this module.
  if (auto *existingDecl = impl->importedModules.lookup(mangledName))
    return *existingDecl;
  auto moduleBuilder = impl->topLevelDecl->getDeclEndBuilder();

  // Resolve the path for this module.
  std::optional<std::string> modulePath =
      resolveModulePath(moduleName, impl->stdlibPath, getSourceMgr(), loc);
  if (!modulePath) {
    emitError(loc, "unable to locate module '") << moduleName << "'";

    // Don't bail if we can't find the module, create a dummy decl so that we
    // can have better error recorvery/messages.
    ASTDecl &moduleDecl =
        declResolver->addErroneousDecl(mangledName, loc, impl->topLevelDecl);
    impl->importedModules.try_emplace(mangledName, &moduleDecl);
    return moduleDecl;
  }

  // Open the module file within the source manager.
  std::string fullPath;
  unsigned fileID = getSourceMgr().AddIncludeFile(*modulePath, loc, fullPath);
  impl->includedFiles.push_back(fullPath);

  // Now that we have a MemoryBuffer, we can lex it, and therefore parse it.
  // do so.
  const llvm::MemoryBuffer *moduleBuffer =
      getSourceMgr().getMemoryBuffer(fileID);
  auto fileLoc = moduleBuilder.getAttr<FileLineColLoc>(fullPath, /*line=*/0,
                                                       /*column=*/0);
  return createModule(moduleName, moduleBuffer, fileLoc);
}

ASTDecl &LitSharedState::getCompilerBuiltInDecl() {
  StringAttr builtinStrAttr =
      getMangledModuleName(getContext(), kCompilerBuiltInStr);
  ASTDecl *entry = impl->importedModules.lookup(builtinStrAttr);
  assert(entry && "_CompilerBuiltin must exist");
  return *entry;
}

ASTDecl &LitSharedState::createModule(StringRef moduleName,
                                      const llvm::MemoryBuffer *moduleBuffer,
                                      FileLineColLoc loc) {
  StringAttr mangledName = getMangledModuleName(getContext(), moduleName);
  LitLexer lexer(*this, moduleBuffer);
  LitLexerCursor endCursor(
      {LitToken::eof, StringRef(moduleBuffer->getBufferEnd() + 1, 0), 0});

  // Create a new decl for this module.
  auto moduleBuilder = impl->topLevelDecl->getDeclEndBuilder();
  Operation *fileOp = moduleBuilder.create<FileModuleOp>(loc, mangledName);
  ASTDecl &moduleDecl = declResolver->addDecl(
      fileOp, lexer.getToken().getLoc(), mangledName, impl->topLevelDecl,
      lexer.getCursor(), endCursor, /*indentation=*/-1);
  // Auto-import the core Lang module declaration.
  moduleDecl.addUnresolvedWildCardImport(
      StringAttr::get(getContext(), kCompilerBuiltInStr),
      lexer.getToken().getLoc());
  impl->importedModules.try_emplace(mangledName, &moduleDecl);
  return moduleDecl;
}

ArrayRef<std::string> LitSharedState::getIncludedFiles() const {
  return impl->includedFiles;
}

/// Given a pointer to the start of a token, find the end of it.
static void adjustTokenEndPoint(LitSharedState &shared, SMLoc &loc) {
  size_t tokenSize = LitLexer::getTokenLength(shared, loc);
  loc = SMLoc::getFromPointer(loc.getPointer() + tokenSize);
}
