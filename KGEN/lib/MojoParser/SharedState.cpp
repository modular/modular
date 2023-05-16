//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of the SharedState class.
//
//===----------------------------------------------------------------------===//

#include "SharedState.h"
#include "ASTDecl.h"
#include "ASTType.h"
#include "DeclResolver.h"
#include "IRValues.h"

#include "Cache/Buffer.h"
#include "Cache/CacheDialect/CachedTransform.h"
#include "KGEN/CompilationOptions.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LLCL/Runtime/Algorithms.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include <filesystem>

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

using llvm::SMLoc;
using llvm::SourceMgr;

static void adjustTokenEndPoint(SharedState &shared, SMLoc &loc);

/// Return the path containing the standard library. Returns nullopt if the
/// standard library cannot be found.
static void getAutoImportPaths(SmallVector<std::string> &paths) {
  // Check if we already have the path set.
  if (auto envDir = llvm::sys::Process::GetEnv("MODULAR_PATH"))
    paths.push_back(
        (std::filesystem::path(*envDir) / "Kernels" / "mojo").string());

  if (auto envDir = llvm::sys::Process::GetEnv("MODULAR_DERIVED_PATH"))
    paths.push_back(
        (std::filesystem::path(*envDir) / "build" / "Kernels" / "mojo")
            .string());

  // If a path was specified via envvar, we're done here.
  if (!paths.empty())
    return;

  // Otherwise, try to find modular relative to the current directory.
  std::filesystem::path path = std::filesystem::current_path();
  while (!path.empty()) {
    if (path.stem() == "modular") {
      paths = {(path / "Kernels" / "mojo").string(),
               (path / ".derived" / "build" / "Kernels" / "mojo").string()};
      return;
    }
    if (!path.has_parent_path())
      break;
    path = path.parent_path();
  }
}

struct SharedState::Impl {
  SymbolTableCollection symbolTables;

  /// A map of symbol tables to unique counters for names within those
  /// symbol tables.
  DenseMap<std::pair<SymbolTable *, StringAttr>, unsigned> symbolTableCounters;

  /// The auto import path (e.g. path to the stdlib), or nullopt if it is not
  /// available.
  SmallVector<std::string> autoImportDirs;

  /// The top-level decl containing everything being parsed.
  ASTDecl *topLevelDecl = nullptr;

  /// This is the AST type that corresponds to TypeCheckErrorType.
  ASTType typeCheckErrorType;
  /// This is the decl for the builtin 'kgen.none' type.
  ASTType noneType;
  NoneAttr noneAttr;

  /// The current set of imported modules.
  llvm::MapVector<StringAttr, std::unique_ptr<ModuleState>> importedModules;
  /// A list of included files used when importing modules. These are used to
  /// generate dependency files.
  SmallVector<std::string> includedFiles;

  /// The cache used to store cached transformations within the parser.
  LLCL::RCRef<Cache::BlobCache<Cache::TransformCacheKey>> transformCache;

  /// The shared operation dominance analysis
  mlir::DominanceInfo domInfo;

  /// Flag indicating if the deps of a module are currently being resolved.
  bool activelyResolvingModuleDeps = false;

  /// Flag indicating if we should validate doc strings while parsing.
  bool validateDocStrings = false;

  /// This keeps track of body decorators for a given declaration, this is
  /// logically part of ASTDecl, but is stored out of line to reduce its size
  /// since these are uncommon.
  DenseMap<const ASTDecl *, std::vector<LexerCursor>> bodyDecorators;
};

SharedState::SharedState(llvm::SourceMgr &sourceMgr, MojoParserConfig &config,
                         bool enableCaching)
    : diags(sourceMgr, config.context, config.useMLIRDiagnostics),
      options(config.options),
      declResolver(std::make_unique<DeclResolver>(*this)),
      runtime(config.runtime), impl(std::make_unique<Impl>()) {
  getAutoImportPaths(impl->autoImportDirs);
  impl->validateDocStrings = config.validateDocStrings;

  config.context->loadDialect<DebugInfo::DebugInfoDialect, HLCF::HLCFDialect,
                              POP::POPDialect, LITDialect,
                              mlir::index::IndexDialect, KGENDialect>();

  // Tell the diagnostics machinery how to find the end of a token lazily when
  // it needs it.
  diags.setTokenEndPointAdjustmentFn(
      [=](SMLoc &loc) { adjustTokenEndPoint(*this, loc); });

  if (options.getDebugInfoLevelForInput()) {
    diBuilder = std::make_unique<DebugInfo::DIBuilder>(config.context);

    // TODO: Dwarf technically has a language for python, but it's not really
    // what we want here AFAICT (our compilation model isn't the same as
    // python's). Figure out what we actually want here (though C works well
    // enough for now).
    diBuilder->initializeCompileUnit(
        llvm::dwarf::DW_LANG_C,
        diBuilder->createFile(diags.getBufferNameIdentifier(), "/"), "Mojo",
        /*isOptimized=*/true, options.getDIEmissionKind());
  }

  // Create a cache for use by the parser.
  if (enableCaching) {
    auto transformCacheBackendOr = Cache::getLocalDefaultBackendChain(
        runtime, ".kgen_cache/lit", KGEN_VERSION_STRING);
    if (failed(transformCacheBackendOr))
      return;
    impl->transformCache =
        LLCL::RCRef<Cache::BlobCache<Cache::TransformCacheKey>>::create(
            transformCacheBackendOr.takeValue());
  }
}

SharedState::~SharedState() { declResolver.reset(); }

bool SharedState::shouldValidateDocStrings() const {
  return impl->validateDocStrings;
}

void SharedState::initialize(ASTDecl &topLevelDecl) {
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

InflightDiag SharedState::emitError(Location loc, const Twine &message) {
  return diags.emitError(loc, message);
}

/// Emit an error through the parser's logic.
InflightDiag SharedState::emitError(llvm::SMLoc loc, const Twine &message) {
  return diags.emitError(loc, message);
}

/// Emit a warning.
InflightDiag SharedState::emitWarning(Location loc, const Twine &message) {
  return diags.emitWarning(loc, message);
}
InflightDiag SharedState::emitWarning(llvm::SMLoc loc, const Twine &message) {
  return diags.emitWarning(loc, message);
}

/// Inflate a lightweight SMLoc into an MLIR Location object for addition
/// into the IR.
Location SharedState::translateLocation(llvm::SMLoc loc) const {
  auto fileLoc = diags.translateLocation(loc);
  return diBuilder ? diBuilder->createScopedLoc(fileLoc) : fileLoc;
}

ASTType SharedState::getTypeCheckErrorType() const {
  return impl->typeCheckErrorType;
}
ASTType SharedState::getNoneType() const { return impl->noneType; }
NoneAttr SharedState::getNoneAttr() const { return impl->noneAttr; }

/// Add declarations for magic things to the builtins decl.
void SharedState::addBuiltinTypes(ASTDecl &builtinsDecl) {
  DeclResolver &resolver = *declResolver;
  MLIRContext *context = getContext();

  // Add a declarations for builtin types.
  impl->noneType = LIT::NoneType::get(context);

  impl->noneAttr = NoneAttr::get(context, impl->noneType.mlirType);

  // Make the type check error type.  Anything that references this will
  // considering it erroneous and already declared as such.
  impl->typeCheckErrorType = TypeCheckErrorType::get(context);

  // Add an empty struct with the specified name to the resolver.
  auto addMagicMLIRDecl = [&](StringRef name, Type magicType) {
    TypedAttr value = TypeConstantAttr::get(magicType);
    resolver.addFullyResolvedDecl(PValue(value), name, builtinsDecl.getLoc(),
                                  &builtinsDecl);
  };

  addMagicMLIRDecl("__mlir_attr", MagicMLIRAttrType::get(context));
  addMagicMLIRDecl("__mlir_op", MagicMLIROpType::get(context));
  addMagicMLIRDecl("__mlir_type", MagicMLIRTypeType::get(context));
}

/// Set the symbol for the specified declaration (known to be an operation)
/// into the MLIR symbol table for its container.  If the symbol is already
/// declared in the same MLIR scope, then return the conflicting operation.
Operation *SharedState::setResolvedDeclSymbol(Operation *declOp) {
  assert(declOp && "Cannot set a symbol for non-operation decl");

  // We look up the symbol in the enclosing symbol table.  For example, for a
  // method in a struct, we use the struct as the symbol table.  For atop-level
  // function we use the global module.
  Operation *parentSymbolTableOp =
      SymbolTable::getNearestSymbolTable(declOp->getParentOp());
  SymbolTable &symTab = impl->symbolTables.getSymbolTable(parentSymbolTableOp);

  // Insert the operation into the symbol table and see if it got renamed.
  // Restore the original position of the operation after.
  Block *prevBlock = declOp->getBlock();
  Block::iterator prevPos = std::next(declOp->getIterator());
  declOp->remove();
  auto resetPos =
      llvm::make_scope_exit([&] { declOp->moveBefore(prevBlock, prevPos); });

  StringAttr origName = SymbolTable::getSymbolName(declOp);
  Operation *existingOp = symTab.lookup(origName);
  if (existingOp && existingOp != declOp) {
    unsigned &counter = impl->symbolTableCounters[{&symTab, origName}];
    SymbolTable::setSymbolName(
        declOp, getUniqueSymbolName(origName.str(), symTab, counter));
  } else {
    existingOp = nullptr;
  }

  auto newName = symTab.insert(declOp);
  assert(newName == SymbolTable::getSymbolName(declOp) &&
         "symbol table insertion changed the name");
  return existingOp;
}

mlir::DominanceInfo &SharedState::getDomInfo() { return impl->domInfo; }

//===----------------------------------------------------------------------===//
// ASTDecl
//===----------------------------------------------------------------------===//

/// Return any decorators that need to be processed as part of body resolution
/// phase for a decl.
ArrayRef<LexerCursor> ASTDecl::getBodyDecorators(SharedState &state) const {
  if (!hasBodyDecorators)
    return {};
  return state.getImpl().bodyDecorators[this];
}

/// During signature resolution, this is called with any decorators that need
/// to persist until body resolution.
void ASTDecl::setBodyDecorators(ArrayRef<LexerCursor> decorators,
                                SharedState &state) {
  if (decorators.empty())
    return;

  state.getImpl().bodyDecorators.insert({this, decorators.vec()});
  hasBodyDecorators = true;
}

//===----------------------------------------------------------------------===//
// ModuleState
//===----------------------------------------------------------------------===//

struct SharedState::ModuleState {
  ModuleState(ASTDecl *decl = nullptr) : decl(decl) {}

  /// Build the cache key for this module.
  Cache::WriteableBufferRef buildCacheKey(const CompilationOptions &options) {
    auto keyBuf = Cache::WriteableBuffer::get();

    // Add the module contents to the cache key.
    keyBuf->write((const char *)contentHash.data(), contentHash.size());

    // Add the module dependencies to the cache key.
    for (auto *dep : dependencies) {
      keyBuf->write((const char *)dep->contentHash.data(),
                    dep->contentHash.size());
    }

    // Add the compilation options to the cache key.
    options.print(*keyBuf);
    return keyBuf;
  }

  /// The decl associated with the module.
  ASTDecl *decl = nullptr;
  /// A hash associated with the modules contents.
  llvm::BLAKE3Result<> contentHash;
  /// The set of other modules that this module depends on.
  llvm::SmallSetVector<ModuleState *, 4> dependencies;
  /// A flag indicating if this module was imported from the cache.
  bool importedFromCache = false;
};

//===----------------------------------------------------------------------===//
// Name Lookup
//===----------------------------------------------------------------------===//

/// Perform a name lookup in the specified scope and return the named
/// declaration as a LookupResult.
auto SharedState::lookupAndResolveDecl(StringRef name, SMLoc loc,
                                       ASTDecl &scope, bool searchParentScopes)
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
  auto lookupInScope = [&](ASTDecl &scope) -> ArrayRef<ASTDecl *> {
    // Check if we already have a declaration for this name in the current
    // scope.
    auto result = scope.lookupInCurrentScope(nameAttr);
    if (!result.empty())
      return result;

    // If the lookup failed, try to resolve any wildcard imports in the scope.
    // Don't try wildcard imports if we wouldn't import this name anyways.
    if (name.startswith("_"))
      return {};

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
      result = scope.lookupInCurrentScope(nameAttr);
      if (!result.empty())
        return result;
    }

    return {};
  };

  auto getEntry = [&]() -> ArrayRef<ASTDecl *> {
    if (!searchParentScopes)
      return lookupInScope(scope);

    ASTDecl *curSearchScope = &scope;
    do {
      ArrayRef<ASTDecl *> e = lookupInScope(*curSearchScope);
      if (!e.empty())
        return e;
    } while ((curSearchScope = curSearchScope->parentDecl));
    return {};
  };

  ArrayRef<ASTDecl *> entry = getEntry();

  // If nothing was found, return a failure.
  if (entry.empty())
    return LookupResult::getFailure();

  // If the lookup succeeded, make sure the signature for the referenced decls
  // are understood.
  for (auto *decl : entry) {
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
  if (!entry.empty() && isa<UnresolvedImportOp>(*entry.front()))
    return lookupAndResolveDecl(name, loc, scope, searchParentScopes);

  // We return a pointer into the TinyPtrVector entry in the scope.  This should
  // be stable because you can't perform a lookup into a decl that has unknown
  // entries, and we just resolved all the signatures for all the decls.
  return LookupResult::getSuccess(entry);
}

/// Perform a name lookup for a member in the specified type.
auto SharedState::lookupAndResolveDecl(StringRef name, SMLoc loc, ASTType scope,
                                       bool searchParentScopes)
    -> LookupResult {
  if (auto *decl = scope.getDecl(*this))
    return lookupAndResolveDecl(name, loc, *decl, searchParentScopes);
  return LookupResult::getFailure();
}

ASTType SharedState::lookupNonparameterizedNamedType(StringRef name,
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
  if (!structOp.getInputParams().empty()) {
    auto diag = emitError(loc, "'")
                << name << "' resolves to a parameterized type";
    diag.attachNote(firstDecl.getLoc()) << "'" << name << "' declared here";
    return {};
  }
  return firstDecl.getSelfType();
}

/// Lookup the `object` type in the specified context and return it if found,
/// otherwise emit an error and return null.
ASTType SharedState::lookupObjectType(llvm::SMLoc loc, ASTDecl &context) {
  return lookupNonparameterizedNamedType("object", loc, context);
}

/// Resolve the absolute path for a given module name. Returns nullopt if the
/// module cannot be found.
static std::optional<std::string>
resolveModulePath(StringRef moduleName,
                  const SmallVector<std::string> &autoImportDirs,
                  llvm::SourceMgr &sourceMgr, llvm::SMLoc includeLoc) {
  // Python has lots of magic rules surrounding how modules get resolved. For
  // now, we just use the available include directories within the source
  // manager and the working directory of where the module is included.
  auto checkPath = [&](StringRef includeDir) -> std::optional<std::string> {
    std::string path = (Twine(includeDir) + "/" + moduleName + ".mojo").str();
    if (std::filesystem::exists(path))
      return path;
    return std::nullopt;
  };

  // Check the auto import directory first.
  for (auto &rawPath : autoImportDirs) {
    if (auto path = checkPath(rawPath))
      return path;
    // Cannot find the file, then check child directories of the auto import
    // directory.
    for (auto &childDir :
         std::filesystem::recursive_directory_iterator(rawPath))
      if (childDir.is_directory())
        if (auto path = checkPath(childDir.path().string()))
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

ASTDecl &SharedState::importModule(StringRef moduleName, llvm::SMLoc loc) {
  return *importModuleState(moduleName, loc).decl;
}

SharedState::ModuleState &SharedState::importModuleState(StringRef moduleName,
                                                         llvm::SMLoc loc) {
  TimeTraceScope<> fullTimeScope(("importModule: " + moduleName).str());

  // Mangle the module name during import to avoid conflicts with symbols that
  // are actually visible. We may import a module, but not directly expose it
  // via its module name.
  auto mangledName = getMangledModuleName(getContext(), moduleName);

  // Check to see if we've already imported this module.
  auto it = impl->importedModules.find(mangledName);
  if (it != impl->importedModules.end())
    return *it->second;
  auto moduleBuilder = impl->topLevelDecl->getDeclEndBuilder();

  // Resolve the path for this module.
  std::optional<std::string> modulePath =
      resolveModulePath(moduleName, impl->autoImportDirs, getSourceMgr(), loc);
  if (!modulePath) {
    emitError(loc, "unable to locate module '") << moduleName << "'";

    // Don't bail if we can't find the module, create a dummy decl so that we
    // can have better error recovery/messages.
    ASTDecl &moduleDecl =
        declResolver->addErroneousDecl(mangledName, loc, impl->topLevelDecl);
    impl->importedModules.insert(
        {mangledName, std::make_unique<ModuleState>(&moduleDecl)});
    return *impl->importedModules.back().second;
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
  return createModuleState(moduleName, moduleBuffer, fileLoc);
}

static ASTType resolveBuiltinModuleType(llvm::SMLoc loc, StringRef moduleName,
                                        StringRef typeName,
                                        SharedState &shared) {
  StringAttr moduleStrAttr =
      getMangledModuleName(shared.getContext(), moduleName);
  auto &impl = shared.getImpl();
  auto it = impl.importedModules.find(moduleStrAttr);
  if (it != impl.importedModules.end()) {
    ASTDecl &moduleDecl = *it->second->decl;
    LookupResult lookup =
        shared.lookupAndResolveDecl(typeName, loc, moduleDecl,
                                    /*searchParentScopes=*/false);
    if (!lookup.isFailure() && !lookup.getIfSuccess().empty())
      return lookup.getIfSuccess()[0]->getSelfType();
  }

  shared.emitError(loc, "internal error: could not find builtin '")
      << typeName << "' type";
  return {};
}

ASTType SharedState::getBuiltinBoolType(llvm::SMLoc loc) {
  return resolveBuiltinModuleType(loc, kBuiltinBoolModuleName, "Bool", *this);
}

ASTType SharedState::getBuiltinTupleLiteralType(llvm::SMLoc loc) {
  return resolveBuiltinModuleType(loc, kBuiltinTupleModuleName, "TupleLiteral",
                                  *this);
}

ASTType SharedState::getBuiltinErrorType(llvm::SMLoc loc) {
  return resolveBuiltinModuleType(loc, kBuiltinErrorModuleName, "Error", *this);
}

ASTType SharedState::getBuiltinIntType(llvm::SMLoc loc) {
  return resolveBuiltinModuleType(loc, kBuiltinIntModuleName, "Int", *this);
}

ASTType SharedState::getBuiltinStringLiteralType(llvm::SMLoc loc) {
  return resolveBuiltinModuleType(loc, kBuiltinStringModuleName,
                                  "StringLiteral", *this);
}

ASTType SharedState::getBuiltinSliceType(llvm::SMLoc loc) {
  return resolveBuiltinModuleType(loc, kBuiltinSliceModuleName, "slice", *this);
}

ASTType SharedState::getBuiltinListLiteralType(llvm::SMLoc loc) {
  return resolveBuiltinModuleType(loc, kBuiltinListModuleName, "ListLiteral",
                                  *this);
}

ASTType SharedState::getBuiltinDoubleType(llvm::SMLoc loc) {
  return resolveBuiltinModuleType(loc, kBuiltinDoubleModuleName, "FloatLiteral",
                                  *this);
}

void SharedState::loadModulesFromCache(
    MutableArrayRef<ModuleState *> moduleStates) {
  // If we don't have a valid cache, we can't do anything.
  if (!impl->transformCache || moduleStates.empty())
    return;

  // Check the cache results for the various modules.
  SmallVector<LLCL::AnyAsyncValueRef> cacheResults;
  for (ModuleState *moduleState : moduleStates) {
    // If the module has already been resolved in any form, we shouldn't
    // try reading it from the cache.
    if (moduleState->decl->resolvedness > DeclResolvedness::unparsed)
      continue;
    Cache::WriteableBufferRef keyBuf = moduleState->buildCacheKey(options);

    auto out = AsyncValueRef<Chain>::allocate(runtime);
    auto f = impl->transformCache->find(
        std::move(keyBuf), LLCL::MLIRLocationDecoder::getEncodedLocation(
                               moduleState->decl->getIfOperation()->getLoc()));
    std::move(f).andThenSync(
        [moduleState, out = out.copy()](
            AsyncValueRef<std::optional<Cache::BufferRef>> &&f) mutable {
          // If the module isn't in the cache, process it as normal. We will
          // attempt to cache it later instead of now, given that we can't
          // reliably resolve everything in the module right now.
          if (f.isError())
            return std::move(out).setToError(f.takeDiagnostic());
          if (!f->has_value())
            return std::move(out).emplace();
          ASTDecl &moduleDecl = *moduleState->decl;
          FileModuleOp moduleOp = cast<FileModuleOp>(moduleDecl);
          TimeTraceScope<> fullTimeScope(
              ("loadModuleFromCache: " + moduleOp.getName()).str());

          // Read the cached IR.
          Block b;
          {
            TimeTraceScope<> timeScope("readBytecodeFile");
            std::unique_ptr<llvm::MemoryBuffer> bytecode =
                llvm::MemoryBuffer::getMemBuffer(
                    (**f)->getBuffer(), /*BufferName=*/"",
                    /*RequiresNullTerminator=*/false);
            mlir::ParserConfig config(moduleOp->getContext(),
                                      /*verifyAfterParse=*/false);

            // Read in the cached bytecode. If we fail, bail and try processing
            // the IR as normal.
            if (failed(mlir::readBytecodeFile(*bytecode, &b, config))) {
              return std::move(out).setToError(LLCL::getMLIRDiagnostic(
                  "failed to read module bytecode", moduleOp.getLoc()));
            }
          }

          // Replace the body of the module with the cached IR.
          FileModuleOp cachedModuleOp = cast<FileModuleOp>(b.front());
          moduleOp.getBodyRegion().takeBody(cachedModuleOp.getBodyRegion());

          // Mark the module as imported from cache.
          moduleState->importedFromCache = true;
          moduleDecl.resolvedness = DeclResolvedness::fully;
          std::move(out).emplace();
        });
    cacheResults.push_back(std::move(out));
  }
  LLCL::await(cacheResults);

  // Now that the cached IR has been read, create decl information for the
  // constructs within the cached IR.
  // TODO: When we can lazy load bytecode, we shouldn't build all of the decls
  // immediately. We can wait to load until the bodies of the decls are actually
  // needed.
  TimeTraceScope<> timeScope("Rebuild ASTDecls");
  SmallVector<ASTDecl *> declsToFill;

  // Populate the decls that we successfully loaded from the cache.
  for (ModuleState *moduleState : moduleStates) {
    // Throw it away if not all the dependencies are also in the cache.
    // TODO: See #10657 - we need to figure out a better way to deal with this,
    //   because this simply throws away work.
    if (moduleState->importedFromCache &&
        llvm::any_of(moduleState->dependencies,
                     [&](auto *state) { return !state->importedFromCache; })) {
      auto moduleOp = cast<FileModuleOp>(*moduleState->decl);
      // Clear out the blocks in this file module and replace with an empty
      // block. This avoids later symbol table issues.
      moduleOp.getBodyRegion().getBlocks().clear();
      moduleOp.getBodyRegion().push_back(new Block);
      moduleState->importedFromCache = false;
      moduleState->decl->resolvedness = DeclResolvedness::unparsed;
      continue;
    }

    if (moduleState->importedFromCache)
      declsToFill.push_back(moduleState->decl);
  }

  // Collect the referenced types that need to be resolved.
  mlir::AttrTypeWalker typeWalker;
  auto resolveTypes = [&](ASTDecl *context, TypeRange types) {
    for (Type type : types)
      typeWalker.walk(type);
  };
  auto resolveParams = [&](ASTDecl *context, ArrayRef<ParamDeclAttr> params) {
    for (ParamDeclAttr param : params)
      typeWalker.walk(param);
  };

  while (!declsToFill.empty()) {
    ASTDecl *container = declsToFill.pop_back_val();
    Operation *containerOp = container->getIfOperation();

    for (Region &region : containerOp->getRegions()) {
      for (Operation &op : region.getOps()) {

        auto addDeclForOp = [&](Operation *op, StringAttr name) -> ASTDecl & {
          ASTDecl &decl = declResolver->addFullyResolvedDecl(
              DeclIRValue(op), name, container->getLoc(), container);
          declsToFill.push_back(&decl);
          return decl;
        };

        if (auto funcOp = dyn_cast<LIT::FuncOp>(&op)) {
          StringRef baseFuncName = funcOp.getName().split('(').first;
          auto &decl = declResolver->addFullyResolvedDecl(
              DeclIRValue(funcOp), StringAttr::get(getContext(), baseFuncName),
              container->getLoc(), container);
          declResolver->declForFuncSymbol[decl.getSymbolRef()] = &decl;

          // Resolve the function types.
          resolveParams(&decl, funcOp.getInputParams());
          resolveParams(&decl, funcOp.getResultParams());
          resolveTypes(&decl, funcOp.getArgumentTypes());
          resolveTypes(&decl, funcOp.getResultTypes());

        } else if (auto importOp = dyn_cast<LIT::UnresolvedImportOp>(&op)) {
          declResolver->addDecl(importOp, container->getLoc(),
                                importOp.getImportNameAttr(), container,
                                container->getCursor(), container->getCursor(),
                                /*indentation=*/0);
        } else if (auto structOp = dyn_cast<StructDeclOp>(&op)) {
          ASTDecl &structDecl =
              addDeclForOp(structOp, structOp.getSymNameAttr());
          structDecl.setSelfType(structDecl.computeSelfTypeForStruct(*this));

          // Resolve the types of any parameters.
          resolveParams(&structDecl, structOp.getInputParams());
        } else if (auto structFieldOp = dyn_cast<StructFieldOp>(&op)) {
          addDeclForOp(structFieldOp, structFieldOp.getNameAttr());
        } else if (auto letOp = dyn_cast<LetRegDeclOp>(&op)) {
          addDeclForOp(letOp, letOp.getNameAttr());
        } else if (auto varOp = dyn_cast<VarLetDeclOp>(&op)) {
          addDeclForOp(varOp, varOp.getNameAttr());
        } else if (auto paramDeclareOp = dyn_cast<ParamDeclareOp>(&op)) {
          addDeclForOp(paramDeclareOp, paramDeclareOp.getName());
        } else if (auto aliasForwardDeclOp =
                       dyn_cast<AliasForwardDeclOp>(&op)) {
          addDeclForOp(aliasForwardDeclOp, aliasForwardDeclOp.getNameAttr());
        }
      }
    }
  }
}

ASTDecl &SharedState::createModule(StringRef moduleName,
                                   const llvm::MemoryBuffer *moduleBuffer,
                                   FileLineColLoc loc) {
  return *createModuleState(moduleName, moduleBuffer, loc).decl;
}

SharedState::ModuleState &
SharedState::createModuleState(StringRef moduleName,
                               const llvm::MemoryBuffer *moduleBuffer,
                               FileLineColLoc loc) {
  StringAttr mangledName = getMangledModuleName(getContext(), moduleName);
  Lexer lexer(*this, moduleBuffer);

  // Create a new decl for this module.
  auto moduleBuilder = impl->topLevelDecl->getDeclEndBuilder();
  Operation *fileOp = moduleBuilder.create<FileModuleOp>(loc, mangledName);
  ASTDecl &moduleDecl = declResolver->addDecl(
      fileOp, lexer.getToken().getLoc(), mangledName, impl->topLevelDecl,
      lexer.getCursor(), LexerCursor::getEOF(moduleBuffer), /*indentation=*/-1);

  // Auto-import the core Lang modules.
  for (StringRef moduleName : kBuiltinModuleNames) {
    moduleDecl.addUnresolvedWildCardImport(
        StringAttr::get(getContext(), moduleName), lexer.getToken().getLoc());
  }

  impl->importedModules.insert(
      {mangledName, std::make_unique<ModuleState>(&moduleDecl)});
  ModuleState &moduleState = *impl->importedModules.back().second;

  // Build a content hash for the module from its input buffer.
  llvm::BLAKE3 contentHash;
  contentHash.update(moduleBuffer->getBuffer());
  moduleState.contentHash = contentHash.final();

  // Resolve the dependencies of the module.
  size_t prevNumModules = impl->importedModules.size() - 1;
  resolveModuleDependencies(moduleState, moduleBuffer->getBuffer());

  // If we aren't currently resolving dependencies, try to load all of the newly
  // imported modules from the cache. We delay cache loading while resolving
  // dependencies so that we properly handle recursively dependent modules.
  if (!impl->activelyResolvingModuleDeps) {
    SmallVector<ModuleState *> modulesToLoad;
    for (auto &[name, moduleState] :
         llvm::drop_begin(impl->importedModules, prevNumModules)) {
      if (moduleState->decl->hasReferenceError)
        continue;
      if (llvm::any_of(moduleState->dependencies, [](ModuleState *dep) {
            return dep->decl->hasReferenceError;
          }))
        continue;
      modulesToLoad.push_back(moduleState.get());
    }
    loadModulesFromCache(modulesToLoad);
  }
  return moduleState;
}

void SharedState::resolveModuleDependencies(ModuleState &moduleState,
                                            StringRef moduleBuffer) {
  ASTDecl &moduleDecl = *moduleState.decl;

  llvm::MapVector<StringAttr, SMLoc> dependencies;

  // Functor used to resolve the module and compute its dependencies via normal
  // parser resolution.
  auto resolveDeclAndComputeDeps = [&]() {
    if (failed(declResolver->resolveFully(moduleDecl, moduleDecl.getLoc())))
      return failure();

    // Walk the body of the module, checking unresolved imports for module
    // dependencies.
    for (auto &[name, decls] : moduleDecl.declsInScope) {
      for (ASTDecl *decl : decls)
        if (auto importOp =
                dyn_cast<UnresolvedImportOp>(decl->getIfOperation()))
          dependencies.insert({importOp.getModuleNameAttr(), decl->getLoc()});
    }
    for (auto it : moduleDecl.unresolvedWildcardImports)
      dependencies.insert(it);
    return mlir::success();
  };

  // For a given textual buffer, we can cache what the dependent module names
  // are. Caching this prevents the need to actually parse the buffer when the
  // content of the module hasn't changed.
  if (impl->transformCache) {
    auto onCacheMiss = [&](Operation *op, Cache::WriteableBufferRef buf,
                           LLCL::AnyAsyncValueRef chain) {
      auto output = LLCL::AsyncValueRef<Cache::BufferRef>::allocate(runtime);
      chain.andThenSync([resolveDeclAndComputeDeps, &dependencies, &moduleDecl,
                         moduleBuffer, output = output.copy(),
                         buf = buf.copy()]() mutable {
        if (failed(resolveDeclAndComputeDeps())) {
          std::move(output).setToError(
              LLCL::getMLIRDiagnostic("failed to resolved body",
                                      moduleDecl.getIfOperation()->getLoc()));
          return;
        }

        // Write the dependencies to the cache. Dependencies are written as a
        // sequence of (name, location) pairs. The location is the offset into
        // the module buffer where the dependency is located.
        llvm::support::endian::Writer writer(*buf, llvm::support::little);
        writer.write((uint64_t)dependencies.size());
        for (auto &[name, loc] : dependencies) {
          writer.write((uint64_t)name.size());
          *buf << name.strref();

          // Sanity check the location pointer, though it should generally
          // always be within the buffer.
          if (loc.getPointer() >= moduleBuffer.data() &&
              loc.getPointer() < moduleBuffer.data() + moduleBuffer.size())
            writer.write((uint64_t)(loc.getPointer() - moduleBuffer.data()));
          else
            writer.write((uint64_t)0);
        }

        std::move(output).emplace(buf.copy());
      });
      return output;
    };
    auto onCacheHit = [&](Operation *op, Cache::BufferRef buf) {
      const char *data = buf->getBufferStart();

      // Functor for reading a uint64_t from the cache buffer.
      auto readInt = [&]() -> uint64_t {
        return llvm::support::endian::readNext<uint64_t, llvm::support::little,
                                               llvm::support::unaligned>(data);
      };

      // Read the dependencies from the cache.
      size_t numDeps = readInt();
      for (size_t i = 0; i < numDeps; ++i) {
        // Read the name.
        size_t nameSize = readInt();
        StringRef name(data, nameSize);
        data += nameSize;

        // Read the location.
        size_t locOffset = readInt();
        auto loc = SMLoc::getFromPointer(moduleBuffer.data() + locOffset);

        // Add the dependency.
        dependencies.insert({StringAttr::get(getContext(), name), loc});
      }
      return buf.copy();
    };

    // Compute the cache key for this module, using the content hash.
    Cache::WriteableBufferRef keyBuf = Cache::WriteableBuffer::get();
    keyBuf->write_impl((const char *)moduleState.contentHash.data(),
                       moduleState.contentHash.size());
    auto output = cachedTransform(
        moduleDecl.getIfOperation(), impl->transformCache.copy(),
        LLCL::AsyncValueRef<Chain>::createReady(runtime), std::move(keyBuf),
        onCacheMiss, onCacheHit);
    await(output);

    // If we don't have a valid cache, just compute the deps directly.
  } else if (failed(resolveDeclAndComputeDeps())) {
    return;
  }

  // Remember if we were actively resolving dependencies before reaching here.
  bool wasImportingAModule = impl->activelyResolvingModuleDeps;
  if (!wasImportingAModule)
    impl->activelyResolvingModuleDeps = true;
  size_t prevNumModules = impl->importedModules.size() - 1;

  // Import all of the dependencies, so that we can resolve their dependencies.
  for (auto [name, loc] : dependencies)
    moduleState.dependencies.insert(&importModuleState(name.getValue(), loc));

  // If we are actively resolving a different module, bail early. That module
  // will handle resolving all of the dependencies of this module, and checking
  // if it's cached. This is necessary to avoid problems with recursive modules.
  if (wasImportingAModule)
    return;

  // At this point, all of the dependent modules are known. Update the modules
  // dependencies to include all dependent modules. We iterate over all of the
  // modules imported during this import, to handle cases of recursive module
  // import.
  bool addedNewDep = false;
  do {
    addedNewDep = false;
    for (auto &it : llvm::drop_begin(impl->importedModules, prevNumModules)) {
      for (unsigned i = 0, e = it.second->dependencies.size(); i < e; ++i)
        for (ModuleState *depState : it.second->dependencies[i]->dependencies)
          addedNewDep |= it.second->dependencies.insert(depState);
    }
  } while (addedNewDep);

  impl->activelyResolvingModuleDeps = false;
}

void SharedState::cacheParsedModules() {
  // If we don't have a valid cache, we can't do anything.
  if (!impl->transformCache)
    return;
  TimeTraceScope<> timeScope("cacheParsedModules");

  SmallVector<LLCL::AnyAsyncValueRef> results;
  for (auto &module : impl->importedModules) {
    if (module.second->importedFromCache)
      continue;
    ModuleState &moduleState = *module.second;
    FileModuleOp moduleOp =
        dyn_cast_if_present<FileModuleOp>(moduleState.decl->getIfOperation());
    if (!moduleOp)
      continue;

    // Re-check if the module is in the cache. If it isn't, we populate it
    // now.
    Cache::BufferRef keyBuffer = moduleState.buildCacheKey(options);
    auto out = AsyncValueRef<Chain>::allocate(runtime);
    auto f = impl->transformCache->contains(
        keyBuffer.copy(),
        LLCL::MLIRLocationDecoder::getEncodedLocation(moduleOp->getLoc()));
    std::move(f).andThenSync(
        [moduleOp, transformCache = impl->transformCache.copy(),
         out = out.copy(), keyBuffer = std::move(keyBuffer)](
            AsyncValueRef<bool> &&alreadyInCache) mutable {
          if (alreadyInCache.isError() || *alreadyInCache)
            return std::move(out).emplace();
          TimeTraceScope<> timeScope(("Caching: " + moduleOp.getName()).str());

          // Write the module to the cache.
          auto writeableTransformResult = Cache::WriteableBuffer::get();
          if (failed(mlir::writeBytecodeToFile(moduleOp,
                                               *writeableTransformResult))) {
            return std::move(out).setToError(LLCL::getMLIRDiagnostic(
                "failed to write bytecode file", moduleOp.getLoc()));
          }
          auto insertResult = transformCache->insert(
              std::move(keyBuffer), std::move(writeableTransformResult));
          insertResult.andThenSync(
              [out = std::move(out)]() mutable { std::move(out).emplace(); });
        });
    results.push_back(std::move(out));
  }
  await(results);
}

ArrayRef<std::string> SharedState::getIncludedFiles() const {
  return impl->includedFiles;
}

/// Given a pointer to the start of a token, find the end of it.
static void adjustTokenEndPoint(SharedState &shared, SMLoc &loc) {
  size_t tokenSize = Lexer::getTokenLength(shared, loc);
  loc = SMLoc::getFromPointer(loc.getPointer() + tokenSize);
}
