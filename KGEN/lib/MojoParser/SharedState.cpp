//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of the SharedState class.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/SharedState.h"
#include "DebugInfo.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/CallEmission.h"
#include "KGEN/MojoParser/ClosureEmitter.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ExprNodes.h"
#include "KGEN/MojoParser/IRValues.h"
#include "KGEN/Support/CompilerProfiling.h"

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/Support/Configuration.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"

#include "Cache/CacheDialect/CachedTransform.h"
#include "LLCL/Runtime/Algorithms.h"
#include "Support/Buffer.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Configuration.h"

#include "Support/Filesystem/Paths.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/bit.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include <filesystem>

#define DEBUG_TYPE "mojo-parser"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

using llvm::SMLoc;
using llvm::SourceMgr;

static void adjustTokenEndPoint(SharedState &shared, SMLoc &loc);

/// Collect all of the default paths used for resolving imports.
static void collectDefaultImportPaths(SmallVector<std::string> &paths) {
  ErrorOr<MojoConfig> cfg = MojoConfig::open();
  if (failed(cfg)) {
    LLVM_DEBUG(llvm::dbgs()
               << "failed to open config: " << cfg.getError() << "\n");
    return;
  }

  // Add any paths specified in the config.
  SmallVector<StringRef> importPaths;
  cfg->getParserImportPaths(importPaths);
  LLVM_DEBUG(llvm::dbgs() << "Using import paths: "
                          << llvm::join(importPaths, ",") << "\n");

  for (StringRef path : importPaths)
    paths.push_back(path.str());
}

struct SharedState::Impl {
  Impl(SharedState &shared, ParserConfig::CachingLevel moduleCachingLevel)
      : sourceNames(shared), moduleCachingLevel(moduleCachingLevel),
        bytecodeParserContext(shared.getContext(), /*verifyAfterParse=*/false) {
  }
  virtual ~Impl() = default;

  SymbolTableCollection symbolTables;

  /// Source name collector.
  SourceNames sourceNames;

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
  /// This is the decl for the builtin 'lit.none' type/attr.
  ASTType noneType;
  NoneAttr noneAttr;

  /// A module state corresponding to the top-level decl. All imported packages
  /// or modules are nested within.
  std::unique_ptr<ModuleState> topLevelModuleState;

  /// A mapping between ASTDecl and the corresponding module state.
  llvm::MapVector<ASTDecl *, ModuleState *> moduleStates;

  /// A mapping between packages and their corresponding module state. A nullptr
  /// entry corresponds to the top level module state.
  /// FIXME(#17327): This only exists to work around the fact that we can't rely
  /// on an ASTDecl's parent reflecting the IR parent. When that issue gets
  /// fixed, this map should be removed in favor of just `moduleStates`.
  DenseMap<PackageOp, ModuleState *> packageStates;

  /// A list of included files used when importing modules. These are used to
  /// generate dependency files.
  SmallVector<std::string> includedFiles;

  /// The set of pre-existing source buffers within the source manager, used if
  /// importing a module whose file is already in the source manager.
  DenseMap<StringRef, int> existingSourceMgrBuffers;

  /// The cache used to store cached transformations within the parser.
  RCRef<Cache::BlobCache<Cache::TransformCacheKey>> transformCache;

  /// The level of module caching enabled for the parser.
  ParserConfig::CachingLevel moduleCachingLevel;

  /// Flag indicating if the deps of a module are currently being resolved.
  bool activelyResolvingModuleDeps = false;

  /// Flag indicating if we should warn on missing doc strings while parsing.
  bool warnMissingDocStrings = false;

  /// If true, use !lit.ref representation for full lifetimes support in Mojo.
  bool experimentalLifetimes = false;

  /// This keeps track of body decorators for a given declaration, this is
  /// logically part of ASTDecl, but is stored out of line to reduce its size
  /// since these are uncommon.
  DenseMap<const ASTDecl *, std::vector<ExprNode *>> bodyDecorators;

  /// The implicit builtin imports added to each module.
  SmallVector<StringAttr> implicitBuiltinImports;

  /// The decl corresponding to the standard library package.
  ModuleState *stdlibPackageState = nullptr;

  /// The parser configuration used when loading bytecode.
  mlir::ParserConfig bytecodeParserContext;

  /// The closure wrapper types that have already been generated, keyed off
  /// name.
  llvm::DenseMap<std::pair<SignatureType, StringAttr>, StructDeclOp>
      closureWrappers;

  /// The capture values and decls associated with their enclosing nested
  /// function.
  llvm::DenseMap<ASTDecl *, llvm::MapVector<ASTDecl *, Capture>>
      capturesInScope;
};

SharedState::SharedState(llvm::SourceMgr &sourceMgr, ParserConfig &config)
    : diags(sourceMgr, config.context, config.useMLIRDiagnostics,
            config.maxNotesPerDiagnostic),
      options(config.options),
      declResolver(std::make_unique<DeclResolver>(*this)),
      parserListener(config.parserListener), runtime(config.runtime),
      parsingStandardLibrary(config.parsingStandardLibrary),
      useBuiltinModule(config.useBuiltinModule),
      impl(std::make_unique<Impl>(*this, config.moduleCachingLevel)) {
  if (!options.searchPaths.empty()) {
    SmallVector<StringRef> paths;
    StringRef(options.searchPaths)
        .split(paths, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    llvm::append_range(impl->autoImportDirs, paths);
  } else {
    collectDefaultImportPaths(impl->autoImportDirs);
  }
  impl->warnMissingDocStrings = config.warnMissingDocStrings;
  impl->experimentalLifetimes = config.experimentalLifetimes;

  preloadAllKGENDialects(config.context);

  // Record any existing buffers in the source manager.
  for (int i = 0, e = sourceMgr.getNumBuffers(); i < e; ++i) {
    int bufferId = i + 1;
    impl->existingSourceMgrBuffers.try_emplace(
        sourceMgr.getMemoryBuffer(bufferId)->getBufferIdentifier(), bufferId);
  }

  // Tell the diagnostics machinery how to find the end of a token lazily when
  // it needs it.
  diags.setTokenEndPointAdjustmentFn(
      [=](SMLoc &loc) { adjustTokenEndPoint(*this, loc); });

  if (options.getDebugInfoLevelForInput() > CompilationOptions::kSynthetic) {
    diBuilder = std::make_unique<DebugInfo::DIBuilder>(config.context);

    // TODO: Dwarf technically has a language for python, but it's not really
    // what we want here AFAICT (our compilation model isn't the same as
    // python's). Figure out what we actually want here (though C works well
    // enough for now).
    diBuilder->initializeCompileUnit(
        options.debugInfoLanguage,
        diBuilder->createFile(diags.getBufferNameIdentifier(), "/"), "Mojo",
        /*isOptimized=*/true, options.getDIEmissionKind());
  }

  // Create a cache for use by the parser.
  if (config.moduleCachingLevel != ParserConfig::kCacheNone) {
    auto transformCacheBackendOr = Cache::getLocalDefaultBackendChain(
        runtime, (std::filesystem::path(".mojo_cache") / "mojo").string(),
        KGEN_VERSION_STRING);
    if (failed(transformCacheBackendOr))
      return;
    impl->transformCache =
        RCRef<Cache::BlobCache<Cache::TransformCacheKey>>::create(
            transformCacheBackendOr.takeValue());
  }
}

SharedState::~SharedState() { declResolver.reset(); }

bool SharedState::shouldWarnMissingDocStrings() const {
  return impl->warnMissingDocStrings;
}

bool SharedState::useExperimentalLifetimes() const {
  return impl->experimentalLifetimes;
}

void SharedState::initialize(ASTDecl &topLevelDecl) {
  assert(!impl->topLevelDecl && "already initialized");
  impl->topLevelDecl = &topLevelDecl;
  impl->topLevelModuleState = std::make_unique<ModuleState>(&topLevelDecl);
  impl->moduleStates[&topLevelDecl] = impl->topLevelModuleState.get();
  impl->packageStates[nullptr] = impl->topLevelModuleState.get();

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

ASTDecl &SharedState::getTopLevelDecl() { return *impl->topLevelDecl; }

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

StringAttr SharedState::getMangledModuleName(MLIRContext *ctx,
                                             StringRef moduleName) {
  return StringAttr::get(ctx, "$" + moduleName);
}

/// Add declarations for magic things to the builtins decl.
void SharedState::addBuiltinTypes(ASTDecl &builtinsDecl) {
  DeclResolver &resolver = *declResolver;
  MLIRContext *context = getContext();

  // Add a declarations for builtin types.
  impl->noneType = KGEN::NoneType::get(context);
  impl->noneAttr = NoneAttr::get(context);

  // Make the type check error type.  Anything that references this will
  // considering it erroneous and already declared as such.
  impl->typeCheckErrorType = TypeCheckErrorType::get(context);

  // Add an empty struct with the specified name to the resolver.
  auto anyRegTypeType = TypeType::get(getContext());
  auto addMagicMLIRDecl = [&](StringRef name, Type magicType) {
    TypedAttr value = TypeConstantAttr::get(magicType, anyRegTypeType);
    resolver.addFullyResolvedDecl(PValue(value), name, builtinsDecl.getLoc(),
                                  &builtinsDecl);
  };

  addMagicMLIRDecl("__mlir_attr", MagicMLIRAttrType::get(context));
  addMagicMLIRDecl("__mlir_op", MagicMLIROpType::get(context));
  addMagicMLIRDecl("__mlir_type", MagicMLIRTypeType::get(context));
}

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

Operation *SharedState::lookupSymbolIn(ASTDecl *container, StringAttr name) {
  Operation *tableOp = container->getIfOperation();
  assert(tableOp && "decl is not an operation");
  return impl->symbolTables.getSymbolTable(tableOp).lookup(name);
}

//===----------------------------------------------------------------------===//
// ASTDecl
//===----------------------------------------------------------------------===//

/// Return any decorators that need to be processed as part of body resolution
/// phase for a decl.
ArrayRef<ExprNode *> ASTDecl::getBodyDecorators(SharedState &state) const {
  if (!hasBodyDecorators)
    return {};
  return state.getImpl().bodyDecorators[this];
}

/// During signature resolution, this is called with any decorators that need
/// to persist until body resolution.
void ASTDecl::setBodyDecorators(ArrayRef<ExprNode *> decorators,
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
  ModuleState(ASTDecl *decl = nullptr) : decl(decl) { contentHash.fill(0); }
  ModuleState(ASTDecl *decl, StringRef sourcePath, bool enableCaching = false)
      : decl(decl), sourcePath(sourcePath.str()),
        canCacheModule(enableCaching) {
    contentHash.fill(0);
  }
  ~ModuleState() {
    // Drop any remaining operations in the reader to avoid dangling
    // unmaterialized operations. If these were neded, they would have been
    // handled already as part of parsing.
    if (bytecodeReader)
      (void)bytecodeReader->finalize([](Operation *) { return false; });
  }

  /// Insert a nested module state.
  ModuleState &insertNestedModule(StringAttr name,
                                  std::unique_ptr<ModuleState> module) {
    nestedModuleAllocations.emplace_back(std::move(module));
    nestedModules.insert({name, nestedModuleAllocations.back().get()});
    return *nestedModuleAllocations.back();
  }

  /// The decl associated with the module or package.
  ASTDecl *decl = nullptr;
  /// An optional bytecode reader, in the case where this decl was loaded from
  /// bytecode as opposed to source.
  std::unique_ptr<mlir::BytecodeReader> bytecodeReader;
  /// The optional source path of this module if it was loaded from source.
  std::optional<std::string> sourcePath;

  //===--------------------------------------------------------------------===//
  // File Module Specific State
  //===--------------------------------------------------------------------===//

  /// Build the cache key for this module.
  WriteableBufferRef buildCacheKey(const CompilationOptions &options) {
    auto keyBuf = WriteableBuffer::get();

    // Add the full module name to the cache key, this ensures proper caching
    // when the same module is in different packages.
    std::string moduleName = getFlattenedSymbolName(
        getFullyResolvedSymbolRef(cast<mlir::SymbolOpInterface>(*decl)));
    keyBuf->write(moduleName.data(), moduleName.size());

    // Add the module contents to the cache key.
    keyBuf->write((const char *)contentHash.data(), contentHash.size());

    // Add the module dependencies to the cache key.
    for (ModuleState *dep : dependencies) {
      keyBuf->write((const char *)dep->contentHash.data(),
                    dep->contentHash.size());
    }

    // Add the compilation options to the cache key.
    options.print(*keyBuf);
    return keyBuf;
  }

  /// A hash associated with the modules contents.
  llvm::BLAKE3Result<> contentHash;
  /// The set of other modules that this module depends on.
  llvm::SmallSetVector<ModuleState *, 4> dependencies;
  /// A flag indicating if this module can be cached.
  bool canCacheModule = false;

  //===--------------------------------------------------------------------===//
  // Package Specific State
  //===--------------------------------------------------------------------===//

  /// The set of nested modules.
  SmallVector<std::unique_ptr<ModuleState>> nestedModuleAllocations;
  DenseMap<StringAttr, ModuleState *> nestedModules;
};

//===----------------------------------------------------------------------===//
// Name Lookup
//===----------------------------------------------------------------------===//

/// Return true if the specified type has a declared member with the specified
/// name.
bool SharedState::typeHasMember(ASTType type, StringRef name, llvm::SMLoc loc) {
  ASTDecl *typeDecl = type.getDecl(*this);
  if (!typeDecl) // MLIR types have no methods.
    return false;
  return typeHasMember(*typeDecl, name, loc);
}

bool SharedState::typeHasMember(ASTDecl &typeDecl, StringRef name,
                                llvm::SMLoc loc) {
  return lookupAndResolveDecl(name, loc, typeDecl,
                              /*searchParentScopes=*/false)
      .isSuccess();
}

/// Perform a name lookup in the specified scope and return the named
/// declaration as a LookupResult.
auto SharedState::lookupAndResolveDecl(StringRef name, SMLoc loc,
                                       ASTDecl &scope, bool searchParentScopes)
    -> LookupResult {

  // Ensure the context is fully resolved, so all its members are known.  It
  // would be bad to look something up in a scope without all members known.
  if (failed(declResolver->resolveFully(scope, loc)))
    return LookupResult::getErroneous();

  auto nameAttr = StringAttr::get(getContext(), name);

  // Look up the name.
  auto lookupInScope = [&](ASTDecl &scope) -> ArrayRef<ASTDecl *> {
    // Check if we already have a declaration for this name in the current
    // scope.
    auto result = scope.lookupInCurrentScope(nameAttr);
    if (!result.empty())
      return result;

    // If the lookup failed, try to resolve any wildcard imports in the scope.
    // We don't know if these imports will actually provide the decl we are
    // looking for, so we have to try until we find one that does.
    for (int i = 0, e = scope.unresolvedWildcardImports.size(); i < e;) {
      auto it = std::next(scope.unresolvedWildcardImports.begin(), i);
      auto [moduleName, locAndIsFullImport] = *it;
      auto [loc, isFullImport] = locAndIsFullImport;

      // Don't try wildcard imports if we wouldn't import this name anyways.
      if (!isFullImport && name.starts_with("_")) {
        ++i;
        continue;
      }
      --e;
      scope.unresolvedWildcardImports.erase(it);

      // Resolve the import. If it fails, don't fail the search immediately,
      // keep checking for something that can resolve the decl we care about.
      if (failed(declResolver->importWildCardDeclsFromModule(
              scope, moduleName, isFullImport, loc)))
        continue;
      // Re-check the lookup in the scope now that the wildcard import has
      // been resolved.
      result = scope.lookupInCurrentScope(nameAttr);
      if (!result.empty())
        return result;
    }

    return {};
  };

  auto getEntry = [&]() -> LookupResult {
    if (!searchParentScopes) {
      ArrayRef<ASTDecl *> result = lookupInScope(scope);
      if (result.empty())
        return LookupResult::getFailure({});
      else
        return LookupResult::getSuccess(result);
    }
    ArrayRef<ASTDecl *> skipped = {};
    ASTDecl *curSearchScope = &scope;
    do {
      ArrayRef<ASTDecl *> e = lookupInScope(*curSearchScope);
      if (!e.empty()) {
        if (isa<StructDeclOp>(*curSearchScope) && !(*e.front()).getIfPValue()) {
          // Skip struct bodies when searching up parent scopes, unless the
          // value is a parameter.
          if (skipped.empty())
            skipped = e;

          continue;
        }
        return LookupResult::getSuccess(e);
      }
    } while ((curSearchScope = curSearchScope->parentDecl));
    // If we found a name in a context that we skip, return it in the failure
    // for diagnostic reporting.
    return LookupResult::getFailure(skipped);
  };

  LookupResult entry = getEntry();

  // If nothing was found, return a failure.
  if (entry.isFailure())
    return entry;
  SmallVector<ASTDecl *> resultDecls(entry.getIfSuccess());

  // If the lookup succeeded, make sure the signature for the referenced decls
  // are understood. Make a copy of the entries to avoid dangling references if
  // we end up invalidating the decl map.
  bool wasUnresolvedImport =
      !resultDecls.empty() && isa<UnresolvedImportOp>(*resultDecls.front());
  for (ASTDecl *decl : resultDecls) {
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
  if (entry.isSuccess() && wasUnresolvedImport)
    return lookupAndResolveDecl(name, loc, scope, searchParentScopes);

  // We return a pointer into the TinyPtrVector entry in the scope.  This should
  // be stable because you can't perform a lookup into a decl that has unknown
  // entries, and we just resolved all the signatures for all the decls.
  return entry;
}

/// Perform a name lookup for a member in the specified type.
auto SharedState::lookupAndResolveDecl(StringRef name, SMLoc loc, ASTType scope,
                                       bool searchParentScopes)
    -> LookupResult {
  if (auto *decl = scope.getDecl(*this))
    return lookupAndResolveDecl(name, loc, *decl, searchParentScopes);
  return LookupResult::getFailure({});
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
  if (!structOp.getParams().empty()) {
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

static ASTDecl *findBuiltinTrait(StringRef traitName, SMLoc location,
                                 ASTDecl *parent, SharedState &shared) {
  LookupResult lookup =
      shared.lookupAndResolveDecl(traitName, location, *parent, true);
  if (!lookup.isFailure() && !lookup.getIfSuccess().empty()) {
    for (ASTDecl *result : lookup.getIfSuccess()) {
      if (auto trait = dyn_cast<TraitDeclOp>(result))
        return result;
    }
  }
  return nullptr;
}

ASTDecl *SharedState::lookupAnyTypeTrait(llvm::SMLoc loc, ASTDecl *context) {
  return findBuiltinTrait("AnyType", loc, context, *this);
}

ASTDecl *SharedState::lookupCopyableTrait(llvm::SMLoc loc, ASTDecl *context) {
  return findBuiltinTrait("Copyable", loc, context, *this);
}

ASTDecl *SharedState::lookupMovableTrait(llvm::SMLoc loc, ASTDecl *context) {
  return findBuiltinTrait("Movable", loc, context, *this);
}

/// Resolve the absolute path for a given module name within the provided
/// directory. Returns nullopt if the module cannot be found.
static std::optional<std::string>
resolveModulePath(SharedState &shared, llvm::SMLoc includeLoc,
                  StringRef moduleName, StringRef includeDir,
                  bool isParsingStandardLibrary) {
  // Gets the name of the file or directory in a case sensitive way. On non-case
  // sensitive systems we cannot just do `path / moduleName` since the
  // constructed path will not adhere to case sensitivity.
  auto getFileName =
      [moduleName = moduleName.str(),
       includeDir =
           includeDir.str()]() -> std::optional<std::filesystem::path> {
  // The file system is always case-sensitive on linux.
#ifdef __linux__
    return std::filesystem::path(includeDir) / moduleName;
#else  // !__linux__
    std::error_code ec;
    auto iter = std::filesystem::directory_iterator(includeDir, ec);
    if (ec)
      return std::nullopt;
    for (const auto &entry : iter)
      if (entry.path().filename().stem().string() == moduleName)
        return entry.path();

    return std::nullopt;
#endif // __linux__
  };

  // If we cannot find a file or directory with the case-sensitive name, then
  // return early.
  auto nameOr = getFileName();
  if (!nameOr)
    return std::nullopt;

  std::filesystem::path name = *nameOr;

  // Check if we have a source package with this name.
  if (Filesystem::isMojoSourcePackagePath(name))
    return name.generic_string();

  // Check for a binary package with this name. We don't enable binary packages
  // when parsing the standard library, as many packages are interdependent,
  // which means we can't serialize their processing.
  std::error_code ec;
  std::string foundName;
  if (!isParsingStandardLibrary) {
    if (std::filesystem::exists(name.replace_extension("mojopkg"), ec))
      foundName = name.string();
    if (std::filesystem::exists(name.replace_extension("📦"), ec)) {
      if (!foundName.empty()) {
        shared.emitError(includeLoc, "ambiguous import, both ")
            << foundName << " and " << name.string()
            << " exist in file system.";
      }
      foundName = name.string();
    }
    if (!foundName.empty())
      return foundName;
  }

  // Otherwise, check for a source module with this name.
  if (std::filesystem::exists(name.replace_extension("mojo"), ec))
    foundName = name.string();
  if (std::filesystem::exists(name.replace_extension("🔥"), ec)) {
    if (!foundName.empty()) {
      shared.emitError(includeLoc, "ambiguous import, both ")
          << foundName << " and " << name.string() << " exist in file system.";
    }
    foundName = name.string();
  }
  if (!foundName.empty())
    return foundName;

  return std::nullopt;
}

/// Resolve the absolute path for a given module name. Returns nullopt if the
/// module cannot be found.
static std::optional<std::string>
resolveModulePath(SharedState &sharedState, StringRef moduleName,
                  llvm::SMLoc includeLoc, bool isParsingStandardLibrary) {
  unsigned includeBufferId =
      sharedState.getSourceMgr().FindBufferContainingLoc(includeLoc);

  std::optional<std::string> result;
  sharedState.traverseImportDirectories(includeBufferId, [&](StringRef dir) {
    // Don't try to resolve modules that reside within a package.
    if (Filesystem::isMojoSourcePackagePath(dir.str())) {
      // TODO: It'd be nice to emit a list of potential modules that the
      // name might correspond with if it did resolve to one inside of this
      // package.
      return WalkResult::advance();
    }
    if ((result = resolveModulePath(sharedState, includeLoc, moduleName, dir,
                                    isParsingStandardLibrary)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result;
}

ASTDecl &SharedState::importModule(StringRef name, PackageOp currentPackage,
                                   llvm::SMLoc loc) {
  ModuleState *moduleState = impl->packageStates[currentPackage];
  assert(moduleState && "unexpected package without a module state");
  return *importModuleState(name, moduleState->decl, loc).decl;
}

SharedState::ModuleState &SharedState::importModuleState(StringRef name,
                                                         ASTDecl *context,
                                                         llvm::SMLoc loc) {
  CompilerTimeTraceScope fullTimeScope(("importModule: " + name).str());

  // Handle the case where the name is comprised of multiple components.
  if (name.contains('.'))
    return importRelativeModuleState(name, context, loc);

  // Otherwise, we're importing an absolute module or package at the top-level.
  return importSubModuleState(name, impl->topLevelDecl, loc);
}

SharedState::ModuleState &SharedState::importSubModuleState(StringRef name,
                                                            ASTDecl *parentDecl,
                                                            llvm::SMLoc loc) {
  // Grab the parent module state.
  ModuleState *parentState = impl->moduleStates[parentDecl];
  assert(parentState && "parent decl must have a module state");

  // Mangle the module name during import to avoid conflicts with symbols that
  // are actually visible. We may import a module, but not directly expose it
  // via its module name.
  StringAttr mangledName = getMangledModuleName(getContext(), name);

  // Check to see if we've already imported this module.
  auto it = parentState->nestedModules.find(mangledName);
  if (it != parentState->nestedModules.end())
    return *it->second;

  // Resolve the path and decl name for this module.
  std::optional<std::string> modulePath;
  auto declName = StringAttr::get(getContext(), name);
  if (parentState->decl != impl->topLevelDecl) {
    if (!parentState->sourcePath) {
      return createErrorModuleState(loc, mangledName, *parentState->decl,
                                    "unable to locate module '" + name + "'");
    }
    modulePath = resolveModulePath(*this, loc, name, *parentState->sourcePath,
                                   parsingStandardLibrary);
  } else {
    // If this is a top-level import, try to resolve a standard library module.
    // We current bundle all of the standard library packages into one mega
    // package, but still want to expose them separately.
    if (impl->stdlibPackageState && name != "stdlib") {
      // Check for an existing module for this name. If we find one, insert it
      // into the parent state and return it.
      auto it = impl->stdlibPackageState->nestedModules.find(mangledName);
      if (it != impl->stdlibPackageState->nestedModules.end()) {
        parentState->nestedModules.insert({mangledName, it->second});
        return *it->second;
      }

      // Otherwise, if the standard library is a source package, check to see if
      // we can resolve a path from it.
      if (impl->stdlibPackageState->sourcePath) {
        modulePath = resolveModulePath(*this, loc, name,
                                       *impl->stdlibPackageState->sourcePath,
                                       parsingStandardLibrary);
        if (modulePath) {
          ModuleState &moduleState = importModuleState(("stdlib." + name).str(),
                                                       impl->topLevelDecl, loc);
          parentState->nestedModules.insert({mangledName, &moduleState});
          return moduleState;
        }
      }
    }

    // Otherwise, go through the normal import path.
    modulePath = resolveModulePath(*this, name, loc, parsingStandardLibrary);
  }
  if (!modulePath) {
    return createErrorModuleState(loc, mangledName, *parentState->decl,
                                  "unable to locate module '" + name + "'");
  }
  auto moduleBuilder = impl->topLevelDecl->getDeclEndBuilder();

  // If the path was a directory, we're importing a source package.
  if (std::filesystem::is_directory(*modulePath)) {
    auto fileLoc = moduleBuilder.getAttr<FileLineColLoc>(
        *modulePath, /*line=*/0, /*column=*/0);
    return createPackageState(declName, mangledName, *modulePath, *parentState,
                              fileLoc);
  }

  // Check if the path is a binary package.
  StringRef pathRef(*modulePath);
  if (pathRef.ends_with(".mojopkg") || pathRef.ends_with(".📦"))
    return createBinaryPackageState(loc, declName, mangledName, *modulePath,
                                    *parentState);

  // Open the module file within the source manager. Reuse an existing file if
  // we've already opened it.
  unsigned fileID = impl->existingSourceMgrBuffers.lookup(pathRef);
  if (!fileID) {
    std::string fullPath;
    fileID = getSourceMgr().AddIncludeFile(*modulePath, loc, fullPath);
    impl->includedFiles.push_back(fullPath);
  }

  // Enable caching for the module if caching is enable and it's not the main
  // file, or if we're caching all modules.
  bool enableCaching = impl->moduleCachingLevel != ParserConfig::kCacheNone;
  if (impl->moduleCachingLevel == ParserConfig::kCacheImports)
    enableCaching = fileID != getSourceMgr().getMainFileID();

  // Now that we have a MemoryBuffer, we can lex it, and therefore parse it.
  // do so.
  const llvm::MemoryBuffer *moduleBuffer =
      getSourceMgr().getMemoryBuffer(fileID);
  auto fileLoc = moduleBuilder.getAttr<FileLineColLoc>(
      moduleBuffer->getBufferIdentifier(), /*line=*/0, /*column=*/0);
  return createModuleState(declName, mangledName, moduleBuffer, *parentState,
                           fileLoc, enableCaching);
}

SharedState::ModuleState &
SharedState::importRelativeModuleState(StringRef name, ASTDecl *parentDecl,
                                       llvm::SMLoc loc) {
  auto emitError = [&](const Twine &message = "") -> ModuleState & {
    return createErrorModuleState(loc, getMangledModuleName(getContext(), name),
                                  *parentDecl, message);
  };

  // If the name starts with a `.`, it is relative to the current package.
  if (name.consume_front(".")) {
    // Find the current package.
    while (!isa<PackageOp>(*parentDecl) && parentDecl->parentDecl)
      parentDecl = parentDecl->parentDecl;
    if (!isa<PackageOp>(*parentDecl))
      return emitError("cannot import relative to a top-level package");

    // Otherwise, this is a package relative to the current parent.
    while (name.consume_front(".")) {
      if (!parentDecl->parentDecl || !isa<PackageOp>(*parentDecl->parentDecl)) {
        return emitError(
            "attempted relative import with no known parent package");
      }
      parentDecl = parentDecl->parentDecl;
    }

    // If the name is empty, we're grabbing the parent package.
    if (name.empty())
      return *impl->moduleStates[parentDecl];
  } else {
    // Otherwise, we're resolving relative to a top-level package.
    StringRef parentName;
    std::tie(parentName, name) = name.split('.');
    parentDecl = importModuleState(parentName, impl->topLevelDecl, loc).decl;
  }

  // The rest of the name resolves a nested module or package from the current
  // parent.
  SmallVector<StringRef> remainingNames;
  name.split(remainingNames, '.');
  name = remainingNames.pop_back_val();
  for (StringRef parentName : remainingNames) {
    // Lookup the next decl in the chain.
    auto lookupResult = lookupAndResolveDecl(parentName, loc, *parentDecl,
                                             /*searchParentScopes*/ false);
    if (lookupResult.getIfSuccess().empty())
      return emitError("'" + parentName +
                       "' does not refer to a nested package");
    parentDecl = lookupResult.getIfSuccess()[0];
    if (!isa<PackageOp>(*parentDecl))
      return emitError("'" + parentName +
                       "' does not refer to a nested package");
  }

  // Now we can import the final decl. If the parent package has an unresolved
  // import, mark it as resolved and import the state for the module.
  if (failed(declResolver->resolveFully(*parentDecl, loc)))
    return emitError();
  TinyPtrVector<ASTDecl *> &existingDecls =
      parentDecl->declsInScope[StringAttr::get(getContext(), name)];
  if (!existingDecls.empty()) {
    ASTDecl *existingDecl = existingDecls.front();

    // The decl already exists, so we can just return it.
    if (isa<FileModuleOp, PackageOp>(*existingDecl))
      return *impl->moduleStates[existingDecl];

    // If the decl isn't an unresolved import, emit an error.
    if (!isa<UnresolvedImportOp>(*existingDecl))
      return emitError("'" + name + "' does not refer to a package or module");
    existingDecls.clear();
  }
  return importSubModuleState(name, parentDecl, loc);
}

bool SharedState::hasBuiltinModule() const { return useBuiltinModule; }

static ASTType resolveBuiltinModuleType(ASTDecl &context, llvm::SMLoc loc,
                                        StringRef typeName,
                                        SharedState &shared) {
  // Unresolved wildcard imports have been added for all builtin modules. Search
  // from the contextual ASTDecl.
  LookupResult lookup = shared.lookupAndResolveDecl(
      typeName, loc, context, /*searchParentScopes=*/true);
  if (!lookup.isFailure() && !lookup.getIfSuccess().empty()) {
    ASTDecl *decl = lookup.getIfSuccess().front();
    if (auto structDecl = dyn_cast<StructDeclOp>(decl))
      return structDecl.bindReference();

    InflightDiag diag = shared.emitError(loc, "builtin '")
                        << typeName << "' identifier does not denote a type";
    diag.attachNote(decl->getLoc())
        << "'" << typeName << "' identifier redeclared here";
    return shared.getTypeCheckErrorType();
  }

  if (!lookup.isErroneous())
    shared.emitError(loc, "could not find builtin '") << typeName << "' type";
  return shared.getTypeCheckErrorType();
}

ASTType SharedState::getBuiltinBoolType(ASTDecl &context, llvm::SMLoc loc) {
  return resolveBuiltinModuleType(context, loc, "Bool", *this);
}

ASTType SharedState::getBuiltinTupleType(ASTDecl &context, llvm::SMLoc loc) {
  return resolveBuiltinModuleType(context, loc, "Tuple", *this);
}

ASTType SharedState::getBuiltinErrorType(ASTDecl &context, llvm::SMLoc loc) {
  return resolveBuiltinModuleType(context, loc, "Error", *this);
}

ASTType SharedState::getBuiltinIntType(ASTDecl &context, llvm::SMLoc loc) {
  return resolveBuiltinModuleType(context, loc, "Int", *this);
}

ASTType SharedState::getBuiltinIntLiteralType(ASTDecl &context,
                                              llvm::SMLoc loc) {
  return resolveBuiltinModuleType(context, loc, "IntLiteral", *this);
}

ASTType SharedState::getBuiltinStringLiteralType(ASTDecl &context,
                                                 llvm::SMLoc loc) {
  return resolveBuiltinModuleType(context, loc, "StringLiteral", *this);
}

ASTType SharedState::getBuiltinSliceType(ASTDecl &context, llvm::SMLoc loc) {
  return resolveBuiltinModuleType(context, loc, "slice", *this);
}

ASTType SharedState::getBuiltinListLiteralType(ASTDecl &context,
                                               llvm::SMLoc loc) {
  return resolveBuiltinModuleType(context, loc, "ListLiteral", *this);
}

ASTType SharedState::getBuiltinVariadicListType(ASTDecl &context,
                                                llvm::SMLoc loc, bool inMem) {
  return resolveBuiltinModuleType(
      context, loc, inMem ? "VariadicListMem" : "VariadicList", *this);
}

ASTType SharedState::getBuiltinDoubleType(ASTDecl &context, llvm::SMLoc loc) {
  return resolveBuiltinModuleType(context, loc, "FloatLiteral", *this);
}

ASTType SharedState::getBuiltinCoroutineType(ASTDecl &context,
                                             llvm::SMLoc loc) {
  return resolveBuiltinModuleType(context, loc, "Coroutine", *this);
}

ASTType SharedState::getBuiltinRaisingCoroutineType(ASTDecl &context,
                                                    llvm::SMLoc loc) {
  return resolveBuiltinModuleType(context, loc, "RaisingCoroutine", *this);
}

ASTType SharedState::getBuiltinCaptureListType(llvm::SMLoc loc) {
  ASTDecl &closureModule =
      importModule("builtin._closure", /*currentPackage=*/nullptr, loc);
  return resolveBuiltinModuleType(closureModule, loc,
                                  "__ParameterClosureCaptureList", *this);
}

ArrayRef<ASTDecl *> SharedState::getBuiltinFunction(ASTDecl &context,
                                                    StringRef moduleName,
                                                    StringRef fnName,
                                                    llvm::SMLoc loc) {
  ASTDecl &module = importModule(moduleName, /*currentPackage=*/nullptr, loc);
  LookupResult result =
      lookupAndResolveDecl(fnName, loc, module, /*searchParentScopes=*/false);
  if (!result.isSuccess() || result.getIfSuccess().empty()) {
    emitError(loc, "internal error: could not find builtin function '")
        << fnName << "'";
    return {};
  }
  ArrayRef<ASTDecl *> decls = result.getIfSuccess();
  if (!isa<LIT::FuncOp>(decls.front())) {
    emitError(loc, "internal error: builtin '")
        << fnName << "' does not refer to a function";
    return {};
  }
  return decls;
}

/// This returns an instance of Tuple[...] with the specified element types
/// installed.
ASTType SharedState::getBuiltinTupleInstantion(ASTDecl &context,
                                               llvm::SMLoc loc,
                                               ArrayRef<Type> elements) {
  auto tupleType = getBuiltinTupleType(context, loc);
  if (tupleType.isTypeCheckErrorType())
    return tupleType;

  // Get the pack parameter from the Tuple type.
  ASTDecl &tupleLiteralDecl = *tupleType.getDecl(*this);
  auto tupleLiteralStruct = cast<StructDeclOp>(tupleLiteralDecl);
  assert(tupleLiteralStruct.getParams().size() == 1);
  ParamDeclAttr tupleParam = tupleLiteralStruct.getParams()[0];

  // Bind the correct element types for the tuple to the tuple type.
  SmallVector<TypedAttr> eltTypes;
  auto anyRegTypeType = TypeType::get(tupleLiteralStruct.getContext());
  for (auto elt : elements)
    eltTypes.push_back(TypeConstantAttr::get(elt, anyRegTypeType));

  // Bind it to a VariadicAttr of the right elements.
  TypedAttr packAttr =
      VariadicAttr::get(eltTypes, cast<VariadicType>(tupleParam.getType()));
  return BindTypeAttr::get(PValue(tupleType), packAttr);
}

void SharedState::loadModulesFromCache(
    MutableArrayRef<ModuleState *> moduleStates) {
  // If we don't have a valid cache, we can't do anything.
  if (!impl->transformCache || moduleStates.empty())
    return;

  // Check the cache results for the various modules.
  for (ModuleState *moduleState : moduleStates) {
    if (!moduleState->canCacheModule)
      continue;
    // If the module has already been resolved in any form, we shouldn't
    // try reading it from the cache.
    if (moduleState->decl->resolvedness > DeclResolvedness::unparsed)
      continue;
    WriteableBufferRef keyBuf = moduleState->buildCacheKey(options);

    auto out = AsyncValueRef<Chain>::allocate(runtime);
    auto f = impl->transformCache->find(
        std::move(keyBuf), LLCL::MLIRLocationDecoder::getEncodedLocation(
                               moduleState->decl->getIfOperation()->getLoc()));
    std::move(f).andThenSync(
        [this, moduleState, out = out.copy()](
            AsyncValueRef<std::optional<BufferRef>> &&f) mutable {
          // If the module isn't in the cache, process it as normal. We will
          // attempt to cache it later instead of now, given that we can't
          // reliably resolve everything in the module right now.
          if (f.isError())
            return std::move(out).setToError(f.takeDiagnostic());
          if (!f->has_value())
            return std::move(out).emplace();
          ASTDecl &moduleDecl = *moduleState->decl;
          FileModuleOp moduleOp = cast<FileModuleOp>(moduleDecl);
          CompilerTimeTraceScope fullTimeScope(
              ("loadModuleFromCache: " + moduleOp.getName()).str());

          // Read the cached IR.
          Block b;
          {
            CompilerTimeTraceScope timeScope("readBytecodeFile");
            auto sourceMgr = std::make_shared<llvm::SourceMgr>();
            sourceMgr->AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(
                                              (**f)->getBuffer(),
                                              /*BufferName=*/"",
                                              /*RequiresNullTerminator=*/false),
                                          SMLoc());
            const llvm::MemoryBuffer *memoryBuf =
                sourceMgr->getMemoryBuffer(sourceMgr->getMainFileID());
            moduleState->bytecodeReader =
                std::make_unique<mlir::BytecodeReader>(
                    memoryBuf->getMemBufferRef(), impl->bytecodeParserContext,
                    /*lazyLoad=*/true, sourceMgr);

            // Read in the cached bytecode. If we fail, bail and try processing
            // the IR as normal.
            if (failed(moduleState->bytecodeReader->readTopLevel(&b))) {
              return std::move(out).setToError(LLCL::getMLIRDiagnostic(
                  "failed to read module bytecode", moduleOp.getLoc()));
            }
          }

          // Replace the module with the cached IR.
          FileModuleOp cachedModuleOp = cast<FileModuleOp>(b.front());
          cachedModuleOp->moveAfter(moduleOp);
          moduleDecl.setIRValue(DeclIRValue(cachedModuleOp));
          SymbolTable &symtab =
              impl->symbolTables.getSymbolTable(moduleOp->getParentOp());
          symtab.erase(moduleOp);
          symtab.insert(cachedModuleOp);

          // Mark the module as imported from cache.
          moduleState->decl->loadedFromBytecode = true;
          moduleDecl.resolvedness = DeclResolvedness::signature;
          std::move(out).emplace();
        });
    LLCL::await(out);
  }
}

void SharedState::importBuiltinModules(ASTDecl &moduleDecl) {
  // Check if this is the first attempt at resolving the builtin modules.
  if (impl->implicitBuiltinImports.empty()) {
    // Import the main standard library package.
    impl->stdlibPackageState =
        &importModuleState("stdlib", impl->topLevelDecl, moduleDecl.getLoc());
    if (failed(declResolver->resolveFully(*impl->stdlibPackageState->decl,
                                          moduleDecl.getLoc())))
      return;

    // Import the builtin package.
    ASTDecl &builtinsPackageDecl =
        *importModuleState("stdlib.builtin", impl->topLevelDecl,
                           moduleDecl.getLoc())
             .decl;
    if (failed(declResolver->resolveFully(builtinsPackageDecl,
                                          moduleDecl.getLoc())))
      return;

    for (StringRef name :
         llvm::make_first_range(builtinsPackageDecl.getDeclsInScope())) {
      // Directly nested modules/packages look like `$foo`.
      if (!name.consume_front("$"))
        continue;
      impl->implicitBuiltinImports.emplace_back(
          StringAttr::get(getContext(), "builtin." + name));
    }
  }

  for (StringAttr import : impl->implicitBuiltinImports)
    moduleDecl.addUnresolvedWildCardImport(import, /*isFullImport=*/false,
                                           moduleDecl.getLoc());
}

ASTDecl &SharedState::createModule(StringRef moduleName,
                                   const llvm::MemoryBuffer *moduleBuffer,
                                   FileLineColLoc loc) {
  StringAttr mangledName = getMangledModuleName(getContext(), moduleName);

  // Create a new module state. This isn't an imported module, so we can only
  // cache if we're caching everything.
  ModuleState &state =
      createModuleState(StringAttr::get(getContext(), moduleName), mangledName,
                        moduleBuffer, *impl->topLevelModuleState, loc,
                        impl->moduleCachingLevel == ParserConfig::kCacheAll);
  return *state.decl;
}

ASTDecl &SharedState::createPackage(StringRef path, StringRef name) {
  auto fileLoc =
      FileLineColLoc::get(getContext(), path, /*line=*/0, /*column=*/0);
  StringAttr mangledName = getMangledModuleName(getContext(), name);
  ModuleState &state =
      createPackageState(StringAttr::get(getContext(), name), mangledName, path,
                         *impl->topLevelModuleState, fileLoc);
  return *state.decl;
}

ASTDecl &SharedState::createBinaryPackage(StringRef path, StringRef name) {
  StringAttr mangledName = getMangledModuleName(getContext(), name);
  ModuleState &state = createBinaryPackageState(
      SMLoc(), mangledName, mangledName, path, *impl->topLevelModuleState);
  return *state.decl;
}

std::optional<std::string> SharedState::getModuleSourcePath(ASTDecl &module) {
  auto it = impl->moduleStates.find(&module);
  if (it == impl->moduleStates.end())
    return std::nullopt;
  return it->second->sourcePath;
}

bool SharedState::isModuleOrPackagePath(const std::filesystem::path &path) {
  // Handle source files.
  if (path.extension() == ".mojo" || path.extension() == ".🔥")
    return true;
  // Handle source packages.
  return Filesystem::isMojoSourcePackagePath(path);
}

SharedState::ModuleState &
SharedState::createModuleState(StringAttr declName, StringAttr mangledName,
                               const llvm::MemoryBuffer *moduleBuffer,
                               ModuleState &parentState, FileLineColLoc loc,
                               bool enableCaching) {
  Lexer lexer(diags, moduleBuffer);

  // Create a new decl for this module.
  auto moduleBuilder = parentState.decl->getDeclEndBuilder();
  Operation *fileOp =
      moduleBuilder.create<FileModuleOp>(loc, mangledName, declName);
  ASTDecl &moduleDecl = declResolver->addDecl(
      fileOp, lexer.getToken().getLoc(), declName, parentState.decl,
      lexer.getCursor(), LexerCursor::getEOF(moduleBuffer), /*indentation=*/-1);

  ModuleState &moduleState = parentState.insertNestedModule(
      mangledName,
      std::make_unique<ModuleState>(
          &moduleDecl, moduleBuffer->getBufferIdentifier(), enableCaching));
  impl->moduleStates[&moduleDecl] = &moduleState;

  // Auto-import the core language modules.
  if (useBuiltinModule)
    importBuiltinModules(moduleDecl);

  // Build a content hash for the module from its input buffer.
  llvm::BLAKE3 contentHash;
  contentHash.update(moduleBuffer->getBuffer());
  moduleState.contentHash = contentHash.final();

  // Resolve the dependencies of the module.
  size_t prevNumModules = impl->moduleStates.size() - 1;
  resolveModuleDependencies(moduleState, parentState.decl,
                            moduleBuffer->getBuffer());

  // If we aren't currently resolving dependencies, try to load all of the newly
  // imported modules from the cache. We delay cache loading while resolving
  // dependencies so that we properly handle recursively dependent modules.
  if (!impl->activelyResolvingModuleDeps) {
    SmallVector<ModuleState *> modulesToLoad;
    for (auto &[name, moduleState] :
         llvm::drop_begin(impl->moduleStates, prevNumModules)) {
      if (moduleState->decl->hasReferenceError)
        continue;
      if (llvm::any_of(moduleState->dependencies, [](ModuleState *dep) {
            return dep->decl->hasReferenceError;
          }))
        continue;
      modulesToLoad.push_back(moduleState);
    }
    loadModulesFromCache(modulesToLoad);
  }

  notifyListenerOnModuleDecl(moduleDecl, moduleDecl.getLoc());
  return moduleState;
}

SharedState::ModuleState &
SharedState::createPackageState(StringAttr declName, StringAttr mangledName,
                                StringRef packagePath, ModuleState &parentState,
                                FileLineColLoc loc) {
  // Create a new decl for this module.
  auto moduleBuilder = parentState.decl->getDeclEndBuilder();
  auto packageOp = moduleBuilder.create<PackageOp>(loc, mangledName, declName);
  ASTDecl &decl =
      declResolver->addDecl(packageOp, SMLoc(), declName, parentState.decl,
                            parentState.decl->getCursor(),
                            parentState.decl->getCursor(), /*indentation=*/-1);

  // Insert the newly created module state.
  ModuleState &moduleState = parentState.insertNestedModule(
      mangledName, std::make_unique<ModuleState>(&decl, packagePath));
  impl->moduleStates[&decl] = &moduleState;
  impl->packageStates[packageOp] = &moduleState;

  return moduleState;
}

SharedState::ModuleState &SharedState::createBinaryPackageState(
    SMLoc loc, StringAttr declName, StringAttr mangledName,
    StringRef packagePath, ModuleState &parentState) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> packageBuffer =
      llvm::MemoryBuffer::getFile(packagePath);
  if (!packageBuffer) {
    return createErrorModuleState(loc, mangledName, *parentState.decl,
                                  "unable to open package file '" +
                                      packagePath + "'");
  }
  StringRef packageBufferRef = (*packageBuffer)->getBuffer();

  // Read the cached package.
  Block *block = parentState.decl->getDeclEndBuilder().getBlock();
  std::unique_ptr<mlir::BytecodeReader> bytecodeReader;
  {
    CompilerTimeTraceScope timeScope("readBytecodeFile");
    auto sourceMgr = std::make_shared<llvm::SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(*packageBuffer), SMLoc());
    const llvm::MemoryBuffer *memoryBuf =
        sourceMgr->getMemoryBuffer(sourceMgr->getMainFileID());
    bytecodeReader = std::make_unique<mlir::BytecodeReader>(
        memoryBuf->getMemBufferRef(), impl->bytecodeParserContext,
        /*lazyLoad=*/true, sourceMgr);

    // Read in the cached bytecode.
    if (failed(bytecodeReader->readTopLevel(block))) {
      return createErrorModuleState(loc, mangledName, *parentState.decl,
                                    "unable to load package '" + packagePath +
                                        "'");
    }

    // Add the package path to the set of included files.
    impl->includedFiles.emplace_back(packagePath.str());
  }

  // Insert a new module decl.
  ASTDecl &decl =
      declResolver->addDecl(&block->back(), SMLoc(), declName, parentState.decl,
                            parentState.decl->getCursor(),
                            parentState.decl->getCursor(), /*indentation=*/-1);
  decl.loadedFromBytecode = true;
  decl.resolvedness = DeclResolvedness::signature;

  // Initialize the module state.
  ModuleState &moduleState = parentState.insertNestedModule(
      mangledName, std::make_unique<ModuleState>(&decl));
  impl->moduleStates[&decl] = &moduleState;
  moduleState.bytecodeReader = std::move(bytecodeReader);
  impl->packageStates[cast<PackageOp>(decl)] = &moduleState;

  // Set the content hash of the package to the parsed buffer.
  moduleState.contentHash = llvm::BLAKE3::hash(ArrayRef<uint8_t>(
      (const uint8_t *)packageBufferRef.data(), packageBufferRef.size()));
  return moduleState;
}

SharedState::ModuleState &
SharedState::createErrorModuleState(SMLoc loc, StringAttr mangledName,
                                    ASTDecl &errorContext,
                                    const Twine &errorMsg) {
  // If the error context hasn't already had an error, emit the provided
  // message.
  if (!std::exchange(errorContext.hasReferenceError, true))
    emitError(loc, errorMsg);

  // Check if we already have an error decl with this name.
  if (auto *it = impl->topLevelModuleState->nestedModules.lookup(mangledName))
    return *it;

  // Otherwise, create one.
  ASTDecl *decl =
      &declResolver->addErroneousDecl(mangledName, loc, impl->topLevelDecl);
  ModuleState &state = impl->topLevelModuleState->insertNestedModule(
      mangledName, std::make_unique<ModuleState>(decl));
  impl->moduleStates[state.decl] = &state;
  return state;
}

void SharedState::resolveModuleDependencies(ModuleState &moduleState,
                                            ASTDecl *parentDecl,
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
        if (auto importOp = dyn_cast<UnresolvedImportOp>(decl))
          dependencies.insert({importOp.getModuleNameAttr(), decl->getLoc()});
    }
    for (auto it : moduleDecl.unresolvedWildcardImports)
      dependencies.insert({it.first, it.second.first});
    return mlir::success();
  };

  // For a given textual buffer, we can cache what the dependent module names
  // are. Caching this prevents the need to actually parse the buffer when the
  // content of the module hasn't changed.
  if (impl->transformCache && moduleState.canCacheModule) {
    auto onCacheMiss = [&](Operation *op, WriteableBufferRef buf,
                           LLCL::AnyAsyncValueRef chain) {
      auto output = LLCL::AsyncValueRef<BufferRef>::allocate(runtime);
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
        llvm::support::endian::Writer writer(*buf, llvm::endianness::little);
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
    auto onCacheHit = [&](Operation *op, BufferRef buf) {
      const char *data = buf->getBufferStart();

      // Functor for reading a uint64_t from the cache buffer.
      auto readInt = [&]() -> uint64_t {
        return llvm::support::endian::readNext<
            uint64_t, llvm::endianness::little, llvm::support::unaligned>(data);
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
    WriteableBufferRef keyBuf = WriteableBuffer::get();
    keyBuf->write_impl((const char *)moduleState.contentHash.data(),
                       moduleState.contentHash.size());
    options.print(*keyBuf << "mojoParser(");
    *keyBuf << ", useBuiltins=" << useBuiltinModule
            << ", experimentalLifetimes=" << useExperimentalLifetimes()
            << ", parsingStdlib=" << parsingStandardLibrary << ")";
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
  size_t prevNumModules = impl->moduleStates.size() - 1;

  // Import all of the dependencies, so that we can resolve their dependencies.
  for (auto [name, loc] : dependencies) {
    moduleState.dependencies.insert(
        &importModuleState(name.getValue(), parentDecl, loc));
  }

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
    for (auto &it : llvm::drop_begin(impl->moduleStates, prevNumModules)) {
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
  CompilerTimeTraceScope timeScope("cacheParsedModules");

  SmallVector<LLCL::AnyAsyncValueRef> results;
  for (auto &[decl, module] : impl->moduleStates) {
    if (!module->canCacheModule || decl->loadedFromBytecode)
      continue;
    FileModuleOp moduleOp =
        dyn_cast_if_present<FileModuleOp>(module->decl->getIfOperation());
    if (!moduleOp)
      continue;

    // Re-check if the module is in the cache. If it isn't, we populate it
    // now.
    BufferRef keyBuffer = module->buildCacheKey(options);
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
          CompilerTimeTraceScope timeScope(
              ("Caching: " + moduleOp.getName()).str());

          // Write the module to the cache.
          auto writeableTransformResult = WriteableBuffer::get();
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

/// Function used to look up and resolve a decl with the given mangled name.
static ASTDecl *lookupAndResolveMangledDecl(SharedState &shared,
                                            StringAttr leafRef, SMLoc loc,
                                            ASTDecl &container,
                                            DeclResolvedness howResolved) {
  // Find the operation in the symbol table of its container.
  auto declOp = shared.lookupSymbolIn<ASTDeclInterface>(&container, leafRef);
  if (!declOp)
    return nullptr;
  // Retrieve the proper decl name.
  StringAttr name = declOp.getDeclName();
  // Perform the lookup.
  LookupResult result = shared.lookupAndResolveDecl(
      name, loc, container, /*searchParentScopes=*/false);

  // Find the entry that matches the full symbol name.
  for (ASTDecl *decl : result.getIfSuccess()) {
    if (decl->getIfOperation() != declOp)
      continue;
    if (failed(shared.declResolver->resolve(*decl, howResolved, loc)))
      return nullptr;
    return decl;
  }
  llvm::report_fatal_error(
      "expected decl in symbol table to appear in lookup: " + name.getValue());
  return nullptr;
}

/// Builds an attribute/type walker to resolve references originating from
/// bytecode decls.
static mlir::AttrTypeWalker
buildBytecodeDeclReferenceResolver(SharedState &shared, ASTDecl &decl) {
  mlir::AttrTypeWalker walker;
  SMLoc loc = decl.getLoc();

  // Given a symbol reference, this functor fully resolves the parents of the
  // symbol assuming that the parent references do not contain any mangling.
  auto resolveParents = [&shared, loc](SymbolRefAttr symbol) -> ASTDecl * {
    // Resolve the top-level container for the reference. This should be a
    // package or module.
    StringRef moduleName = symbol.getRootReference();
    assert(moduleName.starts_with("$") &&
           "expected all references to be bound to a module/package");
    ASTDecl *decl = &shared.importModule(moduleName.drop_front(),
                                         /*currentPackage=*/nullptr, loc);
    if (decl->hasReferenceError ||
        failed(shared.declResolver->resolveFully(*decl, loc)))
      return {};
    ArrayRef<FlatSymbolRefAttr> nestedRefs = symbol.getNestedReferences();
    for (FlatSymbolRefAttr name : nestedRefs.drop_back()) {
      if (!(decl = lookupAndResolveMangledDecl(shared, name.getAttr(), loc,
                                               *decl, DeclResolvedness::fully)))
        return {};
    }
    return decl;
  };

  walker.addWalk([=, &shared](SymbolConstantAttr funcRef) -> WalkResult {
    ASTDecl *moduleDecl = resolveParents(funcRef.getSymbol());
    if (!moduleDecl)
      return WalkResult::interrupt();
    if (lookupAndResolveMangledDecl(shared,
                                    funcRef.getSymbol().getLeafReference(), loc,
                                    *moduleDecl, DeclResolvedness::fully))
      return WalkResult::advance();
    return WalkResult::interrupt();
  });

  auto visitTypeRef = [=, &shared](auto typeRef) -> WalkResult {
    ASTDecl *moduleDecl = resolveParents(typeRef.getSymbol());
    if (!moduleDecl)
      return WalkResult::interrupt();
    // Resolve the base type.
    StringAttr leaf = typeRef.getSymbol().getLeafReference();
    if (!lookupAndResolveMangledDecl(shared, leaf, loc, *moduleDecl,
                                     DeclResolvedness::signature))
      return WalkResult::interrupt();
    return WalkResult::advance();
  };

  walker.addWalk([=](DeclRefType typeRef) { return visitTypeRef(typeRef); });
  walker.addWalk([=](MetaTypeType typeRef) { return visitTypeRef(typeRef); });
  walker.addWalk([=](TraitType typeRef) { return visitTypeRef(typeRef); });

  return walker;
}

LogicalResult
SharedState::resolveDeclFromBytecode(ASTDecl &decl,
                                     DeclResolvedness resolvedness) {
  Operation *declOp = decl.getIfOperation();

  // Collect the referenced types that need to be resolved.
  mlir::AttrTypeWalker typeWalker =
      buildBytecodeDeclReferenceResolver(*this, decl);
  auto resolveTypes = [&](TypeRange types) {
    for (Type type : types)
      typeWalker.walk<mlir::WalkOrder::PreOrder>(type);
  };

  // Handle resolving the signature of the decl.
  if (decl.resolvedness < DeclResolvedness::signature) {
    decl.resolvedness = DeclResolvedness::signature;

    LogicalResult result =
        llvm::TypeSwitch<Operation *, LogicalResult>(declOp)
            .Case([&](LIT::FuncOp funcOp) {
              declResolver->declForFuncSymbol[decl.getSymbolRef()] = &decl;

              // Resolve the references from the signature.
              typeWalker.walk<mlir::WalkOrder::PreOrder>(
                  declOp->getAttrDictionary());
              return success();
            })
            .Case([&](StructDeclOp structOp) {
              // Resolve the types of any parameters.
              typeWalker.walk<mlir::WalkOrder::PreOrder>(
                  structOp.getParamsAttr());
              typeWalker.walk<mlir::WalkOrder::PreOrder>(
                  structOp.getParentTypesAttr());
              if (TypeAttr nmTarget = structOp.getNonmaterializableTargetAttr())
                typeWalker.walk<mlir::WalkOrder::PreOrder>(nmTarget);
              return success();
            })
            .Case([&](TraitDeclOp traitOp) {
              // TODO(traits): Resolve parameter types, when they exist.
              return success();
            })
            .Case([&](UnresolvedImportOp unresolvedImport) {
              // Let the normal decl resolver handling insert aliases and other
              // import behavior.
              if (failed(
                      declResolver->resolveSignature(unresolvedImport, decl)))
                return failure();
              return mlir::success();
            })
            .Case([&](GlobalVarDeclOp varDecl) {
              typeWalker.walk<mlir::WalkOrder::PreOrder>(varDecl.getType());
              return mlir::success();
            })
            .Case([&](AliasDeclOp aliasDecl) {
              typeWalker.walk<mlir::WalkOrder::PreOrder>(aliasDecl.getType());
              typeWalker.walk<mlir::WalkOrder::PreOrder>(aliasDecl.getValue());
              return mlir::success();
            })
            .Case([&](StructFieldOp field) {
              typeWalker.walk<mlir::WalkOrder::PreOrder>(field.getType());
              return mlir::success();
            })
            .Default([](auto) { return mlir::success(); });
    if (failed(result))
      return failure();
  }
  if (resolvedness < DeclResolvedness::fully)
    return success();
  decl.resolvedness = DeclResolvedness::fully;

  // Start body resolution by materializing the regions of this operation from
  // the bytecode reader. To materialize, we need to resolve the bytecode reader
  // from the parent module.
  mlir::BytecodeReader *bytecodeReader = nullptr;
  ASTDecl *parentDecl = &decl;
  do {
    if (!isa<FileModuleOp, PackageOp>(*parentDecl))
      continue;

    ModuleState *moduleState = impl->moduleStates[parentDecl];
    if (moduleState->bytecodeReader) {
      bytecodeReader = &*moduleState->bytecodeReader;
      break;
    }
  } while ((parentDecl = parentDecl->parentDecl));
  assert(bytecodeReader && "bytecode decl doesn't have a bytecode reader");

  // Functor used to resolve references within a single operation.
  auto resolveSingleOp = [&](Operation *op) -> WalkResult {
    if (bytecodeReader->isMaterializable(op) &&
        failed(bytecodeReader->materialize(op)))
      return failure();

    for (Region &region : op->getRegions())
      for (Block &block : region)
        resolveTypes(block.getArgumentTypes());
    resolveTypes(op->getOperandTypes());
    resolveTypes(op->getResultTypes());
    typeWalker.walk<mlir::WalkOrder::PreOrder>(op->getAttrDictionary());
    return mlir::success();
  };

  // If this isn't a container op, we don't need to resolve any nested decls,
  // simply materialize everything nested within.
  if (!isa<FileModuleOp, PackageOp, StructDeclOp, TraitDeclOp>(declOp)) {
    return failure(declOp->walk<mlir::WalkOrder::PreOrder>(resolveSingleOp)
                       .wasInterrupted());
  }

  // Functor to build a decl for a nested operation.
  auto addDeclForOp = [&](Operation *op, StringAttr name) -> ASTDecl & {
    ASTDecl &newDecl = declResolver->addDecl(
        DeclIRValue(op), decl.getLoc(), name, &decl, decl.getCursor(),
        decl.getCursor(), /*indentation=*/-1);
    newDecl.loadedFromBytecode = true;
    return newDecl;
  };

  // If this decl is a package, this is its corresponding module state.
  ModuleState *packageState = nullptr;
  if (isa<PackageOp>(declOp))
    packageState = impl->moduleStates[&decl];

  // Materialize the body of the decl.
  if (bytecodeReader->isMaterializable(declOp)) {
    if (failed(bytecodeReader->materialize(declOp)))
      return failure();
  }

  // Process the parsed region bodies, generating any necessary nested decls.
  SmallVector<Operation *> deferredOps;
  for (Region &region : declOp->getRegions()) {
    for (Operation &op : region.getOps()) {
      TypeSwitch<Operation *>(&op)
          .Case([&](LIT::FuncOp op) { addDeclForOp(op, op.getDeclName()); })
          .Case([&](UnresolvedImportOp op) {
            addDeclForOp(op, op.getImportNameAttr());
          })
          .Case([&](UnresolvedWildcardImportOp op) {
            decl.addUnresolvedWildCardImport(op.getModuleNameAttr(),
                                             op.getFullImport(), decl.getLoc());
          })
          .Case([&](StructDeclOp op) {
            ASTDecl &structDecl = addDeclForOp(op, op.getSymNameAttr());
            structDecl.setSelfType(ASTDecl::computeSelfTypeForStruct(op));
            for (ParamDeclAttr param : op.getParams()) {
              // Add the parameters as accessible member decls. Make sure
              // to demangle the parameter name.
              declResolver->addFullyResolvedDecl(
                  PValue(ParamDeclRefAttr::get(param)),
                  demangleParameterName(param.getName()), structDecl.getLoc(),
                  &structDecl);
            }
          })
          .Case([&](TraitDeclOp op) {
            ASTDecl &traitDecl = addDeclForOp(op, op.getSymNameAttr());
            traitDecl.setSelfType(ASTDecl::computeSelfTypeForTrait(op));
            // TODO(traits): Add decls for parameters, when they exist.
          })
          .Case([&](AliasDeclOp op) {
            addDeclForOp(op, StringAttr::get(op.getContext(),
                                             demangleParameterName(
                                                 op.getParamDecl().getName())));
          })
          .Case([&](StructFieldOp op) { addDeclForOp(op, op.getNameAttr()); })
          .Case([&](GlobalVarDeclOp op) {
            addDeclForOp(op, op.getSymNameAttr());
          })
          .Case<FileModuleOp, PackageOp>([&](auto op) {
            assert(packageState &&
                   "FileModule or Package nested in non-package");
            StringAttr name = op.getNameAttr();
            ASTDecl &decl = addDeclForOp(op, name);

            // Alias this without the `$` to allow users to resolve this nested
            // package/module using the name.
            packageState->decl->declsInScope.insert(
                {StringAttr::get(getContext(), name.getValue().drop_front()),
                 {&decl}});

            // Record a nested module state for this decl.
            ModuleState &moduleState = packageState->insertNestedModule(
                name, std::make_unique<ModuleState>(&decl));
            moduleState.contentHash = packageState->contentHash;

            impl->moduleStates[&decl] = &moduleState;
            if constexpr (std::is_same_v<decltype(op), PackageOp>)
              impl->packageStates[op] = &moduleState;
          })
          .Default([&](Operation *op) { deferredOps.push_back(op); });
    }
  }

  // Resolve references within the deferred operations. These don't have
  // corresponding decls, so we manually resolve them now.
  for (Operation *op : deferredOps)
    if (op->walk(resolveSingleOp).wasInterrupted())
      return failure();

  // After processing the region, make sure any non-signature attributes get
  // resolved.
  typeWalker.walk<mlir::WalkOrder::PreOrder>(declOp->getAttrDictionary());
  return success();
}

LogicalResult SharedState::finalizeImportedBytecodeModules() {
  for (ASTDecl *decl : declResolver->parsedDeclList) {
    if (!decl->loadedFromBytecode ||
        decl->resolvedness != DeclResolvedness::unparsed)
      continue;

    // Clear out decls that weren't materialized to avoid dangling references
    // after they get deleted.
    decl->setIRValue(PValue(BoolAttr::get(getContext(), false)));
  }
  for (auto &module : llvm::make_second_range(impl->moduleStates)) {
    if (!module->bytecodeReader)
      continue;

    // Finalize the bytecode, deleting any operations that weren't materialized.
    if (failed(module->bytecodeReader->finalize(
            [&](Operation *op) { return false; })))
      return failure();
  }
  return success();
}

ArrayRef<std::string> SharedState::getIncludedFiles() const {
  return impl->includedFiles;
}

void SharedState::traverseImportDirectories(
    unsigned importBufferFileId,
    function_ref<WalkResult(StringRef)> callback) const {
  // Python has lots of magic rules surrounding how modules get resolved. For
  // now, we just use the available include directories within the source
  // manager and the working directory of where the module is included.
  SourceMgr &sourceMgr = getSourceMgr();

  // Check the auto import directory first.
  for (auto &rawPath : impl->autoImportDirs) {
    if (callback(rawPath).wasInterrupted())
      return;

    // Cannot find the file, then check child directories of the auto import
    // directory.
    std::error_code ec;
    for (auto &childDir :
         std::filesystem::recursive_directory_iterator(rawPath, ec)) {
      if (ec)
        continue;
      // Skip non-directories and source packages, internal packages should be
      // imported using a relative import.
      if (!childDir.is_directory() ||
          Filesystem::isMojoSourcePackagePath(childDir.path()))
        continue;
      if (callback(childDir.path().string()).wasInterrupted())
        return;
    }
  }

  // Check the working directory.
  if (importBufferFileId) {
    const auto *includeBuffer = sourceMgr.getMemoryBuffer(importBufferFileId);
    std::filesystem::path includerPath(
        includeBuffer->getBufferIdentifier().str());

    // Use the top-most non-package directory.
    do {
      includerPath = includerPath.parent_path();
    } while (Filesystem::isMojoSourcePackagePath(includerPath));

    if (callback(includerPath.string()).wasInterrupted())
      return;
  }

  // Check the include directories.
  for (StringRef includeDir : getSourceMgr().getIncludeDirs())
    if (callback(includeDir).wasInterrupted())
      return;
}

DebugInfo::SourceNameAttr
SharedState::getSourceName(mlir::SymbolOpInterface op) {
  return impl->sourceNames.getSourceName(op);
}

/// Given a valid pointer into a source buffer for some token, return the
/// length of the token by re-lex'ing it.  This is efficient.
static size_t getTokenLength(SharedState &shared, SMLoc loc) {
  // Because we know the pointer is to a valid place in a source buffer, and
  // because we know that all source buffers are NUL terminated, we know that
  // the end of buffer check isn't needed.  This allows us to form a lexer
  // without having to find the MemoryBuffer it came from, saving some expense
  // in diagnostic emission.
  const char *curPtr = loc.getPointer();

  // If the byte is NUL, it is an invalid token and might be end of buffer.
  if (*curPtr == '\0')
    return 0;

  Lexer lexer(shared.diags, StringRef(curPtr, ~0ULL), curPtr);
  return lexer.getToken().getSpelling().size();
}

/// Given a pointer to the start of a token, find the end of it.
static void adjustTokenEndPoint(SharedState &shared, SMLoc &loc) {
  size_t tokenSize = getTokenLength(shared, loc);
  loc = SMLoc::getFromPointer(loc.getPointer() + tokenSize);
}

LIT::StructDeclOp SharedState::getOrCreateClosureWrapper(SMLoc loc,
                                                         SignatureType sig,
                                                         ASTDecl *moduleDecl) {
  if (sig.getNumResultParams()) {
    emitError(loc, "result parameters in closures are not supported yet");
    return {};
  }

  auto fileModuleOp = cast<FileModuleOp>(moduleDecl);
  std::pair<SignatureType, StringAttr> key(sig, fileModuleOp.getSymNameAttr());
  StructDeclOp existing = impl->closureWrappers[key];
  if (!existing) {
    std::string name =
        ASTType(sig).getAsString(/*forDiag=*/true, /*demangleParams=*/true);
    ClosureEmitter emitter(*moduleDecl, *this);
    existing = emitter.createClosureWrapperStructDecl(
        StringAttr::get(getContext(), name), sig, loc);
    impl->closureWrappers[key] = existing;
  }
  return existing;
}

const llvm::MapVector<ASTDecl *, Capture> &
SharedState::getCaptureRangeInScope(ASTDecl &scope) {
  return getImpl().capturesInScope[&scope];
}

void SharedState::addCaptureToScope(ASTDecl &scope, ASTDecl *captureDecl,
                                    Capture capture) {
  getImpl().capturesInScope[&scope].insert({captureDecl, capture});
  if (captureDecl->getParentDecl() != scope.parentDecl) {
    ASTDecl *parentDecl = scope.getNearestDeclOfType<LIT::FuncOp>();
    if (parentDecl)
      addCaptureToScope(*scope.parentDecl, captureDecl, capture);
  }
}

//===----------------------------------------------------------------------===//
// Listener Interface

/// Resolve the given decl in preparation for passing it to the listener for
/// member lookup.
static void resolveDeclForListenerLookup(DeclResolver &declResolver,
                                         ASTDecl &decl, SMLoc loc) {
  // Before passing off to the listener, resolve nested decls. This lets the
  // listener see the full set of declarations, as unresolved imports are
  // generally lazily resolved, and also ensures the availability of things like
  // documentation.
  if (failed(declResolver.resolveFully(decl, loc)))
    return;
  const llvm::MapVector<StringAttr, TinyPtrVector<ASTDecl *>> &decls =
      decl.getDeclsInScope();
  for (int i = 0, e = decls.size(); i < e; ++i) {
    // Resolution may invalidate the decls vector, so we can't rely on
    // iterators here. We also don't fail, because the listener should be
    // tolerant to errors.
    auto &[name, children] = *std::next(decls.begin(), i);
    (void)declResolver.resolveFully(*children.front(), loc);
  }
  // Resolve any pending wildcards in the decl. We don't care about failure
  // here, as we still want to enable lookup for the decls that could be
  // resolved.
  (void)declResolver.resolveAllWildcardImports(decl);
}

/// Return if the given parser listner is interested in the given location.
static bool isListenerInterestedInLoc(ParserListener *listener, SMLoc loc) {
  return listener && listener->isInterestedInLoc(loc);
}

void SharedState::notifyListenerOnAliasDecl(ASTDecl &decl,
                                            SMLoc identifierLoc) {
  if (isListenerInterestedInLoc(parserListener, identifierLoc))
    parserListener->onAliasDecl(&decl, identifierLoc);
}

void SharedState::notifyListenerOnArgumentDecl(ASTDecl &decl,
                                               SMLoc identifierLoc) {
  if (isListenerInterestedInLoc(parserListener, identifierLoc))
    parserListener->onArgumentDecl(&decl, identifierLoc);
}

void SharedState::notifyListenerOnFunctionDecl(ASTDecl &decl,
                                               SMLoc identifierLoc) {
  if (isListenerInterestedInLoc(parserListener, identifierLoc))
    parserListener->onFunctionDecl(&decl, identifierLoc);
}

void SharedState::notifyListenerOnImport(SMLoc importLoc) {
  if (isListenerInterestedInLoc(parserListener, importLoc))
    parserListener->onImport(importLoc);
}

void SharedState::notifyListenerOnImport(
    SMLoc importLoc, function_ref<ASTDecl &()> getPackageDecl) {
  if (!isListenerInterestedInLoc(parserListener, importLoc))
    return;
  parserListener->onImport(
      [&]() -> ASTDecl * {
        ASTDecl &packageDecl = getPackageDecl();
        resolveDeclForListenerLookup(*declResolver, packageDecl, importLoc);
        return &packageDecl;
      },
      importLoc);
}

void SharedState::notifyListenerOnMemberLookup(ASTDecl &decl, SMLoc lookupLoc,
                                               bool searchParentScopes) {
  if (!isListenerInterestedInLoc(parserListener, lookupLoc))
    return;
  parserListener->onMemberLookup(
      [&]() -> ASTDecl * {
        resolveDeclForListenerLookup(*declResolver, decl, lookupLoc);

        // Resolve parent scopes if necessary.
        if (searchParentScopes) {
          ASTDecl *parentDecl = &decl;
          while ((parentDecl = parentDecl->getParentDecl()))
            resolveDeclForListenerLookup(*declResolver, *parentDecl, lookupLoc);
        }
        return &decl;
      },
      lookupLoc, searchParentScopes);
}

void SharedState::notifyListenerOnMemberLookup(
    SMLoc lookupLoc, function_ref<ASTDecl &()> getDeclFn,
    bool searchParentScopes) {
  if (isListenerInterestedInLoc(parserListener, lookupLoc))
    notifyListenerOnMemberLookup(getDeclFn(), lookupLoc, searchParentScopes);
}

void SharedState::notifyListenerOnModuleDecl(ASTDecl &decl,
                                             SMLoc identifierLoc) {
  // TODO: This hook should likely be removed in favor of just `onRef`. It's
  // used to index other modules for the sake of references, but we should just
  // handle this when we see the reference.
  if (parserListener)
    parserListener->onModuleDecl(&decl, identifierLoc);
}

void SharedState::notifyListenerOnModuleImport(ASTDecl &decl,
                                               StringRef spelling, SMLoc loc) {
  if (!isListenerInterestedInLoc(parserListener, loc))
    return;
  if (!decl.getIfOperation())
    return;
  // Grab the names of each of the referenced modules.
  SmallVector<StringRef> moduleNames;
  spelling.split(moduleNames, '.', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  // Skip over relative module markers in the location.
  const char *locPtr = loc.getPointer();
  while (*locPtr == '.')
    ++locPtr;
  loc = SMLoc::getFromPointer(locPtr);

  // Grab the decls for each of the referenced modules.
  SmallVector<ASTDecl *> decls;
  ASTDecl *declIt = &decl;
  for (int i = 0, e = moduleNames.size(); i < e; ++i) {
    decls.push_back(declIt);
    declIt = declIt->getParentDecl();
  }

  // Notify the listener of each module import starting from the parent, so we
  // can skip past the position within the location.
  for (auto [name, decl] : llvm::zip(moduleNames, llvm::reverse(decls))) {
    parserListener->onModuleImport(decl, name, loc);
    loc = SMLoc::getFromPointer(loc.getPointer() + name.size() + 1);
  }
}

void SharedState::notifyListenerOnParameterDecl(ASTDecl &decl,
                                                SMLoc identifierLoc) {
  if (isListenerInterestedInLoc(parserListener, identifierLoc))
    parserListener->onParameterDecl(&decl, identifierLoc);
}

void SharedState::notifyListenerOnStructDecl(ASTDecl &decl,
                                             SMLoc identifierLoc) {
  if (isListenerInterestedInLoc(parserListener, identifierLoc))
    parserListener->onStructDecl(&decl, identifierLoc);
}

void SharedState::notifyListenerOnStructFieldDecl(ASTDecl &decl,
                                                  SMLoc identifierLoc) {
  if (isListenerInterestedInLoc(parserListener, identifierLoc))
    parserListener->onStructFieldDecl(&decl, identifierLoc);
}

void SharedState::notifyListenerOnTraitDecl(ASTDecl &decl,
                                            SMLoc identifierLoc) {
  if (isListenerInterestedInLoc(parserListener, identifierLoc))
    parserListener->onTraitDecl(&decl, identifierLoc);
}

void SharedState::notifyListenerOnVariableDecl(ASTDecl &decl,
                                               SMLoc identifierLoc) {
  if (isListenerInterestedInLoc(parserListener, identifierLoc))
    parserListener->onVariableDecl(&decl, identifierLoc);
}

void SharedState::notifyListenerOnRef(ArrayRef<ASTDecl *> decls,
                                      StringRef spelling, SMLoc loc) {
  if (!loc.isValid())
    return;
  SMLoc endLoc = SMLoc::getFromPointer(loc.getPointer() + spelling.size());
  notifyListenerOnRef(decls, spelling, SourceRange::getByteLevel(loc, endLoc));
}

void SharedState::notifyListenerOnRef(ArrayRef<ASTDecl *> decls,
                                      StringRef spelling, SourceRange range) {
  if (isListenerInterestedInLoc(parserListener, range.getStart()))
    parserListener->onRef(decls, spelling, diags.convertToSMRange(range));
}

/// Return the location of the identifier in the given expression.
static SourceRange getIdentifierLocFromExpr(const ExprNode *expr) {
  if (auto attribute = dyn_cast<AttributeRefNode>(expr))
    return attribute->getAttributeNameRange();

  // For post-fix expression, ensure we get the location from the base, not the
  // operator.
  if (auto subscript = dyn_cast<SubscriptNode>(expr))
    return getIdentifierLocFromExpr(subscript->base);
  if (auto call = dyn_cast<CallNode>(expr))
    return getIdentifierLocFromExpr(call->callee);
  return expr->getRange();
}

void SharedState::notifyListenerOnRef(ArrayRef<ASTDecl *> decls,
                                      StringRef spelling,
                                      const ExprNode *expr) {
  notifyListenerOnRef(decls, spelling, getIdentifierLocFromExpr(expr));
}

/// Returns if the parser listener should be notified on references for the
/// given call syntax.
static bool shouldNotifyListenerForCall(CallSyntax syntax) {
  switch (syntax) {
  case CallSyntax::kDirectCall:
  case CallSyntax::kMethodCall:
  case CallSyntax::kAttribute:
    return true;
  case CallSyntax::kIndirectCall:
  case CallSyntax::kTypeCall:
  case CallSyntax::kOperator:
  case CallSyntax::kReversedOperator:
  case CallSyntax::kSubscript:
  case CallSyntax::kImplicitConvert:
  case CallSyntax::kDestructor:
  case CallSyntax::kTupleGetItem:
    return false;
  }
  llvm_unreachable("unknown call syntax");
}

void SharedState::notifyListenerOnRef(ArrayRef<ASTDecl *> decls,
                                      StringRef spelling, const ExprNode *expr,
                                      CallSyntax syntax) {
  if (shouldNotifyListenerForCall(syntax))
    notifyListenerOnRef(decls, spelling, expr);
}

void SharedState::notifyListenerOnCall(ArrayRef<ASTDecl *> decls,
                                       SMLoc rParenLoc,
                                       const CallOperands &callOperands) {
  if (isListenerInterestedInLoc(parserListener, rParenLoc))
    parserListener->onCall(decls, rParenLoc, callOperands);
}

void SharedState::notifyListenerOnParameterBinding(ArrayRef<ASTDecl *> decls,
                                                   llvm::SMLoc rsquareLoc,
                                                   ArrayRef<Operand> operands) {
  if (isListenerInterestedInLoc(parserListener, rsquareLoc)) {
    SmallVector<ExprNode *> parameters = llvm::map_to_vector(
        operands, [](const Operand &operand) { return operand.value; });
    parserListener->onParameterBinding(decls, rsquareLoc, parameters);
  }
}
