//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of the SharedState class.
//
//===----------------------------------------------------------------------===//

#include "CallEmission.h"
#include "ClosureEmitter.h"
#include "DebugInfo.h"
#include "ExprEmitter.h"
#include "ExprNodes.h"

#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoParser/IRValues.h"
#include "KGEN/MojoParser/SharedState.h"

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/Support/Configuration.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "KGEN/ToolCommon/InitAllDialects.h"

#include "Support/Buffer.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Configuration.h"
#include "Support/Filesystem/Paths.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/bit.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/SourceMgr.h"

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

//===----------------------------------------------------------------------===//
// BytecodeResolutionReferenceWalker
//===----------------------------------------------------------------------===//

namespace {
/// This class defines an attribute and type walker that resolves references to
/// decls defined within bytecode files.
class BytecodeResolutionReferenceWalker {
public:
  BytecodeResolutionReferenceWalker(SharedState &shared) : shared(shared) {}

  /// Set and save the context location for the current bytecode resolution.
  llvm::SaveAndRestore<SMLoc> saveResolutionContextLoc(SMLoc loc) {
    return llvm::SaveAndRestore<SMLoc>(this->resolutionContextLoc, loc);
  }

  /// Walk the given attribute or type element, resolving references found
  /// within.
  template <typename T>
  WalkResult walk(T element) {
    const void *key = element.getAsOpaquePointer();

    // Check if we've already walk this element before.
    auto it = visitedAttrTypes.find(key);
    if (it != visitedAttrTypes.end())
      return it->second;

    // Walk this element, bailing if skipped or interrupted.
    WalkResult walkResult = processBytecodeReferences(element);
    if (walkResult.wasInterrupted())
      return visitedAttrTypes[key] = WalkResult::interrupt();
    if (walkResult.wasSkipped())
      return WalkResult::advance();

    // Walk the sub-elements, checking for bytecode references.
    WalkResult result = WalkResult::advance();
    auto walkFn = [&](auto element) {
      if (element && !result.wasInterrupted())
        result = walk(element);
    };
    element.walkImmediateSubElements(walkFn, walkFn);

    visitedAttrTypes.try_emplace(key, WalkResult::advance());
    return result.wasInterrupted() ? result : WalkResult::advance();
  }
  void walkRange(TypeRange types) {
    for (Type type : types)
      walk(type);
  }

private:
  /// Given a symbol reference, fully resolve the parents of the symbol assuming
  /// that the parent references do not contain any mangling.
  ASTDecl *resolveRefParentDecl(SharedState &shared, SymbolRefAttr symbol) {
    // This is a reference to a top-level declaration.
    if (symbol.getNestedReferences().empty())
      return &shared.getTopLevelDecl();

    StringAttr rootAttr = symbol.getRootReference();
    auto nestedRefs = symbol.getNestedReferences().drop_back();
    auto it = resolvedSymbolParents.find({rootAttr, nestedRefs});
    if (it != resolvedSymbolParents.end())
      return it->second;

    // Resolve the top-level container for the reference. This should be a
    // package or module.
    ASTDecl *decl = &shared.importModule(rootAttr, /*currentPackage=*/nullptr,
                                         resolutionContextLoc);
    if (decl->isErroneous() ||
        failed(shared.declResolver->resolveBody(*decl, resolutionContextLoc)))
      return {};
    for (FlatSymbolRefAttr name : nestedRefs) {
      if (!(decl = shared.lookupAndResolveMangledDecl(
                name.getAttr(), resolutionContextLoc, *decl,
                DeclResolvedness::body)))
        return {};
    }
    resolvedSymbolParents.try_emplace({rootAttr, nestedRefs}, decl);
    return decl;
  }

  /// Resolve the reference to a bytecode decl represented by the given symbol.
  ASTDecl *resolveBytecodeReferenceSignature(SharedState &shared,
                                             SymbolRefAttr symbol) {
    ASTDecl *moduleDecl = resolveRefParentDecl(shared, symbol);
    if (!moduleDecl)
      return nullptr;
    return shared.lookupAndResolveMangledDecl(symbol.getLeafReference(),
                                              resolutionContextLoc, *moduleDecl,
                                              DeclResolvedness::signature);
  }

  /// Process the given attributes and types for bytecode references.
  WalkResult processBytecodeReferences(Attribute attr) {
    return TypeSwitch<Attribute, WalkResult>(attr)
        .Case([&](SymbolConstantAttr ref) {
          ASTDecl *decl =
              resolveBytecodeReferenceSignature(shared, ref.getSymbol());
          if (!decl)
            return failure();

          // Don't fully resolve containers, they'll get resolved if something
          // is needed from within them.
          if (isa_and_nonnull<FileModuleOp, PackageOp, StructDeclOp,
                              TraitDeclOp>(decl->getIfOperation()))
            return mlir::success();

          // Fully resolve every other decl.
          return shared.declResolver->resolveBody(*decl, resolutionContextLoc);
        })
        .Default(WalkResult::advance());
  }
  WalkResult processBytecodeReferences(Type type) {
    return TypeSwitch<Type, WalkResult>(type)
        .Case<StructMetaType, LIT::StructType>([&](auto ref) {
          return success(
              resolveBytecodeReferenceSignature(shared, ref.getSymbol()));
        })
        .Case<TraitType>([&](TraitType ref) {
          return success(
              llvm::all_of(ref.getSymbols(), [&](SymbolRefAttr symbol) -> bool {
                return resolveBytecodeReferenceSignature(shared, symbol);
              }));
        })
        .Default(WalkResult::advance());
  }

  /// The parent shared state.
  SharedState &shared;
  /// A mapping from the parent reference of a SymbolRefAttr to the
  /// corresponding resolved ASTDecl.
  DenseMap<std::pair<Attribute, ArrayRef<FlatSymbolRefAttr>>, ASTDecl *>
      resolvedSymbolParents;
  /// The current bytecode resolution context location.
  SMLoc resolutionContextLoc;
  /// The set of cached attributes/types from which nested references have
  /// already been either successfully or erroneously resolved.
  DenseMap<const void *, WalkResult> visitedAttrTypes;
};
} // namespace

//===----------------------------------------------------------------------===//
// SharedState
//===----------------------------------------------------------------------===//

struct SharedState::Impl {
  Impl(SharedState &shared)
      : sourceNames(shared),
        bytecodeParserContext(shared.getContext(), /*verifyAfterParse=*/false),
        bytecodeRefResolutionWalker(shared) {}
  virtual ~Impl() = default;

  /// This MLIR block is owned by SharedState, and vended to clients that have a
  /// need to build Arguments that are potentially unused.  This happens during
  /// function signature type checking, where the arguments are needed to
  /// satisfy lookup requests later in the signature, but where the body may not
  /// actually be generated.  If generated, the arguments are removed from this
  /// block and installed in the actual function.
  Block argumentOwningBlock;

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

  /// Flag indicating if the deps of a module are currently being resolved.
  bool activelyResolvingModuleDeps = false;

  /// Flag indicating if we should diagnose missing doc strings while parsing.
  bool diagnoseMissingDocStrings = false;

  /// Flag indicating if errors should be emitted instead of warnings for
  /// documentation issues.
  bool errorOnInvalidDocStrings = false;

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
  /// signature and module.
  DenseMap<std::pair<GeneratorType, ASTDecl *>, StructDeclOp> closureWrappers;

  /// The capture values and decls associated with their enclosing nested
  /// function.
  DenseMap<ASTDecl *, llvm::MapVector<ASTDecl *, Capture>> capturesInScope;

  /// Function type conversion thunks in each module.
  // The key is an ArrayAttr containing two elements:
  // - The "actual" signature; the type of the underlying function that the
  //   thunk is calling.
  // - The thunk signature, not including the `callee` input parameter (for some
  //   reason).
  //   This is NOT the expected/destination type we're converting to, it's the
  //   actual thunk's signature (this is so generateConversionThunk can know the
  //   "clarifying parameters", see TAPCPTTT).
  DenseMap<Attribute, FnOp> conversionThunks;

  /// This caches non-trivial implicit convertibility checks from one type to
  /// another.
  DenseMap<std::pair<Type, Type>, bool> cachedImplicitConvertibility;

  /// An attribute walker used to resolve bytecode references.
  BytecodeResolutionReferenceWalker bytecodeRefResolutionWalker;
};

/// Ensure `stripFilePrefix` is an absolute path ending in a separator.
static std::string canonicalizeFileCompilationDir(StringRef stripFilePrefix) {
  if (stripFilePrefix.empty())
    return {};

  SmallString<256> workingFileCompilationDir = stripFilePrefix;
  llvm::sys::path::remove_dots(workingFileCompilationDir,
                               /*remove_dot_dot=*/true);
  llvm::sys::fs::make_absolute(workingFileCompilationDir);
  if (!llvm::sys::path::is_separator(workingFileCompilationDir.back()))
    workingFileCompilationDir.append(llvm::sys::path::get_separator());
  return workingFileCompilationDir.str().str();
}

SharedState::SharedState(llvm::SourceMgr &sourceMgr, ParserConfig &config)
    : diags(sourceMgr, config.context, config.useMLIRDiagnostics,
            config.maxNotesPerDiagnostic,
            canonicalizeFileCompilationDir(config.stripFilePrefix),
            /* disableWarnings */ config.options.disableWarnings,
            /*extraContext*/ this),
      options(config.options),
      declResolver(std::make_unique<DeclResolver>(*this)),
      parserListener(config.parserListener),
      disablePrebuiltPackages(config.disablePrebuiltPackages),
      useBuiltinModule(config.useBuiltinModule),
      exportKgenModule(config.exportKgenModule),
      impl(std::make_unique<Impl>(*this)) {
  if (!options.searchPaths.empty()) {
    SmallVector<StringRef> paths;
    StringRef(options.searchPaths)
        .split(paths, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    llvm::append_range(impl->autoImportDirs, paths);
  } else {
    collectDefaultImportPaths(impl->autoImportDirs);
  }
  impl->diagnoseMissingDocStrings = config.diagnoseMissingDocStrings;
  impl->errorOnInvalidDocStrings = config.errorOnInvalidDocStrings;

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
      [this](SMLoc &loc) { adjustTokenEndPoint(*this, loc); });

  if (options.getDebugInfoLevelForInput() > CompilationOptions::kSynthetic) {
    diBuilder = std::make_unique<DebugInfo::DIBuilder>(config.context);

    diBuilder->initializeCompileUnit(
        options.debugInfoLanguage,
        diBuilder->createFile(diags.getBufferNameIdentifier()), "Mojo",
        /*isOptimized=*/true, options.getDIEmissionKind());
  }
}

SharedState::~SharedState() { declResolver.reset(); }

bool SharedState::shouldExportKgenModule() const { return exportKgenModule; }

bool SharedState::shouldDiagnoseMissingDocStrings() const {
  return impl->diagnoseMissingDocStrings;
}

bool SharedState::shouldErrorOnInvalidDocStrings() const {
  return impl->errorOnInvalidDocStrings;
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
  builtinsDecl.resolvedness = DeclResolvedness::body;

  // The outermost scope contains all of the __builtins__ function definitions.
  for (auto &[name, decls] : builtinsDecl.getDeclsInScope())
    declResolver->aliasDecls(decls, name, topLevelDecl.getLoc(), topLevelDecl);

  // Top level is fully resolved now.
  topLevelDecl.resolvedness = DeclResolvedness::body;
}

/// Shared state maintains an MLIR Block and deallocates it when the parser is
/// torn down.  This can be used to allocate BlockArgument's that may or may
/// not get used in the future.
Block &SharedState::getArgumentOwningBlock() {
  return impl->argumentOwningBlock;
}

void SharedState::deleteDecl(ASTDecl &decl) {
  std::optional<StringRef> name = decl.getNameIfOperation();
  if (!name)
    return;
  Operation *op = decl.getIfOperation();

  // Remove from global maps.
  // Func needs a special case since it may or may not be a symbol.
  if (auto func = dyn_cast<FnOp>(op)) {
    if (SymbolRefAttr sym = decl.getSymbolRef())
      declResolver->declForFuncSymbol.erase(sym);
    impl->sourceNames.forgetSourceName(func);
  } else if (auto symbolDecl = dyn_cast<mlir::SymbolOpInterface>(op)) {
    if (SymbolRefAttr sym = decl.getSymbolRef())
      declResolver->declForTypeSymbol.erase(sym);
    impl->sourceNames.forgetSourceName(symbolDecl);
    impl->symbolTables.getSymbolTable(op->getParentOp()).remove(op);
  }
  op->erase();

  // Set the IRValue to nullptr, so that any reference pointing to the decl can
  // check if it's valid.
  decl.setIRValue(nullptr);
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

FileLineColLoc SharedState::createLocation(StringRef filename, unsigned line,
                                           unsigned column) {
  return FileLineColLoc::get(getContext(), diags.getCanonicalFilename(filename),
                             line, column);
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
  impl->noneType = KGEN::NoneType::get(context);
  impl->noneAttr = NoneAttr::get(context);

  // Make the type check error type.  Anything that references this will
  // considering it erroneous and already declared as such.
  impl->typeCheckErrorType = TypeCheckErrorType::get(context);

  // Add an empty struct with the specified name to the resolver.
  auto anyRegTypeType = TypeType::get(getContext());
  auto addMagicMLIRDecl = [&](StringRef name, Type magicType) {
    TypedAttr value = TypeParamAttr::get(magicType, anyRegTypeType);
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

  [[maybe_unused]] auto newName = symTab.insert(declOp);
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
ArrayRef<ExprNode *> ASTDecl::getBodyDecorators() const {
  if (!hasBodyDecorators)
    return {};
  return shared.getImpl().bodyDecorators[this];
}

/// During signature resolution, this is called with any decorators that need
/// to persist until body resolution.
void ASTDecl::setBodyDecorators(ArrayRef<ExprNode *> decorators) {
  if (decorators.empty())
    return;

  shared.getImpl().bodyDecorators.insert({this, decorators.vec()});
  hasBodyDecorators = true;
}

//===----------------------------------------------------------------------===//
// ModuleState
//===----------------------------------------------------------------------===//

struct SharedState::ModuleState {
  ModuleState(ASTDecl *decl = nullptr) : decl(decl) {}
  ModuleState(ASTDecl *decl, StringRef sourcePath)
      : decl(decl), sourcePath(sourcePath.str()) {}
  ~ModuleState() {
    // Drop any remaining operations in the reader to avoid dangling
    // unmaterialized operations. If these were needed, they would have been
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
  /// A temporary module used to load the bytecode.
  ModuleOp tmpModule;
  /// The optional source path of this module if it was loaded from source.
  std::optional<std::string> sourcePath;

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
  if (failed(declResolver->resolveBody(scope, loc)))
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
    if (!scope.unresolvedWildcardImports)
      return {};

    for (size_t i = 0, e = scope.unresolvedWildcardImports->size(); i < e;) {
      auto it = std::next(scope.unresolvedWildcardImports->begin(), i);
      auto [moduleName, locAndIsFullImport] = *it;
      auto [loc, isFullImport] = locAndIsFullImport;

      // Don't try wildcard imports if we wouldn't import this name anyways.
      if (!isFullImport && name.starts_with("_")) {
        ++i;
        continue;
      }
      --e;
      scope.unresolvedWildcardImports->erase(it);

      // Resolve the import. If it fails, don't fail the search immediately,
      // keep checking for something that can resolve the decl we care about.
      if (succeeded(declResolver->importWildCardDeclsFromModule(
              scope, moduleName, isFullImport, loc))) {
        // Re-check the lookup in the scope now that the wildcard import has
        // been resolved.
        result = scope.lookupInCurrentScope(nameAttr);
        if (!result.empty())
          return result;
      }
      e = scope.unresolvedWildcardImports->size();
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
        if (isa<StructDeclOp>(*curSearchScope) &&
            !(*e.front()).getIfIRValue().getIfPValue()) {
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

/// Resolve the absolute path for a given module name within the provided
/// directory. Returns nullopt if the module cannot be found.
static std::optional<std::string> resolveModulePath(SharedState &shared,
                                                    llvm::SMLoc includeLoc,
                                                    StringRef moduleName,
                                                    StringRef includeDir,
                                                    bool ignorePrebuilt) {
  using namespace std::filesystem;

  // Find a path in `includeDir` that is a mojo package for `moduleName`. This
  // is either a directory with an `__init__.mojo` file inside it, a
  // `moduleName.mojo` file, or a `moduleName.mojopkg` file. Of course, the
  // emoji extensions are supported as well, but a conflict is not allowed. Make
  // sure to ignore other `moduleName.*` files that are definitely not mojo
  // packages.
  std::error_code ec;
  auto iter = directory_iterator(includeDir.str(), ec);
  if (ec)
    return std::nullopt;

  // Gets the name of the file or directory in a case sensitive way. On non-case
  // sensitive systems we cannot just do `path / moduleName` since the
  // constructed path will not adhere to case sensitivity.
  std::optional<path> nameOr;
  path source, emoji;
  auto emitConflictError = [&] {
    shared.emitError(includeLoc, "ambiguous import, both ")
        << source.string() << " and " << emoji.string()
        << " exist in the file system.";
  };
  for (const auto &entry : iter) {
    if (entry.path().filename().stem().string() != moduleName)
      continue;

    // If we found a package path, we can return immediately.
    if (Filesystem::isMojoSourcePackagePath(entry.path())) {
      if (exists(source = entry.path() / "__init__.mojo", ec) &&
          exists(emoji = entry.path() / "__init__.🔥", ec))
        emitConflictError();
      return std::filesystem::absolute(entry.path());
    }

    path ext = entry.path().filename().extension();
    if (!ignorePrebuilt && (ext == ".mojopkg" || ext == ".📦")) {
      if (exists(source = path(entry.path()).replace_extension("mojopkg"),
                 ec) &&
          exists(emoji = path(entry.path()).replace_extension("📦"), ec))
        emitConflictError();
      return std::filesystem::absolute(entry.path());
    }
    if (ext == ".mojo" || ext == ".🔥") {
      if (exists(source = path(entry.path()).replace_extension("mojo"), ec) &&
          exists(emoji = path(entry.path()).replace_extension("🔥"), ec))
        emitConflictError();
      return std::filesystem::absolute(entry.path());
    }
  }

  // If we cannot find a file or directory with the case-sensitive name, then
  // return early.
  return std::nullopt;
}

/// Resolve the absolute path for a given module name. Returns nullopt if the
/// module cannot be found.
std::optional<std::string> SharedState::resolveModulePath(StringRef moduleName,
                                                          SMLoc includeLoc) {
  unsigned includeBufferId = getSourceMgr().FindBufferContainingLoc(includeLoc);

  std::optional<std::string> result;
  traverseImportDirectories(includeBufferId, [&](StringRef dir) {
    // Don't try to resolve modules that reside within a package.
    if (Filesystem::isMojoSourcePackagePath(dir.str())) {
      // TODO: It'd be nice to emit a list of potential modules that the
      // name might correspond with if it did resolve to one inside of this
      // package.
      return WalkResult::advance();
    }
    if ((result = ::resolveModulePath(*this, includeLoc, moduleName, dir,
                                      disablePrebuiltPackages)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result;
}

/// Given a path to a mojo source package, return the path of its __init__ file.
static std::string
getPackageInitPath(const std::filesystem::path &packagePathStr) {
  std::filesystem::path initPath = packagePathStr / "__init__.🔥";
  std::error_code ec;
  if (std::filesystem::exists(initPath, ec))
    return initPath.string();
  return (packagePathStr / "__init__.mojo").string();
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
  return importSubModuleState(name, impl->topLevelDecl, loc, loc);
}

SharedState::ModuleState &
SharedState::importSubModuleState(StringRef name, ASTDecl *parentDecl,
                                  llvm::SMLoc loc, llvm::SMLoc identifierLoc) {

  // Grab the parent module state.
  ModuleState *parentState = impl->moduleStates[parentDecl];
  assert(parentState && "parent decl must have a module state");

  // Check to see if we've already imported this module.
  auto declName = StringAttr::get(getContext(), name);
  auto it = parentState->nestedModules.find(declName);
  if (it != parentState->nestedModules.end())
    return *it->second;

  // Resolve the path and decl name for this module.
  std::optional<std::string> modulePath;
  if (parentState->decl != impl->topLevelDecl) {
    if (!parentState->sourcePath) {
      return createErrorModuleState(identifierLoc, declName, *parentState->decl,
                                    "unable to locate module '" + name + "'");
    }
    modulePath = ::resolveModulePath(*this, loc, name, *parentState->sourcePath,
                                     disablePrebuiltPackages);
  } else {
    // If this is a top-level import, try to resolve a standard library module.
    // We current bundle all of the standard library packages into one mega
    // package, but still want to expose them separately.
    if (impl->stdlibPackageState && name != "stdlib") {
      // Check for an existing module for this name. If we find one, insert it
      // into the parent state and return it.
      auto it = impl->stdlibPackageState->nestedModules.find(declName);
      if (it != impl->stdlibPackageState->nestedModules.end()) {
        parentState->nestedModules.insert({declName, it->second});
        return *it->second;
      }

      // Otherwise, if the standard library is a source package, check to see if
      // we can resolve a path from it.
      if (impl->stdlibPackageState->sourcePath) {
        modulePath = ::resolveModulePath(*this, loc, name,
                                         *impl->stdlibPackageState->sourcePath,
                                         disablePrebuiltPackages);
        if (modulePath) {
          ModuleState &moduleState = importModuleState(("stdlib." + name).str(),
                                                       impl->topLevelDecl, loc);
          parentState->nestedModules.insert({declName, &moduleState});
          return moduleState;
        }
      }
    }

    // Otherwise, go through the normal import path.
    modulePath = resolveModulePath(name, loc);
  }
  if (!modulePath) {
    return createErrorModuleState(identifierLoc, declName, *parentState->decl,
                                  "unable to locate module '" + name + "'");
  }

  // If the path was a directory, we're importing a source package.
  if (std::filesystem::is_directory(*modulePath)) {
    auto fileLoc = createLocation(getPackageInitPath(*modulePath), /*line=*/1,
                                  /*column=*/1);
    return createPackageState(declName, *modulePath, *parentState, fileLoc);
  }

  // Check if the path is a binary package.
  StringRef pathRef(*modulePath);
  if (pathRef.ends_with(".mojopkg") || pathRef.ends_with(".📦"))
    return createBinaryPackageState(loc, declName, *modulePath, *parentState);

  // Open the module file within the source manager. Reuse an existing file if
  // we've already opened it.
  unsigned fileID = impl->existingSourceMgrBuffers.lookup(pathRef);
  if (!fileID) {
    std::string fullPath;
    fileID = getSourceMgr().AddIncludeFile(*modulePath, loc, fullPath);
    impl->includedFiles.push_back(fullPath);
  }

  // Now that we have a MemoryBuffer, we can lex it, and therefore parse it.
  // do so.
  const llvm::MemoryBuffer *moduleBuffer =
      getSourceMgr().getMemoryBuffer(fileID);
  auto fileLoc = createLocation(moduleBuffer->getBufferIdentifier(), /*line=*/1,
                                /*column=*/1);
  return createModuleState(declName, moduleBuffer, *parentState, fileLoc);
}

SharedState::ModuleState &
SharedState::importRelativeModuleState(StringRef name, ASTDecl *parentDecl,
                                       llvm::SMLoc loc) {
  llvm::SMLoc identifierLoc = loc.isValid() ? loc : parentDecl->getLoc();
  auto emitError = [&](const Twine &message = "") -> ModuleState & {
    return createErrorModuleState(identifierLoc,
                                  StringAttr::get(getContext(), name),
                                  *parentDecl, message);
  };

  auto adjustIdentifierLoc = [&](unsigned offset) {
    if (loc.isValid())
      return llvm::SMLoc::getFromPointer(loc.getPointer() + offset);
    return loc;
  };

  // If the name starts with a `.`, it is relative to the current package.
  if (name.consume_front(".")) {
    // Find the current package.
    identifierLoc = adjustIdentifierLoc(1);
    while (!isa<PackageOp>(*parentDecl) && parentDecl->parentDecl)
      parentDecl = parentDecl->parentDecl;
    if (!isa<PackageOp>(*parentDecl))
      return emitError("cannot import relative to a top-level package");

    // Otherwise, this is a package relative to the current parent.
    while (name.consume_front(".")) {
      identifierLoc = adjustIdentifierLoc(1);
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
    identifierLoc = adjustIdentifierLoc(parentName.size() + 1);
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
    identifierLoc = adjustIdentifierLoc(parentName.size() + 1);
  }

  // Now we can import the final decl. If the parent package has an unresolved
  // import, mark it as resolved and import the state for the module.
  if (failed(declResolver->resolveBody(*parentDecl, loc)))
    return emitError();
  if (parentDecl->declsInScope) {
    auto it =
        parentDecl->declsInScope->find(StringAttr::get(getContext(), name));
    if (it != parentDecl->declsInScope->end() && !it->second.empty()) {
      TinyPtrVector<ASTDecl *> &existingDecls = it->second;
      ASTDecl *existingDecl = existingDecls.front();

      // The decl already exists, so we can just return it.
      if (isa<FileModuleOp, PackageOp>(*existingDecl))
        return *impl->moduleStates[existingDecl];

      // If the decl isn't an unresolved import, emit an error.
      if (!isa<UnresolvedImportOp>(*existingDecl))
        return emitError("'" + name +
                         "' does not refer to a package or module");
      existingDecls.clear();
    }
  }

  return importSubModuleState(name, parentDecl, loc, identifierLoc);
}

bool SharedState::hasBuiltinModule() const { return useBuiltinModule; }

/// Lookup a builtin trait like `AnyType`, `ImplicitlyDestructible`, `Copyable`,
/// `Movable` etc.  On error this returns null but does not print an error.
ASTDecl *SharedState::lookupBuiltinTrait(StringRef traitName, ASTDecl *context,
                                         SMLoc loc) {
  LookupResult lookup = lookupAndResolveDecl(traitName, loc, *context, true);
  if (!lookup.isFailure() && !lookup.getIfSuccess().empty()) {
    for (ASTDecl *result : lookup.getIfSuccess()) {
      if (auto trait = dyn_cast<TraitDeclOp>(result))
        return result;
    }
  }
  return nullptr;
}

ASTDecl *SharedState::lookupNamedTypeDecl(StringRef name, ASTDecl &context,
                                          llvm::SMLoc loc) {
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
  if (!isa<StructDeclOp>(firstDecl)) {
    auto diag = emitError(loc, "'") << name << "' doesn't resolve to a type";
    diag.attachNote(firstDecl.getLoc()) << "'" << name << "' declared here";
    return {};
  }
  return &firstDecl;
}

ASTType SharedState::lookupNamedType(StringRef name, ASTDecl &context,
                                     llvm::SMLoc loc) {
  if (ASTDecl *decl = lookupNamedTypeDecl(name, context, loc))
    return cast<StructDeclOp>(decl).bindReference();
  return getTypeCheckErrorType();
}

ASTType SharedState::getBuiltinVariadicListType(ASTDecl &context,
                                                llvm::SMLoc loc, bool inMem) {
  return lookupNamedType(inMem ? "VariadicListMem" : "VariadicList", context,
                         loc);
}

ASTDecl *SharedState::getBuiltinCoroutineType(llvm::SMLoc loc) {
  ASTDecl &coroutineModule =
      importModule("builtin.coroutine", /*currentPackage=*/nullptr, loc);
  return lookupNamedTypeDecl("Coroutine", coroutineModule, loc);
}

ASTDecl *SharedState::getBuiltinRaisingCoroutineType(llvm::SMLoc loc) {
  ASTDecl &coroutineModule =
      importModule("builtin.coroutine", /*currentPackage=*/nullptr, loc);
  return lookupNamedTypeDecl("RaisingCoroutine", coroutineModule, loc);
}

ASTType SharedState::getOwnedKwargsDictType(llvm::SMLoc loc) {
  ASTDecl &collectionsModule =
      importModule("stdlib.collections.dict", /*currentPackage=*/nullptr, loc);
  return lookupNamedType("OwnedKwargsDict", collectionsModule, loc);
}

ASTType SharedState::getBuiltinCaptureListType(llvm::SMLoc loc) {
  ASTDecl &closureModule =
      importModule("builtin._closure", /*currentPackage=*/nullptr, loc);
  return lookupNamedType("__ParameterClosureCaptureList", closureModule, loc);
}

ASTType SharedState::getBuiltinStubsMLIRType(llvm::SMLoc loc) {
  ASTDecl &stubsModule =
      importModule("builtin._stubs", /*currentPackage=*/nullptr, loc);
  return lookupNamedType("__MLIRType", stubsModule, loc);
}

ArrayRef<ASTDecl *> SharedState::getBuiltinFunction(ASTDecl &context,
                                                    StringRef moduleName,
                                                    StringRef fnName,
                                                    llvm::SMLoc loc) {
  ASTDecl &module = importModule(moduleName, /*currentPackage=*/nullptr, loc);
  return getBuiltinFunction(module, fnName, loc);
}

ArrayRef<ASTDecl *> SharedState::getBuiltinFunction(ASTDecl &module,
                                                    StringRef fnName,
                                                    llvm::SMLoc loc) {
  LookupResult result =
      lookupAndResolveDecl(fnName, loc, module, /*searchParentScopes=*/false);
  if (!result.isSuccess() || result.getIfSuccess().empty()) {
    emitError(loc, "internal error: could not find builtin function '")
        << fnName << "'";
    return {};
  }
  ArrayRef<ASTDecl *> decls = result.getIfSuccess();
  if (!isa<FnOp>(decls.front())) {
    emitError(loc, "internal error: builtin '")
        << fnName << "' does not refer to a function";
    return {};
  }
  return decls;
}

void SharedState::importBuiltinModules(ASTDecl &moduleDecl) {
  // Check if this is the first attempt at resolving the builtin modules.
  if (impl->implicitBuiltinImports.empty()) {
    // Import the main standard library package.
    impl->stdlibPackageState =
        &importModuleState("stdlib", impl->topLevelDecl, moduleDecl.getLoc());
    if (failed(declResolver->resolveBody(*impl->stdlibPackageState->decl,
                                         moduleDecl.getLoc())))
      return;

    // Import the prelude package.
    ASTDecl &preludePackageDecl =
        *importModuleState("stdlib.prelude", impl->topLevelDecl,
                           moduleDecl.getLoc())
             .decl;
    if (failed(
            declResolver->resolveBody(preludePackageDecl, moduleDecl.getLoc())))
      return;

    for (StringRef name :
         llvm::make_first_range(preludePackageDecl.getDeclsInScope())) {
      impl->implicitBuiltinImports.emplace_back(
          StringAttr::get(getContext(), "prelude." + name));
    }
  }

  for (StringAttr import : impl->implicitBuiltinImports)
    moduleDecl.addUnresolvedWildCardImport(import, /*isFullImport=*/false,
                                           moduleDecl.getLoc());
}

ASTDecl &SharedState::createModule(StringRef moduleName,
                                   const llvm::MemoryBuffer *moduleBuffer,
                                   FileLineColLoc loc) {
  // Create a new module state.
  ModuleState &state =
      createModuleState(StringAttr::get(getContext(), moduleName), moduleBuffer,
                        *impl->topLevelModuleState, loc);
  return *state.decl;
}

ASTDecl &SharedState::createPackage(StringRef path, StringRef name) {
  auto fileLoc = createLocation(getPackageInitPath(path.str()),
                                /*line=*/1, /*column=*/1);
  ModuleState &state =
      createPackageState(StringAttr::get(getContext(), name), path,
                         *impl->topLevelModuleState, fileLoc);
  return *state.decl;
}

ASTDecl &SharedState::createBinaryPackage(StringRef path, StringRef name) {
  ModuleState &state =
      createBinaryPackageState(SMLoc(), StringAttr::get(getContext(), name),
                               path, *impl->topLevelModuleState);
  return *state.decl;
}

std::optional<std::string> SharedState::getModuleSourcePath(ASTDecl &module) {
  auto it = impl->moduleStates.find(&module);
  if (it == impl->moduleStates.end())
    return std::nullopt;
  return it->second->sourcePath;
}

SharedState::ModuleState &
SharedState::createModuleState(StringAttr declName,
                               const llvm::MemoryBuffer *moduleBuffer,
                               ModuleState &parentState, FileLineColLoc loc) {
  Lexer lexer(diags, moduleBuffer);

  // Create a new decl for this module.
  auto moduleBuilder = parentState.decl->getDeclEndBuilder();
  Operation *fileOp = moduleBuilder.create<FileModuleOp>(loc, declName);
  ASTDecl &moduleDecl = declResolver->addDecl(
      fileOp, lexer.getToken().getLoc(), declName, parentState.decl,
      lexer.getCursor(), LexerCursor::getEOF(moduleBuffer), /*indentation=*/-1);

  ModuleState &moduleState = parentState.insertNestedModule(
      declName, std::make_unique<ModuleState>(
                    &moduleDecl, moduleBuffer->getBufferIdentifier()));
  impl->moduleStates[&moduleDecl] = &moduleState;

  // Auto-import the core language modules.
  if (useBuiltinModule)
    importBuiltinModules(moduleDecl);

  notifyListenerOnModuleDecl(moduleDecl, moduleDecl.getLoc());
  return moduleState;
}

SharedState::ModuleState &
SharedState::createPackageState(StringAttr declName, StringRef packagePath,
                                ModuleState &parentState, FileLineColLoc loc) {
  // Create a new decl for this module.
  auto moduleBuilder = parentState.decl->getDeclEndBuilder();
  auto packageOp = moduleBuilder.create<PackageOp>(loc, declName);
  SMLoc declLoc = declResolver->shared.diags.convertLocToSMLoc(loc);
  ASTDecl &decl =
      declResolver->addDecl(packageOp, declLoc, declName, parentState.decl,
                            parentState.decl->getCursor(),
                            parentState.decl->getCursor(), /*indentation=*/-1);

  // Insert the newly created module state.
  ModuleState &moduleState = parentState.insertNestedModule(
      declName, std::make_unique<ModuleState>(&decl, packagePath));
  impl->moduleStates[&decl] = &moduleState;
  impl->packageStates[packageOp] = &moduleState;

  return moduleState;
}

SharedState::ModuleState &
SharedState::createBinaryPackageState(SMLoc loc, StringAttr declName,
                                      StringRef packagePath,
                                      ModuleState &parentState) {
  auto makeError = [&](const Twine &msg) -> ModuleState & {
    return createErrorModuleState(loc, declName, *parentState.decl, msg);
  };

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> packageBuffer =
      llvm::MemoryBuffer::getFile(packagePath);
  if (!packageBuffer)
    return makeError("unable to open package file '" + packagePath + "'");

  // Read the cached package.
  OpBuilder builder = parentState.decl->getDeclEndBuilder();
  Block *block = builder.getBlock();
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
    if (failed(bytecodeReader->readTopLevel(block)))
      return makeError("unable to load package '" + packagePath + "'");

    // Add the package path to the set of included files.
    impl->includedFiles.emplace_back(packagePath.str());
  }

  // The bytecode module includes the package module and any function stubs.
  auto tmpModule = cast<ModuleOp>(block->back());
  if (failed(bytecodeReader->materialize(tmpModule)))
    return makeError("failed to materialize top-level module");

  // Move the package into the current decl.
  auto packageOp = cast<PackageOp>(tmpModule.getBody()->front());
  packageOp->remove();
  builder.insert(packageOp);

  // Process each of the stubs, deduplicating each of them into the shared
  // state. For any added thunks, we have to register a decl for them.
  auto theModule = cast<ModuleOp>(getTopLevelDecl());
  for (auto thunk : llvm::make_early_inc_range(tmpModule.getOps<FnOp>())) {
    Attribute key = thunk.getThunkKeyAttr();
    assert(key && "thunk is missing its key");
    FnOp &registeredThunk = impl->conversionThunks[key];
    if (registeredThunk)
      continue; // thunk already exists
    registeredThunk = thunk;

    // Move the thunk into the top-level and add it as fully resolved.
    if (failed(bytecodeReader->materialize(thunk)))
      return makeError("failed to materialize function thunk");
    thunk->remove();
    theModule.push_back(thunk);
    ASTDecl &thunkDecl = declResolver->addBytecodeDecl(
        &*thunk, thunk.getSourceNameAttr(), &getTopLevelDecl(),
        DeclResolvedness::body);
    declResolver->finalizeFuncSignature(thunk, thunkDecl);
  }

  // Insert a new module decl.
  ASTDecl &decl = declResolver->addBytecodeDecl(
      packageOp, declName, parentState.decl, DeclResolvedness::signature);

  // Initialize the module state.
  ModuleState &moduleState = parentState.insertNestedModule(
      declName, std::make_unique<ModuleState>(&decl));
  moduleState.bytecodeReader = std::move(bytecodeReader);
  moduleState.tmpModule = tmpModule;
  impl->moduleStates[&decl] = &moduleState;
  impl->packageStates[cast<PackageOp>(decl)] = &moduleState;

  return moduleState;
}

SharedState::ModuleState &SharedState::createErrorModuleState(
    SMLoc loc, StringAttr name, ASTDecl &errorContext, const Twine &errorMsg) {
  // Check if we already have an error decl with this name.
  if (auto *it = impl->topLevelModuleState->nestedModules.lookup(name))
    return *it;

  // Emit the error message the first time this error module state is created.
  emitError(loc, errorMsg);

  // Otherwise, create one.
  ASTDecl *decl =
      &declResolver->addErroneousDecl(name, loc, impl->topLevelDecl);
  ModuleState &state = impl->topLevelModuleState->insertNestedModule(
      name, std::make_unique<ModuleState>(decl));
  impl->moduleStates[state.decl] = &state;
  return state;
}

ASTDecl *
SharedState::lookupAndResolveMangledDecl(StringAttr leafRef, SMLoc loc,
                                         ASTDecl &container,
                                         DeclResolvedness howResolved) {
  // When a bytecode module depends on a decl parsed from source, we have to
  // resolve the signatures of all the children of the source decl, because
  // otherwise they won't be registered in the symbol table.
  if (!container.loadedFromBytecode && !container.referencedFromBytecode) {
    container.referencedFromBytecode = true;
    SmallVector<ASTDecl *> toResolve;
    for (auto &[_, children] : container.getDeclsInScope())
      llvm::append_range(toResolve, children);
    for (ASTDecl *child : toResolve)
      if (failed(declResolver->resolveSignature(*child, loc)))
        return nullptr;
  }

  // Find the operation in the symbol table of its container.
  auto declOp = lookupSymbolIn<ASTDeclInterface>(&container, leafRef);
  if (!declOp)
    return nullptr;
  // Retrieve the proper decl name.
  StringAttr name = declOp.getDeclName();

  // If the container is loaded from bytecode, the decl should already be
  // defined within the container decl, look it up directly. This avoids going
  // through the more complex resolution paths (which also resolve other things
  // in the container that aren't needed for this lookup).
  ArrayRef<ASTDecl *> result;
  if (container.loadedFromBytecode) {
    result = container.lookupInCurrentScope(name);
  } else {
    LookupResult lookup = lookupAndResolveDecl(name, loc, container,
                                               /*searchParentScopes=*/false);
    result = lookup.getIfSuccess();
  }

  // Find the entry that matches the full symbol name.
  for (ASTDecl *decl : result) {
    if (decl->getIfOperation() != declOp)
      continue;
    if (failed(declResolver->resolve(*decl, howResolved, loc)))
      return nullptr;
    return decl;
  }
  llvm::report_fatal_error(
      "expected decl in symbol table to appear in lookup: " + name.getValue());
  return nullptr;
}

LogicalResult SharedState::resolveDeclReferencesIn(SMLoc loc, Type type) {
  auto &refWalker = getImpl().bytecodeRefResolutionWalker;
  auto savedContextLoc = refWalker.saveResolutionContextLoc(loc);
  return success(!refWalker.walk(type).wasInterrupted());
}

LogicalResult
SharedState::resolveDeclFromBytecode(ASTDecl &decl,
                                     DeclResolvedness resolvedness) {
  Operation *declOp = decl.getIfOperation();
  auto &refWalker = getImpl().bytecodeRefResolutionWalker;
  auto savedContextLoc = refWalker.saveResolutionContextLoc(decl.getLoc());

  // Handle resolving the signature of the decl.
  if (decl.resolvedness < DeclResolvedness::signature) {
    decl.resolvedness = DeclResolvedness::signature;

    LogicalResult result =
        llvm::TypeSwitch<Operation *, LogicalResult>(declOp)
            .Case([&](FnOp funcOp) {
              declResolver->declForFuncSymbol[decl.getSymbolRef()] = &decl;

              // Resolve the references from the signature.
              refWalker.walk(declOp->getAttrDictionary());
              return success();
            })
            .Case([&](StructDeclOp structOp) {
              // Resolve the types of any parameters.
              refWalker.walk(structOp.getParamsAttr());
              refWalker.walk(structOp.getCanonicalTrait());
              if (TypeAttr nmTarget = structOp.getNonmaterializableTargetAttr())
                refWalker.walk(nmTarget);
              return success();
            })
            .Case([&](TraitDeclOp traitOp) {
              // TODO(traits): Resolve parameter types, when they exist.
              refWalker.walk(traitOp.getCanonicalTrait());
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
              refWalker.walk(varDecl.getType());
              return mlir::success();
            })
            .Case([&](AliasDeclOp aliasDecl) {
              refWalker.walk(aliasDecl.getType());
              if (TypedAttr value = aliasDecl.getValueAttr())
                refWalker.walk(value);
              return mlir::success();
            })
            .Case([&](StructFieldOp field) {
              refWalker.walk(field.getType());
              return mlir::success();
            })
            .Default([](auto) { return mlir::success(); });
    if (failed(result))
      return failure();
  }
  if (resolvedness < DeclResolvedness::body)
    return success();
  decl.resolvedness = DeclResolvedness::body;

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
        refWalker.walkRange(block.getArgumentTypes());
    refWalker.walkRange(op->getOperandTypes());
    refWalker.walkRange(op->getResultTypes());
    refWalker.walk(op->getAttrDictionary());
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
    return declResolver->addBytecodeDecl(op, name, &decl,
                                         DeclResolvedness::unparsed);
  };

  // If this decl is a package, this is its corresponding module state.
  ModuleState *packageState = nullptr;
  if (auto declPackage = dyn_cast<PackageOp>(declOp)) {
    packageState = impl->moduleStates[&decl];

    // Fully resolve any dependencies of the package.
    if (LinkDependencyArrayAttr deps = declPackage.getDependenciesAttr()) {
      for (FlatSymbolRefAttr dep : deps) {
        ASTDecl *depDecl = &importModule(
            dep.getValue(), /*currentPackage=*/nullptr, decl.getLoc());
        if (failed(declResolver->resolveBody(*depDecl, decl.getLoc())))
          return failure();
      }
    }
  }

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
          .Case([&](FnOp op) { addDeclForOp(op, op.getDeclName()); })
          .Case([&](UnresolvedImportOp op) {
            addDeclForOp(op, op.getImportNameAttr());
          })
          .Case([&](UnresolvedWildcardImportOp op) {
            decl.addUnresolvedWildCardImport(op.getModuleNameAttr(),
                                             op.getFullImport(), decl.getLoc());
          })
          .Case([&](StructDeclOp op) {
            ASTDecl &structDecl = addDeclForOp(op, op.getSymNameAttr());
            structDecl.setTypeDeclSelf(ASTDecl::computeSelfTypeForStruct(op));
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
            traitDecl.setTypeDeclSelf(ASTDecl::computeSelfTypeForTrait(op));
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
            StringAttr name = op.getSymNameAttr();
            ASTDecl &decl = addDeclForOp(op, name);

            // Record a nested module state for this decl.
            ModuleState &moduleState = packageState->insertNestedModule(
                name, std::make_unique<ModuleState>(&decl));

            impl->moduleStates[&decl] = &moduleState;
            if constexpr (std::is_same_v<decltype(op), PackageOp>)
              impl->packageStates[op] = &moduleState;
          })
          .Case([&](ConformanceOp op) {
            // Witness tables are considered signature-resolved from the start
            // since there's nothing else to resolve for its "signature". (see
            // CALROC for more).
            ASTDecl &decl = addDeclForOp(op, op.getSymNameAttr());
            decl.resolvedness = DeclResolvedness::signature;
          })
          .Default([&](Operation *op) { deferredOps.push_back(op); });
    }
  }

  // Resolve references within the deferred operations. These don't have
  // corresponding decls, so we manually resolve them now. Walk in pre-order so
  // that nested ops get visited too.
  for (Operation *op : deferredOps)
    if (op->walk(resolveSingleOp).wasInterrupted())
      return failure();

  // After processing the region, make sure any non-signature attributes get
  // resolved.
  refWalker.walk(declOp->getAttrDictionary());
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
    // Erase the temporary ModuleOp that was used to read bytecode.
    module->tmpModule.erase();
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
    for (llvm::sys::fs::recursive_directory_iterator f(rawPath, ec), e; f != e;
         f.increment(ec)) {
      if (ec)
        continue;
      const std::string &path = f->path();
      // Skip non-directories and source packages, internal packages should be
      // imported using a relative import.
      if (!llvm::sys::fs::is_directory(path) ||
          Filesystem::isMojoSourcePackagePath(path))
        continue;
      if (callback(path).wasInterrupted())
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

  // Use ~0U to indicate the end of the buffer, that should be fine as we don't
  // expect tokens to be >= 2^32 charachetrs long.
  // NOTE: We cannot use ~0ULL as it leads to integer overflow when computing
  // end of the StringRef.
  Lexer lexer(shared.diags,
              StringRef(curPtr, std::numeric_limits<uint32_t>::max()), curPtr);
  return lexer.getToken().getSpelling().size();
}

/// Given a pointer to the start of a token, find the end of it.
static void adjustTokenEndPoint(SharedState &shared, SMLoc &loc) {
  size_t tokenSize = getTokenLength(shared, loc);
  loc = SMLoc::getFromPointer(loc.getPointer() + tokenSize);
}

LIT::StructDeclOp
SharedState::getOrCreateClosureWrapper(SMLoc loc, FuncTypeGeneratorType sig,
                                       ASTDecl *moduleDecl) {
  StructDeclOp &existing = impl->closureWrappers[{sig, moduleDecl}];
  if (!existing) {
    std::string name =
        ASTType(sig).getAsString(/*diags=*/this, /*demangleParams=*/true);
    ClosureEmitter emitter(*moduleDecl, *this);
    existing = emitter.createClosureWrapperStructDecl(
        StringAttr::get(getContext(), name), sig, loc);
  }
  return existing;
}

FnOp SharedState::getOrCreateFunctionThunk(Attribute key,
                                           CreateThunkFn create) {
  FnOp &thunk = impl->conversionThunks[key];
  if (!thunk)
    thunk = create(key, getTopLevelDecl());
  return thunk;
}

const llvm::MapVector<ASTDecl *, Capture> &
SharedState::getCaptureRangeInScope(ASTDecl &scope) {
  return getImpl().capturesInScope[&scope];
}

void SharedState::addCaptureToScope(ASTDecl &scope, ASTDecl *captureDecl,
                                    Capture capture) {
  getImpl().capturesInScope[&scope].insert({captureDecl, capture});
  if (captureDecl->getParentDecl() != scope.parentDecl) {
    if (scope.getNearestDeclOfType<FnOp>())
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
  if (failed(declResolver.resolveBody(decl, loc)))
    return;
  ArrayRef<std::pair<StringAttr, TinyPtrVector<ASTDecl *>>> decls =
      decl.getDeclsInScope();
  for (int i = 0, e = decls.size(); i < e; ++i) {
    // Resolution may invalidate the decls vector, so we can't rely on
    // iterators here. We also don't fail, because the listener should be
    // tolerant to errors.
    auto &[name, children] = *std::next(decls.begin(), i);

    // This case sometimes occurs in invalid code in the LSP.
    if (children.empty())
      continue;

    (void)declResolver.resolveBody(*children.front(), loc);
  }
  // Resolve any pending wildcards in the decl. We don't care about failure
  // here, as we still want to enable lookup for the decls that could be
  // resolved.
  (void)declResolver.resolveAllWildcardImports(decl);
}

/// Return if the given parser listener is interested in the given location.
static bool isListenerInterestedInLoc(ParserListener *listener, SMLoc loc) {
  return listener && listener->isInterestedInLoc(loc);
}

void SharedState::notifyListenerOnAliasDecl(ASTDecl &decl,
                                            SMLoc identifierLoc) {
  if (isListenerInterestedInLoc(parserListener, identifierLoc))
    parserListener->onAliasDecl(&decl, identifierLoc);
}

void SharedState::notifyListenerOnArgumentDecl(ASTDecl &decl, StringRef argName,
                                               SMLoc identifierLoc) {
  if (isListenerInterestedInLoc(parserListener, identifierLoc))
    parserListener->onArgumentDecl(&decl, argName, identifierLoc);
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
  case CallSyntax::kMethodCallSynthetic:
  case CallSyntax::kIndirectCall:
  case CallSyntax::kTypeCall:
  case CallSyntax::kOperator:
  case CallSyntax::kReversedOperator:
  case CallSyntax::kSubscript:
  case CallSyntax::kImplicitConvert:
  case CallSyntax::kImplicitCopyInit:
  case CallSyntax::kImplicitMoveInit:
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
                                       SMLoc rParenLoc, CallSyntax syntax,
                                       const CallOperands &callOperands) {
  // Ignore synthetic calls to functions.
  if (syntax == CallSyntax::kMethodCallSynthetic ||
      syntax == CallSyntax::kImplicitConvert ||
      syntax == CallSyntax::kImplicitCopyInit ||
      syntax == CallSyntax::kImplicitMoveInit)
    return;

  if (isListenerInterestedInLoc(parserListener, rParenLoc))
    parserListener->onCall(decls, rParenLoc, callOperands);
}

void SharedState::notifyListenerOnParameterBinding(ArrayRef<ASTDecl *> decls,
                                                   llvm::SMLoc rsquareLoc,
                                                   ArrayRef<Operand> operands) {
  if (isListenerInterestedInLoc(parserListener, rsquareLoc)) {
    SmallVector<ExprNode *> parameters = llvm::map_to_vector(
        operands, [](const Operand &operand) { return operand.expr; });
    parserListener->onParameterBinding(decls, rsquareLoc, parameters);
  }
}

/// These two methods are used to memoize whether a type is implicitly
/// convertible to another type, which includes overload resolution etc.
std::optional<bool> SharedState::getCachedImplicitConvertibility(ASTType from,
                                                                 ASTType to) {
  DenseMap<std::pair<Type, Type>, bool> &cache =
      getImpl().cachedImplicitConvertibility;
  auto it = cache.find({from, to});
  if (it == cache.end())
    return {};

#ifndef NDEBUG
  // If this is the 64th convertibility hit, allow it to fail so we can detect
  // if the cache ever starts to depend on new state like declContext.  This is
  // a small bit a paranoia to make it possible to track down subtle bugs that
  // may happen in the future.
  if ((cache.size() & 63) == 0)
    return {};
#endif
  return it->second;
}
void SharedState::cacheImplicitConvertibility(ASTType from, ASTType to,
                                              bool isConvertible) {
  DenseMap<std::pair<Type, Type>, bool> &cache =
      getImpl().cachedImplicitConvertibility;
  auto [it, newlyInserted] = cache.insert({{from, to}, isConvertible});

  // If the entry is already present, make sure all checks agree.
  if (!newlyInserted)
    assert(it->second == isConvertible &&
           "convertibility cache disagrees from actual computation! Must need "
           "to include more information in the hash key");
}

namespace {
/// This struct is used to fold @always_inline("builtin") functions.
struct BuiltinFunctionFolder {
  SharedState &shared;
  ParameterEvaluator evaluator;
  bool doEmitError;

  // Keep track of the parameter values for each of the live SSA values in the
  // body, and start by binding the argument values.
  DenseMap<Value, TypedAttr> boundValues;

  // For our virtual memory model, we track entire indirect values like
  // var-decls in this map.  lit.struct.ref indexes to subfields are not
  // immediately processed - they are handled by load/store operations, mostly
  // to handle constructors.
  SmallDenseMap<Value, TypedAttr> varDeclSoFar;

  BuiltinFunctionFolder(SharedState &shared, bool doEmitError)
      : shared(shared), doEmitError(doEmitError) {}

  // This helper handles emitting an error (or not) as needed.
  // This helper handles emitting an error (or not) as needed.
  InflightDiag emitError(Location loc) {
    auto result = shared.emitError(loc) << "'@always_inline(\"builtin\")' ";
    if (!doEmitError) // Only emit an error if requested.
      result.abandon();
    return result;
  };

  // Lookup a pre-bound value and check for validity.  This emits an error and
  // returns null if something goes wrong.
  TypedAttr findValue(Value v) {
    auto result = boundValues[v];
    if (!result)
      emitError(v.getLoc()) << "could not resolve operand value";
    return result;
  };

  void recordValue(Value v, TypedAttr attr) {
    assert(evaluator.getReboundType(v.getType()) == attr.getType() &&
           "incorrect fold");
    assert(!boundValues[v] && "value already has a bound value");
    boundValues[v] = attr;
  }

  /// Process the following operation, doing one of three things:
  /// 1) Fold it to a single TypedAttr, returning it.
  /// 2) Return a failure to indicate that the operation is not foldable.
  /// 3) Return a a null TypedAttr to indicate that the operation was processed
  /// but didn't produce a value (e.g. StoreOps).
  FailureOr<TypedAttr> fold(Operation &op);
};
} // end anonymous namespace

/// Process the following operation, doing one of three things:
/// 1) Fold it to a single TypedAttr, returning it.
/// 2) Return a failure to indicate that the operation is not foldable.
/// 3) Return a a null TypedAttr to indicate that the operation was processed
/// but didn't produce a value (e.g. StoreOps).
FailureOr<TypedAttr> BuiltinFunctionFolder::fold(Operation &op) {
  if (auto paramCst = dyn_cast<ParamConstantOp>(op))
    return evaluator.getReboundAttribute(paramCst.getValue());
  if (auto cst = dyn_cast<mlir::index::ConstantOp>(op))
    return TypedAttr(cst.getValueAttr());

  if (auto extract = dyn_cast<LIT::StructExtractOp>(op)) {
    if (auto base = findValue(extract.getOperand()))
      return LIT::StructExtractAttr::get(
          base, extract.getFieldAttr(),
          evaluator.getReboundType(extract.getType()));
  }

  // Handle a simple binary operation that folds to a POC binary op.
  auto foldBinOp = [&](POC opc) -> FailureOr<TypedAttr> {
    if (auto lhs = findValue(op.getOperand(0)))
      if (auto rhs = findValue(op.getOperand(1)))
        return ParamOperatorAttr::get(opc, lhs, rhs);
    return failure();
  };

  // Many index binops fold directly to POC binops.
  if (auto add = dyn_cast<mlir::index::AddOp>(op))
    return foldBinOp(POC::Add);
  if (auto mul = dyn_cast<mlir::index::MulOp>(op))
    return foldBinOp(POC::Mul);
  if (auto andOp = dyn_cast<mlir::index::AndOp>(op))
    return foldBinOp(POC::And);
  if (auto orOp = dyn_cast<mlir::index::OrOp>(op))
    return foldBinOp(POC::Or);
  if (auto xorOp = dyn_cast<mlir::index::XOrOp>(op))
    return foldBinOp(POC::Xor);
  if (auto shlOp = dyn_cast<mlir::index::ShlOp>(op))
    return foldBinOp(POC::Shl);
  if (auto shrOp = dyn_cast<mlir::index::ShrSOp>(op))
    return foldBinOp(POC::Shr);
  if (auto andOp = dyn_cast<POP::AndOp>(op)) // i1 operations.
    return foldBinOp(POC::And);
  if (auto orOp = dyn_cast<POP::OrOp>(op))
    return foldBinOp(POC::Or);
  if (auto xorOp = dyn_cast<POP::XOrOp>(op))
    return foldBinOp(POC::Xor);

  // Sub doesn't have a POC opcode: "x-y" is "x+(y*-1)".
  if (auto sub = dyn_cast<mlir::index::SubOp>(op)) {
    if (auto lhs = findValue(sub.getOperand(0)))
      if (auto rhs = findValue(sub.getOperand(1)))
        return ParamOperatorAttr::getSub(lhs, rhs);
  }

  if (auto cmp = dyn_cast<mlir::index::CmpOp>(op)) {
    if (auto lhs = findValue(cmp.getOperand(0)))
      if (auto rhs = findValue(cmp.getOperand(1))) {
        switch (cmp.getPred()) {
        default:
          // TODO: we don't handle unsigned comparisons in ParamOperatorAttr
          // yet. It can do it, but we don't have a way to pass the unsigned
          // flag through easily.
          break;
        case mlir::index::IndexCmpPredicate::EQ:
          return ParamOperatorAttr::get(POC::EQ, lhs, rhs);
        case mlir::index::IndexCmpPredicate::NE:
          return ParamOperatorAttr::getNE(lhs, rhs);
        case mlir::index::IndexCmpPredicate::SLT:
          return ParamOperatorAttr::get(POC::LT, lhs, rhs);
        case mlir::index::IndexCmpPredicate::SLE:
          return ParamOperatorAttr::get(POC::LE, lhs, rhs);
        case mlir::index::IndexCmpPredicate::SGT:
          return ParamOperatorAttr::get(POC::LT, rhs, lhs);
        case mlir::index::IndexCmpPredicate::SGE:
          return ParamOperatorAttr::get(POC::LE, rhs, lhs);
        }
      }
  }

  if (auto bitcast = dyn_cast<POP::PointerBitcastOp>(op)) {
    if (auto src = findValue(bitcast.getInput()))
      return ParamOperatorAttr::get(
          POC::PtrBitcast, src, evaluator.getReboundType(bitcast.getType()));
  }

  // FIXME(StringLiteral): Remove this operation.
  if (auto strSize = dyn_cast<POP::StringSizeOp>(op)) {
    if (auto str = findValue(strSize.getStr()))
      return POP::StringSizeAttr::get(str.getContext(), str);
  }

  if (auto call = dyn_cast<LIT::CallOp>(op)) {
    SmallVector<TypedAttr> calleeOperands;
    calleeOperands.push_back(evaluator.getReboundAttribute(call.getCallee()));
    for (auto operandVal : call.getOperands()) {
      calleeOperands.push_back(findValue(operandVal));
      if (!calleeOperands.back())
        return failure();
    }

    // Note that the recursive call here always generates an error.  We know
    // that this was inside of a "builtin" function so we're not being
    // called speculatively on an arbitrary function.
    if (auto result =
            shared.foldInlineBuiltinFunction(calleeOperands, op.getLoc(),
                                             /*emitError=*/true))
      return result;
    return failure();
  }

  // For vardecls, the primary pattern we're trying to handle is:
  //   %tmp = lit.var.decl Int
  //   %tmp2 = lit.ref.struct.ger %tmp, value
  //   lit.ref.store %v, %tmp2
  //   lit.load.consume %tmp
  // Which happens in ctors for builtin operations.  Ignore anything more
  // complex.
  if (auto varDecl = dyn_cast<VarDeclOp>(op)) {
    auto eltType = evaluator.getReboundType(varDecl.getType().getElementType());
    ASTDecl *decl = ASTType(eltType).getDecl(shared);
    StructDeclOp structOp;
    if (decl && (structOp = dyn_cast<StructDeclOp>(*decl)) &&
        structOp.getConvention() == TypeConvention::RegisterPassableTrivial) {
      varDeclSoFar[varDecl] = UnknownAttr::get(eltType);
      return TypedAttr();
    }
  }

  if (auto load = dyn_cast<LoadConsumeOp>(op))
    return varDeclSoFar[load.getRef()];

  if (auto load = dyn_cast<RefLoadOp>(op))
    return varDeclSoFar[load.getRef()];

  if (auto ger = dyn_cast<RefStructGEROp>(op))
    return TypedAttr(); // handled by user.

  if (auto store = dyn_cast<RefStoreOp>(op)) {
    TypedAttr value = findValue(store.getValue());
    auto ger = store.getDest().getDefiningOp<RefStructGEROp>();
    if (value && varDeclSoFar[store.getDest()]) {
      // Store of the whole value.
      varDeclSoFar[store.getDest()] = value;
      return TypedAttr();
    }
    if (value && ger && varDeclSoFar[ger.getContainer()]) {
      // Store to a subfield.
      auto gerBase = ger.getContainer();
      auto &varEntry = varDeclSoFar[gerBase];
      auto structType = cast<LIT::StructType>(
          evaluator.getReboundType(gerBase.getType().getElementType()));

      // These asserting casts are checked when the VarDecl is processed.
      auto structOp = cast<StructDeclOp>(*ASTType(structType).getDecl(shared));

      // Form the new struct using all the same fields as before but with the
      // new one replaced.
      SmallVector<std::tuple<StringAttr, TypedAttr>> fields;
      for (auto field : structOp.getFieldDecls()) {
        TypedAttr fieldValue;
        if (field.getName() == ger.getFieldAttr())
          fieldValue = value;
        else
          fieldValue = LIT::StructExtractAttr::get(varEntry, field);
        fields.push_back({field.getNameAttr(), fieldValue});
      }
      varEntry = LITStructAttr::get(fields, structType);
      return TypedAttr();
    }
  }

  if (auto variant = dyn_cast<VariantCreateOp>(op)) {
    if (TypedAttr value = findValue(variant.getOperand())) {
      auto resType =
          cast<VariantType>(evaluator.getReboundType(variant.getType()));
      return TypedAttr(VariantAttr::get(value, variant.getIndex(), resType));
    }
  }

  // We can fold hlcf.if operations in limited form that end with a yield of
  // a single value for which both sides are foldable.
  if (auto ifOp = dyn_cast<HLCF::IfOp>(op)) {
    auto foldBlockWithYield = [&](Block &block) -> FailureOr<TypedAttr> {
      for (Operation &op : block) {
        if (auto yieldOp = dyn_cast<HLCF::YieldOp>(op)) {
          if (yieldOp.getNumOperands() == 1)
            return findValue(yieldOp.getOperand(0));
          emitError(yieldOp.getLoc()) << "can only handle single-result if";
          return failure();
        }

        // Fold the operation.
        FailureOr<TypedAttr> result = fold(op);
        if (failed(result))
          return failure();
        // Otherwise we know this operation. If it returned a value remember it.
        if (TypedAttr val = *result)
          recordValue(op.getResult(0), val);
      }

      // If there is no block terminator then we have malformed IR, presumably
      // due to an already-diagnosed issue.
      return failure();
    };

    if (auto condVal = findValue(ifOp.getCond())) {
      auto trueVal = foldBlockWithYield(ifOp.getThenBlock());
      if (failed(trueVal))
        return trueVal;
      auto falseVal = foldBlockWithYield(ifOp.getElseBlock());
      if (failed(falseVal))
        return falseVal;

      return ParamOperatorAttr::get(POC::Cond, {condVal, *trueVal, *falseVal},
                                    trueVal->getType());
    }
  }

  // Otherwise we don't know what this is, bail out.
  emitError(op.getLoc()) << "does not support MLIR operation "
                         << op.getName().getStringRef();
  return failure();
}

/// Given a parameter expression call to a function marked
/// @always_inline("builtin"), scan the function to form an inlined parameter
/// expression representation of the function given the specified argument
/// values, then return the resultant expression.  If the function cannot be
/// handled as a builtin, emit an error (when emitError is true) and return
/// null.
TypedAttr SharedState::foldInlineBuiltinFunction(ArrayRef<TypedAttr> operands,
                                                 Location callLoc,
                                                 bool emitError) {

  BuiltinFunctionFolder folder(*this, emitError);

  // Resolve the callee and check to verify it is a "builtin" call that is
  // eligible for parameter inlining.
  auto symCst = dyn_cast<SymbolConstantAttr>(operands.front());
  if (!symCst) {
    folder.emitError(callLoc) << "only supports direct calls";
    return {};
  }
  operands = operands.drop_front();

  auto &resolver = getDeclResolver();
  ASTDecl *calleeDecl = resolver.getDeclForFuncSymbol(symCst.getSymbol());
  assert(llvm::isa_and_present<FnOp>(*calleeDecl) && "callee isn't known?");
  auto fnOp = cast<FnOp>(*calleeDecl);
  if (fnOp.getInlineLevel() != InlineLevel::AlwaysBuiltin) {
    folder.emitError(callLoc) << "only supports calls to other "
                                 "'@always_inline(\"builtin\")' functions";
    return {};
  }
  if (failed(resolver.resolveBody(*calleeDecl, calleeDecl->getLoc())) ||
      // Double check to ensure body resolution's check succeeded.
      fnOp.getInlineLevel() != InlineLevel::AlwaysBuiltin) {
    return {}; // Error already diagnosed.
  }

  // The function being called may be a generic function - if so, we need to
  // remap any values and types in the body with parameter values substituted.
  for (auto [decl, value] :
       llvm::zip(fnOp.collectAllParams(/*implOrigins*/ false),
                 symCst.getParamValues()))
    folder.evaluator.setParameterValue(decl, value);

  // Bind the argument values we are provided.
  for (auto [convention, arg, argValue] :
       llvm::zip(fnOp.getFuncTypeGenerator().getArgConventions(),
                 fnOp.getBody()->getArguments(), operands)) {
    if (convention != ArgConvention::ReadReg) {
      folder.emitError(arg.getLoc())
          << "does not support this argument convention";
      return {};
    }
    if (folder.evaluator.getReboundType(arg.getType()) != argValue.getType()) {
      folder.emitError(arg.getLoc()) << "argument type mismatch";
      return {};
    }
    folder.boundValues[arg] = argValue;
  }

  // This function handles a very limited set of operations and no control
  // flow. As such, we can proceed top-down and bail out if we see anything
  // too complex for our little brain.
  for (Operation &op : *fnOp.getBody()) {
    // Handle the final return.
    if (auto ret = dyn_cast<LIT::ReturnOp>(op)) {
      if (ret.getNumOperands() == 1)
        return folder.findValue(ret.getOperand(0));
    }

    // Otherwise it must be an operation that we can fold.
    FailureOr<TypedAttr> result = folder.fold(op);
    if (failed(result))
      return {}; // Error already diagnosed.

    // Otherwise we know this operation. If it returned a value remember it.
    if (TypedAttr val = *result)
      folder.recordValue(op.getResult(0), val);
  }

  // If there is no block terminator then we have malformed IR, presumably
  // due to an already-diagnosed issue.
  return {};
}
