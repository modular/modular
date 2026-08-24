//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/ModuleLoader.h"

#include "ClosureEmitter.h"
#include "ModuleStore.h"

#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/Lexer.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/Support/Configuration.h"
#include "KGEN/Support/MojoPrecompiledFile.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "Support/Filesystem/Paths.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"

#define DEBUG_TYPE "mojo-module-loader"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// ModuleSpec
//===----------------------------------------------------------------------===//

std::optional<ModuleSpec>
ModuleSpec::classify(const std::filesystem::path &path,
                     llvm::StringRef moduleName) {
  // For directory-based module filtering, we must have an exact match.
  if (auto name = path.filename().string();
      moduleName.empty() || name == moduleName) {
    if (Filesystem::isMojoSourcePackagePath(path))
      return ModuleSpec{name, path, ModuleSpec::Kind::SourcePackage};

    std::error_code ec;
    if (std::filesystem::is_directory(path, ec) && !ec)
      return ModuleSpec{name, path, ModuleSpec::Kind::SourceDir};
  }

  // For file-based module filtering, the name must match the filename's stem
  // (i.e., without the final extension).
  if (auto stem = path.filename().stem().string();
      moduleName.empty() || stem == moduleName) {
    if (Filesystem::isMojoBinaryPackagePath(path))
      return ModuleSpec{stem, path, ModuleSpec::Kind::Precompiled};

    if (Filesystem::isMojoSourceFile(path))
      return ModuleSpec{stem, path, ModuleSpec::Kind::SourceModule};
  }

  return std::nullopt;
}

std::string ModuleSpec::canonicalPath() const {
  std::error_code ec;
  std::filesystem::path canonical = std::filesystem::weakly_canonical(path, ec);
  return (ec ? path.lexically_normal() : canonical).string();
}

//===----------------------------------------------------------------------===//
// ModuleLoader
//===----------------------------------------------------------------------===//

/// Collect the import paths configured in the Mojo config, used when no search
/// paths were given explicitly.
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

ModuleLoader::ModuleLoader(SharedState &shared)
    : SharedStateUser(shared), store(std::make_unique<ModuleStore>()) {
  const CompilationOptions &options = shared.options;
  if (!options.searchPaths.empty()) {
    SmallVector<StringRef> paths;
    StringRef(options.searchPaths)
        .split(paths, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    llvm::append_range(autoImportDirs, paths);
  } else {
    collectDefaultImportPaths(autoImportDirs);
  }
  llvm::append_range(autoImportDirs, options.extraSearchPaths);
}

ModuleLoader::~ModuleLoader() = default;

std::optional<ModuleSpec>
ModuleLoader::resolveModulePath(StringRef moduleName, StringRef includeDir,
                                bool ignorePrebuilt,
                                bool isInsideSourcePackage) {
  // Find a path in `includeDir` that is an importable mojo construct matching
  // `moduleName`
  std::error_code ec;
  auto iter = std::filesystem::directory_iterator(includeDir.str(), ec);
  if (ec)
    return std::nullopt;

  // Gets the name of the file or directory in a case sensitive way. On non-case
  // sensitive systems we cannot just do `path / moduleName` since the
  // constructed path will not adhere to case sensitivity.
  std::optional<ModuleSpec> bestMatch;
  for (const auto &entry : iter) {
    if (auto moduleSpec = ModuleSpec::classify(entry.path(), moduleName)) {
      // A package can't legitimately nest a precompiled copy of itself or its
      // own submodules, so this ignores every `.mojoc` candidate when
      // resolving from within a package's own directory, unlike the
      // top-level `-I` search where `.mojoc`-before-`.mojo` precedence still
      // applies to genuine collisions.
      if (moduleSpec->isPrecompiled() &&
          (ignorePrebuilt || isInsideSourcePackage))
        continue;
      if (!bestMatch || moduleSpec->takesImportPrecedence(*bestMatch))
        bestMatch = moduleSpec;
    }
  }

  return bestMatch;
}

std::optional<ModuleSpec> ModuleLoader::resolveModulePath(StringRef moduleName,
                                                          SMLoc includeLoc) {
  unsigned includeBufferId =
      shared.getSourceMgr().FindBufferContainingLoc(includeLoc);

  // A closed (non-directory) candidate in the earliest directory wins
  // outright. Plain directories are namespace portions: they only name the
  // namespace when no closed candidate exists anywhere on the path, and the
  // returned spec records just the first portion; submodule resolution
  // re-derives the full portion set from the spec's namespace components.
  std::optional<ModuleSpec> result;
  std::optional<ModuleSpec> firstPortion;
  traverseImportDirectories(includeBufferId, [&](StringRef dir) {
    // Don't try to resolve modules that reside within a package.
    if (Filesystem::isMojoSourcePackagePath(dir.str())) {
      // TODO: It'd be nice to emit a list of potential modules that the
      // name might correspond with if it did resolve to one inside of this
      // package.
      return WalkResult::advance();
    }
    std::optional<ModuleSpec> candidate =
        resolveModulePath(moduleName, dir, shared.arePrebuiltPackagesDisabled(),
                          /*isInsideSourcePackage=*/false);
    if (!candidate)
      return WalkResult::advance();
    if (candidate->kind != ModuleSpec::Kind::SourceDir) {
      result = candidate;
      return WalkResult::interrupt();
    }
    if (!firstPortion) {
      firstPortion = std::move(candidate);
      firstPortion->namespaceComponents.push_back(moduleName.str());
    }
    return WalkResult::advance();
  });

  return result ? result : firstPortion;
}

SmallVector<std::string>
ModuleLoader::collectNamespacePortions(const ModuleSpec &parentSpec,
                                       unsigned importBufferFileId) {
  assert(parentSpec.isNamespace() && "expected a namespace spec");
  SmallVector<std::string> portions;
  llvm::StringSet<> seenPortions;
  traverseImportDirectories(importBufferFileId, [&](StringRef dir) {
    if (Filesystem::isMojoSourcePackagePath(dir.str()))
      return WalkResult::advance();
    std::filesystem::path portion(dir.str());
    for (const std::string &component : parentSpec.namespaceComponents) {
      std::error_code ec;
      auto iter = std::filesystem::directory_iterator(portion, ec);
      if (ec)
        return WalkResult::advance();
      // Only a plain directory contributes a portion; a source package (or
      // any other kind) owning this name is closed and resolves by itself.
      std::optional<std::filesystem::path> child;
      for (const auto &entry : iter) {
        if (auto spec = ModuleSpec::classify(entry.path(), component);
            spec && spec->kind == ModuleSpec::Kind::SourceDir) {
          child = entry.path();
          break;
        }
      }
      if (!child)
        return WalkResult::advance();
      portion = std::move(*child);
    }
    // Deduplicate: the buffer-derived working directory may coincide with an
    // include directory, and a duplicate portion must not fake an ambiguity.
    std::error_code ec;
    std::filesystem::path canonical =
        std::filesystem::weakly_canonical(portion, ec);
    std::string key = (ec ? portion.lexically_normal() : canonical).string();
    if (seenPortions.insert(key).second)
      portions.push_back(portion.string());
    return WalkResult::advance();
  });
  return portions;
}

SmallVector<ModuleSpec>
ModuleLoader::resolveNamespaceSubModule(StringRef moduleName,
                                        const ModuleSpec &parentSpec,
                                        unsigned importBufferFileId) {
  // Per portion, in-directory precedence picks one candidate. Across
  // portions, closed candidates win over directory candidates (plain
  // directories carry no marker, so a stray non-Mojo directory must not
  // shadow a real module), and directory candidates merge into a single
  // nested namespace. Every closed candidate is returned: more than one is
  // an ambiguity for the caller to report.
  SmallVector<ModuleSpec> closed;
  std::optional<ModuleSpec> firstDir;
  for (const std::string &portion :
       collectNamespacePortions(parentSpec, importBufferFileId)) {
    std::optional<ModuleSpec> candidate = resolveModulePath(
        moduleName, portion, shared.arePrebuiltPackagesDisabled(),
        /*isInsideSourcePackage=*/false);
    if (!candidate)
      continue;
    if (candidate->kind == ModuleSpec::Kind::SourceDir) {
      if (!firstDir)
        firstDir = std::move(candidate);
      continue;
    }
    closed.push_back(std::move(*candidate));
  }
  if (!closed.empty())
    return closed;

  if (!firstDir)
    return {};
  firstDir->namespaceComponents = parentSpec.namespaceComponents;
  firstDir->namespaceComponents.push_back(moduleName.str());
  return {std::move(*firstDir)};
}

/// Return the directory to use as the "working directory" for relative-ish
/// module lookup. This is the directory containing the given buffer's file,
/// walked up past any enclosing packages, falling back to the process's
/// working directory when the buffer identifier has no existing parent
/// directory. Returns an empty path if no absolute directory could be
/// derived.
static std::filesystem::path
deriveWorkingDirectory(const llvm::SourceMgr &sourceMgr,
                       unsigned importBufferFileId) {
  if (!importBufferFileId)
    return {};

  // The buffer identifier usually names a real file, but REPL and LSP
  // docstring code-block wrapper buffers have synthetic names formed by
  // suffixing the real path (e.g. "foo.mojo wrapper_at(42)"). The identifier
  // itself need not exist - only its parent directory does, which for wrapper
  // buffers is the real file's directory. Identifiers with no usable parent
  // (relative compile inputs, "<stdin>", REPL cells) fall back to the
  // process's working directory.
  std::optional<std::filesystem::path> path;
  if (auto *importBuffer = sourceMgr.getMemoryBuffer(importBufferFileId)) {
    std::filesystem::path identifier(importBuffer->getBufferIdentifier().str());
    if (identifier.has_parent_path() &&
        llvm::sys::fs::exists(identifier.parent_path().string()))
      path = std::move(identifier);
  }

  bool pathFromBuffer = path.has_value();

  // An empty relative path absolutizes to the process's working directory.
  SmallString<256> absolute(path.value_or("").string());
  if (llvm::sys::fs::make_absolute(absolute))
    return {};
  path = absolute.str().str();

  // The buffer's identifier names a file path - real, or a synthetic wrapper
  // name in an existing directory - so step up to its containing directory.
  // The process-CWD fallback is already the directory to search. Either way,
  // work back up to the top-most non-package directory.
  if (pathFromBuffer)
    path = path->parent_path();
  while (Filesystem::isMojoSourcePackagePath(*path))
    path = path->parent_path();
  return *path;
}

void ModuleLoader::traverseImportDirectories(
    unsigned importBufferFileId,
    function_ref<WalkResult(StringRef)> callback) const {
  // Python has lots of magic rules surrounding how modules get resolved. For
  // now, we search the auto-import directories, the working directory derived
  // from the importing buffer, and the source manager's include directories,
  // in that order.
  // Check the auto import directories.
  for (auto &rawPath : autoImportDirs) {
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

  // Check the working directory: the entry point's directory, derived once
  // from the main buffer and visible to every import site alike. No import site
  // sees its own file's directory, meaning that resolution is stable and cannot
  // depend on which file triggered it. A null buffer id requests no working
  // directory at all.
  if (importBufferFileId) {
    llvm::SourceMgr &sourceMgr = shared.getSourceMgr();
    std::filesystem::path cwd =
        deriveWorkingDirectory(sourceMgr, sourceMgr.getMainFileID());
    if (!cwd.empty() && callback(cwd.string()).wasInterrupted())
      return;
  }

  // Check the include directories.
  for (StringRef includeDir : shared.getSourceMgr().getIncludeDirs())
    if (callback(includeDir).wasInterrupted())
      return;
}

//===----------------------------------------------------------------------===//
// Origins
//===----------------------------------------------------------------------===//

static std::string mountPathFor(StringRef boundName, ASTDecl &parentDecl) {
  std::string mount;
  if (SymbolRefAttr parent = parentDecl.getSymbolRef()) {
    mount = parent.getRootReference().str();
    for (FlatSymbolRefAttr nested : parent.getNestedReferences())
      mount += ("." + nested.getValue()).str();
    mount += ".";
  }
  mount += boundName;
  return mount;
}

ErrorOr<ModuleOrigin *> ModuleLoader::getOrCreateModuleOrigin(
    const ModuleSpec &spec, StringRef boundName, ASTDecl &parentDecl) {
  // A namespace is several directories under different import roots, so there
  // is no single entity for it to own.
  if (!spec.isSourceModule() && !spec.isSourcePackage() &&
      !spec.isPrecompiled())
    return nullptr;

  std::string canonicalPath = spec.canonicalPath();
  std::string mount = mountPathFor(boundName, parentDecl);

  auto it = store->originsByCanonicalPath.find(canonicalPath);
  if (it != store->originsByCanonicalPath.end()) {
    ModuleOrigin *existing = it->second;
    if (existing->canonicalMount != mount) {
      return Error(
          Twine{spec.isSourceModule() ? "module" : "package"} +
          " imported as '" + existing->canonicalMount +
          "' must not also be imported as '" + mount +
          "'; remove the duplicate import root or file that reaches it twice");
    }
    return existing;
  }

  store->originAllocations.push_back(
      std::make_unique<ModuleOrigin>(canonicalPath, std::move(mount)));
  ModuleOrigin *origin = store->originAllocations.back().get();
  store->originsByCanonicalPath[canonicalPath] = origin;
  return origin;
}

ArrayRef<std::unique_ptr<ModuleOrigin>> ModuleLoader::getOrigins() const {
  return store->originAllocations;
}

//===----------------------------------------------------------------------===//
// Module states
//===----------------------------------------------------------------------===//

void ModuleLoader::initializeTopLevel(ASTDecl &topLevelDecl) {
  store->topLevelModuleState = std::make_unique<ModuleState>(&topLevelDecl);
  ModuleState &state = *store->topLevelModuleState;
  store->moduleStates[&topLevelDecl] = &state;
  store->packageStates[nullptr] = &state;
}

ModuleState &ModuleLoader::getTopLevelState() const {
  assert(store->topLevelModuleState && "loader has not been initialized");
  return *store->topLevelModuleState;
}

ModuleState *ModuleLoader::lookupState(ASTDecl *decl) const {
  return store->moduleStates.lookup(decl);
}

ModuleState *ModuleLoader::lookupPackageState(PackageOp packageOp) const {
  return store->packageStates.lookup(packageOp);
}

void ModuleLoader::setState(ASTDecl &decl, ModuleState &state) {
  store->moduleStates[&decl] = &state;
}

void ModuleLoader::setPackageState(PackageOp packageOp, ModuleState &state) {
  store->packageStates[packageOp] = &state;
}

void ModuleLoader::eraseState(ASTDecl *decl) {
  store->moduleStates.erase(decl);
}

//===----------------------------------------------------------------------===//
// Loading
//===----------------------------------------------------------------------===//

ModuleState &ModuleLoader::createFileModuleState(
    StringAttr declName, ModuleState &parentState, FileLineColLoc loc,
    llvm::SMLoc declLoc, LexerCursor cursor, LexerCursor endCursor,
    const ModuleSpec &spec) {
  // A module's identity is its position, so one file bound under a second name
  // declares a second copy of every type in it.
  ErrorOr<ModuleOrigin *> originOrErr =
      getOrCreateModuleOrigin(spec, declName.getValue(), *parentState.decl);
  if (const char *originError = originOrErr.getError()) {
    return createErrorModuleState(declLoc.isValid() ? declLoc
                                                    : parentState.importLoc,
                                  declName, *parentState.decl, originError);
  }

  auto moduleBuilder = parentState.decl->getDeclEndBuilder();
  Operation *fileOp = FileModuleOp::create(moduleBuilder, loc, declName);
  // Use createUnlistedDecl (not addDecl) so the module is NOT added to
  // parentState.decl->declsInScope. This prevents "leaky imports"; the module
  // stays navigable via ModuleState::nestedModules.
  ASTDecl &moduleDecl = shared.declResolver->createUnlistedDecl(
      fileOp, declLoc, parentState.decl, cursor, endCursor, /*indentation=*/-1);
  shared.declResolver->registerDeclSymbol(&moduleDecl);

  ModuleState &moduleState = parentState.insertNestedModule(
      declName, std::make_unique<ModuleState>(&moduleDecl, spec));
  moduleState.origin = *originOrErr;
  setState(moduleDecl, moduleState);
  return moduleState;
}

ModuleState &ModuleLoader::createModuleState(
    StringAttr declName, const llvm::MemoryBuffer *moduleBuffer,
    ModuleState &parentState, FileLineColLoc loc, const ModuleSpec &spec) {
  // An eagerly-opened module: its cursor points at the freshly-lexed buffer.
  Lexer lexer(shared.diags, moduleBuffer);
  ModuleState &moduleState = createFileModuleState(
      declName, parentState, loc, lexer.getToken().getLoc(), lexer.getCursor(),
      LexerCursor::getEOF(moduleBuffer), spec);
  // An erroneous state carries no module body, so nothing below applies to it.
  if (moduleState.decl->isErroneous())
    return moduleState;

  // Auto-import the core language modules.
  if (LLVM_LIKELY(shared.hasBuiltinModule()))
    shared.importBuiltinModules(*moduleState.decl);
  shared.notifyListenerOnModuleDecl(*moduleState.decl,
                                    moduleState.decl->getLoc());
  return moduleState;
}

ModuleState &ModuleLoader::createDeferredModuleState(ModuleSpec moduleSpec,
                                                     ModuleState &parentState) {
  // A deferred module: the FileModuleOp + decl exist but its file is NOT
  // opened. The decl carries an invalid cursor; it is opened + lexed on first
  // body resolution, at which point materializeDeferredModule sets its real
  // location.
  assert(moduleSpec.isSourceModule() && "Invalid module state");
  auto declNameAttr = StringAttr::get(shared.getContext(), moduleSpec.name);
  FileLineColLoc loc =
      shared.createLocation(moduleSpec.path.string(), /*line=*/1, /*column=*/1);
  return createFileModuleState(declNameAttr, parentState, loc,
                               /*declLoc=*/SMLoc(),
                               /*cursor=*/LexerCursor(),
                               /*endCursor=*/LexerCursor(), moduleSpec);
}

ModuleState &ModuleLoader::createPackageState(ModuleSpec moduleSpec,
                                              ModuleState &parentState,
                                              SMLoc importLoc) {
  StringAttr declName = StringAttr::get(shared.getContext(), moduleSpec.name);
  // Create a new decl for this module. We use createUnlistedDecl instead of
  // addDecl so the package is NOT added to parentState.decl->declsInScope.
  // This prevents "leaky imports" where importing a sub-module makes the
  // parent package globally accessible. The package is still navigable via
  // ModuleState::nestedModules (populated by insertNestedModule below).
  assert(moduleSpec.isSourcePackageLike() && "Invalid package kind");

  // A package's identity is its position, so the same directory bound under a
  // second name declares a second, incompatible copy of every type in it. A
  // namespace gets no origin, so it is exempt without a kind check here.
  ErrorOr<ModuleOrigin *> originOrErr =
      getOrCreateModuleOrigin(moduleSpec, moduleSpec.name, *parentState.decl);
  if (const char *originError = originOrErr.getError()) {
    return createErrorModuleState(importLoc, declName, *parentState.decl,
                                  originError);
  }

  auto loc = shared.createLocation((moduleSpec.isSourcePackage()
                                        ? moduleSpec.path / "__init__.mojo"
                                        : moduleSpec.path)
                                       .string(),
                                   /*line=*/1, /*column=*/1);
  auto moduleBuilder = parentState.decl->getDeclEndBuilder();
  auto packageOp = PackageOp::create(moduleBuilder, loc, declName);
  // Note we intentionally don't set a valid 'loc' here. The real loc is set
  // if/when the module is actually opened on demand.
  ASTDecl &decl = shared.declResolver->createUnlistedDecl(
      static_cast<Operation *>(packageOp), /*loc=*/SMLoc(), parentState.decl,
      parentState.decl->getCursor(), parentState.decl->getCursor(),
      /*indentation=*/-1);
  // Register the symbol so ModuleType::getDecl() works.
  shared.declResolver->registerDeclSymbol(&decl);

  // Insert the newly created module state.
  ModuleState &moduleState = parentState.insertNestedModule(
      declName, std::make_unique<ModuleState>(&decl, moduleSpec));
  moduleState.importLoc = importLoc;
  // Null for a namespace, which owns no single entity.
  moduleState.origin = *originOrErr;
  setState(decl, moduleState);
  setPackageState(packageOp, moduleState);

  return moduleState;
}

std::string ModuleLoader::moduleMountPath(const ModuleState &root,
                                          const ModuleState &target) {
  for (const auto &[name, nested] : root.nestedModules) {
    if (nested == &target)
      return name.getValue().str();
    std::string subPath = moduleMountPath(*nested, target);
    if (!subPath.empty())
      return (name.getValue() + "." + subPath).str();
  }
  return {};
}

ModuleState &ModuleLoader::createBinaryPackageState(SMLoc loc,
                                                    const ModuleSpec &spec,
                                                    ModuleState &parentState) {
  std::string pathStr = spec.path.string();
  auto declNameAttr = StringAttr::get(shared.getContext(), spec.name);
  auto makeError = [&](const Twine &msg) -> ModuleState & {
    return createErrorModuleState(loc, declNameAttr, *parentState.decl, msg);
  };

  // Symbol references recorded in the artifact are rooted at its compiled
  // name, which resolves only for a top-level binding of that name; mounted
  // below the top level, every type escaping the package is unresolvable.
  // TODO(MOCO-4487): lift this once loading re-anchors recorded roots to the
  // mount point.
  if (parentState.decl != &shared.getTopLevelDecl()) {
    std::string mountPath = moduleMountPath(getTopLevelState(), parentState);
    if (!mountPath.empty())
      mountPath += ".";
    mountPath += spec.name;
    return makeError("precompiled package '" + pathStr +
                     "' must be imported directly from an import root, not "
                     "as '" +
                     mountPath + "'");
  }

  // One artifact bound under two names is two packages, and every type it
  // declares exists twice over.
  ErrorOr<ModuleOrigin *> originOrErr =
      getOrCreateModuleOrigin(spec, spec.name, *parentState.decl);
  if (const char *originError = originOrErr.getError())
    return makeError(originError);

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> packageBuffer =
      llvm::MemoryBuffer::getFile(pathStr);
  if (!packageBuffer)
    return makeError("unable to open package file '" + pathStr + "'");

  // Read the cached package.
  OpBuilder builder = parentState.decl->getDeclEndBuilder();
  Block *block = builder.getBlock();
  // bytecodeReader refers to sourceMgr by reference,
  // so sourceMgr lifetime must be same or longer.
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  std::unique_ptr<mlir::BytecodeReader> bytecodeReader;
  {
    CompilerTimeTraceScope timeScope("readBytecodeFile");
    // Create a source manager to extend the lifetime of the package buffer.
    sourceMgr->AddNewSourceBuffer(std::move(*packageBuffer), SMLoc());
    const llvm::MemoryBuffer *memoryBuf =
        sourceMgr->getMemoryBuffer(sourceMgr->getMainFileID());

    auto mlirBufOrErr = getMLIRBufferFromPrecompiledFile(
        *memoryBuf, shared.options.ignoreIncompatiblePrecompiledFileErrors);
    if (mlirBufOrErr.isError())
      return makeError(mlirBufOrErr.takeError().get());
    auto mlirResult = std::move(*mlirBufOrErr);
    // If the package was compressed, add the decompressed buffer to the source
    // manager to extend its lifetime beyond this scope.
    if (mlirResult.ownedData)
      sourceMgr->AddNewSourceBuffer(std::move(mlirResult.ownedData), SMLoc());

    // TODO(MOCO-522): Arcana docs on this lazy loading.
    bytecodeReader = std::make_unique<mlir::BytecodeReader>(
        mlirResult.buffer, shared.getBytecodeParserConfig(),
        /*lazyLoad=*/true, sourceMgr);

    // Read in the cached bytecode.
    if (failed(bytecodeReader->readTopLevel(block)))
      return makeError("unable to load package '" + pathStr + "'");

    // Add the package path to the set of included files.
    shared.addIncludedFile(pathStr);
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
  auto theModule =
      cast_or_null<ModuleOp>(shared.getTopLevelDecl().getIfOperation());
  for (auto thunk : llvm::make_early_inc_range(tmpModule.getOps<FnOp>())) {
    Attribute key = thunk.getThunkKeyAttr();
    assert(key && "thunk is missing its key");
    if (!shared.tryRegisterConversionThunk(key, thunk))
      continue; // thunk already exists

    // Move the thunk into the top-level and add it as fully resolved.
    if (failed(bytecodeReader->materialize(thunk)))
      return makeError("failed to materialize function thunk");
    thunk->remove();
    theModule.push_back(thunk);
    ASTDecl &thunkDecl = shared.declResolver->addBytecodeDecl(
        &*thunk, thunk.getSourceNameAttr(), &shared.getTopLevelDecl(),
        DeclResolvedness::body);
    shared.declResolver->finalizeFuncSignature(thunk, thunkDecl);
  }
  for (auto trait :
       llvm::make_early_inc_range(tmpModule.getOps<TraitDeclOp>())) {
    if (!trait.getClosureSignature().has_value())
      continue;

    FnTypeGeneratorType key = *trait.getClosureSignature();
    auto creation = [&]() -> ASTDecl * {
      if (failed(bytecodeReader->materialize(trait)))
        return nullptr;
      // A closure trait with no methods is a stub from a package that
      // references but does not define the closure type. Skip it so the cache
      // slot stays empty and a later package with the full body can fill it.
      if (trait.getOps<FnOp>().empty())
        return nullptr;
      trait->remove();
      theModule.push_back(trait);
      ASTDecl &traitDecl = shared.declResolver->addBytecodeDecl(
          &*trait, trait.getSymNameAttr(), &shared.getTopLevelDecl(),
          DeclResolvedness::body);
      traitDecl.setTypeDeclSelf(ASTDecl::computeSelfTypeForTrait(trait));
      // Ensure that the trait's methods are registered, too.
      for (auto fn : trait.getOps<FnOp>()) {
        shared.declResolver->addBytecodeDecl(
            fn, fn.getSourceNameAttr(), &traitDecl, DeclResolvedness::body);
      }
      return &traitDecl;
    };
    shared.getClosureEmitter().getOrCreateClosureTrait(key, creation);
  }
  // Insert a new module decl. Use createUnlistedDecl instead of addBytecodeDecl
  // so the package is NOT added to parentState.decl->declsInScope.
  ASTDecl &decl = shared.declResolver->createUnlistedDecl(
      static_cast<Operation *>(packageOp),
      shared.diags.convertLocToSMLoc(packageOp->getLoc()), parentState.decl,
      LexerCursor(), LexerCursor(), /*indentation=*/-1);
  decl.loadedFromBytecode = true;
  decl.resolvedness = DeclResolvedness::signature;
  shared.declResolver->registerDeclSymbol(&decl);

  // Initialize the module state.
  ModuleState &moduleState = parentState.insertNestedModule(
      declNameAttr, std::make_unique<ModuleState>(&decl, spec));
  // Remember where this package was imported. The package's source files are
  // only opened at diagnostic time (they aren't parsed here), so when a decl
  // from this package is lazily materialized we use this to set its location
  // at the import site.
  moduleState.importLoc = loc;
  moduleState.origin = *originOrErr;

  // The reader and the buffers under it belong to the file, so every module
  // bound out of this artifact reaches them through the shared origin. The
  // module cache means an artifact is only ever read once.
  assert(moduleState.origin && "precompiled artifact without an origin");
  ModuleOrigin &origin = *moduleState.origin;
  assert(!origin.bytecodeReader && "artifact read twice");
  origin.bytecodeReader = std::move(bytecodeReader);
  // keep buffer alive for deferred materialize
  origin.sourceMgr = sourceMgr;
  origin.tmpModule = tmpModule;
  origin.bytecodeImportLoc = loc;

  setState(decl, moduleState);
  setPackageState(cast_or_null<PackageOp>(decl.getIfOperation()), moduleState);

  return moduleState;
}

ModuleState &ModuleLoader::createErrorModuleState(SMLoc loc, StringAttr name,
                                                  ASTDecl &errorContext,
                                                  const Twine &errorMsg,
                                                  bool unlisted,
                                                  const Twine &note) {
  // Track the failure in the scope whose lookup failed.
  ModuleState *contextState = lookupState(&errorContext);
  if (!contextState)
    contextState = &getTopLevelState();

  if (!contextState->failedImports) {
    contextState->failedImports.reset(
        new DenseMap<StringAttr, std::unique_ptr<ModuleState>>());
  }
  std::unique_ptr<ModuleState> &state = (*contextState->failedImports)[name];
  if (!state) {
    ASTDecl *decl = &shared.declResolver->addErroneousDecl(
        name, loc, &errorContext, unlisted);
    state = std::make_unique<ModuleState>(decl);
    setState(*decl, *state);
  }

  // Report errors once per import site. This data is lazily allocated.
  if (!state->reportedFailureLocs)
    state->reportedFailureLocs.reset(new SmallVector<SMLoc>());
  if (!llvm::is_contained(*state->reportedFailureLocs, loc)) {
    state->reportedFailureLocs->push_back(loc);
    MojoInflightDiag diag = shared.emitError(loc, errorMsg);
    if (!note.isTriviallyEmpty())
      diag.attachNote(loc) << note;
  }
  return *state;
}
