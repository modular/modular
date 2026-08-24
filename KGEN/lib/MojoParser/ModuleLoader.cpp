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

#include "ModuleStore.h"

#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/Support/Configuration.h"
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

ModuleLoader::ModuleLoader(SharedState &shared) : SharedStateUser(shared) {
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

  auto it = originsByCanonicalPath.find(canonicalPath);
  if (it != originsByCanonicalPath.end()) {
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

  originAllocations.push_back(
      std::make_unique<ModuleOrigin>(canonicalPath, std::move(mount)));
  ModuleOrigin *origin = originAllocations.back().get();
  originsByCanonicalPath[canonicalPath] = origin;
  return origin;
}
