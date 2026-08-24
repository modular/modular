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
//
// Finding modules and packages on disk.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_MODULELOADER_H
#define KGEN_MOJOPARSER_MODULELOADER_H

#include "KGEN/MojoParser/ModuleSpec.h"
#include "KGEN/MojoParser/SharedState.h"

#include <memory>
#include <optional>

namespace M::KGEN::LIT {

/// Module & package handling logic.
///
/// Its primary purpose is resolution: given an import path, work out what it
/// names on disk and describe it as a `ModuleSpec`.
class ModuleLoader : public SharedStateUser {
public:
  ModuleLoader(SharedState &shared);
  ~ModuleLoader();

  /// Resolve the absolute path for a given module name. Returns nullopt if the
  /// module cannot be found.
  std::optional<ModuleSpec> resolveModulePath(StringRef moduleName,
                                              llvm::SMLoc includeLoc);

  /// Resolve the absolute path for a given module name within the provided
  /// directory. Returns nullopt if the module cannot be found.
  ///
  /// \p isInsideSourcePackage is true for submodule search inside a source
  /// package. In that mode, any `.mojoc` candidate is ignored.
  std::optional<ModuleSpec> resolveModulePath(StringRef moduleName,
                                              StringRef includeDir,
                                              bool ignorePrebuilt,
                                              bool isInsideSourcePackage);

  /// Collect the portion directories of the namespace described by
  /// `parentSpec`: for each import directory visible from
  /// `importBufferFileId`, descend the spec's namespace components as plain
  /// directories. Portions are returned in traversal order, deduplicated.
  ///
  /// A "portion" is one root's directory contributing to the namespace. For the
  /// example on `namespaceComponents`, the portions of `foo.bar` are
  ///
  ///   [ one/foo/bar, two/foo/bar ]
  ///
  /// The search path always contains the same directories for every
  /// import site, and portions are recomputed from it at each resolution
  /// step, so a root where `foo` is missing, or is claimed by a source
  /// package or module file (a closed candidate), simply contributes no
  /// portion.
  SmallVector<std::string>
  collectNamespacePortions(const ModuleSpec &parentSpec,
                           unsigned importBufferFileId);

  /// Resolve the name as a submodule of the namespace described by
  /// `parentSpec`, searching every portion. Returns every distinct thing the
  /// name could be: closed (non-directory) candidates win over directory
  /// candidates, which merge into a single nested-namespace spec rather than
  /// competing. More than one element therefore means the import is
  /// ambiguous; several portions provide a closed candidate.
  SmallVector<ModuleSpec>
  resolveNamespaceSubModule(StringRef moduleName, const ModuleSpec &parentSpec,
                            unsigned importBufferFileId);

  /// Traverse the directories available for importing modules and packages,
  /// calling the given callback for each directory found.
  void
  traverseImportDirectories(unsigned importBufferFileId,
                            function_ref<WalkResult(StringRef)> callback) const;

  /// Returns the ModuleOrigin that the spec names, shared by every binding of
  /// the same entity.
  ///
  /// The bound name is in the given parent decl's scope. An origin can carry
  /// one symbol path only, since re-anchoring an artefact rewrites its
  /// references to a single path. Binding one entity under two names is a
  /// dual-mount error.
  ///
  /// Null is a success: a namespace spans several import roots, so names no
  /// single entity.
  ErrorOr<ModuleOrigin *> getOrCreateModuleOrigin(const ModuleSpec &spec,
                                                  StringRef boundName,
                                                  ASTDecl &parentDecl);

  /// Every origin created so far, in creation order.
  ArrayRef<std::unique_ptr<ModuleOrigin>> getOrigins() const;

  //===--------------------------------------------------------------------===//
  // Module states
  //===--------------------------------------------------------------------===//

  /// Create the state the whole import tree nests inside. Called once, when
  /// the top-level decl exists.
  void initializeTopLevel(ASTDecl &topLevelDecl);

  /// The state of the top-level decl, which every import is nested within.
  ModuleState &getTopLevelState() const;

  /// The state this decl was imported as, or null if it names no module.
  ModuleState *lookupState(ASTDecl *decl) const;

  /// The state this package op was imported as, or null. Distinct from
  /// `lookupState` only because one op can be reached through several decls.
  ModuleState *lookupPackageState(PackageOp packageOp) const;

  /// Record `state` as what `decl` was imported as.
  void setState(ASTDecl &decl, ModuleState &state);

  /// Record `state` as what `packageOp` was imported as.
  void setPackageState(PackageOp packageOp, ModuleState &state);

  /// Forget the state of a decl whose import turned out to have failed.
  void eraseState(ASTDecl *decl);

  //===--------------------------------------------------------------------===//
  // Loading
  //===--------------------------------------------------------------------===//

  /// Shared core of createModuleState and createDeferredModuleState: create the
  /// `FileModuleOp` + unlisted decl + nested module state. The caller supplies
  /// the parse cursor (valid for an already-open module, invalid for a deferred
  /// one) and is responsible for importing builtins (eagerly, or at
  /// materialization for a deferred module).
  ModuleState &createFileModuleState(StringAttr declName,
                                     ModuleState &parentState,
                                     FileLineColLoc loc, llvm::SMLoc declLoc,
                                     LexerCursor cursor, LexerCursor endCursor,
                                     const ModuleSpec &spec);

  /// Create a new module state with the given name, location, and body.
  ModuleState &createModuleState(StringAttr declName,
                                 const llvm::MemoryBuffer *moduleBuffer,
                                 ModuleState &parentState, FileLineColLoc loc,
                                 const ModuleSpec &spec);

  /// Create a module state for a source module whose file has not been opened.
  ModuleState &createDeferredModuleState(ModuleSpec moduleSpec,
                                         ModuleState &parentState);

  /// Create a new module state for a package with the given spec, location,
  /// and body. The importLoc, if valid, is the location of the `import` that
  /// pulled the package in. The spec's kind must be a SourcePackage or
  /// SourceDir.
  ModuleState &createPackageState(ModuleSpec moduleSpec,
                                  ModuleState &parentState, SMLoc importLoc);

  /// Create a new module state for a binary package with the given spec.
  ModuleState &createBinaryPackageState(SMLoc loc, const ModuleSpec &spec,
                                        ModuleState &parentState);

  /// Returns the dotted module path of `target` below `root`, or an empty
  /// string when `target` is not nested below `root`.
  static std::string moduleMountPath(const ModuleState &root,
                                     const ModuleState &target);

  /// Create an error module state and emit the given error message. If
  /// `unlisted` is set, the erroneous decl is not registered in
  /// `errorContext`'s name table. A non-empty `note` is attached to the error.
  ModuleState &createErrorModuleState(SMLoc loc, StringAttr name,
                                      ASTDecl &errorContext,
                                      const Twine &errorMsg,
                                      bool unlisted = false,
                                      const Twine &note = {});

private:
  /// The directories searched before the working directory and the source
  /// manager's include directories: configured search paths, or the defaults
  /// from the Mojo config when none were given.
  SmallVector<std::string> autoImportDirs;

  /// What has been imported so far.
  std::unique_ptr<ModuleStore> store;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_MODULELOADER_H
