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

#include <optional>

namespace M::KGEN::LIT {

/// Module & package handling logic.
///
/// Its primary purpose is resolution: given an import path, work out what it
/// names on disk and describe it as a `ModuleSpec`.
class ModuleLoader : public SharedStateUser {
public:
  ModuleLoader(SharedState &shared);

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

private:
  /// The directories searched before the working directory and the source
  /// manager's include directories: configured search paths, or the defaults
  /// from the Mojo config when none were given.
  SmallVector<std::string> autoImportDirs;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_MODULELOADER_H
