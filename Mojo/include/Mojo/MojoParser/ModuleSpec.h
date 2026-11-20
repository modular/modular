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
// The description of an import: the path an import asks for, and the
// importable things found on disk.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_MODULESPEC_H
#define KGEN_MOJOPARSER_MODULESPEC_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

#include <filesystem>
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace M::KGEN::LIT {

class ImportPathAttr;

/// A parsed import path: module-path components plus the number of leading
/// dots of a relative import (0 for an absolute import).
struct ImportPath {
  ImportPath() = default;
  ImportPath(std::initializer_list<llvm::StringRef> components,
             unsigned relativeLevel = 0)
      : components(components), relativeLevel(relativeLevel) {}
  ImportPath(llvm::ArrayRef<llvm::StringRef> components,
             unsigned relativeLevel = 0)
      : components(components.begin(), components.end()),
        relativeLevel(relativeLevel) {}

  /// Render the path as a Mojo ASTDecl name: the joined dotted-string form
  /// used for decl and import-op names. It is ambiguous for components that
  /// themselves contain periods (e.g. escaped identifiers) and is not a
  /// source-faithful rendering for diagnostics.
  std::string toDottedString() const {
    return std::string(relativeLevel, '.') + llvm::join(components, ".");
  }

  /// Convert to/from the unambiguous `#lit.import_path` attribute form,
  /// which import ops should prefer over the dotted-string forms above.
  ImportPathAttr toAttr(mlir::MLIRContext *ctx) const;
  static ImportPath fromAttr(ImportPathAttr attr);

  llvm::SmallVector<llvm::StringRef, 4> components;
  unsigned relativeLevel = 0;
};

/// One importable thing found on disk: what it is called, where it is, and what
/// kind of thing it is. This is the result of resolution, not a record of
/// anything loaded - several specs can name the same entity, and a spec for a
/// file that is never imported is simply discarded.
struct ModuleSpec {
  /// Mojo import kinds. Enumerator order defines resolution priority: a
  /// lower value outranks a higher one when several candidates share a stem.
  enum class Kind {
    SourcePackage,
    Precompiled,
    SourceModule,
    SourceDir,
  };

  /// Importable name: whole filename for directories, stem for files
  std::string name;
  std::filesystem::path path;
  Kind kind;

  /// For a SourceDir resolved from the import path: the dotted chain of
  /// directory names (including `name`) relative to the import roots. A
  /// plain directory is a namespace whose one name may span several roots, so
  /// submodules resolve against these components under every import root, not
  /// against `path`, which records only the first root's portion. Empty for
  /// every other kind, and for directories nested inside a source package,
  /// which stay single-homed and resolve through `path`.
  ///
  /// For example, given `-I one -I two` and
  ///
  ///   one/foo/bar/baz.mojo
  ///   two/foo/bar/qux.mojo
  ///
  /// the name `foo.bar` is one namespace spanning both roots. Its spec is
  ///
  ///   { name = "bar", path = "one/foo/bar", kind = SourceDir,
  ///     namespaceComponents = ["foo", "bar"] }
  ///
  /// and resolving `foo.bar.qux` searches `<root>/foo/bar` under every
  /// root, finding two/foo/bar/qux.mojo even though `path` names the
  /// first root's directory. The closed candidate baz.mojo gets
  ///
  ///   { name = "baz", path = "one/foo/bar/baz.mojo", kind = SourceModule,
  ///     namespaceComponents = [] }
  std::vector<std::string> namespaceComponents = {};

  bool isNamespace() const {
    return kind == Kind::SourceDir && !namespaceComponents.empty();
  }

  /// Return the module classification of a given path, optionally matching a
  /// specific name. Returns std::nullopt if not a (matching) ModuleSpec.
  static std::optional<ModuleSpec> classify(const std::filesystem::path &path,
                                            llvm::StringRef moduleName = "");

  /// Return true if the import candidate (kind, path) takes precedence over
  /// the other. Higher-priority kinds win; between candidates of
  /// the same kind (e.g. directories foo and foo.bar, which share the stem
  /// 'foo'), the lexicographically smaller filename wins so the result does
  /// not depend on the platform's directory iteration order.
  bool takesImportPrecedence(ModuleSpec other) {
    if (kind != other.kind)
      return kind < other.kind;
    return path.filename() < other.path.filename();
  }

  bool isPrecompiled() const { return kind == ModuleSpec::Kind::Precompiled; }

  bool isSourcePackage() const {
    return kind == ModuleSpec::Kind::SourcePackage;
  }

  bool isSourcePackageLike() const {
    return kind == ModuleSpec::Kind::SourcePackage ||
           kind == ModuleSpec::Kind::SourceDir;
  }

  bool isSourceModule() const { return kind == ModuleSpec::Kind::SourceModule; }

  /// The canonical form of `path`, with symlinks resolved: the identity of
  /// the entity this spec names.
  std::string canonicalPath() const;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_MODULESPEC_H
