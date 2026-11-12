//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
#ifndef GENERICML_GRAPHCOMPILER_SUPPORT_MOJOFILE_H
#define GENERICML_GRAPHCOMPILER_SUPPORT_MOJOFILE_H

#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"

#include <string>

namespace M::MOGG {

class MojoFile;

/// Switches `MojoFile` to its declarations section for the duration of the
/// scope, and restores the previous section on exit. Scopes nest: one created
/// inside another saves and restores the declarations section itself.
class DeclarationScope {
public:
  explicit DeclarationScope(MojoFile &file);
  DeclarationScope(const DeclarationScope &) = delete;
  DeclarationScope &operator=(const DeclarationScope &) = delete;
  ~DeclarationScope();

private:
  MojoFile &file;
  MojoBuilder *previous;
};

/// The Mojo source file being emitted: function bodies in order, then the
/// module-scope declarations they need. Mojo allows declarations and function
/// bodies to appear in any order, so the declarations are appended after the
/// bodies that discovered them.
class MojoFile {
public:
  /// The builder receiving text now, so an emitter writes
  /// `state.file->var(...)`: the body, or the declarations section while an
  /// `enterDeclarations()` scope is alive.
  MojoBuilder *operator->() { return current; }
  MojoBuilder &operator*() { return *current; }

  /// The declarations section, for writing text that emits no operations.
  MojoBuilder &declarations() { return declarationsBuilder; }

  /// For a declaration whose body is emitted from IR: until the returned scope
  /// ends, `operator->()` is the declarations section.
  [[nodiscard]] DeclarationScope enterDeclarations() {
    return DeclarationScope(*this);
  }

  /// Whether a declaration is being emitted right now.
  bool inDeclarations() const { return current == &declarationsBuilder; }

  /// The body, then the declarations.
  std::string str() const { return body.str() + declarationsBuilder.str(); }

private:
  friend class DeclarationScope;

  MojoBuilder body;
  MojoBuilder declarationsBuilder;
  MojoBuilder *current = &body;
};

inline DeclarationScope::DeclarationScope(MojoFile &file)
    : file(file), previous(file.current) {
  file.current = &file.declarationsBuilder;
}

inline DeclarationScope::~DeclarationScope() { file.current = previous; }

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_SUPPORT_MOJOFILE_H
