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
// Command-list / access-path IR for `match` pattern preprocessing. Each case
// lowers to a straight-line sequence of commands (equality / enum-tag / or /
// bind).
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_PATTERNMATCHIR_H
#define KGEN_MOJOPARSER_PATTERNMATCHIR_H

#include "Mojo/MojoParser/ExprDest.h"
#include "Mojo/MojoParser/MojoDiags.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"

#include <cstring>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace M::KGEN::LIT {
using llvm::raw_ostream;
class ASTDecl;
class ExprNode;
class IREmitter;
class SharedState;

/// Uniqued access path from the match subject (keyed by pointer address).
struct PatternPath {
  enum Kind { Root, TupleElement, StructField, EnumPayload };
  Kind kind;
  const PatternPath *parent = nullptr; ///< Null for `Root`.
  ASTType type;
  size_t index = 0;     ///< TupleElement / EnumPayload.
  StringAttr fieldName; ///< StructField.

  void print(raw_ostream &os) const;
  LLVM_DUMP_METHOD void dump() const;
};

struct PatternCommand;

/// Immutable command-list view for bump-stored Or alternatives.
struct PatternCommandListRef {
  ArrayRef<const PatternCommand *> commands;

  void print(raw_ostream &os, unsigned indent = 0) const;
  LLVM_DUMP_METHOD void dump() const;
};

/// One step in a case program: test the subject at `path`, nest alternatives,
/// or record a name binding for later emission.
struct PatternCommand {
  enum Kind { Equal, EnumTag, Or, Bind };

  Kind kind;
  const PatternPath *path = nullptr;
  const ExprNode *expr = nullptr; ///< Source pattern (diags / later emit).

  // Equal / EnumTag / Or.
  size_t enumCaseIndex = 0;
  ArrayRef<PatternCommandListRef> orAlternatives;

  // Bind.
  StringRef bindName;
  PatternDeclKind declKind = PatternDeclKind::kNone;

  void print(raw_ostream &os, unsigned indent = 0) const;
  LLVM_DUMP_METHOD void dump() const;
};

/// Name bound by a successful pattern; materialized by the match statement.
struct PatternBoundName {
  StringRef name;
  CValue value;
  PatternDeclKind bindingKind;
};

/// Mutable case program (stack-owned; not bump-allocated).
struct PatternCommandList {
  SmallVector<const PatternCommand *, 8> commands;

  /// Emit HLCF match tests for this case against `subject` at `rootPath`.
  /// On mismatch emits `hlcf.match.next`; on success falls through. Appends
  /// bindings for the caller to materialize (before any guard / body).
  LogicalResult emit(IREmitter &emitter, CValue subject,
                     const PatternPath *rootPath,
                     SmallVectorImpl<PatternBoundName> &bindings) const;

  void print(raw_ostream &os, unsigned indent = 0) const;
  LLVM_DUMP_METHOD void dump() const;
};

/// Bump storage, path uniquing, and binding-mode state for one `__match`.
class PatternMatchBuilder {
public:
  PatternMatchBuilder(ASTDecl &declScope, ExprContext paramContext);

  llvm::BumpPtrAllocator allocator;
  ASTDecl &declScope;
  ExprContext paramContext;
  SharedState &shared;

  IREmitter getParamEmitter();

  PatternDeclKind getPatternKind() const { return patternKind; }
  void setPatternKind(PatternDeclKind kind) { patternKind = kind; }
  PatternDeclKind &patternKindRef() { return patternKind; }

  MojoInflightDiag emitError(SMLoc loc, const Twine &message = {});
  MojoInflightDiag emitWarning(SMLoc loc, const Twine &message = {});

  template <typename T, typename... Args>
  T *create(Args &&...args) {
    return new (allocator.Allocate(sizeof(T), alignof(T)))
        T(std::forward<Args>(args)...);
  }

  template <typename T>
  ArrayRef<T> internArray(ArrayRef<T> values) {
    if (values.empty())
      return {};
    T *storage = static_cast<T *>(
        allocator.Allocate(sizeof(T) * values.size(), alignof(T)));
    std::memcpy(storage, values.data(), sizeof(T) * values.size());
    return {storage, values.size()};
  }

  const PatternPath *getRootPath(ASTType type) {
    return getOrCreatePath(PatternPath::Root, nullptr, type, 0, {});
  }
  const PatternPath *getTupleElement(const PatternPath *parent, size_t index,
                                     ASTType elemType) {
    return getOrCreatePath(PatternPath::TupleElement, parent, elemType, index,
                           {});
  }
  const PatternPath *getStructField(const PatternPath *parent,
                                    StringAttr fieldName, ASTType fieldType) {
    return getOrCreatePath(PatternPath::StructField, parent, fieldType, 0,
                           fieldName);
  }
  const PatternPath *getEnumPayload(const PatternPath *parent, size_t caseIndex,
                                    ASTType payloadType) {
    return getOrCreatePath(PatternPath::EnumPayload, parent, payloadType,
                           caseIndex, {});
  }

  PatternCommand *createEqual(const PatternPath *path, const ExprNode *expr) {
    auto *cmd = create<PatternCommand>();
    cmd->kind = PatternCommand::Equal;
    cmd->path = path;
    cmd->expr = expr;
    return cmd;
  }
  PatternCommand *createEnumTag(const PatternPath *path, const ExprNode *expr,
                                size_t caseIndex) {
    auto *cmd = create<PatternCommand>();
    cmd->kind = PatternCommand::EnumTag;
    cmd->path = path;
    cmd->expr = expr;
    cmd->enumCaseIndex = caseIndex;
    return cmd;
  }
  PatternCommand *createOr(const PatternPath *path, const ExprNode *expr,
                           ArrayRef<PatternCommandListRef> alternatives) {
    auto *cmd = create<PatternCommand>();
    cmd->kind = PatternCommand::Or;
    cmd->path = path;
    cmd->expr = expr;
    cmd->orAlternatives = internArray(alternatives);
    return cmd;
  }
  PatternCommand *createBind(const PatternPath *path, const ExprNode *expr,
                             StringRef name, PatternDeclKind declKind) {
    auto *cmd = create<PatternCommand>();
    cmd->kind = PatternCommand::Bind;
    cmd->path = path;
    cmd->expr = expr;
    cmd->bindName = name;
    cmd->declKind = declKind;
    return cmd;
  }

  PatternCommandListRef internCommandList(const PatternCommandList &list) {
    return {internArray(ArrayRef<const PatternCommand *>(list.commands))};
  }

private:
  struct PathKey {
    const PatternPath *parent;
    PatternPath::Kind kind;
    size_t index;
    Attribute fieldName;
    bool operator==(const PathKey &o) const {
      return parent == o.parent && kind == o.kind && index == o.index &&
             fieldName == o.fieldName;
    }
  };
  struct PathKeyInfo {
    static PathKey getEmptyKey() {
      return {reinterpret_cast<const PatternPath *>(~uintptr_t(0)),
              PatternPath::Root,
              0,
              {}};
    }
    static PathKey getTombstoneKey() {
      return {reinterpret_cast<const PatternPath *>(~uintptr_t(1)),
              PatternPath::Root,
              0,
              {}};
    }
    static unsigned getHashValue(const PathKey &key) {
      return llvm::hash_combine(key.parent, unsigned(key.kind), key.index,
                                key.fieldName);
    }
    static bool isEqual(const PathKey &a, const PathKey &b) { return a == b; }
  };

  const PatternPath *getOrCreatePath(PatternPath::Kind kind,
                                     const PatternPath *parent, ASTType type,
                                     size_t index, StringAttr fieldName) {
    PathKey key{parent, kind, index, fieldName};
    const PatternPath *&slot = pathUniques[key];
    if (slot)
      return slot;
    auto *path = create<PatternPath>();
    path->kind = kind;
    path->parent = parent;
    path->type = type;
    path->index = index;
    path->fieldName = fieldName;
    return slot = path;
  }

  DenseMap<PathKey, const PatternPath *, PathKeyInfo> pathUniques;
  PatternDeclKind patternKind = PatternDeclKind::kBind;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_PATTERNMATCHIR_H
