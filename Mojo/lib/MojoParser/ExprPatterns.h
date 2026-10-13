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

#ifndef KGEN_MOJOPARSER_EXPRPATTERNS_H
#define KGEN_MOJOPARSER_EXPRPATTERNS_H

#include "Mojo/MojoParser/ExprDest.h"
#include "Mojo/MojoParser/Lexer.h"

#include "Support/BumpAllocator.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Compiler.h"

#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
class OpBuilder;
} // namespace mlir

namespace M::KGEN::LIT {
using llvm::raw_ostream;
class ASTDecl;
class ExprNode;
class IREmitter;
class SharedState;

/// Uniqued access path from the match subject (keyed by pointer address).
struct PatternPath {
  enum Kind {
    Root,
    TupleElement,
    StructField,
    EnumPayload,
    ListElement,        // [x, ...]  where index is the index in the list.
    ListElementFromEnd, // [.. *y, x] whehere index is the offset from the end.
    ListSlice, // [a, *x, c] where index is the start and endOffset is the end.
  };
  Kind kind;
  const PatternPath *parent = nullptr; ///< Null for `Root`.
  ASTType type;
  size_t index = 0; ///< Element/payload index or slice start.
  union {
    size_t endOffset = 0; ///< Elements excluded from the end of a list slice.
    StringAttr fieldName; ///< StructField.
  };

  void print(raw_ostream &os) const;
  LLVM_DUMP_METHOD void dump() const;
};

struct PatternCommand;
using PatternCommandList = ArrayRef<const PatternCommand *>;

/// One step in a case program: test the subject at `path`, nest alternatives,
/// or record a name binding for later emission.
struct PatternCommand {
  enum Kind { Equal, EnumTag, ListLength, Or, Bind };

  Kind kind;
  /// The pattern path indicates what subobject is being tested.
  const PatternPath *path = nullptr;
  const ExprNode *expr = nullptr; ///< Source pattern (diags / later emit).

  // ListLength.
  struct ListLengthData {
    size_t minimumLength;
    bool exactLength;
  };
  struct BindData {
    StringRef bindName;
    PatternDeclKind declKind;
  };
  struct EmptyData {};
  union {
    // Equal.
    EmptyData emptyData;
    // EnumTag.
    size_t enumCaseIndex;
    // ListLength.
    ListLengthData listLengthData;
    // Or.
    ArrayRef<PatternCommandList> orAlternatives;
    // Bind.
    BindData bindData;
  };

  static PatternCommand getEqual(const PatternPath *path,
                                 const ExprNode *expr) {
    return {.kind = Equal, .path = path, .expr = expr, .emptyData = {}};
  }
  static PatternCommand getEnumTag(const PatternPath *path,
                                   const ExprNode *expr, size_t enumCaseIndex) {
    return {.kind = EnumTag,
            .path = path,
            .expr = expr,
            .enumCaseIndex = enumCaseIndex};
  }
  static PatternCommand getListLength(const PatternPath *path,
                                      const ExprNode *expr,
                                      size_t minimumLength, bool exactLength) {
    return {.kind = ListLength,
            .path = path,
            .expr = expr,
            .listLengthData = {minimumLength, exactLength}};
  }
  static PatternCommand getOr(const PatternPath *path, const ExprNode *expr,
                              ArrayRef<PatternCommandList> orAlternatives) {
    return {.kind = Or,
            .path = path,
            .expr = expr,
            .orAlternatives = orAlternatives};
  }
  static PatternCommand getBind(const PatternPath *path, const ExprNode *expr,
                                StringRef bindName, PatternDeclKind declKind) {
    return {.kind = Bind,
            .path = path,
            .expr = expr,
            .bindData = {bindName, declKind}};
  }

  void print(raw_ostream &os, unsigned indent = 0) const;
  LLVM_DUMP_METHOD void dump() const;
};

/// Name bound by a successful pattern; materialized by the match statement.
struct PatternBoundName {
  StringRef name;
  CValue value;
  PatternDeclKind bindingKind;
  /// Shared declaration already emitted for an Or-pattern binding.
  VarDeclOp declaration{};
};

/// Parsed `case` arm for `match`: pattern, optional guard, suite cursor, and
/// the preprocessed command list for the pattern.  This is used for reasoning
/// about entire cases and sets thereof.
struct MatchCaseEntry {
  ExprNode *patternExpr;
  ExprNode *guardExpr;
  LexerCursor caseCursor;
  size_t caseIndent;
  /// Indentation of the case body; appended arms copy it.
  size_t bodyIndent;
  PatternCommandList commandList;

  /// This is set to true if exclusivity checking finds that the case is
  /// dynamically unreachable because previous cases cover it.
  bool isUnreachable = false;

  /// True when this case completes coverage of the match subject (an
  /// irrefutable pattern, or the last remaining EnumLike constructor). Later
  /// cases are unreachable, and IR emission should not fall through past this
  /// case — there is no remaining value that could miss every arm.
  bool isConcluding = false;

  /// True when every command is irrefutable (`Bind`), including the empty `_`
  /// pattern, and there is no guard. Such a case cannot fail once control
  /// reaches it.
  ///
  /// When `skipFirstCommand` is set, the leading test is ignored and only the
  /// residual is checked — used to decide whether a trailing `_` is the
  /// exclusive complement of a leading-value cluster.
  bool alwaysMatches(bool skipFirstCommand = false) const {
    // Cases with a guard expression can always fail.
    if (guardExpr)
      return false;
    ArrayRef<const PatternCommand *> commands = commandList;
    if (skipFirstCommand) {
      assert(!commands.empty() &&
             "skipFirstCommand requires a leading command");
      commands = commands.drop_front();
    }
    // Name bindings always succeed, so we can ignore them.
    return llvm::all_of(commands, [](const PatternCommand *command) {
      return command->kind == PatternCommand::Bind;
    });
  }
};

/// Emit state for one command-list walk: path→value memoization + CF emission.
/// `curDeclScope` is the scope new emitters are built against; each emit method
/// takes an `IREmitter` (dynamic with a builder, or param-only for comptime).
struct PatternEmitState {
  ASTDecl &curDeclScope;
  CValue rootSubject;
  const PatternPath *rootPath;
  Location matchLocation;
  DenseMap<const PatternPath *, CValue> pathValues;

  CValue getPathValue(IREmitter &emitter, const PatternPath *path,
                      const ExprNode *expr);

  LogicalResult emitCommands(OpBuilder &builder,
                             ArrayRef<const PatternCommand *> commands,
                             SmallVectorImpl<PatternBoundName> &bindings);
  LogicalResult emitOr(OpBuilder &builder, const PatternCommand &cmd,
                       SmallVectorImpl<PatternBoundName> &bindings);

  /// Given an Equal/EnumTag command, emit the subject and (if an enum) extract
  /// the discriminant.
  CValue emitTestableValue(IREmitter &emitter, const PatternCommand *command);

  /// Compare `value` against the value or enum tag in `command` and return a
  /// scalar bool (PValue in param context).
  RValue emitTestForValue(IREmitter &emitter, const PatternCommand *command,
                          CValue value);

  /// Test the minimum/exact length requirement of a list pattern.
  RValue emitListLengthTest(IREmitter &emitter, const PatternCommand *command);

  /// AND together Equal/EnumTag tests in `commands` into one compile-time
  /// bool attribute. Or alternatives are OR'd with `getLogicalOr` (same as
  /// comptime `or`). Bind commands append to `bindings` (like `emitCommands`)
  /// without contributing to the condition.
  FailureOr<TypedAttr>
  emitComptimeCondition(ArrayRef<const PatternCommand *> commands,
                        SmallVectorImpl<PatternBoundName> &bindings);
};

/// Bump storage, path uniquing, and binding-mode state for one `match`.
class PatternMatchBuilder {
public:
  static constexpr size_t kMaxSuggestedCases = 8;
  PatternMatchBuilder(ASTDecl &declScope, ExprContext paramContext);

  BumpAllocator &allocator;
  ASTDecl &declScope;
  ExprContext paramContext;
  SharedState &shared;

  IREmitter getParamEmitter();

  PatternDeclKind getPatternKind() const { return patternKind; }
  void setPatternKind(PatternDeclKind kind) { patternKind = kind; }
  PatternDeclKind &patternKindRef() { return patternKind; }

  MojoInflightDiag emitError(SMLoc loc, const Twine &message = {});
  MojoInflightDiag emitWarning(SMLoc loc, const Twine &message = {});

  const PatternPath *getRootPath(ASTType type) {
    return getOrCreatePath(PatternPath::Root, nullptr, type, 0, 0, {});
  }
  const PatternPath *getTupleElement(const PatternPath *parent, size_t index,
                                     ASTType elemType) {
    return getOrCreatePath(PatternPath::TupleElement, parent, elemType, index,
                           0, {});
  }
  const PatternPath *getStructField(const PatternPath *parent,
                                    StringAttr fieldName, ASTType fieldType) {
    return getOrCreatePath(PatternPath::StructField, parent, fieldType, 0, 0,
                           fieldName);
  }
  const PatternPath *getEnumPayload(const PatternPath *parent, size_t caseIndex,
                                    ASTType payloadType) {
    return getOrCreatePath(PatternPath::EnumPayload, parent, payloadType,
                           caseIndex, 0, {});
  }
  const PatternPath *getListElement(const PatternPath *parent, size_t index) {
    return getOrCreatePath(PatternPath::ListElement, parent, {}, index, 0, {});
  }
  const PatternPath *getListElementFromEnd(const PatternPath *parent,
                                           size_t offset) {
    return getOrCreatePath(PatternPath::ListElementFromEnd, parent, {}, offset,
                           0, {});
  }
  const PatternPath *getListSlice(const PatternPath *parent, size_t start,
                                  size_t endOffset) {
    return getOrCreatePath(PatternPath::ListSlice, parent, {}, start, endOffset,
                           {});
  }

  PatternCommand *createEqual(const PatternPath *path, const ExprNode *expr) {
    return allocator.intern(PatternCommand::getEqual(path, expr));
  }
  PatternCommand *createEnumTag(const PatternPath *path, const ExprNode *expr,
                                size_t caseIndex) {
    return allocator.intern(PatternCommand::getEnumTag(path, expr, caseIndex));
  }
  PatternCommand *createListLength(const PatternPath *path,
                                   const ExprNode *expr, size_t minimumLength,
                                   bool exactLength) {
    return allocator.intern(
        PatternCommand::getListLength(path, expr, minimumLength, exactLength));
  }
  PatternCommand *createOr(const PatternPath *path, const ExprNode *expr,
                           ArrayRef<PatternCommandList> alternatives) {
    return allocator.intern(PatternCommand::getOr(
        path, expr, allocator.internElements(alternatives)));
  }
  PatternCommand *createBind(const PatternPath *path, const ExprNode *expr,
                             StringRef name, PatternDeclKind declKind) {
    return allocator.intern(
        PatternCommand::getBind(path, expr, name, declKind));
  }

  /// Bump-allocate `commands` and return an immutable command list.
  PatternCommandList internCommandList(ArrayRef<const PatternCommand *> cmds) {
    return {allocator.internElements(cmds)};
  }

  /// Drop a handled leading test and move irrefutable bindings behind the
  /// remaining tests so recursive emission can cluster those tests.
  PatternCommandList internResidualCommandList(PatternCommandList commandList) {
    PatternCommandList remainingCommands = commandList.drop_front();
    SmallVector<const PatternCommand *, 8> reorderedCommands;
    reorderedCommands.reserve(remainingCommands.size());
    for (const PatternCommand *command : remainingCommands)
      if (command->kind != PatternCommand::Bind)
        reorderedCommands.push_back(command);
    for (const PatternCommand *command : remainingCommands)
      if (command->kind == PatternCommand::Bind)
        reorderedCommands.push_back(command);
    return internCommandList(reorderedCommands);
  }

  /// Diagnose non-exhaustive and unreachable `match` cases; set
  /// `MatchCaseEntry::isUnreachable` / `isConcluding` for covered and
  /// coverage-completing arms. When requested, returns up to
  /// kMaxSuggestedCases arms for a finite subject. Otherwise returns an empty
  /// string.
  std::string checkCaseExhaustivityAndUnreachability(
      MutableArrayRef<MatchCaseEntry> caseEntries, const PatternPath *rootPath,
      SMLoc matchLoc, bool offerMissingCases);

  /// Lay out one `case` arm with a `pass` body per pattern, each arm starting
  /// on a new line at `caseIndent` with its body at `bodyIndent`.
  static std::string formatCaseArms(ArrayRef<std::string> patterns,
                                    size_t caseIndent, size_t bodyIndent);

  /// Spell up to kMaxSuggestedCases patterns from the subject's finite domain,
  /// in the order the exhaustiveness check enumerates them: each case of an
  /// enum (`.red`,
  /// or the bare literal `True` for a `Bool`), or each combination of cases
  /// for a tuple or struct of such types (`False, .red`, or
  /// `Pair(a=False, b=True)`). Leaves `out` empty for any other subject.
  void spellSubjectPatterns(const PatternPath *rootPath, SMLoc loc,
                            SmallVectorImpl<std::string> &out);

  /// Recompute coverage for a residual case set produced during recursive
  /// emission, marking the arm that closes its local search space.
  void markResidualConcludingCase(MutableArrayRef<MatchCaseEntry> caseEntries,
                                  const PatternPath *rootPath, SMLoc matchLoc);

private:
  struct PathKey {
    const PatternPath *parent;
    PatternPath::Kind kind;
    size_t index;
    size_t endOffset;
    Attribute fieldName;
    bool operator==(const PathKey &o) const {
      return parent == o.parent && kind == o.kind && index == o.index &&
             endOffset == o.endOffset && fieldName == o.fieldName;
    }
  };
  struct PathKeyInfo {
    static PathKey getEmptyKey() {
      return {reinterpret_cast<const PatternPath *>(~uintptr_t(0)),
              PatternPath::Root,
              0,
              0,
              {}};
    }
    static PathKey getTombstoneKey() {
      return {reinterpret_cast<const PatternPath *>(~uintptr_t(1)),
              PatternPath::Root,
              0,
              0,
              {}};
    }
    static unsigned getHashValue(const PathKey &key) {
      return llvm::hash_combine(key.parent, unsigned(key.kind), key.index,
                                key.endOffset, key.fieldName);
    }
    static bool isEqual(const PathKey &a, const PathKey &b) { return a == b; }
  };

  const PatternPath *getOrCreatePath(PatternPath::Kind kind,
                                     const PatternPath *parent, ASTType type,
                                     size_t index, size_t endOffset,
                                     StringAttr fieldName) {
    PathKey key{parent, kind, index, endOffset, fieldName};
    const PatternPath *&slot = pathUniques[key];
    if (slot)
      return slot;
    auto *path = allocator.create<PatternPath>();
    path->kind = kind;
    path->parent = parent;
    path->type = type;
    path->index = index;
    if (kind == PatternPath::ListSlice)
      path->endOffset = endOffset;
    else if (kind == PatternPath::StructField)
      path->fieldName = fieldName;
    return slot = path;
  }

  DenseMap<PathKey, const PatternPath *, PathKeyInfo> pathUniques;
  PatternDeclKind patternKind = PatternDeclKind::kBind;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_EXPRPATTERNS_H
