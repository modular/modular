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
//
// Mojo text emission helpers.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_SUPPORT_MOJOEMITTER_H
#define GENERICML_GRAPHCOMPILER_SUPPORT_MOJOEMITTER_H

#include "Mojo/KGENDialect/KGENUtils.h"
#include "Support/AssertStream.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FormatVariadic.h"

#include <stack>

namespace M::MOGG {

class MojoBuilder;

/// Mojo source text for a variable (lvalue) or value (rvalue), with helpers
/// for composing expressions.
class MojoValue {
public:
  MojoValue() = default;
  MojoValue(std::string &&str) : text(std::move(str)) {}
  MojoValue(StringLiteral str) : text(str) {}
  MojoValue(StringRef str) : text(str) {}
  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  MojoValue(T integer) : text(std::to_string(integer)) {}
  MojoValue(const llvm::APSInt &integer) {
    SmallVector<char> chars;
    integer.toString(chars);
    this->text.assign(chars.begin(), chars.end());
  }
  MojoValue(const llvm::APFloat &value) {
    assert(value.isFinite() && "inf/nan have no Mojo float literal spelling");
    llvm::SmallString<32> str;
    value.toString(str);
    // `APFloat::toString` renders integral values as bare digits (e.g. "2").
    if (str.find_first_of(".eE") == llvm::StringRef::npos)
      str += ".0";
    text = str.str();
  }
  MojoValue(bool val) : text(val ? "True" : "False") {}
  MojoValue(const char *str) : text(str) {}
  ~MojoValue() = default;

  static MojoValue None() { return MojoValue("None"); }
  MojoValue rebind(const MojoValue &type) const {
    return llvm::formatv("rebind[{}]({})", type.text, text).str();
  }
  MojoValue get(const MojoValue &member) const {
    return llvm::formatv("{}.{}", text, member.text).str();
  }
  MojoValue operator[](const MojoValue &idx) const {
    return llvm::formatv("{}[{}]", text, idx.text).str();
  }
  MojoValue operator[](ArrayRef<MojoValue> values) const;
  /// Positional `ptr[idx]` is deprecated.
  MojoValue pointerOffsetGet(size_t idx) const {
    return llvm::formatv("{}[unsafe_offset={}]", text, idx).str();
  }
  MojoValue operator()(ArrayRef<MojoValue> parameters = {},
                       ArrayRef<MojoValue> operands = {}) const;
  MojoValue equal(const MojoValue &value) const {
    return llvm::formatv("{} = {}", text, value.text).str();
  }
  MojoValue quote() const {
    std::string res;
    llvm::raw_string_ostream os(res);
    os << '"';
    KGEN::printAsMojoStringLiteral(text, os);
    os << '"';
    return MojoValue(res);
  }
  /// Escapes \\, ", \n, and \r for a single-line Mojo string literal.
  MojoValue escape() const;
  MojoValue operator+(const MojoValue &other) const {
    return llvm::formatv("{} + {}", text, other.text).str();
  }
  MojoValue operator*(const MojoValue &other) const {
    return llvm::formatv("{} * {}", text, other.text).str();
  }
  MojoValue operator%(const MojoValue &other) const {
    return llvm::formatv("{} % {}", text, other.text).str();
  }
  MojoValue operator<<(const MojoValue &other) const {
    return llvm::formatv("{} << {}", text, other.text).str();
  }
  MojoValue operator>>(const MojoValue &other) const {
    return llvm::formatv("{} >> {}", text, other.text).str();
  }
  MojoValue bitwise_logical_and(const MojoValue &other) const {
    return llvm::formatv("{} & {}", text, other.text).str();
  }
  MojoValue bitwise_logical_or(const MojoValue &other) const {
    return llvm::formatv("{} | {}", text, other.text).str();
  }
  MojoValue bitwise_logical_xor(const MojoValue &other) const {
    return llvm::formatv("{} ^ {}", text, other.text).str();
  }
  MojoValue div(const MojoValue &other) const {
    return llvm::formatv("{} / {}", text, other.text).str();
  }
  MojoValue idiv(const MojoValue &other) const {
    return llvm::formatv("{} // {}", text, other.text).str();
  }
  MojoValue sdiv(const MojoValue &other) const {
    return llvm::formatv("-((-{}) // {})", text, other.text).str();
  }
  MojoValue eq(const MojoValue &other) const {
    return llvm::formatv("{} == {}", text, other.text).str();
  }
  MojoValue neq(const MojoValue &other) const {
    return llvm::formatv("{} != {}", text, other.text).str();
  }
  MojoValue ge(const MojoValue &other) const {
    return llvm::formatv("{} >= {}", text, other.text).str();
  }
  MojoValue gt(const MojoValue &other) const {
    return llvm::formatv("{} > {}", text, other.text).str();
  }
  MojoValue le(const MojoValue &other) const {
    return llvm::formatv("{} <= {}", text, other.text).str();
  }
  MojoValue lt(const MojoValue &other) const {
    return llvm::formatv("{} < {}", text, other.text).str();
  }

  /// Postfix `^`, transferring ownership of move-only types.
  MojoValue transfer() const { return llvm::formatv("{}^", text).str(); }

  std::string text;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const MojoValue &value) {
  return os << value.text;
}

struct FunctionBody {
  SmallVector<MojoValue> parameters;
  SmallVector<MojoValue> arguments;
};

/// A BindLambdaOp output lowered to a `FusionStruct`. `bindMethodName` is the
/// ManagedTensorSlice helper (e.g. `_bind_to_fused_input`) that attaches it.
struct FusionStructInfo {
  MojoValue instance;
  StringLiteral bindMethodName;
  MojoValue typeName;
};

using EmittedLambdaLowering = FusionStructInfo;

// A prefix plus a unique index (e.g. `bar_45`), resolved by MojoBuilder.
class UniqueMojoIdentifier {
  std::string prefix;
  std::optional<uint32_t> idx;

public:
  UniqueMojoIdentifier(const std::string &prefix)
      : prefix(prefix), idx(std::nullopt) {}

private:
  UniqueMojoIdentifier(const std::string &prefix, std::optional<uint32_t> idx)
      : prefix(prefix), idx(idx) {}

  std::string str() const {
    ASSERT_STREAM(idx.has_value(), << "UniqueMojoIdentifier must be resolved");
    return llvm::formatv("{}{}", prefix, *idx).str();
  }

  friend class MojoBuilder;
};

struct NamedMojoDef {
  UniqueMojoIdentifier name;
  MojoValue type;
  std::optional<MojoValue> defaultValue;
  /// Emits the name without a counter suffix, and no annotation if `type` is
  /// empty, for signatures that must match a Mojo trait (e.g. `self`, `idx`).
  bool verbatim = false;
  /// Emitted before the name (e.g. `mut`); the body still refers to the bare
  /// name. Empty means the default convention.
  llvm::StringRef argConvention = "";

  NamedMojoDef(UniqueMojoIdentifier name, MojoValue type,
               std::optional<MojoValue> defaultValue = std::nullopt)
      : name(name), type(type), defaultValue(defaultValue) {}

  NamedMojoDef(const std::string &name, MojoValue type,
               std::optional<MojoValue> defaultValue = std::nullopt)
      : NamedMojoDef(UniqueMojoIdentifier(name), type, defaultValue) {}

  /// An empty `type` suppresses the annotation, as for `self`.
  static NamedMojoDef
  asVerbatim(llvm::StringRef name, MojoValue type = MojoValue(""),
             std::optional<MojoValue> defaultValue = std::nullopt) {
    NamedMojoDef def(name.str(), type, defaultValue);
    def.verbatim = true;
    return def;
  }
};

using MojoValueCache = std::stack<llvm::StringMap<MojoValue>>;

/// The mlir::OpBuilder equivalent for emitting Mojo text.
class MojoBuilder {
public:
  MojoBuilder() : ss(sourceStr), variableCounters({0}) {}
  ~MojoBuilder() {
    ASSERT_STREAM(comptimeCache.empty(),
                  << "Expected caches of MojoValue to be empty");
  }

  void def(const MojoValue &name, ArrayRef<NamedMojoDef> parameters,
           ArrayRef<NamedMojoDef> arguments,
           std::optional<MojoValue> returnType, bool exported, bool raises,
           bool isAlwaysInline,
           std::function<void(const FunctionBody &)> &&funcBody);

  /// `kNever` is for device-graph build closures: inlining their single-use
  /// bodies yields one enormous function whose Mem2Reg dominates compile time
  /// and serializes the per-function-parallel pipeline.
  enum class LambdaInline { kAutomatic, kAlways, kNever };

  /// Returns the lambda's name. `raises` and `captureConvention` are
  /// independent. A non-empty `captureConvention` (e.g. "var") emits a
  /// `{<captureConvention>}` clause, which Mojo requires for closures that
  /// capture by name; empty emits none.
  MojoValue lambda(StringLiteral lambdaName, ArrayRef<NamedMojoDef> parameters,
                   ArrayRef<NamedMojoDef> arguments,
                   std::optional<MojoValue> returnType, LambdaInline inlineKind,
                   bool isAsync, bool raises, llvm::StringRef captureConvention,
                   std::function<void(const FunctionBody &)> &&lambdaBody);

  [[nodiscard]] MojoBuilder &try_(std::function<void()> &&tryBlock);
  void except(std::function<void(const MojoValue &)> &&exceptBlock);

  MojoBuilder &if_(const MojoValue &cond, std::function<void()> &&ifBlock,
                   bool comptime = false);

  [[nodiscard]] MojoBuilder &elif_(const MojoValue &cond,
                                   std::function<void()> &&elifBlock);
  void else_(std::function<void()> &&elseBlock);

  MojoBuilder &with_(const MojoValue &contextExpr,
                     std::function<void()> &&body);

  /// Returns the new variable's name. With `cache`, reuses an existing variable
  /// holding the same value (like CSE); disable it for side effects.
  [[nodiscard]] MojoValue var(const MojoValue &value, bool cache = true);

  /// Returns the declared variable's name.
  [[nodiscard]] MojoValue varDecl(const MojoValue &type);

  /// Returns the comptime's name. Always cached, since comptime has no side
  /// effects.
  [[nodiscard]] MojoValue comptime(const MojoValue &value,
                                   const MojoValue *type = nullptr);

  [[nodiscard]] MojoValue listLiteral(ArrayRef<MojoValue> values);

  /// Unlike MojoValue::operator(), writes the call as a statement.
  void call(const MojoValue &funcName, ArrayRef<MojoValue> parameters = {},
            ArrayRef<MojoValue> operands = {});

  void verbatim(const MojoValue &text);

  /// Assigns without declaring, e.g. `foo[idx] = 42`.
  void assign(const MojoValue &dst, const MojoValue &value);

  void comment(const std::string &text);

  void newline() { ss << "\n"; }

  void pass();

  void returnsValue(const MojoValue &value);

  void raise(const MojoValue &value);

  /// Emits `_ = value`.
  void keepAlive(const MojoValue &value);

  /// Fields are (name, type) pairs. Example:
  ///   @fieldwise_init
  ///   struct MyStruct:
  ///       var field1: Type1
  ///       var field2: Type2
  void structDecl(const MojoValue &name,
                  ArrayRef<std::pair<MojoValue, MojoValue>> fields,
                  bool fieldwiseInit = true);

  /// Parameters and fields are (name, type) pairs. Example:
  ///   @fieldwise_init
  ///   struct MyStruct[param1: Type1, param2: Type2]:
  ///       var field1: FieldType1
  ///       var field2: FieldType2
  void structDecl(const MojoValue &name,
                  ArrayRef<std::pair<MojoValue, MojoValue>> parameters,
                  ArrayRef<std::pair<MojoValue, MojoValue>> fields,
                  bool fieldwiseInit = true);

  /// `body` emits methods. Always emits @fieldwise_init. Example:
  ///   @fieldwise_init
  ///   struct MyStruct(MyTrait):
  ///       var field1: Type1
  ///       <body callback output>
  void structDecl(const MojoValue &name, ArrayRef<MojoValue> traits,
                  ArrayRef<std::pair<MojoValue, MojoValue>> fields,
                  std::function<void()> body);

  /// Always emits @fieldwise_init. Example:
  ///   @fieldwise_init
  ///   struct MyStruct[param1: Type1](MyTrait):
  ///       var field1: FieldType1
  ///       <body callback output>
  void structDecl(const MojoValue &name,
                  ArrayRef<std::pair<MojoValue, MojoValue>> parameters,
                  ArrayRef<MojoValue> traits,
                  ArrayRef<std::pair<MojoValue, MojoValue>> fields,
                  std::function<void()> body);

  /// Resolved names of a scoped `structDecl`'s parameters, in input order.
  struct StructBody {
    SmallVector<MojoValue> parameters;
  };

  /// Emits `struct Name[params](traits):` and opens its body scope without
  /// pre-declaring members; `body` declares them with `varDecl`, `def`, and
  /// `comptime`. Parameter names are resolved like `def()`'s and passed back
  /// via `StructBody` for use in type strings (e.g. `Self.dtype_3`). Always
  /// emits `@fieldwise_init`.
  void structDecl(const MojoValue &name, ArrayRef<NamedMojoDef> parameters,
                  ArrayRef<MojoValue> traits,
                  std::function<void(const StructBody &)> &&body);

  void import(const MojoValue &mod);

  void importAs(const MojoValue &from, const MojoValue &as);

  void importFrom(const MojoValue &from, const MojoValue &import);

  const std::string &str() const { return sourceStr; }

  void increaseIndentation() {
    indentation.insert(indentation.size(), "    ", 4);
  }

  void decreaseIndentation() { indentation.erase(indentation.size() - 4, 4); }

  UniqueMojoIdentifier getIdentifier(const std::string &prefix = "val") {
    auto prefixWithIndentationLevel =
        prefix + std::string(variableCounters.size(), '_');
    return UniqueMojoIdentifier(prefixWithIndentationLevel,
                                variableCounters.top()++);
  }

  UniqueMojoIdentifier resolveIdentifier(UniqueMojoIdentifier name) {
    if (name.idx.has_value())
      return name;
    else
      return getIdentifier(name.prefix);
  }

  MojoValue newVar(const std::string &name = "val") {
    return resolveIdentifier(name).str();
  }

  MojoValue newVar(UniqueMojoIdentifier name) {
    return resolveIdentifier(name).str();
  }

  /// Opens a function-like scope (def, lambda, struct): identifiers restart at
  /// zero, and so do the block scopes nested in it.
  void pushCounters() {
    variableCounters.push({0});
    blockSeeds.resize(std::max(blockSeeds.size(), variableCounters.size() + 2));
    std::fill(blockSeeds.begin() + variableCounters.size() + 1,
              blockSeeds.end(), 0);
  }

  void popCounters() { variableCounters.pop(); }

  /// Opens a statement block (if/elif/else, try/except, with). The block keeps
  /// its own depth prefix but continues the numbering where the previous block
  /// at this depth of the enclosing function left off: Mojo hoists a nested
  /// `def` to function scope, so sibling blocks that restarted at zero would
  /// redefine each other's closures.
  void pushBlockCounters() {
    size_t depth = variableCounters.size() + 1;
    if (blockSeeds.size() <= depth)
      blockSeeds.resize(depth + 1, 0);
    variableCounters.push(blockSeeds[depth]);
  }

  void popBlockCounters() {
    blockSeeds[variableCounters.size()] = variableCounters.top();
    variableCounters.pop();
  }

  void pushCache() {
    varCache.push({});
    comptimeCache.push({});
  }
  void popCache() {
    varCache.pop();
    comptimeCache.pop();
  }

private:
  void indent() { ss << indentation; }
  template <typename... Ts>
  void statement(Ts &&...args) {
    indent();
    (ss << ... << args);
    newline();
  }

  std::string funcDefBase(ArrayRef<NamedMojoDef> parameters,
                          ArrayRef<NamedMojoDef> operands,
                          std::optional<MojoValue> retType, FunctionBody &body);

  std::string funcDefBase(ArrayRef<NamedMojoDef> parameters,
                          ArrayRef<NamedMojoDef> operands,
                          std::optional<MojoValue> retType,
                          StringRef effectsClause, FunctionBody &body);

  /// Shared by the structDecl overloads; \p body, if set, runs after fields.
  void structDeclImpl(const MojoValue &name,
                      ArrayRef<std::pair<MojoValue, MojoValue>> parameters,
                      ArrayRef<MojoValue> traits,
                      ArrayRef<std::pair<MojoValue, MojoValue>> fields,
                      bool fieldwiseInit, std::function<void()> body);

  std::string sourceStr;
  llvm::raw_string_ostream ss;
  std::string indentation;

  std::stack<uint32_t> variableCounters;

  /// Per depth, the counter the next statement block at that depth resumes
  /// from; see `pushBlockCounters`.
  llvm::SmallVector<uint32_t> blockSeeds;

  // Partial CSE for Mojo text: some EmitsMojoInterface ops emit many variables
  // not represented in the IR. One entry per scope (if, function, lambda).
  MojoValueCache varCache;
  MojoValueCache comptimeCache;
};

} // namespace M::MOGG

#endif
