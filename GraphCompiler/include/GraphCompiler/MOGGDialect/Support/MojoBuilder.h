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
#include <string>

namespace M::MOGG {

class MojoBuilder;

class MojoValue;

/// The text of a Mojo value and every operation on it. `Storage` holds the
/// text: `std::string` for a value built at run time, `llvm::StringRef` for one
/// known at compile time. The bodies follow `MojoValue`, which they return.
template <typename Storage>
class MojoValueOps {
public:
  MojoValueOps() = default;
  constexpr MojoValueOps(Storage text) : text(std::move(text)) {}

  static MojoValue None();
  MojoValue rebind(const MojoValue &type) const;
  MojoValue get(const MojoValue &member) const;
  MojoValue operator[](const MojoValue &index) const;
  MojoValue operator[](ArrayRef<MojoValue> values) const;
  /// Positional `ptr[idx]` is deprecated.
  MojoValue pointerOffsetGet(size_t index) const;
  /// `f(operands)`; comptime parameters go in `[]`, as in Mojo:
  /// `f[parameters](operands)`.
  MojoValue operator()(ArrayRef<MojoValue> operands = {}) const;
  MojoValue equal(const MojoValue &value) const;
  MojoValue quote() const;
  /// Escapes \\, ", \n, and \r for a single-line Mojo string literal.
  MojoValue escape() const;
  MojoValue operator+(const MojoValue &other) const;
  MojoValue operator*(const MojoValue &other) const;
  MojoValue operator%(const MojoValue &other) const;
  MojoValue operator<<(const MojoValue &other) const;
  MojoValue operator>>(const MojoValue &other) const;
  MojoValue bitwise_logical_and(const MojoValue &other) const;
  MojoValue bitwise_logical_or(const MojoValue &other) const;
  MojoValue bitwise_logical_xor(const MojoValue &other) const;
  MojoValue div(const MojoValue &other) const;
  MojoValue idiv(const MojoValue &other) const;
  MojoValue sdiv(const MojoValue &other) const;
  MojoValue eq(const MojoValue &other) const;
  MojoValue neq(const MojoValue &other) const;
  MojoValue ge(const MojoValue &other) const;
  MojoValue gt(const MojoValue &other) const;
  MojoValue le(const MojoValue &other) const;
  MojoValue lt(const MojoValue &other) const;
  /// Postfix `^`, transferring ownership of move-only types.
  MojoValue transfer() const;

  Storage text;
};

/// Mojo source text for a variable (lvalue) or value (rvalue), with helpers for
/// composing expressions.
class MojoValue : public MojoValueOps<std::string> {
public:
  MojoValue() = default;
  MojoValue(std::string &&str) : MojoValueOps(std::move(str)) {}
  MojoValue(StringLiteral str) : MojoValueOps(str.str()) {}
  MojoValue(StringRef str) : MojoValueOps(str.str()) {}
  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  MojoValue(T integer) : MojoValueOps(std::to_string(integer)) {}
  MojoValue(const llvm::APSInt &integer) {
    SmallVector<char> chars;
    integer.toString(chars);
    text.assign(chars.begin(), chars.end());
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
  MojoValue(bool val) : MojoValueOps(std::string(val ? "True" : "False")) {}
  MojoValue(const char *str) : MojoValueOps(std::string(str)) {}
  ~MojoValue() = default;
};

/// A Mojo name known at compile time: a type, a function, a parameter, a field,
/// or a fragment of source. It has the same operations as `MojoValue` over a
/// `StringRef`, so it stays a constant, and converts when a value is needed.
class MojoStaticValue : public MojoValueOps<llvm::StringRef> {
public:
  constexpr MojoStaticValue(llvm::StringLiteral text) : MojoValueOps(text) {}

  /// A string literal constant, e.g. `constexpr MojoStaticValue kInt = "Int"`.
  template <size_t N>
  constexpr MojoStaticValue(const char (&text)[N])
      : MojoValueOps(llvm::StringRef(text, N - 1)) {}

  operator MojoValue() const { return MojoValue(text); }
  constexpr operator llvm::StringRef() const { return text; }

  /// The text as an owned string, for callers that build one (`std::string`
  /// has no conversion from `StringRef`).
  std::string str() const { return text.str(); }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const MojoValue &value) {
  return os << value.text;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const MojoStaticValue &value) {
  return os << value.text;
}

// The operations, shared by `MojoValue` and `MojoStaticValue`. Defined here
// because they return `MojoValue`, which is complete now.
template <typename Storage>
MojoValue MojoValueOps<Storage>::None() {
  return MojoValue("None");
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::rebind(const MojoValue &type) const {
  return llvm::formatv("rebind[{}]({})", type.text, text).str();
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::get(const MojoValue &member) const {
  return llvm::formatv("{}.{}", text, member.text).str();
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::operator[](const MojoValue &index) const {
  return llvm::formatv("{}[{}]", text, index.text).str();
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::operator[](ArrayRef<MojoValue> values) const {
  std::string str;
  llvm::raw_string_ostream stream(str);
  stream << text;
  if (!values.empty()) {
    stream << "[";
    llvm::interleaveComma(values, stream);
    stream << "]";
  }
  return str;
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::pointerOffsetGet(size_t index) const {
  return llvm::formatv("{}[unsafe_offset={}]", text, index).str();
}

template <typename Storage>
MojoValue
MojoValueOps<Storage>::operator()(ArrayRef<MojoValue> operands) const {
  std::string str;
  llvm::raw_string_ostream stream(str);
  stream << text << "(";
  llvm::interleaveComma(operands, stream);
  stream << ")";
  return str;
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::equal(const MojoValue &value) const {
  return llvm::formatv("{} = {}", text, value.text).str();
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::quote() const {
  std::string res;
  llvm::raw_string_ostream os(res);
  os << '"';
  KGEN::printAsMojoStringLiteral(text, os);
  os << '"';
  return MojoValue(res);
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::escape() const {
  std::string res;
  res += '"';
  for (char c : text) {
    switch (c) {
    case '\\':
      res += "\\\\";
      break;
    case '"':
      res += "\\\"";
      break;
    case '\n':
      res += "\\n";
      break;
    case '\r':
      res += "\\r";
      break;
    default:
      res += c;
      break;
    }
  }
  res += '"';
  return MojoValue(res);
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::operator+(const MojoValue &other) const {
  return llvm::formatv("{} + {}", text, other.text).str();
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::operator*(const MojoValue &other) const {
  return llvm::formatv("{} * {}", text, other.text).str();
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::operator%(const MojoValue &other) const {
  return llvm::formatv("{} % {}", text, other.text).str();
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::operator<<(const MojoValue &other) const {
  return llvm::formatv("{} << {}", text, other.text).str();
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::operator>>(const MojoValue &other) const {
  return llvm::formatv("{} >> {}", text, other.text).str();
}

template <typename Storage>
MojoValue
MojoValueOps<Storage>::bitwise_logical_and(const MojoValue &other) const {
  return llvm::formatv("{} & {}", text, other.text).str();
}

template <typename Storage>
MojoValue
MojoValueOps<Storage>::bitwise_logical_or(const MojoValue &other) const {
  return llvm::formatv("{} | {}", text, other.text).str();
}

template <typename Storage>
MojoValue
MojoValueOps<Storage>::bitwise_logical_xor(const MojoValue &other) const {
  return llvm::formatv("{} ^ {}", text, other.text).str();
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::div(const MojoValue &other) const {
  return llvm::formatv("{} / {}", text, other.text).str();
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::idiv(const MojoValue &other) const {
  return llvm::formatv("{} // {}", text, other.text).str();
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::sdiv(const MojoValue &other) const {
  return llvm::formatv("-((-{}) // {})", text, other.text).str();
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::eq(const MojoValue &other) const {
  return llvm::formatv("{} == {}", text, other.text).str();
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::neq(const MojoValue &other) const {
  return llvm::formatv("{} != {}", text, other.text).str();
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::ge(const MojoValue &other) const {
  return llvm::formatv("{} >= {}", text, other.text).str();
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::gt(const MojoValue &other) const {
  return llvm::formatv("{} > {}", text, other.text).str();
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::le(const MojoValue &other) const {
  return llvm::formatv("{} <= {}", text, other.text).str();
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::lt(const MojoValue &other) const {
  return llvm::formatv("{} < {}", text, other.text).str();
}

template <typename Storage>
MojoValue MojoValueOps<Storage>::transfer() const {
  return llvm::formatv("{}^", text).str();
}

// A prefix plus a unique index (e.g. `bar_45`), resolved by MojoBuilder.
class UniqueMojoIdentifier {
  std::string prefix;
  std::optional<uint32_t> idx;

public:
  UniqueMojoIdentifier(const std::string &prefix)
      : prefix(prefix), idx(std::nullopt) {}

  /// The name this identifier was declared with, before it was resolved to a
  /// unique one (`width` for `width_3`).
  llvm::StringRef declaredName() const { return prefix; }

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
  /// The unique name minted for `name` when this definition was emitted
  /// (`width` becomes `width_3`); empty until then.
  MojoValue resolvedName;
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

/// The parameters and arguments of an emitted function body, as declared.
struct FunctionBody {
  SmallVector<NamedMojoDef> parameters;
  SmallVector<NamedMojoDef> arguments;

  /// The parameter declared as `declaredName` (e.g. `kParameterWidth`).
  MojoValue parameter(llvm::StringRef declaredName) const {
    for (const NamedMojoDef &parameter : parameters)
      if (parameter.name.declaredName() == declaredName) {
        ASSERT_STREAM(!parameter.resolvedName.text.empty(),
                      << "parameter '" << declaredName << "' was not emitted");
        return parameter.resolvedName;
      }
    ASSERT_STREAM(false, << "no parameter named '" << declaredName << "'");
    return {};
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
