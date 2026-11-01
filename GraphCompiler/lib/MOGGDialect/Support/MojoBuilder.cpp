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

#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"

namespace M::MOGG {

MojoValue MojoValue::operator[](ArrayRef<MojoValue> values) const {
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

MojoValue MojoValue::operator()(ArrayRef<MojoValue> parameters,
                                ArrayRef<MojoValue> operands) const {
  std::string str;
  llvm::raw_string_ostream stream(str);
  stream << text;
  if (!parameters.empty()) {
    stream << "[";
    llvm::interleaveComma(parameters, stream);
    stream << "]";
  }

  stream << "(";
  llvm::interleaveComma(operands, stream);
  stream << ")";
  return str;
}

MojoValue MojoValue::escape() const {
  std::string res = "\"";
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

MojoValue createTypedVar(const MojoValue &name, const MojoValue &type) {
  return llvm::formatv("{}: {}", name.text, type.text).str();
}

class MojoBuilderScopeGuard {
public:
  /// `freshCounters` marks a function-like scope (def, lambda, struct), whose
  /// identifiers restart at zero. Statement blocks pass false and continue the
  /// numbering of the previous sibling block; see `pushBlockCounters`.
  MojoBuilderScopeGuard(MojoBuilder *e, bool deferExceptCounters = false,
                        bool freshCounters = true)
      : emitter(e), posBeforeBody(0), freshCounters(freshCounters) {
    if (!deferExceptCounters) {
      emitter->increaseIndentation();
      emitter->pushCache();
      // Record position after indentation setup for empty body detection.
      posBeforeBody = emitter->str().size();
    }
    if (freshCounters)
      emitter->pushCounters();
    else
      emitter->pushBlockCounters();
  }
  MojoBuilderScopeGuard() = delete;
  MojoBuilderScopeGuard(const MojoBuilderScopeGuard &other) = delete;
  ~MojoBuilderScopeGuard() {
    // Mojo requires non-empty function bodies.
    if (emitter->str().size() == posBeforeBody)
      emitter->pass();
    emitter->decreaseIndentation();
    emitter->newline();
    emitter->popCache();
    if (freshCounters)
      emitter->popCounters();
    else
      emitter->popBlockCounters();
  }
  MojoBuilder *emitter;
  size_t posBeforeBody;
  bool freshCounters;

  void doDeferredActions() {
    emitter->increaseIndentation();
    emitter->pushCache();
    // Record position after indentation setup for empty body detection.
    posBeforeBody = emitter->str().size();
  }
};

std::string MojoBuilder::funcDefBase(ArrayRef<NamedMojoDef> parameters,
                                     ArrayRef<NamedMojoDef> operands,
                                     std::optional<MojoValue> retType,
                                     FunctionBody &body) {
  return funcDefBase(parameters, operands, retType, /*effectsClause=*/"", body);
}

std::string MojoBuilder::funcDefBase(ArrayRef<NamedMojoDef> parameters,
                                     ArrayRef<NamedMojoDef> operands,
                                     std::optional<MojoValue> retType,
                                     StringRef effectsClause,
                                     FunctionBody &body) {
  std::string str;
  llvm::raw_string_ostream stream(str);
  auto emitNamedDef = [&](const NamedMojoDef &def, StringRef typeSep,
                          SmallVectorImpl<MojoValue> &out) {
    MojoValue name =
        def.verbatim ? MojoValue(def.name.prefix) : newVar(def.name);
    if (!def.argConvention.empty())
      stream << def.argConvention << " ";
    stream << name;
    if (!(def.verbatim && def.type.text.empty()))
      stream << typeSep << def.type;
    if (def.defaultValue.has_value())
      stream << " = " << *def.defaultValue;
    out.push_back(name);
  };
  if (!parameters.empty()) {
    stream << "[";
    llvm::interleaveComma(parameters, stream, [&](const NamedMojoDef &param) {
      emitNamedDef(param, " : ", body.parameters);
    });
    stream << "]";
  }

  stream << "(";
  llvm::interleaveComma(operands, stream, [&](const NamedMojoDef &operand) {
    emitNamedDef(operand, ": ", body.arguments);
  });
  stream << ")";

  if (!effectsClause.empty())
    stream << " " << effectsClause;

  if (retType.has_value())
    stream << " -> " << *retType;

  stream << ":";
  return str;
}

void MojoBuilder::def(const MojoValue &name, ArrayRef<NamedMojoDef> parameters,
                      ArrayRef<NamedMojoDef> arguments,
                      std::optional<MojoValue> returnType, bool exported,
                      bool raises, bool isAlwaysInline,
                      std::function<void(const FunctionBody &)> &&funcBody) {
  newline();

  FunctionBody body;

  MojoBuilderScopeGuard guard(this, true);

  if (isAlwaysInline)
    statement(kDecoratorAlwaysInline);
  if (exported)
    statement("@export");

  // An exported function requires an explicit `abi` effect. Generated kernels
  // use the Mojo ABI, which is a no-op on the function type.
  std::string effects;
  if (exported)
    effects = "abi(\"Mojo\")";
  if (raises) {
    if (!effects.empty())
      effects += " ";
    effects += "raises";
  }
  statement("def ", name,
            funcDefBase(parameters, arguments, returnType, effects, body));

  // Deferred because the guard is created early so argument names use the new
  // counters.
  guard.doDeferredActions();
  funcBody(body);
}

MojoValue
MojoBuilder::lambda(StringLiteral lambdaName, ArrayRef<NamedMojoDef> parameters,
                    ArrayRef<NamedMojoDef> arguments,
                    std::optional<MojoValue> returnType,
                    LambdaInline inlineKind, bool isAsync, bool raises,
                    llvm::StringRef captureConvention,
                    std::function<void(const FunctionBody &)> &&lambdaBody) {
  MojoValue lambdaParam = newVar(lambdaName.str());
  FunctionBody body;

  if (inlineKind == LambdaInline::kAlways)
    statement(kDecoratorAlwaysInline);
  else if (inlineKind == LambdaInline::kNever)
    statement(kDecoratorNoInline);

  // `raises` and the capture clause are independent -- a capturing closure
  // still needs its blanket `{var}`-style clause regardless of `raises`.
  std::string effects;
  if (raises)
    effects = "raises";
  if (!captureConvention.empty()) {
    if (!effects.empty())
      effects += " ";
    effects += llvm::formatv("{{{0}}", captureConvention).str();
  }

  statement(isAsync ? "__async def " : "def ", lambdaParam,
            funcDefBase(parameters, arguments, returnType, effects, body));

  MojoBuilderScopeGuard guard(this);
  lambdaBody(body);
  return lambdaParam;
}

MojoBuilder &MojoBuilder::try_(std::function<void()> &&tryBlock) {
  statement("try:");
  MojoBuilderScopeGuard guard(this, /*deferExceptCounters=*/false,
                              /*freshCounters=*/false);
  tryBlock();

  return *this;
}

void MojoBuilder::except(std::function<void(const MojoValue &)> &&exceptBlock) {
  auto e = newVar("e");
  statement("except ", e, ":");
  MojoBuilderScopeGuard guard(this, /*deferExceptCounters=*/false,
                              /*freshCounters=*/false);
  exceptBlock(e);
}

MojoBuilder &MojoBuilder::if_(const MojoValue &cond,
                              std::function<void()> &&ifBlock, bool comptime) {
  if (comptime)
    statement("comptime if ", cond, ":");
  else
    statement("if ", cond, ":");

  MojoBuilderScopeGuard guard(this, /*deferExceptCounters=*/false,
                              /*freshCounters=*/false);
  ifBlock();
  return *this;
}

MojoBuilder &MojoBuilder::with_(const MojoValue &contextExpr,
                                std::function<void()> &&body) {
  statement("with ", contextExpr, ":");
  MojoBuilderScopeGuard guard(this, /*deferExceptCounters=*/false,
                              /*freshCounters=*/false);
  body();
  return *this;
}

MojoBuilder &MojoBuilder::elif_(const MojoValue &cond,
                                std::function<void()> &&elifBlock) {
  statement("elif ", cond, ":");
  MojoBuilderScopeGuard guard(this, /*deferExceptCounters=*/false,
                              /*freshCounters=*/false);
  elifBlock();
  return *this;
}

void MojoBuilder::else_(std::function<void()> &&elseBlock) {
  statement("else:");
  MojoBuilderScopeGuard guard(this, /*deferExceptCounters=*/false,
                              /*freshCounters=*/false);
  elseBlock();
}

MojoValue MojoBuilder::var(const MojoValue &value, bool cache) {
  std::pair<llvm::StringMapIterBase<M::MOGG::MojoValue, false>, bool> it;
  if (cache) {
    it = varCache.top().insert({value.text, ""});
    if (!it.second)
      return it.first->second;
  }

  auto name = newVar("var");
  statement("var ", name, " = ", value);

  if (cache)
    it.first->second = name;

  return name;
}
MojoValue MojoBuilder::varDecl(const MojoValue &type) {
  auto name = newVar("var");
  statement("var ", createTypedVar(name, type));

  return name;
}
MojoValue MojoBuilder::comptime(const MojoValue &value, const MojoValue *type) {
  auto it = comptimeCache.top().insert({value.text, ""});
  if (!it.second)
    return it.first->second;

  auto name = newVar("param");

  if (type)
    statement("comptime ", name, ": ", *type, " = ", value);
  else
    statement("comptime ", name, " = ", value);

  it.first->second = name;

  return name;
}
MojoValue MojoBuilder::listLiteral(ArrayRef<MojoValue> values) {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << "[";
  llvm::interleaveComma(values, os);
  os << "]";

  return MojoValue(std::move(str));
}
void MojoBuilder::call(const MojoValue &funcName,
                       ArrayRef<MojoValue> parameters,
                       ArrayRef<MojoValue> operands) {
  statement(funcName(parameters, operands));
}

void MojoBuilder::verbatim(const MojoValue &text) { statement(text); }

void MojoBuilder::assign(const MojoValue &dst, const MojoValue &value) {
  statement(dst, " = ", value);
}

void MojoBuilder::comment(const std::string &text) { statement("# ", text); }

void MojoBuilder::pass() { statement("pass"); }

void MojoBuilder::returnsValue(const MojoValue &value) {
  statement("return ", value);
}

void MojoBuilder::raise(const MojoValue &value) { statement("raise ", value); }

void MojoBuilder::keepAlive(const MojoValue &value) {
  statement("_ = ", value);
}

void MojoBuilder::import(const MojoValue &mod) {
  statement("import ", mod.text);
}

void MojoBuilder::importAs(const MojoValue &mod, const MojoValue &as) {
  statement("import ", mod.text, " as ", as.text);
}

void MojoBuilder::importFrom(const MojoValue &from, const MojoValue &import) {
  statement("from ", from.text, " import ", import.text);
}

void MojoBuilder::structDecl(const MojoValue &name,
                             ArrayRef<std::pair<MojoValue, MojoValue>> fields,
                             bool fieldwiseInit) {
  structDeclImpl(name, /*parameters=*/{}, /*traits=*/{}, fields, fieldwiseInit,
                 nullptr);
}

void MojoBuilder::structDecl(
    const MojoValue &name, ArrayRef<std::pair<MojoValue, MojoValue>> parameters,
    ArrayRef<std::pair<MojoValue, MojoValue>> fields, bool fieldwiseInit) {
  structDeclImpl(name, parameters, /*traits=*/{}, fields, fieldwiseInit,
                 nullptr);
}

void MojoBuilder::structDecl(const MojoValue &name, ArrayRef<MojoValue> traits,
                             ArrayRef<std::pair<MojoValue, MojoValue>> fields,
                             std::function<void()> body) {
  structDeclImpl(name, /*parameters=*/{}, traits, fields,
                 /*fieldwiseInit=*/true, std::move(body));
}

void MojoBuilder::structDecl(
    const MojoValue &name, ArrayRef<std::pair<MojoValue, MojoValue>> parameters,
    ArrayRef<MojoValue> traits,
    ArrayRef<std::pair<MojoValue, MojoValue>> fields,
    std::function<void()> body) {
  structDeclImpl(name, parameters, traits, fields,
                 /*fieldwiseInit=*/true, std::move(body));
}

void MojoBuilder::structDecl(const MojoValue &name,
                             ArrayRef<NamedMojoDef> parameters,
                             ArrayRef<MojoValue> traits,
                             std::function<void(const StructBody &)> &&body) {
  newline();

  StructBody structBody;

  MojoBuilderScopeGuard guard(this, /*deferExceptCounters=*/true);

  statement("@fieldwise_init");

  std::string headerStr;
  llvm::raw_string_ostream os(headerStr);
  os << "struct " << name.text;
  if (!parameters.empty()) {
    os << "[";
    llvm::interleaveComma(parameters, os, [&](const NamedMojoDef &param) {
      MojoValue paramName =
          param.verbatim ? MojoValue(param.name.prefix) : newVar(param.name);
      os << paramName.text << ": " << param.type.text;
      if (param.defaultValue.has_value())
        os << " = " << param.defaultValue->text;
      structBody.parameters.push_back(paramName);
    });
    os << "]";
  }
  if (!traits.empty()) {
    os << "(";
    llvm::interleaveComma(traits, os,
                          [&](const MojoValue &t) { os << t.text; });
    os << ")";
  }
  os << ":";
  statement(headerStr);

  guard.doDeferredActions();
  body(structBody);
}

void MojoBuilder::structDeclImpl(
    const MojoValue &name, ArrayRef<std::pair<MojoValue, MojoValue>> parameters,
    ArrayRef<MojoValue> traits,
    ArrayRef<std::pair<MojoValue, MojoValue>> fields, bool fieldwiseInit,
    std::function<void()> body) {
  newline();
  if (fieldwiseInit)
    statement("@fieldwise_init");

  std::string headerStr;
  llvm::raw_string_ostream os(headerStr);
  os << "struct " << name.text;
  if (!parameters.empty()) {
    os << "[";
    llvm::interleaveComma(parameters, os, [&](const auto &p) {
      os << p.first.text << ": " << p.second.text;
    });
    os << "]";
  }
  if (!traits.empty()) {
    os << "(";
    llvm::interleaveComma(traits, os, [&](const auto &t) { os << t.text; });
    os << ")";
  }
  os << ":";
  statement(headerStr);

  increaseIndentation();
  if (body)
    pushCache();

  for (const auto &[fieldName, fieldType] : fields) {
    // The Mojo parser rejects a struct field whose type exposes AnyOrigin
    // (e.g. Pointer[..., MutAnyOrigin]) unless it opts in with this decorator.
    if (llvm::StringRef(fieldType.text).contains("AnyOrigin"))
      statement("@__allow_legacy_any_origin_fields");
    statement("var ", fieldName, ": ", fieldType);
  }

  if (body) {
    newline();
    body();
    popCache();
  }

  decreaseIndentation();
  newline();
}

} // namespace M::MOGG
