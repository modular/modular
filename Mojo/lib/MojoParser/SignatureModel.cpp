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

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/DeclSignaturePrinter.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

namespace M::KGEN {

namespace {

// These records exist only while rendering one signature. AST declarations
// remain the source of truth exposed to callers.

struct RenderedParameter {
  ASTDecl *decl;
  std::string type;
};

struct RenderedArgument {
  ASTDecl *decl = nullptr;
  std::string type;
};

} // namespace

static std::string
mergeConformsToConstraints(ArrayRef<ConstraintAttr> constraints,
                           ParameterEvaluator *evaluator, SharedState &shared,
                           SmallVectorImpl<RenderedParameter> &params);
static StringRef getConventionString(ArgConvention convention);
static void renderParameter(const RenderedParameter &parameter,
                            TypedAttr defaultValue, SharedState &shared,
                            raw_ostream &os);
static void renderArgument(const RenderedArgument &argument,
                           TypedAttr defaultValue, SharedState &shared,
                           raw_ostream &os);
static bool parseConformsToString(StringRef printed, StringRef &paramName,
                                  StringRef &traitStr);

//===----------------------------------------------------------------------===//
// Population
//===----------------------------------------------------------------------===//

static SmallVector<ASTDecl *> getUserParamDecls(ASTDecl &decl) {
  if (auto *function = dyn_cast<FnSignatureASTDecl>(&decl))
    return function->getUserParamDecls();
  if (auto *structDecl = dyn_cast<StructASTDecl>(&decl))
    return structDecl->getUserParamDecls();
  return cast<AliasASTDecl>(decl).getUserParamDecls();
}

static ParameterEvaluator
populateRenderedParameters(ASTDecl &decl, ArrayRef<Type> paramTypes,
                           PogListAttr paramListAttr,
                           SmallVectorImpl<RenderedParameter> &params,
                           std::optional<ASTType> selfType = std::nullopt) {
  auto &shared = decl.getShared();
  DenseMap<StringRef, ASTDecl *> userParams;
  for (ASTDecl *paramDecl : getUserParamDecls(decl))
    userParams.try_emplace(paramDecl->getPogMetadata().getName(), paramDecl);

  ParameterEvaluator evaluator;
  for (auto [idx, paramType] : llvm::enumerate(paramTypes)) {
    StringRef paramName = paramListAttr.getName(idx);
    VariadicKind variadicKind = paramListAttr.getVariadicKind(idx);
    Type reboundType = evaluator.getReboundType(paramType);

    // ALL parameters must be registered with the evaluator, even ones that
    // are hidden in the rendered signature - otherwise downstream type
    // inference loses the name binding.
    evaluator.appendIndexBinding(ParamDeclRefAttr::get(paramName, reboundType));

    auto paramDecl = userParams.find(paramName);
    if (paramDecl == userParams.end())
      continue;

    RenderedParameter info;
    info.decl = paramDecl->second;
    info.type = generateTypeString(shared, reboundType, variadicKind, selfType);
    params.push_back(std::move(info));
  }
  return evaluator;
}

static ParameterEvaluator
populateRenderedParameters(FnSignatureASTDecl &decl,
                           SmallVectorImpl<RenderedParameter> &params,
                           std::optional<ASTType> selfType) {
  FnTypeGeneratorType signature = decl.getSignature();
  if (!signature)
    return {};
  ParameterEvaluator evaluator = populateRenderedParameters(
      static_cast<ASTDecl &>(decl), signature.getInputParamTypes(),
      signature.getParamListAttrs(), params, selfType);
  return evaluator;
}

static std::optional<ASTType> getSelfType(FnSignatureASTDecl &decl) {
  auto *defDecl = dyn_cast<DefASTDecl>(&decl);
  if (!defDecl)
    return std::nullopt;
  if (auto *structDecl = dyn_cast<StructASTDecl>(defDecl->getParentDecl()))
    return structDecl->getSelfType();
  if (auto *traitDecl = dyn_cast<TraitASTDecl>(defDecl->getParentDecl()))
    return ASTType(TraitASTDecl::computeSelfType(traitDecl->getOp()));
  return std::nullopt;
}

static SmallVector<RenderedArgument, 2>
getRenderedArguments(FnSignatureASTDecl &decl, ParameterEvaluator &evaluator,
                     std::optional<ASTType> selfType) {
  FnTypeGeneratorType signature = decl.getSignature();
  if (!signature)
    return {};
  ArrayRef<Type> userArgTypes = signature.getArguments();
  if (auto *defDecl = dyn_cast<DefASTDecl>(&decl)) {
    FnOp fnOp = defDecl->getOp();
    userArgTypes = fnOp.getArgumentTypes();
  }

  auto &shared = decl.getShared();
  ArrayRef<PogMetadataAttr> argPogs = signature.getArgListAttrs().getPogs();
  ArrayRef<ArgConvention> argConventions = signature.getArgConventions();
  SmallVector<RenderedArgument, 2> args;

  for (auto [argIdx, userType, conventionX, pogAttr] :
       llvm::enumerate(userArgTypes, argConventions, argPogs)) {
    ArgConvention convention = conventionX;
    if (signature.isPosVarArg(argIdx))
      convention = signature.getVariadicConvention(argIdx);

    if (convention == ArgConvention::ByRefError)
      continue;
    if (convention == ArgConvention::ByRefResult) {
      // By-ref result types model as the trailing argument; skip implicit
      // (anonymous-named) ones.
      if (pogAttr.getName() == "__result__")
        continue;
    }

    VariadicKind variadicKind =
        signature.getArgListAttrs().getVariadicKind(argIdx);

    Type reboundUserType = evaluator.getReboundType(userType);
    std::string typeString = generateTypeString(
        shared, reboundUserType, variadicKind, selfType, convention);
    // Hide implicit auto-params of the form `arg.member` from the rendered
    // type - they're fully determined by the argument and noisy for humans.
    typeString = stripImplicitArgParams(typeString, pogAttr.getName());
    RenderedArgument info;
    info.type = std::move(typeString);
    args.push_back(std::move(info));
  }

  SmallVector<ASTDecl *> argDecls = decl.getUserArgDecls();
  assert(args.size() == argDecls.size() &&
         "user argument declarations must match the rendered signature");
  for (auto [argument, argDecl] : llvm::zip(args, argDecls))
    argument.decl = argDecl;
  return args;
}

static SmallVector<RenderedArgument, 2>
getRenderedArguments(FnSignatureASTDecl &decl) {
  DeclResolver::DiagnosticDeclContextChanger scope(&decl);
  std::optional<ASTType> selfType = getSelfType(decl);
  SmallVector<RenderedParameter, 2> parameters;
  ParameterEvaluator evaluator =
      populateRenderedParameters(decl, parameters, selfType);
  return getRenderedArguments(decl, evaluator, selfType);
}

static std::optional<RenderedArgument> getRenderedArgument(ASTDecl &decl) {
  if (decl.getArgNumber() < 0 || !decl.isUserArgument())
    return std::nullopt;

  auto *parentFunction =
      dyn_cast_or_null<FnSignatureASTDecl>(decl.getParentDecl());
  if (!parentFunction)
    return std::nullopt;
  SmallVector<ASTDecl *> argDecls = parentFunction->getUserArgDecls();
  auto argDecl = llvm::find(argDecls, &decl);
  if (argDecl == argDecls.end())
    return std::nullopt;

  SmallVector<RenderedArgument, 2> arguments =
      getRenderedArguments(*parentFunction);
  size_t argIndex = std::distance(argDecls.begin(), argDecl);
  if (arguments.size() <= argIndex)
    return std::nullopt;
  return arguments[argIndex];
}

static std::optional<RenderedParameter> getRenderedParameter(ASTDecl &decl) {
  if (decl.getParamNumber() < 0)
    return std::nullopt;

  ASTDecl *parent = decl.getParentDecl();
  SmallVector<RenderedParameter, 2> parameters;
  ParameterEvaluator evaluator;
  ArrayRef<ConstraintAttr> constraints;
  if (auto *function = dyn_cast_or_null<FnSignatureASTDecl>(parent)) {
    std::optional<ASTType> selfType = getSelfType(*function);
    evaluator = populateRenderedParameters(*function, parameters, selfType);
    constraints =
        function->getSignature().getParamListAttrs().getBodyConstraints();
  } else if (auto *structDecl = dyn_cast_or_null<StructASTDecl>(parent)) {
    TypeSignatureType signature = structDecl->getOp().getSignature();
    evaluator =
        populateRenderedParameters(*structDecl, signature.getInputParamTypes(),
                                   signature.getParamListAttrs(), parameters);
    constraints = signature.getParamListAttrs().getBodyConstraints();
  } else if (auto *aliasDecl = dyn_cast_or_null<AliasASTDecl>(parent)) {
    auto signature = dyn_cast<GeneratorType>(aliasDecl->getOp().getType());
    if (!signature)
      return std::nullopt;
    evaluator =
        populateRenderedParameters(*aliasDecl, signature.getInputParamTypes(),
                                   signature.getParamListAttrs(), parameters);
    constraints = signature.getParamListAttrs().getBodyConstraints();
  } else {
    return std::nullopt;
  }

  mergeConformsToConstraints(constraints, &evaluator, decl.getShared(),
                             parameters);
  auto parameter =
      llvm::find_if(parameters, [&](const RenderedParameter &info) {
        return info.decl == &decl;
      });
  if (parameter == parameters.end())
    return std::nullopt;
  return std::move(*parameter);
}

std::string getParameterTypeString(ASTDecl &decl) {
  std::optional<RenderedParameter> parameter = getRenderedParameter(decl);
  return parameter ? parameter->type : std::string();
}

std::string getArgumentTypeString(ASTDecl &decl) {
  std::optional<RenderedArgument> argument = getRenderedArgument(decl);
  return argument ? argument->type : std::string();
}

StringRef getArgumentConventionString(ASTDecl &decl) {
  if (decl.getArgNumber() < 0 || !decl.isUserArgument())
    return {};
  return getConventionString(decl.getArgConvention());
}

std::string AliasASTDecl::getToolingValue() {
  auto generatorType = dyn_cast<GeneratorType>(getOp().getType());
  ParameterEvaluator evaluator;
  if (generatorType) {
    for (auto [index, type] :
         llvm::enumerate(generatorType.getInputParamTypes())) {
      Type reboundType = evaluator.getReboundType(type);
      evaluator.appendIndexBinding(ParamDeclRefAttr::get(
          generatorType.getParamListAttrs().getName(index), reboundType));
    }
  }

  std::optional<Attribute> value = getOp().getValue();
  if (!value)
    return {};
  if (auto generator = dyn_cast<GeneratorAttr>(*value))
    value = evaluator.getReboundAttribute(generator.getBody());
  return generatePValueString(getShared(), *value);
}

static std::pair<ArrayRef<Type>, PogListAttr>
getParameterSignature(ASTDecl &owner) {
  if (auto *function = dyn_cast<FnSignatureASTDecl>(&owner)) {
    FnTypeGeneratorType signature = function->getSignature();
    return {signature.getInputParamTypes(), signature.getParamListAttrs()};
  }
  if (auto *structDecl = dyn_cast<StructASTDecl>(&owner)) {
    TypeSignatureType signature = structDecl->getOp().getSignature();
    return {signature.getInputParamTypes(), signature.getParamListAttrs()};
  }
  auto *aliasDecl = cast<AliasASTDecl>(&owner);
  auto signature = cast<GeneratorType>(aliasDecl->getOp().getType());
  return {signature.getInputParamTypes(), signature.getParamListAttrs()};
}

TypedAttr getReboundPogDefault(ASTDecl &decl) {
  std::optional<TypedAttr> defaultValue = decl.getPogDefault();
  if (!defaultValue)
    return {};

  ASTDecl *owner = decl.getParentDecl();
  if (!owner)
    return {};
  auto [paramTypes, paramList] = getParameterSignature(*owner);

  size_t parameterLimit = paramTypes.size();
  if (decl.getParamNumber() >= 0) {
    StringRef name = decl.getPogMetadata().getName();
    for (size_t index : llvm::seq(paramList.size())) {
      if (paramList.getName(index) == name) {
        parameterLimit = index;
        break;
      }
    }
  }

  ParameterEvaluator evaluator;
  for (size_t index : llvm::seq(parameterLimit)) {
    Type reboundType = evaluator.getReboundType(paramTypes[index]);
    evaluator.appendIndexBinding(
        ParamDeclRefAttr::get(paramList.getName(index), reboundType));
  }
  return cast<TypedAttr>(evaluator.getReboundAttribute(*defaultValue));
}

void printParameter(ASTDecl &decl, raw_ostream &os) {
  if (std::optional<RenderedParameter> parameter = getRenderedParameter(decl))
    renderParameter(*parameter, getReboundPogDefault(decl), decl.getShared(),
                    os);
}

void printArgument(ASTDecl &decl, raw_ostream &os) {
  if (std::optional<RenderedArgument> argument = getRenderedArgument(decl))
    renderArgument(*argument, getReboundPogDefault(decl), decl.getShared(), os);
}

//===----------------------------------------------------------------------===//
// Constraint merging
//===----------------------------------------------------------------------===//

/// Match a proposition that `deShortCircuitCond` rewrote from `a and b` into
/// `cond(a, b, a)`, or from `a or b` into `cond(a, a, b)`, and hand back the
/// operator with the *sugared* operands so their source spelling survives.
///
/// `ASTType::printParam` matches the same lowering structurally, by comparing
/// the condition against the repeated branch. That comparison holds for a
/// `where` clause written in source but not for a conformance condition, whose
/// branches reach here canonicalized and so are no longer equal to the
/// condition. The sugar's expanded form names the operator either way, so match
/// on that instead.
///
/// TODO(MOTO-1632): only the outermost operator is recovered. The recursion
/// relies on each operand carrying the sugar `deShortCircuitCond` attaches,
/// which the inner `and` of a chain like `a and b and c` does not, so it stays
/// in its lowered `b if a else a` spelling.
static bool matchDeShortCircuitedLogical(TypedAttr proposition, POC &opcode,
                                         TypedAttr &lhs, TypedAttr &rhs) {
  auto sugar = dyn_cast<SugarAttr>(proposition);
  if (!sugar)
    return false;
  auto expanded = dyn_cast<ParamOperatorAttr>(sugar.getExpanded());
  if (!expanded)
    return false;
  opcode = expanded.getOpcode();
  if (opcode != POC::And && opcode != POC::Or)
    return false;

  // Peel this level's sugar only, so the operands keep their own, then the
  // `Bool` field access wrapping the lowered `cond`.
  TypedAttr inner = SugarAttr::strip(sugar.getSugared());
  auto extract = dyn_cast_or_null<LIT::StructExtractAttr>(inner);
  if (extract)
    inner = extract.getStructValue();

  auto cond = dyn_cast_or_null<ParamOperatorAttr>(inner);
  if (!cond || cond.getOpcode() != POC::Cond || cond.getOperands().size() != 3)
    return false;

  // The `cond` operand is already a scalar bool; only the branches carry the
  // `Bool` wrapper, so only they get the field access re-applied.
  lhs = cond.getOperands()[0];
  rhs = cond.getOperands()[opcode == POC::And ? 1 : 2];
  if (extract)
    rhs =
        LIT::StructExtractAttr::get(rhs, extract.getField(), extract.getType());
  return true;
}

/// Print `proposition`, spelling a de-short-circuited `and`/`or` as written.
///
/// Recovering the operator here rather than rebuilding a `ParamOperatorAttr`
/// keeps the operands in source order: reconstructing the operator would run
/// them through the attribute's commutative canonicalization and swap them.
static void printProposition(raw_ostream &os, TypedAttr proposition,
                             SharedState &shared) {
  POC opcode;
  TypedAttr lhs, rhs;
  if (matchDeShortCircuitedLogical(proposition, opcode, lhs, rhs)) {
    printProposition(os, lhs, shared);
    os << (opcode == POC::And ? " and " : " or ");
    printProposition(os, rhs, shared);
    return;
  }
  ASTType::printParam(os, proposition, {&shared});
}

/// Split `printed` at each ` and ` that is not nested inside parentheses.
/// Returns false when there is no top-level `and`, leaving `conjuncts` empty.
static bool splitTopLevelConjuncts(StringRef printed,
                                   SmallVectorImpl<StringRef> &conjuncts) {
  StringRef separator = " and ";
  unsigned depth = 0;
  size_t start = 0;
  for (size_t i = 0, e = printed.size(); i < e; ++i) {
    if (printed[i] == '(') {
      ++depth;
    } else if (printed[i] == ')') {
      if (depth)
        --depth;
    } else if (depth == 0 && printed.substr(i).starts_with(separator)) {
      conjuncts.push_back(printed.slice(start, i));
      i += separator.size() - 1;
      start = i + 1;
    }
  }
  if (conjuncts.empty())
    return false;
  conjuncts.push_back(printed.substr(start));
  return true;
}

/// Rewrite a conjunction of `conforms_to` predicates over a single parameter as
/// one predicate over composed traits, so `conforms_to(T, A) and
/// conforms_to(T, B)` reads as `conforms_to(T, A & B)`. That is how the bound
/// would be written by hand, and it lets `mergeConformsToConstraints` fold the
/// result into the parameter's rendered bounds, which it already does for a
/// lone `conforms_to`.
///
/// Returns `printed` unchanged unless every top-level conjunct is a
/// `conforms_to` naming the same parameter.
static std::string mergeConformsToConjuncts(std::string printed) {
  SmallVector<StringRef> conjuncts;
  if (!splitTopLevelConjuncts(printed, conjuncts))
    return printed;

  StringRef param;
  SmallVector<StringRef> traits;
  for (StringRef conjunct : conjuncts) {
    StringRef conjunctParam, traitStr;
    if (!parseConformsToString(conjunct.trim(), conjunctParam, traitStr))
      return printed;
    if (traits.empty())
      param = conjunctParam;
    else if (param != conjunctParam)
      return printed;
    // A trait repeated across conjuncts composes to itself.
    if (!llvm::is_contained(traits, traitStr))
      traits.push_back(traitStr);
  }

  std::string merged;
  llvm::raw_string_ostream os(merged);
  os << "conforms_to(" << param << ", ";
  llvm::interleave(traits, os, " & ");
  os << ")";
  return merged;
}

std::string renderConstraintProposition(TypedAttr proposition,
                                        ParameterEvaluator *evaluator,
                                        SharedState &shared) {
  if (evaluator)
    proposition = evaluator->getReboundAttribute(proposition);

  std::string printed;
  llvm::raw_string_ostream os(printed);
  printProposition(os, proposition, shared);
  return mergeConformsToConjuncts(std::move(printed));
}

static std::string
mergeConformsToConstraints(ArrayRef<ConstraintAttr> constraints,
                           ParameterEvaluator *evaluator, SharedState &shared,
                           SmallVectorImpl<RenderedParameter> &params) {
  std::string remaining;
  if (constraints.empty())
    return remaining;
  llvm::raw_string_ostream os(remaining);

  for (const ConstraintAttr &c : constraints) {
    std::string printed =
        renderConstraintProposition(c.getProposition(), evaluator, shared);

    StringRef paramName, traitStr;
    if (parseConformsToString(printed, paramName, traitStr)) {
      RenderedParameter *match = nullptr;
      for (auto &p : params) {
        if (p.decl->getPogMetadata().getName() == paramName) {
          match = &p;
          break;
        }
      }
      if (match) {
        // Append `" & Trait"` for each trait, skipping the implicit `AnyType`
        // (which is always present and not user-meaningful).
        SmallVector<StringRef> parts;
        traitStr.split(parts, " & ");
        for (StringRef part : parts) {
          StringRef trimmed = part.trim();
          if (trimmed.empty() || trimmed == "AnyType")
            continue;
          match->type += " & ";
          match->type += trimmed.str();
        }
        continue;
      }
    }

    os << " where " << printed;
  }
  return remaining;
}

//===----------------------------------------------------------------------===//
// Rendering
//===----------------------------------------------------------------------===//

static StringRef getConventionString(ArgConvention convention) {
  switch (convention) {
  case ArgConvention::ByRefError:
    llvm_unreachable("error arguments are not rendered");
  case ArgConvention::ByRefResult:
    return "out";
  case ArgConvention::ImmReg:
  case ArgConvention::ImmMem:
    return "imm";
  case ArgConvention::Mut:
    return "mut";
  case ArgConvention::Ref:
  case ArgConvention::MutRef:
    return "ref";
  case ArgConvention::DeinitMem:
    return "deinit";
  case ArgConvention::OwnedMem:
  case ArgConvention::OwnedReg:
    return "var";
  }
  llvm_unreachable("unknown convention");
}

static void renderParameter(const RenderedParameter &p, TypedAttr defaultValue,
                            SharedState &shared, raw_ostream &os) {
  PogMetadataAttr metadata = p.decl->getPogMetadata();
  dumpIdentifierWithType(os, metadata.getName(), p.type,
                         p.decl->getPogVariadicKind());
  if (defaultValue)
    os << " = " << getDefaultValueString(defaultValue, shared);
}

static std::string getArgumentPrefix(ASTDecl &decl) {
  ArgConvention convention = decl.getArgConvention();
  if (convention != ArgConvention::Ref && convention != ArgConvention::MutRef)
    return {};

  auto *parent = cast<FnSignatureASTDecl>(decl.getParentDecl());
  FnTypeGeneratorType signature = parent->getSignature();
  Type type =
      signature.getArguments()[static_cast<size_t>(decl.getArgNumber())];
  return getRefPrefixAsString(decl.getShared(), cast<RefType>(type), signature,
                              /*isRefResult=*/false);
}

static bool isSelfArgument(ASTDecl &decl) {
  auto *parent = cast<FnSignatureASTDecl>(decl.getParentDecl());
  if (!getSelfType(*parent))
    return false;
  if (auto *function = dyn_cast<DefASTDecl>(parent);
      function && function->getOp().getSpecialFunctionInfo().hasSelfResult())
    return decl.getArgConvention() == ArgConvention::ByRefResult;

  // TODO: this is wrong for static methods, but matches the historical
  // doc-tooling behavior - fixing it would change the rendered Self type
  // elision for the first arg of any static method on a struct.
  return decl.getArgNumber() == 0;
}

static void renderArgument(const RenderedArgument &a, TypedAttr defaultValue,
                           SharedState &shared, raw_ostream &os) {
  ArgConvention argConvention = a.decl->getArgConvention();
  StringRef convention = getConventionString(argConvention);
  std::string prefix = getArgumentPrefix(*a.decl);
  if (argConvention != ArgConvention::ImmReg &&
      argConvention != ArgConvention::ImmMem) {
    os << convention;
    // Don't add a space after `ref` if a `[origin]` prefix follows - the
    // prefix supplies its own trailing space.
    if (prefix.empty())
      os << ' ';
  }
  os << prefix;
  bool isSelf = isSelfArgument(*a.decl);
  bool elideType = isSelf && a.type == "Self";
  // A compiler-synthesized implicit self (e.g. the `__call__` requirement of a
  // closure trait) carries no source name; render it as `self` rather than
  // leaving a blank argument slot.
  StringRef name = a.decl->getPogMetadata().getName();
  if (isSelf && name.empty())
    name = "self";
  dumpIdentifierWithType(os, name, a.type, a.decl->getPogVariadicKind(),
                         elideType);
  if (defaultValue)
    os << " = " << getDefaultValueString(defaultValue, shared);
}

namespace {

/// Walk a list of parameters or arguments, emitting passing-kind markers
/// (`/`, `*`, `?`) and per-element renderings. `extract` returns the
/// PassingKind for the given index; `render` writes the i-th element to `os`.
void printPogList(size_t count, llvm::function_ref<PassingKind(size_t)> extract,
                  llvm::function_ref<void(size_t)> render, bool isArgs,
                  bool suppressSlashAfterSelf, llvm::raw_string_ostream &os,
                  SmallVectorImpl<std::pair<unsigned, unsigned>> *offsets) {
  PassingKindPrinter passingKindPrinter(os, count, extract,
                                        suppressSlashAfterSelf, /*slash=*/'/',
                                        /*plus=*/"//");
  os << (isArgs ? "(" : "[");
  bool first = true;
  for (size_t idx = 0; idx < count; ++idx) {
    if (!first)
      os << ", ";
    first = false;
    passingKindPrinter.printOptionalStarSlash(idx);
    unsigned start = os.str().size();
    render(idx);
    if (offsets)
      offsets->push_back({start, static_cast<unsigned>(os.str().size())});
    passingKindPrinter.printOptionalTrailingSlash(idx);
  }
  os << (isArgs ? ")" : "]");
}

} // namespace

void printParameterList(
    ArrayRef<RenderedParameter> params, ParameterEvaluator &evaluator,
    SharedState &shared, llvm::raw_string_ostream &os,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *offsets) {
  printPogList(
      params.size(),
      [&](size_t i) {
        return params[i].decl->getPogMetadata().getPassingKind();
      },
      [&](size_t i) {
        TypedAttr defaultValue;
        if (auto defaultAttr = params[i].decl->getPogDefault())
          defaultValue =
              cast<TypedAttr>(evaluator.getReboundAttribute(*defaultAttr));
        renderParameter(params[i], defaultValue, shared, os);
      },
      /*isArgs=*/false, /*suppressSlashAfterSelf=*/false, os, offsets);
}

static PassingKind
getRenderedArgumentPassingKind(ArrayRef<RenderedArgument> args, size_t index) {
  const RenderedArgument &argument = args[index];
  if (argument.decl->getArgConvention() != ArgConvention::ByRefResult)
    return argument.decl->getPogMetadata().getPassingKind();

  // An initializer moves its trailing out result to the front. Preserve a
  // leading positional-only marker, but otherwise render the result normally.
  if (index == 0) {
    if (args.size() > 1 &&
        args[1].decl->getPogMetadata().getPassingKind() == PassingKind::PosOnly)
      return PassingKind::PosOnly;
    return PassingKind::PosOrKw;
  }
  return args[index - 1].decl->getPogMetadata().getPassingKind();
}

void printArgumentList(
    ArrayRef<RenderedArgument> args, ParameterEvaluator &evaluator,
    SharedState &shared, llvm::raw_string_ostream &os,
    bool suppressSlashAfterSelf,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *offsets) {
  printPogList(
      args.size(),
      [&](size_t i) { return getRenderedArgumentPassingKind(args, i); },
      [&](size_t i) {
        TypedAttr defaultValue;
        if (auto defaultAttr = args[i].decl->getPogDefault())
          defaultValue =
              cast<TypedAttr>(evaluator.getReboundAttribute(*defaultAttr));
        renderArgument(args[i], defaultValue, shared, os);
      },
      /*isArgs=*/true, suppressSlashAfterSelf, os, offsets);
}

//===----------------------------------------------------------------------===//
// Whole-function signature
//===----------------------------------------------------------------------===//

void printRenderedFunctionSignature(
    StringRef name, SmallVectorImpl<RenderedArgument> &args,
    ArrayRef<RenderedParameter> params, StringRef returnType,
    StringRef fnConstraints, bool isInit, bool isMethod,
    FnTypeGeneratorType signature, ParameterEvaluator &evaluator,
    SharedState &shared, llvm::raw_string_ostream &os,
    const SignatureOffsets &offsets) {
  // Initializer methods conventionally render with `self` (the out-arg) at
  // the front of the argument list rather than the back.
  bool hasOutArgument = !args.empty() && args.back().decl->getArgConvention() ==
                                             ArgConvention::ByRefResult;
  if (hasOutArgument && isInit && args.size() != 1)
    std::rotate(args.rbegin(), args.rbegin() + 1, args.rend());

  os << "def " << name;
  if (!params.empty())
    printParameterList(params, evaluator, shared, os, offsets.parameters);
  printArgumentList(args, evaluator, shared, os,
                    /*suppressSlashAfterSelf=*/isMethod, offsets.arguments);

  if (offsets.returnTypeStart)
    *offsets.returnTypeStart = os.str().size();

  // When an out-arg has been hoisted, the return type clause is redundant
  // (the out-arg *is* the result), so suppress it even if the caller passed
  // one along.
  if (!hasOutArgument && !returnType.empty())
    os << " -> " << returnType;
  if (!fnConstraints.empty())
    os << fnConstraints;
}

void printRenderedStructSignature(StringRef name,
                                  ArrayRef<RenderedParameter> params,
                                  StringRef bodyConstraints,
                                  ParameterEvaluator &evaluator,
                                  LIT::SharedState &shared,
                                  llvm::raw_string_ostream &os,
                                  const SignatureOffsets &offsets) {
  os << "struct " << name;

  if (!params.empty())
    printParameterList(params, evaluator, shared, os, offsets.parameters);

  if (!bodyConstraints.empty())
    os << bodyConstraints;
}

void printRenderedAliasSignature(StringRef name, StringRef type,
                                 ArrayRef<RenderedParameter> params,
                                 StringRef bodyConstraints,
                                 ParameterEvaluator &evaluator,
                                 LIT::SharedState &shared,
                                 llvm::raw_string_ostream &os,
                                 const SignatureOffsets &offsets) {
  os << name;
  if (!params.empty())
    printParameterList(params, evaluator, shared, os, offsets.parameters);
  if (!type.empty())
    os << ": " << type;
  if (!bodyConstraints.empty())
    os << bodyConstraints;
}

/// Try to parse a printed constraint string as "conforms_to(ParamName, Traits)"
/// where ParamName is a simple identifier (no dots). If successful, returns
/// true and fills in paramName and traitStr.
static bool parseConformsToString(StringRef printed, StringRef &paramName,
                                  StringRef &traitStr) {
  if (!printed.starts_with("conforms_to(") || !printed.ends_with(")"))
    return false;
  // Extract contents between "conforms_to(" and ")"
  StringRef inner = printed.drop_front(strlen("conforms_to(")).drop_back(1);
  // Split at the first ", " to get paramName and traits
  auto [lhs, rhs] = inner.split(", ");
  if (rhs.empty())
    return false;
  // If either side contains another comma, this is not a simple
  // `conforms_to(ParamName, Trait)` pair. Keep it as an explicit where clause
  // instead of folding an ambiguous value into T's rendered type bounds.
  if (lhs.contains(',') || rhs.contains(','))
    return false;
  // Only merge if the type is a simple identifier (no dots = not an associated
  // type path like T.Element).
  if (lhs.contains('.'))
    return false;
  paramName = lhs;
  traitStr = rhs;
  return true;
}

/// Generate a user-readable representation of the given pvalue.
std::string generatePValueString(SharedState &shared, PValue value) {
  std::string typeName;
  llvm::raw_string_ostream os(typeName);
  ASTType::printParam(os, value, /*ctx=*/{&shared});
  return os.str();
}

/// Unpack a origin into a printable name when it is uttered in a signature
/// position.
std::string getSignatureOrigin(SharedState &shared, TypedAttr origin,
                               FnTypeGeneratorType signature,
                               bool isRefResult) {
  // Strip out extra stuff.
  origin = OriginType::stripMutCastAndRebind(origin);

  // If this is a "ref [_]" argument, don't print the []'s at all.
  if (auto indexRef = sugarDynCast<ParamIndexRefAttr>(origin);
      indexRef && indexRef.getDepth() == 0 && signature &&
      // Ref results always print their origin, it can refer to args.
      !isRefResult) {
    // Assume implicit origins are the current argument.  This isn't correct
    // because one arg can refer to another arg's origin theoretically.
    if (signature.getParamListAttrs().getPassingKind(indexRef.getIndex()) ==
        PassingKind::Implicit)
      return "";
  }

  std::string result;
  llvm::raw_string_ostream os(result);
  ASTType::printRefOriginParam(os, origin, &shared);
  return result;
}

/// Unpack a "ref" argument or result type into a string that can be shown to
/// the user.
std::string getRefPrefixAsString(SharedState &shared, RefType refType,
                                 FnTypeGeneratorType signature,
                                 bool isRefResult) {
  std::string originString =
      getSignatureOrigin(shared, refType.getOrigin(), signature, isRefResult);

  // Include the address space if it is non-default.
  if (!refType.isDefaultAddrSpace()) {
    // It will often be two extract_elements from the inner guts of the actual
    // AddressSpace value. Remove them.
    TypedAttr addrSpace = refType.getAddressSpace();
    // The address space is often the result of one or two `struct_extract`s
    // wrapping the actual value - peel those off so the user sees the inner
    // value rather than the IR plumbing.
    if (auto extractAttr = sugarDynCast<LIT::StructExtractAttr>(addrSpace)) {
      addrSpace = extractAttr.getStructValue();
      if (auto extractAttr2 = sugarDynCast<LIT::StructExtractAttr>(addrSpace))
        addrSpace = extractAttr2.getStructValue();
    }
    if (!originString.empty())
      originString += ", ";
    originString += generatePValueString(shared, addrSpace);
  }

  if (originString.empty())
    return std::string();
  return "[" + originString + "] ";
}

/// Generate a user-readable representation of the given type and variadic kind,
/// with an optional value convention, and parent struct "Self" type.
std::string generateTypeString(SharedState &shared, ASTType type,
                               VariadicKind varKind,
                               std::optional<ASTType> selfType,
                               std::optional<ArgConvention> convention) {
  std::string typeName;
  llvm::raw_string_ostream os(typeName);

  if (varKind == VariadicKind::PosVarArg) {
    if (convention && !isa<ParamListType>(type)) {
      type = RefType::stripRefConvention(type, *convention);
      type = type.getVariadicListInfo().elementType;
      convention = ArgConvention::ImmReg;
    } else {
      type = type.getParameterListInfo().elementType;
    }
  } else if (varKind == VariadicKind::PackVarArg) {
    // VariadicPack needs special printing - its argument isn't a type.
    os << "*";
    if (convention)
      type = RefType::stripRefConvention(type, *convention);
    ASTType::printParam(os, type.getVariadicPackInfo().typeList,
                        /*ctx=*/{&shared});
    return os.str();
  }

  // Process the convention if present.
  if (convention && hasAddress(*convention)) {
    // In some cases variadics are passed directly (which is a hack, but okay).
    // The ABI in these cases is that we pass a variadic of refs. We leave these
    // as is, since eventually (with unpacking) this hack won't be needed.
    if (!isa<ParamListType>(type))
      type = type.getReferenceElementType();
  }

  // Get the value type in a kwargs dictionary.
  if (varKind == VariadicKind::KwVarArg)
    type = type.getKwargsDictValueType();

  ASTTypePrinterContext ctx{&shared, selfType.value_or(ASTType{})};
  os << type.getAsString(ctx);

  return os.str();
}

/// If the argument/parameter is variadic, we put the star (or two stars if
/// variadic keyword) before the identifier.
llvm::Twine prependVariadicIdentifiers(const llvm::Twine &identifier,
                                       VariadicKind varKind) {
  switch (varKind) {
  case VariadicKind::PosVarArg:
  case VariadicKind::PackVarArg:
    return "*" + identifier;
  case VariadicKind::KwVarArg:
    return "**" + identifier;
  default:
    return identifier;
  }
}

// Helper function that dumps an identifier along with an optional
// type. It also takes care of varargs that need to encode * in the name.
void dumpIdentifierWithType(raw_ostream &os, StringRef identifier,
                            StringRef type, VariadicKind varKind,
                            bool elideType) {
  os << prependVariadicIdentifiers(identifier, varKind);
  if (!type.empty() && !elideType)
    os << ": " << type;
}

/// Given a parameter expression corresponding to a default value, return a
/// rendered string for the value.
std::string getDefaultValueString(TypedAttr defaultValue, SharedState &shared) {
  std::string value;
  llvm::raw_string_ostream os(value);
  // Default values are always printed after a type annotation.
  ASTType::printParam(os, defaultValue, /*ctx=*/{&shared},
                      /*hasContextualType=*/true);
  return os.str();
}

/// True if `s` is a Mojo identifier (`[A-Za-z_][A-Za-z0-9_]*`).
static bool isMojoIdentifier(StringRef s) {
  if (s.empty() || (!llvm::isAlpha(s[0]) && s[0] != '_'))
    return false;
  for (char c : s.drop_front())
    if (!llvm::isAlnum(c) && c != '_')
      return false;
  return true;
}

/// True if `param` is a `argName(.member)+` chain (and nothing else).
static bool isMemberAccessOn(StringRef param, StringRef argName) {
  param = param.trim();
  if (!param.consume_front(argName) || !param.consume_front("."))
    return false;
  while (true) {
    auto [seg, rest] = param.split('.');
    if (!isMojoIdentifier(seg))
      return false;
    if (rest.empty())
      return true;
    param = rest;
  }
}

std::string stripImplicitArgParams(StringRef typeStr, StringRef argName) {
  if (argName.empty())
    return typeStr.str();

  std::string result;
  result.reserve(typeStr.size());

  for (size_t i = 0; i < typeStr.size();) {
    if (typeStr[i] != '[') {
      result += typeStr[i++];
      continue;
    }

    // Find the matching `]` and the top-level commas inside.
    size_t depth = 1;
    size_t j = i + 1;
    llvm::SmallVector<size_t> commas;
    for (; j < typeStr.size() && depth > 0; ++j) {
      char c = typeStr[j];
      if (c == '[')
        ++depth;
      else if (c == ']') {
        if (--depth == 0)
          break;
      } else if (c == ',' && depth == 1) {
        commas.push_back(j);
      }
    }
    assert(depth == 0 && "type printer emitted unbalanced brackets");
    if (depth != 0) {
      // Release-build fallback: emit the remainder verbatim.
      result += typeStr.substr(i);
      return result;
    }

    llvm::SmallVector<StringRef> params;
    size_t prev = i + 1;
    for (size_t comma : commas) {
      params.push_back(typeStr.slice(prev, comma));
      prev = comma + 1;
    }
    params.push_back(typeStr.slice(prev, j));

    llvm::SmallVector<std::string> kept;
    for (StringRef p : params) {
      if (isMemberAccessOn(p, argName))
        continue;
      kept.push_back(stripImplicitArgParams(p.trim(), argName));
    }

    if (!kept.empty()) {
      result += '[';
      llvm::interleave(
          kept, [&](const std::string &k) { result += k; },
          [&] { result += ", "; });
      result += ']';
    }

    i = j + 1;
  }
  return result;
}

void printFunctionSignature(DefASTDecl &decl, llvm::raw_string_ostream &os,
                            const SignatureOffsets &offsets) {
  DeclResolver::DiagnosticDeclContextChanger scope(&decl);
  auto &shared = decl.getShared();

  FnOp fnOp = decl.getOp();
  Operation *parentOp = fnOp->getParentOp();
  bool isStatic = fnOp.getIsStatic();
  bool isMethod =
      !isStatic && (isa<StructDeclOp>(parentOp) || isa<TraitDeclOp>(parentOp));
  bool isInit = fnOp.getSpecialFunctionInfo().isInitializer();
  FnTypeGeneratorType signature = fnOp.getFuncTypeGenerator();

  std::optional<ASTType> selfType;
  if (auto parentStruct = dyn_cast<StructDeclOp>(parentOp))
    selfType = ASTType(StructASTDecl::computeSelfType(parentStruct));
  else if (auto parentTrait = dyn_cast<TraitDeclOp>(parentOp))
    selfType = ASTType(TraitASTDecl::computeSelfType(parentTrait));

  SmallVector<RenderedParameter, 2> params;
  ParameterEvaluator evaluator =
      populateRenderedParameters(decl, params, selfType);
  std::string fnConstraints;
  if (auto cs = signature.getParamListAttrs().getBodyConstraints(); !cs.empty())
    fnConstraints = mergeConformsToConstraints(cs, &evaluator, shared, params);

  SmallVector<RenderedArgument, 2> args =
      getRenderedArguments(decl, evaluator, selfType);
  bool hasOutArgument = !args.empty() && args.back().decl->getArgConvention() ==
                                             ArgConvention::ByRefResult;
  std::string returnTypeStr;
  ASTType resultType = signature.getUserResultType();
  if (!hasOutArgument && resultType && !resultType.isNoneType()) {
    std::optional<ArgConvention> convention;
    if (signature.isRefResult()) {
      convention = ArgConvention::Ref;
      returnTypeStr =
          "ref" + getRefPrefixAsString(shared, cast<RefType>(resultType),
                                       signature, /*isRefResult=*/true);
    }
    Type reboundUserResultType =
        evaluator.getReboundType(fnOp.getUserResultType());
    returnTypeStr +=
        generateTypeString(shared, reboundUserResultType, VariadicKind::None,
                           selfType, convention);
  }

  StringRef name = fnOp.getSourceName().value_or(StringRef()).split('(').first;
  printRenderedFunctionSignature(name, args, params, returnTypeStr,
                                 fnConstraints, isInit, isMethod, signature,
                                 evaluator, shared, os, offsets);
}

std::string getFunctionSignature(
    DefASTDecl &decl,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *argumentOffsets,
    unsigned *returnOffset) {
  std::string signature;
  llvm::raw_string_ostream os(signature);
  SignatureOffsets offsets;
  offsets.parameters = parameterOffsets;
  offsets.arguments = argumentOffsets;
  offsets.returnTypeStart = returnOffset;
  printFunctionSignature(decl, os, offsets);
  return signature;
}

std::string getFunctionDeclarationSnippet(
    DefASTDecl &decl,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *argumentOffsets) {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);
  FnTypeGeneratorType functionType = decl.getSignature();
  if (functionType.isAsync())
    os << "__async ";
  unsigned returnOffset = 0;
  std::string signature = getFunctionSignature(decl, parameterOffsets,
                                               argumentOffsets, &returnOffset);
  size_t signatureStart = os.str().size();
  auto adjustOffsets = [&](auto *offsets) {
    for (auto &offset : *offsets) {
      offset.first += signatureStart;
      offset.second += signatureStart;
    }
  };
  if (parameterOffsets)
    adjustOffsets(parameterOffsets);
  if (argumentOffsets)
    adjustOffsets(argumentOffsets);
  os << StringRef(signature).take_front(returnOffset);
  if (functionType.isThrows())
    os << " raises";
  os << StringRef(signature).drop_front(returnOffset);
  return snippet;
}

void printStructSignature(StructASTDecl &decl, llvm::raw_string_ostream &os,
                          const SignatureOffsets &offsets) {
  DeclResolver::DiagnosticDeclContextChanger scope(&decl);
  auto &shared = decl.getShared();
  StructDeclOp structOp = decl.getOp();
  TypeSignatureType signature = structOp.getSignature();
  SmallVector<RenderedParameter, 2> params;
  ParameterEvaluator evaluator =
      populateRenderedParameters(decl, signature.getInputParamTypes(),
                                 signature.getParamListAttrs(), params);
  std::string constraints;
  if (auto cs = signature.getParamListAttrs().getBodyConstraints(); !cs.empty())
    constraints = mergeConformsToConstraints(cs, &evaluator, shared, params);
  printRenderedStructSignature(structOp.getName(), params, constraints,
                               evaluator, shared, os, offsets);
}

std::string getStructSignature(
    StructASTDecl &decl,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *parameterOffsets) {
  std::string signature;
  llvm::raw_string_ostream os(signature);
  SignatureOffsets offsets;
  offsets.parameters = parameterOffsets;
  printStructSignature(decl, os, offsets);
  return signature;
}

std::string getDeclarationSnippet(ASTDecl &decl) {
  std::string snippet;
  llvm::raw_string_ostream os(snippet);

  if (auto *function = dyn_cast<DefASTDecl>(&decl))
    return getFunctionDeclarationSnippet(*function);
  if (auto *alias = dyn_cast<AliasASTDecl>(&decl)) {
    os << "comptime ";
    printAliasSignature(*alias, os);
    if (std::string value = alias->getToolingValue(); !value.empty())
      os << " = " << value;
    return snippet;
  }
  if (auto *variable = dyn_cast<VarASTDecl>(&decl)) {
    if (variable->isArgument()) {
      if (decl.isUserArgument())
        printArgument(decl, os);
      return snippet;
    }
    ASTType type = decl.getToolingType().getReferenceElementType();
    dumpIdentifierWithType(
        os, decl.getName().value_or(StringRef()),
        type.getAsString(ASTTypePrinterContext(&decl.getShared())));
    return "var " + snippet;
  }
  if (auto *structDecl = dyn_cast<StructASTDecl>(&decl))
    return getStructSignature(*structDecl);
  if (auto *trait = dyn_cast<TraitASTDecl>(&decl)) {
    printTraitSignature(*trait, os);
    return snippet;
  }
  if (auto *field = dyn_cast<StructFieldASTDecl>(&decl)) {
    std::string type;
    llvm::raw_string_ostream typeOS(type);
    ASTType(field->getOp().getType())
        .print(typeOS,
               /*ctx=*/{&decl.getShared()});
    dumpIdentifierWithType(os, decl.getName().value_or(StringRef()), type);
    return "var " + snippet;
  }
  if (decl.isArgumentDecl()) {
    if (decl.isUserArgument())
      printArgument(decl, os);
    return snippet;
  }
  if (decl.isUserParameterDecl()) {
    printParameter(decl, os);
    return snippet;
  }
  return {};
}

void printAliasSignature(AliasASTDecl &decl, llvm::raw_string_ostream &os,
                         const SignatureOffsets &offsets) {
  DeclResolver::DiagnosticDeclContextChanger scope(&decl);
  auto &shared = decl.getShared();
  AliasDeclOp aliasOp = decl.getOp();
  auto name = demangleParameterName(aliasOp.getName(), /*forUser=*/true);
  auto generatorType = dyn_cast<GeneratorType>(aliasOp.getType());
  if (!generatorType) {
    os << name;
    return;
  }

  SmallVector<RenderedParameter, 2> params;
  ParameterEvaluator evaluator =
      populateRenderedParameters(decl, generatorType.getInputParamTypes(),
                                 generatorType.getParamListAttrs(), params);
  std::string constraints;
  if (auto cs = generatorType.getParamListAttrs().getBodyConstraints();
      !cs.empty())
    constraints = mergeConformsToConstraints(cs, &evaluator, shared, params);
  printRenderedAliasSignature(name, /*type=*/"", params, constraints, evaluator,
                              shared, os, offsets);
}

void printTraitSignature(TraitASTDecl &decl, llvm::raw_string_ostream &os,
                         const SignatureOffsets &) {
  DeclResolver::DiagnosticDeclContextChanger scope(&decl);
  auto name = demangleParameterName(decl.getOp().getName(), /*forUser=*/true);
  os << "trait " << name;
}

} // namespace M::KGEN
