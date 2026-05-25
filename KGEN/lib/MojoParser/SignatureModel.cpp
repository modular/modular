//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/SignatureModel.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/DeclSignaturePrinter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// Population
//===----------------------------------------------------------------------===//

ParameterEvaluator populateParameterInfos(
    SharedState &shared, ArrayRef<Type> paramTypes, PogListAttr paramListAttr,
    SmallVectorImpl<ParameterInfo> &params, std::optional<ASTType> selfType) {
  ParameterEvaluator evaluator;
  ArrayRef<PogMetadataAttr> pogs = paramListAttr.getPogs();
  for (auto [idx, paramType] : llvm::enumerate(paramTypes)) {
    TypedAttr defaultValue;
    if (auto defaultAttr = paramListAttr.getDefault(idx))
      defaultValue =
          cast<TypedAttr>(evaluator.getReboundAttribute(defaultAttr));
    PassingKind passingKind = paramListAttr.getPassingKind(idx);
    StringRef paramName = paramListAttr.getName(idx);
    VariadicKind variadicKind = paramListAttr.getVariadicKind(idx);
    Type reboundType = evaluator.getReboundType(paramType);

    // ALL parameters must be registered with the evaluator, even ones that
    // are hidden in the rendered signature - otherwise downstream type
    // inference loses the name binding.
    evaluator.appendIndexBinding(ParamDeclRefAttr::get(paramName, reboundType));

    if (shouldExcludeParameterFromDocs(passingKind, paramName))
      continue;

    params.push_back(ParameterInfo{
        paramName,
        generateTypeString(shared, reboundType, variadicKind, selfType),
        passingKind, variadicKind, defaultValue, /*constraints=*/{}});
    if (auto cs = pogs[idx].getConstraints(); !cs.empty()) {
      std::string remaining =
          mergeConformsToConstraints(cs, &evaluator, shared, params);
      if (!remaining.empty())
        params.back().constraints = std::move(remaining);
    }
  }
  return evaluator;
}

void populateArgumentInfos(SharedState &shared, FnTypeGeneratorType signature,
                           ArrayRef<Type> userArgTypes,
                           std::optional<ASTType> selfType,
                           ParameterEvaluator &evaluator,
                           llvm::function_ref<bool()> hasSelfResultFn,
                           SmallVectorImpl<ArgumentInfo> &args) {
  ArrayRef<PogMetadataAttr> argPogs = signature.getArgListAttrs().getPogs();
  ArrayRef<Type> sigTypes = signature.getArguments();
  ArrayRef<ArgConvention> argConventions = signature.getArgConventions();

  for (auto [argIdx, userType, sigType, conventionX, pogAttr] :
       llvm::enumerate(userArgTypes, sigTypes, argConventions, argPogs)) {
    ArgConvention convention = conventionX;
    if (signature.isPosVarArg(argIdx))
      convention = signature.getVariadicConvention(argIdx);

    TypedAttr defaultValue;
    if (auto defaultAttr = signature.getArgListAttrs().getDefault(argIdx))
      defaultValue = evaluator.getReboundAttribute(defaultAttr);

    std::string prefix;
    auto passingKind = pogAttr.getPassingKind();
    auto declConvention = ArgumentConvention::kRead;
    switch (convention) {
    case ArgConvention::ByRefError:
      continue;
    case ArgConvention::ByRefResult:
      // By-ref result types model as the trailing argument; skip implicit
      // (anonymous-named) ones, otherwise treat as `out` and inherit the
      // previous arg's passing kind so it doesn't print with a stray `?`.
      if (pogAttr.getName() == "__result__")
        continue;
      declConvention = ArgumentConvention::kOut;
      passingKind =
          args.empty() ? PassingKind::PosOrKw : args.back().passingKind;
      break;
    case ArgConvention::ReadReg:
    case ArgConvention::ReadMem:
      break;
    case ArgConvention::Mut:
      declConvention = ArgumentConvention::kInOut;
      break;
    case ArgConvention::Ref:
    case ArgConvention::MutRef:
      declConvention = ArgumentConvention::kRef;
      prefix = getRefPrefixAsString(shared, cast<RefType>(sigType), signature,
                                    /*isRefResult=*/false);
      break;
    case ArgConvention::DeinitMem:
      declConvention = ArgumentConvention::kDeinit;
      break;
    case ArgConvention::OwnedMem:
    case ArgConvention::OwnedReg:
      declConvention = ArgumentConvention::kOwned;
      break;
    }

    VariadicKind variadicKind =
        signature.getArgListAttrs().getVariadicKind(argIdx);

    bool isSelf = false;
    if (selfType) {
      if (hasSelfResultFn && hasSelfResultFn())
        isSelf = declConvention == ArgumentConvention::kOut;
      else
        // TODO: this is wrong for static methods, but matches the historical
        // doc-tooling behavior - fixing it would change the rendered Self
        // type elision for the first arg of any static method on a struct.
        isSelf = argIdx == 0;
    }

    Type reboundUserType = evaluator.getReboundType(userType);
    std::string typeString = generateTypeString(
        shared, reboundUserType, variadicKind, selfType, convention);
    // Hide implicit auto-params of the form `arg.member` from the rendered
    // type - they're fully determined by the argument and noisy for humans.
    typeString = stripImplicitArgParams(typeString, pogAttr.getName());
    args.push_back(ArgumentInfo{
        pogAttr.getName(), std::move(prefix), std::move(typeString),
        passingKind, variadicKind, defaultValue, declConvention, isSelf});
  }
}

//===----------------------------------------------------------------------===//
// Constraint merging
//===----------------------------------------------------------------------===//

std::string mergeConformsToConstraints(ArrayRef<ConstraintAttr> constraints,
                                       ParameterEvaluator *evaluator,
                                       SharedState &shared,
                                       SmallVectorImpl<ParameterInfo> &params) {
  std::string remaining;
  if (constraints.empty())
    return remaining;
  llvm::raw_string_ostream os(remaining);

  for (const ConstraintAttr &c : constraints) {
    TypedAttr proposition = c.getProposition();
    if (evaluator)
      proposition = evaluator->getReboundAttribute(proposition);

    std::string printed;
    {
      llvm::raw_string_ostream pos(printed);
      ASTType::printParam(pos, proposition, &shared);
    }

    StringRef paramName, traitStr;
    if (parseConformsToString(printed, paramName, traitStr)) {
      ParameterInfo *match = nullptr;
      for (auto &p : params) {
        if (p.name == paramName) {
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

StringRef getConventionString(ArgumentConvention conv) {
  switch (conv) {
  case ArgumentConvention::kRead:
    return "read";
  case ArgumentConvention::kDeinit:
    return "deinit";
  case ArgumentConvention::kInOut:
    return "mut";
  case ArgumentConvention::kOwned:
    return "var";
  case ArgumentConvention::kRef:
    return "ref";
  case ArgumentConvention::kOut:
    return "out";
  }
  llvm_unreachable("unknown convention");
}

void renderParameterInfo(const ParameterInfo &p, SharedState &shared,
                         raw_ostream &os) {
  dumpIdentifierWithType(os, p.name, p.type, p.variadicKind);
  if (!p.constraints.empty())
    os << p.constraints;
  if (p.defaultValue)
    os << " = " << getDefaultValueString(p.defaultValue, shared);
}

void renderArgumentInfo(const ArgumentInfo &a, SharedState &shared,
                        raw_ostream &os) {
  if (a.convention != ArgumentConvention::kRead) {
    os << getConventionString(a.convention);
    // Don't add a space after `ref` if a `[origin]` prefix follows - the
    // prefix supplies its own trailing space.
    if (a.prefix.empty())
      os << ' ';
  }
  os << a.prefix;
  bool elideType = a.isSelf && a.type == "Self";
  dumpIdentifierWithType(os, a.name, a.type, a.variadicKind, elideType);
  if (a.defaultValue)
    os << " = " << getDefaultValueString(a.defaultValue, shared);
}

namespace {

/// Walk a list of parameters or arguments, emitting passing-kind markers
/// (`/`, `*`, `?`) and per-element renderings. `extract` returns the
/// PassingKind for the given index; `render` writes the i-th element to `os`.
void printInfoList(size_t count,
                   llvm::function_ref<PassingKind(size_t)> extract,
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
    ArrayRef<ParameterInfo> params, SharedState &shared,
    llvm::raw_string_ostream &os,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *offsets) {
  printInfoList(
      params.size(), [&](size_t i) { return params[i].passingKind; },
      [&](size_t i) { renderParameterInfo(params[i], shared, os); },
      /*isArgs=*/false, /*suppressSlashAfterSelf=*/false, os, offsets);
}

void printArgumentList(
    ArrayRef<ArgumentInfo> args, SharedState &shared,
    llvm::raw_string_ostream &os, bool suppressSlashAfterSelf,
    SmallVectorImpl<std::pair<unsigned, unsigned>> *offsets) {
  printInfoList(
      args.size(), [&](size_t i) { return args[i].passingKind; },
      [&](size_t i) { renderArgumentInfo(args[i], shared, os); },
      /*isArgs=*/true, suppressSlashAfterSelf, os, offsets);
}

//===----------------------------------------------------------------------===//
// Whole-function signature
//===----------------------------------------------------------------------===//

void printFunctionSignatureFromInfos(
    StringRef name, SmallVectorImpl<ArgumentInfo> &args,
    ArrayRef<ParameterInfo> params, StringRef returnType,
    StringRef fnConstraints, bool isInit, bool isMethod, SharedState &shared,
    llvm::raw_string_ostream &os, const SignatureOffsets &offsets) {
  // Initializer methods conventionally render with `self` (the out-arg) at
  // the front of the argument list rather than the back.
  bool hasOutArgument =
      !args.empty() && args.back().convention == ArgumentConvention::kOut;
  if (hasOutArgument && isInit) {
    auto passingKind = PassingKind::PosOrKw;
    if (args.size() != 1) {
      std::rotate(args.rbegin(), args.rbegin() + 1, args.rend());
      if (args[1].passingKind == PassingKind::PosOnly)
        passingKind = PassingKind::PosOnly;
    }
    args[0].passingKind = passingKind;
  }

  os << name;
  if (!params.empty())
    printParameterList(params, shared, os, offsets.parameters);
  printArgumentList(args, shared, os, /*suppressSlashAfterSelf=*/isMethod,
                    offsets.arguments);

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

void printStructSignatureFromInfos(StringRef name,
                                   ArrayRef<ParameterInfo> params,
                                   StringRef bodyConstraints,
                                   LIT::SharedState &shared,
                                   llvm::raw_string_ostream &os,
                                   const SignatureOffsets &offsets) {
  os << name;

  if (!params.empty())
    printParameterList(params, shared, os, offsets.parameters);

  if (!bodyConstraints.empty())
    os << bodyConstraints;
}

void printAliasSignatureFromInfos(StringRef name, StringRef type,
                                  ArrayRef<ParameterInfo> params,
                                  LIT::SharedState &shared,
                                  llvm::raw_string_ostream &os,
                                  const SignatureOffsets &offsets) {
  if (type.empty())
    os << name;
  else
    dumpIdentifierWithType(os, name, type);
  if (!params.empty())
    printParameterList(params, shared, os, offsets.parameters);
}

/// Helper function to determine if a parameter should be excluded from docs.
///
/// Mojo parameters fall into several categories:
/// 1. Implicit parameters - internal compiler parameters, always hidden
/// 2. Compiler-synthesized inferred parameters - generated by compiler, hidden
/// 3. Explicitly declared inferred parameters - written in source code, shown
/// 4. Regular parameters - always shown
bool shouldExcludeParameterFromDocs(PassingKind passingKind,
                                    StringRef paramName) {
  // Always exclude implicit parameters (internal compiler parameters)
  if (passingKind == PassingKind::Implicit)
    return true;

  // Ignore name mangled parameters, which are autoparams.
  if (demangleParameterName(paramName, /*forUser*/ true) != paramName)
    return true;

  // Exclude inferred parameters that were synthesized by the compiler
  // (identifiable by empty names).
  return passingKind == PassingKind::Inferred && paramName.empty();
}

/// Try to parse a printed constraint string as "conforms_to(ParamName, Traits)"
/// where ParamName is a simple identifier (no dots). If successful, returns
/// true and fills in paramName and traitStr.
bool parseConformsToString(StringRef printed, StringRef &paramName,
                           StringRef &traitStr) {
  if (!printed.starts_with("conforms_to(") || !printed.ends_with(")"))
    return false;
  // Extract contents between "conforms_to(" and ")"
  StringRef inner = printed.drop_front(strlen("conforms_to(")).drop_back(1);
  // Split at the first ", " to get paramName and traits
  auto [lhs, rhs] = inner.split(", ");
  if (rhs.empty())
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
  ASTType::printParam(os, value, /*forDiag=*/&shared);
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
      convention = ArgConvention::ReadReg;
    } else {
      type = type.getParameterListInfo().elementType;
    }
  } else if (varKind == VariadicKind::PackVarArg) {
    // VariadicPack needs special printing - its argument isn't a type.
    os << "*";
    if (convention)
      type = RefType::stripRefConvention(type, *convention);
    ASTType::printParam(os, type.getVariadicPackInfo().typeList,
                        /*forDiag=*/&shared);
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

  // If this type is the same as the self type, use the "Self" keyword.
  if (selfType && type.isEqualCanon(*selfType))
    os << "Self";
  else
    os << type.getAsString(/*forDiag=*/&shared);

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
  ASTType::printParamAfterType(os, defaultValue, shared);
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

} // namespace M::KGEN
