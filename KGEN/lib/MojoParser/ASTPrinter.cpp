//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of the AST printing logic.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/DeclResolver.h"

#include "ParserEvaluationContext.h"

#include "KGEN/Interpreter/InterpreterAttrs.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPAttrs.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// Given a SymbolRefAttr, return the underlying symbol name.
static StringRef getNameFromSymbolRef(SymbolRefAttr symbol, bool isFunc) {
  StringAttr leaf;
  if (symbol.getNestedReferences().empty())
    leaf = symbol.getRootReference();
  else
    leaf = symbol.getNestedReferences().back().getAttr();

  // Demangle the function name.
  StringRef name = leaf.getValue();
  if (isFunc)
    if (size_t mangleStart = name.find('('); mangleStart != std::string::npos)
      name = name.take_front(mangleStart);
  return name;
}

/// Try to extract a symbol reference and parameter list from a function callee.
/// Returns the symbol being called and the parameters, or a null symbol if
/// decoding failed.
static std::pair<SymbolRefAttr, ArrayRef<TypedAttr>>
tryGetSymbolNameAndParams(TypedAttr param) {
  param = ParamOperatorAttr::stripRebind(param);
  if (auto symbolCst = sugarDynCast<SymbolConstantAttr>(param))
    return {symbolCst.getSymbol(), symbolCst.getParamValues()};
  return {{}, {}};
}

/// If the value is a call to an implicit constructor of the value's type,
/// remove it, otherwise leave the value alone.
///
/// When shared is specified, this makes sure to only strip implicit
/// constructors, but when it is null it will always strip all constructors.
static void removeImplicitCtorCall(TypedAttr &value, SharedState *shared) {
  // Look through SugarAttr to find the underlying apply if present. We only
  // need to look through at most one SugarAttr, because the sugared side has
  // any nested sugars removed already.
  auto valueWithoutSugar = value;
  if (auto sugar = dyn_cast<SugarAttr>(valueWithoutSugar))
    if (shared || sugar.getKind() == SugarKind::AlwaysInlineBuiltin)
      valueWithoutSugar = sugar.getSugared();

  // Implicit constructors are always calls.
  auto op = dyn_cast<ParamOperatorAttr>(valueWithoutSugar);
  if (!op ||
      (op.getOpcode() != POC::Apply &&
       op.getOpcode() != POC::ApplyResultSlot) ||
      op.getOperands().size() != 2) // callee and value to convert.
    return;

  auto [nameAttr, calleeParams] =
      tryGetSymbolNameAndParams(op.getOperands()[0]);
  if (!nameAttr)
    return;
  StringRef name = getNameFromSymbolRef(nameAttr, /*isFunc=*/true);
  if (!name.starts_with("__init__"))
    return;

  if (shared) {
    ASTDecl *decl = shared->getDeclResolver().getDeclForFuncSymbol(nameAttr);
    auto calleeFn = cast<FnOp>(decl->getIfOperation());
    if (!calleeFn.isImplicitConversion())
      return; // If it's not an implicit conversion, don't remove it.
  }

  value = op.getOperands()[1];
}

// Get the name of the enclosing struct from the function symbol reference.
static StringRef tryGetTypeNameFromSymbolRef(SymbolRefAttr symbol) {
  if (symbol.getNestedReferences().size() >= 2)
    return symbol.getNestedReferences().drop_back().back().getValue();
  return {};
}

// If we are a builtin symbol, then just strip everything but the name of the
// type. E.g. Print ::Int instead of stdlib::builtin::int::Int.
static StringRef trimBuiltinNamespace(StringRef nestedSymbolName) {
  // List of common namespace prefixes to trim
  static const StringRef commonPrefixes[] = {
      "stdlib::", "layout::"
      // Add other common prefixes here
  };

  StringRef prettyName(nestedSymbolName);
  for (StringRef prefix : commonPrefixes) {
    if (prettyName.starts_with(prefix)) {
      const size_t lastSeparatorLoc = prettyName.rfind("::");
      if (lastSeparatorLoc != StringRef::npos)
        return prettyName.substr(lastSeparatorLoc);
    }
  }

  return prettyName;
}

static void printSymbol(raw_ostream &os, SymbolRefAttr symbol,
                        SharedState *diagShared, bool isFunc) {
  // When mangling, keep things simple.
  if (diagShared == nullptr) {
    std::string nestedSymbolName;
    llvm::raw_string_ostream buff(nestedSymbolName);
    printNestedSymbolReference(buff, symbol);
    os << trimBuiltinNamespace(nestedSymbolName);
    return;
  }

  // When printing for diagnostics and the user, we can cut things down to make
  // them more readable.
  StringRef name = getNameFromSymbolRef(symbol, isFunc);

  // Remove stdlib:: prefixes.
  name = trimBuiltinNamespace(name);

  // The symbol is mangled and therefore will have parameter type information
  // in it, remove these if present.  This turn things like
  //    `foo[::Intable,::Intable,::Intable,::DType]` -> `foo`.
  name = name.take_front(name.find('['));
  os << name;
}

/// Given a parameter list for a function or struct, print it out in a nice
/// user-readable format (e.g. eliding infer-only and defaulted parameters).
///
/// This needs to handle the case when 'paramInfo' is null, e.g. when mangling.
///
/// The 'typesImplied' boolean indicates when we're in a struct - we can omit
/// implicit conversions to tidy up the printout because struct's can't be
/// overloaded on parameter sets like functions are.
static void printParamList(raw_ostream &os, PogListAttr paramInfo,
                           ArrayRef<TypedAttr> params, SharedState *diagShared,
                           bool typesImplied) {
  if (params.empty())
    return;

  SmallVector<std::pair<StringAttr, TypedAttr>> paramsToPrint;

  // If we're printing for diagnostics, we'll have 'paramInfo'.  In that case we
  // want to avoid printing defaulted parameter values that are the same as
  // their default value.
  if (paramInfo) {
    assert(paramInfo.size() == params.size() &&
           "Unexpected number of bound params");

    ParameterEvaluator evaluator(params);

    // Find out about default parameter values.
    DefaultValueHandler defaultValueHandler(paramInfo);
    bool skippedPositional = false;
    for (auto [idx, pog, paramValue] :
         llvm::enumerate(paramInfo.getPogs(), params)) {

      auto passingKind = pog.getPassingKind();

      // See if this parameter has a default value.  If so, and if the
      // provided value matches it, then don't print the parameter in the
      // list.
      if (auto def = defaultValueHandler.getDefault(idx)) {
        // Make sure to substitute other parameter values in, e.g. so we can
        // handle things like:
        //   struct UnsafePointer[type: AnyType,
        //                        align: Int = _default_alignment[type]()]:
        def = evaluator.getReboundAttribute(def);
        if (paramValue == def && passingKind != PassingKind::PosOnly) {
          // If we skip a posOrKw then include keyword names for any other
          // posOrKw's that come after it.
          skippedPositional |= (passingKind == PassingKind::PosOrKw);
          continue;
        }
      }

      StringAttr name;
      switch (passingKind) {
      case PassingKind::Implicit:
      case PassingKind::Inferred:
        continue; // Don't print implicit parameters at all.
      case PassingKind::PosOnly:
        break; // Never include a name.
      case PassingKind::PosOrKw:
        if (!skippedPositional)
          break; // Don't include a name unless we skipped another one.
        [[fallthrough]];
      case PassingKind::KwOnly:
        name = paramInfo.getName(idx);
        break;
      }
      paramsToPrint.push_back({name, paramValue});
    }

  } else {
    // When generating mangled names, don't include names for parameters since
    // positional information is enough.
    for (TypedAttr paramValue : params)
      paramsToPrint.push_back({StringAttr(), paramValue});
  }

  if (!paramsToPrint.empty()) {
    os << '[';
    llvm::interleaveComma(paramsToPrint, os,
                          [&](std::pair<StringAttr, TypedAttr> param) {
                            if (param.first)
                              os << param.first.strref() << '=';
                            if (typesImplied && diagShared)
                              removeImplicitCtorCall(param.second, diagShared);
                            ASTType::printParam(os, param.second, diagShared);
                          });
    os << ']';
  }
}

/// Print the input parameter types of a generator type/attr.
/// This needs to handle the case when 'paramInfo' is null.
/// An additional attr/type body can be provided that will also be rebound with
/// parameter names (if available in paramInfo) and returned.
template <typename BodyT>
static BodyT printGeneratorInterface(raw_ostream &os,
                                     ArrayRef<Type> inputParamTypes,
                                     PogListAttr paramInfo,
                                     SharedState *diagShared, BodyT body) {
  os << '[';
  if (paramInfo && diagShared) {
    ParameterEvaluator evaluator;
    PassingKindPrinter passingKindPrinter(os, paramInfo);
    DefaultValueHandler defaultValueHandler(paramInfo);
    auto printFn = [&](auto p) {
      auto [i, type] = p;
      passingKindPrinter.printOptionalStarSlash(i);

      Type reboundType = evaluator.getReboundType(type);

      StringRef name = paramInfo.getName(i).strref();
      if (!name.empty()) {
        os << name << ": ";
        evaluator.appendIndexBinding(ParamDeclRefAttr::get(name, reboundType));
      } else {
        // If no name exists, keep as is.
        evaluator.appendIndexBinding(ParamIndexRefAttr::get(i, reboundType));
      }

      ASTType(reboundType).print(os, diagShared);

      if (TypedAttr defaultOr = defaultValueHandler.getDefault(i)) {
        os << " = ";
        ASTType::printParam(os, defaultOr, diagShared);
      }

      passingKindPrinter.printOptionalTrailingSlash(i);
    };
    llvm::interleaveComma(llvm::enumerate(inputParamTypes), os, printFn);

    if constexpr (std::is_base_of_v<Attribute, BodyT>)
      body = cast<BodyT>(evaluator.getReboundAttribute(body));
    else
      body = cast<BodyT>(evaluator.getReboundType(body));
  } else {
    // If no param metadata, just print the types.
    auto printFn = [&](Type type) { ASTType(type).print(os, diagShared); };
    llvm::interleaveComma(inputParamTypes, os, printFn);
  }
  os << ']';

  return body;
}

/// If the parameter being referenced is an auto-parameterization of the
/// current function or struct, dig it out so we can print the correct name.
/// Consider something like:
///    struct S[a: Scalar]:
///       fn f(b: Scalar):
///          use(a.dtype, b.dtype)
/// Both "a.dtype" and "b.dtype" will resolve to a (mangled) string of
/// "dtype", but we would really like to print them as "a.dtype" so the user
/// knows what is going on, and we don't get a T != T error.
///
/// This returns success when handled, failure otherwise.
static void prettyPrintParamName(ParamDeclRefAttr declRef, SharedState &shared,
                                 raw_ostream &os) {
  ASTDecl *curDecl = shared.declResolver->getDeclCurrentlyProcessing();

  // Walk up the decl hierarchy to find the one that contains the parameter.
  PogListAttr paramListAttr;
  ArrayRef<ParamDeclAttr> paramDecls;
  ssize_t paramIdx = -1;
  [[maybe_unused]] size_t numImplicitOrigins = 0;
  while (curDecl) {
    // TODO: we need a decl interface to do this!
    if (auto fnDecl =
            dyn_cast_if_present<LIT::FnOp>(curDecl->getIfOperation())) {
      paramListAttr = fnDecl.getFuncTypeGenerator().getParamListAttrs();
      paramDecls = fnDecl.getParams();
      numImplicitOrigins =
          fnDecl.getFuncTypeGenerator().getNumImplicitOriginDecls();
    }
    if (auto structDecl =
            dyn_cast_if_present<LIT::StructDeclOp>(curDecl->getIfOperation())) {
      paramListAttr = structDecl.getSignature().getParamListAttrs();
      paramDecls = structDecl.getParams();
      numImplicitOrigins = 0;
    }

    if (paramListAttr) {
      assert(paramListAttr.size() + numImplicitOrigins == paramDecls.size() &&
             "Unexpected number of parameters");

      for (auto [idx, param] : llvm::enumerate(paramDecls)) {
        if (param.getName() == declRef.getName()) {
          paramIdx = idx;
          break;
        }
      }
      if (paramIdx != -1)
        break;
    }
    curDecl = curDecl->getParentDecl();
  }

  // If this is an implicit parameter injected due to auto-parameterization,
  // then it will have a uniquing identifier on it, rip that off.
  auto demangledName = demangleParameterName(declRef.getName());

  // If we didn't find it, or is something like an implicit origin reference
  // then there is nothing to do.
  if (paramIdx == -1) {
    os << demangledName;
    return;
  }

  // Handle implicit origins.
  if (size_t(paramIdx) >= paramListAttr.size()) {
    assert(isa<OriginType>(paramDecls[paramIdx].getType()) &&
           "Only unnamed thing should be an implicit origin");
    os << "origin_of(" << demangledName << ")";
    return;
  }

  // If the name wasn't mangled, then it is a normal user parameter, just
  // print it.
  if (demangledName == declRef.getName()) {
    os << demangledName;
    return;
  }

  // Otherwise, check to see if it is an autoparam.  It could be an autoparam of
  // another parameter, or could be an autoparam for an argument type of a
  // function. Check parameters first (handling structs and functions).
  for (auto paramDecl : paramDecls) {
    for (auto p : ASTType(paramDecl.getType()).getParamBindings()) {
      if (p == declRef) {
        // The param found may itself be an autoparam.  Recurse to print it.
        ASTType::printParam(os, ParamDeclRefAttr::get(paramDecl), &shared);
        os << "." << demangledName;
        return;
      }
    }
  }

  // If this is a function, it may be an auto-param for an argument type.
  if (auto fnDecl = dyn_cast_if_present<LIT::FnOp>(curDecl->getIfOperation())) {
    auto fnSig = fnDecl.getFuncTypeGenerator();
    for (auto [idx, argType] :
         llvm::enumerate(fnDecl.getFunctionType().getInputs())) {
      auto printArgName = [&]() {
        auto argName = fnSig.getArgName(idx).strref();
        if (!argName.empty())
          os << argName;
        else
          os << "arg" << idx;
      };

      auto userArgType =
          RefType::stripRefConvention(argType, fnSig.getArgConvention(idx));
      if (llvm::is_contained(ASTType(userArgType).getParamBindings(),
                             declRef)) {
        printArgName();
        os << "." << demangledName;
        return;
      }

      // If is possible that this parameter is an autoparam origin or mut bool
      // for a ref argument.  Check to see if that is the case.
      if (auto refType = dyn_cast<RefType>(argType))
        if (auto refOrigin = dyn_cast<ParamDeclRefAttr>(refType.getOrigin())) {
          if (refOrigin.getName() == declRef.getName()) {
            os << "origin_of(";
            printArgName();
            os << ")";
            return;
          }
        }
    }
  }

  os << demangledName;
}

/// Pretty print a parameter value.
void ASTType::printParam(raw_ostream &os, TypedAttr param,
                         SharedState *diagShared) {

  auto printOperands =
      [&](ArrayRef<TypedAttr> operands, StringRef separator = ", ",
          StringRef lSeparator = "(", StringRef rSeparator = ")") -> void {
    os << lSeparator;
    llvm::interleave(
        operands, os,
        [&](TypedAttr value) {
          // Don't print extracts out of Int.value.
          if (auto extract = dyn_cast<LIT::StructExtractAttr>(value))
            value = extract.getStructValue();
          printParam(os, value, diagShared);
        },
        separator);
    os << rSeparator;
  };

  if (auto bindParams = dyn_cast<BindParamsAttr>(param)) {
    printParam(os, bindParams.getGenerator(), diagShared);
    printOperands(bindParams.getParamValues(), ", ", "[", "]");
    return;
  }
  if (auto genAttr = dyn_cast<GeneratorAttr>(param)) {
    os << "alias";
    TypedAttr reboundBody =
        printGeneratorInterface(os, genAttr.getInputParamTypes(),
                                dyn_cast<PogListAttr>(genAttr.getMetadata()),
                                diagShared, genAttr.getBody());
    os << ' ';
    printParam(os, reboundBody, diagShared);
    return;
  }
  if (auto symbolCst = dyn_cast<SymbolConstantAttr>(param)) {
    printSymbol(os, symbolCst.getSymbol(), diagShared, /*isFunc=*/true);
    if (!symbolCst.getParamValues().empty())
      printOperands(symbolCst.getParamValues(), ", ", "[", "]");
    return;
  }
  if (auto refPack = dyn_cast<RefPackAttr>(param)) {
    llvm::interleaveComma(refPack.getValues(), os, [&](TypedAttr value) {
      printParam(os, value, diagShared);
    });
    return;
  }

  if (auto op = dyn_cast<ParamOperatorAttr>(param)) {
    ArrayRef<TypedAttr> operands = op.getOperands();

    // Sugar the parameter operators the parser can generate.
    switch (op.getOpcode()) {
    case POC::Apply:
    case POC::ApplyResultSlot: {
      // Check if we're applying a known symbol, in which case we can do some
      // more specialized printing.
      auto [nameAttr, calleeParams] =
          tryGetSymbolNameAndParams(operands.front());
      if (!nameAttr) {
        // If we're calling a parameter of function type, print it as a normal
        // call.
        printParam(os, operands.front(), diagShared);
        return printOperands(operands.drop_front());
      }

      ArrayRef<TypedAttr> operandsToPrint = operands.drop_front();
      StringRef name = getNameFromSymbolRef(nameAttr, /*isFunc=*/true);
      // Don't print conversions of boolean's to i1.
      if (name == "__mlir_i1__" && operands.size() == 2)
        return printParam(os, operands.back(), diagShared);

      // Print arithmetic functions using their mathematical form rather than
      // as dunder method calls.
      static SmallDenseMap<StringRef, StringRef> binaryOpNames{
          {"__add__", " + "},     {"__sub__", " - "},
          {"__mul__", " * "},     {"__mod__", " % "},
          {"__truediv__", " / "}, {"__floordiv__", " // "},
          {"__xor__", " ^ "},     {"__and__", " & "},
          {"__or__", " | "},      {"__lshift__", " << "},
          {"__rshift__", " >> "}, {"__eq__", " == "},
          {"__lt__", " < "},      {"__le__", " <= "},
          {"__in__", " in "},     {"__ne__", " != "},
          {"__gt__", " > "},      {"__ge__", " >= "},
          {"__matmul__", " @ "},  {"__pow__", " ** "},
          {"__is__", " is "},     {"__isnot__", " isnot "},
      };
      if (auto it = binaryOpNames.find(name); it != binaryOpNames.end())
        return printOperands(operandsToPrint, /*separator=*/it->second);

      // Print `x.__getitem__(args...)` as `x[args...]`
      if (name == "__getitem__" && !operandsToPrint.empty()) {
        printParam(os, operandsToPrint.front(), diagShared);
        os << '[';
        llvm::interleaveComma(
            operandsToPrint.slice(1), os,
            [&](const TypedAttr &value) { printParam(os, value, diagShared); });
        os << ']';
        return;
      }

      // Try to resolve the symbol to a ASTDecl and then to a FnOp.
      FnOp calleeFn;
      bool calleeIsMethod = false;
      if (diagShared) {
        if (ASTDecl *decl =
                diagShared->getDeclResolver().getDeclForFuncSymbol(nameAttr)) {
          calleeFn = cast<FnOp>(decl->getIfOperation());
          calleeIsMethod = decl->tryGetMethodParentDecl() != nullptr;
        }
      }

      bool calleeIsStatic =
          calleeFn && calleeIsMethod && calleeFn.getIsStatic();

      // If we can tell that this is a method call, print the receiver first.
      if (!operandsToPrint.empty() && calleeIsMethod && !calleeIsStatic) {
        printParam(os, operandsToPrint.front(), diagShared);
        os << '.';
        operandsToPrint = operandsToPrint.drop_front();
      }

      // Special case: struct __init__ constructor calls for literal types
      if (name.starts_with("__init__") && diagShared && operands.size() >= 2) {

        // Helper function to check if this is a literal wrapper by name
        auto isLiteralWrapperName = [](StringRef structName) {
          return structName == "StringLiteral" || structName == "IntLiteral" ||
                 structName == "FloatLiteral" || structName == "Origin";
        };

        // Helper function to try printing just the literal value
        auto tryPrintLiteralValue = [&](ArrayRef<TypedAttr> args) -> bool {
          if (args.size() == 1) {
            printParam(os, args[0], diagShared);
            return true;
          }
          return false;
        };

        // Primary approach: Use symbol structure to get struct name
        StringRef structName = tryGetTypeNameFromSymbolRef(nameAttr);
        if (isLiteralWrapperName(structName)) {
          if (tryPrintLiteralValue(operandsToPrint))
            return;
        }

        // Fallback: Check if the symbol name contains type suffixes
        if (name.contains("[__mlir_type.!kgen.string]") ||
            name.contains("[__mlir_type.!pop.int_literal]") ||
            name.contains("[__mlir_type.!pop.float_literal]")) {
          if (tryPrintLiteralValue(operandsToPrint))
            return;
        }
      }

      // For constructors, print the type name instead of __init__.
      if (name == "__init__" && nameAttr.getNestedReferences().size() >= 2) {
        os << tryGetTypeNameFromSymbolRef(nameAttr);
      } else {
        // Static methods print 'StructName.method', not just 'method'.
        if (calleeIsStatic)
          os << tryGetTypeNameFromSymbolRef(nameAttr) << '.';

        // Otherwise, print the symbol name.
        printSymbol(os, nameAttr, diagShared, /*isFunc=*/true);
      }

      // If there are parameters, print them, eliding infer-only and defaulted
      // parameter values.
      PogListAttr paramInfo;
      if (calleeFn)
        paramInfo = calleeFn.getFullSignature().getParamListAttrs();
      // typesImplied=false because we don't want to elide implicit conversions
      // constructor calls, because the function could be overloaded.  We could
      // check to see if the function is not overloaded and elide it.
      printParamList(os, paramInfo, calleeParams, diagShared,
                     /*typesImplied*/ false);

      // Finally, also print any operands.
      return printOperands(operandsToPrint);
    }
    case POC::Cond: {
      printParam(os, operands[1], diagShared);
      os << " if ";
      auto cond = operands[0];
      // Don't print extracts of Bool.value.
      if (auto extract = dyn_cast<LIT::StructExtractAttr>(cond))
        cond = extract.getStructValue();
      printParam(os, cond, diagShared);
      os << " else ";
      printParam(os, operands[2], diagShared);
      return;
    }
    case POC::Rebind:
      // Just omit the types.
      printParam(os, operands.front(), diagShared);
      return;
    case POC::VariadicGet:
      printParam(os, operands.front(), diagShared);
      os << '[';
      printParam(os, operands.back(), diagShared);
      os << ']';
      return;
    default:
      const char *binOp = nullptr;
      switch (op.getOpcode()) {
      case POC::Add:
        binOp = " + ";
        break;
      case POC::Mul:
      case POC::MulNoWrap:
        binOp = " * ";
        break;
      case POC::Div:
        binOp = " / ";
        break;
      case POC::Mod:
        binOp = " % ";
        break;
      case POC::And:
        binOp = " & ";
        break;
      case POC::Or:
        binOp = " | ";
        break;
      case POC::Xor:
        binOp = " ^ ";
        break;
      case POC::Shl:
        binOp = " << ";
        break;
      case POC::Shr:
        binOp = " >> ";
        break;
      case POC::EQ:
        binOp = " == ";
        break;
      case POC::LT:
        binOp = " < ";
        break;
      case POC::LE:
        binOp = " <= ";
        break;
      case POC::In:
        binOp = " in ";
        break;
      default:
        break;
      }
      // Simple things that show up in integer param expressions.
      if (binOp)
        return printOperands(operands, /*separator=*/binOp);

      // Otherwise, fall back to printing as a parenthesized form like the KGEN
      // printer does.  We don't fall back to the kgen printer because it will
      // print nested subexpressions as KGEN and lose all sugar.
      os << '(' << stringifyEnum(op.getOpcode()) << ' ';
      llvm::interleaveComma(operands, os, [&](TypedAttr operand) {
        printParam(os, operand, diagShared);
      });
      os << ')';
      return;
    }
  }
  if (auto getWitness = dyn_cast<GetWitnessAttr>(param)) {
    printParam(os, getWitness.getTypeValue(), diagShared);
    os << "." << getWitness.getWitnessName().strref();
    return;
  }
  if (auto typeAttr = dyn_cast<TypeParamAttr>(param)) {
    ASTType(typeAttr.getMlirType()).print(os, diagShared);
    return;
  }
  if (auto upcast = dyn_cast<UpcastAttr>(param))
    return printParam(os, upcast.getInputTypeValue(), diagShared);

  if (auto extractAttr = dyn_cast<LIT::StructExtractAttr>(param)) {
    printParam(os, extractAttr.getStructValue(), diagShared);
    os << '.' << extractAttr.getField().getValue();
    return;
  }
  if (auto variadicCst = dyn_cast<VariadicAttr>(param)) {
    // VariadicAttr appears in a pack list, so it doesn't need extra []'s
    // around it.
    llvm::interleaveComma(variadicCst.getValues(), os, [&](TypedAttr value) {
      printParam(os, value, diagShared);
    });
    return;
  }
  if (auto indexRef = dyn_cast<ParamIndexRefAttr>(param)) {
    os << '$';
    if (size_t depth = indexRef.getDepth())
      os << depth << '|';
    os << indexRef.getIndex();
    return;
  }
  if (auto memAttr = dyn_cast<StoreToMemAttr>(param))
    return printParam(os, memAttr.getValue(), diagShared);

  if (auto dtypeAttr = dyn_cast<DTypeConstantAttr>(param)) {
    if (diagShared)
      os << "DType.";
    os << dtypeAttr.getDType().getAsString(/*libForm=*/true);
    return;
  }

  // Special case bool constants instead of printing as 0/1.
  if (auto boolAttr = dyn_cast<BoolAttr>(param)) {
    os << (boolAttr.getValue() ? "True" : "False");
    return;
  }

  if (auto noneAttr = dyn_cast<NoneAttr>(param)) {
    os << "None";
    return;
  }

  if (auto strAttr = dyn_cast<StringAttr>(param)) {
    os << '"';
    printAsMojoStringLiteral(strAttr, os);
    os << '"';
    return;
  }

  /// A StructAttr is due to an inline @always_inline("builtin") initializer.
  /// Elide it if we have the default type with a literal so we don't print
  /// Int(42), but print it if it is something weird like IntLiteral(42)
  if (auto structAttr = dyn_cast<LITStructAttr>(param)) {
    // If the struct has a single element, elide the braces.
    if (diagShared && structAttr.getValues().size() == 1) {
      StringRef typeName;
      if (auto structType = dyn_cast<StructType>(structAttr.getType()))
        typeName = structType.getSymbol().getLeafReference().strref();
      TypedAttr elt = std::get<1>(structAttr.getValues().front());
      if (typeName == "Int" || typeName == "UInt" || typeName == "Bool" ||
          typeName == "Origin" || typeName == "DType") {
        if (auto extract = dyn_cast<LIT::StructExtractAttr>(elt))
          elt = extract.getStructValue();
        printParam(os, elt, diagShared);
        return;
      }
    }

    ASTType(structAttr.getType()).print(os, diagShared);
    os << '(';
    // TODO: Could print keywords for the labels if there is a reason someday.
    llvm::interleaveComma(structAttr.getValues(), os, [&](auto elt) {
      TypedAttr value = std::get<1>(elt);
      if (auto extract = dyn_cast<LIT::StructExtractAttr>(value))
        value = extract.getStructValue();
      printParam(os, value, diagShared);
    });
    os << ')';
    return;
  }

  if (auto convert = dyn_cast<POP::IntLiteralConvertAttr>(param)) {
    printParam(os, convert.getInput(), diagShared);
    return;
  }

  if (auto intLitBin = dyn_cast<POP::IntLiteralBinAttr>(param)) {
    const char *binOp = nullptr;
    switch (intLitBin.getOper().getValue()) {
    case POP::IntLiteralBinKind::Add:
      binOp = " + ";
      break;
    case POP::IntLiteralBinKind::Sub:
      binOp = " - ";
      break;
    case POP::IntLiteralBinKind::Mul:
      binOp = " * ";
      break;
    case POP::IntLiteralBinKind::FloorDiv:
      binOp = " // ";
      break;
    case POP::IntLiteralBinKind::Mod:
      binOp = " % ";
      break;
    case POP::IntLiteralBinKind::Lshift:
      binOp = " << ";
      break;
    case POP::IntLiteralBinKind::Rshift:
      binOp = " >> ";
      break;
    case POP::IntLiteralBinKind::And:
      binOp = " & ";
      break;
    case POP::IntLiteralBinKind::Or:
      binOp = " | ";
      break;
    case POP::IntLiteralBinKind::Xor:
      binOp = " ^ ";
      break;
    }

    return printOperands({intLitBin.getLhs(), intLitBin.getRhs()},
                         /*separator=*/binOp);
  }

  if (auto fpLit = dyn_cast<POP::FloatLiteralAttr>(param)) {
    switch (fpLit.getSpecial().getValue()) {
    case POP::FloatLiteralSpecialValues::NegZero:
      os << "-0.0";
      return;
    case POP::FloatLiteralSpecialValues::Inf:
      os << "inf";
      return;
    case POP::FloatLiteralSpecialValues::NegInf:
      os << "-inf";
      return;
    case POP::FloatLiteralSpecialValues::Nan:
      os << "nan";
      return;
    case POP::FloatLiteralSpecialValues::Normal:
      // Convert to f64 to print out the value.
      auto ctx = fpLit.getContext();
      auto f64Type = POP::SIMDType::get(ctx, 1, DType::f64);
      auto simdVal = cast<POP::SIMDAttr>(
          POP::FloatLiteralConvertAttr::get(ctx, f64Type, fpLit));
      os << simdVal.getValues()[0].getFloatVal();
      return;
    }
  }

  // IntLiteral/FloatLiteral/StringLiteral are stateless values that end up as
  // UnknownAttr.
  if (isa<UnknownAttr>(param) && diagShared) {
    StringRef typeName;
    if (auto structType = dyn_cast<StructType>(param.getType()))
      typeName = structType.getSymbol().getLeafReference().strref();
    if (typeName == "IntLiteral" || typeName == "FloatLiteral" ||
        typeName == "StringLiteral") {
      auto structType = cast<LIT::StructType>(param.getType());
      if (structType.getParamValues().size() == 1) {
        printParam(os, structType.getParamValues()[0], diagShared);
        return;
      }
    }
  }

  // Print ParamDeclRefAttr as the name of the parameter.
  if (auto declRef = dyn_cast<ParamDeclRefAttr>(param)) {
    if (!diagShared)
      // Escape any weird characters in the parameter name that might have
      // been introduced with backticks.
      return printAsMojoStringLiteral(declRef.getName(), os);

    return prettyPrintParamName(declRef, *diagShared, os);
  }

  // These are origins but don't need `origin_of(...)` around them.
  if (auto anyOrig = dyn_cast<AnyOriginAttr>(param)) {
    if (anyOrig.getType().isMutableKnown(true))
      os << "MutableAnyOrigin";
    else if (anyOrig.getType().isMutableKnown(false))
      os << "ImmutableAnyOrigin";
    else
      os << "SomeAnyOrigin";
    return;
  }
  if (auto comptimeOrig = dyn_cast<ComptimeOriginAttr>(param)) {
    os << "ComptimeOrigin";
    return;
  }
  if (auto originField = dyn_cast<OriginFieldAttr>(param)) {
    if (isa<StaticOriginAttr>(originField.getBase())) {
      if (originField.getField().str() == "__constants__" &&
          originField.getType().isMutableKnown(false)) {
        os << "StaticConstantOrigin";
        return;
      }
    }
  }

  // Origins are handled with their own grammar that has `origin_of(x)` on the
  // outside.
  if (isa<OriginType>(param.getType())) {
    os << "origin_of(";
    // Flatten unions into a comma separated list.
    if (auto unionAttr = dyn_cast<OriginUnionAttr>(param)) {
      llvm::interleaveComma(unionAttr.getOperands(), os, [&](TypedAttr param) {
        printOriginParam(os, param, diagShared);
      });
    } else {
      printOriginParam(os, param, diagShared);
    }
    os << ")";
    return;
  }

  if (auto sugar = dyn_cast<SugarAttr>(param)) {
    // Sugared parameters print as their sugar when always_inline("builtin")
    // even for mangled names because the arguments should be unique.  We don't
    // sugar other things though because the identifiers may not be fully
    // qualified. (TODO: it would be great to do this!)
    if (diagShared || sugar.getKind() == SugarKind::AlwaysInlineBuiltin)
      return printParam(os, sugar.getSugared(), diagShared);
    return printParam(os, sugar.getOriginal(), diagShared);
  }

  // Handle other KGEN parameters that it knows about with an ugly fallback.
  // TODO: Remove this - we should cover all attrs here, anything that falls
  // back should be an error/assertion.
  os << KGEN::getParamAsString(param);
}

/// Print the specified parameter like we would in an origin expression, works
/// in an `origin_of(x)` body.
void ASTType::printOriginParam(raw_ostream &os, TypedAttr param,
                               SharedState *diagShared) {

  if (auto originField = dyn_cast<OriginFieldAttr>(param)) {
    if (isa<StaticOriginAttr>(originField.getBase())) {
      if (originField.getField().str() == "__constants__" &&
          originField.getType().isMutableKnown(false)) {
        os << "StaticConstantOrigin";
        return;
      }
    }

    printOriginParam(os, originField.getBase(), diagShared);
    os << '.' << originField.getField().str();
    return;
  }
  if (auto originUnion = dyn_cast<OriginUnionAttr>(param)) {
    os << '{';
    llvm::interleaveComma(originUnion.getOperands(), os, [&](TypedAttr param) {
      printOriginParam(os, param, diagShared);
    });
    os << '}';
    return;
  }

  if (auto indirect = dyn_cast<IndirectOriginAttr>(param)) {
    printOriginParam(os, indirect.getBase(), diagShared);
    os << "[]";
    return;
  }

  if (auto mutcast = dyn_cast<OriginMutCastAttr>(param)) {
    if (mutcast.getType().isMutableKnown(false))
      os << "(muttoimm ";
    else
      os << "(mutcast ";
    printOriginParam(os, mutcast.getOperand(), diagShared);
    os << ")";
    return;
  }

  if (auto anyOrig = dyn_cast<AnyOriginAttr>(param)) {
    if (anyOrig.getType().isMutableKnown(true))
      os << "MutableAnyOrigin";
    else if (anyOrig.getType().isMutableKnown(false))
      os << "ImmutableAnyOrigin";
    else
      os << "SomeAnyOrigin";
    return;
  }

  if (auto comptimeOrig = dyn_cast<ComptimeOriginAttr>(param)) {
    os << "ComptimeOrigin";
    return;
  }

  if (auto originRef = dyn_cast<ImplicitOriginRefAttr>(param)) {
    // TODO: Should improve this when diagShared is present so references to
    // function types know what context they are in and can resolve these.
    os << "*[" << originRef.getDepth() << ',' << originRef.getIndex() << ']';
    return;
  }

  if (auto declRef = dyn_cast<ParamDeclRefAttr>(param)) {
    // If the parameter is an implicit parameter injected due to
    // auto-parameterization, print the thing that is being parameterized.
    StringRef name = declRef.getName();

    if (diagShared)
      name = demangleParameterName(name);
    // Escape any weird characters in the parameter name that might have
    // been introduced with backticks.
    printAsMojoStringLiteral(name, os);
    return;
  }

  if (isa<StructExtractAttr, ParamIndexRefAttr, ParamOperatorAttr>(param))
    return printParam(os, param, diagShared);

  if (auto sugar = dyn_cast<SugarAttr>(param)) {
    param = sugar.getSugared();
    // Remove implicit Origin ctor calls to reduce noise since they're implicit.
    removeImplicitCtorCall(param, diagShared);
    return printOriginParam(os, param, diagShared);
  }

  param.dump();
  llvm_unreachable("unknown origin parameter");
}

/// Given a parameter value of MLIR wrapper type like Bool or Int or DType,
/// dig out the single element of the struct with the specified type.
template <typename T>
static T getSingleElementStructAttr(TypedAttr param) {
  if (auto strParam = dyn_cast<LITStructAttr>(param)) {
    if (strParam.getValues().size() == 1)
      return dyn_cast<T>(std::get<1>(strParam.getValues()[0]));
  }
  return {};
}

void ASTType::print(raw_ostream &os, SharedState *diagShared) const {
  if (!mlirType) {
    os << "<<NULL ASTTYPE>>";
    return;
  }

  Type type = mlirType;
  auto printUserType = [&](SymbolRefAttr symbol, ArrayRef<TypedAttr> params,
                           ASTDecl *typeDecl) {
    // Handle special cases that should be aliased.
    // FIXME(MOCO-367): maintain "typedef" sugar in the type system.
    if (typeDecl &&
        isa_and_nonnull<LIT::StructDeclOp>(typeDecl->getIfOperation())) {
      auto structDecl = cast<LIT::StructDeclOp>(typeDecl->getIfOperation());
      if (params.size() == 1 && structDecl.getDeclName().strref() == "Origin") {
        // Check to see if we have a Bool with a known constant parameter.
        //   #lit.struct<{value: i1 = 1}>
        if (auto value = getSingleElementStructAttr<BoolAttr>(params[0])) {
          os << (value.getValue() ? "MutableOrigin" : "ImmutableOrigin");
          return;
        }
      }

      // Handle SIMD[dt, 1] with various aliases. Note that the dtype and size
      // may be parametric.
      if (params.size() == 2 && structDecl.getDeclName().strref() == "SIMD") {
        // DType and Int looks like #lit.struct<{value: index = 42}>, dig.
        DTypeConstantAttr dtype =
            getSingleElementStructAttr<DTypeConstantAttr>(params[0]);
        IntegerAttr size = getSingleElementStructAttr<IntegerAttr>(params[1]);
        if (size && size.getInt() == 1) {
          // This list should be kept in sync with the aliases in simd.mojo.
          static std::pair<KGENDType, const char *> dtypeAliases[] = {
              {KGENDType::si8, "Int8"},
              {KGENDType::ui8, "UInt8"},
              {KGENDType::si16, "Int16"},
              {KGENDType::ui16, "UInt16"},
              {KGENDType::si32, "Int32"},
              {KGENDType::ui32, "UInt32"},
              {KGENDType::si64, "Int64"},
              {KGENDType::ui64, "UInt64"},
              {KGENDType::si128, "Int128"},
              {KGENDType::ui128, "UInt128"},
              {KGENDType::si256, "Int256"},
              {KGENDType::ui256, "UInt256"},
              {KGENDType::f8e5m2, "Float8_e5m2"},
              {KGENDType::f8e5m2fnuz, "Float8_e5m2fnuz"},
              {KGENDType::f8e4m3fn, "Float8_e4m3fn"},
              {KGENDType::f8e4m3fnuz, "Float8_e4m3fnuz"},
              {KGENDType::f8e3m4, "Float8_e3m4"},
              {KGENDType::bf16, "BFloat16"},
              {KGENDType::f16, "Float16"},
              {KGENDType::f32, "Float32"},
              {KGENDType::f64, "Float64"},
          };
          if (dtype) {
            for (auto [dtypeValue, dtypeName] : dtypeAliases) {
              if (dtype.getDType() == dtypeValue) {
                os << dtypeName;
                return;
              }
            }
          }

          // Otherwise if we know the size is 1, we can use Scalar[] alias,
          // even if the dtype is parametric.
          os << "Scalar[";
          printParam(os, params[0], diagShared);
          os << "]";
          return;
        }
      }
    }

    // Only print the leaf reference when pretty printing types.
    printSymbol(os, symbol, diagShared, /*isFunc=*/false);

    // Print any type parameters if we can find the struct.
    PogListAttr paramInfo;
    if (typeDecl) {
      paramInfo = cast<StructDeclOp>(typeDecl->getIfOperation())
                      .getSignature()
                      .getParamListAttrs();
    }

    printParamList(os, paramInfo, params, diagShared, /*typesImplied*/ true);
  };

  auto printConvention = [&os](ArgConvention conv) {
    if (conv == ArgConvention::OwnedMem)
      os << "var ";
    else if (conv == ArgConvention::Mut)
      os << "mut ";
    else if (conv == ArgConvention::ByRefResult)
      os << "out ";
  };

  auto printRef = [&](RefType refType) {
    os << "ref [";
    printOriginParam(
        os, OriginType::stripMutCastAndFieldExtract(refType.getOrigin()),
        diagShared);
    if (!refType.isDefaultAddrSpace()) {
      os << ", ";
      printParam(os, refType.getAddressSpace(), diagShared);
    }
    os << "] ";
  };

  if (auto structTy = dyn_cast<StructType>(type)) {
    ASTDecl *decl = nullptr;
    if (diagShared)
      decl = ASTType(type).getDecl(*diagShared);
    printUserType(structTy.getSymbol(), structTy.getParamValues(), decl);
  } else if (auto anyStruct = dyn_cast<StructMetaType>(type)) {
    ASTDecl *decl = nullptr;
    if (diagShared)
      decl = ASTType(anyStruct.getType()).getDecl(*diagShared);
    os << "AnyStruct[";
    printUserType(anyStruct.getSymbol(), anyStruct.getParamValues(), decl);
    os << ']';
  } else if (auto traitType = dyn_cast<TraitType>(type)) {
    llvm::interleave(
        traitType.getSymbols(), os,
        [&](SymbolRefAttr symbol) {
          printSymbol(os, symbol, diagShared, /*isFunc=*/false);
        },
        " & ");
  } else if (auto anyTrait = dyn_cast<AnyTraitType>(type)) {
    os << "AnyTrait[";
    ASTType(anyTrait.getTraitType()).print(os, diagShared);
    os << ']';
  } else if (isNoneType()) {
    os << "None";
  } else if (auto ref = dyn_cast<RefType>(type)) {
    printRef(ref);
    ASTType(ref.getElementType()).print(os, diagShared);
  } else if (auto variadic = dyn_cast<VariadicType>(type)) {
    os << "Variadic[";
    ASTType(variadic.getElementType()).print(os, diagShared);
    os << "]";
  } else if (auto sigGen = dyn_cast<FnTypeGeneratorType>(type)) {
    if (sigGen.isAsync())
      os << "async ";
    os << "fn";
    FnType sig = sigGen.getBody();
    if (!sigGen.getInputParamTypes().empty()) {
      sig =
          printGeneratorInterface(os, sigGen.getInputParamTypes(),
                                  sigGen.getParamListAttrs(), diagShared, sig);
    }
    os << '(';
    PassingKindPrinter passingKindPrinter(os, sig.getArgListAttrs());
    bool hadAnyNames = false;
    for (auto [idx, typeX, conventionX] :
         llvm::enumerate(sig.getArguments(), sig.getArgConventions())) {
      ASTType type = typeX;
      ArgConvention convention = conventionX;
      if (isResultSlot(convention))
        continue; // Don't print result in argument list.

      if (idx)
        os << ", ";
      passingKindPrinter.printOptionalStarSlash(idx);

      bool printStar = false;
      if (sig.isPosVarArg(idx)) { // Print with the element of the variadic.
        auto variadic = cast<VariadicType>(type);
        type = variadic.getElementType();
        convention = sig.getPosVarArgConvention(idx);
        printStar = true;
      }

      // The formal type is VariadicPack[] and the thing to print is a pack
      // attribute, not a type.
      StringAttr name = sig.getArgName(idx);
      hadAnyNames |= !name.empty();
      if (sig.isPack(idx)) {
        convention = sig.getPackVarArgConvention(idx);
        printConvention(convention);
        os << '*';
        if (!name.empty())
          os << name.getValue() << ": ";
        else
          os << ' ';
        os << '*';

        TypedAttr variadic =
            ASTType(sig.getIfVariadicPack(idx)).getVariadicPackTypeList();
        printParam(os, variadic, diagShared);
      } else {
        printConvention(convention);

        if (printStar)
          os << '*';

        if (convention == ArgConvention::Ref ||
            convention == ArgConvention::MutRef)
          printRef(cast<RefType>(type));

        if (!name.empty())
          os << name.getValue() << ": ";

        type = RefType::stripRefConvention(type, convention);
        type.print(os, diagShared);
      }

      // Check if we are at the end; if so, we might still have to print a
      // '/'. If we're pretty printing for a diagnostic, and don't have any
      // names, then we don't print the trailing slash. This makes the
      // extremely common case of a source signature `fn(...) -> ...` look
      // nicer.
      if (!diagShared || hadAnyNames)
        passingKindPrinter.printOptionalTrailingSlash(idx);
    }
    os << ')';
    for (auto [enabled, effect] :
         {std::make_pair(sig.isThrows(), "raises"),
          std::make_pair(sig.isCapturing(), "capturing"),
          std::make_pair(sig.isEscaping(), "escaping")})
      if (enabled)
        os << ' ' << effect;
    os << " -> ";
    Type resultType = sig.getUserResultType();

    if (sig.isRefResult()) {
      auto refType = cast<RefType>(resultType);
      printRef(refType);
      resultType = refType.getElementType();
    }

    if (isa<KGEN::NoneType>(resultType))
      os << "None";
    else
      ASTType(resultType).print(os, diagShared);
  } else if (auto paramRef = dyn_cast<ParamType>(type)) {
    if (auto downcast = dyn_cast<DowncastAttr>(paramRef.getParam())) {
      ASTType(downcast.getInputTypeValue()).print(os, diagShared);
      os << "(";
      ASTType(downcast.getType()).print(os, diagShared);
      os << ")";
    } else {
      printParam(os, paramRef.getParam(), diagShared);
    }
  } else if (isa<TypeType>(type)) {
    os << "AnyTrivialRegType";
  } else if (auto fnType = dyn_cast<FunctionType>(type)) {
    os << "fn (";
    llvm::interleaveComma(fnType.getInputs(), os, [&](Type type) {
      ASTType(type).print(os, diagShared);
    });
    os << ") -> (";
    llvm::interleaveComma(fnType.getResults(), os, [&](Type type) {
      ASTType(type).print(os, diagShared);
    });
    os << ')';
  } else if (auto originType = dyn_cast<OriginType>(type)) {
    if (originType.isMutableKnown(true))
      os << "MutableOrigin";
    else if (originType.isMutableKnown(false))
      os << "ImmutableOrigin";
    else {
      os << "Origin[";
      printParam(os, originType.isMutable(), diagShared);
      os << ']';
    }
  } else if (isa<OriginSetType>(type)) {
    // Use "OriginSet" type name instead of the internal "origin.set"
    os << "OriginSet";
  } else if (auto module = dyn_cast<ModuleType>(type)) {
    // Only print the leaf reference when pretty printing types.
    printSymbol(os, module.getSymbol(), diagShared, /*isFunc=*/false);
  } else {
    // Use KGEN pretty printing when printing bare MLIR types for diagnostics.
    if (diagShared)
      printKGENType(os, type);
    else
      os << "__mlir_type." << type;
  }
}

/// This is the same as printParam, but is only used user pretty printing
/// circumstances (not mangling) after emitting a type annotation.  This
/// avoids printing obvious implicit conversion calls.
void ASTType::printParamAfterType(raw_ostream &os, TypedAttr value,
                                  SharedState &shared) {
  removeImplicitCtorCall(value, &shared);

  // It is pretty common for function arguments to use default conversions
  // from the actual value they want, and may not be an
  // always_inline("builtin") constructor, e.g.:
  //   fn example(v: Optional[Int64] = None):
  // Without doing anything fancy, we would get something like:
  //   fn example(v: Optional[Int64] = Optional[Int64](None)):
  // Which is literally what is happening, but not very pretty.  To clean this
  // up, check to see if call is to an implicit constructor, and if so, elide
  // the call.
  printParam(os, value, /*forDiag=*/&shared);
}

/// Convert this type to a human readable string representation so it can be
/// printed out for diagnostics.
raw_ostream &M::KGEN::LIT::operator<<(raw_ostream &os, ASTType astType) {
  if (!astType)
    return os << "<<NULL ASTTYPE>>";
  astType.print(os);
  return os;
}

std::string ASTType::getAsString(SharedState *forDiags) const {
  std::string result;
  llvm::raw_string_ostream os(result);
  print(os, forDiags);

  // Having "@" in mangled names confuses gnu ld and triggers error at linking
  // stage. See issue #6918. So replacing "@" with "_".
  std::replace(result.begin(), result.end(), '@', '_');
  return os.str();
}

/// Get the specified parameter as a string.
std::string ASTType::getParamAsString(TypedAttr param,
                                      SharedState *diagShared) {
  std::string result;
  llvm::raw_string_ostream os(result);
  printParam(os, param, diagShared);
  return os.str();
}

/// Get the specified parameter as a string.
std::string ASTType::getOriginAsString(TypedAttr param,
                                       SharedState *diagShared) {
  std::string result;
  llvm::raw_string_ostream os(result);
  printOriginParam(os, param, diagShared);
  return os.str();
}
