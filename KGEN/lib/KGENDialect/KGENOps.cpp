//===- KGENOps.cpp --------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the KGEN dialect operations.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/ElaboratorOpInterface.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGENVerifyHelper.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// custom<ParamConstantOpValue>
//===----------------------------------------------------------------------===//

static ParseResult parseParamConstantOpValue(OpAsmParser &p, TypedAttr &value,
                                             Type &resultType) {
  if (parseColonTypeOrIndex(p, resultType) || p.parseEqual() || p.parseLess() ||
      parseParamValue(p, value, resultType) || p.parseGreater())
    return failure();
  return success();
}

static void printParamConstantOpValue(OpAsmPrinter &p, Operation *,
                                      Attribute value, Type type) {
  printColonTypeOrIndex(p.getStream(), type);
  p << " = <";
  printParamValue(p, value);
  p << ">";
}

//===----------------------------------------------------------------------===//
// custom<ParamDeclareOpValue>
//===----------------------------------------------------------------------===//

static ParseResult parseParamDeclareOpValue(OpAsmParser &p,
                                            ParamDeclArrayAttr &paramDecls,
                                            TypedAttr &value) {
  std::string varname;
  Type valTy;
  if (p.parseKeywordOrString(&varname) ||
      parseParamConstantOpValue(p, value, valTy))
    return failure();

  paramDecls = p.getBuilder().getAttr<ParamDeclArrayAttr>(
      ParamDeclAttr::get(varname, valTy));
  return success();
}

static void printParamDeclareOpValue(OpAsmPrinter &p, Operation *,
                                     ParamDeclArrayAttr paramDecls,
                                     TypedAttr value) {
  ParamDeclAttr variable = paramDecls.front();
  printParamName(p, variable.getName().getValue());
  printParamConstantOpValue(p, nullptr, value, value.getType());
}

//===----------------------------------------------------------------------===//
// custom<ParameterBindings>
//===----------------------------------------------------------------------===//

static ParseResult parseParameterBindings(OpAsmParser &p,
                                          ParamBindArrayAttr &value) {
  SmallVector<ParamBindAttr> elts;
  if (p.parseCommaSeparatedList(
          OpAsmParser::Delimiter::OptionalLessGreater, [&]() -> ParseResult {
            std::string name;
            Type type;
            TypedAttr value;
            if (p.parseKeywordOrString(&name) ||
                parseColonTypeOrIndex(p, type) || p.parseEqual() ||
                parseParamValue(p, value, type))
              return failure();
            elts.push_back(ParamBindAttr::get(name, value));
            return success();
          }))
    return failure();

  value = ParamBindArrayAttr::get(p.getContext(), elts);
  return success();
}

static void printParameterBindings(OpAsmPrinter &p, Operation *op,
                                   ParamBindArrayAttr value) {
  if (value.empty())
    return;
  p << '<';
  llvm::interleaveComma(value, p, [&](ParamBindAttr bind) {
    printParamName(p, bind.getName());
    printColonTypeOrIndex(p.getStream(), bind.getType());
    p << " = ";
    printParamValue(p, bind.getValue());
  });
  p << '>';
}

//===----------------------------------------------------------------------===//
// custom<ParamAssertOpValue>
//===----------------------------------------------------------------------===//

static ParseResult parseParamAssertOpValue(OpAsmParser &p, TypedAttr &value) {
  return parseParamValue(p, value, p.getBuilder().getI1Type());
}

static void printParamAssertOpValue(OpAsmPrinter &p, Operation *,
                                    Attribute value) {
  printParamValue(p, value);
}

//===----------------------------------------------------------------------===//
// ParamDeclareOp
//===----------------------------------------------------------------------===//

void ParamDeclareOp::build(OpBuilder &builder, OperationState &result,
                           ParamDeclAttr decl, Attribute value) {
  build(builder, result, /*no result types*/ TypeRange{},
        builder.getAttr<ParamDeclArrayAttr>(decl), value);
}

ParamDeclAttr ParamDeclareOp::getParamDecl() {
  assert(getParamDecls().size() == 1 &&
         "ParamDeclareOp only allows a single parameter decl.");
  return (*getParamDecls().begin()).cast<ParamDeclAttr>();
}

//===----------------------------------------------------------------------===//
// ParamAssertOp
//===----------------------------------------------------------------------===//

LogicalResult ParamAssertOp::canonicalize(ParamAssertOp op,
                                          PatternRewriter &rewriter) {
  // If the condition is statically true then we can just remove this op.
  auto cond = op.getCond();
  if (auto intCond = cond.dyn_cast<IntegerAttr>()) {
    // Leave failing conditions, they must be diagnosed at elaboration time.
    if (intCond.getValue().isZero())
      return failure();
    rewriter.eraseOp(op);
    return success();
  }

  // Check to see if this operation only depends on expressions known in the
  // signature of the generator.  If so, we can fold it into the constraint
  // list.
  SmallVector<ParamDeclRefAttr> parameterRefs;
  KGENDeclInterface parent = op->getParentOfType<KGENDeclInterface>();
  if (parent && succeeded(collectParameterReferences(cond, parameterRefs))) {
    ArrayRef<ParamDeclAttr> generatorInputParams = getParamDecls(parent);

    // Check to see if the parameters referenced by the condition are all
    // defined by the generator.  If so, we can fold this into the constraint
    // list.
    if (llvm::all_of(parameterRefs, [&](ParamDeclRefAttr declRef) -> bool {
          return llvm::any_of(generatorInputParams, [&](ParamDeclAttr decl) {
            return decl.getName() == declRef.getName();
          });
        })) {
      // Ok, great, add this to the trait list of the enclosing operation.
      SmallVector<ConstraintAttr> constraints(parent.getConstraints());
      constraints.push_back(
          ConstraintAttr::get(cond, op.getMessageAttr(), op.getLoc()));
      parent.setConstraintsAttr(
          rewriter.getAttr<ConstraintArrayAttr>(constraints));
      op.erase();
      return success();
    }
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// CallParamOp / custom<CallParamCallee>
//===----------------------------------------------------------------------===//

static ParseResult parseCallParamCallee(OpAsmParser &p, TypedAttr &value,
                                        SmallVectorImpl<Type> &operandTypes,
                                        SmallVectorImpl<Type> &resultTypes) {
  Type type;
  auto loc = p.getCurrentLocation();
  if (parseKGENType(p, type) || p.parseColon() ||
      parseParamValue(p, value, type))
    return failure();

  auto regionType = value.getType().dyn_cast<RegionType>();
  if (!regionType)
    return p.emitError(loc, "callee parameter type must be a region type");

  llvm::append_range(operandTypes, regionType.getValues().getInputs());
  llvm::append_range(resultTypes, regionType.getValues().getResults());
  return success();
}

static void printCallParamCallee(OpAsmPrinter &p, Operation *, TypedAttr value,
                                 OperandRange::type_range operandTypes,
                                 mlir::ResultRange::type_range resultTypes) {
  printKGENType(p.getStream(), value.getType());
  p << ": ";
  printParamValue(p, value);
}

LogicalResult CallParamOp::canonicalize(CallParamOp op,
                                        PatternRewriter &rewriter) {
  // If the condition is a known symbol, then replace this with a kgen.call.
  if (auto calleeSymbol = op.getCallee().dyn_cast<SymbolConstantAttr>()) {
    rewriter.replaceOpWithNewOp<CallOp>(
        op, op.getResultTypes(), calleeSymbol.getSymbol().getLeafReference(),
        op.getParamValues(), op.getParamDecls(), op.getOperands());
    return success();
  }

  return failure();
}

LogicalResult CallParamOp::verify() {
  KGENDeclInterface parent =
      getOperation()->getParentOfType<KGENDeclInterface>();
  if (!parent || isa<FuncOp>(parent))
    return emitError(
        "kgen.call_param is only allowed in generators pre-elaboration");

  return success();
}

//===----------------------------------------------------------------------===//
// Logic shared between FuncOp, GeneratorOp, and CallOp
//===----------------------------------------------------------------------===//

/// Parse a parameter list if present.
template <typename AttrT>
static ParseResult
parseParamList(AsmParser &p, SmallVectorImpl<AttrT> &params,
               function_ref<ParseResult(AsmParser &, AttrT &, StringRef, Type)>
                   parseElementFn) {

  // Handle the parameter-decl/parameter-result productions.
  auto parseParamDecl = [&]() -> ParseResult {
    StringAttr name;
    Type type;

    AttrT element;
    if (parseParamName(p, name) || parseColonTypeOrIndex(p, type) ||
        parseElementFn(p, element, name, type))
      return failure();
    params.push_back(element);
    return success();
  };

  // Check to see if we have the () syntax instead of arguments.
  if (succeeded(p.parseOptionalLParen()))
    return p.parseRParen();

  // Otherwise, parse the parameters, we know there is at least one.
  return p.parseCommaSeparatedList(OpAsmParser::Delimiter::None,
                                   parseParamDecl);
}

/// Parse a parameter declaration list if present.
///
///   parameter-decl   ::= identifier (`:` type)?
///   parameter-decl-list  ::= parameter-decl (`,` parameter-decl)* | `(` `)`
ParseResult KGEN::parseParamDecls(AsmParser &p, ParamDeclArrayAttr &result) {
  auto parseElement = [](AsmParser &p, ParamDeclAttr &attr, StringRef name,
                         Type type) -> ParseResult {
    attr = ParamDeclAttr::get(name, type);
    return success();
  };

  // Parse each of the decls.
  SmallVector<ParamDeclAttr> decls;
  if (parseParamList<ParamDeclAttr>(p, decls, parseElement))
    return failure();

  result = ParamDeclArrayAttr::get(p.getContext(), decls);
  return success();
}

/// Print a comma separated parameter declaration list.
void KGEN::printParamDecls(raw_ostream &os, ParamDeclArrayAttr decls) {
  if (decls.empty()) {
    os << "()";
  } else {
    llvm::interleaveComma(decls, os, [&](ParamDeclAttr ref) {
      printParamName(ref.getName().getValue(), os);
      printColonTypeOrIndex(os, ref.getType());
    });
  }
}

/// Parse a parameter binding list if present.
///
///   parameter-bind   ::= identifier (`:` type)? `=` attribute-value
///   parameter-bind-list ::= parameter-bind (`,` parameter-bind)* | `(` `)`
static ParseResult parseParamBinds(AsmParser &p,
                                   SmallVectorImpl<ParamBindAttr> &paramBinds) {
  auto parseElement = [](AsmParser &p, ParamBindAttr &attr, StringRef name,
                         Type type) -> ParseResult {
    TypedAttr value;
    if (p.parseEqual() || parseParamValue(p, value, type))
      return failure();
    attr = ParamBindAttr::get(name, value);
    return success();
  };
  return parseParamList<ParamBindAttr>(p, paramBinds, parseElement);
}

//===----------------------------------------------------------------------===//
// custom<CallOpParams>
//===----------------------------------------------------------------------===//

/// Parse the parameter spec for a call op.
/// parameter-decl   ::= identifier (`:` type)?
/// parameter-bind   ::= identifier (`:` type)? `=` attribute-value

/// param-decl-list  ::= parameter-decl (`,` parameter-decl)* | `(` `)`
/// param-bind-list  ::= parameter-bind (`,` parameter-bind)* | `(` `)`

/// parameter-spec   ::= `<` param-bind-list (`->` param-decl-list)? `>`
static ParseResult parseCallOpParams(OpAsmParser &p,
                                     ParamBindArrayAttr &paramValues,
                                     ParamDeclArrayAttr &paramDecls) {

  if (p.parseOptionalLess()) {
    // If there is no <, then the params of the call op are empty, so set
    // paramValues and paramDecls to empty and return.
    paramValues = ParamBindArrayAttr::get(p.getContext(), {});
    paramDecls = ParamDeclArrayAttr::get(p.getContext(), {});
    return success();
  }

  SmallVector<ParamBindAttr> vals;
  // Parse the input list
  if (parseParamBinds(p, vals))
    return failure();
  paramValues = ParamBindArrayAttr::get(p.getContext(), vals);

  // Check to see if we have results and parse them if so.
  if (succeeded(p.parseOptionalArrow())) {
    if (parseParamDecls(p, paramDecls))
      return failure();
  } else {
    // paramDecls is empty if there is no arrow.
    paramDecls = ParamDeclArrayAttr::get(p.getContext(), {});
  }

  return p.parseGreater();
}

static void printCallOpParams(OpAsmPrinter &p, Operation *op,
                              ParamBindArrayAttr paramValues,
                              ParamDeclArrayAttr paramDecls) {
  if (paramValues.empty() && paramDecls.empty())
    return;
  p << "<";
  if (paramValues.empty())
    p << "()";
  else {
    llvm::interleaveComma(paramValues, p, [&](ParamBindAttr bind) {
      printParamName(p, bind.getName().getValue());
      printColonTypeOrIndex(p.getStream(), bind.getType());
      p << " = ";
      printParamValue(p, bind.getValue());
    });
  }

  if (!paramDecls.empty()) {
    p << " -> ";
    printParamDecls(p.getStream(), paramDecls);
  }
  p << ">";
}

//===----------------------------------------------------------------------===//
// Logic shared between funcs, generators, and generator interfaces
//===----------------------------------------------------------------------===//

/// Parse an parameter list if present.
/// parameter-decl   ::= identifier (`:` type)?
/// parameter-list   ::= parameter-decl (`,` parameter-decl)* | `(` `)`
/// parameter-spec   ::= `<` parameter-list (`->` parameter-list)? `>`
static ParseResult parseOptionalParameterSpec(OpAsmParser &parser,
                                              OperationState &result) {
  // If there is no parameter list, or if it is empty, we're done.
  if (failed(parser.parseOptionalLess()) ||
      succeeded(parser.parseOptionalGreater())) {
    // All kinds have result parameters.
    result.addAttribute("resultParamDecls",
                        ParamDeclArrayAttr::get(parser.getContext(), {}));
    result.addAttribute("paramDecls",
                        ParamDeclArrayAttr::get(parser.getContext(), {}));
    return success();
  }

  ParamDeclArrayAttr paramDecls, resultParamDecls;

  // Parse the input list.
  if (parseParamDecls(parser, paramDecls))
    return failure();

  // Check to see if we have results and parse them if so.
  if (succeeded(parser.parseOptionalArrow())) {
    if (parseParamDecls(parser, resultParamDecls))
      return failure();
  } else {
    resultParamDecls = ParamDeclArrayAttr::get(parser.getContext(), {});
  }

  result.addAttribute("resultParamDecls", resultParamDecls);
  result.addAttribute("paramDecls", paramDecls);

  return parser.parseGreater();
}

/// Parse a constraint specification if present.
/// constraints-spec ::=
///    `constraints` `<` attribute-value (`,` attribute-value)? `>`
static ParseResult parseOptionalConstraints(OpAsmParser &parser,
                                            OperationState &result,
                                            GeneratorOrFuncKind opKind) {
  // Funcs cannot have constraint specifications.
  if (opKind == GeneratorOrFuncKind::func)
    return success();

  SmallVector<ConstraintAttr> constraints;

  if (succeeded(parser.parseOptionalKeyword("constraints"))) {
    auto parseConstraint = [&]() -> ParseResult {
      ConstraintAttr constraint;
      if (parser.parseCustomAttributeWithFallback(constraint))
        return failure();
      constraints.push_back(constraint);
      return success();
    };

    if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::LessGreater,
                                       parseConstraint))
      return failure();
  }
  result.addAttribute("constraints", ConstraintArrayAttr::get(
                                         parser.getContext(), constraints));
  return success();
}

/// Parse either a kgen.generator or kgen.func declaration, depending on what
/// `isGenerator` is set to.
ParseResult KGEN::parseGeneratorOrFunc(OpAsmParser &parser,
                                       OperationState &result,
                                       GeneratorOrFuncKind opKind) {
  using namespace mlir::function_interface_impl;

  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse visibility. If none is provided, use private by default.
  if (failed(mlir::impl::parseOptionalVisibilityKeyword(parser,
                                                        result.attributes)))
    result.addAttribute(SymbolTable::getVisibilityAttrName(),
                        parser.getBuilder().getStringAttr("private"));

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  bool isVariadic = false;
  if (parseOptionalParameterSpec(parser, result) ||
      parseFunctionSignature(parser, /*allowVariadic=*/false, entryArgs,
                             isVariadic, resultTypes, resultAttrs) ||
      parseOptionalConstraints(parser, result, opKind))
    return failure();

  SmallVector<Type> argTypes;
  argTypes.reserve(entryArgs.size());
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);
  Type type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  // If function attributes are present, parse them.
  NamedAttrList parsedAttributes;
  llvm::SMLoc attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();

  // If this is a generator, see if it is an implementation of a generator
  // interface.
  if ((opKind == GeneratorOrFuncKind::generator ||
       opKind == GeneratorOrFuncKind::hlgenerator) &&
      succeeded(parser.parseOptionalKeyword("implements"))) {
    ::mlir::FlatSymbolRefAttr implementsAttr;
    if (parser.parseAttribute(implementsAttr,
                              parser.getBuilder().getType<::mlir::NoneType>(),
                              "implements", result.attributes))
      return failure();
  }

  // Disallow attributes that are inferred from elsewhere in the attribute
  // dictionary.
  for (StringRef disallowed : GeneratorOp::getAttributeNames()) {
    if (parsedAttributes.get(disallowed))
      return parser.emitError(attributeDictLocation, "'")
             << disallowed
             << "' is an inferred attribute and should not be specified in the "
                "explicit attribute dictionary";
  }
  result.attributes.append(parsedAttributes);

  // Add the attributes to the function arguments.
  assert(resultAttrs.size() == resultTypes.size());
  addArgAndResultAttrs(builder, result, entryArgs, resultAttrs);

  // Parse the required function body.
  auto *body = result.addRegion();

  // If this is a generator interface, no body block is allowed.
  if (opKind == GeneratorOrFuncKind::interface)
    return success();

  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseRegion(*body, entryArgs,
                         /*enableNameShadowing=*/false))
    return failure();

  // Function body was parsed, make sure its not empty.
  if (body->empty())
    return parser.emitError(loc, "expected non-empty function body");

  return success();
}

/// Print a parameter list for a generator, func or interface.
static void printParameterList(KGENDeclInterface decl, OpAsmPrinter &p) {
  auto inputParams = decl.getParamDeclsAttr();
  auto resultParams = decl.getResultParamDeclsAttr();
  if (inputParams.empty() && resultParams.empty())
    return;

  p << '<';
  printParamDecls(p.getStream(), inputParams);

  if (!resultParams.empty()) {
    p << " -> ";
    printParamDecls(p.getStream(), resultParams);
  }
  p << '>';
}

/// Print a constraint list for a generator or interface.
static void printConstraints(KGENDeclInterface decl, OpAsmPrinter &p) {
  ArrayRef<ConstraintAttr> constraints = decl.getConstraints();
  if (constraints.empty())
    return;

  p.printNewline();
  p << "  constraints <";
  llvm::interleaveComma(constraints, p, [&](ConstraintAttr constraint) {
    if (constraints.size() > 1) {
      p.printNewline();
      p << "    ";
    }
    constraint.print(p);
  });
  p << ">";
}

void KGEN::printGeneratorOrFunc(OpAsmPrinter &p, mlir::FunctionOpInterface op) {
  using namespace mlir::function_interface_impl;

  // TODO: KGENDeclInterface should inherit from FunctionOpInterface.
  auto opDecl = cast<KGENDeclInterface>((Operation *)op);

  // Print the operation and the function name.
  auto funcName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  p << ' ';

  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility = op->getAttrOfType<StringAttr>(visibilityAttrName))
    if (visibility.getValue() != "private")
      p << visibility.getValue() << ' ';
  p.printSymbolName(funcName);
  printParameterList(opDecl, p);

  ArrayRef<Type> argTypes = op.getArgumentTypes();
  ArrayRef<Type> resultTypes = op.getResultTypes();
  printFunctionSignature(p, op, argTypes, /*isVariadic=*/false, resultTypes);
  printFunctionAttributes(p, op, argTypes.size(), resultTypes.size(),
                          GeneratorOp::getAttributeNames());
  printConstraints(opDecl, p);

  // If this is a generator implementing a generator.interface, include the
  // symbol for the generator interface.
  if (auto implementsAttr =
          op->getAttrOfType<FlatSymbolRefAttr>("implements")) {
    p.printNewline();
    p << "  implements " << implementsAttr;
  }

  p << ' ';
  if (!op.getBody().empty()) {
    p.printRegion(op.getBody(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

/// Verify that a list of parameter declarations from a generator or func
/// matches those of an interface.  This produces an error diagnostic and
/// returns failure when a problem is detected, or returns true if everything is
/// ok.
ParseResult KGEN::verifyParameterList(
    ArrayRef<ParamDeclAttr> originatorParamDecls,
    ArrayRef<ParamDeclAttr> interfaceParamDecls, const char *originatorName,
    mlir::FunctionOpInterface originatorDecl, const char *interfaceName,
    GeneratorInterfaceOp interfaceDecl, const char *parameterKind) {

  auto getParamDeclName = [](ArrayRef<ParamDeclAttr> decls) {
    return llvm::map_range(decls, [](Attribute value) -> StringAttr {
      return value.cast<ParamDeclAttr>().getName();
    });
  };
  auto getParamDeclType = [](ArrayRef<ParamDeclAttr> decls) {
    return llvm::map_range(decls, [](Attribute value) -> Type {
      return value.cast<ParamDeclAttr>().getType();
    });
  };

  if (verifyMatchingLists(getParamDeclName(originatorParamDecls),
                          getParamDeclName(interfaceParamDecls), originatorName,
                          originatorDecl, interfaceName, interfaceDecl,
                          parameterKind, "name") ||
      verifyMatchingLists(getParamDeclType(originatorParamDecls),
                          getParamDeclType(interfaceParamDecls), originatorName,
                          originatorDecl, interfaceName, interfaceDecl,
                          parameterKind, "type"))
    return failure();

  return success();
}

/// Check that the specified generator/interfaces matches signature information
/// with the other interface.
LogicalResult KGEN::verifyDeclMatchesInterface(
    const char *originatorName, mlir::FunctionOpInterface originatorDecl,
    const char *interfaceName, GeneratorInterfaceOp interfaceDecl) {

  auto [originatorInputParamDecls, originatorResultParamDecls] =
      // TODO: KGENDeclInterface should inherit from FunctionOpInterface.
      cast<KGENDeclInterface>((Operation *)originatorDecl).getParameterInfo();
  auto [interfaceInputParamDecls, interfaceResultParamDecls] =
      interfaceDecl.getParameterInfo();

  if (verifyMatchingLists(originatorDecl.getArgumentTypes(),
                          interfaceDecl.getArgumentTypes(), originatorName,
                          originatorDecl, interfaceName, interfaceDecl,
                          "argument", "type") ||
      verifyMatchingLists(originatorDecl.getResultTypes(),
                          interfaceDecl.getResultTypes(), originatorName,
                          originatorDecl, interfaceName, interfaceDecl,
                          "result", "type") ||
      verifyParameterList(originatorInputParamDecls, interfaceInputParamDecls,
                          originatorName, originatorDecl, interfaceName,
                          interfaceDecl, "input parameter") ||
      verifyParameterList(originatorResultParamDecls, interfaceResultParamDecls,
                          originatorName, originatorDecl, interfaceName,
                          interfaceDecl, "result parameter"))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// GeneratorOp
//===----------------------------------------------------------------------===//

std::pair<ArrayRef<ParamDeclAttr>, ArrayRef<ParamDeclAttr>>
GeneratorOp::getParameterInfo() {
  return {getParamDecls(), getResultParamDecls()};
}

ReturnOp GeneratorOp::getReturnOp() {
  return cast<ReturnOp>(getBodyBlock()->getTerminator());
}

/// Parses a KGEN Generator.
ParseResult GeneratorOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseGeneratorOrFunc(parser, result, GeneratorOrFuncKind::generator);
}

// Print the GeneratorOp using the shared printing logic.
void GeneratorOp::print(OpAsmPrinter &p) { printGeneratorOrFunc(p, *this); }

LogicalResult
GeneratorOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (failed(getReturnOp().checkArgumentTypes(getResultParamDecls(),
                                              getResultTypes())))
    return failure();

  // See if the parameter definitions and uses within the generator are
  // structured correctly.
  if (failed(ParameterDeclsAndUses::calculateAndVerify(*this, symbolTable)))
    return failure();

  // If the generator is implementing a generator interface, check that they
  // line up correctly.
  FlatSymbolRefAttr interfaceSym = getImplementsAttr();
  if (!interfaceSym)
    return success();

  // Check that the callee attribute was specified.
  GeneratorInterfaceOp interface = dyn_cast_or_null<GeneratorInterfaceOp>(
      symbolTable.lookupNearestSymbolFrom(*this, interfaceSym));
  if (!interface)
    return emitError() << "'" << interfaceSym
                       << "' does not reference a generator interface";

  // Verify that the signature of this generator matches the signature of the
  // interface.
  return verifyDeclMatchesInterface("generator", *this, "interface", interface);
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

/// Create a func with no body block.  The caller must create it and fill
/// it in.
void FuncOp::build(OpBuilder &builder, OperationState &result, StringAttr name,
                   StringAttr visibility, FunctionType signature,
                   ArrayRef<ParamDeclAttr> outputParams) {
  // Add an attribute for the name and function_type attributes.
  result.addAttribute(SymbolTable::getSymbolAttrName(), name);
  result.addAttribute(SymbolTable::getVisibilityAttrName(), visibility);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(signature));
  result.addAttribute("paramDecls", builder.getAttr<ParamDeclArrayAttr>(
                                        ArrayRef<ParamDeclAttr>()));
  result.addAttribute("resultParamDecls",
                      builder.getAttr<ParamDeclArrayAttr>(outputParams));
  result.addRegion();
}

/// Create a func with an empty body, `argLocs` specifies the locations for
/// all the block arguments.
void FuncOp::build(OpBuilder &builder, OperationState &result, StringAttr name,
                   StringAttr visibility, FunctionType signature,
                   ArrayRef<ParamDeclAttr> outputParams,
                   ArrayRef<Location> argLocs) {
  build(builder, result, name, visibility, signature, outputParams);

  // Create a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);

  // Add arguments to the body block.
  assert(signature.getInputs().size() == argLocs.size() &&
         "incorrect number of arg locs");
  body->addArguments(signature.getInputs(), argLocs);
}

std::pair<ArrayRef<ParamDeclAttr>, ArrayRef<ParamDeclAttr>>
FuncOp::getParameterInfo() {
  return {{}, getResultParamDecls()};
}

ReturnOp FuncOp::getReturnOp() {
  return cast<ReturnOp>(getBodyBlock()->getTerminator());
}

/// Parses a concrete KGEN func.
///
/// operation ::=
///   `kgen.func` function-signature function-attributes? function-body
///
ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseGeneratorOrFunc(parser, result, GeneratorOrFuncKind::func);
}

/// Print the FuncOp. We use a shared printer with the GeneratorOp since it is
/// a superset of what a func is.
void FuncOp::print(OpAsmPrinter &p) { printGeneratorOrFunc(p, *this); }

LogicalResult FuncOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (failed(getReturnOp().checkArgumentTypes(getResultParamDecls(),
                                              getResultTypes())))
    return failure();

  // kgen.func's are not allowed to have input parameter lists.
  if (!getParamDecls().empty())
    return emitError(
        "kgen.func only allows output parameters, not input parameters");

  // See if the parameter definitions and uses within the func are
  // structured correctly.
  FailureOr<ParameterDeclsAndUses> paramInfo =
      ParameterDeclsAndUses::calculateAndVerify(*this, symbolTable);
  if (failed(paramInfo))
    return failure();

  // In a kgen.func, parameters are allowed to be defined (e.g. by calls with
  // output parameters), but not used.  This is because the elaborator must
  // already have been run, lowering these to concrete attibute values.

  // TODO: In the future, we could ban specific uses (e.g. in types) but have an
  // allow-list of operations that can use parameters.  This could be useful if
  // we want something to be able to use the result parameters of a call or
  // something.  Until then, a blanket ban on parameter use is sufficient.
  for (auto &[usingOp, uses] : paramInfo.value().usersAndDeclarers) {
    if (!uses.empty()) {
      auto diag = usingOp->emitError("invalid use of parameter ")
                  << uses[0].getName() << " in kgen.func";
      diag.attachNote(this->getLoc())
          << "within kgen.func '" << getName() << "'";

      return failure();
    }
  }
  return success();
}

// FuncOp doesn't allow constraints, because it doesn't have any input
// parameters.
ArrayRef<ConstraintAttr> FuncOp::getConstraints() { return {}; }
ConstraintArrayAttr FuncOp::getConstraintsAttr() {
  return ConstraintArrayAttr();
}

void FuncOp::setConstraintsAttr(ConstraintArrayAttr attr) {
  assert(0 && "kgen.func doesn't have input parameters to add constraints to");
}

//===----------------------------------------------------------------------===//
// GeneratorInterfaceOp
//===----------------------------------------------------------------------===//

std::pair<ArrayRef<ParamDeclAttr>, ArrayRef<ParamDeclAttr>>
GeneratorInterfaceOp::getParameterInfo() {
  return {getParamDecls(), getResultParamDecls()};
}

/// Parses a KGEN generator interface.
ParseResult GeneratorInterfaceOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  return parseGeneratorOrFunc(parser, result, GeneratorOrFuncKind::interface);
}

// Print the GeneratorInterfaceOp using the shared printing logic.
void GeneratorInterfaceOp::print(OpAsmPrinter &p) {
  printGeneratorOrFunc(p, *this);
}

LogicalResult
GeneratorInterfaceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // See if the parameter definitions and uses within the generator are
  // structured correctly.  These are only defined in the interface and used
  // in the argument list or constraints list.
  return ParameterDeclsAndUses::calculateAndVerify(*this, symbolTable);
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

template <typename CallerRange, typename CalleeRange>
static ParseResult verifyMatchingCallLists(const CallerRange &callerRange,
                                           const CalleeRange &calleeRange,
                                           Operation *caller, Operation *callee,
                                           const char *itemName,
                                           const char *propertyName) {
  return verifyMatchingLists(callerRange, calleeRange, "caller", caller,
                             "callee", callee, itemName, propertyName);
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto calleeAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!calleeAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  auto callee = dyn_cast_or_null<KGENDeclInterface>(
      symbolTable.lookupNearestSymbolFrom(*this, calleeAttr));
  if (!callee)
    return emitError() << "'" << calleeAttr.getValue()
                       << "' does not reference a valid callee";

  // Verify that the callee/caller parameters match.  The parameter names on the
  // results don't need to match, but the parameter names on the argument
  // bindings do.  The types always need to match.
  auto [calleeInputParamDecls, calleeOutputParamDecls] =
      callee.getParameterInfo();

  // Check the parameter values specified to the input parameters.
  ArrayRef<ParamBindAttr> callerInputParams = getParamValues();
  ArrayRef<ParamDeclAttr> callerOutputParamDecls = getParamDecls();

  auto getParamDeclType = [](ArrayRef<ParamDeclAttr> decls) {
    return llvm::map_range(
        decls, [](ParamDeclAttr value) -> Type { return value.getType(); });
  };

  /// Check the input parameter names.  We don't check the result parameter
  /// names because (in general) they are intentionally renamed at the call
  /// site.
  if (verifyMatchingCallLists(
          llvm::map_range(callerInputParams,
                          [](Attribute value) -> Attribute {
                            return value.cast<ParamBindAttr>().getName();
                          }),
          llvm::map_range(calleeInputParamDecls,
                          [](Attribute value) -> Attribute {
                            return value.cast<ParamDeclAttr>().getName();
                          }),
          *this, callee, "input parameter", "name"))
    return failure();

  // We need to substitute and simplify expressions that occur in the argument
  // list and parameter types, e.g.:
  //     kgen.generator @callee1<type: dtype>(%x: !meta.scalar<type>)
  //     kgen.generator @callee2<size>(%x: !meta.simd<size, f32>)
  // ... call @callee1<type: dtype = f32>(%arg1) : (!meta.scalar<f32>) -> ()
  // ... call @callee2<size=4>(%arg2) : (!meta.simd<4, f32>) -> ()
  //
  // This can also occur in parameter types, e.g. for region types (dt vs f32):
  //     kgen.generator @g<dt: dtype, region: () -> !meta.scalar<dt>>(...
  //     call @g<dt: f32, region: () -> !meta.scalar<f32>(...

  // We do this with with ParameterEvaluator which can do the remapping for us.
  ParameterEvaluator evaluator;
  for (auto [value, decl] :
       llvm::zip(callerInputParams, calleeInputParamDecls)) {
    evaluator.setParameterValue(decl, value.getValue());
  }
  auto remapType = [&](Type type) -> Type {
    return evaluator.getReboundType(type);
  };

  // Check input parameter types match.
  if (verifyMatchingCallLists(
          llvm::map_range(callerInputParams,
                          [](Attribute value) -> Type {
                            return value.cast<ParamBindAttr>().getType();
                          }),
          llvm::map_range(getParamDeclType(calleeInputParamDecls), remapType),
          *this, callee, "input parameter", "type") ||

      /// Check result parameter types.
      verifyMatchingCallLists(getParamDeclType(callerOutputParamDecls),
                              getParamDeclType(calleeOutputParamDecls), *this,
                              callee, "output parameter", "type")) {
    return failure();
  }

  // Ok, now that we know the parameters match up, verify that the operand
  // and result types match the callee.
  auto fnType = callee.getFunctionType();
  auto calleeInputTypes = llvm::map_range(fnType.getInputs(), remapType);
  auto calleeResultTypes = llvm::map_range(fnType.getResults(), remapType);

  // Check that the passed in operands, and returned types match our
  // expectations.
  if (verifyMatchingCallLists(getOperandTypes(), calleeInputTypes, *this,
                              callee, "input", "type") ||
      verifyMatchingCallLists(getResultTypes(), calleeResultTypes, *this,
                              callee, "result", "type"))
    return failure();

  // Ok, the call looks good.  Next, make sure that calls within a
  // kgen.func do not pass input arguments.  Input arguments are invalid to a
  // func, so it must be a generator or generator interface, and these will
  // not be elaborated unless they have zero input arguments.
  if (!callerInputParams.empty()) {
    if (auto funcParent = getOperation()->getParentOfType<FuncOp>()) {
      auto diag = emitError() << "cannot call generator with input arguments "
                                 "from concrete kgen.func";
      diag.attachNote(funcParent->getLoc())
          << "within kgen.func '" << funcParent.getName() << "'";
      return failure();
    }
  }

  return success();
}

void CallOp::build(OpBuilder &builder, OperationState &state,
                   TypeRange resultTypes, StringAttr callee,
                   ArrayRef<ParamBindAttr> inputParams,
                   ArrayRef<ParamDeclAttr> resultParams,
                   OperandRange operands) {
  build(builder, state, resultTypes, FlatSymbolRefAttr::get(callee),
        builder.getAttr<ParamBindArrayAttr>(inputParams),
        builder.getAttr<ParamDeclArrayAttr>(resultParams), operands);
}

//===----------------------------------------------------------------------===//
// ParamConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ParamConstantOp::fold(ArrayRef<Attribute> constants) {
  assert(constants.empty() && "kgen.param.constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

/// Containers verify that the operands of this ReturnOp match the specified set
/// of types.
LogicalResult ReturnOp::checkArgumentTypes(ArrayRef<ParamDeclAttr> paramDecls,
                                           TypeRange types) {
  // Check the parameters match up.
  auto returnedParams = getParameters();
  if (returnedParams.size() != paramDecls.size())
    return emitOpError("expected ")
           << paramDecls.size() << " parameters for enclosing op";

  for (size_t i = 0, e = returnedParams.size(); i != e; ++i) {
    auto returned = returnedParams[i].cast<ParamBindAttr>();
    auto decl = paramDecls[i];
    if (returned.getName() != decl.getName())
      return emitOpError("parameter #")
             << i << " is named " << returned.getName() << " but should be "
             << decl.getName();
    if (returned.getType() != decl.getType())
      return emitOpError("parameter #") << i << " has type " << returned
                                        << " but should be " << decl.getType();
  }

  // Verify our result types match up with the enclosing result type.
  if (getNumOperands() != types.size())
    return emitOpError("expected ")
           << types.size() << " operands for enclosing op";

  for (size_t i = 0, e = getNumOperands(); i != e; ++i) {
    if (getOperand(i).getType() != types[i])
      return emitOpError("operand #")
             << i << " has type " << getOperand(i).getType()
             << " but should be " << types[i];
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "KGEN/KGENDialect/KGEN.cpp.inc"

// Generated interface definitions.
#include "KGEN/KGENDialect/ElaboratorOpInterface.cpp.inc"
#include "KGEN/KGENDialect/KGENDeclInterface.cpp.inc"
