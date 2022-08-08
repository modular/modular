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
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// custom<ParamValueOpValue>
//===----------------------------------------------------------------------===//

static ParseResult parseParamValueOpValue(OpAsmParser &p, TypedAttr &value,
                                          Type &resultType) {
  if (parseColonTypeOrIndex(p, resultType) || p.parseEqual() || p.parseLess() ||
      parseParamValue(p, value, resultType) || p.parseGreater())
    return failure();
  return success();
}

static void printParamValueOpValue(OpAsmPrinter &p, Operation *,
                                   Attribute value, Type type) {
  printColonTypeOrIndex(p, type);
  p << " = <";
  printParamValue(p, value);
  p << ">";
}

//===----------------------------------------------------------------------===//
// custom<ParamBindOpValue>
//===----------------------------------------------------------------------===//

static ParseResult parseParamBindOpValue(OpAsmParser &p,
                                         ParamDeclArrayAttr &paramDecls,
                                         TypedAttr &value) {
  std::string varname;
  Type valTy;
  if (p.parseKeywordOrString(&varname) ||
      parseParamValueOpValue(p, value, valTy))
    return failure();

  paramDecls = p.getBuilder().getAttr<ParamDeclArrayAttr>(
      ParamDeclAttr::get(varname, valTy));
  return success();
}

static void printParamBindOpValue(OpAsmPrinter &p, Operation *,
                                  ParamDeclArrayAttr paramDecls,
                                  TypedAttr value) {
  ParamDeclAttr variable = paramDecls.front();
  printParamName(p, variable.getName().getValue());
  printParamValueOpValue(p, nullptr, value, value.getType());
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
            elts.push_back(ParamBindAttr::get(name, type, value));
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
    printColonTypeOrIndex(p, bind.getType());
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
// ParamBindOp
//===----------------------------------------------------------------------===//

void ParamBindOp::build(OpBuilder &builder, OperationState &result,
                        ParamDeclAttr decl, Attribute value) {
  build(builder, result, /*no result types*/ TypeRange{},
        builder.getAttr<ParamDeclArrayAttr>(decl), value);
}

ParamDeclAttr ParamBindOp::getParamDecl() {
  assert(getParamDecls().size() == 1 &&
         "ParamBindOp only allows a single parameter decl.");
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
  if (GeneratorOp parent = op->getParentOfType<GeneratorOp>();
      succeeded(ParameterEvaluator::collectParameterReferences(
          cond, parameterRefs))) {
    ArrayRef<ParamDeclAttr> generatorInputParams =
        getDeclParameterInfo(parent).first;

    // Check to see if the parameters referenced by the condition are all
    // defined by the generator.  If so, we can fold this into the constraint
    // list.
    if (llvm::all_of(parameterRefs, [&](ParamDeclRefAttr declRef) -> bool {
          return llvm::any_of(generatorInputParams, [&](ParamDeclAttr decl) {
            return decl.getName() == declRef.getName();
          });
        })) {
      // Ok, great, add this to the trait list of the enclosing operation.
      auto oldConstraints = parent.getConstraints().getValue();
      SmallVector<Attribute> constraints(oldConstraints.begin(),
                                         oldConstraints.end());
      auto oldMessages = parent.getConstraintMessages().getValue();
      SmallVector<Attribute> constraintMessages(oldMessages.begin(),
                                                oldMessages.end());
      constraints.push_back(cond);
      constraintMessages.push_back(op.getMessageAttr());
      parent.setConstraintsAttr(rewriter.getArrayAttr(constraints));
      parent.setConstraintMessagesAttr(
          rewriter.getArrayAttr(constraintMessages));
      op.erase();
      return success();
    }
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// Logic shared between KernelOp, GeneratorOp, and CallOp
//===----------------------------------------------------------------------===//

/// Parse a parameter list if present.
template <typename AttrT>
static ParseResult
parseParamList(AsmParser &p, SmallVectorImpl<AttrT> &params,
               function_ref<ParseResult(AsmParser &, AttrT &, StringRef, Type)>
                   parseElementFn) {

  // Handle the parameter-decl/parameter-result productions.
  auto parseParamDecl = [&]() -> ParseResult {
    std::string name;
    Type type;

    AttrT element;
    if (p.parseKeywordOrString(&name) || parseColonTypeOrIndex(p, type) ||
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
static ParseResult parseParamDecls(AsmParser &p,
                                   SmallVectorImpl<ParamDeclAttr> &paramDecls) {
  auto parseElement = [](AsmParser &p, ParamDeclAttr &attr, StringRef name,
                         Type type) -> ParseResult {
    attr = ParamDeclAttr::get(name, type);
    return success();
  };
  return parseParamList<ParamDeclAttr>(p, paramDecls, parseElement);
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
    attr = ParamBindAttr::get(name, type, value);
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

  // Check to see if we have results and parse them if so.
  // paramDecls will be empty if there is no arrow.
  SmallVector<ParamDeclAttr> decls;
  if (succeeded(p.parseOptionalArrow())) {
    if (parseParamDecls(p, decls))
      return failure();
  }

  paramValues = ParamBindArrayAttr::get(p.getContext(), vals);
  paramDecls = ParamDeclArrayAttr::get(p.getContext(), decls);

  return p.parseGreater();
}

static void printCallOpParams(OpAsmPrinter &p, Operation *op,
                              ParamBindArrayAttr paramValues,
                              ParamDeclArrayAttr paramDecls) {
  if (paramValues.empty() && paramDecls.empty())
    return;
  p << "<";
  llvm::interleaveComma(paramValues, p, [&](ParamBindAttr bind) {
    printParamName(p, bind.getName().getValue());
    printColonTypeOrIndex(p, bind.getType());
    p << " = ";
    printParamValue(p, bind.getValue());
  });
  if (paramValues.empty())
    p << "()";

  if (!paramDecls.empty()) {
    p << " -> ";
    llvm::interleaveComma(paramDecls, p, [&](ParamDeclAttr ref) {
      printParamName(p, ref.getName().getValue());
      printColonTypeOrIndex(p, ref.getType());
    });
  }
  p << ">";
}

//===----------------------------------------------------------------------===//
// Logic shared between kernels, generators, and generator interfaces
//===----------------------------------------------------------------------===//

/// Parse an parameter list if present.
/// parameter-decl   ::= identifier (`:` type)?
/// parameter-list   ::= parameter-decl (`,` parameter-decl)* | `(` `)`
/// parameter-spec   ::= `<` parameter-list (`->` parameter-list)? `>`
static ParseResult parseOptionalParameterSpec(OpAsmParser &parser,
                                              OperationState &result,
                                              GeneratorOrKernelKind opKind) {
  // If there is no parameter list, or if it is empty, we're done.
  if (failed(parser.parseOptionalLess()) ||
      succeeded(parser.parseOptionalGreater())) {
    result.addAttribute("paramDecls",
                        ParamDeclArrayAttr::get(parser.getContext(), {}));
    if (opKind != GeneratorOrKernelKind::kernel)
      result.addAttribute("numInputParameters",
                          parser.getBuilder().getI32IntegerAttr(0));
    return success();
  }

  SmallVector<ParamDeclAttr> paramDecls;

  // Parse the input list.
  auto loc = parser.getCurrentLocation();
  if (parseParamDecls(parser, paramDecls))
    return failure();

  unsigned numInputs = paramDecls.size();

  // Check to see if we have results and parse them if so.
  if (succeeded(parser.parseOptionalArrow())) {
    if (parseParamDecls(parser, paramDecls))
      return failure();
  }

  result.addAttribute("paramDecls",
                      ParamDeclArrayAttr::get(parser.getContext(), paramDecls));

  // kgen.kernel's are not allowed to have input parameter lists.
  if (opKind == GeneratorOrKernelKind::kernel && numInputs)
    return parser.emitError(
        loc, "kgen.kernel only allows output parameters, not input parameters");

  if (opKind != GeneratorOrKernelKind::kernel)
    result.addAttribute("numInputParameters",
                        parser.getBuilder().getI32IntegerAttr(numInputs));
  return parser.parseGreater();
}

/// Parse a constraint specification if present.
/// constraints-spec ::=
///    `constraints` `<` attribute-value (`,` attribute-value)? `>`
static ParseResult parseOptionalConstraints(OpAsmParser &parser,
                                            OperationState &result,
                                            GeneratorOrKernelKind opKind) {
  // Kernels cannot have constraint specifications.
  if (opKind == GeneratorOrKernelKind::kernel)
    return success();

  SmallVector<Attribute> constraints, constraintMessages;

  if (succeeded(parser.parseOptionalKeyword("constraints"))) {
    // Constraints are always i1.
    Type int1Ty = parser.getBuilder().getI1Type();
    std::string message;

    auto parseConstraint = [&]() -> ParseResult {
      TypedAttr constraint;
      if (parseParamValue(parser, constraint, int1Ty) || parser.parseComma() ||
          parser.parseString(&message))
        return failure();
      constraints.push_back(constraint);
      constraintMessages.push_back(parser.getBuilder().getStringAttr(message));
      return success();
    };

    if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::LessGreater,
                                       parseConstraint))
      return failure();
  }
  result.addAttribute("constraints",
                      parser.getBuilder().getArrayAttr(constraints));
  result.addAttribute("constraintMessages",
                      parser.getBuilder().getArrayAttr(constraintMessages));
  return success();
}

/// Parse either a kgen.generator or kgen.kernel declaration, depending on what
/// `isGenerator` is set to.
ParseResult KGEN::parseGeneratorOrKernel(OpAsmParser &parser,
                                         OperationState &result,
                                         GeneratorOrKernelKind opKind) {
  using namespace mlir::function_interface_impl;

  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse visibility.
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  bool isVariadic = false;
  if (parseOptionalParameterSpec(parser, result, opKind) ||
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
  if ((opKind == GeneratorOrKernelKind::generator ||
       opKind == GeneratorOrKernelKind::hlgenerator) &&
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
  if (opKind == GeneratorOrKernelKind::interface)
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

/// Print a parameter list for a generator, kernel or interface.
static void printParameterList(Operation *decl, OpAsmPrinter &p) {
  auto [inputParams, outputParams] = getDeclParameterInfo(decl);

  if (inputParams.empty() && outputParams.empty())
    return;

  auto printParamDecl = [&](Attribute param) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    printParamName(p, paramAttr.getName().getValue());
    printColonTypeOrIndex(p, paramAttr.getType());
  };

  p << '<';
  if (inputParams.empty())
    p << "()";
  else
    llvm::interleaveComma(inputParams, p, printParamDecl);

  if (!outputParams.empty()) {
    p << " -> ";
    llvm::interleaveComma(outputParams, p, printParamDecl);
  }
  p << '>';
}

/// Print a constraint list for a generator or interface.
static void printConstraints(Operation *decl, OpAsmPrinter &p) {
  auto constraints = getDeclConstraints(decl);
  if (constraints.empty())
    return;

  p << "\n  constraints <";
  llvm::interleaveComma(
      constraints, p,
      [&](std::pair<Attribute, StringAttr> constraintAndMessage) {
        if (constraints.size() > 1)
          p << "\n    ";
        printParamValue(p, constraintAndMessage.first);
        p << ", " << constraintAndMessage.second;
      });
  p << '>';
}

void KGEN::printGeneratorOrKernel(OpAsmPrinter &p,
                                  mlir::FunctionOpInterface op) {
  using namespace mlir::function_interface_impl;

  // Print the operation and the function name.
  auto funcName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  p << ' ';

  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility = op->getAttrOfType<StringAttr>(visibilityAttrName))
    p << visibility.getValue() << ' ';
  p.printSymbolName(funcName);
  printParameterList(op, p);

  ArrayRef<Type> argTypes = op.getArgumentTypes();
  ArrayRef<Type> resultTypes = op.getResultTypes();
  printFunctionSignature(p, op, argTypes, /*isVariadic=*/false, resultTypes);
  printFunctionAttributes(p, op, argTypes.size(), resultTypes.size(),
                          GeneratorOp::getAttributeNames());
  printConstraints(op, p);

  // If this is a generator implementing a generator.interface, include the
  // symbol for the generator interface.
  if (auto implementsAttr = op->getAttrOfType<FlatSymbolRefAttr>("implements"))
    p << "\n  implements " << implementsAttr;

  p << ' ';
  if (!op.getBody().empty()) {
    p.printRegion(op.getBody(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

/// Compare a range of values from an "originator" to a corresponding range of
/// values from a "target".  If the two mismatch, emit an error that tries to
/// explain the issue in a nice way.
template <typename TargetRange, typename OriginatorRange>
static ParseResult verifyMatchingLists(
    const OriginatorRange &originatorRange, const TargetRange &targetRange,
    const char *originatorName, Operation *originator, const char *targetName,
    Operation *target, const char *itemName, const char *propertyName) {
  // Check that the ranges have the same size.  If not, diagnose this.
  size_t numOriginator =
      std::distance(originatorRange.begin(), originatorRange.end());
  size_t numTarget = std::distance(targetRange.begin(), targetRange.end());
  if (numOriginator != numTarget) {
    auto diag = originator->emitOpError(originatorName)
                << " has " << numOriginator << " " << itemName
                << (numOriginator != 1 ? "s" : "") << " but " << targetName
                << " expects " << numTarget;
    diag.attachNote(target->getLoc()) << targetName << " declared here";
    return failure();
  }

  // If they have the same sizes, diagnose any mismatches between their
  // elements.

  // NOTE: llvm::zip doesn't work with LLVM mapped iterators.
  auto targetIt = targetRange.begin();
  auto originatorIt = originatorRange.begin();
  for (size_t itemNum = 0; itemNum != numTarget; ++itemNum) {
    auto targetVal = *targetIt++;
    auto originatorVal = *originatorIt++;
    if (originatorVal == targetVal)
      continue;

    auto diag = originator->emitError(originatorName)
                << ' ' << itemName << " #" << itemNum << " has " << propertyName
                << ' ' << originatorVal << " but " << targetName << " expected "
                << propertyName << ' ' << targetVal;
    diag.attachNote(target->getLoc()) << targetName << " declared here";
    return failure();
  }

  return success();
}

/// Check that the specified generator/interfaces matches signature information
/// with the other interface.
LogicalResult KGEN::verifyDeclMatchesInterface(
    const char *originatorName, mlir::FunctionOpInterface originatorDecl,
    const char *interfaceName, GeneratorInterfaceOp interfaceDecl) {

  auto [originatorInputParamDecls, originatorResultParamDecls] =
      getDeclParameterInfo(originatorDecl);
  auto [interfaceInputParamDecls, interfaceResultParamDecls] =
      getDeclParameterInfo(interfaceDecl);

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

  if (verifyMatchingLists(originatorDecl.getArgumentTypes(),
                          interfaceDecl.getArgumentTypes(), originatorName,
                          originatorDecl, interfaceName, interfaceDecl,
                          "argument", "type") ||
      verifyMatchingLists(originatorDecl.getResultTypes(),
                          interfaceDecl.getResultTypes(), originatorName,
                          originatorDecl, interfaceName, interfaceDecl,
                          "result", "type") ||
      verifyMatchingLists(getParamDeclName(originatorInputParamDecls),
                          getParamDeclName(interfaceInputParamDecls),
                          originatorName, originatorDecl, interfaceName,
                          interfaceDecl, "input parameter", "name") ||
      verifyMatchingLists(getParamDeclType(originatorInputParamDecls),
                          getParamDeclType(interfaceInputParamDecls),
                          originatorName, originatorDecl, interfaceName,
                          interfaceDecl, "input parameter", "type") ||
      verifyMatchingLists(getParamDeclName(originatorResultParamDecls),
                          getParamDeclName(interfaceResultParamDecls),
                          originatorName, originatorDecl, interfaceName,
                          interfaceDecl, "result parameter", "name") ||
      verifyMatchingLists(getParamDeclType(originatorResultParamDecls),
                          getParamDeclType(interfaceResultParamDecls),
                          originatorName, originatorDecl, interfaceName,
                          interfaceDecl, "result parameter", "type"))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// GeneratorOp
//===----------------------------------------------------------------------===//

std::pair<ArrayRef<ParamDeclAttr>, ArrayRef<ParamDeclAttr>>
GeneratorOp::getParameterInfo() {
  return getDeclParameterInfo(getOperation());
}

ReturnOp GeneratorOp::getReturnOp() {
  return cast<ReturnOp>(getBodyBlock()->getTerminator());
}

/// Parses a KGEN Generator.
ParseResult GeneratorOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseGeneratorOrKernel(parser, result,
                                GeneratorOrKernelKind::generator);
}

// Print the GeneratorOp using the shared printing logic.
void GeneratorOp::print(OpAsmPrinter &p) { printGeneratorOrKernel(p, *this); }

LogicalResult GeneratorOp::verifyRegions() {
  if (failed(getReturnOp().checkArgumentTypes(
          getParamDecls().drop_front(getNumInputParameters()),
          getResultTypes())))
    return failure();

  // See if the parameter definitions and uses within the generator are
  // structured correctly.
  return ParameterDeclsAndUses::calculate(*this);
}

LogicalResult
GeneratorOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // If the generator is implementing a generator interface, check that they
  // line up correctly.
  FlatSymbolRefAttr interfaceSym = getImplementsAttr();
  if (!interfaceSym)
    return success();

  // Check that the callee attribute was specified.
  GeneratorInterfaceOp interface = dyn_cast_or_null<GeneratorInterfaceOp>(
      symbolTable.lookupNearestSymbolFrom(*this, interfaceSym));
  if (!interface)
    return emitError() << "'" << interfaceSym.getValue()
                       << "' does not reference a generator interface";

  // Verify that the signature of this generator matches the signature of the
  // interface.
  return verifyDeclMatchesInterface("generator", *this, "interface", interface);
}

//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//

/// Create a kernel with no body block.  The caller must create it and fill
/// it in.
void KernelOp::build(OpBuilder &builder, OperationState &result,
                     StringAttr name, FunctionType signature,
                     ArrayRef<ParamDeclAttr> outputParams) {
  // Add an attribute for the name and function_type attributes.
  result.addAttribute(SymbolTable::getSymbolAttrName(), name);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(signature));
  result.addAttribute("paramDecls",
                      builder.getAttr<ParamDeclArrayAttr>(outputParams));
  result.addRegion();
}

/// Create a kernel with an empty body, `argLocs` specifies the locations for
/// all the block arguments.
void KernelOp::build(OpBuilder &builder, OperationState &result,
                     StringAttr name, FunctionType signature,
                     ArrayRef<ParamDeclAttr> outputParams,
                     ArrayRef<Location> argLocs) {
  build(builder, result, name, signature, outputParams);

  // Create a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);

  // Add arguments to the body block.
  assert(signature.getInputs().size() == argLocs.size() &&
         "incorrect number of arg locs");
  body->addArguments(signature.getInputs(), argLocs);
}

ReturnOp KernelOp::getReturnOp() {
  return cast<ReturnOp>(getBodyBlock()->getTerminator());
}

/// Parses a concrete KGEN Kernel.
///
/// operation ::=
///   `kgen.kernel` function-signature function-attributes? function-body
///
ParseResult KernelOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseGeneratorOrKernel(parser, result, GeneratorOrKernelKind::kernel);
}

/// Print the KernelOp. We use a shared printer with the GeneratorOp since it is
/// a superset of what a kernel is.
void KernelOp::print(OpAsmPrinter &p) { printGeneratorOrKernel(p, *this); }

LogicalResult KernelOp::verifyRegions() {
  if (failed(getReturnOp().checkArgumentTypes(getOutputParameters(),
                                              getResultTypes())))
    return failure();

  // See if the parameter definitions and uses within the kernel are
  // structured correctly.
  return ParameterDeclsAndUses::calculate(*this);
}

//===----------------------------------------------------------------------===//
// GeneratorInterfaceOp
//===----------------------------------------------------------------------===//

std::pair<ArrayRef<ParamDeclAttr>, ArrayRef<ParamDeclAttr>>
GeneratorInterfaceOp::getParameterInfo() {
  return getDeclParameterInfo(getOperation());
}

/// Parses a KGEN generator interface.
ParseResult GeneratorInterfaceOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  return parseGeneratorOrKernel(parser, result,
                                GeneratorOrKernelKind::interface);
}

// Print the GeneratorInterfaceOp using the shared printing logic.
void GeneratorInterfaceOp::print(OpAsmPrinter &p) {
  printGeneratorOrKernel(p, *this);
}

LogicalResult GeneratorInterfaceOp::verify() {
  // See if the parameter definitions and uses within the generator are
  // structured correctly.  These are only defined in the interface and used
  // in the argument list or constraints list.
  return ParameterDeclsAndUses::calculate(*this);
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
  Operation *callee = symbolTable.lookupNearestSymbolFrom(*this, calleeAttr);
  if (!isa_and_nonnull<GeneratorOp, KernelOp, GeneratorInterfaceOp>(callee))
    return emitError() << "'" << calleeAttr.getValue()
                       << "' does not reference a valid callee";

  // Verify that the callee/caller parameters match.  The parameter names on the
  // results don't need to match, but the parameter names on the argument
  // bindings do.  The types always need to match.
  auto [calleeInputParamDecls, calleeOutputParamDecls] =
      getDeclParameterInfo(callee);

  // Check the parameter values specified to the input parameters.
  ArrayRef<ParamBindAttr> callerInputParams = getParamValues();
  ArrayRef<ParamDeclAttr> callerOutputParamDecls = getParamDecls();

  auto getParamDeclType = [](ArrayRef<ParamDeclAttr> decls) {
    return llvm::map_range(
        decls, [](ParamDeclAttr value) -> Type { return value.getType(); });
  };

  /// Check the input parameter names.
  if (verifyMatchingCallLists(
          llvm::map_range(callerInputParams,
                          [](Attribute value) -> Attribute {
                            return value.cast<ParamBindAttr>().getName();
                          }),
          llvm::map_range(calleeInputParamDecls,
                          [](Attribute value) -> Attribute {
                            return value.cast<ParamDeclAttr>().getName();
                          }),
          *this, callee, "input parameter", "name") ||

      // Check input parameter types.
      verifyMatchingCallLists(
          llvm::map_range(callerInputParams,
                          [](Attribute value) -> Type {
                            return value.cast<ParamBindAttr>().getType();
                          }),
          getParamDeclType(calleeInputParamDecls), *this, callee,
          "input parameter", "type") ||

      /// Check result parameter types.
      verifyMatchingCallLists(getParamDeclType(callerOutputParamDecls),
                              getParamDeclType(calleeOutputParamDecls), *this,
                              callee, "output parameter", "type")) {
    return failure();
  }

  // Ok, now that we know the parameters match up, verify that the operand
  // and result types match the callee.
  auto fnType = callee->getAttrOfType<TypeAttr>("function_type")
                    .getValue()
                    .cast<FunctionType>();

  // We need to substitute and simplify expressions that occur in the argument
  // list, e.g.:
  //     kgen.generator @callee1<type: dtype>(%x: !meta.scalar<type>)
  //     kgen.generator @callee2<size>(%x: !meta.simd<size, f32>)
  // ... call @callee1<type: dtype = f32>(%arg1) : (!meta.scalar<f32>) -> ()
  // ... call @callee2<size=4>(%arg2) : (!meta.simd<4, f32>) -> ()
  //
  // We do this with with ParameterEvaluator which can do the remapping for us.
  ParameterEvaluator evaluator;
  for (auto [value, decl] :
       llvm::zip(callerInputParams, calleeInputParamDecls)) {
    evaluator.setParameterValue(decl.cast<ParamDeclAttr>(),
                                value.cast<ParamBindAttr>().getValue());
  }

  auto remapType = [&](Type type) -> Type {
    return evaluator.getReboundType(type, callee->getLoc());
  };

  auto calleeInputTypes = llvm::map_range(fnType.getInputs(), remapType);
  auto calleeResultTypes = llvm::map_range(fnType.getResults(), remapType);

  // Check that the passed in operands, and returned types match our
  // expectations.
  if (verifyMatchingCallLists(getOperandTypes(), calleeInputTypes, *this,
                              callee, "input", "type") ||
      verifyMatchingCallLists(getResultTypes(), calleeResultTypes, *this,
                              callee, "result", "type"))
    return failure();

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
// ParamValueOp
//===----------------------------------------------------------------------===//

OpFoldResult ParamValueOp::fold(ArrayRef<Attribute> constants) {
  assert(constants.empty() && "kgen.param.value has no operands");
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
