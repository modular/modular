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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// custom<ParamValueOpValue>
//===----------------------------------------------------------------------===//

static ParseResult parseParamValueOpValue(OpAsmParser &p, Attribute &value,
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
  printParamValue(p, value, type);
  p << ">";
}

//===----------------------------------------------------------------------===//
// custom<ParamBindOpValue>
//===----------------------------------------------------------------------===//

static ParseResult parseParamBindOpValue(OpAsmParser &p, Attribute &paramDecls,
                                         Attribute &value) {
  std::string varname;
  Type valTy;
  if (p.parseKeywordOrString(&varname) ||
      parseParamValueOpValue(p, value, valTy))
    return failure();

  paramDecls = p.getBuilder().getArrayAttr(ParamDeclAttr::get(varname, valTy));
  return success();
}

static void printParamBindOpValue(OpAsmPrinter &p, Operation *,
                                  ArrayAttr paramDecls, Attribute value) {
  ParamDeclAttr variable = paramDecls.begin()->cast<ParamDeclAttr>();
  printParamName(p, variable.getName().getValue());
  printParamValueOpValue(p, nullptr, value, value.getType());
}

//===----------------------------------------------------------------------===//
// custom<ParameterBindings>
//===----------------------------------------------------------------------===//

static ParseResult parseParameterBindings(OpAsmParser &p, ArrayAttr &value) {
  SmallVector<Attribute> elts;
  if (p.parseCommaSeparatedList(
          OpAsmParser::Delimiter::OptionalLessGreater, [&]() -> ParseResult {
            std::string name;
            Type type;
            Attribute value;
            if (p.parseKeywordOrString(&name) ||
                parseColonTypeOrIndex(p, type) || p.parseEqual() ||
                parseParamValue(p, value, type))
              return failure();
            elts.push_back(ParamBindAttr::get(name, type, value));
            return success();
          }))
    return failure();

  value = p.getBuilder().getArrayAttr(elts);
  return success();
}

static void printParameterBindings(OpAsmPrinter &p, Operation *op,
                                   ArrayAttr value) {
  if (value.empty())
    return;
  p << '<';
  llvm::interleaveComma(value, p, [&](Attribute attr) {
    auto bind = attr.cast<ParamBindAttr>();
    printParamName(p, bind.getName());
    printColonTypeOrIndex(p, bind.getType());
    p << " = ";
    printParamValue(p, bind.getValue(), bind.getType());
  });
  p << '>';
}

//===----------------------------------------------------------------------===//
// Logic shared between KernelOp and GeneratorOp
//===----------------------------------------------------------------------===//

/// Parse an parameter list if present.
/// parameter-decl   ::= identifier (`:` type)?
/// parameter-list   ::= parameter-decl (`,` parameter-decl)* | `(` `)`
/// parameter-spec   ::= `<` parameter-list (`->` parameter-list)? `>`
static ParseResult parseOptionalParameters(OpAsmParser &parser,
                                           OperationState &result,
                                           bool isGenerator) {
  bool hasLessThan = succeeded(parser.parseOptionalLess());

  // kgen.kernel's are not allowed to have parameter lists and don't get
  // parameter attributes.  If we see one (even an empty <>), diagnose with
  // a helpful error.
  if (hasLessThan && !isGenerator)
    return parser.emitError(parser.getCurrentLocation(),
                            "parameters not allowed in kgen.kernel, use "
                            "kgen.generator instead");

  // If there is no parameter list, or if it is empty, we're done.
  if (!hasLessThan || succeeded(parser.parseOptionalGreater())) {
    // kgen.kernel's don't get paramDecl related attributes.
    if (isGenerator) {
      result.addAttribute("paramDecls", parser.getBuilder().getArrayAttr({}));
      result.addAttribute("numInputParameters",
                          parser.getBuilder().getI32IntegerAttr(0));
    }
    return success();
  }

  SmallVector<Attribute> paramDecls;
  // Handle the parameter-decl/parameter-result productions.
  auto parseParamDecl = [&]() -> ParseResult {
    std::string name;
    Type type;
    if (parser.parseKeywordOrString(&name) ||
        parseColonTypeOrIndex(parser, type))
      return failure();
    paramDecls.push_back(ParamDeclAttr::get(name, type));
    return success();
  };

  // Handle the parameter-list production.
  auto parseParamList = [&]() -> ParseResult {
    // Check to see if we have the () syntax instead of arguments.
    if (succeeded(parser.parseOptionalLParen())) {
      return parser.parseRParen();
    }
    // Otherwise, parse the parameters, we know there is at least one.
    return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::None,
                                          parseParamDecl);
  };

  // Parse the input list.
  if (parseParamList())
    return failure();

  unsigned numInputs = paramDecls.size();

  // Check to see if we have results and parse them if so.
  if (succeeded(parser.parseOptionalArrow())) {
    if (parseParamList())
      return failure();
  }

  result.addAttribute("paramDecls",
                      parser.getBuilder().getArrayAttr(paramDecls));
  result.addAttribute("numInputParameters",
                      parser.getBuilder().getI32IntegerAttr(numInputs));
  return parser.parseGreater();
}

/// Parse either a kgen.generator or kgen.kernel declaration, depending on what
/// `isGenerator` is set to.
static ParseResult parseGeneratorOrKernel(OpAsmParser &parser,
                                          OperationState &result,
                                          bool isGenerator) {
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

  if (parseOptionalParameters(parser, result, isGenerator) ||
      // Both have a normal signature of course.
      parseFunctionSignature(parser, /*allowVariadic=*/false, entryArgs,
                             isVariadic, resultTypes, resultAttrs))
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

  // Disallow attributes that are inferred from elsewhere in the attribute
  // dictionary.
  for (StringRef disallowed :
       {SymbolTable::getVisibilityAttrName(), SymbolTable::getSymbolAttrName(),
        getTypeAttrName()}) {
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
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseRegion(*body, entryArgs,
                         /*enableNameShadowing=*/false))
    return failure();

  // Function body was parsed, make sure its not empty.
  if (body->empty())
    return parser.emitError(loc, "expected non-empty function body");

  return success();
}

/// Print a parameter list for a module or instance.
static void printParameterList(ArrayAttr parameters, unsigned numInputs,
                               OpAsmPrinter &p) {
  if (parameters.empty())
    return;

  auto printParamDecl = [&](Attribute param) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    printParamName(p, paramAttr.getName().getValue());
    printColonTypeOrIndex(p, paramAttr.getType());
  };

  p << '<';
  if (numInputs == 0) {
    p << "()";
  } else {
    llvm::interleaveComma(parameters.getValue().take_front(numInputs), p,
                          printParamDecl);
  }
  if (numInputs != parameters.size()) {
    p << " -> ";
    llvm::interleaveComma(parameters.getValue().drop_front(numInputs), p,
                          printParamDecl);
  }

  p << '>';
}

static void printGeneratorOrKernel(OpAsmPrinter &p,
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

  if (auto paramDecls = op->getAttrOfType<ArrayAttr>("paramDecls")) {
    auto numInputs = op->getAttrOfType<IntegerAttr>("numInputParameters");
    printParameterList(paramDecls, numInputs.getValue().getZExtValue(), p);
  }

  ArrayRef<Type> argTypes = op.getArgumentTypes();
  ArrayRef<Type> resultTypes = op.getResultTypes();
  printFunctionSignature(p, op, argTypes, /*isVariadic=*/false, resultTypes);
  printFunctionAttributes(
      p, op, argTypes.size(), resultTypes.size(),
      {visibilityAttrName, "paramDecls", "numInputParameters"});

  p << ' ';
  p.printRegion(op.getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// GeneratorOp
//===----------------------------------------------------------------------===//

ReturnOp GeneratorOp::getReturnOp() {
  return cast<ReturnOp>(getBodyBlock()->getTerminator());
}

/// Parses a KGEN Generator.
ParseResult GeneratorOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseGeneratorOrKernel(parser, result, /*isGenerator=*/true);
}

// Print the GeneratorOp using the shared printing logic.
void GeneratorOp::print(OpAsmPrinter &p) { printGeneratorOrKernel(p, *this); }

LogicalResult GeneratorOp::verifyRegions() {
  if (failed(getReturnOp().checkArgumentTypes(
          getParamDecls().getValue().drop_front(getNumInputParameters()),
          getResultTypes())) ||
      failed(checkParametersInOpBody(*this)))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//

ReturnOp KernelOp::getReturnOp() {
  return cast<ReturnOp>(getBodyBlock()->getTerminator());
}

/// Parses a concrete KGEN Kernel.
///
/// operation ::=
///   `kgen.kernel` function-signature function-attributes? function-body
///
ParseResult KernelOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseGeneratorOrKernel(parser, result, /*isGenerator=*/false);
}

/// Print the KernelOp. We use a shared printer with the GeneratorOp since it is
/// a superset of what a kernel is.
void KernelOp::print(OpAsmPrinter &p) { printGeneratorOrKernel(p, *this); }

LogicalResult KernelOp::verifyRegions() {
  if (failed(getReturnOp().checkArgumentTypes(/*no parameters*/ {},
                                              getResultTypes())) ||
      failed(checkParametersInOpBody(*this)))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// GenerateOp
//===----------------------------------------------------------------------===//

LogicalResult GenerateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto calleeAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!calleeAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  auto generator =
      symbolTable.lookupNearestSymbolFrom<GeneratorOp>(*this, calleeAttr);
  if (!generator)
    return emitError() << "'" << calleeAttr.getValue()
                       << "' does not reference a valid generator";

  // Verify that the operand and result types match the callee.
  auto fnType = generator.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitError("incorrect number of operands for generator");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
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
LogicalResult ReturnOp::checkArgumentTypes(ArrayRef<Attribute> paramDecls,
                                           TypeRange types) {
  // Check the parameters match up.
  auto returnedParams = getParameters();
  if (returnedParams.size() != paramDecls.size())
    return emitOpError("expected ")
           << paramDecls.size() << " parameters for enclosing op";

  for (size_t i = 0, e = returnedParams.size(); i != e; ++i) {
    auto returned = returnedParams[i].cast<ParamBindAttr>();
    auto decl = paramDecls[i].cast<ParamDeclAttr>();
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
