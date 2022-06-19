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
// Logic shared between KernelOp, GeneratorOp, and CallOp
//===----------------------------------------------------------------------===//

enum class GeneratorOrKernelKind {
  kernel,
  generator,
  interface,
};
/// Parse a parameter list if present.
/// parameter-decl   ::= identifier (`:` type)?
/// parameter-bind   ::= identifier (`:` type)? `=` attribute-value

/// if isBinding:
///   parameter-list ::= parameter-bind (`,` parameter-bind)* | `(` `)`
/// else:
///   parameter-list  ::= parameter-decl (`,` parameter-decl)* | `(` `)`
static ParseResult parseParamList(OpAsmParser &p,
                                  SmallVector<Attribute> &paramDecls,
                                  bool isBinding) {

  // Handle the parameter-decl/parameter-result productions.
  auto parseParamDecl = [&]() -> ParseResult {
    std::string name;
    Type type;

    if (p.parseKeywordOrString(&name) || parseColonTypeOrIndex(p, type))
      return failure();
    if (isBinding) {
      Attribute value;
      if (p.parseEqual() || parseParamValue(p, value, type))
        return failure();
      paramDecls.emplace_back(ParamBindAttr::get(name, type, value));
    } else {
      paramDecls.emplace_back(ParamDeclAttr::get(name, type));
    }
    return success();
  };

  // Check to see if we have the () syntax instead of arguments.
  if (succeeded(p.parseOptionalLParen()))
    return p.parseRParen();

  // Otherwise, parse the parameters, we know there is at least one.
  return p.parseCommaSeparatedList(OpAsmParser::Delimiter::None,
                                   parseParamDecl);
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
static ParseResult parseCallOpParams(OpAsmParser &p, ArrayAttr &paramValues,
                                     ArrayAttr &paramDecls) {

  if (p.parseOptionalLess()) {
    // If there is no <, then the params of the call op are empty, so set
    // paramValues and paramDecls to empty and return.
    paramValues = p.getBuilder().getArrayAttr({});
    paramDecls = p.getBuilder().getArrayAttr({});
    return success();
  }

  SmallVector<Attribute> vals;
  // Parse the input list
  if (parseParamList(p, vals, /*isBinding=*/true))
    return failure();

  // Check to see if we have results and parse them if so.
  // paramDecls will be empty if there is no arrow.
  SmallVector<Attribute> decls;
  if (succeeded(p.parseOptionalArrow())) {
    if (parseParamList(p, decls, /*isBinding=*/false))
      return failure();
  }

  paramValues = p.getBuilder().getArrayAttr(vals);
  paramDecls = p.getBuilder().getArrayAttr(decls);

  return p.parseGreater();
}

static void printCallOpParams(OpAsmPrinter &p, Operation *op,
                              ArrayAttr paramValues, ArrayAttr paramDecls) {
  if (paramValues.empty() && paramDecls.empty())
    return;
  p << "<";
  llvm::interleaveComma(paramValues, p, [&](Attribute attr) {
    auto bind = attr.cast<ParamBindAttr>();
    printParamName(p, bind.getName().getValue());
    printColonTypeOrIndex(p, bind.getType());
    p << " = ";
    printParamValue(p, bind.getValue(), bind.getType());
  });
  if (paramValues.empty())
    p << "()";

  if (!paramDecls.empty()) {
    p << " -> ";
    llvm::interleaveComma(paramDecls, p, [&](Attribute attr) {
      auto ref = attr.cast<ParamDeclAttr>();
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
  bool hasLessThan = succeeded(parser.parseOptionalLess());

  // kgen.kernel's are not allowed to have parameter lists and don't get
  // parameter attributes.  If we see one (even an empty <>), diagnose with
  // a helpful error.
  if (opKind == GeneratorOrKernelKind::kernel) {
    if (hasLessThan)
      return parser.emitError(parser.getCurrentLocation(),
                              "parameters not allowed in kgen.kernel, use "
                              "kgen.generator instead");
    // kgen.kernel's don't get paramDecl related attributes.
    return success();
  }

  // If there is no parameter list, or if it is empty, we're done.
  if (!hasLessThan || succeeded(parser.parseOptionalGreater())) {
    result.addAttribute("paramDecls", parser.getBuilder().getArrayAttr({}));
    result.addAttribute("numInputParameters",
                        parser.getBuilder().getI32IntegerAttr(0));
    return success();
  }

  SmallVector<Attribute> paramDecls;

  // Parse the input list.
  if (parseParamList(parser, paramDecls, /*isBinding=*/false))
    return failure();

  unsigned numInputs = paramDecls.size();

  // Check to see if we have results and parse them if so.
  if (succeeded(parser.parseOptionalArrow())) {
    if (parseParamList(parser, paramDecls, /*isBinding=*/false))
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

  // If this is a generator, see if it is an implementation of a generator
  // interface.
  if (opKind == GeneratorOrKernelKind::generator &&
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
  printFunctionAttributes(p, op, argTypes.size(), resultTypes.size(),
                          GeneratorOp::getAttributeNames());

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

//===----------------------------------------------------------------------===//
// GeneratorOp
//===----------------------------------------------------------------------===//

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
          getParamDecls().getValue().drop_front(getNumInputParameters()),
          getResultTypes())) ||
      failed(checkParametersInOpBody(*this)))
    return failure();

  return success();
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

  // Right now we require an exact match on everything.
  if (getFunctionTypeAttr() != interface.getFunctionTypeAttr())
    return emitError("generator has type ")
           << getFunctionTypeAttr() << " but interface " << interfaceSym
           << " expects " << interface.getFunctionTypeAttr();
  if (getParamDeclsAttr() != interface.getParamDeclsAttr())
    return emitError("generator has parameters ")
           << getParamDeclsAttr() << " but interface " << interfaceSym
           << " expects " << interface.getParamDeclsAttr();
  if (getNumInputParameters() != interface.getNumInputParameters())
    return emitError("generator has ")
           << getNumInputParameters() << " input parameters, but interface "
           << interfaceSym << " expects " << interface.getNumInputParameters();
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
  return parseGeneratorOrKernel(parser, result, GeneratorOrKernelKind::kernel);
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
// GeneratorInterfaceOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto calleeAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!calleeAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  Operation *callee = symbolTable.lookupNearestSymbolFrom(*this, calleeAttr);
  if (!isa_and_nonnull<GeneratorOp, KernelOp, GeneratorInterfaceOp>(callee))
    return emitError() << "'" << calleeAttr.getValue()
                       << "' does not reference a valid callee";

  // Verify that the operand and result types match the callee.
  auto fnType = callee->getAttrOfType<TypeAttr>("function_type")
                    .getValue()
                    .cast<FunctionType>();
  if (fnType.getNumInputs() != getNumOperands())
    return emitError("incorrect number of operands for callee");

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

  // Verify that the callee/caller parameters match.  The parameter names on the
  // results don't need to match, but the parameter names on the argument
  // bindings do.  The types always need to match.
  ArrayRef<Attribute> calleeParams;
  unsigned calleeNumInputParams = 0;
  if (isa<KernelOp>(callee)) {
    // Fully defined kernels never have parameters.
    calleeNumInputParams = 0;
  } else {
    assert((isa<GeneratorOp, GeneratorInterfaceOp>(callee)) &&
           "unknown callee");
    calleeParams = callee->getAttrOfType<ArrayAttr>("paramDecls").getValue();
    calleeNumInputParams =
        callee->getAttrOfType<IntegerAttr>("numInputParameters")
            .getValue()
            .getZExtValue();
    assert(calleeNumInputParams <= calleeParams.size());
  }

  // Check the parameter values specified to the input parameters.
  ArrayRef<Attribute> callerInputParams = getParamValues().getValue();
  ArrayRef<Attribute> calleeInputParamDecls =
      calleeParams.take_front(calleeNumInputParams);
  if (callerInputParams.size() != calleeInputParamDecls.size()) {
    auto diag = emitError("call has ")
                << callerInputParams.size()
                << " input parameters, but callee expects "
                << calleeInputParamDecls.size();
    diag.attachNote(callee->getLoc()) << "callee declared here";
    return failure();
  }

  // Input argument names and types need to match.
  unsigned paramNum = 0;
  for (auto [caller, calleeVal] :
       llvm::zip(callerInputParams, calleeInputParamDecls)) {
    auto callerBind = caller.cast<ParamBindAttr>();
    auto calleeDecl = calleeVal.cast<ParamDeclAttr>();
    if (callerBind.getName() != calleeDecl.getName()) {
      auto diag = emitError("call input parameter #")
                  << paramNum << " has name " << callerBind.getName()
                  << " but callee expects " << calleeDecl.getName();
      diag.attachNote(callee->getLoc()) << "callee declared here";
      return failure();
    }

    if (callerBind.getType() != calleeDecl.getType()) {
      auto diag = emitError("call input parameter ")
                  << callerBind.getName() << " passes parameter of type "
                  << callerBind.getType() << " but callee parameter has type "
                  << calleeDecl.getType();
      diag.attachNote(callee->getLoc()) << "callee declared here";
      return failure();
    }
    ++paramNum;
  }

  /// Check the parameter result values.
  ArrayRef<Attribute> callerOutputParamDecls = getParamDecls().getValue();
  ArrayRef<Attribute> calleeOutputParamDecls =
      calleeParams.drop_front(calleeNumInputParams);
  if (callerOutputParamDecls.size() != calleeOutputParamDecls.size()) {
    auto diag = emitError("call has ")
                << callerOutputParamDecls.size()
                << " result parameters when callee expects "
                << calleeOutputParamDecls.size();
    diag.attachNote(callee->getLoc()) << "callee declared here";
    return diag;
  }

  // The result names don't need to match up, but the result types do.
  paramNum = 0;
  for (auto [callerDeclAttr, calleeDeclAttr] :
       llvm::zip(callerOutputParamDecls, calleeOutputParamDecls)) {
    auto callerDecl = callerDeclAttr.cast<ParamDeclAttr>();
    auto calleeDecl = calleeDeclAttr.cast<ParamDeclAttr>();

    if (callerDecl.getType() != calleeDecl.getType()) {
      auto diag = emitError("result parameter #")
                  << paramNum << " has type " << calleeDecl.getType()
                  << " but caller parameter has type " << callerDecl.getType();
      diag.attachNote(callee->getLoc()) << "callee declared here";
      return failure();
    }
    ++paramNum;
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
