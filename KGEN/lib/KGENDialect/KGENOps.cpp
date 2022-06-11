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
// GeneratorOp
//===----------------------------------------------------------------------===//

void GeneratorOp::build(OpBuilder &builder, OperationState &state,
                        StringAttr name, ArrayAttr parameters,
                        ArrayRef<Type> inputTypes, ArrayRef<Type> resultTypes,
                        ArrayRef<NamedAttribute> attrs) {
  state.addAttribute("parameters", parameters);
  buildWithEntryBlock(builder, state, name,
                      builder.getFunctionType(inputTypes, resultTypes), attrs,
                      inputTypes);
}

ReturnOp GeneratorOp::getReturnOp() {
  return cast<ReturnOp>(getBodyBlock()->getTerminator());
}

/// Parse an parameter list if present.
/// parameter-list ::= `<` parameter-decl (`,` parameter-decl)* `>`
/// parameter-decl ::= identifier (`:` type)?
///
static ParseResult parseOptionalParameters(OpAsmParser &parser,
                                           SmallVector<Attribute> &parameters) {
  return parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::OptionalLessGreater, [&]() -> ParseResult {
        std::string name;
        Type type;
        if (parser.parseKeywordOrString(&name))
          return failure();
        if (succeeded(parser.parseOptionalColon())) {
          if (parser.parseType(type))
            return failure();
        } else {
          type = parser.getBuilder().getIntegerType(64, /*isSigned=*/true);
        }

        parameters.push_back(ParamDeclAttr::get(name, type));
        return success();
      });
}

/// Parses a KGEN Generator.
///
/// operation ::=
/// `kgen.generator` function-signature function-attributes? function-body
///
ParseResult GeneratorOp::parse(OpAsmParser &parser, OperationState &result) {
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
  SmallVector<Attribute> parameters;

  if (parseOptionalParameters(parser, parameters) ||
      parseFunctionSignature(parser, /*allowVariadic=*/false, entryArgs,
                             isVariadic, resultTypes, resultAttrs))
    return failure();

  result.addAttribute("parameters", builder.getArrayAttr(parameters));

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

  // Parse the optional function body. The printer will not print the body if
  // its empty, so disallow parsing of empty body in the parser.
  auto *body = result.addRegion();
  llvm::SMLoc loc = parser.getCurrentLocation();
  mlir::OptionalParseResult parseResult =
      parser.parseOptionalRegion(*body, entryArgs,
                                 /*enableNameShadowing=*/false);
  if (parseResult.hasValue()) {
    if (failed(*parseResult))
      return failure();
    // Function body was parsed, make sure its not empty.
    if (body->empty())
      return parser.emitError(loc, "expected non-empty function body");
  }
  return success();
}

/// Print a parameter list for a module or instance.
static void printParameterList(ArrayAttr parameters, OpAsmPrinter &p) {
  if (parameters.empty())
    return;

  p << '<';
  llvm::interleaveComma(parameters, p, [&](Attribute param) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    p << paramAttr.getName().getValue();
    if (!paramAttr.getType().getValue().isSignedInteger(64))
      p << ": " << paramAttr.getType();
  });
  p << '>';
}

// Print the GeneratorOp. Collects argument and result types and passes them to
// helper functions. Drops "void" result since it cannot be parsed back.
void GeneratorOp::print(OpAsmPrinter &p) {
  using namespace mlir::function_interface_impl;
  Operation *op = getOperation();

  // Print the operation and the function name.
  auto funcName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  p << ' ';

  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility = op->getAttrOfType<StringAttr>(visibilityAttrName))
    p << visibility.getValue() << ' ';
  p.printSymbolName(funcName);

  printParameterList(op->getAttrOfType<ArrayAttr>("parameters"), p);

  ArrayRef<Type> argTypes = getArgumentTypes();
  ArrayRef<Type> resultTypes = getResultTypes();
  printFunctionSignature(p, *this, argTypes, /*isVariadic=*/false, resultTypes);
  printFunctionAttributes(p, *this, argTypes.size(), resultTypes.size(),
                          {visibilityAttrName, "parameters"});

  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

LogicalResult GeneratorOp::verifyRegions() {
  return getReturnOp().checkArgumentTypes(getResultTypes());
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

/// Containers verify that the operands of this ReturnOp match the specified set
/// of types.
LogicalResult ReturnOp::checkArgumentTypes(TypeRange types) {
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
