//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LIT dialect operations.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

SymbolRefAttr LIT::getFullyResolvedSymbolRef(mlir::SymbolOpInterface op) {
  SmallVector<FlatSymbolRefAttr> symbols;
  do {
    symbols.push_back(FlatSymbolRefAttr::get(op.getNameAttr()));
  } while ((op = dyn_cast<mlir::SymbolOpInterface>(op->getParentOp())));

  // Form a reference from the symbols we collected.
  if (symbols.size() == 1)
    return symbols.front();
  std::reverse(symbols.begin(), symbols.end());
  return SymbolRefAttr::get(symbols[0].getAttr(),
                            ArrayRef(symbols).drop_front());
}

//===----------------------------------------------------------------------===//
// ExportOp
//===----------------------------------------------------------------------===//

LogicalResult
LIT::ExportOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (getExports().empty())
    return emitOpError("exports must not be empty");

  // Just ensure we're exporting symbols we can see.
  auto module = KGENModule::from(*this, symbolTable);
  for (auto e : getExports().getAsRange<SymbolRefAttr>())
    if (!module.lookup<FuncInterface>(e))
      return emitOpError("could not find referenced symbol '") << e << "'";

  return success();
}

//===----------------------------------------------------------------------===//
// FileModuleOp
//===----------------------------------------------------------------------===//

void FileModuleOp::build(OpBuilder &odsBuilder, OperationState &state,
                         StringAttr name) {
  state.addAttribute(getSymNameAttrName(state.name), name);
  state.addRegion()->push_back(new Block());
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

/// Return a SymbolConstantAttr for this function, optionally bound to a set
/// of parameter bindings.
SymbolConstantAttr LIT::FuncOp::getBoundReference(ParamBindArrayAttr bindings) {
  if (!bindings) // We allow null for convenience.
    bindings = ParamBindArrayAttr::get(getContext(), {});

  // SymbolConstantAttr provides a type for the SymbolRefAttr with the
  // parameters substituted in.  The function reference binds any parameter
  // bindings present on the access (in bindings), which typically concretizes
  // the signature.
  SignatureType resultType = getFullSignature();
  if (!bindings.empty()) {
    resultType = resultType.getSpecializedSignature(
        bindings, [&]() -> InFlightDiagnostic {
          llvm_unreachable("bad bindings specified for getBoundReference");
        });
    assert(resultType && "bad bindings specified for getBoundReference");
  }

  return SymbolConstantAttr::get(getFullyResolvedSymbolRef(*this), bindings,
                                 resultType);
}

ReturnOp LIT::FuncOp::getReturnOp() {
  // Tolerate malformed IR because this is used by the printer.
  if (isExternal() || getBody()->empty())
    return {};
  return dyn_cast<ReturnOp>(getBody()->back());
}

/// Return the normal result type.  This is the same as getResultType unless
/// the function throws, in which case this is dug out of the variant.
Type LIT::FuncOp::getNormalResultType() {
  Type resultType = getResultType();
  if (!getRaises())
    return resultType;

  // We know that the ABI of a raising function will have it return
  // ErrorOr<NormalType>.  ErrorOr is a Variant<Error, NormalType>, and in the
  // corner case where we return an error, it will be Variant<Error> only.
  auto variant = cast<POP::VariantType>(resultType);
  unsigned normalIdx = std::min(variant.getNumTypes() - 1, size_t(1));
  return variant.getType(normalIdx);
}

// These FuncOp attributes are disallowed while parsing since they can
// be inferred. Likewise while printing we ignore them.
static StringRef disallowedAttrNames[] = {
    "constraints",     "implements", "signature",  "sym_name",
    "valueParamNames", "evaluator",  "defaultImpl"};

/// Parses a LIT Generator.
ParseResult LIT::FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type> resultTypes;
  Builder &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  ParamDeclArrayAttr inputParamDecls;
  TypeArrayAttr resultParamTypes;
  ConventionsAttr conventions;
  llvm::SMLoc sigLoc;
  if (parseOptionalParameterSpec(parser, inputParamDecls, resultParamTypes) ||
      parser.getCurrentLocation(&sigLoc) ||
      parseFunctionSignature(parser, entryArgs, resultTypes, conventions))
    return failure();

  ConstraintArrayAttr constraints;
  if (parseOptionalConstraints(parser, constraints))
    return failure();
  result.addAttribute("constraints", constraints);

  SmallVector<Type> argTypes;
  argTypes.reserve(entryArgs.size());
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);
  FunctionType type = builder.getFunctionType(argTypes, resultTypes);
  auto signature =
      parser.getChecked<SignatureType>(parser.getContext(), inputParamDecls,
                                       resultParamTypes, type, conventions);
  if (!signature)
    return failure();

  result.addAttribute(getSignatureAttrName(result.name),
                      TypeAttr::get(signature));

  // Handle keyword argument names.
  SmallVector<StringAttr> names;
  for (OpAsmParser::Argument &arg : entryArgs) {
    StringRef spelling;
    if (arg.ssaName.name.size() < 2)
      return parser.emitError(sigLoc, "arguments requires SSA names");
    if (isdigit(arg.ssaName.name[1])) // %42 -> no name.
      spelling = "";
    else
      spelling = arg.ssaName.name.drop_front();
    names.push_back(builder.getStringAttr(spelling));
  }

  result.addAttribute(getValueParamNamesAttrName(result.name),
                      StringArrayAttr::get(builder.getContext(), names));

  // If function attributes are present, parse them.
  NamedAttrList parsedAttributes;
  llvm::SMLoc attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();

  // If this function implements an interface.
  if (succeeded(parser.parseOptionalKeyword("implements"))) {
    SymbolRefAttr implementsAttr;
    if (parser.parseAttribute(implementsAttr,
                              parser.getBuilder().getType<::mlir::NoneType>(),
                              "implements", result.attributes))
      return failure();
  }

  // Disallow attributes that are inferred from elsewhere in the attribute
  // dictionary.
  for (StringRef disallowed : disallowedAttrNames) {
    if (parsedAttributes.get(disallowed))
      return parser.emitError(attributeDictLocation, "'")
             << disallowed
             << "' is an inferred attribute and should not be specified in the "
                "explicit attribute dictionary";
  }
  result.attributes.append(parsedAttributes);

  // Parse the required function body.
  Region *region = result.addRegion();

  // If this is a generator interface, no body block is allowed.
  if (!isa_and_nonnull<mlir::UnitAttr>(parsedAttributes.get("isInterface")) &&
      parser.parseRegion(*region, entryArgs, /*enableNameShadowing=*/true))
    return failure();

  // Parse an optional evaluator.
  if (parser.parseOptionalKeyword("evaluator"))
    return success();

  Type sigType;
  TypedAttr evaluator;
  if (parseKGENType(parser, sigType) || parser.parseEqual() ||
      parseParamValue(parser, evaluator, sigType))
    return failure();
  result.addAttribute(LIT::FuncOp::getEvaluatorAttrName(result.name),
                      evaluator);
  return success();
}

// Print the LIT::FuncOp using the shared printing logic.
void LIT::FuncOp::print(OpAsmPrinter &p) {
  using namespace mlir::function_interface_impl;

  FuncInterface op = cast<FuncInterface>(getOperation());
  auto func = cast<mlir::FunctionOpInterface>(*op);
  // Print the operation and the function name.
  p << ' ';

  p.printSymbolName(func.getName());
  printOptionalParameterSpec(op.getInputParamDeclsAttr(),
                             op.getResultParamTypesAttr(), p.getStream());

  ArrayRef<Type> argTypes = op.getArgumentTypes();
  printFunctionSignature(p, func.getFunctionBody(), argTypes,
                         op.getResultTypes(), op.getConventions(),
                         op->getAttrOfType<StringArrayAttr>("valueParamNames"));

  // Don't print the following in lit.func.
  SmallVector<StringRef> ignoredAttrNames(
      (ArrayRef<StringRef>(disallowedAttrNames)));

  printFunctionAttributes(p, op, ignoredAttrNames);
  printOptionalConstraints(p, func, cast<DeclInterface>(*op).getConstraints());

  // If this is a generator implementing a generator.interface, include the
  // symbol for the generator interface.
  if (auto implementsAttr = getImplementsAttr()) {
    p.printNewline();
    p << "  implements " << implementsAttr;
  }

  p << ' ';
  if (!func.isExternal())
    p.printRegion(func.getFunctionBody(), /*printEntryBlockArgs=*/false);
  if (SymbolConstantAttr evaluator = getEvaluatorAttr()) {
    p << " evaluator ";
    printKGENType(p.getStream(), evaluator.getType());
    p << " = ";
    printParamValue(evaluator, p.getStream());
  }
}

// Name the arguments of the region with the valueParamNames.
void LIT::FuncOp::getAsmBlockArgumentNames(
    Region &body, llvm::function_ref<void(Value, StringRef)> setNameFn) {
  // Set a name for each argument.
  if (body.empty())
    return;
  Block *bodyBlock = getBody();
  for (auto [arg, name] :
       llvm::zip(bodyBlock->getArguments(), getValueParamNames()))
    setNameFn(arg, name);
}

Region *LIT::FuncOp::getCallableRegion() {
  // If the body is empty, return null to indicate that this is an "external"
  // callable.
  if (getBody()->empty())
    return nullptr;
  return &getBodyRegion();
}

ArrayRef<Type> LIT::FuncOp::getCallableResults() { return getResultTypes(); }

LogicalResult LIT::FuncOp::verifyRegions() {
  // Check that the number of argument labels matches the number of argument
  // types.
  if (getValueParamNames().size() != getFunctionType().getNumInputs())
    return emitOpError("incorrect number of value parameter labels");

  // Interfaces must have empty bodies and cannot have an implements attribute.
  if (getIsInterface()) {
    if (!isExternal())
      return emitOpError("interface expected an empty function body");
    if (getImplementsAttr())
      return emitOpError("@interface and @implements decorators "
                         "cannot be set at the same time");
    return success();
  }

  // Generators must have non-empty bodies terminated by a return.
  if (getFunctionBody().empty() || getBody()->empty())
    return emitOpError("expected non-empty function body");
  if (!getBody()->back().hasTrait<OpTrait::IsTerminator>())
    return emitOpError("expected a terminator");
  ReturnOp returnOp = getReturnOp();
  if (!returnOp)
    return emitOpError("should have a return");

  // Check result types match the ReturnOp.
  return returnOp.checkArgumentTypes(getResultParamTypes(), {getResultTypes()});
}

LogicalResult
LIT::FuncOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // If this function is top-level, see if the parameter definitions and uses
  // within the generator are structured correctly.
  if (isa<ModuleOp>((*this)->getParentOp()) &&
      failed(ParameterDeclsAndUses().calculateAndVerify(*this, symbolTable)))
    return failure();

  // If the generator is implementing a generator interface, check that they
  // line up correctly.
  SymbolRefAttr interfaceSym = getImplementsAttr();
  if (!interfaceSym)
    return success();

  // Check that the callee attribute was specified.
  auto module = KGENModule::from(*this, symbolTable);
  auto interface = module.lookup<GeneratorInterfaceOp>(interfaceSym);
  auto funcInterface = module.lookup<LIT::FuncOp>(interfaceSym);
  if (!interface && (!funcInterface || !funcInterface.getIsInterface()))
    return emitError() << interfaceSym
                       << " does not reference a generator interface";
  TypeArrayAttr interfaceResultParamTypesAttr;
  if (funcInterface)
    interfaceResultParamTypesAttr = funcInterface.getResultParamTypesAttr();
  else
    interfaceResultParamTypesAttr = interface.getResultParamTypesAttr();
  // Result parameters need to match, but input parameters may be inferred.
  if (getResultParamTypesAttr() != interfaceResultParamTypesAttr)
    return emitError() << "lit.func result parameter types must match "
                          "interface types";

  return success();
}

void LIT::FuncOp::build(OpBuilder &builder, OperationState &result) {
  auto context = builder.getContext();

  // Before resolution, we treat the function as having type ()->Error,
  // because parse or other errors forming the signature won't update the
  // representation.  This makes sure that the error case doesn't break
  // invariants (that functions always have a single result).
  auto errorType = builder.getType<TypeCheckErrorType>();
  auto signatureType =
      SignatureType::get(context, ArrayRef<Type>(), {errorType});

  auto emptyParamNames = StringArrayAttr::get(context, {});

  // NOTE: We set an attribute named 'sym_namex' here instead of setting
  // 'sym_name' because we don't /know/ the symbol name on construction and need
  // to set it during signature resolution phase of the parser.
  //
  // Unfortunately, we cannot set it to null because that causes the SymbolTable
  // logic to be extremely cranky and breaks other MLIR invariants.
  //
  // We also cannot completely omit the symbol, because ODS is doing some clever
  // stuff to speed up attribute lookup.  That clever stuff requires that a slot
  // is filled in the attr dict, so we set this thing and remove it when the
  // real name is set.
  result.addAttribute("sym_namex", emptyParamNames);

  result.addAttribute(getValueParamNamesAttrName(result.name), emptyParamNames);
  result.addAttribute(getSignatureAttrName(result.name),
                      TypeAttr::get(signatureType));
  result.addAttribute(getConstraintsAttrName(result.name),
                      ConstraintArrayAttr::get(context, {}));
  result.addRegion()->push_back(new Block());
}

//===----------------------------------------------------------------------===//
// UnwrapOrPropagateOp
//===----------------------------------------------------------------------===//

LogicalResult UnwrapOrPropagateOp::verify() {
  if ((*this)->getParentOfType<TryOp>())
    return success();
  auto func = (*this)->getParentOfType<LIT::FuncOp>();
  if (!func)
    return emitOpError() << "must be contained in a `lit.func`";
  if (func.getConventions().getFnEffects() != FnEffects::Throws)
    return emitOpError()
           << "cannot propagate error in a function that does not throw";
  return success();
}

//===----------------------------------------------------------------------===//
// TryOp
//===----------------------------------------------------------------------===//

static ParseResult parseExceptRegion(OpAsmParser &p, Region &region) {
  OpAsmParser::Argument arg;
  if (p.parseLParen() || p.parseArgument(arg, /*allowType=*/true) ||
      p.parseRParen() || p.parseRegion(region, arg))
    return failure();
  return success();
}

static void printExceptRegion(OpAsmPrinter &p, Operation *op, Region &region) {
  p << '(';
  p.printRegionArgument(region.getArgument(0));
  p << ") ";
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

LogicalResult TryOp::verify() {
  if (getTryRegion().getNumArguments() != 0)
    return emitOpError("expected try region to have zero arguments");
  if (getExceptRegion().getNumArguments() != 1)
    return emitOpError("expected except region to have one arguments");
  if (getElseRegion().getNumArguments() != 0)
    return emitOpError("expected else region to have zero arguments");
  return success();
}

void TryOp::getEntryTargets(ArrayRef<Attribute> operands,
                            SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.empty());
  targets.emplace_back(0);
}

ValueRange TryOp::getEntryArguments(std::optional<unsigned> target) {
  if (!target)
    return {};
  return getRegion(*target).getArguments();
}

//===----------------------------------------------------------------------===//
// TryYieldOp
//===----------------------------------------------------------------------===//

bool TryYieldOp::isParentNode(Operation *op) { return isa<TryOp>(op); }

void TryYieldOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  Region *region = (*this)->getParentRegion();
  // Figure out which region this yield is in.
  if (!isa<TryOp>(region->getParentOp()))
    region = region->getParentRegion();

  // The region indices of the try operation.
  enum { TRY, EXCEPT, ELSE };
  switch (region->getRegionNumber()) {
  case TRY:
    // Yield from the 'try' region branches to the 'else' region.
    targets.emplace_back(ELSE);
    break;
  case EXCEPT:
  case ELSE:
    // Yield from either the 'except' or 'else' regions branches back to the
    // parent operation.
    targets.emplace_back(std::nullopt);
    break;
  default:
    llvm_unreachable("unknown lit.try region");
  }
}

//===----------------------------------------------------------------------===//
// TryRaiseOp
//===----------------------------------------------------------------------===//

bool TryRaiseOp::isParentNode(Operation *op) { return isa<TryOp>(op); }

void TryRaiseOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == 1);
  targets.emplace_back(1, (*this)->getOperands());
}

//===----------------------------------------------------------------------===//
// LetDeclOp / VarDeclOp
//===----------------------------------------------------------------------===//

void LetDeclOp::build(OpBuilder &builder, OperationState &state,
                      Type resultType, StringAttr name) {
  state.addAttribute(getNameAttrName(state.name), name);
  state.addTypes(resultType);
}

void LetDeclOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getName());
}

void VarDeclOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getName());
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "KGEN/LITDialect/LIT.cpp.inc"
