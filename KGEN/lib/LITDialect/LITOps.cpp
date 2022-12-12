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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

ReturnOp LIT::FuncOp::getReturnOp() {
  // Tolerate malformed IR because this is used by the printer.
  if (isExternal() || getBody()->empty())
    return {};
  return dyn_cast<ReturnOp>(getBody()->getTerminator());
}

/// Parses a LIT Generator.
ParseResult LIT::FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseGeneratorOrFunc(parser, result, GeneratorOrFuncKind::litfunc);
}

// Print the LIT::FuncOp using the shared printing logic.
void LIT::FuncOp::print(OpAsmPrinter &p) { printGeneratorOrFunc(p, *this); }

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
  FlatSymbolRefAttr interfaceSym = getImplementsAttr();
  if (!interfaceSym)
    return success();

  // Check that the callee attribute was specified.
  auto module = KGENModule::from(*this, symbolTable);
  auto interface = module.lookup<GeneratorInterfaceOp>(interfaceSym);
  auto funcInterface = module.lookup<LIT::FuncOp>(interfaceSym);
  if (!interface && (!funcInterface || !funcInterface.getIsInterface()))
    return emitError() << "'" << interfaceSym.getValue()
                       << "' does not reference a generator interface";
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

void LIT::FuncOp::build(OpBuilder &builder, OperationState &result,
                        StringAttr name) {
  auto context = builder.getContext();

  // Before resolution, we treat the function as having type ()->Error,
  // because parse or other errors forming the signature won't update the
  // representation.  This makes sure that the error case doesn't break
  // invariants (that functions always have a single result).
  auto errorType = builder.getType<TypeCheckErrorType>();
  auto signatureType =
      SignatureType::get(context, ArrayRef<Type>(), {errorType});
  build(builder, result, name, StringArrayAttr::get(context, {}),
        TypeAttr::get(signatureType), ConstraintArrayAttr::get(context, {}),
        /*isStatic=*/mlir::UnitAttr(), /*isInterface=*/mlir::UnitAttr(),
        /*isDef=*/mlir::UnitAttr(), /*raises=*/mlir::UnitAttr(),
        /*implements=*/FlatSymbolRefAttr());
  result.regions[0]->push_back(new Block());
}

//===----------------------------------------------------------------------===//
// LITTryOp
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

ValueRange TryOp::getEntryArguments(Optional<unsigned> target) {
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
    targets.emplace_back(None);
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
// VarDeclOp
//===----------------------------------------------------------------------===//

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
