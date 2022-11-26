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

//===----------------------------------------------------------------------===//
// LITFuncOp
//===----------------------------------------------------------------------===//

ReturnOp LITFuncOp::getReturnOp() {
  // Tolerate malformed IR because this is used by the printer.
  Block *body = getBody();
  if (body && !body->empty())
    return dyn_cast_or_null<ReturnOp>(&body->back());
  return {};
}

/// Parses a LIT Generator.
ParseResult LITFuncOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseGeneratorOrFunc(parser, result, GeneratorOrFuncKind::litfunc);
}

// Print the LITFuncOp using the shared printing logic.
void LITFuncOp::print(OpAsmPrinter &p) { printGeneratorOrFunc(p, *this); }

// Name the arguments of the region with the valueParamNames.
void LITFuncOp::getAsmBlockArgumentNames(
    Region &body, llvm::function_ref<void(Value, StringRef)> setNameFn) {
  // Set a name for each argument.
  if (body.empty())
    return;
  Block *bodyBlock = getBody();
  for (auto [arg, name] :
       llvm::zip(bodyBlock->getArguments(), getValueParamNames()))
    setNameFn(arg, name);
}

LogicalResult LITFuncOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the number of argument labels matches the number of argument
  // types.
  if (getValueParamNames().size() != getFunctionType().getNumInputs())
    return emitOpError("incorrect number of value parameter labels");

  bool isInterface = getIsInterface();
  Block *body = getBody();

  if (isInterface) {
    if (!body->empty())
      return emitOpError("expected empty function body");

    if (getImplements().has_value())
      return emitOpError("@interface and @implements decorators "
                         "cannot be set at the same time");
  } else {
    if (body->empty())
      return emitOpError("expected non-empty function body");

    // Verify the return operation if this is a non-interface.
    auto returnOp = getReturnOp();
    if (!returnOp)
      return emitOpError("lit.func should have a return");

    // Check result types match the ReturnOp.
    if (failed(returnOp.checkArgumentTypes(getResultParamTypes(),
                                           {getResultTypes()})))
      return failure();
  }

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
  auto funcInterface = module.lookup<LITFuncOp>(interfaceSym);
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

void LITFuncOp::build(OpBuilder &builder, OperationState &result,
                      StringAttr name) {
  auto context = builder.getContext();

  // Before resolution, we treat the function as having type ()->Error,
  // because parse or other errors forming the signature won't update the
  // representation.  This makes sure that the error case doesn't break
  // invariants (that functions always have a single result).
  auto errorType = builder.getType<TypeCheckErrorType>();
  auto functionType = builder.getFunctionType(ArrayRef<Type>(), {errorType});
  build(builder, result, name, StringArrayAttr::get(context, {}),
        TypeAttr::get(functionType), ParamDeclArrayAttr::get(context, {}),
        TypeArrayAttr::get(context, {}), ConstraintArrayAttr::get(context, {}),
        /*isStatic=*/mlir::UnitAttr(), /*isInterface=*/mlir::UnitAttr(),
        /*isDef=*/mlir::UnitAttr(), /*implements=*/FlatSymbolRefAttr());
  result.regions[0]->push_back(new Block());
}

//===----------------------------------------------------------------------===//
// LITStructGEPOp
//===----------------------------------------------------------------------===//

/// Parse the struct field name as a keyword literal.
static ParseResult parseKeywordAsString(OpAsmParser &p, StringAttr &name) {
  StringRef value;
  if (p.parseKeyword(&value))
    return failure();
  name = p.getBuilder().getStringAttr(value);
  return success();
}

/// Print the struct field name as a keyword literal.
static void printKeywordAsString(OpAsmPrinter &p, Operation *op,
                                 StringAttr name) {
  p << name.getValue();
}

/// Given a struct type with bound parameters and a field decl within it,
/// calculate and return the correct type for the field given substitutions.
static Type getReboundFieldType(LITDeclRefType structType, VarDeclOp field) {
  // Ensure the field comes from the struct type in question.
  assert(structType.getSymbol().getRootReference() ==
         cast<LITStructDeclOp>(field->getParentOp()).getNameAttr());

  // Remap the type of the field based on any bound parameter types from the
  // base reference.
  ParameterEvaluator evaluator;
  for (ParamBindAttr bind : structType.getParamValues())
    evaluator.setParameterValue(bind.getDecl(), bind.getValue());

  return evaluator.getReboundType(field.getType().getResolvedElementType());
}

static LogicalResult
verifyStructFieldAndType(SymbolTableCollection &symbolTable, Operation *op,
                         LITDeclRefType ref, StringAttr fieldName, Type type) {
  auto module = KGENModule::from(op, symbolTable);
  auto structDecl = module.lookup<LITStructDeclOp>(ref.getSymbol());
  if (!structDecl)
    return op->emitOpError("expected a struct declaration");

  for (VarDeclOp fieldDecl : structDecl.getFieldDecls()) {
    if (fieldDecl.getName() != fieldName)
      continue;
    Type reboundType = getReboundFieldType(ref, fieldDecl);
    if (reboundType != type)
      return op->emitOpError("cannot extract value of type ")
             << type << " from struct field " << fieldName << " which has type "
             << reboundType;
    return success();
  }

  return op->emitOpError("struct ")
         << ref.getSymbol() << " has no field named " << fieldName;
}

LogicalResult
LITStructGEPOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  TypedAttr refExpr = getContainer().getType().getElementType();
  return verifyStructFieldAndType(
      symbolTable, *this,
      cast<LITDeclRefType>(cast<TypeConstantAttr>(refExpr).getValue()),
      getFieldAttr(),
      ParamRefType::get(getResult().getType().getElementType()));
}

void LITStructGEPOp::build(OpBuilder &builder, OperationState &result,
                           Value structBasePtr, VarDeclOp field) {
  TypedAttr refExpr =
      cast<POP::PointerType>(structBasePtr.getType()).getElementType();
  auto structType =
      cast<LITDeclRefType>(cast<TypeConstantAttr>(refExpr).getValue());
  build(builder, result,
        POP::PointerType::get(getReboundFieldType(structType, field)),
        field.getNameAttr(), structBasePtr);
}

//===----------------------------------------------------------------------===//
// LITStructExtractOp
//===----------------------------------------------------------------------===//

LogicalResult
LITStructExtractOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Type structType = getStructVal().getType();
  return verifyStructFieldAndType(symbolTable, *this,
                                  cast<LITDeclRefType>(structType),
                                  getFieldAttr(), getResult().getType());
}

void LITStructExtractOp::build(OpBuilder &builder, OperationState &result,
                               Value structBasePtr, VarDeclOp field) {
  auto structType = cast<LITDeclRefType>(structBasePtr.getType());
  build(builder, result, getReboundFieldType(structType, field),
        field.getNameAttr(), structBasePtr);
}

//===----------------------------------------------------------------------===//
// LITStructDeclOp
//===----------------------------------------------------------------------===//

/// Struct declarations aren't functions.
FunctionType LITStructDeclOp::getFunctionType() {
  llvm_unreachable("structs don't have function types");
}

/// Verify that the body has no arguments and that the declaration has no result
/// types.
LogicalResult LITStructDeclOp::verify() {
  if (getFields().getNumArguments())
    return emitOpError("expected declaration body to have no arguments");

  if (!getResultParamTypes().empty())
    return emitOpError("unexpected result parameters");

  return success();
}

/// Verify parameter uses.
LogicalResult
LITStructDeclOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return ParameterDeclsAndUses().calculateAndVerify(*this, symbolTable);
}

void LITStructDeclOp::build(OpBuilder &builder, OperationState &result,
                            StringAttr name) {
  auto context = builder.getContext();
  build(builder, result, name, ParamDeclArrayAttr::get(context, {}),
        TypeArrayAttr::get(context, {}));
  result.regions[0]->push_back(new Block());
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
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
