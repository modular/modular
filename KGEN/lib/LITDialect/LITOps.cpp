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
  return cast<ReturnOp>(getBody()->getTerminator());
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

  // Check result types match the ReturnOp.
  if (failed(getReturnOp().checkArgumentTypes(getResultParamTypes(),
                                              {getResultTypes()})))
    return failure();

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
  if (!interface)
    return emitError() << "'" << interfaceSym.getValue()
                       << "' does not reference a generator interface";

  // Result parameters need to match, but input parameters may be inferred.
  if (getResultParamTypesAttr() != interface.getResultParamTypesAttr())
    return emitError() << "lit.func result parameter types must match "
                          "interface types";

  return success();
}

void LITFuncOp::build(OpBuilder &builder, OperationState &result,
                      StringAttr name) {
  auto context = builder.getContext();
  auto functionType =
      builder.getFunctionType(ArrayRef<Type>(), ArrayRef<Type>());
  build(builder, result, name, StringArrayAttr::get(context, {}),
        TypeAttr::get(functionType), ParamDeclArrayAttr::get(context, {}),
        TypeArrayAttr::get(context, {}), ConstraintArrayAttr::get(context, {}),
        /*isStatic=*/mlir::UnitAttr(), FlatSymbolRefAttr());
  result.regions[0]->push_back(new Block());
}

/// If this is a special function like __init__ return the enum that
/// identifies it, otherwise return kNormal.
SpecialFunctionKind LITFuncOp::getSpecialFunctionKind() {
  StringRef nameStr = getName();
  size_t methodSepIdx = nameStr.rfind("::");
  // If this is a method, strip struct/class container name.
  if (methodSepIdx != StringRef::npos)
    nameStr = nameStr.substr(methodSepIdx + 2, nameStr.size());

  if (nameStr.size() < 5 || !nameStr.startswith("__") ||
      !nameStr.endswith("__"))
    return SpecialFunctionKind::kNormal;
  nameStr = nameStr.drop_front(2).drop_back(2);
  if (nameStr == "init")
    return SpecialFunctionKind::kInit;
  if (nameStr == "new")
    return SpecialFunctionKind::kNew;

  // Otherwise, this declaration isn't known.
  return SpecialFunctionKind::kNormal;
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

/// Lookup the declaration for the struct. When checking field types, we can't
/// directly compare operation types to the struct field types because they are
/// parameterized under different domains. We have to rebind them.
static LogicalResult
lookupStructDecl(SymbolTableCollection &symbolTable, Operation *user,
                 RefType ref,
                 std::pair<LITStructDeclOp, ParameterEvaluator> &result) {
  auto module = KGENModule::from(user, symbolTable);
  auto structDecl = module.lookup<LITStructDeclOp>(ref.getName());
  if (!structDecl)
    return user->emitOpError("expected a struct declaration");

  ParameterEvaluator evaluator;
  for (ParamBindAttr bind : ref.getParamValues())
    evaluator.setParameterValue(bind.getDecl(), bind.getValue());

  result = std::make_pair(structDecl, std::move(evaluator));
  return success();
}

static LogicalResult
verifyStructFieldAndType(SymbolTableCollection &symbolTable, Operation *op,
                         RefType ref, StringAttr fieldName, Type type) {

  std::pair<LITStructDeclOp, ParameterEvaluator> structDeclEval;
  if (failed(lookupStructDecl(symbolTable, op, ref, structDeclEval)))
    return failure();
  auto [structDecl, evaluator] = structDeclEval;
  for (VarDeclOp fieldDecl : structDecl.getFieldDecls()) {
    if (fieldDecl.getName() != fieldName)
      continue;
    Type reboundType =
        evaluator.getReboundType(fieldDecl.getType().getResolvedElementType());
    if (reboundType != type)
      return op->emitOpError("cannot extract value of type ")
             << type << " from struct field " << fieldName << " which has type "
             << reboundType;
    return success();
  }

  return op->emitOpError("struct ")
         << ref.getName() << " has no field named " << fieldName;
}

LogicalResult
LITStructExtractOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Type structType = getStructVal().getType();
  return verifyStructFieldAndType(symbolTable, *this, cast<RefType>(structType),
                                  getFieldAttr(), getResult().getType());
}

LogicalResult
LITStructGEPOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  TypedAttr refExpr = getContainer().getType().getElementType();
  return verifyStructFieldAndType(
      symbolTable, *this,
      cast<RefType>(cast<TypeConstantAttr>(refExpr).getValue()), getFieldAttr(),
      ParamRefType::get(getResult().getType().getElementType()));
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
