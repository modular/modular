//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the KGEN dialect operations.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/HLCFDialect/HLCFUtils.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Compiler/ParserUtils.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

using HLCF::parseLoop;
using HLCF::printLoop;

//===----------------------------------------------------------------------===//
// ParamConstantOp
//===----------------------------------------------------------------------===//

static ParseResult parseParamConstantOpValue(OpAsmParser &p, TypedAttr &value,
                                             Type &resultType) {
  if (parseColonTypeOrIndex(p, resultType) || p.parseEqual() || p.parseLess() ||
      parseParamValue(p, value, resultType) || p.parseGreater())
    return failure();
  return success();
}

static void printParamConstantOpValue(OpAsmPrinter &p, Operation *,
                                      TypedAttr value, Type resultType) {
  printColonTypeOrIndex(p, value.getType());
  p << " = <";
  printParamValue(p, value);
  p << ">";
}

/// Parameter materialization operations are not allowed to materialize
/// capturing signature-typed values, since they can be inlined.
template <typename OpT>
static LogicalResult verifyParamValueOp(OpT op) {
  if (op.getValue().getType() == op.getType())
    return success();
  return op.emitOpError() << "parameter type " << op.getValue().getType()
                          << " does not match result type " << op.getType();
}

/// Return true if the parameter value contains symbol constants, making the
/// operation implicit parametric.
static bool containsSymbolConstants(TypedAttr value) {
  mlir::AttrTypeWalker walker;
  walker.addWalk([](SymbolConstantAttr) { return WalkResult::interrupt(); });
  return walker.walk(value).wasInterrupted();
}

LogicalResult ParamConstantOp::verify() { return verifyParamValueOp(*this); }

void ParamConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  // If the type of the value has a registered pretty name, use that for the SSA
  // value name.
  if (std::optional<StringRef> name =
          getContext()->getLoadedDialect<KGENDialect>()->getTypeName(
              getType().getTypeID())) {
    setNameFn(getResult(), *name);
    return;
  }

  // Otherwise, handle some common cases here.
  if (isa<IndexType>(getType())) {
    if (auto intVal = dyn_cast<IntegerAttr>(getValue()))
      setNameFn(getResult(), ("index" + Twine(intVal.getInt())).str());
    else
      setNameFn(getResult(), "index");
  }
}

bool ParamConstantOp::isImplicitlyParametric() {
  return containsSymbolConstants(getValue());
}

void ParamConstantOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {}

void ParamConstantOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {

  // This is a hack designed to stop the CSE of "none" in closures.
  //
  // See MSTDL-1123 for more context, but until cross-device compilation is
  // more robust, we need this to avoid some edge case behavior around None.
  //
  // TODO(MOCO-1101): remove the need for this and make op Pure again.
  if (isa<KGEN::NoneType>(getValue().getType()) &&
      getOperation()->getParentOfType<KGEN::ParamDeclareRegionOp>())
    effects.emplace_back(mlir::MemoryEffects::Allocate::get());
  return;
}

//===----------------------------------------------------------------------===//
// ParamMaterializeOp
//===----------------------------------------------------------------------===//

LogicalResult ParamMaterializeOp::verify() { return verifyParamValueOp(*this); }

bool ParamMaterializeOp::isImplicitlyParametric() {
  return containsSymbolConstants(getValue());
}

void ParamMaterializeOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {}

//===----------------------------------------------------------------------===//
// ParamDeclareOp
//===----------------------------------------------------------------------===//

static ParseResult parseParamDeclareOpValue(OpAsmParser &p,
                                            ParamDeclAttr &paramDecl,
                                            TypedAttr &value) {
  if (parseParamDecl(p, paramDecl) || p.parseEqual() || p.parseLess() ||
      parseParamValue(p, value, paramDecl.getType()) || p.parseGreater())
    return failure();

  return success();
}

static void printParamDeclareOpValue(OpAsmPrinter &p, Operation *,
                                     ParamDeclAttr paramDecl, TypedAttr value) {
  printParamDecl(p, paramDecl);
  p << " = <";
  printParamValue(p, value);
  p << ">";
}

void ParamDeclareOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  walkDef(getParamDecl(), getValue());
}

/// Verify that the type of the declaration matches the type of the attribute.
LogicalResult ParamDeclareOp::verify() {
  if (getParamDecl().getType() == getValue().getType())
    return success();
  return emitOpError("declares a parameter with type ")
         << getParamDecl().getType() << " but parameter expression has type "
         << getValue().getType();
}

//===----------------------------------------------------------------------===//
// ParamDeclareRegionOp
//===----------------------------------------------------------------------===//

static ParseResult parseRegionOnly(OpAsmParser &p,
                                   ParamDeclArrayAttr &inputParams,
                                   TypeAttr &functionType, TypeAttr &type,
                                   InlineLevelAttr &inlineLevel, Region &body) {
  SmallVector<OpAsmParser::Argument> args;
  FunctionType functionTypeValue;
  FuncTypeGeneratorType sigGenType;
  llvm::SMLoc bodyLoc;
  ParamDeclArrayAttr resultParams;
  if (parseFunctionFuncTypeGenerator(p, args, inputParams, resultParams,
                                     functionTypeValue, sigGenType) ||
      parseOptionalInline(p, inlineLevel) || p.getCurrentLocation(&bodyLoc) ||
      p.parseRegion(body, args))
    return failure();
  if (!resultParams.empty())
    return p.emitError(p.getCurrentLocation(), "invalid result parameters");

  // Form the Signature.
  SmallVector<Type> argTypes;
  for (const OpAsmParser::Argument &arg : args)
    argTypes.push_back(arg.type);
  functionType = TypeAttr::get(functionTypeValue);
  type = TypeAttr::get(sigGenType);
  return success();
}

static ParseResult
parseRegionDeclaration(OpAsmParser &p, ParamDeclAttr &paramDecl,
                       ParamDeclArrayAttr &inputParams, TypeAttr &functionType,
                       TypeAttr &type, InlineLevelAttr &inlineLevel,
                       Region &body) {
  StringAttr paramName;

  if (parseParamName(p, paramName) || p.parseEqual() ||
      parseRegionOnly(p, inputParams, functionType, type, inlineLevel, body))
    return failure();
  paramDecl = ParamDeclAttr::get(paramName, type.getValue());
  return success();
}

static void printRegionOnly(OpAsmPrinter &p, Operation *op,
                            ParamDeclArrayAttr inputParams,
                            TypeAttr functionType, TypeAttr type,
                            InlineLevelAttr inlineLevel, Region &body) {
  printFunctionFuncTypeGenerator(p, &body, inputParams, {},
                                 cast<FunctionType>(functionType.getValue()),
                                 cast<FuncTypeGeneratorType>(type.getValue()));
  printOptionalInline(p, inlineLevel.getValue());
  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

static void printRegionDeclaration(OpAsmPrinter &p, Operation *op,
                                   ParamDeclAttr paramDecl,
                                   ParamDeclArrayAttr inputParams,
                                   TypeAttr functionType, TypeAttr type,
                                   InlineLevelAttr inlineLevel, Region &body) {
  printParamName(p, paramDecl.getName());
  p << " = ";
  printRegionOnly(p, op, inputParams, functionType, type, inlineLevel, body);
}

bool ParamDeclareRegionOp::isIsolatedFromAbove(unsigned regionNum) {
  assert(regionNum == 0);
  return getIsolated();
}

void ParamDeclareRegionOp::notifyKnownIsolatedFromAbove(unsigned regionNum) {
  assert(regionNum == 0);
  setIsolated(true);
}

void ParamDeclareRegionOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  walkDef(getParamDecl(), &getBodyRegion());
}

/// This operation has no uses to collect in its current scope.
void ParamDeclareRegionOp::collectParameterUses(
    function_ref<void(Attribute)> scanAttr, function_ref<void(Type)> scanType) {
}

LogicalResult ParamDeclareRegionOp::verify() {
  if (ArrayAttr argsArray = getLLVMArgMetadataArray();
      !argsArray.empty() && argsArray.size() != getNumArguments())
    return emitOpError("LLVMArgMetadataArray size does not equal number of "
                       "arguments, got ")
           << argsArray.size();
  return success();
}

//===----------------------------------------------------------------------===//
// ParamApplyOp
//===----------------------------------------------------------------------===//

static ParseResult parseParamApplyOp(AsmParser &p, ParamDeclAttr &paramDecl,
                                     TypedAttr &callee,
                                     ParameterExprArrayAttr &operands) {
  StringAttr paramName;
  FuncTypeGeneratorType calleeType;
  SmallVector<TypedAttr> operandValues;
  llvm::SMLoc sigLoc;
  if (parseParamName(p, paramName) || p.parseEqual() || p.parseLSquare() ||
      parseKGENType(p, calleeType) || p.parseColon() ||
      p.getCurrentLocation(&sigLoc) || parseParamValue(p, callee, calleeType) ||
      p.parseRSquare() || p.parseLParen() ||
      failableInterleave(
          calleeType.getBody().getArguments(),
          [&](Type type) {
            return parseParamValue(p, operandValues.emplace_back(), type);
          },
          [&] { return p.parseComma(); }) ||
      p.parseRParen())
    return failure();
  if (calleeType.getBody().getNumResults() != 1)
    return p.emitError(sigLoc, "expected callee to have 1 result");
  paramDecl =
      ParamDeclAttr::get(paramName, calleeType.getBody().getResults().front());
  operands = ParameterExprArrayAttr::get(p.getContext(), operandValues);
  return success();
}

static void printParamApplyOp(AsmPrinter &p, Operation *op,
                              ParamDeclAttr paramDecl, TypedAttr callee,
                              ParameterExprArrayAttr operands) {
  printParamName(p, paramDecl.getName());
  p << " = [";
  printKGENType(p, callee.getType());
  p << ": ";
  printParamValue(p, callee);
  p << "](";
  llvm::interleaveComma(operands, p,
                        [&](TypedAttr value) { printParamValue(p, value); });
  p << ')';
}

LogicalResult ParamApplyOp::verify() {
  auto type = cast<FuncTypeGeneratorType>(getCallee().getType());
  if (type.getInputParamTypes().empty())
    return success();
  return emitOpError("callee signature must be concrete");
}

void ParamApplyOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  ParamDefValue def;
  def.exprs.push_back(getCallee());
  llvm::append_range(def.exprs, getOperands());
  walkDef(getParamDecl(), def);
}

void ParamApplyOp::walkDeclarations(
    function_ref<void(ParamDeclAttr)> walkDecl) {
  walkDecl(getParamDecl());
}

void ParamApplyOp::renameDeclarations(ArrayRef<ParamDeclAttr> decls) {
  assert(decls.size() == 1);
  setParamDeclAttr(decls.front());
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

bool ReturnOp::isParentNode(Operation *op) { return isa<FunctionLike>(op); }

void ReturnOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  targets.emplace_back(std::nullopt, getOperands());
}

LogicalResult ReturnOp::verify() {
  auto func = (*this)->getParentOfType<KGEN::FunctionLike>();
  if (!func)
    return emitOpError("expected to be nested inside a function");
  return checkOperandTypes(*this, func.getResultTypes());
}

//===----------------------------------------------------------------------===//
// UnreachableOp
//===----------------------------------------------------------------------===//

/// Unreachable can terminate any control flow operation.
bool UnreachableOp::isParentNode(Operation *op) { return true; }

/// No branch targets.
void UnreachableOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {}

//===----------------------------------------------------------------------===//
// ParamAssertOp
//===----------------------------------------------------------------------===//

/// This operation defines no parameters.
void ParamAssertOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {}

/// This operation is implicitly parametric.
bool ParamAssertOp::isImplicitlyParametric() { return true; }

//===----------------------------------------------------------------------===//
// GeneratorOp
//===----------------------------------------------------------------------===//

static ParseResult parseGeneratorOp(OpAsmParser &p, ExportKindAttr &exportKind,
                                    StringAttr &name, TypeAttr &signatureAttr,
                                    TypeAttr &functionTypeAttr,
                                    ParamDeclArrayAttr &inputParams,
                                    InlineLevelAttr &inlineLevel,
                                    DecoratorsAttr &decorators,
                                    NamedAttrList &attrs, Region &body) {
  if (parseSymbolExport(p, exportKind) || p.parseSymbolName(name))
    return failure();

  SmallVector<OpAsmParser::Argument> args;
  FuncTypeGeneratorType signature;
  FunctionType functionType;
  ParamDeclArrayAttr resultParams;
  if (parseFunctionFuncTypeGenerator(p, args, inputParams, resultParams,
                                     functionType, signature))
    return failure();
  if (!resultParams.empty())
    return p.emitError(p.getCurrentLocation(), "invalid result parameters");

  signatureAttr = TypeAttr::get(signature);
  functionTypeAttr = TypeAttr::get(functionType);

  if (parseOptionalInline(p, inlineLevel) ||
      parseOptionalDecorators(p, decorators) ||
      p.parseOptionalAttrDictWithKeyword(attrs) ||
      p.parseRegion(body, args, /*enableNameShadowing=*/true))
    return failure();
  return success();
}

static void printGeneratorOp(OpAsmPrinter &p, Operation *op,
                             ExportKindAttr exportKind, StringAttr name,
                             TypeAttr signature, TypeAttr functionType,
                             ParamDeclArrayAttr inputParams,
                             InlineLevelAttr inlineLevel,
                             DecoratorsAttr decorators, DictionaryAttr attrs,
                             Region &body) {
  printSymbolExport(p, op, exportKind);
  p << ' ';
  p.printSymbolName(name);
  printFunctionFuncTypeGenerator(
      p, &body, inputParams, /*resultParams=*/{},
      cast<FunctionType>(functionType.getValue()),
      cast<FuncTypeGeneratorType>(signature.getValue()));
  printOptionalInline(p, inlineLevel.getValue());
  printOptionalDecorators(p, op, decorators);

  auto gen = cast<GeneratorOp>(op);
  SmallVector<StringRef, 10> elidedAttrs{
      gen.getExportKindAttrName(),        gen.getSymNameAttrName(),
      gen.getFuncTypeGeneratorAttrName(), gen.getFunctionTypeAttrName(),
      gen.getInputParamsAttrName(),       gen.getInlineLevelAttrName(),
      gen.getDecoratorsAttrName()};
  ArrayAttr emptyArray = ArrayAttr::get(op->getContext(), {});
  if (attrs.get(gen.getLLVMMetadataArrayAttrName()) == emptyArray)
    elidedAttrs.push_back(gen.getLLVMMetadataArrayAttrName());
  if (attrs.get(gen.getLLVMArgMetadataArrayAttrName()) == emptyArray)
    elidedAttrs.push_back(gen.getLLVMArgMetadataArrayAttrName());
  p.printOptionalAttrDictWithKeyword(attrs.getValue(), elidedAttrs);

  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

LogicalResult GeneratorOp::verify() {
  if (ArrayAttr argsArray = getLLVMArgMetadataArray();
      !argsArray.empty() && argsArray.size() != getNumArguments())
    return emitOpError("LLVMArgMetadataArray size does not equal number of "
                       "arguments, got ")
           << argsArray.size();
  return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

static ParseResult parseFuncOp(OpAsmParser &p, ExportKindAttr &exportKind,
                               StringAttr &name, TypeAttr &signature,
                               InlineLevelAttr &inlineLevel,
                               DecoratorsAttr &decorators, NamedAttrList &attrs,
                               Region &body) {
  if (parseSymbolExport(p, exportKind) || p.parseSymbolName(name))
    return failure();

  SmallVector<OpAsmParser::Argument> args;
  FunctionType functionType;
  SmallVector<ArgConvention> conventions;
  FnEffects effects;
  auto parseArg = [&](SmallVectorImpl<Type> &argTypes) -> ParseResult {
    OpAsmParser::Argument &arg = args.emplace_back();
    if (p.parseArgument(arg, /*allowType=*/true) ||
        parseArgConvention(p, conventions.emplace_back()))
      return failure();
    argTypes.push_back(arg.type);
    return success();
  };
  llvm::SMLoc loc = p.getCurrentLocation();
  if (parseSignatureValues(p, parseArg, functionType, effects,
                           /*optionalResultList=*/true))
    return failure();
  auto sig = FuncType::getChecked([&] { return p.emitError(loc); },
                                  functionType, conventions, effects);
  if (!sig)
    return failure();
  signature = TypeAttr::get(GeneratorType::get(/*inputParamTypes=*/{}, sig));

  if (parseOptionalInline(p, inlineLevel) ||
      parseOptionalDecorators(p, decorators) ||
      p.parseOptionalAttrDictWithKeyword(attrs) ||
      p.parseRegion(body, args, /*enableNameShadowing=*/true))
    return failure();
  return success();
}

static void printFuncOp(OpAsmPrinter &p, Operation *op,
                        ExportKindAttr exportKind, StringAttr name,
                        TypeAttr signature, InlineLevelAttr inlineLevel,
                        DecoratorsAttr decorators, DictionaryAttr attrs,
                        Region &body) {
  FuncType sig = cast<FuncTypeGeneratorType>(signature.getValue()).getBody();
  auto func = cast<FuncOp>(op);

  printSymbolExport(p, op, exportKind);
  p << ' ';
  p.printSymbolName(name);
  auto printArg = [&](unsigned i) {
    p.printRegionArgument(body.getArgument(i));
    printArgConvention(p, sig.getArgConvention(i));
  };
  printSignatureValues(p, printArg, sig.getValues(), sig.getArgConventions(),
                       sig.getFnEffects(),
                       /*optionalResultList=*/true);
  printOptionalInline(p, inlineLevel.getValue());
  printOptionalDecorators(p, op, decorators);

  SmallVector<StringRef, 8> elidedAttrs{
      func.getExportKindAttrName(), func.getSymNameAttrName(),
      func.getFuncTypeGeneratorAttrName(), func.getInlineLevelAttrName(),
      func.getDecoratorsAttrName()};
  if (attrs.get(func.getLLVMMetadataAttrName()) ==
      DictionaryAttr::get(op->getContext()))
    elidedAttrs.push_back(func.getLLVMMetadataAttrName());
  if (attrs.get(func.getLLVMArgMetadataAttrName()) ==
      ArrayAttr::get(op->getContext(), {}))
    elidedAttrs.push_back(func.getLLVMArgMetadataAttrName());
  if (attrs.get(func.getCrossDeviceCapturesAttrName()) ==
      StringArrayAttr::get(func.getContext(), {}))
    elidedAttrs.push_back(func.getCrossDeviceCapturesAttrName());
  p.printOptionalAttrDictWithKeyword(attrs.getValue(), elidedAttrs);

  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

LogicalResult FuncOp::verify() {
  FunctionType type = getFunctionType();
  if (type.getNumInputs() != getNumArguments()) {
    return mlir::emitError(getLoc(), "'kgen.func' op function type expected ")
           << type.getNumInputs() << " arguments but region has "
           << getNumArguments();
  }
  for (auto [i, arg, type] :
       llvm::enumerate(getArguments(), type.getInputs())) {
    if (arg.getType() != type) {
      return mlir::emitError(arg.getLoc(), "'kgen.func' op argument #")
             << i << " type is " << arg.getType()
             << " but function type expected " << type;
    }
  }
  if (ArrayAttr argsArray = getLLVMArgMetadata();
      !argsArray.empty() && argsArray.size() != getNumArguments()) {
    return emitOpError("LLVMArgMetadataArray size does not equal number of "
                       "arguments, got ")
           << argsArray.size();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ExternGeneratorOp
//===----------------------------------------------------------------------===//

static ParseResult parseExternGenerator(OpAsmParser &p, TypeAttr &signature,
                                        TypeAttr &functionType,
                                        ParamDeclArrayAttr &inputParams) {
  SmallVector<OpAsmParser::Argument> args;
  FunctionType funcType;
  FuncTypeGeneratorType sigType;
  ParamDeclArrayAttr resultParams;
  if (parseFunctionFuncTypeGenerator(p, args, inputParams, resultParams,
                                     funcType, sigType))
    return failure();
  if (!resultParams.empty())
    return p.emitError(p.getCurrentLocation(), "invalid result parameters");
  functionType = TypeAttr::get(funcType);
  signature = TypeAttr::get(sigType);
  return success();
}

static void printExternGenerator(OpAsmPrinter &p, Operation *op,
                                 TypeAttr signature, TypeAttr functionType,
                                 ParamDeclArrayAttr inputParams) {
  printFunctionFuncTypeGenerator(
      p, /*region=*/nullptr, inputParams,
      /*resultParams=*/{}, cast<FunctionType>(functionType.getValue()),
      cast<FuncTypeGeneratorType>(signature.getValue()));
}

//===----------------------------------------------------------------------===//
// StructGeneratorOp
//===----------------------------------------------------------------------===//

static ParseResult parseStructGeneratorSpec(OpAsmParser &p,
                                            ParamDeclArrayAttr &inputParams,
                                            TypeAttr &valueDomainTypeAttr,
                                            TypeAttr &metaTypeAttr,
                                            Region &body) {
  SmallVector<ParamDeclAttr> paramAttrs;
  if (succeeded(p.parseOptionalLess()) &&
      (parseParamDeclAttrs(p, paramAttrs) || p.parseGreater()))
    return failure();
  inputParams = ParamDeclArrayAttr::get(p.getContext(), paramAttrs);

  // Derive generator type input params.
  SmallVector<Type> inputParamTypes(llvm::map_range(
      paramAttrs, [](ParamDeclAttr decl) { return decl.getType(); }));

  Type metaType;
  Type valueDomainType;
  if (parseColonTypeOrDefault(p, metaType, TypeType::get(p.getContext())) ||
      p.parseEqual() || parseKGENType(p, valueDomainType))
    return failure();
  valueDomainTypeAttr = TypeAttr::get(valueDomainType);
  metaTypeAttr = TypeAttr::get(metaType);

  auto bodyResult = p.parseOptionalRegion(body);
  if (bodyResult.has_value() && failed(*bodyResult))
    return failure();
  if (body.empty())
    body.emplaceBlock();
  return success();
}

static void printStructGeneratorSpec(OpAsmPrinter &p, Operation *op,
                                     ParamDeclArrayAttr inputParams,
                                     TypeAttr valueDomainTypeAttr,
                                     TypeAttr metaTypeAttr, Region &body) {
  if (inputParams && !inputParams.empty()) {
    p << '<';
    printParamDeclAttrs(p, inputParams);
    p << '>';
  }

  printColonTypeOrDefault(p, metaTypeAttr.getValue(),
                          TypeType::get(op->getContext()));
  p << " = ";
  printKGENType(p, valueDomainTypeAttr.getValue());

  if (!body.empty() && !body.front().empty())
    p.printRegion(body);
}

//===----------------------------------------------------------------------===//
// StructInstanceOp
//===----------------------------------------------------------------------===//

static ParseResult parseStructInstanceSpec(OpAsmParser &p,
                                           TypeAttr &valueDomainTypeAttr,
                                           TypeAttr &metaTypeAttr,
                                           Region &body) {
  ParamDeclArrayAttr params;
  mlir::SMLoc startLoc = p.getCurrentLocation();
  if (failed(parseStructGeneratorSpec(p, params, valueDomainTypeAttr,
                                      metaTypeAttr, body)))
    return failure();
  if (!params.empty())
    return p.emitError(startLoc, "struct.instance cannot be parameterized");
  return success();
}

static void printStructInstanceSpec(OpAsmPrinter &p, Operation *op,
                                    TypeAttr valueDomainTypeAttr,
                                    TypeAttr metaTypeAttr, Region &body) {
  printStructGeneratorSpec(p, op, {}, valueDomainTypeAttr, metaTypeAttr, body);
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

static ParseResult
parseCallOp(OpAsmParser &p, SymbolConstantAttr &calleeCst,
            SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
            SmallVectorImpl<Type> &operandTypes,
            SmallVectorImpl<Type> &resultTypes) {
  SymbolRefAttr callee;
  ParameterExprArrayAttr paramValues;
  if (p.parseAttribute(callee) || parseParameterValues(p, paramValues) ||
      p.parseOperandList(operands, AsmParser::Delimiter::Paren) ||
      p.parseColon())
    return failure();

  FuncTypeGeneratorType signature;
  FunctionType functionType;
  if (parseKGENFuncTypeGenerator(p, functionType, signature))
    return failure();
  calleeCst = SymbolConstantAttr::get(callee, signature, paramValues);
  llvm::append_range(operandTypes, functionType.getInputs());
  llvm::append_range(resultTypes, functionType.getResults());
  return success();
}

static void printCallOp(OpAsmPrinter &p, Operation *op,
                        SymbolConstantAttr calleeCst, ValueRange operands,
                        TypeRange operandTypes, TypeRange resultTypes) {
  p << calleeCst.getSymbol();
  printParameterValues(p, calleeCst.getParamValues());
  p << '(';
  p.printOperands(operands);
  p << ") : ";
  printSignatureValues(
      p, FunctionType::get(op->getContext(), operandTypes, resultTypes),
      calleeCst.getType());
}

void CallOp::concretizeCallee(IRRewriter &b, SymbolConstantAttr callee) {
  setCalleeAttr(callee);
}

void CallOp::setCalleeAttr(TypedAttr callee) {
  setCalleeAttr(cast<SymbolConstantAttr>(callee));
}

FailureOr<InlineResult> CallOp::prepInline(mlir::RewriterBase &b) {
  StringAttr label = b.getStringAttr("inlined_cf_scope");
  auto op =
      b.create<HLCF::LoopOp>(getLoc(), getResultTypes(), ValueRange(), label);
  return {{op, [label, &b](Operation *op) {
             b.replaceOpWithNewOp<HLCF::BreakOp>(op, op->getOperands(), label);
           }}};
}

static LogicalResult verifyCallOp(Operation *op, ValueRange args,
                                  FuncType signature) {
  if (failed(verifyCallOperands(op, args, signature)) ||
      failed(verifyCallResults(op, op->getResults(), signature)))
    return failure();
  return success();
}

LogicalResult CallOp::verify() {
  return verifyCallOp(*this, getOperands(), getCalleeType().getBody());
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

LogicalResult CallIndirectOp::verify() {
  return verifyCallOp(*this, getArguments(), getCallee().getType().getBody());
}

//===----------------------------------------------------------------------===//
// CallParamOp
//===----------------------------------------------------------------------===//

void CallParamOp::concretizeCallee(IRRewriter &b, SymbolConstantAttr callee) {
  b.replaceOpWithNewOp<CallOp>(*this, getResultTypes(), callee, getOperands());
}

FailureOr<InlineResult> CallParamOp::prepInline(mlir::RewriterBase &b) {
  // Inlining not supported for this op
  return failure();
}

LogicalResult CallParamOp::verify() {
  return verifyCallOp(*this, getOperands(), getCalleeType().getBody());
}

//===----------------------------------------------------------------------===//
// ParamForOp
//===----------------------------------------------------------------------===//

LogicalResult ParamForOp::verify() {
  if (getNumOperands() != getNumResults()) {
    return emitOpError("has ")
           << getNumOperands() << " operands but " << getNumResults()
           << " results; it should be the same";
  }
  for (auto [i, argTy, resTy] :
       llvm::enumerate(getOperandTypes(), getResultTypes())) {
    if (argTy == resTy)
      continue;
    return emitOpError("operand #")
           << i << " has type " << argTy
           << " but corresponding result has type " << resTy;
  }
  return success();
}

void ParamForOp::getEntryTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  targets.emplace_back(0, getOperands());
  targets.emplace_back(1, getOperands());
}

ValueRange ParamForOp::getEntryArguments(std::optional<unsigned> target) {
  if (!target)
    return getResults();
  if (*target == 0)
    return getBody().getArguments();
  assert(*target == 1);
  return getElseRegion().getArguments();
}

ArrayRef<ParamDeclAttr> ParamForOp::getInputParams() {
  // HACK: The interface requires an ArrayRef, but we only have a single
  // element. Returning `getParamDecl` will cause a reference to a temporary to
  // be formed. Grab the reference directly from the DictionaryAttr. We know
  // alphabetically it will be last attribute.
  assert((*this)->getRegisteredInfo()->getAttributeNames().back() ==
         getParamDeclAttrName());
  static_assert(sizeof(std::pair<Attribute, ParamDeclAttr>) ==
                        sizeof(NamedAttribute) &&
                    alignof(std::pair<Attribute, ParamDeclAttr>) ==
                        alignof(NamedAttribute),
                "hack doesn't work");
  return {&((const std::pair<Attribute, ParamDeclAttr> *)&(*this)
                ->getAttrs()
                .back())
               ->second,
          1};
}

void ParamForOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {}

bool ParamForOp::isImplicitlyParametric() { return true; }

void ParamForOp::collectParameterUsesBelow(
    function_ref<void(Attribute)> scanAttr, function_ref<void(Type)> scanType) {
}

bool ParamForOp::isIsolatedFromAbove(unsigned regionNum) {
  if (regionNum == 0)
    return getBodyIsolated();
  assert(regionNum == 1);
  return getElseIsolated();
}

void ParamForOp::notifyKnownIsolatedFromAbove(unsigned regionNum) {
  if (regionNum == 0)
    return setBodyIsolated(true);
  assert(regionNum == 1);
  return setElseIsolated(true);
}

bool ParamForBreakOp::isParentNode(Operation *op) {
  return isa<ParamForOp>(op);
}

void ParamForBreakOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  // Branch to after the loop operation.
  targets.emplace_back(std::nullopt, getOperands());
}

bool ParamForContinueOp::isParentNode(Operation *op) {
  return isa<ParamForOp>(op);
}

void ParamForContinueOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  // Branch to the beginning of the body region.
  targets.emplace_back(0, getOperands());
  targets.emplace_back(1, getOperands());
}

//===----------------------------------------------------------------------===//
// ParamIfOp
//===----------------------------------------------------------------------===//

bool ParamIfOp::isIsolatedFromAbove(unsigned regionNum) {
  switch (regionNum) {
  case 0:
    return getThenIsolated();
  case 1:
    return getElseIsolated();
  default:
    llvm_unreachable("unknown region number");
  }
}

void ParamIfOp::notifyKnownIsolatedFromAbove(unsigned regionNum) {
  switch (regionNum) {
  case 0:
    setThenIsolated(true);
    break;
  case 1:
    setElseIsolated(true);
    break;
  default:
    llvm_unreachable("unknown region number");
  }
}

void ParamIfOp::getEntryTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.empty());
  targets.emplace_back(0);
  targets.emplace_back(1);
}

ValueRange ParamIfOp::getEntryArguments(std::optional<unsigned> target) {
  if (!target)
    return getResults();
  assert(*target == 0 || *target == 1);
  return {};
}

void ParamIfOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {}

bool ParamIfOp::isImplicitlyParametric() { return true; }

/// This operation has no uses to collect in the scopes it defines.
void ParamIfOp::collectParameterUsesBelow(
    function_ref<void(Attribute)> scanAttr, function_ref<void(Type)> scanType) {
}

//===----------------------------------------------------------------------===//
// ParamYieldOp
//===----------------------------------------------------------------------===//

bool ParamYieldOp::isParentNode(Operation *op) {
  return isa<ParamForOp, ParamIfOp>(op);
}

void ParamYieldOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  // Branch to after the if operation.
  targets.emplace_back(std::nullopt, getOperands());
}

//===----------------------------------------------------------------------===//
// StageClosureOp
//===----------------------------------------------------------------------===//

static ParseResult parseStageClosureOp(OpAsmParser &p, Type &resultType,
                                       Region &body) {
  // we expect the following syntax:
  // kgen.stage_closure = () capturing -> index {
  // } { name = foo }
  FuncTypeGeneratorType signatureType;
  ParamDeclArrayAttr inputParams;
  ParamDeclArrayAttr resultParams;
  FunctionType functionTypeValue;
  SmallVector<OpAsmParser::Argument> args;
  llvm::SMLoc bodyLoc;
  if (p.parseEqual() ||
      parseFunctionFuncTypeGenerator(p, args, inputParams, resultParams,
                                     functionTypeValue, signatureType) ||
      p.getCurrentLocation(&bodyLoc) || p.parseRegion(body, args))
    return failure();
  if (!inputParams.empty() || !resultParams.empty())
    return p.emitError(bodyLoc, "staged closures cannot have parameters");
  resultType = signatureType;
  return success();
}

static void printStageClosureOp(OpAsmPrinter &p, Operation *op,
                                FuncTypeGeneratorType resultType,
                                Region &body) {
  p << "= ";
  printFunctionFuncTypeGenerator(p, &body, {}, {},
                                 resultType.getBody().getValues(), resultType);
  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// CreateClosureOp
//===----------------------------------------------------------------------===//

LogicalResult
CreateClosureOp::inferReturnTypes(MLIRContext *ctx, std::optional<Location> loc,
                                  Adaptor adaptor,
                                  SmallVectorImpl<Type> &results) {
  auto callee = dyn_cast_or_null<TypedAttr>(adaptor.getCalleeAttr());
  if (!callee) {
    return mlir::emitOptionalError(
        loc, "'create_closure' expected TypedAttr 'callee'");
  }
  auto sigGen = dyn_cast<FuncTypeGeneratorType>(callee.getType());
  if (!sigGen) {
    return mlir::emitOptionalError(
        loc,
        "'create_closure' attribute 'callee' must have FuncTypeGeneratorType");
  }

  ValueRange captures = adaptor.getOperands();
  FuncType sig = sigGen.getBody();
  unsigned numCaptures = captures.size();
  if (numCaptures > sig.getNumArguments()) {
    return mlir::emitOptionalError(loc, "provided ", numCaptures,
                                   " operands but callee only has ",
                                   sig.getNumArguments(), " to bind");
  }

  ArrayRef<Type> newArgTypes = sig.getArguments().drop_front(numCaptures);
  ArrayRef<ArgConvention> newArgConvs =
      sig.getArgConventions().drop_front(numCaptures);

  FnEffects effects = sig.getFnEffects();
  if (!captures.empty())
    effects.setCapturing();
  results.push_back(FuncTypeGeneratorType::get(
      sigGen.getInputParamTypes(),
      Builder(ctx).getFunctionType(newArgTypes, sig.getResults()), newArgConvs,
      effects,
      sig.getMetadata() ? sig.getMetadata().getWithBoundPosArgs(numCaptures)
                        : nullptr,
      sigGen.getMetadata()));
  return mlir::success();
}

static ParseResult
parseClosureCaptureTypes(AsmParser &p, TypedAttr callee,
                         ArrayRef<OpAsmParser::UnresolvedOperand> captures,
                         SmallVectorImpl<Type> &captureTypes) {
  auto sigGen = dyn_cast<FuncTypeGeneratorType>(callee.getType());
  if (!sigGen) {
    return p.emitError(p.getCurrentLocation(),
                       "expected type of callee to be FuncTypeGeneratorType");
  }
  FuncType sig = sigGen.getBody();

  unsigned numCaptures = captures.size();
  if (numCaptures > sig.getNumArguments()) {
    return p.emitError(p.getCurrentLocation(), "provided ")
           << numCaptures << " operands but callee only has "
           << sig.getNumArguments() << " to bind";
  }

  ArrayRef<Type> inputs = sig.getArguments().take_front(numCaptures);
  captureTypes.append(inputs.begin(), inputs.end());
  return success();
}

static void printClosureCaptureTypes(AsmPrinter &p, Operation *,
                                     TypedAttr callee, ValueRange captures,
                                     TypeRange captureTypes) {}

LogicalResult CreateClosureOp::verify() {
  FuncType calleeSig = getCalleeType().getBody();
  if (getNumOperands() > calleeSig.getNumArguments()) {
    return emitOpError("provided ")
           << getNumOperands() << " operands but callee only has "
           << calleeSig.getNumArguments() << " to bind";
  }
  unsigned expectedArgs = calleeSig.getNumArguments() - getNumOperands();
  FuncType sig = getType().getBody();
  if (sig.getNumArguments() != expectedArgs) {
    return emitOpError("result signature has ")
           << sig.getNumArguments() << " arguments but expected "
           << expectedArgs;
  }

  for (auto [i, type, argType] :
       llvm::enumerate(getOperandTypes(),
                       calleeSig.getArguments().take_front(getNumOperands()))) {
    if (type != argType) {
      return emitOpError("operand #")
             << i << " has type " << type
             << " but callee argument type expected " << argType;
    }
  }
  for (auto [i, type, argType] :
       llvm::enumerate(sig.getArguments(),
                       calleeSig.getArguments().drop_front(getNumOperands()))) {
    if (type != argType) {
      return emitOpError("result signature argument #")
             << i << " type is " << argType << " but expected to be " << type;
    }
  }

  if (!getCaptures().empty() && !sig.isCapturing())
    return emitOpError("has captures, so result signature must be 'capturing'");
  return success();
}

FailureOr<InlineResult> CreateClosureOp::prepInline(mlir::RewriterBase &b) {
  auto op = b.create<StageClosureOp>(getLoc(), getType());
  return {{op, [](Operation *) {}}};
}

//===----------------------------------------------------------------------===//
// CreateRegStubOp
//===----------------------------------------------------------------------===//

LogicalResult CreateRegStubOp::verify() {
  FuncType calleeSig =
      cast<FuncTypeGeneratorType>(getCallee().getType()).getBody();
  FuncType resSig = getType().getBody();

  if (calleeSig.isThrows() || resSig.isThrows())
    return emitOpError("throwing function not supported");
  for (Type ty : resSig.getResults())
    if (!isa<NoneType>(ty))
      return emitOpError("result signature with output types not suported");

  bool expectPromotedMemOutputs =
      resSig.hasMemoryOnlyResult() && !calleeSig.hasMemoryOnlyResult();
  unsigned expectedArgsCount =
      calleeSig.getNumArguments() + unsigned(expectPromotedMemOutputs);
  if (resSig.getNumArguments() != expectedArgsCount) {
    return emitOpError("result signature has ")
           << resSig.getNumArguments()
           << " arguments, but the expected count is " << expectedArgsCount;
  }

  for (unsigned i = 0, e = resSig.getNumArguments(); i < e; ++i) {
    Type argTy = getOriginalArgType(i);
    Type calleeTy = getCalleeArgType(i);
    if (argTy == calleeTy)
      continue;

    PointerType argPtrTy = dyn_cast<PointerType>(argTy);
    if (!argPtrTy || argPtrTy.getElementType() != calleeTy) {
      return emitOpError("result signature argument #")
             << i << " type is " << argTy
             << " but callee signature argument is " << calleeTy;
    }
  }

  return success();
}

void CreateRegStubOp::build(mlir::OpBuilder &builder,
                            mlir::OperationState &state,
                            mlir::TypedAttr callee) {
  FuncType resultSig = getStubSignatureType(
      cast<FuncTypeGeneratorType>(callee.getType()).getBody());
  auto resultTy = GeneratorType::get(/*inputParamTypes=*/{}, resultSig);
  build(builder, state, resultTy, callee);
}

FuncType CreateRegStubOp::getStubSignatureType(FuncType calleeSign) {
  FunctionType values = calleeSign.getValues();

  // Check if type is a memory type that can be promoted to value.
  // These types will be wrapped in a memory struct.
  auto canLowerToRegPassable = [](Type ty, ArgConvention conv) {
    if (!hasAddress(conv))
      return false;

    PointerType ptrTy = dyn_cast<PointerType>(ty);
    if (!ptrTy)
      return false;
    auto structElemTy = dyn_cast<StructType>(ptrTy.getElementType());
    if (structElemTy && structElemTy.getIsMemoryOnly())
      return false;

    return true;
  };

  SmallVector<Type> newArgTypes;
  for (unsigned i = 0, e = values.getNumInputs(); i < e; ++i) {
    Type argTy = values.getInput(i);
    // Replace register-passable `!kgen.pointer<T> owned_in_mem` with
    // `!kgen.pointer<struct<(T) memoryOnly>> owned_in_mem`:
    // - It guarantees the pointer arguments won't be lowered to by-value.
    // - It also tells LLVM that arguments don't alias.
    if (canLowerToRegPassable(argTy, calleeSign.getArgConvention(i))) {
      PointerType ptrTy = cast<PointerType>(argTy);
      newArgTypes.push_back(PointerType::get(
          StructType::get(calleeSign.getContext(), ptrTy.getElementType(),
                          /*isMemoryOnly=*/true)));
    } else {
      // Other types aren't changed.
      newArgTypes.push_back(argTy);
    }
  }

  return FuncType::get(FunctionType::get(calleeSign.getContext(), newArgTypes,
                                         values.getResults()),
                       calleeSign.getArgConventions());
}

Type CreateRegStubOp::getOriginalArgType(unsigned index) {
  Type rawArgTy = getType().getBody().getValues().getInput(index);
  Type calleeArgTy = getCalleeArgType(index);
  // The type isn't transformed if it's identical to callee.
  if (rawArgTy == calleeArgTy)
    return rawArgTy;

  // Wrapped types are memory types of the form `pointer<struct<(T)
  // memoryOnly>`.
  if (!hasAddress(getType().getBody().getArgConvention(index)))
    return rawArgTy;

  PointerType ptrTy = dyn_cast<PointerType>(rawArgTy);
  if (!ptrTy)
    return rawArgTy;
  auto structElemTy = dyn_cast<StructType>(ptrTy.getElementType());
  if (!structElemTy || structElemTy.getNumElements() != 1 ||
      !structElemTy.getIsMemoryOnly())
    return rawArgTy;

  // Returns pointer<T>.
  return PointerType::get(structElemTy.getElementTypes()[0]);
}

Type CreateRegStubOp::getCalleeArgType(unsigned index) {
  // Some arguments might be promoted to outputs.
  FuncType calleeSigBase = getCalleeSignature().getBody();
  FuncType resSig = getType().getBody();
  bool promotedOutputs =
      resSig.hasMemoryOnlyResult() && !calleeSigBase.hasMemoryOnlyResult();
  if (!promotedOutputs)
    return calleeSigBase.getValues().getInput(index);

  ArgConvention conv = resSig.getArgConvention(index);
  // If `conv` is ByRefResult, the promoted output has to be this argument.
  if (conv == ArgConvention::ByRefResult)
    return calleeSigBase.getValues().getResult(0);

  // A different argument is promoted.
  return calleeSigBase.getValues().getInput(index);
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalOp::verify() {
  if (getCtor() || getDtor() || getPriority()) {
    if (!getCtor() || !getDtor() || !getPriority()) {
      return emitOpError("does not define all of the constructor, destructor, "
                         "or priority values, if one of these values is "
                         "defined, then all must be defined");
    }
  }
  return success();
}

LogicalResult GlobalOp::verifySymbolUses(SymbolTableCollection &symtab) {
  auto module = (*this)->getParentOfType<ModuleOp>();
  auto verifyFunc = [&](SymbolRefAttr ref, StringRef name) -> LogicalResult {
    auto func = symtab.lookupSymbolIn<mlir::FunctionOpInterface>(module, ref);
    if (!func || func.getNumArguments() != 0) {
      return emitOpError() << name << ' ' << ref
                           << " does not reference a function with zero "
                              "arguments and zero results";
    }
    return success();
  };
  if (!getCtor() || !getDtor())
    return success();
  if (failed(verifyFunc(*getCtor(), "constructor")) ||
      failed(verifyFunc(*getDtor(), "destructor")))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// GlobalAddressOp
//===----------------------------------------------------------------------===//

LogicalResult GlobalAddressOp::verifySymbolUses(SymbolTableCollection &symtab) {
  auto global = symtab.lookupSymbolIn<GlobalOp>(
      (*this)->getParentOfType<ModuleOp>(), getGlobal());
  if (!global)
    return emitOpError("does not reference a `pop.global` operation");
  if (global.getType() != getResult().getType().getElementType())
    return emitOpError("result type does not match global type ")
           << global.getType();
  return success();
}

//===----------------------------------------------------------------------===//
// PackCreateOp
//===----------------------------------------------------------------------===//

/// Parses a kgen.pack.create op.
///
/// operation ::=
///   `kgen.pack.create` `(` operands `)` attr-dict `:` result-type
///
/// This is custom because we need to match operands at each index to the
/// resulting pack type element at that index.
static ParseResult parsePackCreateType(AsmParser &p, Type &resultType,
                                       SmallVectorImpl<Type> &elementTypes) {
  llvm::SMLoc loc = p.getCurrentLocation();
  if (p.parseType(resultType))
    return failure();
  auto type = dyn_cast<PackType>(resultType);
  if (!type)
    return p.emitError(loc, "expected a pack type");

  auto variadic = type.getVariadicIfResolved();
  if (!variadic) {
    // We can only infer if we know the elements of the pack type (i.e.: it is
    // backed by a variadic attribute).
    return p.emitError(loc) << "operand types cannot be "
                               "inferred for resulting pack type "
                            << type;
  }

  ArrayRef<TypedAttr> values = variadic.getValues();
  for (TypedAttr value : values)
    elementTypes.push_back(ParamType::get(value));
  return success();
}

static void printPackCreateType(OpAsmPrinter &p, Operation *op, Type resultType,
                                TypeRange elementTypes) {
  p << resultType;
}

LogicalResult PackCreateOp::verify() {
  VariadicAttr elementTypesAttr = getType().getVariadicIfResolved();
  if (!elementTypesAttr)
    return emitOpError() << "cannot create pack with parametric element types";
  ArrayRef<TypedAttr> elementTypes = elementTypesAttr.getValues();
  if (elementTypes.size() != getNumOperands()) {
    return emitOpError() << "expected " << elementTypes.size()
                         << " operands, but got " << getNumOperands();
  }
  for (auto [i, expected, provided] :
       llvm::enumerate(elementTypes, getOperandTypes())) {
    Type type = ParamType::get(expected);
    if (type == provided)
      continue;
    return emitOpError() << "operand #" << i << " should have type " << type
                         << " but got " << provided;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PackExtractOp
//===----------------------------------------------------------------------===//

/// Given a packtype, return the type of the field at the specified index, which
/// may be parametric.
static Type getPackFieldAtIndex(PackType packType, TypedAttr index) {
  // The result type is the type extracted from the type list.  Extract the
  // element from the type list.  This automatically folds if constant.
  auto typeAttr =
      ParamOperatorAttr::get(POC::VariadicGet, packType.getVariadic(), index);
  return ParamType::get(typeAttr);
}

LogicalResult
PackExtractOp::inferReturnTypes(MLIRContext *context,
                                std::optional<Location> loc, Adaptor adaptor,
                                SmallVectorImpl<Type> &inferredReturnTypes) {
  auto emitError = [&](const Twine &msg) -> LogicalResult {
    return mlir::emitOptionalError(loc, msg);
  };

  ValueRange operands = adaptor.getOperands();
  if (operands.size() != 1 || !isa<PackType>(adaptor.getPack().getType()))
    return emitError("expected 1 operand");

  auto indexAttr = dyn_cast_if_present<TypedAttr>(adaptor.getIndexAttr());
  if (!indexAttr || !indexAttr.getType().isIndex())
    return emitError("expected an index attribute");

  auto packType = cast<PackType>(adaptor.getPack().getType());
  inferredReturnTypes.push_back(getPackFieldAtIndex(packType, indexAttr));
  return success();
}

//===----------------------------------------------------------------------===//
// PackGEPOp
//===----------------------------------------------------------------------===//

LogicalResult
PackGEPOp::inferReturnTypes(MLIRContext *context, std::optional<Location> loc,
                            Adaptor adaptor,
                            SmallVectorImpl<Type> &inferredReturnTypes) {
  auto emitError = [&](const Twine &msg) -> LogicalResult {
    return mlir::emitOptionalError(loc, msg);
  };

  ValueRange operands = adaptor.getOperands();
  if (operands.size() != 1 || !isa<PointerType>(adaptor.getPack().getType()))
    return emitError("expected 1 operand");
  auto packType = dyn_cast<PackType>(
      cast<PointerType>(adaptor.getPack().getType()).getElementType());
  if (!packType)
    return emitError("expected pointer to pack type");

  auto indexAttr = dyn_cast_if_present<TypedAttr>(adaptor.getIndexAttr());
  if (!indexAttr || !indexAttr.getType().isIndex())
    return emitError("expected an index attribute");

  inferredReturnTypes.push_back(
      PointerType::get(getPackFieldAtIndex(packType, indexAttr)));
  return success();
}

//===----------------------------------------------------------------------===//
// PackLoadOp
//===----------------------------------------------------------------------===//

LogicalResult PackLoadOp::inferReturnTypes(MLIRContext *ctx,
                                           std::optional<Location> loc,
                                           Adaptor adaptor,
                                           SmallVectorImpl<Type> &types) {
  if (!isa<PackType>(adaptor.getPack().getType()))
    return mlir::emitError(loc.value_or(adaptor.getPack().getLoc()),
                           "expected one !kgen.pack operand, not ")
           << adaptor.getPack().getType();
  auto packType = cast<PackType>(adaptor.getPack().getType());
  // The result type is the same as the input type, but with a layer of pointers
  // stripped off.
  auto mappedTypes =
      ParamOperatorAttr::get(POC::VariadicPtrRemoveMap, packType.getVariadic());
  types.push_back(PackType::get(mappedTypes));
  return success();
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

/// Verify the value type matches the struct element type at the given index.
static LogicalResult verifyStructValueType(Operation *op, StructType container,
                                           IntegerAttr indexAttr,
                                           Type valueType,
                                           StringRef valueKind) {
  ArrayRef<Type> elementTypes = container.getElementTypes();
  size_t index = indexAttr.getInt();
  if (index >= elementTypes.size())
    return op->emitOpError("element index ")
           << index << " out of bounds (>=" << elementTypes.size() << ")";
  if (elementTypes[index] != valueType) {
    return op->emitOpError(valueKind)
           << " type " << valueType
           << " does not match struct element type at index " << index << ": "
           << elementTypes[index];
  }
  return success();
}

LogicalResult StructExtractOp::verify() {
  return verifyStructValueType(*this, getContainer().getType(), getIndexAttr(),
                               getType(), "result");
}

template <typename OpT>
static FailureOr<Type>
inferStructElementType(function_ref<LogicalResult(const Twine &)> emitError,
                       StructType structType, typename OpT::Adaptor adaptor) {
  if (!structType)
    return emitError("expected struct operand");
  mlir::OperationName name(OpT::getOperationName(), structType.getContext());

  IntegerAttr indexAttr = adaptor.getIndexAttr();
  if (!indexAttr)
    return emitError("expected an integer index attribute");
  size_t index = indexAttr.getInt();
  if (index >= structType.getNumElements())
    return emitError("struct element index out of bounds");
  return structType.getElementTypes()[index];
}

LogicalResult StructExtractOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto emitError = [&](const Twine &msg) -> LogicalResult {
    return mlir::emitOptionalError(location, msg);
  };

  ValueRange operands = adaptor.getOperands();
  if (operands.size() != 1)
    return emitError("expected 1 operand");
  auto structType = dyn_cast<StructType>(operands.front().getType());
  FailureOr<Type> type =
      inferStructElementType<StructExtractOp>(emitError, structType, adaptor);
  if (succeeded(type))
    inferredReturnTypes.push_back(*type);
  return type;
}

//===----------------------------------------------------------------------===//
// StructReplaceOp
//===----------------------------------------------------------------------===//

static ParseResult parseStructValueType(AsmParser &p, Type &valueType,
                                        Type structType, IntegerAttr index) {
  ArrayRef<Type> elementTypes =
      llvm::cast<StructType>(structType).getElementTypes();
  if (index.getInt() > static_cast<int64_t>(elementTypes.size()))
    return p.emitError(p.getCurrentLocation(), "element index out of bounds (")
           << index.getInt() << " >= " << elementTypes.size() << ")";
  // Infer the value type from the struct type and index.
  valueType = elementTypes[index.getInt()];
  return success();
}

static void printStructValueType(AsmPrinter &p, Operation *op, Type valueType,
                                 Type structType, IntegerAttr index) {}

LogicalResult StructReplaceOp::verify() {
  return verifyStructValueType(*this, getContainer().getType(), getIndexAttr(),
                               getValue().getType(), "operand");
}

//===----------------------------------------------------------------------===//
// StructGEPOp
//===----------------------------------------------------------------------===//

LogicalResult StructGEPOp::verify() {
  return verifyStructValueType(
      *this, cast<StructType>(getContainer().getType().getElementType()),
      getIndexAttr(), getType().getElementType(), "result");
}

LogicalResult
StructGEPOp::inferReturnTypes(MLIRContext *context,
                              std::optional<Location> location, Adaptor adaptor,
                              SmallVectorImpl<Type> &inferredReturnTypes) {
  auto emitError = [&](const Twine &msg) -> LogicalResult {
    return mlir::emitOptionalError(location, msg);
  };

  ValueRange operands = adaptor.getOperands();
  if (operands.size() != 1)
    return emitError("expected 1 operand");
  auto pointerType = dyn_cast<PointerType>(adaptor.getContainer().getType());
  if (!pointerType)
    return emitError("expected pointer operand");
  auto structType = dyn_cast<StructType>(pointerType.getElementType());
  FailureOr<Type> type =
      inferStructElementType<StructGEPOp>(emitError, structType, adaptor);
  if (succeeded(type))
    inferredReturnTypes.push_back(PointerType::get(*type));
  return type;
}

//===----------------------------------------------------------------------===//
// VariantCreateOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyVariantIndex(Operation *op, VariantType type,
                                        unsigned index) {
  if (index < type.getNumTypes())
    return success();
  return op->emitOpError("variant index ")
         << index << " is out of bounds in range [0, " << type.getNumTypes()
         << ")";
}

LogicalResult VariantCreateOp::verify() {
  if (failed(verifyVariantIndex(*this, getType(), getIndex())))
    return failure();
  Type elementType = getType().getType(getIndex());
  if (elementType == getOperand().getType())
    return success();
  return emitOpError("variant element at index ")
         << getIndex() << " expected type " << elementType
         << " but operand has type " << getOperand().getType();
}

static ParseResult parseVariantElementType(AsmParser &p, Type &type,
                                           Type variantType,
                                           IntegerAttr index) {
  unsigned i = index.getInt();
  auto variant = cast<VariantType>(variantType);
  if (i >= variant.getNumTypes()) {
    return p.emitError(p.getCurrentLocation(),
                       "variant index is out of bounds: ")
           << i;
  }
  type = variant.getType(i);
  return success();
}

static void printVariantElementType(AsmPrinter &p, Operation *op, Type type,
                                    Type variantType, IntegerAttr index) {}

//===----------------------------------------------------------------------===//
// VariantIsOp
//===----------------------------------------------------------------------===//

LogicalResult VariantIsOp::verify() {
  return verifyVariantIndex(*this, getVariant().getType(), getIndex());
}

//===----------------------------------------------------------------------===//
// VariantGetOp
//===----------------------------------------------------------------------===//

LogicalResult VariantGetOp::verify() {
  if (failed(verifyVariantIndex(*this, getVariant().getType(), getIndex())))
    return failure();
  Type elementType = getVariant().getType().getType(getIndex());
  if (elementType == getType())
    return success();
  return emitOpError("variant element at index ")
         << getIndex() << " expected type " << elementType
         << " but operand has type " << getType();
}

LogicalResult VariantGetOp::inferReturnTypes(MLIRContext *,
                                             std::optional<Location> loc,
                                             Adaptor adaptor,
                                             SmallVectorImpl<Type> &types) {
  unsigned index = adaptor.getIndex();
  auto variant = cast<VariantType>(adaptor.getVariant().getType());
  if (index >= variant.getNumTypes())
    return mlir::emitOptionalError(loc, "variant element index ", index,
                                   " is out of bounds");
  types.push_back(variant.getType(index));
  return success();
}

//===----------------------------------------------------------------------===//
// CompileOffloadOp
//===----------------------------------------------------------------------===//

static ParseResult parseCompileOffloadOp(OpAsmParser &p, TypedAttr &targetType,
                                         TypedAttr &emissionKind,
                                         TypedAttr &emissionOption,
                                         TypedAttr &func) {

  if (parseParamValue(p, targetType, KGEN::TargetType::get(p.getContext())) ||
      p.parseComma() || parseIndexParamValue(p, emissionKind) ||
      p.parseComma() || parseStringParam(p, emissionOption) || p.parseComma() ||
      parseTypeParamValue(p, func))
    return failure();
  return success();
}

static void printCompileOffloadOp(OpAsmPrinter &p, Operation *op,
                                  TypedAttr targetType, TypedAttr emissionKind,
                                  TypedAttr emissionOption, TypedAttr func) {

  printParamValue(p, targetType, KGEN::TargetType::get(op->getContext()));
  p << ", ";
  printIndexParamValue(p, emissionKind);
  p << ", ";
  printParamValue(p, emissionOption);
  p << ", ";
  printTypeParamValue(p, func);
}

//===----------------------------------------------------------------------===//
// ClosureInitOp
//===----------------------------------------------------------------------===//

static MemSymbolTripleAttr
assembleMemSymbolTripleAttr(MemSymbolTripleParts symbolTriple,
                            Type captureType) {
  MLIRContext *context = captureType.getContext();

  SymbolConstantAttr copy;
  if (symbolTriple.copy.callee)
    copy = makeSymbol(captureType, symbolTriple.copy.callee,
                      symbolTriple.copy.paramValues);
  SymbolConstantAttr move;
  if (symbolTriple.move.callee)
    move = makeSymbol(captureType, symbolTriple.move.callee,
                      symbolTriple.move.paramValues);
  SymbolConstantAttr del;
  if (symbolTriple.del.callee)
    del = makeSymbol(captureType, symbolTriple.del.callee,
                     symbolTriple.del.paramValues, /*isConstructor=*/false);
  return MemSymbolTripleAttr::get(context, copy, move, del);
}

static ParseResult parseClosureInitValue(
    OpAsmParser &p, TypeAttr &funcTypeGenerator, TypeAttr &functionType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &captures,
    ArrayAttr &moveOrCopyCaptureSymbols, ParamDeclArrayAttr &inputParams,
    InlineLevelAttr &inlineLevel, Region &body,
    SmallVectorImpl<Type> &capturesTypes) {
  if (p.parseLParen())
    return failure();

  // Collect the captures and symbols, if provided.
  SmallVector<MemSymbolTripleParts> memSymParts;
  LogicalResult result = success();
  if (p.parseOptionalRParen()) {
    do {
      if (p.parseOperand(captures.emplace_back()))
        return failure();

      // Build optional symbol.
      if (p.parseOptionalLSquare()) {
        memSymParts.push_back(MemSymbolTripleParts());
      } else {
        if (parseMemSymbolParts(p, memSymParts.emplace_back()))
          return failure();
        if (p.parseRSquare())
          return failure();
      }
      result = p.parseOptionalComma();
    } while (succeeded(result));
    if (p.parseRParen())
      return failure();
  }

  // Parse the function signature and the body.
  if (parseRegionOnly(p, inputParams, functionType, funcTypeGenerator,
                      inlineLevel, body))
    return failure();
  if (p.parseColon() || p.parseLParen())
    return failure();
  if (p.parseOptionalRParen()) {
    if (p.parseTypeList(capturesTypes) || p.parseRParen())
      return failure();
  }

  // Parse the capture types.
  if (capturesTypes.size() != memSymParts.size())
    return p.emitError(p.getCurrentLocation(),
                       "expected symbols to match number of capture types");

  // Now that we have the capture types and the symbols, we can infer the
  // signatures and build the symbol constant attributes.
  SmallVector<Attribute> copyOrMoveSymbols;
  for (auto [symbolTriple, captureType] :
       llvm::zip(memSymParts, capturesTypes)) {
    // If a value is captured by copy or move the del must also be provided
    if (symbolTriple.getIsTrivial()) {
      copyOrMoveSymbols.push_back(UnitAttr::get(p.getContext()));
      continue;
    }
    MemSymbolTripleAttr memSymbolTriple =
        assembleMemSymbolTripleAttr(symbolTriple, captureType);
    if (!memSymbolTriple)
      return p.emitError(p.getCurrentLocation(),
                         "expected move and del symbols");
    copyOrMoveSymbols.push_back(memSymbolTriple);
  }
  moveOrCopyCaptureSymbols = ArrayAttr::get(p.getContext(), copyOrMoveSymbols);
  return success();
}

static void printMemSymbolTriple(OpAsmPrinter &p, MemSymbolTripleAttr triple) {
  p << "[";
  printMemSymbolTripleAttrWithoutType(p, triple.getCopy(), triple.getMove(),
                                      triple.getDel());
  p << "]";
}

// Print function for ClosureInitValue custom format
static void printClosureInitValue(OpAsmPrinter &p, Operation *op,
                                  TypeAttr funcTypeGenerator,
                                  TypeAttr functionType, ValueRange captures,
                                  ArrayAttr moveOrCopyCaptureSymbols,
                                  ParamDeclArrayAttr inputParams,
                                  InlineLevelAttr inlineLevel, Region &body,
                                  TypeRange capturesTypes) {
  p << "(";
  int i = 0;
  int n = captures.size();
  for (auto [capture, symbol] : llvm::zip(captures, moveOrCopyCaptureSymbols)) {
    p << capture;
    if (!isa<UnitAttr>(symbol))
      printMemSymbolTriple(p, cast<MemSymbolTripleAttr>(symbol));
    if (++i < n)
      p << ", ";
  }
  p << ")";
  printRegionOnly(p, op, inputParams, functionType, funcTypeGenerator,
                  inlineLevel, body);
  p << " : ";
  p << '(';
  llvm::interleaveComma(capturesTypes, p,
                        [&](Type type) { p.printType(type); });
  p << ')';
}

LogicalResult ClosureInitOp::verify() {
  if (getCaptures().size() != getMoveOrCopyCaptureSymbols().size())
    return emitOpError("expected symbols to match number of capture types");
  for (Attribute symbol : getMoveOrCopyCaptureSymbols()) {
    if (!isa<MemSymbolTripleAttr, UnitAttr>(symbol))
      return emitOpError(
          "expected symbol constant attribute or unit attribute");
  }
  // If type is pointer it must be a pointer to a closure type and it cannot
  // be register passable.
  if (auto ptr = dyn_cast<PointerType>(getResult().getType())) {
    if (auto closureType = dyn_cast<ClosureType>((ptr.getElementType()))) {
      if (closureType.getClosureMemoryKind() ==
          ClosureMemoryKind::REGISTER_PASSABLE)
        return emitOpError("expected escaping/nonescaping closure type if "
                           "type is a pointer");
      else
        return success();
    }
    return emitOpError("expected closure type");
  }

  // if register passable, it must be a closure type.
  if (auto closureType = dyn_cast<ClosureType>(getResult().getType())) {
    if (closureType.getClosureMemoryKind() !=
        ClosureMemoryKind::REGISTER_PASSABLE)
      return emitOpError(
          "expected register passable closure type if type is not a pointer");
    else
      return success();
  }
  return emitOpError("expected closure type");
}

///===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/KGENDialect/KGEN.cpp.inc"
