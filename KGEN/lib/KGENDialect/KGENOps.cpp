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
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/STLExtras.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

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
  // Forbid the materialization of parameter capturing closures.
  if (auto sig = dyn_cast<SignatureType>(op.getType())) {
    if (sig.isCapturing())
      return op.emitOpError("cannot be used to materialize capturing closures; "
                            "use `kgen.create_closure` instead");
  }

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

OpFoldResult ParamConstantOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

bool ParamConstantOp::isImplicitlyParametric() {
  return containsSymbolConstants(getValue());
}

void ParamConstantOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {}

//===----------------------------------------------------------------------===//
// ParamMaterializeOp
//===----------------------------------------------------------------------===//

LogicalResult ParamMaterializeOp::verify() { return verifyParamValueOp(*this); }

LogicalResult ParamMaterializeOp::canonicalize(ParamMaterializeOp op,
                                               PatternRewriter &rewriter) {
  // Decay to a constant if the parameter value is a constant value with no
  // memory references.
  if (!ParameterAttr::isSimpleConstant(op.getValue()))
    return rewriter.notifyMatchFailure(op, "value is not a simple constant");

  mlir::AttrTypeWalker walker;
  walker.addWalk([&](MemRefAttr ref) {
    for (MemoryBlobAttr blob : ref.getMemory())
      if (blob.getKind() != MemoryKind::ConstGlobal)
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walker.walk(op.getValue()).wasInterrupted())
    return rewriter.notifyMatchFailure(op, "value has memory references");

  rewriter.replaceOpWithNewOp<ParamConstantOp>(op, op.getValue());
  return success();
}

ErrorTreeOrSuccess ParamMaterializeOp::interpret(ArrayRef<Attribute> operands,
                                                 InterpreterState &state) {
  Attribute value = getValue();
  if (ErrorOrSuccess err = state.internalizeMemory(value); err.isError())
    return ErrorTree(getLoc(), err.takeError());
  state.mapResults(value);
  return success();
}

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
  return parseParamDeclaration(p, paramDecl, value);
}

static void printParamDeclareOpValue(OpAsmPrinter &p, Operation *,
                                     ParamDeclAttr paramDecl, TypedAttr value) {
  return printParamDeclaration(p, paramDecl, value);
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

static ParseResult parseRegionDeclaration(
    OpAsmParser &p, ParamDeclAttr &paramDecl, ParamDeclArrayAttr &inputParams,
    ParamDeclArrayAttr &resultParams, TypeAttr &functionType,
    TypeAttr &signature, InlineLevelAttr &inlineLevel, Region &body) {
  StringAttr paramName;
  SmallVector<OpAsmParser::Argument> args;
  FunctionType functionTypeValue;
  SignatureType signatureType;
  llvm::SMLoc bodyLoc;
  if (parseParamName(p, paramName) || p.parseEqual() ||
      parseFunctionSignature(p, args, inputParams, resultParams,
                             functionTypeValue, signatureType) ||
      parseOptionalInline(p, inlineLevel) || p.getCurrentLocation(&bodyLoc) ||
      p.parseRegion(body, args))
    return failure();

  // Form the Signature.
  SmallVector<Type> argTypes;
  for (const OpAsmParser::Argument &arg : args)
    argTypes.push_back(arg.type);
  functionType = TypeAttr::get(functionTypeValue);
  signature = TypeAttr::get(signatureType);
  paramDecl = ParamDeclAttr::get(paramName, signatureType);
  return success();
}

static void printRegionDeclaration(OpAsmPrinter &p, Operation *op,
                                   ParamDeclAttr paramDecl,
                                   ParamDeclArrayAttr inputParams,
                                   ParamDeclArrayAttr resultParams,
                                   TypeAttr functionType, TypeAttr signature,
                                   InlineLevelAttr inlineLevel, Region &body) {
  printParamName(p, paramDecl.getName());
  p << " = ";
  printFunctionSignature(p, &body, inputParams, resultParams,
                         cast<FunctionType>(functionType.getValue()),
                         cast<SignatureType>(signature.getValue()));
  printOptionalInline(p, inlineLevel.getValue());
  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
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

//===----------------------------------------------------------------------===//
// ParamApplyOp
//===----------------------------------------------------------------------===//

static ParseResult parseParamApplyOp(AsmParser &p, ParamDeclAttr &paramDecl,
                                     TypedAttr &callee,
                                     ParameterExprArrayAttr &operands) {
  StringAttr paramName;
  SignatureType calleeType;
  SmallVector<TypedAttr> operandValues;
  llvm::SMLoc sigLoc;
  if (parseParamName(p, paramName) || p.parseEqual() || p.parseLSquare() ||
      parseKGENType(p, calleeType) || p.parseColon() ||
      p.getCurrentLocation(&sigLoc) || parseParamValue(p, callee, calleeType) ||
      p.parseRSquare() || p.parseLParen() ||
      failableInterleave(
          calleeType.getArguments(),
          [&](Type type) {
            return parseParamValue(p, operandValues.emplace_back(), type);
          },
          [&] { return p.parseComma(); }) ||
      p.parseRParen())
    return failure();
  if (calleeType.getNumResults() != 1)
    return p.emitError(sigLoc, "expected callee to have 1 result");
  paramDecl = ParamDeclAttr::get(paramName, calleeType.getResults().front());
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
  auto type = cast<SignatureType>(getCallee().getType());
  if (type.getInputParamTypes().empty() && type.getResultParamTypes().empty())
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
// CostOfOp
//===----------------------------------------------------------------------===//

/// Compute cost of the given function.
static ErrorTreeOrSuccess computeCost(SymbolConstantAttr func, Location loc,
                                      InterpreterState &state, int64_t &loads,
                                      int64_t &stores,
                                      MutableArrayRef<int64_t> compute,
                                      size_t depth) {
  ErrorOr<Region *> body = state.lookupFunctionBody(func.getSymbol());
  if (body.isError())
    return ErrorTree(loc, body.takeError());

  // Count the number of ops in the body, including parents of regions.
  ErrorTreeOrSuccess walkOutcome;

  body.get()->walk([&](Operation *op) -> WalkResult {
    // Don't count constants, terminators, and debug ops.
    if (op->hasTrait<OpTrait::ConstantLike>() ||
        op->hasTrait<OpTrait::IsTerminator>() ||
        llvm::isa_and_present<DebugInfo::DebugInfoDialect>(op->getDialect()))
      return WalkResult::advance();

    // Compute the cost of the function call descending into the function
    // upto 'maxDepth'. Currently, 'maxDepth' is set to 2, which is sufficient
    // to count pop-level operations for exponentiation.
    constexpr size_t maxDepth = 2;
    if (auto call = dyn_cast<CallOp>(op)) {
      if (depth < maxDepth) {
        auto result = computeCost(call.getCallee(), call.getLoc(), state, loads,
                                  stores, compute, depth + 1);
        if (result.isError()) {
          walkOutcome = result.takeError();
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
    }

    // Count memory operations.
    if (auto memOp = dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
      if (memOp.hasEffect<mlir::MemoryEffects::Read>()) {
        ++loads;
        return WalkResult::advance();
      }
      if (memOp.hasEffect<mlir::MemoryEffects::Write>()) {
        ++stores;
        return WalkResult::advance();
      }
    }

    // Count compute operations.
    ComputeKind kind = ComputeKind::Other;
    if (auto computeOp = dyn_cast<ComputeOpInterface>(op))
      kind = computeOp.getComputeKind();

    ++(compute[static_cast<int>(kind)]);

    return WalkResult::advance();
  });

  return walkOutcome;
}

ErrorTreeOrSuccess CostOfOp::interpret(ArrayRef<Attribute> operands,
                                       InterpreterState &state) {
  int64_t loads = 0, stores = 0;
  std::array<int64_t, getMaxEnumValForComputeKind() + 1> compute{};
  auto callee = dyn_cast<SymbolConstantAttr>(getCallee());
  if (!callee)
    return ErrorTree(getLoc(), "callee is not concrete");

  ErrorTreeOrSuccess result =
      computeCost(callee, getLoc(), state, loads, stores, compute, /*depth=*/0);
  if (result.isError())
    return result;

  Builder builder(getContext());
  auto getComputeOpsAttr = [&builder, &compute](ComputeKind kind) {
    return builder.getIndexAttr(compute[static_cast<int>(kind)]);
  };

  state.mapResults({builder.getIndexAttr(loads), builder.getIndexAttr(stores),
                    getComputeOpsAttr(ComputeKind::Addition),
                    getComputeOpsAttr(ComputeKind::Comparison),
                    getComputeOpsAttr(ComputeKind::Division),
                    getComputeOpsAttr(ComputeKind::Multiplication),
                    getComputeOpsAttr(ComputeKind::MultiplyAdd),
                    getComputeOpsAttr(ComputeKind::Other)});
  return success();
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

LogicalResult ParamAssertOp::canonicalize(ParamAssertOp op,
                                          PatternRewriter &rewriter) {
  // If the condition is statically true then we can just remove this op.
  auto cond = op.getCond();
  if (auto intCond = dyn_cast<IntegerAttr>(cond)) {
    // Leave failing conditions, they must be diagnosed at elaboration time.
    if (intCond.getValue().isZero())
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}

/// This operation defines no parameters.
void ParamAssertOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {}

/// This operation is implicitly parametric.
bool ParamAssertOp::isImplicitlyParametric() { return true; }

//===----------------------------------------------------------------------===//
// GeneratorOp
//===----------------------------------------------------------------------===//

/// Parses a KGEN Generator.
ParseResult GeneratorOp::parse(OpAsmParser &parser, OperationState &result) {
  ExportKindAttr exportKind;
  if (parseSymbolExport(parser, exportKind))
    return failure();
  result.addAttribute(getExportKindAttrName(result.name), exportKind);
  return parseGeneratorOrFunc(parser, result, GeneratorOrFuncKind::generator);
}

// Print the GeneratorOp using the shared printing logic.
void GeneratorOp::print(OpAsmPrinter &p) {
  printSymbolExport(p, *this, getExportKindAttr());
  printGeneratorOrFunc(p, *this);
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
  auto sig = SignatureType::getChecked([&] { return p.emitError(loc); },
                                       functionType, conventions, effects);
  if (!sig)
    return failure();
  signature = TypeAttr::get(sig);

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
  auto sig = cast<SignatureType>(signature.getValue());
  auto func = cast<FuncOp>(op);

  printSymbolExport(p, op, exportKind);
  p << ' ';
  p.printSymbolName(name);
  auto printArg = [&](unsigned i) {
    p.printRegionArgument(body.getArgument(i));
    printArgConvention(p, sig.getArgConvention(i));
  };
  printSignatureValues(p, printArg, sig.getValues(), sig,
                       /*optionalResultList=*/true);
  printOptionalInline(p, inlineLevel.getValue());
  printOptionalDecorators(p, op, decorators);

  SmallVector<StringRef, 6> elidedAttrs{
      func.getExportKindAttrName(), func.getSymNameAttrName(),
      func.getSignatureAttrName(), func.getInlineLevelAttrName(),
      func.getDecoratorsAttrName()};
  if (attrs.get(func.getLLVMMetadataAttrName()) ==
      DictionaryAttr::get(op->getContext()))
    elidedAttrs.push_back(func.getLLVMMetadataAttrName());
  p.printOptionalAttrDictWithKeyword(attrs.getValue(), elidedAttrs);

  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// ExternGeneratorOp
//===----------------------------------------------------------------------===//

static ParseResult parseExternGenerator(OpAsmParser &p, TypeAttr &signature,
                                        TypeAttr &functionType,
                                        ParamDeclArrayAttr &inputParams,
                                        ParamDeclArrayAttr &resultParams) {
  SmallVector<OpAsmParser::Argument> args;
  FunctionType funcType;
  SignatureType sigType;
  if (parseFunctionSignature(p, args, inputParams, resultParams, funcType,
                             sigType))
    return failure();
  functionType = TypeAttr::get(funcType);
  signature = TypeAttr::get(sigType);
  return success();
}

static void printExternGenerator(OpAsmPrinter &p, Operation *op,
                                 TypeAttr signature, TypeAttr functionType,
                                 ParamDeclArrayAttr inputParams,
                                 ParamDeclArrayAttr resultParams) {
  printFunctionSignature(p, /*region=*/nullptr, inputParams, resultParams,
                         cast<FunctionType>(functionType.getValue()),
                         cast<SignatureType>(signature.getValue()));
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

static ParseResult
parseCallOp(OpAsmParser &p, SymbolConstantAttr &calleeCst,
            ParamDeclArrayAttr &paramDecls,
            SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
            SmallVectorImpl<Type> &operandTypes,
            SmallVectorImpl<Type> &resultTypes) {
  SymbolRefAttr callee;
  ParameterExprArrayAttr paramValues;
  if (p.parseAttribute(callee) ||
      parseCallOpParams(p, paramValues, paramDecls) ||
      p.parseOperandList(operands, AsmParser::Delimiter::Paren) ||
      p.parseColon())
    return failure();

  SignatureType signature;
  FunctionType functionType;
  if (parseKGENSignature(p, paramDecls, functionType, signature))
    return failure();
  calleeCst = SymbolConstantAttr::get(callee, paramValues, signature);
  llvm::append_range(operandTypes, functionType.getInputs());
  llvm::append_range(resultTypes, functionType.getResults());
  return success();
}

static void printCallOp(OpAsmPrinter &p, Operation *op,
                        SymbolConstantAttr calleeCst,
                        ParamDeclArrayAttr paramDecls, ValueRange operands,
                        TypeRange operandTypes, TypeRange resultTypes) {
  p << calleeCst.getSymbol();
  printCallOpParams(p, op, calleeCst.getParamValues(), paramDecls,
                    calleeCst.getType().getResultParamTypes());
  p << '(';
  p.printOperands(operands);
  p << ") : ";
  printSignatureValues(
      p, FunctionType::get(op->getContext(), operandTypes, resultTypes),
      calleeCst.getType());
}

OperandRange CallOp::getArgOperands() { return getOperands(); }

MutableOperandRange CallOp::getArgOperandsMutable() {
  return getOperandsMutable();
}

mlir::CallInterfaceCallable CallOp::getCallableForCallee() {
  return getCalleeSymbol();
}

void CallOp::concretizeCallee(mlir::IRRewriter &b, SymbolConstantAttr callee) {
  setCalleeAttr(callee);
  setParamDecls({});
}

void CallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  auto symbol = callee.get<SymbolRefAttr>();
  setCalleeAttr(SymbolConstantAttr::get(symbol, getCallee().getType()));
}

void CallOp::setCalleeAttr(TypedAttr callee) {
  setCalleeAttr(cast<SymbolConstantAttr>(callee));
}

//===----------------------------------------------------------------------===//
// CallParamOp
//===----------------------------------------------------------------------===//

LogicalResult CallParamOp::canonicalize(CallParamOp op,
                                        PatternRewriter &rewriter) {
  // If the condition is a known symbol, then replace this with a kgen.call.
  auto callee = dyn_cast<SymbolConstantAttr>(op.getCallee());
  if (!callee)
    return failure();

  rewriter.replaceOpWithNewOp<CallOp>(op, op.getResultTypes(), callee,
                                      op.getParamDecls(), op.getOperands());
  return success();
}

void CallParamOp::concretizeCallee(mlir::IRRewriter &b,
                                   SymbolConstantAttr callee) {
  b.replaceOpWithNewOp<CallOp>(*this, getResultTypes(), callee, getOperands());
}

//===----------------------------------------------------------------------===//
// ParamIfOp
//===----------------------------------------------------------------------===//

static ParseResult parseOptionalParamDecls(AsmParser &p,
                                           ParamDeclArrayAttr &paramDecls) {
  if (p.parseOptionalArrow()) {
    paramDecls = ParamDeclArrayAttr::get(p.getContext(), {});
    return success();
  }
  return parseParamDecls(p, paramDecls);
}

static void printOptionalParamDecls(AsmPrinter &p, Operation *op,
                                    ParamDeclArrayAttr paramDecls) {
  if (paramDecls.empty())
    return;
  p << " -> ";
  printParamDecls(p, paramDecls);
}

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

void ParamIfOp::walkDeclarations(function_ref<void(ParamDeclAttr)> walkDecl) {
  llvm::for_each(getResultParams(), walkDecl);
}

void ParamIfOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  ParamDefValue value(getCond(), {&getThenRegion(), &getElseRegion()});
  for (ParamDeclAttr decl : getResultParams())
    walkDef(decl, value);
}

void ParamIfOp::renameDeclarations(ArrayRef<ParamDeclAttr> decls) {
  setResultParams(decls);
}

bool ParamIfOp::isImplicitlyParametric() { return true; }

/// This operation has no uses to collect in the scopes it defines.
void ParamIfOp::collectParameterUsesBelow(
    function_ref<void(Attribute)> scanAttr, function_ref<void(Type)> scanType) {
}

//===----------------------------------------------------------------------===//
// ParamYieldOp
//===----------------------------------------------------------------------===//

LogicalResult ParamYieldOp::verify() {
  return checkOperandTypes(
      *this, cast<ParamIfOp>((*this)->getParentOp()).getResultTypes());
}

bool ParamYieldOp::isParentNode(Operation *op) { return isa<ParamIfOp>(op); }

void ParamYieldOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  // Branch to after the if operation.
  targets.emplace_back(std::nullopt, getOperands());
}

//===----------------------------------------------------------------------===//
// RebindOp
//===----------------------------------------------------------------------===//

/// Fold away the rebind if the input and output types are the same.
OpFoldResult RebindOp::fold(FoldAdaptor adaptor) {
  if (getInput().getType() == getType())
    return getInput();
  if (auto ptr = dyn_cast_or_null<SymbolicPointerAttr>(adaptor.getInput()))
    return SymbolicPointerAttr::get(ptr.getSlot(), getType());
  return {};
}

/// If the operand to a rebind is defined by a rebind, use the second rebind's
/// operand.
LogicalResult RebindOp::canonicalize(RebindOp op, PatternRewriter &rewriter) {
  RebindOp cur = op, parent;
  // Climb all the way to the top to avoid recursively invoking this pattern.
  while ((parent = cur.getOperand().getDefiningOp<RebindOp>()))
    cur = parent;

  if (cur == op)
    return failure();
  rewriter.modifyOpInPlace(op, [&] { op.setOperand(cur.getOperand()); });
  return success();
}

//===----------------------------------------------------------------------===//
// UndefOp
//===----------------------------------------------------------------------===//

OpFoldResult UndefOp::fold(FoldAdaptor adaptor) {
  return UnknownAttr::get(getType());
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

LogicalResult CallIndirectOp::canonicalize(CallIndirectOp op,
                                           PatternRewriter &b) {
  auto create = op.getCallee().getDefiningOp<CreateClosureOp>();
  if (!create)
    return b.notifyMatchFailure(op.getLoc(), "callee op is not create closure");
  // Replace this with a direct call.
  SmallVector<Value> args = llvm::to_vector(create.getCaptures());
  llvm::append_range(args, op.getArguments());
  b.replaceOpWithNewOp<CallParamOp>(op, op.getResultTypes(), create.getCallee(),
                                    create.getParamDeclsAttr(), args);
  return success();
}

/// CallIndirectOp cannot conform to CallOpInterface, but is very similar since
/// we know the callee at elaboration time.
ErrorTreeOrSuccess CallIndirectOp::interpret(ArrayRef<Attribute> operands,
                                             InterpreterState &state) {
  auto callee = dyn_cast<SymbolConstantAttr>(operands[0]);
  if (!callee)
    return ErrorTree(getLoc(), "couldn't resolve kgen.call_indirect callee");

  auto bodyOr = state.lookupFunctionBody(callee.getSymbol());
  if (bodyOr.isError())
    return ErrorTree(getLoc(), bodyOr.takeError());

  // Function regions are isolated from above, so push a new stack frame. Then,
  // transfer control flow to the beginning of the function body.
  Region &body = **bodyOr;
  state.pushFrame(*this, body.getParentOp());
  state.transferControlFlowTo(&body.front(), operands.drop_front());
  return success();
}

//===----------------------------------------------------------------------===//
// StageClosureOp
//===----------------------------------------------------------------------===//

static ParseResult parseStageClosureOp(OpAsmParser &p, Type &resultType,
                                       Region &body) {
  // we expect the following syntax:
  // kgen.stage_closure = () capturing -> index {
  // } { name = foo }
  SignatureType signatureType;
  ParamDeclArrayAttr inputParams;
  ParamDeclArrayAttr resultParams;
  FunctionType functionTypeValue;
  SmallVector<OpAsmParser::Argument> args;
  llvm::SMLoc bodyLoc;
  if (p.parseEqual() ||
      parseFunctionSignature(p, args, inputParams, resultParams,
                             functionTypeValue, signatureType) ||
      p.getCurrentLocation(&bodyLoc) || p.parseRegion(body, args))
    return failure();
  if (!inputParams.empty() || !resultParams.empty())
    return p.emitError(bodyLoc, "staged closures cannot have parameters");
  resultType = signatureType;
  return success();
}

static void printStageClosureOp(OpAsmPrinter &p, Operation *op,
                                SignatureType resultType, Region &body) {
  p << "= ";
  printFunctionSignature(p, &body, {}, {}, resultType.getValues(), resultType);
  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// CreateClosureOp
//===----------------------------------------------------------------------===//

void CreateClosureOp::build(OpBuilder &b, OperationState &state,
                            TypedAttr callee, ValueRange captures) {
  build(b, state, callee, captures,
        ParamDeclArrayAttr::get(b.getContext(), {}));
}

void CreateClosureOp::build(OpBuilder &b, OperationState &state, Type type,
                            TypedAttr callee, ValueRange captures) {
  build(b, state, type, callee, captures,
        ParamDeclArrayAttr::get(b.getContext(), {}));
}

LogicalResult CreateClosureOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange captures,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &results) {
  auto callee = dyn_cast_or_null<TypedAttr>(attributes.get("callee"));
  if (!callee) {
    return mlir::emitOptionalError(
        loc, "'create_closure' expected TypedAttr 'callee'");
  }
  auto sig = dyn_cast<SignatureType>(callee.getType());
  if (!sig) {
    return mlir::emitOptionalError(
        loc, "'create_closure' attribute 'callee' must have SignatureType");
  }

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
  results.push_back(SignatureType::get(
      OpBuilder(ctx).getFunctionType(newArgTypes, sig.getResults()),
      sig.getInputParamTypes(), sig.getResultParamTypes(), newArgConvs, effects,
      sig.getMetadata() ? sig.getMetadata().getWithBoundPosArgs(numCaptures)
                        : nullptr));
  return mlir::success();
}

static ParseResult
parseClosureCaptureTypes(AsmParser &p, TypedAttr callee,
                         ArrayRef<OpAsmParser::UnresolvedOperand> captures,
                         SmallVectorImpl<Type> &captureTypes) {
  auto sig = dyn_cast<SignatureType>(callee.getType());
  if (!sig) {
    return p.emitError(p.getCurrentLocation(),
                       "expected type of callee to be SignatureType");
  }

  unsigned numCaptures = captures.size();
  if (numCaptures > sig.getNumArguments()) {
    return p.emitError(p.getCurrentLocation(), "provided ")
           << numCaptures << " operands but callee only has "
           << sig.getNumArguments() << " to bind";
  }

  ArrayRef<mlir::Type> inputs = sig.getArguments().take_front(numCaptures);
  captureTypes.append(inputs.begin(), inputs.end());
  return success();
}

static void printClosureCaptureTypes(AsmPrinter &p, Operation *,
                                     TypedAttr callee, ValueRange captures,
                                     TypeRange captureTypes) {}

LogicalResult CreateClosureOp::verify() {
  SignatureType sig = getCalleeType();
  if (getNumOperands() > sig.getNumArguments()) {
    return emitOpError("provided ")
           << getNumOperands() << " operands but callee only has "
           << sig.getNumArguments() << " to bind";
  }
  unsigned expectedArgs = sig.getNumArguments() - getNumOperands();
  if (getType().getNumArguments() != expectedArgs) {
    return emitOpError("result signature has ")
           << getType().getNumArguments() << " arguments but expected "
           << expectedArgs;
  }

  for (auto [i, type, argType] :
       llvm::enumerate(getOperandTypes(),
                       sig.getArguments().take_front(getNumOperands()))) {
    if (type != argType) {
      return emitOpError("operand #")
             << i << " has type " << type
             << " but callee argument type expected " << argType;
    }
  }
  for (auto [i, type, argType] :
       llvm::enumerate(getType().getArguments(),
                       sig.getArguments().drop_front(getNumOperands()))) {
    if (type != argType) {
      return emitOpError("result signature argument #")
             << i << " type is " << argType << " but expected to be " << type;
    }
  }

  if (!getCaptures().empty() && !getType().isCapturing())
    return emitOpError("has captures, so result signature must be 'capturing'");
  return success();
}

ErrorTreeOrSuccess CreateClosureOp::interpret(ArrayRef<Attribute> operands,
                                              InterpreterState &state) {
  // We have no representation for closing over runtime values.
  if (!operands.empty())
    return ErrorTree(getLoc(), "TODO: cannot form a closure at compile time");

  state.mapResults(getCallee());
  return success();
}

//===----------------------------------------------------------------------===//
// SourceLocOp
//===----------------------------------------------------------------------===//

static ParseResult parseIntProperty(OpAsmParser &parser, int64_t &value) {
  return parser.parseInteger(value);
}

static void printIntProperty(OpAsmPrinter &printer, Operation *op,
                             int64_t value) {
  printer << value;
}

/// Core implementation for interpreting kgen.source_loc.
static SmallVector<Attribute> sourceLocInterpretImpl(Operation *callOp,
                                                     MLIRContext *ctx) {
  OpBuilder b(ctx);
  auto strType = b.getType<StringType>();

  if (callOp) {
    FileLineColLoc fileLoc = DebugInfo::extractSourceLoc(callOp->getLoc());
    return {b.getIndexAttr(fileLoc.getLine()),
            b.getIndexAttr(fileLoc.getColumn()),
            StringAttr::get(fileLoc.getFilename().getValue(), strType)};
  }

  auto zero = b.getIndexAttr(0);
  return {zero, zero,
          StringAttr::get("<unknown location in parameter context>", strType)};
}

ErrorTreeOrSuccess SourceLocOp::interpret(ArrayRef<Attribute> operands,
                                          InterpreterState &state) {
  state.mapResults(sourceLocInterpretImpl(
      state.getOrigin(getProperties().getInlineCount()), getContext()));
  return success();
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
// IntLiteralCmp
//===----------------------------------------------------------------------===//

OpFoldResult IntLiteralCmp::fold(FoldAdaptor adaptor) {
  IntLiteralAttr lAttr = dyn_cast_or_null<IntLiteralAttr>(adaptor.getLhs());
  IntLiteralAttr rAttr = dyn_cast_or_null<IntLiteralAttr>(adaptor.getRhs());
  IntLiteralCmpPred pred = adaptor.getPred();
  if (!lAttr || !rAttr)
    return {};
  IPInt l = lAttr.getValue();
  IPInt r = rAttr.getValue();

  switch (pred) {
  case IntLiteralCmpPred::Eq:
    return BoolAttr::get(lAttr.getContext(), l == r);
  case IntLiteralCmpPred::Ne:
    return BoolAttr::get(lAttr.getContext(), l != r);
  case IntLiteralCmpPred::Lt:
    return BoolAttr::get(lAttr.getContext(), l < r);
  case IntLiteralCmpPred::Le:
    return BoolAttr::get(lAttr.getContext(), l <= r);
  case IntLiteralCmpPred::Gt:
    return BoolAttr::get(lAttr.getContext(), l > r);
  case IntLiteralCmpPred::Ge:
    return BoolAttr::get(lAttr.getContext(), l >= r);
  }
  llvm_unreachable("invalid cmp predicate");
}

//===----------------------------------------------------------------------===//
// IntLiteralBinop
//===----------------------------------------------------------------------===//

OpFoldResult IntLiteralBinop::fold(FoldAdaptor adaptor) {
  IntLiteralAttr lAttr = dyn_cast_or_null<IntLiteralAttr>(adaptor.getLhs());
  IntLiteralAttr rAttr = dyn_cast_or_null<IntLiteralAttr>(adaptor.getRhs());
  IntLiteralBinopKind o = adaptor.getOper();
  if (!lAttr || !rAttr)
    return {};
  IPInt l = lAttr.getValue();
  IPInt r = rAttr.getValue();
  IPInt zero(0);
  IPInt one(1);

  IPInt result;
  switch (o) {
  case IntLiteralBinopKind::Add:
    result = l + r;
    break;
  case IntLiteralBinopKind::Sub:
    result = l - r;
    break;
  case IntLiteralBinopKind::Mul:
    result = l * r;
    break;
  case IntLiteralBinopKind::FloorDiv:
    if ((l >= zero) == (r >= zero) || l % r == zero)
      result = l / r;
    else
      result = (l / r) - one;
    break;
  case IntLiteralBinopKind::Mod:
    // Python's mod:
    // The result sign matches the RHS sign.
    // If the signs match, the value is the same as: sign(abs(l) % abs(r)),
    // where sign is determined by the RHS sign. If the signs don't match, the
    // value is the same as: sign((abs(r) - (abs(l) % abs(r))) % abs(r)).
    {
      bool signMatch = (l >= zero) == (r >= zero);
      IPInt L = l.abs();
      IPInt R = r.abs();
      result = (L % R).abs();
      if (!signMatch && result != zero)
        result = R - result;
      if (r < zero)
        result = zero - result;
    }
    break;
  case IntLiteralBinopKind::Lshift:
    result = l << r;
    break;
  case IntLiteralBinopKind::Rshift:
    result = l >> r;
    break;
  case IntLiteralBinopKind::And:
    result = l & r;
    break;
  case IntLiteralBinopKind::Or:
    result = l | r;
    break;
  case IntLiteralBinopKind::Xor:
    result = l ^ r;
    break;
  }

  return IntLiteralAttr::get(lAttr.getContext(), IPInt(result));
}

//===----------------------------------------------------------------------===//
// IntLiteralConvertOp
//===----------------------------------------------------------------------===//

static ErrorTreeOrSuccess intLiteralConvertOpHelper(IPInt invalIP,
                                                    mlir::Type outType,
                                                    IntegerAttr &attrResult,
                                                    Location loc) {
  APInt invalAP = invalIP.getAPInt();
  unsigned outWidth = 64;
  bool isUnsigned = false;
  APInt result;
  if (!outType.isIndex()) {
    outWidth = outType.getIntOrFloatBitWidth();
    if (outType.isUnsignedInteger())
      isUnsigned = true;
  }
  if (invalAP.getBitWidth() > outWidth) {
    std::string msg;
    llvm::raw_string_ostream msgStream(msg);
    msgStream << "integer value " << invalIP << " requires "
              << invalAP.getBitWidth()
              << " bits to store, but the destination bit width is only "
              << outWidth << " bits wide";
    return ErrorTree(loc, Error(msgStream.str()));
  }
  if (isUnsigned)
    result = invalAP.zextOrTrunc(outWidth);
  else
    result = invalAP.sextOrTrunc(outWidth);
  attrResult = IntegerAttr::get(outType, result);
  return success();
}

ErrorTreeOrSuccess IntLiteralConvertOp::interpret(ArrayRef<Attribute> operands,
                                                  InterpreterState &state) {
  assert(!operands.empty() && "IntLiteralConvertOp must have an operand");
  auto inval = ::dyn_cast<IntLiteralAttr>(operands[0]);
  IntegerAttr attrResult;
  ErrorTreeOrSuccess errOrSuccess = intLiteralConvertOpHelper(
      inval.getValue(), getType(), attrResult, getLoc());
  if (errOrSuccess.isError())
    return errOrSuccess;
  state.mapResults(attrResult);
  return success();
}

OpFoldResult IntLiteralConvertOp::fold(FoldAdaptor adaptor) {
  auto in = dyn_cast_if_present<IntLiteralAttr>(adaptor.getInput());
  if (!in)
    return {};
  IntegerAttr attrResult;
  ErrorTreeOrSuccess errOrSuccess =
      intLiteralConvertOpHelper(in.getValue(), getType(), attrResult, getLoc());
  if (errOrSuccess.isError())
    return {};
  return attrResult;
}

//===----------------------------------------------------------------------===//
// IntLiteralBitWidth
//===----------------------------------------------------------------------===//

OpFoldResult IntLiteralBitWidthOp::fold(FoldAdaptor adaptor) {
  auto in = dyn_cast_if_present<IntLiteralAttr>(adaptor.getInput());
  if (!in)
    return {};
  unsigned bits = in.getValue().getAPInt().getBitWidth();
  return IntLiteralAttr::get(in.getContext(), IPInt(bits));
}

//===----------------------------------------------------------------------===//
// IntLiteralToFloatLiteral
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess
IntLiteralToFloatLiteralOp::interpret(ArrayRef<Attribute> operands,
                                      InterpreterState &state) {
  assert(!operands.empty() &&
         "IntLiteralToFloatLiteralOp must have an operand");
  auto inval = ::cast<IntLiteralAttr>(operands[0]);
  FloatLiteralAttr attrResult = FloatLiteralAttr::get(
      inval.getContext(),
      FloatLiteralSpecialValuesAttr::get(inval.getContext(),
                                         FloatLiteralSpecialValues::Normal),
      IPRational(inval.getValue(), IPInt(1)));
  state.mapResults(attrResult);
  return success();
}

OpFoldResult IntLiteralToFloatLiteralOp::fold(FoldAdaptor adaptor) {
  auto in = dyn_cast_if_present<IntLiteralAttr>(adaptor.getInput());
  if (!in)
    return {};
  return FloatLiteralAttr::get(
      in.getContext(),
      FloatLiteralSpecialValuesAttr::get(in.getContext(),
                                         FloatLiteralSpecialValues::Normal),
      IPRational(in.getValue(), IPInt(1)));
}

//===----------------------------------------------------------------------===//
// FloatLiteralIsa
//===----------------------------------------------------------------------===//

OpFoldResult FloatLiteralIsa::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_or_null<FloatLiteralAttr>(adaptor.getInput())) {
    return BoolAttr::get(input.getContext(),
                         input.getSpecial().getValue() == adaptor.getSpecial());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// FloatLiteralCmp
//===----------------------------------------------------------------------===//

static bool isNan(FloatLiteralSpecialValues v) {
  return v == FloatLiteralSpecialValues::Nan;
}
static bool isNegZero(FloatLiteralSpecialValues v) {
  return v == FloatLiteralSpecialValues::NegZero;
}
static bool isInf(FloatLiteralSpecialValues v) {
  return v == FloatLiteralSpecialValues::Inf;
}
static bool isNegInf(FloatLiteralSpecialValues v) {
  return v == FloatLiteralSpecialValues::NegInf;
}
static bool isNormal(FloatLiteralSpecialValues v) {
  return v == FloatLiteralSpecialValues::Normal;
}

/// Helper for float literal comparison.  The lhs/rhs values are only meaningful
/// when lSpecial/rSpecial are normal.
static bool floatLiteralCmpHelper(const FloatLiteralCmpPred &pred,
                                  const FloatLiteralSpecialValues &lSpecial,
                                  const FloatLiteralSpecialValues &rSpecial,
                                  const IPRational &lhs,
                                  const IPRational &rhs) {
  switch (pred) {
  case FloatLiteralCmpPred::Eq:
    if (lSpecial == rSpecial) {
      if (isNormal(lSpecial))
        return lhs == rhs;
      return !isNan(lSpecial);
    }
    // Python treats -0 and 0 as equal.
    if (isNegZero(lSpecial) && isNormal(rSpecial) && rhs == 0)
      return true;
    if (isNegZero(rSpecial) && isNormal(lSpecial) && lhs == 0)
      return true;
    return false;
  case FloatLiteralCmpPred::Ne:
    return !floatLiteralCmpHelper(FloatLiteralCmpPred::Eq, lSpecial, rSpecial,
                                  lhs, rhs);
  case FloatLiteralCmpPred::Lt:
    switch (lSpecial) {
    case FloatLiteralSpecialValues::Normal:
      switch (rSpecial) {
      case FloatLiteralSpecialValues::Normal:
        return lhs < rhs;
      case FloatLiteralSpecialValues::Inf:
        return true;
      case FloatLiteralSpecialValues::NegZero:
        return lhs < 0;
      default:
        return false;
      }
    case FloatLiteralSpecialValues::NegZero:
      switch (rSpecial) {
      case FloatLiteralSpecialValues::Normal:
        // This would be <=, but Python treats -0 as equal to 0, so the RHS
        // needs to be strictly greater than positive zero.
        return IPRational(0) < rhs;
      case FloatLiteralSpecialValues::Inf:
        return true;
      default:
        return false;
      }
    case FloatLiteralSpecialValues::Inf:
    case FloatLiteralSpecialValues::Nan:
      return false;
    case FloatLiteralSpecialValues::NegInf:
      return !isNan(rSpecial) && !isNegInf(rSpecial);
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralCmpPred::Le:
    return floatLiteralCmpHelper(FloatLiteralCmpPred::Lt, lSpecial, rSpecial,
                                 lhs, rhs) ||
           floatLiteralCmpHelper(FloatLiteralCmpPred::Eq, lSpecial, rSpecial,
                                 lhs, rhs);
  case FloatLiteralCmpPred::Gt:
    if (isNan(lSpecial) || isNan(rSpecial))
      return false;
    return !floatLiteralCmpHelper(FloatLiteralCmpPred::Le, lSpecial, rSpecial,
                                  lhs, rhs);
  case FloatLiteralCmpPred::Ge:
    return floatLiteralCmpHelper(FloatLiteralCmpPred::Gt, lSpecial, rSpecial,
                                 lhs, rhs) ||
           floatLiteralCmpHelper(FloatLiteralCmpPred::Eq, lSpecial, rSpecial,
                                 lhs, rhs);
  }
  llvm_unreachable("invalid cmp predicate");
}

OpFoldResult FloatLiteralCmp::fold(FoldAdaptor adaptor) {
  auto lAttr = dyn_cast_or_null<FloatLiteralAttr>(adaptor.getLhs());
  auto rAttr = dyn_cast_or_null<FloatLiteralAttr>(adaptor.getRhs());
  if (!lAttr || !rAttr)
    return {};
  FloatLiteralSpecialValues lSpecial = lAttr.getSpecial().getValue();
  FloatLiteralSpecialValues rSpecial = rAttr.getSpecial().getValue();
  IPRational lhs;
  IPRational rhs;
  if (isNormal(lSpecial)) {
    assert(lAttr.getRational().has_value() &&
           "rational does not have a value when special value is normal");
    lhs = lAttr.getRational().value();
  }
  if (isNormal(rSpecial)) {
    assert(rAttr.getRational().has_value() &&
           "rational does not have a value when special value is normal");
    rhs = rAttr.getRational().value();
  }
  return BoolAttr::get(
      lAttr.getContext(),
      floatLiteralCmpHelper(adaptor.getPred(), lSpecial, rSpecial, lhs, rhs));
}

//===----------------------------------------------------------------------===//
// FloatLiteralBinop
//===----------------------------------------------------------------------===//

static std::tuple<FloatLiteralSpecialValues, IPRational>
floatLiteralAdd(FloatLiteralSpecialValues lSpecial,
                FloatLiteralSpecialValues rSpecial, IPRational lhs,
                IPRational rhs) {
  switch (lSpecial) {
  case FloatLiteralSpecialValues::NegZero:
    if (isNegZero(rSpecial))
      return {FloatLiteralSpecialValues::Normal, 0};
    return {rSpecial, rhs};
  case FloatLiteralSpecialValues::Inf:
    if (isNegInf(rSpecial) || isNan(rSpecial))
      return {FloatLiteralSpecialValues::Nan, 0};
    return {FloatLiteralSpecialValues::Inf, 0};
  case FloatLiteralSpecialValues::NegInf:
    if (isInf(rSpecial) || isNan(rSpecial))
      return {FloatLiteralSpecialValues::Nan, 0};
    return {FloatLiteralSpecialValues::NegInf, 0};
  case FloatLiteralSpecialValues::Nan:
    return {FloatLiteralSpecialValues::Nan, 0};
  case FloatLiteralSpecialValues::Normal:
    if (isNormal(rSpecial))
      return {FloatLiteralSpecialValues::Normal, lhs + rhs};
    return floatLiteralAdd(rSpecial, lSpecial, rhs, lhs);
  }
  llvm_unreachable("unknown FloatLiteral special type");
}

static std::tuple<FloatLiteralSpecialValues, IPRational>
floatLiteralSub(FloatLiteralSpecialValues lSpecial,
                FloatLiteralSpecialValues rSpecial, const IPRational &lhs,
                const IPRational &rhs) {
  switch (lSpecial) {
  case FloatLiteralSpecialValues::NegZero:
    // When adding zeroes, the signs are basically XORed, like with
    // multiplication.
    if (isNegZero(rSpecial))
      return {FloatLiteralSpecialValues::Normal, 0};
    if (isNormal(rSpecial) && rhs == 0)
      return {FloatLiteralSpecialValues::NegZero, 0};
    return floatLiteralSub(FloatLiteralSpecialValues::Normal, rSpecial, 0, rhs);
  case FloatLiteralSpecialValues::Inf:
    if (isInf(rSpecial) || isNan(rSpecial))
      return {FloatLiteralSpecialValues::Nan, 0};
    return {FloatLiteralSpecialValues::Inf, 0};
  case FloatLiteralSpecialValues::NegInf:
    if (isNegInf(rSpecial) || isNan(rSpecial))
      return {FloatLiteralSpecialValues::Nan, 0};
    return {FloatLiteralSpecialValues::NegInf, 0};
  case FloatLiteralSpecialValues::Nan:
    return {FloatLiteralSpecialValues::Nan, 0};
  case FloatLiteralSpecialValues::Normal:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::NegZero:
      return {lSpecial, lhs};
    case FloatLiteralSpecialValues::Inf:
      return {FloatLiteralSpecialValues::NegInf, 0};
    case FloatLiteralSpecialValues::NegInf:
      return {FloatLiteralSpecialValues::Inf, 0};
    case FloatLiteralSpecialValues::Nan:
      return {FloatLiteralSpecialValues::Nan, 0};
    case FloatLiteralSpecialValues::Normal:
      return {FloatLiteralSpecialValues::Normal, lhs - rhs};
    }
  }
  llvm_unreachable("unknown FloatLiteral special type");
}

/// Helper for multiplication, to keep the special case matching table separate.
/// Assumes that at least one of lSpecial and rSpecial is non-normal.
static FloatLiteralSpecialValues
floatLiteralMulSpecialCases(const FloatLiteralSpecialValues &lSpecial,
                            const FloatLiteralSpecialValues &rSpecial,
                            const IPRational &lhs, const IPRational &rhs) {
  switch (lSpecial) {
  case FloatLiteralSpecialValues::NegZero:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
    case FloatLiteralSpecialValues::Inf:
    case FloatLiteralSpecialValues::NegInf:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::NegZero:
      return FloatLiteralSpecialValues::Normal;
    case FloatLiteralSpecialValues::Normal:
      if (rhs < 0)
        return FloatLiteralSpecialValues::Normal;
      return FloatLiteralSpecialValues::NegZero;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::Inf:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
    case FloatLiteralSpecialValues::NegZero:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::NegInf:
      return FloatLiteralSpecialValues::NegInf;
    case FloatLiteralSpecialValues::Inf:
      return FloatLiteralSpecialValues::Inf;
    case FloatLiteralSpecialValues::Normal:
      if (rhs == 0)
        return FloatLiteralSpecialValues::Nan;
      if (rhs < 0)
        return FloatLiteralSpecialValues::NegInf;
      return FloatLiteralSpecialValues::Inf;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::NegInf:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
    case FloatLiteralSpecialValues::NegZero:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::NegInf:
      return FloatLiteralSpecialValues::Inf;
    case FloatLiteralSpecialValues::Inf:
      return FloatLiteralSpecialValues::NegInf;
    case FloatLiteralSpecialValues::Normal:
      if (rhs == 0)
        return FloatLiteralSpecialValues::Nan;
      if (rhs < 0)
        return FloatLiteralSpecialValues::Inf;
      return FloatLiteralSpecialValues::NegInf;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::Nan:
    return FloatLiteralSpecialValues::Nan;
  case FloatLiteralSpecialValues::Normal:
    // The case of both being normal is handled up front, so we don't worry
    // about it here.  Instead just recur with flipped operand order to handle
    // the case that LHS is normal.
    return floatLiteralMulSpecialCases(rSpecial, lSpecial, rhs, lhs);
  }
  llvm_unreachable("unknown FloatLiteral special type");
}

static std::tuple<FloatLiteralSpecialValues, IPRational>
floatLiteralMul(FloatLiteralSpecialValues lSpecial,
                FloatLiteralSpecialValues rSpecial, IPRational lhs,
                IPRational rhs) {
  if (isNormal(lSpecial) && isNormal(rSpecial)) {
    IPRational ratResult = lhs * rhs;
    if (ratResult == 0 && ((lhs < 0) || (rhs < 0)))
      return {FloatLiteralSpecialValues::NegZero, 0};
    return {FloatLiteralSpecialValues::Normal, ratResult};
  }
  return {floatLiteralMulSpecialCases(lSpecial, rSpecial, lhs, rhs), 0};
}

/// Helper to separate the special case logic for division.  Assumes that at
/// least one of lSpecial and rSpecial is non-normal.
static FloatLiteralSpecialValues
floatLiteralDivSpecialCases(const FloatLiteralSpecialValues &lSpecial,
                            const FloatLiteralSpecialValues &rSpecial,
                            const IPRational &lhs, const IPRational &rhs) {
  switch (lSpecial) {
  case FloatLiteralSpecialValues::NegZero:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Inf:
      return FloatLiteralSpecialValues::NegZero;
    case FloatLiteralSpecialValues::NegInf:
      return FloatLiteralSpecialValues::Normal;
    case FloatLiteralSpecialValues::NegZero:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Normal:
      if (rhs == 0)
        return FloatLiteralSpecialValues::Nan;
      if (rhs < 0)
        return FloatLiteralSpecialValues::Normal;
      return FloatLiteralSpecialValues::NegZero;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::Inf:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
    case FloatLiteralSpecialValues::NegZero:
    case FloatLiteralSpecialValues::NegInf:
    case FloatLiteralSpecialValues::Inf:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Normal:
      if (rhs == 0)
        return FloatLiteralSpecialValues::Nan;
      if (rhs < 0)
        return FloatLiteralSpecialValues::NegInf;
      return FloatLiteralSpecialValues::Inf;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::NegInf:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
    case FloatLiteralSpecialValues::NegZero:
    case FloatLiteralSpecialValues::NegInf:
    case FloatLiteralSpecialValues::Inf:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Normal:
      if (rhs == 0)
        return FloatLiteralSpecialValues::Nan;
      if (rhs < 0)
        return FloatLiteralSpecialValues::Inf;
      return FloatLiteralSpecialValues::NegInf;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::Nan:
    return FloatLiteralSpecialValues::Nan;
  case FloatLiteralSpecialValues::Normal:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Inf:
      if (lhs < 0)
        return FloatLiteralSpecialValues::NegZero;
      return FloatLiteralSpecialValues::Normal;
    case FloatLiteralSpecialValues::NegInf:
      if (lhs < 0)
        return FloatLiteralSpecialValues::Normal;
      return FloatLiteralSpecialValues::NegZero;
    case FloatLiteralSpecialValues::NegZero:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Normal:
      llvm_unreachable("double normal case handled above");
    }
  }
  llvm_unreachable("unknown FloatLiteral special type");
}

static std::tuple<FloatLiteralSpecialValues, IPRational>
floatLiteralDiv(FloatLiteralSpecialValues lSpecial,
                FloatLiteralSpecialValues rSpecial, IPRational lhs,
                IPRational rhs) {
  if (isNormal(lSpecial) && isNormal(rSpecial)) {
    if (rhs == 0)
      return {FloatLiteralSpecialValues::Nan, 0};
    IPRational ratResult = lhs / rhs;
    if (lhs == 0 && rhs < 0)
      return {FloatLiteralSpecialValues::NegZero, 0};
    return {FloatLiteralSpecialValues::Normal, ratResult};
  };
  return {floatLiteralDivSpecialCases(lSpecial, rSpecial, lhs, rhs), 0};
}

OpFoldResult FloatLiteralBinop::fold(FoldAdaptor adaptor) {
  FloatLiteralAttr lAttr = dyn_cast_or_null<FloatLiteralAttr>(adaptor.getLhs());
  FloatLiteralAttr rAttr = dyn_cast_or_null<FloatLiteralAttr>(adaptor.getRhs());
  FloatLiteralBinopKind oper = adaptor.getOper();
  if (!lAttr || !rAttr)
    return {};
  FloatLiteralSpecialValues lSpecial = lAttr.getSpecial().getValue();
  FloatLiteralSpecialValues rSpecial = rAttr.getSpecial().getValue();
  IPRational lhs;
  IPRational rhs;
  if (isNormal(lSpecial)) {
    assert(lAttr.getRational().has_value() &&
           "rational has value when special value is normal");
    lhs = lAttr.getRational().value();
  }
  if (isNormal(rSpecial)) {
    assert(rAttr.getRational().has_value() &&
           "rational has value when special value is normal");
    rhs = rAttr.getRational().value();
  }

  auto mkAttr = [&](FloatLiteralSpecialValues resultSpecial,
                    IPRational rational) -> FloatLiteralAttr {
    return FloatLiteralAttr::get(
        lAttr.getContext(),
        FloatLiteralSpecialValuesAttr::get(lAttr.getContext(), resultSpecial),
        rational);
  };

  switch (oper) {
  case FloatLiteralBinopKind::Add: {
    auto [resultSpecial, rational] =
        floatLiteralAdd(lSpecial, rSpecial, lhs, rhs);
    return mkAttr(resultSpecial, rational);
  } break;
  case FloatLiteralBinopKind::Sub: {
    auto [resultSpecial, rational] =
        floatLiteralSub(lSpecial, rSpecial, lhs, rhs);
    return mkAttr(resultSpecial, rational);
  } break;
  case FloatLiteralBinopKind::Mul: {
    auto [resultSpecial, rational] =
        floatLiteralMul(lSpecial, rSpecial, lhs, rhs);
    return mkAttr(resultSpecial, rational);
  } break;
  case FloatLiteralBinopKind::TrueDiv: {
    auto [resultSpecial, rational] =
        floatLiteralDiv(lSpecial, rSpecial, lhs, rhs);
    return mkAttr(resultSpecial, rational);
  } break;
  }
  llvm_unreachable("unknown FloatLiteralBinop type");
}

//===----------------------------------------------------------------------===//
// FloatLiteralConvertOp
//===----------------------------------------------------------------------===//

/// Take an IPRational along with a specification for an output float type and
/// return the IEEE-style float bit string as an APInt.
static APInt floatLiteralConvertGetBitstring(IPRational input,
                                             unsigned totalLength,
                                             unsigned exponentLength,
                                             unsigned bias) {
  // Throughout this function I use “significand” to mean the float value
  // including the digit before the decimal, and “mantissa” to mean just the
  // part after the decimal, IE the bit pattern that is actually present in the
  // float value.  That's not technically correct, but it was helpful for me to
  // distinguish the two.

  unsigned mantissaLength = totalLength - exponentLength - 1;
  IPInt maxExponentZeroBias = (IPInt(1) << exponentLength) - 1;
  IPInt maxExponent = maxExponentZeroBias - bias;
  IPInt minExponent = IPInt(-1) * IPInt(bias - 1);

  // The maxSignificandIPIntLength is longer than the float mantissa bit width
  // to allow for:
  // * leading 0 in IPInt format
  // * most significant 1 bit that is removed in final encoding
  // * extra precision bits to ensure correct rounding
  unsigned maxSignificandIPIntRoundedLength = mantissaLength + 2;
  static const unsigned kSignificandRoundingLength = 3;
  unsigned maxSignificandIPIntLength =
      maxSignificandIPIntRoundedLength + kSignificandRoundingLength;

  // To support subnormal numbers (IE numbers with minimum exponent that have an
  // implicit leading 0 instead of implicit leading 1), we need to support lower
  // exponents during calculation.
  IPInt minCalculationExponent = minExponent - mantissaLength;

  if (input.getNumerator() == 0)
    return APInt(totalLength, 0);

  bool negativeSign = input.getNumerator() < 0;
  APInt signBits = APInt(totalLength, negativeSign ? 1 : 0);
  signBits = signBits << (totalLength - 1);

  IPInt initialNumerator = input.getNumerator().abs();
  const IPInt &denominator = input.getDenominator();
  IPInt significand = initialNumerator / denominator;
  IPInt remainder = initialNumerator % denominator;
  IPInt exponent = 0;
  bool exponentFinalized = false;
  if (significand > 0) {
    // The IPInt encoding of the number will have a leading 0 bit (because it is
    // positive), and the exponent when treating the most significant one bit is
    // one less than the number of bits representing the number with no leading
    // zeroes.
    exponent = significand.getAPInt().getBitWidth() - 2;
    exponentFinalized = true;
  }

  auto keepDoingLongDivision = [&]() -> bool {
    if (remainder == 0)
      return false;
    if (exponent < minCalculationExponent || exponent > maxExponent)
      return false;
    if (significand.getAPInt().getBitWidth() > maxSignificandIPIntLength)
      return false;
    return true;
  };

  // Do long division loop.
  while (keepDoingLongDivision()) {
    unsigned nBitsToShift = denominator.getAPInt().getBitWidth() -
                            remainder.getAPInt().getBitWidth();
    if (nBitsToShift == 0)
      nBitsToShift = 1;
    IPInt nCur = remainder << nBitsToShift;
    if (!exponentFinalized) {
      exponent = exponent - nBitsToShift;
    }
    IPInt quotient = nCur / denominator;
    remainder = nCur % denominator;
    if (quotient > 0)
      exponentFinalized = true;
    significand = (significand << nBitsToShift) + quotient;
  }

  // Early return for obvious zero case because our later logic requires a
  // non-zero significand.
  if (significand == 0)
    return signBits;

  // Pad to mantissa length before performing rounding, etc.
  if (significand.getAPInt().getBitWidth() < maxSignificandIPIntLength) {
    significand = significand << (maxSignificandIPIntLength -
                                  significand.getAPInt().getBitWidth());
  }

  auto performRounding = [](IPInt &significand, IPInt &exponent,
                            unsigned maxSignificandIPIntRoundedLength) {
    APInt roundingBits = significand.getAPInt().extractBits(
        /*numBits=*/significand.getAPInt().getBitWidth() -
            maxSignificandIPIntRoundedLength,
        /*bitPosition=*/0);
    unsigned roundingBitsActualLength = roundingBits.getBitWidth();
    APInt roundingMidpoint = APInt(roundingBitsActualLength, 1)
                             << (roundingBitsActualLength - 1);
    // Truncate bits first.
    significand = significand >> roundingBitsActualLength;
    // Now that we've truncated, rounding either means doing nothing (for
    // round toward zero) or adding one to the significand representation
    // (for rounding away from zero). The default rounding mode for IEEE
    // floats is “round to nearest, ties to even”. It might be good to take
    // an option to do other rounding modes, but for now we just support the
    // default.
    if (roundingBits.ugt(roundingMidpoint))
      significand = significand + 1;
    else if (roundingBits == roundingMidpoint && significand % 2 == 1)
      significand = significand + 1;
    // If rounding up increased digit count, we need to convert that into a
    // larger exponent and re-truncate.
    if (significand.getAPInt().getBitWidth() >
        maxSignificandIPIntRoundedLength) {
      exponent = exponent + 1;
      significand = significand >> 1;
    }
  };

  // Do rounding now unless we are dealing with a subnormal number, which needs
  // some extra handling before rounding.
  if (exponent >= minExponent)
    performRounding(significand, exponent, maxSignificandIPIntRoundedLength);

  if (exponent > maxExponent) {
    // Return +/- infinity.
    APInt exponentOnes = APInt::getAllOnes(exponentLength);
    APInt exponentBits = APInt(totalLength, 0);
    exponentBits.insertBits(exponentOnes, mantissaLength);
    // Mantissa for infinity is zero.
    return signBits | exponentBits;
  }

  // Handle subnormal numbers, including zero valuess.  (I'm not sure whether
  // zero counts technically as a subnormal number, but it fits the subnormal
  // encoding.)
  if (exponent < minExponent) {
    // Below the minExponent we can still convert to subnormal numbers.
    // The subnormal range is tagged with minExponent - 1, but the exponent
    // value is effectively the same as minExponent. However, instead of an
    // implicit leading 1 before the decimal, there is a leading 0. So subnormal
    // numbers cover down to minExponent - (mantissaWidth - 1) exponent, but
    // losing one bit of mantissa precision for each exponent lowering.
    IPInt minSubnormalExponent = minExponent - (mantissaLength - 1);
    if (exponent < minSubnormalExponent) {
      // We could let this fall through and be handled by the shifting and bit
      // mangling, but at this point we know that every bit is zero except
      // (maybe) the sign.
      return signBits;
    }
    IPInt shiftBits = minExponent - exponent;
    IPInt shiftTag = IPInt(1) << (IPInt(significand.getAPInt().getBitWidth()) -
                                  IPInt(2) + shiftBits);
    // The significand is now
    // `01<correct-bit-pattern><at-least-one-extra-bit>`.
    significand = shiftTag + significand;
    exponent = minExponent - 1;
    // If rounding increases the exponent and carries to a new high bit, then we
    // end up at 1000... for the significand with minExponent, and thus the
    // right number.  Cool.
    performRounding(significand, exponent, maxSignificandIPIntRoundedLength);
  }

  // Whether or not the value was subnormal, the significand now has the bit
  // pattern `01<correct-bit-pattern><maybe-extra-bit-due-to-rounding>`.  So we
  // drop the leading 2 bits and the trailing extra bits to arrive at the final
  // bit pattern for the mantissa.

  unsigned extraSignificandBits =
      significand.getAPInt().getBitWidth() - (mantissaLength + 2);
  significand = significand >> extraSignificandBits;
  assert(significand.getAPInt().getBitWidth() == mantissaLength + 2 &&
         "proper mantissa bit length");
  APInt mantissaLowBits = significand.getAPInt().extractBits(
      /*numBits=*/mantissaLength,
      /*bitPosition=*/0);
  APInt mantissaBits = APInt(totalLength, 0);
  mantissaBits.insertBits(mantissaLowBits, /*bitPosition=*/0);

  // Floating point numbers encode the exponent as `bias + exponent`, so that
  // the result is always a natural number, where `bias + exponent = 0`
  // signifies subnormal (including zero) numbers, and all ones is the
  // exponent for infinity and the NAN values.
  exponent = exponent + bias;
  // Place the bits into an APInt at the appropriate place.
  APInt exponentBits = APInt(totalLength, 0);
  exponentBits.insertBits(exponent.getAPInt(), mantissaLength);

  // Combine pieces to get final bit string: <sign><exponent><mantissa>.
  return signBits | exponentBits | mantissaBits;
}

static ErrorTreeOr<FloatAttr>
floatLiteralConvertOpHelper(FloatLiteralSpecialValues special,
                            std::optional<IPRational> inRat, Type outType,
                            Location loc) {
  unsigned totalLength = 0;
  unsigned exponentLength = 0;
  unsigned bias = 0;
  llvm::APFloatBase::Semantics semantics = llvm::APFloatBase::S_IEEEhalf;

  if (outType.isF16()) {
    totalLength = 16;
    exponentLength = 5;
    bias = 15;
    semantics = llvm::APFloatBase::S_IEEEhalf;
  } else if (outType.isBF16()) {
    totalLength = 16;
    exponentLength = 8;
    bias = 127;
    semantics = llvm::APFloatBase::S_BFloat;
  } else if (outType.isF32()) {
    totalLength = 32;
    exponentLength = 8;
    bias = 127;
    semantics = llvm::APFloatBase::S_IEEEsingle;
  } else if (outType.isF64()) {
    totalLength = 64;
    exponentLength = 11;
    bias = 1023;
    semantics = llvm::APFloatBase::S_IEEEdouble;
  } else if (outType.isF80()) {
    totalLength = 80;
    exponentLength = 15;
    bias = 16383;
    semantics = llvm::APFloatBase::S_x87DoubleExtended;
  } else if (outType.isF128()) {
    totalLength = 128;
    exponentLength = 15;
    bias = 16383;
    semantics = llvm::APFloatBase::S_IEEEquad;
  } else {
    return ErrorTree(
        loc, Error("float literal conversion: unsupported output type"));
  }

  APFloat resultValue =
      APFloat::getNaN(llvm::APFloatBase::EnumToSemantics(semantics));
  switch (special) {
  case FloatLiteralSpecialValues::Nan:
    resultValue =
        APFloat::getNaN(llvm::APFloatBase::EnumToSemantics(semantics));
    break;
  case FloatLiteralSpecialValues::Inf:
    resultValue = APFloat::getInf(llvm::APFloatBase::EnumToSemantics(semantics),
                                  /*negative=*/false);
    break;
  case FloatLiteralSpecialValues::NegInf:
    resultValue = APFloat::getInf(llvm::APFloatBase::EnumToSemantics(semantics),
                                  /*negative=*/true);
    break;
  case FloatLiteralSpecialValues::NegZero:
    resultValue = APFloat::getZero(
        llvm::APFloatBase::EnumToSemantics(semantics), /*negative=*/true);
    break;
  case FloatLiteralSpecialValues::Normal: {
    assert(inRat.has_value() && "normal FloatLiteral values have a rational");
    APInt floatBits = floatLiteralConvertGetBitstring(
        inRat.value(), totalLength, exponentLength, bias);
    resultValue =
        APFloat(llvm::APFloatBase::EnumToSemantics(semantics), floatBits);
  } break;
  }
  return FloatAttr::get(outType, resultValue);
}

ErrorTreeOrSuccess
FloatLiteralConvertOp::interpret(ArrayRef<Attribute> operands,
                                 InterpreterState &state) {
  assert(!operands.empty() && "FloatLiteralConvertOp must have an operand");
  auto inval = ::dyn_cast<FloatLiteralAttr>(operands[0]);
  ErrorTreeOr<FloatAttr> errOrAttr = floatLiteralConvertOpHelper(
      inval.getSpecial().getValue(), inval.getRational(), getType(), getLoc());
  if (errOrAttr.hasValue())
    state.mapResults(errOrAttr.getValue());
  else
    return errOrAttr.takeError();
  return success();
}

OpFoldResult FloatLiteralConvertOp::fold(FoldAdaptor adaptor) {
  auto in = dyn_cast_if_present<FloatLiteralAttr>(adaptor.getInput());
  if (!in)
    return {};
  ErrorTreeOr<FloatAttr> errOrAttr = floatLiteralConvertOpHelper(
      in.getSpecial().getValue(), in.getRational(), getType(), getLoc());
  if (errOrAttr.hasValue())
    return errOrAttr.getValue();
  return {};
}

//===----------------------------------------------------------------------===//
// FloatLiteralToIntLiteral
//===----------------------------------------------------------------------===//

static IntLiteralAttr FloatLiteralToIntLiteralOpHelper(FloatLiteralAttr fattr) {
  IPInt result;
  switch (fattr.getSpecial().getValue()) {
  case FloatLiteralSpecialValues::Nan:
  case FloatLiteralSpecialValues::Inf:
  case FloatLiteralSpecialValues::NegInf:
  case FloatLiteralSpecialValues::NegZero:
    result = 0;
    break;
  case FloatLiteralSpecialValues::Normal:
    assert(fattr.getRational().has_value() &&
           "normal FloatLiterals have rational");
    result = fattr.getRational()->getNumerator() /
             fattr.getRational()->getDenominator();
    break;
  }
  return IntLiteralAttr::get(fattr.getContext(), result);
}

ErrorTreeOrSuccess
FloatLiteralToIntLiteralOp::interpret(ArrayRef<Attribute> operands,
                                      InterpreterState &state) {
  assert(!operands.empty() &&
         "FloatLiteralToIntLiteralOp must have an operand");
  auto inval = ::dyn_cast<FloatLiteralAttr>(operands[0]);
  IntLiteralAttr attrResult = FloatLiteralToIntLiteralOpHelper(inval);
  state.mapResults(attrResult);
  return success();
}

OpFoldResult FloatLiteralToIntLiteralOp::fold(FoldAdaptor adaptor) {
  auto in = dyn_cast_if_present<FloatLiteralAttr>(adaptor.getInput());
  if (!in)
    return {};
  return FloatLiteralToIntLiteralOpHelper(in);
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
    elementTypes.push_back(ParamRefType::get(value));
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
    Type type = ParamRefType::get(expected);
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
  return ParamRefType::get(typeAttr);
}

LogicalResult PackExtractOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto emitError = [&](const Twine &msg) -> LogicalResult {
    return mlir::emitOptionalError(loc, msg);
  };
  if (operands.size() != 1 || !isa<PackType>(operands[0].getType()))
    return emitError("expected 1 operand");

  auto indexAttr = dyn_cast_if_present<TypedAttr>(attrs.get("index"));
  if (!indexAttr || !indexAttr.getType().isIndex())
    return emitError("expected an index attribute");

  auto packType = cast<PackType>(operands[0].getType());
  inferredReturnTypes.push_back(getPackFieldAtIndex(packType, indexAttr));
  return success();
}

//===----------------------------------------------------------------------===//
// PackGEPOp
//===----------------------------------------------------------------------===//

LogicalResult PackGEPOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto emitError = [&](const Twine &msg) -> LogicalResult {
    return mlir::emitOptionalError(loc, msg);
  };
  if (operands.size() != 1 || !isa<PointerType>(operands[0].getType()))
    return emitError("expected 1 operand");
  auto packType = dyn_cast<PackType>(
      cast<PointerType>(operands[0].getType()).getElementType());
  if (!packType)
    return emitError("expected pointer to pack type");

  auto indexAttr = dyn_cast_if_present<TypedAttr>(attrs.get("index"));
  if (!indexAttr || !indexAttr.getType().isIndex())
    return emitError("expected an index attribute");

  inferredReturnTypes.push_back(
      PointerType::get(getPackFieldAtIndex(packType, indexAttr)));
  return success();
}

ErrorTreeOrSuccess PackGEPOp::interpret(ArrayRef<Attribute> operands,
                                        InterpreterState &state) {
  auto ptr = dyn_cast_if_present<PointerAttr>(operands[0]);
  auto idxAttr = dyn_cast_if_present<IntegerAttr>(getIndex());
  if (!ptr || !idxAttr)
    return ErrorTree(getLoc(), "non-constant inputs");

  int64_t offset = 0;
  auto variadic =
      getPack().getType().getElementAs<PackType>().getVariadicIfResolved();
  if (!variadic)
    return ErrorTree(getLoc(), "unknown type list");

  ArrayRef<TypedAttr> typeElts = variadic.getValues();

  // Move the address over the elements before the one we are reading.
  unsigned index = idxAttr.getInt();
  for (unsigned i = 0; i != index; ++i) {
    auto eltType = cast<TypeConstantAttr>(typeElts[i]).getValue();
    auto dl = cast<DataLayoutInterface>(eltType);
    offset = llvm::alignTo(offset, *dl.getTypeAlign(state.getTarget()));
    offset += *dl.getTypeSize(state.getTarget());
  }

  // Align the address to the target element.
  Type targetType = cast<TypeConstantAttr>(typeElts[index]).getValue();
  offset = llvm::alignTo(
      offset,
      *cast<DataLayoutInterface>(targetType).getTypeAlign(state.getTarget()));
  state.mapResults(
      PointerAttr::get(ptr.getAddr() + offset, PointerType::get(targetType)));
  return success();
}

//===----------------------------------------------------------------------===//
// StructCreateOp
//===----------------------------------------------------------------------===//

OpFoldResult StructCreateOp::fold(FoldAdaptor adaptor) {
  ArrayRef<Attribute> operands = adaptor.getOperands();
  SmallVector<TypedAttr> values;
  values.reserve(operands.size());
  for (Attribute operand : operands) {
    auto value = llvm::cast_if_present<TypedAttr>(operand);
    if (!value)
      return {};
    values.push_back(value);
  }
  return StructAttr::get(values, getType());
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
                       StructType structType, DictionaryAttr attrs) {
  if (!structType)
    return emitError("expected struct operand");
  mlir::OperationName name(OpT::getOperationName(), attrs.getContext());
  auto indexAttr =
      dyn_cast_if_present<IntegerAttr>(attrs.get(OpT::getIndexAttrName(name)));
  if (!indexAttr)
    return emitError("expected an integer index attribute");
  size_t index = indexAttr.getInt();
  if (index >= structType.getNumElements())
    return emitError("struct element index out of bounds");
  return structType.getElementTypes()[index];
}

LogicalResult StructExtractOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto emitError = [&](const Twine &msg) -> LogicalResult {
    return mlir::emitOptionalError(location, msg);
  };
  if (operands.size() != 1)
    return emitError("expected 1 operand");
  auto structType = dyn_cast<StructType>(operands.front().getType());
  FailureOr<Type> type = inferStructElementType<StructExtractOp>(
      emitError, structType, attributes);
  if (succeeded(type))
    inferredReturnTypes.push_back(*type);
  return type;
}

OpFoldResult StructExtractOp::fold(FoldAdaptor adaptor) {
  if (auto container = adaptor.getContainer())
    return StructExtractAttr::get(cast<TypedAttr>(container),
                                  getIndexAttr().getInt());
  if (auto structCreate = getOperand().getDefiningOp<StructCreateOp>())
    return structCreate.getOperand(adaptor.getIndex());
  return {};
}

//===----------------------------------------------------------------------===//
// StructReplaceOp
//===----------------------------------------------------------------------===//

static ParseResult parseStructValueType(AsmParser &p, Type &valueType,
                                        Type structType, IntegerAttr index) {
  ArrayRef<Type> elementTypes = structType.cast<StructType>().getElementTypes();
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

OpFoldResult StructReplaceOp::fold(FoldAdaptor adaptor) {
  auto value = llvm::cast_if_present<TypedAttr>(adaptor.getValue());
  auto container = dyn_cast_if_present<StructAttr>(adaptor.getContainer());
  if (!value || !container)
    return {};
  SmallVector<TypedAttr> values(container.getValues());
  values[getIndexAttr().getInt()] = value;
  return StructAttr::get(values, getType());
}

//===----------------------------------------------------------------------===//
// StructGEPOp
//===----------------------------------------------------------------------===//

LogicalResult StructGEPOp::verify() {
  return verifyStructValueType(
      *this, cast<StructType>(getContainer().getType().getElementType()),
      getIndexAttr(), getType().getElementType(), "result");
}

LogicalResult StructGEPOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto emitError = [&](const Twine &msg) -> LogicalResult {
    return mlir::emitOptionalError(location, msg);
  };
  if (operands.size() != 1)
    return emitError("expected 1 operand");
  auto pointerType = dyn_cast<PointerType>(operands.front().getType());
  if (!pointerType)
    return emitError("expected pointer operand");
  auto structType = dyn_cast<StructType>(pointerType.getElementType());
  FailureOr<Type> type = inferStructElementType<StructExtractOp>(
      emitError, structType, attributes);
  if (succeeded(type))
    inferredReturnTypes.push_back(PointerType::get(*type));
  return type;
}

ErrorTreeOrSuccess StructGEPOp::interpret(ArrayRef<Attribute> operands,
                                          InterpreterState &state) {
  auto ptr = dyn_cast_if_present<PointerAttr>(operands.front());
  if (!ptr)
    return ErrorTree(getLoc(), "non-constant inputs");

  int64_t offset = 0;
  auto structType = getContainer().getType().getElementAs<StructType>();

  // Move the address over the elements before the one we are reading.
  unsigned index = getIndexAttr().getInt();
  for (unsigned i = 0; i != index; ++i) {
    auto dl = cast<DataLayoutInterface>(structType.getElementTypes()[i]);
    offset = llvm::alignTo(offset, *dl.getTypeAlign(state.getTarget()));
    offset += *dl.getTypeSize(state.getTarget());
  }

  // Align the address to the target element.
  Type targetType = structType.getElementTypes()[index];
  offset = llvm::alignTo(
      offset,
      *cast<DataLayoutInterface>(targetType).getTypeAlign(state.getTarget()));
  state.mapResults(
      PointerAttr::get(ptr.getAddr() + offset, PointerType::get(targetType)));
  return success();
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
// VariantTakeOp
//===----------------------------------------------------------------------===//

LogicalResult VariantTakeOp::verify() {
  if (failed(verifyVariantIndex(*this, getVariant().getType(), getIndex())))
    return failure();
  Type elementType = getVariant().getType().getType(getIndex());
  if (elementType == getType())
    return success();
  return emitOpError("variant element at index ")
         << getIndex() << " expected type " << elementType
         << " but operand has type " << getType();
}

LogicalResult
VariantTakeOp::inferReturnTypes(MLIRContext *, std::optional<Location> loc,
                                ValueRange operands, DictionaryAttr attrs,
                                mlir::OpaqueProperties, RegionRange,
                                SmallVectorImpl<Type> &types) {
  VariantTakeOpAdaptor adaptor(operands, attrs);
  unsigned index = adaptor.getIndex();
  auto variant = cast<VariantType>(adaptor.getVariant().getType());
  if (index >= variant.getNumTypes())
    return mlir::emitOptionalError(loc, "variant element index ", index,
                                   " is out of bounds");
  types.push_back(variant.getType(index));
  return success();
}

///===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/KGENDialect/KGEN.cpp.inc"
