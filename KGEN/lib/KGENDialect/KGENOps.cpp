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
#include "Cache/CacheDialect/CacheOps.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// custom<ParamConstantOpValue>
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

LogicalResult ParamConstantOp::verify() {
  // Forbid the materialization of parameter capturing closures.
  if (auto sig = dyn_cast<SignatureType>(getType())) {
    if (sig.isCapturing())
      return emitOpError("cannot be used to materialize capturing closures; "
                         "use `kgen.create_closure` instead");
    if (!sig.getResultParamTypes().empty() || !sig.getInputParamTypes().empty())
      return emitOpError("cannot materialize parametric signatures; fully bind "
                         "the signature first");
  }

  if (getValue().getType() == getType())
    return success();
  return emitOpError() << "parameter type " << getValue().getType()
                       << " does not match result type " << getType();
}

void ParamConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  // If the type of the value has a registered pretty name, use that for the SSA
  // value name.
  std::optional<StringRef> name =
      getContext()->getLoadedDialect<KGENDialect>()->getTypeName(
          getType().getTypeID());
  if (name)
    setNameFn(getResult(), *name);
}

OpFoldResult ParamConstantOp::fold(FoldAdaptor adaptor) {
  auto constants = adaptor.getOperands();
  assert(constants.empty() && "kgen.param.constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// ParamDeclareOp
//===----------------------------------------------------------------------===//

static ParseResult parseParamDeclareOpValue(OpAsmParser &p,
                                            ParamDeclAttr &paramDecl,
                                            TypedAttr &value) {
  StringAttr name;
  Type type;
  if (parseParamName(p, name) || parseParamConstantOpValue(p, value, type))
    return failure();

  paramDecl = ParamDeclAttr::get(name, value.getType());
  return success();
}

static void printParamDeclareOpValue(OpAsmPrinter &p, Operation *,
                                     ParamDeclAttr paramDecl, TypedAttr value) {
  printParamName(p, paramDecl.getName());
  printParamConstantOpValue(p, nullptr, value, nullptr);
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

static ParseResult
parseRegionDeclaration(OpAsmParser &p, ParamDeclAttr &paramDecl,
                       ParamDeclArrayAttr &inputParams,
                       ParamDeclArrayAttr &resultParams, TypeAttr &functionType,
                       TypeAttr &signature, ConstraintArrayAttr &constraints,
                       AlwaysInlineLevelAttr &alwaysInlineLevel, Region &body) {
  StringAttr paramName;
  SmallVector<OpAsmParser::Argument> args;
  FunctionType functionTypeValue;
  SignatureType signatureType;
  llvm::SMLoc bodyLoc;
  if (parseParamName(p, paramName) || p.parseEqual() ||
      parseFunctionSignature(p, args, inputParams, resultParams,
                             functionTypeValue, signatureType) ||
      parseOptionalAlwaysInline(p, alwaysInlineLevel) ||
      parseOptionalConstraints(p, constraints) ||
      p.getCurrentLocation(&bodyLoc) || p.parseRegion(body, args))
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

static void
printRegionDeclaration(OpAsmPrinter &p, Operation *op, ParamDeclAttr paramDecl,
                       ParamDeclArrayAttr inputParams,
                       ParamDeclArrayAttr resultParams, TypeAttr functionType,
                       TypeAttr signature, ConstraintArrayAttr constraints,
                       AlwaysInlineLevelAttr alwaysInlineLevel, Region &body) {
  printParamName(p, paramDecl.getName());
  p << " = ";
  printFunctionSignature(p, body, inputParams, resultParams,
                         cast<FunctionType>(functionType.getValue()),
                         cast<SignatureType>(signature.getValue()));
  printOptionalAlwaysInline(p, alwaysInlineLevel);
  printOptionalConstraints(p, op, constraints);
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
// ParamForkOp
//===----------------------------------------------------------------------===//

static ParseResult parseParamForkOpValue(OpAsmParser &p,
                                         ParamDeclAttr &paramDecl,
                                         TypedAttr &value) {
  StringAttr name;
  Type valTy;

  if (parseParamName(p, name) || parseColonTypeOrIndex(p, valTy) ||
      p.parseEqual() || p.parseLess() ||
      parseParamValue(p, value, VariadicType::get(valTy)) || p.parseGreater())
    return failure();

  paramDecl = ParamDeclAttr::get(name, valTy);
  return success();
}

static void printParamForkOpValue(OpAsmPrinter &p, Operation *,
                                  ParamDeclAttr paramDecl, TypedAttr value) {
  printParamName(p, paramDecl.getName().getValue());
  printColonTypeOrIndex(
      p,
      ParamRefType::get(cast<VariadicType>(value.getType()).getElementType()));
  p << " = <";
  printParamValue(p, value);
  p << ">";
}

void ParamForkOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  walkDef(getParamDecl(), getValuesAttr());
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
          calleeType.getValueInputs(),
          [&](Type type) {
            return parseParamValue(p, operandValues.emplace_back(), type);
          },
          [&] { return p.parseComma(); }) ||
      p.parseRParen())
    return failure();
  if (calleeType.getValueResults().size() != 1)
    return p.emitError(sigLoc, "expected callee to have 1 result");
  paramDecl =
      ParamDeclAttr::get(paramName, calleeType.getValueResults().front());
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

void ParamApplyOp::concretizeCallee(mlir::IRRewriter &b,
                                    SymbolConstantAttr callee) {
  setCalleeAttr(callee);
}

//===----------------------------------------------------------------------===//
// CostOfOp
//===----------------------------------------------------------------------===//

void CostOfOp::concretizeCallee(mlir::IRRewriter &b,
                                SymbolConstantAttr callee) {
  setCalleeAttr(callee);
}

ErrorTreeOrSuccess CostOfOp::interpret(ArrayRef<Attribute> operands,
                                       InterpreterState &state) {
  return ErrorTree(getLoc(), "TODO: not implemented");
}

//===----------------------------------------------------------------------===//
// ParamEvaluateOp
//===----------------------------------------------------------------------===//

static ParseResult parseParamEvaluateOp(AsmParser &p, ParamDeclAttr &paramDecl,
                                        TypedAttr &evaluator,
                                        TypedAttr &candidates) {
  SignatureType evaluatorType;
  if (parseParamDecl(p, paramDecl) || p.parseEqual() ||
      parseParamValue(p, candidates, VariadicType::get(paramDecl.getType())) ||
      p.parseKeyword("with") || p.parseLSquare() ||
      parseKGENType(p, evaluatorType) || p.parseColon() ||
      parseParamValue(p, evaluator, evaluatorType) || p.parseRSquare())
    return failure();
  return success();
}

static void printParamEvaluateOp(AsmPrinter &p, Operation *op,
                                 ParamDeclAttr paramDecl, TypedAttr evaluator,
                                 TypedAttr candidates) {
  printParamDecl(p, paramDecl);
  p << " = ";
  printParamValue(p, candidates);
  p << " with [";
  printKGENType(p, evaluator.getType());
  p << ": ";
  printParamValue(p, evaluator);
  p << ']';
}

void ParamEvaluateOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  ParamDefValue def;
  def.exprs.push_back(getEvaluator());
  def.exprs.push_back(getCandidates());
  walkDef(getParamDecl(), def);
}

void ParamEvaluateOp::walkDeclarations(
    function_ref<void(ParamDeclAttr)> walkDecl) {
  walkDecl(getParamDecl());
}

void ParamEvaluateOp::renameDeclarations(ArrayRef<ParamDeclAttr> decls) {
  assert(decls.size() == 1);
  setParamDeclAttr(decls.front());
}

LogicalResult ParamEvaluateOp::verify() {
  auto sigType =
      cast<VariadicType>(getCandidates().getType()).getElementAsType();
  if (sigType != getParamDecl().getType())
    return emitOpError("candidates type does not match parameter type");
  auto evalType = cast<SignatureType>(getEvaluator().getType());
  if (!evalType.getInputParamTypes().empty() ||
      !evalType.getResultParamTypes().empty())
    return emitOpError("evaluator cannot be parametric");
  return success();
}

//===----------------------------------------------------------------------===//
// ParamResultBind
//===----------------------------------------------------------------------===//

void ParamResultBindOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  auto scope = (*this)->getParentOfType<DeclInterface>();
  for (auto [decl, value] : llvm::zip(scope.getResultParams(), getParameters()))
    walkDef(decl, value);
}

/// This operation is parametric even if it defines no parameters.
bool ParamResultBindOp::isImplicitlyParametric() { return true; }

LogicalResult ParamResultBindOp::verify() {
  auto scope = (*this)->getParentOfType<DeclInterface>();
  if (!scope)
    return emitOpError("expected to be nested beneath a declaration scope");
  return checkResultParameterTypes(*this, getParameters(), scope);
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

  // Check to see if this operation only depends on expressions known in the
  // signature of the generator.  If so, we can fold it into the constraint
  // list.
  SmallVector<ParamDeclRefAttr> parameterRefs;
  auto parent = op->getParentOfType<DeclInterface>();
  // FIXME: `kgen.param.if` should get constraints.
  if (!parent || isa<ParamIfOp>(parent))
    return failure();

  bool unused;
  collectParameterReferences(cond, parameterRefs, unused);
  ArrayRef<ParamDeclAttr> generatorInputParams = parent.getInputParams();

  // Check to see if the parameters referenced by the condition are all
  // defined by the generator.  If so, we can fold this into the constraint
  // list.
  if (llvm::all_of(parameterRefs, [&](ParamDeclRefAttr declRef) -> bool {
        return llvm::any_of(generatorInputParams, [&](ParamDeclAttr decl) {
          return decl.getName() == declRef.getName();
        });
      })) {
    // Ok, great, add this to the trait list of the enclosing operation.
    SmallVector<ConstraintAttr> constraints(parent.getConstraints());
    auto typedStringAttr = dyn_cast<StringAttr>(op.getMessageAttr());
    if (!typedStringAttr)
      return failure();
    auto msg = StringAttr::get(op->getContext(), typedStringAttr.getValue());
    constraints.push_back(ConstraintAttr::get(cond, msg, op.getLoc()));
    parent.setConstraints(constraints);
    op.erase();
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
// custom<CallOpParams>
//===----------------------------------------------------------------------===//

/// Parse the parameter spec for a call op.
/// parameter-decl   ::= identifier (`:` type)?
/// parameter-bind   ::= identifier (`:` type)? `=` attribute-value

/// param-decl-list  ::= parameter-decl (`,` parameter-decl)* | `(` `)`
/// param-bind-list  ::= parameter-bind (`,` parameter-bind)* | `(` `)`

/// parameter-spec   ::= `<` param-bind-list (`->` param-decl-list)? `>`
static ParseResult parseCallOpParams(OpAsmParser &p,
                                     ParameterExprArrayAttr &paramValues,
                                     ParamDeclArrayAttr &resultDecls,
                                     TypeArrayAttr &resultParamTypes) {

  if (p.parseOptionalLess()) {
    // If there is no <, then the params of the call op are empty, so set
    // paramValues and paramDecls to empty and return.
    paramValues = ParameterExprArrayAttr::get(p.getContext(), {});
    resultDecls = ParamDeclArrayAttr::get(p.getContext(), {});
    resultParamTypes = TypeArrayAttr::get(p.getContext(), {});
    return success();
  }

  // Parse the input list
  SmallVector<TypedAttr> values;
  if (p.parseOptionalLSquare()) {
    if (p.parseCommaSeparatedList([&] {
          return parseParamValueDefaultingToIndex(p, values.emplace_back());
        }))
      return failure();
  } else if (p.parseRSquare()) {
    return failure();
  }
  paramValues = ParameterExprArrayAttr::get(p.getContext(), values);

  // Check to see if we have results and parse them if so.
  if (p.parseOptionalArrow()) {
    resultDecls = ParamDeclArrayAttr::get(p.getContext(), {});
    resultParamTypes = TypeArrayAttr::get(p.getContext(), {});
    return p.parseGreater();
  }

  SmallVector<ParamDeclAttr> decls;
  SmallVector<Type> paramTypes;
  auto parseElt = [&]() -> ParseResult {
    StringAttr declName;
    Type type;
    if (parseParamName(p, declName) || parseColonTypeOrIndex(p, type))
      return failure();
    decls.push_back(ParamDeclAttr::get(declName, type));
    paramTypes.push_back(type);
    return success();
  };
  if (p.parseCommaSeparatedList(parseElt))
    return failure();
  resultDecls = ParamDeclArrayAttr::get(p.getContext(), decls);
  resultParamTypes = TypeArrayAttr::get(p.getContext(), paramTypes);
  return p.parseGreater();
}

static void printCallOpParams(OpAsmPrinter &p, Operation *op,
                              ArrayRef<TypedAttr> paramValues,
                              ArrayRef<ParamDeclAttr> resultDecls,
                              ArrayRef<Type> resultParamTypes) {
  if (paramValues.empty() && resultDecls.empty())
    return;
  p << "<";
  if (paramValues.empty())
    p << "[]";
  else
    llvm::interleaveComma(paramValues, p, [&](TypedAttr value) {
      printColonTypeParamValue(p, value);
    });
  if (!resultDecls.empty()) {
    p << " -> ";
    llvm::interleaveComma(resultDecls, p, [&](ParamDeclAttr decl) {
      printParamName(p, decl.getName());
      printColonTypeOrIndex(p, decl.getType());
    });
  }
  p << ">";
}

//===----------------------------------------------------------------------===//
// GeneratorOp
//===----------------------------------------------------------------------===//

/// Parses a KGEN Generator.
ParseResult GeneratorOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseGeneratorOrFunc(parser, result, GeneratorOrFuncKind::generator);
}

// Print the GeneratorOp using the shared printing logic.
void GeneratorOp::print(OpAsmPrinter &p) { printGeneratorOrFunc(p, *this); }

LogicalResult GeneratorOp::verify() { return verifyOneBlockOrCached(*this); }

Region *GeneratorOp::getCallableRegion() { return &getBodyRegion(); }

ArrayRef<Type> GeneratorOp::getCallableResults() {
  return getFunctionType().getResults();
}

ArrayAttr GeneratorOp::getCallableArgAttrs() { return nullptr; }

ArrayAttr GeneratorOp::getCallableResAttrs() { return nullptr; }

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

/// Parses a concrete KGEN func.
///
/// operation ::=
///   `kgen.func` function-signature function-attributes? function-body
///
ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseGeneratorOrFunc(parser, result, GeneratorOrFuncKind::func);
}

/// Print the FuncOp. We use a shared printer with the GeneratorOp since it is
/// a superset of what a func is.
void FuncOp::print(OpAsmPrinter &p) { printGeneratorOrFunc(p, *this); }

LogicalResult FuncOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // In a kgen.func, parameters are allowed to be defined (e.g. by calls with
  // output parameters), but not used.  This is because the elaborator must
  // already have been run, lowering these to concrete attribute values.
  mlir::AttrTypeWalker walker;
  ParamDeclRefAttr invalidRef;
  Operation *usingOp = nullptr;
  walker.addWalk([&](ParamDeclRefAttr ref) {
    invalidRef = ref;
    return WalkResult::interrupt();
  });
  WalkResult result = walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    auto walkTypes = [&](TypeRange types) -> LogicalResult {
      for (Type type : types) {
        if (walker.walk(type).wasInterrupted()) {
          usingOp = op;
          return failure();
        }
      }
      return success();
    };
    if (failed(walkTypes(op->getResultTypes())))
      return WalkResult::interrupt();
    for (Region &region : op->getRegions())
      if (failed(walkTypes(region.getArgumentTypes())))
        return WalkResult::interrupt();
    if (walker.walk(op->getAttrDictionary()).wasInterrupted()) {
      usingOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!result.wasInterrupted())
    return success();
  assert(invalidRef && usingOp && "expected an invalid reference");
  auto diag = usingOp->emitError("invalid use of parameter ")
              << invalidRef.getName() << " in kgen.func";
  diag.attachNote(getLoc()) << "within kgen.func '" << getName() << "'";

  return failure();
}

LogicalResult FuncOp::verify() {
  if (!llvm::all_of(getMetadata().getInputConventions(),
                    [](ValueInputConvention inputConv) {
                      return inputConv == ValueInputConvention::OwnedInReg;
                    }))
    return emitOpError("can only have default value input conventions");
  return verifyOneBlockOrCached(*this);
}

Region *FuncOp::getCallableRegion() { return &getBodyRegion(); }

ArrayRef<Type> FuncOp::getCallableResults() {
  return getFunctionType().getResults();
}

ArrayAttr FuncOp::getCallableArgAttrs() { return nullptr; }

ArrayAttr FuncOp::getCallableResAttrs() { return nullptr; }

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
  TypeArrayAttr resultParamTypes;
  if (p.parseAttribute(callee) ||
      parseCallOpParams(p, paramValues, paramDecls, resultParamTypes) ||
      p.parseOperandList(operands, AsmParser::Delimiter::Paren) ||
      p.parseColon())
    return failure();

  SignatureType signature;
  if (parseSignatureValues(p, TypeArrayAttr::get(p.getContext(), {}),
                           resultParamTypes, signature))
    return failure();
  calleeCst = SymbolConstantAttr::get(callee, paramValues, signature);
  llvm::append_range(operandTypes, signature.getValueInputs());
  llvm::append_range(resultTypes, signature.getValueResults());
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
  printSignatureValues(p, calleeCst.getType());
}

OperandRange CallOp::getArgOperands() { return getOperands(); }

mlir::CallInterfaceCallable CallOp::getCallableForCallee() {
  return getCalleeSymbol();
}

void CallOp::concretizeCallee(mlir::IRRewriter &b, SymbolConstantAttr callee) {
  setCalleeAttr(callee);
  setParamDecls({});
}

void CallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  auto symbol = callee.get<SymbolRefAttr>();
  setCalleeAttr(KGEN::SymbolConstantAttr::get(symbol, getCallee().getType()));
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
  b.replaceOpWithNewOp<CallOp>(*this, getResultTypes(), callee,
                               ArrayRef<ParamDeclAttr>(), getOperands());
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
  return {};
}

/// If the operand to a rebind is defined by a rebind, use the second rebind's
/// operand.
LogicalResult RebindOp::canonicalize(RebindOp op, PatternRewriter &rewriter) {
  RebindOp cur = op, parent;
  // Climb all the way to the top to avoid recursively invoking this pattern.
  // Do not fold rebinds across parameter domains, because this can lead to
  // collision of name-shadowed parameters #12242.
  auto nearestDecl = cur->getParentOfType<DeclInterface>();
  while ((parent = cur.getOperand().getDefiningOp<RebindOp>()) &&
         parent->getParentOfType<DeclInterface>() == nearestDecl)
    cur = parent;

  if (cur == op)
    return failure();
  rewriter.updateRootInPlace(op, [&] { op.setOperand(cur.getOperand()); });
  return success();
}

//===----------------------------------------------------------------------===//
// ExportOp
//===----------------------------------------------------------------------===//

static ParseResult parseExportOp(OpAsmParser &p, SymbolRefAttr &exported,
                                 StringAttr &alias) {
  if (p.parseOptionalKeyword("as")) {
    alias = exported.getLeafReference();
    return success();
  }
  return p.parseSymbolName(alias);
}

static void printExportOp(OpAsmPrinter &p, Operation *op,
                          SymbolRefAttr exported, StringAttr alias) {
  if (exported.getLeafReference().getValue() == alias)
    return;
  p << " as ";
  p.printSymbolName(alias);
}

LogicalResult ExportOp::verifySymbolUses(SymbolTableCollection &symbolTable) {

  // Just ensure we're exporting a symbol we can see.
  auto module = KGENModule::from(*this, symbolTable);
  SymbolRefAttr exported = getExported();
  auto func = module.lookup<FuncInterface>(exported);
  if (!func)
    return emitOpError("could not find referenced symbol '") << exported << "'";
  if (func.getAlwaysInlineLevel() != AlwaysInlineLevel::Disabled) {
    return func.emitError("function marked 'always_inline' cannot be exported")
               .attachNote(getLoc())
           << "function exported here";
  }
  return success();
}

LogicalResult ExportOp::verify() {
  if (getIsCExport() && !isCIdentifier(getAlias())) {
    return emitError("The alias name is not a valid C identifier, allowed "
                     "characters: [a-zA-Z0-9_]");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CallSignatureOp
//===----------------------------------------------------------------------===//

static ParseResult parseCallSignature(OpAsmParser &p, Type &typeOfCallee,
                                      SmallVectorImpl<Type> &argumentTypes,
                                      SmallVectorImpl<Type> &resultTypes) {
  TypeArrayAttr resultParams;
  TypeArrayAttr parameterValues;
  // We expect the following syntax: call_signature callee(dynamic args) :
  // (argTypes...) -> resultType

  SignatureType signature;
  if (parseSignatureValues(p, parameterValues, resultParams, signature))
    return failure();
  typeOfCallee = signature;
  llvm::append_range(argumentTypes, signature.getValueInputs());
  llvm::append_range(resultTypes, signature.getValueResults());
  return success();
}

static void printCallSignature(OpAsmPrinter &p, Operation *op, Type calleeType,
                               TypeRange argumentTypes, TypeRange resultTypes) {
  // We expect the following syntax: call_signature callee(dynamic args) :
  // (argTypes...) capturing -> calleeResultType
  if (SignatureType sigType = cast<SignatureType>(calleeType)) {
    printSignature(p, sigType);
  }
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
  printFunctionSignature(p, body, {}, {}, resultType.getValues(), resultType);
  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// CreateClosureOp
//===----------------------------------------------------------------------===//

void CreateClosureOp::concretizeCallee(mlir::IRRewriter &b,
                                       SymbolConstantAttr callee) {
  setCalleeAttr(callee);
}

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
  if (!callee)
    return mlir::emitOptionalError(
        loc, "'create_closure' expected TypedAttr 'callee'");
  auto sig = dyn_cast<SignatureType>(callee.getType());
  if (!sig)
    return mlir::emitOptionalError(
        loc, "'create_closure' attribute 'callee' must have SignatureType");

  unsigned numArgs = captures.size();
  if (numArgs > sig.getValueInputs().size()) {
    return mlir::emitOptionalError(loc, "provided ", numArgs,
                                   " operands but callee only has ",
                                   sig.getValueInputs().size(), " to bind");
  }

  SmallVector<Type> newArgTypes;
  SmallVector<ValueInputConvention> newInputConvs;
  for (unsigned i = numArgs, e = sig.getValueInputs().size(); i != e; ++i) {
    newArgTypes.push_back(sig.getValueInputs()[i]);
    newInputConvs.push_back(sig.getValueInputConventions()[i]);
  }

  ArrayRef<TypedAttr> newDefaultArgs = sig.getDefaultArguments();
  if (newArgTypes.size() < newDefaultArgs.size())
    newDefaultArgs = newDefaultArgs.take_back(newArgTypes.size());

  FnEffects effects = sig.getFnEffects();
  if (!captures.empty())
    effects = effects | FnEffects::Capturing;
  results.push_back(SignatureType::get(
      sig.getInputParamTypes(), sig.getResultParamTypes(),
      OpBuilder(ctx).getFunctionType(newArgTypes, sig.getValueResults()),
      MetadataAttr::get(ctx, newInputConvs, newDefaultArgs, effects)));
  return mlir::success();
}

static ParseResult
parseClosureCaptureTypes(AsmParser &p, TypedAttr callee,
                         ArrayRef<OpAsmParser::UnresolvedOperand> captures,
                         SmallVectorImpl<Type> &captureTypes) {
  auto sig = dyn_cast<SignatureType>(callee.getType());
  if (!sig)
    return p.emitError(p.getCurrentLocation(),
                       "expected type of callee to be SignatureType");

  unsigned numArgs = captures.size();
  if (numArgs > sig.getValueInputs().size()) {
    return p.emitError(p.getCurrentLocation(), "provided ")
           << numArgs << " operands but callee only has "
           << sig.getValueInputs().size() << " to bind";
  }

  for (unsigned i = 0; i != numArgs; ++i)
    captureTypes.push_back(sig.getValueInputs()[i]);
  return success();
}

static void printClosureCaptureTypes(AsmPrinter &p, Operation *,
                                     TypedAttr callee, ValueRange captures,
                                     TypeRange captureTypes) {}

LogicalResult CreateClosureOp::verify() {
  SignatureType sig = getCalleeType();
  if (getNumOperands() > sig.getValueInputs().size()) {
    return emitOpError("provided ")
           << getNumOperands() << " operands but callee only has "
           << sig.getValueInputs().size() << " to bind";
  }
  unsigned expectedArgs = sig.getValueInputs().size() - getNumOperands();
  if (getType().getValueInputs().size() != expectedArgs) {
    return emitOpError("result signature has ")
           << getType().getValueInputs().size() << " arguments but expected "
           << expectedArgs;
  }

  for (auto [i, type, argType] :
       llvm::enumerate(getOperandTypes(),
                       sig.getValueInputs().take_front(getNumOperands()))) {
    if (type != argType) {
      return emitOpError("operand #")
             << i << " has type " << type
             << " but callee argument type expected " << argType;
    }
  }
  for (auto [i, type, argType] :
       llvm::enumerate(getType().getValueInputs(),
                       sig.getValueInputs().drop_front(getNumOperands()))) {
    if (type != argType) {
      return emitOpError("result signature argument #")
             << i << " type is " << argType << " but expected to be " << type;
    }
  }

  if (!getCaptures().empty() && !getType().isCapturing())
    return emitOpError("has captures, so result signature must be 'capturing'");
  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/KGENDialect/KGEN.cpp.inc"
