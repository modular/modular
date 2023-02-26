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
#include "Support/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

/// This checks the arguments of the return op against the result parameters and
/// the result types iff the body of the op has not been cached.
template <typename T>
static LogicalResult checkReturnArguments(T op) {
  // If we have a return op, then we can check the argument types. Otherwise, we
  // just don't have the return op.
  if (ReturnOp returnOp = op.getReturnOp())
    return checkResultArgumentTypes(returnOp, returnOp.getParameters(),
                                    op.getResultParams(), op.getResultTypes());
  return success();
}

//===----------------------------------------------------------------------===//
// custom<ParamConstantOpValue>
//===----------------------------------------------------------------------===//

static ParseResult parseParamConstantOpValue(OpAsmParser &p, TypedAttr &value) {
  Type type;
  if (parseColonTypeOrIndex(p, type) || p.parseEqual() || p.parseLess() ||
      parseParamValue(p, value, type) || p.parseGreater())
    return failure();
  return success();
}

static void printParamConstantOpValue(OpAsmPrinter &p, Operation *,
                                      TypedAttr value) {
  printColonTypeOrIndex(p, value.getType());
  p << " = <";
  printParamValue(p, value);
  p << ">";
}

//===----------------------------------------------------------------------===//
// ParamDeclareOp
//===----------------------------------------------------------------------===//

static ParseResult parseParamDeclareOpValue(OpAsmParser &p,
                                            ParamDeclAttr &paramDecl,
                                            TypedAttr &value) {
  StringAttr name;
  if (parseParamName(p, name) || parseParamConstantOpValue(p, value))
    return failure();

  paramDecl = ParamDeclAttr::get(name, value.getType());
  return success();
}

static void printParamDeclareOpValue(OpAsmPrinter &p, Operation *,
                                     ParamDeclAttr paramDecl, TypedAttr value) {
  printParamName(p, paramDecl.getName());
  printParamConstantOpValue(p, nullptr, value);
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
                       TypeAttr &signature, ConstraintArrayAttr &constraints,
                       AlwaysInlineLevelAttr &alwaysInlineLevel, Region &body) {
  StringAttr paramName;
  SmallVector<OpAsmParser::Argument> args;
  SignatureType signatureType;
  llvm::SMLoc bodyLoc;
  if (parseParamName(p, paramName) || p.parseEqual() ||
      parseFunctionSignature(p, args, signatureType) ||
      parseOptionalAlwaysInline(p, alwaysInlineLevel) ||
      parseOptionalConstraints(p, constraints) ||
      p.getCurrentLocation(&bodyLoc) || p.parseRegion(body, args))
    return failure();

  // Form the Signature.
  SmallVector<Type> argTypes;
  for (const OpAsmParser::Argument &arg : args)
    argTypes.push_back(arg.type);
  signature = TypeAttr::get(signatureType);
  paramDecl = ParamDeclAttr::get(paramName, signatureType);
  return success();
}

static void printRegionDeclaration(OpAsmPrinter &p, Operation *op,
                                   ParamDeclAttr paramDecl, TypeAttr signature,
                                   ConstraintArrayAttr constraints,
                                   AlwaysInlineLevelAttr alwaysInlineLevel,
                                   Region &body) {
  auto sig = cast<SignatureType>(signature.getValue());
  printParamName(p, paramDecl.getName());
  p << " = ";
  printFunctionSignature(p, body, sig);
  printOptionalAlwaysInline(p, alwaysInlineLevel);
  printOptionalConstraints(p, op, constraints);
  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

LogicalResult ParamDeclareRegionOp::verifyRegions() {
  auto returnOp = cast<ReturnOp>(getBody()->getTerminator());
  if (failed(checkResultArgumentTypes(returnOp, returnOp.getParameters(),
                                      getResultParams(),
                                      getSignature().getValueResults())))
    return failure();
  if (getBody()->getArgumentTypes() != getSignature().getValueInputs())
    return emitOpError("signature mismatches body");
  return success();
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
      p, cast<VariadicType>(value.getType()).getResolvedElementType());
  p << " = <";
  printParamValue(p, value);
  p << ">";
}

void ParamForkOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  walkDef(getParamDecl(), getValuesAttr());
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

void ReturnOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  for (auto [decl, value] :
       llvm::zip(cast<FuncInterface>((*this)->getParentOp()).getResultParams(),
                 getParameters()))
    walkDef(decl, value);
}

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
  if (parent) {
    bool unused;
    collectParameterReferences(cond, parameterRefs, unused);
    ArrayRef<ParamDeclAttr> generatorInputParams = parent.getInputParamDecls();

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
  }

  return failure();
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
static ParseResult parseCallOpParams(OpAsmParser &p,
                                     ParamBindArrayAttr &paramValues,
                                     ParamDeclArrayAttr &resultDecls,
                                     ParamDeclArrayAttr &resultParams) {

  if (p.parseOptionalLess()) {
    // If there is no <, then the params of the call op are empty, so set
    // paramValues and paramDecls to empty and return.
    paramValues = ParamBindArrayAttr::get(p.getContext(), {});
    resultDecls = resultParams = ParamDeclArrayAttr::get(p.getContext(), {});
    return success();
  }

  // Parse the input list
  if (parseParamBinds(p, paramValues))
    return failure();

  // Check to see if we have results and parse them if so.
  if (p.parseOptionalArrow()) {
    resultDecls = resultParams = ParamDeclArrayAttr::get(p.getContext(), {});
    return p.parseGreater();
  }

  SmallVector<ParamDeclAttr> resultDeclValues, resultParamValues;
  auto parseElt = [&]() -> ParseResult {
    StringAttr declName, sigName;
    Type type;
    if (parseParamName(p, declName) || p.parseEqual() ||
        parseParamName(p, sigName) || parseColonTypeOrIndex(p, type))
      return failure();
    resultDeclValues.push_back(ParamDeclAttr::get(declName, type));
    resultParamValues.push_back(ParamDeclAttr::get(sigName, type));
    return success();
  };
  if (p.parseCommaSeparatedList(parseElt))
    return failure();
  resultDecls = ParamDeclArrayAttr::get(p.getContext(), resultDeclValues);
  resultParams = ParamDeclArrayAttr::get(p.getContext(), resultParamValues);
  return p.parseGreater();
}

static void printCallOpParams(OpAsmPrinter &p, Operation *op,
                              ParamBindArrayAttr paramValues,
                              ParamDeclArrayAttr paramDecls,
                              ParamDeclArrayAttr resultParams) {
  if (paramValues.empty() && paramDecls.empty())
    return;
  p << "<";
  printParamBinds(p, paramValues);
  if (!paramDecls.empty()) {
    p << " -> ";
    llvm::interleaveComma(llvm::zip(paramDecls, resultParams), p,
                          [&](auto pair) {
                            auto [decl, param] = pair;
                            printParamName(p, decl.getName());
                            p << " = ";
                            printParamName(p, param.getName());
                            printColonTypeOrIndex(p, param.getType());
                          });
  }
  p << ">";
}

//===----------------------------------------------------------------------===//
// GeneratorOp
//===----------------------------------------------------------------------===//

ReturnOp GeneratorOp::getReturnOp() {
  return getBodyRegion().empty() ? nullptr
                                 : cast<ReturnOp>(getBody()->getTerminator());
}

/// Parses a KGEN Generator.
ParseResult GeneratorOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseGeneratorOrFunc(parser, result, GeneratorOrFuncKind::generator);
}

// Print the GeneratorOp using the shared printing logic.
void GeneratorOp::print(OpAsmPrinter &p) { printGeneratorOrFunc(p, *this); }

LogicalResult
GeneratorOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // If the generator is implementing a generator interface, check that they
  // line up correctly.
  FlatSymbolRefAttr interfaceSym = getImplementsAttr();
  if (!interfaceSym)
    return success();

  // Check that the callee attribute was specified.
  auto module = KGENModule::from(*this, symbolTable);
  auto interface = module.lookup<GeneratorInterfaceOp>(interfaceSym);
  if (!interface)
    return emitError() << "'" << interfaceSym
                       << "' does not reference a generator interface";

  // Verify that the signature of this generator matches the signature of the
  // interface.
  return verifyDeclMatchesInterface("generator", *this, "interface", interface);
}

LogicalResult GeneratorOp::verifyRegions() {
  if (failed(verifyOneBlockOrCached(*this)))
    return failure();
  return checkReturnArguments(*this);
}

Region *GeneratorOp::getCallableRegion() { return &getBodyRegion(); }

ArrayRef<Type> GeneratorOp::getCallableResults() {
  return getFunctionType().getResults();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

ReturnOp FuncOp::getReturnOp() {
  return getBodyRegion().empty() ? nullptr
                                 : cast<ReturnOp>(getBody()->getTerminator());
}

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
  // already have been run, lowering these to concrete attibute values.
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
  // kgen.func's are not allowed to have input parameter lists.
  if (!getInputParamDecls().empty() || !getResultParams().empty())
    return emitOpError("cannot have input or result parameters");
  if (!llvm::all_of(getMetadata().getInputConventions(),
                    [](ValueInputConvention inputConv) {
                      return inputConv == ValueInputConvention::ByVal;
                    }))
    return emitOpError("can only have default value input conventions");
  return success();
}

LogicalResult FuncOp::verifyRegions() {
  if (failed(verifyOneBlockOrCached(*this)))
    return failure();
  return checkReturnArguments(*this);
}

Region *FuncOp::getCallableRegion() { return &getBodyRegion(); }

ArrayRef<Type> FuncOp::getCallableResults() {
  return getFunctionType().getResults();
}

//===----------------------------------------------------------------------===//
// GeneratorInterfaceOp
//===----------------------------------------------------------------------===//

/// Parses a KGEN generator interface.
ParseResult GeneratorInterfaceOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  if (failed(
          parseGeneratorOrFunc(parser, result, GeneratorOrFuncKind::interface)))
    return failure();

  // Parse an optional evaluator.
  if (parser.parseOptionalKeyword("evaluator")) {
    // If we don't have an evaluator, we must not have a defaultImpl.
    if (succeeded(parser.parseOptionalKeyword("defaultImpl")))
      return mlir::emitError(
          parser.getEncodedSourceLoc(parser.getCurrentLocation()),
          "cannot specify a default without an evaluator");

    return success();
  }

  Type sigType;
  TypedAttr evaluator;
  if (parseKGENType(parser, sigType) || parser.parseEqual() ||
      parseParamValue(parser, evaluator, sigType))
    return failure();
  result.addAttribute(GeneratorInterfaceOp::getEvaluatorAttrName(result.name),
                      evaluator);

  // If it has an evaluator, the user may specify a default.
  if (parser.parseOptionalKeyword("defaultImpl"))
    return success();

  TypedAttr defaultImpl;
  if (parseKGENType(parser, sigType) || parser.parseEqual() ||
      parseParamValue(parser, defaultImpl, sigType))
    return failure();
  result.addAttribute(GeneratorInterfaceOp::getDefaultImplAttrName(result.name),
                      defaultImpl);

  return success();
}

// Print the GeneratorInterfaceOp using the shared printing logic.
void GeneratorInterfaceOp::print(OpAsmPrinter &p) {
  printGeneratorOrFunc(p, *this);
  if (SymbolConstantAttr evaluator = getEvaluatorAttr()) {
    p << " evaluator ";
    printKGENType(p, evaluator.getType());
    p << " = ";
    printParamValue(p, evaluator);
  }

  if (SymbolConstantAttr defaultImpl = getDefaultImplAttr()) {
    p << " defaultImpl ";
    printKGENType(p, defaultImpl.getType());
    p << " = ";
    printParamValue(p, defaultImpl);
  }
}

LogicalResult
GeneratorInterfaceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // If an evaluator was specified, verify its signature.
  SymbolConstantAttr evaluator = getEvaluatorAttr();
  if (!evaluator) {
    if (getDefaultImplAttr())
      return emitOpError(
          "defaultImpl should not exist if evaluator does not exist");
    return success();
  }

  auto module = KGENModule::from(*this, symbolTable);
  auto func = module.lookup<FuncInterface>(evaluator.getSymbol());
  if (!func)
    return emitOpError() << evaluator.getSymbol()
                         << " does not reference a KGEN declaration";

  // Get the specialized callee signature.
  SignatureType funcSignature = func.getSignature().getSpecializedSignature(
      evaluator.getParamValues(), [&] { return emitError(); });
  if (!funcSignature)
    return failure();

  // If a defaultImpl was specified, verify its signature.
  SymbolConstantAttr defaultImpl = getDefaultImplAttr();
  if (!defaultImpl)
    return success();

  func = module.lookup<GeneratorOp>(defaultImpl.getSymbol());
  if (!func)
    return emitOpError("defaultImpl ")
           << defaultImpl.getSymbol() << " must be a generator";

  funcSignature = func.getSignature().getSpecializedSignature(
      defaultImpl.getParamValues(), [&] { return emitError(); });
  if (!funcSignature)
    return failure();

  if (failed(verifyDeclSignaturesMatch("interface", getSignature(), getLoc(),
                                       "referenced defaultImpl", funcSignature,
                                       func.getLoc())))
    return failure();

  return success();
}

/// Return null to indicate that this is an "external" callable.
Region *GeneratorInterfaceOp::getCallableRegion() { return nullptr; }

ArrayRef<Type> GeneratorInterfaceOp::getCallableResults() {
  return getResultTypes();
}

//===----------------------------------------------------------------------===//
// AddressOfOp
//===----------------------------------------------------------------------===//

static ParseResult parseAddressOfOp(OpAsmParser &p,
                                    SymbolConstantAttr &calleeCst,
                                    ParamDeclArrayAttr &paramDecls,
                                    Type &resultType) {
  SymbolRefAttr callee;
  ParamBindArrayAttr paramValues;
  ParamDeclArrayAttr resultParams;
  SignatureType signature;
  if (p.parseAttribute(callee) ||
      parseCallOpParams(p, paramValues, paramDecls, resultParams) ||
      p.parseColon())
    return failure();

  if (parseSignatureValues(p, ParamDeclArrayAttr::get(p.getContext(), {}),
                           resultParams, signature))
    return failure();
  calleeCst = SymbolConstantAttr::get(callee, paramValues, signature);
  resultType = signature.getValues();
  return success();
}

static void printAddressOfOp(OpAsmPrinter &p, Operation *op,
                             SymbolConstantAttr calleeCst,
                             ParamDeclArrayAttr paramDecls, Type resultType) {
  p << calleeCst.getSymbol();
  printCallOpParams(p, op, calleeCst.getParamValues(), paramDecls,
                    calleeCst.getType().getResultParams());
  p << " : ";
  printSignatureValues(p, calleeCst.getType());
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
  ParamBindArrayAttr paramValues;
  ParamDeclArrayAttr resultParams;
  if (p.parseAttribute(callee) ||
      parseCallOpParams(p, paramValues, paramDecls, resultParams) ||
      p.parseOperandList(operands, AsmParser::Delimiter::Paren) ||
      p.parseColon())
    return failure();

  SignatureType signature;
  if (parseSignatureValues(p, ParamDeclArrayAttr::get(p.getContext(), {}),
                           resultParams, signature))
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
                    calleeCst.getType().getResultParams());
  p << '(';
  p.printOperands(operands);
  p << ") : ";
  printSignatureValues(p, calleeCst.getType());
}

OperandRange CallOp::getArgOperands() { return getOperands(); }

mlir::CallInterfaceCallable CallOp::getCallableForCallee() {
  return getCalleeSymbol();
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

//===----------------------------------------------------------------------===//
// ParamConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ParamConstantOp::fold(FoldAdaptor adaptor) {
  auto constants = adaptor.getOperands();
  assert(constants.empty() && "kgen.param.constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// ParamIfOp
//===----------------------------------------------------------------------===//

/// Parses a kgen.param.if op.
///
/// operation ::=
///   `kgen.param.if` `<` condition (`->` result-attrs)? `>` `{`
///     then-region
///   `} else {`
///     else-region
///   `}`
///
ParseResult ParamIfOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the condition.
  TypedAttr conditionAttr;
  if (parser.parseLess() ||
      parseParamValue(parser, conditionAttr, parser.getBuilder().getI1Type()))
    return failure();
  result.addAttribute(ParamIfOp::getCondAttrName(result.name), conditionAttr);

  // We might have result params, parse them if we do.
  auto decls = parser.getBuilder().getAttr<ParamDeclArrayAttr>(
      ArrayRef<ParamDeclAttr>{});
  if (succeeded(parser.parseOptionalArrow()))
    if (parseParamDecls(parser, decls))
      return failure();
  result.addAttribute(ParamIfOp::getParamDeclsAttrName(result.name), decls);

  // Parse any possible result types.
  SmallVector<Type> resultTypes;
  if (parser.parseGreater() || parser.parseOptionalArrowTypeList(resultTypes))
    return failure();
  result.addTypes(resultTypes);

  // Parse the `then` and `else` regions.
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();
  if (parser.parseRegion(*thenRegion) || parser.parseKeyword("else") ||
      parser.parseRegion(*elseRegion))
    return failure();

  return success();
}

/// Print the FuncOp. We use a shared printer with the GeneratorOp since it is
/// a superset of what a func is.
void ParamIfOp::print(OpAsmPrinter &p) {
  p << " <";
  printParamValue(p, getCond(), getCond().getType());
  if (!getParamDecls().empty()) {
    p << " -> ";
    printParamDecls(p, getParamDecls());
  }

  p << "> ";
  p.printRegion(getThenRegion());
  p << " else ";
  p.printRegion(getElseRegion());
}

LogicalResult ParamIfOp::verify() {
  TypeRange resultTypes = getResultTypes();
  auto checkTypesMatch = [&](ValueRange other) -> LogicalResult {
    for (auto [ifResult, operand] : llvm::zip(resultTypes, other))
      if (ifResult != operand.getType())
        return mlir::emitError(operand.getLoc())
               << "expected type " << ifResult << " but got "
               << operand.getType();
    return success();
  };

  // Check that the yields in both have the same input types as result types.
  if (failed(checkTypesMatch(
          getThenRegion().front().getTerminator()->getOperands())))
    return failure();
  if (failed(checkTypesMatch(
          getElseRegion().front().getTerminator()->getOperands())))
    return failure();

  // Check that the result parameters work.
  auto checkResultParams = [&](Operation *terminator) -> LogicalResult {
    if (getParamDecls().empty())
      return success();

    if (!isa<ParamYieldOp>(terminator))
      return emitError("expected a kgen.param.yield in order to return result "
                       "parameters")
                 .attachNote(terminator->getLoc())
             << "unknown terminator defined here";

    auto yieldOp = cast<ParamYieldOp>(terminator);
    for (auto [decl, value] :
         llvm::zip(getParamDecls(), yieldOp.getParameters()))
      if (decl.getType() != value.getType())
        return (mlir::emitError(
                    terminator->getLoc(),
                    "result parameter type did not match, expected ")
                << decl.getType() << " but got " << value.getType())
                   .attachNote(getLoc())
               << "result parameter defined here";

    return success();
  };

  if (failed(checkResultParams(getThenRegion().front().getTerminator())))
    return failure();
  if (failed(checkResultParams(getElseRegion().front().getTerminator())))
    return failure();

  return success();
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
  llvm::for_each(getParamDecls(), walkDecl);
}

void ParamIfOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  ParamDefValue value(getCond(), {&getThenRegion(), &getElseRegion()});
  for (ParamDeclAttr decl : getParamDecls())
    walkDef(decl, value);
}

void ParamIfOp::renameDeclarations(ArrayRef<ParamDeclAttr> decls) {
  setParamDeclsAttr(ParamDeclArrayAttr::get(getContext(), decls));
}

bool ParamIfOp::isImplicitlyParametric() { return true; }

/// This operation has no uses to collect in the scopes it defines.
void ParamIfOp::collectParameterUsesBelow(
    function_ref<void(Attribute)> scanAttr, function_ref<void(Type)> scanType) {
}

//===----------------------------------------------------------------------===//
// ParamYieldOp
//===----------------------------------------------------------------------===//

bool ParamYieldOp::isParentNode(Operation *op) { return isa<ParamIfOp>(op); }

void ParamYieldOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  // Branch to after the if operation.
  targets.emplace_back(std::nullopt, getOperands());
}

void ParamYieldOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  for (auto [decl, value] :
       llvm::zip(cast<ParamIfOp>((*this)->getParentOp()).getParamDecls(),
                 getParameters()))
    walkDef(decl, value);
}

//===----------------------------------------------------------------------===//
// RebindOp
//===----------------------------------------------------------------------===//

/// If either the input or output type are parameterized, return success.
/// Otherwise, require that the concrete input and output types are the same.
LogicalResult RebindOp::verify() {
  SmallVector<ParamDeclRefAttr> inputRefs, outputRefs;
  bool inputConstExpr = false, outputConstExpr = false;
  collectParameterReferences(getInput().getType(), inputRefs, inputConstExpr);
  if (!inputRefs.empty() || inputConstExpr)
    return success();
  collectParameterReferences(getType(), outputRefs, outputConstExpr);
  if (!outputRefs.empty() || outputConstExpr)
    return success();

  if (getInput().getType() == getType())
    return success();

  return emitError("cannot rebind concrete input type ")
         << getInput().getType() << " to different concrete output type "
         << getType();
}

/// Fold away the rebind if the input and output types are the same.
OpFoldResult RebindOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  assert(operands.size() == 1);
  if (getInput().getType() == getType())
    return getInput();
  return {};
}

//===----------------------------------------------------------------------===//
// ExportOp
//===----------------------------------------------------------------------===//

static ParseResult parseExportOp(OpAsmParser &p, SymbolRefAttr &exported,
                                 StringAttr &alias) {
  if (p.parseOptionalKeyword("as")) {
    alias = StringAttr::get(
        p.getContext(),
        makeCWrapperName(exported.getLeafReference().getValue()));
    return success();
  }
  if (p.parseSymbolName(alias))
    return failure();
  return success();
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
  if (!isCIdentifier(getAlias()))
    return emitError("The alias name is not a valid C identifier, allowed "
                     "characters: [a-zA-Z0-9_]");
  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/KGENDialect/KGEN.cpp.inc"
