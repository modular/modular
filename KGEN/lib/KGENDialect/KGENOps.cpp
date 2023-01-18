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
#include "Support/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

// FIXME(5742): KGENDialect should not depend on POPDialect.
#include "Cache/CacheDialect/CacheOps.h"
#include "KGEN/POPDialect/POPTypes.h"

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
                                    op.getResultParamTypes(),
                                    op.getResultTypes());
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
  printColonTypeOrIndex(p.getStream(), value.getType());
  p << " = <";
  printParamValue(p, value);
  p << ">";
}

//===----------------------------------------------------------------------===//
// custom<ParamAssertOpValue>
//===----------------------------------------------------------------------===//

static ParseResult parseParamAssertOpValue(OpAsmParser &p, TypedAttr &value) {
  return parseParamValue(p, value, p.getBuilder().getI1Type());
}

static void printParamAssertOpValue(OpAsmPrinter &p, Operation *,
                                    Attribute value) {
  printParamValue(p, value);
}

//===----------------------------------------------------------------------===//
// ParamDeclareOp
//===----------------------------------------------------------------------===//

static ParseResult parseParamDeclareOpValue(OpAsmParser &p,
                                            ParamDeclArrayAttr &paramDecls,
                                            TypedAttr &value) {
  StringAttr name;
  if (parseParamName(p, name) || parseParamConstantOpValue(p, value))
    return failure();

  paramDecls = p.getBuilder().getAttr<ParamDeclArrayAttr>(
      ParamDeclAttr::get(name, value.getType()));
  return success();
}

static void printParamDeclareOpValue(OpAsmPrinter &p, Operation *,
                                     ParamDeclArrayAttr paramDecls,
                                     TypedAttr value) {
  ParamDeclAttr variable = paramDecls.front();
  printParamName(p, variable.getName().getValue());
  printParamConstantOpValue(p, nullptr, value);
}

void ParamDeclareOp::build(OpBuilder &builder, OperationState &result,
                           ParamDeclAttr decl, Attribute value) {
  build(builder, result, /*no result types*/ TypeRange{},
        builder.getAttr<ParamDeclArrayAttr>(decl), value);
}

ParamDeclAttr ParamDeclareOp::getParamDecl() {
  assert(getParamDecls().size() == 1 &&
         "ParamDeclareOp only allows a single parameter decl.");
  return *getParamDecls().begin();
}

void ParamDeclareOp::setParamDecl(ParamDeclAttr decl) {
  setParamDeclsAttr(ParamDeclArrayAttr::get(decl.getContext(), decl));
}

//===----------------------------------------------------------------------===//
// ParamDeclareRegionOp
//===----------------------------------------------------------------------===//

static ParseResult parseRegionDeclaration(OpAsmParser &p,
                                          ParamDeclArrayAttr &paramDecls,
                                          Region &body) {
  StringAttr paramName;
  if (parseParamName(p, paramName) || p.parseEqual())
    return failure();

  OperationState regionBody(p.getEncodedSourceLoc(p.getCurrentLocation()),
                            RegionBodyOp::getOperationName());
  std::optional<Location> bodyLoc = regionBody.location;
  if (RegionBodyOp::parse(p, regionBody) ||
      p.parseOptionalLocationSpecifier(bodyLoc))
    return failure();
  regionBody.location = *bodyLoc;
  auto bodyOp = cast<RegionBodyOp>(Operation::create(regionBody));
  body.push_back(new Block);
  body.front().push_back(bodyOp);

  paramDecls = ParamDeclArrayAttr::get(
      p.getContext(), ParamDeclAttr::get(paramName, bodyOp.getSignature()));
  return success();
}

static void printRegionDeclaration(OpAsmPrinter &p, Operation *op,
                                   ParamDeclArrayAttr paramDecls,
                                   Region &region) {
  printParamName(p, paramDecls.front().getName());
  p << " =";
  auto body = cast<RegionBodyOp>(region.front().front());
  body.print(p);
  p.printOptionalLocationSpecifier(body.getLoc());
}

//===----------------------------------------------------------------------===//
// ParamSearchOp
//===----------------------------------------------------------------------===//

static ParseResult parseParamSearchOpValue(OpAsmParser &p,
                                           ParamDeclArrayAttr &paramDecls,
                                           ParameterExprArrayAttr &values) {
  std::string varname;
  Type valTy;
  SmallVector<TypedAttr> valuesElts;

  if (p.parseKeywordOrString(&varname) || parseColonTypeOrIndex(p, valTy) ||
      p.parseEqual() ||
      p.parseCommaSeparatedList(OpAsmParser::Delimiter::LessGreater,
                                [&]() -> ParseResult {
                                  TypedAttr elt;
                                  if (parseParamValue(p, elt, valTy))
                                    return failure();
                                  valuesElts.push_back(elt);
                                  return success();
                                }))
    return failure();

  paramDecls = p.getBuilder().getAttr<ParamDeclArrayAttr>(
      ParamDeclAttr::get(varname, valTy));
  values = p.getBuilder().getAttr<ParameterExprArrayAttr>(valuesElts);
  return success();
}

static void printParamSearchOpValue(OpAsmPrinter &p, Operation *,
                                    ParamDeclArrayAttr paramDecls,
                                    ParameterExprArrayAttr values) {
  ParamDeclAttr variable = paramDecls.front();
  printParamName(p, variable.getName().getValue());

  printColonTypeOrIndex(p.getStream(), variable.getType());
  p << " = <";
  llvm::interleaveComma(values, p,
                        [&](TypedAttr elt) { printParamValue(p, elt); });
  p << ">";
}

ParamDeclAttr ParamSearchOp::getParamDecl() {
  assert(getParamDecls().size() == 1 &&
         "ParamSearchOp only allows a single parameter decl.");
  return *getParamDecls().begin();
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
    collectParameterReferences(cond, parameterRefs);
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
      constraints.push_back(
          ConstraintAttr::get(cond, op.getMessageAttr(), op.getLoc()));
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
                                     ParamDeclArrayAttr &paramDecls) {

  if (p.parseOptionalLess()) {
    // If there is no <, then the params of the call op are empty, so set
    // paramValues and paramDecls to empty and return.
    paramValues = ParamBindArrayAttr::get(p.getContext(), {});
    paramDecls = ParamDeclArrayAttr::get(p.getContext(), {});
    return success();
  }

  // Parse the input list
  if (parseParamBinds(p, paramValues))
    return failure();

  // Check to see if we have results and parse them if so.
  if (succeeded(p.parseOptionalArrow())) {
    if (parseParamDecls(p, paramDecls))
      return failure();
  } else {
    // paramDecls is empty if there is no arrow.
    paramDecls = ParamDeclArrayAttr::get(p.getContext(), {});
  }

  return p.parseGreater();
}

static void printCallOpParams(OpAsmPrinter &p, Operation *op,
                              ParamBindArrayAttr paramValues,
                              ParamDeclArrayAttr paramDecls) {
  if (paramValues.empty() && paramDecls.empty())
    return;
  p << "<";
  printParamBinds(p, paramValues);
  if (!paramDecls.empty()) {
    p << " -> ";
    printParamDecls(paramDecls, p.getStream());
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
  // See if the parameter definitions and uses within the generator are
  // structured correctly.
  if (failed(ParameterDeclsAndUses().calculateAndVerify(*this, symbolTable)))
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
  // See if the parameter definitions and uses within the func are
  // structured correctly.
  ParameterDeclsAndUses paramInfo;
  if (failed(paramInfo.calculateAndVerify(*this, symbolTable)))
    return failure();

  // In a kgen.func, parameters are allowed to be defined (e.g. by calls with
  // output parameters), but not used.  This is because the elaborator must
  // already have been run, lowering these to concrete attibute values.

  // TODO: In the future, we could ban specific uses (e.g. in types) but have an
  // allow-list of operations that can use parameters.  This could be useful if
  // we want something to be able to use the result parameters of a call or
  // something.  Until then, a blanket ban on parameter use is sufficient.
  for (auto &[usingOp, uses] : paramInfo.usersAndDeclarers) {
    if (!uses.empty()) {
      auto diag = usingOp->emitError("invalid use of parameter ")
                  << uses[0].getName() << " in kgen.func";
      diag.attachNote(this->getLoc())
          << "within kgen.func '" << getName() << "'";

      return failure();
    }
  }
  return success();
}

LogicalResult FuncOp::verify() {
  // kgen.func's are not allowed to have input parameter lists.
  if (!getInputParamDecls().empty() || !getResultParamTypes().empty())
    return emitOpError("cannot have input or result parameters");
  if (!llvm::all_of(getConventions().getInputConventions(),
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
    printKGENType(p.getStream(), evaluator.getType());
    p << " = ";
    printParamValue(evaluator, p.getStream());
  }

  if (SymbolConstantAttr defaultImpl = getDefaultImplAttr()) {
    p << " defaultImpl ";
    printKGENType(p.getStream(), defaultImpl.getType());
    p << " = ";
    printParamValue(defaultImpl, p.getStream());
  }
}

LogicalResult
GeneratorInterfaceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // See if the parameter definitions and uses within the generator are
  // structured correctly.  These are only defined in the interface and used
  // in the argument list or constraints list.
  if (failed(ParameterDeclsAndUses().calculateAndVerify(*this, symbolTable)))
    return failure();

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

  auto index = IndexType::get(getContext());
  auto evaluatorType = FunctionType::get(
      getContext(), {POP::PointerType::get(getFunctionType()), index}, index);
  auto expectedSignature = SignatureType::get(evaluatorType);

  // Get the specialized callee signature.
  SignatureType funcSignature = func.getSignature().getSpecializedSignature(
      evaluator.getParamValues(), [&] { return emitError(); });
  if (!funcSignature)
    return failure();

  if (failed(verifyDeclSignaturesMatch("interface evaluator", expectedSignature,
                                       getLoc(), "referenced evaluator",
                                       funcSignature, func.getLoc())))
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
  ConventionsAttr conventions;
  SmallVector<Type> inputs, outputs;
  if (p.parseAttribute(callee) ||
      parseCallOpParams(p, paramValues, paramDecls) || p.parseColon() ||
      parseTypesWithConventions(p, inputs, outputs, conventions))
    return failure();
  FunctionType result = p.getBuilder().getFunctionType(inputs, outputs);
  calleeCst = SymbolConstantAttr::get(
      callee, paramValues,
      SignatureType::get(ParamBindArrayAttr::get(p.getContext(), {}),
                         paramDecls, result, conventions));
  resultType = result;
  return success();
}

static void printAddressOfOp(OpAsmPrinter &p, Operation *op,
                             SymbolConstantAttr calleeCst,
                             ParamDeclArrayAttr paramDecls, Type resultType) {
  p << calleeCst.getSymbol();
  printCallOpParams(p, op, calleeCst.getParamValues(), paramDecls);
  p << " : ";
  printTypesWithConventions(p.getStream(), calleeCst.getType().getValueInputs(),
                            calleeCst.getType().getValueResults(),
                            calleeCst.getType().getConventions());
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
  ConventionsAttr conventions;
  if (p.parseAttribute(callee) ||
      parseCallOpParams(p, paramValues, paramDecls) ||
      p.parseOperandList(operands, AsmParser::Delimiter::Paren) ||
      p.parseColon() ||
      parseTypesWithConventions(p, operandTypes, resultTypes, conventions))
    return failure();
  calleeCst = SymbolConstantAttr::get(
      callee, paramValues,
      SignatureType::get(
          ParamBindArrayAttr::get(p.getContext(), {}), paramDecls,
          p.getBuilder().getFunctionType(operandTypes, resultTypes),
          conventions));
  return success();
}

static void printCallOp(OpAsmPrinter &p, Operation *op,
                        SymbolConstantAttr calleeCst,
                        ParamDeclArrayAttr paramDecls, ValueRange operands,
                        TypeRange operandTypes, TypeRange resultTypes) {
  p << calleeCst.getSymbol();
  printCallOpParams(p, op, calleeCst.getParamValues(), paramDecls);
  p << '(';
  p.printOperands(operands);
  p << ") : ";
  printTypesWithConventions(p.getStream(), operandTypes, resultTypes,
                            calleeCst.getType().getConventions());
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
// RegionBodyOp / custom<RegionBody>
//===----------------------------------------------------------------------===//

/// Parse a single-block isolated from above region.
static ParseResult parseRegionBody(OpAsmParser &p, TypeAttr &signature,
                                   ConstraintArrayAttr &constraints,
                                   Region &body) {
  SmallVector<OpAsmParser::Argument> args;
  ParamDeclArrayAttr inputParamDecls;
  TypeArrayAttr resultParamTypes;
  ConventionsAttr conventions;
  SmallVector<Type> resultTypes;
  llvm::SMLoc bodyLoc;
  if (parseOptionalParameterSpec(p, inputParamDecls, resultParamTypes) ||
      parseFunctionSignature(p, args, resultTypes, conventions) ||
      parseOptionalConstraints(p, constraints) ||
      p.getCurrentLocation(&bodyLoc) || p.parseRegion(body, args))
    return failure();

  // Form the Signature.
  SmallVector<Type> argTypes;
  for (const OpAsmParser::Argument &arg : args)
    argTypes.push_back(arg.type);
  signature = TypeAttr::get(SignatureType::get(
      inputParamDecls, resultParamTypes,
      p.getBuilder().getFunctionType(argTypes, resultTypes), conventions));
  return success();
}

/// Print a single-block isolated from above region.
static void printRegionBody(OpAsmPrinter &p, Operation *op, TypeAttr signature,
                            ConstraintArrayAttr constraints, Region &body) {
  auto sig = cast<SignatureType>(signature.getValue());

  printOptionalParameterSpec(sig.getInputParams(), sig.getResultParamTypes(),
                             p.getStream());
  printFunctionSignature(p, body, sig.getValueInputs(), sig.getValueResults(),
                         sig.getConventions());
  printOptionalConstraints(p, op, constraints);
  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

LogicalResult RegionBodyOp::verifyRegions() {
  auto returnOp = cast<ReturnOp>(getBody()->getTerminator());
  if (failed(checkResultArgumentTypes(returnOp, returnOp.getParameters(),
                                      getResultParamTypes(),
                                      getSignature().getValueResults())))
    return failure();
  if (getBody()->getArgumentTypes() != getSignature().getValueInputs())
    return emitOpError("signature mismatches body");
  return success();
}

//===----------------------------------------------------------------------===//
// ParamConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ParamConstantOp::fold(ArrayRef<Attribute> constants) {
  assert(constants.empty() && "kgen.param.constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// RebindOp
//===----------------------------------------------------------------------===//

/// If either the input or output type are parameterized, return success.
/// Otherwise, require that the concrete input and output types are the same.
LogicalResult RebindOp::verify() {
  SmallVector<ParamDeclRefAttr> inputRefs, outputRefs;
  collectParameterReferences(getInput().getType(), inputRefs);
  if (!inputRefs.empty())
    return success();
  collectParameterReferences(getType(), outputRefs);
  if (!outputRefs.empty())
    return success();

  if (getInput().getType() == getType())
    return success();

  return emitError("cannot rebind concrete input type ")
         << getInput().getType() << " to different concrete output type "
         << getType();
}

/// Fold away the rebind if the input and output types are the same.
OpFoldResult RebindOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 1);
  if (getInput().getType() == getType())
    return getInput();
  return {};
}

//===----------------------------------------------------------------------===//
// StructDeclOp
//===----------------------------------------------------------------------===//

/// Verify that the body has no arguments.
LogicalResult StructDeclOp::verify() {
  if (getFields().getNumArguments())
    return emitOpError("expected declaration body to have no arguments");
  return success();
}

/// Verify that there are no duplicate field names.
LogicalResult StructDeclOp::verifyRegions() {
  SmallDenseMap<StringAttr, StructFieldOp, 8> seenFields;
  for (Operation &op : getFields().front()) {
    auto field = dyn_cast<StructFieldOp>(&op);
    if (!field)
      continue;
    auto [it, inserted] = seenFields.try_emplace(field.getNameAttr(), field);
    if (!inserted) {
      return (field.emitError("duplicate struct field ") << field.getNameAttr())
                 .attachNote(it->second.getLoc())
             << "see previous declaration here";
    }
  }
  return success();
}

/// Verify parameter uses.
LogicalResult
StructDeclOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyIfTopLevel(symbolTable);
}

void StructDeclOp::build(OpBuilder &builder, OperationState &result,
                         StringAttr name) {
  auto context = builder.getContext();
  build(builder, result, name, ParamDeclArrayAttr::get(context, {}));
  result.regions[0]->push_back(new Block());
}

//===----------------------------------------------------------------------===//
// StructFieldOp
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

//===----------------------------------------------------------------------===//
// StructCreateOp
//===----------------------------------------------------------------------===//

static ParameterEvaluator getEvaluatorForBoundStructType(DeclRefType refType) {
  ParameterEvaluator evaluator;
  for (ParamBindAttr bind : refType.getParamValues())
    evaluator.setParameterValue(bind.getDecl(), bind.getValue());
  return evaluator;
}

/// Lookup the declaration for the struct. When checking field types, we can't
/// directly compare operation types to the struct field types because they are
/// parameterized under different domains. We have to rebind them.
static StructDeclOp lookupStructDecl(SymbolTableCollection &symbolTable,
                                     Operation *user, DeclRefType ref) {
  auto module = KGENModule::from(user, symbolTable);
  auto structDecl = module.lookup<StructDeclOp>(ref.getSymbol());
  // Currently, this is impossible to fail because the symbol use was verified
  // by the parameter verifier.
  assert(structDecl && "expected a struct declaration");
  return structDecl;
}

/// Verify the reference struct type.
LogicalResult
StructCreateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verify the types of the fields in the operands match those in the
  // struct declaration.
  ParameterEvaluator evaluator = getEvaluatorForBoundStructType(getType());
  StructDeclOp structDecl = lookupStructDecl(symbolTable, *this, getType());
  auto fields = structDecl.getFieldDecls();
  unsigned numFields = std::distance(fields.begin(), fields.end());
  if (numFields != getNumOperands())
    return emitOpError("expected ")
           << numFields << " operands but got " << getNumOperands();
  for (auto [fieldDecl, operand, i] :
       llvm::zip(fields, getOperands(), llvm::seq<unsigned>(0, numFields))) {
    Type reboundType = evaluator.getReboundType(fieldDecl.getType());
    if (reboundType != operand.getType()) {
      return emitOpError("operand #")
             << i << " has type " << operand.getType()
             << " but corresponding struct field " << fieldDecl.getNameAttr()
             << " expected " << fieldDecl.getType();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// StructInsertOp
//===----------------------------------------------------------------------===//

LogicalResult
StructInsertOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  ParameterEvaluator evaluator = getEvaluatorForBoundStructType(getType());
  StructDeclOp structDecl = lookupStructDecl(symbolTable, *this, getType());

  for (StructFieldOp fieldDecl : structDecl.getFieldDecls()) {
    if (fieldDecl.getName() != getFieldAttr())
      continue;
    Type reboundType = evaluator.getReboundType(fieldDecl.getType());
    if (reboundType != getValue().getType())
      return emitOpError("cannot insert value of type ")
             << getValue().getType() << " into struct field " << getFieldAttr()
             << " which expected " << reboundType;
    return success();
  }

  return emitOpError("struct ")
         << getType().getSymbol() << " has no field named " << getFieldAttr();
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

static LogicalResult
verifyStructFieldAndType(SymbolTableCollection &symbolTable, Operation *op,
                         DeclRefType ref, StringAttr fieldName, Type type) {
  ParameterEvaluator evaluator = getEvaluatorForBoundStructType(ref);
  StructDeclOp structDecl = lookupStructDecl(symbolTable, op, ref);

  for (StructFieldOp fieldDecl : structDecl.getFieldDecls()) {
    if (fieldDecl.getName() != fieldName)
      continue;
    Type reboundType = evaluator.getReboundType(fieldDecl.getType());
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
StructExtractOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyStructFieldAndType(symbolTable, *this, getContainer().getType(),
                                  getFieldAttr(), getValue().getType());
}

void StructExtractOp::build(OpBuilder &builder, OperationState &result,
                            Value structBase, StructFieldOp field) {
  auto structType = cast<DeclRefType>(structBase.getType());
  ParameterEvaluator evaluator = getEvaluatorForBoundStructType(structType);
  build(builder, result, evaluator.getReboundType(field.getType()),
        field.getNameAttr(), structBase);
}

//===----------------------------------------------------------------------===//
// StructGEPOp
//===----------------------------------------------------------------------===//

LogicalResult
StructGEPOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  TypedAttr refExpr = getContainer().getType().getElementType();
  return verifyStructFieldAndType(
      symbolTable, *this,
      cast<DeclRefType>(cast<TypeConstantAttr>(refExpr).getValue()),
      getFieldAttr(),
      ParamRefType::get(getResult().getType().getElementType()));
}

void StructGEPOp::build(OpBuilder &builder, OperationState &result,
                        Value structBasePtr, StructFieldOp field) {
  TypedAttr refExpr =
      cast<POP::PointerType>(structBasePtr.getType()).getElementType();
  auto structType =
      cast<DeclRefType>(cast<TypeConstantAttr>(refExpr).getValue());

  ParameterEvaluator evaluator = getEvaluatorForBoundStructType(structType);
  build(builder, result,
        POP::PointerType::get(evaluator.getReboundType(field.getType())),
        field.getNameAttr(), structBasePtr);
}

//===----------------------------------------------------------------------===//
// ExportOp
//===----------------------------------------------------------------------===//

LogicalResult ExportOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (getExports().empty())
    return emitOpError("exports must not be empty");

  // Just ensure we're exporting symbols we can see.
  auto module = KGENModule::from(*this, symbolTable);
  for (auto e : getExports().getAsRange<SymbolRefAttr>()) {
    if (!module.lookup<FuncInterface>(e))
      return emitOpError("could not find referenced symbol '") << e << "'";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/KGENDialect/KGEN.cpp.inc"
