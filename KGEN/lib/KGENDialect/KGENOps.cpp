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

  printColonTypeOrIndex(p, variable.getType());
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
  if (!getInputParamDecls().empty() || !getResultParams().empty())
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
  ConventionsAttr conventions;
  SmallVector<Type> inputs, outputs;
  if (p.parseAttribute(callee) ||
      parseCallOpParams(p, paramValues, paramDecls, resultParams) ||
      p.parseColon() ||
      parseTypesWithConventions(p, inputs, outputs, conventions))
    return failure();
  FunctionType result = p.getBuilder().getFunctionType(inputs, outputs);
  calleeCst = SymbolConstantAttr::get(
      callee, paramValues,
      SignatureType::get(ParamDeclArrayAttr::get(p.getContext(), {}),
                         resultParams, result, conventions));
  resultType = result;
  return success();
}

static void printAddressOfOp(OpAsmPrinter &p, Operation *op,
                             SymbolConstantAttr calleeCst,
                             ParamDeclArrayAttr paramDecls, Type resultType) {
  p << calleeCst.getSymbol();
  printCallOpParams(p, op, calleeCst.getParamValues(), paramDecls,
                    calleeCst.getType().getResultParams());
  p << " : ";
  printTypesWithConventions(p, calleeCst.getType().getValueInputs(),
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
  ParamDeclArrayAttr resultParams;
  ConventionsAttr conventions;
  if (p.parseAttribute(callee) ||
      parseCallOpParams(p, paramValues, paramDecls, resultParams) ||
      p.parseOperandList(operands, AsmParser::Delimiter::Paren) ||
      p.parseColon() ||
      parseTypesWithConventions(p, operandTypes, resultTypes, conventions))
    return failure();
  calleeCst = SymbolConstantAttr::get(
      callee, paramValues,
      SignatureType::get(
          ParamDeclArrayAttr::get(p.getContext(), {}), resultParams,
          p.getBuilder().getFunctionType(operandTypes, resultTypes),
          conventions));
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
  printTypesWithConventions(p, operandTypes, resultTypes,
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
  ParamDeclArrayAttr inputParamDecls, resultParamDecls;
  ConventionsAttr conventions;
  SmallVector<Type> resultTypes;
  llvm::SMLoc bodyLoc;
  if (parseOptionalParameterSpec(p, inputParamDecls, resultParamDecls) ||
      parseFunctionSignature(p, args, resultTypes, conventions) ||
      parseOptionalConstraints(p, constraints) ||
      p.getCurrentLocation(&bodyLoc) || p.parseRegion(body, args))
    return failure();

  // Form the Signature.
  SmallVector<Type> argTypes;
  for (const OpAsmParser::Argument &arg : args)
    argTypes.push_back(arg.type);
  signature = TypeAttr::get(SignatureType::get(
      inputParamDecls, resultParamDecls,
      p.getBuilder().getFunctionType(argTypes, resultTypes), conventions));
  return success();
}

/// Print a single-block isolated from above region.
static void printRegionBody(OpAsmPrinter &p, Operation *op, TypeAttr signature,
                            ConstraintArrayAttr constraints, Region &body) {
  auto sig = cast<SignatureType>(signature.getValue());

  printOptionalParameterSpec(p, sig.getInputParams(), sig.getResultParams());
  printFunctionSignature(p, body, sig.getValueInputs(), sig.getValueResults(),
                         sig.getConventions());
  printOptionalConstraints(p, op, constraints);
  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

LogicalResult RegionBodyOp::verifyRegions() {
  auto returnOp = cast<ReturnOp>(getBody()->getTerminator());
  if (failed(checkResultArgumentTypes(returnOp, returnOp.getParameters(),
                                      getResultParams(),
                                      getSignature().getValueResults())))
    return failure();
  if (getBody()->getArgumentTypes() != getSignature().getValueInputs())
    return emitOpError("signature mismatches body");
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

LogicalResult ExportOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (getExports().empty())
    return emitOpError("exports must not be empty");

  // Just ensure we're exporting symbols we can see.
  auto module = KGENModule::from(*this, symbolTable);
  for (auto e : getExports().getAsRange<SymbolRefAttr>()) {
    auto func = module.lookup<FuncInterface>(e);
    if (!func)
      return emitOpError("could not find referenced symbol '") << e << "'";
    if (func.isForceInline()) {
      return func.emitError("function marked 'force_inline' cannot be exported")
                 .attachNote(getLoc())
             << "function exported here";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/KGENDialect/KGEN.cpp.inc"
