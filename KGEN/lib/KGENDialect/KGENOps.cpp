//===- KGENOps.cpp --------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the KGEN dialect operations.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/ElaboratorOpInterface.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGENVerifyHelper.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"

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
                                      Attribute value, Type type) {
  printColonTypeOrIndex(p.getStream(), type);
  p << " = <";
  printParamValue(p, value);
  p << ">";
}

//===----------------------------------------------------------------------===//
// custom<ParamDeclareOpValue>
//===----------------------------------------------------------------------===//

static ParseResult parseParamDeclareOpValue(OpAsmParser &p,
                                            ParamDeclArrayAttr &paramDecls,
                                            TypedAttr &value) {
  std::string varname;
  Type valTy;
  if (p.parseKeywordOrString(&varname) ||
      parseParamConstantOpValue(p, value, valTy))
    return failure();

  paramDecls = p.getBuilder().getAttr<ParamDeclArrayAttr>(
      ParamDeclAttr::get(varname, valTy));
  return success();
}

static void printParamDeclareOpValue(OpAsmPrinter &p, Operation *,
                                     ParamDeclArrayAttr paramDecls,
                                     TypedAttr value) {
  ParamDeclAttr variable = paramDecls.front();
  printParamName(p, variable.getName().getValue());
  printParamConstantOpValue(p, nullptr, value, value.getType());
}

//===----------------------------------------------------------------------===//
// custom<ParameterValues>
//===----------------------------------------------------------------------===//

static ParseResult parseParameterValues(OpAsmParser &p, ArrayAttr &value) {
  SmallVector<Attribute> elts;
  if (p.parseCommaSeparatedList(
          OpAsmParser::Delimiter::OptionalLessGreater, [&]() -> ParseResult {
            TypedAttr value;
            if (parseParamValueDefaultingToIndex(p, value))
              return failure();
            elts.push_back(value);
            return success();
          }))
    return failure();

  value = ArrayAttr::get(p.getContext(), elts);
  return success();
}

static void printParameterValues(OpAsmPrinter &p, Operation *op,
                                 ArrayAttr value) {
  if (value.empty())
    return;
  p << '<';
  llvm::interleaveComma(value, p, [&](Attribute value) {
    auto valType = value.cast<TypedAttr>().getType();
    if (!valType.isIndex()) {
      p << ":";
      printKGENType(p.getStream(), valType);
      p << " ";
    }
    printParamValue(p, value);
  });
  p << '>';
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

void ParamDeclareOp::build(OpBuilder &builder, OperationState &result,
                           ParamDeclAttr decl, Attribute value) {
  build(builder, result, /*no result types*/ TypeRange{},
        builder.getAttr<ParamDeclArrayAttr>(decl), value);
}

ParamDeclAttr ParamDeclareOp::getParamDecl() {
  assert(getParamDecls().size() == 1 &&
         "ParamDeclareOp only allows a single parameter decl.");
  return (*getParamDecls().begin()).cast<ParamDeclAttr>();
}

//===----------------------------------------------------------------------===//
// ParamAssertOp
//===----------------------------------------------------------------------===//

LogicalResult ParamAssertOp::canonicalize(ParamAssertOp op,
                                          PatternRewriter &rewriter) {
  // If the condition is statically true then we can just remove this op.
  auto cond = op.getCond();
  if (auto intCond = cond.dyn_cast<IntegerAttr>()) {
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
  KGENDeclInterface parent = op->getParentOfType<KGENDeclInterface>();
  if (parent && succeeded(collectParameterReferences(cond, parameterRefs))) {
    ArrayRef<ParamDeclAttr> generatorInputParams = getParamDecls(parent);

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
      parent.setConstraintsAttr(
          rewriter.getAttr<ConstraintArrayAttr>(constraints));
      op.erase();
      return success();
    }
  }

  return failure();
}

/// Parse a parameter binding list if present.
///
///   parameter-bind   ::= identifier (`:` type)? `=` attribute-value
///   parameter-bind-list ::= parameter-bind (`,` parameter-bind)* | `(` `)`
static ParseResult parseParamBinds(AsmParser &p,
                                   SmallVectorImpl<ParamBindAttr> &paramBinds) {
  // Check to see if we have the () syntax instead of arguments.
  if (succeeded(p.parseOptionalLParen())) {
    if (p.parseRParen())
      return failure();
    return success();
  }

  // Handle the parameter-decl/parameter-result productions.
  auto parseParamBind = [&]() -> ParseResult {
    StringAttr name;
    Type type;
    TypedAttr value;

    if (parseParamName(p, name) || parseColonTypeOrIndex(p, type) ||
        p.parseEqual() || parseParamValue(p, value, type))
      return failure();
    paramBinds.push_back(ParamBindAttr::get(name, value));
    return success();
  };

  if (p.parseCommaSeparatedList(OpAsmParser::Delimiter::None, parseParamBind))
    return failure();

  return success();
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

  SmallVector<ParamBindAttr> vals;
  // Parse the input list
  if (parseParamBinds(p, vals))
    return failure();
  paramValues = ParamBindArrayAttr::get(p.getContext(), vals);

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
  if (paramValues.empty())
    p << "()";
  else {
    llvm::interleaveComma(paramValues, p, [&](ParamBindAttr bind) {
      printParamName(p, bind.getName().getValue());
      printColonTypeOrIndex(p.getStream(), bind.getType());
      p << " = ";
      printParamValue(p, bind.getValue());
    });
  }

  if (!paramDecls.empty()) {
    p << " -> ";
    printParamDecls(p.getStream(), paramDecls);
  }
  p << ">";
}

//===----------------------------------------------------------------------===//
// CallParamOp / custom<CallParamCallee>
//===----------------------------------------------------------------------===//

static ParseResult parseCallParamCallee(OpAsmParser &p, TypedAttr &value,
                                        ParamBindArrayAttr &paramValues,
                                        ParamDeclArrayAttr &paramResultDecls,
                                        SmallVectorImpl<Type> &operandTypes,
                                        SmallVectorImpl<Type> &resultTypes) {
  Type type;
  auto loc = p.getCurrentLocation();
  if (p.parseLSquare() || parseKGENType(p, type) || p.parseColon() ||
      parseParamValue(p, value, type) || p.parseRSquare() ||
      parseCallOpParams(p, paramValues, paramResultDecls))
    return failure();

  auto signature = value.getType().dyn_cast<SignatureType>();
  if (!signature)
    return p.emitError(loc, "callee parameter type must be a signature type");

  auto nameGetter = [&](auto attr) -> Attribute { return attr.getName(); };
  auto typeGetter = [&](auto attr) -> Type { return attr.getType(); };

  // Check that the parameter names/types specified match
  // up with the expected ones.
  auto callLoc = p.getEncodedSourceLoc(loc);
  if (verifyMatchingLists(
          llvm::map_range(paramValues, nameGetter),
          llvm::map_range(signature.getInputParams(), nameGetter), "caller",
          callLoc, "callee parameter", callLoc, "input parameter", "name") ||
      verifyMatchingLists(
          llvm::map_range(paramValues, typeGetter),
          llvm::map_range(signature.getInputParams(), typeGetter), "caller",
          callLoc, "callee parameter", callLoc, "input parameter", "type"))
    return failure();

  // We need to substitute and simplify expressions that occur in the argument
  // list and parameter types, e.g.:
  //     kgen.generator @callee1<type: dtype>(%x: !meta.scalar<type>)
  // ... call ((@callee1))<type: dtype = f32>(%arg1) : (!meta.scalar<f32>) -> ()

  ParameterEvaluator evaluator;
  for (auto [bind, decl] : llvm::zip(paramValues, signature.getInputParams())) {
    evaluator.setParameterValue(decl, bind.getValue());
  }
  auto remapType = [&](Type type) -> Type {
    return evaluator.getReboundType(type);
  };

  for (Type inputType : signature.getValues().getInputs())
    operandTypes.push_back(remapType(inputType));
  for (Type resultType : signature.getValues().getResults())
    resultTypes.push_back(remapType(resultType));

  return success();
}

static void printCallParamCallee(OpAsmPrinter &p, Operation *op,
                                 TypedAttr value,
                                 ParamBindArrayAttr paramValues,
                                 ParamDeclArrayAttr paramDecls,
                                 OperandRange::type_range operandTypes,
                                 mlir::ResultRange::type_range resultTypes) {
  p << "[";
  printKGENType(p.getStream(), value.getType());
  p << ": ";
  printParamValue(p, value);
  p << "]";
  printCallOpParams(p, op, paramValues, paramDecls);
}

LogicalResult CallParamOp::canonicalize(CallParamOp op,
                                        PatternRewriter &rewriter) {
  // If the condition is a known symbol, then replace this with a kgen.call.
  if (auto calleeSymbol = op.getCallee().dyn_cast<SymbolConstantAttr>()) {
    rewriter.replaceOpWithNewOp<CallOp>(
        op, op.getResultTypes(), calleeSymbol.getSymbol().getLeafReference(),
        op.getParamValues(), op.getParamDecls(), op.getOperands());
    return success();
  }

  return failure();
}

LogicalResult CallParamOp::verify() {
  KGENDeclInterface parent =
      getOperation()->getParentOfType<KGENDeclInterface>();
  if (!parent || isa<FuncOp>(parent))
    return emitError(
        "kgen.call_param is only allowed in generators pre-elaboration");

  return success();
}

//===----------------------------------------------------------------------===//
// GeneratorOp
//===----------------------------------------------------------------------===//

ReturnOp GeneratorOp::getReturnOp() {
  return cast<ReturnOp>(getBodyBlock()->getTerminator());
}

/// Parses a KGEN Generator.
ParseResult GeneratorOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseGeneratorOrFunc(parser, result, GeneratorOrFuncKind::generator);
}

// Print the GeneratorOp using the shared printing logic.
void GeneratorOp::print(OpAsmPrinter &p) { printGeneratorOrFunc(p, *this); }

LogicalResult
GeneratorOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (failed(getReturnOp().checkArgumentTypes(getResultParamTypes(),
                                              getResultTypes())))
    return failure();

  // See if the parameter definitions and uses within the generator are
  // structured correctly.
  if (failed(ParameterDeclsAndUses::calculateAndVerify(*this, symbolTable)))
    return failure();

  // If the generator is implementing a generator interface, check that they
  // line up correctly.
  FlatSymbolRefAttr interfaceSym = getImplementsAttr();
  if (!interfaceSym)
    return success();

  // Check that the callee attribute was specified.
  GeneratorInterfaceOp interface = dyn_cast_or_null<GeneratorInterfaceOp>(
      symbolTable.lookupNearestSymbolFrom(*this, interfaceSym));
  if (!interface)
    return emitError() << "'" << interfaceSym
                       << "' does not reference a generator interface";

  // Verify that the signature of this generator matches the signature of the
  // interface.
  return verifyDeclMatchesInterface("generator", *this, "interface", interface);
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

/// Create a func with no body block.  The caller must create it and fill
/// it in.
void FuncOp::build(OpBuilder &builder, OperationState &result, StringAttr name,
                   StringAttr visibility, FunctionType signature,
                   ArrayRef<Type> outputParamTypes) {
  // Add an attribute for the name and function_type attributes.
  result.addAttribute(SymbolTable::getSymbolAttrName(), name);
  result.addAttribute(SymbolTable::getVisibilityAttrName(), visibility);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(signature));
  result.addAttribute("paramDecls", builder.getAttr<ParamDeclArrayAttr>(
                                        ArrayRef<ParamDeclAttr>()));
  result.addAttribute("resultParamTypes",
                      builder.getAttr<TypeArrayAttr>(outputParamTypes));
  result.addRegion();
}

/// Create a func with an empty body, `argLocs` specifies the locations for
/// all the block arguments.
void FuncOp::build(OpBuilder &builder, OperationState &result, StringAttr name,
                   StringAttr visibility, FunctionType signature,
                   ArrayRef<Type> outputParamTypes,
                   ArrayRef<Location> argLocs) {
  build(builder, result, name, visibility, signature, outputParamTypes);

  // Create a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);

  // Add arguments to the body block.
  assert(signature.getInputs().size() == argLocs.size() &&
         "incorrect number of arg locs");
  body->addArguments(signature.getInputs(), argLocs);
}

ReturnOp FuncOp::getReturnOp() {
  return cast<ReturnOp>(getBodyBlock()->getTerminator());
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
  if (failed(getReturnOp().checkArgumentTypes(getResultParamTypes(),
                                              getResultTypes())))
    return failure();

  // kgen.func's are not allowed to have input parameter lists.
  if (!getParamDecls().empty())
    return emitError(
        "kgen.func only allows output parameters, not input parameters");

  // See if the parameter definitions and uses within the func are
  // structured correctly.
  FailureOr<ParameterDeclsAndUses> paramInfo =
      ParameterDeclsAndUses::calculateAndVerify(*this, symbolTable);
  if (failed(paramInfo))
    return failure();

  // In a kgen.func, parameters are allowed to be defined (e.g. by calls with
  // output parameters), but not used.  This is because the elaborator must
  // already have been run, lowering these to concrete attibute values.

  // TODO: In the future, we could ban specific uses (e.g. in types) but have an
  // allow-list of operations that can use parameters.  This could be useful if
  // we want something to be able to use the result parameters of a call or
  // something.  Until then, a blanket ban on parameter use is sufficient.
  for (auto &[usingOp, uses] : paramInfo.value().usersAndDeclarers) {
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

// FuncOp doesn't allow constraints, because it doesn't have any input
// parameters.
ArrayRef<ConstraintAttr> FuncOp::getConstraints() { return {}; }
ConstraintArrayAttr FuncOp::getConstraintsAttr() {
  return ConstraintArrayAttr();
}

void FuncOp::setConstraintsAttr(ConstraintArrayAttr attr) {
  assert(0 && "kgen.func doesn't have input parameters to add constraints to");
}

//===----------------------------------------------------------------------===//
// GeneratorInterfaceOp
//===----------------------------------------------------------------------===//

/// Parses a KGEN generator interface.
ParseResult GeneratorInterfaceOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  return parseGeneratorOrFunc(parser, result, GeneratorOrFuncKind::interface);
}

// Print the GeneratorInterfaceOp using the shared printing logic.
void GeneratorInterfaceOp::print(OpAsmPrinter &p) {
  printGeneratorOrFunc(p, *this);
}

LogicalResult
GeneratorInterfaceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // See if the parameter definitions and uses within the generator are
  // structured correctly.  These are only defined in the interface and used
  // in the argument list or constraints list.
  return ParameterDeclsAndUses::calculateAndVerify(*this, symbolTable);
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

template <typename CallerRange, typename CalleeRange>
static ParseResult verifyMatchingCallLists(const CallerRange &callerRange,
                                           const CalleeRange &calleeRange,
                                           Operation *caller, Operation *callee,
                                           const char *itemName,
                                           const char *propertyName) {
  return verifyMatchingLists(callerRange, calleeRange, "caller",
                             caller->getLoc(), "callee", callee->getLoc(),
                             itemName, propertyName);
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto calleeAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!calleeAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  auto callee = dyn_cast_or_null<KGENDeclInterface>(
      symbolTable.lookupNearestSymbolFrom(*this, calleeAttr));
  if (!callee)
    return emitError() << "'" << calleeAttr.getValue()
                       << "' does not reference a valid callee";

  // Verify that the callee/caller parameters match.  The parameter names on the
  // results don't need to match, but the parameter names on the argument
  // bindings do.  The types always need to match.
  auto calleeInputParamDecls = callee.getInputParamDecls();
  auto calleeOutputParamTypes = callee.getResultParamTypes();

  // Check the parameter values specified to the input parameters.
  ArrayRef<ParamBindAttr> callerInputParams = getParamValues();
  ArrayRef<ParamDeclAttr> callerOutputParamDecls = getParamDecls();

  auto getParamDeclType = [](ArrayRef<ParamDeclAttr> decls) {
    return llvm::map_range(
        decls, [](ParamDeclAttr value) -> Type { return value.getType(); });
  };

  /// Check the input parameter names.  We don't check the result parameter
  /// names because (in general) they are intentionally renamed at the call
  /// site.
  if (verifyMatchingCallLists(
          llvm::map_range(
              callerInputParams,
              [](ParamBindAttr value) -> Attribute { return value.getName(); }),
          llvm::map_range(
              calleeInputParamDecls,
              [](ParamDeclAttr value) -> Attribute { return value.getName(); }),
          *this, callee, "input parameter", "name"))
    return failure();

  // We need to substitute and simplify expressions that occur in the argument
  // list and parameter types, e.g.:
  //     kgen.generator @callee1<type: dtype>(%x: !meta.scalar<type>)
  //     kgen.generator @callee2<size>(%x: !meta.simd<size, f32>)
  // ... call @callee1<type: dtype = f32>(%arg1) : (!meta.scalar<f32>) -> ()
  // ... call @callee2<size=4>(%arg2) : (!meta.simd<4, f32>) -> ()
  //
  // This can also occur in parameter types, e.g. for region types (dt vs f32):
  //     kgen.generator @g<dt: dtype, region: () -> !meta.scalar<dt>>(...
  //     call @g<dt: f32, region: () -> !meta.scalar<f32>(...

  // We do this with with ParameterEvaluator which can do the remapping for us.
  ParameterEvaluator evaluator;
  for (auto [bind, decl] :
       llvm::zip(callerInputParams, calleeInputParamDecls)) {
    evaluator.setParameterValue(decl, bind.getValue());
  }
  auto remapType = [&](Type type) -> Type {
    return evaluator.getReboundType(type);
  };

  // Check input parameter types match.
  if (verifyMatchingCallLists(
          llvm::map_range(callerInputParams,
                          [](Attribute value) -> Type {
                            return value.cast<ParamBindAttr>().getType();
                          }),
          llvm::map_range(getParamDeclType(calleeInputParamDecls), remapType),
          *this, callee, "input parameter", "type") ||

      /// Check result parameter types.
      verifyMatchingCallLists(getParamDeclType(callerOutputParamDecls),
                              calleeOutputParamTypes, *this, callee,
                              "output parameter", "type")) {
    return failure();
  }

  // Ok, now that we know the parameters match up, verify that the operand
  // and result types match the callee.
  auto fnType = callee.getFunctionType();
  auto calleeInputTypes = llvm::map_range(fnType.getInputs(), remapType);
  auto calleeResultTypes = llvm::map_range(fnType.getResults(), remapType);

  // Check that the passed in operands, and returned types match our
  // expectations.
  if (verifyMatchingCallLists(getOperandTypes(), calleeInputTypes, *this,
                              callee, "input", "type") ||
      verifyMatchingCallLists(getResultTypes(), calleeResultTypes, *this,
                              callee, "result", "type"))
    return failure();

  // Ok, the call looks good.  Next, make sure that calls within a
  // kgen.func do not pass input arguments.  Input arguments are invalid to a
  // func, so it must be a generator or generator interface, and these will
  // not be elaborated unless they have zero input arguments.
  if (!callerInputParams.empty()) {
    if (auto funcParent = getOperation()->getParentOfType<FuncOp>()) {
      auto diag = emitError() << "cannot call generator with input arguments "
                                 "from concrete kgen.func";
      diag.attachNote(funcParent->getLoc())
          << "within kgen.func '" << funcParent.getName() << "'";
      return failure();
    }
  }

  return success();
}

void CallOp::build(OpBuilder &builder, OperationState &state,
                   TypeRange resultTypes, StringAttr callee,
                   ArrayRef<ParamBindAttr> inputParams,
                   ArrayRef<ParamDeclAttr> resultParams,
                   OperandRange operands) {
  build(builder, state, resultTypes, FlatSymbolRefAttr::get(callee),
        builder.getAttr<ParamBindArrayAttr>(inputParams),
        builder.getAttr<ParamDeclArrayAttr>(resultParams), operands);
}

//===----------------------------------------------------------------------===//
// ParamConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ParamConstantOp::fold(ArrayRef<Attribute> constants) {
  assert(constants.empty() && "kgen.param.constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

/// Containers verify that the operands of this ReturnOp match the specified set
/// of types.
LogicalResult ReturnOp::checkArgumentTypes(ArrayRef<Type> paramResultTypes,
                                           TypeRange types) {
  // Check the parameters match up.
  auto returnedParams = getParameters();
  if (returnedParams.size() != paramResultTypes.size())
    return emitOpError("expected ")
           << paramResultTypes.size() << " parameters for enclosing op";

  for (size_t i = 0, e = paramResultTypes.size(); i != e; ++i) {
    auto expectedTy = paramResultTypes[i];
    auto actualTy = returnedParams[i].cast<TypedAttr>().getType();
    if (actualTy != expectedTy)
      return emitOpError("parameter #") << i << " has type " << actualTy
                                        << " but should be " << expectedTy;
  }

  // Verify our result types match up with the enclosing result type.
  if (getNumOperands() != types.size())
    return emitOpError("expected ")
           << types.size() << " operands for enclosing op";

  for (size_t i = 0, e = getNumOperands(); i != e; ++i) {
    if (getOperand(i).getType() != types[i])
      return emitOpError("operand #")
             << i << " has type " << getOperand(i).getType()
             << " but should be " << types[i];
  }
  return success();
}

//===----------------------------------------------------------------------===//
// RebindOp
//===----------------------------------------------------------------------===//

/// If either the input or output type are parameterized, return success.
/// Otherwise, require that the concrete input and output types are the same.
LogicalResult RebindOp::verify() {
  SmallVector<ParamDeclRefAttr> inputRefs, outputRefs;
  if (failed(collectParameterReferences(getInput().getType(), inputRefs)))
    return failure();
  if (!inputRefs.empty())
    return success();
  if (failed(collectParameterReferences(getType(), outputRefs)))
    return failure();
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
// Precompiled*Op
//===----------------------------------------------------------------------===//

/// Parse a `kgen.precompiled.*` op. They all have almost exactly the same form,
/// so we can use a single function to parse them all. This defines the form of
/// all the `kgen.precompiled.*` ops to be:
///
///   kgen.precompiled.* (public|private) @symbol(...) -> (...) attributes {
///     ...
///   }
///
/// Note that none of them have bodies, but they do keep the function
/// signatures.
static ParseResult parsePrecompiledOp(OpAsmParser &parser,
                                      OperationState &result) {
  using namespace mlir::function_interface_impl;

  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse visibility. If none is provided, use private by default.
  if (failed(mlir::impl::parseOptionalVisibilityKeyword(parser,
                                                        result.attributes)))
    result.addAttribute(SymbolTable::getVisibilityAttrName(),
                        parser.getBuilder().getStringAttr("private"));

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  bool isVariadic = false;
  if (parseFunctionSignature(parser, /*allowVariadic=*/false, entryArgs,
                             isVariadic, resultTypes, resultAttrs))
    return failure();

  // Get the argument types
  SmallVector<Type> argTypes;
  argTypes.reserve(entryArgs.size());
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);

  // Create and store the function type on the op.
  Type type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  // If function attributes are present, parse them.
  NamedAttrList parsedAttributes;
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();

  // Add the processed attr list to the OperationState.
  result.attributes.append(parsedAttributes);

  // Empty body region, always.
  (void)result.addRegion();

  return success();
}

/// Print a `kgen.precompiled.*` op. They all have almost exactly the same form
/// so we use a single function to handle them all. See `parsePrecompiledOp` for
/// an example of the form we want printed.
static void printPrecompiledOp(OpAsmPrinter &p, Operation *op) {
  using namespace mlir::function_interface_impl;
  auto symOp = cast<mlir::SymbolOpInterface>(op);
  auto funcOp = cast<mlir::FunctionOpInterface>(op);

  // Print the operation and the function name.
  p << ' ';

  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility = op->getAttrOfType<StringAttr>(visibilityAttrName))
    if (visibility.getValue() != "private")
      p << visibility.getValue() << ' ';
  p.printSymbolName(symOp.getName());

  ArrayRef<Type> argTypes = funcOp.getArgumentTypes();
  ArrayRef<Type> resultTypes = funcOp.getResultTypes();
  printFunctionSignature(p, op, argTypes, /*isVariadic=*/false, resultTypes);
  printFunctionAttributes(p, op, argTypes.size(), resultTypes.size(),
                          {mlir::SymbolTable::getSymbolAttrName(),
                           mlir::SymbolTable::getVisibilityAttrName(),
                           mlir::FunctionOpInterface::getTypeAttrName()});
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "KGEN/KGENDialect/KGEN.cpp.inc"

// Generated interface definitions.
#include "KGEN/KGENDialect/ElaboratorOpInterface.cpp.inc"
#include "KGEN/KGENDialect/KGENDeclInterface.cpp.inc"
