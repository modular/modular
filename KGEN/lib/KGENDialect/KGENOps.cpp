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
#include "KGEN/KGENDialect/ElaboratorOpInterface.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENCallInterface.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

// FIXME: KGENDialect should not depend on POPDialect.
#include "KGEN/POPDialect/POPTypes.h"

using namespace M;
using namespace KGEN;
using mlir::TypedValue;

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
// custom<ParameterValues>
//===----------------------------------------------------------------------===//

static ParseResult parseParameterValues(OpAsmParser &p,
                                        ParameterExprArrayAttr &value) {
  SmallVector<TypedAttr> elts;
  if (p.parseCommaSeparatedList(
          OpAsmParser::Delimiter::OptionalLessGreater, [&]() -> ParseResult {
            TypedAttr value;
            if (parseParamValueDefaultingToIndex(p, value))
              return failure();
            elts.push_back(value);
            return success();
          }))
    return failure();

  value = ParameterExprArrayAttr::get(p.getContext(), elts);
  return success();
}

static void printParameterValues(OpAsmPrinter &p, Operation *op,
                                 ParameterExprArrayAttr value) {
  if (value.empty())
    return;
  p << '<';
  llvm::interleaveComma(value, p, [&](TypedAttr value) {
    auto valType = value.getType();
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
  KGENDeclInterface parent = op->getParentOfType<KGENDeclInterface>();
  if (parent) {
    collectParameterReferences(cond, parameterRefs);
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
  return cast<ReturnOp>(getBody()->getTerminator());
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
                                              {getResultTypes()})))
    return failure();

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
  auto module = (*this)->getParentOfType<ModuleOp>();
  auto interface = dyn_cast_if_present<GeneratorInterfaceOp>(
      symbolTable.lookupSymbolIn(module, interfaceSym));
  if (!interface)
    return emitError() << "'" << interfaceSym
                       << "' does not reference a generator interface";

  // Verify that the signature of this generator matches the signature of the
  // interface.
  return verifyDeclMatchesInterface("generator", *this, "interface", interface);
}

Region *GeneratorOp::getCallableRegion() { return &getBodyRegion(); }

ArrayRef<Type> GeneratorOp::getCallableResults() {
  return getFunctionType().getResults();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

/// Create a func with no body block.  The caller must create it and fill
/// it in.
void FuncOp::build(OpBuilder &builder, OperationState &result, StringAttr name,
                   FunctionType signature, ArrayRef<Type> resultParamTypes) {
  build(builder, result, name, signature, {}, resultParamTypes);
}

ReturnOp FuncOp::getReturnOp() {
  return cast<ReturnOp>(getBody()->getTerminator());
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
                                              {getResultTypes()})))
    return failure();

  // kgen.func's are not allowed to have input parameter lists.
  if (!getParamDecls().empty())
    return emitError(
        "kgen.func only allows output parameters, not input parameters");

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
  auto func = module.lookup<KGENDeclInterface>(evaluator.getSymbol());
  if (!func)
    return emitOpError("evaluator ")
           << evaluator.getSymbol() << " does not refer to a KGEN declaration";

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

  func = module.lookup<KGENDeclInterface>(defaultImpl.getSymbol());
  if (!func)
    return emitOpError("defaultImpl ")
           << defaultImpl.getSymbol()
           << " does not refer to a KGEN declaration";

  if (!isa<GeneratorOp>(func))
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
// Common CallOp / CallParamOp logic
//===----------------------------------------------------------------------===//

/// Verify invariants for a CallOp or CallParamOp with a callee of a known
/// signature.
static ParseResult verifyCallAndCallee(Operation *theCall,
                                       SignatureType callerSignature,
                                       SignatureType calleeSignature,
                                       ParamBindArrayAttr callerInputParams,
                                       Location calleeLoc) {
  // Get the substituted signature based on the input parameters specified.
  auto emitErrorFn = [&]() { return theCall->emitError(); };
  calleeSignature =
      calleeSignature.getSpecializedSignature(callerInputParams, emitErrorFn);
  if (!calleeSignature)
    return failure();

  return verifyDeclSignaturesMatch("caller", callerSignature, theCall->getLoc(),
                                   "callee", calleeSignature, calleeLoc);
}

/// Verify that regions used as signature parameters match in signature.
template <typename RegionBodyT>
static LogicalResult verifyRegionSignatures(Operation *theCall,
                                            ParamBindArrayAttr values) {
  auto regionValues = llvm::make_filter_range(values, [](ParamBindAttr value) {
    return value.getValue().isa<ParamCallRegionRefAttr>();
  });

  size_t numRegionParams =
      std::distance(regionValues.begin(), regionValues.end());
  if (numRegionParams != theCall->getNumRegions())
    return theCall->emitOpError("expected ")
           << numRegionParams << " body regions but has "
           << theCall->getNumRegions();

  // Ensure each region parameter matches up in order with the regions.
  for (auto &it : llvm::enumerate(regionValues)) {
    auto paramSignature =
        it.value().getValue().getType().template cast<SignatureType>();
    Region &region = theCall->getRegion(it.index());
    auto body = cast<RegionBodyT>(region.front().getTerminator());
    if (region.front().getOperations().size() != 1)
      return theCall->emitOpError("expected region #")
             << it.index() << " to contain only a `kgen.region.body` op";

    auto regionSignature = SignatureType::get(body.getParamDeclsAttr(),
                                              body.getResultParamTypesAttr(),
                                              body.getFunctionType());
    if (failed(verifyDeclSignaturesMatch("region", regionSignature,
                                         body.getLoc(), "parameter",
                                         paramSignature, theCall->getLoc())))
      return failure();
  }
  return success();
}

template <typename BodyOpT>
static ParseResult
parseCallRegionBodies(OpAsmParser &p,
                      SmallVectorImpl<std::unique_ptr<::mlir::Region>> &result,
                      ParamBindArrayAttr paramValues) {
  // We expect one region for each ParamCallRegionRefAttr.
  auto binds = llvm::make_filter_range(paramValues, [](ParamBindAttr bind) {
    return bind.getValue().isa<ParamCallRegionRefAttr>();
  });

  auto parseFn = [&](ParamBindAttr bind) -> ParseResult {
    // Parse the region body operation in-line.
    OperationState regionBody(p.getEncodedSourceLoc(p.getCurrentLocation()),
                              BodyOpT::getOperationName());
    Optional<Location> bodyLoc = regionBody.location;
    if (p.parseKeyword(bind.getName(), " region name") ||
        BodyOpT::parse(p, regionBody) ||
        p.parseOptionalLocationSpecifier(bodyLoc))
      return failure();
    regionBody.location = *bodyLoc;

    // Create a single-block body with only the region body operation.
    auto *body = new Block;
    body->push_back(Operation::create(regionBody));
    auto region = std::make_unique<Region>();
    region->push_back(body);
    result.push_back(std::move(region));
    return success();
  };
  return failableInterleave(binds, parseFn, [&] { return p.parseComma(); });
}

template <typename RegionBodyT>
static void printCallRegionBodies(OpAsmPrinter &p, mlir::RegionRange regions,
                                  ParamBindArrayAttr paramValues) {
  auto binds = llvm::make_filter_range(paramValues, [](ParamBindAttr bind) {
    return bind.getValue().isa<ParamCallRegionRefAttr>();
  });

  auto printFn = [&](auto &bind) {
    p.printNewline();
    p << bind.value().getName().strref();
    Operation *body = regions[bind.index()]->front().getTerminator();
    cast<RegionBodyT>(body).print(p);
    p.printOptionalLocationSpecifier(body->getLoc());
  };
  llvm::interleave(llvm::enumerate(binds), p, printFn, ",");
}

static ParseResult
parseCallRegions(OpAsmParser &p,
                 SmallVectorImpl<std::unique_ptr<::mlir::Region>> &result,
                 ParamBindArrayAttr paramValues) {
  return parseCallRegionBodies<RegionBodyOp>(p, result, paramValues);
}

static void printCallRegions(OpAsmPrinter &p, Operation *op,
                             mlir::RegionRange regions,
                             ParamBindArrayAttr paramValues) {
  return printCallRegionBodies<RegionBodyOp>(p, regions, paramValues);
}

//===----------------------------------------------------------------------===//
// AddressOfOp
//===----------------------------------------------------------------------===//

LogicalResult
AddressOfOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto module = KGENModule::from(*this, symbolTable);
  auto callee = module.lookup<KGENDeclInterface>(getCalleeAttr());
  if (!callee)
    return emitOpError() << getCalleeAttr()
                         << " does not reference a valid callee";

  // Check the parameters and operands align with the requirements of the
  // callee's signature.
  if (verifyCallAndCallee(*this, getSignature(), callee.getSignature(),
                          getParamValuesAttr(), callee->getLoc()))
    return failure();

  // Make sure we don't reference a generator with input parameters inside a
  // `kgen.func`.
  if (!getParamValues().empty()) {
    if (auto funcParent = getOperation()->getParentOfType<FuncOp>()) {
      auto diag = emitError() << "cannot reference generator with input "
                                 "arguments from concrete kgen.func";
      diag.attachNote(funcParent->getLoc())
          << "within kgen.func '" << funcParent.getName() << "'";
      return failure();
    }
  }

  return success();
}

LogicalResult AddressOfOp::verifyRegions() {
  return verifyRegionSignatures<RegionBodyOp>(*this, getParamValuesAttr());
}

FunctionType AddressOfOp::getFunctionType() { return getType(); }

void AddressOfOp::build(OpBuilder &b, OperationState &state, Type type,
                        StringAttr callee, ArrayRef<ParamBindAttr> inputParams,
                        ArrayRef<ParamDeclAttr> resultParams) {
  build(b, state, type, FlatSymbolRefAttr::get(callee),
        b.getAttr<ParamBindArrayAttr>(inputParams),
        b.getAttr<ParamDeclArrayAttr>(resultParams), 0);
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto module = KGENModule::from(*this, symbolTable);
  auto callee = module.lookup<KGENDeclInterface>(getCalleeAttr());
  if (!callee)
    return emitError() << "'" << getCalleeAttr()
                       << "' does not reference a valid callee";

  // Check the parameters and operands align with the requirements of the
  // callee's signature.
  if (verifyCallAndCallee(*this, getSignature(), callee.getFullSignature(),
                          getParamValuesAttr(), callee->getLoc()))
    return failure();

  // Ok, the call looks good.  Next, make sure that calls within a
  // kgen.func do not pass input arguments.  Input arguments are invalid to a
  // func, so it must be a generator or generator interface, and these will
  // not be elaborated unless they have zero input arguments.
  if (!getParamValues().empty()) {
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
                   TypeRange resultTypes, SymbolRefAttr callee,
                   ArrayRef<ParamBindAttr> inputParams,
                   ArrayRef<ParamDeclAttr> resultParams, ValueRange operands) {
  build(builder, state, resultTypes, callee,
        builder.getAttr<ParamBindArrayAttr>(inputParams),
        builder.getAttr<ParamDeclArrayAttr>(resultParams), operands,
        /*numRegions=*/0);
}
void CallOp::build(OpBuilder &builder, OperationState &state,
                   TypeRange resultTypes, StringAttr callee,
                   ArrayRef<ParamBindAttr> inputParams,
                   ArrayRef<ParamDeclAttr> resultParams, ValueRange operands) {
  build(builder, state, resultTypes, FlatSymbolRefAttr::get(callee),
        inputParams, resultParams, operands);
}

OperandRange CallOp::getArgOperands() { return getOperands(); }

mlir::CallInterfaceCallable CallOp::getCallableForCallee() {
  return getCalleeAttr();
}

LogicalResult CallOp::verifyRegions() {
  // Verify the region signatures match region parameter signatures.
  return verifyRegionSignatures<RegionBodyOp>(*this, getParamValuesAttr());
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

  auto signature = dyn_cast<SignatureType>(value.getType());
  if (!signature)
    return p.emitError(loc, "callee parameter type must be a signature type");

  // Get the substituted signature based on the input parameters specified and
  // check that the parameter names/types specified match up with the expected
  // ones.
  auto emitErrorFn = [&]() { return p.emitError(loc); };
  auto substitutedSignature =
      signature.getSpecializedSignature(paramValues, emitErrorFn);
  if (!substitutedSignature)
    return failure();

  llvm::append_range(operandTypes,
                     substitutedSignature.getValues().getInputs());
  llvm::append_range(resultTypes,
                     substitutedSignature.getValues().getResults());
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
  if (auto calleeSymbol = dyn_cast<SymbolConstantAttr>(op.getCallee())) {
    rewriter.replaceOpWithNewOp<CallOp>(
        op, op.getResultTypes(), calleeSymbol.getSymbol().getLeafReference(),
        op.getParamValues(), op.getParamDecls(), op.getOperands());
    return success();
  }

  return failure();
}

void CallParamOp::build(OpBuilder &builder, OperationState &state,
                        TypeRange resultTypes, TypedAttr callee,
                        ArrayRef<ParamBindAttr> inputParams,
                        ArrayRef<ParamDeclAttr> resultParams,
                        ValueRange operands) {
  build(builder, state, resultTypes, callee,
        builder.getAttr<ParamBindArrayAttr>(inputParams),
        builder.getAttr<ParamDeclArrayAttr>(resultParams), operands,
        /*numRegions=*/0);
}

LogicalResult CallParamOp::verifyRegions() {
  KGENDeclInterface parent =
      getOperation()->getParentOfType<KGENDeclInterface>();
  if (!parent || isa<FuncOp>(parent))
    return emitError(
        "kgen.call_param is only allowed in generators pre-elaboration");

  // Check the parameters and operands align with the requirements of the
  // callee's signature.
  auto calleeSignature = cast<SignatureType>(getCallee().getType());
  if (failed(verifyCallAndCallee(*this, getSignature(), calleeSignature,
                                 getParamValuesAttr(), getLoc())))
    return failure();

  // Verify the region signatures match region parameter signatures.
  return verifyRegionSignatures<RegionBodyOp>(*this, getParamValuesAttr());
}

//===----------------------------------------------------------------------===//
// InlinedCallOp
//===----------------------------------------------------------------------===//

LogicalResult InlinedCallOp::verifyRegions() {
  KGENDeclInterface parent =
      getOperation()->getParentOfType<KGENDeclInterface>();
  if (!parent || isa<FuncOp>(parent))
    return emitOpError("is only allowed in generators pre-elaboration");

  // Check the parameters and operands align with the requirements of the
  // callee's signature.
  auto calleeSignature = cast<SignatureType>(getCallee().getType());
  if (failed(verifyCallAndCallee(*this, getSignature(), calleeSignature,
                                 getParamValuesAttr(), getLoc())))
    return failure();

  // Verify the region signatures match region parameter signatures.
  return verifyRegionSignatures<RegionOpenBodyOp>(*this, getParamValuesAttr());
  return success();
}

static ParseResult parseInPlaceCallRegions(
    OpAsmParser &p, SmallVectorImpl<std::unique_ptr<::mlir::Region>> &result,
    ParamBindArrayAttr paramValues) {
  return parseCallRegionBodies<RegionOpenBodyOp>(p, result, paramValues);
}

static void printInPlaceCallRegions(OpAsmPrinter &p, Operation *op,
                                    mlir::RegionRange regions,
                                    ParamBindArrayAttr paramValues) {
  return printCallRegionBodies<RegionOpenBodyOp>(p, regions, paramValues);
}

//===----------------------------------------------------------------------===//
// RegionBodyOp / custom<RegionBody>
//===----------------------------------------------------------------------===//

ReturnOp RegionBodyOp::getReturnOp() {
  return cast<ReturnOp>(getBody()->getTerminator());
}

/// Parse a single-block isolated from above region.
static ParseResult parseRegionBody(OpAsmParser &p, Region &body) {
  SmallVector<OpAsmParser::Argument> args;
  if (p.parseArgumentList(args, AsmParser::Delimiter::Paren,
                          /*allowType=*/true) ||
      p.parseRegion(body, args))
    return failure();
  return success();
}

/// Print a single-block isolated from above region.
static void printRegionBody(OpAsmPrinter &p, Operation *op, Region &body) {
  p << '(';
  llvm::interleaveComma(body.getArguments(), p,
                        [&](BlockArgument arg) { p.printRegionArgument(arg); });
  p << ") ";
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

/// Derive the region's function type from its arguments and result types.
FunctionType RegionBodyOp::getFunctionType() {
  Block *body = getBody();
  return FunctionType::get(getContext(), body->getArgumentTypes(),
                           body->getTerminator()->getOperandTypes());
}

LogicalResult RegionBodyOp::verifyRegions() {
  return getReturnOp().checkArgumentTypes(getResultParamTypes(), None);
}

//===----------------------------------------------------------------------===//
// RegionOpenBodyOp
//===----------------------------------------------------------------------===//

ReturnOp RegionOpenBodyOp::getReturnOp() {
  return cast<ReturnOp>(getBody()->getTerminator());
}

FunctionType RegionOpenBodyOp::getFunctionType() {
  return FunctionType::get(getContext(), getBodyRegion().getArgumentTypes(),
                           getReturnOp().getOperandTypes());
}

LogicalResult RegionOpenBodyOp::verifyRegions() {
  return getReturnOp().checkArgumentTypes(getResultParamTypes(), None);
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

/// Infer the callee type from the input and result types.
static ParseResult parseCallIndirectCalleeType(AsmParser &p, Type &calleeType,
                                               TypeRange inputTypes,
                                               TypeRange resultTypes) {
  calleeType = SignatureType::get(p.getContext(), inputTypes, resultTypes);
  return success();
}

static void printCallIndirectCalleeType(AsmPrinter &p, Operation *op,
                                        Type calleeType, TypeRange inputTypes,
                                        TypeRange resultTypes) {}

/// Require that the signature type has no input or output parameters.
LogicalResult CallIndirectOp::verify() {
  SignatureType calleeType = getCallee().getType();
  if (calleeType.getInputParams().empty() &&
      calleeType.getResultParamTypes().empty())
    return success();
  return emitOpError("requires the signature callee to have no input or output "
                     "parameters")
             .attachNote()
         << "use `bind_signature`";
}

/// Canonicalize `call_indirect(partial_apply) -> call_indirect` by folding the
/// bound arguments into the call, and canonicalize `call_indirect(constant)`
/// into `call_param`.
LogicalResult CallIndirectOp::canonicalize(CallIndirectOp op,
                                           PatternRewriter &rewriter) {
  Operation *calleeOp = op.getCallee().getDefiningOp();
  if (auto constant = dyn_cast_or_null<ParamConstantOp>(calleeOp)) {
    rewriter.replaceOpWithNewOp<CallParamOp>(
        op, op.getResultTypes(), constant.getValue(), ArrayRef<ParamBindAttr>(),
        ArrayRef<ParamDeclAttr>(), op.getInputs());
    return success();
  }

  if (auto bind = dyn_cast_or_null<PartialApplyOp>(calleeOp)) {
    SmallVector<Value> newInputs;
    int64_t totalInputs = op.getInputs().size() + bind.getInputs().size();
    newInputs.reserve(totalInputs);
    auto boundIt = bind.getBoundInputs().begin();
    auto curInputsIt = op.getInputs().begin();
    auto boundInputsIt = bind.getInputs().begin();
    for (int64_t i = 0; i < totalInputs; ++i) {
      if (boundIt == bind.getBoundInputs().end() || i < *boundIt) {
        newInputs.push_back(*curInputsIt++);
      } else {
        ++boundIt;
        newInputs.push_back(*boundInputsIt++);
      }
    }
    rewriter.replaceOpWithNewOp<CallIndirectOp>(op, op.getResultTypes(),
                                                bind.getCallee(), newInputs);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// PartialApplyOp
//===----------------------------------------------------------------------===//

static Type computePartialApplyResultType(Optional<Location> loc,
                                          TypedValue<SignatureType> callee,
                                          ValueRange inputs,
                                          ArrayRef<int64_t> boundInputs) {
  auto emitError = [&](const Twine &msg) -> Type {
    (void)mlir::emitOptionalError(loc, "'kgen.partial_apply' op " + msg);
    return {};
  };
  // Ensure the indices are sorted.
  if (!llvm::is_sorted(boundInputs))
    return emitError("expected indices to be sorted ascending");
  if (boundInputs.size() != inputs.size())
    return emitError("mismatch between number of indices and inputs: " +
                     Twine(boundInputs.size()) + " vs " + Twine(inputs.size()));

  auto origSignature = callee.getType();

  DenseSet<int64_t> seenInputs;
  seenInputs.reserve(boundInputs.size());
  ArrayRef<Type> argumentTypes = origSignature.getValues().getInputs();
  SmallVector<Type> newInputTypes;
  unsigned lastIdx = 0;
  for (auto [input, index] : llvm::zip(inputs, boundInputs)) {
    if (index >= static_cast<int64_t>(argumentTypes.size()))
      return emitError("bound input index is out of range: " + Twine(index));
    if (!seenInputs.insert(index).second)
      return emitError("duplicate bound input index: " + Twine(index));
    if (input.getType() != argumentTypes[index])
      return emitError("input bound to argument #" + Twine(index) +
                       " is incorrect");
    // Pick the types of arguments that aren't bound.
    while (lastIdx++ < index)
      newInputTypes.push_back(argumentTypes[lastIdx - 1]);
  }
  for (; lastIdx < argumentTypes.size(); ++lastIdx)
    newInputTypes.push_back(argumentTypes[lastIdx]);

  assert(newInputTypes.size() == argumentTypes.size() - boundInputs.size());

  auto resultFnType =
      FunctionType::get(origSignature.getContext(), newInputTypes,
                        origSignature.getValues().getResults());
  return SignatureType::get(origSignature.getInputParams(),
                            origSignature.getResultParamTypes(), resultFnType);
}

LogicalResult PartialApplyOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, RegionRange regions, SmallVectorImpl<Type> &types) {

  mlir::OperationName name(getOperationName(), attrs.getContext());
  auto boundInputs =
      attrs.getAs<mlir::DenseI64ArrayAttr>(getBoundInputsAttrName(name));
  if (!boundInputs || operands.empty() ||
      !isa<SignatureType>(operands[0].getType()))
    return mlir::emitOptionalError(loc, "missing required attributes");

  types.push_back(
      computePartialApplyResultType(loc, (TypedValue<SignatureType>)operands[0],
                                    operands.drop_front(), boundInputs));
  return success(types.back() != Type());
}

/// Verify the operation is well-formed. It is not possible to get an
/// ill-formed operation using the pretty syntax, but it is possible from C++.
LogicalResult PartialApplyOp::verify() {
  auto resultType = computePartialApplyResultType(
      getLoc(), getCallee(), getInputs(), getBoundInputs());
  if (!resultType)
    return failure();
  if (resultType != getType())
    return emitOpError("result signature does not match");
  return success();
}

/// Parse the input operands, using `?` to represent a placeholder value.
static ParseResult parseBoundInputs(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &inputs,
    mlir::DenseI64ArrayAttr &boundInputs, SmallVectorImpl<Type> &inputTypes,
    Type &resultType, Type &calleeType) {
  // Parse the binding list `(` ((`?` | operand) (`,` (`?` | operand))*)? `)`.
  SmallVector<int64_t> boundInputIndices;
  if (p.parseLParen())
    return failure();
  if (p.parseOptionalRParen()) {
    int64_t index = 0;
    OpAsmParser::UnresolvedOperand input;
    auto parseElt = [&]() -> ParseResult {
      llvm::SMLoc loc = p.getCurrentLocation();
      if (p.parseOptionalQuestion()) {
        mlir::OptionalParseResult result = p.parseOptionalOperand(input);
        if (result.has_value() && failed(*result))
          return failure();
        if (!result.has_value())
          return p.emitError(loc, "expected '?' or an operand in binding list");
        inputs.push_back(input);
        boundInputIndices.push_back(index);
      }
      ++index;
      return success();
    };
    if (p.parseCommaSeparatedList(parseElt) || p.parseRParen())
      return failure();
  }

  // Parse the input function type `:` functional-type.
  llvm::SMLoc loc = p.getCurrentLocation();
  FunctionType funcType;
  if (p.parseColonType(funcType))
    return failure();
  calleeType = SignatureType::get(funcType);
  boundInputs = p.getBuilder().getDenseI64ArrayAttr(boundInputIndices);

  // Infer the input types from the function type.
  SmallVector<Type> resultTypes;
  int64_t lastIdx = 0;
  int64_t numInputs = funcType.getNumInputs();
  for (int64_t index : boundInputIndices) {
    if (index >= numInputs)
      return p.emitError(loc, "there are more bound inputs than arguments");
    inputTypes.push_back(funcType.getInputs()[index]);
    while (lastIdx++ < index)
      resultTypes.push_back(funcType.getInputs()[lastIdx - 1]);
  }
  for (; lastIdx < numInputs; ++lastIdx)
    resultTypes.push_back(funcType.getInputs()[lastIdx]);

  // Infer the result signature type.
  resultType =
      SignatureType::get(p.getContext(), resultTypes, funcType.getResults());
  return success();
}

static void printBoundInputs(OpAsmPrinter &p, Operation *op, ValueRange inputs,
                             mlir::DenseI64ArrayAttr boundInputs,
                             TypeRange inputTypes, Type resultType,
                             Type calleeType) {
  FunctionType calleeSig = cast<SignatureType>(calleeType).getValues();

  p << '(';
  auto idxIt = boundInputs.asArrayRef().begin();
  int64_t index = 0;
  auto eachFn = [&](int64_t i) {
    if (idxIt == boundInputs.asArrayRef().end() || i < *idxIt) {
      p << '?';
    } else {
      ++idxIt;
      p << inputs[index++];
    }
  };
  llvm::interleaveComma(llvm::seq<int64_t>(0, calleeSig.getNumInputs()), p,
                        eachFn);
  p << ") : " << calleeSig;
}

/// Canonicalize `partial_apply(partial_apply))` by folding the bound operands
/// into the same operation.
LogicalResult PartialApplyOp::canonicalize(PartialApplyOp op,
                                           PatternRewriter &rewriter) {
  auto bind = dyn_cast_or_null<PartialApplyOp>(op.getCallee().getDefiningOp());
  if (!bind)
    return failure();
  // Merge the values and indices together.
  SmallVector<Value> newInputs;
  SmallVector<int64_t> newIndices;
  size_t totalInputs = op.getInputs().size() + bind.getInputs().size();
  newInputs.reserve(totalInputs);
  newIndices.reserve(totalInputs);
  auto lhsRange = llvm::zip(op.getInputs(), op.getBoundInputs());
  auto rhsRange = llvm::zip(bind.getInputs(), bind.getBoundInputs());
  auto lhs = lhsRange.begin(), rhs = rhsRange.begin(), lhsEnd = lhsRange.end(),
       rhsEnd = rhsRange.end();
  while (lhs != lhsEnd && rhs != rhsEnd) {
    auto [lhsInput, lhsIndex] = *lhs;
    auto [rhsInput, rhsIndex] = *rhs;
    if (lhsIndex < rhsIndex) {
      ++lhs;
      newInputs.push_back(lhsInput);
      newIndices.push_back(lhsIndex);
    } else {
      ++rhs;
      newInputs.push_back(rhsInput);
      newIndices.push_back(rhsIndex);
    }
  }
  auto pushTheRest = [&](auto it, auto end) {
    for (; it != end; ++it) {
      auto [input, index] = *it;
      newInputs.push_back(input);
      newIndices.push_back(index);
    }
  };
  pushTheRest(lhs, lhsEnd);
  pushTheRest(rhs, rhsEnd);
  rewriter.replaceOpWithNewOp<PartialApplyOp>(
      op, op.getType(), bind.getCallee(), newInputs, newIndices);
  return success();
}

//===----------------------------------------------------------------------===//
// ParamConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ParamConstantOp::fold(ArrayRef<Attribute> constants) {
  assert(constants.empty() && "kgen.param.constant has no operands");
  return getValueAttr();
}

void ParamConstantOp::build(OpBuilder &b, OperationState &state,
                            TypedAttr value) {
  build(b, state, value.getType(), value);
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

/// Containers verify that the operands of this ReturnOp match the specified set
/// of types.
LogicalResult ReturnOp::checkArgumentTypes(ArrayRef<Type> paramResultTypes,
                                           Optional<TypeRange> types) {
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

  // Verify the result types if they were provided.
  if (!types)
    return success();

  return checkResultTypes(getOperation(), *types);
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

/// Struct declarations aren't functions.
FunctionType StructDeclOp::getFunctionType() {
  llvm_unreachable("structs don't have function types");
}

/// Verify that the body has no arguments and that the declaration has no result
/// types.
LogicalResult StructDeclOp::verify() {
  if (getFields().getNumArguments())
    return emitOpError("expected declaration body to have no arguments");

  if (!getResultParamTypes().empty())
    return emitOpError("unexpected result parameters");

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
  return ParameterDeclsAndUses().calculateAndVerify(*this, symbolTable);
}

void StructDeclOp::build(OpBuilder &builder, OperationState &result,
                         StringAttr name) {
  auto context = builder.getContext();
  build(builder, result, name, ParamDeclArrayAttr::get(context, {}),
        TypeArrayAttr::get(context, {}));
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

/// Lookup the declaration for the struct. When checking field types, we can't
/// directly compare operation types to the struct field types because they are
/// parameterized under different domains. We have to rebind them.
static std::pair<StructDeclOp, ParameterEvaluator>
lookupStructDecl(SymbolTableCollection &symbolTable, Operation *user,
                 DeclRefType ref) {
  auto module = KGENModule::from(user, symbolTable);
  auto structDecl = module.lookup<StructDeclOp>(ref.getSymbol());
  // Currently, this is impossible to fail because the symbol use was verified
  // by the parameter verifier.
  assert(structDecl && "expected a struct declaration");

  ParameterEvaluator evaluator;
  for (ParamBindAttr bind : ref.getParamValues())
    evaluator.setParameterValue(bind.getDecl(), bind.getValue());

  return std::make_pair(structDecl, std::move(evaluator));
}

/// Verify the reference struct type.
LogicalResult
StructCreateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verify the types of the fields in the operands match those in the
  // struct declaration.
  auto [structDecl, evaluator] =
      lookupStructDecl(symbolTable, *this, getType());
  for (auto [fieldDecl, operand, i] :
       llvm::zip(structDecl.getFieldDecls(), getOperands(),
                 llvm::seq<unsigned>(0, getNumOperands()))) {
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
  auto [structDecl, evaluator] =
      lookupStructDecl(symbolTable, *this, getType());

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
  auto [structDecl, evaluator] = lookupStructDecl(symbolTable, op, ref);

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

//===----------------------------------------------------------------------===//
// ListGetOp
//===----------------------------------------------------------------------===//

LogicalResult ListGetOp::verify() {
  auto index = dyn_cast<IntegerAttr>(getIndex());
  Optional<int64_t> length = getList().getType().getResolvedLength();
  if (!index || !length)
    return success();
  if (index.getInt() < 0 || index.getInt() >= *length)
    return emitOpError("list index out-of-range");
  return success();
}

//===----------------------------------------------------------------------===//
// ListMakeOp
//===----------------------------------------------------------------------===//

static bool typeRangeMatches(Type type, TypeRange range) {
  return llvm::all_of(range, [&](Type e) { return type == e; });
}

LogicalResult ListMakeOp::verify() {
  if (getResult().getType().getLength() !=
      Builder(getContext()).getIndexAttr(getNumOperands()))
    return emitOpError("expected result list to have ")
           << getNumOperands() << "elements";
  return success();
}

//===----------------------------------------------------------------------===//
// ListIterateOp
//===----------------------------------------------------------------------===//

static ParseResult
parseIteratePreamble(OpAsmParser &p, OpAsmParser::UnresolvedOperand &list,
                     SmallVectorImpl<OpAsmParser::UnresolvedOperand> &arguments,
                     ParameterExprArrayAttr &init, mlir::AffineMapAttr &map,
                     SmallVectorImpl<Type> &argumentTypes, Type &listType,
                     Region &body) {
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<TypedAttr> initExprs;
  mlir::AffineMap mapValue;
  llvm::SMLoc loc;

  OpAsmParser::Argument singleArg;
  mlir::OptionalParseResult result = p.parseOptionalArgument(singleArg);
  if (result.has_value()) {
    if (failed(*result))
      return failure();
    args.push_back(singleArg);
  } else if (p.parseArgumentList(args, AsmParser::Delimiter::Paren)) {
    return failure();
  }

  if (p.parseKeyword("in") || p.parseOperand(list) || p.parseColon() ||
      p.getCurrentLocation(&loc) || parseKGENType(p, listType) ||
      p.parseLSquare() || p.parseCommaSeparatedList([&] {
        return parseIndexParamValue(p, initExprs.emplace_back());
      }) ||
      p.parseColon() || p.parseAffineMap(mapValue) || p.parseRSquare())
    return failure();

  auto listTy = dyn_cast<ListType>(listType);
  if (!listTy)
    return p.emitError(loc, "expected a list type");
  for (OpAsmParser::Argument &arg : args)
    arg.type = ParamRefType::get(listTy.getElementType());

  if (succeeded(p.parseOptionalLParen())) {
    if (p.parseOptionalRParen()) {
      auto parseArg = [&]() -> ParseResult {
        if (p.parseArgument(args.emplace_back()) || p.parseEqual() ||
            p.parseOperand(arguments.emplace_back()))
          return failure();
        return success();
      };
      if (p.parseCommaSeparatedList(parseArg) || p.parseRParen())
        return failure();
    }
    loc = p.getCurrentLocation();
    if (p.parseArrowTypeList(argumentTypes))
      return failure();
    if (argumentTypes.size() != arguments.size())
      return p.emitError(
                 loc, "expected the same number of result types as arguments: ")
             << arguments.size() << " but got " << argumentTypes.size();
    for (auto &[idx, arg] : llvm::enumerate(
             MutableArrayRef<OpAsmParser::Argument>(args).drop_front(
                 initExprs.size())))
      arg.type = argumentTypes[idx];
  }
  init = ParameterExprArrayAttr::get(p.getContext(), initExprs);
  map = mlir::AffineMapAttr::get(mapValue);
  return p.parseRegion(body, args);
}

static void printIteratePreamble(OpAsmPrinter &p, Operation *op, Value list,
                                 ValueRange arguments,
                                 ParameterExprArrayAttr init,
                                 mlir::AffineMapAttr map,
                                 TypeRange argumentTypes, Type listType,
                                 Region &body) {
  if (init.size() != 1)
    p << '(';
  llvm::interleaveComma(body.getArguments().take_front(init.size()), p);
  if (init.size() != 1)
    p << ')';
  p << " in ";
  p << list << " : ";
  printKGENType(p.getStream(), listType);
  p << " [";
  llvm::interleaveComma(
      init, p, [&](TypedAttr initExpr) { printIndexParamValue(p, initExpr); });
  p << " : ";
  p << map.getValue();
  p << ']';
  if (!arguments.empty()) {
    p << " (";
    llvm::interleaveComma(
        llvm::zip(arguments, body.getArguments().drop_front(init.size())), p,
        [&](auto pair) {
          p << std::get<1>(pair) << " = " << std::get<0>(pair);
        });
    p << ')';
    p.printArrowTypeList(argumentTypes);
  }
  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

LogicalResult ListIterateOp::verify() {
  // Verify errors that are not structurally possible with the custom syntax.
  if (getBody().getNumArguments() != getInit().size() + getArguments().size())
    return emitOpError(
        "expected the number of region arguments to match the number of "
        "indices plus the number of loop-carried values");
  auto elementType = ParamRefType::get(getList().getType().getElementType());
  if (!llvm::all_of(
          llvm::drop_end(getBody().getArgumentTypes(), getArguments().size()),
          [&](Type type) { return type == elementType; }))
    return emitOpError("expected first ")
           << getInit().size() << " argument types to be list element type "
           << elementType;
  if (!llvm::equal(
          llvm::drop_begin(getBody().getArgumentTypes(), getInit().size()),
          getArguments().getTypes()))
    return emitOpError("expected last ")
           << getArguments().size()
           << " argument types to be equal to the initial value types";
  if (getMap().getNumDims() != getInit().size())
    return emitOpError("expected map to have ")
           << getInit().size() << " variable inputs";
  if (getMap().getNumResults() != getInit().size())
    return emitOpError("expected map to have ")
           << getInit().size() << " results";
  if (getMap().getNumSymbols())
    return emitOpError("expected map to have 0 symbolic inputs");
  return success();
}

//===----------------------------------------------------------------------===//
// ListYieldOp
//===----------------------------------------------------------------------===//

LogicalResult ListYieldOp::verify() {
  auto iterate = (*this)->getParentOfType<ListIterateOp>();
  if (getOperandTypes() != iterate.getArguments().getTypes())
    return emitOpError(
        "operand types do not match surrounding iterate arguments");
  return success();
}

//===----------------------------------------------------------------------===//
// IterateOp
//===----------------------------------------------------------------------===//

static ParseResult
parseIterate(OpAsmParser &p,
             SmallVectorImpl<OpAsmParser::UnresolvedOperand> &arguments,
             SmallVectorImpl<Type> &argumentTypes, ParameterExprArrayAttr &init,
             TypedAttr &next, TypedAttr &cond, Region &body) {
  // Parse the parameter declarations `(` (name `:` type)* `)` `in` `[`
  SmallVector<Type> paramTypes;
  SmallVector<ParamDeclAttr> params;
  auto parseParam = [&] {
    StringAttr name;
    if (parseParamName(p, name) ||
        parseColonTypeOrIndex(p, paramTypes.emplace_back()))
      return failure();
    params.push_back(ParamDeclAttr::get(name, paramTypes.back()));
    return mlir::success();
  };
  if (p.parseCommaSeparatedList(AsmParser::Delimiter::Paren, parseParam) ||
      p.parseKeyword("in") || p.parseLSquare())
    return failure();

  // Parse the init values using the types of the parameters
  SmallVector<TypedAttr> initVals;
  initVals.reserve(params.size());
  auto paramTypeIt = paramTypes.begin(), paramTypeE = paramTypes.end();
  auto parseInit = [&]() -> ParseResult {
    if (paramTypeIt == paramTypeE)
      return p.emitError(p.getCurrentLocation(), "too many init values");
    return parseParamValue(p, initVals.emplace_back(), *paramTypeIt++);
  };
  if (p.parseCommaSeparatedList(AsmParser::Delimiter::Paren, parseInit))
    return failure();
  if (paramTypeIt != paramTypeE)
    return p.emitError(p.getCurrentLocation(), "not enough init values");

  auto nextFnType = SignatureType::get(p.getContext(), paramTypes, paramTypes);
  auto condFnType = SignatureType::get(p.getContext(), paramTypes,
                                       p.getBuilder().getI1Type());
  if (p.parseComma() || parseParamValue(p, next, nextFnType) ||
      p.parseComma() || parseParamValue(p, cond, condFnType) ||
      p.parseRSquare())
    return failure();

  SmallVector<OpAsmParser::Argument> bodyArgs;
  if (succeeded(p.parseOptionalLParen())) {
    auto parseArg = [&] {
      if (p.parseArgument(bodyArgs.emplace_back()) || p.parseEqual() ||
          p.parseOperand(arguments.emplace_back()))
        return failure();
      return mlir::success();
    };
    if (p.parseCommaSeparatedList(parseArg) || p.parseRParen() ||
        p.parseArrowTypeList(argumentTypes))
      return failure();
    for (auto [idx, type] : llvm::enumerate(argumentTypes))
      bodyArgs[idx].type = type;
  }

  Region regionBody;
  Optional<Location> regionBodyLoc =
      p.getEncodedSourceLoc(p.getCurrentLocation());
  if (p.parseRegion(regionBody, bodyArgs) ||
      p.parseOptionalLocationSpecifier(regionBodyLoc))
    return failure();
  body.push_back(new Block);
  OpBuilder b(p.getContext());
  b.setInsertionPointToStart(&body.front());
  auto bodyOp = b.create<RegionOpenBodyOp>(
      *regionBodyLoc, params, ArrayRef<Type>(), ArrayRef<ConstraintAttr>());
  bodyOp.getBodyRegion().takeBody(regionBody);

  init = ParameterExprArrayAttr::get(p.getContext(), initVals);
  return success();
}

static void printIterate(OpAsmPrinter &p, Operation *op, ValueRange arguments,
                         TypeRange argumentTypes, ParameterExprArrayAttr init,
                         TypedAttr next, TypedAttr cond, Region &body) {
  p << '(';
  auto bodyOp = cast<RegionOpenBodyOp>(body.front().front());
  llvm::interleaveComma(bodyOp.getParamDecls(), p, [&](ParamDeclAttr param) {
    printParamName(p, param.getName());
    printColonTypeOrIndex(p.getStream(), param.getType());
  });
  p << ") in [(";
  llvm::interleaveComma(init, p, [&](TypedAttr initVal) {
    printParamValue(initVal, p.getStream());
  });
  p << "), ";
  printParamValue(next, p.getStream());
  p << ", ";
  printParamValue(cond, p.getStream());
  p << "] ";
  if (!arguments.empty()) {
    p << '(';
    llvm::interleaveComma(
        llvm::seq<unsigned>(0, arguments.size()), p, [&](unsigned i) {
          p << bodyOp.getBodyRegion().getArgument(i) << " = " << arguments[i];
        });
    p << ')';
    p.printArrowTypeList(argumentTypes);
    p << ' ';
  }
  p.printRegion(bodyOp.getBodyRegion(), /*printEntryBlockArgs=*/false);
  p.printOptionalLocationSpecifier(bodyOp.getLoc());
}

LogicalResult IterateOp::verifyRegions() {
  auto body = cast<RegionOpenBodyOp>(getBody().front().getTerminator());
  if (!body || body != &getBody().front().front())
    return emitOpError("expected a single open region body");

  unsigned numParams = body.getParamDecls().size();
  if (numParams != getInit().size())
    return emitOpError("expected ") << numParams << " init values";
  SmallVector<Type> paramTypes;
  for (auto [init, param] : llvm::zip(getInit(), body.getParamDecls())) {
    if (init.getType() != param.getType())
      return emitOpError("init types do not match parameter types");
    paramTypes.push_back(init.getType());
  }

  auto nextFnSig = SignatureType::get(getContext(), paramTypes, paramTypes);
  if (getNext().getType() != nextFnSig)
    return emitOpError("next function should have type ") << nextFnSig;
  auto condFnSig = SignatureType::get(getContext(), paramTypes,
                                      Builder(getContext()).getI1Type());
  if (getCond().getType() != condFnSig)
    return emitOpError("cond function should have type ") << condFnSig;

  if (body.getReturnOp().getOperandTypes() != getArguments().getTypes())
    return emitOpError("body results should match argument types");

  return success();
}

//===----------------------------------------------------------------------===//
// ExportOp
//===----------------------------------------------------------------------===//

LogicalResult ExportOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (getExports().empty())
    return emitOpError("exports must not be empty");

  // Just ensure we're exporting symbols we can see.
  auto module = KGENModule::from(*this, symbolTable);
  for (auto e : getExports().getAsRange<FlatSymbolRefAttr>()) {
    if (!module.lookup<KGENDeclInterface>(e))
      return emitOpError("could not find referenced symbol '") << e << "'";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "KGEN/KGENDialect/KGEN.cpp.inc"

// Generated interface definitions.
#include "KGEN/KGENDialect/ElaboratorOpInterface.cpp.inc"
#include "KGEN/KGENDialect/KGENCallInterface.cpp.inc"
#include "KGEN/KGENDialect/KGENDeclInterface.cpp.inc"
