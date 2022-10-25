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

// FIXME: KGENDialect should not depend on POPDialect.
#include "KGEN/POPDialect/POPTypes.h"

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
  if (failed(ParameterDeclsAndUses::calculateAndVerify(*this, symbolTable)))
    return failure();

  // If the generator is implementing a generator interface, check that they
  // line up correctly.
  FlatSymbolRefAttr interfaceSym = getImplementsAttr();
  if (!interfaceSym)
    return success();

  // Check that the callee attribute was specified.
  GeneratorInterfaceOp interface = dyn_cast_if_present<GeneratorInterfaceOp>(
      symbolTable.lookupNearestSymbolFrom(*this, interfaceSym));
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
                   FunctionType signature, LinkageAttr linkage,
                   ArrayRef<Type> outputParamTypes) {
  // Add an attribute for the name and function_type attributes.
  result.addAttribute(SymbolTable::getSymbolAttrName(), name);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(signature));
  result.addAttribute("linkage", linkage);
  result.addAttribute("paramDecls", builder.getAttr<ParamDeclArrayAttr>(
                                        ArrayRef<ParamDeclAttr>()));
  result.addAttribute("resultParamTypes",
                      builder.getAttr<TypeArrayAttr>(outputParamTypes));
  result.addRegion();
}

/// Create a func with an empty body, `argLocs` specifies the locations for
/// all the block arguments.
void FuncOp::build(OpBuilder &builder, OperationState &result, StringAttr name,
                   FunctionType signature, LinkageAttr linkage,
                   ArrayRef<Type> outputParamTypes,
                   ArrayRef<Location> argLocs) {
  build(builder, result, name, signature, linkage, outputParamTypes);

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
  if (parser.parseOptionalKeyword("evaluator"))
    return success();
  Type sigType;
  TypedAttr evaluator;
  if (parseKGENType(parser, sigType) || parser.parseEqual() ||
      parseParamValue(parser, evaluator, sigType))
    return failure();
  result.addAttribute(getEvaluatorAttrName(result.name), evaluator);
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
}

LogicalResult
GeneratorInterfaceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // See if the parameter definitions and uses within the generator are
  // structured correctly.  These are only defined in the interface and used
  // in the argument list or constraints list.
  if (failed(ParameterDeclsAndUses::calculateAndVerify(*this, symbolTable)))
    return failure();

  // If an evaluator was specified, verify its signature.
  SymbolConstantAttr evaluator = getEvaluatorAttr();
  if (!evaluator)
    return success();
  auto func = symbolTable.lookupNearestSymbolFrom<KGENDeclInterface>(
      *this, evaluator.getSymbol().getAttr());
  if (!func)
    return emitOpError("evaluator ")
           << evaluator.getSymbol() << " does not refer to a KGEN declaration";

  // Build the expected evaluator signature.
  SmallVector<ParamDeclAttr> decls;
  decls.reserve(evaluator.getParamValues().size());
  for (ParamBindAttr bind : evaluator.getParamValues())
    decls.push_back(bind.getDecl());
  auto index = IndexType::get(getContext());
  auto evaluatorType = FunctionType::get(
      getContext(), {POP::PointerType::get(getFunctionType()), index}, index);
  auto expectedSignature =
      SignatureType::get(ParamDeclArrayAttr::get(getContext(), decls),
                         TypeArrayAttr::get(getContext(), {}), evaluatorType);

  // Get the specialized callee signature.
  SignatureType funcSignature = func.getSignature().getSpecializedSignature(
      evaluator.getParamValues(), [&] { return emitError(); });
  if (!funcSignature)
    return failure();

  if (failed(verifyDeclSignaturesMatch("interface evaluator", expectedSignature,
                                       getLoc(), "referenced evaluator",
                                       funcSignature, func.getLoc())))
    return failure();

  // Make sure the evalutator is public.
  return llvm::TypeSwitch<Operation *, LogicalResult>(func.getOperation())
      .Case<FuncOp, GeneratorOp, GeneratorInterfaceOp>(
          [&](auto func) -> LogicalResult {
            if (func.getLinkage() != Linkage::Public)
              return emitOpError(
                  "expected evaluator function to have public linkage");
            return success();
          })
      .Default([&](Operation *op) {
        return emitOpError("unknown evaluator operation");
      });
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
static LogicalResult verifyRegionSignatures(Operation *theCall) {
  // TODO: We need an interface for call operations.
  auto values = theCall->getAttrOfType<ParamBindArrayAttr>("paramValues");
  assert(values && "expected parameter values");
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
    auto paramSignature = it.value().getValue().getType().cast<SignatureType>();
    Region &region = theCall->getRegion(it.index());
    auto body = cast<RegionBodyOp>(region.front().getTerminator());
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

static ParseResult
parseCallRegions(OpAsmParser &p,
                 SmallVectorImpl<std::unique_ptr<::mlir::Region>> &result,
                 ParamBindArrayAttr paramValues) {
  // We expect one region for each ParamCallRegionRefAttr.
  auto binds = llvm::make_filter_range(paramValues, [](ParamBindAttr bind) {
    return bind.getValue().isa<ParamCallRegionRefAttr>();
  });

  auto parseFn = [&](ParamBindAttr bind) -> ParseResult {
    // Parse the region body operation in-line.
    OperationState regionBody(p.getEncodedSourceLoc(p.getCurrentLocation()),
                              RegionBodyOp::getOperationName());
    Optional<Location> bodyLoc = regionBody.location;
    if (p.parseKeyword(bind.getName(), " region name") ||
        RegionBodyOp::parse(p, regionBody) ||
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

static void printCallRegions(OpAsmPrinter &p, Operation *op,
                             mlir::RegionRange regions,
                             ParamBindArrayAttr paramValues) {
  auto binds = llvm::make_filter_range(paramValues, [](ParamBindAttr bind) {
    return bind.getValue().isa<ParamCallRegionRefAttr>();
  });

  auto printFn = [&](auto &bind) {
    p.printNewline();
    p << bind.value().getName().strref();
    Operation *body = regions[bind.index()]->front().getTerminator();
    cast<RegionBodyOp>(body).print(p);
    p.printOptionalLocationSpecifier(body->getLoc());
  };
  llvm::interleave(llvm::enumerate(binds), p, printFn, ",");
}

//===----------------------------------------------------------------------===//
// AddressOfOp
//===----------------------------------------------------------------------===//

LogicalResult
AddressOfOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto callee = dyn_cast_if_present<KGENDeclInterface>(
      symbolTable.lookupNearestSymbolFrom(*this, getCalleeAttr()));
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
  return verifyRegionSignatures(*this);
}

SignatureType AddressOfOp::getSignature() {
  SmallVector<ParamDeclAttr> callerInputParamDecls;
  auto getBindDecl = [](auto bind) -> ParamDeclAttr { return bind.getDecl(); };
  llvm::append_range(callerInputParamDecls,
                     llvm::map_range(getParamValues(), getBindDecl));

  SmallVector<Type> callerResultParamTypes;
  auto getParamType = [](auto attr) -> Type { return attr.getType(); };
  llvm::append_range(callerResultParamTypes,
                     llvm::map_range(getParamDecls(), getParamType));

  return SignatureType::get(
      ParamDeclArrayAttr::get(getContext(), callerInputParamDecls),
      TypeArrayAttr::get(getContext(), callerResultParamTypes), getType());
}

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

SignatureType CallOp::getSignature() {
  SmallVector<ParamDeclAttr> callerInputParamDecls;
  auto getBindDecl = [](auto bind) -> ParamDeclAttr { return bind.getDecl(); };
  llvm::append_range(callerInputParamDecls,
                     llvm::map_range(getParamValues(), getBindDecl));

  SmallVector<Type> callerResultParamTypes;
  auto getType = [](auto attr) -> Type { return attr.getType(); };
  llvm::append_range(callerResultParamTypes,
                     llvm::map_range(getParamDecls(), getType));

  return SignatureType::get(
      ParamDeclArrayAttr::get(getContext(), callerInputParamDecls),
      TypeArrayAttr::get(getContext(), callerResultParamTypes),
      getFunctionType());
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto calleeAttr = getCalleeAttr();
  if (!calleeAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  auto callee = dyn_cast_if_present<KGENDeclInterface>(
      symbolTable.lookupNearestSymbolFrom(*this, calleeAttr));
  if (!callee)
    return emitError() << "'" << calleeAttr.getValue()
                       << "' does not reference a valid callee";

  // Check the parameters and operands align with the requirements of the
  // callee's signature.
  if (verifyCallAndCallee(*this, getSignature(), callee.getSignature(),
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
                   TypeRange resultTypes, StringAttr callee,
                   ArrayRef<ParamBindAttr> inputParams,
                   ArrayRef<ParamDeclAttr> resultParams, ValueRange operands) {
  build(builder, state, resultTypes, FlatSymbolRefAttr::get(callee),
        builder.getAttr<ParamBindArrayAttr>(inputParams),
        builder.getAttr<ParamDeclArrayAttr>(resultParams), operands,
        /*numRegions=*/0);
}

LogicalResult CallOp::verifyRegions() {
  // Verify the region signatures match region parameter signatures.
  return verifyRegionSignatures(*this);
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

SignatureType CallParamOp::getSignature() {
  // Should move this to a "getSignature" method on CallOp.
  SmallVector<ParamDeclAttr> callerInputParamDecls;
  auto getBindDecl = [](auto bind) -> ParamDeclAttr { return bind.getDecl(); };
  llvm::append_range(callerInputParamDecls,
                     llvm::map_range(getParamValues(), getBindDecl));

  SmallVector<Type> callerResultParamTypes;
  auto getType = [](auto attr) -> Type { return attr.getType(); };
  llvm::append_range(callerResultParamTypes,
                     llvm::map_range(getParamDecls(), getType));

  return SignatureType::get(
      ParamDeclArrayAttr::get(getContext(), callerInputParamDecls),
      TypeArrayAttr::get(getContext(), callerResultParamTypes),
      getFunctionType());
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

  auto calleeSignature = dyn_cast<SignatureType>(getCallee().getType());
  if (!calleeSignature)
    return emitError("kgen.call_param requires callee of signature type");

  // Check the parameters and operands align with the requirements of the
  // callee's signature.
  if (failed(verifyCallAndCallee(*this, getSignature(), calleeSignature,
                                 getParamValuesAttr(), getLoc())))
    return failure();

  // Verify the region signatures match region parameter signatures.
  return verifyRegionSignatures(*this);
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

/// Verify the parameters in the region body.
LogicalResult
RegionBodyOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (failed(getReturnOp().checkArgumentTypes(getResultParamTypes(), None)))
    return failure();

  // Verify the parameter definitions and uses within the region body.
  return ParameterDeclsAndUses::calculateAndVerify(*this, symbolTable);
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

  // Verify our result types match up with the enclosing result type.
  if (getNumOperands() != types->size())
    return emitOpError("expected ")
           << types->size() << " operands for enclosing op";

  for (size_t i = 0, e = getNumOperands(); i != e; ++i) {
    if (getOperand(i).getType() != (*types)[i])
      return emitOpError("operand #")
             << i << " has type " << getOperand(i).getType()
             << " but should be " << (*types)[i];
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

/// The single region is only allowed to contain `struct.field` ops. Verify that
/// there are no duplicate field names.
LogicalResult StructDeclOp::verifyRegions() {
  SmallDenseMap<StringAttr, StructFieldOp, 8> seenFields;
  for (Operation &op : getFields().front()) {
    auto field = dyn_cast<StructFieldOp>(&op);
    if (!field) {
      return emitOpError("expected only `kgen.struct.field` ops in its body")
                 .attachNote(op.getLoc())
             << "invalid child op here";
    }
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
  return ParameterDeclsAndUses::calculateAndVerify(*this, symbolTable);
}

/// Parse a special syntax for the struct fields.
/// field ::= identifier `:` type
static ParseResult parseStructFields(OpAsmParser &p, Region &fields) {
  if (p.parseLBrace())
    return failure();
  Block *body = new Block;
  fields.push_back(body);
  OpBuilder b(p.getContext());
  b.setInsertionPointToStart(body);
  while (p.parseOptionalRBrace()) {
    OperationState field(p.getEncodedSourceLoc(p.getCurrentLocation()),
                         StructFieldOp::getOperationName());
    if (StructFieldOp::parse(p, field))
      return failure();

    Optional<Location> fieldLoc = field.location;
    if (p.parseOptionalLocationSpecifier(fieldLoc))
      return failure();
    field.location = *fieldLoc;

    b.create(field);
  }
  return success();
}

static void printStructFields(OpAsmPrinter &p, Operation *op, Region &fields) {
  p << '{';
  p.printNewline();
  for (Operation &field : fields.front()) {
    p << "  ";
    cast<StructFieldOp>(field).print(p);
    p.printOptionalLocationSpecifier(field.getLoc());
    p.printNewline();
  }
  p << '}';
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
                 RefType ref) {
  FlatSymbolRefAttr name = ref.getName();
  auto structDecl =
      symbolTable.lookupNearestSymbolFrom<StructDeclOp>(user, name.getAttr());
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
         << getType().getName() << " has no field named " << getFieldAttr();
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

static LogicalResult
verifyStructFieldAndType(SymbolTableCollection &symbolTable, Operation *op,
                         RefType ref, StringAttr fieldName, Type type) {
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
         << ref.getName() << " has no field named " << fieldName;
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
      cast<RefType>(cast<TypeConstantAttr>(refExpr).getValue()), getFieldAttr(),
      ParamRefType::get(getResult().getType().getElementType()));
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
  return parseGeneratorOrFunc(parser, result, GeneratorOrFuncKind::precompiled);
}

/// Print a `kgen.precompiled.*` op. They all have almost exactly the same form
/// so we use a single function to handle them all. See `parsePrecompiledOp` for
/// an example of the form we want printed.
static void printPrecompiledOp(OpAsmPrinter &p, Operation *op) {
  auto funcOp = cast<mlir::FunctionOpInterface>(op);
  printGeneratorOrFunc(p, funcOp);
}

//===----------------------------------------------------------------------===//
// PrecompiledLLVMOp
//===----------------------------------------------------------------------===//

void PrecompiledLLVMOp::build(OpBuilder &builder, OperationState &result,
                              FuncOp func, TargetInfoAttr compiledFor,
                              StringRef llvm) {
  build(builder, result, func.getSymNameAttr(), func.getFunctionTypeAttr(),
        func.getLinkageAttr(), func.getParamDeclsAttr(),
        func.getResultParamTypesAttr(), compiledFor,
        builder.getStringAttr(llvm));
}

//===----------------------------------------------------------------------===//
// PrecompiledObjectOp
//===----------------------------------------------------------------------===//

void PrecompiledObjectOp::build(OpBuilder &builder, OperationState &result,
                                PrecompiledLLVMOp func, StringRef object) {
  build(builder, result, func.getSymNameAttr(), func.getFunctionTypeAttr(),
        func.getLinkageAttr(), func.getParamDeclsAttr(),
        func.getResultParamTypesAttr(), func.getCompiledForAttr(),
        builder.getStringAttr(object));
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
