//===- KernelElaborator.cpp - Core kernel elaborator algorithm ------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains logic to lower a file full of kernel into concrete
// implementations of the kernels.
//
//===----------------------------------------------------------------------===//

#include "Internals.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "Support/DType.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include <variant>

using namespace M;
using namespace KGEN;

/// We expect all parameter expressions to simplify down to concrete constants,
/// we don't want anything left as a ParamOperatorAttr or ParamDeclRefAttr.
static bool isSimpleConstant(Attribute attr) {
  return attr.isa<FloatAttr, IntegerAttr, DTypeConstantAttr>();
}

//===----------------------------------------------------------------------===//
// ElaborationDiagnostic class definition
//===----------------------------------------------------------------------===//

class ElaborationDiagnostic;

/// This class models the location of a declaration that failed to expand along
/// with the reason why it failed.
using CalleeExpansionError = std::pair<Location, ElaborationDiagnostic>;

/// This class represents the reason that a kernel or generator could not be
/// elaborated.  It is either a local problem in the kernel (e.g. an operator
/// defining a parameter that is unknown) or it is a call to another set of
/// kernel/generators that had problems expanding.
class ElaborationDiagnostic {
public:
  ElaborationDiagnostic(Location loc, Error error)
      : loc(loc), payload(std::string(error.get())) {}
  ElaborationDiagnostic(Location loc,
                        ArrayRef<CalleeExpansionError> calleeErrors)
      : loc(loc), payload(calleeErrors.vec()) {}

private:
  /// This is the location within the declaration of the failure, e.g. of the
  /// call or other operator with a problem.
  Location loc;

  /// The problem is either:
  ///   1) a local issue represented as an error on an operation.
  ///   2) a transitive issue where expansion of a call failed (the main
  ///      location) due to callees where something inside the callee failed to
  ///      expand.  Each callee has a location of the decl itself + the problem.
  std::variant<std::string, std::vector<CalleeExpansionError>> payload;
};

using KernelOrCalleeError = std::variant<KernelOp, CalleeExpansionError>;

//===----------------------------------------------------------------------===//
// Elaborator class definition
//===----------------------------------------------------------------------===//

namespace {
class Elaborator {
public:
  Elaborator(ModuleOp primary, ModuleOp library)
      : primaryModule(primary), libraryModule(library), symbolTable(primary) {}

  /// Scan the primary and library module to collect all the interfaces,
  /// verifying that any common interfaces are the same.
  ParseResult collectInterfaces();

  /// Return the operation that defines the specified symbol.
  Operation *lookupCallee(StringAttr symbolName) const {
    return symbolTable.lookup(symbolName);
  }

  /// Return all instantiations of the specified declaration (a kernel,
  /// generator, or interface) with teh specified input parameter values.
  /// `insertionPoint` is always a point in the primary module where a new
  /// kernel should be placed if necessary.
  ArrayRef<KernelOrCalleeError>
  getAllInstantiations(Operation *decl, ArrayRef<Attribute> inputParamValues,
                       Operation *insertionPoint);

  /// Insert a variant of an existing kernel into the primary file.
  void insertKernelVariant(KernelOp existing, KernelOp newKernel);

private:
  /// Specialize a kernel body, generating one variant or each viable
  /// instantiation of that body.  Kernels do not have parameters, but they can
  /// invoke interfaces etc which can cause them to produce multiple variants.
  SmallVector<KernelOrCalleeError> specializeKernel(KernelOp kernel);

  /// Specialize a kernel generator with the specified input parameters and
  /// return the generated kernel.  `insertionPoint` is always a point in the
  /// primary module where a new kernel should be placed if necessary.
  SmallVector<KernelOrCalleeError>
  specializeGenerator(GeneratorOp generator,
                      ArrayRef<Attribute> inputParamValues,
                      Operation *insertionPoint);

  /// Specialize a kernel interface with the specified input parameters and
  /// return the generated kernel.  `insertionPoint` is always a point in the
  /// primary module where a new kernel should be placed if necessary.
  SmallVector<KernelOrCalleeError>
  specializeInterface(GeneratorInterfaceOp itf,
                      ArrayRef<Attribute> inputParamValues,
                      Operation *insertionPoint);

  ArrayRef<GeneratorOp> getGeneratorsImplementing(GeneratorInterfaceOp itf) {
    auto it = interfaceImpls.find(itf.getNameAttr());
    return it == interfaceImpls.end() ? ArrayRef<GeneratorOp>() : it->second;
  }

private:
  /// These are the two modules we start with.  The primary module is mutated by
  /// our algorithm, the library module is immutable.
  ModuleOp primaryModule, libraryModule;

  /// This symbol table allows efficient lookups in the primary module.
  SymbolTable symbolTable;

  /// This collects all of the generator implementations of generator
  /// interfaces, across both the primary module and the library.
  DenseMap<StringAttr, SmallVector<GeneratorOp, 4>> interfaceImpls;

  /// This is a cache of already-instantiated declarations.  The key is the
  /// kernel/generator/interface and input parameters, the result are
  /// all-possible kernels that could be generated from this.
  DenseMap<std::pair<Operation *, ArrayAttr>, SmallVector<KernelOrCalleeError>>
      generatedKernels;
};
} // end anonymous namespace

/// Insert a variant of an existing kernel into the primary file.
void Elaborator::insertKernelVariant(KernelOp existing, KernelOp newKernel) {
  auto insertPt = Block::iterator(existing.getOperation());
  symbolTable.insert(newKernel, /*insertionPoint*/ ++insertPt);
}

//===----------------------------------------------------------------------===//
// collectInterfaces and cleanup helpers
//===----------------------------------------------------------------------===//

/// Scan the primary and library module to collect all the interfaces,
/// verifying that any common interfaces are the same.
ParseResult Elaborator::collectInterfaces() {
  // Collect all the generator interfaces in the library module, which will
  // allow cross checking them below.
  DenseMap<StringAttr, GeneratorInterfaceOp> libraryInterfaces;
  for (auto itf : libraryModule.getOps<GeneratorInterfaceOp>())
    libraryInterfaces[itf.getNameAttr()] = itf;

  // Collect all the kernel generators that implement a given interface,
  // starting with the library.  These will already have been type checked
  // within the library.
  for (auto generator : libraryModule.getOps<GeneratorOp>()) {
    if (auto interface = generator.getImplementsAttr())
      interfaceImpls[interface.getAttr()].push_back(generator);
  }

  // Collect the kernel generators from the primary module.  Start by checking
  // that any generator implementations that exist in both modules match in
  // signature exactly.
  for (auto itf : primaryModule.getOps<GeneratorInterfaceOp>()) {
    auto it = libraryInterfaces.find(itf.getNameAttr());
    if (it == libraryInterfaces.end())
      continue;
    if (failed(verifyDeclMatchesInterface("interface", itf, "library interface",
                                          it->second)))
      return failure();
  }

  // If they all match up, collect the generator implementations from the
  // primary module.
  for (auto generator : primaryModule.getOps<GeneratorOp>())
    if (auto interface = generator.getImplementsAttr())
      interfaceImpls[interface.getAttr()].push_back(generator);

  return success();
}

//===----------------------------------------------------------------------===//
// Elaborator Algorithm for one Kernel
//===----------------------------------------------------------------------===//

namespace {
/// This class keeps a set of defined parameter values and is used to evaluate
/// and simplify parameter expressions based on those values.
class ParameterEvaluator {
public:
  ParameterEvaluator() = default;
  ParameterEvaluator(const ParameterEvaluator &) = default;

  /// Given a generator or interface declaration operation, evaluate any
  /// constraints against inputParamValues.  If the constraints are met, return
  /// success, otherwise return why they aren't.
  static ErrorOrSuccess
  evaluateConstraints(Operation *decl, ArrayRef<Attribute> inputParamValues);

  /// Set a value for the specified parameter declaration to the specified
  /// simplified value.
  void setParameterValue(ParamDeclAttr decl, Attribute value) {
    assert(!paramValues.count(decl.getName()) && "parameter already declared!");
    assert(isSimpleConstant(value) && "expression isn't simplified");
    paramValues[decl.getName()] = value;
  }

  /// Given a generic parameter expression, simplify it by folding the
  /// expression according to known parameter values.  This returns an error if
  /// the expression cannot be folded for one reason or another.
  ErrorOr<Attribute> simplifyParameterExpr(Attribute expr);

private:
  /// These are the bound parameter values, captured in simplified form.
  DenseMap<StringAttr, Attribute> paramValues;
};
} // end anonymous namespace

static std::string getString(Attribute attr) {
  std::string str;
  llvm::raw_string_ostream(str) << attr;
  return str;
}

/// Given a generic parameter expression, simplify it by folding the
/// expression according to known parameter values.  This returns an error if
/// the expression cannot be folded for one reason or another.
ErrorOr<Attribute> ParameterEvaluator::simplifyParameterExpr(Attribute expr) {
  // Simple constants don't need simplification.
  if (isSimpleConstant(expr))
    return expr;

  // We can directly substitute declaration references given our known table of
  // bindings.
  if (auto declRef = expr.dyn_cast<ParamDeclRefAttr>()) {
    auto value = paramValues[declRef.getName()];
    assert(value && "Verifier should check that all parameters are defined");
    return value;
  }

  // Simplify operators by recursively simplifying their operands, then
  // refolding the expression.
  if (auto oper = expr.dyn_cast<ParamOperatorAttr>()) {
    SmallVector<Attribute> simplifiedOperands;
    for (auto value : oper.getOperands()) {
      auto simplified = simplifyParameterExpr(value);
      if (simplified.isError())
        return simplified;
      simplifiedOperands.push_back(simplified.takeValue());
    }

    // FIXME: 'index' folding should require target information to simplify
    // things like div.
    auto result = ParamOperatorAttr::get(oper.getOpcode(), simplifiedOperands);
    if (!isSimpleConstant(result))
      return Error("could not simplify operator " + getString(expr));
    return result;
  }

  // Otherwise, we don't know how to simplify this attribute, it's an error.
  return Error("unknown expression to fold: " + getString(expr));
}

/// Given a generator or interface declaration operation, evaluate any
/// constraints against inputParamValues.  If the constraints are met, return
/// success, otherwise return why they aren't.
ErrorOrSuccess
ParameterEvaluator::evaluateConstraints(Operation *decl,
                                        ArrayRef<Attribute> inputParamValues) {
  // If there are no constraints, we are trivially done.
  auto constraints = getDeclConstraints(decl);
  if (constraints.empty())
    return success();

  // Otherwise, we have constraints to evaluate.  Bind each of the input
  // parameter names.
  ParameterEvaluator evaluator;
  auto inputParamDecls = getDeclParameterInfo(decl).first;
  assert(inputParamDecls.size() == inputParamValues.size() &&
         "incorrect number of input parameters");
  for (auto [decl, value] : llvm::zip(inputParamDecls, inputParamValues))
    evaluator.setParameterValue(decl.cast<ParamDeclAttr>(), value);

  // Each constraint must be foldable, and must fold to true.
  for (auto constraint : constraints) {
    ErrorOr<Attribute> result = evaluator.simplifyParameterExpr(constraint);
    if (failed(result))
      return Error("constraint evaluation failure: " +
                   Twine(result.getError()));
    auto resultInt = (*result).dyn_cast<IntegerAttr>();
    if (!resultInt || resultInt.getValue().getBitWidth() != 1)
      return Error("constraint evaluation didn't return true or false");

    // TODO: This isn't a very pretty printing of why the constraint failed.
    if (resultInt.getValue().isZero())
      return Error("constraint failed: " + getString(constraint));
  }

  // If we made it this far, then everything folded to true.
  return success();
}

namespace {
/// This class keeps a set of defined parameter values and is used to evaluate
/// and simplify operations in a kernel based on those values.  If an error
/// happens during rewriting, the diagnostic is filled in and failure() is
/// returned.
class ParameterRewriter : public ParameterEvaluator {
public:
  ParameterRewriter(Elaborator &elaborator, KernelOp kernel,
                    SmallVector<Operation *> opsToRewrite)
      : elaborator(elaborator), kernel(kernel),
        opsToRewrite(std::move(opsToRewrite)) {}

  /// Create a clone of this rewriter, but refer with a clone of the kernel.
  /// This uses operationMap to remap our state onto the newly created kernel.
  ParameterRewriter(const ParameterRewriter &existing,
                    DenseMap<Operation *, Operation *> &operationMap);

  /// Process all the `opsToRewrite`, simplifying this kernel.  If new variants
  /// of this kernel are necessary, they are added to rewriterWorklist.
  LogicalResult
  rewriteOps(SmallVectorImpl<ParameterRewriter> &rewriterWorklist);

  /// Return the kernel we're generating into.
  KernelOp getKernel() const {
    assert(!diagnostic.hasValue() &&
           "can't get the result kernel when a diagnostic was generated");
    return kernel;
  }

  CalleeExpansionError takeDiagnostic() {
    assert(diagnostic.hasValue() &&
           "cannot get diagnostic when none was generated");
    return CalleeExpansionError(kernel->getLoc(),
                                std::move(diagnostic.getValue()));
  }

  /// Generate a error expanding this kernel.  The location specified is the
  /// operation with the problem, and the message is the problem with it.
  LogicalResult error(Location loc, Error message) {
    assert(!diagnostic.hasValue() && "Already emitted an error");
    diagnostic = ElaborationDiagnostic(loc, std::move(message));
    return failure();
  }

  /// Generate an error expanding this kernel for a call expansion problem.  The
  /// location specified is for the call.  Each entry in calleeErrors includes
  /// the location of the declaration that failed to expand along with why it
  /// failed.
  LogicalResult errorCalling(Location callLoc,
                             ArrayRef<CalleeExpansionError> calleeErrors) {
    assert(!diagnostic.hasValue() && "Already emitted an error");
    diagnostic = ElaborationDiagnostic(callLoc, calleeErrors);
    return failure();
  }

private:
  LogicalResult processParamBindOp(ParamBindOp op);
  LogicalResult processParamValueOp(ParamValueOp op);
  LogicalResult processCallOp(CallOp call,
                              SmallVectorImpl<ParameterRewriter> &rewriters);
  void completeCallOpProcessing(CallOp call, KernelOp newCallee);
  void spawnNewKernelClone(CallOp call, KernelOp callee,
                           SmallVectorImpl<ParameterRewriter> &rewriters);
  LogicalResult processGenericOp(Operation *op);

  /// Get the specified attribute with any nested parameter expressions
  /// rewritten.
  Attribute getReboundAttribute(Attribute attr, Location loc);

  /// Get the specified type with any nested parameter expressions rewritten.
  Type getReboundType(Type type, Location loc);

  // This is maintains global information about the file we're generating into.
  Elaborator &elaborator;

  /// This is the kernel we're working on.
  KernelOp kernel;

  /// This is a diagnostic explaining the expansion failure if something goes
  /// wrong.
  Optional<ElaborationDiagnostic> diagnostic;

  /// These are the operations we still need to visit to complete our rewrite.
  SmallVector<Operation *> opsToRewrite;

  /// This caches attributes and Types with parameter references rebound, and
  /// remembers complex attributes that don't have parameter subexprs (noted as
  /// being rebound to themselves).
  DenseMap<Attribute, Attribute> rewrittenAttrs;
  DenseMap<Type, Type> rewrittenTypes;
};
} // end anonymous namespace

/// Create a clone of this rewriter, but refer with a clone of the kernel.
/// This uses operationMap to remap our state onto the newly created kernel.
ParameterRewriter::ParameterRewriter(
    const ParameterRewriter &existing,
    DenseMap<Operation *, Operation *> &operationMap)
    : ParameterEvaluator(existing), elaborator(existing.elaborator),
      rewrittenAttrs(existing.rewrittenAttrs),
      rewrittenTypes(existing.rewrittenTypes) {
  // Remap the kernel itself.
  kernel = cast<KernelOp>(operationMap[existing.kernel]);
  assert(kernel && "didn't remap kernel correctly");

  // Remap the operation worklist.
  opsToRewrite.reserve(existing.opsToRewrite.size());
  for (Operation *op : existing.opsToRewrite) {
    opsToRewrite.push_back(operationMap[op]);
    assert(opsToRewrite.back() && "didn't clone operation correctly?");
  }
}

/// Work the `opsToRewrite` worklist.
LogicalResult ParameterRewriter::rewriteOps(
    SmallVectorImpl<ParameterRewriter> &rewriterWorklist) {
  /// We use a worklist for this so cloned versions of ParameterRewriter can
  /// be created and known where to pick up from.
  while (!opsToRewrite.empty()) {
    Operation *op = opsToRewrite.pop_back_val();

    LogicalResult result = success();
    /// Process an operation that needs to be rewritten/lowered based on the
    /// context of the parameter values we know are defined.
    if (auto bind = dyn_cast<ParamBindOp>(op))
      result = processParamBindOp(bind);
    else if (auto value = dyn_cast<ParamValueOp>(op))
      result = processParamValueOp(value);
    else if (auto call = dyn_cast<CallOp>(op))
      result = processCallOp(call, rewriterWorklist);
    else if (isa<KernelOp>(op))
      /*kernels can define parameters, nothing need be done with them*/;
    else
      result = processGenericOp(op);

    // If processing any operation failed, then this entire kernel elaboration
    // failed.
    if (failed(result))
      return failure();
  }

  // Check that the thing we just built is correct IR!  We want to catch any
  // errors produced by the verify pass, we don't want them to actually get
  // emitted.
  bool hadError = false;
  mlir::ScopedDiagnosticHandler diagHandler(
      kernel.getContext(), [&](Diagnostic &diag) -> LogicalResult {
        (void)error(diag.getLocation(),
                    Twine("verification error: ") + diag.str());
        hadError = true;
        return success();
      });

  LogicalResult verifyResult = verify(kernel);
  assert(hadError == failed(verifyResult) && "Result of verify is unexpected");
  return verifyResult;
}

LogicalResult ParameterRewriter::processParamBindOp(ParamBindOp op) {
  // Simplify the input expression.
  auto errorOrValue = simplifyParameterExpr(op.getValue());
  if (errorOrValue.isError())
    return error(op->getLoc(), errorOrValue.takeError());

  // Bind it to the parameter declaration it is setting.
  setParameterValue(op.getParamDecl(), errorOrValue.takeValue());

  // The param.bind operation serves no other purpose, so we can remove it.
  op->erase();
  return success();
}

LogicalResult ParameterRewriter::processParamValueOp(ParamValueOp op) {
  // ParamValueOp projects a parameter expression into an SSA value.  We can
  // eventually lower this into lower level operators in the target set, but
  // for now we just simplify their operand.
  auto errorOrValue = simplifyParameterExpr(op.getValue());
  if (errorOrValue.isError())
    return error(op->getLoc(), errorOrValue.takeError());

  op.setValueAttr(errorOrValue.takeValue());
  return success();
}

LogicalResult ParameterRewriter::processCallOp(
    CallOp call, SmallVectorImpl<ParameterRewriter> &rewriters) {
  // Evaluating any input parameters.
  SmallVector<Attribute> boundInputParams;
  for (auto param : call.getParamValues()) {
    auto value = simplifyParameterExpr(param.cast<ParamBindAttr>().getValue());
    if (value.isError())
      return error(call->getLoc(), value.takeError());

    boundInputParams.push_back(value.takeValue());
  }

  // Instantiate the callee into one or more KernelOp's, depending on what the
  // callee is.
  auto callee = elaborator.lookupCallee(call.getCalleeAttr().getAttr());
  ArrayRef<KernelOrCalleeError> newCalleesRef =
      elaborator.getAllInstantiations(callee, boundInputParams, kernel);

  // Copy the list of kernels instead of referring to the cache entry to avoid
  // iterator invalidation problems.
  SmallVector<KernelOrCalleeError> newCallees(newCalleesRef.begin(),
                                              newCalleesRef.end());

  // If we found more than one callee to produce then we need to spawn
  // multiple versions of the kernel we are currently constructing, each
  // which get a different callee.
  KernelOp thisCallee;
  for (const KernelOrCalleeError &callee : newCallees) {
    // Ignore erroneous callees.
    if (std::holds_alternative<CalleeExpansionError>(callee))
      continue;
    // We will pursue the first viable callee locally.
    if (!thisCallee)
      thisCallee = std::get<KernelOp>(callee);
    else
      /// All other callees gets spawned as sub-evaluators.
      spawnNewKernelClone(call, std::get<KernelOp>(callee), rewriters);
  }

  // If all the expansions failed, then this call fails overall.
  if (!thisCallee) {
    SmallVector<CalleeExpansionError> errors;
    for (const auto &value : newCalleesRef)
      errors.push_back(std::get<CalleeExpansionError>(value));
    return errorCalling(call->getLoc(), errors);
  }

  // Finally, we can handle the first viable one as our continued progress here.
  completeCallOpProcessing(call, thisCallee);
  return success();
}

void ParameterRewriter::completeCallOpProcessing(CallOp call,
                                                 KernelOp newCallee) {
  // If we resolved the call to a new thing, build a new call to replace the old
  // one.
  OpBuilder b(call);
  auto newCall = b.create<CallOp>(
      call.getLoc(), call.getResultTypes(), newCallee.getNameAttr(),
      /*input params*/ ArrayRef<Attribute>(),
      /*output params*/ call.getParamDecls().getValue(), call.getOperands());

  // The SSA results of the old call go directly to the new call and remove it.
  call->getResults().replaceAllUsesWith(newCall);
  call->erase();

  // Bind the result parameters to the output parameter decls.
  for (auto [decl, bindValue] : llvm::zip(
           newCall.getParamDecls(), newCallee.getReturnOp().getParameters()))
    setParameterValue(decl.cast<ParamDeclAttr>(),
                      bindValue.cast<ParamBindAttr>().getValue());
}

/// Sometimes when we expand a call, we find that there are multiple viable
/// callees that we can generate.  We handle this by spawning new parameter
/// rewriters with state copied from the current one, but which resolve the call
/// to different callees.  This spawns a new rewriter with the specified call
/// resolving to the specified callee.
void ParameterRewriter::spawnNewKernelClone(
    CallOp call, KernelOp callee,
    SmallVectorImpl<ParameterRewriter> &rewriters) {

  // Start by cloning the current WIP kernel to a new copy of it.
  BlockAndValueMapping blocksAndValues;
  DenseMap<Operation *, Operation *> operationMap;
  auto newKernel =
      cast<KernelOp>(cloneOperation(kernel, blocksAndValues, operationMap));

  // Insert the kernel into the output file and auto-unique the symbol.
  elaborator.insertKernelVariant(kernel, newKernel);

  // Generate the new rewriter which will process this.
  auto &newRewriter = rewriters.emplace_back(*this, operationMap);

  // Change the future of this kernel by resolving the call in the new kernel to
  // the specifed callee.
  auto newCall = cast<CallOp>(operationMap[call]);
  newRewriter.completeCallOpProcessing(newCall, callee);
}

/// Get the specified attribute with any nested parameter expressions
/// rewritten.
Attribute ParameterRewriter::getReboundAttribute(Attribute attr, Location loc) {
  // These are common leaf attributes that we know are never parameterized.
  if (attr.isa<IntegerAttr, FloatAttr, StringAttr, SymbolRefAttr,
               DTypeConstantAttr>())
    return attr;

  // If we've already processed this attribute, just reuse the memoized result.
  auto iter = rewrittenAttrs.find(attr);
  if (iter != rewrittenAttrs.end())
    return iter->second;

  // TODO(jeff): MLIR attribute should not carry types!
  if (getReboundType(attr.getType(), loc) != attr.getType()) {
    emitError(loc, "unsupported parameterized type in attribute ") << attr;
    return rewrittenAttrs[attr] = attr;
  }

  // If this is a foldable parameter expression, do it.
  Attribute result = attr;
  if (attr.isa<ParamDeclRefAttr, ParamOperatorAttr>()) {
    auto newVal = simplifyParameterExpr(attr);
    if (!newVal.isError())
      result = newVal.takeValue();

  } else if (auto itf = attr.dyn_cast<mlir::SubElementAttrInterface>()) {
    SmallVector<std::pair<size_t, Attribute>> newAttrs;
    bool changedType = false;
    size_t attrNo = 0;
    itf.walkImmediateSubElements(
        [&](Attribute attr) {
          auto newAttr = getReboundAttribute(attr, loc);
          if (newAttr != attr)
            newAttrs.push_back(std::make_pair(attrNo, newAttr));
          ++attrNo;
        },
        [&](Type type) { changedType = type != getReboundType(type, loc); });
    if (changedType) {
      // TODO: Improve SubElementTypeInterface:
      // https://github.com/llvm/llvm-project/issues/56355
      emitError(loc, "don't know how to rebind parameterized subtypes in ")
          << attr;
    } else if (!newAttrs.empty()) {
      result = itf.replaceImmediateSubAttribute(newAttrs);
    }
  } else {
    emitError(loc, "unknown attribute in parameterized operation ") << attr;
  }

  return rewrittenAttrs[attr] = result;
}

/// Get the specified type with any nested parameter expressions rewritten.
Type ParameterRewriter::getReboundType(Type type, Location loc) {
  // These are known leaf types that don't participate with
  // SubElementTypeInterface and have no attributes or types within them.
  if (type.isa<IntegerType, FloatType, NoneType, IndexType, DTypeType>())
    return type;

  // If we've already processed this type, just reuse the memoized result.
  auto iter = rewrittenTypes.find(type);
  if (iter != rewrittenTypes.end())
    return iter->second;

  Type result = type;
  if (auto itf = type.dyn_cast<mlir::SubElementTypeInterface>()) {
    SmallVector<std::pair<size_t, Attribute>> newAttrs;
    bool changedType = false;
    size_t attrNo = 0;
    itf.walkImmediateSubElements(
        [&](Attribute attr) {
          auto newAttr = getReboundAttribute(attr, loc);
          if (newAttr != attr)
            newAttrs.push_back(std::make_pair(attrNo, newAttr));
          ++attrNo;
        },
        [&](Type type) { changedType = type != getReboundType(type, loc); });
    if (changedType) {
      // TODO: Improve SubElementTypeInterface:
      // https://github.com/llvm/llvm-project/issues/56355
      emitError(loc, "don't know how to rebind parameterized subtypes in ")
          << type;
    } else if (!newAttrs.empty()) {
      result = itf.replaceImmediateSubAttribute(newAttrs);
    }
  } else {
    emitError(loc, "unknown type in parameterized operation ") << type;
  }

  return rewrittenTypes[type] = result;
}

/// Unknown operations are allowed to use types and attributes with parameter
/// references.  Substitute in concrete values for their references.
LogicalResult ParameterRewriter::processGenericOp(Operation *op) {
  // We can rewrite generic references and /uses/ of parameters, but we don't
  // don't know how to calculate the new value for a defined parameter.  If
  // there is a reason to allow open extension of operations that define
  // parameters, we could genericize this into a op interface.
  if (!getParamDecls(op).empty())
    return error(op->getLoc(),
                 "unknown parameter-defining operator in GenerateKernels");

  // Scan all the attributes and types to look for uses of parameters.  We let
  // the walker scan the region hierarchy.
  SmallVector<NamedAttribute> newAttrs;
  bool changedAttrs = false;
  for (const NamedAttribute &namedAttr : op->getAttrs()) {
    newAttrs.push_back(NamedAttribute(
        namedAttr.getName(),
        getReboundAttribute(namedAttr.getValue(), op->getLoc())));
    changedAttrs |= namedAttr.getValue() != newAttrs.back().getValue();
  }
  if (changedAttrs)
    op->setAttrs(newAttrs);

  // Check the types of results to find any parameters embedded in their
  // types.  We don't have to check operands because they are always checked
  // when being defined.
  for (OpResult result : op->getResults())
    result.setType(getReboundType(result.getType(), op->getLoc()));

  // Scan the region list if present.  The walker will automatically recurse
  // for us, but we have to check the block arguments.
  if (op->getNumRegions()) { // Microoptimization: getRegions() is slow.
    for (auto &region : op->getRegions())
      for (auto &block : region)
        for (Value arg : block.getArguments())
          arg.setType(getReboundType(arg.getType(), op->getLoc()));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Elaborator::getAllInstantiations
//===----------------------------------------------------------------------===//

/// This returns a name to use when the specified generator is specialized
/// with the specified input parameters.
static StringAttr mangleParameterValues(GeneratorOp generator,
                                        ArrayRef<Attribute> inputParamValues) {
  Builder b(generator.getContext());
  if (inputParamValues.empty())
    return b.getStringAttr(generator.getName() + "_kernel");

  std::string result;
  llvm::raw_string_ostream os(result);
  os << generator.getName();

  auto inputParamDecls = generator.getParameterInfo().first;
  for (auto [inputDecl, value] : llvm::zip(inputParamDecls, inputParamValues)) {
    os << ',' << inputDecl.cast<ParamDeclAttr>().getName().str() << '=';

    if (auto intAttr = value.dyn_cast<IntegerAttr>()) {
      os << intAttr.getValue();
    } else if (auto floatAttr = value.dyn_cast<FloatAttr>()) {
      SmallString<32> str;
      floatAttr.getValue().toString(str);
      os << str;
    } else if (auto dtypeAttr = value.dyn_cast<DTypeConstantAttr>()) {
      os << dtypeAttr.getDType();
    } else {
      assert(!isSimpleConstant(value) && "not handling all simple constants");
      os << "??";
    }
  }
  return b.getStringAttr(result);
}

/// Specialize a kernel body, generating one variant or each viable
/// instantiation of that body.  Kernels do not have parameters, but they can
/// invoke interfaces etc which can cause them to produce multiple variants.
SmallVector<KernelOrCalleeError> Elaborator::specializeKernel(KernelOp kernel) {
  /// Get a partial ordering of parameter definitions and uses that is listed
  /// "top down" in our evaluation order.
  SmallVector<Operation *> opsToRewrite;
  {
    auto paramInfo = ParameterDeclsAndUses::calculate(kernel);
    if (failed(paramInfo)) {
      kernel->emitError("verification error for kernel");
      return {};
    }
    opsToRewrite = paramInfo->getUsingAndDeclaringOps();
  }

  // We are going to use opsToRewrite as a worklist, so reverse it for efficient
  // pop_back.
  std::reverse(opsToRewrite.begin(), opsToRewrite.end());

  // Start by rewriting this kernel.
  SmallVector<ParameterRewriter, 2> rewriterWorklist;
  rewriterWorklist.emplace_back(*this, kernel, std::move(opsToRewrite));

  // Rewriting kernels may generate other kernel clones.  If so, rewrite them,
  // until we converge.
  SmallVector<KernelOrCalleeError> results;
  while (!rewriterWorklist.empty()) {
    auto rewriter = rewriterWorklist.pop_back_val();
    if (succeeded(rewriter.rewriteOps(rewriterWorklist)))
      results.push_back(rewriter.getKernel());
    else
      results.push_back(rewriter.takeDiagnostic());
  }
  return results;
}

/// Specialize a kernel generator with the specified input parameters and
/// return the symbol name to use for the result, along with an array of
/// ParamBindAttrs for the result attributes.  `insertionPoint` is always a
/// point in the primary module where a new kernel should be placed if
/// necessary.
SmallVector<KernelOrCalleeError>
Elaborator::specializeGenerator(GeneratorOp generator,
                                ArrayRef<Attribute> inputParamValues,
                                Operation *insertionPoint) {
  // We insert specializations of the generator immediately before the generator
  // if it is defined in the primary module.  Otherwise if it is from the
  // library, it would be better to insert it before the first client that
  // needed it (make tests easier to write).
  if (generator->getParentOp() == primaryModule) {
    insertionPoint = generator;
  } else {
    assert(insertionPoint && insertionPoint->getParentOp() == primaryModule);
  }

  auto [inputParamDecls, resultParamDecls] = generator.getParameterInfo();
  assert(inputParamValues.size() == inputParamDecls.size() &&
         "incorrect # input parameter values");

  // TODO (low prio): Some day we could mangle "instantiated from here"
  // information into the location.
  OpBuilder b(insertionPoint);
  auto newKernel = b.create<KernelOp>(
      generator.getLoc(), mangleParameterValues(generator, inputParamValues),
      generator.getFunctionType(), resultParamDecls);

  // Insert the newKernel into the symbol table which will then know about it,
  // but it will also auto-rename the symbol for us in the case of conflicts.
  symbolTable.insert(newKernel);

  // Clone the body of the generator over.
  BlockAndValueMapping mapper;
  generator.getBody().cloneInto(&newKernel.getBody(), mapper);

  // Provide definitions of the input parameters in the body block as bound
  // constants.
  b.setInsertionPoint(&newKernel.getBodyBlock()->front());
  for (auto [inputDecl, inputValue] :
       llvm::zip(inputParamDecls, inputParamValues)) {
    b.create<ParamBindOp>(generator.getLoc(), inputDecl.cast<ParamDeclAttr>(),
                          inputValue);
  }

  // Now that we have a new synthesized generic kernel, run the rewriter over it
  // to specialize its body.
  return specializeKernel(newKernel);
}

/// Specialize a kernel interface with the specified input parameters and
/// return the generated kernel.  `insertionPoint` is always a point in the
/// primary module where a new kernel should be placed if necessary.
SmallVector<KernelOrCalleeError>
Elaborator::specializeInterface(GeneratorInterfaceOp itf,
                                ArrayRef<Attribute> inputParamValues,
                                Operation *insertionPoint) {
  SmallVector<KernelOrCalleeError> result;

  // An interface is an abstraction over multiple generators.  Invoke each of
  // them, collecting the results together into a single result.
  ArrayRef<GeneratorOp> interfaceImpls = getGeneratorsImplementing(itf);
  if (interfaceImpls.empty()) {
    // If we found no implementations, report that problem at the call site as
    // a single diagnostic.
    result.push_back(CalleeExpansionError(
        itf->getLoc(),
        ElaborationDiagnostic(itf->getLoc(),
                              Error(Twine("no implementations of interface '") +
                                    itf.getName() + "' found"))));
    return result;
  }

  for (GeneratorOp gen : interfaceImpls) {
    // Make sure to go through getAllInstantiations so generators are cached
    // and any constraints on the generator itself are validated.
    auto kernels = getAllInstantiations(gen, inputParamValues, insertionPoint);
    result.append(kernels.begin(), kernels.end());
  }
  return result;
}

/// Return all instantiations of the specified declaration (a kernel,
/// generator, or interface) with teh specified input parameter values.
/// `insertionPoint` is always a point in the primary module where a new
/// kernel should be placed if necessary.
ArrayRef<KernelOrCalleeError>
Elaborator::getAllInstantiations(Operation *decl,
                                 ArrayRef<Attribute> inputParamValues,
                                 Operation *insertionPoint) {

  // Check the cache these so multiple uses of the same kernel don't get
  // separate instantiations.
  auto inputParamKey = ArrayAttr::get(decl->getContext(), inputParamValues);
  auto cacheKey = std::make_pair(decl, inputParamKey);
  auto cacheIt = generatedKernels.find(cacheKey);
  if (cacheIt != generatedKernels.end())
    return cacheIt->second;

  SmallVector<KernelOrCalleeError> newCallees;

  auto localError = [&](Error err) {
    newCallees.push_back(CalleeExpansionError(
        decl->getLoc(), ElaborationDiagnostic(decl->getLoc(), std::move(err))));
  };

  // Evaluate any constraints for this declaration to see if this is a viable
  // expansion.  If not, the expansion fails.
  auto constraintResult =
      ParameterEvaluator::evaluateConstraints(decl, inputParamValues);
  if (failed(constraintResult)) {
    localError(constraintResult.takeError());
  } else if (auto kernel = dyn_cast<KernelOp>(decl)) {
    newCallees = specializeKernel(kernel);
  } else if (auto generator = dyn_cast<GeneratorOp>(decl)) {
    newCallees = specializeGenerator(generator, inputParamValues, kernel);
  } else if (auto interface = dyn_cast<GeneratorInterfaceOp>(decl)) {
    newCallees = specializeInterface(interface, inputParamValues, kernel);
  } else {
    localError("call to an unknown kind of declaration");
  }

  auto &result = generatedKernels[cacheKey];
  result = std::move(newCallees);
  return result;
}

//===----------------------------------------------------------------------===//
// generateKernels Driver
//===----------------------------------------------------------------------===//

/// Elaborate kernels in the specified module, incorporating implementation
/// logic from the specified library.
LogicalResult M::elaborateKernels(ModuleOp primary, ModuleOp library) {
  // We currently rely on pointer equivalence between attributes etc when
  // matching across modules, so the modules must be in the same context.  We
  // could relax this restriction in the future if there were a reason to.
  if (primary.getContext() != library.getContext())
    return primary.emitError() << "Cannot generate kernels when primary and "
                                  "library are in different MLIR contexts";
  Elaborator generator(primary, library);

  // Scan the primary and library module to collect all the interfaces,
  // verifying that any common interfaces are the same.
  if (generator.collectInterfaces())
    return failure();

  // Concretize all the kernels at the top-level.
  bool didFail = false;
  SmallVector<KernelOp, 16> kernelsToGenerate;

  // Collect all the kernels to generate in a prepass, because we will be
  // creating new kernels in the primary file that are already concretized and
  // we don't want to reprocess them.
  // FIXME: This isn't correct at all: kernels can call kernels which can lead
  // to them getting deleted.

  // TODO: When expanding a kernel we need to pass in history of prior expansion
  // bindings, which constrains/defines future expansions of the same thing, and
  // we need to return up novel bindings that are done.  For each multi-version
  // we need to track /which way/ we're resolving an ambiguity.  For something
  // like this:
  //    kernel @foo() {
  //      call @someInterface()        // has 5 implementations
  //    }
  //
  //    kernel @bar() { call @foo() }  // has 5 implementations
  //
  //    kernel @baz() {
  //      call @foo()
  //      call @foo()
  //      call @bar()
  //    }
  //
  // We should process @bar before recursing down to @foo.  We should only
  // generate 5 copies of @bar, each of which resolves the call to
  // foo->someInterface in the same direction.  We should not generate 5*5*5
  // copies of @bar that has all pairs of foo/someInterface resolutions.
  for (auto kernel : primary.getOps<KernelOp>())
    kernelsToGenerate.push_back(kernel);

  // Process each kernel.
  for (auto kernel : kernelsToGenerate) {
    // Elaborate the kernel into concrete versions.
    ArrayRef<KernelOrCalleeError> results =
        generator.getAllInstantiations(kernel, {}, kernel);

    /// If the kernel failed to expand into /anything/ then emit an error.
    if (llvm::all_of(results, [](const KernelOrCalleeError &result) -> bool {
          return std::holds_alternative<CalleeExpansionError>(result);
        })) {
      // TODO: We have a TON of information about /why/ it failed, format the
      // result info.
      kernel->emitError("failed to generate any kernels");
      didFail = true;
    }
  }

  // If we failed to expand any kernel, propagate that failure.
  if (didFail)
    return failure();

  // On success, we remove generators and generator interfaces from the file to
  // clean it up.
  for (Operation &op : llvm::make_early_inc_range(primary.getOps())) {
    if (isa<GeneratorOp, GeneratorInterfaceOp>(op))
      op.erase();
  }

  return success();
}
