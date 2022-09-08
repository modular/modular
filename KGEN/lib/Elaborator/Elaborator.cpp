//===- Elaborator.cpp -----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains core logic to lower a file full of kernel into concrete
// implementations of the kernels.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Elaborator.h"

#include "KGEN/KGENDialect/ElaboratorOpInterface.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/KGENPasses.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/ML/DType.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include <variant>

using namespace M;
using namespace KGEN;

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
  ElaborationDiagnostic(Location failureLoc, Error error)
      : failureLoc(failureLoc), payload(std::string(error.get())) {}
  ElaborationDiagnostic(Location failureLoc,
                        ArrayRef<CalleeExpansionError> calleeErrors)
      : failureLoc(failureLoc), payload(calleeErrors.vec()) {}

  bool operator==(const ElaborationDiagnostic &diag) const {
    return failureLoc == diag.failureLoc && payload == diag.payload;
  }

  /// This is the location within the declaration of the failure, e.g. of the
  /// call or other operator with a problem.
  Location getFailureLoc() const { return failureLoc; }

  bool isLocalError() const {
    return std::holds_alternative<std::string>(payload);
  }
  bool isCalleeError() const { return !isLocalError(); }

  StringRef getLocalMessage() const { return std::get<std::string>(payload); }
  MutableArrayRef<CalleeExpansionError> getCalleeErrors() {
    return std::get<std::vector<CalleeExpansionError>>(payload);
  }

private:
  /// This is the location within the declaration of the failure, e.g. of the
  /// call or other operator with a problem.
  Location failureLoc;

  /// The problem is either:
  ///   1) a local issue represented as an error on an operation.
  ///   2) a transitive issue where expansion of a call failed (the main
  ///      location) due to callees where something inside the callee failed to
  ///      expand.  Each callee has a location of the decl itself + the problem.
  std::variant<std::string, std::vector<CalleeExpansionError>> payload;
};

//===----------------------------------------------------------------------===//
// ElaboratedKernel class definition
//===----------------------------------------------------------------------===//

namespace {
/// This class keeps track of one result from binding a generator to a set of
/// input parameters.  It holds both the kernel that gets produced as well as
/// the (transitive) set of generator bindings used to create it.  This is used
/// to ensure that further derived kernels are only done with consistent
/// bindings.
class ElaboratedKernel {
public:
  /// This is the kernel that is produced.
  KernelOp kernel;

  /// These are the bindings used to produce the kernel.  The results are
  /// transitively flattened, so we don't need to maintain a tree of bindings.
  SmallDenseMap<DeclAndInputParamsPair, KernelOp> bindings;

  /// If we have a binding for the specified generator+InputParamSet, return it,
  /// otherwise return null.
  KernelOp getBinding(DeclAndInputParamsPair key) const {
    auto it = bindings.find(key);
    return it != bindings.end() ? it->second : KernelOp();
  }

  /// Return true if the set of bindings in this elaborated kernel are
  /// consistent with the specified set of bindings.
  bool isConsistentWith(const ElaboratedKernel &other) const;

  /// Declare that we're resolving the specified `declAndInputParams` to a
  /// specified callee.  The callee is known to have bindings that are
  /// consistent with ours, but may have additional entries to merge in.
  void addBinding(DeclAndInputParamsPair declAndInputParams,
                  const ElaboratedKernel &newCallee);

  LLVM_DUMP_METHOD void dump() const;

private:
  void addOneBinding(DeclAndInputParamsPair declAndInputParams,
                     KernelOp result) {
    auto &entry = bindings[declAndInputParams];
    assert((entry == KernelOp() || entry == result) &&
           "merged bindings must be consistent with each other");
    entry = result;
  }
};
} // namespace

void ElaboratedKernel::dump() const {
  if (!kernel) {
    llvm::errs() << "NULL ElaboratedKernel\n";
    return;
  }

  llvm::errs() << "ElaboratedKernel @" << KernelOp(kernel).getName() << "\n";
  unsigned entryNo = 0;
  for (auto entry : bindings) {
    StringAttr name = SymbolTable::getSymbolName(entry.first.first);
    llvm::errs() << "  #" << (entryNo++) << " @" << name << entry.first.second
                 << " = @" << entry.second.getNameAttr() << "\n";
  }
}

/// Return true if the set of bindings in this elaborated kernel are
/// consistent with the specified set of bindings.
bool ElaboratedKernel::isConsistentWith(const ElaboratedKernel &other) const {
  for (auto &binding : bindings) {
    if (KernelOp result = other.getBinding(binding.first))
      if (result != binding.second)
        return false;
  }
  return true;
}

/// Declare that we're resolving the specified `declAndInputParams` to a
/// specified callee.  The callee is known to have bindings that are
/// consistent with ours, but may have additional entries to merge in.
void ElaboratedKernel::addBinding(DeclAndInputParamsPair declAndInputParams,
                                  const ElaboratedKernel &newCallee) {

  // Remember the generator+inputParams to resolved callee binding.
  addOneBinding(declAndInputParams, newCallee.kernel);

  // We know the callee is consistent with our current binding set, but it may
  // also have bound generators that we haven't seen yet.  Remember them.
  for (auto &binding : newCallee.bindings)
    addOneBinding(binding.first, binding.second);
}

using ElaboratedKernelOrCalleeError =
    std::variant<ElaboratedKernel, CalleeExpansionError>;

//===----------------------------------------------------------------------===//
// Elaborator class definition
//===----------------------------------------------------------------------===//

namespace {
class Elaborator {
public:
  Elaborator(ModuleOp primary, ArrayRef<OwningOpRef<ModuleOp>> libraryModules)
      : primaryModule(primary), libraryModules(libraryModules), symbolTable() {}

  ModuleOp getPrimaryModule() const { return primaryModule; }

  /// Scan the primary and library module to collect all the interfaces,
  /// verifying that any common interfaces are the same.
  ParseResult collectInterfaces();

  // Check the kernel/generator call graph to reject any recursion.
  ParseResult checkRecursion();

  /// Return the operation that defines the specified symbol.
  Operation *lookupCallee(SymbolRefAttr symbolRef, ModuleOp sourceModule) {
    return symbolTable.lookupNearestSymbolFrom(sourceModule, symbolRef);
  }

  /// Return all instantiations of the specified declaration (a kernel,
  /// generator, or interface) with the specified input parameter values.
  /// `insertionPoint` is always a point in the primary module where a new
  /// kernel should be placed if necessary.
  ArrayRef<ElaboratedKernelOrCalleeError>
  getAllInstantiations(DeclAndInputParamsPair declAndInputParams,
                       Operation *insertionPoint);

  /// Insert a variant of an existing kernel into the primary file.
  void insertKernelVariant(KernelOp existing, KernelOp newKernel);

  ArrayRef<GeneratorOp> getGeneratorsImplementing(GeneratorInterfaceOp itf) {
    auto it = interfaceImpls.find(itf.getNameAttr());
    return it == interfaceImpls.end() ? ArrayRef<GeneratorOp>() : it->second;
  }

private:
  /// Specialize a kernel body, generating one variant or each viable
  /// instantiation of that body.  Kernels do not have parameters, but they can
  /// invoke interfaces etc which can cause them to produce multiple variants.
  ///
  /// SourceModule indicates which module in the included library this
  /// originally came from (likely not the primary module).
  SmallVector<ElaboratedKernelOrCalleeError>
  specializeKernel(KernelOp kernel, ModuleOp sourceModule);

  /// Specialize a kernel generator with the specified input parameters and
  /// return the generated kernel.  `insertionPoint` is always a point in the
  /// primary module where a new kernel should be placed if necessary.
  SmallVector<ElaboratedKernelOrCalleeError>
  specializeGenerator(DeclAndInputParamsPair declAndInputParams,
                      Operation *insertionPoint);

  /// Specialize a kernel interface with the specified input parameters and
  /// return the generated kernel.  `insertionPoint` is always a point in the
  /// primary module where a new kernel should be placed if necessary.
  SmallVector<ElaboratedKernelOrCalleeError>
  specializeInterface(DeclAndInputParamsPair declAndInputParams,
                      Operation *insertionPoint);

  SymbolTable &getPrimaryModuleSymbolTable() {
    return symbolTable.getSymbolTable(primaryModule);
  }

private:
  /// These are the two modules we start with.  The primary module is mutated by
  /// our algorithm, the library modules are immutable.
  ModuleOp primaryModule;
  ArrayRef<OwningOpRef<ModuleOp>> libraryModules;

  /// This symbol table allows efficient lookups in the primary module.
  SymbolTableCollection symbolTable;

  /// This collects all of the generator implementations of generator
  /// interfaces, across both the primary module and the library.
  DenseMap<StringAttr, SmallVector<GeneratorOp, 4>> interfaceImpls;

  /// This is a cache of already-instantiated declarations.  The key is the
  /// kernel/generator/interface and input parameters, the result are
  /// all-possible kernels that could be generated from this.
  DenseMap<DeclAndInputParamsPair, SmallVector<ElaboratedKernelOrCalleeError>>
      generatedKernels;

  /// This keeps track of kernels that were found to be non viable and need to
  /// be removed.  Their body block is empty (no terminator) so they are known
  /// to be invalid.  We keep them around to the end of elaboration to avoid
  /// invalidating iterators.
  std::vector<KernelOp> kernelsToRemove;
};
} // namespace

/// Insert a variant of an existing kernel into the primary file.
void Elaborator::insertKernelVariant(KernelOp existing, KernelOp newKernel) {
  auto insertPt = Block::iterator(existing.getOperation());
  getPrimaryModuleSymbolTable().insert(newKernel,
                                       /*insertionPoint*/ ++insertPt);
}

//===----------------------------------------------------------------------===//
// Elaborator Algorithm for one Kernel
//===----------------------------------------------------------------------===//

namespace {
/// This class keeps a set of defined parameter values and is used to evaluate
/// and simplify operations in a kernel based on those values.  If an error
/// happens during rewriting, the diagnostic is filled in and failure() is
/// returned.
class ParameterRewriter : public ParameterEvaluator {
public:
  ParameterRewriter(Elaborator &elaborator, KernelOp kernel,
                    ModuleOp sourceModule,
                    SmallVector<Operation *> opsToRewrite)
      : elaborator(elaborator), sourceModule(sourceModule),
        opsToRewrite(std::move(opsToRewrite)) {
    elaboratedKernel.kernel = kernel;
  }

  /// Create a clone of this rewriter, but refer with a clone of the kernel.
  /// This uses operationMap to remap our state onto the newly created kernel.
  ParameterRewriter(const ParameterRewriter &existing,
                    DenseMap<Operation *, Operation *> &operationMap);

  /// Process all the `opsToRewrite`, simplifying this kernel.  If new variants
  /// of this kernel are necessary, they are added to rewriterWorklist.
  LogicalResult
  rewriteOps(SmallVectorImpl<ParameterRewriter> &rewriterWorklist);

  /// Return the kernel we're generating into, along with its bindings.
  ElaboratedKernel takeElaboratedKernel() {
    assert(!diagnostic.has_value() &&
           "can't get the result kernel when a diagnostic was generated");
    return std::move(elaboratedKernel);
  }

  /// If elaboration of this kernel fails, then the client can get the error
  /// out.  This also deletes the dead husk of the kernel which may not even
  /// verify correctly.
  CalleeExpansionError takeDiagnosticAndEraseKernel() {
    assert(diagnostic.has_value() &&
           "cannot get diagnostic when none was generated");
    // The kernel is not viable so we need to delete it.  This op can appear in
    // various maps though, so instead of actually deleting it, we just delete
    // its body.  The cleanup pass at the end of elaboration will remove it.
    elaboratedKernel.kernel.getBodyBlock()->clear();
    return CalleeExpansionError(elaboratedKernel.kernel->getLoc(),
                                std::move(diagnostic.value()));
  }

  /// Generate a error expanding this kernel.  The location specified is the
  /// operation with the problem, and the message is the problem with it.
  LogicalResult error(Location loc, Error message) {
    assert(!diagnostic.has_value() && "Already emitted an error");
    diagnostic = ElaborationDiagnostic(loc, std::move(message));
    return failure();
  }

  /// Generate an error expanding this kernel for a call expansion problem.  The
  /// location specified is for the call.  Each entry in calleeErrors includes
  /// the location of the declaration that failed to expand along with why it
  /// failed.
  LogicalResult errorCalling(Location callLoc,
                             ArrayRef<CalleeExpansionError> calleeErrors) {
    assert(!diagnostic.has_value() && "Already emitted an error");
    diagnostic = ElaborationDiagnostic(callLoc, calleeErrors);
    return failure();
  }

private:
  LogicalResult processParamDeclareOp(ParamDeclareOp op);
  LogicalResult processParamConstantOp(ParamConstantOp op);
  LogicalResult processParamAssertOp(ParamAssertOp op);
  LogicalResult processCallOp(CallOp call,
                              SmallVectorImpl<ParameterRewriter> &rewriters);
  void completeCallOpProcessing(CallOp call,
                                DeclAndInputParamsPair calleeAndInputParams,
                                const ElaboratedKernel &newCallee);
  void spawnNewKernelClone(CallOp call,
                           DeclAndInputParamsPair calleeAndInputParams,
                           const ElaboratedKernel &callee,
                           SmallVectorImpl<ParameterRewriter> &rewriters);
  LogicalResult processGenericOp(Operation *op);

  /// This is maintains global information about the file we're generating into.
  Elaborator &elaborator;

  /// This indicates which module this kernel originally came from (e.g. one of
  /// the imported files).  This is important to know so we can correctly
  /// resolve callee symbols.
  ModuleOp sourceModule;

  /// This is the kernel we're working on.
  ElaboratedKernel elaboratedKernel;

  /// This is a diagnostic explaining the expansion failure if something goes
  /// wrong.
  Optional<ElaborationDiagnostic> diagnostic;

  /// These are the operations we still need to visit to complete our rewrite.
  SmallVector<Operation *> opsToRewrite;
};
} // namespace

/// Create a clone of this rewriter, but refer with a clone of the kernel.
/// This uses operationMap to remap our state onto the newly created kernel.
ParameterRewriter::ParameterRewriter(
    const ParameterRewriter &existing,
    DenseMap<Operation *, Operation *> &operationMap)
    : ParameterEvaluator(existing), elaborator(existing.elaborator),
      sourceModule(existing.sourceModule),
      elaboratedKernel(existing.elaboratedKernel) {
  // Remap the kernel operation.
  elaboratedKernel.kernel =
      cast<KernelOp>(operationMap[existing.elaboratedKernel.kernel]);
  assert(elaboratedKernel.kernel && "didn't remap kernel correctly");

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
    if (auto bind = dyn_cast<ParamDeclareOp>(op))
      result = processParamDeclareOp(bind);
    else if (auto value = dyn_cast<ParamConstantOp>(op))
      result = processParamConstantOp(value);
    else if (auto assertOp = dyn_cast<ParamAssertOp>(op))
      result = processParamAssertOp(assertOp);
    else if (auto call = dyn_cast<CallOp>(op))
      result = processCallOp(call, rewriterWorklist);
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
      elaboratedKernel.kernel.getContext(),
      [&](Diagnostic &diag) -> LogicalResult {
        (void)error(diag.getLocation(),
                    Twine("verification error: ") + diag.str());
        hadError = true;
        return success();
      });

  LogicalResult verifyResult = verify(elaboratedKernel.kernel);
  assert(hadError == failed(verifyResult) && "Result of verify is unexpected");
  return verifyResult;
}

LogicalResult ParameterRewriter::processParamDeclareOp(ParamDeclareOp op) {
  // Simplify the input expression.
  auto errorOrValue = concretizeParameterExpr(op.getValue());
  if (errorOrValue.isError())
    return error(op->getLoc(), errorOrValue.takeError());

  // Bind it to the parameter declaration it is setting.
  setParameterValue(op.getParamDecl(), errorOrValue.takeValue());

  // The param.bind operation serves no other purpose, so we can remove it.
  op->erase();
  return success();
}

LogicalResult ParameterRewriter::processParamConstantOp(ParamConstantOp op) {
  // ParamConstantOp projects a parameter expression into an SSA value.  We can
  // eventually lower this into lower level operators in the target set, but
  // for now we just simplify their operand.
  auto errorOrValue = concretizeParameterExpr(op.getValue());
  if (errorOrValue.isError())
    return error(op->getLoc(), errorOrValue.takeError());

  op.setValueAttr(errorOrValue.takeValue());
  return success();
}

LogicalResult ParameterRewriter::processParamAssertOp(ParamAssertOp op) {
  // Check the condition expression.
  auto errorOrValue = concretizeParameterExpr(op.getCond());
  if (errorOrValue.isError())
    return error(op->getLoc(), errorOrValue.takeError());

  auto resultInt = (*errorOrValue).dyn_cast<IntegerAttr>();
  if (!resultInt || resultInt.getValue().getBitWidth() != 1)
    return error(op->getLoc(),
                 "constraint evaluation didn't return true or false");
  // If the constraint evaluated to zero then the assert fails.
  if (resultInt.getValue().isZero())
    return error(op->getLoc(), "constraint failed: " + op.getMessage());

  // The kgen.param.assert op serves no further purpose, so we can remove it.
  op->erase();
  return success();
}

/// Resolve all of input parameters present at the specified call site to
/// concrete constants.  This reports the error and returns null on failure,
/// and returns an array of bound input parameters on success.
static ArrayAttr resolveCallInputParams(CallOp call,
                                        ParameterRewriter &rewriter) {
  SmallVector<Attribute> boundInputParams;
  for (auto param : call.getParamValues()) {
    auto value = rewriter.concretizeParameterExpr(
        param.cast<ParamBindAttr>().getValue());
    if (value.isError())
      return (void)rewriter.error(call->getLoc(), value.takeError()),
             ArrayAttr();

    boundInputParams.push_back(value.takeValue());
  }
  return ArrayAttr::get(call->getContext(), boundInputParams);
}

LogicalResult ParameterRewriter::processCallOp(
    CallOp call, SmallVectorImpl<ParameterRewriter> &rewriters) {
  // Evaluate any input parameters.
  auto inputParamKey = resolveCallInputParams(call, *this);
  if (!inputParamKey)
    return failure();

  // Instantiate the callee into one or more KernelOp's, depending on what the
  // callee is.
  auto callee = dyn_cast_or_null<KGENDeclInterface>(
      elaborator.lookupCallee(call.getCalleeAttr(), sourceModule));
  if (!callee)
    return error(call->getLoc(),
                 Twine("could not find callee '@") +
                     call.getCalleeAttr().getLeafReference().strref() + "'");

  DeclAndInputParamsPair calleeDeclAndInputParams{callee, inputParamKey};

  // If we already have a binding for this decl/inputParam set, then reuse the
  // consistent callee.
  if (KernelOp callee = elaboratedKernel.getBinding(calleeDeclAndInputParams)) {
    ElaboratedKernel elaboratedCallee;
    elaboratedCallee.kernel = callee;
    completeCallOpProcessing(call, calleeDeclAndInputParams, elaboratedCallee);
    return success();
  }

  // Otherwise, this is our first use of this.  Ask the global elaborator for
  // the full set of candidates.
  ArrayRef<ElaboratedKernelOrCalleeError> newCalleesRef =
      elaborator.getAllInstantiations(calleeDeclAndInputParams,
                                      elaboratedKernel.kernel);

  // Copy the list of kernels instead of referring to the cache entry to avoid
  // iterator invalidation problems.
  SmallVector<ElaboratedKernelOrCalleeError> newCallees(newCalleesRef.begin(),
                                                        newCalleesRef.end());

  // If we found more than one callee to produce then we need to spawn
  // multiple versions of the kernel we are currently constructing, each
  // which get a different callee.
  ElaboratedKernel thisCallee;
  for (const ElaboratedKernelOrCalleeError &candidate : newCallees) {
    // Ignore erroneous callees.
    if (std::holds_alternative<CalleeExpansionError>(candidate))
      continue;
    // Ignore the candidate if the elaborated kernel is inconsistent with our
    // current bindings.
    const ElaboratedKernel &calleeCandidate =
        std::get<ElaboratedKernel>(candidate);
    if (!calleeCandidate.isConsistentWith(elaboratedKernel))
      continue;

    // If this is the first viable candidates, then we will pursue it locally.
    if (!thisCallee.kernel)
      thisCallee = calleeCandidate;
    else
      /// All other callees gets spawned as sub-evaluators.
      spawnNewKernelClone(call, calleeDeclAndInputParams, calleeCandidate,
                          rewriters);
  }

  // If all the expansions failed, then this call fails overall.
  if (!thisCallee.kernel) {
    SmallVector<CalleeExpansionError> errors;
    for (const auto &value : newCalleesRef)
      errors.push_back(std::get<CalleeExpansionError>(value));
    return errorCalling(call->getLoc(), errors);
  }

  // Finally, we can handle the first viable one as our continued progress here.
  completeCallOpProcessing(call, calleeDeclAndInputParams, thisCallee);
  return success();
}

void ParameterRewriter::completeCallOpProcessing(
    CallOp call, DeclAndInputParamsPair calleeAndInputParams,
    const ElaboratedKernel &newCallee) {
  // Add a binding to remember that we resolved this call to this candidate,
  // and merge any bindings from it into our set.
  elaboratedKernel.addBinding(calleeAndInputParams, newCallee);

  KernelOp newCalleeKernel = newCallee.kernel;

  // Resolve any bound result types.
  SmallVector<Type> resultTypes;
  for (auto result : call.getResultTypes())
    resultTypes.push_back(getReboundType(result));

  // Now that we resolved the call to a new thing, build a new call to replace
  // the old one.
  OpBuilder b(call);
  auto newCall = b.create<CallOp>(
      call.getLoc(), resultTypes, newCalleeKernel.getNameAttr(),
      /*input params*/ ArrayRef<ParamBindAttr>(),
      /*output params*/ call.getParamDecls(), call.getOperands());

  // The SSA results of the old call go directly to the new call and remove it.
  call->getResults().replaceAllUsesWith(newCall);
  call->erase();

  // Bind the result parameters to the output parameter decls.
  for (auto [decl, bindValue] :
       llvm::zip(newCall.getParamDecls(),
                 newCalleeKernel.getReturnOp().getParameters()))
    setParameterValue(decl.cast<ParamDeclAttr>(),
                      bindValue.cast<ParamBindAttr>().getValue());
}

/// Sometimes when we expand a call, we find that there are multiple viable
/// callees that we can generate.  We handle this by spawning new parameter
/// rewriters with state copied from the current one, but which resolve the call
/// to different callees.  This spawns a new rewriter with the specified call
/// resolving to the specified callee.
void ParameterRewriter::spawnNewKernelClone(
    CallOp call, DeclAndInputParamsPair calleeAndInputParams,
    const ElaboratedKernel &callee,
    SmallVectorImpl<ParameterRewriter> &rewriters) {

  // Start by cloning the current WIP kernel to a new copy of it.
  BlockAndValueMapping blocksAndValues;
  DenseMap<Operation *, Operation *> operationMap;
  auto newKernel = cast<KernelOp>(
      cloneOperation(elaboratedKernel.kernel, blocksAndValues, operationMap));

  // Insert the kernel into the output file and auto-unique the symbol.
  elaborator.insertKernelVariant(elaboratedKernel.kernel, newKernel);

  // Generate the new rewriter which will process this.
  auto &newRewriter = rewriters.emplace_back(*this, operationMap);

  // Change the future of this kernel by resolving the call in the new kernel to
  // the specifed callee.
  auto newCall = cast<CallOp>(operationMap[call]);
  newRewriter.completeCallOpProcessing(newCall, calleeAndInputParams, callee);
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
    // Preserve but ignore the 'paramDecls' attribute on KernelOp.
    if (namedAttr.getName() == "paramDecls") {
      newAttrs.push_back(namedAttr);
      continue;
    }

    newAttrs.push_back(NamedAttribute(
        namedAttr.getName(), getReboundAttribute(namedAttr.getValue())));
    changedAttrs |= namedAttr.getValue() != newAttrs.back().getValue();
  }
  if (changedAttrs)
    op->setAttrs(newAttrs);

  // Check the types of results to find any parameters embedded in their
  // types.  We don't have to check operands because they are always checked
  // when being defined.
  for (OpResult result : op->getResults())
    result.setType(getReboundType(result.getType()));

  // Scan the region list if present.  The walker will automatically recurse
  // for us, but we have to check the block arguments.
  if (op->getNumRegions()) { // Microoptimization: getRegions() is slow.
    for (auto &region : op->getRegions())
      for (auto &block : region)
        for (Value arg : block.getArguments())
          arg.setType(getReboundType(arg.getType()));
  }

  // If the op implements the elaborator interface, indicate it as resolved.
  if (auto elaboratorIface = dyn_cast<ElaboratorOpInterface>(op)) {
    ErrorOrSuccess result = elaboratorIface.finalizeElaboration();
    if (result.isError())
      return error(op->getLoc(), result.takeError());
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
    } else if (auto typeConstant = value.dyn_cast<ConcreteTypeConstantAttr>()) {
      // NOTE: Could use pretty mangling for common cases, e.g. "simd2xf32" or
      // something if these get too verbose.
      os << typeConstant.getValue();
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
SmallVector<ElaboratedKernelOrCalleeError>
Elaborator::specializeKernel(KernelOp kernel, ModuleOp sourceModule) {
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
  rewriterWorklist.emplace_back(*this, kernel, sourceModule,
                                std::move(opsToRewrite));

  // Rewriting kernels may generate other kernel clones.  If so, rewrite them,
  // until we converge.
  SmallVector<ElaboratedKernelOrCalleeError> results;
  while (!rewriterWorklist.empty()) {
    auto rewriter = rewriterWorklist.pop_back_val();

    // If elaborating the kernel succeeded, then we have a viable candidate.
    if (succeeded(rewriter.rewriteOps(rewriterWorklist))) {
      results.push_back(rewriter.takeElaboratedKernel());
    } else {
      // If elaborating the kernel fails, then remember the diagnostic (in case
      // we need to explain why elaboration fails) and remove the broken husk of
      // a kernel that didn't make it.
      results.push_back(rewriter.takeDiagnosticAndEraseKernel());
    }
  }
  return results;
}

/// Specialize a kernel generator with the specified input parameters and
/// return the symbol name to use for the result, along with an array of
/// ParamBindAttrs for the result attributes.  `insertionPoint` is always a
/// point in the primary module where a new kernel should be placed if
/// necessary.
SmallVector<ElaboratedKernelOrCalleeError>
Elaborator::specializeGenerator(DeclAndInputParamsPair declAndInputParams,
                                Operation *insertionPoint) {
  auto generator = cast<GeneratorOp>(declAndInputParams.first);

  // We insert specializations of the generator immediately before the generator
  // if it is defined in the primary module.  Otherwise if it is from the
  // library, it would be better to insert it before the first client that
  // needed it (make tests easier to write).
  if (generator->getParentOp() == primaryModule) {
    insertionPoint = generator;
  } else {
    assert(insertionPoint && insertionPoint->getParentOp() == primaryModule);
  }

  ArrayRef<Attribute> inputParamValues = declAndInputParams.second.getValue();
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
  getPrimaryModuleSymbolTable().insert(newKernel);

  // Clone the body of the generator over.
  BlockAndValueMapping mapper;
  generator.getBody().cloneInto(&newKernel.getBody(), mapper);

  // Provide definitions of the input parameters in the body block as bound
  // constants.
  b.setInsertionPoint(&newKernel.getBodyBlock()->front());
  for (auto [inputDecl, inputValue] :
       llvm::zip(inputParamDecls, inputParamValues)) {
    b.create<ParamDeclareOp>(generator.getLoc(),
                             inputDecl.cast<ParamDeclAttr>(), inputValue);
  }

  // Now that we have a new synthesized generic kernel, run the rewriter
  // over it to specialize its body.
  auto sourceModule = generator->getParentOfType<ModuleOp>();
  return specializeKernel(newKernel, sourceModule);
}

/// Specialize a kernel interface with the specified input parameters and
/// return the generated kernel.  `insertionPoint` is always a point in the
/// primary module where a new kernel should be placed if necessary.
SmallVector<ElaboratedKernelOrCalleeError>
Elaborator::specializeInterface(DeclAndInputParamsPair declAndInputParams,
                                Operation *insertionPoint) {
  auto itf = cast<GeneratorInterfaceOp>(declAndInputParams.first);
  SmallVector<ElaboratedKernelOrCalleeError> result;

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
    auto kernels =
        getAllInstantiations({gen, declAndInputParams.second}, insertionPoint);
    result.append(kernels.begin(), kernels.end());
  }
  return result;
}

/// Return all instantiations of the specified declaration (a kernel,
/// generator, or interface) with teh specified input parameter values.
/// `insertionPoint` is always a point in the primary module where a new
/// kernel should be placed if necessary.
ArrayRef<ElaboratedKernelOrCalleeError>
Elaborator::getAllInstantiations(DeclAndInputParamsPair declAndInputParams,
                                 Operation *insertionPoint) {
  // Check the global cache of instantiations so we only ever instantiate a
  // generator once.
  auto cacheIt = generatedKernels.find(declAndInputParams);
  if (cacheIt != generatedKernels.end())
    return cacheIt->second;

  Operation *decl = declAndInputParams.first;
  SmallVector<ElaboratedKernelOrCalleeError> newCallees;
  auto localError = [&](Error err) {
    auto loc = decl->getLoc();
    newCallees.push_back(
        CalleeExpansionError(loc, ElaborationDiagnostic(loc, std::move(err))));
  };

  // Evaluate any constraints for this declaration to see if this is a viable
  // expansion.  If not, the expansion fails.
  auto constraintResult =
      ParameterEvaluator::evaluateConstraints(declAndInputParams);
  if (failed(constraintResult)) {
    localError(constraintResult.takeError());
  } else if (auto kernel = dyn_cast<KernelOp>(decl)) {
    auto sourceModule = decl->getParentOfType<ModuleOp>();

    // If the kernel being referenced is in an included module, then copy it
    // into the primary module (the primary module must be self contained by the
    // time we are done).  We can/should consider more flexible approaches, e.g.
    // allowing 'extern' references to kernels.
    if (sourceModule != primaryModule) {
      /// Clone the library kernel and insert it at the insertion point.
      Operation *cloned = kernel->clone();
      assert(insertionPoint && "must be set in non-primary modules");
      getPrimaryModuleSymbolTable().insert(cloned,
                                           Block::iterator(insertionPoint));
      kernel = cast<KernelOp>(cloned);
    }

    // FIXME: There is no need to specialize it.  All this does in practice is
    // pull in other recursively referenced kernels.
    newCallees = specializeKernel(kernel, sourceModule);
  } else if (isa<GeneratorOp>(decl)) {
    newCallees = specializeGenerator(declAndInputParams, insertionPoint);
  } else if (isa<GeneratorInterfaceOp>(decl)) {
    newCallees = specializeInterface(declAndInputParams, insertionPoint);
  } else {
    localError("call to an unknown kind of declaration");
  }

  auto &result = generatedKernels[declAndInputParams];
  result = std::move(newCallees);
  return result;
}

//===----------------------------------------------------------------------===//
// generateKernels Driver
//===----------------------------------------------------------------------===//

/// Scan the primary and library modules to collect all the interfaces,
/// verifying that any common interfaces are the same.
ParseResult Elaborator::collectInterfaces() {
  // Collect all the generator interfaces in the library modules, which will
  // allow cross-checking them below. Also, collect all the kernel generators
  // that implement a given interface, starting with the libraries.  These will
  // already have been type checked within the library.
  DenseMap<StringAttr, GeneratorInterfaceOp> libraryInterfaces;

  // Scan the specified module collecting all the generators that implement an
  // interface and checking the interfaces between library files line up.
  auto collectGeneratorsAndInterfaces = [&](ModuleOp module) -> ParseResult {
    for (auto &op : module.getOps()) {
      // Collect interfaces, and if we have seen another one already, verify
      // their signatures match.
      if (auto itf = dyn_cast<GeneratorInterfaceOp>(op)) {
        auto [it, inserted] =
            libraryInterfaces.insert({itf.getNameAttr(), itf});
        if (inserted) // Just remember it on the first hit.
          continue;

        // If this is the second match, check that the signatures match.
        if (failed(verifyDeclMatchesInterface("interface", itf,
                                              "library interface", it->second)))
          return failure();
        continue;
      }

      // If this is a generator, keep track of it.
      if (auto generator = dyn_cast<GeneratorOp>(op))
        if (auto interface = generator.getImplementsAttr())
          interfaceImpls[interface.getAttr()].push_back(generator);

      // Detect common errors cleanly, and report it.
      if (op.getName().getStringRef() == "hlkgen.generator")
        return op.emitError(
            "unlowered hlkgen.generator discovered in KGEN elaborator");
    }
    return success();
  };

  for (auto &module : libraryModules) {
    if (failed(collectGeneratorsAndInterfaces(module.get())))
      return failure();
  }

  // If they all match up, collect the generator implementations from the
  // primary module.
  return collectGeneratorsAndInterfaces(primaryModule);
}

namespace {
class RecursionChecker {
public:
  RecursionChecker(Elaborator &elaborator) : elaborator(elaborator) {}
  ParseResult run();

private:
  ParseResult checkRecursively(Operation *op, ModuleOp module);

  Elaborator &elaborator;
  llvm::SetVector<Operation *> callStack;
  std::vector<Operation *> callStackCalls;
  SmallPtrSet<Operation *, 8> alreadyCheckedOps;
};
} // namespace

/// Check the specified operation and its call-tree by visiting callees in
/// depth-first order.  If we report errors the call-graph, and keep track of
/// known-ok operators to avoid redundant work.
ParseResult RecursionChecker::checkRecursively(Operation *op, ModuleOp module) {
  // If we've already verified that this operator and all its callees are ok,
  // then we're already done.
  if (alreadyCheckedOps.count(op))
    return success();

  // If this op is in the call stack for our depth first traversal, then we've
  // found a cycle, reject it.  Otherwise add it to our call stack
  if (!callStack.insert(op)) {
    assert(callStack.size() == callStackCalls.size());
    size_t i = 0, e = callStack.size();
    // Skip over any leading stack that isn't part of the cycle.
    while (callStack[i] != op)
      ++i;
    // Report
    auto diag =
        op->emitError("declaration involved in recursive elaboration cycle");
    diag.attachNote(callStackCalls[i]->getLoc()) << "through this call";
    for (++i; i != e; ++i) {
      diag.attachNote(callStack[i]->getLoc()) << "to this declaration";
      diag.attachNote(callStackCalls[i]->getLoc()) << "through this call";
    }
    diag.attachNote(op->getLoc()) << "back to this declaration";
    return failure();
  }

  // Okay, we haven't seen this before, check all the calls in the body of the
  // declaration to see what they call.
  bool failed = false;
  op->walk([&](CallOp call) {
    auto callee = elaborator.lookupCallee(call.getCalleeAttr(), module);
    assert(callee && "couldn't resolve callee?");
    callStackCalls.push_back(call);
    if (isa<KernelOp, GeneratorOp>(callee)) {
      // For direct calls, we immediately check the callee.
      if (checkRecursively(callee, module))
        failed = true;
    } else if (auto itf = dyn_cast<GeneratorInterfaceOp>(callee)) {
      // For generator interfaces, we resolve to all the implementations.
      for (auto gen : elaborator.getGeneratorsImplementing(itf)) {
        // Make sure we keep track of the current module we're scanning.
        // TODO: This recursively checks all code in the imported libraries.
        // Will this be needed when these are individually checked on their own?
        ModuleOp genModule = gen->getParentOfType<ModuleOp>();
        if (checkRecursively(gen, genModule))
          failed = true;
      }
    } else {
      call->emitError("unknown callee in elaboration");
      failed = true;
    }
    callStackCalls.pop_back();
  });

  if (failed)
    return failure();

  callStack.pop_back();

  // Okay, we're successful!  Remember that we don't need to process this again.
  alreadyCheckedOps.insert(op);
  return success();
}

ParseResult RecursionChecker::run() {
  // Check all the operations at the top level of the primary module.
  auto module = elaborator.getPrimaryModule();
  for (Operation &op : module.getOps())
    if (checkRecursively(&op, module))
      return failure();
  return success();
}

/// Check the kernel/generator call graph to reject any recursion.
ParseResult Elaborator::checkRecursion() {
  return RecursionChecker(*this).run();
}

/// When a top-level kernel failed to elaborate, this is used to recursively
/// emit a tree of notes indicating why the elaboration tree failed.
static void emitElaborationError(InFlightDiagnostic &diag,
                                 MutableArrayRef<CalleeExpansionError> errors,
                                 unsigned indentDepth) {
  assert(!errors.empty());

  // This is true when we have multiple decls that a call resolved to, e.g. due
  // to an interface.
  bool haveMultipleDecls = false;
  std::string spaces(indentDepth, ' ');

  // Start by grouping the errors by declaration, emitting each one in turn.
  // Iteratively partition out things by declaration.
  while (!errors.empty()) {
    auto declLoc = errors[0].first;
    auto split =
        std::partition(errors.begin(), errors.end(),
                       [&](auto &elt) -> bool { return elt.first == declLoc; });
    haveMultipleDecls |= split != errors.end();

    // Process the batch.
    MutableArrayRef<CalleeExpansionError> batch =
        errors.take_front(split - errors.begin());

    // If there is one error in this batch, or if they are all at the same point
    // and are the same problem, collapse them together.  These forks must have
    // been different earlier in their elaboration but fail for the same reason.
    if (llvm::all_equal(batch))
      batch = batch.take_front();

    // If there are multiple alternative declarations in batches, emit a header
    // that groups each batch.
    if (haveMultipleDecls)
      diag.attachNote(declLoc) << spaces << "failed to expand this declaration";

    for (auto &error : batch) {
      ElaborationDiagnostic &elabError = error.second;
      if (elabError.isLocalError()) {
        diag.attachNote(elabError.getFailureLoc())
            << spaces << "  " << elabError.getLocalMessage();
      } else {
        diag.attachNote(elabError.getFailureLoc())
            << spaces << "  call expansion failed";
        emitElaborationError(diag, elabError.getCalleeErrors(),
                             indentDepth + 4);
      }
    }

    // Drop all the processed elements.
    errors = errors.drop_front(split - errors.begin());
  }
}

static LogicalResult
resolveInclude(IncludeOp include, ArrayRef<std::filesystem::path> searchPaths,
               DenseSet<StringAttr> &loadedFiles,
               SmallVectorImpl<OwningOpRef<ModuleOp>> &includedModules) {
  if (auto [_, didInsert] = loadedFiles.insert(include.getFileNameAttr());
      !didInsert)
    return success();

  std::string modulePath;
  if (std::filesystem::path(include.getFileName().str()).is_absolute()) {
    modulePath = include.getFileName().str();
  } else {
    for (const auto &p : searchPaths) {
      auto testPath = p / std::filesystem::path(include.getFileName().str());
      if (!std::filesystem::exists(testPath))
        continue;

      modulePath = testPath.string();
      break;
    }
    if (modulePath.empty())
      return include->emitError("could not find file '")
             << include.getFileName() << "'";
  }

  auto includedModule =
      mlir::parseSourceFile<ModuleOp>(modulePath, include->getContext());
  if (!includedModule)
    return mlir::emitError(include.getLoc(),
                           "failed to parse included source file");

  // Recursively resolve transitive includes.
  for (auto inc :
       llvm::make_early_inc_range(includedModule->getOps<IncludeOp>()))
    if (failed(resolveInclude(inc, searchPaths, loadedFiles, includedModules)))
      return failure();

  includedModules.push_back(std::move(includedModule));
  include->erase();
  return success();
}

/// Elaborate generators in the specified module, incorporating implementation
/// logic from the specified library.
LogicalResult
M::elaborateGenerators(ModuleOp primary,
                       ArrayRef<std::filesystem::path> searchPaths) {
  SmallVector<OwningOpRef<ModuleOp>> includedModules;
  DenseSet<StringAttr> loadedFiles;
  for (auto include : llvm::make_early_inc_range(primary.getOps<IncludeOp>()))
    if (failed(
            resolveInclude(include, searchPaths, loadedFiles, includedModules)))
      return failure();

  Elaborator elaborator(primary, includedModules);

  // Scan the primary and library module to collect all the interfaces,
  // verifying that any common interfaces are the same.
  if (elaborator.collectInterfaces())
    return failure();

  // Check the kernel/generator call graph to reject any recursion.
  if (elaborator.checkRecursion())
    return failure();

  auto emptyInputParamKey = ArrayAttr::get(primary.getContext(), {});

  // Elaborate the bodies of all generators without input parameters in the
  // primary module.  These are roots that will cause callees to get recursively
  // elaborated.
  // TODO: When we have access control, we can limit this to just the publicly
  // exposed ones.
  bool didFail = false;
  for (auto kernelRoot : primary.getOps<GeneratorOp>()) {
    // Ignore generators with input parameters, they can't be turned into
    // concrete kernels anyway, but will get specialized if anything uses them.
    if (!kernelRoot.getInputParamDecls().empty())
      continue;

    // Elaborate the kernel into concrete versions.
    ArrayRef<ElaboratedKernelOrCalleeError> results =
        elaborator.getAllInstantiations({kernelRoot, emptyInputParamKey},
                                        kernelRoot);

    // If the kernel failed to expand into /anything/ then emit an error.  Note
    // that the kernel will have been deleted.
    if (llvm::all_of(
            results, [](const ElaboratedKernelOrCalleeError &result) -> bool {
              return std::holds_alternative<CalleeExpansionError>(result);
            })) {
      // Collect the errors together.
      SmallVector<CalleeExpansionError> errors;
      for (const auto &value : results)
        errors.push_back(std::get<CalleeExpansionError>(value));
      auto diag = emitError(errors[0].first, "failed to generate any kernels");
      emitElaborationError(diag, errors, /*indentDepth=*/2);
      didFail = true;
    }
  }

  // If we failed to expand any kernel, propagate that failure.
  if (didFail)
    return failure();

  // On success, we remove generators and generator interfaces from the file to
  // clean it up.
  for (Operation &op : llvm::make_early_inc_range(primary.getOps())) {
    if (isa<GeneratorOp, GeneratorInterfaceOp>(op)) {
      op.erase();
      continue;
    }

    /// Non viable kernels will be left with an empty/invalid body.  Remove them
    /// at the end of elaboration.
    if (auto kernel = dyn_cast<KernelOp>(op))
      if (kernel.getBodyBlock()->empty())
        kernel->erase();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ElaborateGeneratorsPass
//===----------------------------------------------------------------------===//

namespace {
/// Run the kernel elaborator as a pass. The elaborator requires imports to be
/// resolved, so first resolve imports and then elaborate.
class ElaborateGeneratorsPass
    : public ElaborateGeneratorsBase<ElaborateGeneratorsPass> {
public:
  void runOnOperation() override {
    ModuleOp theModule = getOperation();

    SmallVector<std::filesystem::path> paths;
    for (const auto &p : searchPaths)
      paths.push_back(p);

    paths.push_back(std::filesystem::path("."));

    if (failed(elaborateGenerators(theModule, paths)))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> M::KGEN::createElaborateGeneratorsPass() {
  return std::make_unique<ElaborateGeneratorsPass>();
}
