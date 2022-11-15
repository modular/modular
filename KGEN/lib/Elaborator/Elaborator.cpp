//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains core logic to parameterized generators into concrete
// function implementations.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Elaborator.h"
#include "KGEN/KGENDialect/ElaboratorOpInterface.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/KGENPasses.h"
#include "SelectFastestFunction.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/ML/DType.h"
#include "Support/TimeProfiler.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include <numeric>
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

/// This class represents the reason that a  generator could not be elaborated.
/// It is either a local problem in the generator (e.g. an operator defining a
/// parameter that is unknown) or it is a call to another set of generators that
/// had problems expanding.
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
// ElaboratedGenerator class definition
//===----------------------------------------------------------------------===//

namespace {
/// This class keeps track of one result from binding a generator to a set of
/// input parameters.  It holds both the func that gets produced as well as
/// the (transitive) set of generator bindings used to create it.  This is used
/// to ensure that further-derived generators are only elaborated with
/// consistent bindings.
class ElaboratedGenerator {
public:
  explicit ElaboratedGenerator(FuncOp func) : func(func) {}

  /// This is the func that is produced.
  FuncOp func;

  /// These are the bindings used to produce the func.  The results are
  /// transitively flattened, so we don't need to maintain a tree of bindings.
  SmallDenseMap<DeclAndInputParamsPair, FuncOp> bindings;

  /// If we have a binding for the specified generator+InputParamSet, return it,
  /// otherwise return null.
  FuncOp getBinding(DeclAndInputParamsPair key) const {
    auto it = bindings.find(key);
    return it != bindings.end() ? it->second : FuncOp();
  }

  /// Return true if the set of bindings in this elaborated func are
  /// consistent with the specified set of bindings.
  bool isConsistentWith(const ElaboratedGenerator &other) const;

  /// Declare that we're resolving the specified `declAndInputParams` to a
  /// specified callee.  The callee is known to have bindings that are
  /// consistent with ours, but may have additional entries to merge in.
  void addBinding(DeclAndInputParamsPair declAndInputParams,
                  const ElaboratedGenerator &newCallee);

  LLVM_DUMP_METHOD void dump() const;

private:
  void addOneBinding(DeclAndInputParamsPair declAndInputParams, FuncOp result) {
    auto &entry = bindings[declAndInputParams];
    assert((entry == FuncOp() || entry == result) &&
           "merged bindings must be consistent with each other");
    entry = result;
  }
};
} // namespace

void ElaboratedGenerator::dump() const {
  if (!func) {
    llvm::errs() << "NULL ElaboratedGenerator\n";
    return;
  }

  llvm::errs() << "ElaboratedGenerator @" << FuncOp(func).getName() << "\n";
  unsigned entryNo = 0;
  for (auto entry : bindings) {
    StringAttr name = SymbolTable::getSymbolName(entry.first.first);
    llvm::errs() << "  #" << (entryNo++) << " @" << name << entry.first.second
                 << " = @" << entry.second.getNameAttr() << "\n";
  }
}

/// Return true if the set of bindings in this elaborated func are
/// consistent with the specified set of bindings.
bool ElaboratedGenerator::isConsistentWith(
    const ElaboratedGenerator &other) const {
  for (auto &binding : bindings) {
    if (FuncOp result = other.getBinding(binding.first))
      if (result != binding.second)
        return false;
  }
  return true;
}

/// Declare that we're resolving the specified `declAndInputParams` to a
/// specified callee.  The callee is known to have bindings that are
/// consistent with ours, but may have additional entries to merge in.
void ElaboratedGenerator::addBinding(DeclAndInputParamsPair declAndInputParams,
                                     const ElaboratedGenerator &newCallee) {

  // Remember the generator+inputParams to resolved callee binding.
  addOneBinding(declAndInputParams, newCallee.func);

  // We know the callee is consistent with our current binding set, but it may
  // also have bound generators that we haven't seen yet.  Remember them.
  for (auto &binding : newCallee.bindings)
    addOneBinding(binding.first, binding.second);
}

using ElaboratedGeneratorOrCalleeError =
    std::variant<ElaboratedGenerator, CalleeExpansionError>;

//===----------------------------------------------------------------------===//
// RegionReferenceAttr
//===----------------------------------------------------------------------===//

/// Region references in the elaborator are encoded as string attributes with
/// signature types. The string value is a key that uniquely identifies a bound
/// region whose ownership has been transferred to the elaborator. The string
/// value is used to mangle the name of instantiated functions.
namespace {
class RegionReferenceAttr : public StringAttr {
public:
  using StringAttr::StringAttr;

  /// Allow implicit conversion for dense maps.
  RegionReferenceAttr(StringAttr impl) : StringAttr(impl) {}

  /// Create a region reference.
  static RegionReferenceAttr get(const Twine &regionName, SignatureType type) {
    return llvm::cast<RegionReferenceAttr>(StringAttr::get(regionName, type));
  }

  /// Support type inquiry.
  static bool classof(Attribute attr) {
    if (auto string = llvm::dyn_cast<StringAttr>(attr))
      return llvm::isa<SignatureType>(string.getType());
    return false;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Elaborator class definition
//===----------------------------------------------------------------------===//

namespace {
class Elaborator {
public:
  /// Initialize the elaborator and its symbol table.
  Elaborator(SymbolTable &symtab, bool enableSearch = false)
      : symtab(symtab), regionOwner(ModuleOp::create(symtab.getOp()->getLoc())),
        enableSearch(enableSearch) {}

  /// Scan the primary and library module to collect all the interfaces,
  /// verifying that any common interfaces are the same.
  ParseResult collectInterfaces();

  /// Return the operation that defines the specified symbol.
  Operation *lookupCallee(FlatSymbolRefAttr symbolRef) {
    return symtab.lookup(symbolRef.getAttr());
  }

  /// Return all instantiations of the specified declaration (a func,
  /// generator, or interface) with the specified input parameter values.
  ArrayRef<ElaboratedGeneratorOrCalleeError>
  getAllInstantiations(DeclAndInputParamsPair declAndInputParams,
                       bool inlined = false);

  /// Insert a variant of an existing func into the primary file.
  void insertFuncVariant(FuncOp existing, FuncOp newFunc);

  /// Indicate that a function should be removed from the module at the end of
  /// elaboration. These functions are either invalid instantiations or
  /// inlined.
  void markFuncForRemoval(FuncOp func) { funcsToRemove.insert(func); }

  /// Return true if the function was invalid or inlined and should be removed
  /// from the module.
  bool shouldRemoveFunc(FuncOp func) { return funcsToRemove.contains(func); }

  /// Returns true if search is enabled.
  bool isSearchEnabled() { return enableSearch; }

  ArrayRef<GeneratorOp> getGeneratorsImplementing(GeneratorInterfaceOp itf) {
    auto it = interfaceImpls.find(itf.getNameAttr());
    return it == interfaceImpls.end() ? ArrayRef<GeneratorOp>() : it->second;
  }

  const DenseMap<GeneratorOp, FuncOp> &
  getFirstConcreteFuncForGenerator() const {
    return firstConcreteFuncForGenerator;
  }

  /// Bind the result parameters of a fully-specialized function and clear them
  /// from the function.
  void bindResultParameters(FuncOp func) {
    ParameterExprArrayAttr &values = resultParams[func];
    assert(!values && "results for function already bound");
    values = func.getReturnOp().getParametersAttr();
    func.setResultParamTypes({});
    func.getReturnOp().setParameters({});
  }

  /// Lookup bound result parameters of a function.
  ParameterExprArrayAttr lookupResultParameters(FuncOp func) {
    auto it = resultParams.find(func);
    assert(it != resultParams.end() && "results parameters not bound");
    return it->second;
  }

  /// This struct contains a region body and the parameter context at its
  /// original parent operation.
  struct RegionBody {
    KGENDeclInterface body;
    ParameterEvaluator evaluator;

    /// Return true if the body is isolated from above.
    bool isIsolatedFromAbove() const {
      return body->hasTrait<OpTrait::IsIsolatedFromAbove>();
    }
  };

  /// These methods provide access to the `regionsReferenced` dictionary. This
  /// tracks regions on kgen.call operations with unique string names. The
  /// elaborator takes ownership of the body by hooking it up to a module as
  /// its virtual root.
  void addRegionReference(RegionReferenceAttr attr, RegionBody regionBody) {
    regionsReferenced[attr] = std::move(regionBody);
    regionOwner->push_back(regionBody.body);
  }

  /// Get a reference region by name. The referenced region can be isolated
  /// from above or not isolated from above.
  const RegionBody &getRegionReferenced(RegionReferenceAttr attr) {
    return regionsReferenced[attr];
  }

  /// Have the elaborator take the nested parameter declarations and uses for
  /// a set of nested parameter scopes. The uses can be looked up later when
  /// elaboration recurses into those nested scopes.
  void takeNestedParameterUses(
      DenseMap<KGENDeclInterface, ParameterDeclsAndUses> &&nestedUses) {
    for (auto entry : nestedUses) {
      auto [_, inserted] = nestedParamDeclsAndUses.try_emplace(
          entry.first, std::move(entry.second));
      assert(inserted && "already found uses for nested scope?");
    }
  }

  /// Try to find cached parameter declarations and uses for the provided
  /// nested scope. Due to nested scopes getting cloned, nested scopes will
  /// only hit the cache at best every other depth. Still, it covers the most
  /// common case of depth 1.
  const ParameterDeclsAndUses *lookupNestedUses(KGENDeclInterface decl) {
    auto it = nestedParamDeclsAndUses.find(decl);
    return it == nestedParamDeclsAndUses.end() ? nullptr : &it->second;
  }

private:
  /// Specialize a func body, generating one variant or each viable
  /// instantiation of that body.  Funcs do not have input parameters, but
  /// they can invoke interfaces etc which can cause them to produce multiple
  /// variants.
  ///
  /// SourceModule indicates which module in the included library this
  /// originally came from (likely not the primary module).
  SmallVector<ElaboratedGeneratorOrCalleeError>
  specializeFunc(FuncOp func, ModuleOp sourceModule, bool inlined);

  /// Specialize a generator with the specified input parameters and return
  /// the generated func.
  SmallVector<ElaboratedGeneratorOrCalleeError>
  specializeGenerator(DeclAndInputParamsPair declAndInputParams, bool inlined);

  /// Specialize a generator interface with the specified input parameters and
  /// return the generated func.
  SmallVector<ElaboratedGeneratorOrCalleeError>
  specializeInterface(DeclAndInputParamsPair declAndInputParams, bool inlined);

  /// Report an error given an interface and an error string - just reduces
  /// boilerplate around CalleeExpansionError creation.
  ElaboratedGeneratorOrCalleeError
  reportCalleeExpansionError(GeneratorInterfaceOp itf, Twine err) {
    return CalleeExpansionError(
        itf->getLoc(), ElaborationDiagnostic(itf->getLoc(), Error(err)));
  };

private:
  /// This symbol table allows efficient lookups across the module.
  SymbolTable &symtab;

  /// This collects all of the generator implementations of generator
  /// interfaces, across both the primary module and the library.
  DenseMap<StringAttr, SmallVector<GeneratorOp, 4>> interfaceImpls;

  /// This map contains bindings for result parameteres from specialized
  /// functions.
  DenseMap<FuncOp, ParameterExprArrayAttr> resultParams;

  /// This is a cache of already-instantiated declarations.  The key is the
  /// generator/interface and input parameters, the result are all-possible
  /// funcs that could be generated from this.
  DenseMap<DeclAndInputParamsPair,
           SmallVector<ElaboratedGeneratorOrCalleeError>>
      generatedFuncs;

  /// This is keeps track of a mapping from named regions (which get pulled
  /// out of kgen.call's during elaboration) to a Block that provides the
  /// body.
  DenseMap<RegionReferenceAttr, RegionBody> regionsReferenced;

  /// The elaborator maintains ownership of all region body parameters through
  /// an owned module that is the parent of all region bodies. The region body
  /// operations require a virtual for parameter scanning.
  OwningOpRef<ModuleOp> regionOwner;

  /// This map contains the parameter declarations and uses for nested
  /// parameter scopes (region bodies).
  DenseMap<KGENDeclInterface, ParameterDeclsAndUses> nestedParamDeclsAndUses;

  /// This map keeps track of the first func that a generator with no
  /// parameters expanded into.  We rename it to have the same symbol as the
  /// original generator in a post-pass.
  DenseMap<GeneratorOp, FuncOp> firstConcreteFuncForGenerator;

  /// This tracks generated functions that should be removed after
  /// elaboration. These functions were either inlined by a parameter rewriter
  /// or are malformed. These functions need to be cleaned up at the end of
  /// the pass.
  DenseSet<FuncOp> funcsToRemove;

  /// Enable search during interface elaboration. This defaults to `false`
  /// because we want search to be opt-in.
  bool enableSearch = false;
};
} // namespace

/// Insert a variant of an existing func into the primary file.
void Elaborator::insertFuncVariant(FuncOp existing, FuncOp newFunc) {
  auto insertPt = Block::iterator(existing.getOperation());
  symtab.insert(newFunc, ++insertPt);
}

//===----------------------------------------------------------------------===//
// Elaborator Algorithm for one func implementation
//===----------------------------------------------------------------------===//

/// The worklist in ParameterRewriter contains commands that may either be an
/// operation to rewrite or a "RegionReturn" command that instructs it to
/// evaluate any returned parameter expressions, pop its evaluator stack, and
/// bind the returned parameters to a set of paramdecls in the callers context.
struct RegionReturn {
  /// This is the location of the call to the parameter region.
  Location callLoc;

  /// This is the location of the return from the region that binds the
  /// parameters.
  Location returnLoc;

  /// This is a set of parameter expressions to evaluate as part of the return,
  /// taken from the ReturnOp's parameters list.
  ParameterExprArrayAttr returnedParamExprs;

  /// This is the set of declarations bound by the returned expressions when the
  /// call is popped off the evaluator stack.
  ParamDeclArrayAttr callerParamDecls;
};

/// Commands are either operations to process or RegionReturn commands.  The
/// RegionReturn commands are heap allocated so they occupy a single word:
/// this keeps each entry in the worklist one pointer.
using RewriterCommandType = llvm::PointerUnion<Operation *, RegionReturn *>;

namespace {
/// This class keeps a set of defined parameter values and is used to evaluate
/// and simplify operations in a func based on those values.  If an error
/// happens during rewriting, the diagnostic is filled in and failure() is
/// returned.
class ParameterRewriter {
public:
  ParameterRewriter(Elaborator &elaborator, FuncOp func, SymbolTable &symtab,
                    ArrayRef<Operation *> opsToRewrite, bool inlinedCallee)
      : elaborator(elaborator), symtab(&symtab),
        sourceModule(cast<ModuleOp>(symtab.getOp())), elaboratedGenerator(func),
        inlinedCallee(inlinedCallee) {
    nextCallRegionID = 0;
    evaluators.push_back(ParameterEvaluator(&symtab));
    commandWorklist.reserve(opsToRewrite.size());
    llvm::append_range(commandWorklist, opsToRewrite);
  }

  /// Create a clone of this rewriter, but refer with a clone of the func.
  /// This uses operationMap to remap our state onto the newly created func.
  ParameterRewriter(const ParameterRewriter &existing,
                    DenseMap<Operation *, Operation *> &operationMap);
  ParameterRewriter(const ParameterRewriter &) = delete;
  ~ParameterRewriter();

  /// Return the evaluator currently being used for our rewrite.  This is the
  /// top of the stack of evaluators we track.
  ParameterEvaluator &getEvaluator() {
    assert(!evaluators.empty());
    return evaluators.back();
  }

  /// Process all the `commandWorklist`, simplifying this func.  If new
  /// variants of this func are necessary, they are added to rewriterWorklist.
  LogicalResult rewriteOps(
      SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriterWorklist);

  /// Return the func we're generating into, along with its bindings.
  ElaboratedGenerator takeElaboratedGenerator() {
    assert(!diagnostic.has_value() &&
           "can't get the result func when a diagnostic was generated");
    return std::move(elaboratedGenerator);
  }

  /// If elaboration of this func fails, then the client can get the error
  /// out.  This also deallocates the body of the dead husk of the func which
  /// may not even verify correctly, it will be removed later.
  CalleeExpansionError takeDiagnosticAndEraseFunc();

  /// Generate a error expanding this generator.  The location specified is
  /// the operation with the problem, and the message is the problem with it.
  LogicalResult error(Location loc, Error message) {
    assert(!diagnostic.has_value() && "Already emitted an error");
    diagnostic = ElaborationDiagnostic(loc, std::move(message));
    return failure();
  }

  /// Generate an error expanding this generator for a call expansion problem.
  /// The location specified is for the call.  Each entry in calleeErrors
  /// includes the location of the declaration that failed to expand along
  /// with why it failed.
  LogicalResult errorCalling(Location callLoc,
                             ArrayRef<CalleeExpansionError> calleeErrors) {
    assert(!diagnostic.has_value() && "Already emitted an error");
    diagnostic = ElaborationDiagnostic(callLoc, calleeErrors);
    return failure();
  }

private:
  LogicalResult processParamDeclareOp(ParamDeclareOp op);
  LogicalResult processParamSearchOp(
      ParamSearchOp op,
      SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriters);
  void spawnParamSearchClone(
      ParamSearchOp searchOp, Attribute value,
      SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriters);
  void completeParamSearchOpProcessing(ParamSearchOp op, Attribute value);

  LogicalResult processParamConstantOp(ParamConstantOp op);
  LogicalResult processParamAssertOp(ParamAssertOp op);

  /// Resolve the input parameters to a call into concrete values. This returns
  /// an array of bound input constants and a flag indicating whether the
  /// instantiated callee will always be inlined.
  FailureOr<std::pair<ArrayAttr, bool>>
  resolveCallInputParams(Operation *call, ArrayRef<ParamBindAttr> inputValues,
                         bool isInlinedCall);

  /// Process either a `kgen.addressof` op or a `kgen.call` op.
  template <typename OpT>
  LogicalResult processGeneratorUser(
      OpT user,
      SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriters) {
    return processGeneratorUserImpl(user, user.getParamDecls(),
                                    user.getParamValues(), user.getCalleeAttr(),
                                    rewriters, /*isInlinedCall=*/false);
  }
  LogicalResult processGeneratorUserImpl(
      Operation *user, ArrayRef<ParamDeclAttr> decls,
      ArrayRef<ParamBindAttr> paramValues, FlatSymbolRefAttr calleeAttr,
      SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriters,
      bool isInlinedCall);
  void
  completeGeneratorUserProcessing(Operation *user,
                                  ArrayRef<ParamDeclAttr> decls,
                                  DeclAndInputParamsPair calleeAndInputParams,
                                  const ElaboratedGenerator &newCallee);
  void spawnNewFuncClone(
      Operation *user, ArrayRef<ParamDeclAttr> decls,
      DeclAndInputParamsPair calleeAndInputParams,
      const ElaboratedGenerator &callee,
      SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriters);

  /// Process a `kgen.call_param` operation by inlining region callees and
  /// simplifying to `kgen.call` for symbol callees.
  LogicalResult processCallParamOp(
      CallParamOp call,
      SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriter);

  /// Process a `kgen.inlined_call` operation by inlining the callee.
  LogicalResult processInlinedCallOp(
      InlinedCallOp call,
      SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriters);

  /// Process a call to a region.
  LogicalResult processRegionCallImpl(KGENCallOpInterface call,
                                      RegionReferenceAttr regionName,
                                      ParamDeclArrayAttr decls,
                                      ArrayRef<ParamBindAttr> paramValues);

  LogicalResult processGenericOp(Operation *op);

  /// This is maintains global information about the file we're generating
  /// into.
  Elaborator &elaborator;

  /// The symbol table for lookups.
  SymbolTable *symtab;

  /// This indicates which module this func originally came from (e.g. one of
  /// the imported files).  This is important to know so we can correctly
  /// resolve callee symbols.
  ModuleOp sourceModule;

  /// This is the generator -> func we're working on.
  ElaboratedGenerator elaboratedGenerator;

  /// This is the diagnostic explaining the expansion failure if something
  /// goes wrong.
  Optional<ElaborationDiagnostic> diagnostic;

  /// This is a stack of evaluators to use to process parameter expressions.
  /// The current evaluator is always at the back() of the list.  Having a
  /// stack of evaluators allows us to maintain scoped parameters when
  /// processing regions.
  SmallVector<ParameterEvaluator, 2> evaluators;

  /// These are the commands that still need to get performed before this func
  /// has been fully evaluated.  These are mostly operations that need to be
  /// rewritten.
  SmallVector<RewriterCommandType> commandWorklist;

  /// This is a counter that gives each region attached to a kgen.call a
  /// unique number (and therefore, unique name).
  unsigned nextCallRegionID;

  /// A flag to indicate whether the elaborated function will be inlined.
  bool inlinedCallee;
};
} // namespace

ParameterRewriter::~ParameterRewriter() {
  // If a ParameterRewriter is aborted without completion, we need to make sure
  // to deallocate any RegionReturn nodes in the commandWorklist.
  for (RewriterCommandType command : commandWorklist)
    if (RegionReturn *rr = dyn_cast<RegionReturn *>(command))
      delete rr;
  commandWorklist.clear();
}

/// Create a clone of this rewriter, but refer with a clone of the func.
/// This uses operationMap to remap our state onto the newly created func.
ParameterRewriter::ParameterRewriter(
    const ParameterRewriter &existing,
    DenseMap<Operation *, Operation *> &operationMap)
    : elaborator(existing.elaborator), symtab(existing.symtab),
      sourceModule(existing.sourceModule),
      elaboratedGenerator(existing.elaboratedGenerator),
      evaluators(existing.evaluators),
      nextCallRegionID(existing.nextCallRegionID),
      inlinedCallee(existing.inlinedCallee) {
  // Remap the func operation.
  elaboratedGenerator.func =
      cast<FuncOp>(operationMap[existing.elaboratedGenerator.func]);
  assert(elaboratedGenerator.func && "didn't remap func correctly");

  // Remap the operation in the command worklist.
  commandWorklist.reserve(existing.commandWorklist.size());
  for (RewriterCommandType command : existing.commandWorklist) {
    if (Operation *op = dyn_cast<Operation *>(command)) {
      commandWorklist.push_back(operationMap[op]);
      assert(commandWorklist.back() && "didn't clone operation correctly?");
    } else {
      RegionReturn *rr = cast<RegionReturn *>(command);
      // Copy RegionReturn commands to a unique pointer.
      commandWorklist.push_back(new RegionReturn(*rr));
    }
  }
}

/// Work the `opsToRewrite` worklist.
LogicalResult ParameterRewriter::rewriteOps(
    SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriterWorklist) {
  // We use a worklist for this so cloned versions of ParameterRewriter can
  // be created and known where to pick up from.
  while (!commandWorklist.empty()) {
    RewriterCommandType command = commandWorklist.pop_back_val();

    // Most commands in the worklist are operations that need to be rewritten.
    if (Operation *op = dyn_cast<Operation *>(command)) {
      LogicalResult result = success();
      // Process an operation that needs to be rewritten/lowered based on the
      // context of the parameter values we know are defined.
      if (auto bind = dyn_cast<ParamDeclareOp>(op))
        result = processParamDeclareOp(bind);
      else if (auto value = dyn_cast<ParamSearchOp>(op))
        result = processParamSearchOp(value, rewriterWorklist);
      else if (auto value = dyn_cast<ParamConstantOp>(op))
        result = processParamConstantOp(value);
      else if (auto assertOp = dyn_cast<ParamAssertOp>(op))
        result = processParamAssertOp(assertOp);
      else if (auto addressof = dyn_cast<AddressOfOp>(op))
        result = processGeneratorUser(addressof, rewriterWorklist);
      else if (auto call = dyn_cast<CallOp>(op))
        result = processGeneratorUser(call, rewriterWorklist);
      else if (auto call = dyn_cast<CallParamOp>(op))
        result = processCallParamOp(call, rewriterWorklist);
      else if (auto call = dyn_cast<InlinedCallOp>(op))
        result = processInlinedCallOp(call, rewriterWorklist);
      else
        result = processGenericOp(op);

      // If processing any operation failed, then this entire func elaboration
      // failed.
      if (failed(result))
        return failure();
      continue;
    }

    // Otherwise we have a RegionReturn operation.
    std::unique_ptr<RegionReturn> rr(cast<RegionReturn *>(command));

    // Evaluate each of the returned parameter expressions in the current
    // scope.
    SmallVector<Attribute> returnedParams;
    for (TypedAttr expr : rr->returnedParamExprs) {
      auto value = getEvaluator().concretizeParameterExpr(expr);
      if (value.isError())
        return error(rr->returnLoc, value.takeError());
      returnedParams.push_back(value.takeValue());
    }
    // Next, pop the evaluator for the now-returned region.
    assert(evaluators.size() > 1 && "Don't have excess evaluators to pop!");
    evaluators.pop_back();

    // Bind each of the result parameter declarations in the callers context.
    assert(rr->callerParamDecls.size() == returnedParams.size());
    for (auto [decl, value] : llvm::zip(rr->callerParamDecls, returnedParams))
      getEvaluator().setOrOverwriteParameterValue(decl, value);
  }

  // If the generated function will be inlined, don't verify it.
  FuncOp func = elaboratedGenerator.func;
  if (inlinedCallee) {
    elaborator.markFuncForRemoval(func);
    return success();
  }

  // Check that the thing we just built is correct IR!  We want to catch any
  // errors produced by the verify pass, we don't want them to actually get
  // emitted.
  std::string verificationErrorStr;
  llvm::raw_string_ostream verificationError(verificationErrorStr);
  Optional<Location> verificationLoc;
  mlir::ScopedDiagnosticHandler diagHandler(
      elaboratedGenerator.func.getContext(),
      [&](Diagnostic &diag) -> LogicalResult {
        // Combine multiple verification errors.
        if (verificationLoc) {
          verificationError << "; " << diag.str();
          verificationLoc =
              FusedLoc::get(verificationLoc->getContext(),
                            {*verificationLoc, diag.getLocation()});
        } else {
          verificationError << diag.str();
          verificationLoc = diag.getLocation();
        }
        return success();
      });

  // Verify the function and invoke its symbol user verifier.
  SymbolTableCollection collection;
  // Avoid recomputing the symbol table.
  SymbolTable &localSymtab = collection.getSymbolTable(func->getParentOp());
  localSymtab = std::move(*symtab);
  auto cleanup =
      llvm::make_scope_exit([&] { *symtab = std::move(localSymtab); });
  auto verifySymbolUses = [&](mlir::SymbolUserOpInterface user) -> WalkResult {
    return user.verifySymbolUses(collection);
  };
  if (failed(verify(func)) || func.walk(verifySymbolUses).wasInterrupted())
    return error(*verificationLoc,
                 Twine("verification error: ") + verificationError.str());

  return success();
}

LogicalResult ParameterRewriter::processParamDeclareOp(ParamDeclareOp op) {
  // Simplify the input expression.
  auto errorOrValue = getEvaluator().concretizeParameterExpr(op.getValue());
  if (errorOrValue.isError())
    return error(op->getLoc(), errorOrValue.takeError());

  // Bind it to the parameter declaration it is setting.
  getEvaluator().setOrOverwriteParameterValue(op.getParamDecl(),
                                              errorOrValue.takeValue());

  // The kgen.param.declare operation serves no other purpose: remove it.
  op->erase();
  return success();
}

LogicalResult ParameterRewriter::processParamSearchOp(
    ParamSearchOp op,
    SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriters) {
  // Loop over all the possible candidates that we will search over, spawning
  // N-1 possibilities to explore.
  std::string errors;
  Attribute firstValid;
  DenseSet<Attribute> seenValues;
  for (Attribute candidate : op.getValues()) {
    // Simplify the input expressions.
    auto errorOrValue = getEvaluator().concretizeParameterExpr(candidate);
    if (errorOrValue.isError()) {
      if (!errors.empty())
        errors += ", ";
      errors += errorOrValue.takeError().get();
      continue;
    }

    Attribute value = errorOrValue.get();

    // If we've already seen this concrete value before, ignore the duplicate.
    if (!seenValues.insert(value).second)
      continue;

    // If this is the first viable value we've seen, remember it.
    if (!firstValid) {
      firstValid = value;
      // If we are not doing search in the elaborator, then we are done after
      // processing the first parameter.
      if (!elaborator.isSearchEnabled())
        break;
    } else {
      // Otherwise, we have to enqueue an exploration of this value.
      spawnParamSearchClone(op, value, rewriters);
    }
  }

  // If all the expansions failed, then this call fails overall.
  if (!firstValid) {
    if (errors.empty())
      return error(op->getLoc(), "no values to search over");
    return error(op->getLoc(), Error(errors));
  }

  completeParamSearchOpProcessing(op, firstValid);
  return success();
}

void ParameterRewriter::spawnParamSearchClone(
    ParamSearchOp searchOp, Attribute value,
    SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriters) {
  // Start by cloning the current WIP func to a new copy of it.
  BlockAndValueMapping blocksAndValues;
  DenseMap<Operation *, Operation *> operationMap;
  auto newFunc = cast<FuncOp>(
      cloneOperation(elaboratedGenerator.func, blocksAndValues, operationMap));

  // Insert the func into the output file and auto-unique the symbol.
  elaborator.insertFuncVariant(elaboratedGenerator.func, newFunc);

  // Generate the new rewriter which will process this.
  auto &newRewriter =
      rewriters.emplace_back(new ParameterRewriter(*this, operationMap));

  // Change the future of this func by resolving the searchOp in the new func to
  // the specifed value.
  auto newSearch = cast<ParamSearchOp>(operationMap[searchOp]);
  newRewriter->completeParamSearchOpProcessing(newSearch, value);
}

void ParameterRewriter::completeParamSearchOpProcessing(ParamSearchOp op,
                                                        Attribute value) {
  // Bind it to the parameter declaration it is setting.
  getEvaluator().setOrOverwriteParameterValue(op.getParamDecl(), value);

  // The kgen.param.search operation serves no other purpose: remove it.
  op->erase();
}

LogicalResult ParameterRewriter::processParamConstantOp(ParamConstantOp op) {
  // ParamConstantOp projects a parameter expression into an SSA value.  We can
  // eventually lower this into lower level operators in the target set, but
  // for now we just simplify their operand.
  auto errorOrValue = getEvaluator().concretizeParameterExpr(op.getValue());
  if (errorOrValue.isError())
    return error(op->getLoc(), errorOrValue.takeError());

  op.setValueAttr(errorOrValue.takeValue());
  return success();
}

LogicalResult ParameterRewriter::processParamAssertOp(ParamAssertOp op) {
  // Check the condition expression.
  auto errorOrValue = getEvaluator().concretizeParameterExpr(op.getCond());
  if (errorOrValue.isError())
    return error(op->getLoc(), errorOrValue.takeError());

  auto resultInt = dyn_cast<IntegerAttr>(errorOrValue.get());
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
/// concrete constants. This reports the error and returns null on failure,
/// and returns an array of bound input parameters on success.
FailureOr<std::pair<ArrayAttr, bool>> ParameterRewriter::resolveCallInputParams(
    Operation *call, ArrayRef<ParamBindAttr> inputValues, bool isInlinedCall) {
  // The region to use for the next "region" input parameter.
  size_t nextRegionInThisCall = 0;

  SmallVector<Attribute> boundInputParams;
  bool inlineCallee = false;
  for (ParamBindAttr param : inputValues) {
    // If this is a region reference, form a binding to the region provided by
    // the call.
    if (auto regionRef = dyn_cast<ParamCallRegionRefAttr>(param.getValue())) {
      Region &region = call->getRegion(nextRegionInThisCall++);

      // Give this reference a unique name, and make a RegionReferenceAttr with
      // the name and SignatureType.
      auto regionRefAttr =
          RegionReferenceAttr::get(elaboratedGenerator.func.getName() +
                                       "_region_" + Twine(nextCallRegionID++),
                                   regionRef.getType());

      // The region in question should only have a single RegionBodyOp
      // operation.  Take it out and hand ownership to the elaborator so
      // references to it get correctly resolved.
      assert(region.hasOneBlock());
      Block &regionBlock = *region.begin();
      auto body = cast<KGENDeclInterface>(&regionBlock.front());
      // Remove the RegionBodyOp from the call's region, and hand ownership of
      // it to the elaborator.
      body->remove();

      // TODO: We could do some content hashing to avoid making a new name for
      // a lexically identical body.  This would reduce some redundant
      // specialization.
      Elaborator::RegionBody regionBody{
          body,
          ParameterEvaluator(symtab, getEvaluator().getParameterValues())};
      inlineCallee |= !regionBody.isIsolatedFromAbove();
      elaborator.addRegionReference(regionRefAttr, std::move(regionBody));
      boundInputParams.push_back(regionRefAttr);
      continue;
    }

    // Otherwise fold the parameter expression in this context to a simple
    // constant.
    auto value = getEvaluator().concretizeParameterExpr(param.getValue());
    if (value.isError())
      return error(call->getLoc(), value.takeError());

    // Check if we have a reference to a non-isolated region.
    auto regionRef = dyn_cast<RegionReferenceAttr>(value.get());
    if (regionRef &&
        !elaborator.getRegionReferenced(regionRef).isIsolatedFromAbove()) {
      // Only the callees of inlined calls can be regions not isolated from
      // above.
      if (!isInlinedCall)
        return error(
            call->getLoc(),
            "non-inlined call to symbol instantiated with non-isolated region");
      // Indicate that the callee will always be inlined.
      inlineCallee = true;
    }

    boundInputParams.push_back(value.takeValue());
  }
  return std::make_pair(ArrayAttr::get(call->getContext(), boundInputParams),
                        inlineCallee);
}

LogicalResult ParameterRewriter::processGeneratorUserImpl(
    Operation *user, ArrayRef<ParamDeclAttr> decls,
    ArrayRef<ParamBindAttr> paramValues, FlatSymbolRefAttr calleeAttr,
    SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriters,
    bool isInlinedCall) {
  // Evaluate any input parameters.
  FailureOr<std::pair<ArrayAttr, bool>> result =
      resolveCallInputParams(user, paramValues, isInlinedCall);
  if (failed(result))
    return failure();
  auto [inputParamKey, inlineCallee] = *result;

  // Instantiate the callee into one or more FuncOp's, depending on what the
  // callee is.
  auto callee = dyn_cast_if_present<KGENDeclInterface>(
      elaborator.lookupCallee(calleeAttr));
  if (!callee)
    return error(user->getLoc(), Twine("could not find callee '") +
                                     calleeAttr.getLeafReference().strref() +
                                     "'");

  // If the callee is an interface that provides an evaluator, resolve the
  // evaluator first.
  if (auto itf = dyn_cast<GeneratorInterfaceOp>(*callee))
    if (SymbolConstantAttr evaluator = itf.getEvaluatorAttr())
      if (failed(processGeneratorUserImpl(
              itf, {}, evaluator.getParamValues().getValue(),
              evaluator.getSymbol(), rewriters, /*isInlinedCall=*/false)))
        return failure();

  DeclAndInputParamsPair calleeDeclAndInputParams{callee, inputParamKey};

  // If we already have a binding for this decl/inputParam set, then reuse the
  // consistent callee.
  if (FuncOp callee =
          elaboratedGenerator.getBinding(calleeDeclAndInputParams)) {
    completeGeneratorUserProcessing(user, decls, calleeDeclAndInputParams,
                                    ElaboratedGenerator(callee));
    return success();
  }

  // Otherwise, this is our first use of this.  Ask the global elaborator for
  // the full set of candidates.
  ArrayRef<ElaboratedGeneratorOrCalleeError> newCalleesRef =
      elaborator.getAllInstantiations(calleeDeclAndInputParams, inlineCallee);

  // Copy the list of funcs instead of referring to the cache entry to avoid
  // iterator invalidation problems.
  SmallVector<ElaboratedGeneratorOrCalleeError> newCallees(
      newCalleesRef.begin(), newCalleesRef.end());

  // If we found more than one callee to produce then we need to spawn
  // multiple versions of the func we are currently constructing, each
  // which get a different callee.
  ElaboratedGenerator thisCallee(/*func=*/nullptr);
  for (const ElaboratedGeneratorOrCalleeError &candidate : newCallees) {
    // Ignore erroneous callees.
    if (std::holds_alternative<CalleeExpansionError>(candidate))
      continue;
    // Ignore the candidate if the elaborated func is inconsistent with our
    // current bindings.
    const ElaboratedGenerator &calleeCandidate =
        std::get<ElaboratedGenerator>(candidate);
    if (!calleeCandidate.isConsistentWith(elaboratedGenerator))
      continue;

    // If this is the first viable candidates, then we will pursue it locally.
    if (!thisCallee.func) {
      thisCallee = calleeCandidate;
    } else if (auto itf = dyn_cast<GeneratorInterfaceOp>(user)) {
      // Prohibit interface evaluators from having multiple implementations.
      return error(itf.getLoc(),
                   Twine("interface @") + itf.getSymName() +
                       " evaluator must resolve to a single implementation");
    } else {
      // All other callees gets spawned as sub-evaluators.
      spawnNewFuncClone(user, decls, calleeDeclAndInputParams, calleeCandidate,
                        rewriters);
    }
  }

  // If all the expansions failed, then this call fails overall.
  if (!thisCallee.func) {
    SmallVector<CalleeExpansionError> errors;
    for (const auto &value : newCallees)
      errors.push_back(std::get<CalleeExpansionError>(value));
    return errorCalling(user->getLoc(), errors);
  }

  // Finally, we can handle the first viable one as our continued progress here.
  completeGeneratorUserProcessing(user, decls, calleeDeclAndInputParams,
                                  thisCallee);
  return success();
}

void ParameterRewriter::completeGeneratorUserProcessing(
    Operation *user, ArrayRef<ParamDeclAttr> decls,
    DeclAndInputParamsPair calleeAndInputParams,
    const ElaboratedGenerator &newCallee) {
  // Add a binding to remember that we resolved this call to this candidate,
  // and merge any bindings from it into our set.
  elaboratedGenerator.addBinding(calleeAndInputParams, newCallee);

  FuncOp newCalleeFunc = newCallee.func;

  // Resolve any bound result types.
  SmallVector<Type> resultTypes;
  for (auto result : user->getResultTypes())
    resultTypes.push_back(getEvaluator().getReboundType(result));

  // Now that we resolved the call to a new thing, build a new call to replace
  // the old one.
  mlir::IRRewriter b{OpBuilder(user)};
  if (isa<CallOp, CallParamOp>(user)) {
    b.replaceOpWithNewOp<CallOp>(user, resultTypes, newCalleeFunc.getNameAttr(),
                                 ArrayRef<ParamBindAttr>(),
                                 ArrayRef<ParamDeclAttr>(),
                                 user->getOperands());

  } else if (isa<AddressOfOp>(user)) {
    b.replaceOpWithNewOp<AddressOfOp>(
        user, resultTypes.front(), newCalleeFunc.getNameAttr(),
        ArrayRef<ParamBindAttr>(), ArrayRef<ParamDeclAttr>());

  } else if (isa<InlinedCallOp>(user)) {
    // Inline the callee.
    BlockAndValueMapping bv;
    for (auto [operand, argument] :
         llvm::zip(newCalleeFunc.getArguments(), user->getOperands()))
      bv.map(operand, argument);
    for (Operation &op : newCalleeFunc.getBody()->without_terminator())
      b.clone(op, bv);
    Operation *terminator =
        b.clone(*newCalleeFunc.getBody()->getTerminator(), bv);
    b.replaceOp(user, terminator->getOperands());
    terminator->erase();

  } else {
    // Update the interface in-place.
    auto itf = cast<GeneratorInterfaceOp>(user);
    itf.setEvaluatorAttr(SymbolConstantAttr::get(
        FlatSymbolRefAttr::get(newCalleeFunc.getSymNameAttr()),
        NoneType::get(itf.getContext())));
  }

  // Bind the result parameters to the output parameter decls.
  for (auto [decl, bindValue] :
       llvm::zip(decls, elaborator.lookupResultParameters(newCallee.func)))
    getEvaluator().setOrOverwriteParameterValue(decl, bindValue);
}

/// Sometimes when we expand a call, we find that there are multiple viable
/// callees that we can generate.  We handle this by spawning new parameter
/// rewriters with state copied from the current one, but which resolve the call
/// to different callees.  This spawns a new rewriter with the specified call
/// resolving to the specified callee.
void ParameterRewriter::spawnNewFuncClone(
    Operation *user, ArrayRef<ParamDeclAttr> decls,
    DeclAndInputParamsPair calleeAndInputParams,
    const ElaboratedGenerator &callee,
    SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriters) {
  // Start by cloning the current WIP func to a new copy of it.
  BlockAndValueMapping blocksAndValues;
  DenseMap<Operation *, Operation *> operationMap;
  auto newFunc = cast<FuncOp>(
      cloneOperation(elaboratedGenerator.func, blocksAndValues, operationMap));

  // Insert the func into the output file and auto-unique the symbol.
  elaborator.insertFuncVariant(elaboratedGenerator.func, newFunc);

  // If the duplicated callee referenced a region not isolated from above, then
  // we need to remap any values that escaped to the cloned function.
  auto isNonIsolatedRegionRef = [&](Attribute attr) {
    if (auto regionRef = dyn_cast<RegionReferenceAttr>(attr))
      return !elaborator.getRegionReferenced(regionRef).isIsolatedFromAbove();
    return false;
  };
  if (llvm::any_of(calleeAndInputParams.second, isNonIsolatedRegionRef)) {
    callee.func->walk([&](Operation *op) {
      for (OpOperand &operand : op->getOpOperands())
        if (Value remapped = blocksAndValues.lookupOrNull(operand.get()))
          operand.set(remapped);
    });
  }

  // Generate the new rewriter which will process this.
  auto &newRewriter =
      rewriters.emplace_back(new ParameterRewriter(*this, operationMap));

  // Change the future of this func by resolving the call in the new func to
  // the specifed callee.
  Operation *newUser = operationMap[user];
  newRewriter->completeGeneratorUserProcessing(newUser, decls,
                                               calleeAndInputParams, callee);
}

LogicalResult ParameterRewriter::processCallParamOp(
    CallParamOp call,
    SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriters) {
  // Simplify the callee expression.
  auto errorOrValue = getEvaluator().concretizeParameterExpr(call.getCallee());
  if (errorOrValue.isError())
    return error(call->getLoc(), errorOrValue.takeError());

  // If the parameter expression is resolved to a symbol, then treat this like a
  // direct call.
  if (auto symbolCst = dyn_cast<SymbolConstantAttr>(errorOrValue.get())) {
    // If there are no bound parameters on the call, use the one on the
    // CallParam.  TODO: Remove.
    if (symbolCst.getParamValues().empty())
      return processGeneratorUserImpl(
          call, call.getParamDecls(), call.getParamValues(),
          symbolCst.getSymbol(), rewriters, /*isInlinedCall=*/false);
    // Otherwise use the ones from the symbol.
    return processGeneratorUserImpl(
        call, call.getParamDecls(), symbolCst.getParamValues().getValue(),
        symbolCst.getSymbol(), rewriters, /*isInlinedCall=*/false);
  }

  // Otherwise, the only other case we support is a call to a region, which is
  // marked with a StringAttr value that has signature type.
  auto regionName = errorOrValue.get().cast<RegionReferenceAttr>();
  return processRegionCallImpl(call, regionName, call.getParamDeclsAttr(),
                               call.getParamValues());
}

LogicalResult ParameterRewriter::processInlinedCallOp(
    InlinedCallOp call,
    SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriters) {
  // Simplify the callee expression.
  auto callee = getEvaluator().concretizeParameterExpr(call.getCallee());
  if (callee.isError())
    return error(call->getLoc(), callee.takeError());

  // If the parameter expression is resolved to a region, then it is inlined
  // anyways.
  if (auto regionName = dyn_cast<RegionReferenceAttr>(callee.get()))
    return processRegionCallImpl(call, regionName, call.getParamDeclsAttr(),
                                 call.getParamValues());

  auto symbolCst = cast<SymbolConstantAttr>(callee.get());
  // If there are no bound parameters on the call, use the one on the
  // CallParam.  TODO: Remove.
  if (symbolCst.getParamValues().empty())
    return processGeneratorUserImpl(
        call, call.getParamDecls(), call.getParamValues(),
        symbolCst.getSymbol(), rewriters, /*isInlinedCall=*/true);
  return processGeneratorUserImpl(
      call, call.getParamDecls(), symbolCst.getParamValues().getValue(),
      symbolCst.getSymbol(), rewriters, /*isInlinedCall=*/true);
}

LogicalResult ParameterRewriter::processRegionCallImpl(
    KGENCallOpInterface call, RegionReferenceAttr regionName,
    ParamDeclArrayAttr decls, ArrayRef<ParamBindAttr> paramValues) {
  assert(regionName.getType().isa<SignatureType>() && "not a region reference");
  const Elaborator::RegionBody &regionBody =
      elaborator.getRegionReferenced(regionName);
  KGENDeclInterface region = regionBody.body;

  // Compute the binding of input parameters to concrete values.
  FailureOr<std::pair<ArrayAttr, bool>> result =
      resolveCallInputParams(call, paramValues, /*isInlinedCall=*/true);
  if (failed(result))
    return failure();
  auto [inputParamKey, _] = *result;

  auto theRegionReturnOp =
      cast<ReturnOp>(region->getRegion(0).front().getTerminator());

  // Push a region return command that contains the output parameters to bind
  // after processing all the operations in the region.
  commandWorklist.push_back(
      new RegionReturn{call->getLoc(), theRegionReturnOp.getLoc(),
                       theRegionReturnOp.getParametersAttr(), decls});

  // Nested regions will have a different parameter namespace than the caller
  // context: names will mean different things inside the region than they did
  // in the caller.  To handle this, we push a ParameterEvaluator scope that
  // represents the bindings within the region body and set the region return to
  // restore back to the previous scope when operations from the region have
  // finished their processing.
  ParameterEvaluator &evaluator = evaluators.emplace_back(regionBody.evaluator);

  // Add bindings for each of the input parameters to the new scope we just
  // pushed, so they are properly bound when the rewriter continues processing
  // the newly cloned operations.
  for (auto [decl, value] :
       llvm::zip(region.getInputParamDecls(), inputParamKey))
    evaluator.setOrOverwriteParameterValue(decl, value);

  auto emitEvaluateConstraintsError = [&](Location loc, Error message) {
    return error(loc, std::move(message));
  };

  // Evaluate any constraints for this declaration to see if this is a viable
  // expansion.  If not, the expansion fails. Only isolated regions have
  // constraints.
  if (failed(evaluateConstraints(region.getConstraintsAttr(), evaluator,
                                 emitEvaluateConstraintsError)))
    return failure();

  // We process the call to the region by cloning its body inline, replacing
  // the call with the newly substituted operations.  While doing this, we
  // need to remap the region's arguments to the call formal parameters.
  BlockAndValueMapping mapper;
  DenseMap<Operation *, Operation *> operationMap;
  Block &bodyBlock = region->getRegion(0).front();
  for (auto [arg, value] :
       llvm::zip(bodyBlock.getArguments(), call->getOperands()))
    mapper.map(arg, value);

  // Clone all of the operations in the block.
  OpBuilder b(call);
  for (auto &bodyOp : bodyBlock)
    b.insert(cloneOperation(&bodyOp, mapper, operationMap));

  // Lookup all the parameter decls and uses in the body of region, we will
  // visit all of them as the evaluator continues processing the ops we just
  // cloned over.
  const ParameterDeclsAndUses *uses = elaborator.lookupNestedUses(region);
  // If we didn't get a cache hit, recompute the uses.
  Optional<ParameterDeclsAndUses> recomputedUses;
  if (!uses) {
    recomputedUses.emplace();
    uses = &recomputedUses.value();
    // Fold the current parameter scope into the declarations. Make the region's
    // parent the virtual root.
    for (auto [name, value] : evaluator.getParameterValues()) {
      recomputedUses->decls.try_emplace(
          name, std::make_pair(region->getParentOp(),
                               ParamDeclAttr::get(
                                   name, cast<TypedAttr>(value).getType())));
    }
    elaborator.takeNestedParameterUses(recomputedUses->calculate(region));
  }

  // Add the parameter-using operations we cloned over from the region to
  // the commandWorklist so we rewrite them.
  auto remapOp = [&](Operation *op) {
    // We don't clone over the region op itself, and will be deleting the
    // return soon though so ignore those.
    if (op == region || op == theRegionReturnOp)
      return;
    commandWorklist.push_back(operationMap[op]);
    assert(commandWorklist.back() && "operation wasn't cloned over correctly?");
  };
  for (Operation *op : uses->constExprOps)
    remapOp(op);
  for (auto &[op, _] : llvm::reverse(uses->usersAndDeclarers))
    remapOp(op);

  // Now that we've cloned all the operations over, we know what the SSA
  // results are supposed to be.  Replace all the uses of the call results
  // with them.
  auto clonedReturn = cast<ReturnOp>(operationMap[theRegionReturnOp]);
  SmallVector<Value> newResults;
  llvm::append_range(newResults, clonedReturn.getOperands());
  clonedReturn->erase();

  // The SSA results of the old call go directly to the new call and remove
  // it.
  call->getResults().replaceAllUsesWith(newResults);
  call->erase();
  return success();
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
                 "unknown parameter-defining operator in elaboration");

  // Scan all the attributes and types to look for uses of parameters.  We let
  // the walker scan the region hierarchy.
  SmallVector<NamedAttribute> newAttrs;
  bool changedAttrs = false;
  for (const NamedAttribute &namedAttr : op->getAttrs()) {
    // Preserve but ignore the 'paramDecls' attribute on FuncOp.
    if (namedAttr.getName() == "paramDecls") {
      newAttrs.push_back(namedAttr);
      continue;
    }

    newAttrs.push_back(NamedAttribute(
        namedAttr.getName(),
        getEvaluator().getReboundAttribute(namedAttr.getValue())));
    changedAttrs |= namedAttr.getValue() != newAttrs.back().getValue();
  }
  if (changedAttrs)
    op->setAttrs(newAttrs);

  // Check the types of results to find any parameters embedded in their
  // types.  We don't have to check operands because they are always checked
  // when being defined.
  for (OpResult result : op->getResults())
    result.setType(getEvaluator().getReboundType(result.getType()));

  // Scan the region list if present.  The walker will automatically recurse
  // for us, but we have to check the block arguments.
  if (op->getNumRegions()) { // Microoptimization: getRegions() is slow.
    for (auto &region : op->getRegions())
      for (auto &block : region)
        for (Value arg : block.getArguments())
          arg.setType(getEvaluator().getReboundType(arg.getType()));
  }

  // If the op implements the elaborator interface, indicate it as resolved.
  if (auto elaboratorIface = dyn_cast<ElaboratorOpInterface>(op)) {
    ErrorOrSuccess result = elaboratorIface.finalizeElaboration();
    if (result.isError())
      return error(op->getLoc(), result.takeError());
  }

  return success();
}

/// If elaboration of this func fails, then the client can get the error
/// out.  This also deallocates the body of the dead husk of the func which
/// may not even verify correctly, it will be removed later.
CalleeExpansionError ParameterRewriter::takeDiagnosticAndEraseFunc() {
  assert(diagnostic.has_value() &&
         "cannot get diagnostic when none was generated");
  // The generator is not viable so we need to delete it.  This op can appear
  // in various maps though, so instead of actually deleting it, we just
  // mark it for removal later.
  elaborator.markFuncForRemoval(elaboratedGenerator.func);
  auto error = CalleeExpansionError(elaboratedGenerator.func->getLoc(),
                                    std::move(diagnostic.value()));

  // Check to see if the error occurs in the scope of a call_param to a region.
  // If so, make sure to add the nested call to the expansion path.
  for (auto command : llvm::reverse(commandWorklist)) {
    RegionReturn *rr = dyn_cast<RegionReturn *>(command);
    if (!rr)
      continue;
    error = CalleeExpansionError(rr->callLoc,
                                 ElaborationDiagnostic(rr->callLoc, error));
  }

  return error;
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
    return b.getStringAttr(generator.getName() + "_concrete");

  std::string result;
  llvm::raw_string_ostream os(result);
  os << generator.getName();

  auto inputParamDecls = generator.getParamDeclsAttr();
  for (auto [inputDecl, value] : llvm::zip(inputParamDecls, inputParamValues)) {
    os << ',' << inputDecl.getName().str() << '=';

    if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
      os << intAttr.getValue();
    } else if (auto floatAttr = dyn_cast<FloatAttr>(value)) {
      SmallString<32> str;
      floatAttr.getValue().toString(str);
      os << str;
    } else if (auto dtypeAttr = dyn_cast<DTypeConstantAttr>(value)) {
      os << dtypeAttr.getDType();
    } else if (auto typeConstant = dyn_cast<ConcreteTypeConstantAttr>(value)) {
      // NOTE: Could use pretty mangling for common cases, e.g. "simd2xf32" or
      // something if these get too verbose.
      os << typeConstant.getValue();
    } else if (auto symbolConstant = dyn_cast<SymbolConstantAttr>(value)) {
      os << symbolConstant.getName().getValue();
    } else if (auto stringConstant = dyn_cast<StringAttr>(value)) {
      os << stringConstant.strref();
    } else {
      assert(!isSimpleConstant(value) && "not handling all simple constants");
      os << "??";
    }
  }
  return b.getStringAttr(result);
}

/// Specialize a func body, generating one variant of each viable
/// instantiation of that body.  funcs do not have parameters, but they can
/// invoke interfaces etc which can cause them to produce multiple variants.
SmallVector<ElaboratedGeneratorOrCalleeError>
Elaborator::specializeFunc(FuncOp func, ModuleOp sourceModule, bool inlined) {
  // Get a partial ordering of parameter definitions and uses that is listed
  // "top down" in our evaluation order.
  ParameterDeclsAndUses uses;
  takeNestedParameterUses(uses.calculate(func));
  SmallVector<Operation *> opsToRewrite;

  // Rewrite all the parameter-using ops in this scope only. We are going to use
  // opsToRewrite as a worklist, so reverse it for efficient pop_back.
  opsToRewrite.reserve(uses.constExprOps.size() +
                       uses.usersAndDeclarers.size());
  for (auto &[op, _] : llvm::reverse(uses.usersAndDeclarers))
    opsToRewrite.push_back(op);
  // Rewrite ops with only constant parameter expressions too.
  llvm::append_range(opsToRewrite, uses.constExprOps);

  // Start by rewriting this func. Use `unique_ptr` for the stack to prevent
  // invalidation.
  SmallVector<std::unique_ptr<ParameterRewriter>> rewriterWorklist;
  rewriterWorklist.emplace_back(new ParameterRewriter(
      *this, func, symtab, std::move(opsToRewrite), inlined));

  // Rewriting funcs may generate other func clones.  If so, rewrite them,
  // until we converge.
  SmallVector<ElaboratedGeneratorOrCalleeError> results;
  while (!rewriterWorklist.empty()) {
    std::unique_ptr<ParameterRewriter> rewriter =
        rewriterWorklist.pop_back_val();

    // If elaborating the func succeeded, then we have a viable candidate.
    if (succeeded(rewriter->rewriteOps(rewriterWorklist))) {
      // Take the result parameters from the rewritten function and bind it in
      // the elaborator.
      ElaboratedGenerator result = rewriter->takeElaboratedGenerator();
      bindResultParameters(result.func);
      results.push_back(std::move(result));

    } else {
      // If elaborating the func fails, then remember the diagnostic (in case
      // we need to explain why elaboration fails) and remove the broken husk of
      // a func that didn't make it.
      results.push_back(rewriter->takeDiagnosticAndEraseFunc());
    }
  }
  return results;
}

/// Specialize a generator with the specified input parameters and return the
/// symbol name to use for the result, along with an array of ParamBindAttrs for
/// the result attributes.
SmallVector<ElaboratedGeneratorOrCalleeError>
Elaborator::specializeGenerator(DeclAndInputParamsPair declAndInputParams,
                                bool inlined) {
  auto generator = cast<GeneratorOp>(declAndInputParams.first);

  ArrayRef<Attribute> inputParamValues = declAndInputParams.second.getValue();
  auto inputParamDecls = generator.getInputParamDecls();
  assert(inputParamValues.size() == inputParamDecls.size() &&
         "incorrect # input parameter values");

  // TODO (low prio): Some day we could mangle "instantiated from here"
  // information into the location.
  OpBuilder b(generator);
  auto newFunc = b.create<FuncOp>(
      generator.getLoc(), mangleParameterValues(generator, inputParamValues),
      generator.getFunctionType(), generator.getResultParamTypes());

  // Insert the newFunc into the symbol table which will then know about it,
  // but it will also auto-rename the symbol for us in the case of conflicts.
  symtab.insert(newFunc);

  // Clone the body of the generator over.
  BlockAndValueMapping mapper;
  generator.getBodyRegion().cloneInto(&newFunc.getBodyRegion(), mapper);

  // Provide definitions of the input parameters in the body block as bound
  // constants.
  b.setInsertionPoint(&newFunc.getBody()->front());
  for (auto [inputDecl, inputValue] :
       llvm::zip(inputParamDecls, inputParamValues)) {
    b.create<ParamDeclareOp>(generator.getLoc(), inputDecl, inputValue);
  }

  // Now that we have a new synthesized generic func, run the rewriter
  // over it to specialize its body.
  auto sourceModule = generator->getParentOfType<ModuleOp>();
  auto result = specializeFunc(newFunc, sourceModule, inlined);

  // If the generator had no parameters, then we want to reuse the same name as
  // the original generator.  We can't do that when we are building the concrete
  // version though because we may have other calls to the generator and those
  // calls get linked to the generator by their symbol.  Additionally,
  // elaboration of any candidate could fail.
  //
  // To handle this, we let the symbol table autorename it, but keep track of
  // the first successful implementation in a map.  We rename it back after the
  // module has finished elaboration.
  if (inputParamValues.empty()) {
    for (auto &candidate : result) {
      if (std::holds_alternative<ElaboratedGenerator>(candidate)) {
        firstConcreteFuncForGenerator.insert(
            {generator, std::get<ElaboratedGenerator>(candidate).func});
        break;
      }
    }
  }

  return result;
}

/// Specialize a generator interface with the specified input parameters and
/// return the generated func.
SmallVector<ElaboratedGeneratorOrCalleeError>
Elaborator::specializeInterface(DeclAndInputParamsPair declAndInputParams,
                                bool inlined) {
  auto itf = cast<GeneratorInterfaceOp>(declAndInputParams.first);
  SmallVector<ElaboratedGeneratorOrCalleeError> result;

  // An interface is an abstraction over multiple generators.  Invoke each of
  // them, collecting the results together into a single result.
  ArrayRef<GeneratorOp> interfaceImpls = getGeneratorsImplementing(itf);
  if (interfaceImpls.empty()) {
    // If we found no implementations, report that problem at the call site as
    // a single diagnostic.
    result.push_back(reportCalleeExpansionError(
        itf, "no implementations of interface '" + itf.getName() + "' found"));
    return result;
  }

  // If a default has been provided and we don't want to do search, then use it.
  Optional<SymbolConstantAttr> defaultImpl = itf.getDefaultImpl();
  if (!enableSearch && defaultImpl.has_value()) {
    // If the SymbolConstant exists, then the callee must exist.
    Operation *defaultImplCallee = lookupCallee(defaultImpl->getSymbol());
    assert(defaultImplCallee != nullptr && "expected defaultImpl to exist");
    // The default impl must be a generator.
    GeneratorOp gen = cast<GeneratorOp>(defaultImplCallee);
    auto funcs =
        getAllInstantiations({gen, declAndInputParams.second}, inlined);
    result.append(funcs.begin(), funcs.end());
    return result;
  }

  for (GeneratorOp gen : interfaceImpls) {
    // Make sure to go through getAllInstantiations so generators are cached
    // and any constraints on the generator itself are validated.
    auto funcs =
        getAllInstantiations({gen, declAndInputParams.second}, inlined);
    result.append(funcs.begin(), funcs.end());
  }

  // If all the results are expansion errors, return them to the caller which
  // will cause elaboration to fail.
  if (llvm::all_of(result, [](ElaboratedGeneratorOrCalleeError kOr) {
        return std::holds_alternative<CalleeExpansionError>(kOr);
      }))
    return result;

  // Move the expansion errors to the end of the vector.
  auto newEnd =
      llvm::remove_if(result, [&](ElaboratedGeneratorOrCalleeError funcOr) {
        return std::holds_alternative<CalleeExpansionError>(funcOr);
      });

  // Only one successful elaboration, we don't have to search, just return it.
  if (newEnd == result.begin() + 1)
    return {*result.begin()};

  SymbolConstantAttr evaluator = itf.getEvaluatorAttr();

  // Truncate the result vector to contain only the successful implementations.
  result.erase(newEnd, result.end());

  // If we don't want to do search, we're done.
  if (!enableSearch)
    return {*result.begin()};

  // If there is no evaluator, return the full vector of instantiations. If the
  // interface is being inlined, we can't benchmark the instantiations because
  // they would not be well-formed.
  if (!evaluator || inlined)
    return result;

  // Pull out the elaboration results that succeeded to provide to the search
  // inputs.
  SmallVector<FuncOp> searchInputs;
  for (const auto &r : result)
    searchInputs.push_back(std::get<ElaboratedGenerator>(r).func);

  auto evalFunc = cast<FuncOp>(lookupCallee(evaluator.getSymbol()));

  ErrorOr<size_t> bestSpecializationIdxOr =
      evaluateSpecializations(evalFunc, symtab, searchInputs);
  if (failed(bestSpecializationIdxOr))
    return {
        reportCalleeExpansionError(itf, bestSpecializationIdxOr.getError())};

  // Find the fastest one and return just that one.
  return {std::move(result[*bestSpecializationIdxOr])};
}

/// Return all instantiations of the specified declaration (a  generator or
/// interface) with the specified input parameter values.
ArrayRef<ElaboratedGeneratorOrCalleeError>
Elaborator::getAllInstantiations(DeclAndInputParamsPair declAndInputParams,
                                 bool inlined) {
  // Check the global cache of instantiations so we only ever instantiate a
  // generator once.
  auto cacheIt = generatedFuncs.find(declAndInputParams);
  if (cacheIt != generatedFuncs.end())
    return cacheIt->second;

  Operation *decl = declAndInputParams.first;
  SmallVector<ElaboratedGeneratorOrCalleeError> newCallees;
  auto localError = [&](Location loc, Error err) {
    newCallees.push_back(CalleeExpansionError(
        decl->getLoc(), ElaborationDiagnostic(loc, std::move(err))));
    return failure();
  };

  // Evaluate any constraints for this declaration to see if this is a viable
  // expansion.  If not, the expansion fails.
  if (failed(evaluateConstraints(decl, declAndInputParams.second.getValue(),
                                 localError, symtab))) {
    /* nothing */
  } else if (auto func = dyn_cast<FuncOp>(decl)) {
    // Nothing to do here. Just bind the result parameters and return the
    // function.
    bindResultParameters(func);
    newCallees.emplace_back(ElaboratedGenerator(func));
  } else if (isa<GeneratorOp>(decl)) {
    newCallees = specializeGenerator(declAndInputParams, inlined);
  } else if (isa<GeneratorInterfaceOp>(decl)) {
    newCallees = specializeInterface(declAndInputParams, inlined);
  } else {
    (void)localError(decl->getLoc(), "call to an unknown kind of declaration");
  }

  return generatedFuncs[declAndInputParams] = std::move(newCallees);
}

//===----------------------------------------------------------------------===//
// Elaborator Driver
//===----------------------------------------------------------------------===//

/// Scan the primary and library modules to collect all the interfaces,
/// verifying that any common interfaces are the same.
ParseResult Elaborator::collectInterfaces() {
  // Collect all the generator interfaces in the library modules, which will
  // allow cross-checking them below. Also, collect all the generators
  // that implement a given interface, starting with the libraries.  These will
  // already have been type checked within the library.
  DenseMap<StringAttr, GeneratorInterfaceOp> libraryInterfaces;

  // Scan the specified module collecting all the generators that implement an
  // interface and checking the interfaces between library files line up.
  for (Operation &op : cast<ModuleOp>(symtab.getOp()).getOps()) {
    // Collect interfaces.
    if (auto itf = dyn_cast<GeneratorInterfaceOp>(op)) {
      if (auto [it, inserted] =
              libraryInterfaces.insert({itf.getNameAttr(), itf});
          inserted)
        continue;
    }

    // If this is a generator, keep track of it.
    if (auto generator = dyn_cast<GeneratorOp>(op))
      if (auto interface = generator.getImplementsAttr())
        interfaceImpls[interface.getAttr()].push_back(generator);

    // Detect common errors cleanly, and report it.
    if (op.getName().getStringRef() == "lit.func")
      return op.emitError("unlowered lit.func discovered in KGEN elaborator");
  }
  return success();
}

namespace {
class RecursionChecker {
public:
  RecursionChecker(Elaborator &elaborator) : elaborator(elaborator) {}
  ParseResult verify(ArrayRef<GeneratorOp> primaryGenerators);

private:
  ParseResult checkRecursively(Operation *op);

  Elaborator &elaborator;
  llvm::SetVector<Operation *> callStack;
  std::vector<Operation *> callStackCalls;
  SmallPtrSet<Operation *, 8> alreadyCheckedOps;
};
} // namespace

/// Check the specified operation and its call-tree by visiting callees in
/// depth-first order.  If we report errors the call-graph, and keep track of
/// known-ok operators to avoid redundant work.
ParseResult RecursionChecker::checkRecursively(Operation *op) {
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
  op->walk([&](Operation *op) {
    FlatSymbolRefAttr calleeAttr;
    if (auto call = dyn_cast<CallOp>(op)) {
      calleeAttr = call.getCalleeAttr();
    } else if (auto addressof = dyn_cast<AddressOfOp>(op)) {
      calleeAttr = addressof.getCalleeAttr();
    } else if (auto itf = dyn_cast<GeneratorInterfaceOp>(op)) {
      if (SymbolConstantAttr evaluator = itf.getEvaluatorAttr())
        calleeAttr = evaluator.getSymbol();
      else
        return;
    } else {
      return;
    }

    auto callee = elaborator.lookupCallee(calleeAttr);
    assert(callee && "couldn't resolve callee?");
    callStackCalls.push_back(op);
    if (isa<FuncOp, GeneratorOp>(callee)) {
      // For direct calls, we immediately check the callee.
      if (checkRecursively(callee))
        failed = true;
    } else if (auto itf = dyn_cast<GeneratorInterfaceOp>(callee)) {
      // Check that interface evaluator doesn't lead to a cycle.
      if (checkRecursively(itf))
        failed = true;

      // For generator interfaces, we resolve to all the implementations.
      for (auto gen : elaborator.getGeneratorsImplementing(itf)) {
        // Make sure we keep track of the current module we're scanning.
        // TODO: This recursively checks all code in the imported libraries.
        // Will this be needed when these are individually checked on their own?
        if (checkRecursively(gen))
          failed = true;
      }
    } else {
      op->emitError("unknown callee in elaboration");
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

ParseResult RecursionChecker::verify(ArrayRef<GeneratorOp> primaryGenerators) {
  // Check all the operations at the top level of the primary module.
  for (GeneratorOp gen : primaryGenerators)
    if (checkRecursively(gen))
      return failure();
  return success();
}

/// When a top-level generator failed to elaborate, this is used to recursively
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

/// Elaborate generators in the specified module, incorporating implementation
/// logic from the specified library.
LogicalResult M::elaborateGenerators(SymbolTable &symtab,
                                     ArrayRef<GeneratorOp> primaryGenerators,
                                     bool enableSearch) {
  TimeTraceScope<> traceScope("elaborate-generators");
  auto primary = cast<ModuleOp>(symtab.getOp());
  Elaborator elaborator(symtab, enableSearch);

  // Scan the primary and library module to collect all the interfaces,
  // verifying that any common interfaces are the same.
  if (elaborator.collectInterfaces())
    return failure();

  // Check the generator call graph to reject any recursion.
  RecursionChecker checker(elaborator);
  if (failed(checker.verify(primaryGenerators)))
    return failure();

  auto emptyInputParamKey = ArrayAttr::get(primary.getContext(), {});

  // Elaborate the bodies of all generators without input parameters in the
  // primary module.  These are roots that will cause callees to get
  // recursively elaborated.
  // TODO: When we have access control, we can limit this to just the publicly
  // exposed ones.
  bool didFail = false;
  for (GeneratorOp generatorRoot : primaryGenerators) {

    // Elaborate the generator into concrete versions.
    ArrayRef<ElaboratedGeneratorOrCalleeError> results =
        elaborator.getAllInstantiations({generatorRoot, emptyInputParamKey});

    // If the generator failed to expand into /anything/ then emit an error.
    // Note that the func will have been deleted.
    if (llvm::all_of(
            results,
            [](const ElaboratedGeneratorOrCalleeError &result) -> bool {
              return std::holds_alternative<CalleeExpansionError>(result);
            })) {
      // Collect the errors together.
      SmallVector<CalleeExpansionError> errors;
      for (const auto &value : results)
        errors.push_back(std::get<CalleeExpansionError>(value));
      auto diag = emitError(errors[0].first, "no viable implementations found");
      emitElaborationError(diag, errors, /*indentDepth=*/2);
      didFail = true;
    }
  }

  // If we failed to expand any funcs, propagate that failure.
  if (didFail)
    return failure();

  // After removing all the generators, we'll rename any direct
  // implementations of them to use their name.
  DenseMap<StringAttr, StringAttr> funcsToRename;
  for (auto [generator, func] : elaborator.getFirstConcreteFuncForGenerator()) {
    // Rename the (auto-renamed) func to match the generator's name.
    funcsToRename[func.getNameAttr()] = generator.getNameAttr();
    // Make sure this isn't about to be removed below.
    assert(!func.getBody()->empty() &&
           "should only include successful expansions");
  }

  // On success, we remove generators and generator interfaces from the file
  // to clean it up.
  for (Operation &op : llvm::make_early_inc_range(primary.getOps())) {
    if (isa<GeneratorOp, GeneratorInterfaceOp>(op)) {
      op.erase();
      continue;
    }

    /// Non viable funcs or inlined funcs will be left with an invalid body.
    /// Remove them at the end of elaboration.
    if (auto func = dyn_cast<FuncOp>(op))
      if (elaborator.shouldRemoveFunc(func))
        func->erase();
  }

  // Perform any renaming at the end.  We cannot use the
  // SymbolTable::replaceAllSymbolUses method, because it doesn't tolerate
  // unregistered operations.  It also doesn't support batch renaming.
  primary->walk([&](Operation *op) {
    // If this is a func being renamed, rename it.
    if (auto func = dyn_cast<FuncOp>(op)) {
      auto it = funcsToRename.find(func.getNameAttr());
      if (it != funcsToRename.end())
        func.setSymNameAttr(it->second);
      return;
    }

    // If this is a reference to a function that got renamed, update its target.
    TypeSwitch<Operation *>(op).Case<CallOp, AddressOfOp>([&](auto op) {
      auto it = funcsToRename.find(op.getCalleeAttr().getLeafReference());
      if (it != funcsToRename.end())
        op.setCalleeAttr(FlatSymbolRefAttr::get(it->second));
    });
  });

  return success();
}

//===----------------------------------------------------------------------===//
// ElaborateGeneratorsPass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_ELABORATEGENERATORS
#define GEN_PASS_DEF_RESOLVEINCLUDES
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
/// Run the elaborator as a pass. The elaborator requires imports to be
/// resolved, so first resolve imports and then elaborate.
struct ElaborateGeneratorsPass
    : public KGEN::impl::ElaborateGeneratorsBase<ElaborateGeneratorsPass> {
  ElaborateGeneratorsPass(SmallVectorImpl<std::string> &includedFiles,
                          const ElaborateGeneratorsOptions &options)
      : ElaborateGeneratorsBase(options), includedFiles(&includedFiles) {}
  using ElaborateGeneratorsBase::ElaborateGeneratorsBase;

  void runOnOperation() override {
    ModuleOp theModule = getOperation();

    SmallVector<std::filesystem::path> paths;
    for (const auto &p : searchPaths)
      paths.push_back(p);

    paths.push_back(std::filesystem::path("."));

    // Extract the top-level, parameterless generators from the main module.
    // These are the only generators that will be elaborated.
    SmallVector<GeneratorOp> primaryGenerators;
    for (auto gen : theModule.getOps<GeneratorOp>())
      if (gen.getInputParamDecls().empty())
        primaryGenerators.push_back(gen);

    SymbolTable symtab(theModule);
    if (failed(resolveIncludes(symtab, paths, includedFiles)))
      return signalPassFailure();

    if (failed(elaborateGenerators(symtab, primaryGenerators, shouldDoSearch)))
      return signalPassFailure();
  }

  /// An optional set of included files that were found during processing.
  SmallVectorImpl<std::string> *includedFiles = nullptr;
};

/// Resolve includes in a pass. This pass only does include resolution.
struct ResolveIncludesPass
    : public KGEN::impl::ResolveIncludesBase<ResolveIncludesPass> {
  using ResolveIncludesBase::ResolveIncludesBase;

  void runOnOperation() override {
    ModuleOp theModule = getOperation();

    SmallVector<std::filesystem::path> paths;
    for (const auto &p : searchPaths)
      paths.push_back(p);
    paths.push_back(std::filesystem::path("."));

    SymbolTable symtab(theModule);
    if (failed(resolveIncludes(symtab, paths)))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
KGEN::createElaborateGenerators(SmallVectorImpl<std::string> &includedFiles,
                                const ElaborateGeneratorsOptions &options) {
  return std::make_unique<ElaborateGeneratorsPass>(includedFiles, options);
}
