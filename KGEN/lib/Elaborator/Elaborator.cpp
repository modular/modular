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
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENPasses.h"
#include "LLCL/CompilerSupport/MLIRLocationDecoder.h"
#include "LLCL/Runtime/Algorithms.h"
#include "SelectFastestFunction.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Compiler/SymbolTableAnalysis.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "Support/STLExtras.h"
#include "SymbolicExpressions.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "elaborator"

using namespace M;
using namespace KGEN;

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
  Elaborator(
      SymbolTableAnalysis &analysis, LLCL::Runtime &runtime,
      LLCL::AsyncSideEffectMap &map,
      LLCL::RCRef<Cache::BlobCache<Cache::TransformCacheKey>> transformCache,
      LLCL::RCRef<Cache::BlobCache<Cache::RegionCacheKey>> regionCache,
      bool enableSearch = false)
      : analysis(analysis), runtime(runtime), asyncMap(map),
        transformCache(std::move(transformCache)),
        regionCache(std::move(regionCache)),
        regionOwner(ModuleOp::create(analysis.getModule().getLoc())),
        enableSearch(enableSearch) {}

  /// Scan the primary and library module to collect all the interfaces,
  /// verifying that any common interfaces are the same.
  ParseResult collectInterfaces();

  /// Return the operation that defines the specified symbol.
  Operation *lookupCallee(SymbolRefAttr symbolRef) {
    assert(isa<FlatSymbolRefAttr>(symbolRef) &&
           "Elaborator doesn't support nested symbols");
    return analysis.getTopLevelSymbolTable().lookup(
        symbolRef.getRootReference());
  }

  /// Return all instantiations of the specified declaration (a func,
  /// generator, or interface) with the specified input parameter values.
  ArrayRef<ErrorTreeOr<ElaboratedGenerator>>
  getAllInstantiations(DeclAndInputParamsPair declAndInputParams,
                       size_t expansionDepth, bool inlined = false);

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

  size_t generatedFuncsMapSize() const {
    size_t r = 0;
    for (const auto &kv : generatedFuncs)
      r += kv.second.size();
    return r;
  }

  /// Bind the result parameters of a fully-specialized function and clear them
  /// from the function.
  void bindResultParameters(FuncOp func) {
    // Make sure the function is inflated - this is a fast no-op if the function
    // has not been deflated.
    asyncMap.mapChained(func, [&](auto ch) {
      return Cache::inflateOp(func, regionCache.copy(), std::move(ch));
    });
    asyncMap.await(func);

    ParameterExprArrayAttr &values = resultParams[func];
    assert(!values && "results for function already bound");
    values = func.getReturnOp().getParametersAttr();

    // Set a new signature that drops the result parameter type list.
    func.setSignature(SignatureType::get(
        func.getInputParamDeclsAttr(),
        /*clear resultParams=*/TypeArrayAttr::get(func.getContext(), {}),
        func.getFunctionType(), func.getConventions()));
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
    RegionBodyOp body;
    IREvaluator evaluator;
    bool isolated;
  };

  /// These methods provide access to the `regionsReferenced` dictionary. This
  /// tracks regions on kgen.call operations with unique string names. The
  /// elaborator takes ownership of the body by hooking it up to a module as
  /// its virtual root.
  void addRegionReference(RegionReferenceAttr attr, RegionBody regionBody) {
    regionsReferenced.try_emplace(attr, std::move(regionBody));
    regionOwner->push_back(regionBody.body);
  }

  /// Get a reference region by name. The referenced region can be isolated
  /// from above or not isolated from above.
  const RegionBody &getRegionReferenced(RegionReferenceAttr attr) {
    return regionsReferenced.find(attr)->second;
  }

  /// Have the elaborator take the nested parameter declarations and uses for
  /// a set of nested parameter scopes. The uses can be looked up later when
  /// elaboration recurses into those nested scopes.
  void takeNestedParameterUses(
      DenseMap<DeclInterface, ParameterDeclsAndUses> &&nestedUses) {
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
  const ParameterDeclsAndUses *lookupNestedUses(DeclInterface decl) {
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
  SmallVector<ErrorTreeOr<ElaboratedGenerator>>
  specializeFunc(FuncOp func, ModuleOp sourceModule, size_t expansionDepth,
                 bool inlined);

  /// Specialize a generator with the specified input parameters and return
  /// the generated func.
  SmallVector<ErrorTreeOr<ElaboratedGenerator>>
  specializeGenerator(DeclAndInputParamsPair declAndInputParams,
                      size_t expansionDepth, bool inlined);

  /// Specialize a generator interface with the specified input parameters and
  /// return the generated func.
  SmallVector<ErrorTreeOr<ElaboratedGenerator>>
  specializeInterface(DeclAndInputParamsPair declAndInputParams,
                      size_t expansionDepth, bool inlined);

  /// Report an error given an interface and an error string - just reduces
  /// boilerplate around CalleeExpansionError creation.
  ErrorTreeOr<ElaboratedGenerator>
  reportCalleeExpansionError(GeneratorInterfaceOp itf, Twine err) {
    return ErrorTree(itf.getLoc(), err);
  };

  /// Instantiate a new evaluator with the given parameters.
  IREvaluator createEvaluator(DenseMap<StringAttr, Attribute> values =
                                  DenseMap<StringAttr, Attribute>()) {
    return {analysis.getTopLevelSymbolTable(), asyncMap, regionCache.copy(),
            transformCache.copy(), std::move(values)};
  }

  /// This symbol table analysis allows efficient lookups across the module.
  SymbolTableAnalysis &analysis;

  /// This provides a runtime reference for the Elaborator and all its
  /// functionality.
  LLCL::Runtime &runtime;

  /// This gives us a map of operation -> in-flight side effect. This is
  /// important because we do async mutations on the IR and we may need await
  /// those mutations to materialize.
  LLCL::AsyncSideEffectMap &asyncMap;

  /// These are the caches the Elaborator will use to run its operations.
  LLCL::RCRef<Cache::BlobCache<Cache::TransformCacheKey>> transformCache;
  LLCL::RCRef<Cache::BlobCache<Cache::RegionCacheKey>> regionCache;

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
           SmallVector<ErrorTreeOr<ElaboratedGenerator>>>
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
  DenseMap<DeclInterface, ParameterDeclsAndUses> nestedParamDeclsAndUses;

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

  /// Let parameter rewriters access the internal fields of the elaborator.
  friend class ParameterRewriter;
};
} // namespace

/// Insert a variant of an existing func into the primary file.
void Elaborator::insertFuncVariant(FuncOp existing, FuncOp newFunc) {
  auto insertPt = Block::iterator(existing.getOperation());
  analysis.getTopLevelSymbolTable().insert(newFunc, ++insertPt);
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
  ParameterRewriter(Elaborator &elaborator, FuncOp func,
                    ArrayRef<Operation *> opsToRewrite, bool inlinedCallee,
                    size_t expansionDepth)
      : elaborator(elaborator),
        sourceModule(cast<ModuleOp>(elaborator.analysis.getModule())),
        elaboratedGenerator(func), inlinedCallee(inlinedCallee),
        expansionDepth(expansionDepth) {
    nextCallRegionID = 0;
    evaluators.push_back(elaborator.createEvaluator());
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
  IREvaluator &getEvaluator() {
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
  ErrorTree takeDiagnosticAndEraseFunc();

  /// Generate an error expanding this generator.  The location specified is
  /// the operation with the problem, and the message is the problem with it.
  LogicalResult error(Location loc, Error message) {
    assert(!diagnostic.has_value() && "Already emitted an error");
    diagnostic = ErrorTree(loc, std::move(message));
    return failure();
  }

  /// Generate an error expanding this generator that occurred while
  /// concretizing a parameter expression.
  LogicalResult error(ErrorTree error) {
    assert(!diagnostic.has_value() && "Already emitted an error");
    diagnostic.emplace(std::move(error));
    return failure();
  }

  /// Generate an error expanding this generator for a call expansion problem.
  /// The location specified is for the call.  Each entry in calleeErrors
  /// includes the location of the declaration that failed to expand along
  /// with why it failed.
  LogicalResult errorCalling(Location callLoc,
                             MutableArrayRef<ErrorTree> calleeErrors) {
    assert(!diagnostic.has_value() && "Already emitted an error");
    diagnostic.emplace(
        ErrorTree(callLoc, "call expansion failed", calleeErrors));
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
  resolveCallInputParams(KGENCallOpInterface call,
                         ArrayRef<ParamBindAttr> inputValues);

  /// Process either a `kgen.addressof` op or a `kgen.call` op.
  template <typename OpT>
  LogicalResult processGeneratorUser(
      OpT user,
      SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriters) {
    return processGeneratorUserImpl(user, user.getCallee(),
                                    user.getParamDecls(), rewriters);
  }
  LogicalResult processGeneratorUserImpl(
      KGENCallOpInterface user, SymbolConstantAttr callee,
      ArrayRef<ParamDeclAttr> decls,
      SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriters);
  LogicalResult completeGeneratorUserProcessing(
      KGENCallOpInterface user, ArrayRef<ParamDeclAttr> decls,
      DeclAndInputParamsPair calleeAndInputParams,
      const ElaboratedGenerator &newCallee, bool inlineCallee);
  LogicalResult spawnNewFuncClone(
      KGENCallOpInterface user, ArrayRef<ParamDeclAttr> decls,
      DeclAndInputParamsPair calleeAndInputParams,
      const ElaboratedGenerator &callee, bool inlineCallee,
      SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriters);

  /// Process a `kgen.call_param` operation by inlining region callees and
  /// simplifying to `kgen.call` for symbol callees.
  LogicalResult processCallParamOp(
      CallParamOp call,
      SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriter);

  /// Process a call to a region.
  LogicalResult processRegionCallImpl(KGENCallOpInterface call,
                                      RegionReferenceAttr regionName,
                                      ParamDeclArrayAttr decls,
                                      ArrayRef<ParamBindAttr> paramValues);

  /// Process a generic operation that does not fit into one of the above types.
  /// Substitute parameters in the operation's attributes and types.
  LogicalResult processGenericOp(Operation *op);

  /// Operations may reference parameters within types in their locations.
  /// Process them by rewriting their locations.
  LogicalResult processLocation(Operation *op);

  /// This is maintains global information about the file we're generating
  /// into.
  Elaborator &elaborator;

  /// This indicates which module this func originally came from (e.g. one of
  /// the imported files).  This is important to know so we can correctly
  /// resolve callee symbols.
  ModuleOp sourceModule;

  /// This is the generator -> func we're working on.
  ElaboratedGenerator elaboratedGenerator;

  /// This is the diagnostic explaining the expansion failure if something
  /// goes wrong.
  Optional<ErrorTree> diagnostic;

  /// This is a stack of evaluators to use to process parameter expressions.
  /// The current evaluator is always at the back() of the list.  Having a
  /// stack of evaluators allows us to maintain scoped parameters when
  /// processing regions.
  SmallVector<IREvaluator, 2> evaluators;

  /// These are the commands that still need to get performed before this func
  /// has been fully evaluated.  These are mostly operations that need to be
  /// rewritten.
  SmallVector<RewriterCommandType> commandWorklist;

  /// This is a counter that gives each region attached to a kgen.call a
  /// unique number (and therefore, unique name).
  unsigned nextCallRegionID;

  /// A flag to indicate whether the elaborated function will be inlined.
  bool inlinedCallee;

  /// This is the depth of this expansion, which is used to cut off overly deep
  /// recursive evaluations.
  size_t expansionDepth;
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
    : elaborator(existing.elaborator), sourceModule(existing.sourceModule),
      elaboratedGenerator(existing.elaboratedGenerator),
      evaluators(existing.evaluators),
      nextCallRegionID(existing.nextCallRegionID),
      inlinedCallee(existing.inlinedCallee),
      expansionDepth(existing.expansionDepth) {
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

  // FIXME: Make this magic number configurable.
  if (expansionDepth >= 128)
    return error(elaboratedGenerator.func.getLoc(),
                 "elaborator expansion is " + Twine(expansionDepth) +
                     " levels deep - infinite recursion?");

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
      ErrorTreeOr<Attribute> value =
          getEvaluator().concretizeParameterExpr(rr->returnLoc, expr);
      if (value.isError())
        return error(value.takeError());
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

  // Verify the function and invoke the symbol user verifier on all its
  // contained ops to verify parameter references.
  auto verifySymbolUses = [&](mlir::SymbolUserOpInterface user) -> WalkResult {
    return user.verifySymbolUses(elaborator.analysis.getSymbolTables());
  };
  if (failed(verify(func)) || func.walk(verifySymbolUses).wasInterrupted())
    return error(*verificationLoc,
                 Twine("verification error: ") + verificationError.str());

  return success();
}

LogicalResult ParameterRewriter::processParamDeclareOp(ParamDeclareOp op) {
  // Simplify the input expression.
  ErrorTreeOr<Attribute> value =
      getEvaluator().concretizeParameterExpr(op.getLoc(), op.getValue());
  if (value.isError())
    return error(value.takeError());

  // Bind it to the parameter declaration it is setting.
  getEvaluator().setOrOverwriteParameterValue(op.getParamDecl(),
                                              value.getValue());

  // The kgen.param.declare operation serves no other purpose: remove it.
  op->erase();
  return success();
}

LogicalResult ParameterRewriter::processParamSearchOp(
    ParamSearchOp op,
    SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriters) {
  // Loop over all the possible candidates that we will search over, spawning
  // N-1 possibilities to explore.
  std::vector<ErrorTree> errors;
  Attribute firstValid;
  DenseSet<Attribute> seenValues;
  LLVM_DEBUG({
    llvm::dbgs() << "Encountered ParamSearchOp with "
                 << std::to_string(op.getValues().size())
                 << " options: " << op.getValuesAttr() << "\n";
  });
  for (Attribute candidate : op.getValues()) {
    // Simplify the input expressions.
    ErrorTreeOr<Attribute> errorOrValue =
        getEvaluator().concretizeParameterExpr(op.getLoc(), candidate);
    if (errorOrValue.isError()) {
      errors.push_back(errorOrValue.takeError());
      continue;
    }

    Attribute value = errorOrValue.takeValue();

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
      return error(op.getLoc(), "no values to search over");
    return error(ErrorTree(
        op.getLoc(), "failed to concretize any search parameters", errors));
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
  ErrorTreeOr<Attribute> value =
      getEvaluator().concretizeParameterExpr(op.getLoc(), op.getValue());
  if (value.isError())
    return error(value.takeError());

  ErrorTreeOr<Type> type =
      getEvaluator().concretizeParameterExpr(op.getLoc(), op.getType());
  if (type.isError())
    return error(type.takeError());

  op.getResult().setType(type.takeValue());
  op.setValueAttr(value.getValue());
  return success();
}

LogicalResult ParameterRewriter::processParamAssertOp(ParamAssertOp op) {
  // Check the condition expression.
  ErrorTreeOr<Attribute> value =
      getEvaluator().concretizeParameterExpr(op.getLoc(), op.getCond());
  if (value.isError())
    return error(value.takeError());

  auto resultInt = dyn_cast<IntegerAttr>(value.takeValue());
  if (!resultInt || resultInt.getValue().getBitWidth() != 1)
    return error(op.getLoc(),
                 "constraint evaluation didn't return true or false");
  // If the constraint evaluated to zero then the assert fails.
  if (resultInt.getValue().isZero())
    return error(op.getLoc(), "constraint failed: " + op.getMessage());

  // The kgen.param.assert op serves no further purpose, so we can remove it.
  op->erase();
  return success();
}

/// Resolve all of input parameters present at the specified call site to
/// concrete constants. This reports the error and returns null on failure,
/// and returns an array of bound input parameters on success.
FailureOr<std::pair<ArrayAttr, bool>>
ParameterRewriter::resolveCallInputParams(KGENCallOpInterface call,
                                          ArrayRef<ParamBindAttr> inputValues) {
  SmallVector<Attribute> boundInputParams;
  bool inlineCallee = false;
  auto regionBodyIt = call.getRegionBodies().begin();
  for (ParamBindAttr param : inputValues) {
    // If this is a region reference, form a binding to the region provided by
    // the call.
    if (auto regionRef = dyn_cast<ParamCallRegionRefAttr>(param.getValue())) {
      // Give this reference a unique name, and make a RegionReferenceAttr with
      // the name and SignatureType.
      auto regionRefAttr =
          RegionReferenceAttr::get(elaboratedGenerator.func.getName() +
                                       "_region_" + Twine(nextCallRegionID++),
                                   regionRef.getType());

      auto body = cast<RegionBodyOp>(**regionBodyIt++);
      // Determine whether the body isolated from unhooking it from its parent.
      bool isolated = operationIsIsolatedFromAbove(body);

      // Remove the RegionBodyOp from the call's region, and hand ownership of
      // it to the elaborator.
      body->remove();

      // TODO: We could do some content hashing to avoid making a new name for
      // a lexically identical body.  This would reduce some redundant
      // specialization.
      Elaborator::RegionBody regionBody{
          body, elaborator.createEvaluator(getEvaluator().getParameterValues()),
          isolated};
      inlineCallee |= !regionBody.isolated;
      elaborator.addRegionReference(regionRefAttr, std::move(regionBody));
      boundInputParams.push_back(regionRefAttr);
      continue;
    }

    // Otherwise fold the parameter expression in this context to a simple
    // constant.
    ErrorTreeOr<Attribute> value =
        getEvaluator().concretizeParameterExpr(call.getLoc(), param.getValue());
    if (value.isError())
      return error(value.takeError());

    // If we have a reference to a non-isolated region, indicate that the callee
    // will always be inlined.
    auto regionRef = dyn_cast<RegionReferenceAttr>(*value);
    if (regionRef && !elaborator.getRegionReferenced(regionRef).isolated)
      inlineCallee = true;

    boundInputParams.push_back(*value);
  }
  return std::make_pair(ArrayAttr::get(call->getContext(), boundInputParams),
                        inlineCallee);
}

LogicalResult ParameterRewriter::processGeneratorUserImpl(
    KGENCallOpInterface user, SymbolConstantAttr callee,
    ArrayRef<ParamDeclAttr> decls,
    SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriters) {
  // Evaluate any input parameters.
  FailureOr<std::pair<ArrayAttr, bool>> result =
      resolveCallInputParams(user, callee.getParamValues());
  if (failed(result))
    return failure();
  auto [inputParamKey, inlineCallee] = *result;

  // Prevent `kgen.addressof` from referencing a function that will be inilned.
  if (inlineCallee && isa<AddressOfOp>(user))
    return error(user.getLoc(),
                 "cannot take the address of a function that will be inlined");

  // Instantiate the callee into one or more FuncOp's, depending on what the
  // callee is.
  auto ref = cast<FlatSymbolRefAttr>(callee.getSymbol());
  auto func = dyn_cast_if_present<FuncInterface>(elaborator.lookupCallee(ref));
  if (!func)
    return error(user->getLoc(), Twine("could not find callee '") +
                                     ref.getAttr().strref() + "'");

  // If the callee is an interface that provides an evaluator, resolve the
  // evaluator first.
  if (auto itf = dyn_cast<GeneratorInterfaceOp>(*func)) {
    if (SymbolConstantAttr evaluator = itf.getEvaluatorAttr()) {
      if (!elaborator.analysis.getTopLevelSymbolTable().lookup<FuncOp>(
              cast<FlatSymbolRefAttr>(evaluator.getSymbol()).getAttr())) {
        evaluators.push_back(elaborator.createEvaluator());
        for (auto [bind, value] :
             llvm::zip(callee.getParamValues(), inputParamKey))
          evaluators.back().setParameterValue(bind.getDecl(), value);
        if (failed(processGeneratorUserImpl(itf, evaluator, {}, rewriters)))
          return failure();
        evaluators.pop_back();
      }
    }
  }

  DeclAndInputParamsPair calleeDeclAndInputParams{func, inputParamKey};

  // If we already have a binding for this decl/inputParam set, then reuse the
  // consistent callee.
  if (FuncOp callee =
          elaboratedGenerator.getBinding(calleeDeclAndInputParams)) {
    return completeGeneratorUserProcessing(
        user, decls, calleeDeclAndInputParams, ElaboratedGenerator(callee),
        inlineCallee);
  }

  // Otherwise, this is our first use of this.  Ask the global elaborator for
  // the full set of candidates.
  ArrayRef<ErrorTreeOr<ElaboratedGenerator>> newCalleesRef =
      elaborator.getAllInstantiations(calleeDeclAndInputParams,
                                      expansionDepth + 1, inlineCallee);

  // Copy the list of funcs instead of referring to the cache entry to avoid
  // iterator invalidation problems.
  SmallVector<ErrorTreeOr<ElaboratedGenerator>> newCallees;
  for (const ErrorTreeOr<ElaboratedGenerator> &newCallee : newCalleesRef)
    newCallees.push_back(newCallee.copy());

  // If we found more than one callee to produce then we need to spawn
  // multiple versions of the func we are currently constructing, each
  // which get a different callee.
  ElaboratedGenerator thisCallee(/*func=*/nullptr);
  for (const ErrorTreeOr<ElaboratedGenerator> &candidate : newCallees) {
    // Ignore erroneous callees.
    if (candidate.isError())
      continue;
    // Ignore the candidate if the elaborated func is inconsistent with our
    // current bindings.
    const ElaboratedGenerator &calleeCandidate = candidate.getValue();
    if (!calleeCandidate.isConsistentWith(elaboratedGenerator))
      continue;

    // If this is the first viable candidates, then we will pursue it locally.
    if (!thisCallee.func) {
      thisCallee = calleeCandidate;
    } else if (auto itf = dyn_cast<GeneratorInterfaceOp>(*user)) {
      // Prohibit interface evaluators from having multiple implementations.
      return error(itf.getLoc(),
                   Twine("interface @") + itf.getSymName() +
                       " evaluator must resolve to a single implementation");
    } else {
      // All other callees gets spawned as sub-evaluators.
      if (failed(spawnNewFuncClone(user, decls, calleeDeclAndInputParams,
                                   calleeCandidate, inlineCallee, rewriters)))
        return failure();
    }
  }

  // If all the expansions failed, then this call fails overall.
  if (!thisCallee.func) {
    SmallVector<ErrorTree> errors;
    for (ErrorTreeOr<ElaboratedGenerator> &value : newCallees)
      errors.push_back(value.takeError());
    return errorCalling(user->getLoc(), errors);
  }

  // Finally, we can handle the first viable one as our continued progress here.
  return completeGeneratorUserProcessing(user, decls, calleeDeclAndInputParams,
                                         thisCallee, inlineCallee);
}

LogicalResult ParameterRewriter::completeGeneratorUserProcessing(
    KGENCallOpInterface user, ArrayRef<ParamDeclAttr> decls,
    DeclAndInputParamsPair calleeAndInputParams,
    const ElaboratedGenerator &newCallee, bool inlineCallee) {
  // Add a binding to remember that we resolved this call to this candidate,
  // and merge any bindings from it into our set.
  elaboratedGenerator.addBinding(calleeAndInputParams, newCallee);

  FuncOp newCalleeFunc = newCallee.func;

  // Resolve any bound result types.
  SmallVector<Type> resultTypes;
  for (Type result : user->getResultTypes()) {
    ErrorTreeOr<Type> type =
        getEvaluator().concretizeParameterExpr(user.getLoc(), result);
    if (type.isError())
      return error(type.takeError());
    resultTypes.push_back(type.takeValue());
  }

  // Now that we resolved the call to a new thing, build a new call to replace
  // the old one.
  mlir::IRRewriter b{OpBuilder(user)};
  if (isa<CallOp, CallParamOp>(user)) {
    if (!inlineCallee) {
      b.replaceOpWithNewOp<CallOp>(
          user, resultTypes,
          SymbolConstantAttr::get(
              FlatSymbolRefAttr::get(newCalleeFunc.getNameAttr()),
              newCalleeFunc.getSignature()),
          user->getOperands());
    } else {
      // Inline the callee.
      BlockAndValueMapping bv;
      for (auto [operand, argument] :
           llvm::zip(newCalleeFunc.getArguments(), user->getOperands()))
        bv.map(operand, argument);

      for (Operation &op : newCalleeFunc.getBody()->without_terminator()) {
        Operation *cloned = b.clone(op, bv);
        cloned->walk([&](Operation *op) {
          op->setLoc(mlir::CallSiteLoc::get(op->getLoc(), user->getLoc()));
        });
      }
      Operation *terminator =
          b.clone(*newCalleeFunc.getBody()->getTerminator(), bv);
      b.replaceOp(user, terminator->getOperands());
      terminator->erase();
    }

  } else if (isa<AddressOfOp>(user)) {
    b.replaceOpWithNewOp<AddressOfOp>(
        user, resultTypes.front(),
        SymbolConstantAttr::get(
            FlatSymbolRefAttr::get(newCalleeFunc.getNameAttr()),
            newCalleeFunc.getSignature()));

  } else {
    // Update the interface in-place.
    auto itf = cast<GeneratorInterfaceOp>(user);
    itf.setEvaluatorAttr(SymbolConstantAttr::get(
        FlatSymbolRefAttr::get(newCalleeFunc.getSymNameAttr()),
        newCalleeFunc.getSignature()));
  }

  // Bind the result parameters to the output parameter decls.
  for (auto [decl, bindValue] :
       llvm::zip(decls, elaborator.lookupResultParameters(newCallee.func)))
    getEvaluator().setOrOverwriteParameterValue(decl, bindValue);

  return success();
}

/// Sometimes when we expand a call, we find that there are multiple viable
/// callees that we can generate.  We handle this by spawning new parameter
/// rewriters with state copied from the current one, but which resolve the call
/// to different callees.  This spawns a new rewriter with the specified call
/// resolving to the specified callee.
LogicalResult ParameterRewriter::spawnNewFuncClone(
    KGENCallOpInterface user, ArrayRef<ParamDeclAttr> decls,
    DeclAndInputParamsPair calleeAndInputParams,
    const ElaboratedGenerator &callee, bool inlineCallee,
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
      return !elaborator.getRegionReferenced(regionRef).isolated;
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
  return newRewriter->completeGeneratorUserProcessing(
      newUser, decls, calleeAndInputParams, callee, inlineCallee);
}

LogicalResult ParameterRewriter::processCallParamOp(
    CallParamOp call,
    SmallVectorImpl<std::unique_ptr<ParameterRewriter>> &rewriters) {
  // Simplify the callee expression.
  ErrorTreeOr<Attribute> value =
      getEvaluator().concretizeParameterExpr(call.getLoc(), call.getCallee());
  if (value.isError())
    return error(value.takeError());

  // If the parameter expression is resolved to a symbol, then treat this like a
  // direct call.
  if (auto symbolCst = dyn_cast<SymbolConstantAttr>(*value)) {
    // If there are no bound parameters on the call, use the one on the
    // CallParam.  TODO: Remove.
    if (symbolCst.getParamValues().empty())
      return processGeneratorUserImpl(
          call,
          SymbolConstantAttr::get(symbolCst.getSymbol(),
                                  call.getParamValuesAttr(),
                                  symbolCst.getType().dropParamValues()),
          call.getParamDecls(), rewriters);
    // Otherwise use the ones from the symbol.
    return processGeneratorUserImpl(call, symbolCst, call.getParamDecls(),
                                    rewriters);
  }

  // Otherwise, the only other case we support is a call to a region, which is
  // marked with a StringAttr value that has signature type.
  auto regionName = cast<RegionReferenceAttr>(*value);
  return processRegionCallImpl(call, regionName, call.getParamDeclsAttr(),
                               call.getParamValues());
}

LogicalResult ParameterRewriter::processRegionCallImpl(
    KGENCallOpInterface call, RegionReferenceAttr regionName,
    ParamDeclArrayAttr decls, ArrayRef<ParamBindAttr> paramValues) {
  assert(regionName.getType().isa<SignatureType>() && "not a region reference");
  const Elaborator::RegionBody &regionBody =
      elaborator.getRegionReferenced(regionName);
  FuncInterface region = regionBody.body;

  // Compute the binding of input parameters to concrete values.
  FailureOr<std::pair<ArrayAttr, bool>> result =
      resolveCallInputParams(call, paramValues);
  if (failed(result))
    return failure();
  auto [inputParamKey, _] = *result;

  if (failed(processLocation(call)))
    return failure();

  auto theRegionReturnOp =
      cast<ReturnOp>(region->getRegion(0).front().getTerminator());

  // Push a region return command that contains the output parameters to bind
  // after processing all the operations in the region.
  commandWorklist.push_back(
      new RegionReturn{call->getLoc(), theRegionReturnOp.getLoc(),
                       theRegionReturnOp.getParametersAttr(), decls});

  // Nested regions will have a different parameter namespace than the caller
  // context: names will mean different things inside the region than they did
  // in the caller.  To handle this, we push a IREvaluator scope that
  // represents the bindings within the region body and set the region return to
  // restore back to the previous scope when operations from the region have
  // finished their processing.
  IREvaluator &evaluator = evaluators.emplace_back(regionBody.evaluator);

  // Add bindings for each of the input parameters to the new scope we just
  // pushed, so they are properly bound when the rewriter continues processing
  // the newly cloned operations.
  for (auto [decl, value] :
       llvm::zip(region.getInputParamDecls(), inputParamKey))
    evaluator.setOrOverwriteParameterValue(decl, value);

  // Evaluate any constraints for this declaration to see if this is a viable
  // expansion.  If not, the expansion fails. Only isolated regions have
  // constraints.
  Optional<ErrorTree> err = evaluateConstraints(
      cast<DeclInterface>(*region).getConstraints(), evaluator);
  if (err)
    return error(std::move(*err));

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
  for (auto &bodyOp : bodyBlock) {
    Operation *clonedOp =
        b.insert(cloneOperation(&bodyOp, mapper, operationMap));
    clonedOp->walk([&](Operation *newOp) {
      newOp->setLoc(mlir::CallSiteLoc::get(newOp->getLoc(), call->getLoc()));
    });
  }

  // Lookup all the parameter decls and uses in the body of region, we will
  // visit all of them as the evaluator continues processing the ops we just
  // cloned over.
  const ParameterDeclsAndUses *uses =
      elaborator.lookupNestedUses(cast<DeclInterface>(*region));
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
    elaborator.takeNestedParameterUses(
        recomputedUses->calculate(cast<DeclInterface>(*region)));
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
    ErrorTreeOr<Attribute> value = getEvaluator().concretizeParameterExpr(
        op->getLoc(), namedAttr.getValue(), /*allowUnknown=*/true);
    if (value.isError())
      return error(value.takeError());

    newAttrs.emplace_back(namedAttr.getName(), value.takeValue());
    changedAttrs |= namedAttr.getValue() != newAttrs.back().getValue();
  }
  if (changedAttrs)
    op->setAttrs(newAttrs);

  if (failed(processLocation(op)))
    return failure();

  // Check the types of results to find any parameters embedded in their
  // types.  We don't have to check operands because they are always checked
  // when being defined.
  for (OpResult result : op->getResults()) {
    ErrorTreeOr<Type> type =
        getEvaluator().concretizeParameterExpr(op->getLoc(), result.getType());
    if (type.isError())
      return error(type.takeError());
    result.setType(type.takeValue());
  }

  // Scan the region list if present.  The walker will automatically recurse
  // for us, but we have to check the block arguments.
  if (op->getNumRegions()) { // Microoptimization: getRegions() is slow.
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (Value arg : block.getArguments()) {
          ErrorTreeOr<Type> type = getEvaluator().concretizeParameterExpr(
              op->getLoc(), arg.getType());
          if (type.isError())
            return error(type.takeError());
          arg.setType(type.takeValue());
        }
      }
    }
  }

  // If the op implements the elaborator interface, indicate it as resolved.
  if (auto elaboratorIface = dyn_cast<ElaboratorOpInterface>(op)) {
    ErrorOrSuccess result = elaboratorIface.finalizeElaboration();
    if (result.isError())
      return error(op->getLoc(), result.takeError());
  }

  return success();
}

LogicalResult ParameterRewriter::processLocation(Operation *op) {
  ErrorTreeOr<Attribute> value = getEvaluator().concretizeParameterExpr(
      op->getLoc(), op->getLoc(), /*allowUnknown=*/true);
  if (value.isError())
    return error(value.takeError());
  op->setLoc(cast<Location>(value.takeValue()));
  return success();
}

/// If elaboration of this func fails, then the client can get the error
/// out.  This also deallocates the body of the dead husk of the func which
/// may not even verify correctly, it will be removed later.
ErrorTree ParameterRewriter::takeDiagnosticAndEraseFunc() {
  assert(diagnostic.has_value() &&
         "cannot get diagnostic when none was generated");
  // The generator is not viable so we need to delete it.  This op can appear
  // in various maps though, so instead of actually deleting it, we just
  // mark it for removal later.
  elaborator.markFuncForRemoval(elaboratedGenerator.func);
  ErrorTree error = std::move(*diagnostic);

  // Check to see if the error occurs in the scope of a call_param to a region.
  // If so, make sure to add the nested call to the expansion path.
  for (auto command : llvm::reverse(commandWorklist)) {
    RegionReturn *rr = dyn_cast<RegionReturn *>(command);
    if (!rr)
      continue;
    error = ErrorTree(rr->callLoc, "call expansion failed", std::move(error));
  }

  return error;
}

//===----------------------------------------------------------------------===//
// Elaborator::getAllInstantiations
//===----------------------------------------------------------------------===//

static void printParameterValue(TypedAttr value, llvm::raw_ostream &os) {
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
    if (auto flat = dyn_cast<FlatSymbolRefAttr>(symbolConstant.getSymbol()))
      os << flat.getValue();
    else
      os << symbolConstant.getSymbol();
  } else if (auto stringConstant = dyn_cast<StringAttr>(value)) {
    os << stringConstant.strref();
  } else if (auto listConstant = dyn_cast<ListAttr>(value)) {
    os << '[';
    llvm::interleave(
        listConstant.getValues(), os,
        [&](TypedAttr value) { printParameterValue(value, os); }, ",");
    os << ']';
  } else {
    value.print(os, /*elideType=*/true);
  }
}

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

  auto inputParamDecls = generator.getInputParamDeclsAttr();
  for (auto [inputDecl, value] : llvm::zip(inputParamDecls, inputParamValues)) {
    os << ',' << inputDecl.getName().str() << '=';
    printParameterValue(value, os);
  }
  return b.getStringAttr(result);
}

/// Specialize a func body, generating one variant of each viable
/// instantiation of that body.  funcs do not have parameters, but they can
/// invoke interfaces etc which can cause them to produce multiple variants.
SmallVector<ErrorTreeOr<ElaboratedGenerator>>
Elaborator::specializeFunc(FuncOp func, ModuleOp sourceModule,
                           size_t expansionDepth, bool inlined) {
  LLVM_DEBUG({
    llvm::dbgs() << std::string(expansionDepth, ' ') << "specializeFunc "
                 << func.getName() << "\n";
  });
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
      *this, func, std::move(opsToRewrite), inlined, expansionDepth));

  // Extract the debug info from the function, if it's present.
  auto oldFuncSp = DebugInfo::extractScope<DebugInfo::DISubprogramAttr>(func);

  // Rewriting funcs may generate other func clones.  If so, rewrite them,
  // until we converge.
  SmallVector<ErrorTreeOr<ElaboratedGenerator>> results;
  size_t counter = 0;
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
      counter++;

    } else {
      LLVM_DEBUG({
        llvm::dbgs() << std::string(expansionDepth, ' ')
                     << "elaboration failed for " << func.getName() << "\n";
      });
      // If elaborating the func fails, then remember the diagnostic (in case
      // we need to explain why elaboration fails) and remove the broken husk of
      // a func that didn't make it.
      results.push_back(rewriter->takeDiagnosticAndEraseFunc());
    }
  }
  LLVM_DEBUG({
    llvm::dbgs() << std::string(expansionDepth, ' ') << "specializeFunc "
                 << func.getName() << " produced " << std::to_string(counter)
                 << " results\n";
  });

  // If we had a subprogram, update uses with the newly elaborated subprogram.
  if (oldFuncSp) {
    auto newFuncSp = DebugInfo::extractScope<DebugInfo::DISubprogramAttr>(func);
    if (newFuncSp != oldFuncSp) {
      DebugInfo::DIAttrTypeReplacer replacer;
      replacer.addReplacement([&](DebugInfo::DISubprogramAttr attr) {
        return attr == oldFuncSp ? newFuncSp : attr;
      });
      replacer.recursivelyReplaceElementsIn(func);
    }
  }

  return results;
}

/// Specialize a generator with the specified input parameters and return the
/// symbol name to use for the result, along with an array of ParamBindAttrs for
/// the result attributes.
SmallVector<ErrorTreeOr<ElaboratedGenerator>>
Elaborator::specializeGenerator(DeclAndInputParamsPair declAndInputParams,
                                size_t expansionDepth, bool inlined) {
  auto generator = cast<GeneratorOp>(declAndInputParams.first);
  LLVM_DEBUG({
    llvm::dbgs() << std::string(expansionDepth, ' ') << "specializeGenerator "
                 << generator.getNameAttr() << "\n";
  });

  ArrayRef<Attribute> inputParamValues = declAndInputParams.second.getValue();
  auto inputParamDecls = generator.getInputParamDecls();
  assert(inputParamValues.size() == inputParamDecls.size() &&
         "incorrect # input parameter values");

  // TODO (low prio): Some day we could mangle "instantiated from here"
  // information into the location.
  OpBuilder b(generator);
  auto newFunc = b.create<FuncOp>(
      generator.getLoc(), mangleParameterValues(generator, inputParamValues),
      SignatureType::get(ParamDeclArrayAttr::get(generator.getContext(), {}),
                         generator.getResultParamTypesAttr(),
                         generator.getFunctionType(),
                         generator.getConventions()));

  // Insert the newFunc into the symbol table which will then know about it,
  // but it will also auto-rename the symbol for us in the case of conflicts.
  analysis.getTopLevelSymbolTable().insert(newFunc);

  // Make sure this generator is inflated.
  asyncMap.mapChained(generator, [&](auto ch) {
    return Cache::inflateOp(generator, regionCache.copy(), std::move(ch));
  });
  asyncMap.await(generator);

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
  auto result = specializeFunc(newFunc, sourceModule, expansionDepth, inlined);

  // If the generator had no parameters, then we want to reuse the same name as
  // the original generator.  We can't do that when we are building the concrete
  // version though because we may have other calls to the generator and those
  // calls get linked to the generator by their symbol.  Additionally,
  // elaboration of any candidate could fail.
  //
  // To handle this, we let the symbol table autorename it, but keep track of
  // the first successful implementation in a map.  We rename it back after the
  // module has finished elaboration.
  FuncOp firstSuccessfulImpl;
  if (inputParamValues.empty()) {
    for (auto &candidate : result) {
      if (candidate) {
        auto it =
            firstConcreteFuncForGenerator.insert({generator, candidate->func});
        firstSuccessfulImpl = it.first->second;
        break;
      }
    }
  }

  // If the generator had debug information, update the debug info for any
  // elaborated instantiations.
  if (DebugInfo::extractScope(generator)) {
    for (auto &candidate : result) {
      if (!candidate)
        continue;
      FuncOp elabFunc = candidate->func;

      // If this function was the first successful instantiation, it will get to
      // inherit the original name of the generator (i.e., nothing to do here).
      if (elabFunc == firstSuccessfulImpl)
        continue;

      // Otherwise, we need to update the sub program to use the new linkage
      // name.
      auto oldSpAttr =
          DebugInfo::extractScope<DebugInfo::DISubprogramAttr>(elabFunc);
      auto newSpAttr = DebugInfo::DISubprogramAttr::get(
          b.getContext(), oldSpAttr.getCompileUnit(), oldSpAttr.getScope(),
          oldSpAttr.getName(), elabFunc.getNameAttr(), oldSpAttr.getFile(),
          oldSpAttr.getLine(), oldSpAttr.getScopeLine(),
          oldSpAttr.getSubprogramFlags(), oldSpAttr.getType());

      DebugInfo::DIAttrTypeReplacer replacer;
      replacer.addReplacement([&](DebugInfo::DISubprogramAttr attr) {
        return attr == oldSpAttr ? newSpAttr : attr;
      });
      replacer.recursivelyReplaceElementsIn(elabFunc);
    }
  }

  return result;
}

/// Specialize a generator interface with the specified input parameters and
/// return the generated func.
///
/// If search is enabled and required (i.e. more than one successful candidate
/// is generated) then the process of "given a list of functions and a way to
/// evaluate them, which is best" is cached. The cache stores the symbol name of
/// the fastest function, which we can then use to shortcut the search process
/// and truncate the result vector to contain only the fastest one as found by a
/// previous run. The search key is comprised of the inputs to search because if
/// any of the implementations change, a new one is added, one is removed, the
/// evaluator changes, or the target changes, we need to redo search.
SmallVector<ErrorTreeOr<ElaboratedGenerator>>
Elaborator::specializeInterface(DeclAndInputParamsPair declAndInputParams,
                                size_t expansionDepth, bool inlined) {
  auto itf = cast<GeneratorInterfaceOp>(declAndInputParams.first);
  LLVM_DEBUG({
    llvm::dbgs() << std::string(expansionDepth, ' ') << "specializeInterface "
                 << itf.getName() << "\n";
  });
  SmallVector<ErrorTreeOr<ElaboratedGenerator>> result;

  // An interface is an abstraction over multiple generators.  Invoke each of
  // them, collecting the results together into a single result.
  ArrayRef<GeneratorOp> interfaceImpls = getGeneratorsImplementing(itf);
  LLVM_DEBUG({
    llvm::dbgs() << std::string(expansionDepth, ' ')
                 << std::to_string(interfaceImpls.size())
                 << " implementations found\n";
  });
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
    auto funcs = getAllInstantiations({gen, declAndInputParams.second},
                                      expansionDepth + 1, inlined);
    for (auto &func : funcs)
      result.push_back(func.copy());
    return result;
  }

  for (GeneratorOp gen : interfaceImpls) {
    // Make sure to go through getAllInstantiations so generators are cached
    // and any constraints on the generator itself are validated.
    ArrayRef<ErrorTreeOr<ElaboratedGenerator>> funcs = getAllInstantiations(
        {gen, declAndInputParams.second}, expansionDepth + 1, inlined);
    LLVM_DEBUG({
      llvm::dbgs() << std::string(expansionDepth, ' ') << gen.getNameAttr()
                   << " produced " << std::to_string(funcs.size())
                   << " candidates for " << itf.getName() << "\n";
    });

    // If there are multiple implementations that failed to expand, group their
    // errors together so we can report them as an umbrella.
    if (interfaceImpls.size() > 1) {
      SmallVector<ErrorTree> errors;
      for (const ErrorTreeOr<ElaboratedGenerator> &func : funcs) {
        if (func.isError())
          errors.push_back(func.getError().copy());
        else
          result.push_back(func.getValue());
      }
      result.push_back(
          ErrorTree(gen.getLoc(), "failed to expand this declaration", errors));
    } else {
      for (const ErrorTreeOr<ElaboratedGenerator> &func : funcs)
        result.push_back(func.copy());
    }
  }

  // If all the results are expansion errors, return them to the caller which
  // will cause elaboration to fail.
  auto isError = [](const auto &kOr) { return kOr.isError(); };
  if (llvm::all_of(result, isError))
    return result;

  // Move the expansion errors to the end of the vector.
  auto newEnd = llvm::remove_if(result, isError);
  // Only one successful elaboration, we don't have to search, just return it.
  if (newEnd == result.begin() + 1) {
    result.erase(result.begin() + 1, result.end());
    return result;
  }

  SymbolConstantAttr evaluator = itf.getEvaluatorAttr();

  // Truncate the result vector to contain only the successful implementations.
  result.erase(newEnd, result.end());
  LLVM_DEBUG({
    llvm::dbgs() << std::string(expansionDepth, ' ') << "Number of results for "
                 << itf.getName() << ": " << result.size() << "\n";
    llvm::dbgs() << std::string(expansionDepth, ' ')
                 << "Total number of generated funcs: "
                 << generatedFuncsMapSize() << "\n";
  });

  // If we don't want to do search, we're done.
  if (!enableSearch) {
    result.erase(result.begin() + 1, result.end());
    return result;
  }

  // If there is no evaluator, return the full vector of instantiations. If the
  // interface is being inlined, we can't benchmark the instantiations because
  // they would not be well-formed.
  if (!evaluator || inlined)
    return result;

  // Store the valid implementations.
  SmallVector<ElaboratedGenerator> candidates;
  for (ErrorTreeOr<ElaboratedGenerator> &candidate : result)
    candidates.push_back(candidate.takeValue());

  auto keyBuf = Cache::WriteableBuffer::get();

  // Pull out the elaboration results that succeeded to provide to the search
  // inputs. We also write the bytecode for each input into the key.
  SmallVector<FuncOp> searchInputs;
  for (const auto &r : result) {
    searchInputs.push_back(r->func);
    mlir::writeBytecodeToFile(searchInputs.back(), *keyBuf);
  }

  // Part of the key is the evaluation function.
  auto evalFunc = cast<FuncOp>(lookupCallee(evaluator.getSymbol()));
  mlir::writeBytecodeToFile(evalFunc, *keyBuf);

  // And finally, the target.
  // TODO(#5584): This needs to use the target we're compiling *for*, not the
  //              host.
  *keyBuf << TargetInfoAttr::getForHost(itf->getContext());

  // Alright - we want to do search now.
  LLCL::AsyncValue::registerTypes<ErrorTreeOr<ElaboratedGenerator>>();

  // This provides the implementation of search. This is the part we actually
  // care about caching because it's the most expensive part.
  auto doSpecialization = [this, evalFunc, searchInputs,
                           candidates](Operation *itfOp,
                                       Cache::WriteableBufferRef toCache,
                                       AnyAsyncValueRef chain) {
    auto out = LLCL::AsyncValueRef<ErrorTreeOr<ElaboratedGenerator>>::allocate(
        runtime);
    chain->andThenSync([this, evalFunc, searchInputs, &candidates, itfOp,
                        chain = chain.copy(), out = out.copy(),
                        toCache = std::move(toCache)]() mutable {
      auto itf = cast<GeneratorInterfaceOp>(itfOp);

      ErrorOr<size_t> bestSpecializationIdxOr = evaluateSpecializations(
          evalFunc, analysis.getTopLevelSymbolTable(), runtime, searchInputs);
      if (failed(bestSpecializationIdxOr)) {
        return out.emplace(reportCalleeExpansionError(
            itf, bestSpecializationIdxOr.getError()));
      }

      // Find the fastest one and return just that one.
      const ElaboratedGenerator &bestResult =
          candidates[*bestSpecializationIdxOr];

      // Finally, cache the result.
      FuncOp resultFunc = bestResult.func;
      *toCache << resultFunc.getName();
      return out.emplace(std::move(bestResult));
    });
    return out;
  };

  auto onCacheHit =
      [this, candidates](Operation *itfOp,
                         Cache::BufferRef cacheContents) -> AnyAsyncValueRef {
    auto out = LLCL::AsyncValueRef<ErrorTreeOr<ElaboratedGenerator>>::allocate(
        runtime);
    StringAttr fastestFuncName =
        StringAttr::get(itfOp->getContext(), cacheContents->getBuffer());

    // Find the fastest function by name.
    auto fastest = llvm::find_if(candidates, [&](ElaboratedGenerator gen) {
      return gen.func.getNameAttr() == fastestFuncName;
    });
    if (fastest == candidates.end()) {
      out.setToError(LLCL::getMLIRDiagnostic(
          Error("could not find " + fastestFuncName.getValue()),
          itfOp->getLoc()));
    } else {
      out.emplace(*fastest);
    }

    return out;
  };

  // Run the transform with the functions we just defined.
  auto xform =
      Cache::cachedTransform(itf, transformCache.copy(),
                             LLCL::AsyncValueRef<Chain>::createReady(runtime),
                             std::move(keyBuf), doSpecialization, onCacheHit);

  LLCL::await(xform);
  result.clear();
  if (xform->isError())
    result.push_back(reportCalleeExpansionError(
        itf, xform->getDiagnostic().getMessage().get()));
  else
    result.push_back(std::move(xform->get<ErrorTreeOr<ElaboratedGenerator>>()));
  return result;
}

/// Return all instantiations of the specified declaration (a  generator or
/// interface) with the specified input parameter values.
ArrayRef<ErrorTreeOr<ElaboratedGenerator>>
Elaborator::getAllInstantiations(DeclAndInputParamsPair declAndInputParams,
                                 size_t expansionDepth, bool inlined) {
  // Check the global cache of instantiations so we only ever instantiate a
  // generator once.
  auto cacheIt = generatedFuncs.find(declAndInputParams);
  if (cacheIt != generatedFuncs.end())
    return cacheIt->second;

  Operation *decl = declAndInputParams.first;
  SmallVector<ErrorTreeOr<ElaboratedGenerator>> newCallees;
  auto localError = [&](ErrorTree err) {
    newCallees.push_back(std::move(err));
  };

  // Evaluate any constraints for this declaration to see if this is a viable
  // expansion.  If not, the expansion fails.
  LLVM_DEBUG({
    llvm::dbgs() << std::string(expansionDepth, ' ')
                 << "getAllInstantiations: ";
  });
  IREvaluator evaluator = createEvaluator();
  Optional<ErrorTree> err = evaluateConstraints(
      decl, declAndInputParams.second.getValue(), evaluator);
  if (err) {
    LLVM_DEBUG({ llvm::dbgs() << "evaluateConstraints failed\n"; });
    localError(std::move(*err));
  } else if (auto func = dyn_cast<FuncOp>(decl)) {
    // Nothing to do here. Just bind the result parameters and return the
    // function.
    LLVM_DEBUG({ llvm::dbgs() << "Func: " << func->getName() << "\n"; });
    bindResultParameters(func);
    newCallees.emplace_back(ElaboratedGenerator(func));
  } else if (isa<GeneratorOp>(decl)) {
    LLVM_DEBUG({
      llvm::dbgs() << "Generator: " << cast<GeneratorOp>(decl).getNameAttr()
                   << "\n";
    });
    newCallees =
        specializeGenerator(declAndInputParams, expansionDepth + 1, inlined);
  } else if (isa<GeneratorInterfaceOp>(decl)) {
    LLVM_DEBUG({
      llvm::dbgs() << "GenInterface: "
                   << cast<GeneratorInterfaceOp>(decl).getNameAttr() << "\n";
    });
    newCallees =
        specializeInterface(declAndInputParams, expansionDepth + 1, inlined);
  } else {
    localError({decl->getLoc(), "call to an unknown kind of declaration"});
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
  for (Operation &op : analysis.getModule().getOps()) {
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

/// Elaborate generators in the specified module, incorporating implementation
/// logic from the specified library.
LogicalResult M::elaborateGenerators(SymbolTableAnalysis &analysis,
                                     LLCL::Runtime &runtime,
                                     ArrayRef<GeneratorOp> primaryGenerators,
                                     bool enableSearch) {
  LLVM_DEBUG({
    llvm::dbgs() << "Elaborating top level generators:\n";
    for (auto generator : primaryGenerators)
      llvm::dbgs() << " * " << generator.getNameAttr() << "\n";
  });
  TimeTraceScope<> traceScope("elaborate-generators");
  ModuleOp primary = analysis.getModule();

  LLCL::AsyncSideEffectMap asyncMap(runtime);

  auto transformCacheBackendOr =
      Cache::getDefaultBackendChain(runtime, ".kgen_cache/transform");
  if (failed(transformCacheBackendOr))
    return primary->emitError() << transformCacheBackendOr.getError();
  auto regionCacheBackendOr =
      Cache::getDefaultBackendChain(runtime, ".kgen_cache/region");
  if (failed(regionCacheBackendOr))
    return primary->emitError() << regionCacheBackendOr.getError();

  auto transformCache =
      LLCL::RCRef<Cache::BlobCache<Cache::TransformCacheKey>>::create(
          transformCacheBackendOr.takeValue());
  auto regionCache =
      LLCL::RCRef<Cache::BlobCache<Cache::RegionCacheKey>>::create(
          regionCacheBackendOr.takeValue());

  // Deflate every op in the primary module. They'll be inflated at
  // specialization time.
  SmallVector<AnyAsyncValueRef> deflates;
  for (auto gen : primary.getOps<GeneratorOp>()) {
    asyncMap.mapChained(gen, [&](auto ch) {
      return Cache::deflateOp(gen, regionCache.copy(), std::move(ch));
    });
  }

  for (auto func : primary.getOps<FuncOp>()) {
    asyncMap.mapChained(func, [&](auto ch) {
      return Cache::deflateOp(func, regionCache.copy(), std::move(ch));
    });
  }

  Elaborator elaborator(analysis, runtime, asyncMap, transformCache.copy(),
                        regionCache.copy(), enableSearch);

  // Scan the primary and library module to collect all the interfaces,
  // verifying that any common interfaces are the same.
  if (elaborator.collectInterfaces())
    return failure();

  auto emptyInputParamKey = ArrayAttr::get(primary.getContext(), {});

  // Elaborate the bodies of all generators without input parameters in the
  // primary module.  These are roots that will cause callees to get
  // recursively elaborated.
  // TODO: When we have access control, we can limit this to just the publicly
  // exposed ones.
  bool didFail = false;
  [[maybe_unused]] size_t candidatesNumber = 0;
  for (GeneratorOp generatorRoot : primaryGenerators) {

    // Elaborate the generator into concrete versions.
    ArrayRef<ErrorTreeOr<ElaboratedGenerator>> results =
        elaborator.getAllInstantiations({generatorRoot, emptyInputParamKey}, 0);

    // If the generator failed to expand into /anything/ then emit an error.
    // Note that the func will have been deleted.
    if (llvm::all_of(results,
                     [](const ErrorTreeOr<ElaboratedGenerator> &result)
                         -> bool { return !result; })) {
      // Collect the errors together.
      ErrorTree error(generatorRoot.getLoc(),
                      "no viable implementations found");
      for (const ErrorTreeOr<ElaboratedGenerator> &value : results)
        error.addCause(value.getError().copy());
      error.emit([&](Location loc) { return emitError(loc); });
      didFail = true;
    }

    LLVM_DEBUG({
      size_t newCandidatesNumber =
          elaborator.getFirstConcreteFuncForGenerator().size();
      llvm::dbgs() << "Finished processing " << generatorRoot.getNameAttr()
                   << ", generated "
                   << std::to_string(newCandidatesNumber - candidatesNumber)
                   << " results.\n";
      candidatesNumber = newCandidatesNumber;
    });
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
  SymbolTable &symtab = analysis.getTopLevelSymbolTable();
  for (Operation &op : llvm::make_early_inc_range(primary.getOps())) {
    if (isa<GeneratorOp, GeneratorInterfaceOp>(op)) {
      symtab.erase(&op);
      continue;
    }

    /// Non viable funcs or inlined funcs will be left with an invalid body.
    /// Remove them at the end of elaboration.
    if (auto func = dyn_cast<FuncOp>(op)) {
      if (elaborator.shouldRemoveFunc(func)) {
        // Operations may have uses in inlined functions, which are invalid.
        // Drop all defined value uses before erasing the function.
        func->dropAllDefinedValueUses();
        symtab.erase(func);
      } else {
        // Make sure all funcs are inflated at the end of this, even if they
        // didn't participate in elaboration.
        asyncMap.map(func, Cache::inflateOp(func, regionCache.copy(),
                                            asyncMap.getChain(func)));
      }
    }
  }

  // Perform any renaming at the end.  We cannot use the
  // SymbolTable::replaceAllSymbolUses method, because it doesn't tolerate
  // unregistered operations.  It also doesn't support batch renaming.
  primary->walk([&](Operation *op) {
    // If this is a func being renamed, rename it.
    if (auto func = dyn_cast<FuncOp>(op)) {
      auto it = funcsToRename.find(func.getNameAttr());
      if (it != funcsToRename.end()) {
        // Keep the symbol table up-to-date.
        symtab.remove(func);
        func.setSymNameAttr(it->second);
        Block::iterator insertPt(func->getNextNode());
        func->remove();
        symtab.insert(func, insertPt);
      }
      return;
    }

    // If this is a reference to a function that got renamed, update its target.
    TypeSwitch<Operation *>(op).Case<CallOp, AddressOfOp>([&](auto op) {
      SymbolConstantAttr callee = op.getCallee();
      auto it = funcsToRename.find(
          cast<FlatSymbolRefAttr>(callee.getSymbol()).getAttr());
      if (it != funcsToRename.end())
        op.setCalleeAttr(SymbolConstantAttr::get(
            FlatSymbolRefAttr::get(it->second), callee.getType()));
    });
  });

  // Await all async values. The rest of the elaborator can't handle asynchrony
  // yet.
  asyncMap.awaitAll();
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
                          LLCL::Runtime &runtime,
                          const ElaborateGeneratorsOptions &options)
      : ElaborateGeneratorsBase(options), includedFiles(&includedFiles) {}
  using ElaborateGeneratorsBase::ElaborateGeneratorsBase;

  void runOnOperation() override {
    auto rt = ConditionallyOwnedPointer<LLCL::Runtime>::allocateIfNeeded(
        runtime, LLCL::createLeakCheckAllocator(LLCL::createMallocAllocator()),
        LLCL::createSingleThreadWorkQueue());

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

    auto &analysis = getAnalysis<SymbolTableAnalysis>();
    if (failed(resolveIncludes(analysis.getTopLevelSymbolTable(), paths,
                               includedFiles)))
      return signalPassFailure();

    if (failed(elaborateGenerators(analysis, *rt, primaryGenerators,
                                   shouldDoSearch)))
      return signalPassFailure();
  }

  /// An optional set of included files that were found during processing.
  SmallVectorImpl<std::string> *includedFiles = nullptr;
  LLCL::Runtime *runtime = nullptr;
};

/// Resolve includes in a pass. This pass only does include resolution.
struct ResolveIncludesPass
    : public KGEN::impl::ResolveIncludesBase<ResolveIncludesPass> {
  using ResolveIncludesBase::ResolveIncludesBase;

  void runOnOperation() override {
    SmallVector<std::filesystem::path> paths;
    for (const auto &p : searchPaths)
      paths.push_back(p);
    paths.push_back(std::filesystem::path("."));

    auto &analysis = getAnalysis<SymbolTableAnalysis>();
    if (failed(resolveIncludes(analysis.getTopLevelSymbolTable(), paths)))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
KGEN::createElaborateGenerators(SmallVectorImpl<std::string> &includedFiles,
                                LLCL::Runtime &runtime,
                                const ElaborateGeneratorsOptions &options) {
  return std::make_unique<ElaborateGeneratorsPass>(includedFiles, runtime,
                                                   options);
}
