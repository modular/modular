//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/HLCFDialect/Analysis/DataFlow.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace POP;
using namespace mlir::dataflow;

namespace {

/// This state represents the known possible types of a variant.
struct VariantTypes {
  static VariantTypes join(const VariantTypes &lhs, const VariantTypes &rhs) {
    auto knownTypes = lhs.knownTypes;
    knownTypes.insert(rhs.knownTypes.begin(), rhs.knownTypes.end());
    return {std::move(knownTypes)};
  }

  bool operator==(const VariantTypes &rhs) const {
    return knownTypes.getArrayRef() == rhs.knownTypes.getArrayRef();
  }

  void print(raw_ostream &os) const {
    os << '{';
    llvm::interleaveComma(knownTypes, os);
    os << '}';
  }

  llvm::SetVector<Type, SmallVector<Type, 4>, SmallPtrSet<Type, 4>> knownTypes;
};

struct VariantState : public Lattice<VariantTypes> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VariantState);

  using Lattice::Lattice;
};

struct KnownVariantAnalysis : public SparseDataFlowAnalysis<VariantState> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KnownVariantAnalysis);

  using SparseDataFlowAnalysis::SparseDataFlowAnalysis;

  void visitOperation(Operation *op, ArrayRef<const VariantState *> operands,
                      ArrayRef<VariantState *> results) override {
    if (auto create = dyn_cast<VariantCreateOp>(op)) {
      VariantTypes value;
      value.knownTypes.insert(create.getOperand().getType());
      propagateIfChanged(results.front(), results.front()->join(value));
      return;
    }
    setAllToEntryStates(results);
  }

  void setToEntryState(VariantState *state) override {
    if (auto variant = dyn_cast<VariantType>(state->getPoint().getType())) {
      SmallVector<Type> types = variant.getParameterizedElementTypes();
      VariantTypes value;
      value.knownTypes.insert(types.begin(), types.end());
      propagateIfChanged(state, state->join(value));
    }
  }
};

/// Subclass DeadCodeAnalysis to inject our transfer function.
struct VariantAwareDeadCodeAnalysis : public HLCF::DeadCodeAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VariantAwareDeadCodeAnalysis);

  using DeadCodeAnalysis::DeadCodeAnalysis;

  LogicalResult visit(mlir::ProgramPoint point) override {
    auto *op = point.dyn_cast<Operation *>();
    auto visit = dyn_cast_if_present<VariantVisitOp>(op);
    if (!visit)
      return DeadCodeAnalysis::visit(point);

    const VariantTypes &value =
        getOrCreateFor<VariantState>(point, visit.getVariant())->getValue();
    // Mark any case region of a known type to be live. If there is a known type
    // that does not have a case region, the default region is live.
    auto markRegionLive = [&](Region *region) {
      auto *executable = getOrCreate<Executable>(&region->front());
      propagateIfChanged(executable, executable->setToLive());
      auto *predecessors = getOrCreate<PredecessorState>(&region->front());
      propagateIfChanged(predecessors,
                         predecessors->join(visit, visit->getOperands()));
    };
    unsigned casesHit = 0;
    for (auto [caseType, region] :
         llvm::zip(visit.getCases(), visit.getRegions())) {
      if (value.knownTypes.contains(caseType)) {
        ++casesHit;
        markRegionLive(region);
      }
    }
    if (!casesHit && visit.hasDefaultRegion())
      markRegionLive(visit.getDefaultRegion());
    return success();
  }
};

/// Subclass SparseConstantPropagation to inject our transfer function.
struct ConstantPropagation : public HLCF::SparseConstantPropagation {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstantPropagation);

  using SparseConstantPropagation::SparseConstantPropagation;

  void visitOperation(Operation *op,
                      ArrayRef<const Lattice<ConstantValue> *> operands,
                      ArrayRef<Lattice<ConstantValue> *> results) override {
    if (auto test = dyn_cast<VariantIsOp>(op)) {
      const VariantTypes &value =
          getOrCreateFor<VariantState>(op, test.getOperand())->getValue();
      if (value.knownTypes.empty())
        return;
      // If the set does not contain the type, optimistically assume false.
      // Otherwise, if the set contains only the type, optimistically assume
      // true. In any other case, we cannot definitively assume a value (and the
      // state reaches a pessimistic fixpoint).
      Attribute cvValue;
      if (!value.knownTypes.contains(test.getTestType()))
        cvValue = BoolAttr::get(op->getContext(), /*value=*/false);
      else if (value.knownTypes.size() == 1)
        cvValue = BoolAttr::get(op->getContext(), /*value=*/true);
      propagateIfChanged(results.front(), results.front()->join(ConstantValue(
                                              cvValue, /*dialect=*/nullptr)));
      return;
    }

    SparseConstantPropagation::visitOperation(op, operands, results);
  }
};

} // namespace

namespace M::KGEN {
#define GEN_PASS_DEF_PRUNEIMPOSSIBLEVARIANTS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct PruneImpossibleVariantsPass
    : M::KGEN::impl::PruneImpossibleVariantsBase<PruneImpossibleVariantsPass> {
  void runOnOperation() override;
};
} // namespace

void PruneImpossibleVariantsPass::runOnOperation() {
  mlir::DataFlowSolver solver;
  solver.load<VariantAwareDeadCodeAnalysis>(getAnalysisManager());
  solver
      .load<HLCF::SparseDataFlowAnalysis<VariantState, KnownVariantAnalysis>>();
  solver.load<ConstantPropagation>();

  // Compute the string attributes once.
  Builder b(&getContext());
  StringAttr symVisibilityAttrName =
      b.getStringAttr(SymbolTable::getVisibilityAttrName());
  StringAttr publicAttr = b.getStringAttr("public");
  StringAttr privateAttr = b.getStringAttr("private");

  SmallPtrSet<StringAttr, 5> exportedSymbols;
  for (auto e : getOperation().getOps<ExportOp>())
    for (auto sym : e.getExports().getAsRange<FlatSymbolRefAttr>())
      exportedSymbols.insert(sym.getAttr());

  std::vector<Operation *> funcOrGenerator;
  for (Operation &op : getOperation().getOps()) {
    TypeSwitch<Operation *>(&op).Case<FuncOp, GeneratorOp>([&](auto funcOrGen) {
      if (exportedSymbols.contains(funcOrGen.getSymNameAttr()))
        op.setAttr(symVisibilityAttrName, publicAttr);
      else
        op.setAttr(symVisibilityAttrName, privateAttr);
      funcOrGenerator.push_back(funcOrGen);
    });
  }

  if (failed(solver.initializeAndRun(getOperation())))
    return signalPassFailure();

  // Functions referred by symbol parameters cannot be rewritten.
  DenseSet<SymbolRefAttr> refd;
  std::vector<CallOp> calls;
  getOperation()->walk([&](Operation *op) {
    op->getAttrDictionary().walkSubAttrs([&](Attribute attr) {
      if (auto symbol = dyn_cast<SymbolConstantAttr>(attr))
        refd.insert(symbol.getSymbol());
    });

    // Replace `pop.variant.is` ops on variants with known types with constants.
    if (auto is = dyn_cast<VariantIsOp>(op)) {
      auto *cv = solver.lookupState<Lattice<ConstantValue>>(is.getResult());
      if (!cv || cv->getValue().isUninitialized() ||
          !cv->getValue().getConstantValue())
        return;
      OpBuilder b(is);
      Value value = b.create<ParamConstantOp>(
          is.getLoc(), cv->getValue().getConstantValue());
      is.replaceAllUsesWith(value);
      is->erase();
    } else if (auto call = dyn_cast<CallOp>(op)) {
      // Save the call op for rewrites later.
      calls.push_back(call);
    }
  });

  // Rewrite the signatures of operations that return variants that are known to
  // be a particular type.
  DenseMap<StringAttr, SmallVector<std::pair<unsigned, Type>>> rewrites;
  for (Operation *op : funcOrGenerator) {
    op->removeAttr(symVisibilityAttrName);
    StringAttr name;
    if (auto gen = dyn_cast<GeneratorOp>(op)) {
      // Don't rewrite generators implementing interfaces.
      if (gen.getImplements())
        continue;
      name = gen.getSymNameAttr();
    } else {
      name = cast<FuncOp>(op).getSymNameAttr();
    }
    if (refd.contains(FlatSymbolRefAttr::get(name)))
      continue;

    auto func = cast<mlir::FunctionOpInterface>(op);

    // Reduce the known variant types across all returns.
    SmallVector<Operation *> returns;
    func.walk([&](Operation *op) {
      if (isa<ReturnOp, HLCF::ReturnOp>(op))
        returns.push_back(op);
    });

    SmallVector<Optional<VariantTypes>> types;
    for (auto [i, type] : llvm::enumerate(func.getResultTypes())) {
      if (!isa<VariantType>(type)) {
        types.push_back(None);
        continue;
      }
      // Merge the known variant types across all reachable returns.
      VariantTypes merged;
      for (Operation *ret : returns) {
        // Ignore the return if its parent block is dead.
        if (!solver.lookupState<Executable>(ret->getBlock())->isLive())
          continue;
        bool reachable = true;
        // Determine if the return is reachable.
        for (Operation &op :
             llvm::reverse(ret->getBlock()->without_terminator())) {
          if (!isa<mlir::CallOpInterface, mlir::RegionBranchOpInterface,
                   HLCF::ControlFlowNode>(op))
            continue;
          auto *preds = solver.lookupState<PredecessorState>(&op);
          reachable = preds && (!preds->allPredecessorsKnown() ||
                                !preds->getKnownPredecessors().empty());
          break;
        }
        // If the return is reachable, merge in its state.
        if (reachable) {
          auto *state = solver.lookupState<VariantState>(ret->getOperand(i));
          merged = VariantTypes::join(merged, state->getValue());
        }
      }
      types.push_back(std::move(merged));
    }

    // Rewrite all variant operands of returns known to be a particular type.
    SmallVector<std::pair<unsigned, Type>> resultRewrites;
    for (auto [idx, type] : llvm::enumerate(types)) {
      if (!type || type->knownTypes.size() != 1)
        continue;
      Type knownType = type->knownTypes.front();
      resultRewrites.emplace_back(idx, knownType);
      for (Operation *ret : returns) {
        OpBuilder b(ret);
        Value result = b.create<VariantGetOp>(ret->getLoc(), knownType,
                                              ret->getOperand(idx));
        ret->setOperand(idx, result);
      }
    }

    if (!resultRewrites.empty()) {
      auto itf = cast<KGENDeclInterface>(op);
      auto sig = itf.getSignature();
      // Rewrite the function type.
      auto fnType = FunctionType::get(&getContext(), sig.getValueInputs(),
                                      returns.front()->getOperandTypes());
      itf.setSignature(SignatureType::get(sig.getInputParams(),
                                          sig.getResultParamTypes(), fnType));
      rewrites.try_emplace(name, std::move(resultRewrites));
    }
  }

  // Go rewrite all the callsites where variants results are known to be a
  // particular type.
  for (CallOp call : calls) {
    // FIXME(Issue#5471): This probably isn't correct for scoped symbols.
    auto it = rewrites.find(call.getCalleeAttr().getRootReference());
    if (it == rewrites.end())
      continue;
    OpBuilder b(&getContext());
    b.setInsertionPointAfter(call);
    // Wrap rewritten results and change their type.
    for (auto [idx, type] : it->second) {
      OpResult result = call->getOpResult(idx);
      auto create =
          b.create<VariantCreateOp>(call.getLoc(), result.getType(), result);
      result.replaceAllUsesExcept(create.getResult(), create);
      result.setType(type);
    }
  }
}
