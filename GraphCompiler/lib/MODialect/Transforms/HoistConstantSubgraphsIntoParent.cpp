//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/Analysis/ParametricTopologyAnalysis.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOInterfaces.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "Mojo/KGENDialect/AttrTypeMangler.h"
#include "Mojo/KGENDialect/KGENOps.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "hoist-constant-subgraphs-into-parent"

using namespace mlir;

namespace M::MO {
#define GEN_PASS_DEF_HOISTCONSTANTSUBGRAPHSINTOPARENT
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"
} // namespace M::MO

using namespace M;
using namespace M::MO;

namespace {

//===----------------------------------------------------------------------===//
// HoistConstantSubgraphsIntoParent Pass
//===----------------------------------------------------------------------===//
//
// Hoists ops out of subgraphs into each call site so passes like
// ExecInvariantCodeMotion can move them to the init block. An argument is
// callsite-invariant when every call site passes the same SSA value; ops that
// depend only on such arguments are hoisted, and the values they feed to the
// rest of the subgraph become new subgraph arguments.
//
// Subgraphs are processed callees first. Only the acyclic portion of the call
// graph is handled; cycles and disconnected components are ignored.
//===----------------------------------------------------------------------===//

enum class ConstLattice {
  NonConst, // Not constant-derived
  Const     // Constant-derived
};

struct LatticeState {

  // A block argument (graph input constness) or an op (constness of all its
  // results).
  using ProgramPoint = llvm::PointerUnion<BlockArgument, Operation *>;

  ProgramPoint getKey(Value v) const {
    if (auto arg = dyn_cast<BlockArgument>(v))
      return arg;
    return v.getDefiningOp();
  }

  ConstLattice query(Value v) const {
    return state.lookup_or(getKey(v), ConstLattice::NonConst);
  }

  ConstLattice query(Operation *v) const {
    return state.lookup_or(v, ConstLattice::NonConst);
  }

  void set(Value v, ConstLattice l) { state[getKey(v)] = l; }

  void set(Operation *v, ConstLattice l) { state[v] = l; }

  bool update(Operation *v, ConstLattice l) {
    auto oldValue = query(v);
    if (oldValue != l) {
      set(v, l);
      return true;
    }

    return false;
  }

  llvm::DenseMap<ProgramPoint, ConstLattice> state;
};

struct HoistConstantSubgraphsIntoParentPass
    : public M::MO::impl::HoistConstantSubgraphsIntoParentBase<
          HoistConstantSubgraphsIntoParentPass> {
  using HoistConstantSubgraphsIntoParentBase::
      HoistConstantSubgraphsIntoParentBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Starting pass\n");

    llvm::DenseMap<GraphOp, SmallVector<CallOp>> graphToCallsites;
    SmallVector<GraphOp> sortedGraphs;

    buildCallGraphAndSortGraphs(moduleOp, graphToCallsites, sortedGraphs);

    for (GraphOp graphOp : sortedGraphs) {
      auto &callsites = graphToCallsites[graphOp];
      [[maybe_unused]] StringRef graphName = graphOp.getName();

      if (callsites.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "Skipping graph with no callsites: "
                                << graphName << "\n");
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "Processing graph: " << graphName << " with "
                              << callsites.size() << " callsites\n");

      processGraph(graphOp, callsites);
    }
  }

private:
  void buildCallGraphAndSortGraphs(
      ModuleOp module,
      llvm::DenseMap<GraphOp, SmallVector<CallOp>> &graphToCallsites,
      SmallVector<GraphOp> &sortedGraphs) {

    llvm::DenseMap<StringRef, GraphOp> symbolToGraphOp;
    // Caller -> callees.
    llvm::DenseMap<GraphOp, llvm::SetVector<GraphOp>> callGraph;

    SmallVector<GraphOp> graphs;
    SmallVector<GraphOp> entryPoints;

    for (GraphOp graphOp : module.getOps<GraphOp>()) {
      StringRef name = graphOp.getName();
      symbolToGraphOp[name] = graphOp;
      graphs.push_back(graphOp);

      if (!graphOp.getIsSubgraph())
        entryPoints.push_back(graphOp);

      LLVM_DEBUG(llvm::dbgs() << "Found graph: " << name << "\n");
    }

    module.walk([&](CallOp callOp) {
      StringRef calleeName = callOp.getCallee();
      auto it = symbolToGraphOp.find(calleeName);
      if (it == symbolToGraphOp.end()) {
        callOp.emitError() << "Could not find callee graph named '"
                           << calleeName << "'";
        signalPassFailure();
        return;
      }

      GraphOp callee = it->second;
      graphToCallsites[callee].push_back(callOp);

      GraphOp parentGraph = callOp->getParentOfType<GraphOp>();
      if (!parentGraph) {
        callOp.emitError() << "Call op must be contained within a GraphOp";
        signalPassFailure();
        return;
      }

      callGraph[parentGraph].insert(callee);

      LLVM_DEBUG(llvm::dbgs() << "Found callsite for graph: " << calleeName
                              << " at " << callOp.getLoc() << "\n");
    });

    LLVM_DEBUG({
      llvm::dbgs() << "Entry points in call graph:\n";
      for (GraphOp entry : entryPoints)
        llvm::dbgs() << "  " << entry.getName() << "\n";
    });

    llvm::DenseSet<GraphOp> visited;

    std::function<void(GraphOp)> dfs = [&](GraphOp node) {
      if (!visited.insert(node).second)
        return;

      if (auto it = callGraph.find(node); it != callGraph.end())
        llvm::for_each(it->second, dfs);

      sortedGraphs.push_back(node);
    };

    for (GraphOp entry : entryPoints)
      dfs(entry);

    std::reverse(sortedGraphs.begin(), sortedGraphs.end());

    LLVM_DEBUG({
      llvm::dbgs() << "Processing graphs in order:\n";
      for (GraphOp g : sortedGraphs)
        llvm::dbgs() << "  " << g.getName() << "\n";
    });
  }

  void processGraph(GraphOp graphOp, ArrayRef<CallOp> callsites) {
    unsigned numArgs = graphOp.getBody().getArguments().size();
    if (numArgs == 0) {
      LLVM_DEBUG(llvm::dbgs() << "Graph has no arguments, skipping\n");
      return;
    }

    llvm::BitVector effectiveInvariantArgMask =
        computeEffectiveInvariantArgMask(graphOp, callsites);

    if (!effectiveInvariantArgMask.any()) {
      LLVM_DEBUG(llvm::dbgs() << "No effectively invariant arguments found for "
                              << graphOp.getName() << "\n");
      return;
    }

    LatticeState lattice;
    performDataflowAnalysis(graphOp, effectiveInvariantArgMask, lattice);

    llvm::SmallSetVector<Operation *, 16> constantOps;
    SetVector<Value> boundaryValues;
    llvm::BitVector preservedArgs = effectiveInvariantArgMask;
    preservedArgs.flip(); // Non-invariant args are always preserved.
    SmallVector<Operation *> sortedConstantOps;

    findConstantOpsAndBoundaryValues(
        graphOp, lattice, effectiveInvariantArgMask, constantOps,
        boundaryValues, preservedArgs, sortedConstantOps);

    if (boundaryValues.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No boundary values to hoist for "
                              << graphOp.getName() << "\n");
      return;
    }

    auto newCallsites = hoistConstantOperations(
        graphOp, callsites, preservedArgs, effectiveInvariantArgMask,
        constantOps, boundaryValues, sortedConstantOps);

    updateSubgraphDefinition(graphOp, preservedArgs, boundaryValues,
                             sortedConstantOps);

    moveChainsToTheEnd(graphOp, newCallsites);
  }

  void moveChainsToTheEnd(GraphOp subgraphGraphOp, ArrayRef<CallOp> callsites) {
    SmallVector<uint64_t> argumentsToErase;
    auto block = subgraphGraphOp.getBodyBlock();
    // Collect first: inserting an argument can reallocate the block's argument
    // list, which would leave an iterator over it dangling.
    SmallVector<BlockArgument> chainArgs;
    for (BlockArgument arg : block->getArguments())
      if (isa<ChainType>(arg.getType()))
        chainArgs.push_back(arg);

    for (BlockArgument arg : chainArgs) {
      auto newArg = block->insertArgument(block->getNumArguments(),
                                          arg.getType(), arg.getLoc());
      arg.replaceAllUsesWith(newArg);
      argumentsToErase.push_back(arg.getArgNumber());
    }

    for (uint64_t idx : llvm::reverse(argumentsToErase))
      block->eraseArgument(idx);

    synchronizeMetadataUpdates(subgraphGraphOp);

    for (CallOp callOp : callsites) {
      OpBuilder builder(callOp);
      SmallVector<Value> newOperands;
      SmallVector<Value> chains;
      for (auto &&operand : callOp->getOperands()) {
        if (isa<ChainType>(operand.getType()))
          chains.push_back(operand);
        else
          newOperands.push_back(operand);
      }
      newOperands.insert(newOperands.end(), chains.begin(), chains.end());

      auto newCallOp = CallOp::create(
          builder, callOp->getLoc(), callOp->getResultTypes(), newOperands,
          callOp.getCalleeAttr(), callOp.getPrefixAttr(),
          /*outputParamDecls=*/{},
          /*arg_attrs=*/{},
          /*res_attrs=*/{});
      callOp->replaceAllUsesWith(newCallOp.getResults());
      callOp->erase();
    }
  }

  llvm::BitVector computeEffectiveInvariantArgMask(GraphOp graphOp,
                                                   ArrayRef<CallOp> callsites) {
    unsigned numArgs = graphOp.getBody().getArguments().size();
    if (numArgs == 0)
      return llvm::BitVector();

    llvm::BitVector effectiveInvariantMask(numArgs, false);

    assert(!callsites.empty() && "Callsites should not be empty here.");

    for (unsigned i = 0; i < numArgs; ++i) {
      Value firstOperandValue = callsites[0]->getOperand(i);
      bool allSameValue = true;
      for (size_t cIdx = 1; cIdx < callsites.size(); ++cIdx) {
        if (callsites[cIdx]->getOperand(i) != firstOperandValue) {
          allSameValue = false;
          break;
        }
      }

      if (allSameValue && callsites.size() > 1) {
        effectiveInvariantMask[i] = true;
        LLVM_DEBUG(llvm::dbgs()
                   << "Arg " << i
                   << " is callsite-invariant (all same parent value: "
                   << firstOperandValue << ").\n");
      } else {
        effectiveInvariantMask[i] = false;
        LLVM_DEBUG(llvm::dbgs()
                   << "Arg " << i << " is NOT callsite-invariant.\n");
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Callsite-invariant arguments mask for graph "
                   << graphOp.getName() << ": ";
      for (unsigned i : effectiveInvariantMask.set_bits())
        llvm::dbgs() << i << " ";
      llvm::dbgs() << "\n";
    });

    return effectiveInvariantMask;
  }

  void performDataflowAnalysis(GraphOp graphOp,
                               const llvm::BitVector &effectiveInvariantArgMask,
                               LatticeState &lattice) {
    ParametricTopologyAnalysis topological(graphOp);

    auto allInputsConstant = [&](Operation *op) {
      return topological.forEachProvider(op, [&](Operation *input) {
        if (input == graphOp)
          return true;
        return lattice.query(input) == ConstLattice::Const;
      }) && llvm::all_of(op->getOperands(), [&](Value v) {
               return lattice.query(v) == ConstLattice::Const;
             });
    };

    Block *body = graphOp.getBodyBlock();
    SmallVector<Operation *> worklist;

    for (auto [idx, arg] : llvm::enumerate(body->getArguments())) {
      lattice.set(arg, effectiveInvariantArgMask[idx] ? ConstLattice::Const
                                                      : ConstLattice::NonConst);
    }

    for (Operation &op : body->without_terminator())
      worklist.push_back(&op);

    while (!worklist.empty()) {
      Operation *op = worklist.back();
      worklist.pop_back();

      bool allOperandsConst =
          (op->getNumOperands() > 0 || isa<KGEN::ParamDeclareOp>(op)) &&
          allInputsConstant(op);

      ConstLattice newState =
          allOperandsConst ? ConstLattice::Const : ConstLattice::NonConst;

      if (lattice.update(op, newState)) {
        topological.forEachUser(op, [&](Operation *user) {
          if (!user->hasTrait<OpTrait::IsTerminator>() &&
              !llvm::is_contained(worklist, user))
            worklist.push_back(user);

          return true;
        });
      }
    }
  }

  // Boundary values are invariant-derived values used by non-hoisted ops.
  void findConstantOpsAndBoundaryValues(
      GraphOp graphOp, const LatticeState &lattice,
      const llvm::BitVector &effectiveInvariantArgMask,
      llvm::SmallSetVector<Operation *, 16> &constantOps,
      SetVector<Value> &boundaryValues, llvm::BitVector &preservedArgs,
      SmallVector<Operation *> &sortedConstantOps) {

    Block *body = graphOp.getBodyBlock();

    for (Operation &op : body->without_terminator()) {
      if (lattice.query(&op) == ConstLattice::Const)
        constantOps.insert(&op);
    }

    // An invariant argument used by a non-hoisted op must be preserved.
    for (unsigned i : effectiveInvariantArgMask.set_bits()) {
      BlockArgument arg = body->getArgument(i);

      for (Operation *user : arg.getUsers()) {
        if (!constantOps.contains(user)) {
          preservedArgs[i] = true;
          LLVM_DEBUG(
              llvm::dbgs()
              << "Arg " << i
              << " is callsite-invariant but used by non-invariant-derived op: "
              << *user << ", keeping it\n");
          break;
        }
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Arguments to preserve: ";
      for (unsigned i : preservedArgs.set_bits())
        llvm::dbgs() << i << " ";
      llvm::dbgs() << "\n";
    });

    sortedConstantOps.reserve(constantOps.size());
    llvm::DenseSet<Operation *> visited;

    std::function<void(Operation *)> visitOp = [&](Operation *op) {
      if (visited.contains(op))
        return;

      for (Value operand : op->getOperands()) {
        if (Operation *defOp = operand.getDefiningOp()) {
          if (constantOps.contains(defOp))
            visitOp(defOp);
        }
      }

      visited.insert(op);
      sortedConstantOps.push_back(op);
    };

    for (Operation *op : constantOps) {
      if (!visited.contains(op))
        visitOp(op);
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Invariant-derived ops in topological order:\n";
      for (Operation *op : sortedConstantOps)
        llvm::dbgs() << "  " << *op << "\n";
    });

    for (Operation *op : constantOps) {
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (!constantOps.contains(user) &&
              !user->hasTrait<OpTrait::IsTerminator>()) {
            boundaryValues.insert(result);
            break;
          }
        }
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Found " << constantOps.size()
                   << " invariant-derived ops and " << boundaryValues.size()
                   << " boundary values\n";

      llvm::dbgs() << "Boundary values: ";
      for (Value boundaryVal : boundaryValues)
        llvm::dbgs() << boundaryVal << "\n";

      llvm::dbgs() << "\n";
    });
  }

  SmallVector<CallOp> hoistConstantOperations(
      GraphOp subgraphGraphOp, ArrayRef<CallOp> callsites,
      const llvm::BitVector &preservedArgs,
      const llvm::BitVector &effectiveInvariantArgMask,
      const llvm::SmallSetVector<Operation *, 16> &constantOps,
      const SetVector<Value> &boundaryValues,
      const SmallVector<Operation *> &sortedConstantOps) {

    Block *body = subgraphGraphOp.getBodyBlock();
    KGEN::ParameterUseDefGraph subgraphParamUseDefGraph(
        subgraphGraphOp.getBody());
    KGEN::ParameterCollector::Analysis analysisCache;
    subgraphParamUseDefGraph.calculate(analysisCache);
    DenseMap<StringAttr, KGEN::ParamDeclaration> &subgraphDecls =
        subgraphParamUseDefGraph.decls;
    llvm::SetVector<StringAttr> subgraphDeclNames;
    subgraphDeclNames.insert_range(
        llvm::make_first_range(llvm::make_filter_range(
            subgraphDecls,
            [&](const std::pair<StringAttr, KGEN::ParamDeclaration> &decl) {
              return decl.second.declOp != subgraphGraphOp;
            })));

    SmallVector<CallOp> newCallSites;
    for (CallOp callOp : callsites) {
      OpBuilder builder(callOp);
      IRMapping valueMapping;
      GraphOp parentGraphOp = callOp->getParentOfType<MO::GraphOp>();
      Region &parentGraphRegion = parentGraphOp.getBody();
      KGEN::AttrTypeMangler::Cache manglerCache;
      KGEN::AttrTypeMangler mangler(manglerCache);
      KGEN::ParameterUseDefGraph paramGraph(parentGraphRegion);
      paramGraph.calculate(analysisCache);
      KGEN::NameUniquer uniquer(paramGraph, paramGraph);
      mangler.populate(builder, uniquer, subgraphDeclNames, paramGraph);

      assert(parentGraphOp && "expected parent graph op");

      for (unsigned i : effectiveInvariantArgMask.set_bits())
        valueMapping.map(body->getArgument(i), callOp.getOperand(i));

      for (Operation *op : sortedConstantOps) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Cloning invariant-derived operation: " << *op << "\n");
        Operation *clonedOp = builder.clone(*op, valueMapping);

        if (auto itf = dyn_cast<KGEN::ParamOpInterface>(*clonedOp)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Processing ParamOpInterface for operation: "
                     << *clonedOp << "\n");
          SmallVector<KGEN::ParamDeclAttr> newDecls;
          itf.walkDeclarations([&](KGEN::ParamDeclAttr decl) {
            LLVM_DEBUG(llvm::dbgs()
                       << "Walking declaration: " << decl.getName() << "\n");
            newDecls.push_back(
                mangler.mangleDecl(decl, /*needsMangling=*/true));
          });
          cast<KGEN::ParamOpInterface>(clonedOp).renameDeclarations(newDecls);
          // Register the new declarations in the call scope's param graph.
          for (KGEN::ParamDeclAttr decl : newDecls) {
            LLVM_DEBUG(llvm::dbgs() << "Adding declaration to param graph: "
                                    << decl.getName() << "\n");
            paramGraph.decls.try_emplace(
                decl.getName(), KGEN::ParamDeclaration{decl.getType(), clonedOp,
                                                       &parentGraphRegion});
          }
          parentGraphOp.setCounter(parentGraphOp.getCounter() +
                                   newDecls.size());
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "Mangling elements in operation: " << *clonedOp << "\n");
        mangler.mangleElementsIn(clonedOp);

        for (unsigned i = 0; i < op->getNumResults(); ++i)
          valueMapping.map(op->getResult(i), clonedOp->getResult(i));
      }

      SmallVector<Value> newCallOperands;

      for (unsigned i : preservedArgs.set_bits())
        newCallOperands.push_back(callOp.getOperand(i));

      for (Value boundaryVal : boundaryValues) {
        Value mappedVal = valueMapping.lookup(boundaryVal);
        newCallOperands.push_back(mappedVal);
      }

      auto newCallOp = CallOp::create(
          builder, callOp.getLoc(), callOp.getResultTypes(), newCallOperands,
          callOp.getCalleeAttr(), callOp.getPrefixAttr(),
          /*outputParamDecls=*/{},
          /*arg_attrs=*/{},
          /*res_attrs=*/{});

      callOp.replaceAllUsesWith(newCallOp.getResults());

      LLVM_DEBUG(llvm::dbgs() << "Replaced call at " << callOp.getLoc()
                              << " with new call\n");

      newCallSites.push_back(newCallOp);
      callOp.erase();
    }

    return newCallSites;
  }

  void
  updateSubgraphDefinition(GraphOp graphOp,
                           const llvm::BitVector &preservedArgs,
                           const SetVector<Value> &boundaryValues,
                           const SmallVector<Operation *> &sortedConstantOps) {

    Block *body = graphOp.getBodyBlock();
    unsigned numArgs = body->getArguments().size();

    // Reverse order for safe erasure; BitVector has no reverse set_bits().
    SmallVector<unsigned> argsToDelete;
    for (int i = numArgs - 1; i >= 0; --i)
      if (!preservedArgs[i])
        argsToDelete.push_back(i);

    unsigned insertionIndex = numArgs;
    for (Value boundaryVal : boundaryValues) {
      BlockArgument newArg = body->insertArgument(
          insertionIndex++, boundaryVal.getType(), graphOp.getLoc());
      boundaryVal.replaceAllUsesWith(newArg);
      LLVM_DEBUG(llvm::dbgs()
                 << "Replaced boundary value with new arg " << newArg << "\n");
    }

    for (auto it = sortedConstantOps.rbegin(); it != sortedConstantOps.rend();
         ++it) {
      LLVM_DEBUG(llvm::dbgs() << "Erasing constant op: " << **it << "\n");
      (*it)->erase();
    }

    for (unsigned idx : argsToDelete) {
      LLVM_DEBUG(llvm::dbgs() << "Erasing non-preserved arg: " << idx << "\n");
      body->eraseArgument(idx);
    }

    synchronizeMetadataUpdates(graphOp);
  }
};

} // namespace
