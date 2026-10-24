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

#include "AsyncRT/CompilerSupport/Context.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Mojo/TransformUtils/SCCUtils.h"
#include "Support/ML/CommonAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_MERGEIDENTICALFUNCS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

namespace M::MOGG {

namespace {

// A stub with its precomputed hash, so repeated insertions don't rehash.
struct KernelStubHashEntry {
  MOGG::FuncOp op;
  unsigned hash;

  // Will be called in isEqual only.
  bool operator==(const KernelStubHashEntry &other) const {
    return other.op == op;
  }
};

struct KernelStubOpEquivalenceInfo
    : public llvm::DenseMapInfo<KernelStubHashEntry> {

  static unsigned getHashValue(const KernelStubHashEntry cStubOp) {
    return cStubOp.hash;
  }
  static bool isEqual(KernelStubHashEntry lhs, KernelStubHashEntry rhs) {
    if (lhs.op == rhs.op)
      return true;

    // Check if properties are equivalent modulo sym_name and fusionInfo
    // attributes.
    auto lhsProperties = lhs.op.getProperties();
    auto rhsProperties = rhs.op.getProperties();

    lhsProperties.sym_name = rhsProperties.sym_name = {};
    lhsProperties.fusionInfo = rhsProperties.fusionInfo = {};

    if (lhsProperties != rhsProperties)
      return false;

    // Check if discardable attributes are equivalent.
    DictionaryAttr lhsDictAttr = lhs.op->getRawDictionaryAttrs();
    DictionaryAttr rhsDictAttr = rhs.op->getRawDictionaryAttrs();
    if (lhsDictAttr.size() != rhsDictAttr.size())
      return false;

    for (auto [lhsAttr, rhsAttr] : llvm::zip_equal(lhsDictAttr, rhsDictAttr)) {
      if (lhsAttr.getName() != rhsAttr.getName())
        return false;

      if (lhsAttr.getValue() == rhsAttr.getValue())
        continue;

      auto attrName = lhsAttr.getName();
      if (attrName != MO::kOperationDeviceRefAttrName)
        return false;

      // Do not compare the device ID, just the label.
      if (cast<DeviceRefAttr>(lhsAttr.getValue()).getLabel() !=
          cast<DeviceRefAttr>(rhsAttr.getValue()).getLabel())
        return false;
    }

    return mlir::OperationEquivalence::isRegionEquivalentTo(
        &lhs.op.getBodyRegion(), &rhs.op.getBodyRegion(),
        /*flags=*/mlir::OperationEquivalence::Flags::IgnoreLocations);
  }
};

static unsigned getHashValue(const MOGG::FuncOp cStubOp) {
  llvm::hash_code hash = {};
  MOGG::FuncOp stubOp = const_cast<MOGG::FuncOp &>(cStubOp);

  auto properties = stubOp.getProperties();
  properties.sym_name = {};
  properties.fusionInfo = {};

  stubOp->getName().hashOpProperties(mlir::PropertyRef(
      stubOp->getName().getOpPropertiesTypeID(), &properties));

  for (NamedAttribute namedAttr : cStubOp->getDiscardableAttrDictionary()) {
    StringAttr attrName = namedAttr.getName();

    // Do not hash the device ID, just the label.
    if (attrName == MO::kOperationDeviceRefAttrName) {
      hash = llvm::hash_combine(
          hash, cast<DeviceRefAttr>(namedAttr.getValue()).getLabel());
      continue;
    }

    hash = llvm::hash_combine(hash, namedAttr);
  }

  stubOp.getBodyBlock().walk([&](Operation *op) {
    hash = llvm::hash_combine(
        hash,
        mlir::OperationEquivalence::computeHash(
            op, /*hashOperands=*/mlir::OperationEquivalence::ignoreHashValue,
            /*hashResults=*/mlir::OperationEquivalence::ignoreHashValue,
            /*flags=*/mlir::OperationEquivalence::Flags::IgnoreLocations));
  });

  return hash;
}

// The edge type is `FuncCallOpInterface` so `CallGraphBase::build` finds both
// `mogg.call` and `mogg.call.async` edges, and a caller stub is
// canonicalized before it is hashed.
struct CallGraphNode : public KGEN::SCCNode<CallGraphNode, MOGG::FuncOp,
                                            MOGG::FuncCallOpInterface> {
  using Base =
      KGEN::SCCNode<CallGraphNode, MOGG::FuncOp, MOGG::FuncCallOpInterface>;
  using Base::Base;

  unsigned hash;
};

struct CallStubOrCalleeName {
  MOGG::FuncCallOpInterface callOp;
  StringAttr name;

  CallStubOrCalleeName(MOGG::FuncCallOpInterface call)
      : callOp(call), name(call.getCalleeSymbol().getAttr()) {}
  CallStubOrCalleeName(StringAttr n) : callOp(nullptr), name(n) {}
};

struct CallGraph : public KGEN::SCCGraph<CallGraph, CallGraphNode,
                                         /*BottomUpRewrite=*/true> {
  bool doAnalysis(CallGraphNode *node) { return false; }
  // Rewrites callsites in `node` that couldn't be inserted in the hash table.
  void doRewrite(CallGraphNode *node);

  DenseSet<KernelStubHashEntry, KernelStubOpEquivalenceInfo> uniqueFuncSet;

  // Renamings of top-level kernels (no kernel callers), which the CallGraph
  // doesn't replace; applied at the end of runOnOperation.
  llvm::DenseMap<StringAttr, StringAttr> topLevelReplacements;

  // Replacement -> replaced, used at the end to make the choice independent of
  // the non-deterministic execution order.
  llvm::DenseMap<StringAttr, SmallVector<CallStubOrCalleeName>>
      reverseReplacements;

  // The mutex to protect uniqueFuncSet accesses.
  std::mutex setAccessMutex;
};

void CallGraph::doRewrite(CallGraphNode *node) {
  for (auto [caller, callee] : node->callsites) {
    MOGG::FuncOp replacementCallee = nullptr;
    {
      std::lock_guard<std::mutex> lock(setAccessMutex);
      replacementCallee =
          uniqueFuncSet.insert(KernelStubHashEntry{callee->func, callee->hash})
              .first->op;
      reverseReplacements[replacementCallee.getSymNameAttr()].emplace_back(
          caller);
    }

    if (replacementCallee != callee->func) {
      // Merge locations so error messages and NVTX trace details keep every
      // merged stub's Python call site.
      replacementCallee->setLoc(mlir::FusedLoc::get(
          replacementCallee->getContext(),
          {replacementCallee->getLoc(), callee->func->getLoc()}));
      caller.setCalleeSymbol(
          FlatSymbolRefAttr::get(replacementCallee.getSymNameAttr()));
    }
  }

  node->hash = getHashValue(node->func);
  // A stub with no callers inserts itself; a collision with another top-level
  // stub is recorded to replace external calls afterwards (see
  // mergeIdenticalKernelStubs below).
  if (node->callers.empty()) {
    MOGG::FuncOp replacementCallee = nullptr;
    std::lock_guard<std::mutex> lock(setAccessMutex);
    replacementCallee =
        uniqueFuncSet.insert(KernelStubHashEntry{node->func, node->hash})
            .first->op;
    if (replacementCallee != node->func) {
      replacementCallee->setLoc(mlir::FusedLoc::get(
          node->func->getContext(),
          {replacementCallee->getLoc(), node->func->getLoc()}));
    }
    reverseReplacements[replacementCallee.getSymNameAttr()].emplace_back(
        node->func.getSymNameAttr());
    topLevelReplacements[node->func.getSymNameAttr()] =
        replacementCallee.getSymNameAttr();
  }
}

void mergeIdenticalKernelStubs(ModuleOp module) {
  SymbolTable symTable(module);
  CallGraph cg;
  cg.build(module, symTable);
  cg.run(*loadContext(module.getContext())->get<AsyncRT::CPUDevice>());

  // Dedup picked the first candidate, which depends on execution order. Sort
  // the renamed stubs by name and pick the first for determinism.
  llvm::DenseMap<StringAttr, StringAttr> finalReplacements;
  for (auto &[replacement, replacedCallOps] : cg.reverseReplacements) {
    if (replacedCallOps.size() == 1)
      continue;

    auto finalCandidate =
        *llvm::min_element(replacedCallOps, [](CallStubOrCalleeName lhs,
                                               CallStubOrCalleeName rhs) {
          return lhs.name.compare(rhs.name) < 0;
        });
    finalReplacements[replacement] = finalCandidate.name;

    // Fuse locations to maintain provenance information.
    if (FuncOp accumulatedOp = symTable.lookup<FuncOp>(replacement))
      if (FuncOp finalOp = symTable.lookup<FuncOp>(finalCandidate.name))
        if (finalOp != accumulatedOp)
          finalOp->setLoc(mlir::FusedLoc::get(
              finalOp->getContext(),
              {finalOp->getLoc(), accumulatedOp->getLoc()}));

    // Entries with a null callOp are top-level stubs from the
    // `node->callers.empty()` path; the walk below handles those.
    for (auto &callOp : replacedCallOps)
      if (callOp.callOp)
        callOp.callOp.setCalleeSymbol(
            FlatSymbolRefAttr::get(finalCandidate.name));
  }

  // Rewrite callers that aren't call-graph nodes to the deduplicated callee:
  //   - CompiledCallOp in mgp.ModelOps, and
  //   - FuncCallOpInterface ops inside pre-outline
  //     mogg.routine.async ops.
  // A graph-edge callee has a caller, so is never in `topLevelReplacements`.
  module.walk([&](Operation *op) {
    StringAttr calleeName;
    if (auto execOp = dyn_cast<MOGG::CompiledCallOp>(op))
      calleeName = execOp.getCalleeSymbolAttr().getAttr();
    else if (auto stubCall = dyn_cast<MOGG::FuncCallOpInterface>(op))
      calleeName = stubCall.getCalleeSymbol().getAttr();
    else
      return;

    auto it = cg.topLevelReplacements.find(calleeName);
    if (it == cg.topLevelReplacements.end())
      return;

    auto replacement = finalReplacements.find(it->second);
    if (replacement == finalReplacements.end())
      return;

    auto newRef = mlir::FlatSymbolRefAttr::get(replacement->second);
    if (auto execOp = dyn_cast<MOGG::CompiledCallOp>(op))
      execOp.setCalleeSymbolAttr(newRef);
    else
      cast<MOGG::FuncCallOpInterface>(op).setCalleeSymbol(newRef);
  });
}

struct MergeIdenticalFuncs
    : public MOGG::impl::MergeIdenticalFuncsBase<MergeIdenticalFuncs> {
  using MergeIdenticalFuncsBase::MergeIdenticalFuncsBase;

  void runOnOperation() override { mergeIdenticalKernelStubs(getOperation()); }
};

} // namespace

} // namespace M::MOGG
