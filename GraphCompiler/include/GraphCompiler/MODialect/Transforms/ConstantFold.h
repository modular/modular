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

#ifndef GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLD_H
#define GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLD_H

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Support/AssertStream.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallSet.h"

#include <numeric>
namespace M::MO {

struct Subgraph {
  llvm::SetVector<Operation *> ops;
  // Ops with uses outside the subgraph. Outputs may depend on other outputs.
  llvm::SetVector<Operation *> outputs;
  Subgraph(llvm::SetVector<Operation *> ops) : ops(std::move(ops)) {
    findOutputs();
  }

  /// Remove op and its dependents from the subgraph.
  void remove(Operation *op) {
    llvm::SmallVector<Operation *> opsToRemove = {op};
    while (!opsToRemove.empty()) {
      Operation *op = opsToRemove.pop_back_val();
      for (auto dep : op->getUsers())
        if (ops.contains(dep))
          opsToRemove.push_back(dep);
      ops.remove(op);
    }
    outputs.clear();
    findOutputs();
  }

  bool fullyContains(Value v) const {
    return llvm::all_of(v.getUsers(),
                        [this](Operation *user) { return ops.contains(user); });
  }

  /// Parameters declared and used in this subgraph, in topological order.
  llvm::SetVector<StringAttr> paramDeclarations() const;

  llvm::SmallVector<ConstantOp> constants() const {
    auto filtered = llvm::make_filter_range(
        ops, [](Operation *op) { return dyn_cast<ConstantOp>(op); });
    return llvm::to_vector(llvm::map_range(
        filtered, [](Operation *op) { return dyn_cast<ConstantOp>(op); }));
  }

private:
  void findOutputs() {
    for (Operation *op : ops) {
      if (dyn_cast<ConstantOp>(op))
        continue;
      for (auto user : op->getUsers()) {
        if (dyn_cast<MOControlOpInterface>(user) || !ops.contains(user)) {
          outputs.insert(op);
          break;
        }
      }
    }
  }
};

namespace {
struct ValueParamCompare {
  bool operator()(const Operation *lhs, const Operation *rhs) const {
    return hash_value(lhs) < hash_value(rhs);
  }
};
} // namespace

// Returns the connected components of the undirected subgraph.
static SmallVector<Subgraph> getSubgraphComponents(const Subgraph &subgraph) {
  if (subgraph.ops.empty())
    return {};

  llvm::EquivalenceClasses<Operation *> newSubgraphs;
  for (Operation *curOp : subgraph.ops) {
    newSubgraphs.insert(curOp);
    for (Operation *other : curOp->getUsers())
      if (subgraph.ops.contains(other))
        newSubgraphs.unionSets(other, curOp);
  }

  SmallVector<Subgraph> answer;
  for (auto it = newSubgraphs.begin(), end = newSubgraphs.end(); it != end;
       ++it) {
    if (!(*it)->isLeader())
      continue;

    answer.emplace_back(llvm::SetVector<Operation *>{
        newSubgraphs.member_begin(*(*it)), newSubgraphs.member_end()});
  }
  return answer;
}

struct ConstantFoldPolicy {
  // Returns the subgraphs, or subsets of them, to fold; anything not returned
  // is left unfolded.
  virtual SmallVector<Subgraph>
  shouldFold(SmallVector<Subgraph> subgraphs) const = 0;
  virtual ~ConstantFoldPolicy() = default;
};

// Fold the largest subgraph that does not increase memory used by more than
// a factor of the space used by the non-folded subgraph.
struct FactorMemIncrease : public ConstantFoldPolicy {
  FactorMemIncrease(double factor, int64_t threshold)
      : factor(factor), threshold(threshold) {}

  // 1 allows no increase; 2 allows up to 100% more memory.
  double factor;
  /// Fold everything under this size in bytes.
  int64_t threshold;

  SmallVector<Subgraph>
  shouldFold(SmallVector<Subgraph> subgraphs) const override {
    SmallVector<Subgraph> foldThese;
    while (!subgraphs.empty()) {
      Subgraph subgraph = subgraphs.pop_back_val();
      std::optional<int64_t> maybeInitialMem = constantMemUsage(subgraph);

      std::optional<int64_t> maybeOutputMem = outputMemUsage(subgraph);

      // Memory growth is unknown with parameterized inputs, so don't fold.
      if (!maybeInitialMem.has_value())
        continue;

      int64_t initialMem = *maybeInitialMem;

      int64_t factorLimit = floor(initialMem * factor);
      int64_t totalThreshold = std::max(factorLimit, threshold);

      auto memoryOk = [&]() {
        return maybeOutputMem.has_value() && *maybeOutputMem <= totalThreshold;
      };
      bool maybeBcastsOutputs = true;

      // TODO: this is N^2, we should improve it #18475
      while (!memoryOk() || maybeBcastsOutputs) {
        // A parameterized output shape has unknown size; don't fold.
        if (!maybeOutputMem.has_value())
          break;

        if (subgraph.outputs.empty())
          break;

        // Drop the largest outputs until under the memory threshold.
        if (!memoryOk()) {
          Operation *maxValue = *std::max_element(
              subgraph.outputs.begin(), subgraph.outputs.end(),
              [&subgraph, totalThreshold](Operation *a, Operation *b) {
                std::optional<int64_t> maybeSizeA = valueSize(subgraph, a);
                std::optional<int64_t> maybeSizeB = valueSize(subgraph, b);

                // Treat unknown sizes as over the threshold.
                int64_t sizeA = maybeSizeA.has_value() ? maybeSizeA.value()
                                                       : totalThreshold + 1;
                int64_t sizeB = maybeSizeB.has_value() ? maybeSizeB.value()
                                                       : totalThreshold + 1;

                return sizeA < sizeB;
              });

          // TODO: this is N^2, we should improve it #18475
          subgraph.remove(maxValue);

          // New outputs may be broadcasts.
          maybeBcastsOutputs = true;
        }

        // Folding broadcasts never saves memory (they can be view ops), and
        // "broadcast sinking" optimizations rely on them.
        if (maybeBcastsOutputs) {
          maybeBcastsOutputs = false;
          SmallVector<Operation *> initOutputs(subgraph.outputs.begin(),
                                               subgraph.outputs.end());
          for (Operation *op : initOutputs) {
            if (isa<BroadcastToOp, Static::BroadcastToOp>(op)) {
              maybeBcastsOutputs = true;
              subgraph.remove(op);
            }
          }
        }

        maybeOutputMem = outputMemUsage(subgraph);
      }

      // Removal may split the graph, and one large component could mask
      // another's memory growth, so only fold single-component graphs.
      SmallVector<Subgraph> result = getSubgraphComponents(subgraph);
      ASSERT_STREAM(!result.empty(), "Expect non-empty subgraph!");

      // Skip subgraphs whose outputs are all constants or parameters; shape
      // inference cleans up the latter.
      if (result.size() == 1 && memoryOk() &&
          !llvm::all_of(subgraph.outputs, [](Operation *op) {
            return dyn_cast<ConstantOp>(op) ||
                   dyn_cast<KGEN::ParamDeclareOp>(op);
          })) {
        foldThese.push_back(subgraph);
      } else if (result.size() > 1) {
        // Revisit each component separately.
        subgraphs.append(result.begin(), result.end());
      }
    }
    return foldThese;
  }
  ~FactorMemIncrease() override = default;

  // Size in bytes, or std::nullopt if unknown (e.g. parameterized shape).
  static std::optional<int64_t> valueSize(Value v) {
    auto tensor = dyn_cast<TensorType>(v.getType());
    if (tensor) {
      std::optional<int64_t> numElements = tensor.getNumElements();
      if (!numElements.has_value())
        return {};
      return numElements.value() * tensor.getDtype().getSizeInBytes();
    }

    // Index and shape values have nominal size.
    return 0;
  }

  static std::optional<int64_t> valueSize(const Subgraph &subgraph,
                                          Operation *op) {
    int64_t sum = 0;
    for (auto result : op->getResults()) {
      if (!subgraph.fullyContains(result)) {
        std::optional<int64_t> localSize = valueSize(result);
        if (!localSize.has_value())
          return {};
        sum += localSize.value();
      }
    }
    return sum;
  }

  static std::optional<int64_t> valueSize(ConstantOp op) {
    return valueSize(op.getResult());
  }

  // Bytes used by the subgraph's outputs, or std::nullopt if unknown.
  static std::optional<int64_t> outputMemUsage(const Subgraph &subgraph) {
    int64_t outputSize = 0;
    for (auto output : subgraph.outputs) {
      std::optional<int64_t> localSize = valueSize(subgraph, output);

      if (!localSize.has_value())
        return {};

      outputSize += localSize.value();
    }
    return outputSize;
  }

  // Bytes used by the subgraph's internal constants, or std::nullopt if
  // unknown.
  static std::optional<int64_t> constantMemUsage(const Subgraph &subgraph) {
    int64_t internalSize = 0;
    for (auto op : subgraph.ops) {
      if (!subgraph.outputs.contains(op)) {
        auto constant = dyn_cast<ConstantOp>(op);
        if (constant) {
          auto localSize = valueSize(constant);
          if (!localSize.has_value())
            return {};
          internalSize += localSize.value();
        }
      }
    }
    return internalSize;
  }
};

SmallVector<Subgraph>
constantSubgraphs(MO::GraphOp graph,
                  KGEN::ParameterCollector::Analysis &paramCache);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLD_H
