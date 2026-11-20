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

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_SEQUENCEFUSIONANALYSIS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_SEQUENCEFUSIONANALYSIS_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"

#include "GraphCompiler/MODialect/Analysis/ParametricTopologyAnalysis.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOSHDialect/MOSHOps.h"

#include <functional>

namespace M::MOGG {

using BooleanFunctionOnOperation =
    const std::function<bool(mlir::Operation *)> &;

/// Eligibility is defined as ops that should be considered fusible.
using EligibilityFunction = BooleanFunctionOnOperation;

/// Whether two eligible ops can fuse together (e.g. same device, fusion
/// interfaces).
using CompatibilityFunction =
    const std::function<bool(mlir::Operation *, mlir::Operation *)> &;

/// Whether an op is heavy enough that its scheduling matters when preserving
/// parallelism.
using ComputationallySignificantFunction = BooleanFunctionOnOperation;

/// If an operation can be duplicated during sequence fusion.
using CanDuplicate = BooleanFunctionOnOperation;

using FusionSubGraphs = llvm::SmallVector<llvm::SmallVector<mlir::Operation *>>;

/// All functions here return deterministically ordered results, because passes
/// using this analysis must emit deterministic IR.

/// Returns subgraphs of operations that can be fused together while maintaining
/// the following guarantee: Any eligible ops that has a path in G or reverse(G)
/// to another eligible op that does not go through a non eligible op will be
/// fused. Will not fuse across blocks.
FusionSubGraphs floodFillSequenceFusionAnalysis(
    Region &region, MO::ParametricTopologyAnalysis &completeDataflowGraph,
    EligibilityFunction isEligible, CompatibilityFunction areCompatible);

/// If preserveParallelism is set to true, no additional dependency between
/// computationally significant ops is created. Otherwise, it fuses anything
/// as long as no cycle is created. Will not fuse across blocks.
FusionSubGraphs transitiveDependencySequenceFusionAnalysis(
    Region &region, MO::ParametricTopologyAnalysis &completeDataflowGraph,
    EligibilityFunction isEligible, CompatibilityFunction areCompatible,
    ComputationallySignificantFunction computationallySignificant,
    bool preserveParallelism);

/// Fuses an operation down if it can fuse with all its users. Unless
/// `canDuplicate` holds, operations with more than one user can't be fused
/// down.
FusionSubGraphs sourceDestinationSequenceFusionAnalysis(
    Region &region, MO::ParametricTopologyAnalysis &completeDataflowGraph,
    EligibilityFunction isEligible, CompatibilityFunction areCompatible,
    CanDuplicate canDuplicate,
    uint64_t maximumFusionSetSize = std::numeric_limits<uint64_t>::max());

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_SEQUENCEFUSIONANALYSIS_H
