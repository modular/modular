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
#ifndef GENERICML_GRAPH_COMPILER_MOGGSUPPORT_FUSIONUTILS_H
#define GENERICML_GRAPH_COMPILER_MOGGSUPPORT_FUSIONUTILS_H

#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "Support/AssertStream.h"

using namespace M;

namespace M::MOGG {

//===----------------------------------------------------------------------===//
// FusionInfo helper functions
//===----------------------------------------------------------------------===//

FusionLeafAttr createFusionLeaf(Operation *op);

template <typename GroupAttr>
FusionInfoAttr createFusionInfo(MLIRContext *ctx, ArrayRef<Operation *> ops,
                                ArrayRef<uint32_t> operandIndices = {}) {

  auto getFusionInfo = [](Operation *op) -> Attribute {
    if (auto info = op->getAttrOfType<FusionInfoAttr>("fusionInfo"))
      return info.getRoot();
    return createFusionLeaf(op);
  };

  SmallVector<Attribute> children;
  SmallVector<uint32_t> mergedIndices(operandIndices.begin(),
                                      operandIndices.end());
  for (auto *op : ops) {
    Attribute child = getFusionInfo(op);
    // Flatten nested same-type fusions, e.g.
    // Elementwise(a, Elementwise(b, c)) -> Elementwise(a, b, c).
    if (auto sameGroup = dyn_cast<GroupAttr>(child)) {
      children.append(sameGroup.getChildren().begin(),
                      sameGroup.getChildren().end());
      if constexpr (std::is_same_v<GroupAttr, FusionEpilogueAttr> ||
                    std::is_same_v<GroupAttr, FusionPrologueAttr>) {
        auto inner = sameGroup.getOperandIndices();
        mergedIndices.append(inner.begin(), inner.end());
      }
      continue;
    }
    children.push_back(child);
  }

  if constexpr (std::is_same_v<GroupAttr, FusionEpilogueAttr> ||
                std::is_same_v<GroupAttr, FusionPrologueAttr>)
    return FusionInfoAttr::get(ctx,
                               GroupAttr::get(ctx, children, mergedIndices));
  else
    return FusionInfoAttr::get(ctx, GroupAttr::get(ctx, children));
}

/// Fuses topologically-sorted GenericCompiledRegionOps into one region and
/// replaces uses of their results. The caller must erase the old regions.
AsyncRoutineOp fuseAndReplaceCompiledRegions(bool allowDuplication,
                                             ArrayRef<Operation *> toFuse);

/// Intercepts inlining of an operation; returns true if handled. Arguments are
/// the builder, the IRMapping to look up/update, the op, its source region,
/// and the destination region.
using fuseRegionInlineCallBackT =
    llvm::function_ref<bool(OpBuilder &, mlir::IRMapping &, Operation *,
                            MOGG::AsyncRoutineOp, MOGG::AsyncRoutineOp)>;

// Aliased constants (MAXPLAT-70) are not lifted to init, so optimizations that
// rely on that lifting skip them.
bool dependsOnNoAliasConstantLikesOnly(mlir::Operation *operation);

// Kind of fused-tensor lambda built for a kernel output, and the kernel
// operand type it pairs with.
enum class KernelOutputFusionLambdaType {
  Output,                 // Store lambda (FusedOutputTensor).
  OutputLegacyForCompute, // Store lambda for mixed-precision
                          // (_FusedComputeOutputTensor).
  Compute,                // Return lambda (_FusedComputeOutputTensor).
  ComputeTile, // TileTensor return lambda (_FusedComputeOutputTileTensor).
  OutputTile,  // Tile store lambda; the fusion owns the store
               // (_FusedOutputTileTensor).
  // TODO(GEX-3912): SIMD reuses `Compute` for standalone and epilogue fusion,
  // but tile needs `ElementwiseTile` because standalone writes into `dst` while
  // the epilogue transforms `val`. An inout-`dst` tile compute interface would
  // collapse this into `ComputeTile`.
  ElementwiseTile, // Pure elementwise tile fusion writing into `dst`.
};

// Returns true if `op`'s device (from the enclosing kernel stub, else `op`) has
// `tile_based_fusion = true` in `moduleOp`'s `mo.device_info_mapping`.
bool isTileCodegenForOp(mlir::Operation *op, mlir::ModuleOp moduleOp);

} // namespace M::MOGG

#endif // GENERICML_GRAPH_COMPILER_MOGGSUPPORT_FUSIONUTILS_H
