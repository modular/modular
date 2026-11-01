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

#ifndef GENERICML_GRAPHCOMPILER_INTERFACES_MAPTOMOGGSTATE_H
#define GENERICML_GRAPHCOMPILER_INTERFACES_MAPTOMOGGSTATE_H

#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace M::MO {
class InvokeShapeFuncOp;
} // namespace M::MO

namespace M::MAP {

// Per-kernel context for MAPToMOGG's recursive lowering of one
// `map.iter.opaque` (see `LowerableToMOGGInterface`); rebuilt per kernel.
struct MAPToMOGGState {
  MAPToMOGGState(MOGG::MojoRegistry &registry, mlir::Operation *op)
      : registry(registry), builder(op) {}

  MOGG::MojoRegistry &registry;
  mlir::OpBuilder builder;

  // Whether an epilogue feeding a `_FusedComputeOutputTensor` may become a
  // compute lambda; mirrors the `EmitMojo` option.
  bool enableComputeLambdas = true;

  // Maps the kernel's inputs and output block args (and epilogue results) to
  // their reconstructed `!mogg.tensor`s. Nested regions remap their own
  // entries before lowering their body.
  mlir::IRMapping mapping;

  // Set by `IterOpaqueOp::lowerToMOGG` before any nested op is lowered.
  DeviceRefAttr device;
  // Unique dynamic dims, in first-seen order, to their `!mogg.dim` block
  // args (null until the block args exist).
  llvm::MapVector<mlir::Attribute, mlir::BlockArgument> dimArgs;
  // The trailing `!mogg.context` block args: one, or one per device (in
  // `getUniqueDevicesFromKernel` order) for a distributed kernel taking a
  // `DeviceContextArray`, which `ConformCallOperands` packs into a tuple.
  llvm::SmallVector<mlir::Value> contextArgs;

  // Each dynamic index op in a functor body (a `map.index.reshape` or a
  // `map.index.slice.select`), to the `!mogg.shape` captures it evaluates
  // against at runtime.
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Value>>
      indexShapeArgs;

  mlir::LogicalResult lowerOp(mlir::Operation *op);
  // Lowers `block`'s ops in order, skipping those `shouldLower` rejects; a
  // predicate may rely on earlier ops already being mapped.
  mlir::LogicalResult lowerRegionBody(
      mlir::Block &block,
      llvm::function_ref<bool(mlir::Operation *)> shouldLower =
          [](mlir::Operation *) { return true; });
};

/// Lowers a `mo.invoke_shape_func` into a `mogg.routine`, and erases it.
mlir::LogicalResult lowerInvokeShapeFuncToMOGG(MO::InvokeShapeFuncOp op,
                                               MAPToMOGGState &state);

} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_INTERFACES_MAPTOMOGGSTATE_H
