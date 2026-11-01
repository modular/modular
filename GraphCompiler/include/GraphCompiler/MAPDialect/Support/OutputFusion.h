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
//
// Generic output fusion: inline a `map.iter.elementwise` consumer's body into
// the `map.iter.epilogue` of the `map.iter.opaque` producer that feeds it, at
// the producer's store site. A consumer reading the result at an untransformed
// index fuses as-is; one reading it through a single `map.index.reshape` fuses
// with that reshape relocated onto the store index. The generic enforces every
// correctness constraint; a wrapper narrows which edges fuse with a
// `shouldFuse` predicate over the producer's epilogue store and the consumer
// load.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_OUTPUTFUSION_H
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_OUTPUTFUSION_H

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MOOps.h"

#include <functional>

namespace M::MAP {

/// Whether `load` reads its tensor at the consumer's own loop index, with no
/// index transform in the path -- the plain epilogue case.
bool consumerReadsUntransformed(LoadOp load);

/// Whether `load` reads its tensor through a single `map.index.reshape` of the
/// consumer's loop index (composing the reshape with another transform, or
/// reading directly, is false) -- the reshape-epilogue case.
bool consumerReadsThroughReshape(LoadOp load);

/// Whether `load`'s consumer is a pure reshape: it stores exactly the value it
/// loaded through the reshape, with no per-element compute in between. Fusing
/// it is valid but only relabels the store index, so a wrapper may decline it.
bool consumerIsPureReshape(LoadOp load);

/// Whether `consumer` is a `mo.mutable.store` that can fuse into `dst`. Its
/// lowering (`lowerMutableStoreToForeach`) returns only a chain and stores into
/// its buffer input. It fuses when `dst` doesn't read that buffer, its input
/// chain is fresh (`mo.chain.create`), and nothing uses its output chain.
bool isFusibleMutableStore(IterElementwiseOp consumer, mlir::Operation *dst);

/// The policy a wrapper supplies to narrow generic output fusion.
struct OutputFusionConfig {
  /// Whether the edge from `epilogueStore` (the producer's epilogue store) to
  /// `consumerLoad` (where the consumer reads the producer's result) may fuse.
  /// The generic has already checked every correctness constraint.
  std::function<bool(StoreOp epilogueStore, LoadOp consumerLoad)> shouldFuse;
};

/// Fuses output edges in `graphOp` accepted by `config`, processing producers
/// topologically and draining each producer's fusible consumer chain. Every
/// fused edge is stamped `FusionEpilogue`. Returns how many kernels it fused.
unsigned fuseOutput(MO::GraphOp graphOp, const OutputFusionConfig &config);

/// Grows `dst` with one destination-passing result per type in `extraTypes`,
/// each with a matching trailing output block argument. Results are fixed at
/// construction so this rebuilds the op; the body region moves across, leaving
/// the ops inside it (the epilogue) valid. New results go after the existing
/// destination-passing ones, which `IterOpaqueOp::verify` requires.
IterOpaqueOp growOutputs(IterOpaqueOp dst, mlir::TypeRange extraTypes,
                         mlir::RewriterBase &rewriter);

} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_OUTPUTFUSION_H
