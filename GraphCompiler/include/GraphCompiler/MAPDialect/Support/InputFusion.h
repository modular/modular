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
// Generic input fusion: inline a producer `map.iter.elementwise` body at the
// load site(s) where a consumer reads its result, replacing the producer's
// store. The load site is either in a consumer foreach body or behind an
// existing `map.iter.prologue` capture of a `map.iter.opaque` consumer. The
// generic enforces every correctness constraint; a wrapper narrows which edges
// fuse with a `shouldFuse` predicate over the producer store and consumer load.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_INPUTFUSION_H
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_INPUTFUSION_H

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MOOps.h"

#include <cstdint>
#include <functional>

namespace M::MAP {

/// Whether `op`'s body contains any `IndexTransformOpInterface` op. A pure
/// elementwise kernel has none.
bool isViewForeach(IterElementwiseOp op);

/// Whether `store`'s value flows from a transformed load, i.e. its kernel is a
/// view producer. Follows the store's dataflow, not mere presence of a
/// transform.
bool producesThroughView(StoreOp store);

/// Whether `op` is a view and nothing else: it has an index transform but no
/// elementwise compute. Once a view fuses into an elementwise kernel the result
/// still reads through a transform, but it is no longer free to re-derive.
bool isPureViewForeach(IterElementwiseOp op);

/// Whether duplicating `op` into each of its consumers is not a net
/// memory-traffic loss: at most one of its inputs is a large tensor, since
/// every extra copy re-reads all of them (JSC-30 / MSTDL-2759).
bool isCheapToDuplicate(IterElementwiseOp op);

/// Whether `load`'s index runs through a `map.index.reshape`. A reshape changes
/// rank, so `LoopInvariantViewMotion` can't hoist a view fused above it;
/// nothing may fuse into such a load.
bool loadIndexCrossesReshape(LoadOp load);

/// Whether a load of `tensor`, a kernel body's block argument, already crosses
/// a reshape (`loadIndexCrossesReshape`), so no view may fuse into it.
bool anyLoadCrossesReshape(mlir::Value tensor);

/// Clones `src`'s body into `dst` at each load of `src`'s result, at the
/// load's own index, and replaces that input with `src`'s inputs. Loads of
/// one index in one block share a clone. Leaves `src` and duplicate inputs in
/// place. `dst` is a structured kernel (elementwise or rowwise) whose every use
/// of that input is a load.
void spliceProducer(IterOpInterface dst, IterElementwiseOp src,
                    mlir::RewriterBase &rewriter);

/// The policy a wrapper supplies to narrow generic input fusion.
struct InputFusionConfig {
  /// Whether the edge from `producerStore` (a producer kernel's store) to
  /// `consumerLoad` (where the consumer reads that result) may fuse. The
  /// generic has already checked every correctness constraint.
  std::function<bool(StoreOp producerStore, LoadOp consumerLoad)> shouldFuse;
  /// Whether a producer with more than one use may be duplicated (the splice
  /// clones it into each consumer). A single-use producer always fuses.
  std::function<bool(IterElementwiseOp src)> canDuplicate;
  /// Cap on kernels fused into one consumer. Duplication clones a cheap
  /// multi-use producer into every consumer, so an elementwise chain with
  /// fan-out could otherwise grow a few enormous kernels. Unbounded by default
  /// for callers that never duplicate.
  uint64_t maxFusionSetSize = UINT64_MAX;
};

/// Greedily fuses producers into their consumers in `graphOp`, processing
/// consumers bottom-up and draining each consumer's producer chain. The stamped
/// `FusionInfo` follows the fusion itself: a prologue landing is `Prologue`, a
/// foreach landing `View` for a view producer else `Elementwise`. Returns how
/// many producers it fused.
unsigned fuseInput(MO::GraphOp graphOp, const InputFusionConfig &config);

} // namespace M::MAP

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_SUPPORT_INPUTFUSION_H
