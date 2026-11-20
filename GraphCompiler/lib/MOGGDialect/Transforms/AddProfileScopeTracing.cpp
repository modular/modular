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

#include "GraphCompiler/MGPDialect/Dialect/MGPOpInterfaces.h"
#include "GraphCompiler/MOGGDialect/MOGGInterfaces.h" // IWYU: needed for type declarations used via MOGG::
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "GraphCompiler/MOGGDialect/Support/ProfileScopeLabels.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_ADDPROFILESCOPETRACING
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

namespace M::MOGG {

namespace {

struct AddProfileScopeTracing
    : public MOGG::impl::AddProfileScopeTracingBase<AddProfileScopeTracing> {
  using AddProfileScopeTracingBase::AddProfileScopeTracingBase;

  /// Synthesize profiler ranges from contiguous ops sharing the
  /// same profile scope.
  void addProfileScopeTracing(MOGG::FuncOp kernelOp, bool traceUnlabelled) {
    // Use the kernel op locations since MergeIdenticalKernel fuses
    // a single location into FuncOp.
    auto kernelScopeLayers = collectProfileScopeLayers(kernelOp->getLoc());
    if (kernelScopeLayers.size() > 1) {
      llvm::SetVector<ProfileScopeLayer> observedLayers;
      for (Operation &op : kernelOp.getBodyBlock().getOperations())
        for (ProfileScopeLayer &layer : collectProfileScopeLayers(op.getLoc()))
          observedLayers.insert(layer);
      if (observedLayers.size() < kernelScopeLayers.size())
        return;
    }

    auto ctx = kernelOp->getContext();
    auto &kernelBody = kernelOp.getBodyBlock();
    Location loc = kernelOp->getLoc();
    auto rangeTy = MOGG::ProfileScopeRangeType::get(ctx);
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(&kernelBody);

    SmallVector<Operation *> traceable;
    for (Operation &op : kernelBody.without_terminator()) {
      // Async lifetime/signaling bookkeeping belong to no user scope, so they
      // neither open a run nor break one.
      if (isa<MOGG::AsyncReadyOp>(op))
        continue;
      traceable.push_back(&op);
    }

    if (traceable.empty())
      return;

    SmallVector<ProfileScopeLayerSet> layerSets;
    for (Operation *op : traceable) {
      // Ops never individually traced (host-side bookkeeping such as buffer
      // lookups or tensor-view construction) never define or break a run.
      auto opIt = mlir::Block::iterator(op);
      if (!opDoesSignificantWork(op)) {
        if (!layerSets.empty())
          layerSets.back().last = opIt;
        continue;
      }
      auto layers = collectProfileScopeLayers(op->getLoc());
      if (layers.empty()) {
        // Unlabeled ops are glue: extend an open run, otherwise ignore.
        if (!layerSets.empty())
          layerSets.back().last = opIt;
        continue;
      }
      if (!layerSets.empty() && layerSets.back().layers == layers) {
        layerSets.back().last = opIt;
        continue;
      }
      layerSets.emplace_back(opIt, opIt, std::move(layers));
    }

    // Optional fallback range over a fully unlabeled body; off by default.
    if (layerSets.empty() && traceUnlabelled)
      layerSets.emplace_back(mlir::Block::iterator(traceable.front()),
                             mlir::Block::iterator(traceable.back()));

    // Ancestor scope ranges go before the leaf runs so that, at a shared
    // start/end anchor, insertion order alone makes the ancestor start first
    // and end last. See addAncestorProfileScopeRuns.
    size_t maxLayerLen = 0;
    for (const ProfileScopeLayerSet &layerSet : layerSets)
      if (layerSet.layers.size() == 1)
        maxLayerLen = std::max(maxLayerLen, layerSet.layers.front().size());

    llvm::SmallVector<ProfileScopeLayerSet> allLayerSets;
    for (size_t depth = 0; depth + 1 < maxLayerLen; ++depth)
      addAncestorProfileScopeRuns(layerSets, depth, allLayerSets);
    allLayerSets.append(layerSets.begin(), layerSets.end());

    std::string regionName = kernelOp.getSymName().str();
    for (ProfileScopeLayerSet &layerSet : allLayerSets) {
      if (!llvm::any_of(layerSet.ops(), [](Operation &op) {
            return opDoesSignificantWork(&op);
          }))
        continue;

      std::string name = formatProfileScope(layerSet.layers, true) + regionName;
      auto colors = collectRunColors(layerSet.ops());
      auto resolvedColor = resolveProfileScopeColor(colors);
      mlir::StringAttr colorAttr;
      if (resolvedColor)
        colorAttr = StringAttr::get(ctx, *resolvedColor);
      auto beginOp = MOGG::ProfileScopeBeginOp::create(
          builder, loc, rangeTy, StringAttr::get(ctx, name), colorAttr);
      beginOp->moveBefore(&*layerSet.first);
      auto endOp =
          MOGG::ProfileScopeEndOp::create(builder, loc, beginOp.getRange());
      endOp->moveAfter(&*layerSet.last);
    }
  }

  /// Collects every distinct profile scope color in range, to pick the color
  /// (if any) the range spanning these ops claims.
  static llvm::SmallVector<std::string>
  collectRunColors(llvm::iterator_range<mlir::Block::iterator> range) {
    llvm::SetVector<std::string> colors;
    for (Operation &op : range) {
      auto opColors = collectProfileScopeColors(op.getLoc());
      for (std::string &color : opColors)
        colors.insert(std::move(color));
    }
    return colors.takeVector();
  }

  /// Heuristic excluding host-side bookkeeping (buffer lookups, tensor-view
  /// construction, shape math) from traced ranges.
  static bool opDoesSignificantWork(Operation *op) {
    return isa<FuncCallOpInterface, M::MGP::MGPHasDeviceOpInterface>(op);
  }

  /// Group adjacent single-layer entries of `leafLayerSets` that share the
  /// same length-`(depth + 1)` layer prefix into coarser ancestor ranges,
  /// appending one `ProfileScopeLayerSet` per group of two or more leaf layer
  /// sets to `outLayerSets`.
  ///
  /// A peer set (`layers.size() > 1`, ops merged in from an unrelated scope by
  /// fusion) has no meaningful ancestor and never participates. A leaf layer
  /// set whose layer is *exactly* `depth + 1` long doesn't participate at that
  /// depth either: its depth-`d` prefix would just be a second copy of the
  /// leaf range that already covers the identical span.
  static void addAncestorProfileScopeRuns(
      llvm::ArrayRef<ProfileScopeLayerSet> leafLayerSets, size_t depth,
      llvm::SmallVector<ProfileScopeLayerSet> &outLayerSets) {
    const ProfileScopeLayerSet *groupFirst = nullptr;
    const ProfileScopeLayerSet *groupLast = nullptr;
    llvm::ArrayRef<std::string> groupPrefix;
    size_t groupSize = 0;

    auto flush = [&] {
      if (groupSize > 1)
        outLayerSets.push_back({groupFirst->first,
                                groupLast->last,
                                {MOGG::ProfileScopeLayer(groupPrefix.begin(),
                                                         groupPrefix.end())}});
      groupFirst = nullptr;
      groupSize = 0;
    };

    for (const ProfileScopeLayerSet &layerSet : leafLayerSets) {
      bool participates = layerSet.layers.size() == 1 &&
                          layerSet.layers.front().size() > depth + 1;
      if (participates) {
        llvm::ArrayRef<std::string> prefix =
            llvm::ArrayRef<std::string>(layerSet.layers.front())
                .take_front(depth + 1);
        if (groupFirst && prefix == groupPrefix) {
          groupLast = &layerSet;
          ++groupSize;
          continue;
        }
        flush();
        groupFirst = groupLast = &layerSet;
        groupPrefix = prefix;
        groupSize = 1;
        continue;
      }
      flush();
    }
    flush();
  }

  void runOnOperation() override {
    auto kernelOp = getOperation();

    // MOGG-ABI stubs are handled by other emission paths.
    if (kernelOp.getAbi() != MOGG::ABIKind::MGP)
      return;

    addProfileScopeTracing(kernelOp, profileScopeTracingTraceUnlabelled);
  }
};

} // namespace

} // namespace M::MOGG
