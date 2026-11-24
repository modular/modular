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
// Merges the neighbouring dims of each fused kernel (an elementwise loop or a
// reduction) where every tensor the kernel touches lays them out one after the
// other, so the kernel does less index math and runs longer vectorized rows.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Support/AssertStream.h"
#include "mlir/IR/Builders.h"

using namespace M;
using namespace mlir;

namespace M::MAP {

#define GEN_PASS_DEF_FLATTENSHAPES
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h.inc"

namespace {

// Group sizes from the pairs that may merge.
SmallVector<int64_t> groupsOf(ArrayRef<bool> merges) {
  SmallVector<int64_t> groups = {1};
  for (bool merge : merges) {
    if (merge)
      ++groups.back();
    else
      groups.push_back(1);
  }
  return groups;
}

// Whether `body` lowers to `foreach_fusion_tile` (the device's
// `tile-based-fusion` option). That primitive takes rank-2 loops only, so such
// a loop keeps its rank. A body whose device can't be found keeps it too.
bool lowersToTileForeach(IterElementwiseBodyOp body) {
  auto owner = body->getParentOfType<MO::MOHasDeviceInterface>();
  if (!owner)
    return true;
  DeviceRefAttr device = MO::getDeviceOrDefault(owner);
  return device.getLabel() != M::kCPULabel &&
         MO::DeviceInfoMapping(body->getParentOfType<ModuleOp>())
             .get(device)
             .getTileBasedFusion();
}

// Narrows `merges` to the pairs every op in `body` can take, asking each op
// that implements the interface. False when an op blocks flattening outright:
// a kernel call that reads its index, or any other op that computes one or
// hands one to a region, since it would see a different index at the merged
// rank.
bool bodyAllowsMerges(Region &body, MutableArrayRef<bool> merges) {
  WalkResult blocked = body.walk([&](Operation *op) {
    if (auto call = dyn_cast<CallElementwiseOp>(op))
      return call.readsIndex() ? WalkResult::interrupt()
                               : WalkResult::advance();
    if (auto flattenable = dyn_cast<FlattenableOpInterface>(op)) {
      flattenable.canFlattenDims(merges);
      return WalkResult::advance();
    }
    return computesIndex(op) ? WalkResult::interrupt() : WalkResult::advance();
  });
  return !blocked.wasInterrupted();
}

// Narrows `merges` to the pairs every capture of `kernel` lays out one row
// after the other. False when a capture can't take the merged shape at all:
// something other than a load or a store uses it, one of them reads at an
// index computed inside the kernel, its rank isn't `rank`, or its layout is
// unknown. Flattening retypes the kernel's index arguments, so an index an op
// computes would keep its old rank. A capture nothing reads is left alone.
template <typename KernelOp>
bool capturesAllowMerges(KernelOp kernel, size_t rank,
                         MutableArrayRef<bool> merges) {
  auto atKernelIndex = [&](Operation *user) {
    Value index;
    if (auto load = dyn_cast<LoadOp>(user))
      index = load.getIndex();
    else if (auto store = dyn_cast<StoreOp>(user))
      index = store.getIndex();
    else
      return false;
    return isa<BlockArgument>(index);
  };
  Block &block = kernel.getBody().front();
  for (auto [capture, arg] : llvm::zip_equal(
           kernel.getCaptures(),
           block.getArguments().take_back(kernel.getCaptures().size()))) {
    if (arg.use_empty())
      continue;
    if (!llvm::all_of(arg.getUsers(), atKernelIndex))
      return false;
    if (MO::castFromTensorLike(capture.getType()).getShape().getRank() != rank)
      return false;
    std::optional<Layout> layout = layoutOf(capture);
    if (!layout)
      return false;
    for (size_t d = 0; d + 1 < rank; ++d)
      merges[d] = merges[d] && layout->merges(d);
  }
  return true;
}

// Clears the pairs whose merge would create a new parametric dim: the merged
// extent is the product of both, parametric unless both are static. Only a
// kernel directly in the graph's body can declare the parameter naming it.
void dropUndeclarableMerges(Operation *kernel, ArrayRef<TypedAttr> dims,
                            MutableArrayRef<bool> merges) {
  auto opaque = kernel->getParentOfType<IterOpaqueOp>();
  // In the graph's own body, so `mergeShape` can declare the parameter.
  if (isa<MO::GraphOp>(opaque->getParentOp()))
    return;
  for (size_t d = 0; d + 1 < dims.size(); ++d)
    merges[d] = merges[d] && MOSH::ShapeAttr::getStaticDim(dims[d]) &&
                MOSH::ShapeAttr::getStaticDim(dims[d + 1]);
}

// Creates a `map.index.collapse.stride` flattening each captured operand the
// body reads, and retypes the matching block argument. The kernel indexes its
// captures with its own index, so their dims merge as the kernel's do.
template <typename KernelOp>
void flattenCaptures(KernelOp kernel, FlattenDriver &driver) {
  MLIRContext *ctx = kernel.getContext();
  OpBuilder builder(kernel);
  DenseI64ArrayAttr groupsAttr = builder.getDenseI64ArrayAttr(driver.groups);
  Block &block = kernel.getBody().front();
  for (auto [operand, arg] : llvm::zip_equal(
           kernel.getCapturesMutable(),
           block.getArguments().take_back(kernel.getCaptures().size()))) {
    if (arg.use_empty())
      continue;
    Value capture = operand.get();
    MO::TensorType inTy = MO::castFromTensorLike(capture.getType());
    MOSH::ShapeAttr shape = mergeShape(inTy.getShape(), driver);
    auto outTy = MO::TensorType::get(shape, inTy.getElementType());
    auto view = CollapseStrideOp::create(
        builder, kernel.getLoc(), outTy,
        ViewSignatureAttr::get(ctx, inTy.getShape(), shape), groupsAttr,
        capture);
    operand.set(view);
    arg.setType(outTy);
  }
}

// The group sizes `body`'s loop dims merge into, or nothing when the loop
// stays as it is: its body computes an index or calls a kernel reading its
// index, a capture's layout is unknown, no pair of dims may merge, or it lowers
// to a tile foreach.
std::optional<SmallVector<int64_t>>
planElementwise(IterElementwiseBodyOp body) {
  // 1) Skip rank 0-1 and tile elementwise kernels.
  size_t rank = body.getShape().getRank();
  if (rank < 2 || lowersToTileForeach(body))
    return std::nullopt;

  // 2) Figure out which dims can be flattened.
  SmallVector<bool> merges(rank - 1, true);
  if (!bodyAllowsMerges(body.getBody(), merges) ||
      !capturesAllowMerges(body, rank, merges))
    return std::nullopt;

  // 3) Drop unsupported new parametric dims.
  dropUndeclarableMerges(body, body.getShape().getValues(), merges);

  // 4) Return the final groups of dims that can be flattened.
  SmallVector<int64_t> groups = groupsOf(merges);
  if (groups.size() == rank)
    return std::nullopt;
  return groups;
}

// Rewrites the driver's loop over its groups: each capture is read through a
// collapse view, and the loop's index and every select's drop to the merged
// rank.
//
// A [4, 6] loop whose captures are contiguous merges to one dim. The collapse
// views are what let the body read the same memory at rank 1:
//
//   map.iter.opaque (%a, %o) {        map.iter.opaque (%a, %o) {
//                                       %ca = collapse.stride<[4,6]->[24]>(%a)
//                                       %co = collapse.stride<[4,6]->[24]>(%o)
//     elementwise.body<[4, 6]>   =>     elementwise.body<[24]>
//       index (%i: index<2>)              index (%i: index<1>)
//       capturing {%x=%a, %y=%o} {        capturing {%x=%ca, %y=%co} {
//         %v = map.load %x[%i]              %v = map.load %x[%i]
//         map.store %y[%i] = f(%v)          map.store %y[%i] = f(%v)
//       }                                 }
//   }                                 }
//
// A capture read through a broadcast replicates the two outer dims, so those
// merge with each other but not with the innermost, and groups is [2, 1]:
//
//   elementwise.body<[4, 8, 16]>   =>   elementwise.body<[32, 16]>
//     index (%i: index<3>)                index (%i: index<2>)
//
// An op in the body takes the merged rank itself. A concat select on the outer
// axis merges the dims after the axis into each of its inputs:
//
//   concat.select<[3, 16], [5, 16] -> [8, 16], axis = 0>
//   =>
//   concat.select<[48], [80] -> [128], axis = 0>
void flattenElementwise(FlattenDriver &driver) {
  auto body = cast<IterElementwiseBodyOp>(driver.body);
  MLIRContext *ctx = body.getContext();
  MOSH::ShapeAttr loopShape = mergeShape(body.getShape(), driver);

  // 1) Flatten the operands the kernel captures.
  flattenCaptures(body, driver);

  // 2) Rewrite the kernel's shape and its index type.
  Block &block = body.getBody().front();
  block.getArgument(0).setType(IndexType::get(ctx, driver.groups.size()));
  body.setShapeAttr(loopShape);

  // 3) Flatten the other ops in the body.
  body.getBody().walk([&](FlattenableOpInterface op) { op.flatten(driver); });
}

// The group sizes the dims of `body`'s shape merge into, or nothing. As for a
// loop, with two differences: the reduced axis merges with neither neighbor,
// since a row is folded along it, and the innermost extent's rule doesn't
// apply, since the Row API picks the kernel's width.
std::optional<SmallVector<int64_t>> planRowwise(IterRowwiseBodyOp body) {
  // 1) Skip rank 0-1 kernels.
  ArrayRef<TypedAttr> dims = body.getShape().getValues();
  size_t rank = dims.size();
  if (rank < 2)
    return std::nullopt;

  // 2) Figure out which dims can be flattened.
  SmallVector<bool> merges(rank - 1, true);
  if (!bodyAllowsMerges(body.getBody(), merges) ||
      !capturesAllowMerges(body, rank, merges))
    return std::nullopt;

  // 3) Don't flatten the reduced dim.
  int64_t axis = body.getAxis();
  if (axis > 0)
    merges[axis - 1] = false;
  if (axis + 1 < static_cast<int64_t>(rank))
    merges[axis] = false;

  // 4) Drop unsupported new parametric dims.
  dropUndeclarableMerges(body, dims, merges);

  // 5) Return the final groups of dims that can be flattened.
  SmallVector<int64_t> groups = groupsOf(merges);
  if (groups.size() == rank)
    return std::nullopt;
  return groups;
}

// Rewrites the driver's rowwise body over its groups: each capture is read and
// written through a collapse view, the row and every fold's index drop to the
// merged rank, and the axis moves to its group.
//
// A [4, 6, 8] reduction over the innermost dim merges the two row dims; the
// reduced dim stays on its own, and the output collapses with them:
//
//  map.iter.opaque (%x, %o) {      map.iter.opaque (%x, %o) {
//                                  %cx = collapse.stride<[4,6,8]->[24,8]>(%x)
//                                  %co = collapse.stride<[4,6,1]->[24,1]>(%o)
//    rowwise.body<[4, 6, 8],   =>  rowwise.body<[24, 8], axis = 1>
//                 axis = 2>
//      index (%r: index<3>)          index (%r: index<2>)
//      map.iter.reduce ReduceSum     map.iter.reduce ReduceSum
//        index (%i: index<3>)          index (%i: index<2>)
//      map.store %y[%r] = %acc       map.store %y[%r] = %acc
//  }                               }
//
// Dims on both sides of the axis merge, each side on its own, and the axis
// moves to its group:
//
//   rowwise.body<[2, 3, 5, 4, 6], axis = 2>   (groups [2, 1, 2])
//   =>
//   rowwise.body<[6, 5, 24], axis = 1>
//
// Every phase of the row follows the folds to the merged rank:
//
//   map.iter.tile   index (%i: index<3>)   =>   index (%i: index<2>)
//   map.iter.reduce index (%i: index<3>)   =>   index (%i: index<2>)
//   map.iter.map    index (%j: index<3>)   =>   index (%j: index<2>)
void flattenRowwise(FlattenDriver &driver) {
  auto body = cast<IterRowwiseBodyOp>(driver.body);
  MLIRContext *ctx = body.getContext();
  MOSH::ShapeAttr shape = mergeShape(body.getShape(), driver);

  // 1) Flatten the operands the kernel captures.
  flattenCaptures(body, driver);

  // 2) Rewrite the kernel's shape and axis, and its index type.
  Block &block = body.getBody().front();
  block.getArgument(0).setType(IndexType::get(ctx, driver.groups.size()));
  body.setShapeAttr(shape);
  int64_t axis = 0;
  for (int64_t covered = 0; covered < static_cast<int64_t>(body.getAxis());
       covered += driver.groups[axis++]) {
  }
  body.setAxis(axis);

  // 3) Flatten the other ops in the body.
  body.getBody().walk([&](FlattenableOpInterface op) { op.flatten(driver); });
}

struct FlattenShapesPass : public impl::FlattenShapesBase<FlattenShapesPass> {
  using Base::Base;

  void runOnOperation() override {
    FlattenDriver driver;
    // A loop body holds no other, and the rewrite below changes the one just
    // visited, so never walk into it.
    getOperation().walk([&](IterElementwiseBodyOp body) {
      std::optional<SmallVector<int64_t>> groups = planElementwise(body);
      if (!groups)
        return WalkResult::skip();
      driver.groups = std::move(*groups);
      driver.body = body;
      flattenElementwise(driver);
      ++numLoopsFlattened;
      return WalkResult::skip();
    });

    getOperation().walk([&](IterRowwiseBodyOp body) {
      std::optional<SmallVector<int64_t>> groups = planRowwise(body);
      if (!groups)
        return WalkResult::skip();
      driver.groups = std::move(*groups);
      driver.body = body;
      flattenRowwise(driver);
      ++numLoopsFlattened;
      return WalkResult::skip();
    });
  }
};

} // namespace

} // namespace M::MAP
