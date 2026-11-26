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
// This file implements `MAP_FlattenableOpInterface` for the MAP dialect's
// operations: which pairs of a loop's dims each op lets `map-flatten-shapes`
// merge, how the op rewrites itself for the merged groups, and the layout a
// view hands its reader. `layoutOf` and `mergeShape`, which the pass calls
// to drive all of that, are here too.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Support/AssertStream.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"

using namespace M;
using namespace mlir;
using namespace M::MAP;

// Every kernel `map-flatten-shapes` sees sits in a `map.iter.opaque`: the loop
// bodies by their ODS parent, and the opaque calls, prologues and epilogues by
// how fusion builds them. So the lookup of that parent below can't fail, and a
// caller that planned with `dropUndeclarableMerges` never trips the assert
// below.
MOSH::ShapeAttr M::MAP::mergeShape(MOSH::ShapeAttr shape,
                                   FlattenDriver &driver) {
  Operation *body = driver.body;
  MLIRContext *ctx = body->getContext();
  ArrayRef<TypedAttr> dims = shape.getValues();
  SmallVector<TypedAttr> merged;
  size_t first = 0;
  for (int64_t size : driver.groups) {
    ArrayRef<TypedAttr> group = dims.slice(first, size);
    first += size;
    if (size == 1) {
      merged.push_back(group.front());
      continue;
    }
    int64_t product = 1;
    bool allStatic = true;
    for (TypedAttr dim : group) {
      if (std::optional<int64_t> v = MOSH::ShapeAttr::getStaticDim(dim))
        product *= *v;
      else
        allStatic = false;
    }
    if (allStatic) {
      merged.push_back(MOSH::ShapeAttr::getDim(ctx, product));
      continue;
    }
    // Parameters are only declared in a graph's own body: not in
    // `mo.parallel` bodies, and not in control-flow regions, which MEF can't
    // take them from.
    auto opaque = body->getParentOfType<IterOpaqueOp>();
    ASSERT_STREAM(isa<MO::GraphOp>(opaque->getParentOp()),
                  << "a symbolic product needs a parameter, which only a "
                     "kernel in the graph's own body can declare");
    TypedAttr expr =
        MOSH::ShapeAttr::get(group, shape.getType()).getSymbolicNumElements();
    // A group whose other extents are 1 is already a named dim. Declaring an
    // alias of it would be wrong, not just redundant: MGP lowering maps a name
    // declared before its value is known to a placeholder, and an alias would
    // keep that placeholder after it is resolved and erased.
    if (isa<KGEN::ParamDeclRefAttr>(expr)) {
      merged.push_back(expr);
      continue;
    }
    TypedAttr &ref = driver.declared[expr];
    if (!ref) {
      auto graph = body->getParentOfType<MO::GraphOp>();
      std::string name = graph.getNewParamName("param_flat_");
      Type dimType = MOSH::ShapeAttr::getDimType(ctx);
      OpBuilder builder(opaque);
      KGEN::ParamDeclareOp::create(builder, body->getLoc(),
                                   KGEN::ParamDeclAttr::get(name, dimType),
                                   expr);
      ref = KGEN::ParamDeclRefAttr::get(StringAttr::get(ctx, name), dimType);
    }
    merged.push_back(ref);
  }
  return MOSH::ShapeAttr::get(merged, shape.getType());
}

std::optional<Layout> M::MAP::layoutOf(Value value) {
  auto tensorTy = dyn_cast<MO::TensorType>(value.getType());
  auto bufferTy = dyn_cast<MO::BufferType>(value.getType());
  if (!tensorTy && !bufferTy)
    return std::nullopt;
  // A layout attribute reorders the data, so its strides aren't row-major.
  if (tensorTy && tensorTy.getLayoutAttr())
    return std::nullopt;
  size_t rank = MO::castFromTensorLike(value.getType()).getShape().getRank();

  Operation *def = value.getDefiningOp();
  if (!def) {
    // A kernel input or output of the enclosing `map.iter.opaque`.
    auto arg = cast<BlockArgument>(value);
    if (isa<IterOpaqueOp>(arg.getOwner()->getParentOp()) &&
        (tensorTy || bufferTy))
      return Layout::rowMajor(rank);
    return std::nullopt;
  }
  auto flattenable = dyn_cast<FlattenableOpInterface>(def);
  if (!flattenable)
    return std::nullopt;
  std::optional<Layout> in =
      def->getNumOperands() ? layoutOf(def->getOperand(0)) : std::nullopt;
  return flattenable.getOutputLayout(in);
}

// Whether the slice keeps all of dim `d` from its start: start 0 and the same
// extent in and out. Equal unknown extents (`?`) can be different values, so
// with a run-time stop only a static extent shows the dim is whole.
static bool keepsWholeDim(SliceSpecAttr spec, size_t d) {
  TypedAttr in = spec.getSignature().getIn().getValues()[d];
  TypedAttr out = spec.getSignature().getOut().getValues()[d];
  return spec.getStarts()[d] == 0 && out == in &&
         (!ShapedType::isDynamic(spec.getStops()[d]) ||
          MOSH::ShapeAttr::getStaticDim(out));
}

static bool isCut(SliceSpecAttr spec, size_t d) {
  return spec.getSteps()[d] != 1 || !keepsWholeDim(spec, d);
}

static std::optional<Layout> reshapeLayout(ReshapeSpecAttr spec,
                                           std::optional<Layout> in) {
  // A reshape only keeps the elements' order on a row-major input.
  size_t inRank = spec.getSignature().getIn().getRank();
  if (!in || llvm::any_of(in->zero, [](bool z) { return z; }) ||
      !llvm::all_of(in->contiguous, [](bool c) { return c; }) ||
      in->zero.size() != inRank)
    return std::nullopt;
  return Layout::rowMajor(spec.getSignature().getOut().getRank());
}

static std::optional<Layout> sliceLayout(SliceSpecAttr spec,
                                         std::optional<Layout> in) {
  if (!in)
    return std::nullopt;
  ArrayRef<int64_t> steps = spec.getSteps().asArrayRef();
  size_t rank = spec.getSignature().getOut().getRank();
  Layout out{in->zero, SmallVector<bool>(rank - 1, false)};
  for (size_t d = 0; d + 1 < rank; ++d) {
    // The step scales dim d's stride, and a cut dim d+1 no longer spans its
    // input row: either breaks the pair. A start on dim d only moves the base
    // pointer.
    out.contiguous[d] = in->contiguous[d] && steps[d] == 1 &&
                        steps[d + 1] == 1 && keepsWholeDim(spec, d + 1);
  }
  return out;
}

static std::optional<Layout> broadcastLayout(BroadcastSpecAttr spec,
                                             std::optional<Layout> in) {
  if (!in)
    return std::nullopt;
  ArrayRef<TypedAttr> inDims = spec.getSignature().getIn().getValues();
  ArrayRef<TypedAttr> outDims = spec.getSignature().getOut().getValues();
  size_t rank = outDims.size();
  size_t delta = rank - inDims.size();
  // Per out dim: replicated (stride 0), kept (the input's stride), or decided
  // at run time (a symbolic input dim that could be 1).
  enum Kind { kZero, kKept, kUnknown };
  SmallVector<Kind> kinds;
  for (size_t i = 0; i < rank; ++i) {
    std::optional<int64_t> dim =
        i < delta ? std::optional<int64_t>(1)
                  : MOSH::ShapeAttr::getStaticDim(inDims[i - delta]);
    kinds.push_back(!dim ? kUnknown : *dim == 1 ? kZero : kKept);
  }
  Layout out{SmallVector<bool>(rank, false),
             SmallVector<bool>(rank - 1, false)};
  for (size_t i = 0; i < rank; ++i)
    out.zero[i] = kinds[i] == kZero;
  for (size_t i = 0; i + 1 < rank; ++i)
    out.contiguous[i] =
        kinds[i] == kKept && kinds[i + 1] == kKept && in->contiguous[i - delta];
  return out;
}

static std::optional<Layout> transposeLayout(TransposeSpecAttr spec,
                                             std::optional<Layout> in) {
  if (!in)
    return std::nullopt;
  ArrayRef<int64_t> perm = spec.getPerm().asArrayRef();
  if (llvm::any_of(perm, [](int64_t p) { return p < 0; }))
    return std::nullopt;
  size_t rank = perm.size();
  Layout out{SmallVector<bool>(rank, false),
             SmallVector<bool>(rank - 1, false)};
  for (size_t i = 0; i < rank; ++i)
    out.zero[i] = in->zero[perm[i]];
  for (size_t i = 0; i + 1 < rank; ++i)
    out.contiguous[i] = perm[i + 1] == perm[i] + 1 && in->contiguous[perm[i]];
  return out;
}

// The select picks a window of each dim. Merged with a neighbor, a cut dim
// (sliced or strided) would no longer be one window of the merged dim, so it
// merges with neither neighbor; the whole dims around it still merge. For
// example, slicing [2, 4, 8, 16] to [2, 2, 8, 16] (dim 1 cut) keeps 2 and 4
// apart and merges 8 and 16: [2, 4, 128] -> [2, 2, 128].
void SliceSelectOp::canFlattenDims(MutableArrayRef<bool> merges) {
  SliceSpecAttr spec = getSpec();
  // This is stricter than needed: a cut dim could start a group that isn't the
  // innermost one, e.g. dims 0 and 1 of `[A, B, C]` when something else
  // already keeps B and C apart. That merge is given up here to decide each
  // pair on its own.
  for (size_t d = 0; d < merges.size(); ++d)
    merges[d] = merges[d] && !isCut(spec, d) && !isCut(spec, d + 1);
}

// The select's spec and index name the loop's dims, so they take the merged
// shape with the loop, one start, stop and step per merged dim. A group of
// several dims holds no cut, so it runs whole: start 0, step 1, up to its
// merged extent. Slicing [2, 4, 8, 16] to [2, 2, 8, 16] on dim 1, with 8 and
// 16 merged:
//
//   slice.select<[2, 4, 8, 16] -> [2, 2, 8, 16],
//     starts = [0, 1, 0, 0], stops = [2, 3, 8, 16], steps = [1, 1, 1, 1]>
//     (%i) : !map.index<4>
//   =>
//   slice.select<[2, 4, 128] -> [2, 2, 128],
//     starts = [0, 1, 0], stops = [2, 3, 128], steps = [1, 1, 1]>
//     (%i) : !map.index<3>
void SliceSelectOp::flatten(FlattenDriver &driver) {
  SliceSpecAttr spec = getSpec();
  MOSH::ShapeAttr in = mergeShape(spec.getSignature().getIn(), driver);
  MOSH::ShapeAttr out = mergeShape(spec.getSignature().getOut(), driver);
  // `canFlattenDims` leaves only uncut dims in a group of several, so such a
  // group runs whole: start 0, step 1, up to its merged extent.
  SmallVector<int64_t> starts, stops, steps;
  size_t first = 0;
  for (auto [group, size] : llvm::enumerate(driver.groups)) {
    if (size == 1) {
      starts.push_back(spec.getStarts()[first]);
      stops.push_back(spec.getStops()[first]);
      steps.push_back(spec.getSteps()[first]);
    } else {
      starts.push_back(0);
      stops.push_back(out.getIntDim(group));
      steps.push_back(1);
    }
    first += size;
  }
  setSpecAttr(SliceSpecAttr::get(getContext(), in, out, starts, stops, steps));
  getBody().getArgument(0).setType(
      IndexType::get(getContext(), driver.groups.size()));
}

// The inputs interleave across the dims before the axis: one row of each
// input in turn. So the dim before the axis stays apart from the axis; the
// axis and the dims after it merge, since each input lays them out as one run.
// For example, [3, 16] ++ [5, 16] on axis 0 becomes [48] ++ [80] -> [128],
// while [4, 6] ++ [4, 10] on axis 1 keeps its two dims.
void ConcatSelectOp::canFlattenDims(MutableArrayRef<bool> merges) {
  if (int64_t axis = getSpec().getAxis(); axis > 0)
    merges[axis - 1] = false;
}

// The inputs' shapes, the result's and the axis name the loop's dims, so they
// take the merged shape with the loop, and the axis moves to its group. A
// concat on axis 2 of a [2, 3, 10, 5] loop, merging 0 with 1 and 2 with 3:
//
//   concat.select<[2, 3, 4, 5], [2, 3, 6, 5] -> [2, 3, 10, 5], axis = 2>
//     (%i) : !map.index<4>
//   =>
//   concat.select<[6, 20], [6, 30] -> [6, 50], axis = 1>
//     (%i) : !map.index<2>
void ConcatSelectOp::flatten(FlattenDriver &driver) {
  ConcatSpecAttr spec = getSpec();
  SmallVector<MOSH::ShapeAttr> ins;
  for (MOSH::ShapeAttr in : spec.getInputShapes())
    ins.push_back(mergeShape(in, driver));
  MOSH::ShapeAttr out = mergeShape(spec.getResultShape(), driver);
  // The axis's group starts at the axis, which merges with no dim before it.
  int64_t axis = 0;
  for (int64_t covered = 0; covered < spec.getAxis();
       covered += driver.groups[axis++]) {
  }
  setSpecAttr(ConcatSpecAttr::get(getContext(), ins, out, axis));
  auto indexType = IndexType::get(getContext(), driver.groups.size());
  for (Region &region : getInputs())
    region.getArgument(0).setType(indexType);
}

// A constant folded into the kernel is a dense tensor of its own, so it is
// row-major and keeps no pair apart.
std::optional<Layout> ConstantOp::getOutputLayout(std::optional<Layout>) {
  return Layout::rowMajor(
      MO::castFromTensorLike(getType()).getShape().getRank());
}

// The stride views and the index ops of the same kind share a spec, whose
// in-shape is the tensor read and whose out-shape is the loop's: their layout
// rules are the same. The stride views answer for a capture once LIVM has
// hoisted a load's index op into a view; the index ops answer for the same
// hop while it is still inside a kernel, which passes that walk a load's
// index ops need (the reshape-epilogue refusal).
// A reshape only keeps the order of the elements when it reads a dense
// tensor. Over a broadcast or a cut input it answers "unknown", which gives up
// on the kernel; over a dense one its output is row-major and keeps no pair
// apart.
std::optional<Layout> ReshapeStrideOp::getOutputLayout(std::optional<Layout>) {
  return Layout::rowMajor(
      MO::castFromTensorLike(getResult().getType()).getShape().getRank());
}

// A slice keeps its input's strides. A pair stays contiguous when the inner
// dim is taken whole (see `keepsWholeDim`) and neither dim is stepped; where
// the outer dim starts only moves the base pointer. Slicing [2, 4, 8] to
// [2, 2, 8] cuts dim 1, so 2 and 4 stay apart while 4 and 8 still merge.
std::optional<Layout> SliceStrideOp::getOutputLayout(std::optional<Layout> in) {
  return sliceLayout(getSpec(), in);
}

// A broadcast replicates a dim by giving it stride 0. Two replicated dims
// merge, and so do two kept dims that were contiguous in the input, but a
// replicated dim and a kept one never do. A symbolic input dim could be 1 at
// run time, so it isn't known to be either and merges with neither neighbor.
std::optional<Layout>
BroadcastStrideOp::getOutputLayout(std::optional<Layout> in) {
  return broadcastLayout(getSpec(), in);
}

// A transpose reads its input with permuted strides. A pair of its dims is
// contiguous when the permutation left the two adjacent and in order and they
// were contiguous in the input already, so [b, 512, h, w] -> [b, h, w, 512]
// merges h and w and nothing else. Replicated dims follow the permutation, so
// two of those merge wherever they land.
std::optional<Layout>
TransposeStrideOp::getOutputLayout(std::optional<Layout> in) {
  return transposeLayout(getSpec(), in);
}

// A fold reads at an index of the rowwise kernel's shape and keeps no dims of
// its own, so it takes any merge the kernel's tensors allow. Keeping the
// reduced axis apart, which would otherwise fold rows together, is the pass's
// own rule, in `planRowwise`.
void IterReduceOp::canFlattenDims(MutableArrayRef<bool>) {
  // Clears nothing: the interface default keeps every pair apart, and
  // this op takes any merge.
}

// A per-element phase of a rowwise kernel (a norm's scale, a softmax's divide)
// reads and stores at the kernel's index and keeps no dims of its own, so it
// takes any merge the kernel's tensors allow. Its index takes the merged rank:
// in a softmax over the last axis of [B, S, W] merging B and S, the phase runs
// at a rank-2 index of [B*S, W].
void IterMapOp::canFlattenDims(MutableArrayRef<bool>) {
  // Clears nothing: the interface default keeps every pair apart, and
  // this op takes any merge.
}

// The phase reads and stores at the kernel's index, so it only takes the
// merged rank:
//
//   map.iter.map index (%j: !map.index<3>)
//   =>
//   map.iter.map index (%j: !map.index<2>)
void IterMapOp::flatten(FlattenDriver &driver) {
  getIndexArg().setType(IndexType::get(getContext(), driver.groups.size()));
}

// The row cache computes, at the kernel's index, a value the phases read back
// at their own index; like them, it takes any merge and the merged rank.
void IterTileOp::canFlattenDims(MutableArrayRef<bool>) {
  // Clears nothing: the interface default keeps every pair apart, and
  // this op takes any merge.
}

// The cache is filled at the kernel's index, so it only takes the merged
// rank:
//
//   map.iter.tile index (%i: !map.index<3>)
//   =>
//   map.iter.tile index (%i: !map.index<2>)
void IterTileOp::flatten(FlattenDriver &driver) {
  getIndexArg().setType(IndexType::get(getContext(), driver.groups.size()));
}

// The fold's index addresses the kernel's shape, so it only takes the merged
// rank. A sum over the last axis of [B, S, W], merging B and S:
//
//   map.iter.reduce @ReduceSum index (%i: !map.index<3>)
//   =>
//   map.iter.reduce @ReduceSum index (%i: !map.index<2>)
void IterReduceOp::flatten(FlattenDriver &driver) {
  getIndexArg().setType(IndexType::get(getContext(), driver.groups.size()));
}

// In a prologue or an epilogue, the reshape takes the kernel's index to the
// shape the captures are read in, its in-shape, which `map-flatten-shapes`
// merges. For example, an epilogue of a [M, N] kernel that reads its tensors in
// [M, 4, N/4] through `reshape [M, N] -> [M, 4, N/4]` merges 4 and N/4 when
// they are contiguous: the reshape becomes [M, N] -> [M, N], and the pass then
// drops it.
void ReshapeIndexOp::flatten(FlattenDriver &driver) {
  ViewSignatureAttr signature = getSpec().getSignature();
  MOSH::ShapeAttr in = mergeShape(signature.getIn(), driver);
  setSpecAttr(ReshapeSpecAttr::get(
      getContext(),
      ViewSignatureAttr::get(getContext(), in, signature.getOut())));
  getResult().setType(IndexType::get(getContext(), in.getRank()));
}

std::optional<Layout>
ReshapeIndexOp::getOutputLayout(std::optional<Layout> in) {
  return reshapeLayout(getSpec(), in);
}

std::optional<Layout> SliceIndexOp::getOutputLayout(std::optional<Layout> in) {
  return sliceLayout(getSpec(), in);
}

std::optional<Layout>
BroadcastIndexOp::getOutputLayout(std::optional<Layout> in) {
  return broadcastLayout(getSpec(), in);
}

std::optional<Layout>
TransposeIndexOp::getOutputLayout(std::optional<Layout> in) {
  return transposeLayout(getSpec(), in);
}
