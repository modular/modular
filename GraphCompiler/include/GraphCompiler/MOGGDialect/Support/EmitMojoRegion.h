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
#ifndef GENERICML_GRAPHCOMPILER_SUPPORT_EMITMOJOREGION_H
#define GENERICML_GRAPHCOMPILER_SUPPORT_EMITMOJOREGION_H

#include "GraphCompiler/MOGGDialect/Support/MojoBuilder.h"
#include "Support/AssertStream.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

namespace M::MOGG {

/// Every region the emitted code can be in. A region's kind is also its
/// identity: `EmitMojoState::currentRegion<Region>()` finds the innermost one
/// of the same kind.
enum class EmitMojoRegionKind {
  /// The width and alignment of an elementwise lambda.
  kElementwiseLambda,
  /// What the ops of a `mogg.call.rowwise` body print against.
  kRowwiseBody,
};

/// A construct whose data the ops emitted inside it read: an elementwise
/// lambda's parameters, a rowwise body. Regions inherit from
/// `EmitMojoRegion<Region, Kind>`, so the state holds all of them on one stack
/// and finds one by its kind.
class EmitMojoRegionBase {
public:
  virtual ~EmitMojoRegionBase() = default;

  /// Which region this is, compared when a region is looked up.
  virtual EmitMojoRegionKind kind() const = 0;
};

/// The only per-region code the mechanism needs: the kind it answers with.
template <typename Region, EmitMojoRegionKind Kind>
class EmitMojoRegion : public EmitMojoRegionBase {
public:
  EmitMojoRegionKind kind() const override { return Kind; }

  /// The kind as a constant, for `EmitMojoState::findRegion`.
  static constexpr EmitMojoRegionKind kKind = Kind;
};

/// The regions currently being emitted, innermost last.
using EmitMojoRegionStack =
    llvm::SmallVector<std::unique_ptr<EmitMojoRegionBase>>;

/// Pops the region that was pushed when it was created, and asserts that
/// regions are popped in the reverse order they were pushed.
class EmitMojoRegionScope {
public:
  EmitMojoRegionScope(EmitMojoRegionStack &regions, size_t regionCount)
      : regions(regions)
#ifndef NDEBUG
        ,
        regionCount(regionCount)
#endif

  {
  }
  EmitMojoRegionScope(const EmitMojoRegionScope &) = delete;
  EmitMojoRegionScope &operator=(const EmitMojoRegionScope &) = delete;
  ~EmitMojoRegionScope() {
    ASSERT_STREAM(regions.size() == regionCount + 1,
                  << "regions must be popped in the reverse order they were "
                     "pushed");
    regions.pop_back();
  }

private:
  EmitMojoRegionStack &regions;
#ifndef NDEBUG
  size_t regionCount;
#endif
};

/// The width and alignment of a lambda that runs one SIMD operation per element
/// of a tile: a fusion functor's method, a `foreach` closure, or a rowwise
/// phase. Both are names the body can read; asking for one the enclosing lambda
/// does not have is an error here rather than an empty name in the emitted
/// text.
class ElementwiseLambdaRegion
    : public EmitMojoRegion<ElementwiseLambdaRegion,
                            EmitMojoRegionKind::kElementwiseLambda> {
public:
  /// Empty: asking for a width or alignment then asserts, which is what an op
  /// reached outside an elementwise lambda wants.
  ElementwiseLambdaRegion() = default;

  ElementwiseLambdaRegion(MojoValue width, MojoValue alignment)
      : widthValue(std::move(width)), alignmentValue(std::move(alignment)) {}

  MojoValue width() const {
    ASSERT_STREAM(!widthValue.text.empty(),
                  << "the enclosing lambda has no width");
    return widthValue;
  }

  MojoValue alignment() const {
    ASSERT_STREAM(!alignmentValue.text.empty(),
                  << "the enclosing lambda has no alignment");
    return alignmentValue;
  }

private:
  MojoValue widthValue;
  MojoValue alignmentValue;
};

/// What the ops of a `mogg.call.rowwise` body print against: names in the Row
/// API body the kernel prints, and its element types.
class RowwiseBodyRegion
    : public EmitMojoRegion<RowwiseBodyRegion,
                            EmitMojoRegionKind::kRowwiseBody> {
public:
  RowwiseBodyRegion(Block *body, MojoValue params, MojoValue rank,
                    MojoValue rowCoord, MojoValue axisSize, MojoValue dtype,
                    MojoValue accum)
      : body(body), params(std::move(params)), rank(std::move(rank)),
        rowCoord(std::move(rowCoord)), axisSize(std::move(axisSize)),
        dtype(std::move(dtype)), accum(std::move(accum)) {}

  Block *body = nullptr;
  MojoValue params;
  MojoValue rank;
  MojoValue rowCoord;
  MojoValue axisSize;
  MojoValue dtype;
  MojoValue accum;
  /// Set by the body's `mogg.rowwise.tile`, which prints them.
  MojoValue load;
  MojoValue row;
  /// The tile the phase printing now reads.
  MojoValue tile;
  /// Each `mogg.rowwise.cache`'s `compute` closure, which the cached Row
  /// overloads take again to recompute off the register tier.
  llvm::DenseMap<Value, MojoValue> cacheComputes;
};

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_SUPPORT_EMITMOJOREGION_H
