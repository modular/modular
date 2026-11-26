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

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPINTERFACES_H
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPINTERFACES_H

#include "GraphCompiler/MAPDialect/Dialect/MAPAttrs.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace M::MAP {
/// Whether `op`, which doesn't implement `MaxSimdWidthInterface`, produces an
/// index or hands one to its regions: index computation the width inference
/// can't follow.
bool computesIndex(::mlir::Operation *op);

// What `RowwisePhaseOpInterface::getReadCache` returns.
class IterCacheOp;

/// A tensor seen through views, for `map-flatten-shapes`: whether each dim's
/// stride is known to be 0, and whether each pair of adjacent dims (d, d+1) is
/// laid out one row after the other, i.e. stride[d] == extent[d+1] *
/// stride[d+1].
struct Layout {
  ::llvm::SmallVector<bool> zero;
  ::llvm::SmallVector<bool> contiguous;

  static Layout rowMajor(size_t rank) {
    return {::llvm::SmallVector<bool>(rank, false),
            ::llvm::SmallVector<bool>(rank ? rank - 1 : 0, true)};
  }

  /// Whether dims d and d+1 may merge into one.
  bool merges(size_t d) const {
    return contiguous[d] || (zero[d] && zero[d + 1]);
  }
};

/// The loop `map-flatten-shapes` is flattening and the sizes of the groups its
/// dims merge into, set for each loop, and the parameters declared so far for
/// symbolic products, shared by every loop of the graph.
struct FlattenDriver {
  ::llvm::SmallVector<int64_t> groups;
  /// The kernel being flattened: a loop body, an opaque kernel's call, or a
  /// prologue or an epilogue.
  ::mlir::Operation *body = nullptr;
  ::llvm::DenseMap<::mlir::Attribute, ::mlir::TypedAttr> declared;
};

/// The layout of `value` from the views between it and the row-major tensor
/// they read (a kernel input or output, or a constant); nullopt when unknown.
std::optional<Layout> layoutOf(::mlir::Value value);

/// Merges `shape`'s dims by the driver's groups; a symbolic product becomes a
/// graph parameter, since a kernel's dims must be named. The plan clears the
/// merges a loop outside the graph's own body can't declare, so a planned
/// loop's groups always merge.
MOSH::ShapeAttr mergeShape(MOSH::ShapeAttr shape, FlattenDriver &driver);
} // namespace M::MAP

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOpInterfaces.h.inc"

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPINTERFACES_H
