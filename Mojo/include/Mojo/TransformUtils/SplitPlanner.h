//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef KGEN_TRANSFORMUTILS_SPLITPLANNER_H
#define KGEN_TRANSFORMUTILS_SPLITPLANNER_H

#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <vector>

namespace M {
/// A symbol of a module that a splitter partitions. The splitters for LLVM IR
/// and for the LLVM dialect describe their symbols with this struct and share
/// the planners below.
struct SplitSymbol {
  /// The symbols that this one references, as sorted and unique indices.
  SmallVector<unsigned, 4> dependencies;
  /// Orders the outline candidates of equal saving.
  StringRef name;
  /// The instructions of a function, which estimate the cost to optimize it.
  /// Only outlining reads this.
  uint64_t instructionCount = 0;
  bool isFunction = false;
  /// A function without a body, or a global without an initializer.
  bool isDeclaration = false;
  /// The symbol roots a slice.
  bool isAnchor = false;
  /// The users of the symbol share a slice, as the users of a mutable global
  /// must.
  bool tiesUsers = false;
  /// An internal function that `planSplitPerExported` can give a slice of its
  /// own.
  bool canOutline = false;
};

/// A partition of a module, as indices into its `SplitSymbol`s.
struct SplitPlan {
  /// The symbols that each slice defines, in index order. A slice declares
  /// the other symbols that its members reference.
  std::vector<std::vector<unsigned>> slices;
  /// The functions that `planSplitPerExported` gave a slice of their own, in
  /// index order. The other slices declare them.
  std::vector<unsigned> outlineRoots;
};

/// Gives each anchor a slice with its transitive closure. Two closures share a
/// slice when one reaches the anchor of the other, because an anchor must stay
/// unique. They also share one when both reach a symbol with `tiesUsers`.
///
/// A nonzero `outlineThreads` gives an internal function that several anchors
/// reach a slice of its own, when that keeps the estimated critical path over
/// that many threads from growing.
///
/// The larger slices come first, so that they start first. With fewer than
/// two anchors the plan has no slices, and the caller keeps the module whole.
SplitPlan planSplitPerExported(ArrayRef<SplitSymbol> symbols,
                               unsigned outlineThreads = 0);

/// Gives each anchor a slice of its own. A function anchor that reaches a
/// symbol with `tiesUsers` and has other users joins the slices of its users
/// instead. A slice defines its root and what the root reaches without
/// passing a function with a slice of its own, which the slice declares. The
/// slices come in the index order of their roots, and a symbol in an earlier
/// slice roots none.
SplitPlan planSplitPerFunction(ArrayRef<SplitSymbol> symbols);
} // namespace M

#endif // KGEN_TRANSFORMUTILS_SPLITPLANNER_H
