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

#ifndef GRAPHCOMPILER_DRIVER_DEVICE_DEBUGALLOCPOISON_H
#define GRAPHCOMPILER_DRIVER_DEVICE_DEBUGALLOCPOISON_H

#include "Support/ML/DType.h"
#include "Support/ML/TensorSpec.h"
#include "Support/SymbolExport.h"
#include "llvm/ADT/StringRef.h"
#include <cstddef>
#include <cstdint>
#include <optional>

namespace M::GraphCompiler::Driver {

/// The dtype-aware fill pattern of
/// MODULAR_DEBUG_DEVICE_ALLOCATOR=uninitialized-poison.
struct PoisonPattern {
  uint64_t bits;
  size_t patternBytes;
  const char *patternKind; // "largest-finite" or "byte-fill"; for diagnostics.
};

/// Returns `std::nullopt` for sub-byte floats (lanes can't be written at byte
/// granularity) and fp8 (no `max_finite`, or it collides with saturated
/// quantization values). Keep in sync with the Mojo poison-check set.
MODULAR_VISIBILITY_EXPORT std::optional<PoisonPattern>
getPoisonPattern(const TensorSpec &spec);

/// Fills `totalBytes` bytes of host-visible memory at `ptr` with `pattern`.
MODULAR_VISIBILITY_EXPORT void fillHostMemory(void *ptr, size_t totalBytes,
                                              const PoisonPattern &pattern);

/// What the debug allocator does with one allocation.
struct AllocationPoisonDecision {
  uint64_t ordinal; ///< Position in the current epoch, counted from 0.
  /// Poison if selected, zero if not. Empty leaves the contents alone
  /// (`sync-only`, or a selected dtype with no pattern).
  std::optional<PoisonPattern> fill;
};

/// Decides how to fill one allocation, consumes an ordinal, and logs it.
/// `hostAlloc` only labels the log line.
///
/// Unselected allocations are zero-filled so both arms of a bisection stay
/// deterministic. A selected allocation with no poison pattern is left alone:
/// zeroing would hide it from bisection, at the cost of full determinism.
///
/// The selection comes from `MODULAR_DEBUG_ALLOC_POISON_SELECT` (config key
/// `debug.alloc_poison_select`). It is either one of these modes on its own:
///
///   (unset) / `all`    poison every allocation (the default)
///   `none`             poison nothing; zero-fill everything
///   `sync-only`        write nothing at all, but keep the synchronization
///                      after every device allocation
///
/// or one or both of these filters, joined by `&`, all of which must match:
///
///   `dtype=si64,ui32`  poison only these element types, spelled as `DType`
///                      prints them
///   `ord=100-2000`     poison only these allocation ordinals, inclusive;
///                      `ord=100` selects one
///
/// e.g. `dtype=si64&ord=100-2000`. Not `;`: `modular.cfg` reads it as a
/// comment.
///
/// A malformed clause (including a mode mixed with clauses, or a repeated
/// `dtype=`/`ord=`) is ignored with a warning, so a typo widens the selection;
/// silently selecting nothing would read as "no uninitialized reads here".
///
/// `sync-only` separates the allocator's two effects: the fill, and a device
/// sync after every allocation, which alone can hide a race. Still
/// deterministic under `sync-only` means a race; non-deterministic means a
/// genuine uninitialized read. `none` can't tell, since zero-fill is
/// deterministic too.
///
/// `MODULAR_DEBUG_ALLOC_POISON_SELECT_FILE` (`debug.alloc_poison_select_file`)
/// names a file with the same grammar, checked on every allocation. Each
/// change starts a new epoch and **resets ordinals to zero**, so rewriting it
/// between requests addresses "the n-th allocation of this request" and lets
/// a bisection run within one boot. A change is a new mtime, size or file
/// identity, not new content, so a measurement can be repeated. mtimes can be
/// too coarse, so write a temp file and rename it over this one.
///
/// Ordinals are stable across runs only if allocation order is (it depends on
/// host threading too). The `DEBUG_ALLOC_POISON` log line lists every ordinal
/// with dtype and size, so diffing logs checks alignment and one run's log
/// indexes the next.
MODULAR_VISIBILITY_EXPORT AllocationPoisonDecision
decidePoison(const TensorSpec &spec, bool hostAlloc);

/// Evaluates `selectionSpec` without consuming an ordinal or logging, for
/// testing the grammar.
MODULAR_VISIBILITY_EXPORT AllocationPoisonDecision decidePoisonForTesting(
    llvm::StringRef selectionSpec, const TensorSpec &spec, uint64_t ordinal);

/// Testing only: points the selection at file `path` and clears cached state.
/// An empty `path` re-reads the configured selection.
MODULAR_VISIBILITY_EXPORT void
resetPoisonSelectionForTesting(llvm::StringRef path);

} // namespace M::GraphCompiler::Driver

#endif // GRAPHCOMPILER_DRIVER_DEVICE_DEBUGALLOCPOISON_H
