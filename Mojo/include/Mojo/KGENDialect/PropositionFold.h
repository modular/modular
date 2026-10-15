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
//
// Clause-level interface to the proposition fold: insert a proposition into a
// folded conjunction, or test what inserting it would do. Ordinary code builds
// conjunctions with `ParamOperatorAttr::get(POC::And, ...)` and does not need
// this header.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_PROPOSITIONFOLD_H
#define KGEN_KGENDIALECT_PROPOSITIONFOLD_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "llvm/ADT/SmallVector.h"

namespace M::KGEN {

enum class ClauseInsertResult {
  Redundant,     // The folded set already entails the proposition.
  Added,         // The proposition is independent of the set.
  Contradiction, // The proposition and the set are jointly unsatisfiable.
};

/// Insert one already-canonical proposition into an already-folded `clauses`.
/// Nested AND flattens into sequential inserts (Contradiction wins; else Added
/// if any leaf was Added; else Redundant). True is Redundant; False is
/// Contradiction. On Contradiction, `clauses` is unspecified.
ClauseInsertResult insertClause(SmallVectorImpl<TypedAttr> &clauses,
                                TypedAttr canonicalProp);

/// What inserting `canonicalGoal` into the already-canonical
/// `canonicalAssumption` would return, without building the conjunction. A
/// False assumption entails every goal, so the result is Redundant.
ClauseInsertResult testClause(TypedAttr canonicalAssumption,
                              TypedAttr canonicalGoal);

} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_PROPOSITIONFOLD_H
