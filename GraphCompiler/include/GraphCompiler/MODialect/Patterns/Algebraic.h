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

#pragma once

#include "mlir/IR/PatternMatch.h"

namespace M::MO {

/// Adds patterns that perform semantic-preserving algebraic simplifications:
/// arithmetic identities, math canonicalizations, and op-level rewrites such
/// as ConvToMatmul, BatchMatmul[1] -> Matmul, FlattenNestedConcat, and
/// MergeParallelMatmul.
///
/// `enableFastMath` adds patterns that are only valid under fast-math
/// semantics. `enableUndefinedMath` adds patterns that rewrite operations
/// whose behavior is undefined (e.g. div-by-zero, mod-by-one) to a fixed
/// constant.
void populateAlgebraicPatterns(mlir::RewritePatternSet &patterns,
                               bool enableFastMath = false,
                               bool enableUndefinedMath = false);

/// Adds forward-direction mathematical canonicalizations
/// (e.g. `rsqrt(x) -> 1/sqrt(x)`, `relu(x) -> max(x, 0)`, `mul(x, 2) -> x + x`,
/// `pow(x, 2) -> x * x`).
void populateForwardCanonicalizationPatterns(mlir::RewritePatternSet &patterns);

/// Adds backward-direction mathematical canonicalizations — the inverses of
/// the forward set.
void populateBackwardCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns);

} // namespace M::MO
