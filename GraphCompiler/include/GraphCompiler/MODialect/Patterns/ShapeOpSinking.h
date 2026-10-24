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

/// Adds shape-op-sinking patterns: rewrites that re-shuffle the graph by
/// pushing reshape/broadcast/transpose/split/slice ops through neighboring
/// elementwise / reduction / matmul ops to expose more fusion and folding.
void populateShapeOpSinkingPatterns(mlir::RewritePatternSet &patterns);

} // namespace M::MO
