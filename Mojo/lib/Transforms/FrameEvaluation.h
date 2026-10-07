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

#ifndef KGEN_TRANSFORMS_FRAMEEVALUATION_H
#define KGEN_TRANSFORMS_FRAMEEVALUATION_H

#include "FrameData.h"

namespace M::KGEN {

/// Puts every value that is live across a suspension point into the frame.
/// Forwards to `evaluateFrameByStateNumbering` until the liveness
/// analysis lands.
/// Selected with the `use-liveness-frame-evaluation` option of
/// `lower-async-functions`. Matches the `FrameEvaluator` signature.
void evaluateFrameByLiveness(FrameData &frameData, FuncOp originalFunction,
                             mlir::DominanceInfo &domInfo, Value errorValue,
                             Value resultValue, FrameStateTransform transform,
                             bool isHot);

} // namespace M::KGEN

#endif // KGEN_TRANSFORMS_FRAMEEVALUATION_H
