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

#ifndef KGEN_TRANSFORMS_LEGACYFRAMEEVALUATION_H
#define KGEN_TRANSFORMS_LEGACYFRAMEEVALUATION_H

#include "FrameData.h"

namespace M::KGEN {

/// Computes op states by exploring control flow paths, then puts every value
/// whose definition and use are in different states into the frame. Matches
/// the `FrameEvaluator` signature.
void evaluateOldFrame(FrameData &frameData, FuncOp originalFunction,
                      mlir::DominanceInfo &domInfo, Value errorValue,
                      Value resultValue, FrameStateTransform transform,
                      bool isHot);

} // namespace M::KGEN

#endif // KGEN_TRANSFORMS_LEGACYFRAMEEVALUATION_H
