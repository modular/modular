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

#ifndef KGEN_TRANSFORMUTILS_LOOPUTILS_H
#define KGEN_TRANSFORMUTILS_LOOPUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN {
/// Return whether skipping `op` on the iteration that exits is unobservable.
/// In a top-tested loop the exiting test comes first, so every iteration runs
/// whole. When the test follows part of the body instead, that part runs one
/// more time than the ops behind it, so `raise-for-loops` raises the loop
/// only when what precedes the test is droppable, and `rotate-loops` leaves
/// such an op where it is rather than duplicating it.
bool isDroppableOnExitIteration(Operation *op);
} // namespace M::KGEN

#endif // KGEN_TRANSFORMUTILS_LOOPUTILS_H
