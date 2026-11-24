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

#ifndef GENERICML_MGPTRANSFORMS_FIXUPISOLATEDFROMABOVE_H
#define GENERICML_MGPTRANSFORMS_FIXUPISOLATEDFROMABOVE_H

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "mlir/IR/Builders.h"

#include "GraphCompiler/MGPDialect/Transforms/IndexSet.h"

namespace M::MGP {

/// Returns a copy of inModelOp with the 'init' and 'execute' blocks rewritten
/// to ensure all control flow ops are correctly IsolatedFromAbove. Any free
/// values in sub-blocks are captured by new sub-block arguments, and the
/// overall control flow ops are rewritten to include (and, for loops, return)
/// those arguments. The returned model is built after inModelOp, the caller
/// is responsible for erasing inModelOp.
///
/// TODO(#15830): Can mildly generalize so can be also used for MO control
/// flow ops.
ModelOp fixupIsolatedFromAbove(mlir::OpBuilder &builder, ModelOp inModelOp);

} // namespace M::MGP

#endif // GENERICML_MGPTRANSFORMS_FIXUPISOLATEDFROMABOVE_H
